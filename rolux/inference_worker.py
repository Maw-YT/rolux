"""
Thread B — TensorRT Depth Anything V2 async inference.

Memory topology (minimize H↔D copies without requiring CuPy JIT / CUDA_PATH):
  1. Capture thread leaves a host BGR frame in a depth-1 slot.
  2. OpenCV preprocess on pinned host → NCHW float (fast SIMD path).
  3. Single cudaMemcpyAsync H2D into the TRT-bound input buffer.
  4. IExecutionContext.execute_async_v3 on a dedicated non-default stream.
  5. Single D2H of the compact depth map → CPU colormap for the overlay.

Inference stays on GPU; only the network I/O tensors cross the bus.
"""

from __future__ import annotations

import ctypes
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from rolux.capture_worker import CapturedFrame
from rolux.config import RoluxConfig
from rolux.win32_utils import WindowRect

try:
    from cuda.bindings import driver as cuda  # type: ignore
    from cuda.bindings import runtime as cudart  # type: ignore
except ImportError:  # pragma: no cover
    from cuda import cuda, cudart  # type: ignore

import tensorrt as trt  # type: ignore


def _cuda_check(result) -> None:
    if isinstance(result, tuple):
        err, *rest = result
        if err != cudart.cudaError_t.cudaSuccess and (
            not hasattr(cuda, "CUresult") or err != cuda.CUresult.CUDA_SUCCESS
        ):
            # Some bindings only expose runtime errors.
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"CUDA error: {err}")
        return rest[0] if len(rest) == 1 else tuple(rest)
    if result != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {result}")


@dataclass
class DepthPacket:
    """Depth / normals / scene for the shader → overlay path."""

    rgb: np.ndarray  # HxW gray depth, or HxWx3 BGR display after shaders
    rect: WindowRect
    capture_ts: float
    infer_ms: float
    main_bgr: Optional[np.ndarray] = None  # Roblox color @ network res
    normal_bgr: Optional[np.ndarray] = None  # unused (GPU normals)
    depth_f: Optional[np.ndarray] = None  # HxW float32 depth in [0,1] for normals


class _TrtLogger(trt.ILogger):
    def __init__(self) -> None:
        super().__init__()

    def log(self, severity: trt.ILogger.Severity, msg: str) -> None:  # noqa: N802
        if severity <= trt.ILogger.Severity.WARNING:
            print(f"[TRT] {msg}")


class InferenceWorker(threading.Thread):
    def __init__(
        self,
        config: RoluxConfig,
        in_slot: list,
        in_lock: threading.Lock,
        out_slot: list,
        out_lock: threading.Lock,
        stop_event: threading.Event,
        frame_ready: Optional[threading.Event] = None,
        attach_main: bool = True,
    ) -> None:
        super().__init__(name="RoluxInfer", daemon=True)
        self.cfg = config
        self.in_slot = in_slot
        self.in_lock = in_lock
        self.out_slot = out_slot
        self.out_lock = out_lock
        self.stop_event = stop_event
        self.frame_ready = frame_ready
        self.attach_main = attach_main

        self._engine: Optional[trt.ICudaEngine] = None
        self._context: Optional[trt.IExecutionContext] = None
        self._stream = None
        self._stream_ptr: int = 0

        self._d_input: int = 0
        self._d_output: int = 0
        self._h_input: Optional[np.ndarray] = None  # pinned NCHW
        self._h_output: Optional[np.ndarray] = None  # pinned depth
        self._h_input_ptr: int = 0
        self._h_output_ptr: int = 0

        self._input_name: str = ""
        self._output_name: str = ""
        self._out_shape: tuple[int, ...] = ()
        self._net_h: int = int(config.input_h)
        self._net_w: int = int(config.input_w)
        self._in_dtype = np.dtype(np.float32)
        self._out_dtype = np.dtype(np.float32)

        mean = np.asarray(config.imagenet_mean, dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(config.imagenet_std, dtype=np.float32).reshape(3, 1, 1)
        self._mean = mean
        self._std = std

        # EMA of the depth range so the normalized scale stays stable frame-to-frame.
        self._ema_min: Optional[float] = None
        self._ema_max: Optional[float] = None
        # Runtime-toggleable (GUI). Mirrors cfg.stabilize_depth_range at construction.
        self.stabilize = bool(getattr(config, "stabilize_depth_range", True))

    def _load_engine(self, path: Path) -> None:
        logger = _TrtLogger()
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {path}")
        self._engine = engine
        self._context = engine.create_execution_context()

        inputs, outputs = [], []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append(name)
            else:
                outputs.append(name)
        if not inputs or not outputs:
            raise RuntimeError("Engine missing I/O tensors")
        self._input_name = inputs[0]
        self._output_name = outputs[0]

        engine_shape = tuple(self._engine.get_tensor_shape(self._input_name))
        # Fixed engines ignore set_input_shape; dynamic ones need an explicit size.
        if any(int(d) < 0 for d in engine_shape):
            shape = (1, 3, int(self.cfg.input_h), int(self.cfg.input_w))
            if not self._context.set_input_shape(self._input_name, shape):
                raise RuntimeError(f"set_input_shape failed for {shape}")
        else:
            shape = tuple(int(d) for d in engine_shape)
            eh, ew = int(shape[2]), int(shape[3])
            if eh != self.cfg.input_h or ew != self.cfg.input_w:
                print(
                    f"[Rolux] engine input is {ew}x{eh} — GUI size "
                    f"{self.cfg.input_w}x{self.cfg.input_h} ignored "
                    f"(rebuild engine for a different size)"
                )
            if not self._context.set_input_shape(self._input_name, shape):
                # Some TRT builds still want an explicit set even for fixed shapes.
                pass

        self._net_h = int(self._context.get_tensor_shape(self._input_name)[2])
        self._net_w = int(self._context.get_tensor_shape(self._input_name)[3])
        self._out_shape = tuple(self._context.get_tensor_shape(self._output_name))

    @property
    def network_size(self) -> tuple[int, int]:
        """(height, width) actually used by the loaded engine."""
        return (self._net_h, self._net_w)

    def _np_dtype(self, trt_dt) -> np.dtype:
        if trt_dt == trt.float16:
            return np.dtype(np.float16)
        if trt_dt == trt.float32:
            return np.dtype(np.float32)
        raise RuntimeError(f"Unsupported TRT dtype: {trt_dt}")

    def _alloc_cuda(self) -> None:
        assert self._context is not None and self._engine is not None
        err, stream = cudart.cudaStreamCreateWithFlags(cudart.cudaStreamNonBlocking)
        _cuda_check(err)
        self._stream = stream
        self._stream_ptr = int(stream)

        in_shape = tuple(self._context.get_tensor_shape(self._input_name))
        out_shape = tuple(self._context.get_tensor_shape(self._output_name))
        self._out_shape = out_shape
        self._in_dtype = self._np_dtype(self._engine.get_tensor_dtype(self._input_name))
        self._out_dtype = self._np_dtype(self._engine.get_tensor_dtype(self._output_name))

        in_bytes = int(np.prod(in_shape)) * self._in_dtype.itemsize
        out_bytes = int(np.prod(out_shape)) * self._out_dtype.itemsize

        # OPT: device I/O allocated once for the process lifetime.
        err, d_in = cudart.cudaMalloc(in_bytes)
        _cuda_check(err)
        err, d_out = cudart.cudaMalloc(out_bytes)
        _cuda_check(err)
        self._d_input = int(d_in)
        self._d_output = int(d_out)

        if not (
            self._context.set_tensor_address(self._input_name, self._d_input)
            and self._context.set_tensor_address(self._output_name, self._d_output)
        ):
            raise RuntimeError("set_tensor_address failed")

        # OPT: page-locked host staging for async H2D / D2H.
        err, h_in = cudart.cudaMallocHost(in_bytes)
        _cuda_check(err)
        err, h_out = cudart.cudaMallocHost(out_bytes)
        _cuda_check(err)
        self._h_input_ptr = int(h_in)
        self._h_output_ptr = int(h_out)

        self._h_input = np.frombuffer(
            (ctypes.c_uint8 * in_bytes).from_address(self._h_input_ptr),
            dtype=self._in_dtype,
        ).reshape(in_shape)
        self._h_output = np.frombuffer(
            (ctypes.c_uint8 * out_bytes).from_address(self._h_output_ptr),
            dtype=self._out_dtype,
        ).reshape(out_shape)

    def setup(self) -> None:
        if self._context is not None:
            return
        path = Path(self.cfg.engine_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing TensorRT engine: {path}\n"
                "Build it with trtexec (see README Model setup), e.g.\n"
                "  trtexec --onnx=models/your_model.onnx "
                "--saveEngine=models/your_model.engine ..."
            )
        self._load_engine(path)
        self._alloc_cuda()
        print(
            f"[Rolux] TRT ready | in={self._input_name} "
            f"{self._context.get_tensor_shape(self._input_name)} {self._in_dtype} | "  # type: ignore
            f"out={self._output_name} {self._out_shape} {self._out_dtype}"
        )

    def _preprocess(self, bgr: np.ndarray) -> None:
        """BGR uint8 (prefer already network-sized) → pinned NCHW → async H2D."""
        assert self._h_input is not None
        th, tw = self._net_h, self._net_w
        if bgr.shape[0] != th or bgr.shape[1] != tw:
            bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        # mean applied in 0–1 space (after scalefactor); std applied after.
        blob = cv2.dnn.blobFromImage(
            bgr,
            scalefactor=1.0 / 255.0,
            size=(tw, th),
            mean=self.cfg.imagenet_mean,
            swapRB=True,
            crop=False,
        )
        blob[0, 0] /= self.cfg.imagenet_std[0]
        blob[0, 1] /= self.cfg.imagenet_std[1]
        blob[0, 2] /= self.cfg.imagenet_std[2]
        if self._in_dtype == np.float16:
            np.copyto(self._h_input, blob.astype(np.float16))
        else:
            np.copyto(self._h_input, blob)

        _cuda_check(
            cudart.cudaMemcpyAsync(
                self._d_input,
                self._h_input_ptr,
                self._h_input.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream,
            )
        )

    def _colormap_depth(self, out_h: int, out_w: int) -> tuple[np.ndarray, np.ndarray]:
        """D2H depth → uint8 preview + float32 [0,1] for GPU normals."""
        assert self._h_output is not None
        _ = (out_h, out_w)
        _cuda_check(
            cudart.cudaMemcpyAsync(
                self._h_output_ptr,
                self._d_output,
                self._h_output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                self._stream,
            )
        )
        _t_sync = time.perf_counter()
        _cuda_check(cudart.cudaStreamSynchronize(self._stream))
        self._sync_ms = (time.perf_counter() - _t_sync) * 1000.0

        depth = np.asarray(self._h_output)
        while depth.ndim > 2:
            depth = depth[0]
        depth_f = depth.astype(np.float32, copy=False)
        dmin = float(depth_f.min())
        dmax = float(depth_f.max())
        if self.stabilize:
            a = float(self.cfg.depth_range_alpha)
            if self._ema_min is None:
                self._ema_min, self._ema_max = dmin, dmax
            else:
                # Expand instantly toward new extremes (no clipping); contract slowly.
                self._ema_min = min(dmin, self._ema_min + (dmin - self._ema_min) * a)
                self._ema_max = max(dmax, self._ema_max + (dmax - self._ema_max) * a)
            dmin, dmax = self._ema_min, self._ema_max
        else:
            self._ema_min = self._ema_max = None
        denom = max(dmax - dmin, 1e-6)
        depth_01 = (depth_f - dmin) / denom
        depth_u8 = (depth_01 * 255.0 * self.cfg.depth_gain).clip(0, 255).astype(np.uint8)
        return np.ascontiguousarray(depth_u8), np.ascontiguousarray(depth_01)

    def run(self) -> None:
        # GUI may have already called setup() so the engine is warm before capture starts.
        if self._context is None:
            self.setup()
        assert self._context is not None
        try:
            import ctypes

            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 15  # TIME_CRITICAL
            )
        except Exception:
            pass

        self._last_ts: Optional[float] = None
        self._sync_ms = 0.0

        # Rolling per-stage timing (printed once/sec) to locate pipeline stalls.
        _acc = {"n": 0, "wait": 0.0, "pre": 0.0, "sync": 0.0, "post": 0.0, "total": 0.0, "skip": 0}
        _acc_t = time.perf_counter()
        _prev_end = time.perf_counter()

        while not self.stop_event.is_set():
            with self.in_lock:
                packet: Optional[CapturedFrame] = self.in_slot[0]

            if packet is None:
                time.sleep(0.0)  # yield only
                continue

            ts = packet.capture_ts
            if ts == self._last_ts:
                time.sleep(0.0)
                continue
            self._last_ts = ts

            # Snapshot geometry now; resize copies pixels off the shared capture buf.
            rect = packet.rect
            t0 = time.perf_counter()
            try:
                self._preprocess(packet.bgr)
                t_pre = time.perf_counter()

                # If a newer frame arrived during preprocess, skip this TRT call.
                with self.in_lock:
                    newest = self.in_slot[0]
                if newest is not None and newest.capture_ts > ts:
                    self._last_ts = None  # allow immediate retry on newest
                    _acc["skip"] += 1
                    continue

                ok = self._context.execute_async_v3(self._stream_ptr)
                if not ok:
                    print("[Rolux] execute_async_v3 returned False")
                    continue
                rgb, depth_f = self._colormap_depth(rect.height, rect.width)
                t_end = time.perf_counter()
                infer_ms = (t_end - t0) * 1000.0

                _acc["n"] += 1
                _acc["wait"] += (t0 - _prev_end) * 1000.0        # idle between iterations
                _acc["pre"] += (t_pre - t0) * 1000.0             # cpu preprocess + H2D enqueue
                _acc["sync"] += self._sync_ms                    # gpu execute + D2H (wall)
                _acc["post"] += (t_end - t_pre) * 1000.0 - self._sync_ms  # cpu colormap
                _acc["total"] += infer_ms
                _prev_end = t_end
                if t_end - _acc_t >= 1.0 and _acc["n"] > 0:
                    n = _acc["n"]
                    print(
                        f"[perf/infer] {n}/s | total={_acc['total']/n:5.1f}ms "
                        f"wait={_acc['wait']/n:5.1f} pre={_acc['pre']/n:4.1f} "
                        f"gpu_sync={_acc['sync']/n:4.1f} post={_acc['post']/n:4.1f} "
                        f"| stale_skips={_acc['skip']}"
                    )
                    _acc = {"n": 0, "wait": 0.0, "pre": 0.0, "sync": 0.0, "post": 0.0, "total": 0.0, "skip": 0}
                    _acc_t = t_end

                # Drop stale results — never publish depth older than the latest capture.
                with self.in_lock:
                    newest = self.in_slot[0]
                if newest is not None and newest.capture_ts > ts:
                    self._last_ts = None
                    continue

                with self.out_lock:
                    self.out_slot[0] = DepthPacket(
                        rgb=rgb,
                        rect=rect,
                        capture_ts=ts,
                        infer_ms=infer_ms,
                        main_bgr=packet.color_bgr if self.attach_main else None,
                        normal_bgr=None,
                        depth_f=depth_f,
                    )
                if self.frame_ready is not None:
                    self.frame_ready.set()
            except Exception as exc:
                print(f"[Rolux] infer error: {exc}")
                time.sleep(0.001)

        self._cleanup()

    def _cleanup(self) -> None:
        try:
            if self._d_input:
                cudart.cudaFree(self._d_input)
            if self._d_output:
                cudart.cudaFree(self._d_output)
            if self._h_input_ptr:
                cudart.cudaFreeHost(self._h_input_ptr)
            if self._h_output_ptr:
                cudart.cudaFreeHost(self._h_output_ptr)
            if self._stream is not None:
                cudart.cudaStreamDestroy(self._stream)
        except Exception:
            pass
