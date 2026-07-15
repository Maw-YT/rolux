#!/usr/bin/env python3
"""
export_trt.py — convert Depth Anything V2 ONNX → TensorRT .engine

TensorRT 11.x notes:
  - EXPLICIT_BATCH is gone (always-on).
  - BuilderFlag.FP16 is gone — networks are strongly typed; precision comes
    from the ONNX graph. Prefer the FP16 ONNX, or run ModelOpt AutoCast.

Example:
  python export_trt.py --onnx models/depth_anything_v2_vits_fp16.onnx --height 392 --width 392
  python export_trt.py --onnx models/depth_anything_v2_vits.onnx --autocast --height 392 --width 392
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _network_flags(trt) -> int:
    """TRT ≤10 used EXPLICIT_BATCH; TRT 11+ creates strongly-typed nets with flags=0."""
    flag_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if flag_enum is None:
        return 0
    if hasattr(flag_enum, "EXPLICIT_BATCH"):
        return 1 << int(flag_enum.EXPLICIT_BATCH)
    # TRT 11: strong typing is implicit; optional STRONGLY_TYPED is a no-op / legacy.
    return 0


def _enable_fp16_if_supported(trt, builder, config) -> bool:
    """Enable FP16 tactics on TRT ≤10. Returns True if a weak-typing flag was set."""
    if hasattr(trt.BuilderFlag, "FP16"):
        if hasattr(builder, "platform_has_fast_fp16") and not builder.platform_has_fast_fp16:
            print("[export] WARNING: platform_has_fast_fp16 is False; building anyway")
        config.set_flag(trt.BuilderFlag.FP16)
        print("[export] BuilderFlag.FP16 enabled (TRT ≤10 weak typing)")
        return True
    print(
        "[export] TRT 11+: no BuilderFlag.FP16 — precision follows ONNX dtypes "
        "(use FP16 ONNX or --autocast)"
    )
    return False


def maybe_autocast(onnx_path: Path, out_path: Path, height: int, width: int) -> Path:
    """
    Convert FP32 ONNX → mixed FP16 for TRT 11 strong typing via ModelOpt AutoCast.
    keep_io_types=True leaves image/depth as FP32; internal compute is FP16.
    """
    import numpy as np
    import onnx
    from modelopt.onnx.autocast import convert_to_mixed_precision

    calib = out_path.with_name(f"calib_image_{height}.npz")
    # Input tensor name for fabio-sim DA-V2 export is "image".
    np.savez(calib, image=np.random.randn(1, 3, height, width).astype(np.float32))
    print(f"[export] ModelOpt AutoCast FP16: {onnx_path} → {out_path}")
    model = convert_to_mixed_precision(
        onnx_path=str(onnx_path),
        low_precision_type="fp16",
        keep_io_types=True,
        calibration_data=str(calib),
        providers=["cpu"],
    )
    onnx.save(model, str(out_path))
    print(f"[export] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    height: int,
    width: int,
    workspace_gb: float,
    want_fp16: bool,
) -> None:
    import tensorrt as trt  # type: ignore

    print(f"[export] TensorRT {trt.__version__}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(_network_flags(trt))
    parser = trt.OnnxParser(network, logger)

    print(f"[export] parsing ONNX: {onnx_path}")
    onnx_bytes = onnx_path.read_bytes()
    if not parser.parse(onnx_bytes):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    # OPT: large workspace lets TRT pick faster tactics.
    workspace = int(workspace_gb * (1 << 30))
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)

    if want_fp16:
        _enable_fp16_if_supported(trt, builder, config)

    # Dynamic shape profile if the ONNX graph uses symbolic H/W.
    try:
        input_tensor = network.get_input(0)
        in_shape = list(input_tensor.shape)
        print(f"[export] ONNX input '{input_tensor.name}' shape={in_shape} dtype={input_tensor.dtype}")
        needs_profile = any(d == -1 for d in in_shape)
        if needs_profile:
            profile = builder.create_optimization_profile()
            # NCHW: freeze batch=1, lock H/W to the RoLux capture downsample size.
            fixed = (1, 3, height, width)
            profile.set_shape(input_tensor.name, fixed, fixed, fixed)
            config.add_optimization_profile(profile)
            print(f"[export] optimization profile locked to {fixed}")
        for i in range(network.num_outputs):
            out = network.get_output(i)
            print(f"[export] ONNX output '{out.name}' shape={list(out.shape)} dtype={out.dtype}")
    except Exception as exc:
        print(f"[export] profile setup skipped: {exc}")

    print("[export] building engine (this can take several minutes)…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise SystemExit("engine build returned None")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    print(f"[export] wrote {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")


def smoke_onnx_check(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        return
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print(f"[export] onnx.checker OK | ir={model.ir_version}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Depth Anything V2 ONNX → TensorRT engine")
    ap.add_argument("--onnx", type=Path, required=True, help="Path to .onnx checkpoint")
    ap.add_argument(
        "--engine",
        type=Path,
        default=Path("models/depth_anything_v2_vits_fp16.engine"),
        help="Output .engine path",
    )
    ap.add_argument("--height", type=int, default=392, help="Network input H")
    ap.add_argument("--width", type=int, default=392, help="Network input W")
    ap.add_argument("--workspace-gb", type=float, default=4.0, help="TRT workspace GB")
    ap.add_argument(
        "--fp32",
        action="store_true",
        help="Skip FP16 weak-typing flag (TRT≤10) / skip AutoCast hint",
    )
    ap.add_argument(
        "--autocast",
        action="store_true",
        help="Run ModelOpt AutoCast to FP16 before build (TRT 11 recommended for FP32 ONNX)",
    )
    args = ap.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"ONNX not found: {args.onnx}")

    onnx_path = args.onnx
    if args.autocast and not args.fp32:
        cast_out = onnx_path.with_name(onnx_path.stem + "_autocast_fp16.onnx")
        try:
            onnx_path = maybe_autocast(onnx_path, cast_out, args.height, args.width)
        except ImportError as exc:
            raise SystemExit(
                "FP16 AutoCast requires: pip install \"nvidia-modelopt[onnx]\"\n"
                f"Original error: {exc}"
            ) from exc

    smoke_onnx_check(onnx_path)
    build_engine(
        onnx_path=onnx_path,
        engine_path=args.engine,
        height=args.height,
        width=args.width,
        workspace_gb=args.workspace_gb,
        want_fp16=not args.fp32,
    )


if __name__ == "__main__":
    main()
