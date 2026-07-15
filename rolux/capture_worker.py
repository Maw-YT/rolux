"""
Thread A — DXGI desktop capture (dxcam) cropped to Roblox.

Crash avoidance (hard-learned):
  - NEVER camera.start() — background DXGI thread races COM StageSurface → AV.
  - NEVER grab(region=...) — StageSurface rebuild on region crops → AV.
  - Only synchronous full-output grab() + NumPy crop is stable.
  - CoInitialize COM on this thread before touching DXGI.

Requires the overlay HWND to call SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)
when recording is off so DXGI sees Roblox under the overlay. When recording is
on (overlay visible to OBS etc.), internal capture switches to PrintWindow on
the Roblox HWND so RoLux never reads its own overlay back.

Falls back to PrintWindow if dxcam is unavailable.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import win32api
import win32con

from rolux.config import RoluxConfig
from rolux.win32_utils import (
    HwndCapturer,
    WindowRect,
    find_window_hwnd,
    get_client_screen_rect,
    is_window_foreground,
)


@dataclass
class CapturedFrame:
    bgr: np.ndarray  # network-sized (th, tw, 3) — depth model input
    color_bgr: np.ndarray  # display-res color (capped at shader_max_dim) — what you see
    rect: WindowRect
    capture_ts: float
    hwnd: int
    focused: bool


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        config: RoluxConfig,
        out_slot: list,
        slot_lock: threading.Lock,
        stop_event: threading.Event,
        status: Optional[dict] = None,
    ) -> None:
        super().__init__(name="RoluxCapture", daemon=True)
        self.cfg = config
        self.out_slot = out_slot
        self.slot_lock = slot_lock
        self.stop_event = stop_event
        self.status = status if status is not None else {}

        self._hwnd: Optional[int] = None
        self._rect: Optional[WindowRect] = None
        self._frames = 0
        self._cap_fps_frames = 0
        self._cap_fps_t = time.perf_counter()
        self._unfocus_streak = 0
        self._net_w = config.input_w
        self._net_h = config.input_h
        self._small = np.empty((self._net_h, self._net_w, 3), dtype=np.uint8)
        self._camera = None
        self._mon_origin = (0, 0)
        self._fallback = HwndCapturer()
        # When True: overlay is visible to recorders; we capture via PrintWindow
        # so RoLux never feeds back its own overlay. Live-toggled from the GUI.
        self.allow_screen_capture = bool(getattr(config, "allow_screen_capture", False))
        self._dxcam_ok = False

    def set_network_size(self, height: int, width: int) -> None:
        """Match capture downscale to the loaded TensorRT engine."""
        self._net_h = int(height)
        self._net_w = int(width)
        self._small = np.empty((self._net_h, self._net_w, 3), dtype=np.uint8)

    def _resolve_hwnd(self) -> Optional[int]:
        if self._hwnd is None or self._frames % 45 == 0:
            self._hwnd = find_window_hwnd(self.cfg.window_title_substring)
        if self._hwnd is not None:
            try:
                import win32gui

                if not win32gui.IsWindow(self._hwnd):
                    self._hwnd = None
            except Exception:
                self._hwnd = None
        return self._hwnd

    def _publish(
        self,
        bgr_roi: np.ndarray,
        rect: WindowRect,
        hwnd: int,
        focused: bool,
        capture_ts: float,
    ) -> None:
        if bgr_roi.size == 0 or float(bgr_roi.mean()) < 1.5:
            return
        cv2.resize(
            bgr_roi,
            (self._net_w, self._net_h),
            dst=self._small,
            interpolation=cv2.INTER_LINEAR,
        )
        small = self._small.copy()

        # Full-res color for display / shaders (capped at shader_max_dim so the
        # bus copy stays bounded). This is decoupled from the 392px depth input
        # so what you actually SEE keeps its native sharpness.
        cap = max(256, int(getattr(self.cfg, "shader_max_dim", 1280)))
        rh, rw = bgr_roi.shape[:2]
        cscale = min(1.0, float(cap) / float(max(rw, rh)))
        if cscale < 1.0:
            color = cv2.resize(
                bgr_roi,
                (max(1, int(rw * cscale)), max(1, int(rh * cscale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            color = bgr_roi.copy()
        color = np.ascontiguousarray(color)

        self._rect = rect
        with self.slot_lock:
            self.out_slot[0] = CapturedFrame(
                bgr=small,
                color_bgr=color,
                rect=rect,
                capture_ts=capture_ts,
                hwnd=hwnd,
                focused=focused,
            )
        self._frames += 1
        self._cap_fps_frames += 1
        now = time.perf_counter()
        if now - self._cap_fps_t >= 1.0:
            self.status["capture_fps"] = self._cap_fps_frames / max(
                1e-6, now - self._cap_fps_t
            )
            self._cap_fps_frames = 0
            self._cap_fps_t = now

    def _ensure_dxcam(self) -> bool:
        if self._camera is not None:
            return True
        try:
            import dxcam

            # COM must be initialized on this thread before DXGI.
            try:
                import pythoncom

                pythoncom.CoInitialize()
            except Exception:
                pass

            self._camera = dxcam.create(
                output_idx=0,
                device_idx=self.cfg.device_id,
                output_color="BGR",
                max_buffer_len=2,
            )
            try:
                primary = win32api.MonitorFromPoint((0, 0), win32con.MONITOR_DEFAULTTOPRIMARY)
                info = win32api.GetMonitorInfo(primary)
                left, top, _, _ = info["Monitor"]
                self._mon_origin = (int(left), int(top))
            except Exception:
                self._mon_origin = (0, 0)
            self.status["capture_backend"] = "dxcam"
            print("[Rolux] capture backend: DXGI/dxcam grab()+crop (no start/region)")
            return True
        except Exception as exc:
            print(f"[Rolux] dxcam unavailable ({exc})")
            self._camera = None
            return False

    def _crop(self, full: np.ndarray, rect: WindowRect) -> Optional[np.ndarray]:
        mon_left, mon_top = self._mon_origin
        left = int(rect.left - mon_left)
        top = int(rect.top - mon_top)
        right = int(rect.right - mon_left)
        bottom = int(rect.bottom - mon_top)
        h, w = full.shape[:2]
        left = max(0, min(left, w - 1))
        top = max(0, min(top, h - 1))
        right = max(left + 1, min(right, w))
        bottom = max(top + 1, min(bottom, h))
        if right - left < 64 or bottom - top < 64:
            return None
        return full[top:bottom, left:right]

    def _run_dxcam_frame(self, hwnd: int, rect: WindowRect, focused: bool) -> bool:
        """Grab one frame via DXGI crop. Returns True if a frame was published."""
        assert self._camera is not None
        ts = time.perf_counter()
        try:
            full = self._camera.grab()
        except Exception:
            return False
        if full is None:
            return False
        roi = self._crop(full, rect)
        if roi is None:
            return False
        self._publish(roi, rect, hwnd, focused, ts)
        self.status["capture_backend"] = "dxcam"
        return True

    def _run_printwindow_frame(self, hwnd: int, focused: bool) -> bool:
        """Grab one frame via PrintWindow. Returns True if a frame was published."""
        captured = self._fallback.grab(hwnd)
        if captured is None:
            return False
        bgr, rect = captured
        self._publish(bgr, rect, hwnd, focused, time.perf_counter())
        self.status["capture_backend"] = (
            "printwindow (recording)" if self.allow_screen_capture else "printwindow"
        )
        return True

    def run(self) -> None:
        try:
            import ctypes

            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 2
            )
        except Exception:
            pass

        self._dxcam_ok = self._ensure_dxcam()
        if self.allow_screen_capture:
            print(
                "[Rolux] recording mode: overlay visible to capture apps; "
                "internal capture via PrintWindow (no overlay feedback)"
            )

        while not self.stop_event.is_set():
            hwnd = self._resolve_hwnd()
            if hwnd is None:
                self.status.update(roblox_found=False, focused=False)
                time.sleep(0.05)
                continue

            focused = is_window_foreground(hwnd)
            self.status.update(roblox_found=True, focused=focused, hwnd=hwnd)
            if self.cfg.require_focus and not focused:
                self._unfocus_streak += 1
                if self._unfocus_streak >= 10:
                    time.sleep(0.002)
                    continue
            else:
                self._unfocus_streak = 0

            use_pw = self.allow_screen_capture or not self._dxcam_ok or self._camera is None
            if use_pw:
                if not self._run_printwindow_frame(hwnd, focused):
                    time.sleep(0.002)
                continue

            rect = get_client_screen_rect(hwnd)
            if rect is None:
                continue
            if not self._run_dxcam_frame(hwnd, rect, focused):
                time.sleep(0.0)

        self._camera = None
        self._fallback.close()

    @property
    def current_rect(self) -> Optional[WindowRect]:
        return self._rect
