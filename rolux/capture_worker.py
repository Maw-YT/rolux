"""
Thread A — capture Roblox with a size-aware backend split:

  SMALL / windowed  → PrintWindow (HWND)
      DXGI desktop-dup only fires when the desktop is dirty; a small Roblox
      window on a static desktop starves (~20 FPS). PrintWindow follows the game.

  LARGE / fullscreen → DXGI grab()+crop
      PrintWindow copies the full client every frame (slow at 1080p+).
      DXGI was already buttery for fullscreen; use it when the client covers
      enough of the monitor that dirty updates keep up.

Crash avoidance for DXGI (hard-learned):
  - NEVER camera.start() — background DXGI thread races COM StageSurface → AV.
  - NEVER grab(region=...) — StageSurface rebuild on region crops → AV.
  - Only synchronous full-output grab() + NumPy crop is stable.
  - CoInitialize COM on this thread before touching DXGI.

Recording mode always uses PrintWindow so RoLux never reads its own overlay.
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

# Client area vs monitor area. Below this → PrintWindow; at/above → DXGI.
_DXGI_AREA_FRAC = 0.50


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
        self._dx_output_idx = 0
        self._pw = HwndCapturer()
        # When True: overlay is visible to recorders; force PrintWindow so we
        # never DXGI-capture our own overlay pixels.
        self.allow_screen_capture = bool(getattr(config, "allow_screen_capture", False))
        # Live-tunable (GUI "Shader render scale"); caps color_bgr bus size.
        self.shader_max_dim = max(256, int(getattr(config, "shader_max_dim", 960)))
        self._dxcam_ok = False
        self._dx_none_streak = 0

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
        cap = max(256, int(self.shader_max_dim))
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

    def _monitor_info_for_rect(self, rect: WindowRect) -> tuple[int, int, int, int, int]:
        """Return (output_idx_guess, mon_left, mon_top, mon_w, mon_h)."""
        cx = (int(rect.left) + int(rect.right)) // 2
        cy = (int(rect.top) + int(rect.bottom)) // 2
        mon = win32api.MonitorFromPoint((cx, cy), win32con.MONITOR_DEFAULTTONEAREST)
        info = win32api.GetMonitorInfo(mon)
        ml, mt, mr, mb = info["Monitor"]
        mon_w = max(1, int(mr - ml))
        mon_h = max(1, int(mb - mt))

        # Map monitor to dxcam output index by enumerating monitors in order.
        output_idx = 0
        try:
            mons = win32api.EnumDisplayMonitors(None, None)
            for i, (hmon, _hdc, _mrect) in enumerate(mons):
                if int(hmon) == int(mon):
                    output_idx = i
                    break
        except Exception:
            output_idx = 0
        return output_idx, int(ml), int(mt), mon_w, mon_h

    def _prefer_dxgi(self, rect: WindowRect) -> bool:
        """Large clients: DXGI. Small/windowed: PrintWindow."""
        try:
            _idx, _ml, _mt, mon_w, mon_h = self._monitor_info_for_rect(rect)
            client_area = max(1, int(rect.width) * int(rect.height))
            mon_area = max(1, mon_w * mon_h)
            return (client_area / mon_area) >= _DXGI_AREA_FRAC
        except Exception:
            # If we can't tell, prefer DXGI only for very large absolute sizes.
            return int(rect.width) * int(rect.height) >= 1280 * 720

    def _ensure_dxcam(self, rect: WindowRect) -> bool:
        output_idx, ml, mt, _mw, _mh = self._monitor_info_for_rect(rect)
        if self._camera is not None and self._dx_output_idx == output_idx:
            self._mon_origin = (ml, mt)
            return True

        # Recreate on monitor change.
        self._camera = None
        try:
            import dxcam

            try:
                import pythoncom

                pythoncom.CoInitialize()
            except Exception:
                pass

            self._camera = dxcam.create(
                output_idx=int(output_idx),
                device_idx=self.cfg.device_id,
                output_color="BGR",
                max_buffer_len=2,
            )
            self._dx_output_idx = int(output_idx)
            self._mon_origin = (ml, mt)
            self._dxcam_ok = True
            print(f"[Rolux] DXGI/dxcam ready (output={output_idx}, origin={ml},{mt})")
            return True
        except Exception as exc:
            print(f"[Rolux] dxcam unavailable ({exc})")
            self._camera = None
            self._dxcam_ok = False
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

    def _run_printwindow_frame(self, hwnd: int, focused: bool, reason: str = "") -> bool:
        captured = self._pw.grab(hwnd)
        if captured is None:
            return False
        bgr, rect = captured
        self._publish(bgr, rect, hwnd, focused, time.perf_counter())
        if self.allow_screen_capture:
            tag = "printwindow (recording)"
        elif reason:
            tag = f"printwindow ({reason})"
        else:
            tag = "printwindow"
        self.status["capture_backend"] = tag
        return True

    def run(self) -> None:
        try:
            import ctypes

            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 2
            )
        except Exception:
            pass

        print(
            "[Rolux] capture: PrintWindow when windowed/small, "
            f"DXGI when client ≥ {_DXGI_AREA_FRAC:.0%} of monitor"
        )
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

            rect = get_client_screen_rect(hwnd)
            if rect is None:
                time.sleep(0.002)
                continue

            # Recording: always HWND capture (never see our own overlay).
            if self.allow_screen_capture:
                if not self._run_printwindow_frame(hwnd, focused, reason="recording"):
                    time.sleep(0.002)
                continue

            want_dxgi = self._prefer_dxgi(rect)
            if want_dxgi and self._ensure_dxcam(rect):
                if self._run_dxcam_frame(hwnd, rect, focused):
                    self._dx_none_streak = 0
                    continue
                # No desktop frame this tick. A few misses are normal; sustained
                # misses (alt-tab chrome, etc.) → fall back to PrintWindow.
                self._dx_none_streak += 1
                if self._dx_none_streak < 8:
                    time.sleep(0.001)
                    continue

            # Small window, or DXGI starved / unavailable.
            if not self._run_printwindow_frame(
                hwnd, focused, reason=("windowed" if not want_dxgi else "dxgi-miss")
            ):
                time.sleep(0.002)
            else:
                self._dx_none_streak = 0

        self._camera = None
        self._pw.close()

    @property
    def current_rect(self) -> Optional[WindowRect]:
        return self._rect
