"""Win32 helpers: locate Roblox, track geometry, HWND bitmap capture, click-through."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

import numpy as np
import win32con
import win32gui
import win32ui

user32 = ctypes.windll.user32
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL
user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.SetWindowDisplayAffinity.restype = wintypes.BOOL


@dataclass(frozen=True)
class WindowRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def region(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


# PrintWindow flags
PW_CLIENTONLY = 0x1
PW_RENDERFULLCONTENT = 0x2  # needed for DX/D3D games like Roblox


def find_window_hwnd(title_substring: str) -> Optional[int]:
    needle = title_substring.lower()
    found: list[int] = []

    def _enum(hwnd: int, _: object) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if title and needle in title.lower():
            found.append(hwnd)

    win32gui.EnumWindows(_enum, None)
    return found[0] if found else None


def get_client_screen_rect(hwnd: int) -> Optional[WindowRect]:
    if not hwnd or not win32gui.IsWindow(hwnd):
        return None
    try:
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        origin = win32gui.ClientToScreen(hwnd, (left, top))
        br = win32gui.ClientToScreen(hwnd, (right, bottom))
        return WindowRect(origin[0], origin[1], br[0], br[1])
    except win32gui.error:
        return None


def is_window_foreground(hwnd: int) -> bool:
    try:
        return win32gui.GetForegroundWindow() == hwnd
    except win32gui.error:
        return False


def set_click_through(hwnd: int, enable: bool = True) -> None:
    try:
        style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        if enable:
            style |= win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOOLWINDOW
        else:
            style &= ~win32con.WS_EX_TRANSPARENT
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, style)
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0,
            0,
            0,
            0,
            win32con.SWP_NOMOVE
            | win32con.SWP_NOSIZE
            | win32con.SWP_NOACTIVATE
            | win32con.SWP_FRAMECHANGED,
        )
    except Exception as exc:
        print(f"[Rolux] set_click_through failed: {exc}")


# Windows 10 2004+: exclude this HWND from DXGI Desktop Duplication / screenshots.
WDA_NONE = 0x0
WDA_EXCLUDEFROMCAPTURE = 0x11


def exclude_from_capture(hwnd: int, enable: bool = True) -> None:
    """
    So DXGI (dxcam) sees Roblox *under* our overlay, not the overlay pixels.
    Critical for low-latency capture while covering the game with depth.
    """
    try:
        ok = user32.SetWindowDisplayAffinity(
            int(hwnd), WDA_EXCLUDEFROMCAPTURE if enable else WDA_NONE
        )
        if not ok:
            print("[Rolux] SetWindowDisplayAffinity failed (overlay may pollute DXGI capture)")
    except Exception as exc:
        print(f"[Rolux] exclude_from_capture: {exc}")


class HwndCapturer:
    """
    Reusable PrintWindow capturer — keeps GDI bitmap/DCs across frames.

    Recreating CreateCompatibleBitmap every frame was a major latency source.
    Size changes (resolution switch) trigger a one-shot rebuild.
    """

    def __init__(self) -> None:
        self._hwnd: Optional[int] = None
        self._w = 0
        self._h = 0
        self._hwnd_dc = None
        self._mfc_dc = None
        self._save_dc = None
        self._bitmap = None
        # Double buffer so inference can read frame N while we write N+1.
        self._buf_a: Optional[np.ndarray] = None
        self._buf_b: Optional[np.ndarray] = None
        self._use_a = True

    def close(self) -> None:
        self._release()

    def _release(self) -> None:
        try:
            if self._bitmap is not None:
                win32gui.DeleteObject(self._bitmap.GetHandle())
        except Exception:
            pass
        try:
            if self._save_dc is not None:
                self._save_dc.DeleteDC()
        except Exception:
            pass
        try:
            if self._mfc_dc is not None:
                self._mfc_dc.DeleteDC()
        except Exception:
            pass
        try:
            if self._hwnd_dc is not None and self._hwnd:
                win32gui.ReleaseDC(self._hwnd, self._hwnd_dc)
        except Exception:
            pass
        self._hwnd = None
        self._w = self._h = 0
        self._hwnd_dc = self._mfc_dc = self._save_dc = self._bitmap = None

    def _ensure(self, hwnd: int, w: int, h: int) -> bool:
        if (
            self._hwnd == hwnd
            and self._w == w
            and self._h == h
            and self._bitmap is not None
        ):
            return True
        self._release()
        try:
            hwnd_dc = win32gui.GetDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bitmap = win32ui.CreateBitmap()
            bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
            save_dc.SelectObject(bitmap)
            self._hwnd = hwnd
            self._w, self._h = w, h
            self._hwnd_dc = hwnd_dc
            self._mfc_dc = mfc_dc
            self._save_dc = save_dc
            self._bitmap = bitmap
            self._buf_a = np.empty((h, w, 3), dtype=np.uint8)
            self._buf_b = np.empty((h, w, 3), dtype=np.uint8)
            return True
        except Exception as exc:
            print(f"[Rolux] capturer init failed: {exc}")
            self._release()
            return False

    def grab(self, hwnd: int) -> Optional[tuple[np.ndarray, WindowRect]]:
        if not hwnd or not win32gui.IsWindow(hwnd):
            return None
        rect = get_client_screen_rect(hwnd)
        if rect is None or rect.width < 64 or rect.height < 64:
            return None
        w, h = rect.width, rect.height
        if not self._ensure(hwnd, w, h):
            return None
        assert self._save_dc is not None and self._bitmap is not None
        assert self._buf_a is not None and self._buf_b is not None

        ok = user32.PrintWindow(
            int(hwnd),
            int(self._save_dc.GetSafeHdc()),
            PW_CLIENTONLY | PW_RENDERFULLCONTENT,
        )
        if not ok and self._mfc_dc is not None:
            self._save_dc.BitBlt((0, 0), (w, h), self._mfc_dc, (0, 0), win32con.SRCCOPY)

        bits = self._bitmap.GetBitmapBits(True)
        src = np.frombuffer(bits, dtype=np.uint8)
        # Bitmap stride may pad rows to 4-byte boundaries; use reported size.
        info = self._bitmap.GetInfo()
        bw, bh = int(info["bmWidth"]), int(info["bmHeight"])
        bgra = src.reshape((bh, bw, 4))
        dst = self._buf_a if self._use_a else self._buf_b
        self._use_a = not self._use_a
        # Copy BGR into the free buffer (inference may still hold the other).
        np.copyto(dst, bgra[:h, :w, :3])
        return dst, rect


def capture_hwnd_bgr(hwnd: int) -> Optional[tuple[np.ndarray, WindowRect]]:
    """One-shot capture (prefer HwndCapturer in the hot loop)."""
    cap = HwndCapturer()
    try:
        return cap.grab(hwnd)
    finally:
        cap.close()
