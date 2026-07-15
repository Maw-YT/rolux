"""
Native depth overlay — dedicated thread, covers the Roblox client rect.

Uses win32gui.SetWindowPos (ctypes HWND_TOPMOST=-1 is unsafe on Win64) and
UpdateLayeredWindow so the cover always matches the live client size/position.
"""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from typing import Callable, Optional

import cv2
import numpy as np
import win32api
import win32con
import win32gui

from rolux.config import RoluxConfig
from rolux.inference_worker import DepthPacket
from rolux.win32_utils import exclude_from_capture, get_client_screen_rect

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
SW_HIDE = 0
SW_SHOWNOACTIVATE = 4
ULW_ALPHA = 0x00000002
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
DIB_RGB_COLORS = 0
BI_RGB = 0
PM_REMOVE = 0x0001

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
)


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", ctypes.c_long),
        ("biHeight", ctypes.c_long),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", ctypes.c_long),
        ("biYPelsPerMeter", ctypes.c_long),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", ctypes.c_uint),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class SIZE(ctypes.Structure):
    _fields_ = [("cx", ctypes.c_long), ("cy", ctypes.c_long)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
    ]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", wintypes.BYTE),
        ("BlendFlags", wintypes.BYTE),
        ("SourceConstantAlpha", wintypes.BYTE),
        ("AlphaFormat", wintypes.BYTE),
    ]


user32.CreateWindowExW.argtypes = [
    wintypes.DWORD,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.DWORD,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HWND,
    wintypes.HMENU,
    wintypes.HINSTANCE,
    wintypes.LPVOID,
]
user32.CreateWindowExW.restype = wintypes.HWND
user32.RegisterClassW.argtypes = [ctypes.POINTER(WNDCLASS)]
user32.RegisterClassW.restype = wintypes.ATOM
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.UpdateLayeredWindow.argtypes = [
    wintypes.HWND,
    wintypes.HDC,
    ctypes.POINTER(POINT),
    ctypes.POINTER(SIZE),
    wintypes.HDC,
    ctypes.POINTER(POINT),
    wintypes.COLORREF,
    ctypes.POINTER(BLENDFUNCTION),
    wintypes.DWORD,
]
user32.UpdateLayeredWindow.restype = wintypes.BOOL
gdi32.CreateDIBSection.argtypes = [
    wintypes.HDC,
    ctypes.POINTER(BITMAPINFO),
    wintypes.UINT,
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.HANDLE,
    wintypes.DWORD,
]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]
gdi32.SelectObject.restype = wintypes.HANDLE
gdi32.DeleteObject.argtypes = [wintypes.HANDLE]
gdi32.DeleteDC.argtypes = [wintypes.HDC]


@WNDPROC
def _wnd_proc(hwnd, msg, wparam, lparam):
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


_CLASS = "RoLuxDepthOverlayCover"
_registered = False


def ensure_dpi_aware() -> None:
    """Match physical pixel coords with Roblox (Per-Monitor DPI)."""
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def _register() -> None:
    global _registered
    if _registered:
        return
    wc = WNDCLASS()
    wc.lpfnWndProc = _wnd_proc
    wc.hInstance = kernel32.GetModuleHandleW(None)
    wc.hCursor = user32.LoadCursorW(None, 32512)
    wc.lpszClassName = _CLASS
    user32.RegisterClassW(ctypes.byref(wc))
    _registered = True


class DepthOverlay(threading.Thread):
    """Owns the overlay HWND on this thread; always sized to the Roblox client."""

    UNFOCUS_HIDE_TICKS = 10

    def __init__(
        self,
        config: RoluxConfig,
        depth_slot: list,
        depth_lock: threading.Lock,
        status: dict,
        stop_event: threading.Event,
        frame_ready: threading.Event,
        on_stats: Optional[Callable[[float, float, float], None]] = None,
    ) -> None:
        super().__init__(name="RoluxOverlay", daemon=True)
        self.cfg = config
        self.depth_slot = depth_slot
        self.depth_lock = depth_lock
        self.status = status
        self.stop_event = stop_event
        self.frame_ready = frame_ready
        self.on_stats = on_stats

        self._opacity = float(max(0.05, min(1.0, config.overlay_opacity)))
        self._hwnd: Optional[int] = None
        self._visible = False
        self._last_ts = -1.0
        self._has_frame = False
        self._last_gray: Optional[np.ndarray] = None
        self._unfocus_streak = 0
        self._frames = 0
        self._t_fps = time.perf_counter()
        self._last_infer_ms = 0.0
        self._last_e2e_ms = 0.0
        self._ready = threading.Event()
        self._placed_once = False
        self._last_place: Optional[tuple[int, int, int, int]] = None

        self._net_w = config.input_w
        self._net_h = config.input_h

        # MemDC + DIB resized when Roblox client size changes.
        self._dib_w = 0
        self._dib_h = 0
        self._hdc_screen = None
        self._hdc_mem = None
        self._hbmp = None
        self._old_bmp = None
        self._bits = None

    @property
    def hwnd(self) -> Optional[int]:
        return self._hwnd

    def wait_ready(self, timeout: float = 2.0) -> bool:
        return self._ready.wait(timeout)

    def set_opacity(self, value: float) -> None:
        self._opacity = float(max(0.05, min(1.0, value)))
        # Next UpdateLayeredWindow applies the new constant alpha.
        if self._visible and self._last_gray is not None:
            rect = self._live_rect()
            if rect is not None:
                self._present(self._last_gray, rect)

    def destroy(self) -> None:
        self.stop_event.set()
        self.frame_ready.set()
        self.join(timeout=2.0)

    def _create(self) -> None:
        ensure_dpi_aware()
        _register()
        ex = (
            WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOPMOST
            | WS_EX_TOOLWINDOW
            | WS_EX_NOACTIVATE
        )
        # Start 1x1 hidden — real size comes from live Roblox rect.
        hwnd = user32.CreateWindowExW(
            ex,
            _CLASS,
            "RoLux Overlay",
            WS_POPUP,
            0,
            0,
            1,
            1,
            None,
            None,
            kernel32.GetModuleHandleW(None),
            None,
        )
        if not hwnd:
            raise ctypes.WinError()
        self._hwnd = int(hwnd)
        exclude_from_capture(self._hwnd, enable=True)
        user32.ShowWindow(self._hwnd, SW_HIDE)
        self._ready.set()
        print(f"[Rolux] overlay hwnd={self._hwnd} (UpdateLayeredWindow cover)")

    def set_exclude_from_capture(self, enable: bool) -> None:
        """
        When True (default), Win+PrintScreen / Snipping Tool cannot see the overlay
        (needed so DXGI captures Roblox underneath). Set False to allow screenshots.
        """
        if self._hwnd:
            exclude_from_capture(self._hwnd, enable=bool(enable))
            print(
                "[Rolux] overlay "
                + ("hidden from screenshots/DXGI" if enable else "VISIBLE to screenshots")
            )

    def _free_dib(self) -> None:
        if self._hdc_mem and self._old_bmp:
            gdi32.SelectObject(self._hdc_mem, self._old_bmp)
        if self._hbmp:
            gdi32.DeleteObject(self._hbmp)
        if self._hdc_mem:
            gdi32.DeleteDC(self._hdc_mem)
        if self._hdc_screen:
            user32.ReleaseDC(None, self._hdc_screen)
        self._hdc_screen = None
        self._hdc_mem = None
        self._hbmp = None
        self._old_bmp = None
        self._bits = None
        self._dib_w = 0
        self._dib_h = 0

    def _ensure_dib(self, width: int, height: int) -> bool:
        if width < 2 or height < 2:
            return False
        if self._dib_w == width and self._dib_h == height and self._bits is not None:
            return True
        self._free_dib()

        self._hdc_screen = user32.GetDC(None)
        self._hdc_mem = gdi32.CreateCompatibleDC(self._hdc_screen)
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        bits = ctypes.c_void_p()
        self._hbmp = gdi32.CreateDIBSection(
            self._hdc_mem,
            ctypes.byref(bmi),
            DIB_RGB_COLORS,
            ctypes.byref(bits),
            None,
            0,
        )
        if not self._hbmp or not bits:
            self._free_dib()
            return False
        self._old_bmp = gdi32.SelectObject(self._hdc_mem, self._hbmp)
        self._bits = bits
        self._dib_w = width
        self._dib_h = height
        return True

    def _live_rect(self):
        hwnd = self.status.get("hwnd")
        if hwnd:
            rect = get_client_screen_rect(int(hwnd))
            if rect is not None and rect.width >= 64 and rect.height >= 64:
                return rect
        return None

    def _hide(self) -> None:
        if self._visible and self._hwnd:
            win32gui.ShowWindow(self._hwnd, win32con.SW_HIDE)
            self._visible = False

    def _present(self, image: np.ndarray, rect) -> None:
        """Stretch depth/shader BGR to client size and cover Roblox via UpdateLayeredWindow."""
        assert self._hwnd is not None
        left, top = int(rect.left), int(rect.top)
        width, height = int(rect.width), int(rect.height)
        if width < 64 or height < 64:
            return

        # Accept HxW gray, HxWx1, or HxWx3 BGR (shader output).
        if image.ndim == 2:
            bgr = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 1:
            d = image[:, :, 0]
            bgr = np.stack([d, d, d], axis=-1)
        else:
            bgr = image[:, :, :3]

        if bgr.shape[0] != height or bgr.shape[1] != width:
            scaled = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        else:
            scaled = bgr
        scaled = np.ascontiguousarray(scaled, dtype=np.uint8)

        if not self._ensure_dib(width, height):
            return

        buf = (ctypes.c_ubyte * (width * height * 4)).from_address(self._bits.value)
        bgra = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        bgra[:, :, 0] = scaled[:, :, 0]
        bgra[:, :, 1] = scaled[:, :, 1]
        bgra[:, :, 2] = scaled[:, :, 2]
        bgra[:, :, 3] = 255

        # Move/size with win32gui — safe HWND_TOPMOST on Win64.
        flags = (
            win32con.SWP_NOACTIVATE
            | win32con.SWP_SHOWWINDOW
            | win32con.SWP_NOCOPYBITS
        )
        win32gui.SetWindowPos(
            self._hwnd,
            win32con.HWND_TOPMOST,
            left,
            top,
            width,
            height,
            flags,
        )

        pt_dst = POINT(left, top)
        size = SIZE(width, height)
        pt_src = POINT(0, 0)
        blend = BLENDFUNCTION(
            AC_SRC_OVER, 0, int(self._opacity * 255) & 0xFF, 0  # constant alpha
        )
        ok = user32.UpdateLayeredWindow(
            self._hwnd,
            None,
            ctypes.byref(pt_dst),
            ctypes.byref(size),
            self._hdc_mem,
            ctypes.byref(pt_src),
            0,
            ctypes.byref(blend),
            ULW_ALPHA,
        )
        if not ok and not self._placed_once:
            err = ctypes.GetLastError()
            print(f"[Rolux] UpdateLayeredWindow failed err={err}")
        self._placed_once = True
        self._visible = True

    def _should_show(self, found: bool, focused: bool) -> bool:
        if not found:
            self._unfocus_streak = self.UNFOCUS_HIDE_TICKS
            return False
        if not self.cfg.require_focus:
            self._unfocus_streak = 0
            return True
        if focused:
            self._unfocus_streak = 0
            return True
        self._unfocus_streak += 1
        return self._unfocus_streak < self.UNFOCUS_HIDE_TICKS

    def _pump(self) -> None:
        msg = MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def run(self) -> None:
        try:
            ctypes.windll.kernel32.SetThreadPriority(
                ctypes.windll.kernel32.GetCurrentThread(), 15
            )
        except Exception:
            pass

        self._create()

        while not self.stop_event.is_set():
            self.frame_ready.wait(timeout=0.002)
            self.frame_ready.clear()
            self._pump()

            with self.depth_lock:
                packet: Optional[DepthPacket] = self.depth_slot[0]

            found = bool(self.status.get("roblox_found", False))
            focused = bool(self.status.get("focused", False))
            show = self._should_show(found, focused)
            self.status["overlay_visible"] = bool(show and self._has_frame)

            if not show:
                self._hide()
                continue

            rect = self._live_rect()
            if rect is None and packet is not None:
                rect = packet.rect
            if rect is None:
                continue

            # New depth frame, or geometry changed — keep cover locked to Roblox.
            new_frame = packet is not None and packet.capture_ts != self._last_ts
            if new_frame:
                assert packet is not None
                self._last_gray = packet.rgb
                self._last_ts = packet.capture_ts
                self._last_infer_ms = packet.infer_ms
                self._has_frame = True
                self._frames += 1
                now = time.perf_counter()
                self._last_e2e_ms = (now - packet.capture_ts) * 1000.0
                if now - self._t_fps >= 1.0:
                    fps = self._frames / max(1e-6, now - self._t_fps)
                    self.status["overlay_fps"] = fps
                    self.status["e2e_ms"] = self._last_e2e_ms
                    if self.on_stats:
                        try:
                            self.on_stats(fps, self._last_infer_ms, self._last_e2e_ms)
                        except Exception:
                            pass
                    self._frames = 0
                    self._t_fps = now

            if self._last_gray is None:
                continue

            place = (rect.left, rect.top, rect.width, rect.height)
            geom_changed = place != self._last_place
            if new_frame or geom_changed or not self._visible:
                self._present(self._last_gray, rect)
                self._last_place = place

        self._free_dib()
        if self._hwnd:
            try:
                win32gui.DestroyWindow(self._hwnd)
            except Exception:
                pass
            self._hwnd = None
