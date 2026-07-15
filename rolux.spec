# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for RoLux — no-console onedir build (dist/RoLux/RoLux.exe).

TensorRT/CUDA ship gigabytes of DLLs; RoLux only *runs* prebuilt engines, so we
bundle just the runtime libs (nvinfer + nvinfer_plugin + the CUDA DLLs they
depend on) and skip the ~1.8 GB nvinfer_builder_resource_* build-only blobs.

Build:  pyinstaller rolux.spec --noconfirm
Run:    dist/RoLux/RoLux.exe   (put your models/ next to it — see README)
"""

import glob
import importlib.util
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules


def _pkg_dir(name):
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.origin and os.path.basename(spec.origin) == "__init__.py":
        return os.path.dirname(spec.origin)
    if spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    return None


datas = [("shaders", "shaders")]
binaries = []
hiddenimports = [
    "OpenGL.platform.win32",
    "OpenGL.arrays.ctypesarrays",
    "OpenGL.arrays.numpymodule",
    "OpenGL.arrays.lists",
    "OpenGL.arrays.numbers",
    "OpenGL.arrays.strings",
    "win32gui",
    "win32con",
    "win32api",
    "win32ui",
    "pywintypes",
    "tensorrt",
    "tensorrt_bindings",
    "tensorrt_libs",
]
hiddenimports += collect_submodules("OpenGL")

# --- TensorRT python bindings (the tensorrt.pyd + pure-python wrappers) -------
for pkg in ("tensorrt", "tensorrt_bindings"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# --- cuda-python bindings -----------------------------------------------------
for pkg in ("cuda.bindings", "cuda"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# --- dxcam --------------------------------------------------------------------
try:
    d, b, h = collect_all("dxcam")
    datas += d
    binaries += b
    hiddenimports += h
except Exception:
    pass

# --- Native runtime DLLs (hand-picked; skip the giant builder-only blobs) -----
_trt_libs = _pkg_dir("tensorrt_libs")
if _trt_libs:
    # tensorrt_libs/__init__.py ctypes-loads every DLL beside it; ship only the
    # runtime ones. It skips nvinfer_builder_resource_* itself at import time.
    for _name in ("nvinfer_11.dll", "nvinfer_plugin_11.dll", "nvonnxparser_11.dll"):
        _p = os.path.join(_trt_libs, _name)
        if os.path.isfile(_p):
            binaries.append((_p, "tensorrt_libs"))
    _init = os.path.join(_trt_libs, "__init__.py")
    if os.path.isfile(_init):
        datas.append((_init, "tensorrt_libs"))

# CUDA DLLs that nvinfer (and cuda.bindings) link against. Placed alongside
# nvinfer in tensorrt_libs so the OS loader resolves them from that directory.
_nvidia = _pkg_dir("nvidia")
if _nvidia:
    for _name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"):
        _matches = glob.glob(os.path.join(_nvidia, "**", _name), recursive=True)
        if _matches:
            binaries.append((_matches[0], "tensorrt_libs"))


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["rthook_cuda.py"],
    excludes=["matplotlib", "pytest", "onnx", "onnxruntime", "onnxconverter_common"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RoLux",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="RoLux",
)
