"""PyInstaller runtime hook: make bundled TensorRT / CUDA DLLs discoverable.

The TensorRT bindings and cuda-python load their native DLLs by name, relying
on the OS loader search path. We drop the DLLs in ``_internal/tensorrt_libs``;
this hook registers that folder (and the bundle root) so both libraries resolve
their dependencies (nvinfer -> cublas/cudart, cuda.bindings -> cudart).
"""

import os
import sys

_base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))

for _sub in ("tensorrt_libs", "."):
    _d = os.path.join(_base, _sub)
    if os.path.isdir(_d):
        try:
            os.add_dll_directory(_d)
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")
