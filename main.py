#!/usr/bin/env python3
"""
RoLux — TensorRT depth overlay for Roblox.

  python main.py

When frozen with PyInstaller it runs windowless; on first launch it drops an
editable ``shaders/`` folder next to the executable and uses that directory as
the working directory so the relative ``shaders/`` / ``models/`` / ``presets/``
paths resolve beside the .exe.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _bootstrap_frozen() -> None:
    """Prepare the runtime layout when running as a PyInstaller executable."""
    if not getattr(sys, "frozen", False):
        return

    exe_dir = Path(sys.executable).resolve().parent
    bundle_dir = Path(getattr(sys, "_MEIPASS", exe_dir))

    # Make relative resource paths (shaders/, models/, presets/, captures/)
    # resolve next to the executable rather than wherever it was launched from.
    try:
        os.chdir(exe_dir)
    except OSError:
        pass

    # Ship the shader effects with the exe, but expose an editable copy beside
    # it so users can tweak/add shaders and the GUI's temp-folder flow works.
    bundled_shaders = bundle_dir / "shaders"
    external_shaders = exe_dir / "shaders"
    if bundled_shaders.is_dir() and not external_shaders.exists():
        try:
            shutil.copytree(
                bundled_shaders,
                external_shaders,
                ignore=shutil.ignore_patterns("temp"),
            )
        except OSError:
            pass


_bootstrap_frozen()

from rolux.gui import run_gui


def main() -> int:
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
