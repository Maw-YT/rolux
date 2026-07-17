"""Persist RoLux GUI / pipeline settings (separate from shader presets)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path("settings.json")
SETTINGS_VERSION = 1

_DEFAULTS: dict[str, Any] = {
    "version": SETTINGS_VERSION,
    "window_title": "Roblox",
    "network_size": 392,
    "target_fps": 144,
    "shader_max_dim": 960,
    "overlay_opacity": 1.0,
    "require_focus": True,
    "allow_screen_capture": False,
    "temporal": True,
    "engine_path": "models/depth_anything_v2_vits_fp16.engine",
    "last_preset": "",
    "geometry": "500x780",
}


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    data = dict(_DEFAULTS)
    try:
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key in _DEFAULTS:
                    if key == "version":
                        continue
                    if key in raw:
                        data[key] = raw[key]
    except Exception as exc:
        print(f"[Rolux] settings load failed ({exc}) — using defaults")
    return data


def save_settings(data: dict[str, Any], path: Path = SETTINGS_PATH) -> None:
    out = dict(_DEFAULTS)
    out.update(data)
    out["version"] = SETTINGS_VERSION
    try:
        path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"[Rolux] settings save failed: {exc}")
