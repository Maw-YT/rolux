"""
RoLux presets — save/load the whole effect chain (ReShade-style).

A preset captures, for every shader in the folder:
  - enabled: whether it runs (``name.glsl`` vs disabled ``name.glsl.off``)
  - params:  the numeric ``#define`` values

Applying a preset renames files to match the enabled state and rewrites the
``#define`` values in place, which the ShaderWorker hot-reloads.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DEFINE_RE = re.compile(r"^(#define\s+(\w+)\s+)(-?\d+\.?\d*)(.*)$")
_ANNOT_RE = re.compile(
    r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]\s*(.*)"
)

PRESET_EXT = ".json"
PRESET_VERSION = 1


@dataclass
class ShaderParam:
    name: str
    value: float
    is_int: bool
    vmin: float
    vmax: float
    step: float
    desc: str
    line_idx: int
    prefix: str
    suffix: str


def format_value(value: float, is_int: bool) -> str:
    """Numeric -> GLSL literal (ints stay ints, floats keep a decimal point)."""
    if is_int:
        return str(int(round(value)))
    s = f"{float(value):.4f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def parse_params(path: Path) -> tuple[list[str], list[ShaderParam]]:
    """Extract slider-editable ``#define NAME value // [min,max,step] desc``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return [], []
    lines = text.splitlines()
    params: list[ShaderParam] = []
    for i, line in enumerate(lines):
        m = _DEFINE_RE.match(line)
        if not m:
            continue
        prefix, name, valstr, rest = m.group(1), m.group(2), m.group(3), m.group(4)
        is_int = "." not in valstr
        value = float(valstr)
        vmin, vmax, step, desc = 0.0, max(1.0, value * 2.0), (1.0 if is_int else 0.05), ""
        am = _ANNOT_RE.search(rest)
        if am:
            vmin, vmax, step = float(am.group(1)), float(am.group(2)), float(am.group(3))
            desc = am.group(4).strip()
        params.append(
            ShaderParam(name, value, is_int, vmin, vmax, step, desc, i, prefix, rest)
        )
    return lines, params


def write_param_values(path: Path, values: dict[str, float]) -> None:
    """Rewrite the ``#define`` values named in ``values`` (others untouched)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    changed = False
    for i, line in enumerate(lines):
        m = _DEFINE_RE.match(line)
        if not m:
            continue
        name = m.group(2)
        if name not in values:
            continue
        is_int = "." not in m.group(3)
        lines[i] = f"{m.group(1)}{format_value(values[name], is_int)}{m.group(4)}"
        changed = True
    if changed:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def shader_files(shaders_dir: Path) -> list[Path]:
    if not shaders_dir.is_dir():
        return []
    found = list(shaders_dir.glob("*.glsl")) + list(shaders_dir.glob("*.glsl.off"))
    return sorted(found, key=lambda p: p.name)


def _base_name(path: Path) -> str:
    """Enabled/disabled file -> canonical ``name.glsl`` key."""
    return path.name[:-4] if path.suffix == ".off" else path.name


def collect_preset(shaders_dir: Path) -> dict:
    """Snapshot current enabled state + param values of every shader."""
    shaders: dict[str, dict] = {}
    for path in shader_files(shaders_dir):
        _, params = parse_params(path)
        shaders[_base_name(path)] = {
            "enabled": path.suffix == ".glsl",
            "params": {p.name: (int(p.value) if p.is_int else p.value) for p in params},
        }
    return {"version": PRESET_VERSION, "shaders": shaders}


def apply_preset(shaders_dir: Path, data: dict) -> None:
    """Rename files to match `enabled` and rewrite `#define` values."""
    shaders = data.get("shaders", {})
    for base, cfg in shaders.items():
        enabled_path = shaders_dir / base
        disabled_path = shaders_dir / (base + ".off")
        current = enabled_path if enabled_path.is_file() else (
            disabled_path if disabled_path.is_file() else None
        )
        if current is None:
            continue  # shader referenced by preset isn't present

        want_enabled = bool(cfg.get("enabled", True))
        target = enabled_path if want_enabled else disabled_path
        if current != target:
            try:
                current.rename(target)
            except OSError:
                target = current  # keep going, still write params
        params = cfg.get("params", {})
        if params:
            write_param_values(target, {k: float(v) for k, v in params.items()})


def list_presets(presets_dir: Path) -> list[Path]:
    if not presets_dir.is_dir():
        return []
    return sorted(presets_dir.glob(f"*{PRESET_EXT}"), key=lambda p: p.name.lower())


def save_preset(presets_dir: Path, name: str, data: dict) -> Path:
    presets_dir.mkdir(parents=True, exist_ok=True)
    stem = name[:-len(PRESET_EXT)] if name.endswith(PRESET_EXT) else name
    path = presets_dir / f"{stem}{PRESET_EXT}"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_preset(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
