"""Central tuning knobs for the RoLux capture → TRT → overlay pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RoluxConfig:
    # --- Target window ---
    window_title_substring: str = "Roblox"

    # --- Capture ---
    target_fps: int = 144
    # When True: capture + overlay only while Roblox is the foreground window.
    require_focus: bool = True

    # --- Depth Anything V2 / TensorRT ---
    input_h: int = 392
    input_w: int = 392
    engine_path: Path = Path("models/depth_anything_v2_vits_fp16.engine")
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # --- Depth range stabilization (for SSR / normals) ---
    # DA-V2 output is renormalized per frame; a raw per-frame min/max makes the
    # depth scale flicker, which boils SSR reflections. Smooth the range with an
    # EMA that expands instantly (never clips new near/far) but contracts slowly.
    stabilize_depth_range: bool = True
    depth_range_alpha: float = 0.1  # contraction speed toward the current frame

    # --- Overlay ---
    # Depth-only cover: 1.0 = fully opaque (no Roblox pixels show through).
    overlay_opacity: float = 1.0
    depth_gain: float = 1.0

    # --- Shaders ---
    # Folder of *.glsl fragment shaders (hot-reloaded, run in sorted name order).
    shaders_dir: Path = Path("shaders")
    # Run the GLSL chain at this max edge (upscaled from network res). Higher = sharper, slower.
    shader_max_dim: int = 1280

    # --- Temporal accumulation (history buffer) ---
    # Denoises the composited output frame-to-frame. SSR/SSRTGI jitter their
    # sampling each frame, so accumulating history averages the noise into a
    # clean image on static scenes; depth rejection + neighborhood color
    # clamping discard history on motion/disocclusion to avoid ghosting.
    temporal_accumulation: bool = True
    temporal_alpha: float = 0.85         # history weight when accepted (0 = off, ~0.9 = strong)
    temporal_depth_reject: float = 0.02  # normalized depth delta that invalidates history
    # Edge-aware frame-to-frame smoothing of the depth map before normals/SSR.
    depth_temporal_filter: bool = True

    # --- CUDA ---
    device_id: int = 0
