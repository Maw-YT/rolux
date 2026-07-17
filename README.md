# RoLux

A ReShade-style post-processing overlay for **Roblox** — with **no hooking, no
injection, and no DLLs**. RoLux captures the game window off the desktop
compositor, estimates a depth buffer with **Depth Anything V2** (TensorRT),
derives view-space normals from it, and runs a hot-reloadable GLSL effect chain
(SSR, ray-traced GI/AO, bloom, DOF, color grading, and more) on top, presenting
the result through a click-through, capture-excluded overlay window.

Because it only reads the framebuffer that Windows already shows and paints an
overlay on top, RoLux never touches the Roblox process.

> ⚠️ Depth here is **monocular / relative** (no real camera matrices), so
> effects like SSR/GI are convincing but stylized approximations, not physically
> exact. See per-shader notes in `shaders/README.md`.

## How it works

```
capture (DXGI if large / PrintWindow if small) ──▶ client BGR
        │                       │
        │ (full-res color)      │ (392px downscale)
        ▼                       ▼
   scene texture          Depth Anything V2 (TensorRT, FP16)
        │                       │  depth buffer
        │                       ▼
        │                 depth → view-space normals (GPU)
        ▼                       ▼
   GLSL effect chain (hot-reloaded *.glsl, sorted by name)
        ▼
   temporal resolve (history buffer, depth-rejected accumulation)
        ▼
   layered click-through overlay (WDA_EXCLUDEFROMCAPTURE)
```

Each stage runs on its own thread (capture / inference / shaders / overlay)
connected by lock-guarded latest-frame slots.

## Requirements

- Windows 10 2004+ (needs `WDA_EXCLUDEFROMCAPTURE`)
- NVIDIA GPU with **CUDA 12.x** and **TensorRT 10+** installed
- Python 3.10+

```bash
pip install -r requirements.txt
```

## Model setup

The depth model is **not** committed (TensorRT engines are GPU/driver specific
and the ONNX weights are large). Build the engine locally:

1. Grab a Depth Anything V2 ONNX (e.g. the ViT-S export from
   [fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX))
   and place it in `models/`.
2. Build a TensorRT engine:

   ```bash
   # FP16 ONNX:
   python export_trt.py --onnx models/depth_anything_v2_vits_fp16.onnx --height 392 --width 392
   # or auto-cast an FP32 ONNX to mixed FP16 (TRT 11):
   python export_trt.py --onnx models/depth_anything_v2_vits.onnx --autocast --height 392 --width 392
   ```

3. The default engine path is `models/depth_anything_v2_vits_fp16.engine`
   (changeable in the GUI).

## Run

```bash
python main.py
```

Then in the control panel: pick your engine, **Start**, and click into Roblox.
The overlay covers the client area and follows it as you move/resize the window.
Unfocus Roblox to hide it.

## Features

- **Depth + normals** from Depth Anything V2 (TensorRT FP16), full-res color kept
  separate from the 392px depth input so the image stays sharp.
- **Hot-reloaded GLSL chain** — drop a `*.glsl` in `shaders/`; it recompiles live.
- **ReShade-style GUI** — tabbed, dark, with an effect toggle list and live
  sliders that edit each shader's `#define` values on disk.
- **Presets** — save/load the whole chain (enabled effects + all values) as JSON.
- **Persistent settings** — GUI knobs (render scale, FPS, opacity, engine path,
  last preset, etc.) restore from `settings.json` on launch.
- **Temporal accumulation** — a history buffer denoises SSR/GI via depth-rejected
  accumulation with neighborhood color clamping (toggleable).
- **Session sandbox** — edits go to a throwaway `shaders/temp` copy; the pristine
  originals in `shaders/` are the defaults, restorable with the **Reset** button.

### Included effects

`fxaa`, `ssr` (screen-space reflections), `ssrtgi` (ray-traced GI: AO + one-bounce
color bleed), `fog`, `sharpen`, `dof`, `bloom`, `saturation`, `tonemap`,
`colorbalance`, `gamma`, `vignette`, `chromatic aberration`, `film grain`,
`lens distortion`, `letterbox`. See [`shaders/README.md`](shaders/README.md) for
uniforms, chain order, and tuning.

## Project layout

```
main.py               entry point
export_trt.py         ONNX → TensorRT engine builder
rolux/
  config.py           tuning knobs
  capture_worker.py   DXGI (large/FS) + PrintWindow (small/windowed)
  inference_worker.py TensorRT depth inference
  shader_worker.py    GLSL chain + normals + temporal resolve
  overlay_ui.py       layered click-through overlay window
  gui.py              control panel
  presets.py          preset save/load + #define parsing
  win32_utils.py      window/geometry/capture helpers
  normals.py          CPU depth→normals fallback
shaders/              hot-reloaded *.glsl effects (+ presets in presets/)
models/               build your TensorRT engine here (gitignored weights)
```

## Notes / limitations

- Monocular depth is renormalized per frame; RoLux EMA-stabilizes the range, but
  fast camera motion can still cause mild shimmer.
- No motion vectors, so temporal accumulation reprojects at same-UV with depth
  rejection — great for static/slow scenes, falls back to the current frame on
  motion.
- `third_party/` (RT-MonoDepth, ZipDepth experiments) is not committed.

## Disclaimer

RoLux does not read, modify, or inject into the Roblox process — it only
post-processes what is already displayed and overlays a separate window. Use at
your own discretion and in line with Roblox's terms.
