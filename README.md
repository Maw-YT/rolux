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

Depth engines are **not** committed (GPU/driver specific). Build one locally
with NVIDIA’s `trtexec`, then point the GUI at the `.engine`.

### Get `trtexec`

1. Join the free [NVIDIA Developer Program](https://developer.nvidia.com/tensorrt)
   and open the TensorRT download page.
2. Download the **TensorRT zip** for your CUDA version (Windows x64).
3. Unzip it somewhere permanent (e.g. `C:\TensorRT`).
4. Add the `bin` folder to your `PATH` (that directory contains `trtexec.exe`).

   ```powershell
   $env:Path = "C:\TensorRT\bin;" + $env:Path
   trtexec --help
   ```

### Build an engine

1. Put a Depth Anything V2 ONNX in `models/` (e.g. ViT-S from
   [fabio-sim/Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX)).
2. Build with `trtexec`:

   ```bash
   trtexec --onnx=models/depth_anything_v2_vits_fp16.onnx ^
     --saveEngine=models/depth_anything_v2_vits_fp16.engine ^
     --fp16 ^
     --minShapes=image:1x3x392x392 ^
     --optShapes=image:1x3x392x392 ^
     --maxShapes=image:1x3x392x392
   ```

3. In the GUI, select your `.engine` (default path is
   `models/depth_anything_v2_vits_fp16.engine`).


## Run

```bash
python main.py
```

Then in the control panel: pick your engine, **Start**, and click into Roblox.
The overlay covers the client area and follows it as you move/resize the window.
Unfocus Roblox to hide it.

## Build the .exe (PyInstaller)

RoLux ships a no-console onedir build via [`rolux.spec`](rolux.spec). The frozen
app starts windowless and, on first launch, copies the bundled `shaders/` folder
next to the executable so you can edit effects beside the `.exe`.

1. Install dependencies (including PyInstaller) in your venv:

   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. Build from the repo root:

   ```bash
   python -m PyInstaller rolux.spec --noconfirm
   ```

3. Output:

   ```
   dist/RoLux/RoLux.exe
   dist/RoLux/_internal/   # runtime DLLs / Python libs
   ```

4. Next to `RoLux.exe`, place the runtime folders the app expects:

   ```
   dist/RoLux/
     RoLux.exe
     models/          # your .engine (e.g. depth_anything_v2_vits_fp16.engine)
     presets/         # optional — copy from the repo presets/
     settings.json    # optional — your saved GUI settings
     shaders/         # auto-created on first launch if missing
   ```

   Example (PowerShell):

   ```powershell
   Copy-Item -Recurse models\*.engine dist\RoLux\models\ -ErrorAction SilentlyContinue
   New-Item -ItemType Directory -Force dist\RoLux\models, dist\RoLux\presets | Out-Null
   Copy-Item presets\* dist\RoLux\presets\ -ErrorAction SilentlyContinue
   Copy-Item settings.json dist\RoLux\ -ErrorAction SilentlyContinue
   ```

5. Run `dist\RoLux\RoLux.exe` from that folder (working directory must be the
   `dist\RoLux` directory so relative `models/` / `presets/` / `settings.json`
   paths resolve).

Notes:

- Rebuild after pulling shader or Python changes: the same `python -m PyInstaller
  rolux.spec --noconfirm` command overwrites `dist/RoLux/`.
- TensorRT **builder** blobs are intentionally not bundled (large / unused at
  runtime). You still need a local NVIDIA driver + CUDA-compatible GPU to run
  engines.
- `build/` and `dist/` are gitignored; do not commit them.

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
rolux.spec            PyInstaller onedir build (-> dist/RoLux/RoLux.exe)
rthook_cuda.py        PyInstaller runtime hook for TensorRT/CUDA DLLs
rolux/
  config.py           tuning knobs
  app_settings.py     persistent GUI settings (settings.json)
  capture_worker.py   DXGI (large/FS) + PrintWindow (small/windowed)
  inference_worker.py TensorRT depth inference
  shader_worker.py    GLSL chain + normals + temporal resolve
  overlay_ui.py       layered click-through overlay window
  gui.py              control panel
  presets.py          preset save/load + #define parsing
  win32_utils.py      window/geometry/capture helpers
  normals.py          CPU depth->normals fallback
shaders/              hot-reloaded *.glsl effects (+ presets in presets/)
models/               your TensorRT .engine (gitignored; build with trtexec)
presets/              saved effect-chain presets (JSON)
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
