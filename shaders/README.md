# RoLux fragment shaders

Drop any `*.glsl` file here. RoLux hot-reloads them and runs every file in
sorted filename order each frame.

Built-in uniforms (all optional to declare — missing ones are skipped):
  sampler2D uScene      — original Roblox RGB
  sampler2D uDepth      — depth buffer (grayscale in .r/.g/.b)
  sampler2D uDepthF     — float relative depth (R32F; sample `.r`, ~[0,1])
  sampler2D uNormal     — view-space normals from depth (GPU, not a height bump)
                          decode: vec3 n = texture(uNormal, vUV).xyz * 2.0 - 1.0;
  sampler2D uMain       — previous pass output (Roblox for the first pass)
  sampler2D uHistory    — PREVIOUS FRAME's resolved output (temporal reprojection)
  sampler2D uPrevDepthF — previous frame's depth (R32F) for disocclusion tests
  float     uHistoryValid — 1.0 if uHistory/uPrevDepthF hold a usable prior frame
  float     uTime       — seconds since start
  vec2      uResolution — buffer size in pixels

Decode normals in GLSL with:
  vec3 n = texture(uNormal, vUV).xyz * 2.0 - 1.0;

Required varyings / output when writing a full shader:
  in  vec2 vUV;
  out vec4 fragColor;

You can also drop a body-only snippet (no #version); RoLux wraps it.

The final pass is what the overlay shows (instead of raw depth).

## Included

- `00_depth_view.glsl` — grayscale depth visualization (debug)
- `05_fxaa.glsl` — fast approximate anti-aliasing
- `10_ssr.glsl` — screen-space reflections
- `15_ssrtgi.glsl` — ray-traced GI: ambient occlusion + one-bounce color bleed
- `18_fog.glsl` — depth-based distance fog
- `20_sharpen.glsl` — unsharp-mask sharpening
- `25_dof.glsl` — depth of field (depth-driven bokeh blur)
- `30_bloom.glsl` — bright-pass bloom / glow
- `40_saturation.glsl` — saturation + vibrance
- `45_tonemap.glsl` — Reinhard / ACES / filmic tonemapping
- `47_colorbalance.glsl` — white balance + lift/gain/gamma
- `50_gamma.glsl` — exposure / contrast / gamma
- `60_vignette.glsl` — darkened frame edges
- `65_chromatic.glsl` — chromatic aberration
- `70_grain.glsl` — animated film grain
- `80_lensdistortion.glsl` — barrel / pincushion distortion
- `90_letterbox.glsl` — cinematic bars

## Effect chain order

Passes run in sorted filename order, and each pass reads `uMain` = the previous
pass's output (`uScene` for the first pass). The numeric prefixes set a sensible
post order: reflections → sharpen → bloom → color grade.

The color passes (`20`–`50`) chain via `uMain`, so **disable
`00_depth_view.glsl`** when using them — otherwise it writes depth into `uMain`
and the color passes grade the depth buffer instead of the scene. `10_ssr.glsl`
reads `uScene` directly, so it's safe anywhere in the order.

## Temporal accumulation (history buffer)

The ShaderWorker keeps a persistent, ping-ponged **history buffer** of the last
resolved frame. After the effect chain runs, a built-in *temporal resolve*
blends the new frame with the reprojected history:

- **Depth rejection** — where `|uDepthF − uPrevDepthF|` exceeds
  `temporal_depth_reject`, history is discarded (handles motion / disocclusion).
- **Neighborhood color clamp** — history is clamped to the 3×3 range of the
  current frame, so ghosting is bounded even without motion vectors.

Because `10_ssr.glsl` and `15_ssrtgi.glsl` jitter their sampling every frame
(via `uTime`), the resolve averages that noise into a clean, stable image on
static scenes — this is the main payoff of the history buffer. Control it in
`RoluxConfig`: `temporal_accumulation` (on/off), `temporal_alpha` (history
weight, ~0.85), `temporal_depth_reject` (rejection threshold).

Any shader can also read `uHistory` / `uPrevDepthF` / `uHistoryValid` directly
to do its own per-pass temporal filtering.

## SSR (`10_ssr.glsl`)

Raymarched reflections built for monocular depth. It reconstructs a *pseudo*
view-space from an assumed FOV, reflects the view ray about the depth-derived
normal, marches it in screen space with binary refinement, and composites the
hit color onto reflective surfaces.

Because DepthAnythingV2 depth has no metric scale, the reflection geometry is
shaped by three knobs at the top of the file — tune these first:

- `FOV_DEG` — match your Roblox `FieldOfView` (default 70).
- `Z_NEAR` / `Z_FAR` — pseudo near/far. Widen `Z_FAR` for longer, flatter
  reflections; tighten for punchier local ones.

Other useful knobs:

- `REFLECT_MASK` — `1` reflects only up-facing surfaces (floors/water), `0`
  reflects everything. If reflections land on ceilings/walls, flip `UP_SIGN`.
- `INTENSITY`, `FRESNEL_POW`, `FRESNEL_MIN` — strength and grazing falloff.
- `THICKNESS` / `DEPTH_BIAS` — raise `THICKNESS` if reflections have gaps; raise
  `DEPTH_BIAS` if surfaces self-reflect (speckle).
- `ROUGHNESS` — `0` = mirror; higher blurs the reflection.
- `MAX_STEPS` / `RAY_MAX_DIST` — quality vs. cost (cheap either way).

Reflection stability depends on a stable depth scale. `RoluxConfig`
`stabilize_depth_range` (default on) EMA-smooths the per-frame depth
normalization so reflections don't boil; the shader-side temporal depth filter
in `shader_worker` further denoises the normals it reflects about.

The SSR pass reads `uScene` directly, so it works regardless of chain order.
For SSR-only output you can delete `00_depth_view.glsl` (otherwise it just runs
as a wasted pass before SSR).
