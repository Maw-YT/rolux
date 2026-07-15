"""
Real-time GLSL post chain for RoLux.

Drop any ``*.glsl`` fragment shaders into the ``shaders/`` folder (next to
``main.py``). They are hot-reloaded and run in sorted filename order.

Each pass receives:
  uScene   — original Roblox RGB
  uDepth   — depth buffer (grayscale in .r / .g / .b)
  uDepthF  — float relative depth (R32F, sample .r, ~[0,1])
  uNormal  — view-space normals from depth (GPU, RGB = 0.5+0.5*n)
  uMain    — previous pass output (Roblox RGB for the first pass)
  uTime    — seconds since worker start
  uResolution — vec2(width, height)

The final pass replaces the raw depth overlay.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from rolux.config import RoluxConfig
from rolux.inference_worker import DepthPacket

_VERT = """
#version 330 core
const vec2 POS[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
out vec2 vUV;
void main() {
    vec2 p = POS[gl_VertexID];
    vUV = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

# Fallback if shaders/_normals.frag is missing (prefer the file — hot-reloaded).
_NORMAL_FRAG_FALLBACK = """
#version 330 core
uniform sampler2D uDepthF;
uniform vec2 uResolution;
uniform float uStrength;
in vec2 vUV;
out vec4 fragColor;
float sampleD(vec2 uv) {
    return textureLod(uDepthF, clamp(uv, vec2(0.0), vec2(1.0)), 0.0).r;
}
float tap(vec2 uv, float zc, float thr) {
    float z = sampleD(uv);
    return abs(z - zc) > thr ? zc : z;
}
void main() {
    vec2 t = 1.0 / max(vec2(textureSize(uDepthF, 0)), vec2(1.0));
    float zc = sampleD(vUV);
    float zx1 = tap(vUV + vec2(t.x, 0.0), zc, 0.05);
    float zx0 = tap(vUV - vec2(t.x, 0.0), zc, 0.05);
    float zy1 = tap(vUV + vec2(0.0, t.y), zc, 0.05);
    float zy0 = tap(vUV - vec2(0.0, t.y), zc, 0.05);
    float dx = zx1 - zx0;
    float dy = zy1 - zy0;
    float dx2 = tap(vUV + vec2(2.0*t.x,0.0), zc, 0.08) - tap(vUV - vec2(2.0*t.x,0.0), zc, 0.08);
    float dy2 = tap(vUV + vec2(0.0,2.0*t.y), zc, 0.08) - tap(vUV - vec2(0.0,2.0*t.y), zc, 0.08);
    dx -= 0.5 * dx2;
    dy -= 0.5 * dy2;
    if (length(vec2(dx, dy)) < 2.5e-4) {
        fragColor = vec4(0.5, 0.5, 1.0, 1.0);
        return;
    }
    float s = 56.0 * max(uStrength, 0.5);
    vec3 n = normalize(vec3(-dx * s, dy * s, 1.0));
    n.z = max(n.z, 0.05);
    n = normalize(n);
    fragColor = vec4(clamp(n * 0.5 + 0.5, 0.0, 1.0), 1.0);
}
"""

_DEFAULT_PREAMBLE = """
#version 330 core
uniform sampler2D uScene;
uniform sampler2D uDepth;
uniform sampler2D uDepthF;
uniform sampler2D uNormal;
uniform sampler2D uMain;
uniform sampler2D uHistory;
uniform sampler2D uPrevDepthF;
uniform float uHistoryValid;
uniform float uTime;
uniform vec2 uResolution;
in vec2 vUV;
out vec4 fragColor;
"""

# Built-in temporal resolve: blend the current composited frame with the
# reprojected history. Depth rejection discards history where the scene moved;
# a 3x3 neighborhood color clamp caps ghosting even without motion vectors.
_TEMPORAL_FRAG = """
#version 330 core
uniform sampler2D uCurrent;
uniform sampler2D uHistory;
uniform sampler2D uDepthF;
uniform sampler2D uPrevDepthF;
uniform float uAlpha;
uniform float uHistoryValid;
uniform float uDepthReject;
uniform vec2 uResolution;
in vec2 vUV;
out vec4 fragColor;
void main() {
    vec3 cur = texture(uCurrent, vUV).rgb;
    if (uHistoryValid < 0.5) { fragColor = vec4(cur, 1.0); return; }
    vec2 t = 1.0 / uResolution;
    vec3 mn = cur, mx = cur;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec3 s = texture(uCurrent, vUV + vec2(float(x), float(y)) * t).rgb;
            mn = min(mn, s); mx = max(mx, s);
        }
    }
    vec3 hist = clamp(texture(uHistory, vUV).rgb, mn, mx);
    float dc = texture(uDepthF, vUV).r;
    float dp = texture(uPrevDepthF, vUV).r;
    float reject = step(uDepthReject, abs(dc - dp));
    float a = uAlpha * (1.0 - reject);
    fragColor = vec4(mix(cur, hist, a), 1.0);
}
"""


class ShaderWorker(threading.Thread):
    """OpenGL fullscreen passes over (scene, depth) → display packet."""

    def __init__(
        self,
        config: RoluxConfig,
        in_slot: list,
        in_lock: threading.Lock,
        out_slot: list,
        out_lock: threading.Lock,
        stop_event: threading.Event,
        depth_ready: threading.Event,
        frame_ready: threading.Event,
        shaders_dir: Optional[Path] = None,
        status: Optional[dict] = None,
    ) -> None:
        super().__init__(name="RoluxShaders", daemon=True)
        self.cfg = config
        self.in_slot = in_slot
        self.in_lock = in_lock
        self.out_slot = out_slot
        self.out_lock = out_lock
        self.stop_event = stop_event
        self.depth_ready = depth_ready
        self.frame_ready = frame_ready
        self.shaders_dir = (shaders_dir or Path("shaders")).resolve()
        self.status = status if status is not None else {}
        self.shader_max_dim = max(256, int(getattr(config, "shader_max_dim", 960)))

        self._last_ts: Optional[float] = None
        self._t0 = time.perf_counter()
        self._shader_sig: tuple = ()
        self._gl_ok = False

        self._window = None
        self._programs: list = []
        self._normal_prog = None
        self._fbo_a = None
        self._fbo_b = None
        self._fbo_n = None
        self._tex_scene = None
        self._tex_depth = None
        self._tex_depth_f = None
        self._tex_normal = None
        self._tex_a = None
        self._tex_b = None
        self._vao = None
        self._w = 0
        self._h = 0
        self._df_w = 0
        self._df_h = 0
        self._readback = None
        self._readback_flip: Optional[np.ndarray] = None
        self._scene_flip: Optional[np.ndarray] = None
        self._pbo: list = []
        self._pbo_idx = 0
        self._pbo_ready = False
        self._want_u8_depth = False
        self._reload_tick = 0

        # Temporal accumulation (history buffer).
        self._temporal_prog = None
        self._tex_hist = [None, None]   # ping-pong history (prev resolved output)
        self._fbo_hist = [None, None]
        self._hist_cur = 0
        self._hist_valid = False
        self._tex_prev_df = None        # previous frame depth (R32F, df res)
        self._prev_df: Optional[np.ndarray] = None
        self.hist_on = bool(getattr(config, "temporal_accumulation", True))
        self.hist_alpha = float(getattr(config, "temporal_alpha", 0.85))
        self.hist_depth_reject = float(getattr(config, "temporal_depth_reject", 0.02))
        self.depth_filter_on = bool(getattr(config, "depth_temporal_filter", True))
        self._save_normals = threading.Event()
        self._save_overlay = threading.Event()
        self._captures_dir = Path("captures")
        self._normal_strength = 1.0
        self._normals_path = self.shaders_dir / "_normals.frag"
        self._normals_mtime: int = -1

        # Edge-aware temporal smoothing of depth_f, applied before it ever
        # reaches the GPU. Denoises static/flat surfaces frame-to-frame
        # without smearing real edges or moving objects.
        self._depth_prev: Optional[np.ndarray] = None
        self.temporal_alpha = 0.6        # max weight given to history on a static pixel
        self.temporal_edge_thr = 0.02     # depth delta above which we treat it as real motion, not noise
        self.temporal_max_drift = 0.02    # hard cap: history can never sit further than this from the true current sample

    def run(self) -> None:
        try:
            self._init_gl()
            self._gl_ok = True
            print(f"[Rolux] shader GL ready | watching {self.shaders_dir}")
        except Exception as exc:
            self._gl_ok = False
            print(f"[Rolux] shader GL unavailable ({exc}) — passthrough depth")

        self.shaders_dir.mkdir(parents=True, exist_ok=True)

        _acc = {"n": 0, "wait": 0.0, "proc": 0.0}
        _acc_t = time.perf_counter()
        _prev_end = time.perf_counter()

        while not self.stop_event.is_set():
            self.depth_ready.wait(timeout=0.05)
            if self.stop_event.is_set():
                break
            self.depth_ready.clear()

            with self.in_lock:
                packet: Optional[DepthPacket] = self.in_slot[0]
            if packet is None:
                continue
            ts = packet.capture_ts
            if ts == self._last_ts:
                continue
            self._last_ts = ts

            _t0 = time.perf_counter()
            try:
                out_rgb = self._process(packet)
            except Exception as exc:
                print(f"[Rolux] shader error: {exc}")
                out_rgb = self._depth_to_bgr(packet.rgb)
            _t1 = time.perf_counter()
            _acc["n"] += 1
            _acc["wait"] += (_t0 - _prev_end) * 1000.0
            _acc["proc"] += (_t1 - _t0) * 1000.0
            _prev_end = _t1
            if _t1 - _acc_t >= 1.0 and _acc["n"] > 0:
                n = _acc["n"]
                res = getattr(self, "_pt_res", (0, 0))
                print(
                    f"[perf/shader] {n}/s @ {res[0]}x{res[1]} | wait={_acc['wait']/n:5.1f}ms "
                    f"process={_acc['proc']/n:5.1f}ms | "
                    f"reload={getattr(self,'_pt_reload',0.0):4.1f} "
                    f"upload={getattr(self,'_pt_upload',0.0):4.1f} "
                    f"gpu_issue={getattr(self,'_pt_issue',0.0):4.1f} "
                    f"readback={getattr(self,'_pt_readback',0.0):5.1f}"
                )
                _acc = {"n": 0, "wait": 0.0, "proc": 0.0}
                _acc_t = _t1

            with self.out_lock:
                self.out_slot[0] = DepthPacket(
                    rgb=out_rgb,
                    rect=packet.rect,
                    capture_ts=packet.capture_ts,
                    infer_ms=packet.infer_ms,
                    main_bgr=packet.main_bgr,
                    normal_bgr=None,
                    depth_f=packet.depth_f,
                )
            self.frame_ready.set()

        self._shutdown_gl()

    def request_save_normals(self) -> None:
        """Dump the next frame's uNormal buffer to captures/normals_*.png."""
        self._save_normals.set()

    def request_save_overlay(self) -> None:
        """Dump the next composited overlay frame to captures/overlay_*.png."""
        self._save_overlay.set()

    def _depth_to_bgr(self, depth: np.ndarray) -> np.ndarray:
        if depth.ndim == 2:
            return np.ascontiguousarray(np.stack([depth, depth, depth], axis=-1))
        if depth.ndim == 3 and depth.shape[2] == 1:
            d = depth[:, :, 0]
            return np.ascontiguousarray(np.stack([d, d, d], axis=-1))
        return np.ascontiguousarray(depth[:, :, :3])

    def _process(self, packet: DepthPacket) -> np.ndarray:
        import cv2

        depth = packet.rgb
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = np.ascontiguousarray(depth, dtype=np.uint8)

        scene = packet.main_bgr
        if scene is None:
            scene = self._depth_to_bgr(depth)
        else:
            scene = np.ascontiguousarray(scene[:, :, :3], dtype=np.uint8)

        tw = max(64, int(packet.rect.width))
        th = max(64, int(packet.rect.height))
        max_dim = max(256, int(self.shader_max_dim))
        scale = min(1.0, float(max_dim) / float(max(tw, th)))
        tw = max(64, int(tw * scale))
        th = max(64, int(th * scale))
        if scene.shape[0] != th or scene.shape[1] != tw:
            scene = cv2.resize(scene, (tw, th), interpolation=cv2.INTER_LINEAR)
        if depth.shape[0] != th or depth.shape[1] != tw:
            depth = cv2.resize(depth, (tw, th), interpolation=cv2.INTER_LINEAR)

        if not self._gl_ok:
            return self._depth_to_bgr(depth)

        _tr = time.perf_counter()
        self._reload_if_needed()
        self._pt_reload = (time.perf_counter() - _tr) * 1000.0
        if not self._programs:
            self.status["shaders"] = []
            return self._depth_to_bgr(depth)

        depth_f = packet.depth_f
        if depth_f is None:
            depth_f = depth.astype(np.float32) * (1.0 / 255.0)
        else:
            depth_f = np.ascontiguousarray(depth_f, dtype=np.float32)

        depth_f = self._temporal_filter_depth(depth_f)

        return self._run_chain(scene, depth, depth_f)

    def _temporal_filter_depth(self, depth_f: np.ndarray) -> np.ndarray:
        """Blend depth_f with the previous frame, but only where the depth
        hasn't really changed. This kills the per-frame network noise that
        shows up as speckle/mottling in the normal buffer, while leaving
        real edges and moving objects untouched (no lasting ghost trails).
        """
        if not self.depth_filter_on:
            self._depth_prev = None
            return depth_f
        prev = self._depth_prev
        if prev is None or prev.shape != depth_f.shape:
            self._depth_prev = depth_f.copy()
            return depth_f

        diff = np.abs(depth_f - prev)

        # Soft falloff rather than a hard on/off cutoff: partial motion gets
        # partial blending instead of snapping fully one way or the other.
        w = np.clip(1.0 - diff / self.temporal_edge_thr, 0.0, 1.0)
        alpha = self.temporal_alpha * w
        blended = alpha * prev + (1.0 - alpha) * depth_f

        # Hard drift clamp: this is what actually kills ghost trails. Without
        # it, a slow-moving edge can have diff-just-under-threshold for many
        # frames in a row, and each blend nudges the result a little further
        # behind the real depth than the last — the history never catches up,
        # and you see a trailing smear. Clamping means no matter how many
        # consecutive frames get blended, the result can never sit further
        # than temporal_max_drift away from the true current sample.
        drift = self.temporal_max_drift
        blended = np.clip(blended, depth_f - drift, depth_f + drift)
        blended = blended.astype(np.float32, copy=False)

        self._depth_prev = blended
        return blended

    def _init_gl(self) -> None:
        import glfw
        from OpenGL import GL

        if not glfw.init():
            raise RuntimeError("glfw.init failed")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        win = glfw.create_window(16, 16, "RoLux GL", None, None)
        if not win:
            glfw.terminate()
            raise RuntimeError("glfw.create_window failed")
        glfw.make_context_current(win)
        self._window = win
        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)
        self._load_normal_program(force=True)
        print(f"[Rolux] normal shader: {self._normals_path}")
        try:
            self._temporal_prog = self._link_program(_TEMPORAL_FRAG)
            print(f"[Rolux] temporal resolve ready (accumulation={'on' if self.hist_on else 'off'})")
        except Exception as exc:
            self._temporal_prog = None
            print(f"[Rolux] temporal resolve compile failed ({exc}) — accumulation disabled")

    def _shutdown_gl(self) -> None:
        if not self._gl_ok:
            return
        try:
            from OpenGL import GL
            import glfw

            self._delete_targets()
            for p in self._programs:
                GL.glDeleteProgram(p)
            self._programs.clear()
            if self._normal_prog:
                GL.glDeleteProgram(self._normal_prog)
                self._normal_prog = None
            if self._temporal_prog:
                GL.glDeleteProgram(self._temporal_prog)
                self._temporal_prog = None
            if self._vao:
                GL.glDeleteVertexArrays(1, [self._vao])
            if self._window is not None:
                glfw.destroy_window(self._window)
            glfw.terminate()
        except Exception:
            pass
        self._gl_ok = False

    def _shader_files(self) -> list[Path]:
        if not self.shaders_dir.is_dir():
            return []
        return sorted(self.shaders_dir.glob("*.glsl"))

    def _load_normal_program(self, force: bool = False) -> None:
        from OpenGL import GL

        path = self._normals_path
        try:
            mtime = path.stat().st_mtime_ns if path.is_file() else -1
        except OSError:
            mtime = -1
        if not force and mtime == self._normals_mtime and self._normal_prog:
            return

        if path.is_file():
            src = path.read_text(encoding="utf-8")
            src_tag = path.name
        else:
            src = _NORMAL_FRAG_FALLBACK
            src_tag = "fallback"
            print(f"[Rolux] missing {path.name} — using embedded fallback")

        try:
            prog = self._link_program(src)
        except Exception as exc:
            print(f"[Rolux] normal shader compile failed ({src_tag}): {exc}")
            if self._normal_prog is None:
                prog = self._link_program(_NORMAL_FRAG_FALLBACK)
            else:
                return
        if self._normal_prog:
            try:
                GL.glDeleteProgram(self._normal_prog)
            except Exception:
                pass
        self._normal_prog = prog
        self._normals_mtime = mtime
        print(f"[Rolux] normal program loaded ({src_tag})")

    def _reload_if_needed(self) -> None:
        # Hot-reload built-in normals + user *.glsl (stat disk every ~15 frames).
        self._reload_tick += 1
        if self._reload_tick == 1 or self._reload_tick % 15 == 0:
            self._load_normal_program(force=False)
            files = self._shader_files()
            sig = tuple((p.name, p.stat().st_mtime_ns) for p in files)
            if sig != self._shader_sig:
                self._shader_sig = sig
                self._compile_all(files)

    def _compile_shader(self, src: str, stage: int) -> int:
        from OpenGL import GL

        sid = GL.glCreateShader(stage)
        GL.glShaderSource(sid, src)
        GL.glCompileShader(sid)
        if not GL.glGetShaderiv(sid, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(sid).decode("utf-8", errors="replace")
            GL.glDeleteShader(sid)
            raise RuntimeError(log)
        return sid

    def _link_program(self, frag_src: str) -> int:
        from OpenGL import GL

        text = frag_src.strip()
        if not text.startswith("#version"):
            if "void main" in text:
                text = _DEFAULT_PREAMBLE + "\n" + text
            else:
                text = _DEFAULT_PREAMBLE + "\nvoid main() {\n" + text + "\n}\n"

        vs = self._compile_shader(_VERT, GL.GL_VERTEX_SHADER)
        fs = self._compile_shader(text, GL.GL_FRAGMENT_SHADER)
        prog = GL.glCreateProgram()
        GL.glAttachShader(prog, vs)
        GL.glAttachShader(prog, fs)
        GL.glLinkProgram(prog)
        GL.glDeleteShader(vs)
        GL.glDeleteShader(fs)
        if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(prog).decode("utf-8", errors="replace")
            GL.glDeleteProgram(prog)
            raise RuntimeError(log)
        return prog

    def _compile_all(self, files: list[Path]) -> None:
        from OpenGL import GL

        for p in self._programs:
            try:
                GL.glDeleteProgram(p)
            except Exception:
                pass
        self._programs = []
        names: list[str] = []
        for path in files:
            try:
                src = path.read_text(encoding="utf-8")
                prog = self._link_program(src)
                self._programs.append(prog)
                names.append(path.name)
                print(f"[Rolux] shader loaded: {path.name}")
            except Exception as exc:
                print(f"[Rolux] shader compile failed ({path.name}): {exc}")
        self.status["shaders"] = names
        self._want_u8_depth = "00_depth_view.glsl" in names
        if not names:
            print("[Rolux] no active shaders — showing raw depth")

    def _delete_targets(self) -> None:
        from OpenGL import GL

        for tex in (
            self._tex_scene,
            self._tex_depth,
            self._tex_depth_f,
            self._tex_normal,
            self._tex_a,
            self._tex_b,
            self._tex_hist[0],
            self._tex_hist[1],
        ):
            if tex:
                GL.glDeleteTextures(1, [int(tex)])
        for fbo in (self._fbo_a, self._fbo_b, self._fbo_n, self._fbo_hist[0], self._fbo_hist[1]):
            if fbo:
                GL.glDeleteFramebuffers(1, [int(fbo)])
        if self._pbo:
            GL.glDeleteBuffers(len(self._pbo), self._pbo)
        self._tex_scene = self._tex_depth = self._tex_depth_f = None
        self._tex_normal = self._tex_a = self._tex_b = None
        self._fbo_a = self._fbo_b = self._fbo_n = None
        self._tex_hist = [None, None]
        self._fbo_hist = [None, None]
        self._pbo = []
        self._pbo_ready = False
        self._hist_valid = False
        self._w = self._h = 0
        self._df_w = self._df_h = 0
        self._readback_flip = None
        self._scene_flip = None

    def _make_tex(self) -> int:
        from OpenGL import GL

        tex = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        return tex

    def _ensure_targets(self, w: int, h: int) -> None:
        from OpenGL import GL

        if w == self._w and h == self._h and self._fbo_a is not None:
            return
        self._delete_targets()
        self._w, self._h = w, h
        self._tex_scene = self._make_tex()
        self._tex_depth = self._make_tex()
        self._tex_a = self._make_tex()
        self._tex_b = self._make_tex()

        for tex in (self._tex_scene, self._tex_depth, self._tex_a, self._tex_b):
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,
                GL.GL_RGB8,
                w,
                h,
                0,
                GL.GL_RGB,
                GL.GL_UNSIGNED_BYTE,
                None,
            )

        self._fbo_a = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_a)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._tex_a, 0
        )

        self._fbo_b = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_b)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._tex_b, 0
        )

        # Ping-pong history targets for temporal accumulation.
        for i in range(2):
            tex = self._make_tex()
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None
            )
            fbo = int(GL.glGenFramebuffers(1))
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
            GL.glFramebufferTexture2D(
                GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0
            )
            self._tex_hist[i] = tex
            self._fbo_hist[i] = fbo
        self._hist_cur = 0
        self._hist_valid = False  # first frame after a resize has no valid history

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self._readback = np.empty((h, w, 3), dtype=np.uint8)
        self._readback_flip = np.empty((h, w, 3), dtype=np.uint8)
        self._scene_flip = np.empty((h, w, 3), dtype=np.uint8)
        if self._pbo:
            GL.glDeleteBuffers(len(self._pbo), self._pbo)
        self._pbo = [int(x) for x in GL.glGenBuffers(2)]
        pack_bytes = w * h * 3
        for pbo in self._pbo:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo)
            GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, pack_bytes, None, GL.GL_STREAM_READ)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._pbo_idx = 0
        self._pbo_ready = False

    def _upload(self, tex: int, rgb: np.ndarray) -> None:
        from OpenGL import GL

        rgb = np.ascontiguousarray(np.flipud(rgb))
        h, w = rgb.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb
        )

    def _upload_raw(self, tex: int, img: np.ndarray, fmt) -> None:
        """Upload an already-oriented (flipped) 3-channel buffer, no CPU copy."""
        from OpenGL import GL

        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, GL.GL_UNSIGNED_BYTE, img)

    def _upload_gray(self, tex: int, gray_flipped: np.ndarray) -> None:
        """Expand a single-channel (already flipped) depth to RGB and upload."""
        import cv2
        from OpenGL import GL

        rgb = cv2.cvtColor(gray_flipped, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb)

    def _ensure_depth_f(self, w: int, h: int) -> None:
        from OpenGL import GL

        if self._tex_depth_f and self._df_w == w and self._df_h == h and self._fbo_n:
            return
        if self._tex_depth_f:
            GL.glDeleteTextures(1, [int(self._tex_depth_f)])
        if self._tex_normal:
            GL.glDeleteTextures(1, [int(self._tex_normal)])
        if self._tex_prev_df:
            GL.glDeleteTextures(1, [int(self._tex_prev_df)])
        if self._fbo_n:
            GL.glDeleteFramebuffers(1, [int(self._fbo_n)])

        self._tex_depth_f = self._make_tex()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_depth_f)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_R32F, w, h, 0, GL.GL_RED, GL.GL_FLOAT, None
        )

        # Normals rendered at native depth size (not upscaled shader res).
        self._tex_normal = self._make_tex()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_normal)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, w, h, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None
        )
        self._fbo_n = int(GL.glGenFramebuffers(1))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_n)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self._tex_normal, 0
        )
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        # Previous-frame depth (for temporal reprojection/rejection).
        self._tex_prev_df = self._make_tex()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_prev_df)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_R32F, w, h, 0, GL.GL_RED, GL.GL_FLOAT, None
        )
        self._prev_df = None

        self._df_w, self._df_h = w, h
        self._readback_n = np.empty((h, w, 3), dtype=np.uint8)
        print(f"[Rolux] normals @ {w}x{h} (native depth)")

    def _upload_depth_f(self, depth_f: np.ndarray) -> None:
        from OpenGL import GL

        d = np.ascontiguousarray(np.flipud(depth_f.astype(np.float32, copy=False)))
        h, w = d.shape[:2]
        self._ensure_depth_f(w, h)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_depth_f)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RED, GL.GL_FLOAT, d
        )

    def _save_fbo_png(self, fbo: int, w: int, h: int, prefix: str) -> Path:
        from datetime import datetime

        import cv2
        from OpenGL import GL

        self._captures_dir.mkdir(parents=True, exist_ok=True)
        path = self._captures_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, self._readback)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        bgr = np.ascontiguousarray(np.flipud(self._readback)[:, :, ::-1])
        cv2.imwrite(str(path), bgr)
        print(f"[Rolux] saved {path}")
        self.status["last_capture"] = str(path)
        return path

    def _save_normals_png(self) -> Path:
        from datetime import datetime

        import cv2
        from OpenGL import GL

        w, h = self._df_w, self._df_h
        buf = getattr(self, "_readback_n", None)
        if buf is None or buf.shape[0] != h or buf.shape[1] != w:
            buf = np.empty((h, w, 3), dtype=np.uint8)
            self._readback_n = buf
        self._captures_dir.mkdir(parents=True, exist_ok=True)
        path = self._captures_dir / f"normals_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_n)
        GL.glReadPixels(0, 0, w, h, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, buf)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        bgr = np.ascontiguousarray(np.flipud(buf)[:, :, ::-1])
        cv2.imwrite(str(path), bgr)
        print(f"[Rolux] saved {path} ({w}x{h})")
        self.status["last_capture"] = str(path)
        return path

    def _generate_normals(self, w: int, h: int) -> None:
        from OpenGL import GL

        assert self._normal_prog is not None
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_n)
        GL.glViewport(0, 0, w, h)
        GL.glUseProgram(self._normal_prog)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_depth_f)
        loc = GL.glGetUniformLocation(self._normal_prog, "uDepthF")
        if loc >= 0:
            GL.glUniform1i(loc, 0)
        loc = GL.glGetUniformLocation(self._normal_prog, "uResolution")
        if loc >= 0:
            GL.glUniform2f(loc, float(w), float(h))
        loc = GL.glGetUniformLocation(self._normal_prog, "uStrength")
        if loc >= 0:
            GL.glUniform1f(loc, float(self._normal_strength))

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

    def _run_chain(
        self, scene_bgr: np.ndarray, depth: np.ndarray, depth_f: np.ndarray
    ) -> np.ndarray:
        from OpenGL import GL
        import glfw

        glfw.make_context_current(self._window)
        h, w = depth.shape[:2]
        self._ensure_targets(w, h)

        _tu = time.perf_counter()
        import cv2

        scene_bgr = np.ascontiguousarray(scene_bgr)
        cv2.flip(scene_bgr, 0, dst=self._scene_flip)
        self._upload_raw(self._tex_scene, self._scene_flip, GL.GL_BGR)
        if self._want_u8_depth:
            self._upload_raw(self._tex_depth, cv2.flip(depth, 0), GL.GL_RED)
        self._upload_depth_f(depth_f)
        self._pt_upload = (time.perf_counter() - _tu) * 1000.0

        _tg = time.perf_counter()
        self._generate_normals(self._df_w, self._df_h)
        if self._save_normals.is_set():
            self._save_normals.clear()
            try:
                self._save_normals_png()
            except Exception as exc:
                print(f"[Rolux] save normals failed: {exc}")

        src_tex = self._tex_scene
        dst_fbo = self._fbo_b
        t = float(time.perf_counter() - self._t0)

        # Temporal history indices: read last frame's resolved output, write this one.
        hist_read = self._hist_cur
        hist_write = 1 - self._hist_cur
        hist_valid = 1.0 if (self.hist_on and self._hist_valid) else 0.0

        GL.glViewport(0, 0, w, h)
        GL.glBindVertexArray(self._vao)

        for prog in self._programs:
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dst_fbo)
            GL.glUseProgram(prog)

            def _loc(name: str) -> int:
                return GL.glGetUniformLocation(prog, name)

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_scene)
            loc = _loc("uScene")
            if loc >= 0:
                GL.glUniform1i(loc, 0)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_depth)
            loc = _loc("uDepth")
            if loc >= 0:
                GL.glUniform1i(loc, 1)

            GL.glActiveTexture(GL.GL_TEXTURE2)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_normal)
            loc = _loc("uNormal")
            if loc >= 0:
                GL.glUniform1i(loc, 2)

            GL.glActiveTexture(GL.GL_TEXTURE3)
            GL.glBindTexture(GL.GL_TEXTURE_2D, src_tex)
            loc = _loc("uMain")
            if loc >= 0:
                GL.glUniform1i(loc, 3)

            GL.glActiveTexture(GL.GL_TEXTURE4)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_depth_f)
            loc = _loc("uDepthF")
            if loc >= 0:
                GL.glUniform1i(loc, 4)

            GL.glActiveTexture(GL.GL_TEXTURE5)
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._tex_hist[hist_read]))
            loc = _loc("uHistory")
            if loc >= 0:
                GL.glUniform1i(loc, 5)

            GL.glActiveTexture(GL.GL_TEXTURE6)
            GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._tex_prev_df))
            loc = _loc("uPrevDepthF")
            if loc >= 0:
                GL.glUniform1i(loc, 6)

            loc = _loc("uHistoryValid")
            if loc >= 0:
                GL.glUniform1f(loc, hist_valid)

            loc = _loc("uTime")
            if loc >= 0:
                GL.glUniform1f(loc, t)
            loc = _loc("uResolution")
            if loc >= 0:
                GL.glUniform2f(loc, float(w), float(h))

            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

            if dst_fbo == self._fbo_b:
                src_tex = self._tex_b
                dst_fbo = self._fbo_a
            else:
                src_tex = self._tex_a
                dst_fbo = self._fbo_b

        result_tex = src_tex
        read_fbo = self._fbo_a if result_tex == self._tex_a else self._fbo_b

        # Temporal resolve: blend the composited frame with reprojected history.
        if self.hist_on and self._temporal_prog and self._tex_hist[hist_write]:
            self._temporal_resolve(result_tex, hist_read, hist_write, hist_valid, w, h)
            read_fbo = self._fbo_hist[hist_write]
            result_tex = self._tex_hist[hist_write]
            self._hist_cur = hist_write
            self._hist_valid = True
            # Stash this frame's depth as "previous" for next frame's rejection.
            try:
                self._upload_to_tex(self._tex_prev_df, depth_f)
            except Exception:
                pass

        if self._save_overlay.is_set():
            self._save_overlay.clear()
            try:
                self._save_fbo_png(read_fbo, w, h, "overlay")
            except Exception as exc:
                print(f"[Rolux] save overlay failed: {exc}")

        self._pt_issue = (time.perf_counter() - _tg) * 1000.0

        _tb = time.perf_counter()
        out = self._pbo_readback(read_fbo, w, h)
        self._pt_readback = (time.perf_counter() - _tb) * 1000.0
        self._pt_res = (w, h)
        return out

    def _pbo_readback(self, read_fbo: int, w: int, h: int) -> np.ndarray:
        """Async pack-buffer readback; returns top-down BGR for the overlay."""
        import ctypes

        import cv2
        from OpenGL import GL

        assert self._readback is not None and self._readback_flip is not None
        pack = w * h * 3
        write_idx = self._pbo_idx
        read_idx = 1 - write_idx

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, read_fbo)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbo[write_idx])
        GL.glReadPixels(0, 0, w, h, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)

        if self._pbo_ready:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, self._pbo[read_idx])
            ptr = GL.glMapBuffer(GL.GL_PIXEL_PACK_BUFFER, GL.GL_READ_ONLY)
            if ptr:
                src = (ctypes.c_ubyte * pack).from_address(int(ptr))
                np.copyto(
                    self._readback,
                    np.frombuffer(src, dtype=np.uint8).reshape(h, w, 3),
                )
                GL.glUnmapBuffer(GL.GL_PIXEL_PACK_BUFFER)
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        else:
            # First frame: no prior PBO — sync read once, then pipeline async.
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, read_fbo)
            GL.glReadPixels(0, 0, w, h, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, self._readback)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            self._pbo_ready = True

        self._pbo_idx = read_idx
        cv2.flip(self._readback, 0, dst=self._readback_flip)
        return self._readback_flip

    def _upload_to_tex(self, tex: int, depth_f: np.ndarray) -> None:
        """Upload an R32F depth map into an existing texture (GL bottom-left origin)."""
        from OpenGL import GL

        d = np.ascontiguousarray(np.flipud(depth_f.astype(np.float32, copy=False)))
        h, w = d.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(tex))
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_RED, GL.GL_FLOAT, d)

    def _temporal_resolve(
        self, cur_tex, hist_read: int, hist_write: int, valid: float, w: int, h: int
    ) -> None:
        from OpenGL import GL

        prog = self._temporal_prog
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._fbo_hist[hist_write])
        GL.glViewport(0, 0, w, h)
        GL.glUseProgram(prog)

        def _loc(name: str) -> int:
            return GL.glGetUniformLocation(prog, name)

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(cur_tex))
        loc = _loc("uCurrent")
        if loc >= 0:
            GL.glUniform1i(loc, 0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._tex_hist[hist_read]))
        loc = _loc("uHistory")
        if loc >= 0:
            GL.glUniform1i(loc, 1)

        GL.glActiveTexture(GL.GL_TEXTURE2)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._tex_depth_f))
        loc = _loc("uDepthF")
        if loc >= 0:
            GL.glUniform1i(loc, 2)

        GL.glActiveTexture(GL.GL_TEXTURE3)
        GL.glBindTexture(GL.GL_TEXTURE_2D, int(self._tex_prev_df))
        loc = _loc("uPrevDepthF")
        if loc >= 0:
            GL.glUniform1i(loc, 3)

        loc = _loc("uAlpha")
        if loc >= 0:
            GL.glUniform1f(loc, float(self.hist_alpha))
        loc = _loc("uHistoryValid")
        if loc >= 0:
            GL.glUniform1f(loc, float(valid))
        loc = _loc("uDepthReject")
        if loc >= 0:
            GL.glUniform1f(loc, float(self.hist_depth_reject))
        loc = _loc("uResolution")
        if loc >= 0:
            GL.glUniform2f(loc, float(w), float(h))

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
