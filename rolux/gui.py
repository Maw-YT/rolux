"""RoLux control panel — ReShade-style Tkinter GUI."""

from __future__ import annotations

import shutil
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk, simpledialog
from typing import Optional

from rolux.capture_worker import CaptureWorker
from rolux.config import RoluxConfig
from rolux.inference_worker import InferenceWorker
from rolux.overlay_ui import DepthOverlay
from rolux.shader_worker import ShaderWorker
from rolux import presets
from rolux.presets import ShaderParam, format_value, parse_params


# --- ReShade-ish palette ---
BG = "#1b1b1b"
PANEL = "#232323"
CARD = "#2b2b2b"
CARD_HI = "#343434"
FG = "#ececec"
MUTED = "#9a9a9a"
ACCENT = "#ff9d2f"       # ReShade orange
ACCENT_DIM = "#c9791f"
DANGER = "#f0707a"
OK = "#8bd96a"
BORDER = "#3a3a3a"
FONT = ("Segoe UI", 10)
FONT_SM = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI Semibold", 11)
FONT_TITLE = ("Segoe UI Semibold", 20)
FONT_MONO = ("Cascadia Mono", 9)


class RoluxApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RoLux")
        self.root.configure(bg=BG)
        self.root.geometry("500x780")
        self.root.minsize(460, 700)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Originals stay pristine; the live session works off shaders/temp,
        # which is (re)created on launch and deleted when RoLux closes.
        self.orig_shaders_dir = Path("shaders").resolve()
        self.shaders_dir = self.orig_shaders_dir / "temp"
        self.presets_dir = Path("presets").resolve()
        self._init_temp_shaders()

        self.stop_event: Optional[threading.Event] = None
        self.capture: Optional[CaptureWorker] = None
        self.infer: Optional[InferenceWorker] = None
        self.shaders: Optional[ShaderWorker] = None
        self.overlay: Optional[DepthOverlay] = None
        self.running = False

        self.capture_slot: list = [None]
        self.capture_lock = threading.Lock()
        self.raw_slot: list = [None]
        self.raw_lock = threading.Lock()
        self.depth_slot: list = [None]
        self.depth_lock = threading.Lock()
        self.status: dict = {"roblox_found": False, "focused": False}

        # Parameter editor state
        self._sel_path: Optional[Path] = None
        self._param_lines: list[str] = []
        self._params: list[ShaderParam] = []
        self._write_after: Optional[str] = None
        self._shader_rows: dict[str, tk.Widget] = {}
        self._shader_enabled: dict[str, bool] = {}
        self._tab_canvases: list = []
        self._active_canvas = None

        self._build_style()
        self._build_ui()
        self._refresh_shader_list()
        self._refresh_presets()
        self._pulse_status()

    # ---------------------------------------------------------------- style
    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(".", background=BG, foreground=FG, fieldbackground=CARD, borderwidth=0)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=FG, font=FONT)
        style.configure("Muted.TLabel", background=PANEL, foreground=MUTED, font=FONT_SM)
        style.configure("MutedBG.TLabel", background=BG, foreground=MUTED, font=FONT_SM)
        style.configure("Title.TLabel", background=BG, foreground=FG, font=FONT_TITLE)
        style.configure("Accent.TLabel", background=BG, foreground=ACCENT, font=FONT_SM)
        style.configure("Stat.TLabel", background=PANEL, foreground=ACCENT, font=FONT_MONO)

        style.configure(
            "TNotebook", background=BG, borderwidth=0, tabmargins=(0, 4, 0, 0)
        )
        style.configure(
            "TNotebook.Tab",
            background=CARD,
            foreground=MUTED,
            padding=(16, 8),
            font=FONT_BOLD,
            borderwidth=0,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", PANEL)],
            foreground=[("selected", ACCENT)],
        )

        style.configure(
            "Accent.TButton",
            background=ACCENT_DIM,
            foreground="#1b1b1b",
            font=FONT_BOLD,
            padding=(14, 8),
            borderwidth=0,
        )
        style.map("Accent.TButton", background=[("active", ACCENT)])
        style.configure(
            "Ghost.TButton", background=CARD, foreground=FG, font=FONT, padding=(10, 6),
            borderwidth=0,
        )
        style.map("Ghost.TButton", background=[("active", CARD_HI)])
        style.configure(
            "Horizontal.TScale", background=PANEL, troughcolor=CARD,
            bordercolor=PANEL, lightcolor=ACCENT, darkcolor=ACCENT,
        )
        style.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=FG)

    def _card(self, parent: tk.Widget) -> ttk.Frame:
        return ttk.Frame(parent, style="Card.TFrame", padding=12)

    # ------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=(18, 14, 18, 6))
        header.pack(fill="x")
        ttk.Label(header, text="RoLux", style="Title.TLabel").pack(side="left")
        ttk.Label(
            header, text="ReShade for Roblox · DA-V2 depth",
            style="Accent.TLabel",
        ).pack(side="left", padx=(10, 0), pady=(10, 0))

        # persistent status strip
        strip = self._card(self.root)
        strip.pack(fill="x", padx=18, pady=(0, 6))
        self.lbl_state = tk.Label(
            strip, text="Idle", bg=PANEL, fg=MUTED, font=FONT_BOLD, anchor="w"
        )
        self.lbl_state.pack(side="left")
        self.lbl_badges = tk.Label(
            strip, text="Roblox — · Focus —", bg=PANEL, fg=MUTED, font=FONT_MONO
        )
        self.lbl_badges.pack(side="right")

        # actions
        actions = ttk.Frame(self.root, padding=(18, 0, 18, 6))
        actions.pack(fill="x")
        self.btn_start = ttk.Button(actions, text="▶  Start", style="Accent.TButton", command=self.start)
        self.btn_stop = ttk.Button(actions, text="■  Stop", style="Ghost.TButton", command=self.stop)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.btn_stop.pack(side="left", fill="x", expand=True)
        self.btn_stop.state(["disabled"])

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=14, pady=(6, 14))
        self.nb = nb
        self.tab_home = ttk.Frame(nb, style="TFrame")
        self.tab_settings = ttk.Frame(nb, style="TFrame")
        self.tab_stats = ttk.Frame(nb, style="TFrame")
        nb.add(self.tab_home, text="Home")
        nb.add(self.tab_settings, text="Settings")
        nb.add(self.tab_stats, text="Statistics")

        self._build_home(self._make_scrollable(self.tab_home))
        self._build_settings(self._make_scrollable(self.tab_settings))
        self._build_stats(self._make_scrollable(self.tab_stats))

        nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self._on_tab_changed()

        self.var_opacity.trace_add("write", self._on_opacity_change)

    def _make_scrollable(self, tab: ttk.Frame) -> ttk.Frame:
        """Wrap a notebook tab in a vertically scrollable canvas; return the
        inner frame to populate. Wheel is routed by _on_mousewheel."""
        canvas = tk.Canvas(tab, bg=BG, highlightthickness=0)
        vsb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, style="TFrame", padding=8)
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        self._tab_canvases.append(canvas)
        return inner

    def _on_tab_changed(self, _event=None) -> None:
        try:
            idx = self.nb.index(self.nb.select())
        except Exception:
            idx = 0
        if 0 <= idx < len(self._tab_canvases):
            self._active_canvas = self._tab_canvases[idx]

    def _on_mousewheel(self, event) -> None:
        c = self._active_canvas
        if c is None:
            return
        first, last = c.yview()
        if first <= 0.0 and last >= 1.0:
            return  # nothing to scroll
        c.yview_scroll(int(-event.delta / 120), "units")

    # -------- Home: techniques list + variable editor (ReShade style) -----
    def _build_home(self, parent: ttk.Frame) -> None:
        pre = self._card(parent)
        pre.pack(fill="x", pady=(0, 8))
        prow = ttk.Frame(pre, style="Card.TFrame")
        prow.pack(fill="x")
        ttk.Label(prow, text="PRESET", style="Muted.TLabel").pack(side="left")
        self.var_preset = tk.StringVar(value="")
        self.cmb_preset = ttk.Combobox(
            pre, textvariable=self.var_preset, state="readonly", font=FONT_SM
        )
        self.cmb_preset.pack(fill="x", pady=(6, 6))
        self.cmb_preset.bind("<<ComboboxSelected>>", lambda e: self._preset_load())
        pbtns = ttk.Frame(pre, style="Card.TFrame")
        pbtns.pack(fill="x")
        ttk.Button(pbtns, text="Load", style="Ghost.TButton", command=self._preset_load).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Button(pbtns, text="Save", style="Ghost.TButton", command=self._preset_save).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(pbtns, text="Save As", style="Ghost.TButton", command=self._preset_save_as).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(pbtns, text="Delete", style="Ghost.TButton", command=self._preset_delete).pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )

        techs = self._card(parent)
        techs.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(techs, style="Card.TFrame")
        row.pack(fill="x")
        ttk.Label(row, text="EFFECTS", style="Muted.TLabel").pack(side="left")
        ttk.Button(row, text="Folder", style="Ghost.TButton", command=self._open_folder).pack(side="right")
        ttk.Button(row, text="Reload", style="Ghost.TButton", command=self._refresh_shader_list).pack(
            side="right", padx=(0, 6)
        )
        ttk.Button(row, text="Reset", style="Ghost.TButton", command=self._reset_shaders).pack(
            side="right", padx=(0, 6)
        )
        self.tech_frame = tk.Frame(techs, bg=PANEL)
        self.tech_frame.pack(fill="x", pady=(8, 0))

        variables = self._card(parent)
        variables.pack(fill="both", expand=True)
        head = ttk.Frame(variables, style="Card.TFrame")
        head.pack(fill="x")
        ttk.Label(head, text="VARIABLES", style="Muted.TLabel").pack(side="left")
        self.lbl_sel = tk.Label(head, text="—", bg=PANEL, fg=ACCENT, font=FONT_MONO)
        self.lbl_sel.pack(side="right")

        # Parameter widgets live directly in the tab's scroll region.
        self.param_host = tk.Frame(variables, bg=PANEL)
        self.param_host.pack(fill="both", expand=True, pady=(8, 0))

    def _build_settings(self, parent: ttk.Frame) -> None:
        s = self._card(parent)
        s.pack(fill="x")

        self.var_title = tk.StringVar(value="Roblox")
        self.var_size = tk.IntVar(value=392)
        self.var_fps = tk.IntVar(value=144)
        self.var_opacity = tk.DoubleVar(value=1.0)
        self.var_focus = tk.BooleanVar(value=True)
        self.var_allow_shot = tk.BooleanVar(value=False)
        self.var_temporal = tk.BooleanVar(value=True)
        self.var_render = tk.IntVar(value=960)
        self.var_engine = tk.StringVar(value=str(Path("models/depth_anything_v2_vits_fp16.engine")))

        self._labeled_entry(s, "Window title contains", self.var_title)

        rowf = ttk.Frame(s, style="Card.TFrame")
        rowf.pack(fill="x", pady=4)
        ttk.Label(rowf, text="Network size", style="Muted.TLabel").pack(anchor="w")
        size_row = ttk.Frame(rowf, style="Card.TFrame")
        size_row.pack(fill="x")
        for val in (392, 518):
            tk.Radiobutton(
                size_row, text=str(val), variable=self.var_size, value=val,
                bg=PANEL, fg=FG, selectcolor=CARD, activebackground=PANEL,
                activeforeground=FG, font=FONT, highlightthickness=0,
            ).pack(side="left", padx=(0, 12))

        self._labeled_scale(s, "Target FPS", self.var_fps, 30, 144, is_int=True)
        self._labeled_scale(
            s, "Shader render scale", self.var_render, 640, 1280, is_int=True,
            command=self._on_render_scale_change,
        )
        self._labeled_scale(s, "Overlay opacity", self.var_opacity, 0.2, 1.0, is_int=False)

        tk.Checkbutton(
            s, text="Allow screen capture & recording",
            variable=self.var_allow_shot,
            command=self._on_allow_shot, bg=PANEL, fg=FG, selectcolor=CARD,
            activebackground=PANEL, activeforeground=FG, font=FONT,
            highlightthickness=0, anchor="w",
        ).pack(fill="x", pady=(8, 2))
        tk.Label(
            s,
            text=(
                "Include RoLux effects in Snipping Tool, OBS, ShadowPlay, and "
                "other recorders. Off by default so DXGI can read Roblox under the overlay."
            ),
            bg=PANEL, fg=MUTED, font=FONT_SM, justify="left", wraplength=420,
        ).pack(fill="x", pady=(0, 4))

        tk.Checkbutton(
            s, text="Only show while Roblox is focused", variable=self.var_focus,
            bg=PANEL, fg=FG, selectcolor=CARD, activebackground=PANEL,
            activeforeground=FG, font=FONT, highlightthickness=0, anchor="w",
        ).pack(fill="x", pady=(8, 4))
        tk.Checkbutton(
            s, text="Temporal denoising (accumulation + depth filters)",
            variable=self.var_temporal, command=self._on_temporal_change,
            bg=PANEL, fg=FG, selectcolor=CARD, activebackground=PANEL,
            activeforeground=FG, font=FONT, highlightthickness=0, anchor="w",
        ).pack(fill="x", pady=(0, 4))

        eng = ttk.Frame(s, style="Card.TFrame")
        eng.pack(fill="x", pady=(6, 0))
        ttk.Label(eng, text="TensorRT engine", style="Muted.TLabel").pack(anchor="w")
        eng_row = ttk.Frame(eng, style="Card.TFrame")
        eng_row.pack(fill="x", pady=(2, 0))
        tk.Entry(
            eng_row, textvariable=self.var_engine, bg=CARD, fg=FG,
            insertbackground=FG, relief="flat", font=FONT_MONO,
        ).pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 6))
        ttk.Button(eng_row, text="…", style="Ghost.TButton", width=3, command=self._browse_engine).pack(
            side="right"
        )

    def _build_stats(self, parent: ttk.Frame) -> None:
        s = self._card(parent)
        s.pack(fill="x")
        self.lbl_roblox = ttk.Label(s, text="Roblox: —", style="Stat.TLabel")
        self.lbl_focus = ttk.Label(s, text="Focus: —", style="Stat.TLabel")
        self.lbl_fps = ttk.Label(s, text="Overlay: — FPS", style="Stat.TLabel")
        self.lbl_infer = ttk.Label(s, text="Infer: — · e2e — ms", style="Stat.TLabel")
        self.lbl_cap = ttk.Label(s, text="Capture: — FPS", style="Stat.TLabel")
        self.lbl_shaders = ttk.Label(s, text="Active: —", style="Stat.TLabel")
        self.lbl_recording = ttk.Label(s, text="Recording: —", style="Stat.TLabel")
        for w in (
            self.lbl_roblox, self.lbl_focus, self.lbl_cap, self.lbl_fps,
            self.lbl_infer, self.lbl_shaders, self.lbl_recording,
        ):
            w.pack(anchor="w", pady=2)

        shots = self._card(parent)
        shots.pack(fill="x", pady=(8, 0))
        ttk.Label(shots, text="CAPTURE", style="Muted.TLabel").pack(anchor="w", pady=(0, 6))
        srow = ttk.Frame(shots, style="Card.TFrame")
        srow.pack(fill="x")
        self.btn_save_n = ttk.Button(srow, text="Save normals PNG", style="Ghost.TButton", command=self._save_normals)
        self.btn_save_o = ttk.Button(srow, text="Save overlay PNG", style="Ghost.TButton", command=self._save_overlay)
        self.btn_save_n.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.btn_save_o.pack(side="left", fill="x", expand=True)
        self.btn_save_n.state(["disabled"])
        self.btn_save_o.state(["disabled"])

    # ------------------------------------------------------------- widgets
    def _labeled_entry(self, parent: tk.Widget, label: str, var: tk.Variable) -> None:
        ttk.Label(parent, text=label, style="Muted.TLabel").pack(anchor="w")
        tk.Entry(
            parent, textvariable=var, bg=CARD, fg=FG, insertbackground=FG,
            relief="flat", font=FONT,
        ).pack(fill="x", pady=(2, 8), ipady=5)

    def _labeled_scale(self, parent, label, var, frm, to, is_int, command=None) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=4)
        top = ttk.Frame(row, style="Card.TFrame")
        top.pack(fill="x")
        ttk.Label(top, text=label, style="Muted.TLabel").pack(side="left")
        val_lbl = ttk.Label(top, text="", style="Stat.TLabel")
        val_lbl.pack(side="right")

        def _update(_=None) -> None:
            v = var.get()
            val_lbl.configure(text=f"{int(v)}" if is_int else f"{float(v):.2f}")
            if command is not None:
                command()

        ttk.Scale(row, from_=frm, to=to, variable=var, command=lambda _: _update()).pack(
            fill="x", pady=(2, 0)
        )
        _update()

    # ------------------------------------------------------- shader list
    def _shader_files(self) -> list[Path]:
        if not self.shaders_dir.is_dir():
            return []
        found = list(self.shaders_dir.glob("*.glsl")) + list(self.shaders_dir.glob("*.glsl.off"))
        return sorted(found, key=lambda p: p.name)

    def _refresh_shader_list(self) -> None:
        for w in self.tech_frame.winfo_children():
            w.destroy()
        self._shader_rows.clear()
        self._shader_enabled.clear()
        files = self._shader_files()
        if not files:
            tk.Label(self.tech_frame, text="No shaders in folder", bg=PANEL, fg=MUTED, font=FONT_SM).pack(anchor="w")
            return
        for path in files:
            enabled = path.suffix == ".glsl"
            display = path.name if enabled else path.name[:-4]  # strip .off
            row = tk.Frame(self.tech_frame, bg=PANEL)
            row.pack(fill="x", pady=1)
            var = tk.BooleanVar(value=enabled)
            cb = tk.Checkbutton(
                row, variable=var, bg=PANEL, activebackground=PANEL,
                selectcolor=CARD, highlightthickness=0, bd=0,
                command=lambda p=path, v=var: self._toggle_shader(p, v.get()),
            )
            cb.pack(side="left")
            fg = FG if enabled else MUTED
            name_btn = tk.Label(row, text=display, bg=PANEL, fg=fg, font=FONT, anchor="w", cursor="hand2")
            name_btn.pack(side="left", fill="x", expand=True)
            name_btn.bind("<Button-1>", lambda e, p=path: self._select_shader(p))
            self._shader_rows[display] = name_btn
            self._shader_enabled[display] = enabled
        # auto-select the SSR shader if present, else the first
        pick = next((p for p in files if "ssr" in p.name.lower()), files[0])
        self._select_shader(pick)

    def _toggle_shader(self, path: Path, enable: bool) -> None:
        try:
            if enable and path.suffix == ".off":
                new = path.with_name(path.name[:-4])
            elif not enable and path.suffix == ".glsl":
                new = path.with_name(path.name + ".off")
            else:
                return
            if self._sel_path == path:
                self._sel_path = new
            path.rename(new)
        except OSError as exc:
            messagebox.showerror("RoLux", f"Could not toggle shader:\n{exc}")
        self._refresh_shader_list()

    def _select_shader(self, path: Path) -> None:
        self._sel_path = path
        name = path.name[:-4] if path.suffix == ".off" else path.name
        self.lbl_sel.configure(text=name)
        for disp, lbl in self._shader_rows.items():
            if disp == name:
                lbl.configure(fg=ACCENT)
            else:
                lbl.configure(fg=FG if self._shader_enabled.get(disp, True) else MUTED)
        self._param_lines, self._params = parse_params(path)
        self._build_param_widgets()

    def _build_param_widgets(self) -> None:
        for w in self.param_host.winfo_children():
            w.destroy()
        if not self._params:
            tk.Label(self.param_host, text="No adjustable #define parameters", bg=PANEL, fg=MUTED, font=FONT_SM).pack(
                anchor="w", padx=4, pady=6
            )
            return
        for p in self._params:
            self._param_row(p)

    def _param_row(self, p: ShaderParam) -> None:
        wrap = tk.Frame(self.param_host, bg=PANEL)
        wrap.pack(fill="x", padx=2, pady=(2, 4))
        top = tk.Frame(wrap, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top, text=p.name, bg=PANEL, fg=FG, font=FONT_SM, anchor="w").pack(side="left")
        val_lbl = tk.Label(top, text="", bg=PANEL, fg=ACCENT, font=FONT_MONO)
        val_lbl.pack(side="right")
        if p.desc:
            tk.Label(wrap, text=p.desc, bg=PANEL, fg=MUTED, font=("Segoe UI", 8), anchor="w").pack(fill="x")

        var = tk.DoubleVar(value=p.value)

        def _fmt(v: float) -> str:
            return str(int(round(v))) if p.is_int else f"{v:.3f}".rstrip("0").rstrip(".")

        def _on_change(_=None) -> None:
            raw = var.get()
            snapped = p.vmin + round((raw - p.vmin) / p.step) * p.step if p.step else raw
            snapped = max(p.vmin, min(p.vmax, snapped))
            if p.is_int:
                snapped = round(snapped)
            p.value = snapped
            val_lbl.configure(text=_fmt(snapped))
            self._schedule_write()

        ttk.Scale(wrap, from_=p.vmin, to=p.vmax, variable=var, command=lambda _: _on_change()).pack(
            fill="x", pady=(1, 0)
        )
        val_lbl.configure(text=_fmt(p.value))

    def _schedule_write(self) -> None:
        if self._write_after is not None:
            try:
                self.root.after_cancel(self._write_after)
            except Exception:
                pass
        self._write_after = self.root.after(160, self._write_params)

    def _write_params(self) -> None:
        self._write_after = None
        if not self._sel_path or not self._params:
            return
        lines = self._param_lines
        for p in self._params:
            if 0 <= p.line_idx < len(lines):
                lines[p.line_idx] = f"{p.prefix}{format_value(p.value, p.is_int)}{p.suffix}"
        try:
            self._sel_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"[Rolux] param write failed: {exc}")

    # ------------------------------------------------------ temp shaders
    def _init_temp_shaders(self) -> None:
        """(Re)create shaders/temp as a fresh copy of the pristine originals."""
        try:
            if self.shaders_dir.exists():
                shutil.rmtree(self.shaders_dir, ignore_errors=True)
            self.shaders_dir.mkdir(parents=True, exist_ok=True)
            for p in self.orig_shaders_dir.iterdir():
                if p.is_file() and (
                    p.suffix in (".glsl", ".frag") or p.name.endswith(".glsl.off")
                ):
                    shutil.copy2(p, self.shaders_dir / p.name)
        except Exception as exc:
            print(f"[Rolux] temp shaders init failed ({exc}); using originals")
            self.shaders_dir = self.orig_shaders_dir

    def _cleanup_temp_shaders(self) -> None:
        """Delete shaders/temp (guarded so originals are never removed)."""
        try:
            if (
                self.shaders_dir != self.orig_shaders_dir
                and self.shaders_dir.name == "temp"
                and self.shaders_dir.exists()
            ):
                shutil.rmtree(self.shaders_dir, ignore_errors=True)
        except Exception:
            pass

    def _reset_shaders(self) -> None:
        if not messagebox.askyesno(
            "RoLux",
            "Reset ALL shaders to default?\nThis discards this session's edits and re-enables every effect.",
        ):
            return
        self._flush_pending_writes()
        self._write_after = None
        self._init_temp_shaders()
        self._refresh_shader_list()
        self.lbl_state.configure(text="Shaders reset to default", fg=OK)

    # ---------------------------------------------------------- presets
    def _flush_pending_writes(self) -> None:
        if self._write_after is not None:
            try:
                self.root.after_cancel(self._write_after)
            except Exception:
                pass
            self._write_params()

    def _refresh_presets(self, select: Optional[str] = None) -> None:
        names = [p.stem for p in presets.list_presets(self.presets_dir)]
        self.cmb_preset.configure(values=names)
        if select and select in names:
            self.var_preset.set(select)
        elif self.var_preset.get() not in names:
            self.var_preset.set(names[0] if names else "")

    def _preset_load(self) -> None:
        name = self.var_preset.get().strip()
        if not name:
            return
        path = self.presets_dir / f"{name}{presets.PRESET_EXT}"
        if not path.is_file():
            messagebox.showerror("RoLux", f"Preset not found:\n{path}")
            return
        try:
            self._flush_pending_writes()
            presets.apply_preset(self.shaders_dir, presets.load_preset(path))
        except Exception as exc:
            messagebox.showerror("RoLux", f"Failed to load preset:\n{exc}")
            return
        self._refresh_shader_list()
        self.lbl_state.configure(text=f"Loaded preset '{name}'", fg=OK)

    def _preset_save(self) -> None:
        name = self.var_preset.get().strip()
        if not name:
            self._preset_save_as()
            return
        self._do_save_preset(name)

    def _preset_save_as(self) -> None:
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self.root)
        if not name:
            return
        self._do_save_preset(name.strip())

    def _do_save_preset(self, name: str) -> None:
        try:
            self._flush_pending_writes()
            data = presets.collect_preset(self.shaders_dir)
            presets.save_preset(self.presets_dir, name, data)
        except Exception as exc:
            messagebox.showerror("RoLux", f"Failed to save preset:\n{exc}")
            return
        self._refresh_presets(select=name)
        self.lbl_state.configure(text=f"Saved preset '{name}'", fg=OK)

    def _preset_delete(self) -> None:
        name = self.var_preset.get().strip()
        if not name:
            return
        if not messagebox.askyesno("RoLux", f"Delete preset '{name}'?"):
            return
        try:
            (self.presets_dir / f"{name}{presets.PRESET_EXT}").unlink(missing_ok=True)
        except OSError as exc:
            messagebox.showerror("RoLux", f"Could not delete:\n{exc}")
        self._refresh_presets()

    def _open_folder(self) -> None:
        try:
            self.shaders_dir.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["explorer", str(self.shaders_dir)])
        except Exception as exc:
            messagebox.showinfo("RoLux", f"Shaders folder:\n{self.shaders_dir}\n\n{exc}")

    # -------------------------------------------------------------- misc
    def _browse_engine(self) -> None:
        path = filedialog.askopenfilename(
            title="Select TensorRT engine",
            filetypes=[("TensorRT engine", "*.engine"), ("All files", "*.*")],
        )
        if path:
            self.var_engine.set(path)

    def _on_opacity_change(self, *_args) -> None:
        if self.overlay is not None:
            self.overlay.set_opacity(float(self.var_opacity.get()))

    def _on_allow_shot(self) -> None:
        allow = bool(self.var_allow_shot.get())
        self.status["allow_screen_capture"] = allow
        if self.overlay is not None:
            self.overlay.set_exclude_from_capture(not allow)
        self._update_recording_label()
        if allow:
            msg = "Recording ON — effects visible to capture apps"
            color = OK
        else:
            msg = "Recording OFF — overlay hidden from capture (best for DXGI)"
            color = MUTED
        if self.running:
            self.lbl_state.configure(text=msg, fg=color)
        else:
            self.lbl_state.configure(text=f"{msg} (applies on Start)", fg=color)

    def _update_recording_label(self) -> None:
        lbl = getattr(self, "lbl_recording", None)
        if lbl is None:
            return
        if bool(self.var_allow_shot.get()):
            lbl.configure(text="Recording: effects visible to capture apps")
        else:
            lbl.configure(text="Recording: overlay hidden from capture (default)")

    def _on_temporal_change(self) -> None:
        """Toggle all temporal features live (and remember for next start)."""
        on = bool(self.var_temporal.get())
        if self.shaders is not None:
            self.shaders.hist_on = on
            self.shaders.depth_filter_on = on
        if self.infer is not None:
            self.infer.stabilize = on

    def _on_render_scale_change(self) -> None:
        if self.shaders is not None:
            self.shaders.shader_max_dim = max(256, int(self.var_render.get()))

    def _save_normals(self) -> None:
        if self.shaders is None:
            messagebox.showinfo("RoLux", "Start the overlay first.")
            return
        self.shaders.request_save_normals()
        self.lbl_state.configure(text="Saving normals…", fg=ACCENT)

    def _save_overlay(self) -> None:
        if self.shaders is None:
            messagebox.showinfo("RoLux", "Start the overlay first.")
            return
        self.shaders.request_save_overlay()
        self.lbl_state.configure(text="Saving overlay…", fg=ACCENT)

    def _set_stats(self, fps: float, infer_ms: float, e2e_ms: float = 0.0) -> None:
        self.lbl_fps.configure(text=f"Overlay: {fps:.0f} FPS")
        self.lbl_infer.configure(text=f"Infer: {infer_ms:.1f} · e2e {e2e_ms:.0f} ms")
        cap = self.status.get("capture_fps")
        if cap is not None:
            self.lbl_cap.configure(text=f"Capture: {float(cap):.0f} FPS")

    def _pulse_status(self) -> None:
        found = bool(self.status.get("roblox_found", False))
        focused = bool(self.status.get("focused", False))
        self.lbl_roblox.configure(text=f"Roblox: {'found' if found else 'missing'}")
        self.lbl_focus.configure(text=f"Focus: {'yes' if focused else 'no'}")
        self.lbl_badges.configure(
            text=f"Roblox {'●' if found else '○'}  Focus {'●' if focused else '○'}",
            fg=OK if (found and focused) else MUTED,
        )
        cap = self.status.get("capture_fps")
        if self.running and cap is not None:
            self.lbl_cap.configure(text=f"Capture: {float(cap):.0f} FPS")
        sh = self.status.get("shaders")
        if isinstance(sh, list):
            self.lbl_shaders.configure(text=("Active: " + ", ".join(sh)) if sh else "Active: (none)")
        self._update_recording_label()
        last = self.status.pop("last_capture", None) if self.running else None
        if last:
            self.lbl_state.configure(text=f"Saved {Path(str(last)).name}", fg=OK)
        elif self.running:
            cur = self.lbl_state.cget("text")
            if not (str(cur).startswith("Saving") or str(cur).startswith("Saved ")):
                self.lbl_state.configure(text="Running", fg=OK)
        else:
            self.lbl_state.configure(text="Idle", fg=MUTED)
        self.root.after(200, self._pulse_status)

    # ------------------------------------------------------------- run
    def start(self) -> None:
        if self.running:
            return
        engine = Path(self.var_engine.get())
        if not engine.is_file():
            messagebox.showerror("RoLux", f"Engine not found:\n{engine}")
            return

        cfg = RoluxConfig(
            window_title_substring=self.var_title.get().strip() or "Roblox",
            target_fps=int(self.var_fps.get()),
            require_focus=bool(self.var_focus.get()),
            input_h=int(self.var_size.get()),
            input_w=int(self.var_size.get()),
            engine_path=engine,
            overlay_opacity=float(self.var_opacity.get()),
            shaders_dir=self.shaders_dir,
            shader_max_dim=max(256, int(self.var_render.get())),
            allow_screen_capture=bool(self.var_allow_shot.get()),
            temporal_accumulation=bool(self.var_temporal.get()),
            depth_temporal_filter=bool(self.var_temporal.get()),
            stabilize_depth_range=bool(self.var_temporal.get()),
        )

        self.stop_event = threading.Event()
        self.depth_ready = threading.Event()
        self.frame_ready = threading.Event()
        self.capture_slot[0] = None
        self.raw_slot[0] = None
        self.depth_slot[0] = None
        self.status.clear()
        self.status.update(roblox_found=False, focused=False, shaders=[])

        self.infer = InferenceWorker(
            cfg, self.capture_slot, self.capture_lock, self.raw_slot, self.raw_lock,
            self.stop_event, frame_ready=self.depth_ready, attach_main=True,
        )
        self.shaders = ShaderWorker(
            cfg, self.raw_slot, self.raw_lock, self.depth_slot, self.depth_lock,
            self.stop_event, self.depth_ready, self.frame_ready,
            shaders_dir=cfg.shaders_dir, status=self.status,
        )
        self.capture = CaptureWorker(
            cfg, self.capture_slot, self.capture_lock, self.stop_event, status=self.status,
        )

        try:
            self.lbl_state.configure(text="Loading engine…", fg=ACCENT)
            self.root.update_idletasks()
            self.infer.setup()
            nh, nw = self.infer.network_size
            self.capture.set_network_size(nh, nw)
            if int(self.var_size.get()) != nh:
                self.var_size.set(nh)
                self.lbl_state.configure(text=f"Engine is {nw}×{nh}", fg=ACCENT)
                self.root.update_idletasks()
        except Exception as exc:
            messagebox.showerror("RoLux", f"Failed to load TensorRT engine:\n{exc}")
            self.infer = self.capture = self.shaders = None
            self.stop_event = None
            return

        try:
            import ctypes

            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080
            )
        except Exception:
            pass

        def _stats(fps: float, ms: float, e2e: float = 0.0) -> None:
            self.root.after(0, lambda: self._set_stats(fps, ms, e2e))

        self.overlay = DepthOverlay(
            cfg, self.depth_slot, self.depth_lock, self.status,
            self.stop_event, self.frame_ready, on_stats=_stats,
        )
        self.overlay.start()
        self.overlay.wait_ready(2.0)
        allow_cap = bool(cfg.allow_screen_capture)
        self.status["allow_screen_capture"] = allow_cap
        self.overlay.set_exclude_from_capture(not allow_cap)
        self._update_recording_label()

        self.shaders.start()
        self.infer.start()
        self.capture.start()
        self.running = True
        self.btn_start.state(["disabled"])
        self.btn_stop.state(["!disabled"])
        self.btn_save_n.state(["!disabled"])
        self.btn_save_o.state(["!disabled"])
        self.lbl_state.configure(text="Running — click Roblox", fg=OK)
        self.root.after(100, self._focus_roblox)

    def _focus_roblox(self) -> None:
        try:
            import win32con
            import win32gui

            from rolux.win32_utils import find_window_hwnd

            hwnd = find_window_hwnd(self.var_title.get().strip() or "Roblox")
            if not hwnd:
                self.lbl_state.configure(text="Waiting for Roblox…", fg=ACCENT)
                self.root.after(500, self._focus_roblox)
                return
            try:
                import ctypes

                ctypes.windll.user32.AllowSetForegroundWindow(-1)
            except Exception:
                pass
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            self.lbl_state.configure(text="Running", fg=OK)
        except Exception as exc:
            print(f"[Rolux] focus Roblox: {exc}")
            self.lbl_state.configure(text="Running — click Roblox", fg=OK)

    def stop(self) -> None:
        if not self.running:
            return
        if self.stop_event is not None:
            self.stop_event.set()
        if getattr(self, "depth_ready", None) is not None:
            self.depth_ready.set()
        if getattr(self, "frame_ready", None) is not None:
            self.frame_ready.set()
        if self.overlay is not None:
            self.overlay.destroy()
            self.overlay = None
        if self.shaders is not None:
            self.shaders.join(timeout=2.0)
            self.shaders = None
        if self.capture is not None:
            self.capture.join(timeout=2.0)
            self.capture = None
        if self.infer is not None:
            self.infer.join(timeout=2.0)
            self.infer = None
        self.stop_event = None
        self.running = False
        self.btn_start.state(["!disabled"])
        self.btn_stop.state(["disabled"])
        self.btn_save_n.state(["disabled"])
        self.btn_save_o.state(["disabled"])
        self.lbl_state.configure(text="Idle", fg=MUTED)
        self.lbl_fps.configure(text="Overlay: — FPS")
        self.lbl_infer.configure(text="Infer: — · e2e — ms")
        self.lbl_cap.configure(text="Capture: — FPS")
        self.lbl_shaders.configure(text="Active: —")

    def _on_close(self) -> None:
        self.stop()
        self._cleanup_temp_shaders()
        self.root.destroy()


def run_gui() -> int:
    from rolux.overlay_ui import ensure_dpi_aware

    ensure_dpi_aware()
    root = tk.Tk()
    RoluxApp(root)
    root.mainloop()
    return 0
