"""
VIDEX Programming Language — Continuous-Time Cinema Engine (Stages 3–4)
=======================================================================

Pipeline Stages 3 & 4:
    VIDEX Code -> Lexer -> Parser -> Validator -> [CONTINUOUS CINEMA ENGINE] -> MP4

Core Architecture:
  1. Continuous Scene Segments (`SceneSegment`):
     Consecutive `second` blocks that stay in the same world share ONE unbroken
     continuous camera & object state timeline — zero boundary resets, zero
     frozen holds, and zero sudden velocity snaps.
  2. Dual-Stream Overlapping Transitions:
     When a scene transitions (`fade`, `dissolve`, `wipe`, `slide`, `zoom in`,
     `zoom out`, `cut`), BOTH outgoing scene A(t) and incoming scene B(t)
     continue animating live throughout the entire overlap window and blend in
     gamma-correct linear light (`gamma = 2.2`).
  3. True Sub-Pixel 2.5D/3D Rendering:
     - Floating-point `cv2.remap` 2.5D depth-parallax camera motion
     - 5-layer 3D ray-traced Planet Earth (continents, ocean solar glint,
       super-rotating cloud deck + shadows, night city lights, Rayleigh halo)
     - Biomechanical slow wingbeat + thermal soar glide on Golden Eagle
     - Seamless periodic alpine mist, warm solar bloom & wind-blown snow motes
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
import subprocess
import tempfile
import wave

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from videx_mograph import (
    MoGraphStudio,
    blend_mograph_transition,
    is_light_mograph_scene,
    is_mograph_object,
    is_mograph_scene,
    render_kinetic_text_layer,
    write_mograph_electronic_score,
)
from videx_parser import Command, SecondBlock, VideoProgram


# ══════════════════════════════════════════════════════════════════════════════
# COLOR, ANCHOR, EASING & LINEAR-LIGHT OPTICS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (12, 14, 18),
    "blue": (72, 168, 255),
    "red": (239, 71, 88),
    "green": (64, 205, 130),
    "yellow": (255, 214, 82),
    "orange": (255, 148, 66),
    "purple": (168, 102, 255),
    "cyan": (76, 228, 245),
    "gold": (255, 206, 96),
    "silver": (205, 214, 226),
    "warm": (255, 204, 148),
    "cold": (176, 218, 255),
}

SIZE_SCALE: dict[str, float] = {
    "small": 0.42,
    "medium": 1.0,
    "large": 1.42,
}

SPEED_DELTA: dict[str, float] = {
    "slow": 0.34,
    "medium": 0.62,
    "fast": 0.95,
}

TIME_COLOR_GRADE: dict[str, tuple[tuple[float, float, float], float]] = {
    "sunrise": ((1.05, 0.99, 0.93), 1.02),
    "golden hour": ((1.11, 0.98, 0.85), 1.04),
    "sunset": ((1.13, 0.90, 0.83), 0.96),
    "morning": ((1.01, 1.01, 1.02), 1.03),
    "afternoon": ((1.03, 1.01, 0.97), 1.05),
    "dawn": ((0.94, 0.96, 1.08), 0.90),
    "dusk": ((0.95, 0.88, 1.08), 0.84),
    "night": ((0.89, 0.95, 1.09), 0.90),
    "midnight": ((0.80, 0.88, 1.14), 0.78),
}


def resolve_rgb(color_str: str) -> tuple[int, int, int]:
    c = color_str.strip().lower()
    if c in COLOR_RGB:
        return COLOR_RGB[c]
    if c.startswith("#"):
        hex_part = c[1:]
        if len(hex_part) in (3, 4):
            hex_part = "".join(ch * 2 for ch in hex_part[:3])
        if len(hex_part) >= 6:
            return (
                int(hex_part[0:2], 16),
                int(hex_part[2:4], 16),
                int(hex_part[4:6], 16),
            )
    return (255, 255, 255)


def anchor_normalized(position: str) -> tuple[float, float]:
    mapping: dict[str, tuple[float, float]] = {
        "center": (0.50, 0.50),
        "left": (0.18, 0.50),
        "right": (0.82, 0.50),
        "top": (0.50, 0.20),
        "bottom": (0.50, 0.84),
        "top left": (0.15, 0.26),
        "top right": (0.76, 0.13),
        "top center": (0.50, 0.18),
        "bottom left": (0.22, 0.84),
        "bottom right": (0.78, 0.84),
        "bottom center": (0.50, 0.835),
    }
    return mapping.get(position.lower(), (0.50, 0.50))


def load_font(
    size_px: int,
    *,
    serif: bool = False,
    mono: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if mono:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    elif serif:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=max(11, int(size_px)))
    return ImageFont.load_default()


def ease_c2(t: float) -> float:
    """Quintic Perlin smoothstep with zero velocity AND zero acceleration at ends."""
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def ease_gentle_flow(t: float) -> float:
    """
    Smooth cinema drone curve: gentle ease-in, steady airborne glide, gentle
    cushion — never whips violently and never slams into a dead stop mid-shot.
    """
    t = max(0.0, min(1.0, t))
    return 0.65 * t + 0.35 * ease_c2(t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def aces_filmic_tonemap(linear_rgb: np.ndarray) -> np.ndarray:
    """Narkowicz ACES filmic tone curve for smooth cinema highlight roll-off."""
    x = np.maximum(linear_rgb, 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(mapped, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS TIMELINE CURVE & SCENE SEGMENT DATA MODELS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScalarTrack:
    """Piecewise smooth $C^2$ trajectory of a continuous scalar parameter."""

    default: float = 0.0
    keyframes: list[tuple[float, float]] = field(default_factory=list)
    flow_mode: bool = False  # True for continuous camera/object glide

    def add_point(self, frame_pos: float, value: float) -> None:
        for i, (f_existing, _) in enumerate(self.keyframes):
            if abs(f_existing - frame_pos) < 0.01:
                self.keyframes[i] = (frame_pos, value)
                return
        self.keyframes.append((frame_pos, value))
        self.keyframes.sort(key=lambda kv: kv[0])

    def last_value(self) -> float:
        return self.keyframes[-1][1] if self.keyframes else self.default

    def evaluate(self, frame_pos: float) -> float:
        if not self.keyframes:
            return self.default
        if frame_pos <= self.keyframes[0][0]:
            return self.keyframes[0][1]
        if frame_pos >= self.keyframes[-1][0]:
            return self.keyframes[-1][1]
        for idx in range(len(self.keyframes) - 1):
            f0, v0 = self.keyframes[idx]
            f1, v1 = self.keyframes[idx + 1]
            if f0 <= frame_pos <= f1:
                raw = (frame_pos - f0) / max(f1 - f0, 1e-5)
                eased = ease_gentle_flow(raw) if self.flow_mode else ease_c2(raw)
                return _lerp(v0, v1, eased)
        return self.keyframes[-1][1]


@dataclass
class ObjectTrack:
    """Continuous per-object motion, scale, rotation, opacity & effect curves."""

    name: str
    appear_frame: float
    remove_frame: float = 1e9
    x: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.50, flow_mode=True))
    y: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.50, flow_mode=True))
    scale: ScalarTrack = field(default_factory=lambda: ScalarTrack(1.0, flow_mode=True))
    rotation: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.0, flow_mode=True))
    opacity: ScalarTrack = field(default_factory=lambda: ScalarTrack(1.0))
    glow: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.0))
    color: tuple[int, int, int] = (56, 189, 248)
    glow_color: tuple[int, int, int] = (72, 168, 255)
    shadow: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    file: str | None = None


@dataclass
class TextEvent:
    content: str = ""
    x: float = 0.50
    y: float = 0.835
    size: float = 50.0
    color: tuple[int, int, int] = (255, 255, 255)
    start_frame: float = 0.0
    fade_in_frames: float = 24.0
    fade_out_frames: float = 0.0
    glow: float = 0.0
    glow_color: tuple[int, int, int] = (255, 255, 255)


@dataclass
class ObjectState:
    """Evaluated instantaneous snapshot of an object at global frame `f`."""

    name: str
    x: float = 0.50
    y: float = 0.50
    vx: float = 0.0
    vy: float = 0.0
    local_age_s: float = 0.0
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0
    glow: float = 0.0
    color: tuple[int, int, int] = (56, 189, 248)
    glow_color: tuple[int, int, int] = (72, 168, 255)
    shadow: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    file: str | None = None


@dataclass
class SceneState:
    """Evaluated instantaneous snapshot of a scene at global frame `f`."""

    background: str = "space galaxy"
    background_file: str | None = None
    background_blur: float = 0.0
    time_of_day: str = "night"
    weather: str = "clear"
    season: str = "summer"
    light_color: tuple[int, int, int] = (210, 225, 255)
    light_brightness: float = 0.78
    cam_pan_x: float = 0.0
    cam_tilt_y: float = 0.0
    cam_zoom: float = 1.0
    cam_shake: float = 0.0
    scene_effects: dict[str, float] = field(default_factory=dict)
    objects: dict[str, ObjectState] = field(default_factory=dict)
    texts: list[tuple[TextEvent, float, float]] = field(default_factory=list)  # (ev, alpha, rise_px)
    narration: str = ""
    narration_alpha: float = 0.0


@dataclass
class SceneSegment:
    """
    A continuous single-world shot spanning `[render_start, render_end]`,
    allowing smooth multi-second camera curves and live dual-scene transitions!
    """

    background: str = "space galaxy"
    background_file: str | None = None
    time_of_day: str = "night"
    weather: str = "clear"
    season: str = "summer"
    nominal_start: float = 0.0
    nominal_end: float = 26.0
    render_start: int = 0
    render_end: int = 26
    transition_in: str | None = None
    transition_frames: int = 0

    light_r: ScalarTrack = field(default_factory=lambda: ScalarTrack(210.0))
    light_g: ScalarTrack = field(default_factory=lambda: ScalarTrack(225.0))
    light_b: ScalarTrack = field(default_factory=lambda: ScalarTrack(255.0))
    light_brightness: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.78))
    bg_blur: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.0))
    cam_pan_x: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.0, flow_mode=True))
    cam_tilt_y: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.0, flow_mode=True))
    cam_zoom: ScalarTrack = field(default_factory=lambda: ScalarTrack(1.0, flow_mode=True))
    cam_shake: ScalarTrack = field(default_factory=lambda: ScalarTrack(0.0))

    scene_effects: dict[str, ScalarTrack] = field(default_factory=dict)
    objects: dict[str, ObjectTrack] = field(default_factory=dict)
    texts: list[TextEvent] = field(default_factory=list)
    narration_text: str = ""
    narration_frame: float = -1.0

    def sample(self, f: float, fps: int, total_frames: int) -> SceneState:
        st = SceneState(
            background=self.background,
            background_file=self.background_file,
            background_blur=self.bg_blur.evaluate(f),
            time_of_day=self.time_of_day,
            weather=self.weather,
            season=self.season,
            light_color=(
                int(round(self.light_r.evaluate(f))),
                int(round(self.light_g.evaluate(f))),
                int(round(self.light_b.evaluate(f))),
            ),
            light_brightness=self.light_brightness.evaluate(f),
            cam_pan_x=self.cam_pan_x.evaluate(f),
            cam_tilt_y=self.cam_tilt_y.evaluate(f),
            cam_zoom=self.cam_zoom.evaluate(f),
            cam_shake=self.cam_shake.evaluate(f),
            scene_effects={k: tr.evaluate(f) for k, tr in self.scene_effects.items()},
        )

        for name, otrack in self.objects.items():
            if f < otrack.appear_frame - 0.5 or f > otrack.remove_frame:
                continue
            x_now = otrack.x.evaluate(f)
            y_now = otrack.y.evaluate(f)
            x_prev = otrack.x.evaluate(max(0.0, f - 1.0))
            y_prev = otrack.y.evaluate(max(0.0, f - 1.0))
            # Soft entry fade across first 12 frames so objects never hard-pop into existence
            entry_alpha = ease_c2(min(1.0, (f - otrack.appear_frame + 1.0) / 14.0))
            st.objects[name] = ObjectState(
                name=name,
                x=x_now,
                y=y_now,
                vx=x_now - x_prev,
                vy=y_now - y_prev,
                local_age_s=max(0.0, (f - otrack.appear_frame) / max(fps, 1)),
                scale=otrack.scale.evaluate(f),
                rotation=otrack.rotation.evaluate(f),
                opacity=otrack.opacity.evaluate(f) * entry_alpha,
                glow=otrack.glow.evaluate(f),
                color=otrack.color,
                glow_color=otrack.glow_color,
                shadow=otrack.shadow,
                flip_h=otrack.flip_h,
                flip_v=otrack.flip_v,
                file=otrack.file,
            )

        for tev in self.texts:
            if f >= tev.start_frame:
                raw_in = min(1.0, (f - tev.start_frame + 1.0) / max(1.0, tev.fade_in_frames))
                alpha = ease_c2(raw_in)
                # Subtle cinema upward float during fade-in (14px settling smoothly to 0px)
                rise_px = (1.0 - ease_c2(raw_in)) * 14.0
                if tev.fade_out_frames > 0 and self.render_end - f <= tev.fade_out_frames:
                    alpha *= ease_c2(max(0.0, (self.render_end - f) / tev.fade_out_frames))
                st.texts.append((tev, alpha, rise_px))

        if self.narration_text and self.narration_frame >= 0 and f >= self.narration_frame:
            n_in = ease_c2(min(1.0, (f - self.narration_frame + 1.0) / max(fps * 0.75, 8.0)))
            st.narration = self.narration_text
            st.narration_alpha = n_in

        return st


# ══════════════════════════════════════════════════════════════════════════════
# PROCEDURAL NOISE & STUDIO HYBRID ASSET BANK
# ══════════════════════════════════════════════════════════════════════════════


def _value_noise_2d(h: int, w: int, grid_y: int, grid_x: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.random((grid_y, grid_x), dtype=np.float32)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)


def _fbm_2d(h: int, w: int, base_grid: int = 8, octaves: int = 4, seed: int = 42) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 0.5
    total = 0.0
    freq = base_grid
    for o in range(octaves):
        gy = max(2, freq)
        gx = max(2, int(freq * w / max(h, 1)))
        acc += amp * _value_noise_2d(h, w, gy, gx, seed + o * 17)
        total += amp
        amp *= 0.5
        freq *= 2
    return np.clip(acc / max(total, 1e-6), 0.0, 1.0)


def _periodic_cloud_map(h: int, w: int, seed: int = 991) -> np.ndarray:
    """Horizontally 360°-periodic field with zero wrap seam under advection."""
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 0.5
    total = 0.0
    for o, gy in enumerate((6, 12, 24, 48, 96)):
        gx = gy * 2
        rng = np.random.default_rng(seed + o * 29)
        grid = rng.random((gy, gx), dtype=np.float32)
        tiled = np.concatenate([grid[:, -2:], grid, grid[:, :2]], axis=1)
        up = cv2.resize(tiled, (int(w * (gx + 4) / gx), h), interpolation=cv2.INTER_CUBIC)
        margin = int(round(w * 2 / gx))
        acc += amp * up[:, margin : margin + w]
        total += amp
        amp *= 0.52
    return np.clip(acc / max(total, 1e-6), 0.0, 1.0)


class StudioAssetBank:
    """Manages high-resolution scene plates, 2.5D depth fields & 3D shaders."""

    def __init__(self, width: int, height: int, base_dir: Path) -> None:
        self.width = width
        self.height = height
        self.pad_w = int(round(width * 1.22))
        self.pad_h = int(round(height * 1.22))
        self.base_dir = base_dir
        self.assets_dir = Path(__file__).resolve().parent / "assets"

        self._scene_cache: dict[tuple[str, str | None], tuple[np.ndarray, np.ndarray]] = {}
        self._earth_albedo: np.ndarray | None = None
        self._earth_clouds: np.ndarray | None = None
        self._earth_cities: np.ndarray | None = None
        self._eagle_rgba: np.ndarray | None = None
        self.mograph = MoGraphStudio(width, height)

        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        self._screen_x = xx
        self._screen_y = yy
        self._norm_u = (xx / max(width - 1, 1)) * 2.0 - 1.0
        self._norm_v = (yy / max(height - 1, 1)) * 2.0 - 1.0
        self._vignette = np.clip(
            1.0 - 0.22 * (self._norm_u ** 2 * 0.72 + self._norm_v ** 2 * 0.90),
            0.58,
            1.0,
        )[..., None].astype(np.float32)

        self._mist_noise = _periodic_cloud_map(height, width, seed=404)
        rng = np.random.default_rng(777)
        self._particles = np.column_stack(
            [
                rng.uniform(0.0, 1.0, 140),
                rng.uniform(0.28, 0.92, 140),
                rng.uniform(0.35, 1.15, 140),
                rng.uniform(0.8, 2.0, 140),
                rng.uniform(0.0, math.tau, 140),
            ]
        ).astype(np.float32)
        self._stars_twinkle = np.column_stack(
            [
                rng.integers(2, width - 2, 150),
                rng.integers(2, height - 2, 150),
                rng.uniform(1.1, 3.0, 150),
                rng.uniform(0.0, math.tau, 150),
            ]
        ).astype(np.float32)

    def _load_image_rgb(self, path: Path, target_w: int, target_h: int) -> np.ndarray:
        img = Image.open(path).convert("RGB").resize(
            (target_w, target_h), Image.Resampling.LANCZOS
        )
        return np.asarray(img, dtype=np.float32)

    def get_scene_and_depth(
        self, theme: str, file_path: str | None
    ) -> tuple[np.ndarray, np.ndarray]:
        key = (theme.lower(), file_path)
        if key in self._scene_cache:
            return self._scene_cache[key]

        rgb: np.ndarray | None = None
        if file_path:
            cand = Path(file_path) if Path(file_path).is_absolute() else (self.base_dir / file_path)
            if cand.is_file():
                rgb = self._load_image_rgb(cand, self.pad_w, self.pad_h)

        t_low = theme.lower()
        if rgb is None:
            if "space" in t_low or "galaxy" in t_low or "star" in t_low:
                plate = self.assets_dir / "space_galaxy.jpg"
                if plate.is_file():
                    rgb = self._load_image_rgb(plate, self.pad_w, self.pad_h)
                    yy, xx = np.mgrid[0 : self.pad_h, 0 : self.pad_w].astype(np.float32)
                    nu = (xx / self.pad_w) * 2.0 - 1.0
                    nv = (yy / self.pad_h) * 2.0 - 1.0
                    center_dim = 0.50 + 0.50 * np.clip((nu ** 2 + nv ** 2) * 0.95, 0.0, 1.0)
                    rgb = rgb * center_dim[..., None]
                else:
                    rgb = self._procedural_space(self.pad_w, self.pad_h)
            elif "mountain" in t_low or "himalaya" in t_low:
                plate = self.assets_dir / "himalayas_sunrise.jpg"
                if plate.is_file():
                    rgb = self._load_image_rgb(plate, self.pad_w, self.pad_h)
                else:
                    rgb = self._procedural_mountains(self.pad_w, self.pad_h)
            else:
                rgb = self._procedural_scenic(self.pad_w, self.pad_h, t_low)

        v_ramp = np.linspace(0.20, 1.0, self.pad_h, dtype=np.float32)[:, None]
        luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        detail = np.clip(np.hypot(gx, gy) * 1.3, 0.0, 1.0)
        if "space" in t_low or "galaxy" in t_low:
            depth = np.clip(0.30 + 0.42 * luma, 0.22, 0.85)
        else:
            depth = np.clip(v_ramp * 0.78 + detail * 0.24 * v_ramp, 0.20, 1.0)
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=20.0, sigmaY=20.0)

        self._scene_cache[key] = (rgb, depth)
        return rgb, depth

    def sample_camera_subpixel(
        self,
        bg_rgb: np.ndarray,
        depth: np.ndarray,
        pan_x: float,
        tilt_y: float,
        zoom: float,
        shake_px: tuple[float, float] = (0.0, 0.0),
    ) -> np.ndarray:
        """Continuous float32 sub-pixel camera sampler with smooth 2.5D parallax."""
        ch, cw, _ = bg_rgb.shape
        margin_x = (cw - self.width) * 0.5
        margin_y = (ch - self.height) * 0.5

        inv_z = 1.0 / max(zoom, 0.5)
        cx_pad = cw * 0.5
        cy_pad = ch * 0.5

        base_x = cx_pad + (self._screen_x - self.width * 0.5) * inv_z + shake_px[0]
        base_y = cy_pad + (self._screen_y - self.height * 0.5) * inv_z + shake_px[1]

        # Sub-pixel bilinear sample of smooth depth field so depth NEVER steps by integer index!
        local_depth = cv2.remap(
            depth,
            base_x.astype(np.float32),
            base_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        map_x = base_x + pan_x * margin_x * (0.44 + 0.78 * local_depth)
        map_y = base_y + tilt_y * margin_y * (0.44 + 0.78 * local_depth)

        return cv2.remap(
            bg_rgb,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _ensure_earth_layers(self) -> None:
        if self._earth_albedo is not None:
            return
        eh, ew = 720, 1440
        eq_path = self.assets_dir / "earth_equirect.jpg"
        if eq_path.is_file():
            self._earth_albedo = self._load_image_rgb(eq_path, ew, eh)
        else:
            cont = _fbm_2d(eh, ew, base_grid=6, octaves=5, seed=512)
            ocean = np.array([12.0, 52.0, 132.0], dtype=np.float32)
            land = np.array([46.0, 128.0, 68.0], dtype=np.float32)
            self._earth_albedo = np.where((cont > 0.51)[..., None], land, ocean)

        self._earth_clouds = _periodic_cloud_map(eh, ew, seed=991)

        alb = self._earth_albedo
        is_ocean = (alb[..., 2] > alb[..., 1] * 1.08) & (alb[..., 2] > alb[..., 0] * 1.25)
        lat = np.abs(np.linspace(-1.0, 1.0, eh, dtype=np.float32))[:, None]
        habitable = (~is_ocean) & (lat < 0.68)
        sparks = _fbm_2d(eh, ew, base_grid=42, octaves=3, seed=2048)
        fine = _fbm_2d(eh, ew, base_grid=120, octaves=2, seed=4096)
        cities = np.clip((sparks - 0.58) * 3.4, 0.0, 1.0) * np.clip((fine - 0.45) * 2.8, 0.0, 1.0)
        self._earth_cities = cities * habitable.astype(np.float32)

    def render_earth_rgba(
        self,
        diameter: int,
        rotation_deg: float,
        glow: float,
        glow_color: tuple[int, int, int],
    ) -> Image.Image:
        self._ensure_earth_layers()
        assert self._earth_albedo is not None
        assert self._earth_clouds is not None
        assert self._earth_cities is not None

        pad = int(diameter * 0.34)
        size = diameter + 2 * pad
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = cy = (size - 1) * 0.5
        r = max(diameter * 0.5, 1.0)
        nx = (xx - cx) / r
        ny = (yy - cy) / r
        r2 = nx * nx + ny * ny
        inside = r2 <= 1.0
        nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))

        tilt_rad = math.radians(17.0)
        cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)
        ry = ny * cos_t - nz * sin_t
        rz = ny * sin_t + nz * cos_t

        base_lon = math.radians(58.0 + rotation_deg * 1.65)
        lon = np.arctan2(nx, rz) + base_lon
        lat = np.arcsin(np.clip(ry, -1.0, 1.0))

        eh, ew, _ = self._earth_albedo.shape
        u_float = np.mod((lon / math.tau) + 0.5, 1.0) * (ew - 1)
        v_float = np.clip((lat / math.pi) + 0.5, 0.0, 1.0) * (eh - 1)

        albedo = cv2.remap(
            self._earth_albedo,
            u_float.astype(np.float32),
            v_float.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )

        u_cloud = np.mod(u_float + rotation_deg * 0.95, float(ew))
        cloud_raw = cv2.remap(
            self._earth_clouds,
            u_cloud.astype(np.float32),
            v_float.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        cloud_shadow_raw = cv2.remap(
            self._earth_clouds,
            np.mod(u_cloud + 5.5, float(ew)).astype(np.float32),
            np.clip(v_float - 3.5, 0.0, eh - 1).astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        cloud_cover = np.clip((cloud_raw - 0.50) * 2.45, 0.0, 0.90)
        shadow_mul = 1.0 - 0.34 * np.clip((cloud_shadow_raw - 0.50) * 2.3, 0.0, 1.0)

        cities = cv2.remap(
            self._earth_cities,
            u_float.astype(np.float32),
            v_float.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )

        lx, ly, lz = -0.52, -0.26, 0.81
        ndotl = nx * lx + ny * ly + nz * lz
        day_light = np.clip(ndotl, 0.0, 1.0) ** 0.80

        hx, hy, hz = lx, ly, lz + 1.0
        h_norm = math.sqrt(hx * hx + hy * hy + hz * hz)
        ndoth = np.clip((nx * hx + ny * hy + nz * hz) / h_norm, 0.0, 1.0)
        is_ocean = (albedo[..., 2] > albedo[..., 1] * 1.06) & (
            albedo[..., 2] > albedo[..., 0] * 1.20
        )
        ocean_glint = (ndoth ** 28.0) * is_ocean.astype(np.float32) * (1.0 - cloud_cover)

        shaded = albedo * shadow_mul[..., None] * (0.045 + 1.12 * day_light[..., None])
        shaded += ocean_glint[..., None] * np.array([255.0, 236.0, 204.0], dtype=np.float32) * 0.55

        cloud_lit = (0.07 + 1.05 * np.clip(ndotl + 0.06, 0.0, 1.0))[..., None] * 252.0
        shaded = shaded * (1.0 - cloud_cover[..., None]) + cloud_lit * cloud_cover[..., None]

        night_mask = np.clip((-ndotl + 0.06) * 5.0, 0.0, 1.0) * (1.0 - cloud_cover * 0.65)
        shaded += (cities * night_mask)[..., None] * np.array([255.0, 192.0, 105.0], dtype=np.float32) * 1.45

        fresnel = ((1.0 - nz) ** 3.1) * (0.28 + 0.78 * np.clip(ndotl + 0.25, 0.0, 1.0))
        shaded += fresnel[..., None] * np.array([76.0, 168.0, 255.0], dtype=np.float32) * 0.92

        rgba = np.zeros((size, size, 4), dtype=np.float32)
        dist_norm = np.sqrt(r2)

        atmo = (
            np.exp(-np.clip(dist_norm - 0.985, 0.0, 4.0) * (10.5 - 4.5 * glow)) * (0.42 + 0.50 * glow)
            + np.exp(-np.clip(dist_norm - 0.985, 0.0, 4.0) * 3.6) * (0.08 + 0.24 * glow)
        )
        outside = ~inside
        gc = np.array(glow_color, dtype=np.float32)
        rgba[outside, :3] = gc
        rgba[outside, 3] = np.clip(atmo[outside] * 255.0, 0.0, 255.0)

        edge_aa = np.clip((1.0 - dist_norm) * r * 0.75 + 0.5, 0.0, 1.0)
        rgba[inside, :3] = np.clip(shaded[inside], 0.0, 255.0)
        rgba[inside, 3] = edge_aa[inside] * 255.0

        return Image.fromarray(np.clip(rgba, 0.0, 255.0).astype(np.uint8), mode="RGBA")

    def render_eagle_rgba(
        self,
        span_px: int,
        flight_age_s: float,
        bank_deg: float,
        opacity: float,
        shadow: float,
    ) -> Image.Image:
        """
        Natural large-raptor flight biomechanics:
        Two slow, deep majestic 1.15-second wingbeats as the eagle climbs onto
        the mountain thermal, transitioning smoothly into an outstretched soar
        glide into the golden sunrise distance (zero mechanical buzzing!).
        """
        if self._eagle_rgba is None:
            eagle_path = self.assets_dir / "golden_eagle.png"
            if eagle_path.is_file():
                raw = np.asarray(Image.open(eagle_path).convert("RGB"), dtype=np.float32)
                max_c = np.max(raw, axis=-1)
                alpha = np.clip((max_c - 9.0) / 34.0, 0.0, 1.0) ** 0.72
                alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.8)
                ys, xs = np.where(alpha > 0.04)
                if len(xs) > 10 and len(ys) > 10:
                    x0, x1 = max(0, int(xs.min()) - 12), min(raw.shape[1], int(xs.max()) + 12)
                    y0, y1 = max(0, int(ys.min()) - 12), min(raw.shape[0], int(ys.max()) + 12)
                    raw = raw[y0:y1, x0:x1]
                    alpha = alpha[y0:y1, x0:x1]
                self._eagle_rgba = np.dstack([raw, alpha * 255.0]).astype(np.float32)

        # Slow ~0.88 Hz wingbeat damping smoothly into serene thermal soaring glide
        glide_envelope = math.exp(-max(0.0, flight_age_s - 0.2) * 0.62)
        flap_wave = math.sin(flight_age_s * math.tau * 0.88) * glide_envelope
        # Tiny natural air-current micro-breathing even during full thermal soar
        soar_breath = math.sin(flight_age_s * 2.1) * 0.06

        if self._eagle_rgba is not None:
            sh0, sw0, _ = self._eagle_rgba.shape
            target_w = max(48, int(span_px * 1.48))
            target_h = max(32, int(target_w * (sh0 / max(sw0, 1))))
            base = cv2.resize(
                self._eagle_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA
            )

            pad_y = int(target_h * 0.45)
            pad_x = int(target_w * 0.18)
            canvas_h = target_h + 2 * pad_y
            canvas_w = target_w + 2 * pad_x
            padded = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
            padded[pad_y : pad_y + target_h, pad_x : pad_x + target_w] = base

            yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
            norm_wing = np.abs(xx - canvas_w * 0.5) / max(target_w * 0.48, 1.0)
            flap_amp = (flap_wave * 0.21 + soar_breath) * target_h
            body_bob = -flap_wave * (target_h * 0.04)

            src_y = yy + flap_amp * (norm_wing ** 1.55) + body_bob
            src_x = xx + flap_wave * 0.035 * (xx - canvas_w * 0.5) * norm_wing

            warped = cv2.remap(
                padded,
                src_x.astype(np.float32),
                src_y.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            rot_mat = cv2.getRotationMatrix2D((canvas_w * 0.5, canvas_h * 0.5), bank_deg, 1.0)
            warped = cv2.warpAffine(
                warped,
                rot_mat,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            out = np.zeros_like(warped)
            if shadow > 0.01:
                sh_alpha = cv2.GaussianBlur(warped[..., 3], (0, 0), sigmaX=max(3.0, span_px * 0.04))
                M_sh = np.float32([[1, 0, span_px * 0.055], [0, 1, span_px * 0.095]])
                sh_shifted = cv2.warpAffine(sh_alpha, M_sh, (canvas_w, canvas_h))
                out[..., 0] = 16.0
                out[..., 1] = 20.0
                out[..., 2] = 34.0
                out[..., 3] = np.clip(sh_shifted * 0.38 * shadow * opacity, 0.0, 255.0)

            bird_a = (warped[..., 3] / 255.0) * opacity
            sh_a = out[..., 3] / 255.0
            comb_a = bird_a + sh_a * (1.0 - bird_a)
            safe_a = np.maximum(comb_a, 1e-5)
            out[..., :3] = (
                warped[..., :3] * bird_a[..., None]
                + out[..., :3] * (sh_a * (1.0 - bird_a))[..., None]
            ) / safe_a[..., None]
            out[..., 3] = comb_a * 255.0
            return Image.fromarray(np.clip(out, 0.0, 255.0).astype(np.uint8), mode="RGBA")

        w, h = max(48, int(span_px * 1.4)), max(36, int(span_px * 0.9))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = w * 0.5, h * 0.54
        s = span_px * 0.42
        flap = flap_wave * (h * 0.14)
        poly = [
            (cx, cy - s * 0.10),
            (cx + s * 0.58, cy - s * 0.30 - flap),
            (cx + s * 0.96, cy - s * 0.16 - flap * 1.3),
            (cx + s * 0.18, cy + s * 0.09),
            (cx, cy + s * 0.28),
            (cx - s * 0.18, cy + s * 0.09),
            (cx - s * 0.96, cy - s * 0.16 - flap * 1.3),
            (cx - s * 0.58, cy - s * 0.30 - flap),
        ]
        draw.polygon(poly, fill=(42, 30, 24, int(245 * opacity)))
        return layer

    @staticmethod
    def _procedural_space(w: int, h: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx, ny = (xx / w) * 2.0 - 1.0, (yy / h) * 2.0 - 1.0
        canvas = np.zeros((h, w, 3), dtype=np.float32) + np.array([5.0, 8.0, 20.0])
        band = np.exp(-((0.55 * nx + 0.83 * ny) ** 2) / 0.14)
        neb = _fbm_2d(h, w, base_grid=7, octaves=5, seed=101)
        canvas += (band * neb)[..., None] * np.array([58.0, 96.0, 175.0])
        return np.clip(canvas, 0.0, 255.0)

    @staticmethod
    def _procedural_mountains(w: int, h: int) -> np.ndarray:
        yy, _ = np.mgrid[0:h, 0:w].astype(np.float32)
        v = (yy / max(h - 1, 1))[..., None]
        top = np.array([28.0, 52.0, 108.0], dtype=np.float32)
        bot = np.array([245.0, 186.0, 118.0], dtype=np.float32)
        return np.clip(top * (1.0 - v) + bot * v, 0.0, 255.0)

    @staticmethod
    def _procedural_scenic(w: int, h: int, theme: str) -> np.ndarray:
        yy, _ = np.mgrid[0:h, 0:w].astype(np.float32)
        v = (yy / max(h - 1, 1))[..., None]
        noise = _fbm_2d(h, w, base_grid=8, octaves=4, seed=abs(hash(theme)) % 10000)[..., None]
        top = np.array([24.0, 64.0, 122.0], dtype=np.float32)
        bot = np.array([76.0, 152.0, 134.0], dtype=np.float32)
        return np.clip(top * (1.0 - v) + bot * v + (noise - 0.5) * 55.0, 0.0, 255.0)


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUOUS-TIME VIDEX VIDEO ENGINE
# ══════════════════════════════════════════════════════════════════════════════


class VidexVideoEngine:
    """Compiles a ``VideoProgram`` into continuous scene acts & encodes MP4."""

    def __init__(self, *, base_dir: Path | None = None, show_hud: bool = False) -> None:
        self.base_dir = base_dir or Path.cwd()
        self.show_hud = show_hud

    def compile_segments(self, program: VideoProgram, total_frames: int) -> list[SceneSegment]:
        """
        Compile AST `SecondBlock`s into continuous `SceneSegment` acts mapped
        proportionally across `[0, total_frames)` so every scene breathes with
        natural cinematic pacing, continuous motion, and zero dead tail frames.
        """
        fps = program.fps

        if not program.seconds:
            seg = SceneSegment(
                nominal_start=0.0,
                nominal_end=float(total_frames - 1),
                render_start=0,
                render_end=total_frames - 1,
            )
            seg.cam_pan_x.add_point(0.0, -0.14)
            seg.cam_pan_x.add_point(float(total_frames - 1), 0.14)
            seg.cam_zoom.add_point(0.0, 1.0)
            seg.cam_zoom.add_point(float(total_frames - 1), 1.06)
            for cmd in program.global_commands:
                self._compile_command(
                    seg, cmd, program, frame_pos=0.0, block_start=0.0, block_end=float(total_frames - 1)
                )
            return [seg]

        blocks = sorted(program.seconds, key=lambda b: b.second_number)
        num_blocks = len(blocks)

        # Proportional act duration: 3 acts in a 10s film -> ~86.7 frames (3.33s) per act!
        frames_per_block = total_frames / max(num_blocks, 1)

        segments: list[SceneSegment] = []
        current_seg: SceneSegment | None = None

        for b_idx, block in enumerate(blocks):
            b_start = b_idx * frames_per_block
            b_end = (b_idx + 1) * frames_per_block - 1.0

            # Peek whether this block starts a brand-new environment or explicit transition
            explicit_bg: tuple[str, str | None] | None = None
            for fb in block.frames:
                for cmd in fb.commands:
                    if cmd.action == "set_background":
                        explicit_bg = (
                            str(cmd.params["theme"]),
                            cmd.params.get("file") if isinstance(cmd.params.get("file"), str) else None,
                        )
                        break
                if explicit_bg is not None:
                    break

            starts_new_segment = (
                current_seg is None
                or block.transition is not None
                or (
                    explicit_bg is not None
                    and explicit_bg[0].lower() != current_seg.background.lower()
                )
            )

            if starts_new_segment:
                bg_name = explicit_bg[0] if explicit_bg else (current_seg.background if current_seg else "space galaxy")
                bg_file = explicit_bg[1] if explicit_bg else (current_seg.background_file if current_seg else None)
                if current_seg is not None and block.transition is not None:
                    if is_mograph_scene(bg_name) or is_mograph_scene(current_seg.background):
                        trans_len = max(14, min(20, int(round(frames_per_block * 0.24))))
                    else:
                        trans_len = min(42, max(18, int(round(frames_per_block * 0.46))))
                else:
                    trans_len = 0
                current_seg = SceneSegment(
                    background=bg_name,
                    background_file=bg_file,
                    nominal_start=b_start,
                    nominal_end=b_end,
                    render_start=max(0, int(round(b_start - trans_len * 0.5))),
                    render_end=int(round(b_end)),
                    transition_in=block.transition,
                    transition_frames=trans_len,
                )
                # Seed subtle gentle zoom drift so even unpanned scenes breathe alive
                current_seg.cam_zoom.add_point(current_seg.render_start, 1.0)
                if current_seg is not None and segments:
                    # Extend outgoing segment right through the full dissolve window so BOTH animate live!
                    segments[-1].render_end = max(
                        segments[-1].render_end,
                        current_seg.render_start + trans_len,
                    )
                segments.append(current_seg)
            else:
                assert current_seg is not None
                current_seg.nominal_end = b_end
                current_seg.render_end = int(round(b_end))

            assert current_seg is not None

            for fb in sorted(block.frames, key=lambda x: x.index):
                norm_u = (max(1, min(fps, fb.index)) - 1.0) / max(fps - 1, 1)
                mapped_f = b_start + norm_u * max(b_end - b_start, 1.0)
                for cmd in fb.commands:
                    self._compile_command(
                        current_seg,
                        cmd,
                        program,
                        frame_pos=mapped_f,
                        block_start=b_start,
                        block_end=b_end,
                    )

        # Ensure final segment smoothly glides right to `total_frames - 1` with subtle zoom cushion
        for seg in segments:
            seg_end = float(seg.render_end)
            # Extend any continuous camera motion smoothly across the ENTIRE segment window
            if len(seg.cam_pan_x.keyframes) == 1:
                seg.cam_pan_x.add_point(seg_end, seg.cam_pan_x.last_value())
            elif len(seg.cam_pan_x.keyframes) >= 2 and seg.cam_pan_x.keyframes[-1][0] < seg_end:
                # Extrapolate gentle continuation so camera never hits a sudden brick wall!
                f0, v0 = seg.cam_pan_x.keyframes[-2]
                f1, v1 = seg.cam_pan_x.keyframes[-1]
                rate = (v1 - v0) / max(f1 - f0, 1.0)
                seg.cam_pan_x.add_point(
                    seg_end, max(-0.92, min(0.92, v1 + rate * (seg_end - f1) * 0.65))
                )

            if len(seg.cam_tilt_y.keyframes) >= 2 and seg.cam_tilt_y.keyframes[-1][0] < seg_end:
                f0, v0 = seg.cam_tilt_y.keyframes[-2]
                f1, v1 = seg.cam_tilt_y.keyframes[-1]
                rate = (v1 - v0) / max(f1 - f0, 1.0)
                seg.cam_tilt_y.add_point(
                    seg_end, max(-0.92, min(0.92, v1 + rate * (seg_end - f1) * 0.68))
                )

            seg.cam_zoom.add_point(seg_end, max(seg.cam_zoom.last_value(), 1.075))

            # Extend rotating planets cleanly through outgoing transition overlap!
            for otrack in seg.objects.values():
                if len(otrack.rotation.keyframes) >= 2 and otrack.rotation.keyframes[-1][0] < seg_end:
                    f0, r0 = otrack.rotation.keyframes[-2]
                    f1, r1 = otrack.rotation.keyframes[-1]
                    rate = (r1 - r0) / max(f1 - f0, 1.0)
                    otrack.rotation.add_point(seg_end, r1 + rate * (seg_end - f1))

        segments[-1].render_end = max(segments[-1].render_end, total_frames - 1)
        return segments

    @staticmethod
    def _compile_command(
        seg: SceneSegment,
        cmd: Command,
        program: VideoProgram,
        *,
        frame_pos: float,
        block_start: float,
        block_end: float,
    ) -> None:
        act = cmd.action
        p = cmd.params
        fps = program.fps

        if act == "set_background":
            seg.background = str(p["theme"])
            if isinstance(p.get("file"), str):
                seg.background_file = str(p["file"])

        elif act == "set_background_time":
            seg.time_of_day = str(p["time"])

        elif act == "set_background_weather":
            seg.weather = str(p["weather"])

        elif act == "set_background_season":
            seg.season = str(p["season"])

        elif act == "set_background_blur":
            if not seg.bg_blur.keyframes:
                seg.bg_blur.add_point(float(seg.render_start), 0.0)
            seg.bg_blur.add_point(frame_pos, max(0.0, float(p["blur"])))

        elif act == "place_object":
            name = str(p["name"])
            ax, ay = anchor_normalized(str(p["position"]))
            # If placed at the opening keyframe of a transitioned scene, seed right at
            # `seg.render_start` so the incoming hero rig is already live inside the wipe/portal!
            at_scene_open = abs(frame_pos - seg.nominal_start) < 1.5
            appear_f = float(seg.render_start) if at_scene_open else frame_pos
            otrack = ObjectTrack(
                name=name,
                appear_frame=appear_f,
                file=str(p["file"]) if isinstance(p.get("file"), str) else None,
            )
            otrack.x.add_point(appear_f, ax)
            otrack.y.add_point(appear_f, ay)
            otrack.scale.add_point(appear_f, 1.0)
            otrack.rotation.add_point(appear_f, 0.0)
            otrack.opacity.add_point(appear_f, 1.0)
            otrack.glow.add_point(appear_f, 0.0)
            seg.objects[name] = otrack

        elif act == "remove_object":
            name = str(p["name"])
            if name in seg.objects:
                seg.objects[name].remove_frame = frame_pos

        elif act == "flip_object":
            name = str(p["name"])
            if name in seg.objects:
                axis = str(p.get("axis", "horizontally")).lower()
                if "vert" in axis:
                    seg.objects[name].flip_v = not seg.objects[name].flip_v
                else:
                    seg.objects[name].flip_h = not seg.objects[name].flip_h

        elif act == "scale_object":
            name = str(p["name"])
            if name in seg.objects:
                target_sc = SIZE_SCALE.get(str(p["size"]).lower(), 1.0)
                seg.objects[name].scale.add_point(frame_pos, target_sc)

        elif act == "rotate_object":
            name = str(p["name"])
            if name in seg.objects:
                ot = seg.objects[name]
                start_f = ot.appear_frame
                if not ot.rotation.keyframes:
                    ot.rotation.add_point(start_f, 0.0)
                target_f = max(frame_pos, block_end)
                ot.rotation.add_point(target_f, ot.rotation.last_value() + float(p["degrees"]))

        elif act == "move_object":
            # CRITICAL SEMANTIC FIX: `move object` animates smoothly FROM current
            # keyframe position TO target across `[frame_pos, block_end]`!
            name = str(p["name"])
            if name in seg.objects:
                ot = seg.objects[name]
                cur_x = ot.x.evaluate(frame_pos)
                cur_y = ot.y.evaluate(frame_pos)
                ot.x.add_point(frame_pos, cur_x)
                ot.y.add_point(frame_pos, cur_y)

                tx, ty = anchor_normalized(str(p["position"]))
                px = p.get("pixels")
                if isinstance(px, (int, float)):
                    dx, dy = tx - cur_x, ty - cur_y
                    norm_len = math.hypot(dx, dy) or 1.0
                    # Natural scenic arc distance across the full 1080p frame
                    dist_norm = max(float(px) / max(program.width, 1), 0.46)
                    dest_x = max(0.06, min(0.94, cur_x + (dx / norm_len) * dist_norm))
                    dest_y = max(0.08, min(0.90, cur_y + (dy / norm_len) * dist_norm * 0.85))
                else:
                    dest_x, dest_y = tx, ty

                # Single unbroken horizontal span -> pure smooth aerodynamic glide without mid-flight dip!
                ot.x.add_point(block_end, dest_x)
                ot.y.add_point(0.5 * (frame_pos + block_end), max(0.06, 0.5 * (cur_y + dest_y) - 0.024))
                ot.y.add_point(block_end, dest_y)

        elif act == "set_object_color":
            name = str(p["name"])
            if name in seg.objects:
                seg.objects[name].color = resolve_rgb(str(p["color"]))

        elif act == "set_object_opacity":
            name = str(p["name"])
            if name in seg.objects:
                seg.objects[name].opacity.add_point(
                    frame_pos, max(0.0, min(1.0, float(p["opacity"])))
                )

        elif act == "set_light_color":
            r, g, b = resolve_rgb(str(p["color"]))
            seg.light_r.add_point(frame_pos, float(r))
            seg.light_g.add_point(frame_pos, float(g))
            seg.light_b.add_point(frame_pos, float(b))

        elif act == "set_light_brightness":
            seg.light_brightness.add_point(
                frame_pos, max(0.20, min(1.45, float(p["brightness"])))
            )

        elif act == "camera_pan":
            direction = str(p["direction"]).lower()
            delta = SPEED_DELTA.get(str(p["speed"]).lower(), 0.5)
            signed = -delta if direction == "left" else delta
            start_f = float(seg.render_start) if not seg.cam_pan_x.keyframes else min(frame_pos, block_start)
            cur_v = seg.cam_pan_x.evaluate(start_f)
            seg.cam_pan_x.add_point(start_f, cur_v)
            seg.cam_pan_x.add_point(block_end, max(-0.92, min(0.92, cur_v + signed)))

        elif act == "camera_tilt":
            direction = str(p["direction"]).lower()
            delta = SPEED_DELTA.get(str(p["speed"]).lower(), 0.5)
            signed = delta if direction == "down" else -delta
            # Start gentle tilt right from scene start -> glide continuously through block_end!
            start_f = float(seg.render_start) if not seg.cam_tilt_y.keyframes else min(frame_pos, block_start)
            cur_v = -0.16 if not seg.cam_tilt_y.keyframes and direction == "down" else seg.cam_tilt_y.evaluate(start_f)
            seg.cam_tilt_y.add_point(start_f, cur_v)
            seg.cam_tilt_y.add_point(block_end, max(-0.92, min(0.92, cur_v + signed)))

        elif act == "camera_shake":
            lvl = {"low": 3.0, "small": 3.0, "medium": 7.5, "high": 14.0, "heavy": 18.0}.get(
                str(p.get("intensity", "medium")).lower(), 7.5
            )
            seg.cam_shake.add_point(frame_pos, lvl)

        elif act == "set_camera_zoom":
            z_str = str(p["zoom"]).lower()
            z_val = 1.0
            if "extreme" in z_str:
                z_val = 1.30
            elif "close" in z_str:
                z_val = 1.16
            seg.cam_zoom.add_point(frame_pos, z_val)

        elif act == "place_text":
            tx, ty = anchor_normalized(str(p["position"]))
            # If text is placed right at the last keyframe of an act, start fading gently a bit earlier
            graceful_start = min(frame_pos, block_end - max(fps * 0.65, 12.0))
            seg.texts.append(
                TextEvent(
                    content=str(p["content"]),
                    x=tx,
                    y=ty,
                    start_frame=max(float(seg.nominal_start), graceful_start),
                    fade_in_frames=float(max(18, fps)),
                )
            )

        elif act == "set_text_color":
            if seg.texts:
                seg.texts[-1].color = resolve_rgb(str(p["color"]))

        elif act == "set_text_size":
            if seg.texts:
                raw_sz = str(p["size"]).lower()
                if raw_sz in ("small", "medium", "large"):
                    seg.texts[-1].size = {"small": 32.0, "medium": 52.0, "large": 76.0}[raw_sz]
                else:
                    seg.texts[-1].size = float(p["size"])

        elif act == "fade_text_in":
            if seg.texts:
                seg.texts[-1].fade_in_frames = float(max(int(p["frames"]), int(fps * 0.85)))

        elif act == "fade_text_out":
            if seg.texts:
                seg.texts[-1].fade_out_frames = float(max(1, int(p["frames"])))

        elif act == "add_text_effect":
            if seg.texts and str(p["effect"]).lower() == "glow":
                seg.texts[-1].glow = 1.0
                seg.texts[-1].glow_color = resolve_rgb(str(p["color"]))

        elif act == "add_effect":
            target = p.get("object")
            effect = str(p["effect"]).lower()
            lvl = float(p.get("level", 1.0))
            if isinstance(target, str) and target in seg.objects:
                ot = seg.objects[target]
                if effect == "glow":
                    ot.glow_color = resolve_rgb(str(p["color"]))
                    ot.glow.add_point(frame_pos, lvl)
                elif effect == "shadow":
                    ot.shadow = lvl
            else:
                if effect not in seg.scene_effects:
                    seg.scene_effects[effect] = ScalarTrack(0.0)
                    seg.scene_effects[effect].add_point(float(seg.render_start), 0.0)
                seg.scene_effects[effect].add_point(frame_pos, lvl)

        elif act == "narrate":
            seg.narration_text = str(p["text"])
            seg.narration_frame = min(frame_pos, block_end - max(fps * 1.1, 20.0))

    # ── Render Loop with Live Overlapping Dual-Scene Linear-Light Transitions ─

    def render(self, program: VideoProgram, output_path: str | Path) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fps = program.fps
        total_frames = max(1, int(round(program.total_duration * fps)))
        assets = StudioAssetBank(program.width, program.height, self.base_dir)
        segments = self.compile_segments(program, total_frames)

        # Sample primary timeline for audio cue sync
        sampled_for_audio = [
            self._active_primary_segment(segments, f).sample(float(f), fps, total_frames)
            for f in range(total_frames)
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "soundtrack.wav"
            has_mograph = any(is_mograph_scene(seg.background) for seg in segments)
            if has_mograph:
                trans_times = [
                    (seg.render_start + seg.transition_frames * 0.5) / max(fps, 1)
                    for seg in segments[1:]
                    if seg.transition_frames > 0
                ]
                write_mograph_electronic_score(
                    wav_path=wav_path,
                    duration_sec=total_frames / fps,
                    transition_times_sec=trans_times,
                )
            else:
                self._write_cinematic_score(
                    wav_path=wav_path,
                    duration_sec=total_frames / fps,
                    fps=fps,
                    timeline=sampled_for_audio,
                )

            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{program.width}x{program.height}",
                "-pix_fmt", "rgb24",
                "-r", str(fps),
                "-i", "-",
                "-i", str(wav_path),
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                str(out_path),
            ]

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            assert proc.stdin is not None

            try:
                for f_idx in range(total_frames):
                    rgb = self._render_composited_frame(
                        program=program,
                        assets=assets,
                        segments=segments,
                        global_frame=f_idx,
                        total_frames=total_frames,
                    )
                    proc.stdin.write(rgb.tobytes())
            finally:
                proc.stdin.close()
                stderr_out = ""
                if proc.stderr is not None:
                    stderr_out = proc.stderr.read().decode("utf-8", errors="replace")
                    proc.stderr.close()
                ret = proc.wait()
                if ret != 0:
                    raise RuntimeError(f"FFmpeg encoding failed (exit {ret}): {stderr_out}")

        return out_path

    @staticmethod
    def _active_primary_segment(segments: list[SceneSegment], f: int) -> SceneSegment:
        chosen = segments[0]
        for seg in segments:
            if f >= seg.render_start:
                chosen = seg
        return chosen

    def _render_composited_frame(
        self,
        *,
        program: VideoProgram,
        assets: StudioAssetBank,
        segments: list[SceneSegment],
        global_frame: int,
        total_frames: int,
    ) -> np.ndarray:
        fps = program.fps

        # Check if we are inside a live dual-stream transition overlap between two segments!
        blended_rgb: np.ndarray | None = None
        primary_state: SceneState | None = None

        for idx in range(1, len(segments)):
            inc = segments[idx]
            outg = segments[idx - 1]
            t_start = inc.render_start
            t_end = inc.render_start + inc.transition_frames
            if inc.transition_frames > 0 and t_start <= global_frame < t_end:
                raw_alpha = (global_frame - t_start + 1.0) / (inc.transition_frames + 1.0)
                st_out = outg.sample(float(global_frame), fps, total_frames)
                st_in = inc.sample(float(global_frame), fps, total_frames)
                rgb_out = self._render_single_frame(
                    program=program,
                    assets=assets,
                    state=st_out,
                    global_frame=global_frame,
                    total_frames=total_frames,
                )
                rgb_in = self._render_single_frame(
                    program=program,
                    assets=assets,
                    state=st_in,
                    global_frame=global_frame,
                    total_frames=total_frames,
                )
                if is_mograph_scene(st_out.background) or is_mograph_scene(st_in.background):
                    blended_rgb = blend_mograph_transition(
                        rgb_out, rgb_in, inc.transition_in or "slide", raw_alpha
                    )
                else:
                    blended_rgb = self._blend_transition_linear_light(
                        rgb_out, rgb_in, inc.transition_in or "fade", raw_alpha
                    )
                primary_state = st_in
                break

        if blended_rgb is None:
            seg = self._active_primary_segment(segments, global_frame)
            primary_state = seg.sample(float(global_frame), fps, total_frames)
            blended_rgb = self._render_single_frame(
                program=program,
                assets=assets,
                state=primary_state,
                global_frame=global_frame,
                total_frames=total_frames,
            )

        # Gentle cinema fade-from-black opening (15 frames) & fade-to-black closing (18 frames)
        if total_frames >= 48:
            open_gain = ease_c2(min(1.0, (global_frame + 1.0) / 15.0))
            close_gain = ease_c2(min(1.0, (total_frames - global_frame) / 18.0))
            envelope = open_gain * close_gain
            if envelope < 0.999:
                blended_rgb = np.clip(
                    blended_rgb.astype(np.float32) * envelope, 0.0, 255.0
                ).astype(np.uint8)

        if self.show_hud and primary_state is not None:
            blended_rgb = self._draw_frame_hud(
                blended_rgb, program, primary_state, global_frame, total_frames
            )

        return blended_rgb

    def _render_mograph_frame(
        self,
        *,
        program: VideoProgram,
        assets: StudioAssetBank,
        state: SceneState,
        global_frame: int,
    ) -> np.ndarray:
        """Pure sub-pixel analytical SDF vector motion-graphics frame renderer."""
        w, h = program.width, program.height
        fps = program.fps
        time_s = global_frame / max(fps, 1)
        mg = assets.mograph
        light_stage = is_light_mograph_scene(state.background)

        canvas = mg.render_stage_canvas(
            theme=state.background,
            global_frame=global_frame,
            fps=fps,
            cam_pan_x=state.cam_pan_x,
            cam_tilt_y=state.cam_tilt_y,
            cam_zoom=state.cam_zoom,
        )

        cam_px_shift = -state.cam_pan_x * (w * 0.025)
        cam_py_shift = -state.cam_tilt_y * (h * 0.025)

        for obj in state.objects.values():
            if obj.opacity <= 0.005:
                continue
            fx_x = obj.x * w + cam_px_shift
            fy_y = obj.y * h + cam_py_shift
            low = obj.name.strip().lower()

            if "physics lanes" in low:
                mg.render_physics_lanes(
                    canvas,
                    local_age_s=obj.local_age_s,
                    fps=fps,
                    opacity=obj.opacity,
                )
            elif "orbital rig" in low:
                mg.render_orbital_rig(
                    canvas,
                    cx=fx_x,
                    cy=fy_y,
                    scale=obj.scale,
                    rotation_deg=obj.rotation,
                    local_age_s=obj.local_age_s,
                    global_time_s=time_s,
                    opacity=obj.opacity,
                )
            elif "kinetic cascade" in low:
                mg.render_kinetic_cascade(
                    canvas,
                    cx=fx_x,
                    cy=fy_y,
                    scale=obj.scale,
                    local_age_s=obj.local_age_s,
                    global_time_s=time_s,
                    opacity=obj.opacity,
                )
            elif "equalizer" in low or "waveform" in low:
                mg.render_audio_equalizer(
                    canvas,
                    cx=fx_x,
                    cy=fy_y,
                    scale=obj.scale,
                    local_age_s=obj.local_age_s,
                    global_time_s=time_s,
                    opacity=obj.opacity,
                )
            else:
                mg.render_atomic_vector_object(
                    canvas,
                    name=obj.name,
                    cx=fx_x,
                    cy=fy_y,
                    vx_px=obj.vx * w,
                    vy_px=obj.vy * h,
                    scale=obj.scale,
                    rotation_deg=obj.rotation,
                    opacity=obj.opacity,
                    color=obj.color,
                    glow=max(obj.glow, 0.35),
                )

        frame_img = Image.fromarray(np.clip(canvas, 0.0, 255.0).astype(np.uint8), mode="RGB").convert("RGBA")

        for tev, alpha, _ in state.texts:
            if alpha > 0.008:
                raw_prog = min(
                    1.0,
                    max(0.0, (global_frame - tev.start_frame + 1.0) / max(1.0, tev.fade_in_frames)),
                )
                render_kinetic_text_layer(
                    frame_img,
                    text=tev.content,
                    cx=tev.x * w,
                    cy=tev.y * h,
                    font_px=int(round(tev.size * (h / 1080.0))),
                    color=tev.color,
                    progress=raw_prog * (alpha if tev.fade_out_frames > 0 else 1.0),
                    light_stage=light_stage,
                    accent_color=tev.glow_color if tev.glow > 0.01 else None,
                )

        if state.narration and state.narration_alpha > 0.015:
            self._draw_narration_bar(
                frame_img=frame_img,
                narration=state.narration,
                alpha=state.narration_alpha,
            )

        out_f32 = np.asarray(frame_img.convert("RGB"), dtype=np.float32)
        # Subtle 0.35-LSB anti-banding chroma dither on smooth studio gradients
        dither = np.random.default_rng(30000 + global_frame).uniform(-0.35, 0.35, size=(h, w, 1)).astype(np.float32)
        return np.clip(out_f32 + dither, 0.0, 255.0).astype(np.uint8)

    def _render_single_frame(
        self,
        *,
        program: VideoProgram,
        assets: StudioAssetBank,
        state: SceneState,
        global_frame: int,
        total_frames: int,
    ) -> np.ndarray:
        if is_mograph_scene(state.background):
            return self._render_mograph_frame(
                program=program,
                assets=assets,
                state=state,
                global_frame=global_frame,
            )
        w, h = program.width, program.height
        time_s = global_frame / max(program.fps, 1)

        bg_canvas, depth_canvas = assets.get_scene_and_depth(
            state.background, state.background_file
        )

        shake_xy = (0.0, 0.0)
        if state.cam_shake > 0.05:
            shake_xy = (
                math.sin(global_frame * 1.9) * state.cam_shake,
                math.cos(global_frame * 2.5) * state.cam_shake * 0.75,
            )

        scene_f32 = assets.sample_camera_subpixel(
            bg_rgb=bg_canvas,
            depth=depth_canvas,
            pan_x=state.cam_pan_x,
            tilt_y=state.cam_tilt_y,
            zoom=state.cam_zoom,
            shake_px=shake_xy,
        )

        if state.background_blur > 0.1:
            scene_f32 = cv2.GaussianBlur(scene_f32, (0, 0), sigmaX=state.background_blur)

        bg_low = state.background.lower()
        if "space" in bg_low or "galaxy" in bg_low:
            scene_f32 = self._apply_space_scintillation(scene_f32, assets, time_s)
        elif "mountain" in bg_low or "himalaya" in bg_low:
            scene_f32 = self._apply_mountain_atmosphere(scene_f32, assets, state, time_s)

        tod_rgb, tod_exp = TIME_COLOR_GRADE.get(state.time_of_day.lower(), ((1.0, 1.0, 1.0), 1.0))
        light_rgb = np.array(state.light_color, dtype=np.float32) / 255.0
        gain = (
            np.array(tod_rgb, dtype=np.float32)
            * (0.76 + 0.24 * light_rgb)
            * tod_exp
            * (0.74 + 0.42 * state.light_brightness)
        )
        lin = (scene_f32 / 255.0) ** 2.2 * gain[None, None, :]
        tonemapped = (aces_filmic_tonemap(lin * 1.18) ** (1.0 / 2.2)) * 255.0

        tonemapped = self._apply_weather_and_fx(tonemapped, assets, state, time_s)
        tonemapped *= assets._vignette

        frame_img = Image.fromarray(np.clip(tonemapped, 0.0, 255.0).astype(np.uint8), mode="RGB").convert("RGBA")

        cam_px_shift = -state.cam_pan_x * (w * 0.036)
        cam_py_shift = -state.cam_tilt_y * (h * 0.036)

        for obj in state.objects.values():
            if obj.opacity <= 0.005:
                continue
            fx_x = obj.x * w + cam_px_shift
            fy_y = obj.y * h + cam_py_shift
            name_low = obj.name.lower()

            if obj.file:
                cand = Path(obj.file) if Path(obj.file).is_absolute() else (self.base_dir / obj.file)
                if cand.is_file():
                    sprite = Image.open(cand).convert("RGBA")
                    td = max(32, int(min(w, h) * 0.30 * obj.scale))
                    sprite.thumbnail((td, td), Image.Resampling.LANCZOS)
                else:
                    sprite = self._render_badge_object(obj, min(w, h))
            elif "earth" in name_low or "planet" in name_low:
                diameter = max(48, int(min(w, h) * 0.42 * obj.scale * state.cam_zoom))
                sprite = assets.render_earth_rgba(
                    diameter=diameter,
                    rotation_deg=obj.rotation + time_s * 5.2,
                    glow=obj.glow,
                    glow_color=obj.glow_color,
                )
            elif "eagle" in name_low or "bird" in name_low:
                span = max(40, int(min(w, h) * 0.25 * obj.scale))
                # Gentle natural banking angle aligned smoothly with flight heading!
                heading_deg = math.degrees(math.atan2(-obj.vy, max(abs(obj.vx), 1e-4))) * 0.35
                bank = max(-14.0, min(14.0, heading_deg - 4.0 + math.sin(obj.local_age_s * 1.1) * 2.5))
                sprite = assets.render_eagle_rgba(
                    span_px=span,
                    flight_age_s=obj.local_age_s,
                    bank_deg=bank,
                    opacity=obj.opacity,
                    shadow=obj.shadow,
                )
            else:
                sprite = self._render_badge_object(obj, min(w, h))

            if obj.flip_h:
                sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if obj.flip_v:
                sprite = sprite.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            self._alpha_composite_subpixel(frame_img, sprite, fx_x, fy_y)

        for tev, alpha, rise_px in state.texts:
            if alpha > 0.01:
                self._draw_glow_text(
                    frame_img=frame_img,
                    text=tev.content,
                    cx=int(tev.x * w),
                    cy=int(round(tev.y * h + rise_px)),
                    font_px=int(tev.size * (h / 1080.0)),
                    color=tev.color,
                    alpha=alpha,
                    glow=tev.glow,
                    glow_color=tev.glow_color,
                    serif=True,
                )

        if state.narration and state.narration_alpha > 0.015:
            self._draw_narration_bar(
                frame_img=frame_img,
                narration=f'“{state.narration}”',
                alpha=state.narration_alpha,
            )

        final_rgb = np.asarray(frame_img.convert("RGB"), dtype=np.float32)
        grain_rng = np.random.default_rng(10000 + global_frame)
        grain = grain_rng.normal(0.0, 0.50, size=(h // 2, w // 2, 1)).astype(np.float32)
        grain_full = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        return np.clip(final_rgb + grain_full, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _alpha_composite_subpixel(
        dest_img: Image.Image, sprite: Image.Image, center_x: float, center_y: float
    ) -> None:
        top_left_x = center_x - sprite.width * 0.5
        top_left_y = center_y - sprite.height * 0.5
        ix, iy = int(math.floor(top_left_x)), int(math.floor(top_left_y))
        fx, fy = top_left_x - ix, top_left_y - iy

        arr = np.asarray(sprite, dtype=np.float32)
        padded = np.zeros((sprite.height + 2, sprite.width + 2, 4), dtype=np.float32)
        padded[:-2, :-2] = arr
        M = np.float32([[1.0, 0.0, fx], [0.0, 1.0, fy]])
        shifted = cv2.warpAffine(
            padded,
            M,
            (sprite.width + 2, sprite.height + 2),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        sub_sprite = Image.fromarray(np.clip(shifted, 0.0, 255.0).astype(np.uint8), mode="RGBA")
        dest_img.alpha_composite(sub_sprite, dest=(ix, iy))

    @staticmethod
    def _apply_space_scintillation(
        rgb: np.ndarray, assets: StudioAssetBank, time_s: float
    ) -> np.ndarray:
        out = rgb.copy()
        h, w, _ = out.shape
        for sx, sy, freq, phase in assets._stars_twinkle:
            x, y = int(sx), int(sy)
            if 1 <= x < w - 1 and 1 <= y < h - 1:
                pulse = 0.5 + 0.5 * math.sin(time_s * freq + phase)
                boost = pulse * 125.0
                out[y, x] = np.minimum(255.0, out[y, x] + boost)
                out[y - 1 : y + 2, x - 1 : x + 2] = np.minimum(
                    255.0, out[y - 1 : y + 2, x - 1 : x + 2] + boost * 0.22
                )
        return out

    @staticmethod
    def _apply_mountain_atmosphere(
        rgb: np.ndarray,
        assets: StudioAssetBank,
        state: SceneState,
        time_s: float,
    ) -> np.ndarray:
        out = rgb.copy()
        h, w, _ = out.shape

        # Sub-pixel continuous horizontal advection of periodic valley mist (ZERO integer jumps!)
        shift_px = (time_s * w * 0.018) % w
        map_x = np.mod(assets._screen_x - shift_px, float(w)).astype(np.float32)
        mist_slice = cv2.remap(
            assets._mist_noise,
            map_x,
            assets._screen_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        v_mask = np.clip((assets._screen_y / h - 0.54) / 0.40, 0.0, 1.0)
        mist_amount = np.clip((mist_slice - 0.46) * 1.40, 0.0, 0.40) * v_mask
        mist_color = np.array([216.0, 224.0, 240.0], dtype=np.float32)
        out = out * (1.0 - mist_amount[..., None] * 0.52) + mist_color * (mist_amount[..., None] * 0.52)

        sun_u = 0.365 - state.cam_pan_x * 0.055
        sun_v = 0.285 - state.cam_tilt_y * 0.055
        du = (assets._screen_x / w - sun_u) * (w / max(h, 1))
        dv = assets._screen_y / h - sun_v
        r2 = du * du + dv * dv
        shimmer = 0.94 + 0.06 * math.sin(time_s * 1.4)
        warm_bloom = np.exp(-r2 / 0.055) * 0.22 * shimmer
        anamorphic = np.exp(-(dv * dv) / 0.00055) * np.exp(-(du * du) / 0.16) * 0.14 * shimmer
        out += (warm_bloom + anamorphic)[..., None] * np.array([255.0, 198.0, 118.0], dtype=np.float32)

        overlay = np.zeros_like(out)
        for x0, y0, spd, rad, ph in assets._particles:
            px = ((x0 + time_s * 0.024 * spd) % 1.0) * (w - 1)
            py = (y0 + math.sin(time_s * 1.1 + ph) * 0.007) * (h - 1)
            cv2.circle(
                overlay,
                (int(round(px)), int(round(py))),
                max(1, int(round(rad))),
                (255.0, 232.0, 198.0),
                -1,
                lineType=cv2.LINE_AA,
            )
        return np.clip(out + overlay * 0.26, 0.0, 255.0)

    @staticmethod
    def _apply_weather_and_fx(
        rgb: np.ndarray,
        assets: StudioAssetBank,
        state: SceneState,
        time_s: float,
    ) -> np.ndarray:
        out = rgb
        weather = state.weather.lower()
        fx = state.scene_effects

        if "snow" in weather or "blizzard" in weather or fx.get("snow", 0.0) > 0:
            h, w, _ = out.shape
            snow_layer = np.zeros_like(out)
            for x0, y0, spd, rad, ph in assets._particles:
                px = ((x0 + math.sin(time_s + ph) * 0.03) % 1.0) * (w - 1)
                py = ((y0 + time_s * 0.18 * spd) % 1.0) * (h - 1)
                cv2.circle(snow_layer, (int(px), int(py)), max(1, int(rad + 1)), (255, 255, 255), -1, cv2.LINE_AA)
            out = out + snow_layer * 0.65

        if "rain" in weather or "storm" in weather or fx.get("rain", 0.0) > 0:
            h, w, _ = out.shape
            rain_layer = np.zeros_like(out)
            for x0, y0, spd, _, _ in assets._particles:
                px = int(((x0 + time_s * 0.12) % 1.0) * (w - 1))
                py = int(((y0 + time_s * 0.75 * spd) % 1.0) * (h - 1))
                cv2.line(rain_layer, (px, py), (px - 4, py + 18), (185, 210, 240), 1, cv2.LINE_AA)
            out = out * 0.88 + rain_layer * 0.55

        if "fog" in weather or fx.get("smoke", 0.0) > 0:
            out = out * 0.68 + 72.0

        if fx.get("black and white", 0.0) > 0:
            luma = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])[..., None]
            out = np.repeat(luma, 3, axis=-1)

        if fx.get("vintage", 0.0) > 0:
            sepia = np.array(
                [[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]],
                dtype=np.float32,
            )
            out = out @ sepia.T

        if fx.get("glitch", 0.0) > 0:
            shift = int(6 + 6 * math.sin(time_s * 19.0))
            out = out.copy()
            out[..., 0] = np.roll(out[..., 0], shift, axis=1)
            out[..., 2] = np.roll(out[..., 2], -shift, axis=1)

        return out

    @staticmethod
    def _blend_transition_linear_light(
        prev_rgb: np.ndarray,
        curr_rgb: np.ndarray,
        trans_type: str,
        raw_alpha: float,
    ) -> np.ndarray:
        """
        True Cinema Linear-Light Cross-Dissolve (`gamma = 2.2` + $C^2$ ease):
        Prevents mid-dissolve muddiness so starlight & sunrise glow melt smoothly!
        """
        a = ease_c2(raw_alpha)
        h, w, _ = curr_rgb.shape
        t = trans_type.lower()

        if "cut" in t:
            return curr_rgb

        if "wipe" in t:
            # Soft feathered optical cinema wipe (not a harsh 1px razor line!)
            x_norm = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
            feather = 0.09
            mask = np.clip((a * (1.0 + 2 * feather) - feather - x_norm) / feather, 0.0, 1.0)
            p_lin = (prev_rgb.astype(np.float32) / 255.0) ** 2.2
            c_lin = (curr_rgb.astype(np.float32) / 255.0) ** 2.2
            mixed = ((p_lin * (1.0 - mask) + c_lin * mask) ** (1.0 / 2.2)) * 255.0
            return np.clip(mixed, 0.0, 255.0).astype(np.uint8)

        if "slide" in t:
            shift_f = w * (1.0 - a)
            M_prev = np.float32([[1.0, 0.0, -w * a * 0.35], [0.0, 1.0, 0.0]])
            M_curr = np.float32([[1.0, 0.0, shift_f], [0.0, 1.0, 0.0]])
            slid_prev = cv2.warpAffine(prev_rgb, M_prev, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            slid_curr = cv2.warpAffine(curr_rgb, M_curr, (w, h), borderMode=cv2.BORDER_CONSTANT)
            mask = (np.arange(w)[None, :, None] >= shift_f).astype(np.float32)
            return np.clip(slid_prev * (1.0 - mask) + slid_curr * mask, 0.0, 255.0).astype(np.uint8)

        # Default: Optical push-zoom + linear-light cross-dissolve (`fade` / `dissolve` / `zoom`)
        out_zoom = 1.0 + 0.10 * a
        in_zoom = 1.06 - 0.06 * a
        M_out = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.0, out_zoom)
        M_in = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.0, in_zoom)
        warped_prev = cv2.warpAffine(prev_rgb, M_out, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        warped_curr = cv2.warpAffine(curr_rgb, M_in, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        p_lin = (warped_prev.astype(np.float32) / 255.0) ** 2.2
        c_lin = (warped_curr.astype(np.float32) / 255.0) ** 2.2
        # Subtle warm optical bloom at midpoint of transition
        bloom = math.sin(a * math.pi) * 0.045
        mixed_lin = p_lin * (1.0 - a) + c_lin * a + bloom
        mixed_srgb = (np.clip(mixed_lin, 0.0, 1.0) ** (1.0 / 2.2)) * 255.0
        return np.clip(mixed_srgb, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _render_badge_object(obj: ObjectState, ref_dim: int) -> Image.Image:
        radius = max(24, int(ref_dim * 0.11 * obj.scale))
        size = radius * 3
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        c = size // 2
        draw.ellipse(
            (c - radius, c - radius, c + radius, c + radius),
            fill=(88, 152, 236, int(235 * obj.opacity)),
            outline=(255, 255, 255, int(220 * obj.opacity)),
            width=3,
        )
        font = load_font(max(14, radius // 3))
        bbox = draw.textbbox((0, 0), obj.name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((c - tw / 2, c - th / 2), obj.name, font=font, fill=(255, 255, 255, int(245 * obj.opacity)))
        return canvas

    @staticmethod
    def _draw_glow_text(
        *,
        frame_img: Image.Image,
        text: str,
        cx: int,
        cy: int,
        font_px: int,
        color: tuple[int, int, int],
        alpha: float,
        glow: float,
        glow_color: tuple[int, int, int],
        serif: bool = False,
    ) -> None:
        w, h = frame_img.size
        font = load_font(max(18, font_px), serif=serif)
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx, ty = int(cx - tw * 0.5), int(cy - th * 0.5)

        if glow > 0.01:
            glow_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            g_draw = ImageDraw.Draw(glow_img)
            gr, gg, gb = glow_color
            g_draw.text((tx, ty), text, font=font, fill=(gr, gg, gb, int(215 * alpha * glow)))
            glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=max(8.0, font_px * 0.28)))
            frame_img.alpha_composite(glow_img)

        draw.text((tx + 2, ty + 3), text, font=font, fill=(8, 12, 22, int(175 * alpha)))
        r, g, b = color
        draw.text((tx, ty), text, font=font, fill=(r, g, b, int(255 * alpha)))
        frame_img.alpha_composite(overlay)

    @staticmethod
    def _draw_narration_bar(frame_img: Image.Image, narration: str, alpha: float) -> None:
        w, h = frame_img.size
        font = load_font(max(16, int(h * 0.025)), serif=False)
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        bbox = draw.textbbox((0, 0), narration, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = int(w * 0.024), int(h * 0.012)
        cx, cy = w // 2, int(h * 0.932)
        rect = (cx - tw // 2 - pad_x, cy - th // 2 - pad_y, cx + tw // 2 + pad_x, cy + th // 2 + pad_y)
        draw.rounded_rectangle(
            rect,
            radius=max(8, int(h * 0.018)),
            fill=(10, 15, 26, int(185 * alpha)),
            outline=(255, 226, 176, int(95 * alpha)),
            width=1,
        )
        draw.text(
            (cx - tw // 2, cy - th // 2 - 2),
            narration,
            font=font,
            fill=(248, 250, 255, int(248 * alpha)),
        )
        frame_img.alpha_composite(overlay)

    @staticmethod
    def _draw_frame_hud(
        rgb: np.ndarray,
        program: VideoProgram,
        state: SceneState,
        global_frame: int,
        total_frames: int,
    ) -> np.ndarray:
        img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        font = load_font(max(13, int(program.height * 0.018)), mono=True)
        sec_idx = global_frame // program.fps + 1
        local_f = global_frame % program.fps + 1
        hud_line = (
            f"VIDEX ENGINE  |  SEC {sec_idx:02d}  FRAME {local_f:02d}/{program.fps:02d} "
            f"(#{global_frame + 1:03d}/{total_frames:03d})  |  "
            f"PAN {state.cam_pan_x:+.3f}  TILT {state.cam_tilt_y:+.3f}  ZOOM {state.cam_zoom:.3f}x"
        )
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        bbox = draw.textbbox((0, 0), hud_line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rounded_rectangle((22, 20, 46 + tw, 36 + th), radius=8, fill=(8, 12, 20, 168))
        draw.text((34, 27), hud_line, font=font, fill=(144, 226, 255, 235))
        img.alpha_composite(overlay)
        return np.asarray(img.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _write_cinematic_score(
        *,
        wav_path: Path,
        duration_sec: float,
        fps: int,
        timeline: list[SceneState],
    ) -> None:
        sr = 44100
        n_samples = max(1, int(round(duration_sec * sr)))
        t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

        swell = 0.70 + 0.30 * np.sin(2.0 * math.pi * 0.11 * t)
        sunrise_shift = np.clip((t - duration_sec * 0.28) / (duration_sec * 0.18), 0.0, 1.0)
        left = (
            0.26 * np.sin(2.0 * math.pi * 130.81 * t)
            + 0.20 * np.sin(2.0 * math.pi * 196.00 * t + 0.25)
            + 0.14 * np.sin(2.0 * math.pi * 293.66 * t)
            + 0.11 * sunrise_shift * np.sin(2.0 * math.pi * 369.99 * t)
        ) * swell
        right = (
            0.26 * np.sin(2.0 * math.pi * 130.81 * t)
            + 0.20 * np.sin(2.0 * math.pi * 196.00 * t - 0.25)
            + 0.14 * np.sin(2.0 * math.pi * 329.63 * t)
            + 0.11 * sunrise_shift * np.sin(2.0 * math.pi * 493.88 * t)
        ) * swell

        rng = np.random.default_rng(2026)
        raw_air = rng.normal(0.0, 0.024, n_samples).astype(np.float32)
        kernel = np.ones(64, dtype=np.float32) / 64.0
        soft_wind = np.convolve(raw_air, kernel, mode="same") * sunrise_shift
        left += soft_wind
        right += np.roll(soft_wind, 220)

        narrate_frame = next((i for i, st in enumerate(timeline) if st.narration), -1)
        if narrate_frame >= 0:
            cue_time = narrate_frame / max(fps, 1)
            for idx, freq in enumerate((523.25, 659.25, 783.99, 1046.50)):
                onset = cue_time + idx * 0.18
                dt = np.maximum(0.0, t - onset)
                bell_env = (t >= onset).astype(np.float32) * np.exp(-dt * 2.0)
                bell = 0.12 * np.sin(2.0 * math.pi * freq * dt) * bell_env
                left += bell
                right += bell * 0.94

        fade_len = min(int(0.65 * sr), n_samples // 4)
        env = np.ones(n_samples, dtype=np.float32)
        if fade_len > 1:
            ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            env[:fade_len] *= ramp
            env[-fade_len:] *= ramp[::-1]

        stereo = np.stack([left * env, right * env], axis=-1)
        pcm16 = np.clip(stereo * 0.44 * 32767.0, -32767.0, 32767.0).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
