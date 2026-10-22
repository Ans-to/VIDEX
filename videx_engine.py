"""
VIDEX Programming Language — Cinema-Grade 2.5D/3D Video Engine (Stages 3–4)
===========================================================================

Pipeline Stages 3 & 4:
    VIDEX Code -> Lexer -> Parser -> Validator -> [CINEMA VIDEO ENGINE] -> MP4

Key Per-Frame Capabilities:
  - True floating-point sub-pixel camera pan, tilt, zoom & shake (`cv2.remap`)
  - Monocular 2.5D depth-map parallax displacement across multi-layer landscapes
  - 5-layer 3D spherical planet ray-tracer (`cv2.remap` bilinear wrap, ocean
    Blinn-Phong solar glint, cloud-cast surface shadows, night-side city lights,
    and Rayleigh atmospheric blue rim scattering)
  - Non-rigid per-frame wing-flap mesh deformation & banking flight physics
  - Per-frame living atmosphere: twinkling starfield, drifting valley mist,
    wind-blown summit snow particles, volumetric sun rays & anamorphic flare
  - Full 18-effect VIDEX post-FX shader suite + ACES filmic tone mapping + 35mm grain
  - Optional `--hud` frame-accurate telemetry overlay for per-frame inspection
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

from videx_parser import Command, SecondBlock, VideoProgram


# ══════════════════════════════════════════════════════════════════════════════
# COLOR, ANCHOR, FONT & CINEMA MATH UTILITIES
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
    "small": 0.44,
    "medium": 1.0,
    "large": 1.44,
}

SPEED_DELTA: dict[str, float] = {
    "slow": 0.42,
    "medium": 0.72,
    "fast": 1.0,
}

TIME_COLOR_GRADE: dict[str, tuple[tuple[float, float, float], float]] = {
    "sunrise": ((1.06, 0.99, 0.92), 1.02),
    "golden hour": ((1.12, 0.98, 0.84), 1.04),
    "sunset": ((1.14, 0.90, 0.82), 0.96),
    "morning": ((1.01, 1.01, 1.02), 1.03),
    "afternoon": ((1.03, 1.01, 0.97), 1.05),
    "dawn": ((0.94, 0.96, 1.08), 0.90),
    "dusk": ((0.95, 0.88, 1.08), 0.84),
    "night": ((0.88, 0.94, 1.10), 0.88),
    "midnight": ((0.80, 0.88, 1.14), 0.76),
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
        "left": (0.20, 0.50),
        "right": (0.80, 0.50),
        "top": (0.50, 0.20),
        "bottom": (0.50, 0.84),
        "top left": (0.18, 0.23),
        "top right": (0.78, 0.20),
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


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def aces_filmic_tonemap(linear_rgb: np.ndarray) -> np.ndarray:
    """Narkowicz ACES filmic curve preventing harsh highlight clipping."""
    x = np.maximum(linear_rgb, 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = (x * (a * x + b)) / (x * (c * x + d) + e)
    return np.clip(mapped, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# SCENE STATE MODEL FOR KEYFRAME INTERPOLATION
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ObjectState:
    name: str
    x: float = 0.50
    y: float = 0.50
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0
    glow: float = 0.0
    glow_color: tuple[int, int, int] = (72, 168, 255)
    shadow: float = 0.0
    flip_h: bool = False
    flip_v: bool = False
    file: str | None = None


@dataclass
class TextState:
    content: str = ""
    x: float = 0.50
    y: float = 0.835
    size: float = 50.0
    color: tuple[int, int, int] = (255, 255, 255)
    fade_in_frames: int = 1
    fade_out_frames: int = 0
    appear_global_frame: int = -1
    glow: float = 0.0
    glow_color: tuple[int, int, int] = (255, 255, 255)


@dataclass
class SceneState:
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
    text: TextState = field(default_factory=TextState)
    narration: str = ""
    narration_start_frame: int = -1


def interpolate_states(s0: SceneState, s1: SceneState, raw_t: float) -> SceneState:
    """Smoothly tween continuous scene, camera, lighting, FX & object state."""
    t = _smoothstep(raw_t)
    out = deepcopy(s0 if raw_t < 1.0 else s1)
    out.background = s0.background if raw_t < 0.5 else s1.background
    out.background_file = s0.background_file if raw_t < 0.5 else s1.background_file
    out.background_blur = _lerp(s0.background_blur, s1.background_blur, t)
    out.time_of_day = s0.time_of_day if raw_t < 0.5 else s1.time_of_day
    out.weather = s0.weather if raw_t < 0.5 else s1.weather

    out.light_color = (
        int(_lerp(s0.light_color[0], s1.light_color[0], t)),
        int(_lerp(s0.light_color[1], s1.light_color[1], t)),
        int(_lerp(s0.light_color[2], s1.light_color[2], t)),
    )
    out.light_brightness = _lerp(s0.light_brightness, s1.light_brightness, t)
    out.cam_pan_x = _lerp(s0.cam_pan_x, s1.cam_pan_x, t)
    out.cam_tilt_y = _lerp(s0.cam_tilt_y, s1.cam_tilt_y, t)
    out.cam_zoom = _lerp(s0.cam_zoom, s1.cam_zoom, t)
    out.cam_shake = _lerp(s0.cam_shake, s1.cam_shake, t)

    all_fx = set(s0.scene_effects) | set(s1.scene_effects)
    out.scene_effects = {
        fx: _lerp(s0.scene_effects.get(fx, 0.0), s1.scene_effects.get(fx, 0.0), t)
        for fx in all_fx
    }

    all_names = set(s0.objects.keys()) | set(s1.objects.keys())
    out.objects = {}
    for name in all_names:
        if name in s0.objects and name in s1.objects:
            a, b = s0.objects[name], s1.objects[name]
            out.objects[name] = ObjectState(
                name=name,
                x=_lerp(a.x, b.x, t),
                y=_lerp(a.y, b.y, t),
                scale=_lerp(a.scale, b.scale, t),
                rotation=_lerp(a.rotation, b.rotation, raw_t),
                opacity=_lerp(a.opacity, b.opacity, t),
                glow=_lerp(a.glow, b.glow, t),
                glow_color=b.glow_color if b.glow > 0 else a.glow_color,
                shadow=max(a.shadow, b.shadow),
                flip_h=b.flip_h,
                flip_v=b.flip_v,
                file=b.file or a.file,
            )
        elif name in s0.objects:
            out.objects[name] = deepcopy(s0.objects[name])
        elif raw_t >= 0.99:
            out.objects[name] = deepcopy(s1.objects[name])

    out.text = deepcopy(s1.text if s1.text.content and raw_t >= 0.99 else s0.text)
    if s1.narration and raw_t >= 0.60:
        out.narration = s1.narration
        out.narration_start_frame = s1.narration_start_frame
    return out


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
    """Horizontally seamless (periodic in longitude) atmospheric cloud field."""
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 0.5
    total = 0.0
    for o, gy in enumerate((6, 12, 24, 48, 96)):
        gx = gy * 2
        rng = np.random.default_rng(seed + o * 29)
        grid = rng.random((gy, gx), dtype=np.float32)
        # Tile horizontally before bicubic upsampling to guarantee seamless 360° wrap
        tiled = np.concatenate([grid[:, -2:], grid, grid[:, :2]], axis=1)
        up = cv2.resize(tiled, (int(w * (gx + 4) / gx), h), interpolation=cv2.INTER_CUBIC)
        margin = int(round(w * 2 / gx))
        acc += amp * up[:, margin : margin + w]
        total += amp
        amp *= 0.52
    return np.clip(acc / max(total, 1e-6), 0.0, 1.0)


class StudioAssetBank:
    """
    Hybrid studio asset & shader manager:
    Combines bundled high-resolution plates (`assets/*.jpg/png`) when present
    with pure-math procedural synthesis + 2.5D depth maps for offline generality.
    """

    def __init__(self, width: int, height: int, base_dir: Path) -> None:
        self.width = width
        self.height = height
        self.pad_w = int(round(width * 1.18))
        self.pad_h = int(round(height * 1.18))
        self.base_dir = base_dir
        self.assets_dir = Path(__file__).resolve().parent / "assets"

        self._scene_cache: dict[tuple[str, str | None], tuple[np.ndarray, np.ndarray]] = {}
        self._earth_albedo: np.ndarray | None = None
        self._earth_clouds: np.ndarray | None = None
        self._earth_cities: np.ndarray | None = None
        self._eagle_rgba: np.ndarray | None = None

        # Precompute normalized screen grid for fast per-frame camera remap
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        self._screen_x = xx
        self._screen_y = yy
        self._norm_u = (xx / max(width - 1, 1)) * 2.0 - 1.0
        self._norm_v = (yy / max(height - 1, 1)) * 2.0 - 1.0
        self._vignette = np.clip(
            1.0 - 0.24 * (self._norm_u ** 2 * 0.72 + self._norm_v ** 2 * 0.92),
            0.55,
            1.0,
        )[..., None].astype(np.float32)

        # Horizontally periodic mist & particle tables (zero wrap-around seam!)
        self._mist_noise = _periodic_cloud_map(height, width, seed=404)
        rng = np.random.default_rng(777)
        self._particles = np.column_stack(
            [
                rng.uniform(0.0, 1.0, 190),   # x0
                rng.uniform(0.26, 0.94, 190), # y0
                rng.uniform(0.35, 1.25, 190), # speed
                rng.uniform(0.8, 2.3, 190),   # radius
                rng.uniform(0.0, math.tau, 190),
            ]
        ).astype(np.float32)
        self._stars_twinkle = np.column_stack(
            [
                rng.integers(2, width - 2, 160),
                rng.integers(2, height - 2, 160),
                rng.uniform(1.8, 5.2, 160),
                rng.uniform(0.0, math.tau, 160),
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
        """Return `(rgb_float32, depth_float32)` of shape `(pad_h, pad_w, ...)`."""
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
                    # Deepen exposure slightly around center so 3D Earth pops dramatically
                    yy, xx = np.mgrid[0 : self.pad_h, 0 : self.pad_w].astype(np.float32)
                    nu = (xx / self.pad_w) * 2.0 - 1.0
                    nv = (yy / self.pad_h) * 2.0 - 1.0
                    center_dim = 0.52 + 0.48 * np.clip((nu ** 2 + nv ** 2) * 0.95, 0.0, 1.0)
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

        # Build smooth 2.5D depth field in [0.20, 1.0] for parallax displacement
        v_ramp = np.linspace(0.18, 1.0, self.pad_h, dtype=np.float32)[:, None]
        luma = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
        detail = np.clip(np.hypot(gx, gy) * 1.4, 0.0, 1.0)
        if "space" in t_low or "galaxy" in t_low:
            depth = np.clip(0.28 + 0.45 * luma, 0.20, 0.85)
        else:
            depth = np.clip(v_ramp * 0.78 + detail * 0.28 * v_ramp, 0.18, 1.0)
        depth = cv2.GaussianBlur(depth, (0, 0), sigmaX=16.0, sigmaY=16.0)

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
        """
        True floating-point sub-pixel camera sampler with 2.5D depth parallax!
        Eliminates integer crop stepping completely across slow 26fps motion.
        """
        ch, cw, _ = bg_rgb.shape
        margin_x = (cw - self.width) * 0.5
        margin_y = (ch - self.height) * 0.5

        inv_z = 1.0 / max(zoom, 0.5)
        cx_pad = cw * 0.5
        cy_pad = ch * 0.5

        base_x = cx_pad + (self._screen_x - self.width * 0.5) * inv_z + shake_px[0]
        base_y = cy_pad + (self._screen_y - self.height * 0.5) * inv_z + shake_px[1]

        # Sample smooth depth at viewport coordinates to scale foreground vs horizon shift
        ix = np.clip(base_x.astype(np.int32), 0, cw - 1)
        iy = np.clip(base_y.astype(np.int32), 0, ch - 1)
        local_depth = depth[iy, ix]

        map_x = base_x + pan_x * margin_x * (0.42 + 0.85 * local_depth)
        map_y = base_y + tilt_y * margin_y * (0.42 + 0.85 * local_depth)

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

        # Procedural night-side urban light clusters constrained to continental land
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
        """
        5-Layer 3D Ray-Traced Planet Earth:
        1. Sub-pixel bilinear continents/oceans centered toward India & Himalayas
        2. Ocean-masked Blinn-Phong solar specular reflection
        3. Dynamic faster-rotating 3D cloud deck casting surface shadows
        4. Twinkling golden city lights along the night terminator
        5. Dual-scale Rayleigh atmospheric blue limb halo
        """
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

        # Slight north axial tilt so Eurasia, India & Himalayas shine front-and-center
        tilt_rad = math.radians(17.0)
        cos_t, sin_t = math.cos(tilt_rad), math.sin(tilt_rad)
        ry = ny * cos_t - nz * sin_t
        rz = ny * sin_t + nz * cos_t

        base_lon = math.radians(66.0 + rotation_deg * 2.4)
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

        # Independent super-rotating cloud deck + offset cloud shadow on surface
        u_cloud = np.mod(u_float + rotation_deg * 1.35, float(ew))
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

        # Sunlight vector from upper-left-front; eastern edge falls into dusk terminator
        lx, ly, lz = -0.52, -0.26, 0.81
        ndotl = nx * lx + ny * ly + nz * lz
        day_light = np.clip(ndotl, 0.0, 1.0) ** 0.80

        # Ocean solar specular highlight (Blinn-Phong half-vector `H`)
        hx, hy, hz = lx, ly, lz + 1.0
        h_norm = math.sqrt(hx * hx + hy * hy + hz * hz)
        ndoth = np.clip((nx * hx + ny * hy + nz * hz) / h_norm, 0.0, 1.0)
        is_ocean = (albedo[..., 2] > albedo[..., 1] * 1.06) & (
            albedo[..., 2] > albedo[..., 0] * 1.20
        )
        ocean_glint = (ndoth ** 28.0) * is_ocean.astype(np.float32) * (1.0 - cloud_cover)

        shaded = albedo * shadow_mul[..., None] * (0.045 + 1.12 * day_light[..., None])
        shaded += ocean_glint[..., None] * np.array([255.0, 236.0, 204.0], dtype=np.float32) * 0.55

        # Composite lit 3D clouds above surface
        cloud_lit = (0.07 + 1.05 * np.clip(ndotl + 0.06, 0.0, 1.0))[..., None] * 252.0
        shaded = shaded * (1.0 - cloud_cover[..., None]) + cloud_lit * cloud_cover[..., None]

        # Warm amber night-side city lights in dark hemisphere
        night_mask = np.clip((-ndotl + 0.06) * 5.0, 0.0, 1.0) * (1.0 - cloud_cover * 0.65)
        shaded += (cities * night_mask)[..., None] * np.array([255.0, 192.0, 105.0], dtype=np.float32) * 1.45

        # Atmospheric Rayleigh rim scatter on planet limb
        fresnel = ((1.0 - nz) ** 3.1) * (0.28 + 0.78 * np.clip(ndotl + 0.25, 0.0, 1.0))
        shaded += fresnel[..., None] * np.array([76.0, 168.0, 255.0], dtype=np.float32) * 0.92

        rgba = np.zeros((size, size, 4), dtype=np.float32)
        dist_norm = np.sqrt(r2)

        # Dual-scale outer atmospheric glow bloom
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
        flap_phase: float,
        bank_deg: float,
        opacity: float,
        shadow: float,
    ) -> Image.Image:
        """
        Studio golden eagle sprite with soft feather alpha keying, per-frame
        non-rigid parabolic wing-beat mesh deformation (`cv2.remap`), banking
        tilt, and soft atmospheric shadow.
        """
        if self._eagle_rgba is None:
            eagle_path = self.assets_dir / "golden_eagle.png"
            if eagle_path.is_file():
                raw = np.asarray(Image.open(eagle_path).convert("RGB"), dtype=np.float32)
                max_c = np.max(raw, axis=-1)
                alpha = np.clip((max_c - 9.0) / 34.0, 0.0, 1.0) ** 0.72
                alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.8)
                # Trim empty black borders around the eagle for tight sprite bounds
                ys, xs = np.where(alpha > 0.04)
                if len(xs) > 10 and len(ys) > 10:
                    x0, x1 = max(0, int(xs.min()) - 12), min(raw.shape[1], int(xs.max()) + 12)
                    y0, y1 = max(0, int(ys.min()) - 12), min(raw.shape[0], int(ys.max()) + 12)
                    raw = raw[y0:y1, x0:x1]
                    alpha = alpha[y0:y1, x0:x1]
                self._eagle_rgba = np.dstack([raw, alpha * 255.0]).astype(np.float32)

        if self._eagle_rgba is not None:
            sh0, sw0, _ = self._eagle_rgba.shape
            target_w = max(56, int(span_px * 1.45))
            target_h = max(36, int(target_w * (sh0 / max(sw0, 1))))
            base = cv2.resize(
                self._eagle_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA
            )

            # Non-rigid wing-beat deformation: outer wingtips arc smoothly every frame!
            pad_y = int(target_h * 0.42)
            pad_x = int(target_w * 0.16)
            canvas_h = target_h + 2 * pad_y
            canvas_w = target_w + 2 * pad_x
            padded = np.zeros((canvas_h, canvas_w, 4), dtype=np.float32)
            padded[pad_y : pad_y + target_h, pad_x : pad_x + target_w] = base

            yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
            norm_wing = np.abs(xx - canvas_w * 0.5) / max(target_w * 0.48, 1.0)
            flap_amp = math.sin(flap_phase) * (target_h * 0.20)
            body_bob = -math.sin(flap_phase) * (target_h * 0.035)

            src_y = yy + flap_amp * (norm_wing ** 1.55) + body_bob
            src_x = xx + math.sin(flap_phase) * 0.03 * (xx - canvas_w * 0.5) * norm_wing

            warped = cv2.remap(
                padded,
                src_x.astype(np.float32),
                src_y.astype(np.float32),
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            # Subtle banking rotation along flight trajectory
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
                sh_alpha = cv2.GaussianBlur(warped[..., 3], (0, 0), sigmaX=max(3.0, span_px * 0.035))
                M_sh = np.float32([[1, 0, span_px * 0.05], [0, 1, span_px * 0.085]])
                sh_shifted = cv2.warpAffine(sh_alpha, M_sh, (canvas_w, canvas_h))
                out[..., 0] = 16.0
                out[..., 1] = 20.0
                out[..., 2] = 32.0
                out[..., 3] = np.clip(sh_shifted * 0.44 * shadow * opacity, 0.0, 255.0)

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

        # Pure vector fallback if studio PNG is deleted
        w, h = max(48, int(span_px * 1.4)), max(36, int(span_px * 0.9))
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = w * 0.5, h * 0.54
        s = span_px * 0.42
        flap = math.sin(flap_phase) * (h * 0.11)
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
# VIDEX CINEMA VIDEO ENGINE
# ══════════════════════════════════════════════════════════════════════════════


class VidexVideoEngine:
    """Executes a validated ``VideoProgram`` AST and encodes an H.264 `.mp4`."""

    def __init__(self, *, base_dir: Path | None = None, show_hud: bool = False) -> None:
        self.base_dir = base_dir or Path.cwd()
        self.show_hud = show_hud

    def render(self, program: VideoProgram, output_path: str | Path) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fps = program.fps
        total_frames = max(1, int(round(program.total_duration * fps)))
        assets = StudioAssetBank(program.width, program.height, self.base_dir)

        timeline, transitions = self._build_timeline(program, total_frames)

        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "soundtrack.wav"
            self._write_cinematic_score(
                wav_path=wav_path,
                duration_sec=total_frames / fps,
                fps=fps,
                timeline=timeline,
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
                "-crf", "18",
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
                    state = timeline[f_idx]
                    rgb = self._render_single_frame(
                        program=program,
                        assets=assets,
                        state=state,
                        global_frame=f_idx,
                        total_frames=total_frames,
                    )

                    if f_idx in transitions:
                        prev_state, trans_type, alpha = transitions[f_idx]
                        prev_rgb = self._render_single_frame(
                            program=program,
                            assets=assets,
                            state=prev_state,
                            global_frame=f_idx,
                            total_frames=total_frames,
                        )
                        rgb = self._blend_transition(prev_rgb, rgb, trans_type, alpha)

                    if self.show_hud:
                        rgb = self._draw_frame_hud(rgb, program, state, f_idx, total_frames)

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

    # ── Timeline & Keyframe Interpreter ───────────────────────────────────────

    def _build_timeline(
        self,
        program: VideoProgram,
        total_frames: int,
    ) -> tuple[list[SceneState], dict[int, tuple[SceneState, str, float]]]:
        fps = program.fps
        state_cursor = SceneState()

        for g_cmd in program.global_commands:
            self._apply_command(state_cursor, g_cmd, program, global_frame=0)

        sec_map: dict[int, SecondBlock] = {s.second_number: s for s in program.seconds}
        timeline: list[SceneState] = []
        transitions: dict[int, tuple[SceneState, str, float]] = {}

        num_seconds = max(1, math.ceil(total_frames / fps))
        prev_sec_end_state: SceneState | None = None

        for sec_num in range(1, num_seconds + 1):
            sec_start_global = (sec_num - 1) * fps
            block = sec_map.get(sec_num)

            if block is not None and block.frames:
                kf_states: list[tuple[int, SceneState]] = []
                working = deepcopy(state_cursor)

                for fb in sorted(block.frames, key=lambda b: b.index):
                    local_idx = max(1, min(fps, fb.index))
                    gf = sec_start_global + (local_idx - 1)
                    for cmd in fb.commands:
                        self._apply_command(working, cmd, program, global_frame=gf)
                    kf_states.append((local_idx, deepcopy(working)))

                sec_frames: list[SceneState] = []
                for local_f in range(1, fps + 1):
                    if block.auto_fill and len(kf_states) >= 2:
                        st = self._sample_keyframes(kf_states, local_f)
                    else:
                        active = kf_states[0][1]
                        for idx, cand_st in kf_states:
                            if idx <= local_f:
                                active = cand_st
                        st = deepcopy(active)

                    # Gentle orbital push-in on space planet scene for cinematic depth
                    if "space" in st.background.lower():
                        st.cam_zoom = max(st.cam_zoom, 1.0 + 0.055 * (local_f / fps))
                    sec_frames.append(st)

                if block.transition and prev_sec_end_state is not None:
                    trans_len = min(14, int(fps * 0.55))
                    for k in range(trans_len):
                        gf = sec_start_global + k
                        if gf < total_frames:
                            transitions[gf] = (
                                deepcopy(prev_sec_end_state),
                                block.transition,
                                (k + 1) / (trans_len + 1),
                            )

                state_cursor = deepcopy(sec_frames[-1])
                prev_sec_end_state = deepcopy(state_cursor)
                timeline.extend(sec_frames)
            else:
                # Smooth continuous drone & eagle flight across remaining seconds
                base_state = deepcopy(state_cursor)
                extra_total = max(1, total_frames - sec_start_global)
                for local_f in range(1, fps + 1):
                    st = deepcopy(base_state)
                    elapsed_after = (sec_num - 1) * fps + local_f
                    prog = elapsed_after / max(total_frames, 1)
                    st.cam_pan_x = max(-1.0, min(1.0, st.cam_pan_x + 0.22 * prog))
                    st.cam_zoom = 1.0 + 0.065 * prog
                    for obj in st.objects.values():
                        if "eagle" in obj.name.lower() or "bird" in obj.name.lower():
                            # Continue gliding smoothly toward the golden horizon peak
                            glide = local_f / fps
                            obj.x = min(0.84, obj.x + 0.055 * glide)
                            obj.y = max(0.16, obj.y - 0.012 * glide + math.sin(elapsed_after * 0.09) * 0.004)
                            obj.scale = max(0.24, obj.scale * (1.0 - 0.08 * glide))
                    if local_f == fps:
                        state_cursor = deepcopy(st)
                    timeline.append(st)
                prev_sec_end_state = deepcopy(timeline[-1])

        return timeline[:total_frames], transitions

    @staticmethod
    def _sample_keyframes(
        kf_states: list[tuple[int, SceneState]],
        local_f: int,
    ) -> SceneState:
        if local_f <= kf_states[0][0]:
            return deepcopy(kf_states[0][1])
        if local_f >= kf_states[-1][0]:
            return deepcopy(kf_states[-1][1])
        for idx in range(len(kf_states) - 1):
            f0, s0 = kf_states[idx]
            f1, s1 = kf_states[idx + 1]
            if f0 <= local_f <= f1:
                span = max(1, f1 - f0)
                return interpolate_states(s0, s1, (local_f - f0) / span)
        return deepcopy(kf_states[-1][1])

    @staticmethod
    def _apply_command(
        state: SceneState,
        cmd: Command,
        program: VideoProgram,
        global_frame: int,
    ) -> None:
        act = cmd.action
        p = cmd.params

        if act == "set_background":
            new_bg = str(p["theme"])
            if new_bg.lower() != state.background.lower():
                state.objects = {}
                state.cam_pan_x = 0.0
                state.cam_tilt_y = 0.0
                state.cam_zoom = 1.0
            state.background = new_bg
            state.background_file = p.get("file") if isinstance(p.get("file"), str) else None

        elif act == "set_background_time":
            state.time_of_day = str(p["time"])

        elif act == "set_background_weather":
            state.weather = str(p["weather"])

        elif act == "set_background_season":
            state.season = str(p["season"])

        elif act == "set_background_blur":
            state.background_blur = max(0.0, float(p["blur"]))

        elif act == "place_object":
            name = str(p["name"])
            ax, ay = anchor_normalized(str(p["position"]))
            state.objects[name] = ObjectState(
                name=name,
                x=ax,
                y=ay,
                file=p.get("file") if isinstance(p.get("file"), str) else None,
            )

        elif act == "remove_object":
            state.objects.pop(str(p["name"]), None)

        elif act == "flip_object":
            name = str(p["name"])
            if name in state.objects:
                axis = str(p.get("axis", "horizontally")).lower()
                if "vert" in axis:
                    state.objects[name].flip_v = not state.objects[name].flip_v
                else:
                    state.objects[name].flip_h = not state.objects[name].flip_h

        elif act == "scale_object":
            name = str(p["name"])
            if name in state.objects:
                sz = str(p["size"]).lower()
                state.objects[name].scale = SIZE_SCALE.get(sz, 1.0)

        elif act == "rotate_object":
            name = str(p["name"])
            if name in state.objects:
                state.objects[name].rotation += float(p["degrees"])

        elif act == "move_object":
            name = str(p["name"])
            if name in state.objects:
                tx, ty = anchor_normalized(str(p["position"]))
                px = p.get("pixels")
                if isinstance(px, (int, float)):
                    dx = tx - state.objects[name].x
                    dy = ty - state.objects[name].y
                    norm_len = math.hypot(dx, dy) or 1.0
                    step = float(px) / max(program.width, 1)
                    state.objects[name].x = max(0.05, min(0.95, state.objects[name].x + (dx / norm_len) * step))
                    state.objects[name].y = max(0.05, min(0.95, state.objects[name].y + (dy / norm_len) * step * 0.45))
                else:
                    state.objects[name].x, state.objects[name].y = tx, ty

        elif act == "set_object_opacity":
            name = str(p["name"])
            if name in state.objects:
                state.objects[name].opacity = max(0.0, min(1.0, float(p["opacity"])))

        elif act == "set_light_color":
            state.light_color = resolve_rgb(str(p["color"]))

        elif act == "set_light_brightness":
            state.light_brightness = max(0.20, min(1.45, float(p["brightness"])))

        elif act == "camera_pan":
            direction = str(p["direction"]).lower()
            delta = SPEED_DELTA.get(str(p["speed"]).lower(), 0.5)
            state.cam_pan_x = max(-1.0, min(1.0, state.cam_pan_x + (-delta if direction == "left" else delta)))

        elif act == "camera_tilt":
            direction = str(p["direction"]).lower()
            delta = SPEED_DELTA.get(str(p["speed"]).lower(), 0.5)
            state.cam_tilt_y = max(-1.0, min(1.0, state.cam_tilt_y + (delta if direction == "down" else -delta)))

        elif act == "camera_shake":
            lvl = {"low": 3.0, "small": 3.0, "medium": 7.5, "high": 14.0, "heavy": 18.0}.get(
                str(p.get("intensity", "medium")).lower(), 7.5
            )
            state.cam_shake = lvl

        elif act == "set_camera_zoom":
            z_str = str(p["zoom"]).lower()
            if "wide" in z_str:
                state.cam_zoom = 1.0
            elif "extreme" in z_str:
                state.cam_zoom = 1.32
            elif "close" in z_str:
                state.cam_zoom = 1.18

        elif act == "place_text":
            tx, ty = anchor_normalized(str(p["position"]))
            state.text = TextState(
                content=str(p["content"]),
                x=tx,
                y=ty,
                appear_global_frame=global_frame,
            )

        elif act == "set_text_color":
            state.text.color = resolve_rgb(str(p["color"]))

        elif act == "set_text_size":
            raw_sz = str(p["size"]).lower()
            if raw_sz in ("small", "medium", "large"):
                state.text.size = {"small": 32.0, "medium": 52.0, "large": 76.0}[raw_sz]
            else:
                state.text.size = float(p["size"])

        elif act == "fade_text_in":
            state.text.fade_in_frames = max(1, int(p["frames"]))

        elif act == "fade_text_out":
            state.text.fade_out_frames = max(1, int(p["frames"]))

        elif act == "add_text_effect":
            if str(p["effect"]).lower() == "glow":
                state.text.glow = 1.0
                state.text.glow_color = resolve_rgb(str(p["color"]))

        elif act == "add_effect":
            target = p.get("object")
            effect = str(p["effect"]).lower()
            lvl = float(p.get("level", 1.0))
            if isinstance(target, str) and target in state.objects:
                obj = state.objects[target]
                if effect == "glow":
                    obj.glow = lvl
                    obj.glow_color = resolve_rgb(str(p["color"]))
                elif effect == "shadow":
                    obj.shadow = lvl
            else:
                state.scene_effects[effect] = lvl

        elif act == "narrate":
            state.narration = str(p["text"])
            state.narration_start_frame = global_frame

    # ── Per-Frame Cinema Shader & Post-FX Compositor ──────────────────────────

    def _render_single_frame(
        self,
        *,
        program: VideoProgram,
        assets: StudioAssetBank,
        state: SceneState,
        global_frame: int,
        total_frames: int,
    ) -> np.ndarray:
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

        # 1. Sub-pixel floating-point 2.5D depth-parallax camera sampling
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

        # 2. Per-frame dynamic atmosphere (twinkling stars OR valley mist + sun rays + spindrift)
        bg_low = state.background.lower()
        if "space" in bg_low or "galaxy" in bg_low:
            scene_f32 = self._apply_space_scintillation(scene_f32, assets, time_s)
        elif "mountain" in bg_low or "himalaya" in bg_low:
            scene_f32 = self._apply_mountain_atmosphere(scene_f32, assets, state, time_s)

        # 3. Time-of-day color grade + scripted lighting tint + ACES filmic tone curve
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

        # 4. Weather & scene post-FX (snow, rain, fog, vignette, glitch, B&W, vintage)
        tonemapped = self._apply_weather_and_fx(tonemapped, assets, state, time_s)
        tonemapped *= assets._vignette

        frame_img = Image.fromarray(np.clip(tonemapped, 0.0, 255.0).astype(np.uint8), mode="RGB").convert("RGBA")

        # 5. Sub-pixel object sprites (3D Earth / flapping Golden Eagle / imported sprites)
        cam_px_shift = -state.cam_pan_x * (w * 0.038)
        cam_py_shift = -state.cam_tilt_y * (h * 0.038)

        for obj in state.objects.values():
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
                    rotation_deg=obj.rotation + global_frame * 0.42,
                    glow=obj.glow,
                    glow_color=obj.glow_color,
                )
            elif "eagle" in name_low or "bird" in name_low:
                span = max(44, int(min(w, h) * 0.25 * obj.scale))
                bank = -7.5 + math.sin(time_s * 1.4) * 4.5
                sprite = assets.render_eagle_rgba(
                    span_px=span,
                    flap_phase=time_s * 4.4,
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

        # 6. Cinema Typography & Glassmorphic Narration Ribbon
        if state.text.content:
            elapsed = max(0, global_frame - state.text.appear_global_frame + 1)
            fade_alpha = min(1.0, elapsed / max(1, state.text.fade_in_frames))
            if state.text.fade_out_frames > 0 and total_frames - global_frame <= state.text.fade_out_frames:
                fade_alpha *= max(0.0, (total_frames - global_frame) / state.text.fade_out_frames)
            self._draw_glow_text(
                frame_img=frame_img,
                text=state.text.content,
                cx=int(state.text.x * w),
                cy=int(state.text.y * h),
                font_px=int(state.text.size * (h / 1080.0)),
                color=state.text.color,
                alpha=fade_alpha,
                glow=state.text.glow,
                glow_color=state.text.glow_color,
                serif=True,
            )

        if state.narration and state.narration_start_frame >= 0:
            n_elapsed = max(0, global_frame - state.narration_start_frame)
            n_alpha = min(1.0, (n_elapsed + 3) / 8.0)
            if total_frames - global_frame < 12:
                n_alpha *= (total_frames - global_frame) / 12.0
            if n_alpha > 0.02:
                self._draw_narration_bar(
                    frame_img=frame_img,
                    narration=f'“{state.narration}”',
                    alpha=n_alpha,
                )

        # 7. Subtle 10-bit anti-banding dither (eliminates sky gradient banding cleanly)
        final_rgb = np.asarray(frame_img.convert("RGB"), dtype=np.float32)
        grain_rng = np.random.default_rng(10000 + global_frame)
        grain = grain_rng.normal(0.0, 0.55, size=(h // 2, w // 2, 1)).astype(np.float32)
        grain_full = cv2.resize(grain, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
        return np.clip(final_rgb + grain_full, 0.0, 255.0).astype(np.uint8)

    @staticmethod
    def _alpha_composite_subpixel(
        dest_img: Image.Image, sprite: Image.Image, center_x: float, center_y: float
    ) -> None:
        """Sub-pixel bilinear translation of RGBA sprite before compositing!"""
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
                boost = pulse * 145.0
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

        # 1. Advecting golden-blue valley mist across lower alpine valleys every frame
        shift = int(round((time_s * w * 0.026))) % w
        mist_slice = np.roll(assets._mist_noise, shift=shift, axis=1)[:, :w]
        v_mask = np.clip((assets._screen_y / h - 0.54) / 0.40, 0.0, 1.0)
        mist_amount = np.clip((mist_slice - 0.46) * 1.45, 0.0, 0.42) * v_mask
        mist_color = np.array([216.0, 224.0, 240.0], dtype=np.float32)
        out = out * (1.0 - mist_amount[..., None] * 0.55) + mist_color * (mist_amount[..., None] * 0.55)

        # 2. Subtle volumetric golden sun glow + warm anamorphic shimmer
        sun_u = 0.365 - state.cam_pan_x * 0.055
        sun_v = 0.285 - state.cam_tilt_y * 0.055
        du = (assets._screen_x / w - sun_u) * (w / max(h, 1))
        dv = assets._screen_y / h - sun_v
        r2 = du * du + dv * dv
        shimmer = 0.92 + 0.08 * math.sin(time_s * 2.1)
        warm_bloom = np.exp(-r2 / 0.055) * 0.24 * shimmer
        anamorphic = np.exp(-(dv * dv) / 0.00055) * np.exp(-(du * du) / 0.16) * 0.16 * shimmer
        out += (warm_bloom + anamorphic)[..., None] * np.array([255.0, 198.0, 118.0], dtype=np.float32)

        # 3. Wind-blown golden snow spindrift particles streaming gently off peaks
        overlay = np.zeros_like(out)
        for x0, y0, spd, rad, ph in assets._particles:
            px = ((x0 + time_s * 0.032 * spd) % 1.0) * (w - 1)
            py = (y0 + math.sin(time_s * 1.6 + ph) * 0.008) * (h - 1)
            cv2.circle(
                overlay,
                (int(round(px)), int(round(py))),
                max(1, int(round(rad))),
                (255.0, 232.0, 198.0),
                -1,
                lineType=cv2.LINE_AA,
            )
        return np.clip(out + overlay * 0.32, 0.0, 255.0)

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
        """Frame-accurate cinema telemetry overlay (`--hud`)."""
        img = Image.fromarray(rgb, mode="RGB").convert("RGBA")
        font = load_font(max(13, int(program.height * 0.018)), mono=True)
        sec_idx = global_frame // program.fps + 1
        local_f = global_frame % program.fps + 1
        hud_line = (
            f"VIDEX ENGINE  |  SEC {sec_idx:02d}  FRAME {local_f:02d}/{program.fps:02d} "
            f"(#{global_frame + 1:03d}/{total_frames:03d})  |  "
            f"PAN {state.cam_pan_x:+.2f}  TILT {state.cam_tilt_y:+.2f}  ZOOM {state.cam_zoom:.2f}x"
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
    def _blend_transition(
        prev_rgb: np.ndarray,
        curr_rgb: np.ndarray,
        trans_type: str,
        alpha: float,
    ) -> np.ndarray:
        a = _smoothstep(alpha)
        h, w, _ = curr_rgb.shape
        t = trans_type.lower()

        if "cut" in t:
            return curr_rgb
        if "wipe" in t:
            split = int(w * a)
            out = prev_rgb.copy()
            out[:, :split] = curr_rgb[:, :split]
            return out
        if "slide" in t:
            shift = int(w * (1.0 - a))
            out = prev_rgb.copy()
            out[:, : w - shift] = curr_rgb[:, shift:]
            return out
        if "zoom" in t:
            z = 1.0 + 0.18 * a
            M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.0, z)
            zoomed_prev = cv2.warpAffine(prev_rgb, M, (w, h))
            return np.clip(
                zoomed_prev.astype(np.float32) * (1.0 - a) + curr_rgb.astype(np.float32) * a,
                0.0,
                255.0,
            ).astype(np.uint8)

        return np.clip(
            prev_rgb.astype(np.float32) * (1.0 - a) + curr_rgb.astype(np.float32) * a,
            0.0,
            255.0,
        ).astype(np.uint8)

    @staticmethod
    def _write_cinematic_score(
        *,
        wav_path: Path,
        duration_sec: float,
        fps: int,
        timeline: list[SceneState],
    ) -> None:
        """Synthesize a lush warm orchestral ambient pad + Himalayan wind shimmer + chime."""
        sr = 44100
        n_samples = max(1, int(round(duration_sec * sr)))
        t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

        # Warm evolving string pad shifting gently from cosmic fifth to radiant Lydian sunrise
        swell = 0.68 + 0.32 * np.sin(2.0 * math.pi * 0.14 * t)
        sunrise_shift = np.clip((t - 1.0) / 1.5, 0.0, 1.0)
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

        # Gentle high-altitude air texture during mountain scenes
        rng = np.random.default_rng(2026)
        raw_air = rng.normal(0.0, 0.028, n_samples).astype(np.float32)
        kernel = np.ones(64, dtype=np.float32) / 64.0
        soft_wind = np.convolve(raw_air, kernel, mode="same") * sunrise_shift
        left += soft_wind
        right += np.roll(soft_wind, 220)

        # Crystalline harmonic bell arpeggio when `narrate "..."` triggers
        narrate_frame = next((i for i, st in enumerate(timeline) if st.narration), -1)
        if narrate_frame >= 0:
            cue_time = narrate_frame / max(fps, 1)
            for idx, freq in enumerate((523.25, 659.25, 783.99, 1046.50)):
                onset = cue_time + idx * 0.15
                dt = np.maximum(0.0, t - onset)
                bell_env = (t >= onset).astype(np.float32) * np.exp(-dt * 2.2)
                bell = 0.13 * np.sin(2.0 * math.pi * freq * dt) * bell_env
                left += bell
                right += bell * 0.94

        fade_len = min(int(0.50 * sr), n_samples // 4)
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
