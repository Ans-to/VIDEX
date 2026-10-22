"""
VIDEX Programming Language — Keyframe Interpolator & Video Engine (Stages 3–4)
==============================================================================

Pipeline Stages 3 & 4:
    VIDEX Source Code -> Lexer -> Parser -> [INTERPRETER -> VIDEO ENGINE] -> MP4

Evaluates a parsed ``VideoProgram`` AST across frames (`26 fps` VIDEX standard),
interpolates properties between keyframes (`fill frames automatically`), renders
high-definition procedural scenes & objects with NumPy + Pillow, synthesizes an
ambient stereo soundtrack, and encodes a browser-ready H.264 `.mp4` video.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from pathlib import Path
import subprocess
import tempfile
import wave

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from videx_parser import Command, SecondBlock, VideoProgram


# ══════════════════════════════════════════════════════════════════════════════
# COLOR, POSITION & FONT HELPERS
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
    "warm": (255, 198, 138),
    "cold": (168, 214, 255),
}

SIZE_SCALE: dict[str, float] = {
    "small": 0.48,
    "medium": 1.0,
    "large": 1.42,
}

SPEED_DELTA: dict[str, float] = {
    "slow": 0.36,
    "medium": 0.65,
    "fast": 1.0,
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
        "top left": (0.20, 0.22),
        "top right": (0.78, 0.22),
        "top center": (0.50, 0.18),
        "bottom left": (0.22, 0.84),
        "bottom right": (0.78, 0.84),
        "bottom center": (0.50, 0.85),
    }
    return mapping.get(position.lower(), (0.50, 0.50))


def load_font(size_px: int, *, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        if serif
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=max(12, int(size_px)))
    return ImageFont.load_default()


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ══════════════════════════════════════════════════════════════════════════════
# SCENE STATE MODEL FOR KEYFRAME EVALUATION & TWEENING
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
    file: str | None = None


@dataclass
class TextState:
    content: str = ""
    x: float = 0.50
    y: float = 0.85
    size: float = 50.0
    color: tuple[int, int, int] = (255, 255, 255)
    alpha: float = 0.0
    fade_frames: int = 1
    appear_global_frame: int = -1
    glow: float = 0.0
    glow_color: tuple[int, int, int] = (255, 255, 255)


@dataclass
class SceneState:
    background: str = "space galaxy"
    background_file: str | None = None
    time_of_day: str = "night"
    weather: str = "clear"
    season: str = "summer"
    light_color: tuple[int, int, int] = (210, 225, 255)
    light_brightness: float = 0.78
    cam_pan_x: float = 0.0   # [-1.0, +1.0] viewport shift
    cam_tilt_y: float = 0.0  # [-1.0, +1.0] viewport shift
    cam_zoom: float = 1.0
    objects: dict[str, ObjectState] = field(default_factory=dict)
    text: TextState = field(default_factory=TextState)
    narration: str = ""
    narration_start_frame: int = -1


def interpolate_states(s0: SceneState, s1: SceneState, raw_t: float) -> SceneState:
    """Smoothly interpolate continuous scene properties between two keyframes."""
    t = _smoothstep(raw_t)
    out = deepcopy(s0 if raw_t < 1.0 else s1)
    out.background = s0.background if raw_t < 0.5 else s1.background
    out.background_file = s0.background_file if raw_t < 0.5 else s1.background_file
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
                file=b.file or a.file,
            )
        elif name in s0.objects:
            out.objects[name] = deepcopy(s0.objects[name])
        elif raw_t >= 0.99:
            out.objects[name] = deepcopy(s1.objects[name])

    out.text = deepcopy(s1.text if s1.text.content and raw_t >= 0.99 else s0.text)
    if s1.narration and raw_t >= 0.65:
        out.narration = s1.narration
        out.narration_start_frame = s1.narration_start_frame
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PROCEDURAL ASSET SYNTHESIS (BACKGROUNDS & OBJECT SPRITES)
# ══════════════════════════════════════════════════════════════════════════════


def _value_noise_2d(h: int, w: int, grid_y: int, grid_x: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.random((grid_y, grid_x), dtype=np.float32)
    img = Image.fromarray((coarse * 255.0).astype(np.uint8), mode="L")
    resized = img.resize((w, h), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _fbm_2d(h: int, w: int, base_grid: int = 8, octaves: int = 4, seed: int = 42) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 0.5
    total_amp = 0.0
    freq = base_grid
    for o in range(octaves):
        acc += amp * _value_noise_2d(h, w, max(2, freq), max(2, int(freq * w / max(h, 1))), seed + o * 17)
        total_amp += amp
        amp *= 0.5
        freq *= 2
    return acc / max(total_amp, 1e-6)


class ProceduralAssetFactory:
    """Generates high-definition cached backgrounds & object sprites in memory."""

    def __init__(self, width: int, height: int, base_dir: Path) -> None:
        self.width = width
        self.height = height
        self.pad_w = int(round(width * 1.16))
        self.pad_h = int(round(height * 1.16))
        self.base_dir = base_dir
        self._bg_cache: dict[tuple[str, str, str | None], np.ndarray] = {}
        self._earth_surface: np.ndarray | None = None
        self._eagle_sprite: Image.Image | None = None

    def get_background(self, theme: str, time_of_day: str, file_path: str | None) -> np.ndarray:
        key = (theme.lower(), time_of_day.lower(), file_path)
        if key in self._bg_cache:
            return self._bg_cache[key]

        if file_path:
            candidate = Path(file_path)
            if not candidate.is_absolute():
                candidate = self.base_dir / candidate
            if candidate.is_file():
                loaded = Image.open(candidate).convert("RGB").resize(
                    (self.pad_w, self.pad_h), Image.Resampling.LANCZOS
                )
                arr = np.asarray(loaded, dtype=np.float32)
                self._bg_cache[key] = arr
                return arr

        t_low = theme.lower()
        if "space" in t_low or "galaxy" in t_low or "star" in t_low:
            arr = self._make_space_galaxy_canvas(self.pad_w, self.pad_h)
        elif "mountain" in t_low or "himalaya" in t_low:
            arr = self._make_himalayas_canvas(self.pad_w, self.pad_h, time_of_day)
        else:
            arr = self._make_generic_scenic_canvas(self.pad_w, self.pad_h, t_low)

        self._bg_cache[key] = arr
        return arr

    @staticmethod
    def _make_space_galaxy_canvas(w: int, h: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / w) * 2.0 - 1.0
        ny = (yy / h) * 2.0 - 1.0

        # Deep space base
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        canvas[..., 0] = 5.0
        canvas[..., 1] = 8.0
        canvas[..., 2] = 20.0

        # Diagonal galactic band + fBm nebula clouds
        band_dist = np.abs(0.55 * nx + 0.83 * ny + 0.06)
        band = np.exp(-(band_dist ** 2) / 0.14)
        neb1 = _fbm_2d(h, w, base_grid=6, octaves=5, seed=101)
        neb2 = _fbm_2d(h, w, base_grid=11, octaves=4, seed=205)

        cyan_glow = band * np.clip(neb1 * 1.4 - 0.32, 0.0, 1.0)
        violet_glow = np.exp(-(band_dist ** 2) / 0.30) * np.clip(neb2 * 1.35 - 0.36, 0.0, 1.0)

        canvas[..., 0] += cyan_glow * 34.0 + violet_glow * 96.0
        canvas[..., 1] += cyan_glow * 98.0 + violet_glow * 42.0
        canvas[..., 2] += cyan_glow * 165.0 + violet_glow * 138.0

        # Crisp starfield with realistic stellar color temperatures
        rng = np.random.default_rng(2026)
        num_stars = int((w * h) / 1450)
        sx = rng.integers(1, w - 1, size=num_stars)
        sy = rng.integers(1, h - 1, size=num_stars)
        bright = rng.power(0.35, size=num_stars) * 255.0
        tint = rng.choice(
            np.array([[1.0, 0.98, 0.94], [0.80, 0.90, 1.0], [1.0, 0.86, 0.72]], dtype=np.float32),
            size=num_stars,
        )
        star_colors = bright[:, None] * tint
        canvas[sy, sx] = np.maximum(canvas[sy, sx], star_colors)
        # Soft cross halo on brightest stars
        bright_mask = bright > 215.0
        for bx, by, bc in zip(sx[bright_mask], sy[bright_mask], star_colors[bright_mask]):
            canvas[by - 1 : by + 2, bx - 1 : bx + 2] = np.maximum(
                canvas[by - 1 : by + 2, bx - 1 : bx + 2], bc * 0.55
            )

        return np.clip(canvas, 0.0, 255.0)

    @staticmethod
    def _make_himalayas_canvas(w: int, h: int, time_of_day: str) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        v = yy / max(h - 1, 1)
        u = xx / max(w - 1, 1)

        # Sunrise sky gradient: deep twilight royal blue -> rose violet -> warm gold horizon
        sky = np.zeros((h, w, 3), dtype=np.float32)
        c_top = np.array([24.0, 38.0, 82.0], dtype=np.float32)
        c_mid = np.array([152.0, 92.0, 118.0], dtype=np.float32)
        c_hor = np.array([255.0, 196.0, 116.0], dtype=np.float32)

        upper = np.clip(v / 0.46, 0.0, 1.0)[..., None]
        lower = np.clip((v - 0.46) / 0.30, 0.0, 1.0)[..., None]
        sky = np.where(
            v[..., None] < 0.46,
            c_top * (1.0 - upper) + c_mid * upper,
            c_mid * (1.0 - lower) + c_hor * lower,
        )

        # Rising sun glow near upper-right mountain saddle
        sun_u, sun_v = 0.64, 0.40
        aspect = w / max(h, 1)
        dist = np.sqrt(((u - sun_u) * aspect) ** 2 + (v - sun_v) ** 2)
        sun_halo = np.exp(-(dist ** 2) / 0.045)
        sun_core = np.exp(-(dist ** 2) / 0.0016)
        sky += sun_halo[..., None] * np.array([115.0, 82.0, 36.0], dtype=np.float32)
        sky += sun_core[..., None] * np.array([180.0, 165.0, 130.0], dtype=np.float32)

        # Soft high-altitude cirrus clouds
        clouds = _fbm_2d(h, w, base_grid=7, octaves=4, seed=77)
        cloud_mask = np.clip((clouds - 0.52) * 2.4, 0.0, 1.0) * np.clip(1.0 - v * 1.45, 0.0, 1.0)
        sky += cloud_mask[..., None] * np.array([85.0, 58.0, 48.0], dtype=np.float32)

        # 5 layered parallax Himalayan mountain ridges with true 2D facet shading
        x_idx = np.arange(w, dtype=np.float32) / max(w - 1, 1)
        facet_tex = _fbm_2d(h, w, base_grid=14, octaves=5, seed=313)
        fine_tex = _fbm_2d(h, w, base_grid=34, octaves=3, seed=617)
        layers = [
            # (base_v, amp, base_freq, seed, rock_rgb, snow_rgb, snow_span, haze)
            (0.49, 0.15, 2.2, 11, (76, 84, 122), (255, 224, 198), 0.13, 0.55),
            (0.57, 0.17, 2.8, 23, (56, 66, 102), (252, 232, 214), 0.15, 0.38),
            (0.66, 0.19, 2.4, 37, (38, 48, 80), (246, 240, 238), 0.16, 0.22),
            (0.76, 0.17, 3.1, 53, (26, 34, 58), (232, 238, 246), 0.14, 0.12),
            (0.87, 0.13, 2.6, 71, (16, 22, 36), (196, 208, 224), 0.10, 0.05),
        ]

        for base_v, amp, base_freq, lseed, rock_rgb, snow_rgb, snow_span, haze in layers:
            rng = np.random.default_rng(lseed)
            profile = np.zeros_like(x_idx)
            freq = base_freq
            a = 0.55
            for _ in range(5):
                phase = rng.uniform(0.0, math.tau)
                ridge_wave = 1.0 - np.abs(np.sin(x_idx * freq * math.pi + phase))
                profile += a * ridge_wave
                freq *= 1.92
                a *= 0.50
            profile = (profile - profile.min()) / max(float(np.ptp(profile)), 1e-5)
            ridge_y = (base_v - amp * (profile ** 1.18)) * h

            below = yy >= ridge_y[None, :]
            depth_px = np.maximum(0.0, yy - ridge_y[None, :])
            depth = depth_px / max(h * snow_span, 1.0)

            # Diagonal couloir spurs branching downward from peaks into 3D faces
            spur_shift = x_idx[None, :] + (depth_px / max(h, 1)) * 0.24
            spur_wave = np.sin(spur_shift * base_freq * 5.5 * math.pi + lseed)
            surface_relief = (
                (1.0 - depth * 0.45) * 0.55
                + facet_tex * 0.55
                + fine_tex * 0.22
                + spur_wave * 0.16 * np.exp(-depth * 0.7)
            )
            gy, gx = np.gradient(surface_relief)
            # Sunlight arriving from upper-right horizon
            sunlit = np.clip(0.52 + gx * 14.0 - gy * 6.0, 0.08, 1.0)

            snow_factor = np.clip(
                1.18 - depth * 0.92 - (1.0 - facet_tex) * 0.48 + sunlit * 0.22,
                0.0,
                1.0,
            )
            rock = np.array(rock_rgb, dtype=np.float32) * (0.58 + 0.68 * sunlit[..., None])
            snow = np.array(snow_rgb, dtype=np.float32) * (0.64 + 0.44 * sunlit[..., None])
            m_col = rock * (1.0 - snow_factor[..., None]) + snow * snow_factor[..., None]

            # Warm golden atmospheric mist pooling in deep valleys
            haze_col = np.array([232.0, 184.0, 152.0], dtype=np.float32)
            valley_mist = haze * np.clip(depth * 0.42, 0.0, 0.72)
            m_col = m_col * (1.0 - valley_mist[..., None]) + haze_col * valley_mist[..., None]

            sky = np.where(below[..., None], m_col, sky)

        return np.clip(sky, 0.0, 255.0)

    @staticmethod
    def _make_generic_scenic_canvas(w: int, h: int, theme: str) -> np.ndarray:
        yy, _ = np.mgrid[0:h, 0:w].astype(np.float32)
        v = (yy / max(h - 1, 1))[..., None]
        noise = _fbm_2d(h, w, base_grid=8, octaves=4, seed=abs(hash(theme)) % 10000)[..., None]
        top = np.array([26.0, 62.0, 118.0], dtype=np.float32)
        bot = np.array([82.0, 148.0, 132.0], dtype=np.float32)
        return np.clip(top * (1.0 - v) + bot * v + (noise - 0.5) * 55.0, 0.0, 255.0)

    def render_earth_rgba(
        self,
        diameter: int,
        rotation_deg: float,
        glow: float,
        glow_color: tuple[int, int, int],
    ) -> Image.Image:
        """Render a 3D ray-shaded rotating planet Earth with clouds & rim glow."""
        if self._earth_surface is None:
            # Equirectangular procedural Earth map (width=1024, height=512)
            eh, ew = 360, 720
            continents = _fbm_2d(eh, ew, base_grid=6, octaves=5, seed=512)
            moisture = _fbm_2d(eh, ew, base_grid=9, octaves=4, seed=818)
            clouds = _fbm_2d(eh, ew, base_grid=11, octaves=5, seed=991)
            lat = np.linspace(-1.0, 1.0, eh, dtype=np.float32)[:, None]

            tex = np.zeros((eh, ew, 3), dtype=np.float32)
            # Deep & shallow oceans
            shallow = np.clip((continents - 0.36) / 0.16, 0.0, 1.0)
            ocean = (
                np.array([10.0, 44.0, 112.0])[None, None, :] * (1.0 - shallow[..., None])
                + np.array([24.0, 108.0, 178.0])[None, None, :] * shallow[..., None]
            )
            # Lush green forests + golden desert + mountain snow
            desert = np.clip((0.52 - moisture) * 2.6, 0.0, 1.0)
            land = (
                np.array([38.0, 128.0, 62.0])[None, None, :] * (1.0 - desert[..., None])
                + np.array([186.0, 158.0, 96.0])[None, None, :] * desert[..., None]
            )
            is_land = continents > 0.515
            tex = np.where(is_land[..., None], land, ocean)

            # Polar ice caps
            polar = np.clip((np.abs(lat) - 0.76) * 5.0 + (continents - 0.5) * 0.4, 0.0, 1.0)
            tex = tex * (1.0 - polar[..., None]) + 242.0 * polar[..., None]

            # Swirling atmospheric clouds
            cloud_alpha = np.clip((clouds - 0.53) * 2.5, 0.0, 0.88)
            tex = tex * (1.0 - cloud_alpha[..., None]) + 250.0 * cloud_alpha[..., None]
            self._earth_surface = tex

        pad = int(diameter * 0.32)
        size = diameter + 2 * pad
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cx = cy = (size - 1) * 0.5
        r = diameter * 0.5
        nx = (xx - cx) / max(r, 1.0)
        ny = (yy - cy) / max(r, 1.0)
        r2 = nx * nx + ny * ny
        inside = r2 <= 1.0

        nz = np.sqrt(np.clip(1.0 - r2, 0.0, 1.0))
        lon = np.arctan2(nx, nz) + math.radians(rotation_deg * 3.0)
        lat_ang = np.arcsin(np.clip(ny, -1.0, 1.0))

        eh, ew, _ = self._earth_surface.shape
        u_idx = (np.mod((lon / math.tau) + 0.5, 1.0) * (ew - 1)).astype(np.int32)
        v_idx = (np.clip((lat_ang / math.pi) + 0.5, 0.0, 1.0) * (eh - 1)).astype(np.int32)

        albedo = self._earth_surface[v_idx, u_idx]

        # Directional sunlight from upper-left-front + subtle night ambient + Fresnel rim
        lx, ly, lz = -0.48, -0.30, 0.82
        ndotl = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        shade = 0.08 + 1.06 * (ndotl ** 0.82)
        fresnel = ((1.0 - nz) ** 3.2) * (0.35 + 0.65 * ndotl)

        rgb = albedo * shade[..., None] + np.array([78.0, 168.0, 255.0]) * fresnel[..., None] * 0.85

        rgba = np.zeros((size, size, 4), dtype=np.float32)

        # Atmospheric outer halo + scripted glow effect (`add effect glow ... color blue`)
        dist_norm = np.sqrt(r2)
        base_atmo = np.exp(-np.clip(dist_norm - 0.98, 0.0, 4.0) * (9.5 - 4.8 * glow)) * (0.36 + 0.58 * glow)
        outside = ~inside
        gc = np.array(glow_color, dtype=np.float32)
        rgba[outside, :3] = gc
        rgba[outside, 3] = np.clip(base_atmo[outside] * 255.0, 0.0, 255.0)

        # Anti-aliased limb edge + planet body
        edge_aa = np.clip((1.0 - dist_norm) * r * 0.65 + 0.5, 0.0, 1.0)
        rgba[inside, :3] = np.clip(rgb[inside], 0.0, 255.0)
        rgba[inside, 3] = edge_aa[inside] * 255.0

        return Image.fromarray(np.clip(rgba, 0.0, 255.0).astype(np.uint8), mode="RGBA")

    def render_eagle_rgba(
        self,
        span_px: int,
        flap_phase: float,
        opacity: float,
        shadow: float,
    ) -> Image.Image:
        """Render a soaring golden eagle silhouette with gentle wing motion & shadow."""
        w = max(48, int(span_px * 1.4))
        h = max(36, int(span_px * 0.9))
        flap = math.sin(flap_phase) * (h * 0.09)

        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        cx, cy = w * 0.5, h * 0.54
        s = span_px * 0.42

        wing_poly = [
            (cx, cy - s * 0.10),                        # head / beak tip
            (cx + s * 0.14, cy - s * 0.05),             # right shoulder
            (cx + s * 0.58, cy - s * 0.30 - flap),      # right mid-wing
            (cx + s * 0.96, cy - s * 0.16 - flap * 1.3),# right wingtip
            (cx + s * 0.62, cy + s * 0.02 - flap * 0.5),# right trailing edge
            (cx + s * 0.18, cy + s * 0.09),             # body right
            (cx + s * 0.14, cy + s * 0.34),             # tail fan right
            (cx, cy + s * 0.28),                        # tail center
            (cx - s * 0.14, cy + s * 0.34),             # tail fan left
            (cx - s * 0.18, cy + s * 0.09),             # body left
            (cx - s * 0.62, cy + s * 0.02 - flap * 0.5),# left trailing edge
            (cx - s * 0.96, cy - s * 0.16 - flap * 1.3),# left wingtip
            (cx - s * 0.58, cy - s * 0.30 - flap),      # left mid-wing
            (cx - s * 0.14, cy - s * 0.05),             # left shoulder
        ]

        if shadow > 0.01:
            sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            sh_draw = ImageDraw.Draw(sh)
            offset_poly = [(px + span_px * 0.045, py + span_px * 0.075) for px, py in wing_poly]
            sh_draw.polygon(offset_poly, fill=(14, 16, 26, int(115 * opacity * shadow)))
            sh = sh.filter(ImageFilter.GaussianBlur(radius=max(2.0, span_px * 0.03)))
            layer = Image.alpha_composite(layer, sh)

        bird = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        b_draw = ImageDraw.Draw(bird)
        b_draw.polygon(wing_poly, fill=(38, 28, 24, int(245 * opacity)))
        layer = Image.alpha_composite(layer, bird)
        return layer


# ══════════════════════════════════════════════════════════════════════════════
# VIDEX VIDEO ENGINE (INTERPRETER + MP4 ENCODER)
# ══════════════════════════════════════════════════════════════════════════════


class VidexVideoEngine:
    """Executes a ``VideoProgram`` AST and outputs an H.264 `.mp4` video file."""

    def __init__(self, *, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or Path.cwd()

    def render(self, program: VideoProgram, output_path: str | Path) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fps = program.fps
        total_frames = max(1, int(round(program.total_duration * fps)))
        assets = ProceduralAssetFactory(program.width, program.height, self.base_dir)

        # 1. Build per-frame SceneState timeline + transition map
        timeline, transitions = self._build_timeline(program, total_frames)

        # 2. Synthesize ambient soundtrack + narration audio cue to temp WAV
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = Path(tmp_dir) / "soundtrack.wav"
            self._write_ambient_audio(
                wav_path=wav_path,
                duration_sec=total_frames / fps,
                fps=fps,
                timeline=timeline,
            )

            # 3. Stream RGB frames directly into H.264 ffmpeg pipe with AAC audio
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
                "-b:a", "160k",
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

                    # Apply inter-second scene transition if active on this frame
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

        # Apply any top-level commands for single-line / blockless VIDEX scripts
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
                # Fresh scene objects when background explicitly changes at keyframe 1
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
                        sec_frames.append(self._sample_keyframes(kf_states, local_f))
                    else:
                        # Step-hold most recent manual frame
                        active = kf_states[0][1]
                        for idx, st in kf_states:
                            if idx <= local_f:
                                active = st
                        sec_frames.append(deepcopy(active))

                if block.transition and prev_sec_end_state is not None:
                    trans_len = min(12, fps // 2)
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
                # Subtle living continuation for remaining duration after scripted seconds
                for local_f in range(1, fps + 1):
                    living = deepcopy(state_cursor)
                    drift = ((sec_num - 1) * fps + local_f) * 0.0018
                    living.cam_pan_x = max(-1.0, min(1.0, living.cam_pan_x + math.sin(drift) * 0.14))
                    for obj in living.objects.values():
                        if "earth" in obj.name.lower():
                            obj.rotation += 0.35 * local_f
                        elif "eagle" in obj.name.lower():
                            obj.x = min(0.90, obj.x + 0.0022 * local_f)
                    state_cursor = living if local_f == fps else state_cursor
                    timeline.append(living)
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
                # Entering a new environment clears prior scene objects cleanly
                state.objects = {}
                state.cam_pan_x = 0.0
                state.cam_tilt_y = 0.0
            state.background = new_bg
            state.background_file = p.get("file") if isinstance(p.get("file"), str) else None

        elif act == "set_background_time":
            state.time_of_day = str(p["time"])

        elif act == "set_background_weather":
            state.weather = str(p["weather"])

        elif act == "set_background_season":
            state.season = str(p["season"])

        elif act == "place_object":
            name = str(p["name"])
            ax, ay = anchor_normalized(str(p["position"]))
            state.objects[name] = ObjectState(
                name=name,
                x=ax,
                y=ay,
                file=p.get("file") if isinstance(p.get("file"), str) else None,
            )

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
                    state.objects[name].y = max(0.05, min(0.95, state.objects[name].y + (dy / norm_len) * step * 0.4))
                else:
                    state.objects[name].x, state.objects[name].y = tx, ty

        elif act == "set_object_opacity":
            name = str(p["name"])
            if name in state.objects:
                state.objects[name].opacity = max(0.0, min(1.0, float(p["opacity"])))

        elif act == "set_light_color":
            state.light_color = resolve_rgb(str(p["color"]))

        elif act == "set_light_brightness":
            state.light_brightness = max(0.15, min(1.5, float(p["brightness"])))

        elif act == "camera_pan":
            direction = str(p["direction"]).lower()
            delta = SPEED_DELTA.get(str(p["speed"]).lower(), 0.5)
            state.cam_pan_x = max(-1.0, min(1.0, state.cam_pan_x + (-delta if direction == "left" else delta)))

        elif act == "camera_tilt":
            direction = str(p["direction"]).lower()
            delta = SPEED_DELTA.get(str(p["speed"]).lower(), 0.5)
            state.cam_tilt_y = max(-1.0, min(1.0, state.cam_tilt_y + (delta if direction == "down" else -delta)))

        elif act == "place_text":
            tx, ty = anchor_normalized(str(p["position"]))
            state.text = TextState(
                content=str(p["content"]),
                x=tx,
                y=ty,
                alpha=1.0,
                appear_global_frame=global_frame,
            )

        elif act == "set_text_color":
            state.text.color = resolve_rgb(str(p["color"]))

        elif act == "set_text_size":
            raw_sz = str(p["size"]).lower()
            if raw_sz in ("small", "medium", "large"):
                state.text.size = {"small": 32.0, "medium": 50.0, "large": 74.0}[raw_sz]
            else:
                state.text.size = float(p["size"])

        elif act == "fade_text_in":
            state.text.fade_frames = max(1, int(p["frames"]))

        elif act == "add_text_effect":
            if str(p["effect"]).lower() == "glow":
                state.text.glow = 1.0
                state.text.glow_color = resolve_rgb(str(p["color"]))

        elif act == "add_effect":
            target = p.get("object")
            effect = str(p["effect"]).lower()
            if isinstance(target, str) and target in state.objects:
                obj = state.objects[target]
                if effect == "glow":
                    obj.glow = 1.0
                    obj.glow_color = resolve_rgb(str(p["color"]))
                elif effect == "shadow":
                    obj.shadow = 1.0

        elif act == "narrate":
            state.narration = str(p["text"])
            state.narration_start_frame = global_frame

    # ── Frame Compositor ──────────────────────────────────────────────────────

    def _render_single_frame(
        self,
        *,
        program: VideoProgram,
        assets: ProceduralAssetFactory,
        state: SceneState,
        global_frame: int,
        total_frames: int,
    ) -> np.ndarray:
        w, h = program.width, program.height
        bg_canvas = assets.get_background(
            state.background, state.time_of_day, state.background_file
        )
        ch, cw, _ = bg_canvas.shape
        margin_x = (cw - w) * 0.5
        margin_y = (ch - h) * 0.5

        x0 = int(round(max(0.0, min(cw - w, margin_x + state.cam_pan_x * margin_x))))
        y0 = int(round(max(0.0, min(ch - h, margin_y + state.cam_tilt_y * margin_y))))
        cropped = bg_canvas[y0 : y0 + h, x0 : x0 + w].copy()

        # Apply lighting color temperature & subtle exposure shaping
        light_rgb = np.array(state.light_color, dtype=np.float32) / 255.0
        tint_mix = 0.72 + 0.28 * light_rgb
        exposure = 0.62 + 0.52 * state.light_brightness
        lit = np.clip(cropped * tint_mix[None, None, :] * exposure, 0.0, 255.0).astype(np.uint8)

        frame_img = Image.fromarray(lit, mode="RGB").convert("RGBA")

        # Render placed objects with parallax shift relative to camera
        cam_px_shift = -state.cam_pan_x * (w * 0.045)
        cam_py_shift = -state.cam_tilt_y * (h * 0.045)

        for obj in state.objects.values():
            cx = int(round(obj.x * w + cam_px_shift))
            cy = int(round(obj.y * h + cam_py_shift))
            name_low = obj.name.lower()

            if obj.file:
                cand = Path(obj.file) if Path(obj.file).is_absolute() else (self.base_dir / obj.file)
                if cand.is_file():
                    sprite = Image.open(cand).convert("RGBA")
                    target_dim = max(32, int(min(w, h) * 0.30 * obj.scale))
                    sprite.thumbnail((target_dim, target_dim), Image.Resampling.LANCZOS)
                else:
                    sprite = self._render_badge_object(obj, min(w, h))
            elif "earth" in name_low or "planet" in name_low:
                diameter = max(48, int(min(w, h) * 0.36 * obj.scale))
                sprite = assets.render_earth_rgba(
                    diameter=diameter,
                    rotation_deg=obj.rotation + global_frame * 0.45,
                    glow=obj.glow,
                    glow_color=obj.glow_color,
                )
            elif "eagle" in name_low or "bird" in name_low:
                span = max(44, int(min(w, h) * 0.19 * obj.scale))
                sprite = assets.render_eagle_rgba(
                    span_px=span,
                    flap_phase=global_frame * 0.34,
                    opacity=obj.opacity,
                    shadow=obj.shadow,
                )
            else:
                sprite = self._render_badge_object(obj, min(w, h))

            px = cx - sprite.width // 2
            py = cy - sprite.height // 2
            frame_img.alpha_composite(sprite, dest=(px, py))

        # Render scene typography (`"The Himalayas"` with fade-in + outer glow)
        if state.text.content:
            elapsed = max(0, global_frame - state.text.appear_global_frame + 1)
            fade_alpha = min(1.0, elapsed / max(1, state.text.fade_frames))
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

        # Render sleek narration caption pill when `narrate "..."` is active
        if state.narration and state.narration_start_frame >= 0:
            n_elapsed = max(0, global_frame - state.narration_start_frame)
            n_alpha = min(1.0, (n_elapsed + 3) / 8.0)
            if total_frames - global_frame < 10:
                n_alpha *= (total_frames - global_frame) / 10.0
            if n_alpha > 0.02:
                self._draw_narration_bar(
                    frame_img=frame_img,
                    narration=f'“{state.narration}”',
                    alpha=n_alpha,
                )

        return np.asarray(frame_img.convert("RGB"), dtype=np.uint8)

    @staticmethod
    def _render_badge_object(obj: ObjectState, ref_dim: int) -> Image.Image:
        radius = max(24, int(ref_dim * 0.11 * obj.scale))
        size = radius * 3
        canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        c = size // 2
        if obj.glow > 0.01:
            glow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            g_draw = ImageDraw.Draw(glow_layer)
            r, g, b = obj.glow_color
            g_draw.ellipse(
                (c - radius - 14, c - radius - 14, c + radius + 14, c + radius + 14),
                fill=(r, g, b, int(145 * obj.opacity)),
            )
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=14))
            canvas = Image.alpha_composite(canvas, glow_layer)
            draw = ImageDraw.Draw(canvas)

        draw.ellipse(
            (c - radius, c - radius, c + radius, c + radius),
            fill=(88, 152, 236, int(235 * obj.opacity)),
            outline=(255, 255, 255, int(220 * obj.opacity)),
            width=3,
        )
        font = load_font(max(14, radius // 3))
        bbox = draw.textbbox((0, 0), obj.name, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (c - tw / 2, c - th / 2),
            obj.name,
            font=font,
            fill=(255, 255, 255, int(245 * obj.opacity)),
        )
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
        tx = int(cx - tw * 0.5)
        ty = int(cy - th * 0.5)

        if glow > 0.01:
            glow_img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            g_draw = ImageDraw.Draw(glow_img)
            gr, gg, gb = glow_color
            g_draw.text((tx, ty), text, font=font, fill=(gr, gg, gb, int(220 * alpha * glow)))
            glow_img = glow_img.filter(ImageFilter.GaussianBlur(radius=max(6.0, font_px * 0.22)))
            frame_img.alpha_composite(glow_img)

        # Subtle contrast shadow for crisp legibility over bright snow/sky
        draw.text((tx + 2, ty + 3), text, font=font, fill=(10, 14, 24, int(150 * alpha)))
        r, g, b = color
        draw.text((tx, ty), text, font=font, fill=(r, g, b, int(255 * alpha)))
        frame_img.alpha_composite(overlay)

    @staticmethod
    def _draw_narration_bar(frame_img: Image.Image, narration: str, alpha: float) -> None:
        w, h = frame_img.size
        font = load_font(max(16, int(h * 0.026)), serif=False)
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        bbox = draw.textbbox((0, 0), narration, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad_x, pad_y = int(w * 0.022), int(h * 0.012)
        cx, cy = w // 2, int(h * 0.935)
        rect = (
            cx - tw // 2 - pad_x,
            cy - th // 2 - pad_y,
            cx + tw // 2 + pad_x,
            cy + th // 2 + pad_y,
        )
        draw.rounded_rectangle(
            rect,
            radius=max(8, int(h * 0.018)),
            fill=(12, 18, 30, int(172 * alpha)),
            outline=(255, 255, 255, int(75 * alpha)),
            width=1,
        )
        draw.text(
            (cx - tw // 2, cy - th // 2 - 2),
            narration,
            font=font,
            fill=(242, 246, 255, int(245 * alpha)),
        )
        frame_img.alpha_composite(overlay)

    @staticmethod
    def _blend_transition(
        prev_rgb: np.ndarray,
        curr_rgb: np.ndarray,
        trans_type: str,
        alpha: float,
    ) -> np.ndarray:
        a = _smoothstep(alpha)
        if "wipe" in trans_type:
            split = int(curr_rgb.shape[1] * a)
            out = prev_rgb.copy()
            out[:, :split] = curr_rgb[:, :split]
            return out
        return np.clip(
            prev_rgb.astype(np.float32) * (1.0 - a) + curr_rgb.astype(np.float32) * a,
            0.0,
            255.0,
        ).astype(np.uint8)

    @staticmethod
    def _write_ambient_audio(
        *,
        wav_path: Path,
        duration_sec: float,
        fps: int,
        timeline: list[SceneState],
    ) -> None:
        """Synthesize a warm cinematic ambient pad + gentle narration chime."""
        sr = 44100
        n_samples = max(1, int(round(duration_sec * sr)))
        t = np.linspace(0.0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

        # Warm fifth + major ninth ambient pad (C3, G3, D4, E4) with slow breathing LFO
        lfo = 0.72 + 0.28 * np.sin(2.0 * math.pi * 0.18 * t)
        left = (
            0.30 * np.sin(2.0 * math.pi * 130.81 * t)
            + 0.22 * np.sin(2.0 * math.pi * 196.00 * t + 0.3)
            + 0.14 * np.sin(2.0 * math.pi * 293.66 * t)
        ) * lfo
        right = (
            0.30 * np.sin(2.0 * math.pi * 130.81 * t)
            + 0.22 * np.sin(2.0 * math.pi * 196.00 * t - 0.3)
            + 0.14 * np.sin(2.0 * math.pi * 329.63 * t)
        ) * lfo

        # Add soft crystalline bell arpeggio when `narrate "..."` triggers
        narrate_frame = next(
            (i for i, st in enumerate(timeline) if st.narration), -1
        )
        if narrate_frame >= 0:
            cue_time = narrate_frame / max(fps, 1)
            for idx, freq in enumerate((523.25, 659.25, 783.99)):
                onset = cue_time + idx * 0.16
                dt = np.maximum(0.0, t - onset)
                bell_env = (t >= onset).astype(np.float32) * np.exp(-dt * 2.4)
                bell = 0.16 * np.sin(2.0 * math.pi * freq * dt) * bell_env
                left += bell
                right += bell * 0.92

        # Smooth fade-in and fade-out envelopes
        fade_len = min(int(0.45 * sr), n_samples // 4)
        env = np.ones(n_samples, dtype=np.float32)
        if fade_len > 1:
            ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            env[:fade_len] *= ramp
            env[-fade_len:] *= ramp[::-1]

        stereo = np.stack([left * env, right * env], axis=-1)
        pcm16 = np.clip(stereo * 0.42 * 32767.0, -32767.0, 32767.0).astype(np.int16)

        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
