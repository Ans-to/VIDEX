"""
VIDEX Motion Graphics Vector, Physics & Kinetic Typography Engine
=================================================================

Provides pure analytical Signed Distance Field (SDF) sub-pixel vector rendering,
damped harmonic spring + exponential bezier motion curves, multi-subframe 180-deg
shutter motion blur, masked kinetic typography, broadcast motion-graphics transitions
(whip-pan shutter push, 3-ribbon angled stinger wipe, circular iris portal match-cut,
3D perspective card flip), and 120-BPM frame-accurate audio-visual synthesis
(at VIDEX's standard 26 fps, 120 BPM = 2.0 beats/sec = EXACTLY 13.000 frames/beat!).
"""

from __future__ import annotations

import math
from pathlib import Path
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ══════════════════════════════════════════════════════════════════════════════
# 1. MOTION PHYSICS & EASING CURVES
# ══════════════════════════════════════════════════════════════════════════════


def clamp01(t: float) -> float:
    return 0.0 if t <= 0.0 else (1.0 if t >= 1.0 else t)


def ease_linear(t: float) -> float:
    return clamp01(t)


def ease_quintic_c2(t: float) -> float:
    """Zero velocity & zero acceleration at both endpoints (6t^5 - 15t^4 + 10t^3)."""
    u = clamp01(t)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def ease_expo_snap(t: float) -> float:
    """Explosive launch with long silky magnetic settle (exact 0 at t=0, 1 at t=1)."""
    u = clamp01(t)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    norm = 1.0 - math.pow(2.0, -10.5)
    return (1.0 - math.pow(2.0, -10.5 * u)) / norm


def ease_whip_in_out(t: float) -> float:
    """Steep symmetric exponential S-curve whose peak velocity hits at t = 0.5."""
    u = clamp01(t)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    if u < 0.5:
        return 0.5 * math.pow(2.0, 16.0 * (u - 0.5))
    return 1.0 - 0.5 * math.pow(2.0, -16.0 * (u - 0.5))


def ease_spring(t: float, omega: float = 15.5, zeta: float = 0.56) -> float:
    """
    Closed-form underdamped harmonic spring response with gentle anticipation
    wind-up and exact pin to 1.0 at t >= 1.0.
    """
    u = clamp01(t)
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    wd = omega * math.sqrt(max(1.0 - zeta * zeta, 1e-4))
    env = math.exp(-zeta * omega * u)
    val = 1.0 - env * (
        math.cos(wd * u) + (zeta * omega / wd) * math.sin(wd * u)
    )
    # Smoothly pin final 8% window to exact 1.0 so resting state has zero micro-jitter
    tail = clamp01((u - 0.90) / 0.10)
    return val * (1.0 - tail) + 1.0 * tail


# ══════════════════════════════════════════════════════════════════════════════
# 2. MOGRAPH THEME DETECTION & SHARED 120-BPM AUDIO-VISUAL ENVELOPES
# ══════════════════════════════════════════════════════════════════════════════

MOGRAPH_STAGE_KEYWORDS: tuple[str, ...] = (
    "blueprint",
    "motion grid",
    "mograph",
    "neon cyber",
    "orbital stage",
    "editorial light",
    "kinetic paper",
    "swiss grid",
    "obsidian stage",
    "audio stage",
    "dark studio",
)

MOGRAPH_OBJECT_KEYWORDS: tuple[str, ...] = (
    "physics lanes",
    "orbital rig",
    "kinetic cascade",
    "equalizer",
    "waveform",
    "orb",
    "ring",
    "arc",
    "prism",
    "hexagon",
    "capsule",
    "pill",
    "square",
    "diamond",
    "pulse wave",
    "reticle",
)


def is_mograph_scene(background_name: str) -> bool:
    low = background_name.strip().lower()
    return any(k in low for k in MOGRAPH_STAGE_KEYWORDS)


def is_light_mograph_scene(background_name: str) -> bool:
    low = background_name.strip().lower()
    return any(k in low for k in ("editorial", "paper", "swiss", "light", "ivory"))


def is_mograph_object(name: str) -> bool:
    low = name.strip().lower()
    return any(k in low for k in MOGRAPH_OBJECT_KEYWORDS)


def mograph_audio_envelopes(time_s: float) -> tuple[float, float, float]:
    """
    Shared deterministic 120-BPM (2.0 Hz = exact 13 frames/beat @ 26 fps!)
    envelopes used BOTH by the 44.1 kHz synthesizer AND the visual beat rigs
    for 100.000% zero-drift audio-visual synchronization!

    Returns: `(kick_env, snare_env, arp_env)` in `[0.0, 1.0]`.
    """
    beat_pos = time_s * 2.0  # 120 BPM -> 2.0 beats per second
    frac_beat = beat_pos - math.floor(beat_pos)
    beat_idx = int(math.floor(beat_pos))

    # Crisp exponential kick decay on every beat
    kick_env = math.exp(-frac_beat * 7.2)

    # Snare / clap accent on backbeats (beats 1, 3, 5, ...)
    if beat_idx % 2 == 1:
        snare_env = math.exp(-frac_beat * 9.5)
    else:
        snare_env = 0.0

    # 16th-note arpeggiator shimmer (8 notes per second = 4 per beat)
    sixteenth_frac = (beat_pos * 4.0) - math.floor(beat_pos * 4.0)
    arp_env = math.exp(-sixteenth_frac * 5.2)
    return kick_env, snare_env, arp_env


# ══════════════════════════════════════════════════════════════════════════════
# 3. ANALYTICAL SUB-PIXEL SIGNED DISTANCE FIELD (SDF) VECTOR PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════
# Every shape evaluates exact float32 Euclidean signed distance on its tight ROI
# bounding box `[y0:y1, x0:x1]`, giving true sub-pixel positioning (0.01px smooth
# glides), razor-sharp 1.0px anti-aliasing, zero rotation blur, and optional glow!


def _roi_bounds(
    w: int, h: int, cx: float, cy: float, half_ext_x: float, half_ext_y: float
) -> tuple[int, int, int, int] | None:
    x0 = max(0, int(math.floor(cx - half_ext_x)))
    x1 = min(w, int(math.ceil(cx + half_ext_x)) + 1)
    y0 = max(0, int(math.floor(cy - half_ext_y)))
    y1 = min(h, int(math.ceil(cy + half_ext_y)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _composite_sdf_roi(
    canvas: np.ndarray,
    bounds: tuple[int, int, int, int],
    sdf: np.ndarray,
    color_rgb: tuple[float, float, float],
    alpha: float,
    glow_px: float = 0.0,
    glow_color: tuple[float, float, float] | None = None,
) -> None:
    if alpha <= 0.002:
        return
    x0, y0, x1, y1 = bounds
    roi = canvas[y0:y1, x0:x1]

    if glow_px > 0.25:
        gc = glow_color if glow_color is not None else color_rgb
        glow_falloff = np.exp(-np.maximum(sdf, 0.0) / max(glow_px, 0.5)) * (0.42 * alpha)
        g_col = np.array(gc, dtype=np.float32)[None, None, :]
        roi += g_col * glow_falloff[..., None]

    # Exact 1.1px analytical pixel coverage ramp (smooth across fractional pixel centers!)
    cov = np.clip(0.5 - sdf / 1.15, 0.0, 1.0) * alpha
    c_arr = np.array(color_rgb, dtype=np.float32)[None, None, :]
    roi[:] = roi * (1.0 - cov[..., None]) + c_arr * cov[..., None]


def draw_sdf_circle(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    color_rgb: tuple[float, float, float],
    *,
    alpha: float = 1.0,
    stroke_w: float = 0.0,
    squash_x: float = 1.0,
    squash_y: float = 1.0,
    angle_deg: float = 0.0,
    glow_px: float = 0.0,
    glow_color: tuple[float, float, float] | None = None,
) -> None:
    h, w, _ = canvas.shape
    pad = max(glow_px * 3.2, 4.0) + stroke_w
    ext = radius * max(squash_x, squash_y) + pad
    b = _roi_bounds(w, h, cx, cy, ext, ext)
    if b is None:
        return
    x0, y0, x1, y1 = b
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    if abs(angle_deg) > 0.01:
        rad = math.radians(-angle_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        rx = dx * ca - dy * sa
        ry = dx * sa + dy * ca
    else:
        rx, ry = dx, dy

    # Elliptical distance normalization for area-preserving velocity squash & stretch
    sx = max(squash_x, 0.25)
    sy = max(squash_y, 0.25)
    q_len = np.hypot(rx / sx, ry / sy)
    avg_scale = math.sqrt(sx * sy)
    sdf = (q_len - radius) * avg_scale
    if stroke_w > 0.0:
        sdf = np.abs(sdf) - stroke_w * 0.5
    _composite_sdf_roi(canvas, b, sdf, color_rgb, alpha, glow_px, glow_color)


def draw_sdf_capsule_segment(
    canvas: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
    radius: float,
    color_rgb: tuple[float, float, float],
    *,
    alpha: float = 1.0,
    glow_px: float = 0.0,
    glow_color: tuple[float, float, float] | None = None,
) -> None:
    h, w, _ = canvas.shape
    pad = max(glow_px * 3.0, 4.0) + radius
    cx, cy = 0.5 * (ax + bx), 0.5 * (ay + by)
    hx = 0.5 * abs(bx - ax) + pad
    hy = 0.5 * abs(by - ay) + pad
    b = _roi_bounds(w, h, cx, cy, hx, hy)
    if b is None:
        return
    x0, y0, x1, y1 = b
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    pax = xx - ax
    pay = yy - ay
    bax = bx - ax
    bay = by - ay
    denom = max(bax * bax + bay * bay, 1e-5)
    proj = np.clip((pax * bax + pay * bay) / denom, 0.0, 1.0)
    sdf = np.hypot(pax - bax * proj, pay - bay * proj) - radius
    _composite_sdf_roi(canvas, b, sdf, color_rgb, alpha, glow_px, glow_color)


def draw_sdf_round_rect(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    half_w: float,
    half_h: float,
    corner_r: float,
    color_rgb: tuple[float, float, float],
    *,
    angle_deg: float = 0.0,
    alpha: float = 1.0,
    stroke_w: float = 0.0,
    glow_px: float = 0.0,
    glow_color: tuple[float, float, float] | None = None,
) -> None:
    h, w, _ = canvas.shape
    diag = math.hypot(half_w, half_h) + max(glow_px * 3.0, 4.0) + stroke_w
    b = _roi_bounds(w, h, cx, cy, diag, diag)
    if b is None:
        return
    x0, y0, x1, y1 = b
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    if abs(angle_deg) > 0.01:
        rad = math.radians(-angle_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        rx = dx * ca - dy * sa
        ry = dx * sa + dy * ca
    else:
        rx, ry = dx, dy

    cr = min(corner_r, half_w, half_h)
    qx = np.abs(rx) - (half_w - cr)
    qy = np.abs(ry) - (half_h - cr)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    sdf = outside + inside - cr
    if stroke_w > 0.0:
        sdf = np.abs(sdf) - stroke_w * 0.5
    _composite_sdf_roi(canvas, b, sdf, color_rgb, alpha, glow_px, glow_color)


def draw_sdf_regular_polygon(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    n_sides: int,
    color_rgb: tuple[float, float, float],
    *,
    angle_deg: float = 0.0,
    alpha: float = 1.0,
    stroke_w: float = 0.0,
    glow_px: float = 0.0,
    glow_color: tuple[float, float, float] | None = None,
) -> None:
    """Exact Inigo Quilez signed distance to a regular N-gon at arbitrary rotation."""
    h, w, _ = canvas.shape
    ext = radius + max(glow_px * 3.0, 5.0) + stroke_w
    b = _roi_bounds(w, h, cx, cy, ext, ext)
    if b is None:
        return
    x0, y0, x1, y1 = b
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    rad = math.radians(-angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    px = dx * ca - dy * sa
    py = dx * sa + dy * ca

    an = math.pi / max(3, n_sides)
    en_x, en_y = math.cos(an), math.sin(an)
    bn = np.mod(np.arctan2(px, py), 2.0 * an) - an
    plen = np.hypot(px, py)
    qx = plen * np.cos(bn)
    qy = np.abs(plen * np.sin(bn))
    qx -= radius * en_x
    qy -= radius * en_y
    qy_clamped = qy + np.minimum(-qy, 0.0)
    sdf = np.hypot(qx, np.maximum(qy_clamped, 0.0)) * np.sign(qx)
    # Exact interior distance fix when qx <= 0
    sdf = np.where(qx > 0.0, np.hypot(qx, np.maximum(qy, 0.0)), qx)
    if stroke_w > 0.0:
        sdf = np.abs(sdf) - stroke_w * 0.5
    _composite_sdf_roi(canvas, b, sdf, color_rgb, alpha, glow_px, glow_color)


def draw_sdf_ring_arc(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    thickness: float,
    start_deg: float,
    sweep_deg: float,
    color_rgb: tuple[float, float, float],
    *,
    alpha: float = 1.0,
    glow_px: float = 0.0,
    glow_color: tuple[float, float, float] | None = None,
) -> None:
    """
    Exact analytical SDF Trim-Path circular arc with rounded end-caps at ANY
    fractional radius, thickness, start angle, and sweep angle (`0..360 deg`).
    """
    sweep = max(0.0, min(360.0, sweep_deg))
    if sweep <= 0.15 or alpha <= 0.002:
        return
    if sweep >= 359.6:
        draw_sdf_circle(
            canvas,
            cx,
            cy,
            radius,
            color_rgb,
            alpha=alpha,
            stroke_w=thickness,
            glow_px=glow_px,
            glow_color=glow_color,
        )
        return

    h, w, _ = canvas.shape
    half_t = thickness * 0.5
    ext = radius + half_t + max(glow_px * 3.0, 5.0)
    b = _roi_bounds(w, h, cx, cy, ext, ext)
    if b is None:
        return
    x0, y0, x1, y1 = b
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy

    # Rotate coordinate system so arc is symmetric around +Y axis with aperture `ap`
    mid_rad = math.radians(start_deg + sweep * 0.5 + 90.0)
    ca, sa = math.cos(-mid_rad), math.sin(-mid_rad)
    px = np.abs(dx * ca - dy * sa)
    py = dx * sa + dy * ca

    ap = math.radians(sweep * 0.5)
    sc_x, sc_y = math.sin(ap), math.cos(ap)
    r_len = np.hypot(px, py)
    ring_dist = np.abs(r_len - radius)
    cap_dist = np.hypot(px - radius * sc_x, py - radius * sc_y)
    in_sector = (sc_y * px > sc_x * py)
    sdf = np.where(in_sector, cap_dist, ring_dist) - half_t
    _composite_sdf_roi(canvas, b, sdf, color_rgb, alpha, glow_px, glow_color)


# ══════════════════════════════════════════════════════════════════════════════
# 4. FONT LOADER & MASKED BASELINE-CLIPPED KINETIC TYPOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════


def _font(px: int, *, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    paths = (
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
        if mono
        else [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for p in paths:
        if Path(p).is_file():
            return ImageFont.truetype(p, size=max(11, int(round(px))))
    return ImageFont.load_default()


def draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    cx: float,
    cy: float,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int, int],
    tracking_px: float = 0.0,
) -> tuple[float, float]:
    if not text:
        return 0.0, 0.0
    advs = [draw.textlength(ch, font=font) for ch in text]
    total_w = sum(advs) + tracking_px * max(len(text) - 1, 0)
    bbox = draw.textbbox((0, 0), text, font=font)
    th = bbox[3] - bbox[1]
    cursor_x = cx - total_w * 0.5
    ty = cy - th * 0.5 - bbox[1] * 0.35
    for ch, adv in zip(text, advs):
        draw.text((cursor_x, ty), ch, font=font, fill=fill)
        cursor_x += adv + tracking_px
    return total_w, float(th)


def render_kinetic_text_layer(
    frame_img: Image.Image,
    *,
    text: str,
    cx: float,
    cy: float,
    font_px: int,
    color: tuple[int, int, int],
    progress: float,
    light_stage: bool = False,
    accent_color: tuple[int, int, int] | None = None,
) -> None:
    """
    Broadcast After-Effects Masked Kinetic Typography Reveal:
      - Razor horizontal clip plane: characters slide smoothly UP through an
        invisible mask (`y_offset = (1 - expo) * 1.15 * height`) with zero
        blurry drop-shadow mud!
      - Dynamic tracking tighten (`+9px -> +2.5px`) as the title locks in.
      - Animated hairline accent bar sweeping outward beneath display titles.
    """
    u = clamp01(progress)
    if u <= 0.005:
        return
    w, h = frame_img.size
    eased = ease_expo_snap(u)
    tracking = (2.2 + (1.0 - eased) * 7.5) * (font_px / 64.0)
    font = _font(font_px, mono=False)

    # Render onto local clipped strip so the bottom mask cut is 100% razor-clean!
    strip_h = int(font_px * 1.65)
    strip_w = w
    strip = Image.new("RGBA", (strip_w, strip_h), (0, 0, 0, 0))
    s_draw = ImageDraw.Draw(strip)

    # Automatic contrast guard if white text was left on light editorial stage
    r, g, b = color
    if light_stage and (r + g + b) > 650:
        r, g, b = (15, 23, 42)

    rise_offset = (1.0 - eased) * (strip_h * 0.88)
    alpha_i = int(round(255.0 * min(1.0, u * 3.2)))
    tw, _ = draw_tracked_text(
        s_draw,
        text,
        cx=cx,
        cy=strip_h * 0.50 + rise_offset,
        font=font,
        fill=(r, g, b, alpha_i),
        tracking_px=tracking,
    )

    paste_y = int(round(cy - strip_h * 0.50))
    frame_img.alpha_composite(strip, dest=(0, paste_y))

    # Crisp animated kinetic accent rule under major display headlines
    if font_px >= 46 and tw > 20:
        bar_u = ease_expo_snap(clamp01((u - 0.14) / 0.86))
        if bar_u > 0.01:
            bar_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            b_draw = ImageDraw.Draw(bar_overlay)
            ac = accent_color or ((37, 99, 235) if light_stage else (56, 189, 248))
            half_bar = min(tw * 0.34, w * 0.16) * bar_u
            by = int(round(cy + font_px * 0.68))
            b_draw.rounded_rectangle(
                (cx - half_bar, by - 2, cx + half_bar, by + 2),
                radius=2,
                fill=(ac[0], ac[1], ac[2], int(235 * min(1.0, u * 2.5))),
            )
            frame_img.alpha_composite(bar_overlay)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MOTION GRAPHICS STUDIO STAGE CANVASES & SET-PIECE CHOREOGRAPHY
# ══════════════════════════════════════════════════════════════════════════════


class MoGraphStudio:
    """High-precision vector stage renderer & parametric motion-graphics rigs."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        self.xx = xx
        self.yy = yy
        self.nx = (xx / max(width - 1, 1)) * 2.0 - 1.0
        self.ny = (yy / max(height - 1, 1)) * 2.0 - 1.0

    # ── Stage Backgrounds ────────────────────────────────────────────────────

    def render_stage_canvas(
        self,
        theme: str,
        global_frame: int,
        fps: int,
        cam_pan_x: float,
        cam_tilt_y: float,
        cam_zoom: float,
    ) -> np.ndarray:
        """Render crisp 100%-fidelity studio vector background with sub-pixel grid."""
        w, h = self.width, self.height
        time_s = global_frame / max(fps, 1)
        kick, snare, _ = mograph_audio_envelopes(time_s)
        low = theme.strip().lower()
        sx = w / 1920.0
        sy = h / 1080.0

        pan_px = cam_pan_x * (w * 0.06)
        tilt_px = cam_tilt_y * (h * 0.06)

        if is_light_mograph_scene(low):
            # ACT III: High-Contrast Swiss Editorial Warm Ivory Studio (#F5F2EB)
            r2 = self.nx ** 2 * 0.65 + self.ny ** 2 * 0.85
            base = np.empty((h, w, 3), dtype=np.float32)
            base[..., 0] = 246.0 - 13.0 * r2
            base[..., 1] = 243.0 - 14.0 * r2
            base[..., 2] = 235.0 - 16.0 * r2

            # Crisp sub-pixel architectural grid lines
            step = 80.0 * sx
            gx = np.abs(np.mod(self.xx + pan_px - w * 0.5 + step * 0.5, step) - step * 0.5)
            gy = np.abs(np.mod(self.yy + tilt_px - h * 0.5 + step * 0.5, step) - step * 0.5)
            grid_cov = np.maximum(np.clip(1.0 - gx / 1.1, 0.0, 1.0), np.clip(1.0 - gy / 1.1, 0.0, 1.0))
            ink_grid = np.array([203.0, 213.0, 225.0], dtype=np.float32)[None, None, :]
            base = base * (1.0 - grid_cov[..., None] * 0.52) + ink_grid * (grid_cov[..., None] * 0.52)
            act_tag = "03 // SWISS EDITORIAL CASCADE"
            is_light = True
        elif "neon" in low or "orbital" in low:
            # ACT II: Deep Cosmic Indigo + Dual Electric Cyan & Violet Radial Pools
            r2 = self.nx ** 2 + self.ny ** 2
            base = np.empty((h, w, 3), dtype=np.float32)
            base[..., 0] = 9.0 + 10.0 * (1.0 - np.clip(r2, 0.0, 1.0))
            base[..., 1] = 10.0 + 14.0 * (1.0 - np.clip(r2, 0.0, 1.0))
            base[..., 2] = 26.0 + 28.0 * (1.0 - np.clip(r2, 0.0, 1.0))

            # Dual ambient color pools pulsing subtly on 120-BPM beat
            d_left = (self.nx + 0.45) ** 2 + (self.ny + 0.15) ** 2
            d_right = (self.nx - 0.45) ** 2 + (self.ny - 0.18) ** 2
            pulse = 1.0 + 0.18 * kick
            base += np.exp(-d_left / 0.55)[..., None] * np.array([12.0, 62.0, 96.0], dtype=np.float32) * pulse
            base += np.exp(-d_right / 0.55)[..., None] * np.array([58.0, 24.0, 96.0], dtype=np.float32) * pulse

            # Polar concentric guide rings
            pr = np.hypot((self.xx - w * 0.5 + pan_px * 0.4), (self.yy - h * 0.5 + tilt_px * 0.4))
            ring_mod = np.abs(np.mod(pr, 130.0 * sx) - 65.0 * sx)
            rcov = np.clip(1.0 - np.abs(ring_mod - 64.0 * sx) / 1.2, 0.0, 1.0)
            base += rcov[..., None] * np.array([28.0, 44.0, 76.0], dtype=np.float32) * 0.55
            act_tag = "02 // TRIM PATHS & ORBITAL HARMONICS"
            is_light = False
        elif "obsidian" in low or "audio" in low:
            # ACT IV: Pure Obsidian Audio-Reactive Stage with Horizon Glow
            base = np.empty((h, w, 3), dtype=np.float32)
            base[..., 0] = 6.0
            base[..., 1] = 9.0
            base[..., 2] = 17.0
            d_mid = (self.nx * 0.72) ** 2 + ((self.ny - 0.05) * 1.25) ** 2
            aura = (0.85 + 0.35 * kick + 0.20 * snare)
            base += np.exp(-d_mid / 0.42)[..., None] * np.array([14.0, 56.0, 92.0], dtype=np.float32) * aura
            base += np.exp(-((self.ny - 0.12) ** 2) / 0.035)[..., None] * np.array(
                [26.0, 20.0, 10.0], dtype=np.float32
            ) * aura
            # Subtle dot matrix grid
            step = 64.0 * sx
            gx = np.mod(self.xx + pan_px * 0.5, step) - step * 0.5
            gy = np.mod(self.yy + tilt_px * 0.5, step) - step * 0.5
            dots = np.clip(1.8 * sx - np.hypot(gx, gy), 0.0, 1.0)
            base += dots[..., None] * np.array([34.0, 52.0, 78.0], dtype=np.float32) * 0.65
            act_tag = "04 // AUDIO-REACTIVE SPECTRUM & FINALE"
            is_light = False
        else:
            # ACT I Default: Precision Blueprint Motion Lab (#070C17)
            r2 = self.nx ** 2 * 0.7 + self.ny ** 2 * 0.9
            base = np.empty((h, w, 3), dtype=np.float32)
            base[..., 0] = 8.0 + 8.0 * (1.0 - np.clip(r2, 0.0, 1.0))
            base[..., 1] = 13.0 + 16.0 * (1.0 - np.clip(r2, 0.0, 1.0))
            base[..., 2] = 24.0 + 28.0 * (1.0 - np.clip(r2, 0.0, 1.0))

            minor = 60.0 * sx
            major = 240.0 * sx
            gx_m = np.abs(np.mod(self.xx + pan_px + minor * 0.5, minor) - minor * 0.5)
            gy_m = np.abs(np.mod(self.yy + tilt_px + minor * 0.5, minor) - minor * 0.5)
            minor_cov = np.maximum(np.clip(1.0 - gx_m / 1.1, 0.0, 1.0), np.clip(1.0 - gy_m / 1.1, 0.0, 1.0))

            gx_M = np.abs(np.mod(self.xx + pan_px + major * 0.5, major) - major * 0.5)
            gy_M = np.abs(np.mod(self.yy + tilt_px + major * 0.5, major) - major * 0.5)
            major_cov = np.maximum(np.clip(1.0 - gx_M / 1.15, 0.0, 1.0), np.clip(1.0 - gy_M / 1.15, 0.0, 1.0))

            base += minor_cov[..., None] * np.array([14.0, 26.0, 44.0], dtype=np.float32) * 0.72
            base += major_cov[..., None] * np.array([24.0, 52.0, 86.0], dtype=np.float32) * 0.65
            act_tag = "01 // MOTION PHYSICS & SHUTTER LAB"
            is_light = False

        # Subtle studio corner registration marks + live frame/beat header bar
        self._draw_stage_chrome(base, act_tag=act_tag, global_frame=global_frame, fps=fps, is_light=is_light, kick=kick)
        return np.clip(base, 0.0, 255.0)

    def _draw_stage_chrome(
        self,
        canvas: np.ndarray,
        *,
        act_tag: str,
        global_frame: int,
        fps: int,
        is_light: bool,
        kick: float,
    ) -> None:
        w, h = self.width, self.height
        sx = w / 1920.0
        line_col = (100.0, 116.0, 139.0) if is_light else (71.0, 98.0, 136.0)
        accent = (244.0, 63.0, 94.0) if is_light else (56.0, 189.0, 248.0)

        # Corner registration crosshairs
        mx, my = 56.0 * sx, 48.0 * (h / 1080.0)
        arm = 11.0 * sx
        for cx, cy in ((mx, my), (w - mx, my), (mx, h - my), (w - mx, h - my)):
            draw_sdf_capsule_segment(canvas, cx - arm, cy, cx + arm, cy, 1.1 * sx, line_col, alpha=0.75)
            draw_sdf_capsule_segment(canvas, cx, cy - arm, cx, cy + arm, 1.1 * sx, line_col, alpha=0.75)

        # Beat pulse LED indicator in top-right corner
        led_x = w - 92.0 * sx
        led_y = 48.0 * (h / 1080.0)
        draw_sdf_circle(
            canvas,
            led_x,
            led_y,
            (4.5 + 3.0 * kick) * sx,
            accent,
            alpha=0.95,
            glow_px=(6.0 + 10.0 * kick) * sx,
        )

        # Crisp mono metadata strip via lightweight top bar overlay
        bar_h = max(36, int(78 * (h / 1080.0)))
        bar = Image.new("RGBA", (w, bar_h), (0, 0, 0, 0))
        b_draw = ImageDraw.Draw(bar)
        mono = _font(int(17 * sx), mono=True)
        txt_col = (30, 41, 59, 215) if is_light else (148, 179, 218, 215)
        sec = global_frame // max(fps, 1)
        ff = global_frame % max(fps, 1)
        left_label = f"VIDEX STUDIO  ·  {act_tag}"
        right_label = f"TC 00:00:{sec:02d}:{ff:02d}  [{fps} FPS]"
        b_draw.text((int(84 * sx), int(36 * (h / 1080.0))), left_label, font=mono, fill=txt_col)
        rb = b_draw.textbbox((0, 0), right_label, font=mono)
        rw = rb[2] - rb[0]
        b_draw.text((int(w - 118 * sx - rw), int(36 * (h / 1080.0))), right_label, font=mono, fill=txt_col)

        bar_arr = np.asarray(bar, dtype=np.float32)
        ba = (bar_arr[..., 3:4] / 255.0)
        canvas[:bar_h] = canvas[:bar_h] * (1.0 - ba) + bar_arr[..., :3] * ba

    # ── Set-Piece 1: 4-Lane Motion Physics & Shutter Blur Benchmark Lab ──────

    def render_physics_lanes(
        self,
        canvas: np.ndarray,
        local_age_s: float,
        fps: int,
        opacity: float,
    ) -> None:
        w, h = self.width, self.height
        sx, sy = w / 1920.0, h / 1080.0

        # Choreograph clean forward shuttle (0.12s -> 1.42s), brief hold, return shuttle (1.72s -> 2.82s)
        def shuttle_param(age: float) -> tuple[float, int]:
            if age < 0.12:
                return 0.0, 1
            if age < 1.42:
                return (age - 0.12) / 1.30, 1
            if age < 1.72:
                return 1.0, 1
            if age < 2.84:
                return (age - 1.72) / 1.12, -1
            return 1.0, -1

        x_left = 455.0 * sx
        x_right = 1425.0 * sx
        span = x_right - x_left

        # Vertical target alignment laser line so Spring overshoot past target is unmistakable!
        draw_sdf_capsule_segment(
            canvas,
            x_right,
            h * 0.195,
            x_right,
            h * 0.675,
            1.3 * sx,
            (94.0, 234.0, 212.0),
            alpha=0.42 * opacity,
        )
        lanes = [
            ("01  LINEAR", "v(t) = const", ease_linear, (56.0, 189.0, 248.0)),
            ("02  CUBIC C²", "6t⁵ - 15t⁴ + 10t³", ease_quintic_c2, (168.0, 85.0, 247.0)),
            ("03  EXPO SNAP", "1 - 2⁻¹⁰·⁵ᵗ", ease_expo_snap, (251.0, 191.0, 36.0)),
            ("04  SPRING ζ=.56", "underdamped harmonic", ease_spring, (251.0, 113.0, 133.0)),
        ]
        lane_ys = [h * (0.235 + idx * 0.132) for idx in range(4)]
        dt_frame = 1.0 / max(fps, 1)

        label_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ldraw = ImageDraw.Draw(label_layer)
        f_title = _font(int(21 * sx), mono=True)
        f_sub = _font(int(15 * sx), mono=True)

        for idx, (title, formula, fn, col) in enumerate(lanes):
            ly = lane_ys[idx]
            # Calibrated rail track + 10% graduation ticks + start/end gates
            draw_sdf_capsule_segment(
                canvas, x_left, ly, x_right, ly, 2.2 * sy, (36.0, 54.0, 82.0), alpha=0.85 * opacity
            )
            for k in range(11):
                tx = x_left + span * (k / 10.0)
                th_tick = 11.0 * sy if k in (0, 5, 10) else 5.5 * sy
                draw_sdf_capsule_segment(
                    canvas,
                    tx,
                    ly - th_tick,
                    tx,
                    ly + th_tick,
                    1.2 * sx,
                    (64.0, 92.0, 130.0),
                    alpha=0.75 * opacity,
                )

            u_raw, direction = shuttle_param(local_age_s)
            prog = fn(u_raw)
            curr_x = (x_left + span * prog) if direction == 1 else (x_right - span * prog)

            # Luminous progress beam behind puck
            anchor_x = x_left if direction == 1 else x_right
            draw_sdf_capsule_segment(
                canvas,
                min(anchor_x, curr_x),
                ly,
                max(anchor_x, curr_x),
                ly,
                3.2 * sy,
                col,
                alpha=0.55 * opacity,
                glow_px=7.0 * sx,
            )

            # Evaluate 7-subframe shutter positions across [age - 0.60*dt, age] for TRUE 180-deg shutter blur!
            sub_xs: list[float] = []
            for s_idx in range(7):
                sub_age = max(0.0, local_age_s - (s_idx / 6.0) * 0.62 * dt_frame)
                su, sdir = shuttle_param(sub_age)
                sp = fn(su)
                sub_xs.append((x_left + span * sp) if sdir == 1 else (x_right - span * sp))

            inst_speed_px = abs(sub_xs[0] - sub_xs[-1]) / 0.62
            stretch_x = min(1.48, 1.0 + inst_speed_px * 0.0042)
            squash_y = 1.0 / math.sqrt(stretch_x)
            puck_r = 21.0 * sy

            # Integrate shutter sub-samples from oldest -> newest with energy conservation
            weights = [0.09, 0.11, 0.13, 0.15, 0.16, 0.17, 0.19]
            for sx_pos, w_sub in zip(reversed(sub_xs), weights):
                draw_sdf_circle(
                    canvas,
                    sx_pos,
                    ly,
                    puck_r,
                    col,
                    alpha=w_sub * opacity,
                    squash_x=stretch_x,
                    squash_y=squash_y,
                    glow_px=10.0 * sx if w_sub > 0.18 else 0.0,
                )
            # Crisp white center core dot
            draw_sdf_circle(
                canvas,
                curr_x,
                ly,
                6.0 * sy,
                (255.0, 255.0, 255.0),
                alpha=0.95 * opacity,
                squash_x=stretch_x,
            )

            # Arrival shockwave ring right as puck hits target gate
            if 0.86 <= u_raw <= 1.0:
                ring_u = (u_raw - 0.86) / 0.14
                gate_x = x_right if direction == 1 else x_left
                draw_sdf_circle(
                    canvas,
                    gate_x,
                    ly,
                    puck_r * (1.0 + 1.55 * ease_expo_snap(ring_u)),
                    col,
                    alpha=(1.0 - ring_u) * 0.65 * opacity,
                    stroke_w=2.4 * sx,
                )

            # Crisp left & right telemetry labels
            cr, cg, cb = int(col[0]), int(col[1]), int(col[2])
            ldraw.text(
                (int(110 * sx), int(ly - 21 * sy)),
                title,
                font=f_title,
                fill=(cr, cg, cb, int(245 * opacity)),
            )
            ldraw.text(
                (int(110 * sx), int(ly + 5 * sy)),
                formula,
                font=f_sub,
                fill=(148, 163, 184, int(195 * opacity)),
            )
            vel_txt = f"{inst_speed_px:5.1f} px/f"
            ldraw.text(
                (int(1575 * sx), int(ly - 11 * sy)),
                vel_txt,
                font=f_title,
                fill=(226, 232, 240, int(230 * opacity)),
            )

        la = np.asarray(label_layer, dtype=np.float32)
        a_ch = la[..., 3:4] / 255.0
        canvas[:] = canvas * (1.0 - a_ch) + la[..., :3] * a_ch

    # ── Set-Piece 2: Concentric Trim-Path Rings & Harmonic Satellites ────────

    def render_orbital_rig(
        self,
        canvas: np.ndarray,
        cx: float,
        cy: float,
        scale: float,
        rotation_deg: float,
        local_age_s: float,
        global_time_s: float,
        opacity: float,
    ) -> None:
        sx = (self.width / 1920.0) * scale
        kick, snare, arp = mograph_audio_envelopes(global_time_s)
        reveal = ease_expo_snap(clamp01(local_age_s / 0.95))

        cy = cy - 42.0 * (self.height / 1080.0)
        # 1. Precision 60-tick rotating vernier degree dial
        dial_r = 305.0 * sx * (0.85 + 0.15 * reveal)
        dial_rot = local_age_s * -16.0
        for k in range(60):
            ang = math.radians(k * 6.0 + dial_rot)
            len_t = (15.0 if k % 5 == 0 else 7.0) * sx
            r0 = dial_r - len_t * 0.5
            r1 = dial_r + len_t * 0.5
            col = (56.0, 189.0, 248.0) if k % 5 == 0 else (71.0, 98.0, 145.0)
            draw_sdf_capsule_segment(
                canvas,
                cx + math.cos(ang) * r0,
                cy + math.sin(ang) * r0,
                cx + math.cos(ang) * r1,
                cy + math.sin(ang) * r1,
                (1.6 if k % 5 == 0 else 1.0) * sx,
                col,
                alpha=0.72 * reveal * opacity,
            )

        # 2. Five Concentric Trim-Path Arc Rings + Comet-Trail Satellites
        rings = [
            (262.0, 6.5, 48.0, 235.0, (56.0, 189.0, 248.0), 1.0),
            (216.0, 8.5, -66.0, 195.0, (168.0, 85.0, 247.0), -1.35),
            (170.0, 7.0, 84.0, 265.0, (251.0, 113.0, 133.0), 1.75),
            (126.0, 9.0, -108.0, 165.0, (251.0, 191.0, 36.0), -2.2),
            (84.0, 5.5, 132.0, 290.0, (52.0, 211.0, 153.0), 2.8),
        ]
        for r_base, thick, spd, max_sweep, col, sat_mul in rings:
            rr = r_base * sx * (0.75 + 0.25 * reveal)
            start_a = local_age_s * spd + rotation_deg * 0.4
            sweep_a = max_sweep * ease_expo_snap(clamp01(local_age_s / 0.85)) * (
                0.88 + 0.12 * math.sin(local_age_s * 2.4 + r_base * 0.02)
            )
            draw_sdf_ring_arc(
                canvas,
                cx,
                cy,
                rr,
                thick * sx,
                start_a,
                sweep_a,
                col,
                alpha=0.92 * opacity,
                glow_px=9.0 * sx,
            )

            # Orbiting harmonic satellite + 10-step glowing comet trail
            head_deg = local_age_s * 68.0 * sat_mul + r_base
            for t_i in range(9, -1, -1):
                lag_deg = head_deg - t_i * 4.2 * sat_mul
                rad = math.radians(lag_deg)
                px = cx + math.cos(rad) * rr
                py = cy + math.sin(rad) * rr
                frac = 1.0 - t_i / 10.0
                draw_sdf_circle(
                    canvas,
                    px,
                    py,
                    (2.5 + 6.5 * frac) * sx,
                    (255.0, 255.0, 255.0) if t_i == 0 else col,
                    alpha=(0.88 if t_i == 0 else 0.34 * frac) * reveal * opacity,
                    glow_px=(10.0 * sx if t_i == 0 else 0.0),
                    glow_color=col,
                )

        # 3. Beat-Reactive Shockwave Ring + Central Rotating Neon Prism Core
        shock_u = (global_time_s * 2.0) % 1.0
        draw_sdf_circle(
            canvas,
            cx,
            cy,
            (54.0 + 165.0 * ease_expo_snap(shock_u)) * sx,
            (56.0, 189.0, 248.0),
            alpha=(1.0 - shock_u) * 0.42 * opacity,
            stroke_w=2.2 * sx,
        )

        core_scale = 1.0 + 0.14 * kick + 0.05 * arp
        draw_sdf_regular_polygon(
            canvas,
            cx,
            cy,
            58.0 * sx * core_scale,
            6,
            (56.0, 189.0, 248.0),
            angle_deg=local_age_s * 52.0 + rotation_deg,
            alpha=0.95 * opacity,
            stroke_w=5.5 * sx,
            glow_px=14.0 * sx,
        )
        draw_sdf_regular_polygon(
            canvas,
            cx,
            cy,
            34.0 * sx * (1.0 + 0.18 * snare),
            4,
            (251.0, 191.0, 36.0),
            angle_deg=-local_age_s * 78.0,
            alpha=0.96 * opacity,
            glow_px=10.0 * sx,
        )

    # ── Set-Piece 3: Swiss / Bauhaus Staggered Geometric Cascade ─────────────

    def render_kinetic_cascade(
        self,
        canvas: np.ndarray,
        cx: float,
        cy: float,
        scale: float,
        local_age_s: float,
        global_time_s: float,
        opacity: float,
    ) -> None:
        sx = (self.width / 1920.0) * scale
        sy = (self.height / 1080.0) * scale
        kick, snare, _ = mograph_audio_envelopes(global_time_s)

        # 8 Bauhaus cards/shapes cascading in with crisp 0.09s (2.3-frame) stagger!
        # Palette: Deep Ink (#0F172A), Cobalt (#2563EB), Coral (#F43F5E), Amber (#F59E0B), Emerald (#10B981), Violet (#7C3AED)
        ink = (15.0, 23.0, 42.0)
        cobalt = (37.0, 99.0, 235.0)
        coral = (244.0, 63.0, 94.0)
        amber = (245.0, 158.0, 11.0)
        emerald = (16.0, 185.0, 129.0)
        violet = (124.0, 58.0, 237.0)

        cy = cy - 34.0 * sy
        items = [
            (-520.0, -155.0, "box", cobalt, 70.0, 45.0),
            (-265.0, -155.0, "arc", coral, 74.0, -90.0),
            (265.0, -155.0, "hex", emerald, 72.0, 30.0),
            (520.0, -155.0, "ring", ink, 68.0, 0.0),
            (-520.0, 95.0, "pill", coral, 86.0, 0.0),
            (-265.0, 95.0, "tri", amber, 74.0, 60.0),
            (265.0, 95.0, "diamond", violet, 72.0, 45.0),
            (520.0, 95.0, "arc2", cobalt, 74.0, 180.0),
        ]

        for idx, (ox, oy, kind, col, sz, base_ang) in enumerate(items):
            t_in = clamp01((local_age_s - idx * 0.085) / 0.62)
            if t_in <= 0.005:
                continue
            sp = ease_spring(t_in, omega=16.5, zeta=0.54)
            px = cx + ox * sx
            py = cy + (oy - 42.0 * (1.0 - sp)) * sy
            s_cur = max(0.05, sp) * sz * sx * (1.0 + 0.05 * kick)
            spin = base_ang * sp + local_age_s * (26.0 if idx % 2 == 0 else -26.0)

            # Soft offset architectural drop shadow on warm ivory paper
            sh_col = (203.0, 198.0, 188.0)
            if kind == "box":
                draw_sdf_round_rect(canvas, px + 8 * sx, py + 10 * sy, s_cur, s_cur, 18 * sx, sh_col, angle_deg=spin, alpha=0.65 * opacity)
                draw_sdf_round_rect(canvas, px, py, s_cur, s_cur, 18 * sx, col, angle_deg=spin, alpha=opacity)
                draw_sdf_circle(canvas, px, py, s_cur * 0.34, (245.0, 242.0, 235.0), alpha=opacity)
            elif kind in ("arc", "arc2"):
                draw_sdf_ring_arc(canvas, px + 7 * sx, py + 9 * sy, s_cur, 24 * sx, spin, 250.0 * ease_expo_snap(t_in), sh_col, alpha=0.65 * opacity)
                draw_sdf_ring_arc(canvas, px, py, s_cur, 24 * sx, spin, 250.0 * ease_expo_snap(t_in), col, alpha=opacity)
            elif kind == "hex":
                draw_sdf_regular_polygon(canvas, px + 7 * sx, py + 9 * sy, s_cur, 6, sh_col, angle_deg=spin, alpha=0.65 * opacity)
                draw_sdf_regular_polygon(canvas, px, py, s_cur, 6, col, angle_deg=spin, alpha=opacity)
            elif kind == "ring":
                draw_sdf_circle(canvas, px + 7 * sx, py + 9 * sy, s_cur, sh_col, stroke_w=20 * sx, alpha=0.65 * opacity)
                draw_sdf_circle(canvas, px, py, s_cur, col, stroke_w=20 * sx, alpha=opacity)
                draw_sdf_circle(canvas, px, py, s_cur * 0.36 * (1.0 + 0.25 * snare), amber, alpha=opacity)
            elif kind == "pill":
                slide = math.sin(local_age_s * 3.1) * 24.0 * sx
                draw_sdf_round_rect(canvas, px, py, s_cur * 1.15, s_cur * 0.52, s_cur * 0.50, ink, alpha=opacity)
                draw_sdf_circle(canvas, px + slide, py, s_cur * 0.36, coral, alpha=opacity)
            elif kind == "tri":
                draw_sdf_regular_polygon(canvas, px + 7 * sx, py + 9 * sy, s_cur, 3, sh_col, angle_deg=spin, alpha=0.65 * opacity)
                draw_sdf_regular_polygon(canvas, px, py, s_cur, 3, col, angle_deg=spin, alpha=opacity)
            elif kind == "diamond":
                draw_sdf_regular_polygon(canvas, px + 7 * sx, py + 9 * sy, s_cur, 4, sh_col, angle_deg=spin, alpha=0.65 * opacity)
                draw_sdf_regular_polygon(canvas, px, py, s_cur, 4, col, angle_deg=spin, alpha=opacity, stroke_w=16 * sx)

        # Central Portal Medallion right at screen center `(w*0.5, h*0.5)` so `zoom in` iris portal matches it seamlessly!
        t_portal = clamp01(local_age_s / 0.55)
        p_scale = ease_spring(t_portal, omega=15.0, zeta=0.60) * (1.0 + 0.08 * kick)
        pcx, pcy = self.width * 0.5, self.height * 0.5 - 38.0 * sy
        draw_sdf_circle(canvas, pcx, pcy, 108.0 * sx * p_scale, ink, alpha=opacity)
        draw_sdf_ring_arc(
            canvas,
            pcx,
            pcy,
            132.0 * sx * p_scale,
            8.0 * sx,
            local_age_s * 72.0,
            285.0,
            cobalt,
            alpha=opacity,
        )
        draw_sdf_regular_polygon(
            canvas,
            pcx,
            pcy,
            52.0 * sx * p_scale,
            6,
            (56.0, 189.0, 248.0),
            angle_deg=-local_age_s * 54.0,
            stroke_w=6.0 * sx,
            alpha=opacity,
        )
        draw_sdf_circle(canvas, pcx, pcy, 16.0 * sx * (1.0 + 0.30 * snare), coral, alpha=opacity)

    # ── Set-Piece 4: 36-Band Audio-Reactive Spectrum & Waveform Finale ───────

    def render_audio_equalizer(
        self,
        canvas: np.ndarray,
        cx: float,
        cy: float,
        scale: float,
        local_age_s: float,
        global_time_s: float,
        opacity: float,
    ) -> None:
        sx = (self.width / 1920.0) * scale
        sy = (self.height / 1080.0) * scale
        kick, snare, arp = mograph_audio_envelopes(global_time_s)
        reveal = ease_expo_snap(clamp01(local_age_s / 0.70))

        # Hero central emblem ring above spectrum
        emblem_y = cy - 155.0 * sy
        emblem_r = (86.0 + 13.0 * kick) * sx * reveal
        draw_sdf_ring_arc(
            canvas,
            cx,
            emblem_y,
            emblem_r * 1.24,
            5.0 * sx,
            global_time_s * 64.0,
            270.0,
            (251.0, 191.0, 36.0),
            alpha=0.92 * opacity,
            glow_px=9.0 * sx,
        )
        draw_sdf_circle(
            canvas,
            cx,
            emblem_y,
            emblem_r,
            (56.0, 189.0, 248.0),
            alpha=0.95 * opacity,
            stroke_w=6.0 * sx,
            glow_px=15.0 * sx,
        )
        # Play icon pointing RIGHT (▶) inside hero emblem ring!
        draw_sdf_regular_polygon(
            canvas,
            cx + emblem_r * 0.06,
            emblem_y,
            emblem_r * 0.52,
            3,
            (255.0, 255.0, 255.0),
            angle_deg=-90.0 + math.sin(global_time_s * 2.0) * 5.0,
            alpha=0.96 * opacity,
            glow_px=8.0 * sx,
        )

        # 36 symmetric rounded capsule spectrum bars locked to kick/snare/arp envelopes!
        n_bars = 36
        spacing = 34.0 * sx
        start_x = cx - (n_bars - 1) * spacing * 0.5
        base_y = cy + 46.0 * sy

        for i in range(n_bars):
            dist_center = abs(i - (n_bars - 1) * 0.5) / ((n_bars - 1) * 0.5)
            low_w = math.exp(-(dist_center ** 2) / 0.11)
            mid_w = math.exp(-((dist_center - 0.46) ** 2) / 0.065)
            high_w = math.exp(-((dist_center - 0.84) ** 2) / 0.06)
            travel_wave = 0.5 + 0.5 * math.sin(global_time_s * 9.5 - dist_center * 8.5)

            energy = (
                0.08
                + 0.92 * low_w * kick
                + 0.72 * mid_w * (0.25 + 0.75 * arp) * travel_wave
                + 0.78 * high_w * (0.18 + 0.82 * snare)
            ) * reveal
            bar_h = max(7.0 * sy, min(148.0 * sy, energy * 145.0 * sy))
            bx = start_x + i * spacing

            # Smooth gradient from Gold -> Cyan -> Violet-Rose across spectrum
            if dist_center < 0.5:
                u_c = dist_center / 0.5
                col = (
                    251.0 * (1.0 - u_c) + 56.0 * u_c,
                    191.0 * (1.0 - u_c) + 189.0 * u_c,
                    36.0 * (1.0 - u_c) + 248.0 * u_c,
                )
            else:
                u_c = (dist_center - 0.5) / 0.5
                col = (
                    56.0 * (1.0 - u_c) + 244.0 * u_c,
                    189.0 * (1.0 - u_c) + 63.0 * u_c,
                    248.0 * (1.0 - u_c) + 165.0 * u_c,
                )

            draw_sdf_capsule_segment(
                canvas,
                bx,
                base_y - bar_h,
                bx,
                base_y + bar_h * 0.42,
                6.2 * sx,
                col,
                alpha=0.94 * opacity,
                glow_px=6.5 * sx,
            )

    # ── Atomic Parametric Vector Shape Renderer (with Shutter Motion Blur!) ──

    def render_atomic_vector_object(
        self,
        canvas: np.ndarray,
        *,
        name: str,
        cx: float,
        cy: float,
        vx_px: float,
        vy_px: float,
        scale: float,
        rotation_deg: float,
        opacity: float,
        color: tuple[int, int, int],
        glow: float,
    ) -> None:
        sx = (self.width / 1920.0) * scale
        low = name.strip().lower()
        col_f = (float(color[0]), float(color[1]), float(color[2]))
        speed = math.hypot(vx_px, vy_px)
        heading = math.degrees(math.atan2(vy_px, vx_px)) if speed > 0.5 else rotation_deg
        stretch = min(1.42, 1.0 + speed * 0.004)
        squash = 1.0 / math.sqrt(stretch)

        steps = 5 if speed > 2.5 else 1
        for k in range(steps - 1, -1, -1):
            frac = k / max(steps - 1, 1) * 0.62 if steps > 1 else 0.0
            px = cx - vx_px * frac
            py = cy - vy_px * frac
            sub_a = (opacity / steps) * (1.25 - 0.5 * (k / max(steps, 1)))
            g_px = (12.0 * glow * sx) if k == 0 else 0.0

            if "ring" in low or "arc" in low:
                draw_sdf_ring_arc(
                    canvas, px, py, 56.0 * sx, 10.0 * sx, rotation_deg, 275.0, col_f, alpha=sub_a, glow_px=g_px
                )
            elif "hex" in low or "prism" in low:
                draw_sdf_regular_polygon(
                    canvas, px, py, 52.0 * sx, 6, col_f, angle_deg=rotation_deg, alpha=sub_a, stroke_w=7.0 * sx, glow_px=g_px
                )
            elif "square" in low or "diamond" in low:
                draw_sdf_round_rect(
                    canvas, px, py, 44.0 * sx, 44.0 * sx, 10.0 * sx, col_f, angle_deg=rotation_deg, alpha=sub_a, glow_px=g_px
                )
            elif "capsule" in low or "pill" in low:
                draw_sdf_round_rect(
                    canvas, px, py, 76.0 * sx, 28.0 * sx, 26.0 * sx, col_f, angle_deg=rotation_deg, alpha=sub_a, glow_px=g_px
                )
            else:
                draw_sdf_circle(
                    canvas,
                    px,
                    py,
                    30.0 * sx,
                    col_f,
                    alpha=sub_a,
                    squash_x=stretch,
                    squash_y=squash,
                    angle_deg=heading,
                    glow_px=g_px,
                )


# ══════════════════════════════════════════════════════════════════════════════
# 6. BROADCAST MOTION-GRAPHICS TRANSITIONS
# ══════════════════════════════════════════════════════════════════════════════


def blend_mograph_transition(
    prev_rgb: np.ndarray,
    curr_rgb: np.ndarray,
    trans_type: str,
    raw_alpha: float,
) -> np.ndarray:
    """
    Signature broadcast motion-graphics transitions between two live streams:
      - `slide`: Whip-Pan Momentum Push with 9-tap linear-light shutter blur + RGB fringe
      - `wipe`:  16-deg Angled 3-Ribbon Chromatic Stinger Wipe (Cyan / Coral / Gold)
      - `zoom in` / `zoom out`: Sub-pixel SDF Expanding Circular Iris Portal Match-Cut
      - `spin` / `flip`: 3D Perspective Card Flip with cosine specular shading
    """
    h, w, _ = curr_rgb.shape
    t = trans_type.strip().lower()
    u = clamp01(raw_alpha)

    if "slide" in t:
        # Derivative estimates instantaneous whip velocity for real shutter blur!
        eps = 0.025
        a_now = ease_whip_in_out(u)
        a_prev = ease_whip_in_out(clamp01(u - eps))
        vel_norm = abs(a_now - a_prev) / eps  # Peaks smoothly around u = 0.5

        # Composite horizontal push canvas of width `2*w`
        combo = np.concatenate([prev_rgb.astype(np.float32), curr_rgb.astype(np.float32)], axis=1)
        center_shift = a_now * w
        blur_span = min(w * 0.16, vel_norm * w * 0.022)

        # Integrate 9 horizontal linear-light shutter samples across `[shift - blur, shift + blur]`
        acc_lin = np.zeros((h, w, 3), dtype=np.float32)
        taps = np.linspace(-0.5, 0.5, 9, dtype=np.float32)

        for tap in taps:
            M_tap = np.float32([[1.0, 0.0, -float(center_shift + tap * blur_span)], [0.0, 1.0, 0.0]])
            sample = cv2.warpAffine(
                combo,
                M_tap,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            acc_lin += (sample / 255.0) ** 2.2

        acc_lin /= len(taps)
        out = (np.clip(acc_lin, 0.0, 1.0) ** (1.0 / 2.2)) * 255.0

        # Subtle velocity-proportional RGB prism fringe at peak whip speed (0 at rest!)
        fringe = int(round(min(12.0, blur_span * 0.09)))
        if fringe >= 1:
            out[..., 0] = np.roll(out[..., 0], fringe, axis=1)
            out[..., 2] = np.roll(out[..., 2], -fringe, axis=1)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    if "wipe" in t:
        # 16-degree Angled 3-Ribbon Stinger Wipe (sleek graphic ribbons showing BOTH live scenes!)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        diag = (xx / w) + (yy / h - 0.5) * 0.28
        aa = 1.6 / w

        lead = -0.22 + 1.56 * ease_whip_in_out(u)
        e1 = lead
        e2 = lead - 0.068
        e3 = lead - 0.132
        e4 = lead - 0.188

        out = prev_rgb.astype(np.float32).copy()
        for edge, col in (
            (e1, (56.0, 189.0, 248.0)),
            (e2, (244.0, 63.0, 94.0)),
            (e3, (251.0, 191.0, 36.0)),
        ):
            m = np.clip((edge - diag) / aa, 0.0, 1.0)[..., None]
            c_arr = np.array(col, dtype=np.float32)[None, None, :]
            out = out * (1.0 - m) + c_arr * m

        m_in = np.clip((e4 - diag) / aa, 0.0, 1.0)[..., None]
        out = out * (1.0 - m_in) + curr_rgb.astype(np.float32) * m_in
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    if "zoom" in t:
        # Expanding Circular Iris Portal Match-Cut centered on the stage medallion!
        a = ease_whip_in_out(u)
        M_out = cv2.getRotationMatrix2D((w * 0.5, h * 0.465), 0.0, 1.0 + 0.22 * a)
        M_in = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.0, 0.88 + 0.12 * a)
        w_prev = cv2.warpAffine(prev_rgb, M_out, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)
        w_curr = cv2.warpAffine(curr_rgb, M_in, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101).astype(np.float32)

        max_r = math.hypot(w * 0.55, h * 0.60)
        portal_r = a * max_r
        pcx, pcy = w * 0.5, h * (0.465 * (1.0 - a) + 0.50 * a)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.hypot(xx - pcx, yy - pcy)
        sdf = dist - portal_r
        mask = np.clip(0.5 - sdf / 1.6, 0.0, 1.0)[..., None]
        out = w_prev * (1.0 - mask) + w_curr * mask

        # Luminous portal rim ring
        rim = np.exp(-np.abs(sdf) / 7.5)[..., None] * math.sin(u * math.pi)
        out += rim * np.array([96.0, 205.0, 255.0], dtype=np.float32) * 0.85
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    # Fallback: Linear-light optical dissolve
    a = ease_quintic_c2(u)
    p_lin = (prev_rgb.astype(np.float32) / 255.0) ** 2.2
    c_lin = (curr_rgb.astype(np.float32) / 255.0) ** 2.2
    mixed = ((p_lin * (1.0 - a) + c_lin * a) ** (1.0 / 2.2)) * 255.0
    return np.clip(mixed, 0.0, 255.0).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# 7. 120-BPM ELECTRONIC MOTION-GRAPHICS STEREO SYNTH SOUNDTRACK
# ══════════════════════════════════════════════════════════════════════════════


def write_mograph_electronic_score(
    *,
    wav_path: Path,
    duration_sec: float,
    transition_times_sec: list[float],
) -> None:
    """
    Synthesizes a crisp 44.1 kHz stereo 120-BPM motion-graphics electronic score
    (punchy sub-kick on every exact 13-frame beat @ 26 fps, filtered snare claps,
    warm minor-9th synth pad progression, crystalline 16th-note stereo arpeggio,
    and filtered whoosh risers peaking on every scene transition!).
    """
    sr = 44100
    n = max(1, int(round(duration_sec * sr)))
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)

    beat_pos = t * 2.0  # 120.0 BPM -> exact 0.500s per beat = 13.000 frames @ 26 fps!
    frac_beat = np.mod(beat_pos, 1.0)
    beat_idx = np.floor(beat_pos).astype(np.int32)

    # 1. Punchy 120-BPM sub-bass kick drum with pitch sweep (155 Hz -> 46 Hz)
    kick_env = np.exp(-frac_beat * 7.2)
    kick_phase = 2.0 * math.pi * (46.0 * (frac_beat * 0.5) + 12.5 * (1.0 - np.exp(-frac_beat * 14.0)))
    kick = 0.34 * np.sin(kick_phase) * kick_env

    # 2. Crisp backbeat snare/hat texture on odd beats
    rng = np.random.default_rng(707)
    white = rng.normal(0.0, 1.0, n).astype(np.float32)
    bright_noise = white - np.roll(white, 1)  # High-pass crisp air
    snare_mask = (beat_idx % 2 == 1).astype(np.float32)
    snare = 0.09 * bright_noise * np.exp(-frac_beat * 12.0) * snare_mask

    # Soft 8th-note hi-hat tick
    eighth_frac = np.mod(beat_pos * 2.0, 1.0)
    hat = 0.035 * bright_noise * np.exp(-eighth_frac * 22.0)

    # 3. 16th-note stereo crystalline arpeggiator (C minor 9th / Fsus2 progression across 4 acts)
    notes_hz = np.array([261.63, 311.13, 392.00, 466.16, 587.33, 466.16, 392.00, 311.13], dtype=np.float32)
    sixteenth_idx = np.floor(beat_pos * 4.0).astype(np.int32)
    sixteenth_frac = np.mod(beat_pos * 4.0, 1.0)
    arp_freqs = notes_hz[sixteenth_idx % len(notes_hz)]
    arp_env = np.exp(-sixteenth_frac * 5.2)
    arp_sig = 0.12 * (
        np.sin(2.0 * math.pi * arp_freqs * (sixteenth_frac * 0.125))
        + 0.4 * np.sin(4.0 * math.pi * arp_freqs * (sixteenth_frac * 0.125))
    ) * arp_env
    pan_lfo = 0.5 + 0.36 * np.sin(2.0 * math.pi * 0.5 * t)

    # 4. Warm deep stereo synth pad foundation
    pad_l = 0.11 * np.sin(2.0 * math.pi * 130.81 * t) + 0.08 * np.sin(2.0 * math.pi * 196.00 * t + 0.2)
    pad_r = 0.11 * np.sin(2.0 * math.pi * 130.81 * t) + 0.08 * np.sin(2.0 * math.pi * 233.08 * t - 0.2)

    # 5. Filtered stereo whoosh impacts locked right onto every scene transition!
    whoosh_l = np.zeros(n, dtype=np.float32)
    whoosh_r = np.zeros(n, dtype=np.float32)
    smooth_noise = np.convolve(white, np.ones(18, dtype=np.float32) / 18.0, mode="same")
    for tc in transition_times_sec:
        env_w = np.exp(-((t - tc) ** 2) / 0.032)
        sweep_f = 180.0 + 420.0 * np.exp(-((t - tc) ** 2) / 0.05)
        whoosh_tonal = 0.14 * np.sin(2.0 * math.pi * sweep_f * (t - tc)) * env_w
        pan_sweep = np.clip((t - tc) * 2.5 + 0.5, 0.0, 1.0)
        whoosh_l += (smooth_noise * 0.16 * env_w + whoosh_tonal) * (1.0 - 0.45 * pan_sweep)
        whoosh_r += (smooth_noise * 0.16 * env_w + whoosh_tonal) * (0.55 + 0.45 * pan_sweep)

    left = kick + snare + hat + pad_l + arp_sig * pan_lfo + whoosh_l
    right = kick + snare + hat + pad_r + arp_sig * (1.0 - pan_lfo) + whoosh_r

    # Clean master envelope
    in_len = min(int(0.25 * sr), n // 6)
    out_len = min(int(0.55 * sr), n // 5)
    master = np.ones(n, dtype=np.float32)
    if in_len > 1:
        master[:in_len] *= np.linspace(0.0, 1.0, in_len, dtype=np.float32)
    if out_len > 1:
        master[-out_len:] *= np.linspace(1.0, 0.0, out_len, dtype=np.float32)

    stereo = np.stack([np.tanh(left * master * 1.15), np.tanh(right * master * 1.15)], axis=-1)
    pcm16 = np.clip(stereo * 0.78 * 32767.0, -32767.0, 32767.0).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
