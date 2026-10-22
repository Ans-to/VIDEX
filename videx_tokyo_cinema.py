"""
VIDEX Hyperreal Anamorphic Cinema Engine — "Tokyo Seed Courier"
===============================================================

Renders `24 fps` (`360` frames = `15.00s`) 4K UHD (`3840x2160`) 2.39:1
Panavision-style anamorphic cinema with:
  - Shot 1 [0:00–0:03]: 35mm low-angle rain-slicked Tokyo night alley slow dolly-in,
    flickering cyan/magenta neon & puddle reflections, rising grate steam, darting
    stray cat left-to-right with puddle reflection, courier entrance & ballistic
    chrome-case water shake spray.
  - Shot 2 [0:03–0:07]: 50mm handheld micro-shake MCU, true optical 2x anamorphic
    oval rack focus from foreground lens raindrops to courier's eyes, rolling coat
    raindrops, smartwatch tap -> blooming emerald 3D hologram map + single glitch,
    visible cold breath vapor + whispered "Wrong block."
  - Shot 3 [0:07–0:11]: 24mm lateral tracking rooftop chase with barrel distortion,
    whipping laundry lines, 180-deg shutter-blurred rooftop gap leap, snapped-cable
    electric sparks, sweeping red drone searchlight + horizontal anamorphic flare,
    slide behind AC unit & chrome case pop revealing heartbeat-pulsing blue seed.
  - Shot 4 [0:11–0:15]: 100mm macro 120fps->24fps slow-motion push-in, intensifying
    seed heartbeat, growing bioluminescent light roots crawling across wet glove,
    animated blue caustics on courier's smiling face, teal/amber cinema grade,
    subtle chromatic aberration & 4K 35mm film grain, hard cut to pure black at
    14.42s synchronized to a single resonant crystal chime.
  - Strictly ZERO text overlays, ZERO watermarks, ZERO jump cuts.
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import tempfile
import wave

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


def _clamp01(v: float) -> float:
    return 0.0 if v <= 0.0 else (1.0 if v >= 1.0 else v)


def _smoothstep(t: float) -> float:
    u = _clamp01(t)
    return u * u * (3.0 - 2.0 * u)


def _smootherstep(t: float) -> float:
    u = _clamp01(t)
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def _heartbeat(time_s: float, period: float = 0.86, phase: float = 0.0) -> float:
    """Physiological double-peak ('lub-dub') cardiac pulse envelope in [0, 1]."""
    u = ((time_s - phase) % period) / period
    lub = math.exp(-((u - 0.10) ** 2) / 0.0022)
    dub = 0.72 * math.exp(-((u - 0.27) ** 2) / 0.0032)
    return min(1.0, lub + dub)


def is_tokyo_courier_program(program_name: str, scenes: list[str]) -> bool:
    combined = (program_name + " " + " ".join(scenes)).lower()
    return (
        "tokyo" in combined
        or "courier" in combined
        or ("rooftop" in combined and "seed" in combined)
    )


def _periodic_steam_field(h: int, w: int, seed: int = 313) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    amp = 0.5
    tot = 0.0
    for o, gy in enumerate((5, 10, 20, 40)):
        gx = gy * 2
        rng = np.random.default_rng(seed + o * 37)
        g = rng.random((gy, gx), dtype=np.float32)
        tiled = np.concatenate([g[-2:, :], g, g[:2, :]], axis=0)
        tiled = np.concatenate([tiled[:, -2:], tiled, tiled[:, :2]], axis=1)
        up = cv2.resize(
            tiled,
            (int(w * (gx + 4) / gx), int(h * (gy + 4) / gy)),
            interpolation=cv2.INTER_CUBIC,
        )
        mx = int(round(w * 2 / gx))
        my = int(round(h * 2 / gy))
        acc += amp * up[my : my + h, mx : mx + w]
        tot += amp
        amp *= 0.54
    return np.clip(acc / max(tot, 1e-6), 0.0, 1.0)


class TokyoCinemaRenderer:
    """Multi-layer 2.5D optical & physical renderer for `Tokyo Seed Courier`."""

    # Internal working active 2.39:1 anamorphic frame inside 1920x1080 -> Lanczos 4K 3840x2160
    WORK_W = 1920
    WORK_H = 1080
    BAR_PX = 138  # 1080 - 2*138 = 804 -> 1920 / 804 = 2.388:1 Panavision Anamorphic!
    ACT_H = WORK_H - 2 * BAR_PX  # 804 px

    def __init__(self, assets_dir: Path) -> None:
        self.assets_dir = assets_dir
        self.pad_w = int(round(self.WORK_W * 1.20))  # 2304
        self.pad_h = int(round(self.ACT_H * 1.28))   # 1029

        yy, xx = np.mgrid[0 : self.ACT_H, 0 : self.WORK_W].astype(np.float32)
        self.xx = xx
        self.yy = yy
        self.u = xx / (self.WORK_W - 1.0)
        self.v = yy / (self.ACT_H - 1.0)

        # Load all 7 studio hyperreal plates with aspect-preserving cover scaling
        self.alley_bg = self._load_cover("tokyo_shot1_alley_bg.jpg", focus_y=0.58)
        self.alley_courier = self._load_cover("tokyo_shot1_courier_alley.png", focus_y=0.50)
        self.mcu_courier = self._load_cover("tokyo_courier_ref.png", focus_y=0.52)
        self.rooftop_bg = self._load_cover("tokyo_shot3_rooftop_bg.jpg", focus_y=0.50)
        self.rooftop_leap = self._load_cover("tokyo_shot3_leap.png", focus_y=0.50)
        self.rooftop_ac = self._load_cover("tokyo_shot3_ac_box.png", focus_y=0.47)
        self.macro_seed = self._load_cover("tokyo_shot4_macro_seed.png", focus_y=0.50)

        # Extract soft-feathered foreground alpha mattes where background matches
        self.leaper_rgba = self._extract_leaper_sprite()

        # Precompute Macro Seed root-less base state so bioluminescent roots crawl live!
        self.macro_clean_glove, self.macro_roots_only = self._decompose_macro_seed_roots()

        # Depth fields for smooth 2.5D camera parallax
        self.depth_alley = self._compute_alley_depth(self.alley_bg)
        self.depth_rooftop = self._compute_rooftop_depth(self.rooftop_bg)

        # Procedural volumetric steam & caustics textures
        self.steam_tex = _periodic_steam_field(self.ACT_H, self.WORK_W, seed=515)

        # Deterministic particle banks (rain streaks, water spray, cable sparks, lens drops, coat beads, spores)
        rng = np.random.default_rng(2049)
        self.rain_streaks = rng.uniform(0.0, 1.0, (220, 4)).astype(np.float32)
        self.spray_drops = np.column_stack(
            [
                rng.uniform(-210.0, 260.0, 46),   # vx px/s
                rng.uniform(-320.0, -70.0, 46),   # vy px/s upward burst
                rng.uniform(1.6, 4.2, 46),        # radius px
                rng.uniform(0.0, 0.24, 46),       # stagger delay s
            ]
        ).astype(np.float32)
        self.sparks = np.column_stack(
            [
                rng.uniform(-340.0, 290.0, 64),   # vx
                rng.uniform(-260.0, 60.0, 64),    # vy
                rng.uniform(1.2, 3.4, 64),        # radius
                rng.uniform(0.0, 0.45, 64),       # delay s
            ]
        ).astype(np.float32)
        self.lens_drops = np.column_stack(
            [
                rng.uniform(0.06, 0.94, 34),
                rng.uniform(0.08, 0.92, 34),
                rng.uniform(5.0, 19.0, 34),
                rng.uniform(0.0, 1.0, 34),
            ]
        ).astype(np.float32)
        self.coat_beads = np.column_stack(
            [
                rng.uniform(0.24, 0.77, 28),      # u on raincoat shoulders/chest
                rng.uniform(0.52, 0.90, 28),      # starting v
                rng.uniform(0.020, 0.058, 28),    # downward roll speed v/s
                rng.uniform(1.5, 3.2, 28),        # droplet radius px
                rng.uniform(0.0, 6.28, 28),       # slip-stick phase
            ]
        ).astype(np.float32)
        self.spores = np.column_stack(
            [
                rng.uniform(0.08, 0.92, 55),
                rng.uniform(0.08, 0.92, 55),
                rng.uniform(0.008, 0.024, 55),
                rng.uniform(1.2, 3.2, 55),
                rng.uniform(0.0, 6.28, 55),
            ]
        ).astype(np.float32)

    # ── Aspect-Ratio-Preserving Cover Loader & Matte Decomposition ───────────

    def _load_cover(self, filename: str, focus_y: float = 0.5) -> np.ndarray:
        path = self.assets_dir / filename
        img = Image.open(path).convert("RGB")
        iw, ih = img.size
        scale = max(self.pad_w / iw, self.pad_h / ih)
        nw = max(self.pad_w, int(round(iw * scale)))
        nh = max(self.pad_h, int(round(ih * scale)))
        resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
        left = int(round((nw - self.pad_w) * 0.5))
        top = int(round((nh - self.pad_h) * _clamp01(focus_y)))
        cropped = resized.crop((left, top, left + self.pad_w, top + self.pad_h))
        return np.asarray(cropped, dtype=np.float32)

    def _extract_leaper_sprite(self) -> np.ndarray:
        """Isolates the leaping courier from `rooftop_leap` inside the gap region."""
        h, w, _ = self.rooftop_leap.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        un, vn = xx / w, yy / h
        box_mask = (
            np.clip((un - 0.39) * 35.0, 0.0, 1.0)
            * np.clip((0.61 - un) * 35.0, 0.0, 1.0)
            * np.clip((vn - 0.24) * 35.0, 0.0, 1.0)
            * np.clip((0.68 - vn) * 35.0, 0.0, 1.0)
        )
        diff = np.linalg.norm(self.rooftop_leap - self.rooftop_bg, axis=-1)
        alpha = np.clip((diff - 26.0) / 38.0, 0.0, 1.0) * box_mask
        # Clean stray red searchlight mist outside tight courier body region
        body_ellipse = np.exp(-(((un - 0.50) / 0.085) ** 2 + ((vn - 0.455) / 0.19) ** 2) ** 2)
        alpha = cv2.GaussianBlur(alpha * np.clip(body_ellipse * 1.5, 0.0, 1.0), (0, 0), sigmaX=1.4)
        return np.dstack([self.rooftop_leap, alpha * 255.0]).astype(np.float32)

    def _decompose_macro_seed_roots(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Separates the glowing cyan light-root filaments on the black leather glove
        from the dark glove texture using OpenCV morphological top-hat / inpainting
        so the roots physically GROW outward from 0% -> 100% across Shot 4!
        """
        base = self.macro_seed.copy()
        h, w, _ = base.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        un, vn = xx / w, yy / h

        # Seed core sits at `(u ≈ 0.428, v ≈ 0.345)`; glove spans `u < 0.52`
        dist_seed = np.hypot((un - 0.428) * 1.45, vn - 0.345)
        glove_zone = (un < 0.515) & (dist_seed > 0.095)

        cyan_excess = base[..., 2] * 0.65 + base[..., 1] * 0.45 - base[..., 0] * 0.55
        filament_mask = ((cyan_excess > 42.0) & glove_zone).astype(np.uint8) * 255
        filament_mask = cv2.dilate(filament_mask, np.ones((5, 5), np.uint8), iterations=1)

        inpainted_u8 = cv2.inpaint(
            np.clip(base, 0.0, 255.0).astype(np.uint8),
            filament_mask,
            inpaintRadius=6,
            flags=cv2.INPAINT_TELEA,
        )
        clean_glove = inpainted_u8.astype(np.float32)
        roots_delta = np.maximum(base - clean_glove, 0.0)
        return clean_glove, roots_delta

    @staticmethod
    def _compute_alley_depth(rgb: np.ndarray) -> np.ndarray:
        h, w, _ = rgb.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        un = np.abs(xx / w - 0.5) * 2.0  # Walls on sides are close (high depth), center alley is far!
        vn = yy / h                      # Ground floor is very close!
        depth = np.clip(0.25 + 0.55 * un + 0.65 * np.clip((vn - 0.55) * 2.2, 0.0, 1.0), 0.18, 1.0)
        return cv2.GaussianBlur(depth, (0, 0), sigmaX=18.0)

    @staticmethod
    def _compute_rooftop_depth(rgb: np.ndarray) -> np.ndarray:
        h, w, _ = rgb.shape
        yy, _ = np.mgrid[0:h, 0:w].astype(np.float32)
        vn = yy / h
        depth = np.clip(0.16 + 0.88 * (vn ** 1.35), 0.15, 1.0)
        return cv2.GaussianBlur(depth, (0, 0), sigmaX=16.0)

    def _sample_parallax(
        self,
        plate: np.ndarray,
        depth: np.ndarray | None,
        pan_x: float,
        tilt_y: float,
        zoom: float,
        shake_px: tuple[float, float] = (0.0, 0.0),
        barrel_k: float = 0.0,
    ) -> np.ndarray:
        ph, pw, _ = plate.shape
        mx = (pw - self.WORK_W) * 0.5
        my = (ph - self.ACT_H) * 0.5
        inv_z = 1.0 / max(zoom, 0.5)

        nx = (self.xx - self.WORK_W * 0.5) / (self.WORK_W * 0.5)
        ny = (self.yy - self.ACT_H * 0.5) / (self.WORK_W * 0.5)
        if abs(barrel_k) > 1e-5:
            r2 = nx * nx + ny * ny
            radial = 1.0 + barrel_k * r2
            sx_off = nx * radial * (self.WORK_W * 0.5)
            sy_off = ny * radial * (self.WORK_W * 0.5)
        else:
            sx_off = self.xx - self.WORK_W * 0.5
            sy_off = self.yy - self.ACT_H * 0.5

        bx = pw * 0.5 + sx_off * inv_z + shake_px[0]
        by = ph * 0.5 + sy_off * inv_z + shake_px[1]

        if depth is not None:
            d_loc = cv2.remap(depth, bx, by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            fac = 0.42 + 0.78 * d_loc
        else:
            fac = 1.0

        map_x = (bx + pan_x * mx * fac).astype(np.float32)
        map_y = (by + tilt_y * my * fac).astype(np.float32)
        return cv2.remap(
            plate, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )

    # ── Reusable Cinema Visual FX Layers ─────────────────────────────────────

    def _add_rain_streaks(
        self,
        frame: np.ndarray,
        time_s: float,
        *,
        intensity: float = 1.0,
        wind_x: float = -12.0,
        speed: float = 1.0,
    ) -> None:
        if intensity <= 0.01:
            return
        overlay = np.zeros_like(frame)
        for u0, v0, depth_r, ph in self.rain_streaks:
            fall_rate = (1.35 + 1.85 * depth_r) * speed
            cur_v = (v0 + time_s * fall_rate) % 1.0
            cur_u = (u0 + time_s * (wind_x / self.WORK_W) * fall_rate + 0.01 * math.sin(ph)) % 1.0
            x0 = cur_u * self.WORK_W
            y0 = cur_v * self.ACT_H
            streak_len = (16.0 + 34.0 * depth_r) * speed
            dx = wind_x * (0.4 + 0.8 * depth_r)
            alpha = float((0.10 + 0.24 * float(depth_r)) * intensity)
            cv2.line(
                overlay,
                (int(round(float(x0))), int(round(float(y0)))),
                (int(round(float(x0 + dx))), int(round(float(y0 + streak_len)))),
                (float(185.0 * alpha), float(215.0 * alpha), float(238.0 * alpha)),
                1 if float(depth_r) < 0.72 else 2,
                lineType=cv2.LINE_AA,
            )
        frame += overlay

    @staticmethod
    def _draw_running_cat(
        frame: np.ndarray, progress: float, time_s: float
    ) -> None:
        """
        Draws a sleek sprinting stray cat silhouette + wet asphalt mirror
        reflection + 5-subframe shutter blur darting left -> right across Shot 1!
        """
        h, w, _ = frame.shape
        overlay = np.zeros((h, w, 4), dtype=np.float32)

        for tap in range(5):
            u_sub = progress - tap * 0.011
            if not (-0.08 <= u_sub <= 1.06):
                continue
            cx = u_sub * w
            stride = time_s * 17.5 - tap * 0.35
            cy = h * 0.855 - abs(math.sin(stride)) * (h * 0.016)
            sc = h * 0.085
            wt = 0.28 - tap * 0.04

            pil = Image.new("RGBA", (int(sc * 3.4), int(sc * 2.2)), (0, 0, 0, 0))
            d = ImageDraw.Draw(pil)
            ox, oy = sc * 1.7, sc * 0.95
            leg_swing = math.sin(stride) * sc * 0.42

            # Sleek feline body + arched tail + ears + galloping fore/hind legs
            col_cat = (12, 16, 24, int(235 * wt * 3.0))
            d.ellipse((ox - sc * 0.62, oy - sc * 0.22, ox + sc * 0.55, oy + sc * 0.20), fill=col_cat)
            # Head + pointed ears
            d.ellipse((ox + sc * 0.42, oy - sc * 0.36, ox + sc * 0.82, oy - sc * 0.02), fill=col_cat)
            d.polygon([(ox + sc * 0.48, oy - sc * 0.32), (ox + sc * 0.55, oy - sc * 0.54), (ox + sc * 0.63, oy - sc * 0.32)], fill=col_cat)
            d.polygon([(ox + sc * 0.64, oy - sc * 0.32), (ox + sc * 0.72, oy - sc * 0.53), (ox + sc * 0.77, oy - sc * 0.30)], fill=col_cat)
            # Whipping tail
            d.line(
                [
                    (ox - sc * 0.58, oy - sc * 0.06),
                    (ox - sc * 0.98, oy - sc * 0.24 + math.cos(stride) * sc * 0.12),
                    (ox - sc * 1.32, oy - sc * 0.42),
                ],
                fill=col_cat,
                width=max(3, int(sc * 0.09)),
            )
            # Galloping forelegs & hindlegs
            d.line([(ox + sc * 0.38, oy + sc * 0.10), (ox + sc * 0.52 + leg_swing, oy + sc * 0.54)], fill=col_cat, width=max(3, int(sc * 0.09)))
            d.line([(ox + sc * 0.26, oy + sc * 0.10), (ox + sc * 0.22 - leg_swing * 0.6, oy + sc * 0.52)], fill=col_cat, width=max(3, int(sc * 0.09)))
            d.line([(ox - sc * 0.42, oy + sc * 0.08), (ox - sc * 0.46 - leg_swing, oy + sc * 0.54)], fill=col_cat, width=max(3, int(sc * 0.10)))
            d.line([(ox - sc * 0.52, oy + sc * 0.08), (ox - sc * 0.78 + leg_swing * 0.5, oy + sc * 0.50)], fill=col_cat, width=max(3, int(sc * 0.10)))

            sp = np.asarray(pil, dtype=np.float32)
            sh, sw, _ = sp.shape

            # Paste cat + inverted puddle reflection below paws
            for is_refl in (False, True):
                s_use = sp[::-1].copy() if is_refl else sp
                if is_refl:
                    s_use[..., 3] *= 0.36
                    s_use[..., 0] += 18.0
                    s_use[..., 2] += 36.0
                px0 = int(round(cx - sw * 0.5))
                py0 = int(round((cy + sc * 0.46) if is_refl else (cy - sh * 0.45)))
                x_a, x_b = max(0, px0), min(w, px0 + sw)
                y_a, y_b = max(0, py0), min(h, py0 + sh)
                if x_b > x_a and y_b > y_a:
                    sub = s_use[y_a - py0 : y_b - py0, x_a - px0 : x_b - px0]
                    a = (sub[..., 3:4] / 255.0)
                    frame[y_a:y_b, x_a:x_b] = frame[y_a:y_b, x_a:x_b] * (1.0 - a) + sub[..., :3] * a

        # Tiny cyan-tinted puddle splash glints where paws kick water
        if 0.02 <= progress <= 0.98:
            sx_p = progress * w
            sy_p = h * 0.905
            for k in range(5):
                ang = k * 0.7 + time_s * 19.0
                rx = sx_p - k * 14.0
                ry = sy_p - abs(math.sin(ang)) * 12.0
                cv2.circle(frame, (int(round(rx)), int(round(ry))), 2, (165.0, 232.0, 255.0), -1, cv2.LINE_AA)

    # ── SHOT 1 [0:00–0:03]: EXT. RAIN-SLICKED TOKYO ALLEY – NIGHT ────────────

    def render_shot1_alley(self, time_s: float) -> np.ndarray:
        """`time_s` in `[0.0, 3.2]`: 35mm low-angle slow dolly-in."""
        u_shot = _clamp01(time_s / 3.0)
        zoom = 1.00 + 0.135 * _smootherstep(u_shot)
        tilt = 0.34 - 0.54 * _smootherstep(u_shot)  # Low street grate -> tilts up to flickering neon!
        pan = -0.05 + 0.09 * _smootherstep(u_shot)

        bg = self._sample_parallax(self.alley_bg, self.depth_alley, pan, tilt, zoom)

        # Courier enters smoothly into frame (`0.72s -> 1.55s`) with subtle shake motion on box (`1.62s -> 2.15s`)
        enter_u = _smootherstep((time_s - 0.72) / 0.82)
        if enter_u > 0.004:
            box_shake_y = 0.0
            if 1.58 <= time_s <= 2.22:
                env_s = math.sin(_clamp01((time_s - 1.58) / 0.64) * math.pi)
                box_shake_y = math.sin((time_s - 1.58) * 26.0) * 9.5 * env_s
            c_shift_x = -(1.0 - enter_u) * 85.0
            c_shift_y = (1.0 - enter_u) * 26.0 + box_shake_y * 0.35
            fg = self._sample_parallax(
                self.alley_courier,
                self.depth_alley,
                pan,
                tilt,
                zoom,
                shake_px=(-c_shift_x, -c_shift_y),
            )
            bg = bg * (1.0 - enter_u) + fg * enter_u

            # Ballistic specular water spray flying off chrome box (`1.62s -> 2.38s`)!
            if 1.62 <= time_s <= 2.42:
                bx_c = self.WORK_W * 0.485
                by_c = self.ACT_H * 0.735
                for vx, vy, rad, dly in self.spray_drops:
                    dt = time_s - 1.62 - float(dly)
                    if 0.0 < dt < 0.65:
                        px = bx_c + float(vx) * dt
                        py = by_c + float(vy) * dt + 0.5 * 980.0 * dt * dt
                        fade = 1.0 - dt / 0.65
                        col = (165.0 * fade + 80.0, 232.0 * fade + 20.0, 255.0)
                        cv2.circle(
                            bg,
                            (int(round(px)), int(round(py))),
                            max(1, int(round(rad))),
                            col,
                            -1,
                            cv2.LINE_AA,
                        )

        # Flickering cyan & magenta neon practicals + synchronous wet puddle highlights!
        flicker_cyan = 1.0 + 0.26 * math.sin(time_s * 31.0) * (
            1.0 if (1.85 < time_s < 2.75 or 0.45 < time_s < 0.72) else 0.22
        )
        flicker_mag = 1.0 + 0.14 * math.sin(time_s * 17.3 + 1.1)
        d_cyan = ((self.u - 0.645) / 0.16) ** 2 + ((self.v - 0.19) / 0.22) ** 2
        d_mag = ((self.u - 0.675) / 0.15) ** 2 + ((self.v - 0.41) / 0.20) ** 2
        d_puddle_c = ((self.u - 0.64) / 0.20) ** 2 + ((self.v - 0.88) / 0.16) ** 2
        bg += np.exp(-d_cyan)[..., None] * np.array([12.0, 42.0, 52.0], dtype=np.float32) * (flicker_cyan - 0.85)
        bg += np.exp(-d_mag)[..., None] * np.array([46.0, 10.0, 38.0], dtype=np.float32) * (flicker_mag - 0.88)
        bg += np.exp(-d_puddle_c)[..., None] * np.array([10.0, 34.0, 44.0], dtype=np.float32) * (flicker_cyan - 0.85)

        # Planar puddle raindrop ripple glints in foreground asphalt (`v > 0.74`)
        ripple_wave = np.sin(
            np.hypot((self.u - 0.58) * 42.0, (self.v - 0.86) * 95.0) - time_s * 9.5
        )
        puddle_mask = np.clip((self.v - 0.74) * 4.0, 0.0, 1.0) * np.exp(-((self.u - 0.58) ** 2) / 0.09)
        bg += (np.maximum(ripple_wave, 0.0) ** 3 * puddle_mask)[..., None] * np.array(
            [14.0, 28.0, 36.0], dtype=np.float32
        )

        # Rising volumetric grate steam on foreground left (`u = 0.04..0.38`)
        map_sx = np.mod(self.xx - time_s * 26.0, float(self.WORK_W)).astype(np.float32)
        map_sy = np.mod(self.yy + time_s * 68.0, float(self.ACT_H)).astype(np.float32)
        steam = cv2.remap(self.steam_tex, map_sx, map_sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        steam_envelope = (
            np.exp(-((self.u - 0.17) ** 2) / 0.026)
            * np.clip((self.v - 0.25) / 0.55, 0.0, 1.0)
            * np.clip((1.04 - self.v) * 3.0, 0.0, 1.0)
        )
        steam_alpha = np.clip((steam - 0.42) * 1.45, 0.0, 0.42) * steam_envelope
        bg = bg * (1.0 - steam_alpha[..., None] * 0.55) + np.array([148.0, 182.0, 202.0], dtype=np.float32) * (
            steam_alpha[..., None] * 0.55
        )

        # Stray cat darting swiftly across frame left -> right (`0.20s -> 1.52s`)!
        if 0.18 <= time_s <= 1.56:
            self._draw_running_cat(bg, progress=(time_s - 0.20) / 1.30, time_s=time_s)

        self._add_rain_streaks(bg, time_s, intensity=0.88, wind_x=-9.0, speed=1.0)
        return bg

    # ── SHOT 2 [0:03–0:07]: MEDIUM CLOSE-UP – COURIER'S FACE (`50mm`) ────────

    def render_shot2_mcu(self, time_s: float) -> np.ndarray:
        """`time_s` in `[2.75, 7.20]`: 50mm handheld micro-shake + optical rack focus + emerald hologram."""
        local_t = max(0.0, time_s - 3.0)
        u_shot = _clamp01(local_t / 4.0)

        # Authentic 50mm human shoulder-rig micro-shake
        shx = 5.2 * math.sin(time_s * 4.3) + 2.8 * math.sin(time_s * 8.9 + 0.7)
        shy = 4.0 * math.cos(time_s * 3.7 + 0.4) + 2.3 * math.sin(time_s * 7.6)
        zoom = 1.02 + 0.075 * _smootherstep(u_shot)
        pan = -0.04 + 0.08 * _smootherstep(u_shot)

        subj = self._sample_parallax(
            self.mcu_courier, None, pan, -0.02 + 0.04 * u_shot, zoom, shake_px=(shx, shy)
        )

        # Rolling individual water beads down the translucent raincoat (`local_t >= 0`)!
        for u0, v0, spd, rad, ph in self.coat_beads:
            slip = v0 + local_t * spd + 0.004 * math.sin(local_t * 6.0 + ph)
            cur_v = 0.52 + ((slip - 0.52) % 0.42)
            cur_u = u0 + 0.008 * math.sin(cur_v * 22.0 + ph) - shx / self.WORK_W
            px, py = int(round(cur_u * self.WORK_W)), int(round(cur_v * self.ACT_H))
            if 2 <= px < self.WORK_W - 2 and 2 <= py < self.ACT_H - 2:
                cv2.circle(subj, (px, py + 1), max(1, int(round(rad))), (18.0, 26.0, 30.0), -1, cv2.LINE_AA)
                cv2.circle(subj, (px, py), max(1, int(round(rad))), (195.0, 248.0, 232.0), -1, cv2.LINE_AA)

        # Blooming 3D Emerald Holographic Street Map (`t = 4.18s -> 7.15s`) + Single Glitch (`5.20s–5.38s`)!
        holo_u = _smootherstep((time_s - 4.18) / 0.55)
        if holo_u > 0.005:
            is_glitch = 5.20 <= time_s <= 5.38
            holo_layer = self._render_emerald_hologram(time_s, holo_u, is_glitch, shx, shy)
            subj += holo_layer

            # Interactive cool emerald key-light bounce on courier's face & raincoat!
            face_key = np.exp(-(((self.u - 0.535) / 0.17) ** 2 + ((self.v - 0.36) / 0.28) ** 2))
            flick = 1.45 if is_glitch else (0.94 + 0.06 * math.sin(time_s * 14.0))
            subj += face_key[..., None] * np.array([6.0, 38.0, 26.0], dtype=np.float32) * holo_u * flick

        # Cold-air visible breath vapor plume on whispered "Wrong block." (`5.62s -> 6.85s`)!
        if 5.60 <= time_s <= 6.95:
            b_prog = _clamp01((time_s - 5.60) / 1.30)
            b_env = math.sin(b_prog * math.pi) ** 0.75
            mouth_u = 0.545 + 0.055 * b_prog
            mouth_v = 0.445 + 0.030 * b_prog - 0.025 * (b_prog ** 2)
            spread_u = 0.022 + 0.065 * b_prog
            spread_v = 0.020 + 0.052 * b_prog
            map_bx = np.mod(self.xx - time_s * 42.0, float(self.WORK_W)).astype(np.float32)
            map_by = np.mod(self.yy - time_s * 18.0, float(self.ACT_H)).astype(np.float32)
            bt = cv2.remap(self.steam_tex, map_bx, map_by, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            puff = np.exp(-(((self.u - mouth_u) / spread_u) ** 2 + ((self.v - mouth_v) / spread_v) ** 2))
            vapor = puff * (0.45 + 0.75 * bt) * b_env * 0.52
            subj = subj * (1.0 - vapor[..., None] * 0.45) + np.array([196.0, 228.0, 224.0], dtype=np.float32) * (
                vapor[..., None] * 0.45
            )

        self._add_rain_streaks(subj, time_s, intensity=0.52, wind_x=-6.0, speed=0.85)

        # TRUE OPTICAL 2x ANAMORPHIC OVAL RACK FOCUS (`3.00s -> 4.30s`):
        # Pulls focus smoothly from refractive raindrops ON THE LENS -> COURIER'S EYES!
        rack = _smootherstep((time_s - 3.12) / 1.18)  # 0 = focus on lens drops; 1 = tack-sharp eyes!
        sharp_ref = subj.copy()
        subj_sigma_x = (1.0 - rack) * 6.8
        subj_sigma_y = (1.0 - rack) * 11.8  # 2x Anamorphic vertical oval bokeh squeeze!
        if subj_sigma_x > 0.35:
            subj = cv2.GaussianBlur(subj, (0, 0), sigmaX=subj_sigma_x, sigmaY=subj_sigma_y)

        # Photorealistic Refractive Glass Water Droplets on Camera Lens (NO outline circles — pure refraction!)
        if rack < 0.96:
            refract_canvas = subj.copy()
            for du, dv, drad, tint_u in self.lens_drops:
                cx_d = float(du) * self.WORK_W
                cy_d = ((float(dv) + local_t * 0.006) % 1.0) * self.ACT_H
                r_px = float(drad) * 1.35
                x0 = max(0, int(cx_d - r_px - 2))
                x1 = min(self.WORK_W, int(cx_d + r_px + 3))
                y0 = max(0, int(cy_d - r_px * 1.25 - 2))
                y1 = min(self.ACT_H, int(cy_d + r_px * 1.25 + 3))
                if x1 <= x0 + 2 or y1 <= y0 + 2:
                    continue
                yy_l, xx_l = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                nx_d = (xx_l - cx_d) / r_px
                ny_d = (yy_l - cy_d) / (r_px * 1.18)
                rr2 = nx_d * nx_d + ny_d * ny_d
                inside_d = np.clip((1.0 - rr2) * 4.5, 0.0, 1.0)[..., None]
                # Spherical water meniscus refracts sharp scene colors + tiny warm/cyan rim glint
                sx_r = np.clip(cx_d - nx_d * r_px * 2.4, 0.0, self.WORK_W - 1.0)
                sy_r = np.clip(cy_d - ny_d * r_px * 2.4, 0.0, self.ACT_H - 1.0)
                refracted = cv2.remap(sharp_ref, sx_r, sy_r, cv2.INTER_LINEAR)
                rim_glint = np.exp(-((rr2 - 0.78) ** 2) / 0.035)[..., None] * 55.0
                refract_canvas[y0:y1, x0:x1] = (
                    refract_canvas[y0:y1, x0:x1] * (1.0 - inside_d)
                    + (refracted * 1.14 + rim_glint) * inside_d
                )
            if rack > 0.06:
                refract_canvas = cv2.GaussianBlur(
                    refract_canvas, (0, 0), sigmaX=rack * 7.5, sigmaY=rack * 12.5
                )
            blend_d = (1.0 - rack) ** 1.25
            subj = subj * (1.0 - blend_d) + refract_canvas * blend_d
        return subj

    def _render_emerald_hologram(
        self, time_s: float, bloom: float, is_glitch: bool, shx: float, shy: float
    ) -> np.ndarray:
        """Draws crisp, vivid 3D emerald street-block hologram floating above the wrist."""
        h, w = self.ACT_H, self.WORK_W
        canvas = np.zeros((h, w, 3), dtype=np.float32)
        wx = w * 0.665 - shx
        wy = h * 0.885 - shy
        hx = w * 0.652 - shx
        hy = wy - (h * 0.285) * bloom

        # Volumetric emerald projection beam rising out of the smartwatch bezel
        cone_pts = np.array(
            [
                [wx - 22, wy],
                [wx + 22, wy],
                [hx + 215 * bloom, hy + 34 * bloom],
                [hx - 215 * bloom, hy + 34 * bloom],
            ],
            dtype=np.int32,
        )
        cone = np.zeros((h, w, 3), dtype=np.float32)
        cv2.fillPoly(cone, [cone_pts], (24.0, 215.0, 138.0))
        cone = cv2.GaussianBlur(cone, (0, 0), sigmaX=20.0, sigmaY=16.0)
        canvas += cone * 0.52 * bloom

        # Subtle dark emerald holographic glass plane so bright green 3D lines pop with high contrast!
        plane = np.zeros((h, w), dtype=np.float32)
        rot = 0.52 + 0.14 * math.sin(time_s * 1.4)
        ca, sa = math.cos(rot), math.sin(rot)
        scale_g = 148.0 * bloom

        def iso_pt(gx: float, gy: float, gz: float = 0.0) -> tuple[int, int]:
            rx = gx * ca - gy * sa
            ry = gx * sa + gy * ca
            sx_p = hx + rx * scale_g
            sy_p = hy + ry * scale_g * 0.42 - gz * scale_g
            return int(round(sx_p)), int(round(sy_p))

        base_quad = np.array([iso_pt(-1.18, -1.18), iso_pt(1.18, -1.18), iso_pt(1.18, 1.18), iso_pt(-1.18, 1.18)], dtype=np.int32)
        cv2.fillPoly(plane, [base_quad], 1.0)
        plane = cv2.GaussianBlur(plane, (0, 0), sigmaX=6.0)
        # Scanline modulation across holographic deck
        scan = 0.72 + 0.28 * np.sin(self.yy * 1.4 - time_s * 28.0)
        canvas += (plane * scan)[..., None] * np.array([8.0, 74.0, 48.0], dtype=np.float32) * bloom

        grid = np.zeros((h, w, 3), dtype=np.float32)
        em_bright = (65.0, 255.0, 178.0) if not is_glitch else (165.0, 255.0, 225.0)
        em_dim = (26.0, 185.0, 118.0)

        for g in np.linspace(-1.1, 1.1, 7):
            cv2.line(grid, iso_pt(float(g), -1.1), iso_pt(float(g), 1.1), em_dim, 2, cv2.LINE_AA)
            cv2.line(grid, iso_pt(-1.1, float(g)), iso_pt(1.1, float(g)), em_dim, 2, cv2.LINE_AA)

        # Wireframe 3D street blocks rising off the map plane
        blocks = [(-0.65, -0.55, 0.35, 0.34), (0.22, -0.60, 0.42, 0.52), (-0.70, 0.25, 0.40, 0.28), (0.28, 0.22, 0.38, 0.62)]
        for bx0, by0, bw, bh_z in blocks:
            b00, b10 = iso_pt(bx0, by0, 0), iso_pt(bx0 + bw, by0, 0)
            b11, b01 = iso_pt(bx0 + bw, by0 + bw, 0), iso_pt(bx0, by0 + bw, 0)
            t00, t10 = iso_pt(bx0, by0, bh_z), iso_pt(bx0 + bw, by0, bh_z)
            t11, t01 = iso_pt(bx0 + bw, by0 + bw, bh_z), iso_pt(bx0, by0 + bw, bh_z)
            for p_a, p_b in ((t00, t10), (t10, t11), (t11, t01), (t01, t00), (b00, t00), (b10, t10), (b11, t11), (b01, t01)):
                cv2.line(grid, p_a, p_b, em_bright, 2, cv2.LINE_AA)

        # Glowing route polyline + pulsing destination marker
        route = [iso_pt(-0.9, 0.7, 0.02), iso_pt(-0.1, 0.7, 0.02), iso_pt(-0.1, -0.1, 0.02), iso_pt(0.47, 0.41, 0.02)]
        for i in range(len(route) - 1):
            cv2.line(grid, route[i], route[i + 1], (155.0, 255.0, 215.0), 3, cv2.LINE_AA)
        target_xy = iso_pt(0.47, 0.41, 0.04)
        ping_col = (255.0, 95.0, 82.0) if time_s >= 5.20 else (160.0, 255.0, 220.0)
        cv2.circle(grid, target_xy, int(8 + 6 * abs(math.sin(time_s * 9.0))), ping_col, 3, cv2.LINE_AA)
        cv2.circle(grid, (int(round(wx)), int(round(wy))), int(11 * bloom), (110.0, 255.0, 195.0), -1, cv2.LINE_AA)

        # Single Hologram Glitch (`5.20s – 5.38s`): unmistakable horizontal tear + RGB split!
        if is_glitch:
            rng_g = np.random.default_rng(int(time_s * 1000))
            for _ in range(12):
                y_sl = int(rng_g.integers(max(0, int(hy - 150)), min(h - 14, int(wy))))
                h_sl = int(rng_g.integers(5, 16))
                shift_x = int(rng_g.integers(-55, 56))
                grid[y_sl : y_sl + h_sl] = np.roll(grid[y_sl : y_sl + h_sl], shift_x, axis=1)
            grid[..., 0] = np.roll(grid[..., 1], 14, axis=1) * 0.75

        glow = cv2.GaussianBlur(grid, (0, 0), sigmaX=8.5)
        return canvas + grid * 1.15 + glow * 0.95

    # ── SHOT 3 [0:07–0:11]: WIDE TRACKING ROOFTOP CHASE (`24mm`) ─────────────

    def render_shot3_rooftop(self, time_s: float) -> np.ndarray:
        """`time_s` in `[6.75, 11.20]`: 24mm barrel tracking leap -> AC unit slide & seed case open."""
        local_t = max(0.0, time_s - 7.0)

        # Sub-beat A (`7.00s – 9.05s`): Wide lateral rooftop sprint, leap, cable sparks & searchlight
        # Sub-beat B (`8.85s – 11.00s`): Smooth AC-unit foreground wipe into crouched chrome-box reveal
        ac_wipe = _smootherstep((time_s - 8.82) / 0.42)

        out_a: np.ndarray | None = None
        if ac_wipe < 0.998:
            track_u = _clamp01(local_t / 2.05)
            # Landing impact jolt at `t = 8.50s` when feet hit corrugated metal roof!
            jolt = math.exp(-max(0.0, time_s - 8.50) * 14.0) * (14.0 if time_s >= 8.50 else 0.0)
            pan_center = -0.92 + 1.78 * _smootherstep(track_u)
            tilt_center = 0.14 - 0.24 * math.sin(track_u * math.pi)  # Dynamic drone arc up over leap apex!
            # True 5-subframe 180-deg lateral tracking motion blur on the hyperreal leap plate (ZERO cutout halos!)
            shutter_vel = math.sin(track_u * math.pi) * 0.055
            out_a = np.zeros((self.ACT_H, self.WORK_W, 3), dtype=np.float32)
            for tap in np.linspace(-0.5, 0.5, 5):
                out_a += self._sample_parallax(
                    self.rooftop_leap,
                    self.depth_rooftop,
                    pan_center + float(tap) * shutter_vel,
                    tilt_center,
                    zoom=1.06 + 0.06 * track_u,
                    shake_px=(math.sin(time_s * 16.0) * 2.6, jolt * math.cos((time_s - 8.50) * 38.0)),
                    barrel_k=-0.052,  # Authentic 24mm wide-angle barrel distortion!
                )
            out_a /= 5.0

            # Wind displacement ripple across hanging laundry zone (`u = 0.50..0.74, v = 0.30..0.54`)
            wind_mask = (
                np.exp(-(((self.u - 0.62) / 0.14) ** 2 + ((self.v - 0.42) / 0.11) ** 2))
            ).astype(np.float32)
            wx_map = (self.xx + np.sin(self.xx * 0.055 - time_s * 15.0) * 8.5 * wind_mask).astype(np.float32)
            wy_map = (self.yy + np.cos(self.xx * 0.045 - time_s * 13.0) * 5.5 * wind_mask).astype(np.float32)
            out_a = cv2.remap(out_a, wx_map, wy_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

            # Sweeping red security drone searchlight cone + horizontal anamorphic lens flare!
            beam_cx = self.WORK_W * (0.28 + 0.30 * math.sin((local_t - 0.2) * 1.65))
            beam_cy = self.ACT_H * 0.145
            ang_sweep = math.radians(32.0 + 36.0 * math.sin(local_t * 2.15))
            dir_x, dir_y = math.cos(ang_sweep), math.sin(ang_sweep)
            rx = self.xx - beam_cx
            ry = self.yy - beam_cy
            proj = np.maximum(rx * dir_x + ry * dir_y, 0.0)
            perp = np.abs(-rx * dir_y + ry * dir_x)
            cone_w = 14.0 + proj * 0.20
            beam = np.exp(-((perp / cone_w) ** 2)) * np.exp(-proj / (self.WORK_W * 0.62)) * (proj > 0)
            out_a += beam[..., None] * np.array([155.0, 22.0, 26.0], dtype=np.float32) * 0.68
            cv2.circle(out_a, (int(round(beam_cx)), int(round(beam_cy))), 6, (255.0, 165.0, 165.0), -1, cv2.LINE_AA)

            # Horizontal Panavision anamorphic streak flare across screen!
            flare_strength = 0.55 + 0.45 * math.sin(local_t * 3.1)
            v_dist = (self.yy - beam_cy) / (self.ACT_H * 0.014)
            u_dist = (self.xx - beam_cx) / (self.WORK_W * 0.34)
            anamorphic_streak = np.exp(-(v_dist * v_dist)) * np.exp(-(u_dist * u_dist))
            out_a += anamorphic_streak[..., None] * np.array([255.0, 82.0, 96.0], dtype=np.float32) * flare_strength

            # Snapped overhead cable sparks (`8.18s -> 9.05s`)!
            if time_s >= 8.18:
                sp_origin_x = self.WORK_W * 0.72
                sp_origin_y = self.ACT_H * 0.24
                for svx, svy, srad, sdly in self.sparks:
                    dt = time_s - 8.18 - float(sdly)
                    if 0.0 < dt < 0.72:
                        px = sp_origin_x + float(svx) * dt
                        py = sp_origin_y + float(svy) * dt + 0.5 * 1150.0 * dt * dt
                        px0 = sp_origin_x + float(svx) * max(0.0, dt - 0.03)
                        py0 = sp_origin_y + float(svy) * max(0.0, dt - 0.03) + 0.5 * 1150.0 * max(0.0, dt - 0.03) ** 2
                        fade = 1.0 - dt / 0.72
                        cv2.line(
                            out_a,
                            (int(round(px0)), int(round(py0))),
                            (int(round(px)), int(round(py))),
                            (255.0, 205.0 * fade + 40.0, 110.0 * fade),
                            max(1, int(round(srad))),
                            cv2.LINE_AA,
                        )
            self._add_rain_streaks(out_a, time_s, intensity=0.95, wind_x=-22.0, speed=1.35)

        out_b: np.ndarray | None = None
        if ac_wipe > 0.002:
            b_u = _clamp01((time_s - 8.85) / 2.15)
            zoom_b = 1.02 + 0.13 * _smootherstep(b_u)
            pan_b = 0.16 - 0.24 * _smootherstep(b_u)
            out_b = self._sample_parallax(
                self.rooftop_ac,
                None,
                pan_b,
                -0.04 + 0.07 * b_u,
                zoom_b,
                shake_px=(3.2 * math.sin(time_s * 5.1), 2.5 * math.cos(time_s * 4.4)),
                barrel_k=-0.035,
            )

            # Seed inside open chrome box (`u ≈ 0.47, v ≈ 0.65`) pulses like a living heartbeat!
            hb = _heartbeat(time_s, period=0.78, phase=9.80) if time_s >= 9.65 else 0.0
            open_env = _smootherstep((time_s - 9.55) / 0.45)
            d_seed = ((self.u - 0.472) / 0.13) ** 2 + ((self.v - 0.655) / 0.19) ** 2
            pulse_gain = open_env * (0.35 + 1.05 * hb)
            out_b += np.exp(-d_seed)[..., None] * np.array([24.0, 138.0, 255.0], dtype=np.float32) * pulse_gain
            # Sweeping red searchlight glow behind the AC unit
            d_red = ((self.u - (0.55 + 0.14 * math.sin(time_s * 2.3))) / 0.22) ** 2 + ((self.v - 0.18) / 0.18) ** 2
            out_b += np.exp(-d_red)[..., None] * np.array([92.0, 14.0, 18.0], dtype=np.float32) * 0.48
            self._add_rain_streaks(out_b, time_s, intensity=0.78, wind_x=-15.0, speed=1.15)

        if out_a is None:
            assert out_b is not None
            return out_b
        if out_b is None:
            return out_a

        # Smooth foreground AC-unit dark column wipe from wide rooftop into crouched close shot
        edge = -0.15 + 1.30 * ac_wipe
        m = np.clip((edge - self.u) / 0.14, 0.0, 1.0)[..., None]
        dark_band = np.exp(-(((self.u - edge) / 0.045) ** 2))[..., None]
        blended = out_a * (1.0 - m) + out_b * m
        return blended * (1.0 - 0.72 * dark_band)

    # ── SHOT 4 [0:11–0:15]: EXTREME CLOSE-UP – THE SEED (`120fps -> 24fps`) ──

    def render_shot4_macro_seed(self, time_s: float) -> np.ndarray:
        """
        `time_s` in `[10.75, 15.00]`:
          - `11.00s .. 14.416s` (`Frames 264 .. 345`): Slow-motion macro push-in,
            intensifying seed heartbeat, crawling bioluminescent light roots on
            glove, animated blue caustics & subtle smile on courier's face.
          - `>= 14.416s` (`Frame 346 .. 359`): HARD CINEMA CUT TO PURE BLACK on chime!
        """
        if time_s >= 14.416:
            return np.zeros((self.ACT_H, self.WORK_W, 3), dtype=np.float32)

        local_t = max(0.0, time_s - 11.0)
        u_macro = _clamp01(local_t / 3.42)

        # Progressive sinuous radial growth mask so light-roots crawl live across the wet glove!
        ph, pw, _ = self.macro_clean_glove.shape
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        un, vn = xx / pw, yy / ph
        ang = np.arctan2(vn - 0.345, (un - 0.428) * 1.45)
        dist = np.hypot((un - 0.428) * 1.45, vn - 0.345)
        wiggle = 0.045 * np.sin(ang * 5.0 + local_t * 0.9) + 0.025 * np.cos(ang * 9.0)
        reach = 0.075 + 0.62 * _smootherstep(_clamp01((local_t - 0.15) / 2.85)) + wiggle
        crawl_mask = np.clip((reach - dist) / 0.055, 0.0, 1.0)[..., None]
        tip_glow = np.exp(-(((dist - reach) / 0.022) ** 2))[..., None] * 1.65

        hb = _heartbeat(time_s, period=0.96, phase=11.05)
        intensity_ramp = 0.55 + 0.85 * u_macro + 0.42 * hb

        plate = self.macro_clean_glove + self.macro_roots_only * crawl_mask * (intensity_ramp + tip_glow * 0.35)

        # Subtle natural upward corner-of-mouth smile displacement on courier's softly lit face (`u > 0.60`)
        smile_amt = _smootherstep(_clamp01((local_t - 1.15) / 1.85))
        if smile_amt > 0.01:
            corner_mask = (
                np.exp(-(((un - 0.785) / 0.065) ** 2 + ((vn - 0.635) / 0.075) ** 2))
                + 0.65 * np.exp(-(((un - 0.680) / 0.045) ** 2 + ((vn - 0.640) / 0.065) ** 2))
            ).astype(np.float32)
            map_sx = (xx - corner_mask * (pw * 0.0045) * smile_amt).astype(np.float32)
            map_sy = (yy + corner_mask * (ph * 0.0165) * smile_amt).astype(np.float32)
            plate = cv2.remap(plate, map_sx, map_sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # Ultra-smooth 120fps -> 24fps macro push-in
        zoom = 1.00 + 0.125 * _smootherstep(u_macro)
        frame = self._sample_parallax(
            plate,
            None,
            pan_x=-0.04 + 0.07 * u_macro,
            tilt_y=0.02 - 0.05 * u_macro,
            zoom=zoom,
            shake_px=(1.1 * math.sin(time_s * 1.3), 0.9 * math.cos(time_s * 1.1)),  # Slow-mo float!
        )

        # Seed core intensification + horizontal sapphire anamorphic bloom!
        d_core = ((self.u - 0.428) / 0.085) ** 2 + ((self.v - 0.345) / 0.16) ** 2
        d_hstreak = ((self.v - 0.345) / 0.022) ** 2 + ((self.u - 0.428) / 0.30) ** 2
        frame += np.exp(-d_core)[..., None] * np.array([28.0, 145.0, 255.0], dtype=np.float32) * (
            0.22 + 0.48 * u_macro + 0.34 * hb
        )
        frame += np.exp(-d_hstreak)[..., None] * np.array([45.0, 175.0, 255.0], dtype=np.float32) * (
            0.12 + 0.26 * u_macro + 0.20 * hb
        )

        # Animated Blue Optical Water Caustics dancing across courier's smiling face (`u > 0.56`)!
        t_slow = local_t * 0.85  # Overcranked 120fps dreamy speed
        c1 = np.sin(self.u * 46.0 + np.sin(self.v * 29.0 - t_slow * 2.1) * 2.4 + t_slow * 1.7)
        c2 = np.cos(self.v * 41.0 + np.cos(self.u * 33.0 + t_slow * 1.9) * 2.2 - t_slow * 1.5)
        caustic_net = np.power(np.clip(1.0 - np.abs(c1 + c2) * 0.52, 0.0, 1.0), 4.0)
        face_mask = np.exp(-(((self.u - 0.735) / 0.16) ** 2 + ((self.v - 0.44) / 0.44) ** 2))
        frame += (caustic_net * face_mask)[..., None] * np.array([18.0, 128.0, 235.0], dtype=np.float32) * (
            0.24 + 0.44 * u_macro + 0.16 * hb
        )

        # 120fps slow-motion suspended bioluminescent spores drifting weightlessly!
        spore_layer = np.zeros_like(frame)
        for su, sv, sspd, srad, sph in self.spores:
            px = ((su + math.sin(local_t * 0.55 + sph) * 0.022) % 1.0) * self.WORK_W
            py = ((sv - local_t * sspd) % 1.0) * self.ACT_H
            twink = 0.45 + 0.55 * math.sin(local_t * 2.2 + sph)
            cv2.circle(
                spore_layer,
                (int(round(px)), int(round(py))),
                max(1, int(round(srad))),
                (65.0 * twink, 195.0 * twink, 255.0 * twink),
                -1,
                cv2.LINE_AA,
            )
        frame += cv2.GaussianBlur(spore_layer, (0, 0), sigmaX=1.4) * 0.65
        return frame

    # ── Master Seamless Compositor + Teal/Amber Anamorphic Cinema Grade ──────

    def render_frame_1080p_anamorphic(self, frame_idx: int, fps: int = 24) -> np.ndarray:
        time_s = frame_idx / float(fps)

        # Hard cut to pure black at `14.42s` (`Frame 346..359`) — NO noise, PURE #000000 black!
        if frame_idx >= 346:
            return np.zeros((self.WORK_H, self.WORK_W, 3), dtype=np.uint8)

        # Seamless hidden cinema transitions (ZERO jump cuts, ZERO face morphing!)
        if time_s < 2.75:
            act_rgb = self.render_shot1_alley(time_s)
        elif time_s < 3.17:
            # Transition 1 -> 2: Dark rain-slick alley pillar passing close across lens!
            w_u = _smootherstep((time_s - 2.75) / 0.42)
            a = self.render_shot1_alley(time_s)
            b = self.render_shot2_mcu(time_s)
            edge = -0.18 + 1.36 * w_u
            mask = np.clip((edge - self.u) / 0.13, 0.0, 1.0)[..., None]
            pillar_shadow = np.exp(-(((self.u - edge) / 0.055) ** 2))[..., None]
            act_rgb = (a * (1.0 - mask) + b * mask) * (1.0 - 0.86 * pillar_shadow)
        elif time_s < 6.79:
            act_rgb = self.render_shot2_mcu(time_s)
        elif time_s < 7.21:
            # Transition 2 -> 3: Fast lateral/upward 9-tap shutter whip pan into Rooftop Chase!
            w_u = (time_s - 6.79) / 0.42
            ease_w = _smootherstep(w_u)
            a = self.render_shot2_mcu(time_s)
            b = self.render_shot3_rooftop(time_s)
            combo = np.concatenate([a, b], axis=1)
            shift = ease_w * self.WORK_W
            blur_px = math.sin(_clamp01(w_u) * math.pi) * (self.WORK_W * 0.095)
            acc = np.zeros_like(a)
            for tap in np.linspace(-0.5, 0.5, 9):
                M = np.float32([[1.0, 0.0, -float(shift + tap * blur_px)], [0.0, 1.0, 0.0]])
                acc += cv2.warpAffine(combo, M, (self.WORK_W, self.ACT_H), borderMode=cv2.BORDER_REPLICATE)
            act_rgb = acc / 9.0
        elif time_s < 10.75:
            act_rgb = self.render_shot3_rooftop(time_s)
        elif time_s < 11.17:
            # Transition 3 -> 4: Optical push-in sapphire bloom from open case seed -> macro seed!
            w_u = _smootherstep((time_s - 10.75) / 0.42)
            a = self.render_shot3_rooftop(time_s)
            b = self.render_shot4_macro_seed(time_s)
            M_a = cv2.getRotationMatrix2D((self.WORK_W * 0.47, self.ACT_H * 0.65), 0.0, 1.0 + 0.35 * w_u)
            M_b = cv2.getRotationMatrix2D((self.WORK_W * 0.43, self.ACT_H * 0.35), 0.0, 0.86 + 0.14 * w_u)
            wa = cv2.warpAffine(a, M_a, (self.WORK_W, self.ACT_H), borderMode=cv2.BORDER_REFLECT_101)
            wb = cv2.warpAffine(b, M_b, (self.WORK_W, self.ACT_H), borderMode=cv2.BORDER_REFLECT_101)
            sapphire_flash = math.sin(_clamp01((time_s - 10.75) / 0.42) * math.pi)
            act_rgb = wa * (1.0 - w_u) + wb * w_u + np.array([18.0, 95.0, 210.0], dtype=np.float32) * (
                sapphire_flash * 0.30
            )
        else:
            act_rgb = self.render_shot4_macro_seed(time_s)

        # Cinematic Teal-Shadows / Amber-Highlights Color Grade + ACES Highlight Roll-Off
        graded = self._apply_teal_amber_grade(act_rgb, frame_idx=frame_idx)

        # Gentle 10-frame opening fade from black (`0.0s -> 0.42s`)
        if frame_idx < 10:
            graded *= _smootherstep((frame_idx + 1.0) / 10.0)

        out = np.zeros((self.WORK_H, self.WORK_W, 3), dtype=np.uint8)
        out[self.BAR_PX : self.BAR_PX + self.ACT_H] = np.clip(graded, 0.0, 255.0).astype(np.uint8)
        return out

    def _apply_teal_amber_grade(self, rgb: np.ndarray, *, frame_idx: int) -> np.ndarray:
        lin = np.clip(rgb / 255.0, 0.0, 2.5) ** 2.2
        luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]

        # Lift shadows gently toward deep cinema teal; warm mid-highs toward amber skin/lantern tones
        shadow_w = np.clip(1.0 - luma * 3.2, 0.0, 1.0)[..., None]
        high_w = np.clip((luma - 0.18) * 1.4, 0.0, 1.0)[..., None]
        lin = (
            lin
            + shadow_w * np.array([0.0012, 0.0052, 0.0082], dtype=np.float32)
            + high_w * lin * np.array([0.075, 0.022, -0.045], dtype=np.float32)
        )

        # Subtle anamorphic edge vignette
        nx = (self.u - 0.5) * 2.0
        ny = (self.v - 0.5) * 2.0
        vig = np.clip(1.0 - 0.16 * (nx * nx * 0.85 + ny * ny * 0.65), 0.72, 1.0)[..., None]
        lin *= vig

        # ACES filmic roll-off
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        x = np.maximum(lin * 1.12, 0.0)
        mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
        srgb = (mapped ** (1.0 / 2.2)) * 255.0

        # Subtle organic 35mm photochemical luma/chroma grain (cut-to-black frames stay 100% pure 0)
        rng = np.random.default_rng(90000 + frame_idx)
        grain = rng.normal(0.0, 3.1, size=(self.ACT_H // 2, self.WORK_W // 2, 1)).astype(np.float32)
        grain_up = cv2.resize(grain, (self.WORK_W, self.ACT_H), interpolation=cv2.INTER_LINEAR)[..., None]
        return srgb + grain_up


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL FOLEY, WHISPERED DIALOGUE & CHIME SOUND DESIGN (`44.1 kHz` STEREO)
# ══════════════════════════════════════════════════════════════════════════════


def _bandpass_whisper_formants(
    noise: np.ndarray, sr: int, f_center: float, bw: float
) -> np.ndarray:
    """2nd-order digital resonator filter for realistic aspirated human whisper formants."""
    r = math.exp(-math.pi * bw / sr)
    theta = 2.0 * math.pi * f_center / sr
    a1 = -2.0 * r * math.cos(theta)
    a2 = r * r
    gain = (1.0 - r) * math.sqrt(1.0 + r * r - 2.0 * r * math.cos(2.0 * theta))
    from scipy.signal import lfilter

    return lfilter([gain], [1.0, a1, a2], noise).astype(np.float32)


def write_tokyo_courier_soundtrack(wav_path: Path, duration_sec: float = 15.0) -> None:
    """
    Synthesizes ALL 9 scripted spatial audio layers locked frame-for-frame to the film:
      1. Rain patter (stereo multi-band + individual raincoat droplets, hard cut at 14.416s)
      2. Distant Tokyo city hum + neon transformer warmth
      3. Smartwatch crystalline beep (`4.18s`) + hologram glitch zap (`5.22s`)
      4. Whispered `"Wrong block."` (`5.65s – 6.45s` vocal formant whisper synthesis)
      5. Vinyl raincoat fabric swishes (`1.65s`, `7.75s`, `9.20s`)
      6. Corrugated sheet-metal rooftop landing clang (`8.50s`) + electric sparks (`8.20s–9.05s`) + case latch (`9.68s`)
      7. Security drone multi-rotor Doppler whir (`7.05s – 10.85s`)
      8. Deep sub-bass cardiac heartbeat pulses on seed reveal (`9.95s..14.35s`)
      9. Single soft crystalline harmonic chime right on the `14.416s` hard cut to black!
    """
    from scipy.signal import butter, sosfilt

    sr = 44100
    n = int(round(duration_sec * sr))
    t = np.linspace(0.0, duration_sec, n, endpoint=False, dtype=np.float32)
    cut_idx = int(round(14.416 * sr))  # Frame 346 @ 24 fps -> exact picture cut to black!

    rng = np.random.default_rng(204901)
    white_l = rng.normal(0.0, 1.0, n).astype(np.float32)
    white_r = rng.normal(0.0, 1.0, n).astype(np.float32)

    # 1. Stereo Rain Patter + Individual Raincoat Water Droplet Patter
    sos_rain = butter(2, [550.0, 5200.0], btype="bandpass", fs=sr, output="sos")
    rain_l = sosfilt(sos_rain, white_l).astype(np.float32) * 0.048
    rain_r = sosfilt(sos_rain, white_r).astype(np.float32) * 0.048
    # Attenuate rain dreamingly during 120fps slow-motion macro Shot 4 (`11.0s..14.416s`)
    macro_duck = 1.0 - 0.62 * np.clip((t - 11.0) / 0.8, 0.0, 1.0)
    rain_l *= macro_duck
    rain_r *= macro_duck

    # Individual water droplet plinks on raincoat / puddles
    for _ in range(210):
        dt_on = float(rng.uniform(0.05, 14.25))
        i0 = int(dt_on * sr)
        ln = min(int(0.035 * sr), n - i0)
        if ln > 10:
            tt = np.arange(ln, dtype=np.float32) / sr
            f_drop = float(rng.uniform(950.0, 2900.0))
            plink = 0.026 * np.sin(2.0 * math.pi * f_drop * (1.0 + 1.4 * tt) * tt) * np.exp(-tt * 145.0)
            pan = float(rng.uniform(0.15, 0.85))
            rain_l[i0 : i0 + ln] += plink * pan
            rain_r[i0 : i0 + ln] += plink * (1.0 - pan)

    # 2. Distant City Hum + Neon Ballast Drone
    sos_city = butter(2, 165.0, btype="lowpass", fs=sr, output="sos")
    hum_noise = sosfilt(sos_city, white_l + white_r).astype(np.float32) * 0.055
    city_l = hum_noise + 0.022 * np.sin(2.0 * math.pi * 55.0 * t) + 0.011 * np.sin(2.0 * math.pi * 110.2 * t)
    city_r = hum_noise + 0.022 * np.sin(2.0 * math.pi * 55.0 * t + 0.4) + 0.011 * np.sin(2.0 * math.pi * 165.0 * t)

    fx_l = np.zeros(n, dtype=np.float32)
    fx_r = np.zeros(n, dtype=np.float32)

    def add_mono_event(onset_s: float, sig: np.ndarray, pan: float = 0.5) -> None:
        i0 = int(round(onset_s * sr))
        if i0 >= n:
            return
        ln = min(len(sig), n - i0)
        fx_l[i0 : i0 + ln] += sig[:ln] * math.sqrt(1.0 - pan)
        fx_r[i0 : i0 + ln] += sig[:ln] * math.sqrt(pan)

    # 3. Smartwatch Crystalline Beep (`4.18s`) + Single Hologram Glitch Zap (`5.21s`)
    tt_beep = np.arange(int(0.24 * sr), dtype=np.float32) / sr
    beep = (
        0.13 * np.sin(2.0 * math.pi * 1760.0 * tt_beep) * np.exp(-tt_beep * 28.0)
        + 0.11 * np.sin(2.0 * math.pi * 2637.0 * np.maximum(0.0, tt_beep - 0.065))
        * (tt_beep >= 0.065)
        * np.exp(-np.maximum(0.0, tt_beep - 0.065) * 22.0)
    )
    add_mono_event(4.18, beep.astype(np.float32), pan=0.62)

    tt_gl = np.arange(int(0.17 * sr), dtype=np.float32) / sr
    glitch_zap = (
        0.10
        * np.sign(np.sin(2.0 * math.pi * (340.0 + 920.0 * np.sin(tt_gl * 95.0)) * tt_gl))
        * white_l[: len(tt_gl)]
        * np.exp(-tt_gl * 14.0)
    )
    add_mono_event(5.20, glitch_zap.astype(np.float32), pan=0.62)

    # 4. Aspirated Formant Whisper: `"Wrong block."` (`5.65s – 6.45s`)
    w_len = int(0.82 * sr)
    w_noise = rng.normal(0.0, 1.0, w_len).astype(np.float32)
    wt = np.arange(w_len, dtype=np.float32) / sr
    # Word 1 ("Wrong": `0.00..0.34s` — rounded labiovelar `/w r ɔ ŋ/`)
    env_w1 = (np.maximum(np.sin(np.clip(wt / 0.34, 0.0, 1.0) * math.pi), 0.0) ** 0.7) * (wt < 0.34)
    w1 = (
        _bandpass_whisper_formants(w_noise, sr, 420.0, 130.0) * 0.9
        + _bandpass_whisper_formants(w_noise, sr, 820.0, 170.0) * 0.8
        + _bandpass_whisper_formants(w_noise, sr, 2050.0, 320.0) * 0.45
    ) * env_w1
    # Word 2 ("block": `/b/` burst at `0.42s` -> `/l ɑ/` `0.42..0.66s` -> `/k/` aspirated release `0.69..0.79s`)
    env_w2 = (np.maximum(np.sin(np.clip((wt - 0.42) / 0.24, 0.0, 1.0) * math.pi), 0.0) ** 0.65) * ((wt >= 0.42) & (wt < 0.66))
    w2 = (
        _bandpass_whisper_formants(w_noise, sr, 690.0, 150.0) * 0.9
        + _bandpass_whisper_formants(w_noise, sr, 1180.0, 210.0) * 0.85
        + _bandpass_whisper_formants(w_noise, sr, 2450.0, 380.0) * 0.55
    ) * env_w2
    k_burst = _bandpass_whisper_formants(w_noise, sr, 2850.0, 900.0) * np.exp(-np.maximum(0.0, wt - 0.69) * 38.0) * (wt >= 0.69) * 0.75
    whisper_sig = (w1 + w2 + k_burst) * 0.16
    add_mono_event(5.65, whisper_sig.astype(np.float32), pan=0.52)

    # 5. Raincoat Vinyl Fabric Swishes (`1.64s` box shake, `7.65s` rooftop leap, `9.12s` AC slide)
    sos_swish = butter(2, [1200.0, 4800.0], btype="bandpass", fs=sr, output="sos")
    for onset, dur, gain_s, pan_s in ((1.64, 0.42, 0.11, 0.46), (7.62, 0.55, 0.13, 0.50), (9.08, 0.48, 0.12, 0.58)):
        ln = int(dur * sr)
        st = np.linspace(0.0, 1.0, ln, dtype=np.float32)
        sw = sosfilt(sos_swish, rng.normal(0.0, 1.0, ln)).astype(np.float32) * (np.maximum(np.sin(st * math.pi), 0.0) ** 1.4) * gain_s
        add_mono_event(onset, sw, pan=pan_s)

    # 6. Heavy Corrugated Metal Rooftop Landing Clang (`8.50s`) + Electric Cable Sparks (`8.20s`) + Chrome Latch (`9.66s`)
    tt_cl = np.arange(int(0.85 * sr), dtype=np.float32) / sr
    clang = (
        0.26 * np.sin(2.0 * math.pi * 68.0 * tt_cl) * np.exp(-tt_cl * 16.0)
        + 0.13 * np.sin(2.0 * math.pi * 194.0 * tt_cl) * np.exp(-tt_cl * 9.5)
        + 0.10 * np.sin(2.0 * math.pi * 289.0 * tt_cl) * np.exp(-tt_cl * 11.0)
        + 0.08 * np.sin(2.0 * math.pi * 431.0 * tt_cl) * np.exp(-tt_cl * 14.0)
        + 0.06 * np.sin(2.0 * math.pi * 698.0 * tt_cl) * np.exp(-tt_cl * 19.0)
    )
    add_mono_event(8.50, clang.astype(np.float32), pan=0.60)

    # Electric cable snap + crackle sparks (`8.20s – 8.95s`)
    tt_sp = np.arange(int(0.75 * sr), dtype=np.float32) / sr
    crackle = (
        (rng.random(len(tt_sp)) > 0.965).astype(np.float32)
        * rng.uniform(-1.0, 1.0, len(tt_sp)).astype(np.float32)
        * np.exp(-tt_sp * 3.2)
        * 0.24
    )
    add_mono_event(8.20, crackle, pan=0.68)

    # Brushed-chrome delivery case pneumatic latch pop (`9.66s`)
    tt_lt = np.arange(int(0.28 * sr), dtype=np.float32) / sr
    latch = (
        0.15 * np.sin(2.0 * math.pi * 1180.0 * tt_lt) * np.exp(-tt_lt * 65.0)
        + 0.10 * white_r[: len(tt_lt)] * np.exp(-tt_lt * 24.0)
    )
    add_mono_event(9.66, latch.astype(np.float32), pan=0.48)

    # 7. Security Quadcopter Drone Multi-Rotor Doppler Whir (`6.95s – 10.95s`)
    d_mask = np.clip((t - 6.95) / 0.45, 0.0, 1.0) * np.clip((11.05 - t) / 0.55, 0.0, 1.0)
    doppler_f = 138.0 + 18.0 * np.sin((t - 7.0) * 1.6)
    drone_ph = np.cumsum(2.0 * math.pi * doppler_f / sr)
    drone_sig = (
        0.055 * np.sin(drone_ph)
        + 0.042 * np.sin(drone_ph * 2.01)
        + 0.026 * np.sin(drone_ph * 3.03)
    ) * d_mask
    d_pan = 0.5 + 0.35 * np.sin((t - 7.0) * 1.5)
    fx_l += drone_sig * (1.0 - d_pan)
    fx_r += drone_sig * d_pan

    # 8. Deep Sub-Bass Heartbeat Pulses on Seed Reveal (`9.85s -> 14.416s`) + Ethereal Bioluminescent Pad
    sub_bass = np.zeros(n, dtype=np.float32)
    pulse_times = [9.95, 10.73, 11.18, 12.14, 13.10, 14.02]
    for idx_p, pt in enumerate(pulse_times):
        amp_p = 0.24 + 0.05 * idx_p
        for sub_off, sub_g in ((0.0, 1.0), (0.17, 0.72)):  # True double cardiac "lub-dub"!
            ln = int(0.42 * sr)
            tp = np.arange(ln, dtype=np.float32) / sr
            thump = amp_p * sub_g * np.sin(2.0 * math.pi * (44.0 - 12.0 * tp) * tp) * np.exp(-tp * 8.5)
            i0 = int(round((pt + sub_off) * sr))
            if i0 < n:
                m_ln = min(ln, n - i0)
                sub_bass[i0 : i0 + m_ln] += thump[:m_ln]

    glow_env = np.clip((t - 10.6) / 3.8, 0.0, 1.0) ** 1.3
    shimmer_l = 0.055 * glow_env * (np.sin(2.0 * math.pi * 261.63 * t) + 0.6 * np.sin(2.0 * math.pi * 392.00 * t))
    shimmer_r = 0.055 * glow_env * (np.sin(2.0 * math.pi * 329.63 * t) + 0.6 * np.sin(2.0 * math.pi * 493.88 * t))

    left = rain_l + city_l + fx_l + sub_bass + shimmer_l
    right = rain_r + city_r + fx_r + sub_bass + shimmer_r

    # DRAMATIC HARD AUDIO CUT RIGHT AT `14.416s` (Frame 346) — ALL RAIN & HUM VANISH INTO PURE BLACK!
    pre_fade = int(0.012 * sr)
    left[cut_idx - pre_fade : cut_idx] *= np.linspace(1.0, 0.0, pre_fade, dtype=np.float32)
    right[cut_idx - pre_fade : cut_idx] *= np.linspace(1.0, 0.0, pre_fade, dtype=np.float32)
    left[cut_idx:] = 0.0
    right[cut_idx:] = 0.0

    # 9. Single Pure Crystalline Chime Ringing Out Alone Over Black (`14.416s -> 15.00s`)!
    rem = n - cut_idx
    tc = np.arange(rem, dtype=np.float32) / sr
    chime = (
        0.26 * np.sin(2.0 * math.pi * 1046.50 * tc) * np.exp(-tc * 4.4)
        + 0.13 * np.sin(2.0 * math.pi * 2093.00 * tc + 0.3) * np.exp(-tc * 6.5)
        + 0.07 * np.sin(2.0 * math.pi * 3142.00 * tc) * np.exp(-tc * 9.0)
        + 0.09 * np.sin(2.0 * math.pi * 523.25 * tc) * np.exp(-tc * 3.6)
    )
    tail_out = np.ones(rem, dtype=np.float32)
    end_f = min(int(0.08 * sr), rem)
    tail_out[-end_f:] *= np.linspace(1.0, 0.0, end_f, dtype=np.float32)
    left[cut_idx:] += chime * tail_out
    right[cut_idx:] += np.roll(chime, min(88, rem // 4)) * tail_out * 0.96

    stereo = np.stack([np.tanh(left * 1.25), np.tanh(right * 1.25)], axis=-1)
    pcm16 = np.clip(stereo * 0.85 * 32767.0, -32767.0, 32767.0).astype(np.int16)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def render_tokyo_courier_mp4(
    *,
    output_path: Path,
    assets_dir: Path,
    width: int = 3840,
    height: int = 2160,
    fps: int = 24,
    duration_sec: float = 15.0,
) -> Path:
    """
    Renders the complete `15.00s` (`360` frames @ `24 fps`) 4K UHD (`3840x2160`)
    anamorphic cinema film with subtle chromatic aberration & temporal 4K grain.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(round(duration_sec * fps))
    renderer = TokyoCinemaRenderer(assets_dir)

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "tokyo_courier.wav"
        write_tokyo_courier_soundtrack(wav_path, duration_sec=duration_sec)

        # Native SIMD Lanczos 4K upscale + subtle lateral anamorphic chromatic aberration!
        vf_chain = (
            f"scale={width}:{height}:flags=lanczos,"
            "rgbashift=rh=-2:bh=2:rv=1:bv=-1:edge=smear"
        )

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{renderer.WORK_W}x{renderer.WORK_H}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-i", str(wav_path),
            "-vf", vf_chain,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "224k",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
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
                rgb = renderer.render_frame_1080p_anamorphic(f_idx, fps=fps)
                proc.stdin.write(rgb.tobytes())
        finally:
            proc.stdin.close()
            err = ""
            if proc.stderr is not None:
                err = proc.stderr.read().decode("utf-8", errors="replace")
                proc.stderr.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"FFmpeg 4K encode failed (rc={rc}): {err}")

    return output_path
