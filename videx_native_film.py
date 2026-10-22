"""
VIDEX Native Cinema Director — "Tokyo Seed Courier", 100% software-rendered
==========================================================================

There is no photography in this film. Every pixel of every frame is generated
by `videx_raytrace.py`, the VIDEXX signed-distance-field software renderer run
in this repository, and every sample of audio by the in-repo synthesizer in
`videx_tokyo_cinema.py`.

Shot list (15.00 s, 24 fps, 360 frames, 2.39:1 anamorphic):

  [0:00–0:03] EXT. RAIN-SLICKED TOKYO ALLEY — 35 mm low-angle dolly-in, wet
              asphalt reflections, steam plume, darting cat, neon blades,
              courier steps in and shakes water off the chrome case.
  [0:03–0:07] MEDIUM CLOSE-UP — 50 mm handheld micro-shake, anamorphic rack
              focus from lens droplets to the courier's eyes, smartwatch tap,
              emerald holographic map bloom + glitch, whispered "Wrong block.",
              visible cold breath.
  [0:07–0:11] WIDE TRACKING SHOT — 24 mm lateral rooftop chase with barrel
              distortion, whipping laundry, drone searchlight, gap leap,
              slide behind the AC unit, chrome case pops open on a pulsing seed.
  [0:11–0:15] EXTREME CLOSE-UP — 120 mm macro push-in on the seed, cardioid
              heartbeat, roots of light crawling across the glove, smile, hard
              cut to black on a single chime at 14.42 s.
"""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg
import numpy as np

import videx_raytrace as rt
from videx_tokyo_cinema import write_tokyo_courier_soundtrack

FPS = 24
DURATION = 15.0
TOTAL_FRAMES = int(round(FPS * DURATION))
ASPECT = 2.39

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def smoothstep(a: float, b: float, x: float) -> float:
    if b == a:
        return 0.0 if x < a else 1.0
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3.0 - 2.0 * t)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_io(t: float) -> float:
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def heartbeat(t: float, period: float = 0.80) -> float:
    u = (t % period) / period
    lub = math.exp(-((u - 0.09) ** 2) / 0.0022)
    dub = 0.68 * math.exp(-((u - 0.26) ** 2) / 0.0030)
    return min(1.0, lub + dub)


def shake(t: float, amp: float, seed: float = 0.0) -> tuple[float, float]:
    """Deterministic handheld micro-shake (two octaves of smooth wander)."""
    a = math.sin(t * 7.3 + seed) * 0.6 + math.sin(t * 13.1 + seed * 2.1) * 0.4
    b = math.sin(t * 9.7 + seed * 1.7) * 0.6 + math.sin(t * 15.3 + seed * 3.3) * 0.4
    return a * amp, b * amp


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def camera(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    width: int,
    height: int,
    hfov_deg: float,
    roll_deg: float = 0.0,
    jitter: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    ex, ey, ez = eye
    f = np.array([target[0] - ex, target[1] - ey, target[2] - ez], dtype=np.float64)
    n = np.linalg.norm(f)
    f = f / (n if n > 1e-9 else 1.0)
    up = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, up)
    r /= max(np.linalg.norm(r), 1e-9)
    u = np.cross(r, f)
    if roll_deg:
        ca, sa = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        r, u = r * ca + u * sa, u * ca - r * sa
    cam = np.zeros(15)
    cam[0:3] = ex, ey, ez
    cam[3:6] = f
    cam[6:9] = r
    cam[9:12] = u
    cam[12] = (width * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
    cam[13] = jitter[0]
    cam[14] = jitter[1]
    return cam


# ---------------------------------------------------------------------------
# The courier rig — 48 floats of pose, driven by one continuous function
# ---------------------------------------------------------------------------


def rig_standing(x0: float, z0: float, yaw: float = 0.0, detail: float = 1.0,
                 case_x: float | None = None, case_z: float | None = None,
                 lean: float = 0.0, crouch: float = 0.0, run: float = 0.0,
                 run_phase: float = 0.0, air: float = 0.0) -> np.ndarray:
    J = np.zeros(48, dtype=np.float64)
    hip_y = 0.94 - 0.30 * crouch
    chest_y = 1.36 - 0.42 * crouch + lean * 0.06
    head_y = 1.625 - 0.50 * crouch + lean * 0.10
    J[0:3] = x0, hip_y, z0
    J[3:6] = x0, chest_y, z0 + lean * 0.12
    J[6:9] = x0, chest_y + 0.16, z0 + lean * 0.14
    J[9:12] = x0, head_y, z0 + 0.012 + lean * 0.16
    J[12:15] = x0 - 0.10, hip_y - 0.44, z0 + 0.02
    J[15:18] = x0 - 0.11, 0.08, z0 + 0.02
    J[18:21] = x0 + 0.10, hip_y - 0.44, z0 - 0.02
    J[21:24] = x0 + 0.11, 0.08, z0 - 0.02
    J[24:27] = x0 - 0.245, chest_y + 0.065, z0
    J[27:30] = x0 - 0.315, chest_y - 0.20, z0 + 0.03
    J[30:33] = x0 + 0.245, chest_y + 0.065, z0
    J[33:36] = x0 + 0.325, chest_y - 0.20, z0 + 0.03
    J[36:39] = x0 - 0.300, hip_y + 0.02, z0 + 0.115
    J[39:42] = x0 + 0.310, hip_y, z0 + 0.095
    J[42] = yaw
    J[43] = 0.0
    J[44] = case_x if case_x is not None else x0 - 0.30
    J[45] = 0.86
    J[46] = case_z if case_z is not None else z0 + 0.17
    J[47] = detail

    # ---- run cycle: opposed arm/leg swing, torso counter-rotation ----
    if run > 0.01:
        sw = math.sin(run_phase)
        co = math.cos(run_phase)
        J[12] += -0.10 * sw * run * 0.5      # left knee
        J[13] += 0.16 * max(0.0, sw) * run
        J[15] += -0.22 * sw * run
        J[18] += 0.10 * sw * run * 0.5       # right knee
        J[19] += 0.16 * max(0.0, -sw) * run
        J[21] += 0.22 * sw * run
        # arms pump opposite the legs, elbows bent
        J[27] += -0.16 * sw * run
        J[28] += 0.10 * sw * run
        J[29] += -0.12 * abs(sw) * run
        J[33] += 0.16 * sw * run
        J[34] += 0.10 * -sw * run
        J[35] += 0.12 * abs(sw) * run
        J[4] += 0.012 * abs(co) * run
        J[10] += -0.010 * abs(co) * run
    if air > 0.01:
        # tuck both legs and swing the arms forward mid-leap
        J[13] += 0.30 * air
        J[16] += 0.30 * air
        J[19] += 0.34 * air
        J[22] += 0.34 * air
        J[29] += 0.10 * air
        J[35] += 0.10 * air
    return J


def rig_lit_detail(J: np.ndarray, detail: float) -> np.ndarray:
    J[47] = detail

    # ---- run cycle: opposed arm/leg swing, torso counter-rotation ----
    if run > 0.01:
        sw = math.sin(run_phase)
        co = math.cos(run_phase)
        J[12] += -0.10 * sw * run * 0.5      # left knee
        J[13] += 0.16 * max(0.0, sw) * run
        J[15] += -0.22 * sw * run
        J[18] += 0.10 * sw * run * 0.5       # right knee
        J[19] += 0.16 * max(0.0, -sw) * run
        J[21] += 0.22 * sw * run
        # arms pump opposite the legs, elbows bent
        J[27] += -0.16 * sw * run
        J[28] += 0.10 * sw * run
        J[29] += -0.12 * abs(sw) * run
        J[33] += 0.16 * sw * run
        J[34] += 0.10 * -sw * run
        J[35] += 0.12 * abs(sw) * run
        J[4] += 0.012 * abs(co) * run
        J[10] += -0.010 * abs(co) * run
    if air > 0.01:
        # tuck both legs and swing the arms forward mid-leap
        J[13] += 0.30 * air
        J[16] += 0.30 * air
        J[19] += 0.34 * air
        J[22] += 0.34 * air
        J[29] += 0.10 * air
        J[35] += 0.10 * air
    return J


# ---------------------------------------------------------------------------
# Shot 1 — rain-slicked Tokyo alley, 35 mm low-angle dolly-in
# ---------------------------------------------------------------------------

ALLEY_LIGHTS = (
    (-2.05, 4.15, 6.6, 0.10, 0.88, 1.00, 26.0),
    (2.05, 4.35, 5.2, 1.00, 0.18, 0.72, 24.0),
    (2.00, 1.35, 9.4, 1.00, 0.60, 0.28, 15.0),
    (-0.72, 2.72, 7.8, 1.00, 0.64, 0.30, 8.0),
    (0.00, 5.10, 16.0, 0.42, 0.60, 0.95, 13.0),
    (-2.05, 3.25, 11.2, 0.95, 0.52, 0.16, 19.0),
)


def shot1(t: float, width: int, height: int):
    """0.0 – 3.0 s. Low-angle dolly-in, 35 mm, shallow depth of field."""
    u = t / 3.0
    dolly = ease_io(u)
    eye = (0.0, 0.46 + 0.13 * dolly, 2.10 + 1.55 * dolly)
    tx, ty, tz = 0.30, 1.30 + 0.10 * dolly, 8.30
    hfov = lerp(52.0, 44.0, dolly)
    sx, sy = shake(t, 0.0016, 1.0)
    cam = camera(eye, (tx, ty, tz), width, height, hfov, roll_deg=0.6, jitter=(sx, sy))

    # courier steps into frame from screen-left and settles
    enter = smoothstep(0.15, 1.5, t)
    x0 = lerp(-1.05, 0.42, enter)
    z0 = lerp(9.2, 8.45, ease_io(u))
    yaw = lerp(-0.55, 0.18, smoothstep(0.6, 2.2, t))
    J = rig_standing(x0, z0, yaw=yaw, detail=0.0,
                     case_x=x0 - 0.30, case_z=z0 + 0.17)
    # looks up at the flickering sign at the end of the shot
    look = smoothstep(2.0, 3.0, t)
    J[43] = -0.34 * look

    cat_x = lerp(-2.4, 2.4, ease_io(min(1.0, (t - 0.35) / 2.3)))
    return cam, 0, J, 0.6, dict(cat=cat_x, case_open=0.0, steam=1.0, seed=0.0, holo=0.0,
                               rain=1.0, drone_x=0.0, drone_y=0.0)


# ---------------------------------------------------------------------------
# Shot 2 — medium close-up on the face, 50 mm, anamorphic rack focus
# ---------------------------------------------------------------------------


def shot2(t: float, width: int, height: int):
    """3.0 – 7.0 s. Handheld MCU: lens droplets rack to the eyes, watch, holo."""
    u = (t - 3.0) / 4.0
    # camera pushes in 4 cm and drifts a touch to the right
    eye = (0.09 - 0.03 * u, 1.600 + 0.012 * smoothstep(0.0, 1.0, u), 6.95 - 0.12 * u)
    target = (0.0, 1.600, 8.02)
    hfov = lerp(33.0, 28.0, ease_io(u))
    sx, sy = shake(t, 0.0032, 4.0)
    cam = camera(eye, target, width, height, hfov, roll_deg=-0.8, jitter=(sx, sy))

    J = rig_standing(0.0, 8.0, yaw=math.pi, detail=1.0)
    # warm ramen-shop rim light lives behind him in this shot

    # tiny head life: a slow nod, then the whispered line
    J[43] = 0.03 * math.sin(t * 1.6) - 0.05 * smoothstep(5.2, 5.9, t)
    J[10] += 0.004 * math.sin(t * 2.1)

    glow = 0.0
    if t > 4.15:
        glow = smoothstep(4.15, 4.7, t)
        if 5.18 < t < 5.30:
            glow *= 0.35  # the single hologram glitch
    return cam, 0, J, 0.6, dict(cat=-2.2, case_open=0.0, steam=1.0, seed=0.0,
                               holo=glow, rain=1.0, drone_x=0.0, drone_y=0.0)


# ---------------------------------------------------------------------------
# Shot 3 — rooftop chase, 24 mm lateral tracking
# ---------------------------------------------------------------------------


def shot3(t: float, width: int, height: int):
    """7.0 – 11.0 s. Drone-style lateral track with a leap across the gap."""
    u = (t - 7.0) / 4.0

    # run 7.0–9.3, leap 9.3–10.05, land + slide 10.05–11.0
    run = smoothstep(7.0, 9.25, t)
    x0 = lerp(-0.35, 0.55, run)
    z0 = lerp(2.2, 13.6, run)
    crouch = 0.28
    lean = 0.9
    airborne = 0.0
    if 9.25 <= t <= 10.10:
        p = (t - 9.25) / 0.85
        airborne = math.sin(p * math.pi)
        z0 = lerp(13.6, 19.6, p)
        crouch = 0.28 - 0.55 * math.sin(p * math.pi) + 0.9 * max(0.0, (p - 0.62)) * 1.6
        lean = lerp(0.9, 0.25, p)
    elif t > 10.10:
        p = min(1.0, (t - 10.10) / 0.9)
        z0 = lerp(19.6, 21.4, ease_io(p))
        crouch = lerp(0.95, 0.62, ease_io(p))
        lean = 0.25

    ground = 0.0 if z0 < 15.0 else (0.0 if t < 9.25 else -1.35)
    J = rig_standing(x0, z0, yaw=0.0, detail=0.0,
                     case_x=x0 + 0.02, case_z=z0 + 0.24,
                     crouch=crouch, lean=lean,
                     run=1.0 if (t < 9.25) else 0.0,
                     run_phase=t * 9.6, air=airborne)
    J[1] += ground + airborne * 0.95
    J[4] += ground + airborne * 0.95
    J[7] += ground + airborne * 0.95
    J[10] += ground + airborne * 0.95
    J[13] += ground + airborne * 0.95
    J[16] += ground + airborne * 0.95
    J[19] += ground + airborne * 0.95
    J[22] += ground + airborne * 0.95
    J[25] += ground + airborne * 0.95
    J[28] += ground + airborne * 0.95
    J[31] += ground + airborne * 0.95
    J[34] += ground + airborne * 0.95
    J[45] += ground + airborne * 0.95
    # legs tuck during the leap
    if airborne > 0.05:
        J[16] += 0.28 * airborne
        J[22] += 0.34 * airborne
    case_open = smoothstep(10.45, 10.85, t)
    # ---- camera: ride 3.1 m to his left, slightly behind, tracking his chest ----
    eye = (x0 - 2.85 + 0.10 * math.sin(u * 5.0), 1.42 + ground * 0.5 + airborne * 0.25,
           z0 - 3.90)
    target = (x0 + 0.05, 1.02 + ground * 0.85 + airborne * 0.55, z0 + 0.30)
    sx, sy = shake(t, 0.0055, 9.0)
    cam = camera(eye, target, width, height, 50.0,
                 roll_deg=-2.0 + 1.2 * u, jitter=(sx, sy))
    return cam, 1, J, 1.55, dict(cat=0.0, case_open=case_open, steam=0.0,
                               seed=1.0 * case_open, holo=0.0, rain=1.0,
                               drone_x=-2.6 + 3.4 * u, drone_y=2.35)


# ---------------------------------------------------------------------------
# Shot 4 — extreme macro on the seed, 120 mm, 120 fps -> 24 fps feel
# ---------------------------------------------------------------------------


def shot4(t: float, width: int, height: int):
    """11.0 – 15.0 s. Overcranked macro push-in, roots of light, cut to black."""
    u = (t - 11.0) / 4.0
    slow = u ** 1.55  # overcranked time
    eye = (0.070 - 0.030 * slow, 0.150 - 0.040 * slow, 0.345 - 0.095 * slow)
    target = (0.004, 0.020, 0.004)
    hfov = lerp(46.0, 35.0, ease_io(slow))
    sx, sy = shake(t * 0.45, 0.0013, 21.0)
    cam = camera(eye, target, width, height, hfov, roll_deg=1.0, jitter=(sx, sy))

    J = np.zeros(48, dtype=np.float64)
    J[44:47] = 0.0, 0.0, 0.0
    pulse = 0.45 + 0.75 * heartbeat(t, 0.80) * smoothstep(11.0, 11.6, t)
    pulse *= lerp(0.75, 1.35, slow)
    grow = smoothstep(11.15, 14.1, t) ** 1.25
    return cam, 2, J, 0.46, dict(cat=0.0, case_open=1.0, steam=0.0, seed=pulse,
                                 holo=grow, rain=0.0, drone_x=0.0, drone_y=0.0)


SHOTS = (shot1, shot2, shot3, shot4)
SHOT_END = (3.0, 7.0, 11.0, 15.0)

SEED_LIGHTS = (
    (0.00, 0.032, 0.00, 0.35, 0.72, 1.00, 9.0),    # the seed itself
    (0.075, 0.020, 0.060, 0.45, 0.70, 1.00, 2.6),  # bounce onto the fingers
    (-0.02, 0.26, -0.16, 0.62, 0.78, 1.00, 2.2),   # cold rim from the night sky
)
ROOF_LIGHTS = (
    (0.0, 7.0, 22.0, 0.48, 0.66, 1.00, 34.0),     # cold sky dome bounce
    (7.5, 2.4, 22.0, 1.00, 0.40, 0.30, 9.0),      # warm city glow, screen-right
    (-7.0, 1.8, 18.0, 0.22, 0.52, 1.00, 14.0),    # cyan neon, screen-left
    (0.0, 2.2, 6.0, 0.30, 0.55, 1.00, 8.0),       # cool ambience near the camera
    (2.2, 3.4, 11.0, 1.00, 0.20, 0.16, 11.0),     # drone searchlight core
    (-2.2, 0.8, 12.0, 0.55, 0.78, 1.00, 6.0),     # cool practical by the AC unit
)
ALLEY_SHOT_LIGHTS = {
    0: ALLEY_LIGHTS,
    1: (
        (-1.30, 2.10, 7.70, 0.42, 0.82, 1.00, 6.5),
        (1.35, 1.98, 7.55, 1.00, 0.52, 0.58, 3.4),
        (0.00, 1.94, 7.10, 1.00, 0.90, 0.82, 1.05),
        (-0.20, 1.55, 8.30, 0.20, 0.90, 0.62, 2.0),
        (0.30, 1.40, 8.60, 1.00, 0.62, 0.32, 1.6),
    ),
}


def build_params(shot_idx: int, extra: dict, t: float) -> np.ndarray:
    lights = ALLEY_SHOT_LIGHTS.get(shot_idx) if shot_idx in (0, 1) else (
        ROOF_LIGHTS if shot_idx == 2 else SEED_LIGHTS)
    n = len(lights)
    p = np.zeros(10 + n * 8, dtype=np.float64)
    p[0] = extra["cat"]
    p[1] = extra["case_open"]
    p[2] = 1.0
    p[3] = extra["steam"]
    p[4] = extra["rain"]
    p[5] = extra["seed"]
    p[6] = extra["holo"]
    p[7] = extra["drone_x"]
    p[8] = extra["drone_y"]
    p[9] = n
    for i, L in enumerate(lights):
        b = 10 + i * 8
        p[b:b + 7] = L
        p[b + 7] = 0.4
    if shot_idx == 3:
        # the seed's own radiance drives every practical in the macro shot
        k = 0.35 + 0.85 * extra["seed"]
        for i in range(n):
            b = 10 + i * 8
            p[b + 6] *= k if i == 0 else (0.45 + 0.55 * extra["seed"])
    return p


# ---------------------------------------------------------------------------
# Post-processing — all native NumPy
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Native atmosphere overlays (rain, lens droplets, hologram, breath)
# ---------------------------------------------------------------------------


def _rain_overlay(rgb: np.ndarray, t: float, frame_idx: int,
                  density: float = 1.0, depth: np.ndarray | None = None,
                  slant: float = 0.22, gain: float = 0.55) -> np.ndarray:
    """Procedural falling rain: thin wind-slanted streaks with depth falloff."""
    import cv2

    h, w = rgb.shape[:2]
    rng = np.random.default_rng(4242)
    n = int(240 * density)
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n)
    spd = rng.uniform(1.2, 2.6, n)
    ln = rng.uniform(0.012, 0.032, n) * h
    bright = rng.uniform(0.05, 0.22, n)

    layer = np.zeros((h, w), dtype=np.float32)
    fall = (t * 1.9) % 1.0
    for i in range(n):
        x = (xs[i] + t * 26.0 * (0.5 + spd[i] * 0.3)) % (w + 40) - 20
        y = ((ys[i] / h) + fall * spd[i]) % 1.0 * h
        x0 = int(x - ln[i] * slant)
        y0 = int(y)
        x1 = int(x + ln[i] * slant)
        y1 = int(y + ln[i])
        if 0 <= y0 < h and 0 <= y1 < h and 0 <= x0 < w and 0 <= x1 < w:
            cv2.line(layer, (x0, y0), (x1, y1), float(bright[i]), 1, cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (3, 3), 0.6)
    if depth is not None:
        far = np.clip((depth - 1.5) / 9.0, 0.0, 1.0)
        layer = layer * (0.18 + 0.82 * far)
    tint = np.array([0.58, 0.74, 1.00], dtype=np.float32)
    return rgb + layer[..., None] * tint * gain


def _searchlight(rgb: np.ndarray, t: float, origin: tuple[float, float],
                 target: tuple[float, float], width_px: float,
                 strength: float) -> np.ndarray:
    """Volumetric red drone searchlight cone, drawn in screen space."""
    import cv2

    if strength <= 0.01:
        return rgb
    h, w = rgb.shape[:2]
    ox, oy = origin[0] * w, origin[1] * h
    tx, ty = target[0] * w, target[1] * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # distance from the cone axis
    ax, ay = tx - ox, ty - oy
    L2 = max(ax * ax + ay * ay, 1.0)
    proj = ((xx - ox) * ax + (yy - oy) * ay) / L2
    proj_c = np.clip(proj, 0.0, 1.0)
    px = ox + ax * proj_c
    py = oy + ay * proj_c
    dist = np.hypot(xx - px, yy - py)
    cone = np.exp(-(dist / (width_px * (0.25 + proj_c))) ** 2)
    cone *= (proj > 0.02) * (proj < 1.15)
    cone *= np.clip(1.0 - proj * 1.15, 0.0, 1.0) ** 0.8
    # dust motes drifting through the beam
    rng = np.random.default_rng(77)
    motes = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 2.2)
    cone *= 0.75 + 0.45 * np.abs(motes)
    tint = np.array([1.00, 0.16, 0.10], dtype=np.float32)
    return rgb + cone[..., None] * tint * strength


def _lens_droplets(rgb: np.ndarray, t: float, strength: float,
                   seed: int = 7) -> np.ndarray:
    """
    Water beaded on the front element: each droplet refracts and magnifies the
    image behind it, with a bright specular kick at the top.
    """
    import cv2

    h, w = rgb.shape[:2]
    rng = np.random.default_rng(seed)
    n = 26
    cx = rng.uniform(0, w, n)
    cy = rng.uniform(0, h, n)
    rad = rng.uniform(0.012, 0.045, n) * min(w, h)
    out = rgb.copy()
    for i in range(n):
        r = rad[i]
        x0 = int(max(0, cx[i] - r))
        x1 = int(min(w, cx[i] + r + 1))
        y0 = int(max(0, cy[i] - r))
        y1 = int(min(h, cy[i] + r + 1))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = (xx - cx[i]) / r
        dy = (yy - cy[i]) / r
        d2 = dx * dx + dy * dy
        inside = d2 < 1.0
        if not np.any(inside):
            continue
        # refraction: sample the background magnified and flipped near the rim
        k = 0.42
        sx = np.clip(cx[i] + dx * r * k, 0, w - 1).astype(np.float32)
        sy = np.clip(cy[i] + dy * r * k, 0, h - 1).astype(np.float32)
        refr = cv2.remap(rgb, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        edge = np.clip((d2 - 0.55) / 0.45, 0.0, 1.0)
        shade = (1.0 - edge * 0.55)[..., None]
        patch = refr * shade + edge[..., None] * np.array([0.06, 0.09, 0.12])
        # specular highlight, upper-left of each bead
        spec = np.exp(-(((dx + 0.42) ** 2 + (dy + 0.48) ** 2) * 26.0))[..., None]
        patch = patch + spec * np.array([0.55, 0.62, 0.70])
        m = inside[..., None]
        out[y0:y1, x0:x1] = np.where(m, patch, out[y0:y1, x0:x1])
    return out


def _hologram(rgb: np.ndarray, t: float, strength: float, anchor: tuple[float, float],
              size: float = 0.34) -> np.ndarray:
    """
    Emerald holographic street map blooming over the courier's wrist:
    perspective grid, blocks, scan-lines, glow and a single horizontal glitch.
    """
    import cv2

    if strength <= 0.01:
        return rgb
    h, w = rgb.shape[:2]
    ax = int(anchor[0] * w)
    ay = int(anchor[1] * h)
    gw = int(size * w)
    gh = int(size * h * 1.05)
    if gw < 8 or gh < 8:
        return rgb

    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    u = xx / gw * 2.0 - 1.0
    v = yy / gh * 2.0 - 1.0
    # fake 3D tilt: map v to depth
    persp = 1.0 / (1.25 + 0.55 * (v + 1.0))
    gu = u * persp * 3.2
    gv = (v - 0.15) * persp * 5.0

    grid = np.zeros((gh, gw), dtype=np.float32)
    fu = np.abs(gu - np.round(gu))
    fv = np.abs(gv - np.round(gv))
    grid += np.clip(1.0 - fu * 26.0, 0.0, 1.0)
    grid += np.clip(1.0 - fv * 26.0, 0.0, 1.0) * 0.8
    # a few raised blocks
    blk = np.zeros((gh, gw), dtype=np.float32)
    rng = np.random.default_rng(11)
    for _ in range(14):
        bu = rng.uniform(-1.0, 1.0)
        bv = rng.uniform(-0.9, 0.9)
        bw = rng.uniform(0.06, 0.16)
        bh = rng.uniform(0.05, 0.13)
        m = (np.abs(gu - bu * 3.2) < bw * 3.2) & (np.abs(gv - bv * 5.0) < bh * 5.0)
        blk[m] = np.maximum(blk[m], 0.55)
    scan = 0.55 + 0.45 * np.sin((yy + t * 380.0) * 0.55)
    fade = np.clip(1.0 - (u * u + (v * 1.05) ** 2) * 0.55, 0.0, 1.0)
    alpha = np.clip((grid * 0.9 + blk * 0.5) * scan * fade, 0.0, 1.4) * strength

    tint = np.array([0.18, 1.00, 0.62], dtype=np.float32)
    emis = alpha[..., None] * tint * 1.25

    # single glitch: shear a horizontal band sideways
    if 5.18 < t < 5.30:
        band = (yy > gh * 0.42) & (yy < gh * 0.52)
        emis[band] = np.roll(emis[band], int(gw * 0.07), axis=1)
        emis[..., 0] *= 1.35

    canvas = np.zeros((h, w, 3), dtype=np.float32)
    y0 = max(0, ay - gh // 2)
    x0 = max(0, ax - gw // 2)
    y1 = min(h, y0 + gh)
    x1 = min(w, x0 + gw)
    if y1 <= y0 or x1 <= x0:
        return rgb
    canvas[y0:y1, x0:x1] = emis[: y1 - y0, : x1 - x0]
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.4)
    glow = cv2.GaussianBlur(canvas, (0, 0), 9.0)
    return rgb + canvas + glow * 1.3


def _breath(rgb: np.ndarray, t: float, strength: float, origin: tuple[float, float]) -> np.ndarray:
    """Cold visible breath at the whispered line."""
    import cv2

    if strength <= 0.01:
        return rgb
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ox, oy = origin[0] * w, origin[1] * h
    dx = (xx - ox) / (w * 0.10)
    dy = (yy - oy) / (h * 0.16)
    cloud = np.exp(-(dx * dx + dy * dy))
    rng = np.random.default_rng(31)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 12.0)
    cloud = np.clip(cloud * (0.55 + 0.55 * noise), 0.0, 1.0) * strength
    return rgb + cloud[..., None] * np.array([0.16, 0.19, 0.22], dtype=np.float32)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    import cv2

    return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)


def bloom(rgb: np.ndarray, strength: float = 0.32) -> np.ndarray:
    import cv2

    small = cv2.resize(rgb, (max(8, rgb.shape[1] // 4), max(8, rgb.shape[0] // 4)),
                       interpolation=cv2.INTER_AREA)
    br = np.clip(small - 0.85, 0.0, None)
    wide = _blur(br, 6.0)
    tight = _blur(br, 1.8)
    glow = (wide * 0.7 + tight * 0.5)
    up = cv2.resize(glow, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return rgb + up * strength


def anamorphic_streak(rgb: np.ndarray, strength: float = 0.30) -> np.ndarray:
    """Horizontal blue lens flare off the practicals — real anamorphic habit."""
    import cv2

    small = cv2.resize(rgb, (max(8, rgb.shape[1] // 3), max(8, rgb.shape[0] // 3)),
                       interpolation=cv2.INTER_AREA)
    hi = np.clip(small - 1.6, 0.0, None)
    k = max(5, (small.shape[1] // 26) * 2 + 1)
    streak = cv2.blur(hi, (k, 1))
    streak = cv2.blur(streak, (k, 1))
    up = cv2.resize(streak, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    tint = np.array([0.25, 0.55, 1.00], dtype=np.float32)
    return rgb + up * tint * strength


def depth_of_field(rgb: np.ndarray, depth: np.ndarray, focus: float, strength: float,
                   max_blur: float = 9.0) -> np.ndarray:
    """CoC-driven blend of two blur levels using the renderer's depth buffer."""
    import cv2

    coc = np.clip(np.abs(depth - focus) / max(focus, 1e-3), 0.0, 1.0)
    coc = coc ** 0.8 * strength
    mid = _blur(rgb, 2.2)
    far = _blur(rgb, max_blur)
    w = np.clip(coc, 0.0, 1.0)[..., None]
    w2 = np.clip(coc - 0.45, 0.0, 1.0)[..., None] * 1.4
    out = rgb * (1.0 - w) + mid * w
    out = out * (1.0 - w2) + far * w2
    return out


def chromatic_aberration(rgb: np.ndarray, amount: float = 1.2) -> np.ndarray:
    """Lateral CA: red scale up, blue scale down about the frame centre."""
    import cv2

    h, w = rgb.shape[:2]
    out = np.empty_like(rgb)
    out[..., 1] = rgb[..., 1]
    for ch, s in ((0, 1.0 + amount / 1000.0), (2, 1.0 - amount / 1000.0)):
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0.0, s)
        out[..., ch] = cv2.warpAffine(rgb[..., ch], M, (w, h),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
    return out


def barrel_warp(rgb: np.ndarray, k: float) -> np.ndarray:
    import cv2

    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - w * 0.5) / (w * 0.5)
    ny = (yy - h * 0.5) / (h * 0.5)
    r2 = nx * nx + ny * ny
    f = 1.0 + k * r2
    mapx = (nx * f * w * 0.5 + w * 0.5).astype(np.float32)
    mapy = (ny * f * h * 0.5 + h * 0.5).astype(np.float32)
    return cv2.remap(rgb, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def tonemap_to_srgb(rgb: np.ndarray, exposure: float, grade: float = 1.0) -> np.ndarray:
    x = np.clip(rgb * exposure, 0.0, None)
    # ACES-ish filmic curve
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    y = np.clip(y, 0.0, 1.0)
    # teal shadows / amber highlights split tone
    lum = 0.2126 * y[..., 0] + 0.7152 * y[..., 1] + 0.0722 * y[..., 2]
    sh = (1.0 - lum)[..., None]
    hi = lum[..., None]
    y = y + sh * np.array([-0.030, 0.010, 0.045], dtype=np.float32) * grade
    y = y + hi * np.array([0.040, 0.012, -0.028], dtype=np.float32) * grade
    y = np.clip(y, 0.0, 1.0)
    return y ** (1.0 / 2.2)


def add_grain(rgb: np.ndarray, frame_idx: int, amount: float = 0.016) -> np.ndarray:
    rng = np.random.default_rng(1000 + frame_idx)
    h, w = rgb.shape[:2]
    n = rng.normal(0.0, 1.0, (h, w, 1)).astype(np.float32)
    n = np.repeat(n, 3, axis=2)
    # slightly coarser chroma grain
    n2 = rng.normal(0.0, 1.0, (h, w, 3)).astype(np.float32) * 0.4
    lum = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    wgt = (1.0 - np.clip(lum, 0.0, 1.0) * 0.55)[..., None]
    return np.clip(rgb + (n + n2) * amount * wgt, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------


def render_film_frame(frame_idx: int, width: int, height: int) -> np.ndarray:
    t = frame_idx / FPS

    # hard cut to black at 14.42 s on the single chime
    if t >= 14.416:
        return np.zeros((height, width, 3), dtype=np.uint8)

    shot_idx = 0
    for i, end in enumerate(SHOT_END):
        if t < end:
            shot_idx = i
            break
    fn = SHOTS[shot_idx]
    cam, stage, J, exposure, extra = fn(t, width, height)
    params = build_params(shot_idx, extra, t)

    rgb, depth = rt.render_frame(width, height, cam, stage, J, t, params)

    # --- lens & film emulation -------------------------------------------
    if shot_idx == 2:
        rgb = barrel_warp(rgb, 0.055)
        focus = float(np.median(depth))
        rgb = depth_of_field(rgb, depth, focus, 0.55, max_blur=7.0)
    elif shot_idx == 1:
        # rack focus: foreground lens droplets -> the courier's eyes
        focus = float(np.median(depth))
        if t < 4.3:
            focus = lerp(0.30, focus, smoothstep(3.30, 4.35, t))
        rgb = depth_of_field(rgb, depth, focus, 0.85, max_blur=10.0)
    elif shot_idx == 3:
        rgb = depth_of_field(rgb, depth, float(np.median(depth)), 0.80, max_blur=9.0)
    else:
        focus = float(np.median(depth[depth < 40.0])) if np.any(depth < 40.0) else 8.0
        rgb = depth_of_field(rgb, depth, focus, 0.55, max_blur=7.0)

    # ---- atmosphere, keyed to the shot ----
    if shot_idx == 0:
        rgb = _rain_overlay(rgb, t, frame_idx, density=1.0, depth=depth)
    elif shot_idx == 1:
        # foreground lens beading that racks out of focus early in the shot
        lens_amt = 1.0 - smoothstep(3.55, 4.45, t)
        if lens_amt > 0.02:
            rgb = _lens_droplets(rgb, t, lens_amt)
        rgb = _rain_overlay(rgb, t, frame_idx, density=0.55, depth=depth)
        holo = 0.0
        if t > 4.15:
            holo = smoothstep(4.15, 4.62, t)
            if 5.18 < t < 5.30:
                holo *= 0.4
        if holo > 0.01:
            rgb = _hologram(rgb, t, holo, anchor=(0.700, 0.880), size=0.26)
        breath = smoothstep(5.12, 5.38, t) * (1.0 - smoothstep(5.95, 6.55, t))
        if breath > 0.01:
            rgb = _breath(rgb, t, breath * 0.55, origin=(0.495, 0.665))
    elif shot_idx == 2:
        rgb = _rain_overlay(rgb, t, frame_idx, density=0.40, depth=depth,
                            slant=0.34, gain=0.42)
        beam = smoothstep(7.0, 7.6, t) * (1.0 - smoothstep(10.5, 11.0, t))
        if beam > 0.01:
            rgb = _searchlight(rgb, t, origin=(0.72, 0.02),
                               target=(0.42, 0.55), width_px=30.0,
                               strength=0.055 * beam)

    rgb = bloom(rgb, 0.30 if shot_idx != 3 else 0.26)
    rgb = anamorphic_streak(rgb, 0.22 if shot_idx != 3 else 0.30)
    srgb = tonemap_to_srgb(rgb, exposure, grade=1.0 if shot_idx != 3 else 0.7)
    srgb = chromatic_aberration(srgb, 1.4 if shot_idx != 0 else 1.0)
    srgb = add_grain(srgb, frame_idx, 0.014 if shot_idx != 3 else 0.010)

    # 2.39:1 letterbox is baked into the render aspect already
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_native_film(
    output_path: Path,
    *,
    width: int = 1280,
    height: int = 536,
    fps: int = FPS,
    duration: float = DURATION,
    progress: bool = True,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = int(round(duration * fps))
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "tokyo_native.wav"
        write_tokyo_courier_soundtrack(wav_path, duration_sec=duration)

        ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
            "-r", str(fps), "-i", "-",
            "-i", str(wav_path),
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "224k",
            "-shortest", "-movflags", "+faststart",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        try:
            for f_idx in range(total):
                proc.stdin.write(render_film_frame(f_idx, width, height).tobytes())
                if progress and f_idx % 12 == 0:
                    print(f"  frame {f_idx + 1}/{total}  ({f_idx / fps:5.2f}s)", flush=True)
        finally:
            proc.stdin.close()
            err = ""
            if proc.stderr is not None:
                err = proc.stderr.read().decode("utf-8", errors="replace")
                proc.stderr.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"FFmpeg encode failed (rc={rc}): {err}")
    return output_path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render the native VIDEX Tokyo film.")
    ap.add_argument("--out", default="output/tokyo_seed_courier.mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=536)
    ap.add_argument("--preview", type=int, default=0,
                    help="render only N evenly spaced stills to /tmp instead")
    ap.add_argument("--frames", type=int, nargs="*", default=None,
                    help="explicit frame indices for --preview")
    args = ap.parse_args()

    if args.preview:
        import cv2

        for i in range(args.preview):
            fi = int(i * (TOTAL_FRAMES - 1) / max(1, args.preview - 1))
            if args.frames:
                fi = args.frames[i] if i < len(args.frames) else fi
            frame = render_film_frame(fi, args.width, args.height)
            path = f"/tmp/native_{i:02d}_{fi:03d}.png"
            cv2.imwrite(path, frame[..., ::-1])
            print(path, flush=True)
    else:
        out = render_native_film(Path(args.out), width=args.width, height=args.height)
        print(f"wrote {out}")
