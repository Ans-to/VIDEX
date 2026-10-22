"""
VIDEX Native Software Renderer — pure-NumPy/Numba signed-distance-field ray marcher
==================================================================================

Every single pixel of every frame of "Tokyo Seed Courier" is *computed* here.
There are no photographs, no downloaded plates, no external image assets and no
image-generation services anywhere in this pipeline: the set, the buildings, the
courier's body, face and glove, the cat, the drone, the seed, the rain and the
lighting are all analytic signed-distance-field geometry shaded by our own
physically-motivated integrator.

Rendering model
---------------
  * SDF ray marching with adaptive epsilon and a 96-step hard cap.
  * Tetrahedral finite-difference normals.
  * Analytic key / fill / rim / practical lighting with per-light soft shadows
    and ambient occlusion.
  * Wet-surface environment reflections (neon sign lobes + sky dome) so puddles
    and chrome carry the alley's cyan & magenta.
  * Single-scattering volumetric fog & steam integrated in closed form per ray.
  * Emission materials for neon tubes, hologram scan-lines, the blue seed and
    the bioluminescent roots of light.
  * Linear-depth buffer output so the film pipeline can rack focus in post.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------------
# Hashes & value noise
# ---------------------------------------------------------------------------


@njit(inline="always", fastmath=True)
def _hash21(x: float, y: float, seed: float) -> float:
    v = math.sin(x * 127.1 + y * 311.7 + seed * 74.7) * 43758.5453
    return v - math.floor(v)


@njit(inline="always", fastmath=True)
def smoothstep_local(a: float, b: float, x: float) -> float:
    if b <= a:
        return 0.0 if x < a else 1.0
    t = (x - a) / (b - a)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return t * t * (3.0 - 2.0 * t)


@njit(inline="always", fastmath=True)
def _mod(a: float, b: float) -> float:
    return a - math.floor(a / b) * b


@njit(inline="always", fastmath=True)
def _fract(a: float) -> float:
    return a - math.floor(a)


@njit(inline="always", fastmath=True)
def _vnoise3(x: float, y: float, z: float, seed: float) -> float:
    ix = math.floor(x)
    iy = math.floor(y)
    iz = math.floor(z)
    fx = x - ix
    fy = y - iy
    fz = z - iz
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    uz = fz * fz * (3.0 - 2.0 * fz)

    c000 = _hash21(ix + iz * 17.0, iy - iz * 9.0, seed)
    c100 = _hash21(ix + 1.0 + iz * 17.0, iy - iz * 9.0, seed)
    c010 = _hash21(ix + iz * 17.0, iy + 1.0 - iz * 9.0, seed)
    c110 = _hash21(ix + 1.0 + iz * 17.0, iy + 1.0 - iz * 9.0, seed)
    c001 = _hash21(ix + (iz + 1.0) * 17.0, iy - (iz + 1.0) * 9.0, seed)
    c101 = _hash21(ix + 1.0 + (iz + 1.0) * 17.0, iy - (iz + 1.0) * 9.0, seed)
    c011 = _hash21(ix + (iz + 1.0) * 17.0, iy + 1.0 - (iz + 1.0) * 9.0, seed)
    c111 = _hash21(ix + 1.0 + (iz + 1.0) * 17.0, iy + 1.0 - (iz + 1.0) * 9.0, seed)

    x00 = c000 + (c100 - c000) * ux
    x10 = c010 + (c110 - c010) * ux
    x01 = c001 + (c101 - c001) * ux
    x11 = c011 + (c111 - c011) * ux
    y0 = x00 + (x10 - x00) * uy
    y1 = x01 + (x11 - x01) * uy
    return y0 + (y1 - y0) * uz


@njit(inline="always", fastmath=True)
def _fbm3(x: float, y: float, z: float, seed: float, octaves: int) -> float:
    amp = 0.5
    tot = 0.0
    norm = 0.0
    f = 1.0
    for i in range(octaves):
        tot += amp * _vnoise3(x * f, y * f, z * f, seed + i * 13.0)
        norm += amp
        amp *= 0.5
        f *= 2.03
    return tot / max(norm, 1e-6)


# ---------------------------------------------------------------------------
# SDF primitives (scalar triples — zero allocation in the inner loop)
# ---------------------------------------------------------------------------


@njit(inline="always", fastmath=True)
def sd_sphere(px, py, pz, cx, cy, cz, r):
    dx = px - cx
    dy = py - cy
    dz = pz - cz
    return math.sqrt(dx * dx + dy * dy + dz * dz) - r


@njit(inline="always", fastmath=True)
def sd_capsule(px, py, pz, ax, ay, az, bx, by, bz, r):
    pax = px - ax
    pay = py - ay
    paz = pz - az
    bax = bx - ax
    bay = by - ay
    baz = bz - az
    denom = bax * bax + bay * bay + baz * baz
    h = 0.0 if denom < 1e-12 else (pax * bax + pay * bay + paz * baz) / denom
    h = 0.0 if h < 0.0 else (1.0 if h > 1.0 else h)
    dx = pax - bax * h
    dy = pay - bay * h
    dz = paz - baz * h
    return math.sqrt(dx * dx + dy * dy + dz * dz) - r


@njit(inline="always", fastmath=True)
def sd_ellipsoid(px, py, pz, cx, cy, cz, rx, ry, rz):
    dx = (px - cx) / rx
    dy = (py - cy) / ry
    dz = (pz - cz) / rz
    k0 = math.sqrt(dx * dx + dy * dy + dz * dz)
    ex = dx / rx
    ey = dy / ry
    ez = dz / rz
    k1 = math.sqrt(ex * ex + ey * ey + ez * ez)
    if k1 < 1e-9:
        return -min(rx, min(ry, rz))
    return k0 * (k0 - 1.0) / k1


@njit(inline="always", fastmath=True)
def sd_rbox(px, py, pz, cx, cy, cz, bx, by, bz, r):
    qx = abs(px - cx) - bx
    qy = abs(py - cy) - by
    qz = abs(pz - cz) - bz
    mx = 0.0 if qx < 0.0 else qx
    my = 0.0 if qy < 0.0 else qy
    mz = 0.0 if qz < 0.0 else qz
    inner = qx if qx > qy else qy
    inner = inner if inner > qz else qz
    inner = 0.0 if inner < 0.0 else inner
    return math.sqrt(mx * mx + my * my + mz * mz) + inner - r


@njit(inline="always", fastmath=True)
def sd_round_cone(px, py, pz, ax, ay, az, bx, by, bz, r1, r2):
    """Capsule with linearly varying radius — coats, flares, tapered limbs."""
    bax = bx - ax
    bay = by - ay
    baz = bz - az
    l2 = bax * bax + bay * bay + baz * baz
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / max(l2, 1e-9)
    pax = px - ax
    pay = py - ay
    paz = pz - az
    y = pax * bax + pay * bay + paz * baz
    z = y - l2
    x2 = (pax * l2 - bax * y) ** 2 + (pay * l2 - bay * y) ** 2 + (paz * l2 - baz * y) ** 2
    y2 = y * y * l2
    z2 = z * z * l2
    k = math.copysign(1.0, rr) * rr * rr * x2
    if math.copysign(1.0, z) * a2 * z2 > k:
        return math.sqrt(x2 + z2) * il2 - r2
    if math.copysign(1.0, y) * a2 * y2 < k:
        return math.sqrt(x2 + y2) * il2 - r1
    return (math.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1


@njit(inline="always", fastmath=True)
def sd_plane_y(px, py, pz, h):
    return py - h


@njit(inline="always", fastmath=True)
def sd_plane_x(px, py, pz, x0):
    return px - x0


@njit(inline="always", fastmath=True)
def op_union(a, b):
    return a if a < b else b


@njit(inline="always", fastmath=True)
def op_sub(a, b):
    return -a if -a > b else b


@njit(inline="always", fastmath=True)
def op_smin(a, b, k):
    h = k - abs(a - b)
    if h < 1e-6:
        return a if a < b else b
    h = h / k
    return (a if a < b else b) - h * h * k * 0.25


@njit(inline="always", fastmath=True)
def op_smax(a, b, k):
    return -op_smin(-a, -b, k)


# ---------------------------------------------------------------------------
# Material identifiers
# ---------------------------------------------------------------------------

M_ASPHALT = 0
M_CONCRETE = 1
M_METAL = 2
M_CHROME = 3
M_NEON = 4
M_SKIN = 5
M_RAINCOAT = 6
M_FABRIC = 7
M_HOLO = 8
M_SEED = 9
M_LEATHER = 10
M_WINDOW = 11
M_WATER = 12
M_HAIR = 13
M_FUR = 14
M_ROOF = 15
M_EYE = 16
M_LIP = 17
M_GLASS = 18
M_STEEL = 19
M_CLOUD = 20
M_FACADE = 21
M_MOUTH = 24
M_ROOT = 25
M_FAR = 26
M_LANTERN = 22
M_VENDING = 23


# ---------------------------------------------------------------------------
# The courier: a fully procedural humanoid built from a 24-joint rig.
#
# joints layout (world space, metres):
#   0 pelvis   1 chest    2 neck     3 head
#   4-6  L shoulder / elbow / wrist
#   7-9  R shoulder / elbow / wrist
#   10-12 L hip / knee / ankle
#   13-15 R hip / knee / ankle
#   16 head-up hint   17 case centre  18-19 case extents
# ---------------------------------------------------------------------------


@njit(inline="always", fastmath=True)
def _body_sdf(px, py, pz, J):
    d = 1e9

    # legs
    d = op_smin(d, sd_capsule(px, py, pz, J[10], J[11], J[12], J[13], J[14], J[15], 0.082), 0.06)
    d = op_smin(d, sd_capsule(px, py, pz, J[13], J[14], J[15], J[16], J[17], J[18], 0.062), 0.05)
    d = op_smin(d, sd_capsule(px, py, pz, J[19], J[20], J[21], J[22], J[23], J[24], 0.082), 0.06)
    d = op_smin(d, sd_capsule(px, py, pz, J[22], J[23], J[24], J[25], J[26], J[27], 0.062), 0.05)
    # boots
    d = op_smin(d, sd_rbox(px, py, pz, J[16], J[17] - 0.012, J[18] + 0.045, 0.05, 0.038, 0.095, 0.022), 0.03)
    d = op_smin(d, sd_rbox(px, py, pz, J[25], J[26] - 0.012, J[27] + 0.045, 0.05, 0.038, 0.095, 0.022), 0.03)

    # torso
    d = op_smin(d, sd_capsule(px, py, pz, J[0], J[1], J[2], J[3], J[4], J[5], 0.150), 0.09)
    # chest volume
    d = op_smin(d, sd_ellipsoid(px, py, pz, (J[3] + J[6]) * 0.5, (J[4] + J[7]) * 0.5,
                                (J[5] + J[8]) * 0.5, 0.185, 0.150, 0.130), 0.07)
    # neck + head
    d = op_smin(d, sd_capsule(px, py, pz, J[6], J[7], J[8], J[9], J[10], J[11], 0.055), 0.04)
    return d


@njit(inline="always", fastmath=True)
def _arms_sdf(px, py, pz, J):
    d = 1e9
    d = op_smin(d, sd_capsule(px, py, pz, J[12], J[13], J[14], J[15], J[16], J[17], 0.058), 0.05)
    d = op_smin(d, sd_capsule(px, py, pz, J[15], J[16], J[17], J[18], J[19], J[20], 0.047), 0.04)
    d = op_smin(d, sd_capsule(px, py, pz, J[21], J[22], J[23], J[24], J[25], J[26], 0.058), 0.05)
    d = op_smin(d, sd_capsule(px, py, pz, J[24], J[25], J[26], J[27], J[28], J[29], 0.047), 0.04)
    # gloves at wrists
    d = op_smin(d, sd_ellipsoid(px, py, pz, J[18], J[19], J[20], 0.062, 0.050, 0.075), 0.03)
    d = op_smin(d, sd_ellipsoid(px, py, pz, J[27], J[28], J[29], 0.062, 0.050, 0.075), 0.03)
    return d


@njit(inline="always", fastmath=True)
def _head_sdf_local(lx, ly, lz):
    """Head, local space: +z forward (face), +y up, origin at the skull centre."""
    # cranium: widest just above the ears
    d = sd_ellipsoid(lx, ly, lz, 0.0, 0.020, -0.008, 0.0745, 0.0940, 0.0900)
    d = op_smax(d, -(lz + 0.0760), 0.05)
    # forehead
    d = op_smin(d, sd_ellipsoid(lx, ly, lz, 0.0, 0.0360, 0.0480, 0.0600, 0.0480, 0.0420), 0.045)
    # mid-face mass (cheeks + upper jaw)
    d = op_smin(d, sd_ellipsoid(lx, ly, lz, 0.0, -0.0120, 0.0300, 0.0600, 0.0600, 0.0700), 0.050)
    # jaw: ear -> chin, tapering
    d = op_smin(d, sd_round_cone(lx, ly, lz, -0.0600, -0.0180, -0.0040,
                                 -0.0110, -0.0800, 0.0560, 0.0225, 0.0165), 0.032)
    d = op_smin(d, sd_round_cone(lx, ly, lz, 0.0600, -0.0180, -0.0040,
                                 0.0110, -0.0800, 0.0560, 0.0225, 0.0165), 0.032)
    # chin
    d = op_smin(d, sd_ellipsoid(lx, ly, lz, 0.0, -0.0805, 0.0530, 0.0225, 0.0195, 0.0210), 0.018)
    # cheekbones
    d = op_smin(d, sd_sphere(lx, ly, lz, -0.0505, 0.0000, 0.0455, 0.0225), 0.038)
    d = op_smin(d, sd_sphere(lx, ly, lz, 0.0505, 0.0000, 0.0455, 0.0225), 0.038)
    # brow ridge
    d = op_smin(d, sd_capsule(lx, ly, lz, -0.0425, 0.0345, 0.0655, 0.0425, 0.0345, 0.0655, 0.0122), 0.030)
    # nose: short bridge, small tip
    d = op_smin(d, sd_capsule(lx, ly, lz, 0.0, 0.0255, 0.0700, 0.0, -0.0105, 0.0915, 0.0088), 0.017)
    d = op_smin(d, sd_sphere(lx, ly, lz, 0.0, -0.0155, 0.0940, 0.0118), 0.011)
    d = op_smin(d, sd_sphere(lx, ly, lz, -0.0135, -0.0230, 0.0870, 0.0085), 0.009)
    d = op_smin(d, sd_sphere(lx, ly, lz, 0.0135, -0.0230, 0.0870, 0.0085), 0.009)
    # mouth: carve the labial crease, then build both lips proud of the face
    crease = sd_capsule(lx, ly, lz, -0.0205, -0.0505, 0.0855, 0.0205, -0.0505, 0.0855, 0.0038)
    d = op_smax(d, -crease, 0.006)
    d = op_smin(d, sd_ellipsoid(lx, ly, lz, 0.0, -0.0425, 0.0862, 0.0235, 0.0072, 0.0080), 0.008)
    d = op_smin(d, sd_ellipsoid(lx, ly, lz, 0.0, -0.0580, 0.0840, 0.0205, 0.0082, 0.0082), 0.008)
    # chin dimple + jaw shading relief
    d = op_smin(d, sd_sphere(lx, ly, lz, 0.0, -0.0865, 0.0620, 0.0130), 0.016)
    # ears
    d = op_smin(d, sd_ellipsoid(lx, ly, lz, -0.0840, 0.0020, -0.0120, 0.0095, 0.0240, 0.0155), 0.011)
    d = op_smin(d, sd_ellipsoid(lx, ly, lz, 0.0840, 0.0020, -0.0120, 0.0095, 0.0240, 0.0155), 0.011)
    # eye sockets
    d = op_smax(d, -sd_ellipsoid(lx, ly, lz, -0.0295, 0.0185, 0.0880, 0.0212, 0.0118, 0.0150), 0.0045)
    d = op_smax(d, -sd_ellipsoid(lx, ly, lz, 0.0295, 0.0185, 0.0880, 0.0212, 0.0118, 0.0150), 0.0045)
    return d


@njit(inline="always", fastmath=True)
def _hair_sdf_local(lx, ly, lz, t):
    # smooth wet hair cap hugging the crown, swept back
    # wet hair shell hugging the crown, swept back off the forehead
    d = sd_ellipsoid(lx, ly, lz, 0.0, 0.030, -0.012, 0.0815, 0.0985, 0.0975)
    d = op_smax(d, -(lz - 0.0480), 0.048)
    d = op_smax(d, -(ly + 0.012), 0.040)
    # soft fringe rolling forward over the brow, smoothed into the cap
    fr = sd_ellipsoid(lx, ly, lz, 0.0, 0.0680, 0.0320, 0.0730, 0.0250, 0.0440)
    d = op_smin(d, fr, 0.045)
    # tapered temple wedges
    d = op_smin(d, sd_round_cone(lx, ly, lz, -0.0800, 0.016, 0.002, -0.0730, -0.036, 0.018, 0.012, 0.005), 0.028)
    d = op_smin(d, sd_round_cone(lx, ly, lz, 0.0800, 0.016, 0.002, 0.0730, -0.036, 0.018, 0.012, 0.005), 0.028)
    return d


@njit(inline="always", fastmath=True)
def map_head_world(px, py, pz, hx, hy, hz, yaw, pitch, detail):
    """
    Returns (distance, material) of the courier's head in world space.
    `detail` selects eye / lip / hair materials for the close-ups.
    """
    cy = math.cos(-yaw)
    sy = math.sin(-yaw)
    cp = math.cos(-pitch)
    sp = math.sin(-pitch)
    dx = px - hx
    dy = py - hy
    dz = pz - hz
    # yaw about Y
    lx = dx * cy - dz * sy
    lz0 = dx * sy + dz * cy
    # pitch about X
    ly = dy * cp - lz0 * sp
    lz = dy * sp + lz0 * cp

    d = _head_sdf_local(lx, ly, lz)
    mat = M_SKIN

    if detail > 0.5:
        hair = _hair_sdf_local(lx, ly, lz, 0.0)
        if hair < d:
            d = hair
            mat = M_HAIR
        # eyeballs sit back inside the almond openings
        e = sd_sphere(lx, ly, lz, -0.0295, 0.0185, 0.0688, 0.0165)
        if e < d:
            d = e
            mat = M_EYE
        e = sd_sphere(lx, ly, lz, 0.0295, 0.0185, 0.0688, 0.0165)
        if e < d:
            d = e
            mat = M_EYE
        # lash line along the upper edge of each opening
        for sx in (-1.0, 1.0):
            lash = sd_capsule(lx, ly, lz, sx * 0.0140, 0.0255, 0.0862,
                              sx * 0.0450, 0.0230, 0.0818, 0.0019)
            if lash < d:
                d = lash
                mat = M_HAIR
        # lips
        lip = op_union(
            sd_ellipsoid(lx, ly, lz, 0.0, -0.0425, 0.0862, 0.0235, 0.0072, 0.0080),
            sd_ellipsoid(lx, ly, lz, 0.0, -0.0580, 0.0840, 0.0205, 0.0082, 0.0082),
        )
        if lip < d - 1e-4:
            d = lip
            mat = M_LIP
        # mouth crease
        crease = sd_capsule(lx, ly, lz, -0.0205, -0.0505, 0.0858, 0.0205, -0.0505, 0.0858, 0.0024)
        if crease < d:
            d = crease
            mat = M_MOUTH
    return d, mat


# ---------------------------------------------------------------------------
# Stage 1 — Rain-slicked Tokyo alley
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _courier_sdf(px, py, pz, J):
    """Procedural human: raincoat, trousers, boots, leather gloves, chrome case."""
    d = 1e9
    mat = M_FABRIC

    # ---------- legs in dark trousers ----------
    dk = sd_round_cone(px, py, pz, J[12], J[13], J[14], J[15], J[16], J[17], 0.076, 0.056)
    dk2 = sd_round_cone(px, py, pz, J[18], J[19], J[20], J[21], J[22], J[23], 0.076, 0.056)
    dl = sd_round_cone(px, py, pz, J[0], J[1], J[2], J[12], J[13], J[14], 0.100, 0.076)
    dr = sd_round_cone(px, py, pz, J[0], J[1], J[2], J[18], J[19], J[20], 0.100, 0.076)
    legs = min(min(dl, dr), min(dk, dk2))
    if legs < d:
        d = legs
        mat = M_FABRIC

    # ---------- boots ----------
    bt = sd_rbox(px, py, pz, J[15], 0.046, J[17] + 0.050, 0.052, 0.046, 0.100, 0.026)
    bt2 = sd_rbox(px, py, pz, J[21], 0.046, J[23] + 0.050, 0.052, 0.046, 0.100, 0.026)
    boots = min(bt, bt2)
    if boots < d:
        d = boots
        mat = M_LEATHER

    # ---------- torso: tapered chest ----------
    torso = sd_round_cone(px, py, pz, J[0], J[1] - 0.02, J[2], J[3], J[4] - 0.02, J[5], 0.130, 0.150)
    torso = op_smin(torso, sd_ellipsoid(px, py, pz, J[3], J[4] + 0.01, J[5], 0.150, 0.115, 0.098), 0.06)
    if torso < d:
        d = torso
        mat = M_FABRIC

    # ---------- flared translucent raincoat ----------
    shoulder_y = J[4] - 0.012
    hem_y = J[1] - 0.30
    coat = sd_round_cone(px, py, pz, J[3], shoulder_y, J[5],
                         J[3], hem_y, J[2] - 0.01, 0.170, 0.196)
    # belt: a thin ring that breaks the silhouette at the waist
    belt = sd_round_cone(px, py, pz, J[3], J[1] + 0.02, J[2] + 0.004,
                         J[3], J[1] - 0.03, J[2] + 0.004, 0.196, 0.186)
    coat = op_smin(coat, sd_round_cone(px, py, pz, J[24], J[25] + 0.015, J[26],
                                       J[30], J[31] + 0.015, J[32], 0.072, 0.072), 0.075)
    coat = op_smin(coat, sd_round_cone(px, py, pz, J[24], J[25], J[26], J[27], J[28], J[29], 0.080, 0.066), 0.05)
    coat = op_smin(coat, sd_round_cone(px, py, pz, J[30], J[31], J[32], J[33], J[34], J[35], 0.080, 0.066), 0.05)
    coat = op_smin(coat, sd_round_cone(px, py, pz, J[27], J[28], J[29], J[36], J[37], J[38], 0.066, 0.054), 0.04)
    coat = op_smin(coat, sd_round_cone(px, py, pz, J[33], J[34], J[35], J[39], J[40], J[41], 0.066, 0.054), 0.04)
    coat = op_smax(coat, -belt, 0.008)
    # pop-up collar: a short ring at the base of the neck, cut away at the front
    collar = sd_round_cone(px, py, pz, J[9], J[10] - 0.150, J[11] - 0.012,
                           J[9], J[10] - 0.098, J[11] - 0.012, 0.078, 0.064)
    collar = op_smax(collar, -(sd_rbox(px, py, pz, J[9], J[10] - 0.10, J[11] + 0.60,
                                       6.0, 0.14, 0.55, 0.01)), 0.03)
    coat = op_smin(coat, collar, 0.05)
    # vertical cloth folds: cheap angular ripple in the coat's distance field
    ang = math.atan2(px - J[3], pz - J[5])
    fold_amt = 1.0 if py < J[1] else 0.15
    fold = (math.sin(ang * 7.0 + py * 1.8) * 0.0022 + math.sin(ang * 13.0 - py * 4.0) * 0.0010) * fold_amt
    coat = coat + fold
    if coat < d:
        d = coat
        mat = M_RAINCOAT

    # ---------- glove fists ----------
    gl = sd_ellipsoid(px, py, pz, J[36], J[37], J[38], 0.052, 0.045, 0.062)
    gr = sd_ellipsoid(px, py, pz, J[39], J[40], J[41], 0.052, 0.045, 0.062)
    gloves = min(gl, gr)
    if gloves < d:
        d = gloves
        mat = M_LEATHER

    # ---------- neck + head ----------
    neck = sd_round_cone(px, py, pz, J[6], J[7] - 0.055, J[8], J[9], J[10] - 0.070, J[11], 0.062, 0.052)
    hr, hm = map_head_world(px, py, pz, J[9], J[10], J[11], J[42], J[43], J[46])
    if neck < hr:
        if neck < d:
            d = neck
            mat = M_SKIN
    else:
        if hr < d:
            d = hr
            mat = hm
    return d, mat


@njit(inline="always", fastmath=True)
def _facade(px, py, pz, side, t):
    """
    Procedural Tokyo facade shader for the two alley walls.

    Returns (ar, ag, ab, er, eg, eb, bump) where `bump` is a vertical relief
    value used to perturb the wall normal. Everything (concrete panels, lit
    windows, vertical neon kanji blades, pipes, air-conditioners, awnings) is
    an analytic function of the wall coordinates — no textures, no photos.
    """
    u = pz
    v = py
    # ---------------- base concrete / tile panels ----------------
    panel = _fbm3(u * 1.6, v * 2.1, side * 13.0, 17.0, 4)
    tiles = 0.5 + 0.5 * math.sin(u * 7.5) * math.sin(v * 7.5)
    g = 0.026 + 0.042 * panel + 0.014 * tiles
    ar = g * 1.06
    ag = g * 1.00
    ab = g * 1.02
    er = 0.0
    eg = 0.0
    eb = 0.0
    bump = panel * 0.35

    # ---------------- ground-floor shopfronts ----------------
    shop = _mod(u + 40.0, 5.6)
    if v < 2.55 and shop > 0.9 and shop < 4.6:
        # dark interior with a few concentrated practicals inside
        ar, ag, ab = 0.012, 0.012, 0.015
        er, eg, eb = 0.012, 0.010, 0.008
        bump = -0.4
        # ceiling strip lights
        strip = _mod(v - 0.10, 1.35)
        if strip < 0.10 and abs(_mod(u, 1.55) - 0.75) < 0.55:
            er, eg, eb = 0.95, 0.62, 0.30
        # hanging warm bulbs deep inside
        bulb = _mod(u * 1.6, 1.0)
        if _mod(v - 1.62, 0.9) < 0.10 and 0.35 < bulb < 0.45:
            er, eg, eb = 1.35, 0.80, 0.34
        # shelving silhouettes
        if 0.55 < v < 1.95 and _mod(v * 6.0, 1.0) < 0.14:
            er *= 0.12
            eg *= 0.11
            eb *= 0.10
        # glass reflection of the street neon (cyan/magenta sheen on the glass)
        gl = math.exp(-abs(_mod(u, 5.6) - 2.75) * 0.8)
        ar += 0.020 * gl
        ag += 0.034 * gl
        ab += 0.048 * gl
        # mullions & door frame
        if _mod(u, 1.15) < 0.08 or v < 0.16 or v > 2.44:
            ar, ag, ab = 0.030, 0.030, 0.033
            er, eg, eb = 0.0, 0.0, 0.0
            bump = 0.6

    # ---------------- awning band ----------------
    if 2.55 <= v < 2.85 and shop > 0.7 and shop < 4.8:
        ar, ag, ab = 0.09, 0.05, 0.055
        er, eg, eb = 0.0, 0.0, 0.0
        if _mod(u * 3.0, 1.0) < 0.5:
            ar, ag, ab = 0.15, 0.07, 0.06
        bump = 0.8

    # ---------------- upper window grid ----------------
    if v > 3.2 and v < 7.1:
        wu = _mod(u + 40.0, 3.05)
        wv = _mod(v - 3.2, 1.42)
        if 0.55 < wu < 2.35 and 0.28 < wv < 1.20:
            cell_u = math.floor((u + 40.0) / 3.05)
            cell_v = math.floor((v - 3.2) / 1.42)
            lit = _hash21(cell_u + side * 31.0, cell_v, 7.0)
            ar, ag, ab = 0.020, 0.024, 0.032
            bump = -0.5
            if lit > 0.46:
                warm = 0.45 + 0.75 * _hash21(cell_u, cell_v + 3.0, 9.0)
                if lit > 0.86:
                    er = 0.30 * warm
                    eg = 0.44 * warm
                    eb = 0.58 * warm
                else:
                    er = 0.62 * warm
                    eg = 0.40 * warm
                    eb = 0.19 * warm
                # curtains / blinds slicing the glow
                bl = _mod(v * 26.0, 1.0)
                if bl < 0.45:
                    er *= 0.28
                    eg *= 0.26
                    eb *= 0.25
                # distant human silhouette in a few windows
                if lit > 0.63 and lit < 0.70:
                    du = (wu - 1.45) / 0.32
                    dv = (wv - 0.55) / 0.42
                    if du * du + dv * dv < 1.0:
                        er *= 0.10
                        eg *= 0.10
                        eb *= 0.11
            # window frame
            if wu < 0.62 or wu > 2.28 or wv < 0.35 or wv > 1.13:
                ar, ag, ab = 0.075, 0.072, 0.070
                er, eg, eb = 0.0, 0.0, 0.0
                bump = 0.7
            # window sill
            if wv < 0.34:
                ar, ag, ab = 0.10, 0.098, 0.095
                er, eg, eb = er * 0.2, eg * 0.2, eb * 0.2
                bump = 1.0

    # ---------------- vertical neon kanji blade signs ----------------
    # Sign bands every 4.6 m, alternating colour capsules
    band = math.floor((u + 40.0) / 4.6)
    bu = _mod(u + 40.0, 4.6)
    sign_y0 = 2.95 + 0.25 * math.sin(band * 1.7)
    if 1.05 < bu < 2.05 and sign_y0 < v < sign_y0 + 2.35:
        # sign cabinet
        ar, ag, ab = 0.035, 0.035, 0.040
        bump = 1.1
        hue = _hash21(band, side, 13.0)
        if hue < 0.34:
            sr, sg, sb = 0.12, 0.92, 1.00   # cyan
        elif hue < 0.67:
            sr, sg, sb = 1.00, 0.16, 0.72   # magenta
        else:
            sr, sg, sb = 1.00, 0.62, 0.18   # sodium amber
        flick = 0.72 + 0.28 * math.sin(t * 21.0 + band * 2.0) * math.sin(t * 2.7 + band)
        # glyph strokes: pseudo-kanji built from horizontal + vertical bars
        lu = (bu - 1.05) / 1.0
        lv = (v - sign_y0) / 2.35
        glyph = 0.0
        char_row = math.floor(lv * 4.0)
        if char_row < 3.0:
            cy = _mod(lv * 4.0, 1.0)
            cx = lu
            # each "character" is a 3x4 stroke lattice with hash-driven omissions
            ci = math.floor(cx * 3.0)
            cj = math.floor(cy * 4.0)
            fx = _mod(cx * 3.0, 1.0)
            fy = _mod(cy * 4.0, 1.0)
            present = _hash21(char_row * 7.0 + ci, cj + band, 21.0)
            if present > 0.42:
                if fx > 0.18 and fx < 0.82 and fy > 0.20 and fy < 0.80:
                    if (fy < 0.36) or (fy > 0.60 and fy < 0.76) or (fx < 0.40):
                        glyph = 1.0
        if glyph > 0.5:
            er = sr * 4.4 * flick
            eg = sg * 4.4 * flick
            eb = sb * 4.4 * flick
            ar, ag, ab = 0.02, 0.02, 0.02
        # sign edge glow spill
        elif lu < 0.06 or lu > 0.94 or lv < 0.03 or lv > 0.97:
            er = sr * 0.55 * flick
            eg = sg * 0.55 * flick
            eb = sb * 0.55 * flick

    # ---------------- service pipes ----------------
    pu = _mod(u + 12.0, 6.4)
    if 0.42 < pu < 0.60 and v < 7.6:
        shade = 0.085 + 0.05 * math.sin(pu * 60.0)
        ar, ag, ab = shade, shade * 1.03, shade * 1.08
        er, eg, eb = er * 0.2, eg * 0.2, eb * 0.2
        bump = 1.4

    # ---------------- AC condensers on brackets ----------------
    au = _mod(u + 30.0, 5.3)
    if 4.0 < v < 6.9 and 2.9 < au < 3.95 and _hash21(math.floor((u + 30.0) / 5.3), side, 33.0) > 0.45:
        ar, ag, ab = 0.11, 0.115, 0.125
        bump = 1.2
        # louvre slats
        if _mod(v * 40.0, 1.0) < 0.55:
            ar, ag, ab = 0.06, 0.062, 0.068

    return ar, ag, ab, er, eg, eb, bump


@njit(inline="always", fastmath=True)
def _roof_shade(px, py, pz):
    """Corrugated metal / rusted roof albedo."""
    m = _mod(pz * 18.0, 1.0)
    corr = 0.5 + 0.5 * math.sin(m * math.pi * 2.0)
    n = _fbm3(px * 2.4, py * 2.4, pz * 2.4, 211.0, 4)
    rust = _fbm3(px * 5.0, pz * 5.0, 3.0, 223.0, 3)
    g = 0.030 + 0.030 * corr + 0.030 * n
    rr = max(0.0, rust - 0.55)
    return g * (1.10 + 0.55 * rr), g * (1.00 + 0.12 * rr), g * (0.94 - 0.22 * rr)


@njit(cache=True, fastmath=True)
def map_rooftop(px, py, pz, J, t, params):
    """Rooftop chase: launch deck, gap, one-storey-lower landing deck, skyline."""
    d = 1e9
    mat = M_ROOF

    if pz < 15.0:
        if py < d:
            d = py
            mat = M_ROOF
    else:
        db = py + 1.35
        if db < d:
            d = db
            mat = M_ROOF

    dst = py + 14.0
    if dst < d and pz > 13.4 and pz < 20.6:
        d = dst
        mat = M_ASPHALT

    pp = sd_rbox(px, py, pz, 0.0, 0.34, 14.55, 5.2, 0.34, 0.16, 0.02)
    if pp < d:
        d = pp
        mat = M_CONCRETE
    pp2 = sd_rbox(px, py, pz, -5.4, 0.34, 2.0, 0.16, 0.34, 16.6, 0.02)
    pp3 = sd_rbox(px, py, pz, 5.4, 0.34, 2.0, 0.16, 0.34, 16.6, 0.02)
    ppx = min(pp2, pp3)
    if ppx < d:
        d = ppx
        mat = M_CONCRETE
    lp = sd_rbox(px, py, pz, 0.0, -1.01, 19.4, 5.2, 0.34, 0.16, 0.02)
    if lp < d:
        d = lp
        mat = M_CONCRETE

    # ---- gated clutter: only evaluated when the ray is anywhere near it ----
    if abs(pz - 11.6) < 4.0 and abs(px) < 6.0:
        ac = sd_rbox(px, py, pz, 3.05, 0.45, 11.6, 0.90, 0.45, 0.62, 0.04)
        if ac < d:
            d = ac
            mat = M_METAL
        acf = sd_rbox(px, py, pz, 3.05, 0.45, 11.15, 0.82, 0.38, 0.06, 0.02)
        if acf < d:
            d = acf
            mat = M_METAL

    if abs(pz - 6.0) < 4.0 and abs(px + 3.4) < 4.0:
        wt = sd_round_cone(px, py, pz, -3.4, 0.0, 6.0, -3.4, 1.55, 6.0, 0.95, 0.95)
        if wt < d:
            d = wt
            mat = M_METAL
        wtl = sd_round_cone(px, py, pz, -3.4, 1.55, 6.0, -3.4, 1.80, 6.0, 0.95, 0.30)
        if wtl < d:
            d = wtl
            mat = M_METAL

    if abs(pz - 2.2) < 4.0 and abs(px + 2.0) < 3.5:
        ant = sd_round_cone(px, py, pz, -2.0, 0.0, 2.2, -2.0, 6.4, 2.2, 0.09, 0.035)
        if ant < d:
            d = ant
            mat = M_METAL

    if abs(pz - 9.0) < 6.0 and abs(px) < 6.0:
        for i in range(5):
            fi = float(i)
            vx = -4.2 + fi * 2.0
            vz = 4.0 + fi * 2.1
            if abs(vz - pz) > 2.0 or abs(vx - px) > 2.0:
                continue
            vp = sd_round_cone(px, py, pz, vx, 0.0, vz, vx, 0.62 + 0.2 * fi, vz, 0.11, 0.11)
            if vp < d:
                d = vp
                mat = M_METAL
            cap = sd_round_cone(px, py, pz, vx, 0.62 + 0.2 * fi, vz, vx, 0.70 + 0.2 * fi, vz, 0.16, 0.20)
            if cap < d:
                d = cap
                mat = M_METAL

    if abs(pz - 10.4) < 4.5 and abs(px + 4.3) < 4.0:
        hut = sd_rbox(px, py, pz, -4.3, 1.20, 10.4, 1.10, 1.20, 1.40, 0.03)
        if hut < d:
            d = hut
            mat = M_CONCRETE

    # whipping laundry lines (only in the alley-run zone)
    if pz > 3.0 and pz < 16.0 and abs(px) < 7.0:
        for i in range(7):
            fi = float(i)
            z0 = 6.2 + fi * 1.15
            if abs(z0 - pz) > 3.2:
                continue
            y0 = 1.85 + 0.16 * math.sin(fi * 1.3)
            for seg in range(3):
                fs = float(seg)
                axs = -5.0 + 10.0 * fs / 3.0
                bxs = -5.0 + 10.0 * (fs + 1.0) / 3.0
                sag = 0.24 * math.sin(fs * 1.2) + 0.30 * math.sin(t * 2.6 + fi * 1.7 + fs * 0.9)
                sagb = 0.24 * math.sin((fs + 1.0) * 1.2) + 0.30 * math.sin(t * 2.6 + fi * 1.7 + (fs + 1.0) * 0.9)
                ln = sd_capsule(px, py, pz, axs, y0 - sag, z0, bxs, y0 - sagb, z0, 0.007)
                if ln < d:
                    d = ln
                    mat = M_FABRIC
            if _hash21(fi, 3.0, 17.0) > 0.35:
                hx = -3.2 + fi * 1.15
                flap = 0.06 * math.sin(t * 5.2 + fi)
                sh = sd_rbox(px, py, pz, hx + flap, y0 - 0.52, z0, 0.34, 0.46, 0.028, 0.02)
                if sh < d:
                    d = sh
                    mat = M_FABRIC

    # distant skyline: only sampled once the ray is already past the roof
    if pz > 20.0:
        for i in range(10):
            fi = float(i)
            bx = -34.0 + fi * 6.8
            bh = 6.0 + 16.0 * _hash21(fi, 5.0, 31.0)
            bz = 30.0 + 9.0 * _hash21(fi, 9.0, 37.0)
            bd = 3.0 + 4.0 * _hash21(fi, 11.0, 41.0)
            bl = sd_rbox(px, py, pz, bx, -14.0 + bh * 0.5, bz, 3.4, bh * 0.5, bd, 0.04)
            if bl < d:
                d = bl
                mat = M_FAR
        nb = sd_rbox(px, py, pz, 19.0, 2.0, 40.0, 9.0, 16.0, 12.0, 0.05)
        if nb < d:
            d = nb
            mat = M_FAR

    # drone, gated to a small box around it
    dxp = params[7]
    dyp = params[8] + 0.10 * math.sin(t * 4.1)
    dzp = 10.5 - 1.4 * t
    if abs(px - dxp) < 1.2 and abs(py - dyp) < 1.2 and abs(pz - dzp) < 1.2:
        dr = sd_rbox(px, py, pz, dxp, dyp, dzp, 0.19, 0.075, 0.19, 0.03)
        if dr < d:
            d = dr
            mat = M_METAL
        for i in range(4):
            sx = -1.0 if i % 2 == 0 else 1.0
            sz = -1.0 if i < 2 else 1.0
            arm = sd_capsule(px, py, pz, dxp, dyp, dzp, dxp + sx * 0.26, dyp + 0.03, dzp + sz * 0.26, 0.018)
            if arm < d:
                d = arm
                mat = M_METAL
            rot = sd_round_cone(px, py, pz, dxp + sx * 0.26, dyp + 0.05, dzp + sz * 0.26,
                                dxp + sx * 0.26, dyp + 0.09, dzp + sz * 0.26, 0.115, 0.115)
            if rot < d:
                d = rot
                mat = M_GLASS

    if abs(pz - J[11]) < 1.6 and abs(px - J[9]) < 1.2 and py < 2.4:
        cd, cm = _courier_sdf(px, py, pz, J)
        if cd < d:
            d = cd
            mat = cm
    cs = sd_rbox(px, py, pz, J[44], J[45], J[46], 0.185, 0.125, 0.072, 0.028)
    if cs < d:
        d = cs
        mat = M_CHROME
    if params[1] > 0.02:
        sd_ = sd_sphere(px, py, pz, J[44], J[45] + 0.06, J[46], 0.052)
        if sd_ < d:
            d = sd_
            mat = M_SEED
    return d, mat


@njit(cache=True, fastmath=True)
def map_seed(px, py, pz, J, t, params):
    """
    Macro stage, local space (metres), +z toward camera:
      * the seed glows at the origin, resting in the cupped palm
      * fingers frame it from the left, the thumb from the right, palm below
      * the courier's face sits behind at z = -0.60, far out of focus
    params: 5 seed glow, 6 root growth
    """
    d = 1e9
    mat = M_LEATHER

    # ---------------- palm cupped below the seed ----------------
    palm = sd_rbox(px, py, pz, 0.0, -0.085, -0.010, 0.082, 0.042, 0.072, 0.046)
    if palm < d:
        d = palm
        mat = M_LEATHER
    # heel of the hand, rolling away from camera
    heel = sd_ellipsoid(px, py, pz, -0.030, -0.115, -0.085, 0.072, 0.055, 0.070)
    if heel < d:
        d = heel
        mat = M_LEATHER

    # ---------------- fingers framing the seed from the left ----------------
    for i in range(3):
        fi = float(i)
        fx = -0.070 - fi * 0.016
        fz = 0.048 - fi * 0.030
        top = 0.030 - fi * 0.010
        p1 = (fx + 0.012, -0.058, fz + 0.028)
        p2 = (fx, -0.004, fz + 0.010)
        p3 = (fx + 0.014, top - 0.020, fz - 0.030)
        p4 = (fx + 0.036, top - 0.036, fz - 0.046)
        seg1 = sd_capsule(px, py, pz, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], 0.0142)
        if seg1 < d:
            d = seg1
            mat = M_LEATHER
        seg2 = sd_capsule(px, py, pz, p2[0], p2[1], p2[2], p3[0], p3[1], p3[2], 0.0128)
        if seg2 < d:
            d = seg2
            mat = M_LEATHER
        seg3 = sd_capsule(px, py, pz, p3[0], p3[1], p3[2], p4[0], p4[1], p4[2], 0.0112)
        if seg3 < d:
            d = seg3
            mat = M_LEATHER
        kn = sd_sphere(px, py, pz, p2[0], p2[1], p2[2], 0.0148)
        if kn < d:
            d = kn
            mat = M_LEATHER

    # ---------------- thumb wrapping from the right ----------------
    th1 = sd_capsule(px, py, pz, 0.086, -0.090, 0.010, 0.072, -0.020, 0.042, 0.0190)
    if th1 < d:
        d = th1
        mat = M_LEATHER
    th2 = sd_capsule(px, py, pz, 0.072, -0.020, 0.042, 0.046, 0.010, 0.030, 0.0158)
    if th2 < d:
        d = th2
        mat = M_LEATHER

    # ---------------- wrist leading out of the lower-left frame ----------------
    cf = sd_round_cone(px, py, pz, -0.115, -0.240, -0.150, -0.045, -0.120, -0.040, 0.062, 0.052)
    if cf < d:
        d = cf
        mat = M_LEATHER

    # leather grain: fine stipple relief so the glove never reads as plastic
    if mat == M_LEATHER:
        d += _vnoise3(px * 120.0, py * 120.0, pz * 120.0, 71.0) * 0.0012

    # ---------------- the seed ----------------
    seed = sd_sphere(px, py, pz, 0.0, 0.006, 0.0, 0.0345)
    if seed < d:
        d = seed
        mat = M_SEED

    # ---------------- roots of light crawling over the palm ----------------
    grow = params[6]
    if grow > 0.01:
        for i in range(7):
            fi = float(i)
            a = -2.55 + fi * 0.62
            ca = math.cos(a)
            sa = math.sin(a)
            # three-point polyline that hugs the palm surface
            r1 = 0.030
            r2 = 0.030 + grow * (0.030 + 0.012 * _hash21(fi, 1.0, 59.0))
            r3 = 0.030 + grow * (0.062 + 0.022 * _hash21(fi, 3.0, 61.0))
            bend = 0.45 * math.sin(fi * 2.3)
            b1x, b1z = ca * r1, sa * r1
            b2x = math.cos(a + bend) * r2
            b2z = math.sin(a + bend) * r2
            b3x = math.cos(a + bend * 1.7) * r3
            b3z = math.sin(a + bend * 1.7) * r3
            y1 = -0.010
            y2 = -0.041 - 0.006 * math.sin(fi * 1.7)
            y3 = -0.050 + 0.004 * math.cos(fi)
            rt1 = sd_capsule(px, py, pz, b1x, y1, b1z, b2x, y2, b2z, 0.00125)
            if rt1 < d:
                d = rt1
                mat = M_ROOT
            if grow > 0.35:
                rt2 = sd_capsule(px, py, pz, b2x, y2, b2z, b3x, y3, b3z, 0.00090)
                if rt2 < d:
                    d = rt2
                    mat = M_ROOT

    # ---------------- the courier's face behind, soft in bokeh ----------------
    if pz < -0.26:
        hr, hm = map_head_world(px, py, pz, 0.030, 0.075, -0.470, 3.06, 0.46, 1.0)
        if hr < d:
            d = hr
            mat = hm

    return d, mat


@njit(inline="always", fastmath=True)
def map_alley(px, py, pz, J, t, params):
    """params: 0 cat_x, 1 case_open, 2 flicker, 3 steam, 4 rain, 5 seed_glow, 6 holo"""
    d = 1e9
    mat = M_CONCRETE

    # ================= wet asphalt =================
    if py < d:
        d = py
        mat = M_ASPHALT

    # kerbs
    if abs(pz - 12.0) < 26.0:
        kb = sd_rbox(px, py, pz, -2.14, 0.052, pz, 0.10, 0.055, 1.0, 0.012)
        kb2 = sd_rbox(px, py, pz, 2.14, 0.052, pz, 0.10, 0.055, 1.0, 0.012)
        kb = min(kb, kb2)
        if kb < d:
            d = kb
            mat = M_CONCRETE

    # drain grate
    gr = sd_rbox(px, py, pz, 0.85, 0.010, 6.4, 0.50, 0.014, 0.38, 0.02)
    if gr < d:
        d = gr
        mat = M_STEEL
    if _hash21(math.floor(px * 2.0), math.floor(pz * 2.0), 5.0) > 0.55:
        pass

    # ================= walls (flat planes + facade shader) =================
    lw = px + 2.24
    if lw < d:
        d = lw
        mat = M_FACADE
    rw = 2.24 - px
    if rw < d:
        d = rw
        mat = M_FACADE

    # ================= end of the alley: cross street =================
    fw = sd_rbox(px, py, pz, 0.0, 5.0, 34.0, 1.6, 5.0, 0.8, 0.05)
    if fw < d:
        d = fw
        mat = M_FACADE

    # ================= overhead cables =================
    for i in range(6):
        zc = 1.0 + float(i) * 4.2
        if abs(zc - pz) > 5.0:
            continue
        sag = 0.32 + 0.14 * math.sin(float(i) * 1.9)
        c1 = sd_capsule(px, py, pz, -2.24, 5.6, zc, 0.0, 5.6 - sag, zc + 0.4, 0.014)
        c2 = sd_capsule(px, py, pz, 0.0, 5.6 - sag, zc + 0.4, 2.24, 5.75, zc + 0.8, 0.014)
        cc = min(c1, c2)
        if cc < d:
            d = cc
            mat = M_METAL

    # ================= hanging paper lanterns =================
    for i in range(4):
        zc = 3.2 + float(i) * 4.6
        lx = -0.92 + 0.66 * math.sin(float(i) * 2.3)
        lb = sd_sphere(px, py, pz, lx, 2.78, zc, 0.135)
        if lb < d:
            d = lb
            mat = M_LANTERN
        cord = sd_capsule(px, py, pz, lx, 2.90, zc, lx, 5.3, zc, 0.005)
        if cord < d:
            d = cord
            mat = M_METAL

    # ================= vending machine =================
    vm = sd_rbox(px, py, pz, 1.98, 0.86, 9.4, 0.22, 0.86, 0.36, 0.02)
    if vm < d:
        d = vm
        mat = M_VENDING

    # ================= crates, bin, bike =================
    cr = sd_rbox(px, py, pz, -1.74, 0.22, 12.9, 0.32, 0.22, 0.28, 0.02)
    if cr < d:
        d = cr
        mat = M_METAL
    bn = sd_round_cone(px, py, pz, 1.76, 0.0, 13.9, 1.76, 0.76, 13.9, 0.24, 0.28)
    if bn < d:
        d = bn
        mat = M_METAL
    # parked bicycle silhouette
    bw1 = sd_capsule(px, py, pz, -1.55, 0.42, 15.6, -1.15, 0.62, 16.5, 0.018)
    bw2 = sd_capsule(px, py, pz, -1.15, 0.62, 16.5, -1.52, 0.86, 16.4, 0.018)
    bw3 = sd_capsule(px, py, pz, -1.52, 0.86, 16.4, -1.72, 0.44, 17.3, 0.018)
    bw = min(min(bw1, bw2), bw3)
    if bw < d:
        d = bw
        mat = M_METAL

    # ================= stray cat =================
    cx0 = params[0]
    cz0 = 7.85
    cb = sd_ellipsoid(px, py, pz, cx0, 0.205, cz0, 0.078, 0.070, 0.195)
    if cb < d:
        d = cb
        mat = M_FUR
    ch = sd_sphere(px, py, pz, cx0, 0.272, cz0 - 0.215, 0.056)
    if ch < d:
        d = ch
        mat = M_FUR
    ce1 = sd_round_cone(px, py, pz, cx0 - 0.030, 0.302, cz0 - 0.226, cx0 - 0.044, 0.350, cz0 - 0.232, 0.019, 0.002)
    if ce1 < d:
        d = ce1
        mat = M_FUR
    ce2 = sd_round_cone(px, py, pz, cx0 + 0.030, 0.302, cz0 - 0.226, cx0 + 0.044, 0.350, cz0 - 0.232, 0.019, 0.002)
    if ce2 < d:
        d = ce2
        mat = M_FUR
    for i in range(4):
        sx = -1.0 if (i % 2 == 0) else 1.0
        sz = -1.0 if i < 2 else 1.0
        ph = math.sin(t * 13.0 + float(i) * 1.7)
        l1 = sd_round_cone(px, py, pz, cx0 + sx * 0.042, 0.185, cz0 + sz * 0.125,
                           cx0 + sx * 0.052, 0.008, cz0 + sz * 0.145 + 0.060 * ph, 0.019, 0.012)
        if l1 < d:
            d = l1
            mat = M_FUR
    tail = sd_capsule(px, py, pz, cx0, 0.220, cz0 + 0.185, cx0 + 0.065, 0.360, cz0 + 0.290, 0.015)
    if tail < d:
        d = tail
        mat = M_FUR

    # ================= courier + chrome case =================
    if abs(pz - J[11]) < 1.6 and abs(px - J[9]) < 1.2 and py < 2.4:
        cd, cm = _courier_sdf(px, py, pz, J)
        if cd < d:
            d = cd
            mat = cm
    cs = sd_rbox(px, py, pz, J[44], 0.86, J[45], 0.185, 0.125, 0.072, 0.028)
    if cs < d:
        d = cs
        mat = M_CHROME
    hd = sd_capsule(px, py, pz, J[44], 1.02, J[45], J[44], 1.09, J[45], 0.013)
    if hd < d:
        d = hd
        mat = M_METAL

    return d, mat


# ---------------------------------------------------------------------------
# Shading helpers
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _albedo(mat, px, py, pz, t):
    if mat == M_ASPHALT:
        n = _fbm3(px * 3.4, py * 3.4, pz * 3.4, 11.0, 4)
        v = 0.055 + 0.085 * n
        return v, v * 1.02, v * 1.12
    if mat == M_CONCRETE:
        n = _fbm3(px * 1.7, py * 1.9, pz * 1.7, 23.0, 5)
        v = 0.075 + 0.135 * n
        return v * 1.05, v, v * 0.98
    if mat == M_METAL:
        n = _fbm3(px * 6.0, py * 6.0, pz * 6.0, 31.0, 3)
        v = 0.10 + 0.07 * n
        return v, v * 1.03, v * 1.10
    if mat == M_CHROME:
        return 0.52, 0.55, 0.60
    if mat == M_STEEL:
        return 0.20, 0.21, 0.23
    if mat == M_NEON:
        return 0.030, 0.030, 0.034
    if mat == M_SKIN:
        n = _fbm3(px * 42.0, py * 42.0, pz * 42.0, 41.0, 3)
        base = 0.170 + 0.045 * n
        return base * 1.00, base * 0.735, base * 0.640
    if mat == M_RAINCOAT:
        return 0.085, 0.105, 0.130
    if mat == M_FABRIC:
        n = _fbm3(px * 8.0, py * 8.0, pz * 8.0, 53.0, 3)
        v = 0.055 + 0.045 * n
        return v, v * 1.03, v * 1.12
    if mat == M_LANTERN:
        pulse = 0.86 + 0.14 * math.sin(t * 6.0 + px * 3.0)
        return 1.55 * pulse, 0.78 * pulse, 0.34 * pulse
    if mat == M_VENDING:
        rows = _mod(py * 7.0, 1.0)
        if py > 1.02 and rows < 0.68:
            return 0.10, 0.30, 0.34
        return 0.045, 0.050, 0.060
    if mat == M_HOLO:
        return 0.02, 0.35, 0.22
    if mat == M_SEED:
        return 0.10, 0.55, 1.00
    if mat == M_ROOT:
        return 0.20, 0.80, 1.00
    if mat == M_LEATHER:
        n = _fbm3(px * 55.0, py * 55.0, pz * 55.0, 61.0, 4)
        n2 = _fbm3(px * 160.0, py * 160.0, pz * 160.0, 67.0, 2)
        v = 0.030 + 0.036 * n + 0.018 * n2
        return v * 1.00, v * 0.96, v * 0.94
    if mat == M_WINDOW:
        return 0.22, 0.26, 0.30
    if mat == M_WATER:
        return 0.03, 0.04, 0.06
    if mat == M_HAIR:
        return 0.028, 0.024, 0.022
    if mat == M_FUR:
        v = 0.075 + 0.05 * _fbm3(px * 30.0, py * 30.0, pz * 30.0, 71.0, 3)
        return v, v * 0.95, v * 0.88
    if mat == M_ROOF:
        return _roof_shade(px, py, pz)
    if mat == M_EYE:
        return 0.26, 0.24, 0.22
    if mat == M_LIP:
        n = _fbm3(px * 120.0, py * 120.0, pz * 120.0, 251.0, 2)
        base = 0.135 + 0.030 * n
        return base * 1.05, base * 0.58, base * 0.55
    if mat == M_MOUTH:
        return 0.045, 0.022, 0.022
    if mat == M_GLASS:
        return 0.12, 0.14, 0.17
    if mat == M_GLASS:
        return 0.12, 0.14, 0.17
    return 0.12, 0.12, 0.13


@njit(inline="always", fastmath=True)
def _sky(dirx, diry, dirz, city_glow):
    """Night sky dome: deep indigo zenith, sodium-tinted horizon haze."""
    up = diry
    if up < 0.0:
        up = 0.0
    zenith = 0.0030, 0.0045, 0.0130
    horizon = 0.0280, 0.0205, 0.0300
    f = math.exp(-up * 5.2)
    r = zenith[0] + (horizon[0] - zenith[0]) * f
    g = zenith[1] + (horizon[1] - zenith[1]) * f
    b = zenith[2] + (horizon[2] - zenith[2]) * f
    # distant city bounce light band
    band = math.exp(-abs(up - 0.06) * 12.0)
    r += 0.070 * band * city_glow
    g += 0.042 * band * city_glow
    b += 0.032 * band * city_glow
    return r, g, b


@njit(inline="always", fastmath=True)
def _neon_env(dirx, diry, dirz, t):
    """
    Cheap directional environment radiance: cyan & magenta practicals plus the
    warm ramen-shop pool. This is what wets the asphalt with colour.
    """
    r = 0.0
    g = 0.0
    b = 0.0
    # cyan lobe, screen-left and above
    d1 = dirx * -0.62 + diry * 0.30 + dirz * 0.72
    e1 = math.exp(-(1.0 - d1) * 7.0) * 0.95
    r += 0.10 * e1
    g += 0.72 * e1
    b += 0.95 * e1
    # magenta lobe, screen-right
    d2 = dirx * 0.66 + diry * 0.16 + dirz * 0.73
    e2 = math.exp(-(1.0 - d2) * 7.2) * 0.90
    r += 0.92 * e2
    g += 0.14 * e2
    b += 0.66 * e2
    # warm sodium pool low and ahead (ramen shop)
    d3 = dirx * 0.10 - diry * 0.32 + dirz * 0.94
    e3 = math.exp(-(1.0 - d3) * 6.0) * 0.75
    r += 0.95 * e3
    g += 0.55 * e3
    b += 0.22 * e3
    # flickering cold fill from the overhead tube
    d4 = dirx * 0.0 + diry * 0.96 + dirz * 0.28
    e4 = math.exp(-(1.0 - d4) * 9.0) * (0.55 + 0.45 * math.sin(t * 31.0))
    r += 0.24 * e4
    g += 0.44 * e4
    b += 0.52 * e4
    return r, g, b


@njit(cache=True, fastmath=True)
def _scene_map(px, py, pz, stage, J, t, params):
    if stage == 0:
        return map_alley(px, py, pz, J, t, params)
    if stage == 1:
        return map_rooftop(px, py, pz, J, t, params)
    if stage == 2:
        return map_seed(px, py, pz, J, t, params)
    return 1e9, M_CONCRETE


@njit(cache=True, fastmath=True)
def _calc_normal(px, py, pz, stage, J, t, params, eps):
    d0 = _scene_map(px, py, pz, stage, J, t, params)[0]
    dx = _scene_map(px + eps, py, pz, stage, J, t, params)[0] - d0
    dy = _scene_map(px, py + eps, pz, stage, J, t, params)[0] - d0
    dz = _scene_map(px, py, pz + eps, stage, J, t, params)[0] - d0
    l = math.sqrt(dx * dx + dy * dy + dz * dz)
    if l < 1e-9:
        return 0.0, 1.0, 0.0
    return dx / l, dy / l, dz / l


@njit(cache=True, fastmath=True)
def _soft_shadow(px, py, pz, lx, ly, lz, stage, J, t, params, max_t):
    res = 1.0
    tt = 0.03
    for _ in range(16):
        hx = px + lx * tt
        hy = py + ly * tt
        hz = pz + lz * tt
        d = _scene_map(hx, hy, hz, stage, J, t, params)[0]
        if d < 0.001:
            return 0.0
        res = min(res, 9.0 * d / tt)
        tt += min(max(d, 0.03), 0.75)
        if tt > max_t:
            break
    res = 0.0 if res < 0.0 else res
    return res * res * (3.0 - 2.0 * res)


@njit(cache=True, fastmath=True)
def _ambient_occlusion(px, py, pz, nx, ny, nz, stage, J, t, params):
    occ = 0.0
    sca = 1.0
    for i in range(4):
        h = 0.025 + 0.13 * i
        d = _scene_map(px + nx * h, py + ny * h, pz + nz * h, stage, J, t, params)[0]
        occ += (h - d) * sca
        sca *= 0.82
    return max(0.0, min(1.0, 1.0 - 2.1 * occ))


# ---------------------------------------------------------------------------
# Practical lights per stage (returns px, py, pz, r, g, b, intensity, radius)
# ---------------------------------------------------------------------------


@njit(inline="always", fastmath=True)
def _n_lights(params):
    return int(params[9])


@njit(inline="always", fastmath=True)
def _light_at(i, params):
    b = 10 + i * 8
    return (params[b], params[b + 1], params[b + 2],
            params[b + 3], params[b + 4], params[b + 5],
            params[b + 6], params[b + 7])


@njit(cache=True, fastmath=True)
def _emission(mat, px, py, pz, t, params):
    """Self-illumination (radiance) of emissive materials."""
    if mat == M_NEON:
        # identify the nearest sign to give it its practical colour
        best = 1e9
        r = 0.0
        g = 0.0
        b = 0.0
        for i in range(6):
            if i == 0:
                sx, sy, sz, cr, cg, cb = -2.86, 3.55, 6.2, 0.10, 0.90, 1.00
            elif i == 1:
                sx, sy, sz, cr, cg, cb = -2.86, 2.55, 8.6, 0.95, 0.18, 0.80
            elif i == 2:
                sx, sy, sz, cr, cg, cb = 2.86, 3.95, 5.4, 1.00, 0.14, 0.68
            elif i == 3:
                sx, sy, sz, cr, cg, cb = 2.86, 2.75, 10.4, 0.95, 0.55, 0.18
            elif i == 4:
                sx, sy, sz, cr, cg, cb = -2.86, 4.55, 13.2, 0.20, 0.85, 1.00
            else:
                sx, sy, sz, cr, cg, cb = 2.86, 4.35, 15.0, 1.00, 0.28, 0.62
            dd = (px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2
            if dd < best:
                best = dd
                r, g, b = cr, cg, cb
        flick = 0.80 + 0.20 * math.sin(t * 24.0) * math.sin(t * 2.3)
        return r * 5.5 * flick, g * 5.5 * flick, b * 5.5 * flick
    if mat == M_WINDOW or mat == M_GLASS:
        n = _fbm3(px * 2.0, py * 2.0, pz * 2.0, 97.0, 2)
        w = 0.35 + 0.85 * n
        warm = _hash21(px * 0.7, pz * 0.7, 5.0)
        if warm > 0.72:
            return 0.30 * w, 0.52 * w, 0.62 * w
        return 1.05 * w, 0.68 * w, 0.30 * w
    if mat == M_LANTERN:
        pulse = 0.86 + 0.14 * math.sin(t * 6.0 + px * 3.0)
        return 1.55 * pulse, 0.78 * pulse, 0.34 * pulse
    if mat == M_VENDING:
        rows = _mod(py * 7.0, 1.0)
        if py > 1.02 and rows < 0.68:
            return 0.10, 0.30, 0.34
        return 0.045, 0.050, 0.060
    if mat == M_HOLO:
        return 0.10 * params[6], 1.55 * params[6], 0.95 * params[6]
    if mat == M_SEED:
        return 0.0, 0.0, 0.0
    if mat == M_ROOT:
        return 0.0, 0.0, 0.0
    return 0.0, 0.0, 0.0


@njit(cache=True, fastmath=True)
def _steam_density(px, py, pz, t, params):
    """Volumetric density: grate steam column + general alley haze."""
    base = 0.011
    # steam column rising from the grate at (0.85, 0, 6.4)
    dx = px - 0.85
    dz = pz - 6.4
    rad = 0.30 + 0.16 * py
    r2 = dx * dx + dz * dz
    plume = math.exp(-r2 / (rad * rad)) * math.exp(-py * 0.55)
    # slow turbulent wobble
    wob = _vnoise3(px * 2.4, py * 1.6 - t * 0.55, pz * 2.4, 5.0)
    plume *= 0.45 + 0.85 * wob
    if py < 0.0:
        plume = 0.0
    return base + plume * 0.16 * params[3]


@njit(cache=True, fastmath=True)
def _render_pixel(ox, oy, oz, dx, dy, dz, stage, J, t, params):
    """March one primary ray; return (r, g, b, depth) in linear radiance."""
    tmax = 46.0
    tt = 0.02
    hit = False
    mat = 0
    prev_d = 1e9
    fog_acc = 0.0
    trans = 1.0
    fog_r = 0.0
    fog_g = 0.0
    fog_b = 0.0

    steps = 0
    max_steps = 46 if stage == 2 else 110
    while steps < max_steps:
        px = ox + dx * tt
        py = oy + dy * tt
        pz = oz + dz * tt
        d, m = _scene_map(px, py, pz, stage, J, t, params)
        if d < (0.0030 if stage == 2 else 0.0016 * tt + 0.0009):
            hit = True
            mat = m
            break
        step_min = 0.0026 if stage == 2 else 0.022
        # volumetric integration (single scattering, analytic per step)
        dens = _steam_density(px, py, pz, t, params)
        if dens > 1e-5:
            step_len = min(max(d, 0.03), 0.55)
            fog_acc += dens * step_len
            lr = 0.0
            lg = 0.0
            lb = 0.0
            for i in range(_n_lights(params)):
                lx, ly, lz, cr, cg, cb, it, rad = _light_at(i, params)
                ldx = lx - px
                ldy = ly - py
                ldz = lz - pz
                l2 = ldx * ldx + ldy * ldy + ldz * ldz
                inv = 1.0 / (1.0 + l2 * 1.15)
                lr += cr * it * inv
                lg += cg * it * inv
                lb += cb * it * inv
            fw = dens * step_len * math.exp(-fog_acc * 1.4)
            fog_r += lr * fw * 0.055
            fog_g += lg * fw * 0.055
            fog_b += lb * fw * 0.055
        prev_d = d
        tt += min(max(d * 0.92, step_min), 0.85)
        if tt > tmax:
            break
        steps += 1

    if not hit:
        r, g, b = _sky(dx, dy, dz, 1.0)
        # sky picks up a little of the alley's neon signature
        nr, ng, nb = _neon_env(dx, dy, dz, t)
        r += nr * 0.16
        g += ng * 0.16
        b += nb * 0.16
        return r + fog_r, g + fog_g, b + fog_b, 60.0

    px = ox + dx * tt
    py = oy + dy * tt
    pz = oz + dz * tt
    eps = max(0.0016, tt * 0.0012)
    nx, ny, nz = _calc_normal(px, py, pz, stage, J, t, params, eps)

    if mat == M_FAR:
        # distant towers: dark mass with sparse window lights, veiled in haze
        wu = _mod(px * 0.9, 1.55)
        wv = _mod(py * 0.9, 1.30)
        ar, ag, ab = 0.010, 0.012, 0.020
        er, eg, eb = 0.0, 0.0, 0.0
        if 0.30 < wu < 1.15 and 0.26 < wv < 0.95:
            lit = _hash21(math.floor(px * 0.9 / 1.55), math.floor(py * 0.9 / 1.30), 23.0)
            if lit > 0.72:
                warm = 0.4 + 0.6 * _hash21(lit * 91.0, 5.0, 27.0)
                er = 0.075 * warm
                eg = 0.052 * warm
                eb = 0.030 * warm
    elif mat == M_FACADE:
        side = 1.0 if px < 0.0 else -1.0
        ar, ag, ab, er, eg, eb, bump = _facade(px, py, pz, side, t)
        du = 0.035
        b1 = _facade(px, py, pz + du, side, t)[6]
        b2 = _facade(px, py + du, pz, side, t)[6]
        gz = (b1 - bump) * 0.085
        gy = (b2 - bump) * 0.085
        nx = side
        ny = -gy
        nz = -gz
        nl = math.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= nl
        ny /= nl
        nz /= nl
    else:
        ar, ag, ab = _albedo(mat, px, py, pz, t)
        er, eg, eb = _emission(mat, px, py, pz, t, params)

    # ambient occlusion + sky ambient
    ao = _ambient_occlusion(px, py, pz, nx, ny, nz, stage, J, t, params)
    sky_r, sky_g, sky_b = _sky(ny * 0.5 + 0.5, ny, 0.0, 0.6)
    env_r, env_g, env_b = _neon_env(nx, ny, nz, t)
    amb_r = (sky_r * 0.55 + env_r * 0.42) * ao
    amb_g = (sky_g * 0.55 + env_g * 0.42) * ao
    amb_b = (sky_b * 0.55 + env_b * 0.42) * ao

    lr = ar * amb_r
    lg = ag * amb_g
    lb = ab * amb_b

    # specular response
    shin = 22.0
    spec_k = 0.04
    if mat == M_WATER or mat == M_CHROME:
        shin = 240.0
        spec_k = 0.55
    elif mat == M_ASPHALT:
        shin = 70.0
        spec_k = 0.12
    elif mat == M_RAINCOAT:
        shin = 120.0
        spec_k = 0.16
    elif mat == M_LEATHER:
        shin = 40.0
        spec_k = 0.18
    elif mat == M_SKIN:
        shin = 26.0
        spec_k = 0.06
    elif mat == M_SEED:
        shin = 140.0
        spec_k = 0.45

    for i in range(_n_lights(params)):
        lx, ly, lz, cr, cg, cb, inten, rad = _light_at(i, params)
        if inten <= 0.0:
            continue
        ldx = lx - px
        ldy = ly - py
        ldz = lz - pz
        l2 = ldx * ldx + ldy * ldy + ldz * ldz
        ll = math.sqrt(l2) + 1e-6
        ldx /= ll
        ldy /= ll
        ldz /= ll
        ndl = nx * ldx + ny * ldy + nz * ldz
        if ndl <= 0.0:
            continue
        atten = inten / (1.0 + l2 * 1.05)
        if atten < 0.045:
            continue
        # only the strong practicals pay for a soft shadow trace
        if stage >= 1 and i > 1:
            shadow = 1.0
        elif atten > 0.30:
            shadow = _soft_shadow(px + nx * 0.004, py + ny * 0.004, pz + nz * 0.004,
                                  ldx, ldy, ldz, stage, J, t, params, min(ll, 14.0))
        else:
            shadow = 0.82
        if shadow <= 0.001:
            continue
        wrap = ndl
        if mat == M_SKIN or mat == M_LIP:
            wrap = min(1.0, (ndl + 0.28) / 1.28)
        dif = wrap * atten * shadow
        if mat == M_SKIN or mat == M_LIP:
            # cheap subsurface: warm bleed through the terminator
            ss = atten * shadow * 0.30 * (1.0 - ndl)
            lr += ar * cr * dif + cr * ss * 0.30
            lg += ag * cg * dif + cg * ss * 0.10
            lb += ab * cb * dif + cb * ss * 0.06
        else:
            lr += ar * cr * dif
            lg += ag * cg * dif
            lb += ab * cb * dif
        # Blinn-Phong specular
        hx = ldx - dx
        hy = ldy - dy
        hz = ldz - dz
        hl = math.sqrt(hx * hx + hy * hy + hz * hz) + 1e-6
        ndh = (nx * hx + ny * hy + nz * hz) / hl
        if ndh > 0.0:
            sp = spec_k * (ndh ** shin) * atten * shadow * 0.55
            lr += cr * sp
            lg += cg * sp
            lb += cb * sp

    # wet-surface environment reflection (this is the neon in the puddles)
    refl_k = 0.0
    if mat == M_ASPHALT:
        puddle = _fbm3(px * 1.35, py * 1.35, pz * 1.35, 131.0, 3)
        refl_k = 0.18 + 0.72 * max(0.0, min(1.0, (puddle - 0.42) * 5.0))
    elif mat == M_WATER:
        refl_k = 0.95
    elif mat == M_CHROME:
        refl_k = 0.55
    elif mat == M_RAINCOAT:
        refl_k = 0.30

    if refl_k > 0.01:
        rdx = dx - 2.0 * (dx * nx + dy * ny + dz * nz) * nx
        rdy = dy - 2.0 * (dx * nx + dy * ny + dz * nz) * ny
        rdz = dz - 2.0 * (dx * nx + dy * ny + dz * nz) * nz
        # ripple-perturbed reflection for wet asphalt
        if mat != M_CHROME:
            rp = _vnoise3(px * 9.0, py * 2.0, pz * 9.0 + t * 0.8, 151.0)
            rdx += (rp - 0.5) * 0.10
            rdz += (rp - 0.5) * 0.06
        fr, fg, fb = _neon_env(rdx, rdy, rdz, t)
        sr, sg, sb = _sky(rdx, rdy, rdz, 0.8)
        fres = 0.28 + 0.72 * (1.0 - max(0.0, min(1.0, -(dx * nx + dy * ny + dz * nz))))
        lr += (fr + sr) * refl_k * fres * 0.85
        lg += (fg + sg) * refl_k * fres * 0.85
        lb += (fb + sb) * refl_k * fres * 0.85

    # translucent raincoat: fresnel rim + faint body transmission
    if mat == M_RAINCOAT:
        fres = (1.0 - max(0.0, min(1.0, -(dx * nx + dy * ny + dz * nz)))) ** 2.4
        lr += 0.09 * fres
        lg += 0.115 * fres
        lb += 0.15 * fres
        lr *= 0.82
        lg *= 0.86
        lb *= 0.90

    # ---- the seed: a translucent bioluminescent membrane over a hot core ----
    if mat == M_SEED:
        inv = 1.0 / 0.0345
        nx_ = (px - 0.0) * inv
        ny_ = (py - 0.012) * inv
        nz_ = (pz - 0.0) * inv
        fres = (1.0 - max(0.0, min(1.0, -(dx * nx_ + dy * ny_ + dz * nz_)))) ** 2.2
        veins = _fbm3(px * 46.0, py * 46.0, pz * 46.0, 257.0, 4)
        vein = math.exp(-((veins - 0.52) ** 2) * 90.0)
        # organic mottling so the membrane never reads as flat noise
        mott = _fbm3(px * 120.0, py * 120.0, pz * 120.0, 263.0, 3)
        core = math.exp(-(px * px + (py - 0.012) ** 2 + pz * pz) * 2200.0)
        g = params[5] * 0.9
        shade = 0.75 + 0.50 * mott
        lr = (0.020 + 0.10 * vein + 0.55 * core + 0.16 * fres) * g * shade
        lg = (0.140 + 0.34 * vein + 1.10 * core + 0.58 * fres) * g * shade
        lb = (0.420 + 0.72 * vein + 1.65 * core + 1.05 * fres) * g * shade

    # ---- roots of light: hot at the seed, fading as they crawl out ----
    if mat == M_ROOT:
        rad = math.sqrt(px * px + pz * pz)
        fall = math.exp(-rad * 8.0)
        pulse = 0.55 + 0.45 * params[1]
        lr += (0.080 + 0.55 * fall) * params[5] * pulse
        lg += (0.330 + 1.70 * fall) * params[5] * pulse
        lb += (0.700 + 2.70 * fall) * params[5] * pulse

    # eyes: a deliberate, art-directed iris (no blown-out chrome balls)
    if mat == M_EYE:
        cyw = math.cos(-J[42])
        syw = math.sin(-J[42])
        ex = px - J[9]
        ez = pz - J[11]
        lx = ex * cyw - ez * syw
        ly = py - J[10]
        side = -1.0 if lx < 0.0 else 1.0
        radial = math.sqrt((lx - side * 0.0295) ** 2 + (ly - 0.0185) ** 2)
        # sclera: cool off-white, gently shaded by the key direction
        sclera = 0.105 + 0.055 * max(0.0, min(1.0, (lx * side + 0.01) * 8.0))
        lr, lg, lb = sclera * 1.02, sclera * 0.99, sclera * 0.96
        if radial < 0.0152:
            fibre = 0.5 + 0.5 * math.sin(math.atan2(ly - 0.0185, lx - side * 0.0295) * 26.0)
            lr = 0.058 + 0.034 * fibre
            lg = 0.036 + 0.020 * fibre
            lb = 0.019 + 0.010 * fibre
            # limbal ring darkens the iris edge
            edge = smoothstep_local(0.0116, 0.0152, radial)
            lr *= (1.0 - 0.65 * edge)
            lg *= (1.0 - 0.65 * edge)
            lb *= (1.0 - 0.65 * edge)
        if radial < 0.0062:
            lr, lg, lb = 0.010, 0.009, 0.009
        # single soft corneal catchlight, upper-left of the iris
        hl = math.exp(-(((lx - side * 0.0330) ** 2 + (ly - 0.0268) ** 2) * 240000.0))
        lr += hl * 0.85
        lg += hl * 0.92
        lb += hl * 1.00
    lr += er
    lg += eg
    lb += eb

    # distance fog extinction + accumulated in-scattering
    ext = math.exp(-fog_acc * 1.15 - tt * 0.020)
    lr = lr * ext + fog_r
    lg = lg * ext + fog_g
    lb = lb * ext + fog_b
    return lr, lg, lb, tt


@njit(cache=True, parallel=True, fastmath=True)
def render_frame(w, h, cam, stage, J, t, params):
    """
    Render one frame. `cam` is
    [px,py,pz, fx,fy,fz, rx,ry,rz, ux,uy,uz, focal_px, jitter_x, jitter_y]
    Returns (rgb float32 [h,w,3] linear, depth float32 [h,w]).
    """
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    depth = np.full((h, w), 60.0, dtype=np.float32)
    ox = cam[0]
    oy = cam[1]
    oz = cam[2]
    fx = cam[3]
    fy = cam[4]
    fz = cam[5]
    rx = cam[6]
    ry = cam[7]
    rz = cam[8]
    ux = cam[9]
    uy = cam[10]
    uz = cam[11]
    focal = cam[12]
    jx = cam[13]
    jy = cam[14]

    for y in prange(h):
        v = (h * 0.5 - (y + 0.5)) / focal + jy
        for x in range(w):
            u = ((x + 0.5) - w * 0.5) / focal + jx
            dx = fx + rx * u + ux * v
            dy = fy + ry * u + uy * v
            dz = fz + rz * u + uz * v
            inv = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            dx *= inv
            dy *= inv
            dz *= inv
            r, g, b, dep = _render_pixel(ox, oy, oz, dx, dy, dz, stage, J, t, params)
            rgb[y, x, 0] = r
            rgb[y, x, 1] = g
            rgb[y, x, 2] = b
            depth[y, x] = dep
    return rgb, depth
