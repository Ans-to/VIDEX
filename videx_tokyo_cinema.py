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



def _native_film_module():
    """Lazy import so the soundtrack stays importable on its own."""
    import videx_native_film

    return videx_native_film


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
    n = min(n, max(cut_idx + 1, n))
    cut_idx = min(cut_idx, n)

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
    assets_dir: Path | None = None,
    width: int = 3840,
    height: int = 2160,
    fps: int = 24,
    duration_sec: float = 15.0,
) -> Path:
    """
    Render "Tokyo Seed Courier" with the in-repo native software renderer.

    `assets_dir` is accepted for API compatibility and is deliberately unused:
    the film is 100% synthesised (SDF ray marching + procedural audio), so
    there are no photographs, sprites, textures or downloaded plates anywhere
    in the frame pipeline.
    """
    film = _native_film_module()

    # 2.39:1 anamorphic; pick the height that matches the requested width
    if abs(width / height - film.ASPECT) > 0.02:
        height = int(round(width / film.ASPECT))
    height -= height % 2
    width -= width % 2

    return film.render_native_film(
        output_path,
        width=width,
        height=height,
        fps=fps,
        duration=duration_sec,
    )
