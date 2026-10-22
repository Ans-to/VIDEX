# VIDEX — Cinema Video Generation Programming Language

**VIDEX** is a plain-English video generation programming language where source code compiles directly into cinema-grade `1920x1080` MP4 video output at **26 frames per second** (`26 fps` VIDEX standard).

```text
VIDEX Source Code ─▶ Lexer ─▶ Parser ─▶ Semantic Validator ─▶ 2.5D/3D Cinema Engine ─▶ MP4 Video
```

---

## Architecture & Pipeline Stages

| File | Stage | Responsibility |
| --- | --- | --- |
| [`videx_lexer.py`](videx_lexer.py) | **Stage 1 — Lexer** | Character-by-character scanner, greedy multi-word phrase grouper, context-aware classifier across all **20 VIDEX token types**, smart asset resolution, and fuzzy typo error suggestions (`VidexLexerError`). |
| [`videx_parser.py`](videx_parser.py) | **Stage 2 — Parser** | Builds a structured `VideoProgram` AST supporting timelines, keyframes, manual frames, `repeat N times` loops, camera motion/shake, lighting, object transforms, typography, full visual post-FX, and audio. |
| [`videx_validator.py`](videx_validator.py) | **Stage 2.5 — Validator** | Enforces H.264 `yuv420p` even-resolution constraints, clamps out-of-bounds frame indices, merges duplicate keyframes, and auto-instantiates implicitly referenced objects per VIDEX's self-healing philosophy. |
| [`videx_engine.py`](videx_engine.py) | **Stages 3 & 4 — Cinema Engine** | Sub-pixel `float32` `cv2.remap` 2.5D depth-parallax camera, 5-layer ray-shaded 3D Planet Earth, non-rigid per-frame wing-beat mesh deformation, volumetric sun rays & valley mist, ACES filmic tone mapping, and stereo AAC audio synthesis. |
| [`videx.py`](videx.py) | **CLI Driver** | End-to-end compiler & renderer with optional `--hud` frame-telemetry burn-in. |
| [`assets/`](assets/) | **Studio Textures** | High-resolution seamless equirectangular Earth map, deep-field galaxy nebula, Himalayan sunrise plate, and isolated golden eagle sprite (with full procedural fallbacks when offline). |
| [`examples/earth_from_space.videx`](examples/earth_from_space.videx) | **Example Script** | Complete `"Earth from Space"` multi-scene program. |
| [`output/earth_from_space.mp4`](output/earth_from_space.mp4) | **Rendered Video** | `1920x1080 @ 26fps` H.264 + stereo AAC output (`260` frames, zero frozen frames). |

---

## Continuous Motion & Cinema Engine Architecture

1. **Continuous-Time $C^2$ Motion Compiler (`compile_segments`, `ScalarTrack`, `ObjectTrack`)** — Proportional act mapping spans the full declared video duration (`set total duration to ...`) without dead static tails. Consecutive `second` blocks on the same background merge automatically into one continuous shot. `place object` anchors the entry keyframe while `move object`, `rotate object`, `scale object`, `camera pan`, and `camera tilt` evaluate smooth quintic/hermite curves across time with zero boundary jumps or frozen frames.
2. **Dual-Stream Linear-Light Overlapping Transitions (`_blend_transition_linear_light`)** — Scene transitions (`fade`, `wipe`, `slide`) render **both** the outgoing and incoming scenes live simultaneously across a ~1.5-second overlap window and blend in $\gamma = 2.2$ linear light with gentle push-zoom motion continuity.
3. **Sub-Pixel Floating-Point Camera, Sprite Compositing & 2.5D Depth Parallax (`cv2.remap`)** — Eliminates 1px integer quantization jitter across backgrounds, sprites, and advecting valley mist.
4. **5-Layer 3D Planet Ray-Tracer** — Sub-pixel bilinear continents/oceans (`BORDER_WRAP`), ocean-masked Blinn-Phong solar glint, independent super-rotating 3D cloud deck casting surface shadows, night-side golden city lights, and dual-scale Rayleigh blue atmospheric rim bloom.
5. **Organic Aerodynamic Wing-Beat & Thermal Soar** — Smooth `0.88 Hz` power strokes that damp gracefully into an open-wing thermal glide across the summit ridge with sub-pixel banking and soft shadow.
6. **ACES Filmic Color Science & Anti-Banding Dither** — Smooth highlight roll-off, fade-in/fade-out bookends, optical vignette, and 10-bit-style chroma dither.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Run the standalone VIDEX Lexer (zero external dependencies)
python3 videx_lexer.py

# 2. Compile & render a VIDEX program to 1080p MP4
python3 videx.py examples/earth_from_space.videx -o output/earth_from_space.mp4

# 3. Render with live frame-by-frame camera/timeline telemetry HUD burned in
python3 videx.py examples/earth_from_space.videx -o output/debug_hud.mp4 --hud

# 4. Run full automated test suite (12 unit + integration tests)
python3 -m unittest discover -v
```
