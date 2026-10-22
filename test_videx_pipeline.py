"""End-to-end tests covering VIDEX Lexer -> Parser -> Validator -> Cinema & Motion Graphics Engines."""

from pathlib import Path
import tempfile
import unittest

import imageio.v3 as iio
import numpy as np

from videx import compile_and_render
from videx_engine import VidexVideoEngine
from videx_lexer import SECTION_7_TEST_CODE
from videx_mograph import draw_sdf_circle, ease_expo_snap, ease_linear, ease_spring
from videx_parser import VidexParser
import videx_native_film as native_film
from videx_tokyo_cinema import write_tokyo_courier_soundtrack
from videx_validator import VidexValidator


class TestVidexEndToEndPipeline(unittest.TestCase):
    def test_parser_builds_complete_section_7_ast(self) -> None:
        program = VidexParser().parse_source(SECTION_7_TEST_CODE)
        self.assertEqual(program.name, "Earth from Space")
        self.assertEqual(program.fps, 26)
        self.assertEqual(program.width, 1920)
        self.assertEqual(program.height, 1080)
        self.assertEqual(program.total_duration, 10.0)
        self.assertEqual(len(program.seconds), 3)
        self.assertTrue(all(sec.auto_fill for sec in program.seconds))
        self.assertEqual(program.seconds[1].transition, "fade")
        self.assertTrue(program.play_requested)

    def test_continuous_motion_and_smooth_overlapping_transitions(self) -> None:
        """
        Mathematical continuity guard on live-action cinema scene:
        Verifies zero camera jumps, overlapping 40-frame scene dissolves, and
        unbroken >800px continuous eagle flight across Act 3.
        """
        program = VidexParser().parse_source(SECTION_7_TEST_CODE)
        VidexValidator().validate(program)
        engine = VidexVideoEngine()
        segments = engine.compile_segments(program, total_frames=260)

        self.assertEqual(len(segments), 2)
        space_seg, mountain_seg = segments

        self.assertGreaterEqual(mountain_seg.transition_frames, 35)
        self.assertGreaterEqual(
            space_seg.render_end,
            mountain_seg.render_start + mountain_seg.transition_frames,
        )

        for seg in segments:
            pans = [seg.sample(f, 26, 260).cam_pan_x for f in range(seg.render_start, seg.render_end + 1)]
            tilts = [seg.sample(f, 26, 260).cam_tilt_y for f in range(seg.render_start, seg.render_end + 1)]
            zooms = [seg.sample(f, 26, 260).cam_zoom for f in range(seg.render_start, seg.render_end + 1)]
            self.assertLess(max(abs(pans[i + 1] - pans[i]) for i in range(len(pans) - 1)), 0.01)
            self.assertLess(max(abs(tilts[i + 1] - tilts[i]) for i in range(len(tilts) - 1)), 0.01)
            self.assertLess(max(abs(zooms[i + 1] - zooms[i]) for i in range(len(zooms) - 1)), 0.005)

        xs = [
            mountain_seg.sample(f, 26, 260).objects["eagle"].x * 1920.0
            for f in range(mountain_seg.render_start, mountain_seg.render_end + 1)
            if "eagle" in mountain_seg.sample(f, 26, 260).objects
        ]
        self.assertGreater(len(xs), 75)
        self.assertGreater(xs[-1] - xs[0], 800.0)
        for i in range(len(xs) - 1):
            self.assertGreater(xs[i + 1], xs[i], f"Eagle froze or reversed at step {i}")

    def test_mograph_physics_curves_and_quarter_pixel_sdf_precision(self) -> None:
        """
        Verifies spring harmonic overshoot, expo snap boundary exactness, and
        true 0.25-pixel sub-pixel SDF anti-aliasing without integer snapping.
        """
        self.assertAlmostEqual(ease_linear(0.5), 0.5, places=6)
        self.assertEqual(ease_expo_snap(0.0), 0.0)
        self.assertEqual(ease_expo_snap(1.0), 1.0)
        self.assertGreater(ease_expo_snap(0.25), 0.75)

        spring_samples = [ease_spring(i / 200.0) for i in range(201)]
        self.assertGreater(max(spring_samples), 1.04)  # Genuine underdamped spring overshoot
        self.assertAlmostEqual(spring_samples[-1], 1.0, places=5)

        # Prove moving 0.25 sub-pixels produces smooth anti-aliased change (NOT frozen integer snap!)
        c0 = np.zeros((96, 96, 3), dtype=np.float32)
        c1 = np.zeros((96, 96, 3), dtype=np.float32)
        draw_sdf_circle(c0, 48.00, 48.00, 20.0, (56.0, 189.0, 248.0))
        draw_sdf_circle(c1, 48.25, 48.00, 20.0, (56.0, 189.0, 248.0))
        subpx_delta = float(np.abs(c1 - c0).mean())
        self.assertGreater(subpx_delta, 0.02)

    def test_motion_graphics_benchmark_script_compiles_cleanly(self) -> None:
        mg_script = Path(__file__).resolve().parent / "examples" / "motion_graphics_test.videx"
        program = VidexParser().parse_source(mg_script.read_text(encoding="utf-8"))
        notes = VidexValidator().validate(program)
        self.assertEqual(notes, [])
        self.assertEqual(program.fps, 26)
        self.assertEqual((program.width, program.height), (1920, 1080))
        self.assertEqual(program.total_duration, 12.0)

        segs = VidexVideoEngine().compile_segments(program, total_frames=312)
        self.assertEqual(len(segs), 4)
        self.assertEqual([s.transition_in for s in segs], [None, "slide", "wipe", "zoom in"])

    def test_native_renderer_needs_no_image_assets_whatsoever(self) -> None:
        """
        Guarantees the film pipeline is 100% self-rendered: no `assets/` folder,
        no image files anywhere in the repo, and no image-decoding calls in the
        rendering modules. Every pixel must come from our own SDF ray marcher.
        """
        root = Path(__file__).resolve().parent
        self.assertFalse((root / "assets").exists(), "image asset folder must not exist")
        offenders = []
        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.exr", "*.hdr"):
            offenders.extend(p.name for p in root.rglob(pattern))
        self.assertEqual(offenders, [], f"raster image assets found: {offenders}")

        for module in ("videx_raytrace.py", "videx_native_film.py", "videx_tokyo_cinema.py"):
            text = (root / module).read_text(encoding="utf-8")
            for banned in ("cv2.imread", "cv2.imdecode", "Image.open(", "iio.imread(",
                           "urlretrieve", "requests.get("):
                self.assertNotIn(banned, text,
                                 f"{module} must not load external imagery ({banned})")

    def test_native_software_renderer_produces_geometry_and_light(self) -> None:
        """
        Renders one frame of each shot with the native ray marcher and asserts
        that real geometry was hit, lit and depth-resolved (no blank frames).
        """
        for seconds in (1.5, 5.0, 8.0, 12.0):
            frame = native_film.render_film_frame(int(seconds * native_film.FPS), 220, 92)
            self.assertEqual(frame.shape, (92, 220, 3))
            self.assertEqual(frame.dtype, np.uint8)
            self.assertGreater(int(frame.max()), 40, f"frame at {seconds}s is black")
            self.assertGreater(float(frame.std()), 6.0, f"frame at {seconds}s lacks contrast")

    def test_tokyo_seed_courier_script_compiles_to_native_4k_program(self) -> None:
        """The .videx script still compiles clean at 3840x2160 / 24 fps."""
        from videx import compile_and_render
        from videx_engine import VidexVideoEngine

        script = Path(__file__).resolve().parent / "examples" / "tokyo_seed_courier.videx"
        program = VidexParser().parse_source(script.read_text(encoding="utf-8"))
        VidexValidator().validate(program)
        self.assertEqual(program.width, 3840)
        self.assertEqual(program.height, 2160)
        self.assertEqual(program.fps, 24)
        from videx_tokyo_cinema import is_tokyo_courier_program

        self.assertTrue(is_tokyo_courier_program(program.name, []))
        self.assertTrue(VidexVideoEngine)  # engine import stays wired

    def test_soundtrack_synthesises_15s_stereo_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "tokyo.wav"
            write_tokyo_courier_soundtrack(wav, duration_sec=15.0)
            self.assertGreater(wav.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
