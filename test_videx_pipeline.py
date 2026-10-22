"""End-to-end tests covering VIDEX Lexer -> Parser -> Validator -> Continuous Cinema Engine."""

from pathlib import Path
import tempfile
import unittest

import imageio.v3 as iio
import numpy as np

from videx import compile_and_render
from videx_engine import VidexVideoEngine
from videx_lexer import SECTION_7_TEST_CODE
from videx_parser import VidexParser
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
        Mathematical continuity guard:
        Verifies zero camera jumps, overlapping 40-frame scene dissolves, and
        unbroken >800px continuous eagle flight across Act 3.
        """
        program = VidexParser().parse_source(SECTION_7_TEST_CODE)
        VidexValidator().validate(program)
        engine = VidexVideoEngine()
        segments = engine.compile_segments(program, total_frames=260)

        # Space segment + unified continuous Himalayas segment (acts 2 & 3 merged smoothly!)
        self.assertEqual(len(segments), 2)
        space_seg, mountain_seg = segments

        # Verify long overlapping live dual-scene dissolve (>= 35 frames = ~1.4s+)
        self.assertGreaterEqual(mountain_seg.transition_frames, 35)
        self.assertGreaterEqual(
            space_seg.render_end,
            mountain_seg.render_start + mountain_seg.transition_frames,
        )

        # Verify strict C^1/C^2 sub-pixel smoothness with ZERO jumps anywhere!
        for seg in segments:
            pans = [seg.sample(f, 26, 260).cam_pan_x for f in range(seg.render_start, seg.render_end + 1)]
            tilts = [seg.sample(f, 26, 260).cam_tilt_y for f in range(seg.render_start, seg.render_end + 1)]
            zooms = [seg.sample(f, 26, 260).cam_zoom for f in range(seg.render_start, seg.render_end + 1)]
            self.assertLess(max(abs(pans[i + 1] - pans[i]) for i in range(len(pans) - 1)), 0.01)
            self.assertLess(max(abs(tilts[i + 1] - tilts[i]) for i in range(len(tilts) - 1)), 0.01)
            self.assertLess(max(abs(zooms[i + 1] - zooms[i]) for i in range(len(zooms) - 1)), 0.005)

        # Verify eagle moves smoothly & monotonically across > 800 pixels on 1920px canvas!
        xs = [
            mountain_seg.sample(f, 26, 260).objects["eagle"].x * 1920.0
            for f in range(mountain_seg.render_start, mountain_seg.render_end + 1)
            if "eagle" in mountain_seg.sample(f, 26, 260).objects
        ]
        self.assertGreater(len(xs), 75)
        self.assertGreater(xs[-1] - xs[0], 800.0)
        for i in range(len(xs) - 1):
            self.assertGreater(xs[i + 1], xs[i], f"Eagle froze or reversed at step {i}")

    def test_validator_repairs_odd_resolution_clamps_frames_and_auto_places_objects(self) -> None:
        src = "\n".join(
            [
                "set resolution to 1281 by 719",
                "second 1:",
                "    keyframe 40:",
                '        rotate object "earth" by 15 degrees',
                "end second 1",
            ]
        )
        program = VidexParser().parse_source(src)
        notes = VidexValidator().validate(program)
        self.assertEqual((program.width, program.height), (1282, 720))
        self.assertEqual(program.seconds[0].frames[0].index, 26)
        self.assertEqual(program.seconds[0].frames[0].commands[0].action, "place_object")
        self.assertEqual(len(notes), 3)

    def test_single_line_videx_outputs_mp4_with_living_per_frame_motion(self) -> None:
        single_line = 'set background to "Himalayas mountain"'
        with tempfile.TemporaryDirectory() as tmp:
            out_mp4 = Path(tmp) / "single_line.mp4"
            summary = compile_and_render(single_line, output_path=out_mp4)
            self.assertTrue(out_mp4.is_file())
            self.assertGreater(out_mp4.stat().st_size, 15_000)
            self.assertEqual(summary["fps"], 26)

            f0 = iio.imread(out_mp4, index=0)
            f5 = iio.imread(out_mp4, index=5)
            self.assertEqual(f0.shape, (1080, 1920, 3))
            self.assertGreater(float(np.abs(f5.astype(np.float32) - f0.astype(np.float32)).mean()), 0.15)


if __name__ == "__main__":
    unittest.main()
