"""End-to-end tests covering VIDEX Lexer -> Parser -> Video Engine -> MP4."""

from pathlib import Path
import tempfile
import unittest

import imageio.v3 as iio

from videx import compile_and_render
from videx_lexer import SECTION_7_TEST_CODE
from videx_parser import VidexParser


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

    def test_single_line_videx_outputs_mp4_automatically(self) -> None:
        single_line = 'set background to "Himalayas mountain"'
        with tempfile.TemporaryDirectory() as tmp:
            out_mp4 = Path(tmp) / "single_line.mp4"
            summary = compile_and_render(single_line, output_path=out_mp4)
            self.assertTrue(out_mp4.is_file())
            self.assertGreater(out_mp4.stat().st_size, 10_000)
            self.assertEqual(summary["fps"], 26)

            meta = iio.immeta(out_mp4, plugin="pyav") if False else None
            first_frame = iio.imread(out_mp4, index=0)
            self.assertEqual(first_frame.shape, (1080, 1920, 3))
            self.assertGreater(float(first_frame.mean()), 15.0)


if __name__ == "__main__":
    unittest.main()
