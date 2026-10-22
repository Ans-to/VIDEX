"""
VIDEX Compiler & Video Engine — End-to-End CLI Runner
=====================================================

Executes the full VIDEX pipeline:
    VIDEX Code -> Lexer -> Parser -> Interpreter -> Video Engine -> MP4 Output
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import time

from videx_engine import VidexVideoEngine
from videx_lexer import SECTION_7_TEST_CODE, VidexLexer, VidexLexerError
from videx_parser import VidexParser, VidexParserError


def slugify_video_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "videx_output"


def compile_and_render(
    source: str,
    *,
    output_path: str | Path | None = None,
    base_dir: Path | None = None,
) -> dict[str, object]:
    """Run Lexer -> Parser -> Video Engine on ``source`` and write an `.mp4`."""
    t0 = time.perf_counter()

    # Stage 1: Lexer
    lexer = VidexLexer()
    tokens = lexer.tokenize(source)

    # Stage 2: Parser
    parser = VidexParser()
    program = parser.parse(tokens)

    # Destination path
    if output_path is None:
        filename = f"{slugify_video_name(program.name)}.mp4"
        dest = Path("output") / filename
    else:
        dest = Path(output_path)

    # Stage 3 & 4: Interpreter + Video Engine
    engine = VidexVideoEngine(base_dir=base_dir or Path.cwd())
    rendered_path = engine.render(program, dest)
    elapsed = time.perf_counter() - t0

    return {
        "video_name": program.name,
        "tokens_count": len(tokens),
        "seconds_blocks": len(program.seconds),
        "resolution": f"{program.width}x{program.height}",
        "fps": program.fps,
        "duration_seconds": program.total_duration,
        "total_frames": int(round(program.total_duration * program.fps)),
        "output_path": str(rendered_path),
        "render_time_sec": round(elapsed, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compile & render a VIDEX source file to MP4.")
    ap.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to a .videx source file (defaults to Section 7 'Earth from Space' example).",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .mp4 path (defaults to output/<video_name>.mp4).",
    )
    args = ap.parse_args()

    if args.input_file:
        src_path = Path(args.input_file)
        source = src_path.read_text(encoding="utf-8")
        base_dir = src_path.parent
    else:
        source = SECTION_7_TEST_CODE
        base_dir = Path.cwd()

    try:
        summary = compile_and_render(source, output_path=args.output, base_dir=base_dir)
    except (VidexLexerError, VidexParserError) as err:
        print(err)
        raise SystemExit(1) from err

    print("═" * 62)
    print("🎬 VIDEX PIPELINE COMPLETE")
    print("═" * 62)
    print(f"  Video Title   : {summary['video_name']}")
    print(f"  Tokens Lexed  : {summary['tokens_count']}")
    print(f"  Scene Seconds : {summary['seconds_blocks']} scripted blocks")
    print(f"  Resolution    : {summary['resolution']} @ {summary['fps']} fps")
    print(f"  Duration      : {summary['duration_seconds']}s ({summary['total_frames']} frames)")
    print(f"  Output MP4    : {summary['output_path']}")
    print(f"  Render Time   : {summary['render_time_sec']}s")
    print("═" * 62)


if __name__ == "__main__":
    main()
