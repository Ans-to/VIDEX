# VIDEX — Video Generation Programming Language

**VIDEX** is a plain-English video generation programming language where source code compiles directly into rendered `1920x1080` MP4 video output at **26 frames per second** (`26 fps` VIDEX standard) with zero external authoring tools required.

```text
VIDEX Source Code ──▶ Lexer ──▶ Parser ──▶ Keyframe Interpreter ──▶ Video Engine ──▶ MP4 Video
```

---

## Core Language Philosophy

- **Everything is a Frame** (`1 second = 26 frames` standard)
- **Every Line is a Scene** — even a single line of VIDEX produces a playable `.mp4`
- **Code IS the Script** — human-readable plain-English statements
- **Smart Asset Resolution** — procedural generation by default, or direct file imports via `from "..."`

---

## Project Structure

| File | Role |
| --- | --- |
| [`videx_lexer.py`](videx_lexer.py) | **Stage 1 — Lexer**: Character-by-character scanner, phrase grouper, context-sensitive classifier across all 20 VIDEX token types, and typo-suggesting error handler (`VidexLexerError`). |
| [`videx_parser.py`](videx_parser.py) | **Stage 2 — Parser**: Builds a structured `VideoProgram` AST (`SecondBlock`, `FrameBlock`, `Command`). |
| [`videx_engine.py`](videx_engine.py) | **Stages 3 & 4 — Interpreter & Video Engine**: Keyframe tweening (`fill frames automatically`), procedural 1080p scene synthesis, stereo ambient soundtrack generation, and H.264/AAC `.mp4` encoding. |
| [`videx.py`](videx.py) | **CLI Compiler & Renderer**: Compiles any `.videx` file directly into an `.mp4` video. |
| [`examples/earth_from_space.videx`](examples/earth_from_space.videx) | Official `"Earth from Space"` multi-scene example script. |
| [`output/earth_from_space.mp4`](output/earth_from_space.mp4) | Rendered `1920x1080 @ 26fps` H.264 + AAC output of `earth_from_space.videx`. |
| [`test_videx_lexer.py`](test_videx_lexer.py) | Unit tests for all 20 token types, all 13 keyword categories, smart assets, compact units, and lexer errors. |
| [`test_videx_pipeline.py`](test_videx_pipeline.py) | End-to-end integration tests for `Lexer -> Parser -> Video Engine -> MP4`. |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Note: `videx_lexer.py` uses only the Python standard library and runs with zero external dependencies.)*

### 2. Run the VIDEX Lexer

```bash
# Tokenize the built-in "Earth from Space" example (309 tokens)
python3 videx_lexer.py

# Tokenize any custom .videx file
python3 videx_lexer.py examples/earth_from_space.videx
```

Example token output:

```js
{
  type: "KEYWORD",
  value: "set",
  line: 1,
  position: 0
}

{
  type: "KEYWORD",
  value: "background",
  line: 1,
  position: 4
}

{
  type: "KEYWORD",
  value: "to",
  line: 1,
  position: 15
}

{
  type: "VALUE",
  value: "Himalayas mountain",
  line: 1,
  position: 18
}
```

### 3. Compile & Render a Video (`.mp4`)

```bash
python3 videx.py examples/earth_from_space.videx -o output/earth_from_space.mp4
```

### 4. Run Automated Tests

```bash
python3 -m unittest discover -v
```

---

## Supported Token Types (All 20 Types)

`KEYWORD` · `VALUE` · `NUMBER` · `UNIT` · `COLOR` · `POSITION` · `SIZE` · `DIRECTION` · `EFFECT` · `FILE` · `SPEED` · `ANGLE` · `TIME_OF_DAY` · `WEATHER` · `SEASON` · `SEPARATOR` · `COMMENT` · `NEWLINE` · `INDENT` · `EOF`

---

## Example VIDEX Program

```videx
// Earth from Space - VIDEX Test File
set video name to "Earth from Space"
set frame rate to 26
set resolution to 1920 by 1080
set total duration to 10 seconds

second 1:
    keyframe 1:
        set background to "space galaxy"
        place object "earth" at center
        scale object "earth" to large
        set light color to cold
        set light brightness to 60 percent

    keyframe 26:
        rotate object "earth" by 5 degrees
        add effect glow to object "earth" color blue
        camera pan left speed slow

    fill frames automatically
end second 1
```
