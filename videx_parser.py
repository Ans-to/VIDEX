"""
VIDEX Programming Language — Parser (Stage 2)
==============================================

Pipeline Stage 2:
    VIDEX Source Code -> Lexer -> [PARSER] -> Interpreter / Video Engine -> MP4

Consumes the structured token stream produced by ``VidexLexer`` and constructs
a typed Abstract Syntax Tree (AST) representing video metadata, timeline second
blocks, manual frame / keyframe blocks, transitions, and visual/audio commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from videx_lexer import Token, TokenType, VidexLexer


# ══════════════════════════════════════════════════════════════════════════════
# AST NODE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class Command:
    """Single executable scene instruction inside a frame or keyframe."""

    action: str  # e.g. "set_background", "place_object", "scale_object", ...
    params: dict[str, object] = field(default_factory=dict)
    line: int = 0


@dataclass
class FrameBlock:
    """Represents either `frame N:` or `keyframe N:` inside a `second` block."""

    index: int
    is_keyframe: bool
    commands: list[Command] = field(default_factory=list)
    line: int = 0


@dataclass
class SecondBlock:
    """Represents a `second N: ... end second N` timeline section."""

    second_number: int
    transition: str | None = None
    frames: list[FrameBlock] = field(default_factory=list)
    auto_fill: bool = False
    line: int = 0


@dataclass
class VideoProgram:
    """Root AST node for a complete VIDEX program."""

    name: str = "Untitled VIDEX Video"
    fps: int = 26
    width: int = 1920
    height: int = 1080
    total_duration: float = 3.0
    duration_explicit: bool = False
    seconds: list[SecondBlock] = field(default_factory=list)
    global_commands: list[Command] = field(default_factory=list)
    play_requested: bool = False


class VidexParserError(Exception):
    """Syntax error raised when token sequence violates VIDEX grammar."""

    def __init__(self, line: int, detail: str) -> None:
        self.line = line
        super().__init__(f"❌ VIDEX PARSER ERROR\nLine {line}: {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# VIDEX PARSER
# ══════════════════════════════════════════════════════════════════════════════


class VidexParser:
    """Line-oriented recursive-descent Parser for VIDEX token streams."""

    def parse_source(self, source: str) -> VideoProgram:
        tokens = VidexLexer(skip_comments=True).tokenize(source)
        return self.parse(tokens)

    def parse(self, tokens: Sequence[Token]) -> VideoProgram:
        lines = self._group_logical_lines(tokens)
        program = VideoProgram()

        current_second: SecondBlock | None = None
        current_frame: FrameBlock | None = None

        for stmt in lines:
            head = stmt[0]
            words = [t.value for t in stmt]

            # 1. `play video` / `end`
            if words == ["play", "video"]:
                program.play_requested = True
                continue

            if words == ["end"]:
                current_frame = None
                current_second = None
                continue

            # 2. `end second N`
            if len(stmt) >= 2 and words[0] == "end" and words[1] == "second":
                current_frame = None
                current_second = None
                continue

            # 3. `second N:`
            if (
                len(stmt) >= 3
                and words[0] == "second"
                and stmt[1].type == TokenType.NUMBER
                and stmt[-1].type == TokenType.SEPARATOR
            ):
                sec_num = int(float(stmt[1].value))
                current_second = SecondBlock(second_number=sec_num, line=head.line)
                program.seconds.append(current_second)
                current_frame = None
                continue

            # 4. `keyframe N:` or `frame N:`
            if (
                len(stmt) >= 3
                and words[0] in ("keyframe", "frame")
                and stmt[1].type == TokenType.NUMBER
                and stmt[-1].type == TokenType.SEPARATOR
            ):
                if current_second is None:
                    current_second = SecondBlock(second_number=1, line=head.line)
                    program.seconds.append(current_second)
                frame_num = int(float(stmt[1].value))
                current_frame = FrameBlock(
                    index=frame_num,
                    is_keyframe=(words[0] == "keyframe"),
                    line=head.line,
                )
                current_second.frames.append(current_frame)
                continue

            # 5. `fill frames automatically`
            if words[:2] == ["fill", "frames"]:
                if current_second is not None:
                    current_second.auto_fill = True
                continue

            # 6. `transition with <type>`
            if len(stmt) >= 3 and words[:2] == ["transition", "with"]:
                trans_type = " ".join(words[2:])
                if current_second is not None:
                    current_second.transition = trans_type
                continue

            # 7. Top-level video configuration commands (`set video name ...`, etc.)
            if self._try_parse_header_setting(stmt, program):
                continue

            # 8. Scene / object / text / light / camera / sound commands
            cmd = self._parse_command(stmt)
            if current_frame is not None:
                current_frame.commands.append(cmd)
            elif current_second is not None:
                # Command directly inside `second N:` applies at frame 1
                default_frame = FrameBlock(index=1, is_keyframe=True, line=head.line)
                default_frame.commands.append(cmd)
                current_second.frames.append(default_frame)
                current_frame = default_frame
            else:
                # Single-line / blockless VIDEX scripts work automatically too!
                program.global_commands.append(cmd)

        if not program.duration_explicit:
            program.total_duration = float(max(len(program.seconds), 1))

        return program

    @staticmethod
    def _group_logical_lines(tokens: Sequence[Token]) -> list[list[Token]]:
        grouped: list[list[Token]] = []
        current: list[Token] = []

        for tok in tokens:
            if tok.type in (TokenType.COMMENT, TokenType.INDENT):
                continue
            if tok.type == TokenType.NEWLINE:
                if current:
                    grouped.append(current)
                    current = []
                continue
            if tok.type == TokenType.EOF:
                if current:
                    grouped.append(current)
                    current = []
                break
            current.append(tok)

        if current:
            grouped.append(current)
        return grouped

    @staticmethod
    def _try_parse_header_setting(stmt: Sequence[Token], program: VideoProgram) -> bool:
        words = [t.value for t in stmt]

        # set video name to "..."
        if len(stmt) >= 5 and words[:4] == ["set", "video", "name", "to"]:
            program.name = stmt[4].value
            return True

        # set frame rate to 26 [fps]
        if len(stmt) >= 5 and words[:4] == ["set", "frame", "rate", "to"]:
            program.fps = max(1, int(float(stmt[4].value)))
            return True

        # set resolution to 1920 by 1080
        if len(stmt) >= 6 and words[:3] == ["set", "resolution", "to"] and words[4] == "by":
            program.width = int(float(stmt[3].value))
            program.height = int(float(stmt[5].value))
            return True

        # set total duration to X [seconds]
        if len(stmt) >= 5 and words[:4] == ["set", "total", "duration", "to"]:
            program.total_duration = max(0.1, float(stmt[4].value))
            program.duration_explicit = True
            return True

        return False

    @staticmethod
    def _parse_command(stmt: Sequence[Token]) -> Command:
        words = [t.value for t in stmt]
        line = stmt[0].line

        # Background commands
        if len(stmt) >= 4 and words[:3] == ["set", "background", "to"]:
            file_path = None
            if "from" in words:
                idx = words.index("from")
                if idx + 1 < len(stmt):
                    file_path = stmt[idx + 1].value
            return Command(
                "set_background",
                {"theme": stmt[3].value, "file": file_path},
                line=line,
            )

        if len(stmt) >= 5 and words[:3] == ["set", "background", "time"] and words[3] == "to":
            return Command("set_background_time", {"time": stmt[4].value}, line=line)

        if len(stmt) >= 5 and words[:3] == ["set", "background", "weather"] and words[3] == "to":
            return Command("set_background_weather", {"weather": stmt[4].value}, line=line)

        if len(stmt) >= 5 and words[:3] == ["set", "background", "season"] and words[3] == "to":
            return Command("set_background_season", {"season": stmt[4].value}, line=line)

        # Object commands
        if len(stmt) >= 3 and words[:2] == ["place", "object"]:
            name = stmt[2].value
            pos = "center"
            file_path = None
            if "from" in words:
                f_idx = words.index("from")
                if f_idx + 1 < len(stmt):
                    file_path = stmt[f_idx + 1].value
            if "at" in words:
                at_idx = words.index("at")
                if at_idx + 1 < len(stmt):
                    pos = stmt[at_idx + 1].value
            return Command(
                "place_object",
                {"name": name, "position": pos, "file": file_path},
                line=line,
            )

        if len(stmt) >= 5 and words[:2] == ["scale", "object"] and words[3] == "to":
            return Command(
                "scale_object",
                {"name": stmt[2].value, "size": stmt[4].value},
                line=line,
            )

        if len(stmt) >= 5 and words[:2] == ["rotate", "object"] and words[3] == "by":
            return Command(
                "rotate_object",
                {"name": stmt[2].value, "degrees": float(stmt[4].value)},
                line=line,
            )

        if len(stmt) >= 5 and words[:2] == ["move", "object"] and words[3] == "to":
            pixels = None
            if "by" in words:
                by_idx = words.index("by")
                if by_idx + 1 < len(stmt):
                    pixels = float(stmt[by_idx + 1].value)
            return Command(
                "move_object",
                {"name": stmt[2].value, "position": stmt[4].value, "pixels": pixels},
                line=line,
            )

        if (
            len(stmt) >= 6
            and words[:2] == ["set", "object"]
            and words[3] == "opacity"
            and words[4] == "to"
        ):
            return Command(
                "set_object_opacity",
                {"name": stmt[2].value, "opacity": float(stmt[5].value) / 100.0},
                line=line,
            )

        # Light commands
        if len(stmt) >= 5 and words[:4] == ["set", "light", "color", "to"]:
            return Command("set_light_color", {"color": stmt[4].value}, line=line)

        if len(stmt) >= 5 and words[:4] == ["set", "light", "brightness", "to"]:
            return Command(
                "set_light_brightness",
                {"brightness": float(stmt[4].value) / 100.0},
                line=line,
            )

        # Camera commands
        if len(stmt) >= 3 and words[0] == "camera" and words[1] in ("pan", "tilt"):
            speed = "medium"
            if "speed" in words:
                s_idx = words.index("speed")
                if s_idx + 1 < len(stmt):
                    speed = stmt[s_idx + 1].value
            return Command(
                f"camera_{words[1]}",
                {"direction": stmt[2].value, "speed": speed},
                line=line,
            )

        if len(stmt) >= 5 and words[:4] == ["set", "camera", "angle", "to"]:
            return Command("set_camera_angle", {"angle": stmt[4].value}, line=line)

        if len(stmt) >= 5 and words[:4] == ["set", "camera", "zoom", "to"]:
            return Command("set_camera_zoom", {"zoom": stmt[4].value}, line=line)

        # Text commands
        if len(stmt) >= 3 and words[:2] == ["place", "text"]:
            pos = "center"
            if "at" in words:
                at_idx = words.index("at")
                if at_idx + 1 < len(stmt):
                    pos = stmt[at_idx + 1].value
            return Command(
                "place_text",
                {"content": stmt[2].value, "position": pos},
                line=line,
            )

        if len(stmt) >= 5 and words[:4] == ["set", "text", "color", "to"]:
            return Command("set_text_color", {"color": stmt[4].value}, line=line)

        if len(stmt) >= 5 and words[:4] == ["set", "text", "size", "to"]:
            return Command("set_text_size", {"size": stmt[4].value}, line=line)

        if len(stmt) >= 5 and words[:3] == ["fade", "text", "in"]:
            frames = 10
            if "over" in words:
                o_idx = words.index("over")
                if o_idx + 1 < len(stmt):
                    frames = int(float(stmt[o_idx + 1].value))
            return Command("fade_text_in", {"frames": frames}, line=line)

        if len(stmt) >= 3 and words[:2] == ["add", "text"]:
            color = "white"
            if "color" in words:
                c_idx = words.index("color")
                if c_idx + 1 < len(stmt):
                    color = stmt[c_idx + 1].value
            return Command(
                "add_text_effect",
                {"effect": stmt[2].value, "color": color},
                line=line,
            )

        # Visual effects on objects or scene
        if len(stmt) >= 3 and words[:2] == ["add", "effect"]:
            effect_name = stmt[2].value
            target_obj = None
            color = "white"
            if "object" in words:
                o_idx = words.index("object")
                if o_idx + 1 < len(stmt):
                    target_obj = stmt[o_idx + 1].value
            if "color" in words:
                c_idx = words.index("color")
                if c_idx + 1 < len(stmt):
                    color = stmt[c_idx + 1].value
            return Command(
                "add_effect",
                {"effect": effect_name, "object": target_obj, "color": color},
                line=line,
            )

        # Narration & sound
        if len(stmt) >= 2 and words[0] == "narrate":
            return Command("narrate", {"text": stmt[1].value}, line=line)

        if len(stmt) >= 4 and words[:3] == ["play", "sound", "from"]:
            return Command("play_sound", {"file": stmt[3].value}, line=line)

        return Command("generic", {"tokens": [t.to_dict() for t in stmt]}, line=line)
