"""
VIDEX Programming Language — Lexer (Tokenizer)
==============================================

Pipeline Stage 1:
    VIDEX Source Code -> [LEXER] -> Parser -> Interpreter -> Video Engine -> MP4

Reads VIDEX plain-English source code character by character, groups multi-word
phrases where appropriate, classifies every lexeme into one of the 20 VIDEX
token types with exact 1-based line numbers and 0-based column positions, and
emits structured tokens ready for consumption by the VIDEX Parser.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import Enum
import json
from typing import Iterator, Sequence


# ══════════════════════════════════════════════════════════════════════════════
# 1. TOKEN TYPES (All 20 VIDEX Token Types — Section 4)
# ══════════════════════════════════════════════════════════════════════════════


class TokenType(str, Enum):
    """Enumeration of all 20 token types recognized by the VIDEX Lexer."""

    KEYWORD = "KEYWORD"         # 1.  Reserved words of the VIDEX language
    VALUE = "VALUE"             # 2.  Text values (quoted or bare asset descriptors)
    NUMBER = "NUMBER"           # 3.  Numeric literals (integers and decimals)
    UNIT = "UNIT"               # 4.  Measurement units (seconds, frames, px, fps, ...)
    COLOR = "COLOR"             # 5.  Named colors or hex color codes (#FF5733)
    POSITION = "POSITION"       # 6.  Spatial screen positions (center, top left, ...)
    SIZE = "SIZE"               # 7.  Relative sizes (small, medium, large)
    DIRECTION = "DIRECTION"     # 8.  Motion/camera directions (up, down, left, ...)
    EFFECT = "EFFECT"           # 9.  Visual effects (glow, blur, lens flare, ...)
    FILE = "FILE"               # 10. Asset filenames (.png, .jpg, .jpeg, .mp4, .mp3, .wav)
    SPEED = "SPEED"             # 11. Motion/playback speeds (slow, medium, fast)
    ANGLE = "ANGLE"             # 12. Camera angles/framings (drone view, wide shot, ...)
    TIME_OF_DAY = "TIME_OF_DAY" # 13. Lighting times of day (sunrise, golden hour, ...)
    WEATHER = "WEATHER"         # 14. Weather presets (clear, thunder storm, ...)
    SEASON = "SEASON"           # 15. Seasonal presets (spring, summer, autumn, winter)
    SEPARATOR = "SEPARATOR"     # 16. Colon block separator (:)
    COMMENT = "COMMENT"         # 17. Comments starting with //
    NEWLINE = "NEWLINE"         # 18. End of statement line (\n)
    INDENT = "INDENT"           # 19. Leading spaces/tabs at start of block lines
    EOF = "EOF"                 # 20. End of source stream


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPLETE KEYWORD & VOCABULARY DEFINITIONS (Sections 4, 5, 6, 8)
# ══════════════════════════════════════════════════════════════════════════════

KEYWORD_CATEGORIES: dict[str, tuple[str, ...]] = {
    "VIDEO_SETUP": (
        "set", "video", "name", "to", "frame", "rate", "resolution", "by",
        "total", "duration", "play", "end", "seconds",
    ),
    "SCENE_AND_STRUCTURE": (
        "second", "keyframe", "fill", "frames", "automatically",
        "transition", "with", "start", "scene",
    ),
    "BACKGROUND": (
        "background", "color", "blur", "brightness", "move", "zoom",
        "layer", "time", "weather", "season", "on", "from", "custom", "path",
    ),
    "CAMERA": (
        "camera", "angle", "position", "zoom", "rotate", "shake",
        "follow", "focus", "pan", "tilt", "intensity", "speed",
    ),
    "TEXT": (
        "place", "text", "at", "size", "font", "style", "shadow",
        "outline", "glow", "animate", "fade", "in", "out", "over",
        "move", "add", "remove",
    ),
    "OBJECT": (
        "object", "scale", "flip", "opacity", "layer", "front", "back",
        "place", "move", "rotate", "remove", "color",
    ),
    "EFFECT": (
        "effect", "add", "level", "type",
    ),
    "SOUND": (
        "sound", "music", "volume", "stop", "narrate", "play", "percent",
    ),
    "LIGHT": (
        "light", "spotlight", "ambient", "source", "warm", "cold",
    ),
    "TIMING": (
        "wait", "for", "speed", "up", "slow", "down", "repeat", "times",
        "loop", "forever", "if", "is", "otherwise",
    ),
    "TRANSITION": (
        "fade", "cut", "slide", "dissolve", "spin", "flip", "wipe",
        "zoom in", "zoom out",
    ),
    "DIRECTION_AND_POSITION": (
        "left", "right", "up", "down", "top", "bottom", "center",
        "forward", "backward", "horizontally", "vertically",
    ),
    "PROPERTY": (
        "bold", "italic", "normal", "large", "medium", "small",
        "high", "low", "bright", "dark", "heavy",
    ),
}

KEYWORDS: frozenset[str] = frozenset(
    word
    for category_words in KEYWORD_CATEGORIES.values()
    for word in category_words
)

UNITS: frozenset[str] = frozenset({
    "seconds", "frames", "pixels", "percent", "degrees",
    # Compact & singular unit forms supported per Section 8 Rule 6 (5s, 26fps, 100px)
    "s", "ms", "fps", "px", "deg", "%",
})

NAMED_COLORS: frozenset[str] = frozenset({
    "blue", "red", "white", "black", "green", "yellow", "orange", "purple",
    "pink", "brown", "gray", "grey", "cyan", "magenta", "gold", "silver",
    "violet", "indigo", "turquoise", "teal", "navy", "maroon", "olive",
    "coral", "salmon", "beige", "ivory", "crimson", "amber", "emerald",
    "lavender", "mint", "peach", "plum", "tan", "aqua", "lime", "bronze",
    "charcoal", "cream", "transparent",
})

POSITIONS: frozenset[str] = frozenset({
    "center", "left", "right", "top", "bottom",
    "top left", "top right", "bottom left", "bottom right",
    "top center", "bottom center",
})

SIZES: frozenset[str] = frozenset({
    "small", "medium", "large",
})

DIRECTIONS: frozenset[str] = frozenset({
    "up", "down", "left", "right", "forward", "backward",
})

EFFECTS: frozenset[str] = frozenset({
    "glow", "blur", "fade", "zoom", "shadow", "fire", "smoke",
    "sparkle", "electric", "explosion", "ripple", "glitch",
    "vignette", "lens flare", "rain", "snow", "vintage",
    "black and white", "color grade",
})

FILE_EXTENSIONS: tuple[str, ...] = (
    ".png", ".jpg", ".jpeg", ".mp4", ".mp3", ".wav",
)

SPEEDS: frozenset[str] = frozenset({
    "slow", "medium", "fast",
})

ANGLES: frozenset[str] = frozenset({
    "front", "back", "top", "bottom", "left side", "right side",
    "diagonal left", "diagonal right", "close up",
    "extreme close up", "wide shot", "medium shot",
    "over shoulder", "first person", "drone view",
})

TIMES_OF_DAY: frozenset[str] = frozenset({
    "sunrise", "sunset", "morning", "afternoon", "night",
    "golden hour", "midnight", "dawn", "dusk",
})

WEATHERS: frozenset[str] = frozenset({
    "clear", "rainy", "snowy", "foggy", "stormy",
    "thunder storm", "thunderstorm", "blizzard", "sandstorm",
})

SEASONS: frozenset[str] = frozenset({
    "spring", "summer", "autumn", "winter",
})

# Multi-word phrases indexed by word-tuple (longest match checked first: 3 words -> 2 words)
MULTI_WORD_PHRASES: dict[tuple[str, ...], TokenType] = {
    **{tuple(p.split()): TokenType.POSITION for p in POSITIONS if " " in p},
    **{tuple(p.split()): TokenType.EFFECT for p in EFFECTS if " " in p},
    **{tuple(p.split()): TokenType.ANGLE for p in ANGLES if " " in p},
    **{tuple(p.split()): TokenType.TIME_OF_DAY for p in TIMES_OF_DAY if " " in p},
    **{tuple(p.split()): TokenType.WEATHER for p in WEATHERS if " " in p},
    **{tuple(p.split()): TokenType.KEYWORD for p in KEYWORDS if " " in p},
}

MAX_PHRASE_WORDS: int = max(len(k) for k in MULTI_WORD_PHRASES)

# Complete suggestion pool for typo correction (Section 8 Rule 7)
ALL_KNOWN_TERMS: tuple[str, ...] = tuple(
    sorted(
        KEYWORDS
        | UNITS
        | NAMED_COLORS
        | POSITIONS
        | SIZES
        | DIRECTIONS
        | EFFECTS
        | SPEEDS
        | ANGLES
        | TIMES_OF_DAY
        | WEATHERS
        | SEASONS
    )
)


# ══════════════════════════════════════════════════════════════════════════════
# 3. TOKEN MODEL & LEXER EXCEPTION (Sections 8 & 9)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Token:
    """
    A single lexical token produced by ``VidexLexer``.

    Supports both attribute access (``token.type``, ``token.value``,
    ``token.line``, ``token.position``) and dictionary-style access
    (``token["type"]``, ``dict(token)``) for seamless Parser integration.
    """

    type: TokenType
    value: str
    line: int
    position: int

    def to_dict(self) -> dict[str, str | int]:
        return {
            "type": self.type.value,
            "value": self.value,
            "line": self.line,
            "position": self.position,
        }

    def format_display(self) -> str:
        """Format token using the exact Section 9 block layout."""
        return (
            "{\n"
            f'  type: "{self.type.value}",\n'
            f"  value: {json.dumps(self.value)},\n"
            f"  line: {self.line},\n"
            f"  position: {self.position}\n"
            "}"
        )

    def __str__(self) -> str:
        return self.format_display()

    # Mapping protocol helpers so Parser code can inspect tokens either as
    # objects (`token.type`) or dicts (`token["type"]`, `dict(token)`).
    def __getitem__(self, key: str) -> str | int:
        return self.to_dict()[key]

    def keys(self) -> tuple[str, ...]:
        return ("type", "value", "line", "position")

    def values(self) -> tuple[str | int, ...]:
        return (self.type.value, self.value, self.line, self.position)

    def items(self) -> Iterator[tuple[str, str | int]]:
        return iter(self.to_dict().items())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


class VidexLexerError(Exception):
    """
    Exception raised when the VIDEX Lexer encounters invalid syntax or an
    unknown word, formatted per Section 8 Rule 7.
    """

    def __init__(
        self,
        line: int,
        word: str,
        suggestion: str | None = None,
        position: int = 0,
        detail: str | None = None,
    ) -> None:
        self.line = line
        self.word = word
        self.suggestion = suggestion
        self.position = position

        if detail is not None:
            formatted = f"❌ VIDEX LEXER ERROR\nLine {line}: {detail}"
        else:
            lines = [
                "❌ VIDEX LEXER ERROR",
                f'Line {line}: Unknown word "{word}"',
            ]
            if suggestion:
                lines.append(f"Did you mean: {suggestion}?")
            formatted = "\n".join(lines)

        super().__init__(formatted)


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row.append(
                min(
                    prev_row[j] + 1,        # deletion
                    curr_row[j - 1] + 1,    # insertion
                    prev_row[j - 1] + cost, # substitution
                )
            )
        prev_row = curr_row
    return prev_row[-1]


def suggest_closest_keyword(word: str) -> str:
    """
    Find the closest valid VIDEX keyword or vocabulary term for ``word``
    combining normalized edit distance, prefix affinity, and sequence similarity.
    """
    target = word.lower()
    best_term = "set"
    best_score = -1.0

    for candidate in ALL_KNOWN_TERMS:
        dist = _levenshtein_distance(target, candidate)
        max_len = max(len(target), len(candidate), 1)
        edit_sim = 1.0 - (dist / max_len)
        seq_sim = SequenceMatcher(None, target, candidate).ratio()
        prefix_bonus = 0.08 if candidate[:1] == target[:1] else 0.0
        score = (0.55 * edit_sim) + (0.45 * seq_sim) + prefix_bonus
        if score > best_score:
            best_score = score
            best_term = candidate

    return best_term


# ══════════════════════════════════════════════════════════════════════════════
# 4. INTERNAL RAW ITEM (Stage 1 Character Scanner -> Stage 2 Classifier)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _RawLexeme:
    kind: str  # "INDENT" | "COMMENT" | "STRING" | "HEX_COLOR" | "NUMBER" | "UNIT" | "COLON" | "WORD"
    value: str
    line: int
    position: int


# ══════════════════════════════════════════════════════════════════════════════
# 5. MAIN VIDEX LEXER IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════


class VidexLexer:
    """
    Character-by-character Lexer for the VIDEX video programming language.

    Usage:
        lexer = VidexLexer()
        tokens = lexer.tokenize(source_code)
    """

    def __init__(self, *, skip_comments: bool = False) -> None:
        """
        Args:
            skip_comments: When ``False`` (default), comment lines starting with
                ``//`` emit a ``COMMENT`` token without lexing comment text.
                When ``True``, comments are omitted entirely from output tokens
                (convenient when feeding directly into a Parser).
        """
        self.skip_comments = skip_comments
        self._source: str = ""
        self._length: int = 0
        self._index: int = 0
        self._line: int = 1
        self._col: int = 0

    # ── Character Cursor Primitives (Section 8 Rule 1) ────────────────────────

    def _current_char(self) -> str | None:
        if self._index >= self._length:
            return None
        return self._source[self._index]

    def _peek_char(self, offset: int = 1) -> str | None:
        target = self._index + offset
        if target >= self._length:
            return None
        return self._source[target]

    def _advance(self) -> str | None:
        ch = self._current_char()
        if ch is None:
            return None
        self._index += 1
        if ch == "\n":
            self._line += 1
            self._col = 0
        else:
            self._col += 1
        return ch

    # ── Public API ────────────────────────────────────────────────────────────

    def tokenize(self, source: str) -> list[Token]:
        """
        Tokenize VIDEX source code into a list of ``Token`` objects ending
        with ``TokenType.EOF``.
        """
        self._source = source
        self._length = len(source)
        self._index = 0
        self._line = 1
        self._col = 0

        tokens: list[Token] = []

        while self._current_char() is not None:
            line_number = self._line
            raw_items = self._scan_line()
            line_has_newline = False

            if self._current_char() == "\r":
                self._advance()
            newline_col = self._col
            if self._current_char() == "\n":
                line_has_newline = True
                self._advance()

            # Empty / whitespace-only lines -> skip completely (Section 8 Rule 6)
            if not raw_items:
                continue

            # Comment-only line handling
            if len(raw_items) == 1 and raw_items[0].kind == "COMMENT":
                if not self.skip_comments:
                    tokens.append(
                        Token(
                            type=TokenType.COMMENT,
                            value=raw_items[0].value,
                            line=line_number,
                            position=raw_items[0].position,
                        )
                    )
                continue

            # Classify tokens on this non-empty statement line
            line_tokens = self._classify_line(raw_items)
            tokens.extend(line_tokens)

            if line_has_newline and line_tokens:
                tokens.append(
                    Token(
                        type=TokenType.NEWLINE,
                        value="\n",
                        line=line_number,
                        position=newline_col,
                    )
                )

        tokens.append(
            Token(
                type=TokenType.EOF,
                value="",
                line=self._line,
                position=self._col,
            )
        )
        return tokens

    # ── Stage 1: Character-by-Character Line Scanner ──────────────────────────

    def _scan_line(self) -> list[_RawLexeme]:
        """Scan a single line character by character up to ``\\r``, ``\\n``, or EOF."""
        items: list[_RawLexeme] = []

        # 1. Track leading spaces/tabs at column 0 for block structure (INDENT)
        indent_chars: list[str] = []
        while (ch := self._current_char()) in (" ", "\t"):
            indent_chars.append(ch)
            self._advance()

        # Peek ahead: if line is blank, don't emit INDENT
        next_ch = self._current_char()
        if next_ch in (None, "\r", "\n"):
            return items

        # If line starts with // after optional indentation, treat whole line as comment
        if next_ch == "/" and self._peek_char() == "/":
            comment_pos = self._col
            comment_text = self._scan_comment()
            return [_RawLexeme("COMMENT", comment_text, self._line, comment_pos)]

        if indent_chars:
            items.append(_RawLexeme("INDENT", "".join(indent_chars), self._line, 0))

        # 2. Scan remaining characters on the line one character at a time
        while (ch := self._current_char()) not in (None, "\r", "\n"):
            if ch in (" ", "\t"):
                self._advance()
                continue

            # Inline // comment
            if ch == "/" and self._peek_char() == "/":
                comment_pos = self._col
                comment_text = self._scan_comment()
                if not self.skip_comments:
                    items.append(
                        _RawLexeme("COMMENT", comment_text, self._line, comment_pos)
                    )
                break

            # Quoted string value or asset filename ("...")
            if ch == '"':
                start_col = self._col
                text_value = self._scan_quoted_string()
                items.append(_RawLexeme("STRING", text_value, self._line, start_col))
                continue

            # Hex color literal (#FF5733, #FFF, #FF5733AA)
            if ch == "#":
                start_col = self._col
                hex_color = self._scan_hex_color()
                items.append(_RawLexeme("HEX_COLOR", hex_color, self._line, start_col))
                continue

            # Block separator (:)
            if ch == ":":
                start_col = self._col
                self._advance()
                items.append(_RawLexeme("COLON", ":", self._line, start_col))
                continue

            # Standalone percent unit symbol (%)
            if ch == "%":
                start_col = self._col
                self._advance()
                items.append(_RawLexeme("UNIT", "%", self._line, start_col))
                continue

            # Numeric literal + optional attached unit (e.g., 26, 5s, 26fps, 100px, 60%)
            if ch.isdigit() or (
                ch == "-" and (self._peek_char() or "").isdigit()
            ):
                items.extend(self._scan_number_and_optional_unit())
                continue

            # Word / identifier token
            if ch.isalpha():
                start_col = self._col
                word = self._scan_word()
                items.append(_RawLexeme("WORD", word, self._line, start_col))
                continue

            # Unrecognized symbol character
            bad_col = self._col
            bad_ch = ch
            self._advance()
            raise VidexLexerError(
                line=self._line,
                word=bad_ch,
                position=bad_col,
                detail=f'Unexpected character "{bad_ch}" at position {bad_col}',
            )

        return items

    def _scan_comment(self) -> str:
        chars: list[str] = []
        while (ch := self._current_char()) not in (None, "\r", "\n"):
            chars.append(ch)
            self._advance()
        return "".join(chars)

    def _scan_quoted_string(self) -> str:
        line_num = self._line
        start_col = self._col
        self._advance()  # Opening quote '"'
        chars: list[str] = []

        while True:
            ch = self._current_char()
            if ch in (None, "\r", "\n"):
                raise VidexLexerError(
                    line=line_num,
                    word='"' + "".join(chars),
                    position=start_col,
                    detail=f"Unterminated string literal starting at position {start_col}",
                )
            if ch == "\\" and self._peek_char() in ('"', "\\"):
                self._advance()
                escaped = self._advance()
                if escaped is not None:
                    chars.append(escaped)
                continue
            if ch == '"':
                self._advance()  # Closing quote '"'
                return "".join(chars)
            chars.append(ch)
            self._advance()

    def _scan_hex_color(self) -> str:
        line_num = self._line
        start_col = self._col
        self._advance()  # '#'
        hex_digits: list[str] = []

        while (ch := self._current_char()) is not None and (
            ch.isalnum() or ch == "_"
        ):
            hex_digits.append(ch)
            self._advance()

        raw_hex = "".join(hex_digits)
        valid_hex = all(c in "0123456789abcdefABCDEF" for c in raw_hex)
        if not valid_hex or len(raw_hex) not in (3, 4, 6, 8):
            raise VidexLexerError(
                line=line_num,
                word=f"#{raw_hex}",
                position=start_col,
                detail=f'Invalid hex color "#{raw_hex}" (expected #RGB, #RGBA, #RRGGBB, or #RRGGBBAA)',
            )
        return f"#{raw_hex.upper()}"

    def _scan_number_and_optional_unit(self) -> list[_RawLexeme]:
        line_num = self._line
        num_col = self._col
        num_chars: list[str] = []

        if self._current_char() == "-":
            num_chars.append("-")
            self._advance()

        while (ch := self._current_char()) is not None and ch.isdigit():
            num_chars.append(ch)
            self._advance()

        if self._current_char() == "." and (self._peek_char() or "").isdigit():
            num_chars.append(".")
            self._advance()
            while (ch := self._current_char()) is not None and ch.isdigit():
                num_chars.append(ch)
                self._advance()

        result = [_RawLexeme("NUMBER", "".join(num_chars), line_num, num_col)]

        # Section 8 Rule 6: Numbers immediately followed by units (5s, 26fps, 100px, 60%)
        next_ch = self._current_char()
        if next_ch == "%":
            unit_col = self._col
            self._advance()
            result.append(_RawLexeme("UNIT", "%", line_num, unit_col))
        elif next_ch is not None and next_ch.isalpha():
            unit_col = self._col
            unit_raw = self._scan_word()
            unit_norm = unit_raw.lower()
            if unit_norm in UNITS:
                result.append(_RawLexeme("UNIT", unit_norm, line_num, unit_col))
            else:
                suggestion = suggest_closest_keyword(unit_raw)
                raise VidexLexerError(
                    line=line_num,
                    word=unit_raw,
                    suggestion=suggestion,
                    position=unit_col,
                )

        return result

    def _scan_word(self) -> str:
        chars: list[str] = []
        while (ch := self._current_char()) is not None and (
            ch.isalnum() or ch in ("_", "-")
        ):
            chars.append(ch)
            self._advance()
        return "".join(chars)

    # ── Stage 2: Phrase Grouping & Context-Aware Classification ───────────────

    @staticmethod
    def _is_file_string(text: str) -> bool:
        stripped = text.strip().lower()
        return any(
            stripped.endswith(ext) and len(stripped) > len(ext)
            for ext in FILE_EXTENSIONS
        )

    def _classify_line(self, raw_items: Sequence[_RawLexeme]) -> list[Token]:
        tokens: list[Token] = []
        i = 0
        n = len(raw_items)

        while i < n:
            item = raw_items[i]

            if item.kind == "INDENT":
                tokens.append(
                    Token(TokenType.INDENT, item.value, item.line, item.position)
                )
                i += 1
                continue

            if item.kind == "COMMENT":
                tokens.append(
                    Token(TokenType.COMMENT, item.value, item.line, item.position)
                )
                i += 1
                continue

            if item.kind == "COLON":
                tokens.append(
                    Token(TokenType.SEPARATOR, ":", item.line, item.position)
                )
                i += 1
                continue

            if item.kind == "HEX_COLOR":
                tokens.append(
                    Token(TokenType.COLOR, item.value, item.line, item.position)
                )
                i += 1
                continue

            if item.kind == "NUMBER":
                tokens.append(
                    Token(TokenType.NUMBER, item.value, item.line, item.position)
                )
                i += 1
                continue

            if item.kind == "UNIT":
                tokens.append(
                    Token(TokenType.UNIT, item.value, item.line, item.position)
                )
                i += 1
                continue

            if item.kind == "STRING":
                str_type = (
                    TokenType.FILE
                    if self._is_file_string(item.value)
                    else TokenType.VALUE
                )
                tokens.append(
                    Token(str_type, item.value, item.line, item.position)
                )
                i += 1
                continue

            # item.kind == "WORD"
            # Step A: Greedy multi-word phrase matching (3 words -> 2 words)
            matched_phrase = False
            for span in range(min(MAX_PHRASE_WORDS, n - i), 1, -1):
                if all(raw_items[i + k].kind == "WORD" for k in range(span)):
                    phrase_tuple = tuple(
                        raw_items[i + k].value.lower() for k in range(span)
                    )
                    phrase_type = MULTI_WORD_PHRASES.get(phrase_tuple)
                    if phrase_type is not None:
                        tokens.append(
                            Token(
                                type=phrase_type,
                                value=" ".join(phrase_tuple),
                                line=item.line,
                                position=item.position,
                            )
                        )
                        i += span
                        matched_phrase = True
                        break
            if matched_phrase:
                continue

            # Step B: Single-word classification (case-insensitive per Section 8 Rule 8)
            word_lower = item.value.lower()
            token_type = self._classify_word_in_context(
                word_lower=word_lower,
                prior_tokens=tokens,
                remaining_items=raw_items[i + 1 :],
            )

            if token_type is not None:
                # Preserve original case for bare asset descriptors like `mountains` in
                # `set background to mountains from "001.png"`, lowercase reserved tokens.
                normalized_val = (
                    item.value if token_type == TokenType.VALUE else word_lower
                )
                tokens.append(
                    Token(token_type, normalized_val, item.line, item.position)
                )
                i += 1
                continue

            # Step C: Unknown word -> raise formatted VIDEX LEXER ERROR with suggestion
            suggestion = suggest_closest_keyword(item.value)
            raise VidexLexerError(
                line=item.line,
                word=item.value,
                suggestion=suggestion,
                position=item.position,
            )

        return tokens

    def _classify_word_in_context(
        self,
        *,
        word_lower: str,
        prior_tokens: Sequence[Token],
        remaining_items: Sequence[_RawLexeme],
    ) -> TokenType | None:
        """
        Resolve token classification for single words, using surrounding statement
        tokens to cleanly disambiguate words shared across VIDEX categories
        (e.g., `left` as POSITION vs DIRECTION, `frames` in `fill frames` vs
        `10 frames`, `fade`/`zoom` as verb/property KEYWORD vs EFFECT).
        """
        non_indent = [t for t in prior_tokens if t.type != TokenType.INDENT]
        prev = non_indent[-1] if non_indent else None
        prev_words = [t.value.lower() for t in non_indent]
        next_word = (
            remaining_items[0].value.lower()
            if remaining_items and remaining_items[0].kind == "WORD"
            else None
        )

        # 1. Section 6 WAY 3: Unquoted asset label before `from "<file>"`
        #    e.g., `set background to mountains from "001.png"` -> KEYWORD VALUE KEYWORD FILE
        if (
            word_lower not in ALL_KNOWN_TERMS
            and prev is not None
            and prev.value in ("to", "object")
            and len(remaining_items) >= 2
            and remaining_items[0].kind == "WORD"
            and remaining_items[0].value.lower() == "from"
            and remaining_items[1].kind == "STRING"
            and self._is_file_string(remaining_items[1].value)
        ):
            return TokenType.VALUE

        # 2. Units (`seconds`, `frames`, `pixels`, `percent`, `degrees`, `s`, `fps`, `px`)
        if word_lower == "frames":
            # `fill frames automatically` -> KEYWORD; `10 frames` or standalone -> UNIT
            if prev is not None and prev.value == "fill":
                return TokenType.KEYWORD
            return TokenType.UNIT
        if word_lower in UNITS:
            return TokenType.UNIT

        # 3. Unambiguous single-word domain categories
        if word_lower in NAMED_COLORS:
            return TokenType.COLOR
        if word_lower in TIMES_OF_DAY:
            return TokenType.TIME_OF_DAY
        if word_lower in WEATHERS:
            return TokenType.WEATHER
        if word_lower in SEASONS:
            return TokenType.SEASON

        # 4. Speed vs Size vs Timing Keyword (`slow`, `medium`, `fast`, `small`, `large`)
        if word_lower == "fast":
            return TokenType.SPEED
        if word_lower == "slow":
            # `slow down` command starts with KEYWORD `slow`; speed values use SPEED
            if prev is None and next_word == "down":
                return TokenType.KEYWORD
            return TokenType.SPEED
        if word_lower == "medium":
            if (prev is not None and prev.type == TokenType.SPEED) or (
                "speed" in prev_words
            ):
                return TokenType.SPEED
            return TokenType.SIZE
        if word_lower in SIZES:
            return TokenType.SIZE

        # 5. Spatial categories: ANGLE vs POSITION vs DIRECTION vs layer KEYWORD
        if word_lower in ("front", "back"):
            if "layer" in prev_words:
                return TokenType.KEYWORD
            return TokenType.ANGLE

        if word_lower in ("top", "bottom"):
            if "angle" in prev_words or (
                prev is not None and prev.type == TokenType.ANGLE
            ):
                return TokenType.ANGLE
            return TokenType.POSITION

        if word_lower == "center":
            return TokenType.POSITION

        if word_lower in ("up", "down", "forward", "backward"):
            return TokenType.DIRECTION

        if word_lower in ("left", "right"):
            if prev is not None and prev.type == TokenType.DIRECTION:
                return TokenType.DIRECTION
            if prev is not None and prev.value in (
                "pan", "tilt", "move", "slide", "wipe", "flip", "rotate",
            ):
                return TokenType.DIRECTION
            return TokenType.POSITION

        # 6. Visual Effects (`glow`, `blur`, `fade`, `zoom`, `shadow`, + pure effects)
        if word_lower in EFFECTS:
            # Command verbs at start of statement: e.g., `fade text in over 10 frames`
            if prev is None and next_word is not None and next_word not in EFFECTS:
                if word_lower in KEYWORDS:
                    return TokenType.KEYWORD
            # Transition keyword after `transition with`: e.g., `transition with fade`
            if (
                len(prev_words) >= 2
                and prev_words[-2] == "transition"
                and prev_words[-1] == "with"
                and word_lower in KEYWORDS
            ):
                return TokenType.KEYWORD
            # Object/camera/background property setter: e.g., `set camera zoom to wide shot`
            if (
                prev_words[:1] == ["set"]
                and next_word == "to"
                and word_lower in KEYWORDS
            ):
                return TokenType.KEYWORD
            return TokenType.EFFECT

        # 7. Standard VIDEX Keywords
        if word_lower in KEYWORDS:
            return TokenType.KEYWORD

        return None


def tokenize_videx(source: str, *, skip_comments: bool = False) -> list[Token]:
    """Convenience functional entry point wrapping ``VidexLexer.tokenize``."""
    return VidexLexer(skip_comments=skip_comments).tokenize(source)


def format_tokens(tokens: Sequence[Token]) -> str:
    """Format a token stream in the Section 9 multi-block display style."""
    return "\n\n".join(token.format_display() for token in tokens)


# ══════════════════════════════════════════════════════════════════════════════
# 6. SECTION 7 OFFICIAL TEST PROGRAM & CLI RUNNER
# ══════════════════════════════════════════════════════════════════════════════

SECTION_7_TEST_CODE: str = """\
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

second 2:
    transition with fade

    keyframe 1:
        set background to "Himalayas mountain"
        set background time to sunrise
        set background weather to clear
        set camera angle to drone view
        set camera zoom to wide shot

    keyframe 13:
        camera tilt down speed slow
        set light color to warm
        set light brightness to 80 percent

    keyframe 26:
        place text "The Himalayas" at bottom center
        set text color to white
        set text size to 50
        fade text in over 10 frames
        add text glow color white

    fill frames automatically
end second 2

second 3:
    keyframe 1:
        place object "eagle" at top left
        move object "eagle" to top right by 500 pixels
        add effect shadow to object "eagle"

    keyframe 26:
        scale object "eagle" to small
        set object "eagle" opacity to 50 percent
        narrate "Welcome to the roof of the world"

    fill frames automatically
end second 3

play video
end
"""


def main() -> None:
    """Tokenize either a provided `.videx` file path or the Section 7 test suite."""
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
    else:
        source = SECTION_7_TEST_CODE

    lexer = VidexLexer()
    try:
        tokens = lexer.tokenize(source)
    except VidexLexerError as exc:
        print(exc)
        sys.exit(1)

    print("═" * 60)
    print(f"VIDEX LEXER OUTPUT — Total Tokens: {len(tokens)}")
    print("═" * 60)
    print(format_tokens(tokens))


if __name__ == "__main__":
    main()
