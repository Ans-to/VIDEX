"""Comprehensive unit tests for the VIDEX Lexer (`videx_lexer.py`)."""

import unittest

from videx_lexer import (
    KEYWORD_CATEGORIES,
    KEYWORDS,
    SECTION_7_TEST_CODE,
    TokenType,
    VidexLexer,
    VidexLexerError,
    tokenize_videx,
)


class TestVidexLexer(unittest.TestCase):
    def setUp(self) -> None:
        self.lexer = VidexLexer()

    def test_section_9_exact_positions_and_format(self) -> None:
        tokens = self.lexer.tokenize('set background to "Himalayas mountain"')
        # Expect 4 statement tokens + EOF (no trailing newline in string)
        self.assertEqual(len(tokens), 5)
        self.assertEqual(
            [t.to_dict() for t in tokens[:4]],
            [
                {"type": "KEYWORD", "value": "set", "line": 1, "position": 0},
                {"type": "KEYWORD", "value": "background", "line": 1, "position": 4},
                {"type": "KEYWORD", "value": "to", "line": 1, "position": 15},
                {
                    "type": "VALUE",
                    "value": "Himalayas mountain",
                    "line": 1,
                    "position": 18,
                },
            ],
        )
        self.assertEqual(tokens[4].type, TokenType.EOF)
        # Verify exact Section 9 string display layout
        self.assertEqual(
            str(tokens[3]),
            '{\n  type: "VALUE",\n  value: "Himalayas mountain",\n  line: 1,\n  position: 18\n}',
        )
        # Verify dict-style access compatibility for downstream Parser
        self.assertEqual(tokens[0]["type"], "KEYWORD")
        self.assertEqual(dict(tokens[0])["value"], "set")

    def test_all_20_token_types_recognized(self) -> None:
        code = "\n".join(
            [
                "// Header comment",
                'set background to mountains from "001.png"',
                "set background season to autumn",
                "set background time to golden hour",
                "set background weather to thunder storm",
                "set camera angle to extreme close up",
                "second 1:",
                "    scale object \"earth\" to large",
                "    move object \"earth\" to bottom right by 100 pixels",
                "    camera pan right speed fast",
                "    add effect black and white to object \"earth\" color #FF5733",
            ]
        )
        tokens = self.lexer.tokenize(code)
        found_types = {t.type for t in tokens}
        self.assertEqual(found_types, set(TokenType))

    def test_all_keyword_categories_present(self) -> None:
        self.assertEqual(len(KEYWORD_CATEGORIES), 13)
        for category, words in KEYWORD_CATEGORIES.items():
            for phrase in words:
                tokens = self.lexer.tokenize(phrase)
                non_eof = [t for t in tokens if t.type != TokenType.EOF]
                self.assertEqual(
                    len(non_eof),
                    1,
                    f"Phrase {phrase!r} in {category} should lex as 1 token, got {non_eof}",
                )

    def test_section_6_all_three_asset_reference_ways(self) -> None:
        # Way 1: General word
        t1 = self.lexer.tokenize('set background to "mountains"')
        self.assertEqual(
            [(t.type, t.value) for t in t1[:-1]],
            [
                (TokenType.KEYWORD, "set"),
                (TokenType.KEYWORD, "background"),
                (TokenType.KEYWORD, "to"),
                (TokenType.VALUE, "mountains"),
            ],
        )

        # Way 2: Multi-word description in quotes
        t2 = self.lexer.tokenize('set background to "Times Square New York"')
        self.assertEqual(t2[3].type, TokenType.VALUE)
        self.assertEqual(t2[3].value, "Times Square New York")

        # Way 3: Import user files (including unquoted descriptor before `from "..."`)
        t3a = self.lexer.tokenize('set background to mountains from "001.png"')
        self.assertEqual(
            [(t.type, t.value) for t in t3a[:-1]],
            [
                (TokenType.KEYWORD, "set"),
                (TokenType.KEYWORD, "background"),
                (TokenType.KEYWORD, "to"),
                (TokenType.VALUE, "mountains"),
                (TokenType.KEYWORD, "from"),
                (TokenType.FILE, "001.png"),
            ],
        )

        t3b = self.lexer.tokenize('place object "dog" from "mydog.png" at left')
        self.assertEqual(
            [(t.type, t.value) for t in t3b[:-1]],
            [
                (TokenType.KEYWORD, "place"),
                (TokenType.KEYWORD, "object"),
                (TokenType.VALUE, "dog"),
                (TokenType.KEYWORD, "from"),
                (TokenType.FILE, "mydog.png"),
                (TokenType.KEYWORD, "at"),
                (TokenType.POSITION, "left"),
            ],
        )

        t3c = self.lexer.tokenize('play sound from "mysound.mp3"')
        self.assertEqual(t3c[3], self.lexer.tokenize('"mysound.mp3"')[0].__class__(TokenType.FILE, "mysound.mp3", 1, 16))

        t3d = self.lexer.tokenize('set background to video from "ocean_clip.mp4"')
        self.assertEqual(t3d[5].type, TokenType.FILE)
        self.assertEqual(t3d[5].value, "ocean_clip.mp4")

    def test_compact_numbers_with_units(self) -> None:
        tokens = self.lexer.tokenize("wait for 5s\nset frame rate to 26fps\nmove by 100px\n")
        pairs = [(t.type, t.value) for t in tokens if t.type in (TokenType.NUMBER, TokenType.UNIT)]
        self.assertEqual(
            pairs,
            [
                (TokenType.NUMBER, "5"),
                (TokenType.UNIT, "s"),
                (TokenType.NUMBER, "26"),
                (TokenType.UNIT, "fps"),
                (TokenType.NUMBER, "100"),
                (TokenType.UNIT, "px"),
            ],
        )

    def test_case_insensitivity(self) -> None:
        t_upper = tokenize_videx('SET VIDEO NAME TO "Clip"')
        t_lower = tokenize_videx('set video name to "Clip"')
        t_mixed = tokenize_videx('Set Video Name To "Clip"')
        self.assertEqual(t_upper, t_lower)
        self.assertEqual(t_mixed, t_lower)

    def test_skip_comments_option_and_empty_line_numbers(self) -> None:
        src = "// First comment\n\n\nset frame rate to 26\n"
        tokens = tokenize_videx(src, skip_comments=True)
        self.assertEqual(tokens[0].type, TokenType.KEYWORD)
        self.assertEqual(tokens[0].line, 4)

    def test_unknown_word_error_format_and_suggestion(self) -> None:
        bad_code = 'set video name to "Demo"\nset backgrond to "space"\n'
        with self.assertRaises(VidexLexerError) as ctx:
            self.lexer.tokenize(bad_code)

        err_text = str(ctx.exception)
        self.assertIn("❌ VIDEX LEXER ERROR", err_text)
        self.assertIn('Line 2: Unknown word "backgrond"', err_text)
        self.assertIn("Did you mean: background?", err_text)

    def test_section_7_full_example_program(self) -> None:
        tokens = self.lexer.tokenize(SECTION_7_TEST_CODE)
        self.assertGreater(len(tokens), 150)
        self.assertEqual(tokens[0].type, TokenType.COMMENT)
        self.assertEqual(tokens[-1].type, TokenType.EOF)
        self.assertEqual(len(KEYWORDS) >= 85, True)


if __name__ == "__main__":
    unittest.main()
