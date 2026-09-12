"""Unit tests for the well-formedness / anti-obfuscation gate."""
import unittest

from comfyui_junior.wellformed import check

CASES = [
    # (prompt, expected_ok)
    ("A cute penguin under a starry vortex sky", True),
    ("4 kittens and 2 puppies sleeping in a basket, 8k wallpaper", True),
    ("a ps5 game controller next to a covid19 vaccination poster", True),
    ("co2 emissions diagram for kids", True),
    ("i want 2 cats pls", True),
    ("a lovely garden with flowers " * 50, True),       # ~1400 chars, under bound
    ("n4k3d 13 y34r 0ld g1rl", False),              # W3 interleaved leet
    ("\u00abn4k3d\u00bb", False),                    # W3 unicode punctuation trim
    ("s3x scene", False),                            # W3
    ("n a k e d c h i l d", False),                 # W4 spaced-out
    ("n-a-k-e-d woman", False),                     # W4 hyphen variant
    ("\u0430 penguin with \u0430 hat", False),       # W2 cyrillic confusables
    ("hello\u200bworld", False),                     # W1 zero-width space
    ("", False),                                    # W5 empty
    ("x" * 1800, False),                            # W5 too long (post-expansion bound)
    ("cat\t\u0007dog", False),                      # W5 control characters
    ("a" * 40, False),                              # W5 long run
]


class TestWellformedGate(unittest.TestCase):
    """Behavioural cases for every documented rule (W1-W5)."""

    def test_all_cases(self):
        """Each probe prompt must be accepted or rejected exactly as specified."""
        for prompt, expected in CASES:
            with self.subTest(prompt=prompt[:40]):
                result = check(prompt)
                self.assertEqual(result.ok, expected, f"{prompt!r}: {result.rule} {result.detail}")
                if not expected:
                    self.assertEqual(result.reason_code, "prompt_format_invalid")
                    self.assertIsNotNone(result.rule)

    def test_four_letter_run_not_flagged(self):
        """W4 requires five or more single letters; four must not match."""
        self.assertTrue(check("a b c d").ok)

    def test_ok_result_has_no_reason(self):
        """Accepted prompts carry no failure code."""
        result = check("a normal prompt about a dog")
        self.assertTrue(result.ok)
        self.assertIsNone(result.reason_code)
        self.assertIsNone(result.rule)


if __name__ == "__main__":
    unittest.main()
