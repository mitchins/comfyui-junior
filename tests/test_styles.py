"""Unit tests for the explicit style templates."""
import unittest

from comfyui_junior.styles import STYLE_TEMPLATES, VALID_STYLES, apply_style_template


class TestStyleTemplates(unittest.TestCase):
    """Allowlist, expansion and passthrough behaviour of style templates."""

    def test_none_is_default_and_always_valid(self):
        """None, empty string and 'none' all pass the prompt through unchanged."""
        for style in (None, "", "none"):
            out, applied = apply_style_template("a cute penguin", style)
            self.assertEqual(out, "a cute penguin")
            self.assertIsNone(applied)

    def test_expansion_is_additive_suffix(self):
        """Every template appends its suffix without altering the user text."""
        for style, suffix in STYLE_TEMPLATES.items():
            out, applied = apply_style_template("a farm", style)
            self.assertEqual(applied, style)
            self.assertTrue(out.startswith("a farm"))
            self.assertTrue(out.endswith(suffix))

    def test_known_templates_present(self):
        """The four documented styles exist in the allowlist."""
        self.assertEqual(set(VALID_STYLES), {"none", "real_photo", "cartoon", "colouring_sheet"})

    def test_unknown_style_rejected(self):
        """Values outside the allowlist raise ValueError (API: invalid_style)."""
        for bad in ("colouring", "COLOURING_SHEET", "unsafe", "", " ", "../../etc"):
            if bad == "":
                continue  # empty is the documented default passthrough
            with self.subTest(style=bad):
                with self.assertRaises(ValueError):
                    apply_style_template("x", bad)

    def test_blank_prompt_with_template_rejected(self):
        """A template cannot be applied to a blank prompt."""
        with self.assertRaises(ValueError):
            apply_style_template("   ", "colouring_sheet")

    def test_expansion_deterministic(self):
        """Identical inputs expand identically."""
        self.assertEqual(
            apply_style_template("a farm", "cartoon")[0],
            apply_style_template("a farm", "cartoon")[0],
        )


if __name__ == "__main__":
    unittest.main()
