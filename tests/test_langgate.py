"""Unit tests for the English-envelope language gate (no model file needed)."""
import unittest
from unittest.mock import patch

from comfyui_junior import langgate


class _FakeBinding:
    """Stand-in for the fastText native binding object (model.f)."""

    def __init__(self, label, prob):
        self._pred = [(prob, f"__label__{label}")]

    def predict(self, *args, **kwargs):
        """Match the native binding's return shape (list of (prob, label))."""
        return self._pred


class _FakeModel:
    """Minimal stand-in for the fastText model object."""

    def __init__(self, label, prob):
        self.f = _FakeBinding(label, prob)


class TestLanguageGate(unittest.TestCase):
    """Operating-point behaviour: refuse only confidently non-English input."""

    def test_short_prompt_passes_through_without_model(self):
        """Prompts under three words skip measurement entirely."""
        result = langgate.check("a cat")
        self.assertTrue(result.ok)
        self.assertIsNone(result.reason_code)

    def test_confident_english_passes(self):
        """High-confidence English is accepted."""
        with patch.object(langgate, "_model", _FakeModel("en", 0.95)):
            result = langgate.check("a cute penguin under a starry sky")
        self.assertTrue(result.ok)

    def test_confident_non_english_refused(self):
        """Non-English top-1 at or above the threshold is refused."""
        with patch.object(langgate, "_model", _FakeModel("fr", 0.97)):
            result = langgate.check("un pingouin mignon sous un ciel etoile")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason_code, "unsupported_language")
        self.assertEqual(result.language, "fr")

    def test_uncertain_non_english_falls_through(self):
        """Low-confidence non-English falls through to the classifier."""
        with patch.object(langgate, "_model", _FakeModel("ca", 0.35)):
            result = langgate.check("a mixed sort of prompt text")
        self.assertTrue(result.ok)
        self.assertEqual(result.language, "ca")


if __name__ == "__main__":
    unittest.main()
