"""Unit tests for the production pipeline gate (classifier mocked)."""
import unittest
from unittest.mock import patch

import torch

from comfyui_junior import pipeline
from comfyui_junior.classifier import ClassificationResult, DIMS
from comfyui_junior.pipeline import (
    FAILURE_LANGUAGE,
    FAILURE_POLICY,
    FAILURE_PROMPT_FORMAT,
    EvaluationBypassGate,
    ProductionGate,
)


def _result(decision="PASS"):
    """Build a classifier result with uniform levels for the given decision."""
    level = 0 if decision == "PASS" else 3
    return ClassificationResult(
        decision=decision,
        levels={d: level for d in DIMS},
        probabilities={d: [0.9] * w for d, w in zip(DIMS, [3, 2, 2, 2, 2, 2])},
        reasons=["sexual level 3 >= 2"] if decision == "BLOCK" else [],
    )


class _StubClassifier:
    """Classifier stand-in with a configurable classify() outcome."""

    dtype = torch.float16

    def __init__(self, outcome="PASS"):
        self.outcome = outcome

    def classify(self, prompt):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return _result(self.outcome)


class _FakeBinding:
    """Stand-in for the fastText native binding object (model.f)."""

    def __init__(self, label, prob):
        self._pred = [(prob, f"__label__{label}")]

    def predict(self, *args, **kwargs):
        """Match the native binding's return shape."""
        return self._pred


class _FakeLid:
    """Stand-in for the fastText model; confidently English by default."""

    def __init__(self, label="en", prob=0.95):
        self.f = _FakeBinding(label, prob)


class TestProductionGate(unittest.TestCase):
    """Stage ordering, failure codes and fail-closed behaviour."""

    def setUp(self):
        """Install a confident-English language model for stage 2."""
        self._lid_patch = patch.object(pipeline.langgate, "_model", _FakeLid())
        self._lid_patch.start()
        self.addCleanup(self._lid_patch.stop)

    def test_malformed_prompt_rejected_before_classifier(self):
        """Obfuscated input fails at stage 1 with prompt_format_invalid."""
        gate = ProductionGate(_StubClassifier())
        result = gate.check("n4k3d p3opl3")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, FAILURE_PROMPT_FORMAT)
        self.assertEqual(result.gate, "wellformed")

    def test_foreign_language_rejected(self):
        """Confidently non-English input fails at stage 2 (model mocked)."""
        gate = ProductionGate(_StubClassifier())
        with patch.object(pipeline.langgate, "_model", _FakeLid("fr", 0.99)):
            result = gate.check("un pingouin mignon sous un ciel etoile")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, FAILURE_LANGUAGE)

    def test_block_decision(self):
        """A classifier BLOCK yields content_policy_violation with reasons."""
        gate = ProductionGate(_StubClassifier("BLOCK"))
        result = gate.check("a perfectly well formed english prompt")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, FAILURE_POLICY)
        self.assertEqual(result.gate, "classifier")
        self.assertTrue(result.reasons)

    def test_pass_decision(self):
        """A benign prompt passes all three stages."""
        gate = ProductionGate(_StubClassifier())
        result = gate.check("a cute penguin under a starry vortex sky")
        self.assertTrue(result.ok)
        self.assertEqual(result.gate, "classifier")

    def test_classifier_error_fails_closed(self):
        """A crashing classifier fails closed as content_policy_violation."""
        gate = ProductionGate(_StubClassifier(RuntimeError("gpu gone")))
        result = gate.check("a cute penguin")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, FAILURE_POLICY)
        self.assertEqual(result.gate, "classifier_error")


class TestEvaluationBypassGate(unittest.TestCase):
    """Evaluation instances permit every prompt by design."""

    def test_bypass_permits_unsafe_prompt(self):
        """The owner-operated bypass returns ok with an explicit marker."""
        result = EvaluationBypassGate().check("anything at all")
        self.assertTrue(result.ok)
        self.assertEqual(result.gate, "evaluation_bypass")


if __name__ == "__main__":
    unittest.main()
