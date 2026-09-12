"""Production prompt gate: the only path from a raw prompt to a permit.

    prompt
      -> well-formedness / anti-obfuscation gate   (prompt_format_invalid)
      -> English-envelope gate                      (unsupported_language)
      -> single-view FP16 v29db classifier          (content_policy_violation)
      -> policy_v6 PASS/BLOCK

The pipeline is unconditional in appliance mode: no flag, request field,
header, or query parameter bypasses any stage, and an internal error fails
closed.

Evaluation instances (owner-operated, enabled only by the
``JUNIOR_EVALUATION_INSTANCE`` environment variable read once at process
start) inject :class:`EvaluationBypassGate` instead of the classifier-backed
gate. The request-handler code path is identical in both modes — only the
injected gate object differs — so there is exactly one code path to maintain.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from comfyui_junior import langgate, wellformed
from comfyui_junior.classifier import (
    POLICY_VERSION,
    ClassificationResult,
    JuniorSafetyClassifier,
)

logger = logging.getLogger("comfyui_junior.pipeline")

#: Public failure code for malformed or obfuscated prompts.
FAILURE_PROMPT_FORMAT = "prompt_format_invalid"

#: Public failure code for confidently non-English prompts.
FAILURE_LANGUAGE = "unsupported_language"

#: Public failure code for policy violations and fail-closed errors.
FAILURE_POLICY = "content_policy_violation"


@dataclass
class GateResult:
    """Outcome of the production prompt gate.

    Attributes:
        ok: ``True`` only when every stage permits generation.
        failure_code: one of the three public failure codes, or ``None``.
        detail: child-appropriate explanation for the client.
        gate: which stage decided (logs/diagnostics).
        classification: classifier result when stage 3 ran.
        policy: frozen policy identifier.
        reasons: machine-readable reasons for BLOCK results.
    """

    ok: bool
    failure_code: Optional[str] = None
    detail: Optional[str] = None
    gate: Optional[str] = None
    classification: Optional[ClassificationResult] = None
    policy: str = POLICY_VERSION
    reasons: List[str] = field(default_factory=list)


class ProductionGate:
    """Composes the three production stages; constructed once at startup."""

    def __init__(self, classifier: JuniorSafetyClassifier):
        """Bind the gate to a loaded FP16 v29db classifier.

        Args:
            classifier: qualified classifier instance (FP16 on CUDA in
                production; CPU instances are tolerated for diagnostics only
                and are refused by the application startup checks).
        """
        if classifier.dtype is not None and str(classifier.dtype) != "torch.float16":
            logger.warning("ProductionGate constructed with non-FP16 classifier (diagnostic only)")
        self.classifier = classifier

    def check(self, prompt: str) -> GateResult:
        """Run a prompt through well-formedness, language and classifier stages.

        Args:
            prompt: the style-expanded prompt that would be rendered.

        Returns:
            A :class:`GateResult`; ``ok`` is ``True`` only for prompts that
            passed all three stages. Classifier exceptions fail closed.
        """
        wf = wellformed.check(prompt)
        if not wf.ok:
            return GateResult(False, FAILURE_PROMPT_FORMAT,
                              "That text looks scrambled or uses disguised letters.",
                              gate="wellformed")

        lang = langgate.check(prompt)
        if not lang.ok:
            return GateResult(False, FAILURE_LANGUAGE,
                              f"Sorry, we can only understand English right now (detected: {lang.language}).",
                              gate="language_envelope")

        try:
            res = self.classifier.classify(prompt)
        except Exception:
            logger.exception("classifier failure — failing closed")
            return GateResult(False, FAILURE_POLICY,
                              "We couldn't check that prompt, so we can't use it.",
                              gate="classifier_error")

        if res.decision == "BLOCK":
            return GateResult(False, FAILURE_POLICY,
                              "That prompt isn't allowed because it asks for unsafe content.",
                              gate="classifier", classification=res, reasons=res.reasons)
        return GateResult(True, gate="classifier", classification=res, reasons=res.reasons)


class EvaluationBypassGate:
    """Deliberate owner-operated bypass injected on evaluation instances.

    Instances launched with ``JUNIOR_EVALUATION_INSTANCE=1`` exist so the
    owner can measure the unfiltered model externally (including gauging
    false positives of the safety stack) with any client-side judge. This
    object implements the same ``check`` interface as :class:`ProductionGate`
    and permits every prompt; the application marks such instances loudly in
    ``/health`` and disables the child-facing frontend.
    """

    def check(self, prompt: str) -> GateResult:
        """Permit the prompt unconditionally (evaluation instance semantics)."""
        return GateResult(True, gate="evaluation_bypass",
                          reasons=["JUNIOR_EVALUATION_INSTANCE=1 set by owner"])
