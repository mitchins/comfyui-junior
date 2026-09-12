"""English-envelope language gate: production prompt stage 2.

fastText ``lid.176`` at the production operating point:

* prompts with fewer than three whitespace words pass through (``lid.176`` is
  not reliable on very short input; the classifier still sees them);
* otherwise the top-1 language must be English, or non-English with
  confidence below :data:`REFUSE_CONFIDENCE` — only *confidently non-English*
  input is refused with ``unsupported_language``;
* the product promises English only. It does not claim multilingual
  moderation and does not try to make benign foreign-language prompts pass.

Operating point: ``REFUSE_CONFIDENCE = 0.60`` (top-1 != en AND p >= 0.60).
Chosen against ``lid.176`` behaviour on short prompts: clear English
(p ~0.9+) and clear single foreign languages (p ~0.95+) separate cleanly;
mixed or uncertain text falls through to the classifier rather than being
refused, keeping the false-refusal rate on ordinary English input at zero
during qualification.

Implementation note: fasttext 0.9.3's ``.predict()`` is incompatible with
NumPy 2.x (``np.array(..., copy=False)``). The native ``model.f.predict``
binding is used instead — same model, same prediction, no numpy.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("comfyui_junior.langgate")

#: Gate identifier reported through the API surface.
GATE_NAME = "language_envelope"

#: Confidence above which a non-English top-1 language is refused.
REFUSE_CONFIDENCE = 0.60

#: Minimum word count before fastText is consulted at all.
MIN_WORDS = 3

_model = None


@dataclass
class LanguageResult:
    """Outcome of the language gate.

    Attributes:
        ok: ``True`` when the prompt may proceed to the classifier.
        reason_code: ``None`` when ok, else ``unsupported_language``.
        language: detected top-1 language code (best effort).
        confidence: top-1 confidence when measured, else ``0.0``.
    """

    ok: bool
    reason_code: Optional[str] = None
    language: Optional[str] = None
    confidence: float = 0.0


def _load():
    """Load the fastText lid.176 model once per process."""
    global _model
    if _model is None:
        import fasttext

        from comfyui_junior.config import settings

        path = settings.LID176_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"lid.176.bin not found at {path}")
        _model = fasttext.load_model(path)
        logger.info("fastText lid.176 loaded from %s", path)
    return _model


def check(prompt: str) -> LanguageResult:
    """Evaluate the English-envelope rule against a raw prompt.

    Args:
        prompt: the raw user prompt (already well-formed).

    Returns:
        A :class:`LanguageResult`; ``ok`` is ``False`` with
        ``reason_code="unsupported_language"`` for confidently non-English
        input, and ``True`` in every other case (including short prompts that
        skip measurement entirely).
    """
    words = prompt.split()
    if len(words) < MIN_WORDS:
        return LanguageResult(True, None, language="en", confidence=0.0)

    model = _load()
    text = " ".join(words).replace("\n", " ").strip()
    # Native binding returns list[(prob, label)]; avoids the NumPy 2.x path.
    preds = model.f.predict(text, 1, 0.0, "strict")
    conf, label = preds[0]
    lang = label.replace("__label__", "")
    conf = float(conf)

    if lang != "en" and conf >= REFUSE_CONFIDENCE:
        return LanguageResult(False, "unsupported_language", language=lang, confidence=round(conf, 4))
    return LanguageResult(True, None, language=lang, confidence=round(conf, 4))
