"""Well-formedness / anti-obfuscation gate: production prompt stage 1.

Replaces canonicalisation in the serving path (v29 rationale: require
well-formed language instead of repairing obfuscated input). Obfuscated text
is rejected with ``prompt_format_invalid`` rather than repaired and
classified: repairing in front of an English-only lexical encoder reopens
prompt-engineering games, and the frozen classifier was never qualified on
canonicalised text in production.

Deterministic, unicode-based, no ML. A prompt is rejected when any rule fires:

* **W1** zero-width / format (``Cf``) characters anywhere — homoglyph glue.
* **W2** mixed-script confusables: Cyrillic/Greek letters that render as
  Latin, embedded in predominantly-Latin text. Predominantly non-Latin text
  belongs to the language gate, not this one.
* **W3** interleaved leet inside words: alphanumeric tokens whose digits do
  not form one contiguous block at a token edge (``4k``/``ps5``/``covid19``
  pass; ``n4k3d``/``s3x`` fail).
* **W4** spaced-out words: five or more consecutive single letters separated
  by single spaces/hyphens/dots (``n a k e d``).
* **W5** structural junk: empty, longer than 1700 characters (the 1500-char
  request contract plus style-template expansion headroom), control
  characters, or a single character run longer than 31. Prompts that fit the
  character bound but exceed the classifier's token window are rejected
  fail-closed at classification time instead (see ``classifier.py``).

All thresholds are deliberately conservative: normal child-typed prompts
(``a 4 year old cat``, ``8k space poster``, ``i want 2 cats``) must pass.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

#: Gate identifier reported through the API surface.
GATE_NAME = "wellformed"

# Cyrillic/Greek glyphs visually confusable with Latin (identical to the
# frozen canonicalizer's map to avoid drift).
_CONFUSABLES = set("авекмнорстухіѕјԁɡαβεζηικμνορτυχ")

_SINGLE_LETTER_RUN = re.compile(r"(?:[A-Za-z](?:[\s.\-]{1,2})){4,}[A-Za-z](?![A-Za-z])")
_LONG_RUN = re.compile(r"(.)\1{31,}")
_TOKEN = re.compile(r"[^\s]+")

# Strip ALL leading/trailing non-alphanumerics (ASCII and Unicode punctuation
# alike) so tokens like "«n4k3d»" cannot dodge the interior-digit check.
_TOKEN_TRIM = re.compile(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$")

# Tokens whose interior digits are known-benign technical vocabulary.
_DIGIT_TOKEN_ALLOWLIST = {
    "4k", "8k", "2k", "16k", "24k", "32k", "1080p", "720p", "480p",
    "mp3", "mp4", "h264", "h265", "y2k", "gpt4", "gpt5", "covid19",
    "co2", "3d", "2d", "5g", "4g", "usb3", "usb4",
}


@dataclass
class WellformedResult:
    """Outcome of the well-formedness gate.

    Attributes:
        ok: ``True`` when the prompt may proceed to the language gate.
        reason_code: ``None`` when ok, else ``prompt_format_invalid``.
        rule: which rule fired (``W1``..``W5``), for logs only.
        detail: human-readable detail for logs only.
    """

    ok: bool
    reason_code: Optional[str] = None
    rule: Optional[str] = None
    detail: Optional[str] = None


def _has_edge_digit_block(token: str) -> bool:
    """Return True when all digits in the token form one contiguous block at a token edge."""
    digit_positions = [i for i, c in enumerate(token) if c.isdigit()]
    if not digit_positions:
        return True
    n = len(token)
    lo, hi = digit_positions[0], digit_positions[-1]
    contiguous = hi - lo + 1 == len(digit_positions)
    return contiguous and (lo == 0 or hi == n - 1)


def check(prompt: str) -> WellformedResult:
    """Evaluate the anti-obfuscation rules against a raw prompt.

    Args:
        prompt: the raw user prompt.

    Returns:
        A :class:`WellformedResult`; ``ok`` is ``False`` with
        ``reason_code="prompt_format_invalid"`` when any rule fires.
    """

    def fail(rule: str, detail: str) -> WellformedResult:
        return WellformedResult(False, "prompt_format_invalid", rule, detail)

    if not prompt or not prompt.strip():
        return fail("W5", "empty prompt")
    if len(prompt) > 1700:
        return fail("W5", f"prompt too long ({len(prompt)} > 1700 chars)")
    if _LONG_RUN.search(prompt):
        return fail("W5", "repetition run > 31 chars")

    # W1: format/zero-width characters (checked before any normalisation).
    if any(unicodedata.category(c) == "Cf" for c in prompt):
        return fail("W1", "zero-width/format character present")
    if any(unicodedata.category(c) == "Cc" for c in prompt):
        return fail("W5", "control character present")

    # W2: confusables embedded in mostly-Latin text.
    latin = sum(1 for c in prompt if "a" <= c <= "z" or "A" <= c <= "Z")
    confusables = sum(1 for c in prompt if c in _CONFUSABLES)
    if confusables > 0 and latin >= 0.7 * max(1, latin + confusables):
        return fail("W2", f"{confusables} mixed-script confusable character(s) in Latin text")

    # W3: interior leet digits.
    for tok in _TOKEN.findall(prompt):
        t = _TOKEN_TRIM.sub("", tok)
        if not t or t.lower() in _DIGIT_TOKEN_ALLOWLIST:
            continue
        has_alpha = any(c.isalpha() for c in t)
        has_digit = any(c.isdigit() for c in t)
        if has_alpha and has_digit and all("a" <= c.lower() <= "z" or c.isdigit() for c in t):
            if not _has_edge_digit_block(t):
                return fail("W3", f"interleaved digits inside token {t!r}")

    # W4: spaced-out single letters.
    m = _SINGLE_LETTER_RUN.search(prompt)
    if m:
        return fail("W4", f"spaced-out letter sequence near {m.group(0)!r}")

    return WellformedResult(True)
