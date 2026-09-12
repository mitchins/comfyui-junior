"""Explicit style templates appended to the rendered prompt.

Colouring-sheet rendering needs precise wording to produce an empty printable
outline sheet, and the photo/cartoon looks benefit from deliberate wording
too. Rather than sniffing prompts (transparent rewriting), the behaviour is
explicit: the request carries a ``style`` field chosen from a fixed allowlist
and the server expands the selected template *before* the safety gates
(classify-what-you-render). No implicit rewriting of user text happens
anywhere, and ``none`` — the default — is always valid.

Templates are additive-only suffixes with safety-neutral wording, validated
by measurement (see ``docs/RTX3060_QUALIFICATION.md`` and the bake-off
records for the qualification method and receipts).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

#: Style template suffixes keyed by API value.
STYLE_TEMPLATES: Dict[str, str] = {
    "real_photo": (
        ", a realistic photograph, natural lighting, true-to-life colours "
        "and textures, shot on a camera"
    ),
    "cartoon": (
        ", a bright colourful cartoon illustration with clean bold shapes "
        "and a friendly cheerful style"
    ),
    "colouring_sheet": (
        ", simple thick black line art on a plain white background, completely "
        "blank inside the lines, no shading, no colour fill, ready to print"
    ),
}

#: Every accepted ``style`` value, including the default no-op.
VALID_STYLES: Tuple[str, ...] = ("none",) + tuple(STYLE_TEMPLATES)


def apply_style_template(prompt: str, style: Optional[str]) -> Tuple[str, Optional[str]]:
    """Expand an explicitly selected style template.

    Args:
        prompt: the raw user prompt.
        style: requested style value; ``None``, ``""`` and ``"none"`` all mean
            no template (the default, always valid).

    Returns:
        Tuple of ``(render_prompt, applied_template)`` where
        ``applied_template`` is ``None`` when no template was applied.

    Raises:
        ValueError: when ``style`` is outside the allowlist, or a template is
            requested for a blank prompt.
    """
    if style is None or style == "" or style == "none":
        return prompt, None
    if style not in STYLE_TEMPLATES:
        raise ValueError(f"unknown style '{style}'; valid: {list(VALID_STYLES)}")
    if not prompt or not prompt.strip():
        raise ValueError("prompt required with style template")
    return prompt.rstrip(". ,") + STYLE_TEMPLATES[style], style
