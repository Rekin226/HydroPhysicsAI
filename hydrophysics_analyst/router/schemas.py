"""The Route: every field an enum (or a number on the page's grid) — no free text anywhere.

The same object comes out of the rules parser and the LLM router; the LLM's schema is
:func:`route_schema`, in the JSON Schema subset structured outputs supports (``anyOf`` with
null for optional fields, ``additionalProperties: false`` everywhere, enums not bounds).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .. import presets as PR
from ..page import PageBundle

CLARIFY = ("ambiguous_town", "missing_year", "missing_cut", "out_of_grid", "unknown")
INHERIT = ("cuts", "start", "year", "town", "by")


@dataclass
class Route:
    preset: str | None = None
    params: dict = field(default_factory=dict)
    inherit: list[str] = field(default_factory=list)
    refusal: str | None = None
    clarify: dict | None = None
    lang: str = "en"
    confidence: float = 0.0
    source: str = "rules"

    def to_json(self) -> dict:
        return asdict(self)


def _nullable(s: dict) -> dict:
    return {"anyOf": [s, {"type": "null"}]}


def route_schema(b: PageBundle, presets: tuple[str, ...]) -> dict:
    e = PR.enums(b)
    cuts = {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                       "required": ["class", "pct"],
                                       "properties": {"class": {"type": "string", "enum": e["classes"]},
                                                      "pct": {"type": "integer", "enum": PR.PCT}}}}
    props = {
        "preset": _nullable({"type": "string", "enum": list(presets)}),
        "cuts": _nullable(cuts),
        "start": _nullable({"type": "integer", "enum": e["starts"]}),
        "year": _nullable({"type": "integer", "enum": e["years"]}),
        "town": _nullable({"type": "string", "enum": e["towns"]}),
        "by": _nullable({"type": "string", "enum": ["head", "gwh"]}),
        "target_m": _nullable({"type": "number", "enum": [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0]}),
        "refusal": _nullable({"type": "string", "enum": [r for r in PR.REFUSALS if r != "provider_refusal"]}),
        "clarify": _nullable({"type": "string", "enum": list(CLARIFY)}),
        "inherit": {"type": "array", "items": {"type": "string", "enum": list(INHERIT)}},
        "answer_lang": {"type": "string", "enum": ["en", "zh"]},
    }
    return {"type": "object", "additionalProperties": False, "required": list(props),
            "properties": props}


PRESET_PARAMS = {
    "policy_head": ("cuts", "start", "year"),
    "start_compare": ("cuts", "year"),
    "lever_rank": ("year", "by"),
    "township_internals": ("cuts", "start", "year", "town"),
    "rail_internals": ("cuts", "start", "year"),
    "solved_compare": ("year",),
    "cheapest_for_head": ("target_m", "start", "year"),
}


def from_llm_json(d: dict, b: PageBundle) -> Route:
    """The LLM's structured output -> Route (params limited to the preset's own)."""
    preset = d.get("preset")
    params = {k: d.get(k) for k in PRESET_PARAMS.get(preset or "", ()) if d.get(k) is not None}
    clarify = {"reason": d["clarify"], "candidates": []} if d.get("clarify") else None
    return Route(preset=preset, params=params, inherit=list(d.get("inherit") or []),
                 refusal=d.get("refusal"), clarify=clarify, lang=d.get("answer_lang") or "en",
                 confidence=1.0, source="llm")
