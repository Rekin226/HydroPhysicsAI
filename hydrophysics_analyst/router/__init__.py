"""Question routing: hard refusals, then rules, then (if needed) one LLM structured-output call.

Order (spec §8): ``hard_refusal`` on the raw question first — a hit skips the LLM entirely;
then the rules parser; the LLM only when the rules are not confident; LLM errors fall back to
the rules' route or a clarification. Every path ends in an enum-only :class:`Route`.
"""
from __future__ import annotations

import os

from ..page import PageBundle
from .rules import RulesRouter, detect_lang, hard_refusal
from .schemas import Route

V1_ROUTED = ("policy_head", "start_compare", "lever_rank", "township_internals", "rail_internals",
             "solved_compare")


def routed_presets() -> tuple[str, ...]:
    extra = ("cheapest_for_head",) if os.environ.get("ANALYST_ENABLE_CHEAPEST") == "1" else ()
    return V1_ROUTED + extra


def route(b: PageBundle, question: str, prev: dict | None = None, *, llm=None,
          threshold: float = 0.9) -> tuple[Route, dict]:
    """Returns (route, info) where info records which stage decided and any LLM error."""
    info: dict = {"stage": "hard_refusal"}
    ref = hard_refusal(question, b)
    if ref:
        return Route(preset="refuse", params={"reason": ref}, refusal=ref,
                     lang=detect_lang(question), confidence=1.0), info
    r = RulesRouter(b).route(question, prev)
    info["stage"] = "rules"
    if r.confidence >= threshold or llm is None:
        return r, info
    try:
        lr = llm.route(question, prev)
        info["stage"] = "llm"
        lr.lang = lr.lang or r.lang
        return lr, info
    except Exception as e:  # noqa: BLE001  (no key, cache miss, network) -> the rules' answer
        info["llm_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return r, info
