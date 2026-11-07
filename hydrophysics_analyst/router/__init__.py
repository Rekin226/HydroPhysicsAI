"""Question routing: hard refusals, then the rules (refuse or clarify only), then the LLM.

Order (spec §8, ruling R9.2 of 2026-11-07): ``hard_refusal`` on the raw question first (with the
deterministic guards of ``router.guards``) — a hit skips the LLM entirely; then the rules parser, which may only **refuse or clarify**: it never
routes a preset on its own. Every preset is chosen by one LLM structured-output call. The rules'
own parse travels along as a hint (``info["rules_hint"]``, logged in the audit record); it is not
put into the LLM's prompt, so the LLM routes the question as if the rules had not read it.
Without an LLM (``--no-llm``, no key) or when the call fails, a question the rules did not refuse
or clarify gets the ``no_llm`` clarification (use the examples or the form). Every path ends in
an enum-only :class:`Route`.

Why (docs/analyst/EVAL_2026-11-07.md): on held-out v3 every unsafe answer of the deployed
pipeline came from the rules stage routing a must-refuse question to a preset, which the
pipeline then trusted; the arm in which the LLM routed every preset had none.
"""
from __future__ import annotations

import os

from ..page import PageBundle
from . import guards as G
from .rules import RulesRouter, detect_lang, hard_refusal
from .schemas import Route

V1_ROUTED = ("policy_head", "start_compare", "lever_rank", "township_internals", "rail_internals",
             "solved_compare", "how_sure")


def routed_presets() -> tuple[str, ...]:
    extra = ("cheapest_for_head",) if os.environ.get("ANALYST_ENABLE_CHEAPEST") == "1" else ()
    return V1_ROUTED + extra


def no_llm(lang: str) -> Route:
    """The answer to a question only the LLM may route, when there is no LLM to route it."""
    return Route(lang=lang, confidence=0.0, clarify={"reason": "no_llm", "candidates": []})


def rules_decides(r: Route, threshold: float = 0.9) -> bool:
    """The rules stage's route stands only when it is a refusal, or a confident clarification
    (ruling R9.2: the rules may refuse or clarify; they never answer with a preset)."""
    if r.refusal is not None:
        return True
    return r.clarify is not None and r.confidence >= threshold


def _changhua_label(r: Route, question: str) -> Route:
    """A Changhua refusal records which lever the question reads, so the refusal cites the
    right label (C13a head lever, C13b class lever, C13c subsidence lever)."""
    if r.refusal == "changhua_lever":
        r.params = dict(r.params) | {"lever": G.changhua_kind(question)}
    return r


def bare_clarification(r: Route) -> Route:
    """The rules' clarification without the preset or parameters they read: nothing in it can
    be answered as a preset (the shell offers one-click candidates only for a ``for_preset``)."""
    return Route(lang=r.lang, confidence=r.confidence,
                 clarify={"reason": r.clarify["reason"], "candidates": list(r.clarify.get("candidates") or [])})


def route(b: PageBundle, question: str, prev: dict | None = None, *, llm=None,
          threshold: float = 0.9) -> tuple[Route, dict]:
    """Returns (route, info) where info records which stage decided, the rules' hint and any
    LLM error."""
    info: dict = {"stage": "hard_refusal"}
    ref = hard_refusal(question, b)
    if ref:
        return _changhua_label(Route(preset="refuse", params={"reason": ref}, refusal=ref,
                                     lang=detect_lang(question), confidence=1.0), question), info
    r = RulesRouter(b).route(question, prev)
    info["stage"] = "rules"
    if rules_decides(r, threshold):
        return _changhua_label(bare_clarification(r) if r.clarify else r, question), info
    info["rules_hint"] = {"preset": r.preset, "params": r.params, "confidence": r.confidence,
                          "clarify": (r.clarify or {}).get("reason")}
    # without the LLM a low-confidence clarification of the rules stays a clarification, but bare:
    # no preset, no parameters (the rules may not choose a preset, not even as a button)
    fallback = bare_clarification(r) if r.clarify is not None else no_llm(r.lang)
    if llm is None:
        return fallback, info
    try:
        lr = llm.route(question, prev)
        info["stage"] = "llm"
        lr.lang = lr.lang or r.lang
        return _changhua_label(lr, question), info
    except Exception as e:  # noqa: BLE001  (no key, cache miss, network): never the rules' preset
        info["llm_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return fallback, info
