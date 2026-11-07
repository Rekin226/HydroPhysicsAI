"""One question, end to end: route -> preset -> validate -> audit.

The validator runs on every answer. A preset render that fails it is a bug: the user gets a
refusal frame with the deep link instead (``fallback: validator``), and the violations are logged.
"""
from __future__ import annotations

import dataclasses as dc
import time
from pathlib import Path

from . import audit
from . import presets as PR
from .llm.backend import AnthropicBackend, cost_usd
from .llm.cache import CachedBackend
from .page import PageBundle
from .router import route, routed_presets
from .router.llm import LLMRouter
from .session import Session
from .validate import validate


def make_llm(b: PageBundle, *, model: str | None = None, no_api: bool = False, backend=None,
             calls_dir: Path = Path("results/analyst/calls"), fallbacks: bool = True,
             inference_geo: str | None = None) -> LLMRouter:
    inner = backend or AnthropicBackend(model or "claude-opus-5-5", fallbacks=fallbacks,
                                        inference_geo=inference_geo)
    cached = CachedBackend(inner, calls_dir, no_api=no_api,
                           tag={"page": b.page_sha256[:16], "ledger": b.ledger_sha256[:16]})
    return LLMRouter(b, cached, presets=routed_presets())


def answer_route(b: PageBundle, r, lang: str):
    if r.preset and r.preset != "clarify" and not r.clarify:
        return PR.run(b, r.preset, r.params, lang)
    cl = r.clarify or {"reason": "unknown", "candidates": []}
    return PR.clarification(b, r.preset, r.params, cl["reason"], cl.get("candidates"), lang)


def ask(b: PageBundle, question: str, *, lang: str | None = None, use_llm: bool = True,
        no_api: bool = False, model: str | None = None, audit_dir: Path | None = audit.DEFAULT_DIR,
        user: str = "cli", session: Session | None = None, llm: LLMRouter | None = None):
    t0 = time.monotonic()
    session = session or Session()
    if llm is None and use_llm:
        llm = make_llm(b, model=model, no_api=no_api)
    r, info = route(b, question, session.prev, llm=llm if use_llm else None)
    lang = lang or r.lang
    ans = answer_route(b, r, lang)
    violations = validate(ans, b)
    if violations:
        fb = PR.refuse(b, {"reason": "no_run"}, lang)
        ans = dc.replace(fb, deep_link=ans.deep_link, fallback="validator")
    else:
        session.update(ans.preset, ans.params)
    rec = {"user": user, "lang": lang, "question": question, "route": r.to_json(), "stage": info.get("stage"),
           "llm_error": info.get("llm_error"), "rules_hint": info.get("rules_hint"), "answer_preset": ans.preset if not ans.clarify else "clarify",
           "answer_params": ans.params, "answer_text": ans.text(), "citations": ans.citations,
           "refusal": ans.refusal, "clarify": ans.clarify, "fallback": ans.fallback,
           "violations": [v.__dict__ for v in violations], "deep_link": ans.deep_link,
           "page_sha256": b.page_sha256, "ledger_sha256": b.ledger_sha256,
           "strings_sha256": b.strings_sha256, "latency_ms": round(1000 * (time.monotonic() - t0))}
    if info.get("stage") == "llm" and llm is not None and llm.last is not None:
        L = llm.last
        rec |= {"model": L.model, "request_id": L.request_id, "usage": L.usage,
                "cache_hit": getattr(llm.backend, "last_hit", None), "cost_usd": cost_usd(L.model, L.usage)}
    if audit_dir is not None:
        rec["id"] = audit.write(rec, audit_dir)
    return ans, violations, rec
