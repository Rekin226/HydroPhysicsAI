"""The LLM router: one stateless structured-output call per question.

The model sees the question, the previous resolved parameters (so "same for 2030" is a patch,
not a transcript) and a frozen system prompt: the role (route, never compute), the project
rule, what each status code allows, the presets with examples, the payload's own enums, the
refusal reasons, and a digit-free topic tag per claim. It never sees a payload number, a ledger
``claim`` one-liner (those carry withdrawn numbers), ``wording_en`` or ``notes``. Its output
schema (``router.schemas.route_schema``) is enums only, so nothing it writes reaches the user.
"""
from __future__ import annotations

import json

from .. import binding as BD
from .. import query as Q
from ..page import PageBundle
from .schemas import Route, from_llm_json, route_schema

PRESET_DOCS = {
    "policy_head": "Head change under a policy: cuts per water-use class (percent on a five-percent grid; "
                   "retiring a class is a cut of one hundred percent), a start year and a reporting year. "
                   "Example: 'what does cutting aquaculture by a third from the later start do?' -> ask to clarify "
                   "the cut, since a third is not on the grid.",
    "start_compare": "Start now or later: the same cuts from each start year, compared at a reporting year.",
    "lever_rank": "Ranking of the water-use classes by head rise ('head') or by head rise per energy given up "
                  "('gwh'). Model internals.",
    "township_internals": "One township's model-internal subsidence row under a policy (labelled, size untested). "
                          "Needs cuts, start, year and a township from the enum.",
    "rail_internals": "High-speed rail: the head answer plus the rail's model-internal distortion note. "
                      "Never a yes/no or a verdict.",
    "solved_compare": "The page's full-model runs side by side at a reporting year.",
    "cheapest_for_head": "Lowest-energy page lever setting that reaches a head target at the registered wells.",
    "how_sure": "How far the result is tested and when the registered tests are scored (no parameters).",
}
REFUSAL_DOCS = {
    "changhua_lever": "any question about Changhua levers or cuts",
    "subsidence_size": "how much subsidence or how many centimetres are avoided, at fan or county scale",
    "area": "area sinking faster than a threshold",
    "recommendation": "which policy to choose or recommend",
    "volumes_money": "pumping volumes, money, budgets",
    "observations": "individual wells, stations or benchmarks",
    "beyond_horizon": "years after the page's last year",
    "out_of_domain": "anything not about this decision page",
    "township_head": "head change for a single township",
    "no_policy": "a township or rail question with no policy at all",
}


def system_prompt(b: PageBundle, presets: tuple[str, ...]) -> str:
    c = Q.ctx(b)
    st = {x["id"]: x["status_current"] for x in b.ledger["claims"]}
    classes = "; ".join(f"{cl['key']} = {cl['en']} / {cl['zh']}" for cl in c.classes)
    towns = "; ".join(f"{t['en']} / {t['zh']}" for t in c.towns)
    topics = "\n".join(f"- {cid} [{st.get(cid, '?')}]: {t}" for cid, t in BD.topics().items())
    docs = "\n".join(f"- {p}: {PRESET_DOCS[p]}" for p in presets if p in PRESET_DOCS)
    refs = "\n".join(f"- {r}: {d}" for r, d in REFUSAL_DOCS.items())
    return f"""You route questions about a groundwater twin's decision page to one fixed preset with parameters.
You never compute, estimate or state a number, and nothing you write is shown to the user: the page's own
code computes every number and its own sentences answer. Project rule: the language model is for routing
and extraction, never for filling physical values.

Fill the JSON schema. Choose exactly one of: a preset with its parameters, a refusal reason, or a
clarification reason. Use only enum values. If the question continues the previous one (for example
"same for the later start", "and irrigation?"), copy the unchanged parameters from "previous" and list
them in "inherit". If a required parameter is missing and not in "previous", set clarify. If a cut is not
on the five-percent grid, set clarify to out_of_grid. Default the reporting year to the page's last year
and the start to the first start only when the question gives no year at all. answer_lang is the question's
language (zh for Chinese).

What each claim status allows: S, D, N may be answered; L only as labelled model internals; W never.
Questions whose answer would be a W quantity must be refused with the matching reason.

Presets:
{docs}

Refusal reasons (use these instead of a preset):
{refs}

Water-use classes (key = English / Chinese): {classes}
Townships (English / Chinese; the first is the pooled Changhua panel, only for its labelled spillover row): {towns}
Start years: {", ".join(str(s) for s in c.starts)}. Reporting years: {c.years[c.YR + 1]} to {c.years[-1]}.

Claims the presets touch:
{topics}
"""


class LLMRouter:
    def __init__(self, b: PageBundle, backend, *, presets: tuple[str, ...], effort: str = "low"):
        self.b, self.backend, self.presets, self.effort = b, backend, presets, effort
        self.system = system_prompt(b, presets)
        self.schema = route_schema(b, presets)
        self.last = None

    def request(self, question: str, prev: dict | None) -> tuple[list, list]:
        system = [{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}]
        user = json.dumps({"question": question, "previous": prev}, ensure_ascii=False)
        tail = json.dumps({"page_sha256": self.b.page_sha256[:16], "ledger_sha256": self.b.ledger_sha256[:16]})
        return system, [{"role": "user", "content": [{"type": "text", "text": user},
                                                     {"type": "text", "text": tail}]}]

    def route(self, question: str, prev: dict | None = None) -> Route:
        system, messages = self.request(question, prev)
        r = self.backend.complete(system, messages, effort=self.effort, max_tokens=1024,
                                  output_format={"type": "json_schema", "schema": self.schema})
        self.last = r
        if r.stop_reason == "refusal":
            return Route(preset="refuse", params={"reason": "provider_refusal"}, refusal="provider_refusal",
                         confidence=1.0, source="llm")
        try:
            d = json.loads(r.text)
        except (json.JSONDecodeError, TypeError):
            return Route(clarify={"reason": "unknown", "candidates": []}, source="llm")
        rt = from_llm_json(d, self.b)
        if rt.refusal == "no_policy":          # same meaning as the rules' clarification: ask for the cut
            rt.refusal, rt.clarify = None, {"reason": "missing_cut", "candidates": []}
        if rt.refusal:
            rt.preset, rt.params = "refuse", {"reason": rt.refusal}
        elif rt.clarify:
            pass                                   # answered as a clarification (answer_route)
        elif rt.preset and rt.preset not in self.presets:
            rt.preset, rt.params, rt.refusal = "refuse", {"reason": "disabled"}, "disabled"
        if prev and rt.inherit:
            for k in rt.inherit:
                if k not in rt.params and k in (prev.get("params") or {}):
                    rt.params[k] = prev["params"][k]
        return rt
