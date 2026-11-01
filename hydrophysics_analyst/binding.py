"""Claim binding: which ledger claims govern each query value, and what it may be called.

Two sources must agree mechanically (spec §6): the ledger's ``display_map`` (payload paths,
arrays and JS functions -> claim ids) and ``rules_v1.json`` (quantity, region, aggregation ->
claims, class, required page label). A rule may only cite claims the display map gives for the
value's paths and functions; anything else is a ledger gap for the lead to rule on.

Status -> class: any W -> ``withheld``; the cost tile C08 -> ``labelled`` (the ledger allows it in
the answer row with its own page label, ``page_scoping.answer_tiles_allowed_note``); any other L
-> ``internal`` (only beside its page label); otherwise ``answer``.
"""
from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from hydrophysics.twin import claims as C

from .page import PageBundle
from .query import QueryResult, Snippet, Value

RULES = Path(__file__).with_name("rules_v1.json")


@dataclass(frozen=True)
class Rule:
    quantity: str
    region: str
    aggregation: str
    claims: tuple[str, ...]
    answer_class: str
    required_label: tuple[str, ...]
    offered: bool
    verdict_ban: bool = False


class LedgerGap(LookupError):
    """A query value has no rule, or its rule is not offered."""


@lru_cache(maxsize=4)
def load_rules(path: str = str(RULES)) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rules() -> dict[tuple[str, str, str], Rule]:
    out = {}
    for r in load_rules()["rules"]:
        out[(r["quantity"], r["region"], r["aggregation"])] = Rule(
            r["quantity"], r["region"], r["aggregation"], tuple(r["claims"]), r["answer_class"],
            tuple(r["required_label"]), bool(r["offered"]), bool(r.get("verdict_ban")))
    return out


def rule_for(quantity: str, region: str, aggregation: str) -> Rule:
    r = rules().get((quantity, region, aggregation))
    if r is None:
        raise LedgerGap(f"no rule for {quantity}/{region}/{aggregation}")
    return r


def ledger_ids(paths, functions, ledger: dict) -> set[str]:
    dm = ledger["display_map"]
    ids: set[str] = set()
    for p in paths:
        if p.startswith("arrays."):
            name = p.split(".", 1)[1]
            for pat, v in dm["arrays"].items():
                if fnmatch.fnmatchcase(name, pat):
                    ids.update(v)
        else:
            ids.update(dm["payload"].get(p, []))
    for fn in functions:
        ids.update(dm["functions"].get(fn, []))
    return ids


def class_of(claim_ids, ledger: dict) -> str:
    st = {c["id"]: c["status_current"] for c in ledger["claims"]}
    codes = {code for cid in claim_ids for code in C.codes(st[cid])}
    if "W" in codes:
        return "withheld"
    if "L" in codes:
        return "labelled" if set(claim_ids) == {"C08"} else "internal"
    return "answer"


def status_of(claim_ids, ledger: dict) -> tuple[tuple[str, str], ...]:
    st = {c["id"]: c["status_current"] for c in ledger["claims"]}
    return tuple((cid, st[cid]) for cid in claim_ids)


def snippet_claims(key: str, ledger: dict) -> tuple[str, ...]:
    i18n = ledger["display_map"]["i18n"]
    ids: list[str] = []
    for k in key.split("+"):
        for cid in i18n.get(k, []):
            if cid not in ids:
                ids.append(cid)
    return tuple(ids)


def bind_value(v: Value, ledger: dict) -> Value:
    r = rule_for(v.quantity, v.region, v.aggregation)
    if not r.offered:
        raise LedgerGap(f"{v.quantity}/{v.region}/{v.aggregation} is not offered")
    return replace(v, claims=r.claims, status=status_of(r.claims, ledger),
                   answer_class=class_of(r.claims, ledger))


def bind(res: QueryResult, b: PageBundle) -> QueryResult:
    vals = tuple(bind_value(v, b.ledger) for v in res.values)
    snips = tuple(replace(s, claims=snippet_claims(s.key, b.ledger)) for s in res.snippets)
    return replace(res, values=vals, snippets=snips)


def required_labels(v: Value) -> tuple[str, ...]:
    return rule_for(v.quantity, v.region, v.aggregation).required_label


def scoring() -> dict:
    return load_rules()["scoring"]


def topics() -> dict:
    return load_rules()["topics"]


def lexicon() -> dict:
    return load_rules()["lexicon"]


def is_snippet(x) -> bool:
    return isinstance(x, Snippet)
