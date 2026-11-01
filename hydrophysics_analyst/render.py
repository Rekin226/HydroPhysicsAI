"""Answers as spans: page sentences, page-formatted values, reviewed frames and labels.

No span is ever written by an LLM. ``snippet`` spans are page ``T`` strings (or ledger label
strings) rendered with the page's ``tr``; ``value`` spans are query values formatted as the page
formats them; ``frame``/``label``/``echo`` spans come from ``strings.json``. The validator
(``validate.py``) treats snippet and value spans as trusted by construction and checks the rest.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from . import jsport as J
from .page import PageBundle
from .query import QueryResult

STRINGS = Path(__file__).with_name("strings.json")
PAGE_KINDS = ("snippet", "value")


@lru_cache(maxsize=1)
def strings() -> dict:
    return json.loads(STRINGS.read_text(encoding="utf-8"))


def S(lang: str, key: str) -> str:
    st = strings()
    return st.get(lang, st["en"]).get(key) or st["en"][key]


@dataclass(frozen=True)
class Span:
    kind: str                       # snippet | value | frame | label | echo
    text: str
    key: str = ""                   # i18n key / strings.json key / value key
    claims: tuple[str, ...] = ()
    cls: str = ""                   # for values: answer | labelled | internal
    block: str = "answer"           # answer | cost | internals | refusal | meta
    inline: bool = True             # values: shown inside their sentence (else on their own)
    unit: str = ""


@dataclass
class Answer:
    preset: str
    lang: str
    params: dict
    echo: tuple[Span, ...]
    spans: tuple[Span, ...]
    citations: tuple[tuple[str, str], ...]
    deep_link: str
    banners: tuple[str, ...] = ()
    fallback: str | None = None
    refusal: str | None = None
    clarify: dict | None = None
    results: tuple[QueryResult, ...] = field(default_factory=tuple, repr=False)
    table: tuple[tuple[str, ...], ...] = ()

    def text(self) -> str:
        """What the user reads (tags stripped), echo first."""
        sep = "\n"
        parts = [echo_text(self.echo, self.lang)] if self.echo else []
        parts += [S(self.lang, b) for b in self.banners]
        cur, block = [], None
        for s in self.spans:
            if s.kind == "value" and s.inline:
                continue                  # values appear inside their snippets
            if block is not None and s.block != block and cur:
                parts.append(" ".join(cur))
                cur = []
            block = s.block
            cur.append(J.strip_tags(s.text) + (f" {s.unit}" if s.kind == "value" and s.unit else ""))
        if cur:
            parts.append(" ".join(cur))
        if self.table:
            parts += ["  ".join(r) for r in self.table]
        if self.citations:
            parts.append(" ".join(f"[{c} {st}]" for c, st in self.citations))
        if self.deep_link:
            parts.append(f"{S(self.lang, 'frame_open_page')}: {self.deep_link}")
        return sep.join(p for p in parts if p)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("results", None)
        d["text"] = self.text()
        return d


def echo_text(spans, lang: str) -> str:
    return S(lang, "echo_preset") + ": " + S(lang, "echo_sep").join(J.strip_tags(s.text) for s in spans)


def citations_of(spans) -> tuple[str, ...]:
    """The claims that govern what is stated: the numbers shown (value spans) and, for a
    refusal, the claims it refuses on. Caveat sentences that merely mention other claims
    (``mi_note``, ``rank_limit``) add no citation, so a W claim is never listed beside an answer."""
    return tuple(dict.fromkeys(c for s in spans if s.kind == "value" or s.block == "refusal"
                               for c in s.claims))


def render_echo(b: PageBundle, preset: str, params: dict, lang: str) -> tuple[Span, ...]:
    """The interpretation, built from page strings: preset name, policy (``policyPhrase``),
    start month (``jan``), year (``dec``), quantity and scope (``hw_n2``/``tHead``), town."""
    from . import query as Q
    c = Q.ctx(b)
    out = [Span("label", S(lang, f"preset_{preset}"), key=f"preset_{preset}", block="meta")]
    if "cuts" in params or "factors" in params:
        f = params.get("factors") or Q.Policy.from_cuts(cuts_dict(params["cuts"], b), params.get("start", c.starts[0]), c.K).f
        out.append(Span("snippet", f"{S(lang, 'echo_policy')}: " + Q.policy_phrase(b, f, lang),
                        key="pp", block="meta"))
    if params.get("start") is not None and preset != "start_compare":
        out.append(Span("snippet", f"{S(lang, 'echo_start')} " + c.tr(lang, "jan", {"y": params["start"]}),
                        key="jan", block="meta"))
    if params.get("year") is not None:
        out.append(Span("snippet", f"{S(lang, 'echo_year')} " + c.tr(lang, "dec", {"y": params["year"]}),
                        key="dec", block="meta"))
    if preset in ("policy_head", "start_compare", "solved_compare", "cheapest_for_head") and c.n2u:
        out.append(Span("snippet", f"{S(lang, 'echo_scope')}: " + c.tr(lang, "tHead"), key="tHead", block="meta"))
    if params.get("by"):
        out.append(Span("label", f"{S(lang, 'echo_metric')}: " + S(lang, f"metric_{params['by']}"),
                        key="metric", block="meta"))
    if params.get("town") is not None:
        i = int(params["town"])
        tw = c.towns[i]
        name = c.TN(i, lang) if tw.get("pooled") else f"{tw['zh']} {tw['en']}"
        if tw.get("pooled") and c.nochp():
            # C02 spillover label beside any Changhua name (changhua_lever_rule)
            name += " (" + b.ledger["page_scoping"]["changhua_lever_rule"]["labels"][lang][2] + ")"
        out.append(Span("snippet", f"{S(lang, 'echo_town')}: {name}", key="town_name", block="meta"))
    return tuple(out)


def cuts_dict(cuts, b: PageBundle | None = None) -> dict[int, int]:
    """Route cuts (list of {class, pct} or {class_key: pct}) -> {class index: pct}."""
    from . import query as Q
    keys = [cl["key"] for cl in Q.ctx(b).classes] if b is not None else None
    out: dict[int, int] = {}
    items = cuts.items() if isinstance(cuts, dict) else [(x["class"], x["pct"]) for x in cuts]
    for k, pct in items:
        idx = k if isinstance(k, int) else keys.index(k)
        out[int(idx)] = int(pct)
    return out


def spans_from(res: QueryResult, *, block: str = "answer", keys: tuple[str, ...] | None = None,
               lead: Span | None = None) -> list[Span]:
    """Snippet spans (optionally a subset by key) and the value spans they carry."""
    vals = {v.key: v for v in res.values}
    out: list[Span] = [lead] if lead else []
    for s in res.snippets:
        if keys is not None and s.key not in keys and not any(s.key.startswith(k) for k in keys):
            continue
        out.append(Span("snippet", s.text, key=s.key, claims=s.claims, block=block))
        for vk in s.values:
            v = vals[vk]
            out.append(Span("value", v.text, key=v.key, claims=v.claims, cls=v.answer_class, block=block))
    return out
