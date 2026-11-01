"""The fixed, tested questions the analyst answers (spec §7).

Each preset validates its parameters against the bound page, runs the deterministic query
layer, binds the values to claims, and composes an :class:`~render.Answer` from page sentences.
v1 presets: policy_head, start_compare, lever_rank, township_internals, rail_internals,
solved_compare, refuse. ``cheapest_for_head`` is built but off until ruling D6
(``ANALYST_ENABLE_CHEAPEST=1``); ``how_sure`` is a phase-5 preset.

Parameter schemas are JSON Schema in the subset structured outputs supports (``anyOf`` only,
``additionalProperties: false`` everywhere, enums instead of numeric bounds); the LLM router
reuses them.
"""
from __future__ import annotations

import os
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from . import binding as BD
from . import query as Q
from .page import PageBundle
from .render import (
    Answer,
    Chart,
    ChartSeries,
    S,
    Span,
    citations_of,
    cuts_dict,
    render_echo,
    spans_from,
)

PCT = list(range(0, 101, 5))
REFUSALS = ("changhua_lever", "subsidence_size", "area", "recommendation", "volumes_money",
            "observations", "beyond_horizon", "out_of_domain", "provider_refusal",
            "township_head", "no_policy", "no_run", "disabled")
V1 = ("policy_head", "start_compare", "lever_rank", "township_internals", "rail_internals",
      "solved_compare", "refuse")


class ParamError(ValueError):
    """Parameters that are not on the page's grid; becomes a clarification, never coerced."""

    def __init__(self, reason: str, candidates: list | None = None):
        super().__init__(reason)
        self.reason, self.candidates = reason, candidates or []


# ---------------------------------------------------------------------------------------
# schemas (built per bundle: enums come from the payload)
# ---------------------------------------------------------------------------------------
def enums(b: PageBundle) -> dict:
    c = Q.ctx(b)
    return {"classes": [cl["key"] for cl in c.classes], "years": c.years[c.YR + 1:],
            "starts": c.starts, "towns": [t["en"] for t in c.towns] + [t["zh"] for t in c.towns],
            "solved": [s["name"] for s in c.solved]}


def _cuts_schema(e: dict) -> dict:
    return {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                       "required": ["class", "pct"],
                                       "properties": {"class": {"type": "string", "enum": e["classes"]},
                                                      "pct": {"type": "integer", "enum": PCT}}}}


def schemas(b: PageBundle) -> dict[str, dict]:
    e = enums(b)
    year = {"type": "integer", "enum": e["years"]}
    start = {"type": "integer", "enum": e["starts"]}

    def obj(props: dict) -> dict:
        return {"type": "object", "additionalProperties": False, "required": list(props),
                "properties": props}
    return {
        "policy_head": obj({"cuts": _cuts_schema(e), "start": start, "year": year}),
        "start_compare": obj({"cuts": _cuts_schema(e), "year": year}),
        "lever_rank": obj({"year": year, "by": {"type": "string", "enum": ["head", "gwh"]}}),
        "township_internals": obj({"cuts": _cuts_schema(e), "start": start, "year": year,
                                   "town": {"type": "string", "enum": e["towns"]}}),
        "rail_internals": obj({"cuts": _cuts_schema(e), "start": start, "year": year}),
        "solved_compare": obj({"year": year}),
        "cheapest_for_head": obj({"target_m": {"type": "number", "enum": [0.1, 0.2, 0.25, 0.3, 0.4,
                                                                          0.5, 0.75, 1.0]},
                                  "start": start, "year": year}),
        "refuse": obj({"reason": {"type": "string", "enum": list(REFUSALS)}}),
    }


# ---------------------------------------------------------------------------------------
# parameter checks
# ---------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    for ch in " -_()（）鄉鎮市縣":
        s = s.replace(ch, "")
    return s


def resolve_town(b: PageBundle, town) -> int:
    c = Q.ctx(b)
    if isinstance(town, int):
        if 0 <= town < len(c.towns):
            return town
        raise ParamError("ambiguous_town")
    t = str(town)
    for i, tw in enumerate(c.towns):
        if t in (tw["en"], tw["zh"]):
            return i
    n = _norm(t)
    hits = [i for i, tw in enumerate(c.towns) if n and (n in _norm(tw["en"]) or n in _norm(tw["zh"])
                                                        or _norm(tw["en"]).startswith(n))]
    if len(hits) == 1:
        return hits[0]
    raise ParamError("ambiguous_town", [c.towns[i]["en"] for i in hits] or [t["en"] for t in c.towns])


NEEDS = {"policy_head": ("start", "year"), "start_compare": ("year",), "lever_rank": ("year", "by"),
         "township_internals": ("start", "year"), "rail_internals": ("start", "year"),
         "solved_compare": ("year",), "cheapest_for_head": ("start", "year"), "how_sure": ()}


def _check(b: PageBundle, preset: str, p: dict) -> dict:
    c = Q.ctx(b)
    p = {k: v for k, v in dict(p).items() if v is not None}
    # the page's own defaults (S.start = first start, S.year = horizon, S.rank = "head")
    defaults = {"start": c.starts[0], "year": c.years[-1], "by": "head"}
    for k in NEEDS.get(preset, ()):
        p.setdefault(k, defaults[k])
    if "year" in p:
        if int(p["year"]) > c.years[-1]:
            raise ParamError("beyond_horizon")
        if int(p["year"]) not in c.years:
            raise ParamError("out_of_grid")
        p["year"] = int(p["year"])
    if "start" in p:
        if int(p["start"]) not in c.starts:
            raise ParamError("out_of_grid", c.starts)
        p["start"] = int(p["start"])
    if preset in ("township_internals", "rail_internals", "start_compare") and not p.get("cuts"):
        raise ParamError("missing_cut")                   # never fall through to the baseline
    if "cuts" in p:
        raw = p["cuts"] if isinstance(p["cuts"], list) else [{"class": k, "pct": v} for k, v in p["cuts"].items()]
        if any(x.get("pct") is None for x in raw):
            raise ParamError("missing_cut")               # a class named without a size
        d = cuts_dict(p["cuts"], b)
        if not any(d.values()) and preset in ("township_internals", "rail_internals", "start_compare"):
            raise ParamError("missing_cut")            # policy_head without cuts = the baseline
        if any(v not in PCT for v in d.values()) or any(k >= c.K for k in d):
            raise ParamError("out_of_grid")
        p["cuts"] = [{"class": c.classes[k]["key"], "pct": v} for k, v in sorted(d.items()) if v]
    if "town" in p:
        p["town"] = resolve_town(b, p["town"])
    return p


# ---------------------------------------------------------------------------------------
# composition helpers
# ---------------------------------------------------------------------------------------
def _policy(b: PageBundle, p: dict, start: int | None = None) -> Q.Policy:
    c = Q.ctx(b)
    return Q.Policy.from_cuts(cuts_dict(p.get("cuts", []), b), start or p.get("start", c.starts[0]), c.K)


def _head_spans(res: Q.QueryResult) -> list[Span]:
    out = spans_from(res, keys=("hl_",))
    out += spans_from(res, keys=("tW_r",), block="meta")
    out += spans_from(res, keys=("prov_",), block="meta")
    return out


def _cost_spans(b: PageBundle, res: Q.QueryResult, lang: str) -> list[Span]:
    vals = {v.key: v for v in res.values}
    if "gwh" not in vals:
        return spans_from(res, keys=("no_energy",), block="cost")
    g = vals["gwh"]
    out = spans_from(res, keys=("tCost",), block="cost")
    out.append(Span("value", g.text, key=g.key, claims=g.claims, cls=g.answer_class, block="cost",
                    inline=False, unit="GWh/yr"))
    out += spans_from(res, keys=("tC_r", "tC_d", "tC_metered"), block="cost")
    return out


def _answer(b: PageBundle, preset: str, params: dict, lang: str, spans: list[Span],
            results: list[Q.QueryResult], link: str, *, refusal=None, table=(), clarify=None,
            charts=()) -> Answer:
    cites = citations_of(spans)
    st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
    return Answer(preset=preset, lang=lang, params=params,
                  echo=render_echo(b, preset, params, lang) if preset != "refuse" else (),
                  spans=tuple(spans), citations=tuple((cid, st[cid]) for cid in cites),
                  deep_link=link, banners=tuple(banners(b)), refusal=refusal,
                  results=tuple(results), table=tuple(table), clarify=clarify, charts=tuple(charts))


# ---------------------------------------------------------------------------------------
# charts: drawn by the shell from the answer's own values (no numbers of their own)
# ---------------------------------------------------------------------------------------
def _series_refs(res: Q.QueryResult, i: int, prefix: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
    vs = [v for v in res.values if v.key.startswith(prefix) and v.key[len(prefix):].isdigit()]
    return tuple(int(v.key[len(prefix):]) for v in vs), tuple(f"r{i}:{v.key}" for v in vs)


def head_line(b: PageBundle, results: list[Q.QueryResult], lang: str, *, starts=None) -> list[Chart]:
    """Head change at N2's wells by year: one series per policy result (or per start)."""
    c = Q.ctx(b)
    series, xs, marks = [], (), []
    for i, res in enumerate(results):
        for s0 in (starts or [None]):
            prefix = f"head_{s0}_y" if s0 else "head_y"
            x, refs = _series_refs(res, i, prefix)
            if not refs:
                continue
            xs = x
            st = s0 or (res.policy.start if res.policy else None)
            label = c.tr(lang, "from_y", {"y": st}) if starts else c.tr(lang, "jan", {"y": st})
            series.append(ChartSeries(label, refs))
            if st and (st, c.tr(lang, "jan", {"y": st})) not in marks:
                marks.append((st, c.tr(lang, "jan", {"y": st})))
    if not series:
        return []
    key = "tHead" if c.n2u else "tHead_l2"
    return [Chart("line", c.tr(lang, key), key, "m", tuple(series), x=xs, marker=tuple(marks))]


def cost_share(b: PageBundle, res: Q.QueryResult, i: int, lang: str) -> list[Chart]:
    c = Q.ctx(b)
    if not any(v.key == "gwh_pct" for v in res.values) or res.src in ("base", "none"):
        return []
    return [Chart("share", c.tr(lang, "tCost"), "tCost", "%", (ChartSeries("", (f"r{i}:gwh_pct",)),),
                  block="cost")]


def banners(b: PageBundle) -> list[str]:
    import datetime as dt
    out = []
    if "dev_pending" in b.banners:
        out.append("banner_dev_pending")
    if "dev_unbound" in b.banners:
        out.append("banner_dev_unbound")
    today = dt.date.fromisoformat(os.environ.get("ANALYST_TODAY", dt.date.today().isoformat()))
    st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
    for cid, s in BD.scoring().items():
        if cid in st and "D" in st[cid].split("+") and today >= dt.date.fromisoformat(s["not_before"]):
            out.append("banner_stale_wording")
            break
    return out


# ---------------------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------------------
def policy_head(b, p, lang):
    pol = _policy(b, p)
    if pol.is_base():
        res = BD.bind(Q.headline(b, pol, p["year"], lang), b)
        return _answer(b, "policy_head", p, lang, spans_from(res, keys=("hl_",)), [res], res.deep_link)
    res = BD.bind(Q.headline(b, pol, p["year"], lang), b)
    if res.src == "none":
        return _answer(b, "policy_head", p, lang, spans_from(res, keys=("hl_none", "prov_")), [res],
                       res.deep_link)
    spans = _head_spans(res) + _cost_spans(b, res, lang)
    charts = head_line(b, [res], lang) + cost_share(b, res, 0, lang)
    return _answer(b, "policy_head", p, lang, spans, [res], res.deep_link, charts=charts)


def start_compare(b, p, lang):
    c = Q.ctx(b)
    spans, results = [], []
    for s0 in c.starts:
        res = BD.bind(Q.headline(b, _policy(b, p, s0), p["year"], lang), b)
        results.append(res)
        spans += spans_from(res, keys=("hl_",))
    note = BD.bind(Q.start_compare(b, _policy(b, p, c.starts[0]).f, p["year"], lang), b)
    results.append(note)
    spans += spans_from(note, keys=("start_note",))
    if results[0].src != "none":
        spans += spans_from(results[0], keys=("tW_r",))
    charts = head_line(b, [note], lang, starts=c.starts)
    # the note result is last in ``results``; re-index its refs
    charts = [Chart(ch.kind, ch.title, ch.title_key, ch.unit,
                    tuple(ChartSeries(sr.label, tuple(r.replace("r0:", f"r{len(results) - 1}:", 1)
                                                      for r in sr.refs)) for sr in ch.series),
                    x=ch.x, marker=ch.marker, block=ch.block) for ch in charts]
    return _answer(b, "start_compare", p, lang, spans, results, results[0].deep_link, charts=charts)


def lever_rank(b, p, lang):
    c = Q.ctx(b)
    res = BD.bind(Q.rank(b, p["year"], p["by"], lang), b)
    lead = Span("label", S(lang, "label_model_internals"), key="label_model_internals", block="internals")
    spans = spans_from(res, keys=("rank_unit",), block="internals", lead=lead)
    spans += [Span("value", v.text, key=v.key, claims=v.claims, cls=v.answer_class, block="internals")
              for v in res.values]
    spans += spans_from(res, keys=("rank_limit",), block="internals")
    spans.append(Span("snippet", c.tr(lang, "ev_decision"), key="ev_decision", block="internals"))
    link = Q.deep_link(b, Q.Policy((1.0,) * c.K, c.starts[0]), p["year"], lang=lang, tab="rank")
    cats = tuple(row[0] for row in res.table[1:])
    order = [k for k in range(c.K) if c.CN(k, lang) in cats]
    order.sort(key=lambda k: cats.index(c.CN(k, lang)))
    series = tuple(ChartSeries(c.tr(lang, "from_y", {"y": s0}),
                               tuple(f"r0:rank_{k}_{s0}" for k in order)) for s0 in c.starts)
    unit_snip = next((s.text for s in res.snippets if s.key.startswith("rank_unit")), "")
    unit_key = next((s.key for s in res.snippets if s.key.startswith("rank_unit")), "")
    charts = [Chart("bars", unit_snip, unit_key, "m" if p["by"] == "head" else "m/100 GWh/yr", series,
                    x=cats, block="internals")]
    return _answer(b, "lever_rank", p, lang, spans, [res], link, table=res.table, charts=charts)


def township_internals(b, p, lang):
    pol = _policy(b, p)
    if pol.is_base():
        return refuse(b, {"reason": "no_policy"}, lang)
    res = BD.bind(Q.township(b, pol, p["year"], p["town"], lang), b)
    if not res.values:
        return refuse(b, {"reason": "no_run"}, lang)
    spans = spans_from(res, keys=("mi_note", "town_name", "tw_"), block="internals")
    link = Q.deep_link(b, pol, p["year"], lang=lang, sel=p["town"], tab="towns", mi=True)
    return _answer(b, "township_internals", p, lang, spans, [res], link)


def rail_internals(b, p, lang):
    pol = _policy(b, p)
    head = BD.bind(Q.headline(b, pol, p["year"], lang), b)
    spans = spans_from(head, keys=("hl_",)) + spans_from(head, keys=("tW_r",))
    rail = BD.bind(Q.rail(b, pol, p["year"], lang), b)
    spans.append(Span("frame", S(lang, "frame_rail_verdict"), key="frame_rail_verdict", block="internals"))
    spans += spans_from(rail, keys=("rail_note", "no_rail"), block="internals")
    link = Q.deep_link(b, pol, p["year"], lang=lang, tab="hsr")
    return _answer(b, "rail_internals", p, lang, spans, [head, rail], link)


def solved_compare(b, p, lang):
    c = Q.ctx(b)
    spans, results = [], []
    for x in c.solved:
        res = BD.bind(Q.headline(b, Q.Policy(tuple(float(v) for v in x["factors"]), int(x["start"])),
                                 p["year"], lang), b)
        results.append(res)
        spans += _head_spans(res) + _cost_spans(b, res, lang)
    if not results:
        return refuse(b, {"reason": "no_run"}, lang)
    cats = tuple(Q.policy_phrase(b, x["factors"], lang) for x in c.solved)
    charts = [Chart("bars", c.tr(lang, "tHead" if c.n2u else "tHead_l2"), "tHead" if c.n2u else "tHead_l2",
                    "m", (ChartSeries(c.tr(lang, "dec", {"y": p["year"]}),
                                      tuple(f"r{i}:head" for i in range(len(results)))),), x=cats)]
    if all(any(v.key == "gwh" for v in r.values) for r in results):
        charts.append(Chart("bars", c.tr(lang, "tCost"), "tCost", "GWh/yr",
                            (ChartSeries("", tuple(f"r{i}:gwh" for i in range(len(results)))),),
                            x=cats, block="cost"))
    return _answer(b, "solved_compare", p, lang, spans, results, results[0].deep_link, charts=charts)


def cheapest_for_head(b, p, lang):
    if os.environ.get("ANALYST_ENABLE_CHEAPEST") != "1":
        return refuse(b, {"reason": "disabled"}, lang)
    res = Q.cheapest_for_head(b, float(p["target_m"]), p["year"], p["start"], lang)
    if res.policy is None:
        return refuse(b, {"reason": "no_run"}, lang)
    res = BD.bind(res, b)
    spans = [Span("frame", S(lang, "frame_cheapest"), key="frame_cheapest")]
    spans += _head_spans(res) + _cost_spans(b, res, lang)
    q = dict(p) | {"factors": res.policy.f}
    charts = head_line(b, [res], lang) + cost_share(b, res, 0, lang)
    return _answer(b, "cheapest_for_head", q, lang, spans, [res], res.deep_link, charts=charts)


def how_sure(b, p, lang):
    """How far the result is tested: the page's evidence line and registered-test lines. W
    claims these lines merely mention are not cited (a W claim is never listed beside an answer)."""
    import dataclasses as dc
    import datetime as dt

    from hydrophysics.twin import claims as C
    c = Q.ctx(b)
    res = BD.bind(Q.evidence(b, lang), b)
    st = {x["id"]: x["status_current"] for x in b.ledger["claims"]}
    spans = []
    for s in spans_from(res, block="evidence"):
        keep = tuple(x for x in s.claims if "W" not in C.codes(st.get(x, "")))
        spans.append(dc.replace(s, claims=keep))
    today = dt.date.fromisoformat(os.environ.get("ANALYST_TODAY", dt.date.today().isoformat()))
    due = [cid for cid, sc in BD.scoring().items()
           if "D" in C.codes(st.get(cid, "")) and today >= dt.date.fromisoformat(sc["not_before"])]
    if due:
        spans.append(Span("frame", S(lang, "frame_scoring_due"), key="frame_scoring_due", block="meta"))
    link = Q.deep_link(b, Q.Policy((1.0,) * c.K, c.starts[0]), c.years[-1], lang=lang, tab="trust")
    return _answer(b, "how_sure", p, lang, spans, [res], link)


REFUSE_PAGE = {   # the page's own sentence shown with a refusal, and the claims it cites
    "changhua_lever": ("ledger:changhua_label", ("C13a", "E20")),
    "subsidence_size": ("more_metrics", ("C03",)),
    "area": ("more_metrics", ("C05",)),
    "recommendation": ("ev_decision", ()),
    "beyond_horizon": ("year_end", ()),
}


def refuse(b, p, lang):
    c = Q.ctx(b)
    reason = p["reason"]
    spans = [Span("frame", S(lang, f"refuse_{reason}"), key=f"refuse_{reason}", block="refusal")]
    key, cl = REFUSE_PAGE.get(reason, (None, ()))
    if key == "ledger:changhua_label":
        lab = b.ledger["page_scoping"]["changhua_lever_rule"]["labels"][lang][0]
        spans.append(Span("snippet", lab, key="changhua_label", claims=cl, block="refusal"))
    elif key == "more_metrics":
        spans.append(Span("snippet", c.tr(lang, "more_metrics", {"n": 4}), key=key, claims=cl, block="refusal"))
    elif key == "year_end":
        spans.append(Span("snippet", c.tr(lang, "year_end", {"y": c.years[-1]}), key=key, block="refusal"))
    elif key:
        spans.append(Span("snippet", c.tr(lang, key), key=key, claims=cl, block="refusal"))
    layer = "dsubs" if reason in ("subsidence_size", "area") else "dhead"
    link = (Q.deep_link(b, Q.Policy((1.0,) * c.K, c.starts[0]), c.years[-1], layer=layer, lang=lang,
                        mi=reason in ("subsidence_size", "area"))
            if reason in ("subsidence_size", "area", "recommendation", "beyond_horizon") else "")
    return _answer(b, "refuse", p, lang, spans, [], link, refusal=reason)


def clarification(b: PageBundle, preset: str | None, params: dict, reason: str, candidates, lang: str) -> Answer:
    key = reason if reason in ("ambiguous_town", "missing_year", "missing_cut", "out_of_grid") else "unknown"
    if reason == "beyond_horizon":
        return refuse(b, {"reason": "beyond_horizon"}, lang)
    spans = [Span("frame", S(lang, f"clarify_{key}"), key=f"clarify_{key}", block="meta")]
    return Answer(preset="clarify", lang=lang, params=dict(params) | {"for_preset": preset},
                  echo=(), spans=tuple(spans), citations=(), deep_link="", banners=tuple(banners(b)),
                  clarify={"reason": key, "candidates": list(candidates or [])})


def examples(b: PageBundle) -> list[dict]:
    """Clickable example questions for the shell, built from the bound page: each is a preset with
    parameters (answered without the LLM) and a label made of page strings in en and zh."""
    c = Q.ctx(b)
    keys = [cl["key"] for cl in c.classes]
    Y, s0 = c.years[-1], c.starts[0]
    s1 = c.starts[1] if len(c.starts) > 1 else s0

    def cut(k, pct):
        return [{"class": keys[k], "pct": pct}] if k < len(keys) else []
    towns = [i for i, t in enumerate(c.towns) if not t.get("pooled")]
    town = next((i for i in towns if c.towns[i]["en"] == "Tuku"), towns[0] if towns else None)
    ex = [("policy_head", {"cuts": cut(0, 30), "start": s0, "year": Y}),
          ("policy_head", {"cuts": cut(1, 100), "start": s0, "year": Y}),
          ("policy_head", {"cuts": cut(1, 30), "start": s1, "year": Y}),
          ("policy_head", {"cuts": [{"class": k, "pct": 20} for k in keys], "start": s0, "year": Y}),
          ("start_compare", {"cuts": cut(1, 100), "year": Y}),
          ("lever_rank", {"year": Y, "by": "head"}),
          ("lever_rank", {"year": Y, "by": "gwh"}),
          ("rail_internals", {"cuts": cut(0, 30), "start": s0, "year": Y}),
          ("solved_compare", {"year": Y}),
          ("how_sure", {})]
    if town is not None:
        ex.insert(7, ("township_internals", {"cuts": cut(1, 100), "start": s0, "year": Y, "town": town}))
    out = []
    for preset, params in ex:
        lab = {}
        for lang in ("en", "zh"):
            echo = render_echo(b, preset, params, lang)
            parts = [_strip(sp.text.split(": ", 1)[-1]) for sp in echo[1:] if sp.key not in ("tHead",)]
            lab[lang] = " · ".join(parts) if parts else S(lang, f"preset_{preset}")
        out.append({"preset": preset, "params": params, "label": lab,
                    "group": {lg: S(lg, f"preset_{preset}") for lg in ("en", "zh")}})
    return out


def _strip(t: str) -> str:
    from .jsport import strip_tags
    return strip_tags(t)


@dataclass(frozen=True)
class Preset:
    name: str
    run: Callable[[PageBundle, dict, str], Answer]


PRESETS: dict[str, Preset] = {n: Preset(n, f) for n, f in (
    ("policy_head", policy_head), ("start_compare", start_compare), ("lever_rank", lever_rank),
    ("township_internals", township_internals), ("rail_internals", rail_internals),
    ("solved_compare", solved_compare), ("cheapest_for_head", cheapest_for_head), ("how_sure", how_sure),
    ("refuse", refuse))}


def run(b: PageBundle, preset: str, params: dict, lang: str = "en") -> Answer:
    """Validate params on the page grid and answer; bad params become a clarification."""
    if preset not in PRESETS:
        return clarification(b, None, params, "unknown", [], lang)
    try:
        p = _check(b, preset, params)
    except ParamError as e:
        return clarification(b, preset, params, e.reason, e.candidates, lang)
    return PRESETS[preset].run(b, p, lang)
