"""The shell's question library: grouped example questions, try-a-refusal questions, and the
claim-status legend.

Display only. Every preset example is a preset with parameters on the bound page's grid, answered
through ``/api/answer`` (no LLM), exactly like :func:`presets.examples`, whose chips it extends and
regroups. The labels are plain questions built from the page's own class and township names; the
answer, its echo and its claims still come from the page. The try-a-refusal questions go through
``/api/ask`` and are refused by the deterministic guards before any LLM call
(``tests/test_analyst_showcase.py`` checks that with the LLM off).

Kept out of ``eval.ROUTE_CODE`` on purpose: adding or rewording an example changes no route and
must not move the eval's code pins.
"""

from __future__ import annotations

from . import query as Q
from .page import PageBundle

# (id, title en, title zh, one-line description en, zh). Order is the library's order.
GROUPS = (
    ("one_use", "Cut one water use", "減少單一用水",
     "The answer the page supports: the Yunlin head response to a pumping cut.",
     "頁面支持的答案：減少抽水後雲林的地下水位反應。"),
    ("size", "How big a cut?", "減抽幅度",
     "One water use at different cut sizes.",
     "同一類用水，不同的減抽幅度。"),
    ("combine", "Combine cuts", "組合減抽",
     "Several water uses at once, including the page's headline lever.",
     "同時減少多類用水，包括頁面的主要情境。"),
    ("timing", "Start now or later", "現在或稍後開始",
     "The same cut from each start year.",
     "同樣的減抽，從不同的起始年開始。"),
    ("sooner", "Earlier years", "較近的年份",
     "The same cut read at earlier reporting years.",
     "同樣的減抽，在較早的報告年份檢視。"),
    ("rank", "Rank the water uses", "比較各類用水",
     "Which cut moves the head most, and per unit of pumping energy given up.",
     "哪一類減抽最能提升水位，以及每單位放棄抽水能源的效果。"),
    ("sure", "How sure is this?", "可信度",
     "What the registered tests support, and the fully solved runs.",
     "登記檢驗支持哪些結果，以及完整求解的情境。"),
    ("internals", "Model internals", "模型內部數值",
     "Numbers whose size the tests do not support, shown under an amber label.",
     "檢驗不支持其大小的數值，以琥珀色標示。"),
)

REFUSALS_GROUP = ("refusals", "What it won't answer", "不回答的問題",
                  "Try one: these are refused by fixed rules, before any language model is called.",
                  "試試看：這些問題由固定規則拒答，不會呼叫語言模型。")

# Typed questions the hard guards refuse (changhua_lever, subsidence_size, beyond_horizon,
# township_head). Text only; the server routes them like any typed question.
TRY_REFUSALS = (
    ("How many centimetres of subsidence does Yunlin avoid if aquaculture pumping stops?",
     "停止養殖抽水，雲林可以少沉陷幾公分？"),
    ("What if Changhua cuts its irrigation pumping 30%?",
     "如果彰化減少農業灌溉抽水30%，會怎樣？"),
    ("What will the groundwater head be in 2040?",
     "2040年的地下水位會是多少？"),
    ("What is the groundwater level in Tuku township in 2030?",
     "土庫鎮2030年的地下水位是多少？"),
)

# The ledger's status codes in Traditional Chinese (the English comes from the ledger itself).
STATUS_ZH = {
    "S": "經登記檢驗支持（設計期間的檢驗另有標示）",
    "D": "由登記的前瞻檢驗決定（尚待結果）",
    "N": "依紀錄收窄、誠實的措辭",
    "L": "模型內部、描述性或介面數值，已標示",
    "W": "已自答案撤回（只在模型內部顯示，或不顯示）",
}
SHOWN_STATUSES = ("S", "D", "N", "L", "W")


def _cuts_text(c, cuts: list[dict], lang: str) -> str:
    names = {cl["key"]: cl[lang] for cl in c.classes}
    if len(cuts) == len(c.classes) and len({x["pct"] for x in cuts}) == 1:
        pct = cuts[0]["pct"]
        return f"Cut all pumping {pct}%" if lang == "en" else f"全部抽水減少{pct}%"
    if len(cuts) == 1 and cuts[0]["pct"] == 100:
        n = names[cuts[0]["class"]]
        return f"Stop {n.lower()} pumping" if lang == "en" else f"停止{n}抽水"
    if lang == "en":
        parts = [f"{names[x['class']].lower()} {x['pct']}%" for x in cuts]
        return ("Cut " + " and ".join(parts)) if len(parts) > 1 else \
            f"Cut {names[cuts[0]['class']].lower()} pumping {cuts[0]['pct']}%"
    parts = [f"{names[x['class']]}減少{x['pct']}%" for x in cuts]
    return "、".join(parts) if len(parts) > 1 else f"{names[cuts[0]['class']]}抽水減少{cuts[0]['pct']}%"


def label(b: PageBundle, preset: str, p: dict, lang: str) -> str:
    """A plain question for one preset example, in the page's own class and township names."""
    c = Q.ctx(b)
    y = p.get("year")
    if preset == "policy_head":
        w = _cuts_text(c, p["cuts"], lang)
        return (f"{w} from {p['start']}: head change by {y}?" if lang == "en"
                else f"{p['start']}年起{w}，到{y}年地下水位變化多少？")
    if preset == "start_compare":
        w, s0, s1 = _cuts_text(c, p["cuts"], lang), c.starts[0], c.starts[-1]
        return (f"{w} in {s0} or in {s1}: what difference does the start make by {y}?" if lang == "en"
                else f"{w}，{s0}年或{s1}年開始，到{y}年差多少？")
    if preset == "lever_rank":
        if p.get("by") == "gwh":
            return (f"Rank the water uses by head rise per unit of pumping energy given up, {y}" if lang == "en"
                    else f"依每單位放棄抽水能源的水位上升，比較各類用水（{y}年）")
        return (f"Rank the water uses by modelled head rise, {y}" if lang == "en"
                else f"依模擬的水位上升，比較各類用水（{y}年）")
    if preset == "township_internals":
        w, town = _cuts_text(c, p["cuts"], lang), c.towns[p["town"]][lang]
        return (f"{w} from {p['start']}: what does the model show for {town} by {y}?" if lang == "en"
                else f"{p['start']}年起{w}，模型對{town}到{y}年的結果？")
    if preset == "rail_internals":
        w = _cuts_text(c, p["cuts"], lang)
        return (f"{w} from {p['start']}: what does the model show along the high-speed rail by {y}?"
                if lang == "en" else f"{p['start']}年起{w}，模型對高鐵沿線到{y}年的結果？")
    if preset == "solved_compare":
        return (f"How do the fully solved scenarios compare by {y}?" if lang == "en"
                else f"完整求解的情境到{y}年如何比較？")
    if preset == "how_sure":
        return "How far are these results tested?" if lang == "en" else "這些結果經過多少檢驗？"
    raise ValueError(preset)


def _items(b: PageBundle) -> list[tuple[str, str, dict]]:
    c = Q.ctx(b)
    keys = [cl["key"] for cl in c.classes]
    Y, s0, s1 = c.years[-1], c.starts[0], c.starts[-1]

    def cut(key, pct):
        return [{"class": key, "pct": pct}] if key in keys else []

    def every(pct):
        return [{"class": k, "pct": pct} for k in keys]

    def ph(cuts, start=s0, year=Y):
        return {"cuts": cuts, "start": start, "year": year}

    towns = [i for i, t in enumerate(c.towns) if not t.get("pooled")]

    def town(name, fallback):
        return next((i for i in towns if c.towns[i]["en"] == name),
                    towns[fallback] if len(towns) > fallback else None)

    early = [y for y in (s0 + 1, s0 + 3) if y in c.years and y < Y]
    out = [
        ("one_use", "policy_head", ph(cut("irrigation", 30))),
        ("one_use", "policy_head", ph(cut("aquaculture", 100))),
        ("one_use", "policy_head", ph(cut("livestock", 50))),
        ("one_use", "policy_head", ph(cut("industry", 30))),
        ("one_use", "policy_head", ph(cut("domestic", 20))),
        ("size", "policy_head", ph(cut("irrigation", 10))),
        ("size", "policy_head", ph(cut("irrigation", 50))),
        ("size", "policy_head", ph(cut("irrigation", 100))),
        ("combine", "policy_head", ph(every(30))),
        ("combine", "policy_head", ph(every(20))),
        ("combine", "policy_head", ph(cut("irrigation", 20) + cut("aquaculture", 50))),
        ("timing", "start_compare", {"cuts": cut("aquaculture", 100), "year": Y}),
        ("timing", "start_compare", {"cuts": cut("irrigation", 30), "year": Y}),
        ("timing", "policy_head", ph(cut("aquaculture", 30), start=s1)),
        *[("sooner", "policy_head", ph(cut("irrigation", 30), year=y)) for y in early],
        ("rank", "lever_rank", {"year": Y, "by": "head"}),
        ("rank", "lever_rank", {"year": Y, "by": "gwh"}),
        *[("rank", "lever_rank", {"year": y, "by": "head"}) for y in early[-1:]],
        ("sure", "how_sure", {}),
        ("sure", "solved_compare", {"year": Y}),
        ("internals", "rail_internals", ph(cut("irrigation", 30))),
    ]
    for name, fb, cuts in (("Tuku", 0, cut("aquaculture", 100)), ("Mailiao", 1, cut("irrigation", 30))):
        i = town(name, fb)
        if i is not None:
            out.append(("internals", "township_internals", {**ph(cuts), "town": i}))
    seen, uniq = set(), []
    for g, preset, p in out:
        if preset != "how_sure" and not p.get("cuts", True):
            continue                                   # a class this page does not have
        k = (preset, repr(p))
        if k not in seen:
            seen.add(k)
            uniq.append((g, preset, p))
    return uniq


def examples(b: PageBundle) -> list[dict]:
    """Preset examples for ``/api/meta`` (same shape as :func:`presets.examples`, plus ``gid``)."""
    titles = {g[0]: {"en": g[1], "zh": g[2]} for g in GROUPS}
    return [{"preset": preset, "params": p, "gid": g, "group": titles[g],
             "label": {lg: label(b, preset, p, lg) for lg in ("en", "zh")}}
            for g, preset, p in _items(b)]


def library(b: PageBundle) -> dict:
    """Group order and descriptions, try-a-refusal questions and the status legend."""
    codes = b.ledger.get("status_codes", {})
    return {
        "groups": [{"id": g[0], "title": {"en": g[1], "zh": g[2]}, "desc": {"en": g[3], "zh": g[4]}}
                   for g in GROUPS + (REFUSALS_GROUP,)],
        "refusals": [{"gid": REFUSALS_GROUP[0], "question": {"en": en, "zh": zh}}
                     for en, zh in TRY_REFUSALS],
        "statuses": [{"code": k, "en": codes[k], "zh": STATUS_ZH.get(k, codes[k])}
                     for k in SHOWN_STATUSES if k in codes],
    }

