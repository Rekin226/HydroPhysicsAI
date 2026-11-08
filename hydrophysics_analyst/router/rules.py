"""Deterministic question parser (en/zh). Since ruling R9.2 (2026-11-07) its route stands only
when it refuses or clarifies; a preset it reads is a hint, and the LLM routes every preset
(``router.route``).

Dictionaries come from the bound page's payload (class names, township names, years, starts);
patterns cover percentages, "retire/停止", start and target years, per-GWh ranking, the rail,
start comparisons and follow-up patches ("same for 2026", 「改成2030年」).

``hard_refusal`` runs on the raw question before anything else and overrides any route, rules
or LLM (spec §8): any Changhua mention (except asking for the pooled township row by name), money
or volumes, recommendations, individual observations, years or leads after the horizon, area,
subsidence size, a single township's water level (ruling D3). The Changhua, horizon, subsidence
and township-head checks are the deterministic guards of ``router.guards``.
"""
from __future__ import annotations

import re

from .. import query as Q
from ..page import PageBundle
from . import guards as G
from .schemas import Route

CLASS_WORDS = {
    "irrigation": [r"irrigat\w*", r"agricultur\w*", r"farm\w*", r"crops?", "灌溉", "農業", "農田"],
    "aquaculture": [r"aquacultur\w*", r"fish\s*farm\w*", r"fish\w*", r"aqua\b", "養殖", "魚塭", "漁業"],
    "livestock": [r"livestock", r"animal\w*", r"pigs?", r"poultry", "畜牧", "養豬", "畜產"],
    "domestic": [r"domestic", r"household\w*", r"residential", r"drinking", "民生", "生活", "家用", "自來水"],
    "industry": [r"industr\w*", r"factor(?:y|ies)", "工業", "工廠"],
    "other": [r"other\s+(?:use|pumping|class)\w*", "其他"],
}
ALL_WORDS = [r"all\s+(?:pumping|classes|uses|water)", r"every\s+class", r"across\s+the\s+board",
             "全部類別", "所有類別", "全部用水", "所有用水", "各類別", "全部抽水"]   # 全面停抽 = stop one class completely
RETIRE = [r"retir\w*", r"\bstop\w*", r"\bban\w*", r"\bend\b", r"eliminat\w*", r"phase\s*out",
          "停止", "全面停", "禁止", "完全停"]
HALF = [r"\bhalv\w*", r"\bhalf\b", "減半", "一半"]
CHANGHUA = r"\bchanghua\b|彰化"
TOWN_INTENT = [r"\btownships?\b", r"\btowns?\b", r"what\s+happens\s+to", r"\bin\s+\w+\s+township",
               "鄉鎮", "會怎樣", "會如何", "這個鎮", "地區"]
SUBS = [r"subsid\w*", r"\bsink\w*", r"\bsank\b", r"\bcm\b", r"centimet\w*", "下陷", "沉陷", "公分", "地層"]
AREA = [r"\barea\b", r"km²", r"km2", r"square\s+k", "面積", "平方公里"]
MONEY = [r"hectares?", r"\bacres?\b", r"land\s+area", "公頃", "甲地", "土地面積", r"\bmoney\b", r"\bnt\$", r"\$", r"dollars?", r"\bbudget\b", r"\bprice\b", r"\bnt\s*d\b",
         r"cubic", r"m³", r"m3\b", r"\bvolume\w*", r"\btons?\b", "預算", "費用", "立方", "抽水量", "噸",
         # 元 only as money (next to a number or a money word): 元長鄉 is a Yunlin township
         r"[\d一二兩三四五六七八九十百千萬億]\s*(?:萬|億)?\s*元(?!長)", r"(?:多少|幾|幾百|幾千|幾萬)\s*(?:萬|億)?\s*元",
         "新台幣", "台幣", "錢", "成本", "經費", "花費", "補償", "補助", "價格", "金額"]
RECOMMEND = [r"recommend\w*", r"should\s+(?:we|i|the)", r"which\s+(?:policy|option)\s+(?:is\s+)?best",
             r"best\s+(?:policy|option|choice)", r"what\s+should", "建議", "應該", "最好的政策", "最佳方案",
             "該選",
             # "the best groundwater policy", 最佳的抽水方案 (not "best estimate", not a lever ranking)
             r"\bbest\s+(?:\S+\s+){0,3}?(?:policy|policies|option|options|choice|strategy|strategies|plan|"
             r"scenario|course\s+of\s+action)\b",
             r"最(?:佳|好)的?[^，,。？?；;]{0,8}(?:方案|政策|選擇|策略|做法)"]
OBS = [r"benchmark\w*", r"\bstations?\b", r"\bgnss\b", r"\bgps\b", r"\breadings?\b", r"which\s+well",
       r"specific\s+well", r"well\s+(?:no|#)", r"individual\s+wells?",
       "水準點", "單一觀測井", "哪一口井", "測站", "觀測站", "監測站", "讀數",
       # one named well's record (not the registered wells, which are the head answer's unit)
       r"\b(?:the|a|that|this|one)\s+(?:\w+\s+){0,2}monitoring\s+well\b(?!s)",
       r"(?:那個|這個|某個|那口|這口|某口|一口|那个|这个|某个|这口)[^，,。？?]{0,8}(?:監測井|觀測井|监测井|观测井)",
       "觀測值", "观测值", "實測值", "实测值",
       # a well's observed data (not the registered wells' mean, which is the head answer)
       r"(?:觀測井|監測井|水井)[^，,。？?；;]{0,8}(?:資料|數據|紀錄|記錄|觀測)",
       r"\b(?:observed|measured|recorded)\s+(?:\w+\s+)?(?:wells?|stations?|benchmarks?)\s+"
       r"(?:data|records?|readings?|series|values?)\b",
       r"\bwells?\s+(?:data|records?)\s+(?:from|at|for|of|in)\b"]
RAIL = [r"\bhsr\b", r"high[-\s]speed\s+rail\w*", r"\bthsr\b", r"\brail\w*", r"\btrain\w*", "高鐵", "鐵路"]
RANK = [r"\brank\w*", r"which\s+(?:class|lever|use|sector)", r"most\s+effective", r"biggest\s+(?:head\s+)?(?:rise|effect)",
        r"per\s+gwh", r"compare\s+(?:the\s+)?classes", "排序", "哪個類別", "哪一類", "效益最高", "最有效"]
PER_GWH = [r"per\s+gwh", r"每\s*gwh", r"per\s+(?:unit\s+of\s+)?energy", r"efficien\w*", r"\bcheap\w*\s+class",
           "每度電", "每單位", "用電效益", "最省電的類別", "效率"]
STARTCMP = [r"\bwait\w*", r"\bdelay\w*", r"\blater\b", r"\bor\s+(?:in\s+)?20\d\d", r"(?:20\d\d)\s+(?:vs\.?|versus)\s+(?:20\d\d)",
            r"start\s+(?:now|early|late)", "延後", "晚一點", "等到", "還是", "比較早", "提早"]
SOLVED = [r"full[-\s]model", "完整模型", r"two\s+(?:full\s+)?runs", r"compare\s+(?:the\s+)?(?:two|both)\s+(?:policies|scenarios|runs)",
          r"solved", "完整模擬", "兩個方案", "兩個情境", "比較兩"]
CHEAP = [r"cheapest", r"least\s+energy", r"lowest\s+energy", r"minimum\s+(?:cut|energy)", "最省電", "最少用電", "最低用電"]
# "as of <year>" is a reporting date, not a start (round 10, A5: it read "As of 2028" as a start
# off the page and clarified out_of_grid)
FROM = (r"(?:from|starting(?:\s+in)?|start(?:s|ed|ing)?\s+(?:in\s+)?|beg(?:an|un|inning|ins?)\s+(?:in\s+)?|"
        r"launched\s+in|implemented\s+in)\s*")
# "from 2023 to 2028", 從2023年到2028年: a span of reporting years, its first year is no start
YEAR_SPAN = r"\s*(?:年)?\s*(?:to|until|till|through|thru|and|-|–|—|~|～|到|至)\s*(?:20\d\d)"
HOWSURE = [r"how\s+(?:sure|confident|reliable|trustworthy|certain)", r"can\s+(?:we|i)\s+trust", r"\bvalidat\w*",
           r"how\s+(?:well\s+)?tested", r"when\s+(?:is|will)\s+(?:it|this)\s+(?:be\s+)?(?:tested|scored)", r"\buncertain\w*",
           "可信", "可靠", "多準", "信心", "驗證", "檢驗", "不確定"]
VAGUE_TIME = [r"a\s+few\s+years", r"some\s+years", r"in\s+the\s+future", r"eventually", r"down\s+the\s+road",
              "幾年後", "未來幾年", "以後", "將來"]
ADD = [r"\balso\b", r"\bplus\b", r"\bon\s+top\b", r"\badd(?:ing)?\b", r"as\s+well", r"\btoo\b",
       "再加上", "加上", "另外", "同時", "也"]
DELAY_TO = r"(?:延後|等|delay\w*|wait\w*|postpon\w*|push\w*\s+back)[^，,。.;；]{0,6}?(?:到|until|to|till)?\s*"
FOLLOW = [r"^\s*(?:and\s+|what\s+about\s+|how\s+about\s+)?(?:the\s+)?same\b", r"\binstead\b",
          r"^\s*(?:and|what about|how about)\b", "改成", "換成", "那", "同樣", "如果改", "呢"]


def _has(pats, s: str) -> bool:
    return any(re.search(p, s, re.I) for p in pats)


def _norm(s: str) -> str:
    """NFKC and simplified -> traditional (router.zh): every pattern and page name is traditional."""
    return G.norm(s)


def detect_lang(q: str) -> str:
    return "zh" if re.search(r"[一-鿿]", q) else "en"


def _pooled_names(b: PageBundle) -> list[str]:
    return [n for t in Q.ctx(b).towns if t.get("pooled") for n in (t["en"], t["zh"])]


def _yunlin_town(b: PageBundle, s: str) -> bool:
    """A page township other than the pooled Changhua row is named."""
    c = Q.ctx(b)
    t = _town_of(b, s)
    hits = [t] if isinstance(t, int) else (t or [])
    return any(not c.towns[i].get("pooled") for i in hits)


def hard_refusal(q: str, b: PageBundle) -> str | None:
    s = _norm(q)
    via = G.changhua_ref(s)
    ch = bool(via - {"fan"}) or any(n and _norm(n) in s for n in _pooled_names(b))
    # the only allowed Changhua question (guards.pooled_row_ok): the pooled township row's
    # labelled spillover, asked as that row by the name Changhua, under an explicitly Yunlin
    # policy, with no head word (C13a) and no lever that is Changhua's; a Changhua township or a
    # geographic reference is always refused
    pooled_ok = ch and G.pooled_row_ok(s)
    # round 10 (R10.2a): a Yunlin question that names Changhua (or its geography) only to leave
    # it out (A1), and one how-sure question about Changhua's direction test (A3), pass on. The
    # pooled row stays answerable only as a row (round 9's pooled_row_ok; ledger C02: Changhua
    # cells are labelled "spillover of a Yunlin cut ... not a Changhua lever")
    if ch and not (pooled_ok or G.yunlin_only_ok(s) or G.changhua_how_sure_ok(s, b)):
        return "changhua_lever"
    if _has(MONEY, s):
        return "volumes_money"
    if _has(RECOMMEND, s):
        return "recommendation"
    if _has(OBS, s):
        return "observations"
    if G.beyond_horizon(s, b):
        return "beyond_horizon"
    if _has(AREA, s) and _has(SUBS + [r"sink", r"rate", "速率"], s):
        return "area"
    # a named Yunlin township (its internals row, C11) and the pooled row are exempt, unless an
    # area-scope word (fan, county-wide, whole, total, 全縣, 整個, 沖積扇 ...) asks the size wider
    # round 10 (A4): one township referred to but not named, under a policy, is clarified by the
    # rules (ambiguous_town), not refused; an area-scope word still overrides it
    if G.subsidence_size(s) and (G.area_scope(s) or not (_yunlin_town(b, s) or pooled_ok
                                                          or G.unnamed_township(s, b))):
        return "subsidence_size"
    if "fan" in via and G.has_lever(s):                # a lever over the whole fan includes Changhua
        return "changhua_lever"
    if G.township_head(s, b):                         # ruling D3, whatever else is asked
        return "township_head"
    return None


def _town_of(b: PageBundle, s: str):
    c = Q.ctx(b)
    hits = []
    for i, t in enumerate(c.towns):
        names = [t["en"], _norm(t["zh"])] if t.get("pooled") else G.town_names(t)
        for n in names:
            if n and len(n) >= 2 and re.search(re.escape(n), s, re.I):
                hits.append(i)
                break
    if not hits and re.search(CHANGHUA, s, re.I):
        hits = [i for i, t in enumerate(c.towns) if t.get("pooled")]
    return hits[0] if len(hits) == 1 else (hits if hits else None)


def _class_spans(s: str) -> list[tuple[int, int]]:
    out = []
    for k, (_, pats) in enumerate(CLASS_WORDS.items()):
        for p in pats:
            for m in re.finditer(p, s, re.I):
                out.append((m.start(), k))
    return sorted(out)


def _pcts(s: str) -> list[tuple[int, int]]:
    out = []
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|％|percent|per\s*cent|個百分點)", s, re.I):
        out.append((m.start(), round(float(m.group(1)))))
    for m in re.finditer(r"([一二三四五六七八九十]|\d)\s*成", s):
        zh = "一二三四五六七八九十"
        v = m.group(1)
        n = zh.index(v) + 1 if v in zh else int(v)
        out.append((m.start(), n * 10))
    for p in HALF:
        for m in re.finditer(p, s, re.I):
            out.append((m.start(), 50))
    for p in RETIRE:
        for m in re.finditer(p, s, re.I):
            out.append((m.start(), 100))
    return sorted(out)


def parse_cuts(b: PageBundle, s: str) -> list[dict] | None:
    c = Q.ctx(b)
    keys = [cl["key"] for cl in c.classes]
    classes = _class_spans(s)
    pcts = _pcts(s)
    if _has(ALL_WORDS, s):
        v = pcts[0][1] if pcts else None
        return [{"class": k, "pct": v} for k in keys] if v is not None else [{"class": k, "pct": None} for k in keys]
    if not classes:
        return None
    seen: dict[int, int | None] = {}
    for pos, k in classes:
        if k in seen:
            continue
        # nearest percentage, preferring one after the class word, then before
        after = [p for p in pcts if p[0] >= pos]
        before = [p for p in pcts if p[0] < pos]
        cand = (after[0] if after and (not before or after[0][0] - pos <= pos - before[-1][0] + 6)
                else before[-1] if before else None)
        seen[k] = cand[1] if cand else None
        if cand:
            pcts = [p for p in pcts if p is not cand] if len(classes) > 1 else pcts
    return [{"class": keys[k], "pct": v} for k, v in seen.items()]


def parse_years(b: PageBundle, s: str) -> tuple[int | None, int | None]:
    """(start, year) from the question; ``None`` when not given."""
    c = Q.ctx(b)
    start = year = None
    for m in re.finditer(r"(?<!\d)(20\d\d)(?!\d)", s):
        y = int(m.group(1))
        pre = s[max(0, m.start() - 22):m.start()].lower()
        post = s[m.end():m.end() + 6]
        clause = re.split(r"[。；;？?！!，,:：]", s[:m.start()])[-1]
        # a span ("from 2023 to 2028") is a start only when a lever acts on it in the same clause
        # ("cut irrigation 30% from 2028 to 2032"; review finding 7)
        span = (re.match(YEAR_SPAN, s[m.end():], re.I) and re.search(r"(?:from|between|從|自)\s*$", pre, re.I)
                and not G.has_lever(clause))
        # "as of <year>" is a reporting year, except a year before the first page start with a
        # lever in the question ("As of 2025, Yunlin cuts irrigation 30%": a start off the page)
        as_of = re.search(r"(?:as\s+of|截至|截止)\s*$", pre, re.I)
        if as_of and y < c.starts[0] and G.has_lever(s):
            start = start or -1
        elif not span and (re.search(FROM + r"$", pre, re.I)
                           or re.match(r"\s*(?:年)?\s*(?:開始|起(?!來)|實施|啟動|上路)", post)
                           or re.search(r"(?:自|從)\s*$", pre)):
            start = y if y in c.starts else (start or -1)     # -1: a start not on the page
        else:
            year = y
    return start, year


def _target_m(s: str) -> float | None:
    m = re.search(r"\+?\s*(\d+(?:\.\d+)?)\s*(?:m\b|meters?|metres?|公尺|米)", s, re.I)
    return float(m.group(1)) if m else None


class RulesRouter:
    def __init__(self, b: PageBundle):
        self.b = b

    def _both_starts(self, s: str) -> bool:
        """Both page starts named, or one start against 'now' (立即 / 現在 / immediately)."""
        c = Q.ctx(self.b)
        named = {int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", s)} & set(c.starts)
        now = _has([r"\bnow\b", r"immediately", r"right\s+away", r"\btoday\b", "現在", "立即", "馬上", "早做"], s)
        cmp_ = _has([r"\bvs\.?\b", r"versus", r"compar\w*", r"\bor\b", r"\band\b", "比較", "跟", "和", "還是", "差"], s)
        return cmp_ and (len(named) >= 2 or (len(named) == 1 and now))

    def route(self, question: str, prev: dict | None = None) -> Route:
        b = self.b
        c = Q.ctx(b)
        s = _norm(question)
        lang = detect_lang(s)
        ref = hard_refusal(s, b)
        if ref:
            return Route(preset="refuse", params={"reason": ref}, refusal=ref, lang=lang, confidence=1.0)
        cuts = parse_cuts(b, s)
        start, year = parse_years(b, s)
        if start == -1:                       # an explicit start year the page does not have
            return Route(lang=lang, confidence=1.0, clarify={"reason": "out_of_grid", "candidates": c.starts})
        town = _town_of(b, s)
        # round 10 (A4): one township, not named, asked for its subsidence under a policy
        unnamed = town is None and G.unnamed_township(s, b) and G.subsidence_words(s)
        follow = prev is not None and (_has(FOLLOW, s) or (len(s) < 30 and not _has(RAIL + RANK + SOLVED, s)))
        # ruling D3 (a single township's water level) is refused by hard_refusal's guard above
        # intent
        if _has(HOWSURE, s) and not cuts and not _has(RAIL + RANK, s):
            return Route(preset="how_sure", params={}, lang=lang, confidence=1.0)
        if _has(RAIL, s):
            preset = "rail_internals"
        elif _has(CHEAP, s) and _target_m(s) is not None:
            preset = "cheapest_for_head"
        elif _has(RANK, s) and not cuts:
            preset = "lever_rank"
        elif _has(SOLVED, s) or (_has([r"compar\w*", "比較"], s)
                                  and any(x["name"].lower() in s.lower() for x in c.solved)):
            preset = "solved_compare"
        elif ((_has(STARTCMP, s) or self._both_starts(s)) and (cuts or follow) and not town):
            preset = "start_compare"
        elif town is not None or _has(TOWN_INTENT, s) or unnamed:
            preset = "township_internals"
        elif follow and cuts and prev.get("preset") in ("start_compare", "township_internals", "rail_internals"):
            preset = prev["preset"]           # "and also aquaculture 10%?" keeps the previous question type
        elif cuts:
            preset = "policy_head"
        elif follow:
            preset = prev.get("preset")
        else:
            return Route(lang=lang, confidence=0.0, clarify={"reason": "unknown", "candidates": []})
        params: dict = {}
        inherit: list[str] = []
        prevp = (prev or {}).get("params", {}) if follow else {}
        # follow-up: a bare start-year value patches the start
        if follow and start is None and year is not None and year in c.starts and "start" in prevp \
                and not re.search(r"(?:by|in|until|到|至)\s*" + str(year), s, re.I):
            start, year = year, None
        if preset in ("policy_head", "start_compare", "township_internals", "rail_internals"):
            if cuts and follow and _has(ADD, s) and prevp.get("cuts"):
                merged = {x["class"]: x["pct"] for x in prevp["cuts"]} | {x["class"]: x["pct"] for x in cuts}
                params["cuts"] = [{"class": k, "pct": v} for k, v in merged.items()]
            elif cuts:
                params["cuts"] = cuts
            elif "cuts" in prevp:
                params["cuts"], _ = prevp["cuts"], inherit.append("cuts")
        if preset in ("policy_head", "township_internals", "rail_internals", "cheapest_for_head"):
            if start and start > 0:
                params["start"] = start
            elif "start" in prevp:
                params["start"], _ = prevp["start"], inherit.append("start")
            else:
                params["start"] = c.starts[0]
        if preset == "start_compare" and year in c.starts and (
                re.search(DELAY_TO + str(year), s, re.I) or self._both_starts(s)
                or not re.search(r"(?:by|in\s+dec\w*|到|至)\s*" + str(year), s, re.I)):
            year = None                       # "waiting until 2030": a start, not a reporting year
        if preset == "start_compare" and year is None and self._both_starts(s):
            ys = [int(y) for y in re.findall(r"(?<!\d)(20\d\d)(?!\d)", s) if int(y) not in c.starts]
            year = ys[-1] if ys else None
        if preset != "refuse":
            if year:
                params["year"] = year
            elif "year" in prevp:
                params["year"], _ = prevp["year"], inherit.append("year")
            else:
                params["year"] = c.years[-1]
        if preset == "township_internals":
            if isinstance(town, int):
                params["town"] = town
            elif "town" in prevp:
                params["town"], _ = prevp["town"], inherit.append("town")
            else:
                # A4: the unnamed-township subsidence question is a confident clarification
                return Route(preset=preset, params=params, lang=lang, confidence=1.0 if unnamed else 0.6,
                             clarify={"reason": "ambiguous_town",
                                      "candidates": [c.towns[i]["en"] for i in (town or range(len(c.towns)))]})
        if preset == "lever_rank":
            params["by"] = "gwh" if _has(PER_GWH, s) else ("gwh" if prevp.get("by") == "gwh" and follow else "head")
        if preset == "cheapest_for_head":
            params["target_m"] = _target_m(s)
        conf = 1.0
        if preset in ("policy_head", "start_compare", "township_internals", "rail_internals") and (
                not params.get("cuts") or any(x["pct"] is None for x in params["cuts"])):
            conf = 0.4
        if start == -1 or (_has(VAGUE_TIME, s) and not year):
            conf = min(conf, 0.5)
        # a follow-up from which nothing new could be read ("...at the later date instead") is
        # not a confident patch: hand it to the LLM rather than repeat the previous answer
        if follow and not (cuts or start or year or isinstance(town, int) or _has(PER_GWH, s)):
            conf = min(conf, 0.5)
        return Route(preset=preset, params=params, inherit=inherit, lang=lang, confidence=conf)
