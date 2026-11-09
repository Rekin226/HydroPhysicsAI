"""Round 11 (ruling R11.1): a tighten-only post-route step, run after the hard refusals, the rules
and the LLM have routed a question (``router.route``).

It may do exactly two things, and nothing else:

(a) turn a **preset** answer into a clarification: a preset that reports at a year (every routed
    preset but ``how_sure``), for a question that leads forward by an indefinite number of years
    ("a few years from now", 過幾年之後, 未來幾年) about the reporting year (not about when the
    policy starts, not past or duration wording) and gives no reporting year in any form, is
    clarified ``missing_year``. When unsure, it does not fire. OPS.md: the page's default year (2032) applies "when the question gives
    none"; a question asking about "a few years from now" asks about a year it does not give, so
    the default would answer a different question. The LLM's own prompt says the same ("only
    when the question gives no year at all"). This is the indefinite twin of the guards' leads:
    ``guards.beyond_horizon`` reads "in N years" / N年後 as start + N, and here N is not given.
(b) change one clarification reason into another: the catch-all ``unknown`` becomes
    ``out_of_grid`` when the question names a start (with a start verb) or a cut size (that
    belongs to a cut) off the page grid (OPS.md:
    "out_of_grid (a size, start or year off the page grid)").

A refusal (``refusal`` set or preset ``refuse``) is returned as it is, so the step can never turn a
refusal into anything else; a preset route is changed only when it would be answered as that
preset (not refused or clarified at run time, not replaced by the validator); a clarification
stays a clarification; no route ever becomes a preset.
Both changes return a bare clarification (no preset, no parameters), like
``router.bare_clarification``. Proof by construction in :func:`tighten`, by property tests in
tests/test_analyst_tighten_r11.py. Developed after held-out v5 was spent (v5_034, v5_061), using
the spent v3/v4/v5 and the dev set; unscored.
"""
from __future__ import annotations

import re

from .. import presets as PR
from .. import query as Q
from ..page import PageBundle
from . import guards as G
from .rules import parse_years
from .schemas import Route

# the reasons presets.clarification shows as themselves; any other is shown as "unknown"
CLARIFY_SHOWN = ("ambiguous_town", "missing_year", "missing_cut", "out_of_grid", "no_llm")
# the routed presets that report at a year (schemas.PRESET_PARAMS); how_sure takes none
YEAR_PRESETS = ("policy_head", "start_compare", "township_internals", "rail_internals",
                "lever_rank", "solved_compare", "cheapest_for_head")

# (a) a FORWARD lead by an indefinite number of years ("a couple of" is two years to the guards,
# so it is not here). Past and duration wording is not a lead: "for / over / during / since /
# after many years", "in some years" (= in certain years), 過去幾年, 這幾年間, 幾年內 (review F1).
_EN_V = r"(?:a\s+few|few|several|some|a\s+number\s+of|many)"
VAGUE_LEAD_EN = [
    rf"\b{_EN_V}\s+(?:more\s+)?years?\s+(?:from\s+now|later|hence|on|ahead|out|"
    r"down\s+the\s+(?:road|line)|after\s+(?:that|this)|"
    r"after\s+the\s+(?:cut|policy|change|reduction))\b",
    r"\bin\s+(?:a\s+few|several)\s+(?:more\s+)?years?(?!\s+of\b)\b",
    r"\bin\s+the\s+years\s+(?:ahead|to\s+come)\b",
]
_ZH_V = r"(?<![十百這近前頭])(?:好幾|幾|數|若干)\s*個?"
VAGUE_LEAD_ZH = [
    rf"{_ZH_V}\s*(?:多)?\s*年\s*(?:後|之後|以後|過後)",
    rf"(?:再過|(?<!再)過(?!去)|未來|往後)\s*的?\s*{_ZH_V}\s*年(?!\s*(?:間|內|期間|來))",
]
# the vague phrase is about WHEN THE POLICY STARTS, not the reporting year: a start verb in the
# same clause ("start a few years later", 過幾年再做, 晚幾年開始) ...
START_WORDS = [r"\bstart\w*", r"\bbegin\w*", r"\bbegan\b", r"\bbegun\b", r"\bimplement\w*",
               r"\bintroduc\w*", r"\blaunch\w*", r"\bdelay\w*", r"\bwait\w*", r"\bpostpon\w*",
               r"\bdo(?:ing)?\s+it\b", r"\bact(?:ing)?\b", r"\bplan\w*", r"\bwe\s+(?:will|'ll)\b",
               r"\bcut\w*\s+(?:\w+\s+){0,3}?(?:a\s+few|several|some|many)\s+years\s+later",
               "實施", "開始", "啟動", "上路", "推動", "執行", "再做", "才做", "再減", "才減", "晚",
               "延"]
# ... or the question compares acting now with acting later
COMPARE_NOW_LATER = [r"\bnow\b[^.?!;]{0,60}\bor\b", r"\bor\b[^.?!;]{0,40}\blater\b", r"\bvs\.?\b",
                     r"\bversus\b", r"\bcompar\w*", r"\bbetter\b", "現在.{0,12}(?:跟|和|還是|或|與)",
                     "差多少", "比較", "還是"]
# "how many years until it recovers": asks for a time, gives none (要幾年後…才)
ASKS_HOW_LONG = [r"要\s*(?:好幾|幾|數)\s*年", r"(?:幾|數)\s*年\s*(?:後|之後)?[^，,。？?；;]{0,12}才",
                 r"\bhow\s+(?:many|long)\b"]
CLAUSE = r"[。；;？?！!，,:：.]"

# (iii) a definite year in other forms: 二〇三〇, a ROC year without 民國 (一二一年), "in 32",
# "year 32", "3 yrs", next year / 明年 / 後年, the end of the decade / run / page, the final year.
# When in doubt the question counts as giving a year, so (a) does not fire.
DEFINITE_YEAR = [
    r"[二兩]\s*[〇零]\s*[〇零一二三四五六七八九]\s*[〇零一二三四五六七八九]",
    r"\b(?:in|by|at|for|until|till|to)\s+(?:the\s+)?(?:year\s+)?(?:20)?[2-4]\d\b(?!\s*(?:%|percent|m\b|cm|years?|yrs?))",
    r"\byear\s+[2-4]\d\b",
    r"\b(?:\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:-\s*)?(?:yrs?|years?)\b",
    r"\bnext\s+year\b", r"\bthis\s+year\b", r"\blast\s+year\b", r"\byear\s+after\s+next\b",
    r"\bend\s+of\s+(?:the\s+)?(?:decade|run|page|projection|horizon|period)\b",
    r"\b(?:final|last)\s+year\b", r"\bpage'?s?\s+(?:final|last|end)\b", r"\bthe\s+page\s+ends\b",
    "明年", "後年", "今年", "大後年", "最後一年", "期末",
]


def _has(pats, s: str) -> bool:
    return any(re.search(p, s, re.I) for p in pats)


def _clause_at(s: str, i: int) -> str:
    a = max((m.end() for m in re.finditer(CLAUSE, s[:i])), default=0)
    m = re.search(CLAUSE, s[i:])
    return s[a:i + m.start()] if m else s[a:]


def vague_lead(s: str) -> bool:
    """The question asks about a reporting time an indefinite number of years ahead ("a few
    years from now", 過幾年之後): a forward lead whose clause is not about when the policy starts,
    in a question that does not compare now with later or ask how long something takes."""
    s = G.norm(s)
    if _has(COMPARE_NOW_LATER + ASKS_HOW_LONG, s):
        return False
    for p in VAGUE_LEAD_EN + VAGUE_LEAD_ZH:
        for m in re.finditer(p, s, re.I):
            if not _has(START_WORDS, _clause_at(s, m.start())):
                return True
    return False


def _roc_zh(s: str) -> bool:
    """A ROC year in Chinese numerals without 民國 (一二一年, 一百二十一年)."""
    for m in re.finditer(r"([〇零一二三四五六七八九十百]{3,6})\s*年", s):
        n = G.zh_number(m.group(1))
        if n is not None and 100 <= n <= 199:
            return True
    return False


def gives_year(s: str, b: PageBundle) -> bool:
    """The question names a reporting year: a calendar, spelled, two-digit or ROC year, a
    definite lead ("in 3 years", 三年後) the guards read as one, or one of ``DEFINITE_YEAR``."""
    s = G.norm(s)
    if parse_years(b, s)[1] is not None or G.spelled_years(s):
        return True
    if re.search(r"(?:^|[\s(（])['’‘](\d{2})(?![\d%])", s):
        return True
    if re.search(r"(?:民國|ROC\s*)\s*(\d{2,3}|[〇零一二三四五六七八九十百]+)\s*年?|(?<![\d.])(1[1-9]\d)\s*年",
                 s, re.I) or _roc_zh(s):
        return True
    if _has(DEFINITE_YEAR, s):
        return True
    for pat, _ in G.EN_LEAD:
        if any(G._en_number(m.group(1)) is not None for m in re.finditer(pat, s, re.I)):
            return True
    for pat, _ in G.ZH_LEAD:
        if any(G.zh_number(m.group(1)) is not None for m in re.finditer(pat, s)):
            return True
    return False


def missing_year(s: str, b: PageBundle) -> bool:
    """(a): an indefinite forward lead and no reporting year."""
    return vague_lead(s) and not gives_year(s, b)


# (b) a start the rules stage does not read as one, with a start verb: 拖到2029年才實施,
# 2029年才開始, "not until 2029", "delay it until 2027". "wait until 2032" / 等到2032年 / 晚到2031年
# with an outcome after it is a reporting year, not a start (review F2).
_START_ZH = r"(?:開始|實施|啟動|上路|執行|推動|施行|生效|做|減|砍)"
LATE_START = [
    rf"(?:拖|延|延後|延遲|推遲|晚)\s*(?:到|至)\s*(20\d\d)\s*年?\s*(?:才|再)?\s*{_START_ZH}",
    rf"(?<!\d)(20\d\d)\s*年?\s*(?:才|再)\s*{_START_ZH}",
    r"\bnot\s+(?:until|till)\s+(20\d\d)\b",
    r"\b(?:delay\w*|postpon\w*|push\w*\s+back|hold\w*\s+off)\b[^.;?!]{0,30}?\b(?:until|to|till)\s+"
    r"(20\d\d)\b",
    r"\bwait\w*\s+until\s+(20\d\d)\s+(?:to|before)\s+(?:start|begin|cut|implement|act)\w*",
    r"\b(?:takes?|taking|took)\s+effect\s+(?:only\s+)?(?:in|from)\s+(20\d\d)\b",
]
# a cut size: the number must belong to a cut ("cut by 12%", "a 22% domestic cut", 減18%),
# not to a share or a change ("Irrigation is 68% of pumping", 佔43%, "head rose 12%")
CUT_BEFORE = [r"\bcut\w*", r"\breduc\w*", r"\blower\w*", r"\bdecreas\w*", r"\bcurb\w*", r"\btrim\w*",
              r"\bfall\w*", r"\bfell\b", r"\bdrop\w*", r"\bdeclin\w*", r"\bshrink\w*", r"\bretir\w*",
              r"\bmake\s+it\b", r"\bchange\s+it\s+to\b", r"\bset\s+it\s+to\b",
              "減", "降", "砍", "削", "少抽"]
CUT_AFTER = [r"^\s*(?:\w+\s+){0,3}?(?:cut|reduction|decrease)s?\b", r"^\s*instead\b",
             r"^\s*(?:的)?\s*(?:減|削減|減少|減抽)"]
_EN_DEN = {"half": 2, "third": 3, "quarter": 4, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
           "eighth": 8, "ninth": 9, "tenth": 10}
_EN_NUMR = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9}
FRACTIONS = [  # (pattern, numerator group, denominator group, kind)
    (r"\b(a|one|two|three|four|five|six|seven|eight|nine)[\s\-]+(half|third|quarter|fourth|fifth|"
     r"sixth|seventh|eighth|ninth|tenth)s?\b", "en"),
    (r"(?<![\d/])(\d)\s*/\s*(\d{1,2})(?![\d/])", "dig"),
    (r"([一二兩三四五六七八九十]+)\s*分\s*之\s*([一二兩三四五六七八九十]+)", "zh"),
]
ZH_PCT = r"百分之\s*([〇零一二兩三四五六七八九十百]+)"


# "12% of the farms", "a third of the farms": a share of something, not the cut size
SHARE_OF = (r"^\s*of\b(?!\s+(?:the\s+|its\s+|their\s+)?(?:\w+\s+)?(?:pumping|use|water|"
            r"withdrawals?|abstraction)\b)")
# a fraction is a cut size only right after the cut ("cut by a third", 減三分之一, "make it a third")
FRACTION_CUT = (r"(?:\bby|\bcut\w*|\breduc\w*|\blower\w*|\bmake\s+it|\bchange\s+it\s+to|"
                r"\bset\s+it\s+to|減少?|降低?|砍|削減?)\s*(?:about|roughly|around|約|大約)?\s*$")


def _cut_number(s: str, a: int, z: int) -> bool:
    """The number at s[a:z] is a cut size: a cut word just before it in its clause, or a cut
    noun just after it, and not a share ("68% of pumping" is a share; "12% of the pumping" is
    a cut)."""
    pre = re.split(CLAUSE, s[:a])[-1][-30:]
    post = re.split(CLAUSE, s[z:])[0][:25]
    if re.search(SHARE_OF, post, re.I):
        return False
    return _has(CUT_BEFORE, pre) or _has(CUT_AFTER, post)


def _sizes(s: str) -> list[float]:
    """Cut sizes in percent read from the question (only numbers that belong to a cut)."""
    out = []
    for m in re.finditer(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|％|percent|per\s*cent|個百分點)", s, re.I):
        if _cut_number(s, m.start(), m.end()):
            out.append(float(m.group(1)))
    for m in re.finditer(ZH_PCT, s):
        n = G.zh_number(m.group(1))
        if n is not None and _cut_number(s, m.start(), m.end()):
            out.append(float(n))
    for m in re.finditer(r"([一二三四五六七八九十]|\d)\s*成", s):
        v = m.group(1)
        n = "一二三四五六七八九十".index(v) + 1 if not v.isdigit() else int(v)
        if _cut_number(s, m.start(), m.end()):
            out.append(10.0 * n)
    for pat, kind in FRACTIONS:
        for m in re.finditer(pat, s, re.I):
            if kind == "en":
                num, den = _EN_NUMR.get(m.group(1).lower()), _EN_DEN.get(m.group(2).lower())
            elif kind == "dig":
                num, den = int(m.group(1)), int(m.group(2))
            else:                                   # 三分之一: denominator first
                den, num = G.zh_number(m.group(1)), G.zh_number(m.group(2))
            if not num or not den or num > den:
                continue
            frac = 100.0 * num / den
            pre = re.split(CLAUSE, s[:m.start()])[-1]
            if re.search(r"剩(?:下)?\s*(?:原本|原來)?\s*的?\s*$", pre):   # keep a third = cut two thirds
                out.append(100.0 - frac)
            elif re.search(FRACTION_CUT, pre, re.I) and not re.search(
                    SHARE_OF, re.split(CLAUSE, s[m.end():])[0], re.I):
                out.append(frac)
    return out


def off_grid_start(s: str, b: PageBundle) -> bool:
    s = G.norm(s)
    c = Q.ctx(b)
    if parse_years(b, s)[0] == -1:
        return True
    return any(int(m.group(1)) not in c.starts for p in LATE_START for m in re.finditer(p, s, re.I))


def off_grid_size(s: str) -> bool:
    """A cut size off the five-percent grid (``presets.PCT``), from a percentage, 百分之N, N成
    or a fraction that belongs to a cut (六分之三 is 50 %, on the grid)."""
    s = G.norm(s)
    return any(round(v, 6) not in PR.PCT for v in _sizes(s))


def off_grid(s: str, b: PageBundle) -> bool:
    """(b): a start or a cut size off the page grid. A year after the page is refused upstream
    (``beyond_horizon``); a year before it is history, not a grid miss, and is not read here."""
    return off_grid_start(s, b) or off_grid_size(s)


def is_refusal(r: Route) -> bool:
    return r.refusal is not None or r.preset == "refuse"


def outcome(r: Route) -> str:
    """What ``ask.answer_route`` makes of a route, line for line: ``refusal``, ``clarify``,
    ``preset`` or ``error``. A preset (not ``clarify``) with a falsy ``clarify`` runs the preset
    (``refuse`` is a refusal); anything else is a clarification with ``(r.clarify or
    unknown)["reason"]`` (a clarify without a reason raises: ``error``), and
    ``presets.clarification`` answers a ``beyond_horizon`` clarification as a refusal."""
    if r.preset and r.preset != "clarify" and not r.clarify:
        return "refusal" if r.preset == "refuse" else "preset"
    cl = r.clarify or {"reason": "unknown", "candidates": []}
    if "reason" not in cl:
        return "error"
    return "refusal" if cl["reason"] == "beyond_horizon" else "clarify"


def shown_reason(r: Route) -> str:
    """The reason a clarification route is answered with (``presets.clarification``)."""
    reason = (r.clarify or {"reason": "unknown"}).get("reason")
    return reason if reason in CLARIFY_SHOWN else "unknown"


def answered_as_preset(b: PageBundle, r: Route) -> bool:
    """The preset route is answered as that preset: ``presets.run`` neither refuses (no policy,
    no run, disabled) nor clarifies (bad parameters), and the validator would not replace the
    answer by a refusal (``ask.ask``). Only then may (a) turn it into a clarification, so a
    preset route that would be refused stays refused."""
    from ..validate import validate
    try:
        ans = PR.run(b, r.preset, r.params, r.lang or "en")
        return ans.refusal is None and ans.clarify is None and not validate(ans, b)
    except Exception:  # noqa: BLE001  an answer that fails is left to the caller, unchanged
        return False


def _clarify(r: Route, reason: str, candidates=()) -> Route:
    return Route(lang=r.lang, confidence=r.confidence, source=r.source,
                 clarify={"reason": reason, "candidates": list(candidates)})


def decide(b: PageBundle, question: str, kind: str, preset: str | None,
           reason: str | None) -> str | None:
    """The step's decision from the answer's kind (``preset``, ``clarify``, ``refusal``), the
    preset and the clarification reason shown: ``"missing_year"`` (a), ``"out_of_grid"`` (b) or
    ``None`` (no change). A recorded eval row's ``got`` is the answer's kind, so this replays
    the step on recorded rows; :func:`tighten` also checks a preset route is answered as one."""
    if kind == "preset" and preset in YEAR_PRESETS and missing_year(question, b):
        return "missing_year"
    if kind == "clarify" and reason == "unknown" and off_grid(question, b):
        return "out_of_grid"
    return None


def tighten(b: PageBundle, question: str, prev: dict | None, r: Route) -> tuple[Route, str | None]:
    """(route, change): the route after the step and what changed (``None`` when nothing).
    ``prev`` is accepted for the router's signature; a follow-up that leads by "a few years"
    asks about a year it does not give, whatever year the previous turn had."""
    del prev
    if is_refusal(r):                      # a refusal is final: never anything else
        return r, None
    kind = outcome(r)
    if kind in ("refusal", "error"):       # beyond_horizon as a clarification; a malformed route
        return r, None
    reason = shown_reason(r) if kind == "clarify" else None
    d = decide(b, question, kind, r.preset, reason)
    if d == "missing_year" and not answered_as_preset(b, r):
        return r, None
    if d == "missing_year":                # (a) preset -> clarification
        return _clarify(r, "missing_year"), f"{r.preset}->clarify:missing_year"
    if d == "out_of_grid":                 # (b) clarify unknown -> clarify out_of_grid
        cand = Q.ctx(b).starts if off_grid_start(question, b) else ()
        return _clarify(r, "out_of_grid", cand), "clarify:unknown->clarify:out_of_grid"
    return r, None
