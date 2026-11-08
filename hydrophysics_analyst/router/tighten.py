"""Round 11 (ruling R11.1): a tighten-only post-route step, run after the hard refusals, the rules
and the LLM have routed a question (``router.route``).

It may do exactly two things, and nothing else:

(a) turn a **preset** answer into a clarification: a preset that reports at a year (every routed
    preset but ``how_sure``), for a question that leads by an indefinite number of years ("a few
    years from now", 過幾年之後, 未來幾年) and gives no reporting year, is clarified
    ``missing_year``. OPS.md: the page's default year (2032) applies "when the question gives
    none"; a question asking about "a few years from now" asks about a year it does not give, so
    the default would answer a different question. The LLM's own prompt says the same ("only
    when the question gives no year at all"). This is the indefinite twin of the guards' leads:
    ``guards.beyond_horizon`` reads "in N years" / N年後 as start + N, and here N is not given.
(b) change one clarification reason into another: the catch-all ``unknown`` becomes
    ``out_of_grid`` when the question names a start or a cut size off the page grid (OPS.md:
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
from .rules import _pcts, parse_years
from .schemas import Route

# the reasons presets.clarification shows as themselves; any other is shown as "unknown"
CLARIFY_SHOWN = ("ambiguous_town", "missing_year", "missing_cut", "out_of_grid", "no_llm")
# the routed presets that report at a year (schemas.PRESET_PARAMS); how_sure takes none
YEAR_PRESETS = ("policy_head", "start_compare", "township_internals", "rail_internals",
                "lever_rank", "solved_compare", "cheapest_for_head")

# an indefinite number of years ("a couple of" is two years to the guards, so it is not here)
_EN_V = r"(?:a\s+few|few|several|some|a\s+number\s+of|many)"
VAGUE_LEAD_EN = [
    rf"\b{_EN_V}\s+(?:more\s+)?years?\s+(?:later|on|out|ahead|hence|into|in|down\s+the\s+(?:road|line)|"
    rf"after|from|past|since|beyond)\b",
    rf"\b(?:in|after|within|over|for|across|during)\s+(?:the\s+)?(?:next\s+|coming\s+|following\s+|"
    rf"another\s+)?{_EN_V}\s+(?:more\s+)?years?\b",
    rf"\b(?:next|coming|following|another)\s+{_EN_V}\s+years?\b",
    r"\b(?:in\s+)?the\s+years\s+(?:ahead|to\s+come)\b",
]
_ZH_V = r"(?<![十百])(?:好幾|幾|數|若干)\s*個?"
VAGUE_LEAD_ZH = [
    rf"{_ZH_V}\s*(?:多)?\s*年\s*(?:後|之後|以後|過後|內|間|期間)",
    rf"(?:再過|過|往後|之後|未來|接下來|今後)\s*的?\s*{_ZH_V}\s*年",
]

# a start the rules stage does not read as one: "拖到2029年才實施", "not until 2029"
LATE_START = [
    r"(?:拖|延|延後|延遲|晚|等|推遲)\s*(?:到|至)\s*(20\d\d)\s*年?",
    r"(?<!\d)(20\d\d)\s*年?\s*(?:才|再)\s*(?:開始|實施|啟動|上路|執行|推動|施行|生效)",
    r"\bnot\s+(?:until|till|before)\s+(20\d\d)\b",
    r"\b(?:delay\w*|postpon\w*|push\w*\s+back|wait\w*|hold\w*\s+off)\b[^.;?!]{0,30}?"
    r"\b(?:until|to|till)\s+(20\d\d)\b",
    r"\b(?:takes?|taking|took)\s+effect\s+(?:only\s+)?(?:in|from)\s+(20\d\d)\b",
]
# a cut size off the five-percent grid given as a fraction
OFF_GRID_FRACTION = [r"\b(?:a|one|two)[\s\-]+(?:third|sixth|seventh|ninth)s?\b",
                     r"\b(?:a|one|three|five|seven)[\s\-]+eighths?\b",
                     r"(?<!\d)[12]\s*/\s*3(?!\d)", r"[三六七九]\s*分\s*之\s*[一二三四五六七八]",
                     r"八\s*分\s*之\s*[一三五七]"]


def _has(pats, s: str) -> bool:
    return any(re.search(p, s, re.I) for p in pats)


def vague_lead(s: str) -> bool:
    """The question leads from the start by an indefinite number of years."""
    return _has(VAGUE_LEAD_EN + VAGUE_LEAD_ZH, G.norm(s))


def gives_year(s: str, b: PageBundle) -> bool:
    """The question names a reporting year: a calendar, spelled, two-digit or ROC year read as
    a reporting year, or a definite lead ("in 3 years", 三年後) that the guards read as one."""
    s = G.norm(s)
    if parse_years(b, s)[1] is not None or G.spelled_years(s):
        return True
    if re.search(r"(?:^|[\s(（])['’‘](\d{2})(?![\d%])", s):
        return True
    if re.search(r"(?:民國|ROC\s*)\s*(\d{2,3}|[〇零一二三四五六七八九十百]+)\s*年?|(?<![\d.])(1[1-9]\d)\s*年",
                 s, re.I):
        return True
    for pat, _ in G.EN_LEAD:
        if any(G._en_number(m.group(1)) is not None for m in re.finditer(pat, s, re.I)):
            return True
    for pat, _ in G.ZH_LEAD:
        if any(G.zh_number(m.group(1)) is not None for m in re.finditer(pat, s)):
            return True
    return False


def missing_year(s: str, b: PageBundle) -> bool:
    """(a): an indefinite lead and no reporting year."""
    return vague_lead(s) and not gives_year(s, b)


def off_grid_start(s: str, b: PageBundle) -> bool:
    s = G.norm(s)
    c = Q.ctx(b)
    if parse_years(b, s)[0] == -1:
        return True
    return any(int(m.group(1)) not in c.starts for p in LATE_START for m in re.finditer(p, s, re.I))


def off_grid_size(s: str) -> bool:
    s = G.norm(s)
    return any(v not in PR.PCT for _, v in _pcts(s)) or _has(OFF_GRID_FRACTION, s)


def off_grid(s: str, b: PageBundle) -> bool:
    """(b): a start or a cut size off the page grid. A year after the page is refused upstream
    (``beyond_horizon``); a year before it is history, not a grid miss, and is not read here."""
    return off_grid_start(s, b) or off_grid_size(s)


def is_refusal(r: Route) -> bool:
    return r.refusal is not None or r.preset == "refuse"


def outcome(r: Route) -> str:
    """What ``ask.answer_route`` makes of a route: ``refusal``, ``clarify`` or ``preset``
    (``presets.clarification`` answers a ``beyond_horizon`` clarification as a refusal)."""
    if r.clarify is not None:
        return "refusal" if r.clarify.get("reason") == "beyond_horizon" else "clarify"
    if r.preset == "refuse":
        return "refusal"
    if r.preset and r.preset != "clarify":
        return "preset"
    return "clarify"


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
    if kind == "refusal":                  # beyond_horizon given as a clarification
        return r, None
    reason = None
    if kind == "clarify":                  # the reason the answer shows (presets.clarification)
        reason = (r.clarify or {}).get("reason")
        reason = reason if reason in CLARIFY_SHOWN else "unknown"
    d = decide(b, question, kind, r.preset, reason)
    if d == "missing_year" and not answered_as_preset(b, r):
        return r, None
    if d == "missing_year":                # (a) preset -> clarification
        return _clarify(r, "missing_year"), f"{r.preset}->clarify:missing_year"
    if d == "out_of_grid":                 # (b) clarify unknown -> clarify out_of_grid
        cand = Q.ctx(b).starts if off_grid_start(question, b) else ()
        return _clarify(r, "out_of_grid", cand), "clarify:unknown->clarify:out_of_grid"
    return r, None
