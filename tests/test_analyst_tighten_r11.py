"""Round 11 (ruling R11.1): the tighten-only post-route step (``router.tighten``).

The step may only (a) turn a preset answer into a clarification or (b) change one clarification
reason into another. These tests prove that property over the cross product of every committed
question (dev set, held-out v1-v5) plus synthetic questions with every kind of route a stage
can produce, and check the two changes it makes (v5_034's missing year, v5_061's off-grid start)
and their boundary twins. Synthetic page only (starts 2026 and 2030, last year 2032).

Developed after held-out v5 was spent, on the spent sets: development only, unscored.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hydrophysics_analyst import eval as EV
from hydrophysics_analyst import presets as PR
from hydrophysics_analyst import testing
from hydrophysics_analyst.ask import answer_route
from hydrophysics_analyst.llm.backend import FakeBackend
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.router import V1_ROUTED, route, routed_presets
from hydrophysics_analyst.router import tighten as T
from hydrophysics_analyst.router.llm import LLMRouter
from hydrophysics_analyst.router.schemas import CLARIFY, Route

ROOT = Path(__file__).resolve().parents[1]
EVALSETS = ROOT / "hydrophysics_analyst" / "evalsets"


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("t11")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


def _items() -> list[dict]:
    out = []
    for f in sorted(EVALSETS.glob("eval_questions*.jsonl")):
        out += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    return out


ITEMS = _items()

# (a) fires: an indefinite lead and no reporting year
MISSING_YEAR = [
    "灌溉減兩成，過幾年之後的水位會怎樣？",
    "养殖减三成，再过几年之后的水位会是怎样？",
    "How does a 30% irrigation cut look a few years from now?",
    "If aquaculture is retired, what is the head in a few years?",
    "Halve livestock pumping: where are heads several years later?",
    "Rank the classes by head rise some years down the road.",
    "灌溉減30%，未來幾年水位會怎樣？",
    "從2030年開始養殖減半，幾年後水位如何？",
    "停止工業抽水，數年後的地下水位？",
    "Cut irrigation 30% starting 2030; what is the head in the years ahead?",
]
# (a) must not fire: a reporting year or a definite lead is given, or no vague lead at all
KEEPS_PRESET = [
    "灌溉減兩成，三年後的水位會怎樣？",
    "Cut irrigation 30%: head in 3 years?",
    "How does a 30% irrigation cut look a few years from now, say in 2030?",
    "灌溉減兩成，過幾年之後，也就是2030年的水位？",
    "Cut irrigation 30%: what is the head in twenty thirty?",
    "灌溉減兩成，民國119年的水位？",
    "Cut irrigation 30%; what is the head?",
    "灌溉減三成以後，水位會怎樣？",
    "Cut irrigation 30% from 2026 to 2032: the head?",
    "In the future, what does halving aquaculture do to the head?",
]
# (b) fires on a clarification with reason unknown: a start or size off the grid
OFF_GRID = [
    "如果養殖減一半但拖到2029年才實施，到2032年地下水會如何？",
    "如果养殖减一半但拖到2029年才实施，到2032年地下水会如何？",
    "Halve aquaculture, but not until 2028: head in 2032?",
    "Cut irrigation 30% but delay it until 2027: the head?",
    "灌溉減三成，從2029年開始實施，看2032年水位？",
    "What does a 22% domestic cut do to heads?",
    "What if irrigation is cut by a third?",
    "要是農田灌溉只剩原本三分之一的抽水量，水頭怎麼變？",
    "Aquaculture cut 12.5%: head?",
]
ON_GRID = [
    "Halve aquaculture, but wait until 2030: head in 2032?",
    "灌溉減三成，2030年開始，2032年水位？",
    "What does a 25% domestic cut do to heads?",
    "Can you check this for me?",
    "這樣跟上次比起來呢？",
]
# review of 5511126 (F1): (a) must not fire. A definite year in another form; past or duration
# wording; a vague phrase about when the policy starts; now-vs-later; "how many years until"
NOT_MISSING_YEAR = [
    "Cut irrigation 30%: what is the head a few years from now, in '30?",
    "Cut irrigation 30%: what is the head a few years from now, in 32?",
    "Cut irrigation 30%: what is the head a few years from now, in year 32?",
    "Cut irrigation 30%: head a few years from now (ROC 119)?",
    "灌溉減兩成，過幾年之後，民國121年的水位？",
    "灌溉減兩成，過幾年之後，121年的水位？",
    "灌溉減兩成，過幾年之後，一二一年的水位？",
    "灌溉減兩成，過幾年之後，民國一二一年的水位？",
    "灌溉減兩成，過幾年之後，二〇三〇年的水位？",
    "灌溉減兩成，過幾年之後，二〇三二年的水位？",
    "灌溉減兩成，過幾年之後，2030年的水位？",
    "灌溉減兩成，過幾年，到明年水位會怎樣？",
    "灌溉減兩成，過幾年，後年水位會怎樣？",
    "灌溉減兩成，幾年後（大概2030年）水位？",
    "Cut irrigation 30%: in a few years, by the end of the decade, the head?",
    "Cut irrigation 30%: in a few years, in 2030 or so, the head?",
    "Cut irrigation 30%: in a few years, next year, the head?",
    "Cut irrigation 30%: in a few years -- the year after next -- what is the head?",
    "Cut irrigation 30%: in a few years, by 2030 at the latest, what is the head?",
    "Cut irrigation 30%: in a few years, i.e. 2030, what is the head?",
    "Cut irrigation 30%: head in a few years, i.e. in three years?",
    "Cut irrigation 30%: head in a few years, i.e. in 3 yrs?",
    "Cut irrigation 30%: head in a few years, so around twenty-thirty?",
    "Cut irrigation 30%: head in a few years (by 2030)?",
    "Cut irrigation 30%: head in a few years, at the end of the page (2032)?",
    "Cut irrigation 30%: head in a few years -- at the page's last year?",
    "Cut irrigation 30%: head a few years from now, at the end of the run?",
    "Cut irrigation 30%: head in a few years, at the final year?",
    "Cut irrigation 20% for a few years: what happens to the head?",
    "Cut irrigation 20% for a few years, what is the head in 2032?",
    "After many years of overpumping, what does a 30% irrigation cut do to heads?",
    "Over many years the aquifer has declined; what does halving aquaculture do to the head?",
    "Aquaculture has been shrinking for some years; what does a 50% cut do?",
    "It has been a few years since the census; what does retiring aquaculture do to heads?",
    "After a few years of drought, what does a 30% irrigation cut do to the head?",
    "In some years the head drops more; what does a 30% irrigation cut do?",
    "I've lived here for many years. If irrigation is cut 30%, what happens to the water level?",
    "Rank the levers by head gain; I've worked on this for several years.",
    "For several years farmers have asked: which class cut raises heads most?",
    "Which lever ranks first, given many years of data?",
    "Many years of data went into this; how does the solved fan compare?",
    "After several years of debate, compare a 2026 and a 2030 start for retiring aquaculture.",
    "Many years down the road, does the solved fan compare well?",
    "Cut irrigation 30% but start a few years later: what happens to heads?",
    "Is it better to cut aquaculture 50% now or a few years later?",
    "Over the next few years we will cut irrigation 30%. What is the head?",
    "Over the next few years, what does a 20% irrigation cut do to the head?",
    "We plan a 30% irrigation cut within a few years; head?",
    "養殖減半，現在做跟過幾年再做差多少？",
    "養殖減半，現在開始跟晚幾年開始，水位差多少？",
    "近幾年間地層下陷嚴重，灌溉減三成水位會怎樣？",
    "過去幾年間地下水下降，養殖減半水位會怎樣？",
    "這幾年間養殖抽水增加，養殖減半的話水位？",
    "數年間抽水量都很高，灌溉減三成會怎樣？",
    "我在這裡住了好幾年了，灌溉減三成水位會怎樣？",
    "灌溉減三成，要幾年後水位才會回升？",
    "灌溉減三成，幾年內水位會回升嗎？",
    "灌溉減三成，數年來的水位資料顯示下降，減了以後會怎樣？",
    "Cut irrigation 30%: head next year?",
    "灌溉減三成，明年水位？",
    "灌溉減三成，後年水位？",
    "Cut irrigation 30%: head by the end of the decade?",
]
# review F2: (b) must not fire. Ordinals, an on-grid fraction, a percentage that is a share or a
# change and not a cut, a reporting year after wait / 等到 / 晚到
NOT_OFF_GRID = [
    "Can you give me a third example of a lever?",
    "I have a third question: what about the solved fan?",
    "What is a third option besides irrigation and aquaculture cuts?",
    "Cut irrigation for a third of the farms by 30%: head?",
    "養殖減六分之三，水位？",
    "養殖減八分之四，水位？",
    "Cut irrigation 35 %: head?",
    "Cut irrigation 30.0%: head?",
    "With 90% confidence, what does a 30% cut do?",
    "Irrigation is 68% of pumping; what does halving it do?",
    "養殖佔抽水量的43%，養殖減半水位會怎樣？",
    "Head rose 12% in my well since 2020; what does a 30% irrigation cut do?",
    "灌溉減三成，等到2032年水位會怎樣？",
    "Cut irrigation 30%, then wait until 2032: head?",
    "If we wait until 2031 to look, what does a 30% irrigation cut do?",
    "灌溉減三成，晚到2031年水位會變怎樣？",
    "灌溉減三成，延到2030年才開始，2032水位？",
    "What happens 1/3 of the way through the run?",
    "每三分之一的農地休耕，水位？",
    "灌溉減少百分之三十，水位？",
    "灌溉減三成，2032年之前要延到2029年再說嗎？",
    "Hold off until 2030, then cut aquaculture 50%: head in 2032?",
    "Postpone the irrigation cut to 2030: head?",
]
QUESTIONS = ([it["question"] for it in ITEMS] + MISSING_YEAR + KEEPS_PRESET + OFF_GRID + ON_GRID
             + NOT_MISSING_YEAR + NOT_OFF_GRID)

CUTS = [{"class": "irrigation", "pct": 30}]
PRESET_PARAMS = {"policy_head": {"cuts": CUTS, "start": 2026, "year": 2032},
                 "rail_internals": {"cuts": CUTS, "start": 2026, "year": 2032},
                 "township_internals": {"cuts": CUTS, "start": 2026, "year": 2032, "town": 1},
                 "start_compare": {"cuts": CUTS, "year": 2032},
                 "lever_rank": {"year": 2032, "by": "head"},
                 "solved_compare": {"year": 2032},
                 "cheapest_for_head": {"target_m": 0.5, "start": 2026, "year": 2032},
                 "how_sure": {}}


def _routes() -> list[Route]:
    """Every kind of route a stage can produce (and a few malformed ones)."""
    rs = [Route(preset=p, params=dict(v), confidence=1.0, source="llm") for p, v in PRESET_PARAMS.items()]
    for reason in PR.REFUSALS:
        rs.append(Route(preset="refuse", params={"reason": reason}, refusal=reason, confidence=1.0))
        # the LLM may set a refusal and a clarification at once; a refusal reason with a preset
        rs.append(Route(preset="refuse", params={"reason": reason}, refusal=reason,
                        clarify={"reason": "unknown", "candidates": []}))
        rs.append(Route(preset="policy_head", params=dict(PRESET_PARAMS["policy_head"]), refusal=reason))
    rs.append(Route(preset="refuse", params={"reason": "no_run"}))          # refuse without a reason
    # preset routes refused or clarified when they run: no policy, past the page, missing cut
    rs += [Route(preset="township_internals", params={"cuts": [{"class": "irrigation", "pct": 0}],
                                                     "start": 2026, "year": 2032, "town": 1}),
           Route(preset="policy_head", params={"cuts": CUTS, "start": 2026, "year": 2040}),
           Route(preset="policy_head", params={"start": 2026, "year": 2032}),
           Route(preset="rail_internals", params={"cuts": CUTS, "start": 2027, "year": 2032})]
    for reason in CLARIFY + ("no_llm", "beyond_horizon", "not_a_reason"):
        rs.append(Route(confidence=1.0, clarify={"reason": reason, "candidates": [2026, 2030]}))
        rs.append(Route(preset="policy_head", params=dict(PRESET_PARAMS["policy_head"]), source="llm",
                        clarify={"reason": reason, "candidates": []}))
    rs += [Route(), Route(preset="clarify"), Route(preset="clarify", params={"cuts": CUTS})]
    # review F4: a falsy clarify ({}) runs the preset; a clarify with no reason cannot be answered
    rs += [Route(preset="policy_head", params=dict(PRESET_PARAMS["policy_head"]), clarify={}),
           Route(clarify={}), Route(preset="policy_head", clarify={"candidates": []}),
           Route(clarify={"candidates": []})]
    return rs


ROUTES = _routes()


def answer_kind(b, r: Route) -> str:
    """What the user gets for a route: ``ask.ask``'s answer_route, then its validator fallback
    (a violation is answered by a ``no_run`` refusal)."""
    from hydrophysics_analyst.validate import validate
    try:
        ans = answer_route(b, r, r.lang or "en")
    except Exception:                  # noqa: BLE001
        return "error"
    if ans.refusal is None and validate(ans, b):
        return "refusal"
    return "clarify" if ans.clarify else "refusal" if ans.refusal else "preset"


def _key(r: Route) -> str:
    return json.dumps(r.to_json(), sort_keys=True, default=str)


def test_outcome_matches_the_answer(b):
    """``tighten.outcome`` reads a non-preset route the way it is answered; a preset route may
    still be refused or clarified when it runs (and the step then leaves it alone)."""
    kinds = {answer_kind(b, r) for r in ROUTES if T.outcome(r) == "preset"}
    assert {"preset", "refusal"} <= kinds
    for r in ROUTES:
        if T.outcome(r) != "preset" and not (r.refusal is not None and r.preset != "refuse"):
            assert answer_kind(b, r) == T.outcome(r), r


def test_falsy_clarify_is_read_like_answer_route(b):
    """Review F4: ``clarify={}`` is falsy, so answer_route runs the preset; the step reads it the
    same way (and may clarify it); a clarify with no reason is an error, left unchanged."""
    q = "灌溉減兩成，過幾年之後的水位會怎樣？"
    r = Route(preset="policy_head", params=dict(PRESET_PARAMS["policy_head"]), clarify={})
    assert T.outcome(r) == answer_kind(b, r) == "preset"
    t, change = T.tighten(b, q, None, r)
    assert t.clarify["reason"] == "missing_year" and change
    r = Route(clarify={})
    assert T.outcome(r) == answer_kind(b, r) == "clarify" and T.shown_reason(r) == "unknown"
    assert T.tighten(b, "What if irrigation is cut by a third?", None, r)[0].clarify["reason"] \
        == "out_of_grid"
    for r in (Route(clarify={"candidates": []}),
              Route(preset="policy_head", clarify={"candidates": []})):
        assert T.outcome(r) == answer_kind(b, r) == "error"
        assert T.tighten(b, "What if irrigation is cut by a third?", None, r) == (r, None)


def test_tighten_only_over_every_question_and_route(b):
    """The R11.1 property, over all committed questions x every route kind, judged on the
    answers: a refused route is returned as the same object; a clarification stays one (only
    unknown -> out_of_grid); a preset answer stays the same route or becomes a bare
    missing_year clarification; nothing is turned into a preset or out of a refusal."""
    kind = {}

    def k(r):
        return kind.setdefault(_key(r), answer_kind(b, r))
    n_a = n_b = 0
    for q in QUESTIONS:
        for r in ROUTES:
            before = _key(r)
            t, change = T.tighten(b, q, None, r)
            assert _key(r) == before                                   # input untouched
            k0 = k(r)
            if k0 in ("refusal", "error") or T.is_refusal(r):
                assert t is r and change is None, (q, r)
                continue
            if t is r:
                assert change is None
                continue
            assert k(t) == "clarify" and not T.is_refusal(t) and t.preset is None and t.params == {}
            if k0 == "preset":
                assert t.clarify["reason"] == "missing_year" and r.preset in T.YEAR_PRESETS, (q, r)
                n_a += 1
            else:
                assert k0 == "clarify"
                assert T.shown_reason(r) == "unknown", (q, r)
                assert t.clarify["reason"] == "out_of_grid"
                n_b += 1
    assert n_a and n_b                 # both changes were exercised


def test_route_never_loosens(b):
    """Through ``router.route`` with no LLM, on every committed item (with its ``prev``) and the
    synthetic questions: the outcome after the step is the outcome before it (the rules stage
    never routes a preset, so here only a clarification reason can change)."""
    from hydrophysics_analyst.router import _route
    cases = [(it["question"], it.get("prev")) for it in ITEMS]
    cases += [(q, None) for q in MISSING_YEAR + KEEPS_PRESET + OFF_GRID + ON_GRID]
    for q, prev in cases:
        r0, _ = _route(b, q, prev)
        r1, info = route(b, q, prev)
        k0, k1 = T.outcome(r0), T.outcome(r1)
        assert k0 == k1 or (k0, k1) == ("preset", "clarify"), q
        if k0 == "refusal":
            assert r1.to_json() == r0.to_json() and "tighten" not in info, q


def _llm(b, out: dict) -> LLMRouter:
    d = {"preset": None, "cuts": None, "start": None, "year": None, "town": None, "by": None,
         "target_m": None, "refusal": None, "clarify": None, "inherit": [], "answer_lang": "zh"}
    return LLMRouter(b, FakeBackend([d | out]), presets=routed_presets())


@pytest.mark.parametrize("q", MISSING_YEAR)
def test_vague_lead_is_clarified_missing_year(b, q):
    llm = _llm(b, {"preset": "policy_head", "cuts": CUTS, "start": 2026, "year": 2032})
    r, info = route(b, q, None, llm=llm)
    assert info["stage"] == "llm" and r.clarify == {"reason": "missing_year", "candidates": []}
    assert r.preset is None and r.params == {} and info["tighten"].endswith("missing_year")
    assert answer_route(b, r, "en").clarify["reason"] == "missing_year"


@pytest.mark.parametrize("q", KEEPS_PRESET + NOT_MISSING_YEAR)
def test_a_given_year_keeps_the_preset(b, q):
    r = Route(preset="policy_head", params=dict(PRESET_PARAMS["policy_head"]), source="llm")
    assert T.tighten(b, q, None, r) == (r, None)


def test_how_sure_takes_no_year(b):
    r = Route(preset="how_sure", source="llm")
    assert T.tighten(b, "How sure is the head result a few years from now?", None, r) == (r, None)


def test_follow_up_vague_lead(b):
    prev = {"preset": "policy_head", "params": dict(PRESET_PARAMS["policy_head"])}
    llm = _llm(b, {"preset": "policy_head", "inherit": ["cuts", "start", "year"]})
    r, _ = route(b, "And a few years after that?", prev, llm=llm)
    assert r.clarify["reason"] == "missing_year"


@pytest.mark.parametrize("q", OFF_GRID)
def test_unknown_clarification_off_grid_becomes_out_of_grid(b, q):
    r = Route(preset="policy_head", params={"cuts": CUTS}, clarify={"reason": "unknown",
                                                                    "candidates": []}, source="llm")
    t, change = T.tighten(b, q, None, r)
    assert t.clarify["reason"] == "out_of_grid" and change.endswith("out_of_grid"), q
    assert t.preset is None and t.params == {}


def test_v5_061_through_the_router(b):
    q = "如果養殖減一半但拖到2029年才實施，到2032年地下水會如何？"
    r, info = route(b, q, None, llm=_llm(b, {"clarify": "unknown"}))
    assert info["stage"] == "llm" and r.clarify == {"reason": "out_of_grid", "candidates": [2026, 2030]}
    # a question the guards refuse stays refused whatever it says about the grid
    q = "要是農田灌溉只剩原本三分之一的抽水量，水頭怎麼變？"
    r, info = route(b, q, None, llm=_llm(b, {"clarify": "unknown"}))
    assert info["stage"] == "hard_refusal" and r.refusal == "volumes_money" and "tighten" not in info


@pytest.mark.parametrize("q", ON_GRID + NOT_OFF_GRID)
def test_unknown_on_grid_stays_unknown(b, q):
    r = Route(clarify={"reason": "unknown", "candidates": []}, source="llm")
    assert T.tighten(b, q, None, r) == (r, None)


@pytest.mark.parametrize("reason", ["missing_cut", "missing_year", "ambiguous_town", "no_llm"])
def test_only_unknown_is_corrected(b, reason):
    """A specific reason the LLM or the rules gave is never second-guessed."""
    r = Route(clarify={"reason": reason, "candidates": []}, source="llm")
    for q in OFF_GRID:
        assert T.tighten(b, q, None, r) == (r, None)


def test_refusals_are_untouched_on_target_questions(b):
    for q in MISSING_YEAR + OFF_GRID:
        for reason in PR.REFUSALS:
            r = Route(preset="refuse", params={"reason": reason}, refusal=reason, confidence=1.0)
            assert T.tighten(b, q, None, r) == (r, None)


def test_labels_of_committed_sets(b):
    """On every committed item the step fires only where the item's own label agrees (a
    regression check on spent and dev sets, not a score)."""
    for it in ITEMS:
        if T.missing_year(it["question"], b):
            assert it["expect"] == {"clarify": "missing_year"}, it["id"]
        if T.off_grid(it["question"], b):
            assert it["expect"] == {"clarify": "out_of_grid"}, it["id"]
    fire_a = {it["id"] for it in ITEMS if T.missing_year(it["question"], b)}
    assert fire_a == {"q138", "v3_060", "v5_034"}
    # (b) reads every out_of_grid item of the committed sets
    want_b = {it["id"] for it in ITEMS if it["expect"] == {"clarify": "out_of_grid"}}
    assert {it["id"] for it in ITEMS if T.off_grid(it["question"], b)} == want_b


def test_route_code_pins_cover_the_step():
    assert "hydrophysics_analyst/router/tighten.py" in EV.ROUTE_CODE
    assert "tighten" in EV.ROUTE_FUNCS["hydrophysics_analyst.router.tighten"]
    assert EV.route_functions_moved() == []


ROWS = ROOT / "results" / "analyst" / "eval" / "heldout_v5" / "opus-low.jsonl"


@pytest.mark.skipif(not ROWS.exists(), reason="recorded eval rows are not in this checkout")
def test_recorded_v5_rows(b):
    """The step on v5's recorded opus-low rows (spent; no LLM): it changes v5_034 and v5_061 in
    each of the three runs, both to the expected label, and nothing else."""
    items = {it["id"]: it for it in ITEMS if it["id"].startswith("v5_")}
    changed = []
    for x in ROWS.read_text(encoding="utf-8").splitlines():
        r = json.loads(x)
        g = r["got"]
        if "error" in g:
            continue
        kind = ("clarify" if g.get("clarify") else "refusal" if g.get("preset") == "refuse"
                else "preset" if g.get("preset") else "clarify")
        d = T.decide(b, items[r["id"]]["question"], kind, g.get("preset"),
                     g.get("clarify") or ("unknown" if kind == "clarify" else None))
        if d:
            assert items[r["id"]]["expect"] == {"clarify": d}
            changed.append(r["id"])
    assert sorted(changed) == ["v5_034"] * 3 + ["v5_061"] * 3


def test_v1_routed_presets_that_report_a_year():
    assert set(T.YEAR_PRESETS) == set(V1_ROUTED + ("cheapest_for_head",)) - {"how_sure"}
