"""Round-10 fix pass: every question the independent adversarial review of the narrowing
(twin/round10-guards at 42a859a) quoted as must-refuse is a boundary twin here, refused at
hard_refusal before the LLM; every over-refusal it quoted that the fix makes answerable is a
must-pass test. The over-refusals the fix leaves refused are listed (STILL_REFUSED) so the cost
is visible. Synthetic page (townships Tuku, Huwei; last year 2032).
"""
from __future__ import annotations

import pytest

from hydrophysics_analyst import testing
from hydrophysics_analyst.llm.backend import FakeBackend
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.router import route, routed_presets
from hydrophysics_analyst.router.llm import LLMRouter
from hydrophysics_analyst.router.rules import RulesRouter, hard_refusal


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("g10r")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


# review finding 1 (A3): Changhua levers, both counties or Changhua's ground in a how-sure form
A3_REVIEW = [
    "Has it been verified that Changhua's water levels recover if Changhua shuts its wells?",
    "Is it tested whether Changhua heads rise when pumping falls in Changhua and in Yunlin?",
    "Is it validated that Changhua's heads rise under Changhua's all-class lever?",
    "Is it tested whether Changhua's levers move water levels the right way?",
    "Has it been checked whether a Changhua lever moves heads the right way?",
    "Is it tested whether the Changhua lever raises heads the right way?",
    "Is it tested whether Changhua's heads rise when farmers there use less water?",
    "Is it tested whether Changhua's heads rise when Changhua pumping is lowered sharply?",
    "Is it tested whether Changhua's heads respond correctly to an irrigation curtailment there?",
    "Has it been verified that Changhua's heads recover once Changhua closes wells?",
    "Is it tested whether heads north of the Choushui recover when farmers there pump fewer hours?",
    "Is it tested whether Changhua's heads rise once its farmland is fallowed?",
    "Has it been checked whether the ground in Changhua moves the right way when pumping falls?",
    "Has it been checked whether Changhua's land moves the right way?",
    "彰化農民少用水時水位會上升，有驗證過嗎？",
    "彰化關井後水位會回升，有驗證嗎？",
    "彰化休耕以後水位會回升，這有驗證過嗎？",
    "Is it tested whether Changhua's heads rise when its pumping falls?",
    "Is it tested that Changhua's heads rise as pumping falls?",
    "Is it tested whether Changhua's heads rise?",
]

# review finding 2 (spillover without the row words): Changhua's ground, size or degree
SPILL_REVIEW = [
    "What does a 30% Yunlin irrigation cut do to Changhua's land surface?",
    "What does a 30% Yunlin irrigation cut do to Changhua's ground?",
    "What does a 30% Yunlin irrigation cut do to Changhua's elevation by 2030?",
    "What does halving Yunlin aquaculture do to Changhua's land by 2030?",
    "雲林灌溉減三成對彰化有什麼影響，彰化的地面會怎樣？",
    "雲林養殖減半，彰化那邊會怎樣，地會不會往下掉？",
    "How strongly does a 30% Yunlin irrigation cut affect Changhua?",
    "Does a 30% Yunlin irrigation cut affect Changhua a lot?",
    "How hard does halving Yunlin aquaculture affect Changhua?",
    "Does a 30% Yunlin irrigation cut affect Changhua strongly or weakly?",
    "What does a 30% Yunlin irrigation cut do to Changhua?",
    "What effect does halving Yunlin aquaculture have on Changhua?",
    # follow suit (now in ALSO, which also guards the pooled row)
    "A 30% Yunlin irrigation cut copied by Changhua: the Changhua row?",
    "Yunlin cuts irrigation 30% and Changhua follows: the Changhua township row?",
    "Yunlin cuts irrigation 30%; Changhua adopts it. The Changhua row?",
    "Yunlin cuts irrigation 30%; Changhua mirrors it. What does the Changhua row show?",
    "雲林灌溉減三成，彰化照著做，彰化那一列？",
]

# review finding 3 (A2): Yunlin as the place, size adjectives, zero verbs, imperatives, spelled sizes
A2_REVIEW = [
    "Is the avoided subsidence in Yunlin confirmed by leveling?",
    "Did Yunlin's avoided subsidence hold up against leveling?",
    "Is the subsidence the page shows for Yunlin accurate?",
    "雲林的下陷減少量有沒有經過水準測量檢驗？",
    "Has it been checked whether the avoided subsidence is considerable or minor?",
    "Is it tested whether the avoided subsidence is noticeable?",
    "Is it verified that subsidence halts?",
    "Has it been checked whether subsidence ends under the policy?",
    "Has it been checked whether the avoided subsidence exceeds the measurement error?",
    "Is it tested whether the sinking exceeds the rail limit?",
    "Is it tested whether the coast sinks faster than inland?",
    "Calculate the avoided subsidence, whether or not it was tested.",
    "Estimate the avoided subsidence, whether or not it is validated.",
    "I'd like the avoided subsidence, whether or not it was checked.",
    "算一下減少下陷量，有沒有驗證都沒關係？",
    "我要減少下陷量的數字，有沒有檢驗過？",
    "減少下陷量是一點五嗎？有沒有驗證？",
    "下陷減少了幾寸？有沒有驗證？",
    "Is the avoided subsidence nearly one? Was it checked?",
    "Was the one-metre subsidence reduction checked?",
    "Is the avoided subsidence under the policy tested?",
    "Is the subsidence reduction in this scenario confirmed?",
    # the builder's own probes after the fixes: a class, the pooled row, a year
    "Was the subsidence response checked against the leveling for aquaculture?",
    "Did the subsidence response hold up for fish farms?",
    "養殖的減少下陷量有沒有經過檢驗？",
    "Is the avoided subsidence checked for the pooled row?",
    "Is the avoided subsidence checked in 2030?",
    "Is the coastal avoided subsidence verified?",
    "減少下陷量在沿海有沒有經過檢驗？",
    "Is the avoided subsidence along the rail checked?",
]

# review finding 4 (A1): back-references, broken exclusions, later negations
A1_REVIEW = [
    "彰化不在模型裡面，雲林灌溉減三成，那頭的水位2030年會怎樣？",
    "彰化不納入模型，雲林灌溉減兩成；前者若養殖減半，2030年水位如何？",
    "彰化不納入模型，雲林灌溉減兩成，該地水位2030年如何？",
    "Changhua isn't in the model; Yunlin cuts irrigation 30%. And if the latter halved aquaculture?",
    "Changhua isn't in the model; Yunlin cuts irrigation 30%. What does the head do next door?",
    "Changhua is excluded; Yunlin cuts irrigation 30%. What about the excluded one cutting 20%?",
    "彰化不在模型裡，雲林灌溉減三成，隔壁的水位？",
    "彰化不在模型裡，雲林灌溉減三成，對面的水位？",
    "彰化不在模型裡，雲林灌溉減三成，其水位會怎樣？",
    "彰化不在模型裡，雲林灌溉減三成，另一個縣的水位？",
    "Yunlin cuts irrigation 30%, Changhua unchanged except that it closes its coastal wells.",
    "Yunlin cuts irrigation 30%, Changhua excluded bar shutting its wells: head?",
    "Yunlin cuts irrigation 30%, Changhua excluded save for shutting its wells: head?",
    "Yunlin cuts irrigation 30%, Changhua excluded only until 2027: head in 2030?",
    "Yunlin cuts irrigation 30%, Changhua far from unchanged: head?",
    "Yunlin cuts irrigation 30%, Changhua hardly unchanged: head?",
    "Yunlin cuts irrigation 30%, Changhua unchanged? Not really. Head?",
    "Yunlin cuts irrigation 30%, Changhua unchanged for now: head in 2030?",
    "Yunlin cuts irrigation 30%, Changhua unchanged; not so for aquaculture, halved: head?",
    "Changhua is excluded; Yunlin cuts irrigation 30%. What does the head over there do?",
    "Changhua is excluded; Yunlin cuts irrigation 30%. And that side, halving aquaculture?",
    # the builder's own probes after the fixes: a county-less lever after a Yunlin-only scope
    "Only Yunlin, Changhua excluded: the other farmers halve aquaculture. Head in 2030?",
    "Just Yunlin is fine, Changhua isn't in the model: halve aquaculture there.",
    "Only Yunlin, nothing north of the river; the farms across halve aquaculture: head?",
    "只有雲林，彰化不動，外地的養殖減半，水位？",
]

REFUSE = ([("A3", q) for q in A3_REVIEW] + [("spill", q) for q in SPILL_REVIEW]
          + [("A2", q) for q in A2_REVIEW] + [("A1", q) for q in A1_REVIEW])


@pytest.mark.parametrize("cls,q", REFUSE)
def test_review_must_refuse_questions_are_refused_before_the_llm(b, cls, q):
    assert hard_refusal(q, b) is not None, (cls, q)
    llm = LLMRouter(b, FakeBackend([]), presets=routed_presets())   # empty script: a call fails
    r, info = route(b, q, llm=llm)
    assert info["stage"] == "hard_refusal" and r.preset == "refuse", (cls, q)


@pytest.mark.parametrize("q", A3_REVIEW + SPILL_REVIEW[:12] + A1_REVIEW)
def test_review_changhua_questions_are_changhua_refusals(b, q):
    assert hard_refusal(q, b) == "changhua_lever", q


@pytest.mark.parametrize("q", A2_REVIEW)
def test_review_subsidence_questions_are_size_refusals(b, q):
    assert hard_refusal(q, b) == "subsidence_size", q


# review finding 5: over-refusals the fix makes answerable (must pass the guards)
MUST_PASS = [
    # named Yunlin rivers are not Changhua's north bank
    "Under a 30% Yunlin irrigation cut, what happens to heads along the north bank of the Beigang "
    "River by 2030?",
    "Under a 30% Yunlin irrigation cut, what happens to heads on the north bank of Beigang creek?",
    "Cut irrigation 30%: what happens on the north bank of the Huwei River?",
    "Cut irrigation 30%: what happens on the north bank of the Xinhuwei River by 2030?",
    # "the county" does not override the named township
    "How much less does Tuku sink if irrigation is cut 30%? Just the township, not the county.",
    "If aquaculture is retired, how much less does Huwei sink by 2030, the county's rail town?",
    "In the county's Tuku township, how much less is the sinking under a 30% irrigation cut?",
    # A1 near-misses
    "Changhua aside, what does a 30% Yunlin irrigation cut do to heads by 2030?",
    "Not counting Changhua, Yunlin cuts irrigation 30%: head in 2030?",
    "Setting Changhua aside, Yunlin halves aquaculture: head in 2031?",
    "Ignore Changhua: Yunlin cuts irrigation 30%, head in 2030?",
    "Yunlin cuts irrigation 30% and Changhua does nothing: head in 2030?",
    "Yunlin cuts irrigation 30%, Changhua stays the same: head in 2030?",
    "Yunlin cuts irrigation 30%, with no change in Changhua: head in 2030?",
    "只有雲林灌溉減三成，彰化沒有動作，2030年水位？",
    "雲林養殖減半，彰化不動，2030年水位？",
    # the A3 form that stays answerable
    "Has Changhua's pumping direction been tested at held-out wells?",
    "Is there any sign that pumping on the Changhua side moves water levels the right way, judged "
    "at wells the model was never tuned on?",
    "How sure are you that Changhua's water levels respond in the right direction?",
    "彰化抽水對水位的影響方向，有沒有在沒調參的井上驗證過？",
]


@pytest.mark.parametrize("q", MUST_PASS)
def test_review_over_refusals_now_pass(b, q):
    assert hard_refusal(q, b) is None, q


# over-refusals the review quoted that stay refused (safety first): listed, not tested as passes
STILL_REFUSED = [
    "What effect does halving Yunlin aquaculture have on Changhua?",   # spillover without row words
    "Is the subsidence the county's model shows tested at all?",       # how-sure with 'county's'
]


@pytest.mark.parametrize("q", STILL_REFUSED)
def test_review_over_refusals_left_refused(b, q):
    assert hard_refusal(q, b) is not None, q


# review finding 7 (A5 accuracy): a span's first year is a start when a lever acts on it
@pytest.mark.parametrize("q,reason", [
    ("Cut irrigation 30% from 2028 to 2032: head?", "out_of_grid"),
    ("As of 2025, Yunlin cuts irrigation 30%: head in 2030?", "out_of_grid"),
])
def test_review_a5_starts_off_the_page_are_clarified(b, q, reason):
    r = RulesRouter(b).route(q)
    assert (r.clarify or {}).get("reason") == reason, (q, r)
