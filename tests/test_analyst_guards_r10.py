"""Round-10 narrowing of the deterministic guards (ruling R10.2a), each exemption paired with
must-refuse "boundary twins" on the other side of it, in English, traditional and simplified
Chinese.

A1  a Yunlin question that names Changhua (or its geography) only to leave it out, or as the
    pooled row's spillover target
A2  how-sure questions about WHETHER the subsidence size was tested
A3  one how-sure question about Changhua's direction test
A4  one township referred to but not named, under a policy: clarified, not refused
A5  "as of <year>" and year spans are not read as a start by the rules stage

The exemption wordings were written by the round-10 builder, who used held-out v3/v4 (spent) as
development material; the twins were written from the rulings, not from any held-out set.
Synthetic page only (townships 土庫鎮 Tuku and 虎尾鎮 Huwei, last year 2032).
"""
from __future__ import annotations

import pytest

from hydrophysics_analyst import testing
from hydrophysics_analyst.llm.backend import FakeBackend
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.router import guards as G
from hydrophysics_analyst.router import route, routed_presets
from hydrophysics_analyst.router.llm import LLMRouter, system_prompt
from hydrophysics_analyst.router.rules import RulesRouter, hard_refusal, parse_years


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("g10")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


def _t2s() -> dict:
    from hydrophysics_analyst.router.zh import S2T
    return {ord(v): k for k, v in S2T.items() if v != "台"}


T2S = _t2s()


def sc(s: str) -> str:
    """The simplified-Chinese spelling of a traditional question."""
    out = s.translate(T2S)
    assert out != s, s            # the fold must have something to fold
    return out


# --- A1: Yunlin-only levers, Changhua left out or the spillover target --------------------------
A1_PASS = [
    "Cut Yunlin irrigation 30%, Changhua unchanged: what happens to the head by 2030?",
    "Yunlin, not Changhua, cuts aquaculture 40%: head in 2031?",
    "If Yunlin halves aquaculture and Changhua does not cut, what does the head do in 2030?",
    "Only Yunlin cuts irrigation 20%; nothing in Changhua changes. Head in 2030?",
    "Rank Yunlin's water-use classes by head rise; the county north of Yunlin isn't in the model.",
    "Unlike Changhua, Yunlin is modelled: what does a 25% irrigation cut do to Yunlin heads?",
    "Yunlin farms cut irrigation 15%, leaving the far bank of the Choushui out: head in 2029?",
    "What does a 30% Yunlin irrigation cut do to Changhua?",
    "How does retiring Yunlin aquaculture spill over into Changhua?",
    "雲林灌溉減三成，彰化維持現狀，2030年的水位？",
    "只算雲林：養殖減半，不包括彰化，水位會怎樣？",
    "雲林自己的灌溉減兩成，彰化完全不動，2031年水位如何？",
    "雲林灌溉減三成對彰化有什麼影響？",
    "Yunlin County cuts irrigation 20%, Changhua unchanged: head in 2030?",
]
A1_PASS += [sc("雲林灌溉減三成，彰化維持現狀，2030年的水位？"), sc("只算雲林：養殖減半，不包括彰化，水位會怎樣？")]

A1_TWINS = [   # must refuse changhua_lever
    # a Changhua-owned lever that also names Yunlin
    "Yunlin cuts irrigation 30%, and Changhua cuts aquaculture 20%: head?",
    "Cut irrigation 30% in Changhua, not in Yunlin: head in 2030?",
    "Yunlin keeps its pumping; Changhua halves irrigation: what does the head do?",
    "雲林灌溉減三成，彰化養殖減兩成，水位會怎樣？",
    # negation inversions
    "Not Yunlin but Changhua cuts irrigation 30%: what happens to the head?",
    "Unlike Yunlin, if Changhua cut irrigation 30%, what would the head do?",
    "Changhua, not Yunlin, cuts irrigation 30%: head in 2030?",
    "Yunlin doesn't cut; Changhua does: irrigation 30%. Head?",
    "Changhua is not excluded: cut Yunlin irrigation 30%, head in 2030?",
    "Changhua isn't unchanged: Yunlin cuts irrigation 30%, head?",
    "不是雲林，是彰化減灌溉三成，水位會怎樣？",
    "不像雲林，彰化如果減灌溉三成，水位會怎樣？",
    "雲林灌溉減三成，彰化不是不動，而是減養殖兩成，水位？",
    # Changhua comes back in a later clause, or is added
    "Yunlin cuts irrigation 30%, Changhua unchanged: what is the head in Changhua?",
    "Yunlin cuts irrigation 30% while Changhua stays as is: what does its head do?",
    "雲林灌溉減三成，彰化那邊不動，它的水位會怎樣？",
    "Yunlin cuts irrigation 30%; the county to the north does the same: head?",
    "Yunlin cuts irrigation 30%; Changhua isn't left out: it cuts too.",
    "除了彰化，雲林也減灌溉三成，水位會怎樣？",
    "Cut irrigation 30% in Yunlin and nothing on the north bank of the Choushui, plus aquaculture there: head?",
    # spillover target with a Changhua head, no Yunlin, or Changhua following
    "What does a 30% Yunlin irrigation cut do to Changhua's water level?",
    "雲林灌溉減三成對彰化的水位有什麼影響？",
    "What does a 30% irrigation cut do to Changhua?",
    "What does a Yunlin irrigation cut do to Changhua if it also cuts?",
    # a Changhua township is never an exclusion
    "Cut Yunlin irrigation 30%, Erlin unchanged: head in 2030?",
    # another place in a lever clause, or Yunlin itself left out (found by the builder's probes)
    "Yunlin cuts irrigation 30% but not Changhua; what if the north bank cut aquaculture 20%?",
    "Ignoring Yunlin, and with Changhua unchanged, cut irrigation 30%: head?",
    "Changhua unchanged, Yunlin unchanged, cut irrigation 30% across the fan: head?",
    "Only Yunlin, nothing on the north side of the river, irrigation cut 30%: how much less does the "
    "river's north side sink?",
    "雲林灌溉減三成，彰化不動，溪北的養殖減半呢？",
    # the spillover target never carries a subsidence size, a unit or a magnitude
    "雲林減灌溉三成，對彰化影響多少公分？",
    "What does a 30% Yunlin irrigation cut do to Changhua's subsidence in cm?",
    "How much does a 30% Yunlin irrigation cut affect Changhua?",
    # double negatives by a comparative, and other counties in a lever clause
    "Yunlin cuts irrigation 30%; Changhua isn't cutting less.",
    "Yunlin cuts irrigation 30%, Changhua doesn't cut less.",
    "Yunlin cuts irrigation 30%, Changhua unchanged; Chiayi cuts aquaculture 20%.",
    "雲林灌溉減三成，彰化維持現狀，嘉義減養殖。",
    "雲林灌溉減三成，彰化不減，但更多停抽。",
    "雲林減灌溉三成對彰化的地層下陷有什麼影響？",
]
A1_TWINS += [sc("不像雲林，彰化如果減灌溉三成，水位會怎樣？"), sc("雲林灌溉減三成，彰化那邊不動，它的水位會怎樣？"),
             sc("不是雲林，是彰化減灌溉三成，水位會怎樣？")]


@pytest.mark.parametrize("q", A1_PASS)
def test_a1_yunlin_only_questions_pass_the_changhua_guard(b, q):
    assert hard_refusal(q, b) is None, q


@pytest.mark.parametrize("q", A1_TWINS)
def test_a1_boundary_twins_are_changhua_refusals(b, q):
    assert hard_refusal(q, b) == "changhua_lever", q


@pytest.mark.parametrize("q", [
    "Yunlin cuts irrigation 30%, Changhua unchanged: how much less does Yunlin sink overall?",
    "雲林灌溉減三成，彰化維持現狀，全縣會少沉多少？",
])
def test_a1_does_not_open_a_subsidence_size(b, q):
    assert hard_refusal(q, b) == "subsidence_size", q


def test_a1_reflexive_ziji_after_yunlin_is_yunlins_own():
    assert not G.changhua_owns_a_lever("雲林縣自己的魚塭減抽四成")
    assert G.changhua_owns_a_lever("彰化自己的魚塭減抽四成")
    assert G.changhua_owns_a_lever("雲林不動，自己的魚塭減抽四成")


# --- A2: whether the subsidence size was tested ---------------------------------------------------
A2_PASS = [
    "Was the ground-sinking response ever checked against the leveling surveys?",
    "Did the avoided-subsidence result hold up against leveling?",
    "Has the subsidence reduction been confirmed against measurements?",
    "I wonder if the subsidence response was ever validated.",
    "減少下陷量有沒有經過檢驗？",
    "模型的減少沉陷量能不能當真？",
    "地層下陷的結果有沒有跟水準測量比對過？",
    "減少下陷量準不準？", "下陷減少量經得起水準測量檢驗嗎？",
]
A2_PASS += [sc("減少下陷量有沒有經過檢驗？"), sc("模型的減少沉陷量能不能當真？")]

A2_TWINS = [   # must refuse subsidence_size
    # a size asked inside a how-sure form
    "How sure are you of the 3 cm of avoided subsidence?",
    "How sure are you of the 3cm subsidence reduction?",
    "Is the avoided subsidence of 2 checked?",
    "Was the 0.9 subsidence reduction confirmed?",
    "Did the subsidence reduction hold up, and how big is it?",
    "Was the subsidence reduction checked? Show it.",
    "減少3公分的下陷能不能當真？",
    "減少下陷量有沒有經過檢驗？是多少？",
    "你對減少三公分的下陷有多少信心？",
    # a lever or an area scope inside the how-sure form
    "Is the subsidence reduction from a 30% irrigation cut tested?",
    "Has the county-wide subsidence reduction been checked against leveling?",
    "灌溉減三成的減少下陷量有沒有經過檢驗？",
    "全縣的減少下陷量能不能當真？",
    "Was the county's avoided subsidence checked against the leveling?",
    "Is the avoided subsidence positive? Was it checked?",
    "Is the avoided subsidence zero? Was it checked?",
    "Has it been confirmed that subsidence halves?",
    "下陷減少一半可信嗎？",
    "減少下陷量有兩成嗎？有沒有驗證？",
    "Has the avoided subsidence for the county been confirmed?",
]
A2_TWINS += [sc("減少下陷量有沒有經過檢驗？是多少？"), sc("全縣的減少下陷量能不能當真？")]


@pytest.mark.parametrize("q", A2_PASS)
def test_a2_whether_the_subsidence_size_was_tested_passes(b, q):
    assert hard_refusal(q, b) is None and G.how_sure_only(q), q


@pytest.mark.parametrize("q", A2_TWINS)
def test_a2_boundary_twins_are_subsidence_refusals(b, q):
    assert hard_refusal(q, b) == "subsidence_size", q


# --- A3: how sure is Changhua's direction test ----------------------------------------------------
A3_PASS = [
    "Has Changhua's pumping direction been tested at held-out wells?",
    "How sure are you that Changhua's water levels respond in the right direction?",
    "Is there evidence that pumping north of the Choushui moves heads the right way?",
    "When is the Changhua direction test scored?",
    "彰化抽水對水位的影響方向，有沒有在沒調參的井上驗證過？",
    "彰化的水位反應方向可信嗎？",
]
A3_PASS += [sc("彰化抽水對水位的影響方向，有沒有在沒調參的井上驗證過？")]

A3_TWINS = [   # must refuse changhua_lever: Changhua's direction asked as a value, or a lever
    "Which way do Changhua's water levels move? Is that tested?",
    "Has it been tested which way Changhua's head moves?",
    "Is Changhua's head rising, and is that verified?",
    "Do Changhua's heads go up or down, and how sure are you?",
    "How sure are you that Changhua's head rises 2 m by 2030?",
    "Is it verified that Changhua's head rises when Changhua cuts irrigation 30%?",
    "Is there evidence Changhua's head recovers if it pumps less?",
    "What is the direction of Changhua's head response, and is it tested?",
    "Is there evidence that Changhua's subsidence slows down the right way?",
    "Is it tested that the head responds the right way in both counties?",
    "Has Erlin's head response direction been tested?",
    "How sure are you that Changhua's heads move the right way by a meter?",
    "Is it verified that Changhua's heads rise by half when pumping is lower?",
    "彰化的水位會上升還是下降？有驗證嗎？",
    "彰化減抽後水位往哪個方向變，這有驗證嗎？",
    "彰化的水位會回升嗎？可信嗎？",
]
A3_TWINS += [sc("彰化的水位會上升還是下降？有驗證嗎？"), sc("彰化減抽後水位往哪個方向變，這有驗證嗎？")]


@pytest.mark.parametrize("q", A3_PASS)
def test_a3_how_sure_about_changhuas_direction_test_passes(b, q):
    assert hard_refusal(q, b) is None, q


@pytest.mark.parametrize("q", A3_TWINS)
def test_a3_boundary_twins_are_refused(b, q):
    assert hard_refusal(q, b) in ("changhua_lever", "subsidence_size"), q
    assert hard_refusal(q, b) is not None


# --- A4: one unnamed township's internals row is clarified ----------------------------------------
A4_CLARIFY = [
    "How much less would that coastal township sink if aquaculture is halved?",
    "If irrigation is cut 30%, how much less does the township I can't name sink in the model internals?",
    "灌溉減三成的話，那個鄉的地層下陷會少多少？",
    "養殖減半，靠山那個鎮在模型內部少沉多少？",
]
A4_CLARIFY += [sc("灌溉減三成的話，那個鄉的地層下陷會少多少？")]

A4_TWINS = [   # area-scope subsidence sizes: subsidence_size; township heads: township_head
    ("How much less would every township sink if aquaculture is halved?", "subsidence_size"),
    ("Which township sinks least after a 30% irrigation cut?", "subsidence_size"),
    ("How much less would the average township sink under a 30% irrigation cut?", "subsidence_size"),
    ("How much less would the townships on the coast sink if aquaculture is halved?", "subsidence_size"),
    ("How much less would that township and the whole county sink under a 30% cut?", "subsidence_size"),
    ("How much less would a township on the alluvial fan sink if irrigation is cut 30%?", "subsidence_size"),
    ("How much less would that township sink?", "subsidence_size"),           # no policy
    ("灌溉減三成，各鄉鎮的下陷會少多少？", "subsidence_size"),
    ("哪個鄉的下陷在灌溉減三成後減少最多？", "subsidence_size"),
    ("灌溉減三成，那個鄉和全縣會少沉多少？", "subsidence_size"),
    ("What does the water level in that coastal township do if aquaculture is halved?", "township_head"),
    ("那個鄉的地下水位在養殖減半後會怎樣？", "township_head"),
]
A4_TWINS += [(sc("灌溉減三成，各鄉鎮的下陷會少多少？"), "subsidence_size"),
             (sc("那個鄉的地下水位在養殖減半後會怎樣？"), "township_head")]


@pytest.mark.parametrize("q", A4_CLARIFY)
def test_a4_unnamed_township_internals_is_clarified_by_the_rules(b, q):
    assert hard_refusal(q, b) is None, q
    llm = LLMRouter(b, FakeBackend([]), presets=routed_presets())   # never called
    r, info = route(b, q, llm=llm)
    assert info["stage"] == "rules" and r.clarify["reason"] == "ambiguous_town" and r.preset is None, (q, r)


@pytest.mark.parametrize("q,reason", A4_TWINS)
def test_a4_boundary_twins_are_refused(b, q, reason):
    assert hard_refusal(q, b) == reason, q


def test_a4_a_named_township_still_routes_through_the_llm(b):
    # the confident clarification is for the unnamed township only
    r, info = route(b, "How much less would Tuku sink if aquaculture is halved?", llm=testing.echo_llm(b))
    assert info["stage"] == "llm"


# --- A5: "as of <year>" is not a start ------------------------------------------------------------
@pytest.mark.parametrize("q,want", [
    ("As of 2028, which water-use class gives the biggest head rise per GWh?", (None, 2028)),
    ("As of 2031, what does a 30% irrigation cut do to the head?", (None, 2031)),
    ("From 2023 to 2028, how does the head change under a 30% irrigation cut?", (None, 2028)),
    ("從2027年到2029年，灌溉減三成的水位？", (None, 2029)),
    ("Starting in 2030, cut irrigation 30%: head in 2032?", (2030, 2032)),
    ("Starting in 2028, cut irrigation 30%: head in 2032?", (-1, 2032)),     # a start off the page
    ("2028年開始灌溉減三成，2032年水位？", (-1, 2032)),
])
def test_a5_start_years(b, q, want):
    from hydrophysics_analyst.router.guards import norm
    assert parse_years(b, norm(q)) == want, q


@pytest.mark.parametrize("q", ["As of 2028, which water-use class gives the biggest head rise per GWh?",
                               "截至2028年，哪一類用水每GWh的水位效益最高？"])
def test_a5_as_of_is_not_an_out_of_grid_clarification(b, q):
    r = RulesRouter(b).route(q)
    assert (r.clarify or {}).get("reason") != "out_of_grid", (q, r)


@pytest.mark.parametrize("q", [
    "As of 2035, what does a 30% irrigation cut do to the head?",
    "As of 2040, which class gives the biggest head rise?",
    "As of twenty thirty-five, where is the head with aquaculture retired?",
    "截至2036年，灌溉減三成的水位？", "截至民國125年，灌溉減三成的水位？", "截至2036年，灌溉减三成的水位？",
    "From 2030 to 2035, how does the head change under a 30% irrigation cut?",
])
def test_a5_twins_beyond_the_horizon_stay_refused(b, q):
    assert hard_refusal(q, b) == "beyond_horizon", q


# --- every boundary twin is refused before the LLM, in both languages -----------------------------
ALL_TWINS = ([("changhua_lever", q) for q in A1_TWINS] + [("subsidence_size", q) for q in A2_TWINS]
             + [(None, q) for q in A3_TWINS] + [(r, q) for q, r in A4_TWINS])


@pytest.mark.parametrize("reason,q", ALL_TWINS)
def test_twins_never_reach_the_llm(b, reason, q):
    llm = LLMRouter(b, FakeBackend([]), presets=routed_presets())   # empty script: a call fails
    r, info = route(b, q, llm=llm)
    assert info["stage"] == "hard_refusal" and r.preset == "refuse"
    assert reason is None or r.refusal == reason


def test_twins_cover_both_languages_per_class():
    from hydrophysics_analyst.router.rules import detect_lang
    for twins in (A1_TWINS, A2_TWINS, A3_TWINS, [q for q, _ in A4_TWINS]):
        assert {detect_lang(q) for q in twins} == {"en", "zh"}


# --- the LLM prompt follows the narrowed guards (R10.2b) ------------------------------------------
def test_the_llm_prompt_describes_the_narrowed_changhua_rule(b):
    p = system_prompt(b, routed_presets())
    assert "any question about Changhua levers or cuts" not in p
    for phrase in ("leaves Changhua out", "spillover", "direction test", "whether the subsidence size",
                   "not named", "as of"):
        assert phrase in p, phrase


def test_the_fold_map_covers_the_round_ten_words():
    from hydrophysics_analyst.router.zh import to_trad
    pairs = {"当真": "當真", "维持原状": "維持原狀", "照旧": "照舊", "不考虑": "不考慮", "不纳入": "不納入",
             "跟着": "跟著", "连同": "連同", "不仅": "不僅", "影响": "影響", "迹象": "跡象", "没调": "沒調",
             "独立": "獨立", "双": "雙", "无": "無", "涨还是跌": "漲還是跌", "核对": "核對", "证实": "證實",
             "经得起": "經得起", "准不准": "準不準", "信得过": "信得過", "评估": "評估", "水准测量": "水準測量",
             "哪个方向": "哪個方向", "上升还是下降": "上升還是下降", "那个乡": "那個鄉", "这个镇": "這個鎮",
             "差异": "差異", "比较": "比較", "不动": "不動", "不变": "不變", "现状": "現狀", "反应": "反應"}
    for s_, t_ in pairs.items():
        assert to_trad(s_) == t_, s_
