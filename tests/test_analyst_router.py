"""Rules router (en/zh), hard refusals, follow-up patches; LLM router with a fake backend."""
from __future__ import annotations

import json
import re

import pytest

from hydrophysics_analyst import testing
from hydrophysics_analyst.llm.backend import FakeBackend, NormalizedResponse
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.router import route, routed_presets
from hydrophysics_analyst.router.llm import LLMRouter
from hydrophysics_analyst.router.rules import RulesRouter, hard_refusal


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("r")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


AQ30 = [{"class": "aquaculture", "pct": 30}]
CASES = [
    ("What does a 30% aquaculture cut from 2030 do by 2032?", "policy_head", {"cuts": AQ30, "start": 2030, "year": 2032}),
    ("養殖減30%，2030年開始，2032年水位？", "policy_head", {"cuts": AQ30, "start": 2030, "year": 2032}),
    ("Retire aquaculture from 2026", "policy_head", {"cuts": [{"class": "aquaculture", "pct": 100}], "start": 2026}),
    ("停止養殖抽水對2032年水位影響？", "policy_head", {"cuts": [{"class": "aquaculture", "pct": 100}], "year": 2032}),
    ("灌溉減三成", "policy_head", {"cuts": [{"class": "irrigation", "pct": 30}]}),
    ("Halve irrigation pumping", "policy_head", {"cuts": [{"class": "irrigation", "pct": 50}]}),
    ("Cut all pumping 20%", "policy_head", {}),
    ("Does it protect the HSR if we cut irrigation 30%?", "rail_internals", {"cuts": [{"class": "irrigation", "pct": 30}]}),
    ("灌溉減30%對高鐵有影響嗎？", "rail_internals", {}),
    ("Which class gives the biggest head rise per GWh?", "lever_rank", {"by": "gwh"}),
    ("Rank the water-use classes", "lever_rank", {"by": "head"}),
    ("哪個類別每度電效益最高？", "lever_rank", {"by": "gwh"}),
    ("Compare cut30 and retiring aquaculture", "solved_compare", {}),
    ("比較兩個完整模擬", "solved_compare", {}),
    ("Does waiting until 2030 matter for retiring aquaculture?", "start_compare", {"year": 2032}),
    ("What happens to Tuku if we retire aquaculture?", "township_internals", {"town": 1}),
    ("土庫鎮如果灌溉減半會怎樣？", "township_internals", {"town": 1, "cuts": [{"class": "irrigation", "pct": 50}]}),
    ("What happens to Changhua township if we retire aquaculture?", "township_internals", {"town": 0}),
]
REFUSE = [
    ("Cut Changhua aquaculture 30%", "changhua_lever"),
    ("Reduce Changhua aquaculture 30%", "changhua_lever"),
    ("彰化減少抽水", "changhua_lever"),
    ("彰化養殖減30%", "changhua_lever"),
    ("How many cm of subsidence do we avoid by cutting irrigation 30%?", "subsidence_size"),
    ("減少多少地層下陷？", "subsidence_size"),
    ("How big is the area sinking faster than 3 cm/yr?", "area"),
    ("What will it cost in NT$?", "volumes_money"),
    ("要花多少預算？", "volumes_money"),
    ("Which policy should we choose?", "recommendation"),
    ("你建議哪個方案？", "recommendation"),
    ("What about 2035?", "beyond_horizon"),
    ("What does benchmark X12 show?", "observations"),
]


@pytest.mark.parametrize("q,preset,sub", CASES)
def test_rules_cases(b, q, preset, sub):
    r = RulesRouter(b).route(q)
    assert r.preset == preset and r.confidence >= 0.9, (q, r)
    for k, v in sub.items():
        assert r.params.get(k) == v, (q, k, r.params)
    assert r.lang == ("zh" if re.search(r"[一-鿿]", q) else "en")


@pytest.mark.parametrize("q,reason", REFUSE)
def test_hard_refusals(b, q, reason):
    assert hard_refusal(q, b) == reason


def test_follow_up_patches(b):
    prev = {"preset": "policy_head", "params": {"cuts": AQ30, "start": 2026, "year": 2032}}
    R = RulesRouter(b)
    r = R.route("same for 2030", prev)
    assert r.preset == "policy_head" and r.params["start"] == 2030 and r.params["cuts"] == AQ30
    r = R.route("改成2030年開始", prev)
    assert r.params["start"] == 2030
    r = R.route("what about by 2028", prev)
    assert r.params["year"] == 2028 and r.params["start"] == 2026


def test_unclear_cuts_are_not_confident(b):
    assert RulesRouter(b).route("How much does cutting irrigation by a third help?").confidence < 0.9
    assert RulesRouter(b).route("Hello there").preset is None


def _llm(b, script):
    return LLMRouter(b, FakeBackend(script), presets=routed_presets())


def _out(**kw):
    d = {"preset": None, "cuts": None, "start": None, "year": None, "town": None, "by": None,
         "target_m": None, "refusal": None, "clarify": None, "inherit": [], "answer_lang": "en"}
    return d | kw


def test_llm_route_used_when_rules_unsure(b):
    llm = _llm(b, [_out(preset="policy_head", cuts=[{"class": "irrigation", "pct": 35}], start=2026, year=2031)])
    r, info = route(b, "How much does cutting irrigation by about a third help by 2031?", llm=llm)
    assert info["stage"] == "llm" and r.params["cuts"][0]["pct"] == 35 and r.source == "llm"


def test_hard_refusal_never_reaches_the_llm(b):
    llm = _llm(b, [])                                   # empty script: any call fails the test
    for q in ("reduce Changhua aquaculture 30 %", "彰化減少抽水"):
        r, info = route(b, q, llm=llm)
        assert r.refusal == "changhua_lever" and info["stage"] == "hard_refusal"


def test_provider_refusal_and_bad_json(b):
    r, _ = route(b, "something vague about irrigation", llm=_llm(b, [NormalizedResponse("", "refusal")]))
    assert r.refusal == "provider_refusal"
    r, _ = route(b, "something vague about irrigation", llm=_llm(b, ["not json"]))
    assert r.clarify["reason"] == "unknown"


def test_llm_inherit_copies_previous_params(b):
    prev = {"preset": "policy_head", "params": {"cuts": AQ30, "start": 2026, "year": 2032}}
    llm = _llm(b, [_out(preset="policy_head", start=2030, inherit=["cuts", "year"])])
    r, _ = route(b, "hmm and if that began at the later date instead", prev, llm=llm)
    assert r.params == {"start": 2030, "cuts": AQ30, "year": 2032}


def test_disabled_preset_from_llm_is_refused(b, monkeypatch):
    monkeypatch.delenv("ANALYST_ENABLE_CHEAPEST", raising=False)
    llm = _llm(b, [_out(preset="cheapest_for_head", target_m=0.5, start=2026, year=2032)])
    llm.presets = routed_presets()
    r, _ = route(b, "lowest energy way to get half a metre?", llm=llm)
    assert r.preset in ("refuse", "cheapest_for_head")
    if r.preset == "refuse":
        assert r.refusal == "disabled"


def test_system_prompt_is_stable_and_number_free(b):
    from hydrophysics.twin import claims
    llm = _llm(b, [])
    s1, _ = llm.request("q1", None)
    s2, _ = llm.request("q2", {"preset": "policy_head", "params": {}})
    assert s1 == s2
    text = s1[0]["text"]
    c = b.meta
    allowed = {str(y) for y in c["years"]} | {str(s) for s in c["starts"]}
    claim_digits = {m for cid in re.findall(r"\b[CEZ]\d+[a-e]?\b", text) for m in re.findall(r"\d+", cid)}
    for tok in re.findall(r"\d+(?:\.\d+)?", text):
        assert tok in allowed or tok in claim_digits, tok
    led = claims.load_ledger(allow_pending=True)
    for c_ in led["claims"]:
        for num in re.findall(r"\d+\.\d+", c_["claim"] + c_["wording_en"]):
            assert num not in text, (c_["id"], num)


def test_route_schema_is_enum_only(b):
    sch = _llm(b, []).schema
    for k, v in sch["properties"].items():
        inner = v.get("anyOf", [v])[0]
        assert inner.get("type") in ("string", "integer", "number", "array") or "enum" in inner
        if inner.get("type") == "string":
            assert "enum" in inner, k
    assert json.dumps(sch).count('"oneOf"') == 0


def test_llm_refusals_and_clarifications_are_not_overwritten(b):
    for out, want in ((_out(refusal="out_of_domain"), ("refuse", "out_of_domain")),
                      (_out(preset="township_internals", clarify="missing_cut"), (None, "missing_cut"))):
        r, _ = route(b, "some unclear words here", llm=_llm(b, [out]))
        if want[0] == "refuse":
            assert r.preset == "refuse" and r.refusal == want[1]
        else:
            assert r.clarify["reason"] == want[1] and r.refusal is None


def test_a_start_year_off_the_page_is_out_of_grid(b):
    r = RulesRouter(b).route("從2027年開始畜牧業減少一半，水位會怎樣？")
    assert r.clarify["reason"] == "out_of_grid" and r.confidence == 1.0


def test_eval_found_rules_bugs(b):
    R = RulesRouter(b)
    r = R.route("灌溉減30%跟養殖退場這兩個完整模型跑次，2028年比起來如何？")       # 比起來 is not 起
    assert r.preset == "solved_compare" and r.params["year"] == 2028
    assert hard_refusal("How many hectares of fish ponds would we have to retire?", b) == "volumes_money"
    prev = {"preset": "rail_internals", "params": {"cuts": [{"class": "irrigation", "pct": 20}],
                                                   "start": 2026, "year": 2032}}
    r = R.route("那如果再加上養殖減10%呢？", prev)
    assert r.preset == "rail_internals" and len(r.params["cuts"]) == 2


@pytest.mark.parametrize("q", ["What would a 30% irrigation cut do to heads in Huwei township by 2032?",
                               "土庫鎮在養殖全面停抽的情境下，2030年水位會怎麼變化？",
                               "虎尾鎮的地下水位未來會怎麼變？"])
def test_township_water_level_is_refused_per_d3(b, q):
    r = RulesRouter(b).route(q)
    assert r.refusal == "township_head" and r.confidence == 1.0


def test_township_subsidence_and_what_happens_stay_internals(b):
    for q in ("What happens to Tuku if we retire aquaculture?", "土庫鎮如果灌溉減半，地層下陷會少幾公分？"):
        assert RulesRouter(b).route(q).preset == "township_internals"


# --- ruling R9.2 (2026-11-07): the rules stage may only refuse or clarify ---------------------
@pytest.mark.parametrize("q,preset,sub", CASES)
def test_the_rules_stage_never_answers_with_a_preset(b, q, preset, sub):
    r, info = route(b, q, llm=None)                     # no LLM: the rules' parse is only a hint
    assert r.preset is None and r.refusal is None and r.clarify["reason"] == "no_llm", (q, r)
    assert info["stage"] == "rules" and info["rules_hint"]["preset"] == preset


@pytest.mark.parametrize("q,preset,sub", CASES)
def test_presets_come_only_from_the_llm_path(b, q, preset, sub):
    llm = testing.echo_llm(b)                           # mocked LLM (echoes the rules' reading)
    r, info = route(b, q, llm=llm)
    assert info["stage"] == "llm" and r.source == "llm" and r.preset == preset, (q, r)
    assert len(llm.backend.calls) == 1


# Must be refused (subsidence size, C03 withdrawn) but worded so the old rules read it as a
# confident policy_head: "settle" is not one of their subsidence words.
SETTLE = "How much less would the land settle if aquaculture were retired from 2026?"


def test_a_must_refuse_question_the_rules_used_to_route_never_reaches_a_preset_from_them(b):
    old = RulesRouter(b).route(SETTLE)
    assert old.preset == "policy_head" and old.confidence >= 0.9      # what the rules read
    r, info = route(b, SETTLE, llm=None)
    assert r.preset is None and r.clarify["reason"] == "no_llm" and info["stage"] == "rules"
    refusing = _llm(b, [_out(refusal="subsidence_size")])
    r, info = route(b, SETTLE, llm=refusing)
    assert info["stage"] == "llm" and r.refusal == "subsidence_size" and r.preset == "refuse"


def test_an_llm_error_falls_back_to_a_clarification_not_the_rules_preset(b):
    class Broken:
        def route(self, q, prev=None):
            raise RuntimeError("network down")
    r, info = route(b, "Retire aquaculture from 2026", llm=Broken())
    assert r.preset is None and r.clarify["reason"] == "no_llm" and "network down" in info["llm_error"]
    assert info["rules_hint"]["preset"] == "policy_head"


def test_rules_refusals_and_confident_clarifications_stand_without_the_llm(b):
    llm = _llm(b, [])                                   # empty script: any call fails the test
    r, info = route(b, "What would a 30% irrigation cut do to heads in Huwei township by 2032?", llm=llm)
    assert r.refusal == "township_head" and info["stage"] == "rules"
    r, info = route(b, "從2027年開始畜牧業減少一半，水位會怎樣？", llm=llm)
    assert r.clarify["reason"] == "out_of_grid" and info["stage"] == "rules"


def test_a_low_confidence_rules_clarification_goes_to_the_llm_and_stays_a_clarification(b):
    q = "What happens to the town if irrigation is cut 30%?"     # rules: ambiguous town, unsure
    assert RulesRouter(b).route(q).clarify["reason"] == "ambiguous_town"
    r, info = route(b, q, llm=None)
    assert r.clarify["reason"] == "ambiguous_town" and info["stage"] == "rules"
    r, info = route(b, q, llm=_llm(b, [_out(preset="township_internals",
                                            cuts=[{"class": "irrigation", "pct": 30}],
                                            start=2026, year=2032, town="Tuku")]))
    assert info["stage"] == "llm" and r.preset == "township_internals"


def test_no_llm_clarification_has_its_own_page_free_sentence(b):
    from hydrophysics_analyst.ask import answer_route
    for lang in ("en", "zh"):
        r, _ = route(b, "Retire aquaculture from 2026", llm=None)
        a = answer_route(b, r, lang)
        assert a.clarify["reason"] == "no_llm" and a.spans[0].key == "clarify_no_llm"
        assert a.citations == () and a.deep_link == ""
