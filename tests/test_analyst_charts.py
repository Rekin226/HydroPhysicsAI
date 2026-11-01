"""Example-question chips and in-chat charts: built only from the answer's own values."""
from __future__ import annotations

import dataclasses as dc

import pytest

from hydrophysics_analyst import presets as PR
from hydrophysics_analyst import testing
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.render import Chart, ChartSeries, resolve
from hydrophysics_analyst.validate import validate


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("ch")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_example_chip_answers_cleanly(b, lang):
    exs = PR.examples(b)
    assert len(exs) >= 8 and {e["preset"] for e in exs} >= set(PR.V1) - {"refuse"}
    for ex in exs:
        assert ex["label"][lang] and ex["group"][lang]
        a = PR.run(b, ex["preset"], ex["params"], lang)
        assert a.clarify is None and a.refusal is None, ex
        assert validate(a, b) == [], ex


def test_chart_points_are_answer_values(b):
    cases = (("policy_head", {"cuts": [{"class": "aquaculture", "pct": 30}], "start": 2030, "year": 2032}, 2),
             ("start_compare", {"cuts": [{"class": "aquaculture", "pct": 100}], "year": 2032}, 1),
             ("lever_rank", {"year": 2032, "by": "gwh"}, 1),
             ("solved_compare", {"year": 2032}, 2))
    for preset, params, n in cases:
        a = PR.run(b, preset, params, "en")
        assert len(a.charts) == n, preset
        for ch in a.charts:
            for sr in ch.series:
                for ref in sr.refs:
                    v = resolve(a, ref)
                    assert v.answer_class != "withheld"
            if any(resolve(a, r).answer_class == "internal" for sr in ch.series for r in sr.refs):
                assert ch.block == "internals"
    line = PR.run(b, "start_compare", cases[1][1], "en").charts[0]
    assert len(line.series) == 2 and {m[0] for m in line.marker} == {2026, 2030}


def test_no_chart_for_baseline_refusals_or_internals_only_presets(b):
    for preset, params in (("policy_head", {"cuts": [], "start": 2026, "year": 2032}),
                           ("refuse", {"reason": "subsidence_size"}),
                           ("township_internals", {"cuts": [{"class": "aquaculture", "pct": 100}],
                                                   "start": 2026, "year": 2032, "town": 1}),
                           ("how_sure", {})):
        assert PR.run(b, preset, params, "en").charts == ()


def test_validator_rejects_bad_charts(b):
    a = PR.run(b, "policy_head", {"cuts": [{"class": "aquaculture", "pct": 30}], "start": 2030, "year": 2032}, "en")
    rank = PR.run(b, "lever_rank", {"year": 2032, "by": "head"}, "en")

    def rules(x):
        return {v.rule for v in validate(x, b, reproduce=False)}
    foreign = Chart("line", a.charts[0].title, "tHead", "m", (ChartSeries("", ("r0:not_a_value",)),))
    assert "charts" in rules(dc.replace(a, charts=(foreign,)))
    bad_title = dc.replace(a.charts[0], title_key="made_up_title")
    assert "charts" in rules(dc.replace(a, charts=(bad_title,)))
    outside = dc.replace(rank.charts[0], block="answer")                  # internals drawn as an answer
    assert "charts" in rules(dc.replace(rank, charts=(outside,)))
    assert rules(a) == set() and rules(rank) == set()


def test_server_view_resolves_charts_and_meta_lists_examples(b, tmp_path):
    from hydrophysics_analyst.server import App
    app = App(b, use_llm=False, audit_dir=tmp_path)
    meta = app.meta()
    assert meta["examples"] and all("preset" in e and "params" in e for e in meta["examples"])
    v = app.answer({"preset": "start_compare", "params": {"cuts": [{"class": "aquaculture", "pct": 100}],
                                                          "year": 2032}, "lang": "en"}, "t")
    ch = v["charts"][0]
    assert ch["kind"] == "line" and len(ch["series"]) == 2 and ch["marker"]
    pts = ch["series"][0]["points"]
    assert pts[0]["x"] < pts[-1]["x"] and all(isinstance(p["y"], float) and p["text"] for p in pts)
