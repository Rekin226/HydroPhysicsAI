"""The numpy port of the page's runtime math (browser parity: hydrophysics_analyst.parity)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hydrophysics_analyst import query as Q
from hydrophysics_analyst import testing
from hydrophysics_analyst.page import PageBundle


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("q")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


@pytest.fixture(scope="module")
def b_nobasis(tmp_path_factory):
    d = tmp_path_factory.mktemp("q0")
    page = testing.build_synthetic_page(d, basis=False)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


def test_policy_sources(b, b_nobasis):
    assert Q.policy_fields(b, Q.Policy((1.0,) * 6, 2026)).src == "base"
    solved = Q.policy_fields(b, Q.Policy.from_cuts({0: 30}, 2026))
    assert solved.src == "solved" and solved.hasW
    assert np.array_equal(solved.dW, np.asarray(b["n2Unit"]["solved"][solved.solved], np.float32))
    assert Q.policy_fields(b, Q.Policy.from_cuts({0: 50}, 2026)).src == "basis"
    assert Q.policy_fields(b_nobasis, Q.Policy.from_cuts({0: 50}, 2026)).src == "none"


def test_basis_superposition_matches_a_float32_hand_sum(b):
    p = Q.Policy.from_cuts({0: 50, 1: 20}, 2030)
    pol = Q.policy_fields(b, p)
    c = Q.ctx(b)
    n, si = c.A * c.Y, 1
    want = np.zeros(n, np.float32)
    for k in (0, 1):
        w = 1 - p.f[k]
        row = b.array("dSubs")[(si * c.K + k) * n:(si * c.K + k + 1) * n]
        want = (want.astype(np.float64) + w * row.astype(np.float64)).astype(np.float32)
    assert np.array_equal(pol.dS, want)
    dw = np.zeros(c.Y, np.float32)
    for k in (0, 1):
        dw = (dw.astype(np.float64) + (1 - p.f[k]) * np.asarray(b["n2Unit"]["basis"][si * c.K + k])).astype(np.float32)
    assert np.array_equal(pol.dW, dw)


def test_hash_rounded_factors():
    assert Q.Policy.from_cuts({0: 55}, 2026).f[0] == 0.45
    assert Q.Policy.from_cuts({0: 70}, 2026).f[0] == 0.3


def test_head_where_prefers_n2(b):
    r = Q.headline(b, Q.Policy.from_cuts({0: 30}, 2026), 2032)
    head = r.value("head")
    assert head.region == "n2_yunlin_wells" and head.text.startswith("+")
    assert "N2's 84 Yunlin" in r.snippets[0].text


def test_cost_is_the_weighted_class_energy(b):
    r = Q.headline(b, Q.Policy.from_cuts({0: 30, 1: 100}, 2026), 2032)
    want = 0.3 * testing.GWH[0] + 1.0 * testing.GWH[1]
    assert abs(r.value("gwh").value - want) < 1e-9


def test_headline_branches(b):
    c = Q.ctx(b)
    assert Q.headline(b, Q.Policy.from_cuts({0: 30}, 2026), c.years[c.YR]).snippets[0].key == "hl_fit"
    assert Q.headline(b, Q.Policy((1.0,) * 6, 2026), 2032).snippets[0].key == "hl_base"
    assert Q.headline(b, Q.Policy.from_cuts({0: 30}, 2030), 2028).snippets[0].key == "hl_early"
    assert Q.headline(b, Q.Policy.from_cuts({0: 30}, 2030), 2032).snippets[0].key == "hl_pol"


def test_rank_cm_is_withheld(b):
    r = Q.rank(b, 2032, "cm")
    assert not r.values and r.withheld[0].claim == "C03"


def test_deep_link_round_trip(b):
    p = Q.Policy.from_cuts({0: 55, 1: 100}, 2030)
    link = Q.deep_link(b, p, 2031, lang="zh")
    back = Q.hash_read(b, link)
    assert back == {"f": p.f, "start": 2030, "year": 2031, "lang": "zh"}


def test_start_note_only_inside_the_page_tolerance(b):
    r = Q.start_compare(b, Q.Policy.from_cuts({1: 100}, 2026).f, 2032)
    a0, a1 = r.value("head_2026_horizon").value, r.value("head_2030_horizon").value
    has = any(s.key == "start_note" for s in r.snippets)
    assert has == (abs(a0 - a1) <= max(.01, .03 * abs(a0)))


def test_no_internal_quantity_leaves_the_module(b):
    withheld = {"fieldAvoid", "avoid", "yunAvoid", "areaBase", "areaPol", "band", "share"}
    for fn in (lambda: Q.headline(b, Q.Policy.from_cuts({0: 30}, 2026), 2032),
               lambda: Q.township(b, Q.Policy.from_cuts({0: 30}, 2026), 2032, 1),
               lambda: Q.rail(b, Q.Policy.from_cuts({0: 30}, 2026), 2032),
               lambda: Q.rank(b, 2032, "head")):
        r = fn()
        keys = {v.key for v in r.values}
        assert not keys & withheld
        assert all(dataclasses.is_dataclass(v) for v in r.values)
