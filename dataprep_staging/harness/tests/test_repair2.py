"""CPU unit tests for the WP A1 second repair round: prereg3 (strict scopes, membership
blocking) and coverage3 (synthetic data only; pre-registrations are written under tmp_path)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from harness import coverage, coverage2, coverage3, prereg2, prereg3


# ----------------------------------------------------------------------------- helpers
def _reg(tmp_path):
    """Nested fresh units like validation_units/v2: big contains small; other is disjoint;
    block is a temporal unit; dt.t1 is a design-time unit overlapping small."""
    reg = tmp_path / "units.json"
    reg.write_text(json.dumps({
        "units": [
            {"id": "fresh.big", "role": "fresh", "kind": "cells", "cells": list(range(1, 11)),
             "wells": ["w1", "w2", "w3", "w4", "w5"]},
            {"id": "fresh.small", "role": "fresh", "kind": "cells", "cells": [1, 2, 3],
             "wells": ["w1", "w2"]},
            {"id": "fresh.other", "role": "fresh", "kind": "cells", "cells": [20, 21],
             "wells": ["w9"]},
            {"id": "fresh.folds", "role": "fresh", "kind": "folds",
             "folds": [{"fold": 0, "test": ["w2", "w9"]}, {"fold": 1, "test": ["w7"]}]},
            {"id": "fresh.block", "role": "fresh", "kind": "months", "start": "2016-01",
             "end": "2018-12"},
            {"id": "dt.t1", "role": "design_time", "kind": "cells", "cells": [1, 2, 3, 4, 5],
             "wells": ["w1", "w2", "w3"]},
            {"id": "dt.lev", "role": "design_time", "kind": "folds",
             "folds": [{"fold": 0, "test": ["bm1", "bm2"]}]},
        ],
        "withdrawn": {"fresh.old": {"replaced_by": "dt.t1", "reason": "looked at"}}}))
    return reg


def _doc(pid, tests, kind="test", fresh=None, external=None):
    na = prereg3.not_applicable
    fresh = fresh if fresh is not None else sorted({u for t in tests for u in t["fresh_units"]})
    d = {"id": pid, "wp": "TST", "title": "t", "kind": kind, "tests": tests,
         "zones": na("none"), "priors": na("none"), "burn_in": 12, "folds": na("none"),
         "fresh_units": list(fresh), "thresholds": {"bias": 0.3},
         "gate_scores": [{"name": "bias", "weight": 1.0}], "primary_model": "design 1",
         "inputs": na("none"), "interpretation": "x"}
    if external is not None:
        d["external_units"] = external
    return d


def _t(tid, scope="heads:2012-01..2022-12", units=("fresh.small",)):
    return [{"test_id": tid, "data_scope": scope, "fresh_units": list(units)}]


def _kw(tmp_path, reg):
    return {"prereg_dir": tmp_path / "prereg", "rows_dir": tmp_path / "rows",
            "registries": [reg]}


def _kw2(tmp_path):
    return {"prereg_dir": tmp_path / "prereg", "rows_dir": tmp_path / "rows"}


# ----------------------------------------------------------------------------- scope grammar
def test_scope_windows_strict_grammar():
    w = prereg3.scope_windows
    m = lambda y, mo: 12 * y + mo - 1  # noqa: E731
    assert w("heads:2016-01-01..2018-12-31") == [(m(2016, 1), m(2018, 12))]
    assert w("2012-01..2015-12,2019-01..2022-12") == [(m(2012, 1), m(2015, 12)),
                                                      (m(2019, 1), m(2022, 12))]
    assert w("heads:2012-01 .. 2015-12; 2019") == [(m(2012, 1), m(2015, 12)),
                                                   (m(2019, 1), m(2019, 12))]
    assert w("jiji_annual:2012..2025") == [(m(2012, 1), m(2025, 12))]
    assert w("insar:2020-01") == [(m(2020, 1), m(2020, 1))]
    assert w("numerical:sparse_vs_torch:cwa_theta_789d0e45:2012-01..2022-12") == [
        (m(2012, 1), m(2022, 12))]
    assert w("openmeteo_et0:2012-01..2021-12:st1,st42") == [(m(2012, 1), m(2021, 12))]
    assert w("2016,2017") == [(m(2016, 1), m(2017, 12))]          # adjacent years merge
    # unbounded: anything not fully understood
    for s in ("*", "", "heads", "simulation:coverage_interval", "heads:2016-2018",
              "heads:201601..201812", "gnss_wra+rings+piles:2020-01..2021-12:extra_2021",
              "heads:before:2016", "heads:2016 onwards", "heads:after 2019",
              "heads:2016-13", "heads:2016-02-30", "heads:2018..2016", "heads:<2016",
              "heads:2012..2015:2019..2022", "heads:preorigin:2016",
              "heads_2012_2022", "heads:2016-01-01T00:00..2018"):
        assert w(s) is None, s
    ov = prereg3.scopes_overlap
    assert ov("heads:2016-01-01..2018-12-31", "heads:2017-01..2018-12")
    assert ov("2012-01..2015-12,2019-01..2022-12", "heads:2020-01..2022-12")
    assert not ov("2012-01..2015-12,2019-01..2022-12", "heads:2016-01..2018-12")
    assert ov("heads:2016-2018", "heads:2030")                   # unbounded overlaps all


def test_prereg2_misread_scopes_the_verifier_found():
    # documents the holes this module repairs
    assert prereg2.scope_window("heads:2016-01-01..2018-12-31") == (2016 * 12, 2016 * 12)
    assert not prereg2.scopes_overlap("heads:2016-01-01..2018-12-31",
                                      "heads:2017-01..2018-12")
    assert not prereg2.scopes_overlap("2012-01..2015-12,2019-01..2022-12",
                                      "heads:2020-01..2022-12")


# ----------------------------------------------------------------------------- verifier probes
def test_probe_day_dates_and_multi_window_scopes_block(tmp_path):
    reg = _reg(tmp_path)
    kw = _kw(tmp_path, reg)
    prereg3.record_look(["fresh.small"], "heads:2016-01-01..2018-12-31", "TST", "diagnose",
                        **kw)
    with pytest.raises(prereg3.PreregistrationError, match="looks at the test data"):
        prereg3.write_prereg(_doc("A", _t("A.t", "heads:2017-01..2018-12")),
                             units_registry=reg, **kw)
    prereg3.record_look(["fresh.other"], "heads:2012-01..2015-12,2019-01..2022-12", "TST",
                        "diagnose", **kw)
    with pytest.raises(prereg3.PreregistrationError):
        prereg3.write_prereg(_doc("B", _t("B.t", "heads:2020-01..2022-12",
                                          units=("fresh.other",))), units_registry=reg, **kw)
    # a disjoint window is not blocked
    prereg3.write_prereg(_doc("C", _t("C.t", "heads:2016-01..2018-12",
                                      units=("fresh.other",))), units_registry=reg, **kw)
    # prereg2 would have let A through (the verifier's end-to-end probe)
    kw2 = _kw2(tmp_path)
    prereg2.write_prereg(_doc("A2", _t("A2.t", "heads:2017-01..2018-12")),
                         units_registry=reg, **kw2)
    with pytest.raises(prereg3.PreregistrationError, match="postdates looks"):
        prereg3.assert_can_score("A2", "TST", **kw)


def test_probe_nested_units_block_and_disclose(tmp_path):
    reg = _reg(tmp_path)
    kw = _kw(tmp_path, reg)
    prereg3.write_prereg(_doc("BIG", _t("BIG.t", units=("fresh.big",))), units_registry=reg,
                         **kw)
    prereg3.assert_can_score("BIG", "TST", **kw)
    # a later test on a unit nested in fresh.big is refused (prereg2 compared ids only)
    with pytest.raises(prereg3.PreregistrationError, match="fresh.big"):
        prereg3.write_prereg(_doc("SMALL", _t("SMALL.t")), units_registry=reg, **kw)
    kw2 = _kw2(tmp_path)
    prereg2.write_prereg(_doc("SMALL2", _t("SMALL2.t")), units_registry=reg, **kw2)
    with pytest.raises(prereg3.PreregistrationError, match="postdates looks"):
        prereg3.assert_can_score("SMALL2", "TST", **kw)
    # a fold unit sharing a well (w2) is refused too; a disjoint unit is not
    with pytest.raises(prereg3.PreregistrationError):
        prereg3.write_prereg(_doc("FOLDS", _t("FOLDS.t", units=("fresh.folds",))),
                             units_registry=reg, **kw)
    prereg3.write_prereg(_doc("OTHER", _t("OTHER.t", units=("fresh.other",))),
                         units_registry=reg, **kw)
    # the temporal unit is universal in space: the 2012-22 scoring of fresh.big touched
    # 2016-18 everywhere; a test outside 2016-18 on the same temporal unit cannot exist,
    # and a look outside 2016-18 does not block it
    with pytest.raises(prereg3.PreregistrationError):
        prereg3.write_prereg(_doc("BLK", _t("BLK.t", units=("fresh.block",))),
                             units_registry=reg, **kw)


def test_post_registration_looks_are_disclosed_by_membership(tmp_path):
    reg = _reg(tmp_path)
    kw = _kw(tmp_path, reg)
    prereg3.write_prereg(_doc("T1", _t("T1.t")), units_registry=reg, **kw)
    prereg3.write_prereg(_doc("T2", _t("T2.t", units=("fresh.big",))), units_registry=reg,
                         **kw)
    prereg3.assert_can_score("T2", "TST", **kw)
    ok = prereg3.assert_can_score("T1", "TST", **kw)        # registered before T2's look
    post = ok["post_registration_looks"]
    assert any(x["look_test_id"] == "T2.t" and ("fresh.big", "fresh.small") in
               [tuple(p) for p in x["pairs"]] for x in post)


def test_temporal_unit_window(tmp_path):
    reg = _reg(tmp_path)
    kw = _kw(tmp_path, reg)
    prereg3.record_look(["fresh.small"], "heads:2019-01..2022-12", "TST", "diagnose", **kw)
    prereg3.write_prereg(_doc("BLK1", _t("BLK1.t", units=("fresh.block",))),
                         units_registry=reg, **kw)
    # leveling benchmarks are not part of a temporal unit's head-data domain
    prereg3.record_look(["dt.lev"], "leveling:2012-01..2021-12", "TST", "diagnose", **kw)
    prereg3.write_prereg(_doc("BLK3", _t("BLK3.t", units=("fresh.block",))),
                         units_registry=reg, **kw)
    prereg3.record_look(["fresh.other"], "heads:2018-06..2019-03", "TST", "diagnose", **kw)
    with pytest.raises(prereg3.PreregistrationError):
        prereg3.write_prereg(_doc("BLK2", _t("BLK2.t", units=("fresh.block",))),
                             units_registry=reg, **kw)


def test_design_time_looks_block_only_after_the_declaration(tmp_path):
    reg = _reg(tmp_path)
    kw = _kw(tmp_path, reg)
    prereg3.record_look(["dt.t1"], "heads:2012-01..2022-12", "TST", "design_time_seed",
                        seeded_time="2026-09-01T00:00:00Z", **kw)
    prereg3.write_prereg(_doc("DECL", prereg3.not_applicable("a declaration"),
                              kind="declaration", fresh=["fresh.small", "fresh.other"]),
                         units_registry=reg, **kw)
    # history before the declaration does not block a test on fresh.small
    prereg3.write_prereg(_doc("S1", _t("S1.t")), units_registry=reg, **kw)
    # a look on the overlapping design-time unit after the declaration does
    prereg3.record_look(["dt.t1"], "heads:2012-01..2022-12", "TST", "diagnose", **kw)
    with pytest.raises(prereg3.PreregistrationError, match="dt.t1"):
        prereg3.write_prereg(_doc("S2", _t("S2.t", "heads:2015-01..2016-12")),
                             units_registry=reg, **kw)
    with pytest.raises(prereg3.PreregistrationError):
        prereg3.write_prereg(_doc("S3", _t("S3.t", "heads:2020-01..2020-12")),
                             units_registry=reg, **kw)
    # S1 was registered before that look: it may still be scored, and discloses the look
    ok = prereg3.assert_can_score("S1", "TST", **kw)
    assert any(("dt.t1", "fresh.small") in [tuple(p) for p in x["pairs"]]
               for x in ok["post_registration_looks"])
    # a withdrawn id resolves to its replacement
    assert prereg3.membership_index(kw["prereg_dir"], [reg]).canon("fresh.old") == "dt.t1"


def test_unknown_units_need_members(tmp_path):
    reg = _reg(tmp_path)
    kw = _kw(tmp_path, reg)
    with pytest.raises(prereg3.PreregistrationError, match="members"):
        prereg3.record_look(["sim.x"], "simulation:coverage", "TST", "diagnose", **kw)
    with pytest.raises(prereg3.PreregistrationError):
        prereg3.record_look(["sim.x"], "simulation", "TST", "diagnose",
                            members={"sim.x": {"none": ""}}, **kw)
    prereg3.record_look(["sim.x"], "simulation", "TST", "diagnose",
                        members={"sim.x": {"none": "simulation draws"}}, **kw)
    prereg3.record_look(["ext.wells"], "heads:2012-01..2022-12", "TST", "diagnose",
                        members={"ext.wells": {"ids": ["w2"], "description": "one well"}},
                        **kw)
    # declared members make the external look comparable: it overlaps fresh.small via w2,
    # but it is not a fresh unit and fresh.small was never declared: not eligible to block
    # a new registration ... until fresh.small is declared
    idx = prereg3.membership_index(kw["prereg_dir"], [reg])
    assert idx.overlap("ext.wells", "fresh.small") is True
    assert idx.overlap("sim.x", "fresh.small") is False
    assert idx.overlap("legacy.unit", "fresh.small") is None
    # an external unit without members is refused in a prereg3 registration
    t = [{"test_id": "E.t", "data_scope": "simulation", "fresh_units": ["sim.e"],
          "external_units": {"sim.e": "draws"}}]
    with pytest.raises(prereg3.PreregistrationError, match="description"):
        prereg3.write_prereg(_doc("E", t), units_registry=reg, **kw)
    t[0]["external_units"] = {"sim.e": {"description": "draws", "none": "simulation"}}
    prereg3.write_prereg(_doc("E", t), units_registry=reg, **kw)
    ok = prereg3.assert_can_score("E", "TST", **kw)
    assert ok["tests"] == ["E.t"]


def test_legacy_unknown_looks_are_disclosed_not_blocking(tmp_path):
    reg = _reg(tmp_path)
    kw, kw2 = _kw(tmp_path, reg), _kw2(tmp_path)
    prereg2.record_look(["legacy.unit"], "heads:2012-01..2022-12", "TST", "diagnose", **kw2)
    prereg3.write_prereg(_doc("L", _t("L.t")), units_registry=reg, **kw)
    ok = prereg3.assert_can_score("L", "TST", **kw)
    assert ok["unverifiable_prior_looks"] and \
        ok["unverifiable_prior_looks"][0]["units"] == ["legacy.unit"]
    a = prereg3.audit(kw["prereg_dir"], kw["rows_dir"], [reg])
    assert a["looks_on_units_of_unknown_membership"] == {"legacy.unit": 1}


# ----------------------------------------------------------------------------- coverage3
def test_coverage3_anova_equals_frozen_interval():
    rng = np.random.default_rng(3)
    m = np.full(14, 13.0)
    pi = rng.beta(0.9, 0.1, size=(500, 14))
    Y = rng.binomial(m.astype(int)[None, :].repeat(500, 0), pi).astype(float)
    a = coverage.interval_from_totals_batch(Y, m)
    b = coverage3.interval_batch(Y, m, "anova")
    assert np.array_equal(a["lo"], b["lo"]) and np.array_equal(a["hi"], b["hi"])
    u = coverage3.interval_batch(Y, m, "upper90")
    assert np.all(u["n_eff"] <= a["n_eff"] + 1e-9)          # never narrower than the frozen
    assert np.allclose(coverage3.icc_quantile_batch(Y, m, 0.10),
                       coverage2.icc_upper_batch(Y, m, 0.90))


def test_coverage3_all_hits_and_singletons():
    m = np.full(7, 12.0)
    allhit = np.full((1, 7), 12.0)
    a = coverage.interval_from_totals_batch(allhit, m)
    u = coverage3.interval_batch(allhit, m)
    assert a["lo"][0] > 0.9                         # frozen: n_eff = n, misses 0.90
    assert u["n_eff"][0] == pytest.approx(7.0) and u["lo"][0] < 0.9
    ones = np.ones(30)
    y = np.r_[np.ones(27), np.zeros(3)][None, :]
    r = coverage3.interval_batch(y, ones)
    assert r["n_eff"][0] == pytest.approx(30.0)     # singleton clusters: n_eff = n


def test_coverage3_gate_reports_calibration():
    hits = np.r_[np.ones(170), np.zeros(12)].astype(bool)
    rng = np.random.default_rng(0)
    rng.shuffle(hits)
    clusters = np.repeat(np.arange(14), 13)
    g = coverage3.coverage_gate(hits, clusters)
    assert g["method"] == coverage3.DEFAULT_METHOD == "upper90" and g["rule"] == "gate-3"
    assert g["k_clusters"] == 14 and g["calibrated"] is False and "calibration_note" in g
    clusters = np.repeat(np.arange(26), 7)
    g = coverage3.coverage_gate(hits, clusters)
    assert g["calibrated"] is True and "calibration_note" not in g
    g2 = coverage2.coverage_gate(hits, clusters)
    assert g["n_eff_lower"] == pytest.approx(g2["n_eff_lower"])
    assert g["window_applies"] == g2["window_applies"]
