"""CPU unit tests for the WP A1 repair round: prereg2, coverage2, gaps_guarded (synthetic data
only; preregistrations are written under tmp_path)."""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest
from harness import coverage, coverage2, gaps, gaps_guarded, prereg, prereg2


# ----------------------------------------------------------------------------- helpers
def _reg(tmp_path):
    reg = tmp_path / "units.json"
    reg.write_text(json.dumps({
        "units": [{"id": "fresh.u1", "role": "fresh"}, {"id": "fresh.u2", "role": "fresh"},
                  {"id": "dt.u0", "role": "design_time"}],
        "withdrawn": {"fresh.old": {"replaced_by": "dt.old", "reason": "looked at"}}}))
    return reg


def _doc(pid, tests, kind="test", fresh=("fresh.u1",), inputs=None):
    na = prereg2.not_applicable
    return {"id": pid, "wp": "TST", "title": "t", "kind": kind, "tests": tests,
            "zones": na("none"), "priors": na("none"), "burn_in": 12, "folds": na("none"),
            "fresh_units": list(fresh), "thresholds": {"bias": 0.3},
            "gate_scores": [{"name": "bias", "weight": 1.0}], "primary_model": "design 1",
            "inputs": inputs if inputs is not None else na("none"), "interpretation": "x"}


def _t(tid, scope="heads:2012-01..2022-12", units=("fresh.u1",)):
    return [{"test_id": tid, "data_scope": scope, "fresh_units": list(units)}]


def _kw(tmp_path):
    return {"prereg_dir": tmp_path / "prereg", "rows_dir": tmp_path / "rows"}


# ----------------------------------------------------------------------------- scopes
def test_scope_windows_and_overlap():
    w = prereg2.scope_window
    assert w("heads:2012-01..2022-12") == (2012 * 12, 2022 * 12 + 11)
    assert w("jiji_annual:2012..2025") == (2012 * 12, 2025 * 12 + 11)
    assert w("insar:2020-01") == (2020 * 12, 2020 * 12)
    assert w("openmeteo_et0:st1:2022") == (2022 * 12, 2022 * 12 + 11)
    assert w("*") is None and w("openmeteo_et0:gridcells") is None
    assert w("x:cwa_theta_789d0e45") is None and w("cells_1030") is None
    ov = prereg2.scopes_overlap
    assert ov("heads:2012-01..2022-12", "heads:2016-01..2018-12")
    assert ov("leveling:2012-01..2022-12", "leveling_elevation:2012-01..2021-12")
    assert not ov("heads:2012-01..2022-12", "heads:2023-01..2026-09")
    assert ov("heads", "rain:2012-01..2012-12") and ov("*", "heads:2016-01..2018-12")


def test_parse_time_mixed_formats():
    a = prereg2.parse_time("2026-10-12T02:41:17+00:00")
    b = prereg2.parse_time("2026-10-12T02:41:17.000001Z")
    c = prereg2.parse_time("2026-10-12T02:41:17Z")
    assert a == c and b > a
    # string comparison would get this wrong ('Z' sorts after '.')
    assert "2026-10-12T02:41:17Z" > "2026-10-12T02:41:17.000001Z"


# ----------------------------------------------------------------------------- verifier probes
def test_probe_a_revised_design_after_a_scored_one_is_refused(tmp_path):
    reg, kw = _reg(tmp_path), _kw(tmp_path)
    prereg2.write_prereg(_doc("A", _t("A.t")), units_registry=reg, **kw)
    prereg2.assert_can_score("A", "TST", **kw)
    with pytest.raises(prereg2.PreregistrationError, match="looks at the test data"):
        prereg2.write_prereg(_doc("B", _t("B.t")), units_registry=reg, **kw)
    # the first tool allowed exactly this (verifier probe a)
    prereg.write_prereg(_doc("B1", _t("B1.t")), units_registry=reg, **kw)
    prereg.assert_can_score("B1", "TST", **kw)
    # and the repaired scorer refuses B1, whose registration postdates A's scoring look
    with pytest.raises(prereg2.PreregistrationError, match="postdates looks"):
        prereg2.assert_can_score("B1", "TST", **kw)
    # two designs registered before any look may both be scored (declared candidates)
    prereg2.write_prereg(_doc("C", _t("C.t", units=("fresh.u2",)), fresh=("fresh.u2",)),
                         units_registry=reg, **kw)
    prereg2.write_prereg(_doc("D", _t("D.t", units=("fresh.u2",)), fresh=("fresh.u2",)),
                         units_registry=reg, **kw)
    prereg2.assert_can_score("C", "TST", **kw)
    ok = prereg2.assert_can_score("D", "TST", **kw)
    assert any(x["look_test_id"] == "C.t" for x in ok["post_registration_looks"])
    # re-scoring the same registered test is allowed (looks after the registration)
    prereg2.assert_can_score("C", "TST", **kw)


def test_probe_b_overlapping_scopes_block_and_disjoint_do_not(tmp_path):
    reg, kw = _reg(tmp_path), _kw(tmp_path)
    prereg2.record_look(["fresh.u1"], "heads:2012-01..2022-12", "TST", "diagnose", **kw)
    with pytest.raises(prereg2.PreregistrationError):
        prereg2.write_prereg(_doc("E", _t("E.t", "heads:2016-01..2018-12")),
                             units_registry=reg, **kw)
    with pytest.raises(prereg2.PreregistrationError):
        prereg2.write_prereg(_doc("F", _t("F.t", "heads_daily:2020-01..2020-03")),
                             units_registry=reg, **kw)
    with pytest.raises(prereg2.PreregistrationError):
        prereg2.write_prereg(_doc("G", _t("G.t", "heads")), units_registry=reg, **kw)
    prereg2.write_prereg(_doc("H", _t("H.t", "heads:2023-01..2026-09")), units_registry=reg,
                         **kw)
    # the first tool let the 2016-18 test through (verifier probe b)
    assert prereg.blocking_looks(_t("E.t", "heads:2016-01..2018-12"), None,
                                 kw["prereg_dir"]) == []


def test_probe_c_looks_are_validated(tmp_path):
    reg, kw = _reg(tmp_path), _kw(tmp_path)
    with pytest.raises(prereg2.PreregistrationError):
        prereg2.record_look(["fresh.u1"], "heads", "TST", "diagnose", prereg_id="nope", **kw)
    prereg2.write_prereg(_doc("J", _t("J.t")), units_registry=reg, **kw)
    with pytest.raises(prereg2.PreregistrationError):
        prereg2.record_look(["fresh.u1"], "heads", "TST", "diagnose", prereg_id="J",
                            test_id="other", **kw)
    with pytest.raises(prereg2.PreregistrationError):
        prereg2.record_look(["fresh.u1"], "heads", "TST", "diagnose", test_id="J.t", **kw)
    with pytest.raises(prereg2.PreregistrationError):
        prereg2.record_look(["fresh.u1"], "heads", "TST", "score", prereg_id="J",
                            test_id="J.t", **kw)
    with pytest.raises(ValueError):
        prereg2.record_look(["fresh.u1"], "heads", "TST", "design_time_seed", prereg_id="J",
                            seeded_time="2026-01-01T00:00:00Z", **kw)
    with pytest.raises(ValueError):
        prereg2.record_look(["fresh.u1"], "heads", "TST", "diagnose",
                            seeded_time="2026-01-01T00:00:00Z", **kw)
    prereg2.record_look(["fresh.u1"], "heads", "TST", "diagnose", prereg_id="J",
                        test_id="J.t", **kw)
    # a look under an unregistered id (written by the first tool) blocks under the repair
    prereg.record_look(["fresh.u2"], "rain:2012-01..2012-12", "TST", "score",
                       prereg_id="not_in_index", test_id="x", prereg_dir=kw["prereg_dir"])
    with pytest.raises(prereg2.PreregistrationError):
        prereg2.write_prereg(_doc("K", _t("K.t", "rain:2012-06..2013-06",
                                          units=("fresh.u2",)), fresh=("fresh.u2",)),
                             units_registry=reg, **kw)
    a = prereg2.audit(kw["prereg_dir"], kw["rows_dir"], [reg])
    assert any(x["prereg_id"] == "not_in_index" for x in a["looks_with_unknown_prereg_id"])


def test_scoring_refuses_changed_inputs_wrong_wp_and_registry_change(tmp_path):
    reg, kw = _reg(tmp_path), _kw(tmp_path)
    src = tmp_path / "code.py"
    src.write_text("x = 1\n")
    prereg2.write_prereg(_doc("L", _t("L.t"), inputs=[str(src)]), units_registry=reg, **kw)
    with pytest.raises(prereg2.PreregistrationError, match="registered by WP"):
        prereg2.assert_can_score("L", "OTHER", **kw)
    src.write_text("x = 2\n")
    with pytest.raises(prereg2.PreregistrationError, match="inputs"):
        prereg2.assert_can_score("L", "TST", **kw)
    src.write_text("x = 1\n")
    reg.write_text(reg.read_text().replace("looked at", "changed"))
    with pytest.raises(prereg2.PreregistrationError, match="registry"):
        prereg2.assert_can_score("L", "TST", **kw)


def test_registry_roles_withdrawn_ids_and_declarations(tmp_path):
    reg, kw = _reg(tmp_path), _kw(tmp_path)
    bad = _doc("M", _t("M.t", units=("dt.u0",)), fresh=("dt.u0",))
    assert any("cannot be a fresh unit" in e for e in prereg2.validate_prereg(bad, reg))
    t = _t("N.t", units=("fresh.old",))
    t[0]["external_units"] = {"fresh.old": "smuggled"}
    assert any("withdrawn" in e for e in prereg2.validate_prereg(_doc("N", t), reg))
    na = prereg2.not_applicable
    dec = _doc("O", na("declaration"), kind="declaration", fresh=("fresh.u1", "fresh.u2"))
    dec["gate_scores"] = na("declaration")
    dec["primary_model"] = na("declaration")
    prereg2.record_look(["fresh.u2"], "*", "TST", "diagnose", **kw)
    with pytest.raises(prereg2.PreregistrationError, match="declared_fresh_unit_look"):
        prereg2.write_prereg(dec, units_registry=reg, **kw)
    dec2 = dict(dec, id="O2", fresh_units=["fresh.u1"])
    path, sha = prereg2.write_prereg(dec2, units_registry=reg, **kw)
    assert oct(os.stat(path).st_mode & 0o777) == "0o444"
    doc, sha2 = prereg2.load_prereg("O2", **kw)
    assert sha2 == sha and doc["tool"] == "harness.prereg2"
    dec3 = dict(dec, id="O3", fresh_units=["dt.u0"])
    assert any("role" in e for e in prereg2.validate_prereg(dec3, reg))


def test_both_tools_read_each_others_records(tmp_path):
    reg, kw = _reg(tmp_path), _kw(tmp_path)
    prereg2.write_prereg(_doc("P", _t("P.t")), units_registry=reg, **kw)
    d1, _ = prereg.load_prereg("P", **kw)
    prereg2.assert_can_score("P", "TST", **kw)
    assert prereg.read_chained(kw["prereg_dir"] / "looks.jsonl")[-1]["tool"] == \
        "harness.prereg2"
    assert d1["schema"] == "dataprep-prereg/2"


# ----------------------------------------------------------------------------- coverage2
def test_icc_upper_bound_properties():
    rng = np.random.default_rng(3)
    m = np.full(5, 6.0)
    Y = rng.binomial(6, 0.9, size=(2000, 5)).astype(float)
    rho = coverage.icc_anova_batch(Y, m)
    ru = coverage2.icc_upper_batch(Y, m)
    assert (ru >= rho - 1e-12).all() and (ru <= 1).all()
    assert coverage2.icc_upper_batch(np.array([[6, 6, 6, 6, 6.0]]), m)[0] == 1.0   # no variance
    assert coverage2.icc_upper_batch(np.array([[6, 0, 6, 6, 6.0]]), m)[0] == 1.0   # MSW = 0
    assert coverage2.icc_upper_batch(np.array([[5, 5, 5, 5, 5.0]]), m)[0] == 0.0   # MSB = 0
    assert coverage2.icc_upper_batch(np.array([[3.0]]), np.array([6.0]))[0] == 1.0
    assert coverage2.icc_upper_batch(np.ones((1, 30)), np.ones(30))[0] == 0.0


def test_gate2_window_only_where_n_eff_is_shown_large():
    # 5 clusters x 6 rows, rho_hat truncated to 0: gate-1 applied the window, gate-2 does not
    hits = np.r_[[1] * 5 + [0], [1] * 6, [1] * 5 + [0], [1] * 6, [1] * 4 + [0] * 2].astype(bool)
    cl = np.repeat(np.arange(5), 6)
    g1 = coverage.coverage_gate(hits, cl)
    g2 = coverage2.coverage_gate(hits, cl)
    assert g1["rho"] == 0.0 and g1["window_applies"]
    assert not g2["window_applies"] and g2["needs_pooling"] and g2["n_eff_lower"] < 30
    assert g2["lo"] == g1["lo"] and g2["hi"] == g1["hi"]          # same interval
    # 30 independent rows: the plan's window applies (n_eff = 30 exactly)
    h = np.r_[np.ones(27), np.zeros(3)].astype(bool)
    g = coverage2.coverage_gate(h, np.arange(30))
    assert g["window_applies"] and g["n_eff_lower"] == 30
    # many clusters: the window applies
    rng = np.random.default_rng(5)
    h = rng.random(74 * 42) < 0.9
    g = coverage2.coverage_gate(h, np.repeat(np.arange(74), 42))
    assert g["window_applies"] and g["n_eff_lower"] > 30 and g["rule"] == "gate-2"


def test_pool_small_strata_continues_past_a_stratum_without_neighbour():
    rng = np.random.default_rng(7)

    def st(n, off):
        return rng.random(n) < 0.9, np.arange(n) + off
    strata = {"iso": st(5, 0), "a": st(10, 100), "b": st(12, 200), "c": st(15, 300)}
    nb = {"a": ["b"], "b": ["a", "c"], "c": ["b"]}
    old = coverage.pool_small_strata(strata, nb)
    new = coverage2.pool_small_strata(strata, nb)
    assert "iso" in old and len(old) == 4          # the first harness stopped at 'iso'
    assert "iso" in new and new["iso"]["under_sized"]
    assert any(set(v["members"]) == {"a", "b", "c"} for v in new.values())


# ----------------------------------------------------------------------------- gaps_guarded
def test_guarded_generators_refuse_post_origin_and_match_frozen_masks():
    months = [str(p) for p in pd.period_range("2012-01", "2022-12", freq="M")]
    late = [str(p) for p in pd.period_range("2012-01", "2026-09", freq="M")]
    rng = np.random.default_rng(0)
    obs = rng.random((80, len(months))) > 0.05
    targets = list(range(80))
    a = list(gaps.head_masks("S", obs, targets))
    b = list(gaps_guarded.head_masks("S", obs, targets, months))
    assert all(np.array_equal(x[1], y[1]) and x[2] == y[2] for x, y in zip(a, b, strict=True))
    m1, c1 = gaps.replica_test_period(obs, targets, 0, window=(86, 131))
    m2, c2 = gaps_guarded.replica_test_period(obs, targets, 0, (86, 131), months)
    assert np.array_equal(m1, m2) and c1 == c2
    obs_late = rng.random((80, len(late))) > 0.05
    with pytest.raises(gaps.LeakageError):
        gaps_guarded.replica_test_period(obs_late, targets, 0, (132, 177), late)
    with pytest.raises(gaps.LeakageError):
        next(gaps_guarded.head_masks("N", obs_late, targets, late))
    gaps_guarded.replica_test_period(obs_late, targets, 0, (132, 177), late,
                                     allow_post_origin=True)
    with pytest.raises(ValueError):
        gaps_guarded.replica_test_period(obs, targets, 0, (86, 131), months[:-1])
    with pytest.raises(gaps.LeakageError):
        gaps_guarded.ring_windows("2015-01", "2023-06")
    assert len(gaps_guarded.ring_windows()) == 72
    with pytest.raises(gaps.LeakageError):
        gaps_guarded.ring_path_years(range(2018, 2023))
    assert len(gaps_guarded.ring_path_years()) == 5
    with pytest.raises(gaps.LeakageError):
        gaps_guarded.leveling_withheld((2022, 2023))
    days = pd.date_range("2012-01-01", "2023-02-28", freq="D")
    with pytest.raises(gaps.LeakageError):
        gaps_guarded.inject_gnss_steps(days, 0)
    s1, e1 = gaps.inject_gnss_steps(len(days[:4000]), 0)
    s2, e2 = gaps_guarded.inject_gnss_steps(days[:4000], 0)
    assert np.array_equal(s1, s2) and e1 == e2
    Y = rng.normal(size=(30, len(months)))
    y1, ev1 = gaps.inject_offsets_steps(Y, 0, n_paired=3, n_steps=3, margin=3,
                                        paired_len=(3, 12))
    y2, ev2 = gaps_guarded.inject_offsets_steps(Y, 0, months, n_paired=3, n_steps=3, margin=3,
                                                paired_len=(3, 12))
    assert np.array_equal(y1, y2) and ev1 == ev2
    assert set(gaps_guarded.GUARDS) == set(gaps.FAMILIES)
