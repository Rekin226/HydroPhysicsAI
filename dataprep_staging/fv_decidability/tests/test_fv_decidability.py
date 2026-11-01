"""CPU tests of FV.decidability.v1 (no GPU, no forward run, nothing registered on the real
registry, no look logged there, no observation opened). Synthetic inputs only; the real
registry and results/ tree are read (never written) by the declaration checks.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/fv_decidability/tests
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for _p in (str(X), str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"),
           str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import osse_fv as O  # noqa: E402
import prereg_fv as P  # noqa: E402


# ----------------------------------------------------------------------------- static
def test_no_home_paths_or_usernames_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json", ".md"):
            assert not pat.search(p.read_text(encoding="utf-8")), p


def test_future_annotations_everywhere():
    for p in (X / "osse_fv.py", X / "prereg_fv.py", Path(__file__)):
        assert "from __future__ import annotations" in p.read_text(encoding="utf-8"), p


def test_runner_carries_the_declared_recipe():
    assert P.runner_problems() == []
    assert set(O.FROZEN) >= {O.SELF, O.RUNNER}
    assert O.WIN == (178, 190) and O.T_RUN == 190
    ax = pd.date_range("2012-01-01", periods=O.T_RUN, freq="MS")
    assert f"{ax[O.WIN[0]]:%Y-%m}" == "2026-11" and f"{ax[O.WIN[1] - 1]:%Y-%m}" == "2027-10"
    assert f"{ax[O.T_FIT]:%Y-%m}" == "2023-01" == f"{O.ANOM_MONTHS[0]:%Y-%m}"


def test_readings_are_one_sided_with_inclusive_threshold():
    assert O.reading(0.5) == "weak_by_design"
    assert O.reading(0.0) == "weak_by_design"
    assert O.reading(0.5000001) == "not_shown_weak"
    assert O.reading(None) is None
    txt = json.dumps(O.WORDING, ensure_ascii=False).lower()
    assert "informative" not in txt and "calibrated" not in txt


# ----------------------------------------------------------------------------- arms
def test_arm_plan_and_files_are_forward_compatible(tmp_path):
    from hydrophysics.twin.recorded_pumping import load_recorded_multiplier, parse_recorded_spec

    plan = O.check_plan()
    assert [(t, d) for _, t, d in plan] == [("R0", 0), ("Q0", 0), ("R1c", 0), ("X5R1", 0),
                                            ("R0", 1)]
    assert len(O.arm_names("R0")) == 1 + 58 + 2
    assert all(len(O.arm_names(t)) == 1 + 58 + 1 for t in ("Q0", "R1c", "X5R1"))
    pool = 1.0 + 0.1 * np.random.default_rng(0).standard_normal((5, 12))
    rec = O.write_arms(tmp_path / "arms", pool)
    with pytest.raises(FileExistsError):
        O.write_arms(tmp_path / "arms", pool)
    specs = O.recorded_specs("R0", str(tmp_path / "arms"))
    assert len(specs) == 60
    proj = pd.date_range("2023-01-01", periods=O.HORIZON, freq="MS")
    for s in specs[:2] + specs[-2:]:
        name, path, start, county = parse_recorded_spec(s)
        assert county == "yunlin" and name in O.arm_names("R0")
        m = load_recorded_multiplier(path, start=start, county=county)
        m.check_window(proj)
        assert m.sha256 == rec["files"][name]["sha256"]
    imp = load_recorded_multiplier(str(tmp_path / "arms/imp_2024_07.csv"))
    assert list(imp.months.strftime("%Y-%m")) == ["2024-07"]
    assert float(imp.values["*"][0]) == 1.0 + O.DELTA
    chk = O.read_series_csv(tmp_path / "arms/chk_4.csv")
    a, _ = O.draw_multipliers(pool, O.rng_for(O.STREAM_ANOM, 0, 1))
    assert np.allclose(chk, a, rtol=1e-12, atol=0)


# ----------------------------------------------------------------------------- pool
def _pump_table(frozen_from=None, flagged_year=None, scale=1e6, flat_from=None, trend=0.0):
    rng = np.random.default_rng(3)
    rows = []
    prev = None
    for y in range(2010, 2023):
        vals = scale * (1 + 0.2 * rng.standard_normal(12)) + 5 * scale * (1 + trend * (y - 2012))
        if frozen_from is not None and y >= frozen_from and prev is not None:
            vals = prev.copy()
        if flat_from is not None:                       # (year, month): constant from then on
            fy, fmo = flat_from
            for m in range(12):
                if (y, m + 1) >= (fy, fmo):
                    vals[m] = 7.25 * scale
        for m in range(12):
            rows.append({"county": "yunlin", "scope": "in_grid", "class": "all",
                         "date": f"{y}-{m + 1:02d}", "value": vals[m],
                         "flag": 9 if (y == flagged_year and m == 3) else 0})
            rows.append({"county": "yunlin", "scope": "county", "class": "all",
                         "date": f"{y}-{m + 1:02d}", "value": -1.0, "flag": 0})
        prev = vals
    return pd.DataFrame(rows)


def test_anomaly_pool_full_and_ratio_definition():
    df, info = O.anomaly_pool(_pump_table())
    assert info["pool_years"] == list(range(2012, 2023)) and info["label"] == "full pool"
    r = df[[f"m{m:02d}" for m in range(1, 13)]].to_numpy()
    assert np.allclose(r.mean(axis=0), 1.0)          # ratio to the 2012-22 month mean


def test_anomaly_pool_freeze_rule_and_flags():
    df, info = O.anomaly_pool(_pump_table(frozen_from=2017, flagged_year=2013))
    assert info["first_frozen_year"] == 2017
    assert info["pool_years"] == [2012, 2014, 2015, 2016]
    assert info["label"].startswith("2012-16 blocks only")
    assert info["excluded_incomplete_or_flagged"] == [2013]
    assert O.pool_matrix(df).shape == (4, 12)
    assert np.allclose(O.pool_matrix(df).mean(axis=0), 1.0)     # normalised on the pool
    with pytest.raises(O.EngineeringFailure):
        O.anomaly_pool(_pump_table(frozen_from=2014))


def test_mid_year_freeze_marks_its_year():
    """A freeze from 2016-07 (constant months) is caught in 2016 by the run rule, not only in
    2017 by the year-on-year rule."""
    df, info = O.anomaly_pool(_pump_table(flat_from=(2016, 7)))
    assert info["first_frozen_year"] == 2016
    assert info["pool_years"] == [2012, 2013, 2014, 2015]
    assert info["label"].startswith("2012-16 blocks only")
    E = np.array([[1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                  [13.0, 14, 15, 16, 17, 18, 19, 20, 20, 21, 22, 23]])
    assert not O.frozen_months(E, 10.0).any()                   # a run of 2 is not frozen
    E[1, 9] = 20.0
    fm = O.frozen_months(E, 10.0)
    assert fm[1, 7:10].all() and fm.sum() == 3
    E2 = np.vstack([np.full(12, 5.0), 100.0 + np.arange(12.0)])
    E2[0, 10:] = np.nan
    assert O.frozen_months(E2, 5.0)[0, :10].all() and not O.frozen_months(E2, 5.0)[1].any()


def test_pool_normalisation_removes_a_level_shift():
    """A trending series drawn in random year order must not carry a level shift: the pool
    mean of the multiplier is 1 in every month, whatever the trend or truncation."""
    df, info = O.anomaly_pool(_pump_table(trend=0.05, frozen_from=2017))
    pool = O.pool_matrix(df)
    assert np.allclose(pool.mean(axis=0), 1.0)
    lv = info["report_only_pool_level_vs_2012_22"]
    assert len(lv) == 12 and all(x < 1.0 for x in lv)          # pool earlier, lower energy


def test_draw_multipliers_are_calendar_year_blocks():
    pool = np.arange(36, dtype=float).reshape(3, 12)
    a, pick = O.draw_multipliers(pool, O.rng_for(1, 2, 3))
    b, _ = O.draw_multipliers(pool, O.rng_for(1, 2, 3))
    assert a.shape == (58,) and np.array_equal(a, b)
    for y in range(5):
        seg = a[12 * y:12 * y + 12]
        assert np.array_equal(seg, pool[pick[y]][:len(seg)])


# ----------------------------------------------------------------------------- noise
def test_ar1_fit_recovers_parameters_and_respects_support():
    rng = np.random.default_rng(11)
    sd = np.array([0.5, 1.0, 2.0])
    phi = np.array([0.2, 0.6, 0.9])
    e = O.simulate_ar1(sd, phi, 20000, rng)
    s, p, n, npairs = O.ar1_fit(e + 3.0)
    assert np.allclose(s, sd, rtol=0.05) and np.allclose(p, phi, atol=0.03)
    short = np.full((2, 132), np.nan)
    short[0, :11] = rng.standard_normal(11)
    short[1, ::2] = rng.standard_normal(66)              # no consecutive pairs
    s, p, n, npairs = O.ar1_fit(short)
    assert np.isnan(s[0]) and np.isfinite(s[1]) and np.isnan(p).all()
    v, filled = O.fill_by_county(np.array([1.0, np.nan, 3.0, np.nan]),
                                 np.array(["A", "A", "B", "C"]))
    assert list(v) == [1.0, 1.0, 3.0, 2.0] and list(filled) == [False, True, False, True]


def test_departure_uses_zr0_and_the_observed_months():
    W = 3
    t = np.arange(O.T_RUN, dtype=float)
    h0 = np.zeros((W, O.T_RUN))
    hk = np.tile(0.01 * t, (W, 1))                     # a trending recipe difference
    mask = np.zeros((W, O.T_FIT), bool)
    mask[0] = True
    mask[1, :24] = True
    mask[2, -24:] = True
    D = O.departure(hk, h0, mask)
    win = 0.01 * t[O.WIN[0]:O.WIN[1]]
    assert np.allclose(D[0], win - 0.01 * np.arange(O.T_FIT).mean())
    assert np.allclose(D[1], win - 0.01 * np.arange(24).mean())
    assert np.allclose(D[2], win - 0.01 * np.arange(O.T_FIT - 24, O.T_FIT).mean())
    assert np.allclose(O.departure(hk + 5.0, h0 + 5.0, mask), D)   # code-shared offsets cancel
    with pytest.raises(O.EngineeringFailure):
        O.departure(hk, h0, np.zeros((W, O.T_FIT), bool))
    ids = ["09000001", "07000002"]
    obs = np.full((2, O.T_FIT), np.nan)
    obs[0, ::3] = 1.5
    obs[1, :30] = -2.0
    m = O.mask_table(ids, obs)
    assert list(m.columns[[1, -1]]) == ["2012-01", "2022-12"]
    assert set(np.unique(m.iloc[:, 1:].to_numpy())) == {0, 1}      # availability only


def test_null_tolerance_matches_simulation():
    rng = np.random.default_rng(5)
    W, M = 150, 12
    sd = rng.uniform(0.3, 1.0, W)
    phi = rng.uniform(0.2, 0.8, W)
    d = 0.03 * rng.standard_normal((W, M)) + 0.005
    t = O.null_tolerance(d, sd, phi, k=400)
    hits = []
    for k in range(3000):
        e = O.simulate_ar1(sd, phi, M, O.rng_for(9, 0, k))
        gap = np.mean(np.mean(d ** 2 - 2 * d * e, axis=1))
        hits.append(gap < 0)
    assert 0.05 < t["p_approx"] < 0.95
    assert abs(np.mean(hits) - t["p_approx"]) < 0.03
    assert t["tolerance"] >= t["p_approx"]
    far = O.null_tolerance(np.full((W, M), 2.0), sd, phi, k=500)
    assert far["p_approx"] < 1e-12 and abs(far["tolerance"] - 3 / 500) < 1e-9


# ----------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke") / "run"
    return out, O.run_smoke(out, k_truth=6, k_null=12)


def test_smoke_end_to_end(smoke):
    out, rep = smoke
    assert rep["status"] == "scored" and rep["smoke"] is True
    assert rep["superposition"]["pass"] and len(rep["superposition"]["checks"]) == O.N_CHECK
    assert rep["null_world"]["pass"]
    assert len(rep["engineering"]["evaluate_selfcheck"]) == O.N_SELFCHECK
    assert all(c["identical"] for c in rep["engineering"]["evaluate_selfcheck"])
    assert rep["engineering"]["zr0_meta"]["scenarios"] == ["baseline"]
    zd = rep["zr0_vs_registered_report_only"]
    assert zd["report_only"] and zd["hindcast_max_abs_m"] < 1e-5      # float32 storage only
    assert 0 < zd["window_datum_removed_max_abs_m"] < 0.1
    assert rep["engineering"]["repro"]["X5R1"]["pass"]
    assert set(rep["truth_worlds"]) == set(O.TAGS)
    pl = rep["P"]["level"]
    assert np.isclose(pl["p"], np.mean(list(pl["by_truth"].values())))
    assert rep["readings"]["level"] == O.reading(pl["p"])
    assert set(rep["readings"]["coverage"]) == {"Yunlin", "Changhua"}
    assert (out / "smoke_result.json").is_file()
    with pytest.raises(FileExistsError):
        O.run_smoke(out, k_truth=1, k_null=1)


def test_superposition_gate_fails_on_a_nonlinear_world(tmp_path):
    rep = O.run_smoke(tmp_path / "nl", k_truth=2, k_null=2, nonlinear=40.0)
    assert rep["status"] == "not_evaluable" and not rep["superposition"]["pass"]
    assert "truth_worlds" not in rep and "null_world" not in rep
    assert rep["readings"]["level"] is None


def test_tampered_arm_is_an_engineering_failure(tmp_path):
    Pth = O.build_synthetic(tmp_path / "syn")
    meta = json.loads((Path(Pth.arms_dir) / "arms.json").read_text())
    p = Path(Pth.arms_dir) / meta["files"]["chk_2"]["file"]
    p.write_text(p.read_text().replace("2027-10,", "2027-10,1"))     # one value changed
    with pytest.raises(O.EngineeringFailure, match="chk_2"):
        O.score(Pth, smoke=True, k_truth=1, k_null=1)


def test_declared_tolerance_is_enforced_before_gate_s(tmp_path):
    """A wrong declared tolerance is a gate-E failure even in a world that would fail S:
    E is complete before any S number (nothing is returned, nothing written)."""
    Pth = O.build_synthetic(tmp_path / "syn", nonlinear=40.0)
    logs = []
    with pytest.raises(O.EngineeringFailure, match="null tolerance"):
        O.score(Pth, smoke=True, k_truth=1, k_null=3,
                decl={"null_world": {"tolerance": 0.123}}, log=logs.append)
    assert not any("gate S" in x for x in logs)


def test_zr0_metadata_is_gate_e(tmp_path):
    Pth = O.build_synthetic(tmp_path / "syn")
    with np.load(Pth.zr0, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    d["member_labels"] = np.array(["stage3_datum_gate", "fold0"])
    d["n_members"] = 2
    np.savez_compressed(Pth.zr0, **d)
    with pytest.raises(O.EngineeringFailure, match="ZR0"):
        O.score(Pth, smoke=True, k_truth=1, k_null=1)


def test_check_attempts(tmp_path):
    f = tmp_path / "x.npz"
    f.write_bytes(b"abc")
    att = tmp_path / "attempts.jsonl"
    att.write_text(json.dumps({"event": "forward", "truth": "R0", "rc": 0,
                               "npz_sha256": O.sha256(f), "attempt": 1}) + "\n")
    assert O.check_attempts(att, {"R0": f})["R0"]["attempt"] == 1
    f.write_bytes(b"abd")
    with pytest.raises(O.EngineeringFailure):
        O.check_attempts(att, {"R0": f})


# ----------------------------------------------------------------------------- declaration
def test_declaration_validates_and_is_not_ready_without_export(tmp_path):
    from harness import prereg3

    d = P.doc(prereg_dir=tmp_path)
    assert prereg3.validate_prereg(json.loads(json.dumps(d, default=O._jsonable))) == []
    assert d["kind"] == "declaration" and d["frozen_code"].keys() == set(O.FROZEN)
    assert d["readings"]["threshold"] == 0.5
    assert "informative" in d["readings"]["rule"]       # only to say it does not exist
    assert P.TESTS not in d["inputs"]                   # a test edit cannot void it
    assert d["disclosures"]["tests"]["path"] == P.TESTS
    assert set(O.zr0_thetas()) | {O.zr0_column()} <= set(d["inputs"])
    assert d["zr0"]["forward_out"] == O.ZR0_NPZ and "upper bound" not in json.dumps(d)
    pend = P.readiness(d, prereg_dir=tmp_path)
    assert any("use classification" in x for x in pend)
    d2 = P.doc(prereg_dir=tmp_path, use_confirmation="lead, 2026-11-02, design_time_evidence")
    assert d2["replay"]["use_confirmation"].startswith("lead")
    assert not any("use classification" in x for x in P.readiness(d2, prereg_dir=tmp_path))
    if not (REPO / O.INPUTS_DIR / "summary.json").exists():
        assert any(x.startswith("export:") for x in pend)
        assert d["null_world"]["tolerance"] is None


def test_write_refuses_when_not_ready(tmp_path, capsys):
    sha = O.sha256(X / "prereg_fv.py")
    rc = P.main(["--write", "--quiet", "--expect-self-sha", sha], prereg_dir=tmp_path,
                rows_dir=tmp_path / "rows")
    assert rc == 1 and "REFUSED" in capsys.readouterr().out
    assert not (tmp_path / f"{O.ID}.json").exists()


def test_write_and_bind_in_a_tmp_registry(tmp_path, monkeypatch):
    """The document registers in a tmp registry, and the scorer's step 0 accepts it."""
    from harness import prereg3

    monkeypatch.setattr(P, "readiness", lambda d, prereg_dir=None, now=None: [])
    sha = O.sha256(X / "prereg_fv.py")
    rows = tmp_path / "rows"
    rows.mkdir()
    assert P.main(["--write", "--quiet", "--expect-self-sha", sha, "--use-confirmation",
                   "test"], prereg_dir=tmp_path, rows_dir=rows) == 0
    doc, dsha = O.load_registered(prereg_dir=tmp_path, rows_dir=rows)
    assert doc["id"] == O.ID and dsha == prereg3.read_chained(
        tmp_path / "INDEX.jsonl")[-1]["sha256"]
    with pytest.raises(prereg3.PreregistrationError, match="already exists"):
        P.main(["--write", "--quiet", "--expect-self-sha", sha], prereg_dir=tmp_path,
               rows_dir=rows)                         # never twice


def test_log_look_in_a_tmp_registry(tmp_path):
    inp = tmp_path / "inputs"
    inp.mkdir()
    (inp / "summary.json").write_text(json.dumps(
        {"files": {"noise_ar1.csv": "a" * 64, "anomaly_pool.csv": "b" * 64}}))
    wells = tmp_path / "wells.csv"
    pd.DataFrame({"station_id": ["09000001", "07000002"]}).to_csv(wells, index=False)
    rows = tmp_path / "rows"
    rows.mkdir()
    reg = tmp_path / "reg"
    reg.mkdir()
    rec = O.log_look(prereg_dir=reg, rows_dir=rows, inputs_dir=str(inp), wells_csv=str(wells))
    assert rec["units"] == [O.LOOK_UNIT] and rec["purpose"] == O.LOOK_PURPOSE
    assert P.design_look("a" * 64, prereg_dir=reg)["data_scope"] == O.LOOK_SCOPE
    assert P.design_look("c" * 64, prereg_dir=reg) is None
