"""CPU tests of the round 7 rank 8 census audit: synthetic products only (no census, head or
model file is read), nothing written to the real registry or results/.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider \\
        dataprep_staging/ch_audit/tests
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(REPO), str(X), str(REPO / "dataprep_staging")):
    if p not in sys.path:
        sys.path.insert(0, p)
import audit_census as A  # noqa: E402
import audit_lib as L  # noqa: E402


# ----------------------------------------------------------------------------- static
def test_no_home_paths_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(X.rglob("*")):
        if p.is_file() and p.suffix in (".py", ".sh", ".json"):
            assert not pat.search(p.read_text()), p


def test_future_annotations_everywhere():
    for p in sorted(X.rglob("*.py")):
        assert "from __future__ import annotations" in p.read_text(), p


def test_pinned_scorer_untouched():
    from twfe_mirror import BS_PATH, BS_SHA, sha256
    assert sha256(REPO / BS_PATH) == BS_SHA


# ----------------------------------------------------------------------------- pairs
def _months(T: int, first_month: int = 1) -> np.ndarray:
    return (np.arange(T) + first_month - 1) % 12 + 1


def test_pair_starts_follow_c0_phases():
    mo = _months(24)                       # Jan..Dec twice
    a = L.pair_starts("A", mo, 0, 24)
    b = L.pair_starts("B", mo, 0, 24)
    assert set(mo[a]) == {1, 3, 5, 7, 9, 11}
    assert set(mo[b]) == {2, 4, 6, 8, 10, 12}
    assert (a + 1 < 24).all() and (b + 1 < 24).all()
    assert 23 not in b                      # a Dec-Jan pair leaving the window is excluded


def test_reshape_keeps_pair_totals_and_recovers_reference_shape():
    T = 48
    mo = _months(T)
    true = 100 + 40 * np.cos(2 * np.pi * (mo - 7) / 12)
    X = np.vstack([true, true, true])
    for i, ph in ((0, "A"), (1, "B")):
        st = L.pair_starts(ph, mo, 0, T)
        tot = X[i, st] + X[i, st + 1]
        X[i, st] = X[i, st + 1] = tot / 2
    phase = np.array(["A", "B", "monthly"])
    R, cnt = L.reshape_pairs(X, phase, np.zeros(3, int), true[None, :], mo, 0, T)
    np.testing.assert_allclose(R[0], true, rtol=1e-12)
    np.testing.assert_allclose(R[1, 1:-1], true[1:-1], rtol=1e-12)
    np.testing.assert_allclose(R[2], true)                # monthly meters untouched
    assert cnt["pairs_resplit"] == cnt["pairs_even_split"] > 0
    for i, ph in ((0, "A"), (1, "B")):
        st = L.pair_starts(ph, mo, 0, T)
        np.testing.assert_allclose(R[i, st] + R[i, st + 1], X[i, st] + X[i, st + 1])


def test_uneven_and_missing_pairs_are_left_alone():
    mo = _months(12)
    X = np.array([[10.0, 10.0, 7.0, 3.0, np.nan, 5.0, 4, 4, 4, 4, 4, 4]])
    ref = np.arange(1, 13, dtype=float)[None, :]
    R, cnt = L.reshape_pairs(X, np.array(["A"]), np.zeros(1, int), ref, mo, 0, 12)
    assert R[0, 2] == 7.0 and R[0, 3] == 3.0             # an uneven (read) split
    assert np.isnan(R[0, 4]) and R[0, 5] == 5.0          # a missing half
    assert R[0, 0] == pytest.approx(20 * 1 / 3)
    assert cnt["pairs_even_split"] == 4


def test_even_split_attenuation_matches_theory():
    T = 132
    mo = _months(T)
    yr = 2012 + np.arange(T) // 12
    true = 1 + 0.3 * np.cos(2 * np.pi * (mo - 0.5) / 12)
    ev = []
    for ph in ("A", "B"):
        x = true.copy()
        st = L.pair_starts(ph, mo, 0, T)
        x[st] = x[st + 1] = (true[st] + true[st + 1]) / 2
        ev.append(x)
    amp_true = L.harmonic(true, mo, yr)["amp_rel"]
    amp_ev = L.harmonic(ev[0] + ev[1], mo, yr)["amp_rel"]
    assert amp_ev / amp_true == pytest.approx(L.expected_amp_ratio(1.0), abs=2e-3)
    assert L.expected_amp_ratio(1.0) == pytest.approx(math.cos(math.pi / 12) ** 2)
    # one phase alone carries a period-2 artefact; the A + B mix cancels it
    assert L.sawtooth(ev[0], mo, yr)["saw_to_annual"] > 0.05
    assert L.sawtooth(ev[0] + ev[1], mo, yr)["saw_to_annual"] < 1e-6


def test_smooth2_removes_sawtooth_and_has_zero_phase():
    T = 132
    mo = _months(T)
    yr = 2012 + np.arange(T) // 12
    s = 1 + 0.3 * np.cos(2 * np.pi * (mo - 0.5 - 6) / 12) + 0.05 * (-1.0) ** np.arange(T)
    sm = L.smooth2(s)
    assert L.sawtooth(sm, mo, yr)["saw_amp_rel"] < 2e-3
    h0, h1 = L.harmonic(s, mo, yr), L.harmonic(sm, mo, yr)
    assert abs(L.circ_days(h1["peak_day"], h0["peak_day"])) < 1.0
    assert h1["amp_rel"] / h0["amp_rel"] == pytest.approx((1 + math.cos(math.pi / 6)) / 2,
                                                          abs=1e-2)


# ----------------------------------------------------------------------------- retention
def _pt():
    pumps = pd.DataFrame({
        "sid": ["a", "b", "c", "d", "e"], "PUMP_HP": [10, 10, 10, 1, 0],
        "電號1": ["m1", "m1", "m2", "m3", "m4"], "county": ["changhua"] * 5,
        "cls": ["irrigation", "irrigation", "irrigation", "industry", "other"],
        "cell": [1, -1, 2, 3, 4]})
    dates = pd.date_range("2012-01-01", "2012-12-01", freq="MS")
    rows = []
    for sid, kwh in (("a", 500.0), ("b", 500.0), ("c", 100.0), ("d", 5000.0), ("e", 50.0)):
        rows.append(pd.DataFrame({"pump": sid, "datetime": dates, "electricity_kwh": kwh}))
    return pumps, pd.concat(rows, ignore_index=True)


def test_meter_table_matches_clean_census():
    from hydrophysics.twin.pumping import clean_census
    pumps, kwh = _pt()
    pumps["PURPOSE"] = "灌溉用水"
    pt = L.meter_table(pumps, kwh, "2012-01-01", "2013-01-01")
    kp, kk, rep = clean_census(pumps.drop(columns=["county", "cls", "cell"]), kwh,
                               cap_duty=1.0, t0="2012-01-01", t1="2013-01-01")
    assert set(kp["sid"]) == set(pt.loc[~pt["dropped"], "sid"])
    assert pt["e_dedup"].sum() / 1e6 == pytest.approx(rep["kwh_dedup_GWh"].sum())
    assert pt.loc[~pt["dropped"], "e_dedup"].sum() == pytest.approx(kk["electricity_kwh"].sum())
    # m1 shared by a and b counted once; d is over its rating; e has no HP and is kept
    assert pt["e_raw"].sum() == pytest.approx(12 * (500 + 500 + 100 + 5000 + 50))
    assert pt.set_index("sid").loc["d", "dropped"] and not pt.set_index("sid").loc["e",
                                                                                   "dropped"]


def test_retention_scopes_buckets_and_memory_figure():
    pumps, kwh = _pt()
    pt = L.meter_table(pumps, kwh, "2012-01-01", "2013-01-01")
    ret = L.retention(pt, 1.0)
    ch = ret["changhua"]
    dedup = 12 * (500 + 100 + 5000 + 50) / 1e6
    kept = 12 * (500 + 100 + 50) / 1e6
    assert ch["all"]["all"]["dedup_GWh_yr"] == pytest.approx(dedup)
    assert ch["all"]["all"]["kept_GWh_yr"] == pytest.approx(kept)
    assert ch["in_grid"]["all"]["kept_GWh_yr"] == pytest.approx(12 * (250 + 100 + 50) / 1e6)
    duty_d = 5000 / (1 * L.KW_PER_HP * L.HOURS_PER_MONTH)
    lab = [k for k in ch["all"]["all"]["dropped_by_duty"]
           if k.startswith("(3") or k.startswith("(10")]
    got = sum(ch["all"]["all"]["dropped_by_duty"][k]["GWh_yr"] for k in lab)
    assert duty_d > 3 and got == pytest.approx(12 * 5000 / 1e6)
    assert ch["all"]["all"]["kept_GWh_yr_at_cap"]["none"] == pytest.approx(dedup)
    mf = L.memory_figure(ret)
    assert mf["candidates"]["kept/dedup, all pumps, all classes"] == pytest.approx(kept / dedup)
    assert ("withdrawn" in mf["reading"]) == (not mf["reproduced_by"])


def test_decide_rules():
    def met(amp, peak, saw, rng):
        return {"amp_rel": amp, "peak_day": peak, "saw_to_annual": saw, "clim_range": rng}
    ret = {"changhua": {"in_grid": {"irrigation": {
        "kept_GWh_yr": 100.0,
        "dropped_by_duty": {"(1, 1.5]": {"GWh_yr": 6.0}}}}}}
    seas = {"changhua": {c: {"delivered": met(0.31, 200, 0.01, 0.6),
                             "resplit": met(0.33, 203, 0.0, 0.65)} for c in ("all",
                                                                           "irrigation")}}
    d = L.decide(seas, ret, {"changhua": 0.002})
    assert d["D1_changhua_forcing_version"]["decision"] == "model_census_v1_as_is"
    assert "known bias" in d["D1_changhua_forcing_version"]["nature"]
    assert "expected_amp_ratio_theory" in d["D1_changhua_forcing_version"]["rows"]["all"]
    assert d["D1b_duty_cap"]["decision"] == "cap_member_required"
    assert d["D2_seasonal_amplitude_gates"]["decision"] == "unsmoothed"
    assert d["annual_sum_max_rel_change"]["within"]
    seas["changhua"]["all"]["delivered"] = met(0.25, 230, 0.2, 0.5)
    d = L.decide(seas, ret, {"changhua": 0.02})
    assert d["D1_changhua_forcing_version"]["decision"] == "resplit_product_required"
    assert d["D2_seasonal_amplitude_gates"]["decision"] == "smooth2"
    assert not d["annual_sum_max_rel_change"]["within"]


# ----------------------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    d = tmp_path_factory.mktemp("audit")
    mc, po = A.build_synthetic(d / "syn")
    return d, A.run(d / "out", mc, po, A.synthetic_grid())


def test_smoke_end_to_end(smoke):
    d, rep = smoke
    on_disk = json.loads((d / "out" / "audit.json").read_text())
    assert on_disk["decisions"]["D1_changhua_forcing_version"]["decision"] == \
        rep["decisions"]["D1_changhua_forcing_version"]["decision"]
    assert rep["model_check"]["passed"]
    es = rep["even_split"]["changhua"]
    assert es["energy_even_split_share"] == pytest.approx(1.0)
    assert es["pairs_resplit"] == es["pairs_even_split"] > 0
    # the synthetic truth is a pure annual cosine: the re-split recovers its amplitude
    s = rep["seasonal"]["changhua"]["irrigation"]
    assert s["resplit"]["amp_rel"] == pytest.approx(s["monthly_meters_only"]["amp_rel"],
                                                    rel=0.02)
    assert s["delivered"]["amp_rel"] / s["resplit"]["amp_rel"] == pytest.approx(
        es["expected_amp_ratio_from_even_split"], abs=0.01)
    assert rep["decisions"]["annual_sum_max_rel_change"]["within"]
    # unknown billing codes are resolved by the detected phase (C0's convention)
    assert es["energy_share_phase_from_detected"] > 0
    assert es["energy_share_phase_unresolved"] == 0.0
    row = rep["decisions"]["D1_changhua_forcing_version"]["rows"]["irrigation"]
    assert row["expected_amp_ratio_theory"] == pytest.approx(L.expected_amp_ratio(
        s["bimonthly_delivered"]["mean_kwh_per_month"] / s["delivered"]["mean_kwh_per_month"]),
        abs=1e-6)
    fp = rep["decisions"]["forcing_parity_changhua"]
    assert fp["within"] and abs(fp["rel_diff"]) < 1e-9


def test_resolve_phase_follows_c0():
    out = L.resolve_phase(["A", "unknown", "unknown", "monthly", "unknown"],
                          ["B", "B", "monthly", "A", "both-flat"])
    assert out.tolist() == ["A", "B", "monthly", "monthly", "unknown"]


def test_forcing_parity():
    ret = {"changhua": {"in_grid": {"all": {"kept_GWh_yr": 100.0}}}}
    assert L.forcing_parity(ret, {"changhua": {"in_grid_GWh_yr_delivered": 101.5}})["within"]
    assert not L.forcing_parity(ret, {"changhua": {"in_grid_GWh_yr_delivered": 97.0}})[
        "within"]


def test_run_is_written_once_and_never_into_the_registry(smoke, tmp_path):
    d, _ = smoke
    with pytest.raises(SystemExit, match="written once"):
        A.run(d / "out", d / "syn/model_census", d / "syn/preorigin", A.synthetic_grid())
    with pytest.raises(SystemExit, match="preregistration"):
        A.run(tmp_path / "preregistration" / "x", d / "syn/model_census",
              d / "syn/preorigin", A.synthetic_grid())


def test_manifest_use_is_checked(tmp_path):
    mc, po = A.build_synthetic(tmp_path / "syn")
    (po / "MANIFEST.json").write_text(json.dumps({"allowed_uses": ["validator"]}))
    with pytest.raises(SystemExit, match="model_design"):
        A.run(tmp_path / "out", mc, po, A.synthetic_grid())
    (po / "MANIFEST.json").unlink()                       # fails closed
    with pytest.raises(SystemExit, match="missing"):
        A.run(tmp_path / "out", mc, po, A.synthetic_grid())
    assert not (tmp_path / "out" / "audit.json").exists()


def test_look_is_logged_before_any_product_is_read(tmp_path, monkeypatch):
    mc, po = A.build_synthetic(tmp_path / "syn")
    reg = tmp_path / "reg"

    def boom(*a, **k):
        assert (reg / "looks.jsonl").exists()         # the look precedes the first read
        raise SystemExit("crash")
    monkeypatch.setattr(A, "retention_part", boom)
    with pytest.raises(SystemExit, match="crash"):
        A.run(tmp_path / "out", mc, po, A.synthetic_grid(), log_look=True, prereg_dir=reg)
    rec = json.loads((reg / "looks.jsonl").read_text().splitlines()[0])
    assert rec["purpose"] == "model_design" and rec["units"] == [A.LOOK_UNIT]
    assert not (tmp_path / "out" / "audit.json").exists()


def test_look_spec_records_in_a_tmp_registry(tmp_path):
    from harness import prereg3
    kw = A.look_spec()
    rec = prereg3.record_look(kw.pop("units"), kw.pop("data_scope"), kw.pop("wp"),
                              kw.pop("purpose"), prereg_dir=tmp_path, rows_dir=tmp_path / "r",
                              **kw)
    assert rec["purpose"] == "model_design" and rec["units"] == [A.LOOK_UNIT]
    assert rec["scope_windows"]
    lines = (tmp_path / "looks.jsonl").read_text().splitlines()
    assert len(lines) == 1
