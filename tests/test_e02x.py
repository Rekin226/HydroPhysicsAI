"""CPU tests of round-8 rank 5 (dataprep_staging/e02x): the departure-transfer library, the
U1.ext_transfer_power.v1 world and decision rule, the member extraction gate, the sd_fit_rel
operator, the artefact, the GOV precondition entry, the declaration builders and the scorer.

Synthetic only: every array, run file, product MANIFEST and observation is generated here in
pytest's tmp dir. No head product is opened, nothing under results/ is written, and
``prereg3.write_prereg`` is replaced by a function that fails the test if it is ever reached.
Tests that need the gitignored registry (results/dataprep/preregistration) skip without it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
STAGING = REPO / "dataprep_staging"
X = STAGING / "e02x"
PATHS = [str(X), str(STAGING), str(STAGING / "e02d"), str(STAGING / "e02c")]

_saved = list(sys.path)
sys.path[:0] = [p for p in PATHS if p not in sys.path]
import build_artefact_e02x as BA  # noqa: E402
import e02xio as io  # noqa: E402
import e02xlib as L  # noqa: E402
import members_e02x as ME  # noqa: E402
import power_e02x as PW  # noqa: E402
import prereg_e02x as PR  # noqa: E402
import score_e02x as SC  # noqa: E402
import sdfit_e02x as SF  # noqa: E402
from harness import prereg3  # noqa: E402

sys.path[:] = _saved

UTC = dt.timezone.utc
PREREG = REPO / "results" / "dataprep" / "preregistration"
NEED_REGISTRY = pytest.mark.skipif(not (PREREG / "INDEX.jsonl").exists(),
                                   reason="the registry (results/, gitignored) is absent")


@pytest.fixture(autouse=True)
def _paths_and_guards(monkeypatch):
    """dataprep_staging on sys.path for each test only; no data product may be opened and
    no pre-registration may be written."""
    saved = list(sys.path)
    sys.path[:0] = [p for p in PATHS if p not in sys.path]
    from common import guard

    def refuse(*a, **k):
        raise AssertionError("tests must not open a data product")

    def no_write(*a, **k):
        raise AssertionError("write_prereg must never be called by the builder or its tests")

    monkeypatch.setattr(guard, "open_product", refuse)
    monkeypatch.setattr(prereg3, "write_prereg", no_write)
    yield
    sys.path[:] = saved


# ----------------------------------------------------------------------------- synthetic
def _donors(rng=None) -> pd.DataFrame:
    rows = []
    for c, zones, n in (("Yunlin", ["prox", "mid", "dist"], 30),
                        ("Changhua", ["prox", "mid", "dist"], 24)):
        for i in range(n):
            z = zones[i % 3]
            g = z if c == "Yunlin" or z == "prox" else "mid+dist"
            rows.append({"station_id": f"{'09' if c == 'Yunlin' else '07'}0{i:05d}",
                         "county": c, "zone": z, "group": f"{c}:{g}",
                         "subgroup": f"{c}:{['L1', 'L2', 'L34'][i % 3]}"})
    return pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)


def _ext() -> pd.DataFrame:
    rows = []
    for c, n in L.EXPECTED_EXT.items():
        pre = {"Yunlin": "09", "Changhua": "07", "Nantou": "08"}[c]
        for i in range(n):
            rows.append({"station_id": f"{pre}9{i:05d}", "county": c,
                         "zone": ["prox", "mid", "dist"][i % 3], "cell": i, "layer": 1,
                         "layer_group": ["L1", "L2", "L34"][(i // 3) % 3]})
    d = _donors()
    groups = {c: sorted(set(d.loc[d.county == c, "group"])) for c in L.GATED}
    subs = {c: sorted(set(d.loc[d.county == c, "subgroup"])) for c in L.GATED}
    return L.ext_groups(pd.DataFrame(rows), groups, subs, {c: "zone" for c in L.GATED})


# ----------------------------------------------------------------------------- library
def test_unit_names_use_the_real_counts():
    assert L.N_EXT == 66 and L.N_GATED == 60
    assert L.unit_id("primary") == "E02x:ext_heads_2026-12_2027-10_wells60"
    assert L.unit_id("fallback") == "E02x:ext_heads_2027-01_2027-10_wells60"
    assert L.nantou_unit_id("primary") == "E02x:ext_heads_2026-12_2027-10_nantou6"
    assert L.prereg_id("primary") == "E02x.ext_transfer_2026-12.v1"
    assert len(L.months("primary")) == 11 and len(L.months("fallback")) == 10
    assert L.MIN_WELLS_COUNTY == 20 and L.POINT_WINDOW == (0.85, 0.95)
    assert L.MIN_DONORS == 5 and L.MIN_MONTHS == 6 and L.D.GATE_METHOD == "upper90"


def test_group_mapping_merges_and_nantou_report_only():
    e = _ext()
    assert L.map_to_group("Changhua", "dist", ["Changhua:prox", "Changhua:mid+dist"]) == \
        "Changhua:mid+dist"
    assert L.map_to_group("Yunlin", "dist", ["Changhua:prox"]) is None
    ch = e[e.county == "Changhua"]
    assert set(ch["group"]) == {"Changhua:prox", "Changhua:mid+dist"}
    na = e[e.county == "Nantou"]
    assert set(na["role"]) == {"report_only"} and all(k.startswith("zone:")
                                                      for k in na["donor_key"])
    assert set(e.loc[e.county.isin(L.GATED), "role"]) == {"gate_candidate"}
    m = L.donor_masks(e, _donors())
    assert m["zone:dist"].sum() == 18 and m["group:Changhua:mid+dist"].sum() == 16


def test_departure_matches_brute_force_and_needs_five_donors():
    rng = np.random.default_rng(0)
    d, e = _donors(), _ext()
    R = rng.normal(size=(len(d), 4))
    R[rng.random(R.shape) < 0.3] = np.nan
    masks = L.donor_masks(e, d)
    keys = e["donor_key"].tolist()
    D = L.departure(R, masks, keys)
    for j, k in enumerate(keys):
        for t in range(4):
            v = R[masks[k], t]
            v = v[np.isfinite(v)]
            if v.size >= 5:
                assert abs(np.median(v) - D[j, t]) < 1e-12
            else:
                assert np.isnan(D[j, t])
    Rs = np.full_like(R, np.nan)
    Rs[:4] = 1.0                                   # 4 donors at most in any group
    assert np.isnan(L.departure(Rs, masks, keys)).all()


def test_transfer_cancels_a_shared_group_shock():
    rng = np.random.default_rng(1)
    d, e = _donors(), _ext()
    masks, keys = L.donor_masks(e, d), e["donor_key"].tolist()
    Rd, Re = rng.normal(size=(len(d), 6)), rng.normal(size=(len(e), 6))
    e1 = L.transfer_relative(Re, Rd, masks, keys)
    gd = d["group"].to_numpy()
    ge = e["group"].to_numpy()
    shock = {g: rng.normal(size=6) * 5 for g in set(gd)}
    gated = e["county"].isin(L.GATED).to_numpy()
    Rd2 = Rd + np.stack([shock[g] for g in gd])
    Re2 = Re.copy()
    Re2[gated] += np.stack([shock[g] for g in ge[gated]])
    e2 = L.transfer_relative(Re2, Rd2, masks, keys)
    assert np.allclose(e1[gated], e2[gated], atol=1e-12)


def test_sigma_ens_rel_and_idio():
    rng = np.random.default_rng(2)
    d, e = _donors(), _ext()
    masks, keys = L.donor_masks(e, d), e["donor_key"].tolist()
    common = rng.normal(size=(36, 1, 5))
    s0 = L.sigma_ens_rel(np.repeat(common, len(e), 1), np.repeat(common, len(d), 1), masks,
                         keys)
    assert np.allclose(s0, 0.0)                     # a spread shared by all wells cancels
    Me = rng.normal(size=(36, len(e), 5))
    s1 = L.sigma_ens_rel(Me, np.repeat(common, len(d), 1), masks, keys)
    assert (s1 > 0).all()
    sig = L.sigma_idio(np.full(len(e), 0.3), s1, np.full(len(e), 0.01), np.full((1, 5), 4.0))
    assert np.allclose(sig, np.sqrt(0.09 + s1 ** 2 + 0.16))
    assert L.drift_tau2({"Yunlin": {"include_drift": True, "tau2_per_year2": 0.02},
                         "Changhua": {"include_drift": False, "tau2_per_year2": 0.5}},
                        "Changhua") == 0.0
    assert L.drift_tau2({"Yunlin": {"include_drift": True, "tau2_per_year2": 0.02},
                         "Changhua": {"include_drift": True, "tau2_per_year2": 0.05}},
                        "Nantou") == 0.05


# ----------------------------------------------------------------------------- power
def _design(seed=11):
    u1, sha, ext, prov = PW.synthetic_inputs(seed)
    return PW.design_spec(u1, ext, seed, sha, prov)


def test_power_design_grid_seed_and_self_check():
    with pytest.raises(TypeError):
        PW.design_spec(*PW.synthetic_inputs(1)[:1], PW.synthetic_inputs(1)[2], None)
    d = _design()
    assert d["seed"] == 11 and d["self_check_seed"] == 12
    assert {w: v["months"] for w, v in d["windows"].items()} == {"primary": 11,
                                                                 "fallback": 10}
    for c in L.GATED:
        cd = d["counties"][c]
        assert cd["ext"]["wells"] == L.EXPECTED_EXT[c]
        assert cd["a_x"] == [0.0, max(cd["a_s"])]
        assert cd["donor_dropout"] == [0.0, 0.2, 0.4] == cd["ext_dropout"]   # U1's grid
        cs = PW.cells(cd)
        assert sum(x["gating"] for x in cs) == 3 * 3 * 2 * 2 * 3 * 3 * 3
        assert PW.self_check(cd, d["self_check_seed"])["ok"]
        r = PW.run_cell(cd, cs[5], 11, 0, 0, 30, 20, 3)
        assert 0 <= r["P_fail"] <= 1 and r["k_p"] > 0
    assert d["decision"]["false_fail_max"] == 0.12 and d["decision"]["power_min"] == 0.80
    assert d["arithmetic_bounds"]["Changhua"]["report_only_by_arithmetic"]
    assert "WITHOUT external or donor dropout" in d["calibration_disclosure"]
    with pytest.raises(io.Refused):
        PW.run(d, 10, 10, 12, smoke=True)           # a seed other than the design's


def test_power_decision_and_registration_rule():
    rows = [{"gating": True, "heavy": False, "coverage": cv, "P_fail": pf, "cell": i}
            for i, (cv, pf) in enumerate(((0.9, 0.05), (0.9, 0.12), (0.75, 0.95),
                                          (0.75, 0.81), (0.8, 0.5)))]
    dd = PW.decide(rows)
    assert dd["p_gate_pass"] and dd["false_fail_max"] == 0.12
    rows[1]["P_fail"] = 0.13                         # 0.12 is the ceiling (E02d had 0.20)
    assert not PW.decide(rows)["p_gate_pass"]
    ok, no = {"p_gate_pass": True}, {"p_gate_pass": False}
    r = PW.registration_rule({"primary": {"Yunlin": ok, "Changhua": ok},
                              "fallback": {"Yunlin": no, "Changhua": ok}})
    assert r["primary"] == {"registrable": True, "roles": {"Yunlin": "primary",
                                                           "Changhua": "report-only",
                                                           "Nantou": "report-only"}}
    assert not r["fallback"]["registrable"]
    free = {c: dict(v, report_only_by_arithmetic=False)
            for c, v in L.arithmetic_bounds().items()}          # a counterfactual floor
    r2 = PW.registration_rule({"primary": {"Yunlin": ok, "Changhua": ok}}, free)
    assert r2["primary"]["roles"]["Changhua"] == "primary"


def test_arithmetic_bounds_from_declared_constants():
    b = L.arithmetic_bounds()
    ch, yl = b["Changhua"]["p_at_least_floor_by_dropout"], b["Yunlin"][
        "p_at_least_floor_by_dropout"]
    assert abs(ch["0.2"] - 0.2965) < 1e-3 and abs(ch["0.4"] - 0.00516) < 1e-4
    assert abs(yl["0.4"] - 0.8180) < 1e-3
    assert b["Changhua"]["report_only_by_arithmetic"]
    assert not b["Yunlin"]["report_only_by_arithmetic"]
    assert L.p_survive(23, 20, 0.0) == 1.0
    assert "0.296531" in PR.CHANGHUA_ARITH and "REPORT-ONLY BY ARITHMETIC" in PR.CHANGHUA_ARITH
    assert "0.818" in PR.YUNLIN_BOUND


def test_breakdown_suppresses_small_groups():
    H = np.ones((8, 3))
    lab = np.array(["a"] * 5 + ["b"] * 3)
    out = L.breakdown(H, lab)
    assert out["a"]["coverage"] == 1.0
    assert out["b"]["coverage"] is None and "suppressed" in out["b"]


def test_power_smoke_refuses_results_and_needs_seed(tmp_path):
    assert PW.main(["smoke", "--seed", "3", "--out", str(REPO / "results" / "x")]) == 1
    with pytest.raises(SystemExit):
        PW.main(["smoke", "--out", str(tmp_path)])  # --seed is required
    with pytest.raises(SystemExit):
        PW.main(["run"])


# ----------------------------------------------------------------------------- members
def _npz(path: Path, heads: np.ndarray, label: str, n_members: int, dates, origin: int):
    np.savez(path, heads_mean=heads[None], scenario_names=np.array(["baseline"]),
             n_members=np.array(n_members), member_labels=np.array([label]),
             dates=np.array(dates), origin=np.array(origin))


def test_member_extract_gate(tmp_path):
    rng = np.random.default_rng(5)
    T, C = 30, 40
    dates = [str(x.date()) for x in pd.date_range("2012-01-01", periods=T, freq="MS")]
    labels = ["a", "b"]
    donors = pd.DataFrame({"station_id": ["d1", "d2", "d3"], "cell": [0, 5, 9],
                           "layer": [1, 2, 4]})
    ext = pd.DataFrame({"station_id": ["e1", "e2"], "cell": [3, 7], "layer": [1, 3]})
    md = tmp_path / "runs"
    md.mkdir()
    allm = []
    for lab in labels:
        x0 = rng.normal(size=(4, C, T))
        x1 = x0 + np.concatenate([np.zeros((4, C, 10)), rng.normal(size=(4, C, T - 10))], 2)
        _npz(md / f"{lab}_ic0.npz", x0, lab, 1, dates, 9)
        _npz(md / f"{lab}_ic1.npz", (x0 + x1) / 2, lab, 2, dates, 9)
        allm += [x0, x1]
    M = np.stack(allm)
    page = tmp_path / "page.npz"
    _npz(page, M.mean(axis=0), "page", 36, dates, 9)
    ld, cd = donors["layer"].to_numpy() - 1, donors["cell"].to_numpy()
    ref = {"members": M[:, ld, cd, :], "station_id": donors["station_id"].to_numpy(),
           "member": np.array(["a", "a", "b", "b"]), "ic": np.array([0, 1, 0, 1])}
    arr, gate = ME.extract(donors, ext, labels, md, page, ref, last=dates[-1][:7])
    assert gate["ok"], gate
    assert arr["members_ext"].shape == (4, 2, T)
    le, ce = ext["layer"].to_numpy() - 1, ext["cell"].to_numpy()
    assert np.allclose(arr["members_ext"], M[:, le, ce, :])
    bad = dict(ref, members=ref["members"] + 2e-3)
    _, g2 = ME.extract(donors, ext, labels, md, page, bad, last=dates[-1][:7])
    assert not g2["ok"] and g2["max_abs_donor_vs_members_wells_m"] > 1e-3


# ----------------------------------------------------------------------------- sd read
def test_sd_fit_rel_operator_and_plan_reads_nothing(capsys):
    rng = np.random.default_rng(6)
    d, e = _donors(), _ext()
    T = 40
    dates = pd.date_range("2012-01-01", periods=T, freq="MS")
    shock = {g: rng.normal(size=T) * 3 for g in set(d["group"])}
    rd = np.stack([shock[g] for g in d["group"]]) + rng.normal(0, 0.1, (len(d), T))
    re_ = np.zeros((len(e), T))
    for i, r in e.iterrows():
        base = shock[r["group"]] if r["county"] in L.GATED else 0.0
        re_[i] = base + rng.normal(0, 0.5, T)
    re_[0, 11:] = np.nan                             # 11 months: no sd_fit_rel
    out = SF.compute(re_, rd, e, d, dates)
    assert list(out.columns) == list(SF.WELL_COLS)
    g = out[out.county == "Yunlin"].set_index("station_id")
    assert np.isnan(g.loc[e.station_id[0], "sd_fit_rel_m"])
    ok = g["sd_fit_rel_m"].dropna()
    assert ((ok > 0.35) & (ok < 0.75)).all()         # the group shock was removed
    assert SF.main(["plan"]) == 0
    txt = capsys.readouterr().out
    assert "L1_clean/v3" in txt and L.PRE_UNIT in txt


def test_sd_read_refuses_without_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "REPO", tmp_path)
    (tmp_path / "results" / "dataprep" / "preregistration").mkdir(parents=True)
    with pytest.raises(io.Refused):
        SF.run_real("primary", prereg_dir=tmp_path / "results" / "dataprep" /
                    "preregistration", rows_dir=tmp_path / "rows")


# ----------------------------------------------------------------------------- artefact
def test_artefact_build():
    rng = np.random.default_rng(7)
    d, e = _donors(), _ext()
    T = 214
    dates = [str(x.date()) for x in pd.date_range("2012-01-01", periods=T, freq="MS")]
    wells = e.assign(sd_fit_rel_m=0.4, t_centre_year=2017.5, n_fit=100, n_rel=100)
    wells.loc[3, "sd_fit_rel_m"] = np.nan
    members = {"members_ext": rng.normal(size=(36, len(e), T)),
               "members_donor": rng.normal(size=(36, len(d), T)),
               "ext_station_id": e["station_id"].to_numpy(),
               "donor_station_id": d["station_id"].to_numpy(), "dates": np.array(dates)}
    ms = L.months("primary")
    pred = pd.DataFrame([{"station_id": s, "date": str(m.date()), "predicted_m": 10.0}
                         for s in e["station_id"] for m in ms])
    dr = {"decisions": {"Yunlin": {"include_drift": True, "tau2_per_year2": 0.001},
                        "Changhua": {"include_drift": False, "tau2_per_year2": 0.002}},
          "lead_definition": "datum_centre"}
    arts = BA.build(wells, d, members, pred, dr)
    a = arts["primary"]
    assert list(a.columns) == list(BA.ART_COLS)
    assert a["station_id"].nunique() == len(e) - 1 and len(arts["fallback"]) < len(a)
    assert (a["sigma_idio_m"] >= a["sd_fit_rel_m"] - 1e-12).all()
    assert (a.loc[a.county == "Changhua", "drift_m"] == 0).all()
    assert (a.loc[a.county == "Yunlin", "drift_m"] > 0).all()
    assert set(a["role"]) == {"gate_candidate", "report_only"}
    with pytest.raises(io.EngineeringFailure):         # a missing lead definition is an error
        BA.build(wells, d, members, pred, {"decisions": dr["decisions"]})


# ----------------------------------------------------------------------------- GOV
def _hw():
    spec = importlib.util.spec_from_file_location(
        "test_e02x_hw", STAGING / "gov_heads_window" / "heads_window.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _product(root: Path, hw, name: str, *, window="keep", fit_window_end="2029-10") -> Path:
    d = root / name / "v1"
    d.mkdir(parents=True)
    (d / "heads.csv").write_text("station_id,date,head_m,flag\n", encoding="utf-8")
    (d / "validation.json").write_text("{}\n", encoding="utf-8")
    rec = {"path": "heads.csv", "sha256": "0" * 64}
    m = hw.instantiate(hw.template(), version="v1", created="2028-01-10",
                       inputs=[dict(rec, path="data/synthetic.csv")], code=[rec],
                       git_commit="0" * 40, fit_window_end=fit_window_end, outputs=[rec],
                       validation_report="validation.json",
                       gate={"passed": True, "criteria": "synthetic"},
                       window=None if window in ("keep", None) else tuple(window))
    if window is None:
        del m["parameters"][hw.FIELD]
    (d / "MANIFEST.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return d


def test_gov_precondition_entry_is_private_and_needs_a_declared_window(tmp_path):
    hw = _hw()
    before = sorted(hw.PRECONDITIONS)
    full = _product(tmp_path, hw, "full")
    ok, why = SC.precondition(full, "primary")
    assert ok, why
    y1 = _product(tmp_path, hw, "y1", window=("2026-11", "2027-10"), fit_window_end="2027-10")
    assert SC.precondition(y1, "primary")[0] and SC.precondition(y1, "fallback")[0]
    none = _product(tmp_path, hw, "none", window=None)
    ok, why = SC.precondition(none, "primary")
    assert not ok and "declares no" in why
    late = _product(tmp_path, hw, "late", window=("2027-11", "2029-10"))
    assert not SC.precondition(late, "primary")[0]
    assert sorted(hw.PRECONDITIONS) == before        # no other module object is touched
    with pytest.raises(io.Refused):                   # B2: wrong path / no MANIFEST / unreadable
        SC.precondition(tmp_path / "missing", "primary")
    (tmp_path / "empty").mkdir()
    with pytest.raises(io.Refused):
        SC.precondition(tmp_path / "empty", "primary")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "MANIFEST.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(io.Refused):
        SC.precondition(tmp_path / "broken", "primary")
    with pytest.raises(io.EngineeringFailure):        # S1: heads_window.py not GOV's
        SC._hw("0" * 64)
    assert SC.HW_SHA.startswith("9a953039")
    assert L.prereg_id("primary") not in _hw().PRECONDITIONS
    e = SC.gov_entry("fallback")
    assert e["months"] == ("2027-01", "2027-10") and e["needs_window"] and e["code"] is None


# ----------------------------------------------------------------------------- scorer
def _frames(rng, win="primary", shift=0.0):
    d, e = _donors(), _ext()
    ms = L.months(win)
    shock = {g: rng.normal(size=len(ms)) * 2 for g in set(d["group"])}
    zone_shock = {z: rng.normal(size=len(ms)) * 2 for z in ("prox", "mid", "dist")}
    drow, erow, art = [], [], []
    for _, r in d.iterrows():
        for j, m in enumerate(ms):
            drow.append({"station_id": r.station_id, "date": str(m.date()), "predicted_m": 5.0,
                         "observed_m": 5.0 + shock[r.group][j] + zone_shock[r.zone][j] * 0
                         + rng.normal(0, 0.1)})
    for _, r in e.iterrows():
        for j, m in enumerate(ms):
            base = shock[r.group][j] if r.county in L.GATED else 0.0
            erow.append({"station_id": r.station_id, "date": str(m.date()), "predicted_m": 3.0,
                         "observed_m": 3.0 + base + rng.normal(0, 1.0) + shift})
            art.append({"station_id": r.station_id, "county": r.county, "role": r.role,
                        "zone": r.zone, "group": r.group, "donor_key": r.donor_key,
                        "date": str(m.date()), "predicted_m": 3.0, "sigma_idio_m": 1.0})
    return pd.DataFrame(drow), pd.DataFrame(erow), pd.DataFrame(art), d


def test_outcome_and_wording():
    rng = np.random.default_rng(8)
    drows, erows, art, d = _frames(rng)
    roles = {"Yunlin": "primary", "Changhua": "report-only", "Nantou": "report-only"}
    out = SC.outcome(drows, erows, art, d, roles, "primary")
    y = out["counties"]["Yunlin"]
    assert y["verdict"] in ("pass", "fail") and y["wells"] == 37 and 0.8 < y["coverage"] < 0.97
    assert set(y["by_zone_report_only"]) == {"prox", "mid", "dist"}
    assert out["counties"]["Nantou"]["verdict"] == "not_evaluable"   # 6 < 20, report-only
    doc = {"id": L.prereg_id("primary"), "page_mapping": {"wording": PR.WORDING}}
    w = SC.wording(doc, out)
    assert "report-only" in w["Changhua"] and "undecided" in w["Nantou"]
    assert not any("calibrated" in s for s in w.values())
    _, erows2, _, _ = _frames(np.random.default_rng(8), shift=3.0)
    bad = SC.outcome(drows, erows2, art, d, roles, "primary")
    assert bad["counties"]["Yunlin"]["verdict"] == "fail"
    art2 = art.copy()
    art2.loc[0, "predicted_m"] = 3.5
    with pytest.raises(io.EngineeringFailure):
        SC.outcome(drows, erows, art2, d, roles, "primary")


def _ns(tmp_path, out="out"):
    return argparse.Namespace(window="primary", window_product=tmp_path / "prod",
                              step5_dir=tmp_path / "s5", ext_step5_dir=tmp_path / "x5",
                              e02c_report=None, e02d_report=None, n3f_report=None,
                              out=tmp_path / out)


def test_scorer_refuses_before_registration_and_logs_every_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "REPO", tmp_path)
    pdir = tmp_path / "prereg"
    pdir.mkdir()
    a = _ns(tmp_path)
    for _ in range(2):
        with pytest.raises(io.Refused):
            SC.run_score(a, prereg_dir=pdir)
    rows = [json.loads(s) for s in (a.out / "attempts.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in rows] == ["refused", "refused"]
    assert "not registered" in rows[0]["why"]
    assert (tmp_path / SC.RUN_ATTEMPTS).is_file()
    assert not (a.out / "report.json").exists()


def test_scorer_looks_before_rows_writes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "REPO", tmp_path)
    rng = np.random.default_rng(9)
    drows, erows, art, d = _frames(rng)
    doc = {"id": L.prereg_id("primary"), "wp": "E02x",
           "tests": [{"test_id": L.test_id("primary"),
                      "role_by_county": {"Yunlin": "primary", "Changhua": "primary"}},
                     {"test_id": L.nantou_test_id("primary")}],
           "page_mapping": {"wording": PR.WORDING}}
    calls = []
    monkeypatch.setattr(io, "registered", lambda *a_, **k: (doc, "f" * 64))
    monkeypatch.setattr(io, "now", lambda: dt.datetime(2028, 1, 20, tzinfo=UTC))
    monkeypatch.setattr(SC, "verify_code", lambda *a_: {"heads_window_sha256": SC.HW_SHA})
    monkeypatch.setattr(SC, "precondition", lambda *a_: (True, "synthetic"))
    monkeypatch.setattr(SC, "order_status", lambda *a_: {"ok": True})
    monkeypatch.setattr(SC, "_check_artefact", lambda doc_, w: art)
    import wells_e02x as WX

    monkeypatch.setattr(WX, "u1_design", lambda *a_, **k: ({}, "x"))
    monkeypatch.setattr(WX, "donors_table", lambda *a_, **k: d)
    monkeypatch.setattr(SC, "_e02_files", lambda: (None, None, None))
    monkeypatch.setattr(SC, "_n3f_files", lambda *a_: (None, None, None))

    def fake_score(pid, wp, test_id=None, **k):
        calls.append(("look", test_id))
        return {"tests": [test_id], "post_registration_looks": []}

    def fake_rows(step5, *a_):
        calls.append(("rows", Path(step5).name))
        return (drows if Path(step5).name == "s5" else erows), {"synthetic": True}

    monkeypatch.setattr(prereg3, "assert_can_score", fake_score)
    monkeypatch.setattr(SC, "verified_rows", fake_rows)
    a = _ns(tmp_path)
    rep = SC.run_score(a)
    assert calls == [("look", L.test_id("primary")), ("look", L.nantou_test_id("primary")),
                     ("rows", "s5"), ("rows", "x5")]
    assert rep["outcome"]["counties"]["Yunlin"]["role"] == "primary"
    assert rep["outcome"]["counties"]["Nantou"]["role"] == "report-only"
    with pytest.raises(io.Refused):
        SC.run_score(a)                               # scored once
    rows = [json.loads(s) for s in (a.out / "attempts.jsonl").read_text().splitlines()]
    assert [r["outcome"] for r in rows] == ["scored", "refused"]
    monkeypatch.setattr(SC, "precondition", lambda *a_: (False, "no grant"))
    b = _ns(tmp_path, "out_ne")
    calls.clear()
    r = SC.run_score(b)
    assert r["verdict"] == "not_evaluable" and calls == []          # nothing read, no look

    def refuse_manifest(*a_):
        raise io.Refused("no MANIFEST.json")

    monkeypatch.setattr(SC, "precondition", refuse_manifest)
    c = _ns(tmp_path, "out_nm")
    with pytest.raises(io.Refused):                   # B2: refused, logged, retryable
        SC.run_score(c)
    assert not (c.out / "report.json").exists()
    assert json.loads((c.out / "attempts.jsonl").read_text().splitlines()[0])[
        "outcome"] == "refused"
    seen = []
    monkeypatch.setattr(SC, "order_status", lambda *a_: {"ok": False, "x": {"ok": False,
                                                                            "why": "E02d"}})
    monkeypatch.setattr(SC, "precondition", lambda *a_: seen.append(1) or (True, ""))
    with pytest.raises(io.Refused):                   # the order check runs first
        SC.run_score(_ns(tmp_path, "out_order"))
    assert seen == []
    monkeypatch.setattr(io, "now", lambda: dt.datetime(2028, 1, 10, tzinfo=UTC))
    with pytest.raises(io.Refused):
        SC.run_score(_ns(tmp_path, "early"))


def test_thin_rows_need_an_n3f_record(tmp_path):
    ms = L.months("primary")
    pred = pd.DataFrame([{"station_id": "091", "date": str(m.date()), "predicted_m": 1.0,
                          "lower_m": 0.0, "upper_m": 2.0, "baseline_m": 1.0} for m in ms])
    s5 = tmp_path / "s5"
    s5.mkdir()
    pred.assign(observed_m=1.1).to_csv(s5 / "evaluation.csv", index=False)
    with pytest.raises(io.Refused):
        SC.verified_rows(s5, pred, None, {}, "primary")
    with pytest.raises(io.Refused):
        SC.verified_rows(s5, pred, None, {}, "primary",
                         {"evaluate_error": "ValueError: 25 wells < 30",
                          "evaluation_sha256": "0" * 64})
    rows, prov = SC.verified_rows(s5, pred, None, {}, "primary",
                                  {"evaluate_error": "ValueError: 25 wells < 30",
                                   "evaluation_sha256": io.sha256(s5 / "evaluation.csv")})
    assert len(rows) == 11 and "columns only" in prov["verify"]


def test_sdfit_csv_write_is_retry_safe(tmp_path):
    p = tmp_path / "wells_e02x.csv"
    p.write_text("stale\n")
    h = SF._write_csv_retry_safe(pd.DataFrame({"a": [1]}), p)
    assert p.read_text().startswith("a") and h == io.sha256(p)
    assert not (tmp_path / "wells_e02x.csv.partial").exists()


@NEED_REGISTRY
def test_e02d_registered_member_gate():
    assert ME.e02d_registered_gate().startswith(ME.E02D_GATE_SHA_PREFIX)


# ----------------------------------------------------------------------------- runner
def test_runner_dry_run_matches_e02d_command():
    sh = X / "run_e02x.sh"
    subprocess.run(["bash", "-n", str(sh)], check=True)
    env = dict(os.environ, DRY="1")
    r = subprocess.run(["bash", str(sh), "members", "all"], env=env, capture_output=True,
                       text=True, check=True, cwd=REPO)
    lines = [s for s in r.stdout.splitlines() if s.strip()]
    assert len(lines) == 36 and all("--data" not in s for s in lines)
    r1 = subprocess.run(["bash", str(sh), "members", "split1"], env=env, capture_output=True,
                        text=True, check=True, cwd=REPO)
    r2 = subprocess.run(["bash", str(sh), "members", "split2"], env=env, capture_output=True,
                        text=True, check=True, cwd=REPO)
    assert r1.stdout.splitlines() + r2.stdout.splitlines() == r.stdout.splitlines()
    d = subprocess.run(["bash", str(STAGING / "e02d" / "run_e02d.sh"), "members", "fold0"],
                       env=env, capture_output=True, text=True, check=True, cwd=REPO)
    mine = [s for s in lines if "/fold0/" in s]
    assert [s.replace("e02x_members", "M") for s in mine] == \
        [s.replace("e02d_members", "M") for s in d.stdout.splitlines() if s.strip()]


# ----------------------------------------------------------------------------- drafts
@NEED_REGISTRY
def test_drafts_validate_against_the_registry():
    """The drafts build from the registry and metadata only and validate under prereg3."""
    if not (REPO / L.N3F_DIR / "wells.csv").is_file():
        pytest.skip("N3F's export (results/, gitignored) is absent")
    dp = PR.draft_prereg_power(20261203)
    assert prereg3.validate_prereg(dp) == []
    assert dp["design"]["seed"] == 20261203
    assert {c: v["n"] for c, v in dp["well_list"].items()} == L.EXPECTED_EXT
    de = PR.draft_prereg_e02x("primary")
    assert prereg3.validate_prereg(de) == []
    t0 = de["tests"][0]
    assert t0["fresh_units"] == [L.unit_id("primary")]
    assert len(t0["external_units"][L.unit_id("primary")]["ids"]) == 60
    assert de["tests"][1]["fresh_units"] == [L.nantou_unit_id("primary")]
    assert "evc_pro_ext" in de["page_mapping"]["pass"]
    assert t0["role_by_county"]["Changhua"].startswith("report-only (by arithmetic")
    assert dp["arithmetic_bounds"]["Changhua"]["report_only_by_arithmetic"]
    assert "0.818" in dp["yunlin_bound"] and "0.818" in de["disclosures"]["yunlin_bound"]
    for k in ("calibration", "changhua", "nantou", "fallback_deadline", "members"):
        assert de["disclosures"][k]
    assert "IDENTITY" in de["disclosures"]["members"]
    assert "ABOVE 0.95" in de["disclosures"]["product_mix"]
    assert PR.e02x_problems("primary", de, "x")       # nothing upstream exists yet
