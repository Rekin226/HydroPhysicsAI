"""CPU tests of I1.CH_backcast.v1 and heads/L1_pre2012 v1 (round 7 rank 18) and of the
hindcast panel assembly (rank 13), on SYNTHETIC files only: no ls_cache record, no head and
no model run is touched; registries, looks and products live in pytest's tmp dirs.

    cd <repository root>
    ~/miniforge3/envs/hydro/bin/python -m pytest -q -p no:cacheprovider dataprep_staging/ch_x
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(X), str(REPO / "dataprep_staging"), str(TESTS)):
    if p not in sys.path:
        sys.path.insert(0, p)
import avail_l1pre as AV  # noqa: E402
import bc_lib as BL  # noqa: E402
import build_l1pre as BLD  # noqa: E402
import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402
import hindcast as HC  # noqa: E402
import prereg_bc as PB  # noqa: E402
import score_bc as SB  # noqa: E402
from test_chx import fake_wells  # noqa: E402

BCW = L.month_list(*K.BC_WINDOW)


# ----------------------------------------------------------------------------- availability
def test_available_months_counts_days_with_enough_hours():
    full = pd.date_range("2010-01-01", "2010-01-31 23:00", freq="h")
    half = pd.date_range("2010-02-01", "2010-02-15 23:00", freq="h")
    thin = pd.date_range("2010-03-01", "2010-03-31", freq="D")        # 1 sample per day
    outside = pd.date_range("2012-01-01", "2012-01-31 23:00", freq="h")
    got = BL.available_months(full.append(half).append(thin).append(outside))
    assert got == ["2010-01"]


def test_nest_wells_and_feasibility():
    xy = {"a": (100.0, 0.0), "b": (2000.0, 0.0), "c": None}
    assert BL.nest_wells(xy, np.array([[0.0, 0.0]])) == {"a"}
    months = {f"07{i:06d}": BCW[:20] for i in range(11)}
    months["07999999"] = BCW[:5]
    county = {s: "changhua" for s in months}
    f = BL.feasibility(months, county, {"07000000"})
    assert f["n_eligible"] == 10 and f["passed"] and f["n_excluded_nest"] == 1
    f2 = BL.feasibility(months, county, {"07000000", "07000001"})
    assert not f2["passed"]


def test_parse_xy_either_order():
    assert BL.parse_xy("190000 2650000") == (190000.0, 2650000.0)
    assert BL.parse_xy("2650000 190000") == (190000.0, 2650000.0)
    assert BL.parse_xy("bad") is None


# ----------------------------------------------------------------------------- QC
def synth_series(seed=0, level=5.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", "2011-12-31 23:00", freq="h")
    v = level + 0.3 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 365)) \
        + 0.01 * rng.standard_normal(len(idx))
    s = pd.Series(v, index=idx)
    s.iloc[1000] = level + 40.0                                      # spike
    s.iloc[5000:5010] = -999998.0                                    # sentinels
    s = s.drop(s.index[(s.index >= "2011-06-05") & (s.index < "2011-06-25")])   # thin June
    return s


def test_qc_monthly_drops_artefacts_and_thin_months():
    out = BL.qc_monthly(synth_series(), elev=10.0, depth=100.0)
    m = out["monthly"]
    assert "2011-06" not in m.index and len(m) == 23
    assert np.allclose(m.to_numpy(), 5.0, atol=0.35)
    assert out["n_clean"] < out["n_raw"]


def test_build_tables_and_manifest_in_tmp(tmp_path):
    st = pd.DataFrame({"sid": ["07000001", "07000002"], "county": "changhua",
                       "x": [190000.0, 191000.0], "y": [2650000.0, 2651000.0],
                       "layer_code": ["1", "2"], "WellElevation": [10.0, 12.0],
                       "WellDepth": [100.0, 80.0]})
    tb = BLD.build_tables(list(st["sid"]), st, loader=lambda s: synth_series(int(s[-1])))
    val = BLD.validate_tables(tb)
    assert val["passed"] and val["post_qc_counts"]["n_wells_ge_min_months"] == 2
    d = tmp_path / "v1"
    d.mkdir()
    raw = tmp_path / "raw.parquet"
    raw.write_bytes(b"x")
    m = BLD.write_product(d, tb, val, list(st["sid"]), "0" * 64, [str(raw)])
    assert m["allowed_uses"] == list(K.L1PRE_ALLOWED)
    assert set(K.L1PRE_FORBIDDEN) <= set(m["forbidden_uses"])
    assert m["fit_window_end"] == "2011-12" and "local_only" in m
    assert (d / "MANIFEST.json").exists()


# ----------------------------------------------------------------------------- scoring
def bc_panel(seed=0, W=12, k=1.0):
    rng = np.random.default_rng(seed)
    T = len(BCW)
    sid = np.array([f"07{i:06d}" for i in range(W)])
    x = 175000.0 + 6000 * (np.arange(W) % 4)
    y = 2640000.0 + 6000 * (np.arange(W) // 4)
    bl = L.well_blocks(sid, x, y)
    a = 0.6 * rng.standard_normal(T)
    c = -rng.uniform(0.5, 2.5, W)[:, None] * a[None, :]
    B = 20.0 + rng.normal(0, 2, (W, 1)) + np.tile(np.sin(np.arange(T) / 2.0), (W, 1))
    P = {"months": np.array(BCW), "sid": sid, "x": x, "y": y, "c": c, "B": B, "A": B + c,
         "block": np.array([bl[s] for s in sid]), "a_ch": a}
    obs = B + k * c + 0.05 * rng.standard_normal((W, T)) \
        + 0.5 * np.sin(2 * np.pi * np.arange(T) / 12)[None]
    return P, obs


def test_score_backcast_direction_and_rules():
    P, obs = bc_panel()
    r = SB.score_panel(P, obs, np.zeros(obs.shape, int))
    assert r["verdict"] == "direction_supported" and r["passed"]
    assert r["form"] == K.BC_PRIMARY_FORM == "calmonth_twoway"
    assert r["beta_primary"]["beta"] == pytest.approx(1.0, abs=0.1)
    assert r["informative_gate"]["transformed"] and r["informative_gate"]["passes"]
    P2, obs2 = bc_panel(k=-1.0)
    assert SB.score_panel(P2, obs2, np.zeros(obs2.shape, int))["verdict"] == "not_supported"
    obs3 = obs.copy()
    obs3[3:, :10] = np.nan
    r3 = SB.score_panel(P, obs3, np.zeros(obs3.shape, int))
    assert r3["verdict"] == "not_evaluable" and not r3["passed"]


def test_calmonth_basis_removes_well_month_means_and_year_shocks():
    w = np.repeat(["a", "b"], 24)
    m = np.tile(BCW, 2)
    v = np.tile(np.arange(24.0) % 12, 2) + np.repeat([0.0, 5.0], 24)
    assert np.allclose(L.fe_resid(v, L.fe_basis(w, m, "calmonth")), 0.0)
    shock = np.tile(np.r_[np.zeros(12), np.ones(12)], 2)            # a common 2011 shift
    assert np.allclose(L.fe_resid(v + shock, L.fe_basis(w, m, "calmonth_twoway")), 0.0)
    assert not np.allclose(L.fe_resid(v + shock, L.fe_basis(w, m, "calmonth")), 0.0)


def test_month_of_sample_effects_absorb_a_common_drought_year():
    P, _ = bc_panel()
    T = len(BCW)
    drought = -1.5 * np.asarray(P["a_ch"])                    # tracks the pumping anomaly
    obs = P["B"] + drought[None, :] + 0.01 * np.random.default_rng(4).standard_normal(
        (len(P["sid"]), T))                                   # k = 0
    r = SB.score_panel(P, obs, np.zeros(obs.shape, int))
    assert r["verdict"] == "not_supported"
    assert abs(r["beta_primary"]["beta"]) < 0.05
    assert r["report_only_calmonth_only"]["ci90"][0] > 0      # the v0 form is fooled


def test_bc_power_is_report_only_and_small_k0():
    P, _ = bc_panel(W=16)
    sid = np.asarray(P["sid"])
    blocks = dict(zip(sid, P["block"], strict=True))
    inf = L.informative_from_contrast(P["c"], sid, blocks, list(P["months"]),
                                      form=K.BC_PRIMARY_FORM)
    pw = L.bc_power(P, {"sigma_tot_m": 1.318, "block_share": 0.439, "phi": 0.859,
                        "kappa_m": 0.296}, inf, reps=20, B=200)
    assert pw["report_only"] and 0.0 <= pw["p_direction_k0"] <= 0.5
    assert pw["p_direction_k1"] >= pw["p_direction_k0"]


def test_calendar_climatology():
    R = np.tile(np.arange(12.0), (3, 11))
    run = L.month_list(*K.RUN_MONTHS)
    out = HC.calendar_climatology(R, run, L.month_list("2007-01", "2007-12"))
    assert np.allclose(out, np.tile(np.arange(12.0), (3, 1)))


# ----------------------------------------------------------------------------- end to end
def _setup_avail(tmp_path, monkeypatch, n=12):
    from harness import prereg3

    monkeypatch.setattr(IO, "REPO", tmp_path)
    ids = [f"07{i:06d}" for i in range(n)]
    st = pd.DataFrame({"sid": ids + ["09000001"],
                       "CountyName": ["彰化縣"] * n + ["雲林縣"],
                       "LocationByTWD97": [f"{175000 + 6000 * (i % 4)} "
                                           f"{2640000 + 6000 * (i // 4)}" for i in range(n)]
                       + ["190000 2610000"],
                       "GroundwaterLayerCode": ["1"] * (n + 1),
                       "GroundwaterZoneIdentifier": [50] * (n + 1),
                       "WellElevation": [10.0] * (n + 1), "WellDepth": [90.0] * (n + 1)})
    (tmp_path / K.STATIONS).parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(tmp_path / K.STATIONS)
    lc = tmp_path / K.LS_CACHE
    lc.mkdir(parents=True)
    for i, s in enumerate(ids):
        ser = synth_series(i)
        pd.DataFrame({"value": ser.to_numpy()}, index=pd.Index(ser.index, name="datetime")
                     ).to_parquet(lc / K.LS_RAW.format(sid=s))
    pd.DataFrame({"sub_id": ["r"], "lon": [0], "lat": [0], "x": [175100.0],
                  "y": [2640000.0]}).to_csv(tmp_path / K.MLCW_STATIONS, index=False)
    pdir = tmp_path / "prereg"
    rec = AV.look_record(AV.candidates())
    prereg3.record_look(rec["units"], rec["data_scope"], rec["wp"], rec["purpose"],
                        members=rec["members"], note=rec["note"], prereg_dir=pdir,
                        rows_dir=tmp_path / "rows", registries=[])
    return ids, pdir


def test_availability_count_refuses_without_look_then_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "REPO", tmp_path)
    with pytest.raises(SystemExit, match="log the availability look"):
        AV.count(tmp_path / "a", prereg_dir=tmp_path / "empty")
    ids, pdir = _setup_avail(tmp_path, monkeypatch)
    rep = AV.count(tmp_path / K.BC_DIR / "avail", prereg_dir=pdir)
    assert rep["nest_excluded"] == ["07000000"]
    assert rep["feasibility"]["n_eligible"] == len(ids) - 1 and rep["feasibility"]["passed"]
    assert all(n == 23 for n in rep["n_months_by_well"].values())


def test_backcast_registration_build_and_score_end_to_end(tmp_path, monkeypatch):
    from harness import prereg3

    ids, pdir = _setup_avail(tmp_path, monkeypatch)
    rows = tmp_path / "rows"
    AV.count(tmp_path / K.BC_DIR / "avail", prereg_dir=pdir)
    s0 = tmp_path / SB.S0
    s0.parent.mkdir(parents=True)
    s0.write_text(json.dumps({"passed": True, "max_abs_diff_m": 0.0,
                              "swap_gate": {"passed": True}}))
    d = PB.doc()
    assert "conservativ" not in d["interpretation"]
    assert d["pregates"]["s0"]["swap_gate"] == {"passed": True}
    assert d["pregates"]["availability"]["passed"] and d["pregates"]["s0"]["passed"]
    assert prereg3.validate_prereg(d, units_registry=None, prereg_dir=pdir,
                                   registries=[]) == []
    d["inputs"] = [x for x in d["inputs"] if isinstance(x, dict)]
    _, h = prereg3.write_prereg(d, prereg_dir=pdir, rows_dir=rows, units_registry=None,
                                registries=[])
    reg_ids, sha = BLD.registered_ids(pdir, rows)
    assert sha == h and "07000000" not in reg_ids
    with pytest.raises(SystemExit, match="product_build look"):
        BLD.build(pdir, rows)
    BLD.log_look(pdir, rows)
    with pytest.raises(SystemExit, match="already logged"):
        BLD.log_look(pdir, rows)
    assert BLD.log_look(pdir, rows, skip_if_logged=True) is None

    def crash(sid):
        raise OSError("disk")
    with pytest.raises(OSError):
        BLD.build(pdir, rows, loader=crash)                   # an engineering retry
    assert not (tmp_path / K.L1PRE_V1).exists()
    d_built = BLD.build(pdir, rows)
    assert d_built == tmp_path / K.L1PRE_V1 and (d_built / "MANIFEST.json").exists()
    assert not (d_built.parent / ".partial").exists()
    with pytest.raises(SystemExit, match="exists"):
        BLD.build(pdir, rows)
    P, obs = bc_panel(W=len(reg_ids))
    P["sid"] = np.array(reg_ids)
    bl = L.well_blocks(P["sid"], P["x"], P["y"])
    P["block"] = np.array([bl[s] for s in P["sid"]])
    hcd = tmp_path / "hc"
    hcd.mkdir()
    pz = hcd / "bc_panel.npz"
    np.savez(pz, **P)
    (hcd / "bc_panel.json").write_text(json.dumps({
        "panel_sha256": IO.sha256(pz), "code_head": K.WT_HEAD,
        "theta_sha256": d["recipe"]["theta_sha256"]}))
    monkeypatch.setattr(IO, "l1_monthly", lambda sids, months, **kw: (
        obs, np.zeros(obs.shape, int), ["synthetic"]))
    stub = lambda P: {"computed": False, "reason": "test"}  # noqa: E731
    rep = SB.run_score(SimpleNamespace(panel=pz), prereg_dir=pdir, rows_dir=rows,
                       power_fn=stub)
    assert rep["verdict"] == "direction_supported"
    assert rep["report_only_synthetic_power"]["reason"] == "test"
    assert any(lk["purpose"] == "product_build" for lk in rep["post_registration_looks"])
    with pytest.raises(SystemExit, match="scored once"):
        SB.run_score(SimpleNamespace(panel=pz), prereg_dir=pdir, rows_dir=rows,
                     power_fn=stub)


# ----------------------------------------------------------------------------- chx panel
class FakeRun:
    lookup: dict = {}

    def __init__(self, device="cpu"):
        rng = np.random.default_rng(0)
        self.inp = SimpleNamespace(dates=pd.date_range("2012-01-01", periods=132, freq="MS"))
        self.E = rng.uniform(1.0, 2.0, (700, 132))
        self.changhua = np.arange(700) < 400

    def yunlin_cells(self):
        return (np.arange(700) >= 400) & (np.arange(700) < 600)     # 600-699: other

    def heads(self, E=None):
        E = self.E if E is None else E
        h = -1e-3 * np.cumsum(E, axis=1)
        return np.stack([h, 2 * h, 3 * h, 4 * h])

    def grid_cells(self, x, y):
        return np.array([self.lookup[(float(a), float(b))] for a, b in zip(x, y, strict=True)])


def test_chx_panel_assembly_with_a_fake_model(tmp_path, monkeypatch):
    t = fake_wells(tmp_path / K.EXT_WELLS)
    monkeypatch.setattr(IO, "REPO", tmp_path)
    FakeRun.lookup = {(float(r.x), float(r.y)): int(r.cell) for r in t.itertuples()}
    monkeypatch.setattr(HC, "R1cRun", FakeRun)
    monkeypatch.setattr(HC, "theta_sha", lambda: "t" * 64)
    s = HC.chx(tmp_path / "hc", "cpu", check_twinmodel=False)
    z = np.load(tmp_path / "hc" / "chx_panel.npz")
    assert z["c_ch"].shape == (23, 120) and s["checks"]["cells_match_export_changhua"]
    run = FakeRun()
    Hb = run.heads(L.pattern_energy(run.E, L.month_list(*K.RUN_MONTHS), run.changhua, 2012))
    i = 0
    assert np.allclose(z["B_ch"][i], Hb[z["layer_ch"][i], z["cell_ch"][i], 12:])
    yun = run.yunlin_cells()
    Hc = run.heads(L.pattern_energy(run.E, L.month_list(*K.RUN_MONTHS), yun, 2012))
    assert np.allclose(z["C_ch"][i], Hc[z["layer_ch"][i], z["cell_ch"][i], 12:])
    assert z["yunlin_cells"].sum() == 200 and s["checks"]["n_cells"]["other"] == 100
    assert set(z["block_ch"][np.isin(z["sid_ch"], K.NEST)]) == {K.NEST_BLOCK}
    assert s["rows_sha256_changhua"] == IO.rows_sha(IO.changhua_ext())
    with pytest.raises(SystemExit, match="exists"):
        HC.chx(tmp_path / "hc", "cpu", check_twinmodel=False)


# ----------------------------------------------------------------------------- fake model
class FakeModel:
    """A linear stand-in for R1cRun: heads respond to cumulative energy and recharge."""

    def __init__(self, months, A=50):
        rng = np.random.default_rng(1)
        self.months = months
        self.E = rng.uniform(1.0, 2.0, (A, len(months)))
        self.recharge = rng.uniform(0.0, 1e-3, (A, len(months)))
        self.changhua = np.arange(A) < 30
        self._sw = self._extra = None
        self.calls = []

    def run_from(self, E, R, h0=None, month0=None, sw=None, q_extra=None):
        self.calls.append(np.asarray(E))
        h = -1e-2 * np.cumsum(E, axis=1) + 10.0 * np.cumsum(R, axis=1)
        return np.stack([h, 2 * h, 3 * h, 4 * h])

    def heads(self, E=None):
        return self.run_from(self.E if E is None else E, self.recharge, month0=1)

    def grid_cells(self, x, y):
        return np.array([int(a) % 50 for a in x])


def test_s0_identity_and_swap_gate_with_a_fake_model(tmp_path, monkeypatch):
    run = L.month_list(*K.RUN_MONTHS)
    fm = FakeModel(run)
    monkeypatch.setattr(HC, "theta_sha", lambda: "t" * 64)
    monkeypatch.setattr(HC, "census_v3_energy", lambda m, months: (1.05 * m.E, {}))
    r = HC.s0(tmp_path / "a", "cpu", model=fm)
    assert r["identity_passed"] and r["swap_gate"]["passed"] and r["passed"]
    assert r["swap_gate"]["energy_ratio_changhua_2012_22"] == pytest.approx(1.05)
    assert r["swap_gate"]["contrast_corr"] == pytest.approx(1.0)
    monkeypatch.setattr(HC, "census_v3_energy", lambda m, months: (1.3 * m.E, {}))
    r2 = HC.s0(tmp_path / "b", "cpu", model=FakeModel(run))
    assert r2["identity_passed"] and not r2["swap_gate"]["energy_ratio_ok"]
    assert not r2["passed"]
    with pytest.raises(SystemExit, match="exists"):
        HC.s0(tmp_path / "a", "cpu", model=fm)


def test_backcast_panel_with_a_fake_model(tmp_path, monkeypatch):
    bc_run = L.month_list(*K.BC_RUN)
    fm = FakeModel(L.month_list(*K.RUN_MONTHS))
    E3 = np.random.default_rng(5).uniform(1.0, 2.0, (50, len(bc_run)))
    monkeypatch.setattr(HC, "census_v3_energy", lambda m, months: (E3, {}))
    monkeypatch.setattr(HC, "theta_sha", lambda: "t" * 64)
    st = pd.DataFrame({"sid": ["07000003", "07000001", "07000002", "09000001"],
                       "x": [3.0, 1.0, 2.0, 4.0], "y": [0.0] * 4,
                       "layer_code": ["2", "1", "7", "1"],
                       "county": ["changhua", "changhua", "changhua", "yunlin"]})
    sp = tmp_path / "station_meta.parquet"
    st.to_parquet(sp)
    s = HC.backcast(sp, tmp_path / "bc", "cpu", model=fm)
    z = np.load(tmp_path / "bc" / "bc_panel.npz")
    assert list(z["sid"]) == ["07000001", "07000003"] and list(z["not_placed"]) == ["07000002"]
    assert list(z["layer"]) == [0, 1] and s["n_placed"] == 2
    assert z["c"].shape == (2, 24) and list(z["months"]) == BCW
    EB = fm.calls[1]
    assert np.array_equal(EB[:, :12], E3[:, :12])                          # 2007 unchanged
    assert np.array_equal(EB[~fm.changhua], E3[~fm.changhua])
    jan08 = bc_run.index("2008-01")
    assert np.array_equal(EB[fm.changhua, jan08], E3[fm.changhua, 0])     # 2007-01 pattern
    clim = HC.calendar_climatology(fm.recharge, L.month_list(*K.RUN_MONTHS), bc_run)
    H = fm.run_from(E3, clim)
    assert np.allclose(z["A"][1], H[1, 3, bc_run.index("2010-01"):])
