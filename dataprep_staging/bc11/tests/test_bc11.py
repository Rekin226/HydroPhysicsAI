"""CPU tests of I1.backcast_2010_11.v1 (round 8 plan #12) on SYNTHETIC arrays and files only:
no ls_cache record, no head value and no model run is touched. Registries, looks, products,
panels and reports live in pytest's tmp dirs (``chx_io.REPO`` is pointed at the tmp dir);
the one test that reads the real registry COPIES its preregistration directory into a tmp
dir (looks.jsonl / INDEX.jsonl metadata only) and skips cleanly when results/ is absent.

    cd <repository root>
    CUDA_VISIBLE_DEVICES="" PYTHONPATH=dataprep_staging ~/miniforge3/envs/hydro/bin/python \
        -m pytest -q -p no:cacheprovider dataprep_staging/bc11/tests
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(REPO / "dataprep_staging"), str(REPO / "dataprep_staging" / "ch_x"), str(X),
          str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)

import avail_bc11 as A  # noqa: E402
import bc11_const as K  # noqa: E402
import bc11_lib as BL  # noqa: E402
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402

RNG = np.random.default_rng(12345)


# ----------------------------------------------------------------------------- ids
def test_fresh_units_are_new_ids_never_looked_at_under_these_names():
    others = {K.AVAIL_UNIT, K.PRODUCT_UNIT, K.E26_UNIT, *K.I1CHB_UNITS, *K.X5EV_UNITS}
    assert not set(K.UNITS) & others
    assert all(not u.startswith("fresh.") for u in K.UNITS)
    assert K.PRODUCT_UNIT not in K.UNITS and K.PRODUCT_UNIT != K.AVAIL_UNIT
    assert K.ID == A.BC11_ID and K.AVAIL_UNIT == A.UNIT
    assert K.SCOPE == "heads:2010-01..2011-12"
    assert len(set(K.TESTS)) == 4 and len(set(K.G0_SEEDS.values())) == 4


def test_axes():
    assert BL.WIN_M[0] == "2010-01" and BL.WIN_M[11] == K.ORIGIN and BL.WIN_M[-1] == "2011-12"
    assert BL.FIT_M[0] == "2012-01" and len(BL.FIT_M) == 132 and BL.RUN_M[0] == "2007-01"


# ----------------------------------------------------------------------------- boundaries
@pytest.mark.parametrize("r, ub, ok", [(1.0, 0.999, True), (1.0 + 1e-12, 0.9, False),
                                       (0.9, 1.0, False), (0.9, 1.0 - 1e-12, True),
                                       (float("nan"), 0.5, False), (0.5, float("inf"), False)])
def test_ratio_gate_edges(r, ub, ok):
    assert BL.ratio_gate(r, ub) is ok


def test_g1_needs_both_baselines():
    assert BL.g1_gate(0.9, 0.95, 0.9, 0.95)
    assert not BL.g1_gate(0.9, 0.95, 1.01, 0.95)
    assert not BL.g1_gate(0.9, 0.95, 0.9, 1.0)
    assert not BL.g1_gate(1.0, 1.0, 0.9, 0.95)


@pytest.mark.parametrize("c, ok", [(0.85, True), (0.95, True), (0.9, True),
                                   (0.85 - 1e-12, False), (0.95 + 1e-12, False),
                                   (float("nan"), False)])
def test_coverage_edges(c, ok):
    assert BL.coverage_ok(c) is ok


@pytest.mark.parametrize("p, f, ok", [(0.6, 0.2, True), (0.6 - 1e-12, 0.0, False),
                                      (1.0, 0.2 + 1e-12, False), (None, 0.0, False)])
def test_g0_gate_edges(p, f, ok):
    assert BL.g0_gate(p, f)["decidable"] is ok
    assert not BL.g0_gate(1.0, 0.0, informative=False)["decidable"]


def _fair_arrays(scale: float):
    """One well: CL = 0, obs 2011 alternating +-1, model error +-scale (ratio = scale)."""
    of = np.zeros((1, 132))
    mf = np.zeros((1, 132))
    alt = np.array([1.0, -1.0] * 6)
    os_ = alt[None, :]
    ms = (alt + scale * alt)[None, :]
    return mf, of, ms, os_


def test_fair_cl_ratio_edge_125():
    r = BL.fair_cl(*_fair_arrays(1.25), shape=False)
    assert r["rmse_ratio_fair_cl"] == 1.25 and r["passed"]
    r = BL.fair_cl(*_fair_arrays(1.25 + 1e-9), shape=False)
    assert not r["passed"]
    # E07's shape clause bites at this ratio (r2 -0.5625 against 0 - 0.05)
    assert not BL.fair_cl(*_fair_arrays(1.25))["passed"]
    assert BL.fair_cl(*_fair_arrays(0.2))["passed"]


def test_fair_cl_equals_e07_fair_rule_fields():
    from hydrophysics.twin.drift_diag import fair_temporal_verdict

    W = 7
    t = np.arange(144)
    seas = np.sin(2 * np.pi * t / 12)
    obs = 10 + RNG.normal(0, 2, (W, 1)) + seas + 0.3 * RNG.standard_normal((W, 144))
    obs[RNG.random((W, 144)) < 0.1] = np.nan
    obs[0, 132:] = np.nan                                   # no held-out month: not scored
    pred = obs + RNG.normal(0, 1, (W, 1)) + 0.2 * RNG.standard_normal((W, 144))
    pred = np.where(np.isfinite(pred), pred, 5.0)
    ref = fair_temporal_verdict(pred, obs, 132, month_offset=0)
    mine = BL.fair_cl(pred[:, :132], obs[:, :132], pred[:, 132:], obs[:, 132:])
    for a, b in (("rmse_datum_model_m", "rmse_datum_model_m"), ("rmse_clim_m", "rmse_clim_m"),
                 ("r2_shape_datum_model", "r2_shape_datum_model"),
                 ("r2_shape_clim", "r2_shape_clim"), ("n_wells_scored", "n_wells_scored")):
        assert np.isclose(mine[a], ref[b], rtol=0, atol=1e-12), a


def test_equal_station_ratio_is_e02_arithmetic():
    OF = pytest.importorskip("osse_fv")
    W, T = 9, 12
    obs = RNG.standard_normal((W, T))
    pred = obs + 0.5 * RNG.standard_normal((W, T))
    base = obs + RNG.standard_normal((W, T))
    m = RNG.random((W, T)) > 0.2
    rows = [(f"w{w}", t, pred[w, t], base[w, t], obs[w, t]) for w in range(W) for t in range(T)
            if m[w, t]]
    f = pd.DataFrame(rows, columns=["station_id", "date", "predicted_m", "baseline_m",
                                    "observed_m"]).assign(hit=1.0)
    g = OF.g0_e02(f, {"min_stations": 1, "min_months": 1, "coverage_bounds": [0, 1],
                     "max_rmse_ratio": 1.0})
    mine = BL.equal_station_ratio(BL.station_msq(pred, obs, m), BL.station_msq(base, obs, m))
    assert np.isclose(mine, g["ratio"], rtol=0, atol=1e-12)


def test_bootstrap_is_seeded_and_shared():
    a, b = BL.boot_index(10, 50, 7), BL.boot_index(10, 50, 7)
    assert np.array_equal(a, b) and not np.array_equal(a, BL.boot_index(10, 50, 8))
    msq = np.ones(10)
    assert BL.boot_ub(msq * 0.25, msq, a) == 0.5


# ----------------------------------------------------------------------------- E04 recipe
def _fit_world(W=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(132)
    seas = 2 * np.sin(2 * np.pi * t / 12)
    a = np.zeros((W, 132))
    for k in range(1, 132):
        a[:, k] = 0.8 * a[:, k - 1] + rng.standard_normal(W)
    return 20 + seas + a


def test_fit_stat_is_hybrid_register_arithmetic():
    from hydrophysics.twin.coverage import _residual_sd
    from hydrophysics.twin.hybrid import lag1_rho

    obs = _fit_world()
    f = BL.fit_stat(obs)
    cl = np.stack([obs[:, k::12].mean(axis=1) for k in range(12)], axis=1)
    assert np.allclose(f["cl_month"], cl)
    rho, _ = lag1_rho(obs - cl[:, BL.moy(BL.FIT_M)], 132)
    assert np.allclose(f["rho"], rho) and np.allclose(f["rho_used"], rho)
    assert np.allclose(f["sd"], _residual_sd(obs, cl[:, BL.moy(BL.FIT_M)], 132))


def test_stat_forecast_persists_the_origin_anomaly_and_never_reads_2011():
    obs_fit = _fit_world()
    fit = BL.fit_stat(obs_fit)
    CL = BL.cl_on(fit["cl_month"], BL.WIN_M)
    win = CL + 1.0
    win[0, 10:12] = np.nan                       # well 0: last 2010 month is October
    win[1, 8:12] = np.nan                        # well 1: last is August (gap 4 > 3)
    fc = BL.stat_forecast(fit, win)
    k = np.arange(1, 13)
    r = fit["rho_used"]
    assert np.allclose(fc["pred"][2, 12:], CL[2, 12:] + r[2] ** k)
    assert np.allclose(fc["pred"][0, 12:], CL[0, 12:] + r[0] ** (k + 2))
    assert fc["last"][0] == 9 and fc["last"][1] == -1 and not fc["keep"][1]
    assert np.allclose(fc["cl_adj"][2], CL[2] + 1.0) and np.allclose(fc["persist"][2, 12:],
                                                                      win[2, 11])
    w2 = win.copy()
    w2[:, 12:] = 999.0
    fc2 = BL.stat_forecast(fit, w2)
    for key in ("pred", "lower", "upper", "cl_adj", "keep"):
        assert np.array_equal(fc[key], fc2[key], equal_nan=True), key


def _fc(W, err_pred, err_cl, err_adj, hit_frac=0.9):
    obs = np.zeros((W, 24))
    pred = np.zeros((W, 24))
    pred[:, 12:] = err_pred
    sd = np.ones(W)
    fc = {"pred": pred, "cl": np.full((W, 24), err_cl), "cl_adj": np.full((W, 24), err_adj),
          "persist": np.zeros((W, 24)), "keep": np.ones(W, bool),
          "lower": np.full((W, 24), -1.0), "upper": np.full((W, 24), 1.0)}
    n_miss = int(round((1 - hit_frac) * W * 12))
    flat = np.zeros(W * 12, bool)
    flat[:n_miss] = True
    miss = flat.reshape(W, 12)
    fc["upper"][:, 12:][miss] = -0.5                   # those cells fall outside
    return fc, obs, sd


def test_g1_score_arithmetic_and_wells_floor():
    county = np.array(["yunlin"] * 15 + ["changhua"] * 10)
    fc, obs, _ = _fc(25, 0.5, 1.0, 2.0)
    r = BL.g1_score(fc, obs, county, B=200)
    assert r["verdict"] == "pass" and np.isclose(r["ratio_vs_cl"], 0.5)
    assert np.isclose(r["ratio_vs_cl_level_adjusted"], 0.25) and np.isclose(r["coverage90"], 0.9)
    assert r["county_lines"]["changhua"]["g1_passed"]           # misses all sit in Yunlin rows
    assert r["county_lines"]["changhua"]["coverage90"] == 1.0
    fc, obs, _ = _fc(25, 0.5, 1.0, 0.5)                    # ties the adjusted CL: ub = 1.0
    r = BL.g1_score(fc, obs, county, B=200)
    assert r["ratio_vs_cl_level_adjusted"] == 1.0 and not r["g1_passed"]
    fc, obs, _ = _fc(25, 0.5, 1.0, 2.0, hit_frac=0.8)
    assert not BL.g1_score(fc, obs, county, B=200)["g1b_passed"]
    fc, obs, _ = _fc(24, 0.5, 1.0, 2.0)
    assert BL.g1_score(fc, obs, county[:24], B=200)["verdict"] == "not_evaluable"
    fc, obs, _ = _fc(25, 0.5, 1.0, 2.0)
    obs[0, 12:19] = np.nan                                 # 5 observed 2011 months: dropped
    r = BL.g1_score(fc, obs, county, B=200)
    assert r["verdict"] == "not_evaluable" and r["dropped_well_index"] == [0]
    county9 = np.array(["yunlin"] * 16 + ["changhua"] * 9)
    fc, obs, _ = _fc(25, 0.5, 1.0, 2.0)
    assert BL.g1_score(fc, obs, county9, B=200)["county_lines"]["changhua"]["verdict"] == \
        "not_evaluable"


# ----------------------------------------------------------------------------- G0
def test_g0_g1_pipeline_on_tiny_worlds_is_seeded():
    county = np.array(["yunlin"] * 14 + ["changhua"] * 12)
    mask = np.ones((26, 24), bool)
    mask[:, :6] = False
    a = BL.run_g0_g1(county, mask, reps=2, seed=99, B=50)
    b = BL.run_g0_g1(county, mask, reps=2, seed=99, B=50)
    assert a == b
    assert [c["kind"] for c in a["cells"]].count("ar") == 9
    assert [c["kind"] for c in a["cells"]].count("null") == 3
    assert set(a["gate"]) >= {"power", "false_pass", "decidable"}
    assert all(0.0 <= c["p_pass"] <= 1.0 for c in a["cells"])


def _phys(W=24, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(132)
    mf = 15 + rng.normal(0, 3, (W, 1)) + np.sin(2 * np.pi * t / 12)[None] \
        + 0.01 * t[None] * rng.uniform(0.5, 1.5, (W, 1))
    F1 = mf[:, :12] + rng.normal(0, 0.5, (W, 12)) - 3.0
    F0 = F1 + rng.normal(0, 0.8, (W, 12))
    county = np.array(["yunlin"] * (W // 2) + ["changhua"] * (W - W // 2))
    return mf, F1, F0, county


def test_g0_g2_and_g3_pipelines_on_tiny_worlds():
    mf, F1, F0, county = _phys(W=30)
    sig = np.full(len(county), 0.3)
    mask = np.ones(F1.shape, bool)
    r2 = BL.run_g0_g2(mf, F1, sig, county, mask, reps=2, seed=5)
    assert set(r2["gate"]) == set(K.COUNTIES) and len(r2["cells"]) == 19      # + 1 RO cell
    r3 = BL.run_g0_g3(mf, F1, F0, sig, county, mask, reps=2, seed=6, B=50)
    assert len(r3["cells"]) == 18 and "decidable" in r3["gate"]
    assert r3 == BL.run_g0_g3(mf, F1, F0, sig, county, mask, reps=2, seed=6, B=50)


def test_g3_is_value_shown_only_when_f1_beats_f0():
    mf, F1, F0, _ = _phys(W=30)
    d = np.zeros(30)
    assert BL.g3_score(F1, F0, d, F1 + 0.01, B=100)["passed"]
    assert not BL.g3_score(F1, F0, d, F0 + 0.01, B=100)["passed"]
    assert BL.g3_score(F1[:24], F0[:24], d[:24], F1[:24], B=100)["verdict"] == "not_evaluable"


def yr_panel(W=14, k=1.0, seed=0):
    rng = np.random.default_rng(seed)
    T = 24
    sid = np.array([f"09{i:06d}" for i in range(W)])
    x = 175000.0 + 6000 * (np.arange(W) % 4)
    y = 2600000.0 + 6000 * (np.arange(W) // 4)
    bl = L.well_blocks(sid, x, y)
    a = 0.6 * rng.standard_normal(T)
    c = -rng.uniform(0.5, 2.5, W)[:, None] * a[None, :]
    B = 20.0 + rng.normal(0, 2, (W, 1)) + np.tile(np.sin(np.arange(T) / 2.0), (W, 1))
    P = {"months": np.array(BL.WIN_M), "sid_yl": sid, "c_yl": c, "B_yl": B, "A_yl": B + c,
         "block_yl": np.array([bl[s] for s in sid]), "a_yl": a}
    obs = B + k * c + 0.05 * rng.standard_normal((W, T)) \
        + 0.5 * np.sin(2 * np.pi * np.arange(T) / 12)[None]
    return P, obs


def test_yunlin_arm_direction_and_floor():
    P, obs = yr_panel(k=1.0)
    r = BL.score_yr(P, obs, None, B=200)
    assert r["verdict"] == "direction_supported" and r["form"] == K.YR_FORM
    P, obs = yr_panel(k=-1.0)
    assert BL.score_yr(P, obs, None, B=200)["verdict"] == "not_supported"
    P, obs = yr_panel(W=9)
    assert BL.score_yr(P, obs, None, B=200)["verdict"] == "not_evaluable"
    P, obs = yr_panel()
    obs[:, :7] = np.nan                                    # 17 months each: none eligible
    assert BL.score_yr(P, obs, None, B=200)["verdict"] == "not_evaluable"


def test_g0_yr_plumbing_and_cells():
    P, _ = yr_panel()
    par = {"sigma_tot_m": 0.5, "block_share": 0.4, "phi": 0.86, "kappa_m": 0.2}
    r = BL.run_g0_yr(P, np.ones((14, 24), bool), par, reps=2, seed=4, B=50)
    assert r["plumbing"]["ok"] and len(r["cells"]) == 5
    assert sum(c["report_only"] for c in r["cells"]) == 1
    assert "decidable" in r["gate"]
    Pz = dict(P, c_yl=np.zeros_like(P["c_yl"]))
    rz = BL.run_g0_yr(Pz, np.ones((14, 24), bool), par, reps=2, seed=4, B=50)
    assert not rz["gate"]["decidable"] and rz["cells"] == []


def test_swap_gate_county_matches_e26_rule():
    HC = pytest.importorskip("hindcast")
    cn = RNG.standard_normal((4, 30, 120))
    c3 = 1.05 * cn + 0.05 * RNG.standard_normal(cn.shape)
    mine = BL.swap_gate_county(cn, c3, 100.0, 104.0)
    ref = HC.swap_gate(cn, c3, 100.0, 104.0)
    assert mine["passed"] == ref["passed"] is True
    assert np.isclose(mine["contrast_corr"], ref["contrast_corr"])
    assert not BL.swap_gate_county(cn, c3, 100.0, 116.0)["passed"]
    assert BL.swap_gate_county(cn, c3, 100.0, 115.0)["energy_ratio_ok"]


# ----------------------------------------------------------------------------- page runner
class FakePage:
    def __init__(self, A=60, T=132, seed=0):
        rng = np.random.default_rng(seed)
        self.inp = SimpleNamespace(dates=pd.date_range("2012-01-01", periods=T, freq="MS"),
                                   grid=None)
        self.E = rng.uniform(1.0, 2.0, (A, T))
        self.recharge = rng.uniform(0.0, 1.0, (A, T))
        self.A = A

    def ensemble(self, E, R, month0=None):
        h = 10.0 - np.cumsum(np.asarray(E, float) - np.asarray(R, float), axis=-1) * 0.01
        return np.stack([h, h + 1, h + 2, h + 3])

    def native(self):
        return self.ensemble(self.E, self.recharge)

    def county_cells(self, county):
        return (np.arange(self.A) < 30) if county == "yunlin" else (np.arange(self.A) >= 30)

    def grid_cells(self, x, y):
        return (np.asarray(x, float) // 1000).astype(int) % self.A


def test_e0_compare():
    import page_run as PR

    m = FakePage()
    nat = m.native()
    ref = {"scenario_names": np.array(["baseline", "cut30"]),
           "dates": np.array([str(d) for d in pd.date_range("2012-01-01", periods=252,
                                                             freq="MS")]),
           "heads_mean": np.stack([np.concatenate([nat, nat[..., :120]], axis=-1)] * 2)}
    assert PR.e0_compare(nat, ref)["passed"]
    ref["heads_mean"] = ref["heads_mean"] + 0.0101
    assert not PR.e0_compare(nat, ref)["passed"]
    ref["scenario_names"] = np.array(["cut30", "baseline"])
    with pytest.raises(SystemExit, match="axis or scenario"):
        PR.e0_compare(nat, ref)


def test_backcast_panel_with_a_fake_model(tmp_path, monkeypatch):
    import page_run as PR

    monkeypatch.setattr(IO, "REPO", tmp_path)
    m = FakePage(T=132)
    E3 = np.random.default_rng(1).uniform(1.0, 2.0, (60, len(BL.RUN_M)))
    monkeypatch.setattr(PR.HC, "census_v3_energy", lambda mm, months: (E3, {"fake": True}))
    monkeypatch.setattr(PR.HC, "code_head", lambda: K.WT_HEAD)
    page = np.random.default_rng(2).normal(10, 1, (3, 4, 60, 252))
    monkeypatch.setattr(PR, "page_forward", lambda keys=None: {"heads_mean": page})
    st = pd.DataFrame({"sid": [f"09{i:06d}" for i in range(12)] + ["07000001", "07000002"],
                       "x": [1000.0 * i for i in range(14)], "y": 2600000.0,
                       "layer_code": ["1", "2"] * 7,
                       "county": ["yunlin"] * 12 + ["changhua"] * 2})
    sp = tmp_path / "station_meta.parquet"
    pytest.importorskip("pyarrow")
    st.to_parquet(sp)
    summ = PR.backcast(sp, tmp_path / "hc", "cpu", model=m)
    z = np.load(tmp_path / "hc" / Path(K.PANEL).name)
    assert summ["n_placed"] == 14 and list(z["months"]) == BL.WIN_M
    assert z["F1"].shape == (14, 24) and z["c_yl"].shape == (12, 24)
    assert np.allclose(z["c_yl"], z["A_yl"] - z["B_yl"])
    lay0, cel0 = int(z["layer"][0]), int(z["cell"][0])
    assert np.allclose(z["native_fit"][0], page[0, lay0, cel0, :132])
    # F0 is F1's pumping climatology: identical calendar months in every year of the run
    with pytest.raises(SystemExit, match="exists"):
        PR.backcast(sp, tmp_path / "hc", "cpu", model=m)


# ----------------------------------------------------------------------------- end to end
def ts_series(seed=0, level=5.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", "2011-12-31 23:00", freq="h")
    v = level + 0.3 * np.sin(2 * np.pi * np.arange(len(idx)) / (24 * 365)) \
        + 0.01 * rng.standard_normal(len(idx))
    return pd.Series(v, index=idx)


N_YL, N_CH = 14, 12


def ids_and_counties():
    ids = [f"09{i:06d}" for i in range(N_YL)] + [f"07{i:06d}" for i in range(N_CH)]
    return ids, {s: ("yunlin" if s.startswith("09") else "changhua") for s in ids}


def setup_world(tmp_path, monkeypatch):
    """A tmp repository: station table, ls_cache files, the availability look (logged in the
    tmp registry) and its avail.json."""
    from harness import prereg3

    pytest.importorskip("pyarrow")
    monkeypatch.setattr(IO, "REPO", tmp_path)
    ids, county = ids_and_counties()
    zh = {"yunlin": "雲林縣", "changhua": "彰化縣"}
    st = pd.DataFrame({"sid": ids, "CountyName": [zh[county[s]] for s in ids],
                       "LocationByTWD97": [f"{170000 + 6000 * (i % 5)} "
                                           f"{2610000 + 6000 * (i // 5)}"
                                           for i in range(len(ids))],
                       "GroundwaterLayerCode": ["1", "2"] * (len(ids) // 2),
                       "GroundwaterZoneIdentifier": [50] * len(ids),
                       "WellElevation": [10.0] * len(ids), "WellDepth": [90.0] * len(ids)})
    (tmp_path / K.STATIONS).parent.mkdir(parents=True, exist_ok=True)
    st.to_parquet(tmp_path / K.STATIONS)
    lc = tmp_path / "chou-shui-data/data/ls_cache"
    lc.mkdir(parents=True)
    for i, s in enumerate(ids):
        ser = ts_series(i)
        pd.DataFrame({"value": ser.to_numpy()}, index=pd.Index(ser.index, name="datetime")
                     ).to_parquet(lc / f"gw__{s}__raw.parquet")
    pdir = tmp_path / "prereg"
    rows = tmp_path / "rows"
    lk = prereg3.record_look(
        [K.AVAIL_UNIT], A.SCOPE, K.WP, "availability_check",
        members={K.AVAIL_UNIT: {"ids": ids, "window": "2009-10..2011-12",
                                "description": "synthetic"}},
        note=f"[{A.AVAIL_ID}] {A.AL.sha_tag(A.constants())}; {A.RULE_TEXT}",
        prereg_dir=pdir, rows_dir=rows, registries=[])
    months = {s: BL.WIN_M for s in ids}
    feas = A.stop_rule(months, county, set())
    p = tmp_path / K.AVAIL_OUT
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"look_time": lk["time"], "months_by_well": months,
                             "feasibility": feas}))
    return ids, county, pdir, rows


def obs_world(ids, county, seed=0):
    """Synthetic 2012-22 and 2010-11 heads (never real)."""
    rng = np.random.default_rng(seed)
    W = len(ids)
    t = np.arange(156)
    a = np.zeros((W, 156))
    for k in range(1, 156):
        a[:, k] = 0.9 * a[:, k - 1] + 0.3 * rng.standard_normal(W)
    full = 20 + rng.normal(0, 3, (W, 1)) + np.sin(2 * np.pi * t / 12)[None] + a
    return full[:, 24:], full[:, :24]


def test_end_to_end_order_refusals_and_score_once(tmp_path, monkeypatch):
    import baselines_bc11 as BAS
    import build_l1pre_v2 as BV
    import prereg_bc11 as PB
    import score_bc11 as SC
    from harness import prereg3

    ids, county, pdir, rows = setup_world(tmp_path, monkeypatch)
    # 1. the product: refused before its own look; the look is on the build unit only
    with pytest.raises(SystemExit, match="product_build look first"):
        BV.build(pdir)
    lk = BV.log_look(pdir, rows, registries=[])
    assert lk["units"] == [K.PRODUCT_UNIT] and not lk["members"][K.PRODUCT_UNIT].get(
        "universal")
    with pytest.raises(SystemExit, match="already logged"):
        BV.log_look(pdir, rows, registries=[])
    assert BV.log_look(pdir, rows, registries=[], skip_if_logged=True) is None

    def crash(sid):
        raise OSError("disk")
    with pytest.raises(OSError):
        BV.build(pdir, loader=crash)
    assert not (tmp_path / K.PRODUCT).exists()
    d_built = BV.build(pdir)
    man = json.loads((d_built / "MANIFEST.json").read_text())
    assert man["allowed_uses"] == list(K.PRODUCT_ALLOWED) and man["version"] == "v2"
    assert man["parameters"]["build_look"]["unit"] == K.PRODUCT_UNIT
    with pytest.raises(SystemExit, match="exists"):
        BV.build(pdir)

    # 2. model-only files (synthetic stand-ins for the lead's GPU runs)
    obs_fit, obs_win = obs_world(ids, county)
    W = len(ids)
    rng = np.random.default_rng(7)
    native = obs_fit + rng.normal(0, 0.5, (W, 1)) + 0.2 * rng.standard_normal((W, 132))
    F1 = obs_win + rng.normal(0, 0.5, (W, 1)) + 0.3 * rng.standard_normal((W, 24))
    yl = [i for i, s in enumerate(ids) if county[s] == "yunlin"]
    sid = np.array(ids)
    x = 170000.0 + 6000 * (np.arange(W) % 5)
    y = 2610000.0 + 6000 * (np.arange(W) // 5)
    blk = L.well_blocks(sid[yl], x[yl], y[yl])
    c_yl = -rng.uniform(0.5, 2.0, (len(yl), 1)) * rng.standard_normal(24)[None]
    panel = {"months": np.array(BL.WIN_M), "sid": sid,
             "county": np.array([county[s] for s in ids]), "x": x, "y": y,
             "layer": np.zeros(W, int), "cell": np.arange(W), "not_placed": np.array([], str),
             "native_fit": native, "F1": F1, "F0": F1 + 0.5 * rng.standard_normal((W, 24)),
             "sid_yl": sid[yl], "A_yl": F1[yl], "B_yl": F1[yl] - c_yl, "c_yl": c_yl,
             "block_yl": np.array([blk[s] for s in sid[yl]]),
             "a_yl": rng.standard_normal(24)}
    pz = tmp_path / K.PANEL
    pz.parent.mkdir(parents=True)
    np.savez(pz, **panel)
    import page_run as PR

    # the inputs a real draft hashes, as tmp stand-ins
    for path in (K.RULINGS, f"{K.CENSUS_V3}/MANIFEST.json", f"{K.CL_PRODUCT}/MANIFEST.json",
                 K.PAGE_FORWARD, *K.PAGE_THETAS):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("{}")
    shas = K.code_shas()
    (tmp_path / K.PANEL_JSON).write_text(json.dumps({
        "panel_sha256": IO.sha256(pz), "code_head": K.WT_HEAD,
        "thetas_sha256": PR.theta_shas(), "code_sha256": shas}))
    swap_c = {"yunlin": {"passed": True, "energy_ratio_ok": True},
              "changhua": {"passed": False, "energy_ratio_ok": True}}
    for path, obj in ((K.E0_OUT, {"passed": True, "max_abs_diff_m": 1e-6, "code_head":
                                  K.WT_HEAD, "code_sha256": shas}),
                      (K.SWAP_OUT, {"passed": False, "counties": swap_c,
                                    "code_head": K.WT_HEAD, "code_sha256": shas})):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(json.dumps(obj))
    l1_calls = []

    def fake_l1(sids, months, product, use, window_end):
        l1_calls.append((product, months[0], months[-1]))
        if product == K.CL_PRODUCT:
            assert months[-1] <= "2022-12" and window_end == "2022-12"
            return obs_fit[[ids.index(s) for s in sids]], None, ["fake 2012-22"]
        assert product == K.PRODUCT and use == "design_time_evidence"
        events.append("product_read")
        return obs_win[[ids.index(s) for s in sids]], np.zeros((len(sids), 24), int), ["fake"]
    events = []
    BAS.build(pz, tmp_path / Path(K.BASELINES).parent,
              loader=lambda s, m: fake_l1(s, m, K.CL_PRODUCT, K.CL_USE, K.CL_WINDOW_END))
    gates = {"g1": {"power": 0.7, "false_pass": 0.05, "decidable": True},
             "g2": {"yunlin": {"decidable": True}, "changhua": {"decidable": False}},
             "g3": {"decidable": True}, "yr": {"decidable": True}}
    for key, path in K.G0_FILES.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(json.dumps({
            "gate": gates[key], "reps": 1, "code_sha256": shas,
            "created": "2026-01-01T00:00:00+00:00" if key == "g1" else "2099-01-01"}))
    att = tmp_path / K.ATTEMPTS
    rows_att = [{"step": st, "rc": 0} for st in (*K.ATTEMPT_STEPS, f"e0_{K.WT_HEAD}")]
    rows_att.insert(0, {"step": "swap", "rc": 1, "note": "an engineering retry"})
    att.write_text("".join(json.dumps(r) + "\n" for r in rows_att))

    # 3. the draft: valid, not blocked by the availability or product_build looks
    d = PB.doc(prereg_dir=pdir, registries=[])
    assert d["roles"] == {"g1": "gating", "g2": {"yunlin": "gating", "changhua": "report_only"},
                          "yr": "gating", "g3": "report_only"}      # swap failed: G3 RO
    assert prereg3.validate_prereg(d, units_registry=None, prereg_dir=pdir,
                                   registries=[]) == []
    assert PB.blocking(d, prereg_dir=pdir, registries=[]) == []
    ov = d["prior_looks_disclosed"]["probe"][K.UNIT_YL]["overlapping_looks"]
    assert {r["unit"] for r in ov} == {K.AVAIL_UNIT, K.PRODUCT_UNIT}
    assert d["pregates"]["product"]["manifest_sha256"] != "missing"
    assert PB.readiness(d) == []                         # S2 bindings all hold
    assert d["attempts_disclosed"][0]["rc"] == 1          # the failed retry is disclosed
    assert d["recipe"]["worktree_head"] == K.WT_HEAD
    assert "spinup_discontinuity_f1_2011_12_vs_native_2012_01" in d["pregates"]["panel"]
    # S2 mutations: each one alone makes the draft not ready
    bad = json.loads((tmp_path / K.G0_FILES["g2"]).read_text())
    bad["code_sha256"] = {**shas, f"{K.X}/bc11_lib.py": "0" * 64}
    (tmp_path / K.G0_FILES["g2"]).write_text(json.dumps(bad))
    assert any("code shas differ" in x for x in PB.readiness(PB.doc(pdir, [])))
    bad["code_sha256"] = shas
    (tmp_path / K.G0_FILES["g2"]).write_text(json.dumps(bad))
    sw = json.loads((tmp_path / K.SWAP_OUT).read_text())
    (tmp_path / K.SWAP_OUT).write_text(json.dumps({**sw, "code_head": K.WT_HEAD + "-dirty"}))
    assert any("code_head" in x for x in PB.readiness(PB.doc(pdir, [])))
    (tmp_path / K.SWAP_OUT).write_text(json.dumps(sw))
    att.write_text(att.read_text() + json.dumps({"step": "hindcast", "rc": 0}) + "\n")
    assert any("hindcast has 2" in x for x in PB.readiness(PB.doc(pdir, [])))
    att.write_text("".join(json.dumps(r) + "\n" for r in rows_att))
    d = PB.doc(prereg_dir=pdir, registries=[])
    assert PB.readiness(d) == []
    # no 2010-11 head has been read by anything so far
    assert all(c[0] != K.PRODUCT for c in l1_calls)
    with pytest.raises(SystemExit):                    # not registered: refused, nothing read
        SC.run_score(prereg_dir=pdir, rows_dir=rows, l1=fake_l1)
    assert "product_read" not in events

    # 4. register in the tmp registry, then the product build refuses
    # write_prereg hashes plain-path inputs against the REAL repository root
    # (common.paths.REPO), not the tmp one: keep only the code entries in the tmp registry
    d["inputs"] = [x for x in d["inputs"] if isinstance(x, dict)]
    _, h = prereg3.write_prereg(d, prereg_dir=pdir, rows_dir=rows, units_registry=None,
                                registries=[])
    with pytest.raises(SystemExit, match="registered"):
        BV.log_look(pdir, rows, registries=[])

    # 5. score once; the scoring look precedes the product read
    real_acs = prereg3.assert_can_score

    def spy_acs(*a, **kw):
        events.append("score_look")
        return real_acs(*a, **kw)
    monkeypatch.setattr(prereg3, "assert_can_score", spy_acs)
    rep = SC.run_score(prereg_dir=pdir, rows_dir=rows, l1=fake_l1)
    assert events == ["score_look", "product_read"]
    assert rep["prereg_sha256"] == h and rep["G1"]["role"] == "gating"
    assert rep["G1"]["result"]["verdict"] in ("pass", "fail")
    assert rep["G2"]["changhua"]["role"] == "report_only"
    assert rep["G2"]["changhua"]["line"] == SC.WORDING["report_only"]
    assert rep["G3"]["role"] == "report_only"
    assert rep["YR"]["result"]["verdict"] in ("direction_supported", "not_supported",
                                              "not_informative")
    assert "f_l_memory_line" in rep["report_only"]
    assert (tmp_path / K.REPORT).exists() and (tmp_path / K.ATTEMPTS).exists()
    with pytest.raises(SystemExit, match="scored once"):
        SC.run_score(prereg_dir=pdir, rows_dir=rows, l1=fake_l1)


def test_scorer_refuses_changed_frozen_code(tmp_path, monkeypatch):
    import score_bc11 as SC
    from harness import prereg3

    monkeypatch.setattr(prereg3, "load_prereg", lambda *a, **k: (
        {"frozen_code": {p: "0" * 64 for p in SC.FROZEN}}, "s"))
    with pytest.raises(SystemExit, match="frozen code differs"):
        SC.verify_registration()


def test_scorer_refuses_drifted_pregate_files(tmp_path, monkeypatch):
    import score_bc11 as SC

    monkeypatch.setattr(IO, "REPO", tmp_path)
    doc = {"pregates": {k: {"sha256": "x", "path": p} for k, p in SC.PREGATES.items()},
           "g0": {k: {"sha256": "missing"} for k in K.G0_FILES}}
    doc["pregates"]["availability"]["passed"] = True
    doc["pregates"]["e0"]["passed"] = True
    with pytest.raises(SystemExit, match="differs from the registration"):
        SC.check_files(doc)


# ----------------------------------------------------------------------------- real registry
def test_new_units_not_blocked_by_the_logged_availability_looks(tmp_path):
    """prereg3's own blocking logic on a COPY of the real registry (metadata only): the new
    fresh units are not blocked by the logged availability look, X5EV or E26, nor by a
    simulated product_build look on its own unit; the same looks WOULD block a test reusing
    the availability unit, E26's unit or the product_build unit."""
    from harness import prereg3

    src = REPO / "results/dataprep/preregistration"
    avail = REPO / K.AVAIL_OUT
    if not (src / "looks.jsonl").exists() or not avail.exists():
        pytest.skip("results/ registry not present (CI)")
    import prereg_bc11 as PB

    pdir = tmp_path / "preregistration"
    shutil.copytree(src, pdir)
    w = PB.wells()
    ids = w["yunlin"] + w["changhua"]
    lk = _avail_look(pdir)
    assert lk is not None, "the availability look is logged on the real registry"
    prereg3.record_look([K.PRODUCT_UNIT], K.SCOPE, K.WP, K.PRODUCT_PURPOSE,
                        members={K.PRODUCT_UNIT: {"ids": ids, "window": K.WINDOW_TEXT,
                                                  "description": "simulated (tmp copy)"}},
                        note="simulated product_build look on a tmp copy",
                        prereg_dir=pdir, rows_dir=tmp_path / "rows")
    d = {"tests": PB.tests(w)}
    assert PB.blocking(d, prereg_dir=pdir) == []
    for reused in (K.AVAIL_UNIT, K.E26_UNIT, K.PRODUCT_UNIT):
        t = [{"test_id": "probe.reuse", "data_scope": K.SCOPE, "fresh_units": [reused]}]
        assert prereg3.blocking_looks(t, None, pdir, prereg3.membership_index(pdir),
                                      test_time=pd.Timestamp.now(tz="UTC").to_pydatetime()), \
            reused


def _avail_look(pdir):
    import avail_lib as AL

    return AL.logged_look(K.AVAIL_UNIT, pdir)


def test_g0_studies_read_no_head(tmp_path, monkeypatch):
    """The G0 entry point runs on the availability counts and a model panel only: every head
    reader refuses, and the result file is written once."""
    import build_l1pre as BLD
    import g0_bc11 as G

    monkeypatch.setattr(IO, "REPO", tmp_path)

    def refuse(*a, **k):
        raise AssertionError("a head reader was called by a G0 study")
    monkeypatch.setattr(IO, "l1_monthly", refuse)
    monkeypatch.setattr(BLD, "load_raw", refuse)
    ids, county = ids_and_counties()
    months = {s: BL.WIN_M for s in ids}
    p = tmp_path / K.AVAIL_OUT
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"months_by_well": months,
                             "feasibility": A.stop_rule(months, county, set())}))
    rep = G.study("g1", tmp_path / "g0", reps=1)
    assert rep["seed"] == K.G0_SEEDS["g1"] and rep["reps"] == 1
    with pytest.raises(FileExistsError):
        G.study("g1", tmp_path / "g0", reps=1)
    P, _ = yr_panel()
    pz = tmp_path / "panel.npz"
    np.savez(pz, sid=P["sid_yl"], **P)
    rep = G.study("yr", tmp_path / "g0", panel=pz, reps=1)
    assert "gate" in rep and rep["params_from"] == K.G0_PARAMS_FROM


# ----------------------------------------------------------------------------- review (S3)
def test_level_adjusted_cl_uses_the_mean_2010_anomaly_only():
    fit = BL.fit_stat(_fit_world())
    CL = BL.cl_on(fit["cl_month"], BL.WIN_M)
    win = np.array(CL)
    win[:, :12] += np.arange(1.0, 13.0)              # non-constant 2010 anomalies, mean 6.5
    win[:, 12:] += 100.0                             # 2011 never enters the level
    win[3, 0] = np.nan                               # observed months only: mean of 2..12 = 7
    fc = BL.stat_forecast(fit, win)
    assert np.allclose(fc["level"][:3], 6.5) and np.isclose(fc["level"][3], 7.0)
    assert np.allclose(fc["cl_adj"][:3], CL[:3] + 6.5)
    assert np.allclose(fc["persist"][0, 12:], win[0, 11])     # December's value, not the mean


def test_bootstrap_upper_bound_is_the_095_quantile():
    rng = np.random.default_rng(3)
    mp, mb = rng.uniform(0.2, 1.5, 30), rng.uniform(0.5, 1.2, 30)
    idx = BL.boot_index(30, 400, 11)
    r = np.sqrt(mp[idx].mean(axis=1)) / np.sqrt(mb[idx].mean(axis=1))
    ub = BL.boot_ub(mp, mb, idx)
    assert ub == float(np.quantile(r, 0.95)) and ub != float(np.quantile(r, 0.90))
    assert K.BOOT_Q == 0.95


def test_cell_gate_power_is_mean_of_power_cells_and_false_is_max_of_null_cells():
    rows = ([{"kind": "ar", "p": p} for p in (0.9, 0.3, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6)]
            + [{"kind": "null", "p": p} for p in (0.05, 0.19, 0.10)]
            + [{"kind": "report_only_offset", "p": 1.0}, {"kind": "report_only_offset",
                                                         "p": 0.0}])
    g = BL.cell_gate(rows, "p", "ar", "null")
    assert np.isclose(g["power"], 0.6) and g["false_pass"] == 0.19 and g["decidable"]


def test_run_g0_g1_gate_uses_the_9_ar_cells_and_the_max_null(monkeypatch):
    """Each world reveals its cell; only (rho 0.80, share 0) of the AR cells and (rho 0,
    share 0.30) of the null cells pass, and the report-only cells always pass."""
    def fake_world(county, rho, share, rng, T):
        return np.full((len(county), T), 10 * rho + share)

    def fake_score(y, mask, county, seed, B=0):
        v = y[0, 30]
        report_only = y[0, 5] != v                       # only the offset cells perturb 2010-11
        ok = report_only or np.isclose(v, 8.0) or np.isclose(v, 0.30)
        return {"verdict": "pass", "passed": bool(ok), "g1_passed": True, "g1b_passed": True,
                "ratio_vs_cl": 0.9, "ub90_vs_cl": 0.95, "ratio_vs_cl_level_adjusted": 0.9,
                "coverage90": 0.9}
    monkeypatch.setattr(BL, "ar_world", fake_world)
    monkeypatch.setattr(BL, "g1_world_score", fake_score)
    r = BL.run_g0_g1(np.array(["yunlin"] * 3), np.ones((3, 24), bool), reps=2, seed=1)
    assert np.isclose(r["gate"]["power"], 1 / 9) and r["gate"]["false_pass"] == 1.0
    assert [c["p_pass"] for c in r["cells"] if c["kind"] == "report_only_offset"] == [1.0, 1.0]


def test_stat_keep_needs_24_fitted_months():
    obs = _fit_world(W=4)
    obs[0, 23:] = np.nan                                   # 23 fitted months
    obs[1, 24:] = np.nan                                   # 24 fitted months
    fit = BL.fit_stat(obs)
    win = BL.cl_on(fit["cl_month"], BL.WIN_M) + 0.5
    keep = BL.stat_forecast(fit, win)["keep"]
    assert list(fit["n_fit"][:2]) == [23, 24] and not keep[0] and keep[1]


def _g2_arrays(n_yl, n_ch, seed=0):
    rng = np.random.default_rng(seed)
    W = n_yl + n_ch
    of = rng.normal(0, 1, (W, 132))
    mf = of + 0.1
    os_ = rng.normal(0, 1, (W, 12))
    county = np.array(["yunlin"] * n_yl + ["changhua"] * n_ch)
    return mf, of, os_ + 0.05, os_, county


def test_g2_county_floor_is_ten_wells():
    r = BL.g2_score(*_g2_arrays(10, 9))
    assert r["changhua"]["verdict"] == "not_evaluable" and r["yunlin"]["verdict"] != \
        "not_evaluable"
    assert BL.g2_score(*_g2_arrays(10, 10))["changhua"]["verdict"] in ("pass", "fail")


def test_g2_and_g3_g0_null_truth_differs_from_power_truth(monkeypatch):
    mf, F1, F0, county = _phys(W=30)
    seen = []
    real = BL.phys_world

    def spy(model_fit, model_s, sigma, cty, rho, share, rng, truth_s):
        seen.append(np.array(truth_s))
        return real(model_fit, model_s, sigma, cty, rho, share, rng, truth_s)
    monkeypatch.setattr(BL, "phys_world", spy)
    sig = np.full(30, 0.3)
    m = np.ones(F1.shape, bool)
    BL.run_g0_g2(mf, F1, sig, county, m, reps=1, seed=1)
    assert np.allclose(seen[0], F1) and np.allclose(seen[9], BL.model_clim_s(mf))
    assert not np.allclose(seen[0], seen[9])
    seen.clear()
    BL.run_g0_g3(mf, F1, F0, sig, county, m, reps=1, seed=1, B=20)
    assert np.allclose(seen[0], F1) and np.allclose(seen[9], F0)


def test_phys_world_noise_is_in_time_order(monkeypatch):
    monkeypatch.setattr(BL, "ar_world", lambda c, r, s, g, T: np.tile(np.arange(T, dtype=float),
                                                                       (len(c), 1)))
    of, os_ = BL.phys_world(np.zeros((2, 132)), None, np.ones(2), np.array(["a", "b"]), 0.9,
                            0.0, None, np.zeros((2, 12)))
    assert list(os_[0]) == list(range(12)) and of[0, 0] == 12.0      # 2011 precedes 2012


def test_g1_keeps_a_well_with_exactly_six_scored_months_and_needs_g1b():
    county = np.array(["yunlin"] * 15 + ["changhua"] * 10)
    fc, obs, _ = _fc(25, 0.5, 1.0, 2.0)
    obs[0, 18:] = np.nan                                  # 6 observed 2011 months
    r = BL.g1_score(fc, obs, county, B=200)
    assert r["n_wells_scored"] == 25 and r["dropped_well_index"] == []
    fc, obs, _ = _fc(25, 0.5, 1.0, 2.0, hit_frac=0.8)
    r = BL.g1_score(fc, obs, county, B=200)
    assert r["g1_passed"] and not r["g1b_passed"] and not r["passed"]
    assert r["verdict"] == "fail"


def test_yunlin_gate_reads_power_from_the_rho_083_cell():
    cells = [{"k": 1.0, "rho": 0.0, "report_only": False, "p_pass": 0.95},
             {"k": 1.0, "rho": -0.83, "report_only": False, "p_pass": 0.59},
             {"k": 0.0, "rho": 0.0, "report_only": False, "p_pass": 0.05},
             {"k": 0.0, "rho": -0.83, "report_only": False, "p_pass": 0.15},
             {"k": 1.0, "rho": -0.83, "report_only": True, "p_pass": 0.99}]
    g = BL.yr_gate(cells)
    assert g["power"] == 0.59 and g["false_pass"] == 0.15 and not g["decidable"]


def test_roles_need_the_swap_pre_gates():
    import prereg_bc11 as PB

    dec = {"decidable": True}
    g0 = {"g1": {"gate": dec}, "g2": {"gate": {"yunlin": dec, "changhua": dec}},
          "yr": {"gate": dec}, "g3": {"gate": dec}}

    def pg(yl_pass, yl_er, ch_er, all_pass):
        return {"swap": {"passed": all_pass, "counties": {
            "yunlin": {"passed": yl_pass, "energy_ratio_ok": yl_er},
            "changhua": {"passed": False, "energy_ratio_ok": ch_er}}}}
    r = PB.roles(pg(True, True, True, False), g0)
    assert r == {"g1": "gating", "g2": {"yunlin": "gating", "changhua": "gating"},
                 "yr": "gating", "g3": "report_only"}
    r = PB.roles(pg(False, True, False, False), g0)
    assert r["yr"] == "report_only" and r["g2"]["changhua"] == "report_only"
    assert r["g2"]["yunlin"] == "gating"
    assert PB.roles(pg(True, True, True, True), {**g0, "g1": {"gate": {"decidable": False}}}
                    )["g1"] == "report_only"


def test_load_window_returns_no_row_outside_2010_2011(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    import build_l1pre_v2 as BV

    monkeypatch.setattr(IO, "REPO", tmp_path)
    idx = pd.date_range("2009-06-01", "2012-03-31 23:00", freq="h")
    v = np.where(idx < pd.Timestamp("2010-01-01"), -12345.0, 5.0)
    lc = tmp_path / "chou-shui-data/data/ls_cache"
    lc.mkdir(parents=True)
    pd.DataFrame({"value": v}, index=pd.Index(idx, name="datetime")).to_parquet(
        lc / "gw__09000001__raw.parquet", row_group_size=500)
    s = BV.load_window("09000001")
    assert s.index.min() >= pd.Timestamp("2010-01-01")
    assert s.index.max() < pd.Timestamp("2012-01-01") and (s != -12345.0).all()
    assert len(s) == int(((idx >= "2010-01-01") & (idx < "2012-01-01")).sum())
    import inspect
    assert inspect.signature(BV.build).parameters["loader"].default is BV.load_window


def test_e0_failure_exits_3_and_ladder_rules(tmp_path, monkeypatch):
    import page_run as PR
    import prereg_bc11 as PB

    monkeypatch.setattr(IO, "REPO", tmp_path)
    monkeypatch.setattr(PR, "e0_compare", lambda n, r: {"max_abs_diff_m": 0.5, "tol_m": 0.01,
                                                        "passed": False, "shape": []})
    monkeypatch.setattr(PR, "page_forward", lambda keys=None: {})
    monkeypatch.setattr(PR.HC, "code_head", lambda: K.E0_LADDER[0])
    m = FakePage()
    with pytest.raises(SystemExit) as e:
        PR.e0(tmp_path / Path(K.E0_OUT_FMT.format(head=K.E0_LADDER[0])).parent, "cpu", model=m)
    assert e.value.code == 3
    ok, probs = PB.e0_passing(PB.e0_rungs())
    assert ok is None and any("not run" in p for p in probs)
    p1 = tmp_path / K.E0_OUT_FMT.format(head=K.E0_LADDER[1])
    p1.parent.mkdir(parents=True)
    p1.write_text(json.dumps({"passed": False}))
    ok, probs = PB.e0_passing(PB.e0_rungs())
    assert ok is None and any("NOT EVALUABLE" in p for p in probs)
    p1.write_text(json.dumps({"passed": True, "code_head": K.E0_LADDER[1]}))
    ok, probs = PB.e0_passing(PB.e0_rungs())
    assert ok["head"] == K.E0_LADDER[1] and probs == []


def test_g1_study_refuses_after_the_product_look_and_with_model_inputs(tmp_path, monkeypatch):
    import g0_bc11 as G

    monkeypatch.setattr(IO, "REPO", tmp_path)
    with pytest.raises(SystemExit, match="avail.json only"):
        G.g1_refusals(tmp_path / "panel.npz", None, tmp_path / "prereg")
    (tmp_path / K.PRODUCT).mkdir(parents=True)
    with pytest.raises(SystemExit, match="before the product"):
        G.g1_refusals(None, None, tmp_path / "prereg")


def test_no_head_reads_guard_refuses_ls_cache_and_heads_products(tmp_path):
    import g0_bc11 as G

    f = tmp_path / "chou-shui-data/data/ls_cache/gw__x__raw.parquet"
    f.parent.mkdir(parents=True)
    f.write_text("x")
    h = tmp_path / "results/dataprep/heads/L1_pre2012/v2/heads_monthly.parquet"
    h.parent.mkdir(parents=True)
    h.write_text("x")
    with G.no_head_reads():
        for p in (f, h):
            with pytest.raises(G.HeadRead), open(p):
                pass
        with pytest.raises(G.HeadRead):
            IO.l1_monthly(["a"], ["2010-01"])
        (tmp_path / "ok.txt").write_text("fine")                # other files are allowed
    assert f.read_text() == "x"                                  # restored afterwards
