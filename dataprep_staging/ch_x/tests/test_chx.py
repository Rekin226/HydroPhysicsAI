"""CPU tests of I1.CHX.v1 (round 7 rank 13): no model run, no GPU, nothing registered in the
real registry, no look logged there, nothing written outside pytest's tmp dirs. Every panel,
well table and observation here is SYNTHETIC.

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
for p in (str(X), str(REPO / "dataprep_staging"), str(REPO / "dataprep_staging/n2_policy")):
    if p not in sys.path:
        sys.path.insert(0, p)
import bootstrap_scorer as BS  # noqa: E402
import chj_forcing as CJ  # noqa: E402
import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import chx_lib as L  # noqa: E402
import g0_chx as G  # noqa: E402
import prereg_chx as P  # noqa: E402
import score_chx as S  # noqa: E402

WIN = L.month_list(*K.CHX_WINDOW)
RUN = L.month_list(*K.RUN_MONTHS)
PARAMS = {"sigma_tot_m": 1.318, "block_share": 0.439, "phi": 0.859, "kappa_m": 0.296}


# ----------------------------------------------------------------------------- helpers
def fake_wells(path: Path, n_ch: int = 23, n_yl: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    ids = [f"07{i:06d}" for i in range(n_ch - 4)] + list(K.NEST)
    for i, s in enumerate(ids):
        rows.append({"station_id": s, "county": "Changhua", "x": 180000.0 + 3000 * (i % 7),
                     "y": 2650000.0 + 4000 * (i // 7), "layer_code": "1", "layer": 0,
                     "cell": 100 + i, "block5": "x", "kept": True,
                     "n_fit_months": int(rng.integers(24, 132)), "residual_sd_m": 9.9,
                     "datum_m": 9.9})
    for i in range(n_yl):
        rows.append({"station_id": f"09{i:06d}", "county": "Yunlin", "x": 190000.0,
                     "y": 2620000.0 + 5000 * i, "layer_code": "2", "layer": 1,
                     "cell": 500 + i, "block5": "y", "kept": True, "n_fit_months": 60,
                     "residual_sd_m": 9.9, "datum_m": 9.9})
    rows.append({"station_id": "07999999", "county": "Changhua", "x": 1.0, "y": 1.0,
                 "layer_code": "", "layer": -1, "cell": -1, "block5": "none", "kept": False,
                 "n_fit_months": 3, "residual_sd_m": np.nan, "datum_m": np.nan})
    t = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(path, index=False)
    return t


def synth_panel(seed: int = 1, k_true: float = 1.0) -> tuple[dict, np.ndarray]:
    """A chx_panel-shaped dict and synthetic obs with obs - B = k c + noise."""
    rng = np.random.default_rng(seed)
    W, T = K.N_EXT_CHANGHUA, len(WIN)
    sid = np.array([f"07{i:06d}" for i in range(W - 4)] + list(K.NEST))
    x = 180000.0 + 3000 * (np.arange(W) % 7)
    y = 2650000.0 + 4000 * (np.arange(W) // 7)
    bl = L.well_blocks(sid, x, y)
    a = 0.4 * np.sin(2 * np.pi * np.arange(T) / 12) + 0.15 * rng.standard_normal(T)
    c = -rng.uniform(1, 3, W)[:, None] * a[None, :]
    B = 10.0 + rng.normal(0, 1, (W, 1)) + np.zeros((W, T))
    cp = 0.3 * rng.standard_normal((W, T))
    P_ = {"months": np.array(WIN), "sid_ch": sid, "x_ch": x, "y_ch": y,
          "block_ch": np.array([bl[s] for s in sid]), "c_ch": c, "B_ch": B, "A_ch": B + c,
          "C_ch": B + c - cp, "c_placebo_ch": cp, "a_ch": a}
    obs = B + k_true * c + 0.05 * rng.standard_normal((W, T))
    obs[:, :30] = np.nan
    return P_, obs


# ----------------------------------------------------------------------------- arms
def test_pattern_energy_replaces_only_the_cells_after_the_pattern_year():
    rng = np.random.default_rng(0)
    E = rng.uniform(1, 2, (5, len(RUN)))
    cells = np.array([True, False, True, False, False])
    Z = L.pattern_energy(E, RUN, cells, 2012)
    assert np.array_equal(Z[~cells], E[~cells])
    assert np.array_equal(Z[:, :12], E[:, :12])
    for t, m in enumerate(RUN):
        if int(m[:4]) > 2012:
            assert np.array_equal(Z[cells, t], E[cells, int(m[5:7]) - 1])
    Zk = L.pattern_energy(np.stack([E, 2 * E]), RUN, cells, 2012)
    assert np.allclose(Zk[1], 2 * Z)
    with pytest.raises(ValueError):
        L.pattern_energy(E[:, 3:], RUN[3:], cells, 2012)


def test_county_anomaly_is_zero_for_a_pure_seasonal_cycle():
    mo = np.array([int(m[5:7]) for m in RUN])
    E = np.tile(1.0 + 0.5 * np.sin(mo / 2.0), (3, 1))
    assert np.allclose(L.county_anomaly(E, RUN, np.array([True, True, False])), 0.0)


def test_blocks_merge_the_nest_and_keep_5km_squares():
    sid = ["07000001", *K.NEST]
    b = L.well_blocks(sid, [182000.0, 1e5, 2e5, 3e5, 4e5], [2.65e6] * 5)
    assert {b[s] for s in K.NEST} == {K.NEST_BLOCK}
    assert b["07000001"] == BS.block_ids([182000.0], [2.65e6])[0]


# ----------------------------------------------------------------------------- primary
def test_score_direction_recovers_k_and_maps_direction():
    P_, obs = synth_panel(k_true=1.0)
    r = S.score_panel(P_, obs, np.where(np.isfinite(obs), 0, 9))
    pr = r["primary"]
    assert pr["direction"] and pr["verdict"] in K.DIRECTION_VERDICTS
    assert pr["form"] == K.CHX_PRIMARY_FORM == "twoway"
    assert pr["beta_primary"]["beta"] == pytest.approx(1.0, abs=0.05)
    assert pr["informative_gate"]["transformed"] and pr["informative_gate"]["passes"]
    assert "report_only_detrended" in pr and "placebo_report_only" in r


def test_score_direction_negative_slope_is_not_supported():
    P_, obs = synth_panel(k_true=-1.0)
    pr = S.score_panel(P_, obs, np.zeros(obs.shape, int))["primary"]
    assert not pr["direction"] and pr["verdict"] == "not_supported"


def test_uninformative_contrast_gives_not_informative():
    P_, obs = synth_panel()
    P_["c_ch"] = P_["c_ch"] * 1e-4
    pr = S.score_panel(P_, obs, np.zeros(obs.shape, int))["primary"]
    assert pr["verdict"] == "not_informative" and not pr["direction"]


def test_eligibility_drops_short_wells_and_flags_2_plus():
    P_, obs = synth_panel()
    obs[0, :] = np.nan
    obs[0, -5:] = 1.0
    fl = np.zeros(obs.shape, int)
    fl[1, :] = 3
    r = L.eligible(L.long_panel(P_["sid_ch"], WIN, obs, P_["B_ch"], P_["c_ch"], fl))
    assert P_["sid_ch"][0] in r["dropped_wells"]
    assert P_["sid_ch"][1] not in set(r["well"])


def test_fe_basis_matches_least_squares_and_detrend_form_matches_scorer():
    rng = np.random.default_rng(7)
    w = np.repeat(["a", "b", "c"], 24)
    m = np.tile(L.month_list("2013-01", "2014-12"), 3)
    keep = rng.random(72) > 0.2
    w, m = w[keep], m[keep]
    v = rng.normal(size=w.size)
    t = L.month_t(m)
    X = np.hstack([(w[:, None] == np.array(["a", "b", "c"])).astype(float),
                   (w[:, None] == np.array(["a", "b", "c"])) * t[:, None],
                   (m[:, None] == np.unique(m)).astype(float)])
    direct = v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    assert np.allclose(L.fe_resid(v, L.fe_basis(w, m, "twoway")), direct, atol=1e-10)
    rows = {"well": w, "month": m, "t": t}
    a, _ = L.transform(v, v, rows, "detrended")
    b = L.fe_resid(v, L.fe_basis(w, m, "detrended"))
    assert np.allclose(a, b, atol=1e-10)
    with pytest.raises(ValueError):
        L.fe_basis(w, m, "bogus")


def test_twoway_absorbs_a_common_confounder_and_trend():
    P_, _ = synth_panel()
    rng = np.random.default_rng(2)
    g = -2.0 * P_["a_ch"] + 0.1 * rng.standard_normal(len(WIN))     # tracks the anomaly
    obs = P_["B_ch"] + g[None, :] + L.hinge(WIN)[None, :]            # k = 0
    pr = S.score_panel(P_, obs, np.zeros(obs.shape, int))["primary"]
    assert abs(pr["beta_primary"]["beta"]) < 1e-8 and not pr["direction"]
    assert pr["report_only_detrended"]["beta"] > 0.5                  # the old form is fooled


def test_detrended_rmse_equals_direct_computation():
    rng = np.random.default_rng(5)
    w = np.repeat(["a", "b"], 30)
    t = np.tile(np.arange(30.0), 2)
    e, c = rng.normal(size=60), rng.normal(size=60)
    got = L.detrended_rmse(e, c, w, t)
    def dt_(v, m):
        x = np.vstack([np.ones(m.sum()), t[m]]).T
        return v[m] - x @ np.linalg.lstsq(x, v[m], rcond=None)[0]
    res = [dt_(e, w == s) for s in ("a", "b")]
    assert got["ZR"]["equal_well"] == pytest.approx(np.sqrt(np.mean([np.mean(r ** 2)
                                                                      for r in res])))


# ----------------------------------------------------------------------------- G0
def test_g0_plumbing_trend_and_gate():
    pan = G.synthetic_panel()
    assert pan["mask"].sum(axis=1).tolist() == np.clip(pan["counts"], 0, 120).tolist()
    assert L.plumbing_exact(pan)["ok"]
    h = L.hinge(WIN)
    assert h[0] == 0 and h[-1] == pytest.approx(K.TREND_AMP_M)
    assert pan["trend_sign"] in (1.0, -1.0)
    cell = L.g0_cell(pan, PARAMS, 0.0, 0.0, np.random.default_rng(0), 20, B=200)
    assert 0.0 <= cell["p_direction"] <= 1.0 and cell["replicates"] == 20
    cells = [{"k": k, "rho": r, "trend_mult": 1.0, "het_corr": 0.0, "p_direction": p}
             for k, r, p in ((1.0, -0.83, 0.7), (0.0, 0.0, 0.1), (0.0, -0.83, 0.19))]
    cells.append({"k": 0.0, "rho": -0.83, "trend_mult": 1.0, "het_corr": 0.5,
                  "p_direction": 0.9})                               # report-only: ignored
    assert L.g0_gate(cells)["decidable"]
    cells[2]["p_direction"] = 0.25
    assert not L.g0_gate(cells)["decidable"]
    assert not L.g0_gate([], informative=False)["decidable"]


def test_run_g0_stops_at_an_uninformative_contrast():
    pan = G.synthetic_panel(scale=1.0)                  # transformed RMS < 0.05 m
    assert not pan["informative"]["passes"]
    res = L.run_g0(pan, PARAMS, reps=3, B=50)
    assert res["cells"] == [] and res["gate"]["reading"].startswith("not_informative")
    assert not res["gate"]["decidable"] and res["plumbing"]["ok"]


def test_world_heterogeneous_loading():
    pan = G.synthetic_panel(scale=3.0)
    p0 = {**PARAMS, "sigma_tot_m": 0.0}
    a = L.world(pan, 0.0, 0.0, np.random.default_rng(1), p0, trend_mult=0.0, het_mult=0.0)
    b = L.world(pan, 0.0, 0.0, np.random.default_rng(1), p0, trend_mult=0.0, het_mult=1.0,
                het_corr=1.0)
    ok = pan["mask"] & (np.abs(np.nan_to_num(a)) > 1e-3)
    expect = np.broadcast_to((1.0 + pan["load_z"])[:, None], a.shape)
    assert ok.sum() > 100 and np.allclose(b[ok] / a[ok], expect[ok])


def test_trend_sign_biases_upward():
    T = len(WIN)
    h = L.hinge(WIN)
    c = np.tile(h, (4, 1)) * np.array([[1.0], [2.0], [0.5], [1.0]])
    m = np.ones((4, T), bool)
    assert L.trend_sign(c, m, h) == 1.0
    assert L.trend_sign(-c, m, h) == -1.0


def test_g0_smoke_cli_writes_once(tmp_path):
    args = ["--smoke", "--out", str(tmp_path), "--reps", "4", "--scale", "3"]
    assert G.main(args) == 0
    r = json.loads((tmp_path / "g0_smoke_twoway_x3.json").read_text())
    assert r["smoke"] and r["plumbing"]["ok"] and len(r["cells"]) == 8
    assert len(r["report_only_cells"]) == len(K.G0_REPORT_CELLS)
    assert r["design"]["form"] == "twoway"
    with pytest.raises(SystemExit):
        G.main(args)
    with pytest.raises(SystemExit):
        G.main(["--out", str(tmp_path), "--panel", "x.npz", "--scale", "2"])


# ----------------------------------------------------------------------------- CH-J
def test_q0_rest_fractions_follow_calibrate_flow():
    f = CJ.q0_rest_fractions(0.0, pump_layer=1)
    assert np.allclose(f, [0.5, 0.5, 0.0, 0.0])
    assert np.allclose(CJ.q0_rest_fractions(None, 1), [0, 1, 0, 0])
    assert np.allclose(CJ.q0_rest_fractions(2.0, 0), [1, 0, 0, 0])   # no split in layer 0


def test_county_layer_fractions_keep_q0_placement_outside_changhua():
    ch = np.array([True, False, True])
    rest = CJ.q0_rest_fractions(-1.0, 1)
    F = CJ.county_layer_fractions(ch, rest)
    assert np.allclose(F.sum(axis=0), 1.0)
    assert np.allclose(F[:, 0], K.CHJ_SHARES_CHANGHUA) and np.allclose(F[:, 1], rest)
    with pytest.raises(ValueError):
        CJ.county_layer_fractions(ch, rest, shares=(0.5, 0.5, 0.1, 0.1))
    with pytest.raises(ValueError):
        CJ.county_layer_fractions(ch, [0.0, 0.9, 0.0, 0.0])


def test_freeze_partition_frees_only_changhua_sw_entries():
    q0 = {"log_T_mid": 1.0, "log_S_mid": 2.0, "log_eta": 3.0}
    keys = ["log_T_mid", "log_T_mid_changhua", "log_S_mid", "log_eta", "log_sw_scale"]
    free, pinned = CJ.freeze_partition(keys)
    assert free == ["log_T_mid_changhua", "log_sw_scale[0]", "log_sw_scale[2]"]
    assert "log_sw_scale[1]" in pinned and "log_sw_scale[3]" in pinned
    assert "log_sw_scale" not in free + pinned
    out = CJ.pin_frozen(keys, q0)
    assert out["pinned"]["log_S_mid"] == 2.0 and out["pinned"]["log_eta"] == 3.0
    assert out["pinned"]["log_sw_scale[1]"] == pytest.approx(np.log(0.15))
    assert out["pinned"]["log_sw_scale[3]"] == pytest.approx(np.log(0.25))
    assert out["start_free"]["log_T_mid_changhua"] == 1.0
    assert out["start_free"]["log_sw_scale[2]"] == pytest.approx(np.log(0.25))
    with pytest.raises(KeyError):
        CJ.pin_frozen(["log_L_mid"], q0)


def test_merge_fields_adds_deliveries_and_validates():
    d = np.array(["2012-01-01", "2012-02-01"])
    w1 = {"sw_m_per_day": np.ones((3, 2)), "perc_m_per_day": np.full((3, 2), 0.5), "dates": d,
          "dx": 1000.0, "n_active": 3}
    x5 = {"sw_m_per_day": np.full((3, 2), 0.25), "dates": d, "dx": 1000.0, "n_active": 3}
    m = CJ.merge_fields(w1, x5)
    assert np.allclose(m["sw_m_per_day"], 1.25) and np.allclose(m["perc_m_per_day"], 0.5)
    mc = CJ.merge_fields(w1, x5, np.array([True, False, True]))
    assert np.allclose(mc["sw_m_per_day"][1], 0.0) and np.allclose(mc["perc_m_per_day"][1], 0)
    assert np.allclose(mc["sw_m_per_day"][[0, 2]], 1.25)
    with pytest.raises(ValueError):
        CJ.merge_fields(w1, x5, np.array([True, False]))
    x5["sw_m_per_day"] = -x5["sw_m_per_day"]
    with pytest.raises(ValueError):
        CJ.merge_fields(w1, x5)


def test_chj_gate_statistics():
    t = np.arange(len(WIN))
    seas = np.sin(2 * np.pi * t / 12)
    obs = np.tile(seas, (3, 1)) + 0.01 * t
    assert L.seasonal_amplitude(obs[0], WIN) > 0
    r = L.amplitude_ratios(1.2 * obs, obs, WIN)
    assert np.allclose(r, 1.2)
    assert L.county_mean_corr(2 * obs, obs) == pytest.approx(1.0)
    assert L.chj_sw_outcome(0.01) == "co_located_recharge_rejected"
    assert L.chj_sw_outcome(0.3) == "off_bound"
    nh = L.chj_noharm_gates({"yunlin_kfold": 0.5, "yunlin_county_corr": 0.6,
                             "oof_leveling": 0.4, "changhua_leveling": 0.1},
                            {"yunlin_kfold": 0.505, "yunlin_county_corr": 0.6,
                             "oof_leveling": 0.41, "changhua_leveling": 0.08})
    assert nh["passed"]
    fw = L.chj_forward_gates({"subsidence_2023_25_changhua_m": -0.01,
                              "subsidence_2023_25_yunlin_m": 0.1,
                              "rising_share_changhua": 0.1}, {"rising_share_changhua": 0.2})
    assert not fw["passed"]


# ----------------------------------------------------------------------------- registration
def test_doc_validates_in_a_tmp_registry(tmp_path, monkeypatch):
    from harness import prereg3

    fake_wells(tmp_path / K.EXT_WELLS)
    monkeypatch.setattr(IO, "REPO", tmp_path)
    d = P.doc()
    assert d["fresh_units"] == [K.CHX_UNIT]
    assert all(t["fresh_units"] == [K.CHX_UNIT] for t in d["tests"])
    assert K.CHX_UNIT_YL in d["tests"][1]["external_units"]
    assert {K.R1C_COLUMN, K.Q0_THETA, K.COUNTY_MAP} <= {x for x in d["inputs"]
                                                       if isinstance(x, str)}
    assert d["g0_spec"]["form"] == K.CHX_PRIMARY_FORM and "expected_on_stand_in" in d["g0_spec"]
    assert d["screens"]["ids"] == sorted(d["screens"]["ids"]) and len(d["screens"]["ids"]) == 23
    assert d["kind"] == "test" and d["tests"][0]["test_id"] == K.CHX_TEST
    assert d["scorer"]["sha256"] == BS.file_sha256()
    assert prereg3.validate_prereg(d, units_registry=None, prereg_dir=tmp_path / "p",
                                   registries=[]) == []


def _register(tmp_path, monkeypatch):
    from harness import prereg3

    fake_wells(tmp_path / K.EXT_WELLS)
    monkeypatch.setattr(IO, "REPO", tmp_path)
    d = P.doc()
    d["inputs"] = [x for x in d["inputs"] if isinstance(x, dict)]
    pdir, rows = tmp_path / "prereg", tmp_path / "rows"
    _, h = prereg3.write_prereg(d, prereg_dir=pdir, rows_dir=rows, units_registry=None,
                                registries=[])
    return d, pdir, rows, h


def _fake_outputs(tmp_path, d, k_true=1.0, decidable=True):
    P_, obs = synth_panel(k_true=k_true)
    P_["sid_ch"] = np.array(d["screens"]["ids"])
    hc = tmp_path / "hc"
    hc.mkdir()
    pz = hc / "chx_panel.npz"
    np.savez(pz, **P_)
    json.dump({"panel_sha256": IO.sha256(pz), "code_head": K.WT_HEAD,
               "theta_sha256": d["recipe"]["theta_sha256"],
               "rows_sha256_changhua": d["screens"]["rows_sha256"]},
              (hc / "chx_panel.json").open("w"))
    g = {"smoke": False, "panel_sha256": IO.sha256(pz), "plumbing": {"ok": True},
         "design": {"replicates_per_cell": K.G0_REPLICATES, "form": K.CHX_PRIMARY_FORM},
         "code_sha256": {p: IO.sha256(X / p) for p in ("g0_chx.py", "chx_lib.py",
                                                       "chx_const.py")},
         "gate": {"decidable": decidable, "power_k1_rho083": 0.7 if decidable else 0.3,
                  "false_pass_k0_max_rho": 0.1}}
    gz = tmp_path / "g0.json"
    gz.write_text(json.dumps(g))
    return pz, gz, obs


def test_score_end_to_end_in_tmp_registry(tmp_path, monkeypatch):
    d, pdir, rows, h = _register(tmp_path, monkeypatch)
    pz, gz, obs = _fake_outputs(tmp_path, d)
    monkeypatch.setattr(IO, "l1_monthly", lambda sids, months, **kw: (
        obs, np.where(np.isfinite(obs), 0, 9), ["synthetic"]))
    a = SimpleNamespace(panel=pz, g0=gz, chj_panel=None, chj_metrics=None)
    rep = S.run_score(a, prereg_dir=pdir, rows_dir=rows)
    assert rep["prereg_sha256"] == h and rep["direction"] and rep["heads_read"]
    assert rep["chj_secondary"]["status"] == "not_run"
    assert (tmp_path / S.REPORT).exists()
    looks = [json.loads(s) for s in (pdir / "looks.jsonl").read_text().splitlines()]
    assert [lk["purpose"] for lk in looks] == ["score"]
    with pytest.raises(SystemExit, match="scored once"):
        S.run_score(a, prereg_dir=pdir, rows_dir=rows)


def test_not_decidable_reads_no_head(tmp_path, monkeypatch):
    d, pdir, rows, _ = _register(tmp_path, monkeypatch)
    pz, gz, _ = _fake_outputs(tmp_path, d, decidable=False)

    def boom(*a, **k):
        raise AssertionError("a head was read")
    monkeypatch.setattr(IO, "l1_monthly", boom)
    rep = S.run_score(SimpleNamespace(panel=pz, g0=gz, chj_panel=None, chj_metrics=None),
                      prereg_dir=pdir, rows_dir=rows)
    assert rep["verdict"] == "not_decidable" and rep["heads_read"] is False
    assert not (pdir / "looks.jsonl").exists()


def test_not_informative_reads_no_head_and_needs_no_g0(tmp_path, monkeypatch):
    d, pdir, rows, _ = _register(tmp_path, monkeypatch)
    pz, _, _ = _fake_outputs(tmp_path, d)
    P_ = dict(np.load(pz))
    P_["c_ch"] = P_["c_ch"] * 1e-3
    np.savez(pz, **P_)
    s = json.loads((pz.parent / "chx_panel.json").read_text())
    s["panel_sha256"] = IO.sha256(pz)
    (pz.parent / "chx_panel.json").write_text(json.dumps(s))

    def boom(*a, **k):
        raise AssertionError("a head was read")
    monkeypatch.setattr(IO, "l1_monthly", boom)
    rep = S.run_score(SimpleNamespace(panel=pz, g0=tmp_path / "missing.json", chj_panel=None,
                                      chj_metrics=None), prereg_dir=pdir, rows_dir=rows)
    assert rep["verdict"] == "not_informative" and rep["heads_read"] is False
    assert not (pdir / "looks.jsonl").exists()


def _chj_files(tmp_path, P_, obs, seeds=K.CHJ_SEEDS, frozen="2026-11-02T00:00:00+00:00"):
    d = tmp_path / "chj"
    d.mkdir(exist_ok=True)
    arr = {"q0_ch": P_["A_ch"]}
    met = {"seeds": {}}
    for sd in seeds:
        arr[f"chjA_ch_{sd}"] = P_["A_ch"]
        arr[f"chjB_ch_{sd}"] = P_["B_ch"]
        arr[f"chjA_yl_{sd}"] = obs[:5] * 1.1
        met["seeds"][str(sd)] = {
            "noharm": {"yunlin_kfold": 0.5, "yunlin_county_corr": 0.6, "oof_leveling": 0.4,
                       "changhua_leveling": 0.1},
            "q0": {"yunlin_kfold": 0.5, "yunlin_county_corr": 0.6, "oof_leveling": 0.4,
                   "changhua_leveling": 0.1},
            "forward": {"subsidence_2023_25_changhua_m": 0.01,
                        "subsidence_2023_25_yunlin_m": 0.02, "rising_share_changhua": 0.1},
            "forward_q0": {"rising_share_changhua": 0.2}, "sw_fraction_changhua": 0.3}
    np.savez(d / "chj_panel.npz", **arr)
    (d / "chj_metrics.json").write_text(json.dumps(met))
    eng = {"hook_commit": "abc1234", "hook_sha256": "h" * 64, "forcing_version": "v9",
           "forcing_sha256": "f" * 64, "pregate": {"passed": True, "max_abs_diff_m": 1e-9},
           "frozen_at": frozen, "producer_sha256": "p" * 64}
    (d / "chj_engineering.json").write_text(json.dumps(eng))
    (d / "chj_panel.json").write_text(json.dumps({
        "engineering_sha256": IO.sha256(d / "chj_engineering.json"),
        "producer_sha256": "p" * 64, "chj_panel_sha256": IO.sha256(d / "chj_panel.npz"),
        "chj_metrics_sha256": IO.sha256(d / "chj_metrics.json"),
        "created": "2026-11-05T00:00:00+00:00"}))
    return d


def test_chj_block_gates_only_with_verified_provenance(tmp_path):
    P_, obs = synth_panel()
    obs_yl = obs[:5]
    doc = {"created": "2026-11-01T12:00:00+00:00"}
    d = _chj_files(tmp_path, P_, obs)
    r = S.chj_block(P_, obs, obs_yl, d / "chj_panel.npz", d / "chj_metrics.json", doc,
                    d / "chj_engineering.json")
    assert r["status"] == "run" and r["provenance"]["gating"]
    assert set(r["seeds"]) == {str(s) for s in K.CHJ_SEEDS}
    assert all(v["external"]["direction_ok"] for v in r["seeds"].values())
    assert r["passed_all_seeds"] == r["report_only_all_seeds_pass"]
    # no engineering record -> report-only, nothing gates
    r2 = S.chj_block(P_, obs, obs_yl, d / "chj_panel.npz", d / "chj_metrics.json", doc, None)
    assert r2["status"] == "report_only" and not r2["passed_all_seeds"]
    # one seed only -> report-only
    (tmp_path / "one").mkdir()
    d1 = _chj_files(tmp_path / "one", P_, obs, seeds=(2027,))
    r3 = S.chj_block(P_, obs, obs_yl, d1 / "chj_panel.npz", d1 / "chj_metrics.json", doc,
                     d1 / "chj_engineering.json")
    assert r3["status"] == "report_only" and any("seeds" in p for p in
                                                 r3["provenance"]["problems"])
    # frozen after the 21-day deadline -> report-only
    (tmp_path / "late").mkdir()
    dl = _chj_files(tmp_path / "late", P_, obs, frozen="2026-12-01T00:00:00+00:00")
    r4 = S.chj_block(P_, obs, obs_yl, dl / "chj_panel.npz", dl / "chj_metrics.json", doc,
                     dl / "chj_engineering.json")
    assert r4["status"] == "report_only" and any("deadline" in p for p in
                                                 r4["provenance"]["problems"])


def test_scorer_refuses_a_changed_frozen_file(tmp_path, monkeypatch):
    d, pdir, rows, _ = _register(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "frozen_code", lambda: {**d["frozen_code"],
                                                   S.FROZEN[1]: "0" * 64})
    with pytest.raises(SystemExit, match="frozen code differs"):
        S.verify_registration(pdir, rows)


def test_scorer_refuses_a_foreign_panel(tmp_path, monkeypatch):
    d, pdir, rows, _ = _register(tmp_path, monkeypatch)
    pz, gz, _ = _fake_outputs(tmp_path, d)
    s = json.loads((pz.parent / "chx_panel.json").read_text())
    s["code_head"] = "abcdef0"
    (pz.parent / "chx_panel.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="step 1"):
        S.run_score(SimpleNamespace(panel=pz, g0=gz, chj_panel=None, chj_metrics=None),
                    prereg_dir=pdir, rows_dir=rows)


def test_runner_carries_the_declared_lines():
    sh = (X / "run_chx.sh").read_text()
    for s in P.RUNNER_LINES:
        assert s in sh.splitlines()
    assert "/" + "home/" not in sh


def test_no_home_paths_in_committed_files():
    for f in X.rglob("*.py"):
        assert "/" + "home/" not in f.read_text(), f
