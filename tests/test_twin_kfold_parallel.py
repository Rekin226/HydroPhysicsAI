"""K-fold anomaly verdict, per-fold parallel runs + merge, and the datum-aware forward
(2026-09-26).

- the own-mean anomaly verdict sits beside the legacy absolute one and follows its rule;
- ``--only-fold K`` jobs merged by ``merge_folds`` reproduce a sequential run's metrics
  bit for bit (CPU), and a merge refuses missing folds or a mixed configuration;
- a ``--well-datum fit`` fold records its own datum, a forward member uses it, and the
  forward twin subtracts it from observed heads at every injection after month 0 while
  the month-0 field stays the calibration's own."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from test_twin_forward import _inputs, _theta_file  # noqa: E402
from test_twin_well_datum import OFFSETS, _problem, _truth  # noqa: E402

from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    _flow_cfg,
    _flow_fit_cols,
    _write_theta,
    fit_flow,
    kfold_wells,
    main,
    write_fit_summary,
    write_fold_file,
    write_kfold_outputs,
    write_per_entry_npz,
)
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.kfold_scores import (  # noqa: E402
    KFOLD_ANOM_MARGIN_MIN,
    kfold_anomaly_scores,
    own_mean_anomaly,
    rescore,
)
from hydrophysics.twin.merge_folds import merge  # noqa: E402

D = torch.float64
EXACT = pytest.mark.filterwarnings("ignore:_cg did not converge")
SIDS = ["w0", "w1", "w2", "w3"]
TIME_COLS = ["fit_time_s", "gate_time_s"]


# ---------------------------------------------------------------------------------------
# the anomaly verdict
# ---------------------------------------------------------------------------------------
def test_anomaly_verdict_scores_departures_from_each_series_own_mean():
    rng = np.random.default_rng(0)
    obs = rng.normal(0.0, 1.0, (5, 30)) + np.arange(5)[:, None] * 10.0
    # the model has the shape exactly but every level wrong; IDW the levels, no shape
    pred = obs + rng.normal(0.0, 8.0, (5, 1))
    idw = np.repeat(obs.mean(axis=1, keepdims=True), 30, axis=1)
    s = kfold_anomaly_scores(pred, idw, obs)
    assert s["r2_anom_kfold"] == pytest.approx(1.0)
    assert s["r2_anom_idw"] == pytest.approx(0.0, abs=1e-12)   # a flat series scores 0
    assert s["verdict_anom_kfold"] == "PASS" and s["n_wells_anom"] == 5
    assert s["r2_anom_well_median_kfold"] == pytest.approx(1.0)
    # the rule is margin >= KFOLD_ANOM_MARGIN_MIN: a tie passes, as pre-registered
    assert KFOLD_ANOM_MARGIN_MIN == 0.0
    assert kfold_anomaly_scores(idw, idw, obs)["verdict_anom_kfold"] == "PASS"
    assert kfold_anomaly_scores(idw, pred, obs)["verdict_anom_kfold"] == "FAIL"


def test_anomaly_means_are_taken_over_the_months_scored_for_both():
    obs = np.array([[1.0, 2.0, 3.0, np.nan]])
    pred = np.array([[0.0, 1.0, np.nan, 50.0]])
    idw = np.array([[1.0, 1.0, 1.0, 1.0]])
    m = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(idw)
    a = own_mean_anomaly(pred, m)
    # month 2 is unscored for the model, month 3 for the observations: mean over 0-1 only
    assert a[0, :2] == pytest.approx([-0.5, 0.5]) and np.isnan(a[0, 2:]).all()


def test_rescore_reads_a_per_entry_dump_and_checks_the_recorded_legacy_r2(tmp_path):
    rng = np.random.default_rng(1)
    recs = []
    for f in range(2):
        obs = rng.normal(0.0, 1.0, (3, 12)) + 5.0
        recs.append({"fold": f, "n_held": 3, "entry": np.arange(3) + 3 * f,
                     "nn_dist": np.ones(3), "x": np.zeros(3), "y": np.zeros(3),
                     "layer": np.zeros(3, dtype="int64"), "obs": obs,
                     "pred": obs + rng.normal(0.0, 0.3, obs.shape),
                     "idw": obs + rng.normal(0.0, 0.6, obs.shape)})
    write_per_entry_npz(str(tmp_path / "stage3_per_entry.npz"), recs)
    out = rescore(str(tmp_path))
    assert out["n_entries"] == 6 and out["n_folds"] == 2
    assert out["verdict_kfold"] == "PASS" and out["verdict_anom_kfold"] == "PASS"
    row = pd.read_csv(tmp_path / "stage3_kfold_anom.csv").iloc[0]
    assert row["r2_anom_kfold"] == pytest.approx(out["r2_anom_kfold"])


# ---------------------------------------------------------------------------------------
# parallel folds + merge == sequential
# ---------------------------------------------------------------------------------------
def _args(**kw):
    a = dict(param_mode="homogeneous", log_t_min_proximal=None, zone_blend_km=0.0,
             no_forcing=False, boundaries="none", meter_filter="dedupe-cap", fix_eta=None,
             fix_head_extra=None, pump_split=False, return_flow=False, l_min=None,
             holdout_months=0, loss="level", delay_storage="off", rivers="none",
             river_set="choushui", sw_recharge=None, delay_u0="eq", aquitard_storage="off",
             sw_components="delivered", river_c_split="none", no_backfill=False, epochs=3,
             seed=0, compile_matvec=False, dx=1000.0, ic_merged_proximal=False,
             ground_elev="wells", strict_coverage=False, dem_npz="x.npz",
             ic_layered_proximal=False, proximal_layered=False, lr=0.05)
    a.update(kw)
    return SimpleNamespace(**a)


def _case(well_datum="fit"):
    g, h0, rch, rf, idx, lay, well_xy = _problem()
    obs = _truth() + torch.tensor(OFFSETS, dtype=D)[:, None]
    obs = obs + 0.3 * torch.sin(torch.arange(obs.shape[1], dtype=D))[None] * torch.tensor(
        [1.0, -0.5, 0.8, 0.2], dtype=D)[:, None]
    kw = dict(n_layers=4, epochs=3, lr=0.05, n_folds=2, seed=0, well_xy=well_xy,
              obs_h0=obs[:, 0].numpy(), recharge_field=rf, well_datum=well_datum,
              sids=SIDS)
    return g, obs, idx, lay, rch, rf, kw


def _full_fit(g, obs, idx, lay, rch, rf, well_datum):
    _, h0, *_ = _problem()
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    ins = fit_flow(m, obs, idx, lay, rch, epochs=3, lr=0.05, h0=h0, recharge_field=rf,
                   well_datum=well_datum)
    datum_meta = None
    if well_datum == "fit":
        datum_meta = {"well_datum": dict(zip(SIDS, map(float, ins["well_datum"]),
                                             strict=True)),
                      "well_datum_mode": "fit", "well_datum_sd": 5.0,
                      "well_datum_stats": {"n": 4}}
    return ins, datum_meta


def _seed_dir(d, ins, datum_meta):
    d.mkdir(parents=True, exist_ok=True)
    _write_theta(str(d / "stage3_theta.json"), ins["theta"],
                 {"param_mode": "homogeneous", **(datum_meta or {})})
    pd.DataFrame({"sid": SIDS}).to_csv(d / "stage3_wells.csv", index=False)


@EXACT
@pytest.mark.parametrize("well_datum", ["off", "fit"])
def test_only_fold_jobs_merge_to_the_sequential_run_bit_for_bit(tmp_path, well_datum):
    g, obs, idx, lay, rch, rf, kw = _case(well_datum)
    ins, datum_meta = _full_fit(g, obs, idx, lay, rch, rf, well_datum)
    cfg = json.loads(json.dumps(_flow_cfg(_args(), g.n_active, None, None, None, None,
                                          None, "abc123")))
    fit = _flow_fit_cols(ins, None, datum_meta)

    seq = tmp_path / "seq"
    _seed_dir(seq, ins, datum_meta)
    gate = kfold_wells(g, obs, idx, lay, rch, dump_path=str(seq / "stage3_per_entry.npz"),
                       **kw)
    write_kfold_outputs(str(seq), cfg, fit, gate, (1, 0.5), 10.0, 20.0, SIDS)

    par = tmp_path / "par"
    _seed_dir(par, ins, datum_meta)
    write_fit_summary(str(par), cfg, fit, (1, 0.5), 10.0, SIDS)
    for k in (1, 0):                                   # order of the jobs is irrelevant
        gk = kfold_wells(g, obs, idx, lay, rch, only_fold=k, **kw)
        assert [f["fold"] for f in gk["per_fold"]] == [k]
        write_fold_file(str(par), gk, cfg, (0, 0.0), 3.0, SIDS, dump=True)
    merged = merge(str(par), log=lambda *_: None)

    assert merged["r2_kfold"] == gate["r2_kfold"] and merged["r2_idw"] == gate["r2_idw"]
    assert merged["r2_anom_kfold"] == gate["r2_anom_kfold"]
    a = pd.read_csv(seq / "stage3_flow.csv")
    b = pd.read_csv(par / "stage3_flow.csv")
    pd.testing.assert_frame_equal(a.drop(columns=TIME_COLS), b.drop(columns=TIME_COLS))
    assert b["gate_time_s"].iloc[0] == 6.0                 # the folds' summed compute
    assert "verdict_anom_kfold" in a.columns and "verdict_kfold" in a.columns
    assert (seq / "stage3_fold_thetas.json").read_text() == \
        (par / "stage3_fold_thetas.json").read_text()
    pd.testing.assert_frame_equal(pd.read_csv(seq / "stage3_kfold_wells.csv"),
                                  pd.read_csv(par / "stage3_kfold_wells.csv"))
    za, zb = np.load(seq / "stage3_per_entry.npz"), np.load(par / "stage3_per_entry.npz")
    assert list(za.keys()) == list(zb.keys())
    for key in za:
        assert np.array_equal(za[key], zb[key], equal_nan=True), key
    ta = json.loads((seq / "stage3_theta.json").read_text())["meta"]["gate"]
    tb = json.loads((par / "stage3_theta.json").read_text())["meta"]["gate"]
    assert ta == tb and ta["verdict_anom"] in ("PASS", "FAIL")
    thetas = json.loads((par / "stage3_fold_thetas.json").read_text())
    if well_datum == "fit":
        # each fold records its own datum for its KEPT wells only
        for f in thetas:
            held = set(np.load(par / "stage3_per_entry.npz")["entry"][
                np.load(par / "stage3_per_entry.npz")["fold"] == f["fold"]])
            assert set(f["well_datum"]) == {SIDS[i] for i in range(4) if i not in held}
    else:
        assert not any("well_datum" in f for f in thetas)


@EXACT
def test_merge_refuses_a_missing_fold_or_a_mixed_configuration(tmp_path):
    g, obs, idx, lay, rch, rf, kw = _case("off")
    ins, _ = _full_fit(g, obs, idx, lay, rch, rf, "off")
    cfg = _flow_cfg(_args(), g.n_active, None, None, None, None, None, "abc123")
    fit = _flow_fit_cols(ins, None, None)
    _seed_dir(tmp_path, ins, None)
    write_fit_summary(str(tmp_path), cfg, fit, (0, 0.0), 1.0, SIDS)
    g0 = kfold_wells(g, obs, idx, lay, rch, only_fold=0, **kw)
    write_fold_file(str(tmp_path), g0, cfg, (0, 0.0), 1.0, SIDS)
    with pytest.raises(FileNotFoundError, match=r"missing for fold\(s\) \[1\]"):
        merge(str(tmp_path), log=lambda *_: None)
    g1 = kfold_wells(g, obs, idx, lay, rch, only_fold=1, **kw)
    # an option outside the CSV (the learning rate) is still part of the recipe checked
    other = _flow_cfg(_args(lr=0.2), g.n_active, None, None, None, None, None, "abc123")
    write_fold_file(str(tmp_path), g1, other, (0, 0.0), 1.0, SIDS)
    with pytest.raises(ValueError, match=r"different configuration.*\['recipe'\]"):
        merge(str(tmp_path), log=lambda *_: None)
    # options that legitimately differ between the jobs (device, --out, ...) do not count
    same = _flow_cfg(_args(device="cuda", out="elsewhere", only_fold=1,
                           dump_predictions=True), g.n_active, None, None, None, None, None,
                     "abc123")
    write_fold_file(str(tmp_path), g1, same, (0, 0.0), 1.0, SIDS)
    assert merge(str(tmp_path), log=lambda *_: None)["n_folds"] == 2
    with pytest.raises(ValueError, match="outside 0..1"):
        kfold_wells(g, obs, idx, lay, rch, only_fold=2, **kw)


def test_cli_validates_only_fold():
    with pytest.raises(SystemExit, match="cannot be combined with --fit-only"):
        main(["--only-fold", "0", "--fit-only"])
    with pytest.raises(SystemExit, match=r"must be in 0\.\.4"):
        main(["--only-fold", "5", "--n-folds", "5"])


# ---------------------------------------------------------------------------------------
# datum-aware forward
# ---------------------------------------------------------------------------------------
def _datum_theta(tmp_path, fold_datum=None):
    from hydrophysics.twin.calibrate_flow import well_datum_stats

    tmp_path.mkdir(parents=True, exist_ok=True)
    p, f = _theta_file(tmp_path)
    obj = json.loads(Path(p).read_text())
    obj["meta"].update({"well_datum": {"a": 2.0, "b": -1.5, "c": 4.0},
                        "well_datum_mode": "fit", "well_datum_sd": 5.0,
                        "well_datum_stats": well_datum_stats(np.array([2.0, -1.5, 4.0]))})
    Path(p).write_text(json.dumps(obj))
    folds = json.loads(Path(f).read_text())
    if fold_datum is not None:
        folds[0]["well_datum"] = fold_datum
    Path(f).write_text(json.dumps(folds))
    return p, f


def test_fold_members_carry_their_own_datum_and_fall_back_to_the_fit(tmp_path):
    from hydrophysics.twin.forward import load_members, member_datum

    p, f = _datum_theta(tmp_path, fold_datum={"a": 1.0, "b": -2.0})   # "c" was held out
    main_m, fold_m = load_members([p, f])
    assert member_datum(main_m, ["a", "b", "c"]) == pytest.approx([2.0, -1.5, 4.0])
    assert member_datum(fold_m, ["a", "b", "c"]) == pytest.approx([1.0, -2.0, 4.0])
    assert fold_m.meta["well_datum_source"] == "fold0"
    assert main_m.meta["well_datum"] == {"a": 2.0, "b": -1.5, "c": 4.0}   # not mutated
    # a fold without a datum (every run before 2026-09-26) inherits the in-sample one
    p2, f2 = _datum_theta(tmp_path / "old")
    assert member_datum(load_members([p2, f2])[1], ["a", "b", "c"]) == \
        pytest.approx([2.0, -1.5, 4.0])


def test_restart_injects_observed_heads_minus_the_datum():
    from hydrophysics.twin.forward import nudge_to_observations

    inp = _inputs()
    A, origin = inp.grid.n_active, len(inp.dates) - 1
    h_model = torch.zeros(4, A, dtype=D)
    d = np.array([2.0, -1.5, 4.0])
    with_d = nudge_to_observations(inp, h_model, origin, 1.0, taper_km=5.0, datum=d)
    shifted = _inputs()
    shifted.obs_h = inp.obs_h - d[:, None]
    ref = nudge_to_observations(shifted, h_model, origin, 1.0, taper_km=5.0)
    assert torch.equal(with_d, ref)
    assert not torch.equal(with_d, nudge_to_observations(inp, h_model, origin, 1.0,
                                                         taper_km=5.0))
    # the month-0 field is the calibration's own h0: raw heads, whatever the datum
    assert torch.equal(inp.initial_heads(0), _inputs().initial_heads(0))


def test_forward_run_keeps_h0_and_subtracts_the_datum_at_later_injections(tmp_path):
    from hydrophysics.twin.compaction import VEPColumn
    from hydrophysics.twin.forward import load_members, parse_scenario, run

    inp = _inputs()
    col = VEPColumn(n_sites=1, dt_days=30.0)
    scen = [(parse_scenario("base:irrigation=1")[0], 1.0)]
    (tmp_path / "plain").mkdir()
    p0, _ = _theta_file(tmp_path / "plain")
    pd_, _ = _datum_theta(tmp_path / "datum")
    kw = dict(horizon=2, gain=1.0, ic_members=0, ic_sigma=0.0, seed=0, col=col,
              device="cpu", log=lambda *_: None, hindcast_gain=1.0, hindcast_every=6,
              restart_taper_km=5.0)
    plain = run(inp, load_members([p0]), scen, **kw)
    datum = run(inp, load_members([pd_]), scen, **kw)
    assert datum["well_datum_members"] == 1 and "well_datum_members" not in plain
    # month 0 and the free-running months before the first nudge are identical ...
    assert np.array_equal(plain["heads_mean"][..., :6], datum["heads_mean"][..., :6])
    # ... the first nudge (month 6) injects obs - d
    assert not np.array_equal(plain["heads_mean"][..., 6], datum["heads_mean"][..., 6])
    # and with the datum shifted into the observations the plain run is the datum run
    shifted = _inputs()
    d = np.array([2.0, -1.5, 4.0])
    shifted.obs_h = inp.obs_h - d[:, None]
    shifted.obs_h[:, 0] = inp.obs_h[:, 0]                    # h0 from the raw month 0
    ref = run(shifted, load_members([p0]), scen, **kw)
    assert np.array_equal(ref["heads_mean"], datum["heads_mean"])
    assert np.array_equal(ref["subs_mean"], datum["subs_mean"])
