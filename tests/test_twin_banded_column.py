"""The opt-in banded compaction column (2026-09-29) and its 182 km step scorer.

The per-zone column's parameters jump on the 182 km mid/distal line, and the projected
field steps with them (14.7 cm east vs -2.0 cm west of the line, 2022-2032), which the
leveling network does not show. ``calibrate_coupled --configs banded`` puts the column
parameters on bands along the easting, mixed per cell by hat weights, with a smoothness
penalty between neighbouring bands. All CPU and synthetic.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hydrophysics.twin.column_step import (
    first_column_contrast,
    policy_summary,
    slope_per_year,
    step_fit,
)
from hydrophysics.twin.zones import band_centres_km, band_weights, column_band_weights


def _xy(*x_km: float) -> np.ndarray:
    return np.column_stack([np.asarray(x_km) * 1000.0, np.full(len(x_km), 2.6e6)])


def _col(ske=1e-3, skv=2e-2, tau=365.0, h=0.0) -> dict:
    return {"log_ske": float(np.log(ske)), "log_skv": float(np.log(skv)),
            "log_tau": float(np.log(tau)), "h_pc0": float(h)}


# ---- band geometry ------------------------------------------------------------------------
def test_hat_band_weights_are_a_partition_of_unity_and_continuous():
    c = band_centres_km(np.array([160.0, 225.0]), 6)
    assert np.allclose(c, [160, 173, 186, 199, 212, 225])
    x = np.linspace(150.0, 235.0, 851)
    w = band_weights(_xy(*x), c, "hat")
    assert w.shape == (6, x.size)
    assert np.allclose(w.sum(0), 1.0) and (w >= 0).all()
    # continuous: 0.1 km apart never moves any weight by more than 0.1 / 13
    assert np.abs(np.diff(w, axis=1)).max() <= 0.1 / 13.0 + 1e-9
    # one-hot on the centres, the end band beyond the ends
    on = band_weights(_xy(*c), c, "hat")
    assert np.allclose(on, np.eye(6))
    assert np.allclose(band_weights(_xy(140.0, 240.0), c, "hat")[[0, -1], [0, 1]], 1.0)


def test_step_band_weights_are_one_hot_nearest_centre():
    c = [170.0, 180.0, 190.0]
    w = band_weights(_xy(166.0, 174.0, 176.0, 186.0, 199.0), c, "step")
    assert np.array_equal(w.argmax(0), [0, 0, 1, 2, 2])
    assert np.allclose(w.sum(0), 1.0) and set(np.unique(w)) == {0.0, 1.0}


def test_band_geometry_rejects_bad_input():
    with pytest.raises(ValueError, match="at least 2"):
        band_centres_km(np.array([1.0, 2.0]), 1)
    with pytest.raises(ValueError, match="increasing"):
        band_weights(_xy(1.0), [2.0, 1.0])
    with pytest.raises(ValueError, match="kind"):
        band_weights(_xy(1.0), [1.0, 2.0], "linear")
    with pytest.raises(ValueError, match="band centres"):
        column_band_weights({"banded": [_col()] * 3, "band_centres_km": [1.0, 2.0]}, _xy(1.0))


def test_column_param_sets_reads_every_column_layout():
    # compaction imports torch at module level; the torch-less CI job must still collect
    pytest.importorskip("torch")
    from hydrophysics.twin.compaction import column_param_sets

    shared = _col()
    assert column_param_sets(shared) == [shared]
    assert column_param_sets({"zonal": [shared] * 3}) == [shared] * 3
    assert column_param_sets({"banded": [shared] * 7}) == [shared] * 7


# ---- the step scorer ---------------------------------------------------------------------
def test_step_fit_recovers_a_step_under_a_trend_and_a_quadratic_in_y():
    rng = np.random.default_rng(0)
    x = rng.uniform(176.0, 188.0, 400)
    y = rng.uniform(2600.0, 2640.0, 400)
    yc = y - 2620.0
    rate = 1.0 + 0.2 * (x - 182) + 0.05 * yc - 0.003 * yc ** 2 + 2.5 * (x >= 182)
    rate = rate + rng.normal(0.0, 0.05, x.size)
    for w in (2.0, 4.0, 6.0):
        f = step_fit(x, y, rate, 182.0, w)
        assert abs(f["step"] - 2.5) < 4 * f["se"] + 0.02 and f["n_east"] and f["n_west"]
    none = step_fit(x, y, rate - 2.5 * (x >= 182), 182.0, 4.0)
    assert abs(none["step"]) < 4 * none["se"] + 0.02
    one_side = step_fit(x[x > 183], y[x > 183], rate[x > 183], 182.0, 4.0)
    assert np.isnan(one_side["step"]) and one_side["n_west"] == 0


def test_first_column_contrast_and_slope():
    x = np.array([180.5, 181.5, 182.5, 183.5])
    c = first_column_contrast(np.array([1.0, 2.0, 10.0, 20.0]), x, 182.0)
    assert c == {"east": 10.0, "west": 2.0, "n_east": 1, "n_west": 1}
    t = np.arange(24) / 12.0
    s = np.stack([3.0 * t + 1.0, -t])
    assert np.allclose(slope_per_year(s, t), [3.0, -1.0])


def test_policy_summary_reads_the_scenario_start_and_the_avoided_subsidence():
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=72, freq="MS")
    t = np.arange(72) / 12.0
    base = np.tile(0.5 * t, (3, 1))                                   # (A, T) cm
    pol = base.copy()
    pol[:, 36:] -= 1.0                                                # a one-time rebound
    subs = np.stack([base, pol])
    out = policy_summary(subs, [str(d) for d in dates], 24, ["baseline", "cut30"],
                         ["baseline: baseline (no change)",
                          "cut30: irrigation x0.7 from 2023-01"])
    assert len(out) == 1
    assert out[0]["avoided_horizon_cm"] == pytest.approx(1.0)
    assert out[0]["avoided_first_year_cm"] == pytest.approx(1.0)
    assert out[0]["rate_last5_base_cm_yr"] == pytest.approx(
        out[0]["rate_last5_policy_cm_yr"], abs=0.3)


def test_score_run_on_a_synthetic_forward_npz(tmp_path):
    from hydrophysics.twin.column_step import main

    nx, ny, T, origin = 12, 3, 48, 23
    mask = np.ones((ny, nx), dtype=bool)
    x0 = 176_000.0
    xs = x0 + (np.nonzero(mask)[1] + 0.5) * 1000.0
    t = np.arange(T) / 12.0
    rate = np.where(xs >= 182_000.0, 3.0, 1.0)                        # cm/yr, a 2 cm/yr step
    subs = (rate[:, None] * t[None, :]) / 100.0                        # metres
    fw = tmp_path / "fw.npz"
    np.savez(fw, dates=np.array([str(d)[:10] for d in
                                 np.arange("2020-01", "2024-01", dtype="datetime64[M]")]),
             origin=origin, subs_mean=np.stack([subs, subs * 0.9]).astype("float32"),
             scenario_names=np.array(["baseline", "cut10"]),
             scenarios=np.array(["baseline: baseline (no change)",
                                 "cut10: irrigation x0.9 from 2022-01"]),
             mask=mask, nx=nx, ny=ny, dx=1000.0, x0=x0, y0=2_600_000.0,
             rheology_labels=np.array(["col"]),
             forward_options=json.dumps({"column_bands": [7]}))
    cdir = tmp_path / "col"
    cdir.mkdir()
    (cdir / "vep_banded_leveling.json").write_text(json.dumps(
        {"banded": [_col()] * 7, "config": "banded", "band_lambda": 1e-3,
         "band_kind": "hat", "band_centres_km": list(range(160, 230, 10))}))
    (cdir / "stage4_column.csv").write_text(
        "config,r2_insample,r2_outoffold,rings_independent_r2\nbanded,0.66,0.65,0.1\n")
    out = tmp_path / "score"
    main(["--run", "b", str(fw), str(cdir), "--windows", "2,4", "--out", str(out),
          "--data", str(tmp_path / "no_such_dir")])
    js = json.loads((tmp_path / "score.json").read_text())
    r = js["runs"][0]
    assert r["config"] == "banded" and r["n_columns"] == 7 and r["band_lambda"] == 1e-3
    assert r["leveling_r2_outoffold"] == 0.65
    assert r["first_column_forward_cm"]["east"] > r["first_column_forward_cm"]["west"]
    for s in r["steps"]:
        assert s["hindcast_step_cm_yr"] == pytest.approx(2.0, abs=0.01)
        assert s["forecast_step_cm_yr"] == pytest.approx(2.0, abs=0.01)
    assert r["policy"][0]["scenario"] == "cut10"


# ---- the column: fit, JSON, forward loading ------------------------------------------------
def _cc():
    pytest.importorskip("torch")
    pytest.importorskip("pyproj")
    import hydrophysics.twin.calibrate_coupled as cc

    return cc


def test_banded_column_with_equal_bands_equals_one_shared_column():
    torch = pytest.importorskip("torch")
    cc = _cc()
    from hydrophysics.twin.compaction import VEPColumn

    old = (cc.N_BANDS, cc.BAND_LAMBDA)
    try:
        cc.N_BANDS, cc.BAND_LAMBDA = 5, 0.0
        model = cc._make("banded", "cpu")
    finally:
        cc.N_BANDS, cc.BAND_LAMBDA = old
    assert isinstance(model, cc._BandedColumns) and len(model) == 5
    assert float(model.penalty().detach()) == 0.0
    heads = torch.linspace(0.0, -4.0, 20).repeat(6, 4, 1)          # (n, L, T)
    x = np.linspace(160.0, 225.0, 6)
    w = torch.tensor(band_weights(_xy(*x), band_centres_km(x, 5), "hat").T,
                     dtype=torch.float32)
    shared = VEPColumn(n_sites=1)
    assert torch.allclose(cc._predict(model, heads, w), shared(heads.mean(1)), atol=1e-6)


def test_banded_penalty_pulls_neighbouring_bands_together():
    torch = pytest.importorskip("torch")
    cc = _cc()
    x = np.linspace(170.0, 200.0, 12)
    centres = band_centres_km(x, 4)
    w = torch.tensor(band_weights(_xy(*x), centres, "hat").T, dtype=torch.float32)
    heads = torch.linspace(0.0, -3.0, 24).repeat(12, 4, 1)
    # the target: west cells compact 4x more than east cells, a sharp contrast
    skv = torch.tensor(np.where(x < 185.0, 0.08, 0.02), dtype=torch.float32)
    from hydrophysics.twin.compaction import vep_compaction

    n = x.size
    obs = vep_compaction(heads.mean(1), torch.full((n,), float(np.log(1e-3))),
                         torch.log(skv), torch.full((n,), float(np.log(365.0))),
                         torch.zeros(n)).detach()
    mask = torch.ones_like(obs)
    rough = {}
    for lam in (0.0, 10.0):
        m = cc._BandedColumns(4, lam)
        cc._fit(m, heads, obs, mask, w, epochs=150, lr=0.05)
        rough[lam] = float(sum(((m.stacked(k)[1:] - m.stacked(k)[:-1]) ** 2).sum()
                               for k in ("log_ske", "log_skv", "log_tau", "h_pc0")))
        js = cc.column_json(m)
        assert len(js["banded"]) == 4 and js["band_lambda"] == lam
        assert len(cc.tau_at_ceiling(js, 24)) == 4
    assert rough[10.0] < 0.5 * rough[0.0]


def test_forward_loads_a_banded_column_json(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyproj")
    from hydrophysics.twin.forward import (
        ZonalColumn,
        blended_column,
        compaction,
        load_or_fit_vep,
        release_fast_startup,
    )

    cols = [_col(skv=0.01 * (i + 1), tau=20.0 if i == 0 else 900.0, h=3.0 if i == 0 else 1.0)
            for i in range(4)]
    centres = [170.0, 180.0, 190.0, 200.0]
    p = {"banded": cols, "band_centres_km": centres, "band_kind": "hat",
         "band_lambda": 1e-3, "config": "banded", "hpc0_guard_days": None}
    f = tmp_path / "vep_banded_leveling.json"
    f.write_text(json.dumps(p))
    xy = _xy(168.0, 181.9, 182.1, 205.0)
    with pytest.raises(ValueError, match="band_xy"):
        load_or_fit_vep(str(f), None, None, "cpu", zone_of_cell=np.zeros(4, dtype=int))
    col, meta = load_or_fit_vep(str(f), None, None, "cpu", band_xy=xy)
    assert isinstance(col, ZonalColumn) and col.zone_w.shape == (4, 4)
    heads = torch.linspace(0.0, -2.0, 12).repeat(4, 4, 1).permute(1, 0, 2)   # (L, A, T)
    s = compaction(col, heads)
    ref = blended_column(col.cols, torch.tensor(band_weights(xy, centres), dtype=torch.float64),
                         heads.mean(0).to(torch.float32)).detach().numpy()
    assert np.allclose(s, ref, atol=1e-6)
    # no step at 182 km: two cells 0.2 km apart differ by far less than across bands
    assert abs(s[1, -1] - s[2, -1]) < 0.05 * abs(s[0, -1] - s[3, -1])
    # A1 release reads the bands too; the zone blend override does not apply
    q, changed = release_fast_startup(p, 365.0)
    assert changed == [0] and q["banded"][0]["h_pc0"] == 0.0 and q["banded"][1]["h_pc0"] == 1.0
    with pytest.raises(ValueError, match="banded"):
        load_or_fit_vep(str(f), None, None, "cpu", band_xy=xy, blend_km=2.0)


def test_band_init_zonal_copies_each_zone_and_kfold_uses_the_fold_start():
    torch = pytest.importorskip("torch")
    cc = _cc()
    zonal = cc._make("zonal", "cpu")
    with torch.no_grad():
        for i, c in enumerate(zonal):
            c.log_tau.fill_(float(np.log(100.0 * (i + 1))))
            c.h_pc0.fill_(float(i))
    band_zone = np.array([2, 2, 1, 1, 0])
    m = cc.banded_from_zonal(zonal, band_zone, 1e-4)
    assert isinstance(m, cc._BandedColumns) and m.lam == 1e-4
    assert np.allclose(m.stacked("h_pc0").detach().numpy(), [2, 2, 1, 1, 0])
    # the k-fold hook: called once per fold with that fold's kept rows only
    n = 10
    heads = torch.linspace(0.0, -2.0, 8).repeat(n, 4, 1)
    obs = torch.zeros(n, 8)
    mask = torch.ones(n, 8)
    zid = torch.zeros(n, dtype=torch.long)
    seen = []

    def make(keep):
        seen.append(len(keep))
        return cc._make("shared", "cpu")

    cc.kfold_sites("shared", heads, obs, mask, zid, 2, 0.01, "cpu", n_folds=5, make=make)
    assert seen == [8] * 5


def test_hpc0_guard_holds_per_cell_between_a_slow_and_a_fast_band():
    """Review 2026-09-29: the per-band A1 guard let hat cells between a slow band with
    h_pc0 > 0 and a fast band have tau < D and h_pc0 > 0 (250 cells at 197-201 km in the
    lambda 1e-5 CPU fit). ``--band-hpc0-guard cell`` caps the slow band so no cell does."""
    torch = pytest.importorskip("torch")
    cc = _cc()
    x = np.linspace(170.0, 200.0, 3001)
    centres = [170.0, 180.0, 190.0, 200.0]
    w = band_weights(_xy(*x), centres, "hat")                             # (4, n)

    def cells(m):
        tau = np.exp(m.stacked("log_tau").detach().numpy() @ w)
        h = m.stacked("h_pc0").detach().numpy() @ w
        return int(((tau < 365.0) & (h > 1e-9)).sum())

    def model(kind="hat", h_fast=0.0):
        m = cc._BandedColumns(4, 0.0, kind=kind)
        with torch.no_grad():
            for c, tau, h in zip(m, (3000.0, 2000.0, 900.0, 30.0), (4.0, 3.0, 2.0, h_fast),
                                 strict=True):
                c.log_tau.fill_(float(np.log(tau)))
                c.h_pc0.fill_(h)
        return m

    old = (cc.HPC0_GUARD_DAYS, cc.BAND_HPC0_GUARD)
    try:
        cc.HPC0_GUARD_DAYS, cc.BAND_HPC0_GUARD = None, "cell"
        m = model()
        cc._guard_hpc0_banded(m)                                          # off: no change
        assert cells(m) > 0 and float(m[2].h_pc0) == 2.0
        cc.HPC0_GUARD_DAYS, cc.BAND_HPC0_GUARD = 365.0, "band"            # the default
        m = model()
        cc._guard_hpc0_banded(m)
        assert cells(m) > 0 and float(m[2].h_pc0) == 2.0
        cc.BAND_HPC0_GUARD = "cell"
        m = model()
        with torch.no_grad():
            cc._guard_hpc0_banded(m)
        assert cells(m) == 0 and float(m[2].h_pc0) == 0.0
        assert [float(m[k].h_pc0) for k in (0, 1)] == [4.0, 3.0]          # far bands kept
        # a negative fast offset leaves room: the cap is positive and tight
        m = model(h_fast=-1.0)
        with torch.no_grad():
            cc._guard_hpc0_banded(m)
        cap = float(m[2].h_pc0)
        assert 0.0 < cap < 2.0 and cells(m) == 0
        with torch.no_grad():
            m[2].h_pc0.fill_(cap + 0.05)
        assert cells(m) > 0
        m = model(kind="step")                                            # no mixing
        with torch.no_grad():
            cc._guard_hpc0_banded(m)
        assert float(m[2].h_pc0) == 2.0
    finally:
        cc.HPC0_GUARD_DAYS, cc.BAND_HPC0_GUARD = old
