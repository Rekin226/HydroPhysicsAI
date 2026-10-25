"""Recorded-pumping arm of a frozen forward run (N2 draft hook, 2026-10-24): a monthly
multiplier on the metered pumping past the origin, everything else frozen. Synthetic
inputs only; CPU."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.recorded_pumping import (
    from_series,
    load_recorded_multiplier,
    parse_recorded_spec,
)
from hydrophysics.twin.scenario import BASELINE, PumpingScenario

# ---------------------------------------------------------------------------------------
# the multiplier on its own (numpy only)
# ---------------------------------------------------------------------------------------
FUT = pd.date_range("2023-01-01", periods=12, freq="MS")      # projection months


def _E(A=5, T=12, seed=0):
    rng = np.random.default_rng(seed)
    return {"irrigation": rng.uniform(50.0, 150.0, (A, T)),
            "aquaculture": rng.uniform(5.0, 15.0, (A, T))}


def test_multiplier_one_is_bit_identical_to_the_baseline():
    E = _E()
    rec = from_series(FUT.strftime("%Y-%m"), {"multiplier": np.ones(12)})
    a = BASELINE.apply(E, FUT)
    b = PumpingScenario("PR", recorded=rec).apply(E, FUT)
    assert a.dtype == b.dtype and a.tobytes() == b.tobytes()


def test_point_nine_lowers_abstraction_in_exactly_the_targeted_months():
    E = _E()
    months = ["2023-03", "2023-04", "2023-05"]
    rec = from_series(months, {"multiplier": [0.9, 0.9, 0.9]})
    base = BASELINE.apply(E, FUT)
    pr = PumpingScenario("PR", recorded=rec).apply(E, FUT)
    hit = FUT.isin(pd.DatetimeIndex(months))
    assert hit.sum() == 3
    # untouched months: bit for bit; targeted months: each class x 0.9, then summed
    assert pr[:, ~hit].tobytes() == base[:, ~hit].tobytes()
    expect = E["irrigation"][:, hit] * 0.9 + E["aquaculture"][:, hit] * 0.9
    assert np.array_equal(pr[:, hit], expect)
    assert (pr[:, hit] < base[:, hit]).all()


def test_gaps_and_months_after_the_last_row_keep_one():
    E = _E()
    rec = from_series(["2023-02", "2023-05"], {"multiplier": [1.2, 0.8]})
    pr = PumpingScenario("PR", recorded=rec).apply(E, FUT)
    base = BASELINE.apply(E, FUT)
    changed = np.flatnonzero((pr != base).any(axis=0))
    assert changed.tolist() == [1, 4]


def test_per_class_columns_scale_only_their_class():
    E = _E()
    rec = from_series(FUT.strftime("%Y-%m"), {"irrigation": np.full(12, 0.5)})
    pr = PumpingScenario("PR", recorded=rec).apply(E, FUT)
    assert np.array_equal(pr, E["irrigation"] * 0.5 + E["aquaculture"])
    # per-class efficiencies: forward applies the scenario class by class
    aq = PumpingScenario("PR", recorded=rec).apply({"aquaculture": E["aquaculture"]}, FUT)
    assert aq.tobytes() == E["aquaculture"].tobytes()


def test_recorded_then_policy_factors_compose():
    E = _E()
    rec = from_series(["2023-01"], {"multiplier": [0.9]})
    pr = PumpingScenario("PR", factors={"irrigation": 0.5}, recorded=rec).apply(E, FUT)
    assert np.allclose(pr[:, 0], E["irrigation"][:, 0] * 0.9 * 0.5
                       + E["aquaculture"][:, 0] * 0.9)


def test_county_scope_touches_only_that_countys_cells():
    E = _E()
    ch = np.array([True, True, False, False, False])
    rec = from_series(["2023-01"], {"multiplier": [0.5]}, county="yunlin").bind(5, ch)
    pr = PumpingScenario("PR", recorded=rec).apply(E, FUT)
    base = BASELINE.apply(E, FUT)
    assert pr[ch].tobytes() == base[ch].tobytes()
    assert np.array_equal(pr[~ch, 0], (E["irrigation"][~ch, 0] * 0.5
                                       + E["aquaculture"][~ch, 0] * 0.5))
    with pytest.raises(ValueError, match="county flags"):
        from_series(["2023-01"], {"multiplier": [0.5]}, county="yunlin").bind(5)
    with pytest.raises(ValueError, match="bound to the model grid"):
        PumpingScenario("PR", recorded=from_series(["2023-01"], {"multiplier": [0.5]},
                                                   county="yunlin")).apply(E, FUT)


def test_per_cell_field_applies_cell_by_cell():
    E = _E()
    f = np.linspace(0.5, 1.5, 5)[None, :]                          # (1 month, 5 cells)
    rec = from_series(["2023-06"], {"multiplier": f}).bind(5)
    pr = PumpingScenario("PR", recorded=rec).apply(E, FUT)
    tot = E["irrigation"][:, 5] + E["aquaculture"][:, 5]
    assert np.allclose(pr[:, 5], tot * f[0])


# ---------------------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------------------
def test_refuses_rows_before_the_start():
    with pytest.raises(ValueError, match="before the start month"):
        from_series(["2026-10", "2026-11"], {"multiplier": [1.1, 1.2]}, start="2026-11")


def test_refuses_a_start_before_the_projection():
    E = _E()
    rec = from_series(["2022-12", "2023-01"], {"multiplier": [1.1, 1.2]})
    with pytest.raises(ValueError, match="before the first projected month"):
        PumpingScenario("PR", recorded=rec).apply(E, FUT)
    # a later start is fine even if its months are a subset
    later = from_series(["2023-06"], {"multiplier": [1.1]}, start="2023-03")
    PumpingScenario("PR", recorded=later).apply(E, FUT)


def test_refuses_a_month_past_the_horizon_or_off_the_monthly_grid():
    E = _E()
    past = from_series(["2023-12", "2024-01"], {"multiplier": [1.0, 0.9]})
    with pytest.raises(ValueError, match="not on the projection's monthly grid"):
        PumpingScenario("PR", recorded=past).apply(E, FUT)
    with pytest.raises(ValueError, match="not on the monthly grid"):
        from_series(["2023-01-15"], {"multiplier": [0.9]})
    with pytest.raises(ValueError, match="duplicate"):
        from_series(["2023-01", "2023-01"], {"multiplier": [0.9, 0.8]})
    with pytest.raises(ValueError, match="increasing"):
        from_series(["2023-02", "2023-01"], {"multiplier": [0.9, 0.8]}, start="2023-01")


def test_refuses_a_series_for_another_grid():
    E = _E(A=5)
    with pytest.raises(ValueError, match="wrong grid"):
        from_series(["2023-01"], {"multiplier": np.ones((1, 7))}).bind(5)
    with pytest.raises(ValueError, match="wrong grid"):
        from_series(["2023-01"], {"multiplier": [0.9]}, n_active=7).bind(5)
    with pytest.raises(ValueError, match="wrong grid"):
        from_series(["2023-01"], {"multiplier": np.ones((1, 7))}, n_active=5)
    bound = from_series(["2023-01"], {"multiplier": [0.9]}).bind(7)
    with pytest.raises(ValueError, match="wrong grid"):
        PumpingScenario("PR", recorded=bound).apply(E, FUT)
    with pytest.raises(ValueError, match="wrong grid"):
        from_series(["2023-01"], {"multiplier": [0.9]}, county="yunlin").bind(
            5, np.zeros(4, bool))


def test_refuses_bad_values_and_columns():
    with pytest.raises(ValueError, match="non-finite"):
        from_series(["2023-01"], {"multiplier": [np.nan]})
    with pytest.raises(ValueError, match="negative"):
        from_series(["2023-01"], {"multiplier": [-0.1]})
    with pytest.raises(ValueError, match="not both"):
        from_series(["2023-01"], {"multiplier": [1.0], "irrigation": [0.9]})
    with pytest.raises(ValueError, match="unknown class"):
        from_series(["2023-01"], {"golf": [0.9]})
    with pytest.raises(ValueError, match="county must be"):
        from_series(["2023-01"], {"multiplier": [0.9]}, county="taipei")


# ---------------------------------------------------------------------------------------
# files and the CLI spec
# ---------------------------------------------------------------------------------------
def test_csv_and_npz_load_with_hash_and_start(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("month,multiplier\n2026-11,1.1\n2026-12,0.95\n")
    rec = load_recorded_multiplier(str(p))
    assert rec.start == pd.Timestamp("2026-11-01") and len(rec.months) == 2
    assert len(rec.sha256) == 64 and rec.record()["n_months"] == 2
    with pytest.raises(ValueError, match="before the start month"):
        load_recorded_multiplier(str(p), start="2026-12")
    q = tmp_path / "r.npz"
    np.savez(q, month=np.array(["2023-01", "2023-02"]), irrigation=np.array([0.9, 1.0]),
             start=np.array("2023-01"), n_active=np.array(5))
    rn = load_recorded_multiplier(str(q))
    assert rn.n_active == 5 and set(rn.values) == {"irrigation"}
    with pytest.raises(ValueError, match="wrong grid"):
        rn.bind(6)
    bad = tmp_path / "bad.csv"
    bad.write_text("month,golf\n2023-01,0.9\n")
    with pytest.raises(ValueError, match="unknown column"):
        load_recorded_multiplier(str(bad))


def test_parse_recorded_spec():
    assert parse_recorded_spec("PR=a/b.csv") == ("PR", "a/b.csv", None, None)
    assert parse_recorded_spec("PR=a/b.csv@2026-11#yunlin") == ("PR", "a/b.csv", "2026-11",
                                                                 "yunlin")
    with pytest.raises(ValueError):
        parse_recorded_spec("no-equals")


# ---------------------------------------------------------------------------------------
# through the frozen forward run (torch)
# ---------------------------------------------------------------------------------------
def _inputs(T=24, nx=6, ny=3):
    import torch

    from hydrophysics.twin.grid import FanGrid
    from hydrophysics.twin.inputs import TwinInputs

    g = FanGrid(nx=nx, ny=ny, dx=1000.0, x0=200_000.0, y0=0.0,
                mask=np.ones((ny, nx), dtype=bool))
    A = g.n_active
    dates = pd.date_range("2021-01-01", periods=T, freq="MS")     # origin 2022-12
    well_xy = np.array([[200_500.0, 500.0], [204_500.0, 1500.0], [205_500.0, 2500.0]])
    obs_idx = np.array([g.active_index(x, y) for x, y in well_xy])
    obs_layer = np.array([0, 1, 1])
    obs_h = np.tile(np.array([[3.0], [4.0], [6.0]]), (1, T)) + 0.1 * np.arange(T)
    E = {"irrigation": np.full((A, T), 100.0), "aquaculture": np.full((A, T), 10.0)}
    return TwinInputs(grid=g, hf=None, dates=dates, obs_h=obs_h,
                      obs_h_filled=obs_h.copy(), obs_idx=obs_idx,
                      obs_layer=obs_layer, well_xy=well_xy, sids=["a", "b", "c"],
                      ground_elev=torch.full((A,), 12.0, dtype=torch.float64),
                      E_by_class=E,
                      recharge_field=torch.full((A, T), 1e-4, dtype=torch.float64))


def _theta_file(tmp_path):
    theta = {"log_T_proximal": [np.log(500.0)], "log_S_proximal": [np.log(1e-3)],
             "log_T_mid": [np.log(300.0)] * 4, "log_S_mid": [np.log(1e-3)] * 4,
             "log_T_distal": [np.log(100.0)] * 4, "log_S_distal": [np.log(1e-3)] * 4,
             "log_L_mid": [np.log(1e-4)] * 3, "log_L_distal": [np.log(1e-4)] * 3,
             "log_eta": np.log(0.3), "log_head_extra": np.log(30.0),
             "recharge_frac_logit": 0.0,
             "log_C_coast": [np.log(50.0)] * 4, "log_C_apex": [np.log(50.0)]}
    meta = {"param_mode": "zonal", "boundaries": "coast-apex", "zone_boundaries": "205,203",
            "meter_filter": "none", "eta_classes": None}
    p = tmp_path / "stage3_theta.json"
    p.write_text(json.dumps({"theta": theta, "meta": meta}))
    return str(p)


def _run(tmp_path, scenarios, horizon=6, **kw):
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyproj")
    from hydrophysics.twin.compaction import VEPColumn
    from hydrophysics.twin.forward import load_members, run

    inp = _inputs()
    members = load_members([_theta_file(tmp_path)])
    col = VEPColumn(n_sites=1, dt_days=30.0)
    return inp, run(inp, members, scenarios, horizon=horizon, gain=1.0, ic_members=0,
                    ic_sigma=0.0, seed=0, col=col, device="cpu", log=lambda *_: None,
                    restart_taper_km=5.0, column_heads="free", **kw)


def test_frozen_run_with_multiplier_one_is_bit_identical(tmp_path):
    months = pd.date_range("2023-01-01", periods=6, freq="MS").strftime("%Y-%m")
    ones = PumpingScenario("PR", recorded=from_series(months, {"multiplier": np.ones(6)}))
    _, frozen = _run(tmp_path, [(BASELINE, 1.0)])
    _, both = _run(tmp_path, [(BASELINE, 1.0), (ones, 1.0)])
    # the zero-response arm is untouched by adding the arm, and the arm at 1 equals it
    assert frozen["heads_mean"][0].tobytes() == both["heads_mean"][0].tobytes()
    assert frozen["subs_mean"][0].tobytes() == both["subs_mean"][0].tobytes()
    assert both["heads_mean"][1].tobytes() == both["heads_mean"][0].tobytes()
    assert both["subs_mean"][1].tobytes() == both["subs_mean"][0].tobytes()


def test_frozen_run_point_nine_changes_heads_only_from_its_start(tmp_path):
    rec = from_series(["2023-03", "2023-04"], {"multiplier": [0.9, 0.9]})
    inp, res = _run(tmp_path, [(BASELINE, 1.0), (PumpingScenario("PR", recorded=rec), 1.0)])
    T = len(inp.dates)
    h0, h1 = res["heads_mean"][0], res["heads_mean"][1]
    # hindcast, origin and the first two projected months (2023-01, 2023-02): bit-identical
    k = T + 2                       # index of 2023-03 in hindcast + projection
    assert res["dates"][k] == pd.Timestamp("2023-03-01")
    assert h1[..., :k].tobytes() == h0[..., :k].tobytes()
    # less pumping -> higher heads in the pumped layer from the first targeted month on
    assert (h1[1, :, k] > h0[1, :, k]).all()
    assert (h1[1, :, k:] >= h0[1, :, k:] - 1e-12).all()


def test_frozen_run_refuses_a_series_that_does_not_fit_before_any_rollout(tmp_path):
    early = from_series(["2022-12"], {"multiplier": [0.9]})
    with pytest.raises(ValueError, match="before the first projected month"):
        _run(tmp_path, [(BASELINE, 1.0), (PumpingScenario("PR", recorded=early), 1.0)])
    late = from_series(["2023-07"], {"multiplier": [0.9]})
    with pytest.raises(ValueError, match="monthly grid"):
        _run(tmp_path, [(BASELINE, 1.0), (PumpingScenario("PR", recorded=late), 1.0)])
    other = from_series(["2023-01"], {"multiplier": [0.9]}).bind(99)
    with pytest.raises(ValueError, match="wrong grid"):
        _run(tmp_path, [(BASELINE, 1.0), (PumpingScenario("PR", recorded=other), 1.0)])


def test_frozen_run_per_class_efficiencies_take_the_class_multiplier(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pyproj")
    from hydrophysics.twin.forward import future_forcing

    inp = _inputs()
    rec = from_series(["2023-01"], {"irrigation": [0.9]})
    scen = PumpingScenario("PR", recorded=rec)
    E3, _, _ = future_forcing(inp, scen, horizon=3, eta_classes=["irrigation", "aquaculture"])
    B3, _, _ = future_forcing(inp, BASELINE, horizon=3,
                              eta_classes=["irrigation", "aquaculture"])
    assert np.array_equal(E3[0, :, 0].numpy(), B3[0, :, 0].numpy() * 0.9)
    assert E3[0, :, 1:].numpy().tobytes() == B3[0, :, 1:].numpy().tobytes()
    assert E3[1].numpy().tobytes() == B3[1].numpy().tobytes()
