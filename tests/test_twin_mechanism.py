"""Track 4: rebound vs slowed creep (``twin.mechanism``) and the opt-in aquitard column.

The regression that decides between the two mechanisms must return ``k = 0`` on a pure
rebound (elastic compaction plus a head-independent creep) and the true ``k`` on a rate
that depends on the head level; the aquitard term must be zero at t=0, never swell, and
slow when heads rise; and the default column must be the historical one exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.mechanism import (
    annual_means,
    band_compaction,
    fe_ols,
    rate_table,
    seasonal_slope,
    t_ppf,
    virgin_by_year,
)


def _dates(n=132, start="2012-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def _synthetic_sites(k_level: float, n_sites: int = 12, seed: int = 0, noise: float = 5e-4):
    """Drawdown with a seasonal cycle and random multi-year swings; compaction elastic
    (5 mm/m) plus a creep of 2 cm/yr, plus ``k_level`` m/yr per metre of drawdown level."""
    rng = np.random.default_rng(seed)
    d = _dates()
    C, D = {}, {}
    for s in range(n_sites):
        yearly = np.repeat(rng.normal(0.0, 1.5, len(d) // 12 + 1), 12)[:len(d)]
        yearly = pd.Series(yearly).rolling(3, min_periods=1).mean().to_numpy()
        dd = 1.0 * np.sin(2 * np.pi * np.arange(len(d)) / 12) + yearly
        t = np.arange(len(d)) / 12.0
        creep = 0.02 * t + k_level * np.cumsum(dd - dd.mean()) / 12.0
        c = 0.005 * dd + creep + rng.normal(0.0, noise, len(d))
        C[f"s{s}"] = pd.Series(c, index=d)
        D[f"s{s}"] = pd.Series(dd, index=d)
    return C, D


def test_pure_rebound_gives_no_level_effect():
    C, D = _synthetic_sites(k_level=0.0)
    r = fe_ols(rate_table(C, D), "R", ["dD0", "lev1"])
    assert r["coef"]["dD0"] == pytest.approx(0.005, abs=1e-3)
    assert abs(r["coef"]["lev1"]) < 1e-3
    lo, hi = r["ci95"]["lev1"]
    assert lo < 0.0 < hi


def test_level_dependent_rate_is_recovered():
    k = 0.006
    C, D = _synthetic_sites(k_level=k)
    r = fe_ols(rate_table(C, D), "R", ["dD0", "lev1"])
    # R_y = e dD0 + k * level_y = (e + k) dD0 + k lev1 (annual means)
    assert r["coef"]["lev1"] == pytest.approx(k, rel=0.15)
    assert r["ci95"]["lev1"][0] > 0.0
    ry = fe_ols(rate_table(C, D), "R", ["dD0", "lev1"], year_fe=True)
    assert ry["coef"]["lev1"] == pytest.approx(k, rel=0.25)
    rt = fe_ols(rate_table(C, D), "R", ["dD0", "lev1"], trend=True)
    assert rt["n"] == r["n"] and rt["trend"]


def test_rate_table_skips_gaps():
    C, D = _synthetic_sites(k_level=0.0, n_sites=1)
    c = C["s0"].copy()
    c[c.index.year == 2016] = np.nan                  # a year without surveys
    tab = rate_table({"s0": c}, D)
    assert 2016 not in set(tab.year) and 2017 not in set(tab.year)
    assert tab.year.min() == 2014                      # needs heads two years back
    assert 2016 not in annual_means(c).dropna().index


def test_seasonal_slope_recovers_elastic():
    d = _dates()
    rng = np.random.default_rng(1)
    dd = pd.Series(2.0 * np.sin(2 * np.pi * np.arange(len(d)) / 12), index=d)
    c = 0.004 * dd + 0.02 * np.arange(len(d)) / 12 + rng.normal(0, 1e-4, len(d))
    r = seasonal_slope(c, dd)
    assert r["best_lag"] == 0
    assert r["slope"] == pytest.approx(0.004, rel=0.05)


def test_virgin_by_year():
    d = _dates(36)
    D = pd.Series(np.r_[np.full(12, 1.0), np.full(12, 3.0), np.full(12, 2.0)], index=d)
    v = virgin_by_year(D)
    assert list(v.index) == [2013, 2014]
    assert v[2013] == pytest.approx(2.0) and v[2014] == 0.0


def test_band_compaction_partitions_total():
    d = pd.date_range("2014-01-15", periods=48, freq="30D")
    depth = np.array([2.0, 50.0, 110.0, 190.0, 260.0])
    shorten = np.array([0.0, 0.002, 0.005, 0.009, 0.012])          # m/yr, grows with depth
    t = np.arange(len(d)) / 12.0
    pos = pd.DataFrame(depth[None, :] - shorten[None, :] * t[:, None], index=d,
                       columns=[f"NO{i + 1}" for i in range(len(depth))])
    out = band_compaction(pos)
    assert (out["total"].dropna() >= -1e-12).all()
    bands = out.drop(columns="total")
    assert np.allclose(bands.sum(axis=1), out["total"], atol=1e-12)
    assert out["total"].iloc[-1] == pytest.approx(0.012 * t[-1], rel=0.05)


def test_band_compaction_drops_ring_lost_for_a_year():
    d = pd.date_range("2014-01-15", periods=48, freq="30D")
    pos = pd.DataFrame({"NO1": 2.0, "NO2": 100.0, "NO3": 200.0, "NO4": 300.0}, index=d)
    pos.loc[pos.index.year == 2016, "NO4"] = np.nan                 # 25 % gone, one year
    out = band_compaction(pos)
    assert out["total"].notna().all()


def test_t_ppf_close_to_table():
    assert t_ppf(0.975, 13) == pytest.approx(2.160, abs=2e-3)
    assert t_ppf(0.80, 13) == pytest.approx(0.870, abs=2e-3)


# ---------------------------------------------------------------------------------------
# the opt-in aquitard column
# ---------------------------------------------------------------------------------------
def test_default_column_is_historical():
    torch = pytest.importorskip("torch")
    from hydrophysics.twin.compaction import VEPColumn, column_compaction, vep_compaction

    col = VEPColumn(n_sites=2)
    assert [n for n, _ in col.named_parameters()] == ["log_ske", "log_skv", "log_tau", "h_pc0"]
    assert col.param_keys() == ("log_ske", "log_skv", "log_tau", "h_pc0")
    h = -torch.linspace(0, 5, 40).repeat(2, 1)
    ref = vep_compaction(h, col.log_ske, col.log_skv, col.log_tau, col.h_pc0, 30.0)
    assert torch.equal(col(h), ref)
    p = {k: float(getattr(col, k).detach()[0]) for k in col.param_keys()}
    assert torch.allclose(column_compaction(h[:1].double(), p), ref[:1].double(), atol=1e-6)
    with pytest.raises(ValueError):
        VEPColumn(n_sites=1, creep="bjerrum")


def test_aquitard_term_physics():
    torch = pytest.importorskip("torch")
    from hydrophysics.twin.compaction import aquitard_compaction

    lt = torch.log(torch.tensor([3652.5], dtype=torch.float64))
    ls = torch.log(torch.tensor([0.1], dtype=torch.float64))
    flat = torch.zeros(1, 240, dtype=torch.float64)
    c = aquitard_compaction(flat, ls, lt, torch.tensor([4.0], dtype=torch.float64))[0]
    assert c[0] == 0.0
    assert torch.all(torch.diff(c) >= 0)                            # never swells
    assert float(c[-1]) < 0.1 * 4.0                                 # bounded by Ska*h_a0
    # rate ~ Ska (u - h) / tau_a at the start
    assert float(c[1]) == pytest.approx(0.1 * 4.0 * (1 - np.exp(-30 / 3652.5)), rel=1e-6)
    # a sustained 1 m rise slows the rate (not just a step); 5 m (above u) stops it
    up1 = flat.clone()
    up1[:, 60:] += 1.0
    up5 = flat.clone()
    up5[:, 60:] += 5.0
    c1 = aquitard_compaction(up1, ls, lt, torch.tensor([4.0], dtype=torch.float64))[0]
    c5 = aquitard_compaction(up5, ls, lt, torch.tensor([4.0], dtype=torch.float64))[0]
    rate0 = float(c[120] - c[108])
    rate1 = float(c1[120] - c1[108])
    assert rate1 < 0.8 * rate0
    assert float(c5[120] - c5[108]) == pytest.approx(0.0, abs=1e-12)


def test_aquitard_column_adds_to_vep_and_trains():
    torch = pytest.importorskip("torch")
    from hydrophysics.twin.compaction import VEPColumn, column_compaction

    col = VEPColumn(n_sites=1, creep="aquitard").double()
    assert col.param_keys()[-3:] == ("log_ska", "log_tau_a", "h_a0")
    h = -torch.linspace(0, 3, 60, dtype=torch.float64)[None, :]
    base = VEPColumn(n_sites=1).double()
    # h_a0 = 0 at init: the interbeds lag the falling head, so the term only adds
    assert torch.all(col(h) >= base(h) - 1e-12)
    with torch.no_grad():
        col.h_a0.fill_(3.0)
    p = {k: float(getattr(col, k).detach()[0]) for k in col.param_keys()}
    assert torch.allclose(column_compaction(h, p), col(h), atol=1e-12)
    loss = col(h)[:, -1].sum()
    loss.backward()
    for k in ("log_ska", "log_tau_a", "h_a0"):
        assert getattr(col, k).grad is not None and torch.isfinite(getattr(col, k).grad).all()


def test_coupled_twin_carries_creep_option():
    torch = pytest.importorskip("torch")
    from hydrophysics.twin.compaction import VEPColumn
    from hydrophysics.twin.coupled import CoupledTwin
    from hydrophysics.twin.grid import FanGrid

    g = FanGrid(nx=3, ny=3, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((3, 3), dtype=bool))
    m = CoupledTwin(g, n_layers=4)
    assert m.column.creep == "vep"
    m2 = CoupledTwin(g, n_layers=4, creep="aquitard")
    fitted = VEPColumn(n_sites=1, creep="aquitard")
    with torch.no_grad():
        fitted.h_a0.fill_(2.5)
        fitted.log_ske.fill_(-7.0)
    m2.load_column(fitted)
    assert float(m2.column.h_a0.detach()) == pytest.approx(2.5)
    m.load_column(fitted)                                   # a VEP twin takes the VEP part
    assert float(m.column.log_ske.detach()) == pytest.approx(-7.0)


def test_step_response_separates_mechanisms():
    pytest.importorskip("torch")
    import math

    from hydrophysics.twin.mechanism import step_response, step_summary

    drv = np.zeros((1, 180))
    vep = {"log_ske": math.log(5e-3), "log_skv": math.log(2e-2), "log_tau": math.log(3650.0),
           "h_pc0": 2.0}
    s_vep = step_summary(step_response(vep, drv, 60), 60)
    assert s_vep["rebound_first_year_m"] < 0
    assert max(abs(r) for r in s_vep["rate_change_m_per_yr"][1:]) < 1e-9   # pure rebound


# ---------------------------------------------------------------------------------------
# the regression engine and the leveling side
# ---------------------------------------------------------------------------------------
def test_fe_ols_matches_dummy_regression():
    """Partialling the fixed effects out (FWL) gives the dummy regression's slopes."""
    rng = np.random.default_rng(3)
    n_s, n_y = 8, 9
    df = pd.DataFrame([{"site": f"s{s}", "year": 2012 + y, "x1": rng.normal(),
                        "x2": rng.normal()} for s in range(n_s) for y in range(n_y)])
    df["R"] = (0.3 * df.x1 - 0.2 * df.x2 + df.site.str[1:].astype(int) * 0.1
               + 0.05 * (df.year - 2016) * (df.site == "s1") + rng.normal(0, 0.01, len(df)))
    for kw in ({}, {"trend": True}, {"year_fe": True}):
        r = fe_ols(df, "R", ["x1", "x2"], **kw)
        cols = [df.x1, df.x2] + [(df.site == g).astype(float) for g in sorted(df.site.unique())]
        if kw.get("trend"):
            cols += [(df.site == g) * (df.year - df.year[df.site == g].mean())
                     for g in sorted(df.site.unique())]
        if kw.get("year_fe"):
            cols += [(df.year == v).astype(float) for v in sorted(df.year.unique())[1:]]
        beta = np.linalg.lstsq(np.column_stack(cols), df.R.to_numpy(), rcond=None)[0]
        assert r["coef"]["x1"] == pytest.approx(beta[0], abs=1e-10)
        assert r["coef"]["x2"] == pytest.approx(beta[1], abs=1e-10)
    with pytest.raises(ValueError):
        fe_ols(df.assign(cl=np.arange(len(df)) % 2), "R", ["x1"], cluster="cl")


def _nest(phase=0.0, period=41.0):
    d = pd.date_range("2012-01-31", "2022-12-31", freq="ME")
    t = np.arange(len(d))
    return pd.Series(np.sin(2 * np.pi * t / 12)
                     + 0.8 * np.sin(2 * np.pi * t / period + phase), index=d)


def test_leveling_table_and_sustained_effect():
    from hydrophysics.twin.mechanism import leveling_table

    D = _nest()
    nests = {"n1": ((0.0, 0.0), D), "n2": ((10000.0, 0.0), _nest(1.3, 29.0))}
    surveys = pd.to_datetime([f"{y}-05-15" for y in range(2013, 2023)])
    sub, bxy = {}, {}
    for i, (x, nid) in enumerate([(100.0, "n1"), (400.0, "n1"), (10200.0, "n2")]):
        Dn = nests[nid][1]
        # rate follows the interval-mean level (persistent), plus an elastic step
        levels, s = [], [0.0]
        for t0, t1 in zip(surveys[:-1], surveys[1:], strict=True):
            m0, m1 = t0 + pd.offsets.MonthEnd(0), t1 + pd.offsets.MonthEnd(0)
            lv = Dn[(Dn.index > m0) & (Dn.index <= m1)].mean() - Dn.mean()
            gap = (t1 - t0).days / 365.25
            s.append(s[-1] + gap * (0.02 + 0.005 * lv + 0.002 * (Dn[m1] - Dn[m0])))
            levels.append(lv)
        sub[f"b{i}"] = pd.Series(s, index=surveys)
        bxy[f"b{i}"] = (x, 0.0)
    sub["far"] = sub["b0"]
    bxy["far"] = (5000.0, 5000.0)                       # out of range of every nest
    tab = leveling_table(sub, bxy, nests)
    assert set(tab.site) == {"b0", "b1", "b2"} and set(tab.nest) == {"n1", "n2"}
    assert {"R", "dDe", "levI", "levP", "dLevP"} <= set(tab.columns)
    r = fe_ols(tab.dropna(), "R", ["dDe", "levI", "dLevP"], cluster="nest")
    assert r["coef"]["levI"] == pytest.approx(0.005, rel=0.05)    # the sustained effect
    assert r["coef"]["dDe"] == pytest.approx(0.002, rel=0.05)


def test_rate_anomaly_r2_and_agreement():
    from hydrophysics.twin.mechanism import rate_anomaly_agreement, rate_anomaly_r2

    C, D = _synthetic_sites(k_level=0.004, n_sites=4)
    assert rate_anomaly_r2(C, C) == pytest.approx(1.0)
    shifted = {k: v + 0.01 * np.arange(len(v)) / 12 for k, v in C.items()}   # +1 cm/yr
    assert rate_anomaly_r2(shifted, C) == pytest.approx(1.0)       # site mean removed
    tab = rate_table(C, D)
    a = rate_anomaly_agreement(tab.assign(R=2 * tab.R), tab)
    assert a["corr"] == pytest.approx(1.0) and a["slope_model_per_obs"] == pytest.approx(2.0)


def test_step_response_aquitard_keeps_slowing():
    pytest.importorskip("torch")
    import math

    from hydrophysics.twin.mechanism import step_response, step_summary

    drv = np.zeros((1, 180))
    p = {"log_ske": math.log(5e-3), "log_skv": math.log(1e-5), "log_tau": math.log(365.0),
         "h_pc0": 0.0, "log_ska": math.log(0.1), "log_tau_a": math.log(7305.0), "h_a0": 4.0}
    s = step_summary(step_response(p, drv, 60), 60)
    assert s["rebound_first_year_m"] < 0
    assert all(r < -1e-4 for r in s["rate_change_m_per_yr"][1:4])    # slowed creep


def test_matched_compare_difference_is_the_gap():
    """The difference regression's held-level effect equals leveling minus ring."""
    from hydrophysics.twin.mechanism import matched_compare

    rng = np.random.default_rng(3)
    rows = []
    for nest in range(8):
        for bm in range(3):
            for yr in range(2014, 2022):
                dde, levi, levp = rng.normal(0, 1, 3)
                ring = 0.02 + 0.004 * levi + 0.002 * dde + rng.normal(0, 1e-3)
                lev = ring + 0.001 * dde + rng.normal(0, 1e-3)
                rows.append({"site": f"{nest}|{bm}", "nest": str(nest), "year": yr,
                             "R_leveling": lev, "R_ring": ring, "dDe": dde, "levI": levi,
                             "levP": levp, "dLevP": levp - levi})
    out = matched_compare(pd.DataFrame(rows))
    assert out["corr_raw"] > 0.8
    d = out["R_diff_sustained"]["coef"]["levI"]
    gap = (out["R_leveling_sustained"]["coef"]["levI"]
           - out["R_ring_sustained"]["coef"]["levI"])
    assert d == pytest.approx(gap, abs=1e-12)
    assert abs(d) < 1e-3


def test_head_regime_counts_new_minima():
    from hydrophysics.twin.mechanism import head_regime

    drv = np.array([[5.0, 3.0, 4.0, 4.0, 2.5], [5.0, 3.0, 4.0, 4.5, 4.0]])
    r = head_regime(drv, origin=2, t_pol=3)
    assert r["n_rings_new_minima"] == 1
    assert r["proj_min_minus_record_min_m"] == pytest.approx([-0.5, 1.0])
