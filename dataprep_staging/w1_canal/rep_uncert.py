"""Repaired 2026 uncertainty of surface_water_v4 (W1 repair round): error model, members and
the coverage check. Settings and the rationale are in rep_config.py.

Functions only; rep_build.py runs them after the A1 declaration W1.ext2026_uncertainty.r1.
Nothing here reads an observation except through the arguments it is given.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from w1_canal import diversion
from w1_canal import ext_model as M
from w1_canal import rep_config as RC
from w1_canal.config import DISTRICTS

NAME = {"chc": "changhua", "yl": "yunlin"}


# ----------------------------------------------------------------------------- chain folds
def chain_folds(base: dict, P: pd.Series, eval_years: list[int]) -> tuple[dict, dict]:
    """single[t]: chain for year t fitted without t; double[(t, y)]: chain for year y fitted
    without t and y (t, y evaluation years, t != y). Same chain as ext_model.chain_test
    (nested spec selection, beta and scales refitted, R of the previous year)."""
    years = list(RC.FIT_YEARS)
    dd_all = diversion.make_daily(base["I"], base["jiji"], years)
    single, double = {}, {}
    for t in eval_years:
        single[t] = M.chain_year(base, P, t, [x for x in years if x != t], dd_all=dd_all)
        for y in eval_years:
            if y != t:
                double[(t, y)] = M.chain_year(base, P, y, [x for x in years if x not in (t, y)],
                                              dd_all=dd_all)
    return single, double


def structural_alt_2026(base: dict, P: pd.Series) -> dict:
    """The 2026 chain with the winter weight set to WINTER_ALT (structural alternative for the
    winter-weight-0 months); everything else as the shipped 2026 chain."""
    alt = dict(base)
    ff = dict(base["final_fit"])
    ff["winter"] = RC.WINTER_ALT
    alt["final_fit"] = ff
    return M.chain_year(alt, P, RC.EXT_YEAR, list(RC.FIT_YEARS), R_year=M.E.R_EXT_FROM_YEAR)


# ----------------------------------------------------------------------------- residuals
def ss_draws(base: dict) -> tuple[dict, pd.DatetimeIndex]:
    dr = base["draws"]
    dates = pd.to_datetime(dr["dates"])
    return {"chc": np.asarray(dr["draws_changhua"], float),
            "yl": np.asarray(dr["draws_yunlin"], float)}, dates


def residual_pool(preds: dict, ss: dict, dates: pd.DatetimeIndex, pool_years: list[int]
                  ) -> dict:
    """r[d]: (Y, K, 12) residual trajectories D_k(y, m) - c_y(m) of the pool years."""
    out = {}
    for d in DISTRICTS:
        blocks = []
        for y in pool_years:
            sel = np.asarray(dates.year == y)
            if sel.sum() != 12:
                raise ValueError(f"full/v1 draws do not cover {y}")
            c = np.asarray(preds[y][d], float)
            blocks.append(ss[d][:, sel] - c[None, :])
        out[d] = np.stack(blocks)
    return out


def params(r: np.ndarray) -> dict:
    """Scale parameters of one district's pool (r: (Y, K, 12))."""
    n = r.shape[0]
    R = r.sum(2)                      # (Y, K); identical across k up to rounding
    Ry = R.mean(1)
    rmse = float(np.sqrt(np.mean(Ry ** 2)))
    sd = float(Ry.std(ddof=0))
    f = rmse / sd if sd > 0 else 1.0
    g = float(stats.t.ppf(RC.PRED_QUANTILE, n) / stats.norm.ppf(RC.PRED_QUANTILE))
    h = RC.SILVERMAN * n ** (-0.2)
    s_m = r.reshape(-1, r.shape[2]).std(0, ddof=0)
    pi = s_m / s_m.sum() * sd if s_m.sum() > 0 else np.zeros_like(s_m)
    return {"n_years": n, "annual_rmse": rmse, "annual_bias": float(Ry.mean()),
            "annual_sd": sd, "annual_spread_across_k": float(np.ptp(R, axis=1).max()),
            "f": f, "g": g, "h": h, "scale": f * g, "pi": pi, "s_m": s_m}


def members(c: dict, r: dict, prm: dict, iy: np.ndarray, k: np.ndarray, eps: np.ndarray,
            structural: np.ndarray) -> tuple[dict, dict]:
    """Members X[d] (N, M) for a target with value c[d] (M months, M <= 12): residual
    trajectory (iy, k) of the pool + year-level shock, shrunk by sqrt(1 + h^2), centred on the
    members' mean, scaled by f * g, added to the value, truncated at 0; structural months 0.
    Returns (X, pre-truncation X)."""
    X, X_pre = {}, {}
    for d in DISTRICTS:
        p = prm[d]
        mm = len(c[d])
        e = r[d][iy, k, :mm] + p["h"] * eps[:, None] * p["pi"][None, :mm]
        e = e / np.sqrt(1.0 + p["h"] ** 2)
        e = e - e.mean(0, keepdims=True)
        pre = np.asarray(c[d], float)[None, :] + p["scale"] * e
        pre[:, structural[:mm]] = 0.0
        X_pre[d] = pre
        X[d] = np.clip(pre, 0.0, None)
    return X, X_pre


def full_pool_members(n_years: int, n_k: int, seed: int) -> tuple:
    iy = np.repeat(np.arange(n_years), n_k)
    k = np.tile(np.arange(n_k), n_years)
    eps = np.random.default_rng(seed).standard_normal(len(iy))
    return iy, k, eps


def sampled_members(n_years: int, n_k: int, n: int, seed: int) -> tuple:
    """Balanced pool years (each year n // Y or n // Y + 1 times, shuffled), random draw
    index, standard-normal shock."""
    rng = np.random.default_rng(seed)
    iy = np.resize(rng.permutation(n_years), n)
    rng.shuffle(iy)
    k = rng.integers(0, n_k, size=n)
    eps = rng.standard_normal(n)
    return iy, k, eps


def structural_mask(final_fit: dict, months: pd.DatetimeIndex) -> np.ndarray:
    w = diversion.wvec(final_fit["s_chc"], final_fit["winter"])
    return np.array([w[m.month] == 0 for m in months])


# ----------------------------------------------------------------------------- coverage
def interval(s: np.ndarray, level: float = RC.LEVEL) -> tuple[float, float]:
    lo, hi = np.percentile(s, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return float(lo), float(hi)


def coverage_rows(base: dict, single: dict, double: dict, eval_years: list[int],
                  obs_annual: dict, obs_janmay: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Double leave-one-year-out coverage of the annual (and Jan-May) district totals."""
    ss, dates = ss_draws(base)
    months12 = pd.date_range("2001-01-01", periods=12, freq="MS")
    smask = structural_mask(base["final_fit"], months12)
    rows, jrows = [], []
    for t in eval_years:
        pool = [y for y in eval_years if y != t]
        r = residual_pool({y: double[(t, y)] for y in pool}, ss, dates, pool)
        prm = {d: params(r[d]) for d in DISTRICTS}
        iy, k, eps = full_pool_members(len(pool), ss["chc"].shape[0], RC.COVERAGE_SEED + t)
        c = {d: np.asarray(single[t][d], float) for d in DISTRICTS}
        X, _ = members(c, r, prm, iy, k, eps, smask)
        for d in DISTRICTS:
            s = X[d].sum(1)
            lo, hi = interval(s)
            o = float(obs_annual[d][t])
            rows.append({"year": t, "district": d, "obs": o, "chain": float(c[d].sum()),
                         "p05": lo, "p50": float(np.median(s)), "p95": hi,
                         "hit": bool(lo <= o <= hi), "width": hi - lo,
                         "f": prm[d]["f"], "g": prm[d]["g"], "h": prm[d]["h"],
                         "pool_rmse": prm[d]["annual_rmse"], "pool_bias": prm[d]["annual_bias"]})
            if t in obs_janmay[d]:
                sj = X[d][:, :5].sum(1)
                lo, hi = interval(sj)
                o = float(obs_janmay[d][t])
                jrows.append({"year": t, "district": d, "obs": o, "chain": float(c[d][:5].sum()),
                              "p05": lo, "p50": float(np.median(sj)), "p95": hi,
                              "hit": bool(lo <= o <= hi), "width": hi - lo})
    return pd.DataFrame(rows), pd.DataFrame(jrows)


def gate(df: pd.DataFrame) -> dict:
    """A1 gate-2 (clusters = years) and, for continuity, Clopper-Pearson 95 % on n_eff =
    number of years (hits scaled). ``df`` holds year, hit, obs and the interval as
    (p05, p95) (annual, Jan-May) or (lo, hi) (inflow); district and month when present."""
    from harness import coverage2
    lo_c, hi_c = ("p05", "p95") if "p05" in df.columns else ("lo", "hi")
    hits = df.hit.to_numpy(bool)
    g = coverage2.coverage_gate(hits, df.year.to_numpy())
    g = {k: (float(v) if isinstance(v, (float, np.floating)) else
             bool(v) if isinstance(v, (bool, np.bool_)) else v)
         for k, v in g.items() if not isinstance(v, np.ndarray)}
    n_c = df.year.nunique()
    k_eff = hits.sum() * n_c / len(hits)
    lo = 0.0 if k_eff == 0 else float(stats.beta.ppf(0.025, k_eff, n_c - k_eff + 1))
    hi = 1.0 if k_eff >= n_c else float(stats.beta.ppf(0.975, k_eff + 1, n_c - k_eff))
    width = df[hi_c] - df[lo_c]
    if "district" in df.columns:
        mean_width = {NAME[d]: float(width[df.district == d].mean())
                      for d in DISTRICTS if (df.district == d).any()}
    else:
        mean_width = {"all": float(width.mean())}
    misses = []
    for _, r in df[~df.hit].iterrows():
        m = {"year": int(r["year"])}
        for c in ("district", "month"):
            if c in df.columns:
                m[c] = r[c] if c == "district" else int(r[c])
        m["side"] = "over-predicted" if r["obs"] < r[lo_c] else "under-predicted"
        misses.append(m)
    return {"n": int(len(hits)), "hits": int(hits.sum()), "coverage": float(hits.mean()),
            "n_years": int(n_c), "a1_gate2": g, "clopper_pearson_95_neff_years": [lo, hi],
            "cp_contains_target": bool(lo <= RC.COVERAGE_TARGET <= hi),
            "mean_width": mean_width, "misses": misses}


# ----------------------------------------------------------------------------- inflow
def inflow_observed_months(comp: pd.DataFrame) -> set:
    """Months whose every composite day has flag <= INFLOW_MAX_FLAG (observed)."""
    f = pd.Series(comp["flag"].to_numpy(int), index=pd.to_datetime(comp["date"]))
    ok = f.le(RC.INFLOW_MAX_FLAG).groupby(f.index.to_period("M")).all()
    return {p.to_timestamp() for p, v in ok.items() if v}


def inflow_rms(preds: dict, Q: pd.Series, pool_years: list[int]) -> np.ndarray:
    """Per calendar month RMS of (q_pred - Q) over the pool years (12,)."""
    e = np.stack([np.asarray(preds[y]["q_pred"], float)
                  - Q.reindex(M.months_of([y])).to_numpy() for y in pool_years])
    return np.sqrt(np.mean(e ** 2, axis=0))


def inflow_coverage(single: dict, double: dict, Q: pd.Series, eval_years: list[int],
                    observed: set) -> pd.DataFrame:
    rows = []
    for t in eval_years:
        pool = [y for y in eval_years if y != t]
        rms = inflow_rms({y: double[(t, y)] for y in pool}, Q, pool)
        g = float(stats.t.ppf(RC.PRED_QUANTILE, len(pool)) / stats.norm.ppf(RC.PRED_QUANTILE))
        half = RC.INFLOW_Z * g * rms
        q = np.asarray(single[t]["q_pred"], float)
        for i, m in enumerate(M.months_of([t])):
            if m not in observed:
                continue
            o = float(Q.loc[m])
            lo, hi = max(0.0, q[i] - half[i]), q[i] + half[i]
            rows.append({"year": t, "month": m.month, "obs": o, "pred": float(q[i]),
                         "lo": lo, "hi": hi, "hit": bool(lo <= o <= hi)})
    return pd.DataFrame(rows)
