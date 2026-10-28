"""WP X5 library: a two-input transfer-function (Riley / preconsolidation) model of ring-band
compaction, its baselines, the data it reads and the site score.

Model, per ring site and depth band, on a monthly axis (t in years from 2012-01)::

    c(t) = a + r t - A * R[x_e; tau_e](t) + A0 * x_i(t) + B * R[x_i; tau_v](t)

* ``hpc(t)`` is the preconsolidation head: the minimum of the burn-in heads (2012-13) and of
  every head since 2014-01 up to month t (a running prior minimum);
* input 1, ``x_e = h - hpc >= 0``: head above its running prior minimum (elastic,
  recoverable; gain ``A``);
* input 2, ``x_i = hpc_burn - hpc >= 0``: how far the head has gone below the burn-in
  minimum, i.e. the cumulative depth of new lows (inelastic: immediate gain ``A0``, delayed
  gain ``B`` with time constant ``tau_v``);
* ``R[x; tau]`` is the exponential response (``y_t = (1 - d) x_t + d y_{t-1}``,
  ``d = exp(-30 d / tau)``, the recurrence of ``hydrophysics.twin.compaction.vep_compaction``);
* ``a + r t`` is the residual creep of pre-2012 drawdown (decades-long clay time constants
  are not identifiable from ten years; Lees 2022).

With ``A = A0 = Ske``, ``B = Skv``, ``tau_e = 0`` and ``tau_v = tau`` this is exactly the twin's
VEP column started at ``h_pc0 = hpc_burn - h(2012-01)`` plus a linear trend, which is what
the synthetic check uses to generate data. ``A, A0, B >= 0``; ``(tau_e, tau_v)`` are chosen
on a fixed grid by least squares, the gains by bounded linear least squares.

Baselines: persistence (the 2018-19 OLS rate continued from the anchor survey) and a linear
model on head level (``c = a + r t + k h``, the same trend, no preconsolidation split).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import x5const as K  # noqa: E402

# ------------------------------------------------------------------ time axis


def month_axis(t0: str = K.T0, t1: str = K.T1) -> pd.PeriodIndex:
    return pd.period_range(t0, t1, freq="M")


def in_window(months: pd.PeriodIndex, win: tuple[str, str]) -> np.ndarray:
    a, b = pd.Period(win[0], "M"), pd.Period(win[1], "M")
    return np.asarray((months >= a) & (months <= b))


def t_years(months: pd.PeriodIndex) -> np.ndarray:
    base = pd.Period(K.T0, "M")
    return np.array([(m - base).n for m in months], dtype=float) / 12.0


# ------------------------------------------------------------------ inputs


def exp_response(x: np.ndarray, tau_days: float, dt_days: float = K.DT_DAYS) -> np.ndarray:
    """Exponential response of a monthly input (``tau <= 0``: instantaneous)."""
    x = np.asarray(x, dtype=float)
    if tau_days <= 0:
        return x.copy()
    d = math.exp(-dt_days / tau_days)
    y = np.empty_like(x)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = (1.0 - d) * x[t] + d * y[t - 1]
    return y


def precon_inputs(h: np.ndarray, months: pd.PeriodIndex) -> dict:
    """``hpc_burn``, ``hpc`` (running prior minimum), ``x_e``, ``x_i`` of a filled head series."""
    h = np.asarray(h, dtype=float)
    burn = in_window(months, K.BURN)
    hpc_burn = float(np.nanmin(h[burn]))
    hpc = np.full_like(h, hpc_burn)
    cur = hpc_burn
    for t in range(len(h)):
        if not burn[t]:
            cur = min(cur, h[t])
        hpc[t] = cur
    return {"hpc_burn": hpc_burn, "hpc": hpc, "x_e": h - hpc, "x_i": hpc_burn - hpc}


# ------------------------------------------------------------------ fits


def _design_tf(inp: dict, ty: np.ndarray, tau_e: float, tau_v: float, vep_form: bool):
    cache = inp["_cache"]
    if ("e", tau_e) not in cache:
        cache[("e", tau_e)] = exp_response(inp["x_e"], tau_e)
    if ("v", tau_v) not in cache:
        cache[("v", tau_v)] = exp_response(inp["x_i"], tau_v)
    fe, fv = cache[("e", tau_e)], cache[("v", tau_v)]
    if vep_form:   # A = A0 (one elastic gain on -h), tau_e = 0
        return np.column_stack([np.ones_like(ty), ty, inp["x_i"] - inp["x_e"], fv]), 2
    return np.column_stack([np.ones_like(ty), ty, -fe, inp["x_i"], fv]), 2


def fit_tf(h: np.ndarray, obs: np.ndarray, months: pd.PeriodIndex, fit_mask: np.ndarray,
           vep_form: bool = False) -> dict:
    """Grid over (tau_e, tau_v), bounded least squares for the gains. ``obs`` NaN where no
    ring survey; only ``fit_mask & finite(obs)`` months are fitted. Returns the parameters
    and the full monthly prediction ``pred`` (m)."""
    inp = precon_inputs(h, months)
    inp["_cache"] = {}
    ty = t_years(months)
    m = fit_mask & np.isfinite(obs)
    y = obs[m]
    no_new_low = float(np.max(inp["x_i"][m], initial=0.0)) < 1e-9
    tau_es = (0.0,) if vep_form else K.TAU_E_GRID
    tau_vs = (K.TAU_V_GRID[0],) if no_new_low else K.TAU_V_GRID
    best = None
    for te in tau_es:
        for tv in tau_vs:
            X, nfree = _design_tf(inp, ty, te, tv, vep_form)
            keep = np.ones(X.shape[1], dtype=bool)
            if no_new_low:   # the new-low columns are identically 0 over the fit months
                keep[3:] = False   # TF: x_i and R[x_i]; VEP form: R[x_i] (col 2 stays)
            lo = np.r_[np.full(nfree, -np.inf), np.zeros(X.shape[1] - nfree)]
            hi = np.full(X.shape[1], np.inf)
            Xk = X[m][:, keep]
            sol = lsq_linear(Xk, y, bounds=(lo[keep], hi[keep]), method="bvls")
            coef = np.zeros(X.shape[1])
            coef[keep] = sol.x
            sse = float(((Xk @ sol.x - y) ** 2).sum())
            if best is None or sse < best["sse"] - 1e-15:
                best = {"sse": sse, "tau_e": te, "tau_v": tv, "coef": coef, "X": X}
    c = best["coef"]
    out = {"n_fit": int(m.sum()), "sse": best["sse"], "a": float(c[0]), "r_m_yr": float(c[1]),
           "tau_e_days": float(best["tau_e"]),
           "tau_v_days": float("nan") if no_new_low else float(best["tau_v"]),
           "no_new_low_in_fit": bool(no_new_low), "pred": best["X"] @ c,
           "D_fit_m": float(np.max(inp["x_i"][fit_mask], initial=0.0))}
    if vep_form:
        out.update({"A": float(c[2]), "A0": float(c[2]), "B": float(c[3])})
    else:
        out.update({"A": float(c[2]), "A0": float(c[3]), "B": float(c[4])})
    return out


def fit_linear_head(h: np.ndarray, obs: np.ndarray, months: pd.PeriodIndex,
                    fit_mask: np.ndarray) -> dict:
    """Baseline B2: ``c = a + r t + k h`` by OLS on the fit months."""
    ty = t_years(months)
    X = np.column_stack([np.ones_like(ty), ty, np.asarray(h, dtype=float)])
    m = fit_mask & np.isfinite(obs)
    coef, *_ = np.linalg.lstsq(X[m], obs[m], rcond=None)
    return {"n_fit": int(m.sum()), "a": float(coef[0]), "r_m_yr": float(coef[1]),
            "k_m_per_m": float(coef[2]), "pred": X @ coef}


def persistence_rate(obs: np.ndarray, months: pd.PeriodIndex) -> tuple[float, int]:
    """Baseline B1: OLS slope (m/yr) of the ring months in PERSIST; NaN below support."""
    m = in_window(months, K.PERSIST) & np.isfinite(obs)
    if m.sum() < K.MIN_PERSIST_OBS:
        return float("nan"), int(m.sum())
    return float(np.polyfit(t_years(months)[m], obs[m], 1)[0]), int(m.sum())


# ------------------------------------------------------------------ test-window scoring


def anchor_index(obs: np.ndarray, months: pd.PeriodIndex) -> int | None:
    idx = np.flatnonzero(in_window(months, K.ANCHOR) & np.isfinite(obs))
    return int(idx[-1]) if len(idx) else None


def test_errors(pred: np.ndarray, obs: np.ndarray, months: pd.PeriodIndex,
                ia: int) -> np.ndarray:
    """Errors (m) of the change since the anchor survey, at the ring months of TEST."""
    m = in_window(months, K.TEST) & np.isfinite(obs)
    return (pred[m] - pred[ia]) - (obs[m] - obs[ia])


def persistence_pred(rate: float, months: pd.PeriodIndex, ia: int) -> np.ndarray:
    ty = t_years(months)
    return rate * (ty - ty[ia])


def rmse(e: np.ndarray) -> float:
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.mean(e ** 2))) if e.size else float("nan")


# ------------------------------------------------------------------ data: rings


def ring_bands(t_end: str = K.T1) -> dict[str, pd.DataFrame]:
    """{site: monthly band compaction (m, positive = shortening)}, twin convention
    (``twin.mechanism.ring_positions`` + ``band_compaction``), ring positions truncated at
    ``t_end`` BEFORE the bands are formed, so later surveys never shape them. Only the depth
    bands are kept (not ``total``). Index: monthly periods."""
    from hydrophysics.twin.mechanism import band_compaction, ring_positions

    pos = ring_positions(str(K.RING_DDIR))
    end = pd.Period(t_end, "M").to_timestamp(how="end")
    out = {}
    for site, p in pos.items():
        p = p.loc[:end]
        if p.dropna(how="all").empty:
            continue
        b = band_compaction(p)
        b = b[[c for c in b.columns if c != "total"]]
        b.index = b.index.to_period("M")
        out[site] = b
    return out


def band_mid_depth(label: str) -> float:
    a, b = label.rstrip("m").split("-")
    return 0.5 * (float(a) + float(b))


# ------------------------------------------------------------------ data: heads


def ring_sites() -> pd.DataFrame:
    s = pd.read_csv(K.RING_SITES, dtype={"wells": str})
    s["wells"] = s["wells"].map(lambda v: [w.strip().zfill(8) for w in str(v).split(",")])
    return s


def load_heads(sids: list[str], use: str = K.HEAD_USE, window_end: str = K.T1) -> pd.DataFrame:
    """Monthly C20 heads (flags 0/1) of ``sids`` through the loader guard ->
    DataFrame indexed by monthly period, one column per sid (NaN where missing)."""
    from common import guard

    p = guard.open_product(K.HEADS_DIR, use, window_end=window_end)
    df = p.read_table("heads_monthly.parquet", columns=["sid", "date", "value", "flag", "c20"])
    df = df[df.sid.isin(sids) & df.flag.isin(K.HEAD_FLAGS) & df.c20.astype(bool)]
    df = df.assign(month=pd.to_datetime(df.date).dt.to_period("M"))
    w = df.pivot_table(index="month", columns="sid", values="value", aggfunc="mean")
    return w


def well_depths(sids: list[str]) -> pd.Series:
    from common import guard

    p = guard.open_product(K.HEADS_DIR, K.HEAD_USE)   # the pre-origin station table
    m = p.read_table("station_meta_preorigin.parquet", columns=K.META_COLUMNS)
    return m[m.sid.isin(sids)].set_index("sid")["WellDepth"].astype(float)


def fill_head(s: pd.Series, months: pd.PeriodIndex) -> np.ndarray | None:
    """Gap rule: interpolate inside gaps <= MAX_GAP months; every month from the end of
    BURN on must then be present, and BURN needs MIN_BURN_MONTHS observed months; leading
    missing months take the first value. None if the rule fails."""
    x = s.reindex(months).astype(float)
    burn = in_window(months, K.BURN)
    if int(x[burn].notna().sum()) < K.MIN_BURN_MONTHS:
        return None
    xi = x.interpolate(limit=K.MAX_GAP, limit_area="inside")
    post = ~burn
    if xi[post].isna().any():
        return None
    return xi.bfill().to_numpy(dtype=float)


def nest_mean_head(W: pd.DataFrame, wells: list[str], months: pd.PeriodIndex) -> np.ndarray | None:
    """Declared alternative input: mean over the valid nest wells of each well's anomaly
    about its FIT-window mean (the twin's nest convention, without the test years)."""
    fit = in_window(months, (K.BURN[0], K.FIT[1]))
    cols = []
    for w in wells:
        if w not in W.columns:
            continue
        f = fill_head(W[w], months)
        if f is None:
            continue
        cols.append(f - np.mean(f[fit]))
    if not cols:
        return None
    return np.mean(np.vstack(cols), axis=0)


def choose_well(wells: list[str], depths: pd.Series, W: pd.DataFrame, months: pd.PeriodIndex,
                mid: float) -> tuple[str | None, np.ndarray | None]:
    """Primary input: the valid nest well whose WellDepth is nearest the band's mid-depth."""
    cand = []
    for w in wells:
        if w not in W.columns or w not in depths.index or not np.isfinite(depths[w]):
            continue
        f = fill_head(W[w], months)
        if f is not None:
            cand.append((abs(float(depths[w]) - mid), w, f))
    if not cand:
        return None, None
    cand.sort(key=lambda c: (c[0], c[1]))
    return cand[0][1], cand[0][2]


# ------------------------------------------------------------------ assembly


def build_inputs(t_end: str = K.T1, use: str = K.HEAD_USE) -> tuple[list[dict], pd.PeriodIndex]:
    """One record per (site, band) of the 14 ring sites: ring band series on the monthly
    axis (NaN where unsurveyed), the primary head (depth-matched nest well) and the declared
    alternative (nest-mean anomaly). Missing inputs are None; nothing is dropped here."""
    months = month_axis(K.T0, t_end)
    sites = ring_sites()
    all_w = sorted({w for ws in sites.wells for w in ws})
    W = load_heads(all_w, use=use, window_end=t_end)
    depths = well_depths(all_w)
    rings = ring_bands(t_end)
    recs = []
    for _, s in sites.iterrows():
        bands = rings.get(s.site)
        hn = nest_mean_head(W, s.wells, months)
        labels = list(bands.columns) if bands is not None else []
        for j, lab in enumerate(labels):
            obs = bands[lab].reindex(months).to_numpy(dtype=float)
            well, hp = choose_well(s.wells, depths, W, months, band_mid_depth(lab))
            recs.append({"site": s.site, "band_index": j, "band": lab, "obs": obs,
                         "well": well, "well_depth_m": (float(depths[well]) if well else None),
                         "h_primary": hp, "h_nest": hn})
        if not labels:
            recs.append({"site": s.site, "band_index": -1, "band": None, "obs": None,
                         "well": None, "well_depth_m": None, "h_primary": None,
                         "h_nest": hn})
    return recs, months


def analyse_band(h: np.ndarray | None, obs: np.ndarray | None,
                 months: pd.PeriodIndex) -> dict:
    """Fit TF (primary), VEP-form TF (declared alternative) and both baselines on FIT, and
    the test-window errors of each. ``scorable`` is False (with ``why``) when a support rule
    fails; then no error is returned."""
    if obs is None:
        return {"scorable": False, "why": "no ring band"}
    if h is None:
        return {"scorable": False, "why": "no valid head series"}
    fit = in_window(months, K.FIT)
    n_fit = int((fit & np.isfinite(obs)).sum())
    n_test = int((in_window(months, K.TEST) & np.isfinite(obs)).sum())
    ia = anchor_index(obs, months)
    rate, n_pers = persistence_rate(obs, months)
    why = []
    if n_fit < K.MIN_FIT_OBS:
        why.append(f"fit months {n_fit} < {K.MIN_FIT_OBS}")
    if n_test < K.MIN_TEST_OBS:
        why.append(f"test months {n_test} < {K.MIN_TEST_OBS}")
    if ia is None:
        why.append("no anchor survey in 2019-07..2019-12")
    if not np.isfinite(rate):
        why.append(f"persistence months {n_pers} < {K.MIN_PERSIST_OBS}")
    if why:
        return {"scorable": False, "why": "; ".join(why), "n_fit": n_fit, "n_test": n_test}
    tf = fit_tf(h, obs, months, fit)
    vep = fit_tf(h, obs, months, fit, vep_form=True)
    lin = fit_linear_head(h, obs, months, fit)
    inp = precon_inputs(h, months)
    test = in_window(months, K.TEST)
    e = {"tf": test_errors(tf["pred"], obs, months, ia),
         "vep": test_errors(vep["pred"], obs, months, ia),
         "lin": test_errors(lin["pred"], obs, months, ia),
         "pers": test_errors(persistence_pred(rate, months, ia), obs, months, ia)}
    return {"scorable": True, "n_fit": n_fit, "n_test": n_test, "anchor": str(months[ia]),
            "tf": tf, "vep": vep, "lin": lin, "pers_rate_m_yr": rate, "errors": e,
            "D_fit_m": float(np.max(inp["x_i"][fit], initial=0.0)),
            "D_test_new_m": float(np.max(inp["x_i"][test], initial=0.0)
                                  - inp["x_i"][ia]),
            "hpc_burn": inp["hpc_burn"]}


def site_scores(band_results: list[dict], key_h: str = "primary") -> pd.DataFrame:
    """Per site: RMSE (mm) of each method pooled over the site's scorable bands, and the
    declared pass rule (TF <= IMPROVE x persistence AND TF <= IMPROVE x linear-head)."""
    rows = []
    for site in dict.fromkeys(r["site"] for r in band_results):
        rs = [r for r in band_results if r["site"] == site and r[key_h]["scorable"]]
        row = {"site": site, "n_bands_scored": len(rs)}
        for meth in ("tf", "vep", "lin", "pers"):
            e = np.concatenate([r[key_h]["errors"][meth] for r in rs]) if rs else np.array([])
            row[f"rmse_{meth}_mm"] = rmse(e) * 1000.0 if e.size else float("nan")
        row["n_months_scored"] = int(sum(len(r[key_h]["errors"]["tf"]) for r in rs))
        ok = bool(rs) and (row["rmse_tf_mm"] <= K.IMPROVE * row["rmse_pers_mm"]) and (
            row["rmse_tf_mm"] <= K.IMPROVE * row["rmse_lin_mm"])
        row["pass"] = bool(ok)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ synthetic truth


def twin_column(h: np.ndarray, months: pd.PeriodIndex, ske: float, skv: float,
                tau_days: float) -> np.ndarray:
    """Compaction (m) of the twin's own VEP column (``hydrophysics.twin.compaction.
    vep_compaction``, read-only import) on one monthly head series, normally consolidated at
    the burn-in minimum (``h_pc0 = hpc_burn - h[0]``)."""
    import torch

    from hydrophysics.twin.compaction import vep_compaction

    hpc_burn = precon_inputs(h, months)["hpc_burn"]
    ht = torch.tensor(np.asarray(h, dtype=float)[None, :], dtype=torch.float64)

    def p(v):
        return torch.tensor([v], dtype=torch.float64)

    with torch.no_grad():
        c = vep_compaction(ht, p(math.log(ske)), p(math.log(skv)), p(math.log(tau_days)),
                           p(hpc_burn - float(h[0])), K.DT_DAYS)
    return c.numpy()[0]


def synth_obs(h: np.ndarray, months: pd.PeriodIndex, case: dict, mask: np.ndarray,
              rng: np.random.Generator, noise_m: float) -> np.ndarray:
    """Synthetic ring band: twin column + linear trend + iid noise, NaN outside ``mask``."""
    c = twin_column(h, months, case["ske"], case["skv"], case["tau_days"])
    c = c + case["trend_m_yr"] * t_years(months) + rng.normal(0.0, noise_m, len(c))
    return np.where(mask, c, np.nan)
