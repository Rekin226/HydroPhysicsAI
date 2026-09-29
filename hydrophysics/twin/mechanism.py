"""Rebound or slowed creep? What the compaction rings and the leveling network say.

    python -m hydrophysics.twin.mechanism --out results/twin/mechanism

The two flow models agree on how much subsidence a pumping cut avoids by 2032 and disagree
on how (STATE §0). In the per-well datum model a cut gives a one-time rebound within
months and the creep left over from past drawdown goes on at the same rate. In the
previous model the cut also slows that creep, so the avoided subsidence grows for years.
The leveling network sees only the sum. This module asks the record directly.

Two measurements, on the 14 multi-layer compaction wells (MLCW rings, depth-resolved,
monthly, 2010-2025) and on the leveling benchmarks next to a head-observation nest:

1. **Elastic response**: compaction per metre of head change at seasonal frequency
   (:func:`seasonal_slope`), from the monthly anomalies about a 12-month running mean.
2. **Does the multi-year rate depend on the head?** (:func:`rate_table`,
   :func:`fe_ols`). With ``D`` the drawdown (minus the head anomaly, m) and ``C`` the
   compaction (m), both as calendar-year means, the annual rate is regressed per site on

       R_y = a_site + e * dD_y + k * lev_{y-1}  (+ g * dD_{y-1})

   ``dD_y`` is this year's change in drawdown (the elastic and instant-plastic step) and
   ``lev_{y-1}`` last year's drawdown relative to the site's mean. A pure rebound (the
   datum model's mechanism) predicts ``k = 0`` and ``g = 0``: once the step is taken the
   rate is what it was. Creep that slows when heads recover predicts ``k > 0``: a year of
   higher heads lowers the NEXT year's rate.

The leveling benchmarks within 1.5 km of a head nest (:func:`leveling_analysis`) are
regressed per survey interval on the drawdown step between the two surveys (``dDe``), the
interval's mean level (``levI``) and the previous year's (``levP``); in the ``sustained``
form the ``levI`` coefficient is the rate change a level held for two surveys leaves
(zero for a pure rebound). :func:`matched_ring_leveling` puts the rings and the benchmarks
next to them on identical intervals, so a difference between the two is ground motion
outside the ring interval, not the design.

The same regression is run on each flow model's own hindcast at the ring cells (its column
driven by its heads, from the forward archive) and on its column driven by the observed
heads, and each model's column is given a sustained +1 m head rise (:func:`step_response`)
in the hindcast and from the policy start in the projection, so that the effect it implies
can be set against the confidence interval the record gives. ``mde80`` is the smallest
effect the record could have detected (two-sided 5 %, 80 % power). :func:`ring_fit_comparison`
fits the historical VEP column and the opt-in aquitard variant
(``compaction.aquitard_compaction``) to the rings on the observed heads, leave one site out.

Reads only; writes a new output directory. CPU: a minute, plus ~0.5 h for the ring fit
(``--ring-fit-epochs 0`` skips it).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd

from ..subsidence import _decode_mlcw_name, load_mlcw_stations

# ring-interval depth bands (m below the top ring); the last is open-ended
BAND_EDGES_M = (0.0, 100.0, 200.0)
MIN_MONTHS = 9          # a calendar-year mean needs this many monthly values
NEST_RADIUS_M = 1000.0  # a head-observation well belongs to a ring site within this range
MAD_K_MONTHLY = 6.0     # per-well monthly despike (sentinel values survive heads.MAD_K)


# ---------------------------------------------------------------------------------------
# data: rings
# ---------------------------------------------------------------------------------------
def ring_positions(ddir: str) -> dict[str, pd.DataFrame]:
    """{site -> DataFrame of ring depths (m), columns sorted shallow -> deep}."""
    out = {}
    pat = os.path.join(ddir, "ls_cache", "clean", "ls-wra-mlcw-obs__*.parquet")
    for f in sorted(glob.glob(pat)):
        df = pd.read_parquet(f).sort_index()
        out[_decode_mlcw_name(f)] = df[df.mean().sort_values().index]
    return out


def band_compaction(pos: pd.DataFrame, edges=BAND_EDGES_M,
                    min_cover: float = 0.9) -> pd.DataFrame:
    """Ring depths -> monthly compaction (m, positive = the interval shortens) of the whole
    monitored interval (``total``) and of each depth band, re-zeroed to the first month.

    A band runs between the rings nearest its two edges (the deepest ring closes the last
    band), so the bands partition ``total``. Rings observed in less than ``min_cover`` of
    the samples, or in less than half of the samples of any surveyed year, are dropped, so
    a ring lost for a year never removes that year from a band."""
    cover = pos.notna().mean()
    by_year = pos.notna().groupby(pos.index.year).mean().min()
    pos = pos.loc[:, (cover >= min_cover) & (by_year >= 0.5)]
    depth = pos.mean().to_numpy()
    if pos.shape[1] < 2:
        raise ValueError("fewer than two usable rings")
    idx = [int(np.argmin(np.abs(depth - (depth[0] + e)))) for e in edges]
    idx = sorted(set(idx + [pos.shape[1] - 1]))
    monthly = pos.groupby(pos.index.to_period("M")).median()
    monthly.index = monthly.index.to_timestamp("M")
    cols = {}

    def interval(i: int, j: int) -> pd.Series:
        sep = monthly.iloc[:, j] - monthly.iloc[:, i]
        fvi = sep.first_valid_index()
        return (sep.loc[fvi] - sep) if fvi is not None else sep * np.nan

    cols["total"] = interval(0, pos.shape[1] - 1)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        cols[f"{depth[a] - depth[0]:.0f}-{depth[b] - depth[0]:.0f}m"] = interval(a, b)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------------------
# data: heads
# ---------------------------------------------------------------------------------------
def _xy(s) -> tuple[float, float] | None:
    """``LocationByTWD97`` in either order ('x y' or 'y x') -> (x, y) metres."""
    try:
        a, b = (float(v) for v in str(s).split()[:2])
    except (ValueError, TypeError):
        return None
    x, y = (a, b) if a < 1.0e6 else (b, a)
    return (x, y) if 140000 <= x <= 240000 and 2580000 <= y <= 2700000 else None


def _monthly_well(path: str, k: float = MAD_K_MONTHLY) -> pd.Series:
    """One well's monthly median head, with a MAD screen on the monthly values."""
    s = pd.read_parquet(path)
    s = s[s.columns[0]].dropna().sort_index()
    m = s.groupby(s.index.to_period("M")).median()
    med = m.median()
    mad = (m - med).abs().median() or (m.std() or 1.0)
    m = m[(m - med).abs() <= k * mad]
    m.index = m.index.to_timestamp("M")
    return m


def nest_drawdown(stations: pd.DataFrame, wells_dir: str, xy: tuple[float, float],
                  radius_m: float = NEST_RADIUS_M) -> tuple[pd.Series, list[str]]:
    """Monthly drawdown anomaly (m, positive = head below its mean) of the head-observation
    nest at ``xy``: the mean over the nest's wells (all aquifers) of each well's head
    anomaly. Anomalies, not levels, so a well missing a month cannot shift the mean."""
    anoms, used = [], []
    for _, r in stations.iterrows():
        p = _xy(r["LocationByTWD97"])
        if p is None or math.hypot(p[0] - xy[0], p[1] - xy[1]) > radius_m:
            continue
        f = os.path.join(wells_dir, f"{r['sid']}.parquet")
        if not os.path.exists(f):
            continue
        m = _monthly_well(f)
        if len(m) < 24:
            continue
        anoms.append(-(m - m.mean()))
        used.append(str(r["sid"]))
    if not anoms:
        return pd.Series(dtype="float64"), used
    return pd.concat(anoms, axis=1).mean(axis=1).sort_index(), used


# ---------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------
def annual_means(s: pd.Series, min_months: int = MIN_MONTHS) -> pd.Series:
    """Calendar-year means of a monthly series, NaN where fewer than ``min_months``."""
    s = s.dropna()
    g = s.groupby(s.index.year)
    m = g.mean()
    return m.where(g.count() >= min_months)


def seasonal_slope(C: pd.Series, D: pd.Series, window: int = 12,
                   lags=(0, 1, 2)) -> dict:
    """Elastic response at seasonal frequency: the OLS slope of the compaction anomaly on
    the drawdown anomaly (m/m), each about its centred ``window``-month running mean, at
    each head lead ``lag`` (months by which compaction trails head). Months where either
    running mean has fewer than ``window - 2`` values are dropped."""
    idx = pd.date_range(min(C.index.min(), D.index.min()), max(C.index.max(), D.index.max()),
                        freq="ME")
    c, d = C.reindex(idx), D.reindex(idx)

    def anom(x):
        rm = x.rolling(window, center=True, min_periods=window - 2).mean()
        return x - rm

    ca, da = anom(c), anom(d)
    out = {}
    for lag in lags:
        j = pd.concat([ca, da.shift(lag)], axis=1).dropna()
        if len(j) < 24:
            out[lag] = {"slope": float("nan"), "r2": float("nan"), "n": len(j)}
            continue
        x, y = j.iloc[:, 1].to_numpy(), j.iloc[:, 0].to_numpy()
        x = x - x.mean()
        y = y - y.mean()
        b = float(x @ y / max(x @ x, 1e-12))
        r2 = 1.0 - float(((y - b * x) ** 2).sum()) / max(float((y ** 2).sum()), 1e-12)
        out[lag] = {"slope": b, "r2": r2, "n": len(j)}
    best = max(out, key=lambda k: (-1.0 if np.isnan(out[k]["r2"]) else out[k]["r2"]))
    return {"by_lag": out, "best_lag": best, "slope": out[best]["slope"],
            "r2": out[best]["r2"]}


def virgin_by_year(D: pd.Series) -> pd.Series:
    """Per calendar year: how far the drawdown went past its previous maximum (m, >= 0),
    the new load a preconsolidation (VEP) column turns into plastic compaction. The first
    year of the record has no past and is NaN."""
    d = D.dropna()
    if d.empty:
        return pd.Series(dtype="float64")
    out = {}
    years = sorted(set(d.index.year))
    for y in years[1:]:
        past = d[d.index.year < y].max()
        cur = d[d.index.year == y].max()
        out[y] = max(0.0, float(cur - past))
    return pd.Series(out, dtype="float64")


def rate_table(C: dict[str, pd.Series], D: dict[str, pd.Series],
               min_months: int = MIN_MONTHS, years: tuple[int, int] | None = None
               ) -> pd.DataFrame:
    """Per site and year: the annual rate ``R`` (m/yr, difference of consecutive calendar-
    year compaction means), the drawdown change ``dD0`` over the same pair of years, the
    previous changes ``dD1``, ``dD2`` and the drawdown levels ``lev1``, ``lev2``, ``lev3``
    of one, two and three years before, about the site's mean (NaN where the head record
    is too short; each regression drops its own incomplete rows). Only consecutive year
    pairs enter (no rate is spread across a gap)."""
    rows = []
    for site, c in C.items():
        if site not in D or D[site].empty:
            continue
        ca, da = annual_means(c, min_months), annual_means(D[site], min_months)
        dmean = float(da.mean())
        vir = virgin_by_year(D[site])
        for y in ca.index:
            if years is not None and not (years[0] <= y <= years[1]):
                continue
            vals = [ca.get(y), ca.get(y - 1), da.get(y), da.get(y - 1), da.get(y - 2)]
            if any(v is None or not np.isfinite(v) for v in vals):
                continue
            def g(yr, da=da):
                v = da.get(yr)
                return float(v) if v is not None and np.isfinite(v) else np.nan

            rows.append({"site": site, "year": int(y), "R": float(ca[y] - ca[y - 1]),
                         "dD0": g(y) - g(y - 1), "dD1": g(y - 1) - g(y - 2),
                         "dD2": g(y - 2) - g(y - 3), "lev1": g(y - 1) - dmean,
                         "lev2": g(y - 2) - dmean, "lev3": g(y - 3) - dmean,
                         "V0": float(vir.get(y, np.nan)), "V1": float(vir.get(y - 1, np.nan))})
    return pd.DataFrame(rows)


def t_ppf(p: float, df: float) -> float:
    """Student-t quantile: scipy's when installed, else the Cornish-Fisher expansion about
    the normal quantile (within 1e-3 of it for df >= 5, which is all this module uses)."""
    try:
        from scipy import stats
    except ImportError:
        from statistics import NormalDist

        z = NormalDist().inv_cdf(p)
        g1 = (z ** 3 + z) / 4.0
        g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
        g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
        g4 = (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160.0
        return float(z + g1 / df + g2 / df ** 2 + g3 / df ** 3 + g4 / df ** 4)
    return float(stats.t.ppf(p, df))


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def fe_ols(df: pd.DataFrame, y: str, xs: list[str], group: str = "site",
           cluster: str | None = None, trend: bool = False, year_fe: bool = False) -> dict:
    """OLS of ``y`` on ``xs`` with a fixed effect (and optionally a linear year trend) per
    ``group``; ``year_fe`` adds one per year, so that only the differences between sites
    within a year identify the slopes (a common shock, a survey-campaign offset or a
    fan-wide drought cannot). Standard errors are cluster-robust (CR1) over ``cluster``
    (default ``group``), with a delete-one-cluster jackknife beside them; the reported
    ``se`` is the larger. Critical values from t with (clusters - 1) degrees of freedom."""
    cluster = cluster or group
    d = df.dropna(subset=[y, *xs]).reset_index(drop=True)
    groups = d[group].astype(str).to_numpy()
    cl = d[cluster].astype(str).to_numpy()
    # the fixed effects are partialled out within each group (Frisch-Waugh-Lovell: the
    # same slopes and residuals as the dummy regression); groups must nest in clusters so
    # that the sandwich and the jackknife on the partialled design are exact
    if pd.DataFrame({"g": groups, "c": cl}).groupby("g").c.nunique().max() > 1:
        raise ValueError(f"every {group!r} must sit in one {cluster!r}")
    W = np.column_stack([d[x].to_numpy(dtype="float64") for x in xs])
    if year_fe:
        yrs = d["year"].to_numpy()
        W = np.column_stack([W] + [(yrs == v).astype("float64")
                                   for v in sorted(set(yrs))[1:]])
    yy = d[y].to_numpy(dtype="float64")
    Wr, yr = W.copy(), yy.copy()
    n_fe = 0
    t_all = d["year"].to_numpy(dtype="float64") if trend else None
    for g in sorted(set(groups)):
        sel = groups == g
        B = np.ones((int(sel.sum()), 1))
        if trend:
            tg = t_all[sel]
            B = np.column_stack([B, tg - tg.mean()])
        n_fe += B.shape[1]
        proj = B @ np.linalg.pinv(B)
        Wr[sel] -= proj @ W[sel]
        yr[sel] -= proj @ yy[sel]
    X = Wr
    beta, res = _ols(X, yr)
    k = len(xs)
    n = X.shape[0]
    p = X.shape[1] + n_fe
    uc = sorted(set(cl))
    G = len(uc)
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in uc:
        sc = X[cl == c].T @ res[cl == c]
        meat += np.outer(sc, sc)
    scale = (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))
    V = scale * XtX_inv @ meat @ XtX_inv
    se_cr = np.sqrt(np.maximum(np.diag(V)[:k], 0.0))
    jk = []
    for c in uc:
        keep = cl != c
        b_k, _ = _ols(X[keep], yr[keep])
        jk.append(b_k[:k])
    jk = np.array(jk)
    se_jk = np.sqrt((G - 1) / G * ((jk - jk.mean(axis=0)) ** 2).sum(axis=0))
    se = np.maximum(se_cr, se_jk)
    tcrit = t_ppf(0.975, max(G - 1, 1))
    tpow = t_ppf(0.80, max(G - 1, 1))
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    return {"coef": dict(zip(xs, beta[:k].tolist(), strict=True)),
            "se": dict(zip(xs, se.tolist(), strict=True)),
            "se_cr1": dict(zip(xs, se_cr.tolist(), strict=True)),
            "se_jackknife": dict(zip(xs, se_jk.tolist(), strict=True)),
            "ci95": {x: [float(beta[i] - tcrit * se[i]), float(beta[i] + tcrit * se[i])]
                     for i, x in enumerate(xs)},
            "mde80": {x: float((tcrit + tpow) * se[i]) for i, x in enumerate(xs)},
            "n": int(n), "n_clusters": G, "trend": trend, "year_fe": year_fe,
            "r2": 1.0 - float((res ** 2).sum()) / max(ss_tot, 1e-18)}


def mde(se: float, n_clusters: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest true effect detected with ``power`` at two-sided level ``alpha``."""
    df = max(n_clusters - 1, 1)
    return float((t_ppf(1 - alpha / 2, df) + t_ppf(power, df)) * se)


# ---------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------
def column_params(vep_json: str) -> list[dict]:
    """The zonal (or single) column parameters of a ``calibrate_coupled`` JSON, with the
    aquitard keys when the column carries them (``--creep aquitard``)."""
    from .compaction import AQUITARD_KEYS, VEP_KEYS

    with open(vep_json) as fh:
        d = json.load(fh)
    return [{k: float(z[k]) for k in VEP_KEYS + AQUITARD_KEYS if k in z}
            for z in d.get("zonal", [d])]


def step_response(params: dict, drv: np.ndarray, t_step: int, dh: float = 1.0,
                  dt_days: float = 30.0) -> np.ndarray:
    """Compaction change (m) caused by raising the column's driving head by ``dh`` m from
    month ``t_step`` on: ``column(drv + dh*step) - column(drv)``, ``drv`` (n, T) heads.
    A pure rebound is a negative step that then stays flat; slowed creep keeps falling."""
    import torch

    from .compaction import column_compaction

    h = torch.tensor(np.atleast_2d(drv), dtype=torch.float64)
    h2 = h.clone()
    h2[:, t_step:] += dh
    p = {k: torch.tensor([v], dtype=torch.float64) for k, v in params.items()}
    with torch.no_grad():
        base = column_compaction(h, p, dt_days)
        up = column_compaction(h2, p, dt_days)
    return (up - base).numpy()


def step_summary(delta: np.ndarray, t_step: int) -> dict:
    """From a monthly step response (n, T): the rebound in the first 12 months and the
    change in annual rate in each of the next years (m/yr, negative = slower sinking),
    both as the mean over rows."""
    d = np.asarray(delta).mean(axis=0)
    yr = [d[t_step + 12 * i: t_step + 12 * (i + 1)].mean()
          for i in range((len(d) - t_step) // 12)]
    return {"rebound_first_year_m": float(yr[0]) if yr else float("nan"),
            "rate_change_m_per_yr": [float(b - a) for a, b in zip(yr[:-1], yr[1:],
                                                                    strict=True)]}


def column_on_series(params: dict, h: pd.Series, max_gap: int = 6,
                     dt_days: float = 30.0) -> pd.Series | None:
    """A column's compaction (m) driven by one monthly head series: gaps up to ``max_gap``
    months are interpolated, and the longest gap-free stretch is used. None if that
    stretch is shorter than three years."""
    import torch

    from .compaction import column_compaction

    idx = pd.date_range(h.index.min(), h.index.max(), freq="ME")
    x = h.reindex(idx).interpolate(limit=max_gap, limit_area="inside")
    ok = x.notna().to_numpy()
    best, run, start, bstart = 0, 0, 0, 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if run == 1:
            start = i
        if run > best:
            best, bstart = run, start
    if best < 36:
        return None
    seg = x.iloc[bstart:bstart + best]
    t = torch.tensor(seg.to_numpy()[None, :], dtype=torch.float64)
    p = {k: torch.tensor([v], dtype=torch.float64) for k, v in params.items()}
    with torch.no_grad():
        c = column_compaction(t, p, dt_days)
    return pd.Series(c.numpy()[0], index=seg.index)


def ring_cells(npz, xy: np.ndarray) -> np.ndarray:
    """Active-cell index of each ring site in a forward archive (-1 if off the grid)."""
    mask = np.asarray(npz["mask"])
    dx = float(npz["dx"])
    rows, cols = np.nonzero(mask)
    cent = np.column_stack([float(npz["x0"]) + (cols + 0.5) * dx,
                            float(npz["y0"]) + (rows + 0.5) * dx])
    out = []
    for p in np.atleast_2d(xy):
        d2 = ((cent - p) ** 2).sum(1)
        c = int(np.argmin(d2))
        out.append(c if math.sqrt(d2[c]) <= dx else -1)
    return np.array(out)


def model_ring_series(npz, cells: np.ndarray, names: list[str], rheology: int = 0,
                      scenario: int = 0) -> tuple[dict, dict, np.ndarray]:
    """A forward archive's hindcast at the ring cells: compaction (its column on its own
    free-run heads) and drawdown anomaly of the layer-mean head, monthly, record only."""
    T = int(npz["origin"]) + 1
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(npz["dates"])[:T])) + pd.offsets.MonthEnd(0)
    subs = np.asarray(npz["subs_mean_by_rheology"][rheology, scenario][:, :T], dtype="float64")
    head = np.asarray(npz["heads_mean"][scenario][:, :, :T], dtype="float64").mean(axis=0)
    C, D = {}, {}
    for n, c in zip(names, cells, strict=True):
        if c < 0:
            continue
        C[n] = pd.Series(subs[c] - subs[c, 0], index=dates)
        h = head[c]
        D[n] = pd.Series(-(h - h.mean()), index=dates)
    return C, D, head[cells[cells >= 0]]


# ---------------------------------------------------------------------------------------
# the analysis
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# the two columns on the rings, observed heads, leave one site out
# ---------------------------------------------------------------------------------------
def ring_arrays(C: dict[str, pd.Series], D: dict[str, pd.Series], t0: str = "2012-01-31",
                t1: str = "2022-12-31", max_gap: int = 6):
    """Rings on a common monthly axis -> (heads (n, T), obs (n, T), mask (n, T), names,
    dates). Heads are minus the drawdown anomaly, gaps up to ``max_gap`` months
    interpolated and the ends held; ``obs`` is re-zeroed to each site's first observed
    month, and the column prediction is compared after the same re-zeroing."""
    dates = pd.date_range(t0, t1, freq="ME")
    H, Ob, M, names = [], [], [], []
    for s, c in C.items():
        if s not in D or D[s].empty:
            continue
        h = (-D[s]).reindex(dates).interpolate(limit=max_gap, limit_area="inside")
        if h.isna().mean() > 0.2:
            continue
        h = h.ffill().bfill()
        o = c.reindex(dates)
        m = o.notna() & h.notna()
        if m.sum() < 24:
            continue
        first = int(np.argmax(m.to_numpy()))
        o = (o - o.iloc[first]).fillna(0.0)
        H.append(h.to_numpy())
        Ob.append(o.to_numpy())
        M.append(m.to_numpy())
        names.append(s)
    return np.array(H), np.array(Ob), np.array(M), names, dates


def _clamp(col, h_range: float, T: int) -> None:
    import torch

    with torch.no_grad():
        col.log_ske.clamp_(math.log(1e-6), math.log(1e-1))
        col.log_skv.clamp_(math.log(1e-5), math.log(1.0))
        col.log_tau.clamp_(math.log(1.0), math.log(T * col.dt_days))
        col.h_pc0.clamp_(-h_range, h_range)
        if col.creep == "aquitard":
            from .compaction import AQUITARD_BOUNDS

            for k, (lo, hi) in AQUITARD_BOUNDS.items():
                getattr(col, k).clamp_(lo, hi)


def fit_shared(H: np.ndarray, Ob: np.ndarray, M: np.ndarray, creep: str = "vep",
               epochs: int = 600, lr: float = 0.05):
    """One parameter set for every row (``VEPColumn(n_sites=1)`` broadcasts), masked MSE
    after re-zeroing each row's prediction at its first observed month."""
    import torch

    from .compaction import VEPColumn

    h = torch.tensor(H, dtype=torch.float64)
    o = torch.tensor(Ob, dtype=torch.float64)
    m = torch.tensor(M, dtype=torch.float64)
    first = torch.argmax(m, dim=1)
    col = VEPColumn(n_sites=1, creep=creep).to(torch.float64)
    opt = torch.optim.Adam(col.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = float(h.max() - h.min())
    for _ in range(epochs):
        opt.zero_grad()
        p = col(h)
        p = p - p.gather(1, first[:, None])
        loss = (((p - o) ** 2) * m).sum() / m.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(col.parameters(), 1.0)
        opt.step()
        sched.step()
        _clamp(col, rng, H.shape[1])
    return col


def predict(col, H: np.ndarray, M: np.ndarray) -> np.ndarray:
    import torch

    h = torch.tensor(H, dtype=torch.float64)
    first = torch.argmax(torch.tensor(M, dtype=torch.float64), dim=1)
    with torch.no_grad():
        p = col(h)
        return (p - p.gather(1, first[:, None])).numpy()


def loso_rings(H, Ob, M, creep: str, epochs: int = 600) -> tuple[np.ndarray, list[dict]]:
    """Leave-one-site-out predictions (n, T) of a shared column, and each fold's params."""
    pred = np.zeros_like(Ob)
    params = []
    for i in range(H.shape[0]):
        keep = np.arange(H.shape[0]) != i
        col = fit_shared(H[keep], Ob[keep], M[keep], creep=creep, epochs=epochs)
        pred[i] = predict(col, H[i:i + 1], M[i:i + 1])[0]
        params.append({k: float(getattr(col, k).detach().reshape(-1)[0])
                       for k in col.param_keys()})
    return pred, params


def masked_r2(p: np.ndarray, o: np.ndarray, m: np.ndarray) -> float:
    m = m.astype(bool)
    r = o[m] - p[m]
    return 1.0 - float((r ** 2).sum()) / max(float(((o[m] - o[m].mean()) ** 2).sum()), 1e-18)


def rate_anomaly_r2(pred: dict[str, pd.Series], obs: dict[str, pd.Series]) -> float:
    """Skill on the year-to-year variation of the rate: annual rates (differences of
    calendar-year means) of prediction and observation, each minus its site's mean rate,
    pooled R2. A site's mean rate is its initial disequilibrium, which no shared column can
    transfer; the variation about it is what the head drives."""
    p_all, o_all = [], []
    for n, o in obs.items():
        ro = annual_means(o).diff()
        rp = annual_means(pred[n]).diff()
        j = pd.concat([ro, rp], axis=1).dropna()
        j = j[j.index.isin([y for y in j.index if y - 1 in annual_means(o).dropna().index])]
        if len(j) < 3:
            continue
        o_all.append(j.iloc[:, 0] - j.iloc[:, 0].mean())
        p_all.append(j.iloc[:, 1] - j.iloc[:, 1].mean())
    o_v, p_v = np.concatenate(o_all), np.concatenate(p_all)
    return 1.0 - float(((o_v - p_v) ** 2).sum()) / max(float((o_v ** 2).sum()), 1e-18)


def rate_anomaly_agreement(model_tab: pd.DataFrame, obs_tab: pd.DataFrame) -> dict:
    """A model's year-to-year ring rates against the rings': both annual-rate tables
    joined on (site, year), each rate minus its site's mean over the joined years. The
    correlation is scale-free (the column is the whole-depth subsidence, the rings cover
    the top 200-300 m); the slope is model per observed."""
    j = model_tab[["site", "year", "R"]].merge(obs_tab[["site", "year", "R"]],
                                               on=["site", "year"], suffixes=("_m", "_o"))
    if len(j) < 5:
        return {"n": int(len(j))}
    for c in ("R_m", "R_o"):
        j[c] = j[c] - j.groupby("site")[c].transform("mean")
    corr = float(np.corrcoef(j.R_m, j.R_o)[0, 1])
    slope = float((j.R_m * j.R_o).sum() / max(float((j.R_o ** 2).sum()), 1e-18))
    return {"corr": corr, "slope_model_per_obs": slope, "n": int(len(j))}


def ring_fit_comparison(C: dict[str, pd.Series], D: dict[str, pd.Series],
                        epochs: int = 600) -> dict:
    """VEP vs VEP + aquitard term, each one shared parameter set, on the ring totals with
    the observed nest heads: pooled leave-one-site-out R2 of the compaction, and the
    level regression rerun on each column's held-out predictions (does the column carry
    the rate-on-level dependence the rings show?)."""
    H, Ob, M, names, dates = ring_arrays(C, D)
    out = {"n_sites": len(names), "sites": names}
    for creep in ("vep", "aquitard"):
        pred, params = loso_rings(H, Ob, M, creep, epochs=epochs)
        Cp = {n: pd.Series(np.where(M[i], pred[i], np.nan), index=dates)
              for i, n in enumerate(names)}
        Dn = {n: D[n] for n in names}
        tab = rate_table(Cp, Dn)
        full = fit_shared(H, Ob, M, creep=creep, epochs=epochs)
        out[creep] = {
            "loso_r2": masked_r2(pred, Ob, M),
            "loso_rate_anomaly_r2": rate_anomaly_r2(Cp, {n: pd.Series(
                np.where(M[i], Ob[i], np.nan), index=dates) for i, n in enumerate(names)}),
            "insample_r2": masked_r2(predict(full, H, M), Ob, M),
            "level": fe_ols(tab, "R", ["dD0", "lev1"]) if len(tab) else None,
            "params_full": {k: float(getattr(full, k).detach().reshape(-1)[0])
                            for k in full.param_keys()},
            "params_folds": params}
    obs_tab = rate_table({n: pd.Series(np.where(M[i], Ob[i], np.nan), index=dates)
                          for i, n in enumerate(names)}, {n: D[n] for n in names})
    out["observed_level"] = fe_ols(obs_tab, "R", ["dD0", "lev1"])
    return out


# ``level``: a persistent head-level effect on the rate. ``lag``: a transient (delayed)
# one. ``lag_level``/``lag2_level``: both at once -- the level coefficient is then the part
# of the response still present after one/two years, which is the policy question.
REGRESSIONS = {
    "level": ["dD0", "lev1"],
    "lag": ["dD0", "dD1"],
    "lag_level": ["dD0", "dD1", "lev2"],
    "lag2_level": ["dD0", "dD1", "dD2", "lev3"],
    "level_virgin": ["dD0", "lev1", "V0", "V1"],
}
VARIANTS = {"": {}, "_trend": {"trend": True}, "_yearfe": {"year_fe": True}}


def regress_all(tab: pd.DataFrame, cluster: str = "site") -> dict:
    out = {}
    for name, xs in REGRESSIONS.items():
        for suffix, kw in VARIANTS.items():
            try:
                out[name + suffix] = fe_ols(tab, "R", xs, group="site", cluster=cluster, **kw)
            except (np.linalg.LinAlgError, ValueError, KeyError) as e:
                out[name + suffix] = {"error": str(e)}
    return out


def recovery_experiment(C: dict[str, pd.Series], D: dict[str, pd.Series]) -> pd.DataFrame:
    """Per site: the mean annual rate before the 2021 drought (2015-2020), across the
    2021 -> 2023 gap (drought and recovery; the rings have no 2022 surveys) and after
    (2023 -> 2024), with the drawdown for the same years where heads exist."""
    rows = []
    for s, c in C.items():
        ca = annual_means(c)
        da = annual_means(D[s]) if s in D else pd.Series(dtype="float64")
        pre = [ca.get(y) - ca.get(y - 1) for y in range(2015, 2021)
               if np.isfinite(ca.get(y, np.nan)) and np.isfinite(ca.get(y - 1, np.nan))]
        row = {"site": s, "rate_2014_2020_cm": 100 * float(np.mean(pre)) if pre else np.nan,
               "rate_2021_2023_cm": 100 * float((ca.get(2023, np.nan) - ca.get(2021, np.nan))
                                                / 2.0),
               "rate_2023_2024_cm": 100 * float(ca.get(2024, np.nan) - ca.get(2023, np.nan))}
        for y in (2020, 2021, 2022, 2023, 2024):
            row[f"D_{y}_m"] = float(da.get(y, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------
# the leveling network next to a head nest
# ---------------------------------------------------------------------------------------
def head_nests(stations: pd.DataFrame, wells_dir: str, radius_m: float = 300.0
               ) -> dict[str, tuple[tuple[float, float], pd.Series]]:
    """{nest id (first six digits of the well id) -> (xy, monthly drawdown anomaly)} for
    every observation nest with cached well records."""
    xy = {}
    for _, r in stations.iterrows():
        p = _xy(r["LocationByTWD97"])
        if p is not None and os.path.exists(os.path.join(wells_dir, f"{r['sid']}.parquet")):
            xy.setdefault(str(r["sid"])[:6], []).append(p)
    out = {}
    for nid, pts in xy.items():
        c = tuple(np.mean(np.array(pts), axis=0))
        d, used = nest_drawdown(stations, wells_dir, c, radius_m=radius_m)
        if len(d.dropna()) >= 60:
            out[nid] = (c, d)
    return out


def leveling_table(sub: dict[str, pd.Series], bxy: dict[str, tuple[float, float]],
                   nests: dict, radius_m: float = 1500.0, min_gap_yr: float = 0.7,
                   max_gap_yr: float = 1.4) -> pd.DataFrame:
    """Survey-to-survey rates at benchmarks within ``radius_m`` of a head nest.

    Per consecutive survey pair ``(t0, t1]`` about a year apart: ``R`` the sinking rate
    (m/yr), ``dDe`` the drawdown change between the two survey months (the elastic step;
    the surveys share a season, so this is small), ``levI`` the mean drawdown over the
    interval about the nest's mean (the level the rate should follow if creep depends on
    it) and ``levP`` the same over the twelve months before ``t0``."""
    ids = list(nests)
    cxy = np.array([nests[i][0] for i in ids])
    rows = []
    for sid, s in sub.items():
        if sid not in bxy:
            continue
        d2 = ((cxy - np.array(bxy[sid])) ** 2).sum(1)
        k = int(np.argmin(d2))
        if math.sqrt(d2[k]) > radius_m:
            continue
        nid = ids[k]
        D = nests[nid][1]
        dmean = float(D.mean())
        s = s.dropna()
        for (t0, v0), (t1, v1) in zip(s.items(), list(s.items())[1:], strict=False):
            gap = (t1 - t0).days / 365.25
            if not min_gap_yr <= gap <= max_gap_yr:
                continue
            m0, m1 = t0 + pd.offsets.MonthEnd(0), t1 + pd.offsets.MonthEnd(0)
            inside = D[(D.index > m0) & (D.index <= m1)]
            before = D[(D.index > m0 - pd.DateOffset(months=12)) & (D.index <= m0)]
            if m0 not in D.index or m1 not in D.index or len(inside.dropna()) < 9:
                continue
            rows.append({"site": sid, "nest": nid, "year": int(t1.year),
                         "R": float((v1 - v0) / gap), "dDe": float(D[m1] - D[m0]),
                         "levI": float(inside.mean() - dmean),
                         "levP": float(before.mean() - dmean) if len(before.dropna()) >= 9
                         else np.nan})
    tab = pd.DataFrame(rows).dropna(subset=["dDe"])
    if len(tab):
        # levI + levP * (levP - levI) reparametrised: the levI coefficient of the
        # "sustained" regression is the rate change a level held for two surveys leaves
        tab["dLevP"] = tab["levP"] - tab["levI"]
    return tab


def leveling_analysis(ddir: str, stations_path: str, wells_dir: str,
                      radius_m: float = 1500.0) -> tuple[dict, pd.DataFrame]:
    """The level-vs-step regression on the leveling benchmarks next to a head nest, with a
    fixed effect per benchmark and standard errors clustered by nest."""
    from .leveling import load_panel, site_subsidence, site_xy

    stn = pd.read_parquet(stations_path)
    stn = stn[stn.GroundwaterZoneIdentifier == 50]
    nests = head_nests(stn, wells_dir)
    panel = load_panel(ddir)
    sub = site_subsidence(panel, "2012-01-01", "2023-01-01", min_obs=5, max_rate=0.5)
    tab = leveling_table(sub, site_xy(panel), nests, radius_m=radius_m)
    out = {"n_nests": int(tab.nest.nunique()) if len(tab) else 0,
           "n_benchmarks": int(tab.site.nunique()) if len(tab) else 0,
           "radius_m": radius_m}
    specs = {"level": ["dDe", "levI"], "level_prev": ["dDe", "levI", "levP"],
             "sustained": ["dDe", "levI", "dLevP"]}
    for name, xs in specs.items():
        for suffix, kw in VARIANTS.items():
            try:
                out[name + suffix] = fe_ols(tab, "R", xs, group="site", cluster="nest", **kw)
            except (np.linalg.LinAlgError, ValueError, KeyError) as e:
                out[name + suffix] = {"error": str(e)}
    return out, tab


def _interval_drivers(D: pd.Series, t0, t1) -> tuple[float, float, float] | None:
    """(dDe, levI, levP) of :func:`leveling_table` for one interval, None if short."""
    m0, m1 = t0 + pd.offsets.MonthEnd(0), t1 + pd.offsets.MonthEnd(0)
    inside = D[(D.index > m0) & (D.index <= m1)].dropna()
    before = D[(D.index > m0 - pd.DateOffset(months=12)) & (D.index <= m0)].dropna()
    if m0 not in D.index or m1 not in D.index or len(inside) < 9 or len(before) < 9:
        return None
    if not (np.isfinite(D[m0]) and np.isfinite(D[m1])):
        return None
    dm = float(D.mean())
    return float(D[m1] - D[m0]), float(inside.mean() - dm), float(before.mean() - dm)


def matched_ring_leveling(C: dict[str, pd.Series], D: dict[str, pd.Series],
                          site_xy_: dict[str, tuple[float, float]], sub: dict[str, pd.Series],
                          bxy: dict[str, tuple[float, float]], radius_m: float = 1000.0,
                          min_gap_yr: float = 0.7, max_gap_yr: float = 1.4) -> pd.DataFrame:
    """Ring and leveling rates over IDENTICAL intervals: every survey pair of a benchmark
    within ``radius_m`` of a ring site, the ring's compaction over the same two months,
    and the ring nest's drivers. Any difference in the regressions is then the ground
    motion outside the ring interval (below it, or in the top metres), not the design."""
    rows = []
    for site, c in C.items():
        if site not in D or D[site].empty:
            continue
        x, y = site_xy_[site]
        for sid, s in sub.items():
            if sid not in bxy or math.hypot(bxy[sid][0] - x, bxy[sid][1] - y) > radius_m:
                continue
            s = s.dropna()
            for (t0, v0), (t1, v1) in zip(s.items(), list(s.items())[1:], strict=False):
                gap = (t1 - t0).days / 365.25
                if not min_gap_yr <= gap <= max_gap_yr:
                    continue
                m0, m1 = t0 + pd.offsets.MonthEnd(0), t1 + pd.offsets.MonthEnd(0)
                if not (np.isfinite(c.get(m0, np.nan)) and np.isfinite(c.get(m1, np.nan))):
                    continue
                dr = _interval_drivers(D[site], t0, t1)
                if dr is None:
                    continue
                rows.append({"site": f"{site}|{sid}", "nest": site, "year": int(t1.year),
                             "R_leveling": float((v1 - v0) / gap),
                             "R_ring": float((c[m1] - c[m0]) / gap),
                             "dDe": dr[0], "levI": dr[1], "levP": dr[2],
                             "dLevP": dr[2] - dr[1]})
    return pd.DataFrame(rows)


def matched_compare(mt: pd.DataFrame) -> dict:
    """Rings vs leveling on identical intervals, compared directly (review 2026-09-29):
    the correlation of the two rates (raw, and about each benchmark's mean) and the
    held-level (``sustained``) effect of the ring rate, the leveling rate and their
    DIFFERENCE (leveling minus ring) under each variant. Two separate intervals that
    overlap are not a disagreement; only the difference regression tests one."""
    d = mt.dropna(subset=["R_leveling", "R_ring"]).copy()
    if len(d) < 5:
        return {"n": int(len(d))}
    a = d.R_leveling - d.groupby("site").R_leveling.transform("mean")
    b = d.R_ring - d.groupby("site").R_ring.transform("mean")
    out: dict = {"n": int(len(d)),
                 "corr_raw": float(np.corrcoef(d.R_leveling, d.R_ring)[0, 1]),
                 "corr_within_benchmark": float(np.corrcoef(a, b)[0, 1])}
    d["R_diff"] = d.R_leveling - d.R_ring
    for ycol in ("R_leveling", "R_ring", "R_diff"):
        for suffix, kw in VARIANTS.items():
            try:
                out[f"{ycol}_sustained{suffix}"] = fe_ols(
                    d, ycol, ["dDe", "levI", "dLevP"], group="site", cluster="nest", **kw)
            except (np.linalg.LinAlgError, ValueError, KeyError) as e:
                out[f"{ycol}_sustained{suffix}"] = {"error": str(e)}
    return out


def head_regime(drv: np.ndarray, origin: int, t_pol: int) -> dict:
    """Does a projection's baseline head at the ring cells go below its record minimum
    after ``t_pol``? A VEP column slows under a sustained rise only while heads set new
    minima, so this, not the column, decides whether a model's policy effect grows
    (review 2026-09-29). Per ring: projected minimum minus record minimum (m)."""
    gap = drv[:, t_pol:].min(axis=1) - drv[:, :origin + 1].min(axis=1)
    return {"proj_min_minus_record_min_m": gap.tolist(),
            "n_rings_new_minima": int((gap < 0).sum()), "n_rings": int(len(gap))}


def load_rings_and_heads(ddir: str, stations_path: str, wells_dir: str,
                         long_heads: bool = False) -> tuple[dict, dict, pd.DataFrame]:
    """-> ({site: band compaction DataFrame}, {site: drawdown Series}, site table)."""
    st = load_mlcw_stations(os.path.join(ddir, "mlcw_stations.csv"))
    pos = ring_positions(ddir)
    stn = pd.read_parquet(stations_path)
    stn = stn[stn.GroundwaterZoneIdentifier == 50]
    C, D, meta = {}, {}, []
    for _, r in st.iterrows():
        name = r["sub_id"]
        if name not in pos:
            continue
        C[name] = band_compaction(pos[name])
        if long_heads:
            d, used = _long_head(ddir, stn, (r["x"], r["y"]))
        else:
            d, used = nest_drawdown(stn, wells_dir, (r["x"], r["y"]))
        D[name] = d
        meta.append({"site": name, "x": float(r["x"]), "y": float(r["y"]),
                     "wells": ",".join(used), "n_wells": len(used),
                     "bands": ",".join(C[name].columns)})
    return C, D, pd.DataFrame(meta)


def _long_head(ddir: str, stn: pd.DataFrame, xy) -> tuple[pd.Series, list[str]]:
    """Drawdown anomaly from the 2010-2025 ``ls_cache`` records (single wells) in range."""
    anoms, used = [], []
    for _, r in stn.iterrows():
        p = _xy(r["LocationByTWD97"])
        if p is None or math.hypot(p[0] - xy[0], p[1] - xy[1]) > NEST_RADIUS_M:
            continue
        f = os.path.join(ddir, "ls_cache", f"gw__{r['sid']}.parquet")
        if not os.path.exists(f):
            continue
        m = _monthly_well(f)
        if len(m) < 60:
            continue
        anoms.append(-(m - m.mean()))
        used.append(str(r["sid"]))
    if not anoms:
        return pd.Series(dtype="float64"), used
    return pd.concat(anoms, axis=1).mean(axis=1).sort_index(), used


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="rebound vs slowed creep on the ring record")
    ap.add_argument("--data", default=None)
    ap.add_argument("--stations", default="AMP_V2/data/fan_stations.parquet")
    ap.add_argument("--wells-dir", default="AMP_V2/data/wells")
    ap.add_argument("--models", default=(
        "datum=results/twin_forward/datum_gate.npz:"
        "results/twin_runs/stage3_datum_gate/coupled_leveling/vep_zonal_leveling.json:"
        "results/twin_runs/stage3_datum_gate/stage3_theta.json,"
        "previous=results/twin_forward/physical_spread_apex.npz:"
        "results/twin_runs/stage3_spreadL_gate/coupled_leveling/vep_zonal_leveling.json:"
        "results/twin_runs/stage3_spreadL_gate/stage3_theta.json"),
        help="label=forward.npz:column.json:theta.json,...")
    ap.add_argument("--step-year", type=int, default=2016)
    ap.add_argument("--policy-start", default="2026-01-01",
                    help="month of the projection-regime +1 m step (the policies' start)")
    ap.add_argument("--ring-fit-epochs", type=int, default=600,
                    help="epochs per fit of the VEP vs VEP+aquitard ring comparison "
                         "(0 = skip it)")
    ap.add_argument("--out", default="results/twin/mechanism")
    args = ap.parse_args(argv)

    from ..config import Config
    from .zones import fan_zones

    ddir = str(Config(data_dir=args.data).data_dir if args.data else Config().data_dir)
    os.makedirs(args.out, exist_ok=True)
    result: dict = {"data": {}, "models": {}}

    # ---- the record ------------------------------------------------------------------
    C, D, meta = load_rings_and_heads(ddir, args.stations, args.wells_dir)
    meta.to_csv(os.path.join(args.out, "rings_sites.csv"), index=False)
    seas_rows = []
    for s, df in C.items():
        for band in df.columns:
            if D[s].empty:
                continue
            r = seasonal_slope(df[band], D[s])
            seas_rows.append({"site": s, "band": band, "slope_m_per_m": r["slope"],
                              "r2": r["r2"], "lag_months": r["best_lag"],
                              "slope_lag0": r["by_lag"][0]["slope"],
                              "r2_lag0": r["by_lag"][0]["r2"]})
    seas = pd.DataFrame(seas_rows)
    seas.to_csv(os.path.join(args.out, "rings_seasonal.csv"), index=False)
    tot = seas[seas.band == "total"]
    result["data"]["seasonal_total"] = {
        "median_slope_mm_per_m": 1000 * float(tot.slope_m_per_m.median()),
        "iqr_mm_per_m": [1000 * float(tot.slope_m_per_m.quantile(q)) for q in (0.25, 0.75)],
        "median_r2": float(tot.r2.median()), "median_lag_months": float(tot.lag_months.median()),
        "n_sites": int(len(tot))}

    tabs = {}
    Ctot = {s: df["total"] for s, df in C.items()}
    tabs["rings_total"] = rate_table(Ctot, D)
    for bi in range(len(BAND_EDGES_M)):
        Cb = {s: df.iloc[:, 1 + bi] for s, df in C.items() if df.shape[1] > 1 + bi}
        tabs[f"rings_band{bi}"] = rate_table(Cb, D)
    for k, t in tabs.items():
        t.to_csv(os.path.join(args.out, f"{k}_annual.csv"), index=False)
        result["data"][k] = regress_all(t)
        result["data"][k]["R_mean_cm_per_yr"] = 100 * float(t.R.mean()) if len(t) else None
    # the 2010-2025 single-well records for the six sites that have them: 2023-24 enter
    _, Dl, _ = load_rings_and_heads(ddir, args.stations, args.wells_dir, long_heads=True)
    Dl = {s: v for s, v in Dl.items() if not v.empty}
    tl = rate_table({s: Ctot[s] for s in Dl}, Dl)
    tl.to_csv(os.path.join(args.out, "rings_total_longheads_annual.csv"), index=False)
    result["data"]["rings_total_longheads"] = regress_all(tl)

    # ---- the leveling network, and rings vs leveling over identical intervals --------
    lev_out, lev_tab = leveling_analysis(ddir, args.stations, args.wells_dir)
    lev_tab.to_csv(os.path.join(args.out, "leveling_intervals.csv"), index=False)
    result["data"]["leveling"] = lev_out
    from .leveling import load_panel, site_subsidence, site_xy

    panel = load_panel(ddir)
    sub = site_subsidence(panel, "2012-01-01", "2023-01-01", min_obs=5, max_rate=0.5)
    mt = matched_ring_leveling(Ctot, D, {r.site: (r.x, r.y) for r in meta.itertuples()},
                               sub, site_xy(panel))
    mt.to_csv(os.path.join(args.out, "rings_vs_leveling_intervals.csv"), index=False)
    result["data"]["matched"] = {"n_intervals": int(len(mt)),
                                 "n_benchmarks": int(mt.site.nunique()) if len(mt) else 0}
    for ycol in ("R_leveling", "R_ring"):
        for name, xs in {"level_prev": ["dDe", "levI", "levP"],
                         "sustained": ["dDe", "levI", "dLevP"]}.items():
            result["data"]["matched"][f"{ycol}_{name}"] = fe_ols(
                mt, ycol, xs, group="site", cluster="nest") if len(mt) else None
    result["data"]["matched_compare"] = matched_compare(mt) if len(mt) else None
    rec =recovery_experiment(Ctot, {**D, **Dl})
    rec.to_csv(os.path.join(args.out, "rings_recovery.csv"), index=False)
    result["data"]["recovery"] = {
        "median_rate_2014_2020_cm": float(rec.rate_2014_2020_cm.median()),
        "median_rate_2021_2023_cm": float(rec.rate_2021_2023_cm.median()),
        "median_rate_2023_2024_cm": float(rec.rate_2023_2024_cm.median()),
        "n_sites": int(rec.rate_2023_2024_cm.notna().sum())}
    if args.ring_fit_epochs > 0:
        print(f"ring fit: VEP vs VEP+aquitard, leave one site out, {args.ring_fit_epochs} "
              "epochs per fold (CPU, several minutes)", flush=True)
        result["ring_fit"] = ring_fit_comparison(Ctot, D, epochs=args.ring_fit_epochs)

    # ---- the models ------------------------------------------------------------------
    xy = meta[["x", "y"]].to_numpy()
    names = meta.site.tolist()
    for spec in [m for m in args.models.split(",") if m.strip()]:
        label, rest = spec.split("=", 1)
        npz_path, col_path, theta_path = rest.split(":")
        z = np.load(npz_path, allow_pickle=True)
        cells = ring_cells(z, xy)
        Cm, Dm, drv = model_ring_series(z, cells, names)
        params = column_params(col_path)
        with open(theta_path) as fh:
            zb = json.load(fh).get("zone_boundaries", "205,182")
        prox, dist = (float(v) for v in str(zb).split(",")[:2])
        zones = fan_zones(xy[cells >= 0], prox, dist)
        # the same column on the OBSERVED nest heads: does its mechanism, whatever heads
        # drive it, produce the rate-on-level dependence the rings show?
        zone_of = dict(zip([n for n, c in zip(names, cells, strict=True) if c >= 0],
                           zones, strict=True))
        Co, Do = {}, {}
        for n, dser in D.items():
            if n in zone_of and not dser.empty:
                c_sim = column_on_series(params[int(zone_of[n])], -dser)
                if c_sim is not None:
                    Co[n], Do[n] = c_sim, dser
        to = rate_table(Co, Do)
        to = to[to.year.isin(set(tabs["rings_total"].year))]
        tm = rate_table(Cm, Dm)
        tm.to_csv(os.path.join(args.out, f"model_{label}_annual.csv"), index=False)
        tm_matched = tm[tm.year.isin(set(tabs["rings_total"].year))]
        seas_m = [seasonal_slope(Cm[s], Dm[s])["by_lag"][0]["slope"] for s in Cm]
        t_step = 12 * (args.step_year - 2012)
        resp = np.zeros_like(drv)
        for zi, p in enumerate(params):
            sel = zones == zi
            if sel.any():
                resp[sel] = step_response(p, drv[sel], t_step)
        # the same +1 m step inside the projection (the policy regime: heads no longer
        # set new minima), from the policy start month, on the archive's baseline heads
        dates_all = pd.DatetimeIndex(pd.to_datetime(np.asarray(z["dates"])))
        t_pol = int(np.searchsorted(dates_all, pd.Timestamp(args.policy_start)))
        drv_all = np.asarray(z["heads_mean"][0], dtype="float64").mean(axis=0)[cells[cells >= 0]]
        resp_f = np.zeros_like(drv_all)
        for zi, p in enumerate(params):
            sel = zones == zi
            if sel.any():
                resp_f[sel] = step_response(p, drv_all[sel], t_pol)
        # the ring-fitted shared columns on this model's projected heads: what the rings'
        # mechanism would give in the policy regime (ring interval only, not leveling)
        ringfit_proj = {c: step_summary(step_response(
            result["ring_fit"][c]["params_full"], drv_all, t_pol), t_pol)
            for c in ("vep", "aquitard")} if result.get("ring_fit") else None
        result["models"][label] = {
            "head_regime": head_regime(drv_all, int(z["origin"]), t_pol),
            "step_plus1m_projection_ringfit": ringfit_proj,
            "npz": npz_path, "column": col_path, "zones_at_rings": zones.tolist(),
            "step_plus1m_projection": step_summary(resp_f, t_pol),
            "policy_start": args.policy_start,
            "seasonal_median_mm_per_m": 1000 * float(np.nanmedian(seas_m)),
            "regressions": regress_all(tm), "regressions_matched_years": regress_all(tm_matched),
            "regressions_observed_heads": regress_all(to),
            "R_mean_cm_per_yr": 100 * float(tm.R.mean()),
            "ring_rate_anomaly": rate_anomaly_agreement(tm, tabs["rings_total"]),
            "step_plus1m": step_summary(resp, t_step), "step_year": args.step_year,
            "column_params": params}

    with open(os.path.join(args.out, "mechanism.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    _print(result)


def _print(res: dict) -> None:
    def line(tag, r, x):
        if "coef" not in r:
            return
        print(f"  {tag:<28} {x}: {100 * r['coef'][x]:+.3f} cm/yr per m  "
              f"(95% CI {100 * r['ci95'][x][0]:+.3f} .. {100 * r['ci95'][x][1]:+.3f}; "
              f"MDE {100 * r['mde80'][x]:.3f}; n={r['n']}, clusters={r['n_clusters']})")

    print("seasonal elastic (rings, total):", res["data"]["seasonal_total"])
    for k, v in res["data"].items():
        if not k.startswith("rings"):
            continue
        print(k, "mean R", v.get("R_mean_cm_per_yr"))
        for rk in v:
            if isinstance(v[rk], dict) and "coef" in v[rk]:
                for x in v[rk]["coef"]:
                    line(rk, v[rk], x)
    print("recovery:", res["data"]["recovery"])
    for key in ("leveling", "matched"):
        blk = res["data"].get(key, {})
        for rk, v in blk.items():
            if isinstance(v, dict) and "coef" in v:
                for x in v["coef"]:
                    line(f"{key} {rk}", v, x)
    mc = res["data"].get("matched_compare") or {}
    if "corr_raw" in mc:
        print(f"matched: rate corr raw {mc['corr_raw']:.3f}, within benchmark "
              f"{mc['corr_within_benchmark']:.3f}")
        for rk in [k for k in mc if k.startswith("R_diff")]:
            line(f"matched {rk}", mc[rk], "levI")
    rf = res.get("ring_fit")
    if rf:
        ob = rf["observed_level"]
        print(f"ring fit ({rf['n_sites']} sites, observed heads): observed level "
              f"{100 * ob['coef']['lev1']:+.3f} / dD0 {100 * ob['coef']['dD0']:+.3f} cm/yr/m")
        for creep in ("vep", "aquitard"):
            r = rf[creep]
            lv = r["level"]
            print(f"  {creep:<9} LOSO R2 {r['loso_r2']:+.3f}  in-sample {r['insample_r2']:+.3f}"
                  f"  LOSO rate-anomaly R2 {r['loso_rate_anomaly_r2']:+.3f}"
                  f"  held-out level {100 * lv['coef']['lev1']:+.3f} "
                  f"(CI {100 * lv['ci95']['lev1'][0]:+.3f}..{100 * lv['ci95']['lev1'][1]:+.3f})"
                  f" dD0 {100 * lv['coef']['dD0']:+.3f}")
    for lab, m in res["models"].items():
        print(f"model {lab}: ring rate anomalies {m['ring_rate_anomaly']}")
        hr = m.get("head_regime") or {}
        print(f"model {lab}: projected heads below the record minimum at "
              f"{hr.get('n_rings_new_minima')}/{hr.get('n_rings')} ring cells; ring-fit "
              f"columns from the policy start: {m.get('step_plus1m_projection_ringfit')}")
        print(f"model {lab}: +1 m from {m['policy_start']} (projection) "
              f"{m['step_plus1m_projection']}")
        print(f"model {lab}: seasonal {m['seasonal_median_mm_per_m']:.2f} mm/m, "
              f"R {m['R_mean_cm_per_yr']:.2f} cm/yr, step {m['step_plus1m']}")
        for src in ("regressions", "regressions_observed_heads"):
            for rk in REGRESSIONS:
                for x in REGRESSIONS[rk]:
                    line(f"{src[12:] or 'own heads'} {rk}", m[src][rk], x)


if __name__ == "__main__":
    main()
