"""WP X5 library: leveling and ring panels, treatment geometry, synthetic DiD, placebos and
the long-difference secondary.

Synthetic difference-in-differences follows Arkhangelsky, Athey, Hirshberg, Imbens & Wager
(2021, AER 111(12), doi 10.1257/aer.20190159) as implemented in the authors' R package
``synthdid`` (``synthdid_estimate`` with its defaults: Frank-Wolfe weights with intercepts,
zeta_omega = (N1 T1)^(1/4) sigma, zeta_lambda = 1e-6 sigma, sigma = sd of the controls'
pre-period first differences, a 100-iteration pass, sparsification v <= max(v)/4 -> 0, then
up to 1e4 iterations with min decrease 1e-5 sigma).

No function here reads an outcome value unless its name says so (``read_leveling_values``,
``read_ring_values``); the geometry and survey-date functions read locations and dates only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import x5const as K
from numba import njit

LEVELING_COLS = ["sid", "datetime", "x_3826", "y_3826", "town"]


# ============================================================================ geometry
def proxy_wells() -> np.ndarray:
    """(n, 2) TWD97 coordinates of the Yunlin Taiwan Water wells of the 2022 inventory."""
    w = pd.read_csv(K.WELLS_LOCATED, usecols=["county", "agency", "x_twd97", "y_twd97"])
    t = w[(w.county == "Yunlin") & (w.agency == K.TWC_AGENCY)]
    return t[["x_twd97", "y_twd97"]].to_numpy(float)


def grid_centroids() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(K.ZONE_MAP, allow_pickle=True)
    return z["centroids"].astype(float), z["changhua"].astype(bool)


def _mindist(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return d.min(1), d.argmin(1)


def read_survey_dates() -> pd.DataFrame:
    """Survey dates, location and township of every benchmark (no elevation value)."""
    p = pd.read_parquet(K.LEVELING_DATA_DIR / "ls_cache/ls-wra-lsp-obs__choushui_panel.parquet",
                        columns=LEVELING_COLS)
    p["datetime"] = pd.to_datetime(p["datetime"])
    return p.drop_duplicates(["sid", "datetime"]).sort_values(["sid", "datetime"])


def annual_picks(p: pd.DataFrame) -> pd.DataFrame:
    """One survey per benchmark and year: the one nearest to 15 May."""
    q = p.copy()
    q["year"] = q.datetime.dt.year
    q["off"] = (q.datetime.dt.dayofyear - K.SURVEY_TARGET_DOY).abs()
    q = q.sort_values(["sid", "year", "off", "datetime"])
    return q.drop_duplicates(["sid", "year"]).drop(columns="off")


def unit_table(p: pd.DataFrame | None = None) -> pd.DataFrame:
    """Benchmarks with geometry, sets and date-only balance flags (no elevation value)."""
    if p is None:
        p = read_survey_dates()
    a = annual_picks(p)
    u = a.groupby("sid").agg(x=("x_3826", "first"), y=("y_3826", "first"),
                             town=("town", "first"))
    C, ch = grid_centroids()
    dg, ig = _mindist(u[["x", "y"]].to_numpy(float), C)
    u["ongrid"] = dg <= K.ONGRID_M
    u["changhua"] = ch[ig]
    u["dW"], _ = _mindist(u[["x", "y"]].to_numpy(float), proxy_wells())
    # date balance: a pick in every year YEARS[0]-1..YEARS[-1] and every interval admissible
    piv = a.pivot(index="sid", columns="year", values="datetime")
    yrs = [K.YEARS[0] - 1] + K.YEARS
    ok = pd.Series(True, index=piv.index)
    for y in yrs:
        ok &= piv[y].notna() if y in piv.columns else False
    for y in K.YEARS:
        if y in piv.columns and y - 1 in piv.columns:
            dt = (piv[y] - piv[y - 1]).dt.days / 365.25
            ok &= dt.between(K.DT_MIN, K.DT_MAX)
    u["date_balanced"] = ok.reindex(u.index).fillna(False).astype(bool)
    u["T1"] = u.town.isin(K.T1_TOWNS)
    u["GC"] = u.town.isin(K.GC_TOWNS)
    t1 = u[u.T1 & u.ongrid][["x", "y"]].to_numpy(float)
    u["dT1"], _ = _mindist(u[["x", "y"]].to_numpy(float), t1)
    u["blk"] = ((u.x // K.BLOCK_M).astype(int).astype(str) + "_"
                + (u.y // K.BLOCK_M).astype(int).astype(str))
    return u


def arm_sets(u: pd.DataFrame, arm: str, r_treat: float = K.R_TREAT,
             r_buffer: float = K.R_BUFFER) -> tuple[pd.Index, pd.Index]:
    """(treated, donors) for an arm over usable units ``u`` (already filtered).

    T2 (primary): treated within r_treat of a proxy well; donors farther than r_buffer from
    every proxy well, outside the T1 townships and farther than 3 km from any T1 benchmark.
    T1: treated in the T1 townships; donors as T2's donors (always > 6 km from proxy wells).
    GC3: treated = 北港 benchmarks (Golden Corridor 2020 extension, no B02 stop); donors as T2.
    """
    base_donor = (u.dW > K.R_BUFFER) & (~u.T1) & (u.dT1 > K.T1_DONOR_GAP)
    if arm == "T2":
        tr = u.dW <= r_treat
        don = (u.dW > r_buffer) & (~u.T1) & (u.dT1 > K.T1_DONOR_GAP)
    elif arm == "T1":
        tr, don = u.T1, base_donor
    elif arm == "GC3":
        tr = u.town == "北港鎮"
        don = base_donor & ~tr
    else:
        raise ValueError(arm)
    return u.index[tr & ~don], u.index[don & ~tr]


# ============================================================================ outcomes
def read_leveling_values(units: pd.Index) -> pd.DataFrame:
    """OUTCOME READ: wide S[sid, year] (cm/yr, + = sinking) for the given benchmarks, from the
    annual picks; NaN where an interval is inadmissible."""
    p = pd.read_parquet(K.LEVELING_DATA_DIR / "ls_cache/ls-wra-lsp-obs__choushui_panel.parquet",
                        columns=["sid", "datetime", "elev_m"])
    p["datetime"] = pd.to_datetime(p["datetime"])
    p = p[p.sid.isin(set(units))].drop_duplicates(["sid", "datetime"])
    a = annual_picks(p.assign(x_3826=np.nan, y_3826=np.nan, town=""))
    e = a.pivot(index="sid", columns="year", values="elev_m")
    t = a.pivot(index="sid", columns="year", values="datetime")
    out = {}
    for y in sorted(set(e.columns)):
        if y - 1 not in e.columns:
            continue
        dt = (t[y] - t[y - 1]).dt.days / 365.25
        s = -(e[y] - e[y - 1]) / dt * 100.0
        out[y] = s.where(dt.between(K.DT_MIN, K.DT_MAX))
    return pd.DataFrame(out).reindex(units)


def interval_bounds(units: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(t0, t1) wide tables of the annual-pick interval bounds (dates only)."""
    a = annual_picks(read_survey_dates())
    t = a.pivot(index="sid", columns="year", values="datetime").reindex(units)
    t0 = pd.DataFrame({y: t[y - 1] for y in t.columns if y - 1 in t.columns})
    t1 = pd.DataFrame({y: t[y] for y in t.columns if y - 1 in t.columns})
    return t0, t1


# ============================================================================ census energy
class Energy:
    """Census kWh within a radius of each benchmark, annualised over an interval (GWh/yr in
    the product's model units)."""

    def __init__(self, units: pd.DataFrame, radius: float):
        z = np.load(K.CENSUS_CELLS, allow_pickle=True)
        self.E = np.nan_to_num(z[K.CENSUS_VAR].astype(float))
        self.unit = str(z["unit"])
        self.month = pd.PeriodIndex(z["dates"], freq="M")
        C, _ = grid_centroids()
        d = np.sqrt(((units[["x", "y"]].to_numpy(float)[:, None, :] - C[None]) ** 2).sum(-1))
        self.near = {s: np.where(d[i] <= radius)[0] for i, s in enumerate(units.index)}
        self.cum = np.concatenate([np.zeros((self.E.shape[0], 1)), np.cumsum(self.E, 1)], 1)

    def window(self, sid: str, t0, t1, shift_years: int = 0) -> float:
        if pd.isna(t0) or pd.isna(t1):
            return np.nan
        a = (pd.Timestamp(t0) + pd.DateOffset(years=shift_years)).to_period("M")
        b = (pd.Timestamp(t1) + pd.DateOffset(years=shift_years)).to_period("M")
        i0, i1 = self.month.searchsorted(a), self.month.searchsorted(b)
        n = i1 - i0
        if n <= 0 or i1 > len(self.month):
            return np.nan
        cells = self.near[sid]
        tot = (self.cum[cells, i1] - self.cum[cells, i0]).sum()
        return float(tot / n * 12 / 1e6)

    def table(self, t0: pd.DataFrame, t1: pd.DataFrame, shift_years: int = 0) -> pd.DataFrame:
        return pd.DataFrame({y: [self.window(s, t0.at[s, y], t1.at[s, y], shift_years)
                                 for s in t0.index] for y in t0.columns}, index=t0.index)


# ============================================================================ SDID
@njit(cache=True)
def _fw(A, b, x, eta, zeta2, min_dec2, max_iter):
    n = A.shape[0]
    Ax = A @ x
    prev = np.inf
    t = 0
    while t < max_iter:
        g = (Ax - b) @ A + eta * x
        i = np.argmin(g)
        dx = -x.copy()
        dx[i] = 1.0 - x[i]
        if np.any(dx != 0.0):
            derr = A[:, i] - Ax
            den = derr @ derr + eta * (dx @ dx)
            step = -(g @ dx) / den if den > 0 else 0.0
            step = min(1.0, max(0.0, step))
            x = x + step * dx
            Ax = Ax + step * derr
        err = Ax - b
        val = zeta2 * (x @ x) + (err @ err) / n
        t += 1
        if t >= 2 and not (prev - val > min_dec2):
            break
        prev = val
    return x


def _sparsify(v: np.ndarray) -> np.ndarray:
    v = v.copy()
    v[v <= v.max() / 4] = 0.0
    return v / v.sum()


def _weights(Y: np.ndarray, zeta: float, min_dec: float) -> np.ndarray:
    """``synthdid::sc.weight.fw`` with intercept, two passes around sparsification. Y has the
    target in its last column; rows are the observations."""
    Yc = Y - Y.mean(0)
    A = np.ascontiguousarray(Yc[:, :-1])
    b = np.ascontiguousarray(Yc[:, -1])
    n, k = A.shape
    x0 = np.full(k, 1.0 / k)
    eta = n * zeta ** 2
    x = _fw(A, b, x0, eta, zeta ** 2, min_dec ** 2, 100)
    return _fw(A, b, _sparsify(x), eta, zeta ** 2, min_dec ** 2, 10000)


def sdid(Y: np.ndarray, N0: int, T0: int) -> dict:
    """Synthetic DiD ATT on Y (controls first, then treated; pre periods first)."""
    Y = np.asarray(Y, float)
    N, T = Y.shape
    N1, T1 = N - N0, T - T0
    sigma = np.std(np.diff(Y[:N0, :T0], axis=1), ddof=1)
    zeta_o = (N1 * T1) ** 0.25 * sigma
    zeta_l = 1e-6 * sigma
    min_dec = 1e-5 * sigma
    Yc = np.vstack([np.hstack([Y[:N0, :T0], Y[:N0, T0:].mean(1, keepdims=True)]),
                    np.hstack([Y[N0:, :T0].mean(0, keepdims=True),
                               Y[N0:, T0:].mean(keepdims=True).reshape(1, 1)])])
    lam = _weights(Yc[:N0, :], zeta_l, min_dec)
    om = _weights(Yc[:, :T0].T.copy(), zeta_o, min_dec)
    wr = np.concatenate([-om, np.full(N1, 1.0 / N1)])
    wc = np.concatenate([-lam, np.full(T1, 1.0 / T1)])
    tau = float(wr @ Y @ wc)
    tr_pre = Y[N0:, :T0].mean(0)
    sc_pre = om @ Y[:N0, :T0]
    off = (tr_pre - sc_pre).mean()
    return {"tau": tau, "omega": om, "lam": lam, "sigma": float(sigma),
            "pre_rmse": float(np.sqrt(((tr_pre - sc_pre - off) ** 2).mean())),
            "n_eff_donors": float(1.0 / (om ** 2).sum()), "N0": N0, "N1": N1,
            "T0": T0, "T1": T1}


def block_draws(blocks: np.ndarray, n_target: int, B: int, seed: int) -> list[np.ndarray]:
    """Pseudo-treated row sets: random orders of the donor blocks, accumulated until the set
    holds at least ``n_target`` units (placebo-in-space by blocks)."""
    rng = np.random.default_rng(seed)
    labs = np.unique(blocks)
    out = []
    for _ in range(B):
        sel, n = [], 0
        for lb in rng.permutation(labs):
            idx = np.where(blocks == lb)[0]
            sel.append(idx)
            n += len(idx)
            if n >= n_target:
                break
        out.append(np.sort(np.concatenate(sel)))
    return out


def sdid_with_placebo(Yt: np.ndarray, Yd: np.ndarray, donor_blocks: np.ndarray, T0: int,
                      B: int, seed: int) -> dict:
    """ATT of treated rows Yt against donor rows Yd, plus block placebos among the donors."""
    est = sdid(np.vstack([Yd, Yt]), len(Yd), T0)
    taus = []
    for rows in block_draws(donor_blocks, len(Yt), B, seed):
        mask = np.zeros(len(Yd), bool)
        mask[rows] = True
        if mask.all() or (~mask).sum() < 2:
            continue
        taus.append(sdid(np.vstack([Yd[~mask], Yd[mask]]), int((~mask).sum()), T0)["tau"])
    taus = np.asarray(taus)
    return summarize(est, taus)


def summarize(est: dict, taus: np.ndarray) -> dict:
    tau = est["tau"]
    se = float(np.std(taus, ddof=1)) if len(taus) > 1 else float("nan")
    n = len(taus)
    return {"tau": tau, "se_placebo": se,
            "ci90": [tau - K.Z90 * se, tau + K.Z90 * se],
            "p_one_sided_less_sinking": float((1 + (taus <= tau).sum()) / (1 + n)),
            "p_two_sided": float((1 + (np.abs(taus) >= abs(tau)).sum()) / (1 + n)),
            "n_placebo": n, "placebo_mean": float(taus.mean()) if n else float("nan"),
            "N0": est["N0"], "N1": est["N1"], "T0": est["T0"], "T1": est["T1"],
            "pre_rmse": est["pre_rmse"], "n_eff_donors": est["n_eff_donors"],
            "sigma": est["sigma"], "lambda": [float(v) for v in est["lam"]],
            "_taus": [float(v) for v in taus]}


def loo_placebo(Yt: np.ndarray, Yd: np.ndarray, T0: int) -> dict:
    """Small panels (rings): every donor in turn as the pseudo-treated unit."""
    est = sdid(np.vstack([Yd, Yt]), len(Yd), T0)
    taus = []
    for i in range(len(Yd)):
        m = np.ones(len(Yd), bool)
        m[i] = False
        taus.append(sdid(np.vstack([Yd[m], Yd[~m]]), int(m.sum()), T0)["tau"])
    return summarize(est, np.asarray(taus))


# ============================================================================ rings
def read_ring_values() -> pd.DataFrame:
    """OUTCOME READ: annual May-to-May band compaction rates (cm/yr, + = compaction), flag-0
    months of the primary series up to 2021-12; NaN when no flag-0 month lies within the
    tolerance of May."""
    r = pd.read_parquet(K.RINGS, columns=["site", "series", "band", "month", "value", "flag"])
    r = r[(r.series == K.RING_SERIES) & (r.flag == 0) & (r.month <= "2021-12")
          & r.band.isin(K.RING_BANDS)]
    r["m"] = pd.PeriodIndex(r.month, freq="M")
    rows = []
    for (site, band), g in r.groupby(["site", "band"]):
        g = g.set_index("m").value
        pick = {}
        for y in [K.RING_YEARS[0] - 1] + K.RING_YEARS:
            may = pd.Period(f"{y}-05", "M")
            c = [may + k for k in sorted(range(-K.RING_MAY_TOL_MONTHS,
                                               K.RING_MAY_TOL_MONTHS + 1), key=abs)]
            hit = [m for m in c if m in g.index]
            if hit:
                pick[y] = (hit[0], float(g[hit[0]]))
        for y in K.RING_YEARS:
            if y in pick and y - 1 in pick:
                (m1, v1), (m0, v0) = pick[y], pick[y - 1]
                n = (m1 - m0).n
                rows.append({"site": site, "band": band, "year": y,
                             "rate": (v1 - v0) / n * 12 / 10.0})
    return pd.DataFrame(rows)


# ============================================================================ secondary
def ols_slope(dr: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    return np.linalg.lstsq(Xc, dr - dr.mean(), rcond=None)[0]


def block_boot(df: pd.DataFrame, fn, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blks = df.blk.unique()
    groups = {b: df[df.blk == b] for b in blks}
    out = []
    for _ in range(B):
        pick = rng.choice(blks, len(blks))
        out.append(fn(pd.concat([groups[b] for b in pick])))
    return np.asarray(out)


def twfe(S: pd.DataFrame, X: dict[str, pd.DataFrame]) -> np.ndarray:
    """Two-way FE (unit, year) slopes of S on the regressors in X (alternating projections on
    the observed cells)."""
    long = S.stack().rename("S").to_frame()
    for k, v in X.items():
        long[k] = v.stack().reindex(long.index)
    long = long.dropna()
    sid = long.index.get_level_values(0).to_numpy()
    yr = long.index.get_level_values(1).to_numpy()
    D = long.to_numpy(float)
    for _ in range(100):
        D = D - pd.DataFrame(D).groupby(sid).transform("mean").to_numpy()
        D = D - pd.DataFrame(D).groupby(yr).transform("mean").to_numpy()
    return np.linalg.lstsq(D[:, 1:], D[:, 0], rcond=None)[0]
