"""WP S3 validation (v3 code, repair round 3 S3r, 2026-10-18; acceptance spec:
results/dataprep/reports/S3_evidence/acceptance_spec.json; v3 pre-registration
S3.gnss_clean_acceptance.v3). Unchanged from v2 apart from this note: A1, A2, A3, the
common-mode checks and the twin scoring are the v2 definitions (the seed is an argument).

v2 changes against v1 (A2 only): the criterion must hold for EACH common mode the product
offers for removal, ``ngl_offfan`` (the plan's off-fan common mode, the S4 reference) and
``wra_hf`` (v1 gated on ``wra_hf`` alone, whose high-pass leaves almost no December-minus-
January content to remove, so its "survives the removal" half was nearly vacuous); the
output adds each common mode's own 2021 extra, the pooled GNSS extra and its ratio to the
uncorrected one per variant, the interannual spread of the ``ngl_offfan`` extra over the
pre-origin record, and the per-GNSS-station (unique station) summary. ``monthly`` sorts its
month index (the v1 post-registration fix).

A1  injected steps (product pipeline; v3 seed 707, 2/3/5/10 cm x 10 repetitions);
A2  the 2021 extra compaction of rings and piles survives the common-mode removal at
    co-located WRA GNSS stations;
A3  excess-rate stability at co-located rings (reported);
plus the common-mode diagnostics (scatter before/after each candidate common mode, PCA
first-mode uniformity) and the after-the-fact scoring of the frozen forward archive against
GNSS rates, raw and CM-removed.

Reads the product tables written by ``build.py`` (passed in memory) and L0 rings/piles.
"""
from __future__ import annotations

import injection as INJ
import io_gnss as io
import model as M
import numpy as np
import pandas as pd
import pipeline as P

RING_DIR = io.REPO / "chou-shui-data" / "data"
PILES = {"TKJS_281_29": ("TKJS", "土庫 300 m"), "NTUH_282_29": ("NTUH", "中科虎尾 300 m"),
         "SJES": ("SJES", "溪州 300 m"), "STES": ("STES", "秀潭 70 m")}


# ----------------------------------------------------------------------------- helpers
def r2_rmse(pred: np.ndarray, obs: np.ndarray) -> dict:
    """R2 (1 - SSE/SST), RMSE and bias over the sites where both are finite (the same
    definition as the twin's ring/GNSS continuation test; reimplemented here so that this
    WP never imports ``hydrophysics/twin/ring_holdout.py``, which another session edits)."""
    m = np.isfinite(pred) & np.isfinite(obs)
    if m.sum() < 3:
        return {"r2": float("nan"), "rmse": float("nan"), "bias": float("nan"),
                "n": int(m.sum())}
    p, o = pred[m], obs[m]
    sst = float(((o - o.mean()) ** 2).sum())
    sse = float(((p - o) ** 2).sum())
    return {"r2": 1.0 - sse / sst if sst > 0 else float("nan"),
            "rmse": float(np.sqrt(sse / m.sum())), "bias": float((p - o).mean()),
            "n": int(m.sum())}


def _dt(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.assign(date=pd.to_datetime(daily.date))


def monthly(daily: pd.DataFrame, col: str, flags=(0, 1), min_days: int = 10) -> pd.DataFrame:
    """station x month table of the median of ``col`` over days with the given flags."""
    d = _dt(daily)
    d = d[d.flag.isin(flags)].dropna(subset=[col])
    d = d.assign(month=d.date.dt.to_period("M").dt.to_timestamp())
    g = d.groupby(["station", "month"])[col].agg(["median", "count"])
    g = g[g["count"] >= min_days]["median"].unstack("station")
    # unstack does not guarantee a sorted month index (stations starting at different
    # months); label slicing in ols_rate needs it sorted (fixed 2026-10-12 after the
    # pre-registration: the full-variant NGL scoring raised on an unsorted index; label
    # look-ups in A2 never depended on the order)
    return g.sort_index()


def ann_increment(s: pd.Series, year: int) -> float:
    """mean(December) - mean(January) of a monthly series (index = month start)."""
    a = s.get(pd.Timestamp(f"{year}-01-01"), np.nan)
    b = s.get(pd.Timestamp(f"{year}-12-01"), np.nan)
    return float(b - a)


def rings_monthly(windows=(("2020-01-01", "2021-12-01"),)
                  ) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """{site: monthly total compaction (m, + = shortening)} between the shallowest and the
    deepest of the rings read in >= 80 % of the months of EVERY window in ``windows``
    (A2: 2020-2021; A3: 2020-2021 and 2023-02..2025-03, because the deepest rings are not
    read after the 2022 instrument change), and site coordinates."""
    from hydrophysics.twin.mechanism import ring_positions  # read-only repository import
    pos = ring_positions(str(RING_DIR))
    st = pd.read_csv(RING_DIR / "mlcw_stations.csv")
    out = {}
    for site, p in pos.items():
        mon = p.groupby(p.index.to_period("M")).median()
        mon.index = mon.index.to_timestamp()
        w = mon.loc[windows[0][0]:windows[0][1]]
        cols = [c for c in mon.columns
                if all(mon.loc[a:b, c].notna().mean() >= 0.8 for a, b in windows)]
        if len(cols) < 2:
            continue
        z = w[cols].mean()
        top, bot = z.idxmin(), z.idxmax()
        sep = mon[bot] - mon[top]
        fv = sep.first_valid_index()
        out[site] = -(sep - sep.loc[fv])
    return out, st.rename(columns={"sub_id": "site"})[["site", "x", "y"]]


def piles_monthly() -> tuple[dict[str, pd.Series], pd.DataFrame]:
    meta = pd.read_parquet(io.LS_CACHE / "ls-wra-dbm-obs__meta.parquet",
                           columns=["_station_id", "TWD97坐標_X", "TWD97坐標_Y"])
    xy = {"TKJS_281_29": "TKJS(1)", "NTUH_282_29": "NTUH(1)", "SJES": "SJES", "STES": "STES"}
    out, rows = {}, []
    for f, (_, label) in PILES.items():
        d = pd.read_parquet(io.LS_CACHE / f"ls-wra-dbm-obs__{f}.parquet", columns=["value"])
        s = pd.to_numeric(d["value"], errors="coerce")
        s.index = pd.to_datetime(d.index)
        mon = s.resample("MS").mean() / 1000.0          # mm -> m, + = compression
        out[f] = mon
        r = meta[meta._station_id == xy[f]].iloc[0]
        rows.append({"pile": f, "label": label, "x": float(r["TWD97坐標_X"]),
                     "y": float(r["TWD97坐標_Y"])})
    return out, pd.DataFrame(rows)


def pairs(stations: pd.DataFrame, refs: pd.DataFrame, key: str, max_d: float = 1000.0
          ) -> list[tuple[str, str, float]]:
    out = []
    w = stations[stations.family == "wra"]
    for _, s in w.iterrows():
        d = np.hypot(refs.x - s.x_twd97, refs.y - s.y_twd97)
        j = int(np.argmin(d.to_numpy()))
        if d.iloc[j] <= max_d:
            out.append((s.station, refs.iloc[j][key], float(d.iloc[j])))
    return out


# ----------------------------------------------------------------------------- A2 / A3
#: A2 (v2): the common modes offered for removal; the criterion must hold for each
A2_GATING = ("cm_removed_ngl_offfan", "cm_removed_wra_hf")


def cm_extra(cms: pd.DataFrame, col: str, year: int = 2021) -> dict:
    """The common mode's own extra (height, mm): D(year) - D(year - 1), D = December mean
    minus January mean of the daily common mode, with the days per month; plus, for a common
    mode with a long record, the same statistic for every earlier year pair (interannual
    spread)."""
    c = cms[col].copy()
    c.index = pd.DatetimeIndex(pd.to_datetime(c.index))
    m = c.groupby(c.index.to_period("M")).agg(["mean", "count"])

    def D(y):
        a, b = pd.Period(f"{y}-01", "M"), pd.Period(f"{y}-12", "M")
        if a not in m.index or b not in m.index:
            return np.nan
        if m.loc[a, "count"] < 10 or m.loc[b, "count"] < 10:
            return np.nan
        return float(m.loc[b, "mean"] - m.loc[a, "mean"])
    out = {"extra_mm": 1000 * (D(year) - D(year - 1)), "D_mm": {}}
    for y in (year - 1, year):
        out["D_mm"][str(y)] = 1000 * D(y)
    years = sorted({p.year for p in m.index})
    hist = {}
    for y in years[1:]:
        e = D(y) - D(y - 1)
        if np.isfinite(e):
            hist[str(y)] = round(1000 * e, 2)
    out["extra_by_year_mm"] = hist
    others = np.array([v for k, v in hist.items() if k != str(year)])
    if others.size >= 5:
        out["other_years_sd_mm"] = float(np.std(others, ddof=1))
        out["other_years_n"] = int(others.size)
        out["rank_abs_of_year"] = int((np.abs(others) >= abs(out["extra_mm"])).sum())
    return out


def test_2021_extra(wra_daily: pd.DataFrame, cms: pd.DataFrame, stations: pd.DataFrame
                    ) -> dict:
    rings, rxy = rings_monthly()
    piles, pxy = piles_monthly()
    g_raw = -monthly(wra_daily, "value")
    variants = {"cm_removed_wra_hf": "wra_hf", "uncorrected": None,
                "cm_removed_pilot_onfan_fullband": "wra_onfan_fullband",
                "cm_removed_ngl_offfan": "ngl_offfan"}
    gser = {}
    for name, col in variants.items():
        if col is None:
            gser[name] = g_raw
            continue
        d = _dt(wra_daily).merge(cms[[col]].rename(columns={col: "cmx"}), left_on="date",
                                 right_index=True, how="left")
        d["v"] = d["value"] - d["cmx"]
        gser[name] = -monthly(d, "v")
    rows = []
    for st, site, dist in pairs(stations, rxy, "site"):
        if site in rings:
            rows.append(("ring", st, site, dist, rings[site]))
    for st, pile, dist in pairs(stations, pxy, "pile"):
        rows.append(("pile", st, pile, dist, piles[pile]))
    table = []
    for kind, st, ref, dist, rs in rows:
        rec = {"kind": kind, "gnss": st, "reference": ref, "distance_m": round(dist, 1),
               "ref_D2020_mm": 1000 * ann_increment(rs, 2020),
               "ref_D2021_mm": 1000 * ann_increment(rs, 2021)}
        rec["ref_extra_mm"] = rec["ref_D2021_mm"] - rec["ref_D2020_mm"]
        for name, g in gser.items():
            if st not in g:
                rec[f"{name}_extra_mm"] = np.nan
                continue
            s = g[st]
            rec[f"{name}_extra_mm"] = 1000 * (ann_increment(s, 2021) - ann_increment(s, 2020))
        table.append(rec)
    T = pd.DataFrame(table)
    out = {"pairs": T.round(2).to_dict(orient="records")}
    ev = T[(T.ref_extra_mm > 3.0)]
    for name in variants:
        e = ev.dropna(subset=[f"{name}_extra_mm"])
        if len(e) == 0:
            out[name] = {"n_evaluable": 0}
            continue
        same = float((np.sign(e[f"{name}_extra_mm"]) == np.sign(e.ref_extra_mm)).mean())
        ratio = float(e[f"{name}_extra_mm"].sum() / e.ref_extra_mm.sum())
        half = float((e[f"{name}_extra_mm"] >= 0.5 * e.ref_extra_mm).mean())
        rec = {"n_evaluable": int(len(e)), "same_sign_fraction": same,
               "pooled_amplitude_ratio": ratio, "fraction_at_least_half": half,
               "pooled_gnss_extra_mm": float(e[f"{name}_extra_mm"].sum()),
               "pooled_ref_extra_mm": float(e.ref_extra_mm.sum()),
               "pass": bool(same >= 0.75 and ratio >= 0.5)}
        # same pairs, uncorrected, for the removal effect
        both = e.dropna(subset=["uncorrected_extra_mm"])
        if len(both) and both.uncorrected_extra_mm.sum() != 0:
            rec["ratio_to_uncorrected_pooled"] = float(both[f"{name}_extra_mm"].sum()
                                                       / both.uncorrected_extra_mm.sum())
        # unique GNSS stations (a station paired with a ring and a pile counts once: its
        # reference extra is the mean over its evaluable pairs); reported, not gating
        u = e.groupby("gnss").agg(g=(f"{name}_extra_mm", "first"), r=("ref_extra_mm", "mean"))
        rec["unique_gnss"] = {"n": int(len(u)),
                              "same_sign_fraction": float((np.sign(u.g) == np.sign(u.r)).mean()),
                              "pooled_amplitude_ratio": float(u.g.sum() / u.r.sum())}
        out[name] = rec
    for col in ("ngl_offfan", "wra_hf", "wra_onfan_fullband"):
        if col in cms:
            out[f"cm_own_extra_{col}"] = cm_extra(cms, col)
    e = ev.dropna(subset=["cm_removed_wra_hf_extra_mm", "uncorrected_extra_mm"])
    out["ratio_cmr_to_uncorrected_pooled_wra_hf"] = float(
        e.cm_removed_wra_hf_extra_mm.sum() / e.uncorrected_extra_mm.sum()) \
        if len(e) and e.uncorrected_extra_mm.sum() != 0 else np.nan
    out["n_pairs"] = int(len(T))
    out["n_evaluable"] = int(len(ev))
    out["gating_variants"] = list(A2_GATING)
    out["pass"] = bool(all(out.get(v, {}).get("pass", False) for v in A2_GATING))
    return out


def ols_rate(s: pd.Series, t0: str, t1: str, min_n: int = 12) -> tuple[float, int]:
    s = s.dropna().loc[t0:t1]
    if len(s) < min_n:
        return np.nan, len(s)
    yrs = (s.index - s.index[0]).days / 365.25
    return float(np.polyfit(yrs, s.to_numpy(float), 1)[0]), len(s)


def excess_stability(wra_daily: pd.DataFrame, cms: pd.DataFrame, stations: pd.DataFrame
                     ) -> dict:
    rings, rxy = rings_monthly((("2020-01-01", "2021-12-01"), ("2023-02-01", "2025-03-01")))
    d = _dt(wra_daily).merge(cms[["wra_hf"]], left_on="date", right_index=True, how="left")
    d["v"] = d["value"] - d["wra_hf"].fillna(0.0)
    g = -monthly(d, "v")
    rows = []
    for st, site, dist in pairs(stations, rxy, "site"):
        if site not in rings or st not in g:
            continue
        rec = {"gnss": st, "site": site, "distance_m": round(dist, 1)}
        for w, (a, b) in {"pre": ("2020-01-01", "2021-12-01"),
                          "post": ("2023-02-01", "2025-03-01")}.items():
            rg, ng = ols_rate(g[st], a, b)
            rr, nr = ols_rate(rings[site], a, b)
            rec[f"{w}_gnss_mm_yr"] = 1000 * rg
            rec[f"{w}_ring_mm_yr"] = 1000 * rr
            rec[f"{w}_excess_mm_yr"] = 1000 * (rg - rr)
            rec[f"{w}_n_months"] = min(ng, nr)
        rec["excess_change_mm_yr"] = rec["post_excess_mm_yr"] - rec["pre_excess_mm_yr"]
        rows.append(rec)
    T = pd.DataFrame(rows)
    return {"table": T.round(2).to_dict(orient="records"),
            "median_pre_excess_mm_yr": float(T.pre_excess_mm_yr.median()),
            "median_post_excess_mm_yr": float(T.post_excess_mm_yr.median()),
            "median_abs_change_mm_yr": float(T.excess_change_mm_yr.abs().median()),
            "n_sites": int(len(T)), "pass": "reported (no threshold)"}


# ----------------------------------------------------------------------------- CM checks
def scatter_table(daily: pd.DataFrame, resid_col: str, cms: pd.DataFrame,
                  members: list[str], cols: list[str], t0: str, t1: str) -> dict:
    d = _dt(daily)
    d = d[d.flag.isin([0, 1]) & d.station.isin(members)]
    R = d.pivot(index="date", columns="station", values=resid_col).loc[t0:t1]
    out = {}
    base = [float(R[k].std()) for k in R if R[k].notna().sum() >= 300]
    out["none"] = 1000 * float(np.median(base))
    for c in cols:
        cm = cms[c].reindex(R.index)
        v = [float((R[k] - cm).std()) for k in R if (R[k] - cm).notna().sum() >= 300]
        out[c] = 1000 * float(np.median(v)) if v else np.nan
    return {k: round(v, 2) for k, v in out.items()}


def pca_check(daily: pd.DataFrame, members: list[str], cms: pd.DataFrame, window: int
              ) -> dict:
    d = _dt(daily)
    d = d[d.flag.isin([0, 1]) & d.station.isin(members)]
    R = d.pivot(index="date", columns="station", values="resid")
    hp = R - R.rolling(window, center=True, min_periods=window // 3).mean()
    keep = hp.notna().mean(axis=1) >= 0.8
    X = hp[keep]
    X = X.loc[:, X.notna().mean() >= 0.8]
    X = (X - X.mean()).fillna(0.0)
    u, s, vt = np.linalg.svd(X.to_numpy(), full_matrices=False)
    load = vt[0] * np.sign(vt[0].sum())
    pc1 = pd.Series(u[:, 0] * s[0] * np.sign(vt[0].sum()), index=X.index)
    off = cms["ngl_offfan"].reindex(X.index)
    off_hp = off - off.rolling(window, center=True, min_periods=window // 3).mean()
    return {"n_days": int(len(X)), "n_stations": int(X.shape[1]),
            "pc1_variance_fraction": float(s[0] ** 2 / (s ** 2).sum()),
            "pc1_loadings_positive_fraction": float((load > 0).mean()),
            "pc1_loadings_cv": float(load.std() / abs(load.mean())),
            "corr_pc1_wra_hf": float(pc1.corr(cms["wra_hf"].reindex(X.index))),
            "corr_pc1_ngl_offfan_highpass": float(pc1.corr(off_hp))}


# ----------------------------------------------------------------------------- scoring
def _monthly_variant(daily: pd.DataFrame, cms: pd.DataFrame | None, col: str | None,
                     value: str = "value", flags=(0, 1)) -> pd.DataFrame:
    if col is None:
        return monthly(daily, value, flags=flags)
    d = _dt(daily).merge(cms[[col]].rename(columns={col: "cmx"}), left_on="date",
                         right_index=True, how="left")
    d["v"] = d[value] - d["cmx"]
    return monthly(d, "v", flags=flags)


def twin_scores(daily: pd.DataFrame, cms: pd.DataFrame, stations: pd.DataFrame,
                family: str, t0: str, t1: str, members: list[str] | None = None) -> dict:
    """After-the-fact scoring of the frozen forward archive against GNSS subsidence rates
    (OLS over monthly values in [t0, t1], >= 12 months; sinking positive), raw and with each
    common mode removed. The model's own common mode is removed the same way: zero for the
    NGL off-fan median (every reference station is off the grid) and for the WRA
    high-pass median (``Params.hp_window`` days; the monthly model has no sub-monthly
    content); for the on-fan
    full-band median (diagnostic) the model's mean over the same member stations' cells."""
    z = np.load(io.REPO / "results/twin_forward/datum_gate_innov.npz", allow_pickle=False)
    dates = pd.DatetimeIndex(pd.to_datetime(z["dates"]))
    subs = np.asarray(z["subs_mean_by_rheology"][0, 0], dtype=float)
    st = stations[(stations.family == family) & (stations.twin_cell >= 0)]
    variants = {"raw_cleaned": (None, "value", (0, 1)),
                "raw_vendor_uncleaned": (None, "value_raw", (0, 1, 2, 3, 4, 5, 6, 8, 9)),
                "cm_removed_ngl_offfan": ("ngl_offfan", "value", (0, 1))}
    if family == "wra":
        variants["cm_removed_wra_hf"] = ("wra_hf", "value", (0, 1))
        variants["cm_removed_wra_onfan_fullband_diag"] = ("wra_onfan_fullband", "value", (0, 1))
    mem_cells = []
    if members is not None:
        mem_cells = [int(c) for c in stations[(stations.family == family)
                                               & stations.station.isin(members)
                                               & (stations.twin_cell >= 0)].twin_cell]
    model_cm_onfan = pd.Series(subs[mem_cells].mean(axis=0) if mem_cells else 0.0,
                               index=dates)
    out = {}
    for label, (col, value, flags) in variants.items():
        Mv = _monthly_variant(daily, cms, col, value=value, flags=flags)
        mcm = model_cm_onfan if col == "wra_onfan_fullband" else 0.0
        rows = []
        for _, s in st.iterrows():
            k = s.station
            if k not in Mv:
                continue
            obs = -Mv[k]
            ro, n = ols_rate(obs, t0, t1)
            mod = pd.Series(subs[int(s.twin_cell)], index=dates) - mcm
            rm, _ = ols_rate(mod, t0, t1, min_n=1)
            rb, _ = ols_rate(obs, "2020-01-01", "2022-12-01", min_n=24)
            rows.append({"station": k, "n_months": n, "obs_rate_cm_yr": 100 * ro,
                         "model_rate_cm_yr": 100 * rm, "base_continue_cm_yr": 100 * rb})
        T = pd.DataFrame(rows)
        if len(T):
            T = T[T.n_months >= 12]
        o = T.obs_rate_cm_yr.to_numpy(float) if len(T) else np.array([])
        out[label] = {"n_stations": int(len(T)),
                      "model_archive": r2_rmse(T.model_rate_cm_yr.to_numpy(float), o)
                      if len(T) else {},
                      "base_continue_2020_22": r2_rmse(T.base_continue_cm_yr.to_numpy(float), o)
                      if len(T) else {},
                      "base_zero": r2_rmse(np.zeros(len(T)), o) if len(T) else {},
                      "table": T.round(3).to_dict(orient="records")}
    out["window"] = [t0, t1]
    out["model"] = ("results/twin_forward/datum_gate_innov.npz subs_mean_by_rheology[0, 0] "
                    "(coupled_leveling, baseline scenario), monthly, cell of each station")
    return out


# ----------------------------------------------------------------------------- A1
def injection_acceptance(p: M.Params, ref: list[str], window_end, seed: int,
                         amps=(0.02, 0.03, 0.05, 0.10), n_rep: int = 10,
                         threshold: float = 0.90) -> tuple[dict, pd.DataFrame]:
    """Injected-step recovery of the product pipeline on series cut at ``window_end``
    (None = full record): WRA and NGL families separately."""
    out = {}
    wser = P.wra_series(window_end)
    bw = P.run_wra(p, window_end=window_end, series=wser)
    tw = INJ.run_trials("wra", p, bw, window_end, list(amps), n_rep, seed, series=wser)
    nser, _ = P.ngl_series(window_end)
    bn = P.run_ngl(p, ref=ref, window_end=window_end, series=nser)
    tn = INJ.run_trials("ngl", p, bn, window_end, list(amps), n_rep, seed, ref=ref,
                        series=nser)
    for name, t in (("wra", tw), ("ngl", tn)):
        sm = INJ.summarise(t)
        sm["threshold"] = threshold
        sm["pass"] = bool(sm["recovered"] >= threshold)
        out[name] = sm
    out["pass"] = bool(out["wra"]["pass"] and out["ngl"]["pass"])
    return out, pd.concat([tw, tn])
