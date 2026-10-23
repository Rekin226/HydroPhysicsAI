"""2026 extension of the full variant (plan W1 step 6): basin rain index, rainfall-runoff
regression for the inflow composite, leave-one-year-out chain test, and the 2026 series.

Everything here reads published W1 outputs (surface_water_v4/full/v1) and the L0 CODIS
records; the settings are in ext_config.py (fixed before any score, pre-registered by
ext_prereg.py).

* Rain index. Daily station records come from hydrophysics.twin.cwa_rain.build (numeric
  unflagged days only, trace = 0, newest ID of a chain wins, nothing interpolated). A
  station-month counts when >= 80 % of its days are numeric; its total is scaled by days /
  numeric days. Each record is normalised by its normal, the mean over the 12 calendar
  months of its mean valid total of that calendar month in 2012-2022 (normal ratio method,
  so a missing station does not shift the level); a sub-basin's index
  is the mean ratio of its valid records times the mean of their normals; the basin index is
  the drainage-area-weighted mean of the three sub-basins (a sub-basin with no valid record
  takes the basin-wide mean ratio).
* Regression. Q_m (10^8 m3, monthly volume of the full variant's composite I) on P_m,
  P_{m-1}, P_{m-2} with calendar-month intercepts, linear or log-log (log1p rain, Duan
  smearing on back-transform); the spec with the lower leave-one-year-out monthly RMSE on the
  training years is used (nested selection inside the outer leave-one-year-out).
* Chain. Daily I of a predicted year = monthly volume / days; D(day) as in diversion.simulate
  with (beta, a_chc, a_yl) refitted without the held-out year, the full variant's shares and
  winter weight, and R = previous year's industrial + public supply. Only evaluation years
  (every month from November of the previous year to December has an index) are scored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from w1_canal import diversion
from w1_canal import ext_config as E
from w1_canal.config import DISTRICTS, product_dir_base
from w1_canal.sources import jiji_annual

SEC_DAY = 86400.0
FULL_V1 = product_dir_base("full") / "v1"


# ----------------------------------------------------------------------------- rain index
def station_daily(raw_dir=None, end: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame,
                                                                 dict]:
    from hydrophysics.twin import cwa_rain
    chains = {rec: ch for rec, (_sb, _nm, ch) in E.STATIONS.items()}
    daily, src, qa = cwa_rain.build(str(raw_dir or E.RAW_DIR), start=E.FETCH_START,
                                    end=end or E.FETCH_END, chains=chains,
                                    order=list(E.STATIONS))
    return daily, src, qa


def station_monthly(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly totals (mm) with >= MIN_DAY_FRACTION numeric days (scaled), and the counts."""
    g = daily.resample("MS")
    tot, cnt = g.sum(min_count=1), g.count()
    days = pd.Series(tot.index.days_in_month, index=tot.index)
    ok = cnt.ge(E.MIN_DAY_FRACTION * days, axis=0)
    scaled = tot.mul(days, axis=0) / cnt.where(cnt > 0)
    return scaled.where(ok), cnt


def basin_index(pm: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, dict]:
    lo, hi = E.NORMAL_WINDOW
    win = pm.loc[lo:hi]
    normal = win.groupby(win.index.month).mean().mean()
    ratio = pm / normal
    area = {sb: a for sb, (_g, a) in E.SUBBASINS.items()}
    members = {sb: [r for r, (s, _n, _c) in E.STATIONS.items() if s == sb] for sb in area}
    all_ratio = ratio.mean(axis=1)
    sub = {}
    for sb, recs in members.items():
        r = ratio[recs].mean(axis=1).fillna(all_ratio)
        sub[sb] = r * normal[recs].mean()
    sub = pd.DataFrame(sub)
    w = pd.Series(area) / sum(area.values())
    P = (sub * w).sum(axis=1).where(sub.notna().all(axis=1))
    n_valid = pm.notna().sum(axis=1)
    P = P.where(n_valid >= E.MIN_STATIONS_MONTH)
    info = {"normal_mm_per_month": normal.round(2).to_dict(),
            "members": members, "area_weights": w.round(4).to_dict(),
            "n_valid_records": n_valid.astype(int)}
    return P, sub, info


# ----------------------------------------------------------------------------- regression
def design(P: pd.Series, months: pd.DatetimeIndex, spec: str) -> np.ndarray:
    cols = [np.asarray(months.month == k, dtype=float) for k in range(1, 13)]
    for lag in range(E.LAGS + 1):
        x = P.shift(lag).reindex(months).to_numpy(dtype=float)
        cols.append(np.log1p(x) if spec == "log_lag2" else x)
    return np.column_stack(cols)


@dataclass
class Fit:
    spec: str
    coef: np.ndarray
    smear: float


def fit_spec(P: pd.Series, Q: pd.Series, months: pd.DatetimeIndex, spec: str) -> Fit:
    X = design(P, months, spec)
    y = Q.reindex(months).to_numpy(dtype=float)
    if spec == "log_lag2":
        y = np.log(y)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    coef, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    smear = float(np.mean(np.exp(y[ok] - X[ok] @ coef))) if spec == "log_lag2" else 1.0
    return Fit(spec, coef, smear)


def predict(f: Fit, P: pd.Series, months: pd.DatetimeIndex) -> np.ndarray:
    z = design(P, months, f.spec) @ f.coef
    q = np.exp(z) * f.smear if f.spec == "log_lag2" else z
    return np.clip(q, 0.0, None)


def months_of(years) -> pd.DatetimeIndex:
    years = list(years)
    return pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-01", freq="MS")


def loo_rmse(P, Q, years, spec) -> float:
    err = []
    for y in years:
        tr = [t for t in years if t != y]
        f = fit_spec(P, Q, pd.DatetimeIndex(np.concatenate([months_of([t]) for t in tr])),
                     spec)
        m = months_of([y])
        err.append(predict(f, P, m) - Q.reindex(m).to_numpy())
    e = np.concatenate(err)
    return float(np.sqrt(np.nanmean(e ** 2)))


def select_and_fit(P, Q, years) -> tuple[Fit, dict]:
    scores = {s: loo_rmse(P, Q, years, s) for s in E.SPECS}
    best = min(scores, key=scores.get)
    m = pd.DatetimeIndex(np.concatenate([months_of([t]) for t in years]))
    return fit_spec(P, Q, m, best), scores


# ----------------------------------------------------------------------------- chain
def load_full_v1() -> dict:
    comp = pd.read_parquet(FULL_V1 / "inflow_composite_daily.parquet")
    inflow = pd.Series(comp["value"].to_numpy(float), index=pd.to_datetime(comp["date"]))
    val = json.loads((FULL_V1 / "validation.json").read_text())
    ff = val["variant_validation"]["final_fit"]
    mon = pd.read_csv(FULL_V1 / "district_monthly.csv", parse_dates=["date"])
    den = {d: mon[mon.district == nm].set_index("date")["value"]
           for d, nm in (("chc", "changhua"), ("yl", "yunlin"))}
    sd = {d: mon[mon.district == nm].set_index("date")["sd"]
          for d, nm in (("chc", "changhua"), ("yl", "yunlin"))}
    draws = np.load(FULL_V1 / "district_draws.npz", allow_pickle=True)
    jiji, _ = jiji_annual()
    Q = (inflow * SEC_DAY / 1e8).resample("MS").sum()
    return {"I": inflow, "Q": Q, "final_fit": ff, "den": den, "sd": sd, "draws": draws,
            "jiji": jiji}


def r_cms(jiji: pd.DataFrame, year: int, ndays: int) -> float:
    return float((jiji.loc[year, "industrial_1e8_m3"] + jiji.loc[year, "public_1e8_m3"])
                 * 1e8 / (ndays * SEC_DAY))


def daily_from_monthly(qm: np.ndarray, months: pd.DatetimeIndex, R: float
                       ) -> diversion.Daily:
    """Daily drivers of one (possibly partial) year from monthly volumes (10^8 m3)."""
    y = months[0].year
    idx = pd.date_range(months[0], months[-1] + pd.offsets.MonthEnd(0), freq="D")
    per_day = {m.month: q * 1e8 / (m.days_in_month * SEC_DAY) for m, q in zip(months, qm,
                                                                               strict=True)}
    inflow = np.array([per_day[d.month] for d in idx])
    mon = idx.month.to_numpy()
    return diversion.Daily(idx, inflow, np.full(len(idx), R), [y], mon, mon - 1, 12)


def chain_year(base: dict, P: pd.Series, year: int, train_years, use_actual_inflow=False,
               R_year=None, dd_all=None) -> dict:
    """Held-out (or extension) year's monthly district deliveries from the chain."""
    ff = base["final_fit"]
    s_c, s_y, w = ff["s_chc"], ff["s_yl"], ff["winter"]
    jiji = base["jiji"]
    years_all = list(E.FIT_YEARS)
    if dd_all is None:
        dd_all = diversion.make_daily(base["I"], jiji, years_all)
    targets = diversion.annual_targets(jiji, years_all)
    p, rms = diversion.fit(dd_all, targets, s_c, s_y, w, True, fit_years=list(train_years))
    if year in years_all:
        months = months_of([year])
    else:
        months = pd.date_range(f"{year}-01-01", P.dropna().index.max(), freq="MS")
    sel = {}
    if use_actual_inflow:
        qm = base["Q"].reindex(months).to_numpy()
        sel = {"spec": "actual_inflow"}
    else:
        f, scores = select_and_fit(P, base["Q"], list(train_years))
        qm = predict(f, P, months)
        sel = {"spec": f.spec, "inner_loo_rmse": scores}
    ry = (year - 1) if R_year is None else R_year
    ndays = 366 if pd.Timestamp(f"{ry}-12-31").dayofyear == 366 else 365
    dd = daily_from_monthly(qm, months, r_cms(jiji, ry, ndays))
    mc, my = diversion.simulate(dd, *p, s_c, s_y, w)
    k = len(months)
    return {"months": months, "chc": mc[:k], "yl": my[:k], "q_pred": qm, "beta_a": p,
            "fit_rms": rms, **sel}


def paired_bootstrap(e_a: np.ndarray, e_b: np.ndarray, B: int, seed: int, level: float
                     ) -> dict:
    """RMSE(a) - RMSE(b) with a percentile paired bootstrap over units, and a sign test."""
    from scipy import stats
    rng = np.random.default_rng(seed)
    n = len(e_a)
    d0 = float(np.sqrt(np.mean(e_a ** 2)) - np.sqrt(np.mean(e_b ** 2)))
    idx = rng.integers(0, n, size=(B, n))
    d = np.sqrt((e_a[idx] ** 2).mean(1)) - np.sqrt((e_b[idx] ** 2).mean(1))
    lo, hi = np.percentile(d, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    wins = int((np.abs(e_a) < np.abs(e_b)).sum())
    ties = int((np.abs(e_a) == np.abs(e_b)).sum())
    p = float(stats.binomtest(wins, n - ties).pvalue) if n - ties > 0 else 1.0
    return {"rmse_diff": d0, "boot_lo": float(lo), "boot_hi": float(hi), "sign_wins": wins,
            "sign_n": n - ties, "sign_p_two_sided": p}


def evaluation_years(P: pd.Series) -> list[int]:
    """Fit years whose months from November of the previous year to December all have an
    index (the rain lags of January reach back two months)."""
    out = []
    for y in E.FIT_YEARS:
        win = P.reindex(pd.date_range(f"{y - 1}-11-01", f"{y}-12-01", freq="MS"))
        if win.notna().all():
            out.append(y)
    return out


def chain_test(base: dict, P: pd.Series) -> dict:
    """Leave-one-year-out over FIT_YEARS, scored on the evaluation years: annual and monthly
    errors of the rain-driven chain, the actual-inflow reference, annual climatology (mean of
    the other 13 fit years) and persistence."""
    years = list(E.FIT_YEARS)
    jiji = base["jiji"]
    dd_all = diversion.make_daily(base["I"], jiji, years)
    rows, monthly = [], []
    for y in evaluation_years(P):
        tr = [t for t in years if t != y]
        rc = chain_year(base, P, y, tr, dd_all=dd_all)
        ra = chain_year(base, P, y, tr, use_actual_inflow=True, R_year=y, dd_all=dd_all)
        for d in DISTRICTS:
            col = "changhua_irrigation_1e8_m3" if d == "chc" else "yunlin_irrigation_1e8_m3"
            obs = float(jiji.loc[y, col])
            clim = float(jiji.loc[tr, col].mean())
            pers = float(jiji.loc[y - 1, col])
            den = base["den"][d].reindex(rc["months"]).to_numpy()
            rows.append({"year": y, "district": d, "obs": obs,
                         "rain_chain": float(rc[d].sum()), "actual_inflow": float(ra[d].sum()),
                         "climatology": clim, "persistence": pers,
                         "rain_chain_janSep": float(rc[d][:9].sum()),
                         "denton_janSep": float(den[:9].sum()),
                         "spec": rc["spec"], "beta": float(rc["beta_a"][0])})
            for k, m in enumerate(rc["months"]):
                monthly.append({"year": y, "month": m.month, "district": d,
                                "rain_chain": float(rc[d][k]), "denton": float(den[k]),
                                "q_pred": float(rc["q_pred"][k]),
                                "q_obs": float(base["Q"].loc[m])})
    ann = pd.DataFrame(rows)
    mon = pd.DataFrame(monthly)
    out = {"annual": ann, "monthly": mon, "scores": {}, "compare": {},
           "evaluation_years": evaluation_years(P)}
    for d in DISTRICTS:
        a = ann[ann.district == d]
        e = {m: (a[m] - a.obs).to_numpy() for m in ("rain_chain", "actual_inflow",
                                                    "climatology", "persistence")}
        out["scores"][d] = {m: {"rmse": float(np.sqrt(np.mean(v ** 2))),
                                "bias": float(np.mean(v))} for m, v in e.items()}
        out["compare"][d] = {
            "rain_chain_vs_climatology": paired_bootstrap(
                e["rain_chain"], e["climatology"], E.BOOT_B, E.BOOT_SEED, E.BOOT_LEVEL),
            "rain_chain_vs_persistence": paired_bootstrap(
                e["rain_chain"], e["persistence"], E.BOOT_B, E.BOOT_SEED + 1, E.BOOT_LEVEL)}
        js = (a.rain_chain_janSep - a.denton_janSep).to_numpy()
        out["scores"][d]["rain_chain_janSep_vs_denton_rmse"] = float(np.sqrt(np.mean(js ** 2)))
        mm = mon[mon.district == d]
        out["scores"][d]["monthly_rmse_vs_denton"] = float(
            np.sqrt(np.mean((mm.rain_chain - mm.denton) ** 2)))
    qe = mon[mon.district == "chc"]
    out["inflow_monthly_rmse"] = float(np.sqrt(np.mean((qe.q_pred - qe.q_obs) ** 2)))
    clim_q = qe.groupby("month").q_obs.transform("mean")
    out["inflow_monthly_rmse_climatology_insample"] = float(
        np.sqrt(np.mean((clim_q - qe.q_obs) ** 2)))
    out["spec_counts"] = ann[ann.district == "chc"].spec.value_counts().to_dict()
    return out


def shipping_rule(test: dict) -> dict:
    """Pre-registered: the rain-driven chain ships for 2026 if its LOO annual RMSE is below
    annual climatology's in both districts (point); the label follows the 90 % paired
    bootstrap interval; otherwise the 2026 months ship as the monthly climatology of the
    full variant's Denton series 2012-2025."""
    pt = {d: test["scores"][d]["rain_chain"]["rmse"] < test["scores"][d]["climatology"]["rmse"]
          for d in DISTRICTS}
    iv = {d: test["compare"][d]["rain_chain_vs_climatology"]["boot_hi"] < 0 for d in DISTRICTS}
    ships_chain = all(pt.values())
    if ships_chain and all(iv.values()):
        label = "rain-driven extension beats annual climatology (90 % interval excludes zero)"
    elif ships_chain:
        label = "rain-driven extension not distinguishable from climatology on the annual " \
                "leave-one-year-out test"
    else:
        label = "climatology (rain-driven extension did not beat annual climatology)"
    return {"beats_point": pt, "interval_below_zero": iv, "ships": "rain_chain"
            if ships_chain else "climatology", "label": label}
