"""WP S3 (v3 code, repair round 3 S3r, 2026-10-18): turn family runs into the product tables
(plan 2.4 long tables).

v3 change against v2: the own-series z that decides a step's basis (flag 1 "own_series" or
flag 3 "common_mode_only") uses the v3 step-size estimator (level difference of the series
minus the global trend and seasonal terms) and its empirical-null se on the station's own
series (``model.local_null_sigma``), the same calibration as detection (repair R1'/R2). Everything else is v2, including the full-variant
jump-cluster flag (method ``jump_cluster``), which stays full-variant only (its rule was
written after seeing 2023-25 data; U7).

Tables (all heights in metres, dates as ISO ``YYYY-MM-DD`` / ``YYYY-MM`` strings):

* ``daily``: one row per observed day and station: ``value_raw`` (L0 height), ``value``
  (height with the step catalogue removed; no common mode removed), ``correction_m``,
  ``sd`` (1-sigma: station high-frequency noise and the step-correction sd in quadrature),
  ``resid`` (residual of the station's linear + annual + semiannual model of ``value``),
  ``flag`` and ``method`` (plan 2.4: 0 observed; 1 corrected by a step backed by metadata,
  the plan catalogue, or a step evident in the station's own series; 3 corrected by a step
  that is significant only after the neighbour common mode is subtracted; 9 suspect:
  gross or screen outlier, flagged excursion, or after an offset across a gap too long to
  tie), ``product_version``, ``source_ids``.
* ``monthly``: station-month median of ``value`` over the days with flags 0, 1 and 3 (at
  least ``MIN_DAYS``), with ``n_days``, ``sd`` and ``flag`` = the largest flag included.
* ``steps``: the step catalogue with evidence, size, se, own-series z and the flag it gives.
* ``stations``: coordinates (TWD97 TM2), twin cell, role, notes.
* ``common_mode`` (daily) and ``common_mode_monthly``: separate series, never removed from
  ``value``.
"""
from __future__ import annotations

import math

import io_gnss as io
import model as M
import numpy as np
import pandas as pd

MIN_DAYS = 10
META_EVIDENCE = ("equipment", "earthquake", "catalogue")
#: jump-cluster rule (full variant only): a consecutive-day raw height change larger than
#: JUMP_M is a jump; a calendar month with >= JUMP_PER_MONTH jumps is a jump month; every
#: observed day of a run of >= JUMP_MONTHS consecutive jump months is flagged 9
JUMP_M = 0.03
JUMP_PER_MONTH = 3
JUMP_MONTHS = 3


def jump_cluster_mask(dates: pd.DatetimeIndex, y: np.ndarray) -> np.ndarray:
    """True on the days of a jump cluster (see the constants above)."""
    fin = np.isfinite(y)
    n = len(y)
    out = np.zeros(n, bool)
    if fin.sum() < 2:
        return out
    idx = np.flatnonzero(fin)
    d = pd.DatetimeIndex(dates)
    dd = np.diff(d[idx].to_numpy()).astype("timedelta64[D]").astype(int)
    jump = (np.abs(np.diff(y[idx])) > JUMP_M) & (dd == 1)
    per = pd.Series(jump.astype(int), index=d[idx[1:]].to_period("M")).groupby(level=0).sum()
    months = pd.period_range(d[idx[0]].to_period("M"), d[idx[-1]].to_period("M"), freq="M")
    jm = per.reindex(months, fill_value=0) >= JUMP_PER_MONTH
    flagged = set()
    run: list = []
    for m, v in jm.items():
        if v:
            run.append(m)
            continue
        if len(run) >= JUMP_MONTHS:
            flagged.update(run)
        run = []
    if len(run) >= JUMP_MONTHS:
        flagged.update(run)
    if flagged:
        out = fin & np.asarray(d.to_period("M").isin(sorted(flagged)))
    return out


def step_rows(family: str, r: M.StationResult, p: M.Params) -> list[dict]:
    """Steps of one station with the own-series z (no common mode) and the flag they give
    to the samples after them."""
    yv = np.where(r.ok, r.y, np.nan)
    y_adj = M.detrended(yv, r.fit, 0.0, r.dates)
    idxs = [int(r.dates.searchsorted(s.date)) for s in r.steps]
    # own-series empirical-null scale: the station's step-free residual, no common mode
    su_self = M.local_null_sigma(r.fit.resid, p.local_half, p.min_obs, p.size_order, idxs)
    rows = []
    for s, j in zip(r.steps, idxs, strict=True):
        c, se, n = M.local_step(y_adj, r.dates, j, p.local_half, idxs, float(su_self[j]),
                                p.size_order)
        z_self = abs(c) / se if np.isfinite(c) and np.isfinite(se) and se > 0 else float("nan")
        meta = any(e in s.evidence for e in META_EVIDENCE)
        self_ev = bool(np.isfinite(z_self) and z_self >= p.z_min and abs(c) >= p.delta_min)
        if meta:
            basis, flag = "metadata_or_catalogue", 1
        elif self_ev:
            basis, flag = "own_series", 1
        else:
            basis, flag = "common_mode_only", 3
        row = s.row()
        row.update({"family": family, "station": r.station, "size_own_series_m": c,
                    "se_own_series_m": se, "z_own_series": z_self, "basis": basis,
                    "flag_after": flag})
        rows.append(row)
    return rows


def daily_table(family: str, r: M.StationResult, steps: list[dict], version: str,
                source_id: str, jump_flag: bool = False) -> pd.DataFrame:
    fin = np.isfinite(r.y)
    n = len(r.y)
    flag = np.zeros(n, dtype=np.uint8)
    method = np.array(["observed"] * n, dtype=object)
    # step corrections: flag of the latest-basis step applied (3 dominates 1)
    has1 = np.zeros(n, bool)
    has3 = np.zeros(n, bool)
    for s in steps:
        if not np.isfinite(s["size_m"]) or s["size_m"] == 0.0:
            continue
        j = int(r.dates.searchsorted(pd.Timestamp(s["date"])))
        if s["flag_after"] == 3:
            has3[j:] = True
        else:
            has1[j:] = True
    flag[has1] = 1
    method[has1] = "step_corrected"
    flag[has3] = 3
    method[has3] = "step_corrected_cm_only_step"
    screen = fin & ~r.ok & ~r.gross & ~r.transient
    jc = jump_cluster_mask(r.dates, r.y) if jump_flag else np.zeros(n, bool)
    for mask, lab in ((jc, "jump_cluster"), (r.untied, "after_untied_offset"),
                      (screen, "screen_outlier"), (r.transient & fin, "excursion"),
                      (r.gross, "gross_outlier")):
        flag[mask] = 9
        method[mask] = lab
    sd = np.sqrt(r.sigma_hf ** 2 + r.corr_se ** 2)
    df = pd.DataFrame({
        "family": family, "station": r.station,
        "date": r.dates.strftime("%Y-%m-%d"),
        "value_raw": r.y, "value": r.y - r.correction, "correction_m": r.correction,
        "sd": sd, "resid": r.fit.resid, "flag": flag, "method": method,
        "product_version": version, "source_ids": source_id})
    return df[fin].reset_index(drop=True)


def monthly_table(daily: pd.DataFrame, rho: dict[str, float], sig: dict[str, float]
                  ) -> pd.DataFrame:
    d = daily[daily.flag.isin([0, 1, 3])].copy()
    d["month"] = d.date.str.slice(0, 7)
    g = d.groupby(["family", "station", "month"])
    out = g.agg(value=("value", "median"), n_days=("value", "size"), flag=("flag", "max"),
                corr_sd=("sd", "mean"), product_version=("product_version", "first"),
                source_ids=("source_ids", "first")).reset_index()
    out = out[out.n_days >= MIN_DAYS].copy()
    # sd of a monthly median: hf noise inflated for lag-1 correlation and the median's
    # efficiency (pi/2), plus the step-correction sd (not reduced by averaging)
    s_hf = out.station.map(sig).astype(float)
    r1 = out.station.map(rho).astype(float).clip(0, 0.95)
    infl = np.sqrt((1 + r1) / (1 - r1))
    corr_se = np.sqrt(np.clip(out.corr_sd ** 2 - s_hf ** 2, 0.0, None))
    out["sd"] = np.sqrt((s_hf * infl) ** 2 * (math.pi / 2) / out.n_days + corr_se ** 2)
    out["flag"] = out.flag.astype(np.uint8)
    out["method"] = np.where(out.flag == 0, "monthly_median_observed",
                             np.where(out.flag == 1, "monthly_median_step_corrected",
                                      "monthly_median_incl_cm_only_step"))
    return out[["family", "station", "month", "value", "sd", "n_days", "flag", "method",
                "product_version", "source_ids"]].rename(columns={"month": "date"})


def station_row(family: str, name: str, r: M.StationResult | None, xy: tuple[float, float],
                cell: int, role: str, note: str = "") -> dict:
    row = {"family": family, "station": name, "x_twd97": xy[0], "y_twd97": xy[1],
           "twin_cell": int(cell), "role": role, "note": note}
    if r is not None:
        obs = r.dates[np.isfinite(r.y)]
        row.update({"first_day": str(obs[0].date()), "last_day": str(obs[-1].date()),
                    "n_days": int(np.isfinite(r.y).sum()),
                    "n_flag0_or_1_or_3": int(r.ok.sum() - (r.untied & r.ok).sum()),
                    "n_steps": len(r.steps), "rate_m_yr": r.fit.rate,
                    "annual_amp_m": float(np.hypot(*r.fit.beta[2:4])),
                    "sigma_hf_m": r.sigma_hf, "rho_hf": r.rho,
                    "notes": " | ".join(r.notes)})
    return row


def cm_monthly(cms: pd.DataFrame) -> pd.DataFrame:
    c = cms.copy()
    c.index = pd.DatetimeIndex(pd.to_datetime(c.index))
    cols = [k for k in c.columns if not k.endswith("_n") and not k.endswith("_sd")]
    g = c[cols].groupby(c.index.to_period("M"))
    m = g.mean()
    n = g.count()
    out = m.copy()
    for k in cols:
        out[f"{k}_n_days"] = n[k]
    out.index = out.index.strftime("%Y-%m")
    out.index.name = "date"
    return out.reset_index()


def wra_xy() -> dict[str, tuple[float, float]]:
    m = io.wra_meta()
    return {k: (float(r.X_3826), float(r.Y_3826)) for k, r in m.iterrows()}


def ncku_xy() -> dict[str, tuple[float, float]]:
    m = io.ncku_meta()
    return {f"NCKU_{k}": (float(r.e_twd97), float(r.n_twd97)) for k, r in m.iterrows()}


def ngl_xy(stations: list[str]) -> dict[str, tuple[float, float]]:
    out = {}
    for k in stations:
        df = io.read_tenv3(io.ngl_tenv3_path(k))
        x, y = io.lonlat_to_twd97(np.array([df.lon.median()]), np.array([df.lat.median()]))
        out[k] = (float(x[0]), float(y[0]))
    return out
