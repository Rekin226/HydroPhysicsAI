"""Daily Jiji-weir inflow composite with per-year gauge QC (plan W1 step 2).

I(day) = 寶石橋 H075 + 內茅埔 H049 + 水里 H064, then rescaled per year to the weir's published
annual inflow. Steps, in order, for the years of one variant:

1. Explicit exclusions (config.QC_EXCLUDE): 內茅埔 in August 2024 (玉峰橋 2024-25 is not part
   of the composite) -> flag 9 on the gauge, value removed.
2. Gaps of at most 15 days (whole run length, not the first 15 days of a longer gap) are
   filled by linear interpolation in time (flag 4); edge gaps by the nearest value.
3. Remaining gaps are filled with a scaled donor (flag 3): the other composite gauges' sum
   times the ratio of median annual ratios (fitted on the variant's fit years only).
4. Any gauge-year whose annual volume ratio to the weir inflow is more than 50 % off its
   median (median over the fit years) is replaced in full by the scaled donor (flag 3).
5. The composite's annual volume is rescaled to the weir's published inflow.

The medians in steps 3-4 are fitted parameters: the pre-origin variant computes them on
2012-2022 only (U3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from w1_canal.config import DONOR_RATIO_TOL, GAP_FILL_DAYS, INFLOW_GAUGES, QC_EXCLUDE

SEC_DAY = 86400.0


def _runs(mask: np.ndarray):
    """(start, stop) index pairs of consecutive True runs."""
    m = np.r_[False, mask, False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True))


def fill_short_gaps(s: pd.Series, max_len: int) -> tuple[pd.Series, np.ndarray]:
    """Fill NaN runs of length <= max_len (linear in time inside, nearest at the edges)."""
    v = s.to_numpy(dtype=float).copy()
    filled = np.zeros(len(v), dtype=bool)
    interp = s.interpolate(method="time", limit_area="inside").to_numpy()
    for a, b in _runs(np.isnan(v)):
        if b - a > max_len:
            continue
        if a == 0 and b < len(v):
            v[a:b] = v[b]
        elif b == len(v) and a > 0:
            v[a:b] = v[a - 1]
        elif a > 0 and b < len(v):
            v[a:b] = interp[a:b]
        else:
            continue
        filled[a:b] = True
    return pd.Series(v, index=s.index), filled


def build_composite(daily: pd.DataFrame, jiji: pd.DataFrame, years, fit_years) -> dict:
    """Composite for ``years`` with QC parameters fitted on ``fit_years``.

    ``daily``: long table (year, station_no, month, day, q_cms) from parse_yearbook.
    Returns dict with the daily frame (I_cms, composite_raw_cms, per-gauge q and flags),
    the annual scale factors, the QC log and the fitted medians."""
    years = list(years)
    fit_years = [y for y in fit_years if y in years]
    d = daily[daily.station_no.isin(INFLOW_GAUGES) & daily.year.isin(years)].copy()
    d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day))
    q = d.pivot_table(index="date", columns="station_no", values="q_cms", aggfunc="first")
    idx = pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31", freq="D")
    q = q.reindex(idx)[list(INFLOW_GAUGES)]
    flags = pd.DataFrame(np.where(q.isna(), 8, 0), index=idx, columns=q.columns, dtype="uint8")
    log = []
    # 1. explicit exclusions
    for s, items in QC_EXCLUDE.items():
        if s not in q.columns:
            continue
        for y, months in items:
            sel = (idx.year == y) & (np.isin(idx.month, months) if months else True)
            if sel.any():
                q.loc[sel, s] = np.nan
                flags.loc[sel, s] = 9
                log.append(f"{s} {y} months {list(months) if months else 'all'}: excluded "
                           f"(plan W1 step 1), {int(sel.sum())} days")
    # 2. short gaps
    for s in q.columns:
        filled_s, f = fill_short_gaps(q[s], GAP_FILL_DAYS)
        n = int(f.sum())
        if n:
            q[s] = filled_s
            flags.loc[f, s] = 4
            log.append(f"{s}: {n} days in gaps <= {GAP_FILL_DAYS} d filled by interpolation")
    inflow = jiji["inflow_1e8_m3"].reindex(years).astype(float)
    # 3-4. annual ratio to the weir inflow; medians from the fit years
    ann = (q * SEC_DAY / 1e8).groupby(idx.year).sum(min_count=1)
    ratio = ann.div(inflow, axis=0)
    # ratios of a gauge-year with long gaps are not representative: use complete years only
    complete = q.notna().groupby(idx.year).mean() >= 0.98
    med = pd.Series({s: float(ratio.loc[[y for y in fit_years if complete.loc[y, s]], s]
                              .median()) for s in q.columns})
    off = (ratio / med - 1).abs() > DONOR_RATIO_TOL
    for y in years:
        bad = [s for s in q.columns if off.loc[y, s] or not complete.loc[y, s]]
        for s in bad:
            donors = [g for g in q.columns if g not in bad]
            if not donors:
                raise ValueError(f"{y}: no clean donor gauge for {s}")
            k = med[s] / med[donors].sum()
            sel = idx.year == y
            donor = q.loc[sel, donors].sum(axis=1, min_count=len(donors)) * k
            if off.loc[y, s]:
                q.loc[sel, s] = donor
                flags.loc[sel, s] = np.where(flags.loc[sel, s] == 9, 9, 3)
                log.append(f"{y} {s}: annual ratio {ratio.loc[y, s]:.3f} vs median "
                           f"{med[s]:.3f} (> {DONOR_RATIO_TOL:.0%} off) -> whole year replaced "
                           f"by {'+'.join(donors)} x {k:.3f}")
            else:
                gap = q.loc[sel, s].isna()
                q.loc[sel & q[s].isna().to_numpy(), s] = donor[gap.to_numpy()]
                flags.loc[sel & (flags[s].to_numpy() == 8), s] = 3
                log.append(f"{y} {s}: {int(gap.sum())} days in gaps > {GAP_FILL_DAYS} d filled "
                           f"by {'+'.join(donors)} x {k:.3f}")
    if q.isna().any().any():
        raise ValueError(f"composite still has gaps: {q.isna().sum().to_dict()}")
    comp = q.sum(axis=1)
    comp_ann = comp.groupby(idx.year).sum() * SEC_DAY / 1e8
    scale = inflow / comp_ann
    inflow_d = comp * pd.Series(idx.year, index=idx).map(scale)
    # ratios after QC (reported)
    ann2 = (q * SEC_DAY / 1e8).groupby(idx.year).sum()
    return {
        "q": q, "flags": flags, "I_cms": inflow_d, "composite_raw_cms": comp,
        "scale": scale, "median_ratio": med, "ratio_before": ratio, "ratio_after":
        ann2.div(inflow, axis=0), "log": log, "fit_years": fit_years,
    }


def composite_table(c: dict, product_version: str) -> pd.DataFrame:
    """Daily long table of the composite (value in cms) with sd, flag and method.

    The value is the gauge sum rescaled to the weir's published annual inflow. sd: the
    composite's day-level error is not observable; it is set to the robust scatter of the
    log annual scale factors over the fit years (relative), doubled on days where a gauge
    was donor-filled. Flag: 1 (observed, corrected by the weir's own annual inflow) when
    all three gauges are observed, 4 when the worst gauge was interpolated, 3 when donor-
    filled."""
    s = np.log(c["scale"].loc[c["fit_years"]])
    rel = float(1.4826 * np.median(np.abs(s - np.median(s))))
    worst = c["flags"].replace({9: 3}).max(axis=1)        # excluded days are donor days
    flag = np.select([worst == 0, worst == 4, worst == 3], [1, 4, 3], 3).astype("uint8")
    method = np.select([flag == 1, flag == 4], ["gauge_sum_rescaled_to_weir_inflow",
                                                "gauge_sum_with_interpolated_gauge"],
                       "gauge_sum_with_scaled_donor_gauge")
    inflow_d = c["I_cms"]
    sd = inflow_d.to_numpy() * rel * np.where(flag == 3, 2.0, 1.0)
    out = pd.DataFrame({"date": inflow_d.index.strftime("%Y-%m-%d"),
                        "value": inflow_d.to_numpy(),
                        "sd": sd, "flag": flag, "method": method,
                        "product_version": product_version,
                        "source_ids": "wra_hydro_yearbook:H075+H049+H064;jiji_weir_annual:inflow",
                        "composite_raw_cms": c["composite_raw_cms"].to_numpy()})
    for g in c["q"].columns:
        out[f"q_{g}_cms"] = c["q"][g].to_numpy()
        out[f"flag_{g}"] = c["flags"][g].to_numpy()
    out.attrs["rel_sd"] = rel
    return out
