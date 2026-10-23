"""Task 2: TPC county 農林漁牧 monthly sales against the census county totals.

Usage: python -I validate.py OUT_JSON

Windows: the overlap starts 2016-01 (d007012 has no month before it; 2012-2015 has no county
sector data) and ends 2026-06 (last fully recorded census month). Pre-origin overlap
2016-01..2022-12 and post-origin overlap 2023-01..2026-06 are reported separately.

Per county and window, for the census whole-county total (all classes), its agricultural
classes (irrigation + aquaculture + livestock), and the in-grid series the hybrid layer uses:

* ratio: census / TPC over the window (sum ratio) and per calendar year;
* anomaly correlation: monthly values minus the window's month-of-year mean, divided by it;
  also on two-month bill pairs (Jan+Feb, Mar+Apr, ...), and lagged (TPC shifted -2..+2);
* seasonal match: correlation of the 12 month-of-year means (each normalised by its own
  mean), the amplitude ratio (sd of the normalised profiles) and the peak months;
* annual: correlation of calendar-year totals (complete years).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import panel  # noqa: E402

WINDOWS = {"pre_2016-01_2022-12": ("2016-01-01", "2022-12-01"),
           "post_2023-01_2026-06": ("2023-01-01", "2026-06-01"),
           "all_2016-01_2026-06": ("2016-01-01", "2026-06-01")}


def rel_anom(s: pd.Series) -> pd.Series:
    clim = s.groupby(s.index.month).transform("mean")
    return s / clim - 1.0


def corr(a: pd.Series, b: pd.Series) -> float | None:
    ok = a.notna() & b.notna()
    if ok.sum() < 6:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def pairs(s: pd.Series) -> pd.Series:
    key = s.index.year * 100 + (s.index.month + 1) // 2
    g = s.groupby(key)
    out = g.sum(min_count=2)
    out.index = pd.to_datetime([f"{k // 100}-{2 * (k % 100) - 1:02d}-01" for k in out.index])
    return out


def compare(cen: pd.Series, tpc: pd.Series) -> dict:
    ok = cen.notna() & tpc.notna()
    c, t = cen[ok], tpc[ok]
    ac, at = rel_anom(c), rel_anom(t)
    pc, pt = pairs(c), pairs(t)
    prof_c = c.groupby(c.index.month).mean()
    prof_t = t.groupby(t.index.month).mean()
    nc, nt = prof_c / prof_c.mean(), prof_t / prof_t.mean()
    yrs = c.groupby(c.index.year).count()
    full = yrs.index[yrs == 12]
    yc, yt = c.groupby(c.index.year).sum(), t.groupby(t.index.year).sum()
    lagged = {}
    for k in (-2, -1, 0, 1, 2):
        ts = rel_anom(tpc.shift(-k)[ok])            # k > 0: TPC of month t + k
        lagged[f"tpc_t{k:+d}"] = corr(ac, ts)
    return {
        "n_months": int(ok.sum()),
        "mean_census_GWh_per_month": round(float(c.mean()), 3),
        "mean_tpc_GWh_per_month": round(float(t.mean()), 3),
        "ratio_census_over_tpc": round(float(c.sum() / t.sum()), 4),
        "ratio_by_year": {int(y): round(float(yc[y] / yt[y]), 4) for y in yc.index},
        "anomaly_corr_monthly": corr(ac, at),
        "anomaly_corr_bill_pairs": corr(rel_anom(pc), rel_anom(pt)),
        "anomaly_corr_lagged": lagged,
        "anomaly_sd_census": round(float(ac.std()), 4),
        "anomaly_sd_tpc": round(float(at.std()), 4),
        "seasonal_profile_corr": float(np.corrcoef(nc, nt)[0, 1]),
        "seasonal_amplitude_ratio_census_over_tpc": round(float(nc.std() / nt.std()), 4),
        "peak_month_census": int(nc.idxmax()), "peak_month_tpc": int(nt.idxmax()),
        "low_month_census": int(nc.idxmin()), "low_month_tpc": int(nt.idxmin()),
        "seasonal_profile_census": [round(float(x), 3) for x in nc],
        "seasonal_profile_tpc": [round(float(x), 3) for x in nt],
        "annual_corr": corr(yc.loc[full], yt.loc[full]) if len(full) >= 3 else None,
        "annual_n_years": int(len(full)),
    }


def main(out: str) -> None:
    tpc = panel.tpc_wide()
    c0, reads = panel.census_c0_county()
    hyb = panel.census_hybrid()
    reads += hyb.reads
    res = {"reads": reads, "notes": hyb.notes + [
        "TPC d007012 starts 2016-01: the 2012-01..2015-12 part of the requested window has "
        "no county sector data and is not compared",
        "census monthly values of bimonthly meters are TPC's even split of a two-month bill "
        "(about 92-95 % of kept meters): the bill-pair statistic removes that split"],
        "counties": {}}
    for key in ("changhua", "yunlin"):
        r = {}
        for wname, (a, b) in WINDOWS.items():
            t = tpc[f"ag_{key}"].loc[a:b]
            rec_mask = hyb.recorded.loc[a:b]
            series = {
                "census_county_all": c0[f"all_{key}"].loc[a:b],
                "census_county_agri_classes": c0[f"agri_{key}"].loc[a:b],
                "census_county_irrigation": c0[f"irr_{key}"].loc[a:b],
                "census_ingrid_hybrid": hyb.county[key].loc[a:b].where(rec_mask),
            }
            r[wname] = {k: compare(v, t) for k, v in series.items()}
        res["counties"][key] = r
        for wname in WINDOWS:
            x = r[wname]["census_ingrid_hybrid"]
            print(key, wname, "ratio", x["ratio_census_over_tpc"], "anom r",
                  round(x["anomaly_corr_monthly"], 3), "pairs r",
                  round(x["anomaly_corr_bill_pairs"], 3), "lags",
                  {k: round(v, 3) for k, v in x["anomaly_corr_lagged"].items()},
                  "season r", round(x["seasonal_profile_corr"], 3), "amp",
                  x["seasonal_amplitude_ratio_census_over_tpc"])
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
