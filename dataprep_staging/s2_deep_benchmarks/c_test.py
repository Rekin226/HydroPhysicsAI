"""WP S2 test C: the deep-benchmark feeds against the printed table of the 007001 reports.

The 007001 reports (彰化與雲林地區地層下陷監測及分析 資料分析月報) state their method: "GNSS、
地下水位、深層水準樁取每月平均值加以計算" (GNSS, groundwater level and deep benchmarks are
computed from monthly mean values). Test C therefore compares the printed 單月變化量 with the
difference of consecutive monthly means of the feed (``monthly_mean_increments``); the
month-end convention of the product (test A) is reported beside it.

Pure functions; the rules are fixed in the pre-registration S2r2.c_fresh_cy007001.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DAYS = 20          # full days (flag 0-1) a month needs for its mean
TOL_MM = 2.0           # plan WP S2 threshold, read strictly (every unit)
MIN_R = 0.85           # plan WP S2 mapping threshold
MIN_N_R = 5            # months a sensor needs for its correlation


def monthly_means(daily: pd.DataFrame) -> pd.DataFrame:
    """Per sensor, feed and month: mean of the full-day values (flags 0-1), day count, and
    whether any day of the month carries a removed step (flag 1)."""
    d = daily[daily["flag"].isin([0, 1])]
    g = d.groupby(["sensor_id", "feed", "month"])
    out = g["value"].agg(["mean", "size"]).rename(columns={"mean": "mean_mm", "size": "n_days"})
    out["has_flag1"] = g["flag"].apply(lambda f: bool((f == 1).any()))
    return out.reset_index()


def _prev(m: str) -> str:
    return str(pd.Period(m, freq="M") - 1)


def monthly_mean_increments(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Monthly-mean increments (mm, rising = compression) per sensor and month.

    The increment of month m is mean(m) - mean(m-1) of ONE feed with at least MIN_DAYS full
    days in both months: the raw (catalogued) feed when it qualifies, else the vendor feed.
    Flag: 9 when the product's increment of month m or m-1 for that feed is flag 9 (a suspect
    stitch or unremoved step inside the interval), 1 when a removed step lies in either month
    (raw), else 0; 8 when no feed qualifies."""
    mm = monthly_means(daily).set_index(["sensor_id", "feed", "month"])
    fl = {}
    for r in monthly.itertuples():
        fl[(r.sensor_id, "raw", r.month)] = int(r.raw_flag)
        fl[(r.sensor_id, "vendor", r.month)] = int(r.vendor_flag)
    rows = []
    for sid in sorted(daily["sensor_id"].unique()):
        months = sorted(daily.loc[daily.sensor_id == sid, "month"].unique())
        for m in months:
            p = _prev(m)
            rec = {"sensor_id": sid, "month": m, "feed": "", "inc_mm": np.nan, "flag": 8,
                   "n_days": 0, "n_days_prev": 0}
            for feed in ("raw", "vendor"):
                k, kp = (sid, feed, m), (sid, feed, p)
                if k in mm.index and kp in mm.index and mm.at[k, "n_days"] >= MIN_DAYS \
                        and mm.at[kp, "n_days"] >= MIN_DAYS:
                    f9 = fl.get(k, 8) == 9 or fl.get(kp, 8) == 9
                    f1 = bool(mm.at[k, "has_flag1"] or mm.at[kp, "has_flag1"])
                    rec.update({"feed": feed,
                                "inc_mm": float(mm.at[k, "mean_mm"] - mm.at[kp, "mean_mm"]),
                                "flag": 9 if f9 else (1 if f1 else 0),
                                "n_days": int(mm.at[k, "n_days"]),
                                "n_days_prev": int(mm.at[kp, "n_days"])})
                    break
            rows.append(rec)
    return pd.DataFrame(rows)


def printed_units(printed: pd.DataFrame, last_month: str, first_month: str | None = None
                  ) -> pd.DataFrame:
    """One printed value per (sensor, month): the verified report-month column, else the
    verified prior-year column of the report twelve months later."""
    v = printed[(printed["flag"] == 0) & printed["sensor_id"].astype(bool)
                & (printed["month"] <= last_month)]
    if first_month is not None:
        v = v[v["month"] >= first_month]
    v = v.assign(rank=np.where(v["column"] == "report_month", 0, 1))
    v = v.sort_values(["sensor_id", "month", "rank"]).drop_duplicates(["sensor_id", "month"])
    return v.drop(columns="rank").reset_index(drop=True)


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score(units: pd.DataFrame, feed: pd.DataFrame, flags_ok=(0, 1)) -> dict:
    """Units: printed (sensor_id, month, value = compression mm). Feed: (sensor_id, month,
    inc_mm, flag). Returns per-unit table and the C criteria."""
    j = units[["sensor_id", "month", "value", "printed_mm", "column", "report"]].merge(
        feed[["sensor_id", "month", "inc_mm", "flag", "feed"]], on=["sensor_id", "month"],
        how="left")
    j["testable"] = j["flag"].isin(list(flags_ok)) & j["inc_mm"].notna()
    t = j[j.testable].copy()
    t["diff_mm"] = t["value"] - t["inc_mm"]
    t["ok"] = t["diff_mm"].abs() <= TOL_MM + 1e-9
    per = {}
    for sid, g in t.groupby("sensor_id"):
        r = pearson(g["value"], g["inc_mm"]) if len(g) >= MIN_N_R else float("nan")
        per[sid] = {"n": int(len(g)), "n_ok": int(g.ok.sum()),
                    "median_abs_mm": float(g.diff_mm.abs().median()),
                    "max_abs_mm": float(g.diff_mm.abs().max()), "r": r,
                    "r_testable": bool(len(g) >= MIN_N_R),
                    "r_pass": bool(len(g) >= MIN_N_R and r >= MIN_R)}
    a = t["diff_mm"].abs()
    res = {"n_units": int(len(j)), "n_testable": int(len(t)),
           "n_untestable_by_flag": {str(k): int(v) for k, v in
                                    j[~j.testable]["flag"].fillna(-1).astype(int)
                                    .value_counts().items()},
           "n_ok": int(t.ok.sum()), "share_ok": float(t.ok.mean()) if len(t) else float("nan"),
           "median_abs_mm": float(a.median()) if len(t) else float("nan"),
           "p95_abs_mm": float(a.quantile(0.95)) if len(t) else float("nan"),
           "max_abs_mm": float(a.max()) if len(t) else float("nan"),
           "per_sensor": per,
           "tolerance_pass": bool(len(t) > 0 and bool(t.ok.all())),
           "correlation_pass": bool(any(p["r_testable"] for p in per.values()) and all(
               p["r_pass"] for p in per.values() if p["r_testable"])),
           "failures": t[~t.ok][["sensor_id", "month", "value", "inc_mm", "flag", "feed",
                                 "diff_mm", "report", "column"]].round(3).to_dict("records")}
    res["pass"] = bool(res["tolerance_pass"] and res["correlation_pass"])
    return {"summary": res, "units": j.assign(
        diff_mm=j["value"] - j["inc_mm"])}
