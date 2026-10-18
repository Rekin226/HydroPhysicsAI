"""Once-daily records (the R-h find): monthly means from one reading per day, their
representativeness error, and a persistence screen (registered test ``H3.snapshot``).

Well 09030212's hourly record stops on 2018-12-31 like the other wells of R-h, but the twin's
own cache continues it with one reading per day at 00:00 from 2019-01-01 to 2022-12-31. L1 v3
keeps those days (flag 0) but marks them ``day_valid = False`` (its day rule needs >= 12
hours), so they never reach a C20 month. This module turns them into monthly values and
measures, on pre-origin hourly records only, how well a month of 00:00 readings represents
the C20 monthly mean:

* ``snapshot_errors``: for every well with an hourly record, the raw L0 reading at 00:00 of
  each L1 flag-0/1 valid day in 2012-01..2018-12; a month with >= 20 such readings that is an
  L1 C20 month gives the error ``mean(readings) - C20 value``;
* ``sigma_snap``: the pooled RMSE of those errors over the wells of the same raw layer code,
  the reviewed well excluded;
* ``screen_daily``: day-to-day jumps >= 1 m of the once-daily series that persist (median of
  the next 10 readings minus median of the previous 10 has the jump's sign and at least half
  its size); months holding a persistent jump, months between two persistent jumps of
  opposite sign less than 365 days apart, and every month after an unpaired persistent jump
  of >= 3 m are excluded;
* ``snapshot_months``: months 2019-01..2022-12 with >= 20 screened once-daily readings.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from h3_recovered_wells import config as C

sys.path[:0] = [str(C.REPO / "dataprep_staging"), str(C.REPO)]
from common import guard  # noqa: E402

TEST_ID = C.TEST_SNAP
SNAP_START, SNAP_END = "2019-01-01", "2022-12-31"   # the once-daily era (pre-origin part)
EST_END = "2018-12-31"                             # hourly era used for the estimate
MIN_DAYS = 20                                      # C20 analogue
MIN_MONTHS_WELL = 24                               # wells entering the pooled estimate
JUMP_M, STEP_M, PERSIST_N, PAIR_DAYS = 1.0, 3.0, 10, 365


def l1_daily(sids=None) -> pd.DataFrame:
    hp = guard.open_product(C.L1_DIR, "selection", window_end=C.PRE_END)
    d = hp.read_table("heads_daily.parquet",
                      columns=["sid", "date", "value", "flag", "day_valid", "n_samples"])
    d["sid"] = d["sid"].astype(str)
    if sids is not None:
        d = d[d.sid.isin(set(sids))]
    d["date"] = pd.to_datetime(d["date"])
    return d


def l1_monthly() -> pd.DataFrame:
    hp = guard.open_product(C.L1_DIR, "selection", window_end=C.PRE_END)
    m = hp.read_table("heads_monthly.parquet", columns=["sid", "date", "flag", "c20", "value"])
    m["sid"] = m["sid"].astype(str)
    m = m[m.flag.isin([0, 1]) & m.c20.astype(bool)]
    m["date"] = pd.to_datetime(m["date"])
    return m


def raw_midnight(sid: str) -> pd.Series:
    """Raw L0 reading at 00:00 of each day (first sample in [00:00, 01:00)), 2012-2018."""
    p = C.AMP_WELLS / f"{sid}.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    s = pd.read_parquet(p)["value"]
    s.index = pd.to_datetime(s.index)
    s = s.where(s.abs() < 1000).dropna().sort_index()
    s = s[(s.index >= f"{C.START}-01") & (s.index <= f"{EST_END} 23:59:59")]
    s = s[s.index.hour == 0]
    return s.groupby(s.index.normalize()).first()


def snapshot_errors(sids, daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid in sids:
        snap = raw_midnight(sid)
        if snap.empty:
            continue
        ok_days = daily[(daily.sid == sid) & daily.flag.isin([0, 1])
                        & daily.day_valid.astype(bool)].date
        snap = snap[snap.index.isin(set(ok_days))]
        if snap.empty:
            continue
        g = snap.groupby(snap.index.to_period("M")).agg(["mean", "count"])
        g = g[g["count"] >= MIN_DAYS]
        mm = monthly[monthly.sid == sid].set_index(monthly[monthly.sid == sid].date
                                                   .dt.to_period("M"))["value"]
        j = g.join(mm.rename("c20"), how="inner")
        for per, r in j.iterrows():
            rows.append({"sid": sid, "month": str(per), "snap_mean": float(r["mean"]),
                         "n_snap": int(r["count"]), "c20": float(r.c20),
                         "err": float(r["mean"] - r.c20)})
    return pd.DataFrame(rows)


def sigma_snap(err: pd.DataFrame, sids_same_layer, exclude) -> dict:
    e = err[err.sid.isin(set(sids_same_layer)) & ~err.sid.isin(set(exclude))]
    n_by = e.groupby("sid").size()
    keep = n_by[n_by >= MIN_MONTHS_WELL].index
    e = e[e.sid.isin(keep)]
    v = e.err.to_numpy()
    return {"sigma_snap": float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan"),
            "n_months": int(len(v)), "n_wells": int(len(keep)),
            "median_well_rmse": float(e.groupby("sid").err.apply(
                lambda x: np.sqrt(np.mean(x ** 2))).median()) if len(v) else float("nan"),
            "bias": float(np.mean(v)) if len(v) else float("nan")}


def screen_daily(s: pd.Series) -> dict:
    """Persistence screen of a once-daily series (index: days)."""
    s = s.sort_index()
    v = s.to_numpy(float)
    d = np.diff(v)
    jumps = []
    for i in np.where(np.abs(d) >= JUMP_M)[0]:
        pre = v[max(0, i + 1 - PERSIST_N):i + 1]
        post = v[i + 1:i + 1 + PERSIST_N]
        if len(pre) < 5 or len(post) < 5:
            continue
        shift = np.median(post) - np.median(pre)
        if np.sign(shift) == np.sign(d[i]) and abs(shift) >= 0.5 * abs(d[i]):
            jumps.append({"date": s.index[i + 1], "jump": float(d[i]), "shift": float(shift)})
    bad = set()
    used = set()
    for a, ja in enumerate(jumps):
        bad.add(ja["date"].to_period("M"))
        for b in range(a + 1, len(jumps)):
            jb = jumps[b]
            if b in used or np.sign(jb["jump"]) == np.sign(ja["jump"]):
                continue
            if (jb["date"] - ja["date"]).days <= PAIR_DAYS:
                for per in pd.period_range(ja["date"].to_period("M"), jb["date"].to_period("M"),
                                           freq="M"):
                    bad.add(per)
                used.update({a, b})
                break
    for a, ja in enumerate(jumps):
        if a not in used and abs(ja["jump"]) >= STEP_M:
            for per in pd.period_range(ja["date"].to_period("M"),
                                       pd.Period(SNAP_END[:7], "M"), freq="M"):
                bad.add(per)
    return {"persistent_jumps": [{"date": str(j["date"].date()), "jump": j["jump"],
                                  "shift": j["shift"]} for j in jumps],
            "excluded_months": sorted(str(p) for p in bad)}


def snapshot_months(sid: str, daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Monthly means of the screened once-daily readings, 2019-01..2022-12."""
    d = daily[(daily.sid == sid) & daily.flag.isin([0, 1]) & ~daily.day_valid.astype(bool)
              & (daily.n_samples >= 1) & daily.value.notna()
              & (daily.date >= SNAP_START) & (daily.date <= SNAP_END)]
    s = d.set_index("date")["value"]
    scr = screen_daily(s)
    per = s.index.to_period("M")
    g = s.groupby(per).agg(["mean", "count"])
    g = g[(g["count"] >= MIN_DAYS) & ~g.index.astype(str).isin(scr["excluded_months"])]
    out = pd.DataFrame({"month": g.index.astype(str), "value": g["mean"].to_numpy(),
                        "n_days": g["count"].to_numpy()})
    return out, scr
