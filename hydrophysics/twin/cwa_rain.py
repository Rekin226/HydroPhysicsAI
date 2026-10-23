"""Rebuild the twin's 26 daily rain gauges from the CWA CODIS archive.

The vendor rainfall cache (``chou-shui-data/data/rf_timeseries.csv``, 2012-2022) has no
missing values because it zero-fills every gap: gauge rf11 (田中, C0G850) closed on
2019-10-29 and the cache carries 0 mm for ~26 months including all of 2021-22, gauge rf15
(崙背) is 0 mm for the second half of 2015 while its station was being upgraded. The CWA
station inventory (``data_fetch_api/new_measurements/cwa_station_inventory/``) records
every closure and renumbering, so each gauge is a *chain* of station IDs. CODIS serves one
station ID and one calendar year per request; this module fetches every (station, year)
of every chain, keeps the raw responses with their hashes, builds one daily table with
missing values left missing, and compares it day by day with the vendor cache.

Rules (no value is invented):

* a day is numeric only when CODIS prints an unflagged number; CWA's missing-value mark
  (``Accumulationf`` > 0), sentinels and absent days are NaN;
* trace (CWA prints ``-9.8`` for < 0.5 mm) is recorded as 0.0 mm, an interval-censored
  amount whose monthly contribution is below the 0.1 mm print resolution;
* where two IDs of a chain overlap (the new station usually starts before the old one is
  closed) the newer station's finite value is used, else the older one;
* nothing is interpolated: the recharge loader (``calibrate_flow.monthly_mean_rate``)
  takes a month's mean of the finite days and masks months with less than 80 % of days,
  and the IDW down-weights missing gauges per month.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .field_observations import read_codis_rainfall

CODIS_URL = "https://codis.cwa.gov.tw/api/station"
RECORD_START = "2012-01-01"
TAIPEI = timezone(timedelta(hours=8))

# rf_id -> [(station_id, stn_type, first day, last day or None), ...] oldest first, from the
# CWA station inventory (stationStartDate / stationEndDate). Coordinates of a renumbered
# gauge move by at most ~2 km (田中 C0G850 -> manned 467270); ``rf_stations.csv`` keeps the
# vendor coordinates. rf24 (臺西, C0K530) has no CWA predecessor: the vendor's 2012-03/2015
# values match no CWA station within 15 km and are left missing here.
CHAINS: dict[str, list[tuple[str, str, str, str | None]]] = {
    "rf1": [("C0G650", "auto_C0", "1992-05-06", None)],
    "rf2": [("C0G660", "auto_C0", "2011-11-01", None)],
    "rf3": [("C1G670", "auto_C0", "1992-05-05", "2015-05-27"),
            ("C0G720", "auto_C0", "2015-03-12", None)],
    "rf4": [("C1G700", "auto_C0", "1992-05-05", "2015-06-05"),
            ("C0G730", "auto_C0", "2015-07-01", None)],
    "rf5": [("C0G770", "auto_C0", "2011-06-02", None)],
    "rf6": [("C0G780", "auto_C0", "2011-08-01", None)],
    "rf7": [("C0G800", "auto_C0", "2011-04-12", None)],
    "rf26": [("C0G810", "auto_C0", "2011-04-27", None)],
    "rf8": [("C0G820", "auto_C0", "2011-05-05", None)],
    "rf9": [("C0G830", "auto_C0", "2011-05-25", None)],
    "rf10": [("C0G840", "auto_C0", "2011-04-01", "2024-01-08"),
             ("C2G840", "agr", "2024-01-08", None)],
    "rf11": [("C0G850", "auto_C0", "2011-08-01", "2019-10-29"),
             ("467270", "cwb", "2020-04-10", None)],
    "rf12": [("C0G860", "auto_C0", "2011-05-25", None)],
    "rf13": [("C0G870", "auto_C0", "2011-07-19", "2024-01-08"),
             ("C2G870", "agr", "2024-01-08", None)],
    "rf14": [("C0G890", "auto_C0", "2011-02-18", None)],
    "rf15": [("C1K250", "auto_C0", "1992-12-15", "2015-06-08"),
             ("C0K250", "auto_C0", "2015-06-08", "2025-08-15"),
             ("C2K250", "agr", "2025-08-16", None)],
    "rf16": [("C0K280", "auto_C0", "1992-12-05", "2025-08-15"),
             ("C2K280", "agr", "2025-08-16", None)],
    "rf17": [("C0K291", "auto_C0", "2010-09-27", None)],
    "rf18": [("C0K330", "auto_C0", "1992-12-12", None)],
    "rf19": [("C1K340", "auto_C0", "1992-12-24", "2015-05-28"),
             ("C0K390", "auto_C0", "2015-05-07", None)],
    "rf20": [("C1K310", "auto_C0", "1992-12-17", "2015-05-28"),
             ("C0K400", "auto_C0", "2015-03-10", None)],
    "rf21": [("C1K350", "auto_C0", "1992-12-05", "2015-05-26"),
             ("C0K410", "auto_C0", "2015-03-23", None)],
    "rf22": [("C1K230", "auto_C0", "1992-05-05", "2015-05-27"),
             ("C0K420", "auto_C0", "2015-05-06", None)],
    "rf23": [("C1K270", "auto_C0", "1992-12-24", "2015-05-28"),
             ("C0K430", "auto_C0", "2015-04-29", None)],
    "rf24": [("C0K530", "auto_C0", "2015-03-28", None)],
    "rf25": [("C1G691", "auto_C0", "2011-10-06", None)],
}
GAUGE_ORDER = ["rf1", "rf2", "rf3", "rf4", "rf5", "rf6", "rf7", "rf26", "rf8", "rf9", "rf10",
               "rf11", "rf12", "rf13", "rf14", "rf15", "rf16", "rf17", "rf18", "rf19", "rf20",
               "rf21", "rf22", "rf23", "rf24", "rf25"]
TRACE = "trace_less_than_0.5_mm"
NUMERIC = "numeric_unmarked"


def yesterday_taipei() -> str:
    return (datetime.now(TAIPEI) - timedelta(days=1)).strftime("%Y-%m-%d")


def requests_for(chains: dict, start: str, end: str) -> list[tuple[str, str, str, int]]:
    """Every (rf_id, station, stn_type, year) whose inventory window meets [start, end]."""
    out = []
    y0, y1 = int(start[:4]), int(end[:4])
    for rf_id, chain in chains.items():
        for sid, typ, first, last in chain:
            a = max(int(first[:4]), y0)
            b = min(int(last[:4]) if last else y1, y1)
            for year in range(a, b + 1):
                out.append((rf_id, sid, typ, year))
    return out


def year_window(year: int, start: str, end: str) -> tuple[str, str]:
    lo = max(f"{year}-01-01", start)
    hi = min(f"{year}-12-31", end)
    return lo, hi


def raw_path(raw_dir: str, sid: str, year: int) -> str:
    return os.path.join(raw_dir, f"{sid}_{year}.json")


def fetch_one(sid: str, typ: str, lo: str, hi: str, timeout: float = 120.0) -> bytes:
    form = {"type": "report_month", "date": f"{lo[:4]}-01-01T00:00:00",
            "start": f"{lo}T00:00:00", "end": f"{hi}T00:00:00",
            "stn_ID": sid, "stn_type": typ}
    req = urllib.request.Request(CODIS_URL, data=urllib.parse.urlencode(form).encode(),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch(raw_dir: str, start: str = RECORD_START, end: str | None = None,
          chains: dict | None = None, pause: float = 0.5, retries: int = 3,
          refresh_year: int | None = None, log=print) -> dict:
    """Download every (station, year) not yet on disk; the current year is re-fetched when
    ``refresh_year`` names it. Writes ``<sid>_<year>.json`` plus a ``.meta.json`` with the
    sha256, the request and the UTC fetch time (the importer's ``acquired_at``)."""
    end = end or yesterday_taipei()
    chains = chains or CHAINS
    os.makedirs(raw_dir, exist_ok=True)
    done = skipped = failed = 0
    for rf_id, sid, typ, year in requests_for(chains, start, end):
        path = raw_path(raw_dir, sid, year)
        if os.path.exists(path) and year != refresh_year:
            skipped += 1
            continue
        lo, hi = year_window(year, start, end)
        body = None
        for attempt in range(retries):
            try:
                body = fetch_one(sid, typ, lo, hi)
                payload = json.loads(body)
                if payload.get("code") != 200:
                    raise ValueError(f"CODIS code {payload.get('code')}: "
                                     f"{payload.get('message')}")
                break
            except Exception as exc:  # noqa: BLE001 - network; retried, then reported
                log(f"{rf_id} {sid} {year}: attempt {attempt + 1} failed: {exc}")
                body = None
                time.sleep(5.0 * (attempt + 1))
        if body is None:
            failed += 1
            continue
        fetched_at = datetime.now(timezone.utc).isoformat()
        with open(path, "wb") as fh:
            fh.write(body)
        meta = {"rf_id": rf_id, "station_id": sid, "stn_type": typ, "start": lo, "end": hi,
                "url": CODIS_URL, "fetched_at": fetched_at,
                "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body),
                "count": json.loads(body).get("metadata", {}).get("count")}
        with open(path.replace(".json", ".meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        done += 1
        log(f"{rf_id} {sid} {year}: {meta['count']} days")
        time.sleep(pause)
    return {"fetched": done, "skipped": skipped, "failed": failed, "end": end}


def station_daily(raw_dir: str, sid: str, years: list[int], start: str, end: str
                  ) -> tuple[pd.Series, dict]:
    """One station's daily mm (NaN where not numeric; trace = 0.0) and its QA counts."""
    parts, counts = [], {}
    for year in years:
        path = raw_path(raw_dir, sid, year)
        meta_path = path.replace(".json", ".meta.json")
        if not os.path.exists(path):
            counts["missing_file"] = counts.get("missing_file", 0) + 1
            continue
        with open(meta_path) as fh:
            meta = json.load(fh)
        lo, hi = year_window(year, start, end)
        try:
            df = read_codis_rainfall(path, station_id=sid, start=lo, end=hi,
                                     acquired_at=meta["fetched_at"])
        except ValueError as exc:
            if "no observations" in str(exc):
                counts["empty_year"] = counts.get("empty_year", 0) + 1
                continue
            raise
        for status, n in df.qa_status.value_counts().items():
            counts[status] = counts.get(status, 0) + int(n)
        value = df.value.where(df.qa_status != TRACE, 0.0)
        day = pd.DatetimeIndex(df.period_start.dt.tz_localize(None))
        parts.append(pd.Series(value.to_numpy(float), index=day))
    if not parts:
        return pd.Series(dtype=float), counts
    s = pd.concat(parts).sort_index()
    if s.index.duplicated().any():
        raise ValueError(f"{sid}: duplicate days across year files")
    return s, counts


def chain_daily(raw_dir: str, chain: list, start: str, end: str
                ) -> tuple[pd.Series, pd.Series, dict]:
    """Merge a gauge's station chain -> (mm, source station per day, QA counts by station).
    The newest station's finite value wins where stations overlap."""
    index = pd.date_range(start, end, freq="D")
    value = pd.Series(np.nan, index=index)
    source = pd.Series("", index=index, dtype=object)
    qa = {}
    y0, y1 = int(start[:4]), int(end[:4])
    for sid, _typ, first, last in chain:   # oldest first; later ones overwrite
        years = list(range(max(int(first[:4]), y0), min(int(last[:4]) if last else y1, y1) + 1))
        s, counts = station_daily(raw_dir, sid, years, start, end)
        qa[sid] = counts
        s = s.reindex(index)
        has = s.notna()
        value[has] = s[has]
        source[has] = sid
    return value, source, qa


def build(raw_dir: str, start: str = RECORD_START, end: str | None = None,
          chains: dict | None = None, order: list[str] | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    end = end or yesterday_taipei()
    chains = chains or CHAINS
    order = order or [k for k in GAUGE_ORDER if k in chains]
    values, sources, qa = {}, {}, {}
    for rf_id in order:
        v, s, q = chain_daily(raw_dir, chains[rf_id], start, end)
        values[rf_id], sources[rf_id], qa[rf_id] = v, s, q
    daily = pd.DataFrame(values)
    daily.index.name = "date time"
    src = pd.DataFrame(sources)
    src.index.name = "date time"
    return daily, src, qa


def compare_with_cache(new: pd.DataFrame, cache: pd.DataFrame, tol: float = 0.05
                       ) -> pd.DataFrame:
    """Day-by-day agreement of the rebuilt record with the vendor cache on their overlap.

    ``cache_zero_codis_wet`` counts days the cache prints 0 while CODIS has rain (the
    zero-fill signature); ``codis_missing_cache_value`` days the cache has a number CODIS
    does not (vendor-only source or a flagged CODIS total)."""
    idx = new.index.intersection(cache.index)
    rows = []
    for c in new.columns:
        a, b = new.loc[idx, c], cache.loc[idx, c] if c in cache else pd.Series(np.nan, idx)
        both = a.notna() & b.notna()
        equal = both & ((a - b).abs() <= tol)
        zero_wet = both & (b == 0) & (a > 0)
        rows.append({"gauge": c, "days": int(len(idx)), "both_numeric": int(both.sum()),
                     "equal": int(equal.sum()),
                     "cache_zero_codis_wet": int(zero_wet.sum()),
                     "cache_zero_codis_wet_mm": float(a[zero_wet].sum()),
                     "codis_missing_cache_value": int((a.isna() & b.notna()).sum()),
                     "codis_missing_cache_nonzero": int((a.isna() & (b > 0)).sum()),
                     "other_disagreements": int((both & ~equal & ~zero_wet).sum()),
                     "other_mad_mm": float((a - b)[both & ~equal & ~zero_wet].abs().mean())
                     if int((both & ~equal & ~zero_wet).sum()) else 0.0,
                     "codis_mm_total": float(a.sum()), "cache_mm_total": float(b.sum())})
    return pd.DataFrame(rows)


def monthly_summary(new: pd.DataFrame, cache: pd.DataFrame, min_fraction: float = 0.8
                    ) -> dict:
    """Months (on the cache period) where the rebuilt monthly rate differs by > 0.5 mm/day."""
    idx = new.index.intersection(cache.index)
    a = new.loc[idx].resample("ME")
    cnt, days = a.count(), a.mean().index.days_in_month.to_numpy()[:, None]
    ma = a.mean().where(cnt >= min_fraction * days)
    mb = cache.loc[idx].resample("ME").mean()
    diff = (ma - mb[ma.columns])
    big = diff.abs() > 0.5
    return {"months": int(len(ma)), "gauge_months": int(ma.size),
            "codis_month_missing": int(ma.isna().sum().sum()),
            "gauge_months_changed_over_0.5_mm_day": int(big.sum().sum()),
            "changed_by_gauge": {c: int(big[c].sum()) for c in ma.columns if big[c].sum()},
            "mean_rate_codis_mm_day": float(np.nanmean(ma.to_numpy())),
            "mean_rate_cache_mm_day": float(np.nanmean(mb[ma.columns].to_numpy()))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download every (station, year) of every gauge chain")
    f.add_argument("--raw", required=True, help="directory for the raw CODIS responses")
    f.add_argument("--start", default=RECORD_START)
    f.add_argument("--end", default=None, help="last day (default: yesterday, Taipei)")
    f.add_argument("--refresh-year", type=int, default=None,
                   help="re-fetch this year's files (the current year grows)")
    b = sub.add_parser("build", help="assemble the daily table and compare with the cache")
    b.add_argument("--raw", required=True)
    b.add_argument("--start", default=RECORD_START)
    b.add_argument("--end", default=None)
    b.add_argument("--cache", default="chou-shui-data/data/rf_timeseries.csv")
    b.add_argument("--out-csv", required=True, help="daily table (NaN where missing)")
    b.add_argument("--report", required=True, help="comparison report json")
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        res = fetch(args.raw, args.start, args.end, refresh_year=args.refresh_year)
        print(json.dumps(res))
        return
    daily, src, qa = build(args.raw, args.start, args.end)
    cache = pd.read_csv(args.cache, index_col=0, parse_dates=True)
    comp = compare_with_cache(daily, cache)
    report = {"start": args.start, "end": args.end or yesterday_taipei(), "raw": args.raw,
              "cache": args.cache, "gauges": list(daily.columns),
              "chains": {k: [list(t) for t in v] for k, v in CHAINS.items()},
              "qa_counts": qa, "comparison": comp.to_dict(orient="records"),
              "monthly": monthly_summary(daily, cache),
              "days_numeric": {c: int(daily[c].notna().sum()) for c in daily.columns},
              "last_numeric_day": {c: (str(daily[c].last_valid_index().date())
                                       if daily[c].notna().any() else None)
                                   for c in daily.columns}}
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    daily.round(1).to_csv(args.out_csv, float_format="%.1f")
    src.to_csv(args.out_csv.replace(".csv", "_source.csv"))
    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(comp.to_string(index=False))
    print(json.dumps(report["monthly"], indent=1))


if __name__ == "__main__":
    main()
