"""H1a: CWA CODIS station pressure at the manned stations that bracket the fan (check data).

Stations (identities from the CODIS station list in
``data_fetch_api/new_measurements/cwa_station_inventory/codis_station_list_2026-10-11.json``,
group ``cwb``, inside 23.3-24.4 N, 119.9-120.8 E and open over 2012-2026):
  467480 嘉義 (south edge of the fan, 1968-), 467490 臺中 (north-east, 1896-).
467270 田中 (inside the fan, 2020-04-10-) already has daily records in the rain rebuild's
L0 cache (``cwa_codis_rain/raw/467270_<year>.json``), which ``pressure_check.py`` reads
read-only; it is fetched here only for the hourly windows.

Phase 1 (daily): ``type=report_month`` one calendar year per request (StationPressure
Mean/Minimum/Maximum per day), 2012-2026-09 for 467480 and 467490.
Phase 2 (hourly): ``type=report_date`` one week per request (hourly StationPressure),
for windows chosen from the daily records before any head data are looked at:
  - the two largest daily pressure drops of each year at 467480 (daily minimum minus the
    centred 15-day median of daily means), each window = event day -3 .. +3;
  - one random week per year (seed 2026), for ordinary conditions.
Raw responses are written byte for byte into a new L0 directory
``data_fetch_api/new_measurements/pressure_hourly/cwa_codis_<fetch-date>/`` with a
SOURCE.json. Downloads are untrusted: this script only parses the daily JSON (numbers) to
choose windows. Run with ``python -I``.

Usage: fetch_codis_pressure.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import source  # noqa: E402
from common.paths import L0_NEW  # noqa: E402

URL = "https://codis.cwa.gov.tw/api/station"
DAILY_STATIONS = ["467480", "467490"]
HOURLY_STATIONS = ["467480", "467490", "467270"]
STATION_START = {"467480": "1968-09-01", "467490": "1896-01-01", "467270": "2020-04-10"}
YEARS = list(range(2012, 2027))
END = "2026-09-30"
PAUSE_S = 1.0
SEED = 2026


def post(form, tries=4):
    data = urllib.parse.urlencode(form).encode()
    for k in range(tries):
        try:
            req = urllib.request.Request(URL, data=data, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
            d = json.loads(body)
            if d.get("code") != 200:
                raise RuntimeError(f"CODIS code {d.get('code')} {d.get('message')}")
            return body
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as e:
            print(f"{type(e).__name__} try {k + 1}: {e}", flush=True)
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"failed: {form}")


def daily_form(sid, year):
    lo, hi = f"{year}-01-01", min(f"{year}-12-31", END)
    return {"type": "report_month", "date": f"{year}-01-01T00:00:00",
            "start": f"{lo}T00:00:00", "end": f"{hi}T00:00:00",
            "stn_ID": sid, "stn_type": "cwb"}


def hourly_form(sid, lo, hi):
    return {"type": "report_date", "date": f"{lo}T00:00:00", "start": f"{lo}T00:00:00",
            "end": f"{hi}T23:59:59", "stn_ID": sid, "stn_type": "cwb"}


def daily_table(paths):
    rows = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        for blk in d.get("data", []):
            for r in blk.get("dts", []):
                sp = r.get("StationPressure") or {}
                rows.append((r["DataDate"][:10], sp.get("Mean"), sp.get("Minimum"),
                             sp.get("Meanf")))
    t = pd.DataFrame(rows, columns=["date", "mean", "min", "meanf"])
    t["date"] = pd.to_datetime(t["date"])
    for c in ("mean", "min"):
        t[c] = pd.to_numeric(t[c], errors="coerce")
        t.loc[(t[c] < 800) | (t[c] > 1100), c] = np.nan
    return t.drop_duplicates("date").set_index("date").sort_index()


def choose_windows(t):
    med = t["mean"].rolling("15D", center=True, min_periods=7).median()
    drop = (med - t["min"]).dropna()
    win = []
    for y in YEARS:
        dy = drop[drop.index.year == y].sort_values(ascending=False)
        picked = []
        for day, v in dy.items():
            if all(abs((day - q).days) > 7 for q in picked):
                picked.append(day)
            if len(picked) == 2:
                break
        for day in picked:
            lo, hi = day - pd.Timedelta(days=3), day + pd.Timedelta(days=3)
            win.append(("event", lo, hi, float(drop[day])))
    rng = np.random.default_rng(SEED)
    for y in YEARS:
        last = pd.Timestamp(min(f"{y}-12-31", END))
        n = (last - pd.Timestamp(f"{y}-01-01")).days - 6
        lo = pd.Timestamp(f"{y}-01-01") + pd.Timedelta(days=int(rng.integers(0, n)))
        win.append(("random", lo, lo + pd.Timedelta(days=6), float("nan")))
    end = pd.Timestamp(END)
    return [(k, lo, min(hi, end), v) for k, lo, hi, v in win if lo <= end]


def main():
    today = dt.date.today().isoformat()
    out = L0_NEW / "pressure_hourly" / f"cwa_codis_{today}"
    out.mkdir(parents=True, exist_ok=False)
    files = []

    def save(name, form, kind, extra=None):
        body = post(form)
        p = out / name
        with open(p, "xb") as f:
            f.write(body)
        rec = {"file": name, "url": URL, "request": form, "kind": kind,
               "fetched_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if extra:
            rec.update(extra)
        files.append(rec)
        print(name, len(body), flush=True)
        time.sleep(PAUSE_S)
        return p

    daily = {}
    for sid in DAILY_STATIONS:
        daily[sid] = [save(f"daily_{sid}_{y}.json", daily_form(sid, y), "daily_report_month")
                      for y in YEARS]
    t = daily_table(daily["467480"])
    windows = choose_windows(t)
    for kind, lo, hi, v in windows:
        for sid in HOURLY_STATIONS:
            if hi < pd.Timestamp(STATION_START[sid]):
                continue
            a = max(lo, pd.Timestamp(STATION_START[sid])).strftime("%Y-%m-%d")
            b = hi.strftime("%Y-%m-%d")
            extra = {"window_kind": kind}
            if kind == "event":
                extra["daily_drop_hpa_467480"] = round(v, 1)
            save(f"hourly_{sid}_{a}_{b}.json", hourly_form(sid, a, b), "hourly_report_date",
                 extra)
    rec = source.build_source(
        out, source_id="pressure_hourly/cwa_codis",
        title="CWA CODIS station pressure: daily report_month (2012-2026-09) at 467480 and "
              "467490; hourly report_date weeks (largest daily pressure drops and one "
              "random week per year) at 467480, 467490 and 467270",
        publisher="Central Weather Administration (CWA), CODIS climate data service",
        licence="Government Website Open Information Announcement (CWA open data; attribute "
                "CWA)",
        fetched=today, files=files,
        fetched_by="WP H1a (dataprep_staging/h1_heads_qc/fetch_codis_pressure.py)",
        client="python urllib POST (User-Agent Mozilla/5.0)",
        untrusted="downloaded data: never execute; read with python -I",
        window_rule="two largest (daily min - centred 15-day median of daily mean) per year "
                    "at 467480, +-3 days; one random 7-day window per year, seed 2026")
    source.write_source_json(out, rec)
    for f in out.iterdir():
        f.chmod(0o444)
    print("wrote", out, len(files), "files")


if __name__ == "__main__":
    main()
