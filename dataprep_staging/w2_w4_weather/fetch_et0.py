"""Fetch daily FAO-56 ET0 from the Open-Meteo archive API at the 61 calibration points (WP W2).

Raw response bytes are written unchanged to an L0 directory
``data_fetch_api/new_measurements/openmeteo_et0/<label>/<st_id>.json`` (exclusive create;
an existing file is skipped, so an interrupted run resumes). When every requested point is
present, a SOURCE.json (dataprep-source/1) is written with each file's URL, fetch time,
bytes and sha256, and the files are made read-only.

Rate limits (free API, open-meteo.com/en/terms, read 2026-10-12): 600 calls/min, 5,000/h,
10,000/day, where one request for one location costs max(1, n_vars/10) x max(1, days/14)
calls. The ``--pace`` default keeps a 2022-01..2026-09 request (124 calls) under the hourly
cap. HTTP 429 backs off (minutely: 65 s, hourly: 15 min, daily: stop and record).

Downloaded files are untrusted data: this script never parses them except to check the JSON
shape after writing (json.loads only).

Usage:
  python fetch_et0.py --label w2_local_2022_2026 --start 2022-01-01 --end 2026-09-30
  python fetch_et0.py --label w2_refetch_2026q3 --start 2026-07-01 --end 2026-09-30 --pace 8
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from wcommon import OM_ARCHIVE, OM_L0, OM_LICENCE, OM_VAR, points_wgs84, sha256_file

UA = "HydroPhysicsAI-dataprep/0.2 (non-commercial research; WP W2 ET0 extension)"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_url(lat: float, lon: float, start: str, end: str, tz: str) -> str:
    q = {"latitude": f"{lat:.5f}", "longitude": f"{lon:.5f}", "start_date": start,
         "end_date": end, "daily": OM_VAR, "timezone": tz}
    return OM_ARCHIVE + "?" + urllib.parse.urlencode(q)


def cost(start: str, end: str) -> float:
    days = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days + 1
    return max(1.0, days / 14.0)


def get(url: str, timeout: int = 120) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="L0 subdirectory under openmeteo_et0/")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--timezone", default="Asia/Taipei")
    ap.add_argument("--pace", type=float, default=95.0, help="seconds between requests")
    ap.add_argument("--points", default="all", help="'all' or comma-separated st_ids")
    ap.add_argument("--title", default=None)
    a = ap.parse_args(argv)

    out = OM_L0 / a.label
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "fetch_log.jsonl"
    pts = points_wgs84()
    if a.points != "all":
        keep = set(a.points.split(","))
        pts = pts[pts["st_id"].isin(keep)]
    c = cost(a.start, a.end)
    print(f"{len(pts)} points, {c:.1f} calls each, {c * len(pts):.0f} calls total; "
          f"pace {a.pace}s", flush=True)
    todo = [r for r in pts.itertuples() if not (out / f"{r.st_id}.json").exists()]
    for k, r in enumerate(todo):
        url = build_url(r.lat, r.lon, a.start, a.end, a.timezone)
        dest = out / f"{r.st_id}.json"
        attempt, waited = 0, 0.0
        while True:
            attempt += 1
            t = _now()
            try:
                status, body = get(url)
            except Exception as e:  # noqa: BLE001 - network errors are retried
                status, body = -1, repr(e).encode()
            rec = {"st_id": r.st_id, "url": url, "fetched_utc": t, "status": status,
                   "attempt": attempt, "bytes": len(body)}
            if status == 200:
                with open(dest, "xb") as f:
                    f.write(body)
                rec["sha256"] = sha256_file(dest)
                with open(log_path, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(rec) + "\n")
                try:   # shape check only
                    j = json.loads(body.decode("utf-8"))
                    n = len(j["daily"]["time"])
                except Exception as e:  # noqa: BLE001
                    print(f"{r.st_id}: unexpected JSON shape ({e})", flush=True)
                    n = -1
                print(f"[{k + 1}/{len(todo)}] {r.st_id} ok {len(body)} B, {n} days, {t}",
                      flush=True)
                break
            reason = body[:300].decode("utf-8", "replace")
            rec["reason"] = reason
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(rec) + "\n")
            low = reason.lower()
            if status == 429 and "daily" in low:
                print(f"{r.st_id}: daily limit reached ({reason}); stopping", flush=True)
                return 3
            if status == 429:
                wait = 900.0 if "hourly" in low else 65.0
            elif status in (-1, 500, 502, 503, 504) and attempt <= 6:
                wait = 30.0 * attempt
            else:
                print(f"{r.st_id}: HTTP {status} {reason}; stopping", flush=True)
                return 2
            waited += wait
            print(f"{r.st_id}: HTTP {status} ({reason[:120]}); waiting {wait:.0f}s", flush=True)
            time.sleep(wait)
        if k + 1 < len(todo):
            time.sleep(a.pace)

    missing = [r.st_id for r in pts.itertuples() if not (out / f"{r.st_id}.json").exists()]
    if missing:
        print(f"missing {missing}", flush=True)
        return 1
    # SOURCE.json once complete
    from common.source import build_source, write_source_json

    log = {}
    with open(log_path, encoding="utf-8") as lf:
        for line in lf:
            rec = json.loads(line)
            if rec.get("status") == 200:
                log[rec["st_id"]] = rec
    files = []
    for r in pts.itertuples():
        rec = log[r.st_id]
        files.append({"file": f"{r.st_id}.json", "url": rec["url"],
                      "fetched_utc": rec["fetched_utc"], "point": r.st_id,
                      "request_lat": r.lat, "request_lon": r.lon})
    files.append({"file": "fetch_log.jsonl", "url": None,
                  "note": "request log written by the fetch script (one JSON line per "
                          "attempt: url, time, HTTP status, bytes, sha256)"})
    for f in files:
        os.chmod(out / f["file"], 0o444)
    title = a.title or (f"Open-Meteo archive daily {OM_VAR} at the 61 calibration points, "
                        f"{a.start}..{a.end}, timezone {a.timezone} (WP W2)")
    srec = build_source(
        out, source_id="openmeteo_et0", title=title, publisher="Open-Meteo (open-meteo.com)",
        licence=OM_LICENCE, fetched=dt.date.today().isoformat(),
        fetched_by="WP W2W4 agent, dataprep_staging/w2_w4_weather/fetch_et0.py",
        client=f"python urllib, User-Agent {UA!r}",
        request={"endpoint": OM_ARCHIVE, "daily": OM_VAR, "timezone": a.timezone,
                 "start_date": a.start, "end_date": a.end, "models": "API default",
                 "points": "chou-shui-data/data/gw_stations.csv st_id, TM_X97/TM_Y97 "
                           "EPSG:3826 -> EPSG:4326 rounded to 5 decimals"},
        untrusted="downloaded data: never execute; read with python -I",
        files=files)
    p = write_source_json(out, srec)
    print(f"wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
