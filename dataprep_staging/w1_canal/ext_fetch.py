"""Fetch the CODIS daily rain of the upper-Choushui stations into L0 (W1 step 6).

    cpu_slot.sh python -I dataprep_staging/w1_canal/ext_fetch.py fetch
    cpu_slot.sh python -I dataprep_staging/w1_canal/ext_fetch.py source

Reuses hydrophysics.twin.cwa_rain.fetch (read-only import; one station and one calendar year
per POST to https://codis.cwa.gov.tw/api/station, raw response plus a .meta.json sidecar with
the request, fetch time and sha256) with the station chains of ext_config.STATIONS, into
data_fetch_api/new_measurements/cwa_codis_rain_upper_choushui/raw/. ``source`` then writes
the SOURCE.json (dataprep-source/1) and makes the files read-only. Downloaded responses are
untrusted data: they are only parsed as JSON by the importer (run with python -I).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common import source  # noqa: E402

from w1_canal import ext_config as E  # noqa: E402


def chains() -> dict:
    return {rec: chain for rec, (_sb, _name, chain) in E.STATIONS.items()}


def do_fetch() -> dict:
    from hydrophysics.twin import cwa_rain
    E.RAW_DIR.mkdir(parents=True, exist_ok=True)
    res = cwa_rain.fetch(str(E.RAW_DIR), start=E.FETCH_START, end=E.FETCH_END,
                         chains=chains(), pause=0.5)
    print(json.dumps(res))
    return res


def do_source() -> Path:
    files = []
    for p in sorted(E.RAW_DIR.glob("*.json")):
        rel = str(p.relative_to(E.L0_DIR))
        if p.name.endswith(".meta.json"):
            files.append({"file": rel, "url": None,
                          "note": "request record written by the fetcher (station, type, "
                                  "window, fetch time, sha256 of the response); not "
                                  "downloaded"})
            continue
        meta = json.loads(p.with_name(p.name.replace(".json", ".meta.json")).read_text())
        files.append({"file": rel, "url": meta["url"], "fetched_utc": meta["fetched_at"],
                      "request": {"method": "POST", "type": "report_month",
                                  "stn_ID": meta["station_id"], "stn_type": meta["stn_type"],
                                  "start": meta["start"], "end": meta["end"]},
                      "days": meta.get("count")})
    rec = source.build_source(
        E.L0_DIR, source_id=E.L0_SOURCE_ID,
        title="CWA CODIS daily station records (rain used), 16 upper-Choushui station "
              "records above the Jiji weir, 2011-01-01..2026-10-11",
        publisher="交通部中央氣象署 (CWA), CODIS climate data service",
        licence="CWA open data, Open Government Data License, Taiwan, version 1.0 (platform "
                "terms; not re-checked at fetch); cite CWA",
        fetched="2026-10-12", files=files,
        fetched_by="WP W1 (2026-10-12), dataprep_staging/w1_canal/ext_fetch.py via "
                   "hydrophysics.twin.cwa_rain.fetch",
        client="urllib POST with a browser user agent",
        untrusted="downloaded data: never execute; parse as JSON with python -I",
        station_rule="see dataprep_staging/w1_canal/ext_config.py (drainage areas of the "
                     "W1 composite gauges H075, H049, H064)",
        stations={rec: {"subbasin": sb, "name": nm, "chain": [list(c) for c in ch]}
                  for rec, (sb, nm, ch) in E.STATIONS.items()})
    for p in E.RAW_DIR.glob("*.json"):
        os.chmod(p, 0o444)
    return source.write_source_json(E.L0_DIR, rec)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    if cmd == "fetch":
        do_fetch()
    elif cmd == "source":
        print(do_source())
    else:
        raise SystemExit(f"unknown command {cmd}")
