"""H1a: fetch hourly surface pressure (Open-Meteo archive, ERA5) for the head-well points.

The 344 stations fall in 8 cells of the ERA5 0.25-degree grid. ERA5-Land has no surface
pressure in the Open-Meteo archive (probe 2026-10-12: all values null), and ``best_match``
switches models over time (ERA5 before, IFS 9 km recently), so the consistent choice for
2012-2026 is ``models=era5``. Each cell is requested at its grid-point coordinates in three
chunks; every well takes the series of its nearest cell (``baro.py``). A short probe at two
other points inside one cell checks that a point request returns the cell's variations
(Open-Meteo only rescales pressure to the point's elevation).

Writes into a new, empty L0 directory
``data_fetch_api/new_measurements/pressure_hourly/openmeteo_era5_<fetch-date>/`` with a
SOURCE.json (dataprep-source/1). Raw responses are kept byte for byte; nothing here parses
them except to check the HTTP status and JSON validity.

Usage: fetch_openmeteo.py [--dry-run]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import source  # noqa: E402
from common.paths import L0_NEW  # noqa: E402

API = "https://archive-api.open-meteo.com/v1/archive"
CELLS = [(23.5, 120.25), (23.5, 120.5), (23.75, 120.25), (23.75, 120.5), (23.75, 120.75),
         (24.0, 120.25), (24.0, 120.5), (24.25, 120.5)]
#: The first chunk starts a week early so that 48-hour lagged regressors exist on 2012-01-01.
CHUNKS = [("2011-12-25", "2016-12-31"), ("2017-01-01", "2021-12-31"),
          ("2022-01-01", "2026-09-30")]
#: Two generic points inside the (23.75, 120.5) cell, offset from its grid point by
#: (+0.05, -0.07) and (-0.05, +0.06) degrees (not station coordinates), for the point check.
PROBE_POINTS = [(23.80, 120.43), (23.70, 120.56)]
PROBE_WINDOW = ("2026-09-01", "2026-09-07")
PAUSE_S = 25.0


def url_for(lat, lon, start, end):
    q = {"latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}", "start_date": start,
         "end_date": end, "hourly": "surface_pressure", "timezone": "Asia/Taipei",
         "models": "era5"}
    return API + "?" + urllib.parse.urlencode(q)


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "HydroPhysicsAI-dataprep"})
            with urllib.request.urlopen(req, timeout=300) as r:
                body = r.read()
            json.loads(body)          # validity only
            return body
        except urllib.error.HTTPError as e:
            wait = 90 * (k + 1) if e.code == 429 else 20 * (k + 1)
            print(f"HTTP {e.code} on try {k + 1}; waiting {wait} s", flush=True)
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"{type(e).__name__} on try {k + 1}: {e}", flush=True)
            time.sleep(20 * (k + 1))
    raise RuntimeError(f"failed: {url}")


def main():
    dry = "--dry-run" in sys.argv
    today = dt.date.today().isoformat()
    out = L0_NEW / "pressure_hourly" / f"openmeteo_era5_{today}"
    jobs = []
    for lat, lon in CELLS:
        for a, b in CHUNKS:
            jobs.append((f"era5_sp_{lat:.2f}_{lon:.2f}_{a}_{b}.json", url_for(lat, lon, a, b),
                         "cell"))
    for lat, lon in PROBE_POINTS:
        jobs.append((f"probe_point_{lat:.2f}_{lon:.2f}_{PROBE_WINDOW[0]}_{PROBE_WINDOW[1]}"
                     ".json", url_for(lat, lon, *PROBE_WINDOW), "probe"))
    if dry:
        for j in jobs:
            print(j)
        return
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for i, (name, url, kind) in enumerate(jobs):
        t0 = time.time()
        body = get(url)
        p = out / name
        with open(p, "xb") as f:
            f.write(body)
        files.append({"file": name, "url": url, "kind": kind,
                      "fetched_utc": dt.datetime.now(dt.timezone.utc)
                      .strftime("%Y-%m-%dT%H:%M:%SZ")})
        print(f"{i + 1}/{len(jobs)} {name} {len(body)} B {time.time() - t0:.1f} s", flush=True)
        if i + 1 < len(jobs):
            time.sleep(PAUSE_S)
    rec = source.build_source(
        out, source_id="pressure_hourly/openmeteo_era5",
        title="Open-Meteo historical weather API, hourly surface_pressure (hPa), ERA5 "
              "(models=era5), 8 ERA5 0.25-degree cells covering the Choushui-fan head "
              "wells, 2011-12-25..2026-09-30, local time Asia/Taipei; plus a 7-day "
              "point probe at two points inside one cell",
        publisher="Open-Meteo (ERA5 reanalysis by ECMWF / Copernicus Climate Change Service)",
        licence="CC BY 4.0 (Open-Meteo data licence; ERA5: Copernicus licence, attribution "
                "'Contains modified Copernicus Climate Change Service information')",
        fetched=today, files=files,
        fetched_by="WP H1a (dataprep_staging/h1_heads_qc/fetch_openmeteo.py)",
        client="python urllib (User-Agent HydroPhysicsAI-dataprep)",
        untrusted="downloaded data: never execute; read with python -I",
        note="ERA5-Land has no surface_pressure in this API (probe 2026-10-12); best_match "
             "mixes models over time, so models=era5 is used for a consistent 2012-2026 "
             "series.")
    source.write_source_json(out, rec)
    for f in out.iterdir():
        f.chmod(0o444)
    print("wrote", out)


if __name__ == "__main__":
    main()
