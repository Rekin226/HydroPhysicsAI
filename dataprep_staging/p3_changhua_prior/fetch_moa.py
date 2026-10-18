"""WP P3 step 1: fetch the MOA township crop survey (UnitId 038, TownCropData) and the township
rice survey (J51, TownRiceData) for the pre-origin years 2012-2022 (ROC 101-111).

Raw response pages are stored byte for byte (one file per request, untrusted data, never
executed); paging only parses the JSON to count records. The J51 endpoint ignores the city
filter (plan P3 step 1), so its pages hold every county and are filtered at read time.

Usage (run with python -I, output root passed as an argument):
    python -I fetch_moa.py <data_fetch_api/new_measurements> 101 102 ... 111
Writes <root>/moa_crop_038_p3/ and <root>/moa_rice_J51_p3/ with a fetch log
(fetch_log.json); SOURCE.json is written separately by write_sources.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request

BASE_038 = "https://data.moa.gov.tw/Service/OpenData/FromM/TownCropData.aspx"
BASE_J51 = "https://data.moa.gov.tw/Service/OpenData/FromM/TownRiceData.aspx"
CITIES = ("彰化縣", "雲林縣")


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "HydroPhysicsAI-dataprep-P3"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read()
        except Exception as exc:  # network hiccup: retry with backoff
            if attempt == 3:
                raise
            print("retry", url, exc, flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def fetch_paged(base: str, params: dict, top: int, outdir: str, stem: str, log: list) -> int:
    skip, n, k = 0, 0, 0
    while True:
        q = dict(params, **{"$top": top, "$skip": skip})
        url = base + "?" + urllib.parse.urlencode(q, safe="$")
        raw = get(url)
        fn = f"{stem}_p{k}.json"
        path = os.path.join(outdir, fn)
        if os.path.exists(path):
            raise FileExistsError(path)
        with open(path, "xb") as f:
            f.write(raw)
        page = json.loads(raw.decode("utf-8"))
        log.append({"file": fn, "url": url, "fetched_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "records": len(page)})
        n += len(page)
        print(fn, len(page), flush=True)
        if len(page) < top:
            return n
        skip += top
        k += 1
        time.sleep(1)


def main() -> None:
    root = sys.argv[1]
    years = sys.argv[2:]
    d38 = os.path.join(root, "moa_crop_038_p3")
    dj = os.path.join(root, "moa_rice_J51_p3")
    os.makedirs(d38, exist_ok=True)
    os.makedirs(dj, exist_ok=True)
    log38: list = []
    logj: list = []
    for year in years:
        for city in CITIES:
            fetch_paged(BASE_038, {"year": year, "city": city, "UnitId": "038"}, 5000, d38,
                        f"crop_{city}_{year}", log38)
            time.sleep(1)
        fetch_paged(BASE_J51, {"year": year}, 1000, dj, f"rice_{year}", logj)
        time.sleep(1)
    for d, lg in ((d38, log38), (dj, logj)):
        with open(os.path.join(d, "fetch_log.json"), "x", encoding="utf-8") as f:
            json.dump(lg, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
