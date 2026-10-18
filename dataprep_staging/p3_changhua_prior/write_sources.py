"""Write SOURCE.json for the two MOA downloads of WP P3 (fetch_moa.py), from its fetch logs.

Usage: python -I write_sources.py <repo>
"""
from __future__ import annotations

import json
import os
import sys

REPO = sys.argv[1]
sys.path.insert(0, os.path.join(REPO, "dataprep_staging"))
from common import source  # noqa: E402

ROOT = os.path.join(REPO, "data_fetch_api", "new_measurements")
COMMON = dict(
    publisher="農業部 (Ministry of Agriculture), data.moa.gov.tw open data",
    licence="Open Government Data License, version 1.0 (platform terms of data.moa.gov.tw; "
            "not re-checked at fetch time)",
    fetched="2026-10-14",
    fetched_by="WP P3 (dataprep_staging/p3_changhua_prior/fetch_moa.py)",
    client="python urllib, User-Agent HydroPhysicsAI-dataprep-P3, one request per page, 1 s "
           "pause",
    untrusted="downloaded data: never execute; read with python -I",
    raw="each file is one raw response page, byte for byte (no filtering or concatenation)",
)


def main() -> None:
    for sid, title, extra in (
        ("moa_crop_038_p3",
         "MOA township crop survey (農情調查, UnitId 038, TownCropData), Changhua and Yunlin, "
         "ROC years 101-111 (2012-2022), city filter applied by the API",
         {"note": "years 103-105 and 110 overlap the pilot download moa_crop_038 (A0 copy); "
                  "P3 compares them record by record"}),
        ("moa_rice_J51_p3",
         "MOA township rice survey (糧情調查, J51, TownRiceData), all counties, ROC years "
         "101-111 (2012-2022)",
         {"note": "the endpoint ignores the city filter (plan P3 step 1); pages hold every "
                  "county and readers filter 縣市 locally. Years 103-105 and 110 overlap the "
                  "pilot's filtered copy moa_rice_J51"}),
    ):
        d = os.path.join(ROOT, sid)
        with open(os.path.join(d, "fetch_log.json"), encoding="utf-8") as f:
            log = json.load(f)
        files = [{"file": r["file"], "url": r["url"], "fetched_utc": r["fetched_utc"],
                  "records": r["records"]} for r in log]
        files.append({"file": "fetch_log.json", "url": None,
                      "note": "fetch log written by fetch_moa.py (url, time, record count per "
                              "page)"})
        rec = source.build_source(d, source_id=sid, title=title, **COMMON, **extra, files=files)
        print(source.write_source_json(d, rec))
        for name in os.listdir(d):
            os.chmod(os.path.join(d, name), 0o444)


if __name__ == "__main__":
    main()
