"""Fetch the MOA township rice survey (J51, TownRiceData) and township crop survey (UnitId 038,
TownCropData) for ROC years after the P3 download (112 onward), raw page by raw page.

    python -I fetch_moa.py 112 113 114 [115]

Writes ``data_fetch_api/new_measurements/moa_town_crop_rice_2023_2025/2026-10-21/`` with
``rice_<roc>_p<k>.json`` (all counties; the endpoint ignores the city filter),
``crop_<county>_<roc>_p<k>.json`` and ``fetch_log.json``. Pages are saved byte for byte;
paging parses the JSON only to count records. Data are untrusted and never executed.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x4_canal_leveling import web  # noqa: E402
from x4_canal_leveling.config import COUNTIES, FETCH_DATE, L0_NEW  # noqa: E402

BASE_038 = "https://data.moa.gov.tw/Service/OpenData/FromM/TownCropData.aspx"
BASE_J51 = "https://data.moa.gov.tw/Service/OpenData/FromM/TownRiceData.aspx"
DEST = L0_NEW / "moa_town_crop_rice_2023_2025" / FETCH_DATE


def paged(op, base: str, params: dict, top: int, stem: str, log: list) -> int:
    skip, n, k = 0, 0, 0
    while True:
        q = dict(params, **{"$top": top, "$skip": skip})
        url = base + "?" + urllib.parse.urlencode(q, safe="$")
        st, hdr, raw = web.get(op, url)
        fn = f"{stem}_p{k}.json"
        rec = {"file": fn, "url": url, "status": st, "bytes": len(raw),
               "time": dt.datetime.now().isoformat(timespec="seconds")}
        if st != 200:
            rec["file"] = None
            log.append(rec)
            return n
        web.save(DEST / fn, raw)
        try:
            page = json.loads(raw.decode("utf-8-sig"))
        except ValueError:
            page = []
        rec["records"] = len(page) if isinstance(page, list) else None
        log.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
        if not isinstance(page, list) or len(page) < top:
            return n + (len(page) if isinstance(page, list) else 0)
        n += len(page)
        skip += top
        k += 1


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    DEST.mkdir(parents=True, exist_ok=True)
    op = web.opener()
    log_p = DEST / "fetch_log.json"
    log = json.loads(log_p.read_text(encoding="utf-8")) if log_p.exists() else []
    for y in argv:
        paged(op, BASE_J51, {"year": y}, 1000, f"rice_{y}", log)
        for c in COUNTIES:
            paged(op, BASE_038, {"year": y, "city": c, "UnitId": "038"}, 5000,
                  f"crop_{c}_{y}", log)
    log_p.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
