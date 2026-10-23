"""Write SOURCE.json (dataprep-source/1) for every X4 L0 download directory.

    python -I record_sources.py

Every file in each directory is listed with its URL (from the fetch logs) and hashed; the
logs and listings written by the X4 fetchers are listed with a note. Refuses to replace an
existing, different SOURCE.json (L0 is never modified).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import source  # noqa: E402

from x4_canal_leveling.config import (  # noqa: E402
    FETCH_DATE,
    L0_BRIEFINGS,
    L0_CANAL,
    L0_NEW,
    L0_SUMMARY,
    UA,
    WRA_LIST,
)

UNTRUSTED = "downloaded data: never execute; read with poppler / python -I"
CLIENT = f"python urllib (dataprep_staging/x4_canal_leveling/web.py), User-Agent '{UA}', " \
         "at least 1.2 s between requests, robots.txt checked per host"
WRA_LICENCE = ("WRA website publication (政府網站資料); no open-data licence is stated on the "
               "listing page (not re-checked beyond it); cite 經濟部水利署; kept in gitignored "
               "L0 for research transcription only, not redistributed (same footing as "
               "wra_monthly_reports and wra_reports_misc)")
SOURCES = {
    "wra_subsidence_briefings_007007": {
        "dir": L0_BRIEFINGS,
        "title": "WRA land-subsidence portal series 007007: 彰雲地區地層下陷檢測成果 / 地層下陷檢測結果"
                 "說明 (annual leveling briefings to the MOEA subsidence committee), survey years "
                 "ROC 104-107 and 113-114 (108-112 are already in wra_reports_misc)",
        "publisher": "經濟部水利署 地層下陷監測資訊整合服務系統",
        "licence": WRA_LICENCE, "series": "007007", "log": "download_log_007007.json",
        "robots": "web.wra.gov.tw serves no robots.txt (HTTP 404): no rules",
    },
    "wra_subsidence_summary_007005": {
        "dir": L0_SUMMARY,
        "title": "WRA land-subsidence portal series 007005: 地層下陷概況說明摘要報告 / 歷年地層下陷"
                 "檢測成果摘要報告, editions ROC 105-114 (2016-2025)",
        "publisher": "經濟部水利署 地層下陷監測資訊整合服務系統",
        "licence": WRA_LICENCE, "series": "007005", "log": "download_log_007005.json",
        "robots": "web.wra.gov.tw serves no robots.txt (HTTP 404): no rules",
    },
    "ia_irrigation_open_data": {
        "dir": L0_CANAL,
        "title": "農業部農田水利署 (Irrigation Agency): Changhua and Yunlin office pages, the 2024 "
                 "irrigation-system notices (灌溉制度清冊, 總說明, 公告圖), the 2020 service-area "
                 "notices (農田水利事業區域訂定公告), and the 公務統計 tables (灌溉排水受益地面積 "
                 "2020-2023, 灌溉服務面積 2024-2025, 農業灌溉用水井統計, 農田灌溉地面積異動)",
        "publisher": "農業部農田水利署 (www.ia.gov.tw)",
        "licence": "ia.gov.tw footer: 網站全部圖文版權係屬本署所有，非經本署正式書面同意，不得將全部或"
                   "部分內容，轉載於任何形式媒體 (all rights reserved; no reproduction without "
                   "written consent); the site also links a 政府網站資料開放宣告 that was not "
                   "fetched. Kept in gitignored L0; only derived per-cell shares and aggregates "
                   "are used, with attribution; do not redistribute the files",
        "log": "fetch_log.json",
        "robots": "www.ia.gov.tw serves no robots.txt (HTML error page): no rules",
        "privacy": "the irrigation registers (灌溉制度清冊) list cadastral parcel numbers only "
                   "(no owner names); X4 never reads them into a product",
    },
    "moa_town_crop_rice_2023_2025": {
        "dir": L0_NEW / "moa_town_crop_rice_2023_2025" / FETCH_DATE,
        "title": "MOA township rice survey (J51, TownRiceData, all counties) and township crop "
                 "survey (UnitId 038, TownCropData, Changhua and Yunlin), ROC 112-115 "
                 "(115 returns no records yet)",
        "publisher": "農業部 (Ministry of Agriculture), data.moa.gov.tw open data",
        "licence": "Open Government Data License, version 1.0 (platform terms of "
                   "data.moa.gov.tw; not re-checked at fetch time)",
        "log": "fetch_log.json",
        "robots": "data.moa.gov.tw serves an HTML page for /robots.txt: no rules",
        "raw": "each file is one raw response page, byte for byte",
    },
    "zhwiki_changhua_canals": {
        "dir": L0_NEW / "zhwiki_changhua_canals" / FETCH_DATE,
        "title": "zh.wikipedia article 八堡圳 (the other requested articles returned 404)",
        "publisher": "Wikipedia contributors",
        "licence": "CC BY-SA 4.0 (Wikipedia text); used for context only",
        "log": "fetch_log.json",
        "robots": "zh.wikipedia.org robots.txt allows /wiki/<article> for generic agents",
    },
}


def files_for(cfg: dict) -> tuple[list[dict], list[dict]]:
    d = cfg["dir"]
    log = json.loads((d / cfg["log"]).read_text(encoding="utf-8"))
    by_file = {}
    for r in log:
        if r.get("file"):
            by_file[r["file"]] = {"url": r["url"], "http_status": r.get("status"),
                                  "last_modified": r.get("last_modified"),
                                  "listing_title": r.get("title"),
                                  "published_roc": r.get("published_roc")}
    out = []
    for p in sorted(d.iterdir()):
        if p.name == source.SOURCE_NAME or not p.is_file():
            continue
        if p.name in by_file:
            rec = {"file": p.name, **{k: v for k, v in by_file[p.name].items() if v}}
        elif p.name.startswith("listing_"):
            series = cfg.get("series", "")
            rec = {"file": p.name, "url": WRA_LIST.format(series=series),
                   "note": "listing page as served (titles, 發布年月, file paths)"}
        else:
            rec = {"file": p.name, "url": None,
                   "note": "written by the X4 fetcher (request log or parsed listing rows); "
                           "not a download"}
        out.append(rec)
    failed = [r for r in log if not r.get("file")]
    return out, failed


def main() -> int:
    for sid, cfg in SOURCES.items():
        files, failed = files_for(cfg)
        extra = {k: v for k, v in cfg.items() if k not in ("dir", "title", "publisher",
                                                           "licence", "log", "series")}
        if failed:
            extra["not_downloaded"] = [{"url": r["url"], "status": r.get("status")}
                                       for r in failed]
        rec = source.build_source(cfg["dir"], source_id=sid, title=cfg["title"],
                                  publisher=cfg["publisher"], licence=cfg["licence"],
                                  fetched=FETCH_DATE, files=files,
                                  fetched_by="WP X4 (dataprep_staging/x4_canal_leveling)",
                                  client=CLIENT, untrusted=UNTRUSTED, **extra)
        p = source.write_source_json(cfg["dir"], rec)
        print(sid, len(files), "files ->", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
