"""Write SOURCE.json (dataprep-source/1) for each X2 L0 directory from the fetch log.

Every file the fetcher saved is listed with its URL, request method, UTC time and note; the
record's licence and terms are fixed here per source. A directory gets its record once
(``common.source.write_source_json`` refuses to change an existing one).

Usage: python -I finalize_sources.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import source  # noqa: E402

from x2_public_supply import x2const as K  # noqa: E402

META = {
    "twc_statistics": {
        "title": "台灣自來水事業統計年報 (Taiwan Water Corporation statistical yearbooks) 96-114 "
                 "(data years 2007-2025) and the yearbook list pages",
        "publisher": "台灣自來水股份有限公司 (Taiwan Water Corporation), 會計處",
        "licence": "Public download on www.water.gov.tw (台灣自來水事業統計年報, nodeId=4571); "
                   "page metadata DC.Rights = 臺灣自來水股份有限公司; no open-data licence is "
                   "stated on the page: cite the publisher, keep the PDFs out of git and do not "
                   "redistribute; derived numbers with attribution",
        "terms_note": "robots.txt (2026-10-21) has two 'User-agent: *' groups, 'Allow: /' "
                      "and 'Disallow: /'; merged per RFC 9309 2.2.2 the equivalent Allow wins, "
                      "so the fetch was treated as allowed (Python's urllib parser, which drops "
                      "the BOM-prefixed first group, reads it as disallowed). Flagged for the "
                      "user. 24 requests at >= 1.5 s intervals; no login, token or CAPTCHA.",
    },
    "wra_water_use_annual": {
        "title": "經濟部水利署 各項用水統計資料庫 (web.wra.gov.tw/Annualwater): annual report "
                 "listings ROC 100-113 and four reports (113 生活/各標的/各區域, 105 生活)",
        "publisher": "經濟部水利署 (maintained by 逢甲大學海峽兩岸科技研究中心)",
        "licence": "WRA website publication; the site footer reads '經濟部水利署，轉載必究' "
                   "(reproduction reserved): cite, do not redistribute",
        "terms_note": "no robots.txt on web.wra.gov.tw (HTTP 404 page); listing pages fetched "
                      "by the public form's year postback; >= 1.5 s between requests. These "
                      "reports split water use by county and use, not tap water by source: no "
                      "groundwater number for public supply was found in them.",
    },
    "datagovtw_twc_metadata": {
        "title": "data.gov.tw metadata of Taiwan Water datasets 152911 (monthly supply by major "
                 "system), 86105 (annual; the API returned no record) and 25677 (public-supply "
                 "information), and the 152911 resource syst8.csv (rolling: one month, 2026-08)",
        "publisher": "台灣自來水股份有限公司 via data.gov.tw (國家發展委員會)",
        "licence": "Open Government Data License, version 1.0 (data.gov.tw licence code 1)",
        "terms_note": "data.gov.tw /robots.txt returned an HTTP 500 HTML page (no rules); "
                      "water.gov.tw robots.txt as in twc_statistics. These datasets give "
                      "supply volumes by system, not groundwater by source; syst8.csv holds "
                      "only the latest month (rolling file).",
    },
}


def main() -> None:
    log = [json.loads(ln) for ln in K.FETCH_LOG.read_text(encoding="utf-8").splitlines()
           if ln.strip()]
    for key, meta in META.items():
        d = K.L0[key]
        if not d.exists():
            continue
        files = []
        for r in log:
            if r.get("source_key") != key or "dest" not in r or "sha256" not in r:
                continue
            files.append({"file": r["file"], "url": r["url"], "method": r["method"],
                          "fetched_utc": r["utc"], "note": r.get("note", ""),
                          "content_type": r.get("content_type", ""),
                          **({"post_fields": r["post_fields"]} if "post_fields" in r else {}),
                          **({"last_modified": r["last_modified"]}
                             if r.get("last_modified") else {})})
        on_disk = sorted(p.name for p in d.iterdir() if p.name != "SOURCE.json")
        if sorted(f["file"] for f in files) != on_disk:
            raise SystemExit(f"{d}: log and directory disagree")
        rec = source.build_source(
            d, source_id=key, title=meta["title"], publisher=meta["publisher"],
            licence=meta["licence"], fetched=K.FETCH_DATE, files=files,
            fetched_by="WP X2 (dataprep_staging/x2_public_supply/fetch.py)",
            client=f"python urllib, user agent '{K.USER_AGENT}'",
            terms=meta["terms_note"],
            untrusted="downloaded data: never execute; read with poppler / python -I")
        p = source.write_source_json(d, rec)
        print("written", p, len(files), "files")


if __name__ == "__main__":
    main()
