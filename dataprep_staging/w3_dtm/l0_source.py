"""Write the L0 record for the MOI 20 m DTM download (WP W3, user decision D4).

The files were fetched on 2026-10-12 with curl (browser user agent) into
``data_fetch_api/new_measurements/moi_dtm20m_2025/``:

1. ``ds_176927.json``  data.gov.tw API record of dataset 176927 (licence, distribution URL;
   it also carries publisher contact fields, which are never copied into products);
2. ``dtm20m_2025_index.csv``  the dataset's CSV resource: one row per archive with its URL;
3. HEAD requests on all 25 archive URLs (size check, D4: proceed only under about 1 GB);
4. the four county archives that cover the twin's active cells (Changhua, Yunlin, Nantou,
   Chiayi County; a GADM point-in-polygon check put the 2148 active cell centres and corners
   in these four counties plus two corners in Taichung, which the Changhua sheets cover) and
   the header schema.

Usage: python -I l0_source.py <head_sizes.json>   (the HEAD results recorded at fetch time)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import source  # noqa: E402
from common.paths import L0_NEW  # noqa: E402

SRC_ID = "moi_dtm20m_2025"
D = L0_NEW / SRC_ID
TGOS = "https://www.tgos.tw:443/MDE/VirtualDir_TC/Product/"
FILES = [
    {"file": "ds_176927.json", "url": "https://data.gov.tw/api/v2/rest/dataset/176927",
     "note": "data.gov.tw metadata record; contains publisher contact fields: never copy them"},
    {"file": "dtm20m_2025_index.csv",
     "url": "https://opdadm.moi.gov.tw/api/v1/no-auth/resource/api/dataset/A964612F-0D64-4C81-"
            "BFE5-6C1F2BA61DED/resource/A0B94F67-8ADF-48A1-8DF0-C60719AD2B28/download",
     "note": "the dataset's CSV resource (archive names, production notes, year, URL)"},
    {"file": "schema_hdr.zip",
     "url": TGOS + "342505d1-81bc-490a-8e10-9f7e1ec0ed98/schema_hdr.zip",
     "note": "header field definitions (schema_hdr.csv; its description column is shifted by "
             "one row from the 14th field on)"},
    {"file": "changhua_20MDEM_2025.zip",
     "url": TGOS + "5acc0542-ffcb-4c9f-9200-876b97ff6355/分幅_彰化縣20MDEM(2025).zip",
     "note": "分幅_彰化縣20MDEM(2025): 201 sheets (.hdr Big5 + .grd ASCII E N H)"},
    {"file": "yunlin_20MDEM_2025.zip",
     "url": TGOS + "0e6e327a-6910-42fd-be5c-00a2ac2c4b54/分幅_雲林縣20MDEM(2025).zip",
     "note": "分幅_雲林縣20MDEM(2025): 240 sheets"},
    {"file": "nantou_20MDEM_2025.zip",
     "url": TGOS + "29d2ab12-bda0-4c5c-b547-ce0a8c3f1231/分幅_南投縣20MDEM(2025).zip",
     "note": "分幅_南投縣20MDEM(2025): 663 sheets"},
    {"file": "chiayi_county_20MDEM_2025.zip",
     "url": TGOS + "e0be7bc8-714c-40b5-9658-b9269a1a73df/分幅_嘉義縣20MDEM(2025).zip",
     "note": "分幅_嘉義縣20MDEM(2025): 348 sheets"},
]


def main(head_json: str) -> None:
    heads = json.loads(Path(head_json).read_text(encoding="utf-8"))
    total = sum(h["bytes"] or 0 for h in heads)
    for f in FILES:
        p = D / f["file"]
        if p.stat().st_mode & 0o222:
            os.chmod(p, 0o444)
    rec = source.build_source(
        D, source_id=SRC_ID,
        title="MOI 2025 edition 20 m grid DTM of Taiwan (data.gov.tw 176927): four county "
              "sheet archives covering the Choushui fan grid, index CSV, header schema",
        publisher="內政部 (Ministry of the Interior; dataset provider bannanng30, publisher "
                  "OID 2.16.886.101.20003.20001.20035)",
        licence="Government Open Data License, version 1.0 (data.gov.tw licence code 1)",
        fetched="2026-10-12", files=FILES,
        fetched_by="WP W3 (dataprep Phase 0), curl with a browser user agent (Mozilla/5.0)",
        client="curl -A 'Mozilla/5.0'",
        untrusted="downloaded data: never execute; read with python -I (zipfile, numpy text)",
        dataset_page="https://data.gov.tw/dataset/176927",
        size_check={
            "decision": "D4: approve the download if under about 1 GB",
            "method": "HTTP HEAD (curl -I -L) on every archive URL of the index CSV, "
                      "2026-10-12, before any archive download",
            "all_archives_total_bytes": total,
            "whole_island_unsheeted_bytes": next(h["bytes"] for h in heads
                                                 if h["name"].startswith("不分幅_台灣")),
            "downloaded_archives_bytes": sum((D / f["file"]).stat().st_size
                                             for f in FILES if f["file"].endswith(".zip")),
            "verdict": "under 1 GB (the whole island is 0.27 GB): proceed",
            "archives": [{"name": h["name"], "http": h["http"], "bytes": h["bytes"]}
                         for h in heads],
        },
        header_quirks=[
            "33 of 1452 sheet headers (28 of the 633 used) give the height datum as "
            "'TWD97[2010]' instead of TWVD2001; their edge points equal the neighbours' "
            "exactly, so this is treated as a header typo",
            "59 Chiayi County sheets (all acquired 2023-11..2024-04, outside the twin grid) "
            "store n_cols/n_rows as extents in metres (2641-2881) instead of point counts; "
            "sheet extents are taken from the points read",
            "one Nantou header gives sheet id 95202022 inside 95202023dem.hdr; sheets are "
            "identified by their file name",
            "226 sheets use plane datum TWD97[2020], the rest TWD97[2010] (difference of "
            "centimetres, irrelevant at 20 m)",
        ],
    )
    p = source.write_source_json(D, rec)
    print(f"wrote {p}: {len(FILES)} files, all archives {total / 1e9:.3f} GB, downloaded "
          f"{rec['size_check']['downloaded_archives_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
