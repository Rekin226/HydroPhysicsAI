"""Archive the open-data deep-pile dumps (data.gov.tw 58339) into L0 with a SOURCE.json.

An earlier session (2026-10-11) pulled the WRA open-data deep-benchmark API twice into its
scratchpad: once with a plain client (10:35, the API's default order, capped at 100,000 rows,
which returned only 土庫國中(1) and (2) from 2016) and once through headless Chromium because
the host had started answering plain clients with a script challenge (16:13, sorted by
observation time descending, again capped at 100,000 rows: all nine sensors from 2023-09-21 to
2025-04-16). These dumps are the only copy of the agency's raw hourly feed the project has, and
the scratchpad is temporary, so WP S2 copies them byte for byte into
``data_fetch_api/new_measurements/wra_opendata_dbm_58339/`` (read-only) and records their
provenance. The files are untrusted data: this script only hashes and copies them.

Usage: python -I archive_opendata.py <dump_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import source  # noqa: E402
from common.paths import L0_NEW  # noqa: E402

DEST = L0_NEW / "wra_opendata_dbm_58339"
API = "https://opendata.wra.gov.tw/api/v2/34551a0e-170d-4514-9385-e4080b616eca"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    src = Path(argv[0])
    files = [
        {"file": "res58339_default_order.json",
         "src": "res58339.json",
         "url": f"{API}?format=JSON",
         "note": "query string reconstructed: the fetching session kept no URL log. Content is "
                 "the first 100,000 rows in the API's default order: BM_TKJS(1) 2016-06-23.."
                 "2025-02-24 (74,846 rows) and BM_TKJS(2) 2016-07-13..2025-02-24 (25,154 rows). "
                 "File mtime 2026-10-11 10:35:46 +0800.",
         "fetch_method": "plain HTTP client"},
        {"file": "res58339_latest_desc.json",
         "src": "res58339_latest.json",
         "url": f"{API}?sort=observationtime%20desc&format=JSON",
         "note": "the 100,000 most recent rows: nine sensors 2023-09-21..2025-04-16 (the feed "
                 "stops on 2025-04-16). File mtime 2026-10-11 16:13:29 +0800.",
         "fetch_method": "headless Chromium --dump-dom (the host answers plain clients with a "
                         "TSPD script challenge), JSON body extracted from the DOM"},
    ]
    for f in files:
        source.copy_into_l0(src / f["src"], DEST / f["file"])
    rec = source.build_source(
        DEST,
        source_id="wra_opendata_dbm_58339",
        title="WRA open data: deep benchmark (深層樁) hourly compression readings "
              "(data.gov.tw dataset 58339)",
        publisher="經濟部水利署 (Water Resources Agency), via opendata.wra.gov.tw",
        licence="Open Government Data License, version 1.0 (政府資料開放授權條款－第1版; the "
                "data.gov.tw default, attribution required; not re-verified on the dataset "
                "page on 2026-10-12)",
        fetched="2026-10-11",
        files=[{k: v for k, v in f.items() if k != "src"} for f in files],
        provenance="Fetched on 2026-10-11 by an earlier Claude Code session of this project into "
                   "its temporary scratchpad (probe/); copied byte for byte into L0 by WP S2 on "
                   "2026-10-12 (dataprep_staging/s2_deep_benchmarks/archive_opendata.py).",
        caveats=[
            "Both responses are capped at 100,000 rows; no paging or filter parameter was found.",
            "Field 'depth' is the reading in mm (rising = compression) despite its name.",
            "Raw agency feed: sensor steps and resets are present (2024-02-29/03-01 at every "
            "pile, 溪州 2024-03-05 and 2025-03-06, 灣內 2024-03 and a sign inversion after "
            "2024-08-08); see results/dataprep/deep_benchmarks/ for the step catalogue.",
            "No contact or personal fields: deepbenchmarkchinesename, depth, identifier, "
            "observationtime, countycode.",
        ],
        copied_into_l0="2026-10-12",
    )
    p = source.write_source_json(DEST, rec)
    print(f"wrote {p}")
    for f in rec["files"]:
        print(f["file"], f["bytes"], f["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
