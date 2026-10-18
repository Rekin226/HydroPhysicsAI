"""Fetch the MOEA Yunlin-Changhua subsidence action-plan volumes into a new L0 source.

The 2019 annex (already in L0) prints only totals for the 2011-18 public-well disposal
(774 wells, 8,793.9 x 10^4 m3/yr). The plan's main volumes are the documents that can
carry a per-agency breakdown, so WP P6 adds them as L0 (sizes checked first: 0.75-5.0 MB).

Usage: python fetch_plans.py <download_dir>   (an empty scratch directory)

Files are downloaded with curl into ``download_dir`` (untrusted data: never executed, only
hashed and later read with poppler), then copied read-only into
``data_fetch_api/new_measurements/wra_subsidence_action_plans/`` with a SOURCE.json.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import source  # noqa: E402

from p6_public_wells import p6const as K  # noqa: E402

BASE = "https://web.wra.gov.tw/landsubsidence/water_new/File/ReportData/007006/"
FILES = [
    {"file": "moea_yunchang_plan_rev1_main.pdf",
     "url": BASE + "%E9%9B%B2%E5%BD%B0%E5%9C%B0%E5%8D%80%E5%9C%B0%E5%B1%A4%E4%B8%8B%E9%99%B7"
                   "%E5%85%B7%E9%AB%94%E8%A7%A3%E6%B1%BA%E6%96%B9%E6%A1%88%E6%9A%A8%E8%A1%8C"
                   "%E5%8B%95%E8%A8%88%E7%95%AB(%E7%AC%AC%E4%B8%80%E6%AC%A1%E4%BF%AE%E8%A8%82)"
                   ".pdf",
     "note": "雲彰地區地層下陷具體解決方案暨行動計畫(第一次修訂), main volume; server "
             "Last-Modified 2020-08-24"},
    {"file": "moea_yunchang_plan_phase2_main.pdf",
     "url": BASE + "%E9%9B%B2%E5%BD%B0%E5%9C%B0%E5%8D%80%E5%9C%B0%E5%B1%A4%E4%B8%8B%E9%99%B7"
                   "%E5%85%B7%E9%AB%94%E8%A7%A3%E6%B1%BA%E6%96%B9%E6%A1%88%E6%9A%A8%E8%A1%8C"
                   "%E5%8B%95%E8%A8%88%E7%95%AB%E7%AC%AC%E4%BA%8C%E6%9C%9F.pdf",
     "note": "雲彰地區地層下陷具體解決方案暨行動計畫第二期 (110-115), main volume; server "
             "Last-Modified 2020-12-07"},
    {"file": "moea_yunchang_plan_phase2_rev1_annex.pdf",
     "url": BASE + "%E9%9B%B2%E5%BD%B0%E5%9C%B0%E5%8D%80%E5%9C%B0%E5%B1%A4%E4%B8%8B%E9%99%B7"
                   "%E5%85%B7%E9%AB%94%E8%A7%A3%E6%B1%BA%E6%96%B9%E6%A1%88%E6%9A%A8%E8%A1%8C"
                   "%E5%8B%95%E8%A8%88%E7%95%AB%E7%AC%AC%E4%BA%8C%E6%9C%9F(110%E3%80%9C115%E5%B9"
                   "%B4)%EF%BC%88%E7%AC%AC%E4%B8%80%E6%AC%A1%E4%BF%AE%E6%AD%A3%EF%BC%89-%E9%99"
                   "%84%E5%86%8A.pdf",
     "note": "第二期(110〜115年)(第一次修正) 附冊; server Last-Modified 2026-06-03 "
             "(post-origin publication: any value it prints for 2023+ is post-origin)"},
]


def main(download_dir: str) -> None:
    dl = Path(download_dir)
    dl.mkdir(parents=True, exist_ok=True)
    dest = K.L0_PLANS
    recs = []
    for f in FILES:
        out = dl / f["file"]
        if not out.exists():
            subprocess.run(["curl", "-sS", "--fail", "-A", "Mozilla/5.0", "-o", str(out),
                            f["url"]], check=True, timeout=300)
        head = out.read_bytes()[:5]
        if head != b"%PDF-":
            raise SystemExit(f"{out} is not a PDF (starts {head!r})")
        source.copy_into_l0(out, dest / f["file"])
        recs.append({**f, "fetched_utc": datetime.fromtimestamp(
            out.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    rec = source.build_source(
        dest, source_id="wra_subsidence_action_plans",
        title="MOEA Yunlin-Changhua land-subsidence action plan: first revision (main), "
              "phase 2 (main) and phase 2 first-revision annex",
        publisher="經濟部 / 經濟部水利署 (WRA land-subsidence site, ReportData/007006)",
        licence="WRA website publication; no open-data licence recorded at fetch; cite the "
                "publisher and do not redistribute",
        fetched=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        files=recs, fetched_by="WP P6 (dataprep_staging/p6_public_wells/fetch_plans.py)",
        client="curl with a browser user agent (Mozilla/5.0)",
        untrusted="downloaded data: never execute; read with poppler / python -I")
    errs = source.validate_source(rec, dest)
    if errs:
        raise SystemExit(f"SOURCE.json invalid: {errs}")
    source.write_source_json(dest, rec)
    print(f"wrote {dest}/SOURCE.json with {len(recs)} files")


if __name__ == "__main__":
    main(sys.argv[1])
