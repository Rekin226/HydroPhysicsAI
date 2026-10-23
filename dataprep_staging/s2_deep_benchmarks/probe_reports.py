"""Check the WRA monthly-report address for 地層下陷監測整合分析月報 newer than the local copies.

The integrated monthly reports sit at a predictable address
(``.../File/ReportData/007003/<ROC year>年<month>月 地層下陷監測整合分析月報.pdf``); the local
copies (``data_fetch_api/new_measurements/wra_monthly_reports/``) run from 2024-01 to 2026-01.
This script asks the server for the first bytes of every month from 2023-01 to the current
month, plus spelling variants for the months after 2026-01, and records the HTTP status,
Content-Length and Last-Modified. Nothing is downloaded beyond 1 kB per probe and nothing is
executed. A month that answers 200/206 with a PDF content type and is not in the local copy is
a new report to transcribe (``hydrophysics.twin.report_extract``).

Usage: python -I probe_reports.py <out.json>
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://web.wra.gov.tw/landsubsidence/water_new/File/ReportData/007003/"
LOCAL = Path(__file__).resolve().parents[2] / "data_fetch_api/new_measurements/wra_monthly_reports"


def names(roc: int, month: int, variants: bool) -> list[str]:
    out = [f"{roc}年{month}月 地層下陷監測整合分析月報.pdf"]
    if variants:
        out += [f"{roc}年{month}月地層下陷監測整合分析月報.pdf",
                f"{roc}年{month:02d}月 地層下陷監測整合分析月報.pdf",
                f"{roc}年{month:02d}月地層下陷監測整合分析月報.pdf",
                f"{roc}年{month}月　地層下陷監測整合分析月報.pdf"]
    return out


def probe(url: str) -> dict:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-1023",
                                               "User-Agent": "Mozilla/5.0 (dataprep S2 probe)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            head = r.read(1024)
            return {"status": r.status, "content_type": r.headers.get("Content-Type"),
                    "content_range": r.headers.get("Content-Range"),
                    "last_modified": r.headers.get("Last-Modified"),
                    "pdf_magic": head[:5] == b"%PDF-"}
    except urllib.error.HTTPError as e:
        return {"status": e.code, "content_type": e.headers.get("Content-Type") if e.headers
                else None, "pdf_magic": False}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"status": None, "error": f"{type(e).__name__}: {e}", "pdf_magic": False}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    today = dt.date.today()
    local = sorted(p.stem.replace("integrated_", "") for p in LOCAL.glob("integrated_*.pdf"))
    rows = []
    y, m = 2023, 1
    while (y, m) <= (today.year, today.month):
        ym = f"{y}-{m:02d}"
        variants = ym > "2026-01"
        for name in names(y - 1911, m, variants):
            url = BASE + urllib.parse.quote(name)
            r = probe(url)
            rows.append({"month": ym, "name": name, "url": url, "local_copy": ym in local, **r})
        m += 1
        if m == 13:
            y, m = y + 1, 1
    found = sorted({r["month"] for r in rows if r.get("pdf_magic")})
    new = [mo for mo in found if mo not in local]
    out = {"schema": "s2-report-probe/1", "probed_on": today.isoformat(),
           "base": BASE, "local_months": local, "months_available": found,
           "new_months_not_in_local_copy": new, "probes": rows}
    Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
    Path(argv[0]).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("probed_on", "months_available",
                                          "new_months_not_in_local_copy")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
