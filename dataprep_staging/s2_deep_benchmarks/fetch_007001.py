"""Archive the WRA Changhua-Yunlin monthly data-analysis reports (series 007001) into L0.

The WRA land-subsidence portal lists six report series under
``RealtimeMonitor/GetDataReport/<id>``. Series 007003 is the integrated monthly report whose
table 3 the twin already transcribes (2024-01..2026-01; the listing's newest entry is
115年01月, checked 2026-10-12). Series 007001, 彰化與雲林地區地層下陷監測及分析 資料分析月報,
is the Changhua-Yunlin monitoring project's own monthly report, 2020-02..2025-12. Its table 1
(彰雲地區深層水準樁 觀測成果) prints, for seven deep benchmarks, the change of the report month
and of the same month a year earlier in millimetres (one decimal; positive = rebound), plus two
18-month cumulative changes. It covers the pre-origin years the integrated series lacks.

This script reads the listing pages, range-probes every listed PDF for its size (aborts above
``MAX_TOTAL_MB``), downloads each once into
``data_fetch_api/new_measurements/wra_cy_monthly_reports_007001/`` (read-only copies, names
``cy_<YYYY-MM>.pdf`` by report month, original names in SOURCE.json) and writes SOURCE.json.
The files are untrusted data: this script only checks the PDF magic bytes and hashes them.

Usage: python -I fetch_007001.py <scratch_dir>
"""
from __future__ import annotations

import datetime as dt
import html
import http.cookiejar
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import source  # noqa: E402
from common.paths import L0_NEW  # noqa: E402

SITE = "https://web.wra.gov.tw/landsubsidence/water_new"
LIST = SITE + "/RealtimeMonitor/GetDataReport/007001"
HOME = SITE + "/Home/IndexReNew"
DEST = L0_NEW / "wra_cy_monthly_reports_007001"
UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
MAX_TOTAL_MB = 600.0
TITLE = re.compile(r"^(\d{3})年度?彰化與雲林地區地層下陷監測及分析_(\d{1,2})月資料分析月報")


def opener():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", UA)]
    return op


def get(op, url: str, referer: str | None = None, rng: str | None = None):
    req = urllib.request.Request(url)
    if referer:
        req.add_header("Referer", referer)
    if rng:
        req.add_header("Range", rng)
    return op.open(req, timeout=120)


def parse_listing(text: str) -> tuple[list[dict], list[int]]:
    names = re.findall(r'<td data-th="名稱"[^>]*>([^<]+)</td>', text)
    pub = re.findall(r'<td data-th="發布年月"[^>]*>([^<]*)</td>', text)
    paths = []
    for h in re.findall(r'href="(/landsubsidence/water_new/RealtimeMonitor/Download\?[^"]+)"',
                        text):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(html.unescape(h)).query)
        paths.append(q.get("filepath", [""])[0])
    pages = sorted({int(x) for x in re.findall(r"GetDataReport/007001\?Page=(\d+)", text)})
    if not (len(names) == len(pub) == len(paths)):
        raise RuntimeError(f"listing columns differ: {len(names)} {len(pub)} {len(paths)}")
    rows = [{"title": n.strip(), "published_roc": p.strip(), "filepath": f}
            for n, p, f in zip(names, pub, paths, strict=True)]
    return rows, pages


def report_month(title: str) -> str | None:
    m = TITLE.match(title)
    if not m:
        return None
    return f"{int(m[1]) + 1911}-{int(m[2]):02d}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    scratch = Path(argv[0])
    scratch.mkdir(parents=True, exist_ok=True)
    op = opener()
    get(op, HOME).read()
    first = get(op, LIST, referer=HOME).read().decode("utf-8", "replace")
    (scratch / "listing_007001_p1.html").write_text(first, encoding="utf-8")
    rows, pages = parse_listing(first)
    for pg in [p for p in pages if p > 1]:
        t = get(op, f"{LIST}?Page={pg}", referer=LIST).read().decode("utf-8", "replace")
        (scratch / f"listing_007001_p{pg}.html").write_text(t, encoding="utf-8")
        rows += parse_listing(t)[0]
        time.sleep(1.0)
    for r in rows:
        r["report_month"] = report_month(r["title"])
        name = r["filepath"].replace("\\", "/").split("/")[-1]   # some rows use backslashes
        r["original_name"] = name
        r["url"] = SITE + "/File/ReportData/007001/" + urllib.parse.quote(name)
    # one file per distinct path; listing rows whose path repeats another row's are kept
    # in the record with a note (the agency's link, not a second file)
    by_path: dict[str, dict] = {}
    for r in rows:
        if r["filepath"] in by_path:
            by_path[r["filepath"]].setdefault("also_listed_as", []).append(
                {"title": r["title"], "published_roc": r["published_roc"]})
        else:
            by_path[r["filepath"]] = r
    uniq = list(by_path.values())
    total = 0
    for r in uniq:
        try:
            with get(op, r["url"], rng="bytes=0-1023") as resp:
                head = resp.read(1024)
                cr = resp.headers.get("Content-Range") or ""
                r["last_modified"] = resp.headers.get("Last-Modified")
                r["status"] = resp.status
        except urllib.error.HTTPError as exc:
            head, cr, r["status"], r["last_modified"] = b"", "", exc.code, None
        r["pdf_magic"] = head[:5] == b"%PDF-"
        r["bytes_remote"] = int(cr.split("/")[-1]) if "/" in cr else None
        total += r["bytes_remote"] or 0
        time.sleep(0.5)
    missing = [r for r in uniq if not r["pdf_magic"]]
    uniq = [r for r in uniq if r["pdf_magic"]]
    probe = {"probed": dt.datetime.now().isoformat(timespec="seconds"), "n_listed": len(rows),
             "n_files": len(uniq), "total_mb": round(total / 1e6, 1), "rows": rows}
    (scratch / "probe_007001.json").write_text(json.dumps(probe, ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    print(json.dumps({k: probe[k] for k in ("n_listed", "n_files", "total_mb")}))
    if total / 1e6 > MAX_TOTAL_MB:
        raise SystemExit(f"total {total / 1e6:.0f} MB exceeds {MAX_TOTAL_MB} MB; not downloaded")
    if missing:
        print("listed but not served (recorded, not downloaded):",
              [(r["original_name"], r["status"]) for r in missing])
    dl = scratch / "pdf"
    dl.mkdir(exist_ok=True)
    files = []
    for r in uniq:
        stem = f"cy_{r['report_month']}" if r["report_month"] else "cy_unknown"
        fn = f"{stem}.pdf"
        k = 2
        while any(f["file"] == fn for f in files):
            fn = f"{stem}_{k}.pdf"
            k += 1
        tmp = dl / fn
        if not tmp.exists() or tmp.stat().st_size != r["bytes_remote"]:
            with get(op, r["url"]) as resp:
                tmp.write_bytes(resp.read())
            time.sleep(0.5)
        if tmp.read_bytes()[:5] != b"%PDF-" or tmp.stat().st_size != r["bytes_remote"]:
            raise SystemExit(f"{fn}: not a complete PDF")
        source.copy_into_l0(tmp, DEST / fn)
        rec = {"file": fn, "url": r["url"], "original_name": r["original_name"],
               "listing_title": r["title"], "published_roc": r["published_roc"],
               "report_month": r["report_month"], "last_modified": r["last_modified"]}
        if r.get("also_listed_as"):
            rec["note"] = ("the listing links these other rows to the same file (agency "
                           f"link; no separate file): {r['also_listed_as']}")
        files.append(rec)
    for pg in [1, *[p for p in pages if p > 1]]:
        source.copy_into_l0(scratch / f"listing_007001_p{pg}.html",
                            DEST / f"listing_007001_p{pg}.html")
        files.append({"file": f"listing_007001_p{pg}.html",
                      "url": LIST if pg == 1 else f"{LIST}?Page={pg}",
                      "note": "listing page (titles, 發布年月, file paths) as served"})
    rec = source.build_source(
        DEST, source_id="wra_cy_monthly_reports_007001",
        title="WRA 彰化與雲林地區地層下陷監測及分析 資料分析月報 (Changhua-Yunlin land-subsidence "
              "monitoring monthly data-analysis reports, portal series 007001), 2020-02..2025-12",
        publisher="經濟部水利署 地層下陷監測資訊整合服務系統 (Water Resources Agency land-subsidence "
                  "portal)",
        licence="Government website publication (政府網站資料); no explicit open-data licence on "
                "the listing page; kept in gitignored L0 for research transcription only, not "
                "redistributed (same footing as wra_monthly_reports)",
        fetched=dt.date.today().isoformat(),
        files=files,
        provenance="Fetched 2026-10-12 by WP S2 round 2 "
                   "(dataprep_staging/s2_deep_benchmarks/fetch_007001.py): listing pages with a "
                   "session cookie and referer, each PDF range-probed for size, then downloaded "
                   "once.",
        caveats=[
            "Report months follow the listing title (民國 year + month). Table 1 of each report "
            "prints the report month and the same month a year earlier.",
            "Signs: the tables print positive = rebound, negative = compression (the feeds read "
            "rising = compression).",
            "Listing rows sharing one file path are recorded once with a note.",
            "Listed but not served on 2026-10-12 (HTTP status): "
            + json.dumps([{"title": r["title"], "file": r["original_name"],
                           "status": r["status"]} for r in missing], ensure_ascii=False),
        ])
    p = source.write_source_json(DEST, rec)
    print(f"wrote {p} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
