"""Archive WRA land-subsidence portal report series into L0 (X4 leveling part).

Series (portal ``RealtimeMonitor/GetDataReport/<id>``):

* ``007007`` 地層下陷檢測結果說明 / 彰雲地區地層下陷檢測成果: the annual leveling briefings to
  the MOEA subsidence committee (one per survey year);
* ``007005`` 歷年地層下陷檢測成果摘要報告: the summary report of the survey results.

Steps (each request at least 1.2 s after the previous one; no robots.txt is served):

    python -I fetch_wra.py list <series>          save the listing pages, print the rows
    python -I fetch_wra.py get <series> [regex]   download listed PDFs whose title matches

Files land in ``data_fetch_api/new_measurements/<source_id>/2026-10-21/``; SOURCE.json is
written by ``record_sources.py`` once all files are in. The PDFs are untrusted data: this
script checks only the PDF magic bytes.
"""
from __future__ import annotations

import html
import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x4_canal_leveling import web  # noqa: E402
from x4_canal_leveling.config import (  # noqa: E402
    L0_BRIEFINGS,
    L0_SUMMARY,
    WRA_FILE,
    WRA_HOME,
    WRA_LIST,
)

DEST = {"007007": L0_BRIEFINGS, "007005": L0_SUMMARY}


def parse_listing(text: str, series: str) -> tuple[list[dict], list[int]]:
    names = re.findall(r'<td data-th="名稱"[^>]*>([^<]+)</td>', text)
    pub = re.findall(r'<td data-th="發布年月"[^>]*>([^<]*)</td>', text)
    paths = []
    for h in re.findall(r'href="(/landsubsidence/water_new/RealtimeMonitor/Download\?[^"]+)"',
                        text):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(html.unescape(h)).query)
        paths.append(q.get("filepath", [""])[0])
    pages = sorted({int(x) for x in re.findall(rf"GetDataReport/{series}\?Page=(\d+)", text)})
    if not (len(names) == len(pub) == len(paths)):
        raise RuntimeError(f"listing columns differ: {len(names)} {len(pub)} {len(paths)}")
    rows = [{"title": html.unescape(n).strip(), "published_roc": p.strip(), "filepath": f}
            for n, p, f in zip(names, pub, paths, strict=True)]
    return rows, pages


def listing(series: str) -> list[dict]:
    dest = DEST[series]
    dest.mkdir(parents=True, exist_ok=True)
    op = web.opener()
    web.get(op, WRA_HOME)
    url = WRA_LIST.format(series=series)
    st, _, body = web.get(op, url, referer=WRA_HOME)
    if st != 200:
        raise SystemExit(f"listing {url}: HTTP {st}")
    web.save(dest / f"listing_{series}_p1.html", body)
    rows, pages = parse_listing(body.decode("utf-8", "replace"), series)
    for pg in [p for p in pages if p > 1]:
        st, _, b = web.get(op, f"{url}?Page={pg}", referer=url)
        web.save(dest / f"listing_{series}_p{pg}.html", b)
        rows += parse_listing(b.decode("utf-8", "replace"), series)[0]
    for r in rows:
        name = r["filepath"].replace("\\", "/").split("/")[-1]
        r["original_name"] = name
        r["url"] = WRA_FILE.format(series=series, name=urllib.parse.quote(name))
    (dest / f"rows_{series}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    return rows


def safe_name(title: str, k: int) -> str:
    m = re.match(r"^(\d{3})", title)
    stem = f"roc{m[1]}" if m else f"row{k:02d}"
    return stem


def download(series: str, pattern: str | None) -> list[dict]:
    dest = DEST[series]
    rows = json.loads((dest / f"rows_{series}.json").read_text(encoding="utf-8"))
    op = web.opener()
    web.get(op, WRA_HOME)
    got, seen = [], set()
    for k, r in enumerate(rows):
        if pattern and not re.search(pattern, r["title"]):
            continue
        if r["filepath"] in seen:
            continue
        seen.add(r["filepath"])
        fn = f"{safe_name(r['title'], k)}_{k:02d}.pdf"
        st, hdr, body = web.get(op, r["url"], referer=WRA_LIST.format(series=series))
        ok = st == 200 and body[:5] == b"%PDF-"
        rec = {"title": r["title"], "published_roc": r["published_roc"], "url": r["url"],
               "original_name": r["original_name"], "status": st,
               "last_modified": hdr.get("Last-Modified"), "pdf": ok}
        if ok:
            web.save(dest / fn, body)
            rec["file"] = fn
            rec["bytes"] = len(body)
        got.append(rec)
        print(json.dumps({k2: rec[k2] for k2 in ("title", "status", "pdf")}, ensure_ascii=False))
    log = dest / f"download_log_{series}.json"
    old = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    log.write_text(json.dumps(old + got, ensure_ascii=False, indent=1), encoding="utf-8")
    return got


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("list", "get") or argv[1] not in DEST:
        print(__doc__)
        return 2
    if argv[0] == "list":
        for r in listing(argv[1]):
            print(r["published_roc"], r["title"], r["original_name"])
    else:
        download(argv[1], argv[2] if len(argv) > 2 else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
