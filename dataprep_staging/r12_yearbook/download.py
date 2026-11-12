"""Build out/yearbook_urls.csv from crawl results and download the selected groundwater volumes."""
from __future__ import annotations

import os  # noqa
# ruff: noqa
import csv, hashlib, json, os, sys, time, urllib.parse, urllib.request
ROOT = os.environ.get("HYDRO_YEARBOOK_ROOT", "data_fetch_api/new_measurements/wra_hydro_yearbook_gw")
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
BASE = "https://gweb.wra.gov.tw/wrhygis/ebooks/ebook/ebook/hyb%d/"
rows = []
for fn in ("crawl_test.json", "crawl_survey.json", "crawl_survey2.json"):
    d = json.load(open(f"{ROOT}/tmp/{fn}"))
    for y, v in d.items():
        gw = [f for f in v["files"] if "地下水" in f and f.lower().endswith(".pdf")]
        for f in gw:
            rows.append((int(y), f, BASE % int(y) + urllib.parse.quote(f)))
        if not gw:
            rows.append((int(y), "", ""))
rows.sort()
with open(f"{ROOT}/out/yearbook_urls.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["year", "relative_path", "url", "note"])
    for y, f, u in rows:
        w.writerow([y, f, u, "" if u else "no groundwater-level volume linked from the year index"])
want = set(map(int, sys.argv[1:]))
dl = []
for y, f, u in rows:
    if y not in want or not u: continue
    dst = f"{ROOT}/raw/gw_{y}.pdf"
    status = "cached"
    if not os.path.exists(dst):
        status = "fail"
        for k in range(4):
            try:
                req = urllib.request.Request(u, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=300) as r, open(dst + ".part", "wb") as o:
                    while True:
                        b = r.read(1 << 20)
                        if not b: break
                        o.write(b)
                os.rename(dst + ".part", dst); status = "ok"; break
            except Exception as e:
                print("retry", y, e, file=sys.stderr); time.sleep(5 * (k + 1))
        time.sleep(2)
    if os.path.exists(dst):
        h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
        dl.append([y, u, os.path.basename(dst), os.path.getsize(dst), h, status])
    else:
        dl.append([y, u, "", 0, "", status])
    print(dl[-1][0], dl[-1][3], dl[-1][5], file=sys.stderr)
with open(f"{ROOT}/out/downloads.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["year", "url", "file", "size_bytes", "sha256", "status"]); w.writerows(dl)
