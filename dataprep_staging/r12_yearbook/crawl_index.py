"""Crawl the static ebook index pages of one yearbook year and list all linked files."""
from __future__ import annotations

import os  # noqa
# ruff: noqa
import sys, time, re, json, urllib.request, urllib.parse
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
BASE = "https://gweb.wra.gov.tw/wrhygis/ebooks/ebook/ebook/hyb%d/"

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for k in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:
            err = e; time.sleep(2 * (k + 1))
    print("FAIL", url, err, file=sys.stderr); return None

def crawl(year, maxpages=6):
    base = BASE % year
    seen, queue, files = set(), ["default.htm"], {}
    while queue and len(seen) < maxpages:
        p = queue.pop(0)
        if p in seen: continue
        seen.add(p)
        raw = get(urllib.parse.urljoin(base, urllib.parse.quote(p, safe="/%")))
        time.sleep(1.0)
        if raw is None: continue
        txt = None
        for enc in ("utf-8", "big5", "cp950"):
            try: txt = raw.decode(enc); break
            except Exception: pass
        if txt is None: txt = raw.decode("big5", "ignore")
        for m in re.finditer(r'''(?:href|src)\s*=\s*["']?([^"' >]+)''', txt, re.I):
            h = m.group(1)
            if h.startswith(("http", "file:", "javascript", "#", "mailto", "%3c")): continue
            h = urllib.parse.unquote(h).replace("\\", "/")
            full = urllib.parse.urljoin(p, h) if "/" in p else h
            full = re.sub(r"#.*$", "", full)
            if re.search(r"\.html?$", full, re.I):
                if full not in seen: queue.append(full)
            else:
                files.setdefault(full, p)
    return sorted(seen), files

if __name__ == "__main__":
    out = {}
    for y in map(int, sys.argv[1:]):
        pages, files = crawl(y)
        out[y] = {"pages": pages, "files": files}
        print(y, len(pages), "pages", len(files), "files", file=sys.stderr)
    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
