"""List the attachments of a saved 農田水利署 (ia.gov.tw) article page (untrusted; parse only).

    python -I ia_attachments.py <page.html> [...]   -> one JSON line per attachment

Each row: page, attachment title, the file name the site gives in the link title, the
download URL (``/zh-TW/fapi/AttFile?...``) and the printed format.
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

BASE = "https://www.ia.gov.tw"
_ROW = re.compile(r"<tr>(.*?)</tr>", re.S)
_LINK = re.compile(r'<a class="downloadTable-link" href="([^"]+)" title=":?([^"]*)"')
_FMT = re.compile(r"格式：<b>([^<]+)</b>")


def attachments(path: str | Path) -> list[dict]:
    raw = Path(path).read_bytes().decode("utf-8", "replace")
    i = raw.find('id="c"')
    j = raw.find("</main>", max(i, 0))
    body = raw[i:j] if i >= 0 and j > i else raw
    out = []
    for row in _ROW.findall(body):
        m = _LINK.search(row)
        if not m:
            continue
        cell = row.split("<a", 1)[0]
        title = " ".join(re.sub(r"<[^>]+>", " ", cell).split())
        f = _FMT.search(row)
        out.append({"page": Path(path).name, "title": html.unescape(title),
                    "file_name": html.unescape(m[2]),
                    "url": BASE + html.unescape(m[1]), "format": f[1] if f else None})
    return out


def main(argv: list[str]) -> int:
    for p in argv:
        for a in attachments(p):
            print(json.dumps(a, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
