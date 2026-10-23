"""Minimal reader for OpenDocument spreadsheets (.ods) without odfpy (untrusted; parse only).

    python -I ods_read.py <file.ods> [max_rows]

Reads ``content.xml`` from the zip, expands repeated rows/columns (capped) and returns each
sheet as a list of rows of strings. No formula is evaluated; cell text is taken as shown.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

NS = {"table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
      "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
      "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0"}
_T = "{" + NS["table"] + "}"
_X = "{" + NS["text"] + "}"
MAX_REPEAT = 64
MAX_BYTES = 50_000_000


def _cell_text(c: ET.Element) -> str:
    parts = []
    for p in c.iter(_X + "p"):
        parts.append("".join(p.itertext()))
    return "\n".join(parts).strip()


def read_ods(path: str | Path) -> dict[str, list[list[str]]]:
    with zipfile.ZipFile(path) as z:
        info = z.getinfo("content.xml")
        if info.file_size > MAX_BYTES:
            raise ValueError("content.xml too large")
        root = ET.fromstring(z.read("content.xml"))
    out = {}
    for t in root.iter(_T + "table"):
        rows = []
        for r in t.iter(_T + "table-row"):
            rr = min(int(r.get(_T + "number-rows-repeated", "1")), MAX_REPEAT)
            cells = []
            for c in r:
                if c.tag not in (_T + "table-cell", _T + "covered-table-cell"):
                    continue
                cr = min(int(c.get(_T + "number-columns-repeated", "1")), MAX_REPEAT)
                cells.extend([_cell_text(c)] * cr)
            while cells and cells[-1] == "":
                cells.pop()
            for _ in range(rr):
                rows.append(list(cells))
        while rows and not any(rows[-1]):
            rows.pop()
        out[t.get(_T + "name", f"sheet{len(out)}")] = rows
    return out


def main(argv: list[str]) -> int:
    n = int(argv[1]) if len(argv) > 1 else 60
    for name, rows in read_ods(argv[0]).items():
        print(f"===== {name} ({len(rows)} rows)")
        for r in rows[:n]:
            print(" | ".join(x.replace("\n", " ") for x in r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
