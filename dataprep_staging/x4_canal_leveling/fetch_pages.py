"""Fetch a list of public pages or files into one X4 L0 directory (polite, logged).

    python -I fetch_pages.py <dest_key> <name>=<url> [<name>=<url> ...]

``dest_key`` is ``canal`` (data_fetch_api/new_measurements/ia_irrigation_open_data/
2026-10-21/) or a source id from ``EXTRA_DEST``. Every request is appended to
``fetch_log.json`` in the destination (url, status, bytes, content type, time). Bodies are
saved byte for byte and never executed or parsed here.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x4_canal_leveling import web  # noqa: E402
from x4_canal_leveling.config import FETCH_DATE, L0_CANAL, L0_NEW  # noqa: E402

EXTRA_DEST = {
    "canal": L0_CANAL,
    "chcg": L0_NEW / "chcg_irrigation_notices" / FETCH_DATE,
    "moa": L0_NEW / "moa_town_crop_rice_2023_2025" / FETCH_DATE,
    "wra": L0_NEW / "wra_subsidence_briefings_007007" / FETCH_DATE,
    "wiki": L0_NEW / "zhwiki_changhua_canals" / FETCH_DATE,
}


def fetch(dest: Path, items: list[tuple[str, str]], referer: str | None = None) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    op = web.opener()
    log_p = dest / "fetch_log.json"
    log = json.loads(log_p.read_text(encoding="utf-8")) if log_p.exists() else []
    out = []
    for name, url in items:
        if "/" in name or name.startswith("."):
            raise ValueError(f"bad file name {name!r}")
        st, hdr, body = web.get(op, url, referer=referer)
        rec = {"file": name if st == 200 else None, "url": url, "status": st,
               "bytes": len(body), "content_type": hdr.get("Content-Type"),
               "last_modified": hdr.get("Last-Modified"),
               "time": dt.datetime.now().isoformat(timespec="seconds")}
        if st == 200:
            web.save(dest / name, body)
        log.append(rec)
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    log_p.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in EXTRA_DEST:
        print(__doc__)
        return 2
    items = []
    for a in argv[1:]:
        name, _, url = a.partition("=")
        items.append((name, url))
    fetch(EXTRA_DEST[argv[0]], items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
