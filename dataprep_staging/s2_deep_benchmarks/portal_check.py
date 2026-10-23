"""Check the WRA land-subsidence portal for integrated monthly reports newer than 2026-01.

Round 1 probed the predictable file address (``probe_reports.py``). Round 2 also reads the
portal's own listing of series 007003 (整合分析月報告, ``RealtimeMonitor/GetDataReport/007003``,
which needs a session cookie and a referer), so a renamed file would not be missed, and
re-probes the address for 2026-02..the current month with the spelling variants. Only the
parsed titles, publication months and HTTP statuses are written; nothing is executed.

Usage: python -I portal_check.py <out.json>
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from s2_deep_benchmarks import probe_reports as P  # noqa: E402
from s2_deep_benchmarks.fetch_007001 import HOME, SITE, get, opener  # noqa: E402

LIST = SITE + "/RealtimeMonitor/GetDataReport/007003"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    op = opener()
    get(op, HOME).read()
    t = get(op, LIST, referer=HOME).read().decode("utf-8", "replace")
    names = re.findall(r'<td data-th="名稱"[^>]*>([^<]+)</td>', t)
    pub = re.findall(r'<td data-th="發布年月"[^>]*>([^<]*)</td>', t)
    rows = [{"title": n.strip(), "published_roc": p.strip()}
            for n, p in zip(names, pub, strict=True)]
    newest = rows[0]["title"] if rows else None
    today = dt.date.today()
    probes = []
    y, m = 2026, 2
    while (y, m) <= (today.year, today.month):
        for name in P.names(y - 1911, m, True):
            url = P.BASE + urllib.parse.quote(name)
            probes.append({"month": f"{y}-{m:02d}", "name": name, **P.probe(url)})
        m += 1
        if m == 13:
            y, m = y + 1, 1
    found = sorted({p["month"] for p in probes if p.get("pdf_magic")})
    out = {"schema": "s2-portal-check/1", "checked": dt.datetime.now().isoformat(
               timespec="seconds"),
           "listing_url": LIST, "listing_page1": rows, "newest_listed": newest,
           "address_probe_months_found": found, "address_probes": probes,
           "conclusion": ("no integrated monthly report newer than 2026-01" if not found and
                          newest and newest.startswith("115年01月") else "see listing")}
    Path(argv[0]).parent.mkdir(parents=True, exist_ok=True)
    Path(argv[0]).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("checked", "newest_listed",
                                          "address_probe_months_found", "conclusion")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
