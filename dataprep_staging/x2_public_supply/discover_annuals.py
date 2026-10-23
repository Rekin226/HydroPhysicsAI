"""List the WRA water-use annual reports (web.wra.gov.tw/Annualwater/annuals.aspx) by year.

The page is a public ASP.NET form: choosing a year in ``fyears`` posts back and lists that
year's report PDFs. This script posts back once per requested ROC year (saving each listing
page into L0 ``wra_water_use_annual``) and writes the parsed link table to
``reports/X2_public_supply/evidence/annual_report_links.json``.

Usage: python -I discover_annuals.py <roc_year> [<roc_year> ...]
"""
from __future__ import annotations

import html
import json
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2_public_supply import fetch  # noqa: E402
from x2_public_supply import x2const as K  # noqa: E402

PAGE = "https://web.wra.gov.tw/Annualwater/annuals.aspx"
FIELD = "ctl00$ContentPlaceHolder1$fyears"


def parse_links(page: str) -> list[dict]:
    out = []
    for m in re.finditer(r'<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
        href = html.unescape(m.group(1))
        if "annualreports/" not in href:
            continue
        label = re.sub(r"<[^>]+>|\s+", " ", m.group(2)).strip()
        out.append({"label": label, "href": href,
                    "url": urllib.parse.urljoin(PAGE, href)})
    return out


def main(years: list[str]) -> None:
    d = K.L0["wra_water_use_annual"]
    base = (d / "annuals_index_113.html").read_text(encoding="utf-8", errors="replace")
    res_path = K.EVID / "annual_report_links.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    res.setdefault("113", parse_links(base))
    for y in years:
        name = f"annuals_index_{y}.html"
        if (d / name).exists():
            page = (d / name).read_text(encoding="utf-8", errors="replace")
        else:
            fetch.postback("wra_water_use_annual", name, PAGE, FIELD, {FIELD: y},
                           note=f"annual reports listing for ROC year {y} (form postback)",
                           page_html=base)
            page = (d / name).read_text(encoding="utf-8", errors="replace")
        res[y] = parse_links(page)
        print(y, len(res[y]), [r["label"] for r in res[y]])
    K.EVID.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
