"""Find the pages of the WRA subsidence summary reports and briefings that hold the tables
and statements X4 transcribes (poppler text; the PDFs are untrusted and never executed).

    python -I lev_pages.py            -> prints the page plan as JSON

Kinds of page:

* ``area_county``   表1-1 歷年主要縣市顯著下陷面積統計表 (significant-subsidence area by county)
* ``survey_dates``  表1-2 水準檢測外業期程 (field-survey dates of the year)
* ``max_rate``      表1-4 地層下陷地區最大年下陷速率統計表 (maximum annual rate by region)
* ``interval_chc``  表2-1 彰化地區下陷面積分析表 (Changhua per survey interval)
* ``interval_yl``   表2-2 雲林地區下陷面積分析表 (Yunlin per survey interval)
* ``statement``     the 分區說明 paragraphs on the Changhua and Yunlin results of the year
* ``hsr``           高鐵 (high-speed rail) pier settlement table of a briefing
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x4_canal_leveling.config import L0_BRIEFINGS, L0_CY007001, L0_MISC, L0_SUMMARY  # noqa: E402

SUMMARIES = sorted(L0_SUMMARY.glob("roc1*.pdf"))
BRIEFINGS = sorted(L0_BRIEFINGS.glob("roc1*.pdf")) + sorted(
    L0_MISC.glob("wra_cy_subsidence_results_1*.pdf"))
#: 007001 monthly reports whose 綜合說明 states the year's leveling result (June/July issues)
CY_MONTHLY = [L0_CY007001 / f"cy_{y}-{m}.pdf" for y in (2021, 2024, 2025)
              for m in ("06", "07")]


def page_text(pdf: Path, page: int) -> str:
    return subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf),
                           "-"], check=True, capture_output=True, text=True).stdout


def n_pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], check=True, capture_output=True,
                         text=True).stdout
    return int(next(x.split()[-1] for x in out.splitlines() if x.startswith("Pages")))


def norm(t: str) -> str:
    return re.sub(r"\s+", "", t)


def is_toc(t: str) -> bool:
    return "....." in t or "目錄" in t[:200]


RULES = {
    "area_county": lambda t: "歷年主要縣市顯著下陷面積統計表" in t and "彰化" in t
    and re.search(r"\d+\.\d", t) is not None,
    "survey_dates": lambda t: "水準檢測外業期程" in t and "測量期間" in t,
    "max_rate": lambda t: "最大年下陷速率統計表" in t and "雲" in t and "彰化" in t,
    "interval_chc": lambda t: "彰化地區" in t and "下陷面積分析表" in t and "觀測期距" in t,
    "interval_yl": lambda t: "雲林地區" in t and "下陷面積分析表" in t and "觀測期距" in t,
    "statement": lambda t: ("彰化地區" in t or "雲林" in t) and ("3公分/年" in t or "3cm/yr" in t)
    and ("鄉鎮" in t) and ("水準檢測結果" in t or "平均下陷速率" in t or "下陷速率圖" in t),
    "brief_statement": lambda t: "顯著下陷" in t and ("彰化" in t or "雲林" in t)
    and re.search(r"[\u4e00-\u9fff]{1,2}[鄉鎮市]", t) is not None
    and ("區域" in t or "分布" in t or "最大下陷速率" in t),
    "hsr": lambda t: "高鐵" in t and ("TK" in t or "橋墩" in t) and "沉陷" in t,
}


def plan() -> list[dict]:
    out = []
    for group, pdfs, kinds in (
            ("summary", SUMMARIES, ("area_county", "survey_dates", "max_rate", "interval_chc",
                                    "interval_yl", "statement")),
            ("briefing", BRIEFINGS, ("brief_statement", "hsr")),
            ("cy_monthly", [p for p in CY_MONTHLY if p.exists()], ("statement",))):
        for pdf in pdfs:
            n = n_pages(pdf)
            for p in range(1, min(n, 45 if group == "summary" else 40) + 1):
                t = norm(page_text(pdf, p))
                if is_toc(t):
                    continue
                for k in kinds:
                    if RULES[k](t):
                        out.append({"group": group, "pdf": str(pdf), "page": p, "kind": k})
            if group == "cy_monthly":
                # the leveling paragraph is on the first pages only
                out = [x for x in out if not (x["pdf"] == str(pdf) and x["page"] > 4)]
    return out


def main() -> int:
    pl = plan()
    print(json.dumps(pl, ensure_ascii=False, indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
