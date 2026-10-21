"""Download chosen WRA water-use annual report PDFs into L0 ``wra_water_use_annual``.

Reads the link table written by ``discover_annuals.py``; a report is chosen by ROC year and
a kind keyword matched against the printed label (生活 = domestic, 各標的 = all uses,
各區域 = regional/county analysis, 抗旱 = drought-well survey, 工業 = industry, 農業 =
agriculture). The file is saved as ``wra_wateruse_<roc>_<kind>.pdf`` and must start with
%PDF- (otherwise it is kept, flagged in the log, and the script stops).

Usage: python -I fetch_annual_reports.py <kind> <roc_year> [<roc_year> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2_public_supply import fetch  # noqa: E402
from x2_public_supply import x2const as K  # noqa: E402

KINDS = {"domestic": "生活", "alluses": "各標的", "regional": "各區域", "drought": "抗旱",
         "industry": "工業", "agri": "農業", "yearbook": "用水統計年報"}


def pick(links: list[dict], kind: str) -> dict:
    kw = KINDS[kind]
    hits = [r for r in links if kw in r["label"]]
    if kind == "alluses":
        hits = [r for r in hits if "概況" not in r["label"]]
    if kind == "yearbook":
        hits = [r for r in hits if "各標" not in r["label"]]
    if len(hits) != 1:
        raise SystemExit(f"{kind}: {len(hits)} matching links {[r['label'] for r in hits]}")
    return hits[0]


def main(kind: str, years: list[str]) -> None:
    links = json.loads((K.EVID / "annual_report_links.json").read_text(encoding="utf-8"))
    d = K.L0["wra_water_use_annual"]
    for y in years:
        r = pick(links[y], kind)
        name = f"wra_wateruse_{y}_{kind}.pdf"
        if (d / name).exists():
            print("have", name)
            continue
        rec = fetch.get("wra_water_use_annual", name, r["url"],
                        note=f"ROC {y} ({int(y) + 1911}): {r['label']}")
        head = (d / name).read_bytes()[:5]
        print(name, rec["bytes"], head)
        if head != b"%PDF-":
            raise SystemExit(f"{name} is not a PDF")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
