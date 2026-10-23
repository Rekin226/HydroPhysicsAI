"""Look-log records of WP P5 (A1 ``harness.prereg3.record_look``).

Usage: looks_p5.py WORKDIR KIND
KIND = build_check   the pre-registration build checks (counts, wells per period/county, the
                     day-by-day construction check against L1 v3, census sector coverage);
                     no head index was related to electricity or to the pilot
KIND = report_facts  descriptive facts read for the report after every registered test
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import prereg3  # noqa: E402

from p5_head_signatures import config as C  # noqa: E402

NOTES = {
    "build_check": ("[P5-build] before registration: extraction_summary.json (row and well "
                    "counts by period and county, wells with an accepted barometric response, "
                    "day-by-day construction check vs L1 v3: 1,025,861 days, 2 > 1 mm), "
                    "local_kwh_info.json (pump counts, fan GWh/yr), the sectors3 distribution "
                    "by county, the county x layer table, the TF / EEMD run logs (per-well "
                    "fit values scrolled past; TF: 221 wells, 185 with a parameter at a "
                    "bound), the EOF mode summary (mode 0: 66 % of variance, r -0.83 with the "
                    "differenced fan irrigation kWh, r 0.53 with rain; regional series only), "
                    "and smoke runs of every registered test on randomised copies of the "
                    "index and kWh tables (outputs inspected for sample sizes only); no index "
                    "was related to local electricity or to the pilot on real data)"),
    "report_facts": ("[P5-report] after every registered test: descriptive facts for the "
                     "report and product card (product counts, coverage, TF fit summaries)"),
}


def main():
    work = Path(sys.argv[1])
    kind = sys.argv[2]
    F = pd.read_parquet(work / "well_month_features.parquet", columns=["sid"])
    ids = sorted(F["sid"].unique())
    unit = f"P5:{kind}"
    members = {unit: {"description": NOTES[kind], "ids": ids, "window": "2012-01..2026-09"}}
    rec = prereg3.record_look([unit], "heads+census:2012-01..2026-09", C.WP, kind,
                              members=members, note=NOTES[kind])
    print(rec.get("time"), rec.get("purpose"))


if __name__ == "__main__":
    main()
