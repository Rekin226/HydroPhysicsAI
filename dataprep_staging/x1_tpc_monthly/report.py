"""Task 6: collect the X1 numbers into report.json and data_roles.csv (report.md is prose).

Usage: python -I report.py REPORT_DIR
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.paths import REPO  # noqa: E402

PROD = REPO / "results/dataprep/electricity_x1_tpc/full/v1"


def main(rd: str) -> None:
    rd = Path(rd)
    ev = rd / "evidence"
    inv = json.loads((ev / "inventory_2026-10-12.json").read_text(encoding="utf-8"))
    vin = json.loads((ev / "d007012_vintages.json").read_text(encoding="utf-8"))
    val = json.loads((ev / "validation_tpc_vs_census.json").read_text(encoding="utf-8"))
    cv = json.loads((ev / "cv_final.json").read_text(encoding="utf-8"))
    pv = json.loads((PROD / "validation.json").read_text(encoding="utf-8"))
    dr = json.loads((rd / "draft_declaration/X1.hybrid_tf_tpc.prospective_2026-11.v1.draft.json"
                     ).read_text(encoding="utf-8"))
    keep = ("span", "years", "sectors", "unit", "geography", "licence", "url", "rows",
            "ag_彰化縣", "ag_雲林縣", "agri_columns", "agri_classes", "months", "n_counties",
            "rows_彰化縣", "rows_雲林縣")
    t2 = {}
    for c, r in val["counties"].items():
        t2[c] = {w: {s: {k: x[k] for k in ("n_months", "ratio_census_over_tpc",
                                            "anomaly_corr_monthly", "anomaly_corr_bill_pairs",
                                            "seasonal_profile_corr",
                                            "seasonal_amplitude_ratio_census_over_tpc",
                                            "peak_month_census", "peak_month_tpc",
                                            "annual_corr", "anomaly_corr_lagged")}
                     for s, x in v.items()} for w, v in r.items()}
    t3 = {c: {m: {k: v[k] for k in ("pre", "post", "per_year_ratio")}
              for m, v in r.items()} for c, r in cv["county"].items()}
    out = {
        "wp": "X1", "created": "2026-10-21",
        "task1_inventory": {k: {kk: v.get(kk) for kk in keep if kk in v} for k, v in inv.items()},
        "task1_vintages_d007012": vin,
        "task1_online_check_utc": "2026-10-20T17:42:21Z",
        "task1_online_result": "d007012 and Energy Administration set 236 unchanged online "
                               "(same bytes and Last-Modified 2026-09-16T01:42:57Z for "
                               "d007012); last month 2026-08; no newer month to fetch",
        "task2_validation": t2,
        "task3_cv": {"county": t3, "selection": cv["selection"],
                     "sd_calibration": cv["sd_calibration"], "discs": cv["discs"],
                     "disc_sd": {c: {k: v for k, v in x.items() if k != "sdrel_per_well_all"}
                                 for c, x in cv["disc_sd"].items()},
                     "gate_spec_sha256": cv["gate_spec_sha256"]},
        "task4_product": {"path": str(PROD.relative_to(REPO)), "validation": pv},
        "task5_t3_draft": {"path": "results/dataprep/reports/X1_tpc_monthly/draft_declaration",
                           "id": dr["id"], "tests": [t["test_id"] for t in dr["tests"]],
                           "expected_design_skill": dr["draft_expected_design_skill"]},
    }
    (rd / "report.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n",
                                    encoding="utf-8")
    rows = [
        ("X1.validation.tpc_vs_census", "tpc_d007012/2026-10-12", "validator", "L0"),
        ("X1.validation.tpc_vs_census", "pump_electricity/full/v3", "fit_target",
         "census county_class_monthly (compared, not fitted)"),
        ("X1.cv.county_and_disc_skill", "release_20261012b2/full/electricity_census/v1",
         "fit_target", "census in-grid and 3 km series"),
        ("X1.cv.county_and_disc_skill", "tpc_d007012/2026-10-12", "input", "L0"),
        ("X1.cv.county_and_disc_skill", "moeaea_agri_electricity/2026-10-12", "input", "L0"),
        ("X1.cv.county_and_disc_skill", "release_20261012b2/full/et0/v1", "input", "W2"),
        ("X1.cv.county_and_disc_skill", "chou-shui-data/data/rf_timeseries.csv", "input",
         "no manifest"),
        ("X1.cv.county_and_disc_skill", "release_20261012b2/preorigin/heads_observed/v1",
         "input", "stations.parquet only (well x, y)"),
        ("X1.product.electricity_x1_tpc.full.v1", "release_20261012b2/full/electricity_census"
         "/v1", "fit_target", "flag-0 rows copied"),
        ("X1.product.electricity_x1_tpc.full.v1", "tpc_d007012/2026-10-12", "input", "L0"),
        ("X1.t3.expected_skill", "results/dataprep/reports/I1_hybrid_tf/design_skill.json",
         "design_time_evidence", "I1 forward one-step ratios"),
        ("X1.t3.expected_skill", "X1 cv_final.json", "design_time_evidence",
         "anomaly skill f"),
    ]
    with open(rd / "data_roles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number_id", "source", "role", "product_version"])
        w.writerows(rows)
    print("written", rd / "report.json", rd / "data_roles.csv")


if __name__ == "__main__":
    main(sys.argv[1])
