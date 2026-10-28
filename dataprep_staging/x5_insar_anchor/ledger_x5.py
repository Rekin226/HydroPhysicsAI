"""Ledger rows (U6) for the numbers reported by the X5 follow-up and E1 reports.

    flock results/dataprep/preregistration/.x5.lock python -I ledger_x5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import x5const as K  # noqa: E402
from common import ledger  # noqa: E402
from common.paths import LEDGER_ROWS, rel_to_repo  # noqa: E402

FU = K.REPORTS_DIR / "X5.t1_postfailure_gnss_leveling.v1" / "report.json"
E1 = K.REPORTS_DIR / "X5.post2022_exploration.v2" / "report.json"
S0 = K.OUT / "synthetic" / "S0_result.json"


def main() -> int:
    rows = []
    s0 = json.loads(S0.read_text(encoding="utf-8"))
    for k in ("median_slope_over_lambda", "median_rmse_mm", "seeds_passing_t1_rule"):
        rows.append({"number_id": f"X5.S0.{k}", "source": rel_to_repo(S0), "role": "input",
                     "product_version": f"prereg:{K.PREREG_SYN}",
                     "note": f"synthetic check S0 = {s0[k]}"})
    fu = json.loads(FU.read_text(encoding="utf-8"))
    for k, v in fu["summary"].items():
        for src, role in ((rel_to_repo(K.LEV_PANEL), "design_time_evidence"),
                          (rel_to_repo(K.GNSS), "input"), (rel_to_repo(K.P2_DIR), "input")):
            rows.append({"number_id": f"X5.T1fu.{k}.median", "source": src, "role": role,
                         "product_version": "prereg:X5.t1_postfailure_gnss_leveling.v1",
                         "note": f"post-failure exploratory; median {v['median_mm']} mm "
                                 f"(n {v['n']})"})
    e1 = json.loads(E1.read_text(encoding="utf-8"))
    pre = K.PUBLIC_POST.parent.parent / "preorigin" / "v1" / "county_intervals.csv"
    post = K.PUBLIC_POST / "county_intervals.csv"
    for iv, ent in e1["intervals"].items():
        for cn in ("changhua", "rest"):
            for fr in ("G", "L", "L2"):
                for stat in ("max_sinking_cm_yr", "area_gt2_km2", "area_gt3_km2"):
                    nid = f"X5.E1.{iv}.{cn}.{fr}.{stat}"
                    val = ent["field"][fr][cn][stat]
                    for src, role in ((rel_to_repo(K.OUT / "v1"), "input"),
                                      (rel_to_repo(pre if iv < "2022" else post),
                                       "design_time_evidence")):
                        rows.append({"number_id": nid, "source": src, "role": role,
                                     "product_version": "x5_insar_anchor/v1",
                                     "note": f"exploration (never validation) = {val}"})
    for iv, v in e1["loso"].items():
        rows.append({"number_id": f"X5.E1.loso.{iv}.rmse_mm", "source": rel_to_repo(K.GNSS),
                     "role": "design_time_evidence", "product_version": "x5_insar_anchor/v1",
                     "note": f"exploration LOSO rmse {v['rmse_mm']} mm, median "
                             f"{v['median_diff_mm']} mm (n {v['n']})"})
    p = ledger.append_rows(K.WP, rows, rows_dir=LEDGER_ROWS)
    print("ledger", rel_to_repo(p), len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
