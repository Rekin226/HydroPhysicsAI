"""Ledger rows (U6) for every number X5 reports. Run under the X5 flock.

Usage: flock <lock> python -I ledger_x5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from x5_changhua_water import x5const as K  # noqa: E402

PV = "x5_changhua_water/v1"
PRE = "x5_changhua_water/preorigin/v1"


def r(nid, src, role, pv=PV, note=""):
    return {"number_id": nid, "source": rel_to_repo(src) if isinstance(src, Path) else src,
            "role": role, "product_version": pv, "note": note}


def rows() -> list[dict]:
    w1pre = K.W1_PRE / "district_monthly.csv"
    ind = [K.WORK / "coherence_monthly.csv", K.W1_PRE / "inflow_composite_daily.parquet",
           K.W1_PRE / "gauges_daily_L1.parquet", K.HEADS / "heads_monthly.parquet"]
    out = []
    for nid in ("X5.annual_nonjiji_by_year", "X5.annual_nonjiji_level_2016_2022"):
        out += [r(nid, K.MOA_CSV, "input"), r(nid, K.JIJI_CSV, "input")]
    out += [r("X5.sd_def", K.MOA_CSV, "input", note="Yunlin office minus Jiji 2016-2022"),
            r("X5.sd_def", K.JIJI_CSV, "input")]
    out += [r("X5.synthetic_check", K.JIJI_CSV, "input", note="synthetic truth scale"),
            r("X5.synthetic_check", K.SYN_DIR / "synthetic_check.json", "design_time_evidence")]
    out += [r("X5.fit_b", w1pre, "fit_target", note="Yunlin and Changhua W1 shares "
              "2015-2021 (inferred, flag 5)")] + [r("X5.fit_b", p, "input") for p in ind]
    out += [r("X5C.heldout_r", w1pre, "validator", note="W1 Yunlin 2012-2014, 2022; "
              "inferred reference: agreement, not skill (U1)"),
            r("X5C.heldout_r", K.FIT_DIR / "fit.json", "input"),
            r("X5C.heldout_r", K.JIJI_CSV, "input")] + [r("X5C.heldout_r", p, "input")
                                                        for p in ind]
    out += [r("X5.exploration_2023_2025_r", K.W1_FULL / "district_monthly.csv", "validator",
              note="exploration only (U7); W1 full/v4 is inferred")]
    out += [r("X5.designtime_janmay_2015_2021", K.W1_PRE / "janmay_validation.csv",
              "design_time_evidence", note="observed Jan-May, already seen; in-sample"),
            r("X5.designtime_crop1_share_2012_2015", K.MOA_CSV, "design_time_evidence",
              note="MOA crop-season shares, already seen")]
    out += [r("X5.monthly_changhua_nonjiji", K.FIT_DIR / "fit.json", "input"),
            r("X5.monthly_changhua_nonjiji", K.MOA_CSV, "input"),
            r("X5.monthly_changhua_nonjiji", K.JIJI_CSV, "input")]
    out += [r("X5.preorigin.service_map_exemption", K.X4_PRE, "input", PRE,
              note="2024 canal service map: coherence mask and primary placement"),
            r("X5.preorigin.luimap_exemption", K.V3_NPZ, "input", PRE,
              note="LUIMAP 2024 paddy area: alternative placement")]
    return out


if __name__ == "__main__":
    p = ledger.append_rows(K.WP, rows())
    print("ledger", rel_to_repo(p))
