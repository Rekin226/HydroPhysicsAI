"""WP H4: append the data-role ledger rows (plan 4.1, U6) to ledger/rows/H4.csv.

One role per source per number. Usage: python ledger_h4.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import ledger  # noqa: E402
from common.paths import DATAPREP, LEDGER_ROWS  # noqa: E402

PV = "results/dataprep/subgrid_operator/v1"
HEADS = "results/dataprep/heads/L1_clean/v3"
ENGINE = "results/dataprep/changhua_abstraction/engine/v2"
ELEC = "results/dataprep/yunlin_electricity/preorigin/v2"
CENSUS = "results/dataprep/pump_electricity/census_L1/v3"
UNITS = "results/dataprep/validation_units/v3"
C0 = "results/dataprep/pump_electricity/preorigin/v3"
TPC = "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/parquet"


def rows() -> list[dict]:
    r = []

    def add(num, src, role, note, pv=PV):
        r.append({"number_id": num, "source": src, "role": role, "product_version": pv,
                  "note": note})

    model_inputs = [
        (ENGINE, "base hindcast h_cell (cwa theta, in sample 2012-22), zone T and S, the "
                 "twin's pump split and conversion scalars"),
        (ELEC, "per-pump Chow-Lin kWh (Yunlin, kept pumps) -> Q_j"),
        (CENSUS, "pump coordinates (kept pumps)")]
    # multipliers: fitted on 2012-19 heads
    for num, note in (("H4.multipliers.t1", "a_s, a_l fitted on every well, 2012-02..2019-12"),
                      ("H4.multipliers.t2_folds", "a_s, a_l per A1 site fold, 2012-02..2019-12")):
        add(num, HEADS, "fit_target", "heads_monthly flags 0-1, window_end 2022-12; rows "
                                      "2012-02..2019-12 only. " + note)
        for s, n in model_inputs:
            add(num, s, "input", n)
        if num.endswith("folds"):
            add(num, UNITS, "input", "fresh.folds.wells_all partition read as data")
    # acceptance criteria (held-out scores)
    for k in range(1, 9):
        num = f"H4.A{k}"
        add(num, HEADS, "validator", "heads_monthly flags 0-1: the 2020-01..2022-12 rows are "
                                     "the scored observations; the 2012-02..2019-12 rows set "
                                     "the multipliers, the training-mean datum and the "
                                     "climatology that are part of the score definition")
        for s, n in model_inputs:
            add(num, s, "input", n)
        if k in (2, 5, 6, 8):
            add(num, UNITS, "input", "fresh.folds.wells_all partition read as data (T2 folds)")
    # sensitivity members (reported, not gated)
    sens = json.loads((DATAPREP / "reports/H4_evidence/registered/scores.json").read_text())[
        "sensitivity"]
    for name in sens:
        num = f"H4.sensitivity.{name}"
        add(num, HEADS, "validator", "as H4.A1/A2 (held-out scores of the member)")
        for s, n in model_inputs:
            add(num, s, "input", n)
        add(num, UNITS, "input", "T2 folds")
        if name == "recorded_arm":
            add(num, C0, "input", "recorded arm per pump (TPC even split)")
    add("H4.t2_spatial_only", HEADS, "validator",
        "fold fits scored on the test wells' 2012-02..2019-12 months (in-sample datum and "
        "climatology); reported only")
    for s, n in model_inputs:
        add("H4.t2_spatial_only", s, "input", n)
    add("H4.t2_spatial_only", UNITS, "input", "T2 folds")
    # design-time numbers (no head data)
    for s, n in model_inputs:
        add("H4.design.d0", s, "design_time_evidence", n + " (d0_design_check: pump counts, "
                                                         "kernel shapes, unit corrections)")
    add("H4.design.t0_smoke", ENGINE, "design_time_evidence",
        "synthetic heads from the hindcast and the operator (no observation)")
    # exemptions inherited by the product (U3)
    for name, src, note in (
            ("census_attributes", f"{TPC}/yunlin_census.parquet",
             "2026 census attribute table (coordinates, kept set) via census_L1/v3 and "
             "yunlin_electricity/preorigin/v2"),
            ("residual_unpaired_dec2022", f"{TPC}/yunlin_kwh_long.parquet",
             "Dec-2022 reading of unidentified Dec-Jan-paired meters, via "
             "yunlin_electricity/preorigin/v2"),
            ("zero_check_gate",
             "results/dataprep/reports/P2abc_evidence/repair3/tests/t6_zero_shipped.json",
             "P2abc registered zero check (gate input of yunlin_electricity/preorigin/v2)")):
        add(f"H4.exemption.{name}", src, "input", "exemption (U3) inherited: " + note)
    return r


def main() -> None:
    rr = rows()
    path = ledger.append_rows("H4", rr, rows_dir=LEDGER_ROWS)
    allrows = ledger._read(path)
    conf = ledger.conflicts(allrows)
    print(json.dumps({"fragment": str(path), "appended": len(rr), "total": len(allrows),
                      "conflicts": conf}, indent=1))


if __name__ == "__main__":
    main()
