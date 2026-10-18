"""WP H4 repair round: correct the v1 ledger roles and add the v2 rows (plan 4.1, U6).

1. The round-1 fragment is archived once as ledger/superseded/H4.r1.csv.
2. v1 rows are corrected (verifier minor findings): the heads behind A1-A8, the sensitivity
   members and the T2 spatial-only variant are design_time_evidence, not validator (2020-22 at
   the 158 wells is a design-time unit, plan 4.1); each v1 sensitivity member gets a separate
   multiplier number whose heads role is fit_target; the all_ingrid_counties member names the
   Changhua and Chiayi pump arrays it used.
3. v2 rows are appended (one role per source per number).
Usage: python ledger_h4v2.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from common import ledger  # noqa: E402
from common.paths import DATAPREP, LEDGER_ROWS  # noqa: E402

PV1 = "results/dataprep/subgrid_operator/v1"
PV2 = "results/dataprep/subgrid_operator/v2"
HEADS = "results/dataprep/heads/L1_clean/v3"
ENGINE = "results/dataprep/changhua_abstraction/engine/v2"
ELEC = "results/dataprep/yunlin_electricity/preorigin/v2"
CENSUS = "results/dataprep/pump_electricity/census_L1/v3"
UNITS = "results/dataprep/validation_units/v3"
C0 = "results/dataprep/pump_electricity/preorigin/v3"
GRID = "results/twin_forward/datum_gate_innov.npz"
TPC = "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/parquet"
SUPERSEDED = DATAPREP / "ledger/superseded/H4.r1.csv"
FIX = " [role corrected in the H4 repair round: 2020-22 at the 158 calibration wells is a " \
      "design-time unit (plan 4.1), so these head rows are design_time_evidence]"


def corrected_v1(old: list[dict]) -> list[dict]:
    out = []
    sens_names = set()
    for r in old:
        r = dict(r)
        n = r["number_id"]
        if r["source"] == HEADS and r["role"] == "validator" and (
                n.startswith("H4.A") or n.startswith("H4.sensitivity.")
                or n == "H4.t2_spatial_only"):
            r["role"] = "design_time_evidence"
            r["note"] += FIX
        if n.startswith("H4.sensitivity.") and n.count(".") == 2:
            sens_names.add(n.split(".")[2])
        if n == "H4.sensitivity.all_ingrid_counties" and r["source"] == ELEC:
            r["note"] = ("per-pump Chow-Lin kWh of Yunlin, Changhua and in-grid Chiayi kept pumps "
                         "(pumps_yunlin.npz, pumps_changhua.npz, pumps_chiayi_ingrid.npz) -> Q_j "
                         "[note corrected in the H4 repair round]")
        out.append(r)
    for name in sorted(sens_names):
        num = f"H4.sensitivity.{name}.multipliers"
        out.append({"number_id": num, "source": HEADS, "role": "fit_target",
                    "product_version": PV1, "wp": "H4",
                    "note": "v1 member multipliers fitted on 2012-02..2019-12 heads (flags 0-1) "
                            "[added in the H4 repair round]"})
        for s, note in ((ENGINE, "hindcast, T, S, pump split"), (ELEC, "per-pump kWh"),
                        (CENSUS, "pump coordinates")):
            out.append({"number_id": num, "source": s, "role": "input", "product_version": PV1,
                        "wp": "H4", "note": note})
    return out


def v2_rows() -> list[dict]:
    r = []

    def add(num, src, role, note):
        r.append({"number_id": num, "source": src, "role": role, "product_version": PV2,
                  "note": note})

    inputs = [(ENGINE, "base hindcast h_cell (cwa theta), zone T and S, pump split, conversion"),
              (ELEC, "per-pump Chow-Lin kWh of Yunlin, Changhua and in-grid Chiayi kept pumps "
                     "-> Q_j"),
              (CENSUS, "pump coordinates (kept pumps)"),
              (GRID, "model grid (x0, y0, dx, mask): cell of each well, cell-mean kernels")]

    def inp(num, units=False):
        for s, n in inputs:
            add(num, s, "input", n)
        if units:
            add(num, UNITS, "input", "fresh.folds.wells_all partition read as data")

    add("H4v2.multipliers", HEADS, "fit_target",
        "158 calibration wells, heads_monthly flags 0-1, rows 2012-02..2019-12")
    inp("H4v2.multipliers")
    add("H4v2.multipliers.t2_folds", HEADS, "fit_target",
        "per A1 site fold, calibration wells' 2012-02..2019-12 rows")
    inp("H4v2.multipliers.t2_folds", units=True)
    for k in ("F1", "F2", "F3", "F4"):
        num = f"H4v2.{k}"
        add(num, HEADS, "validator",
            "held-out wells (not calibration wells; fresh for H4), heads_monthly flags 0-1, "
            "2012-02..2022-12 rows are the scored observations (own-mean datum); the "
            "calibration wells' 2012-19 rows set the multipliers (H4v2.multipliers)")
        inp(num)
    for k in ("C1", "C2", "C3", "C4"):
        num = f"H4v2.{k}"
        add(num, HEADS, "design_time_evidence",
            "v1 unit (158 wells, 2020-22 scored; 2012-19 multipliers, datum and climatology); "
            "post-look: necessary condition only")
        inp(num, units=k in ("C2", "C4"))
    add("H4v2.reported.v1unit_placebos", HEADS, "design_time_evidence",
        "placebo p of T1/T2 on the v1 unit (post-look)")
    inp("H4v2.reported.v1unit_placebos", units=True)
    add("H4v2.reported.F2_heldout_years", HEADS, "validator",
        "held-out wells with 2020-22 data: 2012-19 datum/climatology, 2020-22 scored")
    inp("H4v2.reported.F2_heldout_years")
    sc = DATAPREP / "reports/H4_evidence/repair/registered/scores.json"
    names = sorted(json.loads(sc.read_text())["sensitivity"]) if sc.exists() else []
    for name in names:
        num = f"H4v2.sensitivity.{name}"
        add(num, HEADS, "design_time_evidence",
            "member scores on the held-out wells and the v1 unit (reported, never gating)")
        inp(num, units=True)
        add(num + ".multipliers", HEADS, "fit_target",
            "member multipliers, calibration wells 2012-02..2019-12")
        inp(num + ".multipliers")
        if name == "recorded_arm":
            add(num, C0, "input", "recorded arm per pump (TPC even split), all three counties")
    for s, n in inputs:
        add("H4v2.design.d1", s, "design_time_evidence",
            n + " (d1_design_v2: kernel split, unit regressors, support; no head data)")
    add("H4v2.design.d1", f"{HEADS}/station_meta_preorigin.parquet", "design_time_evidence",
        "held-out candidates (coordinates, layer codes)")
    add("H4v2.design.availability", HEADS, "design_time_evidence",
        "month counts (sid, month, flag columns) of the held-out candidates; no value read")
    add("H4v2.design.t1_power", ENGINE, "design_time_evidence",
        "synthetic heads from the hindcast and the operator; real availability masks")
    for name, src, note in (
            ("census_attributes", f"{TPC}/yunlin_census.parquet",
             "2026 census attribute table via census_L1/v3 and yunlin_electricity/preorigin/v2"),
            ("census_attributes_changhua", f"{TPC}/changhua_census.parquet",
             "2026 census attribute table (Changhua) via census_L1/v3"),
            ("census_attributes_chiayi", f"{TPC}/chiayi_census.parquet",
             "2026 census attribute table (Chiayi) via census_L1/v3"),
            ("residual_unpaired_dec2022", f"{TPC}/yunlin_kwh_long.parquet",
             "Dec-2022 reading of unidentified Dec-Jan-paired meters, via "
             "yunlin_electricity/preorigin/v2"),
            ("residual_unpaired_dec2022_changhua", f"{TPC}/changhua_kwh_long.parquet",
             "as above (Changhua)"),
            ("residual_unpaired_dec2022_chiayi", f"{TPC}/chiayi_kwh_long.parquet",
             "as above (Chiayi)"),
            ("zero_check_gate",
             "results/dataprep/reports/P2abc_evidence/repair3/tests/t6_zero_shipped.json",
             "P2abc registered zero check (gate input of yunlin_electricity/preorigin/v2)")):
        add(f"H4v2.exemption.{name}", src, "input", "exemption (U3) inherited: " + note)
    return r


def main() -> None:
    frag = ledger.fragment_path("H4", LEDGER_ROWS)
    SUPERSEDED.parent.mkdir(parents=True, exist_ok=True)
    if not SUPERSEDED.exists():
        shutil.copy2(frag, SUPERSEDED)
    with open(SUPERSEDED, newline="", encoding="utf-8") as f:
        r1 = list(csv.DictReader(f))
    with open(frag, newline="", encoding="utf-8") as f:
        cur = list(csv.DictReader(f))
    later = cur[len(r1):]                      # rows appended after round 1 (e.g. v2 prereg)
    if cur[:len(r1)] != r1 and any(r["number_id"].startswith("H4.A") and r["role"] ==
                                   "design_time_evidence" for r in cur):
        rows = cur                             # already corrected: only append v2 rows
    else:
        rows = corrected_v1(r1) + later
    bad = ledger.conflicts(rows)
    if bad:
        raise SystemExit(f"conflicts in corrected rows: {bad}")
    tmp = frag.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ledger.COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in ledger.COLUMNS})
    tmp.replace(frag)
    path = ledger.append_rows("H4", v2_rows(), rows_dir=LEDGER_ROWS)
    allrows = ledger._read(path)
    print(json.dumps({"fragment": str(path), "round1_rows": len(r1), "total": len(allrows),
                      "conflicts": ledger.conflicts(allrows)}, indent=1))


if __name__ == "__main__":
    main()
