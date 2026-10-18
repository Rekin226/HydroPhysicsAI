"""WP A0 step 5: ledger rows for the numbers A0 reports, then rebuild data_roles.csv.

Usage: python -m common.a0.ledger_rows
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import ledger  # noqa: E402

P = "results/dataprep/_pilots"
TPC_V = "_pilots/tpc_aggregate_electricity@2026-10-12"
PIH_V = "_pilots/physics_inversion_hooks@2026-10-12"
GAP_V = "_pilots/head_gapfill_pilot@2026-10-12"

ROWS = [
    # manifest checker: every recorded hash recomputed
    {"number_id": "A0.check.hashes_reproduced", "source": f"{P}/*/MANIFEST.json",
     "role": "validator", "product_version": "_pilots@2026-10-12",
     "note": "recorded sha256 of pilot code, outputs and named inputs"},
    {"number_id": "A0.check.hashes_reproduced",
     "source": "data_fetch_api/new_measurements/**/SOURCE.json", "role": "validator",
     "product_version": "L0 new_measurements@2026-10-12",
     "note": "recorded bytes and sha256 of every L0 copy"},
    # spot rerun 1: TPC compare_annual.csv
    {"number_id": "A0.rerun.tpc.compare_annual.max_rel_diff",
     "source": "data_fetch_api/new_measurements/tpc_d007012/2026-10-12/d007012.csv",
     "role": "input", "product_version": TPC_V, "note": "L0 copy of the scout's download"},
    {"number_id": "A0.rerun.tpc.compare_annual.max_rel_diff",
     "source": "AMP_V2/data/tpc_pumps.parquet", "role": "input", "product_version": TPC_V,
     "note": "columns selected explicitly; no 戶名1 or 用電地址2"},
    {"number_id": "A0.rerun.tpc.compare_annual.max_rel_diff",
     "source": "AMP_V2/data/pump_kwh_all.parquet", "role": "input", "product_version": TPC_V},
    {"number_id": "A0.rerun.tpc.compare_annual.max_rel_diff",
     "source": f"{P}/tpc_aggregate_electricity/compare_annual.csv", "role": "validator",
     "product_version": TPC_V, "note": "pilot reference output"},
    # spot rerun 2: d_pseudo_changhua.json
    {"number_id": "A0.rerun.pseudo.d_pseudo_changhua.max_rel_diff",
     "source": f"{P}/physics_inversion_hooks/{{inputs_cwa.npz,hindcast_cwa.npz,"
               f"K_pump_f32.npy,M_cwa.npz}}", "role": "input", "product_version": PIH_V,
     "note": "archived pilot inputs (batch-8 fit_full theta; provisional)"},
    {"number_id": "A0.rerun.pseudo.d_pseudo_changhua.max_rel_diff",
     "source": f"{P}/physics_inversion_hooks/d_pseudo_changhua.json", "role": "validator",
     "product_version": PIH_V, "note": "pilot reference output"},
    # spot rerun 3: pilot_final_clean.csv
    {"number_id": "A0.rerun.gapfill.pilot_final_clean.max_rel_diff",
     "source": f"{P}/head_gapfill_pilot/out/{{pilot_[ext_]{{N,S,E,H}}.parquet,"
               f"monthly_long.parquet,well_meta.csv}}", "role": "input",
     "product_version": GAP_V, "note": "archived pilot predictions and matrices"},
    {"number_id": "A0.rerun.gapfill.pilot_final_clean.max_rel_diff",
     "source": f"{P}/audit_2026-10-11/artefact_events.csv", "role": "input",
     "product_version": "_pilots/audit_2026-10-11@2026-10-12"},
    {"number_id": "A0.rerun.gapfill.pilot_final_clean.max_rel_diff",
     "source": f"{P}/head_gapfill_pilot/out/pilot_final_clean.csv", "role": "validator",
     "product_version": GAP_V, "note": "pilot reference output"},
]


def main() -> int:
    p = ledger.append_rows("A0", ROWS)
    out, n = ledger.assemble()
    print(f"fragment {p}; assembled {out} with {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
