"""Ledger rows for the numbers quoted about release_20261012b2 (plan 4.1, rule U6).

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release_b2.repair_ledger

The numbers are those of the release b2 section of docs/DATA_REMEDIATION_STATUS_2026-10.md
and of REBUILD_CHECK.json: the rebuild check, the file-by-file comparison with
release_20261012b, the checker and guard probes, the inventory, and the context numbers of user
ruling (a) that ledger_rows does not already cover. Appends one row per source per number to
``results/dataprep/ledger/rows/L3.csv`` (exact duplicates skipped, conflicts refused), then
reassembles ``results/dataprep/ledger/data_roles.csv`` from every WP fragment.
"""
from __future__ import annotations

from common import ledger

from l3_release_b2 import config as C

EV = "results/dataprep/reports/L3_evidence/repair_b2"
REL = f"{C.RELEASE_DIR}/{C.RELEASE}"
OLD = f"{C.RELEASE_DIR}/{C.SUPERSEDES}"
PV = f"model_ready/{C.RELEASE}"

NUMBERS = {
    # number id suffix: (sources, note)
    "rebuild_files": (
        [f"{EV}/compare.json", f"{REL}/REBUILD_CHECK.json"],
        "77/77 files identical over builds A and B from scratch; final tree equals build A"),
    "diff_vs_release_b": (
        [f"{EV}/regress_final.json", OLD],
        "against release_20261012b: 33 files identical, 44 differ (RELEASE.json, "
        "PRODUCT_CARD.md, 18 MANIFEST.json and 18 validation.json with the release name, b2 "
        "code hashes and git commit, 6 re-packed npz equal except release_component); content "
        "changes only in preorigin/surface_water and preorigin/piles_printed"),
    "piles_rows_kept": (
        [f"{REL}/preorigin/piles_printed/v1", f"{OLD}/preorigin/piles_printed/v1"],
        "pre-origin pile table: 231 rows, byte-identical to release b's"),
    "guard_probes": (
        [f"{EV}/check_final.json"],
        "306 guard probes over 18 manifests (17 uses each), 0 mismatches"),
    "checker": (
        [f"{EV}/common_check_{C.RELEASE}.json", f"{EV}/common_check_{C.SUPERSEDES}.json"],
        "A0 checker: 18 manifests, 0 failures, 385 files re-hashed, in each release"),
    "inventory": (
        [f"{EV}/inventory.json"],
        "13 upstream versions included, 81 product versions excluded, 0 unclassified"),
    "ruling_a_r0_changhua": (
        ["results/dataprep/reports/PHASE0_STATUS.json",
         "results/twin_runs/stage3_datum_gate_cwa"],
        "context of ruling (a): R0 Changhua out-of-fold anomaly RMSE 1.074 m (R1b 1.264 m "
        "still fails this gate)"),
    "ruling_a_w1_head_check": (
        ["results/dataprep/reports/W1.json"],
        "W1 supporting head check (registered not gated) failed in Yunlin layer 1, -0.0087"),
}


def rows() -> list[dict]:
    out = []
    for suffix, (sources, note) in NUMBERS.items():
        for s in sources:
            out.append({"number_id": f"L3.{C.RELEASE}.b2.{suffix}", "source": s,
                        "role": "input", "product_version": PV, "note": note})
    return out


def main() -> int:
    p = ledger.append_rows(C.LEDGER_WP, rows())
    ids = [f"L3.{C.RELEASE}.b2.{k}" for k in NUMBERS]
    print(f"appended to {p}; unledgered {ledger.unledgered(ids)}")
    out, n = ledger.assemble()
    print(f"assembled {out} ({n} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
