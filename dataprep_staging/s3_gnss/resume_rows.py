"""WP S3 resume round: ledger rows for the numbers the resume checks report.

    python -I dataprep_staging/s3_gnss/resume_rows.py

Appends to ``results/dataprep/ledger/rows/S3.csv`` through ``common.ledger.append_rows``
(exact duplicates are skipped; a second role for a source raises before writing).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ledger  # noqa: E402

PRE = "results/dataprep/gnss_clean/preorigin/v1"
FULL = "results/dataprep/gnss_clean/full/v1"
ACC = "results/dataprep/reports/S3_evidence/acceptance"
RES = "results/dataprep/reports/S3_evidence/resume_2026-10-12"
PV = "gnss_clean/preorigin/v1"
RING_SOURCES = ["chou-shui-data/data/ls_cache/clean",
                "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__TKJS_281_29.parquet",
                "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__NTUH_282_29.parquet",
                "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__SJES.parquet",
                "chou-shui-data/data/ls_cache/ls-wra-dbm-obs__STES.parquet"]


def rows() -> list[dict]:
    out = []

    def add(nid, src, role, pv=PV, note=""):
        out.append({"number_id": nid, "source": src, "role": role, "product_version": pv,
                    "note": note})

    for nid in ("S3.resume.A2_repro.max_abs_diff_registered_vs_gate",
                "S3.resume.A2_repro.max_abs_diff_current_vs_gate",
                "S3.resume.A2_repro.n_values"):
        add(nid, PRE, "input", note="published pre-origin tables (wra/daily, common_mode, "
                                    "stations)")
        add(nid, f"{ACC}/registered_code/validate.py", "input",
            note="registered validate.py recovered byte-exact (sha256 461cd6a1...)")
        add(nid, f"{ACC}/validation_preorigin.json", "input", note="gate run's A2")
        for s in RING_SOURCES:
            add(nid, s, "validator")
    for nid in ("S3.resume.A1_repro.max_abs_rate_diff", "S3.resume.A1_repro.rows_differing"):
        add(nid, f"{ACC}/injection_trials_preorigin.csv", "input",
            note="synthetic steps seed 303 (archived trial table)")
        add(nid, f"{ACC}/validation_preorigin.json", "input", note="gate run's A1")
    add("S3.resume.checker.files_rehashed", PRE, "input")
    add("S3.resume.checker.files_rehashed", FULL, "input", pv="gnss_clean/full/v1")
    add("S3.resume.checker.files_rehashed", "data_fetch_api/new_measurements/ngl_gnss",
        "input", pv="L0")
    add("S3.resume.guard_probes_identical", PRE, "input")
    add("S3.resume.guard_probes_identical", FULL, "input", pv="gnss_clean/full/v1")
    add("S3.resume.guard_probes_identical", f"{ACC}/guard_probe.json", "input",
        note="archived probe results")
    add("S3.resume.prereg2_audit", "results/dataprep/preregistration/looks.jsonl", "input",
        pv="n/a", note=f"read-only audit, {RES}/prereg2_audit_S3.json")
    return out


def main() -> int:
    p = ledger.append_rows("S3", rows())
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
