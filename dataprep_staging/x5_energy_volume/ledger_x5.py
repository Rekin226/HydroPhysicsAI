"""Ledger rows (U6) for the numbers the X5 energy-to-volume route reports.

Run under the shared lock, after the reports exist:
    flock results/dataprep/preregistration/.x5.lock python -I ledger_x5.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402

from x5_energy_volume import x5const as K  # noqa: E402

NOTE = "WP X5 energy-to-volume (2026-10-25)"


def rows() -> list[dict]:
    r0 = rel_to_repo(K.REPORT_DIR["step0_ab"] / "report.json")
    rt = rel_to_repo(K.REPORT_DIR["test"] / "report.json")
    srcs_a = [(rel_to_repo(K.WRA_TXT), "input"), (rel_to_repo(K.CENSUS_L1), "input"),
              (rel_to_repo(K.L0_PARQ), "input")]
    srcs_c = [(rel_to_repo(K.LS_CACHE), "input"), (rel_to_repo(K.AMP_WELLS), "input"),
              (rel_to_repo(K.CENSUS_L1), "input"), (rel_to_repo(K.BE_CSV), "input"),
              (rel_to_repo(K.X3_CELL_SHARES), "input"), (rel_to_repo(K.STATIONS), "input"),
              (rel_to_repo(K.WRA_TXT), "validator")]
    out = []
    for num, srcs, ver in (("X5.energy_volume.a_m3_per_kwh", srcs_a, r0),
                           ("X5.energy_volume.step0_gap", srcs_a, r0),
                           ("X5.energy_volume.b_m3_per_kwh",
                            [(rel_to_repo(K.CENSUS_L1), "input"),
                             (rel_to_repo(K.L0_PARQ), "input")], r0),
                           ("X5.energy_volume.c_vs_a", srcs_c, rt)):
        for s, role in srcs:
            out.append({"number_id": num, "source": s, "role": role, "product_version": ver,
                        "note": NOTE})
    return out


if __name__ == "__main__":
    print(ledger.append_rows(K.WP, rows()))
