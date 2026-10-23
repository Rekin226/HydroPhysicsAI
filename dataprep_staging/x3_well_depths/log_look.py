"""Log WP X3's one look at the design-time heads (run once, before the heads are opened).

The head-based layer-response analysis (head_response.py) reads the out-of-fold R0/R1c
observation and prediction matrices, 2012-02..2022-12 only. No head after 2022-12 is read.

Run from the repository root:
    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/x3_well_depths/log_look.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import x3const as K  # noqa: E402,F401  (puts dataprep_staging on sys.path)
from harness import prereg3  # noqa: E402


def main() -> dict:
    look = prereg3.record_look(
        ["I1:design_oof_heads_R0_R1e"], "heads:2012-01..2022-12", "I1", "after_the_fact",
        note="WP X3 (task X3, 2026-10-21): head-based estimate of which aquifer the local "
             "census electricity stress hits (per-well lagged response and nested-site "
             "two-layer mass balance on stage3_datum_gate_cwa/ccz per-entry obs and pred, "
             "2012-02..2022-12); a prior for the Changhua pump-split rung, report only. "
             "results/dataprep/reports/X3_well_depths/report.json")
    return look


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, default=str)[:600])
