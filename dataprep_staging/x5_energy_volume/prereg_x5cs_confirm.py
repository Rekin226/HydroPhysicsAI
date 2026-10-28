"""Declare I1.X5CS.v1's seed-2029 column confirmation, kind 'declaration', before either
seed-2029 column is fitted: the promotion step I1.X5CS.v1 declared for its passing arm.

I1.X5CS.v1 (scored 2026-10-28, report results/dataprep/reports/I1_X5CS/report.json):
q0_csk1 failed only the Changhua rising share (0.207 > 0.193); q0_csk01, gated because
q0_csk1 failed, passed every gate. This fixes the confirmation's gates as numbers-free
comparisons against Q0's three-zone column refitted on the same new partition.

Usage (repo root, under the X5 lock):
    flock results/dataprep/preregistration/.x5.lock python dataprep_staging/x5_energy_volume/prereg_x5cs_confirm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import prereg3  # noqa: E402
from prereg_x5cs import RATE_TOL  # noqa: E402

NA = prereg3.not_applicable


def doc() -> dict:
    return {
        "id": "I1.X5CS.v1.confirm2029", "wp": "I1", "kind": "declaration",
        "title": "Seed-2029 column confirmation of I1.X5CS.v1's passing arm (q0_csk01, "
                 "six-zone county-split column, Ske/Skv <= 0.1) against Q0's three-zone column",
        "priors": (
            "I1.X5CS.v1 on seed-2027 leveling folds: q0_csk01 leveling R2 out of fold 0.682 vs "
            "Q0 0.653 (bar 0.673); Changhua 0.204 vs 0.080; rising share (2023-25 rate < "
            f"-{RATE_TOL} cm/yr) Changhua 0.000 vs 0.143, elsewhere 0.499 vs 0.564; 2023-25 "
            "county mean rate +0.29 / +0.50 cm/yr. q0_csk1 failed the Changhua rising share "
            "only. The cap was chosen after the uncapped follow-up was seen, so the seed-2027 "
            "pass is design-time; this confirmation is the declared promotion step."),
        "arms": {
            "q0_s29": "Q0 heads, Q0's three-zone column recipe (no split, no cap), --seed 2029",
            "q0_csk01_s29": "Q0 heads, q0_csk01's six-zone split theta, --ske-skv-max 0.1, "
                            "--seed 2029"},
        "recipe": (
            "theta copied unchanged from stage3_datum_gate_q0 and stage3_datum_gate_q0_csk01 "
            "to stage3_datum_gate_<arm>; column: calibrate_coupled --target leveling --configs "
            "zonal --hpc0-guard-days 365 --seed 2029 --device cuda [--ske-skv-max 0.1 for "
            "q0_csk01_s29]; forward: I1.X5CS.v1's (batch 27's _forward). Code: main 5f3af18, "
            "worktree ~/HydroPhysicsAI-x5run (unchanged). Scorer: "
            "dataprep_staging/x5_energy_volume/score_x5cs_confirm.py, written before any "
            "seed-2029 metric exists."),
        "zones": "Q0's three column zones (q0_s29); the six county-split zones (q0_csk01_s29)",
        "burn_in": "unchanged",
        "folds": "heads: Q0's (calibrate_flow seed 2027, unchanged); leveling site folds "
                 "calibrate_coupled --seed 2029 (new partition, not seen)",
        "gate_scores": NA("pass/fail column gates only; no weighted score"),
        "primary_model": "q0_csk01_s29",
        "inputs": ["results/dataprep/preregistration/I1.X5CS.v1.json",
                   "results/dataprep/reports/I1_X5CS/report.json"],
        "tests": NA("fresh leveling-fold partition (seed 2029) of the same benchmarks; heads "
                    "and forward unchanged"),
        "fresh_units": NA("the partition is new; the benchmarks are not"),
        "thresholds": {
            "passes_if_all": [
                "leveling out-of-fold R2 (seed 2029) >= q0_s29's + 0.02",
                "Changhua leveling R2 (county:changhua, seed 2029) >= q0_s29's",
                f"rising share (2023-25 forward rate < -{RATE_TOL} cm/yr) <= q0_s29's + 0.05 "
                "in Changhua and elsewhere, each",
                "2023-25 county mean forward rate >= 0 in Changhua and elsewhere"],
            "on_pass": "q0_csk01's column (six zones, Ske/Skv <= 0.1) is promoted as the "
                       "column for Q0's heads and is the column a later Changhua-pumping rung "
                       "uses",
            "on_fail": "no promotion; Q0's three-zone column stays",
            "reported_not_gating": ["seed-2027 vs seed-2029 change of each column metric",
                                    "Ske/Skv per zone; leveling R2 by zone; fan subsidence "
                                    "2023-32; P2f"]},
        "attempt_policy": {
            "scored_runs": "one per arm on seed 2029; no rerun after any metric is seen",
            "engineering_retries": "free before any metric is written, each logged",
            "pre_declared_alternatives": [], "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True, "report": "every attempt, failures included"},
        "interpretation": (
            "A pass says the capped six-zone column's gain over the three-zone column is not "
            "an artefact of one leveling partition. A fail on leveling says the seed-2027 gain "
            "came from that partition."),
        "no_prior_look_statement": "No seed-2029 column exists for Q0's heads.",
        "written_by": "dataprep_staging/x5_energy_volume/prereg_x5cs_confirm.py",
    }


def main() -> int:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(errs)
    print(*prereg3.write_prereg(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
