"""Declare rung I1.X5CS.v1, kind 'declaration', before any of its columns is fitted: the
county-split (six-zone) compaction column with Ske/Skv capped, refitted on unchanged heads.

Why now (seen first): I1.X5R1.v2_postfailure (exploratory) found the six-zone column recovers
most of the leveling fit lost with Changhua pumping, but its Changhua zones fit Ske/Skv 1.5-3.6
(Q0's three zones 0.02-0.6) and the forward run rises after 2022 in every split arm, the Q0
control included. The cap is a physical bound chosen from soil mechanics, not from those
numbers; the follow-up's results were seen before it was chosen, so the rung is design-time
and promotion needs the seed-2029 confirmation.

Usage (repo root, under the X5 lock):
    flock results/dataprep/preregistration/.x5.lock python dataprep_staging/x5_energy_volume/prereg_x5cs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import prereg3  # noqa: E402
from uplift import uplift_share  # noqa: E402

NA = prereg3.not_applicable
RATE_TOL = 0.1          # cm/yr: a cell "rises" if its 2023-25 rate is below -RATE_TOL


def doc() -> dict:
    base = json.loads((REPO / "results/dataprep/preregistration/I1.B.v1.json").read_text())
    gates = base["thresholds"]["passes_against_Q0_if_all"][:6]
    u = uplift_share("q0", tol=RATE_TOL)          # Q0 is spent: design-time reference
    d = {k: base[k] for k in ("burn_in", "folds", "gate_scores", "zones")}
    d.update({
        "id": "I1.X5CS.v1", "wp": "I1", "kind": "declaration",
        "title": "Rung X5CS: county-split six-zone compaction column with Ske/Skv capped, "
                 "refitted on unchanged heads (Q0 primary; q0c reported)",
        "priors": (
            "I1.X5R1.v1 FAIL (Changhua pumping breaks the three-zone column's leveling fit); "
            "I1.X5R1.v2_postfailure PARTIAL: six-zone column leveling R2 out of fold q0 0.697, "
            "q0c 0.597, x5r1 0.598; Changhua 0.24 / 0.23 / 0.24; Changhua Ske/Skv 1.5-3.6 in "
            "the split arms; 2023-25 Changhua mean rate -0.10 / -0.26 / -0.08 cm/yr (uplift). "
            f"Q0's own forward: share of cells with 2023-25 rate < -{RATE_TOL} cm/yr is "
            f"{u['uplift_share_ch']:.3f} in Changhua and {u['uplift_share_rest']:.3f} elsewhere "
            "(design-time reference). Cap values: 1 = the bound under which elastic storage "
            "never exceeds inelastic (primary); 0.1 = the upper end of the clay-aquitard range "
            "in the subsidence literature (Ske/Skv ~0.01-0.1; declared alternative)."),
        "arms": {
            "q0_csk1": "Q0 heads, six-zone column, --ske-skv-max 1 (PRIMARY, gated)",
            "q0_csk01": "Q0 heads, six-zone column, --ske-skv-max 0.1 (declared alternative, "
                        "gated only if q0_csk1 fails)",
            "q0c_csk1": "q0c heads (Q0 + Changhua census), cap 1: reported only",
            "q0c_csk01": "q0c heads, cap 0.1: reported only"},
        "recipe": (
            "theta: the follow-up's split theta (stage3_datum_gate_<q0|q0c>_cs, heads checked "
            "identical to the unsplit theta, max |dh| < 1e-12 m) copied to "
            "stage3_datum_gate_<arm>; column: calibrate_coupled --target leveling --configs "
            "zonal --hpc0-guard-days 365 --seed 2027 --device cuda --ske-skv-max <cap>; "
            "forward: batch 27's _forward. Head metrics (kfold, county corr and mean RMSE, "
            "temporal) are the parent run's, since the heads are unchanged. Code: main "
            "5f3af18, worktree ~/HydroPhysicsAI-x5run (unchanged)."),
        "primary_model": "q0_csk1",
        "inputs": ["results/dataprep/preregistration/I1.B.v1.json",
                   "results/dataprep/preregistration/I1.X5R1.v2_postfailure.json",
                   "results/dataprep/reports/I1_X5R1_followup/report.json",
                   "dataprep_staging/x5_energy_volume/uplift.py"],
        "tests": NA("design-time partition seed 2027; promotion needs the seed-2029 "
                    "confirmation"),
        "fresh_units": NA("see the confirmation rule"),
        "thresholds": {
            "passes_against_Q0_if_all": gates + [
                "leveling out-of-fold R2 >= Q0 + 0.02 (the split's extra zone sets must earn "
                "their place, not merely not hurt)",
                "Changhua leveling R2 (county:changhua, out of fold) >= Q0's",
                f"no added ground rising: in each county, share of cells with 2023-25 forward "
                f"rate < -{RATE_TOL} cm/yr <= Q0's + 0.05 (Changhua <= "
                f"{u['uplift_share_ch'] + 0.05:.3f}, elsewhere <= "
                f"{u['uplift_share_rest'] + 0.05:.3f})"],
            "disclosed": "the kfold, county corr, county mean RMSE and temporal gates pass by "
                         "construction for Q0-head arms (unchanged heads); the q0c arms fail "
                         "cmrmse_ch by construction (seen in I1.X5R1.v1), so they are "
                         "reported only, with the column gates read beside them",
            "promotion": ["only if every gate passes for q0_csk1 (or, if it fails, for "
                          "q0_csk01): refit Q0's three-zone column and the passing arm's "
                          "column with --seed 2029 on the same heads and rescore the column "
                          "gates (leveling, Changhua leveling, no added rising, plausibility)"],
            "reported_not_gating": [
                "Ske/Skv per column zone; leveling R2 by zone; fan subsidence 2023-32; P2f",
                "q0c arms: every metric and the column gates' reading"]},
        "attempt_policy": {
            "scored_runs": "one per arm on seed 2027; no rerun after any gate metric is seen",
            "engineering_retries": "free before any metric is written, each logged",
            "pre_declared_alternatives": ["q0_csk01"],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True, "report": "every attempt, failures included"},
        "interpretation": (
            "A pass says the county-split column with physical storage is a better column for "
            "the current heads (higher leveling fit, no added rising) and is the column a "
            "later Changhua-pumping rung should use. The q0c arms say whether the cap keeps "
            "the follow-up's leveling recovery without the uplift; they cannot pass because "
            "q0c's Changhua heads fail cmrmse_ch. A fail with leveling below Q0 + 0.02 says "
            "the split's gain on Q0 heads came from the unphysical elastic storage."),
        "no_prior_look_statement": "No capped six-zone column exists; only the follow-up's "
                                   "uncapped columns and Q0's forward have been seen.",
        "written_by": "dataprep_staging/x5_energy_volume/prereg_x5cs.py",
    })
    return d


def main() -> int:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(errs)
    print(*prereg3.write_prereg(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
