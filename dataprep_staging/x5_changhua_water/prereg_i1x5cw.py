"""Declare rung I1.X5CW.v1 (route 4: Changhua irrigation water outside the Jiji supply), kind
'declaration', BEFORE rung I1.X5R1.v1 (batch 27) has been scored, so the base-model rule below
is fixed without knowing which way it goes.

Base rule (fixed now): if results/dataprep/reports/I1_X5R1/report.json shows x5r1 passing every
gate, the base recipe is batch 27's x5r1 (Q0 + four-county census with the corrected
irrigation conversion); else, if x5r1t passes every gate, x5r1t; else Q0 (par20 RECIPE).

Arms on that base: ``cws`` (primary) adds ``--sw-recharge x5_t2_service.npz --sw-components
delivered`` (canal service-map placement); ``cwp`` (pre-declared alternative placement, all
Changhua paddy) the same with ``x5_t2_paddy.npz``. The water term enters as model input only;
one learned recharge fraction (log_sw_scale) is added.

Usage (repo root, under the X5 lock):
    flock results/dataprep/preregistration/.x5.lock python dataprep_staging/x5_changhua_water/prereg_i1x5cw.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import prereg3  # noqa: E402
from interannual_metrics import changhua_interannual  # noqa: E402

NA = prereg3.not_applicable
CW = "results/dataprep/x5_changhua_water/preorigin/v1"


def doc() -> dict:
    base = json.loads((REPO / "results/dataprep/preregistration/I1.B.v1.json").read_text())
    gates = base["thresholds"]["passes_against_Q0_if_all"][:6]
    q0 = changhua_interannual("q0")          # Q0 is spent: design-time reference numbers
    a_q0, r_q0 = q0["amplitude_A"], q0["interannual_corr"]
    d = {k: base[k] for k in ("burn_in", "folds", "gate_scores", "zones")}
    d.update({
        "id": "I1.X5CW.v1", "wp": "I1", "kind": "declaration",
        "title": "Rung X5CW: Changhua irrigation water outside the Jiji supply (MOA office use "
                 "minus Jiji Changhua delivery, about 3.5e8 m3/yr 2016-22) as canal recharge; "
                 "service-map placement primary, all-paddy placement as the declared alternative",
        "base_rule": (
            "Fixed before I1.X5R1.v1 is scored: base = batch 27's x5r1 recipe if "
            "results/dataprep/reports/I1_X5R1/report.json has passes_all.x5r1 true; else x5r1t's "
            "if passes_all.x5r1t is true; else Q0's (results/twin_runs/par20/jobs.sh RECIPE). The "
            "chosen base and the report's sha256 are written into the batch's jobs.sh header."),
        "priors": (
            "Inferred-input rung (user direction 2026-10-24/25). Route 4 (X5.changhua_water.v1): "
            "annual volume = MOA Changhua irrigation-office use minus the Jiji weir Changhua "
            "delivery (held data; 2016-22 mean 3.50e8 m3/yr, 5.6-8.9e8 in 2012-15, partly a 2016 "
            "reporting change; definitional sd 2.62e8), timed by the crop calendar x exp(b.z) with "
            "z = LiCSAR coherence and river balance (no heads), Denton-constrained to the annual "
            f"totals. Fields: {CW}/x5_t2_service.npz (canal service map; 777 of 782 non-zero cells "
            f"on Changhua) and {CW}/x5_t2_paddy.npz (all Changhua paddy, 878 cells); both load on "
            "the model grid (inputs.load_sw_recharge). Arms add --sw-recharge <field> "
            "--sw-components delivered to the base recipe; one learned recharge fraction "
            "log_sw_scale in [0.01, 2]. Forward runs past 2022 use the field's month-of-year "
            "climatology. Code: main 5f3af18, worktree ~/HydroPhysicsAI-x5run."),
        "primary_model": "cws (stage3_datum_gate_cws); cwp is the pre-declared alternative "
                         "placement, reported beside it, never substituted",
        "inputs": ["results/dataprep/preregistration/I1.B.v1.json",
                   "results/dataprep/preregistration/I1.X5R1.v1.json",
                   "results/dataprep/reports/X5.changhua_water.v1/report.json",
                   "results/dataprep/reports/X5.changhua_water.v1/rung_spec_DRAFT.json",
                   f"{CW}/MANIFEST.json", f"{CW}/x5_t2_service.npz", f"{CW}/x5_t2_paddy.npz",
                   "dataprep_staging/x5_changhua_water/interannual_metrics.py",
                   "dataprep_staging/i1/diag_changhua_interannual.py"],
        "tests": NA("design-time partition seed 2027; promotion needs a confirmation on the "
                    "never-scored partition seed 2029"),
        "fresh_units": NA("see the confirmation rule"),
        "thresholds": {
            "passes_against_Q0_if_all": gates + [
                f"Changhua interannual amplitude: |ln A_rung| < |ln A_Q0|, A_Q0 = {a_q0:.4f} "
                f"(n {q0['n_amp']}; dataprep_staging/x5_changhua_water/interannual_metrics.py: "
                "median over Changhua held-out entries with >= 60 paired months of "
                "nanstd(interannual(pred)) / nanstd(interannual(obs)))",
                f"Changhua interannual correlation >= Q0 - 0.02 = {r_q0 - 0.02:.4f} (Q0 "
                f"{r_q0:.4f}, n {q0['n_corr']}; median per-entry corr, >= 24 paired months)"],
            "promotion": ["only if every gate passes for cws (or, if cws fails, for cwp as the "
                          "pre-declared alternative): rerun Q0, the base (if not Q0) and the "
                          "passing arm on the never-scored partition (calibrate_flow --seed 2029 "
                          "--fold-site-tol-m 200, column --seed 2029) with the same gates"],
            "reported_not_gating": [
                "the same metrics for the base arm when the base is not Q0 (attribution)",
                "fitted sw_scale (at the 0.01 bound = the model rejects the term)",
                "P2f Yunlin ratio; amplitude and correlation per county; leveling R2 by county",
                "forward fan 2023-32 subsidence"]},
        "attempt_policy": {
            "scored_runs": "one per arm on seed 2027; no rerun after any gate metric is seen",
            "engineering_retries": "free before any metric is written, each logged in "
                                   "gap_queue.log",
            "pre_declared_alternatives": ["cwp"],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included"},
        "interpretation": (
            "Disclosed in advance: R1e-cn put both Changhua canal fractions at the 0.01 bound "
            "and the model may reject this term too; a steady extra recharge damps the head "
            "response, and Q0's Changhua amplitude is already a third of observed, so damping "
            "alone moves the amplitude AWAY from 1; only the term's year-to-year variation (the "
            "2012-15 level vs after 2016) can add interannual forcing. The monthly shape is "
            "essentially the crop calendar. A pass says the held annual volume of Changhua's "
            "non-Jiji water improves Changhua's interannual head response without losing fit; "
            "a fail with sw_scale at its bound says the model cannot use it as recharge."),
        "no_prior_look_statement": "No twin run with these fields exists; I1.X5R1.v1 has not "
                                   "been scored; no gate metric of any arm has been computed.",
        "written_by": "dataprep_staging/x5_changhua_water/prereg_i1x5cw.py",
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
