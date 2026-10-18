"""Write the P4.0 repair-round declaration before ``build_engine_v2`` computes R1-R3.

    cd dataprep_staging
    python -m p4_changhua.prereg_p40_repair

A numerical verification of the tangent-linear model (``engine_tlm``) against the stepwise
sparse twin, not a test scored on data units, so it is a declaration (``harness.prereg2``):
cases, metrics, thresholds and the interpretation are fixed here. It is not part of the
P4.0 acceptance (``P40.engine_acceptance.v1`` stands unchanged).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg2  # noqa: E402
from harness.prereg import not_applicable as NA  # noqa: E402

THETA = "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
TLM = "dataprep_staging/p4_changhua/engine_tlm.py"

DOC = {
    "id": "P40.repair_linearisation.v1",
    "wp": "P40",
    "title": "P4.0 repair round: tangent-linear (lift-feedback) linearisation of the sparse "
             "twin vs the stepwise twin",
    "kind": "declaration",
    "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP P4.0 (forcing-uncertainty "
                    "propagation Var h = K Sigma_q K^T) and section 4.8; verifier minor "
                    "issues on engine.energy_rate_jacobian and the Var h claims",
    "tests": NA("numerical verification of a linearisation against the twin; no data units"),
    "zones": NA("no zonation is chosen"),
    "priors": NA("no fit"),
    "burn_in": NA("no fit; rollouts start from the calibration's own h0 (2012-01)"),
    "folds": NA("no folds"),
    "fresh_units": NA("no scoring on observations"),
    "primary_model": "batch 8's merged cwa theta " + THETA + " (sha256 in the inputs), "
                     "sparse twin of p4_changhua.engine, 131 monthly steps 2012-02..2022-12",
    "inputs": [THETA, TLM],
    "thresholds": {
        "R1_tlm_vs_central_difference": {
            "cases": ["energy: direction = the spread synthetic Changhua energy (188 GWh/yr, "
                      "uniform over Changhua cells, Yunlin monthly shape), about the path that "
                      "includes it; eps 1e-4",
                      "volumetric: direction = an independent abstraction-rate field with sd "
                      "0.3 x the base rate (seed 2029), about the base path (lift feedback of "
                      "the base census pumping active); eps 1e-3"],
            "metric": "max |dh_TLM - dh_FD| / max |dh_FD| over all layers, cells and steps, "
                      "dh_FD = (h(+eps) - h(-eps)) / (2 eps) from SparseTwin.rollout",
            "threshold": "<= 1e-6 in each case",
            "reported_not_gated": "the frozen-lift response (the stored kernels' "
                                  "linearisation) against the same finite difference"},
        "R2_energy_error_variance": {
            "case": "the 39 energy-error factors of engine/v1 (independent 10 % zone-year "
                    "level errors and 10 % errors on each zone's annual cycle of the synthetic "
                    "Changhua abstraction), about the path that includes it",
            "metric": "median over well-months (variance > 1e-6 of the max) of V_MC / V_TLM; "
                      "V_TLM = diag(U U^T) with U the tangent-linear factor responses; V_MC "
                      "from 200 fresh draws (seed 2027) through SparseTwin.rollout",
            "threshold": "inside the chi2 95 % band for 199 degrees of freedom, "
                         "[0.8132, 1.2058]",
            "reported_not_gated": "V_MC / V_frozen (frozen-lift K Sigma K^T), V_TLM / "
                                  "V_frozen, and the per-draw rms of (MC response - TLM "
                                  "response) relative to the TLM response"},
        "R3_volumetric_noise_variance": {
            "case": "independent cell-month abstraction-rate errors, sd 0.3 x the base rate, "
                    "about the base path (lift feedback of the base census pumping active)",
            "metric": "median over well-months of V_step / V_TLM-MC, both from the same 200 "
                      "draws (seed 2028, draw i from default_rng([2028, i])): stepwise "
                      "SparseTwin.rollout with the draw as direct sources vs the draw through "
                      "the tangent-linear model",
            "threshold": "inside [0.98, 1.02]",
            "reported_not_gated": "V_step / V_formula with V_formula = the frozen-lift "
                                  "independent-noise formula on the stored kernels "
                                  "(engine.head_var_independent), V_TLM-MC / V_frozen-MC from "
                                  "the same draws (the feedback effect free of sampling "
                                  "error), V_frozen-MC / V_formula (sampling check)"},
    },
    "gate_scores": NA("pass/fail per threshold; the declaration passes if R1, R2 and R3 all "
                      "pass; it does not enter the P4.0 acceptance"),
    "interpretation": (
        "A pass on R1 means engine_tlm.tlm_rollout is the exact first-order derivative of the "
        "stepwise twin, lift feedback included; the frozen-lift response (the stored kernels) "
        "omits a first-order term. A pass on R2 and R3 means J Sigma J^T through the "
        "tangent-linear model gives the twin's forcing-induced head variance for these error "
        "magnitudes, so the gap between the frozen-lift K Sigma K^T and the stepwise Monte "
        "Carlo is the omitted first-order feedback, not non-linearity; downstream (plan 4.8) "
        "may then use the tangent-linear factor responses instead of the stepwise Monte Carlo "
        "for factor-structured covariances of this size. A fail on any of them means the "
        "tangent-linear model is not a valid substitute and plan 4.8 keeps the stepwise "
        "Monte Carlo for lift-feedback cases. Thresholds are not relaxed after the fact."),
    "prior_exposure": [
        "engine/v1 validation.json: energy-error stepwise MC / frozen-lift formula median "
        "0.695 (p05-p95 0.46-0.94); independent volumetric noise through the frozen-lift "
        "linear rollout / formula median 0.9985 (no lift feedback in that check).",
        "Verifier of the first round: a 1 % energy perturbation gives a stepwise / frozen-lift "
        "variance ratio of 0.699; 0.3 x Q_base volumetric noise with the base lift feedback "
        "active gives an overall rms ratio of 0.991 and a median per-well-month squared ratio "
        "of 0.879 against the frozen-lift linear rollout.",
        "Synthetic unit tests of engine_tlm (p4_changhua/tests/test_tlm.py, 8x6 grid, 4 "
        "layers): tangent-linear vs central difference below 1e-6 relative for energy, "
        "source and rate perturbations; frozen-lift error above 1e-3. No R1-R3 number has "
        "been computed on the twin before this declaration."],
}


def main() -> int:
    path, sha = prereg2.write_prereg(DOC)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
