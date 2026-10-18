"""Write the P4.0 pre-registration (declaration) before the engine's acceptance checks run.

    tools/cpu_slot.sh python -m p4_changhua.prereg_p40      (from dataprep_staging/)

The P4.0 acceptance is a numerical verification of a solver against the repository's own
solver, not a test scored on data units, so it is registered as a declaration: thresholds,
cases, metrics and the interpretation are fixed here, before ``build_engine`` computes them.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg  # noqa: E402

THETA = "results/twin_runs/stage3_datum_gate_cwa/stage3_theta.json"
NA = prereg.not_applicable

DOC = {
    "id": "P40.engine_acceptance.v1",
    "wp": "P40",
    "title": "P4.0 linear-response engine: sparse replica and adjoint kernels, acceptance",
    "kind": "declaration",
    "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP P4, P4.0 (Acceptance)",
    "tests": NA("numerical verification against the repository solver; no data units"),
    "zones": NA("no zonation is chosen; the engine reproduces the calibrated zonal theta"),
    "priors": {"well_datum": "fit_flow convention: N(0, 5 m^2), 12 months per independent "
                             "observation (calibrate_flow._profile_well_datum); the fully "
                             "free datum is a sensitivity only"},
    "burn_in": NA("no fit; the replica starts from the calibration's own h0 (2012-01)"),
    "folds": NA("no folds"),
    "fresh_units": NA("no scoring on observations"),
    "thresholds": {
        "A1_sparse_vs_torch_m": {
            "threshold": "max |h_sparse - h_torch| <= 1e-5 m over all layers, cells and the "
                         "131 steps 2012-02..2022-12",
            "cases": ["base: the cwa deliverable's forcing (the acceptance case)",
                      "extra: base + a synthetic Changhua abstraction (lift feedback on the "
                      "added energy; reported against the same threshold)",
                      "riv: cwa theta with the temporal_rivers MODFLOW-RIV recipe and a "
                      "synthetic canal-recharge field (time-varying operator; reported "
                      "against the same threshold)"],
            "torch": "hydrophysics.twin.forward.rollout on CPU, CUDA hidden, at most 2 "
                     "threads (cpu_slot.sh --slots 2)"},
        "A2_adjoint_vs_forward_rel": {
            "threshold": "max |K_adjoint - h_forward| / max |h_forward| <= 1e-6 per case, for "
                         "the float64 kernels and again for the stored float32 kernels",
            "cases": ["uniform 1 m3/day abstraction in every Changhua cell at step 0 (the "
                      "scout's check), all wells, all lags",
                      "single-cell unit sources in each of the 4 layers (8 random cells per "
                      "layer, seed 2026), all wells, all lags",
                      "a random abstraction field over 131 months: convolution vs stepwise "
                      "linear rollout",
                      "frozen-lift energy pathway: stepwise rollout with an added energy "
                      "field at the base head path vs convolution of the linearised "
                      "abstraction"]},
    },
    "gate_scores": NA("acceptance is pass/fail per threshold; both must pass"),
    "primary_model": "cwa deliverable theta " + THETA + " (provisional until batch 8's "
                     "merged theta is confirmed; the sha256 is recorded in the inputs)",
    "inputs": [THETA],
    "interpretation": (
        "A pass on A1 means the sparse replica reproduces the repository's torch rollout of "
        "the calibrated twin to well below observation error, so engine results stand for "
        "the twin's within the recipe's physics (no delay bed, aquitard store or rivers in "
        "the deliverable; RIV switching verified on a non-calibrated configuration). A pass "
        "on A2 means the stored kernels give the twin's linear response to abstraction "
        "exactly (to float32 rounding) for lags up to the stored length; the kernel "
        "truncation beyond that length is reported separately and is not part of the "
        "gate. A fail on either blocks downstream use of the kernels until fixed; the "
        "thresholds are not relaxed after the fact. The convolution engine is valid only "
        "with fixed lift and no RIV switching; the lift-feedback difference is reported as "
        "a diagnostic, not a gate."),
    "prior_exposure": [
        "Design-time evidence from the scout (_pilots/physics_inversion_hooks/b_timing.json, "
        "c_greens.py output): sparse vs torch max |dh| 1.66e-6 m on this theta (sha256 "
        "789d0e45...); the scout's adjoint-vs-forward impulse check on the Changhua uniform "
        "impulse. A0 reproduced both byte-identically. This WP has run only synthetic-grid "
        "unit tests so far (p4_changhua/tests)."],
}


def main() -> int:
    path, sha = prereg.write_prereg(DOC)
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
