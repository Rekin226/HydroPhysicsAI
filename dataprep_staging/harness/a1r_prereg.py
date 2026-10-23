"""WP A1 repair round: pre-registration of the gate-window evaluation, written with the
repaired tool (``harness.prereg2``) before the registered run.

    python -m harness.a1r_prereg gate_window      (from dataprep_staging/)

Registers ``A1r.gate_window.v1``: the evaluation of rule ``gate-2`` (``harness.coverage2``) by
``harness.gate_window_confirm`` on fresh seeds. That script refuses to run otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import coverage, coverage2, gate_window_confirm, prereg2  # noqa: E402
from harness.coverage_confirm import CONFIGS  # noqa: E402

HERE = Path(__file__).resolve().parent
NA = prereg2.not_applicable


def gate_window_doc() -> dict:
    g = gate_window_confirm
    unit = f"sim.gate_window.seed{g.SEED0}"
    return {
        "id": g.PREREG_ID, "wp": "A1", "kind": "test",
        "title": "Evaluation of the repaired coverage-gate window rule gate-2 (A1 repair round)",
        "tests": [{
            "test_id": g.TEST_ID, "data_scope": "simulation:gate_window",
            "fresh_units": [unit],
            "external_units": {unit: f"numpy default_rng seeds {g.SEED0} + 100*i (draws at "
                                     f"0.90), +100*i+1 / +2 (draws at 0.80 / 0.97), "
                                     f"+100*i+99 (cluster sizes); never drawn before this "
                                     f"registration"},
            "configurations": [{"name": n, "k": k, "m": m, "rho": r, "kind": kd, "role": role}
                               for (n, k, m, r, kd, role) in CONFIGS],
            "n_draws": g.N_DRAWS,
        }],
        "zones": NA("a simulation of the gate rule; no spatial zones"),
        "priors": NA("no model is fitted"),
        "burn_in": NA("no model is fitted"),
        "folds": NA("no folds"),
        "fresh_units": [unit],
        "thresholds": {
            "G1": {"max_window_only_fail_rate_true0.90": g.G1_MAX_WINDOW_ONLY_FAILS,
                   "applies_to": f"every primary configuration with true n_eff < "
                                 f"{g.G1_TRUE_N_EFF_BELOW}"},
            "G2": {"min_window_applies_rate_true0.90": g.G2_MIN_WINDOW_APPLIES,
                   "applies_to": f"every primary configuration with true n_eff >= "
                                 f"{g.G2_MIN_TRUE_N_EFF} and k >= {g.G2_MIN_K}"},
            "G3": "gate-2 and gate-1 intervals identical in every draw",
            "true_n_eff": "n / (1 + (m_w - 1) rho_true), from the configuration's sizes and "
                          "ICC (mixtures: ICC 1/3), known before any draw",
        },
        "gate_scores": [{"name": "G1 and G2 and G3 all hold", "weight": 1.0}],
        "primary_model": (
            f"rule {coverage2.RULE}: interval = frozen harness.coverage interval (cluster-"
            f"adjusted Wilson on n_eff_hat, ANOVA ICC, Student-t on k-1 df, level "
            f"{coverage.GATE_LEVEL}); the 0.85-0.95 point window applies only when the one-"
            f"sided {coverage2.WINDOW_CONF} lower bound n_eff_lo = n / (1 + (m_w - 1) rho_U) is "
            f">= {coverage.N_EFF_WINDOW_MIN}, rho_U the F-based upper bound of the ANOVA ICC "
            "(theta_U = F / F_{0.05}(k-1, n-k), rho_U = (theta_U - 1) / (theta_U + m0 - 1); "
            "rho_U = 1 when MSW = 0)"),
        "inputs": [str(HERE / "coverage.py"), str(HERE / "coverage2.py"),
                   str(HERE / "coverage_sim.py"), str(HERE / "coverage_confirm.py"),
                   str(HERE / "gate_window_confirm.py")],
        "prior_exposure": (
            "The repair was motivated by the A1 verifier's diagnostic of gate-1 (E6 5x6 rho 0.5: "
            "n_eff_hat >= 30 in 47 % of draws, 43 % window-only fails; E11 rho 0.3 35 %; k3x10 "
            "34 %) and by the gate-1 full-rule pass rates already in coverage_confirm.json. "
            "The confidence level of the bound (one-sided 0.95) and the criteria were fixed by "
            "reasoning before any simulation of gate-2. Before this registration gate-2 was "
            "only exercised by unit tests on fixed synthetic inputs (harness/tests/"
            "test_repair.py: a 5x6 stratum, 30 independent rows, one 74x42 draw from seed 5); "
            "no draw with the registered seeds and no simulation study of gate-2 was made."),
        "interpretation": (
            "If G1, G2 and G3 hold, rule gate-2 is the coverage-gate rule of the harness "
            "(harness.coverage2) and the A1 acceptance row for the repaired window rule passes. "
            "A failing criterion is reported with its configurations and numbers as a failure "
            "of that row; the rule, its confidence level and the criteria are not changed after "
            "the run (a changed rule is a new id with new seeds). Window rates, pass rates, "
            "n_eff bias, the bound's calibration, the secondary mixtures and the level-0.90 "
            "interval coverage are reported for information only."),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["gate_window"]:
        raise SystemExit("usage: python -m harness.a1r_prereg gate_window")
    path, sha = prereg2.write_prereg(gate_window_doc())
    print(f"wrote {path} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
