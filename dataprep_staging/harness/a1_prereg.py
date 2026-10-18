"""WP A1's own pre-registrations, written with the A1 tool before the registered runs.

    python -m harness.a1_prereg coverage_confirm      (from dataprep_staging/)

``coverage_confirm`` registers ``A1.coverage_confirm.v1``: the confirmatory simulation of the
coverage-gate interval (``harness.coverage_confirm``) on fresh seeds, with the rule frozen by
the design-time simulation. It must be written before ``python -m harness.coverage_confirm``
runs; that script refuses to run otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPORTS_DIR  # noqa: E402

from harness import coverage, coverage_confirm, prereg  # noqa: E402

HERE = Path(__file__).resolve().parent
NA = prereg.not_applicable


def coverage_confirm_doc() -> dict:
    cc = coverage_confirm
    unit = f"sim.coverage_confirm.seed{cc.SEED0}"
    return {
        "id": cc.PREREG_ID, "wp": "A1", "kind": "test",
        "title": "Confirmatory simulation of the cluster-adjusted coverage interval (A1 "
                 "acceptance) with the rule frozen by the design-time run",
        "tests": [{
            "test_id": cc.TEST_ID, "data_scope": "simulation:coverage_interval",
            "fresh_units": [unit],
            "external_units": {unit: f"numpy default_rng seeds {cc.SEED0} + 100*i (draws), "
                                     f"+100*i+99 (cluster sizes), +100*i+1/+2 (power); "
                                     "never drawn before this registration"},
            "configurations": [
                {"name": n, "k": k, "m": m, "rho": r, "kind": kd, "role": role}
                for (n, k, m, r, kd, role) in cc.CONFIGS],
            "n_draws": cc.N_DRAWS,
        }],
        "zones": NA("a simulation of the gate interval; no spatial zones"),
        "priors": NA("no model is fitted"),
        "burn_in": NA("no model is fitted"),
        "folds": NA("no folds"),
        "fresh_units": [unit],
        "thresholds": {"actual_coverage_min": cc.CRITERION, "true_coverage": coverage.TARGET,
                       "applies_to": "every primary configuration",
                       "secondary": "two-point mixtures (cluster coverage 1.0 w.p. 0.75, 0.6 "
                                    "w.p. 0.25; ICC 1/3) are reported, never gating"},
        "gate_scores": [{"name": "min actual coverage over primary configurations >= 0.89",
                         "weight": 1.0}],
        "primary_model": (f"coverage-gate interval: cluster-adjusted Wilson on n_eff = n / "
                          f"(1 + (m_w - 1) rho_ANOVA), Student-t quantile on k - 1 df, level "
                          f"{coverage.GATE_LEVEL} (harness.coverage DEFAULT_SMALL_K = "
                          f"'{coverage.DEFAULT_SMALL_K}', chosen by the design-time "
                          "simulation coverage_sim.json)"),
        "inputs": [str(HERE / "coverage.py"), str(HERE / "coverage_sim.py"),
                   str(HERE / "coverage_confirm.py"),
                   str(REPORTS_DIR / "A1_evidence" / "coverage_sim.json")],
        "prior_exposure": (
            "Design-time: coverage_sim.json (seeds 20261012 + ...) compared the normal and "
            "Student-t quantiles on the first 15 configurations and selected 't'. A code "
            "smoke test of coverage_confirm.py with seed base 1 and 300 draws (not the "
            "registered seeds) checked that the script runs; its smallest primary coverage "
            "was 0.90 on 300 draws. The registered seeds have never been drawn."),
        "interpretation": (
            "The A1 coverage acceptance passes if every primary configuration reaches an actual "
            "coverage of at least 0.89 with the frozen rule. A primary configuration below "
            "0.89 is reported as a failure of that acceptance for that configuration, with its "
            "number; the rule, the level and the configurations are not changed after the run "
            "(a changed rule is a new id with new seeds). Secondary configurations and the "
            "power and full-rule pass rates are reported for information only."),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["coverage_confirm"]:
        raise SystemExit("usage: python -m harness.a1_prereg coverage_confirm")
    path, sha = prereg.write_prereg(coverage_confirm_doc())
    print(f"wrote {path} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
