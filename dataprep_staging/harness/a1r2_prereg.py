"""WP A1 second repair round: inputs and pre-registration of the coverage3 confirmation,
written with ``harness.prereg3`` before the registered run.

    python -m harness.a1r2_prereg sizes       (writes the pilot cluster sizes file once)
    python -m harness.a1r2_prereg confirm     (registers A1r2.coverage3_confirm.v1)
    python -m harness.a1r2_prereg rules       (declares A1r2.harness_rules.v1, after the run)

``harness.coverage3_confirm`` refuses to run unless the registration exists and predates every
look at its draws.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import guard  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PILOTS, from_repo, rel_to_repo  # noqa: E402

from harness import coverage3, coverage3_confirm, prereg3  # noqa: E402

HERE = Path(__file__).resolve().parent
NA = prereg3.not_applicable
FAMILIES = ("N_net1", "N_net2", "H_half12", "E_lead66", "E_trail48", "S_gap06", "S_gap12",
            "S_gap24")


def write_sizes() -> Path:
    """Cluster (gap) sizes of the gap-fill pilot's families: clean wells, method ens_avg
    (pre-origin output, opened as design_time_evidence). Sizes only; no hit is read."""
    import pandas as pd
    prod = guard.open_product(PILOTS / "head_gapfill_pilot", "design_time_evidence")
    p = prod.path("out/pilot_ensembles.parquet")
    D = pd.read_parquet(p, columns=["Lname", "method", "gid", "suspect", "step_well"])
    D = D[~D.suspect & ~D.step_well & (D.method == "ens_avg")]
    sizes = {f: sorted(int(v) for v in D[D.Lname == f].groupby("gid").size().to_numpy())
             for f in FAMILIES}
    out = from_repo(coverage3_confirm.PILOT_SIZES_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"source": {"path": rel_to_repo(p), "sha256": sha256_file(p),
                      "use": "design_time_evidence",
                      "selection": "clean wells (not suspect, not step_well), method ens_avg, "
                                   "cluster = gid; sizes only"},
           "sizes": sizes}
    with open(out, "x", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    return out


def confirm_doc() -> dict:
    c = coverage3_confirm
    unit = f"sim.coverage3_confirm.seed{c.SEED0}"
    pilot = json.loads(from_repo(c.PILOT_SIZES_FILE).read_text(encoding="utf-8"))["sizes"]
    cfg = c.configs(pilot)
    return {
        "id": c.PREREG_ID, "wp": "A1", "kind": "test",
        "title": "Confirmation of the coverage3 interval (upper90) inside its envelope "
                 "(A1 second repair round)",
        "tests": [{
            "test_id": c.TEST_ID, "data_scope": "simulation:coverage3_interval",
            "fresh_units": [unit],
            "external_units": {unit: {
                "description": f"numpy default_rng seeds {c.SEED0} + 100*i (draws at 0.90), "
                               f"+100*i+1 / +2 (draws at 0.80 / 0.97), +100*i+99 (cluster "
                               f"sizes); never drawn before this registration",
                "none": "simulation draws; no well, benchmark or cell"}},
            "configurations": [{k: v for k, v in x.items() if k != "sizes"}
                               | {"sizes": (x["sizes"] if isinstance(x["sizes"], str)
                                            else f"{len(x['sizes'])} clusters, total "
                                                 f"{int(sum(x['sizes']))}")}
                               for x in cfg],
            "n_draws": c.N_DRAWS,
        }],
        "zones": NA("a simulation of the coverage interval; no spatial zones"),
        "priors": NA("no model is fitted"),
        "burn_in": NA("no model is fitted"),
        "folds": NA("no folds"),
        "fresh_units": [unit],
        "thresholds": {
            "C1": {"min_actual_coverage_true0.90": c.CRITERION,
                   "applies_to": f"every primary configuration ({sum(x['role'] == 'primary' for x in cfg)}"
                                 f"; k >= {c.MIN_K} clusters, ICC <= {c.MAX_RHO})"},
            "level": coverage3.GATE_LEVEL,
            "secondary": "configurations outside the envelope (k < 20) and two-point "
                         "mixtures inside it are reported, never gating",
        },
        "gate_scores": [{"name": "C1 holds", "weight": 1.0}],
        "primary_model": (
            f"coverage3.DEFAULT_METHOD = {coverage3.DEFAULT_METHOD!r}: Wilson score interval "
            "on n_eff = n / (1 + (m_w - 1) rho_U90), rho_U90 the one-sided 0.90 upper "
            "confidence bound of the one-way ANOVA ICC (theta = F / F_{0.10}(k-1, n-k), "
            "rho = (theta - 1) / (theta + m0 - 1), clipped to [0, 1]; rho = 1 when MSW = 0), "
            "Student-t quantile on k - 1 df, level 0.95; envelope k >= 20 clusters"),
        "inputs": [str(HERE / "coverage.py"), str(HERE / "coverage2.py"),
                   str(HERE / "coverage3.py"), str(HERE / "coverage_sim.py"),
                   str(HERE / "coverage3_confirm.py"), str(HERE / "coverage3_design.py"),
                   c.PILOT_SIZES_FILE,
                   "results/dataprep/reports/A1_evidence/repair2/coverage3_design.json"],
        "prior_exposure": (
            "Design-time: the A1 verifier's second-round diagnostic of the frozen interval "
            "(14x13 ICC 0.4/0.5/0.7, 7x12 ICC 0.6, 10x50 ICC 0.5, 12x20 ICC 0.4, the pilot's "
            "N structures at ICC 0.3/0.5); the head gap-fill pilot's ANOVA ICC estimates of its "
            "hit/miss data (up to 0.69) and its cluster sizes; the design sweep "
            "coverage3_design.json (seeds 20291012+, 328 configurations, all seven candidates, "
            "which chose upper90 and k >= 20 by the rule declared in coverage3_design.py). "
            "Before this registration coverage3_confirm.run was executed only as a smoke test "
            "with 200 draws and seed 777 (never the registered seeds). The registered seeds "
            f"{c.SEED0}+ were never drawn."),
        "interpretation": (
            "If C1 holds, the A1 acceptance row 'coverage interval reaches at least 0.89 actual "
            "coverage of a true 0.90' passes for coverage3 within the envelope (k >= 20 "
            "clusters, simulated ICC <= 0.7), and coverage3 (rule gate-3) is the harness's "
            "coverage gate; strata with fewer clusters carry calibrated = False. A failing "
            "configuration is reported with its numbers as a failure of that row; the method, "
            "the envelope and the criterion are not changed after the run (a changed rule is a "
            "new id with new seeds). Secondary configurations, power, gate-3 pass rates, "
            "window-only false fails and level-0.90 coverage are information only."),
    }


def rules_doc() -> dict:
    """Declaration of the harness rules after the second repair round (no new units)."""
    from common.paths import DATAPREP
    v3 = DATAPREP / "validation_units" / "v3"
    ev = "results/dataprep/reports/A1_evidence/repair2/"
    return {
        "id": "A1r2.harness_rules.v1", "wp": "A1", "kind": "declaration",
        "title": "Harness rules after the A1 second repair round: coverage gate gate-3 "
                 "(harness.coverage3) and pre-registration tool harness.prereg3",
        "tests": NA("a declaration of rules; every downstream test writes its own "
                    "pre-registration with harness.prereg3"),
        "zones": NA("no zones are declared"),
        "priors": NA("no model is fitted by A1"),
        "burn_in": NA("no model is fitted by A1"),
        "folds": NA("unchanged: the folds of A1-fresh-units-v2"),
        "fresh_units": NA("no new fresh units: the 15 units declared in A1-fresh-units-v2 "
                          "stand (validation_units/v3 carries them byte-identically)"),
        "supersedes": {"A1-fresh-units-v2.thresholds.coverage_gate": "gate-3 below",
                       "pre-registration tool": "harness.prereg2 -> harness.prereg3"},
        "thresholds": {"coverage_gate": {
            "module": "harness.coverage3", "rule": coverage3.RULE,
            "interval": coverage3.DEFAULT_METHOD + ": Wilson on n_eff from the one-sided 0.90 "
                        "upper bound of the ANOVA ICC (rho = 1 when MSW = 0), Student-t on "
                        "k - 1 df",
            "level": coverage3.GATE_LEVEL, "target": coverage3.TARGET,
            "point_window": list(coverage3.POINT_WINDOW),
            "point_window_if_n_eff_lower_at_least": coverage3.N_EFF_WINDOW_MIN,
            "n_eff_lower": "gate-2 (one-sided 0.95 bound from the F-based upper bound of the "
                           "ICC)",
            "calibrated_envelope": coverage3.ENVELOPE,
            "evidence": [ev + "coverage3_design.json (design-time)",
                         ev + "coverage3_confirm.json (A1r2.coverage3_confirm.v1)"]},
            "preregistration_tool": {"module": "harness.prereg3",
                                     "blocking_rule": prereg3.BLOCKING_RULE}},
        "gate_scores": NA("set by each test's pre-registration"),
        "primary_model": NA("set by each test's pre-registration"),
        "inputs": [str(HERE / "coverage3.py"), str(HERE / "prereg3.py"),
                   str(v3 / "units.json"), str(v3 / "MANIFEST.json"),
                   ev + "coverage3_confirm.json", ev + "a1r2_checks.json"],
        "interpretation": (
            "From this declaration on, coverage verdicts use harness.coverage3 (rule gate-3) and "
            "report coverage, n_eff, n_eff_lower, k, the interval and 'calibrated' (False for "
            "strata with fewer than 20 clusters: pool them first or report the verdict as "
            "uncalibrated); pre-registrations, looks and scores use harness.prereg3, which "
            "blocks by unit membership and parses scopes strictly. Fresh units overlap heavily "
            "(72 of 105 pairs share wells, cells or the 2016-18 block), so every test on fresh "
            "units is registered before the first scoring look on any overlapping unit."),
    }


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["rules"]:
        path, sha = prereg3.write_prereg(rules_doc())
        print(f"wrote {path} sha256 {sha}")
        return 0
    if argv == ["sizes"]:
        print(f"wrote {write_sizes()}")
        return 0
    if argv == ["confirm"]:
        path, sha = prereg3.write_prereg(confirm_doc())
        print(f"wrote {path} sha256 {sha}")
        return 0
    raise SystemExit("usage: python -m harness.a1r2_prereg sizes|confirm|rules")


if __name__ == "__main__":
    raise SystemExit(main())
