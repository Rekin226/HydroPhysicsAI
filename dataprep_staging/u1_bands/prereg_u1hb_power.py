"""Declaration U1.HB.power.v1 (harness.prereg3, kind 'declaration'): the synthetic power
study of E02c's coverage gate set, registered BEFORE its registered run draws a replicate.

Round 5, rank 2. It fixes the generator, the grid, the seeds and replicate count, the gate set
(E02c's T-cov: G0 pooled manifest check, G1 harness.coverage3 gate-3 per county with clusters
= wells, G1b month clusters report-only), the outputs and the decision-to-wording map, and it
binds by sha256 the code that runs it (power_u1hb.py, harness/coverage3.py and its two
imports, hydrophysics/twin/bands.py). No observation is involved, so nothing is sealed and no
unit is declared. The design itself lives in ``power_u1hb.design_spec()``; this script embeds
it, and ``power_u1hb.py run`` refuses unless the registered copy is identical.

Lead's sequence (from the repository root; the ids and the result path are fixed here):

    PY=~/miniforge3/envs/hydro/bin/python
    $PY -I dataprep_staging/u1_bands/prereg_u1hb_power.py --validate
    $PY -I dataprep_staging/u1_bands/prereg_u1hb_power.py --write --expect-self-sha <sha>
    $PY -I dataprep_staging/u1_bands/power_u1hb.py run
    (writes results/dataprep/reports/U1.HB.power.v1/result.json and result.md and prints
     the result sha256 that E02c cites)

``--validate`` writes nothing: it prints the document, its canonical sha256, prereg3's
validation, the dry-run blocking check and this script's own sha256.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (str(REPO / "dataprep_staging"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import power_u1hb as P  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
ID = P.DECL_ID
DEADLINE = dt.datetime(2026, 10, 31, 16, 0, tzinfo=dt.timezone.utc)
E02 = "results/twin/prospective_2026-11"
E02_SHA = {"predictions.csv": "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32",
           "protocol.json": "8a96c0ea9330fd79432eb17a666640ad526728c838b62bd8459cd41de0d908b9",
           "model": "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228"}
THIS = "dataprep_staging/u1_bands/prereg_u1hb_power.py"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def doc() -> dict:
    design = P.design_spec()
    n_cells = len(P.cells())
    return {
        "id": ID, "wp": "U1", "kind": "declaration",
        "title": "Synthetic power of E02c's per-county coverage gate set (gate-3) for the frozen "
                 "E02 head interval, 2026-11..2027-10: false-fail and power under within-well, "
                 "within-month and year-level miss clustering",
        "why": "E02c (round 5, rank 4) will test the registered combined-sigma 90 % head "
               "interval of results/twin/prospective_2026-11 on one 12-month window at 84 "
               "Yunlin and 74 Changhua wells. With misses clustered in wells (per-well datum "
               "drift at a 47-58-month lead), in months and in the window's year, a single "
               "window may fail a calibrated interval (false fail) or pass an under-covering "
               "one (low power). This study measures both on synthetic hits under a declared "
               "grid, so that E02c's PASS and FAIL wording is fixed before E02c is written. It "
               "estimates nothing from heads.",
        "design": design,
        "n_cells": n_cells,
        "replicates_total": n_cells * P.REPLICATES,
        "gates_simulated": {
            "G0": "pooled point coverage of all 1896 well-months (both counties) in [0.85, "
                  "0.95]: the registered manifest check (protocol.json coverage_bounds, "
                  "twin.validation.evaluate interval_coverage), unchanged and conjunctive",
            "G1": "per county, harness.coverage3.gate_from_totals_batch (rule gate-3, method "
                  "upper90) on well clusters (84 / 74 clusters of 12 months): the interval "
                  "must contain 0.90 at level 0.95; the point window [0.85, 0.95] applies "
                  "when n_eff_lower >= 30 (window conf 0.95). Parity with "
                  "harness.coverage3.coverage_gate is a declared self-check (rel 1e-12)",
            "G1b": "per county, the same rule on 12 month clusters: report-only, labelled "
                   "'uncalibrated (k = 12 < 20)'; it gates nothing",
            "outcome": "a county's 'coverage supported' = G0 AND its G1 (E02c's pre-declared "
                       "pooled-vs-county rule); any other combination is a FAIL",
            "not_simulated": "G2 (signed misses) gates only the word 'balanced' in E02c and is "
                             "not part of this study; the harness.coverage (frozen) and "
                             "harness.coverage2 rules are superseded (A1r2.harness_rules.v1) "
                             "and not used"},
        "outputs_per_cell": [
            "P(G0); per county P(G1), P(G1b), P(supported) and P(FAIL) with Monte Carlo se",
            "per county n_eff and n_eff_lower (mean, q05, q50, q95), the share of replicates "
            "in which the point window applies, the share whose interval contains 0.90",
            "the mean estimated coverage (bias check) and the realised binary ANOVA ICC of "
            "well and month clusters (maps the liability-scale grid to the binary scale)"],
        "outputs_summary": [
            "per county: the false-fail rate at true 0.90 (largest over the grid), the power "
            "at 0.75 and at 0.80 (smallest over cells with ICC_month <= 0.15), the detectable "
            "set (report-only), G1b power (report-only), and the resulting PASS and FAIL "
            "wording keys with their EN/ZH text"],
        "decision_to_wording": {
            "rules": P.WORDING_RULES, "texts": P.WORDING,
            "binds": "E02c's T-cov page wording (C14a and E02's coverage line) uses exactly "
                     "the wording keys this study returns for each county; a PASS never reads "
                     "'calibrated'; this study can only restrict wording, never promote",
            "direction": P.WORDING_RULES["direction"],
            "fail_composition": "'fail_shock' = 'fail_plain' + 'fail_shock_suffix': a FAIL "
                                "always reads 'failed its registered gate (coverage x)'; the "
                                "shock clause is context appended after it, never a "
                                "replacement, and the interval wording is withdrawn on a FAIL "
                                "either way"},
        "expected_outcome_disclosure": {
            "statement": "The wording keys are close to fixed before the run, in the "
                         "conservative direction; the study's value is the magnitudes "
                         "(false-fail rates, power, n_eff), reported per cell and per ICC_year "
                         "slice.",
            "fail": "'fail_shock' is expected for both counties: from the year shock alone "
                    "(analytic approximation, not a run), at true coverage 0.90 the pooled "
                    "check G0 fails with probability about 0.36 at ICC_year 0.1 and about "
                    "0.55 at ICC_year 0.2, far above 0.10",
            "pass": "'pass_weak' is likely but not certain: at true 0.75 and ICC_year 0.2 the "
                    "year shock alone lets G0 pass with probability about 0.25, so P(FAIL) "
                    "there is at least about 0.75, near the 0.80 threshold; G1 and the "
                    "heavy-tailed or high-ICC_well cells decide the rest",
            "driver": "the ICC_year axis {0, 0.1, 0.2} was added by the builder beyond the "
                      "plan's 'a year-level shift' to give that shift a declared size; it is "
                      "what drives the false-fail rule. The per-slice false-fail maximum and "
                      "power minimum are reported (report-only) so readers can see it; the "
                      "rules themselves still take the whole grid, as declared",
            "source": "a hand calculation of P(G0) under the year shock (Gaussian liability, "
                      "other effects averaged out) done in review of the code; the smoke run "
                      "(seed 1) had shown the same direction at ICC_year 0.1"},
        "self_checks": {
            "parity": "G1 and G1b of the vectorised scorer equal "
                      "harness.coverage3.coverage_gate on the same hits (coverage, n_eff, "
                      "n_eff_lower, lo, hi to rel 1e-12; pass and window_applies exactly), 40 "
                      "replicates in each of 3 cells, both counties",
            "unbiased": "the mean estimated coverage equals the true coverage within 4 Monte "
                        "Carlo se at ICC = 0 (binomial se) and in two heavily clustered cells",
            "consistency": "bands.py TARGET / GATE_LEVEL / POINT_WINDOW, coverage3 "
                           "DEFAULT_METHOD / RULE / N_EFF_WINDOW_MIN and protocol.json "
                           "coverage_bounds agree with the declared gate parameters",
            "on_failure": "an engineering failure: the run writes the result flagged "
                          "'engineering_failure' and exits non-zero, and it gates nothing. "
                          "The self-check is deterministic and the run refuses any change "
                          "to the bound code, so a fix that touches a bound code file needs "
                          "a new declaration id (this one is then reported as failed for "
                          "engineering reasons); only a cause outside the bound code (e.g. a "
                          "missing or changed protocol.json) is fixed and re-run under this "
                          "id, with the failed result archived and reported"},
        "code_sha256": P.code_sha256(),
        "run": {"command": "python -I dataprep_staging/u1_bands/power_u1hb.py run",
                "result": f"{P.RESULT_DIR}/result.json (+ result.md)",
                "refuses_unless": "this declaration is in INDEX.jsonl with its file sha, its "
                                  "'design' equals power_u1hb.design_spec() and its "
                                  "'code_sha256' equals the current files",
                "runs_once": "a result that is not an engineering failure is never "
                             "overwritten",
                "cost": "CPU only, about 5-10 minutes (extrapolated from one timed "
                        "cell); 0 GPU-h",
                "cited_by": "E02c cites the sha256 of result.json"},
        "smoke_disclosure": "Before registration the builder ran 'power_u1hb.py smoke' (4 "
                            "cells: ICC_well 0 and heavy, ICC_month 0.15, ICC_year 0.1, "
                            "coverage 0.75 and 0.90; 400 replicates; seed 1, not the "
                            "registered seed) into its session scratchpad to prove the code "
                            "runs, and saw its numbers; the test suite re-runs the same smoke "
                            "into temporary directories. One timing run of a single cell "
                            "(ICC_well 0.5, ICC_month 0.05, ICC_year 0, coverage 0.80, 10,000 "
                            "replicates, seed 99) printed only its run time. The grid, the "
                            "gate set, the thresholds and the wording rules were fixed by "
                            "the round-5 plan (rank 2 and its review) and written into the "
                            "code before any of these runs, and were not changed after them. "
                            "After them, the code review (an analytic argument, not the smoke "
                            "numbers) changed three things that set no outcome: the 'fail_shock' "
                            "text now keeps 'fail_plain' as its prefix, the per-ICC_year "
                            "report-only summary was added, and the expected-outcome "
                            "disclosure above was written; the test suite re-ran the smoke "
                            "after these edits. No smoke or timing number sets anything.",
        "deadline": "registered and run before E02c is written, both by 2026-10-31T16:00Z",
        "zones": NA("synthetic draws; no model zones"),
        "priors": {
            "grid_source": "declared, never estimated from heads: the review of rank 2 asked "
                           "for ICC_well up to 0.7 plus a heavy-tailed mixture because at a "
                           "47-58-month lead per-well datum drift clusters misses in wells; "
                           "harness.coverage3 was simulated to ICC 0.7 (A1r2.harness_rules.v1 "
                           "max_rho_simulated); the ICC_year axis {0, 0.1, 0.2} brackets one "
                           "window-wide shock (a dry or wet year) and was chosen with the "
                           "grid, before any draw",
            "known_after_the_fact_numbers_disclosed": "the combined interval family was chosen "
                                                     "on 2026-10-11 after its 2023-26 coverage "
                                                     "(0.92-0.93) was seen; the run spread "
                                                     "alone covered 0.75; hybrid PICP90 0.877. "
                                                     "None of these entered the grid"},
        "burn_in": NA("synthetic draws"),
        "folds": NA("synthetic draws"),
        "tests": NA("declaration of a synthetic study; no observation is scored"),
        "fresh_units": NA("no observation is involved; nothing is sealed"),
        "thresholds": design["thresholds"],
        "gate_scores": NA("no promoting gate; the study only restricts E02c's wording"),
        "primary_model": NA("no model: synthetic hits and the gate code only"),
        "inputs": sorted(set(P.CODE_FILES) | {THIS, f"{E02}/protocol.json",
                                              f"{E02}/manifest.json"}),
        "frozen_interval_bound": {
            "dir": E02, "predictions_sha256": E02_SHA["predictions.csv"],
            "protocol_sha256": E02_SHA["protocol.json"], "model_sha256": E02_SHA["model"],
            "note": "metadata only: the predictions' values are not read by this study"},
        "attempt_policy": "User rule 2026-10-23: engineering failures (crash, bug, failed "
                          "self-check) are fixed and re-run, each logged; the registered "
                          "study runs once on the declared seed; a result is reported as it "
                          "falls and never re-drawn on another seed or grid; at most one "
                          "follow-up declaration with a new id, labelled 'post-hoc, "
                          "exploratory', which can never replace this result in E02c's "
                          "wording; every attempt is listed",
        "u7": "no post-origin observation and no 2023-26 residual is read or used",
        "rulings": {
            "2026-10-29 user, third set (a)": "E02's combined-sigma interval may be labelled "
                                              "'decided by a registered prospective test' "
                                              "with the EN/ZH U7 disclosure",
            "2026-10-29 user, third set (b)": "E02c is registered on I1's unit with per-county "
                                              "gate-3 (clusters = well) and declared "
                                              "pooled-vs-county outcomes; this study fixes "
                                              "its wording"},
        "interpretation": "Synthetic power only. It says how often E02c's gate set fails a "
                          "truly calibrated interval and how often it fails an under-covering "
                          "one under the declared clustering; it says nothing about the E02 "
                          "interval's real coverage, and it is not evidence for any claim.",
        "written_by": THIS,
    }


def canonical_sha(d: dict) -> str:
    return hashlib.sha256((json.dumps(d, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def problems(d: dict) -> list[str]:
    """What stands between validation and registration (empty = ready)."""
    out = list(P.consistency_problems())
    for p in d["inputs"]:
        if not (REPO / p).exists():
            out.append(f"input {p} is missing")
    for name, want in (("predictions.csv", E02_SHA["predictions.csv"]),
                       ("protocol.json", E02_SHA["protocol.json"])):
        f = REPO / E02 / name
        if f.exists() and sha(f) != want:
            out.append(f"{E02}/{name} sha256 differs from the registered E02 value")
    man = REPO / E02 / "manifest.json"
    if man.exists() and json.loads(man.read_text()).get("model_sha256") != E02_SHA["model"]:
        out.append("E02 manifest model_sha256 differs")
    if (REPO / P.RESULT_DIR / "result.json").exists():
        out.append(f"{P.RESULT_DIR}/result.json exists before the declaration")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="dry run: print, check, write nothing")
    g.add_argument("--write", action="store_true", help="register once (the lead only)")
    ap.add_argument("--expect-self-sha", default=None,
                    help="sha256 of this script as reviewed; required with --write")
    ap.add_argument("--allow-late", action="store_true",
                    help="write after the 2026-10-31T16:00Z deadline (flagged in the output)")
    ap.add_argument("--quiet", action="store_true", help="do not print the document")
    a = ap.parse_args(argv)
    d = doc()
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", canonical_sha(d))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    probs = problems(d)
    print("readiness:", probs or "ready")
    block = prereg3.blocking_looks([], None, test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none (no tests, no fresh units)")
    if (REPO / f"results/dataprep/preregistration/{ID}.json").exists():
        probs.append(f"{ID} is already registered")
    self_sha = sha(Path(__file__).resolve())
    print("this script sha256:", self_sha)
    late = dt.datetime.now(dt.timezone.utc) >= DEADLINE
    if late:
        print(f"DEADLINE PASSED ({DEADLINE:%Y-%m-%d %H:%M} UTC)")
    bad = bool(errs or probs or block)
    if a.validate:
        print("valid; not written" if not bad else "NOT READY: fix the problems above")
        return 1 if bad else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --write needs --expect-self-sha equal to this script's sha256")
        return 1
    if late and not a.allow_late:
        print("REFUSED: past the deadline; pass --allow-late only if the lead accepts that "
              "E02c may already be written")
        return 1
    if bad:
        print("REFUSED: not ready to register")
        return 1
    path, h = prereg3.write_prereg(d)
    print(f"registered {path.relative_to(REPO)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
