"""Write N1.policy_response.v1 (harness.prereg3) BEFORE any value of its test data is read.

Steps B and C of the policy-response plan approved on 2026-10-23 (memory:
n1-policy-response-plan-2026-10-23), with step A declared as model-only checks. The models of
step C (batch 24) were being built when this record was written; their theta and column
files did not exist yet and are checked by run_step_c.py (recipe, mask and withheld surveys)
instead of being hashed here. The code that builds them is hashed (par24/code.diff).

Usage: python -I prereg_policy.py [--write]   (default: validate and dry-run the blocking rule)
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import PREREG_DIR, REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg2 import read_chained  # noqa: E402

NA = prereg3.not_applicable
CODE = Path(__file__).resolve().parent
N1 = CODE.parent / "n1_natural_experiments"
UNITS = REPO / "results/dataprep/reports/N1_policy/units_policy.json"
R = REPO / "results/twin_runs"

ATTEMPT_POLICY = (
    "User rule 2026-10-23 ('trying another way must never threaten the test'): engineering "
    "failures (crash, bug, missing file) may be fixed and re-run before the scoring look, "
    "each logged; every test below is scored ONCE; a failed, inconclusive or "
    "not-informative test is reported as such and never re-scored on the same units; the "
    "alternatives listed under 'declared_secondary' are fixed now and reported beside the "
    "primary, never in its place; after a failure at most ONE follow-up registration per "
    "step (new test ids, label 'post-failure, exploratory', never validation); the final "
    "report lists every attempt, failures included.")

DISCLOSURE = {
    "harness_overlaps": "per-test counts of logged looks whose members and windows overlap "
                        "(computed at write time; none blocks under the prereg3 rule)",
    "unlogged_history": [
        "I1 temporal screens and leveling gates (2012-22): logged retroactively 2026-10-23 as "
        "I1|history_disclosure (units I1:temporal_screens_2020_2022, "
        "I1:leveling_gates_2012_2021).",
        "fresh.block.2016_2018 is a registry fresh unit; 66 logged looks (H2, H3, P5, P2, "
        "P4m, P40, N1 C1/C2 using 2016-18 as reference years) overlap it, which is why step "
        "C carries a design-time label under new unit ids rather than validation.",
        "Pre-2012 leveling: no code in the repository or its worktrees reads the panel "
        "before 2012 (leveling_rows, validate_against_leveling, column_step and mechanism "
        "all clip to >= 2012); W3 starts in 2011; exposure in the unlogged September 2026 "
        "sessions cannot be checked.",
    ],
    "designer_knowledge": [
        "Five design agents (2026-10-23) read written docs only. Seen: G1's numbers; R0's "
        "column parameters; the observed median InSAR seasonal amplitude 6.1 mm (S4b); the "
        "model-side G4 Jan-May compaction at 土庫 (37/57 vs 17 mm); I1 fair temporal ratios "
        "2020-22 (1.05-1.11); census Jan-May kWh ratios 2016-18 (forcing). No observed "
        "leveling or head value of 2007-10 or 2016-18 was read.",
        "The drafting session read leveling survey DATES (sid, datetime, coordinates; per "
        "year x month counts 1997-2021) and electricity coverage flags, no elevation and "
        "no head value.",
    ],
}


def tests(u: dict) -> list[dict]:
    lev_pre, wells, lev_c = (u["N1P:lev_pre2012"], u["N1P:wells_m1618"], u["N1P:lev_m1618"])
    b_common = {
        "data_scope": "leveling:2007-01..2010-12", "fresh_units": ["N1P:lev_pre2012"],
        "external_units": {"N1P:lev_pre2012": lev_pre},
        "observations": "leveling panel (hydrophysics.twin.leveling.load_panel), "
                        "site_subsidence 2007-01..2010-12, min_obs 3, max_rate 0.5 m/yr; "
                        "intervals of 0.5-1.5 yr between consecutive surveys",
        "exposure": "pump_electricity/preorigin/v3 cells_total value_all (kWh, all classes), "
                    "annualised over each interval's months (start survey month excluded, "
                    "end included): E3 = cells within 3 km of the benchmark cell, E10 = the "
                    "3-10 km ring; lead placebo = E3 over the 12 months after the interval",
        "estimand": "beta_obs: OLS of interval rate (m/yr, + = sinking) on E3 and E10 with "
                    "benchmark and campaign (end-survey year) fixed effects; 90 % interval "
                    "from a 5 km-block cluster bootstrap, B 2000",
        "model_side": "beta_model: the county's model run with 16 random pulse fields "
                      "(each 10 km block x water year Jun-May 2012-06..2021-05 at 0.8 or "
                      "1.2 of its energy); May-to-May model rates 2013-2021 at the same "
                      "benchmarks' cells minus the base run regressed on the E3 and E10 "
                      "differences (with intercept); the E3 coefficient. Unit check: model "
                      "energy / C0 energy over the county's cells 2012-02..2021-12 within "
                      "[0.8, 1.25] or the run stops (not a score)",
        "verdict": "policy_lib.ratio_verdict on beta_model and the observed 90 % interval: "
                   "'consistent' if beta_model/[CI] lies in [1/1.5, 1.5]; 'inconsistent' "
                   "if it excludes 1; 'inconclusive' otherwise; 'not_informative' if the "
                   "observed interval includes 0; 'not_evaluable' if the lead placebo's "
                   "interval excludes 0 or fewer than 5 clusters",
        "threats": "era: the model cannot run before 2012, so beta_model is the 2013-21 "
                   "derivative and beta_obs the 2007-10 one (preconsolidation state "
                   "differs); only ~3 campaigns; contemporaneous effect only (lags are "
                   "noise); the 2026 census lacks pumps retired before 2026 (C0 caveat); "
                   "reverse causality through canal shortage (farmers pump where canals "
                   "fail) biases beta_obs up",
        "label": "closest to validation available: outcome units with no logged look; the "
                 "unlogged September 2026 history cannot be excluded",
    }
    b = [
        {"test_id": "N1P.B_dose_yunlin", **b_common, "seed": 20261023,
         "what": "Does R0's local subsidence response per unit of nearby pumping match the "
                 "one identified from the 2007-10 Yunlin leveling?",
         "model": "R0: stage3_datum_gate_cwa theta + coupled_leveling/vep_zonal_leveling.json"},
        {"test_id": "N1P.B_dose_changhua", **b_common, "seed": 20261033,
         "what": "Same for Changhua with the census model.",
         "model": "R1c: stage3_datum_gate_ccz theta (census v1, county zones) + batch-24 "
                  "coupled_leveling_z3/vep_zonal_leveling.json (three-zone column)"},
    ]
    c_models = ("Yunlin: R0 recipe --mask-months 2016-01:2018-12 "
                "(stage3_datum_gate_m1618) + its column fitted with --withhold-surveys "
                "2016-01:2018-12; Changhua: R1c recipe with the same mask "
                "(stage3_datum_gate_ccz_m1618) + its --column-zones 3 column, same withheld "
                "surveys (batch 24, par24/run.sh)")
    zr = ("ZR = the same rollout with 2016-01..2018-12 energy replaced cell by cell by the "
          "same calendar month's mean over 2013, 2014, 2019 and 2022")
    c = []
    for county, seed in (("yunlin", 20261123), ("changhua", 20261133)):
        c.append({
            "test_id": f"N1P.C_heads_{county}", "data_scope": "heads:2016-01..2018-12",
            "fresh_units": ["N1P:wells_m1618"], "external_units": {"N1P:wells_m1618": wells},
            "what": f"Masked refit: do the {county} head anomalies of 2016-18 follow the "
                    "model's better than no anomaly (CL) and than zero pumping response (ZR)?",
            "models": c_models, "zero_response": zr,
            "observations": "heads/L1_clean/v3 via n1 run_cpu_tests.observed (scoring_target, "
                            "flags 0-1, c20, 2013-01..2022-12)",
            "anomaly": "n1lib.anomalies: intercept + trend + 2 harmonics on the reference "
                       "years 2013, 2014, 2019, 2022, per series; event = the 36 masked "
                       "months; eligibility >= 18 event months, >= 36 reference months in "
                       ">= 3 reference years",
            "score": "per-well RMS error; equal-well RMSE; D = RMSE_M - RMSE_X, paired "
                     "site-cluster bootstrap (sid[:6]) B 2000, 90 %: model_better / "
                     "baseline_better / indistinguishable; not_evaluable below 5 sites; "
                     "ZR not informative if RMS(a_M - a_ZR) < 0.05 m",
            "comparisons": {"primary_1": "M vs CL", "primary_2": "M vs ZR"},
            "seed": seed, "label": "design-time evidence (2016-18 exposure disclosed)"})
        c.append({
            "test_id": f"N1P.C_lev_{county}", "data_scope": "leveling:2016-01..2018-12",
            "fresh_units": ["N1P:lev_m1618"], "external_units": {"N1P:lev_m1618": lev_c},
            "what": f"Does the {county} column, fitted without the 2016-18 surveys and driven "
                    "by the masked refit's heads, split the 2015-19 subsidence across the "
                    "four years better than a constant rate (CL) and than ZR?",
            "models": c_models, "zero_response": zr + "; the same column on the ZR heads",
            "observations": "leveling panel; per benchmark the survey nearest each May "
                            "2015-2019 (within 45 days), site_subsidence max_rate 0.5 m/yr",
            "metric": "increments 2015-16..2018-19 minus their mean (the split of the total, "
                      "which the kept 2015 and 2019 surveys constrain); model the same at "
                      "the benchmark cell and the same model months; per-benchmark RMS "
                      "error; paired cluster bootstrap over 5 km blocks, B 2000, 90 %; ZR "
                      "not informative if RMS(d_M - d_ZR) < 0.2 cm",
            "comparisons": {"primary_1": "M vs CL", "primary_2": "M vs ZR"},
            "seed": seed + 1, "label": "design-time evidence; tests the timing of the "
                                       "subsidence response, not its 2015-19 total"})
    return b + c


def doc(u: dict, overlaps: dict) -> dict:
    return {
        "id": "N1.policy_response.v1", "wp": "N1", "kind": "test",
        "title": "N1 policy response on data still usable: pre-2012 leveling dose-response "
                 "(B) and masked 2016-18 refits for heads and subsidence (C); step A "
                 "declared (model-only)",
        "tests": tests(u),
        "declared_checks": {
            "A1_compensation": "model-only (no observation read beyond the model's own "
                               "inputs): truth heads = H0 + k x (base - H0) with k in "
                               "{1, 1.5, 2, 3}, where H0 is the rollout with zero pumping "
                               "energy (the pumping-driven part amplified k times); "
                               "synthetic leveling = the model's column on the truth heads "
                               "at the real benchmark cells and survey months 2012-21, plus "
                               "N(0, 0.5 cm); refit the column (calibrate_coupled code "
                               "path) on the base heads; apply the page's cut30 lever "
                               "(irrigation energy x 0.7) from 2013-01 to both systems and "
                               "report, per zone, fitted / true subsidence avoided by "
                               "2021-12; 'compensates' if within [0.8, 1.25] at k = 2. R0 "
                               "and the R1c three-zone model.",
            "A2_ranking": "model-only, on the page's response basis "
                          "(results/twin_forward/response_basis.npz, R0, one member): "
                          "multiply each lever's subsidence response in each column zone "
                          "and county (6 groups) by a factor drawn log-uniform from "
                          "[0.5, 2], 500 draws; per start year report Kendall tau of the "
                          "six levers' fan-total avoided cm at the last year-end against "
                          "the unperturbed ranking, the Jaccard overlap of the top-decile "
                          "cells of the 30 % irrigation cut, and the 5-95 % range of each "
                          "lever's fan total; a ranking claim survives if tau = 1 (same "
                          "order) in >= 90 % of draws. Limitation: R0's basis has no "
                          "Changhua pumping, so Changhua levers are ~0 by construction; a "
                          "census-model basis needs a forward run (not in this record).",
            "label": "model-internal checks, never quoted as skill"},
        "declared_secondary": {
            "B": "the model's hindcast regression with fixed effects (hindcast_fe_beta_E3); "
                 "the E10 ring coefficients",
            "C": "each model scored on the other county; pooled obs-on-model slopes",
        },
        "attempt_policy": ATTEMPT_POLICY,
        "zones": "the recipes' own (R0 3-zone; R1c county-split flow, three-zone column)",
        "priors": "the recipes' own", "burn_in": "2012 (model); B uses 2007-10 data only",
        "folds": NA("B: no folds; C: masked temporal hold-out"),
        "fresh_units": NA("per test"),
        "thresholds": {"ratio_band": 1.5, "informative_head_m": 0.05,
                       "informative_lev_m": 0.002, "min_clusters": 5, "level": 0.90},
        "gate_scores": NA("no gate; reported whichever way they fall"),
        "primary_model": "R0 for Yunlin, R1c (census) for Changhua (R0 has no Changhua "
                         "pumping); user ruling 2026-10-23",
        "inputs": [rel_to_repo(p) for p in (
            CODE / "policy_lib.py", CODE / "run_step_a.py", CODE / "run_step_b.py",
            CODE / "run_step_c.py",
            CODE / "units_policy.py", CODE / "prereg_policy.py", UNITS,
            N1 / "n1lib.py", N1 / "run_cpu_tests.py", N1 / "units_n1.py", N1 / "n1const.py",
            R / "par24/code.diff", R / "par24/run.sh",
            R / "stage3_datum_gate_cwa/stage3_theta.json",
            R / "stage3_datum_gate_cwa/coupled_leveling/vep_zonal_leveling.json",
            R / "stage3_datum_gate_ccz/stage3_theta.json",
            REPO / "results/dataprep/pump_electricity/preorigin/v3/cells_total.npz",
            REPO / "results/dataprep/pump_electricity/preorigin/v3/MANIFEST.json",
            REPO / "results/dataprep/heads/L1_clean/v3/MANIFEST.json")],
        "interpretation": (
            "B 'consistent' supports the twin's subsidence response per unit of local "
            "pumping in that county (the policy page's cm per GWh), with the era and power "
            "caveats; 'inconsistent' means the page's absolute avoided subsidence is off by "
            "more than the band in that county; 'not_informative' means the data cannot "
            "resolve it. C 'model_better' against ZR credits the masked model's response to "
            "the 2016-18 pumping swing (heads) and its timing (subsidence). Nothing here "
            "selects or recalibrates a model scored on 2023+ data (U7)."),
        "u7": "leveling <= 2021-12 and heads <= 2022-12 only; no 2023+ value is read",
        "prior_looks_disclosed": {**DISCLOSURE, "harness_overlaps_counts": overlaps},
        "no_prior_look_statement": "No observed elevation of 2007-10 or 2016-18 and no head "
                                   "value of 2016-18 had been read for this design.",
        "written_by": rel_to_repo(Path(__file__)),
    }


def overlap_counts(u: dict) -> dict:
    idx = prereg3.membership_index()
    for k, v in u.items():
        if not idx.known(k):
            idx.members[idx.canon(k)] = prereg3._declared_members(v)
    now = dt.datetime.now(dt.timezone.utc)
    looks = read_chained(Path(PREREG_DIR) / "looks.jsonl")
    out = {}
    for t in tests(u):
        c = collections.Counter()
        for lk in looks:
            if prereg3.look_hits(lk, t, idx, test_time=now)["overlap"]:
                c[f"{lk['wp']}|{lk['purpose']}"] += 1
        out[t["test_id"]] = dict(sorted(c.items()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    u = json.loads(UNITS.read_text(encoding="utf-8"))["units"]
    d = doc(u, overlap_counts(u))
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    missing = [p for p in d["inputs"] if not (REPO / p).exists()]
    print("missing inputs:", missing or "none")
    for t, c in d["prior_looks_disclosed"]["harness_overlaps_counts"].items():
        print(f"  {t}: {sum(c.values())} overlapping looks")
    if not a.write or errs or missing:
        return 1 if (errs or missing) else 0
    path, sha = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
