"""Write N1.changhua_fix.v1 (harness.prereg3) BEFORE step C's 2016-18 data are read.

Post-failure follow-up of N1.policy_response.v1 step B (Changhua 'inconsistent'), approved
by the user on 2026-10-23: two fixes declared together, scored once each on step C's masked
2016-18 window beside the registered R1c reference. This uses step B's single exploratory
follow-up slot (retry rule); every result is design-time evidence. The 2010-12 leveling
judge was NOT spent: its pre-declared power rule failed
(results/dataprep/reports/N1_policy/fix_judge/power_lev_ch_1012.json).

Usage: python -I prereg_fix.py [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
CODE = Path(__file__).resolve().parent
UNITS = REPO / "results/dataprep/reports/N1_policy/units_policy.json"
R = REPO / "results/twin_runs"
RICE = REPO / "results/dataprep/n1_policy/rice_intensity/v1"

F1 = ("F1, slow rheology: the three-zone leveling column on the masked R1c heads "
      "(stage3_datum_gate_ccz_m1618) fitted with --tau-min-days 3650 (10 yr floor on every "
      "zone's viscous time constant; fixed in advance from R0's mid-zone tau at the 10.8-yr "
      "ceiling and the literature's decades), --hpc0-guard-days 365, --withhold-surveys "
      "2016-01:2018-12 (par25/run.sh lane F1). Motivation: the free three-zone column on R1c "
      "chose a mid-zone tau of 7 d (instant compaction), inflating the year-to-year "
      "subsidence response to pumping that step B measured.")
F2 = ("F2, paddy-weighted return flow: the R1c recipe plus --return-rice-share "
      "n1_policy/rice_intensity/v1 (per-cell township rice-planted area / township area, "
      "2012-22 mean; Changhua mean 0.43, rest 0.35): return-flow logit a + b * intensity, "
      "one learned slope b, one law for the whole fan; refitted with --mask-months "
      "2016-01:2018-12 (stage3_datum_gate_ccr_m1618) + its own three-zone column with the "
      "2016-18 surveys withheld (par25/run.sh lane F2). Motivation: pumped water spread on "
      "flooded paddy returns to layer 0, lowering net local stress per kWh where paddy is "
      "dense; adding the Changhua census raised the fan-wide return fraction 0.42 -> ~0.6.")
GATES = ("Evaluated by run_fix_c.py on model outputs before any test is unlocked: G-a the "
         "column's rest-of-fan out-of-fold leveling R2 (county:rest, non-withheld surveys) "
         ">= the reference column's - 0.02; G-b no uplift: mean modelled subsidence May 2019 "
         "-> May 2022 > 0 over the fan and over Changhua. A fix failing a gate is reported "
         "'failed_design_gate' and not scored.")
SCORE_H = ("step C's heads scorer unchanged (run_step_c.head_table / score_heads: reference "
           "years 2013, 2014, 2019, 2022; >= 18 masked months; paired site bootstrap B 2000, "
           "90 %) plus M vs the R1c reference on the wells both score (paired by well, "
           "sites = sid[:6])")
SCORE_L = ("step C's leveling-partition scorer unchanged (may_cum, partition_deviation, "
           "score_lev: May 2015-2019 increments minus their mean; 5 km-block paired "
           "bootstrap B 2000, 90 %) plus M vs the R1c reference on the same benchmarks")
VERDICT = ("primary: M vs the R1c reference ('model_better' = the fix splits/predicts the "
           "2016-18 Changhua response better than the failed column); secondary: M vs CL and "
           "M vs ZR as in step C")


def tests(u: dict) -> list[dict]:
    wells, lev = u["N1P:wells_m1618"], u["N1P:lev_m1618"]
    lev_common = {"data_scope": "leveling:2016-01..2018-12", "fresh_units": ["N1P:lev_m1618"],
                  "external_units": {"N1P:lev_m1618": lev}, "score": SCORE_L,
                  "verdict": VERDICT, "design_gates": GATES}
    return [
        {"test_id": "N1F.F1_lev_changhua", **lev_common, "model": F1, "seed": 20261134,
         "what": "Does the slow-rheology column split Changhua's 2015-19 subsidence across "
                 "2016-18 better than the failed free-tau column?",
         "label": "post-failure (step B) exploratory, design-time"},
        {"test_id": "N1F.F2_heads_changhua", "data_scope": "heads:2016-01..2018-12",
         "fresh_units": ["N1P:wells_m1618"], "external_units": {"N1P:wells_m1618": wells},
         "model": F2, "score": SCORE_H, "verdict": VERDICT, "design_gates": GATES,
         "seed": 20261124,
         "what": "Do the paddy-return-flow model's masked 2016-18 Changhua head anomalies "
                 "beat the R1c reference's?",
         "label": "post-failure (step B) exploratory, design-time"},
        {"test_id": "N1F.F2_lev_changhua", **lev_common, "model": F2, "seed": 20261134,
         "what": "Same as F1's leveling test for the paddy-return-flow model.",
         "label": "post-failure (step B) exploratory, design-time"},
    ]


def doc(u: dict) -> dict:
    return {
        "id": "N1.changhua_fix.v1", "wp": "N1", "kind": "test",
        "title": "N1 Changhua fixes F1 (slow-rheology column) and F2 (paddy-weighted return "
                 "flow) scored on the masked 2016-18 window beside R1c",
        "tests": tests(u),
        "follows": {"N1.policy_response.v1": "step B Changhua 'inconsistent' (obs -0.50 vs "
                                             "model +0.80 cm/yr per GWh/yr); this record is "
                                             "step B's single exploratory follow-up"},
        "attempt_policy": (
            "User rule 2026-10-23: both fixes are declared here before any 2016-18 value is "
            "read and each test is scored once; this is the ONE post-failure follow-up of "
            "step B, so no further Changhua fix may be scored on 2007-10 or 2016-18 data; a "
            "fix failing its design gate is reported, not repaired and re-scored; the final "
            "report lists every attempt (R1c reference, F1, F2), failures included; two "
            "fixes on one window is a multiple comparison and is reported as such."),
        "zones": "R1c county-split flow; three-zone collapsed column",
        "priors": "the recipe's own", "burn_in": "2012",
        "folds": NA("masked temporal hold-out"), "fresh_units": NA("per test"),
        "thresholds": {"gate_rest_r2_drop": 0.02, "level": 0.90, "min_clusters": 5},
        "gate_scores": NA("no gate on the scores; design gates in each test"),
        "primary_model": "Changhua fixes F1 and F2 against the registered R1c reference "
                         "(Yunlin stays R0; user ruling 2026-10-23)",
        "inputs": [rel_to_repo(p) for p in (
            CODE / "run_fix_c.py", CODE / "run_step_c.py", CODE / "policy_lib.py",
            CODE / "prereg_fix.py", CODE / "build_rice_intensity.py", UNITS,
            RICE / "rice_intensity.npz", RICE / "MANIFEST.json",
            R / "par25/code.diff", R / "par25/run.sh",
            REPO / "results/dataprep/reports/N1_policy/fix_judge/power_lev_ch_1012.json")],
        "interpretation": (
            "'model_better' against the R1c reference means the fix's Changhua response "
            "over the masked 2016-18 swing is closer to the data than the failed column's; "
            "it does not validate the fix (2016-18 is design-time, and the fixes were "
            "motivated by step B's spent data). A fix that wins here becomes a candidate for "
            "the policy page's Changhua levers only with that label; the page keeps 'not "
            "supported' otherwise."),
        "u7": "heads <= 2022-12, leveling <= 2021-12",
        "prior_looks_disclosed": {
            "designer_knowledge": [
                "Step B's results (Changhua obs -0.50, model +0.80), the model-world estimator "
                "calibration, the exposure diagnostic and the R1c three-zone column's own "
                "in-run leveling scores (Changhua out-of-fold R2 -0.64; mid-zone tau 7 d) "
                "were known when F1 and F2 were chosen.",
                "No 2016-18 head or leveling value had been read: step C is held until both "
                "fixes are built (user ruling 2026-10-23)."],
            "harness": "same units as N1.policy_response.v1 step C, declared fresh there at "
                       "03:37 UTC; no look on them since"},
        "no_prior_look_statement": "No head or leveling value of 2016-18 had been read.",
        "written_by": rel_to_repo(Path(__file__)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    u = json.loads(UNITS.read_text(encoding="utf-8"))["units"]
    d = doc(u)
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    missing = [p for p in d["inputs"] if not (REPO / p).exists()]
    print("missing inputs:", missing or "none")
    if not a.write or errs or missing:
        return 1 if (errs or missing) else 0
    path, sha = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
