"""Repair round 1: pre-registration ``H3.repair.v1`` (A1 ``prereg3``), written before the step
screen is computed on any real well.

It answers the verifier's first major finding: in the first round the post-look hold-back
(|bias| > 2 x LOYO RMSE against a neighbour model) was applied to 09030212 only, with a
threshold written after its result, and the admitted wells were never checked for a datum
step across their long gaps. ``H3.step_screen`` applies one procedure to every admitted well
and calibrates the threshold on a null population of model wells (``r1_stepscreen``).

The verifier's own numbers on four admitted wells (s7b's neighbour rule with and without
siblings) were read before this registration; they are disclosed as a design-time look
(``r1_looks.py``) and in ``interpretation``.

Usage: python -I r1_prereg.py [--dry-run]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import pandas as pd  # noqa: E402
from harness import prereg3  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402

PREREG = "H3.repair.v1"
TEST = "H3.step_screen"
RH = "09030212"
UNIT = "H3:stepscreen_wells16_2012-2022"
CODE = ["__init__.py", "config.py", "inventory.py", "snapshot.py",
        "s7b_snapshot_consistency.py", "r1_looks.py", "r1_stepscreen.py", "r1_prereg.py",
        "r1_score.py"]
LIB = ["dataprep_staging/harness/prereg3.py", "dataprep_staging/common/guard.py",
       "dataprep_staging/common/paths.py", "dataprep_staging/common/uses.py"]
EV = "results/dataprep/reports/H3_evidence"
DATA = ["results/dataprep/heads/L1_clean/v3/MANIFEST.json",
        "results/dataprep/heads/L1_clean/v3/heads_monthly.parquet",
        "results/dataprep/heads/L1_clean/v3/station_meta_preorigin.parquet",
        "results/dataprep/validation_units/v3/wells.csv",
        "results/dataprep/validation_units/v3/MANIFEST.json",
        "results/dataprep/heads/recovered_wells/v1/MANIFEST.json",
        "results/dataprep/heads/recovered_wells/v1/recovered_wells.csv",
        f"{EV}/registered_wells.json",
        f"{EV}/registered/H3.snapshot_months_2019_2022.csv",
        f"{EV}/registered/H3.snapshot.json",
        f"{EV}/registered/H3.d_wells.json"]


def wells() -> list[str]:
    W = pd.read_csv(C.REPO / "results/dataprep/heads/recovered_wells/v1/recovered_wells.csv",
                    dtype={"sid": str})
    return sorted(W[W.admitted.astype(bool)].sid) + [RH]


def doc() -> dict:
    ids = wells()
    test = {
        "test_id": TEST, "data_scope": "heads:2012-01..2022-12", "fresh_units": [UNIT],
        "external_units": {UNIT: {
            "description": "level-shift statistic across each well's split (longest internal "
                           "L1 v3 C20 gap; 09030212: its 2019-01 switch to once-daily "
                           "readings) for the 15 wells admitted in heads/recovered_wells/v1 "
                           "and 09030212, against a null of the same statistic at the model "
                           "wells; never computed with this neighbour pool or this null",
            "ids": ids, "window": "2012-01..2022-12"}},
        "metric": "r1_stepscreen.screen_one: neighbour pool = the 158 model wells minus the "
                  "target's site; candidates within 25 km with >= 24 common pre-split C20 "
                  "months, observed in >= 80 % of the target's post-split months, Pearson r "
                  ">= 0.6 on common pre months; top 4 by r; ridge (alpha 1e-3, standardised, "
                  "intercept) fitted on pre months with all neighbours observed (>= 24); "
                  "leave-one-calendar-year-out RMSE (rmse_loyo); bias = mean residual on "
                  "post months with all neighbours observed (>= 12); z = |bias| / rmse_loyo; "
                  "untestable with < 2 neighbours or unmet month minima. Null: the same "
                  "statistic with the same calendar split at every model well (its own site "
                  "excluded); p = (1 + #{null z >= z}) / (1 + n_null)",
        "threshold": "flagged if p <= 0.05 with n_null >= 30 testable model wells; otherwise "
                     "not flagged; untestable if the statistic or the null cannot be formed"}
    gate_rule = (
        "Admission for heads/recovered_wells/v2 (replaces the first round's post-look "
        "hold-back criterion; H3.recovery.v1 otherwise unchanged). (1) The H3.recovery.v1 "
        "envelope on observed months and its bridge rule stand: flag-2 rows only for wells "
        "that pass H3.d_wells while the pooled RMSE rule passes, and only rows with method "
        "daily_bridge_partial (family d tests partial-month bridges; outage bridges were "
        "never tested per well). (2) H3.step_screen: an admitted v1 well that is flagged or "
        "untestable is held back. Its record is then two segments of unknown relative datum, "
        "and neither segment alone meets the envelope (a pre-split segment leaves a trailing "
        "run > 0 months; a post-split segment has a leading run > 66 months or < 36 observed "
        "months); this is checked and recorded per well. Its rows stay in the evidence for "
        "a sensitivity member only. (3) 09030212 stays held back whatever its step-screen "
        "result: its once-daily months have no informative representativeness test "
        "(H3.snapshot passed at a well whose own RMSE, 0.0025 m, is 20x below sigma_snap, "
        "because the point window switched off; the verifier showed that 24 of 52 "
        "estimation wells fail the same test on their own months) and the persistence-screen "
        "constants were chosen after the series was viewed. Its screen result is reported "
        "only, as the uniform test of the first round's hold-back reason. (4) Uses unchanged: "
        "calibration_target only in the pre-declared I1 rung R5, datum_fit (flags 0-1), "
        "validator (product card: filter to flags 0-1, U1), after_the_fact_analysis. (5) The "
        "product's validation report and manifest state the plan acceptance separately from "
        "the release rule: 'each recovered well passes the H2 coverage strata' is recorded "
        "as failed when any admitted well fails H3.d_wells, whatever the release rule says.")
    return {
        "id": PREREG, "wp": C.WP, "kind": "test",
        "title": "WP H3 repair round: one level-shift screen for every recovered well, "
                 "calibrated on a model-well null",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP H3, 2.4-2.5, 4.4-4.7",
        "tests": [test],
        "zones": prereg3.not_applicable("point records"),
        "priors": prereg3.not_applicable("no priors"),
        "burn_in": prereg3.not_applicable("no model run"),
        "folds": prereg3.not_applicable("leave-one-calendar-year-out inside the statistic"),
        "fresh_units": [UNIT],
        "design_time_units": [
            "H3:verifier_neighbour_check_admitted15 (verifier: s7b rule with and without "
            "siblings on the 15 admitted wells; numbers read: 07260241 +0.37 m (2.2x), "
            "09080241 +0.52 m (2.5x), 09150131 +0.24 m (2.1x), 09080221 1.7x; non-sibling: "
            "07260241 2.4x, 09130321 -1.22 m (1.7x))",
            "H3:verifier_snapshot_power_estimation_wells",
            "H3:after_fact_09030212_neighbours (first round: +0.87 m, LOYO RMSE 0.375 m)",
            "H3:inventory_2012-2022 (record extents and gap locations of every station)",
            "H3:d_r100-159_wells16 and H3:snapshot_09030212_2012-2018 (first-round scores)"],
        "thresholds": {TEST: "p <= 0.05 against the model-well null (n_null >= 30)"},
        "gate_scores": [{"name": TEST, "rule": "hold back flagged or untestable wells",
                         "weight": 1.0}],
        "gate_rule": gate_rule,
        "primary_model": "ridge regression of the well on its 4 best-correlated non-sibling "
                         "model-well neighbours, fitted before the split",
        "inputs": ([f"dataprep_staging/h3_recovered_wells/{f}" for f in CODE] + LIB + DATA),
        "reported_not_gated": [
            "per split: null median and 95th percentile of z, the share of model wells with "
            "z >= 2 (the first round's hold-back criterion), the mean null bias (a fan-wide "
            "departure after the split shows here)",
            "09030212's z and p (rule 3: reported only)"],
        "interpretation": (
            "The first round held 09030212 back on |bias| > 2 x LOYO RMSE, a threshold "
            "written after its result, and checked no other well. Under H3.repair.v1 one "
            "procedure is applied to all 15 admitted wells and the threshold comes from the "
            "model wells, which carry the same calendar split without a gap (continuous "
            "records across 2019, screened by H1a's offset detector). A fan-wide departure "
            "after the split (for example the 2021 drought) then raises the null as much as "
            "the targets, and only a well-specific shift is flagged. Neighbours exclude the "
            "target's site because nested wells share the site's instrument history. These "
            "two choices are structural; the verifier's numbers for 07260241, 09080241, "
            "09150131 and 09130321 (and the non-sibling variant it ran) were seen before "
            "this registration, so the test is not blind for those wells; alpha (0.05), the "
            "statistic (the first round's |bias| / LOYO RMSE), the neighbour count, radius "
            "and r threshold (s7b's) were not tuned. Model wells that carry an undetected "
            "step of their own widen the null and lower the power; that bias favours "
            "admission. With 15 wells at alpha 0.05 about 0.75 wells are flagged by chance "
            "under no shift; a false flag only removes a well. No observation dated 2023-01 "
            "or later enters (U7)."),
    }


def main(argv):
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("; ".join(errs))
    if argv and argv[0] == "--dry-run":
        print("valid;", len(d["inputs"]), "inputs;", len(d["tests"][0]["external_units"][UNIT]
                                                          ["ids"]), "wells")
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main(sys.argv[1:])
