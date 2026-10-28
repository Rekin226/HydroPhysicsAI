"""Declare X5.changhua_water.v1 (harness.prereg3, kind 'declaration') BEFORE any X5 metric.

Why a declaration and not a test: the only monthly delivery series for Yunlin is W1's
(surface_water_v4), whose every monthly row is flag 5 ("monthly interval uncalibrated (no
monthly observation exists)"). U1 forbids inferred values as scoring targets, so the held-out
Yunlin check is an AGREEMENT check against an inferred reference, run once with its threshold
fixed here, and never quoted as timing skill. The observed sub-annual delivery numbers that do
exist (Jan-May 2015-2021, MOA crop seasons to 2015) were already seen and are reported as
design-time only.

Usage: flock <lock> python -I prereg_x5.py [--write]   (default: validate only)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from x5_changhua_water import x5const as K  # noqa: E402
from x5_changhua_water.look_x5 import HELDOUT_SCOPE, HELDOUT_UNIT  # noqa: E402

NA = prereg3.not_applicable
CODE = ["__init__.py", "x5const.py", "x5lib.py", "pipeline.py", "coherence.py",
        "synthetic_check.py", "fit_x5.py", "score_x5.py", "look_x5.py", "prereg_x5.py"]


def doc() -> dict:
    inputs = [K.MOA_CSV, K.JIJI_CSV, K.W1_PRE / "district_monthly.csv",
              K.W1_PRE / "inflow_composite_daily.parquet", K.W1_PRE / "gauges_daily_L1.parquet",
              K.W1_PRE / "janmay_validation.csv", K.W1_PRE / "MANIFEST.json",
              K.W1_FULL / "district_monthly.csv", K.W1_FULL / "inflow_composite_daily.parquet",
              K.W1_FULL / "gauges_daily_L1.parquet", K.X4_PRE, K.V3_NPZ, K.COUNTY_MAP,
              K.HEADS / "heads_monthly.parquet", K.HEADS / "station_meta_preorigin.parquet",
              K.HEADS / "MANIFEST.json", K.WORK / "coherence_monthly.csv"]
    inputs += [K.CODE_DIR / c for c in CODE]
    ext = {HELDOUT_UNIT: {
        "description": "W1 surface_water_v4/preorigin/v2 Yunlin monthly district values of "
                       "2012-2014 and 2022 (48 months), inferred (flag 5); not read by X5 "
                       "before the check",
        "none": "inferred district values; no wells, benchmarks or cells"}}
    return {
        "external_units": ext,
        "id": K.PREREG_ID, "wp": K.WP, "kind": "declaration",
        "title": "X5: Changhua irrigation water outside the Jiji supply (annual volume, sd) and "
                 "its monthly timing from coherence, river balance and shallow heads, "
                 "calibrated on Yunlin; agreement check on held-out pre-2023 Yunlin years",
        "tests": [{
            "test_id": K.CHECK_ID, "data_scope": HELDOUT_SCOPE, "fresh_units": [HELDOUT_UNIT],
            "external_units": ext,
            "what": "Does the Yunlin-calibrated timing model, Denton-benchmarked to the Jiji "
                    "Yunlin annual delivery, reproduce W1's Yunlin monthly values in years "
                    "not used to fit it?",
            "metric": "Pearson r over the 48 held-out months (structural-zero months "
                      "included) between the primary prediction and the W1 value",
            "decision_rule": "r >= 0.6: 'agrees with W1'; else 'does not agree'. Run once.",
            "label": "design-time agreement with an inferred reference; never validation of "
                     "timing skill (U1); cannot promote anything by itself"}],
        "estimators": {
            "annual_volume": (
                "V(y) = MOA 35644 Changhua office actual irrigation water (10^8 m3; crop "
                "seasons summed to 2015, annual '全' from 2016) minus Jiji weir "
                "changhua_irrigation(y), 2012-2024, flag 5. 2025 (no MOA row): mean V over "
                "2016-2022, flag 6, sd = hypot(sd_def, sd of V over 2016-2022). "
                "sd_def = sample sd (ddof 1) over 2016-2022 of the Yunlin office's MOA minus "
                "Jiji yunlin_irrigation (the definitional mismatch: Yunlin is nearly all "
                "Jiji, and its office total falls below the Jiji delivery in some years). "
                "sd(y) = hypot(sd_def, sd_step(y)); sd_step(y) = |V(y) - mean V 2016-2022| / 2 "
                "for y <= 2015 (both readings of the 2016 reporting change), else 0."),
            "indicators": {
                "coh": "1 - mean short-pair (<= 12 d) LiCSAR coherence, frame 069A_06694, "
                       "pairs assigned to the month of their midpoint, over pixels in twin "
                       "cells with X4 pre-origin canal service share > 0 for the district "
                       "(Changhua cells for chc, the rest for yl); 2020-01 onward",
                "riv": "monthly mean of max(0, W1 inflow composite - Q(H071 溪州大橋)) in "
                       "cms, Jan-May only (other months missing); W1 pre-origin/v2 to 2022, "
                       "full/v4 from 2023; same series for both districts",
                "head": "county median over layer-1 wells (heads/L1_clean/v3 heads_monthly, "
                        "flags 0-2 via guard use model_design, window_end 2022-12) of the "
                        "month-to-month change of the monthly head, >= 5 wells; none after "
                        "2022 (heads are a forbidden input of a recorded forcing arm)",
                "standardisation": "z = (x - month-of-year mean over 2015-2021) / pooled sd "
                                   "of those anomalies over 2015-2021, per district; "
                                   "missing z = 0"},
            "timing_model": (
                "index(m, y) = calendar_c(month) x exp(sum_k b_k z_k(m, y)); calendar_c = mean "
                "monthly share of W1 pre-origin/v2 for district c over 2015-2021; b fitted by "
                "least squares on the Yunlin W1 monthly shares of 2015-2021 (normalised "
                "within year) and transferred to Changhua"),
            "benchmarking": (
                "PRIMARY: proportional first-difference Denton of the index to the annual "
                "totals over positive-index months (zero-index months stay 0), one run per "
                "block 2012-2022 and 2023-2025 (no post-origin value moves a pre-origin "
                "month). Yunlin check: annual = Jiji yunlin_irrigation. Changhua product: "
                "annual = V(y)."),
            "primary": "T3 = (coh, riv, head) + Denton",
            "declared_alternatives_reported_never_substituted": [
                "T2 = (coh, riv) + Denton (no heads; also the stage-2 rung input, so that "
                "the rung's head scores are not scored on heads that shaped its forcing)",
                "T3 + Chow-Lin without intercept, AR(1) rho by grid maximum likelihood "
                "0..0.95 step 0.05 on the annual aggregates, positive-index months only"],
            "baseline_reported": "calendar only (b = 0) + Denton",
            "monthly_sd": "sqrt((share x sd_A)^2 + (A x sd_share(month))^2), sd_share = RMS "
                          "of leave-one-year-out share residuals of the Yunlin fit over "
                          "2015-2021, per calendar month; labelled uncalibrated (the "
                          "reference is inferred)"},
        "checks": {
            "S0_synthetic": (
                "Runs first. Truth = real Yunlin z, v3's published crop calendar (Jan 0, Feb "
                "1.5, Mar 1.5, Apr 1, May 1, Jun 0.5, Jul 1.5, Aug 1, Sep 1, Oct 1, Nov 0.5, "
                "Dec 0), b* = (coh 0.4, riv 0.3, head 0.2), log-normal noise sd 0.2 per month, "
                "seed 20261025, scaled to Jiji Yunlin annual 2012-2022; same pipeline fit on "
                "2015-2021, predict 2012-2014 and 2022. PASS if all: |b_hat - b*| <= 0.5 |b*| "
                "for every k; held-out r >= 0.9; held-out anomaly r (vs the calendar-only "
                "prediction) >= 0.5; annual sums to 1e-9 relative. If it fails, the real "
                "check is still run and reported, labelled 'pipeline failed its synthetic "
                "check'."),
            "X5C_reported_beside": [
                "r without structural-zero months", "anomaly r vs the calendar-only "
                "baseline (the informative number: W1 is a calendar x supply indicator, so "
                "the calendar alone is expected to clear r 0.6 as well)",
                "per-year shape r and RMSE", "the same for T2, T3 Chow-Lin and the baseline"],
            "exploration_U7": "2023-2025 agreement with W1 full/v4 Yunlin (also inferred), "
                              "labelled exploration; never gates",
            "designtime_reported": [
                "Jan-May 2015-2021 sums of the Jiji-timing prediction per district vs the "
                "observed briefing values (in-sample, already seen)",
                "Jan-Jun share 2012-2015 vs the MOA office crop-1 share (already seen; the "
                "Changhua office total includes non-Jiji water)"],
            "split_rationale": "calibration 2015-2021 = the years whose W1 Jan-May sums X5 "
                               "had already seen (look logged); held-out = every other "
                               "pre-2023 year (2012-2014, 2022); coherence exists from 2020, "
                               "so only 2022 of the held-out years has it",
            "circularity_disclosed": "W1's monthly indicator is built from the same inflow "
                                     "composite as 'riv', so riv-driven agreement is partly "
                                     "shared construction, not independent support"},
        "zones": NA("district totals; no model zones"),
        "priors": NA("no priors; least squares on the calibration years"),
        "burn_in": NA("no model run"),
        "folds": "calibration 2015-2021, held-out 2012-2014 and 2022 (leave-one-year-out "
                 "inside the calibration years for the share sd only)",
        "fresh_units": [HELDOUT_UNIT],
        "thresholds": {"X5C_agreement_r_min": K.THRESHOLD_R, "synthetic": K.SYN_PASS},
        "gate_scores": NA("declaration: agreement check, no weighted gate"),
        "primary_model": "T3 + proportional first-difference Denton",
        "inputs": [rel_to_repo(p) for p in inputs],
        "attempt_policy": {"engineering_retries": "free before any metric is seen (crash, NaN, "
                                                  "missing file), each logged in the report",
                           "scored_runs": "the X5C check once", "exploratory_followups_max": 1,
                           "followup_label": "post-failure, exploratory; never validation; "
                                             "cannot promote",
                           "thresholds_frozen": True},
        "interpretation": (
            "Agreement says the indicator model can carry W1-like monthly timing to years it "
            "was not fitted on; it says nothing about how well either matches the real "
            "monthly diversions, which are not held. The annual non-Jiji volume is an "
            "accounting difference of two held tables with a declared definitional sd; the "
            "monthly Changhua term inherits both uncertainties. Its value for the twin is "
            "decided only by the stage-2 rung (draft spec in the report), not here."),
        "no_prior_look_statement": "X5 has not read W1 Yunlin monthly values of 2012-2014 or "
                                   "2022. Prior design-time inspections are logged by "
                                   "look_x5.py prior (Jan-May 2015-2021 table, MOA office "
                                   "rows, v3 attributes).",
        "written_by": rel_to_repo(Path(__file__)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    if errs or not a.write:
        return 1 if errs else 0
    path, sha = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
