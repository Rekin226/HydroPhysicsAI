"""A1 registrations of the W1 repair round (harness.prereg2), written before the repaired
2026 uncertainty or its coverage is computed.

    python -I dataprep_staging/w1_canal/rep_prereg.py record|uncertainty|prospective

* ``record``: puts the legacy, W1-written pre-registration W1_janmay_benchmark_2026-10-12
  (02:40:16Z, before the A1 tool existed) into the hash-chained INDEX, as the A1 repair round
  asks for looks logged under ids outside the INDEX. Nothing about that test changes.
* ``uncertainty``: declaration W1.ext2026_uncertainty.r1 of the repaired 2026 error model and
  of the coverage rule that sets its label (design-time units; not a fresh test).
* ``prospective``: test W1.ext2026_prospective_obs.v1, the U1-compliant version of the
  registered prospective inflow test (verifier minor issue): only observed composite months
  are scored and the climatology baseline uses only observed months. The original test
  (W1.ext2026.prospective_inflow_2026) stays on file; its target and baseline admit flag-3/4
  values, so this one is the primary prospective test of the 2026 inflow prediction.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.paths import PREREG_DIR, rel_to_repo  # noqa: E402
from harness import prereg2  # noqa: E402

from w1_canal import config as C  # noqa: E402
from w1_canal import ext_config as E  # noqa: E402
from w1_canal import rep_config as RC  # noqa: E402

FULL_V1 = C.product_dir_base("full") / "v1"
FULL_V2 = C.product_dir_base("full") / "v2"
NA = prereg2.not_applicable
RECORD_ID = "W1_janmay_benchmark_2026-10-12.record"
UNC_ID = "W1.ext2026_uncertainty.r1"
PROSP_ID = "W1.ext2026_prospective_obs.v1"
PROSP_TEST = "W1.ext2026.prospective_inflow_2026_observed"
MONTHS = [f"2026-{m:02d}" for m in range(1, 10)]


def record_doc() -> dict:
    legacy = PREREG_DIR / "W1_janmay_benchmark_2026-10-12.json"
    return {
        "id": RECORD_ID, "wp": "W1", "kind": "declaration",
        "title": "Record of the W1-written pre-registration of the Jan-May canal benchmark "
                 "(W1_janmay_benchmark_2026-10-12) and of the looks already logged under it",
        "tests": NA("declaration: the test already ran under W1_janmay_benchmark_2026-10-12"),
        "zones": NA("district series; no spatial zones"),
        "priors": NA("no prior"),
        "burn_in": NA("no model run"),
        "folds": "leave-one-year-out over the Jan-May benchmark years 2015-2021 (as registered)",
        "fresh_units": NA("declaration; W1.janmay.{chc,yl}.2015..2021 are design-time units "
                          "(scout look seeded 2026-10-11T16:56:43Z; looks.jsonl)"),
        "thresholds": "from the record: v4 beats v3 and the climatological fraction in both "
                      "districts with the 90 % paired-bootstrap interval of the RMSE difference "
                      "below zero (B 10000, seed 20261012), decided on the pre-origin variant",
        "gate_scores": NA("declaration"),
        "primary_model": NA("declaration"),
        "inputs": [rel_to_repo(legacy)],
        "records": {
            "id": "W1_janmay_benchmark_2026-10-12", "path": rel_to_repo(legacy),
            "created_utc": "2026-10-12T02:40:16Z",
            "written_by": "dataprep_staging/w1_canal/prereg.py (before harness/prereg.py, "
                          "02:42Z, existed)",
            "looks": ["2026-10-11T16:56:43Z design_time_seed (scout surface_water_monthly)",
                      "2026-10-12T02:45:11Z..02:46Z dry run (logged 06:16:35Z)",
                      "2026-10-12T06:16:59Z product build (preorigin/v1, full/v1)"],
            "outcome": "pre-origin variant 4/4 comparisons with the upper bound below zero "
                       "(Yunlin vs climatology -0.002); full variant 2/4"},
        "interpretation": (
            "Puts the legacy pre-registration's sha256 into the hash-chained INDEX so that "
            "the looks logged under its id can be traced (A1 repair round). Its registered "
            "code and inputs were re-checked against their hashes in the W1 resume and "
            "repair rounds. Nothing here changes the test, its result or its label; the "
            "seven Jan-May years remain design-time units."),
    }


def uncertainty_doc() -> dict:
    return {
        "id": UNC_ID, "wp": "W1", "kind": "declaration",
        "title": "W1 repair round: repaired error model of the 2026 canal deliveries and the "
                 "coverage rule that sets their interval label (plan 4.7)",
        "tests": NA("declaration: a design-time coverage check on units already looked at "
                    "(not a fresh test)"),
        "zones": NA("district series (Changhua, Yunlin); no spatial zones"),
        "priors": NA("no prior"),
        "burn_in": NA("no dynamic model is started"),
        "folds": "double leave-one-year-out over the 11 evaluation years of "
                 "W1.ext2026_rain_runoff.v1 (2012, 2013, 2017-2025): the held-out year's chain "
                 "is the published single-fold prediction; the residual pool of the other "
                 "10 years comes from chains fitted without both years",
        "fresh_units": NA("design-time units only: W1.annual.{chc,yl}.<evaluation years> "
                          "(looked at by W1.ext2026_rain_runoff.v1), W1.janmay.{chc,yl}."
                          "2017..2021 (design-time seed), W1.composite.<observed years> "
                          "(regression targets of the chain)"),
        "thresholds": {
            "annual_rule": "22 hits (11 evaluation years x 2 districts; hit = observed Jiji "
                           "annual district total inside the 5th-95th percentile of the "
                           "ensemble's annual sums), clusters = years, "
                           "harness.coverage2.coverage_gate (gate-2, level 0.95, Student-t "
                           "quantile): pass -> the 2026 intervals carry the label "
                           f"'{RC.LABEL_2026_PASS}'; fail -> '{RC.LABEL_2026_FAIL}'",
            "reported_not_gated": [
                "Clopper-Pearson 95 % on n_eff = number of years (W1's earlier rule)",
                "Jan-May 2017-2021 (briefing table) hits of the same ensembles, 10 hits in "
                "5 year clusters",
                "inflow regression: value +- 1.645 g RMS (RMS over the pool folds by calendar "
                "month, g = t_0.95,n / z_0.95), floored at 0, on months whose composite is "
                "observed (every day flag <= 1); its gate sets the inflow sd label"],
            "historical_rows": "the existing full-variant Jan-May leave-one-year-out "
                               "coverage of the state-space draws (full/v1 "
                               "statespace_loo_coverage.csv, 2015-2021 x 2 districts, 14 "
                               "hits already reported as 10/14) is re-evaluated with the A1 "
                               "gate-2 interval (clusters = years); the 2012-2025 rows are "
                               "labelled 'Jan-May aggregate coverage-checked' if it contains "
                               "0.90, else 'Jan-May aggregate uncalibrated'; monthly scale "
                               "'uncalibrated' either way (rep_config.LABEL_HIST_TEMPLATE)",
            "error_model": "rep_config.py docstring steps 1-6 (residuals D_k(y,m) - c_y(m) "
                           "against the 200 full/v1 state-space draws; f = RMSE/SD of the "
                           "pool's annual residuals; g = t_{0.95,n}/z_{0.95}; smoothed "
                           "bootstrap with h = 1.06 n^-1/5 and a year-level shock profile "
                           "pi(m) = s_m/sum(s) SD, divided by sqrt(1+h^2); members centred "
                           "on their mean, scaled by f g, added to the value, truncated at 0; "
                           "structural-zero months 0 with sd 0)",
            "members": {"n": RC.N_DRAWS, "seed": RC.DRAW_SEED,
                        "coverage_ensembles": "every pool year x every full/v1 draw "
                                              f"(10 x 200), shocks seed {RC.COVERAGE_SEED} + "
                                              "held-out year"},
            "unchanged": "the 2026 values (registered chain), the 2012-2025 values and "
                         "members (full/v1)"},
        "gate_scores": NA("declaration"),
        "primary_model": NA("declaration: no model is selected; the error model is fixed "
                            "above"),
        "inputs": [rel_to_repo(p) for p in (
            HERE / "rep_config.py", HERE / "rep_uncert.py", HERE / "rep_build.py",
            HERE / "ext_model.py", HERE / "ext_config.py", HERE / "diversion.py",
            HERE / "sources.py", FULL_V1 / "MANIFEST.json", FULL_V1 / "district_draws.npz",
            FULL_V1 / "inflow_composite_daily.parquet", FULL_V1 / "validation.json",
            FULL_V1 / "statespace_loo_coverage.csv", FULL_V2 / "MANIFEST.json", FULL_V2 / "district_monthly.csv",
            FULL_V2 / "ext2026_loo_monthly.csv", FULL_V2 / "inflow_predicted_monthly.csv",
            FULL_V2 / "basin_rain_index_monthly.csv", FULL_V2 / "validation.json",
            FULL_V2 / "grid_sw_v4.npz", C.JIJI_CSV, C.BRIEFING_110_PDF,
            E.L0_DIR / "SOURCE.json")],
        "disclosure": (
            "Fixed after the verifier's report and after the chain's annual leave-one-year-"
            "out RMSE and bias per district were known (full/v2: 1.227 / 1.402, bias -0.257 / "
            "+0.504 x 10^8 m3); f uses exactly those quantities. The repaired ensembles and "
            "their coverage had not been computed when this was written. The code was run "
            "on synthetic random inputs (rep_uncert functions) and then once end to end in a "
            "scratch directory with the real chain folds (rain index, composite and Jiji "
            "totals as training data; predictions only) but synthetic full/v1 draws, "
            "synthetic held-out annual and Jan-May totals, a synthetic composite for the "
            "inflow check and a synthetic state-space coverage table, so no real coverage, "
            "hit or interval was computed; that run checked that the single folds reproduce "
            "full/v2's ext2026_loo_monthly.csv (max abs 4.4e-16) and the 2026 chain the "
            "shipped values (2.2e-16). The historical-row re-evaluation uses hits already "
            "reported (10/14). The double leave-one-year-out chain is new."),
        "interpretation": (
            "A pass means the annual-scale error model is not contradicted by the 11 design-"
            "time years (n_eff about 11; weak evidence, A1: the gate has little power with few "
            "clusters); it does not calibrate the monthly or Jan-Sep intervals, which have no "
            "observed target of the same kind and stay labelled 'uncalibrated' at those "
            "scales. A failure labels every 2026 interval 'uncalibrated'. The 2026 values are "
            "post-origin forcing (U4): recorded-forcing arm only, flag 6."),
    }


def prospective_doc() -> dict:
    ext = {f"W1.inflow.{m}": f"monthly composite inflow volume, {m} (2026 yearbook)"
           for m in MONTHS}
    return {
        "id": PROSP_ID, "wp": "W1", "kind": "test",
        "title": "U1-compliant prospective test of the 2026 inflow prediction (observed "
                 "composite months only; observed-only climatology)",
        "tests": [{
            "test_id": PROSP_TEST, "data_scope": "wra_hydro_yearbook:2026-01..2026-09",
            "fresh_units": [f"W1.inflow.{m}" for m in MONTHS], "external_units": ext,
            "metric": "RMSE over the scorable months of 2026-01..2026-09 of the predicted "
                      "monthly composite volume (10^8 m3; value column of "
                      "inflow_predicted_monthly.csv, identical in surface_water_v4/full/v2 "
                      "and v3) against the observed composite volume",
            "scorable_month": "all three gauges (H075 寶石橋, H049 內茅埔, H064 水里) have a "
                              "numeric yearbook daily value on every day of the month, none "
                              "excluded by QC (flag 9), and none of the three 2026 gauge-years "
                              "is more than 50 % off its 2012-2025 median annual ratio to the "
                              "weir inflow (the full/v1 rule that would replace the year by a "
                              "donor, flag 3); no interpolation (flag 4) and no donor (flag 3) "
                              "day enters",
            "observed_composite": "sum of the three gauges' daily mean discharges times the "
                                  "days, rescaled by the 2026 weir published annual inflow "
                                  "divided by the 2026 raw three-gauge annual volume (flag 1, "
                                  "as full/v1); if the 2026 weir inflow is not published or a "
                                  "gauge-year is incomplete, the raw sum is scored and the "
                                  "report says so",
            "baseline": "monthly climatology 2012-2025 of the full/v1 composite volume, "
                        "computed only from months whose every composite day is flag <= 1 "
                        "(U1: no donor-replaced or interpolated value in the baseline)",
            "pass_if": "at least 5 scorable months and RMSE(prediction) < RMSE(climatology) "
                       "over the scorable months; fewer than 5 -> 'not scorable'; a sign test "
                       "on the absolute errors is reported",
            "when": "after the 2026 Hydrological Year Book and the 2026 Jiji annual table are "
                    "published (expected mid-2027)"}],
        "zones": NA("one basin series; no spatial zones"),
        "priors": NA("ordinary least squares regression; no priors"),
        "burn_in": NA("no dynamic model is started"),
        "folds": {"prospective": "single fold: 2026-01..2026-09"},
        "fresh_units": [f"W1.inflow.{m}" for m in MONTHS],
        "thresholds": {"pass_if": "RMSE(prediction) < RMSE(observed-only climatology) over "
                                  ">= 5 scorable months",
                       "relation_to_original": "W1.ext2026.prospective_inflow_2026 "
                       "(W1.ext2026_rain_runoff.v1) stays on file; its target (the composite "
                       "with donor or interpolated days) and baseline (all 2012-2025 months) "
                       "admit flag-3/4 values, contrary to U1; this test is the primary "
                       "prospective test of the 2026 inflow prediction and the original is "
                       "reported only as secondary, with that caveat (two comparisons on the "
                       "same units)"},
        "gate_scores": [{"name": "rmse_prediction_minus_observed_climatology", "weight": 1.0}],
        "primary_model": {
            "prediction": "the registered rain-runoff regression of W1.ext2026_rain_runoff.v1 "
                          "(spec linear_lag2, fitted on 2012-2025), shipped unchanged",
            "use": "surface_water_v4/full/v2 and v3 (U4: recorded-forcing arm only)"},
        "inputs": [rel_to_repo(p) for p in (
            FULL_V2 / "inflow_predicted_monthly.csv", FULL_V1 / "inflow_composite_daily.parquet",
            FULL_V1 / "gauges_daily_L1.parquet", FULL_V1 / "validation.json",
            PREREG_DIR / "W1.ext2026_rain_runoff.v1.json", HERE / "composite.py",
            HERE / "config.py")],
        "interpretation": (
            "Registered before any 2026 yearbook value exists. It narrows the original "
            "prospective test to observed values (U1); it does not change the prediction. A "
            "pass is the first skill evidence for the 2026 extension; until then the 2026 "
            "months carry no skill evidence."),
    }


DOCS = {"record": record_doc, "uncertainty": uncertainty_doc, "prospective": prospective_doc}

if __name__ == "__main__":
    which = sys.argv[1]
    path, sha = prereg2.write_prereg(DOCS[which]())
    print(path, sha)
