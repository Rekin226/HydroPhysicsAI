"""Declare X5.record_low_tf.v1 (harness.prereg3, kind 'declaration'; design-time label).

Route 2 of the 2026-10-25 data audit: the 2021 record-low natural experiment, depth-resolved
(Q1, Q7). Rings 2012-2021 were design-time data for the twin's compaction column (Stage-2
target of calibrate_mlcw, the rings_independent check of calibrate_coupled, the S1a/S1b/S3
ring looks), and the nest heads 2012-2022 are inside many earlier looks, so no unit of this
comparison is fresh: it is declared as design-time evidence for the response shape, with its
thresholds, estimator and alternatives fixed here before any ring or head value is read.

Run under the shared lock:
    flock results/dataprep/preregistration/.x5.lock python prereg_x5.py [--write]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import look_x5  # noqa: E402
import x5const as K  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
CODE = ["x5const.py", "x5lib.py", "synth_x5.py", "score_x5.py", "explore_x5.py", "look_x5.py",
        "prereg_x5.py", "avail_x5.py"]
TWIN = ["hydrophysics/twin/compaction.py", "hydrophysics/twin/mechanism.py",
        "hydrophysics/twin/zones.py"]


def inputs() -> list[str]:
    rings = sorted(glob.glob(str(K.RING_DDIR / "ls_cache" / "clean" / "ls-wra-mlcw-obs__*")))
    return ([rel_to_repo(K.CODE_DIR / c) for c in CODE] + TWIN
            + [rel_to_repo(Path(p)) for p in rings]
            + [rel_to_repo(K.RING_SITES), rel_to_repo(K.PAGE_COLUMN),
               rel_to_repo(K.HEADS_DIR / "heads_monthly.parquet"),
               rel_to_repo(K.HEADS_DIR / "station_meta_preorigin.parquet"),
               rel_to_repo(K.HEADS_DIR / "MANIFEST.json")])


def doc() -> dict:
    return {
        "id": K.PREREG_ID, "wp": K.WP, "kind": "declaration",
        "title": "X5: does ring-band compaction of the 2020-21 record-low years follow head "
                 "relative to its running prior minimum? Two-input transfer-function "
                 "(Riley/preconsolidation) model per ring site and depth band (design-time)",
        "tests": NA("declaration: design-time evidence for response shape, never validation; "
                    "no fresh units exist (rings 2012-21 and nest heads were design-time data)"),
        "external_units": look_x5.members(),
        "checks": {
            "order": "1 synthetic recovery check (synth_x5.py, once); 2 design-time comparison "
                     "(score_x5.py, once); 3 exploration of 2023-25 (explore_x5.py, once, U7)",
            "data": {
                "rings": "the 14 MLCW sites (results/twin/mechanism/rings_sites.csv); ring "
                         "positions from chou-shui-data/data/ls_cache/clean via "
                         "twin.mechanism.ring_positions, TRUNCATED at 2021-12 before "
                         "twin.mechanism.band_compaction forms the bands (edges 0/100/200 m "
                         "below the top ring; the last band closes at the deepest ring); the "
                         "three depth bands are the series, 'total' is not scored",
                "heads": "results/dataprep/heads/L1_clean/v3 heads_monthly, C20 months, flags "
                         "0/1, through the loader guard (use design_time_evidence, window_end "
                         "2021-12); WellDepth from station_meta_preorigin (columns sid, "
                         "WellDepth only)",
                "primary_input": "per band, the nest well (rings_sites.csv wells) whose "
                                 "WellDepth is nearest the band's mid-depth among wells "
                                 "passing the gap rule (ties: lower sid)",
                "gap_rule": "interpolate inside gaps <= 6 months; every month 2014-01..2021-12 "
                            "present after that; >= 12 observed months in 2012-01..2013-12; "
                            "leading gaps take the first value",
                "time_axis": "monthly 2012-01..2021-12; t in years from 2012-01; column step "
                             "30 days (twin VEPColumn)"},
            "estimator_primary": (
                "c(t) = a + r t - A R[x_e; tau_e](t) + A0 x_i(t) + B R[x_i; tau_v](t), with "
                "hpc(t) = min(min of heads 2012-01..2013-12, min of heads 2014-01..t) the "
                "running prior minimum, x_e = h - hpc (head above it: elastic input), "
                "x_i = hpc_burn - hpc (depth of new lows: inelastic input), R the exponential "
                "response y_t = (1-d) x_t + d y_(t-1), d = exp(-30/tau); A, A0, B >= 0 by "
                "bounded least squares (scipy lsq_linear bvls); (tau_e, tau_v) by least "
                "squares over tau_e in {0 (instant), 30, 61, 91, 182} d x tau_v in 21 "
                "log-spaced values 30..7305 d; fitted on the ring months of 2014-01..2019-12; "
                "if x_i is 0 over every fit month the new-low terms are 0 (B = A0 = 0)"),
            "baselines": {
                "persistence": "OLS rate of the band's ring months 2018-01..2019-12 (>= 12), "
                               "continued from the anchor survey",
                "linear_head": "c = a + r t + k h by OLS on the same fit months (same trend, "
                               "head level, no preconsolidation split)"},
            "metric": "per band: errors of the change since the anchor survey (the last ring "
                      "month in 2019-07..2019-12) at every ring month of 2020-01..2021-12; per "
                      "site: RMSE pooled over its scorable bands and months (mm)",
            "support": "a band is scored only if it has >= 24 fit months, >= 12 test months, "
                       ">= 12 persistence months, an anchor survey and a primary head; a site "
                       "with no scored band does not pass",
            "success_rule": "a site passes if RMSE_TF <= 0.80 x RMSE_persistence AND "
                            "RMSE_TF <= 0.80 x RMSE_linear_head; declared success if >= 10 of "
                            "the 14 sites pass",
            "alternatives_declared_reported_never_gating": {
                "vep_form": "the same fit with A = A0 and tau_e = 0 (the twin column's VEP "
                            "form plus a linear trend)",
                "nest_mean_input": "the nest-mean head (mean over the valid nest wells of "
                                   "each well's anomaly about its 2012-2019 mean) for every "
                                   "band, with TF and with the VEP form"},
            "synthetic_check": {
                "what": "for each supported site-band: the twin's own column "
                        "(hydrophysics.twin.compaction.vep_compaction, h_pc0 = burn-in "
                        "minimum - h(2012-01)) driven by the band's REAL primary head "
                        "2012-2021, plus a 3 mm/yr trend and iid Gaussian noise, sampled on "
                        "the band's real surveyed months (pattern only); the X5 estimator "
                        "refitted on 2014-2019 exactly as on real data",
                "truth": list(K.SYN_CASES), "noise_m_gating": K.SYN_NOISE_M,
                "noise_m_sensitivity_not_gating": K.SYN_NOISE_SENS_M,
                "seeds": K.SYN_SEEDS, "seed0": K.SYN_SEED0,
                "identifiable": f"heads reach >= {K.SYN_D_ID_M} m below the burn-in minimum "
                                f"within 2014-01..2019-12",
                "pass_rule": f"in EACH truth case, at >= {K.SYN_FRAC:.0%} of identifiable "
                             f"bands the median over seeds of B_hat/Skv lies in "
                             f"{list(K.SYN_SKV_BAND)} and of tau_v_hat/tau in "
                             f"{list(K.SYN_TAU_BAND)}; fewer than {K.SYN_MIN_BANDS} "
                             f"identifiable bands = not evaluable = fail",
                "if_fail": "the comparison still runs once and is reported; the fitted gains "
                           "and delays are then labelled not interpretable and give no "
                           "stage-2 prior"},
            "exploration_U7": "2023-01..2025-03 (never gates, cannot promote): observed band "
                              "and total-interval OLS rate ratios 2023-25 / 2020-21 against "
                              "those predicted by the TF (FIT fit and a 2014-2021 refit), the "
                              "linear-head refit, and the twin's page column "
                              "(vep_zonal_leveling.json, zone of the site's easting) driven by "
                              "the observed nest-mean head; head change and new-low depth "
                              "after 2021",
            "decision_mapping": {
                "success_and_synthetic_pass": "write a stage-2 rung spec for a depth-banded "
                                              "preconsolidation column (layer heads, burn-in "
                                              "preconsolidation) for the lead's GPU batch",
                "success_synthetic_fail": "report; rung spec limited to the response form "
                                          "(no parameter priors from X5)",
                "fail": "no column change motivated by this route; report which part failed "
                        "(fit-window identifiability vs. response form) and the next route",
                "alternatives": "an alternative passing while the primary fails is reported "
                                "and cannot promote"},
            "attempt_policy": "each script runs once; engineering retries (crash, NaN, missing "
                              "file) before any metric is printed are free and logged in "
                              "results/dataprep/x5_record_low/attempts.log; at most one "
                              "exploratory follow-up after a failure, labelled 'post-failure, "
                              "exploratory; never validation; cannot promote'"},
        "zones": NA("per site and band; the page column's zones only in the exploration"),
        "priors": NA("none: per site-band least squares on a fixed grid"),
        "burn_in": "2012-01..2013-12 heads set the initial preconsolidation head; rings have "
                   "no survey 2012-03..2014-03 and no ring month before 2014-01 is fitted",
        "folds": NA("per site-band time split: fit 2014-01..2019-12, predict 2020-01..2021-12"),
        "fresh_units": NA("no fresh units: rings 2012-2021 were design-time data for the twin "
                          "column, nest heads 2012-2022 lie inside earlier looks"),
        "thresholds": {"improve_factor": K.IMPROVE, "min_pass_sites": K.MIN_PASS_SITES,
                       "n_sites": K.N_SITES, "min_fit_obs": K.MIN_FIT_OBS,
                       "min_test_obs": K.MIN_TEST_OBS, "min_persist_obs": K.MIN_PERSIST_OBS,
                       "max_gap_months": K.MAX_GAP, "syn_skv_band": list(K.SYN_SKV_BAND),
                       "syn_tau_band": list(K.SYN_TAU_BAND), "syn_frac": K.SYN_FRAC,
                       "syn_min_bands": K.SYN_MIN_BANDS, "syn_D_id_m": K.SYN_D_ID_M},
        "gate_scores": NA("design-time label; reported whichever way it falls"),
        "primary_model": "two-input transfer function above (estimator_primary), primary "
                         "head input",
        "inputs": inputs(),
        "interpretation": (
            "A pass says that, at the ring sites in 2014-2021, band compaction follows head "
            "relative to its running prior minimum (new lows load the clay with a delay) "
            "better than a constant rate or a head-level regression: design-time evidence "
            "for the preconsolidation response shape that the twin's VEP column already "
            "encodes, never validation of the twin. A fail says that this shape, fitted on "
            "2014-2019, does not predict the 2020-21 record-low years at >= 10 sites, which "
            "is reported with the part that failed (no new lows in the fit window to learn "
            "from, or the form). Neither outcome measures the policy response size."),
        "no_prior_look_statement": (
            "WP X5 has read no ring compaction value and no head value of any window. It read "
            "the 2026-10-25 audit report, which quotes from other agents: ring rate 2023-25 / "
            "2020-21 median 0.48 (n 13), GNSS 0.51 (n 10), mean head change -0.04 m, 19 of 33 "
            "ring-site wells 0.7-3 m below their prior minimum in 2021, ring band shares "
            "28/32/40 %. X5 logged one availability look (2026-10-25T14:51Z, counts of "
            "surveyed ring months per band and head gap-rule pass/fail; 42 of 42 site-bands "
            "meet the support rules). Rings 2012-2021 and the nest heads were seen by S1a, "
            "S1b, S3, N1, the twin's calibrate_mlcw/calibrate_coupled/mechanism runs and the "
            "audit agents: hence the design-time label."),
        "written_by": rel_to_repo(Path(__file__)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    missing = [p for p in d["inputs"] if not (REPO / p).exists()]
    print("inputs:", len(d["inputs"]), "missing:", missing or "none")
    if not a.write or errs or missing:
        return 1 if (errs or missing) else 0
    path, h = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
