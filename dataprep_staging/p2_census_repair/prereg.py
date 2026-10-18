"""Pre-registrations for WP P2a-c, written BEFORE the registered tests run (plan 4.5).

The A1 pre-registration tool (``dataprep_staging/harness/prereg``) did not exist when these
were written, so this script writes them in the same place and shape as the other WPs did
(``results/dataprep/preregistration/<id>.json``), refusing to overwrite an existing file and
recording the sha256 of each file. The billing pilot's results (design-time evidence) had been
read when these were written; the thresholds are the plan's, not chosen from new results.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402

PREREG = lib.REPO / "results" / "dataprep" / "preregistration"
WRITER = "dataprep_staging/p2_census_repair/prereg.py (A1 tool not available yet)"
PLAN = "docs/DATA_REMEDIATION_PLAN_2026-10.md, WP P2"

COMMON = {
    "census": {"pumps": "AMP_V2/data/tpc_pumps.parquet (columns: sid, TOWN, PURPOSE, PUMP_HP, "
                        "W_TUBE_DIA, 每期用電度數1, 新設日期1, 電號1, TWD97_X, TWD97_Y; never 戶名1 or "
                        "用電地址2)",
               "kwh": "AMP_V2/data/pump_kwh_all.parquet (2007-01..2025-07)"},
    "meter": "pd.factorize(str(電號1)); series = first pump's copy; kept = mean monthly kWh "
             "2012-01..2022-12 <= 1.0 x summed HP x 0.746 x 730 (the twin's frozen dedupe-cap set)",
    "phase": "pilot s02 rule on identical-pair fractions; pre-origin variant: pairs in "
             "2012-01..2022-12; full variant: 2012-01..2025-06",
    "profile": "stagger deconvolution (pilot disagg_lib, lambda 0.01, 12-month share window) of "
               "the kept irrigation phase-A (bill M2) and phase-B (bill M1) meters on the active "
               "grid, using only pairs complete by the variant's end (pre-origin: 2022-12)",
    "chow_lin": "X = [1, p/mean p]; AR(1) residual, rho by likelihood per meter from {0, 0.5, "
                "0.9}; window 2011-01..variant end (pairs entirely inside); each pair's split "
                "clipped to [0, Y] preserving the pair sum",
}


def registrations() -> list[dict]:
    return [
        {
            "id": "P2a.meter_validation.v1",
            "question": "Does the pre-origin Chow-Lin split of bimonthly irrigation bills "
                        "recover truly monthly meters better than the even split at every "
                        "group size from 1 to 300 meters?",
            "units": "kept irrigation meters whose pre-origin phase is 'monthly', finite in "
                     "every month 2012-01..2022-12 with positive total (the pilot's s08b/s11 "
                     "set; design-time units: the pilot chose its method on them)",
            "pseudo_phase": "phase (A/B) of the nearest kept irrigation bimonthly meter "
                            "(first-pump coordinates)",
            "estimator": "production pre-origin operator applied to the even-split record",
            "eval_months": "2012-02..2022-11 (pilot ev slice)",
            "groups": "k-means on coordinates (n_init 3, random_state 0), n_clusters = N // n "
                      "for n in {1, 3, 10, 30, 100, 300}; plus fan-wide (reported)",
            "metric": "pooled skill 1 - SSE_chowlin / SSE_even over all groups",
            "criterion": "skill >= 0.15 at every n in {1, 3, 10, 30, 100, 300} (plan P2a)",
            "secondary_reported_not_gating": [
                "pro-rata by the same profile (baseline)",
                "full variant (profile and fit to 2025-07; eval 2012-02..2025-05)",
                "placebo (year-shuffled profile, 20 seeds) skill on the same meters",
                "meter-level skill of the December-2022 phase-B extrapolation at pseudo-origins",
            ],
        },
        {
            "id": "P2a.sd_coverage.v1",
            "question": "Are the 90 % intervals of the Chow-Lin split calibrated?",
            "sd_model": "standardised error z = (chowlin - truth) / even-split value per group "
                        "and month (zero pairs excluded: their error is exactly 0); half-width "
                        "q90(|z|) per (group-size n in {1,3,10,30,100,300}, calendar month) "
                        "from a calibration half; sd = q90 / 1.645 x even value; cells "
                        "interpolate log-linearly in the effective number of Chow-Lin meters "
                        "(clamped to [1, 300])",
            "split": "validation meters split at the median easting (west/east); k-means "
                     "groups formed within each half; fit q90 on one half's groups, score "
                     "coverage on the other half's groups, both directions, hits pooled",
            "strata": "{1}, {3}, {10}, {30}, {100, 300} meters per group (100 and 300 pooled "
                      "because each half has few such groups, plan 4.4)",
            "criterion": "in every stratum the 95 % Wilson interval of the 90 % coverage, "
                         "with n replaced by the effective n = n / (1 + (m - 1) ICC) (clusters "
                         "= groups, m = mean months per group, ICC of hits within groups), "
                         "contains 0.90 (plan 4.4); the point window 0.85-0.95 applies only "
                         "where the effective n >= 30",
            "on_failure": "the sd is labelled uncalibrated in the manifest and report; the sd "
                          "model is not tuned after seeing the held-out coverage",
        },
        {
            "id": "P2a.head_placebo.v1",
            "question": "Does the real fan-wide profile improve the local-energy head "
                        "regressions beyond regional-signal injection, against a year-shuffled "
                        "placebo profile?",
            "heads": "fan_stations.parquet wells, monthly means 2012-01..2022-11 by the pilot "
                     "s06 rule (daily median, 15 x MAD despike, 15-day rolling median +-5 m, "
                     ">= 10 days); n_months >= 60; layer 1-4; 'y x' coordinates swapped",
            "energy": "pre-origin variant; local energy = sum over kept pumps within r km of "
                      "the well of HP share x meter series (recorded, Chow-Lin, placebo)",
            "covariates": "intercept; high-passed rain lags 0-2 (nearest CODIS-rebuilt gauge "
                          "with >= 90 % months, monthly sum with >= 20 days); high-passed "
                          "Yunlin total kWh and high-passed kWh of kept pumps > 15 km away, "
                          "both computed from the same energy version as the local series",
            "high_pass": "x minus its centred 25-month rolling mean (min 12 months)",
            "statistic": "partial R2 of the high-passed local energy; per well "
                         "D_w(r) = pr2(real Chow-Lin) - mean_k pr2(placebo k), k = 1..20 "
                         "(placebo profiles: calendar years 2007..2022 of the pre-origin "
                         "profile permuted by a derangement, seeds 1..20)",
            "primary": "mean over wells of D_w = mean of D_w(r) over r in {1, 2} km where "
                       "defined; site-cluster bootstrap (site = first 6 characters of the "
                       "station id), 4000 resamples, seed 2026; one-sided "
                       "p = (1 + #{boot mean <= 0}) / 4001",
            "criterion": "p < 0.05 (plan P2a)",
            "on_failure": "P2a ships on the meter validation; the head gain is reported as not "
                          "distinguishable from regional-signal injection",
            "secondary_reported_not_gating": [
                "each radius r in {1, 2, 3, 5} km separately",
                "real Chow-Lin vs recorded with the same covariate design",
                "fixed-nuisance design: both recorded and Chow-Lin regional covariates in "
                "every regression",
                "share of wells where Chow-Lin beats recorded (the pilot's statistic)",
            ],
            "u7": "heads 2012-2022 only; no 2023+ observation enters",
        },
        {
            "id": "P2b.july_backtest.v1",
            "question": "How well does the meter's own 3-year same-month mean impute a missing "
                        "phase-A July?",
            "units": "kept meters with bill M2 and full-variant phase A (pilot s09)",
            "backtest": "rolling origin, July of each year 2013-2024; truth = the meter's "
                        "recorded July (half its Jul-Aug bill); estimate = mean of its July "
                        "values in the three previous years",
            "metric": "fan-total error |sum est - sum truth| / sum truth per year",
            "criterion": "mean over years <= 20 % (plan P2b)",
            "secondary_reported_not_gating": [
                "2013-2022 only (pre-origin years)", "max error", "1 km cell relative RMSE",
                "leave-one-year-out coverage of the cell-level 90 % interval",
                "M0 July/June ratio 2022-2025",
            ],
        },
        {
            "id": "P2c.zero_reproduction.v1",
            "question": "Does the ported zero classification reproduce the pilot?",
            "criteria": [
                "pre-install zero meter-months (pilot definition) within 1 % of 878,738",
                "zero-month counts by kind x length bin (kept A/B/monthly meters, "
                "2012-01..2025-06) each within 1 % of the pilot's run table "
                "(zero_runs_summary.csv: number of runs per class x kind x bin) and of the "
                "total 2,124,833 zero-months",
                "interior runs <= 12 months as a share of zero-months within 1 % of 0.579",
                "irrigation zero frequency in April and August within 1 % of 0.0558 and "
                "0.1563 (pilot census_zeros.json)",
            ],
        },
    ]


def main() -> None:
    PREREG.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = {}
    for r in registrations():
        doc = {"id": r["id"], "wp": "P2abc", "plan_section": PLAN, "common": COMMON, **r,
               "anti_leakage": "pre-origin variant: every fitted quantity (phase, profile, "
                               "beta, rho, sd model) uses data to 2022-12; no 2023+ "
                               "observation enters any gate or threshold (U3, U7)",
               "created_utc": now, "written_by": WRITER}
        p = PREREG / f"{r['id']}.json"
        with open(p, "x", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
            f.write("\n")
        out[r["id"]] = hashlib.sha256(p.read_bytes()).hexdigest()
    lib.save_json({"written_utc": now, "sha256": out}, lib.WORK / "prereg_hashes.json")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
