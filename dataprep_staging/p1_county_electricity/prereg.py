"""WP P1 pre-registration (plan 4.5), written BEFORE the registered tests run.

The A1 pre-registration tool does not exist yet (A1 has not reported), so this script writes
the JSON itself with the fields plan 4.5 names (tests, units, thresholds, seeds, primary
choices, input hashes), refuses to overwrite (mode 'x', then read-only) and records the file's
sha256 in the P1 ledger fragment.

Usage: python -I prereg.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import PREREG_DIR, REPO, rel_to_repo  # noqa: E402

PID = "P1_county_electricity_2026-10-12"
BUILD = REPO / "results/dataprep/county_electricity/_build"
INPUTS = [
    REPO / "data_fetch_api/new_measurements/tpc_d007012/2026-10-12/d007012.csv",
    REPO / "data_fetch_api/new_measurements/moeaea_agri_electricity/2026-10-12/set_236.csv",
    REPO / "data_fetch_api/new_measurements/moeaea_agri_electricity/2026-10-12/set_256.csv",
    REPO / "data_fetch_api/new_measurements/tpc_d003004/2026-10-12/001.zip",
    REPO / "AMP_V2/data/tpc_pumps.parquet",
    REPO / "AMP_V2/data/pump_kwh_all.parquet",
    BUILD / "census/census_class_month_full.parquet",
    BUILD / "census/census_class_month_preorigin.parquet",
    BUILD / "phi/phi_in.json",
]
CODE = [HERE / n for n in ("lib.py", "census_monthly.py", "phi_in.py", "prereg.py")]

PRE = {
    "id": PID,
    "wp": "P1",
    "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP P1, 4.4, 4.5, 4.7",
    "tool": ("hand-written by WP P1: the A1 pre-registration tool does not exist yet. Same "
             "content as plan 4.5 asks; written with mode 'x' and made read-only; sha256 in "
             "ledger/rows/P1.csv (number_id P1.prereg)."),
    "looks_before_this_file": {
        "design_time_numbers_already_known": [
            "source separation a, b, R2 on both windows (plan P1 table, TPC scout, plan "
            "revision); reproduced by P1 before this file: full 2016-01..2025-06 a=0.9356 "
            "[0.8928, 0.9747] b=1.4099 R2=0.9648; pre-origin 2016-01..2022-12 a=0.9591 "
            "[0.9197, 1.0083] b=1.3953 R2=0.9635 (year-block bootstrap, 2000 draws, seed 0)",
            "LOYO extension annual RMS 6.7 % (TPC scout, 2016-2024)",
            "matched-lead backcast WITHOUT trend, shares 2020-23 -> 2016-19 (plan revision): "
            "CH +2.4 +4.8 +2.5 +2.6 %, YL +4.7 +0.1 +1.2 +4.3 %",
            "LOYO backcast 2.5-3.1 % (TPC scout; interpolation, not used for acceptance)",
            "phi_in: P1's own two runs of phi_in.py before this file (MOI township area 0.747, "
            "TW_COUNTY polygon 0.790, planted-weighted 0.823, label 0.823)",
            "census model-unit totals reproduce the TPC pilot's census_class_month.csv "
            "(max abs 5e-6 GWh, CSV rounding)",
        ],
        "not_run_before_this_file": [
            "T1 BV1", "T2 BV2 with trend", "T5 EXT-season", "T6 EXT-2023-25", "T7 EXT-rolling",
            "T8 SYN coverage (both windows, both noise models)", "T9 MIS (all cases)",
            "T4 EXT-LOYO as rebuilt by P1 (the 6.7 % is the scout's number)",
        ],
    },
    "production_choices_fixed_here": {
        "model_units": ("YL_clean = Yunlin cleaned census, all purpose classes "
                        "(hydrophysics.twin.pumping.clean_census, cap_duty 1.0, duty "
                        "statistic over 2012-01..2022-12, full record allocated over the "
                        "kept pumps); census months from 2025-07 are masked (partial July)"),
        "nnls_windows": {"preorigin": ["2016-01", "2022-12"], "full": ["2016-01", "2025-06"]},
        "nnls_bootstrap": {"type": "year blocks with replacement, percentile 90 %",
                           "n_boot": 2000, "seed": 0},
        "backcast": {"months": ["2012-01", "2015-12"], "moy_window": ["2016-01", "2019-12"],
                     "trend_window": {"preorigin": ["2016-01", "2022-12"],
                                      "full": ["2016-01", "2026-08"]},
                     "rule": "share(m,t) = mean share of month m over the moy window + beta * "
                             "(t - mean t of the moy window); beta = OLS slope with 12 "
                             "month-of-year intercepts on the trend window",
                     "sd": "value x monthly relative RMS of the matched-lead test of the same "
                           "variant (T1 preorigin, T2 full), with trend"},
        "transfer_ratio_window": {"preorigin": ["2016-01", "2022-12"],
                                  "full": ["2016-01", "2025-06"]},
        "extension": ("full variant only (U4): YL_clean_hat = YL_ag x r_full(moy) for "
                      "2025-07..2026-08; Changhua = a x YL_clean_hat, both a_pre and a_full "
                      "columns, a_pre primary for the recorded-forcing arm (same estimator "
                      "as the calibration window; no splice)"),
        "estimator_ratio_rule": ("each estimator's ratio = Changhua pump-like / its own Yunlin "
                                 "counterpart over the variant's NNLS window: scaled a; shape "
                                 "mean(CH_ag - b N_YL)/mean(YL_clean); transfer "
                                 "mean(CH_ag r(m))/mean(YL_clean); irrigation-only a_irr "
                                 "(NNLS with P = YL_clean_irrigation); national sub-industry "
                                 "(d003004, weak) CH/YL ratio of the 011+013+032 fitted "
                                 "contributions of an NNLS of county ag on national 011, 012, "
                                 "013, 032 (2021-01 on; preorigin 2021-01..2022-12)"),
        "in_grid_anchor": ("mu_elec_in_grid(t) = a_pre x YL_clean(t) x phi_in, phi_in = "
                           "provisional P1 value (phi_in.json, P3 supersedes)"),
        "prior_log_sd": ("sigma_mu = sqrt(0.5^2 + sigma_ln_phi^2 + b_mis^2), b_mis = the largest "
                         "|ln(mean a_hat / a_eff)| among T9 cases M1-M4 whose |mean a_hat - "
                         "a_eff| exceeds the real-data bootstrap half-width of the same window "
                         "(else 0); preorigin window decides the preorigin prior"),
        "cv_j": ("provisional only: sd/mean of the even-split cleaned census class series "
                 "(irrigation; all other classes) 2012-02..2022-12; P2a's Chow-Lin product "
                 "supersedes (plan P1 step 7)"),
    },
    "tests": {
        "T1_BV1": {
            "role": "gate, preorigin variant (data <= 2022-12 only)",
            "design": "share model with moy window 2019-01..2022-12 and trend window "
                      "2019-01..2022-12 (all pre-origin county data on the far side of the "
                      "target); predict 2016, 2017, 2018 (leads 1-3 years) from the national "
                      "series; annual error 100*(sum pred/sum obs - 1) per county-year",
            "acceptance": "per county (CH and YL separately): RMS over the 3 years <= 5 % and "
                          "|mean bias| <= 3 %, with the trend",
            "also_reported": "same without trend; monthly relative RMS",
        },
        "T2_BV2": {
            "role": "gate, full variant (the plan's matched lead; uses 2023-26 TPC data)",
            "design": "moy window 2020-01..2023-12, trend window 2020-01..2026-08; predict "
                      "2016-2019 (leads 1-4)",
            "acceptance": "per county: RMS <= 5 % and |mean bias| <= 3 %, with the trend",
            "also_reported": "same without trend (reproduces the plan's numbers)",
        },
        "T3_BV_LOYO": {"role": "reported only (interpolation)",
                       "design": "leave one year out 2016-2025, moy shares of the other "
                                 "years, no trend"},
        "T4_EXT_LOYO": {
            "role": "gate, full variant",
            "design": "r(moy) from 2016-01..2025-06 minus year y; predict YL_clean for "
                      "y = 2016..2024; annual error %",
            "acceptance": "RMS over 2016-2024 <= 10 %",
            "also_reported": "RMS over 2016-2022; monthly relative RMS (extension sd)",
        },
        "T5_EXT_season": {"role": "reported only",
                          "design": "leave out each quarter block (JFM, AMJ, JAS, OND) of "
                                    "2016-2024 and 2025 Q1-Q2; r from the remaining months of "
                                    "2016-01..2025-06; quarter-total error %, RMS per quarter "
                                    "and pooled"},
        "T6_EXT_2023_25": {"role": "reported only",
                           "design": "T4 errors for 2023 and 2024; 2025-01..06 block predicted "
                                     "with r from 2016-01..2024-12"},
        "T7_EXT_rolling": {"role": "reported only",
                           "design": "forward-only 14-month blocks starting July 2018..2023; r "
                                     "from 2016-01..(start - 1); error of the 14-month total "
                                     "and of the first 12 months"},
        "T8_SYN_coverage": {
            "role": "gate on the preorigin window; reported (and gate of the full variant) on "
                    "the full window",
            "design": "truth (alpha, beta) = production (a, b) of the window; CH_syn = alpha "
                      "YL_clean + beta N_YL + noise; noise = stationary Gaussian AR(1) with "
                      "the production residual's sd and lag-1 autocorrelation; R = 1000 "
                      "replicates, replicate r uses numpy default_rng(7000 + r) for its noise "
                      "and its bootstrap; per replicate the production NNLS and a year-block "
                      "percentile bootstrap with B = 1000; coverage = share of replicates "
                      "whose 90 % interval contains alpha",
            "acceptance": "plan 4.4: the Wilson 95 % interval of the coverage on n_eff = R "
                          "(independent replicates, design effect 1) contains 0.90, and the "
                          "point coverage lies in 0.85-0.95 (n_eff >= 30)",
            "secondary_noise_model": "reported only: deseasonalised production residuals "
                                     "resampled by year blocks",
            "remedy_if_failed": "reported only, never a pass: a calibration factor k on the "
                                "bootstrap half-widths fitted on replicates 0-999 to give "
                                "0.90 coverage, verified on fresh replicates 1000-1999",
        },
        "T9_MIS": {
            "role": "biases reported and folded into the prior width (plan acceptance 4)",
            "design": "as T8 (same seeds, R = 1000, AR(1) noise) with the Changhua pump "
                      "component alpha x P' instead of alpha x YL_clean: M1 seasonal "
                      "component x 0.5, M2 x 1.5 (P' = P + (k-1) c(m), c = month-of-year mean "
                      "minus window mean), M3 seasonal component shifted one month later "
                      "(P' = P - c(m) + c(m-1)), M4 P' = 1.5 P in Jan-May of 2015, 2020, "
                      "2021; report mean a_hat - alpha and mean a_hat - a_eff, a_eff = alpha "
                      "mean(P')/mean(P) over the window; extra reported case M3b (one month "
                      "earlier)",
            "rule": "see production_choices_fixed_here.prior_log_sd",
        },
    },
    "gates": {
        "preorigin": ["T1_BV1 (both counties)", "T8_SYN_coverage preorigin window",
                      "T9 folded into the prior width"],
        "full": ["T2_BV2 (both counties)", "T4_EXT_LOYO", "T8_SYN_coverage full window",
                 "T9 folded into the prior width"],
        "rule": "gate.passed = every listed criterion passes; no threshold changes after "
                "results",
    },
    "U7_note": ("no head, ring, pile, GNSS or InSAR observation enters any P1 test. T2, T4-T7 "
                "use 2023-26 TPC electricity and census data; they gate only the full "
                "variant, which feeds the recorded-forcing arm and after-the-fact analysis. "
                "The preorigin gate (T1, T8 preorigin) uses data <= 2022-12 only."),
}


def main() -> int:
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    p = PREREG_DIR / f"{PID}.json"
    rec = dict(PRE)
    rec["written_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["inputs"] = [{"path": rel_to_repo(x), "sha256": sha256_file(x)} for x in INPUTS]
    rec["code"] = [{"path": rel_to_repo(x), "sha256": sha256_file(x)} for x in CODE]
    with open(p, "x", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.chmod(p, 0o444)
    h = sha256_file(p)
    ledger.append_rows("P1", [{"number_id": "P1.prereg", "source": rel_to_repo(p),
                               "role": "design_time_evidence",
                               "product_version": "county_electricity/preregistration",
                               "note": f"sha256 {h}; written {rec['written_utc']}"}])
    print(p, h)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
