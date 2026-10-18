"""WP C0: write the pre-registration of the C0 acceptance tests with the repaired A1 tool
(``harness.prereg2``) BEFORE any acceptance number is computed, after the products were built
(their files are hashed here, so they cannot change after the registration).

Usage: python -I prereg_c0.py            (writes results/dataprep/preregistration/C0.census_product.v1.json)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common.hashing import iter_files  # noqa: E402
from harness import prereg2  # noqa: E402

NA = prereg2.not_applicable
W = "2012-01..2022-12"


def unit(test_id: str, uid: str, desc: str, scope: str, what: str, metric: str,
         threshold: str, gating: str) -> dict:
    return {"test_id": test_id, "data_scope": scope, "fresh_units": [uid],
            "external_units": {uid: desc}, "what": what, "metric": metric,
            "threshold": threshold, "gating": gating}


TOL = "every element: |delta| <= 1e-6 kWh + 1e-9 x |reference|"
TESTS = [
    unit("C0.v1.T1a_model_path_identity", "C0:model_path_vs_cached_forcing_2012_2022",
         "the model's own forcing path re-run on AMP_V2 vs the cached calibrated forcing",
         f"model_forcing_E_by_class:{W}",
         "The model's own path (pumping.clean_census cap 1.0, t0 2012-01-01, t1 2023-01-01, "
         "then scenario.energy_by_class on the pickle's grid) re-run on AMP_V2/data/"
         "tpc_pumps.parquet + pump_kwh_all.parquet reproduces E_by_class of results/twin_runs/"
         "stage3_datum_gate_cwa/inputs.local.pkl (cached forcing of the calibrated model), and "
         "the pickle's grid equals results/twin_forward/datum_gate_innov.npz.",
         "max over class x active cell x month 2012-01..2022-12 of |delta|; class key set; "
         "dates; grid mask and parameters", TOL + "; identical class keys, dates and grid",
         "preorigin, full"),
    unit("C0.v1.T1b_c0_pipeline_on_amp", "C0:c0_pipeline_amp_vs_cached_forcing_2012_2022",
         "the C0 pipeline run on AMP_V2 vs the cached calibrated forcing",
         f"model_forcing_E_by_class:{W}",
         "build.process (clean_census kept set on 2012-2022, cap-None re-share of the whole "
         "record, dense per-pump matrix, pre-install structural zeros, sparse per-cell "
         "aggregation) on the AMP_V2 inputs reproduces the same E_by_class.",
         "max |delta| over class x cell x month 2012-01..2022-12", TOL, "preorigin, full"),
    unit("C0.v1.T2a_new_l0_common_rows",
         "C0:c0_pipeline_newL0_masked_vs_cached_forcing_2012_2022",
         "the C0 pipeline on the new L0 Yunlin sheet with AMP-absent readings masked",
         f"model_forcing_E_by_class:{W}",
         "build.process on the new L0 Yunlin sheet (new attributes and row order) with every "
         "pump-month reading the AMP_V2 cache lacks set to missing reproduces E_by_class "
         "(isolates the data difference from the code). Also reported: pump-months where both "
         "sources are finite but differ (expected 0), count and kWh of the masked readings.",
         "max |delta| over class x cell x month 2012-01..2022-12", TOL, "preorigin, full"),
    unit("C0.v1.T2b_product_yunlin_vs_model", "C0:product_yunlin_vs_cached_forcing_2012_2022",
         "the delivered pre-origin product's Yunlin county axis vs the cached forcing",
         f"model_forcing_E_by_class:{W}",
         "The delivered Yunlin part (pump_electricity/preorigin value_by_county['yunlin'], "
         "new L0) against the model's E_by_class, 2012-01..2022-12: the task's 'Yunlin part "
         "reproduces the model's cleaned forcing to within rounding', with a tolerance fixed "
         "here because the new sheet holds readings the AMP_V2 cache lacks (disclosed below).",
         "(a) per month: |sum over classes and cells of delta| / model fan total of the month; "
         "(b) sum of |delta| over all class x cell x months; also reported: max element |delta|",
         "(a) <= 1e-4 in every month and (b) <= 2 x (kWh of the AMP-absent readings in "
         "2012-2022, from T2a) + 1 kWh", "preorigin, full"),
    unit("C0.v1.T3_vs_p1_truth", "C0:product_vs_p1_truth",
         "C0 county / in-grid class x month sums vs P1 census_check/v2 truth",
         "census_clean_county_class_month:2007-01..2026-06",
         "C0 county-scope and in-grid class x month sums (Changhua, Yunlin; both variants; "
         "months <= 2026-06 for full, <= 2022-12 for pre-origin) against P1's independent "
         "census truth (census_check/v2 census_truth_monthly.parquet, series 'clean'), opened "
         "as after_the_fact_analysis because that product's validator use is refused (its "
         "gate failed on C9). A cross-implementation consistency check; reported, not gating.",
         "max |delta| in GWh over county x variant x scope x class x month",
         "<= 1e-6 GWh (reported pass/fail; not part of any gate)", "none (reported)"),
    unit("C0.v1.T4_tpc_coverage", "C0:product_vs_tpc_agri",
         "census coverage of TPC d007012 farm-sector sales", "tpc_agri_vs_census:2016-01..2026-06",
         "Descriptive validation against TPC d007012 農林漁牧 (county_electricity/"
         "tpc_observed/v1 opened as after_the_fact_analysis; its validator use is refused while "
         "its gate fails on G3): annual (calendar-year) coverage census/TPC for both counties, "
         "county and in-grid, all classes and domestic excluded, full variant; month-of-year "
         "coverage profile 2016-2022; Changhua/Yunlin ratios of census and of TPC; monthly "
         "Pearson r of census county total vs TPC (raw and with each window's month-of-year "
         "means removed) for 2016-01..2022-12 and 2016-01..2026-06 at lags -1, 0, +1 (census "
         "month t against TPC month t+lag).",
         "descriptive; plus reproduction of P1 census_check/v2 C8 cov_clean_all (annual, "
         "2016-2025)", "none (descriptive); the C8 reproduction is reported pass/fail at 1e-6 "
         "relative", "none (descriptive)"),
    unit("C0.v1.T5_preorigin_equals_full", "C0:product_variant_consistency",
         "pre-origin vs full products on months <= 2022-12", "c0_products:2007-01..2022-12",
         "Per-pump values (all three counties), kept sets, and per-cell arrays (value, "
         "value_by_county) of the two variants are equal on every month <= 2022-12.",
         "max |delta| kWh; NaN patterns; uid sets", "max |delta| <= 1e-9 kWh, identical NaN "
         "patterns and identical kept uid sets", "preorigin, full"),
    unit("C0.v1.T6_masks_and_internal_consistency", "C0:product_masks_internal",
         "masks and internal consistency of the delivered products",
         "c0_products:2007-01..2026-07",
         "(i) every method-1 (pre_install) pump-month has value 0, every method-1/2 month "
         "precedes the meter's install month, every method-2 month has a positive L0 reading, "
         "and no value carries flags 2-6 (nothing imputed); (ii) full: every post-install "
         "2026-07 pump-month on a phase-A meter (M2, or detected A when the code is unknown) is "
         "NaN, flag 8, method 4, and every class-cell or total containing one is NaN with flag "
         "8; (iii) value NaN <=> flag 8; pump flags in {0, 8, 9}, cell flags in {0, 8, 9}, "
         "even-split flags in {0, 1, 8, 9}; common.flags.validate_grid passes; (iv) per-pump "
         "values equal HP share x the meter's L0 reading wherever the method is not 1; "
         "per-cell arrays equal the sums of the per-pump arrays; the county axis sums to the "
         "total; totals equal class sums; the domestic-excluded total equals the non-domestic "
         "class sum; long-table rows equal the array sums; (v) the kept set equals 'meter duty "
         "<= 1.0 or undefined' recomputed independently from L0 (first pump's series, mean of "
         "non-NaN 2012-2022 months / (HP sum x 0.746 x 730)).",
         "violation counts; max |delta| for (iv)",
         "0 violations of (i)-(iii) and (v); (iv) every element within 1e-6 kWh + 1e-9 "
         "relative", "preorigin (pre-origin part), full (all)"),
    unit("C0.v1.T7_county_coverage", "C0:product_county_set", "counties in the product",
         "census_attributes",
         "Every county with pumps in active cells is in the product: Yunlin and Changhua kept "
         "in-grid pumps > 0; Chiayi census pumps in active cells = 320 (main-session profile, "
         "re-seen at design time); Pingtung census pumps in active cells = 0 (pumping."
         "_cell_lookup on its 97X坐標/97Y坐標).",
         "counts", "Yunlin and Changhua > 0; Chiayi = 320; Pingtung = 0", "all products"),
    unit("C0.v1.T8_privacy", "C0:product_privacy", "privacy scan of every output",
         "c0_outputs",
         "No output column, array or JSON key is 戶名1, 用電地址2, 郵遞區號, 電號, 電號1 or 電號2, "
         "and no output column holds raw meter numbers (any column with >= 1 % of its values in "
         "the set of raw meter numbers counts as a violation).",
         "violation count", "0", "all products"),
    unit("C0.v1.T9_aggregation_vs_model_function",
         "C0:product_ch_cy_vs_model_aggregation_2012_2022",
         "Changhua and Chiayi county axes vs the model's own aggregation",
         f"changhua_chiayi_census_new_l0:{W}",
         "For Changhua and for Chiayi (in-grid pumps), the pre-origin value_by_county arrays "
         "equal scenario.energy_by_class applied to the model's clean_census output (cap 1.0, "
         "t0 2012-01-01, t1 2023-01-01) on the new L0, 2012-01..2022-12.",
         "max |delta| over class x cell x month", TOL, "preorigin, full"),
]

GATES = {
    "pump_electricity/preorigin": ["C0.v1.T1a_model_path_identity", "C0.v1.T1b_c0_pipeline_on_amp",
                                   "C0.v1.T2a_new_l0_common_rows",
                                   "C0.v1.T2b_product_yunlin_vs_model",
                                   "C0.v1.T5_preorigin_equals_full",
                                   "C0.v1.T6_masks_and_internal_consistency (pre-origin part)",
                                   "C0.v1.T7_county_coverage", "C0.v1.T8_privacy",
                                   "C0.v1.T9_aggregation_vs_model_function"],
    "pump_electricity/full": ["all tests of the pre-origin gate",
                              "C0.v1.T6_masks_and_internal_consistency (full part)"],
    "pump_electricity/census_L1": ["C0.v1.T7_county_coverage", "C0.v1.T8_privacy",
                                   "uid unique", "C0.v1.T6 (v) on the L1 'kept' column"],
}


def product_files() -> list[str]:
    prods = lib.load_json(lib.WORK / "products.json")
    out = []
    for d in prods.values():
        out += [lib.rel(p) for p in iter_files(lib.REPO / d)]
    return sorted(out)


def doc() -> dict:
    inputs = ([lib.rel(lib.PARQ / f"{c}_{k}.parquet") for c in ("yunlin", "changhua", "chiayi",
                                                                "pingtung")
               for k in ("census", "kwh_long")]
              + [lib.rel(lib.SOURCE_JSON), lib.rel(lib.AMP_CENSUS), lib.rel(lib.AMP_KWH),
                 lib.rel(lib.MODEL_INPUTS_PKL), lib.rel(lib.GRID_NPZ),
                 lib.rel(lib.P1_CENSUS_CHECK / "census_truth_monthly.parquet"),
                 lib.rel(lib.P1_CENSUS_CHECK / "evaluation.json"),
                 lib.rel(lib.P1_CENSUS_CHECK / "MANIFEST.json"),
                 lib.rel(lib.P1_TPC_OBSERVED / "tpc_county_sector_monthly_observed.parquet"),
                 lib.rel(lib.P1_TPC_OBSERVED / "MANIFEST.json")]
              + [lib.rel(lib.CODE_DIR / f) for f in ("lib.py", "build.py", "assemble.py")]
              + product_files())
    return {
        "id": lib.PREREG_ID,
        "wp": lib.WP,
        "title": "C0 cleaned pump-electricity census product (Yunlin, Changhua, in-grid "
                 "Chiayi): construction checks, reproduction of the model's Yunlin forcing "
                 "2012-2022, TPC coverage (descriptive)",
        "kind": "test",
        "plan_section": "workflow task text WP C0 (2026-10-12 afternoon update); plan 2.3-2.6, "
                        "4.1-4.6; P2b/P2c conventions for the July mask and terminated meters",
        "tests": TESTS,
        "zones": NA("no zones; the product is a cleaned observation table on the twin grid"),
        "priors": NA("no priors"),
        "burn_in": NA("no model run"),
        "folds": NA("no folds; deterministic construction checks"),
        "fresh_units": sorted({u for t in TESTS for u in t["fresh_units"]}),
        "thresholds": {t["test_id"]: t["threshold"] for t in TESTS},
        "gate_scores": NA("conjunctive pass/fail gates, no weights: " + "; ".join(
            f"{k}: {', '.join(v)}" for k, v in GATES.items()) + ". T3 and T4 are reported "
            "only (they rest on P1 products whose validator use the guard refuses)."),
        "gates": GATES,
        "primary_model": "the C0 products built by dataprep_staging/c0_pump_census/build.py and "
                         "assemble.py with lib.py (sha256 recorded under inputs, together with "
                         "every product file): clean_census cap 1.0 with the duty statistic on "
                         "2012-01..2022-12 over the L0 readings as delivered; pre-install months "
                         "structural zeros (flag 0, method pre_install; meter install = earliest "
                         "valid install date of the meter's pumps); post-install NaN readings "
                         "missing (flag 8, never filled; counted as zero in aggregates as the "
                         "model does; an aggregate is flagged 9 when > 10 % of its installed "
                         "kept HP is missing); 2026-07: phase-A (M2, or detected A when the code "
                         "is unknown) meters NaN flag 8 (incomplete Jul-Aug pair), M0/M9 "
                         "readings flag 9; terminated meter = trailing run >= 12 months of "
                         "non-positive or missing readings ending at the variant's last "
                         "complete month (2022-12 / 2026-06); billing phase from the billing "
                         "code (M2 = A, M1 = B, M0/M9 monthly) plus the billing pilot's "
                         "identical-pair detection on 2012-2022 (pre-origin) or 2012-01..2026-06 "
                         "(full); full also ships an even-split alternative (flag 1) where a "
                         "bimonthly bill was posted wholly to the second month; domestic-"
                         "excluded totals as the sensitivity variant.",
        "inputs": inputs,
        "interpretation": {
            "gate": "T1a-T2a and T9 are code-identity checks on identical inputs; T2b is the "
                    "task's 'Yunlin part reproduces the model's cleaned forcing to within "
                    "rounding' with the tolerance fixed here; T5-T8 are construction checks. "
                    "A failure is reported with its number and the gate fails; thresholds are "
                    "not changed after the result. None of these tests uses a 2023+ heads, "
                    "ring, pile, GNSS or InSAR observation (U7); the full-variant parts of T6 "
                    "check masks on 2023-26 electricity, which is forcing, not an observation.",
            "T3_T4": "Reported only. T3 compares with P1's independent implementation; T4's "
                     "coverage ratios are descriptive: TPC 農林漁牧 includes non-pump load and "
                     "its billing month is undocumented; both P1 products are opened as "
                     "after_the_fact_analysis.",
            "not_fresh": "These are deterministic construction and reproduction checks, not "
                         "out-of-sample tests. Several reference facts were known at design "
                         "time (disclosures); the fresh units are the comparisons of the "
                         "delivered products, which did not exist before this registration.",
            "disclosures_before_registration": [
                "Look log [C0-seed]/[C0-look] records (10): L0 profiles of all three "
                "counties (NaN structure, billing groups, pair structure), the model-forcing "
                "reference summary (dates, grid, the 6 class fan totals 2012-2022 of the "
                "cached E_by_class, read at 06:35Z by the first attempt), the kept-set "
                "sensitivity to pre-install coding (Yunlin 2 meters / +0.03 %, Changhua 64 "
                "meters / +0.35 % of 2012-2022 kept kWh), AMP_V2 vs new Yunlin attributes "
                "(69 PURPOSE None vs 'NULL', 84 coordinates NaN in both: 0 class and 0 cell "
                "changes), Chiayi 320 and Pingtung 0 pumps in active cells, and two Chiayi "
                "smoke builds (the first exposed the method-5 override bug, fixed).",
                "Decision taken knowing the coding sensitivity: the duty statistic uses the L0 "
                "readings as delivered (model-identical for Yunlin); the alternative is "
                "reported, not used.",
                "P1 round 2/3 (census_check v1/v2): the new Yunlin sheet has 222 pump-month "
                "readings (137 non-zero, 8.3 MWh, one irrigation pump) absent from AMP_V2; the "
                "model cleaning on the new L0 matches P1 v1's Yunlin series to 1.7e-5 "
                "relative; preorigin == full on months <= 2022-12; annual cov_clean_all (C8) "
                "Changhua 0.39-0.45, Yunlin 0.43-0.55; like-for-like domestic ratio 0.795.",
                "Main session profile (inspection/profile*.json): in-grid Changhua about 188 "
                "GWh/yr 2012-22, Yunlin about 188; July 2026 M2 group missing.",
                "Design-time L0 facts: Changhua 758k of 818k NaN pump-months precede the "
                "install date; post-install gaps peak 2016-2020; 2026-07 M2 zero fraction "
                "0.993-0.998; bimonthly pairs posted wholly to the second month in 12-33 % of "
                "pairs from 2025-07 on; before 2025-07 median 0 and at most 2 %, except the M1 "
                "pair 2021-12/2022-01 (8-9 %).",
                "Products built before this registration (pinned by hash here); no comparison "
                "with any reference (model forcing, P1 truth, TPC) was computed before it.",
            ],
        },
    }


def main() -> int:
    path, sha = prereg2.write_prereg(doc())
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
