"""WP C0 repair round: pre-registration ``C0.census_product.v3`` (A1 tool ``harness.prereg3``),
written after the v3 products were assembled (their files are hashed here) and BEFORE any check
of them is computed. Copy of ``repair_prereg.py`` (v2) with the v3 rule, the corrected R4 sd
recomputation and T5 cell clause, new test and unit ids, and the v2 outcomes disclosed.

Usage: python -I repair_prereg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common.hashing import iter_files  # noqa: E402
from harness import prereg3  # noqa: E402

PREREG_ID = "C0.census_product.v3"
NONE = ("C0 electricity-forcing construction object; contains no registered well, leveling "
        "benchmark or model cell of the validation-unit registries")


def na(reason: str):
    from harness import prereg2
    return prereg2.not_applicable(reason)


def unit(test_id: str, uid: str, desc: str, scope: str, what: str, metric: str,
         threshold: str, gating: str) -> dict:
    return {"test_id": test_id, "data_scope": scope, "fresh_units": [uid],
            "external_units": {uid: {"description": desc, "none": NONE}}, "what": what,
            "metric": metric, "threshold": threshold, "gating": gating}


TOL = "|delta| <= 1e-6 kWh + 1e-9 x |reference| per element"
TESTS = [
    unit("C0.v3.R1_v3_vs_v1_decomposition", "C0r3:v3_vs_v1_decomposition",
         "the v3 per-pump arrays against v1, and the fill against an independent recomputation",
         "c0_products_v1_v3:2007-01..2026-07",
         "For each county and variant: (a) the v3 kept uid set equals the set recomputed "
         "independently from L0 (lib functions only: Dec-Jan-paired rule on the identical-pair "
         "detection of pairs inside 2012-01..2022-11; duty over 2012-2022 without the paired "
         "meters' Dec-2022 value; cap 1.0; Chiayi in-grid); (b) for pumps in both v1 and v3, "
         "values (and NaN patterns) and flags are identical on every pump-month except the "
         "2022-12 entries of Dec-Jan-paired pumps in the pre-origin variant; (c) pre-origin "
         "2022-12 of paired pumps: flag 4 exactly where the meter's Oct and Nov 2022 months, "
         "coded as the product codes them (pre-install NaN or 0 -> 0, positive readings before "
         "the install date kept, installed months as read), are both finite, except a meter "
         "not yet installed in 2022-12 with base 0; value = HP share x rho_bar x base and "
         "sd = HP share x base x sqrt(s^2 + sigma^2) with rho_bar, s, sigma from a second "
         "implementation of the registered ratio fit (2012-2021) that must equal "
         "fill_parameters.json; otherwise flag 8 or a pre-install structural zero; no flag 4 "
         "outside 2022-12; (d) full variant: no flag 4; (e) L1 dec_jan_paired and "
         "meter_duty_2012_2022 equal the recomputation. Method labels may differ (reported). "
         "Reported: pump-months, months and meters where v3 differs from v2 (pre-origin).",
         "counts of violations per item", f"0 violations; fill values and sd {TOL}; fill "
         "parameters within 1e-9; L1 duty within 1e-12", "preorigin (pre part), full (full part)"),
    unit("C0.v3.R2_no_postorigin_dependency", "C0r3:perturbation_invariance",
         "pre-origin v3 rebuilt from a perturbed L0", "tpc_census_kwh_perturbed:2007-01..2026-07",
         "Rebuild the pre-origin variant and the L1 rows of each county with repair_build from "
         "an L0 whose every month >= 2023-01 is overwritten (finite x -> 3x + 100, NaN -> 50) "
         "and whose 2022-12 readings of every pump on a Dec-Jan-paired meter are multiplied by "
         "U(0, 5) (seed 20221231; NaN -> 77). Every per-pump array, the meters table, the "
         "county's value/flag/sd cell arrays and the L1 columns (uid, kept, "
         "meter_duty_2012_2022, hp_share, dec_jan_paired, phase_detected_2012_2022, "
         "meter_hp_sum) must equal the delivered v3. Positive control: v1's pre-origin builder "
         "(build.process) on the same perturbed L0 must differ from the delivered v1 for "
         "Yunlin and Changhua (shows the test detects the v1 leak).",
         "violations (arrays compared at 1e-9 abs, NaN-equal; tables at 1e-9 / 1e-12)",
         "0 violations and the control detects the leak in Yunlin and Changhua",
         "preorigin; census_L1 (L1 part)"),
    unit("C0.v3.R3_yunlin_vs_model_forcing", "C0r3:v3_yunlin_vs_cached_forcing",
         "v3 Yunlin county axis vs the calibrated model's cached forcing",
         "model_forcing_E_by_class:2012-01..2022-12",
         "Task (d) restated for v3: v3 (and v2, v1 for reference) Yunlin value_by_county against "
         "E_by_class of inputs.local.pkl: max monthly relative fan-total delta and sum |delta| "
         "on 2012-01..2022-11, the 2022-12 relative delta, max element. v3 departs from the "
         "model's forcing by design (no post-origin information); the departure is accounted "
         "for exactly by R1 (kept-set flips, Dec-2022 fill) plus v1's T2b (one AMP-absent "
         "pump). Reported, not gating.", "as listed", "none (reported)", "none"),
    unit("C0.v3.R4_masks_and_internal_consistency", "C0r3:v3_masks_internal",
         "v3 masks, flags, sd and aggregation identities", "c0_products_v3:2007-01..2026-07",
         "Both variants: pre-install zeros never imputed (method 1 value 0, methods 1/2 only "
         "before the install date); NaN <=> flag 8 at every level; flag = flag of the method; "
         "pump flags in {0,4,8} (pre-origin) / {0,8,9} (full); aggregate flags in {0,4,9} / "
         "{0,8,9}; sd finite <=> flag 4 at pump level, sd finite <=> n_fill > 0 for "
         "aggregates and long-table rows; flag 4 only in 2022-12 and only on Dec-Jan-paired "
         "pumps; no paired 2022-12 value left as an observation; value = HP share x L0 meter "
         "reading wherever it is an observation; cells = sum of pumps; totals = sum of "
         "classes; domestic-excluded = sum of the other classes; long table = arrays (value, "
         "n_fill); sd of class x cell and of the totals recomputed from the pumps and "
         "fill_parameters.json with rho_bar and s of the meter's class (first pump), the class "
         "the fill uses; missing_expected_kwh recomputed (Yunlin/Changhua cells), >= 0, "
         "0 where no HP is missing; validate_grid / validate_long_table clean; full: 2026-07 "
         "phase-A masks as v1 T6 (ii); kept sets of both variants and the L1 'kept' column "
         "equal the independent recomputation.", "violations",
         f"0 violations; numeric identities {TOL}; sd within 1e-6 kWh",
         "preorigin (pre part), full (full part), census_L1 (kept part)"),
    unit("C0.v3.R5_privacy", "C0r3:v3_privacy", "privacy scan of every v3 output", "*",
         "No forbidden column (戶名1, 用電地址2, 郵遞區號, 電號, 電號1, 電號2) in any parquet, npz "
         "or json key, nor in text files; no 1-D column/array whose values are >= 1 % raw "
         "meter numbers of the four census sheets.", "violations", "0",
         "preorigin, full, census_L1"),
    unit("C0.v3.R6_county_coverage", "C0r3:v3_county_set", "counties in the v3 products", "*",
         "Kept pumps in active cells: Yunlin > 0, Changhua > 0; census pumps in active cells: "
         "Chiayi 320, Pingtung 0; L1 uid unique.", "counts", "as stated",
         "preorigin, full, census_L1"),
    unit("C0.v3.T5_variant_consistency", "C0r3:v3_variant_consistency",
         "pre-origin v3 vs full v3 up to the origin", "c0_products_v3:2007-01..2022-12",
         "Same uid arrays; per-pump values and flags identical on 2007-01..2022-11; at 2022-12 "
         "identical for pumps not on Dec-Jan-paired meters, and flag 4 only on paired ones; "
         "cell value_by_county and the two totals identical on 2007-01..2022-11; class x cell "
         "values at 2022-12 identical in every class x cell entry that holds no Dec-Jan-paired "
         "pump.",
         "max |delta| and mismatch counts", "max |delta| <= 1e-9 kWh, 0 mismatches",
         "preorigin, full"),
    unit("C0.v3.R7_fill_backtest_loyo", "C0r3:fill_backtest_loyo_2012_2021",
         "leave-one-year-out back-test of the Dec fill and its sd on Dec y / Jan y+1, "
         "y = 2012..2021", "tpc_census_kwh:2012-10..2022-01",
         "For every year y in 2012..2021: fit the ratio (second implementation, as R1) on the "
         "other nine years, fill December y of the Dec-Jan-paired kept meters from their Oct-Nov "
         "y base, share to pumps, aggregate in grid to class x cell and to county totals, and "
         "compare with the even-split Dec y / Jan y+1 halves. sd = sqrt((sum s b)^2 + "
         "sum_meters (sigma sum b)^2) with the left-out fit. Reported: coverage of the nominal "
         "90 % interval (|z| <= 1.645) over class x cell groups with fill > 0 (pooled) and over "
         "the 20 Yunlin/Changhua county-year totals, median |relative error| and mean relative "
         "bias. Determines the sd label only (plan 4.7).", "coverage, errors",
         "sd labelled 'calibrated' iff class x cell coverage in [0.80, 0.97] and Yunlin+Changhua "
         "county coverage >= 0.75; otherwise 'uncalibrated' (not gating)", "none (label)"),
    unit("C0.v3.R8_after_the_fact_dec2022", "C0r3:dec2022_fill_vs_jan2023_half",
         "after-the-fact comparison with the post-origin Dec-2022/Jan-2023 halves",
         "tpc_census_kwh:2022-10..2023-01",
         "Uses January-2023 L0 and the v1 values (which contain it); after-the-fact only, never "
         "fed back: in-grid fill vs the v1 value of the same pumps (relative difference), the "
         "share of v1's in-grid 2022-12 that paired meters carried, and the rule's capture of "
         "meters whose Dec-2022 and Jan-2023 readings are equal halves (sensitivity, precision, "
         "untreated equal-half meters and their Dec-2022 kWh share).", "as listed",
         "none (reported)", "none"),
    unit("C0.v3.R9_census_snapshot_sensitivity", "C0r3:census_snapshot_sensitivity",
         "sensitivity run of the census-attribute exemption",
         "tpc_census_kwh:2012-01..2022-12",
         "Counts of census pumps with a valid install date after 2022-12 (and after 2016-12), "
         "their meters and shared meters; pre-origin rebuild of each county without those pumps "
         "(attributes as they could have stood at the origin, as far as install dates tell) "
         "and the change of in-grid energy 2012-2022 (total, by class, max monthly relative, "
         "sum |delta| over class x cell x month).", "as listed", "none (reported)", "none"),
    unit("C0.v3.R10_descriptive", "C0r3:v3_descriptive", "descriptive statistics of v3",
         "c0_products_v3:2007-01..2026-07",
         "In-grid GWh/yr 2012-2022 by county, domestic share, missing_expected share, flag "
         "counts, Dec-2022 in-grid energy and filled cells, fill parameters, build counts.",
         "as listed", "none (reported)", "none"),
]
GATES = {
    "pump_electricity/preorigin": ["C0.v3.R1 (pre-origin part)", "C0.v3.R2",
                                   "C0.v3.R4 (pre-origin part)", "C0.v3.R5", "C0.v3.R6",
                                   "C0.v3.T5"],
    "pump_electricity/full": ["C0.v3.R1 (full part)", "C0.v3.R4 (full part)", "C0.v3.R5",
                              "C0.v3.R6", "C0.v3.T5"],
    "pump_electricity/census_L1": ["C0.v3.R2 (L1 columns)", "C0.v3.R4 (L1 kept column)",
                                   "C0.v3.R5", "C0.v3.R6 (incl. uid unique)"],
}


def v3_files() -> list[str]:
    recs = lib.load_json(lib.EVIDENCE / "repair3" / "products_v3.json")
    out = []
    for d in recs.values():
        out += [lib.rel(p) for p in iter_files(lib.REPO / d)]
    return sorted(out)


def doc() -> dict:
    v1 = lib.PRODUCT_ROOT
    v2 = {k: lib.REPO / v for k, v in
          lib.load_json(lib.EVIDENCE / "repair" / "products_v2.json").items()}
    inputs = ([lib.rel(lib.PARQ / f"{c}_{k}.parquet") for c in ("yunlin", "changhua", "chiayi",
                                                                "pingtung")
               for k in ("census", "kwh_long")]
              + [lib.rel(lib.SOURCE_JSON), lib.rel(lib.AMP_CENSUS), lib.rel(lib.AMP_KWH),
                 lib.rel(lib.MODEL_INPUTS_PKL), lib.rel(lib.GRID_NPZ),
                 lib.rel(lib.EVIDENCE / "checks" / "T2a.json")]
              + [lib.rel(v1 / v / "v1" / f) for v in ("preorigin", "full")
                 for f in ("pumps_yunlin.npz", "pumps_changhua.npz", "pumps_chiayi_ingrid.npz",
                           "cells_by_class.npz", "MANIFEST.json")]
              + [lib.rel(v1 / "census_L1" / "v1" / f) for f in ("census_pumps.parquet",
                                                               "MANIFEST.json")]
              + [lib.rel(v2["preorigin"] / f) for f in ("pumps_yunlin.npz", "pumps_changhua.npz",
                                                        "pumps_chiayi_ingrid.npz",
                                                        "cells_by_class.npz")]
              + [lib.rel(lib.CODE_DIR / f) for f in ("lib.py", "build.py", "checks.py",
                                                     "repair_lib.py", "repair_build.py",
                                                     "repair_assemble.py", "repair3_lib.py",
                                                     "repair3_build.py", "repair3_assemble.py")]
              + [lib.rel(lib.EVIDENCE / "repair" / "design_look.json")]
              + v3_files())
    return {
        "id": PREREG_ID,
        "wp": lib.WP,
        "title": "C0 repair round, v3: pump-electricity without post-origin information in the "
                 "pre-origin variant (v2 design plus the v3 fill base coding), construction "
                 "checks with the corrected R4 sd recomputation and T5 cell clause, perturbation "
                 "invariance, fill back-test",
        "kind": "test",
        "plan_section": "repair task of WP C0 (verifier finding: U3/U4 breach of preorigin/v1); "
                        "plan 2.3 (exemptions), 2.4 (flags), 2.5 (U1-U7), 4.2, 4.7, 4.8",
        "tests": TESTS,
        "zones": na("no zones; electricity forcing table on the twin grid"),
        "priors": na("no priors"),
        "burn_in": na("no model run"),
        "folds": na("R7 is leave-one-year-out over 2012-2021; the other tests are "
                    "deterministic construction checks"),
        "fresh_units": sorted({u for t in TESTS for u in t["fresh_units"]}),
        "thresholds": {t["test_id"]: t["threshold"] for t in TESTS},
        "gate_scores": na("conjunctive pass/fail gates, no weights: " + "; ".join(
            f"{k}: {', '.join(v)}" for k, v in GATES.items()) + ". R3, R7, R8, R9, R10 are "
            "reported only (R7 sets the sd label)."),
        "gates": GATES,
        "primary_model": (
            "the v3 products assembled by repair3_assemble.py (repair_assemble functions) from "
            "repair3_build.py with repair3_lib.py, repair_lib.py and lib.py (hashed under "
            "inputs with every product file): as C0.census_product.v2 (Dec-Jan-paired rule, "
            "pre-origin duty shared by both variants, ratio fill of the paired meters' 2022-12 "
            "with its sd, flags and additions) except the fill base: the Oct and Nov 2022 "
            "months are coded as the product codes them (pre-install NaN or 0 -> structural "
            "zero, positive readings before the install date kept, installed months as read); "
            "a paired kept meter is filled whenever both coded months are finite, except a "
            "meter not yet installed in 2022-12 whose base is 0 (structural zero by date); a "
            "missing installed base month leaves the 2022-12 missing (flag 8)."),
        "inputs": inputs,
        "interpretation": {
            "gate": "R1, R2, R4, R5, R6, T5 are construction, invariance and consistency "
                    "checks; a failure fails the gate with its number and thresholds are not "
                    "changed after the result. R2 is the direct test of the repair. No test "
                    "uses a 2023+ heads, ring, pile, GNSS or InSAR observation (U7). R8 uses "
                    "January-2023 electricity after the fact only. R7 labels the fill sd "
                    "'calibrated' or 'uncalibrated' (plan 4.7); it does not gate.",
            "exemption": "The census attribute table (2026 delivery, data to 2026-07) cannot be "
                         "had as of 2022-12; the v3 manifests declare it as an exemption "
                         "(data_end 2026-07, non_exempt_fit_window_end 2022-12) for the "
                         "exemptable pre-origin uses; R9 is its sensitivity run.",
            "not_fresh": "Construction and reproduction checks, not out-of-sample tests. The v1 "
                         "results are known, and so are the v2 outcomes listed below; the v3 "
                         "rule and the corrected R4/T5 checks were decided after the v2 R4 and "
                         "T5 failures (look [C0-repair-v3-decision]). The R7 years 2012-2021 "
                         "are the fit years of the fill (leave-one-year-out).",
            "disclosures_before_registration": [
                "Verifier findings (seed look [C0-repair-seed]): 99.7 % of M1 meters have "
                "Dec-2022 == Jan-2023; phase-B in-grid 2022-12 energy 3.78 + 4.55 + 0.018 of "
                "28.67 GWh; M1 pair 2021-12/2022-01 equal in 3.5 % / 6.2 % of meter pairs; "
                "billing-code agreement 0.959-0.978 / 0.909-0.912 (Chiayi) / 0.926-0.938.",
                "Design look [C0-repair-design] (pre-origin counts): code x detection "
                "cross-tab; kept-set flips Yunlin +2/-1, Changhua +3/-9, Chiayi 0 (detection "
                "rule); paired kept meters without an Oct-Nov 2022 base 11 / 23 / 45.",
                "Census install months after 2016-12: Yunlin 85, Changhua 79 pumps (after "
                "2022-10: 45 / 35; latest 2026-06 / 2025-12).",
                "v2 (C0.census_product.v2, sha256 33067d13...) registered outcomes: R1 pass "
                "(second scoring after a check-scope fix: the first R1 run fitted Chiayi's "
                "ratio on the in-grid L1 subset instead of the county; kept under "
                "superseded/), R4 FAIL (1,346 violations, all in the sd recomputation, a check "
                "defect: pump class instead of meter class; the meter-class recomputation "
                "matches to 9e-13, 917 filled pumps sit on meters of another class, "
                "[C0-repair-diag-1]), T5 FAIL (1 class x cell entry at 2022-12, 3.5 kWh: a "
                "Changhua paired meter with census install date 2025-09 read before it, set to "
                "a structural zero by date with no fill, [C0-repair-diag-2]), R5 pass, R6 "
                "pass; R2 crashed once in its Chiayi comparison (a check bug, fixed) and was "
                "rerun with R9; R3, R7, R8 and R10 of v2 were computed but not read before this "
                "registration.",
                "v3 build logs (kept pumps/meters per county and variant, identical to v2) "
                "were printed; no fill value, ratio or comparison of v3 was read.",
            ],
        },
    }


def main() -> int:
    path, sha = prereg3.write_prereg(doc())
    print(path, sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
