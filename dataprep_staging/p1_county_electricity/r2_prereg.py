"""WP P1 round 2: pre-registrations written with the A1 tool BEFORE the round-2 tests run.

Round 2 exists because the plan was revised on 2026-10-12 (afternoon, plan header): the
Taiwan Power pump census for Changhua (and Yunlin to 2026-07) was obtained, so P1 becomes a
validator of census coverage and its v1 inference can, for the first time, be scored against
measured Changhua pump electricity. The v1 products (``county_electricity/{preorigin,full}/v1``)
were frozen at 2026-10-12T02:43Z, three hours before the census reached the repository
(SOURCE.json 05:44Z), so no v1 number can have been chosen with it.

Three documents:

* ``P1r2.census_confirmations`` (kind ``declaration``): comparisons whose outcome the plan
  revision or the main session's census profile had already disclosed (a against the census
  ratio, the in-grid fraction and anchor, the monthly shape, the ensemble ranking, the
  coverage ratios, the Yunlin consistency check). Definitions and readings are fixed here,
  but they are after-the-fact confirmations, not fresh validation.
* ``P1r2.census_tests`` (kind ``test``): the two out-of-sample tests of the v1 extension past
  the old census end (Yunlin YL_clean_hat and Changhua a x YL_clean_hat, 2025-07..2026-06),
  with pass/fail thresholds.
* ``P1r2.tpc_observed_gate`` (kind ``test``): the integrity gate of the new observed product
  ``county_electricity/tpc_observed/v1`` (the validator role of the revised plan).

Usage: python -I r2_prereg.py            (writes once; the tool refuses to overwrite)
       python -I r2_prereg.py --dry      (validate only)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from harness import prereg as pr  # noqa: E402

REPO = HERE.parents[1]
CE = "results/dataprep/county_electricity"
NEW = "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026"
NA = pr.not_applicable

V1 = {v: json.loads((REPO / CE / v / "v1/validation.json").read_text(encoding="utf-8"))
      for v in ("preorigin", "full")}
DEC = {v: json.loads((REPO / CE / v / "v1/decomposition.json").read_text(encoding="utf-8"))
       for v in ("preorigin", "full")}


def remedy_interval(v: str) -> list[float]:
    s = DEC[v]["source_separation"]
    k = V1[v]["tests"]["T8_SYN_coverage"][v]["remedy_reported_only"]["k_halfwidth_factor"]
    a, lo, hi = s["a"], s["a_lo"], s["a_hi"]
    return [a - k * (a - lo), a + k * (hi - a), k]


A = {v: DEC[v]["source_separation"] for v in ("preorigin", "full")}
REM = {v: remedy_interval(v) for v in ("preorigin", "full")}
PHI = DEC["preorigin"]["phi_in"]
MU = DEC["preorigin"]["mu_prior"]

TRUTH_DEF = {
    "census": f"{NEW}/parquet/<county>_census.parquet + <county>_kwh_long.parquet (columns "
              "selected explicitly; raw/ and extracted/ never read; 戶名1, 用電地址2 and the "
              "postal code are absent from parquet/)",
    "pump_id": "sid = WELL_NO (stripped, trailing '.0' removed; unique in both counties)",
    "cleaning": "hydrophysics.twin.pumping.clean_census exactly as P1 v1 census_monthly.py: "
                "cap_duty 1.0 with the duty statistic on 2012-01..2022-12 selects the kept "
                "pumps; the kept pumps' full record is then de-duplicated and HP-split with "
                "cap_duty None",
    "variants": "preorigin: the kWh table is truncated to <= 2022-12 before cleaning; full: "
                "every month to 2026-07 (2026-07 is a partial month per SOURCE.json and is "
                "never scored)",
    "classes": "scenario.purpose_class; 'all' = every class kept after cleaning (the "
               "definition of P1's YL_clean); irrigation = class 'irrigation'",
    "in_grid": "pumping._cell_lookup on the kept pumps with the grid of "
               "results/twin_forward/datum_gate_innov.npz (pumps outside the active mask "
               "dropped, never snapped)",
    "units": "GWh/month, model units (cleaned-census kWh)",
}

INPUTS_COMMON = [
    f"{CE}/preorigin/v1/MANIFEST.json", f"{CE}/full/v1/MANIFEST.json",
    f"{CE}/preorigin/v1/validation.json", f"{CE}/full/v1/validation.json",
    f"{CE}/preorigin/v1/decomposition.json", f"{CE}/full/v1/decomposition.json",
    f"{NEW}/SOURCE.json",
]
INPUTS_CENSUS = [f"{NEW}/parquet/{c}_{k}.parquet" for c in ("changhua", "yunlin")
                 for k in ("census", "kwh_long")]

EXPOSURE = [
    "Plan revision note (docs/DATA_REMEDIATION_PLAN_2026-10.md header, file time "
    "2026-10-12T05:45Z): in-grid Changhua about 188 GWh/yr 2012-22, equal to Yunlin's about "
    "188; 97 % of Changhua pumps in active cells; irrigation seasonal CV 0.14 (Changhua) vs "
    "0.234 (Yunlin), alpha about 0.6; 'a about 0.94-0.96 x Yunlin is confirmed by the "
    "census'; Changhua domestic class large (about 49 GWh/yr kept, summer peak).",
    f"{NEW}/inspection/profile2.json (main session, 05:44Z), read by this WP before this "
    "pre-registration: Changhua class month-of-year profiles 2012-22 and seasonal CVs, and "
    "Changhua county cleaned totals for 2025-06..2026-07 (e.g. 2025-06 18.37, 2025-07 18.27, "
    "2026-06 19.13, 2026-07 14.16 GWh). The Yunlin part of that file (beyond its NaN "
    "fractions up to 2016) and inspection/cleaned_monthly_by_class.csv were not read.",
    "No P1 v1 number (a, b, the backcast, phi_in, the extension, the sd) depended on the "
    "census: the v1 manifests were written 2026-10-12T02:43Z and the census arrived 05:44Z.",
]

# ----------------------------------------------------------------------------- declaration
DECLARATION = {
    "id": "P1r2.census_confirmations",
    "wp": "P1",
    "title": "P1 round 2: after-the-fact confirmations of the frozen v1 inference against the "
             "Changhua and Yunlin pump census",
    "kind": "declaration",
    "tests": NA("declaration: every comparison below was partly disclosed before this "
                "pre-registration (prior_exposure), so none is claimed as fresh validation"),
    "zones": NA("county and in-grid scopes only"),
    "priors": NA("no prior is fitted; v1 numbers are frozen"),
    "burn_in": NA("no model run"),
    "folds": NA("no folds; fixed windows below"),
    "fresh_units": NA("units exposed by the plan revision and the census profile"),
    "gate_scores": NA("reported only; the v1 gates are fixed and are not re-decided"),
    "primary_model": "county_electricity/preorigin/v1 (a_pre, phi_in, in-grid anchor) and "
                     "county_electricity/full/v1 (a_full, estimator ensemble), as frozen",
    "inputs": INPUTS_COMMON + INPUTS_CENSUS,
    "truth_definition": TRUTH_DEF,
    "prior_exposure": EXPOSURE,
    "thresholds": {
        "C1_a_vs_census": {
            "truth": "a*_pre = sum CH_clean_all / sum YL_clean_all over 2016-01..2022-12 "
                     "(preorigin cleaning); a*_full = same over 2016-01..2025-06 (full "
                     "cleaning)",
            "reading": "'confirmed' if a* lies in the v1 90 % year-block interval; 'confirmed "
                       "only by the reported remedy interval' if it lies in the k-scaled "
                       "interval but not the raw one; else 'not confirmed'",
            "v1_preorigin": {"a": A["preorigin"]["a"], "raw90": [A["preorigin"]["a_lo"],
                             A["preorigin"]["a_hi"]], "remedy90": REM["preorigin"][:2],
                             "k": REM["preorigin"][2]},
            "v1_full": {"a": A["full"]["a"], "raw90": [A["full"]["a_lo"], A["full"]["a_hi"]],
                        "remedy90": REM["full"][:2], "k": REM["full"][2]},
            "secondary_reported": ["irrigation-only: a_irr (v1) vs sum CH_clean_irr / sum "
                                   "YL_clean_irr on the same windows",
                                   "Changhua domestic class excluded (sensitivity)",
                                   "2012-01..2022-12 window"],
        },
        "C2_misspecification_link": "report a*_pre - a_pre beside v1 T9 bias M1 (seasonal CV x "
                                    "0.5: a_hat - a_eff = -0.095) and the census seasonal-CV "
                                    "ratio CH/YL (all classes and irrigation, 2012-22). "
                                    "Descriptive.",
        "C4_phi_in": {
            "truth": "phi*_in = sum CH_clean_all in active cells / sum CH_clean_all county, "
                     "2012-01..2022-12, preorigin cleaning",
            "reading": "inside the stated band iff |ln(phi*/phi_in)| <= 1.645 x sigma_ln_phi",
            "phi_in": PHI["phi_in"] if "phi_in" in PHI else PHI,
            "sigma_ln_phi": DEC["preorigin"]["in_grid_anchor"]["sigma_ln_phi"],
            "secondary_reported": ["pump-count fraction", "dedup (no cap) electricity "
                                   "fraction", "irrigation-only fraction"],
        },
        "C5_in_grid_anchor": {
            "truth": "sum CH_clean_all in active cells, 2012-01..2022-12 (preorigin cleaning)",
            "estimate": "sum a_pre x YL_clean x phi_in over the same months (v1 preorigin "
                        "changhua_candidates 'scaled', scope in_grid)",
            "reading": "inside the mu prior's 90 % band iff |ln(estimate/truth)| <= 1.645 x "
                       f"{MU['log_sd']:.4f}; also reported against the plan's 0.5 log-sd",
            "secondary_reported": ["county scope: a_pre x YL_clean vs CH_clean_all county",
                                   "2013-01..2022-12 window"],
        },
        "C8_coverage_validator_role": "annual ratios census / TPC d007012 農林漁牧 per county "
                                      "2016-2025 (2026: Jan-Jun) for dedup (cap None) "
                                      "agricultural classes (irrigation, aquaculture, "
                                      "livestock), dedup all classes and cleaned all classes; "
                                      "census CH/YL vs TPC CH/YL. Descriptive; WP C0 owns the "
                                      "coverage check of its own census product.",
        "C9_yunlin_consistency": "new-L0 Yunlin cleaned series (both variants) equal P1 v1 "
                                 "YL_clean on every month <= 2025-06 to 1e-6 GWh -> "
                                 "'identical' (a data check)",
        "T12_monthly_shape": "for 2016-01..2022-12 and for 2012-01..2015-12: monthly relative "
                             "RMS, correlation and deseasonalised correlation of each v1 "
                             "estimator (preorigin scaled, shape, transfer; irrigation-only "
                             "against CH_clean_irr) with CH_clean_all; ratio of month-of-year "
                             "CVs (estimator / truth). Descriptive (the revision disclosed "
                             "the CV ratio).",
        "T13_ensemble_rank": "rank of the v1 members (preorigin and full) by |ln(sum member / "
                             "sum truth)| over each variant's NNLS window, and the z of the "
                             "truth inside the mu prior (log-sd 0.549 around a_pre). "
                             "Descriptive.",
    },
    "interpretation": "These numbers record how the frozen v1 inference compares with the "
                      "measured census. They cannot re-open the v1 gates (T1_BV1 and T8 "
                      "failed and stay failed) and they select nothing: P1 v1 is superseded "
                      "as a Changhua forcing source by WP C0 under the revised plan. A "
                      "'confirmed' C1 means the TPC source separation recovered Changhua's "
                      "pump-electricity level from public aggregates; a C4/C5 outside the "
                      "band means the in-grid conversion (phi_in) was the weak step.",
}

# ----------------------------------------------------------------------------- tests
EXT_V = V1["full"]["tests"]["T4_EXT_LOYO"]["monthly_rel_rms"]
TESTS = {
    "id": "P1r2.census_tests",
    "wp": "P1",
    "title": "P1 round 2: out-of-sample test of the v1 extension past the old census end",
    "kind": "test",
    "tests": [
        {"test_id": "P1r2.T10_ext_yunlin", "data_scope": "census_yunlin:2025-07..2026-06",
         "fresh_units": ["yunlin_census_new_months_2025-07..2026-06"],
         "candidate": "county_electricity/full/v1 yunlin_extension.parquet (YL_clean_hat = "
                      "TPC YL_ag x r(moy), r fitted 2016-01..2025-06)",
         "truth": "YL_clean_all from the new L0, full cleaning"},
        {"test_id": "P1r2.T11_ext_changhua", "data_scope": "census_changhua:2025-07..2026-06",
         "fresh_units": ["changhua_census_months_2025-07..2026-06"],
         "candidate": "county_electricity/full/v1 changhua_candidates.parquet, estimator "
                      "scaled_apre (primary), scaled (secondary), transfer (contrast), scope "
                      "county",
         "truth": "CH_clean_all county from the new L0, full cleaning"},
    ],
    "zones": NA("county scope"),
    "priors": NA("frozen v1 candidates; nothing is fitted"),
    "burn_in": NA("no model run"),
    "folds": NA("one fixed 12-month window"),
    "fresh_units": ["yunlin_census_new_months_2025-07..2026-06",
                    "changhua_census_months_2025-07..2026-06"],
    "gate_scores": NA("reported pass/fail of frozen v1 candidates; v1 gates are not "
                      "re-decided"),
    "primary_model": "county_electricity/full/v1: YL_clean_hat (T10) and scaled_apre (T11)",
    "inputs": INPUTS_COMMON + INPUTS_CENSUS + [f"{CE}/full/v1/yunlin_extension.parquet",
                                               f"{CE}/full/v1/changhua_candidates.parquet"],
    "truth_definition": TRUTH_DEF,
    "month_rule": "2026-07 is excluded (partial per SOURCE.json). Any other month m of "
                  "2025-07..2026-06 is scored only if the number of county meters (unique "
                  "電號1) with a non-missing, non-zero reading in m is at least 0.75 x that "
                  "number in m-12; excluded months are reported.",
    "thresholds": {
        "T10": "pass iff |sum pred / sum obs - 1| <= 0.10 over the scored months (the plan's "
               "extension threshold, annual RMS <= 10 %, applied to the one 12-month window)",
        "T11": "same rule for scaled_apre; scaled and transfer reported",
        "secondary_reported": [f"monthly relative RMS against the v1 monthly sd ({EXT_V:.4f} "
                               "for the extension)", "coverage of pred +- 1.645 sd: Wilson "
                               "95 % interval on n months contains 0.90 (n < 30: no point "
                               "window, plan 4.4)", "per-month relative errors"],
    },
    "prior_exposure": EXPOSURE + [
        "T10: no Yunlin value after 2025-06 from the new L0 had been seen by this WP.",
        "T11: the Changhua county truth totals for these months were visible in profile2.json "
        "(above); the v1 candidate values for these months had not been compared with them.",
    ],
    "interpretation": "A pass shows that extending Yunlin first and carrying Changhua as a x "
                      "extended Yunlin (no splice) tracked the measured census over the first "
                      "year past the old cache. A failure bounds the error of the TPC "
                      "extension that WP P2def would use from 2026-07 on.",
}

GATE = {
    "id": "P1r2.tpc_observed_gate",
    "wp": "P1",
    "title": "Integrity gate of county_electricity/tpc_observed/v1 (observed TPC d007012 "
             "county x sector monthly sales and the Energy Administration national series)",
    "kind": "test",
    "tests": [
        {"test_id": "P1r2.G1_archive_hash", "data_scope": "tpc_archive:2026-10-12",
         "fresh_units": ["tpc_d007012_archive_2026-10-12", "moeaea_236_256_archive_2026-10-12"]},
        {"test_id": "P1r2.G2_completeness", "data_scope": "tpc_archive:2026-10-12",
         "fresh_units": ["tpc_d007012_archive_2026-10-12", "moeaea_236_256_archive_2026-10-12"]},
        {"test_id": "P1r2.G3_internal_share", "data_scope": "tpc_archive:2026-10-12",
         "fresh_units": ["tpc_d007012_archive_2026-10-12"]},
        {"test_id": "P1r2.G4_refetch", "data_scope": "tpc_archive:2026-10-12",
         "fresh_units": ["tpc_d007012_archive_2026-10-12"]},
    ],
    "zones": NA("county tables"),
    "priors": NA("no fitted parameter"),
    "burn_in": NA("no model run"),
    "folds": NA("no folds"),
    "fresh_units": ["tpc_d007012_archive_2026-10-12", "moeaea_236_256_archive_2026-10-12"],
    "gate_scores": NA("the gate is the conjunction G1 and G2 and G3; G4 is reported"),
    "primary_model": "county_electricity/tpc_observed/v1",
    "inputs": ["data_fetch_api/new_measurements/tpc_d007012/2026-10-12/d007012.csv",
               "data_fetch_api/new_measurements/tpc_d007012/2026-10-12/SOURCE.json",
               "data_fetch_api/new_measurements/moeaea_agri_electricity/2026-10-12/set_236.csv",
               "data_fetch_api/new_measurements/moeaea_agri_electricity/2026-10-12/set_256.csv",
               "data_fetch_api/new_measurements/moeaea_agri_electricity/2026-10-12/"
               "SOURCE.json"],
    "thresholds": {
        "G1": "sha256 of d007012.csv, set_236.csv and set_256.csv equal their SOURCE.json "
              "records (pass/fail)",
        "G2": "彰化縣 and 雲林縣: every (sector, month) present in the file occurs exactly once; "
              "農林漁牧 present with a value > 0 in every month 2016-01..2026-08 (128 months); "
              "the merged national agricultural series has exactly one value > 0 per month "
              "2007-01..2026-08 (236 months); where sets 236 and 256 overlap, their values "
              "agree to 1e-9 relative (pass/fail)",
        "G3": "彰化縣 and 雲林縣 rows: |published 用電佔比(%) - 100 x kWh / sum of the county's "
              "sector kWh that month| <= 0.01 percentage point on every row (pass/fail); "
              "other counties reported",
        "G4": "today's re-download of d007012 and sets 236/256 is byte-identical to the "
              "archive, or is archived as a new dated L0 directory; differences in 彰化縣 / "
              "雲林縣 農林漁牧 values for overlapping months are reported (reported only)",
        "slice": "G2 and G3 are also evaluated on the rows dated <= 2022-12 alone and reported "
                 "for pre-origin consumers of the time slice",
    },
    "interpretation": "A passed gate means the observed table is the archived publisher file, "
                      "complete and internally consistent, so it can serve as an independent "
                      "validator of census coverage (the revised P1 role). It says nothing "
                      "about the pump-like share of 農林漁牧 (a non-pump load is included).",
    "prior_exposure": ["v1 (fetch_log.json, 02:39Z) found the re-download byte-identical; v1 "
                       "used these files for its fits; no completeness or share check had "
                       "been run on them as a gate."],
}


def main(dry: bool) -> int:
    docs = [DECLARATION, TESTS, GATE]
    for d in docs:
        errs = pr.validate_prereg(d)
        if errs:
            raise SystemExit(f"{d['id']}: {errs}")
    if dry:
        print("valid:", [d["id"] for d in docs])
        return 0
    # the main session's census profile and the plan revision exposed the confirmations:
    # record that as a design-time look (dated to the profile) before registering
    pr.record_look(["changhua_census_vs_P1_v1"], "census_changhua:2012-01..2026-07", "P1",
                   "design_time_seed", seeded_time="2026-10-12T05:44:00.000000Z",
                   note="recorded by P1 round 2 for the main session: inspection/profile2.json "
                        "and the plan revision note disclosed Changhua census totals and the "
                        "a / in-grid comparisons before P1's round-2 pre-registration")
    out = {}
    for d in docs:
        path, sha = pr.write_prereg(d)
        out[d["id"]] = {"path": str(path.relative_to(REPO)), "sha256": sha}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--dry" in sys.argv))
