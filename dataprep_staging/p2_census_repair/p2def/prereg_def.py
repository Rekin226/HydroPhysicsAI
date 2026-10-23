"""Pre-registrations of WP P2def (A1 tool ``harness.prereg3``), written before any registered
score is computed:

* ``P2def.v1`` (kind test): the P2d mid-2013 freeze back-test (Yunlin and Changhua gate,
  Chiayi reported), the construction checks of the P2d layer (both variants) and of the P2e
  extension, and the P2e extension backtests (x2 selection and sd, x1 acceptance);
* ``P2def.f.layer_response.v1`` (kind declaration): the P2f analysis, fixed before it runs.
  Its units are design-time (the Yunlin nested sites are the billing pilot's s07 units, and
  every candidate well's 2012-2022 heads were read by earlier WPs: a test on fresh units is
  impossible), so its results are design-time evidence.

Usage: python -I prereg_def.py test|decl [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg import not_applicable  # noqa: E402

PREREG_ID = "P2def.v1"
DECL_ID = "P2def.f.layer_response.v1"
NONE_ELEC = ("electricity object of WP P2def (TPC census pumps, meters, townships or TPC "
             "county sales); contains no registered well, leveling benchmark or model cell of "
             "the validation-unit registries")


def unit(desc: str) -> dict:
    return {"description": desc, "none": NONE_ELEC}


def tests() -> list[dict]:
    out = []
    for k in ("yunlin", "changhua", "chiayi"):
        gating = k != "chiayi"
        out.append({
            "test_id": f"P2d.bt.{k}",
            "data_scope": "c0_pump_electricity_v3:2007-01..2022-11",
            "fresh_units": [f"P2d:bt_cohorts_2014_2016_{k}"],
            "external_units": {f"P2d:bt_cohorts_2014_2016_{k}": unit(
                f"{k}: kept pumps (C0 v3) with a valid install month in 2014-01..2016-06, "
                "their township counts and their recorded-arm kWh 2014-01..2022-11, against "
                "the layer rule applied at a mid-2013 freeze (d2_backtest.py)")},
            "what": "plan P2d validation: census frozen at mid-2013 (installs <= 2013-06 and "
                    "readings to 2013-06 known); rate per township x class = kept installs "
                    "2007-2012 / 6; placement by the township's 2007-2012 installs per cell; "
                    "relative age ramp g(c, a) of the 2007-2012 cohorts (readings to 2013-06; "
                    "unmeasured ages hold; classes < 200 cohort pumps use the pooled ramp); "
                    "energy n * g * P_F(c, t), P_F = mean energy per pump of the frozen census; "
                    "monthly cohorts 2014-01..2016-06",
            "metric": "(1) median over townships with >= 10 actual kept installs in "
                      "2014-01..2016-06 of |predicted - actual| / actual installs; (2) fan "
                      "(in-grid) kWh of the cohorts 2014-01..2022-11: |predicted / actual - 1|",
            "threshold": "(1) <= 0.30 and (2) <= 0.20 (plan P2d)",
            "gating": ("part of both census_post2016 gates (preorigin and full); on failure "
                       "the products are published with gate FAILED (only design-time "
                       "evidence and after-the-fact analysis open) and P2d is failed")
            if gating else "reported only (Chiayi: 318 in-grid kept pumps, 15 installs in "
                           "2007-2012; townships cannot reach 10 actual installs)",
            "secondary": ["county-scope kWh error", "annual in-grid errors 2014-2022",
                          "signed total count error", "per-township table"]})
    for v in ("preorigin", "full"):
        out.append({
            "test_id": f"P2d.k1.construction_{v}",
            "data_scope": "c0_pump_electricity_v3:2007-01..2026-07" if v == "full"
            else "c0_pump_electricity_v3:2007-01..2022-12",
            "fresh_units": [f"P2d:layer_{v}"],
            "external_units": {f"P2d:layer_{v}": unit(
                f"the assembled census_post2016/{v} arrays (both members)")},
            "what": "construction checks of the synthetic layer (d3_layer.check): (a) shapes, "
                    "dates, mask and grid hashes equal C0 v3; (b) NaN iff flag 8, values >= 0, "
                    "flag 8 only after the last complete month; (c) value 0 and flag 0 before "
                    "2016-07, flag 5 from 2016-07 to the last complete month; (d) sd = s_rel x "
                    "value by county, sd 0 where flag 0; (e, f) county in-grid energy and "
                    "expected counts recomputed by an explicit cohort loop from the stored "
                    "parameters; (g) aggregates (sum over counties, class totals, no-domestic); "
                    "(i) privacy (no uid, coordinate or meter key); (h, both variants built) "
                    "pre-origin equals full on months <= 2022-11",
            "metric": "violations per item; relative differences",
            "threshold": "every item passes ((e, f, h) relative difference <= 1e-9)",
            "gating": f"part of the census_post2016/{v} gate"})
    for k in ("yunlin", "changhua"):
        out.append({
            "test_id": f"P2e.x2.select_{k}",
            "data_scope": "tpc_census_monthly:2016-01..2026-06",
            "fresh_units": [f"P2e:x2_two_month_{k}"],
            "external_units": {f"P2e:x2_two_month_{k}": unit(
                f"{k}: monthly cleaned-census county totals (C0 full v3) and TPC farm-sector "
                "sales 2016-01..2026-06, rolling origins 2019-06..2026-04")},
            "what": "rolling-origin two-month-ahead extension error of each candidate (flat, "
                    "recent36, trend; Changhua also scaled = a x Yunlin's selected candidate) "
                    "(e2_extension.py)",
            "metric": "RMS over the last 36 origins (2023-05..2026-04) of log(sum pred / sum "
                      "census) of the two months after the origin",
            "threshold": "selection rule: lowest RMS is the county's extension; its monthly "
                         "log RMS over the same origins is the extension sd (uncalibrated)",
            "gating": "selection only (no pass/fail)"})
        out.append({
            "test_id": f"P2e.x1.annual_{k}",
            "data_scope": "tpc_census_monthly:2016-01..2026-06",
            "fresh_units": [f"P2e:x1_annual_{k}"],
            "external_units": {f"P2e:x1_annual_{k}": unit(
                f"{k}: one-year-ahead extensions from origins every June 2019..2025 against "
                "the census totals of the following July-June")},
            "what": "plan P1 extension acceptance applied to the selected candidate: "
                    "one-year-ahead annual relative error, origins June 2019..2025",
            "metric": "RMS over the 7 origins of (sum pred / sum census - 1)",
            "threshold": "<= 0.10 (plan P1 extension rule)",
            "gating": "part of the census_extension/full gate (both counties)"})
    out.append({
        "test_id": "P2e.k1.construction",
        "data_scope": "c0_pump_electricity_v3:2023-01..2026-07",
        "fresh_units": ["P2e:extension_arrays"],
        "external_units": {"P2e:extension_arrays": unit(
            "the assembled census_extension/full arrays 2026-07..2026-08")},
        "what": "construction checks (e3_build.check): (a) shapes, dates, grid hashes; (b) "
                "finite, >= 0, flag 5, sd = sd_rel x value; (c) in-grid sum = county hat x "
                "2023-2025 same-month in-grid share; (d) aggregates; (e) privacy; (f) county "
                "extension recomputed from the selected candidate",
        "metric": "violations per item",
        "threshold": "every item passes ((c) <= 1e-9, (f) <= 1e-12 relative)",
        "gating": "part of the census_extension/full gate"})
    return out


def inputs() -> list[str]:
    files = [D.C0["census_L1"] / "census_pumps.parquet"]
    for v in ("preorigin", "full"):
        files += [D.C0[v] / f"pumps_{D.FILETAG[k]}.npz" for k in D.COUNTIES]
        files += [D.C0[v] / "cells_by_class.npz", D.C0[v] / "county_class_monthly.parquet",
                  D.C0[v] / "MANIFEST.json"]
    files += [D.TPC_CSV, D.P2ABC["full"] / "county_class_monthly.parquet"]
    files += sorted(D.CODE_DIR.glob("*.py"))
    return [D.rel(f) for f in files if f.name != "prereg_def.py"]


def design_known() -> dict:
    ev = D.EVID / "design_time"
    out = {}
    for f in ("d0_census_design.json", "d0_tpc_parse.json"):
        if (ev / f).exists():
            out[f] = D.load_json(ev / f)
    if (ev / "d0_billing_completeness_by_year.csv").exists():
        out["d0_billing_completeness_by_year.csv"] = "see file (P2def-d0-billing-completeness)"
    return out


def doc_test() -> dict:
    ts = tests()
    return {
        "id": PREREG_ID, "wp": D.WP,
        "title": "P2def: post-2016 synthetic census layer (mid-2013 freeze back-test, "
                 "construction), census extension 2026-07..08 (backtests, construction)",
        "kind": "test", "tests": ts,
        "fresh_units": sorted({u for t in ts for u in t["fresh_units"]}),
        "zones": not_applicable("no zones; counties are the strata"),
        "priors": not_applicable("no priors"),
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("rolling origins and a single freeze back-test"),
        "thresholds": {t["test_id"]: t["threshold"] for t in ts},
        "gate_scores": not_applicable(
            "conjunctive gates, no weights: census_post2016/preorigin = P2d.bt.yunlin and "
            "P2d.bt.changhua and P2d.k1.construction_preorigin; census_post2016/full = "
            "P2d.bt.yunlin and P2d.bt.changhua and P2d.k1.construction_full; "
            "census_extension/full = P2e.x1.annual_yunlin and P2e.x1.annual_changhua and "
            "P2e.k1.construction"),
        "primary_model": (
            "P2d: synthetic post-2016 layer per county (Yunlin, Changhua, in-grid Chiayi), "
            "members zero (no layer), low (rate = kept installs 2010-2015 without 2013 / 5) and "
            "high (2010-2015 / 6), unweighted; monthly cohorts of rate/12 per township x class "
            "from 2016-07 to the last complete month (pre-origin 2022-12, full 2026-06; full "
            "2026-07 flag 8); placement by the member's install years per cell; energy n * "
            "g(c, age) * P(c, t), g = relative ramp of the 2010-2015 cohorts (both members; "
            "ages 1-6, held after 6; < 200 cohort pumps -> pooled), P = mean energy per "
            "installed kept pump of the class (C0 v3 recorded arm of the variant); flag 5; sd = "
            "s_rel x value, s_rel = max(|fan kWh error|, median township count error) of the "
            "county's back-test (Chiayi: max of the two), uncalibrated. P2e: per county the "
            "x2-selected candidate applied at origin 2026-06 to TPC 2026-07 and 2026-08; class "
            "x cell by the county's 2023-2025 same-month shares; in-grid Chiayi = its 2023-2025 "
            "same-month ratio to the Yunlin county census x Yunlin's extension; flag 5; sd = "
            "value x monthly log RMS (x2), uncalibrated; recorded-forcing arm only (U4)."),
        "inputs": inputs(),
        "interpretation": {
            "design_time_knowledge": (
                "Known before this registration: (i) county-level install counts by year "
                "for the whole Yunlin census (billing pilot s05 log and the audit: 1,357 in "
                "2014, 781 in 2015, 215 in 2016-H1, all census pumps, kept or not), i.e. the "
                "Yunlin back-test count truth at county level; the 2007-2012 kept install "
                "counts of every county (look [P2def-d0-census-design]) - together they imply "
                "a Yunlin county-level over-prediction of roughly a third; nothing at "
                "township level and nothing about Changhua's 2013-2016 installs or any "
                "cohort's kWh was computed; (ii) P1's after-the-fact T10/T11 (flat r(moy) "
                "extension over-predicted the 2025-07..2026-06 census by 15.8 % in Yunlin) and "
                "C0's annual coverage table (census / TPC drifting down after 2021): the P2e "
                "candidates recent36 and trend were added because of that knowledge, and the "
                "x2/x1 origins include those months; (iii) billing completeness 2016-2026 "
                "(look [P2def-d0-billing-completeness])."),
            "not_fresh": "P2e x1/x2 units overlap P1's scored months (T4 2016-2024 LOYO, T10/"
                         "T11 2025-07..2026-06) and C0's T4 coverage; they are electricity "
                         "objects with members 'none' and do not block, but their outcomes "
                         "are partly known (above). The P2d back-test's Yunlin count total is "
                         "partly known (above).",
            "design_time_checks": "synthetic recovery of the P2d core on a toy census "
                                  "(t0_synthetic.py; no real data); a predict-only run of the "
                                  "back-test may be made to check the code (no truth computed)",
            "plan_reading": {
                "low_high": "plan P2d: low = 2010-15 rate without 2013, high = with the 2013 "
                            "registration surge, applied to every county",
                "backtest_rate_window": "2007-2012 (the 6 calendar years before the freeze "
                                        "year, as 2010-2015 is for the 2016-07 freeze); the "
                                        "surge exclusion has no counterpart there",
                "fan_total": "in-grid total of the county ('fan'); county scope reported",
                "kept": "rates, ramp and truth on C0's kept pumps (model units); dropped "
                        "(duty-capped) pumps carry no model energy",
                "extension_scope": "the census is complete to 2026-06 (task update); P2e "
                                   "covers 2026-07 (also the census + imputation of P2abc "
                                   "full/v2, compared) and 2026-08"},
            "design_known": design_known(),
            "code_at_registration": D.p2def_code()},
    }


def doc_decl() -> dict:
    return {
        "id": DECL_ID, "wp": D.WP,
        "title": "P2f: local-only layer-response ratio at nested head sites, observed and "
                 "modelled (cwa theta), with year-shuffled placebos (declared before running)",
        "kind": "declaration",
        "tests": not_applicable("design-time evidence, not a test on fresh units: the Yunlin "
                                "nested sites are the billing pilot's s07 units (pilot ratio "
                                "0.49-0.52) and every candidate well's 2012-2022 heads were "
                                "read by earlier WPs (registry fresh well units already carry "
                                ">= 38 blocking looks on heads 2012-2022)"),
        "zones": not_applicable("counties Yunlin and Changhua, pooled"),
        "priors": not_applicable("no priors"),
        "burn_in": not_applicable("no model run; existing forward archives"),
        "folds": not_applicable("site bootstrap"),
        "fresh_units": not_applicable("none (design-time units)"),
        "thresholds": {
            "local_response_identified": "site-cluster bootstrap one-sided p < 0.05 for mean "
                                         "D_w > 0 (D_w = partial R2 real local energy minus "
                                         "mean over 20 year-shuffled placebos), per county and "
                                         "layer (1, 2)",
            "ratio_locally_identified": "both layer 1 and layer 2 identified in the county",
            "model_inside_observed": "reported (the I1 rung-R4 test, evaluated on the "
                                     "current cwa theta)"},
        "gate_scores": not_applicable("no gate: evidence product"),
        "primary_model": (
            "heads/L1_clean v3 (flags 0-1, guard slice to 2022-12), months 2012-01..2022-11; "
            "wells layer 1-4, valid coordinates, Yunlin or Changhua (station polygon), >= 60 "
            "observed months; site = site_id; local energy = kept pumps (C0 preorigin v3 "
            "recorded arm, all in-grid counties) within 3 km (2 and 5 km and the P2abc "
            "preorigin v2 Chow-Lin arm reported); covariates: high-passed rain lags 0-2 "
            "(nearest CODIS gauge with >= 90 % valid months), the in-grid total of the well's "
            "county, kWh of kept pumps > 15 km away; slope of high-passed head (x minus the "
            "centred 25-month mean, >= 12) on high-passed local energy, >= 48 usable months; "
            "site ratio L1/L2 = median L1 slope / median L2 slope; statistic median over "
            "sites; 90 % interval 5-95 % of 4000 site-bootstrap medians (seed 2026); model "
            "side: same regression on datum_gate_cwa.npz heads_mean baseline (primary), the "
            "pilot's cwa fit_full hindcast (secondary), datum_gate_chc.npz (supplementary, "
            "another session's Changhua-census run); placebo: 20 year-shuffled local-energy "
            "copies (seeds 6000+r, complete years 2012-2021 deranged); depth proxy: in-grid "
            "kept energy share of meters > 10 HP (2012-2022)."),
        "inputs": [D.rel(D.HEADS_L1 / "heads_monthly.parquet"),
                   D.rel(D.HEADS_L1 / "station_meta_preorigin.parquet"),
                   D.rel(D.HEADS_L1 / "MANIFEST.json"),
                   D.rel(D.FWD_CWA), D.rel(D.FWD_CHC), D.rel(D.THETA_CWA),
                   D.rel(D.PILOT_PHYS / "hindcast_cwa.npz"), D.rel(D.RAIN), D.rel(D.RAIN_ST),
                   D.rel(D.C0["census_L1"] / "census_pumps.parquet")] +
                  [D.rel(D.C0["preorigin"] / f"pumps_{D.FILETAG[k]}.npz") for k in D.COUNTIES]
                  + [D.rel(D.P2ABC["preorigin"] / f"pumps_{D.FILETAG[k]}.npz")
                     for k in D.COUNTIES]
                  + [D.rel(D.CODE_DIR / "f1_layer_response.py"), D.rel(D.CODE_DIR / "dlib.py")],
        "interpretation": {
            "role": "design-time evidence for I1 rung R4 (plan P2f: used in I1 only after R3 "
                    "is accepted); the cwa model side is provisional (batch 8 folds still "
                    "running) and has no Changhua pumping, so its Changhua ratio is not "
                    "meaningful",
            "known_before": "the billing pilot's Yunlin numbers (s07, rain-only covariates, "
                            "stage3_datum_gate theta): observed L1/L2 0.49-0.52, model "
                            "0.20-0.21 at 19-22 sites; L3/L2 0.84-0.85 vs 0.69-0.71",
            "code_at_declaration": D.code_sha(D.CODE_DIR / "f1_layer_response.py",
                                              D.CODE_DIR / "dlib.py")},
    }


def main(which: str, dry: bool) -> None:
    d = doc_test() if which == "test" else doc_decl()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("\n".join(errs))
    if dry:
        print(json.dumps({k: (v if k != "tests" else [t["test_id"] for t in v]
                              if isinstance(v, list) else v)
                          for k, v in d.items() if k not in ("inputs",)},
                         ensure_ascii=False, indent=1, default=str)[:6000])
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main(sys.argv[1], "--dry-run" in sys.argv)
