"""Round-2 pre-registration of WP P2abc (A1 tool ``harness.prereg3``), written after the
stage-1 build (values only) and before any round-2 test statistic was computed.

Usage: python -I prereg_v2.py [--dry-run]
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg2 import not_applicable  # noqa: E402

PREREG_ID = "P2abc.r2.c0_based.v1"
NONE = ("electricity-meter object of WP P2abc (TPC census kWh of the C0 v3 products); "
        "contains no registered well, leveling benchmark or model cell of the validation-unit "
        "registries")
R1_UNITS = {
    "P2r1:yunlin_amp_monthly_irrigation_meters": "round-1 Yunlin AMP_V2 validation meters "
                                                 "(scored 2026-10-12, look [P2-r1-seed-meter-"
                                                 "validation])",
    "P2r1:yunlin_amp_phaseA_julys_2013_2024": "round-1 Yunlin July backtest units (look "
                                              "[P2-r1-seed-july])",
    "P2r1:yunlin_amp_zero_classification": "round-1 Yunlin zero classification (look "
                                           "[P2-r1-seed-zeros])",
}


def none_unit(desc: str) -> dict:
    return {"description": desc, "none": NONE}


def wells_unit(county_zh: str, desc: str) -> dict:
    st = pd.read_parquet(L.REPO / "AMP_V2" / "data" / "fan_stations.parquet",
                         columns=["sid", "CountyName"])
    ids = sorted(st.loc[st["CountyName"] == county_zh, "sid"].astype(str))
    return {"description": desc, "ids": ids, "window": "2012-01..2022-11"}


def inputs() -> list[str]:
    out = []
    for v in ("preorigin", "full"):
        d = L.C0_DIR[v]
        out += [L.rel(d / "MANIFEST.json")]
        for c in L.COUNTIES:
            t = L.FILETAG[c]
            out += [L.rel(d / f"pumps_{t}.npz"), L.rel(d / f"meters_{t}.parquet")]
        out += [L.rel(d / "cells_by_class.npz"), L.rel(d / "cells_total.npz")]
    out += [L.rel(L.C0_L1 / "MANIFEST.json"), L.rel(L.C0_L1 / "census_pumps.parquet")]
    out += [L.rel(p) for p in sorted(L.WORK.iterdir()) if p.suffix in (".npz", ".parquet")]
    out += [L.rel(L.WORK / "stage1.json")]
    out += ["AMP_V2/data/fan_stations.parquet",
            "data_fetch_api/new_measurements/cwa_codis_rain/rf_timeseries_codis.csv",
            "chou-shui-data/data/rf_stations.csv",
            "results/dataprep/_pilots/billing_and_census_gaps/zero_runs_summary.csv",
            "results/dataprep/_pilots/billing_and_census_gaps/census_zeros.json",
            "results/dataprep/yunlin_electricity/_work_p2abc/zeros.json"]
    return out


def code_hashes() -> dict:
    return {L.rel(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(L.CODE_DIR.glob("*.py"))}


def tests() -> list[dict]:
    T = []
    for c in L.COUNTIES:
        ext = {f"P2r2:val_meters_{c}": none_unit(
            f"{c}: kept irrigation meters with pre-origin detected phase monthly, flag 0 and "
            f"finite in every month 2012-01..2022-12, positive total, with coordinates")}
        t = {"test_id": f"P2r2.a1.skill_{c}",
             "data_scope": "c0_pump_electricity_v3:2007-01..2026-07",
             "fresh_units": [f"P2r2:val_meters_{c}"], "external_units": ext,
             "what": "P2a meter validation (plan 4.4 'Billing (P2a)'): pseudo-bimonthly "
                     "monthly meters re-split by the production pre-origin operator of the "
                     "county (Chiayi: Yunlin profile); pseudo phase = nearest kept irrigation "
                     "bimonthly meter of the same county (Chiayi: Yunlin or Chiayi); k-means "
                     "groups (n_init 3, random_state 0), N // n clusters for n in {3, 10, 30, "
                     "100, 300}, n = 1 every meter alone, plus fan-wide; evaluation months "
                     "2012-02..2022-11",
             "metric": "pooled skill 1 - SSE_chowlin / SSE_even over groups, per n",
             "threshold": "skill >= 0.15 at every n in {1, 3, 10, 30, 100, 300} (plan P2a)",
             "gating": ("ships the county's Chow-Lin arm (both variants); on failure the "
                        "county's Chow-Lin arm is replaced by its recorded C0 values (C0 "
                        "methods kept; meter_phase.parquet chowlin_shipped False) in both "
                        "products" if c != "chiayi" else
                        "reported only (16 candidate meters; Chiayi's 316 in-grid pumps follow "
                        "the Yunlin verdict because they use the Yunlin profile)"),
             "secondary": ["pro-rata by the same profile", "placebo skill (year-shuffled "
                           "profile, seeds 1..20) on the same meters and groups",
                           "full variant (profile and fit to 2026-07; evaluation "
                           "2012-02..2026-05)"]}
        if c == "yunlin":
            t["design_time_units"] = ["P2r1:yunlin_amp_monthly_irrigation_meters"]
            t["external_units"]["P2r1:yunlin_amp_monthly_irrigation_meters"] = none_unit(
                R1_UNITS["P2r1:yunlin_amp_monthly_irrigation_meters"])
        T.append(t)
    for c in ("yunlin", "changhua"):
        ext = {f"P2r2:sd_halves_{c}": none_unit(
            f"{c}: the validation meters of P2r2.a1.skill_{c} split at the median easting")}
        t = {"test_id": f"P2r2.a2.sdcov_{c}",
             "data_scope": "c0_pump_electricity_v3:2007-01..2026-07",
             "fresh_units": [f"P2r2:sd_halves_{c}"], "external_units": ext,
             "what": "coverage of the Chow-Lin sd: z = |chowlin - truth| / even per "
                     "group-month (zero even excluded); q90 per (n, calendar month) fitted on "
                     "one half (median easting; k-means groups within each half), scored on "
                     "the other, both directions; strata {1}, {3}, {10}, {30}, {100, 300}; a "
                     "stratum with < 20 clusters is pooled into the next smaller one; clusters "
                     "= groups; harness.coverage3 rule gate-3. Pre-origin variant (2012-02.."
                     "2022-11) labels the pre-origin product; full variant (2012-02..2026-05) "
                     "labels the full product",
             "metric": "coverage3 gate per stratum (coverage, interval, n_eff, k, calibrated)",
             "threshold": "label 'calibrated' iff every stratum passes gate-3 and lies in the "
                          "A1 envelope (k >= 20); otherwise 'uncalibrated' (plan 4.7); the sd "
                          "model is not tuned after this result",
             "gating": "sets the sd label only"}
        if c == "yunlin":
            t["design_time_units"] = ["P2r1:yunlin_amp_monthly_irrigation_meters"]
            t["external_units"]["P2r1:yunlin_amp_monthly_irrigation_meters"] = none_unit(
                R1_UNITS["P2r1:yunlin_amp_monthly_irrigation_meters"])
        T.append(t)
    for c, zh in (("yunlin", "雲林縣"), ("changhua", "彰化縣")):
        u = f"P2r2:head_check_wells_{c}"
        T.append({
            "test_id": f"P2r2.a3.heads_{c}", "data_scope": "heads:2012-01..2022-11",
            "fresh_units": [u],
            "external_units": {u: wells_unit(zh, f"fan stations of {zh} (superset of the wells "
                                                 f"the head check retains), heads 2012-01.."
                                                 f"2022-11")},
            "what": "placebo-controlled head check (plan P2a): see t2_head_check.py docstring; "
                    "wells of the county with >= 60 months 2012-01..2022-11, layer 1-4; "
                    "covariates rain lags 0-2 (nearest CODIS gauge), the county's in-grid kept "
                    "total and kept kWh > 15 km away (same energy version); D_w(r) = pr2(real "
                    "Chow-Lin) - mean pr2(20 year-shuffled placebo splits); primary = mean "
                    "over wells of the mean of D_w(1 km), D_w(2 km); site-cluster bootstrap "
                    "4000, seed 2026",
            "metric": "one-sided bootstrap p of mean D_w > 0",
            "threshold": "p < 0.05 (plan P2a)",
            "gating": "none: P2a ships on the meter validation; on failure the head gain is "
                      "reported as not distinguishable from regional-signal injection",
            "secondary": ["each radius 1, 2, 3, 5 km", "Chow-Lin vs recorded partial R2 and "
                          "the share of wells where Chow-Lin beats recorded (pilot statistic)",
                          "fixed-nuisance design (recorded regional covariates in every fit)",
                          "pooled over both counties"]})
    for c in L.COUNTIES:
        ext = {f"P2r2:july_units_{c}": none_unit(
            f"{c}: kept meters with code M2 and full-variant detected phase A, Julys "
            f"2010-2025")}
        t = {"test_id": f"P2r2.b1.july_{c}",
             "data_scope": "c0_pump_electricity_v3:2010-07..2025-08",
             "fresh_units": [f"P2r2:july_units_{c}"], "external_units": ext,
             "what": "P2b rolling-origin July backtest (plan 4.4 'July 2025 (P2b)'): truth = "
                     "(Jul + Aug) / 2 of year Y, estimate = mean of the same over Y-1..Y-3, "
                     "Y = 2013..2025, recorded C0 full values",
             "metric": "mean over years of |fan-total error| (%)",
             "threshold": "<= 20 % (plan P2b)",
             "gating": ("ships the county's July-2026 imputation in the full product (else "
                        "July 2026 stays NaN, flag 8)" if c != "chiayi" else
                        "ships Chiayi's July-2026 imputation (101 meters) under its own "
                        "verdict; reported"),
             "secondary": ["2013-2022 only", "max", "1 km cell relative RMSE", "Chow-Lin arm",
                           "imputation sd LOYO coverage (coverage3, clusters = years; label)",
                           "M0 July/June ratios 2022-2026"]}
        if c == "yunlin":
            t["design_time_units"] = ["P2r1:yunlin_amp_phaseA_julys_2013_2024"]
            t["external_units"]["P2r1:yunlin_amp_phaseA_julys_2013_2024"] = none_unit(
                R1_UNITS["P2r1:yunlin_amp_phaseA_julys_2013_2024"])
        T.append(t)
    T.append({
        "test_id": "P2r2.c1.zero_port_yunlin",
        "data_scope": "c0_pump_electricity_v3:2007-01..2025-07",
        "fresh_units": ["P2r2:zero_port_yunlin"],
        "design_time_units": ["P2r1:yunlin_amp_zero_classification"],
        "external_units": {"P2r2:zero_port_yunlin": none_unit(
            "Yunlin kept meters of C0 full v3 in the pilot view (see t4_zero_port.py)"),
            "P2r1:yunlin_amp_zero_classification": none_unit(
                R1_UNITS["P2r1:yunlin_amp_zero_classification"])},
        "what": "P2c port check: the pilot's s05(e) classification on the C0 Yunlin record in "
                "the pilot view (July 2025 of code-M2 meters set to 0 as in the AMP_V2 cache; "
                "phase by the pilot rule on 2012-01..2025-06; window 2012-01..2025-06)",
        "metric": "relative difference to the pilot of zero-months in interior runs <= 12, "
                  "interior runs > 12, trailing runs >= 12, trailing runs < 12; irrigation "
                  "zero frequency in April and August",
        "threshold": "each within 1 % (plan P2c)",
        "gating": "part of both products' gates (zero classification)",
        "secondary": ["leading runs, total zero months, run table by cell, pre-install months "
                      "(coding-dependent: C0 codes every pre-install month 0)"]})
    for v in ("preorigin", "full"):
        u = f"P2r2:assembled_{v}"
        T.append({
            "test_id": f"P2r2.k1.construction_{v}",
            "data_scope": ("c0_pump_electricity_v3:2007-01..2022-12" if v == "preorigin" else
                           "c0_pump_electricity_v3:2007-01..2026-07"),
            "fresh_units": [u], "external_units": {u: none_unit(
                f"the assembled yunlin_electricity/{v} product (built after this "
                f"registration)")},
            "what": "construction checks of the assembled product: (a) every re-split pair "
                    "keeps its C0 bill sum; (b) the Chow-Lin arm equals C0 (value, NaN, flag, "
                    "method) wherever its method is not chowlin_split or "
                    "own_3yr_same_month_mean; chowlin_split covers both months of a pair, the "
                    "imputation only C0's open July-2026 phase-A months; (c) C0 pre-install "
                    "months are 0, flag 0 in both arms; (d) the recorded arm's cell arrays "
                    "(value, flag; per county and all) equal C0's cells_by_class on every "
                    "cell-month without an imputed pump; (e) cells = sum of pumps, totals = "
                    "sum of classes, no-domestic = all - domestic, by-county sums = all; (f) "
                    "pump sd finite iff flag 3 or 4 and the cell sd finite iff a contributing "
                    "pump has one; flags in {0, 3, 4, 8, 9}; NaN iff flag 8; (g) privacy: no "
                    "personal column, raw meter number or coordinate in any output; (h) the "
                    "pre-origin manifest lists no input of pump_electricity/full; (i) zero "
                    "table: every listed zero is a 0 reading (or out-of-span missing) of a kept "
                    "meter inside the window",
             "metric": "violations per item", "threshold": "0 violations; 1e-6 kWh + 1e-9 x "
                                                           "|reference| for value equalities",
            "gating": f"yunlin_electricity/{v} gate (with P2r2.c1)"})
    return T


def doc() -> dict:
    ts = tests()
    return {
        "id": PREREG_ID, "wp": L.WP,
        "title": "P2a-c round 2 on the C0 v3 census (Yunlin, Changhua, in-grid Chiayi): "
                 "Chow-Lin meter validation and sd coverage, placebo head check, July "
                 "backtest, zero-classification port, product construction",
        "kind": "test", "tests": ts,
        "fresh_units": sorted({u for t in ts for u in t["fresh_units"]}),
        "zones": not_applicable("no zones; counties are the strata"),
        "priors": not_applicable("no priors"),
        "burn_in": not_applicable("no model run"),
        "folds": not_applicable("half-split (sd coverage) and leave-one-year-out (July sd) "
                                "are defined in the tests"),
        "thresholds": {t["test_id"]: t["threshold"] for t in ts},
        "gate_scores": not_applicable(
            "conjunctive gates, no weights: yunlin_electricity/preorigin = P2r2.k1."
            "construction_preorigin and P2r2.c1.zero_port_yunlin; yunlin_electricity/full = "
            "P2r2.k1.construction_full and P2r2.c1.zero_port_yunlin. P2r2.a1 ships or "
            "replaces each county's Chow-Lin arm, P2r2.b1 ships or withholds each county's "
            "July-2026 imputation, P2r2.a2 and P2r2.b1 set sd labels, P2r2.a3 is reported."),
        "primary_model": (
            "Chow-Lin split of the C0 v3 bimonthly irrigation bills built by "
            "p2_census_repair/v2/b1_build.py (stage-1 arrays hashed under inputs): per-county "
            "fan-wide profile (stagger deconvolution of the in-grid M2&A and M1&B irrigation "
            "population sums, lambda 0.01, 12-month share window, pairs complete by the "
            "variant end; Chiayi uses Yunlin's), X = [1, p/mean p], AR(1) rho in {0, 0.5, "
            "0.9} by likelihood per meter, window 2011-01..variant end, pairs re-split only "
            "where both months are flag-0 observations, clipped to [0, Y]; sd = q90(n_eff, "
            "month)/1.645 x even-split value from the validation groups pooled over both "
            "halves (n_eff = (sum e)^2 / sum e^2, log-linear in n_eff, clamped to [1, 300]); "
            "July 2026 phase-A (full): own 3-year same-month mean, flag 4, sd q90_rel/1.645 x "
            "value (meter level for pumps, cell level for cells)."),
        "inputs": inputs(),
        "interpretation": {
            "round1": "The round-1 registrations P2a.meter_validation.v1, P2a.sd_coverage.v1, "
                      "P2a.head_placebo.v1, P2b.july_backtest.v1 and P2c.zero_reproduction.v1 "
                      "(Yunlin, AMP_V2 cache; written 2026-10-12T02:35:58Z outside the A1 "
                      "index, sha256 recorded in _work_p2abc/prereg_hashes.json and unchanged) "
                      "stand. Four were scored by the interrupted round-1 session "
                      "(looks [P2-r1-seed-*], logged retroactively), and r1_reproduce.py "
                      "recomputed them identically ([P2-r1-repro]). P2a.head_placebo.v1 was "
                      "never scored; it is scored as registered after this registration.",
            "not_fresh": "Yunlin's round-2 units (validation meters, July units, zero port) "
                         "are essentially the round-1 units: the Yunlin L0 sheet equals the "
                         "AMP_V2 cache except one pump, and the kept set differs by 3 meters. "
                         "Their outcomes are known from round 1 (meter skill min 0.229; sd "
                         "coverage 0.878-0.899; July 14.0 %; zero port exact). Changhua's and "
                         "Chiayi's units have never been scored; they are the fresh part. The "
                         "head-check wells and 2012-2022 heads are design-time data of the "
                         "billing pilot (s07 regressions on the same wells and similar "
                         "covariates); what is new is the placebo control, the regional "
                         "covariates and the C0-based Changhua energy.",
            "design_before_registration": "Looks before this registration: [P2-d0] counts "
                                          "(validation candidates Yunlin 1789, Changhua 5406, "
                                          "Chiayi 16; Chow-Lin sets; one-month-missing pairs; "
                                          "M2 July zero shares 2022-2025 per county, which "
                                          "showed the July-2025 'bill in the second month' "
                                          "posting); [P2-b1] stage-1 build (profiles, rho "
                                          "shares, pairs re-split, conservation). No skill, "
                                          "coverage, regression or backtest was computed.",
            "choices": "Per-county profiles (Changhua's irrigation seasonality differs: CV "
                       "0.14 vs 0.234); Chiayi uses Yunlin's profile; C0's December-2022 fill "
                       "is kept in the pre-origin product (the Dec 2022-Jan 2023 bill is open "
                       "at the origin); round 1's own-3-year-mean edge fill is not used. The "
                       "July-2026 imputation base is the Jul-Aug pair mean (the recorded July "
                       "of 2025 is distorted by bills posted to the second month).",
            "u7": "No observation dated 2023-01 or later from heads, rings, piles, GNSS or "
                  "InSAR enters any test. The pre-origin product's fitted quantities (phase, "
                  "profile, beta, rho, sd model, zero classes) use data to 2022-12; its "
                  "census attributes and residual December-2022 halves are C0's declared "
                  "exemptions. The July backtest and the zero port use 2023-2025 electricity "
                  "(full-variant uses).",
            "code_at_registration": code_hashes()},
    }


def main(dry: bool) -> None:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("\n".join(errs))
    if dry:
        print(json.dumps({k: (v if k != "tests" else [t["test_id"] for t in v])
                          for k, v in d.items() if k != "inputs"}, ensure_ascii=False,
                         indent=1)[:4000])
        print(len(d["inputs"]), "inputs")
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
