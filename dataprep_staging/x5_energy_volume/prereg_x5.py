"""Write the X5 energy-to-volume registrations (harness.prereg3), each BEFORE its computation.

* ``step0_ab``  - declaration: STEP 0 gap decomposition (exploratory) and estimators (a), (b)
  (design-time);
* ``synthetic`` - declaration: estimator (c) and its synthetic recovery criterion;
* ``test``      - test: (c) against (a) on fresh units (written after the synthetic check).

Run under the shared registry lock:
    flock results/dataprep/preregistration/.x5.lock python -I prereg_x5.py <which> [--write]
(default: validate and dry-run).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from x5_energy_volume import x5const as K  # noqa: E402

NA = prereg3.not_applicable
HERE = Path(__file__).resolve().parent


def _code(*names: str) -> list[str]:
    return [rel_to_repo(HERE / n) for n in names]


def doc_step0_ab() -> dict:
    return {
        "id": K.PREREG["step0_ab"], "wp": K.WP, "kind": "declaration",
        "title": "X5 energy-to-volume: STEP 0 decomposition of the twin-vs-WRA irrigation "
                 "volume gap (exploratory) and estimators (a) WRA 水電比 volume per census kWh "
                 "and (b) pump physics per pump (design-time)",
        "tests": NA("declaration: exploratory and design-time quantities, no skill claim"),
        "checks": {
            "bases": "census kWh (L0 copy; NaN = pre-install = 0) per pump on three bases: raw "
                     "(every pump's series), dedup (each meter once, its first pump's copy, "
                     "shared over its pumps by rated HP; clean_census cap None) and kept "
                     "(dedup restricted to census_L1 v3 kept meters: clean_census cap 1.0, duty "
                     "2012-01..2022-12, the twin's forcing set); class = the pump's own "
                     "purpose_class; county scope (all census pumps of the county, in and out "
                     "of the twin grid, as WRA's county totals)",
            "a_estimator": "WRA irrigation groundwater (2025 trial report, parsed from the text "
                           "layer by wra.py: Tables 4-1.4..4-1.7, 4-3.19, 4-3.22) divided by "
                           "census irrigation kWh of the same window. PRIMARY (the scored "
                           "test's reference): Jan-May 2011-22, WRA = (14 x printed 2011-24 "
                           "Jan-May mean - printed 2023 - printed 2024) / 12, on the kept "
                           "basis; reported beside: dedup and raw bases, and the printed "
                           "2011-24 Jan-May and annual means over 2011-24 kWh (contain 2023-24: "
                           "reported only, never gating, U7)",
            "b_estimator": "per kept irrigation pump with 0 < HP <= 50 and pipe diameter 1.5-12 "
                           "in: design flow v x pi (D x 0.0254)^2 / 4 at rated power 0.746 x HP "
                           "(Hurr & Litke 1989 rate-time logic with the pipe as the rate): "
                           "m3/kWh = 3600 v A / P_kW; PRIMARY v = 1.5 m/s, county value = "
                           "Jan-May 2011-22 kept-kWh-weighted mean; v = 1.0 and 2.0 reported; "
                           "implied total dynamic head at eta 0.5 reported; (b2) band eta in "
                           "{0.4, 0.5, 0.6} x dynamic head {5, 10, 20} m on the twin's median "
                           "static lift 6.65 m",
            "step0": "irrigation, Jan-May 2011-22: gap = WRA / (c_twin x kept kWh), c_twin = "
                     "0.5 x 367.35 / (6.65 + 40) m3/kWh; split gap = conversion x coverage "
                     "under two hypotheses: dropped energy IS pumping (conversion = WRA / "
                     "(c_twin x dedup), coverage = dedup / kept) or is NOT (conversion = gap). "
                     "Description of the dropped energy (2012-22) per class: meters, GWh/yr, "
                     "duty and HP quantiles, full-load hours, shared-meter, high-voltage and "
                     "in-district shares, bimonthly-repeat share, month-of-year correlation "
                     "with kept irrigation, peak month, HP that would bring the meter to the "
                     "kept irrigation median duty",
            "label": "STEP 0 is exploratory, never validation; (a) and (b) are design-time "
                     "and neither is independent of electricity (WRA's 水電比 method is rated "
                     "flow x kWh / rated power on the same census)",
            "no_prior_look_statement": "the audit (2026-10-25) computed (a) on the kept and "
                                       "dedup bases (7.75/8.25 and 4.59/6.81 m3/kWh, 2011-24) "
                                       "and a pipe-velocity probe on Yunlin 2012-15; read by "
                                       "WP X5 before this declaration"},
        "zones": "county (changhua, yunlin)", "priors": NA("no fit"),
        "burn_in": NA("no fit"), "folds": NA("no fit"),
        "fresh_units": NA("no fresh units: design-time and exploratory quantities"),
        "thresholds": NA("reported with their labels"), "gate_scores": NA("no gate"),
        "primary_model": NA("no model; (a) kept Jan-May 2011-22 and (b) v 1.5 m/s are the "
                            "primary values"),
        "inputs": [rel_to_repo(K.CENSUS_L1), rel_to_repo(K.WRA_TXT),
                   rel_to_repo(K.L0_PARQ / "changhua_kwh_long.parquet"),
                   rel_to_repo(K.L0_PARQ / "yunlin_kwh_long.parquet"),
                   rel_to_repo(K.L0_PARQ / "changhua_census.parquet"),
                   rel_to_repo(K.L0_PARQ / "yunlin_census.parquet"),
                   rel_to_repo(K.PUMPING_PY)]
        + _code("x5const.py", "wra.py", "census.py", "step0_ab.py"),
        "interpretation": "(a) is the volume per kWh the twin must reproduce on its own kWh "
                          "basis to match WRA's irrigation budget; (b) bounds it physically; "
                          "STEP 0 says how much of the gap is conversion and how much is "
                          "energy the cleaning drops, and whether that energy looks like "
                          "pumping. None of these is a skill claim.",
        "written_by": rel_to_repo(Path(__file__)),
    }


C_SPEC = {
    "data": "hourly heads of the well, local time (UTC+8), from chou-shui-data/data/ls_cache/"
            "gw__<sid>__raw.parquet over 2010-01-01..2011-12-31 (the scored window); "
            "gw_timeseries.csv is never read",
    "qc": "sentinels (<= -900 or >= 9000) dropped; |x - median| >= 15 MAD dropped (AMP_V2 "
          "run_ampg_all rule); runs of identical hourly values >= 168 h removed; hourly "
          "mean; gaps <= 6 h interpolated; 15-day centred rolling mean removed; invalid "
          "hours set to 0",
    "amplitude": "AMP v1 (AMP_V2/amp.py amp_v1, imported read-only) at f_pump = 1 cpd, "
                 "bandwidth 0.05 cpd, trend_cut 0.1 cpd, smooth 30 days; monthly mean of the "
                 "envelope over valid hours; a month counts if >= 80 % of its hours are valid",
    "background": "B = hypot(1.49 x |O1|, 0.002 m), |O1| from a least-squares fit of O1, K1, "
                  "P1, S1, M2, S2, N2, S3 over the window's valid hours; A1p = sqrt(max(A1^2 "
                  "- B^2, 0))",
    "pumps": "census_L1 v3 kept irrigation pumps of both counties within 4 km of the well "
             "(TWD97; station coordinates with swapped pairs fixed by magnitude), distance "
             "floor 30 m; kWh = kept basis (meter series x HP share), per month; aquifer share "
             "s_i = X3 depth_layers v1 cell share of the well's aquifer (layer code 1-4 -> "
             "share_L0..L3), county combined share for pumps outside the grid",
    "physics": "periodic Theis: G(m) = F(d) |sum_i s_i E_i(m) / (D_m 86400) (ker + i kei)"
               "(r_i sqrt(w S / T))| / (2 pi T), w = 2 pi / 1 day, F(d) = 2 sin(pi d)/(pi d), "
               "d = 9.3 h / 24 (AMP-G network median duty, AMP_V2 README); pumps in phase; "
               "T = 592 m2/day (geometric centre of Liu et al. 2002, 58-6034 m2/day); S = Ss "
               "x 50 m with Ss = rho g n beta / BE (Jacob 1940; n 0.3, beta 4.4e-10 /Pa) from "
               "the well's accepted L1_clean v3 barometric-efficiency epochs (median BE_long), "
               "else the median Ss of all accepted wells",
    "well_value": "c_w = sum_m A1p(m) / sum_m G(m) over the well's valid Jan-May months "
                  "(>= 3 months needed); the well needs >= 20 kept irrigation pumps within "
                  "2 km",
    "county": "county of a well = majority county of the census pumps within 1.5 km",
    "code": "dataprep_staging/x5_energy_volume/amp_inversion.py (hashed in inputs)",
}


def doc_synthetic() -> dict:
    from x5_energy_volume import synth
    ids = sorted(synth.candidates()["sid"])
    return {
        "id": K.PREREG["synthetic"], "wp": K.WP, "kind": "declaration",
        "title": "X5 estimator (c), diurnal drawdown amplitude inverted through the periodic "
                 "Theis response of the kept irrigation pumps: specification and synthetic "
                 "recovery criterion (declared before the synthetic run)",
        "tests": NA("declaration: synthetic recovery check (design-time); the scored test is "
                    "registered separately after it"),
        "estimator_c": C_SPEC,
        "checks": {
            "synthetic": "for each of the " + str(len(ids)) + " candidate wells (confined, "
                         "LayerAttribute C, with ls_cache rows in 2010-2011): hourly heads "
                         "2010-2011 built from the same pumps, shares and 2010-2011 kept kWh "
                         "with a known c_true in {6, 3} m3/kWh, the estimator's own T and S, "
                         "per-pump on-rate lognormal (median 40 m3/h, sd log 0.5), duty U(6, "
                         "12) h, start N(7, 1.5) h + daily U(-1, 1) h, random pumping days "
                         "meeting each monthly volume; exact frequency-domain periodic Theis "
                         "response; plus O1/K1/P1/M2/S2/N2 tides, 2 mm S1, 1 m seasonal "
                         "cycle, 2 cm/day random walk, 3 mm white noise, 1 cm quantisation, "
                         "5 % block gaps; seed 20261025 + 10 k + c_true (synth.py)",
            "pass_criterion": "for each c_true, the pooled median of c_hat/c_true over wells "
                              "with an estimate, and each county's median, lie in [0.75, "
                              "1.25]. If it fails, the scored test is not registered with "
                              "this estimator; the failure is reported",
            "reported_only": "start-hour spread sd 3 h (misspecified); LSQ extractor",
            "label": "design-time synthetic check; no head observation is read"},
        "zones": NA("well-level"), "priors": NA("declared constants, x5const.py"),
        "burn_in": NA("none"), "folds": NA("none"),
        "fresh_units": NA("synthetic data only"),
        "thresholds": {"band": K.SYNTH_BAND, "c_true": list(K.SYNTH_C_TRUE)},
        "gate_scores": NA("pass/fail criterion above"),
        "primary_model": NA("estimator (c) as specified"),
        "inputs": [rel_to_repo(K.CENSUS_L1), rel_to_repo(K.STATIONS), rel_to_repo(K.BE_CSV),
                   rel_to_repo(K.X3_CELL_SHARES), rel_to_repo(K.X3_COUNTY_SPLIT),
                   rel_to_repo(K.AMP_PY), rel_to_repo(K.GAFD_PY),
                   rel_to_repo(K.L0_PARQ / "changhua_kwh_long.parquet"),
                   rel_to_repo(K.L0_PARQ / "yunlin_kwh_long.parquet")]
        + _code("x5const.py", "census.py", "amp_inversion.py", "synth.py"),
        "candidate_wells": ids,
        "interpretation": "a pass says the extraction chain (QC, AMP v1, background, duty "
                          "and intermittency averaging, quantisation) recovers c when the "
                          "physics and T, S are right; it says nothing about T, S, the "
                          "aquifer shares or the in-phase assumption, which the real test "
                          "carries as declared assumptions",
        "no_prior_look_statement": "no 2010-2011 head value has been read by WP X5 (only row "
                                   "counts of the ls_cache files)",
        "written_by": rel_to_repo(Path(__file__)),
    }


C_SPEC_V2 = {
    **C_SPEC,
    "physics": C_SPEC["physics"].replace(
        "d = 9.3 h / 24 (AMP-G network median duty, AMP_V2 README)",
        "per pump-month daily hours h = min(24, max(9.3 h, H / D_m)), H = kept kWh / "
        "(0.746 x rated HP), 9.3 h where HP is missing (v2 correction 1); F(h / 24) inside "
        "the sum"),
    "background": C_SPEC["background"].replace(
        "A1p = sqrt(max(A1^2 - B^2, 0))",
        "A1p = sqrt(max(A1^2 - B^2 - N(m)^2, 0)) with N(m)^2 the mean of the squared "
        "monthly AMP v1 envelopes at 0.85 and 1.15 cpd (v2 correction 2)"),
    "code": "dataprep_staging/x5_energy_volume/amp_inversion.py and amp_inversion_v2.py "
            "(hashed in inputs)",
}


def doc_synthetic_v2() -> dict:
    d = doc_synthetic()
    d.update({
        "id": "X5.energy_volume.synthetic_c.v2",
        "title": "X5 estimator (c) v2 (census duty per pump-month, side-band noise floor): "
                 "specification and synthetic recovery criterion, declared after the v1 "
                 "synthetic check and before the v2 run",
        "estimator_c": C_SPEC_V2,
        "history": "X5.energy_volume.synthetic_c.v1 ran first (report "
                   "results/dataprep/reports/X5.energy_volume.synthetic_c.v1/report.json) and "
                   "PASSED its declared criterion (pooled median ratio 0.93 at c_true 6, 0.95 "
                   "at 3; county medians 0.89-0.97). Its per-well output nevertheless showed "
                   "two biases of v1: a positive noise-floor bias where the expected signal "
                   "is a few mm (deep or weakly loaded wells: ratios 2-12) and a negative "
                   "bias where pumps that draw close to their rating dominate (a well at "
                   "0.4); v2 corrects both. No 2010-2011 head value has been read",
        "selection_rule": "declared now, on synthetic outcomes only: if v2 passes this "
                          "criterion, the scored test uses v2 as its primary estimator and "
                          "reports v1 beside it; if v2 fails, the test uses v1 (which passed) "
                          "and reports v2 beside it; never substituted after the score",
        "inputs": d["inputs"] + _code("amp_inversion_v2.py", "synth_v2.py")
        + [rel_to_repo(K.DATAPREP / "reports" / K.PREREG["synthetic"] / "report.json")],
    })
    d["checks"] = dict(d["checks"])
    d["checks"]["synthetic"] = d["checks"]["synthetic"].replace(
        "per-pump on-rate lognormal (median 40 m3/h, sd log 0.5)",
        "per-pump on-rate c_true x 0.746 HP x lam m3/h with an unknown load factor lam ~ "
        "U(0.6, 1.0) (lognormal median 40 m3/h where HP is missing); a month that needs "
        "more than 24 h a day runs continuously").replace("(synth.py)", "(synth_v2.py)")
    return d


def doc_test(version: str = "v2") -> dict:
    from x5_energy_volume import amp_inversion as A
    from x5_energy_volume import synth
    spec = C_SPEC_V2 if version == "v2" else C_SPEC
    ids = sorted(synth.candidates()["sid"])
    st = A.stations()
    rows_u = []
    for sid in st.loc[st["attr"] != "C", "sid"]:
        f = K.LS_CACHE / f"gw__{sid}__raw.parquet"
        if f.exists():
            ix = pd.to_datetime(pd.read_parquet(f, columns=[]).index)
            if ((ix >= pd.Timestamp(K.TEST_WINDOW[0])) & (ix < pd.Timestamp(K.TEST_WINDOW[1]))
                    ).any():
                rows_u.append(sid)
    ids_d = sorted(s for s in st.loc[st["attr"] == "C", "sid"]
                   if (K.AMP_WELLS / f"{s}.parquet").exists())
    uid = "X5EV:confined_ls_cache_2010_2011"
    synth_id = ("X5.energy_volume.synthetic_c.v2" if version == "v2"
                else K.PREREG["synthetic"])
    return {
        "id": K.PREREG["test"], "wp": K.WP, "kind": "test",
        "title": "X5 estimator (c) (diurnal drawdown amplitude -> m3/kWh, independent of WRA "
                 "volumes) against estimator (a) (WRA 水電比 per kept kWh), confined wells, "
                 "Jan-May 2010-2011",
        "estimator_version": version,
        "estimator_selection": "the selection rule declared in X5.energy_volume.synthetic_c.v2 "
                               "before its run: v2 primary only if its synthetic check "
                               "passed; v1 PASSED (pooled 0.93/0.95), v2 FAILED (Changhua "
                               "county median 0.63/0.73), so v1 is primary and v2 is "
                               "reported beside it, never substituted. Disclosed: v1 was "
                               "checked only against the v1 generator",
        "estimator_c": spec,
        "tests": [{
            "test_id": K.TEST_ID, "data_scope": K.TEST_SCOPE, "fresh_units": [uid],
            "external_units": {uid: {
                "description": "confined (LayerAttribute C) monitoring wells with an ls_cache "
                               "hourly series in 2010-01..2011-12, the only held head data "
                               "before 2012; their 2010-2011 values have never been read by "
                               "any WP (every 2012-2022 AMP well already sits in an earlier "
                               "fresh unit: H2 gap-fill units hold all 267, P5 within_wells "
                               "217; so no 2012-2019 well is fresh)",
                "ids": ids, "window": "2010-01..2011-12"}},
            "what": "Is the independent head-based volume per kWh within a factor 1.5 of "
                    "WRA's per kept kWh?",
            "metric": "ratio_w = c_w / a(county of the well), a = WRA Jan-May 2011-22 "
                      "irrigation groundwater per kept census irrigation kWh (step0_ab "
                      "report: Changhua and Yunlin values); statistic = median of ratio_w "
                      "over test wells with an estimate (pooled over counties)",
            "decision_rule": "PASS if 1/1.5 <= median <= 1.5; FAIL otherwise",
            "informative_gate": "NOT INFORMATIVE if fewer than 5 wells have an estimate",
            "label": "skill test on fresh units (first read of 2010-2011 heads)"}],
        "reported_units": {
            "X5EV:unconfined_ls_cache_2010_2011_reported": {
                "description": "non-confined wells with ls_cache rows in 2010-2011: the same "
                               "estimator with the elastic S and with S = Sy 0.1; reported "
                               "only, never gating (storage at 1 cpd is not pinned between "
                               "the two)", "ids": sorted(rows_u), "window": "2010-01..2011-12"},
            "X5EV:confined_amp_2012_2019_designtime": {
                "description": "confined wells with AMP_V2 hourly files, Jan-May 2012-2019, "
                               "same estimator; design-time label (these wells sat in "
                               "earlier fresh units); computed after the score, never gating",
                "ids": ids_d, "window": "2012-01..2019-12"}},
        "reported_variants_not_gating": [
            "per-county medians (Changhua has at most 4 confined wells)",
            "well-bootstrap 90 % interval of the pooled median (seed 20261026)",
            "T = 58 and 6034 m2/day (the Liu et al. 2002 range ends)",
            "duty 4 h and 12 h", "S x 0.1 and S x 10", "LSQ 1-cpd extractor (declared "
            "alternative extractor, reported beside, never substituted)",
            "all months instead of Jan-May", "dedup basis on both sides",
            "the other estimator version (v1 or v2), reported beside, never substituted",
            "kept-basis c against a on the dedup basis", "implied per-pump on-rate"],
        "zones": "county of the well (majority county of pumps within 1.5 km)",
        "priors": "declared constants (x5const.py): T 592 m2/day, Ss from BE, b 50 m, duty",
        "burn_in": NA("no fit"), "folds": NA("no fit"),
        "fresh_units": [uid],
        "thresholds": {"ratio_band": K.RATIO_BAND, "min_wells": 5,
                       "min_pumps_2km": K.MIN_PUMPS, "min_janmay_months": K.MIN_JM_MONTHS},
        "gate_scores": NA("single decision rule, reported whichever way it falls"),
        "primary_model": f"estimator (c) {version} as in estimator_c; reference (a) kept "
                         "Jan-May 2011-22",
        "attempt_policy": "engineering retries (crash, missing file, NaN) before any metric are "
                          "free and logged in results/dataprep/x5_energy_volume/attempts.jsonl;"
                          " the score runs once (score_c.py writes report.json once); at most "
                          "one exploratory follow-up after a FAIL, labelled 'post-failure, "
                          "exploratory; never validation; cannot promote'",
        "stage2_rule": "the stage-2 twin rung (conversion fix) is specified with (a) whatever "
                       "the outcome; a PASS labels its conversion 'independently corroborated "
                       "within 1.5x', a FAIL or NOT INFORMATIVE labels it 'set by WRA's "
                       "electricity-based ratio only'",
        "inputs": [rel_to_repo(K.CENSUS_L1), rel_to_repo(K.STATIONS), rel_to_repo(K.BE_CSV),
                   rel_to_repo(K.X3_CELL_SHARES), rel_to_repo(K.X3_COUNTY_SPLIT),
                   rel_to_repo(K.AMP_PY), rel_to_repo(K.GAFD_PY),
                   rel_to_repo(K.DATAPREP / "reports" / K.PREREG["step0_ab"] / "report.json"),
                   rel_to_repo(K.DATAPREP / "reports" / synth_id / "report.json"),
                   rel_to_repo(K.DATAPREP / "reports" / "X5.energy_volume.synthetic_c.v2"
                               / "report.json")]
        + _code("x5const.py", "census.py", "amp_inversion.py", "amp_inversion_v2.py",
                "pump_cache.py", "score_c.py", "synth.py"),
        "interpretation": "PASS: an electricity-free route (heads, literature T, BE storage) "
                          "puts the volume per kept kWh within 1.5x of WRA's, so WRA's ratio "
                          "(8.3 Changhua, 8.7 Yunlin) and not the twin's 3.94 is the better "
                          "conversion; it does not pin T, S or the in-phase assumption, each "
                          "of which moves c roughly in proportion. FAIL: the head route does "
                          "not corroborate WRA at this precision; the reported T range says "
                          "whether a different T inside Liu et al.'s range would. Neither "
                          "outcome is a statement about the twin's skill.",
        "no_prior_look_statement": "WP X5 has read no 2010-2011 head value. Disclosed design "
                                   "history: the audit's aggregate diurnal and tidal amplitudes "
                                   "(2016-2019, 238 wells) and AMP_V2 README aggregates "
                                   "(2012-2022) were read; the synthetic checks used only "
                                   "census data and generated heads",
        "written_by": rel_to_repo(Path(__file__)),
    }


DOCS = {"step0_ab": doc_step0_ab, "synthetic": doc_synthetic,
        "synthetic_v2": doc_synthetic_v2, "test_v2": lambda: doc_test("v2"),
        "test_v1": lambda: doc_test("v1")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=sorted(DOCS))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    d = DOCS[a.which]()
    errs = prereg3.validate_prereg(d)
    print("validate:", errs or "ok")
    missing = [p for p in d["inputs"] if not (K.CODE_DIR.parents[1] / p).exists()]
    print("missing inputs:", missing or "none")
    if not a.write or errs or missing:
        print(json.dumps(d, indent=1, ensure_ascii=False)[:1500])
        return 1 if (errs or missing) else 0
    path, h = prereg3.write_prereg(d)
    print(f"registered {rel_to_repo(path)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
