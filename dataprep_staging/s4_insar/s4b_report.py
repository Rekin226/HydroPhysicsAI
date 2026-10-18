"""WP S4b: write ``results/dataprep/reports/S4b.md`` and ``S4b.json`` from the evidence files
(test_*.json, after.json, explore_ramp.json, the product manifest and the LiCSBAS run info).

    python -I s4b_report.py      (hydro env)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import s4b_config as C  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

REP = DATAPREP / "reports"
PROD = DATAPREP / C.PRODUCT / "v1"


def j(p: Path):
    return json.loads(p.read_text())


def run_info(name: str) -> dict:
    ts = C.RUN_DIR / name / "TS_GEOCml2" / "info"
    p13 = (ts / "13parameters.txt").read_text()
    p15 = (ts / "15parameters.txt").read_text()

    def num(key, txt):
        m = re.search(rf"^{key}:\s*([0-9.]+)", txt, re.M)
        return int(float(m.group(1))) if m else None
    removed = sorted(set((ts / "11removed_image.txt").read_text().split()
                         + (ts / "12removed_image.txt").read_text().split()))
    kept = re.search(r"Kept pixels\s*:\s*(\d+)/(\d+)", p15)
    return {"n_im_all": num("n_im_all", p13), "n_im": num("n_im", p13),
            "n_ifg_all": num("n_ifg_all", p13), "n_ifg": num("n_ifg", p13),
            "n_ifg_bad": num("n_ifg_bad", p13), "removed_epochs": removed,
            "kept_pixels": [int(kept.group(1)), int(kept.group(2))] if kept else None}


def f1(x):
    return "n/a" if x is None else f"{x:.1f}"


def main() -> int:
    ev = C.EVID
    g1, c1 = j(ev / "test_gnss.json"), j(ev / "test_coherence.json")
    y = j(ev / "test_seasonal.json")
    af, ex = j(ev / "after.json"), j(ev / "explore_ramp.json")
    cov = j(PROD / "zone_coverage.json")["time_series_coverage_runF"]
    runs = {k: run_info(k) for k in ("P", "F", "D")}
    prereg = j(DATAPREP / "preregistration" / f"{C.PREREG_ID}.json")
    psha = [json.loads(x)["sha256"] for x in
            (DATAPREP / "preregistration" / "INDEX.jsonl").read_text().splitlines()
            if x.strip() and json.loads(x).get("id") == C.PREREG_ID][0]
    pr = g1["primary"]
    zones = c1["zones"]
    y1, y2 = y["Y1_all_yunlin"], y["Y2_battery"]
    ro = g1["report_only"]
    dec = af["decomp_2025"]
    ngP, ngF = af["ngl_onfan_P"], af["ngl_onfan_F"]

    acceptance = [
        {"criterion": "G1 (gate, plan S4 acceptance): WRA GNSS leave-one-station-out 14-month "
                      f"change RMS, pre-origin run P ({C.GNSS_T0} -> {C.GNSS_T1})",
         "threshold": f"<= {C.GNSS_LOSO_RMS_MAX_MM} mm with >= {C.GNSS_MIN_STATIONS} stations",
         "measured": f"{pr['rms_mm']:.1f} mm over {pr['n_evaluable']} stations (mean "
                     f"{pr['mean_mm']:.1f}, max |err| {pr['max_abs_mm']:.1f} mm)",
         "passed": bool(g1["passed"])}]
    for z, v in zones.items():
        acceptance.append({
            "criterion": f"C1 (plan S4 acceptance): Changhua {z.split('.')[-1]} summer "
                         "coherence coverage, 57 pre-origin summer pairs",
            "threshold": ">= 0.30 of the zone's cells with coherence >= 0.3 (else 'not "
                         "covered')",
            "measured": f"{v['median_frac_coherent']:.3f} ({v['n_cells']} cells; "
                        f"{v['median_frac_cells_with_data']:.2f} of them with data)",
            "passed": bool(v["covered"])})
    acceptance += [
        {"criterion": "Y1 (plan S4 step 8): seasonal amplitude vs census density 5 km, all "
                      "Yunlin cells, run P",
         "threshold": "5 km-block bootstrap p05 of Spearman rho > 0",
         "measured": f"rho {y1['rho_primary']:.3f}, p05 {y1['boot_p05']:.3f}, "
                     f"{y1['cells']['n']} cells",
         "passed": bool(y1["passed"])},
        {"criterion": "Y2 (plan S4 step 8): beat the placebo (15-20 km) and both thickness "
                      "proxies under the P3 transfer battery",
         "threshold": f">= {C.BATTERY_MIN_SPLITS} of 8 splits",
         "measured": f"{y2['wins']} of 8 (all-Yunlin rho: density 5 km "
                     f"{y1['cells']['dens_5km']:.3f}, placebo {y1['cells']['placebo_ring']:.3f}, "
                     f"Ske {y1['cells']['thick_ske_zonal']:.3f}, west distance "
                     f"{y1['cells']['thick_west_of_pinchout_km']:.3f})",
         "passed": bool(y2["passed"])},
        {"criterion": "Asc/desc decomposition from 2022-11 (plan S4 step 6)",
         "threshold": "only if S4a fetched a second geometry from 2022-11",
         "measured": "not possible: 069A_06632 not fetched, 105D has GeoTIFFs only for "
                     f"2025-01..2026-01; 2025 decomposition done after the fact (median "
                     "vertical-equivalent minus vertical "
                     f"{dec['median_vert_eq_minus_up_mm_yr']:.1f} mm/yr)",
         "passed": False},
        {"criterion": "Processing: LiCSBAS in the licsbas env, no atmospheric correction "
                      "(D11), stable off-fan reference, offset-only ties to validators",
         "threshold": "runs P, F, D exit 0; no GNSS/ring/pile/leveling value in the product",
         "measured": f"P {runs['P']['n_ifg']}/{runs['P']['n_ifg_all']} ifgs, "
                     f"{runs['P']['n_im']} epochs; F {runs['F']['n_ifg']}/"
                     f"{runs['F']['n_ifg_all']}, {runs['F']['n_im']}; D {runs['D']['n_ifg']}/"
                     f"{runs['D']['n_ifg_all']}, {runs['D']['n_im']}; all steps rc 0",
         "passed": True},
    ]
    status = "partial"
    lines = [
        "# WP S4b: InSAR processing (LiCSBAS), off-fan referencing, GNSS check, Changhua "
        "coherence and the seasonal-amplitude test (report, 2026-10-14)", "",
        f"Status: **{status}**. The product `results/dataprep/{C.PRODUCT}/v1/` was built from "
        "the LiCSAR frame 069A_06694_131402. The plan's numeric acceptance failed: G1 measured "
        f"{pr['rms_mm']:.1f} mm against the 10 mm threshold, on the pre-origin run. The "
        "product gate is therefore failed, and the loader guard refuses the `validator` use "
        "until the user decides. `after_the_fact_analysis` stays open.", "",
        "Pre-registration: `S4b.insar_acceptance.v1`, written by harness.prereg3 at "
        f"{prereg['created']} (sha256 `{psha[:16]}...`). It was written before any test "
        "statistic existed; the three tests were then scored on the pre-origin run P.", "",
        "## Results", "",
        f"- **G1 failed.** {pr['n_evaluable']} on-fan WRA stations were evaluable, with "
        f"errors from {min(r['err_mm'] for r in pr['rows']):.1f} to "
        f"{max(r['err_mm'] for r in pr['rows']):.1f} mm. The RMS was {pr['rms_mm']:.1f} mm "
        "against the 10 mm threshold. The report-only rows on the same run and window also "
        "fail:",
        f"  - steps corrected by the catalogue size (TKJS and SJES included): "
        f"{ro['steps_corrected']['rms_mm']:.1f} mm over {ro['steps_corrected']['n_evaluable']} "
        "stations;",
        f"  - offset + height reference (no plane): {ro['ref_offset_hgt']['rms_mm']:.1f} mm;",
        f"  - offset-only reference: {ro['ref_offset']['rms_mm']:.1f} mm;",
        f"  - constant U 0.833: {ro['u_constant_0.833']['rms_mm']:.1f} mm;",
        f"  - no GNSS tie: {ro['tie_none']['rms_mm']:.1f} mm, mean {ro['tie_none']['mean_mm']:.1f}"
        " mm (InSAR, referenced to the hills, subsides more than GNSS).",
        "- **C1 per zone.** " + "; ".join(
            f"{k.split('.')[-1]}: {v['median_frac_coherent']:.3f} -> "
            f"{'covered' if v['covered'] else 'NOT covered'}" for k, v in zones.items()) + ".",
        f"  - Z2 is the river-border zone; it misses the threshold by "
        f"{0.30 - zones['dt.zone.Z2']['median_frac_coherent']:.3f}. Its per-year medians are "
        + ", ".join(f"{yy} {vv:.3f}" for yy, vv in zones['dt.zone.Z2']['per_year_median'].items())
        + ".",
        f"  - Z3 passes even though only {zones['dt.zone.Z3']['median_frac_cells_with_data']:.2f}"
        " of its cells have data. The northern band of Changhua lies outside the primary "
        "footprint, and the denominator counts those cells as not coherent.",
        f"  - Report only, over every pre-origin pair (all seasons, 516 pairs): Z1 "
        f"{af['c1_all_pairs']['dt.zone.Z1']['median_frac_coherent']:.3f}, Z2 "
        f"{af['c1_all_pairs']['dt.zone.Z2']['median_frac_coherent']:.3f}, Z3 "
        f"{af['c1_all_pairs']['dt.zone.Z3']['median_frac_coherent']:.3f}. The paddy "
        "summer is what drives Z2 below the threshold.",
        f"- **Y failed: the seasonal amplitude is not a pumping validator.** Y1 passes (rho "
        f"{y1['rho_primary']:.3f}, p05 {y1['boot_p05']:.3f}), but Y2 wins {y2['wins']} of 8 "
        "splits. The thickness proxies correlate with the amplitude more strongly than "
        "census density does, both on all of Yunlin (Ske "
        f"{y1['cells']['thick_ske_zonal']:.3f}, distance west of the pinch-out "
        f"{y1['cells']['thick_west_of_pinchout_km']:.3f}) and on every split. The placebo "
        "density 15-20 km away also beats local density "
        f"({y1['cells']['placebo_ring']:.3f} against {y1['cells']['dens_5km']:.3f}). So the "
        "amplitude is not used inside Changhua, as the plan provides.", "",
        "## After the fact (registered report-only rows; not a gate, U7)", "",
        f"- **Run F** (full 2020-01..2025-12 network), same 18 stations and pre-origin window: "
        f"G1 RMS {af['F_g1']['loso']['rms_mm']:.1f} mm. Over the pilot's post-origin window "
        f"(2021-12-05 -> 2023-02-20) the RMS is {af['pilot_window_F']['loso']['rms_mm']:.1f} mm; "
        "the scout pilot measured 8.2 mm there.",
        "  - The pre-origin run P ends at 2022-12. Its second change window (2022-11-30 +- 30 "
        "d) therefore lies on the network's last epochs, where SBAS has the fewest "
        "connections. This end effect is the most likely reason for the gap between P "
        "(13.2 mm) and F. It is a weakness of the registered design that was not foreseen; "
        "the threshold was not changed.",
        f"  - Between P and F, the cell change 2020-01-12 -> {af['p_vs_f']['last']} differs by "
        f"{af['p_vs_f']['rms_diff_mm']:.1f} mm RMS (median {af['p_vs_f']['median_diff_mm']:.1f}"
        f" mm, correlation {af['p_vs_f']['corr']:.3f}).",
        f"- **NGL on-fan stations** with the WRA tie: RMS {f1(ngP['rms_mm'])} mm (P) and "
        f"{f1(ngF['rms_mm'])} mm (F), over n = {ngP['n']} stations (HUWE has no epoch with "
        "both values in the windows).",
        "  - Per-station errors in P: " + ", ".join(
            f"{s} {v['err_mm']:.1f}" for s, v in ngP['stations'].items()
            if v.get('err_mm') is not None) + " mm.",
        "  - The Changhua-side stations FUNY and PUS2 come out at 4.5 and 15.9 mm.",
        f"- **2025 ascending/descending decomposition** ({dec['n_cells']} cells, 2025-01-23.."
        f"2025-12-11). The vertical-equivalent LOS/U minus the decomposed vertical rate has "
        f"median {dec['median_vert_eq_minus_up_mm_yr']:.1f} mm/yr and RMS "
        f"{dec['rms_vert_eq_minus_up_mm_yr']:.1f} mm/yr; the correlation is "
        f"{dec['corr_vert_eq_up']:.3f}.",
        f"  - The east rate relative to the stable area has median "
        f"{dec['median_east_mm_yr']:.1f} mm/yr (p05..p95 "
        f"{dec['p05_p95_east_mm_yr'][0]:.1f}..{dec['p05_p95_east_mm_yr'][1]:.1f}). This is "
        "the plan's east-west leakage, measured in 2025 only.", "",
        "## Exploratory (unregistered, after the gate failed)", "",
        "This section asks whether the plane in the reference biases the fan. It compares, "
        "for each WRA station, the 2020-01..2025-03 rate of InSAR minus GNSS, using run F and "
        "17 stations (steps and the plan's dropped stations excluded).",
        f"- With the registered plane reference, the difference has a slope of "
        f"{ex['plane']['slope_east_mm_yr_per_km']:.2f} mm/yr per km of easting and a mean of "
        f"{ex['plane']['mean_diff_rate_mm_yr']:.1f} mm/yr.",
        f"- With the offset + height reference, the slope is "
        f"{ex['offset_hgt']['slope_east_mm_yr_per_km']:.2f} mm/yr per km and the mean "
        f"{ex['offset_hgt']['mean_diff_rate_mm_yr']:.1f} mm/yr.",
        f"- Both references have a residual sd of about {ex['plane']['resid_sd_mm_yr']:.1f} "
        f"mm/yr and a rate correlation of {ex['plane']['corr_insar_gnss_rate']:.2f}.",
        "- The plane fitted on the stable pixels trends steadily: its east coefficient grows by "
        "about +0.5 mm/km/yr in LOS. Those pixels lie only east of the fan (x about 196-223 "
        "km), and the fan reaches 50 km further west, so the plane can map tectonic or "
        "atmospheric gradients of the foothills onto the western fan.",
        "- The registered product keeps the plane. Long-term rates in the west of the fan "
        "should therefore be read with this bias in mind; a v2 needs a new registration.", "",
        "## What was done", "",
        "1. **Resume check.** No S4b work existed: no code, product, pre-registration, ledger "
        "fragment or look. S4a's hand-over was used as delivered: insar_licsar_subsets/v1, "
        "notes_for_S4b.json and the licsbas env.",
        "2. **LiCSBAS runs.** Run in the licsbas env inside one CPU slot, with "
        "`s4b_run_licsbas.sh` and `--n_para 1`. The steps were 02 (2 x 2 multilook), 11, 12, "
        "13, 14 and 15, all at LiCSBAS defaults (commit a145e6c4217b). There was no "
        "atmospheric step (D11), and step 16 was not used. Each run worked in a symlinked "
        "work dir; L0 was never written.",
        f"   - **P** (pairs <= 2022-12-31, the gate run): {runs['P']['n_ifg']} of "
        f"{runs['P']['n_ifg_all']} ifgs used, {runs['P']['n_ifg_bad']} rejected by "
        f"11/12, {runs['P']['n_im']} of {runs['P']['n_im_all']} epochs. Removed epochs: "
        f"{', '.join(runs['P']['removed_epochs'])}. The step 15 mask keeps "
        f"{runs['P']['kept_pixels'][0]}/{runs['P']['kept_pixels'][1]} pixels.",
        f"   - **F** (2020-01..2025-12, the product run, without the 4 empty 20260104 pairs): "
        f"{runs['F']['n_ifg']} of {runs['F']['n_ifg_all']} ifgs, {runs['F']['n_im']} of "
        f"{runs['F']['n_im_all']} epochs. The step 15 mask keeps "
        f"{runs['F']['kept_pixels'][0]}/{runs['F']['kept_pixels'][1]} pixels.",
        f"   - **D** (descending 105D, 2025-01..2026-01): {runs['D']['n_ifg']} of "
        f"{runs['D']['n_ifg_all']} ifgs, {runs['D']['n_im']} epochs.",
        "   - Wall times were about 8 min (P), 20 min (F) and 1.5 min (D).",
        "   - LiCSBAS12 rejected about a third of the ifgs on loop closure, and it removed "
        "summer epochs (paddy decorrelation).",
        "3. **Georeference.** Pixels are placed by index on S4a's window geotransform. "
        "LiCSBAS02 took the stretched geotransform of the first pair, as S4a warned; "
        "cum.h5's geotransform is ignored. A multilooked pixel's centre is the centre of its "
        "2 x 2 native block.",
        "4. **Reference (plan step 5, stable areas only).**",
        "   - Stable pixels are land, LiCSAR DEM 100-600 m, at least 2.5 km from every active "
        "cell centre, and inside the LiCSBAS15 mask: 11,361 pixels by geometry, about "
        "3,400-3,800 after the mask.",
        "   - Per epoch, a robust two-pass LS fits offset + east/north plane + height term "
        "and subtracts it everywhere.",
        "   - S3's off-fan GNSS reference could not be used, because the gnss_clean gate "
        "failed. No on-fan GNSS, ring, pile or leveling value enters the product.",
        "   - The vertical-equivalent is LOS / U per pixel (U 0.80-0.85).",
        "5. **Twin cells.** The cell value is the median of the valid multilooked pixels "
        "(at least 5). sd = the cell standard error and the stable-pixel residual sd / U, in "
        "quadrature; it is labelled uncalibrated (plan 4.7). Flags: 0 observed, 8 missing, "
        "9 for an epoch with a stable set below 200 pixels (none occurred).",
        "6. **G1.** WRA L0 heights (raw vendor cache, 7-day rolling median) were used rather "
        "than gnss_clean, which is refused, together with the plan's step catalogue: TKJS "
        "and SJES have steps inside the span and were excluded.",
        "   - The other stations were selected by pre-origin data availability only. LNJS, "
        "MDES and YSLL have no heights in the second window; CYES starts in 2023.",
        "   - The tie is the per-epoch mean (InSAR - GNSS) of the other stations: an offset "
        "only, with no rate or plane.",
        "7. **C1.** Native-resolution cc of the 57 summer pairs (both epochs in June-September "
        "of 2020-2022, btemp <= 24 d), reduced to the per-cell median coherence.",
        "8. **Y.** The P3 battery was rebuilt from the design pilot `a2_robust_transfer.py` "
        "(1091 cells, as recorded) because P3 has not built its product yet.",
        "   - Census density comes from pump_electricity/preorigin/v1 (2020-22, read through "
        "the guard as `validator`).",
        "   - The thickness proxies come from the model's layer geometry: zonal Ske of the "
        "leveling-calibrated VEP column, and distance west of the 205 km aquitard pinch-out.",
        "9. **Product, manifest, ledger.**",
        f"   - Product: `{C.PRODUCT}/v1`, layer L2, fit_window_end 2025-12, gate failed. The "
        "seasonal amplitude and coherence outputs carry their own fit_window_end (2022-12 and "
        "2022-09).",
        "   - Allowed uses: validator and after_the_fact_analysis. Every pre-origin use, "
        "scoring_target and the forcing uses are forbidden.",
        "   - Guard probe: `validator` raises GateNotPassed; `after_the_fact_analysis` "
        "opens; calibration_target, design_time_evidence, scoring_target and selection raise "
        "UseNotAllowed.",
        "   - Checker: the product manifest is ok (73 files), and every licsar SOURCE.json is "
        "ok.",
        "", "## Product coverage (run F, cells with >= 5 valid pixels)", "",
        "; ".join(f"{k}: {v:.2f}" for k, v in cov.items()) + ".", "",
        "## Acceptance", "", "| criterion | threshold | measured | pass |", "|---|---|---|---|"]
    for a in acceptance:
        lines.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                     f"{'yes' if a['passed'] else '**no**'} |")
    deviations = [
        "S3's gnss_clean (gate failed twice) could not supply the off-fan GNSS reference or "
        "the G1 GNSS series. The reference uses stable areas only, which the plan allows, and "
        "G1 uses the L0 WRA heights with the plan's step catalogue.",
        "G1 is scored in a fully pre-origin window (2021-09-30 -> 2022-11-30) on a run built "
        "from pre-origin pairs only (U7), rather than in the pilot's 2021-12 -> 2023-02 "
        "window. The pilot's window is reported after the fact.",
        "The asc/desc decomposition from 2022-11 is not available, because S4a did not fetch "
        "it. A 2025-only decomposition is reported after the fact.",
        "The plan validation rows on leveling 2020-21, rings/piles 2023-25, printed piles and "
        "the bridged-ring gap correlation were not run here. They were not in the S4b task, "
        "and the gap correlation is S1b's.",
        "The stable-area height cap (600 m) and the 2.5 km distance were fixed by S4b from the "
        "geometry alone, before any displacement was read.",
        "The Y rules are this registration's reading of plan step 8: Spearman correlation, a "
        "5 km-block bootstrap, two thickness proxies, and the P3 battery's 6-of-8 count. The "
        "battery was rebuilt from the design pilot because P3 is not built.",
        "Code lives in dataprep_staging/s4_insar/ with the prefix s4b_. S4a's files are "
        "unchanged, because they are hashed by insar_licsar_subsets/v1.",
        "Undisclosed development looks were logged after the fact (dev_D_struct, "
        "s4b_looks2.py). The "
        "structure tests of s4b_evaluate on run D (2025) computed, without printing, GNSS, "
        "census and zone statistics on 2025 data. That data is post-origin and disjoint from "
        "every registered window."]
    issues = [
        "Gate failed (G1 13.2 mm > 10 mm on run P), so the guard refuses insar_069A/v1 as a "
        "validator. S1b (InSAR gap check), S5 and the twin's 2022-26 InSAR validation are "
        "blocked until the user decides between: (a) accepting run F's after-the-fact 9.9 mm, "
        "which rests on 2023+ interferograms and so needs a U7 waiver for a validator-only "
        "product; (b) a new registration with a pre-origin window away from the network end, "
        "for example 2020-02 -> 2021-04 on run P, which no look has touched; (c) keeping "
        "InSAR as after-the-fact analysis only.",
        "The run-P end effect is the likely cause of the failure: P's second window sits on "
        "its last epochs. Any re-test must keep both windows away from a network edge.",
        "Reference ramp: exploratory (unregistered) rates show an InSAR-minus-GNSS slope of "
        "-0.50 mm/yr per km of easting with the registered plane, against +0.15 with offset + "
        "height. The plane fitted on the eastern hills drifts by about 0.5 mm/km/yr in LOS. "
        "Long-term rates in the western fan carry this bias; a v2 with an offset + height "
        "reference (or GNSS-informed ramps) needs a user decision and a new registration.",
        "C1: Z2 (river-border zone, 300 cells) is NOT covered in summer (0.297). Under the "
        "plan, InSAR is not a validator there for summer epochs. Z3 passes, but a third of "
        "its cells have no data (outside the primary footprint).",
        "Y failed: the seasonal amplitude tracks the model's compressible-thickness proxies "
        "(rho 0.66) and the 15-20 km placebo (0.41) more than local census density (0.30). "
        "It must not be used as a Changhua rank validator for P3 or P4m.",
        "Vertical-equivalent: in 2025 the east-west leakage biases LOS/U by a median -2.9 "
        "mm/yr (RMS 3.8 mm/yr) against the asc/desc decomposition. Before 2025 only the "
        "ascending geometry exists.",
        "Fetching 069A_06632_162010 (about 1.09 GB, full-fan ascending from 2022-11) still "
        "needs a user decision. With 105D it would not give a 2022-11 decomposition either, "
        "since 105D has GeoTIFFs only from 2025-01.",
        "The checker run of 2026-10-14 also reports failures that are not S4b's: "
        "ground_elevation v1/v2 and heads/L1_clean v2 (code hashes changed), and the TPC "
        "census SOURCE.json (13 files with missing hashes). These belong to W3, H1 and the "
        "main session.",
        "Before committing dataprep_staging, run the CLAUDE.md Rule 1 scan. s4b_ files hold "
        "no coordinates or data. The model zone boundaries 205 and 182 km already exist in "
        "hydrophysics/twin/zones.py."]
    lines += ["", "## Deviations from the plan", ""] + [f"- {d}" for d in deviations]
    lines += ["", "## Issues for downstream", ""] + [f"- {i}" for i in issues]
    lines += ["", "## Looks logged (A1 look log, harness.prereg3)", "",
              "- development: run D 2025 (19:23Z), plus the after-the-fact disclosure "
              "dev_D_struct (s4b_looks2.py; s4b_looks.py stays as the product hashed it).",
              "- score: G1, C1 (preceded by the extraction look) and Y.",
              "- after_the_fact: run F pre-origin, the pilot window, NGL on-fan, C1 over all "
              "pairs, the 2025 decomposition.",
              "- exploratory_after_the_fact: the ramp diagnostic (s4b_looks2.py "
              "explore_ramp).",
              "- The registration disclosed 50 unverifiable legacy looks: other WPs' units of "
              "unknown membership (H1a heads 23, W1 briefing 6, S2 piles 6, S3 GNSS "
              "injections 6, P1 census 3, W2W4 rain 3, A1 simulation 3). None is an InSAR "
              "displacement or coherence look.", "",
              "## Ledger", "",
              "`results/dataprep/ledger/rows/S4b.csv`: 388 rows over the reported numbers. "
              "Roles: LiCSAR L0, run files, the S4a product and the units registry are "
              "inputs; WRA and NGL GNSS files are validators of the InSAR numbers; the census "
              "product is the validator of the amplitude test. Nothing is unledgered.", "",
              "## Resources", "",
              "- Claude API tokens: 0. GPU: none. New downloads: none (0 bytes).",
              "- CPU: LiCSBAS and every job longer than a minute ran through "
              "tools/cpu_slot.sh, at most 2 slots at once.",
              "- Disk: s4b_run about 1.4 GB (gitignored); product 39 MB.",
              "- Nothing was written under hydrophysics/, tests/ or L0, and nothing was "
              "staged in git."]
    (REP / "S4b.md").write_text("\n".join(lines) + "\n")
    rep = {"wp": "S4b", "status": status, "acceptance": acceptance,
           "products": [{"path": f"results/dataprep/{C.PRODUCT}/v1/", "layer": "L2",
                         "manifest": f"results/dataprep/{C.PRODUCT}/v1/MANIFEST.json",
                         "allowed_uses": "validator (refused while the gate fails), "
                                         "after_the_fact_analysis",
                         "fit_window_end": "2025-12"}],
           "deviations_from_plan": deviations, "issues_for_downstream": issues,
           "tests": {"G1": g1["passed"], "C1": {k: v["covered"] for k, v in zones.items()},
                     "Y": y["passed"]},
           "after_the_fact": {"F_g1_rms_mm": af["F_g1"]["loso"]["rms_mm"],
                              "pilot_window_rms_mm": af["pilot_window_F"]["loso"]["rms_mm"],
                              "ngl_P_rms_mm": ngP["rms_mm"], "ngl_F_rms_mm": ngF["rms_mm"],
                              "p_vs_f_rms_mm": af["p_vs_f"]["rms_diff_mm"],
                              "decomp_2025": dec},
           "exploratory": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                           for k, v in ex.items()},
           "licsbas_runs": runs, "coverage_runF": cov, "claude_api_tokens": 0, "gpu": "none"}
    (REP / "S4b.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("wrote", REP / "S4b.md", REP / "S4b.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
