"""WP S4b repair round: rewrite ``results/dataprep/reports/S4b.md`` and ``S4b.json`` as the WP
now stands (round-1 copies are kept in ``S4b_evidence/round1/``).

    python -I s4b_r2_report.py      (hydro env)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import s4b_config as C  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP  # noqa: E402

REP = DATAPREP / "reports"
V2 = DATAPREP / C.PRODUCT / "v2"
EV = C.EVID


def j(p: Path):
    return json.loads(p.read_text())


def main() -> int:
    g1, c1, y = j(EV / "test_gnss.json"), j(EV / "test_coherence.json"), j(EV / "test_seasonal.json")
    af, ex = j(EV / "after.json"), j(EV / "explore_ramp.json")
    gp, dg, sp = j(EV / "repair_gaps.json"), j(EV / "repair_diag.json"), j(EV / "repair_pf_split.json")
    ns = j(V2 / "network_segments.json")
    cov = j(V2 / "zone_coverage.json")["time_series_coverage_runF"]
    r1 = j(EV / "round1" / "S4b_round1.json")
    msha = sha256_file(V2 / "MANIFEST.json")
    with open(DATAPREP / "ledger" / "rows" / "S4b.csv", newline="", encoding="utf-8") as f:
        led = list(csv.DictReader(f))
    n_led, n_num = len(led), len({r["number_id"] for r in led})
    looks = [json.loads(x) for x in (DATAPREP / "preregistration" / "looks.jsonl").read_text()
             .splitlines() if x.strip()]
    s4 = [lk for lk in looks if lk.get("wp") == "S4b"]
    rep_looks = [lk for lk in s4 if lk["time"] >= "2026-10-13T20:00"]
    pr = g1["primary"]
    ro = g1["report_only"]
    errs = [r["err_mm"] for r in pr["rows"] if r.get("evaluable")]
    z = c1["zones"]
    yrs = {}
    for q in c1["pairs"]:
        yrs[int(str(q)[:4])] = yrs.get(int(str(q)[:4]), 0) + 1
    y1, y2 = y["Y1_all_yunlin"], y["Y2_battery"]
    pf, pcs, rc, yc = dg["pf"], dg["pieces"], dg["refcoef"], dg["yconf"]
    fc = ns["flag_counts_vertical_cells"]
    tot = sum(fc.values())
    gF = gp["F"]["gaps"]
    gP = gp["P"]["gaps"]
    exp, exo = ex["plane"], ex["offset_hgt"]
    dec = af["decomp_2025"]

    acceptance = [a for a in r1["acceptance"] if not a["criterion"].startswith("Processing")]
    acceptance[0]["measured"] += ("; the window spans both 2022 network gaps of run P "
                                  f"({', '.join(a + '/' + b for a, b in gP)}), whose "
                                  "increments LiCSBAS bridges with its linear-velocity "
                                  "constraint")
    for a in acceptance:
        for zid, v in z.items():
            if a["criterion"].startswith(f"C1 (plan S4 acceptance): Changhua {zid.split('.')[-1]}"):
                a["measured"] += (f"; per-year medians 2020 {v['per_year_median']['2020']:.3f}, "
                                  f"2021 {v['per_year_median']['2021']:.3f}, 2022 "
                                  f"{v['per_year_median']['2022']:.3f}")
    acceptance += [
        {"criterion": "Processing: LiCSBAS in the licsbas env, no atmospheric correction (D11), "
                      "stable off-fan reference, offset-only ties to validators",
         "threshold": "runs P, F, D exit 0; no GNSS/ring/pile/leveling value in the product",
         "measured": "P 345/516 ifgs, 108 epochs; F 597/925, 192; D 95/108, 28; all steps rc 0. "
                     "v1 shipped gnss_tie_offsets.csv (per-epoch InSAR - WRA GNSS over the 18 "
                     "G1 stations), so v1 did NOT meet the second clause; v2 drops it and holds "
                     "no GNSS, ring, pile or leveling value",
         "passed": True},
        {"criterion": "Repair: product flags reflect what is observed (plan 2.4, U1): values "
                      "whose offset includes a LiCSBAS-bridged gap increment",
         "threshold": "non-zero flag and segment metadata for every such value",
         "measured": f"v2: flag 4 on {fc.get('4', 0):,} of {tot - fc.get('8', 0):,} finite "
                     f"cell-epochs ({100 * fc.get('4', 0) / (tot - fc.get('8', 0)):.1f} %), "
                     "segment ids per cell-epoch and per pixel, network_segments.json (v1 "
                     "flagged every value 0)",
         "passed": True},
        {"criterion": "Repair: fit_window_end covers every input",
         "threshold": "last month of any input's data",
         "measured": "v2 2026-01 (run D pairs to 2026-01-30; decomposition_2025.npz carries "
                     "2026-01, vertical arrays 2025-12); v1's 2025-12 was wrong",
         "passed": True},
        {"criterion": "Process (A1): every look at test data logged before it is computed",
         "threshold": "no unlogged or late-logged look",
         "measured": "round 1 failed it: the run-D structure tests (19:27-19:31Z) were logged "
                     "at 19:50Z, and the v1 build's gnss_tie_offsets computation (about "
                     "19:47-19:49Z) was never logged; both are now disclosed (the latter at "
                     "20:16:16Z); the repair's own looks were logged before their scripts ran",
         "passed": False},
        {"criterion": "Plan S4 validation table: leveling 2020-21 withheld, rings and piles "
                      "2023-25, printed piles after 2025-05, bridged-ring gap correlation",
         "threshold": "run (the plan sets no threshold)",
         "measured": "not run: outside the S4b task text (the gap correlation belongs to S1b); "
                     "the plan's validation table is incomplete",
         "passed": False},
    ]

    head = [
        "# WP S4b: InSAR processing (LiCSBAS), off-fan referencing, GNSS check, Changhua "
        "coherence and the seasonal-amplitude test (report, 2026-10-14, with repair round)", "",
        "Status: **partial**. The current product is `results/dataprep/insar_069A/v2/` "
        f"(MANIFEST sha256 `{msha[:16]}...`), which supersedes v1. The plan's numeric acceptance "
        f"failed: G1 measured {pr['rms_mm']:.1f} mm against 10 mm on the pre-origin run P. The "
        "gate is therefore failed and the loader guard refuses the `validator` use until the "
        "user decides; `after_the_fact_analysis` stays open.", "",
        "Pre-registration: `S4b.insar_acceptance.v1` (harness.prereg3, 2026-10-13T19:33:12Z, "
        "sha256 `f03377feacd4efb7...`), written before any test statistic existed. The repair "
        "round re-scored nothing and changed no threshold.", "",
        "## Repair round (2026-10-14)", "",
        "An independent verifier found two major and ten minor issues, and no critical one. "
        "All are accepted; one count is corrected (census flag 9). What changed:", "",
        "- **insar_069A/v2** was built from v1's values, which are unchanged, plus LiCSBAS's "
        "network-gap structure:",
        "  - flag 4 marks values that include a gap-bridged increment;",
        "  - segment ids are added per cell-epoch and per pixel, and network_segments.json "
        "is new;",
        "  - gnss_tie_offsets.csv is removed;",
        "  - fit_window_end is corrected to 2026-01;",
        "  - the seasonal-amplitude status no longer offers a physical reading.",
        "  v1 stays published, and immutable, as superseded.",
        "- **The G1 failure explanation is corrected.** The round-1 'SBAS end effect' is "
        "withdrawn. Run P differs from run F at the bridged summer-2022 gaps, not at its end. "
        "See below.",
        "- **The downstream options are rewritten.** Option (b) is not clean: the WRA "
        "stations were looked at over 2020-01..2025-12.",
        "- **The look log is completed.**",
        "  - Late disclosure of the v1 build's tie-offset computation.",
        "  - The verifier's looks, logged on its behalf.",
        "  - Two repair looks, logged before their scripts ran.",
        "  - The round-1 claim 'all looks are logged' is withdrawn.",
        "- **Interpretations are corrected.**",
        "  - Y: the physical reading is withdrawn.",
        "  - The ramp now has its north component.",
        "  - C1 gets per-year medians.",
        "  - The ledger count is fixed (387 rows after round 1).", "",
        "### Response to each verifier finding", "",
        "| finding | verdict | action and evidence |", "|---|---|---|",
        "| MAJOR: network gaps not reported or flagged; G1 'end effect' explanation wrong | "
        "accepted | v2 adds flag 4 and segment ids. "
        f"F gaps: {', '.join(a + '/' + b for a, b in gF)}; P gaps: "
        f"{', '.join(a + '/' + b for a, b in gP)}. Every active-cell pixel has n_gap >= "
        f"{gp['P']['pixel_n_gap']['min']:.0f} (P) and >= {gp['F']['pixel_n_gap']['min']:.0f} "
        "(F). The explanation is replaced by the gap analysis below. |",
        "| MAJOR: option (b) not untouched; 'all looks are logged' false | accepted | "
        "gnss_tie_offsets computation disclosed (look 2026-10-13T20:16:16Z). Option (b) is "
        "rewritten (issues below). The claim is withdrawn, and the file is removed from v2. |",
        "| Y2 interpretation unsupported | accepted | Withdrawn. Reproduced: rho(amplitude, "
        f"easting) {yc['rho_amp_easting']:.3f}; rho(census density 5 km, easting) "
        f"{yc['rho_density5_easting']:.3f}; run-P plane east coefficient annual amplitude "
        f"{rc['P']['east_km']['annual_amp']:.3f} mm/km LOS, i.e. "
        f"{rc['P_east_annual_amp_los_mm_at_30_50km'][0]:.1f}-"
        f"{rc['P_east_annual_amp_los_mm_at_30_50km'][1]:.1f} mm at 30-50 km west of the "
        f"stable-pixel centroid (x {rc['stable_centroid_x_km']:.1f} km, the centroid the "
        "design uses). |",
        "| Ramp summarised incompletely (north) | accepted | Plane InSAR-GNSS rate slope north "
        f"{exp['slope_north_mm_yr_per_km']:+.2f} mm/yr/km (offset + height "
        f"{exo['slope_north_mm_yr_per_km']:+.2f}). Run-F plane drifts east "
        f"{rc['F']['east_km']['rate_per_yr']:+.2f}, north "
        f"{rc['F']['north_km']['rate_per_yr']:+.2f} mm/km/yr (LOS; harmonic fit). |",
        "| fit_window_end 2025-12 | accepted | v2: 2026-01. |",
        "| gnss_tie_offsets.csv inside a validator product | accepted | Removed in v2. The "
        "round-1 processing row was wrong for v1 and is corrected. |",
        "| late, inaccurate development-look disclosure | accepted (process defect) | "
        "Recorded in the acceptance table (process row, fail). The log is append-only and "
        "the late entries stand. |",
        "| plan validation rows not run | accepted | Listed as a 'not run' acceptance row. |",
        "| C1 pooled median dominated by 2020 | accepted (reporting note) | 2022 alone: "
        + ", ".join(f"{k.split('.')[-1]} {v['per_year_median']['2022']:.3f}"
                    for k, v in z.items()) + " (Z1 and Z3 stay above 0.30; Z2 already "
        "fails). A reporting note; the registered rule stands. |",
        "| census density includes flag-9 cell-months | accepted, count corrected | "
        f"Recounted over the census years 2020-22: "
        f"{yc['census_flag9_cell_months_2020_22']['all_active']} flag-9 cell-months of "
        f"{yc['census_flag9_cell_months_2020_22']['of_all_active']:,} on the active cells "
        f"({yc['census_flag9_cell_months_2020_22']['yunlin']} in Yunlin). The verifier's 827 "
        "presumably counts a different span; negligible either way. |",
        "| ledger count 388 | accepted | Round 1 left 387 data rows. Now "
        f"{n_led} rows over {n_num} numbers, 0 conflicts. |",
        "| verifier disclosure | done | Its looks are logged on its behalf (2026-10-13T"
        "20:16:17Z, after_the_fact). |", "",
        "### Corrected analysis of the G1 failure (after the fact, report only)", "",
        "LiCSBAS13 runs NSBAS with gamma 1e-4. Where no valid interferogram spans an epoch "
        "interval, the increment is set by the pixel's linear-velocity constraint alone. "
        f"Run P has two such gaps inside the G1 span ({', '.join(a + '/' + b for a, b in gP)}). "
        "Run F has the same two, plus the 2023 and 2024 summer gaps.", "",
        "- **InSAR only, runs P vs F at active-cell pixels:**",
        f"  - across the 2022-08 gap: RMS {pf['across_gap2'][0]['rms_mm']:.1f} mm, median "
        f"{pf['across_gap2'][0]['median_mm']:+.1f} mm (F - P);",
        f"  - across the 2022-06 gap: RMS {pf['across_gap1'][0]['rms_mm']:.1f} mm;",
        f"  - within-segment intervals: median RMS {pf['within_rms_mm']['median']:.1f} mm, max "
        f"{pf['within_rms_mm']['max']:.1f} (n {pf['within_rms_mm']['n']}); 18-30 d intervals "
        f"max {pf['within_18_30d_rms_mm']['max']:.1f} mm;",
        f"  - P's last segment ({pf['last_segment']['span']}): "
        f"{pf['last_segment']['rms_mm']:.1f} mm;",
        f"  - the G1 change itself: {pf['g1_change']['rms_mm']:.1f} mm.",
        "- **The P-to-F change of the G1 station errors** "
        f"({sp['n_stations']} stations; HNES lacks the 2022-06-06 value). The GNSS term "
        "cancels, so this is InSAR only:",
        f"  - total change: RMS {sp['rms_err_change_mm']:.1f} mm;",
        f"  - cross-gap increments: {sp['rms_gap_part_change_mm']:.1f} mm, correlation "
        f"{sp['corr_err_change_gap_change']:.2f} with the total;",
        f"  - within-segment remainder: {sp['rms_within_segment_part_change_mm']:.1f} mm.",
        "- **Reading.** The registered G1 result depends on how LiCSBAS bridges the two "
        "summer-2022 gaps. Re-fitting the bridges with 2023-25 data moves G1 from "
        f"{pr['rms_mm']:.1f} to {af['F_g1']['loso']['rms_mm']:.1f} mm, while the "
        "interferometric increments barely move. The run-P end is not the cause.",
        "- **Limit.** The absolute 13.2 mm cannot be attributed to the bridges alone. "
        "Single-interval InSAR - GNSS increments within segments already have RMS "
        f"{pcs['P']['within_single_interval_rms_mm']['median']:.1f} mm (median over "
        f"{pcs['P']['within_single_interval_rms_mm']['n']} intervals, run P), so the "
        f"per-gap GNSS errors ({pcs['P']['gap1_rms_mm']:.1f} and "
        f"{pcs['P']['gap2_rms_mm']:.1f} mm, run P) are noise-dominated. "
        "Run F's 9.9 mm also rests on bridged offsets.", "",
        "### Network gaps in the product", "",
        f"- Run F: {len(gp['F']['segments'])} segments: " + "; ".join(
            f"{s['first']}-{s['last']} ({s['n_epochs']} epochs)" for s in gp["F"]["segments"])
        + ".",
        "  Every summer 2022-2024 has a gap. 2025 has none, and run D has none.",
        f"- {100 * (1 - gp['F']['frac_pixels_with_gaps_beyond_network']):.1f} % of the "
        "active-cell pixels have exactly the network gaps.",
        f"- {100 * gp['F']['cell_n_gaps_any_pixel']['frac_cells_equal_to_network']:.1f} % of "
        "the cells with data have no extra cell gap. A cell has a gap when any contributing "
        "pixel has one.",
        f"- v2 flags: 0 {100 * fc.get('0', 0) / tot:.1f} %, 4 {100 * fc.get('4', 0) / tot:.1f} %, "
        f"8 {100 * fc.get('8', 0) / tot:.1f} % of cell-epochs.",
        "- A change between two epochs is interferometric only when both share the "
        "segment id.",
        "- The 2025 decomposition cells have no run-F pixel gap inside 2025.", "",
        "### Looks logged this round (A1 look log, harness.prereg3)", ""]
    for lk in rep_looks:
        head.append(f"- {lk['time']}: {lk['purpose']}, `{lk['data_scope']}`, units "
                    f"{', '.join(lk['units'])}.")
    head += [
        "",
        "Under A1's membership rule, every S4b look at all active cells is eligible to block "
        "later registrations that use A1's 11 fresh spatial units. That covers round 1's "
        "dev_D, Y score, run-F pre-origin, decomposition and dev_D_struct looks, and this "
        "round's verifier, gap-structure and diagnostic looks. These looks contain no head "
        "data (see issues).", "",
        "## Results of the registered tests (pre-origin run P; unchanged)", "",
        f"- **G1 failed.** {pr['n_evaluable']} on-fan WRA stations were evaluable, with errors "
        f"from {min(errs):.1f} to {max(errs):.1f} mm and RMS {pr['rms_mm']:.1f} mm against 10 "
        "mm. Report-only rows "
        "on the same run and window:",
        f"  - steps corrected: {ro['steps_corrected']['rms_mm']:.1f} mm "
        f"({ro['steps_corrected']['n_evaluable']} stations);",
        f"  - offset + height reference: {ro['ref_offset_hgt']['rms_mm']:.1f} mm;",
        f"  - offset-only reference: {ro['ref_offset']['rms_mm']:.1f} mm;",
        f"  - constant U 0.833: {ro['u_constant_0.833']['rms_mm']:.1f} mm;",
        f"  - no GNSS tie: {ro['tie_none']['rms_mm']:.1f} mm (mean "
        f"{ro['tie_none']['mean_mm']:.1f}).",
        "  The window spans both 2022 gaps of run P (see the repair round).",
        "- **C1 per zone.**",
        "  - " + "; ".join(f"{k.split('.')[-1]} {v['median_frac_coherent']:.3f} -> "
                           f"{'covered' if v['covered'] else 'NOT covered'}"
                           for k, v in z.items()) + ".",
        "  - Pairs by year: " + ", ".join(f"{k} {n}" for k, n in yrs.items())
        + ". 2022 alone gives "
        + ", ".join(f"{k.split('.')[-1]} {v['per_year_median']['2022']:.3f}"
                    for k, v in z.items()) + ". 'Covered' may be optimistic for 2022+ summers.",
        "  - Z3 passes, but only 0.66 of its cells have data (the northern band lies outside "
        "the footprint).",
        "  - All pre-origin pairs (report only): Z1 0.741, Z2 0.433, Z3 0.462.",
        f"- **Y failed.** Y1 passes (rho {y1['rho_primary']:.3f}, p05 {y1['boot_p05']:.3f}), but "
        f"Y2 wins {y2['wins']} of 8 splits. The amplitude is not a validator in Changhua.",
        "  - Withdrawn: the round-1 reading that the amplitude 'tracks the thickness "
        "proxies'.",
        "  - Both proxies are functions of easting. The amplitude, census density and the "
        "reference plane's annual cycle all vary east-west, so the design cannot separate "
        "local pumping from east-west structure.",
        "  - Exploratory, after the fact: given easting, partial rho(amplitude, density) is "
        f"{yc['partial_rho_amp_density5_given_easting']:.2f}, and rho(amplitude, northing) is "
        f"{yc['rho_amp_northing']:.2f}.", "",
        "## After the fact (registered report-only rows; not a gate, U7)", "",
        f"- **Run F**, same 18 stations and window: {af['F_g1']['loso']['rms_mm']:.1f} mm.",
        f"  - The pilot's post-origin window (2021-12-05 -> 2023-02-20) gives "
        f"{af['pilot_window_F']['loso']['rms_mm']:.1f} mm; the pilot measured 8.2 mm.",
        "  - Both windows span the two 2022 gaps.",
        f"- **Run P vs F**, cell change 2020-01-12 -> 2022-12-27: RMS difference "
        f"{af['p_vs_f']['rms_diff_mm']:.1f} mm, correlation {af['p_vs_f']['corr']:.3f}.",
        f"- **NGL on-fan stations**, WRA tie: {af['ngl_onfan_P']['rms_mm']:.1f} mm (P), "
        f"{af['ngl_onfan_F']['rms_mm']:.1f} mm (F), n 4.",
        f"- **2025 asc/desc decomposition** ({dec['n_cells']} cells): vertical-equivalent "
        f"minus vertical, median {dec['median_vert_eq_minus_up_mm_yr']:.1f} mm/yr, RMS "
        f"{dec['rms_vert_eq_minus_up_mm_yr']:.1f}, correlation {dec['corr_vert_eq_up']:.3f}.",
        f"  - East rate median {dec['median_east_mm_yr']:.1f} mm/yr.", "",
        "## Exploratory (unregistered, after the gate failed)", "",
        "InSAR-minus-GNSS rate per WRA station, 2020-01..2025-03, run F, "
        f"{exp['n_stations']} stations:",
        f"- Plane reference: slopes {exp['slope_east_mm_yr_per_km']:+.2f} (east) and "
        f"{exp['slope_north_mm_yr_per_km']:+.2f} (north) mm/yr/km; mean "
        f"{exp['mean_diff_rate_mm_yr']:+.1f} mm/yr.",
        f"- Offset + height reference: {exo['slope_east_mm_yr_per_km']:+.2f} and "
        f"{exo['slope_north_mm_yr_per_km']:+.2f} mm/yr/km; mean "
        f"{exo['mean_diff_rate_mm_yr']:+.1f} mm/yr.",
        "- Both have a residual sd of about 6.5 mm/yr.",
        "- The plane is fitted on stable pixels east of the fan (centroid x "
        f"{rc['stable_centroid_x_km']:.1f} km; active cells reach "
        f"{rc['active_cells_x_km_min_max'][0]:.0f} km).",
        f"  - Run-F coefficient drifts: east {rc['F']['east_km']['rate_per_yr']:+.2f}, north "
        f"{rc['F']['north_km']['rate_per_yr']:+.2f} mm/km/yr.",
        f"  - Annual amplitudes: {rc['F']['east_km']['annual_amp']:.2f} and "
        f"{rc['F']['north_km']['annual_amp']:.2f} mm/km (LOS).",
        "- Long-term rates and seasonal amplitudes in the west and north of the fan carry "
        "this extrapolated ramp. A different reference needs a user decision and a new "
        "registration.", ""]

    body = [
        "## What was done (round 1, 2026-10-13; corrections in brackets)", "",
        "1. **Inputs.** S4a's hand-over: insar_licsar_subsets/v1, notes_for_S4b.json and the "
        "licsbas env.",
        "2. **LiCSBAS** in the licsbas env inside CPU slots, steps 02 (2 x 2), 11-15 at "
        "defaults, without atmospheric correction (D11).",
        "   - P (pairs <= 2022-12-31; the gate run): 345/516 ifgs, 108/114 epochs.",
        "   - F (2020-01..2025-12; the product run): 597/925 ifgs, 192/205 epochs.",
        "   - D (descending 105D, 2025-01..2026-01): 95/108 ifgs, 28 epochs.",
        "   - LiCSBAS12 rejected about a third of the ifgs on loop closure and removed summer "
        "epochs. [Not reported in round 1: this left P in 3 and F in 5 network segments.]",
        "3. **Georeference.** Pixels are placed by index on S4a's window geotransform.",
        "4. **Reference.** Stable off-fan pixels (land, DEM 100-600 m, >= 2.5 km from active "
        "cells, LiCSBAS15 mask). A robust per-epoch offset + east/north plane + height fit is "
        "subtracted everywhere. The vertical-equivalent is LOS / U per pixel.",
        "5. **Cells.** Median of >= 5 valid pixels. sd is uncalibrated. [Round-1 flags were 0 "
        "and 8 only; v2 adds 4.]",
        "6. **G1.** WRA L0 heights (7-day median) and the step catalogue (TKJS and SJES "
        "excluded); an offset-only leave-one-out tie.",
        "7. **C1.** 57 summer pairs.",
        "8. **Y.** Battery rebuilt from the design pilot; census from "
        "pump_electricity/preorigin/v1.",
        "9. **Products.**",
        "   - v1 (round 1): gate failed, fit_window_end 2025-12 [wrong], included "
        "gnss_tie_offsets.csv [removed in v2].",
        "   - v2 (repair): fit_window_end 2026-01, flags 0/4/8, segment ids. Guard: "
        "validator raises GateNotPassed; after_the_fact_analysis opens; calibration_target, "
        "scoring_target, selection and design_time_evidence raise UseNotAllowed.",
        "   - The checker passes both insar_069A manifests and every licsar SOURCE.json.", "",
        "## Product coverage (run F, cells with >= 5 valid pixels)", "",
        "; ".join(f"{k}: {v:.2f}" for k, v in cov.items()) + ".", "",
        "## Acceptance", "",
        "| criterion | threshold | measured | pass |", "|---|---|---|---|"]
    for a in acceptance:
        body.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                    f"{'yes' if a['passed'] else '**no**'} |")

    deviations = [d for d in r1["deviations_from_plan"]
                  if not d.startswith("Undisclosed development looks")] + [
        "Disclosure: the round-1 development structure tests on run D (2025) were logged only "
        "after the product was built. They computed WRA LOSO, zone coherence fractions and "
        "amplitude-census correlations without printing them. The v1 build's "
        "gnss_tie_offsets computation was not logged until this repair round.",
        "Repair round: v2 re-uses v1's values (verified hashes) and adds flags from "
        "LiCSBAS's own gap structure (cum.h5 'gap'). Nothing was re-processed or re-tested.",
        "Repair round: values that include a bridged increment get flag 4 (plan 2.4, 'filled "
        "from the series' own temporal model'), since LiCSBAS's gamma constraint is a "
        "linear-velocity model of the pixel's own series. The cell rule (any contributing "
        "pixel) was fixed from the structure alone, before any repair statistic.",
        "Repair round: after s4b_r2_diag.py ran, one comparison was rewritten for the linter "
        "(`LO <= a` -> `a >= LO`). The semantics are identical; the v2 manifest hashes the "
        "rewritten file.",
    ]
    issues = [
        "The gate failed (G1 13.2 mm > 10 mm on run P), so the guard refuses insar_069A v1 and "
        "v2 as a validator. S1b's InSAR gap check, S5 and the twin's 2022-26 InSAR validation "
        "stay blocked until the user chooses among options (a)-(d) below.",
        "Option (a): accept run F's after-the-fact 9.9 mm. That needs a U7 waiver, because F "
        "uses 2023+ interferograms. F's G1 window also spans the two bridged 2022 gaps, so "
        "validator use should then be limited to within-segment changes.",
        "Option (b): a new G1-type registration on the WRA stations in 2020-01..2025-12 is NOT "
        "clean. Those stations were looked at by the G1 score (2021-08..2022-12), the "
        "after-the-fact F windows, explore_ramp (17 stations, rates 2020-01..2025-03), the v1 "
        "build's tie-offset computation (18 stations, every epoch 2020-01..2025-12, disclosed "
        "this round) and the verifier's and this round's diagnostics. A1's blocking rule "
        "decides, and S4b claims no untouched window. Possible clean evidence: epochs after a "
        "new registration date (prospective), or the plan's leveling 2020-21 benchmark-change "
        "row (no InSAR-leveling look exists; A1 must still check it).",
        "Option (c): keep InSAR as after-the-fact analysis only.",
        "Option (d): reprocess to close the summer gaps, for example with relaxed LiCSBAS11/12 "
        "thresholds that keep the bridging interferograms, or a segment-wise treatment. That "
        "makes a new product version and a new registration. Re-test windows must not span a "
        "network gap, unless the test is designed to measure the bridge.",
        "Cause of the G1 failure, corrected: the run-P/F difference is carried by the "
        "LiCSBAS-bridged increments across the summer-2022 gaps (station-error change "
        f"{sp['rms_err_change_mm']:.1f} mm RMS, of which cross-gap "
        f"{sp['rms_gap_part_change_mm']:.1f} mm, correlation "
        f"{sp['corr_err_change_gap_change']:.2f}). It is not a network-end effect. The "
        "absolute error cannot be fully attributed, because single-interval InSAR-GNSS noise "
        "is about 9-11 mm.",
        f"Network gaps: run F has gaps in every summer 2022-2024. Flag 4 is on "
        f"{100 * fc.get('4', 0) / (tot - fc.get('8', 0)):.0f} % of the finite cell values "
        f"({100 * fc.get('4', 0) / tot:.0f} % of all cell-epochs). Use changes within one "
        "segment (2022-08-29..2023-07-31, 2023-08-12..2024-07-01, 2024-08-06..2025-12-11), "
        "or carry the bridged offsets as an explicit extra uncertainty. A 2022-26 cumulative "
        "comparison crosses two to four bridges.",
        "Reference ramp (exploratory): InSAR-minus-GNSS rate slopes are east "
        f"{exp['slope_east_mm_yr_per_km']:+.2f} and north "
        f"{exp['slope_north_mm_yr_per_km']:+.2f} mm/yr/km with the plane reference, and "
        f"{exo['slope_east_mm_yr_per_km']:+.2f} and {exo['slope_north_mm_yr_per_km']:+.2f} "
        "with offset + height. The plane's own coefficients drift "
        f"({rc['F']['east_km']['rate_per_yr']:+.2f} east, "
        f"{rc['F']['north_km']['rate_per_yr']:+.2f} north mm/km/yr) and cycle annually. A "
        "different reference needs a user decision and a new registration.",
        "C1: Changhua Z2 is NOT covered in summer (0.297). On 2022 summers alone Z2 is 0.237, "
        "Z3 0.371 and Z1 0.597, so the 'covered' labels may be optimistic for 2022+.",
        "Y failed, so the amplitude is not a Changhua validator for P3 or P4m. The test cannot "
        "separate local pumping from east-west structure: do not read the failure as evidence "
        "about the compressible thickness.",
        "East-west leakage: in 2025 the vertical-equivalent is biased by -2.9 mm/yr (median) "
        "against the asc/desc decomposition. Before 2025 only the ascending geometry exists.",
        "The per-cell sd is UNCALIBRATED and excludes the bridged-offset uncertainty. The "
        "calibrated accuracy statement is the failed G1.",
        "A1 interaction (for the harness owner or the user): under A1's membership rule, every "
        "S4b look at all active cells is eligible to block later registrations that use the "
        "11 A1 fresh spatial units (fresh.km2027.0-4, fresh.rot_tercile.1-3, "
        "fresh.two_tercile.ME/WM, fresh.river_band_YL10). The windows run 2020-01..2026-01: "
        "round 1 covered 2020-22 and 2025-26, and this round's gap-structure look adds "
        "2023-24. These looks hold InSAR or LiCSBAS network structure only, no head data. A "
        "data-type ruling is needed before the twin's held-out tests are registered.",
        "Fetching 069A_06632_162010 (about 1.09 GB) still needs a user decision. Even with it, "
        "a 2022-11 decomposition lacks a descending partner before 2025-01.",
        "Checker findings that are not S4b's: "
        "data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/SOURCE.json "
        "fails (13 files without hashes) and has 19 L0 orphans; round 1 also saw "
        "ground_elevation and heads/L1_clean code-hash failures.",
        "Before any commit of dataprep_staging, run the CLAUDE.md Rule 1 scan. The s4b_ files "
        "hold no coordinates or data.",
    ]
    tail = [
        "", "## Deviations from the plan", ""] + [f"- {d}" for d in deviations] + [
        "", "## Issues for downstream", ""] + [f"- {i}" for i in issues] + [
        "", "## Looks logged (A1 look log, harness.prereg3)", "",
        f"S4b has {len(s4)} look records. Round 1:",
        "- integrity and development: dev_D (19:23Z), and the late dev_D_struct (19:50Z);",
        "- score: G1, C1 (after its extraction look) and Y;",
        "- after_the_fact: run F pre-origin, the pilot window, NGL on-fan, C1 all pairs, "
        "decomposition 2025;",
        "- exploratory_after_the_fact: explore_ramp.",
        f"Repair round: {len(rep_looks)} records (listed above).",
        "The registration disclosed 50 unverifiable legacy looks from other WPs.", "",
        "## Ledger", "",
        f"`results/dataprep/ledger/rows/S4b.csv`: {n_led} data rows over {n_num} numbers, 0 "
        "conflicts. That is 387 from round 1 plus 190 from the repair. Every number quoted "
        "here has a row.", "",
        "## Resources", "",
        "- Claude API tokens 0, GPU none, downloads 0 bytes.",
        "- CPU: every job ran through tools/cpu_slot.sh, one slot each; every repair job "
        "finished in under 10 minutes.",
        "- Disk: s4b_run about 1.4 GB plus the gap files; product v2 41 MB.",
        "- Nothing was written under hydrophysics/, tests/ or L0, and nothing was staged "
        "in git."]
    (REP / "S4b.md").write_text("\n".join(head + body + tail) + "\n")

    rep = {"wp": "S4b", "status": "partial", "repair_round": "2026-10-14",
           "acceptance": acceptance,
           "products": [
               {"path": "results/dataprep/insar_069A/v2/", "layer": "L2",
                "manifest": f"results/dataprep/insar_069A/v2/MANIFEST.json (sha256 {msha})",
                "allowed_uses": "validator (refused while the gate fails), "
                                "after_the_fact_analysis", "fit_window_end": "2026-01"},
               {"path": "results/dataprep/insar_069A/v1/", "layer": "L2 (superseded)",
                "manifest": "results/dataprep/insar_069A/v1/MANIFEST.json",
                "allowed_uses": "superseded by v2 (no gap flags; ships gnss_tie_offsets.csv; "
                                "fit_window_end understated)", "fit_window_end": "2025-12"}],
           "deviations_from_plan": deviations, "issues_for_downstream": issues,
           "repair": {"gaps": {r: gp[r]["gaps"] for r in ("P", "F", "D")},
                      "flag_counts_v2": fc, "pf": {k: pf[k] for k in (
                          "across_gap1", "across_gap2", "within_rms_mm", "within_18_30d_rms_mm",
                          "last_segment", "g1_change")},
                      "station_split": sp, "refcoef": rc, "yconf": yc,
                      "looks": [{k: lk[k] for k in ("time", "purpose", "data_scope", "units")}
                                for lk in rep_looks]},
           "ledger_rows": n_led, "claude_api_tokens": 0, "gpu": "none"}
    (REP / "S4b.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print("wrote", REP / "S4b.md", REP / "S4b.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
