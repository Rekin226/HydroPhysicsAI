"""Text of the WP P6 report and its ledger rows (called by report_p6.py)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from common.paths import rel_to_repo

from p6_public_wells import p6const as K

PV = "public_wells/v2"
S_PH2 = rel_to_repo(K.SRC["plan_phase2"])
S_REV1 = rel_to_repo(K.SRC["plan_rev1"])
S_ANNEX = rel_to_repo(K.SRC["moea_annex"])
S_CY = rel_to_repo(K.SRC["cy53001"])
S_DW = rel_to_repo(K.SRC["drought_wells"])
S_B110 = rel_to_repo(K.SRC["brief110"])
A22 = rel_to_repo(K.API_CACHE / "p2_at2_2.json")
A25 = rel_to_repo(K.API_CACHE / "p2_at2_5.json")
AB = rel_to_repo(K.API_CACHE / "b110_irr_yl.json")
C0L1 = "results/dataprep/pump_electricity/census_L1/v3"
C0PRE = "results/dataprep/pump_electricity/preorigin/v3"
ENG = "results/dataprep/changhua_abstraction/engine/v2"
N1P = "results/dataprep/natural_experiments/v1"
GRID = rel_to_repo(K.GRID_NPZ)
PROD = "results/dataprep/public_wells/v2"

LEDGER = {
    "P6.R1.wells_2011_2018": [(S_PH2, "input"), (A22, "input"), (S_ANNEX, "validator")],
    "P6.R1.volume_2011_2018": [(S_PH2, "input"), (A22, "input"), (S_ANNEX, "validator")],
    "P6.R2.phase1_totals": [(S_PH2, "input"), (A22, "input")],
    "P6.R2.cross_tables": [(S_PH2, "input"), (A22, "input"), (A25, "input")],
    "P6.R2.control_yuan_vs_phase2": [(S_PH2, "input"), (S_CY, "validator")],
    "P6.T1.transcription": [(A22, "input"), (A25, "input"), (S_PH2, "validator")],
    "P6.D1.drought_wells": [(S_DW, "input"), (N1P, "validator")],
    "P6.inv.annex": [(S_ANNEX, "input")],
    "P6.inv.phase2": [(S_PH2, "input")],
    "P6.inv.rev1": [(S_REV1, "input")],
    "P6.inv.control_yuan": [(S_CY, "input")],
    "P6.C1.plan_counts": [(C0L1, "input")],
    "P6.C1.drought_census_match": [(S_DW, "input"), (C0L1, "input"), (C0PRE, "input")],
    "P6.C1.census_energy": [(C0L1, "input"), (C0PRE, "input")],
    "P6.K1.ratio": [(S_B110, "input"), (AB, "input"), (ENG, "input"), (C0PRE, "input")],
    "P6.K1.wra_m3_per_kwh": [(S_B110, "input"), (AB, "input"), (C0PRE, "input")],
    "P6.M1.member_primary": [(S_PH2, "input"), (A22, "input"), (S_DW, "input"),
                             (C0L1, "input"), (ENG, "input"), (GRID, "input")],
    "P6.M1.member_variants": [(S_PH2, "input"), (A22, "input"), (S_DW, "input"),
                              (C0L1, "input"), (ENG, "input"), (GRID, "input")],
    "P6.M1.pending_tw_changhua": [(S_PH2, "input"), (S_DW, "input"), (GRID, "input")],
    "P6.M1.extra_tuku_huwei": [(S_PH2, "input"), (C0L1, "input"), (GRID, "input")],
    "P6.M1.exemption_sensitivity": [(S_DW, "input"), (C0L1, "input"), (ENG, "input")],
    "P6.atf.credits_vs_janmay": [(S_PH2, "input"), (N1P, "validator")],
    "P6.v1.exemption.drought_well_locations": [(S_DW, "input")],
    "P6.v1.exemption.census_attributes": [(C0L1, "input")],
    "P6.v1.exemption.c0_residual_dec2022": [(C0PRE, "input")],
    "P6.product.checks": [(PROD, "input")],
    "P6.product.v1_superseded": [("results/dataprep/public_wells/v1", "input")],
    "P6.api.tokens": [(A22, "input"), (A25, "input"), (AB, "input")],
}


def _f(x, n=2):
    return f"{x:,.{n}f}"


def compose(d: Path, val: dict, man: dict, fin: dict, clos: pd.DataFrame, ann: pd.DataFrame,
            atf: dict):
    c = val["checks"]
    r1, t1, d1, m1 = c["R1"], c["T1"], c["D1"], c["M1"]
    k1 = val["K1_ratio_preorigin"]
    kp = k1["primary"]
    s = val["summary"]
    sens = val["exemption_sensitivity"]
    tok = val["api_token_usage"]
    op, so = tok["claude-opus-5-5"], tok["claude-sonnet-5-5"]
    cm = {(r["county"], r["agency"][:6]): r for r in val["C1_census"][
        "drought_census_match_summary"]}
    eb = pd.read_csv(d / "census_energy_by_purpose.csv")
    pub = eb[eb.purpose_raw.isin(K.PUBLIC_PURPOSES)].groupby("county").agg(
        n=("n_pumps_in_grid", "sum"), g=("gwh_per_year_in_grid", "sum"))
    ind = eb[eb.purpose_raw.isin(K.INDUSTRIAL_PURPOSES)].groupby("county").agg(
        n=("n_pumps_in_grid", "sum"), g=("gwh_per_year_in_grid", "sum"))
    tot = eb.groupby("county").gwh_per_year_in_grid.sum()
    pc = pd.read_csv(d / "census_purpose_counts.csv")
    ch_pc = pc[pc.county == "changhua"].set_index("purpose_raw")
    fp = fin["guard_probes"]
    wra_kwh = k1["wra_m3_per_census_kwh"]
    disp = s["phase1_disposed_volume_Mm3_per_yr"]
    eq_disp = disp * 1e6 / wra_kwh / 1e6
    eq_tw = 149e6 / wra_kwh / 1e6
    eq_pend = val["pending_tw_changhua"]["volume_1e4m3_per_yr"] * 1e4 / wra_kwh / 1e6
    pubtot = float(pub.g.sum())
    indtot = float(ind.g.sum())
    pr = s["member_primary_in_grid_Mm3_per_yr"]
    cy_ch, yl = s["member_primary_changhua_yunlin_2010"]
    tw_y = atf["tw_yunlin"]["janmay_1e4m3"]
    tw_c = atf["tw_changhua"]["janmay_1e4m3"]
    ia_y = atf["ia_yunlin"]["janmay_1e4m3"]
    ia_y_mean = sum(ia_y[y] for y in range(2016, 2021)) / 5 / 100
    status = "passed" if man["gate"]["passed"] else "failed"

    acc = [
        {"criterion": "Inventory totals reconcile with the MOEA annex totals (plan WP P6 "
                      "acceptance; R1 of P6.inventory_member.v1): public wells disposed "
                      "2011-2018 and their abstraction cut",
         "threshold": "relative difference <= 5 % for wells and for volume",
         "measured": f"wells {r1['N_printed_row_totals_100_107']:.0f} vs 774 "
                     f"({100 * r1['rel_diff_wells']:.2f} %); volume "
                     f"{_f(r1['V_cells_100_107'])} vs 8,793.9 x 10^4 m3/yr "
                     f"({100 * r1['rel_diff_volume']:.4f} %); consistency check: the totals "
                     "were seen before the declaration (disclosed), the threshold is the plan's",
         "passed": bool(r1["passed"])},
        {"criterion": "Transcription of annex table 2-2 and 2-5 (T1): two independent Claude API "
                      "reads agree, every row matches the PDF text layer, totals add up",
         "threshold": "0 read disagreements, 0 text-layer mismatches, arithmetic exceptions "
                      "only where a printed note explains them",
         "measured": "0 / 0 cell disagreements (counts, volumes), 0 text-layer mismatches over "
                     "24 rows of 2-2 and 8 row groups of 2-5; 2 count exceptions (row 105: "
                     "cells 73 vs printed 72; Yunlin Taiwan Sugar column: 8 vs 7), both "
                     "explained by the printed note '(100年1口停用改封填)'; volume sums within "
                     "printed rounding; table 4-5 arithmetic 12/12 ok",
         "passed": bool(t1["passed"])},
        {"criterion": "Drought-well inventory parse (D1) equals the report's table 3 and N1's rows",
         "threshold": "Changhua 49 = 26 + 23; Yunlin 521 = 421 + 98 + 2 with the printed gap "
                      "at number 280; rows equal natural_experiments/v1",
         "measured": "Changhua 49 rows (26 / 23), Yunlin 520 printed rows (420 / 98 / 2; "
                     "number 280 missing in the print); equal to N1 on number, agency, "
                     "township, coordinates and cell (P6 repairs N1's well names)",
         "passed": bool(d1["passed"])},
        {"criterion": "Census PURPOSE reconciliation reproduces the plan's census counts (C1)",
         "threshold": "Yunlin 430 public supply, 326 industrial, 91 domestic-and-public, 9 "
                      "pumps >= 50 HP, 1,341 > 10 HP (plan WP P6 'Gap')",
         "measured": "430 / 326 / 91 / 9 / 1,341, max 100 HP (all equal)",
         "passed": all(val["C1_census"]["plan_counts"]["equal"].values())},
        {"criterion": "Sensitivity member construction (M1): mass balance, time axis, no "
                      "observation",
         "threshold": "max relative error <= 1e-9; months 2007-01..2022-12; no head, ring, pile, "
                      "GNSS or InSAR value read",
         "measured": f"max relative error {m1['max_rel_err']:.1e} over 8 variants; 192 months "
                     "2007-01..2022-12; P6 read no observation of those kinds",
         "passed": bool(m1["passed"])},
        {"criterion": "Anti-leakage (U3, U7): pre-origin uses only through declared exemptions; "
                      "no 2023+ observation in any gate, threshold or member choice",
         "threshold": "non_exempt_fit_window_end <= 2022-12; every post-origin input an "
                      "exemption with a sensitivity; 0 post-origin observations",
         "measured": "non_exempt_fit_window_end 2022-12; 3 exemptions (drought-well locations "
                     "2023-04, census attributes 2026-07, C0 residual Dec-2022 halves 2023-01), "
                     "each with a sensitivity; 0 observations of any kind dated 2023+ read",
         "passed": True},
        {"criterion": "Product manifest, checker and loader guard",
         "threshold": "checker ok; pre-origin uses open through the exemptions; targets, "
                      "baselines, selection refused; two rebuilds byte-identical",
         "measured": f"v2: checker return code {fin['checker_returncode']} "
                     f"({fin['checker_tail'][-1]}); guard 13/13 probes as expected "
                     f"({sum(v['opened'] for v in fp.values())} opened, "
                     f"{sum(not v['opened'] for v in fp.values())} refused); scratch rebuilds "
                     f"identical {fin['determinism']['identical_a_b']}, equal to the product "
                     f"{fin['determinism']['identical_product_a']}; ruff clean; superseded v1 "
                     "fails the checker on the edited build_product.py hash only",
         "passed": bool(fin["checker_returncode"] == 0 and fin["guard_all_as_expected"]
                        and fin["determinism"]["identical_a_b"]
                        and fin["determinism"]["identical_product_a"])},
    ]
    for a in acc:
        a["passed"] = bool(a["passed"])

    key_numbers = [
        f"R1: inventory 2011-2018 = {r1['N_printed_row_totals_100_107']:.0f} wells (annex 774, "
        f"+0.13 %) and {_f(r1['V_cells_100_107'])} x 10^4 m3/yr (annex 8,793.9, -0.0006 %); "
        "cohort cells count 776 (one Taiwan Sugar well stopped in 2011 and filled in 2016 is "
        "counted twice)",
        f"Phase 1 (2011-2019): 1,056 wells disposed, {_f(disp)} Mm3/yr agency-reported cut "
        f"(table; the text prints 13,098.45 x 10^4, 3.6 more); Taiwan Water "
        f"{_f(s['by_agency_Mm3_per_yr']['taiwan_water'])}, irrigation associations "
        f"{_f(s['by_agency_Mm3_per_yr']['irrigation_association'])}, Taiwan Sugar "
        f"{_f(s['by_agency_Mm3_per_yr']['taiwan_sugar'])}, schools "
        f"{_f(s['by_agency_Mm3_per_yr']['school'])}, prison and factories "
        f"{_f(s['by_agency_Mm3_per_yr']['prison'] + s['by_agency_Mm3_per_yr']['factory'])}; "
        f"Yunlin {_f(s['by_county_Mm3_per_yr']['yunlin'])}, Changhua "
        f"{_f(s['by_county_Mm3_per_yr']['changhua'])} Mm3/yr",
        "Disposal types 2011-2019 (table 4-5): reduced 430, stopped 241, filled 385; planned "
        "2020-2024: Changhua Taiwan Water 129 (85 reduced, 44 stopped) with the Niaozuitan lake, "
        "(7,172.25 - 1,842.3) x 10^4 = 53.30 Mm3/yr still pumping through 2022",
        "Public wells outside phase-1 control (phase-2 plan): Taiwan Water 48 (Changhua 46), "
        "Taiwan Sugar 111, Changhua agencies and schools 277, Yunlin 82; Control Yuan 53001: "
        "1,191 under control (Taiwan Water 350 vs 344 in the phase-2 list) and 1,140 other "
        "legal wells",
        "Baseline of the 2011 plan (first revision): Taiwan Water groundwater 1.49 x 10^8 "
        "m3/yr (Changhua 1.03, Yunlin 0.46); groundwater self-supplied outside its service "
        "area 3.18 x 10^8 m3/yr",
        f"Census (C0 L1 v3) reproduces the plan's Yunlin counts exactly (430 / 326 / 91; 9 "
        f">= 50 HP; 1,341 > 10 HP); Changhua 公共用水 {int(ch_pc.loc['公共用水', 'n'])}, "
        f"工業用水 {int(ch_pc.loc['工業用水', 'n'])}, 家用及公共給水 "
        f"{int(ch_pc.loc['家用及公共給水', 'n'])} pumps",
        f"Census in-grid kept-pump energy 2012-2021 (GWh/yr, model units): public-supply "
        f"purposes Changhua {_f(pub.g['changhua'])}, Yunlin {_f(pub.g['yunlin'])}; industrial "
        f"{_f(ind.g['changhua'])} / {_f(ind.g['yunlin'])}; all purposes "
        f"{_f(tot['changhua'], 1)} / {_f(tot['yunlin'], 1)}",
        f"Drought wells with any census pump within 50 m: Taiwan Water Yunlin "
        f"{100 * cm[('Yunlin', 'Taiwan')]['share_any_le50m']:.0f} %, Changhua "
        f"{100 * cm[('Changhua', 'Taiwan')]['share_any_le50m']:.0f} %; irrigation agency "
        f"{100 * cm[('Yunlin', 'Irriga')]['share_any_le50m']:.0f} % / "
        f"{100 * cm[('Changhua', 'Irriga')]['share_any_le50m']:.0f} %; public-supply census "
        f"pump within 50 m: <= {100 * cm[('Yunlin', 'Irriga')]['share_public_le50m']:.0f} %; "
        "the near pumps are small irrigation or household pumps (median 2-3 HP)",
        f"K1 (pre-origin): WRA / model Yunlin Jan-May irrigation volume 2013-2021 = "
        f"{kp['ratio']:.3f} (WRA {kp['wra_mean_Mm3']:.1f}, model {kp['model_mean_Mm3']:.1f} Mm3; "
        f"per-year sd {kp['per_year_sd']:.3f}, {kp['per_year']['2013']:.2f} in 2013 to "
        f"{kp['per_year']['2021']:.2f} in 2021); pilot definition 2012-2021 "
        f"{k1['pilot_definition']['ratio']:.3f}; pilot 2.576; plan 2.58 kept; WRA m3 per "
        f"census kWh {wra_kwh:.2f}",
        f"Member primary (in grid, physical): {_f(pr['2010'], 1)} Mm3/yr before 2011 "
        f"(Changhua cells {_f(cy_ch, 1)}, Yunlin {_f(yl, 1)}), {_f(pr['2012'], 1)} in 2012, "
        f"{_f(pr['2015'], 1)} in 2015, {_f(pr['2018'], 1)} in 2018, {_f(pr['2019'], 1)} in 2019 "
        f"(annual mean), 0 from {s['member_primary_zero_from']} (timing_late "
        f"{s['member_timing_late_zero_from']}); "
        f"{_f(s['member_primary_in_grid_2010_model_units'], 1)} Mm3/yr in model units (/2.58)",
        f"Member components: pending Changhua Taiwan Water "
        f"{_f(s['pending_tw_changhua_in_grid_Mm3_per_yr'], 1)} Mm3/yr in grid (82.6 % of 53.30), "
        f"constant to 2022-12; Tuku-Huwei 2013 measure "
        f"{_f(s['extra_tuku_huwei_in_grid_Mm3_per_yr_2012'], 2)} Mm3/yr until 2013-06",
        f"Location sensitivity (2010, overlap of cell volumes with the primary): census-only "
        f"{sens['loc_census_only']['overlap_with_primary']:.2f}, uniform-county "
        f"{sens['loc_uniform_county']['overlap_with_primary']:.2f}, drought-only "
        f"{sens['loc_drought_only']['overlap_with_primary']:.2f}; county centroid shifts up to "
        f"{max(v.get('yunlin_centroid_shift_km', 0) for v in sens.values()):.1f} km "
        f"(Yunlin) and {max(v.get('changhua_centroid_shift_km', 0) for v in sens.values()):.1f} "
        "km (Changhua); in-grid totals 125.3-130.9 Mm3/yr",
        f"Census-equivalent energy at {wra_kwh:.2f} m3/kWh (order of magnitude): phase-1 cut "
        f"{eq_disp:.1f} GWh/yr, 2011 Taiwan Water groundwater {eq_tw:.1f}, pending Changhua "
        f"Taiwan Water {eq_pend:.1f}, against {pubtot:.1f} (public-supply) + {indtot:.1f} "
        "(industrial) GWh/yr in the census in grid",
        f"After the fact (N1 two-read Jan-May series, never fed back): Yunlin Taiwan Water "
        f"{tw_y[2016] / 100:.2f} / {tw_y[2017] / 100:.2f} / {tw_y[2018] / 100:.2f} / "
        f"{tw_y[2019] / 100:.2f} Mm3 in 2016-2019 (the table credits 16.18 Mm3/yr to 2017 and "
        f"5.76 to 2018); Changhua Taiwan Water {tw_c[2011] / 100:.2f} -> {tw_c[2021] / 100:.2f} "
        f"(2011 -> 2021; credited 18.42 Mm3/yr); Yunlin irrigation agency "
        f"{ia_y[2011] / 100:.2f} in 2011, mean {ia_y_mean:.2f} in 2016-2020 (credited cut "
        "44.52 Mm3/yr)",
        f"Claude API: claude-opus-5-5 {op['calls']} calls, {op['input_tokens']:,} in / "
        f"{op['output_tokens']:,} out; claude-sonnet-5-5 {so['calls']} calls, "
        f"{so['input_tokens']:,} in / {so['output_tokens']:,} out; GPU none",
    ]

    deviations = [
        "Code lives in dataprep_staging/p6_public_wells/ (user decision D12), not "
        "results/dataprep/_staging_code/.",
        "The plan's '2011-18 closure ramp' (774 wells, MOEA annex) is the variant "
        "annex_2011_2018; the primary member uses all phase-1 cohorts 2011-2019 (1,056 wells, "
        "130.9 Mm3/yr) because the phase-2 plan (2020-12, newly added L0) shows the same "
        "programme closing 281 wells / 43.0 Mm3/yr in 2019, including the 13 Taiwan Water wells "
        "the annex reports as delayed to March 2019. Declared before construction.",
        "New L0 source wra_subsidence_action_plans (3 PDFs, 8.7 MB, SOURCE.json; fetched by an "
        "earlier P6 attempt on 2026-10-14): the 2019 annex prints only totals, the phase-2 main "
        "plan holds the per-agency, per-year tables. The phase-2 first-revision annex "
        "(published 2026-06) is in L0 but not used (post-origin).",
        "Plan step 3's 2.58 comes from a design pilot that A0 labels after-the-fact only; P6 "
        "recomputed it pre-origin (K1: 2.469 on 2013-2021, 2.506 on the pilot's 2012-2021) "
        "and, as declared, keeps 2.58 in the member.",
        "Added after the declaration (not part of any declared variant): the component "
        "extra_tuku_huwei_2013 (475 x 10^4 m3/yr, the 2013 Tuku-Huwei-Yuanchang groundwater "
        "reduction measure of annex table 2-9, found while building the inventory), an "
        "'other_cells' row in member_annual.csv, and the repair of N1's drought-well names.",
        "A first build stopped after creating public_wells/v1 (a formatting error in the "
        "product card, before any manifest); the partial directory was moved to "
        "reports/P6_evidence/failed_build/ and v1 was rebuilt from the same code path. The "
        "published v1's PRODUCT_CARD.md then turned out to state that the primary member ramps "
        "to 0 by 2020-07 (the arrays reach 0 from 2019-07), so public_wells/v2 was published "
        "with the corrected card and a 'supersedes' record; all data files of v1 and v2 are "
        "byte-identical. No published version was overwritten. Use v2. Because "
        "build_product.py was edited for v2, the checker now fails v1 on that one code hash "
        "(the W3 precedent of A0fix: superseded versions are recorded, not rebuilt).",
        "Request R-i was not drafted (user decision).",
    ]

    issues = [
        "The member is a sensitivity member only (plan P6 step 3): it must not enter the "
        "primary model. Use rate_phys_m3d__<variant> (m3/day, positive = abstraction; pass "
        "-rate as the engine's q_extra) or rate_model_m3d__primary; add pending_tw_changhua "
        "for the still-pumping Changhua Taiwan Water wells. Pre-origin uses go through 3 "
        "exemptions (drought-well locations, census attributes, C0 residual December-2022).",
        "Agency-credited cuts are not metered pumping. After the fact, WRA's Jan-May public-well "
        "series (N1, 110 briefing, a pre-origin publication) puts the Yunlin Taiwan Water drop "
        "in 2019, while the table credits 21.9 Mm3/yr of it to 2017-2018; the Yunlin "
        "irrigation-agency credit (44.5 Mm3/yr) exceeds that agency's entire 2011 Jan-May "
        "public-well pumping (33.9 Mm3) and its series does not fall by that much. A later member "
        "version (public_wells/v3) built on the measured Jan-May series (needs its own "
        "declaration; seasonality "
        "assumption for annual totals) is the natural next step for N1's closure-ramp test "
        "(C4) and rung R6.",
        "Location proxies disagree at the cell scale (overlap 0.19-0.93 with the primary, "
        "centroid shifts up to about 7 km) although fan totals and the county split agree "
        "within 5 %: report member results for the location variants, not the primary alone.",
        "Changhua Taiwan Water (11th district) still pumped about 53 Mm3/yr from the 129 "
        "pending wells (plus other wells) through 2022; WRA's Jan-May series falls only from "
        "40.2 to 27.2 Mm3 over 2011-2021. This continuing deep stress is absent from the model "
        "and from the closure ramp (component pending_tw_changhua).",
        "The census does not contain the public deep wells: public-supply purposes carry 4.2 "
        "GWh/yr in grid against a census-equivalent 15.7 GWh/yr for the phase-1 cut alone, and "
        "drought wells sit next to small private pumps only. P4m/I1 should treat public and "
        "industrial deep abstraction as missing forcing, not as part of the census classes.",
        "Layers: Control Yuan 53001 puts Yunlin irrigation-association public wells at "
        "120-150 m (aquifer 2) and the Tuku Taiwan Water wells at 244-301 m; the plan's layers "
        "3-4 suit Taiwan Water but not the irrigation wells (variant layers_ia_aq2).",
        "K1: the WRA-to-model ratio falls from 2.74 (2013) to 2.09 (2021) (per-year sd 0.26): "
        "the model's census abstraction grows faster than WRA's estimate. Any use of a single "
        "2.58 should carry that spread.",
        "Source quality: annex table 2-2 counts one Taiwan Sugar well twice (row 105); the "
        "phase-2 text prints 13,098.45 where its table sums to 13,094.85; Control Yuan lists "
        "164 Changhua Taiwan Water wells against 158 in the phase-2 plan; the drought-well "
        "table skips number 280.",
        "N1: its drought-well parser drops every token equal to the well count from the name "
        "('二林 1 號井重鑿' -> '二林 號井重鑿'); coordinates and counts are unaffected. P6's "
        "wells_located.csv has the repaired names.",
        "Maps of the pending Changhua Taiwan Water wells and of the non-phase-1 public wells "
        "(phase-2 plan figures 4-3 and 4-4, pp. 49-50) are not digitised; georeferencing them "
        "would locate those wells better than the drought-well proxy.",
        "ledger/data_roles.csv was not reassembled (shared file); rows/P6.csv is complete.",
    ]

    products = [{
        "path": "results/dataprep/public_wells/v2",
        "layer": "L2",
        "manifest": "results/dataprep/public_wells/v2/MANIFEST.json (supersedes v1: card "
                    "corrected, data byte-identical; + PRODUCT_CARD.md, "
                    "validation.json); inventory.csv, closures.csv, wells_located.csv, "
                    "census_purpose_counts.csv, census_energy_by_purpose.csv, "
                    "drought_census_match(_summary).csv, member_cohorts.csv, "
                    "member_annual.csv, ratio_check.json, sensitivity_member.npz; 3 "
                    "exemptions; gate R1+T1+D1+M1 passed",
        "allowed_uses": "forcing_frozen (sensitivity member only, never the primary model), "
                        "model_design, prior, design_time_evidence (pre-origin, through the 3 "
                        "exemptions), validator, after_the_fact_analysis; scoring_target, "
                        "calibration_target, baseline, datum_fit, selection, initial_conditions "
                        "forbidden",
        "fit_window_end": "2026-07 (non_exempt_fit_window_end 2022-12)"},
        {"path": "data_fetch_api/new_measurements/"
                 "wra_subsidence_action_plans",
         "layer": "L0",
         "manifest": "SOURCE.json (dataprep-source/1, 3 PDFs with url, fetch time, bytes, "
                     "sha256, licence note)",
         "allowed_uses": "raw input; untrusted data; WRA website publication, cite and do not "
                         "redistribute",
         "fit_window_end": "n/a (published 2020-08, 2020-12 and 2026-06)"},
        {"path": "results/dataprep/reports/P6_evidence",
         "layer": "evidence",
         "manifest": "none (text layers, API cache, finalize.json, checker output, "
                     "after-the-fact comparison, failed partial build)",
         "allowed_uses": "provenance only",
         "fit_window_end": "n/a"},
        {"path": "results/dataprep/ledger/rows/P6.csv",
         "layer": "ledger", "manifest": "ledger fragment (explicit sources, one role each)",
         "allowed_uses": "data-role ledger (U6)", "fit_window_end": "n/a"}]

    code = sorted(str(p) for p in K.CODE_DIR.glob("*.py"))
    summary = (
        "WP P6 (Phase-0 inventory) built public_wells/v2 (v1 has the same data and a product-"
        "card error, superseded): a sourced inventory of the public-"
        "supply, Taiwan Sugar and irrigation-association deep wells of Yunlin-Changhua, the "
        "census PURPOSE reconciliation and the fan-scale closure-ramp sensitivity member. The "
        "per-agency, per-year disposal tables come from the 2020 phase-2 action plan (new L0 "
        "source), transcribed by two Claude API reads that agree on every cell and match the "
        "PDF text layer row by row. The plan's acceptance passes: the inventory gives "
        f"{r1['N_printed_row_totals_100_107']:.0f} wells and {_f(r1['V_cells_100_107'])} x 10^4 "
        "m3/yr for 2011-2018 against the annex's 774 and 8,793.9 (0.13 % and 0.0006 %); the "
        "totals were seen before the declaration, which is disclosed. Phase 1 closed 1,056 "
        f"wells (2011-2019) with an agency-reported cut of {_f(disp, 1)} Mm3/yr, and 129 "
        "Changhua Taiwan Water wells (53.3 Mm3/yr) were still pumping through 2022. The census "
        "reproduces the plan's counts exactly but does not contain these wells: public-supply "
        f"purposes carry {pubtot:.1f} GWh/yr in grid, and drought wells sit next to small private "
        "pumps only. The member (primary "
        f"{_f(pr['2010'], 1)} Mm3/yr in grid before 2011, 0 from {s['member_primary_zero_from']}, "
        "physical; /2.58 for "
        "model units) has 7 declared variants and 2 add-on components; the location variants "
        "overlap the primary by 0.19-0.93, so location is a first-order sensitivity. A pre-origin "
        f"recomputation of the plan's 2.58 gives {kp['ratio']:.2f} with a falling trend. After "
        "the fact, WRA's Jan-May public-well series shows the agency credits are not metered "
        "pumping (Yunlin Taiwan Water drop in 2019, not 2017; Yunlin irrigation-agency credit "
        "above its whole 2011 Jan-May pumping). No GPU was used; Claude API "
        f"{op['input_tokens'] + so['input_tokens']:,} input and "
        f"{op['output_tokens'] + so['output_tokens']:,} output tokens (6 calls); nothing under "
        "hydrophysics/, tests/ or existing L0 was written; nothing staged in git; R-i not "
        "drafted.")

    js = {"wp": "P6", "status": status, "summary": summary, "acceptance": acc,
          "products": products, "code_paths": code, "key_numbers": key_numbers,
          "deviations_from_plan": deviations, "issues_for_downstream": issues,
          "report_path": "results/dataprep/reports/P6.md"}

    lines = ["# WP P6: public-supply and industrial deep wells, Phase-0 inventory (report, "
             "2026-10-15)", "", f"Status: **{status}**.", "", summary, "", "## Acceptance", "",
             "| criterion | threshold | measured | pass |", "|---|---|---|---|"]
    for a in acc:
        lines.append(f"| {a['criterion']} | {a['threshold']} | {a['measured']} | "
                     f"{'yes' if a['passed'] else 'no'} |")
    lines += ["", "## Pre-registration and looks", "",
              f"- `{K.PREREG_ID}` (declaration, harness.prereg3) fixes the reconciliation rule "
              "R1 with the plan's 5 % threshold, the transcription rules T1, the drought-well "
              "check D1, the census comparison C1, the pre-origin ratio check K1 and every "
              "assumption of the member M1 (p6const.py hashed), before any of them was "
              "computed.",
              "- Disclosed look (looks.jsonl, 2026-10-14T15:57:12Z, purpose inventory_read): "
              "while locating the tables, P6 read the annex totals and the text layer of annex "
              "table 2-2 and summed its 2011-2018 rows by eye. R1 is therefore reported as a "
              "consistency check of the transcription against the annex.",
              "- P6 read no head, ring, pile, GNSS or InSAR value at any time, so no 2023+ "
              "observation can enter a gate, a threshold or a member choice (U7).", "",
              "## What was built", "",
              "- `fetch_plans.py` (earlier attempt, kept): three MOEA action-plan PDFs into L0 "
              "`wra_subsidence_action_plans/` with SOURCE.json.",
              "- `transcribe_p6.py`: two Claude API reads per page (claude-opus-5-5 on the PDF "
              "page, claude-sonnet-5-5 on the 200-dpi PNG) of annex tables 2-2 and 2-5 and of "
              "the 110 briefing's irrigation chart, cached in P6_evidence/api_cache.",
              "- `tables.py`: verification of the reads against the text layer and the printed "
              "totals; text-layer parsers for table 4-5, annex table 2-3 and the table 2-9 "
              "cross-check.",
              "- `inventory.py`: 33 quoted report facts plus 125 table rows (inventory.csv), the "
              "cohort table (closures.csv), reconciliation R1/R2.",
              "- `census_check.py`: drought wells (N1's parser, names repaired), census purpose "
              "counts, kWh by purpose, nearest-pump matching.",
              "- `ratio.py`: K1. `member.py`: the member and its variants. `build_product.py`, "
              "`finalize_p6.py`, `report_p6.py`, `report_text.py`, `prereg_p6.py`.", "",
              "## Inventory", "",
              "| agency (MOEA phase 1) | Changhua wells | Changhua cut (10^4 m3/yr) | Yunlin "
              "wells | Yunlin cut (10^4 m3/yr) |", "|---|---|---|---|---|"]
    t = clos[clos.status == "disposed"].groupby(["agency", "county"]).agg(
        n=("n_wells", "sum"), v=("volume_1e4m3_per_yr", "sum"))
    for ag in ("taiwan_water", "irrigation_association", "taiwan_sugar", "school", "factory",
               "prison"):
        ch = t.loc[(ag, "changhua")] if (ag, "changhua") in t.index else None
        y = t.loc[(ag, "yunlin")] if (ag, "yunlin") in t.index else None
        lines.append(f"| {ag} | {'' if ch is None else int(ch.n)} | "
                     f"{'' if ch is None else _f(ch.v)} | {'' if y is None else int(y.n)} | "
                     f"{'' if y is None else _f(y.v)} |")
    lines += ["", "Counts are cohort-cell sums (Yunlin Taiwan Sugar 8 includes the well counted "
              "twice; the printed total is 7). By year (all agencies, printed row totals): "
              "2011 22 / 130.3, 2012 214 / 592.1, 2013 45 / 1,234, 2014 91 / 1,304.8, 2015 78 / "
              "1,468, 2016 72 / 973.2, 2017 141 / 2,053.5, 2018 112 / 1,038, 2019 281 / 4,301 "
              "(wells / 10^4 m3/yr).", "",
              "Cross-checks (R2, all consistent): table 2-2 against table 4-5 (actual 2011-2019 "
              "by agency and county, equal), annex table 2-3 (executed counts equal; volumes "
              "within 0.05), annex table 2-5 (irrigation-association counts by year equal; "
              "Taiwan Sugar 12 wells and 559.2 equal; irrigation volume 5,091.75 vs 5,092.1 "
              "printed) and annex table 2-9 (Taiwan Water counts and volumes by year all "
              "printed there). Control Yuan 53001 (2015) lists 1,191 wells under control; the "
              "phase-2 list has 1,185 (the 6 difference is Changhua Taiwan Water, 164 vs 158).",
              "", "## Census PURPOSE reconciliation", "",
              "The census (C0 census_L1/v3) reproduces the plan's Yunlin counts exactly. Its "
              "public-supply pumps are small (max 20 HP in Yunlin, 11 HP in Changhua) and carry "
              f"{_f(pub.g['changhua'])} (Changhua) and {_f(pub.g['yunlin'])} (Yunlin) GWh/yr in "
              f"grid over 2012-2021; industrial pumps {_f(ind.g['changhua'])} and "
              f"{_f(ind.g['yunlin'])}. A drought well has a census pump within 50 m in 24-54 % "
              "of cases, but the nearest pumps are household or irrigation pumps of 2-3 HP, and "
              "a public-supply census pump lies within 50 m of at most 1 % of them. At WRA's "
              f"{wra_kwh:.2f} m3 per census kWh, the phase-1 cut alone would be {eq_disp:.1f} "
              f"GWh/yr and the 2011 Taiwan Water groundwater {eq_tw:.1f} GWh/yr (an order of "
              "magnitude: deep wells need more energy per m3 than the irrigation pumps the "
              "ratio describes). The public deep wells are therefore not in the census; they are "
              "missing forcing in both counties.", "",
              "## Sensitivity member", "",
              "| variant | 2010 in grid (Mm3/yr, physical) | change |", "|---|---|---|"]
    desc = {"primary": "cohorts 2011-2019, stop at mid-year, drought-well and census proxies, "
                       "aquifers 3-4",
            "annex_2011_2018": "cohorts 2011-2018 only (the plan's annex window)",
            "timing_early": "stop at January of the disposal year",
            "timing_late": "stop at January of the next year",
            "layers_ia_aq2": "irrigation-association cohorts in aquifer 2",
            "loc_census_only": "census public-supply pumps for every cohort",
            "loc_drought_only": "drought wells for every cohort",
            "loc_uniform_county": "uniform over the county's active cells (no post-origin "
                                  "location input)"}
    for v in K.VARIANTS:
        lines.append(f"| {v} | {_f(s['member_variants_2010_Mm3'][v], 1)} | {desc[v]} |")
    lines += ["", f"Primary in-grid volume by year (annual means): 2010 {_f(pr['2010'], 1)}, "
              f"2012 {_f(pr['2012'], 1)}, 2015 {_f(pr['2015'], 1)}, 2018 {_f(pr['2018'], 1)}, "
              f"2019 {_f(pr['2019'], 1)}, 2020 0 Mm3/yr (physical; divide by 2.58 for model "
              f"units); zero from {s['member_primary_zero_from']}. "
              "Components: pending_tw_changhua "
              f"{_f(s['pending_tw_changhua_in_grid_Mm3_per_yr'], 1)} Mm3/yr in grid through "
              "2022-12; extra_tuku_huwei_2013 "
              f"{_f(s['extra_tuku_huwei_in_grid_Mm3_per_yr_2012'], 2)} Mm3/yr until 2013-06 "
              "(added after the declaration).", "",
              "Exemption sensitivity (2010 cell volumes): overlap with the primary "
              f"{sens['loc_census_only']['overlap_with_primary']:.2f} (census only), "
              f"{sens['loc_drought_only']['overlap_with_primary']:.2f} (drought wells only), "
              f"{sens['loc_uniform_county']['overlap_with_primary']:.2f} (uniform); Changhua "
              f"cells {_f(sens['loc_census_only']['changhua_Mm3_per_yr'], 1)} / "
              f"{_f(sens['loc_drought_only']['changhua_Mm3_per_yr'], 1)} / "
              f"{_f(sens['loc_uniform_county']['changhua_Mm3_per_yr'], 1)} against "
              f"{_f(sens['primary']['changhua_Mm3_per_yr'], 1)} Mm3/yr. The proxies agree on "
              "the fan and county totals and disagree on where in each county.", "",
              "## WRA-to-model ratio, pre-origin (K1)", "",
              f"WRA's Yunlin Jan-May irrigation pumping (110 briefing p. 18, two reads within "
              f"{100 * val['K1_transcription']['max_abs_diff_frac_axis']:.2f} % of the axis; the "
              "second read put the series on the township axis and was rescaled to the county "
              "axis by the axis titles) against the engine/v2 replica's Yunlin census "
              f"abstraction times the irrigation share: {kp['ratio']:.3f} over 2013-2021 "
              f"(per-year {', '.join(f'{v:.2f}' for v in kp['per_year'].values())}); "
              f"{k1['pilot_definition']['ratio']:.3f} with the pilot's 2012-2021 (2012 lacks "
              "January in the model). The member keeps the plan's 2.58 as declared.", "",
              "## Deviations from the plan", ""]
    lines += [f"- {x}" for x in deviations]
    lines += ["", "## Issues for downstream", ""]
    lines += [f"- {x}" for x in issues]
    lines += ["", "## Key numbers", ""]
    lines += [f"- {x}" for x in key_numbers]
    lines += ["", "## Evidence", "",
              "- Product: `results/dataprep/public_wells/v2/` (MANIFEST.json, PRODUCT_CARD.md, "
              "validation.json); v1 superseded (same data, card error).",
              "- Evidence: `results/dataprep/reports/P6_evidence/` (text_layer/, api_cache/, "
              "finalize.json, checker_product.json, after_the_fact_n1_janmay.json, "
              "failed_build/).",
              f"- Pre-registration: `results/dataprep/preregistration/{K.PREREG_ID}.json`.",
              "- Ledger: `results/dataprep/ledger/rows/P6.csv`.", ""]
    md = "\n".join(lines)

    rows = []
    for nid, srcs in LEDGER.items():
        for src, role in srcs:
            rows.append({"number_id": nid, "source": src, "role": role, "product_version": PV,
                         "note": "WP P6 report number"})
    return md, js, rows
