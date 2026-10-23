"""Public-well inventory of WP P6: report facts, closures, reconciliation (checks R1, R2).

``facts()`` lists every printed quantity P6 takes from the reports, each with its source file,
PDF page, the quote as printed (checked whitespace-insensitively on that page's text layer),
the extraction route and the publication vintage. ``closures()`` is the cohort table
(annex table 2-2: agency column x year 100-108, counts and abstraction cuts) plus the planned
Changhua Taiwan Water disposals (109-113). ``reconcile()`` evaluates R1 (the plan's
acceptance) and the R2 consistency checks.

Only pre-origin publications enter ``inventory.csv`` and ``closures.csv`` (U3): the MOEA 2019
annex, the 2020 first-revision and phase-2 plans, Control Yuan 53001 (2015) and the drought-
well report's own count table (published 2023-04; a declared exemption, like its locations).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p6_public_wells import p6const as K  # noqa: E402
from p6_public_wells import p6lib as L  # noqa: E402

# (id, category, agency, county, quantity, value, unit, year_start, year_end, basis, src, quote)
FACTS = [
    # ---- MOEA 2019 annex (第二次修正 附冊, 2019-10)
    ("A01", "programme_total", "all_public", "both", "public deep wells disposed (filled, "
     "stopped or reduced)", 774, "wells", 2011, 2018, "reported_result", "moea_annex",
     "已處置共計 774 口水井，達成率約 98.3%"),
    ("A02", "programme_total", "all_public", "both", "abstraction cut of the disposed public "
     "wells", 8793.9, "1e4 m3/yr", 2011, 2018, "reported_result", "moea_annex",
     "減抽地下水量達 8,793.9 萬噸/年，達成率約 99.4%"),
    ("A03", "programme_target", "all_public", "both", "2018 target of the public-well "
     "abstraction cut (table 1-1)", 8850.5, "1e4 m3/yr", 2018, 2018, "target", "moea_annex",
     "公有水井處置 減抽水量 8,793.9 8,850.5 99.4"),
    ("A04", "programme_total", "taiwan_water", "yunlin", "Taiwan Water wells delayed by the "
     "Hushan supply, disposed by March 2019", 13, "wells", 2019, 2019, "reported_result",
     "moea_annex", "其中台水公司 13 口水井未能依原規劃期程辦理水井處置作業"),
    # ---- phase-2 main plan (2020-12)
    ("B01", "programme_total", "all_public", "both", "phase-1 controlled public wells",
     1185, "wells", 2011, 2024, "plan", "plan_phase2",
     "第一期計畫處置(減抽、停用或填塞)公有水井計有 1,185 口"),
    ("B02", "programme_total", "all_public", "both", "public wells disposed 100-108", 1056,
     "wells", 2011, 2019, "reported_result", "plan_phase2", "處置水井數共計1,056口"),
    ("B03", "programme_total", "all_public", "both", "abstraction cut of the wells disposed "
     "100-108 (text of annex 2; table 2-2 prints 13,094.85)", 13098.45, "1e4 m3/yr", 2011,
     2019, "reported_result", "plan_phase2", "減抽地下水使用量約13,098.45萬噸/年"),
    ("B04", "programme_target", "all_public", "both", "wells planned to be disposed by 108",
     943, "wells", 2011, 2019, "target", "plan_phase2", "第一期至108年底預定處置水井數為943口"),
    ("B05", "programme_target", "all_public", "both", "abstraction cut planned by 108",
     11810.58, "1e4 m3/yr", 2011, 2019, "target", "plan_phase2", "預定減抽水量11,810.58萬噸/年"),
    ("B06", "pending", "taiwan_water", "changhua", "Changhua Taiwan Water wells left for the "
     "Niaozuitan lake (44 to stop, 85 to reduce)", 129, "wells", 2020, 2024, "plan",
     "plan_phase2", "彰化地區餘129口水井(停用44口、減抽85口)配合鳥嘴潭完工期程"),
    ("B07", "non_phase1", "taiwan_water", "both", "Taiwan Water wells outside phase-1 "
     "control (Changhua 46, Yunlin 2)", 48, "wells", 2020, 2020, "inventory", "plan_phase2",
     "台水公司計有48口非第一期計畫控管水井(彰化46口、雲林2口)"),
    ("B08", "non_phase1", "taiwan_sugar", "both", "Taiwan Sugar wells outside phase-1 "
     "control (30 in 溪湖/溪州/二林, 6 in 虎尾/土庫)", 111, "wells", 2020, 2020, "inventory",
     "plan_phase2", "台糖公司計有111口非第一期計畫控管水井"),
    ("B09", "non_phase1", "agencies_schools", "changhua", "Changhua agency and school wells "
     "outside phase-1 control (20 near the HSR, 14 in subsidence townships)", 277, "wells",
     2020, 2020, "inventory", "plan_phase2", "彰化縣計有277口非第一期計畫控管水井"),
    ("B10", "non_phase1", "agencies_schools", "yunlin", "Yunlin agency and school wells "
     "outside phase-1 control (13 near the HSR, 26 in subsidence townships)", 82, "wells",
     2020, 2020, "inventory", "plan_phase2", "雲林縣計有82口非第一期計畫控管水井"),
    ("B11", "status", "irrigation_association", "both", "reduced (not closed) irrigation-"
     "association wells to keep under control", 318, "wells", 2019, 2019, "inventory",
     "plan_phase2", "其中318口減抽處置水井，應持續控管"),
    ("B12", "status", "irrigation_association", "yunlin", "reduced irrigation-association "
     "wells in subsidence townships (虎尾 5, 大埤 32)", 37, "wells", 2019, 2019, "inventory",
     "plan_phase2", "另雲林農田水利會計有37口減抽水井位於下陷顯著區(虎尾5口、大埤32口)"),
    ("B13", "programme_total", "taiwan_water", "both", "abstraction cut of the 215 Taiwan "
     "Water wells disposed by 108", 6790.8, "1e4 m3/yr", 2011, 2019, "reported_result",
     "plan_phase2", "水井處置後(215 口)減抽量達約 6,790.8 萬噸/年"),
    ("B14", "other_cut", "taiwan_water", "yunlin", "Tuku-Huwei groundwater reduction measure "
     "of 102 (not a counted well disposal; table 2-9)", 475, "1e4 m3/yr", 2013, 2013,
     "reported_result", "plan_phase2", "102 年辦理土庫虎尾地區地下水減抽配套措施，可減抽地下水 475 萬噸/年"),
    ("B15", "status", "taiwan_sugar", "both", "Taiwan Sugar phase-1 wells (8 filled, 4 "
     "reduced)", 12, "wells", 2011, 2015, "reported_result", "plan_phase2",
     "第一期計畫台糖公司已完成12口(8口填塞及4口減抽)水井處置作業"),
    # ---- first revision main plan (2020-08)
    ("C01", "baseline_supply", "taiwan_water", "both", "Taiwan Water groundwater supply in "
     "Yunlin-Changhua (Changhua 1.03, Yunlin 0.46; baseline of the 2011 plan)", 1.49,
     "1e8 m3/yr", 2011, 2011, "inventory", "plan_rev1", "地下水供應每年 1.49 億噸"),
    ("C02", "baseline_supply", "taiwan_water", "changhua", "Changhua Taiwan Water groundwater "
     "supply (28 x 10^4 m3/day)", 1.03, "1e8 m3/yr", 2011, 2011, "inventory", "plan_rev1",
     "其餘每日 28 萬噸均由台水公司抽取地下水供應(約為每年 1.03 億噸)"),
    ("C03", "baseline_supply", "taiwan_water", "yunlin", "Yunlin Taiwan Water groundwater "
     "supply (12.6 x 10^4 m3/day)", 0.46, "1e8 m3/yr", 2011, 2011, "inventory", "plan_rev1",
     "則由抽取地下水供應(約為每年 0.46 億噸)"),
    ("C04", "baseline_supply", "private_self_supply", "both", "groundwater self-supplied "
     "outside the Taiwan Water service area (domestic and industrial)", 3.18, "1e8 m3/yr",
     2011, 2011, "inventory", "plan_rev1", "另每年 3.18 億噸則為抽取地下水"),
    ("C05", "context_pre2011", "taiwan_water_taiwan_sugar", "yunlin", "HSR 3-km public legal "
     "wells programme (completed 97): wells filled (also 28 stopped, 16 kept; cut 583 x 10^4 "
     "m3/yr)", 43, "wells", 2008, 2010, "reported_result", "plan_rev1",
     "迄今計已填塞合法水井 43 口，封閉停用 28 口，留置續用 16 口，年減抽地下水量約 583 萬噸"),
    ("C06", "context_pre2011", "public", "changhua", "Dacheng public legal wells programme "
     "(completed 98): wells filled (cut 94.8 x 10^4 m3/yr)", 22, "wells", 2009, 2009,
     "reported_result", "plan_rev1", "計填塞合法水井 22 口"),
    ("C07", "programme_total", "taiwan_sugar", "both", "Taiwan Sugar wells 100-104: filled "
     "5, reduced 4, stopped 3 (cut 0.06 x 10^8 m3/yr)", 12, "wells", 2011, 2015, "plan",
     "plan_rev1", "100-104 年封填 5 口、減抽 4 口、停用 3 口"),
    # ---- Control Yuan 53001 (2015)
    ("D01", "programme_total", "all_public", "both", "wells under the action plan's project "
     "control (Taiwan Water 350, irrigation associations 590, Taiwan Sugar 12, others 239)",
     1191, "wells", 2011, 2020, "plan", "cy53001", "專案控管處置水井計有 1,191 口"),
    ("D02", "programme_total", "taiwan_water", "both", "Taiwan Water wells under control "
     "(Yunlin 186: 23 in 103, 163 by 106 with Hushan; Changhua 164: 12 in 102, 152 with "
     "Niaozuitan)", 350, "wells", 2013, 2020, "plan", "cy53001", "台灣自來水公司水井(350 口)"),
    ("D03", "programme_total", "irrigation_association", "both", "irrigation-association "
     "wells in three phases (phase 1: 24 along the HSR)", 590, "wells", 2011, 2020, "plan",
     "cy53001", "彰化、雲林農田水利會水井(590 口)"),
    ("D04", "programme_total", "others", "both", "other wells (prison 2, agencies and schools "
     "230, HSR factories 7)", 239, "wells", 2011, 2020, "plan", "cy53001", "其他水井(239 口)"),
    ("D05", "non_phase1", "public_legal", "both", "other legal wells outside project control",
     1140, "wells", 2015, 2015, "inventory", "cy53001", "為 1,140 口"),
    ("D06", "depth", "irrigation_association", "yunlin", "depth of the 27 Yunlin "
     "irrigation-association wells filled near the HSR (aquifer 2; m, lower bound of "
     "120-150)", 120, "m", 2014, 2014, "inventory", "cy53001",
     "封 填 之 水 井 深 度 大 都 在 120~150 公 尺 (含 水 層 2)"),
    ("D07", "depth", "taiwan_water", "yunlin", "depth of the 10 Tuku waterworks deep wells "
     "filled by Taiwan Water (m, lower bound of 244-301; main compaction 220-300 m)", 244,
     "m", 2014, 2014, "inventory", "cy53001", "水井深度約介於 244~301 公尺"),
]


def _vintage_ok(src: str) -> bool:
    return K.VINTAGE.get(src, "9999-12") <= K.ORIGIN


def facts() -> pd.DataFrame:
    rows = []
    for (fid, cat, ag, county, qty, val, unit, y0, y1, basis, src, quote) in FACTS:
        page = L.find_page(src, quote)
        rows.append({"fact_id": fid, "category": cat, "agency": ag, "county": county,
                     "quantity": qty, "value": float(val), "unit": unit, "year_start": y0,
                     "year_end": y1, "basis": basis, "source_id": src,
                     "source_file": K.SRC[src].name, "pdf_page": page, "quote": quote,
                     "quote_found": bool(page and L.find_quote(src, page, quote)),
                     "extraction": "text_layer", "vintage": K.VINTAGE[src],
                     "pre_origin_publication": _vintage_ok(src)})
    return pd.DataFrame(rows)


def table_rows(t45: dict, t23: dict, dwc: dict) -> pd.DataFrame:
    """Inventory rows from table 4-5 (disposal types), annex table 2-3 (targets, execution)
    and the drought-well table 3 counts."""
    rows = []
    src45, p45 = t45["src"], t45["page"]
    for r in t45["rows"]:
        if r["agency"] == "total":
            continue
        for part, col, y0, y1, basis in (("actual", "actual_100_108", 2011, 2019,
                                          "reported_result"),
                                         ("planned", "planned_109_113", 2020, 2024, "plan")):
            rows.append({"fact_id": f"T45.{r['agency']}.{r['county']}.{r['type']}.{part}",
                         "category": "disposal_type", "agency": r["agency"],
                         "county": r["county"], "quantity": f"wells {r['type']} ({part})",
                         "value": float(r[col]), "unit": "wells", "year_start": y0,
                         "year_end": y1, "basis": basis, "source_id": src45,
                         "source_file": K.SRC[src45].name, "pdf_page": p45,
                         "quote": f"表4-5: {r['line']}", "quote_found": True,
                         "extraction": "text_layer_table", "vintage": K.VINTAGE[src45],
                         "pre_origin_publication": _vintage_ok(src45)})
    for part, unit in (("count", "wells"), ("volume", "1e4 m3/yr")):
        for key, r in t23[part].items():
            for col, basis in (("full_target", "target"), ("target_to_108", "target"),
                               ("executed_to_108", "reported_result")):
                rows.append({"fact_id": f"T23.{key}.{part}.{col}", "category":
                             "agency_total", "agency": key, "county": key.split("_")[-1]
                             if key != "total" else "both", "quantity": f"{col} ({part})",
                             "value": float(r[col]), "unit": unit, "year_start": 2011,
                             "year_end": 2024 if col == "full_target" else 2019,
                             "basis": basis, "source_id": t23["src"],
                             "source_file": K.SRC[t23["src"]].name, "pdf_page": t23["page"],
                             "quote": f"附表2-3: {r['line']}", "quote_found": True,
                             "extraction": "text_layer_table",
                             "vintage": K.VINTAGE[t23["src"]],
                             "pre_origin_publication": _vintage_ok(t23["src"])})
    dw_quote = {"Changhua": "彰化縣 49 0 0 0 0 26 23 0 0",
                "Yunlin": "雲林縣 521 0 0 2 0 421 98 0 0"}
    dw_page = L.find_page("drought_wells", dw_quote["Yunlin"])
    for county, chk in dwc.items():
        for ag, n in chk["by_agency_stated_table3"].items():
            rows.append({"fact_id": f"DW.{county}.{ag[:6]}", "category": "drought_wells",
                         "agency": ag, "county": county.lower(), "quantity":
                         "drought wells 2022 (table 3; printed rows "
                         f"{chk['by_agency_parsed'].get(ag)})", "value": float(n),
                         "unit": "wells", "year_start": 2022, "year_end": 2022,
                         "basis": "inventory", "source_id": "drought_wells",
                         "source_file": K.SRC["drought_wells"].name, "pdf_page": dw_page,
                         "quote": dw_quote[county], "quote_found": bool(
                             dw_page and L.find_quote("drought_wells", dw_page,
                                                      dw_quote[county])),
                         "extraction": "text_layer_table", "vintage": K.VINTAGE["drought_wells"],
                         "pre_origin_publication": False})
    return pd.DataFrame(rows)


def closures(t22: dict, t23: dict) -> pd.DataFrame:
    """Cohort table: one row per agency column x year 100-108 with a printed count or volume,
    plus the planned Changhua Taiwan Water disposals 109-113."""
    cr, vr = t22["counts"]["rows"], t22["volumes"]["rows"]
    rows = []
    for roc in K.YEARS_ROC:
        lab = f"{roc}年"
        for j, col in enumerate(K.COHORT_COLUMNS):
            n, v = cr[lab]["values"][j], vr[lab]["values"][j]
            note = vr[lab]["notes"].get(col["key"], "")
            if n is None and v is None and not note:
                continue
            rows.append({"cohort": f"{col['key']}_{roc + 1911}", "agency": col["agency"],
                         "county": col["county"], "column_label": col["label"],
                         "year_roc": roc, "year": roc + 1911, "status": "disposed",
                         "n_wells": n, "volume_1e4m3_per_yr": v, "note": note,
                         "source_file": K.SRC["plan_phase2"].name, "pdf_page": t22["page"],
                         "table": "附表2-2", "extraction": "api_two_reads+text_layer",
                         "flag": 0, "method": "transcribed_report_table"})
    full = t23["volume"]["tw_changhua"]["full_target"]
    done = t23["volume"]["tw_changhua"]["executed_to_108"]
    rows.append({"cohort": "tw_changhua_pending_2020_2024", "agency": "taiwan_water",
                 "county": "changhua", "column_label": "自來水公司(11區)", "year_roc": 109,
                 "year": 2020, "status": "planned_109_113 (Niaozuitan lake)",
                 "n_wells": float(K.PENDING_TW_CH_WELLS), "volume_1e4m3_per_yr": full - done,
                 "note": "表4-5 planned 109-113 (stop 44, reduce 85); volume = 附表2-3 full "
                         "target 7,172.25 - executed to 108 1,842.3",
                 "source_file": K.SRC["plan_phase2"].name, "pdf_page": t23["page"],
                 "table": "表4-5, 附表2-3", "extraction": "text_layer_table", "flag": 5,
                 "method": "difference_of_printed_target_and_execution"})
    out = pd.DataFrame(rows)
    out["volume_Mm3_per_yr"] = out.volume_1e4m3_per_yr / 100.0
    return out


def reconcile(t22: dict, t45: dict, t23: dict, t25: dict, t29: dict | None = None) -> dict:
    cr, vr = t22["counts"]["rows"], t22["volumes"]["rows"]
    ann = [f"{y}年" for y in K.ANNEX_YEARS_ROC]
    n_rowtot = sum(cr[lab]["total"] for lab in ann)
    n_cells = sum(sum(v or 0 for v in cr[lab]["values"]) for lab in ann)
    v_cells = sum(sum(v or 0 for v in vr[lab]["values"]) for lab in ann)
    v_rowtot = sum(vr[lab]["total"] for lab in ann)
    rel_n = abs(n_rowtot - K.ANNEX_WELLS) / K.ANNEX_WELLS
    rel_v = abs(v_cells - K.ANNEX_VOLUME_1E4) / K.ANNEX_VOLUME_1E4
    r1 = {"N_printed_row_totals_100_107": n_rowtot, "N_prime_cells_100_107": n_cells,
          "V_cells_100_107": round(v_cells, 4), "V_printed_row_totals_100_107":
          round(v_rowtot, 4), "annex_wells": K.ANNEX_WELLS,
          "annex_volume_1e4m3": K.ANNEX_VOLUME_1E4, "rel_diff_wells": rel_n,
          "rel_diff_volume": rel_v, "threshold": K.RECONCILE_TOL,
          "passed": bool(rel_n <= K.RECONCILE_TOL and rel_v <= K.RECONCILE_TOL),
          "N_vs_Nprime_cause": "105 (2016) Yunlin Taiwan Sugar: the volume cell prints "
                               "'(100年1口停用改封填)': a well stopped in 100 and filled in 105 "
                               "is counted in both years' cells but once in the row and "
                               "column totals (row 105: cells 73, printed 72; Yunlin Taiwan "
                               "Sugar column: cells 8, printed 7)"}
    # R2: phase-1 totals and cross-tables
    allr = [f"{y}年" for y in K.YEARS_ROC]
    tot_c = sum(cr[lab]["total"] for lab in allr)
    tot_v = sum(sum(v or 0 for v in vr[lab]["values"]) for lab in allr)
    r2 = {"phase1_wells_row_totals": tot_c, "phase1_wells_printed": cr["合計"]["total"],
          "phase1_volume_cells": round(tot_v, 4), "phase1_volume_printed_table":
          vr["合計"]["total"], "phase1_volume_printed_text": 13098.45,
          "text_vs_table_volume_diff": round(13098.45 - vr["合計"]["total"], 4)}
    # per agency/county vs table 4-5 actual and 2-3 executed
    t45_tot = {}
    for r in t45["rows"]:
        t45_tot.setdefault((r["agency"], r["county"]), 0)
        t45_tot[(r["agency"], r["county"])] += r["actual_100_108"]
    per = []
    group = {"taiwan_water": "taiwan_water", "irrigation_association":
             "irrigation_association", "taiwan_sugar": "taiwan_sugar", "factory":
             "schools_factories_prison", "school": "schools_factories_prison", "prison":
             "schools_factories_prison"}
    agg = {}
    for j, col in enumerate(K.COHORT_COLUMNS):
        c = cr["合計"]["values"][j]
        v = vr["合計"]["values"][j]
        ex = t23["count"][col["key"]]["executed_to_108"]
        exv = t23["volume"][col["key"]]["executed_to_108"]
        k = (group[col["agency"]], col["county"])
        agg[k] = agg.get(k, 0) + c
        per.append({"column": col["key"], "t22_count_total": c, "t23_count_executed": ex,
                    "t22_volume_total": v, "t23_volume_executed": exv,
                    "count_equal": c == ex, "volume_abs_diff": round(abs(v - exv), 4)})
    t45_cmp = [{"block": "/".join(k), "t22_sum": agg[k], "t45_actual": t45_tot.get(k),
                "equal": agg[k] == t45_tot.get(k)} for k in sorted(agg)]
    # table 2-5 (irrigation associations and Taiwan Sugar by year)
    rows25 = t25["rows"]
    t25_cmp = []
    for roc in K.YEARS_ROC:
        lab = str(roc)
        r = rows25.get(lab)
        if r is None:
            continue
        j_ch = [c["key"] for c in K.COHORT_COLUMNS].index("ia_changhua")
        j_yl = [c["key"] for c in K.COHORT_COLUMNS].index("ia_yunlin")
        ch = cr[f"{roc}年"]["values"][j_ch] or 0
        yl = cr[f"{roc}年"]["values"][j_yl] or 0
        t25_cmp.append({"year_roc": roc, "t22_ia_changhua": ch, "t25_ia_changhua": r[0] or 0,
                        "t22_ia_yunlin": yl, "t25_ia_yunlin": r[1] or 0,
                        "equal": (ch == (r[0] or 0)) and (yl == (r[1] or 0))})
    ia_vol_22 = sum((vr[f"{y}年"]["values"][j_ch] or 0) + (vr[f"{y}年"]["values"][j_yl] or 0)
                    for y in K.YEARS_ROC)
    ts_vol_22 = sum((vr[f"{y}年"]["values"][2] or 0) + (vr[f"{y}年"]["values"][7] or 0)
                    for y in K.YEARS_ROC)
    tot25 = rows25["合計"]
    r2.update({"per_column_vs_t23": per, "t22_vs_t45_actual": t45_cmp,
               "t22_vs_t45_all_equal": all(x["equal"] for x in t45_cmp),
               "t22_vs_t23_counts_all_equal": all(x["count_equal"] for x in per),
               "t22_vs_t23_volume_max_abs_diff": max(x["volume_abs_diff"] for x in per),
               "t22_vs_t25_ia_counts_by_year": t25_cmp,
               "t22_vs_t25_ia_counts_all_equal": all(x["equal"] for x in t25_cmp),
               "ia_volume_t22": round(ia_vol_22, 4), "ia_volume_t25": tot25[3],
               "ts_volume_t22": round(ts_vol_22, 4), "ts_volume_t25": tot25[7],
               "ts_wells_t25": tot25[6],
               "control_yuan_vs_phase2": {
                   "cy_total": 1191, "phase2_total": 1185,
                   "taiwan_water": {"cy": 350, "phase2": 158 + 186,
                                    "cy_changhua": 164, "phase2_changhua": 158},
                   "irrigation_association": {"cy": 590, "phase2": 51 + 539},
                   "taiwan_sugar": {"cy": 12, "phase2": 5 + 7},
                   "others": {"cy": 239, "phase2": 5 + 97 + 2 + 133 + 2}}})
    if t29 is not None:
        r2["t22_vs_t29_taiwan_water"] = t29
    return {"R1": r1, "R2": r2}
