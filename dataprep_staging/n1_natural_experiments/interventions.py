"""The N1 intervention table: every row has a source, a page and a verification status.

Rows come from four kinds of evidence:

* ``text_layer``: a statement printed in an L0 report; its quote must occur on the cited page
  of the PDF's text layer (checked by :func:`parse_sources.find_quote`, whitespace ignored).
  Statements of plans (the Control Yuan report of 2015-05-19 describes schedules) are
  ``basis = planned``; statements of results are ``reported_achieved`` (agency-reported, not
  re-measured here).
* ``text_layer_table``: a number parsed from a printed table (Jiji Jan-May supply, 110 briefing
  p. 20; printed public-well volumes, pp. 10 and 19; drought-well tables of the 2022 report).
* ``chart_two_reads``: public-well Jan-May volumes read off the briefing charts by two models
  (``merge_charts``); every point agreed within 2 % of the axis.
* ``product``: computed from a gated product (C0 ``pump_electricity/preorigin/v3``).

Rows dated 2023 or later are kept for the record with ``pre_origin = False``; no N1 test, gate
or threshold uses them (U7). Quotes are in Traditional Chinese as printed; translations in
``intervention`` are this WP's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import rel_to_repo  # noqa: E402

from n1_natural_experiments import census_events as CE  # noqa: E402
from n1_natural_experiments import merge_charts as MC  # noqa: E402
from n1_natural_experiments import n1const as K  # noqa: E402
from n1_natural_experiments import parse_sources as P  # noqa: E402

COLUMNS = ["row_id", "category", "intervention", "county", "location", "township",
           "x_twd97", "y_twd97", "date_start", "date_end", "quantity", "unit", "basis",
           "source", "source_page", "quote", "extraction", "verification", "pre_origin",
           "used_by_tests", "identifiable", "notes"]

# --------------------------------------------------------------------------- printed rows
# (row_id, category, intervention, county, location, township, date_start, date_end,
#  quantity, unit, basis, src, page, quote, used_by, identifiable, notes)
TEXT_ROWS = [
    ("M01", "public_well_closure",
     "Public deep wells disposed of (filled, stopped or reduced) by Taiwan Water, Taiwan Sugar, "
     "irrigation associations and county governments, 2011-2018; agency-reported abstraction "
     "cut", "Changhua+Yunlin", "not located (no coordinates printed)", "", "2011-01",
     "2018-12", 87.939, "Mm3/yr (774 wells)", "reported_achieved", "moea_annex", 7,
     "已處置共計774口水井，達成率約98.3%，減抽地下水量達8,793.9萬噸/年",
     "N1.T3_pw_step2019 (Yunlin Taiwan Water part, located by drought wells); N1.G5",
     "partly: only the Yunlin Taiwan Water share has a located proxy (2022 drought wells) and "
     "a Jan-May volume series (charts); the rest has neither locations nor volumes",
     "target 8,850.5; 774 wells is 98.3 % of the plan (Table 1-1, p. 6)"),
    ("M02", "public_well_closure",
     "13 Taiwan Water wells whose disposal waited for Hushan Reservoir supply, completed "
     "March 2019", "Yunlin", "not located", "", "2019-03", "2019-03", 13, "wells",
     "reported_achieved", "moea_annex", 7,
     "其中台水公司13口水井未能依原規劃期程辦理水井處置作業，進度落後原因係湖山水庫供水期程延後，"
     "已於108年3月完成處置", "N1.T3_pw_step2019", "as part of the 2019 Yunlin step (B05)", ""),
    ("M03", "irrigation_pumping_cut",
     "Irrigation groundwater pumping cut (2018 achievement of the action plan)",
     "Changhua+Yunlin", "not located", "", "2011-01", "2018-12", 228.958, "Mm3/yr",
     "reported_achieved", "moea_annex", 6, "減抽水量約2.29億噸/年，已達107年目標值", "",
     "no: no location, no time path; metered census kWh carries whatever part is real",
     "Table 1-1 prints 22,895.8 萬噸/年 against a target of 20,800; the 8,000 ha "
     "dry-crop/afforestation item reached 57.5 %"),
    ("M04", "aquaculture_livestock_cut",
     "Aquaculture and livestock groundwater pumping cut (2018)", "Changhua+Yunlin",
     "not located", "", "2011-01", "2018-12", 24.063, "Mm3/yr", "reported_achieved",
     "moea_annex", 6, "養殖及畜牧用水減抽：至107年實際成果為2,406.3萬噸/年", "",
     "no: no location or time path", "Table 1-1 prints 2,406.2 (0.1 萬噸 below the text)"),
    ("M05", "public_supply_cut",
     "Public supply groundwater cut including water saving (2018)", "Changhua+Yunlin",
     "not located", "", "2011-01", "2018-12", 62.479, "Mm3/yr", "reported_achieved",
     "moea_annex", 6, "6,247.96,49798.9", "", "no", "Table 1-1 row; target 64.97, 98.9 %"),
    ("M06", "surface_supply_switch",
     "Hushan Reservoir completed January 2016 (treatment plant December 2016, full level June "
     "2019); target: Yunlin tap water draws 46 Mm3/yr less groundwater", "Yunlin",
     "Hushan Reservoir and its treatment plant (not in the model grid)", "", "2016-01",
     "2019-06", 46.0, "Mm3/yr (target)", "planned", "moea_annex", 6,
     "湖山水庫已於105年1月興建完成", "N1.T3_pw_step2019",
     "via the Yunlin Taiwan Water public-well step of 2019 (B05)",
     "full level: '108年6月達達滿水位目標'; target quote: '將陸續達成雲林地區自來水減抽地下水0.46億噸/年目標'"),
    ("M07", "surface_supply_switch",
     "Niaozuitan artificial lake (Changhua supply) expected 2022; downstream treatment plant "
     "end of 2022, whole scheme 2024", "Changhua", "not in the model grid", "", "2022-12",
     "2024-12", None, "", "planned", "moea_annex", 6,
     "鳥嘴潭人工湖工程業於107年6月25日完成用地取得，進度略落後，趕工中，預計111年完成", "",
     "no: after the record used here", "152 Changhua Taiwan Water wells wait for it (C04)"),
    ("M08", "managed_recharge_stop",
     "River-bed recharge at 78 sections of the Choushui River, 2011-2015, about 26.4 Mm3/yr; "
     "operation ended at the end of 2015", "Changhua+Yunlin (river)", "Choushui River bed, "
     "78 sections (not located)", "", "2011-01", "2015-12", 26.4, "Mm3/yr (average)",
     "reported_achieved", "moea_annex", 7,
     "100~104年於濁水溪斷面78處推動河槽地下水補注工作，平均每年地下水補注量約2,640萬噸", "",
     "no: the deliverable has no river cells (sw_recharge none, rivers none) and the sections "
     "are not located", "end: '自104年底結束操作後即暫緩推動'"),
    ("M09", "illegal_well_fill", "Illegal wells filled by end 2018", "Changhua+Yunlin",
     "not located", "", "2011-01", "2018-12", 2403, "wells (Changhua 1,050; Yunlin 1,353)",
     "reported_achieved", "moea_annex", 9,
     "總計填塞違法水井2,403口，其中彰化縣1,050口；雲林縣1,353口", "",
     "no: no locations, depths or volumes; mostly shallow farm wells", ""),
    ("M10", "context",
     "Registered wells tagged as producing after the 2013-2016 re-survey (納管水井 tagged)",
     "Changhua+Yunlin", "", "", "2016-12", "2018-12", 256810,
     "wells (Yunlin 126,396; Changhua 130,414)", "reported_achieved", "moea_annex", 8,
     "可出水並貼標籤共計25萬6,810口(雲林126,396口、彰化130,414口)", "", "context only", ""),
    ("M11", "crop_conversion",
     "Golden Corridor programme: crop-1 rice to dry crops within 1.5 km of the HSR south of "
     "the New Huwei River (2013-2020); extended in 2020 to four townships",
     "Yunlin", "HSR corridor; 2020: 虎尾, 土庫, 元長, 北港", "虎尾;土庫;元長;北港", "2013-01",
     "2020-12", None, "", "planned", "moea_annex", 16,
     "109年擴大範圍為新虎尾溪以南地層下陷需加強關注之4鄉鎮(虎尾、土庫、元長及北港)", "",
     "no separate test: the metered census kWh already carries any pumping change; the "
     "programme's area and timing per field are not public", ""),
    ("M12", "context", "Plain afforestation stopped in 2013", "Changhua+Yunlin", "", "",
     "2013-01", "2013-12", None, "", "reported_achieved", "moea_annex", 16,
     "平地造林自102年決策停辦", "", "context only", ""),
    ("C01", "crop_conversion",
     "Golden Corridor target: 2,000 ha of crop-1 rice to water-saving crops, expected "
     "groundwater cut 24 Mm3", "Yunlin", "HSR corridor", "", "2013-01", "2020-12", 24.0,
     "Mm3 (expected)", "planned", "cy53001", 5,
     "輔導廊道內2,000公頃一期作水稻田轉為省水耐旱作物，預計可減抽地下水2,400萬噸", "",
     "no (as M11)", "Control Yuan investigation report dated 2015-05-19"),
    ("C02", "public_well_closure",
     "Wells under the action plan's project control: 1,191 (Taiwan Water 350, irrigation "
     "associations 590, Taiwan Sugar 12, others 239)", "Changhua+Yunlin", "not located", "",
     "2011-08", "2020-12", 1191, "wells", "planned", "cy53001", 11,
     "項下專案控管處置水井計有1,191口", "", "partly (as M01)",
     "the plan's 774 wells disposed by 2018 (M01) are a subset"),
    ("C03", "public_well_closure",
     "Taiwan Water, Yunlin: 186 wells; 23 cut or stopped in 2014 (agricultural water "
     "transferred), the other 163 to be cut, stopped or filled in 2017 once Hushan supplies",
     "Yunlin", "not located", "", "2014-01", "2017-12", 186, "wells", "planned", "cy53001",
     11, "雲林地區186口，其中23口以由農田水利會移用農業用水替代於103年完成減抽、停用",
     "N1.T3_pw_step2019", "the charts (B04) put the large Yunlin step in 2019, not 2017",
     "p. 12 continues: '於湖山水庫完工供水，106年完成減抽、停用、填塞等處置作業'"),
    ("C04", "public_well_closure",
     "Taiwan Water, Changhua: 164 wells; 12 cut or stopped in 2013 (supply from Taichung and "
     "Linnei), the other 152 with the Niaozuitan lake", "Changhua", "not located", "",
     "2013-01", "2024-12", 164, "wells", "planned", "cy53001", 12,
     "彰化地區164口，其中12口以自來水公司增加台中及林內支援量替代於102年完成減抽、停用", "",
     "no CPU test: the Changhua Taiwan Water Jan-May volume falls gradually (B04) and its "
     "wells are not located beyond the 23 drought wells", ""),
    ("C05", "public_well_closure",
     "Irrigation-association wells: 590 in three phases; phase 1 = 24 Yunlin wells along the "
     "HSR", "Changhua+Yunlin", "HSR line (phase 1)", "", "2011-08", "2020-12", 590, "wells",
     "planned", "cy53001", 12, "第一階段優先處置雲林農田水利會位於高鐵沿線水井，計24口", "",
     "no", ""),
    ("C06", "public_well_closure", "Taiwan Sugar wells: 12, all filled in 2015",
     "Changhua+Yunlin", "not located", "", "2015-01", "2015-12", 12, "wells", "planned",
     "cy53001", 12, "於104年悉予填塞", "", "no", ""),
    ("C07", "public_well_closure",
     "Other wells: 239 (prison 2, public agencies and schools 230, HSR-line factories 7), "
     "filled after connection to tap water", "Changhua+Yunlin", "not located", "", "2011-08",
     "2020-12", 239, "wells", "planned", "cy53001", 12, "其他水井(239口)", "", "no", ""),
    ("C08", "public_well_closure",
     "Yunlin irrigation association filled 27 wells (120-150 m, aquifer 2) within 3 km of the "
     "HSR; no clear slowing of subsidence at the benchmarks around them (as of 2015)",
     "Yunlin", "within 3 km of the HSR (not located)", "", "2011-08", "2015-05", 27, "wells",
     "reported_achieved", "cy53001", 6,
     "該水利會封填27口水井後，其範圍內之水準樁下陷量無明顯減緩趨勢", "",
     "no: not located; the agency's own reading is reported, not re-tested", ""),
    ("C09", "public_well_closure",
     "Taiwan Water filled 10 deep wells (244-301 m) of the Tuku waterworks; the agency "
     "reports a good effect on subsidence", "Yunlin", "Tuku waterworks", "土庫", "2011-08",
     "2015-05", 10, "wells", "reported_achieved", "cy53001", 6,
     "該公司封填10口土庫鎮自來水廠深水井後，對於減緩地層下陷已有良好成效", "N1.G5",
     "weakly: the chart puts Tuku Taiwan Water Jan-May pumping at 0.54 Mm3 in 2011, 0.17 in "
     "2012 and about 0 from 2013 (B05): a step at the start of the head record (burn-in)",
     ""),
    ("C10", "illegal_well_fill",
     "Yunlin county filled existing illegal wells 2010-2013: 170, 92, 143, 102 (507 in all; "
     "63 inside severe-subsidence areas)", "Yunlin", "not located", "", "2010-01", "2013-12",
     507, "wells", "reported_achieved", "cy53001", 3,
     "表1雲林縣政府99至102年度封填既有違法水井數量表", "", "no", ""),
    ("B01", "drought_context",
     "2021 drought: rain from 2021-05-30 on raised groundwater levels clearly", "Changhua",
     "", "", "2021-05", "2021-05", None, "", "reported_achieved", "brief110", 5,
     "自110年5月30日開始降雨", "N1.T1_ext_E2021", "context: the event window ends in May", ""),
    ("B02", "public_well_closure",
     "Yunlin Taiwan Water: no Jan-May pumping in the significant-subsidence townships Huwei, "
     "Tuku and Yuanchang since 2019", "Yunlin", "虎尾, 土庫, 元長", "虎尾;土庫;元長", "2019-01",
     "2021-05", 0.0, "Mm3 (Jan-May)", "reported_achieved", "brief110", 19,
     "下陷顯著地區(虎尾、土庫及元長)自108年已無抽水", "", "as B05", ""),
    ("B03", "public_well_closure",
     "Changhua irrigation agency: no Jan-May public-well pumping in Xihu, Xizhou and Erlin",
     "Changhua", "溪湖, 溪州, 二林", "溪湖;溪州;二林", "2011-01", "2021-05", 0.0,
     "Mm3 (Jan-May)", "reported_achieved", "brief110", 10,
     "而下陷顯著地區(溪湖、溪州及二林)已無抽水", "",
     "no: the township chart (112 briefing) shows zero in every year 2011-2023, so there is "
     "no change to test", ""),
    ("B06", "drought_pumping",
     "Yunlin public wells pumped more in the 2021 and 2023 droughts, then went back to cuts",
     "Yunlin", "", "", "2021-01", "2023-05", None, "", "reported_achieved", "brief112", 26,
     "惟110及112年因大旱而增抽", "N1.T2_pw_pulse2021", "yes for 2021 (located proxy)",
     "2023 is post-origin: record only"),
    ("D01", "drought_wells",
     "2021 drought (second half of 2020 to first half of 2021): WRA and Taiwan Water drilled "
     "drought wells (programmes 1.0 and 2.0); other agencies switched on their drought wells",
     "Taiwan", "", "", "2020-07", "2021-06", None, "", "reported_achieved", "drought_wells", 34,
     "本署與臺灣自來水公司共同合作執行抗旱水源1.0及2.0計畫以鑿設抗旱水井",
     "N1.T2_pw_pulse2021", "locations from the 2022 inventory (D02, D03)", ""),
]


def _text_rows() -> tuple[list[dict], list[dict]]:
    rows, checks = [], []
    for (rid, cat, what, cty, loc, town, d0, d1, qty, unit, basis, src, page, quote, used,
         ident, notes) in TEXT_ROWS:
        ok = P.find_quote(src, page, quote)
        checks.append({"row_id": rid, "source": src, "page": page, "quote_found": ok})
        rows.append({"row_id": rid, "category": cat, "intervention": what, "county": cty,
                     "location": loc, "township": town, "x_twd97": None, "y_twd97": None,
                     "date_start": d0, "date_end": d1, "quantity": qty, "unit": unit,
                     "basis": basis, "source": rel_to_repo(K.SRC[src]), "source_page": page,
                     "quote": quote, "extraction": "text_layer",
                     "verification": "quote_found_on_page" if ok else "QUOTE_NOT_FOUND",
                     "pre_origin": d1 <= K.ORIGIN, "used_by_tests": used,
                     "identifiable": ident, "notes": notes})
    return rows, checks


def _jiji_rows() -> list[dict]:
    t = P.jiji_janmay_table()
    by = {r["year"]: r for r in t["rows"]}
    ref = {k: sum(by[y][k] for y in (2016, 2017, 2018, 2019)) / 4.0
           for k in ("changhua_irrigation_1e4m3", "yunlin_irrigation_1e4m3")}
    out = []
    for ev, e in K.EVENTS.items():
        y = e["year"]
        for cty, k in (("Changhua", "changhua_irrigation_1e4m3"),
                       ("Yunlin", "yunlin_irrigation_1e4m3")):
            out.append({
                "row_id": f"J{y}{cty[0]}", "category": "delivery_cut",
                "intervention": f"Jiji weir Jan-May irrigation supply {y} ({ev}); reference "
                                f"mean 2016-2019 {ref[k] / 1e4:.2f} x 10^8 m3",
                "county": cty, "location": "irrigation district (Jiji weir supply)",
                "township": "", "x_twd97": None, "y_twd97": None, "date_start": f"{y}-01",
                "date_end": f"{y}-05", "quantity": round(by[y][k] / 1e4, 4),
                "unit": "10^8 m3 (Jan-May)", "basis": "reported_achieved",
                "source": rel_to_repo(K.SRC["brief110"]), "source_page": t["page"],
                "quote": t["quote"], "extraction": "text_layer_table",
                "verification": "parsed_from_printed_table",
                "pre_origin": True, "used_by_tests": f"N1.T1_ext_{ev}; N1.G1-G3",
                "identifiable": "in R0 only through the metered pumping response (the "
                                "deliverable has no canal-delivery term); directly from rung R3 on",
                "notes": f"ratio to the 2016-2019 mean {by[y][k] / ref[k]:.3f}"})
    return out


def _public_well_rows(series: pd.DataFrame) -> list[dict]:
    out = []
    county = series[series.township == "(county)"]
    for (cty, ag), g in county.groupby(["county", "agency"]):
        s = g.set_index("year").value_1e4m3 / 100.0
        txt = ", ".join(f"{y} {s.loc[y]:.2f}" for y in (2011, 2015, 2018, 2019, 2020, 2021, 2022)
                        if y in s.index)
        out.append({
            "row_id": f"B04{cty[0]}{'T' if '台水' in ag else 'I'}", "category": "public_well_pumping",
            "intervention": f"{ag} public-well Jan-May pumping, {cty} (Mm3): {txt}",
            "county": cty, "location": "county", "township": "", "x_twd97": None,
            "y_twd97": None, "date_start": "2011-01", "date_end": "2022-05",
            "quantity": float(s.loc[2021]), "unit": "Mm3 (Jan-May 2021)",
            "basis": "reported_achieved",
            "source": rel_to_repo(K.SRC["brief112"]) + " (+ brief110 for the printed 2020-21)",
            "source_page": 23 if cty == "Changhua" else 26,
            "quote": "彰化公井抽水量" if cty == "Changhua" else "雲林公井抽水量",
            "extraction": "chart_two_reads",
            "verification": "two model reads agree within 2 % of the axis; 2020-21 printed",
            "pre_origin": True,
            "used_by_tests": ("N1.T2_pw_pulse2021; N1.T3_pw_step2019"
                              if (cty == "Yunlin" and "台水" in ag) else ""),
            "identifiable": ("yes (located by the 2022 drought-well inventory)"
                             if (cty == "Yunlin" and "台水" in ag) else
                             "no test: not located beyond the drought-well subset"),
            "notes": "series in public_well_janmay.csv (2011-2023; 2023 post-origin)"})
    # the Yunlin 2019 step and 2021 pulse, stated as rows of their own
    yt = county[(county.county == "Yunlin") & county.agency.str.contains("台水")]
    s = yt.set_index("year").value_1e4m3 / 100.0
    out.append({"row_id": "B05", "category": "public_well_closure",
                "intervention": "Yunlin Taiwan Water public-well Jan-May pumping step: "
                                f"mean 2016-2018 {s.loc[[2016, 2017, 2018]].mean():.2f} Mm3, "
                                f"2019 {s.loc[2019]:.2f}, 2020 {s.loc[2020]:.2f} (Hushan supply)",
                "county": "Yunlin", "location": "98 Taiwan Water drought wells of 2022 as proxy",
                "township": "", "x_twd97": None, "y_twd97": None, "date_start": "2019-01",
                "date_end": "2020-05",
                "quantity": float(s.loc[2019] - s.loc[[2016, 2017, 2018]].mean()),
                "unit": "Mm3 (Jan-May change)", "basis": "reported_achieved",
                "source": rel_to_repo(K.SRC["brief112"]), "source_page": 26,
                "quote": "台水公司枯水期(1~5月)自108年已無抽水", "extraction": "chart_two_reads",
                "verification": "two model reads agree; 2020 printed", "pre_origin": True,
                "used_by_tests": "N1.T3_pw_step2019",
                "identifiable": "yes, with the located proxy and stated assumptions",
                "notes": "the quote refers to the subsidence townships; the county step is "
                         "read off the chart"})
    out.append({"row_id": "B07", "category": "drought_pumping",
                "intervention": "Yunlin Taiwan Water public wells switched back on in the 2021 "
                                f"drought: Jan-May {s.loc[2020]:.2f} (2020, printed) -> "
                                f"{s.loc[2021]:.2f} Mm3 (2021, printed) -> {s.loc[2022]:.2f} "
                                "(2022)",
                "county": "Yunlin", "location": "98 Taiwan Water drought wells of 2022 as proxy",
                "township": "", "x_twd97": None, "y_twd97": None, "date_start": "2021-01",
                "date_end": "2021-05", "quantity": float(s.loc[2021] - s.loc[2020]),
                "unit": "Mm3 (Jan-May change)", "basis": "reported_achieved",
                "source": rel_to_repo(K.SRC["brief110"]), "source_page": 19,
                "quote": "台水公司抽水量109年49萬噸、110年1,193萬噸",
                "extraction": "text_layer_table", "verification": "parsed_from_printed_text",
                "pre_origin": True, "used_by_tests": "N1.T2_pw_pulse2021",
                "identifiable": "yes, with the located proxy and stated assumptions",
                "notes": ""})
    # townships
    town = series[series.township != "(county)"]
    for (cty, ag, tw), g in town.groupby(["county", "agency", "township"]):
        s = g.set_index("year").value_1e4m3 / 100.0
        if float(s.abs().max()) == 0.0:
            desc = "zero in every year 2011-2023"
        else:
            desc = ", ".join(f"{y} {s.loc[y]:.2f}" for y in (2011, 2012, 2013, 2016, 2019, 2021,
                                                              2022) if y in s.index)
        out.append({"row_id": f"B08{cty[0]}{'T' if '台水' in ag else 'I'}_{tw}",
                    "category": "public_well_pumping",
                    "intervention": f"{ag} Jan-May public-well pumping in {tw} (Mm3): {desc}",
                    "county": cty, "location": f"township {tw}", "township": tw,
                    "x_twd97": None, "y_twd97": None, "date_start": "2011-01",
                    "date_end": "2022-05", "quantity": float(s.loc[2022]),
                    "unit": "Mm3 (Jan-May 2022)", "basis": "reported_achieved",
                    "source": rel_to_repo(K.SRC["brief112"]),
                    "source_page": 23 if cty == "Changhua" else 26,
                    "quote": g.chart_title.iloc[0], "extraction": "chart_two_reads",
                    "verification": "two model reads agree within 2 % of the axis",
                    "pre_origin": True, "used_by_tests": "N1.G5" if tw in ("土庫", "元長", "虎尾")
                    else "", "identifiable": "township only (no well coordinates)",
                    "notes": "values on a small axis: digitization sd in public_well_janmay.csv"})
    return out


def _drought_rows(dw: dict) -> list[dict]:
    out = []
    df = pd.DataFrame(dw["rows"])
    for (cty, ag), g in df.groupby(["county", "agency"]):
        towns = g.township.value_counts()
        out.append({"row_id": f"D02{cty[0]}_{'T' if '台水' in ag else ('S' if '台糖' in ag else 'I')}",
                    "category": "drought_wells",
                    "intervention": f"{ag} drought wells in the 2022 inventory, {cty}: "
                                    f"{len(g)} printed rows; townships "
                                    + ", ".join(f"{k} {v}" for k, v in towns.items()),
                    "county": cty, "location": "coordinates per well in drought_wells.csv",
                    "township": ";".join(towns.index), "x_twd97": None, "y_twd97": None,
                    "date_start": "2022-09", "date_end": "2022-09", "quantity": int(len(g)),
                    "unit": "wells", "basis": "reported_achieved",
                    "source": rel_to_repo(K.SRC["drought_wells"]),
                    "source_page": "annex tables 1-8 (Changhua), 1-10 (Yunlin)",
                    "quote": "附表1-8 彰化縣49口抗旱水井鄰近觀測井彙整表" if cty == "Changhua"
                    else "附表1-10 雲林縣521口抗旱水井分布點位彙整表",
                    "extraction": "text_layer_table",
                    "verification": "row count and agency split checked against the report's "
                                    "table 3",
                    "pre_origin": True,
                    "used_by_tests": ("N1.T2_pw_pulse2021; N1.T3_pw_step2019"
                                      if (cty == "Yunlin" and "台水" in ag) else ""),
                    "identifiable": "locations of wells existing in 2022 (a proxy for where "
                                    "the 2021 drought pumping and the earlier public pumping "
                                    "took place)",
                    "notes": ("the Yunlin table skips row number 280: 520 rows printed against "
                              "the stated 521 (421 irrigation-agency wells stated, 420 printed)"
                              if cty == "Yunlin" and "農水署" in ag else "")})
    return out


def _census_rows() -> list[dict]:
    jm, _ = CE.janmay_kwh()
    ev = CE.event_summary(jm)
    out = []
    for _, r in ev[ev["class"] == "irrigation"].iterrows():
        out.append({"row_id": f"P{r.year}{r.county[0].upper()}", "category": "pumping_response",
                    "intervention": f"Metered irrigation pumping electricity, in-grid "
                                    f"{r.county.title()}, Jan-May {r.year}: "
                                    f"{r.janmay_kwh / 1e6:.2f} GWh against "
                                    f"{r.ref_2016_2019_kwh / 1e6:.2f} (2016-2019 mean)",
                    "county": r.county.title(), "location": "in-grid census cells",
                    "township": "", "x_twd97": None, "y_twd97": None,
                    "date_start": f"{r.year}-01", "date_end": f"{r.year}-05",
                    "quantity": round(float(r.ratio_to_2016_2019), 4),
                    "unit": "ratio to the 2016-2019 Jan-May mean (model-unit kWh)",
                    "basis": "measured (cleaned TPC census)",
                    "source": rel_to_repo(K.C0_PRE), "source_page": "county_class_monthly.parquet",
                    "quote": "", "extraction": "product",
                    "verification": f"product gate passed; max flag {r.max_flag}",
                    "pre_origin": True,
                    "used_by_tests": (f"N1.T1_ext_{r.event} (ZR arm)" if r.county == "yunlin"
                                      else "N1.G1-G3 (ZR arm, from rung R1 on)"),
                    "identifiable": "yes (metered)",
                    "notes": "R0 (the deliverable) carries the Yunlin census only; Changhua "
                             "pumping enters from rung R1"})
    return out


def build() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, dict]:
    rows, qchecks = _text_rows()
    rows += _jiji_rows()
    df_ch, ch_checks = MC.merge()
    series = MC.janmay_series(df_ch)
    rows += _public_well_rows(series)
    dw = P.drought_wells()
    rows += _drought_rows(dw)
    rows += _census_rows()
    tab = pd.DataFrame(rows, columns=COLUMNS)
    checks = {"quotes": qchecks, "quotes_found": int(sum(c["quote_found"] for c in qchecks)),
              "quotes_total": len(qchecks), "charts": ch_checks,
              "drought_wells": dw["checks"],
              "every_row_has_source": bool(tab.source.notna().all() and
                                           (tab.source.astype(str).str.len() > 0).all()),
              "every_row_verified": bool(~tab.verification.str.contains("NOT_FOUND").any())}
    return tab, checks, df_ch, series, dw


if __name__ == "__main__":
    t, c, *_ = build()
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(t[["row_id", "category", "county", "date_start", "date_end", "quantity", "unit",
             "verification"]].to_string())
    print({k: v for k, v in c.items() if k not in ("quotes", "charts")})
    print([q for q in c["quotes"] if not q["quote_found"]])
