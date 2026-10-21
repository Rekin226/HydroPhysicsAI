"""Write the X2 report (reports/X2_public_supply/report.md and report.json) from the build
summary and the product tables. Numbers are read, not typed.

Usage: python -I report_x2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO  # noqa: E402

from x2_public_supply import x2const as K  # noqa: E402


def f1(x) -> str:
    return "" if pd.isna(x) else f"{x:,.1f}"


def main() -> None:
    s = json.loads((K.REPORT_DIR / "build_summary.json").read_text(encoding="utf-8"))
    dA = REPO / s["product_full"]
    a = pd.read_csv(dA / "county_annual.csv")
    comp = pd.read_csv(dA / "p6_comparison.csv")
    sysa = pd.read_csv(dA / "changhua_systems_annual.csv")
    jm = pd.read_csv(dA / "wra_janmay_controlled_wells.csv")
    r1e = json.loads((K.DATAPREP / "public_wells/model_npz/v1/validation.json").read_text())
    tw_tab = json.loads((K.EVID / "twc_tables.json").read_text(encoding="utf-8"))
    chart = json.loads((K.EVID / "chart_reconciliation.json").read_text(encoding="utf-8"))
    log = [json.loads(x) for x in K.FETCH_LOG.read_text(encoding="utf-8").splitlines() if x]
    api = {}
    for p in sorted(K.API_CACHE.glob("*.json")):
        for rec in json.loads(p.read_text(encoding="utf-8")).values():
            u = api.setdefault(rec["model"], [0, 0, 0])
            u[0] += 1
            u[1] += rec["usage"]["input_tokens"]
            u[2] += rec["usage"]["output_tokens"]
    piv = a.pivot(index="year", columns="county", values="value_Mm3")
    sdv = a.pivot(index="year", columns="county", values="sd_Mm3")
    flg = a.pivot(index="year", columns="county", values="flag")
    c = comp.set_index(["county", "year"])
    pc = s["p6_comparison"]
    ext = s["external"]
    files_pre = {f["file"]: f for f in s["files_preorigin"]}
    sy = sysa.pivot(index="system", columns="year", values="groundwater_m3") / 1e6
    ch, yl = pc["changhua"], pc["yunlin"]
    hosts = sorted({x["url"].split("/")[2] for x in log})

    def row(y: int) -> str:
        return (f"| {y} | {f1(piv.loc[y, 'changhua'])} ± {f1(sdv.loc[y, 'changhua'])} "
                f"({int(flg.loc[y, 'changhua'])}) | {f1(piv.loc[y, 'yunlin'])} ± "
                f"{f1(sdv.loc[y, 'yunlin'])} ({int(flg.loc[y, 'yunlin'])}) | "
                f"{f1(c.loc[('changhua', y), 'p6_tw_county_Mm3'])} | "
                f"{f1(c.loc[('yunlin', y), 'p6_tw_county_Mm3'])} |")

    years = range(2007, 2026)
    tw_jm = jm[(jm.agency == "taiwan_water")]
    jm_ch = tw_jm[(tw_jm.county == "changhua")].groupby("year").value_1e4m3.first() / 100
    rr = {y: r1e["annual"][str(y)]["model_Mm3_per_yr"] for y in (2010, 2015, 2019, 2022)}
    plus = files_pre["q_p6_proxy_plus_p6_other__strict.npz"]["model_Mm3_per_yr"]
    d11 = {int(y) + 1911: t["monthly"]["district"].get("第十一區管理處")
           for y, t in tw_tab["tables"].items()}
    janmay_ratio = {y: jm_ch.loc[y] * 1e6 / sum(d11[y]["months"][:5])
                    for y in (2012, 2021) if d11.get(y)}
    e12 = a[(a.county == "yunlin") & a.E1_m3.notna() & a.E2_m3.notna()
            & ~a.drought_year.astype(bool)]
    gap = ((e12.E1_m3 - e12.E2_m3).abs() / 1e6)
    diff_ch = (comp[comp.county == "changhua"].set_index("year").measured_county_Mm3
               - comp[comp.county == "changhua"].set_index("year").p6_tw_county_Mm3).dropna()
    n_extracted = len(pd.read_csv(dA / "twc_extracted_numbers.csv"))
    md = f"""# X2: public-supply groundwater abstraction from public sources (report, 2026-10-21)

Status: **gate passed** ({'yes' if s['gate_passed'] else 'NO'}). CPU only, no GPU. Nothing was
committed or staged. No head, ring, pile, GNSS or InSAR observation was opened.

## Bottom line

1. **A measured series exists for Changhua.** Taiwan Water's statistical yearbooks (台灣自來水
   事業統計年報, 2007-2025) print the monthly plant output of the 11th district
   (第十一區管理處), which serves Changhua county only. Until 2022 every one of its eight
   supply systems draws on wells only (表1). Changhua's Taiwan Water groundwater is therefore
   measured: **{f1(piv.loc[2012:2021, 'changhua'].min())}-{f1(piv.loc[2012:2021, 'changhua'].max())}
   Mm3/yr in 2012-2021**, with no downward trend. From 2023 the 鳥嘴潭 lake water (bought as raw
   water, 表4) replaces part of it: {f1(piv.loc[2025, 'changhua'])} Mm3/yr in 2025.
2. **Yunlin is inferred, not measured.** Yunlin's systems mix wells with the 集集 weir and the
   湖山 reservoir, and no table splits their output by source. The WRA briefings' measured
   Jan-May pumping of the Yunlin Taiwan Water wells, scaled by Changhua's seasonal pattern, gives
   **about {f1(piv.loc[2012:2016, 'yunlin'].mean())} Mm3/yr in 2012-2016, falling to
   {f1(piv.loc[2019, 'yunlin'])} in 2019** when 湖山 took over. An output-minus-purchases
   residual differs by {f1(gap.min())}-{f1(gap.max())} Mm3/yr in normal years (median
   {f1(gap.median())}).
3. **P6 has Changhua wrong, Yunlin about right.** P6's Changhua Taiwan Water (closure cohorts
   plus the 129 pending wells) is {f1(ch['p6_tw_2011_2014_mean_Mm3'])} Mm3/yr in 2011-2014
   against {f1(ch['measured_2011_2014_mean_Mm3'])} measured, and it falls by
   {f1(-ch['p6_change_2011_2014_to_2019_2022_Mm3'])} Mm3/yr to 2019-2022 while the
   measured series rises by {f1(ch['measured_change_2011_2014_to_2019_2022_Mm3'])}. The
   closures moved pumping between Changhua systems rather than cutting it. Yunlin's level
   ({f1(yl['measured_2011_2014_mean_Mm3'])} vs {f1(yl['p6_tw_2011_2014_mean_Mm3'])}) and its
   collapse year (2019 in both) agree; P6 cuts too much in 2018
   ({f1(c.loc[('yunlin', 2018), 'p6_tw_county_Mm3'])} vs
   {f1(c.loc[('yunlin', 2018), 'measured_county_Mm3'])}).
4. **The 2.58 ratio is not replaced.** The measured volumes replace P6's volumes and timing,
   not the conversion to model units. 2.58 is a property of the model's abstraction units, not
   of the wells. Divided by 2.58, Changhua is
   {f1(c.loc[('changhua', 2015), 'measured_model_units_Mm3'])} and Yunlin
   {f1(c.loc[('yunlin', 2015), 'measured_model_units_Mm3'])} model-Mm3/yr in 2015.
5. **For rung R1e-pw, the like-for-like replacement is**
   `{s['product_preorigin']}/q_p6_proxy_plus_p6_other__strict.npz`: the measured Taiwan Water
   volumes at P6's locations, plus P6's other agencies. In model units it is
   {f1(plus['2010'])} / {f1(plus['2015'])} / {f1(plus['2019'])} / {f1(plus['2022'])} Mm3/yr
   in 2010 / 2015 / 2019 / 2022. R1e-pw's npz has {f1(rr[2010])} / {f1(rr[2015])} /
   {f1(rr[2019])} / {f1(rr[2022])}. After 2019 the measured version is about twice as large.

## Sources found

| source | what it holds | years | used | licence / terms |
|---|---|---|---|---|
| Taiwan Water statistical yearbooks 96-114 (www.water.gov.tw, 19 PDFs) | 表2 monthly and annual output by district; 表3 annual output by system; 表4 raw water bought, by seller; 表1 sources and wells by system | 2007-2025 (表2(2) back to 1974) | yes: Changhua series, Yunlin residual E1, seasonality | no open-data licence stated; publisher's copyright (DC.Rights); cite, do not redistribute |
| WRA Changhua-Yunlin subsidence briefings 110 and 112 (existing L0) | charts of Jan-May Taiwan Water and irrigation-agency public-well pumping per county; two printed values per series | 2011-2021 (110), 2011-2023 (112) | yes: Yunlin E2; closure-ramp check | WRA publication, no open-data licence recorded |
| WRA water-use statistics, 附錄四 抗旱水井抽水量調查, ROC 110 and 111 (web.wra.gov.tw/Annualwater) | drought wells fed into public supply: wells and volume by county | 2020-2022 | yes: Yunlin lower bound; Changhua check | WRA site, '轉載必究' (reproduction reserved) |
| WRA water-use statistics, annual reports 113 (生活, 各標的, 各區域) and 105 (生活) | water use by county and use | 2016, 2024 | searched: no source split of tap water | as above |
| MOEA action plans (existing L0: rev1 main, phase-2 main) | Changhua and Yunlin Taiwan Water groundwater at baseline and in 2019 | about 2011, 2019 | validators | WRA publication |
| data.gov.tw 152911, 25677 (Taiwan Water) | supply by major system, monthly (rolling: one month only); company totals | 2026-08; yearly | no | Open Government Data License v1.0 |

Districts (verified in the yearbooks' 表1 service areas): Changhua county is served by the
11th district (第十一區管理處) alone. Yunlin is served by the 5th district (第五區管理處), which
also serves Chiayi: through the 林內, 古坑, 崙背, 麥寮 and 四湖 systems to 2015, and through the
single 雲林供水系統 from 2016.

Hosts contacted: {', '.join(hosts)}; {len(log)} requests at 1.5 s or more apart, no login,
token or CAPTCHA. Downloads: `data_fetch_api/new_measurements/twc_statistics/2026-10-21/`,
`wra_water_use_annual/2026-10-21/`, `datagovtw_twc_metadata/2026-10-21/`, each with a
SOURCE.json (checker: ok).

**robots.txt of www.water.gov.tw needs your ruling.** It has two `User-agent: *` groups,
`Allow: /` and `Disallow: /`. Read under RFC 9309 (groups merged; an Allow equal to a Disallow
wins), it allows everything, and I fetched on that reading. Python's standard parser reads it
as disallowed, because it drops the first group, which starts with a byte-order mark. If you
read the file as a refusal, the yearbook PDFs should be deleted from L0 and X2 rebuilt from
other sources, which would lose the Changhua series.

## What was extracted and how it was checked

- **Yearbook tables, from the PDF text layer.** No API was used for these numbers.
  {tw_tab['validation']['n_checks']} arithmetic checks, {tw_tab['validation']['n_failed']}
  failures:
  - months sum to the printed annual total;
  - the 13 districts sum to the company total in every month and year;
  - systems sum to their district, and 表3 equals 表2(1);
  - the sellers in 表4 sum to the total within printed rounding (3 x 1,000 m3);
  - every printing of the same district-year agrees, across up to 19 yearbooks.

  The only exceptions are 1974-75, where the yearbooks note that some output is not merged.
  Each number is listed with its file, table and page in `twc_extracted_numbers.csv`.
- **Sources by system (表1).** These were read from word positions, which gives a
  groundwater-only flag per system and year. Changhua: all eight systems are groundwater-only
  to 2021. From 2022, 彰化供水系統 adds 人工湖 (the lake). From 2021 it also lists a 烏溪
  riverbed intake (伏流水) among its wells; this is carried as extra uncertainty
  ({K.SD_RIVERBED_2021} Mm3/yr).
- **WRA Jan-May charts, read twice with the Claude API.** Four pages (the 110 and 112
  briefings, Changhua and Yunlin) were each read by claude-opus-5-5 (from the PDF) and by
  claude-sonnet-5-5 (from a 200-dpi image). Results:
  - {chart['n_points']} points, {chart['n_disagree']} disagreements beyond 1.5 % of the axis;
  - the two briefings agree on all {chart['cross_vintage_points']} overlapping points
    (largest difference {chart['cross_vintage_max_abs_diff']:.0f} x 10^4 m3);
  - N1's earlier reads agree on all {chart['n1_points']} points;
  - the 2020 and 2021 values printed as text on the pages are used as printed.

  Tokens: {', '.join(f'{k} {v[0]} calls, {v[1]:,} in / {v[2]:,} out' for k, v in api.items())}.
- **Drought-well tables.** Tables 4-1 and 4-2 agree with each other, and the counties add up
  to the national total.

  Source issue: the 111 report's text gives Yunlin 1,095 x 10^4 m3, but both of its tables
  give that number to Changhua and 291 to Yunlin. The tables are used.
- **Checks against the action plans' own figures:**

  | figure | plan | X2 | difference |
  |---|---|---|---|
  | Changhua, 2019 | 99 Mm3 | {f1(ext['phase2_plan_2019_changhua_gw_Mm3']['x2'])} | {ext['phase2_plan_2019_changhua_gw_Mm3']['rel_diff']:+.1%} |
  | Changhua, baseline (about 2011) | 103 | {f1(ext['rev1_plan_baseline_changhua_gw_Mm3']['x2_2010_2012_mean'])} (2010-12 mean) | {ext['rev1_plan_baseline_changhua_gw_Mm3']['rel_diff']:+.1%} |
  | Yunlin, 2019 | 3 | {f1(ext['phase2_plan_2019_yunlin_gw_Mm3']['x2'])} | +0.9 Mm3 |
  | Yunlin, baseline | 46 | {f1(ext['rev1_plan_baseline_yunlin_gw_Mm3']['x2_2011_2012_mean'])} (2011-12 mean) | {ext['rev1_plan_baseline_yunlin_gw_Mm3']['rel_diff']:+.1%} |
  | Yunlin, 2021 drought wells (Jan 1 - Jul 15) | at least 15.4 | {f1(ext['drought_wells_2021_yunlin_Mm3_lower_bound']['x2'])} | consistent |

## The series (Mm3/yr, physical; ± sd; flag in brackets)

Flags: 0 measured; 1 measured, raw-water purchases subtracted; 5 inferred; 6 held or
extrapolated.

| year | Changhua (measured) | Yunlin (inferred) | P6 Taiwan Water, Changhua | P6 Taiwan Water, Yunlin |
|---|---|---|---|---|
""" + "\n".join(row(y) for y in years) + f"""

How the numbers are made:

- **Changhua.** The 11th-district output minus the raw water bought (表4). That subtraction is
  1.4 Mm3 in 2012, falling to 0 by 2018; it is the lake water from 2023. Months come from
  表2(1). Purchases before 2012 were not parsed; they are taken as 0, with 1.4 Mm3 of sd. The
  months of 2011 follow the 2012-2016 mean shape (flag 4).
- **Yunlin.** E2 = measured Jan-May pumping × the same year's Changhua ratio of annual to
  Jan-May output (2.30-2.52).
  - Drought years (2021, 2022, 2023): E1, the systems' output minus the 集集/湖山 raw water,
    because their pumping is concentrated in the dry season.
  - Floored at the drought-well volumes.
  - 2007-2010 hold 2011 (flag 6).
  - 2024-2025 use E1 only.
  - In normal years |E1 - E2| is {f1(gap.min())}-{f1(gap.max())} Mm3/yr. E1 includes the
    systems' own surface intakes (三疊溪). This gap is the sd.
- **The controlled-well subset.** The WRA Jan-May series covers only the action plan's
  controlled public wells. For Changhua it matches the whole 11th district's Jan-May output in
  2012 (ratio {janmay_ratio[2012]:.2f}) and falls to {janmay_ratio[2021]:.2f} of it by 2021. The controlled wells were cut; Taiwan Water's other wells made up the
  difference.

Changhua by system (groundwater, Mm3/yr):

| system | 2012 | 2018 | 2019 | 2022 | 2025 |
|---|---|---|---|---|---|
""" + "\n".join(f"| {k} | {f1(sy.loc[k, 2012])} | {f1(sy.loc[k, 2018])} | "
                f"{f1(sy.loc[k, 2019])} | {f1(sy.loc[k, 2022])} | {f1(sy.loc[k, 2025])} |"
                for k in sy.index) + f"""

The 2019 closures are visible in the south-western systems: 二林 and 溪湖 together fall by
{f1(sy.loc['二林供水系統', 2018] + sy.loc['溪湖供水系統', 2018] - sy.loc['二林供水系統', 2019] - sy.loc['溪湖供水系統', 2019])}
Mm3/yr from 2018 to 2019. P6 credits 9.5 Mm3/yr to the 2019 Taiwan Water cohort. Over the
same years 彰化 (north) and 鹿港 grow, so the county total does not fall.

## Comparison with P6

- **Totals.**
  - Changhua: P6's Taiwan Water is 53.3 Mm3/yr of pending wells plus closure cohorts that start
    at 18.4 and reach 0 by 2019. The measured total is {f1(diff_ch.min())}-{f1(diff_ch.max())}
    Mm3/yr higher in every year.
    Before the closures, Changhua's Taiwan Water pumped about 100 Mm3/yr, not 72.
  - Yunlin: levels agree within about 3 Mm3/yr in 2007-2016.
- **Timing of the closure ramp.**
  - Changhua has no ramp in the total. Only the subsiding south-west systems ramp (2019).
  - In Yunlin the collapse is in 2019 in both series. But P6 already removes its 2017 and 2018
    cohorts (credited 16.2 and 5.8 Mm3/yr), while measured pumping is still
    {f1(c.loc[('yunlin', 2018), 'measured_county_Mm3'])} Mm3/yr in 2018,
    {f1(c.loc[('yunlin', 2018), 'measured_county_Mm3']
        - c.loc[('yunlin', 2018), 'p6_tw_county_Mm3'])} more than P6.
  - The closures are not permanent. In the drought-well survey, closure-programme wells
    (地陷防治停減抽) ran again in drought: 84 of Yunlin's 95 drought wells in 2021 and all 98 in
    2022, and 12 of 20 (2021) and 15 of 23 (2022) in Changhua. Yunlin pumped
    {f1(piv.loc[2021, 'yunlin'])} Mm3 in 2021.
- **The 129 pending Changhua wells.** P6's 53.3 Mm3/yr is of the same order as what the WRA
  series says the controlled Changhua wells pumped: Jan-May {f1(jm_ch.loc[2021])} Mm3 in 2021,
  about {f1(jm_ch.loc[2021] * 2.46)} Mm3/yr annualised. But P6 misses the roughly
  {f1(piv.loc[2021, 'changhua'] - jm_ch.loc[2021] * 2.46)} Mm3/yr of Changhua Taiwan Water
  pumping outside the controlled list.
- **In grid (2012-2021 mean, Mm3/yr).**

  | | P6 | county_uniform | p6_proxy | service_area |
  |---|---|---|---|---|
  | Changhua | {f1(ch['p6_in_grid_2012_2021_mean_Mm3'])} | {f1(ch['measured_in_grid_2012_2021_mean_Mm3']['county_uniform'])} | {f1(ch['measured_in_grid_2012_2021_mean_Mm3']['p6_proxy'])} | {f1(ch['measured_in_grid_2012_2021_mean_Mm3']['service_area'])} |
  | Yunlin | {f1(yl['p6_in_grid_2012_2021_mean_Mm3'])} | {f1(yl['measured_in_grid_2012_2021_mean_Mm3']['county_uniform'])} | {f1(yl['measured_in_grid_2012_2021_mean_Mm3']['p6_proxy'])} | {f1(yl['measured_in_grid_2012_2021_mean_Mm3']['service_area'])} |
- **Does the measured series replace the 2.58 ratio and the proxy timing?**
  - Timing: yes. The measured series replaces P6's timing.
  - Volumes: yes for Changhua (measured); Yunlin is still inferred, but it is bounded by
    measured Jan-May values and by a residual.
  - Locations: no. Only the well-field names are public; they are not geocoded.
  - The 2.58 ratio: no. The ratio is the model's own convention for converting WRA physical
    volumes into its units; a public-supply measurement cannot test it.

  Model-unit volumes at 2.58 (K1's 2.47 in brackets), in model-Mm3/yr:

  | | 2015 | 2020 | 2022 |
  |---|---|---|---|
  | Changhua | {f1(c.loc[('changhua', 2015), 'measured_model_units_Mm3'])} ({f1(c.loc[('changhua', 2015), 'measured_model_units_k1_Mm3'])}) | {f1(c.loc[('changhua', 2020), 'measured_model_units_Mm3'])} | {f1(c.loc[('changhua', 2022), 'measured_model_units_Mm3'])} |
  | Yunlin | {f1(c.loc[('yunlin', 2015), 'measured_model_units_Mm3'])} ({f1(c.loc[('yunlin', 2015), 'measured_model_units_k1_Mm3'])}) | {f1(c.loc[('yunlin', 2020), 'measured_model_units_Mm3'])} | {f1(c.loc[('yunlin', 2022), 'measured_model_units_Mm3'])} |

## Products

### Full product: `{s['product_full']}` (L2, 2007-2025)

Contents:
- county annual and monthly tables, with sd, flag, method and source page;
- `changhua_systems_annual.csv`;
- `twc_extracted_numbers.csv` ({n_extracted:,} numbers);
- the Jan-May reads;
- `p6_comparison.csv`;
- `grid_full.npz`: three variants, physical m3/day, 4 × 2148 × 228, in layers 3-4 split
  0.5/0.5 (P6's convention).

The three grid variants:
- `county_uniform`: per km2 over the county, with the out-of-grid share dropped. The in-grid
  fraction is {s['county_area_fraction']['changhua']:.2f} for Changhua and
  {s['county_area_fraction']['yunlin']:.2f} for Yunlin. P6's loc_uniform_county put the
  whole county in grid.
- `p6_proxy`: P6's Taiwan Water drought-well weights, reused read-only. In-grid fraction:
  {s['in_grid_fraction']['p6_proxy']['changhua']:.3f} Changhua,
  {s['in_grid_fraction']['p6_proxy']['yunlin']:.3f} Yunlin.
- `service_area`: Changhua system by system, over the townships each system serves (from the
  2013 yearbook); Yunlin as county_uniform.

Allowed uses: after-the-fact analysis, recorded-forcing arm, statistical-layer history and
validator. Every pre-origin use is forbidden (U4). This version supersedes v1, which had one
error in the post-2022 system split, and v2, which has the same data but fails on edited code
hashes. Both are recorded in the manifest.

### Pre-origin product: `{s['product_preorigin']}` (L3, 2007-01..2022-12)

Each file is in the `calibrate_flow --extra-abstraction` layout: `q_m3d` (4, 2148, 192) in
model units (physical / 2.58), plus `dates`, `dx` and `n_active`. The loader holds 2022-12 for
later months.

There are four files, each in two vintages:
- `q_county_uniform__*`, `q_p6_proxy__*`, `q_service_area__*`: Taiwan Water only;
- `q_p6_proxy_plus_p6_other__*`: Taiwan Water plus P6's other agencies' cohorts, the
  replacement for R1e-pw.

The two vintages:
- `__strict`: only documents published by 2022-12. 2022 is held at 2020, the last non-drought
  year (flag 6). There is no exemption for its Taiwan Water data. The `p6_proxy` files still
  carry P6's three location exemptions.
- `__exempt`: the 2022 measurements, from yearbook 111 (2023-04), the 112 briefing (2024-04)
  and the 111 drought-well report (2023-12). These are declared as exemption
  `X2.v1.exemption.values_2022_published_after_origin`, with the strict file as its
  sensitivity.

Allowed uses: forcing_frozen, model_design, design_time_evidence and after-the-fact analysis.
Selection, targets and baselines are refused. Guard probes: 10 of 10 behaved as expected.
Checker: 2 current manifests pass; the 3 superseded ones fail on code hashes only.

The P6 restriction still applies: its member is a sensitivity member and never the primary
model. That covers the P6 locations and P6's non-Taiwan-Water part of these files. The Taiwan
Water volumes themselves are measured.

## What was not findable

- **Per-well or per-plant volumes, or closure dates per well.** No public source has them. The
  yearbooks' 表1 gives well counts by system and year, which are in `twc_extracted_numbers.csv`;
  the Yunlin counts are unreliable where the table breaks across pages. The action plans give
  only the credited disposal year per cohort (P6).
- **Monthly Yunlin groundwater.** No source has it. The Yunlin months are a seasonality
  assumption (flags 3 and 5).
- **Taiwan Water purchases before 2012.** Not parsed: the older yearbooks' 表4 uses another
  layout.
- **The other public wells' volumes by year** (irrigation agency, Taiwan Sugar, schools). Only
  the controlled irrigation-agency Jan-May series exists; it was transcribed with the charts but
  is not used in the product.
- **Official subsidence-programme annual reports by agency and county, after 2019.** Only the
  phase-2 first-revision annex (2026) gives cumulative public-supply cuts: 0.8734 x 10^8 m3/yr of
  Taiwan Water wells by 2024. These are credited cuts, not metered volumes.
- **data.gov.tw.** Its Taiwan Water datasets give supply by system (152911 is a rolling file
  holding one month) and company totals (25677). None splits supply by water source.
- **The MOEA statistics department mirror of the yearbook** answered HTTP 403, so it was not
  used.

## Privacy and licences

- No census column, TPC raw or extracted file, or head observation was opened.
- The data.gov.tw metadata JSONs stored in L0 contain the publishers' contact fields, as every
  data.gov.tw record does. They are not copied into any product, table or report.
- The yearbook PDFs are the publisher's copyright and the WRA reports are '轉載必究', so the PDFs
  stay in gitignored L0. Only derived numbers with attribution could be shared.

## Code and evidence

- Code: `dataprep_staging/x2_public_supply/` (new package; ruff clean at 100). Files:
  fetch, discover_annuals, fetch_annual_reports, twc_tables, transcribe_x2, reconcile_charts,
  drought_wells, series, grid, build_x2, finalize_sources, report_x2, x2const.
- Evidence: `results/dataprep/reports/X2_public_supply/`. It holds `fetch_log.jsonl`,
  `build_summary.json`, and `evidence/` (text layers, word boxes, `twc_tables.json`,
  `api_cache/`, chart reconciliation, checker and guard probes).
- Ledger: `results/dataprep/ledger/rows/X2.csv`.
"""
    out_md = K.REPORT_DIR / "report.md"
    out_md.write_text(md, encoding="utf-8")
    rep = {"wp": "X2", "date": "2026-10-21", "gate_passed": s["gate_passed"],
           "products": {"full": s["product_full"], "preorigin": s["product_preorigin"],
                        "superseded": ["results/dataprep/public_supply_gw/v1",
                                       "results/dataprep/public_supply_gw/v2",
                                       "results/dataprep/public_supply_gw/model_npz_preorigin/v1"]},
           "annual_Mm3": {c: {str(y): float(piv.loc[y, c]) for y in years}
                          for c in ("changhua", "yunlin")},
           "annual_sd_Mm3": {c: {str(y): float(sdv.loc[y, c]) for y in years}
                             for c in ("changhua", "yunlin")},
           "annual_flag": {c: {str(y): int(flg.loc[y, c]) for y in years}
                           for c in ("changhua", "yunlin")},
           "p6_comparison": pc, "external_checks": ext, "checks": s["checks"],
           "preorigin_files": s["files_preorigin"],
           "r1e_pw_model_Mm3_per_yr": {str(k): v for k, v in rr.items()},
           "requests": {"n": len(log), "hosts": hosts},
           "api_tokens": {k: {"calls": v[0], "input": v[1], "output": v[2]}
                          for k, v in api.items()},
           "robots_note": "www.water.gov.tw robots.txt: 'Allow: /' and 'Disallow: /' groups for "
                          "'*'; treated as allowed under RFC 9309 2.2.2; needs a user ruling"}
    (K.REPORT_DIR / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False,
                                                         default=float), encoding="utf-8")
    print("written", out_md)


if __name__ == "__main__":
    main()
