"""Write results/dataprep/reports/X3_well_depths/report.{md,json} from the build results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _f(x, n=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{n}f}"


def _pct(x, n=0):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{n}f} %"


def write(out: Path, prods: dict, ev: dict, dl: dict, hd: dict, pr: dict) -> None:
    cs = dl["county_private"]
    comb = dl["combined"].set_index("county")
    ia = dl["ia_layers"].set_index("county")
    cvd = dl["cv_depth"]
    cvs = dl["cvs"]
    R01 = hd["R01"].set_index(["src", "county"])
    S = hd["S"]
    wra_c = dl["wra_county"]

    def priv(v, c, k):
        r = cs[(cs["variant"] == v) & (cs["county"] == c)].iloc[0]
        return f"{_f(r[k])} ({_f(r[k + '_lo90'])}-{_f(r[k + '_hi90'])})"

    def gain(src, c, layer):
        r = S[(S["src"] == src) & (S["county"] == c) & (S["layer"] == layer)]
        if r.empty:
            return "n/a"
        r = r.iloc[0]
        return f"{_f(r['gain_median'], 1)} ({_f(r['gain_lo90'], 1)} to {_f(r['gain_hi90'], 1)})"

    def ratio(src, c):
        r = R01.loc[(src, c)]
        return f"{_f(r['ratio'])} ({_f(r['ratio_lo90'])}-{_f(r['ratio_hi90'])})"

    def tau(src, c):
        r = R01.loc[(src, c)]
        return f"{_f(r['tau_upper'], 1)} / {_f(r['tau_lower'], 1)}"

    pch, pyl = pr["changhua"], pr["yunlin"]
    g = lambda p, k: p["gross_extraction_share"][k]  # noqa: E731
    recov = hd["recov"]
    syn = hd["syn"]
    md = f"""# X3: public well depths and the pumping split between layers (2026-10-21)

Code: `dataprep_staging/x3_well_depths/` (run `build.py`). Numbers: `report.json`.
Products:
- `{prods['evidence']}`
- `{prods['depth_layers']}`
- `{prods['head_layer_response']}`
- `{prods['split_prior']}`

## Short answer

1. **The hypothesis is not supported.** The public depth data do not show Changhua's pumps
   drawing from a shallower layer than Yunlin's. Weighted by where the census electricity
   is, the share of private-well pumping taken from aquifer 1 (twin layer 0) is:
   - Changhua: {priv('census_initial', 'changhua', 'a1')} from the 2000-2008 well census,
     and {priv('calibrated', 'changhua', 'a1')} after the source's own model calibration;
   - Yunlin: {priv('census_initial', 'yunlin', 'a1')} and
     {priv('calibrated', 'yunlin', 'a1')}.

   In both members Changhua is slightly deeper than Yunlin, not shallower. The difference,
   Changhua minus Yunlin, is
   {_f(pr['changhua_minus_yunlin_shallow']['members']['census_initial'])} and
   {_f(pr['changhua_minus_yunlin_shallow']['members']['calibrated'])}.
2. **Both counties pump mostly from aquifer 1.** The data put about 0.8-0.95 of gross
   extraction in twin layer 0. The twin puts 0.09 there (R0) or 0.01 (R1c), and returns
   another 0.42-0.59 of the pumped volume to layer 0 as irrigation return flow. Its net
   layer-0 stress is therefore negative: an injection. That is far from what the depth data
   say in both counties, not only in Changhua.
3. **The heads agree that the counties do not differ in their vertical pattern.** The ratio
   of the shallow (layer 0) to the layer-1 response to local electricity is the same in
   both counties: {ratio('obs', 'changhua')} in Changhua and {ratio('obs', 'yunlin')} in
   Yunlin. What differs is the size: Changhua's heads respond about half as much in every
   layer.
4. **R1c gets Changhua's vertical pattern wrong.** It predicts a ratio of
   {ratio('pred_R1c', 'changhua')}, about a quarter of the observed. Its layer-1 response is
   too strong ({gain('pred_R1c', 'changhua', 1)} m per GWh/month against
   {gain('obs', 'changhua', 1)} observed) and its layer-0 response too weak. In Yunlin R1c
   matches the observed ratio ({ratio('pred_R1c', 'yunlin')}).
5. **Recommended prior** (gross extraction share, both counties):

   | Layer | Changhua | Yunlin |
   |---|---|---|
   | Layer 0 (aquifer 1) | {_f(g(pch, 'L0_aquifer1')['mean'])} ± {_f(g(pch, 'L0_aquifer1')['sd'])} | {_f(g(pyl, 'L0_aquifer1')['mean'])} ± {_f(g(pyl, 'L0_aquifer1')['sd'])} |
   | Layer 1 (aquifer 2) | {_f(g(pch, 'L1_aquifer2')['mean'])} ± {_f(g(pch, 'L1_aquifer2')['sd'])} | {_f(g(pyl, 'L1_aquifer2')['mean'])} ± {_f(g(pyl, 'L1_aquifer2')['sd'])} |
   | Deeper (aquifers 3-4) | {_f(g(pch, 'L2_L3_aquifers3_4')['mean'])} ± {_f(g(pch, 'L2_L3_aquifers3_4')['sd'])} | {_f(g(pyl, 'L2_L3_aquifers3_4')['mean'])} ± {_f(g(pyl, 'L2_L3_aquifers3_4')['sd'])} |

   For the twin's two-layer split, take `pump_frac_shallow` with the layer-0 values above.
   As a logit-normal: Changhua mu = {_f(pch['two_layer_twin_form']['pump_frac_shallow']['mu_logit'])},
   sd = {_f(pch['two_layer_twin_form']['pump_frac_shallow']['sd_logit'])}; Yunlin
   mu = {_f(pyl['two_layer_twin_form']['pump_frac_shallow']['mu_logit'])},
   sd = {_f(pyl['two_layer_twin_form']['pump_frac_shallow']['sd_logit'])}. The deeper share
   is lumped into layer 1.

   A Changhua-only split is supported only as a test of the twin's structure. The data give
   no county difference to aim at. The same prior applies to Yunlin, whose fitted split is
   equally far from the data.
6. **The census attributes cannot predict well depth.** Horsepower, tube diameter, purpose,
   install year and position do not predict the township depth shares out of block. The
   per-cell product therefore uses the township tables directly.

## 1. Sources found

| Source (L0 id) | What it contains | Used for |
|---|---|---|
| Lin S.-T. 2012, NTU MSc thesis (`ntu_thesis_lin2012_choushui_pumping`) | Split of private-well pumping over aquifers 1-4 for each of 46 townships (Changhua 26, Yunlin 20). Two versions: from the 2000-2008 YunTech/ARDC well census (Table 4-10), and after MODFLOW calibration on 2005-2009 heads (Table 4-14). Also census pumping by use (Table 2-2) and aquifer depths. | The two members of the county and cell split |
| WRA 2013 well-management report, marc_id 74288 (`wra_lib_x3_pages_b`) | Wells found in each township and the number deeper than 100 m: Changhua {int(wra_c.loc['changhua', 'deep_gt100_total'])} of {int(wra_c.loc['changhua', 'wells_total'])} ({_pct(wra_c.loc['changhua', 'deep_count_share'], 2)}); Yunlin {int(wra_c.loc['yunlin', 'deep_gt100_total'])} of {int(wra_c.loc['yunlin', 'wells_total'])} ({_pct(wra_c.loc['yunlin', 'deep_count_share'], 2)}). Also a 2012 Yunlin village sample (600 of 621 wells 20 m deep, mostly 4-inch PVC) and county pumping totals. | Independent rank check; context |
| MOA agricultural statistics yearbooks 96-103 (`moa_ia_well_depth_tables`) | Irrigation-association public wells by depth class. Changhua: {int(ia.loc['changhua', 'wells'])} wells in {int(ia.loc['changhua', 'year'])}, {_pct(ia.loc['changhua', 'share_ge120m'])} deeper than 120 m. Yunlin: {int(ia.loc['yunlin', 'wells'])} wells, {_pct(ia.loc['yunlin', 'share_ge120m'])} deeper than 120 m. | The IA component |
| MOA web pages (`moa_ws_ia_public_wells`) | 590 IA wells; mean pumping 2001-2010 of 60.3 x10^6 t/yr in Yunlin and 6.9 in Changhua; most wells in aquifer 2, most 2010 pumping from aquifer 3. "Farmers mostly pump the shallow aquifer, for cost reasons." | IA share and its layers |
| WRA library catalogue pages (`wra_lib_x3_pages`) | 2015-16 measurement of 2,096 irrigation wells: 119 deep (5.7 %). Shallow wells are aquifer 1, deep wells aquifers 2-4. | Context, consistent with the above |
| Control Yuan 53001 (existing L0), 64326, 54525 (`cy_x3_well_documents`) | Yunlin IA wells closed near the high-speed rail were 120-150 m deep (aquifer 2). Water-company wells are 244-301 m. | Context |
| WRA monitoring stations (AMP_V2 `fan_stations`, = data.gov.tw 32718) | Depth and CGS aquifer code of 283 monitoring wells in the two counties | Depth-to-layer model |

Licences: see `{prods['evidence']}/sources.json` and each SOURCE.json.
- The thesis is author-licensed for worldwide public access; copyright is reserved.
- The MOA yearbooks are government statistics.
- The WRA and Control Yuan documents are public government publications. No open-data
  licence is stated; cite them and do not redistribute.
- Nothing fetched needed a login, and no robots.txt disallowed a fetch. NTU asks for a
  10 s crawl delay, and the fetcher obeyed it.

## 2. Depth distributions and twin layers

**Depth to layer, from the monitoring wells.**
- Validation: spatial block cross-validation, {cvd['n_blocks']} blocks of 10 km, 5 folds.
- A depth-only threshold model reaches accuracy {_f(cvd['depth_thresholds']['accuracy'])}
  (log loss {_f(cvd['depth_thresholds']['log_loss'])}). Its cuts are at
  {', '.join(str(int(c)) for c in dl['cuts'])} m.
- Gradient boosting on depth plus position does no better:
  {_f(cvd['gbm_depth_xy']['accuracy'])} (log loss {_f(cvd['gbm_depth_xy']['log_loss'])}).
  The 90 % interval of the log-loss gain contains zero.
- So the threshold model maps depths to layers. A well shallower than 60 m is in aquifer 1
  with probability 0.92. A well of 60-190 m is in aquifer 2 with probability about 0.65-0.70.

**Private wells: the county split.** Shares of pumping by aquifer, weighted by each
township's 2012-2022 census electricity (90 % township-bootstrap intervals):

| County, member | Aquifer 1 (L0) | Aquifer 2 (L1) | Aquifer 3 (L2) | Aquifer 4 (L3) |
|---|---|---|---|---|
| Changhua, census | {priv('census_initial', 'changhua', 'a1')} | {priv('census_initial', 'changhua', 'a2')} | {priv('census_initial', 'changhua', 'a3')} | {priv('census_initial', 'changhua', 'a4')} |
| Changhua, calibrated | {priv('calibrated', 'changhua', 'a1')} | {priv('calibrated', 'changhua', 'a2')} | {priv('calibrated', 'changhua', 'a3')} | {priv('calibrated', 'changhua', 'a4')} |
| Yunlin, census | {priv('census_initial', 'yunlin', 'a1')} | {priv('census_initial', 'yunlin', 'a2')} | {priv('census_initial', 'yunlin', 'a3')} | {priv('census_initial', 'yunlin', 'a4')} |
| Yunlin, calibrated | {priv('calibrated', 'yunlin', 'a1')} | {priv('calibrated', 'yunlin', 'a2')} | {priv('calibrated', 'yunlin', 'a3')} | {priv('calibrated', 'yunlin', 'a4')} |

- **Where the deeper pumping is.** In Changhua, deeper private pumping sits in the
  subsiding south-central townships: 溪州 (0.49 below aquifer 1), 溪湖, 埤頭, 鹿港, 永靖.
  In Yunlin it is concentrated in 麥寮 and the coastal townships. The calibrated member adds
  aquifer-2 pumping in 虎尾, 崙背 and 莿桐.
- **Count against volume.** By count, under 2 % of wells are deeper than 100 m in either
  county. Their share of the pumped volume is larger, because deep wells are bigger.
- **IA public wells** are deep, mostly in aquifers 2-3. They are about 1 % of Changhua's
  agricultural pumping and 4-11 % of Yunlin's. They push Yunlin slightly deeper.

**Combined gross extraction split** (private plus IA): Changhua layer 0
{_f(comb.loc['changhua', 'a1'])} ± {_f(comb.loc['changhua', 'a1_sd'])}; Yunlin layer 0
{_f(comb.loc['yunlin', 'a1'])} ± {_f(comb.loc['yunlin', 'a1_sd'])}.

The sd combines four parts:
- half the difference between the two members;
- the township bootstrap;
- the uncertainty of the IA share;
- 0.05 for the age of the survey (2000-2008, against 2012-2022 pumping).

## 3. Census join and ML skill

**The join.** The key is the township of each census pump, a public administrative unit.
- 223,830 kept, in-grid pumps installed by 2022-12 are joined, weighted by
  duty x horsepower.
- No name, address or postal code column was read.

**The models.** A learning-from-label-proportions logistic model predicts each pump's
probability of drawing from aquifer 1. Its inputs are log horsepower, tube diameter,
purpose, install year and position. It is fitted so that each township's energy-weighted
mean matches the printed share. A gradient-boosting regression on township-mean attributes
is the benchmark.

**Validation.** Leave-one-block-out, with 6 k-means blocks of townships. RMSE of the
aquifer-1 share:

| Target member | County mean | LLP (attributes) | LLP (+ position) | GBM (township means) |
|---|---|---|---|---|
| census | {_f(cvs['census_initial']['county_mean']['rmse_equal_towns'], 3)} | {_f(cvs['census_initial']['llp_attributes']['rmse_equal_towns'], 3)} | {_f(cvs['census_initial']['llp_attributes_xy']['rmse_equal_towns'], 3)} | {_f(cvs['census_initial']['gbm_town_means']['rmse_equal_towns'], 3)} |
| calibrated | {_f(cvs['calibrated']['county_mean']['rmse_equal_towns'], 3)} | {_f(cvs['calibrated']['llp_attributes']['rmse_equal_towns'], 3)} | {_f(cvs['calibrated']['llp_attributes_xy']['rmse_equal_towns'], 3)} | {_f(cvs['calibrated']['gbm_town_means']['rmse_equal_towns'], 3)} |

- Rank correlations with the held-out townships are near zero or negative.
- **No model beats the county mean**, so no within-township variation is used:
  `{dl['choice']['census_initial']}` / `{dl['choice']['calibrated']}`.
- Census attributes carry no usable depth signal at the scale where labels exist. A small
  neural network is not justified with 46 labels.

**Independent check against the WRA deeper-than-100 m counts** (Spearman over townships):
- census member: {_f(dl['rank']['census_initial']['spearman_all'])} on all
  {dl['rank']['census_initial']['n_towns']} townships;
  {_f(dl['rank']['census_initial']['spearman_survey_counts'])} where the count comes from
  the field survey rather than the water-rights register;
- calibrated member: {_f(dl['rank']['calibrated']['spearman_all'])} on all townships.

**Per-cell product.** `{prods['depth_layers']}/cell_layer_shares.npz` and `.csv`:
- the share of pumped electricity drawn from each twin layer (4 x 2148), the mean of the
  two members;
- sd = sqrt(half-difference^2 + 0.10^2), where 0.10 stands for the unobserved variation
  within a township;
- flag 5, inferred;
- private wells only: IA wells are not separable in the census without owner data.

## 4. Head-based estimate (design-time heads only, 2012-02..2022-12)

**Method.**
- **Data.** Each monitoring well's monthly head, out of fold from the R0 run
  (`stage3_per_entry.npz`). One look was logged: units I1:design_oof_heads_R0_R1e, scope
  heads:2012-01..2022-12, wp I1, after_the_fact.
- **Fit.** The head is regressed on local census electricity (cells within 3 km, lags 0-3
  months), with rain recharge (lags 0-3) and a trend.
- **Gain.** The sum of the electricity coefficients, in m per GWh/month. Median over wells,
  with a 90 % site-bootstrap interval.

| Layer | Changhua observed | Changhua R1c | Yunlin observed | Yunlin R1c |
|---|---|---|---|---|
| Layer 0 (aquifer 1) | {gain('obs', 'changhua', 0)} | {gain('pred_R1c', 'changhua', 0)} | {gain('obs', 'yunlin', 0)} | {gain('pred_R1c', 'yunlin', 0)} |
| Layer 1 (aquifer 2) | {gain('obs', 'changhua', 1)} | {gain('pred_R1c', 'changhua', 1)} | {gain('obs', 'yunlin', 1)} | {gain('pred_R1c', 'yunlin', 1)} |
| Layer 2 (aquifer 3) | {gain('obs', 'changhua', 2)} | {gain('pred_R1c', 'changhua', 2)} | {gain('obs', 'yunlin', 2)} | {gain('pred_R1c', 'yunlin', 2)} |
| Ratio layer 0 / layer 1 | {ratio('obs', 'changhua')} | {ratio('pred_R1c', 'changhua')} | {ratio('obs', 'yunlin')} | {ratio('pred_R1c', 'yunlin')} |
| Lag centroid, layer 0 / layer 1 (months) | {tau('obs', 'changhua')} | {tau('pred_R1c', 'changhua')} | {tau('obs', 'yunlin')} | {tau('pred_R1c', 'yunlin')} |

R0's ratio is {ratio('pred_R0', 'changhua')} in Changhua and {ratio('pred_R0', 'yunlin')}
in Yunlin. Its Changhua responses are weak in every layer.

**What it says.**
- **Same vertical pattern.** The observed ratio is the same in the two counties. The
  intervals overlap almost completely, so the heads give no sign that Changhua's stress
  sits in a different layer.
- **Changhua is weaker in every layer.** The county difference is a uniform factor of about
  one half. That points to a county-wide cause rather than a layer split: more storage,
  more leakage or recharge, less water per kWh, or electricity that does not pump
  groundwater.
- **Layer 1 responds first** in both counties (lag about 0.1-0.3 months, against about
  0.3-1 month in layer 0), and with the strongest gain. Taken alone, that says the stress
  is felt most directly in aquifer 2.
- **The conflict with the depth data.** The depth data put 0.8-0.9 of extraction in
  aquifer 1. The likely reconciliation is that the shallow aquifer's large specific yield
  damps and delays its own drawdown. Pumped water partly drains from aquifer 2 through the
  weak aquitard, which is absent at the fan top. Many layer-0 monitoring wells are also
  screened at 30-70 m, below most farm wells. The heads cannot separate these from a
  deeper stress.
- **R1c, against the observed pattern.** R1c puts nearly all net stress in layer 1. Its
  Changhua ratio is too low (0.13 against 0.52) and its layer-1 gain too high. Moving
  Changhua's stress towards layer 0 would move R1c in the observed direction. The same
  argument holds less strongly in Yunlin, where R1c already matches the ratio.

**The two-layer mass-balance split** at nested sites failed its validation. It is not used.
- Run on the models' own predictions, whose net split is known, it returned:
  - R0: {_f(recov['pred_R0']['estimate_median'])} for a true
    {_f(recov['pred_R0']['truth_net_phi'])};
  - R1c: {_f(recov['pred_R1c']['estimate_median'])} for a true
    {_f(recov['pred_R1c']['truth_net_phi'])}.
- Synthetic recovery with the lumped model itself:
  {'; '.join(f"true {_f(t)} gives median {_f(m)}" for t, m in zip(syn.index, syn['median'], strict=True))}.
- Lateral flow and regional drawdown dominate local month-to-month changes, so a local
  two-layer balance cannot identify the split. massbalance_valid = {hd['mb_valid']}.

**Limits.**
- 3 km electricity sums stand in for the stress, and canal water is not a control.
- There are only 8-10 nested sites per county, and the proximal and distal zones have 0-2
  each.
- Wells of the same layer are screened at different depths.
- The heads are design-time evidence already used by the pilots (plan 4.5). Nothing here
  is validation of a later choice.

## 5. Recommendation for the GPU rung

- **The prior.** Use `{prods['split_prior']}/split_prior.json` as a prior on gross
  extraction:
  - Changhua layer 0: {_f(g(pch, 'L0_aquifer1')['mean'])}, sd {_f(g(pch, 'L0_aquifer1')['sd'])},
    90 % {_f(g(pch, 'L0_aquifer1')['ci90'][0])}-{_f(g(pch, 'L0_aquifer1')['ci90'][1])};
  - Yunlin layer 0: {_f(g(pyl, 'L0_aquifer1')['mean'])}, sd {_f(g(pyl, 'L0_aquifer1')['sd'])};
  - deeper share about 0.05 in each county, lumped into layer 1;
  - for the Changhua-minus-Yunlin difference: 0 ± 0.06.
- **Return flow.** The prior is on extraction, not net stress. The rung must keep the
  return-flow term separate, or the prior applies to the wrong quantity.
- **What a Changhua-only split tests.** It tests whether the twin's layer 0 can carry the
  stress. It does not test a county difference, because the data show none.
- **Yunlin's fitted split.** The same evidence says Yunlin's fitted split (0.01-0.09) is
  also inconsistent with the depth data. The twin's layer 0 may then be absorbing the
  stress through its storage, return flow or leakance settings rather than its split.
- **Expected effect.** If the rung moves Changhua towards about 0.8 shallow, its layer-0
  response should rise towards the observed {gain('obs', 'changhua', 0)} m per GWh/month,
  and its layer-1 response should fall. The observed overall Changhua response is still
  half of Yunlin's. A split alone cannot halve all layers, so another factor is needed.

## 6. What was not findable

- **Pump-level depth in Changhua or Yunlin.** The WRA well registry (納管 / 水權, about
  250,000 registered wells, with depth) is not public. data.gov.tw has no well-registry
  dataset with depth for these counties; it has only the monitoring-well datasets 32718
  and 41571.
- **The YunTech/ARDC 2000-2008 well-census reports themselves.** Only Lin's
  township-level tables derived from them are public.
- **The WRA 105年度 report on 2,096 measured wells.** Only the catalogue abstract is
  public; there is no county split and no full text.
- **The 2/3 and 3/3 volumes of the 2013 well-management study.** No public PDF.
- **CGS borehole logs.** These are only in the CGS interactive geology portal (not
  scraped). The aquifer depths they would give are already in the monitoring-station layer
  codes.
- **Two Control Yuan documents (18748, 64306).** They are served as Word files and were not
  fetched.
- **The NLMA (國土管理署) subsidence PDF.** The site's firewall rejects automated requests.
- **Any depth field in the TPC census.** `well_depth_m` is empty for every kept pump.
"""
    (out / "report.md").write_text(md, encoding="utf-8")
    js = {
        "date": "2026-10-21", "products": prods,
        "transcription_checks": ev,
        "depth_to_aquifer_cv": cvd, "threshold_cuts_m": dl["cuts"],
        "county_private_split": cs.to_dict("records"),
        "ia_layers": dl["ia_layers"].to_dict("records"),
        "county_split_combined": dl["combined"].to_dict("records"),
        "census_ml_cv": cvs, "census_ml_choice": dl["choice"], "wra_rank_check": dl["rank"],
        "head_gains_county_layer": hd["S"].to_dict("records"),
        "head_ratio_L0_L1": hd["R01"].to_dict("records"),
        "head_ratio_L1_L2": hd["R12"].to_dict("records"),
        "massbalance_validation": hd["val"],
        "split_prior": pr,
        "look": {"units": ["I1:design_oof_heads_R0_R1e"], "scope": "heads:2012-01..2022-12",
                 "wp": "I1", "purpose": "after_the_fact", "time": "2026-10-20T17:52:27Z"},
    }
    (out / "report.json").write_text(json.dumps(js, ensure_ascii=False, indent=1,
                                                default=lambda o: o.item()
                                                if hasattr(o, "item") else str(o)),
                                     encoding="utf-8")
    pd.DataFrame(dl["combined"]).to_csv(out / "county_split_combined.csv", index=False)
