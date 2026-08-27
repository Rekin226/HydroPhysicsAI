# Literature review — land subsidence in central Taiwan, and where HydroPhysicsAI should aim

Date: 2026-08-21 (rev. 2, broadened sweep). Sources: OpenAlex (~250M works) and Semantic Scholar.
Method note: after a citation-ranked keyword search missed a major author, the sweep was redone by
**enumeration** — all 859 OpenAlex works matching `subsidence` with a Taiwan-affiliated author, medical
"cage subsidence" noise removed (734 remain), then aggregated by author, institution, topic and year.
Every DOI came from indexed metadata and resolves against doi.org.
Companion to `PHASE2_LITERATURE.md`, which covers physics-informed ML for groundwater but contains
**no** Taiwan, Choushui or subsidence coverage.

---

## 1. Bottom line — four findings that should change what you do next

**(a) Your `Sk` model failed for a known, published reason, and the reason names the fix.**
Tsai & Hsu (2018) analysed multi-layer compaction + head data on this exact fan and concluded the
deformation is **visco-elasto-plastic**: elastic, plastic *and viscous* components, with a time-delayed
response. Lees et al. (2022), modelling 65 years of San Joaquin Valley subsidence, independently found
residual clay compaction time constants are far larger than assumed — decades to centuries. Your
`S = Sk · cumulative_drawdown` is memory-less instantaneous plasticity: no elastic recovery, no viscous
delay, no preconsolidation state. A single scalar cannot absorb a rheology, which is why per-site `Sk`
fits at R²=0.81 in-sample and collapses out-of-sample. The negative result in your README is **correct
and expected**, not a data problem alone.

**(b) Your forecasting track has direct prior art on the same fan.**
Patra & Chu (2023, *J. Hydrol. Reg. Studies*, 29 cites) is titled *"Regional groundwater sequential
forecasting using **global and local LSTM** models"*, study area the **Choushui River Alluvial Fan**,
daily, multi-well — the design of your GlobalForecastLSTM. Chang, Sun, Kow et al. (2025, *J. Hydrol.*,
30 cites) do hybrid deep-learning level forecasting on "Taiwan's largest alluvial fan". Patra & Chu
(2024) extend to ConvLSTM over the whole fan. `PHASE2_LITERATURE.md` claim #3 cites none of them.
**Fix before submission.**

**(c) The whitespace survives a much harder search.**
`"physics-informed" AND (subsidence OR land deformation OR InSAR)` returns **52 works** in all of
OpenAlex; exactly **one** couples heads to subsidence with a PINN (Wang et al. 2025, Dezhou, North China
Plain, 2 cites). `hybrid physics machine learning land subsidence compaction` returns **zero**.
`universal differential equation` + hydrology returns **two**, neither on subsidence. Six groups work
this fan and several hold both halves — but always in *separate papers*. Nobody has put a
differentiable, memory-carrying compaction model here.

**(d) Two threats to the leveling plan that the literature makes explicit — see §4.**
Benchmark elevation change is not pure compaction (tectonics), and "learn `Sk` from hydrogeological
attributes" has a published uncertainty floor set by borehole density.

---

## 2. The landscape: six groups

| Group | Angle | Key works |
|---|---|---|
| **Hwang / Hung** (NYCU, WRA) | The monitoring backbone: leveling, GPS, InSAR, gravity, and the MLCW instrument itself | Hung et al. 2021 *WRR* (`10.1029/2020wr028194`); Hung et al. 2011 *RSE* (`10.1016/j.rse.2010.11.007`); Hung et al. 2009 (`10.1007/s12665-009-0139-9`); Hung et al. 2012 (`10.1016/j.enggeo.2012.07.018`); Hung et al. 2024 (`10.3389/feart.2024.1370626`); K.-H. Chen et al. 2023 (`10.1016/j.enggeo.2023.107021`) |
| **Hone-Jay Chu** (NCKU) | Statistical + deep learning: drawdown functions, spatio-temporal regression, LSTM / ConvLSTM | Chu et al. 2021 (`10.1016/j.ejrh.2021.100808`); Ali et al. 2020 (`10.1007/s10040-020-02211-0`); Patra & Chu 2023 (`10.1016/j.ejrh.2023.101442`); Patra & Chu 2024 (`10.3389/frwa.2024.1471258`); Tatas et al. 2022 (`10.1016/j.ejrh.2022.101289`) |
| **Ku / Liu** (NTOU) | GIS factor maps, PCA, ANN, data reconstruction, pumping-reduction scenarios | Ku & Liu 2023 (`10.1038/s41598-023-31390-5`); Ku et al. 2022 (`10.3390/app122312464`); Liu et al. 2023 (`10.1038/s41598-023-44642-1`); Liu, Ku & Ni 2025 (`10.1038/s41598-025-16454-y`) |
| **Hsu / Tsai** (NCKU) | **Poromechanics** — visco-elasto-plastic rheology, spatially varying parameters | Tsai & Hsu 2018 (`10.1016/j.enggeo.2018.07.025`) |
| **Chuen-Fa Ni** (NCU) | Broadest span — geodetic fusion, SBAS-PSInSAR, hydromechanical modelling, poromechanics, DL | Lu & Ni 2016 (`10.3319/tao.2016.01.29.02(isrs)`); Lu & Ni 2015 (`10.5194/piahs-372-77-2015`); Nguyen & Ni 2024 (`10.3390/rs16203789`); Chao, Borja, Lo & Ni 2024 (`10.1016/j.jhydrol.2024.132108`) |
| **Shih-Jung Wang** (NCU) | **Stochastic poroelasticity + geological-model uncertainty** — how much do you need to know to predict compaction | Wang & Hsu 2009 (`10.1016/j.jhydrol.2009.02.049`); Wang & Hsu 2013 (`10.1016/j.jhydrol.2013.06.047`); Wang et al. 2015 (`10.1007/s12665-014-3970-6`); Tran, Wang & Nguyen 2022 (`10.1016/j.enggeo.2022.106543`); Tran, Wang & Dong 2025 (`10.1016/j.enggeo.2025.107991`) |

**Ni Chuen-Fa (NCU)** — 149 works, ~1,250 citations; 23 on subsidence/deformation, 18 on this fan. He
publishes it as the "Choushui River **Fluvial Plain**" (CRFP), often in *TAO*, IAHS proceedings and EGU
abstracts, frequently as middle author — which is exactly how citation-ranked search loses him. He spans
geodetic fusion (Lu & Ni 2016 cokrige leveling + GPS + PSInSAR, 1993–2008), hydromechanical modelling
(Nugraha, Ni & Nguyen 2023), serious poromechanics with **Ronaldo Borja** (Stanford), and deep learning.
Likely reviewer; plausible collaborator.

**Shih-Jung Wang (NCU)** is the one whose questions bear hardest on your plan. His 2009–2025 program asks
how uncertainty in the *geological model* propagates into flow and subsidence predictions: first-order
second-moment poroelasticity in heterogeneous media (2009), flow↔deformation coupling in randomly
heterogeneous media (2013), a nonlinear stochastic poroelastic model to quantify pumping and subsidence
(2015), model uncertainty at **Huwei Town in Yunlin — inside your study area** (2022), and how **borehole
density** controls the geostatistical properties you can estimate at all (2025).

The critical observation now holds *inside* individual groups, not only across them: **the mechanics work
and the machine-learning work do not overlap.** Chu's models are empirical spatial regressions. Ku's are
static susceptibility maps. Tsai & Hsu have the rheology but fit it well-by-well. Ni and Wang hold both
halves — in separate papers. The bridge, a differentiable rheology whose parameters are learned as a
function of hydrogeology across all sites, is unoccupied.

---

## 3. The physics you are missing, in the order it matters

1. **Effective stress.** Compaction is driven by effective-stress increase as head falls. Framing:
   Gambolati & Teatini 2015 (`10.1002/2014wr016841`); Guzy & Malinowska 2020 review (`10.3390/w12072051`).
2. **Elastic vs inelastic, gated by preconsolidation head.** Recoverable (small `Ske`) above the historical
   minimum, permanent (`Skv`, often 10–100× larger) below it. Your `h₀ − running-min(h)` is exactly a
   *fixed* preconsolidation assumption. Li et al. (2022) show it is **variable** and separate the two
   storage modes (`10.1016/j.jhydrol.2021.127420`). Li & Wang (2025) estimate inelastic skeletal
   storativity from SAR + head in the Beijing Plain (`10.1016/j.ejrh.2024.102161`) — the closest published
   analogue to what you are trying to fit.
3. **Delay.** Thick aquitards drain slowly; the surface lags head by months to decades. The term your
   model lacks, and the one Tsai & Hsu found dominant here.
4. **Depth structure — and it is contested on this fan.** Tsai & Hsu found Young's modulus rising with
   depth and distal→proximal. Lees et al. found >90% of recent San Joaquin subsidence came from the *lower
   confined* aquifer. But Nguyen & Ni (2024) report **half the major compaction on the Choushui fan occurs
   at shallow depth**. Your depth-resolved compaction-well data can speak to an open question, not just a
   detail.
5. **Other mechanisms exist.** Clay dehydration was proposed for the Yunlin coastal area by Liu et al.
   (2001, `10.1007/s002540000193`) — a non-poroelastic contribution your model would silently absorb into
   `Sk`.

**So:** `Sk ~ exp(β₀ + β₁·distance_to_coast)` regresses one parameter of a ≥5-parameter rheology onto one
covariate. The in-sample corr of −0.68 is the marine-clay gradient showing through; out-of-sample failure
is what happens when time constant, preconsolidation state and depth partition vary independently.

---

## 4. Two threats to the leveling plan

**Threat 1 — leveling records tectonics too.** Ching et al. (2011, *JGR*, `10.1029/2011jb008242`) built
Taiwan's modern vertical velocity field from **1,843 precise-leveling benchmarks and 199 continuous GPS
stations, 2000–2008**, and found uplift of 0.2–18.5 mm/yr in the range interior with subsidence on the
flanks and coastal plains — noting western coastal rates depart from geologic rates *because of
groundwater pumping*. Benchmark elevation change is therefore **tectonic motion + anthropogenic
compaction**. On the coastal fan the pumping signal (~2 cm/yr, and up to 6 cm/yr per Nguyen & Ni) swamps
tectonics — but at proximal/inland benchmarks the compaction signal is small, so the *ratio* is worst
exactly where your `Sk` fit is most fragile. Mitigation: subtract a tectonic vertical-rate field (or fit a
per-site linear offset), and report what fraction of variance it absorbs. Untreated, this alone can
corrupt a per-site regression.

**Threat 2 — the attribute covariates may not carry the information.** Your plan is to learn rheological
parameters from hydrogeological attributes (fine fraction, drainage path, depth, distance to coast).
Shih-Jung Wang's group has published precisely on that limit: geological-model uncertainty propagating
into flow and subsidence simulations at **Huwei, Yunlin** (Tran, Wang & Nguyen 2022), uncertainty as a
function of **data sufficiency** (Wang et al. 2022, `10.1007/s10064-022-02832-7`), and **borehole density**
controlling which geostatistical properties are estimable at all (Tran, Wang & Dong 2025). Read these
before designing the attribute network; they set the expectation for how much of `Sk` is knowable from
what you have, and they give you a citable reason if the answer is "not much".

---

## 5. Observables you are not yet using

- **The MLCW instrument paper — cite it.** Hung, Hwang, **Sneed** (USGS), Chen & Chu 2021, *WRR*
  (`10.1029/2020wr028194`): magnetic rings at **25 depths to 300 m**, 1 mm precision, tested across the
  proximal, middle and distal fan. This is the reference for the compaction data you already hold, and
  nearly everyone on this fan cites it (35 citations, including Patra & Chu and Ku & Liu).
- **A much larger head network exists.** Hung et al. 2024 (`10.3389/feart.2024.1370626`) use **233
  groundwater monitoring stations across four aquifers** plus **50 continuous GNSS stations** on the
  CRAF. You are modelling 61 wells.
- **Absolute gravity.** K.-H. Chen, Hwang & Tanaka 2023 (`10.1016/j.enggeo.2023.107021`) estimate
  groundwater mass balance of sandy aquifers in Yunlin from gravity — an independent constraint on storage
  change that is not head and not deformation.
- **Leveling+InSAR fusion is prior art.** Hung et al. 2011, *RSE* (`10.1016/j.rse.2010.11.007`, 89 cites)
  already fuse PS-InSAR with leveling on this fan; Lu & Ni 2016 cokrige three sensor types. Pitch novelty
  as *differentiability*, not fusion.

---

## 6. Four directions, ranked

**① Differentiable visco-elasto-plastic compaction model, trained across all sites — recommended.**
Replace the algebraic `Sk` with a small ODE carrying internal state — elastic strain, inelastic strain,
relaxation time constant — and a neural network mapping site hydrogeology to the rheological parameters.
Train jointly against the 16 compaction wells (depth-resolved, high frequency) and the 556 leveling
benchmarks (surface-only, annual). Score with the same leave-one-site-out gate.
*Why:* the intersection of your UDE machinery, the new leveling data, a published diagnosis of why simple
models fail, and a literature gap of one paper worldwide (zero for the hybrid framing).

**② Learned preconsolidation-head gate.** Narrower version of ①: keep the algebraic form, make the
elastic/inelastic switch a learned function of state rather than a running minimum. Best used as an
ablation *inside* ①. Closest published analogue: Li & Wang 2025 (Beijing Plain).

**③ Joint head↔deformation inversion.** Ali et al. (2021, `10.1016/j.envsoft.2021.105123`) run it backwards
— heads from GPS deformation. More occupied than it looks: Lu & Ni (2016) cokrige leveling + GPS + PSInSAR
here, Lu & Ni (2012) did "inversion of subsidence parameters" on this plain, Nguyen & Ni (2024) integrate
InSAR with heads, multi-layer compaction and borehole logs, and Hung et al. (2011) fused InSAR with
leveling. The *differentiable* version is open; the fusion framing is not.

**④ Pure ML subsidence mapping.** Do not. Ku, Chu and Patra have covered ANN, PCA, spatial regression and
ConvLSTM since 2020, and Liu, Ku & Ni 2025 add LSTM subsidence reconstruction with pumping-reduction
scenarios driven by well electricity use.

---

## 7. The next experiment

1. Build the leveling panel: 556 sites × ~11 annual elevations, differenced to cumulative subsidence
   relative to each site's first survey. Already cached as `ls-wra-lsp-obs__choushui_panel.parquet`.
2. **Remove the tectonic component** (Threat 1) — a per-site linear offset at minimum; report how much
   variance it absorbs.
3. Interpolate monthly heads to each leveling site with your existing IDW, then aggregate to survey dates.
4. Refit the current `Sk` regression on 556 sites instead of 14. **This is the fork in the road.** If
   leave-one-site-out R² is still negative with 40× more sites, the wall is the model form, not sparsity,
   and ① is justified on evidence rather than assertion.
5. Fit the VEP-UDE with the parameter network under the same protocol, ablating: no-viscosity, no-elastic,
   fixed-preconsolidation, no-attribute-network.
6. Report against Chu et al. 2021 as the published empirical baseline for this fan.

---

## 8. Caveats on the new data

- Leveling is **annual** — it constrains long-term rheology and time constants, not seasonal dynamics.
  The compaction wells stay the only high-frequency signal (16 fan sites).
- GNSS from the API begins **2020-01-01**, overlapping only the tail of your 2012–2022 window. Note that
  Hung et al. 2024 work with 50 cGPS stations, so a longer record exists somewhere.
- You have **no InSAR**. Every competing group uses it. Expect it as reviewer question #1; the leveling
  network is a defensible answer only if you argue it explicitly.
- The `Sk`-vs-coast correlation you found is corroborated in spirit — Tsai & Hsu report the same
  proximal→distal gradient in elastic modulus. Cite it.

## 9. Not checked

Chinese-language WRA technical reports and theses (likely the densest source on the leveling network
itself); conference proceedings beyond OpenAlex's index; whether the manuscript's target journal has
published competing Choushui work recently.

---

**Published version.** This review is also published as an Artifact:
<https://claude.ai/code/artifact/2e0b5617-18fe-409e-a949-afa6270ed9e0>
Its source is tracked at `docs/assets/choushui-compaction-gap.html`; to update the published
page, edit that file and re-publish it to the same URL (publishing without the URL creates a
separate artifact instead of updating this one).
