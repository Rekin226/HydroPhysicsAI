# Data remediation plan, October 2026

Written 2026-10-12 and revised the same day after two independent reviews (changes listed in
the Review log at the end). This is a plan: nothing in it has been built into the repository.
It turns the findings of [DATA_AUDIT_2026-10-11.md](DATA_AUDIT_2026-10-11.md) and of eight
scout pilots and three design pilots (run 2026-10-12, CPU only, while GPU batch `hydro-par8`
held the code freeze) into work packages that multi-agent workflows can execute.

How to read the numbers:

- **Pilot numbers** come from scripts the scouts ran. Their scripts and outputs sit in the
  session scratchpad under `plan/<scout>/` (index in Appendix A). The scratchpad lives under
  `/tmp`, so WP A0 copies it into the project's gitignored `results/` tree first.
- Pilot numbers marked **frozen θ** were computed with the calibrated `cwa` parameters held fixed.
  Those parameters were fitted with zero Changhua pumping, and they saw any census that a test
  withheld, so such numbers are optimistic.
- **Verified here** marks a number recomputed while revising this plan (scripts in
  `revise_plan/scripts/` of the same scratchpad, or inline runs on the pilot files).
- **Unverified** marks anything that rests on a search snippet, an assumption or an untested
  route.
- **Uncalibrated** marks an interval that has not passed a coverage check (section 4.7).
- **Model units** means kWh of the de-duplicated, duty-capped Yunlin census
  (`pumping.clean_census`, `dedupe-cap`, cap 1.0), the currency the twin's θ was calibrated in.
  Volumes in m³ follow from the twin's fixed conversion (η 0.5, head_extra 40 m), so they are
  model volumes, not physical ones. In Yunlin, WRA's own Jan-May irrigation volumes are 2.58 times
  the model's (design pilot `p6_wra_janmay.json`).
- **County and in-grid.** A county total is the whole county. Only a fraction φ_in of Changhua
  county (about 0.75 by area, 0.82 by planted area, Design 2 township check) lies inside the
  model grid. Every Changhua magnitude in this plan says which of the two it is.

---

> **Revision, 2026-10-12 (afternoon): the Changhua pump census was obtained.** The Taiwan Power
> pump-electricity census for Changhua, Yunlin, Chiayi and Pingtung, 2007-01..2026-07, was obtained
> (L0 `data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/`). Its raw files
> contain names and addresses. The `parquet/` derivative drops them.
>
> - **Yunlin.** The Yunlin sheet matches the cached census exactly: all 116,769 pumps and all
>   25.9 M pump-months for 2007-01..2025-06.
> - **Changhua.** It has 125,941 pumps, 97 % of them in active cells. After the model's own
>   `clean_census`, in-grid electricity is about 188 GWh/yr over 2012-22, the same as Yunlin's
>   about 188. Its irrigation seasonal CV is 0.14 against Yunlin's 0.234, so α ≈ 0.6.
>
> **Consequences for this plan:**
>
> - Changhua pumping is now measured, not inferred. A new WP **C0** ingests the census for every
>   in-grid county with the same cleaning as Yunlin.
> - **P2** (billing, July mask, zero classes, post-2016 layer, layer evidence) now applies to both
>   counties.
> - The P4 inference machinery (P4.1 E1-E7 and E11, P4.2 members, P4.3 strict pseudo tests and
>   level discrimination, the μ table) is **superseded**. It is replaced by **P4m**, metered-forcing
>   diagnostics with the P4.0 engine. The Phase-1 ladder's first rung becomes "recalibrate with the
>   Changhua census".
> - **P1** (TPC aggregate) becomes a validator of census coverage. Its estimate, a ≈ 0.94-0.96 ×
>   Yunlin, is confirmed by the census.
> - **P3** becomes an out-of-sample test of the land-use/crop transfer priors against the
>   Changhua truth.
> - Request R-a is answered.
> - The heads-only inversion's ≈ 0.25 × Yunlin is explained: it borrowed Yunlin's seasonal shape,
>   while Changhua pumps more evenly, and the free per-well datum hides the mean.
>
> Sections below that describe the Changhua inference are kept as the record of what was planned.

## 1. Executive summary

**What is missing.** Changhua (954 of 2,148 active cells; 74 of 158 test wells) has no
pumping input at all. The Yunlin census has three faults:

- it is frozen at mid-2016;
- 94 % of its meters are billed every two months and split evenly over the pair (not 51 %, as
  the audit first said);
- its record stops at a partial July 2025.

Two further stresses are absent from both counties: large public-supply and industrial deep
wells (the census holds 430 public-supply and 326 industrial pumps, only 9 pumps of 50 HP or
more, maximum 100 HP; verified here), and unmetered, diesel or electricity-theft pumps.

Monitoring records also have gaps:

- **Heads** have gaps and artefacts, and 186 of 344 stations are not used.
- **Compaction rings** were not surveyed in 2022, and the instrument changed across that gap.
- **Leveling** ends in September 2021.
- **GNSS** (the 25 WRA stations) and the deep-benchmark feeds end in March-May 2025. A public
  archive the first draft missed (Nevada Geodetic Laboratory, NGL) holds 11 continuous GNSS
  stations in the fan box, 9 of them running from 2015 or earlier past March 2023 (verified
  here).
- **InSAR**: no Taiwanese agency has published any after 2021.
- **Intervention response**: no WP used the natural experiments already on record (well
  closures, drought rotations).

Surface-water deliveries exist only as annual totals, and ET0 in the calibration cache ends
in 2022.

**What can be recovered, and how well (pilot evidence):**

| Gap | Recoverable to | Method | Pilot accuracy | Not recoverable from public data |
|---|---|---|---|---|
| Changhua pumping | the seasonal stress s per zone (3 zones); a county monthly electricity aggregate | TPC county electricity, a Yunlin temporal basis and a land-use/crop shape, inverted against heads through exact Green's functions, then a joint GPU refit (P1-P4) | The adopted design recovered a withheld region's time variation at 0.94-3.26 × truth in a stress test (frozen θ, optimistic). Its mean follows the electricity prior: 0.43-0.89 of a prior error passes into the estimate. Across all tested designs and regions (12-month burn-in; prior-only and density-transfer baselines excluded) the total bias ran from −71 % to +47 % (verified here). Changhua anomaly RMSE 1.203 → 1.13-1.16 m under cross-validation, which cannot detect a wrong magnitude (P4 fact 6). | The time-mean abstraction μ: no head, leveling or subsidence data in this plan constrain it; only the electricity prior does. Therefore also the split between μ and the seasonal amplitude α, and physical volumes. The pattern inside a zone. Changhua's own seasonal timing. |
| Census frozen at mid-2016 | a bounded synthetic layer (zero, low and high members, unweighted) | Installation rate × age ramp (P2d) | Neither route bounds it well. The heads ramp test is over-dispersed (4 of 8 blocks beyond 2 sd against about 0.4 expected; verified here), so its 4 % bound is uncalibrated. The TPC residual trend is +2.7 ± 2.6 GWh/yr from 2016 and +6.0 ± 2.2 from 2017. | The actual post-2016 pumps and where they are |
| Bimonthly billing (94 % of meters) | monthly values per meter | Chow-Lin disaggregation with a profile from staggered deconvolution (P2a) | Squared error 17-25 % below the even split for groups of 1-300 meters; 47 % below fan-wide. The head check still needs a regional placebo. | Exact monthly values for a single meter |
| July 2025 partial month; census end | masked, or flagged imputation; extension to 2026-08 | NaN mask plus the meter's own 3-year July mean; Yunlin extended by TPC × month-of-year ratio, Changhua = a × extended Yunlin (no estimator splice) | July imputation: 14 % fan-total error. Extension: 6.7 % annual RMS in Yunlin leave-one-year-out | Spatial changes after 2025-06 |
| Head gaps and artefacts | monthly fills with calibrated sd; flags | Daily bridge; neighbour ridge regression plus softImpute; GP or Kalman smoother; barometric correction (H1-H2) | Feb-Mar 2026 outage: RMSE 0.28 m. Gaps of 1-12 months: median error 0.10-0.19 m. Split-conformal 90 % coverage 0.86-0.90 by gap type (0.91 pooled in sample) | Long edge extrapolation (66-month leading gaps: 1.1-1.5 m). Gap filling does not fix head skill (cleaning moves the score by < 0.01). Fills never become calibration targets beyond daily-bridged months (U2). |
| Excluded wells | 16 wells with ≥ 80 % observed monthly coverage and a layer code (Yunlin 13, Changhua 3; verified here) | Same filling machinery (H3); inclusion judged on observed months only | — | Changhua gains only 3 wells. Its gap is a lack of wells, not missing months. |
| Yunlin high-swing wells (Q4, 84 % of the excess error) | a sub-grid correction of the cell-mean head, if it validates | Periodic-Theis observation operator from census pumps near each well (H4) | Untested | Local cones, if H4 fails |
| Ring hiatus in 2022, plus the instrument change | corrected ring positions and a bridged 2022 | Common offset plus depth-scale model (+182 ppm for group A); groups assigned from evidence independent of the jump being corrected; path shaped by the co-located pile (S1) | Synthetic pseudo-gaps: 0.31-0.32 cm RMSE in two windows, 1.35 cm in the worst. Monthly path: 1.8 mm (pile-shaped) or 5.3 mm (climatology). Group misclassification untested so far. | The 2022 surveys themselves. That the scale stayed constant in time is unverified. |
| GNSS steps and noise | cleaned daily heights, WRA and NGL families | Step catalogue; common mode from off-fan stations, kept as a separate product (S3) | WRA: median residual sd 14.3 → 8.4 mm with an on-fan common mode, which also removes real fan-wide motion | GNSS cannot bridge the rings on its own |
| Leveling ends 2021, no InSAR after 2021 | fan-wide deformation 2020-2026 | LiCSAR frame 069A_06694_131402 processed with LiCSBAS, referenced to off-fan GNSS (S4-S5) | 14-month change against GNSS: 8.2 mm RMS without atmospheric correction (pilot on 171 interferograms) | Leveling at individual benchmarks after 2021 (needs a data request) |
| Canal deliveries (annual only) | monthly per district 2012-2025 | Supply-limited indicator from the yearbook gauges, benchmarked with Denton (W1) | Jan-May leave-one-year-out RMSE 0.40 / 0.37 against 0.84 / 1.06 × 10⁸ m³ for v3 (Changhua / Yunlin), n = 7 per district | Split within a month; bank-by-bank intake; 2026 until the 2027 yearbook appears |
| ET0 | 2012-2026 | Open-Meteo at the same 61 points (W2) | 2022 overlap with the cache: daily MAE 0.05 mm/d, r 1.000. The archive serves ET0 to 2026-10-05 (verified here). | — |
| Rain | done: rebuilt from CWA CODIS 2012-2026 (evidence doc §11) | Residual gaps (rf25 before 2018, rf24 before 2015-03, months under 80 % of days) handled by masked IDW; optional kriging check (W4) | Not re-tested; the forcing experiment shows weather barely moves head skill | — |
| Ground elevation | 1 km DTM, corrected for epoch | MOI 20 m DTM checked against held-out leveling benchmarks (W3) | SRTM cell value minus leveling benchmark: median +1.2 to +1.4 m | — |
| Barometric loading on heads | hourly pressure 2012-2026 | Open-Meteo `surface_pressure` and CWA hourly `AirPressure`; barometric efficiency per well (H1) | Pressure availability verified here; correction untested | — |
| Public-supply and industrial deep wells | an inventory of closures and volumes; per-well data only by request | MOEA 2019 annex, WRA briefings, request R-i (P6) | — | Monthly per-well abstraction until R-i is answered |
| Intervention response | a table of natural experiments and pre-registered response tests | Well closures 2011-18, the 2021 drought rotation, 2015 and 2020 spring shortfalls (N1) | — | Effects that the tests cannot identify are stated in 7.3 |

**The central unresolved question.** The two routes to Changhua's seasonal pumping stress
disagree, but by less than the first draft said, once two biases in the comparison are removed.

- **Electricity.** TPC county agricultural sales, after source separation, put Changhua's
  pump-like electricity at 0.94 × Yunlin's cleaned census over 2016-01..2025-06 (county total
  about 193 against 206 GWh/yr). On the pre-origin window 2016-01..2022-12 the share is
  a = 0.959 (90 % year-block bootstrap 0.922-1.004; verified here). Only φ_in of the county lies
  inside the grid.
- **Heads, least squares (frozen θ).** The design pilot placed the whole-county series inside the
  grid (φ_in = 1, verified here in `p1_channels.py`). The least-squares scales the heads accept
  are 0.209 (scaled estimator), 0.156 (shape) and 0.099 (transfer) of that series. A least-squares
  scale on an imperfect regressor is shrunk towards zero (evidence doc §4), and the regressor is
  Yunlin-shaped while Changhua head anomalies correlate with Yunlin kWh anomalies at only
  r ≈ −0.59. The formal sd (0.017) is uncalibrated.
- **Heads, amplitude-matched (frozen θ).** The scale at which the model's Changhua anomaly
  amplitude equals the observed one is about 0.50, 0.42 and 0.59 for the same three estimators
  (quadratic interpolation of the squared amplitude ratio through the pilot's three points;
  approximate; verified here).
- **In grid.** The fit is linear in the scale, so applying φ_in divides every scale exactly by
  φ_in. Against the in-grid electricity series the heads accept about 0.12-0.28 (least squares)
  or 0.51-0.79 (amplitude-matched) of its time variation. The gap is a factor of about 1.3-2 on
  the amplitude-matched reading, not the 3-10 the first draft quoted.

No conflict statistic is quoted until experiment E5 has produced a de-attenuated heads estimate
and the strict pseudo tests have produced τ_H (P4, conflict statistic). The evidence bearing on
the direction of the gap, and the role each piece may play, is in the table below.

What the data can identify is narrower than the question. Heads see only the seasonal stress
s = α·μ·R(ψ): the per-well datum is nearly free (prior sd 5 m, shrinkage median 0.99, minimum
0.96; verified here), and the compaction column has no sensitivity to head level (evidence doc
§6), so neither heads nor leveling see the time-mean μ. A gap in s therefore has four
explanations, and only some of them can be told apart:

- θ is biased by having been calibrated with zero Changhua pumping (tested by the strict
  pseudo-Changhua refits, P4.3a);
- the model under-responds near the river, where it has no rivers (tested by the RIV refit,
  P4.3b);
- Changhua pumps more evenly through the year than Yunlin, at the electricity level of μ
  ("different seasonality");
- part of the TPC "pump-like" load is not groundwater pumping, or the conversion to volume is
  wrong for Changhua.

Heads cannot separate the last two. They are separated only by independent seasonality
evidence (P4.3 decision gate). If that evidence does not discriminate, Changhua outputs are
published as functions of μ and α, and the data requests are escalated. Unmetered, diesel or
electricity-theft pumps would make the heads see more abstraction than electricity, so they
cannot explain the gap.

| Evidence | Points to | Verified | Ledger role |
|---|---|---|---|
| TPC agriculture: Changhua 1.13-1.20 × Yunlin; pump-like share a = 0.94 (0.96 pre-origin) | electricity-level μ | yes (TPC scout; a refitted here) | prior on μ |
| Tagged producing wells, end 2018: Changhua 130,414 against Yunlin 126,396 (census = 0.92 of Yunlin's) | comparable magnitude | yes (MOEA 2019 annex, TPC scout) | context |
| WRA chart: Changhua Jan-May irrigation pumping 0.80 ± 0.03 × Yunlin, r 0.95 with Yunlin | comparable magnitude, possibly a scaled copy of Yunlin | digitized; method unknown | validator only (E7) |
| Irrigation-association public-well pumping 2001-10: Changhua 6.89 against Yunlin 60.26 Mm³/yr | less Changhua groundwater use (public wells only) | yes (MOA page, read through a fetch summary by the TPC scout) | context |
| Median seasonal head amplitude 1.33 m (Changhua) against 3.95 m (Yunlin) | smaller Changhua seasonal stress, or a different aquifer response | yes | context; not a valid pumping test (P3) |
| Canal water per farm hectare 2.99 against 1.90 m/yr | less need to pump in Changhua | yes (v3 build) | context |
| Crop-statistics irrigation prior about 0.5 × the land-use prior (35-57 against 83 GWh/yr in fan) | lower Changhua irrigation pumping | yes (land-use scout) | prior-width input only |
| Layer-2 daily-cycle amplitude 55-80 % of Yunlin's, same timing; Changhua DN seasonality r 0.65 with Yunlin irrigation kWh | Changhua pumps with Yunlin-like timing, somewhat less | yes (head-signature scout) | seasonality evidence and rank validator (P5) |
| TPC Changhua/Yunlin month-of-year ratio 1.09 (Feb) to 1.23 (Aug-Sep) | relatively more summer (non-pump) load in Changhua | yes (TPC scout) | seasonality evidence |
| Unmetered, diesel, electricity-theft pumps | heads would see more than electricity | not quantifiable | cannot explain heads < electricity |

**What will not change.** The registered prospective tests (Nov 2026 - Oct 2027) score the old
model as registered. The headline frozen head score is dominated by about 40 high-swing Yunlin
wells; the only data fix proposed for them is the untested sub-grid operator (H4). Changhua
pumping mainly serves the subsidence and policy question, and it moves the 158-well head ratio
only slightly (section 7).

**How selection stays clean.** No observation from 2023 on (heads, rings, piles, GNSS, InSAR)
enters any gate, ladder decision, member choice or threshold (section 4.2). One primary model is
pre-registered before any 2023-26 re-score; every other model is secondary (I1).

---

## 2. Data architecture

### 2.1 Layers

| Layer | Content | Rule |
|---|---|---|
| **L0 raw** | Vendor caches (`AMP_V2/data/`, `chou-shui-data/data/`), new downloads (`data_fetch_api/new_measurements/<source>/`) | Never modified. Each download directory carries a `SOURCE.json` (the existing convention: fetch date, publisher, URL, bytes, plus sha256 and a licence note, which are added from now on). A source whose file rolls over, such as TPC d007012, is archived under a dated subdirectory at every fetch. |
| **L1 cleaned** | Same values and time base as L0, plus flags (QC events, coordinate fixes, metadata repairs) | A value may change only by a documented, reversible correction (flag 1), for example a GNSS step removed or a ring depth corrected. L0 → L1 is a deterministic script. |
| **L2 filled / inferred** | Values plus 1-sigma sd, flag, method id and provenance | Every non-observed value carries an sd and a method. Each product records an "allowed uses" list (2.5). |
| **L3 model-ready** | Grid arrays on the twin's 1 km grid (`(A, T)`, `(C, A, T)`), with dates, a hash of the grid mask and the grid parameters | Built only from L2 products that passed their validation gate. Comes in a pre-origin variant and a full variant. A `RELEASE.json` lists the product versions it was built from. |

### 2.2 Directory layout (all gitignored)

`chou-shui-data/`, `data_fetch_api/` and `results/` are all ignored in `.gitignore` (checked
2026-10-12). `AMP_V2/data/` is ignored by the `data/` rule.

```
data_fetch_api/new_measurements/          L0 downloads, one directory per source, SOURCE.json each
  tpc_d007012/<fetch-date>/                TPC county x sector monthly sales (rolling file)
  moeaea_agri_electricity/                 Energy Administration national monthly (set 236/256)
  moa_crop_038/, moa_rice_J51/             MOA township crop / rice surveys, one file per county-year
  wra_hydro_yearbook/hyb<YYYY>/            river stage/discharge PDFs 2012-2025 (copying disallowed)
  wra_reports_misc/                        WRA briefings (110/112), MOEA 2019 plan annex, CY 53001, drought wells
  licsar_069A_06694/                       interferogram subsets (range reads), epoch metadata
  ngl_gnss/<fetch-date>/                   NGL DataHoldings.txt and daily IGS20 tenv3 files (about 10 MB)
  pressure_hourly/                         Open-Meteo surface_pressure at well points; CODIS hourly checks
  moi_dtm20m_2025/                         MOI 20 m DTM (size to be checked before download)
  openmeteo_et0/                           ET0 2025-07..2026-09, same 61 points
results/dataprep/                          L1-L3 products and process records
  _pilots/<scout>/                         copies of the 2026-10-12 pilots and review checks (code + small outputs)
  _staging_code/                           Phase-0 code, ported to hydrophysics/dataprep/ after the freeze (D12)
  ledger/data_roles.csv                    which source played which role in which reported number
  preregistration/<id>.json                hashed pre-registrations
  heads/{L1_clean,L2_filled_preorigin,L2_filled_full}/v<N>/
  head_signatures/v<N>/   subgrid_operator/v<N>/
  yunlin_electricity/{preorigin,full}/v<N>/   county_electricity/{preorigin,full}/v<N>/
  changhua_prior/{preorigin,full}/v<N>/   changhua_abstraction/v<N>/
  public_wells/v<N>/   natural_experiments/v<N>/
  rings_bridged/v<N>/   deep_benchmarks/v<N>/   gnss_clean/v<N>/   insar_069A/v<N>/
  subsidence_fused/v<N>/   surface_water_v4/{preorigin,full}/v<N>/   et0/v<N>/   ground_elevation/v<N>/
  forcing_uncertainty/v<N>/   rain_check/v<N>/
  model_ready/release_<YYYYMMDD>/          L3 arrays + RELEASE.json
```

Product directories are versioned and never overwritten. A new version is a new directory.

### 2.3 Manifests

Every L1-L3 product directory holds a `MANIFEST.json`:

```json
{
  "product": "heads/L2_filled_preorigin",
  "version": "v1",
  "created": "2026-10-xx",
  "layer": "L2",
  "inputs":  [{"path": "...", "sha256": "..."}],
  "code":    [{"path": "results/dataprep/_staging_code/heads_fill.py", "sha256": "..."}],
  "git_commit": "<HEAD at build time, if code is in the repo>",
  "parameters": {"fit_window": ["2012-01", "2022-12"], "seed": 1000},
  "fit_window_end": "2022-12",
  "exemptions": [],
  "outputs": [{"path": "fill.parquet", "sha256": "...", "rows": 0}],
  "validation_report": "validation.json",
  "gate": {"passed": true, "criteria": "section 3 WP H2"},
  "allowed_uses": ["initial_conditions", "idw_field"],
  "forbidden_uses": ["scoring_target", "baseline", "datum_fit", "calibration_target"],
  "personal_data": "none"
}
```

`fit_window_end` is the last month of data that any fitted parameter of the product has seen,
inputs included. `exemptions` lists any post-origin input a pre-origin product could not avoid,
each with its ledger row and sensitivity check (rule U3).

A manifest checker recomputes every hash. A product refuses to build if any input hash
differs from its manifest. An L3 release refuses to build from a product whose gate has not
passed.

### 2.4 Flags and uncertainty (common to all products)

Long tables carry `value, sd, flag, method, product_version, source_ids`. Grid arrays carry
`value`, `sd` and `flag` (uint8) arrays with the same shape, plus `dates` and the mask hash.

| flag | meaning |
|---|---|
| 0 | observed, passed QC |
| 1 | observed, corrected by a documented model backed by physical metadata or a co-located instrument (ring depth z′ with an independently assigned group, GNSS step removed, swapped coordinates, barometric loading removed) |
| 2 | completed from sub-period data of the same instrument (daily bridge of a partial or outage month) |
| 3 | filled or split using other sites or instruments (neighbour regression, low-rank completion, pile-shaped ring path, Chow-Lin split of a bill by the fan-wide profile, a step correction corroborated only by neighbour regression) |
| 4 | filled from the series' own temporal model (GP, Kalman smoother, seasonal climatology) |
| 5 | inferred from an aggregate or a model (TPC-derived series, synthetic census layer, Changhua inversion, Denton disaggregation) |
| 6 | extrapolated beyond the tested envelope or beyond the record end (never used for calibration) |
| 8 | missing, not filled |
| 9 | suspect observation (kept in L1; excluded downstream unless a product says otherwise) |

The `method` column keeps the finer distinctions the pilots used. For heads it separates a
partial month completed by the daily bridge from an outage month bridged across.

### 2.5 Use rules (enforced in the loaders)

- **U1.** Values with flags 2-6 never enter scoring targets, baselines (climatology), datum fits
  or the registered prospective scorer. This is the rule that imputed or inferred values are
  never scoring targets.
- **U2.** Calibration targets are observed values (flags 0-1) and, at most, daily-bridged
  months (flag 2), weighted by `σ_obs² / (σ_obs² + sd²)`. σ_obs is the measurement plus
  representativeness error of an observed monthly mean: `σ_obs = max(0.10 m, ρ(k))`, where ρ(k)
  is the RMSE of a k-clean-day partial mean against the full month from the gap-fill pilot
  (0.33 m at k = 20, 0.49 m at k = 15, 0.64 m at k = 10; interpolated linearly in k, reaching the 0.10 m floor for a complete month). Flag-3 and flag-4 head fills are
  deterministic functions of other calibration wells, so weighting them as independent
  observations would double-count those wells: they are used only for initial conditions and
  IDW head fields. A variant that needs them as targets must use a joint likelihood with the
  fill-to-donor cross-covariance, and is declared as a separate ladder rung. Forcing products
  (flags 3 and 5 on electricity and deliveries) are inputs, not targets, and carry their sd into
  the forcing-uncertainty propagation (4.8). Flag 6 never enters calibration.
- **U3.** Anything used to calibrate, to start, to set a prior for, or to design a model that is
  scored on 2023-26 must be a pre-origin variant whose `fit_window_end` is 2022-12 or earlier,
  inputs included. This covers heads (H2), μ_elec and the source separation (P1), the Chow-Lin
  parameters and fan-wide profile (P2a), the crop-statistics years and coefficients (P3), the
  canal β and district scales (W1) and CV_j (P4). An input that cannot be had pre-origin (the
  LUIMAP 2024 snapshot) is an exemption: it is listed in the manifest and the ledger with a
  sensitivity run (P3). The full variants (2012-2026) exist for after-the-fact analysis and the
  statistical layer's history.
- **U4.** Post-origin forcing products feed only the recorded-forcing diagnostic arm (the
  forcing experiment), never the frozen or prospective forecasts. These are the census for
  2023-25, the TPC extension, canal v4 for 2023-25 and ET0 for 2023-26.
- **U5.** Each loader reads `allowed_uses` and `fit_window_end` from the manifest. It raises if
  the caller's declared use is not allowed, or if a pre-origin use meets a product whose
  `fit_window_end` is later than 2022-12 and no listed exemption covers it.
- **U6.** Every reported number has a ledger row naming each data source and its role (fit
  target, prior, prior-width input, input, validator, design-time evidence). A source has one
  role per number (section 4.1).
- **U7.** No observation dated 2023-01 or later (heads, rings, piles, GNSS, InSAR) enters a
  gate, a ladder decision, a member choice, a hyperparameter or a pre-registered threshold
  (section 4.2).

### 2.6 Privacy and licensing

- **Pump census.** The columns 戶名1 (a person's name) and 用電地址2 (an address) are never read
  into any product; loaders select columns explicitly. Pump coordinates and meter ids stay in
  gitignored directories, like all real data.
- **Station metadata.** `ls-wra-dbm-obs-stations.parquet` contains contact names and phone
  numbers. Those columns are never copied.
- **Licences.**
  - Hydrological yearbook PDFs disallow copying. Raw daily extractions stay gitignored; only
    scripts and derived monthly aggregates, with attribution, could be committed (user
    decision, section 6).
  - LiCSAR products carry Copernicus derived-work terms and need a LiCSAR citation.
  - The MOI DTM is under the Government Open Data License v1 (data.gov.tw 176927, page opened
    2026-10-12).
- **Pre-commit.** Run the repository's pre-commit secrets scan (CLAUDE.md Rule 1) on any commit
  that touches docs or code from this plan.

### 2.7 Where code lives

- **During the freeze (Phase 0).** Code goes in `results/dataprep/_staging_code/`, which is
  gitignored. Nothing under `hydrophysics/` imports it. Repository modules may be imported
  read-only, for example `pumping.clean_census`, `calibrate_flow.build_grid` and
  `heads._station_xy`.
- **After the freeze (Phase 1).** Code is ported to `hydrophysics/dataprep/` with CPU-only
  synthetic tests: `pytest.importorskip("torch")`, no `tests.` imports,
  `from __future__ import annotations`, ruff at line length 100.

### 2.8 Dataset release gate

An L3 release is model-ready only when all of these hold:

1. Every manifest validates.
2. Every product's validation report meets its acceptance criteria (section 3).
3. The ledger has rows for the release.
4. A short product card names coverage, known limits and the flags in use.
5. The release has been built twice from scratch with identical hashes.

---

## 3. Work packages

### 3.0 Overview

| WP | Title | Gap | Phase | Depends on | GPU | Effort (agent-days) |
|---|---|---|---|---|---|---|
| A0 | Provenance, layout, pilot preservation | infrastructure | 0 (day 1) | — | no | 1 |
| A1 | Validation and anti-leakage harness | infrastructure | 0 | A0 | no | 2 |
| P1 | County agricultural electricity constraint (TPC) | Changhua magnitude prior; census end | 0 | A0 | no | 1.5 |
| P2 | Yunlin census repair (billing, July 2025, zeros, post-2016, extension, layer evidence) | census defects | 0 | A0, P1 (P2e) | no | 3.5 |
| P3 | Changhua spatial prior (shape maps) | Changhua allocation | 0 | A0, P2a | no | 3 |
| P4 | Changhua abstraction inference (centrepiece) | Changhua pumping | 0-2 | P1, P2, P3, W1, A1, H1a; P4.0 internal | P4.3-P4.4 | 7-10 in Phase 0; 12-15 in all |
| P5 | Head-signal pumping signatures (signal processing / ML / DL benchmark arm) | head-to-electricity extraction | 0 | H1, P2a | no | 4 |
| P6 | Public-supply and industrial deep wells | unmodelled deep stress | 0 (inventory); 3 (data) | A0, P2, R-i | no | 1.5 |
| N1 | Natural experiments for policy response | no intervention-response data | 0-2 | A0, W1, S1a, S2, P4.0 | no (I1 runs) | 2 |
| H1 | Head QC, barometric correction, metadata and coordinate repair (H1a); GroundHeight repair (H1b) | head artefacts | 0 | A0 (H1a); W3 (H1b) | no | 2.5 |
| H2 | Head gap filling with calibrated uncertainty | head gaps | 0 | H1a | no | 2 |
| H3 | Recovery of excluded wells and missing records | unused wells | 0 | H2 | no | 1.5 |
| H4 | Sub-grid observation operator for high-swing wells | Q4 excess error | 0 (CPU); 1 (hook) | H1a, P2a, P4.0 | no | 2 |
| S1 | Rings: instrument-change correction and 2022 bridge (S1a product; S1b independent validation) | ring hiatus | 0 (port in 1) | A0, S2 (S1a); S1a, S3, S4 if approved (S1b) | no | 2 |
| S2 | Deep benchmarks as a first-class series | subsidence after 2021 | 0 | A0 | no | 1 |
| S3 | GNSS cleaning: WRA and NGL families | GNSS noise, steps, short record | 0 | A0 | no | 2 |
| S4 | InSAR from LiCSAR + LiCSBAS | no InSAR after 2021 | 0-1 (user decisions D1, D11) | S3 | no | 6 |
| S5 | Space-time subsidence fusion and leveling extension | leveling ends 2021 | 1 | S1-S4 | no | 5 |
| W1 | Monthly canal deliveries (surface water v4) | annual-only deliveries | 0 | A0 | no | 2.5 |
| W2 | ET0 extension 2023-26 | ET0 ends 2022 / 2025-06 | 0 | A0 | no | 0.5 |
| W3 | Ground elevation (DTM) | SRTM bias, GroundHeight = 0 | 0 | A0, S5 (epoch, optional) | no | 1 |
| W4 | Rain residual gaps (optional) | masked gauge-months | 0 | A0 | no | 0.5 |
| I1 | Integration into the twin, recalibration, re-scoring | model use | 1-2 | all accepted products | yes | 5-6 + GPU |
| R1 | Data requests | measurement in place of inference | 0 (send now) | — | no | 1.5 (drafting) |

Effort is one agent working in one session-day. The Phase-0 rows (every row except S5, I1 and
the Phase 1-2 parts of P4) sum to about 50.5-53.5 agent-days. With four single-threaded agents
inside the CPU budget (5.7), Phase 0 takes about 2.5-3 calendar weeks.

---

### WP A0. Provenance, layout and pilot preservation

- **Gap.** The pilots this plan rests on (about 1.15 GB) live in a session scratchpad under
  `/tmp`. Downloads made by the scouts have no `SOURCE.json`. There is no common product layout.
- **Tasks.**
  1. **Pilots.** Copy every `plan/<scout>/` script and small output (JSON, CSV, logs) into
     `results/dataprep/_pilots/<scout>/` with a manifest. Also copy the review checks
     (`review_cf/`, `revise_plan/scripts/`). Large regenerable arrays are not copied, only their
     generating scripts and hashes:
     - Green's-function kernels, 2 × 178 MB;
     - meter matrices, about 290 MB.
  2. **Downloads.** Move each download into its own L0 directory with a `SOURCE.json`
     (URL, fetch date, bytes, sha256, licence):
     - d007012, d007019, d003004;
     - Energy Administration sets 236/256;
     - MOA 038/J51;
     - yearbooks 2012-2025 (about 160 MB);
     - LiCSAR test subsets (107.6 MB);
     - the WRA 110 briefing, the MOEA 2019 plan annex, Control Yuan 53001 and the 111 drought-well
       report;
     - the review's NGL `DataHoldings.txt` (2.8 MB) and `HUWE.tenv3`, and its Open-Meteo
       pressure and ET0 test responses.
  3. Write the manifest schema (with `fit_window_end` and `exemptions`), the flag table and the
     loader guard (`allowed_uses`, U5) as a small staging library.
  4. Create `ledger/data_roles.csv` and the pre-registration directory.
- **Validation.** The manifest checker reproduces every hash. A spot rerun of three pilots from
  their copied code reproduces their key JSON values to a relative 1e-6:
  - TPC `compare_annual.csv`;
  - `d_pseudo_changhua.json`;
  - `pilot_final_clean.csv`.
- **Outputs.** `results/dataprep/_pilots/`, the L0 directories, `_staging_code/io.py`, the
  ledger.
- **Compute.** Minutes of CPU, about 1.2 GB of disk.
- **Risks.** The billing scout's `census.parquet` and `meters_phase.parquet` hold pump
  coordinates (no names or addresses). They stay under `results/`.

### WP A1. Validation and anti-leakage harness

Shared code, so that every WP scores the same way:

- **Scorers.** Equal-well RMSE, datum-free anomaly RMSE, out-of-time RMSE with the
  **training-mean datum** (not the GLS datum, which made a good inversion look worse than zero
  pumping in design pilot `p3_results_v1_gls_datum.json`), and 90 % coverage, both native and
  split-conformal.
- **Coverage gate.** A cluster-adjusted binomial interval for coverage (Wilson interval on the
  effective n = n / design effect, the design effect estimated from the within-cluster
  correlation of misses). Rule in 4.4.
- **Placebo generators.** A 6-month-shifted copy of any temporal regressor; its own 12-month
  climatology; a year-shuffled Chow-Lin profile that keeps the seasonal shape; kWh more than
  15 km away and the Yunlin total as regional covariates.
- **Artificial-gap generators** for every family in section 4.4, with fixed seeds
  (`1000*ord(family)+r`, `7000+r`).
- **Pseudo-region definitions.**
  - **Design-time units** (already used by the pilots; labelled as such wherever they appear):
    Yunlin easting terciles (374 / 385 / 399 cells), the north and south halves, the 2020-22
    hold-out.
  - **Fresh units** (declared in the pre-registration before any E1-E11 or P4.3 run): easting
    terciles with cut points rotated by half a tercile width; a two-tercile region (two adjacent
    design-time terciles hold 55 or 63 of the 84 Yunlin wells, 35-40 % of all calibration wells,
    the closest Yunlin can come to Changhua's 74 of 158); a new 5-region disjoint k-means partition with seed 2027; and a
    mid-record temporal block, 2016-01..2018-12.
  - Two shaped stress regions: the river-adjacent north half (600 cells, 37 wells) and a 10 km
    river band that mirrors Changhua's Z2.
- **Site-grouped folds.** Nested wells stay together. Spatial blocks are 10 km clusters with a
  5 km buffer for heads; leveling folds use a buffer of at least 10 km.
- **Pre-registration tool.** It writes a JSON (zones, priors, burn-in, folds, fresh units,
  thresholds, gate scores and their weights, the primary model, input hashes), records its
  sha256 in the ledger, and refuses to score a design whose pre-registration postdates any look
  at its test data.
- **Acceptance.** Reproduces the pilot metrics named in A0 to a relative 1e-6. The coverage
  interval reaches at least 0.89 actual coverage of a true 0.90 in 10,000 simulated clustered
  binomial draws.
- **Effort.** 2 agent-days. **Compute.** CPU only.

---

### WP P1. County agricultural electricity constraint (TPC)

- **Gap.** There are no Changhua electricity data, and the Yunlin census ends with a partial
  July 2025.
- **Data (verified by the TPC scout).**
  - **TPC d007012.** The file's title names only residential, services and government, but it
    also carries a 農林漁牧 (agriculture) sector by county and month. It covers 2016-01 to
    2026-08, with no missing months for 彰化縣 or 雲林縣.
  - **Changhua and Yunlin agriculture.** Annual sales are 447-542 and 371-465 GWh/yr
    (ratio 1.13-1.20). The two counties correlate monthly at r 0.97, or 0.95 after removing the
    seasonal cycle.
  - **National series.** The Energy Administration's national monthly agricultural electricity
    covers 2007-01 to 2026-08.
- **Technique.**
  1. **Ingest.** Archive every fetch under a dated directory, because d007012 is a rolling
     window.
  2. **Backcast 2012-2015.** County agriculture = national agriculture × month-of-year county
     share fitted on 2016-2019, with the share trend (about +0.1 percentage point per year)
     included rather than optional. The backcast extrapolates 1-4 years, so it is validated by a
     matched-lead pseudo-backcast (Validation), not by leave-one-year-out.
  3. **Source separation**, two-component NNLS with a year-block bootstrap:
     `CH_ag(t) = a·YL_clean(t) + b·(YL_ag(t) − YL_clean(t))`, on two windows:

     | Window | a (90 % bootstrap) | b | R² | Use |
     |---|---|---|---|---|
     | 2016-01..2025-06 (full) | 0.936 (0.893-0.975) | 1.410 | 0.965 | after-the-fact analysis |
     | 2016-01..2022-12 (pre-origin) | 0.959 (0.922-1.004) | 1.395 | 0.964 | every prior and calibration input of a model scored on 2023-26 (U3) |

     Both rows were refitted here from the scout's candidate file (the full row reproduces the
     scout). The pre-origin share is 2.5 % higher.
  4. **Estimator ensemble**, carried as the spread behind the prior on the Changhua mean:
     - scaled (a·YL_clean);
     - shape (CH_ag − b·N_YL);
     - transfer (CH_ag × r(moy): 1.17-1.27 × Yunlin);
     - irrigation-only (0.83);
     - national sub-industry NNLS (about 0.75 ×, weak).
     The crop-statistics total from P3 (about 0.5 ×) is recorded in the ledger as a
     **prior-width input**: it widens the prior but is never the level and never validates μ.
     The prior's log-sd must cover the whole spread, 0.5-1.27 × Yunlin, which rules out the
     0.15 used in one design.
  5. **In-grid anchor.** `μ_elec,in-grid(t) = a_pre · YL_clean(t) · φ_in`, with φ_in from P3 and
     its uncertainty added in quadrature to the prior's log-sd. County and in-grid values are
     reported in separate, labelled columns.
  6. **Extension past the census end, without a splice.** Extend Yunlin first:
     `YL_clean_hat(t) = TPC_ag_YL(t) · r(moy)` for 2025-07 to 2026-08. Then Changhua
     = `a · YL_clean_hat(t)`, so the Changhua estimator stays the same across the census end.
     The transfer estimator (CH_ag × r(moy)) is the only candidate that runs past 2025-06 on its
     own, and it runs 1.35 × the scaled one (2024: 265.0 against 196.0 GWh; June 2025: 25.44
     against 18.73 GWh; verified here). Splicing it on would put a step of about 35 % into the
     Changhua recorded-forcing arm at 2025-07. Estimators are never spliced.
  7. **CV_j** of the temporal basis is recomputed from the Chow-Lin product actually used (P2a),
     on the pre-origin window. The values 0.264 (irrigation) and 0.197 (other) came from the
     physics scout's even-split Q and are superseded.
- **Validation.**
  - **Extension,** leaving one year out: annual RMS 6.7 % (done). Also hold out whole seasons,
    and score 2023-25 separately.
  - **Backcast, matched lead.** Shares from 2020-23 predicting 2016-19 (verified here):
    Changhua +2.4, +4.8, +2.5, +2.6 % (RMS 3.2 %, all positive); Yunlin +4.7, +0.1, +1.2, +4.3 %
    (RMS 3.25 %). The leave-one-year-out figure (2.5-3.1 %) measures interpolation and is not
    used for acceptance. Report bias and RMS after the share-trend correction.
  - **Synthetic recovery of a and b,** from `α·P_YL + β·N_YL + noise` with the noise matched to
    the observed residual, under the correct model and under four misspecifications:
    - Changhua pump seasonal CV × 0.5 and × 1.5;
    - a 1-month phase shift of the Changhua pump shape;
    - drought-year concentration (Changhua pumping × 1.5 in the springs of 2015, 2020 and 2021,
      the years the W1 benchmark shows short deliveries).
    Report the bias of a under each. If a bias exceeds the bootstrap half-width, widen the μ
    prior by that bias in quadrature.
  - **Acceptance:**
    - extension annual RMS ≤ 10 %;
    - matched-lead backcast RMS ≤ 5 % and |mean bias| ≤ 3 % after the trend correction;
    - under the correct model, the cluster-adjusted interval for the coverage of a's bootstrap
      interval contains 0.90 (4.4);
    - misspecification biases reported and folded into the prior width.
- **Outputs.** `county_electricity/{preorigin,full}/v1/`:
  - `tpc_county_sector_monthly.parquet`;
  - `changhua_candidates.parquet`: one column per estimator, county and in-grid, flags for
    backcast (2012-15) and extension months;
  - `yunlin_extension.parquet`;
  - `decomposition.json` and `misspecification.json`.
- **Compute.** Minutes of CPU. **Effort.** 1.5 agent-days, mostly porting the scout's tools.
- **Risks.**
  - TPC does not define 農林漁牧, which includes non-pump load: livestock cooling, aerators,
    lighting.
  - The pump-like share a cannot be tested out of sample with public data: in Yunlin the pump
    shape is the census itself. Only requests R-a and R-b can validate it.
  - It is unknown whether TPC reports billing months or consumption months.
  - There are no county-by-sector data before 2016.
  - The census includes about 185-230 GWh/yr of "industry" and about 50 GWh/yr of "domestic"
    meters, so coverage ratios must use the agricultural classes only.

### WP P2. Yunlin census repair

**P2a. Billing phase table and Chow-Lin disaggregation.**

- **Phase table** (billing scout, verified):

  | billing code | meters | phase |
  |---|---|---|
  | M2 | 54,790 (plus 11 phase B, 3 monthly) | A: pairs Jan-Feb, Mar-Apr, ... |
  | M1 | 43,154 (plus 6 monthly) | B: pairs Feb-Mar, Apr-May, ... |
  | M0 | 6,284 (plus 8 A/B) | monthly |
  | M9 | 265 (plus 52 A/B) | monthly |

  6,549 meters are monthly (M0 plus M9). The bimonthly meters carry 63.5 % of 2012-22
  duty-capped energy and 92 % of irrigation energy.
- **Why the even split matters.** Phases cluster in space: 68 % of bimonthly energy sits in
  1 km cells that are at least 90 % one phase. The even split is therefore a phase-locked
  two-month boxcar at cell scale.
- **Formulation.** For a meter with pair sums Y = Cx:
  - Model: `x = Xβ + u`, with u AR(1) and ρ chosen by likelihood from {0, 0.5, 0.9}.
  - Estimate: `x̂ = Xβ̂ + ΩC′(CΩC′)⁻¹(Y − CXβ̂)`, where β̂ is the GLS estimate.
  - Indicator: X = [1, p(t)], where p(t) is the fan-wide irrigation profile. p is recovered
    from the phase-A and phase-B population pair sums by
    `min_p ||C_A p − Y_A||² + ||C_B p − Y_B||² + λ||D²p||²`, with λ = 0.01.
  - With one common indicator this is a fixed linear operator per phase, so all 92k
    bimonthly irrigation meters run in seconds.
  - Keep the even split for the small non-irrigation bimonthly share.
  - **Flag 3.** The within-bill split comes from the fan-wide profile of other meters.
  - **Pre-origin variant.** p(t), β̂ and ρ are estimated on data to 2022-12 only (U3); the full
    variant uses data to 2025-06.
- **Validation.** Hold out the truly monthly meters (M0/M9 irrigation, n = 1,785). Give each
  the phase of its nearest bimonthly meter, sum it into pairs and recover it.

  | Metric | Pilot result |
  |---|---|
  | Pooled skill 1 − SSE/SSE_even, groups of 1-300 meters | +0.17 to +0.25 |
  | Pooled skill, fan-wide | +0.47 |
  | Partial R² of local energy in high-passed head regressions, 1 km | improves at 86 of 110 wells (78 %) |
  | Same, 2 km | improves at 84 of 113 wells (74 %) |

  Baselines are the even split and pro-rata by the same profile. Pro-rata and Denton with local
  indicators, and rain/ET0 indicators, were validated as no better than the even split and are
  excluded.

  **The head check needs a regional placebo.** The pilot regressions
  (`s07_layers.py`, `s11_chowlin_production.py`) used rain lags 0-2 as the only covariates.
  Chow-Lin writes the fan-wide profile into every meter, so local series become more regional,
  and the head-signature scout showed the monthly head fingerprint is mostly regional (local-only
  r −0.19 against −0.67 total). The gain fading from 1-2 km to 3-5 km is as consistent with
  regional-signal injection as with better timing. The rerun therefore:
  - adds the Yunlin total and kWh more than 15 km away as covariates, or regresses them out of
    both series first;
  - runs a placebo Chow-Lin whose profile is year-shuffled but keeps the seasonal shape;
  - requires the real profile to beat the placebo with a site-clustered one-sided p < 0.05
    (cluster bootstrap by site).

  **Acceptance:** pooled skill ≥ 0.15 at every group size from 1 to 300 on the meters. If the
  placebo-controlled head check fails, P2a still ships on the meter validation, and the head
  gain is reported as not distinguishable from regional-signal injection.

**P2b. July 2025 and record ends.**

- **Masking.** Any month whose billing pair is incomplete for a phase is NaN for that phase,
  never zero. Phase A in July 2025 is 99.8 % zero because its July-August bill is missing.
- **Optional imputation.** Use the meter's own 3-year same-month mean, flagged 5. In a
  2013-2024 backtest it gave a 14.0 % fan-total error (worst 29.2 %) and a 1 km cell relative
  RMSE of 0.335. Treating July as zero gives a 100 % error.
- **M0 meters.** They may be slightly incomplete in July 2025: the July/June ratio is 1.10
  against 1.16-1.26 in earlier years. Flag them 9.
- **Acceptance:** mean fan-total error ≤ 20 % in the rolling-origin backtest (pilot 14.0 %).

**P2c. Zero classification.**

| Zero type | Treatment |
|---|---|
| Before the install date (878,738 meter-months) | Structural zero |
| Interior runs ≤ 12 months (58 % of zero-months) | Real idling (irrigation zeros are seasonal) |
| Trailing runs ≥ 12 months | Terminated meter: a real zero for that meter, flagged as possible missing replacement abstraction |

A synchronized-zero monitor across billing groups lists the flagged months: 2010-01, 2011-12/
2012-01, 2012-09, 2012-12/2013-01, 2015-09, 2016-01/02, 2019-07/08, 2021-07/08 and 2023-09.
Their cause is unverified. They stay in the data with flag 9 and enter as a sensitivity member
only. **Acceptance:** the classification reproduces the pilot's class counts within 1 % and the
irrigation zero seasonality (5.6 % in April, 15.6 % in August).

**P2d. Post-2016 completion layer**, a bounded synthetic layer.

- **Rule.** Synthetic pumps per township × class from 2016-07 at the 2010-15 install rate:
  - low variant, excluding 2013: 1,327 per year;
  - high variant, including the 2013 registration surge.
- **Ramp.** Energy ramps with pump age (irrigation 725 → 1,039 kWh/yr over six years).
- **Placement.** In proportion to 2010-15 installs per 1 km cell.
- **Size.** By 2022: 7,300-11,300 pumps, 14.0-19.5 GWh/yr, against 206 GWh kept. By 2025:
  11,300-17,500 pumps, 21.6-30.1 GWh/yr.
- **Members.** Carry three, zero, low and high, with equal weight. No evidence in hand can weight
  them:
  - **Heads.** The physics scout's ramp test gave +30 ± 34,388 m³/day by December 2022, but 4 of
    its 8 blocks lie beyond 2 sd (2 positive, 2 negative) against about 0.4 expected under the
    null (verified here), so the SE is underestimated. The median Yunlin residual trend also fell
    from +0.083 to +0.041 m/yr after 2016-07, which is the direction missing pumps would push,
    and θ, calibrated on 2012-22 with the frozen census, can absorb part of a trend in its
    recharge and boundary terms. The SE is inflated by the block over-dispersion (χ²/df) or
    replaced by a block bootstrap, and the result is labelled uncalibrated.
  - **TPC residual trend.** +2.7 ± 2.6 GWh/yr from 2016, but +6.0 ± 2.2 from 2017: the bound
    depends on the start year, and both are reported.
  - **Optional GPU check (Phase 2, 1 job).** Repeat the ramp test with θ refitted on 2012-2016
    only (`--holdout-months 72`, an existing option).
- **Survivorship note.** Terminated meters' last-year energy (6.7 GWh/yr by 2022, 9.2 by 2024)
  is a lower bound on replacement abstraction now on unseen meters.
- **Validation.**
  - Freeze the census at mid-2013, predict the 2014-16 cohorts per township, and compare counts
    and kWh. **Acceptance:** median township count error ≤ 30 % and fan-total kWh error ≤ 20 %.
  - In the twin (I1), check post-2016 Yunlin residual drift for 2017-22 against 2012-16.
- **Risk.** kWh per active meter rose from about 1,530-1,765 (2012-16) to 1,974-2,373
  (2017-24). Pumps added on old accounts may already be counted, so the high member can double
  count.

**P2e. Extension to 2026-08** from P1 step 6, for the recorded-forcing arm only (rule U4).

**P2f. Evidence on aquifer allocation**, used as an acceptance test in I1 rung R4.

- **Layer response at nested sites, as piloted.** The layer-1/layer-2 response ratio to 3 km
  local energy is 0.49 observed (90 % bootstrap 0.44-0.63) against 0.21 in the model (19 sites).
  Layer 3/2 matches (0.85 against 0.69), and so does layer 4/2. The model side came from the
  `stage3_datum_gate` forward archive.
- **Why it needs a rerun before it can gate anything.** The regressions used rain lags as the
  only covariates, so the observed 0.49 is partly a response to the regional seasonal signal,
  which includes canal percolation and ET acting on layer 1. Used as an acceptance test, it would
  invite fitting per-layer fractions and return flow to absorb recharge errors. The rerun:
  - computes both ratios on **local-only slopes** (the Yunlin total and kWh more than 15 km away
    regressed out of heads and energy first);
  - recomputes the model side from the `cwa` θ the plan builds on (verified here in
    `stage3_datum_gate_cwa/stage3_theta.json`: shallow split sigmoid(−2.353) = 8.7 %, return
    flow 41.7 %, spread 0.5 km at its floor), not the `stage3_datum_gate` θ (6.2 %, 45.2 %,
    2.71 km);
  - is used in I1 only after rung R3 (monthly deliveries) has been accepted, so that a
    recharge-timing error is not absorbed by the allocation.
- **Depth proxy.** Meters over 10 HP carry 19 % of kept energy. They are the only plausible
  deep-well class, offered as a depth proxy.

**WP P2 as a whole:**

- **Outputs.** `yunlin_electricity/{preorigin,full}/v1/`:
  - `meter_phase.parquet`;
  - `kwh_by_class_recorded.npz` and `kwh_by_class_chowlin.npz`, per class `(C, A, T)` with flags
    and sd;
  - `synthetic_post2016_{low,high}.npz`;
  - `july2025_mask.parquet`;
  - `zero_class.parquet`;
  - `layer_response_localonly.csv`.
- **Compute.** CPU, under an hour. **Effort.** 3.5 agent-days.
- **Risks.** The M0 validation meters are a different population (median 5 HP against 3 HP). A
  single fan-wide profile ignores differences in the local crop calendar.

### WP P3. Changhua spatial prior (within-zone shape)

- **Gap.** The pattern inside Changhua cannot come from the heads: a 40-cell block has relative
  sd above 100 %, and uniform-block inversions give negative block-density correlation in every
  pseudo region. It must come from land use and crop statistics. Those give **shape only**;
  their level transfers badly (land-use-only bias from −26 % to +276 % in the pseudo regions).
- **Technique.**
  1. Fetch the MOA 農情調查 (038) and 糧情調查 (J51) township surveys for every year 2012-2025
     in both counties, one call per county-year. The J51 city filter is ignored by the API, so
     filter locally.
  2. **Crop-statistics township model** on Yunlin:
     `kWh_town,y = Σ_crop β_crop,y · ha_crop,season,y`, NNLS. Leave-one-out R² is 0.80-0.86
     for 2014/15/16/21, against 0.26-0.40 for a single rate per hectare. The coefficients are
     unstable across years (rice crop 1: 523-1,742 kWh per ha-season), so fit per year with
     shrinkage towards the pooled fit.
  3. Apply the model to Changhua townships, scaled by the **in-fan farmland fraction** computed
     from LUIMAP inside MOI township polygons. The scout scaled by land-area fraction.
  4. Spread within each township by the LUIMAP irrigation shape. Aquaculture and livestock use
     per-hectare rates from LUIMAP: pond 6.1 MWh/ha/yr [5.3-8.0], livestock buildings 5.9
     [5.1-6.8]. Use MOI township polygons: GADM misses half the coastal ponds (1,099 against
     2,125 ha).
  5. **Candidate maps and their gate outcomes on the pilot numbers.** The gate (Validation) was
     applied to the design pilot's eight transfer splits (`a5_scores.csv` and `a4_scores.csv`
     against their `farm_uniform` rows; verified here):

     | Map | Mean ΔAA_5km against uniform | Worst ΔAA_5km | Splits with 5 km correlation ≥ 0.5 | Gate |
     |---|---|---|---|---|
     | LUIMAP random forest (`rf_total`) | +0.047 | −0.007 | 6 of 8 | passes |
     | compA (random forest per class) | +0.055 | −0.015 | 5 of 8 | fails (correlation) |
     | compB (random forest for irrigation and other, per-hectare rates for ponds and livestock) | +0.000 | −0.148 (paddy split, low) | 8 of 8 | fails (AA) |
     | HGB, monotone | +0.026 | −0.033 | 5 of 8 | fails |
     | NNLS on all classes | −0.079 | −0.263 | 6 of 8 | fails |
     | MLP variants | −0.09 to −0.52 | down to −0.75 | 2-6 of 8 | fail |
     | crop-dasymetric (steps 2-4) | not scored | — | — | **precondition** |

     The first draft called compB the preferred robust allocator because it kept 5 km
     correlation ≥ 0.61 on every split; it fails the share-accuracy half of the plan's own rule
     and is kept only as a diagnostic. The criterion is not loosened after seeing these
     numbers. The crop-dasymetric map becomes the primary π only if it passes the same gate on
     the same eight splits. If it fails, the primary π is the LUIMAP random forest, the one map
     that passes on the current pilot.
  6. **Shift diagnostics.**
     - Adversarial AUC 0.95.
     - Kish effective sample size of importance weights: 5-10 %.
     - Canal water per farm hectare is outside Yunlin's 1-99 % range in 86 % of Changhua cells.
     - Upland-crop share is 0.013 against 0.188.
     Every Changhua cell carries an extrapolation flag.
  7. **φ_in.** The in-grid fraction of Changhua county is computed from MOI polygons and
     farmland (the township check gave about 0.75 by area and 0.82 by planted area), never
     assumed. Its uncertainty (at least the area-to-planted-area spread) is exported to P1 and
     P4.
  8. **Pre-origin variant (U3).** The crop-statistics model uses survey years up to 2022 only.
     LUIMAP is a 2024 snapshot and cannot be had pre-origin, so it is an **exemption** recorded
     in the manifest and ledger, with two sensitivity runs: a crop-statistics-only π (township
     totals spread uniformly over in-fan farmland), and an earlier NLSC land-use survey edition
     if one can be obtained (availability unverified).
- **Validation.**
  - **Transfer battery**, 8 splits, used for selection:
    - easting terciles;
    - the most Changhua-like towns by propensity;
    - town halves by paddy share, in both directions;
    - town halves by aquaculture area, in both directions.
  - Random leave-one-township-out and 5 km blocks are reported, not used for selection, because
    they are optimistic: 5 km correlation 0.90 there against 0.58 on the transfer splits.
  - **Acceptance:** AA_5km mean ≥ uniform + 0.03 and never below uniform − 0.02; 5 km
    correlation ≥ 0.5 on at least 6 of 8 splits. AA = 1 − ½ Σ|share_pred − share_true|.
  - Changhua head swing is **not** a valid check: even the true census correlates negatively
    with swing within Yunlin at kernels of 5 km or less.
  - If S4's Yunlin test of InSAR seasonal amplitude passes, that amplitude becomes a rank
    validator of π inside Changhua (never a prior; ledger row in 4.1).
- **Outputs.** `changhua_prior/{preorigin,full}/v1/`:
  - `pi_maps.npz`: the primary map (crop-dasymetric if it passes, otherwise LUIMAP-RF) and the
    diagnostic maps, each with a 200-member township-block bootstrap;
  - `township_crop_areas.parquet` (2012-2025; the pre-origin variant stops at 2022);
  - `features.parquet`;
  - `shift_diagnostics.json`;
  - `transfer_battery.csv`, including the crop-dasymetric rows;
  - `phi_in.json` (value and uncertainty).
  The Nantou apex prior (about 6.5 GWh/yr) is included.
- **Prior-width input only, never the level.** The prior's in-fan Changhua totals:
  - LUIMAP models: 116-194 GWh/yr;
  - crop-dasymetric: 95-112 GWh/yr;
  - for comparison, Yunlin's in-fan census: 183 GWh/yr.
  These widen the μ prior (P1 step 4) and have a ledger row saying so.
- **Compute.** About 30-40 min with 4 workers for the bootstraps (inside the CPU budget, 5.7).
  **Effort.** 3 agent-days.
- **Risks.**
  - The canal-substitution effect cannot be learned inside Yunlin.
  - LUIMAP is a single 2024 snapshot.
  - "Dry field" is a field type, not a crop.
  - The prior is smoother than reality (per-cell p10 85 against 10 MWh in the census).

### WP P4. Changhua abstraction inference (centrepiece)

**What the pilots established.** These facts shape the design.

1. **Linearity.** Pumping enters the solver only through the right-hand side, and the
   deliverable's operator is time-invariant. Heads are therefore exactly linear in abstraction.
   A sparse-LU replica matches torch to 1.7e-6 m, reproduces R² 0.8651 and runs 2012-22 in
   0.25 s.
2. **Kernels.** Exact adjoint Green's functions (158 wells × 2,148 cells × 131 lags) take 166 s
   on CPU.
3. **Footprint.** The response is quasi-steady (65 % of the steady gain at lag 0, 96 % by
   lag 5) and broad (33 % of a Changhua well's response comes from within 5 km).
4. **What the data identify: s, not μ.** Heads see only the time-varying stress times the
   response, `s = α·μ·CV·R(ψ)`.
   - The deliverable's per-well datum carries a prior with sd 5 m, but it is nearly free:
     shrinkage median 0.99, minimum 0.96 (`stage3_datum_gate_cwa/stage3_theta.json` meta;
     verified here). The CPU pilots used a fully free datum.
   - Constant abstraction after a 6-month burn-in has relative sd 6.6.
   - The compaction column has no sensitivity to head level (evidence doc §6: about −0.0005 per
     m against +0.0059 for the rings), so leveling cannot see a constant μ either.
   - Consequence: **μ is set by the electricity prior alone.** No head, leveling or subsidence
     data in this plan constrain it.
5. **Burn-in.** Without one, the start-up transient from the observed-head initial condition
   "identifies" the mean (bias −58 % to −100 %). A 12-month burn-in is mandatory.
6. **Head RMSE misleads.** In the pseudo-west region, uniform blocks reached CV RMSE 1.83 m with
   block correlation −0.49, while the true census scored 2.33 m. A 1-3 parameter fit also gives
   cross-validated and in-sample RMSE that are nearly equal (1.1539 against 1.1526 m in
   `p1_channels.json`), so any phase-matched seasonal regressor passes a CV head-RMSE test.
   Designs are selected on recovery of pseudo-truth and on placebo-controlled tests, never on
   head RMSE.
7. **The conflict, as piloted (frozen θ).** In `p1_channels.py` the whole-county TPC series was
   spread over Changhua's in-grid cells (φ_in = 1; verified here).
   - Least-squares scales the heads accept: 0.209 (scaled estimator; formal sd 0.017,
     uncalibrated), 0.156 (shape), 0.099 (transfer). At scale 1 the scaled series raises
     Changhua anomaly RMSE from 1.20 to 1.81 m.
   - Least squares on an imperfect regressor is attenuated. Matching the model's anomaly
     amplitude to the observed one instead gives scales of about 0.50, 0.42 and 0.59 (quadratic
     interpolation of the squared amplitude ratio through 0.42 at zero, 0.59 / 0.57 / 0.44 at
     the least-squares scale and 1.78 / 2.06 / 1.61 at scale 1; approximate; verified here).
   - The fit is linear in the scale, so against the in-grid series every scale divides exactly
     by φ_in (0.75-0.82): 0.12-0.28 by least squares, 0.51-0.79 amplitude-matched.
8. **A misspecification flag.** When a mean channel was added, the heads preferred *negative*
   Changhua abstraction: θ_mean = −0.187 ± 0.049 (scaled) and −0.191 ± 0.050 (shape)
   (`p1_channels.json`, verified here). With μ ≥ 0 the bound would be active and a Laplace sd
   invalid. The sign points to start-up or initial-condition transients or an unmodelled trend.
   E10 investigates it before any fixed-datum or level variant is used.
9. **The adopted design's own stress test** (`p3_burnin_stress.json`, frozen θ, 12-month
   burn-in, truth with half the region's temporal variation, mean prior sd 0.3; verified here):

   | Region | Time variation recovered (× truth) | Mean bias, prior × 0.7 | prior × 1.0 | prior × 1.3 |
   |---|---|---|---|---|
   | west | 1.20 | −33 % | −5 % | +21 % |
   | mid | 1.48 | −39 % | −19 % | −2 % |
   | east | 3.26 | −40 % | −20 % | −4 % |
   | north | 0.94 | −43 % | −27 % | −17 % |

   Between prior × 0.7 and × 1.3, 0.43-0.89 of the prior's error passed into the estimate.
10. **The full range of pilot biases** (12-month burn-in; prior-only and density-transfer
    baselines excluded) is −71 % (borrowed shape, half variation, north) to +47 % (four blocks,
    south half) (`p3_burnin_stress.json`, `p2_pseudo_region.json`; verified here). The range
    −16 % to +23 % quoted in the first draft holds only for heads-only designs whose borrowed
    shape matches the truth's variation. In the east tercile and the north half the true census
    does not beat zero pumping on heads (2.071 against 2.077 m; 1.386 against 1.360 m), so those
    regions carry no pumping information.

**Adopted design.** Both judges recommended the same combination:

- **Backbone.** Design 1: zone amplitudes × a land-use/crop shape × Yunlin temporal variation,
  with the head-blind mean taken from an electricity prior.
- **From Design 3.** The β-sweep robust conflict harness. The digitized WRA chart, as a
  validation-only check.
- **From Design 2.** The transfer-validated allocator (in P3), the scoring rules, and the
  stress-discrimination refits.
- **Naming.** The output is named **net seasonal stress attributable to Changhua abstraction,
  at zone scale**, unless experiment E4 shows that pumping and co-phased percolation separate.
- **Designs compared on the fresh units** are limited to three, declared in advance: Design 1
  (decomposed, μ from the prior), the heads-only borrowed-shape design, and land-use-only as the
  baseline.

**Formulation.**

Changhua energy, in model units, per cell c and month t, with a log link so that it cannot go
negative:

```
E_c(t) = Σ_k Σ_j π_kj(c) · μ_kj · exp(α_kj·CV_j·v_j(t)) / mean_t[exp(α_kj·CV_j·v_j(t))]
```

- The time mean of E is μ exactly. The linear form `μ·[1 + α·CV·v(t)]` of the first draft goes
  negative when `α·CV·v(t) < −1`; the product reports how often that would have happened.
- **Zones k** (counts verified on the grid by the design pilot; these are design-time choices):
  - Z1: proximal, x ≥ 205 km, 139 cells, 13 wells.
  - Z2: within 10 km of the county border on the river, 300 cells, 24 wells.
  - Z3: the rest, 515 cells, 37 wells.
- **Classes j:** irrigation, and other (aquaculture + livestock + domestic).
- **Within-zone shape** π_kj(c): the primary P3 map, with the diagnostic maps as sensitivity.
- **Temporal variation** v_j(t): Yunlin class electricity from P2a (Chow-Lin, pre-origin
  variant), zero mean and unit sd. CV_j recomputed from the same product (P1 step 7).
- **Conversion to volume**, the twin's own: `q = η·E·3.6e6 / (ρ g max(z_g − h_1 + 40 m, 2 m))`.

Heads at well w:

```
ĥ_w(t) = h0_w(t; ψ) − Σ_c Σ_τ K[w,c,τ](ψ) · q_c(t − τ) + κ_p · Σ_c Σ_τ G0[w,c,τ] · P_c(t − τ)
y_w(t) = ĥ_w(t) + d_w + e_w(t)
```

- d_w is the per-well datum. **One convention everywhere:** the CPU engine uses the same datum
  prior as `fit_flow` (sd 5 m, the same months-per-observation weighting), so the P4.4
  consistency check compares like with like. The fully free datum is reported as a
  sensitivity.
- e_w is AR(1) with φ 0.743 and σ 1.41 m.
- P_c is a crop-calendar paddy-percolation basis built from the W1 monthly deliveries; it is
  used in E4 and kept if E4 says so.
- Months from 2013-01 on enter the fit (12-month burn-in), unless a consistent 2007-2011
  spin-up is used instead (E10).

Priors:

| Parameter | Prior | Basis |
|---|---|---|
| log μ (total, in grid) | N(log μ_elec,in-grid, σ_μ²) | `μ_elec,in-grid = a_pre · YL_clean · φ_in` (P1 step 5). σ_μ² = 0.5² + σ_φ² + any P1 misspecification bias². A log-sd of 0.5 places the whole estimator spread, 0.5-1.27 × Yunlin, within about ±1.3 sd of the centre; the crop-statistics total enters only as a prior-width input. |
| log α | N(0, 0.5²) | — |
| between zones | smoothness, sd 0.3 | — |

**What may and may not be reported:**

- The identified quantity is `s_kj = α_kj · μ_kj · CV_j · R_k(ψ)`.
- μ is prior-only, so α = s/(μ·CV·R) inherits the prior's error inversely. Neither μ nor α is
  ever reported as data-driven.
- **Policy outputs depend on μ**, because a 30 % cut removes 0.3 μ. Every Changhua policy number
  is published as a table over μ ∈ {0.5, 0.75, 1.0, 1.27} × μ_elec,in-grid, with α derived
  from s at each μ, and over the layer split f. The table states that no head, leveling or
  subsidence data in this plan constrain μ.

**Conflict statistic (one fixed definition, in s):**

```
z = (ln s_H − ln s_E) / sqrt(τ_H² + σ_E²)
```

- s_H: the heads-only estimate of the seasonal stress, **de-attenuated**: the E5
  errors-in-variables correction, with amplitude matching and the instrumental-variable fit as
  cross-checks. The least-squares estimate is reported beside it, labelled "attenuated".
- s_E: the electricity-implied seasonal stress, μ_elec,in-grid at Yunlin's seasonality (α = 1).
- σ_E = σ_μ.
- τ_H: the empirical log error of s_H from the **strict** pseudo tests (P4.3), given a posterior
  rather than a point value: scaled inverse-χ² with ν = (number of disjoint fresh regions − 1)
  = 4 degrees of freedom, inflated by κ_shift = (RMS log error on the Changhua-like fresh
  regions) / (RMS log error on the 5 disjoint regions), floored at 1. Until strict tests exist,
  the frozen-θ disjoint-region value is used, labelled optimistic.
- Thresholds use the posterior predictive: z is compared with Student-t quantiles on ν = 4
  (|z| < 2.78 for the two-sided 95 % level), not with ±2.

The pilots reported z anywhere from −0.46 to −4.3 depending on definition. Only this definition
may be reported, and only after E5 and the strict tests.

#### P4.0 Linear-response engine (CPU, staging; rebuilt in Phase 1)

- **Port** the physics scout's code (`b_operator.py`, `c_greens.py`, `inv_lib.py`):
  - `assemble_operator`, with cell-masked T/S/leakance multipliers;
  - `SparseTwin`: LU factorisation, a stepwise rollout with lift feedback, and
    `verify_against_torch` (≤ 1e-5 m);
  - `adjoint_greens`.
- **Add** a stepwise sparse rollout for time-varying cases (RIV switching, lift feedback on the
  added abstraction). The convolution engine is invalid there.
- **Add** the `fit_flow` datum prior (P4 formulation) and the forcing-uncertainty propagation
  `Var h = K Σ_q Kᵀ` used by section 4.8.
- **Rebuild** on batch 8's merged `cwa` θ once the batch finishes. The scouts used `fit_full`
  while folds 2-4 were still running, so every E1 and E3 conclusion is provisional until rerun
  on the merged θ.
- **Bound-pinned parameters.** In the `cwa` θ the distal log T sits at its upper bound in 3 of 4
  layers, leakance at its lower bound in 3 of 6 zone-layers (1 mid, 2 distal), and the pumping
  spread at its 0.5 km floor (`bounds_hit`; verified here). Every identifiability report and
  Laplace sd treats them as non-identified: they are profiled, or their bounds are widened in
  a sensitivity, never differentiated at the bound.
- **Compute.** 0.09 s per LU, 0.25 s per rollout, 166 s per kernel set. About 1.4 GB RAM in
  float64, or 178 MB per kernel in float32 (mmap-able).
- **Acceptance.** Sparse rollout matches torch within 1e-5 m on the merged θ; adjoint kernels
  match a forward impulse within 1e-6 relative.

#### P4.1 Decisive CPU experiments (Phase 0)

Each experiment states its interpretation before it is run. Pre-registration is via A1, and it
names the fresh units (A1) each experiment is scored on. E5 runs first: no conflict number is
quoted before it.

| id | Experiment | Interpretation fixed in advance | Cost |
|---|---|---|---|
| E1 | **Strict-proxy pseudo test.** Withhold the census in each fresh tercile and in the two-tercile region, refit the shared zonal parameters with zero pumping in the region, then run the heads-only inversion. Primary route: zonal log T, log S, leakance, recharge fraction and coast conductance refit on the sparse replica. Cross-check for one region: a warm-started `fit_flow` on CPU torch (read-only import), run inside the CPU budget (5.7). Report the θ shift for every region. | If heads-only underestimates the region by 2× or more, the conflict is plausibly a calibration artefact. If it recovers within the tolerance (P4.3), the heads route is not biased by calibration in Yunlin. Provisional until rerun on the merged θ; superseded by the strict GPU refits of P4.3. | 1-3 h CPU per region (sparse); about 7 h for the torch cross-check at 4 threads, alone in the budget |
| E2 | **Policy invariance.** Compute the cut30 Changhua head change and the steady cone under M_heads (heads-only s, base response) and under electricity-level s with the response variants that keep the base fit (f 0.5 with T×2; Lk×10; T×4), each across the μ grid. | If they agree within 15 %, the magnitude dispute is mainly about volumes, and GPU effort goes to α and layer allocation. Otherwise the policy span carries the factor. Design-3 pilot numbers already show heads-optimal cones of 1.74-2.20 m against 4.55-7.23 m for electricity-level Q. | minutes |
| E3 | **Response-ridge identifiability in Yunlin.** For a pseudo region with metered truth, fix pumping at truth × {0.3, 1, 3} and fit region-masked {f, log T, log Lk}. | If site-CV head skill cannot pick out × 1, Changhua-specific response multipliers are not identifiable from heads and may not be freed without independent priors. | about 20 min (17 s per operator variant) |
| E4 | **Paddy-percolation co-estimation.** v4 monthly deliveries × crop-1/crop-2 calendar, co-estimated with the pumping amplitude. | If the pumping amplitude moves by more than 30 %, the product is reported as net stress, and the percolation term stays in the design. | minutes |
| E5 | **De-attenuation (runs first).** (a) Errors-in-variables: synthetic Changhua truth = common signal + Changhua-specific deviation, variance matched to the TPC Changhua-vs-Yunlin anomaly correlation (r 0.84-0.88); invert with the Yunlin shape; measure the attenuation of s. (b) Amplitude matching on the real heads. (c) Instrumental variables: TPC Changhua agricultural anomalies as the instrument for Changhua stress timing. | The attenuation factor and its uncertainty de-attenuate s_H. If (a)-(c) disagree by more than their uncertainties, z is not reported and the gate uses the bracket they span. Using TPC Changhua anomalies as an instrument makes them an input to s_H, so they stop being a free validator of interannual amplitude (ledger, 4.1). | under 1 h |
| E6 | **Robust fusion β-sweep** on the 5 fresh disjoint regions. Electricity spread widened to 0.5-1.27×; β ∈ {0.3, 0.5, 0.7, 1, 1.4, 2, 3}; fixed z definition; Gaussian and robust-mixture fusion. Includes a pseudo test whose prior is built the Changhua way: TPC Yunlin agriculture × r(moy) × the region's land-use share × the pump-like fraction, not a multiple of the truth. | Report coverage, conflict-detection rate, effective number of regions, the τ_H posterior and κ_shift. Design 3's 12 overlapping regions gave τ 0.246-0.285, which is not an independent estimate. | under 1 h |
| E7 | **WRA chart verification.** Independently re-digitize the 2021 briefing chart 灌溉抽水量比較 (Jan-May irrigation pumping, ROC 100-110, 依納管水井用電度數及水電比推估) and cross-check the overlapping years in the 2023 briefing chart. Test whether the series is a scaled copy of Yunlin: ratio 0.80 ± 0.03, r 0.95 (0.91 detrended), and only r 0.68 with TPC Changhua agriculture (n = 6). Back-calculate the Changhua well electricity the chart implies from WRA's water-to-electricity ratio (8.43 m³ per census kWh in Yunlin, design pilot `p6_wra_janmay.json`). | Validation only. The chart may enter a likelihood only after WRA supplies the table and method (R-b). | under 1 h |
| E8 | **Tidal response, Changhua against Yunlin.** At wells with M2 coherence > 0.8 (45 % of wells; layers 3-4 median 0.95), compare the M2 response in the same layer, after the barometric correction of H1. | Expected to be inconclusive. M2 amplitude decays 0.04-0.12 per km inland in both counties, which looks like ocean loading, not diffusion, and phases scatter by 65-115°. A T_CH ≈ 2× claim needs proper earth-tide and loading modelling. Until then no Changhua T multiplier is freed. | about 1 h |
| E9 | **Changhua nested-site layer ratio.** Layer-1/layer-2 response of Changhua nested sites to the regional pumping shape, on local-only slopes, compared with Yunlin's local-only ratio (P2f rerun). | Independent evidence on the shallow split f for Changhua, and one of the seasonality indicators of the P4.3 gate. | under 1 h |
| E10 | **Mean-channel sign.** Refit the mean, seasonal and anomaly channels with a 24-month burn-in, with a per-well linear trend nuisance, and with an initial condition from a consistent 2007-2011 spin-up. | If θ_mean stays significantly negative in all three, the level channel is misspecified: no fixed-datum or level variant is used, and the issue is logged in 7.3. If it turns non-negative under one of them, that treatment becomes part of the design. | under 1 h |
| E11 | **Gating V1 tests (CV mismatch and μ-prior emulation).** In each fresh region: rescale the withheld census's seasonal amplitude by 0.5 and 1.5 while keeping its mean; generate heads with the sparse replica plus block-resampled real residuals; draw the μ prior from the region's true total perturbed log-normally with sd 0.5, and from the Changhua-way construction of E6. Run the adopted design end to end. | Pass if the recovered s is within ±35 % of truth in at least 4 of 5 regions under every rescaling, α is within ±50 % of truth when the true μ is supplied, and the 90 % interval of s covers truth at a rate whose cluster-adjusted interval contains 0.90. A failure is reported as a limit on s, and P4.3 cannot reach Outcome B. | under 1 h |

#### P4.2 Product v0 (CPU, Phase 0)

- **Members**, written as `changhua_abstraction/v0/E_extra_<member>.npz`. Each holds per-class
  model-unit kWh `(classes, 2148, months)` for 2012-01..2026-08, dates, dx, n_active, the
  member definition, prior and posterior summaries, flags and sha256 provenance.
  - **M_heads:** heads-only MAP of s, with μ at the electricity prior and τ_H as its interval.
  - **M_elec:** s at the electricity anchor (α = 1).
  - **M_post:** the posterior, with the conflict z attached once it may be reported.
- **Z2 (river band)** is no longer prior-only. A prior-only Z2 would sit at α ≈ 1 and μ at the
  electricity anchor, the configuration that raised Changhua anomaly RMSE from 1.20 to 1.81 m
  with amplitude ratio 1.78 (`p1_channels.json` scale 1), on 300 cells and 24 of the 74 wells.
  Two members instead, until the river-band pseudo test passes (P4.3b):
  - **Z2-borrow:** α from the hierarchical posterior of Z1 and Z3, with its sd doubled;
  - **Z2-mean:** μ at the prior, zero anomaly (no seasonal stress).
  The pilots gave −32 % to −58 % bias in the river-adjacent north half, where the true census
  did not improve heads (1.386 against 1.360 m at zero pumping).
- **Scoring:**
  - training-mean datum;
  - site-grouped 5-fold CV over Changhua wells on the fresh folds (10 km clusters, 5 km buffer);
  - **placebo-controlled tests on the fresh units:** the electricity-derived v(t) must beat (i)
    its own 12-month climatology, which isolates the seasonal part, (ii) a rain-deficit or
    Jiji-inflow anomaly index, which tests drought confounding, and (iii) v(t) shifted by six
    months. This is what shows whether the interannual electricity timing is detected at all;
  - the fresh temporal block (2016-18) as an amplitude-only hold-out, labelled "θ in-sample"
    until the masked GPU refit of P4.3 exists; the 2020-22 hold-out is reported as design-time
    evidence;
  - the identifiability report (Fisher sd after the datum, resolution diagonal, trade-off
    correlations with log T, log S, recharge and κ_p, burn-in sensitivity), with bound-pinned
    parameters profiled. Fisher sds are labelled uncalibrated: pilot errors reach up to 28 times
    the 8 % zone sd.
- **Acceptance for entering P4.3:**
  - frozen-θ pseudo |bias of s| < 30 % in at least 2 of the 3 fresh terciles, labelled
    optimistic;
  - E11 passed;
  - v(t) beats all three placebos in the site-grouped CV;
  - E1-E10 reported.
- **Pilot reference values**, all frozen θ, design-time units:

  | Case | Changhua magnitude, m³/day | Anomaly RMSE (CV) | Amplitude ratio |
  |---|---|---|---|
  | Zero pumping | — | 1.203 | 0.42 |
  | Single land-use amplitude | 3.5e5 ± 0.25e5 | 1.154 | — |
  | 10 blocks × 2 shapes | 4.45e5 ± 0.53e5 | 1.139 | 0.64 |

  The 2020-22 hold-out gave 1.306 at zero pumping against 1.131 for the inversion, and the
  estimate fell to 3.66e5 when fitted on data through 2019. The ± values are formal and
  uncalibrated.

#### P4.3 Decisive GPU tests (Phase 1, after batch 8 and the ladder's R*)

Run under 3-way MPS on top of R*:

- **(a) Strict pseudo-Changhua.** Refit θ with the census withheld (heads kept) in each fresh
  tercile, and in the two-tercile region under two Changhua baselines: zero (as calibrated) and
  a fixed mid-level member. Then run the heads-only and fused inversions. Record the θ shift of
  every refit (zonal log T, log S, leakance, recharge fraction, coast conductance). 5 jobs.
- **(b) RIV refit plus the river-band pseudo test.** 1-2 jobs; RIV run time is unverified.
  With RIV the operator varies in time, so inversions use the torch adjoint or the stepwise
  sparse rollout.
- **(c) Seasonal-stress discrimination.** Changhua E fixed so that s is 0.25×, 0.6× and 1.0× the
  electricity-implied s_E, with θ refit each time and per-layer abstraction fractions freed
  fan-wide (the one response change with independent support, P2f). Each level also runs with
  the fresh 2016-18 block masked (a Phase-1 hook, `--mask-months`). Changhua-specific T or
  leakance multipliers are freed only with independent priors (E8, E9). 6 jobs. The θ shift
  between levels is the **real-case θ shift** that (a) must span.

**Scores that drive the gate** (named, with weights, in the pre-registration; all pre-origin):

- out-of-fold Changhua heads 2012-22 on the fresh site folds (anomaly RMSE and amplitude);
- the masked 2016-18 block at Changhua wells, against the three placebos of P4.2;
- out-of-fold Changhua leveling 2012-21 (spatial K-fold, buffer ≥ 10 km), against the
  zero-Changhua baseline computed under the same folds, tested by a spatial block bootstrap;
- ring and pile increments up to 2021, where available.

The 2023-25 forcing experiment, the post-origin piles, the bridged rings 2023-25, NGL and WRA
GNSS from 2022, and InSAR 2022-26 are reported for every level and member but never enter the
gate (U7).

**Decision gate** (pre-registered, stated in s):

| Outcome | Condition | Action |
|---|---|---|
| A, calibration artefact | Strict heads-only s is biased low by 2× or more in the fresh regions (log error below −0.69 and more than 2 τ_H below zero), and the pseudo θ shifts span the real-case shift | M_elec is primary for s; M_heads is kept as a labelled sensitivity member; joint refit (P4.4) |
| B, heads trustworthy | Strict heads-only s within tolerance (absolute log error below max(0.30, 2 τ_H)), E11 passed, pseudo θ shifts span the real-case shift, the de-attenuated z is below the t₄ lower 2.5 % quantile, and the gate scores favour the lower s level | s from heads (M_heads is primary for s). μ stays at the electricity prior. Three hypotheses remain open: different seasonality, a wrong pump-like share or conversion, and river under-response (closed only if (b) passed). The independent seasonality evidence below decides between the first two |
| C, unrepresentative or inconclusive | The pseudo θ shifts do not span the real-case shift (the real shift exceeds the largest pseudo shift by more than 1.5×), or the scores do not separate the levels, or neither A nor B holds | Ship M_heads and M_elec. Changhua policy numbers are published as functions of μ and α. Data requests are escalated |

**Independent seasonality evidence** (decides "different seasonality" against "wrong pump-like
share", only after Outcome B; validators only, ledger rows in 4.1):

- the seasonal coefficient of variation of the Changhua layer 1-2 daily-cycle index (DN, S1)
  relative to Yunlin's (P5), a head-signal proxy for α that does not depend on level;
- the TPC Changhua/Yunlin month-of-year ratio (1.09 in February to 1.23 in August-September);
- Changhua's crop calendar and canal timing (W1: crop-1 share 0.50 against Yunlin's 0.30);
- the Changhua nested-site layer ratio (E9);
- the WRA chart (E7), whose Jan-May window limits it to the spring part of the cycle.

"Different seasonality" is rejected only if at least two of these indicators show Changhua's
seasonal variation at 0.75 or more of Yunlin's. Only then is the electricity pump-like share or
the volume conversion flagged as wrong for Changhua and investigated with R-a and R-b. Otherwise
both hypotheses stay open, and the outputs are given over the (μ, α) table.

**Acceptance for any claim about s** (strict variant):

- |mean bias of s| < 30 % in every fresh tercile, and < 40 % in the river band;
- better than land-use-only on bias in every region;
- block-density correlation > 0.5 before any within-zone pattern is claimed (the pilots fail at
  about 100-cell blocks).

#### P4.4 Joint refit and product v1 (Phase 2)

- **Refit.** Co-estimate `theta['log_mu']` and `theta['log_alpha']` with their prior penalties
  inside `fit_flow._forward`, with `_rollout` unchanged; the adjoint gives their gradients
  without extra solves. Alternatively fix E_extra at the P4.3-selected member(s). Switch on RIV
  and sw v4 if accepted in I1. Then fit plus 5 folds.
- **Consistency.** Recompute K for the joint θ. The CPU MAP, under the same datum prior, must
  agree with the GPU estimate within one posterior sd; otherwise run one more alternation.
- **Uncertainty.** Recalibrate τ_H (posterior and κ_shift), the block and class-share prior sds,
  and the interval coverage on the strict pseudo tests. The frozen values (κ 0.022, τ 0.273) are
  optimistic, and joint coverage was only 0.33-0.5. Forcing uncertainty is propagated into the
  head and subsidence intervals (4.8).
- **Ensemble.** If |z| is below the t₄ 97.5 % quantile, a single posterior is kept. Otherwise
  the gate's members remain separate forward members, combined with equal weight unless a
  stacking rule on pre-origin scores was declared in the pre-registration.
- **Downstream.** Refit the column on leveling, out of fold over Changhua benchmarks, because
  the column pairs with its head field. Then run the forward and the scenarios, with the μ table.

#### P4 validation summary

| id | Test | Circularity guard |
|---|---|---|
| V1 | Pseudo-Changhua in Yunlin on the fresh terciles, the two-tercile region, the river band and the north half; frozen and strict; plus the gating E11 tests (CV mismatch × 0.5 and × 1.5, μ-prior emulation with log-sd 0.5 and the Changhua-way prior) | Census, π and v(t) built outside the region. The adopted μ-prior design is tested, not only the heads-only estimator |
| V2 | Temporal hold-out on the fresh 2016-18 block with θ refitted (masked GPU refit); the electricity v(t) must beat its 12-month climatology, a drought index and a 6-month shift. The 2020-22 hold-out is design-time | Training-mean datum. Before the masked refit exists, labelled "amplitude-only hold-out, θ in-sample" |
| V3 | Retrospective 2023-26 at the 74 Changhua test wells: RMSE ratio to climatology, amplitude ratio, 90 % coverage (target 0.90, cluster-adjusted interval, 4.4). Yunlin must not degrade by more than 2 % RMSE | 2023-26 heads never used in inversion, refit, column, priors, gates or design choice. Forcing is the posterior amplitudes × v(t) from census and TPC only. Abstraction anomalies are taken relative to the pre-origin mean at restart. Primary model pre-registered (I1) |
| V4 | Withheld Changhua wells (10 km clusters, 5 km buffer) | Reported as a transfer check, not as skill (10 km footprint) |
| V5 | Independent subsidence: out-of-fold Changhua leveling against the zero-Changhua baseline recomputed under identical folds with a ≥ 10 km buffer, tested by spatial block bootstrap (the first draft's absolute 0.3 threshold against an in-sample 0.11 is withdrawn); 溪州 pile; bridged rings; NGL GNSS; InSAR 2022-26 (S4) | None of these enters any fit, gate or member choice |
| V6 | Cross-checks, reported but not gated: TPC Changhua annual anomalies against head-inferred interannual amplitudes (unless E5 uses them as an instrument); WRA chart (E7); layer 1-2 daily-cycle rank (P5); InSAR seasonal amplitude rank (S4, if its Yunlin test passes) | Each source has one role per number |

#### P4 expected accuracy (zone or county scale; conditional on the twin's structure)

| Quantity | Expected | Basis |
|---|---|---|
| Seasonal stress s, 3 zones (identified) | Time variation recovered at 0.94-3.26 × truth in the adopted design's stress test (frozen θ, optimistic). The Fisher relative sd (8 % irrigation, 17 % other) is uncalibrated: pilot errors reach up to 28 times it | `p3_burnin_stress.json`; physics scout |
| Time-mean abstraction μ | Prior only. Electricity anchor with log-sd ≥ 0.5 plus φ_in uncertainty. 0.43-0.89 of any error in the anchor's pump-like share passes into the product's mean, and that share cannot be tested with public data; only R-a and R-b can | `p3_burnin_stress.json`; P1 |
| Region-total bias, all designs | −71 % to +47 % across designs and regions (frozen θ, design-time units). −16 % to +23 % only for heads-only designs whose borrowed shape matches the truth's variation | `p2_pseudo_region.json`, `p3_burnin_stress.json`, `d_pseudo_changhua.json` |
| Heads against electricity in s | In grid, 0.12-0.28 (least squares, attenuated) or 0.51-0.79 (amplitude-matched, approximate) of the electricity series (frozen θ). Not a calibrated conflict until E5 and the strict tests | P4 fact 7 |
| Timing | Borrowed. Annual correlation of region totals 0.92-0.99, monthly 0.70-0.96; seasonal amplitude can be off by 1.1-3× | Pseudo tests |
| Within-zone pattern | Prior only; no cell-level claims | Block-density correlation fails |
| Changhua head skill | CV anomaly RMSE 1.203 → 1.13-1.16 m; amplitude ratio 0.42 → 0.59-0.72; 2020-22 hold-out (design-time) 1.306 → 1.11-1.14. These cannot detect a wrong magnitude (fact 6) | Frozen θ |
| Changhua cut30 response | Proportional to μ, which is prior-only: a factor of about 1.65 per prior sd from μ alone, more if E2 shows the Q/response split matters. Published over the μ table | P4 formulation |

- **Compute.**
  - CPU: P4.0-P4.2 under 3 h in total, plus E1 at 3-10 h, inside the CPU budget (5.7).
  - GPU: P4.3 is 12-13 jobs (about 46-50 job-hours) and P4.4 is 6 jobs (about 23 job-hours),
    plus optional alternation and member refits (6-12 jobs, about 23-46 job-hours). Measured
    3.85 h per 500-epoch job under 3-way MPS. One budget table: section 5.6.
- **Effort.** P4.0 1 day; P4.1 4-6; P4.2 2-3; P4.3 2 plus GPU; P4.4 3 plus GPU.
- **Risks.**
  - Start-up artefact (E10).
  - River-adjacent under-response.
  - CV mismatch: Changhua flatter than Yunlin (now a gating test, E11).
  - Structural error absorbed as pumping: the inversion beat the true census on heads in two
    pseudo regions.
  - θ bias from zero-Changhua calibration.
  - Small samples: 74 wells, 13-37 per zone; 5 disjoint regions give τ_H 4 degrees of freedom.
  - The 22 swapped-coordinate stations that `heads.py` drops (fixed in H1 and I1 before the
    final inversion).
  - Any use of 2023-26 observations in a gate or choice invalidates V3 and V5.

### WP P5. Head-signal pumping signatures (signal processing, ML and DL benchmark arm)

**Gap.** The user asked whether pumping, as electricity, can be extracted from heads by reverse
engineering, signal processing, ML or DL, training in Yunlin and applying to Changhua. The
head-signature scout tested this directly on 128 Yunlin wells with 22 monthly head indices.

**What it found:**

- **The monthly level fingerprint is mostly regional.** The median within-well correlation with
  local kWh anomalies is −0.66 to −0.72, but −0.59 with kWh more than 15 km away and −0.71 with
  the Yunlin total. The part specific to the local area is −0.19.
- **The daily cycle carries a cross-well signal only in shallow layers.** Spearman of the
  tide-corrected S1 amplitude against local kWh is 0.55-0.61 in layer 1 (n = 22) and 0.33-0.41
  in layer 2 (n = 58). Month to month it barely follows local kWh (median r 0.03-0.08).
- **Cross-validated prediction is weak.** Predicting local kWh from head indices gives negative
  R² for pooled ML. Layers 1-2 with a single feature give R² 0.11-0.22, against a 95 %
  permutation null of 0.02 or less.
- **The MLP failed** the transfer splits in the design pilots, with total ratios up to 6.1.
- **Changhua has the same daytime-pumping signature.** In layer 2 the head minimum falls at
  19.1 h against 18.2 h, phase coherence is 0.87 against 0.85, and the amplitude is about 55-80 %
  of Yunlin's. Its seasonal timing follows the Yunlin irrigation calendar (r 0.65).

**Verdict.** Heads give Changhua pumping its **presence and timing**, not its magnitude. The
physics inversion (P4) is the only head-based magnitude route. P5 therefore builds benchmarks
and validators, not the forcing.

**Technique.**

1. **Head-index table (L2-derived).** Per well-month, 2012-2026, both counties:
   - tide-corrected S1 amplitude (cubic B-spline trend with 3-day knots, plus harmonics Q1, O1,
     S1, N2, M2, S2, S3, S4; K1+P1 predicted from O1);
   - DN (day-minus-night head change);
   - daily range;
   - 30-min drop events;
   - spring recession slope;
   - seasonal drawdown;
   - nested-pair head difference.
   The vectorised extractor exists; set OMP_NUM_THREADS=1.
2. **Transfer-function models per well.** `h(t) = Σ_i (θ_i * x_i)(t) + noise`, with gamma or
   Hantush response functions fitted with scipy to rain, ET0, regional per-class pumping
   (v_j) and v4 deliveries. The fitted pumping gains are a second, model-independent relative
   intensity measure. No new package is needed.
3. **Signal-decomposition benchmark.** EEMD/HHT plus EOF on first-differenced heads, as in
   Tseng, Lin & Yu (2024, J. Hydrology 633). That paper's method and numbers were checked only
   at abstract level and are unverified.
4. **Tidal M2 response** for T and S. This feeds E8 and needs an ocean-loading model to be
   usable.
5. **Deep learning.** A temporal-convolution or recurrent encoder from daily head anomalies to
   local kWh anomalies, run **only** if steps 2-3 show local signal beyond the placebo, and
   only through the same transfer battery and pseudo-region masking. The prior expectation from
   the pilots is no skill beyond the regional signal.
6. **Barometric correction first.** The first draft said atmospheric S1/S2 could not be removed
   for lack of barometric data. Pressure is available: the CWA hourly met feed already sampled
   in `data_fetch_api/forcing_experiment/met-cwa-met1hr-obs-sample.parquet` has an `AirPressure`
   column, and the Open-Meteo archive (the same API as the ET0 cache) returns hourly
   `surface_pressure` at 23.7N 120.4E from 2012-01-01 (both verified here). H1 removes the
   barometric loading per well before any daily-cycle or tidal index is computed here.

**Validation.**

- Yunlin spatial-block CV with 5 km blocks and townships, plus a Changhua-sized masked half.
- Every relation must beat two placebos (kWh more than 15 km away, and the regional total) and
  a feature-permutation null.
- The local-only partial correlation, with the regional series regressed out first, is always
  reported.
- **Acceptance for any head-derived intensity claim:** out-of-block R² above the permutation
  95th percentile and above the placebo, in at least 4 of 5 folds.

**Role.**

- **Validation-only rank checks for P4:**
  - posterior local density against the Changhua layer 1-2 S1/DN ranks, compared with the same
    statistic computed in Yunlin from the census (0.33-0.61);
  - the Changhua DN seasonality (peaks in March and October).
- **Seasonality evidence for the P4.3 gate:** the seasonal coefficient of variation of the
  Changhua layer 1-2 daily-cycle index relative to Yunlin's, a level-free proxy for α. It is a
  validator there, so it may not also become a prior for α.
- If a head-derived quantity is ever declared as a prior, it stops being a validator for that
  number.

**Outputs, compute and effort.**

- **Outputs.** `head_signatures/v1/`: `well_month_features.parquet` (barometrically corrected),
  `tf_gains.csv`, `seasonal_cv.csv`, `benchmarks.json`.
- **Compute.** CPU hours inside the budget (5.7); the scout's extractor ran on a machine at load
  average about 33.
- **Effort.** 4 agent-days.
- **Acceptance.** The extractor reproduces the scout's indices on the hourly and 10-minute
  overlap (Spearman ≥ 0.95 for daily range and corrected S1, as in the pilot), and the
  barometric step meets the H1 acceptance.
- **Risks.**
  - Daily cycles from ET in shallow wells.
  - Barometric efficiency is estimated per well and may be poorly constrained at quiet wells.
  - O1 is poorly resolved at most wells (K1 correction noise 2-5 mm).
  - Aquaculture pumps run round the clock and leave no daily cycle.

### WP P6. Public-supply and industrial deep wells

- **Gap.** The census holds 430 public-supply (公共用水), 326 industrial (工業用水) and 91
  domestic-and-public pumps, but only 9 pumps of 50 HP or more (maximum 100 HP) and 1,341 above
  10 HP (verified here on the `PURPOSE` and `PUMP_HP` columns only). Large Taiwan Water Corp
  (台水), Taiwan Sugar (台糖) and irrigation-association public deep wells are therefore
  essentially unmodelled in both counties. These wells are deep, which matters for deep
  compaction, and their number changed over time. Verified by the TPC scout in the MOEA 2019
  plan annex: 774 public deep wells disposed of in 2011-2018, cutting abstraction by
  87.939 Mm³/yr; irrigation-association public-well pumping averaged 60.26 Mm³/yr in Yunlin and
  6.89 Mm³/yr in Changhua over 2001-10. For scale, WRA's Yunlin Jan-May irrigation pumping is
  428.7 Mm³ (design pilot `p6_wra_janmay.json`). The 2022 drought-well inventory lists 49
  Changhua wells (26 irrigation agency, 23 台水 district 11) and 521 in Yunlin (421 irrigation
  agency, 98 台水, 2 台糖) with coordinates but no volumes. Unverified: "1,191 public wells in
  the region, mostly 台水", from a search-result summary only
  (https://www.cna.com.tw/news/ahel/201810140131.aspx, not opened).
- **Technique.**
  1. **Inventory** from the MOEA annex, the WRA 彰雲 briefings (transcription, decision D6b) and
     the drought-well report: counts, closure years, volumes, coordinates where printed. No
     personal data are copied.
  2. **Request R-i** (6.2): monthly per-well abstraction 2007-2026 with coordinates, screen
     depth and closure dates, from 台水 districts 5 and 11 and from WRA water-rights records.
  3. **Until the data arrive,** a fan-scale sensitivity member only: the 2011-18 closure ramp
     placed in layers 3-4, located by the drought-well and public-supply census coordinates as a
     proxy. The member is in physical m³, converted to model units with the WRA-to-model ratio
     (2.58) as a stated assumption. It never enters the primary model.
  4. **When R-i arrives,** a separate deep class in `E_by_class`, allocated to layers 3-4, and a
     ladder rung R6 (I1).
- **Validation.** The closure-ramp response at nearby deep wells and piles is one of the N1
  tests. **Acceptance:** inventory totals reconcile with the MOEA annex totals within 5 %.
- **Outputs.** `public_wells/v1/`: `inventory.csv`, `closures.csv`, `sensitivity_member.npz`.
- **Effort.** 1.5 agent-days now; the rest when R-i is answered.
- **Risks.** The closures fall inside 2012-22, so θ's recharge and boundary terms may already
  absorb part of their trend. Until R-i is answered this is a known missing deep stress (7.3).

### WP N1. Natural experiments for policy response

- **Gap.** The audit lists the absence of intervention-response data as a cause of the weak
  policy evidence. The policy numbers rest on the model's response alone. Public natural
  experiments are already in hand:
  - the 2011-18 public deep-well closures (774 wells, −87.9 Mm³/yr; MOEA annex);
  - 2,403 illegal wells filled by end 2018 (Changhua 1,050, Yunlin 1,353);
  - the 2021 drought crop-1 rotation, with Jan-May 2021 deliveries of 1.43 (Changhua) and
    0.79 (Yunlin) × 10⁸ m³;
  - the 2015 and 2020 spring delivery shortfalls (Jan-May 2015: 1.54 and 1.26; 2020: 2.03 and
    1.26 × 10⁸ m³, against 2016-19 means of 3.59 and 2.49; W1 benchmark);
  - the reported 2018 cuts in irrigation pumping (228.958 Mm³/yr) and aquaculture and livestock
    use (24.06 Mm³/yr) (MOEA annex);
  - the census, rings, piles and heads cover all of these.
- **Technique.**
  1. **Intervention table:** date, location (coordinates or township), volume, source,
     verification status.
  2. **Pre-registered tests** using only pre-origin data and held-out wells or years: does the
     model reproduce the head and compaction response to the 2021 delivery cut, and to the
     closure ramp, at wells and rings near them, against zero-response and climatology
     baselines? These years lie inside the calibration window, so a test counts as skill only
     on held-out wells (fresh site folds) or with the year masked in a refit (the
     `--mask-months` hook); otherwise it is labelled a consistency check.
  3. Effects the tests cannot identify (for example closures without locations) are stated in
     7.3.
- **Acceptance.** Every row of the table has a source; every test is pre-registered before it
  runs; results are reported whichever way they fall.
- **Outputs.** `natural_experiments/v1/`: `interventions.csv`, `tests.json`.
- **Effort.** 2 agent-days. **Depends on** W1, S1a, S2, P4.0, and I1 for the twin runs.

---

### WP H1. Head QC, barometric correction, metadata and coordinate repair (L1)

H1 is split in two so that it does not wait for W3: **H1a** (everything below except the
`GroundHeight` repair) depends only on A0; **H1b** (the `GroundHeight` repair) runs after W3.

- **Gaps** (audit plus the gap-fill scout):
  - dropouts to a constant floor (about −33 or −44 m) at 50 wells;
  - spikes at 96 wells;
  - permanent steps (21 wells in 2012-22; 10 in the test period, for example 09180341
    +80.5 m on 2024-08-06);
  - 273 paired datum-offset episodes of ≥ 1 m at 67 wells in 2012-22, 199 of them lasting
    ≥ 15 days, which the 15-day rolling-median screen cannot remove;
  - 65 unpaired offsets of ≥ 3 m at 46 wells;
  - blocks of −999998 sentinels;
  - barometric loading, never removed: it contaminates the daily-cycle and tidal indices, and a
    typhoon's pressure drop can be mistaken for a datum-offset episode;
  - 22 stations with `LocationByTWD97` stored as "y x", and some at "0 0";
  - `GroundHeight` 0.0 (a missing-value code) at 60 of 158 model wells;
  - 19 stations with no layer code.
- **Technique.**
  - Screens: sentinel removal, 15×MAD despike, flat-run screen, dropout screen (15-day rolling
    median plus floor detection against WellDepth).
  - **Barometric correction.** Fetch hourly pressure for 2012-2026: Open-Meteo
    `surface_pressure` at the well points, with CODIS hourly pressure at the CWA manned stations
    that bracket the fan as a check (station identities taken from the CODIS station list).
    Estimate the barometric efficiency of each well by regression deconvolution (Rasmussen and
    Crawford 1997) or from its S2 response, and remove the loading (flag 1) before the tidal fit
    (P5, E8) and before offset-episode detection.
  - **Offset-episode and step detection.** Paired jumps of ≥ 1 m that return; unpaired steps of
    ≥ 3 m. Each is flagged 9. A step is corrected with flag 1 only when physical metadata or a
    co-located instrument (a nested well at the same site) backs it. A correction corroborated
    only by a leave-out neighbour-regression residual above 4 calibrated sd gets flag 3: the
    corrected value is then neighbour-derived, and U2 keeps it out of calibration targets. Real
    events are kept: post-season recovery of about 1 m/day, and typhoon recharge.
  - **Merge.** Historical cache plus new network. Their 2022-10..12 overlap is identical:
    13,314 daily pairs, maximum difference 0.0.
  - **Aggregates.** Daily and monthly, with clean-day counts and the C20 flag: a day needs
    ≥ 12 h, a month ≥ 20 days. The clean-day count sets σ_obs (U2).
  - **Station metadata repair.** Swapped coordinates detected by range and corrected (flag 1)
    with provenance. Coordinates of "0 0" left missing unless WRA metadata supplies them.
    Layer codes for wells without one are proposed from WellDepth (flag 5, unverified, never
    used without review).
  - **H1b, after W3.** `GroundHeight` 0.0 set to missing and replaced by the W3 DTM at the well
    (flag 5) for depth-to-water and transducer-floor checks.
- **Validation and acceptance.**
  - Injected paired offsets of ≥ 1 m and steps of ≥ 3 m are recovered with recall ≥ 0.9; the
    false-flag rate is reported.
  - No flag on the known real events (09200321 and 09190132, recovery of 28 May - 4 June 2024;
    09070311, typhoon recharge of +4.9 m at the end of July 2024).
  - ≥ 95 % agreement with the audit's `artefacts.csv` events (same well, within ±1 day).
  - Barometric step: the median O1 phase coherence over wells improves, and no flagged offset
    episode coincides with a pressure drop of ≥ 20 hPa.
  - Rescore the frozen challenge on L1 monthly means and report the change; the audit expects
    less than 0.01 in the ratio. This is reported, not an acceptance criterion.
- **Prospective tests.** The registered prospective tests keep their registered QC. Any new
  screen is reported as a secondary score only.
- **Outputs.** `heads/L1_clean/v1/`:
  - `heads_daily.parquet` (sid, date, value, n_samples, flag);
  - `heads_monthly.parquet` (with clean-day counts and σ_obs);
  - `events.csv`;
  - `barometric_efficiency.csv` and `pressure_hourly.parquet`;
  - `station_meta_clean.parquet`: only non-personal columns; contact fields are never copied.
- **Compute.** CPU minutes; OPENBLAS_NUM_THREADS=1. **Effort.** 2.5 agent-days.
- **Risk.** Some offset episodes may be real events. Whether each is an artefact is unverified,
  which is why the default is to flag, not correct.

### WP H2. Head gap filling with calibrated uncertainty (L2)

- **Gaps.**
  - 2012-22, 158 model wells: 2,003 of 20,856 well-months missing (1,543 leading, 450 internal,
    10 trailing).
  - 2023-26: 792 of 7,110 missing (316 of them in the 2026-02-11..03-19 outage, 189 trailing).
  - Partial months carry 0.1-0.3 m of representativeness noise. A 20-day partial mean has RMSE
    0.33 m against the full month.
- **Technique** (the pilot's `lib.py` is a working reference):
  1. **Daily bridge** for multi-week daily gaps and partial months: a linear bridge between the
     3-day means at each edge, plus the well's mean deviation curve from other years. The
     monthly mean is then taken over the completed daily record. Gaussian-equivalent sd
     0.25 m.
  2. **Router.** Where any well is observed in the month, a precision-weighted (1/sd²) average
     of two methods:
     - **neighbour ridge regression**: the top 4 wells with r ≥ 0.6 within 25 km that are
       observed through the gap, with an AR(1) residual bridge and a parameter-uncertainty term;
     - **softImpute**: low-rank completion on standardized heads, with λ chosen by 6-month
       block CV (0.004-0.008 × the top singular value), a temporal bridge of the latent factors
       and an AR(1) residual bridge.
     Where no well is observed and no daily edges exist, a per-well GP (RBF + seasonal +
     exponential + white-noise kernel) or a Kalman smoother with rain.
  3. **Calibrated sd.** Multiply each method's sd by a factor per gap type (floor 0.05 m):

     | Gap type | Factor |
     |---|---|
     | Internal ≤ 2 months | 1.0 |
     | Internal 3-23 months | 1.1 |
     | Internal ≥ 24 months | 1.25 |
     | Leading or trailing edges | 2.5 |
     | Months with no observation anywhere | 1.25 |

  4. **Tested envelope.** Fill only wells with ≥ 36 observed months, internal runs ≤ 24 months
     and edge extrapolation ≤ 66 months. Everything else is flag 8.
  5. **Two versions.** Pre-origin (fitted on 2012-22 only) and full (2012-2026).
  6. **Untested methods, recorded.** A deep imputer (BRITS/SAITS type) was not tested: with
     132 months × 267 wells, the low-rank plus regression methods already capture the shared
     structure, and a deep model would be harder to calibrate. It is an optional benchmark on
     the same masks, never a production method without passing them.
  7. **Initial-condition head field.** The initial condition stays IDW unless kriging with an
     external drift or a GP field beats it by leave-one-well-out on January 2012 heads. The
     12-month burn-in is kept either way.
- **Validation.** Masking families mirror the real gaps (section 4.4). Baselines: the twin's
  backfill (linear in the middle, constant at the edges), per-well climatology,
  climatology + AR(1), same-layer IDW of anomalies, and space-time kriging of standardized
  anomalies. Pilot results, recommended method against backfill (RMSE in m, clean truth):

  | Gap | 1 mo | 2 mo | 3 mo | 6 mo | 12 mo | 24 mo | half network 12 mo | leading 66 mo | trailing 48 mo | daily outage window |
  |---|---|---|---|---|---|---|---|---|---|---|
  | recommended | 0.16 | 0.28 | 0.32 | 0.28 | 0.48 | 0.70 | 0.55 | 1.13 | 0.73 | 0.28 (all truth) |
  | twin backfill (all truth) | 0.59 | 1.42 | 1.30 | 1.97 | 2.64 | 3.30 | 2.85 | 2.62 | 2.36 | 1.05 (monthly interp.) |

  Coverage: the pooled 0.912 was computed with gap-type factors tuned on the same 15,691 test
  cells. The out-of-sample split-conformal coverage, 0.86-0.90 by stratum, is the figure quoted.

  **Acceptance:**
  - coverage passes the cluster-adjusted rule of 4.4 in every stratum;
  - RMSE below the backfill, climatology + AR(1), same-layer IDW and space-time kriging in every
    family where the baseline is defined; a family where a baseline wins falls back to that
    baseline;
  - results reported on both all truth and clean truth.
- **Use.**
  - Initial conditions and IDW head fields (all flags up to 4, pre-origin version).
  - Calibration targets: only daily-bridged months (flag 2), weighted per U2. Flag-3 and flag-4
    fills never enter as targets.
  - Never scoring targets (U1).
  - Feb-Mar 2026 is daily-bridged (282 well-months at sd 0.25 m) for continuity of the
    statistical layer only.
- **Outputs.** `heads/L2_filled_{preorigin,full}/v1/fill.parquet`: head_m, head_sd_m, flag,
  method, gap_run_len, product_version.
- **Compute.** CPU minutes with single-threaded BLAS. **Effort.** 2 agent-days.
- **Risks.**
  - Contaminated truth: the top 1 % of cells carry 46 % of squared error. Offsets can leak into
    other wells through the neighbour regression, so H1a must run first.
  - Long edges stay weak.
  - Using the full fill in calibration would leak test information.

### WP H3. Recovery of excluded wells and missing records

- **Gap.** 186 of 344 stations are not in the model: 170 fail the coverage QC and 16 are off
  the grid. 73 on-grid excluded wells have data (Changhua 22, Yunlin 41, other 10).
- **Inclusion on observed months only.** The first draft counted wells at ≥ 80 % monthly
  coverage *after fills* (20 at calibrated sd ≤ 0.5 m, 30 at ≤ 1.0 m), which let imputed months
  qualify a well, including records ending on 2018-12-31 that were extrapolated 48 months. On
  observed months alone (verified here on the pilot's `recovery_hist.csv`), 16 of the 61 on-grid
  excluded wells with a layer code 1-4 reach ≥ 80 % monthly coverage: Yunlin 13, Changhua 3,
  none with a trailing gap. These 16 are the candidates.
- **Wells extrapolated beyond their record end stay out of calibration** (flag 6 after the
  record end). The 18 wells whose 60-minute records stop on 2018-12-31 enter only if their
  post-2018 records are found.
- **Acquisition.**
  - Probe the station data routes for the post-2018 records of those 18 wells (R-h). Whether
    the records exist is unverified.
  - 6 on-grid wells with full coverage have no layer code; assign one from WellDepth, flagged
    and reviewed (unverified).
- **Reconcile counts.** The brief says 17 Changhua and 30 Yunlin excluded wells have ≥ 30 %
  coverage. The twin's own rule gives 13 / 29 / 10 (Changhua / Yunlin / other), and C20 gives
  8 / 29 / 10. Settle on one definition and record it.
- **Use.** Added wells enter calibration only in the pre-declared rung R5 of the I1 ladder. The
  2023-26 test set stays at the registered 158 wells. Added wells with 2023-26 data can form a
  secondary test set that is reported separately.
- **Acceptance.** Each recovered well passes the H2 coverage strata on its observed months, and
  R5 passes its rung gate.
- **Effort.** 1.5 agent-days. **Risk.** Changhua gains only 3 wells.

### WP H4. Sub-grid observation operator for the high-swing wells

- **Gap.** About 40 high-swing Yunlin wells (the top quartile by observed sd) carry 84 % of the
  excess error, with model sd / observed sd of 0.29 at the median. They sit in pumping clusters
  (median 157-241 census pumps within 1 km for Q3-Q4 against 0 for Q1-Q2), so part of their
  swing is a local cone that a 1 km cell-mean head cannot carry. The first draft planned no fix.
- **Technique.** A data-driven correction of the observation operator, with no solver change:
  `h_well(t) = h_cell(t) − Σ_j Q_j(t)·k(r_j; T, S, ω)`, summed over census pumps within 2-3 km.
  - Q_j from the Chow-Lin product (P2a); T and S from the well's zone (profiled where the zone
    parameter sits at a bound).
  - k: the periodic-Theis kernel |K0(r·sqrt(iωS/T))| for the seasonal part (the head-signature
    scout's kernel) and a Thiem log kernel for the slow part.
  - One or two fitted multipliers, fan-wide or per zone, fitted on 2012-19 residual amplitude.
  - Phase 0 runs it on CPU over the sparse-replica hindcast. After the freeze it becomes a
    scorer and observation hook.
- **Validation.** Held-out years 2020-22 and site-grouped CV. Placebo: the same kernel with the
  pump positions shuffled within the county, or with only pumps more than 15 km away.
  **Acceptance:** Q4 excess falls on held-out years and held-out sites, beats both placebos,
  and Q1-Q3 RMSE does not rise by more than 2 %.
- **Role.** A secondary model for the frozen 2023-26 challenge (I1), never a change to the
  registered scorer. If it fails, 7.3 states that the local cones cannot be recovered with this
  grid.
- **Outputs.** `subgrid_operator/v1/`: per-well correction series, fitted multipliers, report.
- **Effort.** 2 agent-days. **Compute.** CPU minutes.

---

### WP S1. Rings: instrument-change correction and 2022 bridge

S1 is split so that its independent checks do not run before their inputs exist: **S1a**
(product, checks a, b, e; depends on A0 and S2) and **S1b** (independent checks c and d; runs
after S3 and, if approved, S4).

- **Gap.**
  - No survey at any of the 14 sites between December 2021 and February 2023 (北辰: July 2024).
  - After the gap, no ring deeper than 293.9 m is read at any site. Before it, the deepest rings
    were at 280-340 m, which points to a new probe or cable.
  - Each site has a common depth offset (−15.5 to −16.3 cm at 8 sites, +50 to +54 cm at 3) and a
    depth-proportional change.
- **Formulation.** `z_post = a_site + (1 + b_group)·z_true`, so `z′ = (z − a_site)/(1 + b_group)`.
  - **Group A:** b = 182.2 ppm (se 7.0, between-site τ 11.4) in the pilot. **Group B:**
    b = 1.9 ppm (se 14.3).
  - b is pooled from deep, low-rate intervals with the site being corrected left out. a_site
    comes from the top-ring jump.
  - Series are restricted to rings read on both sides of the gap (shallower than about 294 m).
- **Group assignment from evidence that does not depend on the jump being corrected.** The pilot
  (`bridge_pilot.py`: `fam = "A" if dS_obs > 0 else "B"`, verified here) assigned each site by
  the sign of its observed whole-interval change across the gap, which is the outcome the
  correction acts on. Three of its ten group-A sites lie within 1 cm of that threshold (內寮
  +0.4, 元長 +0.9, 秀潭 +1.0 cm); a group-A instrument at a site with about 1 cm more real
  compaction would have been called B and received about 0 instead of about 5 cm of correction.
  The within-group τ was also estimated on sign-selected sites, which makes group A look more
  homogeneous than it is. The revised rule uses four independent lines of evidence:
  - the offset cluster (−15.5..−16.3 cm, +50..+54 cm);
  - the site's own deep-interval scale b_deep (`bridge_real_gap.csv`, verified here): 嘉興
    194 ± 13, 新生 181 ± 8, 溪州 172 ± 20, 土庫 167 ± 37, 秀潭 240 ± 40, 內寮 206 ± 80, 湖南
    106 ± 51, 光復 66 ± 66, 北辰 62 ± 91, 拯民 13 ± 20, 僑義 −9 ± 23, 宏崙 −26 ± 47 ppm; none
    at 元長 and 客厝;
  - pile-implied scales (土庫, 溪州, 秀潭, and 光復 and 拯民 via NTUH);
  - GNSS-implied scales (S3, both families).
  A site is assigned when at least two lines agree. Otherwise it stays unassigned and is carried
  as **both members**. 內寮, 元長, 秀潭 and 僑義 are carried as both members in any case
  (near-threshold or uncorroborated: 僑義's offset of +16.2 cm and 元長's −8.9 cm fit neither
  cluster). Group-membership uncertainty enters sigma as the mixture variance, and τ is
  re-estimated within the evidence-assigned groups.
- **2022 interior path.**
  - **Pile-shaped** where a co-located deep benchmark exists: 土庫 (TKJS 300 m), 溪州 (SJES),
    光復 (via NTUH 300 m, 0.6 km away), and 秀潭 (STES 70 m) for the shallow band.
  - **Elsewhere:** linear interpolation plus the site's seasonal climatology.
  - **sd** = σ_b·L combined with a path term of 1.8 mm (pile-shaped) or 5.3 mm (climatology),
    plus the membership term.
  - Never use plain linear interpolation (10.6 mm), a continuous-rate bridge (1.7-2.5 cm error)
    or an offset-only correction (5.3 cm error).
- **The 2012-03..2014-04 hiatus.** No correction: the evidence is inconclusive (−42 ± 23 ppm
  against +80 ppm).
- **Validation.**

  | Check | Part | Pilot result | Revised test |
  |---|---|---|---|
  | (a) Bridged gap compaction against the pre- and post-gap rate bracket | S1a | 10 of 14 sites inside the bracket | Replaced by a pooled standardized difference between bridged and rate-midpoint gap compaction across sites, with its power to detect a ±25 % scale error stated. A count of sites inside the bracket cannot detect a 20-30 % scale error |
  | (b) Pile-implied scale | S1a | 153-203 ppm at group-A sites; −10 ppm at 拯民 (group B) | Group A within ±25 % of the pooled scale; group B within an absolute ±30 ppm, because a relative tolerance is undefined near zero |
  | (c) GNSS-implied scale | S1b, after S3 | 182-225 ppm group A; 25 ppm at 宏崙 (group B) | Within the ±30-40 ppm excess-rate bracket; NGL stations near ring sites added where they exist (to be checked from coordinates) |
  | (d) InSAR gap subsidence across sites | S1b, after S4 | correlation 0.87 with bridged values, 0.10 with raw (9 sites) | Extended to 14 sites; offset-only InSAR ties (S4) |
  | (e) Synthetic pseudo-gaps, every 13-month window 2015-2021 | S1a | group-A jumps only; worst window 1.35 cm (2017, common bias −1.34 cm) | Mixed A/B injections in the observed proportions, run through the full pipeline including classification. Report the misclassification rate and the error it causes, and the worst window, not the mean |

  **Acceptance:**
  - worst-window RMSE ≤ 1.5 cm in the mixed-group rerun. This threshold was set after seeing the
    pilot's 1.35 cm and is labelled post hoc wherever it is quoted;
  - (a) |pooled standardized difference| < 2, with power ≥ 0.8 for a 25 % scale error;
  - (b) as in the table;
  - residual offset across the gap within ±1 cm at ≥ 12 of 14 sites (pilot: 13 within ±0.9 cm,
    元長 +1.6 cm);
  - **misassignment sensitivity:** groups assigned by offset cluster alone, and the change in
    bridged gap compaction reported for every site, 元長 in particular (no independent check).

  Pile checks are not independent at the sites where piles shape the 2022 path, so the
  independent weight there comes from (c), (d) and (e).
- **Outputs.** `rings_bridged/v1/`:
  - `ring_corrections.csv` (with group evidence and membership probabilities);
  - `rings_corrected_<site>.parquet`, one per member where a site carries two;
  - `ring_total_monthly_bridged.csv` (flag, sigma);
  - band series for `mechanism.band_compaction`.
  The pilot product is in `subsidence_gap_sources/out/product/`.
- **Port after the freeze.** `subsidence.mlcw_compaction` takes the deepest ring by mean. That
  ring is no longer read after the gap, so the series ends in December 2021 at five sites and in
  2024-10 at 元長. `band_compaction` carries the step (−0.4 to −3.9 cm at group-A sites). Fix
  both, coordinated with the ring_holdout session (section 5.5).
- **Effort.** 2 agent-days. **Risks.**
  - A common physical anomaly in the deep intervals during 2022, a post-drought year, would bias
    the pooled scale.
  - The scale may drift in time.
  - 北辰's 2.6-year gap.

### WP S2. Deep benchmarks as a first-class series

- **Feeds** (`ls-wra-dbm-obs__*`), 2016-06 or later to 2025-03, continuous through 2022 with
  94-100 % daily coverage. Mapping by matching printed values:

  | Feed | Benchmark |
  |---|---|
  | TKJS_281 | 土庫 300 m |
  | TKJS_282 | 土庫 130 m |
  | TKJS_283 | 土庫 400 m (zeros in 2025) |
  | NTUH_282 | 中科虎尾 300 m |
  | NTUH_281 | 中科虎尾 70 m |
  | SJES | 溪州 300 m |
  | STES | 秀潭 70 m |
  | WNES | 灣內, off the fan, sign inverted |
  | DLIO | Pingtung, out of domain |

- **Technique.**
  - A step catalogue: a step at every pile on 2024-02-29/03-01, a reset at 溪州 in March 2025,
    and 灣內's sign.
  - Daily → monthly aggregation.
  - Merge with the printed monthly-report values for 2024-01..2026-01 (already transcribed and
    verified three ways). Printed values are primary where they overlap, being agency-screened;
    the feed is the check.
  - Extend monthly as WRA publishes (`twin.report_extract`). The 2023 reports are absent from
    the address.
- **Role.**
  - Validator for twin subsidence 2023-26 (never fitted, never in a gate or choice; U7).
  - Path shape and scale check for S1.
  - A calibration point for S4 and S5.
- **Acceptance.** Printed against feed monthly differences ≤ 2 mm after the step catalogue, and
  a mapping correlation ≥ 0.85 for every mapped pile (pilot 0.877-0.994).
- **Outputs.** `deep_benchmarks/v1/` (daily, monthly, mapping, steps).
- **Effort.** 1 agent-day. **Risk.** The ring/pile rate ratio varies by window (0.60-0.91). Use
  increments, not levels.

### WP S3. GNSS cleaning: WRA and NGL families

- **Data.**
  - **WRA:** 25 stations, daily, 2020-01..2025-03. NCKU stations (STES, TK01-03) cover only
    2023-04..2025-03.
  - **NGL (new family).** The public Nevada Geodetic Laboratory archive lists 11 stations in the
    fan box (lon 120.15-120.65, lat 23.55-24.10) in its `DataHoldings.txt` (fetched by the
    review on 2026-10-12; parsed again here):

    | Station | Span | Note |
    |---|---|---|
    | CHUA | 2007-01..2023-08 | Changhua side, height 109 m |
    | LUKN | 2007-01..2024-04 | Changhua coast |
    | PINT | 2007-01..2024-04 | height 328 m |
    | S103 | 2007-01..2024-04 | height 125 m |
    | ERLN | 2007-01..2019-08 | ends before the ring hiatus |
    | PAKU | 2008-03..2023-05 | height 432 m |
    | FUNY | 2012-08..2024-04 | Changhua side |
    | GUKN | 2012-08..2024-04 | height 192 m |
    | DNAN | 2014-08..2024-04 | Yunlin |
    | HUWE | 2014-08..2024-04 | Yunlin |
    | PUS2 | 2016-01..2024-04 | Changhua side |

    The county notes are inferred from the coordinates and are unverified. Ten span
    2021-12..2023-03, and nine start in 2015 or earlier and run past 2023-03. HUWE has
    3,310 daily solutions, 362 of them in 2022, and a 2015-21 vertical trend of −20.5 mm/yr
    (verified here). LSB0, just north of the box (24.15N), runs 2012-01..2026-09. Daily IGS20
    `tenv3` files download anonymously (about 10 MB for all). This gives continuous GNSS across
    the 2022 ring hiatus, in Changhua, and overlapping leveling. Why most series end in 2024-04,
    and whether the operating agency holds later data, is unverified (probe R-j). The NGL
    citation and terms are recorded in `SOURCE.json` at fetch.
- **Model.** `h(t) = a + b·t + annual + semiannual + Σ steps + e`, per station.
- **Common mode from off-fan stations.** The first draft took the common mode as the median
  residual across on-fan stations (10.1 mm daily sd). Every WRA station is on the fan, so that
  median includes real fan-wide non-seasonal deformation, such as the 2021 drought-year
  acceleration and the post-2022 slowdown seen in the rings. Revised:
  - the common mode is estimated from off-fan or low-compaction reference stations: NGL
    candidates PINT (328 m), PAKU (432 m), S103 (125 m), GUKN (192 m), CHUA (109 m) and LSB0
    (122 m), inferred off-fan from their heights only (unverified); or from a PCA/ICA filter
    whose first mode is checked for spatial uniformity against those stations;
  - it is kept as a separate product, never silently removed;
  - the twin is scored against both raw and common-mode-removed series, with the model's own
    common mode (its mean over the same stations' cells) removed the same way.
- **Step catalogue (WRA):**

  | Station | Step |
  |---|---|
  | TKJS | +13.2 cm on 2022-06-17; −9.5 cm around 2024-01-01 |
  | SJES | +10.0 cm on 2022-03-29, plus an unresolved second offset in 2023 (unusable across the gap) |
  | GFES | −5.6 cm on 2020-07-10 |
  | SNES | transient in 2023 |

  Drop LNJS (fitted rate +53.6 cm/yr, residual sd 1.3 m) and YSLL (14 steps). Flag ANES. The
  cause of each step (antenna or monument) is unverified. NGL series get the same detection.
- **Result so far.** With the on-fan median common mode, the WRA median station residual sd
  falls from 14.3 to 8.4 mm. It is recomputed with the off-fan common mode.
- **Role.** Calibration layer for InSAR (S4, off-fan reference), the S1b check, the validator of
  the twin's GNSS holdout, and (NGL) a Changhua subsidence validator independent of leveling
  (V5). GNSS carries full-column and tectonic motion, so it is compared through the "excess"
  relative to the rings, not as a bridge.
- **Validation and acceptance.**
  - Injected steps of ≥ 2 cm recovered within ±5 mm and ±3 days.
  - The 2021 extra compaction seen in the rings and piles survives the common-mode removal at
    co-located sites (same sign, at least half its amplitude).
  - Stability of the excess rate at co-located rings, reported.
- **Outputs.** `gnss_clean/v1/{wra,ngl}/`, `common_mode.parquet`, `steps.csv`.
- **Effort.** 2 agent-days.

### WP S4. InSAR from LiCSAR and LiCSBAS (needs user decisions D1 and D11)

- **Data (verified by the subsidence scout).** Frame 069A_06694_131402, ascending:
  - covers 85 % of the fan box and all 14 ring sites;
  - 326 epochs, 2015-12-16..2026-01-04;
  - one connected network 2020-01-12..2026-01-04, with 27 epochs in 2022;
  - anonymous HTTPS, and HTTP range reads work, so a fan subset is about 13 % of the bytes.
  Two whole-Taiwan frames cover the fan from late 2022: 069A_06632_162010 (ascending) and
  105D_06628_232108 (descending, with a network break 2024-05-28..09-01).
- **Technique.**
  1. **Subset** about 930 interferograms since 2020 (about 1.4 GB, against 8.6 GB full).
  2. **Environment.** LiCSBAS in a **separate** conda-forge environment with GDAL, not the
     `hydro` env. conda-forge answers from this VM (HTTP 200) and mamba is installed in
     `~/miniforge3` (verified here); creating the environment is untested. Run on CPU with
     `--n_para` inside the CPU budget (5.7) under nice.
  3. **Atmosphere (decision D11).** LiCSBAS03op_ERA5 reads `*.icams.sltd.geo.tif` products from
     the LiCSAR portal when they exist and otherwise runs ICAMS (script header, fetched by the
     review). The one epoch checked held only `geo.mli.tif`, so a correction probably needs
     ICAMS, which likely needs an ERA5/CDS account (unverified; a credential that must never be
     committed), or manual GACOS requests.
  4. **Inversion.** Small-baseline time series, clipped to the fan.
  5. **Reference, without coupling to validators.** Per-epoch offsets and ramps are estimated
     from off-fan GNSS stations (S3) and stable areas only. Ties to on-fan GNSS and to leveling
     in the 2020-01..2021-09 overlap are offset-only: no rate or plane is fitted to data that
     later validate the twin. (A pure offset-only scheme would leave orbital ramps in, so ramps
     come from off-fan references instead.)
  6. **Vertical.** Before 2022-11, vertical-equivalent = LOS/U (U = 0.833; the east component of
     −0.54 is a stated bias). From 2022-11, decompose ascending and descending into vertical and
     east.
  7. **Grids.** Resample across processing batches; the grid spacing differs slightly.
  8. **Seasonal amplitude as a spatial validator (new).** In Yunlin first: correlate the InSAR
     annual-harmonic vertical amplitude with census density at 2, 5 and 10 km. It must beat the
     placebo (density more than 15 km away) and a compressible-thickness proxy from the model's
     layer geometry, under the P3 transfer battery. Only if it passes is it used inside Changhua,
     as a rank validator (never a prior) for the π maps and the P4 posterior.
- **Validation.**

  | Check | Pilot result / role |
  |---|---|
  | GNSS, leaving one station out | 9.6 mm time-series RMS; 8.2 mm on the 14-month change |
  | Leveling 2020-21 benchmark changes | leveling withheld |
  | Rings and piles 2023-25 | after subtracting the GNSS excess |
  | Printed piles after 2025-05 | prospective-style check |
  | Bridged-ring gap correlation | all 14 sites |

  **Acceptance (provisional, fixed at pre-registration):** GNSS leave-one-station-out 14-month
  change RMS ≤ 10 mm; in summer pairs, at least 30 % of each Changhua zone's cells coherent
  (coherence ≥ 0.3), otherwise that zone is reported as not covered. Coherence in paddy summers
  has not been checked (only a winter pair, median about 0.46).
- **Role.** Validator for twin subsidence from 2022 to 2026, fan-wide and including Changhua.
  It would be the first spatial subsidence test in Changhua after 2021. It is never a column
  target and never enters a gate (U7). Leveling up to 2021 stays the target.
- **Outputs.** `insar_069A/v1/`: per-epoch vertical-equivalent displacement on a fan grid with
  sd, reference metadata, coherence masks, seasonal amplitude map and its Yunlin test.
- **Compute.** CPU hours (unverified for the full inversion); 2-10 GB disk.
- **Effort.** 6 agent-days.
- **Risks.**
  - Unwrapping errors.
  - No atmospheric correction in the pilot.
  - A −12.6 mm common bias when referenced to one station, hence network referencing.
  - East-west leakage before 2022-11.
  - OPERA DISP-S1 does not cover Taiwan. ASF HyP3 needs an Earthdata login, and whether it
    processes Taiwan is unconfirmed.

### WP S5. Space-time subsidence fusion and leveling extension

- **Gap.** Leveling is annual and ends in 2021-09. Rings, piles, GNSS and InSAR each see
  different parts of the column and different periods.
- **Technique.** A hierarchical space-time model of vertical displacement u(x, t): a separable
  Matérn field in space and a random walk on velocity in time. Each instrument has its own
  observation operator:

  | Instrument | Observation operator |
  |---|---|
  | Leveling | point elevation changes; σ a few mm plus benchmark instability; stepped benchmarks flagged |
  | InSAR | LOS → vertical; per-epoch reference nuisance from off-fan references |
  | GNSS (WRA, NGL) | point; excess-rate nuisance for deep and tectonic motion; common mode as a separate term |
  | Rings and piles | partial-column compaction; operator = depth-fraction parameter (ring/pile rate ratio 0.6-0.9) |

  Inference by sparse GP or INLA-like Laplace on CPU. The output is an annual field (monthly
  where InSAR supports it), 2012-2026, with sd.
- **Validation.**
  - Leave one instrument family out.
  - Leave benchmarks out (spatial blocks).
  - Temporal blocks.
  - Coverage by the rule of 4.4.
- **Role ledger.** The column stays calibrated on leveling up to 2021, as now. After 2021, where
  InSAR and GNSS are absent, the fused field is a random-walk extrapolation of leveling-era
  velocities, the same velocities the column was fitted to, so validating the column there would
  reward persistence. The fused field after 2021 is therefore scored **only** in cells where the
  post-2021 data reduce the posterior variance by at least 50 % against a leveling-only fit (a
  fraction fixed before any score), and always against a "continue the leveling-era rate"
  baseline. A variant that calibrates on the fused field would have to give up every instrument
  in it as a validator, and is not planned.
- **Leveling extension.** Request the 112年 per-benchmark appendix (2022-23) and later years
  (R-e). Until it arrives, InSAR is the fan-wide extension.
- **Outputs.** `subsidence_fused/v1/` with the variance-reduction mask. **Dependencies.** S1-S4.
  **Effort.** 5 agent-days.
- **Risk.** Partial-column operators trade off against the deep-compaction term; report
  posterior correlations.

---

### WP W1. Monthly canal deliveries (surface water v4)

- **Gap.** Deliveries exist only as annual district totals (Jiji Weir table 2006-2025) and
  crop-season tables that end in 2015 or 2020. The v3 calendar assumes the whole 2020 cut fell
  on crop 2. Two sources contradict that: Jan-May 2020 deliveries were already 57 % (Changhua)
  and 51 % (Yunlin) of the 2016-19 mean. On average, v3 over-delivers in Jan-May by +0.56 and
  +0.82 × 10⁸ m³.
- **Technique.**
  1. **Fetch and parse** the WRA Hydrological Year Book discharge volumes for 2012-2025: a POST
     to the listing API plus one PDF per year, 14 PDFs, 159.9 MB, `pdftotext -layout`.
     - 9 Choushui gauges.
     - Compute monthly values from the daily table, not the printed monthly row (they differ by
       more than 5 % in 23 station-months).
     - QC per year: exclude 玉峰橋 in 2024-25 and 內茅埔 in August 2024. Do not difference
       gauges below and above the weir; 彰雲橋 does not balance with the spill.
  2. **Inflow composite.** I(day) = 寶石橋 + 內茅埔 + 水里. Fill gaps up to 15 days. Swap in a
     scaled donor for any year where a gauge's annual ratio is more than 50 % off its median.
     Rescale each year to the weir's published inflow.
  3. **Diversion.** `D(day) = max(0, min(β·I − R, Σ_d C_d))`, where R is industrial plus public
     supply. β is fitted to the annual district totals (pilot 1.12). District demand
     calendars C_d use crop-1 shares chosen by leave-one-year-out on the Jan-May benchmark
     (Changhua 0.50, Yunlin 0.30, winter weight 0).
  4. **Benchmark.** Proportional first-difference Denton to the annual district totals, keeping
     zero months zero. This is the deterministic v4. Chow-Lin in linear form produced 25
     negative months and is excluded.
  5. **State-space version.** Log monthly delivery = log indicator + AR(1). It is fed, as
     observed aggregates, the annual totals 2006-2025, the Jan-May totals 2015-2021 and the
     crop-season splits, each with its own error. Posterior draws become forcing members.
  6. **2026.** No yearbook appears until about mid-2027. Fit a rainfall-runoff regression from
     CWA CODIS mountain stations to the inflow composite, carrying the indicator's annual
     leave-one-year-out error (RMSE 0.89 / 1.28 × 10⁸ m³). Start a forward archive of the WRA
     real-time discharge and weir feeds now.
  7. **Basis for P4 E4.** Deliveries × crop calendar give the paddy-percolation basis.
  8. **Pre-origin variant (U3).** β, a_chc and a_yl are fitted to the 2012-2022 annual district
     totals only, and the Denton benchmark stops at 2022. The full variant (2012-2025) feeds only
     the recorded-forcing arm (U4).
- **Validation.** The primary held-out test is the Jan-May supply per district for 2015-2021,
  from the WRA 110年度 briefing, slide 19. It is never used to fit β or demand scales; the
  shares are chosen leave-one-year-out.

  | Jan-May RMSE (× 10⁸ m³) | Changhua | Yunlin |
  |---|---|---|
  | Supply-limited + Denton (pilot) | 0.40 | 0.37 |
  | Climatological fraction | 0.66 | 0.56 |
  | v3 | 0.84 | 1.06 |

  With n = 7 per district these RMSEs carry no uncertainty by themselves. The comparison is
  therefore reported with a paired bootstrap interval over years for the RMSE difference and a
  paired sign test on the absolute errors (with 7 years, the smallest two-sided sign-test p is
  0.016, reached only if the method wins in all 7).

  **Acceptance:** beat both v3 and the climatological fraction in both districts, with the
  paired-bootstrap 90 % interval of the RMSE difference excluding zero; if it does not exclude
  zero, v4 ships labelled "not distinguishable from climatology on the benchmark".
  Supporting checks:
  - the monthly head-anomaly partial correlation given rain must not get worse (pilot +0.02 to
    +0.06);
  - crop-season shares against MOA 691, reported but not gated.
  The physical A/B test runs in I1 (rung R3).
- **Outputs.** `surface_water_v4/{preorigin,full}/v1/`: district monthly series (Denton and
  state-space draws), the daily inflow composite, a QC report, and the spatial allocation reused
  from v3.
- **Compute.** CPU minutes after parsing. **Effort.** 2.5 agent-days.
- **Risks.**
  - Unstable gauge ratings.
  - A small validation set (7 Jan-May values per district).
  - The split between districts within a month is model-based.
  - β > 1 hints at Taipower releases entering below the gauges (unverified).
  - Licensing of yearbook extractions.

### WP W2. ET0 extension 2023-26

- **Data.**
  - The calibration cache `results/et/openmeteo_et0_2012_2022.npz`: 61 points, daily, to
    2022-12-31.
  - The forcing experiment fetched the same 61 points for 2022-01..2025-06
    (`data_fetch_api/forcing_experiment/et/`, Open-Meteo, GMT days).
  - Checked here: in their 2022 overlap the two agree to a median daily MAE of 0.048 mm/d (max
    0.063) with r 1.000. The small difference is consistent with GMT against local days.
  - The Open-Meteo archive served daily ET0 at 23.7N 120.4E through 2026-10-05 when queried on
    2026-10-12 (verified here from the review's response file). Recent days come from the
    near-real-time stream and may be revised.
- **Technique.**
  - Fetch 2025-07..2026-09 for the same points from the same API, and re-fetch the last 2-3
    months immediately before freezing `et0/v1`.
  - Adopt one day convention (local days, matching the cache) and convert.
  - Splice with source flags.
  - Optionally compare with station ET0 or pan evaporation from CWA agricultural stations
    (availability unverified).
- **Validation.** Overlap consistency (MAE < 0.1 mm/d); monthly bias against the 2012-2021
  climatology reported.
- **Outputs.** `et0/v1/et0_daily_2012_2026.npz` with per-day source and fetch date.
- **Effort.** 0.5 agent-day. **Note.** ET0 feeds `max(rain − ET0, 0)` recharge, so its effect
  on heads is small (recorded weather added nothing in the forcing experiment).

### WP W3. Ground elevation (DTM)

- **Facts.**
  - The `cwa` deliverable uses `--ground-elev dem` (`stage3_theta.json`: `"ground_elev":
    "dem"`), which is SRTM 1 arc-second. SRTM is a surface model with a February 2000 epoch.
  - Grid values: median 15.0 m, minimum −4.7 m; 24 active cells are below 0 m and 101 below 2 m.
  - Checked here against leveling benchmark elevations at their nearest active-cell centroid
    (within 0.71 km): SRTM minus benchmark has median +1.18 m (2012, n = 618), +1.23 m (2016,
    n = 734) and +1.43 m (2020, n = 795). The IQR is about −0.3..+3.1 m, and 30-33 % differ by
    more than 3 m. Point-against-cell representativeness is included. The growth with time is
    consistent with subsidence after 2000, but that attribution is unverified.
  - `GroundHeight` is exactly 0.0 at 60 of 158 model wells (audit of 2026-09-23, quoted in
    `calibrate_flow._load_ground_elev`).
- **Technique.**
  - **Candidate DTM:** MOI 20 m DTM, 2025 edition (data.gov.tw 176927; page opened 2026-10-12:
    CSV, Government Open Data License v1, production year varies by map sheet, file size not
    shown). Check its size before downloading.
  - Aggregate to the 1 km cells by mean and median, keeping the within-cell sd as a
    representativeness error.
  - Correct for epoch with cumulative subsidence between the sheet year and the model epoch,
    from S5 or leveling. A time-varying z(t) is optional.
  - Write a well-point DTM elevation to replace `GroundHeight` 0.0 (feeds H1b).
- **Validation.** Benchmark elevations, point in cell, with the within-cell sd as error. The
  benchmarks used for the epoch correction are not the ones used to validate: the check runs on
  benchmarks held out in spatial blocks (10 km) from the epoch correction.
  **Acceptance:** |median bias| ≤ 0.3 m on the held-out benchmarks.
- **Impact.** The lift is `z − h_1 + 40 m` (floor 2 m), so a 1.2-1.4 m bias is about 3 % of the
  lift. The direct effect on volumes is small. It matters for coastal cells below 0 m, for the
  stated Changhua/Yunlin static lifts (8.7 against 15.8 m) and for transducer-floor QC. The
  priority is low and the work is cheap.
- **Outputs.** `ground_elevation/v1/`. **Effort.** 1 agent-day.

### WP W4. Rain residual gaps (optional, low priority)

- **Status.** Audit item 9 is closed: rain was rebuilt from CWA CODIS for 2012-2026 (evidence
  doc §11, recalibration in batch 8).
- **Residual gaps.** rf25 (下水埔) before 2018 and rf24 (臺西) before 2015-03-28 match no CWA
  station and are left empty by the rebuild; months with fewer than 80 % of days are masked;
  the gap-fill scout found 1-41 % monthly NaN per gauge. The current recharge field uses
  inverse-distance weighting that down-weights missing gauge-months.
- **Technique.** Compare masked IDW with kriging using an elevation drift (the apex gauges sit
  higher), by leave-one-gauge-out on monthly totals.
- **Acceptance.** Kriging is adopted only if it lowers the leave-one-gauge-out RMSE by at least
  10 % fan-wide without worsening any gauge by more than 20 %. Otherwise masked IDW stays.
- **Priority.** Low: the forcing experiment shows recorded weather barely moves head skill.
- **Outputs.** `rain_check/v1/` (report only, unless kriging is adopted). **Effort.** 0.5
  agent-day.

---

### WP I1. Integration into the twin, recalibration and re-scoring (after the freeze)

**Hooks.** Each is minimal and bit-identical when absent. Paths and sha256 are recorded in run
metadata.

- `inputs.load_twin_inputs(extra_energy=PATH)` and `calibrate_flow --extra-energy PATH`: add
  the Changhua per-class arrays into `E_by_class`.
- `--abstraction-basis PATH`: `theta['log_mu']` and `theta['log_alpha']` with prior penalties
  inside `fit_flow._forward`; `_rollout` unchanged.
- `forward.build_model/run` and `scenario.energy_by_class` read the Changhua classes, so the
  policy levers act on Changhua; scenario runs iterate over the μ table (P4).
- `--electricity-product PATH`: the repaired Yunlin kWh (Chow-Lin, pre-origin variant) and the
  synthetic post-2016 members.
- `surface_water` gets a new versioned key `v4` (monthly district volumes); v1-v3 stay
  unchanged.
- `--heads-product` (pre-origin fill, for initial conditions and flag-2 σ-weighted targets) and
  `--well-set` (recovered wells).
- `--mask-months START:END`: mask a mid-record block from the fit (the fresh 2016-18 temporal
  unit, and N1 tests). The existing `--holdout-months N` already masks the last N months.
- `heads._station_xy` accepts swapped coordinates through the H1 repair table. This changes the
  well set, so it defines a new baseline.
- `--ground-elev dtm`.
- Free per-layer abstraction fractions (fan-wide) and a lagged return flow.
- An observation-operator hook for the H4 sub-grid correction (secondary model only).
- `mlcw_compaction` and `band_compaction` fixes, and ring products (coordinated, section 5.5).
- Secondary QC flags in `challenge.monthly_observations` and the prospective scorer, reported
  as secondary scores only.

**Tests.** CPU-only and synthetic:

- sparse operator equals matvec;
- adjoint equals a forward impulse;
- synthetic recovery of μ and α;
- extra-energy bit-identity when absent;
- loader use guards, including the `fit_window_end` check.

**Recalibration ladder.** Each rung is R0 plus one change, except R4, which is built on R3 if R3
is accepted (so that a recharge-timing error is not absorbed by the layer allocation). Every
rung runs two GPU jobs, 3 at a time under MPS: `fit_full`, and a temporal screen with
`--holdout-months 36` (fit 2012-19, score 2020-22) that yields the existing fair
held-out-years verdict (G3).

| Rung | Change against R0 | Rung-specific test |
|---|---|---|
| R0 | Batch 8 merged `cwa`, rerun with the coordinate fix (new well set) | new baseline for comparability |
| R1 | + DTM ground elevation | — |
| R2 | + Yunlin Chow-Lin electricity (synthetic post-2016 members as an unweighted sensitivity) | placebo-controlled head check (P2a) |
| R3 | + sw v4 monthly deliveries (recharge and seepage terms), pre-origin variant | — |
| R4 | (on R3 if accepted) + free per-layer abstraction fractions and lagged return flow | P2f rerun: model layer-1/2 ratio on local-only slopes inside the observed 90 % interval |
| R5 | + recovered wells (H3), pre-declared | — |
| R6 | + public and industrial deep class (P6), only if R-i data arrive | — |
| R* | Every rung that passed | combined baseline for P4.3 / P4.4 |

**Acceptance per rung (all pre-origin, declared before the first rung runs).**

- **Binding at every rung:**
  - the fair held-out-years verdict of its temporal screen is not worse than R0's (RMSE ratio
    within 1 %, verdict unchanged or better);
  - leveling out of fold (spatial K-fold, buffer ≥ 10 km) is not worse;
  - a CPU held-out-site proxy is not worse: the sparse replica with the rung's θ, the per-well
    datum refitted out of fold, site-grouped folds. It is declared as a proxy, because θ saw
    those wells;
  - the rung-specific test in the table.
- **Binding only at R*:** the full held-out-site k-fold gate (5 folds).
- **Reported, never gating:** the 2023-25 forcing-experiment RMSE by swing quartile, and every
  2023-26 subsidence score. If any of them is ever used to choose a rung, a member or a
  threshold, the model is labelled "selection-contaminated", its frozen 2023-26 score is not
  reported as skill, and it can be tested only in a new registered window (U7).

In-sample fit is never an acceptance criterion. Each accepted rung refits the column on
leveling before its subsidence scores are read.

**Re-scoring protocol.**

0. **One primary model, pre-registered before any 2023-26 re-score:** R* plus the Changhua member
   the P4.3 gate selects (under Outcome C, the member named in advance in the P4.3
   pre-registration), with the H4 operator only if it passed its own pre-origin acceptance.
   Every other model (each rung, M_heads, M_elec, M_post, the synthetic-census members, the
   Z2 members) is secondary. Secondary results are reported with the number of comparisons
   stated, and no secondary model is promoted on its 2023-26 score.
1. The frozen 2023-26 challenge is rerun for the primary model and the secondary models. New
   numbers are reported next to the old ones, never in their place.
2. The registered prospective tests (`prospective_2026-11`, `prospective_2026-11_hybrid`) are
   scored with their registered models and protocol. They are not refitted, re-selected or
   re-registered.
3. A remediated model can be tested prospectively only in a **new** window registered before
   any of its data exist.
4. The forcing experiment (recorded forcing, 2023-25) gains Changhua arms: zero, M_heads,
   M_elec and M_post, each with Changhua forcing from census/TPC only. It is a diagnostic.
5. Head and subsidence intervals include the propagated forcing uncertainty (4.8). Members are
   combined with equal weight, unless a stacking rule on pre-origin scores was declared in the
   pre-registration.
6. Changhua and Yunlin are always reported separately.
7. Model card and evidence doc: Changhua policy numbers carry the μ table and their member range,
   and the Changhua output is labelled "net seasonal stress (zone scale)" unless E4 says
   otherwise.

**GPU.** Ladder 12 jobs; R* 7 jobs (fit, 5 folds, temporal screen); P4.3 12-13 jobs; P4.4 6
jobs; optional 7-13 jobs. Totals and wall time in the single budget table of 5.6. Rule: never
edit twin modules while GPU jobs run (project memory), so code changes land in a window with no
GPU job running, and every rung records the git commit.

**Effort.** 5-6 agent-days of code, plus GPU wall time.

### WP R1. Data requests

Requests are listed in section 6.2. Drafting takes about 1.5 agent-days; existing drafts are in
`docs/UNSENT_FIELD_DATA_REQUESTS.md`. Only the user can send them.

---

## 4. Validation and anti-leakage protocol (all WPs)

### 4.1 Data-role ledger

Each reported number has a row in `results/dataprep/ledger/data_roles.csv` with columns
`number_id, source, role, product_version`, where role is one of fit target, prior,
prior-width input, input, validator or design-time evidence. A source plays one role per number.
The cases that matter:

| Source | If used as | Then it cannot be used to |
|---|---|---|
| TPC Changhua total | prior on the Changhua mean μ | validate μ (its monthly anomalies stay free as a validator, because v(t) comes from the Yunlin census, unless E5 uses them as an instrument) |
| TPC Changhua anomalies | instrument in E5 (if used) | validate interannual amplitudes (V6) for the same number |
| Crop-statistics and land-use totals (P3) | prior-width input to μ | be the level of μ, or validate μ |
| Leveling ≤ 2021 | column target | validate the column, except out of fold (spatial K-fold over Changhua benchmarks with a ≥ 10 km buffer, or a column fitted on Yunlin leveling only) |
| Changhua heads 2012-22 | inversion target | validate Changhua head skill in 2012-22, except by cross-validation across wells and masked years |
| Design-time units (easting terciles, north and south halves, the 2020-22 hold-out, Changhua heads as used by the pilots) | design-time evidence | serve as fresh validation units for a pre-registered test (4.5) |
| WRA Jan-May chart | validator (until the source table arrives) | enter any likelihood |
| Deep piles | 2022 path shape for S1 at 4 sites | serve as an independent check of S1 at those sites |
| Land-use / crop prior | within-zone shape π | validate the within-zone pattern |
| Daily-cycle indices and their seasonal CV | validator (rank check; seasonality evidence in the P4.3 gate) | be a prior for the same number, including a prior on α |
| InSAR seasonal amplitude | rank validator in Changhua, only if its Yunlin test passes | be a prior for π or s |
| Jan-May 2015-21 canal benchmark | W1 held-out test, with shares chosen leave-one-year-out | fit β or demand scales |
| Any observation from 2023-01 on: heads, rings, piles, WRA and NGL GNSS, InSAR | validator of the frozen and recorded-forcing runs | enter a gate, ladder decision, member choice, hyperparameter or threshold (U7) |
| Fused subsidence field after 2021 | validator, only where post-2021 data cut its variance by ≥ 50 % | validate the column where it only extrapolates leveling-era velocities |

### 4.2 Time

- The forecast origin is December 2022.
- Every product used for calibration, initial conditions, priors, zone design or burn-in
  choices of anything scored on 2023-26 is a pre-origin product whose `fit_window_end` is
  2022-12 or earlier (U3, U5). This now covers P1, P2a, P3, W1 and CV_j as well as heads.
  Exemptions (the LUIMAP 2024 snapshot) are listed with a sensitivity run.
- Post-origin forcing data enter only the recorded-forcing arm (U4).
- The 2023-26 heads, the full gap fill and the Feb-Mar 2026 bridge never enter any fit.
- **No observation dated 2023-01 or later (heads, rings, piles, GNSS, InSAR) may enter a gate, a
  member choice, a ladder decision, a hyperparameter or a pre-registered threshold (U7).** A
  model chosen with any of them is labelled "selection-contaminated"; its frozen 2023-26 score
  is not reported as skill, and it can be tested only in a new registered window.

### 4.3 Space

- **Pseudo-Changhua tests in Yunlin.** The census is withheld in the region, the land-use prior
  and temporal basis are built outside it, and in the strict variant θ is refitted without it.
- **Fresh units** (A1): rotated terciles, the two-tercile region, a new 5-region disjoint
  partition (seed 2027) for uncertainty calibration. The overlapping terciles, halves and
  k-means sets give no independent estimate of τ; report the effective n.
- **Representativeness.** A pseudo test transfers to Changhua only if its θ shift spans the
  real-case θ shift (P4.3).
- **Site-grouped well folds.** Nested wells stay together; 10 km clusters with a 5 km buffer for
  heads, ≥ 10 km buffer for leveling.

### 4.4 Artificial-gap tests mirroring the real gaps

| Product | Real gap | Artificial test |
|---|---|---|
| Heads (H2) | Feb-Mar 2026 outage (37 days, partial days on both sides) | The same daily window removed in every year 2013-2025 at the 158 wells |
| Heads | 1-24 month internal gaps | Single-well internal gaps of 1, 2, 3, 6, 12 and 24 months (4 × 72) |
| Heads | Late-starting wells; records ending in 2018 | 66-month leading and 48-month trailing gaps (2 × 40 wells) |
| Heads | Jul-Aug 2023 network thinning | 50 % of wells missing for 12 months |
| Heads | Network-wide months | Whole-network 1- and 2-month outages |
| Heads | 2023-26 pattern | Test-period replica on the 2012-26 matrix |
| Heads (H1) | Datum offsets, steps, barometric loading | Injected paired offsets ≥ 1 m and steps ≥ 3 m; known real events must stay unflagged |
| Rings (S1) | 13-month hiatus with an instrument change, groups unknown | Every 13-month window 2015-2021 masked, mixed A/B jumps injected, classification included; report the worst window and the misclassification rate |
| Ring path | Monthly values inside 2022 | Masked calendar years 2017-2021 with known endpoints |
| GNSS (S3) | Equipment steps | Injected steps ≥ 2 cm |
| Billing (P2a) | Bimonthly pair sums | Monthly meters given their nearest bimonthly meter's phase, then pair-summed |
| July 2025 (P2b) | Missing phase-A bill | Rolling-origin July backtest 2013-2024 |
| Census (P2d) | Frozen after mid-2016 | Census frozen at mid-2013; predict 2014-16 cohorts |
| Extension (P1) | Census ends 2025-06 | Leave one year out (2016-2024), plus whole seasons |
| Backcast (P1) | No county data before 2016 | Matched-lead pseudo-backcast (2020-23 shares predicting 2016-19) |
| Canal (W1) | Monthly values unknown | Jan-May 2015-2021, leave one year out, paired bootstrap |
| Changhua pumping (P4) | No metered data | Pseudo-Changhua regions with the census withheld (frozen and strict), CV-mismatch rescaling and μ-prior emulation (E11) |
| InSAR (S4) | Leveling after 2021 absent | Leveling 2020-21 withheld; GNSS leave one station out |

Each family is re-run on the final product with fixed seeds.

**Coverage rule (one target across the plan: 0.90 nominal).** A product passes a stratum if the
cluster-adjusted binomial interval for its 90 % coverage (A1) contains 0.90. The point window
0.85-0.95 is required in addition only for strata with an effective n of at least 30. Smaller
strata are pooled with their nearest neighbour in gap type or region before testing. With
n = 7 per district (W1), 14 ring sites (S1), 5 disjoint regions (E6) or 74 clustered wells
(V3), sampling error alone exceeds a 0.10-wide window, so the point window would decide pass or
fail by chance. V3 uses the same rule; its former 0.80-0.95 target is withdrawn.

### 4.5 Pre-registration and design selection

- Write a hashed pre-registration (A1) before the first Changhua fit and before every decisive
  experiment (E1-E11, P4.3). It names the **fresh units** each test is scored on, the gate
  scores and weights, and the primary model.
- **Design-time evidence is labelled.** The pilots already fitted Changhua heads 2012-22, the
  three easting terciles, the north and south halves and the 2020-22 hold-out, and chose from
  them the zones Z1-Z3, the 12-month burn-in, the two-class basis, the prior log-sd of 0.5 and
  Design 1 itself. Those units are design-time data: results on them are reported as such and
  never as validation of those choices.
- Keep the number of candidate designs compared on the fresh units to the three declared in P4.
  Head RMSE prefers wrong flexible patterns, and 3-5 regions cannot support many comparisons.
- Select on recovery of pseudo-truth (bias of s, seasonal amplitude, block correlation) and on
  placebo-controlled tests, never on fitted-period head RMSE.
- Frozen-θ pseudo results are labelled optimistic everywhere they appear.

### 4.6 Keeping the 2023-26 test and the registered prospective tests clean

- **2023-26 heads.** Used only for scoring. The test set stays the registered 158 wells and the
  scorer's QC is unchanged; new QC appears only as a secondary score.
- **One primary model** is pre-registered before any re-score (I1); every other model is
  secondary, reported with the number of comparisons, and never promoted on its 2023-26 score.
- **Registered prospective windows (Nov 2026 - Oct 2027).** Untouched. They are scored with the
  registered models and gates (RMSE ratio ≤ 1, ≥ 100 stations with ≥ 6 months, coverage
  0.85-0.95). No product from this plan changes their predictions.
- **New models.** A model built from this plan may claim prospective skill only through a new
  registration made before its window's data exist.

### 4.7 Uncertainty must be calibrated before it is quoted

Every interval this plan publishes must be checked for coverage on held-out data of the same
kind (rule of 4.4). The pilots showed why:

- the Laplace zone sd was 8 % while pseudo-test biases ranged from −71 % to +47 %;
- joint tempered coverage was 0.33-0.5;
- the least-squares conflict scale (0.209 ± 0.017) is both attenuated and uncalibrated.

Further rules:

- τ_H is carried as a posterior (scaled inverse-χ², 4 degrees of freedom on 5 disjoint regions),
  inflated by κ_shift for the Yunlin-to-Changhua shift (adversarial AUC 0.95), and propagated
  into z and its thresholds (P4).
- Parameters at a bound (distal log T, three leakances, the pumping spread) are non-identified:
  profiled, never given a Laplace sd.

An interval without a coverage check is labelled "uncalibrated".

### 4.8 Forcing uncertainty is propagated into the twin's intervals

Each L2/L3 forcing product carries an sd or members: the Chow-Lin split, the Denton and
state-space delivery draws, the Changhua posterior and its μ table, the synthetic census
members, the TPC extension (6.7 % annual error), the head-fill sd (initial conditions), and
the ring sigma on the subsidence side.

- **Heads.** Where the response is linear (the time-invariant operator), the forcing covariance
  is pushed through the exact Green's functions, `Var h = K Σ_q Kᵀ`, on CPU (P4.0). Where it is
  not (RIV, lift feedback), forcing draws are run through the stepwise sparse replica.
- **Subsidence.** Forcing draws are pushed through the column by linearisation around the
  member's head path, or by member runs where the column is strongly non-linear.
- The result is added in quadrature to the model's head and subsidence intervals, and every
  coverage check of a new model includes it. The coverage gates were calibrated for the old
  model's interval and are re-checked with the new one.
- **Member combination** is fixed before results are seen: equal weight, or stacking on
  pre-origin scores only, as declared in the pre-registration.
- **Output.** `forcing_uncertainty/v1/`: the forcing covariances or draws, the propagated head
  variance per well-month and its subsidence counterpart.

---

## 5. Sequencing

### 5.1 Phases

| Phase | When | Content |
|---|---|---|
| **0** | Now, until the freeze lifts (decision D13: no GPU job from any `par*` runner, `hydro-par8` or `hydro-par8b`, is running, and evidence doc section 11 is written) | CPU only, inside the CPU budget (5.7). No edits under `hydrophysics/` or `tests/`. WPs: A0 (first), A1, P1, P2, P3, P4.0 (staging), P4.1, P4.2, P5, P6 (inventory), N1 (table and pre-registration), H1a → H2 → H3, H1b, H4, S1a, S2, S3, S4 (if approved), S1b, W1-W4. Send R1. |
| **1** | Freeze lifted; GPU free | A code window with no GPU job running: port staging code to `hydrophysics/dataprep/` with tests, add the I1 hooks (including `--mask-months`), the coordinate fix and the ring-loader fixes (coordinated). Then rebuild P4.0 on the merged `cwa` θ and rerun E1 and E3, run the recalibration ladder R0-R5 and R*, then P4.3 (decisive GPU tests). S5 fusion on CPU in parallel. |
| **2** | P4.3 gate decided | Pre-register the primary model. P4.4 joint refit and folds; column refit; forward and scenarios over the μ table; re-scoring (I1 protocol); N1 tests; evidence doc and model card. |
| **3** | When requested data arrive | TPC Changhua census (R-a) → P4 switches from inference to calibrating η and lift against metered data. WRA 納管水井 electricity (R-b) → a measured validator, or a replacement. Monthly intakes from 中區 (R-c) → definitive W1 test. WRA 2022 ring record (R-d) → confirms S1. 112年 leveling (R-e) → S5 and the column target. Public-well abstraction (R-i) → P6 deep class and rung R6. |

### 5.2 Workflow fan-outs (Phase 0; inside the CPU budget of 5.7)

1. **Wave 1, day 1:** A0, then A1. Everything else waits for A0's pilot copy.
2. **Wave 2, independent:** {P1, P2a-c, H1a, W1}, then {S2 → S1a, S3, W2, W3}, then {P3 (needs
   P2a's class arrays), P5 (needs H1a), H1b (needs W3)}.
3. **Wave 3:** {H2 → H3}, {P2d-f}, {P4.0}, then {H4 (needs P4.0 and P2a)}, {P6 inventory},
   {N1 table}.
4. **Wave 4:** P4.1. **E5 runs first**, from the scouts' products, as soon as A0 is done; no
   conflict number is quoted before it. E2, E4, E6, E7, E10 and E11 follow. E1 and E3 need
   P4.0. Product v0 (P4.2) needs P1, P2a, P3, W1 and a passed E11.
5. **S4,** once the user approves D1 and D11, runs in its own environment and counts as one or
   two of the four CPU slots. S1b runs after S3 and S4.

### 5.3 Critical path

```
A0 -> A1 -> {P1, P2a, P3, W1, H1a} -> P4.0 -> P4.1 (E5 first; E1-E11) -> P4.2 v0 --+
                                   [freeze lifts, D13]                               |
  I1 hooks + tests + P4.0 rebuild on merged theta (E1, E3 rerun)  <------------------+
     -> ladder R0..R5, fit_full + temporal screen per rung (12 jobs, about 15 h wall)
     -> R*: fit + 5 folds + temporal screen (7 jobs, about 9 h wall)
     -> P4.3 strict pseudo / RIV / stress discrimination (12-13 jobs, about 15-17 h wall)
     -> gate -> pre-register the primary model
     -> P4.4 joint refit + 5 folds (6 jobs, about 8 h wall)
     -> column refit -> forward + scenarios over the mu table -> re-scoring (about 0.5 day)
```

The CPU part of the critical path takes about 2-2.5 weeks of calendar time. The GPU part takes
about 2.2-3.2 days of wall time once it starts, if the MPS queue is kept full (5.6).

### 5.4 What waits for the freeze to lift

- Every edit under `hydrophysics/` and `tests/`.
- Every GPU job.
- The final Green's functions and every P4 number quoted as final. The scouts used batch 8's
  `fit_full` θ while folds 2-4 were still running, so E1 and E3 are provisional until rerun.
- Evidence doc section 11 (results of the rain-rebuild recalibration), which gives the
  baseline that rung R0 starts from.

### 5.5 Coordination with the ring_holdout session

A separate session is fixing the reference in `hydrophysics/twin/ring_holdout.py`. This plan
does **not** touch that file. Agreements to settle before S1 is ported:

1. **Two conventions, both reported.** The ring_holdout fix re-references the raw ring record to
   the first 2023 survey, giving the corrected primary score (audit section 3). S1 adds the
   bridged December 2022 level and the 2022 path as an alternative series, reported side by
   side, never mixed in one score.
2. **The S1 product is data.** It lives in `results/dataprep/rings_bridged/` and enters
   `ring_holdout` only through an explicit option that the ring_holdout session adds or
   approves.
3. **Shared files.** The `subsidence.mlcw_compaction` and `mechanism.band_compaction` fixes touch
   files the ring_holdout session may also edit. One session makes them, as agreed in advance,
   after the freeze.
4. **Old results.** Results that used 2023-25 ring levels together with pre-2022 levels are
   re-run under both conventions. 2023-25 rates and increments are unaffected (a scale of
   1.00018 is negligible for rates).

### 5.6 GPU budget (Turing sm_75, fp32/fp64 only; about 0.76 GiB per job; measured 3.85 h per 500-epoch job under 3-way MPS)

This is the single GPU budget of the plan. Wall time assumes 3 jobs in flight, about 1.28 h of
wall per job.

| Block | Jobs | Job-hours | Wall (3-way MPS) |
|---|---|---|---|
| Ladder R0-R5 (`fit_full` + temporal screen per rung) | 12 | about 46 | about 15 h |
| R* (fit, 5 folds, temporal screen) | 7 | about 27 | about 9 h |
| P4.3: strict pseudo (5), RIV (1-2), stress discrimination with masked blocks (6) | 12-13 | about 46-50 | about 15-17 h |
| P4.4: joint refit + 5 folds | 6 | about 23 | about 8 h |
| **Required subtotal** | **37-38** | **about 142-146** | **about 47-49 h** |
| Optional: alternation and member refits | 6-12 | about 23-46 | about 8-15 h |
| Optional: post-2016 ramp test with θ fitted on 2012-16 (P2d) | 1 | about 4 | fills a slot |
| Optional: rung R6, public and industrial deep class (only if R-i data arrive) | 2 | about 8 | about 3 h |
| Downstream chain per final model (column, posterior, forward, scorers) | — | — | about 3 h each, 2-3 models |
| **Total** | **37-53** | **about 142-204** | **about 2.2-3.2 days if kept full** |

RIV run time is unverified and may be longer. If the budget must shrink, the per-rung temporal
screen is the first thing to drop, in favour of the CPU held-out-site proxy plus leveling; that
choice is declared before the ladder starts and saves 12 jobs (about 46 job-hours).

### 5.7 CPU budget while the GPU batch runs

- At most **4 worker threads in total** across all Phase-0 processes, each single-threaded
  (`OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = 1`), all under `nice -n 10`, on
  the 12-CPU, 62 GB machine.
- S4 (LiCSBAS `--n_para`) and the E1 torch cross-check (measured at 4 threads) count inside this
  budget. The torch cross-check runs alone, or at 2 threads beside one other worker.
- Each GPU job is launch-bound on one CPU core. Watch the GPU jobs' epoch time (baseline
  27.7 s/epoch under 3-way MPS) and pause CPU work if it degrades by more than 10 %.
- A second GPU runner session, `hydro-par8b` (created 2026-10-12 03:07), exists, so the batch
  window may outlast batch 8 alone (decision D13).

---

## 6. Decisions for the user, and data requests

### 6.1 Decisions

| # | Decision | What it involves | Recommendation |
|---|---|---|---|
| D1 | InSAR processing (S4) | About 1.4 GB of interferogram subsets by HTTP range reads from the JASMIN LiCSAR portal, or 8.6 GB in full; a new conda-forge environment with GDAL for LiCSBAS, separate from `hydro`; 2-10 GB of disk; CPU hours inside the CPU budget. conda-forge answers from this VM (HTTP 200) and mamba is installed (verified here); creating the environment is untested. | Approve the subset route: it is the only fan-wide subsidence source for 2022 and after March 2025, and the only spatial test in Changhua after 2021. |
| D2 | NASA Earthdata account (ASF HyP3) | Free account, 8,000 credits a month; whether HyP3 processes Taiwan is unconfirmed. OPERA DISP-S1 does not cover Taiwan. | Not needed now; revisit only if LiCSAR coherence fails. |
| D3 | Hydrological yearbooks (W1) | 14 PDFs, 159.9 MB (already downloaded by the scout, to be archived by A0). Copying is disallowed: keep raw extractions gitignored. Decide whether derived monthly district series may be committed with attribution. | Keep all of it gitignored for now; decide on committing derived aggregates when a paper needs them. |
| D4 | MOI 20 m DTM (W3) | Size not shown on the dataset page. Check it before downloading. | Approve if under about 1 GB. |
| D5 | Sending data requests | Only the user can send them (6.2). | Send R-a, R-b and R-c first: they turn inference into measurement. |
| D6 | Claude API spend | (a) Independent second read of the WRA 2021/2023 briefing charts and tables (E7). (b) Transcription of tables from WRA 彰雲 briefings 2019-2023 and the MOEA 2019 plan annex (also feeds P6 and N1). (c) Continuing monthly-report deep-pile extraction. For scale, the 143 deep-pile values took about 181k input and 61k output units. The API is for extraction and discovery, never for filling physical values (project memory). | Approve (a) and (c); approve (b), now that P6 and N1 depend on it. |
| D7 | GPU allocation | About 142-146 required job-hours in Phases 1-2, up to about 204 with the options (5.6). | Run the ladder, R* and P4.3 before anything else on the GPU: P4.3 decides whether the P4.4 hours are spent. |
| D8 | Reporting conventions | Label Changhua output "net seasonal stress (zone scale)"; publish every Changhua policy number over the μ table, stating that μ is prior-only; publish two members if the gate reaches Outcome C. | Accept. |
| D9 | Adding recovered wells to calibration (H3) | Changes the well set and comparability with earlier runs. | Add as a separate ladder rung (R5); keep the 158-well test set fixed. |
| D10 | Optional dependencies | NUTS through pyro (not in the `hydro` env) for the joint posterior. | Not needed: Laplace with importance reweighting suffices for 6-12 amplitudes, with bound-pinned parameters profiled. |
| D11 | InSAR atmospheric correction (S4) | None; ICAMS, which likely needs an ERA5/CDS account (unverified; a credential that must never be committed); or manual GACOS requests. LiCSBAS03op_ERA5 uses portal ICAMS products when present, and the one epoch checked had none. | Start with none (the pilot reached 8.2 mm against GNSS without it) and request GACOS for a test subset; open a CDS account only if you want ICAMS. |
| D12 | Version control of Phase-0 code | `results/` is gitignored (checked), so staging code there is unversioned, which empties the manifests' `git_commit` field. The freeze covers only `hydrophysics/` and `tests/`. | Allow staging code in a new top-level directory (for example `dataprep_staging/`), outside `hydrophysics/` and `tests/`, committed only after the Rule 1 scan, with no data, coordinates or personal fields. Otherwise keep it unversioned and rely on manifest hashes. |
| D13 | Freeze-lift trigger | The first draft named batch `hydro-par8`; a second runner session, `hydro-par8b`, now exists. | The freeze lifts when no GPU job from any `par*` runner is running and evidence doc section 11 is written. |
| D14 | Public and industrial well request (R-i) | A new request to 台水 districts 5 and 11 and to WRA water rights. | Send it with R-a to R-c. |

### 6.2 Data requests that would replace inference with measurement

| id | To | What | Replaces | Status |
|---|---|---|---|---|
| R-a | 台灣電力公司 | 彰化縣 agricultural-power (農業動力) meter census and monthly kWh 2007-2026, the same product as the Yunlin census; for Yunlin, the per-meter billing period and new connections after 2016 | P4 inference (it becomes calibration of η and lift), the untestable pump-like share a, P2d synthetic layer, P2a phase inference | To draft |
| R-b | 經濟部水利署 | The table and method behind the 2021 briefing chart 灌溉抽水量比較 (Changhua and Yunlin Jan-May irrigation pumping 2011-2021, 依納管水井用電度數及水電比推估), and the Changhua registered-well (納管水井) register by township with coordinates and use (130,414 wells tagged as producing, end of 2018) | The WRA chart as validator only; P3 shape (well density); P4 μ | To draft |
| R-c | 中區水資源分署 集集攔河堰管理中心 | Monthly bank-by-bank intakes (計價水量月報表) and the annual 水源調配及水門操作成果報告 | W1 indicator (definitive validation); N1 delivery-cut tests | Drafted, `UNSENT_FIELD_DATA_REQUESTS.md` §4 |
| R-d | 經濟部水利署 | Any 2022 compaction-ring surveys and the 2022-23 ring re-installation or recalibration record | S1 scale, offset and group model (confirmation) | To draft (audit section 5) |
| R-e | 經濟部水利署 | 112年 彰化與雲林地區地層下陷監測及分析: per-benchmark leveling appendix (2022-23) and later years | S5 fused field after 2021; column target extension | To draft. Related NLSC request drafted, §3. |
| R-f | 彰化縣政府 / 漁業署 | Township aquaculture area by species and year | P3 aquaculture class | To draft |
| R-g | 水利署 / 成功大學 | Water-to-electricity (水電比) calibration and pilot raw data | Calibration of η and lift; avoids circular electricity calibration | Drafted, §1 |
| R-h | (data probe, not a letter) | Post-2018 records of the 18 wells whose 60-minute records stop on 2018-12-31 | H3 extrapolation | Probe in Phase 0 |
| R-i | 台灣自來水公司 (districts 5 and 11) / 經濟部水利署 (water rights) | Monthly per-well abstraction 2007-2026 for public-supply and industrial deep wells, with coordinates, screen depth and closure dates; the list of the 774 public deep wells closed in 2011-18 with coordinates | P6 deep class (rung R6); N1 closure-ramp tests | To draft |
| R-j | (data probe, not a letter) | Later data for the NGL-listed stations from their operating agency (which agency operates them is unverified) | S3 NGL family after 2024-04 | Probe in Phase 0 |

---

## 7. Expected impact on the two scores

The two scores are:

- **Head skill:** the frozen 2023-26 free run against climatology at 158 wells, equal-well RMSE
  ratio 1.229 for the innovation run, plus the recorded-forcing diagnostic.
- **Subsidence/policy evidence:** ring, pile and GNSS continuation; the 2012-21 hindcast
  correlation with leveling; and whether the policy levers act where the subsidence is.

Every new number below will come from the pre-registered primary model (I1); other models are
secondary.

### 7.1 Head skill

**Changhua pumping (P4).** Frozen-θ pilots give Changhua cross-validated anomaly RMSE
1.203 → 1.13-1.16 m (−4 % to −6 %) and amplitude ratio 0.42 → 0.59-0.72. The 2020-22 hold-out
(design-time) gives 1.306 → 1.11-1.14 m. These head scores cannot detect a wrong magnitude
(P4 fact 6). Expect a 5-15 % gain at the 74 Changhua test wells, conditional on the joint refit.

On the 158-well headline it is small. This is my arithmetic from the audit's county figures, not
a pilot:

- Changhua's 74 wells carry about a quarter of the model's summed squared error (74 × 1.280²
  against 84 × 2.113²).
- A 10-25 % cut in Changhua MSE therefore lowers the overall RMSE by about 1-3 %.
- The ratio would move from about 1.23 to about 1.19-1.22.

The pumping fix is for the subsidence and policy question; it does not repair the head
headline.

**The Yunlin high-swing wells** (Q4, 84 % of the excess error). The candidates, with what the
pilots measured:

- Billing repair: partial R² of local energy +0.01 to +0.05 at 1-2 km, before the regional
  placebo; it may shrink once the placebo is applied (P2a).
- Monthly canal deliveries: head-anomaly partial correlation +0.02 to +0.06.
- Layer allocation: the model's layer-1 response is 0.21 of layer 2 against 0.49 observed, both
  to be recomputed on local-only slopes with the `cwa` θ (P2f).
- The 1 km cell's inability to carry local cones: the sub-grid observation operator (H4), the
  only proposal aimed at these wells directly. Untested; it enters only as a secondary model
  unless it passes its own pre-origin acceptance.

The expected effect is small to unknown. The I1 ladder measures it by quartile on pre-origin
scores, and the 2023-25 forcing experiment reports it without gating.

**Gap filling and QC (H1-H3).** Under 0.01 on the ratio (audit), better initial conditions,
barometrically clean daily indices, and 16 more candidate calibration wells.

**Recorded-forcing arm 2023-25** (physics 1.519 against climatology 1.422 m). Changhua sits at
zero pumping in that arm today. Adding Changhua TPC-based forcing and monthly canal deliveries
should improve the Changhua wells modestly. No pilot quantifies this.

**Unchanged.** The statistical layer (ratio 0.90) and the registered prospective scores.

### 7.2 Subsidence and policy

**Changhua policy levers.** Today an irrigation cut of 30 % avoids 0.34 cm in Changhua against
1.34 cm in Yunlin, because Changhua has no pumping. With P4 the Changhua response becomes
non-zero, but it is proportional to μ, which no head, leveling or subsidence data in this plan
constrain. It is published over the μ table (0.5-1.27 × the in-grid electricity anchor), a
factor of about 1.65 per prior sd from μ alone, more if E2 shows that the split between Q and
response matters. This is the largest change the plan makes to the policy evidence, and its
size stays conditional on the electricity prior until R-a or R-b is answered.

**Changhua leveling correlation** (2012-21 hindcast 0.11 against Yunlin's 0.46). No pilot
predicts it. The 0.11 baseline was in-sample, so the success test compares out-of-fold
Changhua correlation with the zero-Changhua baseline recomputed under the same folds (buffer
≥ 10 km), by spatial block bootstrap. It may stay low, because the column is zonal and
Changhua's seasonal stress is weakly constrained.

**Ring continuation test** (S1 with the ring_holdout fix). The audit's approximate re-score with
the first-2023 reference:

| Model, rings 2023-25 | RMSE | Bias | R² |
|---|---|---|---|
| Published reference | 2.54 cm | +1.05 cm | +0.29 |
| First-2023 reference | 1.54 cm | +0.07 cm | −0.13 |

The error and the bias improve, while the cross-site R² goes away, because the step had
inflated the observed spread. Report both columns. Sites carried as two group members (S1)
report both.

**New evidence families.** InSAR 2022-2026 fan-wide (S4), the NGL GNSS family (nine stations
from 2015 or earlier past March 2023, several in Changhua) and the fused field (S5, scored only
where post-2021 data cut its variance) would give the first spatial subsidence tests in
Changhua after 2021, and the first fan-wide checks of the 2022 bridge. Deep piles already reach
January 2026. GNSS is scored raw and with the off-fan common mode removed.

**Natural experiments (N1).** The 2011-18 well closures and the 2021 delivery cut give the first
tests of the policy response against data rather than against the model itself; results are
reported whichever way they fall.

**Not resolved by data remediation.** The VEP column's missing sensitivity of rate to head
level (rings +0.0059 per m against about 0 in the column; evidence doc section 6). Only the
InSAR and ring continuation, the N1 tests and the Tuku trial records can address it.

### 7.3 What this plan cannot deliver from public data

- Cell-level Changhua pumping.
- The time-mean Changhua abstraction μ: no head, leveling or subsidence data constrain it, so it
  rests on the electricity prior, and with it the split between μ and α and every Changhua
  policy number.
- Physical abstraction volumes, as opposed to model units.
- Changhua's own seasonal timing (borrowed from Yunlin).
- Constant-in-time abstraction (aquaculture, livestock).
- Abstraction by unmetered, diesel or electricity-theft pumps, in either county. They would make
  heads see more abstraction than electricity, so they cannot explain the heads accepting less.
- Public-supply and industrial deep-well abstraction, until R-i is answered: a known missing
  deep stress.
- The actual post-2016 Yunlin pumps.
- Leveling at benchmarks after 2021.
- Bank-by-bank canal intakes.
- The 2022 ring surveys themselves.
- Intervention effects that N1 cannot identify (for example closures without locations).
- The local cones at the high-swing wells, if H4 fails: they cannot be recovered with a 1 km
  grid.
- If E10 confirms it, a correctly specified level channel for Changhua.

Each data gap here has a matching request in 6.2.

---

## Appendix A. Pilot evidence index

All paths are relative to `results/dataprep/_pilots/` (gitignored), where WP A0 moved the
pilots from the planning session's temporary scratchpad on 2026-10-12.

| Scout / design | Key scripts | Key outputs |
|---|---|---|
| `tpc_aggregate_electricity` | `tools/tpc_county.py`, `tools/shares.py`, `tools/decompose.py`, `tools/shape_nnls.py`, `tools/ch_product.py` | `compare_annual.csv`, `decompose_annual.csv`, `changhua_pump_electricity_candidates.csv`, `ag_county_backcast_2012_2015.csv` |
| `billing_and_census_gaps` | `s02_billing.py`, `s04_disagg_pilot.py`, `s08b_group_size_nearest.py`, `s11_chowlin_production.py`, `s05_census_zeros.py`, `s07_layers.py` | `billing_phase.json`, `group_size_nearest_phase.json`, `chowlin_production.json`, `census_zeros.json`, `layer_response.json` |
| `head_pumping_signatures` | `scripts/well_features2.py`, `placebo.py`, `cv_shallow.py`, `cross_by_layer.py`, `changhua_distance.py`, `tide_coherence.py` | `out/within_well_summary.csv`, `out/placebo.log`, `out/cv_shallow.csv`, `out/tide_coherence.csv` |
| `physics_inversion_hooks` | `b_operator.py`, `c_greens.py`, `inv_lib.py`, `d_pseudo_changhua.py`, `e_changhua_inversion.py`, `f_temporal_holdout.py`, `g_ident_burnin.py` | `d_pseudo_changhua.json`, `e_changhua_inversion.json`, `f_temporal_holdout.json`, `g_ident_burnin.json` |
| `landuse_crop_priors` | `features.py`, `models.py`, `apply.py`, `township_crops.py`, `combine.py` | `cv_results.csv`, `prior_summary.json`, `prior_cells_ci.csv`, `combine.json` |
| `subsidence_gap_sources` | `scripts/bridge_pilot.py`, `bridge_product.py`, `dbm_vs_ring.py`, `gnss_stepfix.py`, `insar_pilot.py` | `out/bridge_real_gap.csv`, `out/bridge_synthetic.csv`, `out/product/`, `out/insar_gnss_loso.csv` |
| `surface_water_monthly` | `parse_hyb.py`, `annual_check.py`, `pilot2.py`, `head_check.py` | `pilot2_report.json`, `monthly_deliveries_pilot2.csv`, `jiji_janmay_supply_2015_2021.csv` |
| `head_gapfill_pilot` | `lib.py`, `s3_pilot.py`, `s4_daily.py`, `s5c_ensemble.py`, `s6_apply.py`, `s7_excluded.py` | `out/pilot_final_clean.csv`, `out/fill_hist.parquet`, `out/recovery_hist.csv`, `out/offset_episodes_hist.csv` |
| `design_physics_inversion` | `p1_channels.py`, `p2_pseudo.py`, `p3_burnin_stress.py` | `p1_channels.json`, `p2_pseudo.json`, `p3_burnin_stress.json` |
| `design_ml_transfer` | `a1_domain_shift.py`, `a2_robust_transfer.py`, `a5_class_composite.py`, `p3_changhua_ml_level.py` | `a1_results.json`, `a2_scores.csv`, `a5_summary.csv`, `p3_results.json`, `p3_results_v1_gls_datum.json` |
| `design_bayesian_fusion` | `fusion_common.py`, `p5_meta_fusion.py`, `p6_wra_janmay.py`, `p8_combined_variants.py`, `tools/digitize_wra.py` | `p5_meta_fusion.json`, `p6_wra_janmay.json`, `p8_combined_variants.csv` |
| this plan | `remediation_plan/dem_vs_leveling.py` | SRTM against leveling benchmarks (W3). The ET0 overlap check (W2) ran inline. |
| review checks, 2026-10-12 (`../review_cf/`) | `scripts/ngl_filter.py`, `scripts/tenv3_check.py` | `dl_ngl/DataHoldings.txt`, `dl_ngl_huwe/HUWE.tenv3`, `dl_openmeteo/om.json` (surface pressure), `dl_openmeteo/et0.json`, `dl_licsbas_era5/era5.py` (script header) |
| this revision (`../revise_plan/`) | `scripts/ngl_check.py`, `scripts/nnls_window.py`; inline runs | NGL station table and HUWE trend (S3); pre-origin source separation (P1); amplitude-matching scales, φ_in in `p1_channels.py`, datum shrinkage and `bounds_hit` (P4); P3 gate on `a5_scores.csv`/`a4_scores.csv`; observed-only H3 counts; matched-lead backcast; census purposes and horsepower (P6); ring b_deep and the group rule (S1); post-2016 block over-dispersion (P2d) |

Values I re-read from these files while writing this plan:

- TPC annual table;
- the pseudo-Changhua west-region rows;
- the electricity-against-heads scales (0.209 ± 0.017; RMSE 1.81 at scale 1);
- the gap-fill strata;
- the canal leave-one-year-out annual skill;
- the ring synthetic rows;
- the design-3 response variants.

## Appendix B. Inconsistencies between scouts, and how this plan resolves them

| Topic | Values seen | Resolution |
|---|---|---|
| Share of bimonthly billing | Audit first pass 51 %; head-signature scout 92 % of cleaned pumps; billing scout 94 % of meters (63.5 % of energy); the physics scout's risk note still says 51 % | 94 % of meters, 63.5 % of duty-capped energy, 92 % of irrigation energy (audit corrected 2026-10-12). The 51 % note is superseded. |
| Changhua cell count | 954 by the TW_COUNTY label (physics scout, Design 1 zones); 852-1,057 in the brief | 954, recorded with the polygon and the labelling rule |
| Changhua first-light magnitude (frozen θ) | 4.45e5 (10 blocks × 2 shapes); 3.5e5 (single amplitude); 2.83e5-4.42e5 (ML shape); 0.43-0.50 M (heads-only, Design 3) | All are frozen-θ heads estimates in model units (about 26-48 GWh/yr equivalent). None is final before P4.3. |
| Conflict z | −0.46 to −4.3 across design pilots | One definition in s (P4): de-attenuated s_H (E5), σ_E = σ_μ, a τ_H posterior from strict pseudo tests, Student-t thresholds on 4 degrees of freedom; reported only after E5 and the strict tests |
| Excluded wells with ≥ 30 % coverage | Brief 17 Changhua / 30 Yunlin; twin rule 13 / 29 / 10; C20 8 / 29 / 10 | H3 fixes one definition and records it |
| Monthly coverage rule | Twin: any reading in the month; audit and gap-fill pilot: C20 | L1 carries day counts; products state their rule |
| Volume currency | Model link 0.265 kWh/m³; WRA Yunlin volumes 2.58 × model | All anchors in model units (kWh of the Yunlin dedupe-cap census); m³ claims state the currency |
| In-grid fraction φ_in | Assumed 0.90 (Design 3, `p3_structural.py`); about 0.75-0.82 (Design 2 township check); φ_in = 1 in the Design 1 scale fit (`p1_channels.py`, verified here) | Computed in P3 from MOI polygons and farmland, with its uncertainty; applied explicitly in μ_elec,in-grid; every heads scale against the county series divides by φ_in |
| Heads-accepted share of electricity | First draft 0.21 ± 0.02 and "3-10×" | Least-squares scales 0.10-0.21 are attenuated and uncalibrated; amplitude-matched 0.42-0.59; in grid 0.12-0.28 and 0.51-0.79. No z until E5 and the strict tests |
| Preferred allocator | First draft: compB (robust 5 km correlation) | compB fails the P3 gate on AA; LUIMAP RF passes; crop-dasymetric must pass the same battery before it is primary |
| Monthly meters | First draft table: 6,284 ("M0 and most M9") | 6,284 M0 + 265 M9 = 6,549 monthly meters |
| Shallow split, return flow, spread used in P2f | `stage3_datum_gate` θ (6.2 %, 45.2 %, 2.71 km) | `cwa` θ (8.7 %, 41.7 %, 0.5 km at its floor) |
| Recoverable excluded wells | 20 (sd ≤ 0.5 m) or 30 (sd ≤ 1.0 m), counted with fills | 16 on observed months only (Yunlin 13, Changhua 3) |
| Phase-0 effort | First draft "45-55"; its own rows summed to 37.5-40.5 | Recomputed from the revised rows: about 50.5-53.5 agent-days |
| GPU total | First draft 90-120 (I1) and 100-130 (5.6) job-hours | One table (5.6): about 142-146 required, up to about 204 with options |

## Appendix C. Technique-by-gap matrix

A = adopted; T = tested and rejected, with the pilot number; D = deferred, with the reason;
— = not applicable. "Reverse engineering" means inverting a physical or accounting model.

| Gap | Reverse engineering | Signal processing | ML | Deep learning | Advanced interpolation |
|---|---|---|---|---|---|
| Changhua seasonal stress s | A: exact Green's-function inversion (P4) | A as validator: daily-cycle and seasonal-CV indices (P5) | T for the level: pooled head-index models R² < 0; shallow single feature 0.11-0.22 | D: run only if P5 steps 2-3 show local signal beyond placebo (prior expectation none) | A: dasymetric township-to-cell spreading (P3) |
| Changhua within-zone pattern | T: block inversion, block-density correlation −0.47 to −0.93 | A as validator, if its Yunlin test passes: InSAR seasonal amplitude (S4) | A: LUIMAP random forest (passes the P3 gate); T: compB, compA, HGB, NNLS, MLP (gate failures, P3 table) | T: MLP transfer total ratios up to 6.1 | A: crop-dasymetric, if it passes the gate |
| Head-to-electricity extraction (user question) | A: through P4 only | T: month-to-month S1 against local kWh, median r 0.03-0.08; level fingerprint mostly regional (local-only r −0.19) | T: cross-validated R² < 0 pooled | D: as above | — |
| EEMD/HHT + EOF decomposition | — | D: benchmark in P5 step 3 (prior art checked at abstract level only) | — | — | — |
| Census frozen after 2016 | T: heads ramp test, over-dispersed, uncalibrated (P2d) | — | D: no post-2016 truth to train on | — | A: install-rate × age-ramp synthetic layer (P2d) |
| Bimonthly billing | A: staggered A/B deconvolution of the fan profile (P2a) | A: Chow-Lin temporal disaggregation | — | D: not needed; a fixed linear operator validates | T: Denton or pro-rata with local indicators, skill −0.19 to −0.36 |
| Census end, July 2025 | A: TPC × month-of-year ratio, Yunlin first (P1) | — | — | — | A: meter's own 3-year same-month mean (14 % error); T: June × ratio methods (24-52 %) |
| Head gaps | — | A: daily bridge (0.28 m) | A: ridge neighbour regression + softImpute; T: DINEOF (1.44 against 0.99 m on 12-month gaps) | D: BRITS/SAITS untested; optional benchmark (H2 step 6) | A: GP or Kalman for network outages; baselines IDW and space-time kriging; KED/GP initial field only if it beats IDW |
| High-swing wells (local cones) | A (untested): periodic-Theis sub-grid operator (H4) | — | D: no transferable target beyond H4 | D | — |
| Barometric loading | — | A: barometric efficiency by regression deconvolution (H1) | — | — | — |
| Ring hiatus | A: offset + depth-scale instrument model (S1) | A: pile-shaped path (1.8 mm) | — | — | T: linear path (10.6 mm), continuous-rate bridge (1.7-2.5 cm), offset-only (5.3 cm) |
| GNSS | — | A: harmonic + step model, off-fan common mode; D: PCA/ICA filter as the alternative | — | — | — |
| InSAR and leveling after 2021 | A: small-baseline inversion (S4) | A: seasonal-amplitude extraction; atmosphere per D11 | — | D: not needed | A: space-time Matérn fusion (S5) |
| Canal deliveries | A: supply-limited diversion indicator (W1) | A: proportional Denton; T: linear Chow-Lin (25 negative months) | — | — | A: state-space disaggregation with draws |
| Rain residual gaps | — | — | — | — | D: kriging with elevation drift against masked IDW (W4, low priority) |
| Ground elevation | — | — | — | — | A: DTM aggregation with representativeness sd (W3) |
| WRA chart magnitude | A as validator: back-calculated Changhua well electricity from the 水電比 (E7) | — | — | — | — |
| Policy response | A: natural-experiment tests through the twin (N1) | — | — | — | — |

## Review log

Revision of 2026-10-12, after two independent reviews. Every change below was made in this
file; nothing under `hydrophysics/` or `tests/` was touched. Numbers marked "verified here" were
recomputed during the revision (Appendix A, last two rows).

**Selection and leakage (critical).**

- Ladder rungs are no longer accepted on the 2023-25 forcing-experiment RMSE. Rung gates are
  pre-origin only: a temporal screen per rung (`--holdout-months 36`, the existing fair
  held-out-years verdict), leveling out of fold, and a declared CPU held-out-site proxy; full
  5-fold only at R*. The budget is in 5.6 (I1).
- New rule U7 and section 4.2: no 2023+ observation enters a gate, ladder decision, member
  choice, hyperparameter or threshold; anything chosen with one is "selection-contaminated".
- The P4.3 gate scores are named and pre-origin; post-origin piles, rings, GNSS and InSAR are
  reported only (P4.3, 4.1).
- One primary model is pre-registered before any 2023-26 re-score; all others are secondary
  and never promoted (I1, 4.6).

**Changhua identification (critical).**

- The gate is restated in s = α·μ·R; μ is declared prior-only (datum shrinkage 0.96-0.99 and
  no level sensitivity in the column). Outcome B now keeps "different seasonality" open and
  blames the electricity route only after independent seasonality evidence (P4.3).
- Changhua policy numbers are published over a μ table (P4, 7.2, D8).
- The negative mean-channel result is a misspecification flag with its own experiment (E10).
- One datum convention for the CPU engine and `fit_flow` (P4 formulation, P4.0, P4.4).

**Accuracy claims (critical and major).**

- The summary and P4 accuracy rows now quote the adopted design's measured recovery
  (0.94-3.26 × truth; 0.43-0.89 of a prior error passes through) and the full bias range
  (−71 % to +47 %); the Fisher sd is labelled uncalibrated (section 1, P4 facts 9-10, P4
  accuracy table).
- The CV-mismatch and μ-prior emulation tests became a gating experiment (E11, V1), including a
  prior built the Changhua way (E6).
- The conflict is restated: least-squares scales are attenuated; amplitude-matched scales and
  φ_in are applied; "3-10×" and "0.21 ± 0.02" are removed from the summary; E5 runs first and z
  uses a de-attenuated s_H with a τ_H posterior and t₄ thresholds (section 1, P4).
- φ_in is applied explicitly in μ_elec,in-grid, with its uncertainty in the prior (P1, P4).

**Design of the tests (major).**

- Fresh validation units (rotated terciles, a two-tercile region, a new disjoint partition, a
  2016-18 temporal block) are declared before E1-E11 and P4.3; tercile and 2020-22 results are
  labelled design-time (A1, 4.5).
- The strict pseudo test adds the two-tercile region under two Changhua baselines, reports θ
  shifts, and must span the real-case shift; bound-pinned parameters are profiled (P4.0,
  P4.3).
- V2 uses a masked θ refit or is labelled "amplitude-only, θ in-sample"; P4.2 acceptance uses
  three placebos instead of "CV RMSE below zero pumping" (P4.2, V2).
- Coverage is gated by a cluster-adjusted interval with one 0.90 target; the point window
  applies only at effective n ≥ 30 (A1, 4.4).
- Regional placebos were added to the P2a head check and the P2f layer ratio; P2f uses
  local-only slopes and the `cwa` θ; R4 runs after R3 (P2, I1).
- U2 defines σ_obs and limits calibration targets to observed and daily-bridged months; H3
  counts observed months only (16 wells, verified here) (2.5, H2, H3).
- Pre-origin variants for P1 (a = 0.959 on 2016-01..2022-12, verified here), P2a, P3 and W1;
  manifests carry `fit_window_end` and exemptions; the LUIMAP 2024 snapshot is a listed
  exemption with sensitivity runs (2.3, 2.5, P1, P2, P3, W1).
- Ring groups are assigned from evidence independent of the corrected jump, near-threshold
  sites carry both members, synthetic tests inject mixed groups, and the bracket count is
  replaced by a quantitative test; the 1.5 cm threshold is labelled post hoc (S1).
- The post-2016 heads test is labelled uncalibrated (over-dispersed blocks, verified here) and
  no longer weights members (P2d, section 1).
- Forcing uncertainty is propagated into head and subsidence intervals, with member
  combination fixed in advance (4.8, I1, P4.4).
- GNSS common mode comes from off-fan stations and is kept as a separate product; InSAR is
  referenced to off-fan stations with offset-only ties to validators (S3, S4).
- The fused field is scored only where post-2021 data cut its variance; the leveling success
  test uses an out-of-fold baseline under the same folds (S5, V5, 7.2).

**Missing data and techniques (major).**

- New WPs: P6 (public-supply and industrial deep wells, census counts verified here), N1
  (natural experiments), H4 (sub-grid operator for the high-swing wells), W4 (rain residual
  gaps, optional).
- NGL GNSS added as a second family (11 stations, verified here) (S3, S1b, S5, V5).
- Barometric correction added to H1 and P5; pressure availability verified here.
- InSAR seasonal amplitude added as a Changhua rank validator after a Yunlin test (S4, P3).
- P3 states the gate outcome per map: compB fails, LUIMAP RF passes, crop-dasymetric must be
  scored before use (verified here).
- Every WP that lacked one now has a numeric acceptance (H1, H3, S2, S3, S4, P2b-d, A0, A1).

**Smaller items.**

- Z2 is no longer prior-only (Z2-borrow and Z2-mean members); the seasonal term uses a log link;
  the Changhua extension uses a × extended Yunlin with no splice; the backcast is validated at
  matched lead; misspecified synthetics added to P1; flags for Chow-Lin (3) and
  neighbour-corroborated steps (3); split-conformal coverage quoted and stronger H2 baselines;
  paired intervals for W1; held-out benchmarks for W3.
- Dependencies fixed: S1 split into S1a/S1b, H1 split into H1a/H1b, P4.0 on the critical path,
  E1/E3 provisional until rerun on the merged θ (3.0, 5.2-5.4).
- One CPU budget (5.7) and one GPU budget (5.6); effort recomputed; D1 and W2 updated with
  verified facts; decisions D11-D14 and requests R-i, R-j added (section 6).
- Evidence table under the central question; technique-by-gap matrix (Appendix C); unmetered
  pumps, public wells and μ listed as unrecoverable (7.3).

**Review points not adopted as written:**
the attribution of the φ_in = 1 fit to Design 3 (it is Design 1's `p1_channels.py`; Design 3
used 0.90); a withheld region holding 45 % of calibration wells (Yunlin cannot supply it; the
two-tercile region holds 35-40 %); purely offset-only InSAR referencing (ramps are taken from
off-fan references instead); and the claim that no data constrain μ (the electricity prior
does; heads, leveling and subsidence do not).
