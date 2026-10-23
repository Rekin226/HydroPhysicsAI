# Data audit: does data quality explain the low head-skill and subsidence/policy scores?

Written 2026-10-11, read-only (batch 8 was running on the GPU; no twin module was edited).
Scope: every input the twin consumes and every observation it is scored on. Scripts and
intermediate tables are in the session scratchpad (`heads_profile.py`, `artefacts.py`,
`clean_rescore.py`); every number below was computed from the files named beside it.

## Verdict

| dimension | observation quality the cause? | missing/coarse data the cause? |
|---|---|---|
| Head prediction skill | **No.** Removing every instrument artefact moves the model/climatology ratio from 1.220 to 1.222. | **Yes, largely.** Given the actual 2023-25 pumping, the physics beats climatology at 3 of 4 well quartiles; the whole remaining deficit sits at ~40 high-swing Yunlin wells whose drivers the pumping data cannot resolve. |
| Subsidence / policy evidence | **Partly.** The ring record has an instrument discontinuity across a 2022 survey hiatus, and the ring continuation test references across it. | **Yes.** No pumping data at all for Changhua (half the model domain): the policy levers barely act there and the hindcast has no spatial skill there. Leveling ends 2021, rings and GNSS end March 2025, no intervention-response data. |

## 1. Head observations: good enough; not the problem

**Test set** (`data_fetch_api/new_measurements/heads_202210_202610/`, 158 wells, scored
through `results/twin/hybrid_datum_gate_innov/evaluation.csv`, 6,186 well-months).

- Coverage: median 42 of 45 months per well. Network-wide holes: **no observation at any
  well in February-March 2026** (10 Feb to 17 Mar logger outage at 150 wells), a 5-6 day
  gap every year-end, and 129-137 of 158 wells in July-August 2023. Missing months are not
  imputed, so they cost sample size, not bias.
- Artefacts in the raw 10-minute records: multi-hour to multi-day dropouts to a constant
  floor (about -33 or -44 m, the transducer reading its own depth) at 50 wells (98
  events), short spikes at 96 wells, and permanent steps at 10 wells (09180341 +80.5 m on
  2024-08-06; 09100211 -12.4 m on 2024-08-26; 07200121, 07040111, 09080111, 07100122,
  07120121, 07020131, 07040121, 07240223). `challenge.monthly_observations` screens only
  ±1000 m sentinels, so these reach the monthly means.
- Effect on the score, re-scored with daily medians and a 15-day rolling-median dropout
  screen: 30 well-months at 19 wells change by more than 0.25 m; equal-well RMSE model
  1.723 → 1.723 m, climatology 1.412 → 1.411, `stat` 1.284 → 1.283; ratio 1.220 → 1.222.
  Dropping the 10 step wells gives 1.229: they hurt the baselines more than the model.
- Many apparent "steps" of 4-10 m are real: continuous recovery of about 1 m/day after the
  spring pumping season (for example 09200321 and 09190132, 28 May to 4 June 2024, across
  a 3-day logger gap) and typhoon recharge (09070311 +4.9 m, end of July 2024).

**Calibration targets** (`AMP_V2/data/wells`, 2012-2022, the same 158 wells). Built the
twin's way (15×MAD despike, monthly mean) and compared with the cleaned build: 95 of
18,852 well-months differ by more than 0.25 m, 12 by more than 1 m; per-well monthly
climatology shifts by median 0.02 m (90th percentile 0.08 m). 21 wells carry a permanent
step somewhere in 2012-22. The twin keeps any month with at least one reading (330
months here have fewer than 20 clean days). None of this is large enough to move the
calibration measurably.

**Where the error actually is.** Per-well excess (model MSE minus climatology MSE):

- Spearman correlation with the well's observed variability: **+0.75** (p < 1e-29); with
  historical coverage −0.03, longest gap −0.04, test coverage +0.01 (none significant).
- The top quartile of wells by observed standard deviation (median 3.1 m; 36 of 40 in
  Yunlin, 29 in layer 2) carries **84 %** of the total excess. Model sd / observed sd is
  0.29 at the median.
- By county (frozen challenge): Changhua 1.280 vs climatology 1.207 m (ratio 1.06);
  Yunlin 2.113 vs 1.593 m (ratio 1.33).
- `GroundHeight = 0` wells (the known missing-value code) score no worse than the others.

## 2. Forcing data: this is where head skill is lost

**The frozen test gives the model no forcing information.** The free run is driven by its
calibration climatology after the December 2022 origin. The forcing experiment
(`results/twin/forcing_experiment/predictions.npz`, 156 wells, 2023-01 to 2025-06)
supplies the recorded inputs:

| equal-well RMSE (m) | all | Q1 (low swing) | Q2 | Q3 | Q4 (high swing) |
|---|---|---|---|---|---|
| physics, climatology forcing | 1.772 | 0.97 | 1.30 | 1.74 | 3.10 |
| physics, recorded pumping | 1.519 | **0.83** | **1.10** | **1.46** | 2.70 |
| climatology baseline | 1.422 | 0.86 | 1.17 | 1.49 | **2.18** |

With recorded pumping the excess falls from 252 to 117 m² summed over wells, and **108 %
of what remains is in Q4** (the other quartiles are net better than climatology). Pooled,
recorded pumping closes about half of the gap to climatology (2.067 → 1.845 vs 1.630 m).
Recorded weather adds nothing (1.848). The model's amplitude stays at 0.36 of observed in
every arm, so whatever drives the big seasonal swings is not in the electricity record.

**Pump census and electricity** (`AMP_V2/data/tpc_pumps.parquet`, `pump_kwh_all.parquet`):

1. **Yunlin only.** All 116,769 census pumps are in 雲林縣 (the fresh station list fetched
   2026-09-30 is the same). 1,057 of the 2,148 active cells (49 %) have no pump; Changhua
   holds 74 of the 158 test wells and has zero modelled abstraction. The 2026-08-22 design
   spec recorded this ("the pump census is Yunlin-only"); it has dropped out of the
   evidence doc and model card.
2. **Frozen at mid-2016.** Install dates run 1,100-1,600 per year from 2005 to 2015 (5,727
   in 2013), 215 in January-June 2016, then about 10 per year. Pumps connected since mid-2016
   are absent; at the earlier rate that is roughly 10,000 pumps (8-10 % by count) by
   2023-25. This is an estimate, not a count.
3. **Mostly bimonthly** (corrected 2026-10-12). The first pass read only code M2 as
   bimonthly (59,595 meters, 51 %). The billing scout of 2026-10-12 showed that M1 and M2
   are the two TPC meter-reading groups, both bimonthly and one month apart (M2 pairs
   Jan-Feb, M1 pairs Feb-Mar). Only M0 and most M9 meters are monthly. That makes about
   94 % of meters bimonthly, carrying 63.5 % of the duty-capped 2012-22 energy and 92 % of
   irrigation energy. Each bill is split evenly over its two months, and reading groups
   cluster in space (68 % of bimonthly energy is in 1 km cells that are at least 90 % one
   phase), so the boxcar does not average out at cell scale.
4. **Registered meters only, no screen depth.** Unmetered or diesel wells are invisible,
   and abstraction is split among aquifers by inference.
5. July 2025 is a partial month (59 % zeros; the M2 group is 99.8 % zero because its
   July-August bill is missing).
6. (Added 2026-10-12.) 22 stations in `fan_stations.parquet` store `LocationByTWD97` as
   "y x" instead of "x y". `heads._station_xy` rejects them as out of range, so the twin
   silently drops those wells. The forcing experiment excluded it (window
   end-exclusive); any later use must too.

The high-swing wells sit in pumping clusters (median 157-241 census pumps within 1 km for
Q3-Q4 against 0 for Q1-Q2) on a 1 km grid, so part of their swing is a local cone that a
cell-mean head cannot carry. Monthly canal deliveries, which track the Yunlin annual head
anomaly (partial r 0.76, evidence doc section 8), exist only as annual totals.

Rainfall was zero-filled and is already rebuilt (evidence doc section 11; recalibration in
progress); the forcing experiment shows weather barely moves head skill, so that rebuild
should not be expected to move this score.

## 3. Subsidence observations

**Compaction rings** (`chou-shui-data/data/ls_cache/clean/ls-wra-mlcw-obs__*`; 14 fan
sites, about monthly from 2010, to March 2025).

- **No survey at any site in 2022.** The record jumps from December 2021 to February 2023.
- **Instrument discontinuity across the gap.** Every ring at a site moves by a common offset
  (新生 −15.7 cm, 土庫 −15.5 cm, 宏崙 +50.6 cm: a change of depth reference), plus a
  depth-proportional term that changes the total monitored interval by −3.9 to +5.0 cm
  (溪州 shallow ring −3.5, deep ring +0.4; consistent with a probe or tape scale change).
  Ring counts also change (宏崙 29 → 24, 溪州 27 → 25). Seven sites show 2-4 cm of net
  *expansion* over the 14 months, which is not physical for sites compacting 1-6 cm/yr.
- **The ring continuation test references across it.** `ring_holdout.window_stats` takes
  the last value on or before 2022-12-31 as the reference, which for rings is December
  2021; the observed "cumulative since December 2022" therefore spans 3.25 years and
  includes the step, while the model and the continued-rate baseline span 2.25 years.
  Approximate re-score from `results/twin/ring_holdout/rings_sites.csv`, observed
  cumulative from each site's first 2023 survey and model cumulative scaled to the same
  span (13 sites):

  | cumulative 2023-25 | published reference | first-2023 reference |
  |---|---|---|
  | model (climatology forcing) RMSE / bias / R² | 2.54 / +1.05 / +0.29 cm | **1.54 / +0.07 / −0.13** |
  | continue prior rate | 3.72 / +2.77 / −0.52 | 2.18 / +1.79 / −1.27 |
  | no change | 3.63 / −2.02 / −0.45 | 3.33 / −3.00 / −4.31 |

  The model's absolute error drops 40 % and its bias vanishes, and it still beats both
  baselines; but the cross-site R² goes away, because the step had been inflating the
  observed spread (observed 2023-25 compaction is a fairly uniform 1.3-5.1 cm). The rate
  scores (OLS inside the window) and the 2012-22 mechanism fits are not affected.
  (The published table quotes R² +0.23; this CSV recompute gives +0.29 on the same
  reference, so treat the right-hand column as approximate until `ring_holdout` is rerun.)

**GNSS** (25 WRA stations): January 2020 to March 2025 only, 79-97 % daily completeness;
ANES, LNJS, SNES and YSLL have 68-113 day-to-day jumps above 3 cm. GNSS carries full-column
and tectonic motion the column does not model.

**Leveling** (`ls-wra-lsp-obs__choushui_panel.parquet`, the column's calibration target):
1,239 benchmarks, 878 used for 2012-22 at the 50 cm/yr screen; 187 benchmarks are marked
lost or destroyed; about 1 % of used sites carry a suspicious step. Quality is acceptable,
but it ends in September 2021 (687 sites last surveyed in 2021) and is annual, which is why
it cannot identify a level-dependent rate (evidence doc section 6).

**Deep benchmarks**: six piles printed in the WRA monthly reports, four on the grid, January
2024 to January 2026, already transcribed and verified (evidence doc section 10). The only
subsidence numbers after May 2025.

## 4. The Changhua gap and the policy question

With no pumping input north of the Yunlin census, Changhua's subsidence comes from the
zonal column fitted to heads that contain no local abstraction:

- 2012-21 hindcast rate vs leveling at the same cells: Yunlin 468 sites, correlation 0.46,
  RMSE 1.67 cm/yr; **Changhua 313 sites, correlation 0.11**, RMSE 1.17 cm/yr.
- Avoided subsidence by 2032 (`results/twin_forward/datum_gate_innov.npz`): irrigation −30 %
  avoids 1.34 cm in Yunlin cells but 0.34 cm in Changhua; aquaculture retired 1.97 vs 0.29
  cm, while baseline forward subsidence is the same in both (3.0 cm). The scenarios cannot
  cut pumping the model does not have.

## 5. What would move the scores

In order of expected effect:

1. **Changhua pumping** (TPC agricultural-pump electricity for 彰化縣, the same product as the
   Yunlin census). Biggest structural gap for the policy question; the vendor dataset does
   not contain it.
2. **A post-2016 census** for Yunlin (new connections, meter changes) and the billing period
   per meter.
3. **Monthly canal deliveries** (request 4, already drafted).
4. **Rings**: fix the `ring_holdout` reference (first post-gap survey, or bridge 2022
   explicitly) once batch 8 finishes, and ask the WRA for any 2022 ring surveys and the
   2022-23 re-installation or recalibration record.
5. **Test QC**: add a dropout screen and step flag to `challenge.monthly_observations` and
   the prospective scorer. This will not change the current numbers, but it stops a future
   +80 m step from reaching a registered score.

Not worth doing for skill: further cleaning of the head records or recalibrating on the
cleaned targets. The measured effect is below 0.01 in the ratio.
