# Twin evidence update, October 2026

Written 2026-10-11 from the audit of the same day. Every number below is read from a
result file named beside it; nothing here is a projection of what a fix might give.
Sections marked *pending* are filled in as their runs finish.

## 1. The corrected observation update does not change the subsidence numbers

The published projection (`results/twin_forward/datum_gate.npz`, built 2026-09-27)
predates the innovation update and restarted its displayed heads with the superseded
field update. Rerun with `--update-method innovation` and otherwise identical flags
(`results/twin_forward/datum_gate_innov.npz`, response basis
`response_basis_innov.npz`):

| quantity | field update (published) | innovation update |
|---|---|---|
| fan-mean forward subsidence 2023-32, baseline | 3.03 cm | 3.03 cm |
| avoided by 2032, irrigation −30 % | 0.94 cm | 0.94 cm |
| avoided by 2032, aquaculture retired | 1.30 cm | 1.30 cm |
| frozen-forecast head RMSE 2023-26, 158 wells (`new_data_challenge*/report.json`) | 2.072 m (ratio 1.243 vs climatology) | 2.048 m (ratio 1.229) |

The subsidence is identical because the column is driven by the free-running heads
(`--column-heads free`), which no observation update touches; only the displayed heads
and their 2023-26 score move. The page is rebuilt on the innovation run so that what it
displays is what the corrected method gives.

## 2. The page's uncertainty bands were too narrow

`twin.coverage` scores how often 2023-26 observations fall inside the run's intervals
(`results/twin/coverage_datum_gate_innov/report.json`, 158 wells, 6,186 well-months,
datum offsets and spreads from fitted months only):

| interval family | nominal 90 % covers | inflation needed | mean 90 % width |
|---|---|---|---|
| ensemble spread (what the page showed) | 74.7 % | 1.86 | 4.8 m |
| well's fitted-period residual sd | 80.2 % | 1.27 | 4.8 m |
| **combined, in quadrature** | **92.6 %** | 0.86 | 7.1 m |
| climatology with its residual sd | 84.5 % | 1.16 | 4.5 m |

The combined interval is the one registered in the prospective protocol. The page keeps
the run spread as its band but labels it as spread, not probability, and prints this
measured coverage beside it.

## 3. A prospective test is registered

`results/twin/prospective_2026-11/` (manifest, protocol, predictions; SHA-256 in the
manifest, registered 2026-10-11): the innovation run's datum-adjusted heads at 158 wells
for November 2026 to October 2027, the climatology baseline, and the combined 90 %
interval. Gate: equal-well RMSE ratio ≤ 1 against climatology, ≥ 100 stations with ≥ 6
months, 90 % interval coverage within 0.85-0.95. Scored with
`python -m hydrophysics.twin.prospective score` once the window closes. No other score
in this repository is prospective.

## 4. The damped head swings are least-squares shrinkage, not a timing defect

`results/twin/diagnostics/pumping_lag.json` (89 wells with usable local electricity
variance): observed and modelled head anomalies respond to the 3 km local electricity
anomaly at the same lag (0 months) with near-identical lag profiles. In sample the
model's anomaly amplitude is 0.73 of observed at correlation 0.65; the MSE-optimal
amplitude for that correlation is about 0.89 of observed, so the fit sits near its
optimum and a larger amplitude would raise the error. Storage variants (×0.25 to ×32,
`update_experiment/dynamics_extended.json`) move the 2023-25 free run by < 0.02 m. The
amplitude can only be recovered by forcing information the model does not have (canal
deliveries, metered abstraction, gauge rainfall) or by a statistical layer on top of the
physics; another calibration sweep is not the next step.

## 5. First out-of-sample subsidence test: 2023-25 rings and GNSS

`twin.ring_holdout` drives the frozen flow model (18 parameter sets) and the deliverable's
column from the free-running record end state through January 2023 to March 2025 under
three forcings and scores each site's cumulative displacement (from December 2022) and
OLS rate against the 14 compaction rings (13 with enough months) and the 19 on-grid WRA
GNSS stations (`results/twin/ring_holdout/report.json`, `rings_sites.csv`,
`gnss_sites.csv`). It is not a clean holdout (the report lists six reasons); it is the
only subsidence observation past the forecast origin.

| 2023-01 to 2025-03 | rings, cumulative (cm) R² / RMSE / bias | rings, rate (cm/yr) R² / RMSE / bias | GNSS, rate (cm/yr) R² / RMSE / bias |
|---|---|---|---|
| model, climatology forcing (the page's projection) | **+0.23 / 2.65 / +1.29** | −2.06 / 1.02 / −0.35 | −1.51 / 2.18 / −1.85 |
| model, recorded pumping | +0.24 / 2.63 / +0.80 | −5.64 / 1.50 / −0.77 | −3.40 / 2.89 / −2.56 |
| model, recorded pumping and weather | +0.23 / 2.65 / +0.86 | −5.26 / 1.45 / −0.68 | −3.19 / 2.82 / −2.49 |
| continue the site's own prior rate | −0.79 / 4.04 / +3.15 | −3.05 / 1.17 / +0.97 | −1.33 / 2.10 / +1.52 |
| no change | −0.45 / 3.63 / −2.02 | −5.23 / 1.45 / −1.33 | −2.56 / 2.59 / −2.20 |
| member p10-p90 covers the observed site value | 15 % (climatology) | 31 % | 5 % |

Reading: the rings slowed sharply after 2022 (site rates 1.7-4.0 cm/yr in 2017-22 fell to
0.6-2.0), so continuing the prior rate overshoots and no-change undershoots; the column's
cumulative displacement sits between them with weak but real skill. The spatial pattern
of rates is not captured by the model or by either baseline. Two signatures matter for
the policy question: the column under-predicts the ongoing rate, and more so when the
recorded (lower) pumping is used, because recovering heads set no new minima and the
VEP column then only creeps; the rings kept compacting. That is the same deficit
section 6 finds. The member spread is far too narrow to be read as a range. GNSS rates
carry full-column and tectonic motion the model does not represent; the model does not
beat the continued-rate baseline there.

## 6. Policy mechanism: the aquitard column does not resolve it (negative result)

The page's policy-timing statement (a pumping cut gives a one-time rebound and does not
slow the ongoing sinking) rests on the VEP column. The 14 compaction rings show a
sustained next-year rate sensitivity to head level of +0.0059 per m (95 % CI
0.0048-0.0071, `results/twin/mechanism/mechanism.json`); the VEP column's own hindcast
gives −0.0005 (CI −0.0013..0.0002). The hypothesis tested here: a zonal column with the
delayed-drainage term (`calibrate_coupled --creep aquitard`), fitted on leveling exactly
as the deliverable's column, keeps the leveling skill and carries that sensitivity.

| column (leveling target, 11-yr tau ceiling) | leveling R² out of fold | rings R² | lev1 on its hindcast (CI) | +1 m step in the projection: sustained rate change |
|---|---|---|---|---|
| VEP (deliverable) | 0.652 | 0.009 | −0.0005 (−0.0013..0.0002) | 0 |
| VEP + aquitard term | 0.656 | 0.003 | −0.0005 (−0.0013..0.0003) | −5e-6 m/yr |
| observed rings | | | **+0.0059 (0.0048..0.0071)** | |

Sources: `results/twin_runs/stage3_datum_gate/coupled_leveling{,_aquitard}/vep_zonal_leveling.json`,
`results/twin/mechanism_aquitard/mechanism.json` (`--recompute-column`). The leveling
fit pushed the new term to its bounds in every zone (mid zone: time constant at the
100-year ceiling with coefficient 0.001; distal zone: at the 1-year floor, acting as
extra elastic storage). The leveling network does not identify a level-dependent rate,
so a column fitted to it cannot carry one, whatever its rheology.

Consequence for the page: the avoided-subsidence number by 2032 stands (both columns
give it), but the statement that the benefit arrives within months and the sinking
rate is unchanged is a property of a column the ring record contradicts. The page
now carries that as an unresolved question beside the number rather than as a
finding. Resolving it needs a target that sees the rate response: the 2023-25 ring
and GNSS continuation (section 5) and the Tuku trial records requested in
[FIELD_DATA_ACQUISITION.md](FIELD_DATA_ACQUISITION.md). Fitting the column on the
rings themselves would make the judge part of the fit and was not done.

## 7. A statistical layer beats climatology where the physics alone does not

Section 1 scores the frozen free run against climatology and loses (ratio 1.23). After
the origin the run knows nothing but its own state, so the only honest forecast
ingredients are each well's climatology, the anomaly observed at the origin with the
rate at which anomalies fade, and the physics' evolution of its updated state.
`twin.hybrid score` builds four pre-specified forecasts from the fitted months only
(`results/twin/hybrid_datum_gate_innov/report.json`, same 158 wells and 6,186 well-months
as the challenge; `rho` is each well's lag-1 anomaly coefficient, median 0.86, nothing
tuned on the test):

| 2023-26, equal-well RMSE (m) | all | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| climatology (`clim`) | 1.667 | 1.804 | 1.620 | 1.515 | 1.273 |
| `clim` + decayed last anomaly (`stat`) | **1.507** (ratio 0.90) | **1.432** | 1.516 | 1.478 | 1.266 |
| `clim` + the physics' own anomaly (`phys`) | 1.669 (1.00) | 1.799 | 1.643 | 1.491 | 1.269 |
| `clim` + half of each (`mix`) | 1.569 (0.94) | 1.587 | 1.566 | 1.473 | 1.257 |
| the frozen free run at datum (section 1) | 2.048 (1.23) | 2.223 | 2.136 | 1.798 | 1.594 |

Reading: anomaly persistence carries real forecast skill for about a year after the
origin (21 % lower error in 2023) and converges to climatology afterwards, as it must.
The physics' anomaly adds nothing on top of climatology at any lead, so the physics'
value for the heads is the scenario *difference* it computes, not the absolute level. An
in-sample regression of the two anomalies on the 2023-26 observations (reported in the
file as a diagnostic, not a forecast) puts the weight on the persisted anomaly at 1.8 and
on the physics' at 0.3: anomalies fade more slowly than a lag-1 model says, which a later
registration may use, but this window cannot select it. With each well's fitted anomaly
spread as the interval, the `stat` 90 % band covered 87.7 % of the observations.

The `stat` forecast is registered prospectively for November 2026 to October 2027
(`results/twin/prospective_2026-11_hybrid/`: 147 wells, 1,764 rows, anomalies persisted
from each well's last observed month, 133 of them September 2026; `model_sha256` is the
hash of its parameter file, no physics enters the point forecast). It is scored with the
same `twin.prospective score` command as section 3's registration, which stays in force
for the physics. Neither registration may be refitted or re-registered for its window.

## 8. A forcing the model lacks: surface-water deliveries track the Yunlin heads

Section 4 named canal deliveries first among the forcings that could carry the missing
anomaly variance. The Jiji Weir's annual operating table (WRA, `水源運用 > 歷年營運水量統計表`,
fetched 2026-10-11, 2006-2025, kept as `chou-shui-data/data/water/jiji_weir_annual_supply.csv`
with its source note) gives the surface water delivered to the Changhua and Yunlin
irrigation districts and the weir's inflow. `twin.forcing_diag`
(`results/twin/diagnostics/jiji_deliveries.json`) correlates each county's annual mean
head anomaly (observed minus the well's climatology) with its district's deliveries, with
the inflow, and with the deliveries once the inflow is regressed out (14 years, 2012-2025):

| annual mean head anomaly | wells | r with district deliveries | r with weir inflow | partial r, deliveries given inflow (p) |
|---|---|---|---|---|
| Changhua, layer 1 | 18 | 0.67 | 0.70 | 0.27 (0.35) |
| Changhua, layer 2 | 37 | 0.62 | 0.57 | 0.34 (0.24) |
| Yunlin, layer 1 | 19 | **0.88** | 0.78 | **0.76 (0.002)** |
| Yunlin, layer 2 | 42 | **0.84** | 0.69 | **0.70 (0.005)** |

Reading: in Yunlin the surface water delivered explains the year's head anomaly beyond
what the basin's wetness explains (a dry year with normal deliveries and a dry year with
cut deliveries, as 2021 and 2023, give different heads); in Changhua it does not separate
from the inflow. Fourteen annual values are a screen, not a forcing, and deliveries are
themselves decided from availability, so no causal weight is claimed. What it establishes
is that the Yunlin district's delivery record is the next forcing to acquire at monthly
resolution and to put into the recharge and pumping-substitution terms; the model has
never seen it. The same probe found the WRA open-data feeds for the weir (daily operations,
code 20503, and the hourly reservoir status) serving the latest day only, and the leveling
"benchmark points" dataset to be a KML of locations without elevations; the monthly intake
history and any 2022+ leveling results therefore have to come from the WRA's hydrological
yearbooks or its monitoring systems, which is what the data scout (section 9) looked for.

## 9. What public data exists for the two gaps: the data scout's verified leads

`twin.data_scout` (Claude API, `claude-opus-5-5` with server-side web search and fetch;
16 searches, 25 fetches, 1.93 M input tokens; `results/twin/data_scout/2026-10-11/`)
was asked for *downloadable* sources for five targets and told to quote the page that
proves each lead. Every lead was then probed by HTTP from here. The model's fetcher was
refused by most `gov.tw` open-data pages, so many leads rest on search snippets and are
marked unverified in `leads.json`; nothing from any lead has been read into the model.
What survives:

| gap | lead (verified = page opened and quoted) | what it gives | how to get it |
|---|---|---|---|
| subsidence past 2021 | WRA 地層下陷監測資訊整合服務系統, monthly 地層下陷監測整合分析月報 PDFs (verified; e.g. December 2024: deep-benchmark compaction, GNSS and heads by site for 彰雲嘉南屏, "雲林土庫鎮 4.9 公分為最大") and the 彰雲 monthly and InSAR bimonthly reports in the same system | monthly site values 2023-26 and the agency's own InSAR results | open PDFs; tables to be extracted and checked against the ring cache where they overlap |
| subsidence past 2021 | 112年 彰化與雲林地區地層下陷監測及分析 report (GPN 1011201949, verified): per-benchmark leveling appendix on CD-ROM; the 2026 leveling standard requires contractors to deliver per-benchmark elevation histories as spreadsheets | 2022-23 leveling at every benchmark, later years in the same series | formal request to the WRA (the standard is the justification) |
| canal deliveries | Jiji Weir annual table (verified here, section 8); the 計價水量月報表 held by 集管中心 | monthly bank-by-bank intake | request to 中區水資源分署 |
| canal deliveries | 農田水利署 actual irrigation use by 管理處 (open data id 691, annual since 2015) | annual district use | open CSV; annual only |
| Tuku trial | 立法院預算中心 audit of the 黃金廊道 programme (zones A/B/C 523/1,242/1,808 ha, phased 2013-20); 公視 summary (verified): 1,056 public deep wells cut, closed or sealed by 2019 | programme areas and dates, not measured responses | public pages; the measured response must come from the WRA reports above |
| InSAR 2023+ | none open from a Taiwanese agency; papers deposit nothing derived; COMET-LiCS frames exist but coverage of 彰雲 unchecked | fan-wide deformation would have to be computed from Sentinel-1 | LiCSBAS on a LiCSAR frame, or ASF HyP3 (Earthdata login) |

The scout's role ends at pointers. Following the first row by hand: the integrated monthly
reports exist at a predictable address from January 2024 to 2026-01 (25 monthly files fetched into
`data_fetch_api/new_measurements/wra_monthly_reports/`, with their source record); their
text tables carry the six deep benchmarks (monthly and annual compression) and the
observation-well heads, while the ring and GNSS values appear only as numbers printed on
figures, so they are not a reliable source for the ring record. The vendor API is not an
extension either: its rings and deep benchmarks end in May 2025 even in files fetched on
30 September 2026, and the open-data deep-pile feed ends in April 2025 (section 10). The
deep-benchmark tables of the monthly reports are therefore the only subsidence numbers
past May 2025; section 10 transcribes and scores them.

## 10. Subsidence evidence to January 2026, transcribed from the printed monthly reports

Every instrument feed the project can reach stops in spring 2025: the vendor API's rings
and deep benchmarks end in May 2025 even in files fetched on 30 September 2026, and the
WRA open-data deep-pile feed (data.gov.tw 58339, read through a real browser because the
host now answers plain clients with a script challenge) ends on 16 April 2025. The only
later subsidence numbers in the public domain are printed: table 3 of each monthly
地層下陷監測整合分析月報 gives, for six deep benchmarks (深層樁), the compression of the
month and of the trailing twelve months in centimetres. Four stand on the model grid
(溪州國小 300 m, 土庫國中 400 m, 秀潭國小 70 m, 中科虎尾 300 m); the deep piles were never
used in any calibration or model choice.

**Transcription with the Claude API, verified three ways** (`twin.report_extract`,
`results/twin/report_extract/deep_piles/`). Each report's table page was transcribed
twice, independently: the page as a PDF document by `claude-opus-5-5` and the page as
poppler layout text by `claude-sonnet-5-5`, both under a structured-output schema that
allows only copying (null where the table prints —). The two agreed on 135
of 143 values; in the other 8 the text model had dropped the 秀潭 row, a third
transcription (the PDF model on the text) agreed with the first, and the pages were read
by eye here. All 143 values are verified; 181 k input and 61 k output tokens.
Two checks on the printed record follow. The trailing-twelve-month identity
Y(m) − Y(m−1) = M(m) − M(m−12) holds within rounding for 48 of 49 testable months (the
exception is the reports' own arithmetic at 秀潭 in March 2025, 1.0 cm; the reports also
switch 土庫國中 between its 400 m and 300 m sensors in three months, which the check keys
on). Against the raw hourly feed, the printed monthly value is matched within 0.15 cm in
44 of 59 overlapping months, and every mismatch coincides with a step in the feed (a
step at every pile on 29 February to 1 March 2024, resets at 溪州 in March 2025 and at
灣內) or with 灣內, whose feed reads with the opposite sign. The printed values, which the
agency has already screened, are used; the feed is the check.

**Cumulative compression** is reconstructed per pile by weighted least squares on the
printed values (monthly differences at weight 1, trailing-twelve-month differences at
0.5 and only where the same sensor was printed twelve months earlier), so a month printed
as — is bridged by the year value and nothing is interpolated across an unconstrained
gap (195 constraints, residual RMS 0.05 cm). The sum of the twelve printed 2024
values agrees with the printed 2024 trailing-year value where both exist (土庫 3.11 vs 3.1 cm).

**Scores** (`twin.deep_pile_holdout`, `results/twin/deep_pile_holdout/`): the forward
archive's column (ensemble mean; 10–90 % band from the year-end members) at the pile's
cell, reference 31 December 2023, against the pile's printed trailing-year compression
in the January 2024 report continued (base_continue) and no displacement (base_zero).

| window | piles | model RMSE (R²) | continue last year | no change | inside 10–90 % |
|---|---|---|---|---|---|
| Jan 2024 – Dec 2025, all on grid | 4 | 1.27 cm (+0.45) | 1.09 cm | 3.53 cm | 2 of 4 |
| same, piles anchored ≥ 200 m | 3 | 0.50 cm (+0.92) | 1.26 cm | 3.94 cm | 2 of 3 |
| Jun 2025 – Jan 2026 (after every feed stopped), all | 4 | 0.70 cm | 2.01 cm | 0.62 cm | |
| same, ≥ 200 m | 3 | 0.57 cm | 2.22 cm | 0.65 cm | |

Per pile over the 24 months (observed / model / band, cm): 土庫國中 4.7 / 4.4 / 3.9–4.9;
溪州國小 1.0 / 1.2 / 0.8–1.8; 中科虎尾 4.8 / 4.0 / 3.7–4.4; 秀潭國小 (70 m) 1.8 / 4.2 / 3.6–4.6.
The median correlation of month-to-month increments is 0.55 (土庫 0.67, 虎尾 0.55,
溪州 0.14): the column follows the spring compression and summer rebound that the piles
record. After the feeds stopped, the column predicted the 2025 summer rebound at 土庫
(−1.1 cm against −1.1 observed), over-predicted it at 虎尾 (−0.8 against +0.1) and is level
with "no change" over those eight months, while continuing the previous year's rate is
off by 2 cm.

**Reading.** On the three piles that see most of the compacting column, the frozen
projection's two-year compaction is within 0.8 cm of the agency's own numbers at every
site, inside its band at two, and well ahead of both baselines; the 70 m pile is
over-predicted by 2.4 cm because the column is total depth, which is the caveat the ring
test already carries. This is the first subsidence evidence past May 2025 and the first
on an instrument family the model never saw. It is not a clean holdout: three of the
four piles stand at schools whose compaction rings were a Stage-2 target through 2022,
the pooled R² rests on three or four sites, and the model runs its calibration
climatology from the 2022 origin. The report carries these caveats verbatim; the page's
evidence list carries the line. The same transcription route extends month by month as
the WRA publishes: the series at the predictable address ran from January 2024 to
January 2026 at the time of writing (the 2023 files are not there), and a new file needs
one `twin.report_extract` run (earlier reports are cached) and one `twin.deep_pile_holdout` run.

## 11. The calibration rainfall was zero-filled; rebuilt from the CWA archive and recalibrated

**What was found.** The CWA station inventory (the old `e-service.cwb.gov.tw/wdps/obs/state.htm`
table, now only on the Wayback Machine, and its live replacement `codis.cwa.gov.tw/api/station_list`;
copies in `data_fetch_api/new_measurements/cwa_station_inventory/`) records a closure the
calibration rainfall cache does not: gauge rf11 (田中, C0G850) stopped on 2019-10-29. The
vendor cache (2012-2022, no missing value anywhere) fills every gap with 0 mm: rf11 is
exactly 0 for 192 days from October 2019 and for the whole of 2021 and 2022 (734 days,
2.9 m of rain at the neighbouring gauges), rf15 (崙背) is 0 for 223 days while its station
was being upgraded in 2015. Because the recharge interpolation counted those zeros as
observations, the 2021-22 recharge field had a dry hole at the fan apex: basin-mean rain
3 % low, 6 % of cells more than 10 % low, 44 cells with 田中 as their nearest gauge.
The same inventory shows two gauges renumbered on 2024-01-08 (北斗 and 芳苑 became
agricultural stations C2G840/C2G870) and nine gauges that changed ID in spring 2015.

**Rebuild** (`twin.cwa_rain`). Each gauge is a chain of CWA station IDs taken from the
inventory; CODIS was asked for every (station, year) of every chain, 399 requests, raw
responses and hashes kept under `data_fetch_api/new_measurements/cwa_codis_rain/raw/`. The
daily table is assembled with nothing invented: a day is a number only when CWA prints an
unflagged one (123,726 gauge-days), CWA's missing-value mark (13,737), sentinels (712) and
absent days are left empty, trace is 0.0. Where a renumbered gauge's two IDs overlap the
newer station wins. The recharge loader now takes a month's mean over its finite days and
masks months with fewer than 80 % of days, and the IDW down-weights a missing gauge-month
instead of reading it as dry. Day by day against the vendor cache on 2012-2022
(`results/twin/cwa_rain/report.json`): 89,907 of 95,295 jointly numeric days agree to
0.05 mm (94 %); the predecessor mapping is exact (土庫 C1K340 matches 365 of 365 days of
2013); 446 days are 0 in the cache and wet in CODIS (3.65 m of rain, 2.93 m of it at rf11);
4,942 days differ otherwise, two-thirds of them on 201 shared dates, nearly all heavy-rain
days of 2021-22 where CODIS carries a revised total; 9,173 cache numbers fall on days
CODIS flags or lacks. Two vendor series are not CWA data at all: rf25 (下水埔) before 2018
and rf24 (臺西) before 2015-03-28 match no CWA station within 15 km; the rebuild leaves them
empty. The rebuilt field changes the basin-mean rain by −6.6 % in 2013, +6.1 % in 2021,
+2.6 % in 2022 and within ±2.5 % in the other years. The record now runs to 2026-10-10
(the vendor cache ended 2022-12-31), so observed rain is available for the whole 2023-26
test period and for the forcing experiment, which had to use reanalysis rain.

The vendor cache is kept as `rf_timeseries_vendor_2012_2022.csv`; `rf_timeseries.csv` is the
rebuild, `rf_timeseries_source.csv` names the station behind every day, and the calibration
now records the rainfall file's hash in `stage3_theta.json`.

**Recalibration** (`results/twin_runs/par8/`, `stage3_datum_gate_cwa`). The deliverable's
recipe was rerun unchanged on the rebuilt rainfall, then the column refitted on leveling
(the column pairs with its head field), the posterior redrawn, the forward archive and
response basis rebuilt with the innovation update, and every evidence tool rerun into
`*_cwa` directories (batch 8, 18 jobs, finished 2026-10-12; `results/twin_runs/stage3_datum_gate_cwa/`,
`results/twin_forward/datum_gate_cwa.npz`, `results/twin/*_cwa/report.json`).

**Results: the rebuilt rainfall changes nothing the evidence can see.** Flow fit: in-sample
R² 0.865 (was 0.865); k-fold R² +0.643 against IDW +0.702 (was +0.643 / +0.702); own-mean
anomaly R² +0.342 against +0.605 (unchanged); the gate still fails. The parameters moved
within their posterior spread except the pumping-spread kernel, which fell from 2.7 km to
its 0.5 km lower bound (posterior median 0.8 km, was 3.2 km) with no change in skill: the
well record does not identify it, and that bound says so, not anything about the aquifer.
Leveling column: R² 0.652 in sample, 0.649 out of fold (was 0.656 / 0.652). Heads 2023-26
on the same 158 wells and 6,186 well-months: the model with its datum 2.042 m (was 2.048 m)
against climatology 1.667 m; the statistical layer stays at 1.507 m because it does not use
the physics; the physics' own anomaly ratio 0.999 (was 1.001). The 90 % ensemble band
covered 74.7 % (was 74.7 %), the combined band 92.9 % (was 92.6 %). Deep piles: 0.55 cm
on the three piles below 200 m, R² 0.90 (was 0.50 cm, 0.92); 1.30 cm on all four (was
1.27); after the feeds stopped 0.65 cm (was 0.70). Rings 2023-25: cumulative R² +0.24
(was +0.23), rates still not captured. Projection: basin-mean subsidence 2026-32 under
the baseline 2.66 cm (was 2.68), avoided by the 30 % irrigation cut 0.52 cm (was 0.50),
by retiring aquaculture 0.56 cm (was 0.58).

**Reading.** The dry hole was real and sat on the recharge field for two years, but recharge
is 4-5 % of rainfall in this calibration and the heads are set by pumping and by the apex
and coastal boundaries; the 2023-25 forcing screen (`results/twin/forcing_experiment/`)
had already shown observed weather moving the head RMSE by 0.03 m and observed pumping by
0.22 m. Correct rainfall was owed to the record, and the record now runs to October 2026
with a station named behind every day, which is what the next recalibration and a
gauge-based forcing experiment need. It is not a route to head skill; section 8's
deliveries and the pumping record are.

**Decision.** The page keeps the registered model (`datum_gate_innov`): both prospective
registrations are hash-bound to it, the recalibrated model is not distinguishable from it
on any test above, and re-registering is ruled out. `stage3_datum_gate_cwa` is the
reference for any later calibration (it records the rainfall file's hash) and its
artefacts are kept beside the registered ones. One tooling note: `merge_folds` now treats
a `git_commit` that differs between the full fit and a fold as provenance (logged) rather
than as a different configuration, since a commit made while the fold jobs were queued
had stopped the merge without changing what they ran.
