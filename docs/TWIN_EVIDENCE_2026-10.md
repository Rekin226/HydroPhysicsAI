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

The subsidence rows are fan-mean changes from December 2022 to December 2032, taken from the
forward summary (`datum_gate_innov.summary.csv`, `subs_forward_cm`). That summary pools 72
runs: the 36 of the page's leveling column and the 36 of the 30-year creep-ceiling column.
The page and its build log show the leveling column alone, which gives a **2.88 cm**
baseline (sd 0.60 over its 36 runs, `datum_gate_innov.members.csv`). The creep-ceiling
column gives 3.18 cm, and their mean is the 3.03 cm above. The avoided amounts agree in
both columns: 0.94 and 1.31 cm for the leveling column, against 0.94 and 1.30 pooled.

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
three forcings and scores each site's cumulative displacement and OLS rate against the
14 compaction rings (13 with enough months) and the 19 on-grid WRA GNSS stations. It is
not a clean holdout (the report lists seven reasons); it is the only subsidence
observation past the forecast origin.

**The rings change instrument across 2022.** No ring site was surveyed between December
2021 and February 2023 (北辰 not until July 2024), and across that gap every ring at a
site jumps by a common amount: −13 to −16 cm at eight sites, −7 cm at 元長, −2 cm at 溪州
and +15 to +50 cm at four (僑義, 北辰, 宏崙, 拯民), a change of depth reference, since the
shallowest ring otherwise moves millimetres a year. At the four the monitored interval
is continuous (its change matches the prior rate times the gap within 0.4 cm); at the
other ten it lengthens by 4 to 8 cm more than its rate implies, consistent with a probe
or tape scale change (`mechanism.ring_hiatuses`, `rings_hiatuses.csv`). The 2012-14
survey gap at eleven sites is not a discontinuity: the shallowest ring is unchanged
across it.
The first version of this test took each ring's last reading before December 2022 as
the reference, which for the rings is December 2021: the observed "cumulative" spanned
3.25 years and the instrument step while the model and the baselines spanned 2.25. The
test now starts each ring's cumulative at its first 2023 survey and differences the
model, the archive and the continued-rate baseline over the same months (February 2023
to March 2025, 2.1 years; `--max-ref-age-days 62`). Rates are OLS slopes inside the
window and did not change; GNSS starts in 2020 and is unaffected.

Rerun with otherwise identical inputs (same parameter sets, column, forcing and forward
archive, sha256-checked in the report): `results/twin/ring_holdout_ref2023/report.json`,
`rings_sites.csv` (with `ref_month`, `end_month` and the hiatus offset per site),
`gnss_sites.csv`, `rings_hiatuses.csv`.

| 2023-01 to 2025-03 | rings, cumulative (cm) R² / RMSE / bias | rings, rate (cm/yr) R² / RMSE / bias | GNSS, rate (cm/yr) R² / RMSE / bias |
|---|---|---|---|
| model, climatology forcing (the page's projection) | **−0.20 / 1.58 / +0.32** | −2.06 / 1.02 / −0.35 | −1.51 / 2.18 / −1.85 |
| model, recorded pumping | −4.03 / 3.24 / −1.86 | −5.64 / 1.50 / −0.77 | −3.40 / 2.89 / −2.56 |
| model, recorded pumping and weather | −3.98 / 3.22 / −1.83 | −5.26 / 1.45 / −0.68 | −3.19 / 2.82 / −2.49 |
| continue the site's own prior rate | −1.27 / 2.18 / +1.79 | −3.05 / 1.17 / +0.97 | −1.33 / 2.10 / +1.52 |
| no change | −4.31 / 3.33 / −3.00 | −5.23 / 1.45 / −1.33 | −2.56 / 2.59 / −2.20 |
| member p10-p90 covers the observed site value | 15 % (climatology) | 23 % | 5 % |

Superseded (reference across the gap, `results/twin/ring_holdout/report.json`): ring
cumulative +0.23 / 2.65 / +1.29 for the climatology run, −0.79 / 4.04 / +3.15 for the
continued rate, −0.45 / 3.63 / −2.02 for no change. The audit's approximate recompute
from that run's site table (−0.13 / 1.54 / +0.07) is close; it scaled the model's span
instead of differencing it over the observed months. The same
rerun fixes the ring *rate* coverage, which had been counted against the cumulative
band: it is 23 / 38 / 46 % for the three arms, not 31 / 0 / 0 %.

Reading: observed 2023-25 compaction is a fairly uniform 1.3-5.1 cm (嘉興 −0.8 cm), so
the step had been inflating the observed spread and with it the R². With the reference
fixed, the climatology run's absolute error falls 40 % and its bias nearly vanishes, and
it still beats both baselines (RMSE 1.58 against 2.18 and 3.33 cm), but it has no
cross-site skill: it puts 5-6 cm at 內寮, 土庫, 秀潭, 宏崙, 光復 and 拯民 where the rings
measured 3.4-5.1, and 0.7-2.2 cm where they measured −0.8 to 3.9. The rings slowed after 2022
(site rates 0.8-4.0 cm/yr in 2017-22, −0.2 to 2.0 inside the window), so continuing the
prior rate overshoots and no-change undershoots. Under recorded pumping the column
compacts up to 4 cm in January-February 2023 (the 2023 dry-season drawdown) and rebounds
afterwards (元長: +4.0 cm from December to February, then −3.9 cm to March 2025), so from
the February reference those two arms under-predict by about 1.9 cm; their score is
dominated by that elastic swing, which the rings, unsurveyed until late February, cannot
confirm or rule out. The rate signatures are unchanged: the column under-predicts the
ongoing rate, more so under the recorded (lower) pumping, because recovering heads set
no new minima and the VEP column then only creeps; the rings kept compacting. That is
the same deficit section 6 finds. The member spread is far too narrow to be read as a
range. GNSS rates carry full-column and tectonic motion the model does not represent;
the model does not beat the continued-rate baseline there.

The page's evidence line can read this report (build with
`--ring-holdout results/twin/ring_holdout_ref2023/report.json`). The live page does not yet
do so. The 2026-10-11 page and its 2026-10-23 rebuild both embed the superseded
`ring_holdout/report.json` (sha256 `71286b0a`, see `docs/TWIN_RELEASE.md`), so their ring
line shows the pre-fix numbers. Built on the corrected report, the line states that each
site's cumulative starts after the 2022 gap and gives RMSE beside R², and the
discontinuity is caveat (vii) in its independence list. The `mechanism` table of mean
ring rates across 2021-23 (`rings_recovery.csv`, `rate_2021_2023_cm`) spans the same
step; it is now flagged per site (`disc_2021_2023`) and its median is taken over
unflagged sites only, which leaves none.

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
(溪州國小 300 m, 土庫國中 400 m, 秀潭國小 70 m, 中科虎尾 300 m). The deep piles were never
used in any calibration. Their scores below were later used to choose between columns
(ruling U7, 2026-10-21, section 12), so since then they are report-only and selection-
contaminated, not independent skill.

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
site, inside its band at two, and well ahead of both baselines (report-only since U7,
2026-10-21: these scores were then used to choose between columns); the 70 m pile is
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
(was +0.23), rates still not captured. Projection: fan-mean subsidence from the end of
January 2026 to December 2032 (`subs_mean`, pooled over both columns) under the baseline
2.66 cm (was 2.68), avoided by the 30 % irrigation cut 0.52 cm (was 0.50), by retiring
aquaculture 0.56 cm (was 0.58). That window starts after the policy's first month, in
which most of the avoided subsidence arrives, so it is not the page's number. On the basis
of section 1 and the page (December 2022, or equally December 2025, to December 2032) the
avoided amounts are 0.96 and 1.26 cm (was 0.94 and 1.30; `datum_gate_cwa.summary.csv`,
`datum_gate_innov.summary.csv`).

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

## 12. Phase 1 outcome and the registered policy-response tests (2026-10-23)

**Phase 1: no candidate replaces the deliverable.** Every rung was declared before it ran
(`results/dataprep/preregistration/I1.*`) and scored against gates fixed by R0, the
`stage3_datum_gate_cwa` recipe. Detail is in `docs/DATA_REMEDIATION_STATUS_2026-10.md`;
the numbers are in `results/dataprep/reports/I1_*/report.json`.

| Rungs | What they added | Outcome |
|---|---|---|
| R1, R1b-R1e | Changhua and Chiayi pump census, canal recharge, a county-split column, public-supply wells | No rung passes the design-time gates. On fresh folds (G2), R1c and R1e-pw fail too |
| R4, R4p | per-layer pumping shares (learned, then measured), lagged return flow | Fail |
| R* | R4p plus a learned Changhua pump-efficiency factor | Fails 5 of 7 gates. The factor went to 1.55, not below 1 |

What Phase 1 shows:
- The Changhua census gives the right seasonal amplitude: predicted to observed goes from 0.39 to between 1.07 and 1.40.
- No rung reproduces Changhua's year-to-year changes: the interannual correlation stays at or below 0.52, except with canal water alone, which reaches 0.63.
- The R1c-based rungs forecast uplift after 2022. That uplift is an artefact of their 6-zone column.
- Changhua's damped head response is not an efficiency effect. The remaining suspect is water the model does not carry, chiefly canal water outside the Jiji supply. Those data are not available.
- R0 stays the deliverable for heads and subsidence (user ruling 2026-10-21). Its 2023-26 deep-pile scores were used to choose between columns, so they are no longer reported as skill (U7).

**Registered tests of the policy response (`N1.phase12_response.v1`).** Results are in
`results/dataprep/reports/N1_evidence/phase12_tests/`. Each was scored once, after
`prereg3.assert_can_score`, which verified the inputs and found no blocking look.

*G1, masked 2015 refit: passes against both baselines.*
- **Setup.** R0's recipe was refitted with Jan-May 2015 hidden from the head targets (`calibrate_flow --mask-months 2015-01:2015-05`, batch 23, `results/twin_runs/stage3_datum_gate_m15/`). The in-sample R² was 0.867; R0's was 0.865.
  - The I1 primary model is R0, because no I1 rung passed, so the registration's two models are this one refit.
- **Scoring.** The refit's own rollout M was scored on the 2015 drought-spring anomaly against two baselines:
  - CL: climatology, an anomaly of 0;
  - ZR: the same rollout with the Jan-May 2015 pumping energy replaced by its reference-year mean, cell by cell.
- **Coverage.** 135 of the 158 calibration wells (63 sites) meet the registered eligibility rule. The paired site bootstrap used B = 2000 and seed 20261024.

| | wells | RMSE M (m) | RMSE baseline (m) | ratio | 90 % interval of the difference (m) | verdict |
|---|---|---|---|---|---|---|
| M vs CL | 135 | 1.490 | 2.062 | 0.72 | -0.674 to -0.462 | model better |
| M vs ZR | 135 | 1.490 | 1.749 | 0.85 | -0.319 to -0.198 | model better |
| Changhua, M vs CL | 59 | 1.211 | 1.362 | 0.89 | -0.209 to -0.105 | model better |
| Yunlin, M vs CL | 76 | 1.675 | 2.472 | 0.68 | -0.906 to -0.687 | model better |

ZR is informative: its anomaly differs from M's by an RMS of 0.63 m, above the 0.05 m floor.
So the improvement over ZR is the model's response to the recorded 2015 pumping, not its
seasonal cycle.

Reported, not gating: the response slope, the observed on the modelled event anomaly
through the origin, is 2.13 (90 % interval 1.93 to 2.39).
- The sign agrees at 96 % of wells, with correlation 0.58.
- The model gets the direction of the 2015 shortfall right almost everywhere, but only about half its size in this event. This is the damping of section 4, now seen out of sample. The event's anomaly mixes canal water and rain with pumping, so the slope is not a correction factor for a pumping change; step C (section 13) measures the pumping response on 2016-18 at 1.21 in Yunlin.
- The external wells of `N1.T1_ext_E2015` give 1.76, so the damping is not an artefact of the calibration wells.

Changhua gains less than Yunlin (ratio 0.89 against 0.68), as in every Phase 1 result.

Caveats:
- Only the targets were masked. The 2015 forcing (pumping energy, rain) stayed in the record, as registered, because the test asks whether the forced response is right.
- The recipe itself was designed on fits that included 2015, so G1 holds out the parameters, not the design.
- The column was not refitted; G1 tests heads only.

*G3, canal-delivery channel: not run.* The registration applies G3 only to a G1 recipe
with a canal-delivery term, and R0 has none.

*G4, deep benchmarks, 2020 and 2021 springs against 2022: not evaluable.*
- No printed sensor has all five Jan-May 2022 months verified: each has 4, and the fifth has no row in `deep_benchmarks_printed/v2`. So none of the 14 sensor-events can be scored.
- The scorer wrote `not_model_better`. `g4/erratum.json` corrects this to `not_evaluable`; results.json is left as written.
- Model side only: the column's Jan-May compaction at every pile cell is larger in 2020 and 2021 than in 2022, which is the registered direction. Example: 37 and 57 mm against 17 mm at 土庫.
- Two further limits stand regardless:
  - the column returns only total compaction, so the sensor's depth interval is not resolved;
  - the product's `positive_means` label is ambiguous and must be checked before any rescore.
- G4 can be rescored only with a fifth 2022 month or under a new registration.

C3 and C5, the masked 2020 and 2021 refits, appear in the plan but were never registered.
They need a new registration before they can run.

**Reading.**
- G1 is the first out-of-sample evidence that the model's head response to a drought-year pumping change is real:
  - it beats both climatology and its own zero-response counterfactual on wells and months it never saw;
  - in both counties.
- The modelled response is smaller than observed; by how much depends on the test. In the 2015 event the observed-on-modelled slope is 2.13, but that anomaly is confounded by canal water and rain. On the masked 2016-18 refits (section 13, design-time) it is 1.21 (1.06-1.36) in Yunlin and 0.85 (0.63-1.05) in Changhua on the census model, whose interval includes 1. The policy numbers inherit this limit: the head change under a pumping cut is likely smaller than the real one in Yunlin. No calibrated factor follows from either test.
  - The column was calibrated on leveling under these same damped heads, so its compressibility may partly make up for it.
  - Whether avoided subsidence is understated is therefore not settled by this test.
- No test here scores the subsidence side of the policy response. G4 has no complete 2022 spring, and the 2023-26 pile record is selection-contaminated. Section 13's steps B and C score it later; the 1-3-year size is not supported.

Scorer provenance. The G1 and G4 scorers
(`dataprep_staging/n1_natural_experiments/run_phase12_tests.py`, sha256 `90a15a4d`, and
`run_g4_piles.py`, `85834cca`) are not bound by `N1.phase12_response.v1`. The registration
(2026-10-13 20:00 UTC) predates them and hashes only `n1lib.py`, `n1const.py` and the data
inputs, and `phase12_tests/results.json` records no scorer hash. What ties them to the
scores:
- their file times: 2026-10-22 12:31 and 12:37 UTC, before the G4 look (12:39 UTC) and the G1 look (16:13 UTC) in `looks.jsonl`;
- commit `ba76923`, which froze them unchanged after the scoring.

`docs/N1_CODE_PROVENANCE_2026-10.md` records the hashes.

## 13. Policy-response tests of heads and subsidence, and the page labels (2026-10-23)

Two registrations, each written before its data were read:
- `N1.policy_response.v1` (sha `6c3a4e0c`): steps A-C below;
- `N1.changhua_fix.v1` (sha `be85ab6b`): two Changhua fixes, step B's one exploratory follow-up.

Every test was scored once, after `prereg3.assert_can_score`. Scripts are in
`dataprep_staging/n1_policy/`; results are in `results/dataprep/reports/N1_policy/`, with
every attempt listed in its `SUMMARY.md`, failures and engineering re-runs included.
Yunlin is scored on R0. Changhua is scored on the R1c census flow with a three-zone leveling
column, because R0 carries no Changhua pumping.

**Which model was tested.** R0 here is `stage3_datum_gate_cwa`: the deliverable's recipe on
the CWA-rebuilt rainfall (theta sha256 `789d0e45`). Steps A1 and B, step C's Yunlin refit
and the registration's inputs all use it. The decision page's levers come from a different
theta: the registered run `datum_gate_innov` (theta `stage3_datum_gate`, sha256 `036a7a17`,
vendor rainfall, commit `fa5cca2`, basis `response_basis_innov.npz`). Section 11 found the
two within noise on every test it ran, and the page was not rebuilt on R0 because both
prospective registrations are bound to `datum_gate_innov` by hash. The verdicts below
therefore transfer to the page's levers through section 11's equivalence. They were not
scored on the page's run itself. A post-hoc, model-only comparison (not a test; see "Post-hoc
model-only checks" below) finds the two models' policy responses within 2-4 % fan-wide.

**Step A, model-only (spends no data).**
- **A1, compensation.** A leveling-fitted column compensates for damped heads fan-wide, but not zone by zone. A1 ran on R0 (`stage3_datum_gate_cwa`) and on R1c with its three-zone column, not on the page's run (`step_a/a1.json`). The table gives fitted/true subsidence avoided by the cut30 lever at 2021-12, when the true head response is k times the model's:

| R0 column zone, county | k = 1.5 | k = 2 |
|---|---|---|
| proximal, Yunlin (130 cells) | 0.66 | 0.75 |
| proximal, Changhua (134) | 0.69 | 0.79 |
| mid, Yunlin (499) | 0.91 | 0.88 |
| mid, Changhua (736) | 0.84 | 0.77 |
| distal, Yunlin (576) | 0.96 | 0.98 |
| distal, Changhua (73) | 1.05 | 1.07 |
| fan | 0.94 | 0.93 |

  The registered rule (fan ratio within 0.8-1.25 at k = 2) calls this "compensates". The proximal cells carry little avoided subsidence (0.68 m summed over Yunlin's 130 cells at k = 2, against 98.7 m in its distal zone). The township-level shortfall therefore sits mainly in mid Changhua, where a head response twice the model's would leave the avoided subsidence 23 % too small (0.77). On R1c the proximal ratios at k = 2 fall to 0.49 (Yunlin) and 0.32 (Changhua), and mid Changhua is 0.91.
- **A2, lever ranking: not about the page.** A2 loaded `results/twin_forward/response_basis.npz` (built 2026-09-20 at commit `2f21f81`, in-sample R² 0.915, a pre-datum-gate model; `step_a/a2.json` records the path). The page uses `response_basis_innov.npz` (`fa5cca2`), and R0's is `response_basis_cwa.npz` (`9715f58`). The registration describes that file as "the page's response basis (R0, one member)", which is wrong. A2 ran as registered on the registered file, but its results describe a superseded model: the 2030 ranking survived, the 2026 ranking failed the rule on a near-tie between "other" and "industry", and hot spots were fragile. "The ranking is robust" is withdrawn as a statement about the page's levers. A2's own result stays as written. A post-hoc, model-only rerun of the identical procedure on the page's basis (A2b, not a test; see "Post-hoc model-only checks" below) says what holds for the page instead: the cm ranking, the per-100-GWh/yr near-tie and the hot spots.

**Step B, 2007-10 leveling dose-response across places** (subsidence per pumping exposure, fixed effects for benchmark and campaign, 5 km block bootstrap).
- **Yunlin: inconclusive.** Observed +0.66 against R0's +0.26 cm/yr per GWh/yr. The model's own three-campaign estimates range from -0.23 to 1.26, so the method cannot separate the two.
- **Changhua: inconsistent.** Observed -0.50 against the census model's +0.80, below every model-world window. Entry and exit pumps explain only 24 % of the exposure variation.
- The 2010-12 leveling is unspent by `looks.jsonl`: its pre-declared power rule failed. It is not a clean judge: the W3 looks of 2026-10-12 read 2011-21 leveling at benchmarks that overlap its 2011-12 surveys, step B's campaigns include May 2010, and step B's one exploratory follow-up is spent (status doc, N1 policy-response tests).
- Declared secondaries (reported, never gating) are in the table under "Declared secondaries" below.

**Step C, masked 2016-18 refits.** Heads are masked, and the column is refitted with the May 2016-18 surveys withheld. Earlier looks touch this window, so it is labelled design-time.

| | M vs no change (CL) | M vs reference-year pumping (ZR) | observed on modelled slope |
|---|---|---|---|
| heads, Yunlin (75 wells) | model better, 1.38 vs 2.21 m | model better (2.29 m) | 1.21 (1.06-1.36) |
| heads, Changhua (59 wells) | model better, 1.07 vs 1.19 m | model better (1.21 m) | 0.85 (0.63-1.05) |
| subsidence partition, Yunlin (355 benchmarks) | baseline better, 1.88 vs 1.00 cm | baseline better (1.08 cm) | 0.22 (0.17-0.27) |
| subsidence partition, Changhua (255 benchmarks) | baseline better, 0.64 vs 0.58 cm | model better (0.71 cm) | 0.40 (0.30-0.49) |

The subsidence partition is the May 2015-19 increments minus their mean. Intervals are 90 %
cluster-bootstrap intervals.

**Changhua fixes, scored beside the R1c reference.**
- **F1, a 10-year floor on the column's viscous time constant.** It failed its design gate: rest-of-fan out-of-fold leveling R² was 0.601 against a floor of 0.655. As registered, it was not scored and not repaired.
- **F2, a paddy-weighted irrigation return flow** (`--return-rice-share`, learned slope 1.84). It passed its gates but changed nothing:
  - heads are indistinguishable from R1c (ratio 1.002);
  - leveling is slightly worse (ratio 1.014).

**Reading.**
- **Heads:** the head response to a pumping change is supported in Yunlin on the R0 recipe (`stage3_datum_gate_cwa` refitted with 2016-18 masked) and in Changhua on the census model (R1c flow, three-zone column). The page's model carries no Changhua pumping, and neither does R0, so neither can represent a Changhua pumping change; the R0 recipe scored on Changhua (the declared cross-county secondary) is marginal: ratio 0.955 to no change, 90 % interval upper bound -0.0002 m. This is a second masked-parameter, design-time window after G1 (section 12), not a fresh period: its parameters are masked, but the recipe was designed on fits that saw it.
  - Size: in Yunlin the modelled response is too small. The observed-on-modelled slope is 1.21 (1.06-1.36) on 2016-18, against 2.13 (1.93-2.39) in G1's 2015 drought spring. G1's event mixes canal water and rain with pumping, so the two numbers measure different things; neither is a calibrated correction. In Changhua the slope is 0.85 (0.63-1.05), not distinguishable from 1.
  - The page template now prints both slopes beside the heads verdict. The live page shows them from its next rebuild.
- **Subsidence, 1-3-year size: not supported.**
  - Year to year in 2016-18, the observed swings do not follow the columns, and a constant rate fits better.
  - Across places in 2007-10, Yunlin sank more than modelled, though inconclusively.
  - The two tests measure different things and disagree in direction, so the size is labelled "not supported", with no direction attached.
  - The column is fitted to the cumulative leveling record, not to a policy difference. In this model the avoided subsidence is an elastic response that arrives within about a year; its 1-3-year size is not supported by the registered tests, and whether a cut changes long-run sinking is untested (the post-hoc elastic split below shows why the model has no other kind of response under mean forcing).
- **Changhua levers: not supported.** The page's model has no Changhua pumping. The census model's response failed step B, and neither fix closed the gap. Step B's exploratory follow-up is spent, so no further Changhua fix may be scored on the 2007-10 or 2016-18 data.

**Post-hoc model-only checks (2026-10-24).** Each item below is post hoc, model-only and
not a test: it reads model outputs only (forward runs, member fields, response bases, the
page payload), reads no observation, scores nothing and is in no registration. The files are
under `results/dataprep/reports/N1_gaps_2026-10-24/`; the scripts that wrote them
(`dataprep_staging/n1_gaps/`, recorded in each file's `provenance`, git head `82da213`) were
committed afterwards in `6ea3b2b`.
- **A2b, lever ranking on the page's basis** (`science-04/`; post hoc, model-only, not a
  test). The identical A2 procedure (seed 20261223, 500 draws per start) rerun on A2's own
  basis reproduces `step_a/a2.json` exactly (`a2_repro_check.json`, largest difference 0.0),
  so only the basis differs. On the page's basis (`a2b_innov.json`, `response_basis_innov.npz`),
  on the page payload (`a2b_page_payload.json`, page `8a5950e8`, rescaled to the solved runs)
  and on R0's basis (`a2b_cwa.json`):
  - the cm ranking (irrigation, aquaculture, livestock, domestic, industry, other) holds in
    100 % of draws at both starts. Industry is above other (0.068 against 0.052 cm fan-mean
    by 2032 on the page's basis); A2's "other above industry" near-tie belongs to the
    superseded 10 km basis only;
  - the per-100-GWh/yr ranking, which A2 never tested, is a near-tie: 2.84-3.10 cm per
    100 GWh/yr across the six classes on the page payload. Its order holds in 3 % of draws on
    the page payload (3.2 % for 2026 starts, 3.0 % for 2030), 10-13 % on the page's raw basis
    and 16-21 % on R0's. It is not a robust ranking. On the raw basis irrigation and
    aquaculture are 3.09 and 3.08; the page's order between them comes from the per-class
    rescaling factors (0.920 irrigation, 0.970 aquaculture, 0.945 the rest), not from the
    model's response;
  - hot spots are more fragile than A2 said: for the irrigation lever the top-decile cell
    Jaccard has 5th percentile 0.00 and median 0.72 (2026 starts; 0.00 and 0.71 for 2030),
    against A2's 5th percentiles of 0.10 (2026 starts) and 0.41 (2030). The aquaculture lever's top decile is
    steadier (5th percentile 0.61, median 1.00).
- **Transfer from R0 to the page's model** (`science-05/transfer_innov_vs_cwa.json`; post
  hoc, model-only, not a test). Avoided subsidence from Dec 2022 to Dec 2032, page model
  (`datum_gate_innov`, spread 2.71 km) against R0 (`datum_gate_cwa`, spread 0.5 km, at its
  lower bound): fan-mean 0.942 against 0.958 cm for cut30 and 1.306 against 1.261 cm for
  retire_aqua (2 % and 4 %); cell correlation 0.9999 and 0.9996; top-decile cell Jaccard 0.98
  and 0.92; township Spearman 0.999 for both, largest township difference 0.04 and 0.23 cm.
  The layer-2 head change at the 158 well cells agrees to a ratio of means of 0.988 and 1.017
  (r 0.999 and 0.998). The cut30 top three townships are the same set, but the leader differs
  (Baozhong on the page's model, Beigang on R0's).
- **Two "Yunlin" cell sets.** The page's headline Yunlin figure and the post-hoc files
  average different cells. The page averages the 1,091 cells of the named Yunlin townships
  (every cell outside the pooled "Changhua (not yet split by township)" panel of 1,057
  cells): cut30 1.49 cm, retire_aqua 2.20 cm by Dec 2032 on the page's model. science-05 and
  science-13 use the model's county mask (`zone_county`, also used by A1): 1,205 Yunlin and
  943 Changhua cells, so about 114 cells the mask calls Yunlin sit in the pooled panel. On
  that mask the Yunlin mean is 1.398 cm for cut30 and 2.093 cm for retire_aqua
  (`transfer_innov_vs_cwa.json` `yunlin_mean_cm`, `elastic_split.json` `yunlin`). Both
  weightings give the same fan mean (0.942 and 1.306 cm). From `twin_app_20261024e.html` on,
  the headline names its cell set ("over the 1,091 cells of the named Yunlin townships").
- **Hot-spot stability** (`science-10/hotspot_stability.json`; post hoc, model-only, not a
  test). Among the ten townships the page names (pooled and low-confidence townships are
  excluded, as on the page):
  - cut30: the page's headline is Baozhong, Beigang, Dongshi; Baozhong leads Beigang by
    0.02 cm. Baozhong leads in 30 of the page model's 36 runs (83 %), in 26 of R0's 36 runs
    (72 %) and in 57 % of the A2b draws for 2026 starts (Huwei leads in 32 %). The top three
    are the same set in every run of both models, but in only 59 % of those draws. R0's ensemble puts Beigang
    first, by 0.003 cm;
  - retire_aqua: the page's headline is Taixi, Shuilin, Mailiao. Taixi leads in 22 of 36
    page runs (61 %), 30 of 36 R0 runs (83 %) and every A2b draw; the top three are the same
    set in 67 % and 83 % of runs;
  - by the file's stated rule (stable if at least 90 % of the page's runs and of the A2b
    draws keep it and R0's ensemble agrees), neither leader and neither top-three set is
    stable. Read each headline as a group of three, not an order.
- **Elastic and inelastic parts** (`science-13/elastic_split.json`, `elastic_context.json`;
  post hoc, model-only, not a test). The page's column (`coupled_leveling`), driven by the
  forward run's ensemble-mean heads, splits the avoided subsidence into its elastic and
  inelastic terms:
  - by Dec 2032 the cut30 benefit (0.942 cm fan-mean) is all elastic, inelastic 0.000 cm,
    and so is retire_aqua's (1.306 cm). By Dec 2026, 98 % and 99 % of it has arrived;
  - the reason: under the forecast's mean forcing no cell's layer-mean head falls below its
    preconsolidation head after Jan 2026, in the baseline or under either policy (margin
    above it: 5th percentile 0.24 m, smallest 0.03 m). The baseline's inelastic forward
    subsidence (3.75 cm fan-mean, from the mid zone's 3960-day creep toward loads set before
    2026) is identical with or without a policy, so a cut changes only the elastic term;
  - lifting the cut30 policy in Jan 2030 (an abrupt return of the heads to the baseline)
    leaves none of the benefit by Dec 2032;
  - limits: ensemble-mean heads, not each run; mean forcing only. A dry year that drew heads
    below that margin would add inelastic loading that a cut could avoid; that case is not
    in the page's runs.
  - Reading (wording chosen by the user, 2026-10-24): in this model the avoided
    subsidence is an elastic response that arrives within about a year; its 1-3-year size
    is not supported by the registered tests, and whether a cut changes long-run sinking is
    untested.

The derived page labels (`dataprep_staging/n1_policy/page_posthoc.py`, output
`results/dataprep/reports/N1_gaps_2026-10-24/page_posthoc.json`) carry the hot-spot, per-GWh
and elastic-split numbers onto the page through `viewer_app --posthoc` (off by default),
each labelled post hoc, model-only, not a test.

**Page labels.** `twin.viewer_app --policy-response
results/dataprep/reports/N1_policy/page_labels.json` puts these verdicts on the model card
and a one-line caveat under the levers, in both languages. The October 23 page
(`results/twin/twin_app_20261023.html`, promoted to `twin_app.html`) was rebuilt from the
same hashed inputs as the October 11 page. Those inputs include the October 11 ring report
(`results/twin/ring_holdout/report.json`, sha `71286b0a`), which section 5 marks superseded,
so that page still prints the old ring line (R² +0.23 against -0.79). It passes
`release inspect` and `browser_check`.

**Corrected page (2026-10-24, not promoted).** `results/twin/twin_app_20261024b.html` is
built with every other input unchanged, but with `--ring-holdout
results/twin/ring_holdout_ref2023/report.json` (sha `92e621d4`, caveat vii) and
`--policy-response results/dataprep/reports/N1_policy/page_labels_v2.json`
(`dataprep_staging/n1_policy/page_labels_v2.py`: the v1 verdicts plus the model each row
was scored on, the design-time flag, step B and the declared secondary; nothing
re-scored). It names the census model for the Changhua head result, labels the 2016-18
window design-time, shows the deep-pile line as report-only (U7), calls the policy gate a
model-internal sanity check and leaves it out of the test count, counts only the
N1 rows scored on the page model's recipe (a presentation choice, open for review), and says the levers act on Yunlin metered pumping only.
It was built from the code fixer's branch alone, before the two fixer branches were
merged, so it lacks the head-response size line, the head-gain caveat and the neutral
subsidence wording below.

**Rebuilt pages (2026-10-24, not promoted).** `results/twin/twin_app_20261024c.html`
(sha256 `499e9025`, build log `abaa18e3`, browser report `fd9a03d8`) is built from the
uncommitted merge of the two fixer branches (release `code_sha256` `e585666f`, which matches
no commit) with the 20261024b inputs plus `--posthoc
results/dataprep/reports/N1_gaps_2026-10-24/page_posthoc.json` (`fb59adcc`). Its command,
the first line of its build log, is:

```bash
python -m hydrophysics.twin.viewer_app --forward results/twin_forward/datum_gate_innov.npz --basis results/twin_forward/response_basis_innov.npz --basemap results/twin/basemap.npz --challenge results/twin/new_data_challenge_innov/report.json --coverage results/twin/coverage_datum_gate_innov/report.json --prospective results/twin/prospective_2026-11/manifest.json --ring-holdout results/twin/ring_holdout_ref2023/report.json --hybrid results/twin/hybrid_datum_gate_innov/report.json --hybrid-registration results/twin/prospective_2026-11_hybrid/manifest.json --deep-piles results/twin/deep_pile_holdout/report.json --policy-response results/dataprep/reports/N1_policy/page_labels_v2.json --posthoc results/dataprep/reports/N1_gaps_2026-10-24/page_posthoc.json --out results/twin/twin_app_20261024c.html
```

It carries the head-response size line, the head-gain caveat, the neutral subsidence wording
and the post-hoc labels. `results/twin/twin_app_20261024d.html` (sha256 `7b6350c3`, build
log `aa3356fb`, browser report `f3fe6583`, `code_sha256` `3461dd9b`) is the same command
with `--out results/twin/twin_app_20261024d.html`; its sources and payload equal
20261024c's, and its only change is that the post-hoc lift sentence (en and zh) says the
heads return abruptly to the baseline. Both pass `release inspect` and `browser_check`.
20261024c is superseded: its lift sentence ("leaves 0.00 cm of its benefit by Dec 2032")
has no qualifier, although science-13 splices the ensemble-mean heads abruptly and the flow
model's own return would take months; it is not a promotion candidate.
`results/twin/twin_app_20261024e.html` (sha256 `e37a36d3`, build log `94becf35`, browser
report `65744226`, `code_sha256` `ea6cadc9`) is the same command with `--out
results/twin/twin_app_20261024e.html`; its sources and build log equal 20261024d's, and its
only change is that the Yunlin figure in the headline names its cell set (en "over the 1,091
cells of the named Yunlin townships", zh 「僅計雲林已命名鄉鎮的 1,091 個網格」; see "Two
'Yunlin' cell sets" above). It passes `release inspect` and `browser_check`.
This page was promoted to `results/twin/twin_app.html` on 2026-10-24 (user decision); the
2026-10-23 page stays as `twin_app_20261023.html`.

**Declared secondaries.** `N1.policy_response.v1` declares them "reported beside the
primary": for step B, the model's hindcast fixed-effects regression (`hindcast_fe_beta_E3`)
and the E10 coefficients; for step C, each model scored on the other county and pooled
observed-on-modelled slopes. Values are copied from `step_b/results.json` (sha256
`207f6d37`) and `step_c/results.json` (`edcd5031`), with step B in cm/yr per GWh/yr
(beta × 1e8) and 90 % intervals:

| step B, 2007-10 leveling | Yunlin (R0) | Changhua (R1c census model) |
|---|---|---|
| E10 coefficient, observed | +0.045 (-0.071 to +0.176) | -0.221 (-0.347 to -0.110) |
| E10 coefficient, model | +0.221 | +0.165 |
| E3 coefficient, model hindcast with fixed effects | +0.141 | +0.004 |
| (primary, E3: observed / model pulse runs) | +0.66 / +0.26 | -0.50 / +0.80 |

| step C, cross-county (masked 2016-18) | vs no change (CL) | vs reference-year pumping (ZR) | observed on modelled slope |
|---|---|---|---|
| Changhua heads, scored with R0 (59 wells) | model better, 1.13 vs 1.19 m (ratio 0.955; interval of the difference -0.116 to -0.0002 m) | model better, 1.20 m (ratio 0.94) | 0.77 (0.51-1.05) |
| Changhua leveling, scored with R0 (255 benchmarks) | baseline better, 0.75 vs 0.58 cm (ratio 1.29) | indistinguishable, 0.77 cm (ratio 0.97) | 0.10 (-0.07-0.24) |
| Yunlin heads, scored with R1c (75 wells) | model better, 1.42 vs 2.21 m (ratio 0.64) | model better, 2.28 m (ratio 0.62) | 1.16 (1.02-1.31) |
| Yunlin leveling, scored with R1c (355 benchmarks) | baseline better, 2.47 vs 1.00 cm (ratio 2.47) | baseline better, 1.25 cm (ratio 1.98) | 0.18 (0.15-0.21) |

The pooled slopes (both counties together) were declared but not computed, because the
scorer wrote none. They are not computed now either: doing so would read spent observations
again after the look. Reading of the secondaries: they agree with the primaries. R0 alone,
with no Changhua pumping, edges past both baselines on Changhua heads (the interval of the
difference ends at -0.0002 m), and no model's column follows the 2016-18 leveling
partition in either county.

**Known defects in the head targets.** Every model cited in sections 11-13 (the page's run,
R0, the G1 and step C refits, R1c) was fitted with the historical defaults of
`heads.build_head_field` and `calibrate_flow.prepare_series`, and every score was computed
on targets built the same way. Nothing was rescored.
- Monthly means weight samples equally. After the hourly-to-10-minute logging switch on 2019-02-20 the late days count six times, which shifts February 2019 by up to 1.7 m at 13 Yunlin wells. `--day-weighted-months` (opt-in) fixes it in `calibrate_flow`, `inputs` and `forward` only. `challenge`, `coverage`, `rescore_temporal`, `explorer3d`, `calibrate_mlcw` and `sk_leveling` still build equal-sample heads, so a day-weighted model (Q0) is evaluated against equal-sample observations.
- Coverage is counted as samples over hours, so 10-minute records pass the 80 % rule at up to six times their true coverage, and the gap check ignores leading and trailing gaps (`--strict-coverage` is opt-in).
- Calibration targets are back-filled: interior gaps are interpolated and leading or trailing gaps filled by the nearest value (`--no-backfill` is opt-in).

**Code provenance.** The batches that produced the N1 models recorded their code as a
diff with no base commit (`results/twin_runs/par2{3,4,5}/code.diff`; no `code.base` file).
The registrations hash the par24 and par25 diffs, and neither `step_c/results.json` nor
`fix_c/results.json` records the tree the scorer imported. Base commits, reconstructed and
checked by comparing each recorded diff, restricted to `hydrophysics/`, with the commit's
diff (index lines ignored):

| batch | used by | recorded diff (sha256) | equivalent to | on this branch? |
|---|---|---|---|---|
| par23 | G1 refit `stage3_datum_gate_m15` | `af53ef55` | `ccb6ff2..d3b365d` (`calibrate_flow.py`; the diff also appends the then-untracked test file) | yes |
| par24 | step C refits and columns, step A1 and step B runs | `2618484a` | `c48e858..294ed2e` (`calibrate_coupled`, `forward`, `explorer3d`) | yes, merged in `c31a5a9` |
| par25 | fixes F1 and F2 | `277d6a50` | `c48e858..8a96b21` (adds `calibrate_flow --return-rice-share`) | yes, merged in `c31a5a9` |

Tests were untracked at run time and are not part of these diffs' `hydrophysics/` part.
`--withhold-surveys`, `--column-zones 3` and `--return-rice-share` reached this branch with
the merge `c31a5a9`; `76ae637` then made `twin.forward` rebuild the merged terms
(`return_rice_b`, the collapsed three-zone column), which it had refused. To rerun a scorer
exactly as it ran, put a checkout of `294ed2e` (steps A-C) or
`8a96b21` (fixes) first on `PYTHONPATH` and run it with `python -P`, as
`results/twin_runs/par24/py.sh` and `par25/py.sh` do. Do not add `-I`. The usage lines of
`run_step_a.py`, `run_step_b.py` and `run_step_c.py` say `py.sh -I`, which drops
`PYTHONPATH` and imports the main tree's twin modules. That is the logged engineering
failure of step B and A1, and the files stay as they are because the registration hashes
them. Hashes and file times: [`N1_CODE_PROVENANCE_2026-10.md`](N1_CODE_PROVENANCE_2026-10.md).

## 14. Inferring what the data do not hold (2026-10-24/25)

After section 13 two claims were open: the 1-3-year subsidence size and the Changhua levers.
The user's direction (2026-10-24) was to try any technique that infers missing quantities
from the data already held. Inferred values may enter only as model inputs, never as the
observations a test is scored against. Four routes were tried:
- A0, an identifiability audit;
- M1, a storage-capped column;
- rung B, the measured public supply as pumping;
- N2, a prospective test.

Each route was declared before it ran and run once, and every attempt is reported.

**A0, identifiability audit** (post hoc, model-only, not a test).
- Script: `dataprep_staging/n1_gaps/identifiability.py`.
- Files: `results/dataprep/reports/N1_gaps_2026-10-24/A0/`.
- The rule was fixed before the run. A quantity is identifiable if its posterior sd/|mean|
  is at most 0.5 and its 50-replicate 90 % coverage lies in [0.85, 0.95].
- Laplace posteriors were computed on R0, Q0, R4p and R1c.

Findings:
- **Head response:** the cut30 head response is identifiable on information alone
  (sd/|mean| 1.3-3.6 % in every model). That holds only given the fixed energy-to-volume
  conversion (eta 0.5, head_extra 40 m), which sits outside the posterior.
  - The nonlinear check on R0 covers 0.68-0.82, and with correlated-noise data 0.54-0.75.
  - So the Laplace intervals are too narrow, and the calibration data do not pin the
    response size.
- **Not identifiable without new data:**
  - the pump shallow share (the posterior equals the prior);
  - proximal Ske and Skv;
  - Changhua T and S in R0 and Q0 (prior-driven);
  - Changhua T in layers 2-3 in R1c;
  - most Changhua T and S in R4p.
- **Identifiable:** Ske and Skv in the mid and distal zones.
- **Rung A not run:** none of the four models has a layer-0 recharge conductance parameter.
  That was the precondition for the post-deadline rung A (river-stage recharge), so by its
  stopping rule rung A is not run.
- **Caveat:** the declared 50-replicate coverage rule is noisy. A calibrated 90 % interval
  falls outside [0.85, 0.95] about 23 % of the time. The rule was applied exactly as
  declared, and 2000-replicate coverage is reported as a diagnostic only.

**M1, a storage-capped elastic column** (post hoc, model-only, not a test).
- Code: local branch `twin/m1-ske-cap`. The cap is opt-in
  (`calibrate_coupled --ske-cap-from-theta`), the default is bit-identical, and there are
  14 tests.
- Gate 0 was fixed before the run: synthetic recovery. The capped column (kappa 1) is
  fitted to leveling made from a known truth. It must recover the avoided subsidence within
  [0.67, 1.5] in every zone at 1, 3 and 7 years, with head damping k of 1 and 2.
- Result file: `M1/m1_synthetic_20261023T232354Z.json` (sha256 `c957d24b`).
- **Result: FAIL.** At k = 2 the mid zone recovers 0.61 at 3 years and 0.66 at 7 years.
  Every other zone and horizon passes.
- The cap removes the uncapped column's distal inflation: the 1-year ratio at k = 2 goes
  from 14.1 to 0.93. It cannot recover the mid zone's inelastic, creep part.
- As declared, M1 is not adopted, kappa is not retuned, and no capped column is fitted to
  real data.

**Rung B, the measured public-supply abstraction** (`I1.B.v1`, sha256 `a3aea7c7`).
- Registered before the run. Batch `results/twin_runs/par26`, code `76ae637` (clean).
- Input: Taiwan Water yearbook volumes (X2) as fixed deep pumping in twin layers 2-3, on Q0
  with the Changhua census.
  - B1 keeps Q0's learned layer split.
  - B2, the pre-declared alternative, fixes the measured X3 shares.
- Partition: design-time, seed 2027.
- Report: `results/dataprep/reports/I1_B/report.json` (sha256 `58546161`), written by
  `dataprep_staging/i1/score_rung_b.py`. That scorer reproduces all nine R* gate results of
  section 12 exactly (`--check rs`).
- Reading: `I1_B/reading.json`.

| gate (against Q0) | Q0 | B1 | B2 |
|---|---|---|---|
| k-fold R² ≥ 0.638 | 0.648 | **0.688** | 0.481 ✗ |
| Changhua county_corr ≥ 0.448 | 0.468 | **0.557** | 0.473 |
| Yunlin county_corr ≥ 0.720 | 0.740 | 0.751 | 0.734 |
| Changhua county-mean RMSE ≤ 0.542 m | 0.531 | 0.679 ✗ | 0.648 ✗ |
| Yunlin county-mean RMSE ≤ 1.000 m | 0.980 | 0.925 | 0.952 |
| leveling out-of-fold R² ≥ 0.633 | 0.653 | 0.085 ✗ | 0.482 ✗ |
| temporal fair ratio ≤ 1.0768, PASS | 1.066 | 1.054 | 1.034 |
| plausibility, 2023-25 rate ≥ 0 cm/yr (Changhua, rest) | +0.27, +0.17 | -0.03, -0.32 ✗ | -0.02, -0.20 ✗ |
| P2f Yunlin in [0.521, 0.954] (B2 only) | -0.007 | 0.050 (reported) | 0.900 |

- **Both arms fail.** No seed-2029 confirmation is run, so no Changhua arm can be added to
  N2 before its 2026-10-31 deadline.
- **What the measured public supply did:**
  - It improves Changhua head timing: B1's Changhua correlation is 0.557 against 0.468,
    and its k-fold is 0.688, the best of any rung.
  - It makes Changhua's heads too loud: amplitude 0.36 → 1.19 of observed.
  - The leveling column refitted on those heads loses the leveling: Changhua -0.73, rest
    of the fan 0.06.
- **What it means:** the declared interpretation ruled the term out only if Changhua's
  correlation did not move. It moved, so the term is neither confirmed nor ruled out. The
  report's own decision line overstates this ("not the missing Changhua term"); `reading.json`
  corrects it.
- **The rebound** (post hoc, model inputs only, not a test):
  - In X2, Changhua's public supply is measured, 31-35 model Mm³/yr every year from 2007 to
    2022.
  - Yunlin's is inferred. It falls from 16-19 to 1.5 in 2019 and 0.5 in 2022, and
    `twin.forward` holds that last value.
  - The rest-of-fan rebound (fan 2023-32 subsidence -0.58 cm in B1, against Q0's +2.93 cm)
    fits deep Yunlin heads still recovering from that inferred drop.
- **Unspent:** the one exploratory follow-up allowed is left unspent. It could not promote,
  and nothing it shows could reach a registration in time.

**N2, a prospective test of the head response to recorded pumping**
(`N2.policy_response.prospective_2026-11.v1`, sha256 `bfbbd6b3`, registered 2026-10-24
00:41 UTC, before its window opens).
- **Models and window:** the frozen page model is primary: `stage3_datum_gate` with its
  `coupled_leveling` column and the forward run `datum_gate_innov`. R0 is the declared
  secondary. The window is 2026-11 to 2027-10, on the 158 wells of the registered
  prospective test.
- **The recorded-pumping arm (PR):**
  - Ruling U4, as the user extended it on 2026-10-24: recorded post-origin pumping may feed
    a separate counterfactual arm, and the registered forecast (ZR) is unchanged.
  - The multiplier r(t) is the Yunlin in-grid electricity of month t (X1) over its 2012-22
    mean for that month of the year (`dataprep_staging/n2_policy/build_r.py`).
  - The contrast c = PR - ZR comes from the same forward run (`twin.forward
    --recorded-pumping`, commit `1fff145`). The PR prediction is ZR + c.
- **The metric:** beta_response, the OLS slope of obs - ZR on c over Yunlin well-months.
  It uses well fixed effects and a 5 km block bootstrap 90 % interval.
- **The verdict:**
  - "response supported": the interval excludes 0, the point estimate lies in [1/1.5, 1.5],
    and RMSE(PR) < RMSE(ZR);
  - "direction supported, size not": beta > 0, but the size or the RMSE condition fails;
  - "not supported": otherwise;
  - "not informative": the RMS of the well-demeaned contrast is below 0.05 m, or fewer than
    6 window months carry a recorded r.
- **Rehearsal** on 2023-01 to 2026-08 (an engineering check; no window value exists yet;
  `results/dataprep/reports/N2_policy/rehearsal/`):
  - 44 months of r, 42 of them flag 0;
  - the contrast is exactly 0 before 2023 and averages -0.50 m in layer 2 over 2023-26.
- **What a pass supports:** the frozen model's head response to real pumping anomalies at
  Yunlin wells, on fresh months. Nothing about the 1-3-year subsidence size, long-run
  sinking or Changhua. N2 is scored after October 2027.

**Where each policy claim stands (2026-10-25).**

| claim | status | decided by | what would close it |
|---|---|---|---|
| head response to a pumping change, Yunlin | supported on design-time masked refits (G1, step C); size not pinned by the calibration data (A0) | N2, registered, scored after 2027-10 | N2 decides |
| head response, Changhua | supported only on the census model R1c; the page model has no Changhua pumping; rung B failed | no registered test | Changhua irrigation-office monthly diversions by intake; per-well public-supply abstraction |
| 1-3-year subsidence size | not supported (steps B and C disagree in direction); M1 failed its synthetic gate | no unspent leveling decides it | a fresh WRA leveling campaign, or monthly compaction-well and GNSS series from 2026-11, registered before they are seen |
| long-run sinking under a cut | untested; in this model the benefit is all elastic (section 13) | none | the same, over several years, including a dry year |
| Changhua levers | not supported (step B, fixes F1 and F2, rungs R1-R4p, R* and B) | none before 2026-10-31 | the Changhua data above, plus pump-test energy-to-volume data |
