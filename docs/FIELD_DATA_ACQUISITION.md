# Field calibration and independent validation: acquisition route

Investigated 2026-10-07 (Taipei). This extends the repository audit and
[decision roadmap](SUBSIDENCE_DECISION_ROADMAP.md). **There is a concrete route to
existing field records; the required raw calibration tables have not yet been
obtained. No independent validation has been completed.**

## First request: WRA / NCKU calibration and Tuku trial records

The downloaded official **2025 Land-Subsidence Prevention and Reclamation** report,
project **MOEAWRA1140109**, identifies a specific data asset: **516 original records,
including 505 water-to-electricity ratio records and 11 water-metered wells**.
The ratio records include well IDs, electrical measurements, discharge and
conversion formulas. Their geographic coverage, measurement methods, dates and
overlap with our electricity census still need confirmation. The 11 metered wells
are a lead for direct volume observations, not proof of usable paired histories.

The [NCKU laboratory's well-discharge project page](https://thl.web2.ncku.edu.tw/p/405-1185-212847,c25896.php?Lang=zh-tw)
names Cheng-Wei Lin as contact and describes a Yunlin electricity-linked inventory
of 116,769 wells, matching our census row count. This supports approaching that
team for provenance and linkage; matching counts alone do not prove identical
datasets. Its published regional pumping estimates use the electricity method and
are not independent measured-volume validation targets.

The report also describes a Xiutan/Tuku pumping-schedule trial, with participating
and nonparticipating wells, six groundwater monitoring sites and GNSS TK01–TK03.
Request raw observations, actual pump-operation logs and intervention assignments.
Participation was voluntary: the groups cannot be treated as randomized controls.

**Custodian route:** WRA's Hydrology Technical Division and the NCKU project team;
ask them to route the request to the water-well management database custodian.
Use the project number rather than relying on personnel named in older news.
Evidence: [official catalog](https://web.wra.gov.tw/WraLib/wSite/books_97847),
[public report PDF](https://web.wra.gov.tw/WraLib/DocDownloadService?marc_id=97847),
printed pp. 3-29 (PDF p. 117), 2-20–2-21 (PDF pp. 62–63), 2-31–2-35.

### Avoid circular electricity calibration

WRA's [2023 trial account](https://www.wra.gov.tw/epaper/Article_Detail.aspx?n=30173&s=8753)
explicitly describes converting monitored electricity into estimated pumping.
Consequently, a field called “pumping volume” or “smart water meter” is insufficient:
we need its measurement provenance. An estimate derived from electricity cannot
serve as independent truth for another electricity-to-volume estimate.

Our catalog already lists `etc-tpc-etc1mon-obs` and `etc-ncku-etc1min-obs`.
The older design spec describes minute-scale electrical measurements as efficiency
calibration data. Electrical measurements alone cannot identify wire-to-water
efficiency: simultaneous discharge and total dynamic head are also required.
Minute power can still improve operating-time estimates and detect non-pump loads.
Catalog presence does not establish that the complete histories are downloaded.

The field table must distinguish **direct flow measurement**, **measured energy**,
**head measurement**, and **derived ratio/volume**. Request instrument calibration,
test date, operating condition and uncertainty. Voltage/current alone are not
active power without phase/power-factor information. Keep shared-meter groups
together in calibration/validation splits, and match actual billing intervals.

## Other acquisition routes

| Route | Verified evidence / access | Value and limit |
|---|---|---|
| Controlled pumping, Yuanchang Elementary School and XiHu Central Park | [Chen et al., 2026, DOI 10.1016/j.enggeo.2026.108783](https://doi.org/10.1016/j.enggeo.2026.108783); [NCKU record](https://researchoutput.ncku.edu.tw/en/publications/estimating-aquifer-separations-and-storage-coefficients-via-seque/). July–August 2023 tests with multi-depth pressure and precise leveling; raw download not located. | Ask authors/WRA for pumping schedules, discharge, screened intervals, pressure and leveling time series with uncertainties. Constrains short-term hydraulic/storage response; does not establish long-term irreversible compaction or paired electricity availability. |
| 2025 leveling update | [Official release, 2026-02-25](https://www.nlsc.gov.tw/NLSC_Content.aspx?n=11987&s=336997&sms=9680). Downloaded eight-page methodology; station-by-year attachment not obtained. | Request machine-readable heights, observation dates, datum/adjustment versions, benchmark changes and ties to GNSS. The network incorporates WRA observations, including 495 Yunlin monitoring points; it is not automatically an independent sensor dataset. Annual leveling cannot by itself validate monthly forecasts. |
| Daily GNSS and depth compaction | Existing local NCKU/WRA caches; [official GNSS observations](https://data.gov.tw/dataset/36694), [station metadata](https://data.gov.tw/dataset/36693). | Obtain latest high-precision daily solutions, QA and maintenance histories, plus ring-depth/datum mappings. These are candidates for withheld forecast targets only after exposure and source-overlap audits. |
| Irrigation Agency, Yunlin Management Office | [Official account of well tests and operations](https://www.iayli.nat.gov.tw/about/Articles?a=10984). Specific historical data availability not confirmed. | Secondary request route for completion/pumping tests, pump operation, canal deliveries and energy records. Public institutional description is not an accessible paired-data archive. |
| WRA IoW portal | [Official portal](https://iot.wra.gov.tw/) describes API services and member access. | Request/inspect dataset metadata through approved access. “Pumping” can mean drainage pumps, and volume may be inferred: verify groundwater use, geography, instrument and units before ingestion. No new account created. |

The NLSC methodology uses constraints from earlier height-network releases. Raw
differences across releases must not be treated as subsidence until the datum and
adjustment changes are reconciled. Its downloaded PDF contains no embedded files;
the release page exposed only that PDF, despite its reference to a separate table.

## Existing trial data: check before commissioning new instruments

Local file metadata inspected without scoring target values:

| Daily NCKU GNSS series | Rows | Cached dates |
|---|---:|---|
| STES | 166 | 2023-04-25 to 2024-05-14 |
| TK01 | 686 | 2023-04-25 to 2025-03-31 |
| TK02 | 664 | 2023-04-25 to 2025-03-31 |
| TK03 | 202 | 2024-08-17 to 2025-03-31 |

Files are under `chou-shui-data/data/ls_cache/ls-ncku-gnss1day-obs__*.parquet`.
The report's TK identifiers match cached series names; coordinates, datum and
instrument lineage still need reconciliation. Request their continuation and the
paired groundwater/pump logs. Sparse STES coverage is not a complete daily record.
These already-cached series cannot simply be declared untouched test data.

### Follow-up: direct API retrieval completed

Used the existing project account for read-only queries on 2026-10-07. Raw files
and SHA-256 manifests are in ignored `results/field_data_acquisition/api/`.
Existing model input caches were not replaced.

- Retrieved metadata for 475 minute-electricity devices, six NCKU groundwater
  stations, four NCKU daily GNSS stations, 1,830 leveling stations and nine deep
  benchmarks. The catalog still lists 19 datasets and no dedicated calibration table.
- GNSS TK01, TK02 and TK03 each supplied 13 additional dates, ending 2025-04-13.
  All overlapping values matched the existing cache. STES supplied no additional
  dates. This is 39 extra station-days, not a new multi-month validation set.
- All six NCKU groundwater endpoints returned empty frames. Explicit historical
  interval rechecks on three stations were also empty; this does not prove that
  the custodian lacks the field records.
- The first three electricity devices had no rows in the initial February 2025
  sample. Latest-record queries returned electrical observations for all three;
  their schema has voltage, current, power and signal quality, with no discharge.
- None of the 475 device `電表號` identifiers matched the census `電號1` after
  conservative string normalization. This is an unresolved identifier crosswalk,
  not evidence that the sites are geographically unrelated. Do not join by row order.
- API leveling metadata still ends at 2021-09-15. It cannot supply the newer
  annual leveling evidence through this route.
- Initial deep-benchmark queries hit the 20,000-row cap at eight of nine stations.
  These responses are explicitly marked potentially truncated; they must not be
  represented as complete histories. Bounded Tuku/Xiutan retrieval uses the existing
  client's recursive time-window splitting to avoid that cap.

The bounded retrieval is now complete for 2024-01-01 through 2025-05-26:

| Deep-benchmark series | Nominal installation depth | Returned rows | Non-finite readings |
|---|---:|---:|---:|
| TKJS(1) | 300 m | 69,425 | 0 |
| TKJS(2) | 130 m | 69,425 | 0 |
| TKJS(3) | 400 m | 512 | 129 |
| STES | 70 m | 68,029 | 0 |

This acquired **207,391 rows**, with no duplicate timestamps and verified query
boundaries. Rows are not independent experimental samples, and missing values
remain missing. The combined `deep_benchmark_candidate_observations.parquet`
retains raw values, source hashes and explicit unconfirmed unit/datum fields;
it marks no row as an independent holdout. `readiness.json` records the unresolved
calibration and validation requirements. No values were silently interpolated,
unit-converted or inserted into the forecasting model.

### Follow-up: monitoring provenance recovered from public reports

Downloaded the complete [2025 monitoring report](https://web.wra.gov.tw/WraLib/wSite/books_97639)
and June, August and December monthly reports. The annual report identifies a
June 2022 equipment replacement at TKJS (printed p. 3-16), a relevant lead for the
previously unexplained June 2022 jump; it supplies neither the exact correction nor
proof that the entire jump is instrumental. Its GNSS solutions use KMNM as reference
(p. 3-26). That processing lineage still needs matching to our API product.

August's table 1 documents TKJS platform gaps in February/March and July 2025.
The July gap is associated with a failed receiver memory card. The existing public
snapshot contains one record within July's reported gap: require recovery/version
provenance, rather than assuming that a later recovered record is invalid.
Saved `tuku_gnss_event_inventory.csv` and `tuku_gnss_event_contrast.json` preserve
the event dates, source page and comparison results.

Monthly logs document unstable Tuku 400 m readings in April–May 2025 and subsequent
logger replacements. Annual compaction plots use compression-positive conventions,
while monthly change tables use compression-negative conventions. Each API series
still needs a verified unit/sign/reference mapping; copying a report convention
onto a generic `value` column is not adequate.

The annual report places raw leveling, adjustment, instrument checks and station
logs in **electronic deliverable appendices 1–11 on the project disc**, rather than
inside its PDF. The PDF has no embedded attachments. Request that deliverable by
project **MOEAWRA1140115**, specifically appendices 3, 4, 6 and 8–11.

## What makes the validation independent

The following is the proposed protocol, not a completed validation result:

1. **Calibration:** use verified direct discharge paired with energy, and
   reconciled pressure/compaction measurements. Fit conversion uncertainty by
   operating condition and pump class; do not let head-fit residuals define the
   only volume calibration. Use construction/screen records to map observations.
2. **Withheld conversion test:** reserve complete meter/pump groups and later
   operating periods before fitting. Score inferred m³ against independently
   measured m³, including bias and uncertainty. Never split duplicate/shared-meter
   records across folds. Check seasonal and equipment coverage before extrapolating.
3. **Withheld forecast test:** have a data custodian sequester selected station/time
   blocks, after checking prior use and overlap with training, assimilation and
   model-selection records. Use 1- and 3-month surface-displacement forecasts first;
   add longer horizons only as observations mature. Freeze code, model, QC rules,
   sites, issue times, input availability and target definitions before release.
   Past data at an issue time may initialize the forecast; target-window data may not.
4. **Prospective confirmation:** timestamp and hash predictions before observations
   arrive; retain original input/publication versions. Evaluate displacement in mm,
   bias, MAE/RMSE, interval coverage and forecast availability against persistence,
   seasonal persistence and a past-only trend baseline on identical target support.
   Quantify uncertainty with site/time blocks, not independent daily-row assumptions.
   Fix decision tolerances from the smallest policy-relevant effect and measurement
   uncertainty before seeing test scores; do not invent a universal acceptable error.
5. **Intervention test:** obtain actual schedules, delivered water volumes, rainfall,
   canal supplies, crop demand and neighboring pumping. Check pretreatment trends,
   selection bias and groundwater interference across comparison areas. Existing
   trials support retrospective mechanism tests; their published results are
   already exposed and cannot become a blind confirmation set. A future agreed
   operating period, evaluated externally with frozen predictions, is stronger.

Changing pumping timing and reducing total volume are separate interventions.
Do not label a schedule shift a successful subsidence remedy merely because it
reduces peak drawdown. Test cumulative surface displacement, delayed compaction
and delivered irrigation water under the actual volume histories. If effects are
smaller than observational/forecast uncertainty, report the ranking as unresolved.

An independent evaluator should control withheld targets and run the frozen
scorecard. Different institutions' files are not independent when they contain the
same observations, or when they have already influenced parameter/model choices.

## Fallback if existing calibration records are insufficient

Arrange a targeted campaign with the well owners and measurement team: calibrated
discharge measurement suitable for the pipe/site, an active-energy logger, water
levels and discharge-pressure/head measurements on a synchronized clock. Record
screen interval, pump configuration, shared loads, valve changes and meter resets.
Repeat across relevant lift and seasonal conditions. Select the number of pumps
after a pilot estimates conversion variability and required decision precision;
11 or 505 historical records do not guarantee adequate local coverage.

For intervention validation, pair surface geodesy with depth compaction and
groundwater pressure; use existing stations where quality permits. A controlled
test or schedule change requires the operators' agreement and a design that
preserves their water-delivery requirements. No field operation was initiated.

## Deliverables and present status

### AquaScope follow-up: useful public data acquired

Checked 2026-10-07 Taipei. Inspected the actual collectors and downloaded archive,
then tested the public endpoints they document. AquaScope was not installed and
no existing training cache was overwritten. Source revision:
`8280f6b180addb9166ea27f35704d97df0ee49e5`; archive revision:
`9a824f79267ead7ad5e96aafb05987a6d2ee55fd`.

| Source / route | Verified acquisition | Project value and limitation |
|---|---|---|
| [AquaScope archive](https://huggingface.co/datasets/Rekin226/aquascope-gauges) | Downloaded station catalog, health/observation manifests and Taiwan rainfall bundle. Current files contain 1,265 CWA catalog stations and 168,842 rainfall rows for 59 stations. | Catalog coverage exceeds mirrored observations. Tuku `C0K390` is listed but has no mirrored series. No Taiwan groundwater, pumping-volume or deformation observations were found in this archive revision. |
| [CWA daily collector](https://github.com/Rekin226/aquascope/blob/8280f6b180addb9166ea27f35704d97df0ee49e5/aquascope/collectors/taiwan_cwa.py), [CODIS](https://codis.cwa.gov.tw/) | Direct public requests obtained 1,405 Tuku `C0K390` daily weather records, 2022-12-01 through 2026-10-05; 1,374 days extend beyond our rainfall cache. | This is existing gauge `rf19`. All 31 December 2022 rainfall values match the local cache exactly. Supports extending observed forcing and checking weather inputs, not independent subsidence validation. |
| [WRA daily groundwater collector](https://github.com/Rekin226/aquascope/blob/8280f6b180addb9166ea27f35704d97df0ee49e5/aquascope/collectors/taiwan_wra.py), [official historical portal](https://gweb.wra.gov.tw/HydroInfo/GroundWaterQuery/) | Retrieved 226 Choushui station entries and 731 daily values per well for Xiutan `09050211` and Tuku `09050321`, `09050331`, `09050341`, covering 2024–2025: 2,924 station-days. | These four existing AMP caches end 2022-12-27. The new daily records support recent groundwater/compaction response checks. They are WRA observations through another delivery route, not independent sensors. We have not acquired 2023 groundwater through this route. |
| [WRA IoT collector](https://github.com/Rekin226/aquascope/blob/8280f6b180addb9166ea27f35704d97df0ee49e5/aquascope/collectors/taiwan_wra_iot.py), [public endpoint](https://iot.wra.gov.tw/groundwaterlevel/stations) | Retrieved a 767-station snapshot, including Tuku (2), (3), (4), stamped 2026-10-07 01:00 +08:00. | Potential input for prospective forecasts. A current snapshot does not establish continuous history or solve the missing trial groundwater records. IoT identifiers differ from historical WRA identifiers; reconcile metadata before joining. |
| WRA reservoir/river, FHY and Civil IoT collectors | Source code inspected; local historical records not acquired in this follow-up. | Potential boundary-condition or surface-water context. River discharge and reservoir releases are not measured irrigation-well discharge or confirmed canal deliveries. No paired electricity/discharge calibration table located in the inspected collectors. |

Raw responses, candidate Parquet tables, exact acquisition scripts, pinned source
files and SHA-256 manifest are in ignored
`results/field_data_acquisition/aquascope/`. `assessment.json` records counts,
overlap checks and incomplete calibration/validation status. No model was fitted
to these records or forecast skill computed.

Acquisition checks and required follow-up:

- Rainfall has no duplicate station-dates. Of the 1,405 records, 26 have nonzero
  source flags and one has a missing flag. Subsequent inspection of CWA's own
  display code resolves the positive-flag meaning: see the quality follow-up below.
- Groundwater arrays contain 731 finite, non-sentinel values each. A separate
  three-day request agrees with its position in the two-year series, checking
  the collector's inferred daily timestamps. Station datum, measurement meaning,
  screened intervals and agency quality status remain to be reconciled. Total
  well depth alone does not establish a screened layer.
- The reviewed daily groundwater collector defaults to monthly means stamped
  mid-month. For forecasting, aggregate explicitly, retain coverage, and mark a
  month's value available only when its contributing observations were available.
  Enforce request boundaries locally; the source code notes that the portal can
  return dates beyond the requested end.
- Civil IoT's reviewed normalizer retains only the first observation per stream;
  it is not a complete historical-series importer. WRA's annual statistics are
  also unsuitable substitutes for monthly/daily targets.
- The codebase already uses AquaScope/Open-Meteo for ET0. Local meteorological
  measurements can check forcing bias, but rain minus ET0 is a water-balance
  proxy, not a direct measurement of groundwater recharge. Observed future rain
  cannot be supplied to a forecast and reported as operational forecast skill.

**Priority:** use the newly acquired weather and groundwater data to prepare a
recent-period, source-audited evaluation with the existing GNSS/depth-compaction
records. Fix quality rules and forecast information cutoffs before scoring;
audit exposure across worktrees before calling any historical period withheld.
The downloads are marked `independent_holdout=False`. Keep the WRA/NCKU request
for directly measured discharge paired with energy, pump-operation logs and
sensor maintenance/datum records active. AquaScope improves access to supporting
observations; it does not close those calibration or independent-validation gaps.

### Quality follow-up: provisional heads, incomplete rain and coordinate conflicts

The [WRA portal](https://gweb.wra.gov.tw/HydroInfo/GroundWaterQuery/) explicitly
labels its immediate heads and simple history charts as not manually checked.
It identifies immediate heads as absolute elevations, but does not establish the
datum version or correction history for our downloaded arrays. Therefore all
2,924 finite daily heads retain provisional quality, unknown datum version and
unknown screen intervals. Finite values are not equivalent to validation approval.

CWA's [shared display function](https://codis.cwa.gov.tw/Common/global.js)
`SpecialVal` adds a star when the associated `f` field is positive. The
[page's own language configuration](https://codis.cwa.gov.tw/StationData/global)
calls this the missing-value mark. We retain the raw numeric totals and flag
magnitudes, but mask the 26 marked totals and one unconfirmed-flag value in the
QA table. This does not assert that those 26 numbers are wholly erroneous; it
prevents incomplete totals being treated as unqualified daily rainfall. CWA's
[data description](https://codis.cwa.gov.tw/pdf_data/Readme.pdf) also documents
trace rainfall and next-day publication by noon. This is a nominal schedule,
not proof that the revision downloaded now existed at an earlier forecast origin.

Implemented `hydrophysics/twin/field_observations.py` as an offline importer:

- Preserve source hashes, raw values, source flags and verified acquisition time.
- Reject station mismatches, duplicate dates, dates outside the request, unknown
  timezones, impossible acquisition times and unexpected undated-array lengths.
- Preserve negative groundwater heads; mask missing/sentinel values. Do not
  convert trace rainfall to zero or drop uncertainty about screens and datum.
- Keep validation approval and independent-holdout flags false. A timing filter
  excludes later revisions from an earlier origin; it does not approve data quality.

The actual imports produce 1,378 numeric unmarked rainfall days, 26 masked marked
days, one masked unknown-flag day and 2,924 provisional groundwater observations.
`tuku_rainfall_qc.parquet` and `tuku_groundwater_qc.parquet` are separate ignored
artifacts. The initial raw/candidate tables remain intact. Reproduce with the
saved local `reconcile_observations.py`; seven synthetic regression tests cover
partial totals, trace/sentinel handling, source/date mismatches and revision leakage.

The downloaded [WRA well metadata](https://opendata.wra.gov.tw/api/v2/3e86faea-e94a-4a91-a870-852d73e83c3d?format=json)
gives total well depths of 134 m for Xiutan and 90/185/269 m for Tuku (2)/(3)/(4).
No screen-top/bottom fields are present. Their stated TWD97 positions agree with
the IoT WGS84 positions transformed to TWD97 within 0.6 m. This supports their
candidate identity mapping; it does not establish depth-matched hydraulic responses.

**Additional model-input issue:** comparing all 61 curated well coordinates with
the WRA metadata found eight discrepancies greater than 10 m. Seven are about
854 m; the eighth has apparently transposed WRA coordinates and remains displaced
even after swapping. For six of the seven 854 m cases, IoT coordinates corroborate
the WRA TWD97 position within 0.7 m; the seventh has no unique IoT name/county match.
Xiutan is one of the corroborated conflicts. Its curated coordinates are about
854 m from both public WRA positions. These are two delivery routes from WRA,
not independent surveys. Do not apply one global coordinate transformation:
53 of the 61 curated positions already match the WRA table within 10 m.

`well_metadata_reconciliation.csv`, `curated_coordinate_check.csv` and
`curated_iot_coordinate_check.csv` record these comparisons locally. No live
coordinates were overwritten. Before a spatial recalibration, reconcile original
survey records and recompute affected station-to-cell and rainfall assignments;
compare forecasts on identical targets. This is a concrete input-quality issue
worth resolving, but no accuracy improvement from correcting it has been measured.

### Coordinate impact and weather-input reliability

The loader takes operator coordinates from `gray_box_input.csv`; ET0 interpolation
also reads `gw_stations.csv`. The discrepant Xiutan location appears in both.
The twin's groundwater observation mapping instead uses `fan_stations.parquet`.
Therefore a discrepancy in the curated table does not by itself prove the twin's
head observations were mapped to the wrong cell.

Prepared an isolated coordinate proposal for the six discrepancies corroborated
by the WRA metadata and IoT positions. Kept the unmatched and conflicting eighth
station unchanged. The diagnostic held cached ET0 values fixed and moved only
their interpolation positions, using the existing interpolation code over 2012–2022:

| Grid | Mean absolute ET0 field change | Maximum absolute ET0 field change |
|---|---:|---:|
| 1 km | 0.000843 mm/day | 0.07736 mm/day |
| 4 km | 0.000861 mm/day | 0.06857 mm/day |

All six proposed locations fall in different 1 km cells; one does at 4 km.
One nearest-rain-gauge identity changes. These are coordinate diagnostics, not
changes to the existing learned rain pairings or active head-observation mapping.
The small mean ET0 relocation effect is insufficient evidence to prioritize it
over measured pumping and deformation validation. Actual weather re-sampling,
static operator features and spatial target matching were not tested here.
The historical ET0 cache has no coordinate metadata, so its sampling locations
cannot be verified from the cache. Do not attach corrected coordinates to those
old values and call them newly sampled weather. Original datasets remain unchanged;
the proposal, input hashes, script and results are saved in
`results/field_data_acquisition/aquascope/coordinate_impact.json` and adjacent files.

Tracing that pathway found and repaired additional defects in `hydrophysics/et.py`:
downloaded values now align by returned dates; missing dates keep their positions.
A complete source outage raises rather than producing zero ET0. New caches retain
the mask of spatially imputed values. Matching caches with invalid shape, missing
or negative ET0 raise. The rain-minus-ET0 transform rejects missing forcing and
shape mismatches rather than silently treating them as zeros. Callers must repair
gaps explicitly and retain that provenance. This does not remove all downstream
models' historical missing-input policies, nor validate legacy cache locations.

Verification: 32 tests passed across ET0, field import, user-supplied data and
forecast smoke tests, with two existing empty-score warnings. No network was used
in those tests. Lint and whitespace checks passed. No new subsidence skill or
intervention-benefit result was produced.

- Downloaded WRA's full 2025 report and NLSC's 2025 methodology, with source URLs,
  retrieval timestamps and SHA-256 hashes in ignored `results/field_data_acquisition/`.
- Saved local trial-series coverage in `local_trial_metadata.json` in that directory.
- Prepared [specific, unsent data requests](UNSENT_FIELD_DATA_REQUESTS.md).
- A supplementary MOENV PDF URL returned HTML instead of a PDF; it is recorded as
  an unsuccessful document acquisition and is not used as acquired evidence.
- No agency/author contacted, raw restricted calibration table acquired, model
  fitted, validation score produced, or intervention benefit claimed in this task.

**Recommended next dependency:** obtain the WRA/NCKU calibration table, measurement
provenance and trial-operation linkage first. In parallel, arrange withheld future
GNSS/compaction targets and resolve datum/screen metadata. This directly addresses
the missing volume calibration and intervention evidence without assuming that a
new model architecture or an entirely new monitoring network is necessary.

## Completion audit and external dependency

Rechecked against downloaded records and current source code on 2026-10-07.
The acquisition work and software repairs are complete to the extent described
above. The requested decision-ready platform and independent field validation
are **not complete**. The repeated blocking dependency is access to custodian-held
measurements, linkage and validation records, rather than another software package.

| Required result | Evidence currently held | Missing evidence / next owner action |
|---|---|---|
| Electricity-to-volume calibration | Monthly census energy; minute electrical samples; report documenting 505 ratio records and 11 metered wells | WRA/NCKU: raw simultaneous measured flow/volume and active energy, test conditions, uncertainty and local applicability. Electrical channels alone do not supply the target. |
| Historical pump/observation linkage | 475 device identifiers, monthly electricity census, trial station lists | Custodian: dated device–meter–well crosswalk and shared-load information; zero exact identifier matches prevent a defensible join. |
| Depth-matched pressure and deformation | Four recent daily groundwater series, depth-benchmark observations and monitoring reports | WRA/project team: actual screen intervals, instrument units/sign/datum, processing versions, maintenance corrections and usable-quality intervals. Total depths do not close this gate. |
| Independent subsidence validation | Existing and newly retrieved GNSS/compaction records, with known overlap and quality concerns | Data custodian/independent evaluator: fixed targets kept out of development, source lineage, frozen prediction protocol and subsequent scoring. No eligible independent holdout has been established. |
| Demonstrated improved near-term forecasts | Repaired code; focused and earlier full-suite software checks; prior baseline comparisons | New calibration and a passed, independently evaluated subsidence scorecard. Software tests and newly downloaded rows are insufficient evidence. |
| Defensible mitigation ranking | Existing simulator and conditional scenario mechanisms; descriptions of historical trials | Operators/project team: actual intervention timing/volumes, water substitution and comparison data; then validated model response and uncertainty. No winning action is established. |
| Decision acceptance | Near-term subsidence/mitigation objective and proposed horizons | Decision owner: actionable locations, acceptable errors relative to intervention effects, water-delivery constraints and costs before final acceptance. |

The first concrete handoff is [request 1](UNSENT_FIELD_DATA_REQUESTS.md) to the
[published NCKU well-discharge project contact](https://thl.web2.ncku.edu.tw/p/405-1185-212847,c25896.php?Lang=zh-tw),
with WRA routing by **MOEAWRA1140109** and the monitoring appendices under
**MOEAWRA1140115**. It is ready for the project owner's institutional submission.
The controlled-pumping-study and NLSC requests are also prepared. No external
message was sent. A subsequent connector check confirmed that Gmail and Drive
are connected. Existing project correspondence provides a route to collaborators;
external outreach has not been explicitly authorized.

Resume field calibration when the requested records or approved access arrive;
first verify measurement provenance and identifier linkage, then calibrate with
pumps/periods held out, and register the independent deformation evaluation before
selecting or ranking scenarios. If the original records cannot be released, ask
the custodian to run the fixed evaluation on their side or organize the targeted
measurement campaign described above. Do not manufacture an independent target
from the same energy estimate or relabel already-used observations as untouched.

### Additional parameter evidence recovered on 2026-10-07

A 58-page May 2022 conceptual-model workshop PDF was recovered from an existing
project email attachment. Its shared-folder link returned not-found through the
connected Drive account; this does not establish whether the folder was deleted
or access has changed. The attachment and correspondence provenance are retained
only in the ignored local acquisition directory.

The workshop identifies well-construction and maintenance reports, original
pumping-test drawdown/recovery records, geological logs, subsidence-well
installation test appendices, and the 2021 deep-compression investigation as
parameter sources (pages 8–17, 21–24, 27–38). These are actionable acquisition
targets, not a newly obtained calibration dataset. Team methods differ in their
use of screen length versus aquifer thickness to convert transmissivity to
conductivity, and in conductivity units (m/day versus m/min). Preserve original
transmissivity, thickness definition, units and test method before conversion.
Distinguish specific yield, specific storage and skeletal compressibility.

The public [NCKU investigator profile](https://thl.web2.ncku.edu.tw/p/406-1185-212998%2Cr3124.php?Lang=zh-tw)
independently confirms leadership of the 2021, 2022 and 2023 compression-parameter
projects. Request their raw tests and depth-resolved parameter tables through the
existing project collaborators or the published contact. Numerical ranges and
model diagrams in a workshop presentation are prior information; they do not
demonstrate a calibrated local model, independent validation or improved forecasts.
The prepared requests now include this additional source family.

The [2023 report catalog](https://web.wra.gov.tw/WraLib/wSite/books_95637)
was subsequently retrieved directly: **MOEAWRA1120106**, ISBN 9789865334253,
GPN 1011201920. Its abstract describes 400 m geological sampling at Ke-Tsuo
Elementary School in Yuanchang, laboratory hydraulic/compression tests, and
layer-specific compression calculations under groundwater-head scenarios.
This strengthens the case for requesting existing measurements before commissioning
new field tests. It does not establish transferability to Tuku or the accuracy
of those scenario calculations. The catalog lists physical holdings including
the Water Resources Planning Branch library (FC036258/FC036259) and WRA's main
library (AC015329). The page has no full-text download button; a direct public
document-service check returned an empty response. Raw test tables and the full
report remain unacquired. The catalog HTML is retained locally for provenance.
