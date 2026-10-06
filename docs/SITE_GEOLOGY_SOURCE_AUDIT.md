# Site geology source audit — 2026-10-06

The original project checkout has station catalogs, head/ring series, model layers and plans, but no station-linked core log, completion diagram or screen-top/bottom table. The scoped `chou-shui-data/data`, `data_fetch_api/new_measurements`, `AMP_V2/data`, and `hydrophysics/sample_data` trees were checked. The existing ring manifest supplies measured depths and TWD97/EPSG:3826 locations, not soil contacts.

## Official sources obtained

- [WRA 2003 installation report, catalog 41979](https://web.wra.gov.tw/wralib/WraLib/wSite/books_41979), PDF button route `https://web.wra.gov.tw/wralib/WraLib/DocDownloadService?marc_id=41979`. The 108-page PDF is saved only in ignored `results/decision_evidence/public_sources/wra_2003_ring_installation_41979.pdf`, SHA-256 `648330a12f831f96fc426110a42ed5a4c7930230c4a01c3f8d756377d25bce22`. PDF pp. 62–63 show the 土庫國中 well log and Table 3-1-21: aquifer 1 0–34 m, confining unit 1 34–51 m, aquifer 2 51–171 m, confining unit 2 171–193 m, aquifer 3 193–246 m, confining unit 3 246–260 m, aquifer 4 below 260 m. It distinguishes well-log interpretation, core interpretation and soil classification; these boundaries are an interpreted site model. PDF p. 67 Table 3-1-22 reports 26 installed rings, 9.85–302.01 m, dated 2003-09-18. The report has site-specific soil laboratory results mostly from the upper 100 m and a comparison with 元長國小, but not an interchangeable 元長 installation log.
- [MOT Institute of Transportation 2017 report B1068760](https://www.iot.gov.tw/uploads/asset/data/661982b7367376304acd4ed0/B1068760.pdf), saved in ignored `results/decision_evidence/public_sources/iot_tuku_B1068760.pdf`, SHA-256 `d2bd83cbdc1796cd6361ee99bafd039dc897dfc72b670673cd287dc2d2f8d9bf`. PDF p. 81 Table 4-1 explicitly places the 土庫國中 subsidence and pressure wells at TWD97 E 187771, N 2620610 m, exactly the ring catalog location. It reports 300 m well depth, pressure sensors at 90/185/269 m, 25 listed sensor rings, and PDF p. 83 Table 4-4 gives 47 soil-class intervals from 0 to 300 m. It does not give groundwater observation-well screen tops/bottoms. The original page image itself labels interval 39 as 49. A depth-sorted extraction is saved only in ignored `results/decision_evidence/public_sources/tuku_soil_intervals_2017.csv`, preserving the printed label alongside the sequence-by-depth. A bounded check found 47 non-overlapping, contiguous intervals from 0 to 300 m and the single printed-label anomaly; derived CSV SHA-256 `9065619ab03a3972351ba181facf869daf18a785d8b7277ee296f6e2df4102f7`. This is a checked source transcription, not calibrated material properties.
- [Geological Survey borehole index, dataset 16799](https://data.gov.tw/dataset/16799), direct official [CSV](https://www.gsmma.gov.tw/uploads/17503844515568fWuXe2C.csv), 387 records, saved in ignored `results/decision_evidence/public_sources/gsmma_boreholes_16799.csv`, SHA-256 `0311afdf29164a31c62dc61098f539791e6b2673d1ec6c34cb2e31129075b8fc`. It gives explicit TWD97 coordinates and IDs, but the downloadable CSV contains only basic borehole metadata even though the portal description also mentions core and well-logging records. Historic `091702G1` 元長 is E 179638.53, N 2616686.92, 200 m deep and 193 m from the ring site. Historic `090551G1` 土庫 is E 187956.33, N 2619620.44, 200 m deep and 1007 m from the 土庫國中 ring site. These are separate borings. The [1999 Geological Survey appendix](https://twgeoref.gsmma.gov.tw/GipOpenWeb/wSite/public/Attachment/1999/19990238/0001.pdf) has lithology for those older borings, but its printed `TM` datum is unspecified; use the later explicit-TWD97 index for separation, not an assumed conversion.
- [WRA well-status dataset 32718](https://data.gov.tw/dataset/32718), cached at `data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet`, places 土庫(2/3/4), IDs ending `09050321/31/41`, at E 187777.40, N 2620597.81 m, depths 90/185/269 m. This agrees with the report sensor-depth sequence but represents later 2012 wells and only a 14 m spatial match; no screen top/bottom is published in the field list. Its TWD67 string duplicates the TWD97 easting, a metadata anomaly. 元長 catalog well `09170241` has coordinate axes reversed in its TWD97 string. Quarantine suspect rows from automatic spatial joins.
- [WRA 元長 construction report catalog 22847](https://web.wra.gov.tw/wralib/WraLib/wSite/books_22847), GPN 1009200996, confirms a ring well installed at 元長 but exposes physical holdings and no PDF download button on the examined page. The full exact-site log and initial ring table were not obtained.

## Current ring sites versus Geological Survey index

Distances use current ring-manifest and official-index TWD97 coordinates. They screen possible regional context, **not** borehole identity or valid transfer of contacts.

| Ring site | Ring ID | Nearest indexed historic boring | Distance | Site-linked installation geology |
|---|---|---|---:|---|
| 僑義國小 | `A130600GWMWQYES` | `072602G1` 溪州 | 2566 m | No exact ring-boring link |
| 元長國小 | `A130600GWMWYCES` | `091702G1` 元長 | 193 m | No exact ring-boring link |
| 光復國小 | `A130600GWMWGFES` | `090402G1` 九隆 | 3102 m | No exact ring-boring link |
| 內寮派駐站 | `A130600GWMWNLPS` | `091701G1` 崙子 | 37 m | No exact ring-boring link |
| 北辰國小 | `A130600GWMWBCES` | `090601G1` 北港 | 338 m | No exact ring-boring link |
| 嘉興國小 | `A130600GWMWJSES` | `090801G1` 嘉興 | 18 m | No exact ring-boring link |
| 土庫國中 | `A130600GWMWTKJS` | `090551G1` 土庫 | 1007 m | Exact-site WRA/IOT report |
| 宏崙國小 | `A130600GWMWHLES` | `090501G1` 宏崙 | 8 m | No exact ring-boring link |
| 客厝國小 | `A130600GWMWKTES` | `091701G1` 崙子 | 2972 m | No exact ring-boring link |
| 拯民國小 | `A130600GWMWJMES` | `090301G1` 虎尾 | 2546 m | No exact ring-boring link |
| 新生國小 | `A130600GWMWSSES` | `072001G1` 趙甲 | 128 m | No exact ring-boring link |
| 湖南國小 | `A130600GWMWHNES` | `071101G1` 溪湖 | 18 m | No exact ring-boring link |
| 溪州國小 | `A130600GWMWHCES` | `072602G1` 溪州 | 439 m | No exact ring-boring link |
| 秀潭國小 | `A130600GWMWSTES` | `090501G1` 宏崙 | 3068 m | No exact ring-boring link |
| 仁和國小 | `A130600GWMWRHES` | `090551G1` 土庫 | 3625 m | No exact ring-boring link |
| 舊庄國小 | `A130600GWMWJJES` | `090802G1` 舊庄 | 28 m | No exact ring-boring link |

## Measurement reconciliation and remaining requests

The current 土庫 ring manifest first survey (2010-01-19) has `NO1`–`NO24` at 8.773–294.374 m; it omits the 2003 installation rings 25 and 26 at 297.04 and 302.01 m. The first 24 2010 depths follow installed order, but differ by -1.077 to -0.067 m, so use dated surveyed depths and verify ring-ID continuity and reference before mapping intervals. The 2017 report lists 25 rings, another inventory-date difference. No screen intervals, full-column site-specific stress-to-strain calibration, or post-2010 ring configuration history has been verified. Do not score the existing scalar total-column archive against a selected ring interval.

For 土庫, independently review the checked 47-interval transcription and link the 24 usable ring IDs and pressure sensor/well identifiers with dates and depth references, then test the observation operator on synthetic profiles before any field claim. For 元長, request or obtain the full WRA GPN 1009200996 installation report and station log for ring ID `A130600GWMWYCES`; keep Geological Survey borehole `091702G1` separate. For the other 14 ring IDs in the table, request the WRA installation/completion package keyed by exact ring ID: dated site coordinates and CRS, lithologic core and geophysical logs, ring placement/maintenance history, pressure-well screened intervals, and vertical datum. No agency request was sent.

## Bounded 土庫 geometry implementation

The public tables are versioned under `hydrophysics/twin/data/` with original printed soil-layer labels and a separate sequence-by-depth column. `tuku_geometry_sources.json` records the exact public report URLs, source-PDF SHA-256 hashes, table page numbers, station ID, TWD97 coordinates and CSV hashes. `site_geometry.py` verifies those hashes and the 47 contiguous 0–300 m intervals, then returns *geometric overlap lengths* only. It does not turn soil classes into strain or allocate the archived total-column compaction. The official WRA catalog page exposes a PDF button whose actual `onclick` targets `DocDownloadService?marc_id=41979`; that exact public response had PDF content type, 108 readable pages and the recorded SHA-256. The IOT report is a later reproduction/analysis, not the original 2003 installation record.

The original WRA table has 26 rings dated 2003-09-18; IOT Table 4-2 lists 25 at a later report date; the local WRA cache contains exactly `NO1`–`NO24` across 143 surveys from 2010-01-19 to 2025-03-12, with 20–24 nonmissing per survey. The first survey has 24 depths. Its 23 adjacent intervals all lie inside the 0–300 m soil table, and each soil/hydro overlap sum equals its measured interval thickness. These exploratory results are in ignored `results/decision_evidence/tuku_geometry_check.json`. Its 2010 depths are 0.067–1.077 m shallower than the corresponding *ordinal* 2003 installation depths. Number matching alone does not establish unchanged physical ring identity, datum or maintenance history, so the code keeps the 2003 installation and 2010 observed-depth versions separate. The 26th installed ring at 302.01 m lies beyond the soil table and is rejected. A mentioned 27-ring reference has not been verified as a dated 土庫 configuration; the attempted public MOEA guidance download returned HTTP 403, so 27 is not used.

The 2003 [WRA report](https://web.wra.gov.tw/wralib/WraLib/wSite/books_41979) Table 3-1-10 (PDF p. 48) publishes ten one-dimensional consolidation tests from sampled depths 9.0–50.8 m: initial void ratio 0.508–1.048, compression index `Cc` 0.102–0.379, rebound index `Cs` 0.012–0.057 and preconsolidation stress 1.54–5.50 kg/cm². Its narrative range excludes the largest `Cc`/`Cs` pair, so any prior must use the sample rows and an explicit outlier rule. These are genuine site samples, potentially useful as *shallow laboratory constraints* after unit conversion and stress-history checks. They do not identify strain coefficients throughout 300 m, delayed drainage times, in-situ effective stress or response to a particular pumping scenario. Source versions also disagree locally: WRA labels samples at 48.5–50.8 m `ML`, within a confining unit ending at 51 m, while IOT Table 4-4 labels 48.3–85 m `SP~GP`. Resolve that contact with the original core/log and sampling record before using either as an exact material map.

**Next one-site prototype:** first verify the ring-number/maintenance crosswalk and the 48–51 m contact; then pair dated pressure observations with an explicitly referenced, depth-resolved strain model and test synthetic interval sums. Freeze choices before scoring later 土庫 ring surveys. The current geometry check is small: 1,000 table-load plus 24-overlap cycles took 2.125 s on one CPU in the existing environment, or about 2 ms per cycle. A physical fit has no measured runtime estimate and is not started; it needs an agreed constitutive law, stress history and reference conventions. No regional extrapolation or field-skill claim follows from this one site.

## Final ring and material-source reconciliation

The additional existing local `data_fetch_api/new_measurements/compaction/土庫國中.parquet` (SHA-256 `3bf5d91c7a78afaa6b5c3a70a343982bbc6783d3c7fc9e867ada0c7ce64a3f4a`) has 218 survey dates from 2003-12-03 to 2025-05-07 and only `NO1`–`NO24`. This source is closer to installation than the old clean cache. It has no 2013 or 2022 surveys and only two in 2012; the older 2010–2025 cache also has only 24 columns. On 2003-12-03, WRA 2003 Table 3-1-22 installed depth minus surveyed depth is 0.064–0.092 m (median 0.078 m) for `NO2`–`NO24`, but 1.075 m for `NO1`. Visual inspection confirms the WRA PDF prints 9.85 m for installed `NO1`; the local first survey gives 8.775 m. That discrepancy cannot be attributed to 2003–2010 compaction. A table error, ring-specific relocation, or a reference/measurement change remain possible. `installed_ring_overlap` therefore refuses intervals beginning at original `NO1`; the printed value stays preserved in the source table.

The later IOT [report](https://www.iot.gov.tw/uploads/asset/data/661982b7367376304acd4ed0/B1068760.pdf) Table 4-5 (PDF p. 85) reproduces WRA 2013 measurements through `NO25`, and Table 4-6 on the same page reports independent May/November 2015 measurements of `NO25` at 296.3655 and 296.3766 m. Thus `NO25` was observable after installation; its absence from the local 24-column API extract does **not** establish sensor failure. No post-installation `NO26` measurement or reason for the API omission was found. The report gives no maintenance/replacement crosswalk for this gap. The 27-ring mention in separate measurement guidance remains unverified for 土庫.

For the 24 shared ring numbers, comparing IOT 2015-05-21 to local 2015-05-15 (six days apart) gives a median local-minus-published depth of 0.0636 m; comparing 2015-11-03 to local 2015-11-10 (seven days apart) gives 0.0625 m. Almost all rings share that offset, consistent with the same ordering under different zero/reference procedures, but neither report documents a transformation and the dates/instruments differ. Two printed IOT values are exceptional: May `NO16` and November `NO2` depart from their respective median offset by more than 0.02 m. Preserve the raw values; do not force-align or correct them. The agreement supports a provisional `NO2`–`NO24` numbering match for *geometry review*, not a proven unchanged installation configuration.

The 48–51 m lithology conflict is also real in the printed sources. WRA 2003 Table 3-1-21 (PDF p. 63) interprets 34–51 m as a confining unit, and Table 3-1-10 (PDF p. 48) classifies samples 48.5–49.3 and 50.0–50.8 m as `ML`. IOT Table 4-4 (PDF p. 83) classifies 48.3–85 m as `SP~GP`. The reports do not state that the later table revises the sampled core, and the IOT table also has an unrelated printed layer-number typo. Keep the two classifications as alternative measured/derived interpretations; do not assign a single material property across this boundary until the original lithologic log, sample positions and later table provenance are reconciled.

**Inputs still required for a physical fit:** a dated WRA ring installation/maintenance and API field crosswalk for `NO1`, `NO25`, `NO26`; vertical survey reference definitions; the original core/geophysical log and sampling record near 48–51 m; screened intervals and datum for the pressure wells; deeper consolidation and hydraulic properties or an explicit independently justified constitutive prior; and a frozen held-out ring/pressure evaluation protocol. The completed code tests overlap geometry and excludes unsupported intervals. It supplies no strain predictions or calibrated full-column parameters.

## One-cell structural sensitivity, without a compaction fit

The bounded command below reads only the two existing forward archives and prints JSON;
it does not start a flow run, fit material parameters, or change the source checkout:

```bash
python -m hydrophysics.twin.tuku_sensitivity --forward \
  results/twin_forward/datum_gate.npz
python -m hydrophysics.twin.tuku_sensitivity --forward \
  results/twin_forward/physical_spread_apex.npz
```

The analysis locates the exact 土庫 site coordinate in the 1 km model grid (active
cell 541) and reads the four saved aquifer-head differences at 2032-12-01. The
selected archive SHA-256 is recorded in ignored
`results/decision_evidence/tuku_datum_sensitivity.json`; the previous archive
SHA-256 is `17a16b8cd6ce9266d9726c6a7ea69622ad4fd11f29351c501be90c8aff366ff8`,
recorded with full values in ignored
`results/decision_evidence/tuku_spread_sensitivity.json`.
The comparison is a **normalized weighted head rise in metres**, not strain,
settlement, subsidence rate, or probability. The WRA support counts only its three
interpreted confining units (53 m); the IOT support counts its CL and ML intervals
(132.8 m). Both assume equal unknown compliance per included metre; a confining
interval gets the average of adjacent model aquifer heads. These stress-driver and
compliance assumptions have not been established at the site. The two sources
disagree at 48–51 m and are kept separate.

| Archive, December 2032 | Cut30 − retire, by aquifer (m head) | WRA support: cut / retire (m proxy) | IOT support: cut / retire (m proxy) |
|---|---|---:|---:|
| Selected datum | +0.135, +1.048, +0.667, +0.557 | 1.828 / 1.120 | 1.832 / 1.154 |
| Previous spread | −0.385, +0.692, +0.350, +0.025 | 0.726 / 0.411 | 0.695 / 0.421 |

In the selected archive, cut30 gives more head rise than retire_aqua in every
aquifer at this cell, so **any nonnegative four-layer head weighting** ranks it
higher on that narrow head metric. In the previous archive, the shallow-layer
difference is negative and the other three are positive, so an unmeasured
shallow-heavy compliance weighting reverses the head-proxy ranking. Neither
support is a calibrated compliance distribution. More importantly, the saved
*whole-column subsidence* comparison in `docs/DECISION_READINESS.md` ranks
retire_aqua higher for 2032 cumulative fan-average avoided displacement in both
archives. This is a different quantity and spatial aggregation; the one-cell
head ranking cannot replace it or validate its mechanism. Neither comparison
ranks verified late *rate* benefits or observed interventions.

A depth-resolved constitutive fit is still blocked by pressure screen/datum
metadata, ring maintenance/reference crosswalk, deeper compressibility and
stress-history information. It would require independently justified drainage,
effective-stress and time-response assumptions, then a prospective depth-matched
ring/surface test. Assigning strain from the present head proxy would fabricate
those missing inputs.

## Independent surface-height candidate found after the geometry check

The official WRA [GNSS measurement dataset](https://data.gov.tw/dataset/36694)
lists 土庫國中 among its continuous surface-height stations; the linked
[station metadata dataset](https://data.gov.tw/dataset/36693) identifies
`A130600LSTKJS` at TWD97 E 187770.14, N 2620603.69 m, about 6.37 m from
the ring coordinate. This is a *separate instrument and ID* from ring
`A130600GWMWTKJS`. The public metadata says GNSS measures one point,
not regional surface subsidence. The official measurement dataset is published
annually; its current post-March-2025 rows were not independently fetched here.

An existing original-checkout cache
`chou-shui-data/data/ls_cache/ls-wra-gnss-obs__TKJS.parquet` (SHA-256
`561f1d53ff27e8ec14451ce6357961aa6dd97b4accf1b30975c20fb21dcc3e16`)
has 1,690 non-null daily heights from 2020-01-01 to 2025-03-31, no duplicate
dates; 2025 has 56 observed days through March. The cache was only read.
It does not supply post-May-2025 prospective surface evidence. The twin
package has no direct reference to this cache, but absence of a code reference
does not prove it was excluded from human model selection or the broader
leveling data. Before even a *retrospective* surface check, audit vertical
datum, GNSS processing offsets and outliers, antenna/site maintenance, frozen
model selection date, and whether a surface displacement operator matches
the saved modeled total column. A single station cannot validate a regional
map or a pumping-policy effect.
