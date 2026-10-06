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
