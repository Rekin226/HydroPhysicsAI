# Existing depth and deformation data: Choushui fan

**Scope:** project data under `chou-shui-data/data/` and
`data_fetch_api/new_measurements/`, plus the existing twin inputs and saved outputs.
Counts below describe files already on the authorized host, not a fresh download.
Raw observations stay in ignored project data paths; this inventory contains only
aggregate coverage and public dataset identifiers.

| Source | Local coverage and geometry | Provenance and limitation |
|---|---|---|
| Magnetic-ring positions | `chou-shui-data/data/ls_cache/clean/`: 14 nonempty Choushui sites, 20–31 rings per site. At the first survey, shallow rings span 1.351–28.921 m and deepest rings 280.339–340.464 m. Records start in 2010 and extend at most to March 2025. | `ls-wra-mlcw-obs` from WRA; [dataset 41572](https://data.gov.tw/dataset/41572) describes depths below ground and metre units. `hydrophysics/subsidence.py` treats `NO1..NO31` values as ring positions in metres. These are measured ring locations, **not** lithology or aquifer boundaries. Each interval is local to its own well. |
| Later magnetic-ring downloads | `chou-shui-data/data/ls_cache/` has 38 downloaded station files, 16 nonempty; its other parquet file is a station catalog. The two sites absent from the old clean cache have a fixed NO1–NO26 interval near 1.4–292 m at their first survey. Their evaluation window contains 26 monthly surveys, February 2023–May 2025. | Already examined in the retrospective protocol at `results/decision_evidence/compaction_protocol.json`. The downloaded files have additional earlier dates at one site. They are not prospective validation or a new source of unseen field skill. |
| Ring-station locations | `data_fetch_api/new_measurements/ls-wra-mlcw-obs-stations.parquet`: 62 stations overall, 38 tagged groundwater zone 50 (Choushui), with TWD97/EPSG:3826 coordinates and installation dates. `chou-shui-data/data/mlcw_stations.csv` locates the 14 old sites. | [WRA station dataset 41571](https://data.gov.tw/dataset/41571). No layer contacts or ring installation litholog in these catalogs. |
| Groundwater-observation wells | `data_fetch_api/new_measurements/gw-wra-gw10min-obs-stations.parquet`: 1,000 stations overall; 344 in zone 50, 343 with `WellDepth`, and layer codes 1–4 at 316 of them. All 14 old ring sites have a catalog well within 1.5 km. | Fields include `WellDepth`, `FinishDepth`, `GroundwaterLayerCode`, ground elevation and TWD97 location. Well depth and aquifer code locate head observations approximately; they do **not** define screen top/bottom, aquitard contacts or a cell-specific strain profile. Depth units need confirmation from the WRA well data dictionary. |
| Deep benchmarks | `data_fetch_api/new_measurements/ls-wra-dbm-obs-stations.parquet`: nine listed stations, eight with nominal installation depths 70–400 m; nested stations include depths 70/300 m and 130/300/400 m. `chou-shui-data/data/ls_cache/` holds time series extending into March 2025. | Official [station metadata 58335](https://data.gov.tw/dataset/58335) says the instrument measures compression from ground surface to installation depth; its depth field is in metres. Official [monitoring metadata 58339](https://data.gov.tw/dataset/58339) describes the same physical interval and warns that one station does not represent a region. Its published field list does **not** state the displacement value's unit, sign or reference datum. The cached generic `value` has not been reconciled to that dictionary. Quarantine it from compaction scoring until these are verified. |
| Surface geodesy | `chou-shui-data/data/ls_cache/`: 26 nonempty GNSS series, generally 2020–March 2025; `ls-wra-lsp-obs__choushui_panel.parquet` has 13,997 elevation records at 1,239 benchmarks, 1997–September 2021. | WRA API IDs `ls-wra-gnss-obs` and `ls-wra-lsp-obs`. Surface displacement needs datum, stable reference, trend and uncertainty handling. The leveling panel does not provide a post-2021 intervention check. |
| Model output | `results/twin_forward/*.npz` has four aquifer-head layers but only `(scenario, cell, month)` total-column compaction. | The fitted VEP column reduces heads to one driver per cell and has no vertical strain state. Ring geometry cannot be matched to its saved total without a new identified vertical model. |

## Consequence for the next implementation

Observation geometry is **partly available and previously underused**.
`python -m hydrophysics.twin.ring_geometry` now writes a hash-checked, ignored local
manifest of each ring source, its first usable survey's ordered depths below ground,
TWD97 coordinates and paired model cell. It checks monotonicity and missingness.
The existing local sources yielded 16 sites in 16 active grid cells, with 20–31 rings
per site; the output stays in `results/decision_evidence/ring_geometry.json` and is
not committed. This needs no new model run or fit. The existing
`observation_operator.ring_interval` can be exercised on synthetic profiles with
those boundaries, but cannot score the current scalar forward archive.

Run from the isolated project worktree, substituting the existing data-root paths:

```bash
python -m hydrophysics.twin.ring_geometry \
  --cache-dir DATA_ROOT/ls_cache/clean \
  --stations-csv DATA_ROOT/mlcw_stations.csv \
  --forward EXISTING_FORWARD.npz \
  --protocol results/decision_evidence/compaction_protocol.json \
  --out results/decision_evidence/ring_geometry.json
```

Site-linked lithology/aquitard contacts are now documented at 土庫國中 in the official WRA 2003 installation report and 2017 MOT summary; see `docs/SITE_GEOLOGY_SOURCE_AUDIT.md`. For the remaining sites, and for any model-wide use, the limiting inputs are verified station-linked contacts and compressible thickness, a screen-interval map, ring-maintenance history, and a measured relation between depth-specific stress and strain. The original twin design explicitly deferred the
geometry source in `docs/superpowers/specs/2026-08-22-choushui-differentiable-twin-design.md`
§11. Search or request the WRA/Geological Survey borehole completion and lithology
records using the station identifiers and coordinates already in the catalogs; useful
queries are `濁水溪沖積扇 地下水觀測井 井柱狀圖`, `濁水溪 水文地質 鑽井 岩心 地層柱狀圖`,
`水利署 地層下陷監測井 磁環 安裝深度 地質剖面`, and
`Choushui alluvial fan borehole litholog aquitard contact`.

An official [1999 Geological Survey fan appendix](https://twgeoref.gsmma.gov.tw/GipOpenWeb/wSite/ct?ctNode=217&mp=6&xItem=117602)
was found and downloaded on 2026-10-06 into ignored
`results/decision_evidence/public_sources/`. Its 180-page scanned PDF (SHA-256
`bb4460c446a3e780a82ddcfa5dbaf4cbe16f39eb89e1560cd634a67241e853e5`)
contains named historic core/geophysical logs, including 土庫 (PDF page 108) and
元長 (PDF page 105). The logs measure lithology at their own borings. Their
identifiers do not match the current observation-well catalog, and their coordinate
field says only `TM`; the datum has not been established. Thus none of those contacts
has yet been assigned to a ring station, screen or model cell. The WRA
[well-construction report catalog](https://web.wra.gov.tw/wralib/WraLib/wSite/books_39283)
has physical copies. A separate WRA 2003 installation report is publicly downloadable for the current 土庫國中 ring station, as detailed in `docs/SITE_GEOLOGY_SOURCE_AUDIT.md`.

### Historic boring and ring-site reconciliation

The same [1999 official appendix, PDF pages 105, 108 and 148–151](https://twgeoref.gsmma.gov.tw/GipOpenWeb/wSite/public/Attachment/1999/19990238/0001.pdf) identifies the historic 元長 boring at E 178806.50, N 2616895.947 m and 土庫 boring at E 187124.529, N 2619830.231 m. A separate table lists a historic observation station 元長 091702 at E 178792.031, N 2616970.687 m; no corresponding 土庫 station was identified in that table. These historic names must not be used as current ring-site identifiers. The ring manifest instead records 元長國小 at E 179484, N 2616803 m and 土庫國中 at E 187771, N 2620610 m in TWD97/EPSG:3826. All coordinates above are transcribed source values, not a common-datum distance comparison.

The appendix calls the older coordinates `TM` and labels northings/eastings in metres, but gives no datum on these pages. The official [survey-methods section, printed page 17](https://twgeoref.gsmma.gov.tw/GipOpenWeb/wSite/public/Attachment/1999/19990159/0016.pdf) describes GPS surveying and comparison to the Taiwan-wide coordinate system without naming a datum in that passage. The separate Geological Survey borehole index (dataset 16799) now gives explicit TWD97 positions for those older borehole IDs: 元長 `091702G1` lies 193 m from the current 元長 ring and 土庫 `090551G1` lies 1007 m from the current 土庫國中 ring. This is a same-datum separation from an independent index, not a datum identification for the old appendix or a borehole-to-ring identity crosswalk. Even a small distance requires stratigraphic correlation and screen/interval matching before transfer. Treat those older logs as regional context. The exact-current-site 土庫 sources and their 24-versus-26 ring discrepancy are inventoried in `docs/SITE_GEOLOGY_SOURCE_AUDIT.md`.

For evaluation, freeze the observation geometry and candidate vertical model before
new measurements. The old 14 ring sites and the two 2023–2025 sites have already
influenced or been examined by model selection, so reuse of them is exploratory.
