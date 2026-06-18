# Choushui Head + Subsidence Explorer — Design

**Date:** 2026-06-19
**Status:** Approved (brainstorming), pending spec review
**Author:** brainstormed with Claude

## Summary

A single self-contained **interactive HTML** (Plotly, no server) — the "Choushui
groundwater + subsidence explorer." One **date slider** animates two linked layers over
the real Zhuoshui alluvial-fan basemap:

1. a **3D groundwater-head surface** `h(x, y, t)`, interpolated (IDW) from the 61
   **observed** wells and masked to the fan polygon, and
2. a **land-subsidence surface** `S(x, y, t)` derived from cumulative head drawdown,
   with its single coefficient **calibrated against the real multi-layer compaction
   well (MLCW) data at 14 georeferenced sites**.

Plus a **validation panel**: predicted-vs-observed compaction across the 14 MLCW sites.

This turns the existing point-well dataset + the (now georeferenced) subsidence data into
a "what happened" digital-twin explorer. It is a visualization/explainer, not a precision
between-well tool (that limitation is documented for the head field).

## Scope decisions (locked)

- **Head surface = IDW of the 61 observed heads**, explicitly labeled as observation
  interpolation — *not* the SpatialPINN field (documented as untrustworthy between wells,
  see `2026-06-16-spatial-pinn-head-field-design.md`). This explorer shows what was
  observed, so observed interpolation is the honest source. The UDE is not used here.
- **Subsidence coupling = per-site calibrated proxy.** The 14 MLCW sites now have
  coordinates (user-provided, EPSG:3826, same CRS as the wells), so calibration uses real
  co-located head↔compaction pairs, not a basin-only aggregate.
- **Monthly time steps**, 2010–2022, to keep the HTML responsive and match MLCW's monthly
  cadence.
- **Standalone HTML** (Plotly frames + slider). No Streamlit/server.
- **Out of scope (YAGNI):** UDE-driven drought/wet *scenario* re-run (a separate
  follow-on); the rivers shapefile (27 MB) — coast + fan outline suffice; GNSS surface
  stations (kept for a later layer).

## Data (verified on disk)

- **GW heads:** 61 wells, daily 2010–2025, coords in `gw_stations.csv` (`TM_X97/TM_Y97`,
  EPSG:3826). Loaded via `hydrophysics.data.load_dataset`.
- **MLCW compaction:** 14 sites under `ls_cache/clean/ls-wra-mlcw-obs__*.parquet`, monthly
  2014–2025, columns `NO1..NO31` = magnetic-ring positions (m) at increasing depth. The
  percent-hex filenames decode (UTF-8) to the Chinese site names; all 14 join 1:1 to the
  user-provided coordinate table.
- **MLCW coordinates:** the 14 `(sub_id, X_3826, Y_3826, lon, lat)` rows provided by the
  user → stored as `chou-shui-data/chou-shui-data/data/mlcw_stations.csv`.
- **Basemap:** `Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.shp` (fan polygon, EPSG:3826)
  and `water/sea_TWD97.shp` (coast). Read with geopandas (installed, 1.1.3).

## Architecture / components

Two new focused modules, each independently testable.

### `hydrophysics/subsidence.py` — the science
- `load_mlcw_stations(path) -> DataFrame[sub_id, x, y]` — read `mlcw_stations.csv`.
- `mlcw_compaction(data_dir) -> dict[sub_id -> pd.Series]` — for each MLCW file, decode the
  name, read `NO1..NO31`, and compute the site's **total cumulative compaction** time
  series = displacement of the shallowest ring relative to the deepest (reference) ring,
  re-zeroed to the first observation. Sanity: compaction is broadly monotonic (land sinks);
  flag/keep sign convention positive = subsidence.
- `head_at(points_xy, data) -> (N, T)` — IDW-interpolate the 61 observed daily heads to
  arbitrary `(x, y)` (reuse the IDW pattern from `field_inputs.RainfallField`; power 2).
- `calibrate_sk(data, mlcw_series, mlcw_xy) -> {sk, r2, pairs}` — for each MLCW site,
  resample head to monthly, compute **cumulative drawdown** `D(t) = Σ max(0, runningmin
  drop)` of the head at that site, pool all (D, compaction) pairs across the 14 sites, and
  fit a single slope `Sk` by least squares through the origin. Return `Sk`, `R²`, and the
  per-site predicted/observed arrays for the validation panel.

### `hydrophysics/explorer.py` — the visualization
- `head_grid(data, n, dates) -> (XX, YY, HH[t])` — IDW the observed heads onto an `n×n`
  grid over the fan bounding box per monthly date; mask to the fan polygon (shapely
  `contains`) → NaN outside.
- `subsidence_grid(HH, sk) -> SS[t]` — `Sk · cumulative_drawdown` per grid cell from the
  head-grid history (same running-min-drop rule as calibration).
- `build_explorer(data, out_html) -> Path` — assemble a Plotly figure:
  - **3D surface** of head (z = head, color = head) over the fan, with animation **frames**
    (one per month) and a **slider** + play button.
  - a second trace/toggle for the **subsidence surface** (z = subsidence) sharing the slider.
  - **markers**: 61 wells and 14 MLCW sites (MLCW sized/colored by observed subsidence).
  - **validation subplot**: predicted-vs-observed compaction scatter at the 14 sites + the
    `Sk`/`R²` annotation.
  - write a self-contained `results/explorer/choushui_explorer.html`
    (`include_plotlyjs="cdn"`).
- CLI `python -m hydrophysics.explorer [--n 60] [--out ...]`.

### Repo integration
- Reuses `hydrophysics.data.load_dataset`, `hydrophysics.config`. Coordinates come from
  `data.attrs[tm_x/tm_y]`. geopandas/shapely for polygon mask; plotly for the figure.
- No change to existing models. New deps already present (plotly, geopandas, shapely).

## Data flow

```
mlcw_stations.csv ─┐
ls_cache/clean/*.parquet ─► mlcw_compaction() ─► per-site total compaction (monthly)
gw wells (data) ───────────► head_at(MLCW xy)  ─► per-site head ─► cumulative drawdown
                                          └─► calibrate_sk() ─► Sk, R², pred/obs pairs
fan polygon, coast ─┐
gw wells (data) ────┴─► head_grid() ─► HH[t] ─► subsidence_grid(Sk) ─► SS[t]
                                   └────────────┬──────────────┘
                                                ▼
                                   build_explorer() ─► choushui_explorer.html
                                   (head surface + subsidence + markers + validation)
```

## Error handling & failure modes

- **MLCW ring reference:** if the deepest ring is itself unstable or a site has irregular
  ring counts, fall back to (shallowest − deepest available) and re-zero; keep sign so
  positive = subsidence. Unit-tested on a synthetic monotone case.
- **IDW outside the well hull:** mask the grid to the fan polygon AND drop cells far from
  any well (so we don't extrapolate into empty corners); NaN renders as a hole.
- **Time alignment:** heads are daily, MLCW monthly — resample heads to month-end means
  before pairing/calibration.
- **HTML size/performance:** monthly steps (~150 frames) and `n≈60` grid keep it light;
  `include_plotlyjs="cdn"` avoids a multi-MB inline bundle. If still heavy, coarsen to
  quarterly — logged, not silent.
- **CRS:** everything stays in EPSG:3826 meters (wells, MLCW, fan polygon all match) — no
  reprojection needed; assert ranges overlap.

## Testing

- `load_mlcw_stations` returns 14 rows with finite x/y in the fan bbox.
- `mlcw_compaction` decodes all 14 filenames and returns monotone-ish positive series.
- `calibrate_sk` on **synthetic** data (compaction = k·drawdown + noise) recovers `k`
  within tolerance and reports `R²` near 1.
- `head_grid` is finite inside the fan polygon and NaN outside; correct shapes.
- `build_explorer` smoke: writes a non-empty `.html` containing a plotly div.

## Deliverable

`results/explorer/choushui_explorer.html` — open in any browser: scrub the slider to watch
the head surface and calibrated subsidence evolve 2010–2022 over the real fan, with wells
and the 14 MLCW sites, and a validation panel grounding the subsidence magnitude in the
measured compaction.
