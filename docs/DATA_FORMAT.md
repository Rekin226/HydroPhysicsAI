# Data format: running HydroPhysicsAI on your own wells

HydroPhysicsAI reads a small set of CSV files from one directory. Point the tool at that
directory with the `HYDROMIND_GW_DATA` environment variable:

```bash
export HYDROMIND_GW_DATA=/path/to/your/gw_data
python -m hydrophysics.run_baselines           # baselines on your data
python -m hydrophysics.train --model ude       # train the operator
```

The bundled sample under `hydrophysics/sample_data/` is a working, runnable example of
exactly this layout. Regenerate it any time with:

```bash
python -m hydrophysics.sample sample_data
```

Everything below is the contract that `hydrophysics/sample.py` writes and
`hydrophysics/data.py` reads. Match it and your data loads.

> **Coordinate system note.** Station coordinates (`TM_X97`, `TM_Y97`) default to
> **EPSG:3826 (TWD97 / Taiwan)**. Set `Config.crs` to your own CRS: any projected EPSG
> (metres) for another region, or `"EPSG:4326"` if `TM_X97`/`TM_Y97` are already
> longitude/latitude. Only the optional ET driver (`hydrophysics.et`) uses the CRS, to
> fetch weather at each well's true location; the core operator/baseline/eval work in
> whatever projected metres you supply. A mismatched CRS now raises a clear error instead
> of silently fetching ET at the wrong place.

---

## Directory layout

```
your_gw_data/
├── gw_timeseries.csv                      # groundwater levels (time series)
├── rf_timeseries.csv                      # rainfall (time series)
├── gw_stations.csv                        # station metadata
├── gray_box_input.csv                     # per-well topology + attributes (REQUIRED)
├── gw_fit_results.csv                     # optional: external gray-box baseline scores
└── intermediate/
    └── gw_coastal_inland_class.csv        # optional: coastal / tidal descriptors
```

## Files

### 1. `gw_timeseries.csv` — groundwater levels (required)
Wide format: first column is the timestamp, then **one column per well**, whose header is
that well's `st_id`.

| column | meaning |
|---|---|
| `date time` (first column) | timestamp, parseable by pandas (e.g. `2019-03-01 00:00:00`). Sub-daily is fine; the loader resamples to daily means. |
| `<st_id>` (one per well) | groundwater level in metres. `st1`, `st2`, … in the sample. Missing values allowed (blank / NaN). |

### 2. `rf_timeseries.csv` — rainfall (required)
Wide format: first column timestamp, then **one column per rain gauge**, header = `rf_id`.

| column | meaning |
|---|---|
| `date time` (first column) | timestamp (daily). |
| `<rf_id>` (one per gauge) | rainfall in mm/day. `rf1`, `rf2`, … in the sample. |

### 3. `gw_stations.csv` — station metadata (required)
One row per well.

| column | meaning |
|---|---|
| `st_id` | well id; must match the column headers in `gw_timeseries.csv`. |
| `TM_X97`, `TM_Y97` | projected coordinates in metres (EPSG:3826 for the ET driver — see note). |
| `Station`, `NAME_C` | numeric station code and a display name (used for labels; not modelled). |

### 4. `gray_box_input.csv` — per-well topology + attributes (required, the key file)
One row per well. This encodes the **hydrological pairing** the model consumes: which
upstream well drives each well, which rain gauge forces it, and the lags. These are
domain decisions you make; the tool does not infer them.

| column | meaning |
|---|---|
| `st_id` | well id (matches `gw_stations.csv`). |
| `ups_id` | id of the **upstream well** that drives this one (its observed level is used as an external driver). Use the well's own id, or leave the coupling weak, if there is no clear upstream neighbour. |
| `ups_lag_days` | lag (days) applied to the upstream driver. |
| `rf_id` | id of the **rain gauge** forcing this well (matches `rf_timeseries.csv`). |
| `lag_days` | lag (days) applied to rainfall. |
| `group` | `coastal` or `inland`. The string `coastal` sets the `is_coastal` attribute; any other value is treated as inland. |
| `active` | `1` to include the well, `0` to skip it. |
| `gw_TM_X97`, `gw_TM_Y97` | the well's coordinates (metres), duplicated here for convenience. |

### 5. `intermediate/gw_coastal_inland_class.csv` — coastal/tidal descriptors (optional)
Extra static attributes used to condition the operator. If absent, the related features
are zero-filled and conditioning degrades gracefully (inland-only networks can skip it).

| column | meaning |
|---|---|
| `st_id` | well id. |
| `dist_to_coast_m` | distance to coast (m). |
| `dom_amp`, `dom_freq_cpd`, `m2_amp` | dominant spectral amplitude/frequency and M2 tidal amplitude (from a tidal analysis of the level record). |
| `is_near_coast`, `is_m2_like`, `group` | boolean/label flags. |

### 6. `gw_fit_results.csv` — external gray-box baseline (optional)
Per-well scores from a separately-calibrated gray-box model, used only as a reference row
in the benchmark table (not recomputed by this repo).

| column | meaning |
|---|---|
| `st_id`, `group_name`, `model` | well id, group, model name. |
| `kge_val`, `rmse_val`, `r2_val` | validation-period scores. |

---

## Programmatic API (skip the CSVs)

If your data is already in memory, you don't need to write any CSV files.

**From arrays** — you have `(n_wells, n_days)` arrays:

```python
from hydrophysics import GWData
from hydrophysics.baselines import climatology_prediction

data = GWData.from_arrays(
    well_ids=["w1", "w2", "w3"],
    dates=dates,                 # length-T, anything pd.to_datetime accepts
    target=levels,               # (3, T) groundwater level
    rainfall=rain,               # (3, T)
    upstream=None,               # optional (3, T); None = no upstream coupling
    attrs=stations_df,           # per-well tm_x/tm_y (+ optional group, dist_to_coast_m…)
    split_date="2019-01-01",
    crs="EPSG:4326",             # your coordinate system (see note above)
)
print(data.validate())           # [] means good to go; otherwise a list of issues
pred = climatology_prediction(data)
```

**From tidy long tables** — "declare your columns" with `column_map`:

```python
from hydrophysics import load_dataset_from_frames

data = load_dataset_from_frames(
    gw_long, rf_long, stations, pairing=None,
    column_map={"DateTime": "date", "WellID": "st_id", "Level": "level",
                "GaugeID": "rf_id", "Rain": "rainfall", "X": "tm_x", "Y": "tm_y"},
    split_date="2019-01-01",
)
```

Canonical columns: `gw_long` → `date, st_id, level`; `rf_long` → `date, rf_id, rainfall`;
`stations` → `st_id, tm_x, tm_y` (+ optional attributes); optional `pairing` →
`st_id, ups_id, rf_id, ups_lag_days, lag_days, group`.

**`GWData.validate()`** returns a list of human-readable issues (missing static features,
dead wells, empty split, non-finite coordinates). Run it before training.

## Minimum to get a result
Required: `gw_timeseries.csv`, `rf_timeseries.csv`, `gw_stations.csv`, `gray_box_input.csv`.
Optional: `intermediate/gw_coastal_inland_class.csv`, `gw_fit_results.csv`.

The train/validation split defaults to `2019-01-01` and is configurable via
`Config.split_date`. See `hydrophysics/config.py` and `hydrophysics/data.py` for the
loader, and `hydrophysics/sample.py` for a script that emits every file above.


## The twin's data cache (`AMP_V2/data/`, `chou-shui-data/data/`)

The Choushui twin (`hydrophysics.twin`) reads two gitignored trees. Neither ships in the
repository; both are rebuilt as below. Paths are the `DEFAULT_PATHS` in
`hydrophysics/twin/calibrate_flow.py`, so no flag is needed once they exist.

| path | content | how it is rebuilt |
|---|---|---|
| `AMP_V2/data/fan_stations.parquet` | every monitoring well in groundwater zone 50 with its metadata (`GroundwaterLayerCode`, `GroundHeight`, `LocationByTWD97`, …) | `python -m hydrophysics.twin.fetch_amp stations` |
| `AMP_V2/data/wells/<sid>.parquet` | hourly water level per well, index `datetime`, column `value` | `python -m hydrophysics.twin.fetch_amp wells` |
| `AMP_V2/data/tpc_pumps.parquet` | registered-pump census inside the fan polygon (`sid`, `電號1` meter number, `PUMP_HP`, `PURPOSE`, `TWD97_X/Y`) | `python -m hydrophysics.twin.fetch_amp pumps --polygon "<fan>.json"` |
| `AMP_V2/data/pump_kwh_all.parquet` | monthly `electricity_kwh` per `pump` | same command (`--kwh-out`) |
| `chou-shui-data/data/Zhuoshui Alluvial Fan/*.json` | the fan polygon (EPSG:4326) | **not available from the API**; from the original data delivery |
| `chou-shui-data/data/rf_timeseries.csv`, `rf_stations.csv`, `gw_stations.csv` | rain gauges and the 61 curated wells (the ET0 cache is keyed on them) | original data delivery |
| `chou-shui-data/data/mlcw_stations.csv` + `ls_cache/` | MLCW compaction sites and the leveling panel used by Stages 1–2 | original data delivery / backup |
| `results/et/openmeteo_et0_2012_2022.npz` | cached ET0 per curated well | committed (`hydrophysics.et`) |

The fetcher takes the API host and account from `WISENVR_BASE_URL`, `WISENVR_USERNAME`
and `WISENVR_PASSWORD` only. It renews the bearer token on expiry and records *successes*
in a `.fetched.json` beside the output, so an interrupted fetch resumes without skipping
what failed.

**Canonical head field.** `heads.build_head_field` on the current cache yields **174 wells
passing QC, 158 inside the 1 km grid, 8.79 % NaN month-cells** (the numbers every run
prints at startup). An earlier cache produced 147 wells at 1.1 % NaN; results recorded
before 2026-09-05 were computed on it. The 174-well field is canonical from 2026-09-11:
it is the one the fetcher reproduces, and every gate since has been run on it.
