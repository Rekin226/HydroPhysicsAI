# Twin release and validation workflow

The application is a **research scenario viewer**. Numerical solver agreement, model
agreement across parameter sets, and pumping-response sanity checks do not establish
forecast skill or a causal intervention effect. The September 2026 implementation
keeps these claims separate and exposes failed checks.

## Verified results, 2026-09-30

| Check | Result | Interpretation |
|---|---|---|
| Two-layer transient flow versus MODFLOW 6.4.4 | Maximum head difference 3.17e-8 m | Synthetic discretization verification |
| Flow water balance | Maximum relative residual 1.19e-9 | Storage, recharge, pumping and boundary fluxes balance |
| Elastic drawdown/recovery versus CSUB and analytic solution | Maximum difference 1.15e-17 m | Elastic limit only; no creep-equivalence claim |
| Newly acquired heads, January 2023–August 2026 | 158 wells, 6,186 scored well-months | Previously unused temporal challenge |
| Frozen forecast, historical-only datum adjustment | RMSE 2.072 m | Worse than seasonal climatology, 1.667 m |
| Frozen forecast without datum adjustment | RMSE 8.076 m | Large spatial offsets remain |
| Offline browser checks | Desktop and mobile pass; no remote requests or JS errors | EN/中文, 3D controls, layout and 2D fallback work |
| Software-rendered policy updates | Desktop p95 0.49–0.56 s; mobile 0.12–0.18 s | Does not certify the 30 FPS device target |

The field challenge has an RMSE ratio of **1.243** against the best simple baseline and
fails the improvement gate. It was evaluated retrospectively after acquiring the data;
it is not a preregistered prospective trial. No calibration was changed to improve this
score. The viewer still uses its December 2022 model origin, with the challenge shown
separately. It does not relabel 2023–2025 as validated because an older test ran 36 months.

## Data acquired locally

The authorized vendor connection was used without copying its credentials into code,
logs, manifests, or the page. Real observations remain in ignored directories.

* Groundwater: 158 in-grid wells; 29,614,215 readings from October 2022 to September 30,
  2026. October–December 2022 overlap matches the old cache exactly at shared valid
  timestamps for 157 wells. Station metadata ends in 2025 even though the live series
  reaches 2026.
* Compaction: 38 catalog stations queried, 16 non-empty responses, 61,010 records in
  total, latest May 2025. The old 14-site ring cache already extended to March 2025;
  those historical ring observations are not a fresh independent holdout.
* Leveling: catalog end dates do not extend beyond 2021.
* Pumping: inspection for the forcing experiment found that the existing electricity
  cache already extends through July 2025. January 2023–June 2025 was assembled using
  the meter population selected on 2012–2022 only; over 99.9% of retained pumps have
  records in every evaluation month. This does not establish completeness of unmetered
  abstraction or validate the electricity-to-volume conversion.
* Weather: the forcing experiment uses locally cached Open-Meteo rainfall and ET0
  reanalysis at the original 61 ET0 locations. Vendor rainfall reads timed out. A
  weather-only seasonal source alignment is fitted on 2012–2021 and checked on 2022;
  it never uses the new head observations. Canal delivery forcing remains absent from
  the frozen model.

The local download audit is under `data_fetch_api/new_measurements/`. Preserve this
directory separately from `AMP_V2/data/`, which is the historical calibration input.
The new groundwater challenge uses complete months through August 2026, omitting the
partial current month. It rejects sentinel heads outside ±1000 m and requires at least
12 observed hours/day on 20 days/month. Monthly means give each retained day equal
weight. Scores give each well equal weight on identical available observations.
The normalized batch has 24 stations older than the 62-day freshness limit as of
September 30, 2026. Its operational freshness check therefore fails even though it
provides enough historical observations for the retrospective challenge.

## Frozen forcing attribution experiment

`hydrophysics.twin.forcing_experiment` compares six trajectories with unchanged flow
parameters: historical climatological forcing; updated pumping only; updated weather
only; both updated; climatology with monthly state updates; and both updated with
monthly state updates. Every update happens **after** that month's prediction is
stored. The primary metric is equal-well RMSE on identical observed months, with at
least 12 months per well. Datum adjustments and simple seasonal/trend baselines use
2012–2022 only. A separate one-step comparison includes previous-month persistence.

The local protocol records parameter hashes, dates, rules and amendments made before
scoring the experimental arms. The earlier failed head challenge was already known;
this is not an untouched prospective holdout. The replay must reproduce both the
historical trajectory and the frozen
forecast within 1 mm before attribution is accepted. Cached forcing and historical
replay files are checked against their audit hashes. Raw measurements and experiment
outputs remain in ignored directories.

```bash
OMP_NUM_THREADS=1 python -m hydrophysics.twin.forcing_experiment \
  --protocol results/twin/forcing_experiment/protocol.json \
  --out results/twin/forcing_experiment --device cuda --compile-matvec
```

The uncached path rebuilds pumping from the original census/electricity files and
weather from the local authorized downloads. For the Open-Meteo alternative, store
daily `precipitation_sum` and `et0_fao_evapotranspiration` JSON responses by original
`st_id` in `data_fetch_api/forcing_experiment/reanalysis/` (2022 through the evaluation
end) and `reanalysis_historical/` (2012–2022). Set `weather_source: "openmeteo"` in the
protocol. The experiment's fixed alignment uses per-cell, calendar-month mean recharge
ratios from 2012–2021. It does not tune this mapping against head skill.

Known future forcing and end-of-month observations make this a retrospective
diagnostic. The weather source substitution, inferred pumping and missing canal term
limit causal attribution. Even an improved short-term head score does not validate
long-horizon policies, subsidence, or uncertainty coverage. The viewer's deployed
scenario forecast is not replaced by these experimental trajectories.

The completed January 2023–June 2025 experiment scores 156 wells and 4,460 common
well-months, using 36 trajectories per arm. The control reproduces the saved forecast
within 3.65e-6 m. Equal-well RMSE is 2.067 m for the control, 1.845 m with updated
pumping, 2.038 m with aligned weather, and 1.848 m with both. Monthly state updates
worsen these scores to 2.181 m under climatological forcing and 1.942 m with both
inputs updated. Seasonal climatology scores 1.630 m. On the separate matched one-step
subset, previous-month persistence scores 1.332 m versus 1.938 m for the fully updated
model.

Updated pumping improves every ensemble trajectory and 126 of 156 scored wells, but
the best arm remains 13.2% worse than seasonal climatology. Updated pumping reduces
the control's mean squared error by about 20%; most error remains. Temporal variability is
still too small: median well-level standard deviation is 0.424 m in the pumping arm
versus 1.189 m observed. State updates improve timing while increasing positive bias.
These results prioritize the observation operator/state update and transient response
for the next investigation; they do not uniquely identify a missing physical process.
The source-substituted weather arm cannot settle the role of actual gauge rainfall.
The failed forecast-skill status remains appropriate.

## Fetching dated observations

Configure `WISENVR_BASE_URL`, `WISENVR_USERNAME`, `WISENVR_PASSWORD` and, when the
authentication route differs, `WISENVR_TOKEN_URL` in the environment. The committed
client never loads credentials from repository files.

```bash
python -m hydrophysics.twin.fetch_amp stations --out results/new-data/stations.parquet
python -m hydrophysics.twin.fetch_amp wells \
  --stations results/new-data/stations.parquet --out results/new-data/wells \
  --start 2023-01-01 --end 2026-10-01
```

The client splits capped responses instead of accepting the API's default newest-20,000
rows as complete history. The API schema also documents `limit=0` as uncapped, but the
bounded reader does not depend on that extension. Resume state is bound to a dataset
and date interval; a different interval requires a separate output directory.

Normalized observation CSVs use `station_id,date,head_m,layer,datum`, with layer 1–4 and
one declared vertical reference. QC rejects duplicates, mixed datums, future dates and
non-finite values. It reports stale stations individually; the update runner requires
all supplied stations to be fresh. The acquired vendor reference is not independently
surveyed and must not be represented as a newly verified physical datum.

```bash
python -m hydrophysics.twin.validation qc observations.csv \
  --max-age-days 62 --out results/observation-qc.json
```

## Independent numerical verification

Install the optional `reference` extra and obtain an official MODFLOW executable
separately. No benchmark code downloads or installs a simulator automatically.

```bash
python -m hydrophysics.twin.reference --mf6 /path/to/mf6 \
  --out results/twin/reference_acceptance
```

The flow case uses heterogeneous transmissivity, two confined layers, interlayer
leakage, general-head boundaries, seasonal recharge and a pumping reduction over 24
monthly steps. Identical stresses and material properties are passed to MODFLOW and
the differentiable solver. The CSUB case uses an active cell driven by a general-head
boundary because CSUB does not compute compaction in constant-head cells; the solved
head history drives the analytic and VEP elastic responses.

Acceptance tolerances are 1e-5 m for heads, 1e-6 for relative water-balance residual,
and 1e-8 m for elastic compaction. Tested with FloPy 3.10.0 and MODFLOW 6.4.4 on Ubuntu
20.04. The current 6.8.1 Linux binary requires a newer system C library on this host.
No operating-system or GPU-driver change was needed.

Repository regression checks passed: 527 tests, 7 skips. The final focused checks for
the viewer, fetching, releases and new-data challenge passed all 71 tests. Ruff and the
lockfile consistency check passed. The changed files contained no vendor credentials;
private downloads remain ignored.

## Evaluating a frozen forecast

```bash
python -m hydrophysics.twin.challenge \
  --forward results/twin_forward/datum_gate.npz \
  --stations AMP_V2/data/fan_stations.parquet \
  --historical-wells AMP_V2/data/wells \
  --new-wells data_fetch_api/new_measurements/heads_202210_202610 \
  --end 2026-09-01 --out results/twin/new_data_challenge
```

Outputs are an aggregate `report.json`, private per-well `predictions.csv` and normalized
`observations.csv`. Model offsets and climatology, climatology-plus-trend, and persistence
baselines use only observations up to the frozen forecast origin. The report hashes the
model and every new well file. It makes no predictive-interval coverage claim.

For a future prospective test, `validation evaluate` accepts a separately frozen
protocol and prediction CSV. Protocol fields are `test_start`, `test_end`,
`registered_before`, `model_sha256`, `baseline`, `max_rmse_ratio` (at most 1),
`min_stations`, `min_months`, and `independent_holdout`. Optional `coverage_bounds`
require calibrated interval bounds. Predictions contain `station_id,date,prediction_origin,
observed_m,predicted_m,baseline_m,model_sha256` and optionally `lower_m,upper_m`.
The origin must precede the holdout; matching hashes, coverage and minimum support are
checked. The software cannot independently prove that a protocol was preregistered:
archive the protocol and predictions before acquiring the test observations.

## Building and checking the viewer

```bash
python -m hydrophysics.twin.viewer_app \
  --forward results/twin_forward/datum_gate.npz \
  --basis results/twin_forward/response_basis_datum.npz \
  --basemap results/twin/basemap.npz \
  --challenge results/twin/new_data_challenge/report.json \
  --out results/twin/twin_app.html
python -m hydrophysics.twin.release inspect results/twin/twin_app.html
python -m hydrophysics.twin.browser_check \
  --page results/twin/twin_app.html --out results/twin/browser_acceptance
```

The default includes the licensed Three.js renderer. The public payload contains
aggregate observations only. Layer geometry is explicitly schematic. Each build hashes
its input artifacts and source tree, records library versions, and states that ensemble
spread is conditional model uncertainty, not calibrated predictive coverage.

Browser acceptance needs the optional `browser` extra and its Chromium installation.
The Ubuntu 20.04 verification used an isolated Playwright 1.48 installation; newer
Playwright browser binaries require a newer host. The check blocks HTTP/HTTPS, exercises
1440-pixel and 390-pixel viewports, tests language and 3D controls, saves screenshots, and
checks a separate WebGL-disabled browser. Timings identify the actual renderer.

## Local releases and guarded updates

```bash
python -m hydrophysics.twin.release publish results/twin/twin_app.html \
  --root results/releases
```

This creates an immutable, content-addressed HTML artifact and manifest, then atomically
switches `current.json`. It is local publication, not deployment. The publisher permits
research status only. Failed candidate checks preserve the current pointer.

`update_cycle` runs an explicitly configured fetch → QC → simulation/assimilation →
render → acceptance → publish workflow, using argument lists without a shell. A file
lock prevents overlapping updates. Nonzero exits, timeouts, stale observations, failed
acceptance reports and mismatched observation lineage retain the previous release.

Configuration fields are `cwd`, `release_root`, `observations`, `page`, `steps`
(non-empty list of command argument lists), `acceptance_reports` (non-empty list of JSON
reports with `passed: true`), and optional `fetch`, `timeout_seconds`, `max_age_days` and
`as_of`. Relative data paths resolve against `cwd`. Commands inherit environment
credentials; their arguments and output are not copied into status files.

The simulation producer must store `observations_sha256` in its forward NPZ. The viewer
accepts `--observations` only when that hash matches, and the update runner also checks
the observation hash and model origin. This is a workflow contract, not an implemented
new-data assimilation algorithm. Configure scientifically validated simulation steps
and acceptance reports before enabling a recurring run. No schedule or external
deployment has been created.

Remaining scientific work is a model/forcing revision that improves the frozen-head
test, calibrated uncertainty, updated forcing and leveling, and observed-intervention
validation. The new observations now make that work possible; an old-model relabeling
would not satisfy it.
