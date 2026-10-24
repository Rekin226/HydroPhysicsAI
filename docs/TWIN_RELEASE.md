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

*Superseded in part by `docs/TWIN_EVIDENCE_2026-10.md` section 1 (innovation update).* The
2.072 m and 1.243 above score the field-update run (`datum_gate.npz`,
`results/twin/new_data_challenge/report.json`). The page now embeds the innovation-update
run, which scores **2.048 m**, ratio **1.229**, on the same 158 wells and 6,186 well-months
(`results/twin/new_data_challenge_innov/report.json`). The table is kept as the
2026-09-30 record.

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

## Rainfall from the CWA archive

The calibration rainfall is rebuilt from CWA CODIS (the vendor cache zero-filled gaps and
ended in 2022; `docs/TWIN_EVIDENCE_2026-10.md` section 11). Each gauge is a chain of station
IDs from the CWA inventory (`cwa_rain.CHAINS`); the fetch is idempotent and keeps raw
responses with hashes, the build leaves missing days empty and compares with the vendor file:

```bash
python -m hydrophysics.twin.cwa_rain fetch --raw data_fetch_api/new_measurements/cwa_codis_rain/raw \
    --refresh-year 2026        # re-fetch the growing current year
python -m hydrophysics.twin.cwa_rain build --raw data_fetch_api/new_measurements/cwa_codis_rain/raw \
    --cache chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv \
    --out-csv data_fetch_api/new_measurements/cwa_codis_rain/rf_timeseries_codis.csv \
    --report results/twin/cwa_rain/report.json
```

Install the built file as `chou-shui-data/data/rf_timeseries.csv` (with its `_source.csv`)
only after reading the comparison. A recalibration on new rainfall is the whole chain:
`results/twin_runs/par8/jobs.sh` holds the deliverable's recipe (full fit + 5 parallel folds
under MPS, `merge_folds`, posterior, two leveling columns, forward with `--update-method
innovation`, response basis, then challenge/coverage/hybrid/deep-pile/ring tools), started
with `tmux new -d -s hydro-par8 "bash results/twin_runs/par8/controller.sh"`. The October
2026 rerun (`stage3_datum_gate_cwa`, `datum_gate_cwa.npz`, `results/twin/*_cwa/`) scored
within noise of the registered model on every test (evidence section 11), so the page above
is still built from `datum_gate_innov`, to which the prospective registrations are bound.

The two models are not reproducible from the same code. The page's theta
(`results/twin_runs/stage3_datum_gate/stage3_theta.json`, sha256 `036a7a17`) was fitted at
commit `fa5cca2` on the vendor rainfall cache (`rf_timeseries_vendor_2012_2022.csv`). That
commit predates the zero-start adjoint fix (`730f65b`, 2026-10-11), and no record says
whether its tree was clean. Rerunning the recipe at the PR head therefore does not
reproduce the page's theta: check out `fa5cca2` and install the vendor rainfall cache as
`rf_timeseries.csv` first. Even then no GPU fit is bit-identical. `stage3_datum_gate_cwa`
(theta sha256 `789d0e45`; recorded commit `9715f58`; the code was at or after `13860ee`,
see `TWIN_THETA_PROVENANCE_2026-10.md`; both include the fix) is the post-fix
reference on the CWA rainfall. It is also the model the N1 and I1 tests scored (evidence
section 13).

Do not commit mid-batch from the checkout the jobs run in: the fold files record `HEAD`,
and `merge_folds` logs (no longer refuses) a commit-only difference.

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
  --forward results/twin_forward/datum_gate_innov.npz \
  --stations AMP_V2/data/fan_stations.parquet \
  --historical-wells AMP_V2/data/wells \
  --new-wells data_fetch_api/new_measurements/heads_202210_202610 \
  --end 2026-09-01 --out results/twin/new_data_challenge_innov
```

This is the report the page embeds. The 2026-09-30 run above used `datum_gate.npz` (the
field update) with `--out results/twin/new_data_challenge`; that report is superseded.

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

The command below reproduces the source manifest of the live page
(`results/twin/twin_app.html`, built 2026-10-23, identical to `twin_app_20261023.html`).
The build log records no command line, so the arguments were recovered from the page's
embedded hashes (`release.sources`), one input file at a time. Run it from the repository
root with `HYDROMIND_GW_DATA` set: the leveling and well summaries are read from there
(`--leveling auto`, `--wells auto`).

```bash
python -m hydrophysics.twin.viewer_app \
  --forward results/twin_forward/datum_gate_innov.npz \
  --basis results/twin_forward/response_basis_innov.npz \
  --basemap results/twin/basemap.npz \
  --challenge results/twin/new_data_challenge_innov/report.json \
  --coverage results/twin/coverage_datum_gate_innov/report.json \
  --prospective results/twin/prospective_2026-11/manifest.json \
  --ring-holdout results/twin/ring_holdout/report.json \
  --hybrid results/twin/hybrid_datum_gate_innov/report.json \
  --hybrid-registration results/twin/prospective_2026-11_hybrid/manifest.json \
  --deep-piles results/twin/deep_pile_holdout/report.json \
  --policy-response results/dataprep/reports/N1_policy/page_labels.json \
  --out results/twin/twin_app.html
python -m hydrophysics.twin.release inspect results/twin/twin_app.html
python -m hydrophysics.twin.browser_check \
  --page results/twin/twin_app.html --out results/twin/browser_acceptance
```

The committed `results/twin/twin_app.html` is the 2026-10-24 page `twin_app_20261024e.html`
(promoted 2026-10-24); it no longer carries the withdrawn "long-run trend" caveat.

The page's inputs, by role in `release.sources` (first 8 hex digits of the sha256).
Roles marked *default* come from `viewer_app`'s defaults, which point at the registered
model `results/twin_runs/stage3_datum_gate/`:

| role | file | sha256 |
|---|---|---|
| forward, rheology | `results/twin_forward/datum_gate_innov.npz` | `cf2c7154` |
| members, member_summary | `datum_gate_innov.members.npz`, `.members.csv` (beside the forward file) | `ca43a405`, `3d440d59` |
| basis | `results/twin_forward/response_basis_innov.npz` | `f2e7c536` |
| basemap | `results/twin/basemap.npz` | `99bbe12f` |
| new_data_challenge | `results/twin/new_data_challenge_innov/report.json` | `b0648b27` |
| coverage | `results/twin/coverage_datum_gate_innov/report.json` | `fe1b950d` |
| prospective | `results/twin/prospective_2026-11/manifest.json` | `d7286551` |
| ring_holdout | `results/twin/ring_holdout/report.json` (**pre-fix reference**, see below) | `71286b0a` |
| hybrid | `results/twin/hybrid_datum_gate_innov/report.json` | `dbd4a314` |
| hybrid_registration | `results/twin/prospective_2026-11_hybrid/manifest.json` | `8e28ac21` |
| deep_piles | `results/twin/deep_pile_holdout/report.json` | `19421c35` |
| policy_response | `results/dataprep/reports/N1_policy/page_labels.json` | `bd41d77e` |
| theta (*default*) | `results/twin_runs/stage3_datum_gate/stage3_theta.json` | `036a7a17` |
| column, rheology_column (*default*) | `stage3_datum_gate/coupled_leveling{,_tau30}/stage4_column.csv` | `5673a25e`, `bb6b4fd2` |
| temporal, rescore, townships, alternative (*default*) | see `viewer_app` `DEFAULT_*` | |

The ring line deliberately reproduces the live page. The 2026-10-11 page and its
2026-10-23 rebuild both embed `results/twin/ring_holdout/report.json`. That report
references each ring across the 2022 instrument gap, which evidence section 5 marks as
superseded. The corrected report is `results/twin/ring_holdout_ref2023/report.json`
(`92e621d4`). Switching the page to it is a page change that needs its own rebuild and
decision; until then, the page's ring line shows the pre-fix numbers.

**Candidate page (2026-10-24, not promoted).** The page's reports are bound to its forward
run by sha256 (`model_sha256`, `forward_sha256`), so the forward, basis and challenge must
be the `_innov` set the page uses. A rebuild that switches to the corrected ring report and
to the v2 policy labels, which name the model behind each N1 row
(`dataprep_staging/n1_policy/page_labels_v2.py`), changes only those two inputs. This is
the command of `results/twin/twin_app_20261024b.html`; it writes a new file and leaves
`twin_app.html` as it is:

```bash
python -m hydrophysics.twin.viewer_app \
  --forward results/twin_forward/datum_gate_innov.npz \
  --basis results/twin_forward/response_basis_innov.npz \
  --basemap results/twin/basemap.npz \
  --challenge results/twin/new_data_challenge_innov/report.json \
  --coverage results/twin/coverage_datum_gate_innov/report.json \
  --prospective results/twin/prospective_2026-11/manifest.json \
  --ring-holdout results/twin/ring_holdout_ref2023/report.json \
  --hybrid results/twin/hybrid_datum_gate_innov/report.json \
  --hybrid-registration results/twin/prospective_2026-11_hybrid/manifest.json \
  --deep-piles results/twin/deep_pile_holdout/report.json \
  --policy-response results/dataprep/reports/N1_policy/page_labels_v2.json \
  --out results/twin/twin_app_20261024b.html
```

`twin_app_20261024b.html` was built before the two fixer branches were merged, so it lacks
the head-response size line (`evc_pr_heads_size`), the head-gain caveat and the neutral
subsidence wording of the merged template.

**Rebuilds from the merged code (2026-10-24, not promoted).** Two pages carry those labels.
Both are built from the uncommitted merge of the two fixer branches on base `82da213`, with
that tree's `hydrophysics` first on `sys.path`, and both pass `release inspect` (no errors)
and `browser_check`. Their command is the 20261024b command above plus `--posthoc
results/dataprep/reports/N1_gaps_2026-10-24/page_posthoc.json` (sha256 `fb59adcc`, off by
default): the post-hoc, model-only labels (hot-spot stability, the per-100-GWh/yr near-tie,
the elastic/inelastic split), which the viewer refuses unless the file names the same
forward run. This is the first line of each build log:

```bash
python -m hydrophysics.twin.viewer_app \
  --forward results/twin_forward/datum_gate_innov.npz \
  --basis results/twin_forward/response_basis_innov.npz \
  --basemap results/twin/basemap.npz \
  --challenge results/twin/new_data_challenge_innov/report.json \
  --coverage results/twin/coverage_datum_gate_innov/report.json \
  --prospective results/twin/prospective_2026-11/manifest.json \
  --ring-holdout results/twin/ring_holdout_ref2023/report.json \
  --hybrid results/twin/hybrid_datum_gate_innov/report.json \
  --hybrid-registration results/twin/prospective_2026-11_hybrid/manifest.json \
  --deep-piles results/twin/deep_pile_holdout/report.json \
  --policy-response results/dataprep/reports/N1_policy/page_labels_v2.json \
  --posthoc results/dataprep/reports/N1_gaps_2026-10-24/page_posthoc.json \
  --out results/twin/twin_app_20261024c.html
```

- `results/twin/twin_app_20261024c.html`: sha256 `499e9025`, release `code_sha256`
  `e585666f`, build log `twin_app_20261024c.build.log` (`abaa18e3`), browser report
  `browser_acceptance_20261024c/report.json` (`fd9a03d8`).
- `results/twin/twin_app_20261024d.html`: the same command with `--out
  results/twin/twin_app_20261024d.html`; sha256 `7b6350c3`, release `code_sha256`
  `3461dd9b`, build log `twin_app_20261024d.build.log` (`aa3356fb`), browser report
  `browser_acceptance_20261024d/report.json` (`f3fe6583`). Its sources and payload equal
  20261024c's; the only change is the post-hoc lift sentence (`cv9_lift`, en and zh), which
  now says the heads return abruptly to the baseline (science-13 splices the ensemble-mean
  heads; the flow model's own return would take months). 20261024c, whose lift sentence has
  no such qualifier, is superseded and not a promotion candidate.
- `results/twin/twin_app_20261024e.html`: the same command with `--out
  results/twin/twin_app_20261024e.html`; sha256 `e37a36d3`, release `code_sha256`
  `ea6cadc9`, build log `twin_app_20261024e.build.log` (`94becf35`), browser report
  `browser_acceptance_20261024e/report.json` (`65744226`). Its sources and build log equal
  20261024d's; the only change is that the headline's Yunlin figure (`hl_yun`, en and zh)
  names its cell set: the 1,091 cells of the named Yunlin townships, not the model's county
  mask (1,205 Yunlin cells) used by science-05 and science-13.

The 20261024e page's `code_sha256` (`ea6cadc9`) is the hash of the package as committed
with it; the earlier candidates (b, c, d) match only intermediate trees and are superseded.
20261024e was promoted to `twin_app.html` on 2026-10-24.

**The release pointer is stale.** `results/releases/current.json` still points to the page
published on 2026-09-30 (`page_sha256` `95f47e20`, forward `datum_gate.npz`, no
`policy_response` source). The labelled pages since then were promoted only by copying
them to `results/twin/twin_app.html`; none went through `release publish`, which would
rewrite `current.json`. The two therefore name different pages; `twin_app.html` is the
current page, and `current.json` is updated only by a `release publish`.

Earlier versions of this section and of the README built from `datum_gate.npz` and
`response_basis_datum.npz`, the field-update run that section 1 supersedes; those commands
do not reproduce the live page. Builds after this change print the full command line as
the first line of their log.

The statistical layer (`twin.hybrid`, climatology plus the decayed last-observed anomaly)
is scored on the same acquisitions as the challenge and registered prospectively beside
the physics; both registrations are scored by `twin.prospective score` after the window:

```bash
python -m hydrophysics.twin.hybrid score --forward results/twin_forward/datum_gate_innov.npz \
  --stations AMP_V2/data/fan_stations.parquet --historical-wells AMP_V2/data/wells \
  --new-wells data_fetch_api/new_measurements/heads_202210_202610 --end 2026-09-01 \
  --out results/twin/hybrid_datum_gate_innov
python -m hydrophysics.twin.hybrid register ... --test-start 2026-11-01 --test-end 2027-10-31 \
  --retrospective results/twin/hybrid_datum_gate_innov/report.json \
  --out results/twin/prospective_2026-11_hybrid
python -m hydrophysics.twin.prospective score --registration results/twin/prospective_2026-11_hybrid \
  --new-wells <acquisitions through 2027-10> --out results/twin/prospective_2026-11_hybrid/score
```

The deep-benchmark line comes from the WRA monthly reports: `twin.report_extract`
transcribes table 3 of each PDF twice with the Claude API (needs `ANTHROPIC_API_KEY`;
`.env` is gitignored) and checks the numbers against each other, against the reports'
own trailing-year arithmetic and against the open-data hourly feed; `twin.deep_pile_holdout`
reconstructs each pile's cumulative compression and scores the frozen column against it:

```bash
set -a; . ./.env; set +a
python -m hydrophysics.twin.report_extract \
  --reports data_fetch_api/new_measurements/wra_monthly_reports \
  --feed <open-data 58339 JSON dumps> --out results/twin/report_extract/deep_piles
python -m hydrophysics.twin.deep_pile_holdout \
  --piles results/twin/report_extract/deep_piles/deep_piles.csv \
  --sites chou-shui-data/data/subsidence/deep_pile_sites.csv \
  --forward results/twin_forward/datum_gate_innov.npz \
  --members results/twin_forward/datum_gate_innov.members.npz \
  --out results/twin/deep_pile_holdout
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
(non-empty list of command argument lists), `acceptance_reports` (the artifact bindings
below), and optional `fetch`, `timeout_seconds`, `max_age_days` and
`as_of`. Relative data paths resolve against `cwd`. Commands inherit environment
credentials; their arguments and output are not copied into status files.

As of October 7, a boolean passing report alone is rejected. Each acceptance entry
must identify all three local files:

```json
{"report": "acceptance.json", "protocol": "protocol.json", "predictions": "predictions.csv"}
```

The runner recomputes `validation.evaluate(predictions, protocol)` and requires an
exact match to the report. It currently supports only `free-running-head-validation`;
this is not a subsidence or policy-effect gate. The protocol must additionally
identify `code_sha256` and `observations_sha256` for publication. Its `model_sha256`
must equal the rendered page's forward-artifact hash. Historical evaluation can
still run without these extra fields, but cannot approve an update.

Render with `--validation-protocol protocol.json --validation-predictions predictions.csv`
so the page's source manifest binds the exact protocol (including task period and
criteria) and predictions. Only hashes/names are embedded; the prediction table is
not published. Code, forward run, observation batch, protocol and prediction hashes
must all match the recomputed report. The accepted manifest retains the report hash,
task, period and input identities. Changed evidence, a changed observation file or a
page modified between checking and publishing retains the previous release.

These checks establish artifact consistency and repeatable scoring. They do not
attest that a manually declared registration date is genuine, that the holdout was
unused, or that measured interventions establish causality. Preserve independently
timestamped prediction issuance and source availability for prospective evaluation.
The standalone `release publish` command remains a local **research-artifact**
publisher; it does not run or claim observational acceptance.

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
