# dataprep_staging

Phase-0 staging code for the October 2026 data-remediation plan
(`docs/DATA_REMEDIATION_PLAN_2026-10.md`; audit in `docs/DATA_AUDIT_2026-10-11.md`).

This folder exists because of user decision D12: the code that builds the data products is
versioned here, outside `hydrophysics/` and `tests/`, which stay frozen while the GPU
calibration batch runs. After the freeze lifts (Phase 1) the code is ported to
`hydrophysics/dataprep/` with CPU-only tests. Nothing under `hydrophysics/` imports this
folder; code here may import repository modules read-only (for example
`hydrophysics.twin.pumping.clean_census`).

## Layout

```
dataprep_staging/
  README.md            this file
  .gitignore           blocks data files (parquet, npy, npz, csv, json, txt, html, pdf, ...)
  tools/cpu_slot.sh    shared CPU-slot wrapper (see below)
  common/              shared library (WP A0)
    paths.py           fixed layout, origin 2022-12
    hashing.py         sha256 of files, trees, arrays, grid masks
    flags.py           flag table (plan 2.4), long-table and grid validators
    uses.py            use vocabulary, pre-origin uses, flags allowed per use, ledger roles
    manifest.py        MANIFEST.json schema (plan 2.3), writer, validator, new_version_dir
    guard.py           loader guard open_product(dir, use[, window_end]): U1-U3, U5, U7
    ledger.py          per-WP ledger fragments, data_roles.csv assembly (plan 4.1, U6)
    source.py          L0 SOURCE.json writer/validator, read-only copy into L0
    check.py           CLI: re-hash every MANIFEST.json and SOURCE.json; report L0 orphans
    tests/             CPU unit tests of the library (synthetic files only)
    a0/                WP A0 one-off scripts (pilot move, L0 copy, pilot manifests, reruns;
                       r2_* = repair round: window audit, r2 pilot manifests, L0 registry,
                       chain rerun, rho(k) check, guard probes, ledger, report)
    a0fix/             WP A0fix scripts (time-encoding probe of the guard, ledger, report)
  harness/             validation and anti-leakage harness (WP A1; shared by every WP)
    scorers.py         equal-well / anomaly / out-of-time RMSE (training-mean datum),
                       native and split-conformal coverage, recovery metrics, U2 sigma_obs
    coverage.py        cluster-adjusted Wilson interval and the plan 4.4 coverage gate
    placebos.py        6-month shift, own climatology, year shuffle, distant/regional kWh
    gaps.py            artificial-gap generators of every plan 4.4 family (fixed seeds)
    units.py           design-time and fresh units, stress regions, site-grouped folds
    prereg.py          pre-registration tool, hash-chained look log, refusal to score
    build_units.py     builds results/dataprep/validation_units/v<N>/ and declares it
    repro_pilots.py, coverage_sim.py, coverage_confirm.py, a1_prereg.py, a1_checks.py,
    a1_report.py       A1 acceptance and evidence scripts
    tests/             CPU unit tests (synthetic data only)
  <wp>/                one directory per work package (WP-specific code only)
```

Data never lives here:

| What | Where (all gitignored) |
|---|---|
| L0 new downloads | `data_fetch_api/new_measurements/<source>/` (+ `SOURCE.json`; rolling sources in `<source>/<fetch-date>/`) |
| L0 vendor caches (read-only) | `AMP_V2/data/`, `chou-shui-data/data/`, existing `data_fetch_api/` files |
| L1-L3 products | `results/dataprep/<product>/v<N>/` with `MANIFEST.json` |
| L0 registry (hashes of pre-convention L0 files) | `results/dataprep/l0_registry/v<N>/` |
| Pilots (design-time evidence) | `results/dataprep/_pilots/<scout>/` with `MANIFEST.json`; `INDEX.json` |
| Ledger | `results/dataprep/ledger/rows/<WP>.csv` -> `ledger/data_roles.csv` |
| Pre-registrations | `results/dataprep/preregistration/<id>.json` (A1 tool, written before the test) |
| WP reports | `results/dataprep/reports/<WP>.md` and `<WP>.json` |

## Rules (binding; see plan sections 2.5, 4 and the Phase-0 hard rules)

- No edits under `hydrophysics/` or `tests/` during the freeze. No GPU (set
  `CUDA_VISIBLE_DEVICES=""` for anything that imports torch).
- Every computation longer than about a minute runs through `tools/cpu_slot.sh`.
- Products are never overwritten: a new version is a new `v<N+1>` directory
  (`manifest.new_version_dir`), and `write_manifest` refuses to replace a manifest.
- L0 is never modified: `source.copy_into_l0` and `write_source_json` refuse to replace
  differing content and make the copies read-only. Downloads are untrusted data: never
  execute them; read them with `python -I` and pass paths as arguments.
- Anything that calibrates, starts, sets a prior for, designs or selects a model scored on
  2023-26 uses pre-origin products (`fit_window_end <= 2022-12`) or a listed exemption (U3);
  the loader guard enforces it (U5). No 2023+ observation enters a gate, threshold,
  hyperparameter or member choice (U7). Imputed or inferred values (flags 2-6) never become
  scoring targets (U1).
- Every reported number gets ledger rows naming each source and its single role (U6).
- Privacy: never read the census columns `戶名1` or `用電地址2`; never copy contact names or
  phones (station metadata, data.gov.tw publisher fields). Pump coordinates stay under
  `results/`.
- Before any commit (only when the user asks): run the CLAUDE.md Rule 1 secrets scan. This
  folder must stay free of data, coordinates and personal fields.

## The CPU-slot wrapper

Four slots are shared by every data-prep agent so that the GPU batch keeps its CPU cores.

```bash
dataprep_staging/tools/cpu_slot.sh python my_job.py args...          # one slot
dataprep_staging/tools/cpu_slot.sh --slots 2 python my_job.py ...    # two slots (rare)
```

It waits for a free slot (`flock` on `results/dataprep/.cpu_slots/slot<i>`), sets
`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` to the slot count and runs the command under
`nice -n 10`. Long jobs: run them in the background and poll; foreground calls time out.

## Using the library

```python
import sys
sys.path.insert(0, "<repo>/dataprep_staging")             # the repository checkout
from common import guard, ledger, manifest, source

# build a product version (an observed table 2012-2026 whose QC parameters saw 2012-2022 only)
d = manifest.new_version_dir("heads/L1_clean")            # results/dataprep/heads/L1_clean/v1
...  # write outputs and validation.json into d
m = manifest.build_manifest(
    d, product="heads/L1_clean", layer="L1",
    inputs=["AMP_V2/data/fan_stations.parquet"], code=["dataprep_staging/h1/qc.py"],
    parameters={"seed": 1000}, fit_window_end="2026-09", validation_report="validation.json",
    gate={"passed": True, "criteria": "plan 3 WP H1"},
    time_slicing={"column": "date", "params_fit_window_end": "2022-12", "encoding": "iso"},
    allowed_uses=["scoring_target", "calibration_target", "datum_fit", "validator"],
    forbidden_uses=["prior"], personal_data="none")
assert not guard.time_slicing_errors(m, d)   # declared encoding + every row of every table
manifest.write_manifest(d, m)

# consume it: declare the use; the guard enforces allowed_uses, fit_window_end, gate, hashes
p = guard.open_product(d, "calibration_target", window_end="2022-12")
df = p.read_table("heads.parquet")          # rows <= 2022-12, flags 0-2 only (U2, U3/U5)
s = guard.open_product(d, "scoring_target").read_table("heads.parquet")   # flags 0-1 (U1)

# ledger rows for a reported number (one role per source per number)
ledger.append_rows("H1", [{"number_id": "H1.steps_flagged", "source": "AMP_V2/data/wells",
                           "role": "input", "product_version": "heads/L1_clean/v1"}])
```

Uses are a fixed vocabulary (`common/uses.py`). Flags per use (U1, U2): `scoring_target`,
`baseline`, `datum_fit`, `prospective_scorer` take flags 0-1; `calibration_target` 0-2.
**Pre-origin uses** (U3/U7; the product must have `fit_window_end <= 2022-12`, a covering
exemption, a time slice or pre-origin outputs): `calibration_target`, `datum_fit`,
`baseline`, `initial_conditions`, `idw_field`, `prior`, `prior_width_input`,
`model_design`, `selection`, `forcing_frozen`, `design_time_evidence`. The others:
`scoring_target`, `prospective_scorer`, `forcing_recorded_arm`, `validator`,
`statistical_layer_history`, `after_the_fact_analysis`. A reference statistic computed from
the test period itself (an NSE denominator) is part of the score: use `scoring_target`.
Ledger roles: `fit_target`, `prior`, `prior_width_input`, `input`, `validator`,
`design_time_evidence`.

Guard rules (repair round, schema `dataprep-manifest/2`):

- **Overrides only tighten.** `preorigin=True` marks a non-pre-origin use as feeding a model
  scored on 2023-26; `preorigin=False` on a pre-origin use raises. `require_gate=True`
  demands a passed gate for a gate-exempt use; `require_gate=False` on a gated use raises.
  `verify=False` only skips the output re-hash and is noted in `Product.notes`.
- **Time slices.** An observed product spanning 2012-2026 (for example `heads/L1_clean`) can
  serve datum fits, baselines and calibration targets only through a slice: the manifest
  declares `time_slicing = {"column": "date", "params_fit_window_end": "YYYY-MM",
  "encoding": "iso"}` (`params_fit_window_end` is the last month any of the product's own
  fitted parameters, such as QC thresholds, saw), and the consumer calls
  `open_product(d, "datum_fit", window_end="2022-12")`; `read_table` then returns rows up to
  the window only, and raw `path()` access is refused.
- **Time encodings (WP A0fix).** The slice column must match a declared
  `time_slicing.encoding` on every row of the table (rows the use's flag filter would drop
  included), or `read_table` raises `guard.TimeEncodingError` (a `ValueError`) before
  returning anything:

  | encoding | column | readable rows for `window_end` W |
  |---|---|---|
  | `iso` (default when absent) | ISO-8601 strings `YYYY-MM`, `YYYY-MM-DD`, `YYYY-MM-DD[T ]hh:mm[:ss[.f]][Z\|+hh:mm]`, or a datetime64 column (naive or tz-aware) | month <= W (the month written in the value) |
  | `yyyymm` | integers `YYYYMM`, 190001-209912, month 01-12 | month <= W |
  | `year` | integer calendar years 1900-2099 | `Y-12` <= W: the whole year inside the window (W = 2022-06 excludes 2022) |

  Refused: integer columns not declared `yyyymm` or `year` (so epoch seconds, milliseconds or
  days can never pass as months), values outside the declared range (2023 declared `yyyymm`,
  202301 declared `year`, YYYYMMDD), floats (an integer column with gaps reads as float:
  drop or fill those rows), booleans, periods, missing values, bare year strings (`"2022"`),
  non-ASCII digits, and object columns that mix strings and numbers. A categorical column is
  checked through its categories. An unknown encoding raises when a window is opened.
  Builders always declare `encoding` explicitly and call
  `guard.time_slicing_errors(m, d)` before `write_manifest` (it requires the key, checks the
  column in every parquet/csv output and that at least one output carries it); the schema
  validator in `manifest.py` does not check the encoding (see the A0fix report). Helpers:
  `guard.time_keys(series, encoding)` (YYYYMM keys of the last month each row covers) and
  `guard.slice_mask(series, encoding, window_end)`.
- **Per-output windows.** Pilots that mix windows give outputs their own `fit_window_end`; a
  pre-origin use may then read only those outputs.
- **Exemptions (U3)** need `input`, `data_end`, `reason`, `ledger_number_id`, `sensitivity`
  and an explicit `uses` list (no `*`) drawn from `initial_conditions`, `idw_field`, `prior`,
  `prior_width_input`, `model_design`, `forcing_frozen`, `design_time_evidence` (never
  targets, fits or selection). The manifest must carry `non_exempt_fit_window_end`, and
  `fit_window_end` must equal the max of it and the exemptions' `data_end`. A pre-origin use
  passes only if `non_exempt_fit_window_end <= 2022-12` and every post-origin exempt input
  lists that use. Exemptions and time slicing do not combine.
- **The writer refuses unreachable uses**: a manifest that allows a pre-origin use the guard
  could never grant, or whose name says pre-origin (`preorigin`, `pre_origin`, `pre-origin`)
  with a post-origin window and no exemption coverage, fails validation.
- `read_table` always reads the `flag` column when the table has one, so a column subset
  cannot bypass U1/U2.

## Checks

```bash
cd dataprep_staging
python -m common.check                      # re-hash every manifest and SOURCE.json
python -m common.check --fail-on-orphans    # also fail on L0 files no record covers
python -m common.ledger assemble            # rebuild ledger/data_roles.csv from rows/*.csv
python -m pytest -q -p no:cacheprovider common/tests
```

Pre-dataprep `SOURCE.json` files (written before 2026-10-12 without sha256) are reported as
`legacy` and checked for presence and size; they are read-only and were not edited. Their
files, and every L0 file under `data_fetch_api/new_measurements/` that no SOURCE.json lists
(the 2026-09-30 vendor-API pulls, `cwa_codis_rain/`, `heads_202210_202610/`,
`compaction/`), are hashed in the L0 registry (`results/dataprep/l0_registry/v<N>/`), which
the checker re-hashes; files covered by neither are reported as orphans.

Ledger sources must be explicit (one file, directory, product version or named source per
row): glob and brace patterns are refused, and a directory conflicts with the files inside
it when their roles differ. `ledger.unledgered(number_ids)` lists reported numbers without
a row.

## Validation harness (WP A1)

**Second repair round (2026-10-12): use `prereg3`, `coverage3` and `gaps_guarded`, and
`validation_units/v3`** (data byte-identical to v2; the manifest documents the post-origin
pilot inputs as design-time labels). `prereg2` and `coverage2` are frozen too now (published
products hash them). Rules declared in `preregistration/A1r2.harness_rules.v1.json`.

```python
from harness import coverage3, prereg3
g = coverage3.coverage_gate(hits, clusters)   # rule gate-3; report g["calibrated"] too
prereg3.write_prereg(doc)                      # then prereg3.assert_can_score(id, wp, ...)
prereg3.record_look(units, scope, wp, purpose, members={...})   # members for own units
```

- **`prereg3`** parses a scope strictly into a union of month windows
  (`heads:2016-01-01..2018-12-31`, `2012-01..2015-12,2019-01..2022-12`); anything it does not
  fully understand (relational words, `2016-2018`, a second time segment) makes the scope
  unbounded, which overlaps everything. Blocking compares unit **membership** (wells,
  benchmarks, fold test sets, cells; a temporal unit spans every calibration well and active
  cell in its months), so a look on `fresh.km2027.0` blocks a later test on
  `fresh.rot_tercile.1`. 72 of the 105 pairs of fresh units overlap: register every test on
  fresh units before the first scoring look on any overlapping unit. External units need
  members in the registration (`{"description": ..., "ids"/"cells"/"window"/"none": ...}`).
- **`coverage3`** (rule gate-3): Wilson interval on n_eff from the one-sided 0.90 upper bound
  of the ICC (`upper90`), Student-t on k - 1 df, level 0.95, gate-2 window rule. It reaches
  0.89 actual coverage for k >= 20 clusters and ICC <= 0.7 (pre-registered
  A1r2.coverage3_confirm.v1, min 0.9094 over 205 configurations); strata with fewer clusters
  come back `calibrated: False` (pool them first or report the verdict as uncalibrated). The
  frozen `coverage` interval falls to 0.75-0.83 with 7-14 clusters at ICC 0.5-0.7.

**First repair round (2026-10-12): use `prereg2`, `coverage2` and `gaps_guarded`.** `prereg`,
`coverage`, `gaps`, `build_units`, `units` and `harness/__init__.py` are frozen because other
products and pre-registrations hash them; the repaired modules sit next to them and share the
same files (`INDEX.jsonl`, `looks.jsonl`, the ledger).

```python
import sys; sys.path.insert(0, "<repo>/dataprep_staging")
from harness import scorers, coverage2, placebos, gaps, gaps_guarded, units, prereg2
```

- **Pre-register before you score (repaired rules).** `prereg2.write_prereg(doc)` checks the
  latest `validation_units` registry by default (v2: fresh units by role; the withdrawn v1
  leveling ids refused; your own units go in `external_units`). A look on a fresh unit blocks
  every later test of that unit whose data scope overlaps (any purpose, test id or
  pre-registration id; scopes overlap unless both carry disjoint `YYYY-MM..YYYY-MM` windows;
  dataset names are not compared). `prereg2.assert_can_score(id, wp, test_id=...)` also
  refuses a changed input, another WP's registration, or a changed units registry, then logs
  the scoring look. `prereg2.record_look` refuses unknown pre-registration ids, test ids
  outside them, and the purpose `score`. `prereg2.audit()` applies the rules to the log.
- **Coverage gate (rule gate-2).** `coverage2.coverage_gate(hits, clusters)`: same interval
  as `coverage`; the 0.85-0.95 window applies only when the one-sided 0.95 lower bound
  `n_eff_lower` is at least 30; `coverage2.pool_small_strata` pools strata below that and
  no longer stops at a stratum without a neighbour. Report `n_eff`, `n_eff_lower`, `k` and
  the interval with every verdict.
- **Origin guards for index-based gaps.** `gaps_guarded.head_masks`, `replica_test_period`,
  `inject_offsets_steps`, `inject_gnss_steps` take the matrix calendar (`months`) and raise
  for post-origin columns or windows; `ring_windows`, `ring_path_years`, `leveling_withheld`
  check their dates. Masks and seeds equal `gaps`.

First-round description (still valid for `scorers`, `placebos`, `gaps`, `units`):

- **Score the same way.** `scorers.equal_well_rmse` (state `agg`: `"rms"` is the twin's,
  `"mean_of_rmse"` the gap-fill pilot's), `scorers.anomaly_rmse` (datum-free),
  `scorers.out_of_time_rmse` (training-mean datum; never a GLS datum),
  `scorers.coverage_native` and `scorers.conformal_split` / `conformal_crossfit_hits`,
  `scorers.recovery_metrics` (pseudo-truth recovery), `scorers.climatology_baseline`
  (refuses a post-origin fit), `scorers.sigma_obs` / `u2_weight` (U2 with the pre-origin
  rho(k) of A0).
- **Coverage gate (plan 4.4).** `coverage.coverage_gate(hits, clusters)`: cluster-adjusted
  Wilson interval on n_eff (ANOVA ICC), Student-t quantile on k - 1 df, level 0.95; pass if
  it contains 0.90, plus the 0.85-0.95 point window when n_eff >= 30; pool small strata
  first with `coverage.pool_small_strata`. With few clusters the interval is wide and the gate
  has little power (see the A1 report).
- **Artificial gaps.** `gaps.FAMILIES` lists every family of plan 4.4 with its seed rule
  (`1000*ord(family)+r`, `7000+r` for the test-period replica). Families whose plan wording
  reaches past 2022 raise unless `allow_post_origin=True` (after-the-fact analysis only).
- **Units.** Read `results/dataprep/validation_units/v<N>/units.json` through
  `guard.open_product(dir, "validator")` (or `"selection"`). Units with role `fresh` were
  declared in `preregistration/A1-fresh-units-v<N>.json`; units with role `design_time`
  carry a seeded look and can never be fresh.
- **Pre-register before you score.** `prereg.write_prereg(doc, units_registry=...)` writes
  `results/dataprep/preregistration/<id>.json` once (read-only), records its sha256 in
  `INDEX.jsonl` and your ledger fragment, and refuses test ids or fresh units that already
  carry a look. Call `prereg.assert_can_score(id, wp, test_id=...)` immediately before
  computing the registered score: it refuses a pre-registration that postdates a look at its
  test data, or whose file changed, and logs the scoring look. Log any other look at test
  data with `prereg.record_look` (an unregistered look taints the unit for later tests).

```bash
cd dataprep_staging
python -m pytest -q -p no:cacheprovider harness/tests
tools/cpu_slot.sh python -m harness.repro_pilots <evidence_dir>    # pilot metrics, 1e-6
```
