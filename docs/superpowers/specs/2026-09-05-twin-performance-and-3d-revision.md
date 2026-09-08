# Twin audit, performance revision, and the route to the 3D scenario viewer

**Date:** 2026-09-05, revised 2026-09-07 · **Status:** Stage-3 primary rule **PASSED**; the
k-fold secondary rule is running

> **Revision note (2026-09-07).** §1 and §2 below were written before the real cause was
> found, and their headline numbers are wrong in a way that matters. The solve was not
> merely "not GPU-bound" — **it was never on the GPU at all**, and on the real problem the
> GPU is ~51× faster, not the 1.1× a synthetic benchmark predicted. §1a and §2a record the
> correction; §1 and §2 are kept because the reasoning they contain is what led to looking
> at device placement in the first place, and a result log should show its own wrong turns.

Revisits the Stage-3 cost estimate that stalled the sub-project, on the premise that new
hardware and an upgraded NVIDIA stack would change it. The premise is half right, and the
half that is wrong matters more than the half that is right.

---

## 1. The ~144 h estimate was never GPU-bound

Spec `2026-08-29` §7.2 stopped the zonal sweep at a measured ~11.3× homogeneous per fold,
projecting ~7 days against 14 h budgeted. The natural reading — the card was too small —
is wrong. Measured on the Quadro RTX 6000 (Turing, sm_75), torch 2.11.0+cu128, at fan
scale (a synthetic 3,224-cell mask, 4 layers, 12,896 unknowns, float64):

| measurement | result | what it rules out |
|---|---|---|
| full rollout, GPU vs CPU | 1.11 s vs 1.25 s | **1.1×.** Not compute-bound — a GPU should crush a CPU here |
| matvec, float64 vs float32 | 257 µs vs 250 µs | not FP64-bound (Turing's 1:32 FP64 rate is irrelevant) |
| `index_add` (atomics) share | 15% of matvec | not atomics-bound |
| 257 µs for 12,896 unknowns | — | kernel-launch overhead, tens of tiny kernels per matvec |

The solve is **launch- and synchronisation-bound**. The problem is small and the kernels
are many; a bigger card does not help, and neither does a better dtype. The old 12 GB card
named in `grid.py`'s docstring was never the constraint.

This is the same pathology `results/bench_port/` already documents for the UDE, where
restructuring plus `torch.compile` bought 59×.

### The dominant single defect

`_cg` evaluated `if relres < tol` every iteration. That is a Python branch on a CUDA
tensor, so it forced a `cudaStreamSynchronize` **per CG iteration** — and the solves are
long (median **346 iterations**, measured on a heterogeneous fan-scale problem).

## 1a. CORRECTION — the solve was never on the GPU (2026-09-07)

`calibrate_flow` constructed its model as `FlowModel(grid, n_layers=4, dt_days=30.0)` at
both sites, with no `device=`. `FlowModel` honours the argument correctly, so `None` put
every parameter on CPU — and everything downstream derives its device from the model
(`dev = model.log_T.device`), so the whole run inherited CPU consistently. No error, no
warning. There was no `--device` flag, and `pick_device()` — present in `train.py` and used
by `bench_port.py` — was never wired into `twin/`. `calibrate_mlcw.py:50` has always
auto-selected CUDA, so the twin's two halves silently disagreed about hardware.

**Measured on the real problem** (2,148 cells × 4 layers, zonal, full pumping forcing):

| device | s/epoch | 1500 epochs |
|---|---|---|
| CPU | **3,452** | ~1,438 h |
| GPU | **67 → 34 avg** | **14.3 h (measured)** |

**~51×**, not the 1.1× §1's synthetic benchmark predicted. That benchmark is the methodological
lesson here: a 3,224-cell synthetic grid with randomised parameters mispredicted the real
system by nearly two orders of magnitude, because it did not reproduce the real fit's
CG-iteration profile. Every historical flow timing in this project — the 4.3 h fit, the
12,733 s/fold, the ~144 h sweep estimate, and the "we need a better GPU" conclusion drawn
from them — was produced by a CPU run caused by one omitted keyword argument.

Fixed: `--device` added (auto-selects CUDA), threaded through both construction sites and
into `kfold_wells`'s per-fold models, and the resolved device is printed at startup so this
cannot recur silently.

## 2a. CORRECTION — `_CG_CHECK_EVERY` is CUDA-only (2026-09-07)

§2's setting was validated on GPU and then applied globally, including to the CPU path it
cannot help: its entire purpose is avoiding `cudaStreamSynchronize`, which does not exist on
CPU, so there it could only run iterations *past* convergence. It is now gated on
`b.is_cuda`. Tests: 89 passed.

## 2. What was changed

| change | speedup | accuracy cost |
|---|---|---|
| `_CG_CHECK_EVERY = 25` — read the residual on a schedule | 1.38× | head/grad rel L2 ~1e-9, grad cosine 1.0000000000 |
| `--compile-matvec` — `torch.compile` the conductance matvec | 1.38× → **1.91×** | unchanged (~1e-9) |
| *rejected:* `_CG_TOL` 1e-8 → 1e-6 | 2.49× | head 6.6e-08, **grad 1.5e-07** |

The tolerance lever is left on the table deliberately. Spec `2026-08-29` §10 flags it as
"the obvious lever", and it is the largest single win available — but this solver feeds a
**pre-registered gate verdict**, and 2.49× is not worth perturbing the gradients that
verdict is computed from. It remains available for exploratory sweeps.

Checking every 25 costs ~4 extra iterations (350 vs 346), because the solves are long
relative to the check interval. The true-residual recompute and warning after the loop are
untouched, so a stalled solve still cannot pass silently. `_CG_CHECK_EVERY = 1` restores
exact per-iteration checking for bit-comparison against results recorded before this.

`cg_check_every` and `compile_matvec` are recorded in the run provenance alongside
`cg_maxiter` and `git_commit`, so a published number still carries proof of its settings.

Verification: `pytest -k "twin or flow or subsid"` → **105 passed, 6 skipped**, including
the Theis and Hantush-limit checks.

**Consequence.** §7.2's `--fit-only` route to the primary rule drops from ~4.3 h to ~2.25 h.
The full five-seed sweep drops from ~7 days to ~3.7 days — still not worth spending before
the in-sample rule has answered, exactly as §7.2 argues.

## 3. Data restored, with one honest discrepancy

`chou-shui-data/` (331 MB) is back. `AMP_V2/data/` was not, and was rebuilt from the
WiseEnvr API: `fan_stations.parquet` (344 zone-50 wells; screen-depth medians 53/119/210/282 m
over 103/134/60/19 wells in layers 1–4, matching §1's table) and `wells/` (344 per-`sid`
parquet, 2012–2023, fetched in yearly windows with recursive splitting against the API's
hard 20,000-row response cap).

**The rebuilt head field is not the recorded one.** `build_head_field` now yields **174
wells (L1 41, L2 86, L3 34, L4 13) at 8.79% NaN month-cells**, against the recorded **147
wells (34/69/31/13) at 1.1%**. Layer 4 matches exactly; every other layer gained wells, and
time coverage is sparser. The API is live and its holdings have moved since August. Any
Stage-1/Stage-2 number recomputed on this field should be expected to differ from the
recorded one, and the difference is the data, not the code.

### Stage 2 re-run: the upgrade is provably neutral, the refetched field is not

`calibrate_mlcw --epochs 2000`, on torch 2.11.0+cu128 with `_CG_CHECK_EVERY = 25`:

| metric | recorded (61 curated) | **re-run (61 curated)** | recorded (147 API) | **re-run (174 API)** |
|---|---|---|---|---|
| `sk_insample` | −0.298 | **−0.298** | +0.190 | +0.123 |
| `sk_loso` (baseline) | −0.556 | **−0.556** | +0.106 | +0.031 |
| `sk_coast_loso` | −4.000 | **−4.000** | −0.120 | −0.275 |
| `vep_loso` | −1.875 | **−1.875** | +0.070 | −0.064 |
| `vep_shared_loso` | +0.324 | **+0.324** | +0.478 | +0.465 |

The legacy column reproduces **every recorded digit**. That is the regression test for both
the torch 2.5.1+cu121 → 2.11.0+cu128 upgrade and the CG change: on identical inputs they
change nothing.

The API column moves, which isolates the cause to the refetched data alone. **The Stage-2
verdict is unaffected — the gate still PASSES**, and by a slightly wider margin than
recorded (+0.434 over baseline, against +0.372), because the baseline degraded more than the
rheology did. The sparser head field hurts the algebraic `Sk` more than the VEP, which is
consistent with §1's finding that head-field density was the binding constraint on `Sk`.

## 4. The 3D scenario viewer (`twin/explorer3d.py`)

Delivered against the restored data: four aquifer surfaces at their measured screen depths
under a deforming ground surface, animated over 132 months, as one self-contained Plotly
HTML.

Subsidence comes from the Stage-2 VEP column fitted as a single global parameter set
against the 14 MLCW magnetic-ring sites — the arm whose LOSO gate passes — then applied per
grid cell.

**Validated, not just rendered.** Scored against the WRA leveling network, which took no
part in the fit:

| grid | sites | pairs | R² | RMSE | bias |
|---|---|---|---|---|---|
| 2 km | 808 | 7,616 | **+0.266** | 8.1 cm | +2.6 cm |
| 1 km | 798 | 7,539 | **+0.241** | 8.3 cm | +3.1 cm |

For scale, Stage 1's algebraic `Sk` scored +0.214 on the same network (tilt-uncorrected).
The protocols are not identical — Stage 1 ran a pooled LOSO over `Sk`, this is a direct
forward application of an MLCW-fitted rheology with no leveling data used at all — so read
this as an independent positive result, not as a like-for-like win.

Scenario response is nonlinear in the right direction (1 km grid, final month):

| drawdown factor | mean | p95 | max | area >30 cm | area >50 cm |
|---|---|---|---|---|---|
| 0.5 | 18.9 cm | 24.0 | 33.0 | 6 km² | 0 km² |
| 1.0 (observed) | 23.1 cm | 33.3 | 51.2 | 214 km² | 1 km² |
| 1.5 | 27.2 cm | 42.6 | 69.4 | 738 km² | 50 km² |
| 2.0 | 31.4 cm | 51.9 | 87.6 | 1,041 km² | 131 km² |

Halving the drawdown does not halve the subsidence (18.9 vs 23.1 cm) — the preconsolidation
gate and the viscous term make the column path-dependent, which is the whole reason Stage 2
rejected the algebraic `Sk`.

### The limitation that decides what this can claim

**The scenario axis is head decline, not pumping rate.** Turning an abstraction rate into a
head field is precisely what the four-layer flow solver does, and its gate has not returned
a verdict — so no pumping → head map here is validated. `scenario_heads` is the seam: when
Stage 3 passes, replace the multiplier with the flow model's simulated head under a modified
abstraction, and every downstream component is unchanged.

## 5. Stage 3 — the pumping drivers, and the PRIMARY RULE VERDICT

### 5.1 The drivers were recoverable after all

An earlier draft of this section called the pumping data "not practically recoverable from
the API". That was wrong, and the error was mine: it rested on a single 504 that turned out
to be transient. `etc-tpc-etc1mon-obs` serves both halves fine —

- **census** (`tpc_pumps.parquet`): the dataset's *station* metadata, 116,769 rows carrying
  exactly `sid, TWD97_X, TWD97_Y, PUMP_HP, PURPOSE`. 99.7% fall inside the fan mask.
- **kWh** (`pump_kwh_all.parquet`): 223 monthly rows per pump, `datetime` +
  `electricity_kwh`. Fetched in ~3 h at 10 workers → **26,039,264 rows, 116,768 pumps,
  2007-01 → 2025-07, 16.23 TWh total, zero failures.**

Two fetcher bugs worth recording because both are generic: the bearer token expires and
`demo.py` never renews it (the first run died at 401 then "succeeded" at failing 2,394
pumps in a minute), and resume state that logs *attempted* rather than *succeeded* items
will permanently skip whatever failed during an outage. Resume state is now derived from
the output shards themselves, which by construction contain only real data.

### 5.2 PRIMARY RULE: **PASS** — the clamp released (2026-09-06)

`--param-mode zonal --fit-only --epochs 1500 --device cuda --compile-matvec`, 14.3 h:

```
wells=158  cells=2148  dx=1000m  n_params=26  forcing=on
in-sample R2=+0.760   fit_time=51,625s
bounds_hit[proximal]={log_T: lo=0/1 hi=0/1, log_S: lo=0/1 hi=0/1}
bounds_hit[mid]     ={log_T: lo=0/4 hi=0/4, log_S: lo=0/4 hi=3/4, log_L: lo=0/3 hi=1/3}
bounds_hit[distal]  ={log_T: lo=3/4 hi=1/4, log_S: lo=0/4 hi=1/4, log_L: lo=0/3 hi=0/3}
cg_maxiter=2000  cg_nonconverged=0  cg_worst_residual=0.000e+00
```

Against the rule pre-registered before any zonal number existed:

| condition | measured | |
|---|---|---|
| pooled lower-clamp `log_T` ≤ 4/9 | 0/1 + 0/4 + 3/4 = **3/9** | ✅ |
| proximal `log_T` not at lower clamp | **0/1**, free | ✅ |

**Clamp released. PASS.** The proximal zone fits `log_T` 6.585 → **T = 725 m²/day**, inside
the 58–6,034 m²/day Liu et al. 2002 measured on this fan. The pin below the 58 m²/day floor
— the strongest evidence against the parameterisation, and the finding that survived the
2026-08-27 retraction — is gone. Zoning transmissivity did what §4 predicted it would.

Three supporting facts:

- **`cg_nonconverged=0`, worst true residual exactly `0.000e+00`** across a 14 h fit. Given
  this solver previously produced gradients with a median true residual of 4.955e-02, a
  clean provenance line is what makes the number publishable.
- **`PLATEAUED (structural)`** — R² converged by epoch **250** (+0.769) and drifted to +0.760
  by 1500. The earlier FAILs cannot be attributed to training budget.
- **R² +0.760 < the old +0.943.** That is the right direction: the retraction records that
  +0.943 "was bought by saturating parameters against their bounds". A lower in-sample fit
  with parameters inside physical bounds is the better model.

**Caveat carried forward.** 3 of 4 distal `log_T` remain at the lower clamp and one at the
upper; `log_S` is at its upper bound in 3 of 4 mid cells. The rule counts only lower-clamp
hits and 3/9 clears 4/9, but the distal zone is still straining and that belongs in any
write-up.

### 5.3 SECONDARY RULE: running

§6 reaches the margin (flow vs IDW on held-out wells) only because the clamp released.
Running: 5 folds, seed 0, `--dump-predictions`, GPU.

**Protocol deviation, declared:** `--epochs 500`, not 1500, justified by the measured
plateau at epoch 250 (2× margin) and cutting ~70 h to ~35 h. It slightly favours the flow
arm, since IDW has no epoch budget — so if the margin lands close, re-run at 1500 before
publishing. `--dump-predictions` also lets the **co-location rate** be verified empirically,
which is what invalidated the 2026-08-27 verdict (95 of 136 held-out entries at zero
distance from a training entry).

## 6. Revised staging

1. ~~Solver performance~~ — done and validated.
2. ~~Restore the pumping drivers~~ — done; 26.0M kWh rows, 116,768 pumps, zero failures.
3. ~~**Stage-3 primary rule**~~ — **PASSED** (§5.2). Clamp released.
4. **Stage-3 secondary rule** — running (§5.3). The margin against IDW.
5. **On PASS — Stage 4 coupling.** `twin/coupled.py` is already built and tested
   (§4a): flow → layer heads → driver → VEP → subsidence, with a subsidence loss reaching
   `log_T` through the implicit adjoint. Warm-start the column from Stage 2, the flow from
   §5.2, then fine-tune jointly per the spec's staging rule.
6. **Then the forward twin.** Swap `explorer3d`'s `scenario_heads` multiplier for a
   `CoupledTwin` run driven by `twin/scenario.PumpingScenario`, and the viewer becomes a
   genuine on-demand pumping-scenario twin rather than a head-space proxy.

**If the secondary rule FAILs — re-parameterise, do not force.** Candidates unchanged: a
transmissivity prior from the Kassie et al. 2023 TEM survey (which §10 already names for
the mid/distal boundary, itself still unjustified), or a PhysicsNeMo FNO surrogate trained
on the classical solve. Note that the primary rule passing means the *parameterisation* is
no longer the prime suspect — a secondary FAIL would point at the zoning geometry or the
head field, not at `log_T`'s bounds.

## 6a. What was built while the gate ran

- `twin/coupled.py` — Stage-4 coupling (see §6.5). 9 tests, including the one that matters:
  a subsidence-only loss produces non-zero, finite gradients on `log_T`/`log_S`/`log_L`.
- `twin/scenario.py` — pumping policies rather than drawdown multipliers. The 23 raw
  `PURPOSE` labels map to six policy classes; **irrigation is 86% of the 457,750 installed
  HP inside the fan**, so a scenario that does not touch irrigation does not move the fan.
  Supports per-class factors, zone restriction, and a start date, plus `climatology()` for
  forcing a run past the end of the record. 12 tests.

## 7. Environment

torch 2.11.0+cu128 (from 2.5.1+cu121), CUDA 12.8, cuDNN 9.19, PhysicsNeMo 2.2.1,
warp-lang 1.17.0, driver 535.230.02 unchanged. The cu128 wheels ship sm_75 and run on
driver 535 via CUDA minor-version compatibility. Forecaster throughput across the upgrade
was flat on GPU (24.81 s → 24.61 s) — the upgrade was an access fee for PhysicsNeMo, not a
speedup. See `~/bench_baseline/COMPARISON.md`.

PhysicsNeMo is installed but **not yet used by any twin code path**. Its first candidate
use is item 4 above.
