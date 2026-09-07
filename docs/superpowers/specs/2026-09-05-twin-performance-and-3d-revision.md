# Twin audit, performance revision, and the route to the 3D scenario viewer

**Date:** 2026-09-05 · **Status:** performance work done and validated; Stage 3 still blocked

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

## 5. Stage 3 remains blocked — on data, not on compute

The gate needs the pumping drivers, and they are absent: `AMP_V2/data/pump_kwh_all.parquet`
and the TPC pump census (whose CLI default still points at a dead session scratchpad,
`calibrate_flow.py:900-904`). Neither is in `chou-shui-data/`.

They are **not practically recoverable from the API**: `etc-tpc-etc1mon-obs` exposes no bulk
endpoint (four candidate paths all 404), a single pump's monthly series returned 504, and
per-station fetching over 116,769 pumps is ~65 h at the observed rate. `--no-forcing` exists
but the module docstring calls that configuration degenerate, so it is not a gate run.

**This is the single blocker on the whole downstream chain**, and the fix is one file pair
from the original workstation or a backup.

## 6. Revised staging

1. ~~Solver performance~~ — done, 1.91×, validated.
2. **Restore the pumping drivers.** Blocking; nothing below can start.
3. **Stage-3 `--fit-only`** (~2.25 h) → the pre-registered rule: clamp released iff pooled
   lower-clamp `log_T` ≤ 4/9 **and** the proximal zone's `log_T` is not at the lower clamp.
4. **On FAIL — re-parameterise, do not force.** The pinned `log_T = log 10` sits below the
   58 m²/day floor Liu et al. 2002 measured here; that is evidence against the
   parameterisation, not against the physics. Candidates: a transmissivity prior from the
   Kassie et al. 2023 TEM survey (§10 already names it for the mid/distal boundary), or a
   PhysicsNeMo FNO surrogate trained on the classical solve.
5. **On PASS — Stage 4 coupling**, then swap `scenario_heads` for the flow model and the
   viewer becomes a genuine pumping-scenario twin.

## 7. Environment

torch 2.11.0+cu128 (from 2.5.1+cu121), CUDA 12.8, cuDNN 9.19, PhysicsNeMo 2.2.1,
warp-lang 1.17.0, driver 535.230.02 unchanged. The cu128 wheels ship sm_75 and run on
driver 535 via CUDA minor-version compatibility. Forecaster throughput across the upgrade
was flat on GPU (24.81 s → 24.61 s) — the upgrade was an access fee for PhysicsNeMo, not a
speedup. See `~/bench_baseline/COMPARISON.md`.

PhysicsNeMo is installed but **not yet used by any twin code path**. Its first candidate
use is item 4 above.
