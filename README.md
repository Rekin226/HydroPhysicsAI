# HydroPhysicsAI

**A differentiable digital twin of the Choushui alluvial fan, Taiwan: pumping policy in, groundwater heads and land subsidence out, gated against held-out data at every stage.**

[![CI](https://github.com/Rekin226/HydroPhysicsAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Rekin226/HydroPhysicsAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20PhysicsNeMo%20%7C%20CUDA-76b900.svg)](#how-it-works)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Project state](docs/superpowers/STATE.md) · [Data](docs/DATA_FORMAT.md) · [Technical writeup](docs/TECHNICAL_WRITEUP.md) · [Model card](MODEL_CARD.md) · [Earlier phase](docs/EARLIER_PHASE.md)

![Hindcast and projected subsidence over the fan, with the policy ensemble](results/figures/twin_overview.png)

---

## Overview

The Choushui fan is Taiwan's largest groundwater basin and its most serious subsidence
problem: unmetered irrigation pumping from four stacked aquifers, and a ground surface
sinking centimetres a year under the high-speed-rail corridor.

This repository is a physics-based twin of that system. A four-layer groundwater solver
on a 1 km grid is driven by the registered-pump electricity census and rain-minus-ET
recharge, coupled to a visco-elasto-plastic compaction column, and calibrated by gradient
descent through the solver's own adjoint. It takes a pumping policy, runs the aquifer
forward, and animates where and how fast the ground sinks, with a parameter ensemble
around every number.

```
pumping policy ──▶ flow solver ──▶ layer heads ──▶ compaction column ──▶ subsidence ──▶ 3D viewer
```

## Results

| | gate | result |
|---|---|---|
| **Flow model** | held-out wells, 5 site-grouped folds, must beat inverse-distance interpolation | **PASS**, R² +0.804 vs +0.702, with a physical pump conversion and a 10 km stress radius |
| **Compaction column** | 798 leveling benchmarks, site-grouped 5-fold | **+0.546** out of fold, bias +0.1 cm |
| **Full chain hindcast** | 798 leveling sites, 36-member ensemble | R² **+0.579**, RMSE 6.2 cm |
| **Projection 2023–2032** | fan-mean subsidence ± ensemble | 10.7 ± 0.6 cm baseline · 9.9 with irrigation cut 30 % · 9.0 with aquaculture retired |
| **Re-run cost** | 18 members × 3 policies × 21 years | 609 s on one GPU; FNO surrogate 13–34× faster again |

Every verdict, including the two failed configurations that preceded the pass, is
recorded in [`docs/superpowers/STATE.md`](docs/superpowers/STATE.md).

**Two caveats govern the policy numbers.** First, the flow gate holds out *wells*: it
validates spatial interpolation under the recorded forcing. A gate that holds out
*years* (fit to 2019, free-run 2020–2022) finds that the model's own dynamics drift
within three years, worse than climatology on per-well anomalies. Projections are
therefore anchored by nudging to observations at the origin, and their decade-scale
trend is the model's, not yet validated. Second, the gated model's pumping stress is
physical (efficiency 0.5, 40 m extra head, irrigation return flow) only because each
cell's electricity is spread over a learned radius that sits at its 10 km bound; the
earlier free fit passed by switching the stress off. Both are stated with numbers in
the state doc.

## Data

All real, on the 2,144 km² fan, 2012–2022, monthly. Only a synthetic sample ships with
the repository; the cache is rebuilt from the WiseEnvr API with
`hydrophysics.twin.fetch_amp` (credentials from the environment only).

| source | size | role |
|---|---|---|
| Water Resources Agency monitoring wells | 174 wells, 4 aquifers, hourly | heads: calibration target and assimilation |
| Taiwan Power Company pump census + electricity | 116,769 pumps, 26 M meter-months | abstraction forcing |
| Rain gauges + ERA5 ET₀ | 26 gauges, daily | recharge |
| Leveling benchmarks | 798 sites, 7,539 surveys | subsidence calibration and validation |
| Multi-layer compaction wells | 14 sites | independent compaction check |

## Quickstart

```bash
pip install -e ".[gpu,viz,explorer,dev]"
export HYDROMIND_GW_DATA=$PWD/chou-shui-data/data

# calibrate the flow model and run its gate (GPU, ~35 h)
python -m hydrophysics.twin.calibrate_flow --param-mode zonal --epochs 500 --n-folds 5 \
    --device cuda --compile-matvec --out results/twin_runs/my_run

# calibrate the compaction column on the gated heads (leveling target)
python -m hydrophysics.twin.calibrate_coupled --theta results/twin_runs/my_run/stage3_theta.json \
    --target leveling --configs shared,zonal --out results/twin_runs/my_run/coupled

# run a policy forward with the fold ensemble, and render it
python -m hydrophysics.twin.forward --theta results/twin_runs/my_run/stage3_theta.json \
    --theta results/twin_runs/my_run/stage3_fold_thetas.json \
    --scenario "cut30:irrigation=0.7@2026-01" --horizon 120 \
    --vep-json results/twin_runs/my_run/coupled/vep_zonal_leveling.json --out results/twin_forward/cut30
python -m hydrophysics.twin.explorer3d --forward-npz results/twin_forward/cut30.npz \
    --out results/twin/explorer3d_forward.html
```

`pytest -q` runs the test suite on CPU in about 13 minutes.

### The application

```bash
python -m hydrophysics.twin.viewer_app --forward results/twin_forward/<run>.npz \
    --basis results/twin_forward/response_basis.npz --out results/twin/twin_app.html
```

`results/twin/twin_app.html` is the twin as something to operate: one self-contained page,
about 8 MB, nothing to install. The fan is a block model, four aquifers at their per-zone
screen depths with the clay aquitards between them, standing on the real ground surface,
and the ground descends as the record plays. A slider per water-use class sets the pumping
policy and the field updates as you drag; the response comes from a precomputed basis, one
solver run per class, combined by superposition, which reproduces a solved half-cut to
within 0.06 cm and a four-year delay to within 0.02 cm. Click any cell for its heads,
its sinking and the ensemble spread. Analyst mode adds layer toggles, head and drawdown
colouring, an exploded view, a west-to-east cut-away and the vertical exaggeration.

Four panels sit beside the block. **Cell** reads one location. **Section** cuts a line
between two clicked points and draws the vertical profile: the aquifers and clay
aquitards at their real depths, the head in each, and the subsidence along the transect.
**Compare** puts your policy and a reference side by side as maps, reports the difference
in centimetres, and can recolour the block by that difference. **Townships** is a sortable
report card for the twenty townships on the fan. Both table and section export as CSV,
and the view saves as an image.

The older Plotly viewer (`explorer3d`) still builds if you want a quick figure.

## How it works

- **Solver.** Five-point finite volume, backward Euler, four aquifers with leakance between
  them, general-head boundaries on the coast and the mountain front. The linear solve is
  matrix-free conjugate gradient in float64; gradients come from an implicit-function
  adjoint, so memory does not grow with the rollout.
- **Forcing.** Electricity converts to abstraction through the pump's own hydraulics, with
  the lift taken from the model's evolving head. The census is de-duplicated across shared
  meters and screened against rated motor capacity. Recharge is rain minus ET₀.
- **Calibration and gates.** Zonal parameters (proximal, mid, distal) fitted by Adam
  through the adjoint. The flow gate is site-grouped k-fold against inverse-distance
  interpolation; the column gate is site-grouped k-fold over the leveling network.
- **Forward twin.** Hindcast, observation nudging at the forecast origin and optionally
  through the record, climatological forcing under named policies, and an ensemble over
  fold fits, Laplace-posterior draws and perturbed initial fields.
- **Surrogate.** A PhysicsNeMo Fourier neural operator trained on the calibrated solver,
  for policy sweeps and large ensembles.

Modules live under `hydrophysics/twin/`; each file's docstring states what it does and
why.

**Hardware.** Everything here runs on a single GPU server: one NVIDIA Quadro RTX 6000
(Turing, sm_75, 24 GB), torch 2.11 on CUDA 12.8, PhysicsNeMo 2.2.1, Ubuntu, long jobs in
tmux. Calibration takes about 35 hours per gate; a policy run takes minutes.

The solver is float64 throughout and uses no mixed precision, because the conjugate-
gradient solve needs the precision, not because of speed. Measured on this card with a
4096-cube matmul:

| precision | TFLOP/s | used for |
|---|---|---|
| fp16 | 61.9 | nothing (accuracy) |
| fp32 | 9.1 | the FNO surrogate |
| bf16 | 5.3 | nothing: Turing has no bf16 tensor cores, so it is slower than fp32 |
| fp64 | 0.32 | the flow solver |

The flow solver still gains about 51× over CPU despite running in the slowest precision,
because it is bound by kernel launches rather than arithmetic. A newer card would help
the surrogate far more than the solver. Stack decisions and the full constraint list are
in `docs/GPU_SERVER.md`.

## Limits

- A free-running continuation drifts within three years (held-out-years gate: RMSE
  6.6 m at the wells against 2.0 m for climatology), so the twin is a hindcast-and-nudged-
  projection tool, not a free forecaster. Fitting anomalies rather than levels is the next
  calibration change.
- The pumping stress has to be spread over about 10 km for the model to generalise,
  which is wider than a well's drawdown cone. It is standing in for something not yet
  modelled, probably that the census locates meters rather than wells. Widening it to
  21 km scores marginally better on the head gate and erases the policy response, so the
  head gate alone is not a sufficient selection criterion. Policy sensitivities are
  consequences of a gated model, not validated forecasts.
- The mid-zone viscous time constant reaches the length of the record; decadal creep is
  bounded by the calibration window.
- The posterior is a local Laplace approximation; parameters at a bound are held, not
  sampled.

## Earlier phase

The project began (2026-06) as a benchmark of one physics-informed neural operator across
61 curated wells against per-well gray-box ODEs, with an assimilated forecaster, a PINN
head field and a 2D explorer. That work is kept in full in
[`docs/EARLIER_PHASE.md`](docs/EARLIER_PHASE.md). It is no longer the headline: the twin
runs on a denser, layer-resolved dataset with the pumping forcing and the subsidence
network the operator benchmark never had, and its gates are physical rather than a
comparison with a baseline scored elsewhere.

## Repository

```
hydrophysics/twin/       the twin: grid, boundaries, flow, pumping, compaction, calibration,
                         forward, uncertainty, surrogate, explorer3d, fetch_amp
hydrophysics/            earlier-phase models, baselines, metrics, forecaster, explorer
docs/superpowers/        design specs, plans and the state ledger
results/twin/            published gate results and the rendered viewer
tests/                   CPU-only test suite
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup,
the GPU install and the evaluation rules, and the
[issues](https://github.com/Rekin226/HydroPhysicsAI/issues) for claimable work.

## Author and license

Abdoul Rachid Ouedraogo, Ph.D., hydrogeology and AI. Also: [AquaScope](https://github.com/Rekin226/aquascope).
MIT, see [LICENSE](LICENSE).
