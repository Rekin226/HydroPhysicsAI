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
| **Compaction column** | 798 leveling benchmarks, site-grouped 5-fold | **+0.589** out of fold, bias +0.1 cm |
| **Full chain hindcast** | 798 leveling sites, 36-member ensemble | R² **+0.579**, RMSE 6.2 cm |
| **Projection 2023–2032** | fan-mean subsidence ± ensemble | 4.5 ± 0.8 cm baseline · 3.6 with irrigation cut 30 % · 2.7 with aquaculture retired (all 18 parameter sets agree on both effects) |
| **Creep uncertainty** | same, with a 30-year creep ceiling (equal leveling skill) | baseline 8.2 cm; the policy effects move by under 0.1 cm |
| **Re-run cost** | 18 members × 3 policies × 21 years | 609 s on one GPU; FNO surrogate 13–34× faster again |

Every verdict, including the two failed configurations that preceded the pass, is
recorded in [`docs/superpowers/STATE.md`](docs/superpowers/STATE.md).

An earlier projection of 10.7 cm was about half artefact: a column start-up release in
the proximal zone and a restart step at the forecast origin. Both are fixed; hindcast skill
is unchanged (state doc §0).

**Two caveats govern the policy numbers.** First, the flow gate holds out *wells*: it
validates spatial interpolation under the recorded forcing. A gate that holds out
*years* (fit to 2019, free-run 2020–2022) fails for every model tried. Most of that error
is a per-well level offset already present in the fitted years, concentrated in the
proximal fan; after removing a per-well datum the model is still 1.5–2x worse than
climatology plus trend. The decade-scale trend is the model's, not yet validated. Second, the gated model's pumping stress is
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
| NLSC tile service | orthophoto and topographic base, 252 tiles at zoom 13 | the ground the viewer recognises |
| SRTM 1 arc-second | terrain over the fan, −5 to 194 m | real elevation, replacing interpolated well collars |

The last two are fetched by `python -m hydrophysics.twin.basemap`; both are public and need
no credentials. Interpolating 279 well collar elevations had put the fan a median 5 m too
low and as much as 83 m out at the foothills.

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
python -m hydrophysics.twin.viewer_app --forward results/twin_forward/physical_spread_fixed.npz \
    --basis results/twin_forward/response_basis.npz --out results/twin/twin_app.html
```

`results/twin/twin_app.html` is the twin as a decision page: one self-contained file of
about 1.8 MB, nothing to install (spec: `docs/superpowers/specs/2026-09-23-twin-decision-app-redesign.md`).
It opens on the answer. A generated headline and six tiles compare the current policy with
business as usual: subsidence avoided by 2032 with its 36-run range, area sinking faster
than a threshold, the high-speed-rail gradient, layer-2 head recovery, the pumping energy
given up, and how many runs agree. Below them sit a 2D map of the policy's change from
baseline, drawn on a fixed symmetric scale, and a linked time series. The series marks the
fitted, tested and not-validated periods and has a policy-minus-baseline panel. A slider per
water-use class and a 2026/2030 start set the policy. When the policy matches a solved run
the page uses the full 36-run fields; otherwise it builds the result from the per-class
response basis and prints that fast estimate's error. Tabs hold township small multiples
(low-confidence townships flagged from the leveling support), the THSR profile, a
cross-section, pinned-policy comparison and "where the model is trusted". The 3D exploded
block is a drawer, built only when opened. A five-step story mode guides first-time
readers, and the page offers EN/中文, keyboard control, table views and CSV/PNG export.
The known limits from `docs/superpowers/STATE.md` are named next to the numbers they
qualify. Rail and river lines are OpenStreetMap traces (`python -m hydrophysics.twin.app.geo`,
committed under `hydrophysics/twin/app/geodata/`).

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

- The held-out-years gate fails (RMSE 6.6 m at the wells against 2.0 m for climatology;
  1.5–2x the climatology-plus-trend baseline after a per-well datum). The error lives in the
  proximal fan, whose deep layers have one well between them; its merged-aquifer
  representation is the next structural change. Slow storage, canal deliveries and river
  boundaries were each screened and none closes the gap.
- The pumping stress has to be spread over about 10 km for the model to generalise,
  which is wider than a well's drawdown cone. It is standing in for something not yet
  modelled, probably that the census locates meters rather than wells. Widening it to
  21 km scores marginally better on the head gate and erases the policy response, so the
  head gate alone is not a sufficient selection criterion. Policy sensitivities are
  consequences of a gated model, not validated forecasts.
- The mid-zone viscous time constant reaches the length of the record; a 30-year ceiling
  fits leveling equally well and adds 3.6 cm to the baseline projection (shown as a range).
- The per-zone column produces a subsidence step at the 182 km mid/distal line that the
  leveling data do not show; the app marks it.
- The posterior is a local Laplace approximation (`--bounded truncnorm` samples parameters
  at a bound).

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
