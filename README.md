# HydroPhysicsAI

**GPU physics-informed neural operators for groundwater — one model across many wells, benchmarked against per-well gray-box ODEs.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20PhysicsNeMo%20%7C%20CUDA-76b900.svg)](#nvidia-gpu-path)

---

## The idea

Classical hydrology calibrates **one ODE per well**: 33-61 separate parameter fits, each blind to the others. HydroPhysicsAI trains **a single physics-informed neural operator across all wells at once**, conditioned on each well's static attributes, on the NVIDIA GPU stack (PyTorch / PhysicsNeMo / CUDA). It is scored in true **simulation mode** (free-running hindcast from an initial condition + forcing, never seeing observed levels) against the per-well gray-box ODE baseline.

> One GPU-trained physics-ML operator, across all 61 wells, aiming to match or beat 61 hand-calibrated ODEs — and run in milliseconds, generalize to wells it never saw, and carry calibrated uncertainty.

Test bed: 61 groundwater monitoring wells on the Zhuoshui alluvial fan, Taiwan, 2012-2022, validated out-of-sample from 2019.

## Why physics-informed, not a black box

The model keeps the gray-box mass-balance ODE (recession + rainfall + upstream coupling + seasonal terms) as its inductive bias, and learns a neural **hypernetwork** that maps well attributes to the ODE parameters. So it stays interpretable (read off `a`, `b`, `k_link` per well), extrapolates better than a pure sequence model, and amortizes: one network conditions on attributes, so it can predict a well it was never calibrated on. That leave-one-well-out generalization is the operator-learning headline.

## Benchmark (simulation mode, validation period)

The bar to beat, computed by the harness in this repo:

| Model | KGE (median) | NSE (median) | RMSE m (median) | Wells |
|---|---|---|---|---|
| **Gray-box ODE** (per-well calibrated, baseline) | **0.736** | — | 0.83 | 61 |
| Climatology (per-well seasonal mean) | 0.446 | -0.20 | 1.63 | 61 |
| Last-value (constant) | undefined* | -0.57 | 1.83 | 61 |
| **Physics-informed neural operator** (this project) | *training on GPU* | | | |

<sub>*A constant prediction has zero variance, so KGE is undefined; skill is read from NSE/RMSE. Forecast-mode persistence scores KGE 0.997 but is deliberately excluded: 1-step-ahead prediction of slow-moving groundwater is trivial and not comparable to a free-running simulation. The harness keeps simulation and forecast modes separate so the comparison stays honest.</sub>

![Per-well gray-box KGE across the Zhuoshui fan](results/phase0/spatial_kge_graybox.png)

*Per-well validation KGE across the fan (gray-box baseline). Most wells are well-modeled (green/yellow); the dark wells are the hard cases a single attribute-conditioned operator should rescue.*

## Status

- **Foundation (done, tested, runs anywhere):** dataset loader, KGE/NSE/RMSE metrics with explicit simulation-vs-forecast modes, gray-box + climatology + last-value baselines, reproducible benchmark, synthetic sample for CI.
- **Models (in progress, GPU):** a working `GlobalGRU` reference model, and the `PhysicsUDE` physics-informed operator skeleton wired to the harness. The UDE forward integration and the PhysicsNeMo port are the active build (see TODOs in `hydrophysics/models/ude.py`).

## Quickstart

```bash
pip install -e .                 # foundation only (numpy/pandas)
pip install -e ".[gpu]"          # + torch, torchdiffeq (on the CUDA machine)
pip install -e ".[gpu,viz,dev]"  # everything

# Reproduce the baselines on the bundled synthetic sample (no real data, no GPU):
python -m hydrophysics.run_baselines

# Train + benchmark a model on the sample:
python -m hydrophysics.train --model gru --epochs 30

# On real data + GPU:
export HYDROMIND_GW_DATA=/path/to/data
python -m hydrophysics.train --model ude \
    --baseline /path/to/gw_fit_results.csv --out results/ude --epochs 300
```

The real Zhuoshui groundwater data is **not** redistributed here (agency-data terms). The repo ships a synthetic sample in `hydrophysics/sample_data/` and reads real data via the `HYDROMIND_GW_DATA` path.

## NVIDIA GPU path

The flagship result targets the NVIDIA stack:

1. **CUDA training.** `train.py` auto-selects `cuda > mps > cpu`. The models are standard PyTorch, so they train on any CUDA GPU as-is.
2. **Differentiable ODE.** Swap the prototype Euler loop in `PhysicsUDE._rollout` for `torchdiffeq.odeint_adjoint` for constant-memory backprop through long rollouts.
3. **PhysicsNeMo.** Port the hypernetwork + integration to NVIDIA PhysicsNeMo to use its physics-ML utilities, mixed precision, and multi-GPU training, and to add a physics-residual loss. This is what converts "scientific ML" into "scientific ML on NVIDIA's framework".
4. **Operator generalization.** Evaluate leave-one-well-out: train on N-1 wells, predict the held-out well from its attributes alone.

## Project structure

```
hydrophysics/
  config.py        data-path resolution (HYDROMIND_GW_DATA env or bundled sample)
  data.py          GWData loader: daily, well-aligned forcing + target + attributes + splits
  metrics.py       KGE / NSE / RMSE (NaN-safe, zero-variance-safe)
  baselines.py     gray-box + climatology + last-value (sim) + persistence (forecast)
  eval.py          benchmark_table + per-well scores + spatial KGE map
  sample.py        synthetic dataset generator (CI / no-data users)
  run_baselines.py freeze + print the baseline tables
  train.py         load -> fit -> simulate -> benchmark (the GPU entry point)
  models/
    base.py        GroundwaterModel interface (fit + simulate)
    gru.py         GlobalGRU reference model (working, GPU-ready)
    ude.py         PhysicsUDE physics-informed operator skeleton (the new method)
results/phase0/    frozen baselines + spatial map
tests/             foundation + model-smoke tests
```

## Author

Abdoul Rachid Ouedraogo, Ph.D. — hydrogeology x AI. Also: [AquaScope](https://github.com/Rekin226/aquascope).

## License

MIT — see [LICENSE](LICENSE).
