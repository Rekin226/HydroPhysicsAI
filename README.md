# HydroPhysicsAI

**GPU physics-informed neural operators for groundwater — one model across many wells, benchmarked against per-well gray-box ODEs.**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20PhysicsNeMo%20%7C%20CUDA-76b900.svg)](#nvidia-gpu-path)

---

## Results at a glance

All on the real 61-well Zhuoshui data, out-of-sample validation from 2019. Median KGE (higher is better) unless noted.

| Task | This repo | Reference | Verdict |
|---|---|---|---|
| **Simulation** (free-running hindcast) | physics-UDE **0.591** | gray-box 0.736 · climatology 0.446 | beats climatology, trails the per-well gray-box |
| **Generalize to unseen wells** (leave-one-well-out) | **0.236** | climatology 0.446 | open problem — does not yet generalize |
| **Forecast, 7-day** (operational, assimilated) | LSTM **0.967** | persistence 0.946 | real skill over persistence |
| **Forecast, 30-day** | LSTM **0.897** | persistence 0.703 | nearly halves the error |
| **Forecast, probabilistic** | CRPS beats persistence, ~90% calibrated | — | sharp, well-calibrated intervals |

Simulation and forecast modes are scored **separately** and are not comparable (forecasting with assimilation is a different, easier task). The headline scientific challenge — one attribute-conditioned operator generalizing to wells it never saw — is the leave-one-well-out row, and it is still unsolved.

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
| **Physics-informed neural operator** (this project) | 0.591 | 0.49 | 1.07 | 61 |

<sub>*A constant prediction has zero variance, so KGE is undefined; skill is read from NSE/RMSE. Forecast-mode persistence scores KGE 0.997 but is deliberately excluded: 1-step-ahead prediction of slow-moving groundwater is trivial and not comparable to a free-running simulation. The harness keeps simulation and forecast modes separate so the comparison stays honest.</sub>

The operator is one network across all 61 wells (attribute-conditioned hypernetwork to ODE parameters, level anchored to each well's training mean, per-well-weighted data + physics-residual loss, 1500 epochs on CUDA). It clears the seasonal climatology baseline comfortably (0.591 vs 0.446 KGE, 0.49 vs -0.20 NSE) and beats the per-well-calibrated gray-box on 18 of 61 wells, but still trails it overall (0.591 vs 0.736). All hyperparameters were chosen on an inner split carved from the pre-2019 training period (inner-train <2018, inner-val 2018); the 2019+ benchmark was evaluated exactly once, so the number is not tuned to the test. The remaining gap is a few hard wells, and the operator-generalization headline (leave-one-well-out: predict a held-out well from its attributes alone) is the next milestone.

![Per-well gray-box KGE across the Zhuoshui fan](results/phase0/spatial_kge_graybox.png)

*Per-well validation KGE across the fan (gray-box baseline). Most wells are well-modeled (green/yellow); the dark wells are the hard cases a single attribute-conditioned operator should rescue.*

## Forecast mode (operational, data-assimilated)

A **separate** track from the simulation benchmark above (do not compare the two — different task). Here a single global attribute-aware LSTM (`hydrophysics/models/forecast_lstm.py`) forecasts the level `h` days ahead using observed levels up to the forecast origin (assimilation) plus forcing, scored on 2019+ against the honest forecast-mode references at each horizon:

| Horizon | LSTM KGE | Persistence KGE | LSTM RMSE m | Persistence RMSE m |
|---|---|---|---|---|
| 1 day | 0.995 | 0.997 | 0.08 | 0.10 |
| 7 days | **0.967** | 0.946 | **0.32** | 0.44 |
| 30 days | **0.897** | 0.703 | **0.55** | 1.08 |

<sub>Median over 61 wells; climatology scores ~0.45 KGE at every horizon. The LSTM adds real skill over persistence at 7 and 30 days (at 30 days it nearly halves the error); the 1-day row is near-trivial for both and shown only for context. Hyperparameters (learning rate) were tuned on an inner pre-2019 split (inner-train <2018, inner-val 2018), then evaluated once on 2019+. Reproduce: `python -m hydrophysics.forecast_eval --device cuda --horizons 1,7,30`.</sub>

**Probabilistic forecasts.** With `--probabilistic` the LSTM emits a Gaussian per horizon (mean + variance, trained by Gaussian NLL), so each forecast is a calibrated distribution — read off any prediction interval or exceedance probability for early warning.

| Horizon | CRPS (LSTM) | CRPS (persistence) | 90% coverage | 90% interval width m |
|---|---|---|---|---|
| 1 day | 0.040 | 0.059 | 0.92 | 0.30 |
| 7 days | **0.138** | 0.260 | 0.89 | 0.78 |
| 30 days | **0.282** | 0.590 | 0.85 | 1.47 |

<sub>CRPS (lower better) beats the persistence-Gaussian baseline at every horizon — nearly half at 30 days. Empirical coverage (PICP) sits near the nominal 0.90, so the intervals are well calibrated out of the box, and they stay sharp. Reproduce: `python -m hydrophysics.forecast_eval --device cuda --probabilistic`.</sub>

## Status

- **Foundation (done, tested, runs anywhere):** dataset loader, KGE/NSE/RMSE metrics with explicit simulation-vs-forecast modes, gray-box + climatology + last-value baselines, reproducible benchmark, synthetic sample for CI.
- **Models (GPU):** a working `GlobalGRU` reference model, and the `PhysicsUDE` physics-informed operator — hypernetwork + stable semi-implicit ODE rollout + physics-residual loss, trained end-to-end on CUDA and benchmarked above. Active build: closing the gap to the gray-box (per-well level anchoring, leave-one-well-out generalization) and the PhysicsNeMo port.
- **Forecasting (GPU):** `GlobalForecastLSTM`, a global attribute-aware multi-horizon forecaster with data assimilation, scored against persistence/climatology by `hydrophysics.forecast_eval`. Beats persistence at 7- and 30-day horizons, with a probabilistic (Gaussian) mode giving calibrated prediction intervals and CRPS/coverage scoring (see Forecast mode above).

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
    --baseline /path/to/gw_fit_results.csv --out results/ude --epochs 1500
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the GPU / CUDA / PhysicsNeMo install, the model contract, and the evaluation rules that keep the benchmark honest.

## Author

Abdoul Rachid Ouedraogo, Ph.D. — hydrogeology x AI. Also: [AquaScope](https://github.com/Rekin226/aquascope).

## License

MIT — see [LICENSE](LICENSE).
