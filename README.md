# HydroPhysicsAI

**GPU physics-informed neural operators for groundwater — one model across many wells, benchmarked against per-well gray-box ODEs.**

[![CI](https://github.com/Rekin226/HydroPhysicsAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Rekin226/HydroPhysicsAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20PhysicsNeMo%20%7C%20CUDA-76b900.svg)](#nvidia-gpu-path)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-live-blue.svg)](https://huggingface.co/spaces/Rekin226/HydroPhysicsAI-demo)

**[▶ Live demo](https://huggingface.co/spaces/Rekin226/HydroPhysicsAI-demo)** · **[Model card](MODEL_CARD.md)** · **[Technical writeup](docs/TECHNICAL_WRITEUP.md)**

---

## Results at a glance

All on the real 61-well Zhuoshui data, out-of-sample validation from 2019. Median KGE (higher is better) unless noted.

| Task | This repo | Reference | Verdict |
|---|---|---|---|
| **Simulation** (free-running hindcast) | physics-UDE **0.591** | gray-box 0.736 · climatology 0.446 | beats climatology, trails the per-well gray-box |
| **Generalize to unseen wells** (leave-one-well-out) | **0.236** | climatology 0.446 | open problem — does not yet generalize |
| **Forecast, 7-day** (operational, assimilated) | LSTM **0.965** | persistence 0.946 | real skill over persistence |
| **Forecast, 30-day** | LSTM **0.899** | persistence 0.703 | nearly halves the error |
| **Forecast, probabilistic** | CRPS beats persistence, ~90% calibrated | — | sharp, well-calibrated intervals |
| **GPU training** (mixed precision) | **14×** over CPU, ~½ the memory | fp32 10.9× | real NVIDIA-stack speedup |
| **NVIDIA PhysicsNeMo port** | KGE **0.591** (`ude_nemo`) | PyTorch UDE 0.591 | runs on the framework, identical skill |

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

![Free-running simulation vs observed](results/figures/simulation_hydrographs.png)

*Free-running PhysicsUDE hindcast vs observed, validation period shaded. Generated from the bundled **synthetic sample** (`python -m hydrophysics.figures`) so the figure reproduces anywhere and ships no real agency series; the same command on real data redraws it for the 61 wells.*

## Forecast mode (operational, data-assimilated)

A **separate** track from the simulation benchmark above (do not compare the two — different task). Here a single global attribute-aware LSTM (`hydrophysics/models/forecast_lstm.py`) forecasts the level `h` days ahead using observed levels up to the forecast origin (assimilation) plus forcing, scored on 2019+ against the honest forecast-mode references at each horizon:

| Horizon | LSTM KGE | Persistence KGE | LSTM RMSE m | Persistence RMSE m |
|---|---|---|---|---|
| 1 day | 0.995 | 0.997 | 0.08 | 0.10 |
| 7 days | **0.965** | 0.946 | **0.32** | 0.44 |
| 30 days | **0.899** | 0.703 | **0.53** | 1.08 |

<sub>Median over 61 wells; climatology scores ~0.45 KGE at every horizon. The LSTM adds real skill over persistence at 7 and 30 days (at 30 days it nearly halves the error); the 1-day row is near-trivial for both and shown only for context. Hyperparameters (learning rate) were tuned on an inner pre-2019 split (inner-train <2018, inner-val 2018), then evaluated once on 2019+. Reproduce: `python -m hydrophysics.forecast_eval --device cuda --horizons 1,7,30`.</sub>

**Probabilistic forecasts.** With `--probabilistic` the LSTM emits a Gaussian per horizon (mean + variance, trained by Gaussian NLL), so each forecast is a calibrated distribution — read off any prediction interval or exceedance probability for early warning.

| Horizon | CRPS (LSTM) | CRPS (persistence) | 90% coverage | 90% interval width m |
|---|---|---|---|---|
| 1 day | 0.041 | 0.059 | 0.92 | 0.29 |
| 7 days | **0.143** | 0.260 | 0.89 | 0.78 |
| 30 days | **0.286** | 0.590 | 0.85 | 1.42 |

<sub>CRPS (lower better) beats the persistence-Gaussian baseline at every horizon — nearly half at 30 days. Empirical coverage (PICP) sits near the nominal 0.90, so the intervals are well calibrated out of the box, and they stay sharp. Reproduce: `python -m hydrophysics.forecast_eval --device cuda --probabilistic`.</sub>

![Probabilistic 30-day forecast with 90% interval](results/figures/forecast_fan.png)

*Probabilistic 30-day forecast from one origin (dashed line): the mean tracks the realized observations and the 90% interval widens with lead time. Bundled-sample figure (`python -m hydrophysics.figures`), reproducible with no real data.*

## Status

- **Foundation (done, tested in CI, runs anywhere):** dataset loader, KGE/NSE/RMSE metrics with explicit simulation-vs-forecast modes, gray-box + climatology + last-value baselines, reproducible benchmark, synthetic sample, GitHub Actions CI (ruff + pytest on Python 3.10–3.12).
- **Models (GPU):** a working `GlobalGRU` reference model, the `PhysicsUDE` physics-informed operator (hypernetwork + stable semi-implicit ODE rollout + physics-residual loss), and `PhysicsNeMoUDE` — the same operator **ported to NVIDIA PhysicsNeMo** with `.mdlus` checkpointing, reproducing the headline KGE 0.591 on CUDA. Active build: closing the gap to the gray-box and leave-one-well-out generalization.
- **Forecasting (GPU):** `GlobalForecastLSTM`, a global attribute-aware multi-horizon forecaster with data assimilation and **bf16 mixed-precision** training (14× over CPU; see GPU performance), scored against persistence/climatology by `hydrophysics.forecast_eval`. Beats persistence at 7- and 30-day horizons, with a probabilistic (Gaussian) mode giving calibrated prediction intervals and CRPS/coverage scoring (see Forecast mode above).
- **Tooling:** `hydrophysics.figures` (reproducible plots) and `hydrophysics.bench` (CPU vs CUDA vs CUDA+AMP throughput).

## Quickstart

```bash
pip install -e .                      # foundation only (numpy/pandas)
pip install -e ".[gpu]"               # + torch, torchdiffeq (on the CUDA machine)
pip install -e ".[nemo]"              # + NVIDIA PhysicsNeMo (for --model ude_nemo)
pip install -e ".[gpu,nemo,viz,dev]"  # everything

# Reproduce the baselines on the bundled synthetic sample (no real data, no GPU):
python -m hydrophysics.run_baselines

# Train + benchmark a model on the sample:
python -m hydrophysics.train --model gru --epochs 30

# Regenerate the figures + the GPU throughput benchmark (reproducible, no real data):
python -m hydrophysics.figures
python -m hydrophysics.bench

# On real data + GPU (PhysicsNeMo port of the operator):
export HYDROMIND_GW_DATA=/path/to/data
python -m hydrophysics.train --model ude_nemo --out results/ude_nemo --epochs 1500
```

The real Zhuoshui groundwater data is **not** redistributed here (agency-data terms). The repo ships a synthetic sample in `hydrophysics/sample_data/` and reads real data via the `HYDROMIND_GW_DATA` path.

## Live demo

An interactive [Gradio demo](https://huggingface.co/spaces/Rekin226/HydroPhysicsAI-demo) lets you pick a well and see the free-running PhysicsUDE simulation and a calibrated probabilistic 30-day forecast. It runs on **synthetic** data (the real agency series can't be redistributed and the Space is public) but exercises the same models and code paths.

```bash
pip install -e ".[gpu,viz]" gradio   # demo deps
python app/app.py                    # run locally at http://localhost:7860

# deploy your own Space (after `hf auth login` with a write token):
python app/deploy_space.py
```

## NVIDIA GPU path

Not aspirational — the flagship runs on the NVIDIA stack today:

1. **PhysicsNeMo port (done).** `PhysicsNeMoUDE` (`--model ude_nemo`) is the operator with its hypernetwork as a native `physicsnemo.Module`. It carries PhysicsNeMo `ModelMetaData` capability flags (AMP / auto-grad), serializes to a single portable `.mdlus` checkpoint (`save_checkpoint` / `load_checkpoint`, architecture + weights together), and reproduces the simulation headline **exactly** — median KGE 0.591 on the real 61-well data, bit-identical to the pure-PyTorch UDE. Install with `pip install -e ".[nemo]"`.
2. **Mixed precision (done).** The forecaster trains under bf16 autocast (`amp=True`). On an RTX 4070 SUPER it is **14× faster than CPU and uses ~half the GPU memory** of fp32 (see GPU performance below). Reproduce: `python -m hydrophysics.bench`.
3. **CUDA training (done).** `train.py` auto-selects `cuda > mps > cpu`; all models are standard PyTorch and train on any CUDA GPU as-is.
4. **Next on the NVIDIA path.** Constant-memory adjoint rollout via `torchdiffeq.odeint_adjoint`, PhysicsNeMo multi-GPU / distributed training, and the leave-one-well-out operator-generalization study.

### GPU performance — forecaster training throughput

| Backend | Throughput | Speedup vs CPU | Peak GPU memory |
|---|---|---|---|
| CPU | 18.5k windows/s | 1.0× | — |
| CUDA fp32 | 201k windows/s | 10.9× | 2682 MB |
| **CUDA bf16-AMP** | **260k windows/s** | **14.1×** | **1362 MB** |

<sub>RTX 4070 SUPER, 40 wells × 6 years, 8 epochs. Mixed precision is both faster *and* roughly halves memory. Reproduce: `python -m hydrophysics.bench`.</sub>

![Forecaster training throughput by backend](results/bench/gpu_benchmark.png)

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
  forecast_eval.py horizon-wise forecast scoring (point + probabilistic)
  figures.py       generate the README figures (hydrographs, fan, bars)
  bench.py         CPU vs CUDA vs CUDA+AMP throughput benchmark
  viz.py           plotting helpers (lazy matplotlib)
  models/
    base.py        GroundwaterModel interface (fit + simulate)
    gru.py         GlobalGRU reference model (working, GPU-ready)
    ude.py         PhysicsUDE physics-informed operator (the new method)
    ude_physicsnemo.py  PhysicsNeMoUDE: the UDE on NVIDIA PhysicsNeMo
    forecast_lstm.py    GlobalForecastLSTM (assimilated, probabilistic, AMP)
results/phase0/    frozen baselines + spatial map
results/figures/   reproducible figures (from the synthetic sample)
results/bench/     GPU throughput benchmark
tests/             foundation + model-smoke + forecast tests (run in CI)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the GPU / CUDA / PhysicsNeMo install, the model contract, and the evaluation rules that keep the benchmark honest.

## Author

Abdoul Rachid Ouedraogo, Ph.D. — hydrogeology x AI. Also: [AquaScope](https://github.com/Rekin226/aquascope).

## License

MIT — see [LICENSE](LICENSE).
