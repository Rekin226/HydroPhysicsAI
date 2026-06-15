# Model Card — HydroPhysicsAI

Physics-informed neural operators for daily groundwater levels on the Zhuoshui alluvial
fan (Taiwan, 61 wells, 2012–2022). This card covers the three trainable models in the
repo and how they are evaluated. Numbers are measured by the harness in this repo; see
the [README](README.md) for the full tables.

## Models

| Model | What it is | Mode |
|---|---|---|
| **PhysicsUDE** | Universal Differential Equation: a gray-box mass-balance ODE skeleton (recession + rainfall + upstream coupling + seasonal) whose per-well parameters are produced by a shared attribute → parameter **hypernetwork**, trained end-to-end through a stable semi-implicit rollout with a physics-residual loss. | Simulation |
| **PhysicsNeMoUDE** | The same operator with its hypernetwork as a native `physicsnemo.Module` (NVIDIA PhysicsNeMo): metadata, AMP capability flags, `.mdlus` checkpointing. Bit-identical results to PhysicsUDE. | Simulation |
| **GlobalForecastLSTM** | One global, attribute-aware seq2seq LSTM that forecasts the level *h* days ahead from observed levels up to the origin (data assimilation) + future forcing. Optional Gaussian head for probabilistic output. | Forecast |

## Intended use

- Research and benchmarking of physics-ML vs classical gray-box ODEs for groundwater.
- **Simulation**: free-running hindcast of levels from an initial condition + forcing,
  for scenario analysis where no live observations are available.
- **Forecast**: short-horizon (1–30 day) operational forecasting with calibrated
  uncertainty, for early-warning / water-management contexts.

### Out of scope

- Not a calibrated operational product for any specific well; not validated outside the
  Zhuoshui fan or outside 2012–2022.
- Does **not** yet generalize to wells unseen in training (see Limitations).
- The public demo runs on **synthetic** data; it illustrates behavior, not real skill.

## Training data

- **Real:** 61 monitoring wells, daily groundwater level (target), paired daily rainfall
  and upstream-well level (drivers), and static per-well attributes. Hourly levels are
  resampled to daily means (dt = 1 day) to match the gray-box ODE. The agency data is
  **not redistributed** (terms); the loader reads it from a configured path.
- **Synthetic sample:** a bundled generator (`hydrophysics.sample`) mirrors the signal
  structure for CI, the demo, and no-data users.
- **Split:** train ≤ 2018, validation from 2019 (out-of-sample). All hyperparameters were
  chosen on an **inner** split carved from pre-2019 data (inner-train < 2018, inner-val
  2018); the 2019+ benchmark was scored exactly once, so reported numbers are not tuned
  to the test set.

## Evaluation

Metrics: KGE, NSE, RMSE (per-well, reported as medians over 61 wells). Simulation and
forecast modes are scored **separately and are not comparable** (forecasting with
assimilation is an easier task).

| Task | Result | Reference |
|---|---|---|
| Simulation (free-running) | PhysicsUDE / PhysicsNeMoUDE **KGE 0.591**, NSE 0.49 | gray-box 0.736 · climatology 0.446 |
| Leave-one-well-out (static attrs) | KGE 0.236 | climatology 0.446 |
| Leave-one-well-out (+ observable signatures) | KGE 0.389 | climatology 0.446 |
| Leave-one-well-out (**+ self-consistency gate**) | **KGE 0.530** | climatology 0.446 |
| Forecast 7-day | LSTM **KGE 0.965** | persistence 0.946 |
| Forecast 30-day | LSTM **KGE 0.899** | persistence 0.703 |
| Forecast probabilistic | CRPS 0.143 / 0.286 (7/30 d), PICP ≈ 0.90 | persistence-Gaussian 0.260 / 0.590 |

GPU: on an RTX 4070 SUPER the forecaster trains 14× faster than CPU under bf16-AMP at
~half the memory (`python -m hydrophysics.bench`).

## Limitations

- **Trails the classical baseline in simulation:** 0.591 vs the per-well gray-box 0.736.
  The operator wins on 18/61 wells; a few hard wells dominate the gap.
- **Generalization to unseen wells — improved, with caveats.** Leave-one-well-out median
  KGE went 0.236 (geographic attrs) → 0.389 (observable history signatures) → **0.530**
  (adding a self-consistency gate: trust the operator only where it reproduces a well's
  own training history, else fall back to climatology). The gated hybrid beats climatology
  (0.446) and erases the catastrophic tail. **But** it is not a wholesale model of unseen
  wells: it equals climatology on the 34/61 wells it can't model, the operator alone still
  loses head-to-head to climatology on most wells, and KGE's unbounded tail means median
  is the trustworthy aggregate (mean is not). Gate threshold and feature set were chosen on
  the inner 2018 split; 2019+ was scored once.
- **Forecast skill depends on assimilation:** the LSTM uses observed levels up to the
  origin; it is not a free-running model and its scores must not be compared to
  simulation mode.
- **Synthetic demo:** the live Space does not reflect real-well accuracy.

## Reproduce

```bash
pip install -e ".[gpu,nemo,viz,dev]"
python -m hydrophysics.run_baselines                 # baselines (synthetic sample)
export HYDROMIND_GW_DATA=/path/to/data               # real data
python -m hydrophysics.train --model ude_nemo --epochs 1500 --out results/ude_nemo
python -m hydrophysics.forecast_eval --device cuda --horizons 1,7,30 --probabilistic
```

## License & citation

MIT (see [LICENSE](LICENSE)). Author: Abdoul Rachid Ouedraogo. Repository:
https://github.com/Rekin226/HydroPhysicsAI
