# One operator for many wells: a physics-informed approach to groundwater prediction

*A technical walkthrough of HydroPhysicsAI — what it does, how it's built, what works,
and what doesn't.*

## The problem

Groundwater management on an alluvial fan means tracking dozens of wells, each with its
own hydrogeology. The classical approach calibrates **one ODE per well**: a small
mass-balance "tank" model (recession + rainfall recharge + upstream coupling + a seasonal
term) fit independently to each station. On the Zhuoshui fan that's 61 separate parameter
sets, each blind to the others, and re-calibration is needed for every new well.

Two questions motivated this project:

1. Can a **single** model, trained across all wells at once and conditioned on each well's
   static attributes, match those 61 hand-calibrated ODEs — and amortize to new wells?
2. Separately, how well can we **forecast** levels operationally, with calibrated
   uncertainty, when recent observations are available?

## The method: a Universal Differential Equation

The flagship is not a black box. It keeps the gray-box mass-balance ODE as a hard
inductive bias and learns only the parameters:

```
dh/dt = -a·(h − z) + b·rain + k·(upstream − h) + c + d_sin·sin(doy) + d_cos·cos(doy)
```

A shared **hypernetwork** maps each well's static attributes → its parameter vector
`(a, z, b, c, k, d_sin, d_cos)`. So one network conditions on attributes and, in
principle, can predict a well it never calibrated on. Three implementation choices matter:

- **Stable integration.** Collecting the terms linear in `h` gives `dh/dt = −g·h + s_t`
  with decay `g = a + k ≥ 0`. We integrate semi-implicitly (backward-Euler on the decay,
  explicit on the forcing): `h_{t+1} = (h_t + s_t)/(1 + g)`. This is *unconditionally
  stable*, so a long free-running hindcast can't blow up even while the hypernetwork is
  still learning — the naive explicit Euler loop it replaces could diverge to NaN.
- **Level anchoring.** The reference level `z` is predicted as a residual around each
  well's training-mean level, so the network doesn't have to regress absolute levels from
  attributes (which wrecks the bias on hard wells).
- **Physics-residual loss.** Beyond fitting observations, we add a teacher-forced
  collocation term that enforces the ODE on consecutive observed days. This constrains the
  *dynamics*, not just the fit — the "informed" half of physics-informed.

## Evaluation discipline (the part that's easy to get wrong)

Groundwater is highly autocorrelated, so a 1-step-ahead "forecast" of tomorrow ≈ today
scores deceptively well. We therefore separate two modes and **never** compare across them:

- **Simulation** — free-running from an initial condition + forcing, never seeing observed
  levels. Honest references: per-well gray-box ODE, seasonal climatology, last-value.
- **Forecast** — predict day *t* using observations up to the origin (assimilation).
  Reference: persistence.

All hyperparameters were tuned on an **inner** split (inner-train < 2018, inner-val 2018)
carved from the pre-2019 training data; the 2019+ benchmark was scored exactly once. No
number in this repo is tuned against the test set.

## Results

**Simulation.** The operator clears the seasonal climatology comfortably (KGE 0.591 vs
0.446; NSE 0.49 vs −0.20) and beats the per-well gray-box on 18 of 61 wells — but trails
it overall (0.591 vs 0.736). A few hard wells dominate the gap.

**Forecasting.** A single global attribute-aware LSTM adds real skill over persistence at
7- and 30-day horizons (30-day KGE 0.899 vs 0.703 — it nearly halves the error). A
Gaussian head gives calibrated intervals out of the box: empirical 90% coverage sits at
~0.90 and CRPS beats a persistence-Gaussian baseline at every horizon.

**The hard part: generalizing to unseen wells.** The operator-learning headline —
predict a *held-out* well that was never calibrated (leave-one-well-out) — started at KGE
0.236, below climatology. The raw station data carries no hydrogeology (just
coordinates), so geographic attributes can't place an unseen well in parameter space.

But a held-out well still has a monitoring *history*. Summarizing that history into six
physically-meaningful signatures — lag-1 autocorrelation (→ recession rate), rainfall
sensitivity (→ recharge gain), upstream coupling (→ `k_link`), seasonal amplitude, level
spread, trend — and conditioning the shared network on those instead of geography lifts
leave-one-well-out from **0.236 to 0.389 median KGE (+65%)**, better on every metric,
nearly closing the gap to climatology (0.446). The feature set was selected on an inner
2018 split and the 2019+ number scored exactly once, so it is leakage-free. It is real
progress, but **not a solved problem**: the median still trails climatology and a tail of
wells remains badly mismodeled. The lesson is that *observable behavior generalizes where
geography does not* — which points squarely at the next step.

## Running on the NVIDIA stack

- **PhysicsNeMo port.** `PhysicsNeMoUDE` makes the hypernetwork a native
  `physicsnemo.Module` — capability metadata + single-file `.mdlus` checkpointing — and
  reproduces the simulation headline exactly on CUDA.
- **Mixed precision.** The forecaster trains under bf16 autocast: 14× faster than CPU and
  ~half the GPU memory of fp32 on an RTX 4070 SUPER.
- **Reproducible everything.** CI (ruff + pytest) on every push; figures and the GPU
  benchmark regenerate from the bundled synthetic sample with no real data or GPU.

## What's next

**Push leave-one-well-out past climatology.** Observable signatures got it from 0.236 to
0.389; the remaining gap is a tail of wells whose free-running simulation diverges. Likely
levers: richer physical attributes if any can be sourced (aquifer transmissivity,
lithology, screen depth), a DeepONet/FNO operator head, regularizing the predicted
parameters toward stable regimes, or robust per-well uncertainty so the bad wells are
flagged rather than trusted. A constant-memory adjoint rollout
(`torchdiffeq.odeint_adjoint`) and PhysicsNeMo multi-GPU training are the natural systems
follow-ups.

---

*See the [README](../README.md) for the full benchmark tables and the
[model card](../MODEL_CARD.md) for intended use and limitations.*
