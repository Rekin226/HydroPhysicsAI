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
leave-one-well-out from 0.236 to **0.389 median KGE**.

Then a methodological check worth dwelling on: **is median KGE even the right metric?**
It is flattering. Per well, the operator still *loses head-to-head to each well's own
climatology on 54% of wells*, and KGE's unbounded negative tail makes the *mean* useless
(a few wells free-run to large negative KGE). But that same per-well view exposes the real
structure — **the operator and climatology are complementary**: the operator adds genuine
skill on ~46% of wells and is untrustworthy on the rest. An oracle that picked the better
of the two per well would score 0.54.

Two more steps followed, each diagnosed rather than guessed. **First, pin the
equilibrium.** Inspecting the wells that still free-ran badly showed a clear signature:
their predicted steady state `h* = mean(source)/g` sat 5–15 m off the well's actual mean
level (bad wells' median offset 5.75 m vs 1.67 m for good wells). The operator had free
degrees of freedom in the absolute level (`z`, `c`) that it couldn't pin down for an
unseen well. The fix is physical: pin each well's free-run equilibrium to its observed
mean (the anchor — available for a held-out well) and let the ODE drive only *deviations*
around the training-period mean forcing. That lifts the operator *alone* to 0.491 median —
**beating climatology head-to-head on 62% of wells**.

**Second, the self-consistency gate**: trust the operator only on wells where it can
free-run-reproduce *their own training history* (training-period KGE ≥ τ, τ=0.3 selected
on the inner split, leakage-free), else fall back to climatology. The final hybrid scores
**0.565 median KGE** with clipped-mean 0.515 (vs climatology's 0.332) — and the held-out
median now nearly matches the in-sample operator (0.591), i.e. predicting a *never-seen*
well is almost as good as having trained on it.

This is honest progress, not a conquest. The full arc is 0.236 → 0.389 → 0.491 → 0.565.
But the gated hybrid is still *worse* than climatology on 8 wells (the gate misplaced
trust, by up to ~1.9 KGE) and merely equals it on the 18 it falls back on.

**Chasing the last false positives — and finding the wall.** Averaging an ensemble of
K=3 operators lifts the held-out median to ≈0.59, *matching the in-sample operator
(0.591)* — a never-calibrated well predicted as well as a trained one. To kill the
worse-than-climatology wells we added an epistemic-uncertainty gate: distrust wells where
the independently-seeded ensemble members *disagree about the 2019+ trajectory*
(leakage-free — only forcing and model outputs). It caught just 2 of the 8. Diagnosing the
rest showed why: they are **non-stationary** — the well's 2019+ genuinely departs from its
training history (climatology fails them too; one sits at −0.8). The ensemble members all
fit the past, so they *agree* on the future *and are all wrong*. Disagreement — or any
training-time signal — cannot flag a regime change it has never seen. That is a property
of the data, not a tuning failure, and it is the honest ceiling of this approach.

The lessons that carry forward — *observable behavior generalizes where geography does
not*, *respect the physics (pin what you can observe)*, *a model should know when not to
be trusted* — and the limit: *self-knowledge can't catch a world that changed.*

## Running on the NVIDIA stack

- **PhysicsNeMo port.** `PhysicsNeMoUDE` makes the hypernetwork a native
  `physicsnemo.Module` — capability metadata + single-file `.mdlus` checkpointing — and
  reproduces the simulation headline exactly on CUDA.
- **Mixed precision.** The forecaster trains under bf16 autocast: 14× faster than CPU and
  ~half the GPU memory of fp32 on an RTX 4070 SUPER.
- **Reproducible everything.** CI (ruff + pytest) on every push; figures and the GPU
  benchmark regenerate from the bundled synthetic sample with no real data or GPU.

## What's next

**Close the last LOWO gap.** Leave-one-well-out now beats climatology (0.565 vs 0.446) and
nearly matches the in-sample operator (0.591), but 8 wells still beat the gate and end up
worse than climatology. Tighter gating (a learned trust score or per-well predictive
uncertainty rather than a single τ), richer physical attributes if any can be sourced
(aquifer transmissivity, lithology, screen depth), or a DeepONet/FNO operator head are the
levers. A constant-memory adjoint rollout (`torchdiffeq.odeint_adjoint`) and PhysicsNeMo
multi-GPU training are the natural systems follow-ups.

---

*See the [README](../README.md) for the full benchmark tables and the
[model card](../MODEL_CARD.md) for intended use and limitations.*
