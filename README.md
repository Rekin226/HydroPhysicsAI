# HydroPhysicsAI

**A differentiable, gated digital twin of the Choushui alluvial fan (Taiwan): pumping policy → groundwater heads → land subsidence, on the NVIDIA GPU stack.**

[![CI](https://github.com/Rekin226/HydroPhysicsAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Rekin226/HydroPhysicsAI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stack](https://img.shields.io/badge/stack-PyTorch%20%7C%20PhysicsNeMo%20%7C%20CUDA-76b900.svg)](#nvidia-gpu-path)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20demo-live-blue.svg)](https://huggingface.co/spaces/Rekin226/HydroPhysicsAI-demo)

**[▶ Live demo](https://huggingface.co/spaces/Rekin226/HydroPhysicsAI-demo)** · **[Model card](MODEL_CARD.md)** · **[Technical writeup](docs/TECHNICAL_WRITEUP.md)** · **[Project state](docs/superpowers/STATE.md)**

---

## What this is

The Choushui fan is Taiwan's largest groundwater basin and its worst subsidence problem:
unmetered irrigation pumping from a stack of four aquifers, and a ground surface that sinks
centimetres a year over the high-speed-rail corridor. This repository is a **digital twin**
of that system: a four-layer differentiable groundwater solver on the fan grid, driven by
the registered-pump electricity census and rain-minus-ET recharge, coupled to a
visco-elasto-plastic compaction column, calibrated by gradient descent through the solver's
own adjoint, and **gated at every stage against held-out data**. It takes a pumping policy
("cut irrigation 30 % from 2026", "retire aquaculture"), runs the aquifer forward, and
animates where and how fast the ground sinks — with a parameter ensemble around it.

```
pumping policy ──▶ flow solver ──▶ layer heads ──▶ VEP column ──▶ subsidence ──▶ 3D viewer
   (census)        4 layers,        nudged to      per zone,        798 leveling     policy
                   open basin       observations   leveling-fit     benchmarks       dropdown
```

## The data

Not a toy set. Everything below is real, on the 2,144 km² fan, 2012–2022, monthly:

| source | size | role |
|---|---|---|
| Water Resources Agency monitoring wells (via the WiseEnvr API) | **174 wells** passing QC, 158 on the 1 km grid, layer-resolved across 4 aquifers, hourly → monthly | heads: calibration target and assimilation |
| Taiwan Power Company pump census + monthly electricity | **116,769 registered pumps**, 26 M meter-months; 2.1 TWh after de-duplication and capacity screening | abstraction forcing (energy → volume through the pump's own hydraulics) |
| Rain gauges + Open-Meteo/ERA5 ET₀ | 26 gauges, daily | recharge = rain − ET₀ |
| Leveling benchmarks | **798 sites**, 7,539 site-surveys | independent subsidence validation, and (since 2026-09-14) the compaction column's calibration target |
| Multi-layer compaction wells (MLCW) | 14 magnetic-ring sites | compaction rings, held out as an independent check |
| Fan polygon, zones, coast and apex geometry | 2,148 active cells at 1 km | grid, boundaries |

The cache is rebuilt from the API with `hydrophysics.twin.fetch_amp` (credentials from
the environment only); layout in `docs/DATA_FORMAT.md`. Only a synthetic sample ships.

## Results at a glance

Every number is out of sample unless marked; verdicts are recorded in
`docs/superpowers/STATE.md` and the design specs, including the failures.

| stage | what is gated | result |
|---|---|---|
| **Stage 3 — flow model** | 5-fold, site-grouped, k-fold R² of held-out wells must beat inverse-distance interpolation | **PASS** (2026-09-14): **+0.757 vs IDW +0.702**, in-sample +0.906. Two earlier configurations failed (−0.236, −0.080); the fixes were an open coastal/apex boundary and a cleaned census |
| **Stage 4 — compaction column on the flow model's heads** | site-grouped 5-fold over 798 leveling benchmarks | **+0.546** out of fold (per-zone column), bias +0.1 cm; +0.371 on the 14 rings it never saw |
| **Hindcast subsidence, full chain** | 798 leveling sites, 18-member ensemble, sequential assimilation | **R² +0.526**, bias 0.0 cm, RMSE 6.6 cm |
| **Policy projection 2023–2032** | fan-mean subsidence, ± over parameter and initial-condition members | baseline 10.0 ± 1.3 cm; irrigation −30 % 9.9 ± 1.3; aquaculture retired 9.5 ± 1.1 |
| **Re-run cost** | 18 members × 3 policies × 252 months | 609 s on one Turing GPU; the FNO surrogate is 13–34× faster again (0.03–0.04 m RMSE vs the solver over 24 months) |
| **Uncertainty** | Laplace posterior around the calibrated vector + fold ensemble + perturbed initial fields | 32 parameters, 10 at a bound and held; posterior sd 0.5–0.7 (mid zone) to 2 (deep distal, coast) |

**The caveat that governs the policy numbers**, stated here because it decides what the
table means: the gated flow model still keeps its pump conversion at its bounds (efficiency
at the floor, extra head at the ceiling) and generalises with ~2 % of the published
abstraction. Holding the conversion at physical values fits nearly as well in sample
(+0.877) but *fails* the gate (+0.626). So heads and subsidence are validated; the
model's *sensitivity* to pumping is not yet. Irrigation return flow is the leading
candidate (in-sample +0.884 at a physical conversion) and its gate is queued. See the
state doc for the ledger.

## Digital twin of the Choushui fan (pumping → heads → subsidence)

The twin lives under `hydrophysics/twin/`: a differentiable four-layer groundwater solver on the fan grid,
driven by the registered-pump electricity census and rain-minus-ET recharge, coupled to a
visco-elasto-plastic compaction column, with a 3D viewer that takes a **pumping policy**
and animates the ground sinking under it, past the end of the data.

```
pumping policy ──▶ flow solver ──▶ layer heads ──▶ VEP column ──▶ subsidence ──▶ 3D viewer
                   (Stage 3)       nudged to obs    (Stage 2 PASS)                 (Stage 5)
                                   at the origin
```

Every stage is gated against held-out data and the verdicts are recorded, not assumed
(`docs/superpowers/STATE.md` is the current ledger):

| stage | what | gate | status |
|---|---|---|---|
| 2 | VEP compaction column, one shared parameter set | leave-one-site-out vs a single-`Sk` baseline | **PASS** (+0.465 vs +0.031) |
| 3 | flow solver, zonal parameterisation, open basin, clean census | clamp release; k-fold R² must beat IDW | **PASS** (2026-09-14): 5-fold +0.757 vs IDW +0.702; pump conversion still at its bounds, so policy sensitivities are under test |
| 4 | flow ↔ compaction coupling | column refit on the flow model's heads, site-grouped 5-fold on 798 leveling benchmarks | **PASS**: per-zone column +0.546 out of fold (Stage-2 column under the same driver +0.299) |
| 5 | 3D scenario twin | — | forward mode ships on the gated parameters (`results/twin/explorer3d_forward.html`) |

Two defects found on 2026-09-11 explain the Stage-3 failures to date and are fixed as
defaults: the solver was a **closed basin** (no coastal outlet, no apex inflow; now
general-head boundaries with learnable conductance, `twin/boundaries.py`), and the pump
census **over-counted electricity 4.6×** (shared meters attached to every pump on them,
plus meters drawing many times their rated motor capacity; now `pumping.clean_census`).

```bash
export HYDROMIND_GW_DATA=$PWD/chou-shui-data/data
# calibrate + gate (GPU, ~1 day at 5 folds); writes results/twin_runs/<stamp>/stage3_*.{csv,json}
python -m hydrophysics.twin.calibrate_flow --param-mode zonal --epochs 500 --n-folds 5 \
    --device cuda --compile-matvec --log-every 25
# run a policy forward: ensemble over the in-sample and fold parameter sets
python -m hydrophysics.twin.forward --theta results/twin_runs/<run>/stage3_theta.json \
    --theta results/twin_runs/<run>/stage3_fold_thetas.json --horizon 120 \
    --scenario "cut30:irrigation=0.7@2026-01" --out results/twin_forward/cut30
# animate it
python -m hydrophysics.twin.explorer3d --forward-npz results/twin_forward/cut30.npz \
    --stride 3 --out results/twin/explorer3d_forward.html
# optional: an FNO surrogate (PhysicsNeMo) of the calibrated solver for sweeps and ensembles
python -m hydrophysics.twin.surrogate build --theta ... --out results/surrogate/data.npz
```

A forward run prints the Stage-3 verdict it inherits and the hindcast skill against
leveling on every invocation; the viewer carries the verdict in its title. The twin never
decides for itself whether its pumping → head map is trustworthy.

## Earlier phase (2026-06): one operator across 61 wells

Before the twin, this repository benchmarked **one physics-informed neural operator
across the 61 curated wells of the original data delivery** against per-well gray-box
ODEs, in free-running simulation and in assimilated forecasting. That work stands as
recorded below, but it is no longer the project's headline: the twin above runs on a
dataset three times as dense in wells, layer-resolved, with the pumping forcing and the
subsidence network the operator benchmark never had, and its gates are physical (held-out
wells, held-out leveling sites) rather than a comparison with a baseline that was scored
elsewhere. The gray-box numbers here are the original project's own evaluation and were
never re-scored on this harness; the comparison is kept for the record, not as a claim.

<details>
<summary>The 61-well operator benchmark, forecaster, PINN field and explorer (click to expand)</summary>

### The idea

Classical hydrology calibrates **one ODE per well**: 33-61 separate parameter fits, each blind to the others. HydroPhysicsAI trains **a single physics-informed neural operator across all wells at once**, conditioned on each well's static attributes, on the NVIDIA GPU stack (PyTorch / PhysicsNeMo / CUDA). It is scored in true **simulation mode** (free-running hindcast from an initial condition + forcing, never seeing observed levels) against the per-well gray-box ODE baseline.

> One GPU-trained physics-ML operator, across all 61 wells, aiming to match or beat 61 hand-calibrated ODEs — and run in milliseconds, generalize to wells it never saw, and carry calibrated uncertainty.

Test bed: 61 groundwater monitoring wells on the Zhuoshui alluvial fan, Taiwan, 2012-2022, validated out-of-sample from 2019.

### Why physics-informed, not a black box

The model keeps the gray-box mass-balance ODE (recession + rainfall + upstream coupling + seasonal terms) as its inductive bias, and learns a neural **hypernetwork** that maps well attributes to the ODE parameters. So it stays interpretable (read off `a`, `b`, `k_link` per well), extrapolates better than a pure sequence model, and amortizes: one network conditions on attributes, so it can predict a well it was never calibrated on. That leave-one-well-out generalization is the operator-learning headline.

### Benchmark (simulation mode, validation period)

The bar to beat, computed by the harness in this repo:

| Model | KGE (median) | NSE (median) | RMSE m (median) | Wells |
|---|---|---|---|---|
| **Gray-box ODE** (per-well calibrated, baseline) | **0.736** | — | 0.83 | 61 |
| Climatology (per-well seasonal mean) | 0.446 | -0.20 | 1.63 | 61 |
| Last-value (constant) | undefined* | -0.57 | 1.83 | 61 |
| Physics-informed neural operator (baseline) | 0.591 | 0.49 | 1.07 | 61 |
| **+ rainfall recharge-memory** (this project) | **0.704** | **0.57** | **0.95** | 61 |
| **+ evapotranspiration driver** (this project) | **0.754** | **0.60** | **0.81** | 61 |

<sub>*A constant prediction has zero variance, so KGE is undefined; skill is read from NSE/RMSE. Forecast-mode persistence scores KGE 0.997 but is deliberately excluded: 1-step-ahead prediction of slow-moving groundwater is trivial and not comparable to a free-running simulation. The harness keeps simulation and forecast modes separate so the comparison stays honest.</sub>

The operator is one network across all 61 wells (attribute-conditioned hypernetwork to ODE parameters, level anchored to each well's training mean, per-well-weighted data + physics-residual loss, 1500 epochs on CUDA). The original instantaneous `b·rain` recharge term underfit the response — aquifers integrate rainfall over weeks to months — so we replaced it with **multi-timescale recharge memory**: the hypernetwork predicts one gain per causal memory state `api_k[t] = decay_k·api_k[t-1] + rain[t]` (decays 0.99/0.95/0.85 ≈ 100/20/6-day timescales). That single change lifts the operator from **0.591 to 0.704 median KGE** (NSE 0.49→0.57, RMSE 1.07→0.95, better on every metric), so **one shared, amortized operator now nearly matches the 61 hand-calibrated gray-box ODEs (0.704 vs 0.736)** — the gap closed from 0.145 to 0.032. The memory timescales were selected on an inner pre-2019 split (inner-train <2018, inner-val 2018) — reproduce that selection with `python results/ude/inner_select.py --device cuda`; the 2019+ benchmark was scored exactly once, and the baseline reproduces at 0.591. Reproduce: `python -m hydrophysics.train --model ude --rain-memory 0.99,0.95,0.85 --epochs 1500`. (The ingredient is borrowed from the per-well [decomposition](#per-well-signal-decomposition--reaching-the-gray-box-accuracy-class) below, now folded into the shared operator.)

### Adding the missing driver: evapotranspiration

Recharge is rainfall *minus* evapotranspiration, but the model only saw rainfall — so it could not explain ET-driven decline. We add the other half: **net recharge = rain − ET₀**, where daily FAO-56 ET₀ comes free (no API key) from Open-Meteo/ERA5 at each well, fetched via the [AquaScope](https://github.com/Rekin226/aquascope) toolkit. That lifts the operator from **0.704 to 0.754 median KGE** (NSE 0.57→0.60, RMSE 0.95→0.81, beats climatology on 58/61 wells) — now **at or above the gray-box's reported 0.736**. Strikingly, the out-of-sample gain (+0.050) is *larger* than the inner-split gain (+0.016) for a physical reason: the 2019+ window contains Taiwan's record **2020–2021 drought**, where decline is ET-driven and a rain-only model is blind — so the ET driver helps most exactly when prediction is hardest. The subtraction coefficient was selected on the inner 2018 split (`python results/ude/inner_select.py --device cuda --et`); 2019+ scored once; the control reproduces at 0.704. Reproduce: `python -m hydrophysics.train --model ude --rain-memory 0.99,0.95,0.85 --et --epochs 1500` (uses the committed ERA5 ET₀ cache; no fetch needed).

![Per-well gray-box KGE across the Zhuoshui fan](results/phase0/spatial_kge_graybox.png)

*Per-well validation KGE (gray-box baseline) over the real study area — the Zhuoshui alluvial-fan outline, rivers, and coastline (from the project shapefiles, TWD97). Each dot is a station at its true coordinates, colored by validation KGE; most wells are well-modeled (green/yellow), the dark wells are the hard cases a single attribute-conditioned operator should rescue. Reproduce: `python -m hydrophysics.maps --kge`.*

![Free-running simulation vs observed](results/figures/simulation_hydrographs.png)

*Free-running PhysicsUDE hindcast vs observed, validation period shaded. Generated from the bundled **synthetic sample** (`python -m hydrophysics.figures`) so the figure reproduces anywhere and ships no real agency series; the same command on real data redraws it for the 61 wells.*

### Per-well signal decomposition — reaching the gray-box accuracy class

A different angle on the simulation task: instead of predicting the raw level, **decompose each well's signal and model the components** — `level(t) = harmonic seasonal + (damped) trend + forcing-driven anomaly + anchor`, where rainfall enters through multi-timescale exponential recharge-memory filters and upstream as a lagged term (ridge-fit on the training residual). Every component is fit on training days only and projected from known forcing, so it is scored in the same leakage-free simulation mode as the gray-box.

| Model (2019+, simulation mode) | median KGE |
|---|---|
| climatology | 0.446 |
| physics-UDE operator (baseline) | 0.591 |
| **decomposition** (per-well) | **0.75** (no-trend variant 0.78) |
| gray-box ODE (reported) | 0.736 |
| physics-UDE operator + recharge-memory | 0.704 |

Its winning ingredient — multi-timescale rainfall recharge-memory — is exactly what we then folded into the shared operator above (lifting it 0.591 → 0.704). So the per-well experiment both reaches gray-box accuracy *and* points the way to improving the headline operator.

This lands in the **same accuracy class as the gray-box** and is well above our UDE and climatology on the identical harness. It is *not* a head-to-head win over the gray-box — that 0.736 is from the original project's own evaluation and is not re-scored here — and it is a **per-well** model (the gray-box's class), not the shared operator, so it answers "how accurately can we simulate?", not "does one operator generalize to unseen wells?". Verified leakage-free: corrupting every validation value leaves predictions unchanged (`tests/test_decomp_smoke.py`). Reproduce: `python results/decomp/final_benchmark.py`.

### Generalizing to unseen wells

The operator-learning headline: train on N−1 wells, then predict a **held-out** well that was never calibrated (k-fold, every well held out once, scored on 2019+). The gray-box can't do this at all — it fits one parameter set per well and has nothing to say about a new one.

Three steps took the held-out-well median KGE from 0.236 to **0.565** — past climatology (0.446) and within reach of the in-sample operator (0.591). Each step was selected on an inner 2018 split; the 2019+ numbers were scored exactly once.

**Step 1 — condition on observed behavior, not geography** (0.236 → 0.389). The raw station data carries no hydrogeology (just coordinates), so geographic attributes alone are weak. But a held-out well still has a monitoring *history*, so we summarize it into six physically-meaningful signatures (lag-1 autocorrelation → recession rate, rainfall sensitivity → recharge gain, upstream coupling → `k_link`, seasonal amplitude, level spread, trend), all from training days only.

**Step 2 — pin the equilibrium** (0.389 → 0.491). Diagnosing the wells that still failed showed they free-run to the *wrong level*: their predicted equilibrium sat 5–15 m off the well's actual mean, because the operator had free degrees of freedom in the absolute level. The fix is physical — pin each well's free-run equilibrium to its observed mean (the anchor, available for held-out wells) and let the ODE model only *deviations* driven by rainfall/upstream anomalies and season. The operator alone now **beats climatology** (0.491 vs 0.446, head-to-head on 62% of wells).

**Step 3 — gate on self-consistency** (0.491 → 0.565). Trust the operator only on wells where it can reproduce *their own training history* (free-run training-period KGE ≥ τ, τ=0.3 selected on the inner split, leakage-free), else fall back to climatology.

| LOWO method | median KGE | clipped-mean | beats climatology per-well | wells KGE<0 |
|---|---|---|---|---|
| climatology (reference) | 0.446 | 0.332 | — | 5 |
| static attributes only | 0.236 | — | 22 / 61 | 23 |
| + observable signatures | 0.389 | 0.207 | 28 / 61 | 18 |
| + equilibrium anchoring (operator) | 0.491 | 0.335 | 38 / 61 | 12 |
| **+ self-consistency gate (hybrid)** | **0.565** | **0.515** | 35 better · 18 tie · 8 worse | **3** |

The final hybrid **beats per-well climatology (0.565 vs 0.446 median)** and is far better on the bounded clipped-mean (0.515 vs 0.332). Honestly: it is *not* a clean sweep — it is worse than climatology on 8 wells (mean −0.41, worst −1.9) where the gate trusted the operator but shouldn't have, and it equals climatology on the 18 wells it falls back on. But on balance it is a clear, leakage-free improvement on a strong baseline, and the operator *alone* now generalizes well enough to beat climatology. Reproduce: `python -m hydrophysics.lowo --device cuda --features observable --anchor-equilibrium --gate 0.3`.

**Going further, and an honest wall.** Averaging an ensemble of K=3 operators (`--ensemble 3`) lifts the held-out median to **≈0.59 — matching the in-sample operator (0.591)**: predicting a never-calibrated well as well as a trained one. But it does *not* remove the worse-than-climatology wells. We chased them with an *uncertainty* gate (distrust wells where independently-seeded operators disagree about the future) — it caught only 2 of 8. The rest are **non-stationary**: their 2019+ genuinely departs from their training history (climatology fails them too, e.g. one well at −0.8), so the ensemble members *agree and are all wrong*. No training-time signal can flag that — it is a property of the data, not a tuning failure. The honest ceiling for this approach is "in-sample-parity median, with a handful of non-stationary wells climatology-gating can't rescue."

![Leave-one-well-out: climatology vs operator vs gated hybrid](results/figures/lowo_improvement.png)

*Per-well held-out KGE. Equilibrium anchoring lifts the operator (blue) above climatology and shrinks its tail; the self-consistency gate (green) trims most of what remains. The held-out-well median (0.56) now nearly matches the in-sample operator (0.59).*

**Do the in-sample forcing wins transfer here? No — instructively.** The recharge-memory and ET that lift the *in-sample* operator to 0.754 do **not** help leave-one-well-out — they slightly *hurt* the gated hybrid (inner-split 0.626 → ~0.61). Two compounding reasons: (1) recharge-memory adds per-well rainfall gains the hypernetwork must predict for an *unseen* well from attributes alone — extra capacity it cannot place; (2) richer forcing lets the operator fit each well's *training* history better, which fools the self-consistency gate into over-trusting wells that don't generalize. The simplest operator generalizes best. A clean **capacity-vs-generalization tradeoff**: forcing that helps when you *have* the well hurts when you *don't*. (`leave_one_well_out` accepts `rain_memory` to reproduce this.)

### Continuous head field (PINN) — a documented negative result

We also tried the more ambitious "physical AI" framing: a single physics-informed neural network learning a **continuous head field** `h(x, y, t)` over the whole fan (2D depth-averaged groundwater-flow PDE, autodiff residual, learned transmissivity field — the NVIDIA PhysicsNeMo wheelhouse). It is implemented (`hydrophysics/models/pinn_field.py`, `--model pinn`, 22 tests incl. an analytic PDE-residual check) and reported honestly here because **it does not beat the per-well models on this data**, and *why* is the interesting part.

| Model (2019+ val, sim mode) | KGE median | beats climatology | reference |
|---|---|---|---|
| climatology | 0.446 | — | — |
| **SpatialPINN** in-sample | 0.334 | 21/61 | gray-box 0.736 · UDE 0.591 |
| **SpatialPINN** LOWO unanchored | 0.105 | 13/61 | UDE LOWO **0.565** |
| SpatialPINN LOWO anchored | 0.189 | 13/61 | — |

`physics_weight` was selected on the inner pre-2019 split; 2019+ scored once. **Why it underperforms** is structural, not a tuning miss: a field conditioned only on `(x, y)` can't give each well its own dynamics, and on a fan where wells are 5–10 km apart with different local behavior, coordinates alone barely place an unseen well (LOWO 0.10). The lumped operator wins LOWO (0.565) precisely because it conditions on each well's *observable history* and *anchors to its observed mean* — per-well information the pure field forgoes (adding only the mean back lifts LOWO 0.10 → 0.19). A competitive field model would be *the UDE plus a spatial deviation field*, not a pure field — left as future work.

The genuine deliverable that survives is the **continuous map**: the learned field is hydrogeologically plausible (a smooth inland→coast head gradient), even though its per-well KGE is modest.

![PINN continuous head field over the fan](results/pinn/head_field_map.png)

*Learned head field `h(x,y,t)` on **2020-12-31 (day index 3287)**, masked to the real fan polygon (study-area shapefiles). High head inland (SE, ~+63 m) declining toward the coast (NW, ~−23 m) — a physically sensible interpolation between the 61 wells (markers). Reproduce: `python -m hydrophysics.maps --headfield --day 3287`. Full write-up: [`docs/superpowers/specs/2026-06-16-spatial-pinn-head-field-design.md`](docs/superpowers/specs/2026-06-16-spatial-pinn-head-field-design.md).*

### Interactive explorer: head + land subsidence

```bash
python -m hydrophysics.explorer --n 50   # writes results/explorer/choushui_explorer.html
```

Builds a standalone, self-contained HTML — a date-slider animation over the real Zhuoshui
fan with a **Head / Subsidence** toggle:

- **Head** surface: monthly IDW interpolation of the 61 **observed** wells (labelled
  interpolation, not a model), masked to the fan polygon, with wells and the 14 multi-layer
  compaction wells (MLCW) marked.
- **Subsidence** surface: `S(x,y,t) = Sk · cumulative head drawdown`, with the single
  coefficient `Sk` calibrated against the real MLCW compaction at the 14 georeferenced
  sites.

A validation panel plots predicted vs observed compaction at those sites. **Honest result:
no head→subsidence coupling we tried generalizes** (`python -m hydrophysics.subsidence_report`):

| coupling | evaluation | R² | verdict |
|---|---|---|---|
| single basin-wide `Sk` | in-sample, pooled (`calibrate_sk_from_pairs`) | −0.28 (compaction) | worse than the mean |
| spatial IDW of per-site `Sk` | reported in the coast-regression design doc; not reproduced by a script in this repo | −2.40 (compaction) | worse |
| `Sk = exp(β₀+β₁·distance_to_coast)` | **leave-one-site-out** (`loso_sk_regression`) | −0.29 (Sk-space gate) · −2.10 (compaction) | **fails the gate** |

Only the coast-regression row above is an honest leave-one-site-out number; the single-`Sk`
row is an in-sample fit (it is fit and scored on the same 14 sites) and is not directly
comparable to it. The Stage-1/Stage-2 twin-track gates below redo this comparison
like-for-like — see "Results" in the differentiable-twin design doc.

Each MLCW site is individually well-explained (per-site `Sk` in-sample R² = 0.81), and `Sk`
tracks distance-to-coast in-sample (corr = −0.68, the marine-clay gradient). But with only
14 widely-spaced sites whose `Sk` spans 28×, **no model predicts a held-out site** — the
distance-to-coast regression's leave-one-site-out Sk-R² is −0.29 (gate: pass if > 0). So the
subsidence *surface* stays an **illustrative single-`Sk` proxy, labelled as such**; what is
trustworthy is the observed-head animation and the real MLCW compaction the panel shows. This
is the same spatial-sparsity wall the PINN hit. Specs:
[explorer](docs/superpowers/specs/2026-06-19-choushui-head-subsidence-explorer-design.md) ·
[coast regression](docs/superpowers/specs/2026-06-19-per-site-sk-coast-regression-design.md).

### Forecast mode (operational, data-assimilated)

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

**Does ET help the forecaster? No — and that is the point.** The ET driver that lifts the *free-running* operator by +0.05 gives the *assimilated* forecaster essentially nothing (2019+: 7-day 0.965 → 0.964, 30-day flat). Same reason the forecaster is strong: it assimilates recent observed levels, which already encode the ET-driven state, so explicit ET is redundant. Put beside the [leave-one-well-out finding](#generalizing-to-unseen-wells) — where the extra forcing actually *hurts* unseen-well generalization — this maps a clean principle: **explicit physics forcing helps most where information is scarcest (free-running operator, +0.05) and least where it is richest (assimilated forecaster, ≈0; or extrapolating to a never-seen well, where the added capacity hurts).**

</details>

## Status

- **Digital twin (GPU):** `hydrophysics.twin` — fan grid, four-layer differentiable flow solver with open boundaries, electricity-census pumping driver, VEP compaction column, Stage-2/3 gates, the forward policy twin with observation nudging and a fold-ensemble spread, the 3D viewer, and a PhysicsNeMo FNO surrogate. See "Digital twin" above and `docs/superpowers/STATE.md`.
- **Foundation (done, tested in CI, runs anywhere):** dataset loader, KGE/NSE/RMSE metrics with explicit simulation-vs-forecast modes, gray-box + climatology + last-value baselines, reproducible benchmark, synthetic sample, GitHub Actions CI (ruff + pytest on Python 3.10–3.12).
- **Models (GPU):** a working `GlobalGRU` reference model, the `PhysicsUDE` physics-informed operator (hypernetwork + stable semi-implicit ODE rollout + physics-residual loss), and `PhysicsNeMoUDE` — the same operator **ported to NVIDIA PhysicsNeMo** with `.mdlus` checkpointing, reproducing the headline simulation result on CUDA. Multi-timescale rainfall **recharge-memory** plus an **evapotranspiration driver** (net recharge = rain − ET₀, ET₀ from Open-Meteo/ERA5 via AquaScope) lift the operator's simulation KGE from 0.591 to **0.754**, reaching/exceeding the per-well gray-box (0.736) with one shared network. Leave-one-well-out generalization improved from 0.236 to **0.565** (above climatology, near in-sample) via observable history signatures + equilibrium anchoring + a self-consistency gate.
- **Forecasting (GPU):** `GlobalForecastLSTM`, a global attribute-aware multi-horizon forecaster with data assimilation and optional **bf16 mixed-precision** training (14× over CPU measured on an Ampere-class card; on the project's Turing server bf16 has no hardware support and the fp32 path is used — see GPU performance), scored against persistence/climatology by `hydrophysics.forecast_eval`. Beats persistence at 7- and 30-day horizons, with a probabilistic (Gaussian) mode giving calibrated prediction intervals and CRPS/coverage scoring (see Forecast mode above).
- **Simulation baselines:** a per-well **signal-decomposition** model (`hydrophysics/decomp.py`) reaching the gray-box accuracy class (median KGE 0.75, leakage-free), and the `SpatialPINN` continuous head field (documented negative result, but a hydrogeologically plausible map).
- **Tooling:** `hydrophysics.figures` (reproducible plots), `hydrophysics.bench` (CPU vs CUDA vs CUDA+AMP throughput), and `hydrophysics.maps` (study-area KGE + PINN head-field maps over the real fan/river/coast shapefiles).

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
2. **Mixed precision (done, hardware-dependent).** The forecaster trains under bf16 autocast (`amp=True`). On an RTX 4070 SUPER (Ada) it is **14× faster than CPU and uses ~half the GPU memory** of fp32 (see GPU performance below). The project's own server is a Turing card with no bf16 support, where `amp=False` fp32 is the path that runs; the twin's solver is float64 throughout and does not use AMP at all. Reproduce: `python -m hydrophysics.bench`.
3. **CUDA training (done).** `train.py` auto-selects `cuda > mps > cpu`; all models are standard PyTorch and train on any CUDA GPU as-is.
4. **GPU-parallel operator rollout (done).** `--rollout scan` evaluates the same semi-implicit recurrence chunk-parallel (banded-triangular matmuls instead of ~4000 sequential kernel launches): **6.7× faster operator training** (259 s → 39 s for 1500 epochs on the real 61-well data), accuracy-neutral within seed variance (5-seed median KGE 0.62 ± 0.04; the sequential-loop runs land inside that spread). Reproduce: `python -m hydrophysics.bench_port`.
5. **Adjoint rollout (done — honest negative).** `--rollout adjoint` integrates the same ODE with `torchdiffeq.odeint_adjoint` (constant-memory backprop, validated by tests). At this scale it is not worth it: the full training graph is only ~10 MiB, and the adaptive continuous-time solver is orders of magnitude slower against daily piecewise-constant forcing — the discrete semi-implicit scan is the right integrator here. bf16 autocast for the operator (`--amp`) is likewise measured and neutral: the operator's training is kernel-launch-bound, not compute-bound (AMP's 14× win is the forecaster's, above).
6. **Not yet.** PhysicsNeMo multi-GPU / distributed training (this box has a single GPU, so scaling claims would be unverifiable here).

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
  lowo.py          leave-one-well-out cross-well generalization (+ gated hybrid)
  decomp.py        per-well signal-decomposition baseline (gray-box-class)
  et.py            Open-Meteo FAO-56 ET0 net-recharge driver (rain - ET0)
  sample.py        synthetic dataset generator (CI / no-data users)
  run_baselines.py freeze + print the baseline tables
  train.py         load -> fit -> simulate -> benchmark (the GPU entry point)
  forecast_eval.py horizon-wise forecast scoring (point + probabilistic)
  figures.py       generate the README figures (hydrographs, fan, bars)
  maps.py          study-area maps: per-well KGE + PINN head field over the fan
  explorer.py      Choushui head + land-subsidence explorer
  subsidence.py    inelastic-compaction (Sk) regression vs cumulative drawdown
  bench.py         CPU vs CUDA vs CUDA+AMP throughput benchmark
  viz.py           plotting helpers (lazy matplotlib)
  models/
    base.py        GroundwaterModel interface (fit + simulate)
    gru.py         GlobalGRU reference model (working, GPU-ready)
    ude.py         PhysicsUDE physics-informed operator (the new method)
    ude_physicsnemo.py  PhysicsNeMoUDE: the UDE on NVIDIA PhysicsNeMo
    forecast_lstm.py    GlobalForecastLSTM (assimilated, probabilistic, AMP)
    pinn_field.py       SpatialPINN continuous head field h(x, y, t) over the fan
  twin/
    grid.py, zones.py, boundaries.py   fan grid, proximal/mid/distal zones, coast + apex GHB
    flow.py            differentiable 4-layer flow solver (implicit adjoint, float64 CG)
    pumping.py         electricity census -> abstraction; clean_census (dedupe + capacity)
    heads.py           layer-resolved QC'd head field from the API wells
    compaction.py      visco-elasto-plastic column; coupled.py  flow <-> column
    calibrate_mlcw.py  Stage-2 gate;  calibrate_flow.py  Stage-3 fit + k-fold gate
    scenario.py        pumping policies + climatology;  inputs.py  one loader for the twin
    forward.py         the forward twin (hindcast, nudging, projection, ensemble)
    uncertainty.py     Laplace posterior of the flow parameters -> ensemble members
    calibrate_coupled.py  Stage 4: refit the column on the flow model's heads, 3 configs
    surrogate.py       PhysicsNeMo FNO surrogate of the calibrated solver
    explorer3d.py      3D viewer (head-decline mode and forward/policy mode)
    fetch_amp.py       env-driven rebuild of the AMP_V2 data cache from the API
results/phase0/    frozen baselines + spatial map
results/figures/   reproducible figures (from the synthetic sample)
results/bench/     GPU throughput benchmark
app/               live Gradio demo (deployed to Hugging Face Spaces)
tests/             foundation + numerics + model-smoke + forecast/explorer tests (CI)
```

## Contributing

Contributions are welcome — from one-line docs fixes to open research on cross-well generalization.

**New here? Start with one of these:**

- 🟢 [**Good first issues**](https://github.com/Rekin226/HydroPhysicsAI/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — newcomer-sized, self-contained (logging, coverage, a small data/summary improvement).
- 🟣 [**Up for grabs**](https://github.com/Rekin226/HydroPhysicsAI/issues?q=is%3Aissue+is%3Aopen+label%3A%22up+for+grabs%22) — claimable tasks across all sizes; comment to claim one.
- 🔵 [**Research / help wanted**](https://github.com/Rekin226/HydroPhysicsAI/issues?q=is%3Aissue+is%3Aopen+label%3Aresearch) — meatier modeling work (adjoint rollout, simulation uncertainty, richer conditioning, a transfer pilot). Negative results documented honestly are valued here.

Each issue lists context, the files to touch, and acceptance criteria. Please comment to claim an issue before starting (especially the larger ones) so work isn't duplicated.

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the GPU / CUDA / PhysicsNeMo install, the model contract, and the evaluation rules that keep the benchmark honest.

## Author

Abdoul Rachid Ouedraogo, Ph.D. — hydrogeology x AI. Also: [AquaScope](https://github.com/Rekin226/aquascope).

## License

MIT — see [LICENSE](LICENSE).
