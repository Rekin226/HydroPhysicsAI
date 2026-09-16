# Earlier phase (2026-06): one operator across 61 wells

This is the project's first phase, kept verbatim for the record. It benchmarked one
physics-informed neural operator across the **61 curated wells** of the original data
delivery (Zhuoshui alluvial fan, 2012–2022, validated out of sample from 2019) against
per-well gray-box ODEs, in free-running simulation and in assimilated forecasting, and
tried a continuous PINN head field and a 2D head/subsidence explorer.

It is no longer the headline. The digital twin (`README.md`) runs on a dataset three
times as dense in wells, layer-resolved, with the pumping forcing and the subsidence
network this phase never had, and its gates are physical (held-out wells, held-out
leveling sites) rather than a comparison with a baseline scored elsewhere. The gray-box
numbers below are the original project's own evaluation and were never re-scored on this
harness; the comparison is a record, not a claim.

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
| Physics-informed neural operator (baseline) | 0.591 | 0.49 | 1.07 | 61 |
| **+ rainfall recharge-memory** (this project) | **0.704** | **0.57** | **0.95** | 61 |
| **+ evapotranspiration driver** (this project) | **0.754** | **0.60** | **0.81** | 61 |

<sub>*A constant prediction has zero variance, so KGE is undefined; skill is read from NSE/RMSE. Forecast-mode persistence scores KGE 0.997 but is deliberately excluded: 1-step-ahead prediction of slow-moving groundwater is trivial and not comparable to a free-running simulation. The harness keeps simulation and forecast modes separate so the comparison stays honest.</sub>

The operator is one network across all 61 wells (attribute-conditioned hypernetwork to ODE parameters, level anchored to each well's training mean, per-well-weighted data + physics-residual loss, 1500 epochs on CUDA). The original instantaneous `b·rain` recharge term underfit the response — aquifers integrate rainfall over weeks to months — so we replaced it with **multi-timescale recharge memory**: the hypernetwork predicts one gain per causal memory state `api_k[t] = decay_k·api_k[t-1] + rain[t]` (decays 0.99/0.95/0.85 ≈ 100/20/6-day timescales). That single change lifts the operator from **0.591 to 0.704 median KGE** (NSE 0.49→0.57, RMSE 1.07→0.95, better on every metric), so **one shared, amortized operator now nearly matches the 61 hand-calibrated gray-box ODEs (0.704 vs 0.736)** — the gap closed from 0.145 to 0.032. The memory timescales were selected on an inner pre-2019 split (inner-train <2018, inner-val 2018) — reproduce that selection with `python results/ude/inner_select.py --device cuda`; the 2019+ benchmark was scored exactly once, and the baseline reproduces at 0.591. Reproduce: `python -m hydrophysics.train --model ude --rain-memory 0.99,0.95,0.85 --epochs 1500`. (The ingredient is borrowed from the per-well [decomposition](#per-well-signal-decomposition--reaching-the-gray-box-accuracy-class) below, now folded into the shared operator.)

## Adding the missing driver: evapotranspiration

Recharge is rainfall *minus* evapotranspiration, but the model only saw rainfall — so it could not explain ET-driven decline. We add the other half: **net recharge = rain − ET₀**, where daily FAO-56 ET₀ comes free (no API key) from Open-Meteo/ERA5 at each well, fetched via the [AquaScope](https://github.com/Rekin226/aquascope) toolkit. That lifts the operator from **0.704 to 0.754 median KGE** (NSE 0.57→0.60, RMSE 0.95→0.81, beats climatology on 58/61 wells) — now **at or above the gray-box's reported 0.736**. Strikingly, the out-of-sample gain (+0.050) is *larger* than the inner-split gain (+0.016) for a physical reason: the 2019+ window contains Taiwan's record **2020–2021 drought**, where decline is ET-driven and a rain-only model is blind — so the ET driver helps most exactly when prediction is hardest. The subtraction coefficient was selected on the inner 2018 split (`python results/ude/inner_select.py --device cuda --et`); 2019+ scored once; the control reproduces at 0.704. Reproduce: `python -m hydrophysics.train --model ude --rain-memory 0.99,0.95,0.85 --et --epochs 1500` (uses the committed ERA5 ET₀ cache; no fetch needed).

![Per-well gray-box KGE across the Zhuoshui fan](results/phase0/spatial_kge_graybox.png)

*Per-well validation KGE (gray-box baseline) over the real study area — the Zhuoshui alluvial-fan outline, rivers, and coastline (from the project shapefiles, TWD97). Each dot is a station at its true coordinates, colored by validation KGE; most wells are well-modeled (green/yellow), the dark wells are the hard cases a single attribute-conditioned operator should rescue. Reproduce: `python -m hydrophysics.maps --kge`.*

![Free-running simulation vs observed](results/figures/simulation_hydrographs.png)

*Free-running PhysicsUDE hindcast vs observed, validation period shaded. Generated from the bundled **synthetic sample** (`python -m hydrophysics.figures`) so the figure reproduces anywhere and ships no real agency series; the same command on real data redraws it for the 61 wells.*

## Per-well signal decomposition — reaching the gray-box accuracy class

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

## Generalizing to unseen wells

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

## Continuous head field (PINN) — a documented negative result

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

## Interactive explorer: head + land subsidence

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

**Does ET help the forecaster? No — and that is the point.** The ET driver that lifts the *free-running* operator by +0.05 gives the *assimilated* forecaster essentially nothing (2019+: 7-day 0.965 → 0.964, 30-day flat). Same reason the forecaster is strong: it assimilates recent observed levels, which already encode the ET-driven state, so explicit ET is redundant. Put beside the [leave-one-well-out finding](#generalizing-to-unseen-wells) — where the extra forcing actually *hurts* unseen-well generalization — this maps a clean principle: **explicit physics forcing helps most where information is scarcest (free-running operator, +0.05) and least where it is richest (assimilated forecaster, ≈0; or extrapolating to a never-seen well, where the added capacity hurts).**
