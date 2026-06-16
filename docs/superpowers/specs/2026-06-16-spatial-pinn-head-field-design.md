# Spatial PINN Head Field — Design

**Date:** 2026-06-16
**Status:** Approved (brainstorming), pending spec review
**Author:** brainstormed with Claude

## Summary

Extend HydroPhysicsAI from a 0D-in-space, per-well lumped UDE to a **continuous
spatial head field** `h(x, y, t)` over the Zhuoshui alluvial fan — a
physics-informed neural network (PINN) trained against the 61 wells as scattered
observation points, regularized by the 2D depth-averaged groundwater flow PDE.

One model, reported three ways:
1. **In-sample simulation** vs the per-well gray-box ODE (bar: KGE 0.736; current
   UDE 0.591).
2. **Leave-one-well-out (LOWO) spatial skill** — predict a well never seen in the
   data loss, from the field the neighbors + physics build. **Headline metric is
   the unanchored (pure-spatial) LOWO score**, reported alongside the anchored
   variant for comparability with today's 0.565.
3. **Continuous head map** `h(x,y,t)` + the learned transmissivity field `T(x,y)`
   across the whole fan, with LOWO as its honesty check.

This is the NVIDIA Physical AI / PhysicsNeMo wheelhouse (PINN over a continuous
domain): plain-PyTorch autodiff first, optional PhysicsNeMo port after.

## Scope decisions (locked)

- **"3D" means a spatial (x, y, t) field**, not a volumetric (x, y, z) aquifer.
  The data has only 2D well coordinates, daily heads, rainfall, and a
  coastal/inland class — no aquifer-layer geometry, conductivity fields,
  screen depths, DEM, or pumping records. A faithful volumetric 3D solve would
  require inventing the subsurface, which would violate this repo's honest,
  out-of-sample-scored benchmark ethos. A depth-averaged 2D field is the most
  physically defensible upgrade the data supports.
- **Drop the per-well "upstream coupling" term.** Lateral flow between wells is
  now carried by the real PDE divergence `∇·(T∇h)`, not the lumped graph proxy.
- **LOWO headline = unanchored.** The held-out well's observed mean (the
  "equilibrium anchor" used to reach 0.565 today) is exactly what a spatial model
  should *earn* from neighbors, not be handed. The unanchored number is the
  headline even if it lands below 0.565; the anchored number is reported next to
  it for continuity.

## Architecture

### Core network
`f_θ(x, y, t, forcing_features) → h`

- **Fourier-feature MLP** (or SIREN) to defeat PINN spectral bias.
- **Nondimensionalization is mandatory**: `x, y → [0,1]` over the fan bounding
  box; `t →` years from the data start; head standardized to zero-mean/unit-std
  over training observations. PINNs are unforgiving about input/output scale.

### Learned interpretable spatial fields
Small auxiliary sub-networks of `(x, y)` only:
- `T(x, y)` — transmissivity, parameterized as `exp(g_φ(x,y))` so it stays
  positive.
- `α(x, y)` — recharge gain on rainfall.
- `d(x, y)` — net discharge / drift, absorbing the unobserved pumping + ET sink.

Storativity `S` starts as a single learned scalar; promote to a field only if
the inner-split says it helps (YAGNI).

### Governing PDE (the inductive bias)
2D depth-averaged transient groundwater flow:

```
S · ∂h/∂t = ∇·(T ∇h) + α(x,y)·R(x,y,t) − d(x,y)
```

- `∂h/∂t`, `∇h`, `∇·(T∇h)` all from autodiff — no mesh, no finite differences.
- **`R(x,y,t)` is a continuous rainfall field** interpolated from the rainfall
  stations (`rf_stations.csv` coords + `rf_timeseries.csv`), e.g. inverse-distance
  weighting or a tiny interpolation net. This is how forcing reaches arbitrary
  collocation points.
- **Western coast boundary**: soft (penalty) sea-level Dirichlet condition built
  from `water/sea_TWD97.shp` + the coastal/inland classification. Other fan
  edges: optional soft no-flow (Neumann) — add only if residuals demand it.

## Data flow

```
rf_stations + rf_timeseries ──► R(x,y,t) rainfall field
gw_stations (x,y) + gw_timeseries (daily head) ──► well observation points
sea_TWD97.shp + coastal class ──► coast boundary geometry
                    │
                    ▼
          nondimensionalize (x,y,t,h)
                    │
        ┌───────────┴────────────┐
        ▼                        ▼
  data loss at wells      physics residual on
  (train period only)     collocation points (domain × time)
        └───────────┬────────────┘
                    ▼
              + boundary penalty
                    ▼
            train (Adam, residual-weight schedule)
                    ▼
   query h at well (x,y) over val dates ──► GroundwaterModel.simulate
                    ▼
            existing harness: KGE / NSE / RMSE
```

## Losses

`L = w_data · MSE_data + w_phys · MSE_residual + w_bc · MSE_boundary`

- `MSE_data`: head misfit at well observation points, **training period only**.
- `MSE_residual`: PDE residual squared at collocation points sampled over the fan
  domain × time window.
- `MSE_boundary`: soft coast (and optional no-flow) condition.
- Loss weights and the network/optimizer hyperparameters are tuned on the
  **inner pre-2019 split** (inner-train < 2018, inner-val 2018), identical to the
  existing discipline. The 2019+ benchmark is evaluated exactly once.

## Validation — three reports from one model

### 1. In-sample simulation
Train on all wells' pre-2019 data; query `h` at each well's `(x,y)` over 2019+;
score with the existing harness. Compare to gray-box 0.736 and current UDE 0.591.

### 2. LOWO (spatial) — headline
For each held-out well:
- Its observations are excluded **entirely** from `MSE_data` (no leakage — see
  testing). Its coordinate `(x,y)` and any static attributes remain available
  (the coordinate *is* the query input).
- Predict its full 2019+ series from the field built by the other wells + physics.
- **Unanchored** (headline): no use of the held-out well's observed mean.
- **Anchored** (secondary): shift to the held-out well's training mean, for
  apples-to-apples with today's 0.565.

### 3. Continuous head map
Render `h(x,y,t)` over the fan through time + the learned `T(x,y)` field. The map
is a product deliverable; its credibility is the LOWO numbers, not eyeballing.

## Repo integration

- **New** `hydrophysics/models/pinn_field.py` — the PINN, learned fields, PDE
  residual, training loop. Implements the existing `GroundwaterModel` interface
  (`simulate` returns per-well series at well coordinates) so `bench.py` and the
  benchmark table wire up unchanged.
- **New** `hydrophysics/field_inputs.py` — rainfall field interpolation + coast
  boundary geometry loading (keeps `pinn_field.py` focused).
- **LOWO** reuses / extends `hydrophysics/lowo.py`; the field model plugs into the
  same leave-one-out loop with the masking guarantee above.
- **PhysicsNeMo**: plain-PyTorch autodiff implementation first; an optional port
  mirrors the existing `ude_physicsnemo.py` pattern (same skill, framework
  utilities + mixed precision).

## Error handling & failure modes

- **PINN non-convergence / data-vs-residual imbalance**: monitor the two loss
  terms separately; tune `w_phys` on the inner split; consider a residual-weight
  ramp (warm up data fit, then enforce physics).
- **Spectral bias** (PINN smears sharp seasonal swings): Fourier features /
  SIREN; report this as a known risk.
- **Units / boundary leakage**: nondimensionalize once, centrally; unit-test the
  residual; keep BC soft initially.
- **Sparse-data overfit of `T(x,y)`**: 61 points is thin for a 2D field —
  regularize `g_φ` (smoothness/L2) and validate via LOWO, not in-sample fit.

## Testing

- **Physics residual correctness**: evaluate the autodiff residual on an
  **analytic 1D diffusion solution** and assert it matches the known answer.
- **LOWO no-leakage**: assert the held-out well's observations never enter
  `MSE_data` (programmatic check, not visual).
- **Interface smoke**: `pinn_field` satisfies `GroundwaterModel` and produces
  finite per-well series of the right shape over the val window.
- **Rainfall field**: interpolation returns station values at station coords and
  finite values between them.

## Staging

1. **Data plumbing** — fan bbox, nondimensionalization, rainfall field, coast BC.
2. **PINN core + in-sample benchmark** — network, learned fields, residual,
   training loop; report vs gray-box.
3. **LOWO spatial eval** — both unanchored (headline) and anchored.
4. **Map rendering + docs** — `h(x,y,t)` + `T(x,y)` figures, README + model-card
   update, optional PhysicsNeMo port.

## Out of scope (YAGNI)

- Volumetric (x, y, z) / multi-layer aquifer modeling — no supporting data.
- Pumping/ET reconstruction beyond the lumped `d(x,y)` sink.
- Real-time data assimilation (that's the separate forecast track).
- FNO / gridded-operator approach (61 scattered points → fabricated grid).
