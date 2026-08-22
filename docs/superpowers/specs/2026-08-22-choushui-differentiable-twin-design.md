# Choushui Differentiable Digital Twin — Design

**Date:** 2026-08-22
**Status:** Approved (brainstorming), pending spec review
**Author:** brainstormed with Claude
**Doubles as:** pre-registration. Gates, baselines and success criteria below are fixed
*before* any run. See §9.

## Summary

A GPU-native, end-to-end **differentiable quasi-3D model of the Choushui River alluvial
fan** that couples four-layer groundwater flow to a visco-elasto-plastic (VEP) compaction
column, learns its spatially varying parameters from hydrogeological attributes, and
inverts for the pumping field from three independent observation channels. Its purpose is
**counterfactual scenario forecasting with calibrated uncertainty**: what happens to
subsidence if abstraction changes.

The scientific gap it fills, established by the review in `docs/LIT_SUBSIDENCE_TAIWAN.md`:
coupled 3D flow-deformation modelling exists on this fan (Ni; Shih-Jung Wang), and
deep learning exists on this fan (Chu; Ku), but **no differentiable model exists anywhere**
— OpenAlex returns 52 works for `"physics-informed" AND (subsidence OR land deformation OR
InSAR)`, exactly one couples heads to subsidence with a PINN (Dezhou, China), and
`hybrid physics machine learning land subsidence compaction` returns **zero**.

## Why the incumbent approach failed (the motivation)

`hydrophysics/subsidence.py` fits `S = Sk · cumulative_drawdown` and reports honestly
negative leave-one-site-out R² (−0.28 to −2.40, README). The literature explains it:
Tsai & Hsu 2018 (`10.1016/j.enggeo.2018.07.025`) show deformation on *this fan* is
visco-elasto-plastic — elastic, plastic **and viscous with a delay** — and Lees et al. 2022
(`10.1029/2021WR031390`) find residual clay compaction time constants of decades. A
memory-less scalar cannot absorb a rheology. The negative result is correct and expected.

## Scope decisions (locked)

- **Quasi-3D, not full Biot.** Vertical strain dominates; the MLCW observations *are* 1D
  vertical profiles, so the model is calibrated at the resolution of the data; and it is
  the physics the subsidence community already accepts (MODFLOW SUB-WT). Full 3D
  poroelasticity + neural-operator surrogate is **Approach B, explicitly deferred** —
  attempted only after this succeeds, seeded by this model as data generator.
- **One-way coupling, flow → compaction.** Compaction-induced storage loss is real but
  second-order, and adding it makes the system stiff. Deferred, not designed in.
- **Implicit differentiation through the linear solve**, not backprop through CG
  iterations. Exact gradients, memory independent of iteration count.
- **Base grid 500 m** (8,567 active cells × 4 layers over the 2,144 km² fan polygon;
  ~1.2 GB stored flow state for 22-year daily backprop). 250 m (34,302 cells, 4.6 GB) is a
  convergence check and a stretch target, not the base.
- **Gradient checkpointing over monthly blocks** for the time loop.
- **No RL.** This is estimation, which gradients solve better. If optimal pumping policy is
  wanted later, differentiability permits gradient-based optimal control, which strictly
  dominates RL for a differentiable model. Deferred.
- **No additional DL module.** The attribute network *is* the learned component; a separate
  neural "pumping predictor" would break the physical closure of the energy relation.
- **Out of scope (YAGNI):** horizontal deformation; seawater intrusion; solute transport;
  land-use change modelling; real-time operational deployment.

## 1. Domain and data

Fan polygon: 2,144 km², bbox 58.6 × 76.6 km, EPSG:3826.

| Channel | Source | Extent | Role |
|---|---|---|---|
| Heads | `gw-wra-gw10min-obs`, zone 50 | **344 wells**, layer-coded (103/134/60/19 across aquifers 1–4), 206 confined / 77 unconfined, screens 17–306 m; 10-min at most sites | State observation; AMP source |
| Compaction (depth-resolved) | `ls-wra-mlcw-obs` | **16 fan sites**, magnetic rings at 25 depths to 300 m, 1 mm precision (Hung et al. 2021 `10.1029/2020wr028194`) | Strongest compaction constraint |
| Leveling | `ls-wra-lsp-obs` | **1,239 benchmarks** in bbox, 556 with ≥5 obs spanning 2012→2019+, ~annual, median −1.9 cm/yr | Spatial compaction constraint |
| Pump census | `etc-tpc-etc1mon-obs` | **116,769 registered pumps**, all Yunlin, with coordinates, `PUMP_HP` (459,124 HP total), purpose class, irrigation district; **monthly kWh 2007-01→2025-07** | Pumping volume (registered) |
| High-frequency power | `etc-ncku-etc1min-obs` | 475 meters, 1-min V/I/P, 2022–2025 | Calibration of pump efficiency η |
| GNSS | `ls-wra-gnss-obs` | 38 stations, daily, **2020-01 onward only** | Late-window deformation check |

**Known coverage limits, to be stated in any write-up:** the pump census is Yunlin-only
(Changhua and Chiayi have monitoring wells but no census); it covers registered pumps only;
and no per-pump screen depth exists, so aquifer allocation must be inferred.

## 2. Forward model

**Flow**, per aquifer layer `l ∈ {1..4}`, five-point finite volume, backward Euler, daily:

```
S_l ∂h_l/∂t = ∇·(T_l ∇h_l) + L_{l-1,l}(h_{l-1} − h_l) + L_{l,l+1}(h_{l+1} − h_l) + R_l − Q_l
```

with harmonic-mean face transmissivities, `R` recharge into layer 1 from rainfall−ET, `Q`
pumping. Gradients via implicit differentiation of the linear solve.

**Compaction**, a 1D VEP column per cell, driven by that cell's layer heads
(`Δσ' = −γ_w Δh`), state `(ε_e, ε_i, h_pc)` per compressible interbed:

- elastic, coefficient `S_ke`, always active, fully recoverable;
- inelastic, coefficient `S_kv` (10–100× larger), gated on `h < h_pc`;
- viscous relaxation toward equilibrium: `τ · dε_i/dt + ε_i = ε_i^eq(h)`.

`h_pc` is a **learned, evolving state**, not a running minimum — the variable-preconsolidation
upgrade of Li et al. 2022 (`10.1016/j.jhydrol.2021.127420`).

Surface subsidence `S(x,y,t) = Σ_l b_l ε_l(t)`.

## 3. Parameterization

One shared network maps per-cell attributes → log of every physical parameter
(`T`, `S`, `L`, `S_ke`, `S_kv`, `τ`, initial `h_pc`). Attributes: fine-grained fraction,
layer thickness, depth to layer, ground elevation, distance to coast, proximal/mid/distal
position. Log-parameterization enforces positivity; published ranges act as soft priors.

**Mandatory ablation, treated as a deliverable, not diagnostics:** attribute-driven
parameters vs free per-site parameters. The gap between them measures how much rheology is
predictable from hydrogeology — the question Shih-Jung Wang's group has published on
directly (`10.1016/j.enggeo.2022.106543` at Huwei, Yunlin, inside this study area;
`10.1016/j.enggeo.2025.107991` on borehole density). The answer is publishable either way.

## 4. Pumping: three channels, over-determined

```
electricity  E(pump, month)  →  Q = η · E / (ρ g · lift)     lift = ground elev − simulated head + losses
AMP          a(well, day)    →  Q_local ≈ a · C(T, S)        C = drawdown coefficient from learned T, S
heads        h(well, t)      →  the state both must explain
```

Only `η` (a few parameters per purpose/HP class, calibrated on the 475 high-frequency
meters) and `C` (supplied by the learned `T`, `S`) are unknown, and both are
low-dimensional. This replaces a low-rank spatiotemporal inversion and removes the largest
technical risk in the project.

**The residual between the two pumping estimates is modelled explicitly as the
unregistered-abstraction field**, not absorbed into noise.

**Physical consequence worth stating:** as heads fall, lift rises, so the same electricity
delivers less water. This energy–water feedback is real, policy-relevant, and unmodelled on
this fan. It also lets a scenario be expressed in something a regulator controls —
electricity supply or tariff to agricultural wells.

### 4.1 AMP v2

Extends Ouédraogo, Hsu & Wang 2023 (`10.1061/JHYEFF.HEENG-5760`), which established AMP on
three Tuku wells from hourly data and stated its own limitation: *"the multiplication of AMP
by the drawdown coefficient results in the average pumping rate… the inclusion of the aquifer
properties will lead to interesting results."* This model supplies those properties.

Verified on the 61 cached wells before writing this spec: **38 of 49 analysable wells have
their dominant 0.5–5 cpd spectral peak at exactly 1.00 cpd**, SNR median **63×** over the
3–5 cpd noise floor (>3× at 48/49); A(1 cpd) median 0.92 cm, p90 4.9 cm, max 11.2 cm; a
2 cpd harmonic exceeding 30% of the 1 cpd line at **24 wells** (duty-cycle structure);
sampling is 10-min at 36 wells. Tidal 1.93 cpd stays 0.1–1.7 cm, negligible as the original
paper argued.

v1's fixed narrow band-pass at 1 cpd assumes stationarity, which the harmonic structure and
the 2021 drought both violate. v2 uses **GAFD + Hilbert (HGT)** per Lin et al. 2023
(`10.3390/s23083785`) to obtain *instantaneous* amplitude and frequency — pumping intensity
**and** schedule — adaptively, without mode mixing or boundary effects. Band-pass vs HGT is
an ablation, not an assumption.

AMP contributes what the census cannot: unregistered wells (hydraulic stress regardless of
registration), aquifer-layer discrimination (the original paper's layer-1 sites read
0.22/0.08 m against layer-2 at 0.02 m), coverage outside Yunlin, and daily rather than
monthly timing.

## 5. Observation model and uncertainty

Three operators, one Gaussian likelihood, weights `1/σ²` from instrument precision (MLCW
1 mm, leveling mm-level, heads cm-level) so no weight is a free tuning knob:

- heads → sample layer `l` at cell `(i,j)`;
- MLCW → compaction *between magnetic-ring depths* (depth-resolved, far stronger than a
  surface number);
- leveling → total column strain **after tectonic detrending** (§6).

**Uncertainty:** deep ensemble, 8–10 independent fits with different seeds and bootstrapped
observations. Scenario bands are ensemble spread. For counterfactuals the dominant
uncertainty is parameter uncertainty in `τ` and `S_kv`, which an ensemble captures honestly.

**Optimization is staged, never joint-from-scratch:** warm-start compaction from Stage 2 and
flow from Stage 3, then fine-tune jointly.

## 6. Two threats, addressed by construction

**Tectonics.** Ching et al. 2011 (`10.1029/2011jb008242`) built Taiwan's vertical velocity
field from 1,843 leveling benchmarks and 199 cGPS: uplift 0.2–18.5 mm/yr inland, subsidence
on the coastal plains. Benchmark elevation change is therefore tectonic + anthropogenic. The
pumping signal swamps tectonics near the coast, but at proximal/inland benchmarks drawdown
is small, so the ratio is worst exactly where the fit is most fragile. **Mitigation:** fit or
subtract a per-site tectonic linear offset and *report the variance it absorbs*.

**Attribute information limit.** See §3 — handled as a measured ablation rather than an
assumption.

## 7. Staging and gates

Each stage is independently valuable and each gate is a genuine continue/kill decision.

| Stage | Deliverable | Gate |
|---|---|---|
| 0 | Data foundation: 344 heads, 1,239 leveling, 16 MLCW, grid + 4-layer geometry, attribute rasters, tectonic detrend | Panel assembled; tectonic correction measured |
| 1 | Refit existing algebraic `Sk` on 556 leveling sites instead of 14 | A number that motivates or descopes Stage 2. If LOSO R² > 0 the wall was sparsity, and the *rheology* is simplified accordingly — the flow/pumping stages still stand, since counterfactuals need them regardless |
| 2 | Differentiable VEP compaction column, calibrated at 16 MLCW sites | Beats algebraic `Sk` on leave-one-site-out at MLCW sites |
| 3 | Differentiable 4-layer flow + AMP v2 + electricity-driven pumping | Predicts held-out wells better than current IDW interpolation |
| 4 | Coupled model, joint calibration on heads + MLCW + leveling | **Positive leave-one-site-out R² on leveling subsidence** |
| 5 | Counterfactual scenarios + ensemble uncertainty | Physically coherent rebound/residual behaviour; calibrated coverage |
| — | Agentic research loop, from Stage 2 onward | Ledger + lab notebook; sealed-gate discipline holds |
| 6 | *(deferred)* Approach B: PhysicsNeMo neural-operator surrogate, 3D visualization, optimal control | Not planned here |

Stages 1 and 2 can each kill or redirect the project within weeks, before the expensive
machinery exists. That ordering is the main risk control.

## 8. Agentic research loop

Extends existing conventions (`train.py` argparse CLI, `results/<name>/`, `inner_select.py`).

**Division of labour:** numeric hyperparameter search goes to ASHA/Bayesian optimization —
an LLM must not do what Optuna does better. The agent's role is reading the accumulated
ledger, forming hypotheses ("variants without the viscous term fail at coastal sites but not
inland"), and proposing **structural** experiments.

**Guardrails:**

1. **Sealed test set by construction** — the data layer refuses held-out sites and 2019+ to
   the agent. Not a convention; an architectural constraint.
2. **Multiple-comparisons accounting** — top-k candidates get *one* sealed-gate evaluation
   at the end, reported *with the number of comparisons made*.
3. Fixed experiment budget and stopping rule per round.
4. Full provenance per run: config hash, seed, git SHA, data version.
5. **Agent proposes, human disposes on anything structural.** Hyperparameters and ablations
   are free; a new physical term needs sign-off.

**Artifacts:** a structured ledger (one row per run) and a running markdown lab notebook,
which becomes the ablation section of any write-up.

## 9. Validation (pre-registered)

**Software correctness.** Theis drawdown (confined, single well) and Hantush (leaky) must be
reproduced. Mass balance to machine precision each timestep. Finite-difference vs autograd
gradient checks. Compaction analytic limits: elastic loading/unloading recovers exactly;
inelastic never recovers; `τ→0` gives the instantaneous limit; `τ→∞` gives no compaction.
Grid convergence 1000 → 500 → 250 m.

**Synthetic twin recovery is the single most important test:** generate from the model with
known parameters, add realistic noise, refit, verify recovery. Parameters not recoverable
from our own synthetic data are meaningless on real data.

**Baselines, named in advance:** persistence/climatology; the existing algebraic `Sk`;
**Chu et al. 2021** spatially-varying drawdown function (`10.1016/j.ejrh.2021.100808`, the
published incumbent for this fan); and a pure-ML model on identical inputs — that last one
is what demonstrates the physics earns its place.

**Held out in space and time:** leave-one-site-out *plus* a temporal holdout (train ≤2018).
The **2021 drought** is the designated out-of-distribution stress test.

**Independent corroboration of learned physics:** against Tsai & Hsu's VEP moduli and their
proximal→distal gradient, published `Sk` ranges, and the contested shallow-vs-deep compaction
split (Nguyen & Ni 2024 report half the major compaction is shallow; Lees et al. report >90%
from the lower confined aquifer in San Joaquin). A model that fits well while learning absurd
parameters must be detectable as a failure.

**Unregistered pumping carries a weaker evidentiary standard** — there is no ground truth. It
is validated only by (a) whether the residual field improves held-out well prediction over
assuming zero, and (b) whether total abstraction stays within WRA's published basin water
balance. Reported as a bounded inference, never as a measurement.

**Uncertainty calibration** via coverage/PICP on held-out data, matching the discipline
already applied to the forecasting track.

**Negative results will be reported.** If a full rheology with 40× more sites still fails to
generalize, that is a strong finding against a literature that has assumed sparsity was the
problem. Pre-committing removes the incentive to rationalize.

## 9a. Decomposition into implementation plans

This spec is deliberately larger than one implementation plan. It decomposes into three,
each written and executed separately, with the gates in §7 as the hand-off points:

- **Plan A — Stages 0–2** (data foundation, the `Sk` refit, the differentiable VEP column).
  Self-contained, needs no flow solver, and settles the central scientific question.
- **Plan B — Stages 3–4** (flow solver, AMP v2, pumping channels, coupling). Begins only
  after Stage 2's gate passes.
- **Plan C — Stage 5 + the research loop hardening** (scenarios, ensembles, calibration).

Only Plan A should be written now. Writing B and C before A's results exist would be
planning against unknowns.

## 10. Interfaces and module boundaries

New modules under `hydrophysics/`, each independently testable:

- `twin/grid.py` — fan polygon → masked grid, 4-layer geometry, attribute rasters.
- `twin/flow.py` — differentiable 4-layer flow solver. Input: parameters, forcing, pumping.
  Output: head field. Depends on nothing but tensors.
- `twin/compaction.py` — differentiable VEP column. Input: layer heads + rheology.
  Output: strain and surface subsidence.
- `twin/pumping.py` — electricity → volume, AMP → stress, unregistered residual.
- `twin/amp.py` — AMP v1 (band-pass) and v2 (GAFD+Hilbert); pure signal processing, no
  model dependency.
- `twin/observe.py` — observation operators and likelihood.
- `twin/params.py` — attribute network.
- `twin/calibrate.py` — staged optimization driver.
- `research_loop/` — experiment spec, ledger, agent interface, guardrails.

Each answers: what does it do, how is it used, what does it depend on. `flow.py` and
`compaction.py` in particular must be usable and testable without any of the others.

## 11. Open questions deferred to implementation planning

- Aquitard thickness/geometry source: derived from well screen depths and layer codes, or
  from published Choushui stratigraphy — decide during Stage 0 once borehole coverage is
  assessed.
- Pump→layer allocation: learned soft assignment vs HP/diameter heuristic; both are
  candidates for the Stage 3 ablation.
- Recharge model: existing ET module (`hydrophysics/et.py`) vs a calibrated fraction of
  rainfall.
