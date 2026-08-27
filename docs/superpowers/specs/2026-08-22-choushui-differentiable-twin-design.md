# Choushui Differentiable Digital Twin — Design

**Date:** 2026-08-22 (rev. 2026-08-23, §4/§4.1 after the AMP_V2 experiment)
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
AMP-G        a(t) · d(t)     →  Q_local ∝ amplitude × duty   rate × duration = volume
heads        h(well, t)      →  the state both must explain
```

Only `η` (a few parameters per purpose/HP class, calibrated on the 475 high-frequency
meters) and the AMP-G scaling are unknown, and both are low-dimensional. This replaces a
low-rank spatiotemporal inversion and removes the largest technical risk in the project.

**Revised 2026-08-23 after the AMP_V2 experiment.** The AMP channel was originally specified
as stress, `Q_local ≈ a · C(T,S)`, with the drawdown coefficient supplied by the learned
parameters. Measurement showed amplitude alone is a weak cross-well discriminator
(ρ = +0.255 against irrigation electricity, n=71) because it confounds pumping with `T` and
`S` — exactly the limitation the 2023 AMP paper flagged. **Duty cycle solves it**: being a
timing property it is not scaled by `T` or `S`, and `volume = amplitude × duty` reaches
ρ = +0.424 (p = 0.007), a 48% improvement on identical wells. The twin therefore ingests
volume, not stress.

**The residual between the two pumping estimates is modelled explicitly as the
unregistered-abstraction field**, not absorbed into noise.

**Physical consequence worth stating:** as heads fall, lift rises, so the same electricity
delivers less water. This energy–water feedback is real, policy-relevant, and unmodelled on
this fan. It also lets a scenario be expressed in something a regulator controls —
electricity supply or tariff to agricultural wells.

### 4.1 AMP-G (built and evaluated — see `AMP_V2/README.md`)

Extends Ouédraogo, Hsu & Wang 2023 (`10.1061/JHYEFF.HEENG-5760`). An unconstrained survey
of 34 wells over 11 years finds a **median 3.9 coherent spectral lines per well**; AMP
measures one. AMP-G detects lines against a local noise floor, complex-demodulates each,
attributes them physically, and inverts the harmonic ladder for **duty cycle**
(`A2/A1 = |cos(πd)|`, fitted jointly with aquifer attenuation `α`).

**What the evaluation established, and what it did not:**

| claim | verdict |
|---|---|
| HGT/GAFD beats the published band-pass on amplitude | **No.** 0.988 vs 0.994 synthetic; +0.231 vs +0.303 per-well real |
| Multi-band amplitude adds information | **No.** +7%, p = 0.19 |
| Sub-daily "irrigation rotation" band helps | **No — it hurts.** ρ 0.202, p = 0.027. Hypothesis rejected |
| Duty cycle is a new, aquifer-independent observable | **Yes.** volume ρ = +0.424 (p = 0.007) vs amplitude +0.286, same 40 wells |
| `α` is a usable T/S diagnostic | **Plausible.** Splits by layer: 0.21 at 38 m vs 0.49 at 119 m |

Bootstrap on the improvement: +0.137, 95% CI **[−0.008, +0.323]**, P(improvement) = 0.964.
Real but not conclusively established — it enters the twin as an observable with honest
uncertainty, never as a headline claim.

**Availability limit, load-bearing for Plan B:** duty needs ≥3 detected harmonics and is
recoverable at only **40 of 161** wells with a signature. The twin must use
`volume = amp × duty` where available and `amp_fund` elsewhere, with different noise
models for the two cases.

**AMP is irrigation-specific.** Seasonal correlation with rice/dry-crop electricity is
+0.66 to +0.75, but −0.17 with aquaculture and −0.15 with domestic. Consequence for §4:
the gap between AMP-implied and electricity-implied abstraction is **not** purely
unregistered pumping — it also contains the non-irrigation purpose mix, which must be
modelled separately before any unregistered-abstraction claim is made.

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

### Results (2026-08-23, commit `011df32`, final fix wave on `feat/choushui-twin-plan-a`)

**Stage 1 — 888 leveling sites** (`python -m hydrophysics.twin.sk_leveling --min-obs 5`):

| tectonic | n_sites | var_removed | sk_single (in-sample) | sk_loso_pooled | coast-regression LOSO (compaction) | coast-regression LOSO (Sk-space) |
|---|---|---|---|---|---|---|
| none | 888 | 0.000 | +0.119 | +0.116 | +0.062 | −0.102 |
| planar | 888 | 0.208 | −0.158 | −0.161 | −0.180 | −0.021 |

Gate: LOSO compaction R² > 0 means spatial sparsity was the wall; still negative means the
model form is. Result: positive (+0.062 to +0.116) with no tectonic correction, negative
once the planar tilt is removed — inconclusive rather than a clean pass, but nowhere near
as negative as the 14-site MLCW numbers below, consistent with sparsity being *part* of
the original problem even though the algebraic form still underperforms once the tectonic
correction is applied.

**Stage 2 — 14 MLCW compaction wells** (`python -m hydrophysics.twin.calibrate_mlcw --epochs 2000`):

| n_sites | n_cells | loss | sk_insample | sk_loso | sk_coast_loso | vep_loso | vep_shared_loso |
|---|---|---|---|---|---|---|---|
| 14 | 1296 | 0.0077 | −0.298 | −0.556 | −4.000 | −0.707 | −0.944 |

`sk_loso` (pooled single-`Sk`, LOSO, −0.556) and `vep_shared_loso` (one global 4-parameter
VEP column fit jointly across all training sites, LOSO, −0.944) are the like-for-like
model-form comparison: both are pooled estimators with no per-site covariates. The shared
VEP arm has only 4 parameters over ~1,296 training cells, so identifiability is not the
limiting factor for that arm -- and it still loses to a single scalar `Sk`. Stated plainly:
**the rheology, tested on equal footing, does not beat the algebraic baseline at these 14
sites.**

The per-site arm (`vep_loso`, 52 free parameters collapsed by a weighted mean of
log-parameters, −0.919) scores worse than the shared arm (−0.841) and worse than
`sk_loso` (−0.556), and 6/14 folds gave zero gradient on `log_skv`/`log_tau` in that
per-site fit (those sites carry only their initialization value into the average). Both
facts are recorded without over-interpreting them: the per-site number is not a clean
rheology result either, since a chunk of its parameters never moved.

Gate: VEP LOSO beats single-`Sk` LOSO on identical (h, obs, mask) arrays. The
`vep_shared_loso` arm (one global 4-parameter column fit jointly across all training
sites, not fit-per-site-then-averaged) is the structural test of the rheology itself,
per-item 2 of the final fix wave.

### Reproducibility and the init-scatter ensemble (2026-08-23)

The pipeline is **fully deterministic**: rerunning the committed code reproduces
`results/twin/stage2_vep_mlcw.csv` byte-for-byte, and `fit_column` is bit-identical across
repeats on both CPU and CUDA. An earlier note in this project claimed a ~0.2 run-to-run
swing; that was an error — the two figures compared came from different code versions
during the fix wave (in-sample loss 0.0077 before the anchor-alignment fix, 0.00466 after),
not from two runs of the same code.

Determinism is not robustness, though. `VEPColumn`'s initialisation is a fixed constant, so
every fit starts from the same point in a landscape already shown to be flat in several
directions. `--ensemble N --init-scatter S` repeats the decisive shared-parameter arm from
starts perturbed by `N(0, S)` in log space. Eight runs at `S = 0.5`:

| statistic | `vep_shared_loso` |
|---|---|
| mean | −0.8588 |
| sd | 0.0426 |
| min / max | −0.9627 / −0.8408 |
| runs beating `sk_loso` (−0.5562) | **0 / 8** |

Six of the eight land within 0.001 of the deterministic-init value (−0.8408), so the
optimiser reaches the same basin from scattered starts; one outlier at −0.963 sets the
spread. The gap to the pooled single-`Sk` baseline is 0.30, roughly **7× the ensemble
standard deviation**. The Stage-2 conclusion is therefore robust to initialisation: at
these 14 sites, the visco-elasto-plastic rheology does not beat a single scalar
out-of-sample, and that is not an artefact of where the optimiser started.

Reproduce with:

```bash
export HYDROMIND_GW_DATA="$(pwd)/chou-shui-data/chou-shui-data/data"
python -m hydrophysics.twin.calibrate_mlcw --epochs 2000 --ensemble 8 --init-scatter 0.5
```

### CORRECTION (2026-08-24): the Stage-2 gate PASSES — the earlier failure was a model bug

Everything recorded above for Stage 2 is superseded. `VEPColumn` computed the
preconsolidation head as `h_pc = min(h_pc0, h[:, 0])` with `h_pc0` initialised to **0.0** —
an absolute head, i.e. the survey datum. Inelastic strain accrues only where `h < h_pc`, so
**at every site whose heads never crossed 0 m the inelastic term was structurally disabled
and `log_skv` / `log_tau` received no gradient at all**. That was **7 of the 14** MLCW sites
(湖南國小, 溪州國小, 僑義國小, 土庫國中, 宏崙國小, 光復國小, 拯民國小). Zero is a datum
choice, not a physical preconsolidation head; the reviewer's "6/14 folds gave zero gradient"
observation was this bug, not a data limitation.

`h_pc0` is now an **offset relative to each site's starting head** — `h_pc = h[:, 0] +
h_pc0`, init 0 meaning "normally consolidated at t = 0" — which is datum-independent.
`tests/test_twin_compaction.py::test_inelastic_gate_opens_for_all_positive_heads` pins it.

Re-running the identical gate (14 sites, 1,296 cells, 2000 epochs):

| model | transfer rule | R² before fix | R² after fix |
|---|---|---|---|
| single `Sk` | in-sample | −0.298 | −0.298 |
| single `Sk` | LOSO, pooled | **−0.556** | **−0.556** |
| coast regression | LOSO | −4.000 | −4.000 |
| VEP | LOSO, per-site + mean transfer | −0.919 | −1.875 |
| **VEP** | **LOSO, shared global parameters** | −0.841 | **+0.324** |

In-sample loss fell from 4.660e-03 to **1.283e-04**, a 36× improvement, because half the
sites regained a working inelastic term. The init-scatter ensemble (8 runs, sd 0.5 in log
space) gives mean **+0.3232**, sd **0.0043**, range +0.3171 to +0.3290, and **8/8 runs beat
the pooled single-`Sk` baseline**.

**Revised Stage-2 verdict: PASS.** The shared-parameter VEP — 4 global parameters, no
per-site covariates, the exact structural analogue of pooled single-`Sk` — achieves a
**positive** leave-one-site-out R² on head→subsidence coupling. Every coupling previously
reported for this fan is negative (README: −0.28 in-sample single-`Sk`, −2.40 spatial-IDW,
−0.29 coast regression; and −0.556 for the honest pooled LOSO baseline computed here).

The per-site arm got *worse* (−1.875), which is consistent: with the inelastic term now
active everywhere, 4 free parameters × 13 sites overfits harder, and a covariate-free mean
is the wrong transfer rule. The pooled arm is the right estimator, and it is the one that
passes.

**Consequence for staging:** §7's gate table says a failing Stage 2 means "do not proceed to
Plan B". It passes, so Plan B (Stages 3–4: the differentiable four-layer flow solver, AMP-G
pumping channels, and coupling) is now justified on evidence.

**Diagnostic gap:** `results/twin/stage2_vep_per_site.csv` is written only for the per-site
arm, not the shared arm. For the record, that per-site arm gets 8 of 14 held-out sites
positive but is destroyed in the pool by two outliers (嘉興國小 −228.9, 北辰國小 −54.6).
Per-fold diagnostics for the shared arm should be added before Plan B.

**Caveat that survives:** Stage 1 is unaffected — the algebraic `Sk` coupling still fails on
888 leveling sites. What passes here is the *rheology with memory*, at the 14 depth-resolved
compaction wells, transferred by a pooled estimator.

### FINAL RESULTS (2026-08-25) — the head field was the binding constraint

The head field had been built from `chou-shui-data`'s curated **61** wells. That selection
came from the gray-box study, which required every well to have an *upstream partner* for
its ODE — a constraint with no bearing on subsidence. The provided raw file holds **174**
wells, and the API exposes **344** on the fan, each with a `GroundwaterLayerCode`.
`twin/heads.py` rebuilds the field from the API network: robust despike at median ± 15·MAD,
≥80% coverage, ≤180 d max gap, valid layer code → **147 wells** (L1 34, L2 69, L3 31,
L4 13), 1.1% NaN month-cells, nothing gap-filled. This is now the default (`--heads api`).

**Stage 1** — 878 leveling sites (10 benchmark resets screened at `--max-rate 0.5`), single-`Sk` LOSO pooled:

| head field | no tectonic correction | tilt removed |
|---|---|---|
| 61 curated wells | +0.116 | **−0.161** |
| **147 API wells** | **+0.214** | **+0.040** |

Head-field density was a real confound: the tilt-corrected gate flips from negative to
positive. Per-layer fields are *worse* than pooled (L1 −0.215, L2 −0.099, L3 +0.036),
which is expected — subsidence integrates compaction over the whole column, so no single
aquifer's head drives it. That is direct motivation for the four-layer solver.

**Stage 2** — 14 compaction wells, 1,296 cells, identical arrays:

| model | evaluation | 61-well heads | **147-well heads** |
|---|---|---|---|
| single `Sk` | in-sample | −0.298 | +0.190 |
| single `Sk` | **LOSO, pooled** (baseline) | −0.556 | **+0.106** |
| coast regression | LOSO | −4.000 | −0.120 |
| VEP | LOSO, per-site + mean | −1.875 | +0.070 |
| **VEP** | **LOSO, shared global** | +0.324 | **+0.478** |

Init-scatter ensemble on the final configuration (8 runs, sd 0.5 log-space): mean
**+0.4777**, sd **0.0023**, range +0.4739 to +0.4812, **8/8 beat the baseline**.

**Verdict.** With the full well network, the sparse-head-field confound is removed and the
picture is consistent: the algebraic `Sk` becomes weakly predictive out-of-sample
(+0.106 at MLCW sites, +0.040 across 878 leveling sites), and the visco-elasto-plastic
rheology adds **+0.37 R²** on top of it (+0.478 vs +0.106) under a pooled estimator with no
per-site covariates. Stage 2 passes decisively; Stage 1 is marginal, which is itself
informative — the algebraic form is adequate at compaction wells and not across the
leveling network.

**Plan B is justified**, and the layer results say what it must be: a genuinely four-layer
solver, not a single-layer proxy.

### Stage-3 result (2026-08-26) — FAIL, provisional

`hydrophysics/twin/{grid,flow,pumping,calibrate_flow}.py`. Four-layer differentiable flow on
a 1 km grid (2,148 active cells), float64, Jacobi-preconditioned CG with an exact adjoint,
verified against Theis and Hantush-style limits. Pumping from the 116,769-pump electricity
census; recharge from 26 rain gauges minus cached ET.

| quantity | value |
|---|---|
| wells / cells / grid | 136 / 2,148 / 1 km |
| parameters | 13 (homogeneous per layer) |
| in-sample R² | +0.126 |
| **10-fold R²** | **−0.204** |
| **IDW baseline R²** (identical folds) | **+0.904** |
| recovered T / S | ~3,000 m²/day / ~9e-4 |
| bounds binding | none |

**GATE: FAIL.** The physics model does not beat inverse-distance weighting.

Two things keep this honest rather than damning. Adding real forcing moved in-sample R² from
−0.022 to +0.126, and the recovered `T` sits inside the measured Choushui range (Liu et al.
2002, 58–6,034 m²/day) — the model behaves, it just does not fit. And `epochs=45` was chosen
to land the 11-fit gate inside two hours, not because the fit converged; by the criterion
stated before the run, an in-sample R² this poor reads as under-training, so **this gate is
provisional**.

The baseline is also genuinely hard: 147 wells over 2,144 km² is ~4 km spacing, and IDW
interpolates directly between the same wells the model is scored against.

Candidate limiters, in test order: (1) epoch budget, (2) homogeneity — 13 parameters cannot
express the fan's proximal-to-distal texture gradient, though Plan A settled that free
per-cell parameters are the wrong fix, (3) pump-layer allocation, currently all into layer 2.

**Plan C (Stage 4 coupling) does not start until this gate passes.**

### Stage-3 re-run (2026-08-27) — RETRACTED. The gate measures the wrong thing.

The 2026-08-26 FAIL above was under-trained, as suspected. Re-running at a converged budget
produced a second FAIL, and chasing an unrelated well-count discrepancy then showed that
result is **also** invalid. Both are recorded here because the reasons matter more than the
numbers.

**Confounder 1 — `--epochs` drives the LR schedule.** `fit_flow` builds
`CosineAnnealingLR(opt, T_max=epochs)`, so epoch *N* of a 400-epoch run is not comparable to
epoch *N* of a 100-epoch run. The "converges at epoch ~75-100" reading was taken off a
400-epoch trace; passing `--epochs 100` anneals to zero by epoch 100 and plateaus at in-sample
R² **+0.684**, whereas the 400-epoch schedule reaches **+0.943**. Same seed, identical at
epoch 1, divergent thereafter. Run the gate at the budget that produced the target trace.

**The re-run** (`--epochs 400 --n-folds 5`, 2h06m) reproduced the reference trace at all 17
logged points exactly and gave:

| quantity | value |
|---|---|
| wells / sites / cells / grid | 136 entries / **66 physical sites** / 2,148 / 1 km |
| parameters | 13 (homogeneous per layer) |
| in-sample R² | +0.9428 |
| 5-fold R² | +0.664 |
| IDW baseline R² (identical folds) | +0.880 |
| bounds binding | `log_T` 3 of 4, `log_S` 2 of 4, `log_eta` 1 |

**Confounder 2 — the folds leak by co-location, so the baseline is a near-oracle.** The
"136-well network" is 66 physical sites carrying layer-coded screens. `_kfold_indices` split
the 136 *entries* at random with no grouping, so co-located screens landed on opposite sides
of the split: **95 of 136 held-out entries (69.9%) sit at zero distance from a training
entry** (per fold: 61%, 93%, 63%, 78%, 56%). `idw_interp` weights by
`1/(d² + 1e-6)`, so a zero-distance source gets weight 1e6 against 1e-6 for a neighbour 1 km
out — a ratio of 1e12. For ~70% of the held-out set the "IDW baseline" is not interpolating,
it is copying another screen in the same borehole.

The flow model cannot exploit that: it reproduces each head through a 13-parameter
homogeneous solve, and the co-located screens sit in different layers. The gate therefore
pits a physics model against a near-oracle, and **+0.664 vs +0.880 is biased toward IDW by
construction**. Neither this FAIL nor the 2026-08-26 one is a physics verdict.

**What survives the retraction.** `bounds_hit` is a property of the fit, not of the split:
`log_T = [log 10, 5.304, log 10, log 10]` pins three of four layers at the **lower clamp**,
T = 10 m²/day, below the 58 m²/day floor Liu et al. 2002 measured at Choushui. So the +0.943
in-sample fit was bought by saturating parameters against their bounds rather than by finding
physical values. That remains the strongest evidence against the current parameterization,
and it is independent of how the folds are drawn.

**Two data-handling facts found on the way.** `calibrate_flow.py` silently drops wells whose
coordinates miss the active mask (`grid.active_index(...) is None` → bare `continue`): 147 →
136, all 11 from 3 sites outside the fan, one of them west of 99.3% of kept wells. The drop
is also grid-dependent — dx=1 km keeps 136, dx=500 m keeps **134** — so the planned
`--dx 500` grid-convergence check is not like-for-like unless the well sets are intersected
first.

**Gate status: unresolved, not failed.** Re-run only after the folds are grouped by physical
site. Plan C stays blocked either way.

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

**Already discharged:** the AMP channel's own validation is complete (`AMP_V2/`), against
1,736,085 monthly electricity readings from 15,019 pumps. Its numbers are fixed above and
are not to be re-tuned during Plan B.

**Negative results will be reported.** If a full rheology with 40× more sites still fails to
generalize, that is a strong finding against a literature that has assumed sparsity was the
problem. Pre-committing removes the incentive to rationalize.

## 9a. Decomposition into implementation plans

This spec is deliberately larger than one implementation plan. It decomposes into three,
each written and executed separately, with the gates in §7 as the hand-off points:

- **Plan A — Stages 0–2** (data foundation, the `Sk` refit, the differentiable VEP column).
  Self-contained, needs no flow solver, and settles the central scientific question.
- **Plan B — Stage 3 only** (grid + layer geometry, differentiable four-layer flow solver,
  electricity-driven pumping, flow calibration gate). Begins only after Stage 2's gate
  passes — it has. *Revised 2026-08-25: originally scoped as Stages 3–4. Split because the
  Stage-3 gate ("predicts held-out wells better than IDW") is a genuine kill point, and
  because the flow solver is a large enough subsystem to warrant its own review surface.*
- **Plan C — Stage 4** (coupling flow to the VEP column, joint calibration against heads +
  MLCW + leveling). Begins only after Stage 3's gate passes.
- **Plan D — Stage 5 + research-loop hardening** (counterfactual scenarios, ensembles,
  uncertainty calibration).

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
