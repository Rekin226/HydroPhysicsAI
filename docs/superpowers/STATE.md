# Project state — where to continue

**Last updated:** 2026-09-18 · Read this first if you are picking the twin up cold.

The goal, stated once so the gates below have a point:

> **A living 3D digital twin of the Choushui fan that takes a pumping scenario, runs the
> aquifer forward in time, and animates where and how fast the ground sinks —
> re-runnable on demand.**

As of 2026-09-14 every link of that chain is built, tested, and the flow model **passes
its k-fold gate** (+0.757 vs IDW +0.702). What is still open is whether its *pumping
response* is physical: the fit still pins the pump conversion at its bounds, so policy
deltas are consequences of a gated model, not yet validated sensitivities. §2 says how
that is being measured.

---

## 1. Where the chain stands

```
pumping policy ──▶ [flow model] ──▶ heads ──▶ [VEP column] ──▶ subsidence ──▶ [3D viewer]
                   GATE PASS         nudged      PASSES           PASSES          BUILT
                   (+0.757 vs        to obs      (Stage 2)        (R² +0.299      (forward mode:
                    IDW +0.702)      at origin                     vs leveling)    policy axis)
```

| stage | status | evidence |
|---|---|---|
| Stage 1 — algebraic `Sk` on leveling | marginal | +0.040 tilt-corrected, 878 sites |
| Stage 2 — VEP compaction | **PASS** | shared-VEP LOSO +0.465 vs baseline +0.031 |
| Stage 3 primary — clamp released | PASS (2026-09-06) on the closed basin | `log_T` 3/9 ≤ 4/9 |
| Stage 3 secondary — margin vs IDW | **FAIL** (2026-09-09, closed basin, raw census) | flow +0.466 vs IDW +0.702 |
| Stage 3 re-run, TDH fix only | **FAIL** (2026-09-12, `results/twin_stage3_tdh/`) | flow +0.622 vs IDW +0.702, margin −0.080 (was −0.236); eta at floor and head_extra at ceiling in every fold |
| Stage 3 re-run, open basin + clean census | **PASS** (2026-09-14, `results/twin_runs/stage3_open_clean/`, published as `results/twin/stage3_zonal_open_clean.csv`) | in-sample +0.906; **5-fold +0.757 vs IDW +0.702, margin +0.055**; 4 of 5 folds beat IDW (fold 4: +0.507 vs +0.746). But eta floor, head_extra ceiling, recharge 7%, apex and layer-1 coast at the Dirichlet ceiling in every fold — see §2 |
| Stage 4 — flow↔VEP coupling | built and exercised | `twin/forward.py` drives the column with the flow model's heads |
| Stage 5 — 3D scenario twin | **built** | `explorer3d.py --forward-npz`: policy dropdown, month slider through the projection, verdict in the title |

Canonical diagnosis: `specs/2026-08-29-choushui-stage3-zonal-remediation-design.md` §7.5.

## 2. The one thing to do next

**Establish whether the gated model's pumping response is physical.**

The gate passed on 2026-09-14 (32 parameters, 500 epochs, 5 site-grouped folds, 35 h on
the GPU). The policy twin, the viewer and the surrogate are being run on its parameters
by the post-gate chain (`results/twin_runs/post_gate_chain.log`). The open question is
below, and the chain's third step measures it.

Two root causes were found on 2026-09-11 (spec §7.5), and both are now calibration
defaults:

1. **The solver was a closed basin.** No coastal outlet, no apex inflow. Now general-head
   boundaries on the coast (h_b = 0) and the apex (h_b = initial head) with learnable
   conductances (`--boundaries coast-apex`, `twin/boundaries.py`).
2. **The census over-counted electricity 4.6×.** Shared meters attached to every pump
   on them, and meters drawing many times their rated motor capacity. Now
   `pumping.clean_census` (`--meter-filter dedupe-cap`): 9.9 → 2.1 TWh.

The command that produced the PASS (defaults now open the basin and clean the census):

```bash
export HYDROMIND_GW_DATA=$PWD/chou-shui-data/data
tmux new -s gate
python -m hydrophysics.twin.calibrate_flow --param-mode zonal --epochs 500 --n-folds 5 \
  --seed 0 --device cuda --compile-matvec --log-every 25 --dump-predictions \
  --out results/twin_runs/stage3_open_clean          # 35 h on the GPU (6 h fit + 5 folds)
```

Defaults now point at the repo data paths, open the basin and clean the census, and the
run writes `stage3_theta.json`, `stage3_fold_thetas.json`, `stage3_wells.csv` and
`stage3_flow.csv` (with `boundaries` and `meter_filter` columns) into `--out`.

Then, whatever the verdict:

```bash
python -m hydrophysics.twin.forward \
  --theta results/twin_runs/stage3_open_clean/stage3_theta.json \
  --theta results/twin_runs/stage3_open_clean/stage3_fold_thetas.json \
  --scenario "cut30:irrigation=0.7@2026-01" --scenario "retire_aqua:aquaculture=0" \
  --horizon 120 --ic-members 2 --out results/twin_forward/open_clean
python -m hydrophysics.twin.explorer3d --forward-npz results/twin_forward/open_clean.npz \
  --stride 3 --out results/twin/explorer3d_forward.html
```

The forward run prints the gate verdict it inherits and its hindcast skill against
leveling. Measured 2026-09-12 with the same shared VEP column, 798 leveling sites:

| head driver for the column | leveling R² | bias |
|---|---|---|
| observed heads, IDW (explorer3d head-decline mode) | +0.242 | +3.1 cm |
| flow model, 2026-09-09 gate parameters (closed, raw census) | −0.036 | +2.3 cm |
| flow model, TDH-gate parameters (closed, raw census, hindcast R² +0.873) | **+0.299** | −0.3 cm |

| flow model, **gated** parameters (open basin, clean census), 18-member ensemble | **+0.299** | +0.7 cm |

The coupled chain beats the observed-head-driven column on the independent leveling
network. The gated run (`results/twin_forward/open_clean.*`, viewer
`results/twin/explorer3d_forward.html`, 43 MB, 86 frames, 3 policies) took 609 s on the
GPU for 18 members × 3 scenarios × 252 months — the "re-runnable on demand" requirement is
met by the solver itself. Fan-mean projection to 2032, ± = spread over the 6 parameter
sets × 3 initial fields:

| policy from 2026-01 | layer-2 head change | forward subsidence (p95) |
|---|---|---|
| baseline | +0.74 ± 0.27 m | 2.72 ± 0.16 cm (5.4 cm) |
| irrigation −30 % | +1.02 ± 0.34 m | 2.64 ± 0.13 cm (5.3 cm) |
| retire aquaculture | +1.71 ± 0.56 m | 2.46 ± 0.06 cm (4.9 cm) |

Read these with the caveat below: the policy *deltas* are small because the gated model
converts electricity at its efficiency floor. The same three policies through the
physical-conversion fit (`results/twin_forward/fixed_eta.*`, one member, not yet gated):

| policy from 2026-01 | layer-2 head change | forward subsidence (p95) |
|---|---|---|
| baseline | +1.01 m | 3.76 cm (7.3 cm) |
| irrigation −30 % | +1.84 m | 3.50 cm (6.7 cm) |
| retire aquaculture | +2.49 m | 3.22 cm (6.0 cm) |

Three times the head response per policy, and a 7 % subsidence reduction for a 30 %
irrigation cut instead of 3 %. Its hindcast leveling skill is +0.274 (bias +2.3 cm).
Which of the two tables is evidence is what the running fixed-eta gate decides.

**The caveat that governs the policy numbers.** Fit +0.906
(+0.868 TDH, +0.758 original). Yet `eta` = 0.05 (floor), `head_extra` = 200 m (ceiling),
`recharge_frac` = 0.07, `C_apex` and layer-1 `C_coast` = 1e5 m²/day (Dirichlet ceiling),
layers 2-4 coast conductance ≈ 0.2-1.4 (closed), proximal `log_T` = 10 m²/day (lower
clamp, in the gravel fan), mid `S` = 0.3 (ceiling) in two layers. The implied abstraction
at those values is ~0.02 ×10⁹ m³/yr, two orders below the published 1.5-2.0. Opening the
basin and cleaning the census raised the fit and did not change the verdict of the
optimiser: it still wants the pumping stress gone and the storage maximal. The cheapest
next diagnostics, both runnable on CPU while the folds finish:

- `calibrate_flow --fix-eta 0.5 --fix-head-extra 40`, **done 2026-09-14**
  (`results/twin_runs/stage3_fixed_eta/`, fit-only, 500 epochs, 4.1 h GPU): in-sample
  **+0.877** against +0.906 free — the price of a physical pumping stress is 0.03 of R².
  And the rest of the model becomes physical with it: recharge fraction **0.43** (was
  0.07), proximal T 57 m²/day (off the clamp, at Liu et al.'s 58 floor), coast conductance
  14-28 m²/day in layers 2-4 (open, was closed), distal T high. Mid-zone S still sits at
  0.3 in three layers and the apex stays Dirichlet. Implied abstraction ≈ 0.75 ×10⁹
  m³/yr, within a factor two of the published range. Its k-fold gate
  (`results/twin_runs/stage3_fixed_eta_gate/`, 2026-09-15): **FAIL, +0.626 vs IDW +0.702**
  (margin −0.076). So a physical pumping stress under the *current* placement of that
  stress (all of it in layer 2, leakage off) generalises worse than the free fit that
  switches the stress off. The verdict is stamped in its theta file. Its policy run
  (`results/twin_forward/fixed_eta.*`) shows the sensitivities such a model has, but that
  model is not gated. What is being measured next is whether moving the stress
  (`--pump-split`, `--l-min`, `--return-flow`) lets a physical conversion pass.
- a data-only check, done 2026-09-13 (`results/twin_runs/diag_head_vs_pumping.csv`, 147
  wells): observed seasonal head amplitude **rises** with cleaned kWh within 3 km
  (Spearman +0.51 overall, +0.57 in aquifer 2, +0.69 in aquifer 3) and the 2012-2022
  trend falls with it (−0.28). The heads carry the pumping signal. What they cannot carry
  is its *concentration*: cleaned kWh per 1 km cell has median 1,170 but p90 255,000 and
  max 1,003,000 kWh/yr, so at a physical conversion the 99th-percentile cell's peak month
  is 3.4×10⁵ m³ — hundreds of metres of drawdown in a confined cell unless storage is
  maximal, which is exactly the fit the optimiser keeps finding (S at 0.3, eta at 0.05).
  The next physics candidates therefore concern *where the stress lands*: pump→layer
  allocation (shallow wells pump layer 1, where S is large), vertical leakage that the
  fit is currently switching off (log_L ≈ −10 to −17), and irrigation return flow.

**If the gate still fails with an open basin and a clean census,** the remaining
suspects are, in order: the river boundaries (Choushui river across the fan; the Wu and
Beigang rivers on the north and south edges are still no-flow), pump-to-layer
allocation (everything pumps layer 2; shallow wells pump layer 1), and irrigation return
flow. Only after those is a different forward model the honest move.

## 3. Remaining gaps, beyond the gate — and what was built for each on 2026-09-14

1. **Uncertainty.** Built: `twin/uncertainty.py`, a Laplace posterior around the
   calibrated vector (Jacobian by central finite differences through the rebuilt model,
   residual variance, weak log-prior; parameters at a bound are reported and held, not
   sampled). Its samples are `--theta` members for the forward twin. Built:
   sequential assimilation through the record (`twin.forward --hindcast-gain g
   --hindcast-every k`), nudging the state toward the observed field every k months.
   Still true: a Laplace posterior is local and linear; the truncated posteriors of
   pinned parameters are not represented. **Run on the gated model (2026-09-15,
   `stage3_open_clean/stage3_posterior.json`):** 32 free parameters, 10 at a bound and
   held; residual sd 6.6 m; posterior sd 0.45-0.7 in log-T/log-S for the mid zone,
   1.3-2.0 for the distal deep layers and the coast conductances (weakly identified).
   Policy run with 18 members (in-sample + 5 folds + 12 posterior draws), sequential
   nudging (gain 0.5 every 12 months) and the per-zone leveling column
   (`results/twin_forward/open_clean_posterior.*`, viewer re-rendered): hindcast
   leveling R² **+0.526**, bias 0.0 cm; 2023-2032 fan-mean subsidence 10.0 ± 1.3 cm
   baseline, 9.5 ± 1.1 cm with aquaculture retired. This is the twin's current
   deliverable.
2. **Subsidence skill.** Built: `twin/calibrate_coupled.py` — Stage 4 in practice. The
   column is refit against the MLCW rings with the *flow model's* heads as driver, in
   three configurations (shared; shared with learnable layer weights; one set per fan
   zone), each scored by leave-one-site-out over the rings and by the independent
   leveling network. The leveling winner is a `vep_<config>.json` the forward twin takes.
   **Result on the gated parameters (2026-09-14, `stage3_open_clean/coupled/`):** the
   ring-fitted columns score *worse* on leveling than the Stage-2 column under the same
   driver — shared +0.036, zonal −0.023, weighted −0.827 (it puts 78 % of the weight on
   aquifer 4 and fits the rings best, LOSO +0.164) — against +0.299 for Stage-2's
   parameters. Fourteen rings cannot constrain a fan-wide field; the leveling network
   (798 sites) can. So a `--target leveling` mode was added (site-grouped 5-fold
   scoring, rings as the independent check). **Result (2026-09-14,
   `stage3_open_clean/coupled_leveling/`):**

   | column, driver = gated flow heads | leveling out-of-fold R² | bias | rings (independent) |
   |---|---|---|---|
   | Stage-2 shared (fitted on observed heads vs rings) | +0.299 (all sites) | +0.7 cm | — |
   | shared, fitted vs leveling | +0.371 | +0.1 cm | +0.015 |
   | **zonal (3 × 4 params), fitted vs leveling** | **+0.546** | +0.1 cm | **+0.371** |

   The per-zone column doubles the fan-wide subsidence skill and still scores +0.371 on
   the 14 rings it never saw. `vep_zonal_leveling.json` is now what the forward twin
   takes (`--vep-json`; per-zone columns are supported since 2026-09-14). Rerun of the
   gated 6-member policy twin with it (`results/twin_forward/open_clean_zonalvep.*`,
   viewer re-rendered): hindcast leveling R² **+0.494**, bias +0.6 cm, RMSE 6.8 cm.
   Projected fan-mean subsidence 2023-2032 is now 10.2 ± 1.9 cm (p95 20 cm) under the
   baseline, 9.4 ± 1.6 cm with aquaculture retired — the Stage-2 column had given 2.7 cm,
   which was the wrong rheology for a fan-wide field. Caveat: the mid-zone viscous time
   constant sits at its ceiling (3,960 d, the record length × dt), so decadal creep is
   bounded by the calibration window. Revisited 2026-09-17 with the ceiling lifted to 30
   years (`--tau-max-years 30`, `coupled_leveling_tau30/`): out-of-fold +0.554 against
   +0.546, and the mid and distal time constants go straight to the new ceiling (30 and
   29 years) with the inelastic coefficient doubling (0.18 → 0.40) to compensate. The
   pair is not identifiable from an 11-year record: a longer tau with a larger Skv gives
   the same creep inside the window and more of it afterwards. Decadal projections
   therefore carry a rheology uncertainty the ensemble does not show; the ceiling is a
   modelling choice and is recorded with each column file (`tau_max_years`).
3. **Where the stress lands.** Built as opt-in calibration options: `--pump-split`
   (learned share of abstraction from layer 1), `--return-flow` (learned irrigation
   return fraction ≤ 0.7 into layer 1), `--l-min` (leakance floor). Queued on the GPU
   after the fixed-eta gate (`results/twin_runs/post_gate2_chain.log`): four fit-only
   diagnostics at the physical conversion (split, floor, return, all three), the best
   one gated, then posterior → column refit → policy runs with assimilation → viewer →
   surrogate on the winner. **Diagnostics so far (2026-09-15, fit-only, 300 epochs,
   eta 0.5 / head_extra 40 m; reference without them +0.877):**

   | option | in-sample R² | what it learned |
   |---|---|---|
   | `--pump-split` | +0.867 | 75 % of abstraction from layer 1; recharge fraction 0.85 |
   | `--l-min 1e-4` | +0.876 | leakance at the new floor in two mid interfaces |
   | `--return-flow` | **+0.884** | return fraction 0.69 (near its 0.7 cap); recharge fraction 0.21 |
   | all three | +0.884 (300 epochs), +0.887 at 500 | gated: **FAIL, 5-fold +0.632 vs IDW +0.702** (2026-09-17, `stage3_all_gate/`) |

   So a physical pumping conversion fits nearly as well as the free fit in sample
   (+0.887 vs +0.906) under every stress placement tried, and **fails the held-out-well
   gate under every one of them** (+0.626 fixed, +0.632 with split + floor + return).
   The free fit that switches the stress off remains the only configuration that
   generalises across wells. The reading is that cell-scale pumping hot spots do not
   transfer to wells the fit never saw, so a model that damps them predicts held-out
   heads better than one that carries them. This is where the twin stands: heads and
   subsidence gated, pumping sensitivity not. For the record, the stress-placement
   model's own chain (`stage3_all_gate/coupled_leveling/`, `results/twin_forward/
   final_zonalvep.*`, viewer `results/twin_forward/all_gate_viewer.html`): per-zone
   column +0.570 out of fold on leveling, full-chain hindcast +0.481 with 26 members,
   and a policy response about three times the gated model's (irrigation −30 %:
   subsidence 5.72 → 5.29 cm over the decade, layer-2 head +0.6 m). Those are the
   sensitivities a physical stress gives; that model does not pass its gate.
   `results/twin/explorer3d_forward.html` is the gated model's run.

   **Spread radius (2026-09-17/18, on top of all three, physical conversion):** 2 km
   +0.895, 4 km +0.908, learned +0.913 with the radius at its 10 km ceiling — the first
   physical-stress configuration to beat the free fit's +0.906 in sample. Its k-fold gate
   (`stage3_spreadL_gate/`) is running.

   **Temporal gate (held-out years, 2026-09-18) — a hard result.** Fit on 2012-2019,
   free-running continuation over 2020-2022 (36 months) from the record's start:

   | configuration | pooled R² | anomaly R² (per-well departures from the fitted mean) | per-well median R² |
   |---|---|---|---|
   | climatology of the fitted years | +0.991 | +0.154 | −0.38 |
   | persistence of the last fitted month | +0.978 | −1.10 | −1.84 |
   | free fit (gated) | +0.907 | **−8.1** | −3.9 |
   | physical + split/floor/return | +0.885 | −10.2 | −10.2 |
   | physical + learned spread | +0.915 | −7.3 | −8.5 |

   Pooled R² is a between-well statistic and says nothing here. On anomalies every
   configuration is far worse than climatology: a free-running continuation of this
   model **drifts within three years**, over a window that contains the 2020-21
   drought. The held-out-*well* gate measures spatial interpolation under the recorded
   forcing; it does not measure whether the model's own dynamics carry the heads
   forward, and they do not yet. This is now the twin's governing caveat, ahead of the
   pumping sensitivity. For the deliverable: projections are anchored by nudging to the
   observed field at the origin (and every 12 months through the record), so the first
   years are held by the data, but the decade-scale trend is the model's and the
   temporal gate says not to trust it. The continuation error in metres, free-running
   and restarted from observations at the origin, is in
   `results/twin_runs/temporal_predictions.npz`. The next honest steps are a coarser
   placement of the stress (smoothing the census over its billing radius, or a learned
   spread kernel) and a gate that scores the *response to pumping* directly, e.g.
   held-out years rather than held-out wells.
4. **Zone boundary and grid convergence.** The mid/distal boundary at 182 km has no
   independent justification (spec §10). The `--dx 500` check can now be made
   like-for-like with `--wells-from <1 km run>/stage3_wells.csv`, but has not been run.
5. **The surrogate is trained on the gated parameters** (`results/surrogate/open_clean_*`,
   2026-09-14, GPU): 256 solver rollouts × 24 months under random policies (6,144
   one-step pairs), 200 epochs, one-step validation rel-L2 **0.0145**; a 24-month
   autoregressive rollout stays within **0.03-0.04 m RMSE** of the solver in every layer,
   at 13-34× the solver's GPU speed (0.08-0.2 s vs 2.7 s per 24-month scenario). It is
   PhysicsNeMo's one job in the twin: sweeps and large ensembles. It inherits the solver's
   physics, so it must be retrained whenever the gated parameters change (the fixed-eta
   model, if it passes).
6. **Per-purpose efficiency** (`--eta-classes`) is implemented and tested but not yet run
   through a gate; the clean census makes irrigation 60% of the energy, so the class
   split matters less than it did.

## 4. Traps that have already cost time

- **Device placement.** Every flow run before 2026-09-07 silently ran on CPU. GPU is
  ~51× faster on the real problem (67 s/epoch vs 3,452). `tests/test_twin_device.py`
  guards it. Always confirm the `device:` line at startup.
- **Synthetic benchmarks mispredicted the real system by two orders of magnitude.** Do not
  size this solver's cost from synthetic grids. A real 132-month rollout is ~210 s on CPU.
- **Long runs need `--log-every`.** `ptrace_scope=1` means py-spy needs sudo after the fact.
- **The WiseEnvr bearer token expires**; `twin/fetch_amp.py` renews it and records
  successes, not attempts.
- **`results/twin/*.csv` are the published results**; `calibrate_flow` now defaults to
  `results/twin_runs/stage3_<UTC stamp>/` so a new run never overwrites them.
  `results/twin_stage3_*/` stay ignored.
- **Old CLI defaults were dead** (doubled `chou-shui-data/chou-shui-data/`, a scratchpad
  path for the census). `DEFAULT_PATHS` in `calibrate_flow.py` is the single table now.
- **Two runs on the GPU at once** roughly double both wall-clocks; the solver is
  launch-bound. Queue gates, do not overlap them.

## 5. Reproducing the recorded Stage-3 runs

```bash
export HYDROMIND_GW_DATA=$PWD/chou-shui-data/data
# the 2026-09-09 verdict (closed basin, raw census, 500 epochs, 5 folds, ~25 h on GPU)
python -m hydrophysics.twin.calibrate_flow --param-mode zonal --epochs 500 --n-folds 5 \
  --seed 0 --device cuda --compile-matvec --log-every 100 --dump-predictions \
  --boundaries none --meter-filter none --out results/twin_stage3_folds_repro
```

Note the epochs: the recorded verdict used **500**, not the 1500 an earlier version of
this file listed. Data that must exist: see `docs/DATA_FORMAT.md`, "The twin's data
cache" (~331 MB + ~1 GB, rebuilt with `twin/fetch_amp.py` except the fan polygon).

## 6. Environment

torch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, PhysicsNeMo 2.2.1, warp-lang 1.17.0, driver
535.230.02, Quadro RTX 6000 (Turing sm_75 — no bf16, no FP8, no FA2). conda env `hydro`.
The twin's solver is float64 and uses no AMP; the FNO surrogate is fp32.
