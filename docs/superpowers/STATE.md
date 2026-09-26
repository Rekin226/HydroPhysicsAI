# Project state: where to continue

**Last updated:** 2026-09-24 · Read §0 first if you are picking the twin up cold; §1-§6
are the history that led there.

The goal, stated once so the gates below have a point:

> **A living 3D digital twin of the Choushui fan that takes a pumping scenario, runs the
> aquifer forward in time, and animates where and how fast the ground sinks -
> re-runnable on demand.**

---

## 0. State on 2026-09-27: the per-well datum model is the deliverable

**The deliverable.** Flow model `results/twin_runs/stage3_datum_gate/`: the physical pump
conversion of the previous model with a learned stress-spread radius, plus merged proximal initial heads
(`--ic-merged-proximal`), SRTM ground elevation (`--ground-elev dem`), the proximal split
at 208 km with a 58 m²/d transmissivity floor, and a **per-well datum** in the observation
operator (`--well-datum fit --well-datum-sd 5`, rms 7.3 m, max 38 m). The datum carries each
well's sub-grid level (single well nests hold 30 m vertical head differences a 1 km cell
cannot); it never enters the solver, the column or the projections. Column
`coupled_leveling/`, projection `results/twin_forward/datum_gate` (36 members, `--apex-hold
calibrated`, restart with the datum subtracted, rheology axis 11 / 30-yr creep ceiling).

| test | previous (10 km, `stage3_spreadL_gate`) | **datum model** | reference |
|---|---|---|---|
| held-out years, fair ratio (≤ 1.25 passes) | 2.12 FAIL | **1.11 PASS** | climatology + trend |
| held-out years, shape R² | −1.03 | **+0.45** | climatology +0.36 |
| unseen wells, head-change (anomaly) R² | −1.43 FAIL | **+0.34** FAIL | IDW +0.61 |
| unseen wells, absolute R² | **+0.80 PASS** | +0.64 FAIL | IDW +0.70 |
| leveling, column out of fold | +0.589 | **+0.652** | |
| leveling, full-chain hindcast | +0.580 | **+0.639** | |
| compaction rings (14, independent) | **+0.295** | +0.009 | |
| policy response | PASS | PASS | |

**Why it was adopted.** The twin turns head *changes* into subsidence under a policy. The
datum model is better on every test of that (held-out years, head changes at unseen wells,
leveling, and the creep uncertainty below) and is the first model to pass the held-out-years
test. It gives up absolute head level at unseen wells, which the column does not use. Two
regressions are stated wherever it is shown: the absolute k-fold now fails, and the 14 rings
lose their skill. At wells it never saw it is still worse than interpolation on both counts:
it is a policy-response tool, not a head interpolator.

The anomaly k-fold verdict (`twin/kfold_scores.py`, pre-registered 2026-09-26) is what
exposed the previous model: it passed the absolute test on levels while its head swings at
unseen wells were several times too large (proximal sd up to 16 m against 1-3 m observed).

| fan-mean forward subsidence 2023-2032 | 11-yr creep ceiling | 30-yr creep ceiling |
|---|---|---|
| baseline | **2.88 ± 0.60 cm** | 3.18 ± 0.66 cm |
| irrigation −30 % from 2026 | 1.93 (avoids 0.95) | 2.25 (avoids 0.93) |
| aquaculture retired from 2026 | 1.57 (avoids 1.31) | 1.88 (avoids 1.30) |

The creep ceiling now moves the baseline by 0.3 cm (it moved the previous model's by 3.4).

**The two models agree on the size of the policy effect by 2032 and disagree on its
mechanism.** In the datum model a pumping cut gives a one-time rebound within months
(irrigation −30 %: 0.93 cm avoided by end 2026, 0.94 by 2032; aquaculture retired: 1.30 /
1.31 cm) and the fan-mean sinking rate over 2028-2032 is 0.30 cm/yr with or without the
policy: the ongoing sinking is creep left over from past drawdown, and a 2030 start gives
the same 2032 number as 2026. In the previous model the effect grew (0.43 / 0.67 / 0.91 cm
for irrigation by 2026 / 2029 / 2032) because the policy slowed the sinking from 0.40 to
0.32 cm/yr. The record has not tested which is right: the leveling network sees the sum,
not the split between elastic rebound and slowed creep. The app states this beside the
avoided-subsidence number (`response_basis_datum`, rebuilt 2026-09-27 on this model;
the fast estimate is within 8.7 % / 3.1 % of the two solved policies).
The previous model's projection under the same apex handling (`physical_spread_apex`): 4.47 /
3.56 / 2.68 cm.

**Built 2026-09-26/27:** `--well-datum` (closed-form per-well offset under a prior; fitted
months only, no leakage), `kfold_scores` (anomaly verdict), `--only-fold` + `merge_folds`
(k-fold as parallel MPS jobs, identical to sequential), datum-aware forward restart and
nudging, `--apex-hold calibrated` (the projection used to re-pin the apex boundary to each
restart field, up to 24 m), `--proximal-layered` and `--ic-layered-proximal` (screened, no
gain). CI: the torch job, red since at least 2026-09-19, is green (plain `pytest` cannot
import `tests.*`).

**The 10 km stress radius was a symptom of the level misfit.** With the datum carrying each
well's level, the full-record fit puts the spread radius at 2.71 km, interior to its bounds,
and the five folds scatter over the whole range (10, 1.94, 0.5, 0.5, 0.5 km). The radius is
not identified once levels are out of the physics' way, so the ensemble spans 0.5-10 km and
the policy effects above already carry that spread. The earlier reading ("the census
locates meters, not wells") is not needed to explain it.

**Still open.** At unseen wells the model is worse than interpolation on head changes
(+0.34 vs +0.61), worst in the proximal fan. The stress radius is not identified. The
2022 recovery is missed by every model. The 182 km column step: see the app's caveat for its
size under this model.

---

## 0b. State on 2026-09-24 (superseded deliverable)

**The deliverable.** Flow model `results/twin_runs/stage3_spreadL_gate/` (physical pump
conversion, 10 km stress spread; head k-fold +0.804 vs IDW +0.702), its per-zone column
`coupled_leveling/`, and the corrected forward run `results/twin_forward/physical_spread_fixed`
(36 members, `--column-hpc0-fast-days 365 --column-heads free --restart-taper-km 5
--save-members yearly`). The decision app `results/twin/twin_app.html` is built from it.

| fan-mean forward subsidence 2023-2032 | 11-yr creep ceiling (deliverable) | 30-yr creep ceiling |
|---|---|---|
| baseline | **4.47 ± 0.77 cm** | 8.18 ± 1.25 cm |
| irrigation −30 % from 2026 | 3.56 (avoids 0.91, p10-p90 0.74-1.04) | 7.17 (avoids 1.01) |
| aquaculture retired from 2026 | 2.68 (avoids 1.79, p10-p90 1.36-2.00) | 6.18 (avoids 2.00) |
| leveling hindcast R² (798 sites) | +0.579 | +0.605 |

All 18 parameter sets agree on the sign of both policy effects. The creep ceiling moves
the baseline by 3.6 cm and the policy effects by under 0.1 cm: decisions are robust to it,
the absolute projection is not (`physical_spread_rheo2_fixed`).

**The earlier 10.68 cm baseline was about half artefact** (2026-09-23): (A1) the proximal
column started 4.7 m under-consolidated with a 24-day time constant and released ~121 cm per
cell in the first months of 2012, invisible to leveling because each site is re-zeroed at its
first survey; (A2) the gain-1 restart at the forecast origin overwrote the proximal deep
layers with IDW heads from distant wells (−21/−25 m). Both are fixed by opt-in forward flags;
leveling skill is unchanged.

**What was closed 2026-09-22 to 09-24** (flags are opt-in; defaults reproduce old results):

| gap | outcome |
|---|---|
| decision app | rebuilt from a visualization literature review (spec `specs/2026-09-23-twin-decision-app-redesign.md`): impact strip first, linked 2D map + time series, difference/swipe/side-by-side, per-run agreement hatching, story mode, township cards, rail profile, caveats beside the numbers they qualify; 3D block is a drawer. 1.8 MB, was 10.7 |
| posterior at bounds | `uncertainty --bounded truncnorm` samples bounded parameters; every earlier posterior had the spread radius unsampled (zero Jacobian column) |
| creep identifiability | forward takes several columns (`--vep-json a,b`) as a rheology axis (table above) |
| 500 m grid | 1 km parameters on the 500 m grid: heads +0.878 vs +0.915, leveling +0.587 vs +0.599: close to converged. A 500 m *refit* landed in a worse basin (+0.810) |
| mid/distal boundary | 172 / 182 / 192 km give in-sample +0.917 / +0.913 / +0.916: insensitive |
| per-purpose efficiency | `--eta-classes` +0.917 vs +0.913 for 6 more parameters: not worth it |
| 182 km subsidence step | caused by the per-zone *column* (swap test 98 %); leveling shows no step. Blended columns fix the step only by going degenerate (rings −0.28 / −0.15): rejected; stated as a caveat |
| constrained column | `--tau-min-days 180 --ske-min 1e-3 --ske-skv-max 0.3`: leveling +0.584, rings +0.309, no degenerate parameter (`coupled_leveling_c180`): equal skill, physically cleaner alternative |
| data | `hydrophysics.twin.data_snapshot` backup with SHA256 manifest outside the repo |
| parallel GPU | 3 jobs under NVIDIA MPS = 2.86x throughput (`results/twin_runs/par*/`) |

**What is still open.**

1. **The held-out-years gate fails for every candidate.** The decomposition
   (`twin/drift_diag.py`) shows 70-92 % of the held-out error is a per-well level offset
   already present in 2012-2019, not drift after 2019. A fair verdict (datum from the fitted
   years only, late-start wells excluded, climatology + trend baseline; pre-registered rule
   datum-RMSE ≤ 1.25x best baseline and shape R² ≥ climatology − 0.05) is written by every new
   run and by `python -m hydrophysics.twin.rescore_temporal`. Screens so far (fair ratio):
   ref10 2.12, slow storage 1.47, canal water 1.83, rivers 1.67, all three 1.58,
   merged proximal IC + DEM ground 1.92, + proximal split + T floor 58: **1.65** (raw RMSE
   6.64 → 5.39 m). Canal water is rejected by the fit (scale → 0.01); slow storage kills the
   policy response.
2. **The proximal zone is the structural problem.** The outlier-well audit found no datum
   errors: proximal layers 3-4 have 1 and 0 wells, so their initial and apex-boundary heads
   came from mid-fan wells (fixed by `--ic-merged-proximal`), 60 wells carry GroundHeight 0.0
   (fixed by `--ground-elev dem`), and single well nests show 30 m vertical gradients that a
   merged proximal aquifer cannot hold. Correcting the proximal initial heads improves heads
   but drops leveling to +0.39 with the candidate's own column: part of the deliverable's
   proximal subsidence skill rested on the wrong initial state. Next: a layered proximal
   aquifer (leakance not pinned), then re-screen. Built 2026-09-25 (opt-in):
   `calibrate_flow --proximal-layered` gives each proximal zone 4 log_T + 4 log_S + 3
   learnable log_L (mid bounds, `--log-t-min-proximal`/`--l-min` respected), starting at
   the merged values so epoch 0 is the merged model; `--ic-layered-proximal` (with
   `--ic-merged-proximal`) keeps per-layer initial heads inside the proximal zone(s), the
   merged value for layers without a well there (proximal L3-L4). Both travel through theta
   meta and the CSVs to forward/policy_gate/uncertainty/surrogate.
   **Screened 2026-09-25, each with its own refit column** (`results/twin_runs/temporal_*layered*`):

   | screen | held-out RMSE | fair ratio | leveling | ds at irrigation x0.7 |
   |---|---|---|---|---|
   | split208 + T floor 58, merged | 5.39 m | 1.65 | +0.391 | −0.91 cm |
   | + `--proximal-layered` | 5.26 | 1.65 | +0.418 | −0.13 |
   | + `--ic-layered-proximal` | 5.20 | 1.67 | +0.389 | −0.51 |
   | `--proximal-layered`, no split | 5.57 | 1.82 | +0.339 | −0.85 |
   | deliverable (merged, old IC) | 6.64 | 2.12 | **+0.589** | −1.25 |

   The layering engaged (proximal leakance left its ceiling in every run) and lowered the
   head error a little, but the fair ratio did not move, leveling stayed near +0.4 and the
   extra freedom absorbed the policy response. **Verdict: no proximal structure tried so far
   beats the deliverable on the chain that matters.** The residual per-well offsets are
   sub-grid (30 m vertical differences within single well nests at 1 km cells); the next
   honest options are a per-well datum term in the observation operator, or treating the
   held-out-years gate as scoring anomalies after that datum, not more aquifer structure.
3. **The 10 km stress radius** remains unexplained (meter vs well location).
4. **The 2022 head recovery** after the drought is missed by every model; rain minus ET0 says
   2022 was dry, so something outside the forcing (canal deliveries resuming, fallowing
   policy) drove it.

---

## 1. Where the chain stands

```
pumping policy ──▶ [flow model] ──▶ heads ──▶ [VEP column] ──▶ subsidence ──▶ [3D viewer]
                   GATE PASS with    nudged      per-zone, vs     +0.526 hindcast   BUILT
                   a PHYSICAL stress to obs      leveling         (798 sites)       (policy axis)
                   (+0.804 vs        at origin   +0.546 o.o.f.
                    IDW +0.702)
```

| stage | status | evidence |
|---|---|---|
| Stage 1, algebraic `Sk` on leveling | marginal | +0.040 tilt-corrected, 878 sites |
| Stage 2, VEP compaction | **PASS** | shared-VEP LOSO +0.465 vs baseline +0.031 |
| Stage 3 primary, clamp released | PASS (2026-09-06) on the closed basin | `log_T` 3/9 ≤ 4/9 |
| Stage 3 secondary, margin vs IDW | **FAIL** (2026-09-09, closed basin, raw census) | flow +0.466 vs IDW +0.702 |
| Stage 3 re-run, TDH fix only | **FAIL** (2026-09-12, `results/twin_stage3_tdh/`) | flow +0.622 vs IDW +0.702, margin −0.080 (was −0.236); eta at floor and head_extra at ceiling in every fold |
| Stage 3 re-run, open basin + clean census | **PASS** (2026-09-14, `results/twin_runs/stage3_open_clean/`, published as `results/twin/stage3_zonal_open_clean.csv`) | in-sample +0.906; **5-fold +0.757 vs IDW +0.702, margin +0.055**; 4 of 5 folds beat IDW (fold 4: +0.507 vs +0.746). But eta floor, head_extra ceiling, recharge 7%, apex and layer-1 coast at the Dirichlet ceiling in every fold, see §2 |
| Stage 4, flow↔VEP coupling | built and exercised | `twin/forward.py` drives the column with the flow model's heads |
| Stage 5, 3D scenario twin | **built** | `explorer3d.py --forward-npz`: policy dropdown, month slider through the projection, verdict in the title |

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
GPU for 18 members × 3 scenarios × 252 months, the "re-runnable on demand" requirement is
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
  **+0.877** against +0.906 free, the price of a physical pumping stress is 0.03 of R².
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
  is 3.4×10⁵ m³, hundreds of metres of drawdown in a confined cell unless storage is
  maximal, which is exactly the fit the optimiser keeps finding (S at 0.3, eta at 0.05).
  The next physics candidates therefore concern *where the stress lands*: pump→layer
  allocation (shallow wells pump layer 1, where S is large), vertical leakage that the
  fit is currently switching off (log_L ≈ −10 to −17), and irrigation return flow.

**If the gate still fails with an open basin and a clean census,** the remaining
suspects are, in order: the river boundaries (Choushui river across the fan; the Wu and
Beigang rivers on the north and south edges are still no-flow), pump-to-layer
allocation (everything pumps layer 2; shallow wells pump layer 1), and irrigation return
flow. Only after those is a different forward model the honest move.

## 3. Remaining gaps, beyond the gate: and what was built for each on 2026-09-14

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
2. **Subsidence skill.** Built: `twin/calibrate_coupled.py`, Stage 4 in practice. The
   column is refit against the MLCW rings with the *flow model's* heads as driver, in
   three configurations (shared; shared with learnable layer weights; one set per fan
   zone), each scored by leave-one-site-out over the rings and by the independent
   leveling network. The leveling winner is a `vep_<config>.json` the forward twin takes.
   **Result on the gated parameters (2026-09-14, `stage3_open_clean/coupled/`):** the
   ring-fitted columns score *worse* on leveling than the Stage-2 column under the same
   driver, shared +0.036, zonal −0.023, weighted −0.827 (it puts 78 % of the weight on
   aquifer 4 and fits the rings best, LOSO +0.164), against +0.299 for Stage-2's
   parameters. Fourteen rings cannot constrain a fan-wide field; the leveling network
   (798 sites) can. So a `--target leveling` mode was added (site-grouped 5-fold
   scoring, rings as the independent check). **Result (2026-09-14,
   `stage3_open_clean/coupled_leveling/`):**

   | column, driver = gated flow heads | leveling out-of-fold R² | bias | rings (independent) |
   |---|---|---|---|
   | Stage-2 shared (fitted on observed heads vs rings) | +0.299 (all sites) | +0.7 cm | n/a |
   | shared, fitted vs leveling | +0.371 | +0.1 cm | +0.015 |
   | **zonal (3 × 4 params), fitted vs leveling** | **+0.546** | +0.1 cm | **+0.371** |

   The per-zone column doubles the fan-wide subsidence skill and still scores +0.371 on
   the 14 rings it never saw. `vep_zonal_leveling.json` is now what the forward twin
   takes (`--vep-json`; per-zone columns are supported since 2026-09-14). Rerun of the
   gated 6-member policy twin with it (`results/twin_forward/open_clean_zonalvep.*`,
   viewer re-rendered): hindcast leveling R² **+0.494**, bias +0.6 cm, RMSE 6.8 cm.
   Projected fan-mean subsidence 2023-2032 is now 10.2 ± 1.9 cm (p95 20 cm) under the
   baseline, 9.4 ± 1.6 cm with aquaculture retired, the Stage-2 column had given 2.7 cm,
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

   **The column belongs to its head field (2026-09-19).** Scored on the 798 leveling
   sites, single member, free-running hindcast:

   | heads | column | leveling R² |
   |---|---|---|
   | free fit | free fit's | +0.552 |
   | physical + spread | its own | **+0.599** |
   | free fit | the physical model's | +0.264 |
   | physical + spread | the free fit's | +0.395 |

   Paired correctly the physical model is the better of the two; crossed, either loses
   about half its skill. The same applies in time: nudging the hindcast toward
   observations every 12 months drops it to +0.391, because the column was calibrated on
   the free-running trajectory. That, not the model, explains the +0.338 the first
   36-member run reported. Ensemble averaging costs almost nothing (+0.572 over six
   members). `--hindcast-gain` now carries the warning.

   **The deliverable, re-run without hindcast nudging (2026-09-19,
   `results/twin_forward/physical_spread_nonudge.*`, viewer
   `results/twin/explorer3d_forward.html`):** 36 members (in-sample + 5 folds + 12
   posterior draws, each from two initial fields), hindcast against 798 leveling sites
   **R² +0.579**, RMSE 6.2 cm, bias -0.8 cm. Fan-mean subsidence 2023-2032: baseline
   10.68 ± 0.57 cm, irrigation -30 % 9.91 ± 0.53, aquaculture retired 9.02 ± 0.50; the
   corresponding layer-2 head changes are +1.09, +1.90 and +2.35 m. This is the twin's
   current state: a gated flow model with a physical pumping stress, a column calibrated
   on its own heads, and a policy response of about 0.8 cm per decade for a 30 % cut in
   irrigation.

   **Spread radius (2026-09-17/18, on top of all three, physical conversion):** 2 km
   +0.895, 4 km +0.908, learned +0.913 with the radius at its 10 km ceiling, the first
   physical-stress configuration to beat the free fit's +0.906 in sample. **Its k-fold
   gate PASSES (2026-09-19, `stage3_spreadL_gate/`, published as
   `results/twin/stage3_zonal_physical_spread.csv`): 5-fold +0.804 vs IDW +0.702, margin
   +0.102, four of five folds ahead**, better than the free fit's +0.757, with eta 0.5,
   40 m extra head, return flow 0.69, 5 % of the stress in layer 1, recharge fraction
   0.16, leakance at the 1e-4 floor in four of six interfaces, and the spread radius on
   its 10 km ceiling in every fold. Spreading the cell-scale stress was the missing
   piece: the same physical conversion that failed at cell scale (+0.632) passes once
   each cell's electricity is applied over a 10 km Gaussian.

   **Re-gated with the bound raised to 25 km (2026-09-20, `stage3_spread25_gate/`,
   published as `results/twin/stage3_zonal_physical_spread.csv`): PASS, 5-fold +0.810 vs
   IDW +0.702, in-sample +0.918, and the radius settles at 21.2 km, interior to the new
   bound in four of five folds.** So the fit does have a preferred stress radius, around
   20 km, not an unbounded appetite for smoothing. This is the twin's model; the free fit
   and the 10 km run are kept as the record of how the pass was reached. A ~20 km radius
   is far larger than a well's cone of depression, so it is standing in for something
   else: most likely that billing coordinates locate the *meter*, not the well, and that
   irrigation districts move water laterally. Worth testing against the well-permit
   coordinates if they can be obtained.

   **And the wider radius is not the better twin (2026-09-20).** Running both through the
   whole chain, the 21 km model wins the head gate by 0.006 and loses everything that
   matters downstream:

   | | 10 km spread | 21 km spread |
   |---|---|---|
   | head k-fold R² | +0.804 | **+0.810** |
   | column out of fold, leveling | **+0.589** | +0.545 |
   | compaction rings, independent | **+0.295** | −0.125 |
   | full-chain hindcast, leveling | **+0.579** | +0.556 |
   | projected baseline subsidence to 2032 | 10.7 cm | 16.4 cm |
   | response to a 30 % irrigation cut | **−0.77 cm** | −0.18 cm |
   | response to retiring aquaculture | **−1.66 cm** | −0.19 cm |

   Spreading the stress over 21 km smooths away the policy signal along with the hot
   spots: that model cannot tell a 30 % irrigation cut from doing nothing. A twin whose
   purpose is policy response fails at its purpose even while passing its gate, so
   **the 10 km model stays the deliverable** and the viewer is rendered from it. The
   lesson is about the gate, not the model: held-out-well R² measures interpolation under
   the recorded forcing and is blind to whether the forcing does any work. Every
   candidate from here should be scored on the policy response and on the leveling chain,
   not on the head gate alone.

   **Temporal gate (held-out years, 2026-09-18): a hard result.** Fit on 2012-2019,
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
   temporal gate says not to trust it. The continuation error over those 36 months, in
   metres at the 158 wells (`results/twin_runs/temporal_predictions.npz`):

   | | bias | RMSE | median abs. error |
   |---|---|---|---|
   | climatology | +1.14 m | 2.01 m | 1.18 m |
   | persistence | +1.90 m | 3.16 m | 1.49 m |
   | free fit, free-running from 2012 | +0.90 m | 6.58 m | 2.72 m |
   | free fit, restarted from the observed field at the origin | +0.35 m | 8.08 m | 2.26 m |
   | physical + split/floor/return, free-running | +0.32 m | 7.31 m | 4.15 m |
   | physical + learned spread, free-running | +0.11 m | 6.29 m | 3.97 m |

   Biases are small; the error is spread, three times climatology's, and restarting from
   the observed field does not help the RMSE (the injected state is not the model's own
   and it relaxes). Two readings, both testable next: the free fit's recharge fraction of
   0.07 makes it nearly blind to rainfall, so it cannot follow the 2020-21 drought; and
   the calibration objective is a level misfit, which the between-well variance
   dominates, so the fit is never asked to get anomalies right. A per-well anomaly term
   in the loss (or fitting anomalies outright) and a held-out-years gate as a first-class
   verdict are the next calibration changes, ahead of any more stress placement.

   **The level anchor sweep (2026-09-20) closes the anomaly-loss experiment.** At weight
   1 and 3 the in-sample fit and the level error come back (+0.916 and +0.925, 3.5 and
   3.1 m) but the shape degrades to −1.29 and −2.15, worse than the level loss's −1.05.
   Only the unanchored weight of 0.1 bought shape (+0.059), and it cost 6.4 m of level.
   There is no setting that wins both: the anomaly loss trades one for the other rather
   than fixing the drift. The held-out-years failure is therefore structural, not a loss
   function choice, and the next candidates are physical: a storage term that can release
   water slowly (the fit pins mid-zone S at 0.3), and forcing the model has never seen
   (surface-water irrigation deliveries, which would explain both the drift and the 21 km
   radius).

   **The anomaly loss, tested 2026-09-19 (`--loss anomaly`, level weight 0.1).** On the
   gate's own metric it looks worse (anomaly R² −20 free, −31 physical, against −8 and
   −7 for the level loss). That metric removes the *observed* fitted-period mean from
   both series, so it charges a model for a level offset as well as for a wrong shape.
   Scoring the shape alone, each series minus its own fitted mean, reverses the reading:

   | fit | shape R² of the continuation | mean level error over the fitted years |
   |---|---|---|
   | climatology | +0.154 | n/a |
   | free, level loss | −0.405 | 3.33 m |
   | free, anomaly loss | −0.037 | 5.58 m |
   | physical + spread, level loss | −1.052 | 3.73 m |
   | physical + spread, anomaly loss | **+0.059** | 6.39 m |

   So the anomaly loss does what it was meant to: it nearly closes the shape gap to
   climatology, on the model that carries a physical pumping stress. It pays for that by
   letting the absolute level drift, because at a level weight of 0.1 the anchor is too
   weak. The next runs restore it (`--level-weight 1` and `3`, queued), and the temporal
   gate now reports shape R² and the level error beside the anomaly score, since the
   pooled and anomaly numbers on their own hid this. The next honest steps are a coarser
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
535.230.02, Quadro RTX 6000 (Turing sm_75, no bf16, no FP8, no FA2). conda env `hydro`.
The twin's solver is float64 and uses no AMP; the FNO surrogate is fp32.
