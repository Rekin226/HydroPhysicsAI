# Model Card: HydroPhysicsAI

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
| **SpatialPINN** | Physics-informed neural network learning a continuous head field `h(x, y, t)` over the fan, regularized by the 2D depth-averaged groundwater-flow PDE (autodiff residual) with learned `T(x,y)/α(x,y)/d(x,y)` fields. **Documented negative result**: not competitive with the per-well models on this data (see Limitations); retained as a continuous-field baseline and for the head-field map. | Simulation |

## Intended use

- Research and benchmarking of physics-ML vs classical gray-box ODEs for groundwater.
- **Simulation**: free-running hindcast of levels from an initial condition + forcing,
  for scenario analysis where no live observations are available.
- **Forecast**: research on short-horizon (1–30 day) forecasts and uncertainty.
  Operational skill and calibrated coverage require validation after the October
  2026 forecast-window corrections below.

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

**Forecast results below are historical and require retraining and reevaluation.** The October 2026
audit reproduced training targets crossing the held-out cutoff, inference depending
on future target availability, and point-score comparisons on different rows. Those
paths have been repaired, but the earlier forecast scores/coverage are not evidence
for the corrected implementation. Perfect future forcing is also a conditional
experiment, not a forecast of rainfall or upstream heads available at issuance.
These findings concern the forecast LSTM/evaluator, not a new re-evaluation of the
separate simulation results.

| Task | Result | Reference |
|---|---|---|
| Simulation (free-running) | PhysicsUDE / PhysicsNeMoUDE **KGE 0.591**, NSE 0.49 | gray-box 0.736 · climatology 0.446 |
| Leave-one-well-out (static attrs) | KGE 0.236 | climatology 0.446 |
| Leave-one-well-out (+ observable signatures) | KGE 0.389 | climatology 0.446 |
| Leave-one-well-out (+ equilibrium anchoring, operator alone) | KGE 0.491 | climatology 0.446 |
| Leave-one-well-out (**+ self-consistency gate**) | **KGE 0.565** | climatology 0.446 · in-sample 0.591 |
| Forecast 7-day | LSTM **KGE 0.965** | persistence 0.946 |
| Forecast 30-day | LSTM **KGE 0.899** | persistence 0.703 |
| Forecast probabilistic | CRPS 0.143 / 0.286 (7/30 d), PICP ≈ 0.90 | persistence-Gaussian 0.260 / 0.590 |

GPU (earlier phase, historical): on an RTX 4070 SUPER, not the project's server, the forecaster trained 14× faster than CPU under bf16-AMP at
~half the memory (`python -m hydrophysics.bench`).

## Limitations

- **Trails the classical baseline in simulation:** 0.591 vs the per-well gray-box 0.736.
  The operator wins on 18/61 wells; a few hard wells dominate the gap.
- **Generalization to unseen wells: improved, with caveats.** Leave-one-well-out median
  KGE went 0.236 (geographic attrs) → 0.389 (observable history signatures) → 0.491
  (pinning each well's free-run equilibrium to its observed mean, fixes a 5–15 m level
  drift) → **0.565** (a self-consistency gate that falls back to climatology where the
  operator can't reproduce a well's own history). The operator alone now beats climatology
  (0.491 vs 0.446), and the held-out median (0.565) nearly matches the in-sample operator
  (0.591); averaging a K=3 ensemble lifts it to ≈0.59, matching in-sample. **But** it is
  not a clean sweep: the hybrid is *worse* than climatology on ~8 wells, and these resist
  fixing, an ensemble-disagreement (uncertainty) gate caught only 2, because the rest are
  **non-stationary** (their 2019+ departs from training; climatology fails them too), which
  no training-time signal can detect. KGE's unbounded tail means median/clipped-mean are
  the trustworthy aggregates (raw mean is not). All choices were made on the inner 2018
  split; 2019+ scored once.
- **Forecast skill depends on assimilation:** the LSTM uses observed levels up to the
  origin; it is not a free-running model and its scores must not be compared to
  simulation mode.
- **SpatialPINN is a negative result, not a contender.** The continuous-field PINN reaches
  in-sample KGE 0.334 and leave-one-well-out 0.105, below climatology (0.446), the UDE
  (0.591/LOWO 0.565), and the gray-box (0.736). A field conditioned only on `(x, y)` cannot
  match per-well dynamics, and on this fan coordinates alone barely place an unseen well; the
  UDE wins by conditioning on per-well observable history and anchoring to the observed mean.
  Its head-field *map* is physically plausible and is the retained deliverable. A competitive
  field model would be the UDE plus a spatial deviation field (future work). `physics_weight`
  was selected on the inner pre-2019 split; 2019+ scored once.
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

## October 2026 evidence update

Interval coverage, the registered prospective tests (the physics and the statistical layer), the 2023-25 ring/GNSS continuation, the negative aquitard-column result, the statistical layer's 2023-26 score and the frozen column's score against the deep benchmarks printed in the WRA monthly reports (January 2024 to January 2026, transcribed with the Claude API and verified; `results/twin/deep_pile_holdout/`; report-only and selection-contaminated since 2026-10-21, ruling U7, because these scores were used to choose between columns) are recorded in [docs/TWIN_EVIDENCE_2026-10.md](docs/TWIN_EVIDENCE_2026-10.md); the page's evidence panel reads those report files directly.

### Groundwater-subsidence twin: status after Phase 1 (2026-10-23)

The page model is `datum_gate_innov` (theta `stage3_datum_gate`), N2's registered primary model; R0 (`stage3_datum_gate_cwa`) and Q0 are ladder baselines, and the N1 verdicts reach the page model only through a post-hoc, model-only comparison. The text below keeps the Phase 1 record.

The deliverable is the R0 twin, `stage3_datum_gate_cwa`: a 4-layer differentiable flow
model with a leveling-calibrated VEP column. No Phase 1 candidate beat it under its
declared gates.

Two models carry the deliverable's recipe. In the I1 and N1 tests below, R0 is
`stage3_datum_gate_cwa` (CWA-rebuilt rainfall; recorded commit `9715f58`; the code was at or
after `13860ee`, see `docs/TWIN_THETA_PROVENANCE_2026-10.md`); the N1 tests for Changhua
use the census model R1c instead. The decision page's
levers and projections come from the registered run `datum_gate_innov` (theta
`stage3_datum_gate`, vendor rainfall, commit `fa5cca2`). Both prospective registrations
are bound to that run by hash, so the page was not rebuilt on R0. Evidence section 11 found
the two within noise on every test it ran.

Head skill:
- **Fitted period.** Out-of-fold k-fold R² is 0.643 on the design folds and 0.718 on a fresh partition. It sits below inverse-distance interpolation on anomalies.
- **Masked test (`N1.G1_mask_E2015`).** Refitted without the Jan-May 2015 targets, it reproduces the 2015 drought-spring anomaly better than climatology (RMSE 1.49 against 2.06 m, ratio 0.72) and better than its own zero-response counterfactual (1.75 m), at 135 wells.
- **Damped amplitude.** The modelled head response is smaller than observed; by how much depends on the test. The observed-on-modelled slope is 2.13 (90 % interval 1.93-2.39) in the 2015 drought spring (G1). That event mixes canal water and rain with pumping, so 2.13 is not a correction factor. On the masked 2016-18 refits (step C, design-time) it is 1.21 (1.06-1.36) in Yunlin and 0.85 (0.63-1.05) in Changhua. The Changhua value was scored on the census model R1c, and its interval includes 1.
- **Policy response, heads (`N1.policy_response.v1`, step C).** Masked 2016-18 refits beat both no change and reference-year pumping in Yunlin (R0; 75 wells, RMSE 1.38 against 2.21 and 2.29 m) and in Changhua (R1c census model, because R0 carries no Changhua pumping; 59 wells, 1.07 against 1.19 and 1.21 m). The verdict is supported in Yunlin on R0, on a design-time window. In Changhua it is supported only on R1c, which failed its own rung (I1.R1c) and is not the page model; the page has no Changhua head lever.

Reported by county, because they differ:

| | Changhua | Yunlin |
|---|---|---|
| Masked 2015 test, RMSE ratio to climatology | 0.89 (59 wells) | 0.68 (76 wells) |
| Median correlation, fresh folds | 0.42 | 0.74 |
| County-mean anomaly RMSE, fresh folds | 0.52 m | 1.08 m |

Changhua's year-to-year head changes are not reproduced by any forcing available. The
suspected missing source is canal water outside the Jiji supply, for which no data are
available.

Subsidence:
- **Fitted period.** The column fits the 2012-2022 leveling with out-of-fold R² 0.65.
- **Not claimed as skill:**
  - its 2023-26 deep-pile score was used to choose between columns, so it is not reported as skill;
  - the registered spring-compaction test (G4) is not evaluable with the printed record.
- **Policy response, 1-3-year size: not supported** (`N1.policy_response.v1`, steps B and C).
  - On 2016-18, a constant sinking rate partitions the May 2015-19 leveling better than the column (observed-on-modelled slope 0.22 in Yunlin, 0.40 in Changhua).
  - Across places in 2007-10, Yunlin sank more than modelled (observed +0.66 against +0.26 cm/yr per GWh/yr, inconclusive).
  - The two tests disagree in direction, so the label is "not supported", with no direction attached.
  - In this model the avoided subsidence is an elastic response that arrives within about a year; its 1-3-year size is not supported by the registered tests, and whether a cut changes long-run sinking is untested.
- **Fitted record.** The column is fitted to the cumulative 2012-2022 leveling, not to a policy difference. The deep piles of 2024-25 are report-only.
- **Step A (model-only).** A leveling-fitted column compensates for damped heads fan-wide (fitted/true avoided 0.93 at twice the damping), not zone by zone: 0.75 in proximal Yunlin and 0.77 in mid Changhua. Step A1 ran on R0 and R1c, not on the page's run. Step A2's lever ranking was computed on a superseded response basis (`response_basis.npz`, 2026-09-20, commit `2f21f81`), so its "ranking robust" is withdrawn as a statement about the page's levers; A2's result is left as written.
- **Pending.** The only fully prospective scores are the 2026-11 to 2027-10 registrations.

Post-hoc model-only checks (2026-10-24; `results/dataprep/reports/N1_gaps_2026-10-24/`).
Each is post hoc, model-only and not a test: model outputs only, no observation read,
nothing scored.
- **A2b, lever ranking on the page's basis** (post hoc, model-only, not a test). The A2 procedure rerun unchanged reproduces A2 on A2's own basis. On the page's basis, the page payload and R0's basis, the cm ranking (irrigation, aquaculture, livestock, domestic, industry, other) holds in 100 % of draws at both starts. The per-100-GWh/yr ranking is a non-robust near-tie (2.84-3.10 cm per 100 GWh/yr on the page; order kept in 3 % of draws). Hot spots are fragile: the irrigation lever's top-decile cell Jaccard has 5th percentile 0.00.
- **Transfer from R0 to the page's model** (post hoc, model-only, not a test). The N1 verdicts were scored on `stage3_datum_gate_cwa` (R0) in Yunlin and R1c in Changhua; the page shows `stage3_datum_gate` (`datum_gate_innov`). Their avoided subsidence by 2032 differs by 2-4 % fan-wide (cut30 0.94 against 0.96 cm, retire_aqua 1.31 against 1.26 cm), with cell correlation 0.9996 or more.
- **Hot-spot stability** (post hoc, model-only, not a test). The cut30 headline leader Baozhong leads in 83 % of the page model's 36 runs, 72 % of R0's and 57 % of the A2b draws; Baozhong, Beigang and Dongshi are the top three in every run. retire_aqua's leader Taixi leads in 61 % of the page's runs. Neither leader is stable by the check's rule; read each headline as a group of three.
- **Elastic split** (post hoc, model-only, not a test). On ensemble-mean heads under mean forcing, no cell falls below its preconsolidation head after Jan 2026, so the avoided subsidence is all elastic (inelastic 0.000 cm of 0.94 cm for cut30 and of 1.31 cm for retire_aqua by 2032), 98-99 % of it arrives by Dec 2026, and none is left by 2032 if cut30 is lifted in Jan 2030 with the heads returned abruptly to the baseline (a splice of the ensemble-mean heads, not the flow model's own months-long return). A dry year below the 0.24 m (5th percentile) margin is not in the runs.

Changhua levers: **not supported.** Neither R0 nor the page's model has Changhua pumping. The census model's
2007-10 dose-response came out opposite in sign to the leveling (observed -0.50 against
+0.80 cm/yr per GWh/yr). Neither registered fix closed the gap: F1 failed its design gate
and F2 changed nothing.

Known defects in the head targets. Every cited fit (the page's run, R0, the G1 and step C
refits, R1c) uses the historical defaults of `heads.build_head_field` and
`calibrate_flow.prepare_series`. None was rescored.
- **Sample-weighted monthly means.** After the hourly-to-10-minute logging switch on 2019-02-20, February 2019 shifts by up to 1.7 m at 13 Yunlin wells. The fix (`--day-weighted-months`) is opt-in and reaches only `calibrate_flow`, `inputs` and `forward`. `challenge`, `coverage`, `rescore_temporal`, `explorer3d`, `calibrate_mlcw` and `sk_leveling` still build equal-sample heads, so a day-weighted model (Q0) is scored against equal-sample observations.
- **Coverage counted against hours.** 10-minute records pass the 80 % coverage rule at up to six times their true coverage (`--strict-coverage` is opt-in).
- **Back-filled targets.** Interior gaps are interpolated and leading or trailing gaps filled by the nearest value, and the result becomes the fit target (`--no-backfill` is opt-in).

Details: [docs/TWIN_EVIDENCE_2026-10.md](docs/TWIN_EVIDENCE_2026-10.md), sections 12 and 13.
