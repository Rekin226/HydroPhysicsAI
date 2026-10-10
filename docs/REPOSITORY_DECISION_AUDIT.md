# Repository audit: subsidence forecasts and mitigation decisions

Assessment: October 6, 2026. Main checkout `ad9bcf6`, with the uncommitted AMP
pilot; isolated decision-evidence worktree `d86c50b` inspected separately.

**October 7 follow-through:** the authorized reliability fixes are implemented
locally. See [implementation and verification](RELIABILITY_IMPLEMENTATION.md) for
their current status. Findings below describe the audited pre-fix state; the
scientific-readiness verdict remains unchanged. Historical scores were not rewritten.

## Verdict

The repository is a substantial research platform, but the available evidence does
not yet support an operational subsidence forecast or a winning mitigation policy.
The next investment should be in **valid measurements, trustworthy evaluation,
calibrated pumping inputs and a small depth-matched experiment**, reusing the
existing solver. Another neural architecture, wholesale migration to MODFLOW, or
a larger parameter search has no demonstrated incremental value at present.

This audit changes the earlier recommendation: delayed drainage, canal supply,
pumping substitution, purpose-specific conversion, differentiable coupling and
uncertainty machinery already exist. Their existence is not the missing milestone;
their independently demonstrated skill and physical identification are.

Software repairs below have direct reliability value. Pumping calibration and a
depth-resolved model are testable scientific hypotheses, not promised accuracy gains.
No amount of code review alone establishes intervention effectiveness.

## Scope and verification

The structural inventory covers 149 Python files and 38,418 lines across the main
package, AMP, application and tests, before this audit's small test edit. I traced
the main paths from inputs through fitting, scoring, scenarios, rendering and local
publishing; reviewed the research state and negative experiments; inventoried 44
saved parameter artifacts; and compared the isolated audit changes with this checkout.
This is a repository-wide scientific/software review with targeted deep checks,
not a claim that every line or every historical result was independently verified.

Work performed:

- Collected 556 tests and ran CPU checks, with bounded long runs. Final outcomes are
  recorded in the verification section below.
- Recomputed the extended head-forecast scorecard directly from the saved arrays,
  using identical finite observation rows across all methods and equal well weight.
- Reproduced neural forecast cutoff leakage, dependence on future target presence,
  unequal point-score support, and acceptance of an unrelated research release report.
- Checked the AMP pilot and optional-dependency behavior. Corrected my new AMP test
  module to skip explicitly when SciPy is absent, consistent with foundation CI.
- Inspected data coverage and source/provenance inventories. I did not re-download all
  vendor records, refit all 44 configurations, rerun GPU training, or conduct field work.
- Kept real data and diagnostic artifacts local. No deployment, commit or external
  data request was made in this audit.

Local diagnostics are in `results/repository_audit/`: module/run inventories,
reproduced findings, recomputed scores, lint output and test logs. These ignored
artifacts are not a portable release by themselves; retain them with the data snapshot.

## Existing capabilities: reuse rather than rebuild

| Area | What is already implemented | Actual remaining gap |
|---|---|---|
| Groundwater physics | Implicit multilayer flow, gradients, water-balance and analytical/reference checks; optional delayed stores | Field dynamics, parameter identification and observation mapping; numerical agreement is not forecast skill |
| Deformation | Elastic/plastic/viscous column; optional delayed aquitard term; zonal and horizontally banded columns | Vertically resolved, measurement-matched deformation and independently constrained stress history; horizontal bands are not depth bands |
| Pumping | Shared-meter cleaning, purpose groups, energy-to-volume conversion, extra hydraulic head and purpose-specific efficiencies | Independently measured conversion, historical meter/well/depth mapping, missing/unregistered abstraction, origin-time availability |
| Surface water | Canal/recharge histories and variants, delivery factors, crop calendar and pumping substitution | Consistent use in the chosen model; measured delivery/recharge response and feasible replacement supply |
| Coupling | Differentiable flow-to-compaction graph in `coupled.py` | The inspected column calibration workflow fits deformation against fixed modeled heads; joint multi-observation fitting is not established by the graph alone |
| Statistical comparators | Persistence, seasonal models, distributed-lag decomposition and head-update experiments | Fair, frozen forecast comparisons for the actual surface/depth subsidence target |
| Uncertainty | Local parameter approximation, bounded ensembles and rheology alternatives | Observed predictive coverage, forcing/measurement errors and structural uncertainty; ensemble agreement is not an outcome probability |
| AMP | Original algorithm, harmonic extension and past-only pilot | No observed incremental forecast gain on the limited eligible sample; do not prioritize production integration |
| Neural surrogate | Solver-trained one-step FNO with rollout evaluation | Can only accelerate its teacher; current input/state and split limitations must be resolved before broader use |
| Application/release | Research viewer, offline assets, source hashes, local immutable publishing, rollback and observation freshness checks | Bind scientific evidence to the exact release and avoid stronger decision labels than the evidence supports |
| Isolated audit branch | Geometry/observation/evidence utilities, synthetic depth information tests and presentation changes | Not integrated into this checkout; distinguish synthetic readiness from field validation when integrating |

Primary code: `hydrophysics/twin/{flow,compaction,pumping,surface_water,scenario,
coupled,calibrate_coupled,uncertainty,surrogate,release,update_cycle}.py`.
The public `app/` groundwater demonstration and the research twin viewer are
different products. The older neural groundwater benchmarks do not certify
subsidence scenarios.

## Verified software findings

P1 means resolve before relying on the affected forecast or release claim; P2
means a material limitation in a particular experimental path. These are not all
failures of the same model: the neural forecast defects do not invalidate the
separately recomputed twin scorecard.

### P1 — Neural training windows cross the held-out boundary

`hydrophysics/models/forecast_lstm.py:113` accepts a training origin when its mask
is true, but does not require its future target window to remain in training.
`fit` passes the origin mask at line 164. A synthetic seven-day horizon yielded
seven training windows with targets inside the holdout, reaching March 17 after
the holdout started March 11. A longer horizon can cross farther.

Require every supervised target in a training sample to precede the training
cutoff. Verify that changing held-out targets cannot change training samples or
fitted parameters with a fixed seed. Historical affected scores need rerunning.

### P1 — Neural inference requires future observed heads to be present

`forecast_lstm.py:115` tests finiteness over both history and the future target;
`_predict` reuses that supervised sample builder at line 210. Hiding future heads
in a synthetic live forecast changed seven finite predictions into zero at the
same origin, despite unchanged observed history and supplied forcing.

Separate supervised target construction from inference. Test the final observed
origin with an explicitly supplied future time/forcing axis and all future heads
missing. The model also explicitly assumes perfect future forcing; report that
conditional experiment separately from a forecast using information available
at issuance.

### P1 — Local publication does not bind acceptance to the published model

`hydrophysics/twin/update_cycle.py:55` checks `passed` and records the report hash,
but does not compare the acceptance report's model identity, prediction artifact,
protocol or task with the rendered run. With fresh, correctly hashed synthetic
observations, an unrelated old passing report allowed a new synthetic page to
reach state `published` in a temporary local directory.

Require explicit identities for the model, data, predictions, protocol, task and
applicable period; compare them against the release manifest before publishing.
Reject a stale/mismatched report and preserve the prior release. This is a defect
in the local research workflow, not evidence that a public production system was
compromised. Existing observation-hash and vintage checks are useful and remain.

### P1 — The corrected head-update method is not the default operational path

`hydrophysics/twin/forward.py:622` still defaults to historical `method="field"`;
normal update/hindcast call sites omit an explicit method. The experimental
innovation method is available and documented but is not automatically used by
these paths. The historical interpolation can inject spatial mapping differences
as state changes, even when the simulated values match observations at wells.

Preserve old-run reproducibility while making the method explicit in new run
configuration and provenance. Promote the innovation path only with the existing
observation-operator invariance checks, matched hindcasts and deformation restart
checks. Do not interpret a state correction as a physical pumping benefit.

### P1 — The suite passes while adjoint solver convergence checks warn

The complete CPU run emitted convergence warnings from `flow.py:213`, including
the anomaly-loss and coupled-gradient tests. Re-running those two existing tests
with convergence warnings treated as errors gave **two failures in this checkout**
and **two passes in the isolated audit worktree**. The two test files have identical
hashes across the checkouts. Logs are `numerical_main.log` and
`numerical_worktree.log` in the diagnostic directory.

The isolated branch starts adjoint solves at zero rather than at the forward head
and handles an exactly zero right-hand side directly. Integrate and regression-test
this existing fix before new calibration, and make unintended convergence failures
fail the relevant tests/run gate. Preserve explicitly intentional nonconvergence
tests. A huge relative residual for a zero/tiny right-hand side does not by itself
quantify head error; this reproduction demonstrates a solver/convergence-control
problem, not the magnitude of any historical forecast error or a new accuracy gain.

### P2 — Legacy point-forecast metrics compare different observation subsets

`hydrophysics/forecast_eval.py:68` masks finite observations only; downstream
metrics separately discard nonfinite predictions for each method. The synthetic
example scored 43 model predictions against 50 available persistence predictions.
This does not establish the direction of bias, but makes comparisons unfair.

Use a common support mask and publish availability/missing-forecast rates. This
finding concerns the legacy point evaluator; the extended twin scores below were
recomputed on common support and matched their report.

### P2 — Surrogate validation and inputs do not cover all supported physics

`hydrophysics/twin/surrogate.py:202` randomly splits one-step samples, allowing
steps from the same generated trajectory on both sides. The builder explicitly
warns that slow-store state and canal deliveries can affect its targets without
being input channels. Aggregate energy also does not preserve purpose allocation
when conversion varies by purpose.

Hold out complete trajectories/scenarios and test multistep stability. Either
include all necessary state/forcing channels or reject incompatible parent models.
Do this only when simulation throughput is the measured bottleneck; a more accurate
emulator cannot repair an inaccurate physical model.

### Fixed during audit — My AMP tests broke a foundation-only installation

The new test file imported SciPy-dependent modules before checking whether SciPy
was available. Foundation CI deliberately installs only the foundation, development
and explorer extras. A controlled missing-SciPy import reproduced collection failure.
The test now uses `pytest.importorskip("scipy")`: it skips cleanly in that case and
all 13 AMP/pumping checks pass in the research environment. This repairs test
portability, not AMP accuracy.

## Scientific and data limits that software tests cannot settle

1. **The target is not reconciled.** Whole-column deformation is not the same
   observation as a specific ring interval, and neither automatically equals
   surface movement. Exact Tuku well screens, head units/datum and instrument
   references remain unverified in the source audit. Total well depth is not a
   screen interval. Fit nothing to a guessed physical mapping.
2. **Better inputs have helped, but do not yet produce sufficient forecasts.**
   The forcing experiment improved head RMSE using updated pumping; the strongest
   simple comparator still won. This is evidence to investigate pumping, not
   proof that a new conversion model will improve subsidence forecasts.
3. **Energy is not a measured volume intervention.** `PumpingScenario` scales
   kWh. The existing conversion depends on efficiency and total dynamic head;
   equal percentage energy changes need not produce equal percentage volumes.
   Existing canal substitution converts volumes using a reference head. Add
   volume targets, dynamic accounting and feasible supply constraints to these
   components instead of writing a parallel pumping system.
4. **Observation and forcing quality differ by path.** Historical head coverage
   counts can be inflated by ten-minute sampling; the opt-in strict mode counts
   months with any data, while the new challenge has stronger hour/day thresholds.
   Pump aggregation drops missing values into zero-initialized energy totals, so
   coverage metadata must distinguish no measurement from no use. The forcing
   experiment's high retained-meter coverage does not prove all abstraction is
   represented. Freeze one target-specific QC/availability contract.
5. **Repeated retrospective selection is not a final independent test.** Many
   configurations have been compared on the same historical windows. The current
   validator verifies self-reported registration dates and independence flags; it
   cannot prove a forecast existed before the target or that a holdout was unused.
   Archive issued predictions, source availability and protocols before evaluation.
6. **Parameter agreement is not calibrated confidence.** The main viewer still
   maps parameter-set agreement to “robust”/“likely”. It includes caveats, but those
   labels are stronger than measured forecast coverage or causal evidence. Reuse
   the isolated audit's more literal presentation. Require observed coverage and
   structural comparisons before giving decision probabilities.
7. **Spatial targeting is not validated.** The known column-zone boundary creates
   a deformation discontinuity; smoothing variants often became degenerate or
   lost ring skill. Do not select priority pumping areas from this discontinuity.
   Distinguish real geological transitions from an imposed numerical zone edge.
8. **Cumulative avoided displacement does not prove sustained mitigation.**
   Existing structures differ in whether intervention benefits mostly occur as
   immediate rebound or continued slowing. Depth-matched deformation, surface
   records and a historical intervention or monitored pilot must discriminate them.
   The policy-response sanity gate is not observed causal validation.

The isolated audit's three-band depth design has conditional algebraic rank 3/3;
larger free designs are rank deficient under the observed correlated drivers.
That supports keeping the first experiment small. It is not a fitted consolidation
law, a forecast score or evidence that the three physical units are correctly mapped.

## What the numerical evidence actually supports

Different rows below use different tasks/support and must not be treated as one
league table.

| Experiment | Evidence | Interpretation |
|---|---|---|
| Frozen field head challenge, Jan 2023–Aug 2026 | Reported 2.072 m model RMSE versus 1.667 m climatology; 158 wells/6,186 well-months | Fails improvement gate; published archive arithmetic was checked by the isolated audit |
| Updated-forcing head experiment | Reported control 2.067 m, updated pumping 1.845 m, comparator 1.630 m | Real improvement relative to the old forcing, insufficient against the baseline |
| Extended one-step head forecasts, same observation rows | **Independently recomputed:** hybrid 0.894190 m, seasonal persistence 0.844684 m, previous month 1.325549 m, physical 1.982789 m, seasonal 1.537337 m; 157 wells/5,908 well-months | Reproduces the saved report; hybrid loses overall to seasonal persistence. Does not measure surface subsidence skill |
| Recent AMP pilot | Electricity 0.647208, +original AMP 0.647521, +version two 0.649405, +duty 0.656655; seasonal persistence 0.637121, native head units | No incremental gain; only 18 eligible test well-months, seven with duty support; local units require source confirmation |
| Whole-column versus aggregate ring comparison | Reported 1.528 versus 1.538 cm/year baseline error | Tiny nominal edge with mismatched depth geometry; insufficient to justify subsidence forecasts |

The extended score recomputation used `results/twin/update_experiment/extended_predictions.npz`,
intersection of finite observations/predictions for all five methods, at least 12
common months per well, and square root of mean per-well mean squared errors.
It checks scoring arithmetic, not a fresh refit or prospective forecast issuance.

## Revised work order and added-value tests

| Priority | Concrete work | Why it adds value | Required evidence before expanding |
|---|---|---|---|
| 1 — Reliability, low compute | Integrate and test the existing adjoint fix; repair the forecast sample/evaluator and publication binding; explicitly configure observation updates; selectively integrate audited evidence/UI utilities | Removes demonstrable sources of solver warnings, misleading scores and release claims | Solver convergence, boundary/future-target invariance, matched support, stale-report rejection and reproducible run identities |
| 2 — Target and measurements | Reconcile one site's pressure screens/units and ring/surface references; archive forecasts and inputs | Makes the predicted quantity observable; identifies which extra field record is useful | Source-backed mapping, clear sign/datum, stable measurement intervals and frozen QC |
| 3 — Subsidence benchmark | Establish observed rate/seasonal/persistence forecasts at proposed 1/3/6/12-month horizons; test one frozen existing physical configuration | Establishes whether any physical complexity improves the user's actual target | Untouched time/site tests, same support, acceptable bias and interval coverage; enough independent observations |
| 4 — Electricity calibration | Reuse meter cleaning and conversion; collect paired energy/discharge/dynamic-head measurements by purpose and setting; propagate conversion uncertainty | Constrains an external driver that already showed forecast sensitivity | Held-out pumps/periods and water-budget checks; then one-factor forecast/scenario ablation |
| 5 — Small physical discrimination test | Reuse delayed physics and differentiable coupling in the source-supported depth design; compare baseline VEP/delayed alternatives | Tests sustained compaction response rather than adding unconstrained parameters | Depth sums, conservation, delayed/inelastic reference cases, identifiable parameters and held-out deformation skill |
| 6 — Feasible interventions | Extend existing scenario/substitution code to volume schedules, supply limits and joint uncertainty | Makes scenario comparisons correspond to implementable actions | Measured action timing/volumes, confounding assessment, sustained-rate response and robust or explicitly unresolved ranking |

No universal RMSE or completion date can be responsibly invented here. Error
tolerance depends on the smallest useful reduction, location/assets and consequences
of wrong decisions. Those criteria should be set before the final validation, not
selected to make the model pass.

Do **not** repeat unconstrained efficiency fitting, horizontal smoothing, storage
or recharge sweeps already screened in `docs/superpowers/STATE.md` without a new
observation or a specific falsifiable hypothesis. Do not prioritize AMP expansion,
FNO/GPU scaling, a wholesale Pastas/CSUB migration, or basin-wide depth complexity
before these gates. Pastas is a possible comparator; CSUB is a useful physical
reference; PEST++ is optional calibration tooling. None is evidence of project skill.

## Verification record

- Repository lint: passed.
- AMP/pumping focused suite after the dependency fix: 13 passed.
- Missing-SciPy simulation after the fix: explicit skip, no collection error.
- Complete CPU suite: **552 passed, 7 skipped, 122 warnings**, 829.47 seconds,
  exit code zero. Skips concerned unavailable real-data fixtures; passing synthetic
  checks does not establish performance on those field records. The initial sandbox
  collection found 556 tests; the completed host-environment run executed 559 cases.
- Solver warnings promoted to errors: **two failures on main, two passes on the
  isolated audit branch**, using identical anomaly-loss/coupled-gradient tests.
  Consequently the ordinary suite's success is not a clean numerical-health verdict.
- Two smaller zero-right-hand-side/large-head examples from the isolated branch
  passed against main; those inputs alone did not expose the problem. The full-suite
  warnings motivated the stronger reproduction above.
- A redundant non-flow subset was stopped once the complete suite finished; it is
  not counted as another completed test run. No audit test jobs remain running.

## Supporting records and methods

- [Revised development plan](SUBSIDENCE_DECISION_ROADMAP.md)
- [AMP experiment](AMP_FORECAST_EXPERIMENT.md)
- [Twin release evidence](TWIN_RELEASE.md) and [assimilation findings](TWIN_ASSIMILATION.md)
- [Research state and rejected experiments](superpowers/STATE.md)
- Isolated worktree: `../HydroPhysicsAI-decision-evidence/docs/DECISION_READINESS.md`,
  `SITE_GEOLOGY_SOURCE_AUDIT.md`, `DEPTH_DATA_INVENTORY.md` and
  `DECISION_EVIDENCE_HANDOFF.md` (paths relative to the repository root).
- [USGS CSUB documentation](https://www.usgs.gov/publications/documentation-skeletal-storage-compaction-and-subsidence-csub-package-modflow-6): reference for delayed and elastic/inelastic compaction.
- [USGS One-Water/Farm Process](https://pubs.usgs.gov/publication/tm6A60): demand/supply accounting precedent.
- [Guantao project account](https://www.hydrosolutions.ch/projects/rehabilitation-and-management-strategies-of-overpumped-aquifers-in-a-changing-climate): pump-test-calibrated electricity inference precedent, not transferable local coefficients.

The engineering audit is a basis for targeted development. Decision readiness
remains unproven until the measurement, predictive and intervention gates pass.
