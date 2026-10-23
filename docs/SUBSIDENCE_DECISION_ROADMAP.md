# From the audit to subsidence forecasts and mitigation decisions

**Assessment date: October 6, 2026. Status: development plan, not a validated
management recommendation.**

**Implementation update, October 7:** the software reliability stage is implemented
locally; see [changes, compatibility and verification](RELIABILITY_IMPLEMENTATION.md).
The measurement, independent field validation and intervention gates remain open.

**Revised after the repository-wide review:** read
[Repository decision audit](REPOSITORY_DECISION_AUDIT.md) first. Delayed drainage,
canal supply/substitution, purpose-specific energy conversion, differentiable
flow/compaction coupling and parameter ensembles already exist. The work below
means validating and extending those components, not rebuilding them. Confirmed
adjoint convergence, forecast-window, live-inference, score-support and release-evidence defects must
be repaired before relying on the affected scores or publishing claims.

The user objective is to forecast near-term land subsidence, identify its main
controllable drivers, and compare feasible interventions that reduce it. A good
short-term forecast alone does not establish what would happen after changing
pumping. Both capabilities must be tested. Proposed forecast horizons are 1, 3,
6 and 12 months; mitigation evaluation also needs longer follow-up to distinguish
temporary rebound from a sustained rate reduction. These horizons are planning
assumptions, not stakeholder-approved acceptance thresholds.

## Recommended architecture and methods

Keep the existing platform, data adapters and viewer. Develop two connected but
separately scored outputs: a continually updated observational forecast and a
physical intervention simulator. Use a shared, versioned observation and forcing
record. Do not treat assimilation increments or statistical forecast corrections
as physical recharge, pumping or intervention benefits.

| Component | Method to investigate | Why it addresses this project's gap | Required evidence |
|---|---|---|---|
| Pumping inputs | Calibrated electricity-to-volume conversion, grouped by pump purpose and hydraulic setting | Uses an existing large external dataset, rather than inferring all pumping from head fluctuations | Paired discharge, electricity and total dynamic head measurements; dated meter/well mapping; uncertainty and held-out conversion checks |
| Seasonal water budget | Extend the existing canal, crop-calendar and substitution code with explicit demand/supply constraints, following the Farm Process concept | Makes existing scenario levers physically feasible and checks water-budget consistency | Crop areas/calendar, irrigation efficiency, surface-water deliveries, return flows and pumping constraints |
| Short-term head response | Test existing distributed-lag/decomposition and seasonal-persistence comparators first; Pastas only for a missing response/noise capability | Cheap tests of amplitude, lag and memory; a package switch alone is not an improvement | Chronological and site holdouts; forcing information actually available at each origin |
| Subsidence mechanism | Map existing elastic/inelastic/delayed components to supported depth intervals; benchmark relevant limits against MODFLOW 6 CSUB | Tests the whole-column/ring mismatch and rebound versus continued consolidation without adding unconstrained complexity | Screen assignments, compressible thicknesses, material/stress-history constraints, depth-matched deformation and surface measurements |
| Calibration and uncertainty | Use existing differentiable coupling and ensembles for a small joint multi-observation experiment; an ensemble smoother is optional | Tests the staged workflow and extends uncertainty beyond parameter spread | Identifiability tests, defensible measurement covariance, held-out interval coverage and sensitivity to alternate structures |
| Scenario selection | Constrained comparisons and a benefit–cost–water-shortfall frontier | Defines “best” using feasible actions and robust outcomes, rather than largest displayed displacement difference | Explicit water supplies, costs, compliance, protected locations and acceptable residual risk |

Pastas represents groundwater responses to multiple stresses through response
functions; it is a useful diagnostic comparator, not proof of causal attribution
or a stand-alone subsidence simulator. [Collenteur et al., 2019](https://doi.org/10.1111/gwat.12925).

CSUB distinguishes elastic and inelastic compaction and can resolve delayed
pressure equilibration inside compressible interbeds. The project already has
lumped delayed stores and an optional delayed compaction term. The missing evidence
is depth-matched field validation and discrimination between those structures,
not the mere existence of delayed drainage in code. Adopting a package
does not supply local geology or field validation. The existing project reference
test checks only the elastic limit; the next reference tests must include delayed
drainage, threshold crossing and recovery. [USGS CSUB documentation](https://www.usgs.gov/publications/documentation-skeletal-storage-compaction-and-subsidence-csub-package-modflow-6).

PEST++ supports ensemble history matching, sequential assimilation and optimization
under uncertainty. It can wrap an existing model rather than requiring a new
simulator. Nonlinear ensemble spread is not guaranteed calibrated predictive
uncertainty: test coverage on observations excluded from fitting and selection.
[PEST++ project](https://github.com/pestpp/pestpp) and
[PESTPP-IES limitations](https://help.pesthomepage.org/pestpp-ies.html).

## Electricity: a central input, with a conversion problem to solve

The [field-data acquisition assessment](FIELD_DATA_ACQUISITION.md) now identifies
an existing WRA/NCKU calibration table and Tuku intervention-trial records to
request, with [unsent request drafts](UNSENT_FIELD_DATA_REQUESTS.md). Raw paired
calibration records and independent validation remain outstanding.

The AquaScope follow-up acquired recent local weather and groundwater records.
Its source checks also found incomplete rainfall totals, provisional WRA heads and
eight coordinate discrepancies among the 61 curated wells. Use the conservative
field importer and [quality findings](FIELD_DATA_ACQUISITION.md) before extending
calibration or evaluating spatial interventions; the new records are not automatically
independent validation targets.

The repository already has monthly electricity histories, pump purpose, installed
power, meter identifiers and locations. `hydrophysics/twin/pumping.py` deduplicates
shared meters and converts energy using efficiency and total dynamic head.
`scenario.py` separates irrigation, aquaculture and other uses. Reuse that work.

For energy E in kWh, a useful physical conversion is:

```
pumped volume [m³] = 3.6e6 × E × wire-to-water efficiency
                    / (water density × gravity × total dynamic head [m])
```

Total dynamic head includes static lift, pumping-induced well drawdown, friction
and discharge requirements. The present code includes an extra-head term because
static lift alone led to implausibly large inferred volumes. That correction is
not an independent measurement of the conversion. Electricity alone cannot
separately identify efficiency and hydraulic head; several combinations can
explain the same energy record.

Recommended next work:

1. Audit date coverage, missing months, duplicated/bimonthly billing, shared meters,
   non-pump loads, purpose labels, historical registry changes and publication lag.
   A missing bill is not zero pumping, and a billing month is not necessarily a
   month of physical use.
2. Obtain simultaneous electricity, discharge and dynamic-head observations on a
   sample stratified by purpose, lift and equipment class. Select sample size from
   observed variability and decision sensitivity; do not assert an arbitrary number
   of pumps is sufficient. Repeat across representative operating conditions.
3. Fit purpose/site-group conversion distributions with partial pooling; reserve
   entire pumps and operating periods for validation. Track unregistered pumping
   and non-pump electricity as separate uncertainties, not whatever residual remains.
4. Use crop water demand, effective rainfall, canal deliveries and return flows as
   independent checks on water balance. They are constraints with uncertainty, not
   a target that guarantees the inferred pumping is correct.
5. Express interventions as dated water-volume schedules by area and aquifer.
   The current `PumpingScenario` scales energy. A 30% energy reduction does not
   generally remain a 30% volume reduction when lift or efficiency changes.
   Report both the requested water reduction and the implied energy trajectory.

A directly relevant implementation precedent is Guantao: the project used pump
electricity as a proxy, with well conversion factors determined by pumping tests,
and combined it with groundwater monitoring and water-budget planning. It supports
the approach, not transfer of its coefficients or claimed benefits to Choushui.
[Project team's account](https://www.hydrosolutions.ch/projects/rehabilitation-and-management-strategies-of-overpumped-aquifers-in-a-changing-climate).

The USGS One-Water/Farm Process approach computes agricultural demand and allocates
available supplies, including groundwater needed after other supplies. Borrow its
accounting principles; adopting that entire software stack is not yet justified.
[USGS One-Water report](https://pubs.usgs.gov/publication/tm6A60).

## Identify the dynamics before ranking interventions

The working causal chain is water demand and supply → pumping by depth/location
→ aquifer pressure → pressure inside compressible layers → effective stress and
delayed strain → measured ground movement. Other surface-motion components and
instrument/reference changes must be accounted for in the measurement model.

Test, rather than assume, the roles of:

- Seasonal pressure decline and recovery: how much deformation is reversible?
- New stress maxima: which layers develop additional permanent compaction?
- Delayed drainage: how much ongoing sinking reflects earlier pressure declines?
- Spatial and vertical pumping distribution: where do equal volume cuts change
  stress in the layers contributing most to subsidence?
- Replacement water and return flow: does a local reduction shift pumping nearby,
  change irrigation recharge, or leave water demand unmet?

Delayed compaction can continue while aquifer heads recover, and permanent
compaction is not generally reversed by restoring head. Consequently a short-lived
uplift is not sufficient evidence of sustained mitigation.
[USGS explanation of residual compaction](https://pubs.usgs.gov/sir/2007/5251/section4.html).

The audit's three-band 土庫 structure is a candidate for an initial prototype,
not established geology or a fitted physical model. The larger free-parameter
structures failed information/rank checks under the recorded head correlations.
Start with identifiable complexity and independent parameter bounds. Add spatial
or material detail only when observations can constrain it.

## Scenario set to compare — no winner selected

Use a common baseline, weather realizations and initial state for paired comparisons.
Change the physical action and all documented consequences, not a plotted head curve.

| Candidate action | Comparison that matters | Feasibility or displacement check |
|---|---|---|
| Uniform pumping-volume reduction | Reference intervention against which targeting can be assessed | Water deficit, compliance and replacement supply |
| Targeted reductions by place and aquifer | Equal total water reduction allocated where compaction sensitivity is greatest | Screen mapping; avoid exporting drawdown to another area |
| Seasonal pumping limits or pressure triggers | Reduce damaging low-head episodes while preserving useful supply | Whether timing changes annual demand, pumping elsewhere or crop production |
| Surface-water substitution plus pumping reduction | Same demand met with less groundwater where supplies permit | Actual canal capacity, drought reliability, return flow and distribution losses |
| Demand reduction or crop/irrigation changes | Net reduction after behavioural and return-flow effects | Cost, crop yield, compliance and rebound in irrigated area |
| Managed recharge where hydrogeologically feasible | Whether recharge reaches the stressed layers at a useful time | Water availability, quality, location and hydraulic connectivity; shallow recharge is not automatically deep pressure recovery |

Compare near-term displacement, subsequent annual sinking rate, spatial hotspots,
and differential movement where infrastructure is the objective. Also report water
shortfall, costs and implementation delay. Compare rankings across conversion,
geology, stress-history, weather and compliance assumptions. If plausible models
disagree on the preferred action, report the disagreement and the observation
that would most reduce it. Do not force a single “best” label.

## Work packages and release criteria

| Order | Concrete deliverable | Pass condition | Current dependency |
|---|---|---|---|
| 0. Repair evidence paths | Integrate the existing adjoint fix; separate forecast training/inference; fair scores, release-to-evidence binding and explicit observation-update configuration | Solver convergence; no held-out training targets; forecasts work without future observed heads; mismatched acceptance reports fail; tests and rerun manifests identify the active code | Verified findings and reproductions in the repository audit; selectively integrate the existing audit worktree |
| 1. Define decision and observations | Site/horizon/action contract; immutable data and QC manifest | Exact target is surface motion or a stated depth interval; units, sign, datum, source versions and availability are documented | Priority is near-term subsidence plus mitigation; geography, actionable levers, costs and acceptable error still need stakeholder specification |
| 2. Establish honest forecast baselines | Rolling-origin surface-displacement forecasts at 1/3/6/12 months; seasonal/rate/persistence baselines; matched scorecards | Sufficient site and temporal coverage under fixed QC; predictions recorded before targets become available | Surface GNSS reference/revision issues; leveling is dated; rings are depth intervals, not direct surface displacement |
| 3. Quantify pumping | Purpose-stratified energy inventory and calibrated volume distributions | Held-out conversion measurements and water-budget consistency support uncertainty bounds | Paired pumping measurements and dated meter/well metadata |
| 4. Verify deformation physics | One-site depth-resolved prototype and delayed/inelastic reference cases | Synthetic conservation, depth sums, limiting cases and recovery tests; then held-out depth-matched and surface skill | Screen/top-bottom records, stress history and measurement references |
| 5. Jointly test and calibrate | Multi-site heads + depth deformation + surface motion; alternate structures | Beats the strongest relevant baseline on untouched periods/sites; acceptable bias and empirically checked uncertainty by horizon and location | Enough independent data; validation targets must not be repeatedly reused for model selection |
| 6. Test intervention response | Feasible volume-based scenarios; paired uncertainty analysis; documented historic intervention comparison or monitored pilot | Direction, timing and sustained-rate response agree with observed interventions after addressing confounding and substitution; rankings robust or explicitly unresolved | Measured action timing/volumes, controls and follow-up observations |
| 7. Release decision support | Reproducible model/data release, visible evidence status, stale-data handling and forecast monitoring | Every displayed claim links to the appropriate passed evidence; failed updates preserve the last verified release | Scientific gates above, not merely software test success |

An absolute “accurate enough” threshold cannot be set from RMSE alone. It must
relate forecast error and uncertainty to the minimum actionable reduction,
protected assets and costs of false reassurance versus unnecessary restrictions.
Predefine those thresholds with the decision owner before final validation. Report
uncertainty in skill differences with time/site dependence respected; do not count
overlapping annual changes or colocated wells as independent replications.

Surface and depth observations complement one another. The USGS combines InSAR,
leveling, GPS and extensometers for subsidence characterization. InSAR line-of-sight
motion requires an explicit observation operator and reference treatment rather
than being equated directly to vertical compaction.
[USGS measurement integration](https://ca.water.usgs.gov/projects/central-valley/delta-mendota-canal-subsidence.html).

## What can be presented now

The platform can be presented as a research demonstrator with the measured failures,
conditional scenarios and this development program. It cannot yet be presented as
a validated prescription for the best pumping intervention. The current evidence
supports investigating independently calibrated electricity inference and validating
the existing water-budget/delayed-compaction components against correctly mapped
measurements. Depth resolution is a conditional next experiment. It does not support a promised
accuracy, avoided-subsidence amount, completion date, or winning management scenario.

Do not begin a basin-wide expensive refit before the observation definitions and
one-site reference tests are ready. CPU diagnostics and a small physical prototype
come first. A numerical experiment cannot substitute for missing well construction,
instrument-reference or measured intervention records.

## Local evidence used

- [AMP forecasting pilot](AMP_FORECAST_EXPERIMENT.md): leakage-safe implementation,
  no observed incremental gain on the small eligible 土庫 subset.
- [Forcing experiments](TWIN_RELEASE.md): updated pumping helped but did not beat
  the seasonal benchmark; dynamics remained too weak.
- [Observation updates](TWIN_ASSIMILATION.md): corrected mapping helped without
  establishing sufficient forecast or intervention skill.
- Latest isolated audit: `docs/DECISION_EVIDENCE_HANDOFF.md`,
  `docs/DECISION_READINESS.md`, `docs/SITE_GEOLOGY_SOURCE_AUDIT.md` and
  `docs/DEPTH_DATA_INVENTORY.md` on branch `audit/decision-evidence`, inspected at
  commit `d86c50b`. These audit files are not yet part of this checkout.
- The broader implementation inventory, verified defects, negative experiments
  and revised work order are in [Repository decision audit](REPOSITORY_DECISION_AUDIT.md).
  The existing policy-response gate tests model behavior, not observed causal
  effectiveness. Passing it is not authorization to claim a validated intervention.
