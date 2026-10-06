# Subsidence decision readiness: Choushui fan

**Status: research comparison, not a validated intervention recommendation.** This
document separates the decision objective (reducing the *rate* of land subsidence) from
head forecasts, one-time elastic rebound, and cumulative avoided displacement. The
existing irrigation and aquaculture levers are hypothetical changes to pumping energy;
their operational feasibility, water substitution, and compliance are not established.

## Saved evidence and provenance

The frozen head challenge is `results/twin/new_data_challenge/` in the existing local
checkout, evaluated from a December 2022 origin through August 2026. Its 158-well,
6,186-well-month equal-well RMSE is 2.072 m after historical-only datum correction versus
1.667 m for seasonal climatology. This was retrospectively scored after the new data were
acquired. The no-adjustment model scores 8.076 m. The diagnostic
`python -m hydrophysics.twin.evidence` verifies the saved aggregate and produces
matched-sample lead-time, year, layer and private per-well scores. Optional event windows
can be supplied as a JSON list of `name`, inclusive `start`, and exclusive `end`, but their
registration timing is not verified; post hoc windows remain descriptive. The model loses to
seasonal climatology in **every** lead-time bucket (1–6, 7–12, 13–24, 25–36, 37+ months)
and every aquifer layer. No saved predictive bounds permit coverage measurement.

The scenario diagnostic reads the existing annual member sidecars, with no new flow or
compaction run. Both saved model versions use the same three named scenarios:
`baseline`, `cut30` (30% irrigation pumping cut from 2026), and `retire_aqua`
(aquaculture pumping retired from 2026). These are conditional model consequences:

| Physical model | Scenario | Avoided by end 2026 | Avoided by end 2032 | 2028–2032 rate reduction |
|---|---|---:|---:|---:|
| Selected datum model | Irrigation −30% | 0.924 cm | 0.941 cm | <0.001 cm/year |
| Selected datum model | Aquaculture retired | 1.295 cm | 1.305 cm | <0.001 cm/year |
| Previous spread model | Irrigation −30% | 0.430 cm | 0.910 cm | 0.0793 cm/year |
| Previous spread model | Aquaculture retired | 1.300 cm | 1.787 cm | 0.0752 cm/year |

The selected model attributes almost the entire difference to the first year, with
essentially no later rate reduction. The previous model predicts a sustained reduction.
Aquaculture retirement has the larger 2032 cumulative modeled effect for 68 of 72 paired
members of the selected model and all 72 members of the previous model; neither model gives
a robust ranking of the *late rate* effects. Agreement on a positive cumulative difference
does **not** identify which mechanism is right or show that either intervention will slow
the long-term subsidence rate. Annual
December states cannot resolve the month of rebound. The 72 paired parameter/initial
condition/rheology trajectories per model are sensitivity samples, not calibrated
probabilities or predictive intervals. Spatial cell fractions are model-grid diagnostics,
not a map of protected people or assets. Scenario inputs are not verified feasible water
management plans.

### What the local ring downloads can test

The 14-site old clean ring cache was used for prior model checks and the 2010–2025
mechanism review. Fourteen downloaded series match **47,735 overlapping ring values**
in that cache to floating-point precision. They add only 27 later survey rows across
those sites and are not an untouched temporal holdout. Two other downloaded sites are
absent from the old clean cache, inside the model grid, and have 26 sampled months
between February 2023 and May 2025. Their ring measurements were not fitted by the
coupled column, although regional leveling data influenced column selection.

A local hashed protocol was written before scoring their measured changes. Its first
version failed a prespecified 12-pair completeness rule because one site's deepest
ring stopped reporting. The original protocol was retained; a missingness-only
amendment fixed the common NO1–NO26 interval at both sites before examining change
values; those ring labels do not imply identical physical depth at both sites. The
amended protocol hash, source hashes and site-specific output are in the
ignored `results/decision_evidence/` directory. This remains **retrospective**: the
observations had already been downloaded, and the 30 annual pairs overlap in time.

| Twelve-month ring change, two sites | Equal-site RMSE |
|---|---:|
| Zero-change baseline | 2.439 cm/year |
| Median annual rate from 14 old sites, using pre-2023 records only | 1.538 cm/year |
| Selected datum model | 1.528 cm/year |
| Previous spread model | 1.666 cm/year |

These model scores compare an unresolved **total-column proxy** with shortening
between the registered NO1–NO26 rings; the saved forward archive has no vertical
compaction profile. They therefore do not establish depth-matched ring accuracy.
`hydrophysics.twin.observation_operator` supplies an exact-boundary depth operator
for future resolved profiles and refuses unsupported partial-layer allocation.

A bounded 土庫 one-cell head-support comparison, using the two official geology
interpretations without a compaction fit, is documented in
`docs/SITE_GEOLOGY_SOURCE_AUDIT.md`. Its selected-model head ranking and the
saved whole-column subsidence ranking differ. Neither is a site strain validation.

The present four-layer flow solver outputs heads by aquifer, but the calibrated VEP
column is driven by their mean (or another single weighted driver) and returns one
scalar compaction history per cell. Its elastic and delayed inelastic terms are
mechanism components of that **same whole column**, not deformation assigned to
known depth intervals. The one-interbed CSUB comparison in `twin.reference` is a
synthetic elastic-limit test, not a depth profile for the fan. Therefore no valid
depth-resolved field history can be exported from existing solver state or reconstructed
from the saved `subs_mean` array. The viewer source labels its ring R² as a
depth-unmatched proxy; previously generated viewer files are unchanged.

The smallest defensible route to such an export is a separately identified vertical
strain model with surveyed cell-by-cell layer boundaries, a head/stress driver for each
compressible interval, and explicit elastic, inelastic and delayed histories per
interval. It must report the existing scalar-column prediction separately from the
new depth model, record depth and displacement datums and parameter/input hashes,
and demonstrate component-sum and reference-solution checks on synthetic columns.
Only then can the ring operator integrate a registered physical interval and compare
it with new held-out ring measurements; fitting the vertical fractions to the already examined
two sites would not be validation.

The selected model's 0.010 cm/year aggregate edge over the historical baseline is
too small to establish useful skill. It improves on that baseline at one site and is
worse at the other. The descriptive year split also reverses: for 22 pairs ending in
2024, the historical baseline scores 1.195 versus the selected model's 1.531 cm/year;
for eight pairs ending by May 2025, the selected model scores 1.521 versus 2.224.
The model proxy is total-column compaction at one 1-km cell while the observation
is ring-interval shortening. Neither measures an observed pumping intervention,
basin-wide surface subsidence, or predictive interval coverage.

Reproduce the saved diagnostics, writing only to an ignored local directory:

```bash
python -m hydrophysics.twin.evidence \
  --challenge-dir /path/to/existing/results/twin/new_data_challenge \
  --out results/decision_evidence
python -m hydrophysics.twin.scenario_evidence \
  --sidecar /path/to/existing/results/twin_forward/datum_gate.members.npz \
            /path/to/existing/results/twin_forward/physical_spread_apex.members.npz \
  --forward /path/to/existing/results/twin_forward/datum_gate.npz \
            /path/to/existing/results/twin_forward/physical_spread_apex.npz \
  --out results/decision_evidence/scenarios.json
python -m hydrophysics.twin.compaction_evidence \
  --protocol results/decision_evidence/compaction_protocol.json \
  --out results/decision_evidence/compaction_report.json
```

`site_scores.csv` contains observation-derived per-well results and stays local and
ignored. The JSON files record hashes of the saved inputs; copying a score to a public
slide without its model, period, metric and caveat would be misleading.

## Prospective compaction protocol to register before new surveys

1. Freeze the model code and parameter hashes, raw scenario trajectories, column
   geometry, measurement operator, intervention definitions, pumping/forcing vintages,
   and the complete site population. Archive those predictions and a time-stamped protocol
   before receiving any new ring, leveling or InSAR measurements. Record exclusions and
   minimum survey coverage now; do not add sites because they later score well.
2. Make the primary endpoint the change in compaction over a registered 12-month
   interval at each site, in cm/year, with an explicitly declared ring-depth or surface
   displacement operator. Score fixed-origin 1-year and 2-year leads separately. Use the
   same available site-periods for every method, equal site weights, and historical-only
   zero-change, site-trend (where prior site data exist), and regional seasonal/rate
   baselines. Disclose site- and region-level errors and missing surveys. Require the
   physical model to beat the strongest registered baseline, not just zero change.
3. Estimate uncertainty from errors in periods and sites excluded from calibration and
   model selection. Report interval coverage by lead, zone and event as well as point
   errors. A parameter-ensemble spread alone is insufficient coverage evidence.
4. To claim **policy impact**, identify an actual, measured change in pumping with its
   location, date, volume, compliance and any replacement canal water or nearby pumping.
   Prespecify comparison sites and confounders such as rainfall and infrastructure
   movement. Test both the immediate rebound and the later rate difference against
   observed controls. The current two hypothetical scenarios have no such intervention
   record, so a causal policy gate cannot run yet.

A separate 土庫 WRA GNSS surface-height cache ends in March 2025, while the
current public source extends through December 2025. The site series contains
unexplained discontinuities and its vertical reference is unresolved; the bounded
raw-contrast diagnostic in `docs/SITE_GEOLOGY_SOURCE_AUDIT.md` is not a
validated surface forecast or a prospective or causal test.

The next eligible measurement period starts after the last used May 2025 ring downloads;
it also needs new independent surface displacement data because an interval ring does
not by itself measure surface subsidence. The number and location of required sites
should be set from power and coverage analysis with the agency, not chosen after seeing
new outcomes.

## Readiness gates and next work

1. **Validate the compaction mechanism (highest impact; high effort).** The current
   selected model has poor 14-ring fit and the two models disagree on rebound versus
   reduced creep. Obtain independent, dated leveling/ring or InSAR observations spanning
   recovery or a documented intervention, with vertical reference and measurement
   uncertainty. Specifically, seek ring/leveling or InSAR measurements after May 2025
   at sites with documented pumping changes; register the site population, depth-to-
   surface mapping, survey intervals, QC, baselines and rate metric **before** acquiring
   those observations. Lock the calibration and validation split before refitting. **Pass:**
   held-out compaction *rate* and cumulative change beat prespecified simple baselines,
   the post-recovery behavior is captured, and the sign and timing of intervention
   response agree with observations. Head skill alone cannot pass this gate.
2. **Establish feasible, measured interventions (high impact; high effort).** Confirm
   pumping volumes rather than relying only on electricity conversion; document canal
   deliveries, substitution pumping, spatial scope and compliance for each proposal.
   **Pass:** a traceable water-budget scenario exists for the action decision makers can
   actually take, including a plausible substitution range.
3. **Resolve rate-ranking robustness (high impact; moderate analysis after data).** Rerun
   paired interventions across model structures, forcing, datum, rheology and the
   mid/distal column boundary. Report annual rates, cumulative change, lag, cell-level
   heterogeneity and the ranking under each assumption. **Pass:** a prespecified ranking
   remains materially stable; if the ranking or sustained rate effect changes with a
   plausible mechanism, present alternatives without a preferred intervention.
4. **Prospectively test head forecasts and uncertainty (supporting gate; moderate effort
   plus observation time).** The 2023–2026 challenge has already been examined and must
   stay exploratory. Register the model hash, scenario/forcing availability, origin,
   sites, lead times, missing-data rule, baselines and metric before future observations.
   Store predictions before ingesting them, then run `hydrophysics.twin.validation`.
   **Pass:** matched-sample fixed-origin skill exceeds the strongest registered baseline
   across decision-relevant horizons, with calibrated interval coverage and disclosed
   failure strata. Rolling one-step well corrections are a separate product and cannot
   validate the aquifer or policy response.
5. **Decision presentation (low implementation effort; dependent on gates above).** Until
   those gates pass, show the offline viewer as a transparent research comparison: the
   two model mechanisms, annual rate curves, cumulative differences and explicit failed
   forecast checks. A specific avoided-centimetre promise or recommended pumping action
   should wait. Ask stakeholders for the actual action, spatial area, planning horizon,
   feasible levers, and minimum meaningful reduction before setting a decision threshold.

The existing synthetic MODFLOW and elastic CSUB comparisons verify numerical behavior,
not field compaction or policy effects. The 182 km zonal column discontinuity and the
unexplained 2022 head recovery remain structural blockers. See
`docs/superpowers/STATE.md` section 0 and `docs/TWIN_RELEASE.md` for the original
experiments and caveats.

### Saved-field boundary sensitivity (exploratory)

At the 182 km easting line, 41 grid rows have four consecutive active 1-km cells.
For the December 2025–December 2026 modeled total-column change, the median
east-minus-west difference after subtracting the average slope of the neighboring
cell pairs is 1.835 cm in the selected baseline and 4.618 cm in the previous
baseline. Under the selected model's irrigation cut and aquaculture retirement the
same excess is 3.111 and 3.497 cm; under the previous model it is 4.861 and
5.840 cm. These are spatial discontinuities in model output, not measured
subsidence rates or proof that the underlying geology is continuous. The model
structures themselves change the excess substantially. The diagnostic does not
smooth or recalibrate either model. Reproduce from the saved monthly archives with
`python -m hydrophysics.twin.boundary_evidence --forward ARCHIVE.npz --out
results/decision_evidence/boundary.json`; pass `--earlier`, `--later`, and
`--boundary-km` for explicitly labeled sensitivity periods and locations. No
physical correction is justified without better depth and geological constraints.

## Audit closure matrix (2026-10-06)

This is the status of evidence in the isolated worktree, **not** a claim that the
decision gate has passed. The three ignored diagnostics were independently rerun into
`results/decision_evidence/recheck_2026_10_06/`; their JSON content matches the
previous `evidence.json`, `scenarios.json`, and `compaction_report.json` exactly.
The source challenge and forward archives reside in the original checkout and were
read without modification. No new model fitting was performed.

| Gate and status | Evidence and issue | Next action; impact, effort, dependency, acceptance |
|---|---|---|
| Archived-score reproducibility: **pass, limited** | Diagnostic rerun reproduced 158 wells/6,186 well-months, 2.072 m datum-model versus 1.667 m climatology RMSE, and both scenario and ring reports exactly. See `results/decision_evidence/recheck_2026_10_06/` and `hydrophysics/twin/evidence.py`. This verifies arithmetic on archived predictions, not how they were trained. | Preserve source hashes, frozen model and execution environment; low effort. Pass only when an independent machine can regenerate the forecasts, not just rescore them. |
| CG numerical convergence: **pass for saved calibrations and final CPU tests; GPU retest pending** | The selected `results/twin_runs/stage3_datum_gate/stage3_flow.csv` records `cg_maxiter=2000`, `cg_nonconverged=0`, `cg_worst_residual=0.0`; the previous `stage3_spreadL_gate` also records zero. The first broad test run passed 128 tests with 89 warnings. A bounded probe found its largest apparent failure (relative residual 1.68e15) had an exactly zero adjoint right-hand side and absolute residual 1.68e-15. The solver now returns the exact zero solution for that case and starts nonzero adjoint solves at zero instead of the unrelated forward head. The broader run passed 129 tests in 413 s with CG warnings treated as errors; one unrelated tensor-to-scalar test warning remains. A subsequent 23-test temporal/datum run passed with persistent strict CG checks on the optimization cases. An independent large-head adjoint regression matched central finite differences for the summed log-transmissivity gradient (analytic -0.0132306204404, numeric -0.0132306208071; relative difference 2.77e-8), and the focused gradient/zero-RHS tests passed 3/3. One exact-fit datum fixture retains a narrow warning exception: its nonzero adjoint RHS and absolute residual are both about 2.5e-17, below float64 machine epsilon in these units. Archived predictions were not recomputed, and GPU performance of the extra zero-RHS check has not been measured. | Keep true-residual counters on every new run; low to moderate effort. Pass production only with zero failed solves or separately bounded error, and benchmark the revised solver on the available GPU before a new heavy calibration. Numerical convergence does not establish field skill. |
| Forecast-origin integrity: **partial** | `hydrophysics/twin/challenge.py` masks history after the December 2022 origin for datum offsets and baselines and scores matched rows through August 2026. `hydrophysics/twin/forward.py:457` and `scenario.py:196` generate future electricity and recharge from month-of-year means of the record ending at origin; no observed 2023–2026 forcing enters that forward call. However, the challenge was scored after acquisition, source vintages and later model-selection exposure are not fully frozen, and weather/pumping forecasts were never tested. | Freeze dated inputs, source vintages, model-selection log and forecast before a new observation period; moderate effort plus elapsed observation time. Pass a prospective fixed-origin score with no post-origin tuning or input, matched historical-only baselines and horizon strata. |
| Head predictive skill: **fail** | The selected model loses to seasonal climatology overall and in every saved lead bucket and layer; its spatial fold gate also records `r2_kfold=0.643` versus IDW `0.702` (FAIL) in the archived `datum_gate.npz` gate metadata. The previous spread model's fold gate passes, but this does not repair the selected model's future RMSE. | Revisit physical/input structure using historical-only development data, then prospectively test a frozen candidate; high impact, high effort. Pass only if the new model beats the strongest registered baseline at decision horizons and material regions. |
| Depth-matched compaction: **fail** | The two-site ring report labels `depth_match_valid=false`; 30 overlapping annual windows and a 0.010 cm/year aggregate RMSE edge over a historical median cannot validate a total-column model against NO1–NO26 ring shortening. `hydrophysics/twin/observation_operator.py` and `ring_geometry.py` provide boundaries and a future operator, but `forward.py` saves no vertical strain state. | The 土庫 one-site soil/ring overlap tests pass as geometry only; resolve the 24 observed versus 25 later-listed versus 26 original ring inventory and a conflicting 48–51 m material label before field scoring. Obtain corresponding verified geometry at other sites, screen intervals and depth stress/strain constraints before fitting a vertical model; high impact, high effort. Pass synthetic component-sum/reference checks and new site/period ring scores at registered depths, beating site-rate and zero-change baselines. |
| Deep-benchmark cross-check: **blocked by data dictionary** | WRA [basic](https://data.gov.tw/dataset/58335) and [monitoring](https://data.gov.tw/dataset/58339) catalogs define ground-surface-to-depth compression and metre installation depth, but do not specify the cached displacement value unit, sign or datum. The cached generic values cannot safely be converted. | Obtain the instrument series dictionary or a station report giving observed-value scale, reference and sign; low analysis effort, external source dependency. Pass a documented transformation and independent comparison at overlapping stations; reject physically impossible scales. |
| Forcing and intervention realism: **fail** | Saved `datum_gate.npz` and `physical_spread_apex.npz` use the same pre-origin seasonal forcing, irrigation-energy ×0.7 and aquaculture-energy ×0 from January 2026; neither scenario represents verified metered abstraction, canal substitution, compliance or measured intervention. The two model structures reverse the late-rate story (near zero versus about 0.08 cm/year). | Obtain dated metered volumes, canal deliveries and plausible substitution ranges; high impact, high effort. Pass water-budget checks and robust rate effects under forcing, calibration and rheology sensitivity. |
| Uncertainty and spatial credibility: **fail** | Member spread is sensitivity, not calibrated predictive coverage. The selected model has a 1.835 cm 2025–2026 excess at the 182 km column boundary; structural alternatives change it. No archived predictive bounds or observed interval coverage exist. | Use withheld site/time residuals for calibrated intervals and geologic support for boundary transitions; moderate to high effort. Pass 50/90% coverage by site, zone, horizon and event and a prespecified spatial-discontinuity check. |
| Causal policy recommendation: **blocked** | Neither saved scenario was an observed intervention, and the depth comparison and predictive gates above fail. | Register an actual action, pumping date/volume and controls, then compare pre/post rates and displacement against confounders; high impact, high effort and agency cooperation. Pass a prespecified, uncertainty-bounded rate reduction at the affected sites, with substitution and adjacent-area effects reported. |

### Station-specific geology search boundary

The existing WRA zone-50 well catalog was joined spatially to all 16 inventoried ring
sites. Examples of **nearby catalog wells, not proven identical borings**, are
`3132015GW09050321` (14 m from 土庫國中), `3132015GW09050211` (5 m from 秀潭國小),
`3132015GW09080321` (52 m from 仁和國小), and `3132015GW09080211` (83 m from
舊庄國小). The catalog supplies total/finished well depth and layer code, but no screen
top/bottom, lithologic interval, core description or ring-installation log. Exact-ID
queries produced no matching completion records. The official 1999 Geological Survey
[fan report appendix](https://twgeoref.gsmma.gov.tw/GipOpenWeb/wSite/ct?ctNode=217&mp=6&xItem=117602)
**does** provide scanned core/geophysical logs of 90 named historical boreholes,
including 土庫 and 元長 (PDF pages 108 and 105 respectively). It was downloaded on
2026-10-06 from the linked public PDF into the ignored
`results/decision_evidence/public_sources/` directory; SHA-256 of the 180-page
appendix is `bb4460c446a3e780a82ddcfa5dbaf4cbe16f39eb89e1560cd634a67241e853e5`.
Those historical boreholes have different identifiers from the current catalog wells;
the scan labels its coordinates `TM` without a verified datum. Their contacts are
**measured at those borings only** and cannot yet be assigned to a ring location or
model cell. The separate WRA library
[well-construction report catalog](https://web.wra.gov.tw/wralib/WraLib/wSite/books_39283)
lists physical copies, but no downloadable site-specific completion log was found.
An official WRA
[geologic-log form example](https://wr.wra.gov.tw/GWMGAPI/Document/GWMGInfo/page3/01.%E5%9C%B0%E8%B3%AA%E6%9F%B1%E7%8B%80%E5%9C%96%28%E7%AF%84%E4%BE%8B%29.pdf)
shows such a record would include screen locations, but is a blank example, not a
site log. The next concrete request to WRA is the completion/log record for those
well identifiers plus the 16 ring installations; do not infer contacts from nearby
well depth, historical borehole labels or ring locations. This is a missing exact-site
source and coordinate-reference blocker, not an SSH authorization blocker.

### Observed-head information and bounded three-band gate

The ignored `results/decision_evidence/tuku_observed_head_information.json` records exact input hashes, 42 common covered months, observed correlations, and six conditional three-well-to-four-layer projections. The three WRA confining coefficients retain rank 3/3 and column-normalized condition about 1 because they act on separate installed NO2–NO24 depth bands. The IOT seven-zone proposal falls to rank 5–6/7; the 29-interval design to 17–22/29. This supersedes using the four-independent-driver synthetic basis test as a reason to fit seven free field coefficients. The measured head value unit/reference, exact screens, and physical WRA/IOT-to-pressure mapping are unverified; these ranks do not measure field coefficient precision or forecast skill. The [exact-source attempt table](SITE_GEOLOGY_SOURCE_AUDIT.md#exact-well-screen-and-head-reference-follow-up-2026-10-06) lists what was checked and the official WRA request route.

| Gate and status | Current evidence | Acceptance and dependency |
|---|---|---|
| Exact 土庫 pressure mapping: **blocked by records** | IDs `3132015GW09050321/31/41` have total depths 90/185/269 m and catalog layer codes 2/2/3, but no published exact screen tops/bottoms or cache water-level datum. The 2017 report lists sensor depths, not screen lengths; its older “water-level elevation” figure is suggestive but cannot certify these cache values. | Obtain per-ID dated completion/casing/screen and well-top survey history plus a water-level field dictionary from WRA. Pass only if each pressure series can be mapped to an independently justified modeled unit and signed, unit-correct head change. No material fit before this gate. |
| Three-band information candidate: **synthetic-only pass; field gate blocked** | WRA confining-band rank 3/3 under all six conditional observed-driver projections; no measured ring value was fitted. Seven-zone and 29-interval free fits are rank deficient. | Keep WRA and IOT structures separate. After mapping records, freeze NO2–NO24 ring intervals, pressure preprocessing, parameter bounds, baseline and temporal holdout before field fitting. Pass depth-matched held-out change and uncertainty tests against registered zero/site-rate baselines, not a fitted retrospective curve. |
| Viewer depth wording: **corrected in isolated worktree** | The 35-field station table exposes `WellDepth`, not screen interval. Viewer and explorer now label medians as total well depths and retain schematic/assumed-thickness caveats. | Recheck source identity and label every exported viewer/deck as schematic. A measured aquifer-boundary claim requires completion and geologic logs. |

**Runnable fast synthetic prototype specification.** In the hydro environment run `python -m hydrophysics.twin.synthetic_depth_identifiability` and examine `designs.WRA_3_confining_coefficients`. It uses the source-separated WRA 34–51, 171–193 and 246–260 m confining intervals, original installed NO2–NO24 rings, three coefficients, four invented independent basis paths and 12 synthetic correlated monthly head paths. The acceptance checks are rank 3/3, noiseless recovery error below 1e-15, ring-to-cumulative and whole-column sum errors below 1e-15 m, and an explicitly synthetic 1 mm-noise/50% relative halfwidth gate. `pytest -q tests/test_twin_synthetic_depth_identifiability.py` exercises these. This is an algebraic prototype only: no consolidation law, drainage time, measured ring fit, pressure datum or intervention attribution is supplied. The observed-head command in the site audit tests input-rank sensitivity separately.

**Bounded integration test selection.** Before launch, the observed-head, synthetic-depth, site/ring geometry, GNSS, decision-evidence, observation-boundary, policy-gate and viewer test files were selected from their test definitions; no training or calibration test file was included. `timeout 25s python -m pytest -q` on those nine files passed **84 tests in 4.28 s**. The flow/forward, optimization, full package, GPU, and known 800-epoch training stages were skipped. The earlier broad ad hoc selection that entered slow compute was stopped; no user job was stopped. Ruff and staged whitespace checks are separate gates.
