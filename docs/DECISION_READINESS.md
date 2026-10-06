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
