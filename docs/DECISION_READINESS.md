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

Reproduce the saved diagnostics, writing only to an ignored local directory:

```bash
python -m hydrophysics.twin.evidence \
  --challenge-dir /path/to/existing/results/twin/new_data_challenge \
  --out results/decision_evidence
python -m hydrophysics.twin.scenario_evidence \
  --sidecar /path/to/existing/results/twin_forward/datum_gate.members.npz \
            /path/to/existing/results/twin_forward/physical_spread_apex.members.npz \
  --out results/decision_evidence/scenarios.json
```

`site_scores.csv` contains observation-derived per-well results and stays local and
ignored. The JSON files record hashes of the saved inputs; copying a score to a public
slide without its model, period, metric and caveat would be misleading.

## Readiness gates and next work

1. **Validate the compaction mechanism (highest impact; high effort).** The current
   selected model has poor 14-ring fit and the two models disagree on rebound versus
   reduced creep. Obtain independent, dated leveling/ring or InSAR observations spanning
   recovery or a documented intervention, with vertical reference and measurement
   uncertainty. Lock the calibration and validation split before refitting. **Pass:**
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
