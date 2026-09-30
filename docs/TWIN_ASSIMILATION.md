# Groundwater observation updates and transient-response audit

The update experiment separates an aquifer state correction from a forecast correction
at monitoring wells. Neither establishes subsidence skill or validates an intervention.
Local results and a browser report live under `results/twin/update_experiment/`.

## What changed

`HeadObservationOperator` samples the same active cell and aquifer layer used during
calibration, then adds a fixed well offset. Its update interpolates the residual
between the observed and predicted well heads. Corrections stay within their observed
layer and decay spatially. Missing observations make no contribution. Wells sharing a
cell contribute their mean residual after removing their individual offsets.

For an observation operator H, interpolation I, fixed offset d, and localized gain G:

- Existing field reconstruction: `h_new = h + G * (I(y - d) - h)`.
- Residual update: `h_new = h + G * I(y - (Hh + d))`.

The second expression leaves the entire field unchanged when the observations are
predicted perfectly. The first generally does not: interpolation of sampled heads
need not reconstruct the original model field. The existing implementation also
applies the calibration's proximal layer-merging rule when reconstructing that field.

The real-grid fixed-point audit found that the old method moves well-cell heads by
0.161 m RMS and grid heads by up to 19.731 m despite zero observation error. Its net
change in modeled storage is about 12.48 million cubic metres. The residual update
produces exactly zero change in this test. The difference between the learned well
datum and the historical residual-mean scoring offset is only 0.056 m RMS.

There are 158 wells but 124 distinct cell/layer combinations, with at most two wells
per combination. A diagnostic oracle using current observations has an approximately
0.637 m RMSE floor under this fixed-cell, fixed-offset observation representation.
That oracle is not a forecast. It shows a representational limitation of the grid.

## API and verification

```python
operator = HeadObservationOperator.from_inputs(inputs, device=heads.device)
updated = operator.update(heads, observed_heads, fixed_offsets,
                          gain=1.0, radius_km=10.0)
```

`forward.nudge_to_observations(..., method="innovation", operator=operator)` exposes the
same opt-in behavior. The historical field method remains the default for reproducing
existing artifacts. Use exactly the same fixed offset during updating and scoring.
Do not estimate offsets from the evaluation interval.

The tests cover perfect-observation invariance, missing layers, localization, shared
cells, float32/float64, gain bounds, historical-only offset fitting, and causal
well-level residual forecasts. Predictions must be saved before the current month's
observation is assimilated. Real observation corrections alter modeled storage; their
increments must be kept separate from physical source and sink budgets.

## Selection and evaluation

The registered update screen compares gains 0.25, 0.5 and 1.0 and localization radii
2, 5 and 10 km, plus no update, the existing updater, and a datum-consistent existing
updater. Settings are selected on 2017–2020. Results for 2021–2022 are reported
separately. The original physics parameters had already been fitted on 2012–2022,
so this is conditional algorithm selection, not independent validation of physics.

The transient screen perturbs storage, transmissivity and vertical leakage separately.
The historical optimum at the upper storage boundary triggered a documented extension
to factors 8, 16 and 32, capped at storativity 0.3. Selected settings are frozen before
the 36-trajectory evaluation on January 2023–June 2025. The no-update and updated
variants are selected separately. No calibrated parameter file is overwritten.

A separate well-level correction is
`base[t] + retention * (observed[t-1] - base[t-1])`.
Its base blends the physical forecast and historical seasonal means; retention and
blend weight are selected using historical years only. It never changes aquifer heads
or supplies a subsidence driver. Missing previous observations fall back to the base.
Seasonal persistence is the special case of a seasonal base and retention one.

The longer replay through August 2026 uses climatological future forcing and previous
observations only. This avoids requiring the next month's actual pumping. All scores
use identical available observations within a comparison and equal well weights.
Previous-observation benchmarks restrict scores to months with the necessary preceding
observation. This conditional coverage is not evidence that every station is fresh.

Results remain retrospective: newer data were examined in earlier experiments. The
pulse tests are model sensitivity diagnostics, and the source-aligned weather and
inferred pumping remain uncertain. A future prospectively frozen evaluation and
independent compaction validation are still required.

## Results

The completed physical comparison uses 155 wells and 4,370 matched well-months:

| Method | RMSE (m) | Bias (m) |
|---|---:|---:|
| Existing field update | 1.963 | +0.701 |
| Existing update with consistent datum | 1.905 | +0.584 |
| Residual update | 1.723 | +0.062 |
| Residual update with bounded storage sensitivity | 1.600 | +0.005 |
| No update, recorded pumping | 1.840 | +0.238 |
| Previous-month persistence | 1.332 | +0.054 |
| Seasonal persistence | 0.855 | −0.018 |

One posterior parameter member exceeded the registered 0.3 storativity ceiling under
the 32× perturbation. Its affected history, offsets and two initial-state trajectories
were recomputed with that ceiling enforced. The historical master-member selection
was unchanged. The unbounded result is preserved as superseded diagnostic output.

The separate January 2023–August 2026 replay scores 157 wells and 5,908 common
well-months. Seasonal persistence scores 0.845 m versus 1.326 m for previous-month
persistence and 0.894 m for the hybrid under climatological future forcing. The hybrid's
small advantage with known pumping in the shorter period is not robust: a paired
three-month time-block interval for its RMSE difference from seasonal persistence is
approximately −0.109 to +0.103 m. This interval concerns aggregate retrospective skill,
not predictive coverage for an individual well.

The result supports the residual update as a correction to the observation mapping,
but does not justify a new physical calibration or policy deployment. The global
storage perturbation mainly changes retention of observations. The physical forecast
gate remains failed against the stronger short-term benchmarks.

## Reproduction

Use a new output directory for a new experiment. The registered input cache is the
locally produced, hash-checked cache from the forcing experiment; it is never shipped.
Run long stages in tmux as required by the project instructions.

```bash
python -m hydrophysics.twin.update_experiment register --out results/twin/update_experiment
python -m hydrophysics.twin.update_experiment select --device cuda
python -m hydrophysics.twin.update_experiment dynamics --device cuda
# Only when the initial historical optimum is storage4:
python -m hydrophysics.twin.update_experiment extend --device cuda
python -m hydrophysics.twin.update_experiment evaluate --device cuda
python -m hydrophysics.twin.well_correction select
python -m hydrophysics.twin.well_correction evaluate
python -m hydrophysics.twin.well_correction extended
```

Historical selection, confirmation, extension, source snapshots, pulse responses,
paired time-block bootstrap intervals, and newer-data predictions are recorded
separately. Numerical and protocol audits do not override a failed forecast gate.
