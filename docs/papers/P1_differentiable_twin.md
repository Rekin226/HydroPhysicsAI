# P1: A differentiable GPU digital twin of an aquifer-subsidence system

**Status:** planned (main paper). **Last updated:** 2026-10-02.

## Target venues (in order)

1. **Geoscientific Model Development** (EGU, open access): model-description and
   evaluation papers; requires public code with a DOI and a data-availability statement.
2. **JAMES** (Journal of Advances in Modeling Earth Systems): differentiable and hybrid
   models.
3. **Computers & Geosciences**: methods with code.

## Working title

A differentiable, GPU-calibrated digital twin of groundwater and land subsidence in the
Choushui alluvial fan, Taiwan, and the tests that decide whether it can be trusted for policy

## Contribution

1. A four-layer finite-volume flow solver in float64, matrix-free conjugate gradient on the
   GPU, gradients by implicit-function adjoint (memory does not grow with the rollout),
   coupled to a visco-elasto-plastic compaction column, calibrated end to end by gradient
   descent. Forcing from a cleaned 116,769-pump electricity census and rain minus ET0.
2. A testing method, with evidence that the usual test misleads:
   - The site-grouped absolute k-fold against IDW passed (R2 +0.804 vs +0.702) a model whose
     head changes at unseen wells were badly wrong (anomaly R2 -1.43 vs IDW +0.61).
   - A per-well datum in the observation operator (sub-grid levels: 30 m vertical head
     differences inside single well nests) gives the first model to pass a fair held-out-years
     test (ratio 1.11, shape R2 +0.45 vs climatology +0.36) and raises leveling skill
     (+0.589 -> +0.652 out of fold).
   - Half of a published projection (10.7 cm) was artefact: a compaction start-up release
     invisible to re-zeroed leveling, and an origin-restart step.
3. Policy result: irrigation -30 % avoids 0.94 cm and retiring aquaculture 1.31 cm of
   fan-mean subsidence by 2032 (baseline 2.9 cm); robust to the creep ceiling (0.3 cm),
   but the timing (rebound vs slowed creep) is not testable on the 2012-2022 record.

## Evidence in the repo

- Verdict history and every number: `docs/superpowers/STATE.md` sections 0-3.
- Deliverable: `results/twin_runs/stage3_datum_gate/` (stage3_flow.csv, stage3_kfold_wells.csv,
  coupled_leveling/), projection `results/twin_forward/datum_gate.*`.
- Previous model for comparison: `results/twin_runs/stage3_spreadL_gate/`,
  `results/twin_forward/physical_spread_apex.*`.
- Fair temporal screen: `results/twin_runs/temporal_ref10_ic_ge_split208_tmin58_datum_sd5/`.
- Tools: `hydrophysics/twin/{flow,calibrate_flow,kfold_scores,drift_diag,rescore_temporal,
  forward,calibrate_coupled,mechanism,residual_kriging}.py`.
- Negative results worth a section or supplement: slow storage, canal water v2/v3, rivers,
  layered proximal aquifer, banded column, regression kriging (all in STATE.md).

## Figures to make

1. Fan map: wells, leveling, rings, zones, THSR (basemap.npz).
2. Solver and calibration diagram (adjoint, column coupling).
3. Absolute vs anomaly k-fold, per well, previous vs datum model.
4. Held-out-years shapes 2020-2022, model vs climatology.
5. Leveling hindcast scatter and maps.
6. Policy projections with ensemble and creep-ceiling bands; rebound-vs-slowed-creep panel.

## Gaps before submission

- [ ] Re-measure CPU/GPU cost (see README gap 1); state calibration wall time honestly.
- [ ] Decide whether the 2023-2026 out-of-sample test (new WiseEnvr pull, separate cache)
      goes in; it would be the strongest validation.
- [ ] Code release tag + Zenodo DOI; data-availability text (no redistribution).
- [ ] Literature: differentiable hydrology (dPL / Shen et al.), GPU groundwater (MODFLOW 6 GPU
      efforts, ParFlow GPU), Choushui subsidence (`docs/LIT_SUBSIDENCE_TAIWAN.md`).
- [ ] Draft with the `paper-agent` skill once the journal is fixed.
