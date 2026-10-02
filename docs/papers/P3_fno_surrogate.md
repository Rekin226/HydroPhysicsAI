# P3: A PhysicsNeMo FNO surrogate for pumping-policy ensembles

**Status:** planned; blocked on retraining the surrogate on the current model.
**Last updated:** 2026-10-02.

## Target venues (in order)

1. **NeurIPS workshops**: Machine Learning and the Physical Sciences (ML4PS), or Tackling
   Climate Change with ML.
2. **AMS Artificial Intelligence for the Earth Systems** (full paper).
3. Merge into P2 as a section if it stays short.

## Working title

A Fourier neural operator surrogate of a calibrated, differentiable aquifer model for fast
pumping-policy sweeps

## Contribution

- The surrogate is trained on a calibrated physics model, not on the record, so it inherits
  the physics' tested behaviour; error is measured against the solver under random policies.
- Earlier result (old deliverable): 256 solver rollouts x 24 months, one-step validation
  rel-L2 about 0.015-0.024, 24-month rollout within 0.03-0.08 m RMSE of the solver, 9-34x
  faster than the GPU solver per scenario.
- Use case: dense policy sweeps and large ensembles for the decision app's sliders.

## Evidence in the repo

- `hydrophysics/twin/surrogate.py`; trained models in `results/surrogate/`
  (`physical_spread_fno` = previous deliverable; `final_fno` = a rejected model).
- Response basis used by the app instead: `results/twin_forward/response_basis_datum.npz`.

## Gaps before submission

- [ ] Retrain on `stage3_datum_gate` (surrogate must match the deliverable). Check the
      surrogate's inputs cover the deliverable's options (merged proximal IC, DEM ground,
      208 km split; no delay store or canal water, so no new input channels are needed).
- [ ] Long-rollout stability (120 months, the projection horizon), not only 24.
- [ ] Compare against the cheaper response-basis superposition the app already uses
      (within 8.7 % / 3.1 % of solved runs): the surrogate must beat it to be worth a paper.
- [ ] Uncertainty: surrogate ensemble over the 18 parameter sets vs solver ensemble.
