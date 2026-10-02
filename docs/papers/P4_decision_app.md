# P4 (optional): Showing scenario impact of a groundwater twin to decision makers

**Status:** optional. **Last updated:** 2026-10-02.

## Target venues

1. **Environmental Modelling & Software** (decision-support and visualization papers).
2. **SoftwareX** (short software paper).

## Contribution

- A decision page designed from the visualization literature (spec with citations:
  `docs/superpowers/specs/2026-09-23-twin-decision-app-redesign.md`): impact strip first,
  linked 2D map and time series, difference / side-by-side / swipe comparison, per-run
  agreement hatching, guided story, township cards, rail profile; the 3D block as a drawer.
- Every verdict and caveat on the page is computed from run files, and caveats sit next to
  the numbers they qualify (including tests the model fails).
- One self-contained file, 1.6 MB (the earlier 3D-first viewer was 10.7 MB).

## Evidence in the repo

- `hydrophysics/twin/viewer_app.py`, `hydrophysics/twin/app/`, `results/twin/twin_app.html`,
  `tests/test_viewer_app.py`.

## Gaps before submission

- [ ] A user evaluation (even a small one with water-agency staff or students): EMS will
      expect evidence that the design helps people decide.
- [ ] Screenshots and a short video; an accessibility check (`design:accessibility-review`).
