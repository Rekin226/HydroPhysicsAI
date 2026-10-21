"""Selection rule and product gate of WP X1, fixed before the cross-validation was run.

This file was written before ``cv.py`` computed any score; its sha256 is recorded in the CV
output and in the product manifest. It is not entered in the harness pre-registration index
(the task forbids editing existing files, and the index is append-only); the hash is the
record.

Folds: forward chaining by calendar year. Test year Y = 2018..2026 (2026 = Jan-Jun); each
model trains on the recorded months before Y-01 (from 2012-01 for ``clim``, from 2016-01 for
the TPC-based models). Months the census does not record (2019-09..2020-06: < 99 % of active
cells with flags 0-5) are neither trained on nor scored.

Selection (primary model): among the TPC-based candidates, the lowest mean over the two
counties of the county RMSE ratio to ``clim`` pooled over the pre-origin test years
2018-2022. The post-origin test years 2023-2026H1 are reported as confirmation and choose
nothing.

sd: the multiplicative 1-sigma of a downscaled month is the RMS of the primary model's
out-of-fold log residuals, per county. For the gate it is calibrated on the 2018-2022 folds
and its 90 % interval coverage is checked on the 2023-2026H1 folds; the product's sd uses
every fold 2018-2026H1.

Product gate (all must hold):
  G1  per county, primary RMSE ratio to clim <= 0.90 on the 2018-2022 folds;
  G2  per county, primary RMSE ratio to clim < 1.00 on the 2023-2026H1 folds;
  G3  per county, 90 % interval coverage on the 2023-2026H1 folds: the Wilson 95 % interval
      of the coverage, with n = number of test years (months within a year are not
      independent), contains 0.90;
  G4  observed months of the product reproduce the census series to 1e-9 GWh;
  G5  every input hash is recorded and every TPC month used has a SOURCE.json archive.
Disc (3 km) series are reported, not gated.
"""
from __future__ import annotations

TEST_YEARS = tuple(range(2018, 2027))
PRE_YEARS = tuple(range(2018, 2023))
POST_YEARS = (2023, 2024, 2025, 2026)
CANDIDATES = ("ratio", "ridge", "ridge_wx", "gbm")
BASELINE = "clim"
G1_MAX_RATIO_PRE = 0.90
G2_MAX_RATIO_POST = 1.00
NOMINAL = 0.90
Z90 = 1.6448536269514722
