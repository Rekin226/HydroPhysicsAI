"""WP I1 hybrid transfer-function layer (design study, 2026-10-20).

A per-well transfer-function (TF) model of monthly heads on recorded drivers, alone or as a
correction of the physics model R0, prototyped on 2012-2022 only and drafted for a prospective
test on head months nobody has fetched. Modules:

* :mod:`.config`       paths, the frozen hyperparameters and the driver-availability lags;
* :mod:`.data`         guarded loading of the 2012-2022 wells, heads, R0 heads and drivers;
* :mod:`.model`        filters, ridge fits and the forecast families;
* :mod:`.inventory`    task 1, driver availability after 2022-12 (dates and coverage only);
* :mod:`.evaluate`     task 2, design-time skill by leave-one-year-out and forward chaining;
* :mod:`.declaration`  task 3, the DRAFT prospective declaration (never registered here).

Every number this package writes is design-time evidence or an inventory (plan 4.5); nothing
here is a gate, a ladder decision or a validation of its own choices (U7).
"""
from __future__ import annotations
