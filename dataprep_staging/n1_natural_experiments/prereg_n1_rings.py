"""Declare the N1 ring-compaction check (harness.prereg3, kind 'declaration').

The plan's N1 step 2 names 'wells and rings near them'. Rings 2012-2021 are design-time data
for the compaction column (they served as out-of-fold validators when the column's calibration
on leveling was chosen; project memory 'calibrate the column on leveling, per zone'), so a
ring check of the event response is declared here with a design-time label, never as a skill
test. It runs in Phase 2 with the I1 primary model's column; no ring value is read now.

Usage: python -I prereg_n1_rings.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import rel_to_repo  # noqa: E402
from harness import prereg3  # noqa: E402

from n1_natural_experiments import n1const as K  # noqa: E402

NA = prereg3.not_applicable


def doc() -> dict:
    return {
        "id": "N1.rings_designtime.v1", "wp": K.WP, "kind": "declaration",
        "title": "N1 ring-compaction check of the 2015, 2020 and 2021 spring events "
                 "(design-time label; Phase 2)",
        "tests": NA("declaration: design-time evidence, never validation"),
        "checks": {
            "C5_rings_events": (
                "For each pre-gap compaction-ring site (the 14 sites of S1a; ring record "
                "2012-2021 as read by the ring_holdout convention, original instrument, no "
                "S1 bridge), the compaction of each depth interval over the survey interval "
                "containing Jan-May of 2015, 2020 and 2021 minus the mean of the same "
                "interval in the reference years 2013, 2014, 2016-2019; model: the I1 "
                "primary model's compaction column (leveling-calibrated) at the ring's cell, "
                "driven by (a) its hindcast heads and (b) the masked-refit heads of G1/C3; "
                "baselines ZR (the column driven by the ZR head path) and CL (zero anomaly). "
                "Metric as n1lib (per-site error, equal-site RMSE, paired site bootstrap, "
                "90 %); with 14 sites the coverage of any interval is uncalibrated (A1). "
                "Label: design-time evidence (rings validated the column's design)."),
            "data_rule": "no ring value after 2021 is read (the 2022 hiatus and the 2023+ "
                         "instrument are excluded; U7)",
            "coordination": "ring loading goes through the ring_holdout session's convention "
                            "(plan 5.5); hydrophysics/twin/ring_holdout.py is not touched"},
        "zones": NA("the column's own"), "priors": NA("none"), "burn_in": "2012 excluded",
        "folds": NA("none"), "fresh_units": NA("no fresh units: design-time check"),
        "thresholds": NA("reported with its label"), "gate_scores": NA("no gate"),
        "primary_model": NA("the I1 primary model's column, when it exists"),
        "inputs": [rel_to_repo(K.CODE_DIR / "n1lib.py"), rel_to_repo(K.CODE_DIR / "n1const.py")],
        "interpretation": "context for the policy response of the column; a ring pass or fail "
                          "is never quoted as skill",
        "no_prior_look_statement": "WP N1 has read no ring value.",
        "written_by": rel_to_repo(Path(__file__)),
    }


def main() -> int:
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit(errs)
    path, sha = prereg3.write_prereg(d)
    print(rel_to_repo(path), sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
