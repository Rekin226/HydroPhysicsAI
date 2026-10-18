"""WP C0: log, in the A1 look log, every look the earlier (interrupted) C0 attempts and this
session took before the pre-registration. Run once, before ``prereg_c0.py``.

The earlier attempts' scratch scripts and outputs are archived in
results/dataprep/reports/C0_evidence/design_time/ (scratch_attempt1 = 06:32-06:53Z,
scratch_attempt2 = 11:11-11:13Z, attempt1_build_chiayi_preorigin = the 06:53Z smoke build).
Times are the file modification times (UTC). The units named here are the objects looked at
(L0 profiles, the model-forcing reference summary, attribute comparisons, smoke builds); the
pre-registered tests score comparisons of the delivered C0 products, which did not exist when
these looks were taken.

Usage: python -I looks_c0.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from harness import prereg2  # noqa: E402

EV = "results/dataprep/reports/C0_evidence/design_time"
SEEDS = [
    ("2026-10-12T06:32:52Z", ["C0:L0_profile_chiayi", "C0:county_set_L0"],
     "tpc_census_chiayi:2007-01..2026-07",
     f"{EV}/scratch_attempt1/explore_chiayi.py: Chiayi census attributes, pumps in active "
     "cells (320), their purposes, classes, billing codes and towns; meter-column checks; raw "
     "(uncleaned) in-grid kWh by year."),
    ("2026-10-12T06:35:13Z", ["C0:model_forcing_reference_summary"],
     "model_forcing_E_by_class:2012-01..2022-12",
     f"{EV}/scratch_attempt1/peek_pkl.py: results/twin_runs/stage3_datum_gate_cwa/"
     "inputs.local.pkl sha256, its dates, grid parameters, mask equality with "
     "datum_gate_innov.npz, and the 2012-2022 fan total of each class of E_by_class (6 "
     "numbers). No C0 output existed; no comparison was made."),
    ("2026-10-12T06:36:11Z", ["C0:L0_profile_yunlin", "C0:L0_profile_changhua",
                              "C0:L0_profile_chiayi"],
     "tpc_census_kwh:2007-01..2026-07",
     f"{EV}/scratch_attempt1/explore_flags.py (+ .json): NaN, zero and pre-install counts, "
     "install-date parse failures, billing-code by month zero/NaN fractions and raw GWh for "
     "7 months (2024-07..2026-07), meters with mixed billing codes."),
    ("2026-10-12T06:39:13Z", ["C0:L0_profile_yunlin", "C0:L0_profile_changhua"],
     "tpc_census_kwh:2012-01..2026-07",
     f"{EV}/scratch_attempt1/explore_pairs.py (+ .json): per pair-start month, the fraction "
     "of identical, first-zero and second-zero bimonthly pairs of first-pump M2/M1 meters."),
    ("2026-10-12T06:40:05Z", ["C0:L0_profile_yunlin", "C0:L0_profile_changhua"],
     "tpc_census_kwh:2007-01..2026-07",
     f"{EV}/scratch_attempt1/explore_nan.py: months with the most post-install NaN, NaN run "
     "structure (leading / interior / trailing), NaN on first-pump meters."),
    ("2026-10-12T06:53:09Z", ["C0:build_smoke_chiayi_preorigin"],
     "c0_build_chiayi:2007-01..2022-12",
     f"{EV}/attempt1_build_chiayi_preorigin/: first build.py run (Chiayi pre-origin "
     "intermediates); its info json (kept counts, method counts, allocation identity, phase "
     "agreement, clean report) was read in the resume session; the full variant crashed on "
     "the pump-level NaN <=> flag-8 invariant (method-5 override bug, fixed)."),
    ("2026-10-12T11:12:39Z", ["C0:L0_profile_yunlin", "C0:L0_profile_changhua",
                              "C0:L0_profile_chiayi", "C0:coding_sensitivity_kept_set",
                              "C0:yunlin_attr_amp_vs_new", "C0:county_set_L0"],
     "tpc_census_kwh:2007-01..2026-07",
     f"{EV}/scratch_attempt2/explore_coding.py (+ .json, .log): kept meters and kept "
     "de-duplicated 2012-2022 kWh per county under the delivered coding and two alternative "
     "pre-install codings; dense kept set == clean_census kept set; billing-code counts; "
     "classes in grid; Chiayi purposes and shared meters; AMP_V2 vs new Yunlin attribute "
     "equality counts; Pingtung pumps in active cells (0)."),
    ("2026-10-12T11:13:14Z", ["C0:yunlin_attr_amp_vs_new"],
     "tpc_census_yunlin_attributes",
     f"{EV}/scratch_attempt2/explore_amp_attr.py: PURPOSE / coordinate differences between "
     "AMP_V2 and the new Yunlin sheet and whether they change class or cell (output printed; "
     "rerun in the resume session, see below)."),
]
NOW = [
    (["C0:yunlin_attr_amp_vs_new"], "tpc_census_yunlin_attributes", "design_time_check",
     "resume session 15:49Z: rerun of explore_amp_attr.py: 69 PURPOSE differ only as None vs "
     "'NULL' (class 'other' both), 84 coordinate differences are NaN in both, 0 class and 0 "
     "cell differences."),
    (["C0:build_smoke_chiayi_preorigin"], "c0_build_chiayi:2007-01..2026-07",
     "design_time_check",
     "resume session 15:50-15:54Z: build.run('chiayi') smoke runs into the scratchpad; the "
     "first reproduced the full-variant crash, the second (after the method-5 fix) ran both "
     "variants; only kept-pump counts and run time were printed."),
]


def main() -> int:
    for t, units, scope, note in SEEDS:
        prereg2.record_look(units, scope, lib.WP, prereg2.SEED_PURPOSE, seeded_time=t,
                            note="[C0-seed] " + note)
    for units, scope, purpose, note in NOW:
        prereg2.record_look(units, scope, lib.WP, purpose, note="[C0-look] " + note)
    print("logged", len(SEEDS), "seeds and", len(NOW), "looks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
