"""WP C0 repair round: look-log records (A1 tool ``harness.prereg3``).

``python -I repair_looks.py seeds``   back-dated design-time seeds: the verifier's findings
                                      (2026-10-13) that this session read before any work
``python -I repair_looks.py design``  the pre-origin design look (repair_design.py), logged
                                      before it runs
``python -I repair_looks.py <key>``   other looks listed in LOOKS, each logged before it runs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from harness import prereg3  # noqa: E402

NONE = {"none": "C0 electricity-forcing construction object; contains no registered well, "
                "benchmark or model cell of the validation-unit registries",
        "description": ""}


def members(units: dict) -> dict:
    return {u: {**NONE, "description": d} for u, d in units.items()}


SEEDS = [
    ("2026-10-13T07:38:00Z",
     {"C0r:verifier_findings_2026-10-13":
      "the independent verifier's C0 findings as read by this session: 99.7 % of M1 meters "
      "have Dec-2022 == Jan-2023 in L0; phase-B in-grid 2022-12 energy 3.78 (Yunlin) + 4.55 "
      "(Changhua) + 0.018 (Chiayi) GWh of 28.67 GWh; M1 pair 2021-12/2022-01 equal in only "
      "3.5 % (Yunlin) / 6.2 % (Changhua) of meter pairs against >= 99.6 % in every other "
      "year 2016-2024; billing-code agreement bimonthly 0.959-0.978 (Yunlin/Changhua), "
      "0.909-0.912 (Chiayi), monthly 0.926-0.938; independent rebuild reproduces v1 exactly"},
     "tpc_census_kwh:2016-01..2024-12",
     "verifier report (relayed in the repair task) read at the start of the repair session; "
     "no computation by this session"),
]

LOOKS = {
    "design": (
        {"C0r:design_phase_crosstab_preorigin":
         "billing code x detected phase (2012-2022) cross-tab of every meter, 3 counties",
         "C0r:design_duty_flips_preorigin":
         "kept-set flips when the Dec-2022 value of Dec-Jan-paired meters leaves the duty "
         "statistic (two candidate rules); counts of missing Oct-Nov 2022 fill bases"},
        "tpc_census_kwh:2007-01..2022-12", "design_time_check",
        "repair_design.py: pre-origin L0 only (truncated to <= 2022-12 at read); counts only, "
        "no fill value, ratio or reference comparison; output "
        "results/dataprep/reports/C0_evidence/repair/design_look.json"),
    "build": (
        {"C0r:design_install_dates_census":
         "valid census install months after 2016-12 and 2022-10 per county (v1 L1 table), "
         "read while debugging the v2 build (a Chiayi paired meter not yet installed in "
         "2022-12)",
         "C0r:design_build_counts_v2":
         "kept pumps, kept in grid and kept meters per county and variant printed by the v2 "
         "builder (smoke and full builds)"},
        "tpc_census_attributes:2007-01..2026-07", "design_time_check",
        "logged after the fact, before the repair pre-registration: Yunlin 85 / Changhua 79 "
        "pumps installed after 2016-12 (45 / 35 after 2022-10; latest 2026-06 / 2025-12; "
        "Chiayi in-grid meters none); v2 kept pumps/meters Yunlin 114,393/105,210, Changhua "
        "115,760/105,068, Chiayi 26,056/24,802 (316 in grid); no fill value, ratio or "
        "comparison was printed or read"),
    "smoke": (
        {"C0r:v2_structural_smoke":
         "array shapes and counts of finite sd / n_fill > 0 entries in the assembled v2 "
         "products (no value, no comparison)"},
        "c0_products_v2:2007-01..2026-07", "design_time_check",
        "structural smoke of the assembled v2 products before the repair pre-registration: "
        "shapes, finite-sd and fill counts only"),
}


def main(argv: list[str]) -> int:
    for key in argv:
        if key == "seeds":
            for t, units, scope, note in SEEDS:
                prereg3.record_look(sorted(units), scope, lib.WP, prereg3.SEED_PURPOSE,
                                    members=members(units), seeded_time=t,
                                    note="[C0-repair-seed] " + note)
            print("logged", len(SEEDS), "seed(s)")
        else:
            units, scope, purpose, note = LOOKS[key]
            prereg3.record_look(sorted(units), scope, lib.WP, purpose, members=members(units),
                                note=f"[C0-repair-{key}] " + note)
            print("logged look", key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
