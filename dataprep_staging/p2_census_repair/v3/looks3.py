"""Look-log records of the P2abc repair round (A1 tool ``harness.prereg3``), each written
before the look it describes runs (or, for the verifier disclosure, as soon as this round
learned of it).

Usage: python -I looks3.py <tag>
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import looks_p2 as LK  # noqa: E402  (v2, read-only)
import p3lib  # noqa: E402,F401

E = LK.NONE_ELEC
SCOPE_FULL = "c0_pump_electricity_v3:2007-01..2026-07"

LOOKS = {
    "P2-r3-verifier-disclosure": dict(
        units={
            "P2r2:zero_port_yunlin": "Yunlin kept meters of C0 full v3 in the pilot view "
                                     "(round-2 zero check units)",
            "P2r2:july_units_yunlin": "Yunlin kept M2 phase-A meters, Julys 2010-2025",
            "P2r2:july_units_changhua": "Changhua kept M2 phase-A meters, Julys 2010-2025",
            "P2r2:july_units_chiayi": "Chiayi kept M2 phase-A meters, Julys 2010-2025",
            "P2r3:billing_pairs_2022_2026": "kept bimonthly and monthly meters of every county, "
                                            "billing pairs and monthly readings 2022-01..2026-07",
        },
        scope=SCOPE_FULL, purpose="diagnose",
        note="Disclosure by P2abc (repair round, 2026-10-14): the independent verifier of WP "
             "P2abc looked at the scored round-2 units after the round-2 report (08:20 CST) and "
             "before this repair round started (08:36 CST); the verifier did not log. It ran "
             "the shipped b2_zeros.classify on the pilot window and population (Yunlin "
             "2012-01..2025-06, kz meters): idle 1,224,444 vs pilot 1,231,196, long_idle "
             "382,664 vs 389,527, terminated 275,756 vs 257,073, trailing_short +0.87 %, "
             "irrigation in-span zero share about 6.1 % in April; recomputed the July backtest "
             "(15.374 / 8.697 / 42.193 %) and the plan's variant (recorded July, 2013-2024: "
             "13.99 / 8.29 / 39.1 %); measured the 2025 billing pattern (Yunlin M2 phase-A "
             "Jul+Aug 2025 8.54 GWh vs 11.05-11.74 in 2022-24, monthly meters 15.36 vs "
             "15.2-16.1, Sep+Oct 2025 10.99 vs 9.0-9.7, active phase-A meters with a zero "
             "Jul-Aug bill 0.164 vs 0.061-0.079; Changhua 11.00 vs 12.4-13.4) and the M0 "
             "flags at 2025-07/08; recomputed meter skill, sd coverage, head-check p values "
             "and the construction checks. These outcomes are known to this round."),
    "P2-r3-explore-2025": dict(
        units={"P2r3:billing_pairs_2022_2026": "kept bimonthly and monthly meters of every "
                                               "county, billing pairs and monthly readings "
                                               "2012-01..2026-07"},
        scope="c0_pump_electricity_v3:2012-01..2026-07", purpose="design",
        note="Design-time exploration for the P2b repair (full product only): per county, "
             "phase and billing pair 2012-2026, the pair-sum totals, the share of active "
             "meters (positive bills before and after) with a zero bill, the ratio of the "
             "following bill to the meter's own same-pair mean of the three previous years "
             "for zero-bill meters, and monthly-meter July/June and August/July ratios. Used "
             "to choose the 2025 masking rule, registered before the repaired backtest and "
             "construction checks are scored."),
    "P2-r3-explore-structural": dict(
        units={"P2r3:zero_structural_yunlin": "Yunlin kept meters: pre-install meter-months "
                                              "of C0 full v3 and of the billing pilot's "
                                              "archive"},
        scope="c0_pump_electricity_v3:2007-01..2025-07", purpose="design",
        note="Design-time check of feasibility only (no statistic): column names and the meter "
             "join key between the billing pilot's archive (meters_phase.parquet: inst, "
             "dropped) and C0 full v3 (meters_yunlin.parquet: install_month); counts are "
             "computed only by the registered P2r3 zero check."),
}


def main(tag: str) -> None:
    lk = LOOKS[tag]
    LK.record(tag, lk["units"], lk["scope"], lk["purpose"], lk["note"])


if __name__ == "__main__":
    main(sys.argv[1])
