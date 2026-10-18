"""Look-log records of WP P2abc (A1 tool ``harness.prereg3``).

``seed_round1``: the interrupted round-1 session (2026-10-12, UTC+8 file times) scored four of
its five pre-registrations on the Yunlin AMP_V2 pipeline before the A1 tool existed and never
logged those looks. They are recorded here as back-dated ``design_time_seed`` looks (the only
back-dated purpose the tool accepts), with the actual computation times taken from the output
files' modification times, before any round-2 registration.

``record``: any other look (design looks, diagnostics) of this WP, logged before it runs.

Usage: python -I looks_p2.py seed_round1
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
from harness import prereg3  # noqa: E402
from harness.prereg2 import read_chained  # noqa: E402

NONE_ELEC = ("electricity-meter object of WP P2abc (TPC census kWh); contains no registered "
             "well, leveling benchmark or model cell of the validation-unit registries")


def _mtime_utc(p: Path) -> str:
    t = dt.datetime.fromtimestamp(os.path.getmtime(p), tz=dt.timezone.utc)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def logged_tags(wp: str = L.WP) -> set:
    out = set()
    for lk in read_chained(prereg3.PREREG_DIR / "looks.jsonl"):
        if lk.get("wp") == wp and lk.get("note", "").startswith("["):
            out.add(lk["note"].split("]")[0] + "]")
    return out


def record(tag: str, units: dict, scope: str, purpose: str, note: str, seeded_time=None):
    """units: {unit_id: members dict}; members {'none': reason} for electricity objects."""
    if f"[{tag}]" in logged_tags():
        print("already logged:", tag)
        return None
    members = {u: (m if isinstance(m, dict) else {"description": m, "none": NONE_ELEC})
               for u, m in units.items()}
    for u, m in members.items():
        m.setdefault("description", u)
    rec = prereg3.record_look(sorted(members), scope, L.WP, purpose, members=members,
                              seeded_time=seeded_time, note=f"[{tag}] {note}")
    print("logged", tag, rec.get("time"))
    return rec


def seed_round1() -> None:
    w = L.PRODUCT_ROOT / "_work_p2abc"
    pre = "results/dataprep/preregistration"
    items = [
        ("P2-r1-seed-portcheck", w / "portcheck.json",
         {"P2r1:pilot_port_check_yunlin_amp": "round-1 port check of the billing pilot (s08b, "
                                              "s11 skills on the pilot's own validation set)"},
         "amp_census_yunlin:2007-01..2025-07",
         "round-1 s0_portcheck.py reproduced the pilot's s08b/s11 skills (max |diff| 2.2e-16); "
         "the pilot's own design-time numbers."),
        ("P2-r1-seed-meter-validation", w / "validation_preorigin.json",
         {"P2r1:yunlin_amp_monthly_irrigation_meters": "round-1 validation meters (Yunlin "
                                                       "AMP_V2, 1784 kept monthly irrigation "
                                                       "meters)"},
         "amp_census_yunlin:2007-01..2025-07",
         f"round-1 s3_validate.py scored {pre}/P2a.meter_validation.v1.json and "
         f"P2a.sd_coverage.v1.json (registered 2026-10-12T02:35:58Z by round-1 prereg.py, "
         "outside the A1 index) on the Yunlin AMP_V2 pipeline: pre-origin skill min 0.229 over "
         "n 1..300 (passed), sd coverage 0.878-0.899 in 5 strata (round-1 rule: passed); "
         "placebo skills; December phase-B edge backtest; full variant. Results file "
         "_work_p2abc/validation_preorigin.json, validation_full.json, s3_validate.log."),
        ("P2-r1-seed-july", w / "july_backtest.json",
         {"P2r1:yunlin_amp_phaseA_julys_2013_2024": "round-1 July backtest units (Yunlin "
                                                    "AMP_V2 kept M2 phase-A meters)"},
         "amp_census_yunlin:2013-07..2024-07",
         f"round-1 s4_july.py scored {pre}/P2b.july_backtest.v1.json: mean fan-total error "
         "14.0 % (max 29.2 %), passed; imputation sd LOYO coverage; M0 July/June ratios "
         "2022-2025 and the July 2025 zero share by billing code (AMP_V2 cache, July 2025 "
         "partial)."),
        ("P2-r1-seed-zeros", w / "zeros.json",
         {"P2r1:yunlin_amp_zero_classification": "round-1 zero classification of Yunlin AMP_V2 "
                                                 "kept meters"},
         "amp_census_yunlin:2007-01..2025-07",
         f"round-1 s5_zeros.py scored {pre}/P2c.zero_reproduction.v1.json: pilot reproduction "
         "exact (run table max rel diff 0; pre-install zeros 878738; irrigation April 0.0558, "
         "August 0.1563), passed; product classes and synchronized-zero months per variant."),
    ]
    for tag, path, units, scope, note in items:
        record(tag, units, scope, "design_time_seed",
               note + f" Computed at {_mtime_utc(path)} (file mtime) by the interrupted "
               "round-1 session; logged retroactively in round 2 before any round-2 "
               "registration.", seeded_time=_mtime_utc(path))


if __name__ == "__main__":
    if sys.argv[1:] == ["seed_round1"]:
        seed_round1()
    else:
        raise SystemExit(__doc__)
