"""Changhua step-0 count-only look on ls_cache 2008-01..2009-12 (round 8 plan #18: closure of
the last unverified Changhua head window). A second reviewer checks this script before it runs.

Two steps, in order, both run by the lead:

``look --validate`` / ``look --write``
    prints / logs (``prereg3.record_look``, purpose ``availability_check``) the look on unit
    I1CHB0:ls_cache_rowcounts_2008_2009 whose members are the fan-zone (zone 50) Changhua
    monitoring wells that have an ``ls_cache/gw__<sid>__raw.parquet`` file (file names and the
    station table only), window 2008-01..2009-12, ids only (never universal). Both run the
    prereg3 blocking probe (looks.jsonl metadata only); ``--write`` puts the stop rule text and
    the sha256 of the stop-rule constants (``constants()``) in the note, so the look scope is
    in looks.jsonl before any count is read. stdout carries the unit, the member count, the
    scope, the constants hash and the probe hits (unit ids and counts), never a member
    station id.
``count``
    refuses unless that look is logged once, its note carries this file's constants hash and
    rule text, and today's candidate list equals the logged members; refuses if the pinned
    output ``OUT`` exists. Then, per member, reads ONLY the ``datetime`` column
    (``pyarrow.parquet.read_schema`` then ``read_table(columns=['datetime'])``); the
    ``value`` column is never decoded or returned (the file is hashed whole for provenance
    and its footer schema is parsed). It lists the 2008-01..2009-12 months whose sampling
    could make a monthly value (>= 20 days with >= 12 distinct hours;
    ``bc_lib.available_months``) and applies the stop rule below. Writes ``OUT`` once (mode
    ``x``); prints only counts and PASS / STOP.

Stop rule (fixed here, before the look; constants below, hashed into the look note):

* PASS iff >= 10 Changhua wells have >= 18 available months within 2008-01..2009-12 in the
  gating count;
* ring-nest wells (within 1000 m of a ring site: mechanism._long_head read their whole
  ls_cache files with no date clip, so their 2008-09 rows may have been read) stay look
  members; with ``EXCLUDE_NEST`` (True, lead decision) they are excluded from the GATING
  count, and the count with them is reported beside it, report-only;
* on PASS: continue only after a ruling on L1_pre2012 v2 coverage of 2008-09, reusing N2CS's
  frozen estimator (any later backcast G0: power >= 0.6, false pass <= 0.2);
* on STOP: record 'last possible Changhua head window checked, not decidable'.

Disclosure: P6's inventory_read look on P6:moea_public_well_tables (scope
moea_public_well_tables:2008-01..2024-12, 2026-10-14) may overlap if those public tables
carry Changhua 2008-09 heads; its membership is not comparable by the probe.

    python dataprep_staging/ch_x/avail_chb0809.py look --validate
    python dataprep_staging/ch_x/avail_chb0809.py look --write
    python dataprep_staging/ch_x/avail_chb0809.py count
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import avail_l1pre as AV  # noqa: E402
import avail_lib as AL  # noqa: E402
import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402

# ----------------------------------------------------------------------------- frozen design
AVAIL_ID = "I1.CH_backcast_2008_09.availability.v1"
UNIT = "I1CHB0:ls_cache_rowcounts_2008_2009"
WINDOW = ("2008-01", "2009-12")
SCOPE = f"heads:{WINDOW[0]}..{WINDOW[1]}"
WP = K.WP
PURPOSE = AL.PURPOSE
COUNTIES = ("changhua",)
OUT = "results/twin_runs/chx/backcast_2008_09/avail/avail.json"   # pinned, written once

MIN_WELLS = 10                             # PASS iff >= 10 wells ...
MIN_MONTHS = 18                            # ... with >= 18 available months in the window
EXCLUDE_NEST = True                        # nest wells out of the gating count (lead decision)
NEST_RADIUS_M = K.BC_NEST_RADIUS_M
STOP_LINE = "last possible Changhua head window checked, not decidable"
P6_LOOK = ("P6:moea_public_well_tables", "moea_public_well_tables:2008-01..2024-12",
           "2026-10-14")


def _nest_phrase() -> str:
    if EXCLUDE_NEST:
        return (f"ring-nest wells (<= {NEST_RADIUS_M:.0f} m of a ring site) stay look members "
                "but are excluded from the gating count (the count with them is reported "
                "beside it, report-only)")
    return (f"ring-nest wells (<= {NEST_RADIUS_M:.0f} m of a ring site) are counted (the "
            "count without them is reported beside it, report-only)")


RULE_TEXT = (
    f"stop rule fixed in code before the look: PASS iff >= {MIN_WELLS} Changhua wells have "
    f">= {MIN_MONTHS} available months within {WINDOW[0]}..{WINDOW[1]}; {_nest_phrase()}; a "
    f"month is available with >= {K.BC_MONTH_MIN_DAYS} days of >= {K.BC_DAY_MIN_HOURS} "
    "distinct hours (datetime only); on PASS continue only after a ruling on L1_pre2012 v2 "
    f"coverage with N2CS's frozen estimator; on STOP record '{STOP_LINE}'")

DISCLOSURE_TEXT = (
    "disclosed: mechanism._long_head read the whole ls_cache files of the wells near ring "
    "sites (no date clip), so their 2008-09 rows may have been read; P6's inventory_read look "
    f"on {P6_LOOK[0]} (scope {P6_LOOK[1]}, {P6_LOOK[2]}) may overlap if those public tables "
    "carry Changhua 2008-09 heads")


def constants() -> dict:
    """Every stop-rule constant; its sha256 goes into the look note."""
    return {"id": AVAIL_ID, "unit": UNIT, "scope": SCOPE, "window": list(WINDOW),
            "counties": list(COUNTIES), "min_wells": MIN_WELLS, "min_months": MIN_MONTHS,
            "exclude_nest": EXCLUDE_NEST, "nest_radius_m": NEST_RADIUS_M,
            "day_min_hours": K.BC_DAY_MIN_HOURS, "month_min_days": K.BC_MONTH_MIN_DAYS,
            "out": OUT, "bounds": "inclusive"}


def stop_rule(months_by_well: dict, county_of: dict, nest: set) -> dict:
    """The fixed stop rule on availability counts (pure; see the module docstring). Both the
    nest-excluded and the nest-included counts are computed from the same months; the one
    ``EXCLUDE_NEST`` names gates, the other is report-only."""
    pool = sorted(s for s in months_by_well if county_of.get(s) in COUNTIES)
    nest = set(nest) & set(pool)
    n = {s: AL.n_in(months_by_well[s], *WINDOW) for s in pool}
    ok_incl = [s for s in pool if n[s] >= MIN_MONTHS]
    ok_excl = [s for s in ok_incl if s not in nest]
    gate, other = (ok_excl, ok_incl) if EXCLUDE_NEST else (ok_incl, ok_excl)
    return {"n_changhua_candidates": len(pool), "n_nest": len(nest),
            "gating": "nest_excluded" if EXCLUDE_NEST else "nest_included",
            "nest_excluded": {"wells": ok_excl, "n": len(ok_excl)},
            "nest_included": {"wells": ok_incl, "n": len(ok_incl)},
            "eligible_wells": gate, "n_eligible": len(gate),
            "report_only_other_count": {"which": ("nest_included" if EXCLUDE_NEST
                                                  else "nest_excluded"), "n": len(other)},
            "passed": len(gate) >= MIN_WELLS, "n_months_by_well": n,
            "constants": constants(), "constants_sha256": AL.constants_sha(constants()),
            "on_stop": STOP_LINE}


def candidates(st=None) -> list[str]:
    return AL.candidates(COUNTIES, st)


def members(ids) -> dict:
    return {"description": "fan-zone (zone 50) Changhua monitoring wells with an ls_cache "
                           "gw__<sid>__raw file: per-month sampling counts "
                           f"{WINDOW[0]}..{WINDOW[1]} from the datetime column only (no "
                           "value); round-8 plan #18 count-only closure look",
            "ids": sorted(map(str, ids)), "window": f"{WINDOW[0]}..{WINDOW[1]}"}


def look_record(ids, prereg_dir=None, registries=None) -> dict:
    m = members(ids)
    ov = AL.prior_overlaps(UNIT, m, SCOPE, prereg_dir, registries)
    note = (f"[{AVAIL_ID}] {AL.sha_tag(constants())}; metadata-only count look (datetime "
            "column only; the value column is never decoded or returned); "
            f"{RULE_TEXT}. {DISCLOSURE_TEXT}. prereg3 probe, earlier looks sharing members in "
            f"{WINDOW[0]}..{WINDOW[1]}: {AL.overlap_summary(ov)}")
    return {"units": [UNIT], "data_scope": SCOPE, "wp": WP, "purpose": PURPOSE,
            "members": {UNIT: m}, "note": note, "probe": ov}


def look(write: bool, prereg_dir=None, registries=None) -> int:
    from harness import prereg3

    rec = look_record(candidates(), prereg_dir, registries)
    m = rec["members"][UNIT]
    errs = prereg3.member_errors(m)
    logged = AL.logged_look(UNIT, prereg_dir) is not None
    print(f"unit {UNIT} | {len(m['ids'])} members | scope {SCOPE} | purpose {PURPOSE}")
    print(AL.sha_tag(constants()))
    print("probe hits (unit | purpose | date | membership | shared members):")
    for ln in AL.probe_lines(rec["probe"]):
        print("  " + ln)
    print("member errors:", errs or "none", "| logged already:", logged)
    if not write:
        return 1 if errs else 0
    if errs:
        raise SystemExit("REFUSED: member errors")
    if logged:
        raise SystemExit("REFUSED: the availability look is already logged")
    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir}
    r = prereg3.record_look(rec["units"], rec["data_scope"], rec["wp"], rec["purpose"],
                            members=rec["members"], note=rec["note"], registries=registries,
                            **kw)
    print(f"logged look at {r['time']}")
    return 0


def count(prereg_dir=None) -> dict:
    lk = AL.logged_look(UNIT, prereg_dir)
    if lk is None:
        raise SystemExit("REFUSED: log the availability look first (look --write)")
    if not AL.note_matches(lk, constants(), RULE_TEXT):
        raise SystemExit("REFUSED: the logged look's note does not carry this file's "
                         "constants hash and rule text")
    target = IO.rp(OUT)
    if target.exists():
        raise SystemExit(f"{OUT} exists; the availability look runs once")
    st = AV.stations()
    ids = AL.logged_ids(lk, UNIT)
    if ids != candidates(st):
        raise SystemExit("REFUSED: today's candidate list differs from the logged members")
    months = AL.months_by_well(ids, WINDOW)
    nest = AL.nest_excluded(ids, st, NEST_RADIUS_M)
    feas = stop_rule(months, AL.county_of(st), nest)
    rep = {"what": f"{AVAIL_ID} (timestamps only)", "unit": UNIT, "look_time": lk["time"],
           "months_by_well": months, "nest_wells": sorted(set(ids) & nest),
           "feasibility": feas, **AL.provenance(ids)}
    AL.write_once(target, rep)
    print(summary(feas))
    return rep


def summary(feas: dict) -> str:
    o = feas["report_only_other_count"]
    return (f"availability 2008-09: gating count ({feas['gating']}) {feas['n_eligible']} "
            f"eligible Changhua wells of {feas['n_changhua_candidates']} (>= {MIN_MONTHS} "
            f"months; {feas['n_nest']} nest wells; need >= {MIN_WELLS}) -> "
            f"{'PASS' if feas['passed'] else 'STOP (' + STOP_LINE + ')'}; report-only "
            f"{o['which']} count {o['n']}")


def main(argv=None, prereg_dir=None, registries=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("look")
    g = lk.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    sub.add_parser("count")
    a = ap.parse_args(argv)
    if a.cmd == "count":
        count(prereg_dir)
        return 0
    return look(a.write, prereg_dir, registries)


if __name__ == "__main__":
    raise SystemExit(main())
