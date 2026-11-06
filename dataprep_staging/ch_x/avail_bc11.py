"""I1.backcast_2010_11.v1 step 0: the registered, metadata-only availability look on ls_cache
2009-10..2011-12 over the fan-zone Yunlin and Changhua wells (round 8 plan #12; user ruling
R8.6a: L1_pre2012 v2 covering Yunlin and Changhua, local, never committed, built only after
this look with the stop rule fixed first).

Two steps, in order, both run by the lead:

``look --validate`` / ``look --write``
    prints / logs (``prereg3.record_look``, purpose ``availability_check``) the look on unit
    I1BC11:ls_cache_dates_2009_2011 whose members are the fan-zone (zone 50) Yunlin and
    Changhua monitoring wells that have an ``ls_cache/gw__<sid>__raw.parquet`` file (file
    names and the station table only), window 2009-10..2011-12, ids only (never universal).
    Both run the prereg3 blocking probe (the earlier looks that share members with the unit
    inside the window; looks.jsonl metadata only) and refuse unless it lists every X5EV and
    I1CHB unit the note discloses. ``--write`` puts the stop rule text and the sha256 of the
    stop-rule constants (``constants()``) in the note. stdout carries the unit, the member
    count, the scope, the constants hash and the probe hits (unit ids and counts), never a
    member station id.
``count``
    refuses unless that look is logged once, its note carries this file's constants hash and
    rule text, and today's candidate list equals the logged members; refuses if the pinned
    output ``OUT`` exists. Then, per member, reads ONLY the ``datetime`` column
    (``pyarrow.parquet.read_schema`` then ``read_table(columns=['datetime'])``); the
    ``value`` column is never decoded or returned (the file is hashed whole for provenance
    and its footer schema is parsed). It lists the 2009-10..2011-12 months whose sampling
    could make a monthly value (>= 20 days with >= 12 distinct hours;
    ``bc_lib.available_months``) and applies the stop rule below. Writes ``OUT`` once (mode
    ``x``); prints only counts and PASS / STOP.

Stop rule (fixed here, before the look; constants below, hashed into the look note):

* a well QUALIFIES for origin 2010-12 iff it has >= 9 available months in 2011 AND >= 1
  available month in 2010-10..2010-12 (the 3 months before 2011-01);
* ring-nest wells (within 1000 m of a ring site: mechanism._long_head read their whole
  ls_cache files) stay look members; with ``EXCLUDE_NEST`` (True, the precedent's rule) they
  are excluded from the GATING count, and the count with them is reported beside it,
  report-only;
* PASS iff >= 25 pooled (Yunlin + Changhua) wells qualify in the gating count; otherwise
  STOP (not evaluable);
* a county line (Yunlin or Changhua alone) is reported only for a county with >= 10
  qualifying wells in the gating count;
* report-only, never gating: the wells that would qualify for a second origin 2009-12
  (>= 9 available months in 2010 and >= 1 in 2009-10..2009-12), both counts.

What the later registration must do (plan #12): I1.backcast_2010_11.v1 declares BRAND-NEW
fresh-unit ids (neither I1BC11:ls_cache_dates_2009_2011 nor E26's I1CHB units), and the
L1_pre2012 v2 ``product_build`` look must not name the test's fresh unit; otherwise these
looks block the test by id.

    python dataprep_staging/ch_x/avail_bc11.py look --validate
    python dataprep_staging/ch_x/avail_bc11.py look --write
    python dataprep_staging/ch_x/avail_bc11.py count
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
BC11_ID = "I1.backcast_2010_11.v1"
AVAIL_ID = "I1.backcast_2010_11.availability.v1"
UNIT = "I1BC11:ls_cache_dates_2009_2011"
WINDOW = ("2009-10", "2011-12")
SCOPE = f"heads:{WINDOW[0]}..{WINDOW[1]}"
WP = K.WP
PURPOSE = AL.PURPOSE
COUNTIES = ("yunlin", "changhua")
RULING = "R8.6a"
OUT = "results/twin_runs/chx/backcast_2010_11/avail/avail.json"   # pinned, written once

MIN_POOLED_WELLS = 25                      # PASS iff >= 25 pooled qualifying wells
MIN_YEAR_MONTHS = 9                        # >= 9 available months of the scored year
SCORED_YEAR = ("2011-01", "2011-12")
ORIGIN = "2010-12"
ORIGIN_WINDOW = ("2010-10", "2010-12")     # >= 1 available month in the 3 months before
MIN_ORIGIN_MONTHS = 1
COUNTY_LINE_MIN_WELLS = 10                 # a county line only with >= 10 qualifying wells
#: report-only second origin (never gating)
ORIGIN2 = "2009-12"
ORIGIN2_YEAR = ("2010-01", "2010-12")
ORIGIN2_WINDOW = ("2009-10", "2009-12")
EXCLUDE_NEST = True                        # nest wells out of the gating count (precedent)
NEST_RADIUS_M = K.BC_NEST_RADIUS_M

#: earlier looks on these members that the note discloses (plan #12, 'Register before look');
#: the probe must list every one of them or the look is not written
X5EV_UNITS = ("X5EV:confined_ls_cache_2010_2011", "X5EV:unconfined_ls_cache_2010_2011_reported")
I1CHB_UNITS = (K.BC_AVAIL_UNIT, K.BC_UNIT)
REQUIRED_PROBE_UNITS = X5EV_UNITS + I1CHB_UNITS
PRINTED_WELL = "07010211"                  # already public in tracked files


def _nest_phrase() -> str:
    if EXCLUDE_NEST:
        return (f"ring-nest wells (<= {NEST_RADIUS_M:.0f} m of a ring site) stay look members "
                "but are excluded from the gating count (the count with them is reported "
                "beside it, report-only)")
    return (f"ring-nest wells (<= {NEST_RADIUS_M:.0f} m of a ring site) are counted (the "
            "count without them is reported beside it, report-only)")


RULE_TEXT = (
    f"stop rule fixed in code before the look: a well qualifies for origin {ORIGIN} iff it "
    f"has >= {MIN_YEAR_MONTHS} available months in {SCORED_YEAR[0]}..{SCORED_YEAR[1]} and "
    f">= {MIN_ORIGIN_MONTHS} available month in {ORIGIN_WINDOW[0]}..{ORIGIN_WINDOW[1]}; "
    f"{_nest_phrase()}; PASS iff >= {MIN_POOLED_WELLS} pooled Yunlin+Changhua wells qualify, "
    f"else STOP (not evaluable); county lines only for a county with >= "
    f"{COUNTY_LINE_MIN_WELLS} qualifying wells; report-only: wells qualifying for a second "
    f"origin {ORIGIN2} (>= {MIN_YEAR_MONTHS} months in {ORIGIN2_YEAR[0]}..{ORIGIN2_YEAR[1]} "
    f"and a month in {ORIGIN2_WINDOW[0]}..{ORIGIN2_WINDOW[1]}); a month is available with >= "
    f"{K.BC_MONTH_MIN_DAYS} days of >= {K.BC_DAY_MIN_HOURS} distinct hours (datetime only)")

DISCLOSURE_TEXT = (
    "disclosed earlier reads of these members: X5EV (X5.energy_volume.c_vs_a.v1, scored "
    f"2026-10-25; units {', '.join(X5EV_UNITS)}) loaded the 2010-11 hourly ls_cache series of "
    "its confined wells for a 1-cpd band estimand and reported its unconfined companion unit; "
    f"three hourly 2010-01-01 values of confined well {PRINTED_WELL} were printed in a "
    "metadata check before X5EV was registered (disclosed 2026-10-25); I1CHB "
    f"(I1.CH_backcast.v1, E26; units {', '.join(I1CHB_UNITS)}) logged a timestamps "
    "availability look, built heads/L1_pre2012 v1 and scored 2010-11 Changhua heads "
    "(2026-11-01), so the Changhua members overlap E26's wells; mechanism.py's long-heads "
    "option (mechanism._long_head) read the whole ls_cache files of the wells near ring sites "
    "(no date clip): those wells are look members and are excluded only from the gating "
    "count; the later climatology baseline CL is refitted on 2012-22 and applied backward to "
    "2010-11 (time-reversed)")


# ----------------------------------------------------------------------------- rule
def constants() -> dict:
    """Every stop-rule constant; its sha256 goes into the look note."""
    return {"id": AVAIL_ID, "unit": UNIT, "scope": SCOPE, "window": list(WINDOW),
            "counties": list(COUNTIES), "min_pooled_wells": MIN_POOLED_WELLS,
            "min_year_months": MIN_YEAR_MONTHS, "scored_year": list(SCORED_YEAR),
            "origin": ORIGIN, "origin_window": list(ORIGIN_WINDOW),
            "min_origin_months": MIN_ORIGIN_MONTHS,
            "county_line_min_wells": COUNTY_LINE_MIN_WELLS, "origin2": ORIGIN2,
            "origin2_year": list(ORIGIN2_YEAR), "origin2_window": list(ORIGIN2_WINDOW),
            "exclude_nest": EXCLUDE_NEST, "nest_radius_m": NEST_RADIUS_M,
            "day_min_hours": K.BC_DAY_MIN_HOURS, "month_min_days": K.BC_MONTH_MIN_DAYS,
            "out": OUT, "bounds": "inclusive"}


def qualifies(months, year=SCORED_YEAR, owin=ORIGIN_WINDOW) -> bool:
    return (AL.n_in(months, *year) >= MIN_YEAR_MONTHS
            and AL.n_in(months, *owin) >= MIN_ORIGIN_MONTHS)


def _counts(wells, county_of) -> dict:
    return {"wells": wells, "n": len(wells),
            "by_county": {c: sum(county_of[s] == c for s in wells) for c in COUNTIES}}


def stop_rule(months_by_well: dict, county_of: dict, nest: set) -> dict:
    """The fixed stop rule on availability counts (pure; see the module docstring). Both the
    nest-excluded and the nest-included counts are computed from the same months; the one
    ``EXCLUDE_NEST`` names gates, the other is report-only."""
    pool = sorted(s for s in months_by_well if county_of.get(s) in COUNTIES)
    nest = set(nest) & set(pool)
    q1 = [s for s in pool if qualifies(months_by_well[s])]
    q2 = [s for s in pool if qualifies(months_by_well[s], ORIGIN2_YEAR, ORIGIN2_WINDOW)]
    excl, incl = _counts([s for s in q1 if s not in nest], county_of), _counts(q1, county_of)
    gate, other = (excl, incl) if EXCLUDE_NEST else (incl, excl)
    o2x = _counts([s for s in q2 if s not in nest], county_of)
    o2i = _counts(q2, county_of)
    per = {s: {"county": county_of[s],
               "n_months_2011": AL.n_in(months_by_well[s], *SCORED_YEAR),
               "n_months_origin_window": AL.n_in(months_by_well[s], *ORIGIN_WINDOW),
               "n_months_2010": AL.n_in(months_by_well[s], *ORIGIN2_YEAR),
               "n_months_origin2_window": AL.n_in(months_by_well[s], *ORIGIN2_WINDOW),
               "nest": s in nest} for s in pool}
    return {"n_candidates": len(pool),
            "n_candidates_by_county": {c: sum(county_of[s] == c for s in pool)
                                       for c in COUNTIES},
            "n_nest": len(nest), "gating": "nest_excluded" if EXCLUDE_NEST else "nest_included",
            "nest_excluded": excl, "nest_included": incl,
            "eligible_wells": gate["wells"], "n_eligible": gate["n"],
            "n_eligible_by_county": gate["by_county"],
            "report_only_other_count": {"which": ("nest_included" if EXCLUDE_NEST
                                                  else "nest_excluded"),
                                        "n": other["n"], "by_county": other["by_county"]},
            "county_line": {c: gate["by_county"][c] >= COUNTY_LINE_MIN_WELLS
                            for c in COUNTIES},
            "passed": gate["n"] >= MIN_POOLED_WELLS,
            "origin2_report_only": {"origin": ORIGIN2, "nest_excluded": o2x,
                                    "nest_included": o2i,
                                    "n_eligible": (o2x if EXCLUDE_NEST else o2i)["n"]},
            "per_well": per, "constants": constants(),
            "constants_sha256": AL.constants_sha(constants())}


# ----------------------------------------------------------------------------- look
def candidates(st=None) -> list[str]:
    return AL.candidates(COUNTIES, st)


def members(ids) -> dict:
    return {"description": "fan-zone (zone 50) Yunlin and Changhua monitoring wells with an "
                           "ls_cache gw__<sid>__raw file: per-month sampling counts "
                           f"{WINDOW[0]}..{WINDOW[1]} from the datetime column only (no "
                           f"value), {BC11_ID}'s step-0 availability look (user ruling "
                           f"{RULING})",
            "ids": sorted(map(str, ids)), "window": f"{WINDOW[0]}..{WINDOW[1]}"}


def look_record(ids, prereg_dir=None, registries=None) -> dict:
    m = members(ids)
    ov = AL.prior_overlaps(UNIT, m, SCOPE, prereg_dir, registries)
    note = (f"[{AVAIL_ID}] {AL.sha_tag(constants())}; metadata-only availability look "
            "(datetime column only; the value column is never decoded or returned); "
            f"{RULE_TEXT}. {DISCLOSURE_TEXT}. prereg3 probe, earlier looks sharing members in "
            f"{WINDOW[0]}..{WINDOW[1]}: {AL.overlap_summary(ov)}")
    return {"units": [UNIT], "data_scope": SCOPE, "wp": WP, "purpose": PURPOSE,
            "members": {UNIT: m}, "note": note, "probe": ov}


def look(write: bool, prereg_dir=None, registries=None) -> int:
    from harness import prereg3

    rec = look_record(candidates(), prereg_dir, registries)
    m = rec["members"][UNIT]
    errs = prereg3.member_errors(m)
    miss = AL.missing_units(rec["probe"], REQUIRED_PROBE_UNITS)
    logged = AL.logged_look(UNIT, prereg_dir) is not None
    print(f"unit {UNIT} | {len(m['ids'])} members | scope {SCOPE} | purpose {PURPOSE}")
    print(AL.sha_tag(constants()))
    print("probe hits (unit | purpose | date | membership | shared members):")
    for ln in AL.probe_lines(rec["probe"]):
        print("  " + ln)
    print("member errors:", errs or "none", "| required probe units missing:", miss or "none",
          "| logged already:", logged)
    if not write:
        return 1 if errs or miss else 0
    if errs or miss:
        raise SystemExit("REFUSED: member errors or a disclosed unit missing from the probe")
    if logged:
        raise SystemExit("REFUSED: the availability look is already logged")
    kw = {} if prereg_dir is None else {"prereg_dir": prereg_dir}
    r = prereg3.record_look(rec["units"], rec["data_scope"], rec["wp"], rec["purpose"],
                            members=rec["members"], note=rec["note"], registries=registries,
                            **kw)
    print(f"logged look at {r['time']}")
    return 0


# ----------------------------------------------------------------------------- count
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
           "months_by_well": months, "n_months_by_well": {s: len(m) for s, m in months.items()},
           "nest_wells": sorted(set(ids) & nest), "feasibility": feas,
           **AL.provenance(ids)}
    AL.write_once(target, rep)
    print(summary(feas))
    return rep


def summary(feas: dict) -> str:
    c = feas["n_eligible_by_county"]
    lines = ", ".join(f"{k} {c[k]} ({'line' if feas['county_line'][k] else 'no line'})"
                      for k in COUNTIES)
    o = feas["report_only_other_count"]
    o2 = feas["origin2_report_only"]
    return (f"availability: gating count ({feas['gating']}) {feas['n_eligible']} qualifying "
            f"pooled wells of {feas['n_candidates']} (>= {MIN_YEAR_MONTHS} months of 2011 and "
            f"a month in {ORIGIN_WINDOW[0]}..{ORIGIN_WINDOW[1]}; {feas['n_nest']} nest wells; "
            f"need >= {MIN_POOLED_WELLS}) -> "
            f"{'PASS' if feas['passed'] else 'STOP (not evaluable)'}; by county: {lines}; "
            f"report-only {o['which']} count {o['n']}; report-only origin {ORIGIN2}: "
            f"{o2['nest_excluded']['n']} nest-excluded, {o2['nest_included']['n']} "
            "nest-included")


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
