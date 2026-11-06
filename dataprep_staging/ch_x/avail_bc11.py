"""I1.backcast_2010_11.v1 step 0: the registered, metadata-only availability look on ls_cache
2009-09..2011-12 over the fan-zone Yunlin and Changhua wells (round 8 plan #12; user ruling
R8.6a: L1_pre2012 v2 covering Yunlin and Changhua, local, never committed, built only after
this look with the stop rule fixed first).

Two steps, in order, both run by the lead:

``look --validate`` / ``look --write``
    prints / logs (``prereg3.record_look``, purpose ``availability_check``) the look on unit
    I1BC11:ls_cache_dates_2009_2011 whose members are the fan-zone (zone 50) Yunlin and
    Changhua monitoring wells that have an ``ls_cache/gw__<sid>__raw.parquet`` file (file
    names and the station table only), window 2009-09..2011-12, ids only (never universal).
    ``--validate`` also runs the prereg3 blocking probe: the earlier looks that share members
    with the unit inside the window (looks.jsonl metadata only). The note carries the stop
    rule and the disclosures plan #12 lists.
``count --out DIR``
    refuses unless that look is logged (and unless today's candidate list equals the logged
    members); then, per member, reads ONLY the ``datetime`` column
    (``pyarrow.parquet.read_table(columns=['datetime'])``; the ``value`` column is never
    read), lists the 2009-09..2011-12 months whose sampling could make a monthly value
    (>= 20 days with >= 12 distinct hours; ``bc_lib.available_months``) and applies the stop
    rule below. Writes avail.json once (mode ``x``); prints only counts and PASS / STOP.

Stop rule (fixed here, before the look; constants below, repeated in the look note):

* a well QUALIFIES for origin 2010-12 iff it has >= 9 available months in 2011 AND >= 1
  available month in 2010-10..2010-12 (the 3 months before 2011-01);
* ring-nest wells (within 1000 m of a ring site, mechanism.py's nest reads; the precedent's
  rule, kept) are excluded before counting;
* PASS iff >= 25 pooled (Yunlin + Changhua) wells qualify; otherwise STOP (not evaluable);
* a county line (Yunlin or Changhua alone) is reported only for a county with >= 10
  qualifying wells;
* report-only, never gating: the wells that would qualify for a second origin 2009-12
  (>= 9 available months in 2010 and >= 1 in 2009-10..2009-12), nest wells excluded.

    python dataprep_staging/ch_x/avail_bc11.py look --validate
    python dataprep_staging/ch_x/avail_bc11.py look --write
    python dataprep_staging/ch_x/avail_bc11.py count --out <run dir>/avail
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import avail_l1pre as AV  # noqa: E402
import avail_lib as AL  # noqa: E402
import chx_const as K  # noqa: E402

# ----------------------------------------------------------------------------- frozen design
BC11_ID = "I1.backcast_2010_11.v1"
AVAIL_ID = "I1.backcast_2010_11.availability.v1"
UNIT = "I1BC11:ls_cache_dates_2009_2011"
WINDOW = ("2009-09", "2011-12")
SCOPE = f"heads:{WINDOW[0]}..{WINDOW[1]}"
WP = K.WP
PURPOSE = AL.PURPOSE
COUNTIES = ("yunlin", "changhua")
RULING = "R8.6a"

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
EXCLUDE_NEST = True                        # ring-nest wells excluded (precedent kept)
NEST_RADIUS_M = K.BC_NEST_RADIUS_M

#: earlier looks on these members that the note discloses (plan #12, 'Register before look')
X5EV_UNITS = ("X5EV:confined_ls_cache_2010_2011", "X5EV:unconfined_ls_cache_2010_2011_reported")
I1CHB_UNITS = (K.BC_AVAIL_UNIT, K.BC_UNIT)
PRINTED_WELL = "07010211"                  # already public in tracked files

RULE_TEXT = (
    f"stop rule fixed in code before the look: a well qualifies for origin {ORIGIN} iff it "
    f"has >= {MIN_YEAR_MONTHS} available months in {SCORED_YEAR[0]}..{SCORED_YEAR[1]} and "
    f">= {MIN_ORIGIN_MONTHS} available month in {ORIGIN_WINDOW[0]}..{ORIGIN_WINDOW[1]}; "
    f"ring-nest wells (<= {NEST_RADIUS_M:.0f} m of a ring site) excluded; PASS iff >= "
    f"{MIN_POOLED_WELLS} pooled Yunlin+Changhua wells qualify, else STOP (not evaluable); "
    f"county lines only for a county with >= {COUNTY_LINE_MIN_WELLS} qualifying wells; "
    f"report-only: wells qualifying for a second origin {ORIGIN2} (>= {MIN_YEAR_MONTHS} "
    f"months in {ORIGIN2_YEAR[0]}..{ORIGIN2_YEAR[1]} and a month in "
    f"{ORIGIN2_WINDOW[0]}..{ORIGIN2_WINDOW[1]}); a month is available with >= "
    f"{K.BC_MONTH_MIN_DAYS} days of >= {K.BC_DAY_MIN_HOURS} distinct hours (datetime only)")

DISCLOSURE_TEXT = (
    "disclosed earlier reads of these members: X5EV (X5.energy_volume.c_vs_a.v1, scored "
    "2026-10-25) loaded the 2010-11 hourly ls_cache series of its confined wells for a 1-cpd "
    "band estimand and reported its unconfined companion unit; three hourly 2010-01-01 values "
    f"of confined well {PRINTED_WELL} were printed in a metadata check before X5EV was "
    "registered (disclosed 2026-10-25); I1CHB (I1.CH_backcast.v1, E26) logged a timestamps "
    "availability look, built heads/L1_pre2012 v1 and scored 2010-11 Changhua heads "
    "(2026-11-01), so the Changhua members overlap E26's wells; mechanism.py's long-heads "
    "option read 2010-25 monthly anomalies of the wells near ring sites (excluded here); "
    "the later climatology baseline CL is refitted on 2012-22 and applied backward to "
    "2010-11 (time-reversed)")


# ----------------------------------------------------------------------------- rule
def qualifies(months, year=SCORED_YEAR, owin=ORIGIN_WINDOW) -> bool:
    return (AL.n_in(months, *year) >= MIN_YEAR_MONTHS
            and AL.n_in(months, *owin) >= MIN_ORIGIN_MONTHS)


def stop_rule(months_by_well: dict, county_of: dict, excluded: set) -> dict:
    """The fixed stop rule on availability counts (pure; see the module docstring)."""
    pool = sorted(s for s in months_by_well if county_of.get(s) in COUNTIES)
    excl = set(excluded) if EXCLUDE_NEST else set()
    per = {s: {"county": county_of[s],
               "n_months_2011": AL.n_in(months_by_well[s], *SCORED_YEAR),
               "n_months_origin_window": AL.n_in(months_by_well[s], *ORIGIN_WINDOW),
               "n_months_2010": AL.n_in(months_by_well[s], *ORIGIN2_YEAR),
               "n_months_origin2_window": AL.n_in(months_by_well[s], *ORIGIN2_WINDOW),
               "nest": s in excl} for s in pool}
    elig = sorted(s for s in pool if s not in excl and qualifies(months_by_well[s]))
    elig2 = sorted(s for s in pool if s not in excl
                   and qualifies(months_by_well[s], ORIGIN2_YEAR, ORIGIN2_WINDOW))
    by_c = {c: sum(county_of[s] == c for s in elig) for c in COUNTIES}
    by_c2 = {c: sum(county_of[s] == c for s in elig2) for c in COUNTIES}
    return {"n_candidates": len(pool),
            "n_candidates_by_county": {c: sum(county_of[s] == c for s in pool)
                                       for c in COUNTIES},
            "n_excluded_nest": len(set(pool) & excl),
            "eligible_wells": elig, "n_eligible": len(elig),
            "n_eligible_by_county": by_c,
            "county_line": {c: by_c[c] >= COUNTY_LINE_MIN_WELLS for c in COUNTIES},
            "passed": len(elig) >= MIN_POOLED_WELLS,
            "origin2_report_only": {"origin": ORIGIN2, "eligible_wells": elig2,
                                    "n_eligible": len(elig2), "n_eligible_by_county": by_c2},
            "per_well": per, "constants": constants()}


def constants() -> dict:
    return {"min_pooled_wells": MIN_POOLED_WELLS, "min_year_months": MIN_YEAR_MONTHS,
            "scored_year": list(SCORED_YEAR), "origin": ORIGIN,
            "origin_window": list(ORIGIN_WINDOW), "min_origin_months": MIN_ORIGIN_MONTHS,
            "county_line_min_wells": COUNTY_LINE_MIN_WELLS, "origin2": ORIGIN2,
            "origin2_year": list(ORIGIN2_YEAR), "origin2_window": list(ORIGIN2_WINDOW),
            "exclude_nest": EXCLUDE_NEST, "nest_radius_m": NEST_RADIUS_M,
            "day_min_hours": K.BC_DAY_MIN_HOURS, "month_min_days": K.BC_MONTH_MIN_DAYS,
            "window": list(WINDOW), "bounds": "inclusive"}


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
    note = (f"[{AVAIL_ID}] metadata-only availability look (datetime column only; value "
            f"never read); {RULE_TEXT}. {DISCLOSURE_TEXT}. prereg3 probe, earlier looks "
            f"sharing members in {WINDOW[0]}..{WINDOW[1]}: {AL.overlap_summary(ov)}")
    return {"units": [UNIT], "data_scope": SCOPE, "wp": WP, "purpose": PURPOSE,
            "members": {UNIT: m}, "note": note, "probe": ov}


# ----------------------------------------------------------------------------- count
def count(out: Path, prereg_dir=None) -> dict:
    lk = AL.logged_look(UNIT, prereg_dir)
    if lk is None:
        raise SystemExit("REFUSED: log the availability look first (look --write)")
    target = Path(out) / "avail.json"
    if target.exists():
        raise SystemExit(f"{target} exists; the availability look runs once")
    st = AV.stations()
    ids = AL.logged_ids(lk, UNIT)
    if ids != candidates(st):
        raise SystemExit("REFUSED: today's candidate list differs from the logged members")
    months = AL.months_by_well(ids, WINDOW)
    nest = AL.nest_excluded(ids, st, NEST_RADIUS_M)
    feas = stop_rule(months, AL.county_of(st), nest)
    rep = {"what": f"{AVAIL_ID} (timestamps only)", "unit": UNIT, "look_time": lk["time"],
           "months_by_well": months, "n_months_by_well": {s: len(m) for s, m in months.items()},
           "nest_excluded": sorted(set(ids) & nest), "feasibility": feas,
           **AL.provenance(ids)}
    AL.write_once(target, rep)
    print(summary(feas))
    return rep


def summary(feas: dict) -> str:
    c = feas["n_eligible_by_county"]
    lines = ", ".join(f"{k} {c[k]} ({'line' if feas['county_line'][k] else 'no line'})"
                      for k in COUNTIES)
    o2 = feas["origin2_report_only"]
    return (f"availability: {feas['n_eligible']} qualifying pooled wells of "
            f"{feas['n_candidates']} (>= {MIN_YEAR_MONTHS} months of 2011 and a month in "
            f"{ORIGIN_WINDOW[0]}..{ORIGIN_WINDOW[1]}; {feas['n_excluded_nest']} nest wells "
            f"excluded; need >= {MIN_POOLED_WELLS}) -> "
            f"{'PASS' if feas['passed'] else 'STOP (not evaluable)'}; by county: {lines}; "
            f"report-only origin {ORIGIN2}: {o2['n_eligible']} wells")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("look")
    g = lk.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    c = sub.add_parser("count")
    c.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "count":
        count(a.out)
        return 0
    from harness import prereg3

    rec = look_record(candidates())
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    if a.validate:
        errs = prereg3.member_errors(rec["members"][UNIT])
        print("member errors:", errs or "none", "| logged already:",
              AL.logged_look(UNIT) is not None)
        return 1 if errs else 0
    if AL.logged_look(UNIT) is not None:
        raise SystemExit("REFUSED: the availability look is already logged")
    r = prereg3.record_look(rec["units"], rec["data_scope"], rec["wp"], rec["purpose"],
                            members=rec["members"], note=rec["note"])
    print(f"logged look at {r['time']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
