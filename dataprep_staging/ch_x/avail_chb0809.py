"""Changhua step-0 count-only look on ls_cache 2008-01..2009-12 (round 8 plan #18: closure of
the last unverified Changhua head window). A second reviewer checks this script before it runs.

Two steps, in order, both run by the lead:

``look --validate`` / ``look --write``
    prints / logs (``prereg3.record_look``, purpose ``availability_check``) the look on unit
    I1CHB0:ls_cache_rowcounts_2008_2009 whose members are the fan-zone (zone 50) Changhua
    monitoring wells that have an ``ls_cache/gw__<sid>__raw.parquet`` file (file names and the
    station table only), window 2008-01..2009-12, ids only (never universal). ``--validate``
    also runs the prereg3 blocking probe (looks.jsonl metadata only). The look scope is in
    looks.jsonl before any count is read.
``count --out DIR``
    refuses unless that look is logged (and unless today's candidate list equals the logged
    members); then, per member, reads ONLY the ``datetime`` column
    (``pyarrow.parquet.read_table(columns=['datetime'])``; the ``value`` column is never
    read), lists the 2008-01..2009-12 months whose sampling could make a monthly value
    (>= 20 days with >= 12 distinct hours; ``bc_lib.available_months``) and applies the stop
    rule below. Writes avail.json once (mode ``x``); prints only counts and PASS / STOP.

Stop rule (fixed here, before the look; constants below, repeated in the look note):

* PASS iff >= 10 Changhua wells have >= 18 available months within 2008-01..2009-12,
  ring-nest wells (within 1000 m of a ring site; avail_l1pre.py's pattern, kept) excluded;
  the count with the nest wells included is reported only, never gating;
* on PASS: continue only after a ruling on L1_pre2012 v2 coverage of 2008-09, reusing N2CS's
  frozen estimator (any later backcast G0: power >= 0.6, false pass <= 0.2);
* on STOP: record 'last possible Changhua head window checked, not decidable'.

    python dataprep_staging/ch_x/avail_chb0809.py look --validate
    python dataprep_staging/ch_x/avail_chb0809.py look --write
    python dataprep_staging/ch_x/avail_chb0809.py count --out <run dir>/avail
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
AVAIL_ID = "I1.CH_backcast_2008_09.availability.v1"
UNIT = "I1CHB0:ls_cache_rowcounts_2008_2009"
WINDOW = ("2008-01", "2009-12")
SCOPE = f"heads:{WINDOW[0]}..{WINDOW[1]}"
WP = K.WP
PURPOSE = AL.PURPOSE
COUNTIES = ("changhua",)

MIN_WELLS = 10                             # PASS iff >= 10 wells ...
MIN_MONTHS = 18                            # ... with >= 18 available months in the window
EXCLUDE_NEST = True                        # ring-nest wells excluded (precedent kept)
NEST_RADIUS_M = K.BC_NEST_RADIUS_M
STOP_LINE = "last possible Changhua head window checked, not decidable"

RULE_TEXT = (
    f"stop rule fixed in code before the look: PASS iff >= {MIN_WELLS} Changhua wells have "
    f">= {MIN_MONTHS} available months within {WINDOW[0]}..{WINDOW[1]}, ring-nest wells "
    f"(<= {NEST_RADIUS_M:.0f} m of a ring site) excluded (count with them reported only); a "
    f"month is available with >= {K.BC_MONTH_MIN_DAYS} days of >= {K.BC_DAY_MIN_HOURS} "
    "distinct hours (datetime only); on PASS continue only after a ruling on L1_pre2012 v2 "
    f"coverage with N2CS's frozen estimator; on STOP record '{STOP_LINE}'")


def stop_rule(months_by_well: dict, county_of: dict, excluded: set) -> dict:
    """The fixed stop rule on availability counts (pure; see the module docstring)."""
    pool = sorted(s for s in months_by_well if county_of.get(s) in COUNTIES)
    excl = set(excluded) if EXCLUDE_NEST else set()
    n = {s: AL.n_in(months_by_well[s], *WINDOW) for s in pool}
    elig = sorted(s for s in pool if n[s] >= MIN_MONTHS and s not in excl)
    with_nest = sorted(s for s in pool if n[s] >= MIN_MONTHS)
    return {"n_changhua_candidates": len(pool),
            "n_excluded_nest": len(set(pool) & excl),
            "eligible_wells": elig, "n_eligible": len(elig),
            "n_ge_min_months_incl_nest_report_only": len(with_nest),
            "passed": len(elig) >= MIN_WELLS, "n_months_by_well": n,
            "constants": {"min_wells": MIN_WELLS, "min_months": MIN_MONTHS,
                          "window": list(WINDOW), "exclude_nest": EXCLUDE_NEST,
                          "nest_radius_m": NEST_RADIUS_M,
                          "day_min_hours": K.BC_DAY_MIN_HOURS,
                          "month_min_days": K.BC_MONTH_MIN_DAYS, "bounds": "inclusive"},
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
    note = (f"[{AVAIL_ID}] metadata-only count look (datetime column only; value never "
            f"read); {RULE_TEXT}. prereg3 probe, earlier looks sharing members in "
            f"{WINDOW[0]}..{WINDOW[1]}: {AL.overlap_summary(ov)}")
    return {"units": [UNIT], "data_scope": SCOPE, "wp": WP, "purpose": PURPOSE,
            "members": {UNIT: m}, "note": note, "probe": ov}


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
           "months_by_well": months, "nest_excluded": sorted(set(ids) & nest),
           "feasibility": feas, **AL.provenance(ids)}
    AL.write_once(target, rep)
    print(summary(feas))
    return rep


def summary(feas: dict) -> str:
    return (f"availability 2008-09: {feas['n_eligible']} eligible Changhua wells of "
            f"{feas['n_changhua_candidates']} (>= {MIN_MONTHS} months; "
            f"{feas['n_excluded_nest']} nest wells excluded; need >= {MIN_WELLS}) -> "
            f"{'PASS' if feas['passed'] else 'STOP (' + STOP_LINE + ')'}")


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
