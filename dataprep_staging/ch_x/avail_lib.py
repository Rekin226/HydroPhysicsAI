"""Shared helpers of the round-8 metadata-only availability looks (avail_bc11.py, plan #12, and
avail_chb0809.py, plan #18). Nothing here reads a head value.

What each helper touches, and nothing else:

* ``candidates``: the station table's metadata columns (``avail_l1pre.stations``: fan zone 50,
  county from ``CountyName``) and the EXISTENCE of ``ls_cache/gw__<sid>__raw.parquet`` (file
  names only);
* ``timestamps_by_well`` / ``months_by_well``: per candidate, ONLY the ``datetime`` column
  (``avail_l1pre.timestamps``: ``pyarrow.parquet.read_schema`` then
  ``pyarrow.parquet.read_table(columns=['datetime'])``); the ``value`` column is never decoded
  or returned (the file is hashed whole for provenance and its footer schema is parsed).
  The monthly-sampling rule is ``bc_lib.available_months`` (a month is available with >= 20
  days holding >= 12 distinct hours), applied to the look's own window;
* ``nest_excluded``: station coordinates and the ring-site table (``bc_lib.nest_wells``, the
  1000 m ring-nest rule of I1.CH_backcast.v1);
* ``logged_look`` / ``prior_overlaps``: the hash-chained ``looks.jsonl`` and the membership
  index (unit ids, member ids, windows, purposes and times of earlier looks; never a value);
* ``constants_sha`` / ``note_matches``: the stop rule's constants are hashed into the look
  note at ``look --write``; ``count`` refuses unless the logged note carries the same hash and
  rule text (no rule shopping between the look and the count);
* ``write_once``: the output (a constant path per script) is opened with mode ``x`` (an
  existing file refuses).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import avail_l1pre as AV  # noqa: E402
import bc_lib as BL  # noqa: E402
import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402

PURPOSE = "availability_check"


def _prereg_dir(prereg_dir=None) -> Path:
    from common.paths import PREREG_DIR

    return Path(prereg_dir or PREREG_DIR)


# ----------------------------------------------------------------------------- metadata
def candidates(counties, st=None) -> list[str]:
    """Fan-zone wells of ``counties`` that have an ls_cache raw file (file names only)."""
    st = AV.stations() if st is None else st
    sub = st[st["county"].isin(tuple(counties))]
    return sorted(s for s in sub["sid"] if AV.raw_path(s).exists())


def county_of(st) -> dict:
    return dict(zip(st["sid"], st["county"], strict=True))


def nest_excluded(ids, st, radius: float = K.BC_NEST_RADIUS_M) -> set:
    """Candidates within ``radius`` of a ring site (bc_lib.nest_wells; coordinates only)."""
    xy = {r.sid: (r.x, r.y) if np.isfinite(r.x) else None for r in st.itertuples()}
    return BL.nest_wells({s: xy.get(s) for s in ids}, AV.ring_xy(), radius=radius)


# ----------------------------------------------------------------------------- timestamps
def months_by_well(ids, window: tuple[str, str]) -> dict:
    """{sid: available months of ``window``} from the datetime column only."""
    return {s: BL.available_months(AV.timestamps(s), window[0], window[1],
                                   day_min_hours=K.BC_DAY_MIN_HOURS,
                                   month_min_days=K.BC_MONTH_MIN_DAYS) for s in ids}


def n_in(months, first: str, last: str) -> int:
    """Number of 'YYYY-MM' months in [first, last] (inclusive)."""
    return sum(first <= m <= last for m in months)


# ----------------------------------------------------------------------------- looks
def logged_looks(unit: str, prereg_dir=None) -> list[dict]:
    from harness import prereg3

    return [lk for lk in prereg3.read_chained(_prereg_dir(prereg_dir) / "looks.jsonl")
            if unit in (lk.get("units") or []) and lk.get("purpose") == PURPOSE]


def logged_look(unit: str, prereg_dir=None) -> dict | None:
    """The single logged availability look on ``unit`` (None if absent; refuses if several)."""
    lks = logged_looks(unit, prereg_dir)
    if len(lks) > 1:
        raise SystemExit(f"REFUSED: {len(lks)} availability looks logged on {unit}; "
                         "the look runs once")
    return lks[0] if lks else None


def logged_ids(lk: dict, unit: str) -> list[str]:
    m = (lk.get("members") or {}).get(unit) or {}
    return sorted(map(str, m.get("ids") or []))


def prior_overlaps(unit: str, members: dict, scope: str, prereg_dir=None,
                   registries=None) -> list[dict]:
    """The prereg3 blocking probe for a new unit: every earlier look whose window intersects
    ``scope`` and whose unit shares members with ``members`` (or has unknown membership).
    Reads looks.jsonl and the membership index only."""
    from harness import prereg3

    d = _prereg_dir(prereg_dir)
    idx = prereg3.membership_index(d, registries)
    # in-memory only: the unit is not declared anywhere yet; nothing is written
    idx.members[unit] = prereg3._declared_members(members)
    ids = frozenset(map(str, members.get("ids") or []))
    test = {"fresh_units": [unit], "data_scope": scope}
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for lk in prereg3.read_chained(d / "looks.jsonl"):
        hits = prereg3.look_hits(lk, test, idx, test_time=now)
        for kind in ("overlap", "unknown"):
            for lu, _ in hits[kind]:
                if lu == unit:
                    continue
                m = idx.members.get(lu)
                shared = None
                if m is not None and not m.get("universal"):
                    shared = len(ids & m["ids"])
                out.append({"time": lk.get("time"), "purpose": lk.get("purpose"),
                            "wp": lk.get("wp"), "unit": lu,
                            "data_scope": lk.get("data_scope"), "membership": kind,
                            "universal": bool(m and m.get("universal")),
                            "n_shared": shared})
    return out


def overlap_summary(rows: list[dict]) -> str:
    """One line per overlapping unit (known membership) plus a count of unknown ones."""
    known = [r for r in rows if r["membership"] == "overlap"]
    unknown = [r for r in rows if r["membership"] == "unknown"]
    parts = []
    for r in known:
        n = "universal" if r["universal"] else f"{r['n_shared']} shared"
        parts.append(f"{r['unit']} ({r['purpose']}, {str(r['time'])[:10]}, {n})")
    s = "; ".join(parts) if parts else "none with known membership"
    if unknown:
        s += f"; plus {len(unknown)} look(s) on units of unknown membership in the window"
    return s


# ----------------------------------------------------------------------------- rule hash
def constants_sha(constants: dict) -> str:
    """sha256 of the canonical JSON of a script's stop-rule constants."""
    return hashlib.sha256(json.dumps(constants, sort_keys=True).encode()).hexdigest()


def sha_tag(constants: dict) -> str:
    return f"constants_sha256={constants_sha(constants)}"


def note_matches(lk: dict, constants: dict, rule_text: str) -> bool:
    """Does the logged look's note carry these constants' hash and this rule text?"""
    note = str(lk.get("note") or "")
    return sha_tag(constants) in note and rule_text in note


def probe_lines(rows: list[dict]) -> list[str]:
    """Probe hits as unit ids and counts only (no member id)."""
    return [f"{r['unit']} | {r['purpose']} | {str(r['time'])[:10]} | {r['membership']} | "
            f"{'universal' if r['universal'] else r['n_shared']}" for r in rows]


def missing_units(rows: list[dict], required) -> list[str]:
    seen = {r["unit"] for r in rows}
    return [u for u in required if u not in seen]


# ----------------------------------------------------------------------------- output
def write_once(target: Path, rep: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)


def provenance(ids) -> dict:
    return {"files_sha256": {s: IO.sha256(AV.raw_path(s)) for s in ids},
            "stations": K.STATIONS, "stations_sha256": IO.sha256(IO.rp(K.STATIONS)),
            "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
