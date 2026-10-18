"""Pre-registration tool, second repair: ``harness.prereg3`` (WP A1, 2026-10-12).

Use this module for every new pre-registration, look and score. ``harness.prereg`` and
``harness.prereg2`` stay importable and unchanged because published products hash them
(``prereg2.py``: surface_water_v4 full/v3); all three tools share ``INDEX.jsonl``,
``looks.jsonl`` and the ledger, and each reads records written by the others.

Plan WP A1: the tool "refuses to score a design whose pre-registration postdates any look at
its test data". The A1 verifier showed two ways round ``prereg2``:

1. **Scopes were misread into narrower windows.** ``prereg2.scope_window`` read
   ``heads:2016-01-01..2018-12-31`` as 2016-01 only (the day component broke the range
   pattern) and kept only the first window of ``2012-01..2015-12,2019-01..2022-12``, so a look
   on those data did not block a 2017-18 or a 2020-22 test. Here a scope is parsed by a strict
   grammar into a UNION of month windows (:func:`scope_windows`), and anything the grammar
   does not fully understand makes the scope unbounded (it overlaps every scope):

   * ``"*"``, an empty scope, or a scope without a time segment -> unbounded;
   * the scope is split on ``:``; a *time segment* is a list of windows separated by ``,``
     ``;`` ``+`` or blanks, each window ``DATE`` or ``DATE..DATE``, ``DATE`` = ``YYYY``,
     ``YYYY-MM`` or ``YYYY-MM-DD`` (valid calendar dates, years 1900-2099); a day widens to
     its whole month, a bare year to its whole year;
   * unbounded if: more than one time segment; a segment that holds a standalone year
     (19xx/20xx) but is not a time segment (``extra_2021``, ``2016-2018``, ``201601``); an
     inverted range; a relational word anywhere (before, after, since, until, through, to,
     from, pre, post, prior, except, excluding, without, not, outside, rest, onward(s),
     preorigin, postorigin, ...) or one of ``< > ! ~``.
   Other segments (dataset names, station lists) are ignored, which only widens the scope.

2. **Blocking compared unit ids, not membership.** The v2 fresh units nest (every well of
   ``fresh.rot_tercile.1`` lies in ``fresh.two_tercile.WM``; ``fresh.folds.wells_YL`` holds all
   84 Yunlin wells), so a test scored on ``fresh.km2027.0`` neither blocked nor disclosed a
   later test on ``fresh.rot_tercile.1``. Here two units overlap when their MEMBERS intersect
   (:func:`membership_index`): wells, leveling benchmarks and fold test sets (``ids``), model
   cells (``cells``); a temporal unit spans the registry's calibration wells and every active
   cell (its head-data domain; leveling benchmarks are not part of it) within its own month
   ``window``, intersected with the scope. Membership comes from every
   ``validation_units/v*/units.json`` (withdrawn ids resolved to their replacements), from
   external units declared with members in a pre-registration, and from the ``members`` of
   looks written by this tool.

The rule (:data:`BLOCKING_RULE`): a look blocks a test if it is dated before the test's
pre-registration, its data window overlaps the test's (scope windows intersected with the unit
windows), and it names (a) the same unit as a fresh unit of the test, or (b) a unit whose
members intersect that fresh unit's members and which is either a registry *fresh* unit or
was looked at on or after the time the test's unit was first declared fresh (the earliest
pre-registration listing it as a fresh unit). Looks on design-time units before that
declaration are the history plan 4.5 labels as design-time evidence; they do not block.
Looks on units whose membership is unknown (legacy looks on undeclared external units) cannot
be compared: they block only by id and are DISCLOSED as unverifiable by
:func:`assert_can_score` and :func:`audit`. :func:`record_look` and :func:`write_prereg`
therefore require members for every unit outside the registries (``{"none": reason}`` for
units that contain no registered well, benchmark or cell, such as simulation draws).

Everything else is ``prereg2``: any purpose blocks, the registered file and inputs must be
intact, looks are validated, the look log stays self-reported (an unlogged look cannot be
detected), and back-dated looks are only ``design_time_seed`` looks.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from pathlib import Path

from common import ledger
from common.hashing import sha256_file
from common.paths import DATAPREP, LEDGER_ROWS, PREREG_DIR, from_repo, rel_to_repo

from harness import __version__ as HARNESS_VERSION
from harness import prereg as _v1
from harness import prereg2 as _v2
from harness.prereg2 import (  # noqa: F401 - re-exported
    AUTO,
    KINDS,
    NA,
    REQUIRED,
    RESERVED_PURPOSES,
    SEED_PURPOSE,
    PreregistrationError,
    changed_inputs,
    load_prereg,
    not_applicable,
    parse_time,
    read_chained,
)

TOOL = "harness.prereg3"
TOOL_VERSION = "3.0.0"
SCHEMA = "dataprep-prereg/3"
BLOCKING_RULE = (
    "a look blocks a test if it is dated before the test's pre-registration, its data window "
    "overlaps the test's (scope windows, a strict union-of-months parse in which anything not "
    "understood is unbounded, intersected with unit windows), and it names the same unit as a "
    "fresh unit of the test, or a unit whose members (wells, benchmarks, fold test sets, "
    "cells; temporal units span every calibration well and active cell) intersect that "
    "fresh unit's members and "
    "which is a registry fresh unit or was looked at on or after that fresh unit was first "
    "declared fresh; looks on units of unknown membership block by id only and are disclosed")
MEMBER_KEYS = ("ids", "cells", "universal", "window", "none", "description")
_UTC = _dt.timezone.utc
_RELATIONAL = frozenset({
    "before", "after", "since", "until", "till", "through", "thru", "to", "from", "pre",
    "post", "prior", "onward", "onwards", "except", "excluding", "exclude", "excl", "without",
    "not", "outside", "rest", "remaining", "complement", "minus", "but", "beyond", "preorigin",
    "postorigin", "upto", "between", "other", "others", "non"})
_SYMBOLS = re.compile(r"[<>!~≤≥]")
_YEAR = re.compile(r"(?<![0-9])(?:19|20)[0-9]{2}(?![0-9])")
_DATE = re.compile(r"((?:19|20)[0-9]{2})(?:-([0-9]{2})(?:-([0-9]{2}))?)?")


def _now() -> str:
    return _dt.datetime.now(_UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ----------------------------------------------------------------------------- scopes
def _date_bounds(tok: str) -> tuple[int, int] | None:
    """(first, last) month index (12 * year + month - 1) of a DATE token, None if invalid."""
    m = _DATE.fullmatch(tok)
    if not m:
        return None
    y, mm, dd = m.groups()
    y = int(y)
    if mm is None:
        return 12 * y, 12 * y + 11
    mo = int(mm)
    if not 1 <= mo <= 12:
        return None
    if dd is not None:
        try:
            _dt.date(y, mo, int(dd))
        except ValueError:
            return None
    return 12 * y + mo - 1, 12 * y + mo - 1


def _time_segment(seg: str) -> list[tuple[int, int]] | None:
    """Windows of a time segment, or None if ``seg`` is not one (see the module doc)."""
    s = re.sub(r"\s*\.\.\s*", "..", seg.strip())
    toks = [t for t in re.split(r"[,;+\s]+", s) if t]
    if not toks:
        return None
    wins = []
    for t in toks:
        parts = t.split("..")
        if len(parts) == 1:
            b = _date_bounds(parts[0])
            if b is None:
                return None
            wins.append(b)
        elif len(parts) == 2:
            a, b = _date_bounds(parts[0]), _date_bounds(parts[1])
            if a is None or b is None or a[0] > b[1]:
                return None
            wins.append((a[0], b[1]))
        else:
            return None
    return wins


def _merge(wins) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(wins):
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def scope_windows(scope) -> list[tuple[int, int]] | None:
    """Union of month windows (12 * year + month - 1, inclusive) of a data scope, or None when
    the scope is unbounded (no time segment, or anything the strict grammar does not
    understand; see the module docstring). None overlaps every scope."""
    if scope is None:
        return None
    s = str(scope).strip()
    if s in ("", "*"):
        return None
    if _SYMBOLS.search(s):
        return None
    words = {w.lower() for w in re.split(r"[^0-9A-Za-z]+", s) if w}
    if words & _RELATIONAL:
        return None
    found = []
    for seg in s.split(":"):
        w = _time_segment(seg)
        if w is not None:
            found.append(w)
        elif _YEAR.search(seg):
            return None
    if len(found) != 1:
        return None
    return _merge(found[0])


def windows_intersect(a, b) -> list[tuple[int, int]] | None:
    """Intersection of two window unions (None = unbounded); [] when they are disjoint."""
    if a is None:
        return None if b is None else list(b)
    if b is None:
        return list(a)
    out = []
    for x0, x1 in a:
        for y0, y1 in b:
            lo, hi = max(x0, y0), min(x1, y1)
            if lo <= hi:
                out.append((lo, hi))
    return _merge(out)


def scopes_overlap(a, b) -> bool:
    """True unless both scopes are bounded and their window unions do not intersect."""
    w = windows_intersect(scope_windows(a), scope_windows(b))
    return w is None or bool(w)


def format_windows(w) -> str:
    if w is None:
        return "unbounded: overlaps every scope"
    return ",".join(f"{a // 12}-{a % 12 + 1:02d}..{b // 12}-{b % 12 + 1:02d}" for a, b in w)


# ----------------------------------------------------------------------------- membership
def _registries() -> list[Path]:
    base = DATAPREP / "validation_units"
    if not base.is_dir():
        return []
    vs = sorted((int(p.name[1:]), p / "units.json") for p in base.iterdir()
                if p.is_dir() and re.fullmatch(r"v\d+", p.name) and (p / "units.json").exists())
    return [p for _, p in vs]


def _registry_domain(reg: dict, path: Path) -> tuple[frozenset, frozenset]:
    """(well ids, cells) a temporal unit of this registry spans: the calibration wells
    (``wells.csv`` next to ``units.json``, else the wells of the cell units) and every active
    cell (``grid.n_active``, else the cells of the cell units). Leveling benchmarks and other
    data sets are not part of it (a temporal unit is a block of the model's head data)."""
    ids, cells = set(), set()
    for u in reg["units"]:
        if u.get("kind") == "cells":
            ids.update(map(str, u.get("wells") or []))
            cells.update(int(c) for c in u.get("cells") or [])
    wc = Path(path).parent / "wells.csv"
    if wc.exists():
        import csv
        with open(wc, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("sid"):
                    ids.add(str(row["sid"]))
                if row.get("cell") not in (None, ""):
                    cells.add(int(row["cell"]))
    n_active = (reg.get("grid") or {}).get("n_active")
    if isinstance(n_active, int) and n_active > 0:
        cells.update(range(n_active))
    return frozenset(ids), frozenset(cells)


def _unit_members(u: dict, domain: tuple[frozenset, frozenset]) -> dict:
    kind = u.get("kind")
    if kind == "months":
        a, b = _date_bounds(u["start"]), _date_bounds(u["end"])
        return {"ids": domain[0], "cells": domain[1], "universal": False,
                "window": [(a[0], b[1])]}
    ids = set(map(str, u.get("wells") or []))
    for f in u.get("folds") or []:
        if isinstance(f, dict):
            ids.update(map(str, f.get("test") or []))
    return {"ids": frozenset(ids), "cells": frozenset(int(c) for c in u.get("cells") or []),
            "universal": False, "window": None}


def _declared_members(v) -> dict | None:
    """Members of an external unit declared as a dict with member keys (None otherwise)."""
    if not isinstance(v, dict) or not (set(v) & {"ids", "cells", "universal", "window",
                                                 "none"}):
        return None
    if v.get("none"):
        return {"ids": frozenset(), "cells": frozenset(), "universal": False, "window": None,
                "none": True}
    win = None
    if v.get("window"):
        win = scope_windows(v["window"])
    return {"ids": frozenset(map(str, v.get("ids") or [])),
            "cells": frozenset(int(c) for c in v.get("cells") or []),
            "universal": bool(v.get("universal")), "window": win}


def member_errors(members: dict) -> list[str]:
    """Errors of a ``members`` declaration (one unit)."""
    if not isinstance(members, dict):
        return ["members must be a dict"]
    e = [f"unknown member key {k!r}" for k in members if k not in MEMBER_KEYS]
    if not set(members) & {"ids", "cells", "universal", "window", "none"}:
        e.append("members need at least one of ids, cells, universal, window or none")
    if members.get("none") and (members.get("ids") or members.get("cells")
                                or members.get("universal")):
        e.append("'none' excludes ids, cells and universal")
    if "none" in members and not (isinstance(members["none"], str) and members["none"]):
        e.append("'none' takes the reason as a string")
    if members.get("window") is not None and scope_windows(members["window"]) is None:
        e.append(f"window {members['window']!r} does not parse as a bounded time window")
    return e


class MembershipIndex:
    """Members, roles, withdrawn ids and first-declaration times of every known unit."""

    def __init__(self, prereg_dir=PREREG_DIR, registries=None):
        self.members: dict[str, dict] = {}
        self.role: dict[str, str] = {}
        self.withdrawn: dict[str, str] = {}
        self.declared: dict[str, _dt.datetime] = {}
        self.sources: dict[str, str] = {}
        regs = list(registries) if registries is not None else _registries()
        self.registries = [rel_to_repo(p) for p in regs]
        for p in regs:                       # oldest first; newer versions override
            reg = json.loads(Path(p).read_text(encoding="utf-8"))
            dom = _registry_domain(reg, Path(p))
            for u in reg["units"]:
                self.members[u["id"]] = _unit_members(u, dom)
                self.role[u["id"]] = u.get("role", "")
                self.sources[u["id"]] = rel_to_repo(p)
            for old, w in (reg.get("withdrawn") or {}).items():
                self.withdrawn[old] = str(w.get("replaced_by", old))
        d = Path(prereg_dir)
        for r in read_chained(d / "INDEX.jsonl"):
            p = d / f"{r['id']}.json"
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            created = parse_time(doc.get("created") or r.get("created"))
            ext = dict(doc.get("external_units") or {}) if isinstance(
                doc.get("external_units"), dict) else {}
            fresh = list(doc.get("fresh_units") or []) if isinstance(
                doc.get("fresh_units"), list) else []
            for t in doc.get("tests") if isinstance(doc.get("tests"), list) else []:
                fresh += list(t.get("fresh_units") or [])
                if isinstance(t.get("external_units"), dict):
                    ext.update(t["external_units"])
            for u, v in ext.items():
                m = _declared_members(v)
                if m is not None and self.canon(u) not in self.members:
                    self.members[self.canon(u)] = m
                    self.sources[self.canon(u)] = f"pre-registration {r['id']}"
            for u in fresh:
                c = self.canon(u)
                if c not in self.declared or created < self.declared[c]:
                    self.declared[c] = created
        for lk in read_chained(d / "looks.jsonl"):
            for u, v in (lk.get("members") or {}).items():
                m = _declared_members(v)
                if m is not None and self.canon(u) not in self.members:
                    self.members[self.canon(u)] = m
                    self.sources[self.canon(u)] = f"look {lk.get('time')}"

    def canon(self, u) -> str:
        u = str(u)
        seen = set()
        while u in self.withdrawn and u not in seen:
            seen.add(u)
            u = self.withdrawn[u]
        return u

    def known(self, u) -> bool:
        return self.canon(u) in self.members

    def overlap(self, u, v) -> bool | None:
        """Do units ``u`` and ``v`` share members? None when either membership is unknown
        (the same id always overlaps)."""
        cu, cv = self.canon(u), self.canon(v)
        if cu == cv:
            return True
        mu, mv = self.members.get(cu), self.members.get(cv)
        if mu is None or mv is None:
            return None
        if mu.get("none") or mv.get("none"):
            return False
        if mu["universal"] or mv["universal"]:
            return True
        return bool(mu["ids"] & mv["ids"]) or bool(mu["cells"] & mv["cells"])

    def window(self, u, scope) -> list | None:
        """Data window of a look or test on unit ``u`` within ``scope``."""
        m = self.members.get(self.canon(u))
        return windows_intersect(scope_windows(scope), m["window"] if m else None)


def membership_index(prereg_dir=PREREG_DIR, registries=None) -> MembershipIndex:
    """The :class:`MembershipIndex` of the current registries, pre-registrations and looks."""
    return MembershipIndex(prereg_dir, registries)


# ----------------------------------------------------------------------------- blocking
def _look_brief(lk: dict) -> dict:
    return {k: lk.get(k) for k in ("time", "wp", "purpose", "test_id", "prereg_id",
                                   "data_scope")}


def look_hits(lk: dict, test: dict, idx: MembershipIndex, *, test_time=None) -> dict:
    """How look ``lk`` relates to the fresh units of ``test``: ``{"block": [(look unit, test
    unit), ...], "overlap": [...], "unknown": [...]}``. ``overlap`` = members and windows
    intersect (whatever the time); ``block`` = the subset that is eligible to block under
    :data:`BLOCKING_RULE`; ``unknown`` = pairs whose membership cannot be compared but whose
    windows intersect. ``test_time`` (datetime) is the declaration time used for units never
    declared fresh before (a new registration: now)."""
    t_lk = parse_time(lk["time"])
    out = {"block": [], "overlap": [], "unknown": []}
    for v in test.get("fresh_units") or []:
        cv = idx.canon(v)
        decl = idx.declared.get(cv, test_time)
        wt = idx.window(cv, test["data_scope"])
        for u in lk.get("units") or []:
            cu = idx.canon(u)
            wl = idx.window(cu, lk.get("data_scope"))
            w = windows_intersect(wl, wt)
            if w is not None and not w:
                continue
            ov = idx.overlap(cu, cv)
            if ov is None:
                out["unknown"].append((cu, cv))
                continue
            if not ov:
                continue
            out["overlap"].append((cu, cv))
            eligible = (cu == cv or idx.role.get(cu) == "fresh"
                        or (decl is not None and t_lk >= decl))
            if eligible:
                out["block"].append((cu, cv))
    return out


def blocking_looks(tests: list[dict], before: str | None, prereg_dir=PREREG_DIR,
                   idx: MembershipIndex | None = None, test_time=None) -> list[dict]:
    """Looks that block ``tests`` under :data:`BLOCKING_RULE`, dated before ``before``
    (None = any time)."""
    idx = idx or membership_index(prereg_dir)
    cut = parse_time(before) if before is not None else None
    hits = []
    for lk in read_chained(Path(prereg_dir) / "looks.jsonl"):
        if cut is not None and parse_time(lk["time"]) >= cut:
            continue
        for t in tests:
            h = look_hits(lk, t, idx, test_time=test_time)
            if h["block"]:
                hits.append({"test_id": t["test_id"], "pairs": sorted(set(h["block"])),
                             "look": _look_brief(lk)})
    return hits


def unverifiable_looks(tests: list[dict], before: str | None, prereg_dir=PREREG_DIR,
                       idx: MembershipIndex | None = None) -> list[dict]:
    """Looks dated before ``before`` on units of unknown membership whose windows overlap the
    tests' (they cannot be compared; disclosed, not blocking)."""
    idx = idx or membership_index(prereg_dir)
    cut = parse_time(before) if before is not None else None
    out = []
    for lk in read_chained(Path(prereg_dir) / "looks.jsonl"):
        if cut is not None and parse_time(lk["time"]) >= cut:
            continue
        for t in tests:
            h = look_hits(lk, t, idx)
            if h["unknown"]:
                out.append({"test_id": t["test_id"],
                            "units": sorted({u for u, _ in h["unknown"]}),
                            "look": _look_brief(lk)})
    return out


# ----------------------------------------------------------------------------- looks
def record_look(units, data_scope: str, wp: str, purpose: str, *, members: dict | None = None,
                test_id: str | None = None, prereg_id: str | None = None,
                seeded_time: str | None = None, note: str = "", prereg_dir=PREREG_DIR,
                rows_dir=LEDGER_ROWS, registries=None) -> dict:
    """Append a look (``prereg2.record_look`` rules) on ``units`` within ``data_scope``.
    Every unit outside the registries and the declared external units needs an entry in
    ``members`` ({unit: {"ids": [...], "cells": [...], "universal": bool, "window": "...",
    "description": "..."}} or {unit: {"none": "reason"}})."""
    if purpose in RESERVED_PURPOSES:
        raise PreregistrationError("scoring looks are logged by assert_can_score")
    if not purpose or not isinstance(purpose, str):
        raise ValueError("a look has a purpose")
    if not units:
        raise ValueError("a look names at least one unit")
    if not isinstance(data_scope, str) or not data_scope.strip():
        raise ValueError("a look has a data scope ('*' = every scope)")
    members = dict(members or {})
    idx = membership_index(prereg_dir, registries)
    errs = []
    for u in units:
        if u in members:
            errs += [f"members[{u!r}]: {x}" for x in member_errors(members[u])]
        elif not idx.known(u):
            errs.append(f"unit {u!r} is not in a units registry or a pre-registration with "
                        f"members; pass members={{{u!r}: {{...}}}}")
    extra = sorted(set(members) - set(map(str, units)))
    if extra:
        errs.append(f"members given for units the look does not name: {extra}")
    if errs:
        raise PreregistrationError("; ".join(errs))
    if seeded_time is not None:
        if purpose != SEED_PURPOSE:
            raise ValueError(f"seeded_time is only for purpose={SEED_PURPOSE!r}")
        if prereg_id is not None or test_id is not None:
            raise ValueError("a back-dated design-time seed has no pre-registration or test id")
        if parse_time(seeded_time) > _dt.datetime.now(_UTC):
            raise ValueError("seeded_time lies in the future")
    if test_id is not None and prereg_id is None:
        raise PreregistrationError("a look with a test id names its pre-registration id")
    if prereg_id is not None:
        doc, _ = _v1.load_prereg(prereg_id, prereg_dir=prereg_dir, rows_dir=rows_dir)
        if test_id is not None:
            ids = [t["test_id"] for t in doc["tests"]] if isinstance(doc["tests"], list) else []
            if test_id not in ids:
                raise PreregistrationError(f"test {test_id!r} is not in {prereg_id!r}")
    now = _now()
    rec = {"time": seeded_time or now, "logged": now, "wp": wp, "purpose": purpose,
           "test_id": test_id, "prereg_id": prereg_id, "units": sorted(map(str, units)),
           "data_scope": data_scope, "scope_windows": format_windows(scope_windows(data_scope)),
           "members": members, "seeded": seeded_time is not None, "note": note, "tool": TOOL}
    return _v1._append_chained(Path(prereg_dir) / "looks.jsonl", rec)


# ----------------------------------------------------------------------------- validation
def _external_dicts(doc: dict):
    if isinstance(doc.get("external_units"), dict):
        yield "external_units", doc["external_units"]
    for i, t in enumerate(doc["tests"] if isinstance(doc.get("tests"), list) else []):
        if isinstance(t.get("external_units"), dict):
            yield f"tests[{i}].external_units", t["external_units"]


def validate_prereg(doc: dict, units_registry=AUTO, prereg_dir=PREREG_DIR,
                    registries=None) -> list[str]:
    """``prereg2.validate_prereg`` plus: every external unit is declared with members
    (a dict with ``description`` and at least one of ids, cells, universal, window, none)."""
    e = _v2.validate_prereg(doc, units_registry)
    if any(x.startswith("missing field") for x in e):
        return e
    idx = membership_index(prereg_dir, registries)
    for where, ext in _external_dicts(doc):
        for u, v in ext.items():
            if idx.canon(u) in idx.members and idx.sources.get(idx.canon(u), "").endswith(
                    "units.json"):
                e.append(f"{where}: {u!r} is a registry unit; do not redeclare it")
                continue
            if not isinstance(v, dict) or "description" not in v:
                e.append(f"{where}: {u!r} needs a dict with 'description' and members "
                         f"(prereg3)")
                continue
            e += [f"{where}[{u!r}]: {x}" for x in member_errors(v)]
    return e


# ----------------------------------------------------------------------------- write / score
def write_prereg(doc: dict, *, prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS,
                 units_registry=AUTO, registries=None) -> tuple[Path, str]:
    """Validate, check against the look log (:data:`BLOCKING_RULE`), hash and write a
    pre-registration once; returns (path, sha256)."""
    reg_path = _v2._registry_path(units_registry)
    if registries is None and reg_path is not None and not (
            isinstance(units_registry, str) and units_registry == AUTO):
        registries = [reg_path]
    errs = validate_prereg(doc, reg_path, prereg_dir, registries)
    if errs:
        raise PreregistrationError("; ".join(errs))
    reg = _v2.load_registry(reg_path)
    d = Path(prereg_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{doc['id']}.json"
    if path.exists():
        raise PreregistrationError(f"{path} already exists")
    tests = doc["tests"] if isinstance(doc["tests"], list) else []
    existing = read_chained(d / "INDEX.jsonl")
    if any(r["id"] == doc["id"] for r in existing):
        raise PreregistrationError(f"pre-registration id {doc['id']!r} already exists")
    clash = sorted({t["test_id"] for t in tests} & {t for r in existing
                                                     for t in r.get("test_ids", [])})
    if clash:
        raise PreregistrationError(f"test ids {clash} already belong to a pre-registration; "
                                   f"a changed design needs new ids")
    idx = membership_index(d, registries)
    for _, ext in _external_dicts(doc):          # members of the new external units
        for u, v in ext.items():
            m = _declared_members(v)
            if m is not None and not idx.known(u):
                idx.members[idx.canon(u)] = m
    now_dt = _dt.datetime.now(_UTC)
    block = blocking_looks(tests, None, d, idx, test_time=now_dt)
    if doc["kind"] == "declaration" and isinstance(doc.get("fresh_units"), list):
        decl = {"test_id": "<declaration>", "data_scope": "*",
                "fresh_units": list(doc["fresh_units"])}
        block += blocking_looks([decl], None, d, idx, test_time=now_dt)
    if block:
        raise PreregistrationError(f"looks at the test data already exist: {block[:20]}")
    out = dict(doc)
    out.update({"schema": SCHEMA, "created": _now(), "tool": TOOL, "tool_version": TOOL_VERSION,
                "harness_version": HARNESS_VERSION, "blocking_rule": BLOCKING_RULE,
                "inputs": _v1._hash_inputs(doc["inputs"]),
                "scope_windows": {t["test_id"]: format_windows(scope_windows(t["data_scope"]))
                                  for t in tests},
                "unverifiable_prior_looks": unverifiable_looks(tests, None, d, idx)[:50]})
    if reg["path"] is not None:
        out["units_registry"] = {"path": rel_to_repo(reg["path"]), "sha256": reg["sha256"]}
    data = (json.dumps(out, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode()
    with open(path, "xb") as f:
        f.write(data)
    os.chmod(path, 0o444)
    sha = hashlib.sha256(data).hexdigest()
    _v1._append_chained(d / "INDEX.jsonl", {"id": doc["id"], "wp": doc["wp"],
                                            "kind": doc["kind"], "created": out["created"],
                                            "sha256": sha, "path": rel_to_repo(path),
                                            "tool": TOOL,
                                            "test_ids": [t["test_id"] for t in tests]})
    ledger.append_rows(doc["wp"], [{
        "number_id": f"{doc['wp']}.prereg.{doc['id']}", "source": rel_to_repo(path),
        "role": "input", "product_version": f"prereg-sha256:{sha}",
        "note": f"pre-registration ({doc['kind']}) written {out['created']} by {TOOL} "
                f"{TOOL_VERSION}"}], rows_dir=rows_dir)
    return path, sha


def assert_can_score(prereg_id: str, wp: str, *, test_id: str | None = None,
                     prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS, verify_inputs: bool = True,
                     log: bool = True, registries=None) -> dict:
    """Refuse to score unless the registered design is intact and predates every look at its
    test data under :data:`BLOCKING_RULE`. On success logs the scoring look(s) and returns
    ``{"prereg", "sha256", "tests", "inputs_verified", "post_registration_looks",
    "unverifiable_prior_looks"}``."""
    doc, sha = load_prereg(prereg_id, prereg_dir=prereg_dir, rows_dir=rows_dir)
    if doc["kind"] != "test":
        raise PreregistrationError(f"{prereg_id!r} is a {doc['kind']}, not a test")
    if wp != doc["wp"]:
        raise PreregistrationError(f"{prereg_id!r} was registered by WP {doc['wp']!r}, not "
                                   f"{wp!r}")
    tests = [t for t in doc["tests"] if test_id is None or t["test_id"] == test_id]
    if not tests:
        raise PreregistrationError(f"{prereg_id!r} has no test {test_id!r}")
    ur = doc.get("units_registry")
    if isinstance(ur, dict) and ur.get("path"):
        rp = from_repo(ur["path"])
        if not rp.exists() or sha256_file(rp) != ur["sha256"]:
            raise PreregistrationError(f"units registry {ur['path']} changed or is missing")
    idx = membership_index(prereg_dir, registries)
    gone = sorted(u for t in tests for u in t["fresh_units"] if u in idx.withdrawn)
    if gone:
        raise PreregistrationError(f"fresh units {gone} of {prereg_id!r} were withdrawn in a "
                                   f"units registry (replaced by "
                                   f"{[idx.withdrawn[u] for u in gone]})")
    block = blocking_looks(tests, doc["created"], prereg_dir, idx)
    if block:
        raise PreregistrationError(f"pre-registration {prereg_id!r} ({doc['created']}) "
                                   f"postdates looks at its test data: {block[:20]}")
    if verify_inputs:
        bad = changed_inputs(doc)
        if bad:
            raise PreregistrationError(f"registered inputs of {prereg_id!r} changed: {bad}")
    created = parse_time(doc["created"])
    post = []
    for lk in read_chained(Path(prereg_dir) / "looks.jsonl"):
        if parse_time(lk["time"]) < created:
            continue
        for t in tests:
            if lk.get("prereg_id") == prereg_id and lk.get("test_id") == t["test_id"]:
                continue
            h = look_hits(lk, t, idx)
            if h["overlap"] or h["unknown"]:
                post.append({"test_id": t["test_id"], "time": lk["time"], "wp": lk["wp"],
                             "purpose": lk["purpose"], "look_test_id": lk.get("test_id"),
                             "pairs": sorted(set(h["overlap"])),
                             "unknown_membership": sorted(set(h["unknown"]))})
    unver = unverifiable_looks(tests, doc["created"], prereg_dir, idx)
    if log:
        for t in tests:
            now = _now()
            _v1._append_chained(Path(prereg_dir) / "looks.jsonl", {
                "time": now, "logged": now, "wp": wp, "purpose": "score",
                "test_id": t["test_id"], "prereg_id": prereg_id,
                "units": sorted(map(str, t["fresh_units"])), "data_scope": t["data_scope"],
                "scope_windows": format_windows(scope_windows(t["data_scope"])),
                "members": {}, "seeded": False, "note": f"prereg sha256 {sha}", "tool": TOOL})
    return {"prereg": doc, "sha256": sha, "tests": [t["test_id"] for t in tests],
            "inputs_verified": bool(verify_inputs), "post_registration_looks": post,
            "unverifiable_prior_looks": unver}


# ----------------------------------------------------------------------------- audit
def audit(prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS, registries=None) -> dict:
    """Apply :data:`BLOCKING_RULE` to the existing log (read-only): registered tests whose
    scoring the rule refuses, tests the rule refuses but ``prereg2`` allowed, looks whose scope
    is unbounded or that ``prereg2`` read differently, per registry fresh unit the looks that
    overlap it by membership, and looks on units of unknown membership."""
    d = Path(prereg_dir)
    idx = membership_index(d, registries)
    idx_rows = read_chained(d / "INDEX.jsonl")
    looks = read_chained(d / "looks.jsonl")
    out: dict = {"index_records": len(idx_rows), "looks_records": len(looks),
                 "registries": idx.registries, "blocking_rule": BLOCKING_RULE}
    refused, newly = [], []
    for r in idx_rows:
        try:
            doc, _ = load_prereg(r["id"], prereg_dir=d, rows_dir=rows_dir)
        except PreregistrationError:
            continue
        if doc.get("kind") != "test" or not isinstance(doc.get("tests"), list):
            continue
        scored = {lk.get("test_id") for lk in looks
                  if lk.get("prereg_id") == r["id"] and lk.get("purpose") == "score"}
        for t in doc["tests"]:
            b3 = blocking_looks([t], doc["created"], d, idx)
            if not b3:
                continue
            b2 = _v2.blocking_looks([t], doc["created"], d, dict(
                (k, {"replaced_by": v}) for k, v in idx.withdrawn.items()))
            rec = {"prereg_id": r["id"], "wp": r["wp"], "test_id": t["test_id"],
                   "created": doc["created"], "scored": t["test_id"] in scored,
                   "n_prior_looks": len(b3), "prior_looks": b3[:10]}
            refused.append(rec)
            if not b2:
                newly.append(rec)
    out["tests_refused_under_prereg3"] = refused
    out["tests_refused_under_prereg3_not_prereg2"] = newly
    scopes = sorted({lk["data_scope"] for lk in looks})
    out["scopes"] = {s: {"prereg3": format_windows(scope_windows(s)),
                         "prereg2": str(_v2.scope_window(s))} for s in scopes}
    diff = []
    for s in scopes:
        w3, w2 = scope_windows(s), _v2.scope_window(s)
        w2l = None if w2 is None else [tuple(w2)]
        if w3 != w2l:
            diff.append({"scope": s, "prereg3": format_windows(w3),
                         "prereg2": format_windows(w2l)})
    out["scopes_read_differently_from_prereg2"] = diff
    fresh = sorted(u for u, r in idx.role.items() if r == "fresh" and u not in idx.withdrawn)
    per = {}
    for v in fresh:
        t = {"test_id": f"<{v}>", "data_scope": "*", "fresh_units": [v]}
        hits = []
        for lk in looks:
            h = look_hits(lk, t, idx)
            if h["overlap"]:
                hits.append({"look": _look_brief(lk), "pairs": sorted(set(h["overlap"])),
                             "eligible_to_block": bool(h["block"])})
        per[v] = {"declared": (idx.declared[v].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                               if v in idx.declared else None),
                  "n_overlapping_looks": len(hits),
                  "n_eligible_to_block": sum(x["eligible_to_block"] for x in hits),
                  "looks": hits[:20]}
    out["fresh_units"] = per
    unknown = {}
    for lk in looks:
        for u in lk.get("units") or []:
            if not idx.known(u):
                unknown.setdefault(idx.canon(u), 0)
                unknown[idx.canon(u)] += 1
    out["looks_on_units_of_unknown_membership"] = dict(sorted(unknown.items()))
    return out
