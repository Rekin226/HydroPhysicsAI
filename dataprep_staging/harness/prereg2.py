"""Pre-registration tool, repaired (WP A1 repair round, 2026-10-12): ``harness.prereg2``.

Use this module for every new pre-registration, look and score. ``harness.prereg`` stays
frozen because two products hash it (county_electricity/census_check v1, surface_water_v4
full/v2); both tools share the same files (``INDEX.jsonl``, ``looks.jsonl``, the ledger), and
records written by either are read by both.

Plan WP A1: the tool "refuses to score a design whose pre-registration postdates any look at
its test data". The first tool enforced less (verifier probes, repair round): a look blocked a
test only if it carried the same test id or no pre-registration id, scopes had to be equal
strings, and ``record_look`` accepted any pre-registration id. This tool enforces the plan's
rule:

* **Any look blocks.** A look on a fresh unit of a test blocks that test if it is dated before
  the test's pre-registration, whatever its purpose, test id or pre-registration id (a design
  registered after another design was scored on the same unit postdates a look at its test
  data). Looks after the registration (re-scoring the same registered test, diagnostics) do
  not block that registration; they are disclosed.
* **Scopes overlap unless provably disjoint.** Dataset names are not compared (the same data
  travel under different names: ``leveling``, ``leveling_elevation``); two scopes are disjoint
  only if both carry a time window (``YYYY-MM..YYYY-MM``, ``YYYY..YYYY``, or one ``YYYY[-MM]``)
  and the windows do not intersect. ``"*"`` and scopes without a window overlap everything.
* **Units are the registry's.** The latest ``validation_units`` registry is loaded by default:
  a fresh unit must have role ``fresh`` there, a withdrawn id (v2: the leveling folds) is
  refused everywhere, and every other unit must be declared in ``external_units``. A
  declaration of fresh units is refused if any look names one of them.
* **Looks are validated.** ``record_look`` refuses a pre-registration id that is not in
  ``INDEX.jsonl`` (or whose file changed), a test id outside that pre-registration, a test id
  without one, and the purpose ``score`` (only :func:`assert_can_score` logs scoring looks).
  Back-dated looks are only ``design_time_seed`` looks without a pre-registration id.
* **The design is the registered one.** :func:`assert_can_score` refuses unless the file
  matches its recorded sha256 (INDEX and ledger), the scoring WP is the registering WP, the
  units registry is unchanged, and (by default) every hashed input still has its registered
  sha256; then it logs the scoring look.

Times are compared as UTC datetimes, not strings. The look log stays self-reported: a look that
is never logged cannot be detected.
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
from harness.prereg import (  # noqa: F401 - re-exported
    KINDS,
    NA,
    REQUIRED,
    PreregistrationError,
    not_applicable,
    read_chained,
)

TOOL = "harness.prereg2"
TOOL_VERSION = "2.0.0"
SCHEMA = "dataprep-prereg/2"
SEED_PURPOSE = "design_time_seed"
RESERVED_PURPOSES = frozenset({"score"})
AUTO = "auto"
BLOCKING_RULE = (
    "a look on a fresh unit of a test blocks the test if it is dated before the test's "
    "pre-registration (any purpose, test id or pre-registration id) and its data scope "
    "overlaps the test's (scopes overlap unless both carry time windows that do not "
    "intersect; dataset names are not compared)")
_UTC = _dt.timezone.utc
_RANGE = re.compile(r"(?<![0-9A-Za-z])((?:19|20)\d{2})(?:-(0[1-9]|1[0-2]))?\.\."
                    r"((?:19|20)\d{2})(?:-(0[1-9]|1[0-2]))?(?![0-9A-Za-z])")
_SINGLE = re.compile(r"(?<![0-9A-Za-z.])((?:19|20)\d{2})(?:-(0[1-9]|1[0-2]))?"
                     r"(?![0-9A-Za-z.])")


# ----------------------------------------------------------------------------- time, scope
def _now() -> str:
    return _dt.datetime.now(_UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_time(s: str) -> _dt.datetime:
    """UTC datetime of an ISO-8601 time string (``Z`` or an offset; naive = UTC)."""
    t = str(s).strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    d = _dt.datetime.fromisoformat(t)
    if d.tzinfo is None:
        d = d.replace(tzinfo=_UTC)
    return d.astimezone(_UTC)


def scope_window(scope: str) -> tuple[int, int] | None:
    """(first, last) month index (12 * year + month - 1) of the time window of a data scope,
    or None when the scope carries none (``"*"``, no year, several single years, inverted)."""
    if scope is None or str(scope).strip() in ("", "*"):
        return None
    s = str(scope)
    m = _RANGE.search(s)
    if m:
        y0, m0, y1, m1 = m.groups()
        a = int(y0) * 12 + (int(m0) - 1 if m0 else 0)
        b = int(y1) * 12 + (int(m1) - 1 if m1 else 11)
        return (a, b) if a <= b else None
    singles = _SINGLE.findall(s)
    if len(singles) == 1:
        y, mm = singles[0]
        if mm:
            a = int(y) * 12 + int(mm) - 1
            return a, a
        return int(y) * 12, int(y) * 12 + 11
    return None


def scopes_overlap(a: str, b: str) -> bool:
    """True unless both scopes carry time windows that do not intersect (see module doc)."""
    wa, wb = scope_window(a), scope_window(b)
    if wa is None or wb is None:
        return True
    return wa[0] <= wb[1] and wb[0] <= wa[1]


# ----------------------------------------------------------------------------- registry
def latest_units_registry() -> Path | None:
    """``units.json`` of the highest ``validation_units/v<N>`` (None if there is none)."""
    base = DATAPREP / "validation_units"
    if not base.is_dir():
        return None
    vs = sorted((int(p.name[1:]), p) for p in base.iterdir()
                if p.is_dir() and re.fullmatch(r"v\d+", p.name) and (p / "units.json").exists())
    return vs[-1][1] / "units.json" if vs else None


def _registry_path(units_registry):
    if units_registry is None:
        return None
    if isinstance(units_registry, str) and units_registry == AUTO:
        return latest_units_registry()
    return Path(units_registry)


def load_registry(units_registry) -> dict:
    """{"path", "sha256", "units" {id: unit}, "withdrawn" {old id: record}} (empty if None)."""
    p = _registry_path(units_registry)
    if p is None:
        return {"path": None, "sha256": None, "units": {}, "withdrawn": {}}
    reg = json.loads(Path(p).read_text(encoding="utf-8"))
    return {"path": p, "sha256": sha256_file(p), "units": {u["id"]: u for u in reg["units"]},
            "withdrawn": dict(reg.get("withdrawn") or {})}


def _canon(units, withdrawn: dict) -> set:
    return {str(withdrawn.get(u, {}).get("replaced_by", u)) for u in units}


# ----------------------------------------------------------------------------- looks
def _looks_path(prereg_dir) -> Path:
    return Path(prereg_dir) / "looks.jsonl"


def _index_path(prereg_dir) -> Path:
    return Path(prereg_dir) / "INDEX.jsonl"


def looks_on(units, prereg_dir=PREREG_DIR, withdrawn: dict | None = None) -> list[dict]:
    """Every logged look that names one of ``units`` (withdrawn ids resolved)."""
    wd = withdrawn or {}
    want = _canon(units, wd)
    return [lk for lk in read_chained(_looks_path(prereg_dir))
            if _canon(lk["units"], wd) & want]


def blocking_looks(tests: list[dict], before: str | None, prereg_dir=PREREG_DIR,
                   withdrawn: dict | None = None) -> list[dict]:
    """Looks that taint the fresh units of ``tests`` under the repaired rule: any look on a
    fresh unit, dated before ``before`` (None = any time), whose data scope overlaps the
    test's (:func:`scopes_overlap`)."""
    wd = withdrawn or {}
    cut = parse_time(before) if before is not None else None
    hits = []
    for lk in read_chained(_looks_path(prereg_dir)):
        if cut is not None and parse_time(lk["time"]) >= cut:
            continue
        lu = _canon(lk["units"], wd)
        for t in tests:
            common = _canon(t.get("fresh_units") or [], wd) & lu
            if not common or not scopes_overlap(lk["data_scope"], t["data_scope"]):
                continue
            hits.append({"test_id": t["test_id"], "units": sorted(common),
                         "look": {k: lk.get(k) for k in ("time", "wp", "purpose", "test_id",
                                                         "prereg_id", "data_scope")}})
    return hits


def _index_ids(prereg_dir) -> dict:
    return {r["id"]: r for r in read_chained(_index_path(prereg_dir))}


def record_look(units, data_scope: str, wp: str, purpose: str, *, test_id: str | None = None,
                prereg_id: str | None = None, seeded_time: str | None = None, note: str = "",
                prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS) -> dict:
    """Append a look: computing any score or diagnostic of any candidate on the test data of
    ``units`` within ``data_scope`` (for example ``"heads:2012-01..2022-12"``; ``"*"`` = every
    scope). Validated as described in the module docstring."""
    if purpose in RESERVED_PURPOSES:
        raise PreregistrationError("scoring looks are logged by assert_can_score; use another "
                                   "purpose (e.g. 'diagnose', 'unregistered_score')")
    if not purpose or not isinstance(purpose, str):
        raise ValueError("a look has a purpose")
    if not units:
        raise ValueError("a look names at least one unit")
    if not isinstance(data_scope, str) or not data_scope.strip():
        raise ValueError("a look has a data scope ('*' = every scope)")
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
           "data_scope": data_scope, "seeded": seeded_time is not None, "note": note,
           "tool": TOOL}
    return _v1._append_chained(_looks_path(prereg_dir), rec)


# ----------------------------------------------------------------------------- validation
def _unit_lists(doc: dict):
    tests = doc["tests"] if isinstance(doc.get("tests"), list) else []
    for i, t in enumerate(tests):
        yield f"tests[{i}].fresh_units", list(t.get("fresh_units") or [])
        yield f"tests[{i}].design_time_units", list(t.get("design_time_units") or [])
        yield f"tests[{i}].external_units", list((t.get("external_units") or {}).keys())
    fu = doc.get("fresh_units")
    if isinstance(fu, list):
        yield "fresh_units", fu
    yield "external_units", list((doc.get("external_units") or {}).keys())


def validate_prereg(doc: dict, units_registry=AUTO) -> list[str]:
    """Schema and unit errors of a pre-registration (empty = valid). The first tool's checks
    plus: withdrawn unit ids are refused anywhere; the fresh units of a declaration must have
    role 'fresh' in the registry or be declared in its top-level ``external_units``."""
    reg_path = _registry_path(units_registry)
    e = _v1.validate_prereg(doc, reg_path)
    if any(x.startswith("missing field") for x in e):
        return e
    reg = load_registry(reg_path)
    for where, ids in _unit_lists(doc):
        for u in ids:
            if u in reg["withdrawn"]:
                w = reg["withdrawn"][u]
                e.append(f"{where}: unit {u!r} was withdrawn in {rel_to_repo(reg['path'])} "
                         f"(replaced by {w.get('replaced_by')!r}: {w.get('reason', '')})")
    if doc.get("kind") == "declaration" and isinstance(doc.get("fresh_units"), list):
        ext = doc.get("external_units") or {}
        for u in doc["fresh_units"]:
            if u in reg["units"]:
                if reg["units"][u].get("role") != "fresh":
                    e.append(f"fresh_units: unit {u!r} has role "
                             f"{reg['units'][u].get('role')!r} in the registry")
            elif reg["path"] is not None and u not in ext:
                e.append(f"fresh_units: unit {u!r} is neither in the units registry nor "
                         f"declared in external_units")
    return e


# ----------------------------------------------------------------------------- write / load
def write_prereg(doc: dict, *, prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS,
                 units_registry=AUTO) -> tuple[Path, str]:
    """Validate, check against the look log, hash and write a pre-registration once; returns
    (path, sha256). ``units_registry='auto'`` = the latest validation_units registry."""
    reg_path = _registry_path(units_registry)
    errs = validate_prereg(doc, reg_path)
    if errs:
        raise PreregistrationError("; ".join(errs))
    reg = load_registry(reg_path)
    d = Path(prereg_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{doc['id']}.json"
    if path.exists():
        raise PreregistrationError(f"{path} already exists")
    tests = doc["tests"] if isinstance(doc["tests"], list) else []
    existing = read_chained(_index_path(d))
    if any(r["id"] == doc["id"] for r in existing):
        raise PreregistrationError(f"pre-registration id {doc['id']!r} already exists")
    clash = sorted({t["test_id"] for t in tests} & {t for r in existing
                                                     for t in r.get("test_ids", [])})
    if clash:
        raise PreregistrationError(f"test ids {clash} already belong to a pre-registration; "
                                   f"a changed design needs new ids")
    block = blocking_looks(tests, None, d, reg["withdrawn"])
    if doc["kind"] == "declaration" and isinstance(doc.get("fresh_units"), list):
        prior = looks_on(doc["fresh_units"], d, reg["withdrawn"])
        block += [{"declared_fresh_unit_look": {k: lk.get(k) for k in (
            "time", "wp", "purpose", "units", "data_scope")}} for lk in prior]
    if block:
        raise PreregistrationError(f"looks at the test data already exist: {block}")
    out = dict(doc)
    out.update({"schema": SCHEMA, "created": _now(), "tool": TOOL, "tool_version": TOOL_VERSION,
                "harness_version": HARNESS_VERSION, "blocking_rule": BLOCKING_RULE,
                "inputs": _v1._hash_inputs(doc["inputs"]),
                "scope_windows": {t["test_id"]: (scope_window(t["data_scope"])
                                                 or "none: overlaps every scope")
                                  for t in tests}})
    if reg["path"] is not None:
        out["units_registry"] = {"path": rel_to_repo(reg["path"]), "sha256": reg["sha256"]}
    data = (json.dumps(out, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode()
    with open(path, "xb") as f:
        f.write(data)
    os.chmod(path, 0o444)
    sha = hashlib.sha256(data).hexdigest()
    _v1._append_chained(_index_path(d), {"id": doc["id"], "wp": doc["wp"], "kind": doc["kind"],
                                         "created": out["created"], "sha256": sha,
                                         "path": rel_to_repo(path), "tool": TOOL,
                                         "test_ids": [t["test_id"] for t in tests]})
    ledger.append_rows(doc["wp"], [{
        "number_id": f"{doc['wp']}.prereg.{doc['id']}", "source": rel_to_repo(path),
        "role": "input", "product_version": f"prereg-sha256:{sha}",
        "note": f"pre-registration ({doc['kind']}) written {out['created']} by {TOOL} "
                f"{TOOL_VERSION}"}], rows_dir=rows_dir)
    return path, sha


def load_prereg(prereg_id: str, *, prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS
                ) -> tuple[dict, str]:
    """Load a pre-registration (either tool's) and verify its sha256 against INDEX.jsonl and
    the ledger."""
    return _v1.load_prereg(prereg_id, prereg_dir=prereg_dir, rows_dir=rows_dir)


def changed_inputs(doc: dict) -> list[dict]:
    """Hashed inputs of a pre-registration whose file is missing or no longer matches."""
    out = []
    inp = doc.get("inputs")
    if not isinstance(inp, list):
        return out
    for rec in inp:
        if not isinstance(rec, dict) or "sha256" not in rec:
            continue
        p = from_repo(rec["path"])
        if not p.exists():
            out.append({"path": rec["path"], "problem": "missing"})
        elif sha256_file(p) != rec["sha256"]:
            out.append({"path": rec["path"], "problem": "sha256 changed"})
    return out


def assert_can_score(prereg_id: str, wp: str, *, test_id: str | None = None,
                     prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS, verify_inputs: bool = True,
                     log: bool = True) -> dict:
    """Refuse to score unless the registered design is intact and predates every look at its
    test data (module docstring). On success logs the scoring look(s) and returns
    ``{"prereg", "sha256", "tests", "inputs_verified", "post_registration_looks"}`` (looks on
    the same units after the registration, for disclosure)."""
    doc, sha = load_prereg(prereg_id, prereg_dir=prereg_dir, rows_dir=rows_dir)
    if doc["kind"] != "test":
        raise PreregistrationError(f"{prereg_id!r} is a {doc['kind']}, not a test")
    if wp != doc["wp"]:
        raise PreregistrationError(f"{prereg_id!r} was registered by WP {doc['wp']!r}, not "
                                   f"{wp!r}")
    tests = [t for t in doc["tests"] if test_id is None or t["test_id"] == test_id]
    if not tests:
        raise PreregistrationError(f"{prereg_id!r} has no test {test_id!r}")
    withdrawn = {}
    ur = doc.get("units_registry")
    if isinstance(ur, dict) and ur.get("path"):
        rp = from_repo(ur["path"])
        if not rp.exists() or sha256_file(rp) != ur["sha256"]:
            raise PreregistrationError(f"units registry {ur['path']} changed or is missing")
        withdrawn = load_registry(rp)["withdrawn"]
    latest = load_registry(AUTO)["withdrawn"]
    gone = sorted({u for t in tests for u in t["fresh_units"]} & set(latest))
    if gone:
        raise PreregistrationError(f"fresh units {gone} of {prereg_id!r} were withdrawn in the "
                                   f"latest units registry: {[latest[u] for u in gone]}")
    withdrawn = {**latest, **withdrawn}
    block = blocking_looks(tests, doc["created"], prereg_dir, withdrawn)
    if block:
        raise PreregistrationError(f"pre-registration {prereg_id!r} ({doc['created']}) "
                                   f"postdates looks at its test data: {block}")
    if verify_inputs:
        bad = changed_inputs(doc)
        if bad:
            raise PreregistrationError(f"registered inputs of {prereg_id!r} changed: {bad}")
    created = parse_time(doc["created"])
    post = []
    for lk in read_chained(_looks_path(prereg_dir)):
        if parse_time(lk["time"]) < created:
            continue
        for t in tests:
            common = _canon(t["fresh_units"], withdrawn) & _canon(lk["units"], withdrawn)
            if common and not (lk.get("prereg_id") == prereg_id
                               and lk.get("test_id") == t["test_id"]):
                post.append({"test_id": t["test_id"], "time": lk["time"], "wp": lk["wp"],
                             "purpose": lk["purpose"], "look_test_id": lk.get("test_id"),
                             "units": sorted(common)})
    if log:
        for t in tests:
            now = _now()
            _v1._append_chained(_looks_path(prereg_dir), {
                "time": now, "logged": now, "wp": wp, "purpose": "score",
                "test_id": t["test_id"], "prereg_id": prereg_id,
                "units": sorted(map(str, t["fresh_units"])), "data_scope": t["data_scope"],
                "seeded": False, "note": f"prereg sha256 {sha}", "tool": TOOL})
    return {"prereg": doc, "sha256": sha, "tests": [t["test_id"] for t in tests],
            "inputs_verified": bool(verify_inputs), "post_registration_looks": post}


# ----------------------------------------------------------------------------- audit
def audit(prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS, registries=()) -> dict:
    """Apply the repaired rules to the existing log (read-only).

    Returns: registered tests whose scoring looks the repaired rule would have refused (looks
    on their fresh units, dated before the registration, overlapping scope), looks naming a
    pre-registration id that is not in INDEX.jsonl, files in ``prereg_dir`` that are not in
    INDEX.jsonl, tests whose hashed inputs changed since registration, and, per units
    registry, the looks on its fresh units and on its withdrawn ids."""
    d = Path(prereg_dir)
    idx = read_chained(_index_path(d))
    looks = read_chained(_looks_path(d))
    ids = {r["id"] for r in idx}
    out: dict = {"index_records": len(idx), "looks_records": len(looks), "chains_intact": True}
    refused, changed, unloadable = [], [], []
    for r in idx:
        try:
            doc, _ = load_prereg(r["id"], prereg_dir=d, rows_dir=rows_dir)
        except PreregistrationError as exc:
            unloadable.append({"id": r["id"], "error": str(exc)[:300]})
            continue
        ch = changed_inputs(doc)
        if ch:
            changed.append({"id": r["id"], "wp": r["wp"], "inputs": ch})
        if doc.get("kind") != "test" or not isinstance(doc.get("tests"), list):
            continue
        scored = {lk.get("test_id") for lk in looks
                  if lk.get("prereg_id") == r["id"] and lk.get("purpose") == "score"}
        for t in doc["tests"]:
            b = blocking_looks([t], doc["created"], d)
            if b:
                refused.append({"prereg_id": r["id"], "wp": r["wp"], "test_id": t["test_id"],
                                "created": doc["created"], "scored": t["test_id"] in scored,
                                "n_prior_looks": len(b),
                                "prior_looks": [x["look"] for x in b][:10]})
    out["tests_refused_under_repaired_rule"] = refused
    out["tests_with_changed_inputs"] = changed
    out["index_records_not_loadable"] = unloadable
    out["looks_with_unknown_prereg_id"] = [
        {k: lk.get(k) for k in ("time", "logged", "wp", "purpose", "test_id", "prereg_id")}
        for lk in looks if lk.get("prereg_id") is not None and lk["prereg_id"] not in ids]
    files = sorted(p.stem for p in d.glob("*.json"))
    out["prereg_files_not_in_index"] = [f for f in files if f not in ids]
    out["registries"] = {}
    for rp in registries:
        reg = load_registry(rp)
        fresh = sorted(u for u, x in reg["units"].items() if x.get("role") == "fresh")
        lf = {}
        for lk in looks:
            for u in set(lk["units"]) & set(fresh):
                lf.setdefault(u, []).append({k: lk.get(k) for k in (
                    "time", "logged", "wp", "purpose", "test_id", "prereg_id", "data_scope")})
        lw = {}
        for lk in looks:
            for u in set(lk["units"]) & set(reg["withdrawn"]):
                lw.setdefault(u, []).append({k: lk.get(k) for k in (
                    "time", "wp", "purpose", "test_id", "prereg_id", "data_scope")})
        out["registries"][rel_to_repo(rp)] = {
            "sha256": reg["sha256"], "n_fresh": len(fresh),
            "fresh_units_with_looks": lf, "withdrawn_units_with_looks": lw}
    return out
