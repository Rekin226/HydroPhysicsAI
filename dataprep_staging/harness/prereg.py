"""Pre-registration tool and look log (plan 4.5, WP A1).

A pre-registration is a JSON file ``results/dataprep/preregistration/<id>.json`` written by
:func:`write_prereg` BEFORE the registered test runs. It names the zones, priors, burn-in,
folds, fresh units, thresholds, gate scores and their weights, the primary model and the
input hashes, plus the interpretation fixed in advance. The tool

* fills ``created`` with the current UTC time (callers cannot back-date it), hashes every
  listed input, writes the file once (mode ``x``, then read-only) and records its sha256 in
  the WP's ledger fragment (``<wp>.prereg.<id>``, role ``input``) and in the hash-chained
  ``INDEX.jsonl``;
* refuses to write a pre-registration whose fresh test units already carry a look at the
  same test, or any unregistered scoring look (a look made without a pre-registration
  taints the unit for every later test);
* refuses to score (:func:`assert_can_score`) a design whose pre-registration postdates any
  such look, or whose file no longer matches its recorded sha256, and logs the scoring look.

Looks live in the hash-chained, append-only ``looks.jsonl``: one line per act of computing a
score of any candidate on a unit's test data (time, WP, test id, pre-registration id, units,
data scope). Design-time units are seeded with looks dated to the pilots, so they can never
serve as fresh validation units. A changed design is a new id; its test ids must be new, and
looks already made on its units by other tests are disclosed by :func:`assert_can_score`.
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path

from common import ledger
from common.hashing import sha256_file
from common.paths import LEDGER_ROWS, PREREG_DIR, from_repo, rel_to_repo

SCHEMA = "dataprep-prereg/1"
REQUIRED = ("id", "wp", "title", "kind", "tests", "zones", "priors", "burn_in", "folds",
            "fresh_units", "thresholds", "gate_scores", "primary_model", "inputs",
            "interpretation")
KINDS = ("test", "declaration")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
NA = "not_applicable"


class PreregistrationError(PermissionError):
    """A pre-registration is invalid, tampered with, or postdates a look at its test data."""


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _paths(prereg_dir) -> tuple[Path, Path, Path]:
    d = Path(prereg_dir)
    return d, d / "INDEX.jsonl", d / "looks.jsonl"


def not_applicable(reason: str) -> dict:
    """Value for a required field that does not apply to this test (a reason is required)."""
    if not reason:
        raise ValueError("a reason is required")
    return {NA: reason}


def _is_na(v) -> bool:
    return isinstance(v, dict) and set(v) == {NA} and bool(v[NA])


# ----------------------------------------------------------------------------- chained logs
def _append_chained(path: Path, rec: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
        prev = hashlib.sha256(lines[-1].encode()).hexdigest() if lines else None
        rec = dict(rec, prev=prev)
        line = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        f.seek(0, os.SEEK_END)
        f.write(line + "\n")
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)
    return rec


def read_chained(path: Path) -> list[dict]:
    """Records of a hash-chained log; raises if the chain is broken (an edited line)."""
    path = Path(path)
    if not path.exists():
        return []
    out, prev = [], None
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        rec = json.loads(ln)
        if rec.get("prev") != prev:
            raise PreregistrationError(f"{path}: hash chain broken before {rec}")
        prev = hashlib.sha256(ln.encode()).hexdigest()
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- looks
def record_look(units, data_scope: str, wp: str, purpose: str, *, test_id: str | None = None,
                prereg_id: str | None = None, prereg_dir=PREREG_DIR, seeded_time: str | None = None,
                note: str = "") -> dict:
    """Append a look: computing any score on the test data of ``units`` within
    ``data_scope`` (for example ``"heads:2012-01..2022-12"``; ``"*"`` = every scope).
    ``seeded_time`` back-dates a look to documented pilot evidence (purpose must be
    ``"design_time_seed"``)."""
    if seeded_time is not None and purpose != "design_time_seed":
        raise ValueError("seeded_time is only for purpose='design_time_seed'")
    if not units:
        raise ValueError("a look names at least one unit")
    _, _, looks = _paths(prereg_dir)
    rec = {"time": seeded_time or _now(), "logged": _now(), "wp": wp, "purpose": purpose,
           "test_id": test_id, "prereg_id": prereg_id, "units": sorted(map(str, units)),
           "data_scope": data_scope, "seeded": seeded_time is not None, "note": note}
    return _append_chained(looks, rec)


def _scope_match(a: str, b: str) -> bool:
    return a == "*" or b == "*" or a == b


def blocking_looks(tests: list[dict], before: str | None, prereg_dir=PREREG_DIR) -> list[dict]:
    """Looks that taint the fresh units of ``tests``: a look at the same test id, or an
    unregistered scoring look (``prereg_id`` None, including design-time seeds), on a fresh
    unit with a matching data scope, dated before ``before`` (None = any time)."""
    _, _, looks = _paths(prereg_dir)
    hits = []
    for lk in read_chained(looks):
        if before is not None and lk["time"] >= before:
            continue
        for t in tests:
            fresh = set(t.get("fresh_units") or [])
            if not fresh & set(lk["units"]):
                continue
            if not _scope_match(lk["data_scope"], t["data_scope"]):
                continue
            if lk.get("test_id") == t["test_id"] or lk.get("prereg_id") is None:
                hits.append({"look": lk, "test_id": t["test_id"],
                             "units": sorted(fresh & set(lk["units"]))})
    return hits


# ----------------------------------------------------------------------------- validation
def _known_units(units_registry) -> dict | None:
    if units_registry is None:
        return None
    with open(units_registry, encoding="utf-8") as f:
        reg = json.load(f)
    return {u["id"]: u for u in reg["units"]}


def validate_prereg(doc: dict, units_registry=None) -> list[str]:
    """Schema errors of a pre-registration document (empty = valid)."""
    e = []
    for k in REQUIRED:
        if k not in doc:
            e.append(f"missing field {k!r} (use not_applicable(reason) if it does not apply)")
    if e:
        return e
    if "created" in doc:
        e.append("'created' is set by the tool; remove it")
    if not (isinstance(doc["id"], str) and _ID.match(doc["id"])):
        e.append(f"bad id {doc['id']!r}")
    if doc["kind"] not in KINDS:
        e.append(f"kind must be one of {KINDS}")
    for k in REQUIRED:
        v = doc[k]
        if v is None or (isinstance(v, (str, list, dict)) and not v):
            e.append(f"field {k!r} is empty (use not_applicable(reason))")
    known = _known_units(units_registry)
    tests = doc["tests"]
    if doc["kind"] == "test":
        if _is_na(tests) or not isinstance(tests, list):
            e.append("a test pre-registration lists its tests")
            tests = []
        gs = doc["gate_scores"]
        if not _is_na(gs):
            if not isinstance(gs, list) or not gs:
                e.append("gate_scores must be a list of {name, weight, ...}")
            else:
                w = [g.get("weight") for g in gs]
                if any(not isinstance(x, (int, float)) for x in w):
                    e.append("every gate score needs a numeric weight")
                elif abs(sum(w) - 1.0) > 1e-9:
                    e.append(f"gate-score weights sum to {sum(w)}, not 1")
        if _is_na(doc["primary_model"]):
            e.append("a test pre-registration names its primary model or design")
    elif not _is_na(tests) and not isinstance(tests, list):
        e.append("tests must be a list or not_applicable")
    seen = set()
    for i, t in enumerate(tests if isinstance(tests, list) else []):
        for k in ("test_id", "data_scope", "fresh_units"):
            if k not in t:
                e.append(f"tests[{i}] missing {k!r}")
        tid = t.get("test_id")
        if tid in seen:
            e.append(f"duplicate test id {tid!r}")
        seen.add(tid)
        if not (isinstance(tid, str) and _ID.match(tid or "")):
            e.append(f"tests[{i}] bad test_id {tid!r}")
        fu = t.get("fresh_units") or []
        if not fu:
            e.append(f"tests[{i}] scores on no fresh unit")
        if known is not None:
            ext = t.get("external_units") or {}
            for u in list(fu) + list(t.get("design_time_units") or []):
                if u not in known and u not in ext:
                    e.append(f"tests[{i}] unit {u!r} is neither in the units registry nor "
                             f"declared in external_units")
            for u in fu:
                if u in known and known[u].get("role") != "fresh":
                    e.append(f"tests[{i}] unit {u!r} has role {known[u].get('role')!r} in the "
                             f"registry; it cannot be a fresh unit")
    inp = doc["inputs"]
    if not _is_na(inp) and not isinstance(inp, list):
        e.append("inputs must be a list of paths or {path, sha256}")
    return e


def _hash_inputs(inputs) -> list[dict]:
    if _is_na(inputs):
        return inputs
    out = []
    for x in inputs:
        rec = dict(x) if isinstance(x, dict) else {"path": str(x)}
        p = from_repo(rec["path"])
        got = sha256_file(p)
        if "sha256" in rec and rec["sha256"] != got:
            raise PreregistrationError(f"input {rec['path']} sha256 {got} != declared "
                                       f"{rec['sha256']}")
        rec.update({"path": rel_to_repo(p), "sha256": got, "bytes": p.stat().st_size})
        out.append(rec)
    return out


# ----------------------------------------------------------------------------- write / load
def write_prereg(doc: dict, *, prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS,
                 units_registry=None) -> tuple[Path, str]:
    """Validate, hash and write a pre-registration; returns (path, sha256)."""
    errs = validate_prereg(doc, units_registry)
    if errs:
        raise PreregistrationError("; ".join(errs))
    d, index, _ = _paths(prereg_dir)
    d.mkdir(parents=True, exist_ok=True)
    tests = doc["tests"] if isinstance(doc["tests"], list) else []
    existing = read_chained(index)
    used = {t for r in existing for t in r.get("test_ids", [])}
    clash = sorted({t["test_id"] for t in tests} & used)
    if clash:
        raise PreregistrationError(f"test ids {clash} already belong to a pre-registration; "
                                   f"a changed design needs new ids")
    if any(r["id"] == doc["id"] for r in existing):
        raise PreregistrationError(f"pre-registration id {doc['id']!r} already exists")
    block = blocking_looks(tests, None, prereg_dir)
    if block:
        raise PreregistrationError(f"looks at the test data already exist: {block}")
    from harness import __version__
    out = dict(doc)
    out["schema"] = SCHEMA
    out["created"] = _now()
    out["harness_version"] = __version__
    out["inputs"] = _hash_inputs(doc["inputs"])
    if units_registry is not None:
        out["units_registry"] = {"path": rel_to_repo(units_registry),
                                 "sha256": sha256_file(units_registry)}
    path = d / f"{doc['id']}.json"
    data = (json.dumps(out, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode()
    with open(path, "xb") as f:
        f.write(data)
    os.chmod(path, 0o444)
    sha = hashlib.sha256(data).hexdigest()
    _append_chained(index, {"id": doc["id"], "wp": doc["wp"], "kind": doc["kind"],
                            "created": out["created"], "sha256": sha,
                            "path": rel_to_repo(path),
                            "test_ids": [t["test_id"] for t in tests]})
    ledger.append_rows(doc["wp"], [{
        "number_id": f"{doc['wp']}.prereg.{doc['id']}", "source": rel_to_repo(path),
        "role": "input", "product_version": f"prereg-sha256:{sha}",
        "note": f"pre-registration ({doc['kind']}) written {out['created']} by the A1 tool"}],
        rows_dir=rows_dir)
    return path, sha


def load_prereg(prereg_id: str, *, prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS
                ) -> tuple[dict, str]:
    """Load a pre-registration and verify its sha256 against INDEX.jsonl and the ledger."""
    d, index, _ = _paths(prereg_dir)
    path = d / f"{prereg_id}.json"
    if not path.exists():
        raise PreregistrationError(f"no pre-registration {prereg_id!r}")
    sha = sha256_file(path)
    rec = [r for r in read_chained(index) if r["id"] == prereg_id]
    if len(rec) != 1:
        raise PreregistrationError(f"{prereg_id!r} is not (uniquely) in {index}")
    if rec[0]["sha256"] != sha:
        raise PreregistrationError(f"{path} changed: sha256 {sha} != recorded "
                                   f"{rec[0]['sha256']}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in ledger.read_all(rows_dir)
            if r["number_id"] == f"{doc['wp']}.prereg.{prereg_id}"]
    if not any(r["product_version"] == f"prereg-sha256:{sha}" for r in rows):
        raise PreregistrationError(f"no ledger row records sha256 {sha} for {prereg_id!r}")
    return doc, sha


def assert_can_score(prereg_id: str, wp: str, *, test_id: str | None = None,
                     prereg_dir=PREREG_DIR, rows_dir=LEDGER_ROWS, log: bool = True) -> dict:
    """Refuse to score unless the pre-registration exists, is unchanged, and predates every
    look at its test data (same test id, or unregistered scoring, on its fresh units). On
    success logs the scoring look and returns ``{"prereg", "sha256", "tests",
    "other_tests_prior_looks"}`` (looks by other tests on the same units, for disclosure)."""
    doc, sha = load_prereg(prereg_id, prereg_dir=prereg_dir, rows_dir=rows_dir)
    if doc["kind"] != "test":
        raise PreregistrationError(f"{prereg_id!r} is a {doc['kind']}, not a test")
    tests = [t for t in doc["tests"] if test_id is None or t["test_id"] == test_id]
    if not tests:
        raise PreregistrationError(f"{prereg_id!r} has no test {test_id!r}")
    block = blocking_looks(tests, doc["created"], prereg_dir)
    if block:
        raise PreregistrationError(f"pre-registration {prereg_id!r} ({doc['created']}) "
                                   f"postdates looks at its test data: {block}")
    _, _, looks = _paths(prereg_dir)
    other = []
    for lk in read_chained(looks):
        for t in tests:
            if (set(t["fresh_units"]) & set(lk["units"]) and lk.get("test_id") != t["test_id"]
                    and lk.get("prereg_id") is not None):
                other.append({"test_id": lk.get("test_id"), "time": lk["time"],
                              "units": sorted(set(t["fresh_units"]) & set(lk["units"]))})
    if log:
        for t in tests:
            record_look(t["fresh_units"], t["data_scope"], wp, "score", test_id=t["test_id"],
                        prereg_id=prereg_id, prereg_dir=prereg_dir)
    return {"prereg": doc, "sha256": sha, "tests": [t["test_id"] for t in tests],
            "other_tests_prior_looks": other}
