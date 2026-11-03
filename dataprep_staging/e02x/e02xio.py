"""Small IO helpers of round-8 rank 5 (dataprep_staging/e02x): file hashes, the registered-
declaration check, write-once JSON, the attempt log and the clock. A copy of the e02d pattern
(``e02d/e02dio.py``) with this directory's own repository handle, so tests can point it at a
fake repository without touching E02d's module."""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.paths import REPO  # noqa: E402  (the repository root; DATAPREP_REPO overrides)

X = "dataprep_staging/e02x"
UTC = dt.timezone.utc


class Refused(RuntimeError):
    """A script refuses to run (registration, order, date or provenance rule)."""


class EngineeringFailure(Refused):
    """An input is inconsistent: no statistic is computed; fix, log and re-run."""


def now() -> dt.datetime:
    return dt.datetime.now(UTC)


def utc_stamp() -> str:
    return now().strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def sha_lines(ids) -> str:
    """sha256 of the sorted unique ids, one per line, newline-terminated (GOV's definition)."""
    return hashlib.sha256("".join(f"{s}\n" for s in sorted(set(map(str, ids))))
                          .encode("utf-8")).hexdigest()


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def rel(p) -> str:
    q = Path(p).resolve()
    try:
        return q.relative_to(Path(REPO).resolve()).as_posix()
    except ValueError:
        return q.as_posix()


def rec(p) -> dict:
    """{path (repo-relative), sha256} of a file."""
    q = repo_path(p)
    return {"path": rel(q), "sha256": sha256(q)}


def prereg_dir_default() -> Path:
    return Path(REPO) / "results" / "dataprep" / "preregistration"


def index_record(prereg_id: str, prereg_dir=None) -> dict | None:
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg_dir_default()
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == prereg_id]
    return rows[-1] if rows else None


def registered(prereg_id: str, prereg_dir=None, frozen: tuple[str, ...] = ()) -> tuple[dict, str]:
    """(document, sha256) of a registered pre-registration; refuses unless it is in INDEX with
    the same file sha and records, under ``frozen_code``, the current sha256 of every file in
    ``frozen`` (repo-relative)."""
    d = Path(prereg_dir) if prereg_dir else prereg_dir_default()
    r = index_record(prereg_id, d)
    if r is None:
        raise Refused(f"{prereg_id} is not registered in {d}")
    path = d / f"{prereg_id}.json"
    if not path.is_file():
        raise Refused(f"{path} is missing")
    sha = sha256(path)
    if sha != r["sha256"]:
        raise Refused(f"{path} sha256 {sha} != INDEX {r['sha256']}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    fc = doc.get("frozen_code") or {}
    bad = [f for f in frozen if fc.get(f) != sha256(Path(REPO) / f)]
    if bad:
        raise Refused(f"code changed since {prereg_id} was registered: {bad}")
    return doc, sha


def write_once(path: Path, obj) -> str:
    """Write JSON once (open mode 'x': refuses an existing file); returns its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True, default=str)
            + "\n").encode()
    with open(path, "xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def append_attempt(path: Path, rec_: dict) -> dict:
    """Append one record to an attempts.jsonl (flock; never rewrites earlier lines)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": utc_stamp(), **rec_}
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    return row


def jsonable(x):
    return json.loads(json.dumps(x, default=float))
