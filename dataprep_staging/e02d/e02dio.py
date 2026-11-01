"""Small IO helpers shared by the round-7 bands scripts (dataprep_staging/e02d): file hashes,
the registered-declaration check, write-once JSON and the deadline clock."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.paths import REPO  # noqa: E402  (the repository root; DATAPREP_REPO overrides)

X = "dataprep_staging/e02d"
UTC = dt.timezone.utc
#: E02d registers before this instant (the N3F-style December fallback precedent)
DEADLINE = dt.datetime(2026, 11, 30, 16, 0, tzinfo=UTC)


class Refused(RuntimeError):
    """A script refuses to run (registration, order, date or provenance rule)."""


class EngineeringFailure(Refused):
    """An input is inconsistent: no statistic is computed; fix, log and re-run."""


def now() -> dt.datetime:
    return dt.datetime.now(UTC)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_json(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def rec(p) -> dict:
    """{path (repo-relative), sha256} of a file."""
    from common.paths import rel_to_repo

    q = repo_path(p)
    return {"path": rel_to_repo(q), "sha256": sha256(q)}


def prereg_dir_default() -> Path:
    from common.paths import PREREG_DIR

    return PREREG_DIR


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
    bad = [f for f in frozen if fc.get(f) != sha256(REPO / f)]
    if bad:
        raise Refused(f"code changed since {prereg_id} was registered: {bad}")
    return doc, sha


def write_once(path: Path, obj) -> str:
    """Write JSON once (refuses an existing file); returns its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True, default=str)
            + "\n").encode()
    with open(path, "xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def jsonable(x):
    return json.loads(json.dumps(x, default=float))
