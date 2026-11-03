"""Small IO helpers of round-8 rank 16's FB part (dataprep_staging/fb_family): file hashes,
the registered-declaration check, write-once JSON, the attempt log and the clock. A copy of the
e02d/e02x pattern with this directory's own repository handle (``common.paths.REPO``, which
``DATAPREP_REPO`` overrides), so tests can point it at a fake repository."""
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

from common.paths import REPO  # noqa: E402

X = "dataprep_staging/fb_family"
UTC = dt.timezone.utc


class Refused(RuntimeError):
    """A script refuses to run (registration, order, date or provenance rule)."""


class EngineeringFailure(Refused):
    """An input is inconsistent: no statistic is computed; fix, log and re-run."""


def now() -> dt.datetime:
    return dt.datetime.now(UTC)


def utc_stamp() -> str:
    return now().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_json(obj) -> str:
    """sha256 of the canonical JSON text (sorted keys, indent 1, trailing newline)."""
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def repo_path(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(REPO) / p


def prereg_dir_default() -> Path:
    return Path(REPO) / "results" / "dataprep" / "preregistration"


def index_record(prereg_id: str, prereg_dir=None) -> dict | None:
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg_dir_default()
    rows = [r for r in prereg3.read_chained(d / "INDEX.jsonl") if r["id"] == prereg_id]
    return rows[-1] if rows else None


def registered(prereg_id: str, prereg_dir=None) -> tuple[dict, str]:
    """(document, sha256) of a registered pre-registration; refuses unless it is listed in
    INDEX.jsonl with the same file sha256."""
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
    return json.loads(path.read_text(encoding="utf-8")), sha


def write_once(path: Path, obj) -> str:
    """Write JSON once (open mode 'x': refuses an existing file); returns its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=True, default=str)
            + "\n").encode()
    with open(path, "xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def append_attempt(path: Path, rec: dict) -> dict:
    """Append one record to an attempts.jsonl (flock; never rewrites earlier lines)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": utc_stamp(), **rec}
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)
    return row


def jsonable(x):
    return json.loads(json.dumps(x, default=float))
