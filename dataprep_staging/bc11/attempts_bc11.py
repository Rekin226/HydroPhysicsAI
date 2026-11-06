"""The attempt log of I1.backcast_2010_11.v1 (``results/twin_runs/chx/backcast_2010_11/
attempts.jsonl``): every output-writing Python entry point appends its OWN row through
``run_step`` -- time, step, exit code, the sha256 of each declared output file as it stands
when the step ends, and the sha256 of every frozen code file. Rows are only appended.

The declaration builder (``prereg_bc11.bindings``) needs, per step, exactly one rc 0 row
whose output shas equal the files now on disk: an output deleted and rerun leaves a second
rc 0 row or a sha mismatch, and either refuses. Every row is disclosed in the declaration.
Nothing here reads a head value.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent / "ch_x"), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bc11_const as K  # noqa: E402
import chx_io as IO  # noqa: E402


def log(step: str, rc: int, outputs=(), **extra) -> dict:
    """Append one attempt row (outputs: repository-relative paths)."""
    row = {"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "step": step, "rc": int(rc),
           "outputs": {str(p): IO.sha_or_missing(IO.rp(p)) for p in outputs},
           "code_sha256": K.code_shas(), **extra}
    p = IO.rp(K.ATTEMPTS)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def run_step(step: str, fn, outputs=()) -> int:
    """Run ``fn()`` and log its row in ``finally`` with the exit code (SystemExit's code, 1
    for any other exception, 0 on success)."""
    rc = 1
    try:
        fn()
        rc = 0
        return 0
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        raise
    finally:
        log(step, rc, outputs)


def read() -> list[dict]:
    p = IO.rp(K.ATTEMPTS)
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
