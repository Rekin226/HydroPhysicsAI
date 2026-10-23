"""WP S1b repair round r3: hashed snapshots of ``reports/S1b.repair.md`` (parts written in order).

    python -I dataprep_staging/s1_rings/s1b/s1b_r3_snapshot.py <label>

Copies the current report to ``S1b_evidence/repair_r3/snapshots/S1b.repair.<label>.<UTC>.md``
(read-only) and appends ``{label, time, sha256, path}`` to ``snapshots/index.jsonl``. A part of
the report is "written before X" when its snapshot time precedes X's look or registration.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from common.paths import rel_to_repo  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402

REPORT = K.DATAPREP / "reports" / "S1b.repair.md"
SNAP = K.EVIDENCE / "repair_r3" / "snapshots"


def snapshot(label: str) -> dict:
    if not label.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"bad label {label!r}")
    SNAP.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dst = SNAP / f"S1b.repair.{label}.{now}.md"
    shutil.copyfile(REPORT, dst)
    os.chmod(dst, 0o444)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    rec = {"label": label, "time": now, "sha256": sha, "path": rel_to_repo(dst),
           "bytes": dst.stat().st_size}
    with open(SNAP / "index.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        raise SystemExit("usage: s1b_r3_snapshot.py <label>")
    print(json.dumps(snapshot(argv[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
