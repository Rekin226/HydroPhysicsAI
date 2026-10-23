"""WP A0 step 1: move the 2026-10-12 pilots into results/dataprep/_pilots/<scout>/.

* Moves (renames, no copy) every scout directory from results/dataprep/pilots_2026-10-12/.
* Copies the review checks that the main session did not copy (plan A0 task 1): the
  planning scratchpad's ``review_cf/`` (scripts and test downloads) and
  ``revise_plan/scripts/`` into ``_pilots/review_cf/`` and ``_pilots/revise_plan/``.
* Leaves a README.md at the old path pointing to the new one.

Idempotent: a scout already moved is skipped. Usage: python -m common.a0.reorganize_pilots
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.paths import DATAPREP, PILOTS  # noqa: E402

OLD = DATAPREP / "pilots_2026-10-12"
#: Planning-session scratchpad that held review_cf/ and revise_plan/ (set by the caller).
SCRATCH = Path(os.environ.get("DATAPREP_PLANNING_SCRATCHPAD", "/nonexistent-scratchpad"))
REVIEW_COPIES = {
    "review_cf": SCRATCH / "review_cf",                  # whole tree (2.9 MB)
    "revise_plan/scripts": SCRATCH / "revise_plan" / "scripts",
}

README_OLD = """# Moved

The pilots that were staged here on 2026-10-12 were moved (not copied) by WP A0 to

    results/dataprep/_pilots/<scout>/

Each scout directory there has a MANIFEST.json (layer "pilot": design-time evidence only,
hashes of every script and output, generators of the large regenerable arrays) and
`_pilots/INDEX.json` lists them. Downloads the scouts made were also copied to their L0
directories under `data_fetch_api/new_measurements/<source>/` with a SOURCE.json each.

Hard-coded paths inside the pilot scripts still name the planning scratchpad
(`<planning scratchpad>/plan/<scout>`); the A0 rerun harness
(`dataprep_staging/common/a0/rerun_pilots.py`) shows how to run them from the new location
without editing the archived copies.
"""


def main() -> int:
    PILOTS.mkdir(parents=True, exist_ok=True)
    moved, skipped = [], []
    if OLD.is_dir():
        for d in sorted(p for p in OLD.iterdir() if p.is_dir()):
            dest = PILOTS / d.name
            if dest.exists():
                skipped.append(d.name)
                continue
            os.rename(d, dest)                          # same filesystem: atomic move
            moved.append(d.name)
    copied = {}
    for rel, src in REVIEW_COPIES.items():
        dest = PILOTS / rel
        if dest.exists():
            copied[rel] = "already present"
            continue
        if not src.is_dir():
            copied[rel] = f"MISSING source {src}"
            continue
        shutil.copytree(src, dest, copy_function=shutil.copy2)
        copied[rel] = f"copied from {src}"
    OLD.mkdir(parents=True, exist_ok=True)
    leftovers = [p.name for p in OLD.iterdir() if p.name != "README.md"]
    (OLD / "README.md").write_text(README_OLD, encoding="utf-8")
    print(json.dumps({"moved": moved, "skipped": skipped, "review_copies": copied,
                      "left_in_old_dir": leftovers}, indent=1))
    return 0 if not leftovers else 1


if __name__ == "__main__":
    raise SystemExit(main())
