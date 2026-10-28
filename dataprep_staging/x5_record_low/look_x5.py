"""Record an X5 look in the shared look log (harness.prereg3.record_look).

Run under the shared lock, short and alone:
    flock results/dataprep/preregistration/.x5.lock python look_x5.py PURPOSE SCOPE NOTE [--prereg]

The unit is the 14 ring sites' band series and their nest heads ("X5:ring_bands_14sites");
its members are the nest wells of results/twin/mechanism/rings_sites.csv (ring sites are
neither registered wells, benchmarks nor cells).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402
from harness import prereg3  # noqa: E402

UNIT = "X5:ring_bands_14sites"


def members() -> dict:
    ids = sorted({w for ws in L.ring_sites().wells for w in ws})
    return {UNIT: {"description": "monthly compaction of the depth bands of the 14 ring "
                                  "(MLCW) sites and the heads of their nest wells (ids)",
                   "ids": ids}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("purpose")
    ap.add_argument("scope")
    ap.add_argument("note")
    ap.add_argument("--prereg", action="store_true", help="tie the look to the declaration")
    a = ap.parse_args(argv)
    rec = prereg3.record_look([UNIT], a.scope, K.WP, a.purpose, members=members(),
                              prereg_id=K.PREREG_ID if a.prereg else None, note=a.note)
    print(json.dumps({k: rec[k] for k in ("time", "purpose", "data_scope", "scope_windows")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
