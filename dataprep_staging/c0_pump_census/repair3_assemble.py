"""WP C0 repair round, v3: assemble the v3 build intermediates (``repair3_build.py``) into the next
product versions with ``repair_assemble``'s functions (layout identical to v2) and record their
paths in results/dataprep/reports/C0_evidence/repair3/products_v3.json.

Usage: python -I repair3_assemble.py <build dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import repair_assemble as ra  # noqa: E402

EV3 = lib.EVIDENCE / "repair3"


def main(argv: list[str]) -> int:
    work = Path(argv[0])
    root = lib.DATAPREP
    grid = lib.load_grid()
    gmeta = lib.grid_meta(grid)
    out = {"census_L1": ra.assemble_l1(work, root)}
    for variant in ("preorigin", "full"):
        out[variant] = ra.assemble_variant(variant, work, root, grid, gmeta)
    rec = {k: lib.rel(v) for k, v in out.items()}
    lib.save_json(rec, work / "products_v3.json")
    lib.save_json(rec, EV3 / "products_v3.json")
    print(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
