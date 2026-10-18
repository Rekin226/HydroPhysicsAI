"""WP S4a: share of the twin's active cells inside each frame's footprint polygon.

Reads each frame's saved ``metadata/<frame>-poly.txt`` (lon lat vertices, L0) and
``twin_cells.npz``; writes ``results/dataprep/insar_tools/s4a_run/footprint_coverage.json``
with, per frame, the fraction (and count) of active cell centres inside the polygon and the
latitude band of the uncovered active cells. Geometry only: no pixel values.

Run with the hydro env: python -I footprint_coverage.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from config import FRAMES, RUN_DIR  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402


def main() -> int:
    z = np.load(RUN_DIR / "twin_cells.npz")
    m = z["mask"].astype(bool)
    pts = np.c_[z["lon"][m], z["lat"][m]]
    out = {}
    union = np.zeros(len(pts), bool)
    for frame, spec in FRAMES.items():
        poly = Path(spec["l0_dir"]) / "metadata" / f"{frame}-poly.txt"
        xy = np.array([[float(v) for v in ln.split()[:2]]
                       for ln in poly.read_text().splitlines() if ln.strip()])
        inside = MPath(xy).contains_points(pts)
        union |= inside
        miss = pts[~inside]
        out[frame] = {"n_active": int(len(pts)), "n_inside": int(inside.sum()),
                      "frac_inside": round(float(inside.mean()), 4),
                      "uncovered_lat_range": ([round(float(miss[:, 1].min()), 3),
                                               round(float(miss[:, 1].max()), 3)]
                                              if len(miss) else None)}
    out["union_all_frames"] = {"frac_inside": round(float(union.mean()), 4)}
    (RUN_DIR / "footprint_coverage.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
