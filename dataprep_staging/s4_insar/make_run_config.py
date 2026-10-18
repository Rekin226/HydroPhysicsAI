"""WP S4a step 1: compute the fan clip box from the twin grid and write the run config.

The box is the bounding box (WGS84 lon/lat) of the twin's 1 km grid
(``datum_gate_innov.npz``: nx, ny, dx, x0, y0 in TWD97 / TM2, EPSG:3826) expanded by
``CLIP_BUFFER_M``. It is written to ``results/dataprep/insar_tools/s4a_run/run_config.json``
(gitignored) together with the off-fan NGL GNSS stations it contains, so that S4b can check
that its reference stations fall inside the subsets.

Run with the hydro env: python -I make_run_config.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dataprep_staging (common)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pyproj  # noqa: E402
from common.hashing import grid_hash, sha256_file  # noqa: E402
from common.paths import L0_NEW, rel_to_repo  # noqa: E402
from config import CLIP_BUFFER_M, FRAMES, RUN_CONFIG, TWIN_GRID  # noqa: E402

NGL_HOLDINGS = L0_NEW / "ngl_gnss" / "2026-10-12" / "DataHoldings.txt"


def clip_box() -> dict:
    z = np.load(TWIN_GRID, allow_pickle=False)
    nx, ny, dx = int(z["nx"]), int(z["ny"]), float(z["dx"])
    x0, y0 = float(z["x0"]), float(z["y0"])
    b = CLIP_BUFFER_M
    tr = pyproj.Transformer.from_crs(3826, 4326, always_xy=True)
    # densify the buffered rectangle's edges so the TM2 -> lon/lat bounding box is exact
    xs = np.linspace(x0 - b, x0 + nx * dx + b, 200)
    ys = np.linspace(y0 - b, y0 + ny * dx + b, 200)
    edge = np.concatenate([
        np.c_[xs, np.full_like(xs, ys[0])], np.c_[xs, np.full_like(xs, ys[-1])],
        np.c_[np.full_like(ys, xs[0]), ys], np.c_[np.full_like(ys, xs[-1]), ys]])
    lon, lat = tr.transform(edge[:, 0], edge[:, 1])
    return {
        "lon0": round(float(np.min(lon)), 4), "lat0": round(float(np.min(lat)), 4),
        "lon1": round(float(np.max(lon)), 4), "lat1": round(float(np.max(lat)), 4),
        "crs": "EPSG:4326",
        "derivation": (f"bounding box of the twin grid (nx={nx}, ny={ny}, dx={dx:g} m, EPSG:3826)"
                       f" expanded by {b:g} m, densified edges, rounded to 1e-4 deg"),
        "twin_grid": rel_to_repo(TWIN_GRID),
        "twin_grid_sha256": sha256_file(TWIN_GRID),
        "twin_grid_hash": grid_hash(nx, ny, dx, x0, y0, z["mask"]),
    }


def ngl_inside(box: dict) -> list[dict]:
    """NGL stations (public DataHoldings.txt) inside the box; S4b picks references from them."""
    out = []
    if not NGL_HOLDINGS.is_file():
        return out
    for line in NGL_HOLDINGS.read_text(errors="replace").splitlines()[1:]:
        f = line.split()
        if len(f) < 9:
            continue
        try:
            lat, lon = float(f[1]), float(f[2])
        except ValueError:
            continue
        lon = lon - 360.0 if lon > 180 else lon
        if box["lon0"] <= lon <= box["lon1"] and box["lat0"] <= lat <= box["lat1"]:
            out.append({"station": f[0], "lat": lat, "lon": lon, "first": f[7], "last": f[8]})
    return out


def main() -> int:
    box = clip_box()
    cfg = {
        "wp": "S4a",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "box": box,
        "frames": {k: {kk: (rel_to_repo(vv) if kk == "l0_dir" else vv) for kk, vv in v.items()}
                   for k, v in FRAMES.items()},
        "ngl_stations_in_box": ngl_inside(box),
        "ngl_holdings": rel_to_repo(NGL_HOLDINGS),
    }
    RUN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    RUN_CONFIG.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(box, indent=1))
    print(f"{len(cfg['ngl_stations_in_box'])} NGL stations inside the box -> {RUN_CONFIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
