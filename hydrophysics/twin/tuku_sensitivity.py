"""Uncalibrated one-cell structural sensitivity from saved four-layer heads.

This is a head-weighted proxy in metres, not settlement, a vertical strain
model, an intervention forecast, or a fitted material law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from hydrophysics.twin.site_geometry import load_tuku_geometry

# Within an aquifer use its modeled head; within a confining unit use
# the mean of its two adjacent aquifers. Both are unverified stress drivers.
_DRIVER = np.array([
    [1, 0, 0, 0],
    [.5, .5, 0, 0],
    [0, 1, 0, 0],
    [0, .5, .5, 0],
    [0, 0, 1, 0],
    [0, 0, .5, .5],
    [0, 0, 0, 1],
])


def support_weights() -> dict[str, np.ndarray]:
    """Normalize two source-based *supports*, assuming equal compliance per m."""
    g = load_tuku_geometry()
    supports = {
        "WRA_2003_confining_units": (g.hydro_bounds_m_bgs,
                                      np.array([0, 1, 0, 1, 0, 1, 0])),
        "IOT_2017_CL_ML": (g.soil_bounds_m_bgs,
                            np.array([c in {"CL", "ML"} for c in g.soil_classes])),
    }
    out = {}
    for name, (bounds, mask) in supports.items():
        coeff = np.zeros(4)
        for (lo, hi), used in zip(bounds, mask, strict=True):
            if not used:
                continue
            for (unit_lo, unit_hi), driver in zip(g.hydro_bounds_m_bgs, _DRIVER, strict=True):
                coeff += max(0.0, min(hi, unit_hi) - max(lo, unit_lo)) * driver
        if coeff.sum() <= 0:
            raise ValueError("Empty compressible support")
        out[name] = coeff / coeff.sum()
    return out


def analyze(path: Path, date: str = "2032-12-01") -> dict:
    """Analyze an existing forward archive at the exact Tuku grid cell."""
    g = load_tuku_geometry()
    with np.load(path, allow_pickle=False) as archive:
        required = {"mask", "heads_mean", "scenario_names", "dates", "x0", "y0", "dx"}
        if not required <= set(archive.files):
            raise ValueError("Forward archive lacks required four-layer data")
        mask = archive["mask"]
        x0, y0, dx = (float(archive[k]) for k in ("x0", "y0", "dx"))
        x, y = (g.source["location"][k] for k in ("x_m", "y_m"))
        ix, iy = int((x - x0) // dx), int((y - y0) // dx)
        if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]):
            raise ValueError("Tuku is outside the active forward grid")
        cell = int(np.count_nonzero(mask[:iy]) + np.count_nonzero(mask[iy, :ix]))
        dates = [str(v)[:10] for v in archive["dates"]]
        if dates.count(date) != 1:
            raise ValueError("Date absent or duplicated in forward archive")
        t = dates.index(date)
        names = [str(v) for v in archive["scenario_names"]]
        if len(set(names)) != len(names) or not {"baseline", "cut30", "retire_aqua"} <= set(names):
            raise ValueError("Forward archive needs three unique named scenarios")
        heads = archive["heads_mean"]
        if heads.ndim != 4 or heads.shape[1] != 4 or heads.shape[2] != int(mask.sum()):
            raise ValueError("Unexpected forward head shape")
        baseline = heads[names.index("baseline"), :, cell, t].astype(float)
        delta = {name: (heads[names.index(name), :, cell, t].astype(float)
                        - baseline) for name in ("cut30", "retire_aqua")}
    if not all(np.isfinite(v).all() for v in delta.values()):
        raise ValueError("Non-finite site heads")
    weights = support_weights()
    difference = delta["cut30"] - delta["retire_aqua"]
    return {
        "archive": str(path),
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "date": date, "site": g.source["ring_station_id"],
        "grid_cell": cell, "grid_xy": [ix, iy],
        "units": "m modeled head rise relative to baseline; NOT settlement",
        "assumptions": [
            "Equal compliance per supported metre, then normalized to unit total",
            "Aquifer uses its own modeled head; confining unit uses adjacent-head mean",
            "No stress-history, screened-interval or material-parameter validation",
        ],
        "delta_head_m": {k: v.tolist() for k, v in delta.items()},
        "cut30_minus_retire_aqua_by_aquifer_m": difference.tolist(),
        "nonnegative_weight_ranking": (
            "cut30_all_weights" if np.all(difference > 0) else
            "retire_aqua_all_weights" if np.all(difference < 0) else
            "depends_on_unknown_weights"
        ),
        "source_supports": {
            name: {
                "normalized_head_weights": w.tolist(),
                "cut30_proxy_m": float(w @ delta["cut30"]),
                "retire_aqua_proxy_m": float(w @ delta["retire_aqua"]),
            } for name, w in weights.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--date", default="2032-12-01")
    args = parser.parse_args()
    print(json.dumps(analyze(args.forward, args.date), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
