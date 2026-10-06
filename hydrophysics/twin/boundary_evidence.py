"""Read-only spatial diagnostic of a saved column-boundary discontinuity.

Adjacent-cell differences are model-grid behavior, not measured geology. No
parameter is changed and no smoothing or new forward simulation is performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .release import atomic_json, sha256


def cross_boundary_pairs(mask: np.ndarray, x0_m: float, dx_m: float,
                         boundary_km: float) -> np.ndarray:
    """Active-cell indices in west-west/west/east/east-east order, same grid row."""
    if mask.ndim != 2 or dx_m <= 0 or not np.isfinite(boundary_km):
        raise ValueError("Invalid grid geometry or boundary")
    rows, cols = np.nonzero(mask)
    lookup = {(int(row), int(col)): i
              for i, (row, col) in enumerate(zip(rows, cols, strict=True))}
    east_col = int(np.searchsorted(x0_m + (np.arange(mask.shape[1]) + .5) * dx_m,
                                    boundary_km * 1000, side="left"))
    west_col = east_col - 1
    if west_col < 1 or east_col + 1 >= mask.shape[1]:
        raise ValueError("Boundary lacks two cells on each side")
    candidates = [(row, [west_col - 1, west_col, east_col, east_col + 1])
                  for row in range(mask.shape[0])]
    pairs = [[lookup[(row, col)] for col in selected]
             for row, selected in candidates
             if all((row, col) in lookup for col in selected)]
    if not pairs:
        raise ValueError("No complete active rows across boundary")
    return np.asarray(pairs, dtype=int)


def jump_diagnostic(field: np.ndarray, pairs: np.ndarray) -> dict:
    """Compare the boundary jump to the mean neighboring grid slope (metres).

    ``field`` contains one scalar per active cell, such as a 12-month cumulative
    change. A nonzero excess is a finite-difference anomaly, not proof of cause.
    """
    if field.ndim != 1 or pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError("Expected one field value per cell and four-cell row pairs")
    q = field[pairs]
    if not np.isfinite(q).all():
        raise ValueError("Nonfinite boundary field")
    jump = q[:, 2] - q[:, 1]
    neighbor_slope = ((q[:, 1] - q[:, 0]) + (q[:, 3] - q[:, 2])) / 2
    excess = jump - neighbor_slope
    def stats(v):
        return {"median_cm": float(np.median(v) * 100),
                "p10_cm": float(np.percentile(v, 10) * 100),
                "p90_cm": float(np.percentile(v, 90) * 100),
                "median_absolute_cm": float(np.median(np.abs(v)) * 100)}
    return {"rows": len(pairs), "east_minus_west": stats(jump),
            "neighbor_slope": stats(neighbor_slope), "excess": stats(excess)}


def evaluate(path: str | Path, *, boundary_km: float = 182.0,
             earlier: str = "2025-12", later: str = "2026-12") -> dict:
    """Diagnose a saved total-column field over two calendar-month states."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        dates = np.asarray(z["dates"]).astype("datetime64[M]")
        a, b = np.datetime64(earlier, "M"), np.datetime64(later, "M")
        if np.count_nonzero(dates == a) != 1 or np.count_nonzero(dates == b) != 1 or b <= a:
            raise ValueError("Requested months must uniquely exist in chronological order")
        ia, ib = int(np.flatnonzero(dates == a)[0]), int(np.flatnonzero(dates == b)[0])
        pairs = cross_boundary_pairs(z["mask"], float(z["x0"]), float(z["dx"]),
                                     boundary_km)
        subs = z["subs_mean"]
        names = [str(v) for v in z["scenario_names"]]
        if subs.ndim != 3 or subs.shape[0] != len(names):
            raise ValueError("Expected scenario/cell/month total-column archive")
        if len(set(names)) != len(names):
            raise ValueError("Scenario names must be unique")
        out = {name: jump_diagnostic(subs[i, :, ib].astype(float)
                                     - subs[i, :, ia].astype(float), pairs)
               for i, name in enumerate(names)}
    return {"schema": 1, "source_sha256": sha256(path),
            "field": "modeled total-column compaction change", "unit": "cm per interval",
            "earlier": earlier, "later": later, "boundary_km_easting": boundary_km,
            "grid_rows": len(pairs), "by_scenario": out,
            "interpretation": "exploratory grid diagnostic; no geological validation or correction"}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--boundary-km", type=float, default=182.0)
    parser.add_argument("--earlier", default="2025-12")
    parser.add_argument("--later", default="2026-12")
    args = parser.parse_args(argv)
    atomic_json(args.out, evaluate(args.forward, boundary_km=args.boundary_km,
                                   earlier=args.earlier, later=args.later))


if __name__ == "__main__":
    main()
