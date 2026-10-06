"""Record measured magnetic-ring geometry without inferring geological layers.

WRA ring values are depths below land surface in metres at each survey. This
read-only manifest fixes the first usable survey, site location and grid cell for
observation-operator work; it does not create a vertical compaction prediction.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..subsidence import _decode_mlcw_name
from .grid import FanGrid
from .release import atomic_json, sha256


def reference_depths(wide: pd.DataFrame) -> dict:
    """First survey with two or more ordered NO rings; depths are metres below ground."""
    if wide.empty or wide.index.has_duplicates:
        raise ValueError("Ring series is empty or has duplicate survey dates")
    ring_cols = [c for c in wide if re.fullmatch(r"NO[1-9][0-9]*", str(c))]
    ring_cols.sort(key=lambda c: int(str(c)[2:]))
    if len(ring_cols) < 2:
        raise ValueError("At least two numbered rings are required")
    for date, row in wide.sort_index()[ring_cols].iterrows():
        depth = pd.to_numeric(row, errors="raise").dropna()
        if len(depth) < 2:
            continue
        values = depth.to_numpy(dtype=float)
        if not np.isfinite(values).all() or values[0] <= 0 or np.any(np.diff(values) <= 0):
            raise ValueError("Ring depths must be finite, positive and increasing by number")
        return {"survey_date": pd.Timestamp(date).isoformat(),
                "depth_datum": "metres below ground surface at survey",
                "ring_depths_m_bgs": {str(k): float(v) for k, v in depth.items()}}
    raise ValueError("No survey contains two ring depths")


def _grid_from_forward(path: Path) -> FanGrid:
    with np.load(path, allow_pickle=False) as z:
        return FanGrid(int(z["nx"]), int(z["ny"]), float(z["dx"]),
                       float(z["x0"]), float(z["y0"]), z["mask"])


def build(cache_dir: Path, stations_csv: Path, forward: Path,
          protocol_path: Path | None = None) -> dict:
    """Hash-checkable geometry for old ring sites and optional later protocol sites."""
    grid = _grid_from_forward(forward)
    stations = pd.read_csv(stations_csv).set_index("sub_id")
    if stations.index.has_duplicates:
        raise ValueError("Duplicate old ring site in coordinate catalog")
    records = []

    def add(site: str, wide: pd.DataFrame, source: Path, x: float, y: float) -> None:
        cell = grid.active_index(x, y)
        if cell is None:
            raise ValueError(f"Ring site outside saved grid: {site}")
        records.append({"site": site, "source_name": source.name,
                        "source_sha256": sha256(source), "x_3826_m": x, "y_3826_m": y,
                        "grid_active_index": cell, **reference_depths(wide)})

    old_files = sorted(cache_dir.glob("ls-wra-mlcw-obs__*.parquet"))
    if not old_files:
        raise ValueError("No historical ring files")
    for source in old_files:
        site = _decode_mlcw_name(str(source))
        if site not in stations.index:
            raise ValueError(f"Historical ring site lacks coordinates: {site}")
        row = stations.loc[site]
        add(site, pd.read_parquet(source), source, float(row.x), float(row.y))

    protocol_sha = None
    if protocol_path is not None:
        protocol = json.loads(protocol_path.read_text())
        protocol_sha = sha256(protocol_path)
        files = protocol["files"]
        catalog_path = Path(files["stations"]["path"])
        if sha256(catalog_path) != files["stations"]["sha256"]:
            raise ValueError("Protocol station catalog hash changed")
        catalog = pd.read_parquet(catalog_path).set_index("sid")
        for i, site in enumerate(protocol["sites"]):
            item = files[f"ring_{i}"]
            source = Path(item["path"])
            if sha256(source) != item["sha256"]:
                raise ValueError("Protocol ring source hash changed")
            raw = pd.read_parquet(source)
            if raw.duplicated(["datetime", "depth"]).any():
                raise ValueError("Duplicate date/ring values")
            wide = raw.pivot(index="datetime", columns="depth", values="value")
            row = catalog.loc[site]
            add(site, wide, source, float(row.X_3826), float(row.Y_3826))
    if len({r["site"] for r in records}) != len(records):
        raise ValueError("Duplicate site across historical and later sources")
    return {"schema": 1, "meaning": "measured ring geometry, not modeled layer contacts",
            "forward_grid_sha256": sha256(forward),
            "old_station_catalog_sha256": sha256(stations_csv),
            "protocol_sha256": protocol_sha, "sites": records}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--stations-csv", type=Path, required=True)
    parser.add_argument("--forward", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    atomic_json(args.out, build(args.cache_dir, args.stations_csv, args.forward,
                                args.protocol))


if __name__ == "__main__":
    main()
