"""Source-checked geometry for the 土庫國中 ring installation.

Overlaps are lengths in metres. They do not allocate observed shortening or
assign compressibility to a soil class or hydrogeologic unit.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DATA = Path(__file__).parent / "data"


@dataclass(frozen=True)
class TukuGeometry:
    soil_bounds_m_bgs: np.ndarray
    soil_classes: tuple[str, ...]
    source_layer_numbers: tuple[int, ...]
    installed_ring_depths_m_bgs: np.ndarray
    hydro_bounds_m_bgs: np.ndarray
    hydro_names: tuple[str, ...]
    source: dict


def _checked_rows(spec: dict) -> list[dict]:
    path = _DATA / spec["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
        raise ValueError(f"Source-version hash mismatch: {path.name}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def load_tuku_geometry() -> TukuGeometry:
    """Load two separately dated official site tables, rejecting altered input."""
    source = json.loads((_DATA / "tuku_geometry_sources.json").read_text())
    if (source["schema"] != 1 or source["site_name"] != "土庫國中"
            or source["ring_station_id"] != "A130600GWMWTKJS"
            or source["location"]["crs"] != "EPSG:3826"
            or (source["location"]["x_m"], source["location"]["y_m"])
            != (187771, 2620610)):
        raise ValueError("Unexpected station identity or coordinate reference")
    soil = _checked_rows(source["soil"])
    if len(soil) != 47 or [int(row["sequence_by_depth"]) for row in soil] != list(range(1, 48)):
        raise ValueError("Expected 47 depth-ordered soil intervals")
    printed = tuple(int(row["source_printed_layer"]) for row in soil)
    if [(i, label) for i, label in enumerate(printed, 1) if i != label] != [(39, 49)]:
        raise ValueError("Unexpected printed soil-layer labels")
    bounds = np.array([(float(row["top_m_bgs"]), float(row["bottom_m_bgs"]))
                       for row in soil], dtype=float)
    if (not np.isfinite(bounds).all() or bounds[0, 0] != 0 or bounds[-1, 1] != 300
            or np.any(bounds[:, 1] <= bounds[:, 0])
            or not np.allclose(bounds[:-1, 1], bounds[1:, 0], rtol=0, atol=1e-9)
            or any(int(row["source_pdf_page"]) != 83 for row in soil)):
        raise ValueError("Soil intervals must cover 0–300 m without gaps or overlaps")
    classes = tuple(row["soil_class"] for row in soil)
    if not set(classes) <= {"SP", "CL", "ML", "SP~GP"}:
        raise ValueError("Unexpected soil-class code")
    rings = _checked_rows(source["rings"])
    if len(rings) != 26 or [int(row["ring_number"]) for row in rings] != list(range(1, 27)):
        raise ValueError("Expected 26 original installation rings")
    depths = np.array([float(row["installed_depth_m_bgs"]) for row in rings], dtype=float)
    if not np.isfinite(depths).all() or depths[0] <= 0 or np.any(np.diff(depths) <= 0):
        raise ValueError("Original ring depths must increase")
    edges = np.asarray(source["hydro_units"]["edges_m_bgs"], dtype=float)
    names = tuple(source["hydro_units"]["names"])
    if (len(names) != 7 or len(edges) != len(names) + 1 or edges[0] != 0
            or edges[-1] != 300 or np.any(np.diff(edges) <= 0)):
        raise ValueError("Unexpected interpreted hydrogeologic boundaries")
    return TukuGeometry(bounds, classes, printed, depths,
                        np.column_stack((edges[:-1], edges[1:])), names, source)


def overlap_lengths(bounds_m_bgs: np.ndarray, shallow_m_bgs: float,
                    deep_m_bgs: float) -> np.ndarray:
    """Return geometric intersection lengths; reject unlogged depths."""
    bounds = np.asarray(bounds_m_bgs, dtype=float)
    if (bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all()
            or np.any(bounds[:, 1] <= bounds[:, 0])
            or not np.allclose(bounds[:-1, 1], bounds[1:, 0], rtol=0, atol=1e-9)):
        raise ValueError("Bounds must be finite, contiguous increasing intervals")
    if (not np.isfinite([shallow_m_bgs, deep_m_bgs]).all()
            or shallow_m_bgs < bounds[0, 0] or deep_m_bgs > bounds[-1, 1]
            or shallow_m_bgs >= deep_m_bgs):
        raise ValueError("Ring interval is reversed or exceeds logged depth")
    lengths = np.maximum(0.0, np.minimum(bounds[:, 1], deep_m_bgs)
                         - np.maximum(bounds[:, 0], shallow_m_bgs))
    if not np.isclose(lengths.sum(), deep_m_bgs - shallow_m_bgs,
                      rtol=0, atol=1e-8):
        raise ValueError("Interval overlap does not conserve measured thickness")
    return lengths


def installed_ring_overlap(geometry: TukuGeometry, shallow_ring: int,
                           deep_ring: int, *, basis: str = "soil") -> np.ndarray:
    """Original installed-ring interval versus soil or interpreted hydro units."""
    if (not 1 <= shallow_ring < deep_ring <= len(geometry.installed_ring_depths_m_bgs)
            or basis not in {"soil", "hydro"}):
        raise ValueError("Invalid installed ring IDs or geometry basis")
    depths = geometry.installed_ring_depths_m_bgs
    bounds = geometry.soil_bounds_m_bgs if basis == "soil" else geometry.hydro_bounds_m_bgs
    return overlap_lengths(bounds, depths[shallow_ring - 1], depths[deep_ring - 1])
