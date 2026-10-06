"""Observation-only rank sensitivity to nearby Tuku head histories.

No screen-to-model-layer mapping, head datum, head unit, or ring outcome is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from hydrophysics.twin.site_geometry import load_tuku_geometry
from hydrophysics.twin.synthetic_depth_identifiability import designs, matrix, ring_depths

WELLS = (
    ("09050321", "3132015GW09050321", "土庫(2)", 90.0, "2"),
    ("09050331", "3132015GW09050331", "土庫(3)", 185.0, "2"),
    ("09050341", "3132015GW09050341", "土庫(4)", 269.0, "3"),
)
START = "2022-12-01"
END = "2026-09-30"
MIN_MONTH_COVERAGE = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def monthly_values(series: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    """Month medians with at least half of scheduled 10-minute slots valid."""
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.has_duplicates:
        raise ValueError("Expected unique datetime index")
    if series.index.tz is not None:
        raise ValueError("Head timezone has not been verified")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Head dates must be ordered")
    values = pd.to_numeric(series, errors="coerce")
    invalid = values.isna() | ~np.isfinite(values) | (values <= -1e5)
    values = values.mask(invalid).loc[START:END]
    median = values.resample("MS").median()
    count = values.resample("MS").count()
    required = np.ceil(median.index.days_in_month * 144 * MIN_MONTH_COVERAGE)
    return median.where(count >= required), count, int(invalid.sum())


def normalized_condition(x: np.ndarray) -> tuple[int, float | None]:
    """Column-scale-free sensitivity; rank tolerance is fixed and reported."""
    norms = np.linalg.norm(x, axis=0)
    scale = np.where(norms == 0, 1.0, norms)
    singular = np.linalg.svd(x / scale, compute_uv=False)
    rank = int(np.count_nonzero(singular > singular[0] * 1e-10))
    return rank, (float(singular[0] / singular[-1])
                  if rank == x.shape[1] else None)


def scenario_table(observed: np.ndarray) -> dict:
    if observed.ndim != 2 or observed.shape[1] != 3 or not np.isfinite(observed).all():
        raise ValueError("Expected finite three-well anomalies")
    geom = load_tuku_geometry()
    structures = designs(geom)
    depths = ring_depths(geom)
    scenarios = {}
    # Four monotone placements of three measured series in four modeled aquifer
    # slots. Zero in an empty slot is a projection, not a physical zero head.
    for slots in combinations(range(4), 3):
        scenarios["ordered_slots_" + "".join(str(s + 1) for s in slots)] = (
            (0, 1, 2), slots)
    # The catalog codes put both shallow wells in layer 2. Choosing either
    # candidate for that code is an explicit unresolved alternative.
    scenarios["catalog_code2_well2_code3_well4"] = ((0, 2), (1, 2))
    scenarios["catalog_code2_well3_code3_well4"] = ((1, 2), (1, 2))
    rows = {}
    for label, (sources, slots) in scenarios.items():
        projected = np.zeros((len(observed), 4))
        for source, slot in zip(sources, slots, strict=True):
            projected[:, slot] = observed[:, source]
        rows[label] = {
            "observed_well_indices_1based": [s + 1 for s in sources],
            "model_aquifer_slots_1based": [s + 1 for s in slots],
            "designs": {
                name: dict(zip(("rank", "column_normalized_condition"),
                               normalized_condition(matrix(design, projected, depths)), strict=True))
                for name, design in structures.items()
            },
        }
    envelope = {}
    for name, design in structures.items():
        conds = [(key, item["designs"][name]["column_normalized_condition"])
                 for key, item in rows.items()]
        valid = [(key, value) for key, value in conds if value is not None]
        envelope[name] = {
            "n_parameters": len(design.parameter_names),
            "min_rank": min(row["designs"][name]["rank"] for row in rows.values()),
            "max_rank": max(row["designs"][name]["rank"] for row in rows.values()),
            "best_condition_scenario": min(valid, key=lambda v: v[1]) if valid else None,
            "worst_finite_condition_scenario": max(valid, key=lambda v: v[1])
            if valid else None,
            "rank_deficient_scenarios": [key for key, value in conds if value is None],
        }
    return {"scenarios": rows, "envelope": envelope}


def analyze(head_dir: Path, metadata_path: Path) -> dict:
    metadata = pd.read_parquet(metadata_path)
    monthly = {}
    provenance = []
    for sid, identifier, name, depth, layer_code in WELLS:
        found = metadata.loc[metadata["sid"].astype(str) == sid]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one metadata row for {sid}")
        row = found.iloc[0]
        checks = (("WellIdentifier", identifier), ("WellName", name),
                  ("GroundwaterLayerCode", layer_code))
        if any(str(row[key]) != target for key, target in checks):
            raise ValueError(f"Well identity or layer code changed for {sid}")
        if float(row["WellDepth"]) != depth or float(row["FinishDepth"]) != depth:
            raise ValueError(f"Well depth changed for {sid}")
        if str(row["LocationByTWD97"]) != "187777.40 2620597.81":
            raise ValueError(f"Well coordinate changed for {sid}")
        path = head_dir / f"{sid}.parquet"
        frame = pd.read_parquet(path)
        if list(frame.columns) != ["value"]:
            raise ValueError(f"Unexpected head schema for {sid}")
        values, counts, invalid = monthly_values(frame["value"])
        monthly[sid] = values
        provenance.append({
            "sid": sid, "well_identifier": identifier, "well_name": name,
            "total_depth_m": depth, "catalog_layer_code": layer_code,
            "file_sha256": sha256(path), "rows": len(frame),
            "invalid_or_sentinel_rows": invalid,
            "first_timestamp": str(frame.index.min()),
            "last_timestamp": str(frame.index.max()),
            "catalog_end_datetime": str(row["end_datetime_from_dataset"]),
            "months_passing_coverage": int(values.notna().sum()),
            "months_below_coverage": [str(t.date()) for t in values.index[values.isna()]],
            "valid_10min_rows_by_month": {str(t.date()): int(v) for t, v in counts.items()},
        })
    common = pd.DataFrame(monthly).loc[START:END].dropna()
    if len(common) < 12:
        raise ValueError("Too few common covered months")
    # Per-series standardization removes unknown common/individual vertical
    # offsets and scale for a dimensionless correlation diagnostic only.
    std = common.std(ddof=1)
    if (std <= 0).any():
        raise ValueError("Constant observed series")
    observed = ((common - common.iloc[0]) / std).to_numpy()
    singular = np.linalg.svd(observed, compute_uv=False)
    information = scenario_table(observed)
    return {
        "status": "OBSERVED_INPUT_INFORMATION_ONLY_NO_SCREEN_MAPPING_OR_FIELD_FIT",
        "metadata_sha256": sha256(metadata_path),
        "monthly_window": [START, END],
        "monthly_aggregation": "median; sentinel <= -100000/nonfinite omitted; >=50% scheduled ten-minute slots",
        "n_common_covered_months": len(common),
        "common_months": [str(t.date()) for t in common.index],
        "well_order": [row["sid"] for row in provenance],
        "well_files": provenance,
        "standardization": "first common month subtracted; divide by each common-window sample SD",
        "head_value_units_and_vertical_reference": "unverified; dimensionless correlation only",
        "well_correlation": common.corr().to_numpy().tolist(),
        "observed_driver_singular_values": singular.tolist(),
        "observed_driver_condition": float(singular[0] / singular[-1]),
        "rank_relative_tolerance": 1e-10,
        "projection_warning": "unfilled modeled layer is zero only in measured-driver projection, not a physical head assumption",
        "mapping_warning": "well total depth and catalog codes do not identify screened intervals; all scenarios conditional",
        **information,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.head_dir, args.metadata), ensure_ascii=False,
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
