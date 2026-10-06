"""Retrospective, uncorrected Tuku GNSS-to-column diagnostic.

The official field is elevation in metres, but its vertical reference and
instrument maintenance are unresolved. Output scores are raw contrasts within
apparently stable segments, never a validation verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydrophysics.twin.site_geometry import load_tuku_geometry

STATION = "A130600LSTKJS"
MIN_DAYS_PER_MONTH = 20
STEP_M = 0.025  # exploratory flag, chosen after looking at this series


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_height(path: Path) -> pd.Series:
    """Read one daily station cache, preserving raw heights and rejecting duplicates."""
    if path.suffix == ".json":
        records = json.loads(path.read_text())
        rows = [row for row in records if row.get("observatoryidentifier") == STATION]
        if not rows or any(row.get("observatoryname") != "土庫國中" for row in rows):
            raise ValueError("Official Tuku GNSS records absent or mislabeled")
        dates = pd.to_datetime([row["observationttime"] for row in rows])
        values = pd.to_numeric([row["height"] for row in rows], errors="raise")
        s = pd.Series(values, index=dates, name="value").sort_index()
    elif path.suffix == ".parquet" and path.name.endswith("__TKJS.parquet"):
        frame = pd.read_parquet(path)
        if list(frame.columns) != ["value"] or not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Expected a daily DatetimeIndex and one height value column")
        s = frame["value"].sort_index().astype(float)
    else:
        raise ValueError("Expected official JSON or named Tuku GNSS parquet cache")
    if (s.empty or s.index.has_duplicates or s.index.tz is not None
            or not np.isfinite(s.to_numpy()).all()
            or not (s.index == s.index.normalize()).all()):
        raise ValueError("GNSS daily dates or values are invalid")
    return s


def station_match(meta_path: Path) -> dict:
    """Require the distinct official GNSS identity near the ring site."""
    meta = pd.read_parquet(meta_path)
    rows = meta.loc[meta["ObservatoryIdentifier"].eq(STATION)]
    if len(rows) != 1 or rows.iloc[0]["ObservatoryName"] != "土庫國中":
        raise ValueError("Tuku GNSS station identity is absent or ambiguous")
    row = rows.iloc[0]
    x, y = float(row["X_3826"]), float(row["Y_3826"])
    loc = load_tuku_geometry().source["location"]
    distance = float(np.hypot(x - loc["x_m"], y - loc["y_m"]))
    if not np.isfinite(distance) or distance > 20:
        raise ValueError("GNSS station is not colocated with the ring")
    return {"station_id": STATION, "crs": "EPSG:3826", "xy_m": [x, y],
            "ring_distance_m": distance}


def step_candidates(s: pd.Series) -> list[dict]:
    """Flag persistent jumps; do not correct or silently interpolate them."""
    candidates = []
    for day, jump in s.diff().dropna().items():
        if abs(jump) < STEP_M:
            continue
        before = s.loc[day - pd.Timedelta(days=15):day - pd.Timedelta(days=1)]
        after = s.loc[day:day + pd.Timedelta(days=14)]
        if min(len(before), len(after)) < 10:
            continue
        shift = float(after.median() - before.median())
        if abs(shift) >= STEP_M:
            candidates.append({"date": day.strftime("%Y-%m-%d"),
                               "one_day_jump_cm": float(jump * 100),
                               "window_median_shift_cm": shift * 100})
    # Consecutive-day candidates describe one change.
    out = []
    for item in candidates:
        if out and (pd.Timestamp(item["date"]) - pd.Timestamp(out[-1]["date"])).days <= 1:
            if abs(item["one_day_jump_cm"]) > abs(out[-1]["one_day_jump_cm"]):
                out[-1] = item
        else:
            out.append(item)
    return out


def _model_column(path: Path, ring_xy: tuple[float, float]) -> tuple[pd.DatetimeIndex, np.ndarray, int]:
    with np.load(path, allow_pickle=False) as z:
        required = {"mask", "x0", "y0", "dx", "dates", "subs_mean", "scenario_names"}
        if not required <= set(z.files):
            raise ValueError("Forward archive lacks site/date/subsidence fields")
        mask = z["mask"]
        dx = float(z["dx"])
        if mask.ndim != 2 or dx <= 0:
            raise ValueError("Invalid model grid")
        ix = int((ring_xy[0] - float(z["x0"])) // dx)
        iy = int((ring_xy[1] - float(z["y0"])) // dx)
        if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1] and mask[iy, ix]):
            raise ValueError("Tuku lies outside the model grid")
        cell = int(np.count_nonzero(mask[:iy]) + np.count_nonzero(mask[iy, :ix]))
        names = [str(v) for v in z["scenario_names"]]
        if names.count("baseline") != 1:
            raise ValueError("Missing or duplicated baseline")
        subs = z["subs_mean"]
        if subs.ndim != 3 or subs.shape[1] != mask.sum():
            raise ValueError("Unexpected whole-column shape")
        curve = subs[names.index("baseline"), cell].astype(float)
        dates = pd.to_datetime(z["dates"])
    if dates.has_duplicates or not np.isfinite(curve).all() or len(curve) != len(dates):
        raise ValueError("Invalid model dates or subsidence")
    return dates.to_period("M").to_timestamp(), curve, cell


def segment_contrasts(s: pd.Series, steps: list[dict], dates: pd.DatetimeIndex,
                      subs_m: np.ndarray) -> list[dict]:
    """Monthly median raw displacement versus whole-column changes, by QC segment."""
    # An event day begins a fresh segment. No across-step differences are scored.
    cuts = [s.index.min()] + [pd.Timestamp(x["date"]) for x in steps]
    cuts += [s.index.max() + pd.Timedelta(days=1)]
    model = pd.Series(subs_m, index=dates)
    flagged_months = {pd.Timestamp(item["date"]).to_period("M").to_timestamp()
                      for item in steps}
    if model.index.has_duplicates:
        raise ValueError("Duplicate model months")
    out = []
    for begin, end in zip(cuts[:-1], cuts[1:], strict=True):
        part = s.loc[(s.index >= begin) & (s.index < end)]
        count = part.resample("MS").count()
        monthly = part.resample("MS").median().loc[count >= MIN_DAYS_PER_MONTH]
        # A partial event month cannot share the model full-month timestamp.
        monthly = monthly.loc[(monthly.index >= pd.Timestamp("2022-12-01"))
                              & ~monthly.index.isin(flagged_months)]
        common = monthly.index.intersection(model.index)
        if len(common) < 3:
            continue
        obs_cm = (monthly.loc[common[0]] - monthly.loc[common]).to_numpy() * 100
        pred_cm = (model.loc[common] - model.loc[common[0]]).to_numpy() * 100
        out.append({
            "start_month": common[0].strftime("%Y-%m"), "end_month": common[-1].strftime("%Y-%m"),
            "n_months": len(common), "months": [d.strftime("%Y-%m") for d in common],
            "endpoint_raw_height_drop_cm": float(obs_cm[-1]),
            "endpoint_model_column_cm": float(pred_cm[-1]),
            "rmse_raw_model_cm": float(np.sqrt(np.mean((obs_cm - pred_cm) ** 2))),
            "rmse_zero_change_cm": float(np.sqrt(np.mean(obs_cm ** 2))),
            "status": "exploratory_raw_contrast_NOT_VALIDATION",
        })
    return out


def analyze(cache: Path, metadata: Path, forward: list[Path]) -> dict:
    station = station_match(metadata)
    series = checked_height(cache)
    steps = step_candidates(series)
    loc = load_tuku_geometry().source["location"]
    results = {}
    for path in forward:
        dates, subs, cell = _model_column(path, (loc["x_m"], loc["y_m"]))
        results[path.stem] = {
            "archive_sha256": _sha256(path), "grid_cell": cell,
            "segments": segment_contrasts(series, steps, dates, subs),
        }
    return {
        "status": "NOT_VALIDATED_OBSERVATION_OPERATOR_AND_REFERENCE_UNRESOLVED",
        "station": station,
        "height_source": str(cache), "height_source_sha256": _sha256(cache),
        "metadata_sha256": _sha256(metadata),
        "n_daily": len(series), "first_day": str(series.index.min().date()),
        "last_day": str(series.index.max().date()),
        "step_threshold_cm": STEP_M * 100, "step_candidates": steps,
        "min_days_per_month": MIN_DAYS_PER_MONTH,
        "limits": [
            "WRA describes height in metres but does not specify the height component or vertical datum here",
            "Station maintenance and processing history not verified; flagged jumps are not corrected",
            "GNSS includes reference, tectonic, station and deeper motion absent from modeled column",
            "Each segment removes its own absolute offset; calendar-month seasonality is not corrected",
            "Model monthly state date versus GNSS calendar-month median timing is not independently verified",
            "Zero-change comparator and raw contrasts are descriptive, not held-out skill",
            "GNSS data predate this analysis; model-selection independence is unverified",
        ],
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", "--height-source", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--forward", required=True, nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cache, args.metadata, args.forward),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
