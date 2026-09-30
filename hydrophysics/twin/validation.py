"""Dated observation QC and locked-protocol evaluation of prospective head predictions.

Evaluation never estimates an offset from held-out observations. Supply physical or
datum-adjusted predictions consistently, with offsets fitted before prediction_origin.
This evaluates head predictions, not the causal effect of a pumping intervention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .release import atomic_json, sha256


def observations(path: str | Path, *, as_of=None, max_age_days=62) -> dict:
    frame = pd.read_csv(path, dtype={"station_id": str, "datum": str})
    required = {"station_id", "date", "head_m", "layer", "datum"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Observation columns missing: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Observation records must be non-empty and complete")
    frame["date"] = pd.to_datetime(frame.date, utc=True, errors="raise")
    if frame.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate station/date observations")
    if not np.isfinite(frame.head_m.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite observed head")
    if not frame.layer.isin([1, 2, 3, 4]).all():
        raise ValueError("Layer must be an integer from 1 through 4")
    if frame.station_id.str.strip().eq("").any() or frame.datum.str.strip().eq("").any():
        raise ValueError("Station and vertical datum must be identified")
    if frame.datum.nunique() != 1:
        raise ValueError("Convert observations to one declared vertical datum before ingestion")
    if (frame.groupby("station_id").layer.nunique() > 1).any():
        raise ValueError("A station cannot change aquifer layer within an observation batch")
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if (frame.date > now).any():
        raise ValueError("Observations cannot be dated in the future")
    if max_age_days < 0:
        raise ValueError("max_age_days must be non-negative")
    ages = (now - frame.groupby("station_id").date.max()).dt.total_seconds() / 86400
    return {"schema": 1, "input_sha256": sha256(path), "rows": len(frame),
            "stations": int(frame.station_id.nunique()), "datum": frame.datum.iloc[0],
            "first_date": frame.date.min().date().isoformat(),
            "last_date": frame.date.max().date().isoformat(),
            "as_of": now.isoformat(), "max_age_days": max_age_days,
            "stale_stations": int((ages > max_age_days).sum()),
            "fresh": bool((ages <= max_age_days).all()),
            "post_2022_rows": int((frame.date >= pd.Timestamp("2023-01-01", tz="UTC")).sum())}


def evaluate(path: str | Path, protocol_path: str | Path) -> dict:
    protocol = json.loads(Path(protocol_path).read_text())
    needed = {"test_start", "test_end", "model_sha256", "baseline", "max_rmse_ratio",
              "min_stations", "min_months", "registered_before", "independent_holdout"}
    if needed - set(protocol):
        raise ValueError(f"Protocol fields missing: {sorted(needed - set(protocol))}")
    start, end = pd.to_datetime([protocol["test_start"], protocol["test_end"]], utc=True)
    if end < start or pd.to_datetime(protocol["registered_before"], utc=True) >= start:
        raise ValueError("Protocol registration must precede the evaluation period")
    if not 0 < float(protocol["max_rmse_ratio"]) <= 1:
        raise ValueError("A prospective improvement gate must require RMSE ratio <= 1")
    for key in ("min_stations", "min_months"):
        value = protocol[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(protocol["baseline"], str) or not protocol["baseline"].strip():
        raise ValueError("The baseline method must be identified")
    if "coverage_bounds" in protocol:
        lo, hi = protocol["coverage_bounds"]
        if not 0 <= lo <= hi <= 1:
            raise ValueError("Coverage bounds must be ordered probabilities")
    frame = pd.read_csv(path, dtype={"station_id": str, "model_sha256": str})
    required = {"station_id", "date", "prediction_origin", "observed_m", "predicted_m",
                "baseline_m", "model_sha256"}
    if required - set(frame):
        raise ValueError(f"Evaluation columns missing: {sorted(required - set(frame))}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Evaluation records must be non-empty and complete")
    if frame.station_id.str.strip().eq("").any():
        raise ValueError("Station must be identified")
    frame["date"] = pd.to_datetime(frame.date, utc=True)
    frame["prediction_origin"] = pd.to_datetime(frame.prediction_origin, utc=True)
    if (frame.prediction_origin >= frame.date).any():
        raise ValueError("Prediction origin must precede every scored observation")
    if (frame.prediction_origin >= start).any():
        raise ValueError("Free-running evaluation cannot assimilate observations inside the holdout")
    if not frame.date.between(start, end).all():
        raise ValueError("Evaluation rows outside the registered period")
    if frame.duplicated(["station_id", "date"]).any():
        raise ValueError("Duplicate station/date predictions")
    if not frame.model_sha256.eq(protocol["model_sha256"]).all():
        raise ValueError("Predictions do not match the registered model")
    values = frame[["observed_m", "predicted_m", "baseline_m"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Evaluation contains non-finite values")
    frame["model_sq"] = (frame.predicted_m - frame.observed_m)**2
    frame["baseline_sq"] = (frame.baseline_m - frame.observed_m)**2
    # Give each station equal weight; retain pooled values as diagnostics.
    station = frame.groupby("station_id")[["model_sq", "baseline_sq"]].mean()
    rmse, baseline = np.sqrt(station.mean().to_numpy())
    ratio = float(rmse / baseline) if baseline > 0 else None
    month_counts = frame.groupby("station_id").date.apply(lambda d: d.dt.strftime("%Y-%m").nunique())
    checks = {"independent_holdout": protocol["independent_holdout"] is True,
              "enough_stations": len(station) >= int(protocol["min_stations"]),
              "enough_months_per_station": bool((month_counts >= int(protocol["min_months"])).all()),
              "beats_baseline": ratio is not None and ratio <= protocol["max_rmse_ratio"]}
    coverage = None
    if {"lower_m", "upper_m"} <= set(frame):
        if not np.isfinite(frame[["lower_m", "upper_m"]].to_numpy(dtype=float)).all():
            raise ValueError("Non-finite interval bounds")
        if (frame.lower_m > frame.upper_m).any():
            raise ValueError("Reversed prediction intervals")
        coverage = float(frame.observed_m.between(frame.lower_m, frame.upper_m).mean())
    if "coverage_bounds" in protocol:
        lo, hi = protocol["coverage_bounds"]
        checks["interval_coverage"] = coverage is not None and lo <= coverage <= hi
    result = {"schema": 1, "task": "free-running-head-validation", "checks": checks,
              "passed": all(checks.values()), "causal_policy_validation": False,
              "rmse_m": float(rmse), "baseline_rmse_m": float(baseline), "rmse_ratio": ratio,
              "bias_m": float((frame.predicted_m - frame.observed_m).mean()),
              "coverage": coverage, "stations": len(station), "rows": len(frame),
              "protocol_sha256": sha256(protocol_path), "predictions_sha256": sha256(path),
              "model_sha256": protocol["model_sha256"], "period": [str(start), str(end)]}
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("qc", "evaluate"))
    ap.add_argument("input", type=Path)
    ap.add_argument("--protocol", type=Path)
    ap.add_argument("--as-of")
    ap.add_argument("--max-age-days", type=int, default=62)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.action == "qc":
        result = observations(args.input, as_of=args.as_of, max_age_days=args.max_age_days)
        passed = result["fresh"]
    else:
        if args.protocol is None:
            ap.error("evaluate needs a registered --protocol")
        result = evaluate(args.input, args.protocol)
        passed = result["passed"]
    atomic_json(args.out, result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
