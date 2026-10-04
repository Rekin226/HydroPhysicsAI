"""Audit a saved fixed-origin head challenge without refitting or reading raw inputs.

The output is retrospective diagnostic evidence, never a prospective pass. Site-level
results contain private observation-derived errors and belong in an ignored directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .release import atomic_json, sha256

METHODS = ("model_raw", "model_datum", "clim", "clim_trend", "persist")


def _metrics(frame: pd.DataFrame) -> dict:
    """Use the challenge's equal-well RMSE and matched rows in every comparison."""
    results = {}
    for method in METHODS:
        error = frame[method] - frame["observed_m"]
        per_well_mse = error.pow(2).groupby(frame["station_id"]).mean()
        results[method] = {
            "rmse_m": float(np.sqrt(per_well_mse.mean())),
            "pooled_rmse_m": float(np.sqrt(error.pow(2).mean())),
            "bias_m": float(error.mean()),
        }
    return results


def _section(frame: pd.DataFrame) -> dict:
    return {"wells": int(frame.station_id.nunique()), "well_months": len(frame),
            "metrics": _metrics(frame)}


def evaluate(challenge_dir: str | Path, event_windows: str | Path | None = None) -> tuple[dict, pd.DataFrame]:
    """Rescore existing challenge rows and reject a mismatched report or site mapping."""
    root = Path(challenge_dir)
    report_path = root / "report.json"
    predictions_path = root / "predictions.csv"
    observations_path = root / "observations.csv"
    report = json.loads(report_path.read_text())
    if report.get("task") != "newly-acquired-retrospective-head-challenge":
        raise ValueError("Expected a saved frozen head challenge")
    if report.get("prospective_validation") is not False:
        raise ValueError("A retrospective diagnostic cannot certify prospective validation")
    origin = pd.Timestamp(report["forecast_origin"])
    predictions = pd.read_csv(predictions_path, dtype={"station_id": str})
    observations = pd.read_csv(observations_path, dtype={"station_id": str})
    needed = {"station_id", "date", "observed_m", *METHODS}
    if needed - set(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(needed - set(predictions))}")
    if {"station_id", "date", "layer"} - set(observations):
        raise ValueError("Observation site, date and layer are required")
    for frame in (predictions, observations):
        if frame.duplicated(["station_id", "date"]).any():
            raise ValueError("Duplicate station/date rows")
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if frame.date.dt.day.ne(1).any():
            raise ValueError("Dates must be monthly starts")
    if predictions.empty:
        raise ValueError("No scored predictions")
    values = predictions[["observed_m", *METHODS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing or non-finite matched predictions")
    if predictions.date.le(origin).any():
        raise ValueError("Scored dates must follow the frozen origin")
    if len(predictions) != report["scored_months"]:
        raise ValueError("Prediction count differs from saved report")
    if predictions.station_id.nunique() != report["stations"]:
        raise ValueError("Well count differs from saved report")
    scored = predictions.merge(observations[["station_id", "date", "layer", "head_m"]],
                               on=["station_id", "date"], how="left", validate="one_to_one")
    if scored[["layer", "head_m"]].isna().any().any():
        raise ValueError("Scored row lacks observation metadata")
    if not np.allclose(scored.observed_m, scored.head_m, rtol=0, atol=1e-9):
        raise ValueError("Scored values disagree with observation metadata")
    if not scored.layer.isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid aquifer layer")
    if scored.groupby("station_id").layer.nunique().gt(1).any():
        raise ValueError("A station cannot change aquifer layer")
    full = _metrics(scored)
    for method in METHODS:
        for metric in ("rmse_m", "pooled_rmse_m", "bias_m"):
            if not np.isclose(full[method][metric], report["metrics"][method][metric],
                              rtol=0, atol=1e-9):
                raise ValueError(f"Saved {method} {metric} cannot be reproduced")
    scored["lead_month"] = ((scored.date.dt.year - origin.year) * 12
                            + scored.date.dt.month - origin.month)
    scored["year"] = scored.date.dt.year
    buckets = ((1, 6), (7, 12), (13, 24), (25, 36), (37, None))
    by_horizon = {}
    for lo, hi in buckets:
        name = f"{lo}+" if hi is None else f"{lo}-{hi}"
        part = scored[scored.lead_month.ge(lo)]
        if hi is not None:
            part = part[part.lead_month.le(hi)]
        if not part.empty:
            by_horizon[name] = _section(part)
    by_year = {str(year): _section(part) for year, part in scored.groupby("year")}
    by_layer = {str(layer): _section(part) for layer, part in scored.groupby("layer")}
    by_event = {}
    if event_windows is not None:
        events = json.loads(Path(event_windows).read_text())
        if not isinstance(events, list):
            raise ValueError("Event windows must be a JSON list")
        for event in events:
            if not isinstance(event, dict) or set(event) != {"name", "start", "end"}:
                raise ValueError("Event requires name, inclusive start and exclusive end")
            name = event["name"]
            start, end = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
            if not isinstance(name, str) or not name.strip() or name in by_event:
                raise ValueError("Event names must be nonempty and unique")
            if start.day != 1 or end.day != 1 or end <= start:
                raise ValueError("Event dates must be ordered monthly starts")
            part = scored[scored.date.ge(start) & scored.date.lt(end)]
            by_event[name] = {"start_inclusive": str(start.date()),
                              "end_exclusive": str(end.date()),
                              **(_section(part) if not part.empty else
                                 {"wells": 0, "well_months": 0, "metrics": None})}
    sites = []
    for station, part in scored.groupby("station_id"):
        metrics = _metrics(part)
        sites.append({"station_id": station, "layer": int(part.layer.iloc[0]),
                      "well_months": len(part),
                      **{f"{method}_rmse_m": metrics[method]["rmse_m"] for method in METHODS}})
    out = {
        "schema": 1, "task": "fixed-origin-head-evidence-diagnostic",
        "prospective_validation": False, "causal_policy_validation": False,
        "forecast_origin": report["forecast_origin"],
        "lead_month_definition": "calendar months after fixed forecast origin; no evaluation-period updates",
        "score_definition": "sqrt(mean of per-well MSE); same scored rows for all methods",
        "source_report_sha256": sha256(report_path),
        "predictions_sha256": sha256(predictions_path),
        "observations_sha256": sha256(observations_path),
        "model_sha256": report["model_sha256"],
        "aggregate": _section(scored), "by_horizon": by_horizon,
        "by_year": by_year, "by_layer": by_layer, "by_event": by_event,
        "event_definitions_sha256": sha256(event_windows) if event_windows else None,
        "event_note": "Optional windows are descriptive; registration timing is not verified",
        "site_scores_file": "site_scores.csv (private local diagnostic)",
        "predictive_interval_coverage": None,
        "coverage_note": "No calibrated prediction bounds in the saved frozen challenge",
        "decision_gate": {"head_skill_passed": bool(report["beats_best_baseline"]),
                          "baseline": report["best_baseline"],
                          "rmse_ratio": report["rmse_ratio"],
                          "policy_skill_passed": False},
    }
    return out, pd.DataFrame(sites)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenge-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True,
                        help="ignored/local directory; includes private site scores")
    parser.add_argument("--events", type=Path,
                        help="optional descriptive event windows JSON; not automatically preregistered")
    args = parser.parse_args(argv)
    result, sites = evaluate(args.challenge_dir, args.events)
    args.out.mkdir(parents=True, exist_ok=True)
    sites.to_csv(args.out / "site_scores.csv", index=False)
    atomic_json(args.out / "evidence.json", result)
    print(json.dumps({"aggregate": result["aggregate"],
                      "decision_gate": result["decision_gate"]}, indent=2))


if __name__ == "__main__":
    main()
