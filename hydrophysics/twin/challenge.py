"""Challenge a frozen forecast against newly acquired heads, without refitting it.

This retrospective test is not a preregistered prospective evaluation. Datum offsets
and all simple baselines use only the historical cache ending at the forecast origin.
New observations never change the model, offset, or baseline fit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .drift_diag import fair_baselines
from .grid import FanGrid
from .heads import build_head_field
from .release import atomic_json, sha256


def monthly_observations(path: Path, min_days: int = 20) -> pd.Series:
    """Reject API sentinels and require 12 observed hours on at least 20 days/month."""
    frame = pd.read_parquet(path)
    if "datetime" in frame:
        frame = frame.set_index("datetime")
    series = pd.to_numeric(frame.value, errors="coerce")
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.where(series.between(-1000, 1000)).dropna()
    if series.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    hourly = series.resample("h").mean()
    daily = hourly.resample("D").mean().where(hourly.resample("D").count() >= 12)
    return daily.resample("MS").mean().where(daily.resample("MS").count() >= min_days)


def score(prediction: np.ndarray, historical: np.ndarray, new: np.ndarray,
          fit_months: int, min_months: int = 12) -> tuple[dict, dict]:
    """Equal-well errors on the same new observations, with train-only datum correction."""
    if prediction.shape != new.shape or historical.shape != new.shape:
        raise ValueError("Predictions and observations must share well/month axes")
    if not 0 < fit_months < new.shape[1]:
        raise ValueError("A non-empty historical and held-out interval is required")
    if min_months < 1:
        raise ValueError("min_months must be positive")
    train = historical.copy()
    train[:, fit_months:] = np.nan
    baseline = fair_baselines(train, fit_months, month_offset=0)
    residual = train[:, :fit_months] - prediction[:, :fit_months]
    count = np.isfinite(residual).sum(axis=1)
    offsets = np.divide(np.nansum(residual, axis=1), count,
                        out=np.full(len(count), np.nan), where=count > 0)
    predictions = {"model_raw": prediction, "model_datum": prediction + offsets[:, None],
                   **baseline}
    valid = np.isfinite(new)
    valid[:, :fit_months] = False
    for values in predictions.values():
        valid &= np.isfinite(values)
    keep = (valid.sum(axis=1) >= min_months) & (count >= 24)
    valid[~keep] = False
    if not valid.any():
        raise ValueError("No wells have sufficient historical and new observations")
    metrics = {}
    for name, values in predictions.items():
        error = np.where(valid, values - new, np.nan)
        per_well_mse = np.nansum(error**2, axis=1)[keep] / valid.sum(axis=1)[keep]
        metrics[name] = {"rmse_m": float(np.sqrt(per_well_mse.mean())),
                         "pooled_rmse_m": float(np.sqrt(np.nanmean(error**2))),
                         "bias_m": float(np.nanmean(error))}
    best = min(baseline, key=lambda k: metrics[k]["rmse_m"])
    denominator = metrics[best]["rmse_m"]
    ratio = metrics["model_datum"]["rmse_m"] / denominator if denominator > 0 else None
    report = {"schema": 1, "task": "newly-acquired-retrospective-head-challenge",
              "prospective_validation": False, "causal_policy_validation": False,
              "metrics": metrics, "best_baseline": best, "rmse_ratio": ratio,
              "beats_best_baseline": ratio is not None and ratio < 1,
              "stations": int(keep.sum()), "scored_months": int(valid.sum()),
              "minimum_new_months_per_station": min_months,
              "datum_adjustment": "constant residual mean estimated only before forecast origin",
              "uncertainty": "No predictive coverage claim; structural and forcing errors omitted"}
    return report, {"predictions": predictions, "valid": valid, "offsets": offsets}


def run(forward: Path, stations: Path, historical_wells: Path, new_wells: Path,
        out: Path, end: str, min_months: int = 12) -> dict:
    with np.load(forward, allow_pickle=False) as saved:
        fw = {k: saved[k] for k in saved.files}
    dates = pd.DatetimeIndex(pd.to_datetime(fw["dates"])).to_period("M").to_timestamp()
    origin = int(fw["origin"])
    grid = FanGrid(int(fw["nx"]), int(fw["ny"]), float(fw["dx"]), float(fw["x0"]),
                   float(fw["y0"]), fw["mask"])
    hf = build_head_field(str(historical_wells), pd.read_parquet(stations),
                          t0=str(dates[0].date()),
                          t1=str((dates[origin] + pd.offsets.MonthBegin()).date()))
    selected = [(i, grid.active_index(*xy)) for i, xy in enumerate(hf.xy)]
    selected = [(i, c) for i, c in selected if c is not None]
    ids = [hf.sids[i] for i, _ in selected]
    historical = np.full((len(ids), len(dates)), np.nan)
    new = np.full_like(historical, np.nan)
    prediction = np.empty_like(historical)
    scenario = list(fw["scenario_names"]).index("baseline")
    sources = []
    for w, (i, cell) in enumerate(selected):
        historical[w, :origin + 1] = hf.heads[i]
        prediction[w] = fw["heads_mean"][scenario, int(hf.layers[i]) - 1, cell]
        path = new_wells / f"{hf.sids[i]}.parquet"
        if path.is_file():
            new[w] = monthly_observations(path).reindex(dates).to_numpy()
            sources.append({"station_id": hf.sids[i], "sha256": sha256(path)})
    new[:, dates >= pd.Timestamp(end)] = np.nan
    report, details = score(prediction, historical, new, origin + 1, min_months)
    report.update(model_sha256=sha256(forward), observations=sources,
                  forecast_origin=str(dates[origin].date()), end_exclusive=end,
                  vertical_reference="vendor water-level reference; not independently surveyed",
                  qc="-1000..1000 m sentinel screen; 12 hours/day; 20 days/month; equal-day means")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    observation_rows = []
    for w, t in zip(*np.where(np.isfinite(new)), strict=True):
        observation_rows.append({"station_id": ids[w], "date": str(dates[t].date()),
                                 "head_m": new[w, t], "layer": int(hf.layers[selected[w][0]]),
                                 "datum": "vendor water-level reference (unverified)"})
    for w, t in zip(*np.where(details["valid"]), strict=True):
        records.append({"station_id": ids[w], "date": str(dates[t].date()),
                        "observed_m": new[w, t],
                        **{name: values[w, t] for name, values in details["predictions"].items()}})
    pd.DataFrame(records).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(observation_rows).to_csv(out / "observations.csv", index=False)
    atomic_json(out / "report.json", report)
    return report


def main(argv=None):
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward", required=True, type=Path)
    ap.add_argument("--stations", required=True, type=Path)
    ap.add_argument("--historical-wells", required=True, type=Path)
    ap.add_argument("--new-wells", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--end", required=True, help="exclusive first day of a month; omit partial months")
    ap.add_argument("--min-months", type=int, default=12)
    args = ap.parse_args(argv)
    result = run(args.forward, args.stations, args.historical_wells, args.new_wells,
                 args.out, args.end, args.min_months)
    print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    main()
