"""Register a prospective head-forecast test before its observations exist, and score it
when they do.

``register`` freezes, with hashes, the frozen run's datum-adjusted well predictions, the
climatology baseline and a prediction interval for a future window, together with the
protocol ``twin.validation.evaluate`` will apply. Nothing in the window is observed at
registration; the manifest records when it was written. ``score`` joins the observations
that have since arrived and runs the locked evaluation. Only this path can produce a
prospective verdict; every other score in the repository is retrospective.

The interval is the ``combined`` family of ``twin.coverage`` (ensemble spread and the
well's fitted-period residual spread in quadrature), the one that covered 92 % of the
2023-26 observations at a nominal 90 %. Offsets, spreads and the baseline use fitted
months only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .challenge import monthly_observations
from .coverage import Z, _residual_sd, assemble, fitted_offsets
from .drift_diag import fair_baselines
from .release import atomic_json, sha256
from .validation import evaluate


def register(forward: Path, stations: Path, historical_wells: Path, out: Path,
             test_start: str, test_end: str, registered_on: str | None = None,
             min_stations: int = 100, min_months: int = 6, level: int = 90) -> dict:
    a = assemble(forward, stations, historical_wells)
    dates, fit = a["dates"], a["fit"]
    historical, predicted, sigma_ens = a["historical"], a["predicted"], a["sigma_ens"]
    offsets, count = fitted_offsets(historical, predicted, fit)
    model = predicted + offsets[:, None]
    clim = fair_baselines(historical.copy(), fit, month_offset=0)["clim"]
    sd = _residual_sd(historical, model, fit)
    sigma = np.sqrt(np.maximum(sigma_ens, 1e-3) ** 2 + sd[:, None] ** 2)
    z = Z[level]
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    window = (dates >= start) & (dates <= end)
    if not window.any():
        raise ValueError("The forward run does not cover the test window")
    keep = (count >= 24) & np.isfinite(sd) & np.isfinite(offsets)
    rows = []
    for w in np.flatnonzero(keep):
        for t in np.flatnonzero(window):
            if not (np.isfinite(model[w, t]) and np.isfinite(clim[w, t])):
                continue
            rows.append({"station_id": a["ids"][w], "layer": int(a["layers"][w]),
                         "date": str(dates[t].date()),
                         "prediction_origin": str(dates[a["origin"]].date()),
                         "predicted_m": float(model[w, t]), "baseline_m": float(clim[w, t]),
                         "lower_m": float(model[w, t] - z * sigma[w, t]),
                         "upper_m": float(model[w, t] + z * sigma[w, t]),
                         "model_sha256": a["model_sha256"]})
    frame = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "predictions.csv", index=False)
    today = registered_on or dt.date.today().isoformat()
    if pd.Timestamp(today) >= start:
        raise ValueError("Registration must precede the test window")
    protocol = {
        "schema": 1, "task": "free-running-head-validation",
        "test_start": test_start, "test_end": test_end, "registered_before": today,
        "independent_holdout": True, "model_sha256": a["model_sha256"],
        "baseline": "month-of-year climatology of each well's fitted months before the origin",
        "max_rmse_ratio": 1.0, "min_stations": int(min_stations), "min_months": int(min_months),
        "coverage_bounds": [0.85, 0.95], "interval": f"central {level} %, combined sigma",
        "update_method": a["update_method"], "forecast_origin": str(dates[a["origin"]].date()),
        "datum": "per-well mean fitted-period residual; fitted months only",
        "forcing_after_origin": "month-of-year climatology of the 2012-2022 record (baseline scenario)",
        "note": "Observations in the window did not exist at registration. Score with "
                "twin.prospective score once the window has closed; do not refit, re-select "
                "or re-register for this window.",
    }
    atomic_json(out / "protocol.json", protocol)
    manifest = {"registered_on": today, "stations": int(frame.station_id.nunique()),
                "rows": len(frame), "predictions_sha256": sha256(out / "predictions.csv"),
                "protocol_sha256": sha256(out / "protocol.json"),
                "forward": str(forward), "model_sha256": a["model_sha256"]}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def score(registration: Path, new_wells: Path, out: Path, as_of: str | None = None) -> dict:
    """Join the observations that have arrived and run the locked evaluation. Fails if the
    registered files changed since registration."""
    manifest = json.loads((registration / "manifest.json").read_text())
    for key, name in (("predictions_sha256", "predictions.csv"), ("protocol_sha256", "protocol.json")):
        if sha256(registration / name) != manifest[key]:
            raise ValueError(f"{name} changed since registration")
    frame = pd.read_csv(registration / "predictions.csv", dtype={"station_id": str})
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date))
    observed = np.full(len(frame), np.nan)
    for sid, grp in frame.groupby("station_id"):
        path = new_wells / f"{sid}.parquet"
        if not path.is_file():
            continue
        series = monthly_observations(path)
        if as_of is not None:
            series = series[series.index < pd.Timestamp(as_of)]
        observed[grp.index] = series.reindex(dates[grp.index]).to_numpy()
    frame["observed_m"] = observed
    frame = frame[np.isfinite(frame.observed_m)]
    protocol = json.loads((registration / "protocol.json").read_text())
    months = frame.groupby("station_id").date.nunique()
    frame = frame[frame.station_id.isin(months[months >= int(protocol["min_months"])].index)]
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "evaluation.csv", index=False)
    result = evaluate(out / "evaluation.csv", registration / "protocol.json")
    result["registration_manifest"] = manifest
    atomic_json(out / "report.json", result)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--forward", required=True, type=Path)
    r.add_argument("--stations", required=True, type=Path)
    r.add_argument("--historical-wells", required=True, type=Path)
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--test-start", required=True)
    r.add_argument("--test-end", required=True)
    r.add_argument("--min-stations", type=int, default=100)
    r.add_argument("--min-months", type=int, default=6)
    s = sub.add_parser("score")
    s.add_argument("--registration", required=True, type=Path)
    s.add_argument("--new-wells", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--as-of", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "register":
        res = register(args.forward, args.stations, args.historical_wells, args.out,
                       args.test_start, args.test_end, min_stations=args.min_stations,
                       min_months=args.min_months)
    else:
        res = score(args.registration, args.new_wells, args.out, args.as_of)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
