"""Origin-safe AMP ablations for monthly well forecasts (research, not aquifer forcing).

Run from the repository with ``python -m AMP_V2.forecast_experiment --help``.
Full-record AMP exports are deliberately not accepted: their symmetric filters leak
future observations. Every spectral calculation below sees only a trailing origin window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .amp import amp_v1, amp_v2
from .ampg import ampg


def origin_features(series: pd.Series, origin, window_days=180) -> dict:
    """Summarize the last completed record; never close gaps by dropping timestamps.

    The last seven days are excluded from the envelope summary to reduce endpoint
    effects. Symmetric filtering is confined to the past window, including its padding.
    Duty is missing unless the response fit has at least three detected harmonics.
    Neither amplitude times duty nor the fitted duty is a metered pumping volume.
    """
    origin = pd.Timestamp(origin)
    if window_days < 60:
        raise ValueError("Use at least 60 days for the spectral window")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is not None:
        raise ValueError("Expected timestamps in one documented local, timezone-naive frame")
    if series.index.has_duplicates:
        raise ValueError("Resolve duplicate timestamps before feature extraction")
    start = origin - pd.Timedelta(days=window_days)
    past = series.loc[(series.index >= start) & (series.index < origin)].sort_index()
    past = past.where(np.isfinite(past) & (past > -100000))
    med = past.median()
    mad = (past - med).abs().median()
    if np.isfinite(mad) and mad > 0:
        past = past.where((past - med).abs() <= 15 * mad)
    grid = pd.date_range(start, origin, freq="h", inclusive="left")
    hourly = past.resample("h").mean().reindex(grid)
    coverage = float(hourly.notna().mean())
    result = {"coverage": coverage, "amp_v1": np.nan, "amp_v2": np.nan,
              "duty": np.nan, "amp_duty": np.nan, "n_harmonics": 0,
              "status": "insufficient_coverage"}
    if coverage < .95:
        return result
    missing = hourly.isna()
    run = missing.groupby((~missing).cumsum()).sum().max()
    if run > 6:
        result["status"] = "long_gap"
        return result
    hourly = hourly.interpolate(limit=6, limit_area="inside")
    if hourly.isna().any():
        result["status"] = "edge_gap"
        return result
    x = hourly.to_numpy(dtype=float)
    segment = slice(-37 * 24, -7 * 24)
    v1 = amp_v1(x, 24., f_pump=1., bandwidth=.05)
    v2 = amp_v2(x, 24., mode="band")
    result["amp_v1"] = float(np.median(v1["amp"][segment]))
    result["amp_v2"] = float(np.median(v2["amp"][segment])) if v2["found"] else np.nan
    # Use the same detrending convention as the historical AMP-G network experiment.
    hp = x - pd.Series(x).rolling(24 * 15, center=True, min_periods=1).mean().to_numpy()
    # Half a cycle/day of local spectral background, expressed in this window's bins.
    # The old 2001-bin default suppresses adjacent harmonics in a 180-day window.
    floor_win = max(5, 2 * int(.5 * window_days / 2) + 1)
    g = ampg(hp, 24., floor_win=floor_win)
    if g["found"]:
        result["n_harmonics"] = 1 + g["n_harm"]
        if result["n_harmonics"] >= 3 and np.isfinite(g["duty_fit"]):
            result["duty"] = float(g["duty_fit"])
            result["amp_duty"] = result["amp_v1"] * result["duty"]
    result["status"] = "ok"
    return result


def monthly_heads(series):
    """Monthly means with at least 80% hourly coverage; no target interpolation."""
    clean = series.where(np.isfinite(series) & (series > -100000)).resample("h").mean()
    monthly = clean.resample("MS").mean()
    count = clean.resample("MS").count()
    return monthly.where(count >= monthly.index.days_in_month * 24 * .8)


def make_panel(well, series, origins, energy, energy_lag_months=2):
    """One row per origin, forecasting that month's mean using earlier observations.

    Energy release times are unknown in the local cache. The explicit lag is an
    availability assumption, not evidence that historical publication was timely.
    """
    if energy_lag_months < 1:
        raise ValueError("Same-month electricity is unavailable to a monthly forecast")
    heads = monthly_heads(series)
    rows = []
    for origin in origins:
        f = origin_features(series, origin)
        row = {"well": well, "origin": origin, "observed": heads.get(origin, np.nan),
               "previous": heads.get(origin - pd.DateOffset(months=1), np.nan),
               "previous2": heads.get(origin - pd.DateOffset(months=2), np.nan),
               "energy": energy.get(origin - pd.DateOffset(months=energy_lag_months), np.nan),
               **f}
        rows.append(row)
    return pd.DataFrame(rows)


def _ridge_predict(train, target, features, alpha):
    x = train[features].to_numpy(float)
    z = target[features].to_numpy(float)
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-12] = 1.
    x = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    z = np.column_stack([np.ones(len(z)), (z - mean) / scale])
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0
    y = (train.observed - train.seasonal_persistence).to_numpy()
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return target.seasonal_persistence.to_numpy() + z @ weights


def score(frame, prediction):
    """Equal-well RMSE, matching the project's main challenge weighting."""
    errors = pd.DataFrame({"well": frame.well.to_numpy(),
                           "square": (frame.observed.to_numpy() - prediction)**2})
    return float(np.sqrt(errors.groupby("well").square.mean().mean()))


def evaluate(panel, train_end="2021-01-01", test_start="2023-01-01"):
    """Choose ridge strength on validation only; test outcomes never change fitting.

    Common eligible rows are used for every arm and baseline. The duty arm includes
    amplitude plus a missingness flag; unsupported duty is imputed from training only.
    """
    if pd.Timestamp(train_end) >= pd.Timestamp(test_start):
        raise ValueError("Training must end before the test starts")
    p = panel.copy()
    p["origin"] = pd.to_datetime(p.origin)
    if p.duplicated(["well", "origin"]).any():
        raise ValueError("Duplicate well/origin rows")
    p["month"] = p.origin.dt.month
    historical = p[p.origin < train_end]
    climatology = historical.groupby(["well", "month"])["observed"].mean()
    p["seasonal"] = [climatology.get((w, m), np.nan) for w, m in zip(p.well, p.month, strict=True)]
    previous_season = np.array([climatology.get((w, (m - 2) % 12 + 1), np.nan)
                               for w, m in zip(p.well, p.month, strict=True)])
    p["anomaly"] = p.previous - previous_season
    p["seasonal_persistence"] = p.seasonal + p.anomaly
    p["change"] = p.previous - p.previous2
    p["sin"] = np.sin(2 * np.pi * p.month / 12)
    p["cos"] = np.cos(2 * np.pi * p.month / 12)
    p["energy"] = np.log1p(p.energy.where(p.energy >= 0))
    p["duty_missing"] = p.amp_duty.isna().astype(float)
    known_duty = p.loc[p.origin < train_end, "amp_duty"].dropna()
    fill = float(known_duty.median()) if len(known_duty) else 0.
    p["amp_duty"] = p.amp_duty.fillna(fill)
    base = ["anomaly", "change", "sin", "cos", "energy"]
    features = {"electricity": base, "electricity_amp_v1": base + ["amp_v1"],
                "electricity_amp_v2": base + ["amp_v2"],
                "electricity_amp_duty": base + ["amp_v1", "amp_duty", "duty_missing"]}
    required = ["observed", "seasonal_persistence", "seasonal", *base, "amp_v1", "amp_v2"]
    p = p.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    train = p[p.origin < train_end]
    valid = p[(p.origin >= train_end) & (p.origin < test_start)]
    test = p[p.origin >= test_start].copy()
    if min(len(train), len(valid), len(test)) < 12:
        raise ValueError("Need at least 12 matched rows in each split after QC")
    rows, choices = {}, {}
    for name, columns in features.items():
        candidates = {a: score(valid, _ridge_predict(train, valid, columns, a))
                      for a in (.1, 1., 10., 100.)}
        alpha = min(candidates, key=candidates.get)
        test[name] = _ridge_predict(train, test, columns, alpha)
        choices[name] = {"alpha": alpha, "validation_rmse": candidates[alpha]}
    names = ["seasonal", "previous", "seasonal_persistence", *features]
    for name in names:
        rows[name] = {"equal_well_rmse": score(test, test[name].to_numpy()),
                      "per_well_rmse": {str(w): score(g, g[name].to_numpy())
                                        for w, g in test.groupby("well")}}
    report = {"status": "retrospective_exploratory_not_deployment_validation",
              "counts": {"train": len(train), "validation": len(valid), "test": len(test)},
              "test_wells": int(test.well.nunique()), "selection": choices, "scores": rows,
              "score_units": "native head input units; site datum and unit verification pending",
              "test_months": sorted(test.origin.dt.strftime("%Y-%m-%d").unique().tolist()),
              "duty_test_rows": int((test.duty_missing == 0).sum()),
              "qc_status_counts": panel.status.value_counts().to_dict()}
    return report, test


def load_series(paths):
    """Join vintages only when overlapping valid measurements agree."""
    pieces = [pd.read_parquet(p)["value"] for p in paths]
    s = pd.concat(pieces).sort_index()
    duplicates = s[s.index.duplicated(keep=False)]
    if len(duplicates):
        spread = duplicates.groupby(level=0).agg(["min", "max"])
        if ((spread["max"] - spread["min"]) > 1e-6).any():
            raise ValueError("Conflicting overlapping head vintages")
    return s.groupby(level=0).first().astype(float)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--recent", type=Path, required=True)
    parser.add_argument("--energy", type=Path, required=True,
                        help="Local CSV with month-start datetime and irrigation energy_kwh")
    parser.add_argument("--well", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2026-09-01", help="Exclusive target-month boundary")
    parser.add_argument("--energy-lag-months", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "protocol.json").exists():
        raise ValueError("Use a new output directory to preserve the previous protocol")
    energy = pd.read_csv(args.energy, parse_dates=["datetime"]).set_index("datetime").energy_kwh
    if energy.index.has_duplicates or not (energy.index.day == 1).all():
        raise ValueError("Energy must have unique month-start dates")
    origins = pd.date_range(args.start, args.end, freq="MS", inclusive="left")
    sources = {w: [args.historical / f"{w}.parquet", args.recent / f"{w}.parquet"]
               for w in args.well}
    hashes = {str(p): sha256(p) for paths in sources.values() for p in paths}
    protocol = {"wells": args.well, "start": args.start, "end_exclusive": args.end,
                "train_end": "2021-01-01", "test_start": "2023-01-01",
                "energy_lag_months": args.energy_lag_months, "window_days": 180,
                "spectral_noise_floor_width_cpd": .5,
                "source_sha256": hashes, "energy_sha256": sha256(args.energy),
                "code_sha256": {p.name: sha256(p) for p in
                                [Path(__file__), Path(__file__).with_name("amp.py"),
                                 Path(__file__).with_name("ampg.py"),
                                 Path(__file__).with_name("gafd.py")]},
                "limits": ["Electricity publication lag is assumed, not verified",
                           "One-month well forecasts, not a physical pumping or subsidence model",
                           "Already inspected periods; not a prospective trial",
                           "Duty-weighted amplitude is not water volume",
                           "Spatially colocated wells are not independent site replications"]}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    panels = []
    for well, paths in sources.items():
        panel = make_panel(well, load_series(paths), origins, energy, args.energy_lag_months)
        panels.append(panel)
        print(f"Completed {len(panels)}/{len(sources)} wells; QC: {panel.status.value_counts().to_dict()}",
              flush=True)
    panel = pd.concat(panels, ignore_index=True)
    panel.to_parquet(args.out / "features.parquet", index=False)
    report, predictions = evaluate(panel)
    predictions.to_parquet(args.out / "predictions.parquet", index=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"counts": report["counts"], "scores": {
        k: v["equal_well_rmse"] for k, v in report["scores"].items()}}, indent=2))


if __name__ == "__main__":
    main()
