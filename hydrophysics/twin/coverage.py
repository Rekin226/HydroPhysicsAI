"""Interval coverage of a frozen forward run on newly acquired heads.

The challenge (``twin.challenge``) scores the point forecast. This module asks the next
question a decision maker needs answered: when the page shows a range, how often does the
observation fall inside it? Three interval families are scored on identical rows:

- ``ensemble``: the forward run's member spread, ``heads_std`` of the baseline scenario,
  as the page's bands use it;
- ``residual``: each well's fitted-period residual standard deviation after its datum
  offset (fitted months only, nothing from the held-out period);
- ``combined``: the two in quadrature.

The climatology baseline gets the same ``residual`` treatment so the model's interval is
compared with a simple one. Datum offsets, residual spreads and baselines use only months
before the forecast origin. This is a retrospective check on a frozen model, not a
preregistered prospective validation, and it says nothing about policy effects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .challenge import monthly_observations
from .drift_diag import fair_baselines
from .grid import FanGrid
from .heads import build_head_field
from .release import atomic_json, sha256

# two-sided standard-normal quantiles of the nominal central levels
Z = {50: 0.6744897501960817, 80: 1.2815515655446004, 90: 1.6448536269514722,
     95: 1.959963984540054}


def score_coverage(observed: np.ndarray, predicted: np.ndarray, sigma: np.ndarray,
                   valid: np.ndarray, keep: np.ndarray) -> dict:
    """Coverage of ``predicted +- z * sigma`` on the rows marked ``valid``.

    ``picp_<L>`` weights every kept well equally (mean of per-well coverages);
    ``picp_pooled_<L>`` weights rows. ``inflation_90`` is the factor the spread would
    need for the nominal 90 % interval to cover 90 % of the rows: 1 means calibrated,
    above 1 under-dispersed, below 1 over-dispersed.
    """
    if not (observed.shape == predicted.shape == sigma.shape == valid.shape):
        raise ValueError("observed, predicted, sigma and valid must share a shape")
    if not valid.any():
        raise ValueError("No valid rows to score")
    if (sigma[valid] <= 0).any() or not np.isfinite(sigma[valid]).all():
        raise ValueError("sigma must be finite and positive on valid rows")
    err = predicted - observed
    out = {}
    for level, z in Z.items():
        inside = np.abs(err) <= z * sigma
        per_well = np.where(valid, inside, np.nan)
        with np.errstate(invalid="ignore"):
            cov_w = np.nanmean(per_well, axis=1)[keep]
        out[f"picp_{level}"] = float(np.nanmean(cov_w))
        out[f"picp_pooled_{level}"] = float(inside[valid].mean())
        out[f"mean_width_{level}_m"] = float(2 * z * sigma[valid].mean())
    standardized = np.abs(err[valid]) / sigma[valid]
    out["q90_abs_standardized_error"] = float(np.quantile(standardized, 0.9))
    out["inflation_90"] = out["q90_abs_standardized_error"] / Z[90]
    out["sigma_mean_m"] = float(sigma[valid].mean())
    out["rmse_m"] = float(np.sqrt(np.mean(err[valid] ** 2)))
    return out


def _residual_sd(historical: np.ndarray, predicted: np.ndarray, fit_months: int,
                 min_months: int = 12) -> np.ndarray:
    """Per-well standard deviation of the fitted-period residual (NaN below support)."""
    res = historical[:, :fit_months] - predicted[:, :fit_months]
    n = np.isfinite(res).sum(axis=1)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(res, axis=1, ddof=1)
    sd[n < min_months] = np.nan
    return sd


def assemble(forward: Path, stations: Path, historical_wells: Path,
             new_wells: Path | None = None, end: str | None = None) -> dict:
    """The forward run's baseline heads at every QC'd well, with the historical record,
    the ensemble spread and (optionally) newly acquired observations, all on the run's
    monthly axis -> dict of arrays ``(W, T)`` plus ``ids``, ``dates``, ``origin``."""
    with np.load(forward, allow_pickle=False) as saved:
        fw = {k: saved[k] for k in saved.files}
    dates = pd.DatetimeIndex(pd.to_datetime(fw["dates"])).to_period("M").to_timestamp()
    origin = int(fw["origin"])
    fit = origin + 1
    grid = FanGrid(int(fw["nx"]), int(fw["ny"]), float(fw["dx"]), float(fw["x0"]),
                   float(fw["y0"]), fw["mask"])
    hf = build_head_field(str(historical_wells), pd.read_parquet(stations),
                          t0=str(dates[0].date()),
                          t1=str((dates[origin] + pd.offsets.MonthBegin()).date()))
    selected = [(i, grid.active_index(*xy)) for i, xy in enumerate(hf.xy)]
    selected = [(i, c) for i, c in selected if c is not None]
    ids = [hf.sids[i] for i, _ in selected]
    scenario = list(fw["scenario_names"]).index("baseline")
    W, T = len(ids), len(dates)
    historical = np.full((W, T), np.nan)
    new = np.full((W, T), np.nan)
    predicted = np.empty((W, T))
    sigma_ens = np.empty((W, T))
    layers = np.empty(W, dtype=int)
    sources = []
    for w, (i, cell) in enumerate(selected):
        historical[w, :fit] = hf.heads[i]
        layer = int(hf.layers[i]) - 1
        layers[w] = layer + 1
        predicted[w] = fw["heads_mean"][scenario, layer, cell]
        sigma_ens[w] = fw["heads_std"][scenario, layer, cell]
        if new_wells is not None:
            path = new_wells / f"{hf.sids[i]}.parquet"
            if path.is_file():
                new[w] = monthly_observations(path).reindex(dates).to_numpy()
                sources.append({"station_id": hf.sids[i], "sha256": sha256(path)})
    if end is not None:
        new[:, dates >= pd.Timestamp(end)] = np.nan
    return {"ids": ids, "dates": dates, "origin": origin, "fit": fit, "layers": layers,
            "historical": historical, "new": new, "predicted": predicted,
            "sigma_ens": sigma_ens, "sources": sources, "update_method": _update_method(fw),
            "model_sha256": sha256(forward)}


def fitted_offsets(historical: np.ndarray, predicted: np.ndarray, fit: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Per-well datum offset (mean fitted-period residual) and its support."""
    residual = historical[:, :fit] - predicted[:, :fit]
    count = np.isfinite(residual).sum(axis=1)
    offsets = np.divide(np.nansum(residual, axis=1), count,
                        out=np.full(len(count), np.nan), where=count > 0)
    return offsets, count


def run(forward: Path, stations: Path, historical_wells: Path, new_wells: Path,
        out: Path, end: str, min_months: int = 12) -> dict:
    a = assemble(forward, stations, historical_wells, new_wells, end)
    ids, dates, origin, fit = a["ids"], a["dates"], a["origin"], a["fit"]
    historical, new, predicted, sigma_ens = (a["historical"], a["new"], a["predicted"],
                                             a["sigma_ens"])
    W, T = historical.shape
    sources = a["sources"]

    # datum offset and residual spread from the fitted months only (as the challenge)
    train = historical.copy()
    offsets, count = fitted_offsets(train, predicted, fit)
    model = predicted + offsets[:, None]
    clim = fair_baselines(train, fit, month_offset=0)["clim"]
    sd_model = _residual_sd(train, model, fit)
    sd_clim = _residual_sd(train, clim, fit)

    valid = np.isfinite(new)
    valid[:, :fit] = False
    for arr in (model, clim, sigma_ens):
        valid &= np.isfinite(arr)
    valid &= np.isfinite(sd_model)[:, None] & np.isfinite(sd_clim)[:, None]
    keep = (valid.sum(axis=1) >= min_months) & (count >= 24)
    valid[~keep] = False
    if not valid.any():
        raise ValueError("No wells have sufficient historical and new observations")
    floor = 1e-3
    families = {
        "model_ensemble": (model, np.maximum(sigma_ens, floor)),
        "model_residual": (model, np.broadcast_to(sd_model[:, None], (W, T))),
        "model_combined": (model, np.sqrt(np.maximum(sigma_ens, floor) ** 2
                                          + sd_model[:, None] ** 2)),
        "clim_residual": (clim, np.broadcast_to(sd_clim[:, None], (W, T))),
    }
    metrics = {name: score_coverage(new, pred, np.asarray(sig), valid, keep)
               for name, (pred, sig) in families.items()}
    report = {
        "schema": 1, "task": "retrospective-interval-coverage",
        "prospective_validation": False, "causal_policy_validation": False,
        "model_sha256": a["model_sha256"], "forecast_origin": str(dates[origin].date()),
        "end_exclusive": end, "stations": int(keep.sum()), "scored_months": int(valid.sum()),
        "minimum_new_months_per_station": min_months,
        "update_method": a["update_method"],
        "intervals": "central two-sided normal intervals: prediction +- z * sigma",
        "ensemble_sigma": "member spread of the baseline scenario (heads_std); parameter "
                          "sets and initial fields only, no forcing or structural error",
        "residual_sigma": "per-well fitted-period residual sd after the datum offset; "
                          "fitted months only",
        "metrics": metrics, "observations": sources,
    }
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    z90 = Z[90]
    for w, t in zip(*np.where(valid), strict=True):
        rows.append({"station_id": ids[w], "date": str(dates[t].date()),
                     "prediction_origin": str(dates[origin].date()),
                     "observed_m": new[w, t], "predicted_m": model[w, t],
                     "baseline_m": clim[w, t], "model_sha256": report["model_sha256"],
                     "sigma_ensemble_m": sigma_ens[w, t], "sigma_residual_m": sd_model[w],
                     "lower_m": model[w, t] - z90 * sd_model[w],
                     "upper_m": model[w, t] + z90 * sd_model[w]})
    pd.DataFrame(rows).to_csv(out / "evaluation.csv", index=False)
    atomic_json(out / "report.json", report)
    return report


def _update_method(fw: dict) -> str:
    """The observation update the run recorded, or 'field (pre-fix default)'."""
    try:
        options = json.loads(str(fw["forward_options"]))
    except (KeyError, ValueError):
        return "unknown"
    return str(options.get("update_method", "field (pre-fix default)"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forward", required=True, type=Path)
    ap.add_argument("--stations", required=True, type=Path)
    ap.add_argument("--historical-wells", required=True, type=Path)
    ap.add_argument("--new-wells", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--end", required=True, help="exclusive first day of a month")
    ap.add_argument("--min-months", type=int, default=12)
    args = ap.parse_args(argv)
    result = run(args.forward, args.stations, args.historical_wells, args.new_wells,
                 args.out, args.end, args.min_months)
    print(json.dumps({k: v for k, v in result.items() if k != "observations"}, indent=2))


if __name__ == "__main__":
    main()
