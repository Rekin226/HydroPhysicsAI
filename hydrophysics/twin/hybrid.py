"""A statistical layer on the frozen physics for head forecasts, fitted before the origin.

``twin.challenge`` showed the frozen free run (climatology forcing after the origin) losing
to month-of-year climatology on the 2023-26 acquisitions. Nothing after the origin is
known to the model but its own state, so the only honest forecast ingredients are the
well's climatology, the anomaly observed at the origin and how fast anomalies fade, and
the physics' own evolution of its updated state. This module builds those from the fitted
months only and scores the combinations exactly as the challenge scores the model:

- ``clim``: month-of-year climatology (the challenge's best baseline);
- ``stat``: ``clim`` plus the last fitted anomaly decayed with the well's lag-1
  autocorrelation (``rho ** k``), the classic anomaly-persistence forecast;
- ``phys``: ``clim`` plus the physics' anomaly (datum-adjusted free run minus its own
  fitted-period month-of-year mean) -- the physics persistence, on the observed climatology;
- ``mix``: ``clim`` plus the average of the two anomalies.

Skill is reported per calendar year because anomaly memory fades: a hybrid can only beat
climatology while the origin is still remembered. The coefficients are fixed a priori
(``rho`` from the fitted months, equal weights), so the 2023-26 result is a retrospective
test of pre-specified forecasts, not a fit; the ``in_sample_diagnostic`` block, which
regresses the anomaly weights on the test observations, is labelled as such and must not
be read as skill. No forecast here is prospective; ``twin.prospective`` is the only
prospective path.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .coverage import Z, _residual_sd, assemble, fitted_offsets, score_coverage
from .drift_diag import fair_baselines
from .release import atomic_json, sha256

CANDIDATES = ("clim", "stat", "phys", "mix")


def lag1_rho(anom: np.ndarray, fit: int, min_pairs: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Per-well lag-1 regression-through-origin coefficient of the fitted-month anomaly,
    clipped to ``[0, 0.99]``; NaN below ``min_pairs`` consecutive observed pairs."""
    a = np.asarray(anom, dtype="float64")[:, :fit]
    x, y = a[:, :-1], a[:, 1:]
    ok = np.isfinite(x) & np.isfinite(y)
    n = ok.sum(axis=1)
    num = np.where(ok, x * y, 0.0).sum(axis=1)
    den = np.where(ok, x * x, 0.0).sum(axis=1)
    rho = np.divide(num, den, out=np.full(len(n), np.nan), where=den > 0)
    rho = np.clip(rho, 0.0, 0.99)
    rho[n < min_pairs] = np.nan
    return rho, n


def persisted_anomaly(anom: np.ndarray, fit: int, rho: np.ndarray, max_gap: int = 3
                      ) -> tuple[np.ndarray, np.ndarray]:
    """``(W, T)`` anomaly persisted from the last fitted observation, ``rho ** k`` times
    it ``k`` months later; zero (no information) where the last fitted observation is more
    than ``max_gap`` months before the origin or ``rho`` is unknown. Also returns the index
    of the month persisted from (-1 when none)."""
    anom = np.asarray(anom, dtype="float64")
    W, T = anom.shape
    out = np.zeros((W, T))
    last = np.full(W, -1)
    k_all = np.arange(T, dtype="float64")
    for w in range(W):
        idx = np.flatnonzero(np.isfinite(anom[w, :fit]))
        if idx.size == 0 or not np.isfinite(rho[w]):
            continue
        t_last = int(idx[-1])
        if fit - 1 - t_last > max_gap:
            continue
        last[w] = t_last
        k = k_all - t_last
        out[w] = np.where(k >= 0, rho[w] ** np.maximum(k, 0.0) * anom[w, t_last], 0.0)
    return out, last


def month_of_year_mean(x: np.ndarray, ok: np.ndarray, fit: int, moy: np.ndarray) -> np.ndarray:
    """``(W, T)`` month-of-year mean of ``x`` over the fitted months flagged ``ok``, with a
    month never observed falling back to the well's fitted mean (as ``fair_baselines``)."""
    x = np.asarray(x, dtype="float64")
    W, T = x.shape
    sel = ok.copy()
    sel[:, fit:] = False
    out = np.full((W, T), np.nan)
    for w in range(W):
        if not sel[w].any():
            continue
        m_fit = float(x[w, sel[w]].mean())
        eff = np.full(12, np.nan)
        for mth in range(12):
            s = sel[w] & (moy == mth)
            if s.any():
                eff[mth] = x[w, s].mean()
        out[w] = np.where(np.isfinite(eff), eff, m_fit)[moy]
    return out


def candidates(historical: np.ndarray, predicted: np.ndarray, fit: int, moy: np.ndarray
               ) -> tuple[dict[str, np.ndarray], dict]:
    """The four pre-specified forecasts on ``(W, T)`` arrays and their fitted ingredients.

    ``historical`` carries observations before the origin only (NaN after); ``predicted``
    is the run's free-running head at the well; ``fit`` is the number of fitted months;
    ``moy`` the calendar month index of each column."""
    historical = np.asarray(historical, dtype="float64")
    train = historical.copy()
    train[:, fit:] = np.nan
    clim = fair_baselines(train, fit, month_offset=0)["clim"]
    offsets, count = fitted_offsets(train, predicted, fit)
    model = predicted + offsets[:, None]
    obs_ok = np.isfinite(train)
    anom_obs = train - clim
    rho, n_pairs = lag1_rho(anom_obs, fit)
    pooled = float(np.nanmedian(rho)) if np.isfinite(rho).any() else 0.0
    rho_used = np.where(np.isfinite(rho), rho, pooled)
    a_stat, last = persisted_anomaly(anom_obs, fit, rho_used)
    a_phys = model - month_of_year_mean(model, obs_ok & np.isfinite(model), fit, moy)
    a_phys = np.where(np.isfinite(a_phys), a_phys, 0.0)
    cands = {"clim": clim, "stat": clim + a_stat, "phys": clim + a_phys,
             "mix": clim + 0.5 * (a_stat + a_phys)}
    info = {"rho": rho, "rho_used": rho_used, "rho_pooled": pooled, "n_pairs": n_pairs,
            "last_fitted_index": last, "offsets": offsets, "count": count,
            "anomaly_stat": a_stat, "anomaly_phys": a_phys, "model_datum": model,
            "sd_anomaly": _residual_sd(train, clim, fit)}
    return cands, info


def _equal_well_rmse(err: np.ndarray, valid: np.ndarray, keep: np.ndarray) -> dict:
    e = np.where(valid, err, np.nan)
    n = valid.sum(axis=1)
    mse = np.divide(np.nansum(e ** 2, axis=1), n, out=np.full(len(n), np.nan), where=n > 0)
    return {"rmse_m": float(np.sqrt(np.nanmean(mse[keep]))),
            "pooled_rmse_m": float(np.sqrt(np.nanmean(e ** 2))),
            "bias_m": float(np.nanmean(e))}


def score_candidates(cands: dict[str, np.ndarray], new: np.ndarray, fit: int,
                     dates: pd.DatetimeIndex, min_months: int = 12, count: np.ndarray | None = None
                     ) -> tuple[dict, np.ndarray, np.ndarray]:
    """Equal-well errors of every candidate on the same post-origin observations, overall
    and per calendar year (a well enters a year with at least three scored months)."""
    valid = np.isfinite(new)
    valid[:, :fit] = False
    for v in cands.values():
        valid &= np.isfinite(v)
    keep = valid.sum(axis=1) >= min_months
    if count is not None:
        keep &= count >= 24
    valid[~keep] = False
    if not valid.any():
        raise ValueError("No wells have sufficient historical and new observations")
    metrics = {k: _equal_well_rmse(v - new, valid, keep) for k, v in cands.items()}
    years = {}
    for year in sorted(set(dates[valid.any(axis=0)].year)):
        vy = valid & (dates.year == year)[None, :]
        ky = vy.sum(axis=1) >= 3
        vy[~ky] = False
        if not vy.any():
            continue
        years[str(year)] = {k: _equal_well_rmse(v - new, vy, ky)["rmse_m"]
                            for k, v in cands.items()}
        years[str(year)]["stations"] = int(ky.sum())
        years[str(year)]["months"] = int(vy.sum())
    return {"metrics": metrics, "per_year": years, "stations": int(keep.sum()),
            "scored_months": int(valid.sum())}, valid, keep


def in_sample_weights(info: dict, clim: np.ndarray, new: np.ndarray, valid: np.ndarray) -> dict:
    """Least-squares weights of the two anomalies on the TEST observations -- a diagnostic
    of how much information each carries, fitted in sample, never a forecast."""
    y = (new - clim)[valid]
    X = np.column_stack([info["anomaly_stat"][valid], info["anomaly_phys"][valid]])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return {"weight_stat": float(coef[0]), "weight_phys": float(coef[1]),
            "rmse_pooled_with_fitted_weights_m": float(np.sqrt(np.mean(resid ** 2))),
            "rmse_pooled_clim_m": float(np.sqrt(np.mean(y ** 2))),
            "note": "weights regressed on the 2023-26 observations themselves; in sample, "
                    "not a forecast and not selectable without a new prospective window"}


def run(forward: Path, stations: Path, historical_wells: Path, new_wells: Path, out: Path,
        end: str, min_months: int = 12) -> dict:
    a = assemble(forward, stations, historical_wells, new_wells, end)
    dates, fit, new = a["dates"], a["fit"], a["new"]
    moy = np.arange(len(dates)) % 12          # the challenge's grouping (month_offset=0)
    cands, info = candidates(a["historical"], a["predicted"], fit, moy)
    cands["model_datum"] = info["model_datum"]
    scores, valid, keep = score_candidates(cands, new, fit, dates, min_months, info["count"])
    clim_rmse = scores["metrics"]["clim"]["rmse_m"]
    ratios = {k: v["rmse_m"] / clim_rmse for k, v in scores["metrics"].items()}
    sd = info["sd_anomaly"]
    sig_ok = np.isfinite(sd)
    v2 = valid & sig_ok[:, None]
    k2 = keep & sig_ok
    sigma = np.broadcast_to(np.where(sig_ok, sd, 1.0)[:, None], new.shape)
    coverage = {k: score_coverage(new, cands[k], np.asarray(sigma), v2, k2)["picp_90"]
                for k in CANDIDATES}
    rho = info["rho"]
    report = {
        "schema": 1, "task": "retrospective-hybrid-head-forecast",
        "prospective_validation": False, "causal_policy_validation": False,
        "model_sha256": a["model_sha256"], "forecast_origin": str(dates[a["origin"]].date()),
        "end_exclusive": end, "update_method": a["update_method"],
        "stations": scores["stations"], "scored_months": scores["scored_months"],
        "candidates": {
            "clim": "month-of-year mean of the fitted months (challenge baseline)",
            "stat": "clim + rho**k * last fitted anomaly (rho: per-well lag-1 coefficient "
                    "of the fitted anomalies, pooled median where fewer than 24 pairs)",
            "phys": "clim + (datum-adjusted free run - its fitted-period month-of-year mean)",
            "mix": "clim + 0.5 * (stat anomaly + phys anomaly)",
            "model_datum": "the challenge's datum-adjusted free run, for reference"},
        "pre_specified": "rho from fitted months; equal weights; nothing tuned on the test",
        "metrics": scores["metrics"], "rmse_ratio_vs_clim": ratios,
        "per_year_rmse_m": scores["per_year"],
        "rho": {"median_own": float(np.nanmedian(rho)), "wells_with_own": int(np.isfinite(rho).sum()),
                "pooled": info["rho_pooled"],
                "p10_p90": [float(np.nanquantile(rho, 0.1)), float(np.nanquantile(rho, 0.9))]},
        "interval_90_coverage_with_fitted_anomaly_sd": coverage,
        "in_sample_diagnostic": in_sample_weights(info, cands["clim"], new, valid),
    }
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for w, t in zip(*np.where(valid), strict=True):
        rows.append({"station_id": a["ids"][w], "date": str(dates[t].date()),
                     "observed_m": new[w, t],
                     **{k: cands[k][w, t] for k in (*CANDIDATES, "model_datum")}})
    pd.DataFrame(rows).to_csv(out / "evaluation.csv", index=False)
    atomic_json(out / "report.json", report)
    return report


def register(forward: Path, stations: Path, historical_wells: Path, new_wells: Path, out: Path,
             test_start: str, test_end: str, registered_on: str | None = None,
             retrospective: Path | None = None, min_stations: int = 100, min_months: int = 6,
             level: int = 90, max_gap: int = 3) -> dict:
    """Register the ``stat`` forecast prospectively for a future window.

    Climatology, ``rho`` and the anomaly spread come from every observation before the
    window (the historical record and the acquisitions since); the anomaly is persisted from
    each well's last observed month, which must lie within ``max_gap`` months of the window.
    The files follow ``twin.prospective`` so ``twin.prospective score`` scores them unchanged;
    ``model_sha256`` is the hash of the parameter file written here, since no physics run is
    involved. Registration must precede the window."""
    a = assemble(forward, stations, historical_wells, new_wells, end=None)
    dates = a["dates"]
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    today = registered_on or dt.date.today().isoformat()
    if pd.Timestamp(today) >= start:
        raise ValueError("Registration must precede the test window")
    window = (dates >= start) & (dates <= end)
    if not window.any():
        raise ValueError("The forward run's axis does not cover the test window")
    obs = np.where(np.isfinite(a["new"]), a["new"], a["historical"])
    t_fit = int((dates < start).sum())
    obs[:, t_fit:] = np.nan                       # nothing from the window, explicitly
    clim = fair_baselines(obs, t_fit, month_offset=0)["clim"]
    anom = obs - clim
    rho, n_pairs = lag1_rho(anom, t_fit)
    pooled = float(np.nanmedian(rho))
    rho_used = np.where(np.isfinite(rho), rho, pooled)
    a_stat, last = persisted_anomaly(anom, t_fit, rho_used, max_gap)
    sd = _residual_sd(obs, clim, t_fit)
    count = np.isfinite(obs[:, :t_fit]).sum(axis=1)
    keep = (count >= 24) & np.isfinite(sd) & (last >= 0)
    pred = clim + a_stat
    z = Z[level]
    out.mkdir(parents=True, exist_ok=True)
    ids = a["ids"]
    params = {"method": "stat: month-of-year climatology + rho**k * last observed anomaly",
              "fitted_through_exclusive": str(dates[t_fit].date()), "rho_pooled": pooled,
              "wells": {ids[w]: {"rho": float(rho_used[w]), "own_rho": bool(np.isfinite(rho[w])),
                                 "pairs": int(n_pairs[w]), "sd_anomaly_m": float(sd[w]),
                                 "last_observed": str(dates[last[w]].date()),
                                 "anomaly_at_origin_m": float(anom[w, last[w]]),
                                 "fitted_months": int(count[w])}
                        for w in np.flatnonzero(keep)},
              "well_axis_forward_sha256": a["model_sha256"]}
    atomic_json(out / "parameters.json", params)
    model_sha = sha256(out / "parameters.json")
    rows = []
    for w in np.flatnonzero(keep):
        for t in np.flatnonzero(window):
            if not (np.isfinite(pred[w, t]) and np.isfinite(clim[w, t])):
                continue
            rows.append({"station_id": ids[w], "layer": int(a["layers"][w]),
                         "date": str(dates[t].date()),
                         "prediction_origin": str(dates[last[w]].date()),
                         "predicted_m": float(pred[w, t]), "baseline_m": float(clim[w, t]),
                         "lower_m": float(pred[w, t] - z * sd[w]),
                         "upper_m": float(pred[w, t] + z * sd[w]), "model_sha256": model_sha})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "predictions.csv", index=False)
    retro = None
    if retrospective is not None and Path(retrospective).is_file():
        rep = json.loads(Path(retrospective).read_text())
        retro = {"report": str(retrospective), "sha256": sha256(Path(retrospective)),
                 "rmse_ratio_vs_clim_2023_26": rep["rmse_ratio_vs_clim"].get("stat"),
                 "picp_90_2023_26": rep["interval_90_coverage_with_fitted_anomaly_sd"].get("stat")}
    protocol = {
        "schema": 1, "task": "free-running-head-validation",
        "test_start": test_start, "test_end": test_end, "registered_before": today,
        "independent_holdout": True, "model_sha256": model_sha,
        "model": "twin.hybrid stat: statistical anomaly persistence on the well's climatology; "
                 "no physics enters the point forecast",
        "baseline": "month-of-year climatology of each well's observed months before the window",
        "max_rmse_ratio": 1.0, "min_stations": int(min_stations), "min_months": int(min_months),
        "coverage_bounds": [0.85, 0.95],
        "interval": f"central {level} %, per-well sd of the fitted anomaly",
        "forecast_origin": "each well's last observed month before the window (parameters.json)",
        "retrospective_basis": retro,
        "note": "Observations in the window did not exist at registration. Score with "
                "twin.prospective score once the window has closed; do not refit, re-select "
                "or re-register for this window.",
    }
    atomic_json(out / "protocol.json", protocol)
    manifest = {"registered_on": today, "stations": int(frame.station_id.nunique()),
                "rows": len(frame), "predictions_sha256": sha256(out / "predictions.csv"),
                "protocol_sha256": sha256(out / "protocol.json"),
                "parameters_sha256": model_sha, "model_sha256": model_sha,
                "origins": frame.groupby("prediction_origin").station_id.nunique().to_dict()}
    atomic_json(out / "manifest.json", manifest)
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("score", "register"):
        p = sub.add_parser(name)
        p.add_argument("--forward", required=True, type=Path)
        p.add_argument("--stations", required=True, type=Path)
        p.add_argument("--historical-wells", required=True, type=Path)
        p.add_argument("--new-wells", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        if name == "score":
            p.add_argument("--end", required=True, help="exclusive first day of a month")
            p.add_argument("--min-months", type=int, default=12)
        else:
            p.add_argument("--test-start", required=True)
            p.add_argument("--test-end", required=True)
            p.add_argument("--retrospective", type=Path, default=None,
                           help="the score report.json this registration rests on")
            p.add_argument("--min-stations", type=int, default=100)
            p.add_argument("--min-months", type=int, default=6)
    args = ap.parse_args(argv)
    if args.cmd == "score":
        rep = run(args.forward, args.stations, args.historical_wells, args.new_wells, args.out,
                  args.end, args.min_months)
        print(json.dumps({k: rep[k] for k in ("metrics", "rmse_ratio_vs_clim",
                                              "per_year_rmse_m", "rho",
                                              "interval_90_coverage_with_fitted_anomaly_sd",
                                              "in_sample_diagnostic")}, indent=2))
    else:
        res = register(args.forward, args.stations, args.historical_wells, args.new_wells,
                       args.out, args.test_start, args.test_end, retrospective=args.retrospective,
                       min_stations=args.min_stations, min_months=args.min_months)
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
