"""Forecast-mode evaluation: horizon-wise skill vs persistence and climatology.

Forecast mode (predict day t0+h using info up to t0) is kept strictly separate from the
simulation benchmark -- comparing the two is the classic groundwater evaluation trap,
since daily levels are highly autocorrelated and short-horizon forecasting looks
trivially good. Here every model is scored against the honest forecast-mode references at
the SAME horizon: persistence (tomorrow = today) and per-well climatology.

    python -m hydrophysics.forecast_eval --device cuda --horizons 1,7,30 --out results/forecast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import climatology_prediction
from .config import Config, default_config
from .data import GWData, load_dataset
from .metrics import all_metrics
from .train import pick_device


def _score_series(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return all_metrics(obs, pred)


def _aggregate(rows: dict[str, dict]) -> dict[str, float]:
    df = pd.DataFrame.from_dict(rows, orient="index")
    return {
        "kge_median": df["kge"].median(),
        "kge_mean": df["kge"].mean(),
        "nse_median": df["nse"].median(),
        "rmse_median": df["rmse"].median(),
        "n_wells": int(df["kge"].notna().sum()),
    }


def horizon_table(
    data: GWData, pred_cube: np.ndarray, horizons: list[int], period: str = "val"
) -> pd.DataFrame:
    """Score the forecaster and the persistence/climatology baselines at each horizon.

    pred_cube: (W, T, H) absolute-level forecasts, [i, t0, h-1] = level at t0+h.
    Target days tau are the chosen period's days; origin is t0 = tau - h.
    """
    level = data.target
    T = data.n_days
    mask = {"val": data.val_mask, "train": data.train_mask,
            "all": np.ones(T, dtype=bool)}[period]
    clim = climatology_prediction(data)             # (W, T) seasonal mean per day
    records = []
    for h in horizons:
        taus = np.where(mask)[0]
        taus = taus[(taus - h >= 0) & (taus < T)]
        m_rows, p_rows, c_rows = {}, {}, {}
        for i, wid in enumerate(data.well_ids):
            tau = taus
            t0 = tau - h
            obs = level[i, tau]
            fc = pred_cube[i, t0, h - 1]
            persist = level[i, t0]
            climf = clim[i, tau]
            ok = np.isfinite(obs)
            m_rows[wid] = _score_series(obs[ok], fc[ok])
            p_rows[wid] = _score_series(obs[ok], persist[ok])
            c_rows[wid] = _score_series(obs[ok], climf[ok])
        records.append({"horizon_d": h, "model": "forecast_lstm", **_aggregate(m_rows)})
        records.append({"horizon_d": h, "model": "persistence", **_aggregate(p_rows)})
        records.append({"horizon_d": h, "model": "climatology", **_aggregate(c_rows)})
    return pd.DataFrame.from_records(records).set_index(["horizon_d", "model"])


def _crps_gaussian(obs: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Gaussian predictive distribution (per point). Lower better."""
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-6)
    z = (obs - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def _persistence_sigma(data: GWData, horizons: list[int]) -> dict[int, np.ndarray]:
    """Per-well std of the h-step increment over training -- the persistence forecast's
    honest Gaussian spread for N(y[t0], sigma^2)."""
    out = {}
    for h in horizons:
        sd = np.ones(data.n_wells, dtype=float)
        for i in range(data.n_wells):
            y = data.target[i]
            d = y[h:] - y[:-h]
            d = d[np.isfinite(d) & data.train_mask[h:]]
            if d.size:
                sd[i] = max(d.std(), 1e-6)
        out[h] = sd
    return out


def probabilistic_horizon_table(
    data: GWData, mean_cube: np.ndarray, sigma_cube: np.ndarray,
    horizons: list[int], period: str = "val", z: float = 1.6449,
) -> pd.DataFrame:
    """Probabilistic skill per horizon: CRPS, 90% interval coverage (PICP), sharpness.

    Reported for the LSTM and a persistence-Gaussian baseline. PICP near 0.90 is well
    calibrated; sharpness is the mean 90% interval width (narrower is better at equal
    coverage). KGE/RMSE on the mean are kept for continuity with the point forecast.
    """
    level = data.target
    T = data.n_days
    mask = {"val": data.val_mask, "train": data.train_mask,
            "all": np.ones(T, dtype=bool)}[period]
    p_sigma = _persistence_sigma(data, horizons)
    records = []
    for h in horizons:
        taus = np.where(mask)[0]
        taus = taus[(taus - h >= 0) & (taus < T)]
        t0 = taus - h
        m_crps, m_cov, m_sharp, p_crps, p_cov = [], [], [], [], []
        m_kge, m_rmse = {}, {}
        for i, wid in enumerate(data.well_ids):
            obs = level[i, taus]
            mu = mean_cube[i, t0, h - 1]
            sig = sigma_cube[i, t0, h - 1]
            ok = np.isfinite(obs) & np.isfinite(mu) & np.isfinite(sig)
            o, mu, sig = obs[ok], mu[ok], sig[ok]
            if o.size == 0:
                continue
            m_crps.append(_crps_gaussian(o, mu, sig).mean())
            m_cov.append(np.mean(np.abs(o - mu) <= z * sig))
            m_sharp.append(np.mean(2 * z * sig))
            m_kge[wid] = all_metrics(o, mu)["kge"]
            m_rmse[wid] = all_metrics(o, mu)["rmse"]
            # persistence-Gaussian baseline
            ps = p_sigma[h][i]
            pmu = level[i, t0][ok]
            p_crps.append(_crps_gaussian(o, pmu, np.full_like(o, ps)).mean())
            p_cov.append(np.mean(np.abs(o - pmu) <= z * ps))
        records.append({"horizon_d": h, "model": "forecast_lstm",
                        "crps_median": float(np.median(m_crps)),
                        "picp90": float(np.mean(m_cov)),
                        "sharpness90": float(np.median(m_sharp)),
                        "kge_median": float(np.median(list(m_kge.values()))),
                        "rmse_median": float(np.median(list(m_rmse.values()))),
                        "n_wells": len(m_crps)})
        records.append({"horizon_d": h, "model": "persistence_gauss",
                        "crps_median": float(np.median(p_crps)),
                        "picp90": float(np.mean(p_cov)),
                        "sharpness90": float("nan"), "kge_median": float("nan"),
                        "rmse_median": float("nan"), "n_wells": len(p_crps)})
    return pd.DataFrame.from_records(records).set_index(["horizon_d", "model"])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Forecast-mode evaluation for GlobalForecastLSTM")
    ap.add_argument("--data", default=None)
    ap.add_argument("--horizons", default="1,7,30", help="comma-separated forecast horizons (days)")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--probabilistic", action="store_true",
                    help="Gaussian forecasts; report CRPS / 90%% coverage / sharpness")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    horizons = [int(x) for x in args.horizons.split(",")]
    cfg = Config(data_dir=Path(args.data)) if args.data else default_config()
    data = load_dataset(cfg)
    print(data.summary())
    device = pick_device(args.device)
    mode = "probabilistic" if args.probabilistic else "point"
    print(f"device: {device} | {mode} forecast | horizons {horizons} | lookback {args.lookback}")

    from .models.forecast_lstm import GlobalForecastLSTM
    model = GlobalForecastLSTM(lookback=args.lookback, horizon=max(horizons),
                              epochs=args.epochs, probabilistic=args.probabilistic, device=device)
    model.fit(data)
    pd.set_option("display.width", 120)

    if args.probabilistic:
        mean, sigma = model.forecast_dist(data)
        table = probabilistic_horizon_table(data, mean, sigma, horizons, period="val")
        print("\n=== PROBABILISTIC forecast skill (validation), by horizon ===")
        print(table.round(3).to_string())
        out_name = "forecast_probabilistic.csv"
    else:
        cube = model.forecast(data)
        table = horizon_table(data, cube, horizons, period="val")
        print("\n=== FORECAST-mode skill (validation), by horizon ===")
        print(table.round(3).to_string())
        out_name = "forecast_benchmark.csv"

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(out / out_name)
        print(f"\nwrote -> {out / out_name}")


if __name__ == "__main__":
    main()
