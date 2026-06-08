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


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Forecast-mode evaluation for GlobalForecastLSTM")
    ap.add_argument("--data", default=None)
    ap.add_argument("--horizons", default="1,7,30", help="comma-separated forecast horizons (days)")
    ap.add_argument("--lookback", type=int, default=90)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    horizons = [int(x) for x in args.horizons.split(",")]
    cfg = Config(data_dir=Path(args.data)) if args.data else default_config()
    data = load_dataset(cfg)
    print(data.summary())
    device = pick_device(args.device)
    print(f"device: {device} | forecast horizons {horizons} | lookback {args.lookback}")

    from .models.forecast_lstm import GlobalForecastLSTM
    model = GlobalForecastLSTM(lookback=args.lookback, horizon=max(horizons),
                              epochs=args.epochs, device=device)
    model.fit(data)
    cube = model.forecast(data)
    table = horizon_table(data, cube, horizons, period="val")

    pd.set_option("display.width", 120)
    print("\n=== FORECAST-mode skill (validation), by horizon ===")
    print(table.round(3).to_string())

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        table.round(4).to_csv(out / "forecast_benchmark.csv")
        print(f"\nwrote -> {out / 'forecast_benchmark.csv'}")


if __name__ == "__main__":
    main()
