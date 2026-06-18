"""Inner-split evaluation: inner-train < 2018, inner-val = 2018. Used to choose variants.
Never touches the 2019+ benchmark."""
import numpy as np
import pandas as pd
from copy import copy

from hydrophysics.config import default_config
from hydrophysics.data import load_dataset, GWData
from hydrophysics.eval import evaluate_predictions
from hydrophysics.baselines import climatology_prediction
from hydrophysics.decomp import DecompModel, DecompConfig


def make_inner(data: GWData) -> GWData:
    """Carve an inner split from the pre-2019 training period.
    inner-train = dates < 2018-01-01, inner-val = 2018 (still < 2019 outer split).
    """
    dates = data.dates
    inner_split = pd.Timestamp("2018-01-01")
    outer_split = pd.Timestamp(data.split_date)
    train_mask = np.asarray(dates < inner_split)
    val_mask = np.asarray((dates >= inner_split) & (dates < outer_split))
    d2 = copy(data)
    d2.train_mask = train_mask
    d2.val_mask = val_mask
    return d2


def score(data, pred, label):
    pw = evaluate_predictions(data, pred, "val")
    kge = pw["kge"]
    clim = climatology_prediction(data)
    cpw = evaluate_predictions(data, clim, "val")
    winrate = float((kge.values > cpw["kge"].values).mean())
    print(f"{label:42s} median={kge.median():.4f}  clipped-mean={kge.clip(lower=-1).mean():.4f}  "
          f"win-vs-clim={winrate:.2%}  n={kge.notna().sum()}")
    return kge.median()


if __name__ == "__main__":
    data = load_dataset(default_config())
    inner = make_inner(data)
    print("INNER split: train days", int(inner.train_mask.sum()), "val(2018) days", int(inner.val_mask.sum()))

    # climatology reference on inner split
    clim = climatology_prediction(inner)
    score(inner, clim, "climatology (ref)")

    variants = {
        "H3 notrend": DecompConfig(n_harmonics=3, use_trend=False),
        "H3 notrend rain3": DecompConfig(n_harmonics=3, use_trend=False, rain_decays=(0.98, 0.9, 0.7)),
        "H3 notrend rain3 lag5": DecompConfig(n_harmonics=3, use_trend=False, rain_decays=(0.98, 0.9, 0.7), ups_lag=5),
        "H3 +trend(linear)": DecompConfig(n_harmonics=3, use_trend=True),
        "H3 +trend damp.999": DecompConfig(n_harmonics=3, use_trend=True, trend_damping=0.999),
        "H3 +trend damp.997": DecompConfig(n_harmonics=3, use_trend=True, trend_damping=0.997),
        "H3 +trend damp.995": DecompConfig(n_harmonics=3, use_trend=True, trend_damping=0.995),
        "H3 +trend damp.99": DecompConfig(n_harmonics=3, use_trend=True, trend_damping=0.99),
        "H3 +trend rain3 damp.997": DecompConfig(n_harmonics=3, use_trend=True, trend_damping=0.997, rain_decays=(0.98, 0.9, 0.7), ups_lag=5),
        "H4 +trend damp.997 rain3 lag5": DecompConfig(n_harmonics=4, use_trend=True, trend_damping=0.997, rain_decays=(0.98, 0.9, 0.7), ups_lag=5),
    }
    results = {}
    for label, cfg in variants.items():
        m = DecompModel(cfg).fit(inner)
        pred = m.simulate(inner)
        results[label] = score(inner, pred, label)
    best = max(results, key=results.get)
    print("\nBEST on inner:", best, "=", round(results[best], 4))
