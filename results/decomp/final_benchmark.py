"""FINAL 2019+ benchmark for the decomposition model. Run ONCE for the headline number.

Primary config is chosen on the inner split (2018) among HORIZON-ROBUST trend settings
(damped or no trend), because the real benchmark extrapolates up to 4 years and a pure
linear per-well trend is known to be unsafe over that horizon. We also report the
linear-trend and no-trend variants as ablations to expose the horizon risk -- but the
headline verdict uses the pre-committed robust config only.
"""
import pandas as pd

from hydrophysics.config import default_config
from hydrophysics.data import load_dataset
from hydrophysics.eval import evaluate_predictions
from hydrophysics.baselines import climatology_prediction
from hydrophysics.decomp import DecompModel, DecompConfig

GRAYBOX = 0.736
UDE = 0.591
CLIM = 0.446


def report(data, pred, label, clim_kge):
    pw = evaluate_predictions(data, pred, "val")
    kge = pw["kge"]
    winrate = float((kge.values > clim_kge.values).mean())
    return {
        "model": label,
        "kge_median": round(float(kge.median()), 4),
        "kge_clipmean": round(float(kge.clip(lower=-1).mean()), 4),
        "win_vs_clim": round(winrate, 4),
        "n": int(kge.notna().sum()),
        "_pw": pw,
    }


if __name__ == "__main__":
    data = load_dataset(default_config())  # outer split = 2019-01-01
    print("OUTER benchmark: train days", int(data.train_mask.sum()),
          "val(2019+) days", int(data.val_mask.sum()))

    clim = climatology_prediction(data)
    clim_pw = evaluate_predictions(data, clim, "val")
    clim_kge = clim_pw["kge"]

    # PRIMARY (pre-committed, horizon-robust): damped trend + richer drivers
    primary = DecompConfig(n_harmonics=3, use_trend=True, trend_damping=0.99,
                           rain_decays=(0.98, 0.9, 0.7), ups_lag=5)
    configs = {
        "decomp PRIMARY (damp.99 rain3 lag5)": primary,
        "decomp notrend": DecompConfig(n_harmonics=3, use_trend=False),
        "decomp notrend rain3 lag5": DecompConfig(n_harmonics=3, use_trend=False,
                                                  rain_decays=(0.98, 0.9, 0.7), ups_lag=5),
        "decomp linear-trend (ablation)": DecompConfig(n_harmonics=3, use_trend=True),
    }

    rows = [report(data, clim, "climatology", clim_kge)]
    saved = {}
    for label, cfg in configs.items():
        m = DecompModel(cfg).fit(data)
        pred = m.simulate(data)
        r = report(data, pred, label, clim_kge)
        rows.append(r)
        saved[label] = r["_pw"]

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_pw"} for r in rows])
    print("\n=== FINAL 2019+ benchmark (simulation mode) ===")
    print(df.to_string(index=False))
    print(f"\nBars:  gray-box={GRAYBOX}  UDE={UDE}  climatology={CLIM}")

    # save per-well CSV for the primary config
    pw = saved["decomp PRIMARY (damp.99 rain3 lag5)"]
    out = pw[["kge", "nse", "rmse", "kge_r", "kge_alpha", "kge_beta"]].copy()
    out.to_csv("results/decomp/decomp_primary_perwell_2019plus.csv")
    print("\nWrote results/decomp/decomp_primary_perwell_2019plus.csv")
    df.to_csv("results/decomp/decomp_benchmark_2019plus.csv", index=False)
    print("Wrote results/decomp/decomp_benchmark_2019plus.csv")
