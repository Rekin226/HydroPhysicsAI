"""Inner-split selection for the PhysicsUDE forcing hyperparameters.

inner-train = dates < 2018-01-01, inner-val = 2018 (still inside the pre-2019 training
period -- this NEVER touches the 2019+ benchmark). This operationalizes the repeated
"selected on the inner split" claim for the two forcing choices the README credits with
lifting simulation KGE (0.591 -> 0.704 -> 0.754):

  * the multi-timescale recharge-memory decays ``rain_memory`` (vs the instantaneous
    ``b * rain`` term), and
  * the ET coefficient in ``net recharge = rain - coef * ET0``.

It mirrors ``results/decomp/inner_split.py``. Like that script it is only meaningful on
the REAL data: the bundled synthetic sample's 2018 inner-val is too thin (and its
coordinates/dates don't match the committed ET0 cache), so on the sample use it just to
confirm the selection runs, not for the numbers.

    python -m results.ude.inner_select --device cuda --epochs 1500 \
        --et --et-cache results/et/openmeteo_et0_2012_2022.npz
"""

from __future__ import annotations

import argparse
from copy import copy

import numpy as np
import pandas as pd

from hydrophysics.baselines import climatology_prediction
from hydrophysics.config import default_config
from hydrophysics.data import GWData, load_dataset
from hydrophysics.eval import evaluate_predictions
from hydrophysics.models.ude import PhysicsUDE
from hydrophysics.train import pick_device


def make_inner(data: GWData) -> GWData:
    """inner-train = dates < 2018-01-01, inner-val = 2018 (< the 2019 outer split)."""
    dates = data.dates
    inner_split = pd.Timestamp("2018-01-01")
    outer_split = pd.Timestamp(data.split_date)
    d2 = copy(data)
    d2.train_mask = np.asarray(dates < inner_split)
    d2.val_mask = np.asarray((dates >= inner_split) & (dates < outer_split))
    return d2


def score(data: GWData, pred: np.ndarray, label: str, clim_kge: pd.Series) -> float:
    kge = evaluate_predictions(data, pred, "val")["kge"]
    winrate = float((kge.values > clim_kge.values).mean())
    print(f"{label:34s} median={kge.median():.4f}  "
          f"clipped-mean={kge.clip(lower=-1).mean():.4f}  win-vs-clim={winrate:.2%}")
    return float(kge.median())


def _select(name: str, candidates: dict, fit_fn, inner: GWData, clim_kge: pd.Series) -> str:
    print(f"\n=== selecting {name} (inner-val = 2018) ===")
    scored = {lab: score(inner, fit_fn(val), lab, clim_kge) for lab, val in candidates.items()}
    best = max(scored, key=scored.get)
    print(f"BEST {name}: {best} = {scored[best]:.4f}")
    return best


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Inner-split selection for PhysicsUDE forcing")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--anchor-equilibrium", action="store_true")
    ap.add_argument("--et", action="store_true", help="also select the ET coefficient")
    ap.add_argument("--et-cache", default="results/et/openmeteo_et0_2012_2022.npz")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    data = load_dataset(default_config())
    inner = make_inner(data)
    print(f"INNER split: train days {int(inner.train_mask.sum())} "
          f"| val(2018) days {int(inner.val_mask.sum())} | device {device}")
    clim_kge = evaluate_predictions(inner, climatology_prediction(inner), "val")["kge"]
    score(inner, climatology_prediction(inner), "climatology (ref)", clim_kge)

    def fit_rm(rm):
        m = PhysicsUDE(rain_memory=rm, device=device, epochs=args.epochs,
                       anchor_equilibrium=args.anchor_equilibrium).fit(inner)
        return m.simulate(inner)

    rm_candidates = {
        "no-memory (b*rain)": (),
        "rain2 (.99,.9)": (0.99, 0.9),
        "rain3 (.99,.95,.85)": (0.99, 0.95, 0.85),
        "rain4 (.99,.95,.85,.5)": (0.99, 0.95, 0.85, 0.5),
    }
    best_rm = _select("rain_memory", rm_candidates, fit_rm, inner, clim_kge)
    rm = rm_candidates[best_rm]

    if args.et:
        from hydrophysics.et import et0_for_data, net_recharge
        et0 = et0_for_data(inner, cache_path=args.et_cache)

        def fit_coef(coef):
            nd = net_recharge(inner, et0=et0, coef=coef)
            m = PhysicsUDE(rain_memory=rm, device=device, epochs=args.epochs,
                           anchor_equilibrium=args.anchor_equilibrium).fit(nd)
            return m.simulate(nd)

        _select("ET coef", {"0.0": 0.0, "0.5": 0.5, "1.0": 1.0}, fit_coef, inner, clim_kge)


if __name__ == "__main__":
    main()
