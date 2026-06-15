"""Leave-one-well-out (LOWO) generalization: the operator-learning headline.

The flagship claim is amortization: one attribute-conditioned network should predict a
well it was never calibrated on. We test that by k-fold cross-well validation -- hold out
a fold of wells, train the hypernetwork on the rest, then free-run-simulate the held-out
wells from their attributes and initial condition alone (their observations never enter
the training loss). Every well is predicted exactly once while held out, and the assembled
prediction is scored on the same 2019+ simulation-mode benchmark as everything else.

This is generalization to UNSEEN wells, strictly harder than the in-domain benchmark
(where all wells are trained on). The gray-box cannot do this at all: it calibrates one
parameter set per well and has nothing to say about a well it never saw.

    python -m hydrophysics.lowo --device cuda --folds 6 --epochs 1500 --out results/ude
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import Config, default_config
from .data import GWData, load_dataset
from .eval import evaluate_predictions
from .models.ude import PhysicsUDE, enriched_features, observable_features
from .train import pick_device


def leave_one_well_out(
    data: GWData, device: str, epochs: int, folds: int = 6, seed: int = 0,
    feature_fn=None,
) -> np.ndarray:
    """Return a (W, T) prediction where each well was predicted while held out.

    Wells are assigned to folds round-robin by index (deterministic). For each fold the
    held-out wells contribute no training signal; they are simulated from their features +
    initial condition only. ``feature_fn`` selects the static-feature builder (default the
    geographic attributes; pass ``enriched_features`` to add observable history
    signatures, which is what lets the operator place an unseen well).
    """
    assign = np.arange(data.n_wells) % folds
    pred = np.full_like(data.target, np.nan)
    for f in range(folds):
        held = assign == f
        model = PhysicsUDE(device=device, epochs=epochs, seed=seed, feature_fn=feature_fn)
        model.fit(data, train_wells=~held)
        sim = model.simulate(data)
        pred[held] = sim[held]
        print(f"fold {f + 1}/{folds}: trained on {int((~held).sum())} wells, "
              f"predicted {int(held.sum())} held-out")
    return pred


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Leave-one-well-out generalization for PhysicsUDE")
    ap.add_argument("--data", default=None)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--device", default=None)
    ap.add_argument("--features", default="observable",
                    choices=["static", "observable", "enriched"],
                    help="static geographic attrs, observable history signatures "
                         "(selected on the inner split), or both")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cfg = (Config(data_dir=Path(args.data)) if args.data else default_config())
    data = load_dataset(cfg)
    print(data.summary())
    device = pick_device(args.device)
    feature_fn = {"static": None, "observable": observable_features,
                  "enriched": enriched_features}[args.features]
    print(f"device: {device} | LOWO {args.folds}-fold | epochs: {args.epochs} "
          f"| features: {args.features}")

    pred = leave_one_well_out(data, device, args.epochs, folds=args.folds,
                              feature_fn=feature_fn)
    per = evaluate_predictions(data, pred, period="val")
    print("\n=== LEAVE-ONE-WELL-OUT (held-out wells, validation) ===")
    print(f"KGE  median {per['kge'].median():.3f} | mean {per['kge'].mean():.3f}")
    print(f"NSE  median {per['nse'].median():.3f}")
    print(f"RMSE median {per['rmse'].median():.3f} m")
    print(f"wells: {int(per['kge'].notna().sum())} | KGE>0.5: {int((per['kge'] > 0.5).sum())}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        per.round(4).to_csv(out / "per_well_lowo.csv")
        print(f"\nwrote -> {out / 'per_well_lowo.csv'}")


if __name__ == "__main__":
    main()
