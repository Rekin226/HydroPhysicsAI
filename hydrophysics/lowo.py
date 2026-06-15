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

from .baselines import climatology_prediction
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
    ap.add_argument("--gate", type=float, default=None, metavar="TAU",
                    help="confidence-gated hybrid: trust the operator only on wells whose "
                         "free-run training-period KGE >= TAU, else fall back to that "
                         "well's climatology. TAU=0.5 was selected on the inner 2018 split.")
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
    clim_pred = climatology_prediction(data)
    # Honest per-well head-to-head against each well's OWN climatology. Median KGE alone
    # is flattering: it hides KGE's unbounded negative tail (so report a clipped mean) and
    # it is not the same as per-well superiority (so report the win-rate vs climatology).
    clim = evaluate_predictions(data, clim_pred, period="val")["kge"]

    def summary(name, p):
        per = evaluate_predictions(data, p, period="val")
        k = per["kge"]
        n = int(k.notna().sum())
        win = int((k > clim).sum())
        print(f"\n=== {name} (held-out wells, validation) ===")
        print(f"KGE  median {k.median():.3f} | mean {k.mean():.2f} "
              f"| clipped[-1,1] mean {k.clip(-1, 1).mean():.3f}")
        print(f"NSE  median {per['nse'].median():.3f} | RMSE median {per['rmse'].median():.3f} m")
        print(f"beats own climatology on {win}/{n} wells ({100 * win / max(n, 1):.0f}%) "
              f"| KGE>0.5 on {int((k > 0.5).sum())} | KGE<0 on {int((k < 0).sum())}")
        return per

    print(f"\n(climatology reference: median KGE {clim.median():.3f})")
    per = summary("LEAVE-ONE-WELL-OUT", pred)

    if args.gate is not None:
        # Trust the operator only where it reproduces a well's OWN training history;
        # otherwise fall back to climatology. The gate signal uses training days only, so
        # it leaks no validation information; TAU is selected on the inner 2018 split.
        train_kge = evaluate_predictions(data, pred, period="train")["kge"]
        trust = (train_kge >= args.gate).to_numpy()
        hybrid = clim_pred.copy()
        hybrid[trust] = pred[trust]
        print(f"\n[gate TAU={args.gate}] trusting operator on {int(trust.sum())}/"
              f"{data.n_wells} wells (rest -> climatology)")
        per = summary(f"GATED HYBRID (TAU={args.gate})", hybrid)
        pred = hybrid

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        per.round(4).to_csv(out / "per_well_lowo.csv")
        print(f"\nwrote -> {out / 'per_well_lowo.csv'}")


if __name__ == "__main__":
    main()
