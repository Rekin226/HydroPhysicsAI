"""Plot the leave-one-well-out improvement: per-well KGE distribution, static vs the
observable-signature features. Reads the per-well CSVs written by the LOWO run.

    python -m hydrophysics.lowo_figure \
        --static results/ude/per_well_lowo_static.csv \
        --observable results/ude/per_well_lowo_observable.csv \
        --out results/figures/lowo_improvement.png
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LOWO improvement figure")
    ap.add_argument("--static", default="results/ude/per_well_lowo_static.csv")
    ap.add_argument("--observable", default="results/ude/per_well_lowo_observable.csv")
    ap.add_argument("--climatology", type=float, default=0.446,
                    help="climatology median KGE reference line")
    ap.add_argument("--out", default="results/figures/lowo_improvement.png")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = pd.read_csv(args.static)["kge"].dropna().to_numpy()
    o = pd.read_csv(args.observable)["kge"].dropna().to_numpy()
    s = np.clip(s, -1, 1)
    o = np.clip(o, -1, 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.violinplot([s, o], showmedians=True, showextrema=False)
    for pc, c in zip(parts["bodies"], ["0.6", "#76b900"]):
        pc.set_facecolor(c)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("k")
    # jittered points
    rng = np.random.default_rng(0)
    for x, vals in [(1, s), (2, o)]:
        ax.scatter(x + rng.uniform(-0.06, 0.06, vals.size), vals, s=12,
                   color="k", alpha=0.4, zorder=3)
    ax.axhline(args.climatology, color="tab:red", ls="--", lw=1.2,
               label=f"climatology median {args.climatology:.2f}")
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"static attrs\nmedian {np.median(s):.2f}",
                        f"+ observable signatures\nmedian {np.median(o):.2f}"])
    ax.set_ylabel("held-out well KGE (clipped to [-1, 1])")
    ax.set_title("Leave-one-well-out generalization: observed-history signatures")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
