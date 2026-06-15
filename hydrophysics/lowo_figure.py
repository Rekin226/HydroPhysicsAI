"""Plot leave-one-well-out per-well KGE for climatology vs the operator vs the
confidence-gated hybrid. Reads the consolidated per-well CSV written by the hybrid run
(columns: clim, operator, hybrid).

    python -m hydrophysics.lowo_figure \
        --csv results/ude/lowo_hybrid_per_well.csv \
        --out results/figures/lowo_improvement.png
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="LOWO comparison figure")
    ap.add_argument("--csv", default="results/ude/lowo_hybrid_per_well.csv")
    ap.add_argument("--out", default="results/figures/lowo_improvement.png")
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv, index_col=0)
    cols = [("clim", "climatology\n(baseline)", "0.6"),
            ("operator", "operator only", "#5b9bd5"),
            ("hybrid", "gated hybrid", "#76b900")]
    data = [np.clip(df[c].dropna().to_numpy(), -1, 1) for c, _, _ in cols]
    clim_med = float(np.median(df["clim"].dropna()))

    fig, ax = plt.subplots(figsize=(9, 4.8))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc, (_, _, color) in zip(parts["bodies"], cols):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("k")
    rng = np.random.default_rng(0)
    for x, vals in enumerate(data, start=1):
        ax.scatter(x + rng.uniform(-0.06, 0.06, vals.size), vals, s=11,
                   color="k", alpha=0.35, zorder=3)
    ax.axhline(clim_med, color="tab:red", ls="--", lw=1.1,
               label=f"climatology median {clim_med:.2f}")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{lbl}\nmedian {np.median(d):.2f}"
                        for (_, lbl, _), d in zip(cols, data)])
    ax.set_ylabel("held-out well KGE (clipped to [-1, 1])")
    ax.set_title("Leave-one-well-out: gating the operator to climatology beats both")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
