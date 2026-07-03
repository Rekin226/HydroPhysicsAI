"""Freeze the baselines and print the Phase-0 benchmark tables.

Usage:
    # your own data (laid out like hydrophysics/sample_data/; see docs/DATA_FORMAT.md)
    export HYDROMIND_GW_DATA=/path/to/your/gw_data
    python -m hydrophysics.run_baselines \\
        --baseline /path/to/your/gray_box_fit_results.csv \\
        --out results/phase0

    # synthetic sample (no real data needed)
    python -m hydrophysics.run_baselines

Writes (when --out is given): benchmark_simulation.csv, benchmark_forecast.csv,
per_well_<baseline>.csv, and a spatial KGE map if matplotlib is available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .baselines import (
    climatology_prediction,
    last_value_prediction,
    load_graybox_baseline,
    persistence_prediction,
)
from .config import Config, default_config
from .data import load_dataset
from .eval import benchmark_table, evaluate_predictions, spatial_scores


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase-0 groundwater baselines")
    ap.add_argument("--data", default=None, help="data dir (else HYDROMIND_GW_DATA / sample)")
    ap.add_argument("--baseline", default=None, help="gray-box gw_fit_results.csv")
    ap.add_argument("--out", default=None, help="output dir for tables/figures")
    args = ap.parse_args(argv)

    if args.data or args.baseline:
        cfg = Config(
            data_dir=Path(args.data) if args.data else default_config().data_dir,
            baseline_results=Path(args.baseline) if args.baseline else None,
        )
    else:
        cfg = default_config()

    data = load_dataset(cfg)
    print(data.summary())

    sim_preds = {
        "climatology": climatology_prediction(data),
        "last_value": last_value_prediction(data),
    }
    fc_preds = {"persistence": persistence_prediction(data)}

    graybox = None
    try:
        graybox = load_graybox_baseline(cfg)
    except FileNotFoundError:
        print("\n(no gray-box baseline configured; skipping it)")

    pd.set_option("display.width", 120)

    sim_table = benchmark_table(data, sim_preds, graybox=graybox, period="val")
    print("\n=== SIMULATION mode (free-running; comparable to gray-box & neural operator) ===")
    print(sim_table.round(3).to_string())

    fc_table = benchmark_table(data, fc_preds, graybox=None, period="val")
    print("\n=== FORECAST mode (1-step-ahead; context only, NOT comparable to above) ===")
    print(fc_table.round(3).to_string())

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        sim_table.round(4).to_csv(out / "benchmark_simulation.csv")
        fc_table.round(4).to_csv(out / "benchmark_forecast.csv")
        for name, pred in {**sim_preds, **fc_preds}.items():
            evaluate_predictions(data, pred, period="val").round(4).to_csv(
                out / f"per_well_{name}.csv"
            )
        if graybox is not None:
            graybox.to_csv(out / "per_well_graybox.csv")
            try:
                import matplotlib.pyplot as plt  # optional

                from .eval import plot_spatial
                gb = graybox.reindex(data.well_ids)
                sp = spatial_scores(data, gb.rename(columns={"kge": "kge"}), metric="kge")
                ax = plot_spatial(sp, metric="kge",
                                  shapefile=None)
                ax.figure.savefig(out / "spatial_kge_graybox.png", dpi=150,
                                  bbox_inches="tight")
                plt.close(ax.figure)
                print(f"\nwrote spatial map -> {out / 'spatial_kge_graybox.png'}")
            except Exception as e:  # pragma: no cover - plotting is optional
                print(f"(spatial map skipped: {e})")
        print(f"\nwrote tables -> {out}")


if __name__ == "__main__":
    main()
