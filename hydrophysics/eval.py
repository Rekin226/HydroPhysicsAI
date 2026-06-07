"""Evaluation harness: score predictions per well and build the benchmark table.

The benchmark table is the headline artifact of the flagship. Every model (and the
persistence and gray-box baselines) is scored on the SAME validation split, and we
report the per-well distribution, not just the mean, because a few hard wells can
dominate the mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GWData
from .metrics import all_metrics


def evaluate_predictions(
    data: GWData, pred: np.ndarray, period: str = "val"
) -> pd.DataFrame:
    """Per-well KGE / NSE / RMSE for ``pred`` over the chosen period.

    period: "val" (default), "train", or "all".
    Returns a DataFrame indexed by well id.
    """
    if pred.shape != data.target.shape:
        raise ValueError(f"pred shape {pred.shape} != target {data.target.shape}")
    mask = {"val": data.val_mask, "train": data.train_mask,
            "all": np.ones(data.n_days, dtype=bool)}[period]

    rows = {}
    for i, wid in enumerate(data.well_ids):
        rows[wid] = all_metrics(data.target[i, mask], pred[i, mask])
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "st_id"
    return out


def _aggregate(per_well: pd.DataFrame) -> dict[str, float]:
    return {
        "kge_median": per_well["kge"].median(),
        "kge_mean": per_well["kge"].mean(),
        "rmse_median": per_well["rmse"].median(),
        "nse_median": per_well["nse"].median(),
        "n_wells": int(per_well["kge"].notna().sum()),
    }


def benchmark_table(
    data: GWData,
    predictions: dict[str, np.ndarray],
    graybox: pd.DataFrame | None = None,
    period: str = "val",
) -> pd.DataFrame:
    """Compare models against the baselines on one table.

    predictions: {model_name: (W, T) array}. Use this for persistence and the neural
        models (anything we can run forward to get a prediction series).
    graybox: optional gray-box per-well scores (from load_graybox_baseline) whose
        validation scores are reported as-is (no prediction series needed).
    """
    records = []
    for name, pred in predictions.items():
        agg = _aggregate(evaluate_predictions(data, pred, period=period))
        records.append({"model": name, **agg})

    if graybox is not None:
        gb = graybox.reindex([str(w) for w in data.well_ids])
        records.append({
            "model": "graybox_ode",
            "kge_median": gb["kge"].median(),
            "kge_mean": gb["kge"].mean(),
            "rmse_median": gb["rmse"].median() if "rmse" in gb else float("nan"),
            "nse_median": float("nan"),
            "n_wells": int(gb["kge"].notna().sum()),
        })

    table = pd.DataFrame.from_records(records).set_index("model")
    return table.sort_values("kge_median", ascending=False)


def spatial_scores(
    data: GWData, per_well: pd.DataFrame, metric: str = "kge"
) -> pd.DataFrame:
    """Join a per-well metric to well coordinates for spatial plotting.

    Returns columns [tm_x, tm_y, <metric>, is_coastal]. Plotting itself is left to the
    caller (matplotlib/geopandas are optional); see ``plot_spatial`` for a default.
    """
    df = per_well[[metric]].copy()
    attrs = data.attrs
    df["tm_x"] = attrs["tm_x"].reindex(df.index).to_numpy()
    df["tm_y"] = attrs["tm_y"].reindex(df.index).to_numpy()
    df["is_coastal"] = attrs["is_coastal"].reindex(df.index).to_numpy()
    return df


def plot_spatial(spatial_df: pd.DataFrame, metric: str = "kge", ax=None, shapefile=None):
    """Scatter wells colored by score over the alluvial fan. Optional, lazy imports.

    Pass ``shapefile`` (a path to the Zhuoshui fan .shp) to draw the fan outline if
    geopandas is installed. Returns the matplotlib Axes.
    """
    import matplotlib.pyplot as plt  # lazy: plotting is optional

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 8))

    if shapefile is not None:
        try:
            import geopandas as gpd  # optional
            gpd.read_file(shapefile).boundary.plot(ax=ax, color="0.7", linewidth=0.8)
        except Exception:
            pass  # fan outline is decorative; skip if geopandas/shapefile unavailable

    sc = ax.scatter(
        spatial_df["tm_x"], spatial_df["tm_y"], c=spatial_df[metric],
        cmap="viridis", s=60, edgecolor="k", linewidth=0.4, vmin=0, vmax=1,
    )
    ax.set_title(f"Per-well {metric.upper()} (validation)")
    ax.set_xlabel("TM_X97 (m)")
    ax.set_ylabel("TM_Y97 (m)")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, label=metric.upper(), shrink=0.7)
    return ax
