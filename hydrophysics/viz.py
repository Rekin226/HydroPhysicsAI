"""Figures for the README: simulation hydrographs, probabilistic forecast fans, and
benchmark bars. matplotlib is a lazy import (optional ``[viz]`` extra) so the foundation
stays dependency-light.

These work on any :class:`GWData`. The committed figures are generated from the bundled
*synthetic* sample (see ``hydrophysics.figures``) so they are fully reproducible and ship
no real agency time series -- only aggregate skill maps are derived from real data.
"""

from __future__ import annotations

import numpy as np

from .data import GWData
from .metrics import kge


def _val_span(data: GWData):
    """First/last date index of the validation period, for shading."""
    idx = np.where(data.val_mask)[0]
    return (idx[0], idx[-1]) if idx.size else (None, None)


def plot_simulation_hydrographs(data: GWData, preds: dict, well_ids, out_path,
                                title: str = "Free-running simulation vs observed"):
    """Observed vs each model's free-running simulation for selected wells.

    preds: {label: (W, T) array}. One stacked subplot per well; the validation period
    (scored, never seen during fit) is shaded. Returns the saved path.
    """
    import matplotlib.pyplot as plt

    n = len(well_ids)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), sharex=True, squeeze=False)
    v0, v1 = _val_span(data)
    for ax, wid in zip(axes[:, 0], well_ids):
        i = data.well_index(wid)
        ax.plot(data.dates, data.target[i], color="0.25", lw=1.1, label="observed")
        for label, pred in preds.items():
            k = kge(data.target[i, data.val_mask], pred[i, data.val_mask])
            ax.plot(data.dates, pred[i], lw=1.1, alpha=0.9, label=f"{label} (val KGE {k:.2f})")
        if v0 is not None:
            ax.axvspan(data.dates[v0], data.dates[v1], color="orange", alpha=0.08)
        ax.set_title(f"Well {wid}", fontsize=10, loc="left")
        ax.set_ylabel("level (m)")
        ax.legend(fontsize=8, loc="best", framealpha=0.85)
    axes[-1, 0].set_xlabel("date")
    fig.suptitle(title, fontsize=12, y=0.997)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_forecast_fan(data: GWData, mean_cube, sigma_cube, well_id, t0_index,
                      out_path, z: float = 1.6449,
                      title: str = "Probabilistic forecast (90% interval)"):
    """Fan chart: the H-step forecast trajectory from one origin with its 90% band.

    mean_cube / sigma_cube: (W, T, H) from GlobalForecastLSTM.forecast_dist.
    t0_index: the forecast origin (day index). Shows recent history, the predicted
    mean, the z*sigma band, and the realized observations. Returns the saved path.
    """
    import matplotlib.pyplot as plt

    i = data.well_index(well_id)
    H = mean_cube.shape[2]
    mean = mean_cube[i, t0_index]
    sigma = sigma_cube[i, t0_index]
    hx = np.arange(1, H + 1)
    fut_dates = [data.dates[t0_index + h] for h in hx if t0_index + h < data.n_days]
    m = len(fut_dates)
    mean, sigma, hx = mean[:m], sigma[:m], hx[:m]

    look = 90
    h0 = max(0, t0_index - look)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(data.dates[h0:t0_index + 1], data.target[i, h0:t0_index + 1],
            color="0.25", lw=1.2, label="observed (history)")
    obs_future = data.target[i, t0_index + 1:t0_index + 1 + m]
    ax.plot(fut_dates, obs_future, "o", color="0.1", ms=3, label="observed (realized)")
    ax.plot(fut_dates, mean, color="tab:blue", lw=1.6, label="forecast mean")
    ax.fill_between(fut_dates, mean - z * sigma, mean + z * sigma,
                    color="tab:blue", alpha=0.20, label="90% interval")
    ax.axvline(data.dates[t0_index], color="orange", ls="--", lw=1, alpha=0.8)
    ax.set_title(f"{title} - well {well_id}, {m}-day horizon", fontsize=11)
    ax.set_xlabel("date")
    ax.set_ylabel("level (m)")
    ax.legend(fontsize=9, loc="best", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_benchmark_bars(scores: dict, out_path, ylabel="median KGE (higher is better)",
                        title="Simulation benchmark", reference: float | None = None):
    """Horizontal bar chart of a scalar score per model.

    scores: {model_label: value}. ``reference`` draws a dashed line (e.g. the gray-box).
    """
    import matplotlib.pyplot as plt

    labels = list(scores)
    vals = [scores[k] for k in labels]
    colors = ["#76b900" if v == max(vals) else "0.6" for v in vals]
    fig, ax = plt.subplots(figsize=(8, 0.7 * len(labels) + 1.5))
    ax.barh(labels, vals, color=colors, edgecolor="k", linewidth=0.5)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=9)
    if reference is not None:
        ax.axvline(reference, color="tab:red", ls="--", lw=1.2,
                   label=f"gray-box {reference:.3f}")
        ax.legend(fontsize=9)
    ax.set_xlabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
