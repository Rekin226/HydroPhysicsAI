"""HydroPhysicsAI live demo (Gradio).

Pick a well and see two things the repo does:
  1. a free-running PhysicsUDE simulation hindcast vs observed (simulation mode), and
  2. a probabilistic 30-day forecast with its 90% interval (forecast mode).

The demo runs on a *synthetic* dataset generated at startup: the real Zhuoshui agency
series cannot be redistributed, and this Space is public. The synthetic generator mirrors
the real signal structure (seasonal recession + rainfall response + upstream coupling),
so it exercises the exact same models and code paths. The headline scores in the GitHub
README are measured on the real 61-well data.

Models are trained once at startup and cached; moving the slider only re-plots.
"""

from __future__ import annotations

import tempfile

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import gradio as gr  # noqa: E402

from hydrophysics.config import Config  # noqa: E402
from hydrophysics.data import load_dataset  # noqa: E402
from hydrophysics.metrics import kge  # noqa: E402
from hydrophysics.sample import write_sample  # noqa: E402

# ---------------------------------------------------------------------------
# One-time setup: build a demo dataset and train the two models (CPU).
# ---------------------------------------------------------------------------
N_WELLS = 8
Z90 = 1.6449   # 90% Gaussian quantile


def _build():
    tmp = tempfile.mkdtemp(prefix="hydro_demo_")
    d = write_sample(tmp, n_wells=N_WELLS, seed=7, start="2014-01-01", end="2019-12-31")
    data = load_dataset(Config(data_dir=d))

    from hydrophysics.models.forecast_lstm import GlobalForecastLSTM
    from hydrophysics.models.ude import PhysicsUDE

    ude = PhysicsUDE(device="cpu", epochs=400, seed=0).fit(data)
    sim = ude.simulate(data)

    fc = GlobalForecastLSTM(lookback=90, horizon=30, hidden=64, epochs=60,
                            probabilistic=True, device="cpu", seed=0).fit(data)
    mean_cube, sigma_cube = fc.forecast_dist(data)
    return data, sim, mean_cube, sigma_cube


DATA, SIM, MEAN, SIGMA = _build()
WELL_IDS = list(DATA.well_ids)
VAL_IDX = np.where(DATA.val_mask)[0]
V0, V1 = int(VAL_IDX[0]), int(VAL_IDX[-1])


def render(well_id: str, origin_frac: float):
    """Re-plot from cached arrays for the chosen well + forecast origin."""
    i = DATA.well_index(well_id)
    H = MEAN.shape[2]
    # forecast origin somewhere in the validation period
    t0 = int(V0 + origin_frac * max(1, (V1 - H - V0)))
    t0 = max(V0, min(t0, DATA.n_days - H - 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # --- panel 1: free-running simulation ---
    val_k = kge(DATA.target[i, DATA.val_mask], SIM[i, DATA.val_mask])
    ax1.plot(DATA.dates, DATA.target[i], color="0.25", lw=1.1, label="observed")
    ax1.plot(DATA.dates, SIM[i], color="tab:green", lw=1.2,
             label=f"PhysicsUDE simulation (val KGE {val_k:.2f})")
    ax1.axvspan(DATA.dates[V0], DATA.dates[V1], color="orange", alpha=0.08)
    ax1.set_title(f"Free-running simulation - well {well_id}  (validation period shaded)",
                  fontsize=11)
    ax1.set_ylabel("level (m)")
    ax1.legend(fontsize=9, loc="best", framealpha=0.85)

    # --- panel 2: probabilistic forecast fan from t0 ---
    mean = MEAN[i, t0]
    sigma = SIGMA[i, t0]
    hx = [t0 + h for h in range(1, H + 1) if t0 + h < DATA.n_days]
    m = len(hx)
    fut_dates = [DATA.dates[t] for t in hx]
    mean, sigma = mean[:m], sigma[:m]
    look = 90
    h0 = max(0, t0 - look)
    ax2.plot(DATA.dates[h0:t0 + 1], DATA.target[i, h0:t0 + 1],
             color="0.25", lw=1.2, label="observed (history)")
    ax2.plot(fut_dates, DATA.target[i, t0 + 1:t0 + 1 + m], "o", color="0.1", ms=3,
             label="observed (realized)")
    ax2.plot(fut_dates, mean, color="tab:blue", lw=1.6, label="forecast mean")
    ax2.fill_between(fut_dates, mean - Z90 * sigma, mean + Z90 * sigma,
                     color="tab:blue", alpha=0.20, label="90% interval")
    ax2.axvline(DATA.dates[t0], color="orange", ls="--", lw=1, alpha=0.8)
    ax2.set_title(f"Probabilistic {m}-day forecast from {DATA.dates[t0].date()}",
                  fontsize=11)
    ax2.set_ylabel("level (m)")
    ax2.set_xlabel("date")
    ax2.legend(fontsize=9, loc="best", framealpha=0.85)

    fig.tight_layout()
    return fig


INTRO = """
# HydroPhysicsAI - live demo

One GPU-trained **physics-informed neural operator** across many groundwater wells.
Pick a well and a forecast origin:

- **Top:** a *free-running* PhysicsUDE simulation (never sees observed levels in the
  validation window) vs the truth - this is the hard "simulation mode" task.
- **Bottom:** a *probabilistic* 30-day forecast with a calibrated 90% interval.

> Running on a **synthetic** dataset (the real Zhuoshui agency series can't be
> redistributed). Same models, same code. Real-data scores are in the
> [GitHub README](https://github.com/Rekin226/HydroPhysicsAI).
"""


def build_ui():
    with gr.Blocks(title="HydroPhysicsAI demo") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            well = gr.Dropdown(WELL_IDS, value=WELL_IDS[0], label="Well")
            origin = gr.Slider(0.0, 1.0, value=0.1, step=0.02,
                               label="Forecast origin (position in validation period)")
        plot = gr.Plot(label="")
        inputs = [well, origin]
        well.change(render, inputs, plot)
        origin.change(render, inputs, plot)
        demo.load(render, inputs, plot)
    return demo


if __name__ == "__main__":
    build_ui().launch()
