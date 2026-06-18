"""HydroPhysicsAI live demo (Gradio).

Pick a well via a descriptive dropdown (its location is highlighted on an interactive
map) and see two things the repo does:
  1. a free-running PhysicsUDE simulation hindcast vs observed (simulation mode), and
  2. a probabilistic 30-day forecast with its 90% interval (forecast mode).

All charts are interactive Plotly figures (zoom, pan, hover tooltips), rendered via
Gradio's gr.Plot.

The demo runs on a *synthetic* dataset generated at startup: the real Zhuoshui agency
series cannot be redistributed, and this Space is public. The synthetic generator mirrors
the real signal structure (seasonal recession + rainfall response + upstream coupling),
so it exercises the exact same models and code paths. The headline scores in the GitHub
README are measured on the real 61-well data.

Models are trained once at startup and cached; changing the well or moving the slider
only re-plots (no retraining).
"""

from __future__ import annotations

import tempfile

import numpy as np
import plotly.graph_objects as go

import gradio as gr

from hydrophysics.config import Config
from hydrophysics.data import load_dataset
from hydrophysics.metrics import kge
from hydrophysics.sample import write_sample

# ---------------------------------------------------------------------------
# One-time setup: build a demo dataset and train the two models (CPU).
# ---------------------------------------------------------------------------
N_WELLS = 8
Z90 = 1.6449   # 90% Gaussian quantile

COASTAL_COLOR = "#1f77b4"   # blue
INLAND_COLOR = "#d62728"    # red
SELECT_COLOR = "#ff7f0e"    # orange highlight


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
H = MEAN.shape[2]


# --- per-well descriptive metadata derived from the synthetic attrs --------
def _well_meta(well_id: str) -> dict:
    a = DATA.attrs.loc[well_id]
    coastal = bool(int(a.get("is_coastal", 0)))
    group = "coastal" if coastal else "inland"
    dist = a.get("dist_to_coast_m", np.nan)
    return {
        "id": well_id,
        "group": group,
        "coastal": coastal,
        "x": float(a["tm_x"]),
        "y": float(a["tm_y"]),
        "dist_km": (float(dist) / 1000.0) if np.isfinite(dist) else None,
    }


META = {w: _well_meta(w) for w in WELL_IDS}


def _label(well_id: str) -> str:
    m = META[well_id]
    n = WELL_IDS.index(well_id) + 1
    dist = f", ~{m['dist_km']:.1f} km to coast" if m["dist_km"] is not None else ""
    return f"Well {n} ({well_id}) - {m['group']}{dist}"


WELL_LABELS = [_label(w) for w in WELL_IDS]
LABEL_TO_ID = dict(zip(WELL_LABELS, WELL_IDS))


def _origin_index(origin_frac: float) -> int:
    """Map slider position -> a forecast origin t0 with a valid cached forecast."""
    t0 = int(V0 + origin_frac * max(1, (V1 - H - V0)))
    t0 = max(V0, min(t0, DATA.n_days - H - 1))
    return t0


def make_map(well_id: str) -> go.Figure:
    """Scatter all wells on (tm_x, tm_y), colored by group, selected one starred."""
    fig = go.Figure()
    for group, color in (("coastal", COASTAL_COLOR), ("inland", INLAND_COLOR)):
        members = [w for w in WELL_IDS if META[w]["group"] == group]
        if not members:
            continue
        fig.add_trace(go.Scatter(
            x=[META[w]["x"] for w in members],
            y=[META[w]["y"] for w in members],
            mode="markers+text",
            marker=dict(size=15, color=color, line=dict(color="white", width=1)),
            text=[f"W{WELL_IDS.index(w) + 1}" for w in members],
            textposition="top center",
            textfont=dict(size=10),
            name=group,
            customdata=members,
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                + group
                + "<br>TM_X=%{x:.0f}  TM_Y=%{y:.0f}<extra></extra>"
            ),
        ))
    m = META[well_id]
    fig.add_trace(go.Scatter(
        x=[m["x"]], y=[m["y"]], mode="markers",
        marker=dict(size=26, color=SELECT_COLOR, symbol="star",
                    line=dict(color="black", width=1)),
        name="selected", hoverinfo="skip", showlegend=True,
    ))
    fig.update_layout(
        title="Synthetic well network (colored by group; selected well starred)",
        xaxis_title="TM_X97 (synthetic local grid)",
        yaxis_title="TM_Y97 (synthetic local grid)",
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def make_sim(well_id: str) -> go.Figure:
    """Panel A: free-running PhysicsUDE simulation vs observed, val period shaded."""
    i = DATA.well_index(well_id)
    val_k = kge(DATA.target[i, DATA.val_mask], SIM[i, DATA.val_mask])
    dates = DATA.dates

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=DATA.target[i], mode="lines", name="observed",
        line=dict(color="#444", width=1.4),
        hovertemplate="%{x|%Y-%m-%d}<br>observed: %{y:.2f} m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=SIM[i], mode="lines",
        name=f"PhysicsUDE simulation (val KGE {val_k:.2f})",
        line=dict(color="#2ca02c", width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>simulated: %{y:.2f} m<extra></extra>",
    ))
    fig.add_vrect(
        x0=dates[V0], x1=dates[V1], fillcolor="orange", opacity=0.10,
        line_width=0, annotation_text="validation", annotation_position="top left",
    )
    fig.update_layout(
        title=f"Free-running simulation - {_label(well_id)}  "
              "(drag the bar below to scrub the record)",
        yaxis_title="level (m)",
        template="plotly_white", height=440,
        margin=dict(l=60, r=20, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        # Scrubber: a draggable overview bar + quick-range buttons to slide a zoom
        # window across the multi-year record.
        xaxis=dict(
            title="date",
            rangeslider=dict(visible=True, thickness=0.09),
            rangeselector=dict(
                buttons=[
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(step="all", label="all"),
                ],
                x=0, y=1.08,
            ),
        ),
    )
    return fig


def make_forecast(well_id: str, origin_frac: float) -> go.Figure:
    """Panel B: probabilistic forecast from the chosen origin, 90% band shaded."""
    i = DATA.well_index(well_id)
    t0 = _origin_index(origin_frac)
    mean = MEAN[i, t0]
    sigma = SIGMA[i, t0]
    hx = [t0 + h for h in range(1, H + 1) if t0 + h < DATA.n_days]
    m = len(hx)
    fut_dates = [DATA.dates[t] for t in hx]
    mean, sigma = mean[:m], sigma[:m]
    lo, hi = mean - Z90 * sigma, mean + Z90 * sigma

    look = 90
    h0 = max(0, t0 - look)
    hist_dates = DATA.dates[h0:t0 + 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_dates, y=DATA.target[i, h0:t0 + 1], mode="lines",
        name="observed (history)", line=dict(color="#444", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} m<extra></extra>",
    ))
    # 90% band: lower bound then upper with fill='tonexty'
    fig.add_trace(go.Scatter(
        x=fut_dates, y=lo, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fut_dates, y=hi, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(31,119,180,0.20)",
        name="90% interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fut_dates, y=mean, mode="lines", name="forecast mean",
        line=dict(color="#1f77b4", width=2.2),
        hovertemplate="%{x|%Y-%m-%d}<br>mean: %{y:.2f} m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fut_dates, y=DATA.target[i, t0 + 1:t0 + 1 + m], mode="markers",
        name="observed (realized)",
        marker=dict(color="#111", size=5),
        hovertemplate="%{x|%Y-%m-%d}<br>realized: %{y:.2f} m<extra></extra>",
    ))
    fig.add_vline(x=DATA.dates[t0], line=dict(color="orange", dash="dash", width=1.2))
    fig.update_layout(
        title=f"Probabilistic {m}-day forecast from {DATA.dates[t0].date()}",
        xaxis_title="date", yaxis_title="level (m)",
        template="plotly_white", height=360,
        margin=dict(l=60, r=20, t=50, b=45),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def render(well_id: str, origin_frac: float):
    """Re-render the map + both charts from cached arrays. No retraining."""
    return make_map(well_id), make_sim(well_id), make_forecast(well_id, origin_frac)


def on_dropdown(label: str, origin_frac: float):
    well_id = LABEL_TO_ID.get(label, WELL_IDS[0])
    return render(well_id, origin_frac)


def on_slider(label: str, origin_frac: float):
    well_id = LABEL_TO_ID.get(label, WELL_IDS[0])
    return make_forecast(well_id, origin_frac)


INTRO = """
# HydroPhysicsAI - live demo

One GPU-trained **physics-informed neural operator** across many groundwater wells.

**Pick a well** from the descriptive dropdown (its position lights up on the map), then
choose a forecast origin:

- **Map:** all wells on their coordinates, colored coastal vs inland. The orange star
  marks the current selection. Coastal wells behave differently (tidal influence,
  shorter recession), so *where* a well sits matters. Hover a marker for its id and
  coordinates.
- **Panel A - simulation:** a *free-running* PhysicsUDE hindcast that never sees observed
  levels in the validation window (shaded), compared to the truth - the hard
  "simulation mode" task. Validation KGE is in the legend.
- **Panel B - forecast:** a *probabilistic* 30-day forecast with a calibrated 90%
  interval (shaded band) from the chosen origin.

All charts are **interactive** - hover for date + value, drag to zoom, double-click to reset.

> **Synthetic / illustrative data.** This public Space runs on a *synthetic* dataset
> generated at startup (coordinates, coastal/inland labels and series are all
> fabricated) - the real Zhuoshui agency series cannot be redistributed. The models and
> code paths are identical to the real pipeline; real-data headline scores live in the
> [GitHub README](https://github.com/Rekin226/HydroPhysicsAI). Nothing shown here is a
> real measurement.
"""


def build_ui():
    with gr.Blocks(title="HydroPhysicsAI demo") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                map_plot = gr.Plot(label="Well map (synthetic)")
                well = gr.Dropdown(WELL_LABELS, value=WELL_LABELS[0],
                                   label="Select a well (highlighted on the map)")
                origin = gr.Slider(0.0, 1.0, value=0.1, step=0.02,
                                   label="Forecast origin (position in validation period)")
            with gr.Column(scale=2):
                sim_plot = gr.Plot(label="Simulation (free-running)")
                fc_plot = gr.Plot(label="Probabilistic forecast")

        # dropdown -> redraw map (re-highlight) + both charts
        well.change(on_dropdown, [well, origin], [map_plot, sim_plot, fc_plot])
        # slider only affects the forecast panel
        origin.change(on_slider, [well, origin], fc_plot)
        # initial draw
        demo.load(on_dropdown, [well, origin], [map_plot, sim_plot, fc_plot])
    return demo


if __name__ == "__main__":
    build_ui().launch()
