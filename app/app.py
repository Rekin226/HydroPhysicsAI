"""HydroPhysicsAI live demo (Gradio).

Pick a real Zhuoshui monitoring station from a descriptive dropdown (its location
lights up on a real-geography map of the alluvial fan) and see two things the repo
does:
  1. a free-running PhysicsUDE simulation hindcast vs "observed" (simulation mode), and
  2. a probabilistic 30-day forecast with its 90% interval (forecast mode).

All charts are interactive Plotly figures (zoom, pan, hover tooltips), rendered via
Gradio's gr.Plot.

DATA POLICY (important):
  * The MAP is REAL geography: the 61 stations sit at their true TWD97 coordinates on
    the actual Zhuoshui alluvial fan, with the fan outline, rivers, and coastline as a
    backdrop, and real station names on hover. Publishing this location/boundary
    metadata was explicitly approved.
  * The plotted TIME SERIES are SYNTHETIC illustrations. Real groundwater / rainfall
    series cannot be redistributed and are never shipped to this public Space. The
    synthetic generator mirrors the real signal structure (seasonal recession +
    rainfall response + upstream coupling) so the exact model code paths run; nothing
    plotted is a real measurement.

Startup: a precomputed artifact (synthetic series + cached PhysicsUDE simulation and
probabilistic-forecast cubes, built once on GPU by ``app/demo_data.py``) is loaded at
launch, so the Space comes up in seconds with no live training. Changing the station
or moving the slider only re-plots.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import gradio as gr

from demo_data import build_synthetic_data, load_artifact
from hydrophysics.metrics import kge

APP_DIR = Path(__file__).resolve().parent
GEO_DIR = APP_DIR / "geo"
ARTIFACT = APP_DIR / "artifact.npz"

Z90 = 1.6449   # 90% Gaussian quantile

COASTAL_COLOR = "#1f77b4"   # blue
INLAND_COLOR = "#d62728"    # red
SELECT_COLOR = "#ff7f0e"    # orange highlight


# ---------------------------------------------------------------------------
# One-time setup: load the precomputed synthetic artifact (fast). Fall back to a
# live CPU build if the artifact is missing (e.g. fresh checkout).
# ---------------------------------------------------------------------------
def _setup():
    if ARTIFACT.exists():
        return load_artifact(ARTIFACT)
    # Fallback: build + train live (slower; used only without a shipped artifact).
    from demo_data import _train_and_predict
    data = build_synthetic_data()
    sim, mean, sigma = _train_and_predict(data, device="cpu", ude_epochs=300, fc_epochs=40)
    return data, sim, mean, sigma


DATA, SIM, MEAN, SIGMA = _setup()
WELL_IDS = list(DATA.well_ids)
VAL_IDX = np.where(DATA.val_mask)[0]
V0, V1 = int(VAL_IDX[0]), int(VAL_IDX[-1])
H = MEAN.shape[2]
# Origins with a cached (non-NaN) forecast, used to clamp the slider.
FC_ORIGINS = np.where(np.isfinite(MEAN[0, :, 0]))[0]


def _load_geo(name: str) -> dict:
    p = GEO_DIR / name
    if not p.exists():
        return {}
    return json.loads(p.read_text())


GEO_FAN = _load_geo("fan.json")
GEO_RIVERS = _load_geo("rivers.json")
GEO_SEA = _load_geo("sea.json")


# --- per-well descriptive metadata (REAL coords/name/group) ----------------
def _well_meta(well_id: str) -> dict:
    a = DATA.attrs.loc[well_id]
    group = str(a.get("group", "inland"))
    dist = a.get("dist_to_coast_m", np.nan)
    return {
        "id": well_id,
        "name": str(a.get("name", well_id)),
        "group": group,
        "coastal": group == "coastal",
        "x": float(a["tm_x"]),
        "y": float(a["tm_y"]),
        "dist_km": (float(dist) / 1000.0) if np.isfinite(dist) else None,
    }


META = {w: _well_meta(w) for w in WELL_IDS}


def _label(well_id: str) -> str:
    m = META[well_id]
    dist = f", ~{m['dist_km']:.1f} km to coast" if m["dist_km"] is not None else ""
    return f"{m['name']} ({well_id}) - {m['group']}{dist}"


WELL_LABELS = [_label(w) for w in WELL_IDS]
LABEL_TO_ID = dict(zip(WELL_LABELS, WELL_IDS))


def _origin_index(origin_frac: float) -> int:
    """Map slider position -> a forecast origin t0 that has a cached forecast."""
    if len(FC_ORIGINS) == 0:
        return V0
    k = int(round(origin_frac * (len(FC_ORIGINS) - 1)))
    k = max(0, min(k, len(FC_ORIGINS) - 1))
    return int(FC_ORIGINS[k])


# ---------------------------------------------------------------------------
# Real-geography map.
# ---------------------------------------------------------------------------
def make_map(well_id: str) -> go.Figure:
    """Real station locations on the Zhuoshui alluvial fan (TWD97 meters).

    Layers (bottom to top): sea fill, fan outline (light fill), rivers (thin blue
    lines), station markers colored coastal/inland, and the selected station starred.
    """
    fig = go.Figure()

    # Sea (coastline / coastal fill) -- light blue.
    for ring in GEO_SEA.get("rings", []):
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor="rgba(173,216,230,0.45)", line=dict(color="rgba(120,160,200,0.6)", width=1),
            hoverinfo="skip", showlegend=False,
        ))

    # Alluvial fan outline -- light tan fill.
    fan_named = False
    for ring in GEO_FAN.get("rings", []):
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor="rgba(222,200,150,0.30)", line=dict(color="rgba(150,120,60,0.8)", width=1.4),
            name="Zhuoshui alluvial fan", hoverinfo="skip",
            showlegend=not fan_named,
        ))
        fan_named = True

    # Rivers -- thin blue lines (one legend entry).
    riv_named = False
    for ln in GEO_RIVERS.get("lines", []):
        xs = [p[0] for p in ln]
        ys = [p[1] for p in ln]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="rgba(40,110,190,0.55)", width=0.8),
            name="rivers", hoverinfo="skip",
            showlegend=not riv_named, legendgroup="rivers",
        ))
        riv_named = True

    # Station markers, colored by group, with real names on hover.
    for group, color in (("coastal", COASTAL_COLOR), ("inland", INLAND_COLOR)):
        members = [w for w in WELL_IDS if META[w]["group"] == group]
        if not members:
            continue
        fig.add_trace(go.Scatter(
            x=[META[w]["x"] for w in members],
            y=[META[w]["y"] for w in members],
            mode="markers",
            marker=dict(size=9, color=color, line=dict(color="white", width=1)),
            name=f"{group} ({len(members)})",
            customdata=[[META[w]["name"], w, META[w]["group"]] for w in members],
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "%{customdata[2]}<br>"
                "TM_X=%{x:.0f}  TM_Y=%{y:.0f}<extra></extra>"
            ),
        ))

    m = META[well_id]
    fig.add_trace(go.Scatter(
        x=[m["x"]], y=[m["y"]], mode="markers",
        marker=dict(size=20, color=SELECT_COLOR, symbol="star",
                    line=dict(color="black", width=1)),
        name="selected", showlegend=True,
        hovertemplate=f"<b>{m['name']}</b> ({well_id})<br>selected<extra></extra>",
    ))

    fig.update_layout(
        title="Real station locations on the Zhuoshui alluvial fan "
              "(displayed series are synthetic illustrations)",
        xaxis_title="TM_X97 (m, TWD97)",
        yaxis_title="TM_Y97 (m, TWD97)",
        template="plotly_white",
        height=460,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal aspect (true geography)
    return fig


def make_sim(well_id: str) -> go.Figure:
    """Panel A: free-running PhysicsUDE simulation vs synthetic truth, val shaded."""
    i = DATA.well_index(well_id)
    val_k = kge(DATA.target[i, DATA.val_mask], SIM[i, DATA.val_mask])
    dates = DATA.dates

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=DATA.target[i], mode="lines", name="synthetic level",
        line=dict(color="#444", width=1.4),
        hovertemplate="%{x|%Y-%m-%d}<br>synthetic: %{y:.2f} m<extra></extra>",
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
              "(drag the bar below to scrub; series are synthetic)",
        yaxis_title="level (m, synthetic)",
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
        name="synthetic history", line=dict(color="#444", width=1.5),
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
        name="synthetic realized",
        marker=dict(color="#111", size=5),
        hovertemplate="%{x|%Y-%m-%d}<br>realized: %{y:.2f} m<extra></extra>",
    ))
    fig.add_vline(x=DATA.dates[t0], line=dict(color="orange", dash="dash", width=1.2))
    fig.update_layout(
        title=f"Probabilistic {m}-day forecast from {DATA.dates[t0].date()} (synthetic)",
        xaxis_title="date", yaxis_title="level (m, synthetic)",
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

One GPU-trained **physics-informed neural operator** across the 61 groundwater
monitoring stations of Taiwan's Zhuoshui alluvial fan.

**Pick a station** from the descriptive dropdown (its true location lights up on the
map), then choose a forecast origin:

- **Map (real geography):** all 61 stations at their actual TWD97 coordinates on the
  real alluvial fan, with the fan outline, rivers, and coastline as a backdrop.
  Markers are colored **coastal** (blue) vs **inland** (red); the orange star is the
  current selection. Coastal wells behave differently (tidal influence, shorter
  recession), so *where* a well sits matters. Hover a marker for its real name and
  coordinates.
- **Panel A - simulation:** a *free-running* PhysicsUDE hindcast that never sees the
  level in the validation window (shaded), compared to the truth - the hard
  "simulation mode" task. Validation KGE is in the legend.
- **Panel B - forecast:** a *probabilistic* 30-day forecast with a calibrated 90%
  interval (shaded band) from the chosen origin.

All charts are **interactive** - hover for date + value, drag to zoom, double-click to reset.

> **Data policy.** Station **locations, names, coastal/inland group, and the GIS map
> layers are REAL** (publishing this location/boundary metadata was approved). The
> plotted **time series are SYNTHETIC illustrations** - the real Zhuoshui agency
> groundwater/rainfall series are not redistributable and never ship to this public
> Space. The synthetic generator mirrors the real signal structure so the exact model
> code paths run; the headline scores in the
> [GitHub README](https://github.com/Rekin226/HydroPhysicsAI) are measured on the real
> data. Nothing plotted here is a real measurement.
"""


def build_ui():
    with gr.Blocks(title="HydroPhysicsAI demo") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                map_plot = gr.Plot(label="Station map (real locations)")
                well = gr.Dropdown(WELL_LABELS, value=WELL_LABELS[0],
                                   label="Select a station (highlighted on the map)")
                origin = gr.Slider(0.0, 1.0, value=0.1, step=0.02,
                                   label="Forecast origin (position in validation period)")
            with gr.Column(scale=2):
                sim_plot = gr.Plot(label="Simulation (free-running, synthetic)")
                fc_plot = gr.Plot(label="Probabilistic forecast (synthetic)")

        # dropdown -> redraw map (re-highlight) + both charts
        well.change(on_dropdown, [well, origin], [map_plot, sim_plot, fc_plot])
        # slider only affects the forecast panel
        origin.change(on_slider, [well, origin], fc_plot)
        # initial draw
        demo.load(on_dropdown, [well, origin], [map_plot, sim_plot, fc_plot])
    return demo


if __name__ == "__main__":
    build_ui().launch()
