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
from dataclasses import replace
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from demo_data import build_synthetic_data, load_artifact

from hydrophysics.metrics import kge

APP_DIR = Path(__file__).resolve().parent
GEO_DIR = APP_DIR / "geo"
ARTIFACT = APP_DIR / "artifact.npz"
OPERATOR_PT = APP_DIR / "models" / "operator.pt"

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
    ude, sim, mean, sigma, et0, net_data = _train_and_predict(
        data, device="cpu", ude_epochs=300, fc_epochs=40)
    # Hand back the same 6-tuple load_artifact returns (rainfall = raw synthetic rain).
    rainfall = np.nan_to_num(data.rainfall).astype(float)
    return net_data, sim, mean, sigma, rainfall, et0


DATA, SIM, MEAN, SIGMA, RAINFALL, ET0 = _setup()


def _load_operator():
    """Rebuild the shipped trained PhysicsUDE operator (CPU), or None.

    The operator ships as a state_dict + config + frozen stats and is loaded with
    torch.load(weights_only=True) (no pickle-of-code execution surface). It is trained on
    NET recharge (rain - ET0), so its frozen rain-memory stats line up with the baseline
    forcing reconstructed in DATA.rainfall. Falls back to None (what-if disabled,
    baseline-only) if the model is missing or unloadable.
    """
    if not OPERATOR_PT.exists():
        return None
    try:
        from demo_data import load_operator
        return load_operator(OPERATOR_PT, device="cpu")
    except Exception as exc:  # pragma: no cover - defensive load
        print(f"[what-if] operator load failed ({exc}); what-if disabled")
        return None


OPERATOR = _load_operator()
# Baseline net recharge (synthetic rain - real ET0) the operator was trained on, and the
# baseline simulation it produces. Computed once; the what-if perturbs around this.
WHATIF_OK = OPERATOR is not None and np.isfinite(RAINFALL).any() and np.isfinite(ET0).any()
# ~1 s CPU forward over all 61 wells when the operator is present, else the shipped sim.
BASE_SIM = OPERATOR.simulate(DATA) if WHATIF_OK else SIM
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


# ---------------------------------------------------------------------------
# Spatial groundwater surface: interpolate the operator's simulated table across
# the fan on any day. Numpy-only inverse-distance weighting on a grid clipped to
# the real fan polygon (no scipy / geopandas dependency on the Space).
# ---------------------------------------------------------------------------
WELL_X = DATA.attrs["tm_x"].to_numpy(dtype=float)
WELL_Y = DATA.attrs["tm_y"].to_numpy(dtype=float)
_FAN_RING = np.array(GEO_FAN["rings"][0], dtype=float) if GEO_FAN.get("rings") else None
GRID_N = 70   # grid resolution per axis (kept modest so re-interpolation is instant)


def _point_in_poly(px, py, poly):
    """Vectorized ray-casting mask: True where (px, py) falls inside ``poly`` (N, 2)."""
    x = px.ravel()
    y = py.ravel()
    inside = np.zeros(x.shape, dtype=bool)
    vx, vy = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = (vy[i] > y) != (vy[j] > y)
        xint = (vx[j] - vx[i]) * (y - vy[i]) / (vy[j] - vy[i] + 1e-12) + vx[i]
        inside ^= cond & (x < xint)
        j = i
    return inside.reshape(px.shape)


def _build_grid():
    """Fan-clipped interpolation grid (GX, GY, inside-fan mask). Computed once."""
    if _FAN_RING is None:
        return None, None, None
    x0, x1 = _FAN_RING[:, 0].min(), _FAN_RING[:, 0].max()
    y0, y1 = _FAN_RING[:, 1].min(), _FAN_RING[:, 1].max()
    gx, gy = np.meshgrid(np.linspace(x0, x1, GRID_N), np.linspace(y0, y1, GRID_N))
    return gx, gy, _point_in_poly(gx, gy, _FAN_RING)


GRID_X, GRID_Y, FAN_MASK = _build_grid()
SURFACE_OK = GRID_X is not None
if SURFACE_OK:
    # Precompute the k-nearest wells + IDW weights per grid cell ONCE (geometry is
    # static); each day only re-weights the cached neighbour values -> instant scrub.
    _gx, _gy = GRID_X.ravel(), GRID_Y.ravel()
    _d = np.sqrt((_gx[:, None] - WELL_X[None, :]) ** 2
                 + (_gy[:, None] - WELL_Y[None, :]) ** 2) + 1e-6
    _K = 12
    _NN = np.argsort(_d, axis=1)[:, :_K]               # (cells, K) nearest well idx
    _DK = np.take_along_axis(_d, _NN, axis=1)
    _W = 1.0 / _DK ** 1.5                              # IDW weights (p=1.5: smoother field)
    _W /= _W.sum(axis=1, keepdims=True)


def _surface(values: np.ndarray) -> np.ndarray:
    """IDW-interpolate per-well ``values`` (W,) onto the fan grid; NaN outside the fan."""
    z = (_W * values[_NN]).sum(axis=1).reshape(GRID_X.shape)
    return np.where(FAN_MASK, z, np.nan)


# --- live recharge-stress monitor (real Open-Meteo) ------------------------
# Per-well WGS84 lon/lat (precomputed from TWD97; ships as a tiny npz so the Space
# needs no pyproj). Enables a live weather pull at each station's true location.
def _load_lonlat() -> dict:
    p = APP_DIR / "well_lonlat.npz"
    if not p.exists():
        return {}
    z = np.load(p, allow_pickle=True)
    return {str(w): (float(la), float(lo))
            for w, la, lo in zip(z["well_ids"], z["lat"], z["lon"], strict=True)}


WELL_LONLAT = _load_lonlat()
try:
    from live import recharge_stress
    LIVE_OK = bool(WELL_LONLAT)
except Exception:  # pragma: no cover - defensive
    recharge_stress = None
    LIVE_OK = False


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

# Per-well validation KGE of the PhysicsUDE simulation, surfaced on map hover so a station
# shows its skill at a glance.
KGE_BY_WELL = {
    w: float(kge(DATA.target[DATA.well_index(w), DATA.val_mask],
                 SIM[DATA.well_index(w), DATA.val_mask]))
    for w in WELL_IDS
}


def _label(well_id: str) -> str:
    m = META[well_id]
    dist = f", ~{m['dist_km']:.1f} km to coast" if m["dist_km"] is not None else ""
    return f"{m['name']} ({well_id}) - {m['group']}{dist}"


WELL_LABELS = [_label(w) for w in WELL_IDS]
LABEL_TO_ID = dict(zip(WELL_LABELS, WELL_IDS, strict=True))


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

    # Station markers, colored by group. Hover gives the full station info (the
    # "click a station for details" the map provides natively).
    def _hover_row(w):
        d = META[w]["dist_km"]
        dist = f"{d:.1f} km to coast" if d is not None else "distance n/a"
        return [META[w]["name"], w, META[w]["group"], dist, KGE_BY_WELL[w]]

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
            customdata=[_hover_row(w) for w in members],
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "%{customdata[2]} · %{customdata[3]}<br>"
                "sim val KGE %{customdata[4]:.2f}<br>"
                "TM_X %{x:.0f}, TM_Y %{y:.0f}"
                "<extra></extra>"
            ),
        ))

    # Selected station: orange star, same rich hover.
    m = META[well_id]
    fig.add_trace(go.Scatter(
        x=[m["x"]], y=[m["y"]], mode="markers",
        marker=dict(size=20, color=SELECT_COLOR, symbol="star",
                    line=dict(color="black", width=1)),
        name="selected", showlegend=True,
        customdata=[_hover_row(well_id)],
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]}) — selected<br>"
            "%{customdata[2]} · %{customdata[3]}<br>"
            "sim val KGE %{customdata[4]:.2f}<extra></extra>"
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Real station locations on the Zhuoshui alluvial fan<br>"
                 "<sub>Displayed series are synthetic illustrations · "
                 "drag to zoom, hover a station for details</sub>",
            x=0.5, xanchor="center", y=0.97, yanchor="top", font=dict(size=15),
        ),
        xaxis_title="TM_X97 (m, TWD97)",
        yaxis_title="TM_Y97 (m, TWD97)",
        template="plotly_white",
        height=560,
        margin=dict(l=65, r=25, t=75, b=80),
        # Legend along the bottom so it never collides with the title.
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0.5, xanchor="center"),
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


def _whatif_sim(rain_mult: float, et_mult: float) -> np.ndarray:
    """Live operator forward pass under perturbed forcing -> (W, T) simulated levels.

    Builds net recharge = synthetic_rain*rain_mult - ET0*et_mult, swaps it into a copy of
    DATA (dataclasses.replace), and re-runs the trained operator over all wells (~1 s on
    CPU). This is a real forward pass through the recharge-memory ODE, not a lookup -- so
    cutting rainfall / raising ET propagates through the rain-memory states and the table
    falls with the aquifer's natural lag.
    """
    rain = np.nan_to_num(RAINFALL)
    et0 = np.nan_to_num(ET0)
    net = (rain * float(rain_mult) - et0 * float(et_mult)).astype("float32")
    perturbed = replace(DATA, rainfall=net)
    return OPERATOR.simulate(perturbed)


def make_whatif(well_id: str, rain_mult: float, et_mult: float) -> go.Figure:
    """What-if panel: baseline hindcast vs a re-simulated drought/wet scenario.

    Plots, for the selected well: the synthetic observed level, the baseline operator
    simulation (rain x1, ET x1), and the what-if simulation under the slider multipliers.
    The validation window is shaded. Drag rainfall down (or ET up) to starve recharge and
    watch the simulated table sink below baseline after the recharge-memory lag.
    """
    i = DATA.well_index(well_id)
    dates = DATA.dates

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=DATA.target[i], mode="lines", name="synthetic level",
        line=dict(color="#888", width=1.1),
        hovertemplate="%{x|%Y-%m-%d}<br>synthetic: %{y:.2f} m<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=BASE_SIM[i], mode="lines", name="baseline sim (rain x1, ET x1)",
        line=dict(color="#1f77b4", width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>baseline: %{y:.2f} m<extra></extra>",
    ))

    perturbed = not (abs(rain_mult - 1.0) < 1e-9 and abs(et_mult - 1.0) < 1e-9)
    if perturbed:
        wsim = _whatif_sim(rain_mult, et_mult)
        drier = rain_mult < 1.0 or et_mult > 1.0
        color = "#d62728" if drier else "#2ca02c"
        fig.add_trace(go.Scatter(
            x=dates, y=wsim[i], mode="lines",
            name=f"what-if (rain x{rain_mult:.2f}, ET x{et_mult:.2f})",
            line=dict(color=color, width=2.0),
            hovertemplate="%{x|%Y-%m-%d}<br>what-if: %{y:.2f} m<extra></extra>",
        ))
        # Shade the gap so the drought/wet response reads at a glance.
        fig.add_trace(go.Scatter(
            x=dates, y=BASE_SIM[i], mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=wsim[i], mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(214,39,40,0.12)" if drier else "rgba(44,160,44,0.12)",
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_vrect(
        x0=dates[V0], x1=dates[V1], fillcolor="orange", opacity=0.08,
        line_width=0, annotation_text="validation", annotation_position="top left",
    )
    sub = ("baseline only - move a slider to create a scenario" if not perturbed
           else "drought scenario: less recharge -> lower table (lagged)" if (rain_mult < 1.0 or et_mult > 1.0)
           else "wetter scenario: more recharge -> higher table (lagged)")
    fig.update_layout(
        title=f"What-if physics (synthetic) - {_label(well_id)}<br><sub>{sub}</sub>",
        yaxis_title="level (m, synthetic)",
        template="plotly_white", height=420,
        margin=dict(l=60, r=20, t=66, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title="date"),
    )
    return fig


def _surface_day(day_frac: float) -> int:
    """Map a 0..1 slider position to a day index across the full record."""
    k = int(round(day_frac * (DATA.n_days - 1)))
    return max(0, min(k, DATA.n_days - 1))


def make_surface(well_id: str, day_frac: float) -> go.Figure:
    """Spatial panel: the operator's simulated water table interpolated across the fan.

    A filled contour of BASE_SIM at the chosen day, clipped to the real fan polygon, with
    the fan outline, rivers, all 61 stations, and the selected station starred. Scrubbing
    the day slider re-weights the cached IDW neighbours (instant) so the basin-wide table
    rises and falls through the seasons. The surface is the operator's hindcast, not raw
    observations — it is what the model thinks the table looks like everywhere, not just at
    the wells.
    """
    t = _surface_day(day_frac)
    fig = go.Figure()
    if SURFACE_OK:
        z = _surface(BASE_SIM[:, t])
        x1d = GRID_X[0, :]
        y1d = GRID_Y[:, 0]
        fig.add_trace(go.Contour(
            x=x1d, y=y1d, z=z, colorscale="Viridis", connectgaps=False,
            colorbar=dict(title="level (m)", thickness=14, len=0.8),
            contours=dict(coloring="fill"),
            hovertemplate="TM_X %{x:.0f}, TM_Y %{y:.0f}<br>table %{z:.2f} m<extra></extra>",
        ))
    # Fan outline for context.
    for ring in GEO_FAN.get("rings", []):
        fig.add_trace(go.Scatter(
            x=[p[0] for p in ring], y=[p[1] for p in ring], mode="lines",
            line=dict(color="rgba(80,60,20,0.8)", width=1.4), hoverinfo="skip",
            showlegend=False,
        ))
    # Rivers, faint, for orientation -- all segments in ONE trace (None-separated) so the
    # figure stays light instead of carrying ~750 separate line traces.
    riv_x, riv_y = [], []
    for ln in GEO_RIVERS.get("lines", []):
        riv_x.extend([p[0] for p in ln] + [None])
        riv_y.extend([p[1] for p in ln] + [None])
    if riv_x:
        fig.add_trace(go.Scatter(
            x=riv_x, y=riv_y, mode="lines",
            line=dict(color="rgba(255,255,255,0.45)", width=0.6), hoverinfo="skip",
            showlegend=False,
        ))
    # All stations as small dots; selected one starred.
    fig.add_trace(go.Scatter(
        x=WELL_X, y=WELL_Y, mode="markers",
        marker=dict(size=4, color="rgba(0,0,0,0.55)"), hoverinfo="skip", showlegend=False,
    ))
    m = META[well_id]
    fig.add_trace(go.Scatter(
        x=[m["x"]], y=[m["y"]], mode="markers",
        marker=dict(size=16, color=SELECT_COLOR, symbol="star",
                    line=dict(color="black", width=1)),
        name="selected", showlegend=False,
        hovertemplate=f"<b>{m['name']}</b> ({well_id})<extra></extra>",
    ))
    note = ("simulated water table across the fan"
            if SURFACE_OK else "spatial surface unavailable (fan geometry missing)")
    fig.update_layout(
        title=f"Basin groundwater surface — {DATA.dates[t].date()} (synthetic)<br>"
              f"<sub>{note} · drag the day slider to scrub through the seasons</sub>",
        xaxis_title="TM_X97 (m, TWD97)", yaxis_title="TM_Y97 (m, TWD97)",
        template="plotly_white", height=520,
        margin=dict(l=65, r=20, t=70, b=50),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _stress_prompt(msg: str, cta: bool = False) -> go.Figure:
    """A placeholder figure carrying a message.

    ``cta=True`` renders a prominent, button-like call-to-action (the initial "click to
    load" state, which would otherwise look like an empty panel). ``cta=False`` is a plain
    gray note used for the unavailable / fetch-failed cases.
    """
    fig = go.Figure()
    if cta:
        fig.add_annotation(
            text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            align="center", font=dict(size=18, color="#1f77b4"),
            bordercolor="#1f77b4", borderwidth=1, borderpad=18,
            bgcolor="rgba(31,119,180,0.06)")
    else:
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=14, color="#555"))
    fig.update_layout(template="plotly_white", height=360,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=40, r=20, t=50, b=30),
                      title="Live recharge-stress monitor (real Open-Meteo)")
    return fig


def make_stress(well_id: str) -> go.Figure:
    """Live panel: real recent weather -> the operator's recharge-memory anomaly.

    Pulls actual precipitation + ET0 at the station's true coordinates from Open-Meteo's
    public ERA5 archive, runs net recharge (rain - ET0) through the same ~100-day
    recharge-memory the operator uses, and standardizes it against this location's own
    multi-year climatology. Positive (green) = the aquifer is recharging faster than
    normal; negative (red) = real drought stress right now. Everything here is real public
    data -- it is the live forcing side of the model, shown where the synthetic level
    cannot be.
    """
    if not LIVE_OK or recharge_stress is None:
        return _stress_prompt("Live monitor unavailable (coordinate lookup missing).")
    lat, lon = WELL_LONLAT.get(well_id, (None, None))
    if lat is None:
        return _stress_prompt("No coordinates for this station.")
    r = recharge_stress(lat, lon)
    if not r.get("ok"):
        return _stress_prompt(r.get("status", "Live data unavailable — try again."))

    dates = np.array(r["dates"], dtype="datetime64[D]")
    z = np.asarray(r["z"], float)
    pos = np.where(z >= 0, z, 0.0)
    neg = np.where(z < 0, z, 0.0)

    fig = go.Figure()
    # Recharging (green, above zero) and depleting (red, below zero) filled to baseline.
    fig.add_trace(go.Scatter(
        x=dates, y=pos, mode="lines", line=dict(width=0), fill="tozeroy",
        fillcolor="rgba(44,160,44,0.35)", name="recharging", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=dates, y=neg, mode="lines", line=dict(width=0), fill="tozeroy",
        fillcolor="rgba(214,39,40,0.35)", name="depleting", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=dates, y=z, mode="lines", line=dict(color="#333", width=1.4),
        name="recharge anomaly",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f}σ<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="#888", width=1))

    cur = r["current"]
    color = "#2ca02c" if cur > 0.5 else "#d62728" if cur < -0.5 else "#555"
    fig.add_annotation(
        x=dates[-1], y=z[-1],
        text=f"<b>now: {cur:+.2f}σ</b><br>{r['label']}",
        showarrow=True, arrowhead=2, ax=-60, ay=-40,
        font=dict(size=12, color=color), bordercolor=color, borderwidth=1,
        bgcolor="rgba(255,255,255,0.85)")
    m = META[well_id]
    fig.update_layout(
        title=f"Live recharge-stress — {m['name']} ({well_id})<br>"
              f"<sub>real Open-Meteo ERA5 · 100-day recharge-memory anomaly vs local "
              f"climatology · {r['status']}</sub>",
        yaxis_title="recharge anomaly (σ vs normal)",
        xaxis_title="date", template="plotly_white", height=400,
        margin=dict(l=60, r=20, t=70, b=70), hovermode="x unified",
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0.5, xanchor="center"))
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
    # Anchor the forecast to the last observed point (the origin t0) so the mean line and
    # the 90% band emanate from the data with no visual gap. The band has zero width at
    # the origin (the level there is observed, not predicted).
    o_date = DATA.dates[t0]
    o_val = float(DATA.target[i, t0])
    fc_x = [o_date] + fut_dates
    mean_a = np.concatenate([[o_val], mean])
    lo_a = np.concatenate([[o_val], lo])
    hi_a = np.concatenate([[o_val], hi])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_dates, y=DATA.target[i, h0:t0 + 1], mode="lines",
        name="synthetic history", line=dict(color="#444", width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} m<extra></extra>",
    ))
    # 90% band (anchored at the origin): lower bound then upper with fill='tonexty'
    fig.add_trace(go.Scatter(
        x=fc_x, y=lo_a, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc_x, y=hi_a, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(31,119,180,0.20)",
        name="90% interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=fc_x, y=mean_a, mode="lines", name="forecast mean",
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


def render(well_id: str, origin_frac: float, rain_mult: float, et_mult: float,
           day_frac: float):
    """Re-render the map + all charts from cached arrays / one operator pass."""
    return (make_map(well_id), make_sim(well_id), make_forecast(well_id, origin_frac),
            make_whatif(well_id, rain_mult, et_mult), make_surface(well_id, day_frac))


def on_dropdown(label: str, origin_frac: float, rain_mult: float, et_mult: float,
                day_frac: float):
    well_id = LABEL_TO_ID.get(label, WELL_IDS[0])
    return render(well_id, origin_frac, rain_mult, et_mult, day_frac)


def on_slider(label: str, origin_frac: float):
    well_id = LABEL_TO_ID.get(label, WELL_IDS[0])
    return make_forecast(well_id, origin_frac)


def on_whatif(label: str, rain_mult: float, et_mult: float):
    """Slider-release handler: re-simulate the selected well under the perturbed forcing."""
    well_id = LABEL_TO_ID.get(label, WELL_IDS[0])
    return make_whatif(well_id, rain_mult, et_mult)


def on_surface(label: str, day_frac: float):
    """Day-slider handler: re-interpolate the basin surface for the chosen day."""
    well_id = LABEL_TO_ID.get(label, WELL_IDS[0])
    return make_surface(well_id, day_frac)


def on_stress(label: str):
    """Fetch-button handler: pull live Open-Meteo weather for the selected station."""
    well_id = LABEL_TO_ID.get(label, WELL_IDS[0])
    return make_stress(well_id)


def on_click(station_id: str, origin_frac: float, rain_mult: float, et_mult: float,
             day_frac: float):
    """Click-to-select handler. A plotly_click on a station marker writes that station's
    id into a hidden textbox (via BIND_JS below); this syncs the dropdown and redraws.
    No-op for clicks that aren't on a station marker."""
    if station_id not in WELL_IDS:
        return (gr.update(),) * 6
    label = _label(station_id)
    return (gr.update(value=label), make_map(station_id), make_sim(station_id),
            make_forecast(station_id, origin_frac), make_whatif(station_id, rain_mult, et_mult),
            make_surface(station_id, day_frac))


# JS bridge: gr.Plot (plotly) has no native click event in Gradio 6.18, so we bind
# plotly's own `plotly_click` on the map markers and push the clicked station id into a
# hidden textbox, whose .change drives selection. A MutationObserver re-binds after every
# re-render (Gradio recreates the plot div). Station markers carry customdata = [name, id,
# group, dist, kge]; we read index 1 (the st_id) and ignore non-station clicks.
BIND_JS = """
() => {
  const findBox = () => document.querySelector('#wellclick textarea, #wellclick input');
  const bind = (gd) => {
    if (!gd || gd.__wbound || !gd.on) return;
    gd.__wbound = true;
    gd.on('plotly_click', (ev) => {
      const pt = ev && ev.points && ev.points[0];
      const cd = pt && pt.customdata;
      const sid = Array.isArray(cd) ? cd[1] : null;
      if (!sid || typeof sid !== 'string') return;
      const box = findBox();
      if (!box) return;
      box.value = sid;
      box.dispatchEvent(new Event('input', { bubbles: true }));
    });
  };
  const scan = () => document.querySelectorAll('.js-plotly-plot').forEach(bind);
  scan();
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
}
"""


INTRO = """
# HydroPhysicsAI - live demo

One GPU-trained **physics-informed neural operator** across the 61 groundwater
monitoring stations of Taiwan's Zhuoshui alluvial fan.

**Pick a station** by clicking it on the map, or from the descriptive dropdown (its true
location lights up with an orange star), then choose a forecast origin:

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


# Geographic panels (map + basin surface) lock equal x/y scale for true geography, so
# their data is portrait (the fan is ~0.77 as wide as it is tall). In a full-width
# container that leaves huge horizontal margins, so we cap their width and center them,
# keeping the figure roughly square instead of a wide rectangle with the fan stranded
# in the middle. Injected as a <style> tag (not gr.Blocks(css=...)): Gradio 6 moved the
# css= kwarg to launch(), which the HF Spaces runtime owns, so a constructor css= would
# be silently dropped on the Space.
STYLE = """
<style>
.geo-square { max-width: 600px; margin-left: auto !important; margin-right: auto !important; }
/* Keep the click-bridge textbox in the DOM (so BIND_JS can find it) but out of sight.
   gr.Textbox(visible=False) is dropped from the DOM entirely in Gradio 6, which would
   silently break map click-to-select. */
.wellclick-hidden { display: none !important; }
</style>
"""


def build_ui():
    with gr.Blocks(title="HydroPhysicsAI demo") as demo:
        gr.HTML(STYLE)
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                map_plot = gr.Plot(label="Station map (real locations) - click a station")
                well = gr.Dropdown(WELL_LABELS, value=WELL_LABELS[0],
                                   label="Select a station (or click it on the map)")
                origin = gr.Slider(0.0, 1.0, value=0.1, step=0.02,
                                   label="Forecast origin (position in validation period)")
            with gr.Column(scale=2):
                sim_plot = gr.Plot(label="Simulation (free-running, synthetic)")
                fc_plot = gr.Plot(label="Probabilistic forecast (synthetic)")

        # --- Interactive what-if physics panel -----------------------------------
        with gr.Group():
            gr.Markdown(
                "## Interactive what-if physics\n"
                "Perturb the forcing on the **selected** station and the operator "
                "**re-simulates live** (a real forward pass through the recharge-memory "
                "ODE). **Drag rainfall down (or ET up) to create a drought** and watch the "
                "groundwater table sink below baseline — with the aquifer's natural lag. "
                "_Series are synthetic; ET0 is real public ERA5._"
                + ("" if WHATIF_OK else
                   "\n\n_(What-if is unavailable: the trained operator or forcing is "
                   "missing from this build — showing the baseline simulation only.)_")
            )
            with gr.Row():
                rain_mult = gr.Slider(0.5, 1.5, value=1.0, step=0.05,
                                      label="Rainfall x  (drag down for drought)",
                                      interactive=WHATIF_OK)
                et_mult = gr.Slider(0.5, 1.5, value=1.0, step=0.05,
                                    label="ET x  (drag up for drought)",
                                    interactive=WHATIF_OK)
            whatif_plot = gr.Plot(label="What-if scenario (baseline vs perturbed, synthetic)")

        # --- Spatial groundwater surface panel -----------------------------------
        with gr.Group():
            gr.Markdown(
                "## Basin groundwater surface\n"
                "The operator's simulated water table **interpolated across the whole "
                "alluvial fan** (inverse-distance weighting on the 61 stations, clipped to "
                "the real fan boundary). **Drag the day slider** to scrub through six years "
                "and watch the basin recharge in the wet season and draw down in the dry. "
                "The orange star is the selected station. _Surface is the model hindcast on "
                "synthetic series; the fan geometry is real._"
                + ("" if SURFACE_OK else
                   "\n\n_(Spatial surface unavailable: the fan geometry is missing from "
                   "this build.)_")
            )
            day = gr.Slider(0.0, 1.0, value=0.55, step=0.01,
                            label="Day (position across the 2014–2019 record)",
                            interactive=SURFACE_OK)
            with gr.Row():
                surface_plot = gr.Plot(
                    label="Basin groundwater surface (synthetic, model hindcast)",
                    elem_classes=["geo-square"])

        # --- Live recharge-stress monitor (real Open-Meteo) ----------------------
        with gr.Group():
            gr.Markdown(
                "## Live recharge-stress monitor — real-time Open-Meteo\n"
                "The groundwater series here are synthetic, so we don't fake a real level. "
                "Instead we show the **real forcing** the operator consumes: click below to "
                "pull **actual recent weather** (precipitation + ET0) at the selected "
                "station's true coordinates from Open-Meteo's public ERA5 archive, run it "
                "through the **same 100-day recharge-memory the operator uses**, and see "
                "whether the aquifer is **recharging (green)** or under **drought stress "
                "(red)** right now — versus this location's own multi-year climatology. "
                "_100% real public data._"
                + ("" if LIVE_OK else
                   "\n\n_(Live monitor unavailable in this build: the coordinate lookup is "
                   "missing.)_")
            )
            fetch_btn = gr.Button("🌦  Pull latest weather for the selected station",
                                  variant="primary", interactive=LIVE_OK)
            stress_plot = gr.Plot(label="Live recharge-stress (real Open-Meteo)")

        # Hidden bridge: BIND_JS writes the clicked station id here; its change selects.
        # CSS-hidden (not visible=False) so the textarea stays in the DOM for the JS bridge
        # to find — Gradio 6 removes visible=False components from the DOM entirely.
        clicked = gr.Textbox(value="", elem_id="wellclick",
                             elem_classes=["wellclick-hidden"])

        sel_inputs = [well, origin, rain_mult, et_mult, day]
        all_plots = [map_plot, sim_plot, fc_plot, whatif_plot, surface_plot]
        # dropdown -> redraw map (re-highlight) + all charts (what-if + surface re-rendered)
        well.change(on_dropdown, sel_inputs, all_plots)
        # map marker click -> sync dropdown + redraw (click-to-select)
        clicked.change(on_click, [clicked, origin, rain_mult, et_mult, day], [well] + all_plots)
        # forecast-origin slider only affects the forecast panel
        origin.change(on_slider, [well, origin], fc_plot)
        # what-if sliders -> live operator re-simulation (on release, not streaming)
        rain_mult.release(on_whatif, [well, rain_mult, et_mult], whatif_plot)
        et_mult.release(on_whatif, [well, rain_mult, et_mult], whatif_plot)
        # day slider -> re-interpolate the basin surface (on release; instant IDW re-weight)
        day.release(on_surface, [well, day], surface_plot)
        # live monitor: explicit button (one Open-Meteo call, cached) -> recharge-stress
        fetch_btn.click(on_stress, [well], stress_plot)
        # initial draw + install the plotly_click -> hidden-textbox JS bridge
        demo.load(on_dropdown, sel_inputs, all_plots)
        demo.load(lambda: (_stress_prompt(
            "⬆&nbsp;&nbsp;<b>Click “Pull latest weather” above</b><br>"
            "<span style='font-size:13px;color:#777'>to load live Open-Meteo recharge "
            "data for the selected station</span>", cta=True)
            if LIVE_OK else
            _stress_prompt("Live monitor unavailable in this build.")),
            None, stress_plot)
        demo.load(js=BIND_JS)
    return demo


# Module-level app object. The Hugging Face Spaces Gradio runtime imports this file and
# launches the top-level ``demo`` itself (managing the queue / SSR proxy); exposing it here
# avoids the "demo not found in __main__" fallback. Locally, run the file directly.
demo = build_ui()

if __name__ == "__main__":
    demo.launch()
