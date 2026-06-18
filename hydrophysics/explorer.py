"""Choushui head + subsidence explorer: grid/mask + interactive Plotly HTML.

Builds a self-contained HTML that animates the IDW observed-head surface and the
calibrated subsidence surface over the real fan, with a validation panel. See
docs/superpowers/specs/2026-06-19-choushui-head-subsidence-explorer-design.md.
"""

from __future__ import annotations

import numpy as np

from .data import GWData
from .subsidence import cumulative_drawdown, idw_interp, monthly_heads, well_xy


def head_grid(data: GWData, poly, n: int = 60):
    """IDW the monthly observed heads onto an n x n grid over the well bbox, masked to
    ``poly`` (a shapely polygon). Returns (XX, YY, HH, dates) with HH shape (Tm, n, n);
    cells outside the polygon are NaN.
    """
    from shapely.geometry import Point

    wxy = well_xy(data)
    xs = np.linspace(wxy[:, 0].min(), wxy[:, 0].max(), n)
    ys = np.linspace(wxy[:, 1].min(), wxy[:, 1].max(), n)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=-1)            # (n*n, 2)
    H, dates = monthly_heads(data)                              # (W, Tm)
    grid = idw_interp(pts, wxy, H)                              # (n*n, Tm)
    inside = np.array([poly.contains(Point(px, py)) for px, py in pts])
    grid[~inside] = np.nan
    HH = grid.T.reshape(len(dates), n, n)                      # (Tm, n, n)
    return XX, YY, HH, dates


def subsidence_grid(HH: np.ndarray, sk: float) -> np.ndarray:
    """Sk * cumulative drawdown per cell, from the (Tm, n, n) head history."""
    Tm = HH.shape[0]
    flat = HH.reshape(Tm, -1).T                                # (cells, Tm)
    D = cumulative_drawdown(flat)                              # (cells, Tm)
    return (sk * D).T.reshape(HH.shape)


def fan_polygon(shp_path):
    """Load the Zhuoshui fan polygon (EPSG:3826) as a single shapely geometry."""
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    return gdf.geometry.union_all()


def build_explorer(data: GWData, poly, stations, compaction, out_html: str,
                   n: int = 60) -> str:
    """Assemble the interactive HTML and write it to ``out_html``. Returns the path.

    A Head/Subsidence toggle switches between the IDW observed-head surface and the
    Sk-scaled subsidence surface; both animate over months via the slider. ``stations``
    (DataFrame[sub_id,x,y]) and ``compaction`` (dict name->Series) may be None when MLCW
    data is unavailable; then Sk falls back to 0 and the validation panel is omitted.
    """
    import os

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .subsidence import calibrate_sk

    XX, YY, HH, dates = head_grid(data, poly, n=n)
    if stations is not None and compaction:
        cal = calibrate_sk(data, stations, compaction)
    else:
        cal = {"sk": 0.0, "r2": float("nan"), "per_site": {}, "D": np.array([]),
               "C": np.array([])}
    SS = subsidence_grid(HH, cal["sk"])
    wxy = well_xy(data)
    zmark = float(np.nanmax(HH)) if np.isfinite(HH).any() else 0.0

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.7, 0.3],
        specs=[[{"type": "surface"}, {"type": "xy"}]],
        subplot_titles=("Groundwater head (m) — IDW of observed wells",
                        f"MLCW validation (Sk={cal['sk']:.4g}, R²={cal['r2']:.2f})"),
    )

    # trace 0: head surface (visible); trace 1: subsidence surface (hidden until toggled)
    fig.add_trace(go.Surface(x=XX, y=YY, z=HH[0], colorscale="Viridis",
                             colorbar=dict(title="head m", x=0.62)), row=1, col=1)
    fig.add_trace(go.Surface(x=XX, y=YY, z=SS[0], colorscale="Reds", visible=False,
                             showscale=False), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=wxy[:, 0], y=wxy[:, 1], z=zmark * np.ones(len(wxy)),
                               mode="markers", marker=dict(size=2, color="white"),
                               name="wells"), row=1, col=1)
    if stations is not None:
        sx = stations["x"].to_numpy(dtype=float)
        sy = stations["y"].to_numpy(dtype=float)
        fig.add_trace(go.Scatter3d(x=sx, y=sy, z=zmark * np.ones(len(sx)),
                                   mode="markers",
                                   marker=dict(size=4, color="red", symbol="diamond"),
                                   name="MLCW sites"), row=1, col=1)

    if cal["D"].size:
        fig.add_trace(go.Scatter(x=cal["sk"] * cal["D"], y=cal["C"], mode="markers",
                                 marker=dict(size=5, color="firebrick"),
                                 name="MLCW pairs"), row=1, col=2)
        lim = float(max(cal["C"].max(), (cal["sk"] * cal["D"]).max(), 1e-6))
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                 line=dict(dash="dash", color="gray"), name="1:1"),
                      row=1, col=2)
        fig.update_xaxes(title_text="predicted subsidence (m)", row=1, col=2)
        fig.update_yaxes(title_text="observed compaction (m)", row=1, col=2)

    frames = []
    for t, dt in enumerate(dates):
        frames.append(go.Frame(name=str(dt.date()),
                               data=[go.Surface(x=XX, y=YY, z=HH[t]),
                                     go.Surface(x=XX, y=YY, z=SS[t])],
                               traces=[0, 1]))
    fig.frames = frames
    steps = [dict(method="animate", label=str(dt.date()),
                  args=[[str(dt.date())], dict(mode="immediate",
                        frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for dt in dates]
    fig.update_layout(
        title="Choushui groundwater head + land subsidence (observed, 2010–2022)",
        sliders=[dict(active=0, steps=steps, x=0.05, len=0.6,
                      currentvalue=dict(prefix="month: "))],
        updatemenus=[dict(type="buttons", direction="right", x=0.05, y=1.12, buttons=[
            dict(label="Head", method="restyle", args=[{"visible": [True, False]}, [0, 1]]),
            dict(label="Subsidence", method="restyle",
                 args=[{"visible": [False, True]}, [0, 1]]),
        ])],
        scene=dict(xaxis_title="TM_X97 (m)", yaxis_title="TM_Y97 (m)",
                   zaxis_title="head (m) / subsidence (m)", aspectmode="auto"),
        margin=dict(l=0, r=0, t=80, b=0),
    )

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn", auto_play=False)
    return out_html


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from .config import Config, default_config
    from .data import load_dataset
    from .subsidence import load_mlcw_stations, mlcw_compaction

    ap = argparse.ArgumentParser(description="Build the Choushui head+subsidence explorer HTML")
    ap.add_argument("--data", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="results/explorer/choushui_explorer.html")
    args = ap.parse_args(argv)

    cfg = Config(data_dir=Path(args.data)) if args.data else default_config()
    data = load_dataset(cfg)
    fan_shp = cfg.data_dir / "Zhuoshui Alluvial Fan" / "Zhuoshui Alluvial Fan.shp"
    poly = fan_polygon(str(fan_shp))
    st_path = cfg.data_dir / "mlcw_stations.csv"
    stations = load_mlcw_stations(st_path) if st_path.exists() else None
    compaction = mlcw_compaction(str(cfg.data_dir)) if stations is not None else None
    out = build_explorer(data, poly, stations, compaction, args.out, n=args.n)
    print(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
