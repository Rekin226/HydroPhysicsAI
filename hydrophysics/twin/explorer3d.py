"""3D viewer for the Choushui twin: layered head volume + subsidence, with scenarios.

Renders the fan as what it physically is -- four stacked aquifers under a deforming
ground surface -- and lets a scenario be dialled in and its subsidence consequence read
off. Output is one self-contained Plotly HTML, the same delivery form as
``hydrophysics.explorer``'s 2D animation.

What drives what
----------------
Head surfaces come from the layer-resolved API well network (``twin.heads``), inverse-
distance interpolated per aquifer onto the fan grid. Subsidence comes from the Stage-2
visco-elasto-plastic column (``twin.compaction.VEPColumn``) fitted as ONE global parameter
set against the MLCW magnetic-ring sites -- the configuration whose LOSO gate passes -- and
then applied per grid cell to the pooled head series at that cell.

Two modes.

**Head-decline mode** (the original): the scenario axis is a multiplier on the observed
drawdown. It needs no flow model and makes no claim about pumping.

**Forward mode** (``--forward-npz``, 2026-09-11): render a ``twin.forward`` run -- the
calibrated flow model's heads under named *pumping policies*, hindcast then projection,
with the Stage-2 column's subsidence on top. The scenario axis is then the policy
(``cut30: irrigation x0.7 from 2026-01``), which is a decision someone can take. The
title carries the Stage-3 verdict recorded with the parameters, because whether the
pumping -> head map is trustworthy is the gate's call and the viewer must not hide it.

    python -m hydrophysics.twin.explorer3d --out results/twin/explorer3d.html
    python -m hydrophysics.twin.explorer3d --forward-npz results/twin_forward/run.npz \
        --out results/twin/explorer3d_forward.html
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..subsidence import idw_interp, load_mlcw_stations, mlcw_compaction
from .compaction import VEPColumn
from .grid import build_grid
from .heads import DEFAULT_LAYERS, build_head_field

# Median screen depth per aquifer, metres below ground, from the zone-50 station metadata
# (WellDepth medians over 103/134/60/19 wells in layers 1-4). Used only to place the
# surfaces vertically in the render; nothing physical depends on it.
LAYER_DEPTH_M = {"1": 53.3, "2": 119.0, "3": 210.5, "4": 282.0}

# Vertical exaggeration. The fan is ~60 km across and ~300 m deep, so a true-scale plot is
# a flat sheet. Subsidence (centimetres) needs its own, much larger factor again.
DEPTH_EXAG = 40.0
SUBS_EXAG = 2000.0


def layer_head_volume(hf, grid, layers: tuple[str, ...] = DEFAULT_LAYERS) -> np.ndarray:
    """IDW each aquifer's wells onto the active cells -> ``(n_layers, n_active, T)``.

    A layer with no wells yields NaN rather than borrowing another layer's heads.
    """
    cent = grid.centroids()
    out = np.full((len(layers), cent.shape[0], hf.heads.shape[1]), np.nan)
    for k, code in enumerate(layers):
        sub = hf.subset(code)
        if len(sub) == 0:
            continue
        out[k] = idw_interp(cent, sub.xy, sub.heads)
    return out


def fit_shared_vep(ddir: str, hf, epochs: int = 2000, lr: float = 0.05,
                   device=None) -> tuple[VEPColumn, dict]:
    """Fit the single global VEP parameter set against the MLCW sites.

    This is Stage 2's ``loso_shared`` arm refit on all sites at once (no hold-out): the
    gate has already established that this configuration generalises, so here it is used
    as a calibrated forward model rather than re-evaluated.
    """
    from .calibrate_mlcw import fit_column

    stations = load_mlcw_stations(os.path.join(ddir, "mlcw_stations.csv"))
    comp = mlcw_compaction(ddir)
    rows_h, rows_o, rows_m = [], [], []
    for _, r in stations.iterrows():
        if r["sub_id"] not in comp:
            continue
        h_site = idw_interp(np.array([[r["x"], r["y"]]], dtype="float64"),
                            hf.xy, hf.heads)[0]
        c = comp[r["sub_id"]].reindex(hf.dates, method="nearest",
                                      tolerance=pd.Timedelta("45D"))
        ok = c.notna().to_numpy()
        if ok.sum() < 24:
            continue
        rows_h.append(h_site)
        rows_o.append(np.nan_to_num(c.to_numpy(dtype="float64")))
        rows_m.append(ok)
    if not rows_h:
        raise SystemExit("no MLCW site had >=24 usable compaction samples")
    h = torch.tensor(np.stack(rows_h), dtype=torch.float32)
    obs = torch.tensor(np.stack(rows_o), dtype=torch.float32)
    mask = torch.tensor(np.stack(rows_m))
    model, info = fit_column(h, obs, mask, epochs=epochs, lr=lr, device=device, n_sites=1)
    info["n_sites_fitted"] = len(rows_h)
    return model, info


def scenario_heads(pooled: np.ndarray, factor: float) -> np.ndarray:
    """Scale each cell's head DECLINE from its own starting head by ``factor``.

    ``factor=1`` is the observed record; 1.5 deepens every drawdown by half again; 0.5
    halves it (a partial-recovery scenario). Heads above the starting level are scaled the
    same way, so a factor never turns a recovery into a decline.

    This is the seam where a validated flow model would replace the multiplier -- see the
    module docstring.
    """
    h0 = pooled[:, :1]
    return h0 + (pooled - h0) * float(factor)


def compaction_field(model: VEPColumn, pooled: np.ndarray, factor: float = 1.0,
                     device=None) -> np.ndarray:
    """Cumulative subsidence per active cell, metres positive-down -> ``(n_active, T)``."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    h = torch.tensor(scenario_heads(pooled, factor), dtype=torch.float32, device=dev)
    h = torch.nan_to_num(h, nan=0.0)
    with torch.no_grad():
        return model.to(dev)(h).cpu().numpy()


def _to_frame(grid, vec: np.ndarray) -> np.ndarray:
    """Scatter an active-cell vector back onto the (ny, nx) mask, NaN outside."""
    out = np.full(grid.mask.shape, np.nan)
    out[grid.mask] = vec
    return out


def build_figure(grid, head_vol: np.ndarray, subs: dict[float, np.ndarray],
                 dates, layers: tuple[str, ...] = DEFAULT_LAYERS):
    """Assemble the Plotly figure: four head surfaces plus a subsiding ground surface."""
    import plotly.graph_objects as go

    xs = (grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx) / 1000.0
    ys = (grid.y0 + (np.arange(grid.ny) + 0.5) * grid.dx) / 1000.0
    factors = sorted(subs)
    base = factors.index(1.0) if 1.0 in factors else 0

    def frame_traces(t: int, factor: float):
        tr = []
        for k, code in enumerate(layers):
            z = _to_frame(grid, head_vol[k, :, t])
            depth = -LAYER_DEPTH_M[code] * DEPTH_EXAG
            # The surface sits at its aquifer's depth; head is carried by surfacecolor, so
            # NaN cells (outside the fan, or a layer with no wells) drop out of the plane.
            tr.append(go.Surface(
                x=xs, y=ys, z=np.where(np.isnan(z), np.nan, depth),
                surfacecolor=z, colorscale="RdYlBu", cmin=np.nanmin(head_vol),
                cmax=np.nanmax(head_vol), showscale=(k == 0),
                colorbar=dict(title="head (m)", x=1.02, len=0.55),
                name=f"aquifer {code} ({LAYER_DEPTH_M[code]:.0f} m)",
                hovertemplate=f"aquifer {code}<br>head %{{surfacecolor:.2f}} m<extra></extra>",
                opacity=0.92,
            ))
        s = _to_frame(grid, subs[factor][:, t])
        tr.append(go.Surface(
            x=xs, y=ys, z=-s * SUBS_EXAG, surfacecolor=s * 100.0,
            colorscale="Inferno_r", cmin=0.0,
            cmax=float(np.nanmax([np.nanmax(v) for v in subs.values()]) * 100.0),
            showscale=True, colorbar=dict(title="subsidence (cm)", x=1.14, len=0.55),
            name="ground surface",
            hovertemplate="subsidence %{surfacecolor:.1f} cm<extra></extra>",
        ))
        return tr

    fig = go.Figure(data=frame_traces(len(dates) - 1, factors[base]))
    fig.frames = [go.Frame(data=frame_traces(t, factors[base]), name=str(t))
                  for t in range(len(dates))]

    steps = [dict(method="animate", label=pd.Timestamp(d).strftime("%Y-%m"),
                  args=[[str(t)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for t, d in enumerate(dates)]
    fig.update_layout(
        title=("Choushui alluvial fan — layered head volume and cumulative subsidence"
               "<br><sub>Scenario axis is head decline, not pumping rate: the "
               "pumping→head map awaits the Stage-3 flow gate.</sub>"),
        scene=dict(
            xaxis_title="easting (km, EPSG:3826)",
            yaxis_title="northing (km)",
            zaxis_title=f"depth (m x{DEPTH_EXAG:g}) / subsidence (m x{SUBS_EXAG:g})",
            aspectmode="manual", aspectratio=dict(x=1.0, y=1.1, z=0.75),
            camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9)),
        ),
        sliders=[dict(active=len(dates) - 1, currentvalue=dict(prefix="month: "),
                      pad=dict(t=48), steps=steps)],
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=1.08,
                          buttons=[
                              dict(label="▶ play", method="animate",
                                   args=[None, dict(frame=dict(duration=110, redraw=True),
                                                    fromcurrent=True)]),
                              dict(label="❚❚ pause", method="animate",
                                   args=[[None], dict(mode="immediate",
                                                      frame=dict(duration=0, redraw=False))]),
                          ])],
        margin=dict(l=0, r=0, t=78, b=0), height=780,
    )
    return fig


def _coarsen(frame: np.ndarray, k: int) -> np.ndarray:
    """Block-mean an (ny, nx) raster by an integer factor, NaN-aware, for rendering."""
    if k <= 1:
        return frame
    ny, nx = frame.shape
    py, px = (-ny) % k, (-nx) % k
    f = np.pad(frame, ((0, py), (0, px)), constant_values=np.nan)
    f = f.reshape(f.shape[0] // k, k, f.shape[1] // k, k)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN blocks outside the fan
        return np.nanmean(np.nanmean(f, axis=3), axis=1)


def build_forward_figure(fw: dict, stride: int = 1, coarsen: int = 1,
                         layers: tuple[str, ...] = DEFAULT_LAYERS, gate: dict | None = None):
    """One figure for a ``twin.forward`` result: a scenario dropdown, a month slider that
    runs through the hindcast into the projection, four head surfaces and the ground.

    ``fw`` is the loaded npz. Frames carry every scenario's traces; the dropdown only
    toggles visibility, so switching policy at any month is instant.
    """
    import plotly.graph_objects as go

    from .grid import FanGrid

    grid = FanGrid(nx=int(fw["nx"]), ny=int(fw["ny"]), dx=float(fw["dx"]),
                   x0=float(fw["x0"]), y0=float(fw["y0"]), mask=fw["mask"])
    dates = pd.DatetimeIndex([pd.Timestamp(str(d)) for d in fw["dates"]])
    origin = int(fw["origin"])
    names = [str(n) for n in fw["scenario_names"]]
    descs = [str(n) for n in fw["scenarios"]]
    heads = fw["heads_mean"]           # (S, L, A, T)
    subs = fw["subs_mean"]             # (S, A, T)
    spread = fw["subs_std"]            # (S, A, T)
    if gate is None:
        try:
            gate = json.loads(str(fw["gate"])).get("gate") or {}
        except Exception:
            gate = {}
    T = heads.shape[-1]
    sel = np.unique(np.r_[np.arange(0, T, max(stride, 1)), origin, T - 1])

    k = max(int(coarsen), 1)
    xs = (grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx) / 1000.0
    ys = (grid.y0 + (np.arange(grid.ny) + 0.5) * grid.dx) / 1000.0
    if k > 1:
        xs = (grid.x0 + (np.arange(int(np.ceil(grid.nx / k))) * k + k / 2) * grid.dx) / 1000.0
        ys = (grid.y0 + (np.arange(int(np.ceil(grid.ny / k))) * k + k / 2) * grid.dx) / 1000.0
    hmin, hmax = float(np.nanmin(heads)), float(np.nanmax(heads))
    smax = float(np.nanmax(subs) * 100.0)

    def traces(t: int, si: int, visible: bool):
        tr = []
        for li, code in enumerate(layers):
            z = _coarsen(_to_frame(grid, heads[si, li, :, t]), k)
            depth = -LAYER_DEPTH_M[code] * DEPTH_EXAG
            tr.append(go.Surface(
                x=xs, y=ys, z=np.where(np.isnan(z), np.nan, depth), surfacecolor=z,
                colorscale="RdYlBu", cmin=hmin, cmax=hmax, showscale=(li == 0),
                colorbar=dict(title="head (m)", x=1.02, len=0.55),
                name=f"{names[si]}: aquifer {code}", visible=visible, opacity=0.92,
                hovertemplate=f"aquifer {code}<br>head %{{surfacecolor:.2f}} m<extra></extra>"))
        sv = _coarsen(_to_frame(grid, subs[si, :, t]), k)
        sd = _coarsen(_to_frame(grid, spread[si, :, t]), k)
        tr.append(go.Surface(
            x=xs, y=ys, z=-sv * SUBS_EXAG, surfacecolor=sv * 100.0, colorscale="Inferno_r",
            cmin=0.0, cmax=smax, showscale=True,
            colorbar=dict(title="subsidence (cm)", x=1.14, len=0.55),
            name=f"{names[si]}: ground", visible=visible, customdata=sd * 100.0,
            hovertemplate="subsidence %{surfacecolor:.1f} cm (±%{customdata:.1f})<extra></extra>"))
        return tr

    n_per = len(layers) + 1
    fig = go.Figure(data=[tr for si in range(len(names))
                          for tr in traces(T - 1, si, si == 0)])
    fig.frames = [go.Frame(data=[tr for si in range(len(names)) for tr in traces(t, si, si == 0)],
                           name=str(t)) for t in sel]
    steps = [dict(method="animate",
                  label=pd.Timestamp(dates[t]).strftime("%Y-%m") + ("" if t <= origin else " ▸"),
                  args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))]) for t in sel]
    buttons = []
    for si in range(len(names)):
        vis = [False] * (n_per * len(names))
        for j in range(n_per):
            vis[si * n_per + j] = True
        buttons.append(dict(label=descs[si], method="restyle", args=[{"visible": vis}]))
    verdict = ("Stage-3 gate " + f"{gate.get('verdict')} (k-fold R² {gate.get('r2_kfold', float('nan')):+.3f} "
               f"vs IDW {gate.get('r2_idw', float('nan')):+.3f})" if gate else
               "Stage-3 gate: no verdict recorded -- pumping→head map unvalidated")
    fig.update_layout(
        title=("Choushui alluvial fan — forward twin: pumping policy → heads → subsidence"
               f"<br><sub>{verdict}. Months after {dates[origin].strftime('%Y-%m')} (▸) are "
               f"projections under climatological forcing; mean of {int(fw['n_members'])} "
               "members, hover shows the spread.</sub>"),
        scene=dict(xaxis_title="easting (km, EPSG:3826)", yaxis_title="northing (km)",
                   zaxis_title=f"depth (m x{DEPTH_EXAG:g}) / subsidence (m x{SUBS_EXAG:g})",
                   aspectmode="manual", aspectratio=dict(x=1.0, y=1.1, z=0.75),
                   camera=dict(eye=dict(x=1.5, y=-1.6, z=0.9))),
        sliders=[dict(active=len(sel) - 1, currentvalue=dict(prefix="month: "),
                      pad=dict(t=48), steps=steps)],
        updatemenus=[
            dict(type="buttons", showactive=False, x=0.02, y=1.08, buttons=[
                dict(label="▶ play", method="animate",
                     args=[None, dict(frame=dict(duration=110, redraw=True), fromcurrent=True)]),
                dict(label="❚❚ pause", method="animate",
                     args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])]),
            dict(type="dropdown", x=0.02, y=1.0, xanchor="left", showactive=True,
                 buttons=buttons)],
        margin=dict(l=0, r=0, t=98, b=0), height=780)
    return fig


def validate_against_leveling(ddir: str, grid, modelled: np.ndarray, dates,
                              min_obs: int = 5, max_rate: float | None = 0.5) -> dict:
    """Score the modelled subsidence field against the WRA leveling benchmarks.

    The viewer would otherwise be a picture with no claim attached. This compares the
    factor-1 (observed-head) field against an *independent* observation network -- leveling
    benchmarks, which played no part in fitting the VEP (that was MLCW magnetic rings) --
    at each benchmark's own survey dates, and reports pooled R² over every site-survey pair.

    Sign convention matches ``leveling.site_subsidence``: positive is sinking, re-zeroed to
    each site's first survey inside the window, so the comparison is like-for-like with the
    model's own re-zeroing at t=0.
    """
    from .leveling import load_panel, site_subsidence, site_xy

    panel = load_panel(ddir)
    obs = site_subsidence(panel, str(dates[0].date()), str(dates[-1].date()),
                          min_obs=min_obs, max_rate=max_rate)
    xy = site_xy(panel)
    cent = grid.centroids()

    pred_pairs, obs_pairs, n_sites = [], [], 0
    for sid, series in obs.items():
        if sid not in xy:
            continue
        d2 = ((cent - np.array(xy[sid])) ** 2).sum(1)
        cell = int(np.argmin(d2))
        if np.sqrt(d2[cell]) > grid.dx:          # benchmark outside the fan mask
            continue
        m = pd.Series(modelled[cell], index=dates)
        aligned = m.reindex(series.index, method="nearest",
                            tolerance=pd.Timedelta("45D"))
        ok = aligned.notna().to_numpy() & np.isfinite(series.to_numpy())
        if ok.sum() < 3:
            continue
        # Re-zero the model at this site's first matched survey so both series measure
        # change from the same instant.
        p = aligned.to_numpy()[ok]
        o = series.to_numpy()[ok]
        pred_pairs.append(p - p[0])
        obs_pairs.append(o - o[0])
        n_sites += 1

    if not pred_pairs:
        return {"n_sites": 0, "r2": float("nan"), "rmse_cm": float("nan")}
    p = np.concatenate(pred_pairs)
    o = np.concatenate(obs_pairs)
    ss_res = float(((o - p) ** 2).sum())
    ss_tot = float(((o - o.mean()) ** 2).sum())
    return {"n_sites": n_sites, "n_pairs": int(o.size),
            "r2": 1.0 - ss_res / max(ss_tot, 1e-12),
            "rmse_cm": float(np.sqrt(((o - p) ** 2).mean()) * 100.0),
            "bias_cm": float((p - o).mean() * 100.0)}


def scenario_table(subs: dict[float, np.ndarray], grid) -> pd.DataFrame:
    """Fan-wide subsidence summary per scenario, in centimetres at the final month."""
    rows = []
    for f in sorted(subs):
        final = subs[f][:, -1] * 100.0
        rows.append({"drawdown_factor": f,
                     "mean_cm": float(np.nanmean(final)),
                     "p95_cm": float(np.nanpercentile(final, 95)),
                     "max_cm": float(np.nanmax(final)),
                     # Thresholds chosen to discriminate: every cell on this fan clears
                     # 10 cm in every scenario, so a 10 cm area column is constant and
                     # tells you nothing. 30/50 cm separate the scenarios.
                     "area_over_30cm_km2": float((final > 30).sum() * (grid.dx / 1000.0) ** 2),
                     "area_over_50cm_km2": float((final > 50).sum() * (grid.dx / 1000.0) ** 2)})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="3D head + subsidence scenario viewer")
    ap.add_argument("--data", default=None, help="data dir (else HYDROMIND_GW_DATA)")
    ap.add_argument("--wells-dir", default="AMP_V2/data/wells")
    ap.add_argument("--stations", default="AMP_V2/data/fan_stations.parquet")
    ap.add_argument("--polygon",
                    default="chou-shui-data/data/Zhuoshui Alluvial Fan/"
                            "Zhuoshui Alluvial Fan.json")
    ap.add_argument("--dx", type=float, default=2000.0,
                    help="render grid spacing (m); coarser than the solver's 1 km because "
                         "this is a visualisation, not a solve")
    ap.add_argument("--epochs", type=int, default=2000, help="VEP fit epochs")
    ap.add_argument("--factors", default="0.5,1.0,1.5,2.0",
                    help="head-decline multipliers to precompute")
    ap.add_argument("--stride", type=int, default=1,
                    help="render every Nth month. Frame count drives the HTML's size far "
                         "more than grid resolution does (132 monthly frames at 1 km is "
                         "~70 MB); --stride 3 gives quarterly frames at a third the size. "
                         "The physics is always computed at full monthly resolution and "
                         "the validation and scenario table always use every month -- "
                         "this subsamples the animation only.")
    ap.add_argument("--out", default="results/twin/explorer3d.html")
    ap.add_argument("--forward-npz", default=None,
                    help="render a twin.forward result (pumping-policy scenarios through "
                         "the calibrated flow model) instead of the head-decline mode")
    ap.add_argument("--coarsen", type=int, default=1,
                    help="forward mode: block-mean the render raster by this factor")
    ap.add_argument("--gate-csv", default=None,
                    help="forward mode: read the Stage-3 verdict for the title from this "
                         "stage3_flow.csv instead of the run's embedded metadata")
    args = ap.parse_args(argv)

    if args.forward_npz:
        fw = np.load(args.forward_npz, allow_pickle=False)
        gate = None
        if args.gate_csv:
            from .forward import _members_from_gate_csv

            gate = _members_from_gate_csv(args.gate_csv)["meta"]["gate"]
        fig = build_forward_figure(fw, stride=args.stride, coarsen=args.coarsen, gate=gate)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out), include_plotlyjs="cdn", auto_play=False)
        print(f"wrote -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB, "
              f"{len(fig.frames)} frames, {len(fw['scenario_names'])} scenarios)")
        return

    cfg = Config(data_dir=Path(args.data)) if args.data else Config()
    ddir = str(cfg.data_dir)
    factors = [float(x) for x in args.factors.split(",") if x.strip()]

    stn = pd.read_parquet(args.stations)
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(args.wells_dir, stn)
    print(f"heads: {len(hf)} wells x {hf.heads.shape[1]} months "
          f"({hf.dates[0].date()} -> {hf.dates[-1].date()})")

    grid = build_grid(args.polygon, dx=args.dx)
    print(f"render grid: {grid.nx}x{grid.ny} @ {grid.dx:.0f} m -> {grid.n_active} active cells")

    head_vol = layer_head_volume(hf, grid)
    pooled = idw_interp(grid.centroids(), hf.xy, hf.heads)

    model, info = fit_shared_vep(ddir, hf, epochs=args.epochs)
    p = {k: float(torch.exp(getattr(model, f"log_{k}")).item()) for k in ("ske", "skv", "tau")}
    print(f"shared VEP fitted on {info['n_sites_fitted']} MLCW sites: "
          f"loss={info['loss']:.3e}  Ske={p['ske']:.3e}  Skv={p['skv']:.3e}  "
          f"tau={p['tau']:.0f} d")

    subs = {f: compaction_field(model, pooled, f) for f in factors}

    base = 1.0 if 1.0 in subs else sorted(subs)[0]
    skill = validate_against_leveling(ddir, grid, subs[base], hf.dates)
    print("\n=== validation vs leveling benchmarks (independent of the VEP fit) ===")
    if skill["n_sites"]:
        print(f"  sites {skill['n_sites']}  pairs {skill['n_pairs']}  "
              f"R2 {skill['r2']:+.3f}  RMSE {skill['rmse_cm']:.1f} cm  "
              f"bias {skill['bias_cm']:+.1f} cm")
    else:
        print("  no leveling benchmark matched a fan cell -- no skill number")

    table = scenario_table(subs, grid)
    print("\n=== scenario summary (final month) ===")
    print(table.round(2).to_string(index=False))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Subsample only what is drawn; skill and scenario numbers above used every month.
    # Always keep the final month, which is where the scenario table is read.
    sel = np.unique(np.r_[np.arange(0, len(hf.dates), max(args.stride, 1)),
                          len(hf.dates) - 1])
    fig = build_figure(grid, head_vol[:, :, sel],
                       {f: v[:, sel] for f, v in subs.items()}, hf.dates[sel])
    fig.write_html(str(out), include_plotlyjs="cdn", auto_play=False)
    table.to_csv(out.with_suffix(".scenarios.csv"), index=False)
    print(f"\nwrote -> {out}\nwrote -> {out.with_suffix('.scenarios.csv')}")


if __name__ == "__main__":
    main()
