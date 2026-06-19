"""Study-area maps for the README: per-well validation KGE and the SpatialPINN head field.

Both maps share a GIS backdrop (alluvial-fan outline + rivers + coastline) loaded from
the lightweight JSON in ``app/geo/`` (TWD97 metres, the same CRS as station tm_x/tm_y and
the PINN's physical coordinates), so everything is drawn in native metres with no
reprojection and an equal aspect.

CLI::

    python -m hydrophysics.maps --kge
    python -m hydrophysics.maps --headfield --day 3287

Run from the repo root with ``HYDROMIND_GW_DATA`` pointing at the real data directory for
the head-field map (the KGE map only needs the committed CSV + geo JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Repo layout: this file is hydrophysics/maps.py, so the root is its parent's parent.
_ROOT = Path(__file__).resolve().parent.parent
_GEO = _ROOT / "app" / "geo"
_STATIONS = _ROOT / "app" / "stations.csv"
_GRAYBOX_CSV = _ROOT / "results" / "phase0" / "per_well_graybox.csv"
_KGE_PNG = _ROOT / "results" / "phase0" / "spatial_kge_graybox.png"
_HEADFIELD_PNG = _ROOT / "results" / "pinn" / "head_field_map.png"


# --- backdrop -------------------------------------------------------------------


def _load_geo() -> dict:
    """Load the fan, rivers and sea JSON. Returns dict with 'fan_ring', 'rivers', 'sea'."""
    fan = json.loads((_GEO / "fan.json").read_text())
    rivers = json.loads((_GEO / "rivers.json").read_text())
    sea = json.loads((_GEO / "sea.json").read_text())
    return {
        "fan_ring": np.asarray(fan["rings"][0], dtype=float),
        "rivers": [np.asarray(line, dtype=float) for line in rivers["lines"]],
        "sea": [np.asarray(ring, dtype=float) for ring in sea["rings"]],
    }


def draw_backdrop(ax, geo: dict | None = None, *, fan_fill: bool = False):
    """Draw the study-area backdrop (sea fill, rivers, fan outline) onto ``ax``.

    Shared by both maps. ``fan_fill`` lightly tints the fan (used under the KGE scatter,
    skipped under the head-field heatmap which already fills the fan). Returns ``geo``.
    """
    geo = geo or _load_geo()

    # Coastline / sea as a light-blue fill underneath everything.
    for ring in geo["sea"]:
        ax.fill(ring[:, 0], ring[:, 1], facecolor="#cfe8f3", edgecolor="none",
                zorder=0)

    if fan_fill:
        ring = geo["fan_ring"]
        ax.fill(ring[:, 0], ring[:, 1], facecolor="#f3efe6", edgecolor="none",
                zorder=0.5)

    # Rivers as thin blue lines.
    for line in geo["rivers"]:
        ax.plot(line[:, 0], line[:, 1], color="#3a7bd5", linewidth=0.5, alpha=0.7,
                zorder=1)

    # Fan outline on top.
    ring = geo["fan_ring"]
    ax.plot(ring[:, 0], ring[:, 1], color="#5a4a30", linewidth=1.4, zorder=2)

    ax.set_aspect("equal")
    ax.set_xlabel("TM_X97 (m)")
    ax.set_ylabel("TM_Y97 (m)")
    return geo


def _fan_bbox(geo: dict, pad: float = 0.02) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of the fan ring, padded by ``pad`` of each span."""
    ring = geo["fan_ring"]
    xmin, ymin = ring.min(axis=0)
    xmax, ymax = ring.max(axis=0)
    dx, dy = (xmax - xmin) * pad, (ymax - ymin) * pad
    return xmin - dx, xmax + dx, ymin - dy, ymax + dy


# --- KGE map --------------------------------------------------------------------


def render_kge_map(out: Path = _KGE_PNG):
    """Per-well validation KGE at true station coordinates over the backdrop."""
    import matplotlib.pyplot as plt

    gb = pd.read_csv(_GRAYBOX_CSV).set_index("st_id")
    stations = pd.read_csv(_STATIONS).set_index("st_id")
    df = stations.join(gb[["kge"]], how="inner").dropna(subset=["kge", "tm_x", "tm_y"])

    geo = _load_geo()
    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    draw_backdrop(ax, geo, fan_fill=True)

    sc = ax.scatter(
        df["tm_x"], df["tm_y"], c=df["kge"], cmap="viridis",
        s=70, edgecolor="k", linewidth=0.5, vmin=0.0, vmax=1.0, zorder=5,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Validation KGE")
    ax.set_title(f"Per-well validation KGE (gray-box ODE), n={len(df)} wells\n"
                 "Zhuoshui alluvial fan")
    xmin, xmax, ymin, ymax = _fan_bbox(geo, pad=0.05)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  ({len(df)} wells, KGE range "
          f"{df['kge'].min():.3f}..{df['kge'].max():.3f})")
    return out


# --- head-field map -------------------------------------------------------------


def render_headfield_map(day: int = 3287, out: Path = _HEADFIELD_PNG,
                         nx: int = 150, ny: int = 150, epochs: int = 1500,
                         device: str | None = None):
    """Train the SpatialPINN on the real data and render its head field on ``day``."""
    import matplotlib.pyplot as plt
    from matplotlib.path import Path as MplPath

    from .config import default_config
    from .data import load_dataset
    from .models.pinn_field import SpatialPINN

    data = load_dataset(default_config())
    if not (0 <= day < data.n_days):
        raise ValueError(f"day {day} out of range [0, {data.n_days})")
    date = pd.Timestamp(data.dates[day]).date()

    geo = _load_geo()
    xmin, xmax, ymin, ymax = _fan_bbox(geo, pad=0.02)
    gx = np.linspace(xmin, xmax, nx)
    gy = np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([GX.ravel(), GY.ravel()])

    # Mask grid cells outside the fan polygon.
    inside = MplPath(geo["fan_ring"]).contains_points(pts)

    print(f"training SpatialPINN ({epochs} epochs) on {data.n_wells} wells ...")
    model = SpatialPINN(hidden=64, epochs=epochs, device=device, seed=0)
    model.fit(data)

    head = model.head_field(pts, day)
    head = np.where(inside, head, np.nan).reshape(GY.shape)
    finite = head[np.isfinite(head)]
    print(f"head field range (masked to fan): {finite.min():.2f}..{finite.max():.2f} m")

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    draw_backdrop(ax, geo, fan_fill=False)
    mesh = ax.pcolormesh(GX, GY, head, cmap="terrain", shading="auto", zorder=1.5)
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.7)
    cbar.set_label("Head (m)")

    stations = pd.read_csv(_STATIONS).set_index("st_id")
    ax.scatter(stations["tm_x"], stations["tm_y"], s=18, facecolor="none",
               edgecolor="k", linewidth=0.6, zorder=5, label="monitoring wells")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    ax.set_title(f"SpatialPINN continuous head field — {date} (day {day})\n"
                 "Zhuoshui alluvial fan")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}  (day {day} = {date})")
    return out, date


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Render README study-area maps.")
    ap.add_argument("--kge", action="store_true", help="render the per-well KGE map")
    ap.add_argument("--headfield", action="store_true",
                    help="train the SpatialPINN and render its head field")
    ap.add_argument("--day", type=int, default=3287, help="day index for the head field")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    if not (args.kge or args.headfield):
        ap.error("pass --kge and/or --headfield")
    if args.kge:
        render_kge_map()
    if args.headfield:
        render_headfield_map(day=args.day, epochs=args.epochs, device=args.device)


if __name__ == "__main__":  # pragma: no cover
    main()
