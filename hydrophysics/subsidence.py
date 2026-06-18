"""Subsidence science for the Choushui explorer: IDW heads, MLCW compaction, Sk fit.

Pure NumPy / pandas (+ pyarrow for parquet). No plotly/geopandas here so the science is
importable and testable on its own. See
docs/superpowers/specs/2026-06-19-choushui-head-subsidence-explorer-design.md.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from .data import GWData


def load_mlcw_stations(path) -> pd.DataFrame:
    """Read the 14 MLCW coordinates -> DataFrame[sub_id, x, y, lon, lat] (EPSG:3826)."""
    df = pd.read_csv(path)
    return df[["sub_id", "x", "y", "lon", "lat"]]


def well_xy(data: GWData) -> np.ndarray:
    """(W, 2) well coordinates in EPSG:3826 meters."""
    x = data.attrs["tm_x"].astype(float).fillna(0.0).to_numpy()
    y = data.attrs["tm_y"].astype(float).fillna(0.0).to_numpy()
    return np.stack([x, y], axis=-1)


def idw_interp(points_xy: np.ndarray, src_xy: np.ndarray, values: np.ndarray,
               power: float = 2.0, eps: float = 1e-6) -> np.ndarray:
    """Inverse-distance interpolate ``values`` (S sources x T) to N query points -> (N, T).

    NaNs in ``values`` are down-weighted per timestep (a source contributes only where it
    has a finite value), so missing observations never poison a cell.
    """
    P = np.asarray(points_xy, dtype="float64")              # (N, 2)
    S = np.asarray(src_xy, dtype="float64")                 # (Sn, 2)
    V = np.asarray(values, dtype="float64")                 # (Sn, T)
    d2 = ((P[:, None, :] - S[None, :, :]) ** 2).sum(-1)      # (N, Sn)
    w = 1.0 / (d2 ** (power / 2.0) + eps)                    # (N, Sn)
    finite = np.isfinite(V).astype("float64")               # (Sn, T)
    num = w @ np.nan_to_num(V)                               # (N, T)
    den = w @ finite                                        # (N, T)
    return num / np.maximum(den, 1e-9)


def monthly_heads(data: GWData) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Resample the observed heads to month-end means -> ((W, Tm) array, Tm dates)."""
    df = pd.DataFrame(data.target.T, index=pd.DatetimeIndex(data.dates))  # (T, W)
    m = df.resample("ME").mean()
    return m.to_numpy().T, m.index


def cumulative_drawdown(h_monthly: np.ndarray) -> np.ndarray:
    """Inelastic cumulative drawdown along the last axis: h[...,0] - runningmin(h) (>=0)."""
    h = np.asarray(h_monthly, dtype="float64")
    runmin = np.minimum.accumulate(h, axis=-1)
    return h[..., :1] - runmin


def _decode_mlcw_name(filename: str) -> str:
    """Decode a percent-hex MLCW filename (UTF-8 bytes as _XX) back to the Chinese name."""
    core = os.path.basename(filename).split("ls-wra-mlcw-obs__")[-1].replace(".parquet", "")
    hexes = re.findall(r"_([0-9A-Fa-f]{2})", core)
    try:
        return bytes(int(h, 16) for h in hexes).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return core


def mlcw_compaction(data_dir: str) -> dict[str, pd.Series]:
    """Per-site total compaction (m), re-zeroed to the first observation, monthly.

    Each MLCW parquet holds magnetic-ring positions ``NO1..NO31`` (m) at increasing depth.
    Total compaction = shortening of the monitored interval = (separation between the
    shallowest and deepest ring at t0) - (separation at t). Positive = subsidence.
    """
    pattern = os.path.join(data_dir, "ls_cache", "clean", "ls-wra-mlcw-obs__*.parquet")
    out: dict[str, pd.Series] = {}
    for f in sorted(glob.glob(pattern)):
        name = _decode_mlcw_name(f)
        df = pd.read_parquet(f).sort_index()
        order = df.mean().sort_values().index            # shallow (small) -> deep (large)
        shallow, deep = df[order[0]], df[order[-1]]
        sep = deep - shallow                             # interval thickness over time
        comp = (sep.iloc[0] - sep).rename("compaction_m")  # >0 as the interval compacts
        out[name] = comp
    return out
