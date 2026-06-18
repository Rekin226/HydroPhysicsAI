"""Subsidence science for the Choushui explorer: IDW heads, MLCW compaction, Sk fit.

Pure NumPy / pandas (+ pyarrow for parquet). No plotly/geopandas here so the science is
importable and testable on its own. See
docs/superpowers/specs/2026-06-19-choushui-head-subsidence-explorer-design.md.
"""

from __future__ import annotations

import glob  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401

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
