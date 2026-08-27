"""Layer-resolved monthly head field from the wisenvr fan-well network.

The project's original head field comes from `chou-shui-data`'s curated 61 wells. That
selection was inherited from the gray-box study, which needed every well to have an
*upstream partner* for its ODE -- a constraint irrelevant to subsidence, which only needs
head at a location. The provided raw file actually holds 174 wells, and the wisenvr API
exposes 344 on the Choushui fan (groundwater zone 50), each carrying a
``GroundwaterLayerCode`` assigning it to one of the four aquifers.

This module rebuilds the head field from those API wells so that
  (a) head-field *density* stops being a confound in the Stage-1 result, and
  (b) heads become layer-resolved, which is what the four-layer flow solver needs.

QC per well: robust despike at median +/- 15*MAD, coverage and max-gap filters, then a
month-end mean. Nothing is gap-filled -- months with no observation stay NaN and the IDW
down-weights them per timestep, the convention ``subsidence.idw_interp`` already uses.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAD_K = 15.0
DEFAULT_LAYERS = ("1", "2", "3", "4")


@dataclass
class HeadField:
    """Monthly heads for a set of wells, with coordinates and aquifer layer."""

    heads: np.ndarray            # (W, T) month-end mean head, NaN where unobserved
    dates: pd.DatetimeIndex
    xy: np.ndarray               # (W, 2) EPSG:3826 metres
    layers: np.ndarray           # (W,) aquifer code as str
    sids: list[str]

    def subset(self, layer: str | None) -> HeadField:
        """Restrict to one aquifer layer; ``None`` keeps every well."""
        if layer is None:
            return self
        m = self.layers == layer
        return HeadField(self.heads[m], self.dates, self.xy[m], self.layers[m],
                         [s for s, k in zip(self.sids, m, strict=True) if k])

    def __len__(self) -> int:
        return self.heads.shape[0]


def _despike(s: pd.Series, k: float = MAD_K) -> pd.Series:
    """Drop wild outliers and sentinel values with a median/MAD screen."""
    med = s.median()
    mad = (s - med).abs().median()
    scale = mad if mad > 0 else (s.std() or 1.0)
    return s[(s - med).abs() <= k * scale]


def _station_xy(row) -> tuple[float, float] | None:
    """Parse ``LocationByTWD97`` ('x y') to metres, rejecting implausible values."""
    try:
        parts = str(row["LocationByTWD97"]).strip().split()
        x, y = float(parts[0]), float(parts[1])
    except (ValueError, IndexError, TypeError):
        return None
    if not (140000 <= x <= 240000 and 2580000 <= y <= 2700000):
        return None
    return x, y


def build_head_field(wells_dir: str, stations: pd.DataFrame,
                     t0: str = "2012-01-01", t1: str = "2023-01-01",
                     min_coverage: float = 0.80, max_gap_days: float = 180.0,
                     layers: tuple[str, ...] = DEFAULT_LAYERS) -> HeadField:
    """Assemble a QC'd monthly head field from cached per-well parquet files."""
    T0, T1 = pd.Timestamp(t0), pd.Timestamp(t1)
    n_hours = int((T1 - T0).total_seconds() // 3600)
    meta = {str(r["sid"]): r for _, r in stations.iterrows()}

    keep_h, keep_xy, keep_layer, keep_sid = [], [], [], []
    for f in sorted(glob.glob(os.path.join(wells_dir, "*.parquet"))):
        sid = os.path.basename(f)[:-8]
        row = meta.get(sid)
        if row is None:
            continue
        code = str(row.get("GroundwaterLayerCode", "")).strip()
        if code not in layers:
            continue
        xy = _station_xy(row)
        if xy is None:
            continue
        s = pd.read_parquet(f)["value"].dropna().sort_index()
        s = s[(s.index >= T0) & (s.index < T1)]
        if s.empty:
            continue
        s = _despike(s)
        if len(s) / max(n_hours, 1) < min_coverage:
            continue
        hourly = s.resample("h").mean()
        miss = hourly.isna()
        if miss.any():
            runs = (miss != miss.shift()).cumsum()[miss]
            if float(runs.value_counts().max()) / 24.0 > max_gap_days:
                continue
        keep_h.append(s.resample("ME").mean())
        keep_xy.append(xy)
        keep_layer.append(code)
        keep_sid.append(sid)

    if not keep_h:
        raise ValueError(f"no wells passed QC in {wells_dir}")
    frame = pd.concat(keep_h, axis=1)
    frame.columns = keep_sid
    frame = frame.reindex(pd.date_range(T0, T1, freq="ME"))
    return HeadField(heads=frame.to_numpy(dtype="float64").T,
                     dates=pd.DatetimeIndex(frame.index),
                     xy=np.array(keep_xy, dtype="float64"),
                     layers=np.array(keep_layer, dtype=object),
                     sids=keep_sid)
