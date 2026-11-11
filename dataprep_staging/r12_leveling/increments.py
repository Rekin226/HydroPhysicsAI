"""Annual May-to-May leveling increments and head features at benchmarks (round-12 exploratory).

Increment d[i, y] = elevation(survey of year y-1) - elevation(survey of year y), metres,
positive = sinking, using each year's spring survey (April-August). Head features come from
observed monitoring wells only (L1_clean v3, flags 0-1), expressed as anomalies against each
well's own 2012-.. mean so that they interpolate between wells, then inverse-distance
weighted to the benchmark per aquifer group.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from dataprep_staging.r12_leveling.paths import DDIR
from dataprep_staging.r12_leveling.paths import HEADS as SHARED
from hydrophysics.twin.leveling import load_panel


def spring_elev(panel: pd.DataFrame, y0: int = 1997, y1: int = 2021) -> pd.DataFrame:
    """sites x years table of the spring-survey elevation (NaN when not surveyed)."""
    p = panel[(panel.datetime.dt.month >= 4) & (panel.datetime.dt.month <= 8)].copy()
    p["year"] = p.datetime.dt.year
    p = p[(p.year >= y0) & (p.year <= y1)].dropna(subset=["elev_m"])
    return p.groupby(["sid", "year"]).elev_m.first().unstack("year")


def increments(elev: pd.DataFrame, max_abs: float = 0.5) -> pd.DataFrame:
    """sites x years table of d[i, y] (m/yr); steps larger than ``max_abs`` are artefacts."""
    d = -(elev.diff(axis=1))
    return d.where(d.abs() <= max_abs)


def hydro_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """June y-1 .. May y belongs to hydrological year y (matches the May survey)."""
    return np.where(dates.month >= 6, dates.year + 1, dates.year)


def well_year_stats(first_year: int = 2013, last_year: int = 2026) -> tuple[dict, pd.DataFrame]:
    """Per-well annual statistics of the head anomaly. -> ({name: wells x years}, meta)."""
    H = pd.read_parquet(SHARED + "heads_monthly.parquet")
    H = H[H.flag <= 1]
    W = H.pivot(index="date", columns="sid", values="value").sort_index()
    W = W.loc["2012-06-01":]
    hy = hydro_year(W.index)
    keep = W.columns[(W.loc[:"2022-12-31"].notna().sum() >= 96)]
    W = W[keep]
    base = W.loc[:"2022-05-31"].mean()                      # fixed reference level
    A = W - base
    g = A.groupby(hy)
    n = g.count()
    mean = g.mean().where(n >= 9)
    amin = g.min().where(n >= 9)
    amax = g.max().where(n >= 9)
    yrs = [y for y in mean.index if first_year <= y <= last_year]
    mean, amin, amax = mean.loc[yrs], amin.loc[yrs], amax.loc[yrs]
    # depth of a new record low below the lowest earlier annual minimum
    prev_low = amin.cummin().shift(1)
    exceed = (prev_low - amin).clip(lower=0.0)
    stats = {"mean": mean.T, "min": amin.T, "dmean": mean.diff().T,
             "amp": (amax - amin).T, "exceed": exceed.T}
    meta = pd.read_parquet(SHARED + "station_meta_clean.parquet").set_index("sid")
    meta = meta.loc[keep, ["x_twd97", "y_twd97", "layer_code", "CountyName", "WellDepth"]]
    return stats, meta


def to_sites(stat: pd.DataFrame, meta: pd.DataFrame, xy: np.ndarray, layers: tuple,
             k: int = 6, power: float = 2.0) -> np.ndarray:
    """IDW of one wells x years table to site coordinates -> (n_sites, n_years)."""
    lay = pd.to_numeric(meta.layer_code, errors="coerce")
    sel = meta.index[lay.isin(layers) & meta.x_twd97.notna()]
    wxy = meta.loc[sel, ["x_twd97", "y_twd97"]].to_numpy("float64")
    V = stat.loc[sel].to_numpy("float64")
    out = np.full((len(xy), V.shape[1]), np.nan)
    for j in range(V.shape[1]):
        ok = np.isfinite(V[:, j])
        if ok.sum() < 3:
            continue
        tree = cKDTree(wxy[ok])
        d, idx = tree.query(xy, k=min(k, int(ok.sum())))
        w = 1.0 / np.maximum(d, 500.0) ** power
        out[:, j] = (w * V[ok, j][idx]).sum(1) / w.sum(1)
    return out


def site_table():
    """-> (d sites x years, xy (n,2), sids, county mask is_changhua)."""
    panel = load_panel(DDIR)
    elev = spring_elev(panel)
    d = increments(elev)
    first = panel.groupby("sid").first()
    xy = first.loc[d.index, ["x", "y"]].to_numpy("float64")
    return d, xy, list(d.index), elev
