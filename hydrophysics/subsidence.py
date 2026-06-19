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
        fvi = sep.first_valid_index()                    # re-zero to first valid sep (NaN-safe)
        base = sep.loc[fvi] if fvi is not None else 0.0
        comp = (base - sep).rename("compaction_m")        # >0 as the interval compacts
        out[name] = comp
    return out


def calibrate_sk_from_pairs(per_site: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Least-squares-through-origin slope Sk over pooled (drawdown, compaction) pairs.

    per_site: {name -> (D, C)} aligned monthly arrays. Returns {sk, r2, per_site, D, C}.
    """
    if not per_site:
        return {"sk": 0.0, "r2": float("nan"), "per_site": {},
                "D": np.array([]), "C": np.array([])}
    Ds = [np.asarray(D, float) for D, _ in per_site.values()]
    Cs = [np.asarray(C, float) for _, C in per_site.values()]
    D = np.concatenate(Ds)
    C = np.concatenate(Cs)
    sk = float(D @ C / (D @ D + 1e-12))
    pred = sk * D
    ss_res = float(((C - pred) ** 2).sum())
    ss_tot = float(((C - C.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"sk": sk, "r2": r2, "per_site": per_site, "D": D, "C": C}


def calibrate_sk(data: GWData, stations: pd.DataFrame,
                 compaction: dict[str, pd.Series]) -> dict:
    """Pair each MLCW site's monthly compaction with the IDW head drawdown at its (x,y),
    then fit one Sk. Returns the calibrate_sk_from_pairs result plus per-site monthly index.
    """
    H, dates = monthly_heads(data)                       # (W, Tm)
    wxy = well_xy(data)
    per_site: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for _, row in stations.iterrows():
        name = row["sub_id"]
        if name not in compaction:
            continue
        h_site = idw_interp(np.array([[row["x"], row["y"]]]), wxy, H)[0]   # (Tm,)
        hser = pd.Series(h_site, index=dates)
        comp = compaction[name].resample("ME").mean()        # align to month-end heads
        idx = hser.index.intersection(comp.index)
        if len(idx) < 6:
            continue
        hh = hser.reindex(idx).to_numpy()
        cc = comp.reindex(idx).to_numpy()
        mask = np.isfinite(hh) & np.isfinite(cc)             # drop months missing either
        if mask.sum() < 6:
            continue
        hh, cc = hh[mask], cc[mask]
        D = hh[:1] - np.minimum.accumulate(hh)               # cumulative drawdown
        C = cc - cc[0]                                       # re-zero compaction
        per_site[name] = (D, C)
    return calibrate_sk_from_pairs(per_site)


def _per_site_sk(per_site: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[float, float]]:
    """Per-site (Sk_i, weight=ΣD²). Sk_i is the through-origin slope of C on D."""
    out = {}
    for name, (D, C) in per_site.items():
        D = np.asarray(D, float)
        C = np.asarray(C, float)
        denom = float(D @ D)
        out[name] = ((D @ C) / (denom + 1e-12), denom)
    return out


def fit_sk_regression(per_site: dict[str, tuple[np.ndarray, np.ndarray]],
                      dist: dict[str, float]) -> dict:
    """Weighted least-squares fit of log(Sk_i) on distance-to-coast.

    Sk(dc) = exp(b0 + b1*dc). Only sites with Sk_i > 0 enter the log fit; each is weighted
    by ΣD² (how well its Sk_i is determined). The predictor is a function of distance ONLY
    (no compaction-derived feature) -> leakage-safe. Returns {b0, b1, r2_insample,
    predict_sk, sk_per_site}.
    """
    sk_w = _per_site_sk(per_site)
    use = [n for n in sk_w if n in dist and sk_w[n][0] > 0]
    dc = np.array([dist[n] for n in use], float)
    y = np.log(np.array([sk_w[n][0] for n in use], float))
    w = np.array([sk_w[n][1] for n in use], float)
    w = w / w.sum()
    X = np.stack([np.ones_like(dc), dc], axis=1)               # (m, 2)
    WX = X * w[:, None]
    beta = np.linalg.solve(X.T @ WX, X.T @ (w * y))            # weighted normal equations
    b0, b1 = float(beta[0]), float(beta[1])

    def predict_sk(d):
        d = np.asarray(d, float)
        return np.exp(b0 + b1 * d)

    sk_true = np.array([sk_w[n][0] for n in use], float)
    pred = predict_sk(dc)
    ss_res = float(((sk_true - pred) ** 2).sum())
    ss_tot = float(((sk_true - sk_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return {"b0": b0, "b1": b1, "r2_insample": r2, "predict_sk": predict_sk,
            "sk_per_site": {n: sk_w[n][0] for n in sk_w}}


def _distances_to_geom(stations: pd.DataFrame, geom) -> dict[str, float]:
    """Distance from each station (x, y) to a shapely geometry. Pure shapely (testable)."""
    from shapely.geometry import Point

    return {row["sub_id"]: float(Point(row["x"], row["y"]).distance(geom))
            for _, row in stations.iterrows()}


def site_distance_to_coast(stations: pd.DataFrame, coast_shp) -> dict[str, float]:
    """Distance (m, EPSG:3826) from each MLCW site to the coastline polygon/line."""
    import geopandas as gpd

    geom = gpd.read_file(coast_shp).geometry.union_all()
    return _distances_to_geom(stations, geom)
