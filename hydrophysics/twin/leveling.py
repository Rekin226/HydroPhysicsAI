"""Leveling benchmarks -> per-site cumulative subsidence.

The WRA leveling network (`ls-wra-lsp-obs`) surveys benchmark orthometric elevation roughly
annually. Cumulative subsidence at a site is its elevation drop relative to the first survey
inside the analysis window, so positive values mean sinking -- the same sign convention as
``subsidence.mlcw_compaction``.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PANEL = os.path.join("ls_cache", "ls-wra-lsp-obs__choushui_panel.parquet")


def load_panel(data_dir: str) -> pd.DataFrame:
    """Read the cached benchmark panel -> DataFrame[sid, datetime, elev_m, x, y]."""
    df = pd.read_parquet(os.path.join(data_dir, PANEL))
    out = df[["sid", "datetime", "elev_m", "x_3826", "y_3826"]].copy()
    out = out.rename(columns={"x_3826": "x", "y_3826": "y"})
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values(["sid", "datetime"]).reset_index(drop=True)


def site_subsidence(panel: pd.DataFrame, t0: str, t1: str,
                    min_obs: int = 5) -> dict[str, pd.Series]:
    """{sid -> cumulative subsidence (m, positive = sinking) re-zeroed to the first survey}.

    Sites with fewer than ``min_obs`` surveys inside [t0, t1) are dropped.
    """
    w = panel[(panel.datetime >= pd.Timestamp(t0)) & (panel.datetime < pd.Timestamp(t1))]
    out: dict[str, pd.Series] = {}
    for sid, g in w.groupby("sid"):
        g = g.sort_values("datetime")
        if len(g) < min_obs:
            continue
        s = pd.Series(g.elev_m.to_numpy(), index=pd.DatetimeIndex(g.datetime))
        out[sid] = (s.iloc[0] - s).rename("subsidence_m")
    return out


def site_xy(panel: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """{sid -> (x, y)} in EPSG:3826 metres, taken from the first row of each site."""
    first = panel.groupby("sid").first()
    return {sid: (float(r.x), float(r.y)) for sid, r in first.iterrows()}


def _site_rate(s: pd.Series) -> float:
    """Least-squares subsidence rate (m/yr) of one site's cumulative series."""
    t = np.array([(i - s.index[0]).days / 365.25 for i in s.index], dtype="float64")
    if t.size < 2 or t.std() == 0:
        return 0.0
    return float(np.polyfit(t, s.to_numpy(dtype="float64"), 1)[0])


def remove_tectonic(sub: dict[str, pd.Series], xy: dict[str, tuple[float, float]],
                    mode: str = "planar") -> tuple[dict[str, pd.Series], dict]:
    """Subtract a global planar vertical-velocity field v(x,y) = a + b*x + c*y.

    Taiwan's tectonic vertical field is spatially smooth (Ching et al. 2011), so three
    global parameters over hundreds of sites absorb a regional tilt without absorbing
    site-specific compaction. ``mode="none"`` returns the input unchanged.
    """
    if mode == "none":
        return dict(sub), {"a": 0.0, "b": 0.0, "c": 0.0, "var_removed": 0.0}
    if mode != "planar":
        raise ValueError(f"unknown tectonic mode: {mode!r}")

    names = [n for n in sub if n in xy]
    rates = np.array([_site_rate(sub[n]) for n in names], dtype="float64")
    X = np.array([[1.0, xy[n][0], xy[n][1]] for n in names], dtype="float64")
    Xc = X.copy()
    Xc[:, 1] -= Xc[:, 1].mean()          # centre the coordinates for conditioning
    Xc[:, 2] -= Xc[:, 2].mean()
    beta, *_ = np.linalg.lstsq(Xc, rates, rcond=None)
    fitted = Xc @ beta
    denom = float(((rates - rates.mean()) ** 2).sum())
    # mean-centre the fit: the intercept carries the mean rate, not the tilt, so leaving it
    # in makes var_removed report ~1.0 whatever the plane does.
    fitted_c = fitted - fitted.mean()
    var_removed = float((fitted_c ** 2).sum() / denom) if denom > 0 else 0.0

    # Subtract only the TILT, never the intercept: the intercept is the basin-mean vertical
    # rate, which on this fan is ~1.7 cm/yr of groundwater compaction, not tectonics.
    # Removing it would drive most sites negative and make a Sk*D >= 0 model unfittable.
    out: dict[str, pd.Series] = {}
    for n, v in zip(names, fitted_c, strict=True):
        s = sub[n]
        t = np.array([(i - s.index[0]).days / 365.25 for i in s.index], dtype="float64")
        out[n] = s - v * t
    return out, {"a": float(beta[0]), "b": float(beta[1]), "c": float(beta[2]),
                 "var_removed": min(max(var_removed, 0.0), 1.0)}
