"""Leveling benchmarks -> per-site cumulative subsidence.

The WRA leveling network (`ls-wra-lsp-obs`) surveys benchmark orthometric elevation roughly
annually. Cumulative subsidence at a site is its elevation drop relative to the first survey
inside the analysis window, so positive values mean sinking -- the same sign convention as
``subsidence.mlcw_compaction``.
"""

from __future__ import annotations

import os

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
