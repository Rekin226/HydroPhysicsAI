"""Round-12 exploratory: leveling panel and fair baselines for the subsidence column.

The column's gate scores pooled R2 over (benchmark, survey) pairs of cumulative subsidence
re-zeroed at each benchmark's first survey inside 2012-01..2022-12 (``calibrate_coupled.
leveling_rows`` / ``_r2``). This module rebuilds that target without the model and scores
model-free predictors on it, so the column's 0.65 has a reference.

Exploratory, design-time. Leveling 2012-2021 is a spent unit; nothing here is a test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from hydrophysics.twin.leveling import load_panel, site_subsidence, site_xy

DDIR = "chou-shui-data/data"
T0, T1 = "2012-01-01", "2022-12-31"


def build(ddir: str = DDIR, t0: str = T0, t1: str = T1, min_obs: int = 5,
          max_rate: float = 0.5):
    """-> (panel, target dict sid -> Series cumulative m, xy dict). Same screen as the column."""
    panel = load_panel(ddir)
    sub = site_subsidence(panel, t0, t1, min_obs=min_obs, max_rate=max_rate)
    xy = site_xy(panel)
    sub = {s: v for s, v in sub.items() if s in xy}
    return panel, sub, xy


def years(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.year + (idx.dayofyear - 1) / 365.25


def elev_curve(panel: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """sid -> (decimal years, elevation m) over the full record (1997-2021)."""
    out = {}
    for sid, g in panel.groupby("sid"):
        g = g.dropna(subset=["elev_m"]).sort_values("datetime")
        if len(g) >= 2:
            out[sid] = (years(pd.DatetimeIndex(g.datetime)), g.elev_m.to_numpy("float64"))
    return out


def pooled_scores(pred: dict[str, np.ndarray], sub: dict[str, pd.Series],
                  sids=None) -> dict:
    """Pooled R2 / RMSE over pairs, excluding each site's first (zero) survey as the column
    gate does not (it keeps the zero pair); both conventions are reported."""
    sids = list(pred) if sids is None else [s for s in sids if s in pred]
    o = np.concatenate([sub[s].to_numpy() for s in sids])
    p = np.concatenate([pred[s] for s in sids])
    ok = np.isfinite(p)
    o, p = o[ok], p[ok]
    r2 = 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()
    # final-survey totals, one number per site
    of = np.array([sub[s].to_numpy()[-1] for s in sids])
    pf = np.array([pred[s][-1] for s in sids])
    k = np.isfinite(pf)
    r2f = 1 - ((of[k] - pf[k]) ** 2).sum() / ((of[k] - of[k].mean()) ** 2).sum()
    return {"n_sites": int(len(sids)), "n_pairs": int(ok.sum()), "r2": float(r2),
            "rmse_cm": float(100 * np.sqrt(((o - p) ** 2).mean())),
            "bias_cm": float(100 * (p - o).mean()), "r2_final": float(r2f),
            "rmse_final_cm": float(100 * np.sqrt(((of[k] - pf[k]) ** 2).mean())),
            "coverage": float(ok.mean())}


def spatial_blocks(xy: np.ndarray, size_m: float = 10_000.0, n_folds: int = 5,
                   seed: int = 2027) -> np.ndarray:
    """Fold id per site: square blocks of ``size_m`` assigned to folds at random."""
    bx = np.floor(xy[:, 0] / size_m).astype(int)
    by = np.floor(xy[:, 1] / size_m).astype(int)
    key = bx * 100_000 + by
    uniq = np.unique(key)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(uniq, rng.permutation(len(uniq)) % n_folds, strict=True))
    return np.array([fold_of[k] for k in key])


def random_folds(n: int, n_folds: int = 5, seed: int = 0) -> np.ndarray:
    """The column gate's historical per-benchmark partition."""
    order = np.random.default_rng(seed).permutation(n)
    f = np.empty(n, dtype=int)
    for i, part in enumerate(np.array_split(order, n_folds)):
        f[part] = i
    return f
