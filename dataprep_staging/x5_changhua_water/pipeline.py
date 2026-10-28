"""The declared X5 timing pipeline, shared by the synthetic check, the fit and the scorer.

calendar (mean monthly share of the reference over the calibration years) x exp(sum_k b_k z_k),
z_k = month-of-year anomaly of indicator k over the calibration years / pooled anomaly sd
(missing = 0); b fitted by least squares on the reference's monthly shares in the
calibration years; the index is benchmarked to the annual totals by proportional
first-difference Denton (primary) or Chow-Lin (declared alternative), one run per block of
years (2012-2022 and 2023-2025 never share a run, so no post-origin value moves a
pre-origin month).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_changhua_water import x5const as K  # noqa: E402
from x5_changhua_water import x5lib as L  # noqa: E402

PRE_BLOCK = list(range(2012, 2023))
POST_BLOCK = list(range(2023, 2026))


def fit(ref_calib: pd.Series, z: pd.DataFrame, keys) -> dict:
    """Calendar and b from the reference's calibration-year months only."""
    years = sorted(set(ref_calib.index.year))
    if not set(years) <= set(K.CALIB_YEARS):
        raise ValueError(f"reference holds non-calibration years {years}")
    sh = L.ref_shares(ref_calib)
    cal = L.calendar(sh, years)
    b = L.fit_b(cal, z, sh, keys, years)
    return {"calendar": cal, "b": b, "keys": list(keys)}


def loyo_share_sd(ref_calib: pd.Series, z: pd.DataFrame, keys) -> np.ndarray:
    """Per calendar month, RMS of leave-one-year-out share residuals in the calibration years."""
    years = sorted(set(ref_calib.index.year))
    res = []
    for y in years:
        keep = [v for v in years if v != y]
        sub = ref_calib[np.isin(ref_calib.index.year, keep)]
        f = fit(sub, z, keys)
        pr = L.predict_shares(f["calendar"], z, f["b"], [y])
        tg = L.ref_shares(ref_calib[ref_calib.index.year == y])
        res.append(pd.Series((pr.reindex(tg.index) - tg).to_numpy(), index=tg.index))
    r = pd.concat(res)
    return np.sqrt((r ** 2).groupby(r.index.month).mean().reindex(range(1, 13))
                   .fillna(0.0).to_numpy())


def monthly(f: dict, z: pd.DataFrame, annual: pd.Series, years, estimator: str = "denton",
            b_override: dict | None = None) -> tuple[pd.Series, dict]:
    """Benchmarked monthly series over ``years`` (one block)."""
    b = f["b"] if b_override is None else b_override
    p = L.index_series(f["calendar"], z, b, years)
    if estimator == "denton":
        return L.denton_pfd(p, annual), {}
    if estimator == "chow_lin":
        return L.chow_lin(p, annual)
    raise ValueError(estimator)


def blocks(f, z, annual, years, estimator="denton", b_override=None) -> pd.Series:
    out = []
    for blk in (PRE_BLOCK, POST_BLOCK):
        ys = [y for y in years if y in blk]
        if ys:
            out.append(monthly(f, z, annual, ys, estimator, b_override)[0])
    return pd.concat(out).sort_index()


def zero_b(keys) -> dict:
    return {k: 0.0 for k in keys}
