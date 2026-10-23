"""Frozen downscaling of WP X1: serialise the primary model and apply it to TPC months.

The primary model (``ridge_wx``, chosen by ``gate_spec``'s rule) is fitted once per county on
the recorded census months 2016-01..2026-06 and written to ``parameters.json``. ``apply``
re-computes the downscaled value of any month from the parameters and the inputs alone, so a
later TPC month can be added without refitting (the procedure a registered test freezes).

Missing-input rules (fixed here, before use):
* own-county TPC 農林漁牧 at t, t-1 or t-2 missing -> the month is not downscaled (flag 8);
* any other feature missing (other county, other sectors, national total, rain, ET0) -> its
  anomaly is set to 0, i.e. the feature takes its month-of-year training mean; the month is
  still downscaled and the row's ``n_missing_features`` says how many were set.
Disc values: county value x the disc's month-of-year share of the county (recorded months
2012-01..2026-06). sd: county multiplicative sd (sdlog, out-of-fold RMS of log errors, all
CV folds); disc relative sd per well (out-of-fold, all folds; pooled county value if a well
lacks one).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import model as M  # noqa: E402

PRIMARY = "ridge_wx"
REQUIRED = ("ag", "ag_l1", "ag_l2")


def to_params(f) -> dict:
    return {"model": f.name, "county": f.county, "cols": list(f.cols),
            "mu_log": [float(v) for v in f.mu_log.to_numpy()],
            "fmeans": {c: [float(v) for v in f.fmeans[c].to_numpy()] for c in f.cols},
            "sd": {c: float(f.sd[c]) for c in f.cols},
            "lo": [float(v) for v in f.lo], "hi": [float(v) for v in f.hi],
            "beta": [float(v) for v in f.beta], "b0": float(f.b0),
            "smear": float(f.smear), "alpha": M.ALPHA,
            "train_resid_sd_log": float(f.train_resid_sd)}


def apply(params: dict, tpc: pd.DataFrame, nat: pd.Series, wx: pd.DataFrame,
          months: pd.DatetimeIndex) -> pd.DataFrame:
    """Downscaled county values (GWh) for ``months``; columns value, n_missing_features."""
    f = M.features(tpc, nat, wx, params["county"], weather=True).reindex(months)
    cols = params["cols"]
    X = f[cols]
    moy = months.month.to_numpy() - 1
    means = np.column_stack([np.asarray(params["fmeans"][c])[moy] for c in cols])
    A = X.to_numpy() - means
    req = np.column_stack([X[c].notna().to_numpy() for c in REQUIRED]).all(axis=1)
    miss = ~np.isfinite(A)
    A = np.where(miss, 0.0, A)
    sd = np.asarray([params["sd"][c] for c in cols])
    Z = np.clip(A / sd, np.asarray(params["lo"]), np.asarray(params["hi"]))
    a = params["b0"] + Z @ np.asarray(params["beta"])
    v = np.exp(np.asarray(params["mu_log"])[moy] + a + params["smear"])
    v = np.where(req, v, np.nan)
    return pd.DataFrame({"value": v, "n_missing_features": miss.sum(axis=1)}, index=months)
