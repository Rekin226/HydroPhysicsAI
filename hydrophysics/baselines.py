"""Baselines the neural operator must beat.

Evaluation has TWO modes, and mixing them is the classic groundwater benchmarking
trap (daily levels are highly autocorrelated, so 1-step forecasting looks trivially
good):

  - SIMULATION (free-running hindcast): given only an initial condition + forcing,
    reproduce the whole validation window without ever seeing observed levels. This is
    what the gray-box ODE does, and how the neural operator MUST be scored for a fair
    comparison. Honest no-skill references here are ``climatology`` and ``last_value``.
  - FORECAST (1-step-ahead): predict day t given the observation at t-1. Persistence
    is the no-skill reference here. Reported for context only; NOT comparable to the
    gray-box simulation scores.

References:
  - Gray-box ODE: per-well calibrated model (one parameter set per well), read from a
    single_tankV2 ``gw_fit_results.csv`` (simulation mode). The published baseline.
  - persistence_prediction: forecast-mode no-skill (tomorrow = today).
  - climatology_prediction / last_value_prediction: simulation-mode no-skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, default_config
from .data import GWData

# Columns of interest from a gray-box gw_fit_results.csv.
_GRAYBOX_COLS = ["st_id", "model", "group_name", "kge_val", "rmse_val", "r2_val"]


def load_graybox_baseline(cfg: Config | None = None) -> pd.DataFrame:
    """Return the gray-box per-well validation scores, indexed by well id.

    Raises FileNotFoundError if no baseline results path is configured/available.
    """
    cfg = cfg or default_config()
    if cfg.baseline_results is None or not cfg.baseline_results.exists():
        raise FileNotFoundError(
            "No gray-box baseline results found. Set Config.baseline_results to a "
            "gw_fit_results.csv (e.g. single_tankV2/workspace/results/final/)."
        )
    df = pd.read_csv(cfg.baseline_results)
    df["st_id"] = df["st_id"].astype(str)
    have = [c for c in _GRAYBOX_COLS if c in df.columns]
    out = df[have].set_index("st_id")
    return out.rename(columns={"kge_val": "kge", "rmse_val": "rmse", "r2_val": "r2"})


def persistence_prediction(data: GWData) -> np.ndarray:
    """FORECAST-mode lag-1 persistence: pred[:, t] = observed target[:, t-1].

    NaN at t = 0. This is the no-skill reference for 1-step-ahead forecasting only.
    Do NOT compare its scores to simulation-mode models (gray-box, neural operator):
    groundwater is so autocorrelated that this looks deceptively strong.
    """
    pred = np.full_like(data.target, np.nan)
    pred[:, 1:] = data.target[:, :-1]
    return pred


def last_value_prediction(data: GWData) -> np.ndarray:
    """SIMULATION-mode no-skill: constant equal to each well's last training value.

    The dumbest free-running "simulation" (hold the last calibration level forever).
    A real simulation model must clearly beat this.
    """
    pred = np.full_like(data.target, np.nan)
    for i in range(data.n_wells):
        train_obs = data.target[i, data.train_mask]
        valid = train_obs[np.isfinite(train_obs)]
        if valid.size:
            pred[i, :] = valid[-1]
    return pred


def climatology_prediction(data: GWData) -> np.ndarray:
    """SIMULATION-mode no-skill: per-well seasonal mean by day-of-year (training only).

    Uses only the calibration period to estimate a climatology, then predicts it for
    every day (including validation). This is the honest seasonal reference a useful
    simulation model should beat.
    """
    pred = np.full_like(data.target, np.nan)
    doy = data.doy
    for i in range(data.n_wells):
        clim = np.full(367, np.nan)
        obs_train = np.where(data.train_mask, data.target[i], np.nan)
        for d in range(1, 367):
            vals = obs_train[doy == d]
            vals = vals[np.isfinite(vals)]
            if vals.size:
                clim[d] = vals.mean()
        # fill any empty day-of-year with the overall training mean
        overall = np.nanmean(obs_train) if np.isfinite(np.nanmean(obs_train)) else 0.0
        clim = np.where(np.isfinite(clim), clim, overall)
        pred[i, :] = clim[doy]
    return pred
