"""Hydrological skill metrics with NaN handling.

All functions take 1-D arrays ``obs`` and ``sim`` of equal length, ignore any
index where either is NaN, and return a float (NaN if fewer than 2 valid pairs).
"""

from __future__ import annotations

import numpy as np


def _valid_pair(obs: np.ndarray, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    if obs.shape != sim.shape:
        raise ValueError(f"shape mismatch: obs {obs.shape} vs sim {sim.shape}")
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    o, s = _valid_pair(obs, sim)
    if o.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((s - o) ** 2)))


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency. 1 is perfect; 0 equals the mean-of-obs baseline."""
    o, s = _valid_pair(obs, sim)
    if o.size < 2:
        return float("nan")
    denom = np.sum((o - o.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((o - s) ** 2) / denom)


def kge_components(obs: np.ndarray, sim: np.ndarray) -> tuple[float, float, float]:
    """Return the (r, alpha, beta) components of the 2009 Kling-Gupta efficiency.

    r     = Pearson correlation
    alpha = std(sim) / std(obs)   (variability ratio)
    beta  = mean(sim) / mean(obs) (bias ratio)
    """
    o, s = _valid_pair(obs, sim)
    if o.size < 2:
        return (float("nan"), float("nan"), float("nan"))
    so, ss = o.std(), s.std()
    mo, ms = o.mean(), s.mean()
    # Treat a near-constant series as zero-variance (floating-point cancellation can
    # leave a std of ~1e-15 for a constant prediction, which would otherwise yield a
    # spurious correlation). Scale the tolerance by the observed magnitude.
    tol = 1e-8 * max(abs(mo), abs(so), 1.0)
    degenerate = so <= tol or ss <= tol
    r = float("nan") if degenerate else float(np.corrcoef(o, s)[0, 1])
    alpha = float(ss / so) if so > tol else float("nan")
    beta = float(ms / mo) if mo != 0 else float("nan")
    return (r, alpha, beta)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta efficiency (Gupta et al., 2009). 1 is perfect."""
    r, alpha, beta = kge_components(obs, sim)
    if not all(np.isfinite([r, alpha, beta])):
        return float("nan")
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def all_metrics(obs: np.ndarray, sim: np.ndarray) -> dict[str, float]:
    """Convenience: KGE, NSE, RMSE plus KGE components in one dict."""
    r, alpha, beta = kge_components(obs, sim)
    return {
        "kge": kge(obs, sim),
        "nse": nse(obs, sim),
        "rmse": rmse(obs, sim),
        "kge_r": r,
        "kge_alpha": alpha,
        "kge_beta": beta,
    }
