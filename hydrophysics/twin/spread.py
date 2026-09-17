"""Spread the pumping stress over a radius before it enters the solver.

Every gate so far has said the same thing: a physical pump conversion fits in sample and
fails on held-out wells, while a fit that damps the stress generalises. The census puts
each pump's electricity in the 1 km cell of its billing coordinate, so the stress lands
as cell-scale hot spots (p99 cell-month 3.4e5 m3); a held-out well next to one is
predicted with a drawdown the observations do not show. A farm's wells are not at its
meter, and a well's cone of depression is not a cell. This module redistributes each
cell's energy with a mass-conserving Gaussian kernel of width ``sigma_km``, either fixed
(``--pump-spread-km``) or learned (``--learn-spread``, bounded 0.5-10 km) so the data
decide how local the stress is.
"""

from __future__ import annotations

import math

import numpy as np
import torch

SPREAD_KM_BOUNDS = (math.log(0.5), math.log(10.0))


def pairwise_d2_km(grid, device=None) -> torch.Tensor:
    """(A, A) squared centroid distances in km^2, float64, on ``device``."""
    c = torch.tensor(grid.centroids() / 1000.0, dtype=torch.float64, device=device)
    return torch.cdist(c, c) ** 2


def spread_matrix(d2_km: torch.Tensor, log_sigma_km: torch.Tensor) -> torch.Tensor:
    """Column-normalised Gaussian kernel ``W`` (A, A): ``W[i, j]`` is the share of source
    cell ``j``'s energy that lands in cell ``i``. Columns sum to one, so fan-wide energy is
    conserved; only where it is applied changes. Differentiable in ``log_sigma_km``."""
    sigma2 = torch.exp(2.0 * log_sigma_km.to(d2_km.dtype))
    K = torch.exp(-d2_km / (2.0 * sigma2))
    return K / K.sum(dim=0, keepdim=True)


def spread_energy(E: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Apply ``W`` to an energy field ``(A, T)`` or ``(C, A, T)`` -> same shape."""
    if E.dim() == 3:
        return torch.einsum("ij,cjt->cit", W, E)
    return W @ E


def spread_energy_numpy(E: np.ndarray, grid, sigma_km: float) -> np.ndarray:
    """Convenience for offline use: numpy in, numpy out, fixed sigma."""
    d2 = pairwise_d2_km(grid)
    W = spread_matrix(d2, torch.tensor(math.log(sigma_km), dtype=torch.float64))
    return spread_energy(torch.tensor(E, dtype=torch.float64), W).numpy()
