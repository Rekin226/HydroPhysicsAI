"""Observation-consistent, layer-local groundwater head corrections.

The observation operator is the calibration's nearest active cell plus a fixed well
offset. Interpolate innovations, not absolute head fields: a perfectly predicted
observation must leave the state unchanged. These corrections add/remove storage;
they are data-assimilation increments, not physical recharge or pumping fluxes.
"""
from __future__ import annotations

import numpy as np
import torch


class HeadObservationOperator:
    """Nearest-cell sampling and localized IDW innovation spreading in each layer.

    Sources use the sampled cell centres, so interpolation and observation sampling
    share the same geometry. Multiple wells in one cell contribute their mean
    innovation; a cell cannot represent independent heads for colocated wells.
    """

    def __init__(self, centroids, cells, layers, n_layers=4, device="cpu"):
        xy = np.asarray(centroids, dtype=float)
        cells, layers = np.asarray(cells, dtype=int), np.asarray(layers, dtype=int)
        if (xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all()
                or cells.ndim != 1 or layers.shape != cells.shape
                or (cells < 0).any() or (cells >= len(xy)).any()
                or (layers < 0).any() or (layers >= n_layers).any()):
            raise ValueError("Invalid observation geometry")
        self.cells = torch.as_tensor(cells, device=device)
        self.layers = torch.as_tensor(layers, device=device)
        self.n_layers, self.n_cells = n_layers, len(xy)
        self.groups = []
        for layer in range(n_layers):
            wells = np.flatnonzero(layers == layer)
            d2 = ((xy[:, None] - xy[cells[wells]][None])**2).sum(axis=-1)
            self.groups.append((torch.as_tensor(wells, device=device),
                                torch.as_tensor(d2, dtype=torch.float64, device=device)))

    @classmethod
    def from_inputs(cls, inp, device="cpu"):
        return cls(inp.grid.centroids(), inp.obs_idx, inp.obs_layer, device=device)

    def observe(self, state, offset=None):
        if tuple(state.shape) != (self.n_layers, self.n_cells):
            raise ValueError("State does not match observation geometry")
        values = state[self.layers, self.cells]
        return values if offset is None else values + torch.as_tensor(offset).to(values)

    def update(self, state, observed, offset=None, gain=1.0, radius_km=5.0):
        if not 0 <= gain <= 1 or not np.isfinite(radius_km) or radius_km <= 0:
            raise ValueError("Gain must be in [0,1] and localization radius must be positive")
        prediction = self.observe(state, offset)
        observed = torch.as_tensor(observed).to(state)
        if observed.shape != prediction.shape:
            raise ValueError("Expected one observation per well")
        finite = torch.isfinite(observed) & torch.isfinite(prediction)
        residual = torch.where(finite, observed - prediction, 0)
        out = state.clone()
        if gain == 0:
            return out
        for layer, (wells, distance2) in enumerate(self.groups):
            if len(wells) == 0:
                continue
            distance2 = distance2.to(state)
            valid = finite[wells]
            weights = valid.to(state.dtype)[None] / (distance2 + 1e-6)
            correction = (weights @ residual[wells]) / weights.sum(dim=1).clamp_min(1e-30)
            nearest2 = torch.where(valid[None], distance2, torch.inf).min(dim=1).values
            taper = torch.exp(-nearest2 / (radius_km * 1000)**2)
            out[layer] += gain * taper * correction
        return out


def historical_offsets(observed, predicted, stop, fallback=None):
    """A fixed observation datum from months strictly before ``stop`` only."""
    residual = observed[:, :stop] - predicted[:, :stop]
    count = np.isfinite(residual).sum(axis=1)
    out = np.zeros(len(observed)) if fallback is None else np.asarray(fallback, float).copy()
    return np.divide(np.nansum(residual, axis=1), count, out=out, where=count > 0)


def lagged_residual_forecast(base, observed, origin_base, origin_observed, retention=1.0):
    """Well-level forecast correction using previous observations only.

    This corrects predictions at monitoring wells; it is not an aquifer state update
    and supplies neither spatial heads nor a mass-conserving intervention response.
    Missing previous observations fall back to the uncorrected base forecast.
    """
    if not 0 <= retention <= 1:
        raise ValueError("Residual retention must be in [0,1]")
    base, observed = np.asarray(base), np.asarray(observed)
    if base.ndim != 2 or observed.shape != base.shape:
        raise ValueError("Expected matching well-by-month arrays")
    previous_base = np.concatenate([np.asarray(origin_base)[:, None], base[:, :-1]], axis=1)
    previous_obs = np.concatenate([np.asarray(origin_observed)[:, None], observed[:, :-1]], axis=1)
    residual = previous_obs - previous_base
    return base + retention * np.where(np.isfinite(residual), residual, 0)
