# hydrophysics/field_inputs.py
"""Domain geometry for the spatial head field: nondimensionalization + rainfall field.

Pure NumPy (no torch) so it is importable in the foundation/test path. The PINN
(``models/pinn_field.py``) consumes these helpers. Constants are absorbed into the
PINN's learned fields, so normalization only needs to be self-consistent: coordinates
to [0,1], time to years, head standardized on training observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import GWData


@dataclass
class Normalizer:
    xmin: float
    ymin: float
    length: float        # shared scale for x and y (preserves aspect ratio)
    h_mu: float
    h_sd: float
    year: float = 365.25

    @classmethod
    def from_data(cls, data: GWData) -> "Normalizer":
        x = data.attrs["tm_x"].astype(float).fillna(0.0).to_numpy()
        y = data.attrs["tm_y"].astype(float).fillna(0.0).to_numpy()
        xr = float(x.max() - x.min())
        yr = float(y.max() - y.min())
        length = max(xr, yr, 1.0)
        h = data.target[:, data.train_mask]
        h = h[np.isfinite(h)]
        h_mu = float(h.mean()) if h.size else 0.0
        h_sd = float(h.std()) + 1e-6 if h.size else 1.0
        return cls(xmin=float(x.min()), ymin=float(y.min()), length=length,
                   h_mu=h_mu, h_sd=h_sd)

    def xy(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.asarray(x) - self.xmin) / self.length, (np.asarray(y) - self.ymin) / self.length

    def tau(self, day_index: np.ndarray) -> np.ndarray:
        return np.asarray(day_index, dtype=float) / self.year

    def h_to_norm(self, h: np.ndarray) -> np.ndarray:
        return (np.asarray(h) - self.h_mu) / self.h_sd

    def h_from_norm(self, hn: np.ndarray) -> np.ndarray:
        return np.asarray(hn) * self.h_sd + self.h_mu
