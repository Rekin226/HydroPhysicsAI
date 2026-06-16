"""SpatialPINN: a physics-informed continuous head field h(x, y, t) over the fan.

The lumped UDE models one ODE per well (0D in space). This learns a single continuous
field across the whole alluvial fan, trained against the wells as scattered observation
points and regularized by the 2D depth-averaged groundwater-flow PDE

    S * dh/dt = div(T grad h) + alpha(x,y) * R(x,y,t) - d(x,y)

All derivatives come from autodiff (no mesh). T, alpha, d are small spatial sub-networks;
S is a learned positive scalar. The model satisfies the GroundwaterModel interface, so
bench.py and the benchmark table wire up unchanged. See
docs/superpowers/specs/2026-06-16-spatial-pinn-head-field-design.md.
"""

from __future__ import annotations

import numpy as np

from ..data import GWData  # noqa: F401
from ..field_inputs import Normalizer, RainfallField, well_coords_norm  # noqa: F401
from .base import GroundwaterModel  # noqa: F401
from .gru import _default_device  # noqa: F401

try:
    import torch
    from torch import nn  # noqa: F401
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


def positional_encoding(coords, n_bands: int):
    """NeRF-style encoding: [coords, sin(2^k pi c), cos(2^k pi c) for k in 0..n_bands-1].

    coords: (N, D) tensor. Returns (N, D + D*2*n_bands). Differentiable.
    """
    feats = [coords]
    for k in range(n_bands):
        freq = (2.0 ** k) * np.pi
        feats.append(torch.sin(freq * coords))
        feats.append(torch.cos(freq * coords))
    return torch.cat(feats, dim=-1)
