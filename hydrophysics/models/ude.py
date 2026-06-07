"""PhysicsUDE: physics-informed Universal Differential Equation (the core new method).

Idea: keep the gray-box ODE skeleton (recession + rainfall + upstream coupling +
seasonal terms) as the inductive bias, but replace the 33-61 hand-calibrated per-well
parameter sets with ONE shared neural network ("hypernetwork") that maps each well's
static attributes to its ODE parameters. Train it jointly across all wells by
backpropagating through a differentiable ODE integration.

Why this is the flagship and not the GRU:
  - It is physics-informed: the mass-balance structure is hard-wired, so it extrapolates
    and stays interpretable (you can read off a, b, k_link per well).
  - It amortizes: one network conditions on attributes, so it can predict a well it was
    never calibrated on (leave-one-well-out) -- the operator-learning headline.

This file is a SKELETON. The forward integration is sketched; the parts marked TODO are
what you implement and tune on the GPU. The class still satisfies the GroundwaterModel
interface so train.py and the benchmark wire up unchanged.

NVIDIA GPU path:
  - Replace the hand-rolled Euler loop with a stiff-safe differentiable solver
    (torchdiffeq.odeint_adjoint) for memory-efficient long rollouts on CUDA.
  - Port the hypernetwork + integration to NVIDIA PhysicsNeMo to use its physics-ML
    utilities, mixed-precision, and multi-GPU training. See README "NVIDIA GPU path".
  - Add a physics-residual loss term (penalize mass-balance violation) alongside the
    data loss -- the "informed" part of physics-informed.
"""

from __future__ import annotations

import numpy as np

from ..data import GWData
from .base import GroundwaterModel
from .gru import _forcing_features, _static_features, _default_device

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


# The ODE parameters the hypernetwork predicts per well. Mirrors the gray-box.
PARAM_NAMES = ["a", "z", "b", "c", "k_link", "d_sin", "d_cos"]


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("PhysicsUDE needs PyTorch. pip install 'hydrophysics[gpu]'.")


class PhysicsUDE(GroundwaterModel):
    name = "physics_ude"

    def __init__(self, hidden: int = 64, epochs: int = 200, lr: float = 1e-2,
                 device: str | None = None, seed: int = 0):
        _require_torch()
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self.device = device or _default_device()
        self.hypernet: nn.Module | None = None
        self._stats: dict = {}

    def _build(self, n_static: int):
        # Hypernetwork: static attributes -> ODE parameters (one vector per well).
        self.hypernet = _HyperNet(n_static, self.hidden, len(PARAM_NAMES)).to(self.device)

    def fit(self, data: GWData) -> "PhysicsUDE":
        torch.manual_seed(self.seed)
        stat = _static_features(data)
        self._stats["stat_mu"] = stat.mean(0)
        self._stats["stat_sd"] = stat.std(0) + 1e-6
        self._build(stat.shape[-1])

        stat_n = torch.from_numpy(
            ((stat - self._stats["stat_mu"]) / self._stats["stat_sd"]).astype("float32")
        ).to(self.device)
        dyn = torch.from_numpy(_forcing_features(data)).to(self.device)   # (W,T,4)
        target = torch.from_numpy(np.nan_to_num(data.target).astype("float32")).to(self.device)
        obs_mask = torch.from_numpy(
            (np.isfinite(data.target) & data.train_mask[None, :]).astype("float32")
        ).to(self.device)
        h0 = torch.from_numpy(np.nan_to_num(self.initial_condition(data)).astype("float32")).to(self.device)

        opt = torch.optim.Adam(self.hypernet.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            params = self.hypernet(stat_n)            # (W, n_params)
            pred = self._rollout(params, dyn, h0, data.doy)   # (W, T)
            data_loss = ((pred - target) ** 2 * obs_mask).sum() / obs_mask.sum().clamp_min(1.0)
            # TODO(GPU): add a physics-residual penalty here (mass-balance consistency).
            loss = data_loss
            loss.backward()
            opt.step()
        return self

    def simulate(self, data: GWData) -> np.ndarray:
        if self.hypernet is None:
            raise RuntimeError("call fit() before simulate()")
        stat = _static_features(data)
        stat_n = torch.from_numpy(
            ((stat - self._stats["stat_mu"]) / self._stats["stat_sd"]).astype("float32")
        ).to(self.device)
        dyn = torch.from_numpy(_forcing_features(data)).to(self.device)
        h0 = torch.from_numpy(np.nan_to_num(self.initial_condition(data)).astype("float32")).to(self.device)
        with torch.no_grad():
            params = self.hypernet(stat_n)
            pred = self._rollout(params, dyn, h0, data.doy)
        return pred.cpu().numpy()

    def _rollout(self, params, dyn, h0, doy) -> "torch.Tensor":
        """Free-running daily Euler integration of the gray-box ODE skeleton.

        params: (W, n_params) -> a, z, b, c, k_link, d_sin, d_cos
        dyn:    (W, T, 4) -> rainfall, upstream, sin(doy), cos(doy)
        h0:     (W,) initial level
        Returns (W, T) simulated levels.

        TODO(GPU): this hand-rolled Euler loop is fine on CPU/MPS for prototyping but is
        slow and memory-heavy for long rollouts. On CUDA, replace with
        torchdiffeq.odeint_adjoint over a vectorized ODE func for constant-memory
        backprop, or move the whole thing into PhysicsNeMo.
        """
        a = torch.nn.functional.softplus(params[:, 0])   # recession >= 0
        z = params[:, 1]
        b = torch.nn.functional.softplus(params[:, 2])
        c = params[:, 3]
        k = torch.sigmoid(params[:, 4])                  # coupling in (0,1)
        d_sin, d_cos = params[:, 5], params[:, 6]

        W, T, _ = dyn.shape
        rain, ups, sin_t, cos_t = dyn[..., 0], dyn[..., 1], dyn[..., 2], dyn[..., 3]
        h = h0.clone()
        out = []
        for t in range(T):
            dh = (-a * (h - z) + b * rain[:, t] + k * (ups[:, t] - h)
                  - c * 0.0  # TODO(GPU): wire tidal amplitude AMP[:,t] as a driver
                  + d_sin * sin_t[:, t] + d_cos * cos_t[:, t])
            h = h + dh
            out.append(h)
        return torch.stack(out, dim=1)


if _HAS_TORCH:
    class _HyperNet(nn.Module):
        """Static well attributes -> ODE parameters."""

        def __init__(self, n_in: int, hidden: int, n_params: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU(),
                nn.Linear(hidden, n_params),
            )

        def forward(self, x):
            return self.net(x)
