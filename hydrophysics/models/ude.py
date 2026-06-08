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

The forward integration, the attribute-conditioned hypernetwork, and the
physics-residual loss are implemented here and satisfy the GroundwaterModel interface,
so train.py and the benchmark wire up unchanged. The rollout is a stable semi-implicit
daily integrator; the level anchoring, per-well loss weighting, and LR schedule were
chosen on an inner pre-2019 tuning split (never the 2019+ benchmark). What remains is
the leave-one-well-out generalization study and the PhysicsNeMo port.

NVIDIA GPU path:
  - The class is plain PyTorch and trains as-is on any CUDA GPU (train.py picks cuda).
  - For very long rollouts where storing every step dominates memory, swap the loop in
    _rollout for torchdiffeq.odeint_adjoint over the same _decay_and_source ODE
    (constant-memory backprop on CUDA).
  - Port the hypernetwork + integration to NVIDIA PhysicsNeMo to use its physics-ML
    utilities, mixed-precision, and multi-GPU training. See README "NVIDIA GPU path".
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

    def __init__(self, hidden: int = 64, epochs: int = 1500, lr: float = 3e-2,
                 physics_weight: float = 0.1, anchor_level: bool = True,
                 normalize_loss: bool = True, lr_min: float | None = 1e-4,
                 device: str | None = None, seed: int = 0):
        _require_torch()
        self.hidden, self.epochs, self.lr, self.seed = hidden, epochs, lr, seed
        self.physics_weight = physics_weight
        # Improvements (validated on an inner pre-2019 tuning split, never on the
        # 2019+ benchmark): anchor each well's level to its training mean, weight every
        # well equally in the loss, and cosine-decay the learning rate to lr_min.
        self.anchor_level = anchor_level
        self.normalize_loss = normalize_loss
        self.lr_min = lr_min
        self.device = device or _default_device()
        self.hypernet: nn.Module | None = None
        self._stats: dict = {}

    def _build(self, n_static: int):
        # Hypernetwork: static attributes -> ODE parameters (one vector per well).
        self.hypernet = _HyperNet(n_static, self.hidden, len(PARAM_NAMES)).to(self.device)

    def _level_stats(self, data: GWData) -> tuple[np.ndarray, np.ndarray]:
        """Per-well mean and std of the level over the calibration period only.

        Used to anchor the absolute level and to weight the loss. Both read only
        training observations -- no peeking past the split -- exactly like the gray-box,
        which is calibrated per well on the same training window.
        """
        tm = data.train_mask
        mu = np.zeros(data.n_wells, dtype="float32")
        sd = np.ones(data.n_wells, dtype="float32")
        for i in range(data.n_wells):
            obs = data.target[i, tm]
            obs = obs[np.isfinite(obs)]
            if obs.size:
                mu[i] = obs.mean()
                sd[i] = obs.std() if obs.std() > 1e-6 else 1.0
        return mu, sd

    def fit(self, data: GWData, train_wells: np.ndarray | None = None) -> "PhysicsUDE":
        """Train the hypernetwork on the calibration period.

        train_wells: optional boolean mask (n_wells,). When given, only those wells
        contribute to the loss -- used for leave-one-well-out, where held-out wells must
        be predicted from their attributes (and own initial condition) alone, never from
        a fitted parameter set. Their attributes/IC/level-anchor are still available.
        """
        torch.manual_seed(self.seed)
        stat = _static_features(data)
        self._stats["stat_mu"] = stat.mean(0)
        self._stats["stat_sd"] = stat.std(0) + 1e-6
        mu_h, sd_h = self._level_stats(data)
        zeros = np.zeros_like(mu_h)
        self._stats["anchor"] = mu_h if self.anchor_level else zeros
        self._build(stat.shape[-1])

        stat_n = torch.from_numpy(
            ((stat - self._stats["stat_mu"]) / self._stats["stat_sd"]).astype("float32")
        ).to(self.device)
        dyn = torch.from_numpy(_forcing_features(data)).to(self.device)   # (W,T,4)
        target = torch.from_numpy(np.nan_to_num(data.target).astype("float32")).to(self.device)
        fit_mask = np.isfinite(data.target) & data.train_mask[None, :]
        if train_wells is not None:
            fit_mask = fit_mask & np.asarray(train_wells, dtype=bool)[:, None]
        obs_mask = torch.from_numpy(fit_mask.astype("float32")).to(self.device)
        h0 = torch.from_numpy(np.nan_to_num(self.initial_condition(data)).astype("float32")).to(self.device)
        anchor = torch.from_numpy(self._stats["anchor"]).to(self.device)   # (W,) level anchor
        scale = torch.from_numpy(sd_h if self.normalize_loss else np.ones_like(sd_h)).to(self.device)

        opt = torch.optim.Adam(self.hypernet.parameters(), lr=self.lr)
        sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs, eta_min=self.lr_min)
                 if self.lr_min is not None else None)
        for _ in range(self.epochs):
            opt.zero_grad()
            params = self.hypernet(stat_n)            # (W, n_params)
            pred = self._rollout(params, dyn, h0, anchor)     # (W, T)
            # Per-well scale weights every well equally (matches the per-well-median
            # KGE we report); scale is 1 when normalize_loss is off.
            data_loss = (((pred - target) / scale[:, None]) ** 2 * obs_mask).sum() \
                / obs_mask.sum().clamp_min(1.0)
            # Physics-residual: enforce the mass-balance ODE on observed training days,
            # teacher-forced (decoupled from the free rollout). This is the "informed"
            # half of physics-informed: it constrains the dynamics, not just the fit.
            phys_loss = self._physics_residual(params, dyn, target, obs_mask, anchor, scale)
            loss = data_loss + self.physics_weight * phys_loss
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
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
        anchor = torch.from_numpy(self._stats["anchor"]).to(self.device)
        with torch.no_grad():
            params = self.hypernet(stat_n)
            pred = self._rollout(params, dyn, h0, anchor)
        return pred.cpu().numpy()

    def _decay_and_source(self, params, dyn, anchor):
        """Cast the gray-box ODE into the affine form  dh/dt = -g*h + s_t.

        params: (W, n_params) -> a, z, b, c, k_link, d_sin, d_cos
        dyn:    (W, T, 4) -> rainfall, upstream, sin(doy), cos(doy)
        anchor: (W,) per-well level anchor (training mean, or zeros if disabled)

        The gray-box mass balance is
            dh/dt = -a*(h - z) + b*rain + k*(ups - h) + c + d_sin*sin + d_cos*cos
        Collecting the terms linear in h gives a decay rate g = a + k (>= 0) and an
        affine forcing s_t that carries everything else. ``c`` is a constant
        source/sink (baseline recharge), replacing the unused tidal placeholder: the
        daily forcing here has no tidal series, so we keep the parameter as a physical
        bias rather than wiring a driver that does not exist.

        The reference level z is predicted as a residual around ``anchor`` so the steady
        state h* ~= s/g sits near the well's observed operating level; without it the
        network must regress each absolute level from attributes alone, which blows up
        the mean bias on the hard wells.

        Returns ``g`` (W,) and ``s`` (W, T).
        """
        a = torch.nn.functional.softplus(params[:, 0])   # recession >= 0
        z = anchor + params[:, 1]                         # level as residual around anchor
        b = torch.nn.functional.softplus(params[:, 2])   # rainfall gain >= 0
        c = params[:, 3]
        k = torch.sigmoid(params[:, 4])                  # upstream coupling in (0,1)
        d_sin, d_cos = params[:, 5], params[:, 6]

        rain, ups, sin_t, cos_t = dyn[..., 0], dyn[..., 1], dyn[..., 2], dyn[..., 3]
        g = a + k                                        # (W,) linear decay coefficient
        s = ((a * z + c)[:, None]
             + k[:, None] * ups
             + b[:, None] * rain
             + d_sin[:, None] * sin_t
             + d_cos[:, None] * cos_t)                   # (W, T) affine forcing
        return g, s

    def _rollout(self, params, dyn, h0, anchor) -> "torch.Tensor":
        """Free-running daily integration of the gray-box ODE skeleton.

        params: (W, n_params); dyn: (W, T, 4); h0: (W,) initial level;
        anchor: (W,) per-well level anchor for the reference-level residual.
        Returns (W, T) simulated levels.

        Integration is semi-implicit (backward Euler on the linear decay term,
        explicit on the forcing) with dt = 1 day to match the gray-box. For decay
        g >= 0 this is unconditionally stable, so a long free-running val hindcast
        cannot blow up even while the hypernetwork is still learning -- the explicit
        Euler loop this replaces could diverge to NaN for large early-training g. The
        update is fully differentiable, so gradients flow through the whole rollout.

        NVIDIA GPU path: for very long rollouts where storing every step is the
        bottleneck, swap this loop for torchdiffeq.odeint_adjoint over the same
        ``_decay_and_source`` ODE (constant-memory backprop), or port to PhysicsNeMo.
        """
        g, s = self._decay_and_source(params, dyn, anchor)
        denom = 1.0 + g                                  # backward-Euler, dt = 1 day
        _, T = s.shape
        h = h0.clone()
        out = []
        for t in range(T):
            h = (h + s[:, t]) / denom
            out.append(h)
        return torch.stack(out, dim=1)

    def _physics_residual(self, params, dyn, target, obs_mask, anchor, scale) -> "torch.Tensor":
        """Mean squared mass-balance residual on observed consecutive training days.

        Teacher-forced: plug the observed level into the ODE right-hand side and require
        the one-day finite difference of the observations to match it,
            (h_obs[t+1] - h_obs[t]) ~= -g*h_obs[t] + s_t.
        This constrains the learned dynamics independently of the free rollout, which
        stabilizes early training and is the standard universal-ODE collocation term.
        The residual is scaled per well so all wells weigh equally. Only day-pairs with
        a finite training observation at both ends contribute.
        """
        g, s = self._decay_and_source(params, dyn, anchor)
        h_t, h_tp1 = target[:, :-1], target[:, 1:]
        rhs = s[:, :-1] - g[:, None] * h_t
        resid = ((h_tp1 - h_t) - rhs) / scale[:, None]
        pair_mask = obs_mask[:, :-1] * obs_mask[:, 1:]
        return (resid ** 2 * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


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
