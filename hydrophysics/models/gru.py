"""GlobalGRU: a single deep-learning model across all wells (reference contestant).

This is the working template, the simplest thing that is a real, GPU-trainable model
sharing one set of weights across every well. It maps the forcing sequence (rainfall,
upstream level, seasonal features) plus static well attributes to the groundwater-level
sequence. Because it never consumes the observed target, running it forward IS a
simulation-mode hindcast, directly comparable to the gray-box.

Use it as the baseline to extend: swap the GRU core for the physics-informed UDE
(ude.py) and port to PhysicsNeMo for the headline result.

Requires torch. Runs on cuda > mps > cpu (see hydrophysics.train.pick_device).

Note on upstream forcing: like the gray-box ODE, this uses observed upstream levels as
an external driver. That keeps the comparison fair (same inputs). A fully-coupled
variant that simulates all wells jointly is a later extension.
"""

from __future__ import annotations

import numpy as np

from ..data import GWData
from .base import GroundwaterModel

try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional dependency
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "GlobalGRU needs PyTorch. Install with: pip install 'hydrophysics[gpu]' "
            "(or pip install torch). The foundation modules do not need torch."
        )


def _forcing_features(data: GWData) -> np.ndarray:
    """Per-well, per-day dynamic forcing -> (W, T, F_dyn). NaNs filled with 0."""
    doy = data.doy.astype(float)
    sin = np.sin(2 * np.pi * doy / 365.25)
    cos = np.cos(2 * np.pi * doy / 365.25)
    W, T = data.target.shape
    feats = np.stack([
        np.nan_to_num(data.rainfall),
        np.nan_to_num(data.upstream),
        np.broadcast_to(sin, (W, T)),
        np.broadcast_to(cos, (W, T)),
    ], axis=-1)
    return feats.astype("float32")


def _static_features(data: GWData) -> np.ndarray:
    """Numeric static attributes per well -> (W, F_static). Standardized later."""
    a = data.attrs
    cols = ["tm_x", "tm_y", "is_coastal", "dist_to_coast_m", "dom_amp",
            "ups_lag_days", "rf_lag_days"]
    mat = []
    for c in cols:
        if c in a.columns:
            mat.append(a[c].astype(float).fillna(0.0).to_numpy())
        else:
            mat.append(np.zeros(len(a)))
    return np.stack(mat, axis=-1).astype("float32")


class GlobalGRU(GroundwaterModel):
    name = "global_gru"

    def __init__(self, hidden: int = 64, layers: int = 2, epochs: int = 60,
                 lr: float = 1e-3, device: str | None = None, seed: int = 0):
        _require_torch()
        self.hidden, self.layers, self.epochs, self.lr = hidden, layers, epochs, lr
        self.seed = seed
        self.device = device or _default_device()
        self.net: nn.Module | None = None
        self._stats: dict[str, np.ndarray] = {}

    # --- standardization computed on the calibration period only ---
    def _standardize_fit(self, dyn, stat, target, train_mask):
        tm = train_mask
        self._stats["dyn_mu"] = dyn[:, tm].reshape(-1, dyn.shape[-1]).mean(0)
        self._stats["dyn_sd"] = dyn[:, tm].reshape(-1, dyn.shape[-1]).std(0) + 1e-6
        self._stats["stat_mu"] = stat.mean(0)
        self._stats["stat_sd"] = stat.std(0) + 1e-6
        t_train = target[:, tm]
        t_train = t_train[np.isfinite(t_train)]
        self._stats["y_mu"] = t_train.mean()
        self._stats["y_sd"] = t_train.std() + 1e-6

    def _to_tensors(self, dyn, stat):
        s = self._stats
        dyn = (dyn - s["dyn_mu"]) / s["dyn_sd"]
        stat = (stat - s["stat_mu"]) / s["stat_sd"]
        W, T, _ = dyn.shape
        stat_seq = np.broadcast_to(stat[:, None, :], (W, T, stat.shape[-1]))
        x = np.concatenate([dyn, stat_seq], axis=-1).astype("float32")
        return torch.from_numpy(x).to(self.device)

    def fit(self, data: GWData) -> "GlobalGRU":
        torch.manual_seed(self.seed)
        dyn = _forcing_features(data)
        stat = _static_features(data)
        self._standardize_fit(dyn, stat, data.target, data.train_mask)

        x = self._to_tensors(dyn, stat)                       # (W, T, F)
        y = (data.target - self._stats["y_mu"]) / self._stats["y_sd"]
        y = torch.from_numpy(np.nan_to_num(y).astype("float32")).to(self.device)
        obs_mask = torch.from_numpy(
            (np.isfinite(data.target) & data.train_mask[None, :]).astype("float32")
        ).to(self.device)

        self.net = _GRUNet(x.shape[-1], self.hidden, self.layers).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self.net(x).squeeze(-1)                    # (W, T)
            loss = ((pred - y) ** 2 * obs_mask).sum() / obs_mask.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
        return self

    def simulate(self, data: GWData) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("call fit() before simulate()")
        dyn = _forcing_features(data)
        stat = _static_features(data)
        x = self._to_tensors(dyn, stat)
        with torch.no_grad():
            pred = self.net(x).squeeze(-1).cpu().numpy()
        return pred * self._stats["y_sd"] + self._stats["y_mu"]


if _HAS_TORCH:
    class _GRUNet(nn.Module):
        def __init__(self, n_in: int, hidden: int, layers: int):
            super().__init__()
            self.gru = nn.GRU(n_in, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            h, _ = self.gru(x)
            return self.head(h)


def _default_device() -> str:
    if not _HAS_TORCH:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
