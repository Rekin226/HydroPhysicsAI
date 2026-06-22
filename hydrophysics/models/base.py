"""Model interface. Every contestant must be scored the same way as the gray-box.

The contract is deliberately tiny:

    model.fit(data)                # learn on the calibration period only
    pred = model.simulate(data)    # (W, T) free-running hindcast over the FULL record

``simulate`` must NOT peek at observed groundwater levels during the validation period.
It receives forcing (rainfall, upstream levels, day-of-year) and static attributes, and
an initial condition taken from the last calibration value. This is what makes the
comparison to the per-well gray-box ODE fair: same information, same task.

The returned (W, T) array is fed straight into ``hydrophysics.eval.benchmark_table``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..data import GWData


def seed_everything(seed: int) -> None:
    """Seed torch + numpy, and on CUDA make cuDNN deterministic.

    The bit-identical reproductions the tests assert hold on CPU regardless; without the
    cuDNN flags a seeded run is NOT reproducible on GPU (cuDNN picks nondeterministic
    algorithms). The models here are small Linear/GRU/LSTM stacks, so forcing deterministic
    cuDNN costs negligible throughput. Call this instead of a bare torch.manual_seed.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GroundwaterModel(ABC):
    """Abstract base for any groundwater simulation model."""

    name: str = "model"

    @abstractmethod
    def fit(self, data: GWData) -> GroundwaterModel:
        """Train on the calibration period (data.train_mask). Return self."""

    @abstractmethod
    def simulate(self, data: GWData) -> np.ndarray:
        """Free-running hindcast over all days. Shape (n_wells, n_days).

        Must not use observed target values on or after the split. Use the last
        calibration value per well as the initial condition.
        """

    def initial_condition(self, data: GWData) -> np.ndarray:
        """Per-well initial level: last finite observation in the calibration period."""
        h0 = np.full(data.n_wells, np.nan)
        for i in range(data.n_wells):
            obs = data.target[i, data.train_mask]
            obs = obs[np.isfinite(obs)]
            if obs.size:
                h0[i] = obs[-1]
        return h0
