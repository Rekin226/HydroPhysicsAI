"""Differentiable visco-elasto-plastic compaction column.

Tsai & Hsu 2018 (10.1016/j.enggeo.2018.07.025) show deformation on the Choushui fan is
visco-elasto-plastic: elastic, plastic *and* viscous with a delay. Lees et al. 2022
(10.1029/2021WR031390) find residual clay time constants of decades. The algebraic
``S = Sk * cumulative_drawdown`` in ``hydrophysics/subsidence.py`` has none of that memory,
which is why it fits in-sample and fails out-of-sample.

This module builds the rheology one term at a time so each has its own analytic test:
elastic (Task 4), preconsolidation-gated inelastic (Task 5), viscous relaxation (Task 6).
All parameters are log-parameterized to keep them positive.
"""

from __future__ import annotations

import torch
from torch import nn


class VEPColumn(nn.Module):
    """Visco-elasto-plastic compaction driven by head, one column per site.

    ``forward(h)`` maps heads ``(n_sites, T)`` in metres to cumulative compaction
    ``(n_sites, T)`` in metres, positive for subsidence and re-zeroed to ``t=0``.
    """

    def __init__(self, n_sites: int, dt_days: float = 30.0, device=None):
        super().__init__()
        self.n_sites = int(n_sites)
        self.dt_days = float(dt_days)
        z = torch.zeros(self.n_sites, device=device)
        self.log_ske = nn.Parameter(z.clone() + torch.log(torch.tensor(1e-3)))
        self.log_skv = nn.Parameter(z.clone() + torch.log(torch.tensor(2e-2)))
        self.log_tau = nn.Parameter(z.clone() + torch.log(torch.tensor(365.0)))
        self.h_pc0 = nn.Parameter(z.clone())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ske = torch.exp(self.log_ske).unsqueeze(1)          # (n, 1)
        drop = (h[:, :1] - h)                                # (n, T), >0 when head falls
        return ske * drop
