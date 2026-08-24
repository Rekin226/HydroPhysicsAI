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
        # offset of the preconsolidation head relative to h[:, 0]; 0 = normally consolidated
        self.h_pc0 = nn.Parameter(z.clone())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        ske = torch.exp(self.log_ske)
        skv = torch.exp(self.log_skv)
        tau = torch.exp(self.log_tau)
        decay = torch.exp(-torch.tensor(self.dt_days, dtype=h.dtype, device=h.device) / tau)
        n, T = h.shape
        # h_pc0 is an OFFSET relative to each site's starting head, not an absolute head.
        # Absolute framing silently disabled the inelastic term wherever a site's heads never
        # crossed the datum: with h_pc0 init 0.0, `min(0, h[0])` pinned the gate at 0 m, so at
        # 7 of 14 Choushui sites (all heads above sea level) it never opened and log_skv /
        # log_tau received no gradient at all. Offsetting from h[:, 0] makes the init mean
        # "normally consolidated at t=0" and is datum-independent.
        h_pc = h[:, 0] + self.h_pc0
        eps_i = torch.zeros(n, dtype=h.dtype, device=h.device)
        eq = torch.zeros(n, dtype=h.dtype, device=h.device)
        out = [torch.zeros(n, dtype=h.dtype, device=h.device)]
        for t in range(1, T):
            below = torch.clamp(h_pc - h[:, t], min=0.0)
            eq = eq + skv * below                     # equilibrium inelastic strain
            h_pc = torch.minimum(h_pc, h[:, t])
            eps_i = eq + (eps_i - eq) * decay         # exact for piecewise-constant forcing
            eps_e = ske * (h[:, 0] - h[:, t])
            out.append(eps_e + eps_i)
        return torch.stack(out, dim=1)
