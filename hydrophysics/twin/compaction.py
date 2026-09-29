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

import math

import torch
from torch import nn


class VEPColumn(nn.Module):
    """Visco-elasto-plastic compaction driven by head, one column per site.

    ``forward(h)`` maps heads ``(n_sites, T)`` in metres to cumulative compaction
    ``(n_sites, T)`` in metres, positive for subsidence and re-zeroed to ``t=0``.

    ``creep="aquitard"`` (opt-in, 2026-09-29) adds the delayed-drainage term of
    :func:`aquitard_compaction` and its three parameters (``log_ska``, ``log_tau_a``,
    ``h_a0``). The default ``"vep"`` registers exactly the historical four parameters.
    """

    def __init__(self, n_sites: int, dt_days: float = 30.0, device=None,
                 creep: str = "vep"):
        super().__init__()
        if creep not in CREEP_MODES:
            raise ValueError(f"creep must be one of {CREEP_MODES}, got {creep!r}")
        self.n_sites = int(n_sites)
        self.dt_days = float(dt_days)
        self.creep = creep
        z = torch.zeros(self.n_sites, device=device)
        self.log_ske = nn.Parameter(z.clone() + torch.log(torch.tensor(1e-3)))
        self.log_skv = nn.Parameter(z.clone() + torch.log(torch.tensor(2e-2)))
        self.log_tau = nn.Parameter(z.clone() + torch.log(torch.tensor(365.0)))
        # offset of the preconsolidation head relative to h[:, 0]; 0 = normally consolidated
        self.h_pc0 = nn.Parameter(z.clone())
        if creep == "aquitard":
            self.log_ska = nn.Parameter(z.clone() + torch.log(torch.tensor(AQUITARD_INIT[0])))
            self.log_tau_a = nn.Parameter(z.clone() + torch.log(torch.tensor(AQUITARD_INIT[1])))
            self.h_a0 = nn.Parameter(z.clone() + AQUITARD_INIT[2])

    def param_keys(self) -> tuple[str, ...]:
        """The parameter names this column carries, in JSON order."""
        return VEP_KEYS + (AQUITARD_KEYS if self.creep == "aquitard" else ())

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out = vep_compaction(h, self.log_ske, self.log_skv, self.log_tau, self.h_pc0,
                             self.dt_days)
        if self.creep == "aquitard":
            out = out + aquitard_compaction(h, self.log_ska, self.log_tau_a, self.h_a0,
                                            self.dt_days)
        return out


VEP_KEYS = ("log_ske", "log_skv", "log_tau", "h_pc0")
AQUITARD_KEYS = ("log_ska", "log_tau_a", "h_a0")
CREEP_MODES = ("vep", "aquitard")
# initial (Ska, tau_a in days, h_a0 in m): h_a0 = 0 starts the term at rest
AQUITARD_INIT = (1e-2, 3652.5, 0.0)
# physical bounds for fitting: Ska in [1e-4, 1] m/m, tau_a in [1, 100] yr, |h_a0| <= 30 m
AQUITARD_BOUNDS = {"log_ska": (math.log(1e-4), math.log(1.0)),
                   "log_tau_a": (math.log(365.25), math.log(100 * 365.25)),
                   "h_a0": (-30.0, 30.0)}


def aquitard_compaction(h: torch.Tensor, log_ska: torch.Tensor, log_tau_a: torch.Tensor,
                        h_a0: torch.Tensor, dt_days: float = 30.0) -> torch.Tensor:
    """Delayed drainage of the clay interbeds: compaction that goes on at a rate set by
    the head (opt-in column term, 2026-09-29; ``twin.mechanism`` for the evidence).

    The interbeds' mean head ``u`` relaxes toward the aquifer head with time constant
    ``tau_a`` (exact for piecewise-constant forcing), starting ``h_a0`` metres above
    ``h[:, 0]`` (the excess head left by past drawdown). Compaction is ``Ska`` times the
    fall of ``u`` below its own past minimum -- inelastic, so a recovery never swells the
    clay back. While ``u`` is falling the rate is ``Ska (u - h) / tau_a``: it depends on
    the head *level*, so a sustained head rise slows it and a rise above ``u`` stops it.
    That is the behaviour the compaction rings show over their top 200-300 m (rate per
    metre of level about as large as the immediate response, for at least three years)
    and that the VEP term, whose creep only relaxes toward a load fixed by past minima,
    cannot produce. The leveling benchmarks beside the same rings, over the same
    intervals, do not show it in the base regression, though the site-trend variant does
    not exclude it (``twin.mechanism``, 2026-09-29): fit it to leveling before trusting it
    in a projection.

    Parameters are ``(1,)`` or ``(n,)``; returns ``(n, T)`` metres, zero at ``t=0``."""
    ska = torch.exp(log_ska)
    a = 1.0 - torch.exp(-torch.tensor(dt_days, dtype=h.dtype, device=h.device)
                        / torch.exp(log_tau_a))
    n, T = h.shape
    u0 = h[:, 0] + h_a0
    u = u0
    u_min = u0
    out = [torch.zeros(n, dtype=h.dtype, device=h.device)]
    for t in range(1, T):
        u = u + a * (h[:, t] - u)
        u_min = torch.minimum(u_min, u)
        out.append(ska * (u0 - u_min))
    return torch.stack(out, dim=1)


def column_compaction(h: torch.Tensor, p: dict, dt_days: float = 30.0) -> torch.Tensor:
    """A column JSON's parameter dict (one column) on heads ``(n, T)``: the VEP term, plus
    the aquitard term when the dict carries its keys. Plain floats are accepted."""
    def par(k):
        return torch.as_tensor(p[k], dtype=h.dtype, device=h.device).reshape(-1)

    out = vep_compaction(h, *(par(k) for k in VEP_KEYS), dt_days)
    if all(k in p for k in AQUITARD_KEYS):
        out = out + aquitard_compaction(h, *(par(k) for k in AQUITARD_KEYS), dt_days)
    return out


def vep_compaction(h: torch.Tensor, log_ske: torch.Tensor, log_skv: torch.Tensor,
                   log_tau: torch.Tensor, h_pc0: torch.Tensor,
                   dt_days: float = 30.0) -> torch.Tensor:
    """The column's recurrence with explicit parameters, each ``(1,)`` or ``(n_sites,)``.

    ``VEPColumn.forward`` is this function on its own parameters. A caller that builds
    per-cell parameters (the ``--zone-blend-km`` mix of per-zone sets) calls it directly.
    """
    ske = torch.exp(log_ske)
    skv = torch.exp(log_skv)
    tau = torch.exp(log_tau)
    decay = torch.exp(-torch.tensor(dt_days, dtype=h.dtype, device=h.device) / tau)
    n, T = h.shape
    # h_pc0 is an OFFSET relative to each site's starting head, not an absolute head.
    # Absolute framing silently disabled the inelastic term wherever a site's heads never
    # crossed the datum: with h_pc0 init 0.0, `min(0, h[0])` pinned the gate at 0 m, so at
    # 7 of 14 Choushui sites (all heads above sea level) it never opened and log_skv /
    # log_tau received no gradient at all. Offsetting from h[:, 0] makes the init mean
    # "normally consolidated at t=0" and is datum-independent.
    h_pc = h[:, 0] + h_pc0
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


def column_param_sets(params: dict) -> list[dict]:
    """The per-column parameter dicts of a column JSON: one per zone (``zonal``), one
    per band (``banded``, opt-in 2026-09-29), or the file itself (a shared column)."""
    for key in ("zonal", "banded"):
        if params.get(key):
            return params[key]
    return [params]
