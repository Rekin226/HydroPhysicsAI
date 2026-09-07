"""Stage 4: the four-layer flow solver coupled to the VEP compaction column.

Until this module the twin was two disconnected halves. ``calibrate_flow`` fitted heads and
never imported ``VEPColumn``; ``calibrate_mlcw`` fitted a rheology driven by *observed*
IDW heads and never saw the flow model. Nothing turned an abstraction rate into a
subsidence field, which is the one thing the twin exists to do.

``CoupledTwin`` closes that loop:

    pumping, recharge --> FlowModel --> layer heads --> driver --> VEPColumn --> subsidence

and the whole path is differentiable, so a loss on subsidence reaches the flow parameters
(`log_T`, `log_S`, `log_L`) through the implicit-function adjoint in ``flow._ImplicitSolve``.
That is what makes joint calibration against heads *and* compaction *and* leveling possible,
and it is the property ``test_twin_coupled`` pins down.

The column driver
-----------------
The VEP column is one-dimensional and takes a single head series per cell, but the flow
model produces four. How they reduce is a physical claim, not a detail:

- ``mean`` (default) -- the layer-mean head. This is the like-for-like substitution for how
  Stage 2 was actually calibrated: ``calibrate_mlcw`` drove the column with the pooled IDW
  head over *all* wells regardless of aquifer, so the mean is what its fitted `Ske`/`Skv`
  were estimated against. Anything else silently re-scales those parameters.
- ``layer`` -- a single aquifer (``driver_layer``, 0-indexed; 1 = the main production
  aquifer that ``calibrate_flow`` pumps from).
- ``weighted`` -- learnable per-layer weights via softmax, letting joint calibration
  discover which aquifers actually drive compaction. Initialised uniform, so it starts
  exactly at ``mean`` and any departure is something the data paid for.

Spec §1's per-layer Stage-1 result (L1 -0.215, L2 -0.099, L3 +0.036, all worse than pooled)
is the evidence that no single aquifer is the right driver, which is why ``mean`` is the
default and ``weighted`` exists rather than a hardcoded layer.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .compaction import VEPColumn
from .flow import _MODEL_DTYPE, FlowModel
from .grid import FanGrid

DRIVERS = ("mean", "layer", "weighted")


class CoupledTwin(nn.Module):
    """Flow -> compaction, end to end.

    ``forward`` returns ``(heads, subsidence)`` with heads ``(n_layers, A, T+1)`` in metres
    and subsidence ``(A, T+1)`` in metres, positive for sinking and re-zeroed to ``t=0``.

    Both halves run in float64. The flow model requires it (see ``flow``'s module
    docstring: float32 CG has an accuracy floor of ~1e-5), and letting the column stay in
    float32 would put a silent precision seam in the middle of a gradient path that has to
    carry a subsidence loss back to ``log_T``.
    """

    def __init__(self, grid: FanGrid, n_layers: int = 4, dt_days: float = 30.0,
                 driver: str = "mean", driver_layer: int = 1, device=None):
        super().__init__()
        if driver not in DRIVERS:
            raise ValueError(f"driver must be one of {DRIVERS}, got {driver!r}")
        self.flow = FlowModel(grid, n_layers=n_layers, dt_days=dt_days, device=device)
        self.column = VEPColumn(n_sites=1, dt_days=dt_days, device=device).to(_MODEL_DTYPE)
        self.driver = driver
        self.driver_layer = int(driver_layer)
        self.n_layers = int(n_layers)
        if driver == "weighted":
            # Uniform init => softmax is exactly 1/n_layers => identical to ``mean`` on the
            # first step. A departure from the mean then has to be earned by the data.
            self.layer_logits = nn.Parameter(
                torch.zeros(n_layers, device=device, dtype=_MODEL_DTYPE))
        else:
            self.register_parameter("layer_logits", None)
        if driver == "layer" and not (0 <= self.driver_layer < n_layers):
            raise ValueError(f"driver_layer {driver_layer} outside 0..{n_layers - 1}")

    # --- driver ---------------------------------------------------------------
    def column_driver(self, heads: torch.Tensor) -> torch.Tensor:
        """Reduce layer heads ``(L, A, T)`` to the single series ``(A, T)`` the column sees."""
        if self.driver == "mean":
            return heads.mean(dim=0)
        if self.driver == "layer":
            return heads[self.driver_layer]
        w = torch.softmax(self.layer_logits, dim=0).to(heads.dtype)
        return (heads * w[:, None, None]).sum(dim=0)

    def layer_weights(self) -> np.ndarray:
        """Current driver weights per layer, for reporting."""
        if self.driver == "mean":
            return np.full(self.n_layers, 1.0 / self.n_layers)
        if self.driver == "layer":
            w = np.zeros(self.n_layers)
            w[self.driver_layer] = 1.0
            return w
        return torch.softmax(self.layer_logits.detach(), dim=0).cpu().numpy()

    # --- forward --------------------------------------------------------------
    def forward(self, h0: torch.Tensor, recharge: torch.Tensor, pumping: torch.Tensor,
                n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        heads = self.flow(h0, recharge, pumping, n_steps)      # (L, A, T+1)
        return heads, self.column(self.column_driver(heads))    # (A, T+1)

    # --- warm starts ----------------------------------------------------------
    def load_column(self, fitted: VEPColumn) -> CoupledTwin:
        """Adopt a Stage-2 fitted column (its shared global parameter set).

        Stage 2's gate passes on the *shared* arm -- one 4-vector for every site -- so the
        only thing worth carrying over is that vector, which is exactly ``n_sites=1``.
        """
        with torch.no_grad():
            for name in ("log_ske", "log_skv", "log_tau", "h_pc0"):
                src = getattr(fitted, name).detach()
                dst = getattr(self.column, name)
                if src.numel() != 1:
                    src = src.mean().reshape(1)   # per-site fit -> its pooled value
                dst.copy_(src.to(dst.dtype).reshape(dst.shape))
        return self

    def freeze_column(self, frozen: bool = True) -> CoupledTwin:
        """Hold the rheology fixed while the flow parameters move (staged calibration).

        The spec's staging rule is "warm-start compaction from Stage 2 and flow from
        Stage 3, then fine-tune jointly" -- this is the first half of that.
        """
        for p in self.column.parameters():
            p.requires_grad_(not frozen)
        return self


def subsidence_loss(pred: torch.Tensor, obs: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """Masked MSE between predicted and observed cumulative subsidence.

    Same masked-mean convention as ``calibrate_mlcw``: cells without an observation
    contribute nothing rather than being counted as zero error.
    """
    m = mask.to(pred.dtype)
    return (((pred - obs) ** 2) * m).sum() / m.sum().clamp(min=1)
