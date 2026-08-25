"""Differentiable multi-layer transient groundwater flow on a masked grid.

Five-point finite volume, backward Euler, monthly steps by default. Each step solves

    (S*A/dt + K) h^{n+1} = S*A/dt h^n + q

with K the symmetric conductance operator. The solve runs matrix-free under no_grad; the
gradient is attached by implicit differentiation (a transposed solve with the same
operator), so memory does not grow with iteration count.

That "does not depend on how tightly CG converged" guarantee holds for the *forward* head
solve only: the implicit-function-theorem backward is exact for whatever ``y`` the forward
solve actually returned, converged or not. The *adjoint* solve (lambda = M^{-1} grad_y,
computed inside backward) is a CG solve in its own right and its convergence matters
directly -- a stalled lambda solve feeds a wrong lambda into both the b-gradient and the
parameter vjp, corrupting every gradient computed from this step. See ``_cg``'s residual
warning.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import nn

from .grid import FanGrid

_CG_TOL = 1e-8
_CG_MAXITER = 400
_CG_RESTART = 50


def _neighbour_index(grid: FanGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Active-cell index pairs (i, j) for every shared face, each face listed once."""
    idx = -np.ones((grid.ny, grid.nx), dtype="int64")
    rows, cols = np.nonzero(grid.mask)
    idx[rows, cols] = np.arange(rows.size)
    a, b = [], []
    for dr, dc in ((0, 1), (1, 0)):
        r2, c2 = rows + dr, cols + dc
        ok = (r2 < grid.ny) & (c2 < grid.nx)
        ok[ok] &= grid.mask[r2[ok], c2[ok]]
        a.append(idx[rows[ok], cols[ok]])
        b.append(idx[r2[ok], c2[ok]])
    return (torch.as_tensor(np.concatenate(a)), torch.as_tensor(np.concatenate(b)))


class _ImplicitSolve(torch.autograd.Function):
    """y = M(params)^{-1} b with an exact adjoint.

    ``op(y, *params)`` must compute ``M(params) @ y`` in a way that is differentiable
    with respect to ``params`` (used only in backward, with grad enabled). ``solve(rhs)``
    solves ``M @ x = rhs`` with the *current* params and always runs under no_grad (used
    both for the forward solve and, since M is symmetric, for the adjoint solve).

    M depends on the log-parameters through a matvec closure, a dependency invisible to
    autograd because the forward solve runs under no_grad. So backward has two jobs:

      1. grad wrt b: lam = M^{-T} grad_y = M^{-1} grad_y (M symmetric), which flows into
         whatever autograd expression built b (e.g. S*area/dt*h_prev + q).
      2. grad wrt each parameter theta that M itself depends on: by the implicit
         function theorem, dh/dtheta|_{b fixed} = -M^{-1} (dM/dtheta) y, so
         dL/dtheta = -lam . (dM/dtheta y) = -d/dtheta (lam . M(theta) y).
         Since op(y, theta) = M(theta) y is linear in y and differentiable in theta when
         evaluated with grad enabled, this is a single vjp through ``op``.
    """

    @staticmethod
    def forward(ctx, b, op, solve, *params):
        with torch.no_grad():
            y = solve(b)
        ctx.save_for_backward(y, *params)
        ctx.op, ctx.solve = op, solve
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, *params = ctx.saved_tensors
        with torch.no_grad():
            lam = ctx.solve(grad_y)          # M is symmetric, so M^T == M

        detached = [p.detach().requires_grad_(True) for p in params]
        with torch.enable_grad():
            My = ctx.op(y, *detached)
            raw_grads = torch.autograd.grad(
                My, detached, grad_outputs=lam, allow_unused=True,
            )
        # allow_unused=True above only to get a clean per-parameter error message below
        # instead of an autograd.grad RuntimeError; every parameter passed in is expected
        # to actually affect the operator. A None here means one silently dropped out of
        # `op` -- exactly the defect class this backward was rewritten to fix (the brief's
        # version returned None for every parameter unconditionally) -- so fail loudly
        # rather than propagate a silent zero gradient.
        for i, g in enumerate(raw_grads):
            if g is None:
                raise RuntimeError(
                    f"_ImplicitSolve.backward: parameter at position {i} did not "
                    "contribute to op(y, *params) -- it dropped out of the operator "
                    "(check that _op actually uses it)."
                )
        param_grads = tuple(-g for g in raw_grads)
        return (lam, None, None) + param_grads


def _warm_started_solver(matvec, x0):
    """Bind ``x0`` now (rather than via a lambda default-arg trick) so the returned
    closure always warm-starts from the head at the time it was created, not whatever
    the loop variable holds when the closure is later invoked (e.g. by
    ``_ImplicitSolve.backward`` for the adjoint solve, well after the forward loop ends).
    """
    x0 = x0.detach()

    def solve(rhs):
        return _cg(matvec, rhs, x0=x0)

    return solve


def _cg(matvec, b, x0=None, tol=_CG_TOL, maxiter=_CG_MAXITER):
    """Matrix-free conjugate gradient for a symmetric positive-definite operator.

    Warns once if the *true* relative residual (recomputed from scratch, not the value
    tracked by the CG recurrence) has not reached ``tol`` within ``maxiter`` iterations.
    The same closure produces both the forward head solve and the adjoint's lambda solve,
    so a stalled solve here silently corrupts the heads *and* every gradient computed from
    them -- this is deliberately not a hard error, since a warned-about near-miss (e.g.
    heads accurate to 1e-6 instead of 1e-8) is often still usable, but it must not pass
    silently.

    Two float32-specific safeguards, both needed:

    1. The convergence check *inside* the loop uses the cheap recurrence residual
       ``r = r - alpha*Ap``, which drifts from the true residual ``b - matvec(x)`` under
       float32 rounding over the ~100-140 iterations these solves typically take --
       confirmed on a plain, non-adversarial case (T=500, single well): the recurrence
       reports relres~9e-9 (converged) while the true residual is ~1e-5, four orders of
       magnitude worse. So every ``_CG_RESTART`` iterations the recurrence is corrected by
       recomputing ``r`` from scratch (the standard remedy for this drift; costs one extra
       matvec per restart), rather than trusting the incremental update indefinitely.
    2. Independent of restarts, the *warning* always recomputes the true residual after
       the loop rather than reusing whatever the recurrence last reported -- the drift
       above means the recurrence's own belief about convergence cannot be trusted even
       right when the loop exits.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    b_norm = (b * b).sum().sqrt().clamp(min=1e-30)
    relres = (r * r).sum().sqrt() / b_norm
    if relres >= tol:
        p = r.clone()
        rs = (r * r).sum()
        for i in range(1, maxiter + 1):
            Ap = matvec(p)
            alpha = rs / (p * Ap).sum().clamp(min=1e-30)
            x = x + alpha * p
            r = b - matvec(x) if i % _CG_RESTART == 0 else r - alpha * Ap
            rs_new = (r * r).sum()
            relres = rs_new.sqrt() / b_norm
            if relres < tol:
                break
            p = r + (rs_new / rs.clamp(min=1e-30)) * p
            rs = rs_new
    true_relres = (b - matvec(x)).pow(2).sum().sqrt() / b_norm
    if true_relres >= tol:
        warnings.warn(
            f"_cg did not converge within maxiter={maxiter}: true relative residual "
            f"{float(true_relres):.3e} exceeds tol={tol:.1e}. Heads and gradients from "
            "this solve may be inaccurate.",
            stacklevel=2,
        )
    return x


class FlowModel(nn.Module):
    """Multi-layer transient flow. ``forward`` returns heads ``(L, A, T+1)``."""

    def __init__(self, grid: FanGrid, n_layers: int = 4, dt_days: float = 30.0,
                 device=None):
        super().__init__()
        self.grid = grid
        self.n_layers = int(n_layers)
        self.dt = float(dt_days)
        self.area = float(grid.dx) ** 2
        A = grid.n_active
        ia, ib = _neighbour_index(grid)
        self.register_buffer("ia", ia.to(device) if device else ia)
        self.register_buffer("ib", ib.to(device) if device else ib)
        z = torch.zeros(self.n_layers, A, device=device)
        self.log_T = nn.Parameter(z.clone() + float(np.log(500.0)))     # m2/day
        self.log_S = nn.Parameter(z.clone() + float(np.log(1e-4)))      # -
        self.log_L = nn.Parameter(torch.zeros(max(self.n_layers - 1, 1), A, device=device)
                                  + float(np.log(1e-4)))               # 1/day

    def _matvec_from(self, T, S):
        """Return a closure applying (S*area/dt + K) to a head vector of shape (L, A)."""
        ia, ib = self.ia, self.ib
        Tf = 2.0 * T[:, ia] * T[:, ib] / (T[:, ia] + T[:, ib]).clamp(min=1e-30)  # harmonic

        def mv(h):
            out = S * self.area / self.dt * h
            dh = h[:, ia] - h[:, ib]
            flux = Tf * dh
            out = out.index_add(1, ia, flux)
            out = out.index_add(1, ib, -flux)
            # Vertical leakage is added in Task 3; layers are independent here.
            return out

        return mv

    def _op(self, h, log_T, log_S):
        """Differentiable M(log_T, log_S) @ h, used only by the adjoint's backward."""
        T = torch.exp(log_T)
        S = torch.exp(log_S)
        return self._matvec_from(T, S)(h)

    def forward(self, h0: torch.Tensor, recharge: torch.Tensor,
                pumping: torch.Tensor, n_steps: int) -> torch.Tensor:
        T = torch.exp(self.log_T)
        S = torch.exp(self.log_S)
        mv = self._matvec_from(T, S)
        h = h0
        out = [h0]
        for t in range(n_steps):
            q = recharge[..., t] * self.area - pumping[..., t]
            b = S * self.area / self.dt * h + q
            # Warm-start CG from the previous head: consecutive backward-Euler steps have
            # similar solutions, so this is free and cuts iterations substantially.
            solve = _warm_started_solver(mv, h)
            h = _ImplicitSolve.apply(b, self._op, solve, self.log_T, self.log_S)
            out.append(h)
        return torch.stack(out, dim=-1)
