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

The model holds its parameters in float64 and casts ``forward``'s inputs to float64 on
entry, returning float64. This is deliberate, not incidental: measured on this solver,
float32 CG has an honest accuracy floor of ~1e-5 regardless of iteration budget (CG's
running residual estimate drifts from the true residual under float32 rounding well
before the true error gets anywhere near a useful tolerance), while float64 reaches 1e-8
or better in fewer iterations than float32's illusory "converged" iteration count. Memory
is not the constraint one might expect: at fan scale (~2,150 cells x 4 layers x 133
monthly steps) a float64 rollout is on the order of 10 MB. Callers (including Task 5's
numpy-built calibration tensors) should not need to remember to pass float64 -- the
model enforces it internally regardless of what dtype it is called with.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
from torch import nn

from .boundaries import COAST_HEAD_M
from .grid import FanGrid

_CG_TOL = 1e-8
# Measured, not guessed: on the real 2148-cell fan grid, the zonal design pins the
# proximal zone's log_L at BOUNDS["log_L"][1] = log(1e-1), i.e. L = 1e-1/day (the
# proximal fan has no aquitards, so its layers equilibrate -- that pin is the design's
# physical claim, not a bug). Bisecting the iterations that pin actually needs: L=1e-1/T=500 -> 947;
# L=1e-1/T=10 -> 1242 (T=10 is the lower clamp the homogeneous fit actually reaches, so
# this is the worst realistic case). The previous cap of 400 silently truncated these
# solves well short of tol -- 1,879 CG solves sampled from the aborted gate run had a
# median true relative residual of 4.955e-02 and a max of 2.542e+04, all worse than the
# 1e-6 usability bar in this function's own docstring, because the truncated result
# feeds the next step's warm start (_warm_started_solver) and residuals compound. The
# operator is still SPD at the pin (200 Rayleigh quotients, min 1.813e+04, none <= 0)
# and converges cleanly to 9.607e-09 at maxiter=2000 -- this is slow convergence, not
# divergence. 2000 gives headroom over the measured worst case of 1242 iterations; CG
# exits as soon as it converges, so the larger cap costs nothing when it isn't needed
# (homogeneous mode converges at 237 iterations and never gets close to either cap).
_CG_MAXITER = 2000
_CG_RESTART = 50
# How often the CG loop *reads* its residual. Reading it means `if relres < tol`, a Python
# branch on a CUDA tensor, which forces a cudaStreamSynchronize -- so checking every
# iteration serialises the host against the device once per iteration. On this hardware the
# whole solve is launch-bound, not compute-bound (measured: a matvec on 12,896 unknowns
# takes 257 us in float64 and 250 us in float32 -- dtype-independent, i.e. dominated by
# kernel launch, not arithmetic; the same rollout runs 1.11 s on the GPU against 1.25 s on
# the CPU), so those syncs are a first-order cost rather than bookkeeping.
#
# Checking every 25 costs almost nothing in extra iterations because the solves are long:
# median 346 iterations per solve on a heterogeneous fan-scale problem, so the loop
# overshoots by ~4 iterations (350 vs 346) and runs 1.38x faster. Combined with a compiled
# matvec (see ``set_compile_matvec``) the rollout is 1.91x faster, at a head/gradient relative
# L2 difference of ~1e-9 against per-iteration checking -- four orders below the seed
# spread the gate is read against, and the gradient cosine similarity is 1.0000000000.
#
# Applies on CUDA only. On CPU there is no sync to avoid, so `_cg` checks every iteration
# there and exits as early as it can -- deferring the check on CPU is pure waste, which
# matters because calibrate_flow ran on CPU until --device was added.
#
# The safety net is unchanged: the *true* residual is still recomputed from scratch after
# the loop and still warns, so a stalled solve cannot pass silently. Set to 1 to restore
# exact per-iteration checking (bit-reproducible against results recorded before this).
# _CG_TOL is deliberately NOT loosened here: 1e-6 buys 2.49x but moves gradients by
# ~1.5e-07, and this solver feeds a pre-registered gate verdict.
_CG_CHECK_EVERY = 25
_MODEL_DTYPE = torch.float64

# Opt-in `torch.compile` on the conductance matvec. Off by default: it is the one change
# here that alters kernel selection, and compiled results are not bit-reproducible across
# inductor cache states (the same caveat `bench_port.py` records for the UDE's compile
# backend). Measured on a heterogeneous fan-scale rollout it takes the batched-check
# speedup from 1.38x to 1.91x for a ~8 s one-time warmup, at an unchanged head/gradient
# relative L2 difference of ~1e-9 -- i.e. compiling costs no additional accuracy over the
# batched check alone. Enable per-run via `set_compile_matvec(True)` or
# `calibrate_flow --compile-matvec`, which records it in the run's provenance.
_COMPILE_MATVEC = False


def set_compile_matvec(enabled: bool) -> None:
    """Turn the compiled matvec on or off for subsequent solves."""
    global _COMPILE_MATVEC
    _COMPILE_MATVEC = bool(enabled)

# Provenance counters for the published gate number (final-review fix I2): a previous
# run of this exact code produced catastrophically corrupt gradients (median true
# relative residual 4.955e-02, max 2.542e+04) and it was caught only by ad-hoc log
# grepping. A caller that reports a result (calibrate_flow.py's stage3_flow.csv) resets
# these with _reset_cg_stats() at the start of a run and reads them back with
# _cg_stats() at the end, so the artifact carries its own proof that gradients were
# sound. Plain module globals, not a class, so incrementing them costs one branch and
# two comparisons per non-convergent solve -- immeasurable next to a CG iteration --
# and _cg's warnings.warn call is untouched, so operators still see every warning on
# stderr exactly as before.
_CG_NONCONVERGED = 0
_CG_WORST_RESIDUAL = 0.0


def _reset_cg_stats() -> None:
    """Zero the non-convergence counters ``_cg`` increments. Call once at the start of
    a run whose result artifact will report ``cg_nonconverged``/``cg_worst_residual``."""
    global _CG_NONCONVERGED, _CG_WORST_RESIDUAL
    _CG_NONCONVERGED = 0
    _CG_WORST_RESIDUAL = 0.0


def _cg_stats() -> tuple[int, float]:
    """``(count, worst true relative residual)`` over every ``_cg did not converge``
    warning since the last ``_reset_cg_stats()`` call. ``(0, 0.0)`` on a clean run."""
    return _CG_NONCONVERGED, _CG_WORST_RESIDUAL


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


def _warm_started_solver(matvec, diag, x0):
    """Bind ``x0`` (and the operator's diagonal, for Jacobi preconditioning) now, rather
    than via a lambda default-arg trick, so the returned closure always warm-starts from
    the head at the time it was created, not whatever the loop variable holds when the
    closure is later invoked (e.g. by ``_ImplicitSolve.backward`` for the adjoint solve,
    well after the forward loop ends).
    """
    x0 = x0.detach()

    def solve(rhs):
        return _cg(matvec, rhs, diag=diag, x0=x0)

    return solve


def _cg(matvec, b, diag=None, x0=None, tol=_CG_TOL, maxiter=_CG_MAXITER):
    """Matrix-free, Jacobi-preconditioned conjugate gradient for an SPD operator.

    Warns once if the *true* relative residual (recomputed from scratch, not the value
    tracked by the CG recurrence) has not reached ``tol`` within ``maxiter`` iterations.
    The same closure produces both the forward head solve and the adjoint's lambda solve,
    so a stalled solve here silently corrupts the heads *and* every gradient computed from
    them -- this is deliberately not a hard error, since a warned-about near-miss (e.g.
    heads accurate to 1e-6 instead of 1e-8) is often still usable, but it must not pass
    silently.

    ``diag``, if given, is the operator's diagonal (S*area/dt plus the sum of face
    conductances touching each cell); ``z = r / diag`` is applied each iteration as a
    Jacobi preconditioner. This matters because calibration (Task 5) clamps log_T to
    span 5 decades (log(1)..log(1e5)), which is exactly the kind of heterogeneity that
    makes the unpreconditioned operator ill-conditioned and stalls CG well short of
    ``tol`` -- Jacobi is the standard, near-free (one elementwise divide per iteration)
    first remedy. ``diag=None`` falls back to plain (unpreconditioned) CG.

    Two float32-legacy safeguards remain even though the model now runs in float64,
    since ``_cg`` itself is dtype-agnostic and these protect it either way:

    1. The convergence check *inside* the loop uses the cheap recurrence residual
       ``r = r - alpha*Ap``, which can drift from the true residual ``b - matvec(x)``
       under floating-point rounding over many iterations. So every ``_CG_RESTART``
       iterations the recurrence is corrected by recomputing ``r`` from scratch (the
       standard remedy for this drift; costs one extra matvec per restart), rather than
       trusting the incremental update indefinitely.
    2. Independent of restarts, the *warning* always recomputes the true residual after
       the loop rather than reusing whatever the recurrence last reported -- drift means
       the recurrence's own belief about convergence cannot be trusted even right when
       the loop exits.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    b_norm = (b * b).sum().sqrt().clamp(min=1e-30)
    relres = (r * r).sum().sqrt() / b_norm

    def precondition(v):
        return v / diag.clamp(min=1e-30) if diag is not None else v

    # Batching the convergence check only pays where reading the residual costs a device
    # synchronisation -- i.e. on CUDA. On CPU there is no sync to avoid, so deferring the
    # check cannot save anything and can only run iterations past convergence (up to
    # _CG_CHECK_EVERY - 1 of them, on every solve). Gate it on where the tensors actually
    # live rather than applying it globally.
    check_every = _CG_CHECK_EVERY if b.is_cuda else 1
    if relres >= tol:
        z = precondition(r)
        p = z.clone()
        rz = (r * z).sum()
        for i in range(1, maxiter + 1):
            Ap = matvec(p)
            alpha = rz / (p * Ap).sum().clamp(min=1e-30)
            x = x + alpha * p
            r = b - matvec(x) if i % _CG_RESTART == 0 else r - alpha * Ap
            if i % check_every == 0:
                relres = (r * r).sum().sqrt() / b_norm
                if relres < tol:
                    break
            z = precondition(r)
            rz_new = (r * z).sum()
            p = z + (rz_new / rz.clamp(min=1e-30)) * p
            rz = rz_new
    true_relres = (b - matvec(x)).pow(2).sum().sqrt() / b_norm
    if true_relres >= tol:
        global _CG_NONCONVERGED, _CG_WORST_RESIDUAL
        r = float(true_relres)
        _CG_NONCONVERGED += 1
        _CG_WORST_RESIDUAL = max(_CG_WORST_RESIDUAL, r)
        warnings.warn(
            f"_cg did not converge within maxiter={maxiter}: true relative residual "
            f"{r:.3e} exceeds tol={tol:.1e}. Heads and gradients from "
            "this solve may be inaccurate.",
            stacklevel=2,
        )
    return x


class FlowModel(nn.Module):
    """Multi-layer transient flow. ``forward`` returns heads ``(L, A, T+1)``.

    Always operates internally in float64 (see module docstring); the constructor's
    ``device`` argument still controls placement, just not dtype.
    """

    def __init__(self, grid: FanGrid, n_layers: int = 4, dt_days: float = 30.0,
                 device=None, boundaries=None):
        super().__init__()
        self.grid = grid
        self.n_layers = int(n_layers)
        self.dt = float(dt_days)
        self.area = float(grid.dx) ** 2
        A = grid.n_active
        ia, ib = _neighbour_index(grid)
        self.register_buffer("ia", ia.to(device) if device else ia)
        self.register_buffer("ib", ib.to(device) if device else ib)
        z = torch.zeros(self.n_layers, A, device=device, dtype=_MODEL_DTYPE)
        self.log_T = nn.Parameter(z.clone() + float(np.log(500.0)))     # m2/day
        self.log_S = nn.Parameter(z.clone() + float(np.log(1e-4)))      # -
        self.log_L = nn.Parameter(
            torch.zeros(max(self.n_layers - 1, 1), A, device=device, dtype=_MODEL_DTYPE)
            + float(np.log(1e-4))                                       # 1/day
        )
        # Open boundaries (2026-09-11, ``boundaries.py``): general-head cells on the coast
        # and at the apex. ``boundaries=None`` keeps the closed basin every result before
        # that date was computed with, bit for bit -- the boundary term is then absent
        # from the operator, the RHS and the adjoint's parameter tuple alike.
        self.boundaries = boundaries
        if boundaries is not None:
            dev = device
            self.register_buffer("coast_idx", torch.as_tensor(
                boundaries.coast_idx, dtype=torch.long, device=dev))
            self.register_buffer("coast_faces", torch.as_tensor(
                boundaries.coast_faces, dtype=_MODEL_DTYPE, device=dev))
            self.register_buffer("apex_idx", torch.as_tensor(
                boundaries.apex_idx, dtype=torch.long, device=dev))
            self.register_buffer("apex_faces", torch.as_tensor(
                boundaries.apex_faces, dtype=_MODEL_DTYPE, device=dev))
            # Prescribed apex heads, (n_layers, n_apex); set from the initial head field
            # by ``set_apex_heads`` before the first solve. Zeros until then, which a
            # caller that forgets will notice as a coast-like sink at the mountain front.
            self.register_buffer("apex_h", torch.zeros(
                self.n_layers, boundaries.n_apex, device=dev, dtype=_MODEL_DTYPE))
            # Conductance C (m2/day) per exposed face: flux = C * faces * (h_b - h). A
            # Dirichlet face at distance dx/2 from the cell centre would be C = 2T, so the
            # init of 100 m2/day is "open, but not pinned" against T ~ 500 at start.
            self.log_C_coast = nn.Parameter(torch.full(
                (self.n_layers, 1), float(np.log(100.0)), device=dev, dtype=_MODEL_DTYPE))
            self.log_C_apex = nn.Parameter(torch.full(
                (1, 1), float(np.log(100.0)), device=dev, dtype=_MODEL_DTYPE))
        else:
            self.register_parameter("log_C_coast", None)
            self.register_parameter("log_C_apex", None)
        # Opt-in extensions (2026-09-23), all absent by default so every recorded result
        # replays bit for bit: river cells (``set_rivers``) and the calibrated delay-bed
        # fields a fit copies back here the way it copies log_T/log_S/log_L back.
        self.rivers = None
        self.river_layer = 0
        self.river_mode = "none"
        self.delay_log_Sd: torch.Tensor | None = None       # (L, A) after a delay fit
        self.delay_log_tau: torch.Tensor | None = None      # (L, A), days
        # --delay-u0 learned: log of the slow store's initial excess head over h0, (L, A)
        self.delay_log_du0: torch.Tensor | None = None
        # --delay-storage aquitard: log storativity / log conductance of the aquitard
        # store between layers k and k+1, each (L-1, A)
        self.aqt_log_Sa: torch.Tensor | None = None
        self.aqt_log_G: torch.Tensor | None = None
        self.fit_log_C_riv: torch.Tensor | None = None      # (n_groups,)
        # --river-stage-season: (n_groups, 12) monthly connection factor, or None
        self.river_season: torch.Tensor | None = None

    @property
    def has_boundaries(self) -> bool:
        return self.boundaries is not None

    # -----------------------------------------------------------------------------------
    # rivers (rivers.py) and the delay bed: extra positive diagonals on the operator
    # -----------------------------------------------------------------------------------
    @property
    def has_rivers(self) -> bool:
        return self.rivers is not None and self.river_mode in ("ghb", "riv")

    def set_rivers(self, rivers, layer: int = 0, mode: str = "ghb") -> FlowModel:
        """Attach a ``rivers.RiverSet``. ``mode`` is ``"ghb"`` (linear head-dependent
        exchange) or ``"riv"`` (MODFLOW RIV: once the aquifer head falls below the river
        bottom the leakage stops depending on it). ``None`` or mode ``"none"`` detaches."""
        if rivers is None or mode == "none":
            self.rivers, self.river_mode = None, "none"
            return self
        if mode not in ("ghb", "riv"):
            raise ValueError(f"river mode must be 'ghb' or 'riv', got {mode!r}")
        if not 0 <= int(layer) < self.n_layers:
            raise ValueError(f"river layer {layer} outside 0..{self.n_layers - 1}")
        dev = self.log_T.device
        self.rivers = rivers
        self.river_layer = int(layer)
        self.river_mode = mode
        self.n_riv_groups = len(rivers.names)
        self.riv_idx = torch.as_tensor(rivers.idx, dtype=torch.long, device=dev)
        self.riv_w = torch.as_tensor(rivers.weight, dtype=_MODEL_DTYPE, device=dev)
        self.riv_group = torch.as_tensor(rivers.group, dtype=torch.long, device=dev)
        self.riv_h = torch.as_tensor(rivers.h_riv, dtype=_MODEL_DTYPE, device=dev)
        self.riv_rbot = torch.as_tensor(rivers.rbot, dtype=_MODEL_DTYPE, device=dev)
        season = getattr(rivers, "season", None)
        self.river_season = (None if season is None else
                             torch.as_tensor(np.asarray(season), dtype=_MODEL_DTYPE, device=dev))
        return self

    def river_month_factor(self, month: int | None) -> torch.Tensor | None:
        """Per-group connection factor for calendar month index ``month`` (0 = January)
        from ``--river-stage-season``, or ``None`` (factor 1) without a season table."""
        if self.river_season is None or month is None:
            return None
        return self.river_season[:, int(month) % 12]

    def river_mask(self, h: torch.Tensor) -> torch.Tensor | None:
        """The lagged RIV switch: ``True`` where the river cell's aquifer head (layer
        ``river_layer`` of ``h``, the PREVIOUS step's head) is above the river bottom, i.e.
        the river is hydraulically connected. ``None`` in ``ghb`` mode (always connected).
        Detached: the switch is a regime choice, not a differentiable quantity."""
        if self.river_mode != "riv":
            return None
        return (h[self.river_layer, self.riv_idx] > self.riv_rbot).detach()

    def river_terms(self, C_riv: torch.Tensor, mask: torch.Tensor | None = None,
                    factor: torch.Tensor | None = None):
        """``(diag_term, rhs_term)`` of the river cells, each ``(L, A)``.

        ``factor`` (per group, from ``river_month_factor``) multiplies the conductance:
        the seasonal connection of a bed the Jiji weir leaves dry in the dry season.

        Per river cell with conductance ``c = C[group] * weight`` (m2/day):
        connected (``mask`` True, or ``ghb``): flux ``c (h_riv - h)``, so ``+c`` on the
        diagonal and ``c h_riv`` on the right-hand side; disconnected: the constant
        ``c (h_riv - rbot)`` on the right-hand side and nothing on the diagonal. The
        diagonal addition is non-negative, so the operator stays SPD.
        """
        L, A = self.n_layers, self.grid.n_active
        dev = self.log_T.device
        zero = torch.zeros(A, dtype=_MODEL_DTYPE, device=dev)
        c = C_riv.to(dtype=_MODEL_DTYPE)[self.riv_group] * self.riv_w
        if factor is not None:
            c = c * factor.to(dtype=_MODEL_DTYPE)[self.riv_group]
        conn = torch.ones_like(c) if mask is None else mask.to(dtype=_MODEL_DTYPE)
        d_row = zero.index_add(0, self.riv_idx, c * conn)
        r_row = zero.index_add(0, self.riv_idx,
                               c * conn * self.riv_h + c * (1.0 - conn) * (self.riv_h
                                                                          - self.riv_rbot))
        rows_d = [d_row if k == self.river_layer else zero for k in range(L)]
        rows_r = [r_row if k == self.river_layer else zero for k in range(L)]
        return torch.stack(rows_d, dim=0), torch.stack(rows_r, dim=0)

    def delay_beta(self, log_Sd: torch.Tensor, log_tau: torch.Tensor) -> torch.Tensor:
        """Backward-Euler exchange coefficient of the lumped delay bed, ``(L, A)``, 1/day:
        ``beta = S_d / (tau + dt)``. With the slow store's head ``u`` condensed out of the
        step, the aquifer row gains ``beta*area`` on its diagonal and ``beta*area*u`` on
        its right-hand side, and ``u' = (tau u + dt h') / (tau + dt)`` afterwards."""
        return torch.exp(log_Sd) / (torch.exp(log_tau) + self.dt)

    def aqt_terms(self, log_Sa: torch.Tensor, log_G: torch.Tensor):
        """Backward-Euler coefficients of the aquitard store between layers k and k+1
        (``--aquitard-storage``), each ``(L-1, A)``: ``(a, g, c, r)`` with ``a = S_a
        area/dt``, ``g = G area``, ``D = a + 2g``, ``c = g^2/D`` and ``r = g a/D``.

        The store's head ``u`` exchanges ``g (h_k - u)`` with the layer above and ``g
        (h_{k+1} - u)`` with the one below. Condensing ``u' = (a u + g h_k' + g
        h_{k+1}')/D`` out of the step gives the 2x2 block ``[[g-c, -c], [-c, g-c]]`` on
        ``(h_k, h_{k+1})`` and ``r u`` on both right-hand sides. The block's eigenvalues
        are ``g`` and ``g a/D >= 0``, so the operator stays SPD; as ``a -> 0`` it is the
        plain leakance ``g/2``, and as ``a`` grows the interface stores water instead of
        passing it. It acts in parallel with ``log_L`` (which the fit can take to its
        floor), so no historical parameter drops out of the operator.
        """
        a = torch.exp(log_Sa) * self.area / self.dt
        g = torch.exp(log_G) * self.area
        D = a + 2.0 * g
        return a, g, g * g / D, g * a / D

    _LAYOUT_ORDER = ("bnd", "delay", "riv", "aqt")

    def operator_layout(self, bnd: bool = False, delay: bool = False,
                        riv: bool = False, aqt: bool = False) -> tuple[str, ...]:
        """Which optional parameter groups follow ``(log_T, log_S[, log_L])`` in the
        tuple ``make_op``'s operator unpacks, in their one fixed order."""
        flags = {"bnd": bnd, "delay": delay, "riv": riv, "aqt": aqt}
        return tuple(g for g in self._LAYOUT_ORDER if flags[g])

    def extra_diag(self, layout: tuple[str, ...], groups: dict,
                   riv_mask: torch.Tensor | None = None,
                   riv_factor: torch.Tensor | None = None) -> torch.Tensor | None:
        """Sum of the optional diagonal terms for ``layout``; ``groups`` maps each group
        name to its tuple of LOG parameters. ``None`` when ``layout`` is empty."""
        extra = None
        for g in layout:
            if g == "bnd":
                lc, la = groups["bnd"]
                d, _ = self.boundary_terms(torch.exp(lc), torch.exp(la))
            elif g == "delay":
                ls, lt = groups["delay"]
                d = self.delay_beta(ls, lt) * self.area
            elif g == "riv":
                (lr,) = groups["riv"]
                d, _ = self.river_terms(torch.exp(lr), riv_mask, riv_factor)
            elif g == "aqt":
                continue          # not diagonal-only: _matvec_from takes it as ``aqt``
            else:
                raise ValueError(f"unknown operator group {g!r}")
            extra = d if extra is None else extra + d
        return extra

    def make_op(self, layout: tuple[str, ...], riv_mask: torch.Tensor | None = None,
                riv_factor: torch.Tensor | None = None):
        """A differentiable ``op(h, *params) = M(params) @ h`` for ``_ImplicitSolve``'s
        backward, for any combination of the optional groups. ``params`` unpacks as
        ``log_T, log_S, [log_L]`` then, for each group in ``layout`` (always in
        ``_LAYOUT_ORDER``): ``bnd`` -> ``(log_C_coast, log_C_apex)``, ``delay`` ->
        ``(log_Sd, log_tau)`` each ``(L, A)``, ``riv`` -> ``(log_C_riv,)``, ``aqt`` ->
        ``(log_Sa, log_G)`` each ``(L-1, A)``. Every group but ``aqt`` adds only to the
        positive diagonal; ``aqt`` adds SPD 2x2 interface blocks (``aqt_terms``), so M
        stays SPD. ``riv_mask``/``riv_factor`` are the RIV connection switch and the
        seasonal factor the forward solve used; the op must be rebuilt when they change.
        """
        layout = tuple(layout)
        if list(layout) != [g for g in self._LAYOUT_ORDER if g in layout]:
            raise ValueError(f"layout {layout} is not in the fixed order {self._LAYOUT_ORDER}")
        sizes = {"bnd": 2, "delay": 2, "riv": 1, "aqt": 2}

        def op(h, *params):
            it = iter(params)
            log_T = next(it)
            log_S = next(it)
            log_L = next(it) if self.n_layers > 1 else None
            groups = {g: tuple(next(it) for _ in range(sizes[g])) for g in layout}
            extra = self.extra_diag(layout, groups, riv_mask, riv_factor)
            aqt = None
            if "aqt" in groups:
                _, g_, c_, _ = self.aqt_terms(*groups["aqt"])
                aqt = (g_, c_)
            mv, _ = self._matvec_from(torch.exp(log_T), torch.exp(log_S),
                                      torch.exp(log_L) if log_L is not None else None,
                                      bdiag=extra, compile_ok=False, aqt=aqt)
            return mv(h)

        return op

    def set_apex_heads(self, h0: torch.Tensor) -> FlowModel:
        """Prescribe the apex boundary head from an initial head field ``(n_layers, A)``.

        The mountain front is held at the initial IDW head for the whole run. In a k-fold
        gate ``h0`` is the fold's own kept-wells field, so a held-out well never reaches
        the boundary it is later scored against.
        """
        if not self.has_boundaries:
            return self
        with torch.no_grad():
            src = h0.to(dtype=_MODEL_DTYPE, device=self.apex_h.device)
            self.apex_h.copy_(src[:, self.apex_idx])
        return self

    def boundary_terms(self, C_coast: torch.Tensor | None, C_apex: torch.Tensor | None):
        """``(diag_term, rhs_term)`` of the general-head boundaries, each ``(L, A)``.

        ``diag_term`` adds ``C * faces`` on every boundary cell's own head (it enters the
        SPD operator; a positive diagonal addition keeps it SPD) and ``rhs_term`` carries
        ``C * faces * h_b`` -- zero on the coast, since sea level is the datum. Both are
        zero tensors when the model has no boundaries, so callers can add them blindly.
        """
        L, A = self.n_layers, self.grid.n_active
        dev = self.log_T.device
        zero = torch.zeros(L, A, dtype=_MODEL_DTYPE, device=dev)
        if not self.has_boundaries or C_coast is None or C_apex is None:
            return zero, zero
        cc = C_coast.to(dtype=_MODEL_DTYPE) * self.coast_faces[None, :]      # (L, n_coast)
        ca = C_apex.to(dtype=_MODEL_DTYPE) * self.apex_faces[None, :]        # (1, n_apex)
        ca = ca.expand(L, -1)
        diag = zero.index_add(1, self.coast_idx, cc)
        diag = diag.index_add(1, self.apex_idx, ca)
        rhs = zero.index_add(1, self.apex_idx, ca * self.apex_h)
        if COAST_HEAD_M != 0.0:
            rhs = rhs.index_add(1, self.coast_idx, cc * COAST_HEAD_M)
        return diag, rhs

    def _C_from_params(self):
        """``(C_coast, C_apex)`` from the model's own registered conductances, or Nones."""
        if not self.has_boundaries:
            return None, None
        return torch.exp(self.log_C_coast), torch.exp(self.log_C_apex)

    def _matvec_from(self, T, S, L=None, bdiag=None, compile_ok: bool = True, aqt=None):
        """Return ``(mv, diag)``: ``mv`` applies (S*area/dt + K + leakage) to a head
        vector of shape (L, A); ``diag`` is that operator's diagonal (S*area/dt, plus
        the sum of face conductances touching each cell, plus the leakances touching
        each layer), used as a Jacobi preconditioner in _cg.

        ``L`` is the leakance (1/day) between layer k and k+1, shape (n_layers-1, A),
        or ``None`` when there is only one layer (nothing to couple). Leakage acts on
        the *vertical* head difference between adjacent layers, independently of the
        horizontal face conductances above. These are LHS operator coefficients
        (``M @ h``), not a physical mass flux -- with ``inter = Lk*(h_k - h_{k+1})``,
        row k of the operator gains ``+inter`` and row k+1 gains ``-inter``, i.e. each
        row picks up ``+Lk`` on its own (diagonal) head and ``-Lk`` on its neighbour's
        (off-diagonal) head. This mirrors the ``ia``/``ib`` convention already used
        for the horizontal conductance term just above (``index_add(..., flux)`` /
        ``index_add(..., -flux)``), and is what keeps the operator symmetric: the
        code is the source of truth here, verified symmetric and SPD by assembling
        the dense operator on a small grid. Written as an explicit +/- accumulation
        into a zeros_like buffer (rather than in place on ``out``/``diag`` directly)
        so it reads the same for any n_layers and stays out-of-place for autograd.

        ``aqt`` (``(g, c)`` from ``aqt_terms``, each ``(L-1, A)``, or ``None``) adds the
        aquitard-store blocks ``[[g-c, -c], [-c, g-c]]`` on every interface.
        """
        ia, ib = self.ia, self.ib
        Tf = 2.0 * T[:, ia] * T[:, ib] / (T[:, ia] + T[:, ib]).clamp(min=1e-30)  # harmonic
        # General-head boundaries (``bdiag``, from ``boundary_terms``): a per-cell
        # conductance on the diagonal only. Positive, so SPD is preserved; ``None`` or
        # zeros reproduces the closed basin.
        stor = S * self.area / self.dt
        if bdiag is not None:
            stor = stor + bdiag

        def mv(h):
            out = stor * h
            dh = h[:, ia] - h[:, ib]
            flux = Tf * dh
            out = out.index_add(1, ia, flux)
            out = out.index_add(1, ib, -flux)
            if self.n_layers > 1:
                # Leakance L (1/day) between layer k and k+1, acting on the head
                # difference. +Lk to both diagonal entries, -Lk to both off-diagonal
                # entries -- symmetric, so the operator stays SPD.
                Lk = L * self.area                              # (L-1, A)
                inter = Lk * (h[:-1] - h[1:])                    # downward-positive flux
                lay = torch.zeros_like(out)
                lay[:-1] = lay[:-1] + inter
                lay[1:] = lay[1:] - inter
                out = out + lay
            if aqt is not None:
                g_a, c_a = aqt
                s_a = c_a * (h[:-1] + h[1:])
                lay = torch.zeros_like(out)
                lay[:-1] = lay[:-1] + g_a * h[:-1] - s_a
                lay[1:] = lay[1:] + g_a * h[1:] - s_a
                out = out + lay
            return out

        diag = stor
        diag = diag.index_add(1, ia, Tf)
        diag = diag.index_add(1, ib, Tf)
        if self.n_layers > 1:
            Lk = L * self.area
            diagL = torch.zeros_like(diag)
            diagL[:-1] = diagL[:-1] + Lk
            diagL[1:] = diagL[1:] + Lk
            diag = diag + diagL
        if aqt is not None:
            g_a, c_a = aqt
            diagA = torch.zeros_like(diag)
            diagA[:-1] = diagA[:-1] + (g_a - c_a)
            diagA[1:] = diagA[1:] + (g_a - c_a)
            diag = diag + diagA
        if _COMPILE_MATVEC and compile_ok:
            mv = torch.compile(mv, dynamic=False)
        return mv, diag

    def _op(self, h, *params):
        """Differentiable M(params) @ h, used only by the adjoint's backward.

        ``params`` is whatever ``operator_params`` produced: ``(log_T, log_S)``, plus
        ``log_L`` when there is more than one layer, plus ``(log_C_coast, log_C_apex)``
        when the model has open boundaries. Only parameters that actually enter the
        operator are ever in the tuple, because ``_ImplicitSolve.backward`` refuses one
        that drops out.
        """
        it = iter(params)
        log_T = next(it)
        log_S = next(it)
        log_L = next(it) if self.n_layers > 1 else None
        rest = tuple(it)
        T = torch.exp(log_T)
        S = torch.exp(log_S)
        L = torch.exp(log_L) if log_L is not None else None
        bdiag = None
        if rest:
            log_C_coast, log_C_apex = rest
            bdiag, _ = self.boundary_terms(torch.exp(log_C_coast), torch.exp(log_C_apex))
        mv, _ = self._matvec_from(T, S, L, bdiag=bdiag)
        return mv(h)

    def operator_params(self, log_T, log_S, log_L=None, log_C_coast=None,
                        log_C_apex=None, delay=None, riv=None, aqt=None) -> tuple:
        """The tuple ``_ImplicitSolve.apply`` gets, in ``_op``'s (and ``make_op``'s)
        unpacking order. ``delay`` is ``(log_Sd, log_tau)``, ``riv`` is ``(log_C_riv,)``
        and ``aqt`` is ``(log_Sa, log_G)``; all default to absent, which is the
        historical tuple."""
        params = [log_T, log_S]
        if self.n_layers > 1:
            params.append(log_L)
        if log_C_coast is not None and log_C_apex is not None:
            params += [log_C_coast, log_C_apex]
        if delay is not None:
            params += list(delay)
        if riv is not None:
            params += list(riv)
        if aqt is not None:
            params += list(aqt)
        return tuple(params)

    def forward(self, h0: torch.Tensor, recharge: torch.Tensor,
                pumping: torch.Tensor, n_steps: int) -> torch.Tensor:
        h0 = h0.to(dtype=_MODEL_DTYPE)
        recharge = recharge.to(dtype=_MODEL_DTYPE)
        pumping = pumping.to(dtype=_MODEL_DTYPE)

        T = torch.exp(self.log_T)
        S = torch.exp(self.log_S)
        L = torch.exp(self.log_L) if self.n_layers > 1 else None
        C_coast, C_apex = self._C_from_params()
        bdiag, brhs = self.boundary_terms(C_coast, C_apex)
        params = self.operator_params(self.log_T, self.log_S,
                                      self.log_L if self.n_layers > 1 else None,
                                      self.log_C_coast, self.log_C_apex)
        mv, diag = self._matvec_from(T, S, L, bdiag=bdiag)
        h = h0
        out = [h0]
        for t in range(n_steps):
            q = recharge[..., t] * self.area - pumping[..., t]
            b = S * self.area / self.dt * h + q + brhs
            # Warm-start CG from the previous head: consecutive backward-Euler steps have
            # similar solutions, so this is free and cuts iterations substantially.
            solve = _warm_started_solver(mv, diag, h)
            h = _ImplicitSolve.apply(b, self._op, solve, *params)
            out.append(h)
        return torch.stack(out, dim=-1)
