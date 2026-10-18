"""P4.0 repair round: the tangent-linear (lift-feedback-aware) linearisation of the sparse twin.

Why this module exists (verifier, P4.0 repair round, 2026-10-12). The stored kernels
(``engine.adjoint_greens`` / ``engine.pump_kernel``) and ``engine.energy_rate_jacobian`` give
the **frozen-lift** response: the abstraction rate of every step is converted with the head of
the base path, ``rate_t = c(h_t) E_t`` with ``c(h) = eta J / (rho g max(z_g - h + h_x, 2 m))
/ dt``. In the twin the lift moves with the perturbed heads, and that feedback is **first
order**, not second order as the docstring of ``engine.energy_rate_jacobian`` says:

    d rate_t = c(h_t) dE_t + c'(h_t) E_t dh_t,        c'(h) = c(h) / (z_g - h + h_x)

(zero where the lift is clipped at 2 m). The second term acts on the *whole* energy of the
path (base census plus any added abstraction), so it changes the response to every
perturbation, volumetric ones included. ``engine.py`` is left unchanged because the manifest
of ``changhua_abstraction/engine/v1`` hashes it; this module carries the correction.

* :func:`rate_factor_dh` - ``c'(h)`` (m3/day per kWh per m of head).
* :func:`tlm_rollout` - the exact first-order (tangent-linear) response of
  ``SparseTwin.rollout`` to perturbations of the spread energy, of the abstraction rate or of
  direct volumetric sources, batched over columns (one multi-right-hand-side LU solve per
  step). ``frozen=True`` drops the feedback term and reproduces the kernel (frozen-lift)
  response.
* :func:`tlm_factor_responses` - head responses ``U (J, W, T)`` of factor fields, so that
  ``Var h = diag(U C U^T)`` (``engine.head_var_factors``) is the feedback-aware
  ``J Sigma J^T`` for factor-structured forcing covariances (plan 4.8).
* :func:`tlm_draw_responses` - the same for independent draws (Monte Carlo through the
  tangent-linear model, e.g. independent cell-month noise, which has too many factors).

Rivers (RIV switching) are refused: the response is not differentiable at a switch; use the
stepwise Monte Carlo there (plan 4.8).
"""
from __future__ import annotations

import sys

import numpy as np

if __package__ in (None, ""):
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from p4_changhua import engine  # noqa: E402

__version__ = "0.2.0"   # P4.0 repair round, 2026-10-12


def _lift(twin: engine.SparseTwin, h_pump: np.ndarray) -> np.ndarray:
    head = twin.ge - np.asarray(h_pump, dtype=np.float64)
    if twin.hx is not None:
        head = head + twin.hx
    return head


def rate_factor_dh(twin: engine.SparseTwin, h_pump: np.ndarray) -> np.ndarray:
    """``d c / d h`` at pump-layer head ``h_pump`` ``(A,)``: ``c(h) / lift`` where the lift
    exceeds ``MIN_LIFT_M`` and 0 where it is clipped (one-sided at the clip)."""
    lift = _lift(twin, h_pump)
    c = twin.rate_factor(h_pump)
    return np.where(lift > engine.MIN_LIFT_M, c / np.maximum(lift, engine.MIN_LIFT_M), 0.0)


def clipped_lift_count(twin: engine.SparseTwin, H: np.ndarray) -> dict:
    """Cell-steps of the path ``H`` ``(L, A, T+1)`` whose lift is at or below the 2 m clip
    (where the feedback derivative is zero and the response is not differentiable)."""
    pl = twin.pump_layer
    lift = np.stack([_lift(twin, H[pl, :, t]) for t in range(H.shape[-1] - 1)], axis=1)
    return {"n_clipped": int((lift <= engine.MIN_LIFT_M).sum()), "n": int(lift.size),
            "min_lift_m": float(lift.min()), "median_lift_m": float(np.median(lift))}


def _cols(x, shape2: tuple[int, int] | None = None):
    """Perturbation array with a trailing column axis (adds one if absent)."""
    if x is None:
        return None, None
    x = np.asarray(x, dtype=np.float64)
    if shape2 is not None and x.ndim == len(shape2):
        return x[..., None], True
    return x, False


def tlm_rollout(twin: engine.SparseTwin, H: np.ndarray, Es: np.ndarray | None, *,
                dEs=None, drate=None, dq=None, frozen: bool = False,
                obs: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Tangent-linear response of :meth:`engine.SparseTwin.rollout` about the path ``H``.

    ``H`` ``(L, A, T+1)`` is the path the rollout produced with total spread energy ``Es``
    ``(A, T)`` (base plus any added abstraction, kWh/month after the spread kernel; ``None``
    if the path has no energy forcing). Perturbations (each ``(..., T)`` or ``(..., T, J)``
    for ``J`` columns): ``dEs`` ``(A, T[, J])`` spread energy (kWh/month), ``drate``
    ``(A, T[, J])`` abstraction rate (m3/day, + = more abstraction; passes through the pump
    split and the return flow), ``dq`` ``(L, A, T[, J])`` direct volumetric sources (m3/day,
    + = injection). Step ``t`` converts energy with the head at its start, ``H[pl, :, t]``,
    exactly as the rollout does, and the feedback term ``c'(h_t) Es_t dh_t`` is included
    unless ``frozen=True`` (then the response is the frozen-lift one the kernels give).

    Returns head changes (m, + = rise): ``(L, A, T+1[, J])``, or ``(W, T[, J])`` at the
    observation wells for steps 1..T when ``obs=(obs_layer, obs_idx)``."""
    if twin.has_rivers:
        raise NotImplementedError("RIV switching is not differentiable; use the stepwise "
                                  "Monte Carlo for river configurations")
    L, A = twin.n_layers, twin.n_active
    T = H.shape[-1] - 1
    dEs, s1 = _cols(dEs, (A, T))
    drate, s2 = _cols(drate, (A, T))
    dq, s3 = _cols(dq, (L, A, T))
    squeeze = [s for s in (s1, s2, s3) if s is not None]
    if not squeeze:
        raise ValueError("no perturbation given")
    if len(set(squeeze)) != 1:
        raise ValueError("perturbations must all have, or all lack, the column axis")
    squeeze = squeeze[0]
    J = next(x.shape[-1] for x in (dEs, drate, dq) if x is not None)
    for x in (dEs, drate, dq):
        if x is not None and (x.shape[-1] != J or x.shape[-2] != T):
            raise ValueError("perturbation shapes disagree with the path or with each other")
    Es = None if Es is None else np.asarray(Es, dtype=np.float64)
    if (Es is not None or dEs is not None) and twin.eta is None:
        raise ValueError("energy forcing or perturbations need log_eta in the scalars")
    pl = twin.pump_layer
    wts = twin.pump_weights()
    lu = twin.lu()
    D = twin.op.stor.reshape(-1)[:, None]
    dh = np.zeros((L * A, J))
    if obs is None:
        out = np.zeros((L, A, T + 1, J))
    else:
        ol, oi = (np.asarray(v) for v in obs)
        rows = ol * A + oi
        out = np.zeros((rows.size, T, J))
    for t in range(T):
        hp = H[pl, :, t]
        dr = np.zeros((A, J))
        if dEs is not None:
            dr += twin.rate_factor(hp)[:, None] * dEs[:, t, :]
        if drate is not None:
            dr += drate[:, t, :]
        if not frozen and Es is not None:
            dr += (rate_factor_dh(twin, hp) * Es[:, t])[:, None] * dh[pl * A:(pl + 1) * A]
        q = np.zeros((L, A, J))
        for lay, w in wts.items():
            q[lay] += w * dr
        if dq is not None:
            q += dq[:, :, t, :]
        dh = lu.solve(D * dh + q.reshape(L * A, J))
        if dh.ndim == 1:
            dh = dh[:, None]
        if obs is None:
            out[:, :, t + 1, :] = dh.reshape(L, A, J)
        else:
            out[:, t, :] = dh[rows]
    return out[..., 0] if squeeze else out


def tlm_factor_responses(twin: engine.SparseTwin, H: np.ndarray, Es: np.ndarray | None,
                         F: np.ndarray, kind: str, obs, *, frozen: bool = False,
                         chunk: int = 64) -> np.ndarray:
    """Head responses ``U`` ``(J, W, T)`` (m, + = rise) of factor fields ``F`` ``(J, A, T)``
    (``kind='energy'``: spread kWh/month; ``'rate'``: m3/day abstraction) through the
    tangent-linear model; ``Var h = diag(U C U^T)`` for ``xi ~ N(0, C)`` and the forcing
    error ``sum_j F_j xi_j`` (``engine.head_var_factors``)."""
    if kind not in ("energy", "rate"):
        raise ValueError("kind is 'energy' or 'rate'")
    F = np.asarray(F, dtype=np.float64)
    out = []
    for s in range(0, F.shape[0], chunk):
        blk = np.moveaxis(F[s:s + chunk], 0, -1)                  # (A, T, j)
        kw = {"dEs": blk} if kind == "energy" else {"drate": blk}
        out.append(np.moveaxis(tlm_rollout(twin, H, Es, frozen=frozen, obs=obs, **kw), -1, 0))
    return np.concatenate(out, axis=0)


def tlm_draw_responses(twin: engine.SparseTwin, H: np.ndarray, Es: np.ndarray | None,
                       draw, n_draws: int, kind: str, obs, *, frozen: bool = False,
                       chunk: int = 50):
    """Monte Carlo through the tangent-linear model: ``draw(i)`` returns draw ``i`` of the
    perturbation (``(A, T)`` for ``kind`` 'energy' or 'rate', ``(L, A, T)`` for 'source').
    Yields ``(i0, R)`` with ``R`` ``(W, T, j)`` the responses of draws ``i0 .. i0 + j - 1``
    (draws regenerated by index, so a stepwise Monte Carlo can use the very same draws)."""
    key = {"energy": "dEs", "rate": "drate", "source": "dq"}[kind]
    for s in range(0, n_draws, chunk):
        blk = np.stack([draw(i) for i in range(s, min(s + chunk, n_draws))], axis=-1)
        yield s, tlm_rollout(twin, H, Es, frozen=frozen, obs=obs, **{key: blk})
