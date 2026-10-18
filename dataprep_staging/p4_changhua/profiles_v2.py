"""P4.0 repair round: converged bound-pinned profiles of the level loss (in-sample diagnostics).

engine/v1's profiles were chord Gauss-Newton steps with a Jacobian frozen at theta-hat, four
iterations and a 0.5 trust box per iteration; the reference refit and most profile points
stopped at the iteration cap, so profile differences partly measured how far each refit got
(verifier, minor issue). This module replaces them with:

1. a **reference refit** of the interior entries (bound-pinned entries held at their bounds)
   by bounded Gauss-Newton with a central-difference Jacobian recomputed at every accepted
   iterate, run until the relative decrease of the objective falls below ``tol`` (or the line
   search fails, or ``n_iter`` is reached; the stop reason is recorded);
2. the Jacobian at the converged reference (Fisher information and the predicted remaining
   decrease, i.e. how stationary the reference is);
3. **local profiles**: each pinned entry moved inside / beyond its bound, interior entries
   re-optimised from the converged reference by chord Gauss-Newton with that Jacobian, again
   to the same relative tolerance.

The objective is ``fit_flow``'s level loss with the per-well datum profiled at fixed
``kappa`` (sum of squared residual rows plus ``kappa d^2`` rows); the caller supplies the
residual function, for back-filled targets (``fit_flow``'s own convention) or observed months
only (plan U2). Everything here is in-sample (the theta's fit window 2012-22) and says
nothing about out-of-sample skill.
"""
from __future__ import annotations

import numpy as np


def central_jacobian(eval_res, x: np.ndarray, step: float = 1e-3) -> tuple[np.ndarray, int]:
    """Central-difference Jacobian of the residual vector ``eval_res(x)`` -> ``(J, n_eval)``."""
    cols = []
    for j in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp[j] += step
        xm[j] -= step
        cols.append((eval_res(xp) - eval_res(xm)) / (2.0 * step))
    return np.stack(cols, axis=1), 2 * x.size


def _bounded_step(J, r, x, lo, hi, trust):
    from scipy.optimize import lsq_linear

    lb = np.maximum(lo - x, -trust)
    ub = np.minimum(hi - x, trust)
    return lsq_linear(J, -r, bounds=(lb, ub), method="bvls").x


def predicted_remaining(J, r, x, lo, hi, trust=np.inf) -> float:
    """Relative decrease of ``|r|^2`` a bounded Gauss-Newton step from ``x`` predicts (0 at a
    bounded stationary point of the linearised problem)."""
    dx = _bounded_step(J, r, x, lo, hi, trust)
    obj = float(r @ r)
    return float((obj - np.sum((r + J @ dx) ** 2)) / obj)


def gn_refit(eval_res, x0: np.ndarray, lo: np.ndarray, hi: np.ndarray, *,
             J_fixed: np.ndarray | None = None, step: float = 1e-3, n_iter: int = 15,
             trust: float = 0.5, tol: float = 1e-6, halvings: int = 4) -> dict:
    """Bounded Gauss-Newton on ``|eval_res(x)|^2`` from ``x0`` inside ``[lo, hi]``.

    ``J_fixed`` given: chord iterations with that Jacobian; otherwise a central-difference
    Jacobian at every accepted iterate. Each step is the bounded linear least-squares step
    within a per-iteration box of ``trust``, with up to ``halvings`` halvings until the
    objective decreases. Stops when the relative decrease of an accepted step is below
    ``tol`` (``converged``), when no halving decreases it (``no_decrease``), or after
    ``n_iter`` iterations (``max_iter``)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    r = eval_res(x)
    obj = float(r @ r)
    n_eval, hist, reason, last_box = 1, [obj], "max_iter", False
    J = J_fixed
    for _ in range(n_iter):
        if J_fixed is None:
            J, ne = central_jacobian(eval_res, x, step)
            n_eval += ne
        dx = _bounded_step(J, r, x, lo, hi, trust)
        accepted = False
        for _h in range(halvings):
            rt = eval_res(x + dx)
            n_eval += 1
            ot = float(rt @ rt)
            if ot < obj:
                accepted = True
                break
            dx = dx / 2.0
        if not accepted:
            reason = "no_decrease"
            break
        last_box = bool(np.abs(dx).max() >= trust - 1e-9)
        rel = (obj - ot) / obj
        x, r, obj = x + dx, rt, ot
        hist.append(obj)
        if rel < tol:
            reason = "converged"
            break
    return {"x": x, "res": r, "obj": obj, "n_iter": len(hist) - 1, "n_eval": n_eval,
            "reason": reason, "history": hist, "last_step_hit_trust_box": last_box}


def lag1_autocorr(res: np.ndarray, mask: np.ndarray) -> dict:
    """Pooled lag-1 autocorrelation of well residuals ``res`` ``(W, T)`` over pairs of
    consecutive observed months (each well's mean over its observed months removed)."""
    m = np.asarray(mask, bool)
    num = den = 0.0
    n_pairs = 0
    for w in range(res.shape[0]):
        rw = np.where(m[w], res[w], np.nan)
        if np.isfinite(rw).sum() < 3:
            continue
        rw = rw - np.nanmean(rw)
        a, b = rw[:-1], rw[1:]
        ok = np.isfinite(a) & np.isfinite(b)
        num += float((a[ok] * b[ok]).sum())
        den += float(np.nansum(rw ** 2))
        n_pairs += int(ok.sum())
    phi = num / den if den > 0 else float("nan")
    return {"phi": phi, "n_pairs": n_pairs,
            "variance_factor_ar1": (1 + phi) / (1 - phi),
            "sd_factor_ar1": float(np.sqrt((1 + phi) / (1 - phi)))}
