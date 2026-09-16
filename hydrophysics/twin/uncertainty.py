"""A posterior for the flow parameters, not just a cross-validation spread.

    python -m hydrophysics.twin.uncertainty --theta <stage3_theta.json> --n-samples 20 \\
        --out results/twin_runs/<run>/stage3_posterior.json

Laplace approximation around the calibrated parameter vector. With the head misfit
Gaussian, the posterior is approximately

    theta ~ N(theta_hat, sigma^2 (J^T J + lambda I)^-1)

where ``J`` is the Jacobian of every observed head with respect to the free parameters,
``sigma^2`` the residual variance of the fit, and ``lambda`` a weak prior that keeps the
unidentified directions finite. ``J`` (~21,000 observations x ~32 parameters) is built
column by column with central finite differences through the rebuilt model: two rollouts
per parameter, a few seconds each on the GPU (the implicit solve defines only a
reverse-mode adjoint, so forward-mode autodiff is not available).

What this is and is not. It is a linearised, local posterior: it says how much the data
constrain each parameter *around the fit*, and it draws samples that the forward twin can
run as members (``twin.forward --theta <posterior.json>``). It is not a full Bayesian
calibration -- a parameter pinned at a bound has a truncated posterior that this cannot
represent, and the samples are clipped to the bounds. Parameters held fixed
(``fix_eta``) or at their bounds are reported as such rather than sampled.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..train import pick_device
from .calibrate_flow import BOUNDS, _base_param_name, _r2, set_compile_matvec
from .forward import Member, build_model, load_members, rollout
from .inputs import load_twin_inputs

# keys in a theta dict that are derived, not parameters
_DERIVED = {"eta", "head_extra_m", "recharge_frac", "C_coast_m2day", "C_apex_m2day",
            "pump_frac_shallow", "return_frac"}


def flatten(theta: dict, fixed: tuple[str, ...] = ()) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """theta dict -> (vector, index) over free parameters, skipping derived and fixed."""
    vec, index = [], []
    for k, v in theta.items():
        if k in _DERIVED or k in fixed:
            continue
        arr = np.atleast_1d(np.asarray(v, dtype="float64")).reshape(-1)
        for i, x in enumerate(arr):
            vec.append(float(x))
            index.append((k, i))
    return np.array(vec), index


def unflatten(vec: np.ndarray, index: list[tuple[str, int]], template: dict) -> dict:
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in template.items()}
    for x, (k, i) in zip(vec, index, strict=True):
        if isinstance(out[k], list):
            out[k][i] = float(x)
        else:
            out[k] = float(x)
    return out


def clip_to_bounds(vec: np.ndarray, index: list[tuple[str, int]]) -> np.ndarray:
    out = vec.copy()
    for j, (k, _) in enumerate(index):
        base = _base_param_name(k)
        if base in BOUNDS:
            lo, hi = BOUNDS[base]
            out[j] = min(max(out[j], lo), hi)
    return out


def predict_at_wells(inp, member: Member, device) -> np.ndarray:
    """Rebuild the model from ``member.theta`` and return heads at the wells, (W, T-1)."""
    model, scalars, _ = build_model(inp.grid, member, device)
    eta_classes = member.meta.get("eta_classes")
    if eta_classes:
        E = torch.tensor(np.stack([inp.E_by_class[c] for c in eta_classes]),
                         dtype=torch.float64)[..., 1:]
    else:
        E = inp.E_total[:, 1:]
    h = rollout(model, scalars, inp.initial_heads(0), E, inp.recharge_field[:, 1:],
                inp.ground_elev)
    return h[torch.as_tensor(inp.obs_layer, device=h.device),
             torch.as_tensor(inp.obs_idx, device=h.device), 1:].cpu().numpy()


def jacobian(inp, member: Member, index, device, eps: float = 1e-3,
             log=print) -> tuple[np.ndarray, np.ndarray]:
    """``(J, residual)``: J is (n_obs, n_params) by central finite differences in the
    log-parameters (two rollouts per parameter, a few seconds each on the GPU); residual
    is pred - obs at the fit. Finite differences rather than forward-mode autodiff because
    the implicit solve defines only a reverse-mode adjoint."""
    obs = inp.obs_h_filled[:, 1:]
    vec = np.array([np.atleast_1d(np.asarray(member.theta[k], dtype="float64")).reshape(-1)[i]
                    for k, i in index])
    t0 = time.perf_counter()
    base = predict_at_wells(inp, member, device)
    resid = (base - obs).reshape(-1)
    log(f"  base prediction R2 {_r2(base, obs):+.4f} ({time.perf_counter() - t0:.1f}s)")
    cols = []
    for j in range(len(index)):
        plus = vec.copy()
        plus[j] += eps
        minus = vec.copy()
        minus[j] -= eps
        p_plus = predict_at_wells(inp, Member(member.label, unflatten(plus, index, member.theta),
                                              member.meta), device)
        p_minus = predict_at_wells(inp, Member(member.label, unflatten(minus, index, member.theta),
                                               member.meta), device)
        cols.append(((p_plus - p_minus) / (2 * eps)).reshape(-1))
        if (j + 1) % 8 == 0 or j == len(index) - 1:
            log(f"  jacobian column {j + 1}/{len(index)} ({time.perf_counter() - t0:.0f}s)")
    return np.stack(cols, axis=1), resid


def laplace(J: np.ndarray, resid: np.ndarray, prior_sd: float = 2.0) -> tuple[np.ndarray, float]:
    """Posterior covariance ``sigma^2 (J^T J / sigma^2 ... )`` with a Gaussian prior of
    width ``prior_sd`` in log-parameter units on every direction."""
    n, p = J.shape
    sigma2 = float((resid ** 2).sum() / max(n - p, 1))
    H = J.T @ J / sigma2 + np.eye(p) / prior_sd ** 2
    cov = np.linalg.inv(H)
    return cov, sigma2


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Laplace posterior of the calibrated flow parameters")
    ap.add_argument("--theta", required=True)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--prior-sd", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile-matvec", action="store_true")
    ap.add_argument("--out", required=True, help="posterior json: samples usable as --theta members")
    args = ap.parse_args(argv)

    set_compile_matvec(args.compile_matvec)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    member = load_members([args.theta])[0]
    fixed = tuple(member.meta.get("fixed") or ())
    if member.meta.get("fix_eta") is not None:
        fixed = fixed + ("log_eta",)
    if member.meta.get("fix_head_extra") is not None:
        fixed = fixed + ("log_head_extra",)
    vec, index = flatten(member.theta, fixed=fixed)
    # parameters sitting on a bound have a truncated posterior the Laplace form cannot
    # hold; report them and keep them where they are
    at_bound = []
    for j, (k, _) in enumerate(index):
        base = _base_param_name(k)
        if base in BOUNDS and (abs(vec[j] - BOUNDS[base][0]) < 1e-9 or abs(vec[j] - BOUNDS[base][1]) < 1e-9):
            at_bound.append(j)
    print(f"free parameters {len(index)}, fixed {list(fixed)}, at a bound "
          f"{[f'{index[j][0]}[{index[j][1]}]' for j in at_bound]}", flush=True)

    inp = load_twin_inputs(dx=args.dx, meter_filter=member.meta.get("meter_filter", "none"),
                           cap_duty=float(member.meta.get("cap_duty", 1.0)))
    J, resid = jacobian(inp, member, index, device)
    cov, sigma2 = laplace(J, resid, prior_sd=args.prior_sd)
    sd = np.sqrt(np.diag(cov))
    print(f"residual sd {np.sqrt(sigma2):.3f} m; posterior sd per parameter:")
    for j, (k, i) in enumerate(index):
        flag = "  (at bound, not sampled)" if j in at_bound else ""
        print(f"  {k}[{i}]: {vec[j]:+.3f} ± {sd[j]:.3f}{flag}")

    rng = np.random.default_rng(args.seed)
    free = [j for j in range(len(index)) if j not in at_bound]
    L = np.linalg.cholesky(cov[np.ix_(free, free)] + 1e-12 * np.eye(len(free)))
    samples = []
    for s in range(args.n_samples):
        v = vec.copy()
        v[free] = vec[free] + L @ rng.standard_normal(len(free))
        v = clip_to_bounds(v, index)
        samples.append({"fold": f"post{s}", "n_held": 0, "r2_kfold": float("nan"),
                        "r2_idw": float("nan"), "theta": unflatten(v, index, member.theta)})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(samples, fh, indent=1)
    with open(args.out.replace(".json", "_cov.json"), "w") as fh:
        json.dump({"index": index, "mean": vec.tolist(), "sd": sd.tolist(), "sigma2": sigma2,
                   "at_bound": at_bound, "cov": cov.tolist(), "prior_sd": args.prior_sd},
                  fh)
    print(f"wrote {args.n_samples} posterior samples -> {args.out} (pass after the theta file "
          "to twin.forward as extra members)")


if __name__ == "__main__":
    main()
