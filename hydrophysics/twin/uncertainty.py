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
calibration. Parameters held fixed (``fix_eta``) are never sampled.

Parameters at a bound (``--bounded``, 2026-09-23):

- ``hold`` (the default, the behaviour since 2026-09-14): a parameter sitting on a bound
  is reported and held where it is; the others are drawn from their marginal Gaussian
  and clipped to the bounds (mass piles up on the bound).
- ``truncnorm``: every bounded parameter is sampled from the Laplace Gaussian
  *truncated to the feasible box*, jointly (exact rejection sampling while the box keeps
  enough mass, else a Gibbs sampler over the coordinate conditionals, Geweke 1991). A
  parameter pinned at a bound then gets the one-sided posterior its curvature implies:
  a half-normal off the bound with the Laplace width, correlated with the rest. With
  ``--bound-mean newton`` the Gaussian is centred on the Gauss-Newton step from the fit
  instead (``theta_hat - cov J^T r / sigma^2`` on the at-bound coordinates), i.e. on the
  unconstrained optimum the data point to, which is usually past the bound -- the draws
  then crowd the bound more tightly. A logit reparameterisation was considered and
  rejected: at the bound the logit is infinite, the Jacobian in the unconstrained space
  vanishes, and the draws depend on an arbitrary offset from the bound.

``--from-cov`` re-draws from a saved ``*_cov.json`` without recomputing the Jacobian (CPU,
seconds). A parameter whose posterior sd equals the prior sd carried no information
(a zero Jacobian column) and is held, not sampled from the prior.

Effective bounds are the calibration's, not only ``BOUNDS``: the leakance floor
``meta["l_min"]`` raises the ``log_L`` floor, and ``--bound KEY=LO,HI`` overrides a bound
that has moved since the fit (the stress-spread ceiling went from 10 to 25 km on
2026-09-19; the ``stage3_spreadL_gate`` deliverable was fitted under 10 km).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..train import pick_device
from .calibrate_flow import (
    BOUNDS,
    RETURN_FRAC_MAX,
    _base_param_name,
    _extension_readouts,
    _r2,
    set_compile_matvec,
    well_datum_vector,
)
from .forward import Member, attach_sw_recharge, build_model, load_members, rollout, sw_hist
from .inputs import input_options, load_twin_inputs

# keys in a theta dict that are derived, not parameters
_DERIVED = {"eta", "head_extra_m", "recharge_frac", "C_coast_m2day", "C_apex_m2day",
            "pump_frac_shallow", "return_frac", "spread_km",
            # physical-unit readouts of the opt-in parameters of 2026-09-23 (the log_*
            # keys are the parameters and are perturbed/sampled like every other one)
            "Sd", "tau_days", "C_riv_m2day", "sw_scale",
            # second round (--delay-u0, --aquitard-storage, S_eff diagnostics)
            "du0_m", "Sa", "G_per_day", "S_eff", "delay_is_elastic"}
_DERIVED_PREFIXES = ("Sd_", "tau_days_", "du0_m_", "Sa_", "G_per_day_", "S_eff_",
                     "delay_is_elastic_")


def _is_derived(key: str) -> bool:
    return key in _DERIVED or key.startswith(_DERIVED_PREFIXES)


def flatten(theta: dict, fixed: tuple[str, ...] = ()) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """theta dict -> (vector, index) over free parameters, skipping derived and fixed."""
    vec, index = [], []
    for k, v in theta.items():
        if _is_derived(k) or k in fixed:
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
    return refresh_derived(out)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def refresh_derived(theta: dict) -> dict:
    """Recompute the derived keys from the parameters they come from, as
    ``calibrate_flow`` writes them. Not cosmetic: ``forward.build_model`` reads the
    stress radius from the derived ``spread_km``, so before 2026-09-23 a perturbed or
    sampled ``log_spread_km`` never reached the model -- its Jacobian column was zero
    (posterior sd = the prior sd) and every draw ran at the fitted radius."""
    th = dict(theta)
    if "log_eta" in th:
        v = th["log_eta"]
        th["eta"] = ([float(np.exp(x)) for x in v] if isinstance(v, list)
                     else float(np.exp(v)))
    if "log_head_extra" in th:
        th["head_extra_m"] = float(np.exp(th["log_head_extra"]))
    if "log_C_coast" in th:
        th["C_coast_m2day"] = [float(np.exp(v)) for v in np.atleast_1d(th["log_C_coast"])]
    if "log_C_apex" in th:
        th["C_apex_m2day"] = float(np.exp(np.atleast_1d(th["log_C_apex"])[0]))
    if "pump_split_logit" in th:
        th["pump_frac_shallow"] = _sigmoid(th["pump_split_logit"])
    if "return_frac_logit" in th:
        th["return_frac"] = RETURN_FRAC_MAX * _sigmoid(th["return_frac_logit"])
    if "log_spread_km" in th:
        th["spread_km"] = float(np.exp(th["log_spread_km"]))
    if "recharge_frac_logit" in th:
        th["recharge_frac"] = _sigmoid(th["recharge_frac_logit"])
    th.update(_extension_readouts(th))
    return th


def effective_bounds(index: list[tuple[str, int]], meta: dict | None = None,
                     overrides: dict[str, tuple[float, float]] | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` per flattened parameter: ``BOUNDS`` by base name (``-inf/+inf`` for
    unbounded logits), the ``log_L`` floor raised to ``meta["l_min"]`` as the calibration
    had it, then ``overrides`` keyed on the base name (or the full key)."""
    meta = meta or {}
    overrides = overrides or {}
    lo = np.full(len(index), -np.inf)
    hi = np.full(len(index), np.inf)
    for j, (k, _) in enumerate(index):
        base = _base_param_name(k)
        if base in BOUNDS:
            lo[j], hi[j] = BOUNDS[base]
        if base == "log_L" and meta.get("l_min") is not None:
            lo[j] = max(lo[j], float(np.log(float(meta["l_min"]))))
        if (k in ("log_T_proximal", "log_T_proximal_w")
                and meta.get("log_t_min_proximal") is not None):
            # --log-t-min-proximal (round 3): the calibration's raised proximal floor
            lo[j] = max(lo[j], float(np.log(float(meta["log_t_min_proximal"]))))
        if base == "log_tau" and meta.get("delay_tau_max_years") is not None:
            hi[j] = float(np.log(365.25 * float(meta["delay_tau_max_years"])))
        if base == "log_tau" and meta.get("delay_tau_min_days") is not None:
            lo[j] = float(np.log(float(meta["delay_tau_min_days"])))
        if base == "log_spread_km" and meta.get("spread_max_km") is not None:
            hi[j] = float(np.log(float(meta["spread_max_km"])))
        for key in (base, k):
            if key in overrides:
                lo[j], hi[j] = overrides[key]
    return lo, hi


def find_at_bound(vec: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                  tol: float = 1e-6) -> list[int]:
    return [j for j in range(len(vec))
            if abs(vec[j] - lo[j]) < tol or abs(vec[j] - hi[j]) < tol]


def newton_shift(J: np.ndarray, resid: np.ndarray, cov: np.ndarray,
                 sigma2: float) -> np.ndarray:
    """Gauss-Newton step from the fit toward the unconstrained optimum of the Laplace
    objective: ``-cov J^T r / sigma^2`` (the prior is centred on the fit, so it adds no
    gradient there)."""
    return -cov @ (J.T @ resid) / sigma2


def _truncnorm_1d(mean: float, sd: float, lo: float, hi: float,
                  rng: np.random.Generator) -> float:
    from scipy.stats import truncnorm

    a, b = (lo - mean) / sd, (hi - mean) / sd
    return float(truncnorm.rvs(a, b, loc=mean, scale=sd, random_state=rng))


def sample_truncated_mvn(mean: np.ndarray, cov: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                         n: int, rng: np.random.Generator, max_draws: int = 1_000_000,
                         burn: int = 500, thin: int = 50, start: np.ndarray | None = None
                         ) -> tuple[np.ndarray, str]:
    """``n`` draws from ``N(mean, cov)`` restricted to the box ``[lo, hi]`` -> (draws,
    method). Rejection (exact, i.i.d.) when the box keeps enough of the Gaussian's mass
    to collect ``n`` draws from ``max_draws`` proposals; otherwise a Gibbs sampler over
    the coordinate conditionals (each a 1-D truncated normal from the precision matrix),
    started at ``start`` (default: ``mean`` projected into the box), ``burn`` sweeps of
    burn-in and ``thin`` sweeps between kept draws."""
    p = len(mean)
    if p == 0:
        return np.zeros((n, 0)), "none"
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(p))
    kept: list[np.ndarray] = []
    drawn, batch = 0, 4096
    while drawn < max_draws:
        z = mean[None, :] + rng.standard_normal((batch, p)) @ L.T
        ok = np.all((z >= lo) & (z <= hi), axis=1)
        kept.extend(z[ok])
        drawn += batch
        if len(kept) >= n:
            return np.stack(kept[:n]), "rejection"
        if drawn >= 16 * batch and len(kept) * max_draws / drawn < n:
            break                                   # the box is too small for rejection
    Q = np.linalg.inv(cov + 1e-12 * np.eye(p))
    x = np.clip(mean if start is None else start, lo, hi).astype("float64").copy()
    # a start exactly on an infinite-density edge is fine; conditionals are 1-D truncated
    out = []
    for sweep in range(burn + n * thin):
        for j in range(p):
            sd = 1.0 / np.sqrt(Q[j, j])
            m = mean[j] - (Q[j] @ (x - mean) - Q[j, j] * (x[j] - mean[j])) / Q[j, j]
            x[j] = _truncnorm_1d(m, sd, lo[j], hi[j], rng)
        if sweep >= burn and (sweep - burn) % thin == thin - 1:
            out.append(x.copy())
    return np.stack(out[:n]), "gibbs"


def draw_posterior(vec: np.ndarray, cov: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   n: int, rng: np.random.Generator, mode: str = "hold",
                   hold: list[int] | tuple[int, ...] = (), at_bound: list[int] | None = None,
                   shift: np.ndarray | None = None) -> tuple[np.ndarray, str]:
    """``n`` parameter vectors from the Laplace posterior -> (draws (n, p), method).

    ``hold`` indices are never sampled. ``mode="hold"`` also holds ``at_bound`` and
    clips the rest (the pre-2026-09-23 behaviour, same random stream). ``"truncnorm"``
    samples every other coordinate from the Gaussian conditioned on the held ones and
    truncated to ``[lo, hi]``; ``shift`` (e.g. ``newton_shift`` masked to the at-bound
    coordinates) moves the Gaussian's centre."""
    at_bound = list(at_bound if at_bound is not None else find_at_bound(vec, lo, hi))
    p = len(vec)
    if mode == "hold":
        free = [j for j in range(p) if j not in at_bound and j not in hold]
        Lc = np.linalg.cholesky(cov[np.ix_(free, free)] + 1e-12 * np.eye(len(free)))
        out = np.tile(vec, (n, 1))
        for s in range(n):
            out[s, free] = vec[free] + Lc @ rng.standard_normal(len(free))
            out[s] = np.clip(out[s], lo, hi)
        return out, "hold"
    if mode != "truncnorm":
        raise ValueError(f"unknown bounded-sampling mode {mode!r}")
    free = [j for j in range(p) if j not in hold]
    held = [j for j in range(p) if j in hold]
    mu = vec.astype("float64").copy()
    if shift is not None:
        mu[free] += np.asarray(shift, dtype="float64")[free]
    if held:
        # condition on the held coordinates at their fitted values (which are also the
        # Gaussian's centre there, so the conditional mean is mu_f and only the
        # covariance shrinks to the Schur complement)
        P = np.linalg.inv(cov + 1e-12 * np.eye(p))
        cov_f = np.linalg.inv(P[np.ix_(free, free)])
    else:
        cov_f = cov
    mu_f = mu[free]
    draws, method = sample_truncated_mvn(mu_f, cov_f, lo[free], hi[free], n, rng,
                                         start=vec[free])
    out = np.tile(vec, (n, 1))
    out[:, free] = draws
    return out, method


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
    attach_sw_recharge(inp, member.meta, log=lambda m: None)
    h = rollout(model, scalars, inp.initial_heads(0), E, inp.recharge_field[:, 1:],
                inp.ground_elev, sw_field=sw_hist(inp))
    return h[torch.as_tensor(inp.obs_layer, device=h.device),
             torch.as_tensor(inp.obs_idx, device=h.device), 1:].cpu().numpy()


def jacobian(inp, member: Member, index, device, eps: float = 1e-3,
             log=print) -> tuple[np.ndarray, np.ndarray]:
    """``(J, residual)``: J is (n_obs, n_params) by central finite differences in the
    log-parameters (two rollouts per parameter, a few seconds each on the GPU); residual
    is pred - obs at the fit (with a ``--well-datum fit`` run's datum added, and J then
    centred per well over time: the datum profiled out). Finite differences rather than
    forward-mode autodiff because the implicit solve defines only a reverse-mode
    adjoint."""
    obs = inp.obs_h_filled[:, 1:]
    vec = np.array([np.atleast_1d(np.asarray(member.theta[k], dtype="float64")).reshape(-1)[i]
                    for k, i in index])
    t0 = time.perf_counter()
    base = predict_at_wells(inp, member, device)
    # --well-datum fit: the per-well datum is an observation-operator nuisance (not in
    # theta, so never perturbed or sampled). The residual is taken at its fitted value;
    # the Jacobian is profiled over it below (review 2026-09-26)
    datum = well_datum_vector(member.meta, inp.sids)
    if datum is not None:
        base = base + datum[:, None]
        log(f"  well datum held at its fit ({int((datum != 0).sum())} wells, rms "
            f"{float(np.sqrt((datum ** 2).mean())):.2f} m)")
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
    J = np.stack(cols, axis=1)
    if datum is not None:
        # A run with a per-well datum explains any per-well constant head shift with d_i,
        # not with the physics: holding d fixed would credit the physics with level
        # information the datum absorbs (an over-confident posterior). Profile d out: each
        # well's rows minus their mean over time (the datum treated as free, which is
        # exact for the long-record wells that keep ~97 % of their mean residual and
        # conservative -- a wider posterior -- for the few the prior shrinks).
        W = datum.shape[0]
        Jw = J.reshape(W, -1, J.shape[1])
        J = (Jw - Jw.mean(axis=1, keepdims=True)).reshape(J.shape)
    return J, resid


def laplace(J: np.ndarray, resid: np.ndarray, prior_sd: float = 2.0) -> tuple[np.ndarray, float]:
    """Posterior covariance ``sigma^2 (J^T J / sigma^2 ... )`` with a Gaussian prior of
    width ``prior_sd`` in log-parameter units on every direction."""
    n, p = J.shape
    sigma2 = float((resid ** 2).sum() / max(n - p, 1))
    H = J.T @ J / sigma2 + np.eye(p) / prior_sd ** 2
    cov = np.linalg.inv(H)
    return cov, sigma2


def _finite_or_none(a: np.ndarray) -> list:
    return [float(x) if np.isfinite(x) else None for x in a]


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
    ap.add_argument("--bounded", choices=("hold", "truncnorm"), default="hold",
                    help="parameters at a bound: hold them (default) or sample the whole "
                         "posterior truncated to the bounds (see the module docstring)")
    ap.add_argument("--bound-mean", choices=("fit", "newton"), default="fit",
                    help="truncnorm only: centre the Gaussian on the fit (half-normal off "
                         "a pinned bound) or on the Gauss-Newton step toward the "
                         "unconstrained optimum for the at-bound coordinates")
    ap.add_argument("--bound", action="append", default=[], metavar="KEY=LO,HI",
                    help="override a bound, in parameter units (log for log_*), e.g. "
                         "log_spread_km=-0.693147,2.302585 for the 0.5-10 km bound the "
                         "spreadL deliverable was fitted under (repeatable)")
    ap.add_argument("--from-cov", default=None,
                    help="re-draw from a saved *_cov.json instead of recomputing the "
                         "Jacobian (CPU, seconds); --bound-mean newton needs a cov file "
                         "written since 2026-09-23 (it stores the gradient)")
    args = ap.parse_args(argv)

    overrides = {}
    for spec in args.bound:
        key, rng_s = spec.split("=", 1)
        lo_s, hi_s = rng_s.split(",")
        overrides[key.strip()] = (float(lo_s), float(hi_s))

    member = load_members([args.theta])[0]
    fixed = tuple(member.meta.get("fixed") or ())
    if member.meta.get("fix_eta") is not None:
        fixed = fixed + ("log_eta",)
    if member.meta.get("fix_head_extra") is not None:
        fixed = fixed + ("log_head_extra",)
    vec, index = flatten(member.theta, fixed=fixed)
    lo, hi = effective_bounds(index, member.meta, overrides)
    at_bound = find_at_bound(vec, lo, hi)
    print(f"free parameters {len(index)}, fixed {list(fixed)}, at a bound "
          f"{[f'{index[j][0]}[{index[j][1]}]' for j in at_bound]}", flush=True)

    grad = None
    if args.from_cov:
        with open(args.from_cov) as fh:
            saved = json.load(fh)
        if [tuple(x) for x in saved["index"]] != index:
            raise SystemExit(f"{args.from_cov}: parameter index differs from {args.theta}")
        cov = np.asarray(saved["cov"], dtype="float64")
        sigma2 = float(saved["sigma2"])
        prior_sd = float(saved.get("prior_sd", args.prior_sd))
        if saved.get("grad") is not None:
            grad = np.asarray(saved["grad"], dtype="float64")
        print(f"posterior read from {args.from_cov} (no Jacobian recomputed)", flush=True)
    else:
        set_compile_matvec(args.compile_matvec)
        device = pick_device(args.device)
        print(f"device: {device}", flush=True)
        inp = load_twin_inputs(dx=args.dx, meter_filter=member.meta.get("meter_filter", "none"),
                               cap_duty=float(member.meta.get("cap_duty", 1.0)),
                               **input_options(member.meta))
        J, resid = jacobian(inp, member, index, device)
        cov, sigma2 = laplace(J, resid, prior_sd=args.prior_sd)
        prior_sd = args.prior_sd
        grad = J.T @ resid / sigma2
    sd = np.sqrt(np.diag(cov))
    # a parameter whose posterior is its prior carried no information (zero Jacobian
    # column): sampling it would only spread the prior into the forward twin
    uninformed = [j for j in range(len(index)) if abs(sd[j] - prior_sd) < 1e-6 * prior_sd]
    print(f"residual sd {np.sqrt(sigma2):.3f} m; posterior sd per parameter:")
    for j, (k, i) in enumerate(index):
        flag = ("  (uninformed: sd = prior, held)" if j in uninformed else
                "  (at bound, held)" if j in at_bound and args.bounded == "hold" else
                "  (at bound, truncated)" if j in at_bound else "")
        print(f"  {k}[{i}]: {vec[j]:+.3f} ± {sd[j]:.3f}{flag}")

    shift = None
    if args.bounded == "truncnorm" and args.bound_mean == "newton":
        if grad is None:
            raise SystemExit("--bound-mean newton needs the gradient; recompute the "
                             "Jacobian (drop --from-cov)")
        full = -cov @ grad
        shift = np.zeros_like(vec)
        shift[at_bound] = full[at_bound]
        print("Gauss-Newton shift on the at-bound coordinates: " + ", ".join(
            f"{index[j][0]}[{index[j][1]}] {shift[j]:+.3f}" for j in at_bound))
    rng = np.random.default_rng(args.seed)
    draws, method = draw_posterior(vec, cov, lo, hi, args.n_samples, rng, mode=args.bounded,
                                   hold=uninformed, at_bound=at_bound, shift=shift)
    print(f"sampler: {method}")
    if args.bounded == "truncnorm" and at_bound:
        print("at-bound parameters, drawn mean ± sd (fit value):")
        for j in at_bound:
            print(f"  {index[j][0]}[{index[j][1]}]: {draws[:, j].mean():+.3f} ± "
                  f"{draws[:, j].std():.3f} ({vec[j]:+.3f})")
    samples = [{"fold": f"post{s}", "n_held": 0, "r2_kfold": float("nan"),
                "r2_idw": float("nan"), "posterior": args.bounded,
                "theta": unflatten(draws[s], index, member.theta)}
               for s in range(args.n_samples)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(samples, fh, indent=1)
    with open(args.out.replace(".json", "_cov.json"), "w") as fh:
        json.dump({"index": index, "mean": vec.tolist(), "sd": sd.tolist(), "sigma2": sigma2,
                   "at_bound": at_bound, "uninformed": uninformed, "cov": cov.tolist(),
                   "prior_sd": prior_sd, "grad": None if grad is None else grad.tolist(),
                   "bounded": args.bounded, "bound_mean": args.bound_mean,
                   "sampler": method, "lo": _finite_or_none(lo), "hi": _finite_or_none(hi),
                   "from_cov": args.from_cov}, fh)
    print(f"wrote {args.n_samples} posterior samples -> {args.out} (pass after the theta file "
          "to twin.forward as extra members)")


if __name__ == "__main__":
    main()
