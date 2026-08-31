"""Stage 3: calibrate the four-layer flow model and run the k-fold cross-validation gate.

The gate is deliberately unkind: the physics model must beat ``subsidence.idw_interp``, the
inverse-distance interpolation every result so far has relied on, scored on the identical
held-out cells. A flow model that cannot beat IDW has not earned its complexity.

Three rulings override the Task-5 brief (see the SDD progress ledger and task-5-report.md
for the full reasoning):

1. **Homogeneous parameters are the primary configuration.** Calibrating one (log_T, log_S)
   pair per layer per CELL gives 4 layers x 2,148 cells x 2 + 3 x 2,148 = 23,628 free
   parameters against 147 wells x 132 months = 19,404 observations -- 1.22 parameters per
   observation, i.e. guaranteed overfitting (this project already measured that exact
   failure mode in Plan A: a 52-parameter per-site model scored -0.919 while the
   4-parameter pooled version scored +0.478 on identical data). ``--param-mode
   {homogeneous,percell}`` defaults to ``homogeneous``: one ``log_T``, one ``log_S`` per
   layer and one ``log_L`` per interface, broadcast across every active cell. This is
   implemented as a *constraint* on ``FlowModel``'s existing per-cell parameters: a small
   ``(n_layers, 1)``-shaped tensor is optimised and expanded to ``(n_layers, n_active)`` on
   every forward call (see ``_rollout`` below), rather than by changing ``FlowModel``'s
   frozen constructor or parameter shapes. Autograd's own broadcast-backward (a sum over
   the expanded axis) does the parameter-sharing gradient correctly with no manual
   bookkeeping.
2. **10-fold cross-validation over wells, not leave-one-out.** Leave-one-well-out over 147
   wells is 147 refits x epochs x (forward + adjoint) linear solves per monthly step --
   about 29.3M solves, not runnable. 10-fold CV over wells is about 13x cheaper and
   statistically adequate at this well count. Named honestly as k-fold everywhere: the
   function is ``kfold_wells``, not ``loso_wells``; the CSV column is ``r2_kfold``, not
   ``r2_loso``. The IDW baseline is scored on exactly the same folds and the same
   held-out cells as the flow model -- non-negotiable, and the reason both are computed
   inside the same fold loop below rather than in separate passes.
3. **Tighter log_T clamp.** ``log(10)..log(2e4)`` m^2/day, not the brief's
   ``log(1)..log(1e5)``. Task 2 measured that CG does not converge across 5 decades of T
   even in float64 with Jacobi preconditioning (true relative residual 1.4e-4 against a
   1e-8 target). The tighter range is grounded in Choushui transmissivity measured at
   0.04-4.19 m^2/min = 58-6,034 m^2/day (Liu et al. 2002).

**Fix round 1 (coordinator ruling, superseding the first headline run).** The first gate
run used ``h0 = 0``, zero recharge, and zero pumping, which is provably parameter-
independent: at every step ``b = S*area/dt*h + q`` with ``q = 0``, so ``b = 0`` whenever
``h`` is already 0, and the unique solution of the SPD system ``M @ h = 0`` is ``h = 0`` for
*any* T, S, L. That run is invalid and is not a Stage-3 result (see task-5-report.md for the
full record, including the proof and its empirical confirmation). This module now wires the
real drivers the brief always intended (``Consumes: ... aggregate_pumps/energy_to_volume``):

- ``h0`` is IDW'd per layer from that layer's observed wells' first month
  (``_idw_initial_heads``), not zero.
- Pumping: the electricity census (``pumping.aggregate_pumps``/``energy_to_volume``),
  applied to a single layer (``pump_layer``, default index 1 -- the main production
  aquifer). Lift (``ground_elevation - simulated head``) is recomputed from the *previous*
  step's head inside the rollout every iteration, since it is a real, evolving
  energy-water feedback, not a static field. One learnable scalar, ``log_eta`` (clamped
  ``log(0.05)..log(0.9)``).
- Recharge: rain (26 gauges) minus cached ET0, IDW'd to the grid, clamped at zero,
  applied to a single layer (``recharge_layer``, default index 0) through one learnable
  scalar recharge fraction in ``[0, 1]`` (a sigmoid, so no BOUNDS entry is needed).

That is 13 free parameters in the homogeneous, 4-layer, both-drivers-active configuration
(11 from Ruling 1 + log_eta + the recharge fraction).
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..subsidence import idw_interp
from . import pumping as pumping_mod
from .flow import FlowModel, _ImplicitSolve, _warm_started_solver
from .grid import build_grid
from .zones import N_ZONES, ZONE_NAMES, fan_zones

# Physically defensible bounds. log_T is tightened per Ruling 3 above (Task-2 CG
# conditioning finding); log_S and log_L keep the brief's bounds. log_eta (fix round 1)
# is the brief's original wire-to-water-efficiency bound.
BOUNDS = {
    "log_T": (math.log(10.0), math.log(2e4)),        # m2/day (Liu et al. 2002)
    "log_S": (math.log(1e-6), math.log(0.3)),        # -
    "log_L": (math.log(1e-8), math.log(1e-1)),       # 1/day
    "log_eta": (math.log(0.05), math.log(0.9)),      # wire-to-water efficiency, -
}


def _clamp_(tensors: dict[str, torch.Tensor]) -> dict[str, int]:
    """Clamp each named tensor into BOUNDS in place; return how many entries sit on a bound."""
    hits: dict[str, int] = {}
    with torch.no_grad():
        for name, par in tensors.items():
            if par is None:
                continue
            lo, hi = BOUNDS[name]
            par.clamp_(min=lo, max=hi)
            hits[name] = int(((par <= lo + 1e-9) | (par >= hi - 1e-9)).sum())
    return hits


def _r2(pred: np.ndarray, obs: np.ndarray) -> float:
    finite = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[finite], obs[finite]
    ss_res = float(((obs - pred) ** 2).sum())
    ss_tot = float(((obs - obs.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _idw_field(grid, xy: np.ndarray, values: np.ndarray) -> np.ndarray:
    """IDW-interpolate scalar point observations to every active grid-cell centroid."""
    pts = grid.centroids()
    v = np.asarray(values, dtype="float64").reshape(-1, 1)
    return idw_interp(pts, np.asarray(xy, dtype="float64"), v)[:, 0]


def _idw_initial_heads(grid, xy: np.ndarray, h0_values: np.ndarray,
                       layer_of: np.ndarray, n_layers: int) -> torch.Tensor:
    """Per-layer IDW of each well's first-observed head to every active cell (fix round 1,
    replacing the earlier ``h0 = 0``). A layer with no wells in the given set falls back to
    the IDW of every well regardless of layer, rather than leaving that layer's initial
    condition at zero. Callers are responsible for passing only the wells that should be
    visible for this construction (``kfold_wells`` passes only the fold's *kept* wells, so
    a held-out well's own head never leaks into the initial condition used to score it).
    """
    A = grid.n_active
    xy = np.asarray(xy, dtype="float64")
    h0_values = np.asarray(h0_values, dtype="float64")
    layer_of = np.asarray(layer_of)
    finite = np.isfinite(h0_values)
    xy, h0_values, layer_of = xy[finite], h0_values[finite], layer_of[finite]
    out = np.zeros((n_layers, A), dtype="float64")
    fallback = _idw_field(grid, xy, h0_values) if len(h0_values) else np.zeros(A)
    for k in range(n_layers):
        m = layer_of == k
        out[k] = _idw_field(grid, xy[m], h0_values[m]) if m.sum() >= 1 else fallback
    return torch.tensor(out, dtype=torch.float64)


def _rollout(model: FlowModel, log_T: torch.Tensor, log_S: torch.Tensor,
             log_L: torch.Tensor | None, h0: torch.Tensor, n_steps: int, *,
             recharge: torch.Tensor | None = None, pumping: torch.Tensor | None = None,
             recharge_field: torch.Tensor | None = None,
             recharge_scale: torch.Tensor | None = None, recharge_layer: int = 0,
             E: torch.Tensor | None = None, log_eta: torch.Tensor | None = None,
             ground_elev: torch.Tensor | None = None, pump_layer: int = 1) -> torch.Tensor:
    """The same backward-Euler rollout as ``FlowModel.forward``, but taking log-parameter
    tensors as arguments instead of reading ``model``'s own registered nn.Parameters, and
    (fix round 1) supporting a *dynamic* forcing mode alongside the original static one.

    This is what makes homogeneous-mode calibration possible without touching
    ``FlowModel``'s constructor: the caller passes in a *broadcast view* of a small
    per-layer tensor (``theta.expand(n_layers, n_active)``), and autograd's own
    expand-backward (a sum over the broadcast axis) correctly accumulates the per-cell
    gradient back onto the small tensor. ``model`` still supplies the grid-derived
    operator machinery (``_matvec_from``, ``_op``, ``area``, ``dt``) -- only the
    parameter *source* differs from ``forward``.

    Two forcing mechanisms, independently switchable, both feeding the same
    ``q = recharge*area - pumping`` convention ``FlowModel.forward`` uses:

    - **Static** (``recharge``/``pumping``, each ``(n_layers, A, n_steps)``): used as-is at
      every step, exactly like ``FlowModel.forward``. This is what ``param_mode="percell"``
      and the synthetic tests use, and what a caller with no real driver data falls back to
      (both default ``None``, contributing nothing).
    - **Dynamic** (fix round 1): ``recharge_field`` (A, n_steps, m/day, already
      rain-minus-ET0 and clamped >= 0) times a differentiable ``recharge_scale`` passed
      through a sigmoid, injected into ``recharge_layer`` only; and/or ``E`` (A, n_steps,
      kWh) converted through ``pumping.energy_to_volume`` using ``log_eta`` and a lift
      computed from ``ground_elev`` minus the *previous* step's simulated head at
      ``pump_layer``, injected as an extraction into ``pump_layer`` only. Lift is
      recomputed every iteration from the evolving head -- the real energy-water feedback
      the pumping module's docstring describes (falling heads make the same electricity
      deliver less water), and the reason this cannot be a precomputed static tensor.
      ``recharge_scale``/``log_eta`` never enter the SPD operator ``M`` (only
      ``log_T``/``log_S``/``log_L`` do), so they need no implicit-function-theorem
      treatment here: they reach the loss purely through ``b``'s ordinary autograd graph,
      and PyTorch backpropagates through the resulting backward-Euler recurrence (via
      ``h[pump_layer]``'s dependence on the *previous* step's ``_ImplicitSolve.apply``
      output) automatically.
    """
    h0 = h0.to(dtype=torch.float64)
    A = h0.shape[-1]
    dev = h0.device
    T = torch.exp(log_T)
    S = torch.exp(log_S)
    if model.n_layers > 1:
        L = torch.exp(log_L)
        params = (log_T, log_S, log_L)
    else:
        L = None
        params = (log_T, log_S)
    mv, diag = model._matvec_from(T, S, L)
    h = h0
    out = [h0]
    for t in range(n_steps):
        layer_q = [torch.zeros(A, dtype=torch.float64, device=dev)
                  for _ in range(model.n_layers)]
        if recharge is not None:
            for k in range(model.n_layers):
                layer_q[k] = layer_q[k] + recharge[k, :, t].to(dtype=torch.float64) * model.area
        if pumping is not None:
            for k in range(model.n_layers):
                layer_q[k] = layer_q[k] - pumping[k, :, t].to(dtype=torch.float64)
        if recharge_field is not None:
            layer_q[recharge_layer] = (
                layer_q[recharge_layer]
                + torch.sigmoid(recharge_scale) * recharge_field[:, t] * model.area
            )
        if E is not None:
            lift = torch.clamp(ground_elev - h[pump_layer], min=pumping_mod.MIN_LIFT_M)
            vol = pumping_mod.energy_to_volume(E[:, t], lift, log_eta)   # m3 for the month
            rate = vol / model.dt                                        # m3/day
            layer_q[pump_layer] = layer_q[pump_layer] - rate
        q = torch.stack(layer_q, dim=0)
        b = S * model.area / model.dt * h + q
        solve = _warm_started_solver(mv, diag, h)
        h = _ImplicitSolve.apply(b, model._op, solve, *params)
        out.append(h)
    return torch.stack(out, dim=-1)


def _make_homogeneous_params(model: FlowModel, use_pumping: bool = False,
                             use_recharge: bool = False) -> dict[str, nn.Parameter]:
    """One (log_T, log_S) per layer and one log_L per interface, shape ``(k, 1)`` so it
    broadcasts against ``(n_layers, n_active)`` via ``.expand``. Initialised from the
    model's own (uniform, per Task 3/4's constructor) starting values.

    Fix round 1: ``use_pumping``/``use_recharge`` each add one more learnable *scalar*
    (not per-layer, not per-cell) -- ``log_eta`` (log-bounded like the others, via BOUNDS)
    and ``recharge_frac_logit`` (a raw scalar passed through a sigmoid at use time, so it
    needs no BOUNDS entry) -- 13 parameters total for a 4-layer model with both drivers
    active, per the coordinator's ruling.
    """
    theta = {
        "log_T": nn.Parameter(model.log_T[:, :1].detach().clone()),
        "log_S": nn.Parameter(model.log_S[:, :1].detach().clone()),
    }
    if model.n_layers > 1:
        theta["log_L"] = nn.Parameter(model.log_L[:, :1].detach().clone())
    dev = model.log_T.device
    if use_pumping:
        theta["log_eta"] = nn.Parameter(
            torch.tensor(float(np.log(0.3)), dtype=torch.float64, device=dev))
    if use_recharge:
        theta["recharge_frac_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))
    return theta


def _base_param_name(name: str) -> str:
    """``"log_T_mid"`` -> ``"log_T"``. Zonal theta keys carry a zone suffix; BOUNDS and
    _clamp_ are keyed on the bare physical name. Only the three known zone suffixes are
    stripped, so ``log_eta`` and ``recharge_frac_logit`` survive untouched.
    """
    for zone in ZONE_NAMES:
        suffix = f"_{zone}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _make_zonal_params(model: FlowModel, use_pumping: bool = False,
                       use_recharge: bool = False) -> dict[str, nn.Parameter]:
    """Structural proximal/mid/distal parameters -- 26 free values for a 4-layer model
    with both drivers, against the homogeneous mode's 13 (spec §5).

    Zones differ in **form**, not only in value:

    - **proximal** is one merged aquifer: a single ``log_T`` and a single ``log_S``
      shared across all four layers, and NO ``log_L`` at all. Its leakage is pinned at
      the top of BOUNDS by ``_expand_zonal`` as a constant. That IS the published
      geology -- thick gravel, indistinct stratification, unrestricted vertical flow --
      encoded structurally rather than left for the optimiser to discover. 2 parameters.
    - **mid** and **distal** keep the full 4 aquifers + 3 aquitards. 11 each.
    - **global**: ``log_eta`` and the recharge fraction, as in homogeneous mode. 2.

    Shapes are ``(k, 1)`` so ``_expand_zonal`` can column-stack them into ``(k, N_ZONES)``
    and gather to ``(k, n_active)``. Initialised from the model's own uniform starting
    values, so a zonal run and a homogeneous run start from the same physics.
    """
    log_T0 = model.log_T[:, :1].detach().clone()
    log_S0 = model.log_S[:, :1].detach().clone()
    theta = {
        # one merged aquifer: mean of the layer starts, a single shared value
        "log_T_proximal": nn.Parameter(log_T0.mean(dim=0, keepdim=True)),
        "log_S_proximal": nn.Parameter(log_S0.mean(dim=0, keepdim=True)),
        "log_T_mid": nn.Parameter(log_T0.clone()),
        "log_S_mid": nn.Parameter(log_S0.clone()),
        "log_T_distal": nn.Parameter(log_T0.clone()),
        "log_S_distal": nn.Parameter(log_S0.clone()),
    }
    if model.n_layers > 1:
        log_L0 = model.log_L[:, :1].detach().clone()
        theta["log_L_mid"] = nn.Parameter(log_L0.clone())
        theta["log_L_distal"] = nn.Parameter(log_L0.clone())
    dev = model.log_T.device
    if use_pumping:
        theta["log_eta"] = nn.Parameter(
            torch.tensor(float(np.log(0.3)), dtype=torch.float64, device=dev))
    if use_recharge:
        theta["recharge_frac_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))
    return theta


def _expand_zonal(theta: dict[str, torch.Tensor], zone_t: torch.Tensor,
                  n_layers: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Column-stack the per-zone tensors into ``(k, N_ZONES)`` and gather to ``(k, A)``.

    The gather ``cols[:, zone_t]`` is advanced indexing, whose backward is a scatter-add:
    every cell's gradient accumulates onto its own zone's small tensor, exactly the way
    homogeneous mode relies on expand-backward summing over the broadcast axis. That is
    what lets ``FlowModel``'s frozen constructor and parameter shapes stay untouched.

    Proximal ``log_L`` is a CONSTANT at the upper bound, not a parameter: the four
    proximal layers equilibrate instead of being independently fitted.
    """
    dev = theta["log_T_mid"].device
    prox_T = theta["log_T_proximal"].expand(n_layers, 1)
    prox_S = theta["log_S_proximal"].expand(n_layers, 1)
    cols_T = torch.cat([prox_T, theta["log_T_mid"], theta["log_T_distal"]], dim=1)
    cols_S = torch.cat([prox_S, theta["log_S_mid"], theta["log_S_distal"]], dim=1)
    assert cols_T.shape == (n_layers, N_ZONES)
    log_T = cols_T[:, zone_t]
    log_S = cols_S[:, zone_t]
    log_L = None
    if n_layers > 1 and "log_L_mid" in theta:
        prox_L = torch.full((n_layers - 1, 1), BOUNDS["log_L"][1],
                            dtype=torch.float64, device=dev)
        cols_L = torch.cat([prox_L, theta["log_L_mid"], theta["log_L_distal"]], dim=1)
        log_L = cols_L[:, zone_t]
    return log_T, log_S, log_L


def _zonal_bounds_hit(theta: dict[str, torch.Tensor]) -> dict[str, dict[str, int]]:
    """Clamp every zonal parameter into BOUNDS in place and report hits **per zone**.

    Spec §6's primary decision rule reads this. Pooling would let an interior mid value
    mask a pinned proximal one, and a model still fitting outside the measured physical
    range gives confident wrong counterfactuals -- the failure that matters at the
    operational bar.
    """
    report: dict[str, dict[str, int]] = {name: {} for name in ZONE_NAMES}
    report["global"] = {}
    for name, par in theta.items():
        base = _base_param_name(name)
        if base not in BOUNDS:
            continue                     # recharge_frac_logit: unconstrained by design
        bucket = next((z for z in ZONE_NAMES if name.endswith(f"_{z}")), "global")
        report[bucket][base] = _clamp_({base: par})[base]
    return report


def fit_flow(model: FlowModel, obs_h: torch.Tensor, obs_idx: torch.Tensor,
             obs_layer: torch.Tensor, recharge: torch.Tensor,
             E=None, ground_elev=None, epochs: int = 1500, lr: float = 0.1,
             init_scatter: float = 0.0, seed: int | None = None,
             param_mode: str = "homogeneous", h0: torch.Tensor | None = None,
             zone_of_cell: np.ndarray | None = None,
             recharge_field: torch.Tensor | None = None,
             pump_layer: int = 1, recharge_layer: int = 0,
             log_every: int = 0) -> dict:
    """Fit log-parameters to observed head series by masked MSE.

    ``obs_h`` is ``(W, T)``; ``obs_idx``/``obs_layer`` locate each well in the active-cell
    vector and the layer stack. ``h0`` is the ``(n_layers, A)`` initial head field (fix
    round 1: build it with ``_idw_initial_heads``, not zero -- see the module docstring for
    why zero is a degenerate choice). ``param_mode`` selects the free-parameter structure:

    - ``"homogeneous"`` (default, Ruling 1): one ``log_T``/``log_S`` per layer and one
      ``log_L`` per interface, broadcast across all active cells, plus (fix round 1) one
      learnable ``log_eta`` when ``E``/``ground_elev`` are given and one learnable
      recharge fraction when ``recharge_field`` is given -- 11-13 parameters for a
      4-layer model. Dynamic pumping/recharge (see ``_rollout``) is only implemented for
      this mode.
    - ``"percell"``: ``FlowModel``'s own per-cell parameters directly (the brief's
      original configuration) -- 23,628 parameters for a 4-layer, 2,148-cell grid. Only
      run this as a secondary check; it is not the headline gate. Dynamic pumping/
      recharge is not wired for this mode (``E``/``ground_elev``/``recharge_field`` are
      ignored here) -- it keeps the original static (here: zero, in every current caller)
      recharge/pumping tensors.
    - ``"zonal"`` (spec 2026-08-29 §5): structural proximal/mid/distal zonation --
      proximal is one merged aquifer (2 parameters, ``log_L`` fixed at its upper bound),
      mid and distal each keep 4 aquifers + 3 aquitards (11 each), plus the same two
      global driver scalars: 26 parameters for a 4-layer model. Needs ``zone_of_cell``.
      ``bounds_hit`` comes back nested by zone, never pooled.
    """
    if param_mode not in ("homogeneous", "percell", "zonal"):
        raise ValueError(
            "param_mode must be 'homogeneous', 'percell' or 'zonal', "
            f"got {param_mode!r}"
        )
    if param_mode == "zonal" and zone_of_cell is None:
        raise ValueError(
            "param_mode='zonal' needs zone_of_cell: an (n_active,) zone id per active "
            "cell, from hydrophysics.twin.zones.fan_zones(grid.centroids())"
        )

    n_steps = recharge.shape[-1]
    A = model.grid.n_active
    dev = model.log_T.device
    if h0 is None:
        h0 = torch.zeros(model.n_layers, A, dtype=torch.float64, device=dev)
    else:
        h0 = h0.to(dtype=torch.float64, device=dev)
    obs_h = obs_h.to(dtype=torch.float64, device=dev)
    obs_idx = obs_idx.to(device=dev)
    obs_layer = obs_layer.to(device=dev)
    recharge = recharge.to(dtype=torch.float64, device=dev)
    if recharge_field is not None:
        recharge_field = recharge_field.to(dtype=torch.float64, device=dev)
    if E is not None:
        E = E.to(dtype=torch.float64, device=dev)
    if ground_elev is not None:
        ground_elev = ground_elev.to(dtype=torch.float64, device=dev)

    if param_mode == "percell":
        pumping = torch.zeros(model.n_layers, A, n_steps, dtype=torch.float64, device=dev)
        if init_scatter > 0.0:
            g = torch.Generator(device="cpu")
            if seed is not None:
                g.manual_seed(int(seed))
            with torch.no_grad():
                for name in ("log_T", "log_S", "log_L"):
                    par = getattr(model, name, None)
                    if par is not None:
                        par.add_(torch.randn(par.shape, generator=g).to(par.device)
                                 * init_scatter)
        free = list(model.parameters())
        opt = torch.optim.Adam(free, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss = torch.tensor(float("nan"))
        hits: dict[str, int] = {}
        for _ in range(epochs):
            opt.zero_grad()
            h = model(h0, recharge, pumping, n_steps)
            pred = h[obs_layer, obs_idx, 1:]
            loss = ((pred - obs_h) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(free, 1.0)
            opt.step()
            sched.step()
            hits = _clamp_({n: getattr(model, n) for n in BOUNDS if hasattr(model, n)})
        with torch.no_grad():
            pred = model(h0, recharge, pumping, n_steps)[obs_layer, obs_idx, 1:]
        n_params = sum(p.numel() for p in free)
        return {"loss": float(loss.detach()), "epochs": epochs, "bounds_hit": hits,
                "r2": _r2(pred.cpu().numpy(), obs_h.cpu().numpy()), "n_params": n_params,
                "param_mode": param_mode}

    # homogeneous: optimise a small (k, 1) tensor per parameter, expanded to (k, A) on
    # every forward call (Ruling 1), plus (fix round 1) the two scalar driver parameters.
    # model's own registered per-cell parameters are left untouched during optimisation
    # and only overwritten (with the final broadcast log_T/log_S/log_L values) at the
    # end, so downstream code that calls model(...) directly sees the calibrated
    # homogeneous field without needing to know how it was fit. log_eta/recharge_frac
    # have no per-cell home on model to copy back into; they travel in the return dict.
    use_pumping = E is not None and ground_elev is not None
    use_recharge = recharge_field is not None
    zone_t = None
    if param_mode == "zonal":
        zone_arr = np.asarray(zone_of_cell, dtype="int64").reshape(-1)
        if zone_arr.shape[0] != A:
            raise ValueError(
                f"zone_of_cell has {zone_arr.shape[0]} entries but the grid has {A} "
                "active cells"
            )
        zone_t = torch.tensor(zone_arr, dtype=torch.long, device=dev)
        theta = _make_zonal_params(model, use_pumping=use_pumping,
                                   use_recharge=use_recharge)
    else:
        theta = _make_homogeneous_params(model, use_pumping=use_pumping,
                                         use_recharge=use_recharge)
    if init_scatter > 0.0:
        g = torch.Generator().manual_seed(int(seed) if seed is not None else 0)
        with torch.no_grad():
            for name, par in theta.items():
                if name == "recharge_frac_logit":
                    continue   # unconstrained scalar; scatter would just re-centre lr=0.5
                par.add_(torch.randn(par.shape, generator=g) * init_scatter)
    free = list(theta.values())
    opt = torch.optim.Adam(free, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss = torch.tensor(float("nan"))
    hits: dict[str, int] = {}

    def _forward() -> torch.Tensor:
        if zone_t is not None:
            log_T, log_S, log_L = _expand_zonal(theta, zone_t, model.n_layers)
        else:
            log_T = theta["log_T"].expand(-1, A)
            log_S = theta["log_S"].expand(-1, A)
            log_L = theta["log_L"].expand(-1, A) if "log_L" in theta else None
        return _rollout(
            model, log_T, log_S, log_L, h0, n_steps,
            recharge=None if use_recharge else recharge,
            pumping=None,
            recharge_field=recharge_field if use_recharge else None,
            recharge_scale=theta.get("recharge_frac_logit"),
            recharge_layer=recharge_layer,
            E=E if use_pumping else None,
            log_eta=theta.get("log_eta"),
            ground_elev=ground_elev,
            pump_layer=pump_layer,
        )

    r2_trace: list[tuple[int, float]] = []
    for _ep in range(epochs):
        opt.zero_grad()
        h = _forward()
        pred = h[obs_layer, obs_idx, 1:]
        loss = ((pred - obs_h) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(free, 1.0)
        opt.step()
        sched.step()
        if log_every and ((_ep + 1) % log_every == 0 or _ep == 0):
            # Periodic in-sample R2 so the TRAJECTORY is visible, not just the endpoint.
            # A gate that fails can fail two ways -- under-trained or structurally
            # inadequate -- and only the trajectory separates them: still climbing means
            # the epoch budget bound, a plateau means the parameterisation did.
            with torch.no_grad():
                _p = _forward()[obs_layer, obs_idx, 1:]
                _r = _r2(_p.cpu().numpy(), obs_h.cpu().numpy())
            r2_trace.append((_ep + 1, _r))
            print(f"      epoch {_ep + 1:4d}: loss={float(loss.detach()):.4g} "
                  f"in-sample R2={_r:+.4f}", flush=True)
        hits = (_zonal_bounds_hit(theta) if zone_t is not None
                else _clamp_({k: v for k, v in theta.items() if k in BOUNDS}))
    with torch.no_grad():
        h = _forward()
        pred = h[obs_layer, obs_idx, 1:]
        if zone_t is not None:
            zT, zS, zL = _expand_zonal(theta, zone_t, model.n_layers)
            model.log_T.copy_(zT)
            model.log_S.copy_(zS)
            if zL is not None and model.n_layers > 1:
                model.log_L.copy_(zL)
        else:
            model.log_T.copy_(theta["log_T"].expand(-1, A))
            model.log_S.copy_(theta["log_S"].expand(-1, A))
            if "log_L" in theta and model.n_layers > 1:
                model.log_L.copy_(theta["log_L"].expand(-1, A))
    n_params = sum(p.numel() for p in free)
    theta_out = {}
    for k, v in theta.items():
        theta_out[k] = (float(v.detach().cpu()) if v.dim() == 0
                        else v.detach().clone().squeeze(-1).cpu().numpy().tolist())
    if "log_eta" in theta_out:
        theta_out["eta"] = float(np.exp(theta_out["log_eta"]))
    if "recharge_frac_logit" in theta_out:
        theta_out["recharge_frac"] = float(1.0 / (1.0 + np.exp(-theta_out["recharge_frac_logit"])))
    return {"loss": float(loss.detach()), "epochs": epochs, "bounds_hit": hits,
            "r2": _r2(pred.cpu().numpy(), obs_h.cpu().numpy()), "n_params": n_params,
            "param_mode": param_mode, "theta": theta_out, "r2_trace": r2_trace}


def _predict_homogeneous(model: FlowModel, fit: dict, h0: torch.Tensor, n_steps: int,
                         recharge: torch.Tensor | None = None,
                         recharge_field: torch.Tensor | None = None,
                         E: torch.Tensor | None = None,
                         ground_elev: torch.Tensor | None = None,
                         recharge_layer: int = 0, pump_layer: int = 1) -> torch.Tensor:
    """Re-run the rollout for evaluation (e.g. at wells held out of a k-fold's fit),
    reusing ``model``'s own calibrated per-cell log_T/log_S/log_L. This serves BOTH
    ``homogeneous`` and ``zonal``: both copy their expanded field back into the model, so
    by this point the parameter *source* is identical and only the field's spatial
    structure differs (constant vs piecewise-constant). The name is historical.

    Also reuses the scalar ``log_eta``/``recharge_frac_logit`` returned in
    ``fit["theta"]`` -- those two scalars have no per-cell home on ``model`` to copy back
    into, so they travel through the fit-result dict instead. Both param_mode="homogeneous"
    and param_mode="zonal" keep these two keys bare (no zone suffix) in ``theta``.
    """
    with torch.no_grad():
        log_T, log_S = model.log_T, model.log_S
        log_L = model.log_L if model.n_layers > 1 else None
        theta = fit.get("theta", {})
        log_eta = (torch.tensor(theta["log_eta"], dtype=torch.float64)
                   if E is not None and "log_eta" in theta else None)
        rfrac = (torch.tensor(theta["recharge_frac_logit"], dtype=torch.float64)
                 if recharge_field is not None and "recharge_frac_logit" in theta else None)
        return _rollout(model, log_T, log_S, log_L, h0, n_steps,
                        recharge=None if recharge_field is not None else recharge,
                        recharge_field=recharge_field, recharge_scale=rfrac,
                        recharge_layer=recharge_layer,
                        E=E if log_eta is not None else None, log_eta=log_eta,
                        ground_elev=ground_elev, pump_layer=pump_layer)


def _kfold_indices(n: int, n_folds: int, seed: int = 0,
                   groups: np.ndarray | None = None) -> list[np.ndarray]:
    """Deterministic, (roughly) equal-sized fold partition of ``range(n)``.

    ``groups`` (length ``n`` labels) makes the split *grouped*: every index sharing a
    label lands in the same fold. Retraction of 2026-08-27: the Choushui head field is
    136 layer-coded screens over only 66 physical sites, so an ungrouped split put 95 of
    136 held-out entries (69.9%) at ZERO distance from a training entry. ``idw_interp``
    weights by ``1/(d^2 + 1e-6)``, so a co-located source outweighs a 1 km neighbour by
    1e12 -- the "baseline" stops interpolating and starts copying another screen in the
    same borehole, and the gate compares the physics model against a near-oracle. Group
    by physical site and that channel is closed.

    Folds are equal-sized in *groups*, so they are only roughly equal in entries.
    """
    rng = np.random.default_rng(seed)
    if groups is None:
        order = rng.permutation(n)
        return [np.sort(chunk) for chunk in np.array_split(order, n_folds)]

    g = np.asarray(groups)
    if g.shape[0] != n:
        raise ValueError(f"groups has length {g.shape[0]}, expected {n}")
    uniq = np.unique(g)
    if n_folds > len(uniq):
        raise ValueError(
            f"n_folds={n_folds} exceeds the {len(uniq)} distinct groups; a grouped split "
            f"cannot fill that many folds. Lower n_folds or drop the grouping.")
    order = rng.permutation(len(uniq))
    return [np.sort(np.flatnonzero(np.isin(g, uniq[chunk])))
            for chunk in np.array_split(order, n_folds)]


def _site_labels(well_xy: np.ndarray, decimals: int = 1) -> np.ndarray:
    """Integer physical-site id per well entry, from rounded coordinates."""
    key = np.round(np.asarray(well_xy, dtype="float64"), decimals)
    return np.unique(key, axis=0, return_inverse=True)[1]


def _colocation_rate(folds: list[np.ndarray], well_xy: np.ndarray | None,
                     n: int) -> float:
    """Fraction of held-out entries sitting at zero distance from a *training* entry.

    This is the number that would have caught the 2026-08-27 leak on sight, so the gate
    reports it permanently rather than leaving it to be rediscovered.
    """
    if well_xy is None:
        return float("nan")
    site = _site_labels(well_xy)
    leaked = held_total = 0
    for held in folds:
        held = np.asarray(held)
        keep = np.setdiff1d(np.arange(n), held)
        train_sites = set(site[keep].tolist())
        leaked += sum(1 for i in held if site[i] in train_sites)
        held_total += len(held)
    return leaked / held_total if held_total else float("nan")


def kfold_wells(grid, obs_h: torch.Tensor, obs_idx: torch.Tensor,
                obs_layer: torch.Tensor, recharge: torch.Tensor, n_layers: int = 4,
                epochs: int = 1500, lr: float = 0.1, n_folds: int = 10,
                param_mode: str = "homogeneous", seed: int = 0,
                well_xy: np.ndarray | None = None, obs_h0: np.ndarray | None = None,
                ground_elev: torch.Tensor | None = None, E: torch.Tensor | None = None,
                recharge_field: torch.Tensor | None = None,
                pump_layer: int = 1, recharge_layer: int = 0,
                zone_of_cell: np.ndarray | None = None,
                dump_path: str | None = None) -> dict:
    """K-fold cross-validation over wells (Ruling 2: k-fold, never leave-one-out).

    Wells are split into ``n_folds`` folds; for each fold the model is refit on the
    other 9/10 of the wells and scored on the held-out fold. The IDW baseline
    (``subsidence.idw_interp``) is computed inside the *same* fold loop, from the exact
    same ``keep``/``held`` well and cell index sets, so the two R^2 numbers are never at
    risk of drifting apart onto non-identical arrays.

    Fix round 1: when ``well_xy``/``obs_h0`` are given, each fold's initial head field is
    rebuilt by IDW from that fold's *kept* wells only (never the held-out ones) -- the
    same identical-arrays/no-leakage discipline Ruling 2 already applies to the IDW
    baseline, now extended to the model's own initial condition, so a held-out well's own
    head can never leak into the forecast used to score it. ``ground_elev``/``E``/
    ``recharge_field`` are physical driver fields independent of which wells are held
    out, so they are reused unchanged across every fold (no leakage risk there).
    """
    if param_mode == "zonal" and zone_of_cell is None:
        raise ValueError(
            "param_mode='zonal' needs zone_of_cell -- the zone assignment is a property "
            "of the grid, not of the fold, so it is passed once and reused unchanged"
        )
    W = obs_h.shape[0]
    xy = grid.centroids()
    n_active = grid.n_active
    n_steps = recharge.shape[-1]
    obs_layer_np = obs_layer.cpu().numpy()
    # Grouped by physical site when coordinates are available (2026-08-27 retraction):
    # co-located screens must never straddle a fold boundary, or the IDW baseline reads
    # the held-out well's own borehole instead of interpolating to it.
    groups = _site_labels(well_xy) if well_xy is not None else None
    if groups is None:
        print("    WARNING: no well_xy, so folds are UNGROUPED -- co-located screens may "
              "straddle folds and the IDW baseline may peek. Pass well_xy.", flush=True)
    folds = _kfold_indices(W, n_folds, seed=seed, groups=groups)
    coloc = _colocation_rate(folds, well_xy, W)
    # Per-entry dump (2026-08-29): the headline gate is one pooled R2, which cannot answer
    # whether the physics model degrades more or less gracefully than IDW as held-out sites
    # get isolated. That question is most of the argument for building a physics model at
    # all, so keep the raw predictions rather than only their pooled summary.
    dump = {"fold": [], "entry": [], "nn_dist": [], "x": [], "y": [], "layer": []}
    dump_arrays = {"pred": [], "idw": [], "obs": []}
    # coordinates used for the nearest-training-entry distance
    dist_xy = (np.asarray(well_xy, dtype="float64") if well_xy is not None
               else grid.centroids()[obs_idx.cpu().numpy()])
    if well_xy is not None:
        n_sites = len(np.unique(groups))
        print(f"    folds grouped by site: {W} entries over {n_sites} physical sites, "
              f"held-out/training co-location rate = {coloc:.3f}", flush=True)
    preds, idws, targets = [], [], []
    per_fold = []
    for f, held in enumerate(folds):
        t_fold = time.perf_counter()
        held = np.asarray(held)
        keep = np.setdiff1d(np.arange(W), held)
        h0_fold = None
        if well_xy is not None and obs_h0 is not None:
            h0_fold = _idw_initial_heads(grid, well_xy[keep], np.asarray(obs_h0)[keep],
                                         obs_layer_np[keep], n_layers)
        m = FlowModel(grid, n_layers=n_layers, dt_days=30.0)
        fit = fit_flow(m, obs_h[keep], obs_idx[keep], obs_layer[keep], recharge,
                       E=E, ground_elev=ground_elev, epochs=epochs, lr=lr,
                       param_mode=param_mode, h0=h0_fold, recharge_field=recharge_field,
                       pump_layer=pump_layer, recharge_layer=recharge_layer,
                       zone_of_cell=zone_of_cell)
        print(f"    fold {f + 1}/{n_folds}: n_held={len(held)} loss={fit['loss']:.4g} "
              f"({time.perf_counter() - t_fold:.1f}s)", flush=True)
        with torch.no_grad():
            h0_eval = (h0_fold if h0_fold is not None
                      else torch.zeros(n_layers, n_active, dtype=torch.float64))
            if (param_mode in ("homogeneous", "zonal")
                    and (E is not None or recharge_field is not None)):
                h = _predict_homogeneous(m, fit, h0_eval, n_steps, recharge=recharge,
                                         recharge_field=recharge_field, E=E,
                                         ground_elev=ground_elev,
                                         recharge_layer=recharge_layer, pump_layer=pump_layer)
            else:
                h = m(h0_eval, recharge,
                     torch.zeros(n_layers, n_active, n_steps, dtype=torch.float64), n_steps)
            p = h[obs_layer[held], obs_idx[held], 1:].numpy()
        src = xy[obs_idx[keep].numpy()]
        tgt = xy[obs_idx[held].numpy()]
        idw = idw_interp(tgt, src, obs_h[keep].numpy())
        preds.append(p)
        idws.append(idw)
        targets.append(obs_h[held].numpy())
        if dump_path is not None:
            dd = np.sqrt(((dist_xy[held][:, None, :] - dist_xy[keep][None, :, :]) ** 2)
                         .sum(-1)).min(axis=1)
            dump["fold"].append(np.full(len(held), f, dtype="int64"))
            dump["entry"].append(np.asarray(held, dtype="int64"))
            dump["nn_dist"].append(dd)
            dump["x"].append(dist_xy[held][:, 0])
            dump["y"].append(dist_xy[held][:, 1])
            dump["layer"].append(obs_layer_np[held])
            dump_arrays["pred"].append(p)
            dump_arrays["idw"].append(idw)
            dump_arrays["obs"].append(obs_h[held].numpy())
        per_fold.append({"fold": f, "n_held": len(held), "fit_loss": fit["loss"],
                         "r2_kfold": _r2(p, obs_h[held].numpy()),
                         "r2_idw": _r2(idw, obs_h[held].numpy())})
    pred = np.concatenate(preds)
    obs = np.concatenate(targets)
    idw_all = np.concatenate(idws)
    if dump_path is not None:
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        np.savez_compressed(
            dump_path,
            **{k: np.concatenate(v) for k, v in dump.items()},
            **{k: np.concatenate(v) for k, v in dump_arrays.items()},
            obs_mean=np.array(float(obs[np.isfinite(obs)].mean())))
        print(f"    wrote per-entry predictions -> {dump_path}", flush=True)
    return {"r2_kfold": _r2(pred, obs), "r2_idw": _r2(idw_all, obs),
            "n_wells": W, "n_folds": n_folds, "per_fold": per_fold,
            "n_sites": int(len(np.unique(groups))) if groups is not None else W,
            "colocation_rate": coloc}


def _load_ground_elev(grid, stn: pd.DataFrame) -> torch.Tensor:
    """IDW the fan stations' ``GroundHeight`` (surface elevation, m) to every active cell."""
    from .heads import _station_xy

    xy, ge = [], []
    for _, row in stn.iterrows():
        gh = row.get("GroundHeight")
        if gh is None or not np.isfinite(gh):
            continue
        p = _station_xy(row)
        if p is None:
            continue
        xy.append(p)
        ge.append(float(gh))
    if not xy:
        raise ValueError("no station carried a finite GroundHeight -- cannot build a "
                         "ground-elevation field for the pumping driver")
    return torch.tensor(_idw_field(grid, np.array(xy), np.array(ge)), dtype=torch.float64)


def _load_pumping_kwh(grid, pump_census_path: str, pump_kwh_path: str,
                      t0: str, t1: str) -> torch.Tensor:
    """Monthly electricity census -> (A, T) kWh per active cell (Task 4's aggregate_pumps)."""
    for p, label in ((pump_census_path, "pump census"), (pump_kwh_path, "pump kWh")):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{label} parquet not found at {p!r} -- fix round 1's pumping driver "
                "needs it; stopping rather than silently skipping it. See "
                "task-5-report.md's fix-round-1 section for where the real 116,769-pump "
                "census currently lives (it has no stable in-repo path yet).")
    pumps = pd.read_parquet(pump_census_path)
    kwh = pd.read_parquet(pump_kwh_path)
    E, _dates = pumping_mod.aggregate_pumps(pumps, kwh, grid, t0, t1)
    return torch.tensor(E, dtype=torch.float64)


def _load_recharge_field(grid, rf_timeseries_path: str, rf_stations_path: str,
                         et_npz_path: str, gw_stations_path: str,
                         t0: str, t1: str) -> torch.Tensor:
    """Monthly (rain - ET0), IDW'd to every active cell, clamped >= 0, in m/day.

    Rainfall: 26 daily rain gauges (``rf_timeseries.csv``/``rf_stations.csv``). ET0: the
    cached OpenMeteo ET0 (``openmeteo_et0_2012_2022.npz``), one daily series per one of
    the original 61 curated wells, matched to coordinates via ``gw_stations.csv``'s
    ``st_id``. Both are aggregated to each calendar month's MEAN daily rate (not the
    monthly total) before IDW, matching ``FlowModel``'s convention that ``recharge`` is a
    rate applied for the whole (assumed-uniform) ``dt``-day step, not a lump sum for the
    month.
    """
    for p, label in ((rf_timeseries_path, "rainfall timeseries"),
                     (rf_stations_path, "rain-gauge stations"),
                     (et_npz_path, "cached ET0"), (gw_stations_path, "gw stations")):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{label} not found at {p!r} -- fix round 1's recharge driver needs it; "
                "stopping rather than silently skipping it.")
    months = pd.date_range(t0, t1, freq="ME")

    rf = pd.read_csv(rf_timeseries_path)
    rf["date time"] = pd.to_datetime(rf["date time"])
    rf = rf.set_index("date time")
    rf_monthly = rf.resample("ME").mean().reindex(months)     # mean daily mm rate/month
    rf_stn = pd.read_csv(rf_stations_path).set_index("rf_id")
    rf_xy = rf_stn.loc[rf_monthly.columns, ["TM_X97", "TM_Y97"]].to_numpy(dtype="float64")

    et = np.load(et_npz_path, allow_pickle=True)
    et_df = pd.DataFrame(et["et0"].T, index=pd.DatetimeIndex(et["dates"]),
                         columns=et["well_ids"])
    et_monthly = et_df.resample("ME").mean().reindex(months)
    gw_stn = pd.read_csv(gw_stations_path).set_index("st_id")
    et_xy = gw_stn.loc[et_monthly.columns, ["TM_X97", "TM_Y97"]].to_numpy(dtype="float64")

    A = grid.n_active
    field = np.zeros((A, len(months)), dtype="float64")
    for i in range(len(months)):
        rain_cell = _idw_field(grid, rf_xy, rf_monthly.iloc[i].to_numpy(dtype="float64"))
        et_cell = _idw_field(grid, et_xy, et_monthly.iloc[i].to_numpy(dtype="float64"))
        field[:, i] = np.clip((rain_cell - et_cell) / 1000.0, a_min=0.0, a_max=None)
    return torch.tensor(field, dtype=torch.float64)


# The pump census (sid, TWD97_X, TWD97_Y, PUMP_HP, PURPOSE; 116,769 rows) has no stable
# in-repo path -- see task-5-report.md's fix-round-1 section. This points at the copy
# Task 4 fetched live from the wisenvr API into this session's scratchpad, which is what
# --pump-census defaults to; pass --pump-census explicitly (or --no-forcing) elsewhere.
_DEFAULT_PUMP_CENSUS = (
    "/tmp/claude-1000/-home-rekin226-Desktop-code-space-HydroPhysicsAI/"
    "55ec15e7-c585-41a8-a463-e69e4ca3c0cf/scratchpad/tpc_pumps.parquet"
)

_PARAM_MODES = ("homogeneous", "percell", "zonal")
_DEFAULT_ZONE_BOUNDARIES = "205,182"


def _parse_zone_boundaries(text: str) -> tuple[float, float]:
    """``"205,182"`` -> ``(205.0, 182.0)``: the proximal/mid and mid/distal eastings in km.

    Spec §4.2 requires re-running the gate at 178 and 186 km, because the mid/distal
    boundary is an equal-width default with no independent justification. A transposed
    pair would silently empty the mid zone, so it is rejected rather than tolerated.
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(
            f"--zone-boundaries wants PROXIMAL_KM,DISTAL_KM (e.g. '205,182'), got {text!r}"
        )
    try:
        proximal_km, distal_km = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"--zone-boundaries values must be numbers, got {text!r}") from exc
    if not distal_km < proximal_km:
        raise ValueError(
            f"--zone-boundaries wants PROXIMAL_KM,DISTAL_KM with the proximal boundary "
            f"east of the distal one; got proximal={proximal_km}, distal={distal_km}"
        )
    return proximal_km, distal_km


def _format_bounds_hit(hits: dict) -> str:
    """Per-zone dicts print one line per zone; flat dicts print as before.

    Spec §6's primary rule reads this, and pooling would let an interior mid value mask
    a pinned proximal one -- so the zonal form is never collapsed into a single number.
    """
    if hits and all(isinstance(v, dict) for v in hits.values()):
        return "\n".join(f"    bounds_hit[{zone}]={vals}" for zone, vals in hits.items())
    return f"  bounds_hit={hits}"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stage-3 flow calibration and k-fold gate")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0,
                    help="fold-assignment seed. Vary it to measure whether a gate "
                         "margin is signal or fold noise; the 2026-08-27 gate came "
                         "down to 0.033, which one split cannot resolve.")
    ap.add_argument("--param-mode", choices=list(_PARAM_MODES), default="homogeneous")
    ap.add_argument("--zone-boundaries", default=_DEFAULT_ZONE_BOUNDARIES,
                    help="PROXIMAL_KM,DISTAL_KM for --param-mode zonal (default "
                         "'205,182'). The mid/distal value is an unjustified default; "
                         "spec 4.2 requires re-running at 178 and 186 to report whether "
                         "the verdict moves.")
    ap.add_argument("--wells-dir", default="AMP_V2/data/wells")
    ap.add_argument("--stations", default="AMP_V2/data/fan_stations.parquet")
    ap.add_argument("--polygon",
                    default="chou-shui-data/chou-shui-data/data/Zhuoshui Alluvial Fan/"
                            "Zhuoshui Alluvial Fan.json")
    ap.add_argument("--pump-census", default=_DEFAULT_PUMP_CENSUS,
                    help="pump census parquet (sid, TWD97_X, TWD97_Y, PUMP_HP, PURPOSE) "
                         "-- see task-5-report.md fix-round-1 for why this has no stable "
                         "repo path yet")
    ap.add_argument("--pump-kwh", default="AMP_V2/data/pump_kwh_all.parquet")
    ap.add_argument("--rf-timeseries",
                    default="chou-shui-data/chou-shui-data/data/rf_timeseries.csv")
    ap.add_argument("--rf-stations",
                    default="chou-shui-data/chou-shui-data/data/rf_stations.csv")
    ap.add_argument("--et-npz", default="results/et/openmeteo_et0_2012_2022.npz")
    ap.add_argument("--gw-stations",
                    default="chou-shui-data/chou-shui-data/data/gw_stations.csv")
    ap.add_argument("--pump-layer", type=int, default=1,
                    help="0-indexed layer that receives pumping (default 1 = layer 2, "
                         "the main production aquifer)")
    ap.add_argument("--recharge-layer", type=int, default=0,
                    help="0-indexed layer that receives rain-minus-ET0 recharge")
    ap.add_argument("--no-forcing", action="store_true",
                    help="skip the pumping/recharge drivers and h0 IC, reproducing the "
                         "original (degenerate, see module docstring) zero-forcing run")
    ap.add_argument("--out", default="results/twin")
    ap.add_argument("--log-every", type=int, default=0,
                    help="print in-sample R2 every N epochs during the fit")
    ap.add_argument("--dump-predictions", action="store_true",
                    help="write per-held-out-entry predictions (flow, IDW, obs) plus\neach entry's distance to the nearest training entry, for degradation-vs-distance\nanalysis without refitting.")
    ap.add_argument("--fit-only", action="store_true",
                    help="run the in-sample fit and skip the k-fold gate")
    args = ap.parse_args(argv)

    from .heads import build_head_field

    grid = build_grid(args.polygon, dx=args.dx)

    zone_of_cell = zone_counts = proximal_km = distal_km = None
    if args.param_mode == "zonal":
        proximal_km, distal_km = _parse_zone_boundaries(args.zone_boundaries)
        zone_of_cell = fan_zones(grid.centroids(), proximal_km=proximal_km,
                                 distal_km=distal_km)
        zone_counts = {name: int((zone_of_cell == i).sum())
                       for i, name in enumerate(ZONE_NAMES)}
        print(f"zones: proximal/mid at {proximal_km:.0f} km, mid/distal at "
              f"{distal_km:.0f} km -> cells {zone_counts}", flush=True)
        empty = [n for n, c in zone_counts.items() if c == 0]
        if empty:
            raise SystemExit(
                f"zone(s) {empty} contain no active cells at these boundaries; "
                "the gate would score a model with fewer zones than it reports"
            )

    stn = pd.read_parquet(args.stations)
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(args.wells_dir, stn)

    idx, lay, series, xy_used = [], [], [], []
    for w in range(len(hf)):
        i = grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1]))
        if i is None:
            continue
        s = hf.heads[w]
        if not np.isfinite(s).all():
            s = pd.Series(s).interpolate(limit_direction="both").to_numpy()
        idx.append(i)
        lay.append(max(int(hf.layers[w]) - 1, 0))
        series.append(s)
        xy_used.append(hf.xy[w])
    obs_h_full = np.stack(series)                                  # (W, 132), raw heads
    well_xy = np.array(xy_used, dtype="float64")
    obs_layer_np = np.array(lay, dtype="int64")
    obs_h0 = obs_h_full[:, 0]                       # each well's first-observed head
    # Fix round 1: the target is the RAW head from month 2 on; month 1 is consumed below
    # to build a real initial condition (_idw_initial_heads) instead of anomaly-shifting
    # the series against a hardcoded h0=0 the way the first (invalid) run did.
    obs_h = torch.tensor(obs_h_full[:, 1:], dtype=torch.float64)
    obs_idx = torch.tensor(idx, dtype=torch.long)
    obs_layer = torch.tensor(lay, dtype=torch.long)
    n_steps = obs_h.shape[1]
    recharge_dummy = torch.zeros(4, grid.n_active, n_steps, dtype=torch.float64)

    ground_elev = E = recharge_field = None
    if not args.no_forcing:
        ground_elev = _load_ground_elev(grid, stn)
        E = _load_pumping_kwh(grid, args.pump_census, args.pump_kwh,
                              "2012-01-01", "2023-01-01")[:, 1:]
        recharge_field = _load_recharge_field(grid, args.rf_timeseries, args.rf_stations,
                                              args.et_npz, args.gw_stations,
                                              "2012-01-01", "2023-01-01")[:, 1:]

    h0_all = _idw_initial_heads(grid, well_xy, obs_h0, obs_layer_np, n_layers=4)

    m = FlowModel(grid, n_layers=4, dt_days=30.0)
    t0 = time.perf_counter()
    ins = fit_flow(m, obs_h, obs_idx, obs_layer, recharge_dummy, E=E, ground_elev=ground_elev,
                   epochs=args.epochs, lr=args.lr, param_mode=args.param_mode, h0=h0_all,
                   recharge_field=recharge_field, pump_layer=args.pump_layer,
                   recharge_layer=args.recharge_layer, log_every=args.log_every,
                   zone_of_cell=zone_of_cell)
    t_fit = time.perf_counter() - t0

    if args.fit_only:
        # Discriminator mode: the in-sample TRAJECTORY separates under-training from a
        # structurally inadequate parameterisation, and costs one fit instead of eleven.
        print(f"wells={obs_h.shape[0]} "
              f"cells={grid.n_active} dx={args.dx:.0f}m param_mode={args.param_mode} "
              f"n_params={ins['n_params']} epochs={args.epochs}")
        print(f"  in-sample R2={ins['r2']:+.3f}  fit_time={t_fit:.1f}s")
        print(_format_bounds_hit(ins["bounds_hit"]))
        tr = ins.get("r2_trace") or []
        if tr:
            print("  trajectory: " + "  ".join(f"{e}:{r:+.3f}" for e, r in tr))
            last = [r for _, r in tr[-3:]]
            if len(last) >= 2:
                drift = last[-1] - last[0]
                print(f"  change over the last {len(last)} logged points: {drift:+.4f} "
                      f"-> {'STILL CLIMBING (under-trained)' if drift > 0.01 else 'PLATEAUED (structural)'}")
        os.makedirs(args.out, exist_ok=True)
        pd.DataFrame(tr, columns=["epoch", "r2_insample"]).to_csv(
            os.path.join(args.out, "stage3_fit_trace.csv"), index=False)
        print(f"wrote {os.path.join(args.out, 'stage3_fit_trace.csv')}")
        return

    t0 = time.perf_counter()
    gate = kfold_wells(grid, obs_h, obs_idx, obs_layer, recharge_dummy, n_layers=4,
                       epochs=args.epochs, lr=args.lr, n_folds=args.n_folds,
                       seed=args.seed,
                       param_mode=args.param_mode, well_xy=well_xy, obs_h0=obs_h0,
                       ground_elev=ground_elev, E=E, recharge_field=recharge_field,
                       pump_layer=args.pump_layer, recharge_layer=args.recharge_layer,
                       dump_path=(os.path.join(args.out, "stage3_per_entry.npz")
                                  if args.dump_predictions else None),
                       zone_of_cell=zone_of_cell)
    t_gate = time.perf_counter() - t0
    print(f"wells={gate['n_wells']} cells={grid.n_active} dx={args.dx:.0f}m "
          f"param_mode={args.param_mode} n_params={ins['n_params']} "
          f"forcing={'off' if args.no_forcing else 'on'} epochs={args.epochs}")
    print(f"  in-sample R2={ins['r2']:+.3f}  fit_time={t_fit:.1f}s")
    print(_format_bounds_hit(ins["bounds_hit"]))
    if "theta" in ins:
        print(f"  theta={ins['theta']}")
    print(f"  {args.n_folds}-fold R2={gate['r2_kfold']:+.3f}   "
          f"IDW baseline R2={gate['r2_idw']:+.3f}  gate_time={t_gate:.1f}s")
    print(f"  folds grouped by site: {gate['n_wells']} entries / {gate['n_sites']} sites   "
          f"co-location rate={gate['colocation_rate']:.3f}")
    if not (gate["colocation_rate"] >= 0.0 and gate["colocation_rate"] < 1e-9):
        print("  WARNING: held-out entries sit at zero distance from training entries, so "
              "the IDW baseline is reading co-located screens rather than interpolating. "
              "Treat the comparison below as biased toward IDW.")
    print(f"GATE ({args.n_folds}-fold): "
          f"{'PASS' if gate['r2_kfold'] > gate['r2_idw'] else 'FAIL'}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stage3_flow.csv")
    pd.DataFrame([{"n_wells": gate["n_wells"], "n_cells": grid.n_active, "dx": args.dx,
                   "param_mode": args.param_mode,
                   "zone_proximal_km": (proximal_km if args.param_mode == "zonal"
                                        else ""),
                   "zone_distal_km": (distal_km if args.param_mode == "zonal" else ""),
                   "zone_cell_counts": (str(zone_counts) if args.param_mode == "zonal"
                                        else ""),
                   "n_params": ins["n_params"],
                   "forcing": "off" if args.no_forcing else "on", "epochs": args.epochs,
                   "n_folds": gate["n_folds"], "seed": args.seed,
                   "n_sites": gate["n_sites"],
                   "colocation_rate": gate["colocation_rate"], "loss": ins["loss"],
                   "r2_insample": ins["r2"], "r2_kfold": gate["r2_kfold"],
                   "r2_idw": gate["r2_idw"], "bounds_hit": str(ins["bounds_hit"]),
                   "theta": str(ins.get("theta", {})),
                   "fit_time_s": t_fit, "gate_time_s": t_gate}]).to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
