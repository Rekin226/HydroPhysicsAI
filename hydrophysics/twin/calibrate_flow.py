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
import json
import math
import os
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..subsidence import idw_interp
from ..train import pick_device
from . import pumping as pumping_mod
from .boundaries import fan_boundaries
from .flow import (
    _CG_CHECK_EVERY,
    _CG_MAXITER,
    FlowModel,
    _cg_stats,
    _ImplicitSolve,
    _reset_cg_stats,
    _warm_started_solver,
    set_compile_matvec,
)
from .grid import build_grid
from .spread import SPREAD_KM_BOUNDS, pairwise_d2_km, spread_energy, spread_matrix
from .zones import (
    N_ZONES,
    PROXIMAL,
    PROXIMAL_W,
    ZONE_NAMES,
    ZONE_NAMES_SPLIT,
    collapse_zones,
    fan_zones,
    zone_blend_weights,
    zone_names,
)

# Physically defensible bounds. log_T is tightened per Ruling 3 above (Task-2 CG
# conditioning finding); log_S and log_L keep the brief's bounds. log_eta (fix round 1)
# is the brief's original wire-to-water-efficiency bound.
RETURN_FRAC_MAX = 0.7      # irrigation return flow cannot exceed this share of pumping


def set_l_min(l_min: float | None) -> None:
    """Raise the leakance floor (``--l-min``, 1/day). The free fits drive log_L to
    1e-8..1e-5, i.e. isolated layers; a physical aquitard on this fan leaks more than
    that, and an isolated production layer is what forces storage to its ceiling."""
    if l_min is not None:
        BOUNDS["log_L"] = (math.log(float(l_min)), BOUNDS["log_L"][1])


BOUNDS = {
    "log_T": (math.log(10.0), math.log(2e4)),        # m2/day (Liu et al. 2002)
    "log_S": (math.log(1e-6), math.log(0.3)),        # -
    "log_L": (math.log(1e-8), math.log(1e-1)),       # 1/day
    "log_eta": (math.log(0.05), math.log(0.9)),      # wire-to-water efficiency, -
    # Total dynamic head = static lift + this. Covers well drawdown, entrance/friction
    # losses and the distribution system's discharge head -- everything the pump works
    # against besides raising water to ground level. Bounded 1-200 m: an irrigation well
    # with no drawdown and gravity delivery sits near the floor, and 200 m is generous for
    # a deep well on sprinklers. See pumping.energy_to_volume for why omitting this term
    # cost the Stage-3 gate a factor of ~12 and pinned log_eta on its lower clamp.
    "log_head_extra": (math.log(1.0), math.log(200.0)),   # m
    # General-head boundary conductance per exposed face, m2/day (boundaries.py). A
    # Dirichlet face sits at C = 2T, so the ceiling clears the log_T ceiling with room;
    # the floor is effectively closed (1e-2 m2/day against T >= 10), which is how the
    # data get to say "no boundary here" if that is what they say.
    "log_C_coast": (math.log(1e-2), math.log(1e5)),
    "log_C_apex": (math.log(1e-2), math.log(1e5)),
    "log_spread_km": SPREAD_KM_BOUNDS,             # spread.py: learned stress radius
    # Opt-in physics of 2026-09-23 (G1/G6), each behind a flag whose default is off:
    # lumped delay bed (--delay-storage): slow-store storativity and time constant (days;
    # the ceiling follows --delay-tau-max-years via set_delay_tau_max)
    "log_Sd": (math.log(1e-5), math.log(0.3)),
    "log_tau": (math.log(30.0), math.log(365.25 * 30.0)),
    # river conductance per unit channel weight, m2/day (--rivers)
    "log_C_riv": (math.log(1e-1), math.log(1e6)),
    # recharge fraction of surface-water irrigation deliveries (--sw-recharge); above 1
    # would say the delivery map is under-scaled, which is worth being able to see
    "log_sw_scale": (math.log(0.01), math.log(2.0)),
    # --delay-u0 learned: the slow store's head above the aquifer's at the record's start
    # (m). The record opens after decades of drawdown, so real interbeds still drain.
    "log_du0": (math.log(0.01), math.log(50.0)),
    # --aquitard-storage: storativity of the aquitard store between two layers (-) and its
    # conductance to each of them (1/day, the log_L range)
    "log_Sa": (math.log(1e-5), math.log(0.3)),
    "log_G": (math.log(1e-8), math.log(1e-1)),
}
DELAY_SD_INIT = 1e-3
DELAY_TAU_INIT_DAYS = 365.0
DELAY_DU0_INIT_M = 1.0
AQT_SA_INIT = 1e-3
AQT_G_INIT = 1e-4
C_RIV_INIT = 1e3
# Liu et al. (2001, 2005): 20-35 % of what Yunlin's canals deliver infiltrates
SW_SCALE_INIT = 0.25
# two components (canal leakage, paddy percolation; surface_water.py --components):
# leakage of delivered water 0.1-0.2 once percolation is separate, k_p ~0.25 (Liu 2001)
SW_SCALE_INIT_2 = (0.15, 0.25)
DELAY_TAU_MIN_DAYS = 30.0

# --- per-well datum in the observation operator (--well-datum fit, opt-in 2026-09-26) ---
WELL_DATUM_MODES = ("off", "fit")
WELL_DATUM_SD_DEFAULT = 5.0
# Monthly residuals of one well are strongly autocorrelated, so a well's n observed months
# carry about n / 12 independent looks at its level. The prior's weight is expressed in
# months with this factor, so it is not swamped by counting 132 correlated months as 132
# independent observations. A documented modelling constant, not a tuned one.
WELL_DATUM_MONTHS_PER_OBS = 12.0


def _profile_well_datum(pred: torch.Tensor, obs_z: torch.Tensor, mask: torch.Tensor | None,
                        sd: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                            torch.Tensor]:
    """The MAP per-well datum ``d`` (``(W, 1)``, detached) for the current prediction.

    The observation operator is ``pred_i(t) = h(cell_i, layer_i, t) + d_i`` with the prior
    ``d_i ~ N(0, sd^2)``. The loss is quadratic in ``d``, so its minimum is exact and
    closed-form (variable projection); the physical parameters are then optimised against
    the profiled loss, and by the envelope theorem their gradient is the joint one.

    Scaling. With per-month residual variance ``s^2`` (the within-well residual variance,
    ``r - mean_i(r)``, which does not depend on ``d``) and ``WELL_DATUM_MONTHS_PER_OBS``
    (``m``) correlated months per independent observation, well ``i``'s ``n_i`` observed
    months carry ``n_eff_i = n_i / m`` independent looks at its level, but never fewer
    than one: ``n_eff_i = max(n_i / m, 1)`` (a few months of a strongly autocorrelated
    series are one look, not a fraction of one). The negative log posterior, multiplied
    by the data term's own normalisation, is

        MSE(pred + d, obs) + sum_i kappa_i d_i^2 / N,
        kappa_i = n_i / n_eff_i * s^2 / sd^2 = min(n_i, m) s^2 / sd^2,

    with ``N`` the number of observed cells, so

        d_i = n_i rbar_i / (n_i + kappa_i) = rbar_i n_eff_i / (n_eff_i + s^2 / sd^2),

    with ``rbar_i = mean_t(obs - pred)_i``.

    A well observed for years (``n_i >> m``) keeps nearly its full mean residual; a well
    with at most ``m`` observed months counts as one observation and keeps
    ``1 / (1 + s^2 / sd^2)`` of it. Returns ``(d, kappa, s2, n)``: ``kappa`` the ``(W, 1)``
    per-well ``kappa_i`` (months) and ``n`` the ``(W, 1)`` observed-month counts."""
    with torch.no_grad():
        r = obs_z - pred
        mf = (torch.ones_like(pred) if mask is None else mask.to(pred.dtype))
        n = mf.sum(dim=1, keepdim=True)
        rbar = (r * mf).sum(dim=1, keepdim=True) / n.clamp_min(1.0)
        s2 = (((r - rbar) ** 2) * mf).sum() / mf.sum().clamp_min(1.0)
        kappa = n.clamp_max(WELL_DATUM_MONTHS_PER_OBS) * s2 / float(sd) ** 2
        d = torch.where(n > 0, n * rbar / (n + kappa).clamp_min(1e-12),
                        torch.zeros_like(rbar))
    return d, kappa, s2, n


def well_datum_vector(meta: dict | None, sids) -> np.ndarray | None:
    """``(W,)`` fitted datum per well of ``sids`` from a theta meta's ``well_datum``
    (``{sid: m}``), 0 for a well without one; ``None`` when the run fitted none. For
    nuisance use only (e.g. ``uncertainty``'s residual): the datum never enters a flow
    solve, a column, a forward projection or subsidence."""
    wd = (meta or {}).get("well_datum")
    if not wd:
        return None
    return np.array([float(wd.get(str(s), 0.0)) for s in sids], dtype="float64")


def well_datum_stats(d: np.ndarray, n: np.ndarray | None = None,
                     kappa: float | None = None) -> dict:
    """Summary of a fitted datum vector for the meta and the CSVs."""
    d = np.asarray(d, dtype="float64").ravel()
    out = {"n": int(d.size), "mean_m": float(d.mean()) if d.size else float("nan"),
           "mean_abs_m": float(np.abs(d).mean()) if d.size else float("nan"),
           "rms_m": float(np.sqrt((d ** 2).mean())) if d.size else float("nan"),
           "max_abs_m": float(np.abs(d).max()) if d.size else float("nan")}
    if kappa is not None:
        # kappa: the long-record prior weight m s^2 / sd^2; well i's is min(n_i, m) / m x it
        out["kappa_months"] = float(kappa)
        if n is not None and np.size(n):
            nn = np.asarray(n, dtype="float64").ravel()
            k_i = np.minimum(nn, WELL_DATUM_MONTHS_PER_OBS) / WELL_DATUM_MONTHS_PER_OBS * kappa
            shrink = np.where(nn > 0, nn / np.maximum(nn + k_i, 1e-12), 0.0)
            out["shrink_min"] = float(shrink.min())
            out["shrink_median"] = float(np.median(shrink))
    return out


def set_delay_tau_max(years: float | None) -> None:
    """Delay-bed time-constant ceiling (``--delay-tau-max-years``, default 30)."""
    if years is not None:
        BOUNDS["log_tau"] = (BOUNDS["log_tau"][0], math.log(365.25 * float(years)))


def set_delay_tau_min(days: float | None) -> None:
    """Delay-bed time-constant floor (``--delay-tau-min-days``, default 30). At tau ~ dt
    the bed is only extra instant storage (``S_d dt/(tau+dt)``), a route around the
    storage ceiling rather than a slow release; 180 days keeps it slow."""
    if days is not None:
        BOUNDS["log_tau"] = (math.log(float(days)), BOUNDS["log_tau"][1])


def set_spread_max_km(km: float | None) -> None:
    """Ceiling of the learned stress radius (``--spread-max-km``, default 25). The
    deliverable (stage3_spreadL_gate) was fitted at 10 km, where it sits."""
    if km is not None:
        BOUNDS["log_spread_km"] = (BOUNDS["log_spread_km"][0], math.log(float(km)))


# Per-zone raised floors, ``{(base, zone): log_lo}``, read by ``_zonal_bounds_hit``. Empty
# by default (every zone uses BOUNDS). Set by ``set_log_t_min_proximal``.
ZONE_LOWER_BOUNDS: dict[tuple[str, str], float] = {}


def set_log_t_min_proximal(m2day: float | None) -> None:
    """``--log-t-min-proximal`` (m2/day): raise the proximal ``log_T`` floor above the
    global 10 m2/day of ``BOUNDS["log_T"]``, in both proximal parts when the zone is
    split. The proximal gravel is the most transmissive part of the fan, and Liu et al.
    (2002) measured 58 m2/day as the fan-wide minimum. ``None`` clears the override."""
    for z in ("proximal", "proximal_w"):
        ZONE_LOWER_BOUNDS.pop(("log_T", z), None)
    if m2day is None:
        return
    if not float(m2day) > 0.0:
        raise ValueError(f"--log-t-min-proximal must be > 0 m2/day, got {m2day}")
    lo = max(math.log(float(m2day)), BOUNDS["log_T"][0])
    if lo >= BOUNDS["log_T"][1]:
        raise ValueError(f"--log-t-min-proximal {m2day} is not below the log_T ceiling")
    for z in ("proximal", "proximal_w"):
        ZONE_LOWER_BOUNDS[("log_T", z)] = lo


def parse_delay_layers(text: str | None, n_layers: int = 4) -> tuple[int, ...] | None:
    """``--delay-layers "1,2,3"`` (0-based layer indices) -> tuple, or ``None`` = all."""
    if text is None or str(text).strip() in ("", "all"):
        return None
    out = tuple(sorted({int(s) for s in str(text).split(",") if s.strip()}))
    if not out or min(out) < 0 or max(out) >= n_layers:
        raise ValueError(f"--delay-layers {text!r}: expected indices in 0..{n_layers - 1}")
    return out


def _git_commit() -> str:
    """Short commit SHA for the result artifact's provenance, or "" if it cannot be
    determined (e.g. not a git checkout) -- never raises, since a multi-hour sweep must
    not die on a provenance nicety.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _clamp_(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, int]]:
    """Clamp each named tensor into BOUNDS in place; return, per parameter, how many
    entries sit on the LOWER bound, how many on the UPPER bound, and the total entry
    count -- ``{"lo": int, "hi": int, "n": int}``.

    Ruling P1: spec 1's documented failure is fitting BELOW the measured physical range
    (three of four log_T layers pinned at the lower clamp, 10 m2/day, under the 58 m2/day
    floor Liu et al. 2002 measured). An upper-bound hit is a different physical statement
    and must never be pooled with a lower-bound one. The clamping behaviour itself is
    unchanged from before this split -- only what gets reported.
    """
    hits: dict[str, dict[str, int]] = {}
    with torch.no_grad():
        for name, par in tensors.items():
            if par is None:
                continue
            lo, hi = BOUNDS[name]
            par.clamp_(min=lo, max=hi)
            n_lo = int((par <= lo + 1e-9).sum())
            n_hi = int((par >= hi - 1e-9).sum())
            hits[name] = {"lo": n_lo, "hi": n_hi, "n": int(par.numel())}
    return hits


def prepare_series(s: np.ndarray, backfill: bool = True) -> np.ndarray:
    """One well's monthly head series as the calibration target.

    ``backfill=True`` (default, every run before 2026-09-23): linear interpolation across
    interior gaps and constant fill of leading/trailing gaps
    (``interpolate(limit_direction="both")``). That fill leaks held-out values into a
    temporal screen: a well that starts in 2020 gets its first held-out head copied over
    every fitted month, into the fit target, the climatology and the initial condition.

    ``backfill=False`` (``--no-backfill``): the series is returned unchanged -- every
    never-observed month stays NaN, interior gaps included (an interior gap straddling
    the fit/held-out split would leak the same way), and ``fit_flow`` masks those months
    out of the loss."""
    s = np.asarray(s, dtype="float64")
    if not backfill or np.isfinite(s).all():
        return s.copy() if not backfill else s
    return pd.Series(s).interpolate(limit_direction="both").to_numpy()


def _masked_mse(pred: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the ``mask``-ed entries only (``obs`` must be finite
    everywhere -- zero-filled where masked -- so no NaN reaches the gradient)."""
    mf = mask.to(pred.dtype)
    return (((pred - obs) ** 2) * mf).sum() / mf.sum().clamp_min(1.0)


def _r2(pred: np.ndarray, obs: np.ndarray) -> float:
    finite = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[finite], obs[finite]
    if obs.size == 0:
        return float("nan")      # e.g. a --no-backfill well never observed in the window
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


def _merged_proximal_heads(grid, h0: torch.Tensor, xy: np.ndarray, h0_values: np.ndarray,
                           zone_of_cell: np.ndarray, well_zone: np.ndarray,
                           layer_of: np.ndarray | None = None
                           ) -> tuple[torch.Tensor, dict]:
    """``--ic-merged-proximal`` (opt-in, 2026-09-23): overwrite the proximal part of a
    per-layer initial head field ``h0`` ``(L, A)`` with one merged-aquifer head.

    The proximal fan has no aquitards (``log_L`` is not even a parameter there), yet the
    per-layer IDW of ``_idw_initial_heads`` fills its layers 3-4 -- which have one and zero
    proximal wells -- from mid-fan wells tens of km west, leaving the apex cells 30-40 m
    below their own layer-1 wells. The layers then equilibrate within the first steps and
    ``set_apex_heads`` pins each layer's apex boundary at the wrong head for the whole run.

    Here, for each proximal zone present in ``zone_of_cell`` (``PROXIMAL`` and, with the
    opt-in split, ``PROXIMAL_W``), every layer of its cells gets the same value: the IDW of
    the finite heads of the wells that sit in that zone, whatever their layer code (they all
    screen one aquifer), interpolated to that zone's cells only. A zone without such wells
    falls back to all proximal wells, and without any keeps ``h0``. Other zones are
    untouched. The apex boundary inherits the merged head through ``set_apex_heads(h0)``.
    ``well_zone`` is the zone id of each well's cell. Returns ``(h0_new, report)``.

    ``layer_of`` (``--ic-layered-proximal``, opt-in 2026-09-25; the 0-based layer of each
    well) keeps the per-layer structure inside the proximal zone(s): layer k of a zone's
    cells is the IDW of that zone's layer-k wells only (the same well set as above, zone
    or all-proximal fallback), and a layer without such a well gets the merged value.
    Nothing outside the proximal zone(s) changes. ``None`` (default) is the merged form.
    """
    out = h0.clone()
    xy = np.asarray(xy, dtype="float64").reshape(-1, 2)
    v = np.asarray(h0_values, dtype="float64").reshape(-1)
    wz = np.asarray(well_zone).reshape(-1)
    zc = np.asarray(zone_of_cell).reshape(-1)
    fin = np.isfinite(v)
    prox = (PROXIMAL, PROXIMAL_W)
    any_prox = fin & np.isin(wz, prox)
    pts = grid.centroids()
    names = zone_names(len(ZONE_NAMES_SPLIT))
    report: dict = {}
    for z in prox:
        cells = np.flatnonzero(zc == z)
        if cells.size == 0:
            continue
        m, src = fin & (wz == z), "zone"
        if not m.any():
            m, src = any_prox, "all proximal"
        if not m.any():
            report[names[z]] = {"n_wells": 0, "source": "none (per-layer IDW kept)"}
            continue
        f = idw_interp(pts[cells], xy[m], v[m].reshape(-1, 1))[:, 0]
        before = out[:, cells].median(dim=1).values.tolist()
        out[:, cells] = torch.as_tensor(f, dtype=out.dtype, device=out.device)[None, :]
        report[names[z]] = {"n_wells": int(m.sum()), "source": src, "n_cells": int(cells.size),
                            "median_before_m": [round(float(b), 2) for b in before],
                            "median_after_m": round(float(np.median(f)), 2)}
        if layer_of is not None:
            lay = np.asarray(layer_of).reshape(-1)
            n_k, med_k = [], []
            for k in range(out.shape[0]):
                mk = m & (lay == k)
                n_k.append(int(mk.sum()))
                if mk.any():
                    fk = idw_interp(pts[cells], xy[mk], v[mk].reshape(-1, 1))[:, 0]
                    out[k, cells] = torch.as_tensor(fk, dtype=out.dtype, device=out.device)
                med_k.append(round(float(out[k, cells].median()), 2))
            report[names[z]].update({"layered": True, "n_wells_per_layer": n_k,
                                     "median_after_per_layer_m": med_k})
    return out, report


def _ic_zone_map(grid, zone_boundaries: str | None) -> np.ndarray:
    """The zone map ``--ic-merged-proximal`` uses: the calibration's own boundaries (with
    the optional proximal split), also for a non-zonal parameterisation."""
    p, d, s = _parse_zone_boundaries(zone_boundaries or _DEFAULT_ZONE_BOUNDARIES,
                                     allow_split=True)
    return fan_zones(grid.centroids(), proximal_km=p, distal_km=d, split_km=s)


def _rollout(model: FlowModel, log_T: torch.Tensor, log_S: torch.Tensor,
             log_L: torch.Tensor | None, h0: torch.Tensor, n_steps: int, *,
             recharge: torch.Tensor | None = None, pumping: torch.Tensor | None = None,
             recharge_field: torch.Tensor | None = None,
             recharge_scale: torch.Tensor | None = None, recharge_layer: int = 0,
             E: torch.Tensor | None = None, log_eta: torch.Tensor | None = None,
             ground_elev: torch.Tensor | None = None, pump_layer: int = 1,
             log_head_extra: torch.Tensor | None = None,
             log_C_coast: torch.Tensor | None = None,
             log_C_apex: torch.Tensor | None = None,
             pump_split_logit: torch.Tensor | None = None,
             return_frac_logit: torch.Tensor | None = None,
             spread_W: torch.Tensor | None = None,
             delay_Sd: torch.Tensor | None = None,
             delay_tau: torch.Tensor | None = None,
             u0: torch.Tensor | None = None,
             log_C_riv: torch.Tensor | None = None,
             sw_field: torch.Tensor | None = None,
             log_sw_scale: torch.Tensor | None = None,
             sw_layer: int | None = None,
             aqt_Sa: torch.Tensor | None = None,
             aqt_G: torch.Tensor | None = None,
             river_month0: int | None = None,
             return_state: bool = False):
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

    Opt-in terms of 2026-09-23, each absent unless its arguments are given (and then the
    historical code path runs unchanged, bit for bit):

    - **Delay bed** (``delay_Sd``/``delay_tau``, log, ``(L, A)``): a slow store of head
      ``u`` per cell and layer exchanging ``A S_d/tau (u - h)`` with the aquifer, the
      first-mode lumping of SUB-style delay-bed drainage. Backward Euler with ``u``
      condensed out: ``beta = S_d/(tau + dt)`` enters the operator's diagonal (implicit
      adjoint) and ``beta A u`` the right-hand side, then ``u' = (tau u + dt h')/(tau +
      dt)``. ``u0`` defaults to ``h0`` (equilibrium). ``return_state=True`` returns
      ``(heads, u_final)`` so a projection can continue from the record's slow state.
    - **Rivers** (``log_C_riv``, one per group, needs ``model.set_rivers``): ``ghb`` or
      lagged-switch ``riv`` exchange on the river layer; see ``FlowModel.river_terms``.
    - **Surface-water irrigation recharge** (``sw_field`` (A, T) m/day of delivered water,
      ``log_sw_scale`` its recharge fraction): added to ``sw_layer`` (default
      ``recharge_layer``). With several components (``sw_field`` (K, A, T), e.g. canal
      deliveries and paddy percolation, ``log_sw_scale`` (K,)) each has its own fraction.
    - **Aquitard store** (``aqt_Sa``/``aqt_G``, log, ``(L-1, A)``; ``--aquitard-storage``):
      a storage node between layers k and k+1, see ``FlowModel.aqt_terms``. Its state
      starts at the mean of the two layers' heads (equilibrium).
    - **Seasonal river connection** (``model.river_season``): ``river_month0`` is the
      calendar month index (0 = January) of step 0; default 1, the calibration record,
      whose first step is February 2012.

    With ``return_state=True`` the second element is the delay bed's ``u`` (a tensor),
    or, when the aquitard store is on, ``{"u": u or None, "ua": ua}``; ``u0`` accepts
    either form back.
    """
    # Anchor the device on the PARAMETERS, not on h0. Every forcing tensor reaching this
    # function is built by a numpy-backed loader (_idw_initial_heads, _ground_elev,
    # _load_pumping_kwh) and therefore arrives on CPU; fit_flow happens to move them
    # internally, but the k-fold *evaluation* path passes the originals straight through.
    # Taking the device from h0 then silently mixed a CPU rollout with CUDA model buffers
    # and blew up 9.5 h into a gate run. The model's parameters are the authority on where
    # this computation lives, so everything is moved to them here, once.
    dev = log_T.device
    h0 = h0.to(dtype=torch.float64, device=dev)
    A = h0.shape[-1]

    def _here(x):
        return None if x is None else x.to(dtype=torch.float64, device=dev)

    recharge = _here(recharge)
    pumping = _here(pumping)
    recharge_field = _here(recharge_field)
    E = _here(E)
    ground_elev = _here(ground_elev)
    # These two are learnable during a fit (already on `dev`, so `.to` is a no-op and the
    # autograd graph is untouched) but are rebuilt as plain CPU scalars from fit["theta"]
    # during evaluation, where they would otherwise meet CUDA tensors inside
    # pumping.energy_to_volume.
    recharge_scale = _here(recharge_scale)
    log_eta = _here(log_eta)
    log_head_extra = _here(log_head_extra)
    log_C_coast = _here(log_C_coast)
    log_C_apex = _here(log_C_apex)
    # Where the pumping stress lands (2026-09-14, three opt-in physics candidates):
    #  * ``pump_split_logit``: a fraction sigmoid(.) of every cell's abstraction is taken
    #    from layer 0 (the shallow aquifer, large storage) instead of ``pump_layer``.
    #    Shallow farm wells do pump layer 1 on this fan; the census carries no depth, so
    #    the fraction is learned.
    #  * ``return_frac_logit``: irrigation return flow -- a fraction 0.7*sigmoid(.) of the
    #    pumped volume infiltrates back into layer 0 the same month (paddy fields lose a
    #    large share of applied water to the shallow aquifer). Net stress falls without
    #    touching the electricity->volume conversion.
    #  * the leakance floor is a BOUNDS change (``--l-min``), not a rollout term.
    pump_split_logit = _here(pump_split_logit)
    return_frac_logit = _here(return_frac_logit)
    if E is not None and spread_W is not None:
        # spatial spread of the stress (spread.py): the same energy, applied over a radius
        E = spread_energy(E, _here(spread_W))
    T = torch.exp(log_T)
    S = torch.exp(log_S)
    L = torch.exp(log_L) if model.n_layers > 1 else None
    # Open boundaries (2026-09-11): only when the model was built with them AND the
    # caller passed conductances; either missing means the closed basin, exactly as
    # before, so recorded results replay.
    use_bnd = model.has_boundaries and log_C_coast is not None and log_C_apex is not None
    bdiag, brhs = model.boundary_terms(torch.exp(log_C_coast) if use_bnd else None,
                                       torch.exp(log_C_apex) if use_bnd else None)
    use_delay = delay_Sd is not None and delay_tau is not None
    use_riv = model.has_rivers and log_C_riv is not None
    use_sw = sw_field is not None and log_sw_scale is not None
    use_aqt = aqt_Sa is not None and aqt_G is not None and model.n_layers > 1
    sw_field = _here(sw_field)
    log_sw_scale = _here(log_sw_scale)
    swl = recharge_layer if sw_layer is None else int(sw_layer)
    season = use_riv and model.river_season is not None
    month0 = 1 if river_month0 is None else int(river_month0)
    ua0 = None
    if isinstance(u0, dict):
        u0, ua0 = u0.get("u"), u0.get("ua")
    extended = use_delay or use_riv or use_aqt
    if not extended:
        params = model.operator_params(log_T, log_S, log_L if model.n_layers > 1 else None,
                                       log_C_coast if use_bnd else None,
                                       log_C_apex if use_bnd else None)
        mv, diag = model._matvec_from(T, S, L, bdiag=bdiag if use_bnd else None)
        op = model._op
    else:
        from .flow import _COMPILE_MATVEC

        rebuilt = use_riv and (model.river_mode == "riv" or season)
        if rebuilt and _COMPILE_MATVEC:
            raise ValueError("--compile-matvec cannot be used with --rivers riv or a river "
                             "season table: the operator is rebuilt whenever the river "
                             "switch or month changes")
        delay_Sd, delay_tau = _here(delay_Sd), _here(delay_tau)
        log_C_riv = _here(log_C_riv)
        aqt_Sa, aqt_G = _here(aqt_Sa), _here(aqt_G)
        layout = model.operator_layout(bnd=use_bnd, delay=use_delay, riv=use_riv,
                                       aqt=use_aqt)
        params = model.operator_params(
            log_T, log_S, log_L if model.n_layers > 1 else None,
            log_C_coast if use_bnd else None, log_C_apex if use_bnd else None,
            delay=(delay_Sd, delay_tau) if use_delay else None,
            riv=(log_C_riv,) if use_riv else None,
            aqt=(aqt_Sa, aqt_G) if use_aqt else None)
        groups = {}
        if use_bnd:
            groups["bnd"] = (log_C_coast, log_C_apex)
        if use_delay:
            groups["delay"] = (delay_Sd, delay_tau)
            tau_d = torch.exp(delay_tau)
            beta_A = model.delay_beta(delay_Sd, delay_tau) * model.area
            u = h0 if u0 is None else _here(u0)
        aqt_mv = None
        if use_aqt:
            a_a, g_a, c_a, r_a = model.aqt_terms(aqt_Sa, aqt_G)
            D_a = a_a + 2.0 * g_a
            aqt_mv = (g_a, c_a)
            ua = 0.5 * (h0[:-1] + h0[1:]) if ua0 is None else _here(ua0)
        static_layout = tuple(g for g in layout if g not in ("riv", "aqt"))
        base_extra = model.extra_diag(static_layout, groups) if static_layout else None
        C_riv = torch.exp(log_C_riv) if use_riv else None
        riv_cache: dict = {}

        def _operator(mask, month):
            """(mv, diag, op, river rhs), rebuilt only when the RIV switch or (with a
            season table) the calendar month changes."""
            key = None if mask is None else mask
            fac = model.river_month_factor(month) if season else None
            mkey = (int(month) % 12) if season else None
            if riv_cache and riv_cache["month"] == mkey and (
                    (key is None and riv_cache["mask"] is None) or (
                    key is not None and riv_cache["mask"] is not None
                    and torch.equal(key, riv_cache["mask"]))):
                return riv_cache["ops"]
            extra = base_extra
            rrhs = None
            if use_riv:
                rdiag, rrhs = model.river_terms(C_riv, mask, fac)
                extra = rdiag if extra is None else extra + rdiag
            m_v, d_g = model._matvec_from(T, S, L, bdiag=extra, compile_ok=not rebuilt,
                                          aqt=aqt_mv)
            ops = (m_v, d_g, model.make_op(layout, riv_mask=mask, riv_factor=fac), rrhs)
            riv_cache["mask"], riv_cache["month"], riv_cache["ops"] = key, mkey, ops
            return ops
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
            # Static lift only; energy_to_volume adds the rest of the total dynamic head.
            # It is NOT clamped here any more -- the clamp belongs after head_extra is
            # added, or an artesian cell (head above ground, 20% of cell-months here) gets
            # floored to MIN_LIFT_M and implies a near-unbounded volume.
            lift = ground_elev - h[pump_layer]
            head_extra = (torch.exp(log_head_extra)
                          if log_head_extra is not None else None)
            if E.dim() == 3:
                # per-purpose efficiency classes: E is (C, A, T) and log_eta is (C,);
                # each class converts with its own eta and the volumes add
                vol = pumping_mod.energy_to_volume(E[:, :, t], lift,
                                                   log_eta.reshape(-1, 1),
                                                   head_extra=head_extra).sum(dim=0)
            else:
                vol = pumping_mod.energy_to_volume(E[:, t], lift, log_eta,
                                                   head_extra=head_extra)  # m3 for the month
            rate = vol / model.dt                                        # m3/day
            if pump_split_logit is not None and pump_layer != 0:
                f_shallow = torch.sigmoid(pump_split_logit)
                layer_q[0] = layer_q[0] - f_shallow * rate
                layer_q[pump_layer] = layer_q[pump_layer] - (1.0 - f_shallow) * rate
            else:
                layer_q[pump_layer] = layer_q[pump_layer] - rate
            if return_frac_logit is not None:
                r_ret = RETURN_FRAC_MAX * torch.sigmoid(return_frac_logit)
                layer_q[0] = layer_q[0] + r_ret * rate
        if use_sw:
            if sw_field.dim() == 3:
                # several components (canal leakage, paddy percolation), one fraction each
                sw_t = (torch.exp(log_sw_scale).reshape(-1, 1) * sw_field[:, :, t]).sum(dim=0)
            else:
                sw_t = torch.exp(log_sw_scale) * sw_field[:, t]
            layer_q[swl] = layer_q[swl] + sw_t * model.area
        q = torch.stack(layer_q, dim=0)
        b = S * model.area / model.dt * h + q + brhs
        if extended:
            mv, diag, op, rrhs = _operator(model.river_mask(h) if use_riv else None,
                                           month0 + t)
            if rrhs is not None:
                b = b + rrhs
            if use_delay:
                b = b + beta_A * u
            if use_aqt:
                ra_u = r_a * ua
                b = b + torch.cat([ra_u, torch.zeros_like(ra_u[:1])], dim=0) \
                    + torch.cat([torch.zeros_like(ra_u[:1]), ra_u], dim=0)
        solve = _warm_started_solver(mv, diag, h)
        h = _ImplicitSolve.apply(b, op, solve, *params)
        if extended and use_delay:
            u = (tau_d * u + model.dt * h) / (tau_d + model.dt)
        if extended and use_aqt:
            ua = (a_a * ua + g_a * (h[:-1] + h[1:])) / D_a
        out.append(h)
    heads = torch.stack(out, dim=-1)
    if return_state:
        u_end = u if extended and use_delay else None
        if extended and use_aqt:
            return heads, {"u": u_end, "ua": ua}
        return heads, u_end
    return heads


DELAY_MODES = ("off", "global", "zonal")
DELAY_U0_MODES = ("eq", "learned")
# --sw-components: which fields of surface_water.py's npz enter the rollout, in order
SW_COMPONENT_KEYS = {"delivered": ("sw_m_per_day",), "percolation": ("perc_m_per_day",),
                     "both": ("sw_m_per_day", "perc_m_per_day")}
SW_COMPONENT_CHOICES = tuple(SW_COMPONENT_KEYS)


def _add_extension_params(theta: dict, model: FlowModel, delay_storage: str = "off",
                          n_riv: int = 0, use_sw: bool | int = False,
                          zonal: bool = False, delay_u0: str = "eq",
                          aquitard: str = "off",
                          zones: tuple[str, ...] = ZONE_NAMES) -> dict:
    """The opt-in parameters of 2026-09-23, appended after the historical ones so a run
    without them builds exactly the historical dict (same keys, same order):

    - ``delay_storage="global"``: ``log_Sd``/``log_tau``, shape ``(1, 1)``;
      ``"zonal"``: ``log_Sd_{zone}``/``log_tau_{zone}`` per fan zone (value shared across
      layers within a zone). The ``_zone`` suffix lets ``_base_param_name`` and
      ``_zonal_bounds_hit`` treat them like every other zonal parameter.
    - ``n_riv > 0``: ``log_C_riv``, ``(n_riv,)``, one conductance per river group.
    - ``use_sw``: ``log_sw_scale``, the recharge fraction of canal deliveries (a scalar),
      or with ``use_sw = K > 1`` components one fraction each, ``(K,)``.
    - ``delay_u0="learned"`` (needs a delay bed): ``log_du0[_zone]``, the slow store's
      initial head above ``h0``.
    - ``aquitard="global"/"zonal"``: ``log_Sa[_zone]``/``log_G[_zone]`` of the aquitard
      store (``FlowModel.aqt_terms``), shared by every interface of a zone.

    ``zones`` is the zonation's names (``ZONE_NAMES_SPLIT`` with the proximal split).
    """
    if delay_storage not in DELAY_MODES:
        raise ValueError(f"delay_storage must be one of {DELAY_MODES}, got {delay_storage!r}")
    if aquitard not in DELAY_MODES:
        raise ValueError(f"aquitard must be one of {DELAY_MODES}, got {aquitard!r}")
    if delay_u0 not in DELAY_U0_MODES:
        raise ValueError(f"delay_u0 must be one of {DELAY_U0_MODES}, got {delay_u0!r}")
    if delay_u0 == "learned" and delay_storage == "off":
        raise ValueError("--delay-u0 learned needs --delay-storage global or zonal")
    dev = model.log_T.device

    def _p(value, shape=(1, 1)):
        return nn.Parameter(torch.full(shape, float(value), dtype=torch.float64, device=dev))

    if delay_storage == "global":
        theta["log_Sd"] = _p(math.log(DELAY_SD_INIT))
        theta["log_tau"] = _p(math.log(DELAY_TAU_INIT_DAYS))
    elif delay_storage == "zonal":
        if not zonal:
            raise ValueError("--delay-storage zonal needs --param-mode zonal; use 'global'")
        for z in zones:
            theta[f"log_Sd_{z}"] = _p(math.log(DELAY_SD_INIT))
        for z in zones:
            theta[f"log_tau_{z}"] = _p(math.log(DELAY_TAU_INIT_DAYS))
    if n_riv > 0:
        theta["log_C_riv"] = _p(math.log(C_RIV_INIT), shape=(int(n_riv),))
    n_sw = int(use_sw)
    if n_sw == 1:
        theta["log_sw_scale"] = nn.Parameter(
            torch.tensor(math.log(SW_SCALE_INIT), dtype=torch.float64, device=dev))
    elif n_sw > 1:
        init = list(SW_SCALE_INIT_2) + [SW_SCALE_INIT] * max(n_sw - 2, 0)
        theta["log_sw_scale"] = nn.Parameter(torch.tensor(
            [math.log(v) for v in init[:n_sw]], dtype=torch.float64, device=dev))
    # appended after every historical key, so a run without them is the historical dict
    if delay_u0 == "learned":
        if delay_storage == "global":
            theta["log_du0"] = _p(math.log(DELAY_DU0_INIT_M))
        else:
            for z in zones:
                theta[f"log_du0_{z}"] = _p(math.log(DELAY_DU0_INIT_M))
    if aquitard == "global":
        theta["log_Sa"] = _p(math.log(AQT_SA_INIT))
        theta["log_G"] = _p(math.log(AQT_G_INIT))
    elif aquitard == "zonal":
        if not zonal:
            raise ValueError("--aquitard-storage zonal needs --param-mode zonal; use 'global'")
        for z in zones:
            theta[f"log_Sa_{z}"] = _p(math.log(AQT_SA_INIT))
        for z in zones:
            theta[f"log_G_{z}"] = _p(math.log(AQT_G_INIT))
    return theta


def _zone_gather(cols: torch.Tensor, zone_t: torch.Tensor) -> torch.Tensor:
    """``(k, N_ZONES)`` per-zone columns -> ``(k, A)`` per-cell values.

    ``zone_t`` is either a long ``(A,)`` zone id (the sharp zonation: a gather) or a float
    ``(N_ZONES, A)`` weight matrix from ``zones.zone_blend_weights`` (``--zone-blend-km``:
    ``cols @ W``). Both are differentiable onto each zone's small tensor."""
    if zone_t.is_floating_point():
        return cols @ zone_t.to(dtype=cols.dtype, device=cols.device)
    return cols[:, zone_t]


def zone_tensor(zone_of_cell: np.ndarray | None, device=None,
                zone_w: np.ndarray | None = None) -> torch.Tensor | None:
    """The ``zone_t`` the ``_expand_*`` helpers take: the ``(N_ZONES, A)`` blend weights
    when given, else the long zone ids, else ``None``."""
    if zone_w is not None:
        return torch.as_tensor(np.asarray(zone_w, dtype="float64"), dtype=torch.float64,
                               device=device)
    if zone_of_cell is None:
        return None
    return torch.as_tensor(np.asarray(zone_of_cell), dtype=torch.long, device=device)


def _expand_zonal_group(theta: dict, zone_t: torch.Tensor | None, n_rows: int,
                        n_active: int, bases: tuple[str, ...]) -> list | None:
    """Global (``base``) or zonal (``base_{zone}``) scalar parameters -> one ``(n_rows,
    A)`` field per base, or ``None`` when ``theta`` has none of them. Zonal values gather
    by ``zone_t`` exactly like ``_expand_zonal`` (advanced-index backward is a scatter-add
    onto each zone)."""
    if bases[0] in theta:
        return [theta[b].reshape(1, 1).expand(n_rows, n_active) for b in bases]
    if f"{bases[0]}_{ZONE_NAMES[0]}" in theta:
        if zone_t is None:
            raise ValueError(f"zonal {bases} parameters need a zone assignment")
        out = []
        for base in bases:
            names = ZONE_NAMES_SPLIT if f"{base}_proximal_w" in theta else ZONE_NAMES
            cols = torch.cat([theta[f"{base}_{z}"].reshape(1, 1) for z in names], dim=1)
            out.append(_zone_gather(cols, zone_t).expand(n_rows, -1))
        return out
    return None


def _expand_zonal_delay(theta: dict, zone_t: torch.Tensor | None, n_layers: int,
                        n_active: int, layers: tuple[int, ...] | None = None
                        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Delay-bed parameters -> ``(log_Sd, log_tau)`` each ``(L, A)``, or ``(None, None)``
    when the run has no delay bed.

    ``layers`` (``--delay-layers``) restricts the bed to those layers: elsewhere
    ``log_Sd`` is the CONSTANT lower bound (not zero, which would drop the term's
    gradient path out of the operator and trip ``_ImplicitSolve``'s check)."""
    got = _expand_zonal_group(theta, zone_t, n_layers, n_active, ("log_Sd", "log_tau"))
    if got is None:
        return None, None
    Sd, tau = got
    if layers is not None:
        keep = torch.zeros(n_layers, 1, dtype=torch.bool, device=Sd.device)
        keep[list(layers)] = True
        floor = torch.full_like(Sd, BOUNDS["log_Sd"][0])
        Sd = torch.where(keep, Sd, floor)
    return Sd, tau


def _expand_delay_du0(theta: dict, zone_t: torch.Tensor | None, n_layers: int,
                      n_active: int) -> torch.Tensor | None:
    """``--delay-u0 learned``: the slow store's initial excess head ``exp(log_du0)`` as
    an ``(L, A)`` field in metres, or ``None`` (equilibrium, ``u0 = h0``)."""
    got = _expand_zonal_group(theta, zone_t, n_layers, n_active, ("log_du0",))
    return None if got is None else torch.exp(got[0])


def _expand_aqt(theta: dict, zone_t: torch.Tensor | None, n_layers: int,
                n_active: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``--aquitard-storage``: ``(log_Sa, log_G)`` each ``(L-1, A)``, or ``(None, None)``."""
    if n_layers < 2:
        return None, None
    got = _expand_zonal_group(theta, zone_t, n_layers - 1, n_active, ("log_Sa", "log_G"))
    return (None, None) if got is None else (got[0], got[1])


def _make_homogeneous_params(model: FlowModel, use_pumping: bool = False,
                             use_recharge: bool = False, n_eta: int = 1,
                             pump_split: bool = False, return_flow: bool = False,
                             learn_spread: bool = False, delay_storage: str = "off",
                             n_riv: int = 0, use_sw: bool | int = False,
                             delay_u0: str = "eq",
                             aquitard: str = "off") -> dict[str, nn.Parameter]:
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
            torch.full((n_eta,), float(np.log(0.3)), dtype=torch.float64, device=dev)
            if n_eta > 1 else
            torch.tensor(float(np.log(0.3)), dtype=torch.float64, device=dev))
        theta["log_head_extra"] = nn.Parameter(
            torch.tensor(float(np.log(40.0)), dtype=torch.float64, device=dev))
    if model.has_boundaries:
        theta["log_C_coast"] = nn.Parameter(model.log_C_coast.detach().clone())
        theta["log_C_apex"] = nn.Parameter(model.log_C_apex.detach().clone())
    if use_pumping and pump_split:
        theta["pump_split_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))      # start at 50/50
    if use_pumping and return_flow:
        theta["return_frac_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))      # start at 0.35
    if use_pumping and learn_spread:
        theta["log_spread_km"] = nn.Parameter(
            torch.tensor(float(np.log(2.0)), dtype=torch.float64, device=dev))
    if use_recharge:
        theta["recharge_frac_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))
    return _add_extension_params(theta, model, delay_storage, n_riv, use_sw, zonal=False,
                                 delay_u0=delay_u0, aquitard=aquitard)


def _base_param_name(name: str) -> str:
    """``"log_T_mid"`` -> ``"log_T"``. Zonal theta keys carry a zone suffix; BOUNDS and
    _clamp_ are keyed on the bare physical name. Only the three known zone suffixes are
    stripped, so ``log_eta`` and ``recharge_frac_logit`` survive untouched.
    """
    for zone in ZONE_NAMES_SPLIT:
        suffix = f"_{zone}"
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _theta_zone_names(theta: dict) -> tuple[str, ...]:
    """The zonation a zonal theta was built for: three zones, or four when it carries the
    proximal split's ``log_T_proximal_w``."""
    return ZONE_NAMES_SPLIT if "log_T_proximal_w" in theta else ZONE_NAMES


def _make_zonal_params(model: FlowModel, use_pumping: bool = False,
                       use_recharge: bool = False, n_eta: int = 1,
                       pump_split: bool = False, return_flow: bool = False,
                       learn_spread: bool = False, delay_storage: str = "off",
                       n_riv: int = 0, use_sw: bool | int = False, delay_u0: str = "eq",
                       aquitard: str = "off", split: bool = False,
                       proximal_layered: bool = False) -> dict[str, nn.Parameter]:
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

    ``split`` (opt-in, ``--zone-boundaries P,D,S``) adds a fourth zone, ``proximal_w``, the
    proximal cells west of S. It has the proximal form: one merged aquifer, 2 parameters.

    ``proximal_layered`` (opt-in, ``--proximal-layered``, 2026-09-25) gives each proximal
    zone the mid/distal form instead: ``log_T``/``log_S`` of shape ``(L, 1)`` and a
    learnable ``log_L_proximal[_w]`` ``(L-1, 1)``. They start at the merged values (the
    layer mean repeated, leakance at the BOUNDS ceiling the merged form pins), so epoch 0
    is the merged model; the keys and their order are otherwise unchanged, with the new
    leakances appended after ``log_L_distal`` and after the split's own entries.
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
    if split:
        theta["log_T_proximal_w"] = nn.Parameter(log_T0.mean(dim=0, keepdim=True))
        theta["log_S_proximal_w"] = nn.Parameter(log_S0.mean(dim=0, keepdim=True))
    if proximal_layered:
        L = model.n_layers
        for z in ("proximal", "proximal_w") if split else ("proximal",):
            for base in ("log_T", "log_S"):
                merged = theta[f"{base}_{z}"].detach()
                theta[f"{base}_{z}"] = nn.Parameter(merged.expand(L, 1).clone())
            if L > 1:
                theta[f"log_L_{z}"] = nn.Parameter(torch.full(
                    (L - 1, 1), BOUNDS["log_L"][1], dtype=torch.float64,
                    device=model.log_T.device))
    dev = model.log_T.device
    if use_pumping:
        theta["log_eta"] = nn.Parameter(
            torch.full((n_eta,), float(np.log(0.3)), dtype=torch.float64, device=dev)
            if n_eta > 1 else
            torch.tensor(float(np.log(0.3)), dtype=torch.float64, device=dev))
        theta["log_head_extra"] = nn.Parameter(
            torch.tensor(float(np.log(40.0)), dtype=torch.float64, device=dev))
    if model.has_boundaries:
        theta["log_C_coast"] = nn.Parameter(model.log_C_coast.detach().clone())
        theta["log_C_apex"] = nn.Parameter(model.log_C_apex.detach().clone())
    if use_pumping and pump_split:
        theta["pump_split_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))      # start at 50/50
    if use_pumping and return_flow:
        theta["return_frac_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))      # start at 0.35
    if use_pumping and learn_spread:
        theta["log_spread_km"] = nn.Parameter(
            torch.tensor(float(np.log(2.0)), dtype=torch.float64, device=dev))
    if use_recharge:
        theta["recharge_frac_logit"] = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float64, device=dev))
    return _add_extension_params(theta, model, delay_storage, n_riv, use_sw, zonal=True,
                                 delay_u0=delay_u0, aquitard=aquitard,
                                 zones=ZONE_NAMES_SPLIT if split else ZONE_NAMES)


def _expand_zonal(theta: dict[str, torch.Tensor], zone_t: torch.Tensor,
                  n_layers: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Column-stack the per-zone tensors into ``(k, N_ZONES)`` and gather to ``(k, A)``.

    The gather ``cols[:, zone_t]`` is advanced indexing, whose backward is a scatter-add:
    every cell's gradient accumulates onto its own zone's small tensor, exactly the way
    homogeneous mode relies on expand-backward summing over the broadcast axis. That is
    what lets ``FlowModel``'s frozen constructor and parameter shapes stay untouched.

    Proximal ``log_L`` is a CONSTANT at the upper bound, not a parameter: the four
    proximal layers equilibrate instead of being independently fitted -- unless the theta
    is ``--proximal-layered``: then ``log_T_proximal[_w]``/``log_S_proximal[_w]`` are
    ``(L, 1)`` (``expand`` is a no-op) and ``log_L_proximal[_w]`` replaces the constant.
    """
    dev = theta["log_T_mid"].device
    prox_T = theta["log_T_proximal"].expand(n_layers, 1)
    prox_S = theta["log_S_proximal"].expand(n_layers, 1)
    split = "log_T_proximal_w" in theta
    extra_T = [theta["log_T_proximal_w"].expand(n_layers, 1)] if split else []
    extra_S = [theta["log_S_proximal_w"].expand(n_layers, 1)] if split else []
    cols_T = torch.cat([prox_T, theta["log_T_mid"], theta["log_T_distal"], *extra_T], dim=1)
    cols_S = torch.cat([prox_S, theta["log_S_mid"], theta["log_S_distal"], *extra_S], dim=1)
    assert cols_T.shape == (n_layers, N_ZONES + int(split))
    log_T = _zone_gather(cols_T, zone_t)
    log_S = _zone_gather(cols_S, zone_t)
    log_L = None
    if n_layers > 1 and "log_L_mid" in theta:
        prox_L = torch.full((n_layers - 1, 1), BOUNDS["log_L"][1],
                            dtype=torch.float64, device=dev)
        prox_Lw = theta.get("log_L_proximal_w", prox_L)
        prox_L = theta.get("log_L_proximal", prox_L)
        cols_L = torch.cat([prox_L, theta["log_L_mid"], theta["log_L_distal"]]
                           + ([prox_Lw] if split else []), dim=1)
        log_L = _zone_gather(cols_L, zone_t)
    return log_T, log_S, log_L


def _zonal_bounds_hit(theta: dict[str, torch.Tensor]) -> dict[str, dict[str, dict[str, int]]]:
    """Clamp every zonal parameter into BOUNDS in place and report hits **per zone**,
    each as the ``{"lo", "hi", "n"}`` shape ``_clamp_`` returns.

    Spec §6's primary decision rule reads this. Pooling would let an interior mid value
    mask a pinned proximal one, and a model still fitting outside the measured physical
    range gives confident wrong counterfactuals -- the failure that matters at the
    operational bar. Ruling P1 further requires lo/hi to stay distinguished per zone:
    a pinned proximal log_T is only the documented failure if it is pinned LOW.
    """
    names = _theta_zone_names(theta)
    report: dict[str, dict[str, dict[str, int]]] = {name: {} for name in names}
    report["global"] = {}
    for name, par in theta.items():
        base = _base_param_name(name)
        if base not in BOUNDS:
            continue                     # recharge_frac_logit: unconstrained by design
        bucket = next((z for z in names if name.endswith(f"_{z}")), "global")
        lo = ZONE_LOWER_BOUNDS.get((base, bucket))
        if lo is not None:               # --log-t-min-proximal: a raised per-zone floor
            with torch.no_grad():
                par.clamp_(min=lo)
        hit = _clamp_({base: par})[base]
        if lo is not None:
            hit["lo"] = int((par <= lo + 1e-9).sum())
        report[bucket][base] = hit
    return report


def _first(v) -> float:
    return float(np.ravel(np.asarray(v, dtype="float64"))[0])


def _extension_readouts(theta_out: dict) -> dict:
    """Physical-unit copies of the opt-in parameters for the theta file: ``Sd[_zone]``,
    ``tau_days[_zone]``, ``C_riv_m2day`` and ``sw_scale``."""
    out = {}
    for suffix in [""] + [f"_{z}" for z in ZONE_NAMES_SPLIT]:
        if f"log_Sd{suffix}" in theta_out:
            out[f"Sd{suffix}"] = math.exp(_first(theta_out[f"log_Sd{suffix}"]))
            out[f"tau_days{suffix}"] = math.exp(_first(theta_out[f"log_tau{suffix}"]))
        if f"log_du0{suffix}" in theta_out:
            out[f"du0_m{suffix}"] = math.exp(_first(theta_out[f"log_du0{suffix}"]))
        if f"log_Sa{suffix}" in theta_out:
            out[f"Sa{suffix}"] = math.exp(_first(theta_out[f"log_Sa{suffix}"]))
            out[f"G_per_day{suffix}"] = math.exp(_first(theta_out[f"log_G{suffix}"]))
    if "log_C_riv" in theta_out:
        out["C_riv_m2day"] = [float(np.exp(v)) for v in np.ravel(theta_out["log_C_riv"])]
    if "log_sw_scale" in theta_out:
        v = np.ravel(theta_out["log_sw_scale"])
        out["sw_scale"] = (math.exp(float(v[0])) if np.ndim(theta_out["log_sw_scale"]) == 0
                           else [float(np.exp(x)) for x in v])
    return out


def _delay_diagnostics(theta_out: dict, dt: float, n_layers: int,
                       layers: tuple[int, ...] | None = None) -> dict:
    """Per zone (or global): ``S_eff = S + S_d dt/(tau + dt)`` per layer, the storage a
    monthly step actually sees (the bed's instantaneous share adds to S), and
    ``delay_is_elastic`` when ``tau < 3 dt``: the bed is then extra elastic storage, not
    a slow release, and the fit has used it to get round the S ceiling."""
    out: dict = {}
    zoned = [f"_{z}" for z in ZONE_NAMES_SPLIT if f"log_S_{z}" in theta_out]
    for suffix in (zoned or [""]):
        dsuf = suffix if f"log_Sd{suffix}" in theta_out else ""
        if f"log_Sd{dsuf}" not in theta_out or f"log_S{suffix}" not in theta_out:
            continue
        Sd = math.exp(_first(theta_out[f"log_Sd{dsuf}"]))
        tau = math.exp(_first(theta_out[f"log_tau{dsuf}"]))
        s = [float(np.exp(v)) for v in np.ravel(theta_out[f"log_S{suffix}"])]
        if len(s) == 1:
            s = s * n_layers                       # proximal: one merged aquifer
        inst = Sd * dt / (tau + dt)
        out[f"S_eff{suffix}"] = [s[k] + (inst if layers is None or k in layers else 0.0)
                                 for k in range(n_layers)]
        out[f"delay_is_elastic{suffix}"] = bool(tau < 3.0 * dt)
    return out


def _scalar_theta(theta: dict, bases: tuple[str, ...], device=None) -> dict:
    th = {}
    for k, v in theta.items():
        if _base_param_name(k) in bases and k.startswith("log_"):
            th[k] = torch.tensor(_first(v), dtype=torch.float64, device=device).reshape(1, 1)
    return th


def delay_fields_from_theta(theta: dict, zone_of_cell: np.ndarray | None, n_layers: int,
                            n_active: int, device=None,
                            layers: tuple[int, ...] | None = None,
                            zone_w: np.ndarray | None = None
                            ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """A theta file's delay-bed parameters -> ``(log_Sd, log_tau)`` each ``(L, A)`` on
    ``device``, or ``(None, None)``. The forward twin and the evaluation paths use it.
    ``layers`` is the run's ``--delay-layers`` (meta ``delay_layers``)."""
    th = _scalar_theta(theta, ("log_Sd", "log_tau"), device)
    if not th:
        return None, None
    zt = zone_tensor(zone_of_cell, device, zone_w)
    return _expand_zonal_delay(th, zt, n_layers, n_active, layers=layers)


def delay_du0_from_theta(theta: dict, zone_of_cell: np.ndarray | None, n_layers: int,
                         n_active: int, device=None,
                         zone_w: np.ndarray | None = None) -> torch.Tensor | None:
    """``--delay-u0 learned``: the slow store's initial excess head (m), ``(L, A)``."""
    th = _scalar_theta(theta, ("log_du0",), device)
    if not th:
        return None
    zt = zone_tensor(zone_of_cell, device, zone_w)
    return _expand_delay_du0(th, zt, n_layers, n_active)


def aqt_fields_from_theta(theta: dict, zone_of_cell: np.ndarray | None, n_layers: int,
                          n_active: int, device=None, zone_w: np.ndarray | None = None
                          ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """``--aquitard-storage``: ``(log_Sa, log_G)`` each ``(L-1, A)``, or ``(None, None)``."""
    th = _scalar_theta(theta, ("log_Sa", "log_G"), device)
    if not th:
        return None, None
    zt = zone_tensor(zone_of_cell, device, zone_w)
    return _expand_aqt(th, zt, n_layers, n_active)


def fit_flow(model: FlowModel, obs_h: torch.Tensor, obs_idx: torch.Tensor,
             obs_layer: torch.Tensor, recharge: torch.Tensor,
             E=None, ground_elev=None, epochs: int = 1500, lr: float = 0.1,
             init_scatter: float = 0.0, seed: int | None = None,
             param_mode: str = "homogeneous", h0: torch.Tensor | None = None,
             zone_of_cell: np.ndarray | None = None,
             recharge_field: torch.Tensor | None = None,
             pump_layer: int = 1, recharge_layer: int = 0,
             log_every: int = 0, fix_eta: float | None = None,
             fix_head_extra: float | None = None, pump_split: bool = False,
             return_flow: bool = False, spread_km: float | None = None,
             learn_spread: bool = False, loss_mode: str = "level",
             level_weight: float = 0.1, delay_storage: str = "off",
             sw_field: torch.Tensor | None = None, sw_layer: int | None = None,
             fix_sw_scale: float | None = None, delay_u0: str = "eq",
             delay_layers: tuple[int, ...] | None = None,
             aquitard: str = "off", zone_w: np.ndarray | None = None,
             proximal_layered: bool = False, well_datum: str = "off",
             well_datum_sd: float = WELL_DATUM_SD_DEFAULT,
             datum_mask: torch.Tensor | None = None) -> dict:
    """Fit log-parameters to observed head series by masked MSE.

    ``well_datum="fit"`` (``--well-datum fit``, opt-in 2026-09-26; homogeneous/zonal):
    each well of ``obs_h`` carries a datum ``d_i`` in the observation operator,
    ``pred_i(t) = h(cell_i, layer_i, t) + d_i``, with a Gaussian prior of sd
    ``well_datum_sd`` m, in both loss variants (``_profile_well_datum`` gives the exact
    MAP and the prior's scaling). The datum is an observation-operator parameter, not a
    physical one: it never enters the flow solve, is not in ``theta`` (so not in
    ``bounds_hit``, fold thetas or any downstream model) and comes back as
    ``well_datum`` (``(W,)``, in ``obs_h`` row order) with ``well_datum_kappa`` (the
    long-record prior weight ``m s^2 / sd^2``) and
    ``well_datum_n``. ``r2`` stays the physical heads' in-sample R2 (comparable to
    every earlier run); ``r2_with_datum`` adds the datum. ``datum_mask`` (``(W, T)``
    bool, default: the loss mask) restricts the months that inform the datum and count
    as its ``n_i`` -- ``main`` passes the months actually observed, so a back-filled
    month (a copy of a neighbouring, possibly held-out, value) never sets a datum and a
    well with no observed month gets ``d = 0``. Default off: bit-identical.

    ``proximal_layered`` (``--proximal-layered``, opt-in 2026-09-25; zonal only): the
    proximal zone(s) get per-layer ``log_T``/``log_S`` and learnable leakances instead of
    the merged aquifer (``_make_zonal_params``). Default off: the merged form.

    ``zone_w`` (``--zone-blend-km``, 2026-09-23) is an ``(N_ZONES, A)`` blend-weight
    matrix (``zones.zone_blend_weights``). Zonal parameters are then mixed per cell, not
    gathered. The parameter count is unchanged. Default ``None``: the sharp zonation.

    Opt-in (2026-09-23, second round): ``delay_u0="learned"`` fits the delay bed's initial
    disequilibrium (``u0 = h0 + exp(log_du0)``), ``delay_layers`` restricts the bed to
    those layers, ``aquitard`` ("global"/"zonal") adds a storage node on every layer
    interface (``FlowModel.aqt_terms``), and ``sw_field`` may be ``(K, A, T)`` with one
    learnable fraction per component. All default off.

    Opt-in (2026-09-23): ``delay_storage`` ("global"/"zonal") adds the lumped delay bed,
    a model built with ``set_rivers`` gets one learnable conductance per river group, and
    ``sw_field`` (A, T, m/day of canal deliveries) adds surface-water irrigation recharge
    with a learnable fraction (or ``fix_sw_scale``). All default off.

    ``fix_eta``/``fix_head_extra`` (2026-09-13) hold the pump energy->volume conversion at
    given physical values instead of learning it. Diagnostic: every free fit so far has
    driven both to their bounds, so this measures what the rest of the model can do when
    the abstraction is what the electricity says it is.

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
    if proximal_layered and param_mode != "zonal":
        raise ValueError("proximal_layered needs param_mode='zonal'")
    if well_datum not in WELL_DATUM_MODES:
        raise ValueError(f"well_datum must be one of {WELL_DATUM_MODES}, got {well_datum!r}")
    use_datum = well_datum == "fit"
    if use_datum and param_mode == "percell":
        raise ValueError("well_datum='fit' is wired for param_mode homogeneous/zonal only")
    if use_datum and not float(well_datum_sd) > 0.0:
        raise ValueError(f"well_datum_sd must be > 0, got {well_datum_sd!r}")

    n_steps = recharge.shape[-1]
    A = model.grid.n_active
    dev = model.log_T.device
    if h0 is None:
        h0 = torch.zeros(model.n_layers, A, dtype=torch.float64, device=dev)
    else:
        h0 = h0.to(dtype=torch.float64, device=dev)
    obs_h = obs_h.to(dtype=torch.float64, device=dev)
    # NaN months in the target (only a --no-backfill run has any) are masked out of the
    # loss; a back-filled target has none, so its loss is bit-for-bit the unmasked one.
    obs_mask = torch.isfinite(obs_h)
    masked = not bool(obs_mask.all())
    obs_z = torch.where(obs_mask, obs_h, torch.zeros_like(obs_h)) if masked else obs_h
    d_mask = None
    if use_datum:
        d_mask = obs_mask if masked else None
        if datum_mask is not None:
            if tuple(datum_mask.shape) != tuple(obs_h.shape):
                raise ValueError(f"datum_mask {tuple(datum_mask.shape)} != obs_h "
                                 f"{tuple(obs_h.shape)}")
            d_mask = datum_mask.to(device=dev, dtype=torch.bool) & obs_mask
    obs_idx = obs_idx.to(device=dev)
    obs_layer = obs_layer.to(device=dev)
    recharge = recharge.to(dtype=torch.float64, device=dev)
    if recharge_field is not None:
        recharge_field = recharge_field.to(dtype=torch.float64, device=dev)
    if E is not None:
        E = E.to(dtype=torch.float64, device=dev)
    if ground_elev is not None:
        ground_elev = ground_elev.to(dtype=torch.float64, device=dev)
    if sw_field is not None:
        sw_field = sw_field.to(dtype=torch.float64, device=dev)
    # The apex boundary holds this run's initial head; a fold's h0 is built from its
    # kept wells only, so this cannot leak a held-out well into the boundary.
    model.set_apex_heads(h0)

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
            loss = (_masked_mse(pred, obs_z, obs_mask) if masked
                    else ((pred - obs_h) ** 2).mean())
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
    n_eta = int(E.shape[0]) if (use_pumping and E.dim() == 3) else 1
    n_riv = int(model.n_riv_groups) if model.has_rivers else 0
    use_sw = sw_field is not None
    n_sw = (int(sw_field.shape[0]) if use_sw and sw_field.dim() == 3 else int(use_sw))
    zone_t = None
    if param_mode == "zonal":
        zone_arr = np.asarray(zone_of_cell, dtype="int64").reshape(-1)
        if zone_arr.shape[0] != A:
            raise ValueError(
                f"zone_of_cell has {zone_arr.shape[0]} entries but the grid has {A} "
                "active cells"
            )
        zone_t = torch.tensor(zone_arr, dtype=torch.long, device=dev)
        # the opt-in proximal split: ids 0-3, one more merged-aquifer zone
        split = bool((zone_arr == PROXIMAL_W).any())
        n_z = N_ZONES + int(split)
        if zone_w is not None:
            zone_t = zone_tensor(None, dev, zone_w)
            if tuple(zone_t.shape) != (n_z, A):
                raise ValueError(f"zone_w must be ({n_z}, {A}), got {tuple(zone_t.shape)}")
        theta = _make_zonal_params(model, use_pumping=use_pumping,
                                   use_recharge=use_recharge, n_eta=n_eta,
                                   pump_split=pump_split, return_flow=return_flow,
                                   learn_spread=learn_spread, delay_storage=delay_storage,
                                   n_riv=n_riv, use_sw=n_sw, delay_u0=delay_u0,
                                   aquitard=aquitard, split=split,
                                   proximal_layered=proximal_layered)
    else:
        theta = _make_homogeneous_params(model, use_pumping=use_pumping,
                                         use_recharge=use_recharge, n_eta=n_eta,
                                         pump_split=pump_split, return_flow=return_flow,
                                         learn_spread=learn_spread,
                                         delay_storage=delay_storage, n_riv=n_riv,
                                         use_sw=n_sw, delay_u0=delay_u0, aquitard=aquitard)
    # spatial spread of the pumping stress: fixed radius, learned radius, or none
    d2_km = (pairwise_d2_km(model.grid, device=dev)
             if use_pumping and (learn_spread or spread_km is not None) else None)
    fixed_W = (spread_matrix(d2_km, torch.tensor(math.log(spread_km), dtype=torch.float64,
                                                  device=dev))
               if d2_km is not None and not learn_spread else None)
    if init_scatter > 0.0:
        g = torch.Generator().manual_seed(int(seed) if seed is not None else 0)
        with torch.no_grad():
            for name, par in theta.items():
                if name.endswith("_logit"):
                    continue   # unconstrained scalars; scatter would just re-centre them
                par.add_(torch.randn(par.shape, generator=g) * init_scatter)
    def _loss(pred: torch.Tensor) -> torch.Tensor:
        """``"level"``: plain MSE on heads (every gate before 2026-09-18). ``"anomaly"``:
        MSE on each well's departures from its own mean over the fitted months, plus
        ``level_weight`` x the MSE of the means. The level misfit is dominated by
        between-well differences of tens of metres, so a level fit is never asked to get
        a well's variations right -- and the held-out-years gate found exactly that."""
        if masked:
            if loss_mode == "level":
                return _masked_mse(pred, obs_z, obs_mask)
            mf = obs_mask.to(pred.dtype)
            n_w = mf.sum(dim=1, keepdim=True)
            pm = (pred * mf).sum(dim=1, keepdim=True) / n_w.clamp_min(1.0)
            om = (obs_z * mf).sum(dim=1, keepdim=True) / n_w.clamp_min(1.0)
            has = (n_w > 0).to(pred.dtype)       # a well never observed carries no level
            lvl = (((pm - om) ** 2) * has).sum() / has.sum().clamp_min(1.0)
            return (_masked_mse(pred - pm, obs_z - om, obs_mask)
                    + level_weight * lvl)
        if loss_mode == "level":
            return ((pred - obs_h) ** 2).mean()
        pm, om = pred.mean(dim=1, keepdim=True), obs_h.mean(dim=1, keepdim=True)
        return (((pred - pm) - (obs_h - om)) ** 2).mean() + level_weight * ((pm - om) ** 2).mean()

    def _loss_datum(pred: torch.Tensor) -> torch.Tensor:
        """``_loss`` with the per-well datum profiled out (``--well-datum fit``).

        ``"level"``: ``MSE(pred + d, obs) + sum kappa_i d_i^2 / N`` (``_profile_well_datum``;
        ``N`` = the loss's observed cells). ``"anomaly"``: the anomaly part is unchanged
        (a constant cancels in it); the level term becomes ``level_weight x
        mean_i[(pm_i + d_i - om_i)^2 + (kappa_i / n_i) d_i^2]`` -- the prior scaled by the
        weight that loss puts on well ``i``'s level, so the MAP datum (shrinkage
        ``n_i / (n_i + kappa_i)``) is the same under both losses. With a ``datum_mask``
        narrower than the loss mask the datum is the MAP of the masked months and the
        loss evaluates it (the physics still sees every loss month, as by default)."""
        d, kappa, _, n_d = _profile_well_datum(pred, obs_z, d_mask, well_datum_sd)
        if loss_mode == "level":
            if masked:
                mse = _masked_mse(pred + d, obs_z, obs_mask)
                n_cells = obs_mask.sum().to(pred.dtype).clamp_min(1.0)
            else:
                mse = ((pred + d - obs_h) ** 2).mean()
                n_cells = float(pred.numel())
            return mse + (kappa * d ** 2).sum() / n_cells
        if masked:
            mf = obs_mask.to(pred.dtype)
            n_w = mf.sum(dim=1, keepdim=True)
            pm = (pred * mf).sum(dim=1, keepdim=True) / n_w.clamp_min(1.0)
            om = (obs_z * mf).sum(dim=1, keepdim=True) / n_w.clamp_min(1.0)
            has = (n_w > 0).to(pred.dtype)
            anom = _masked_mse(pred - pm, obs_z - om, obs_mask)
        else:
            pm, om = pred.mean(dim=1, keepdim=True), obs_h.mean(dim=1, keepdim=True)
            has = torch.ones_like(pm)
            anom = (((pred - pm) - (obs_h - om)) ** 2).mean()
        per_well = ((pm + d - om) ** 2 + kappa / n_d.clamp_min(1.0) * d ** 2) * has
        return anom + level_weight * per_well.sum() / has.sum().clamp_min(1.0)

    fixed: dict[str, torch.Tensor] = {}
    if use_pumping and fix_eta is not None:
        fixed["log_eta"] = torch.full_like(theta.pop("log_eta").detach(),
                                           float(math.log(fix_eta)))
    if use_pumping and fix_head_extra is not None:
        fixed["log_head_extra"] = torch.full_like(theta.pop("log_head_extra").detach(),
                                                  float(math.log(fix_head_extra)))
    if use_sw and fix_sw_scale is not None:
        fixed["log_sw_scale"] = torch.full_like(theta.pop("log_sw_scale").detach(),
                                                float(math.log(fix_sw_scale)))
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
        d_Sd, d_tau = _expand_zonal_delay(theta, zone_t, model.n_layers, A,
                                          layers=delay_layers)
        ext = {}
        if d_Sd is not None:
            ext.update(delay_Sd=d_Sd, delay_tau=d_tau)
            du0 = _expand_delay_du0(theta, zone_t, model.n_layers, A)
            if du0 is not None:
                ext["u0"] = h0 + du0
        a_Sa, a_G = _expand_aqt(theta, zone_t, model.n_layers, A)
        if a_Sa is not None:
            ext.update(aqt_Sa=a_Sa, aqt_G=a_G)
        if n_riv:
            ext["log_C_riv"] = theta["log_C_riv"]
        if use_sw:
            ext.update(sw_field=sw_field, sw_layer=sw_layer,
                       log_sw_scale=theta.get("log_sw_scale", fixed.get("log_sw_scale")))
        return _rollout(
            model, log_T, log_S, log_L, h0, n_steps, **ext,
            recharge=None if use_recharge else recharge,
            pumping=None,
            recharge_field=recharge_field if use_recharge else None,
            recharge_scale=theta.get("recharge_frac_logit"),
            recharge_layer=recharge_layer,
            E=E if use_pumping else None,
            log_eta=theta.get("log_eta", fixed.get("log_eta")),
            log_head_extra=theta.get("log_head_extra", fixed.get("log_head_extra")),
            ground_elev=ground_elev,
            pump_layer=pump_layer,
            log_C_coast=theta.get("log_C_coast"),
            log_C_apex=theta.get("log_C_apex"),
            pump_split_logit=theta.get("pump_split_logit"),
            return_frac_logit=theta.get("return_frac_logit"),
            spread_W=(spread_matrix(d2_km, theta["log_spread_km"])
                      if "log_spread_km" in theta else fixed_W),
        )

    r2_trace: list[tuple[int, float]] = []
    for _ep in range(epochs):
        opt.zero_grad()
        h = _forward()
        pred = h[obs_layer, obs_idx, 1:]
        loss = _loss_datum(pred) if use_datum else _loss(pred)
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
        if model.has_boundaries and "log_C_coast" in theta:
            model.log_C_coast.copy_(theta["log_C_coast"])
            model.log_C_apex.copy_(theta["log_C_apex"])
        d_Sd, d_tau = _expand_zonal_delay(theta, zone_t, model.n_layers, A,
                                          layers=delay_layers)
        if d_Sd is not None:
            model.delay_log_Sd = d_Sd.detach().clone()
            model.delay_log_tau = d_tau.detach().clone()
            du0 = _expand_delay_du0(theta, zone_t, model.n_layers, A)
            model.delay_log_du0 = None if du0 is None else torch.log(du0).detach().clone()
        a_Sa, a_G = _expand_aqt(theta, zone_t, model.n_layers, A)
        if a_Sa is not None:
            model.aqt_log_Sa = a_Sa.detach().clone()
            model.aqt_log_G = a_G.detach().clone()
        if n_riv:
            model.fit_log_C_riv = theta["log_C_riv"].detach().clone()
    n_params = sum(p.numel() for p in free)
    theta_out = {}
    for k, v in list(theta.items()) + list(fixed.items()):
        theta_out[k] = (float(v.detach().cpu()) if v.dim() == 0
                        else v.detach().clone().squeeze(-1).cpu().numpy().tolist())
    if "log_eta" in theta_out:
        theta_out["eta"] = (float(np.exp(theta_out["log_eta"]))
                            if np.ndim(theta_out["log_eta"]) == 0
                            else [float(np.exp(v)) for v in theta_out["log_eta"]])
    if "log_head_extra" in theta_out:
        theta_out["head_extra_m"] = float(np.exp(theta_out["log_head_extra"]))
    if "log_C_coast" in theta_out:
        theta_out["C_coast_m2day"] = [float(np.exp(v)) for v in theta_out["log_C_coast"]]
        theta_out["C_apex_m2day"] = float(np.exp(theta_out["log_C_apex"][0]))
    if "pump_split_logit" in theta_out:
        theta_out["pump_frac_shallow"] = float(1.0 / (1.0 + np.exp(-theta_out["pump_split_logit"])))
    if "return_frac_logit" in theta_out:
        theta_out["return_frac"] = float(RETURN_FRAC_MAX
                                         / (1.0 + np.exp(-theta_out["return_frac_logit"])))
    if "log_spread_km" in theta_out:
        theta_out["spread_km"] = float(np.exp(theta_out["log_spread_km"]))
    elif spread_km is not None:
        theta_out["spread_km"] = float(spread_km)          # fixed, recorded for the forward twin
    if "recharge_frac_logit" in theta_out:
        theta_out["recharge_frac"] = float(1.0 / (1.0 + np.exp(-theta_out["recharge_frac_logit"])))
    theta_out.update(_extension_readouts(theta_out))
    diag = _delay_diagnostics(theta_out, model.dt, model.n_layers, delay_layers)
    theta_out.update(diag)
    elastic = [k[len("delay_is_elastic"):] or "_global" for k, v in diag.items()
               if k.startswith("delay_is_elastic") and v]
    if elastic:
        print(f"    WARNING delay_is_elastic: tau < 3 dt in {elastic} -- the delay bed is "
              "acting as extra instant storage, not a slow release (raise "
              "--delay-tau-min-days)", flush=True)
    out = {"loss": float(loss.detach()), "epochs": epochs, "bounds_hit": hits,
           "r2": _r2(pred.cpu().numpy(), obs_h.cpu().numpy()), "n_params": n_params,
           "param_mode": param_mode, "theta": theta_out, "r2_trace": r2_trace,
           "fixed": sorted(fixed), "loss_mode": loss_mode}
    if use_datum:
        # the MAP datum at the FINAL parameters (the loop's last one lags a step)
        d, kappa, s2, n = _profile_well_datum(pred, obs_z, d_mask, well_datum_sd)
        d_np = d.squeeze(-1).cpu().numpy()
        # well_datum_kappa: the long-record prior weight m s^2 / sd^2 (months); a well with
        # n_i observed months has kappa_i = min(n_i, m) / m x this (_profile_well_datum)
        out.update({"well_datum": d_np,
                    "well_datum_kappa": float(WELL_DATUM_MONTHS_PER_OBS * s2
                                              / float(well_datum_sd) ** 2),
                    "well_datum_sigma_m": float(torch.sqrt(s2)),
                    "well_datum_n": n.squeeze(-1).cpu().numpy(),
                    "well_datum_sd": float(well_datum_sd),
                    "r2_with_datum": _r2(pred.cpu().numpy() + d_np[:, None],
                                         obs_h.cpu().numpy())})
    return out


def sw_scale_tensor(v, device=None) -> torch.Tensor:
    """``log_sw_scale`` from a theta file: a scalar (one component) stays 0-d, a list
    (``--sw-components``) becomes ``(K,)``, matching ``_rollout``'s two forms."""
    if np.ndim(v) == 0:
        return torch.tensor(float(v), dtype=torch.float64, device=device)
    arr = np.ravel(np.asarray(v, dtype="float64"))
    return torch.tensor(arr, dtype=torch.float64, device=device)


def _predict_homogeneous(model: FlowModel, fit: dict, h0: torch.Tensor, n_steps: int,
                         recharge: torch.Tensor | None = None,
                         recharge_field: torch.Tensor | None = None,
                         E: torch.Tensor | None = None,
                         ground_elev: torch.Tensor | None = None,
                         recharge_layer: int = 0, pump_layer: int = 1,
                         sw_field: torch.Tensor | None = None,
                         sw_layer: int | None = None) -> torch.Tensor:
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
        log_head_extra = (torch.tensor(theta["log_head_extra"], dtype=torch.float64)
                          if E is not None and "log_head_extra" in theta else None)
        rfrac = (torch.tensor(theta["recharge_frac_logit"], dtype=torch.float64)
                 if recharge_field is not None and "recharge_frac_logit" in theta else None)
        split = (torch.tensor(theta["pump_split_logit"], dtype=torch.float64)
                 if "pump_split_logit" in theta else None)
        ret = (torch.tensor(theta["return_frac_logit"], dtype=torch.float64)
               if "return_frac_logit" in theta else None)
        W = None
        if E is not None and "spread_km" in theta:
            W = spread_matrix(pairwise_d2_km(model.grid, device=model.log_T.device),
                              torch.tensor(math.log(theta["spread_km"]), dtype=torch.float64,
                                           device=model.log_T.device))
        # opt-in extensions: the delay bed and river conductances a fit copied back onto
        # the model, and the canal-recharge fraction that travels in theta
        ext = {}
        if model.delay_log_Sd is not None:
            ext.update(delay_Sd=model.delay_log_Sd, delay_tau=model.delay_log_tau)
            if model.delay_log_du0 is not None:
                ext["u0"] = h0.to(model.delay_log_du0.device) + torch.exp(model.delay_log_du0)
        if model.aqt_log_Sa is not None:
            ext.update(aqt_Sa=model.aqt_log_Sa, aqt_G=model.aqt_log_G)
        if model.has_rivers and "log_C_riv" in theta:
            ext["log_C_riv"] = torch.tensor(np.ravel(theta["log_C_riv"]), dtype=torch.float64)
        if sw_field is not None and "log_sw_scale" in theta:
            ext.update(sw_field=sw_field, sw_layer=sw_layer,
                       log_sw_scale=sw_scale_tensor(theta["log_sw_scale"]))
        return _rollout(model, log_T, log_S, log_L, h0, n_steps, **ext,
                        recharge=None if recharge_field is not None else recharge,
                        recharge_field=recharge_field, recharge_scale=rfrac,
                        recharge_layer=recharge_layer,
                        E=E if log_eta is not None else None, log_eta=log_eta,
                        log_head_extra=log_head_extra,
                        ground_elev=ground_elev, pump_layer=pump_layer,
                        log_C_coast=model.log_C_coast, log_C_apex=model.log_C_apex,
                        pump_split_logit=split, return_frac_logit=ret, spread_W=W)


def _nanmean_rows(x: np.ndarray) -> np.ndarray:
    """``(W, 1)`` row means over the finite entries; NaN for a row with none (silently)."""
    x = np.asarray(x, dtype="float64")
    n = np.isfinite(x).sum(axis=1, keepdims=True)
    s = np.where(np.isfinite(x), x, 0.0).sum(axis=1, keepdims=True)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _last_finite(x: np.ndarray) -> np.ndarray:
    """``(W, 1)``: each row's last finite entry (NaN for a row with none)."""
    x = np.asarray(x, dtype="float64")
    fin = np.isfinite(x)
    last = x.shape[1] - 1 - np.argmax(fin[:, ::-1], axis=1)
    out = x[np.arange(x.shape[0]), last]
    return np.where(fin.any(axis=1), out, np.nan)[:, None]


def temporal_gate(model: FlowModel, fit: dict, h0: torch.Tensor, obs_full: torch.Tensor,
                  obs_idx: torch.Tensor, obs_layer: torch.Tensor, T_fit: int,
                  E_full, recharge_full, ground_elev, recharge_layer: int = 0,
                  pump_layer: int = 1, sw_full: torch.Tensor | None = None,
                  sw_layer: int | None = None, rmse_k: float = 1.5,
                  keep_arrays: bool = False, datum: np.ndarray | None = None) -> dict:
    """Score a free-running continuation over the months the fit never saw.

    ``datum`` (``(W,)`` m, ``--well-datum fit``): the per-well datum fitted on the fitted
    months only (no held-out value enters it). Every legacy metric, the ``verdict`` and
    ``arrays["pred"]`` stay the PHYSICAL heads, so they remain comparable with every
    other run -- ``policy_gate.rank_key`` ranks on ``verdict``/``rmse_ratio``, and a
    datum there would hand a datum run most of its held-out level error for free
    (review 2026-09-26). The same verdict WITH the datum added unchanged to every month
    comes back beside them as ``datum_*`` keys. ``drift_diag.fair_temporal_verdict``
    removes its own fitted-period datum, so the fair verdict is invariant to ours.

    Since 2026-09-23 it also returns a first-class verdict (``temporal_verdict``): PASS
    needs the continuation's shape R2 to beat climatology's AND its RMSE to stay under
    ``rmse_k`` x climatology's. ``keep_arrays`` adds ``arrays`` (pred, obs, clim) for
    the ``stage3_temporal_pred.npz`` dump.

    The k-fold gate holds out *wells* and asks whether the model interpolates in space
    better than IDW. This one holds out *time*: the model rolls from the record's start
    under the recorded forcing through the fitted months and on across the held-out
    months, and its held-out heads are scored against two baselines a forecaster would
    face -- each well's month-of-year climatology built from the fitted months, and
    persistence of its last fitted value. Beating climatology here means the model's
    response to the forcing carries information about what the heads did next.
    """
    T_full = obs_full.shape[1]
    with torch.no_grad():
        h = _predict_homogeneous(model, fit, h0, T_full, recharge_field=recharge_full,
                                 E=E_full, ground_elev=ground_elev,
                                 recharge_layer=recharge_layer, pump_layer=pump_layer,
                                 sw_field=sw_full, sw_layer=sw_layer)
        pred = h[obs_layer.to(h.device), obs_idx.to(h.device), 1:].cpu().numpy()
    obs = obs_full.cpu().numpy()
    held = slice(T_fit, T_full)
    # month-of-year climatology from the fitted months; obs columns are months 1..T-1 of
    # the record (month 0 is the initial condition), so calendar month = (t + 1) % 12
    months = (np.arange(T_full) + 1) % 12
    clim = np.zeros_like(obs)
    # nan-aware throughout: identical on a back-filled record, and a --no-backfill record
    # (NaN where never observed) gets NaN baselines only for wells unseen in the fit
    for mth in range(12):
        sel_fit = (months[:T_fit] == mth)
        clim[:, months == mth] = _nanmean_rows(obs[:, :T_fit][:, sel_fit])
    persist = np.zeros_like(obs)
    persist[:, held] = np.repeat(_last_finite(obs[:, :T_fit]), T_full - T_fit, axis=1)
    # Pooled R2 over wells is dominated by between-well level differences (tens of
    # metres), which makes climatology trivially strong. The number that tests the
    # response in time is R2 on per-well ANOMALIES from each well's fitted-period mean,
    # plus the median per-well R2, reported alongside the pooled one.
    mean_fit = _nanmean_rows(obs[:, :T_fit])

    def _anom(x):
        return _r2((x[:, held] - mean_fit).reshape(-1), (obs[:, held] - mean_fit).reshape(-1))

    def _median_per_well(x):
        return float(np.nanmedian([_r2(x[w, held], obs[w, held])
                                   for w in range(obs.shape[0])]))

    def _shape(x):
        """R2 of the held-out variation alone: each series minus ITS OWN fitted-period
        mean. ``_anom`` removes the observed mean from both, so it charges a model for a
        level offset as well as for a wrong shape; this one isolates the shape, which is
        what drives compaction (subsidence follows head CHANGE)."""
        return _r2((x[:, held] - x[:, :T_fit].mean(axis=1, keepdims=True)).reshape(-1),
                   (obs[:, held] - mean_fit).reshape(-1))

    def _level_err(x):
        return float(np.nanmean(np.abs(x[:, :T_fit].mean(axis=1) - mean_fit.ravel())))

    out = {"r2_model": _r2(pred[:, held], obs[:, held]),
           "r2_clim": _r2(clim[:, held], obs[:, held]),
           "r2_persist": _r2(persist[:, held], obs[:, held]),
           "r2_anom_model": _anom(pred), "r2_anom_clim": _anom(clim),
           "r2_anom_persist": _anom(persist),
           "r2_well_median_model": _median_per_well(pred),
           "r2_well_median_clim": _median_per_well(clim),
           "r2_well_median_persist": _median_per_well(persist),
           "r2_shape_model": _shape(pred), "r2_shape_clim": _shape(clim),
           "level_err_model_m": _level_err(pred),
           "n_months": int(T_full - T_fit)}
    out.update(temporal_verdict(pred, obs, clim, T_fit, k=rmse_k))
    out["verdict_anom_legacy"] = ("PASS" if out["r2_anom_model"] > out["r2_anom_clim"]
                                  else "FAIL")
    if datum is not None:
        pred_d = pred + np.asarray(datum, dtype="float64").reshape(-1, 1)
        out.update({f"datum_{k}": v for k, v in
                    temporal_verdict(pred_d, obs, clim, T_fit, k=rmse_k).items()
                    if k not in ("r2_shape_clim", "rmse_clim_m", "bias_clim_m", "rmse_k")})
        out["datum_r2_anom_model"] = _anom(pred_d)
    if keep_arrays:
        out["arrays"] = {"pred": pred, "obs": obs, "clim": clim, "T_fit": int(T_fit)}
    return out


def temporal_verdict(pred: np.ndarray, obs: np.ndarray, clim: np.ndarray, T_fit: int,
                     k: float = 1.5) -> dict:
    """The held-out-years verdict (G3), on ``(W, T)`` arrays whose first ``T_fit`` months
    were fitted. PASS iff the continuation's SHAPE R2 (each series minus its own
    fitted-period mean, against the observed departures) beats climatology's AND its
    RMSE over the held-out months is below ``k`` x climatology's. Shape alone lets a
    drifting level through; RMSE alone rewards a flat line; the pair asks for both.
    Strict inequalities: a model exactly at ``k`` x climatology fails."""
    pred, obs, clim = (np.asarray(a, dtype="float64") for a in (pred, obs, clim))
    held = slice(int(T_fit), obs.shape[1])
    mean_fit = _nanmean_rows(obs[:, :T_fit])

    def _shape(x):
        return _r2((x[:, held] - x[:, :T_fit].mean(axis=1, keepdims=True)).reshape(-1),
                   (obs[:, held] - mean_fit).reshape(-1))

    def _rmse(x):
        d = (x[:, held] - obs[:, held]).reshape(-1)
        d = d[np.isfinite(d)]
        return float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")

    def _bias(x):
        d = (x[:, held] - obs[:, held]).reshape(-1)
        d = d[np.isfinite(d)]
        return float(d.mean()) if d.size else float("nan")

    s_m, s_c = _shape(pred), _shape(clim)
    r_m, r_c = _rmse(pred), _rmse(clim)
    ratio = r_m / r_c if r_c > 0 else float("inf")
    ok = bool(s_m > s_c and r_m < float(k) * r_c)
    return {"r2_shape_model": s_m, "r2_shape_clim": s_c, "rmse_model_m": r_m,
            "rmse_clim_m": r_c, "bias_model_m": _bias(pred), "bias_clim_m": _bias(clim),
            "rmse_ratio": ratio, "rmse_k": float(k), "verdict": "PASS" if ok else "FAIL"}


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
                dump_path: str | None = None, device=None, boundaries=None,
                fix_eta: float | None = None, fix_head_extra: float | None = None,
                pump_split: bool = False, return_flow: bool = False,
                spread_km: float | None = None, learn_spread: bool = False,
                loss_mode: str = "level", level_weight: float = 0.1,
                delay_storage: str = "off", rivers: tuple | None = None,
                sw_field: torch.Tensor | None = None, sw_layer: int | None = None,
                fix_sw_scale: float | None = None, delay_u0: str = "eq",
                delay_layers: tuple[int, ...] | None = None,
                aquitard: str = "off", zone_w: np.ndarray | None = None,
                ic_zone_of_cell: np.ndarray | None = None,
                proximal_layered: bool = False, ic_layered: bool = False,
                well_datum: str = "off",
                well_datum_sd: float = WELL_DATUM_SD_DEFAULT,
                datum_mask: torch.Tensor | None = None) -> dict:
    """K-fold cross-validation over wells (Ruling 2: k-fold, never leave-one-out).

    ``well_datum="fit"``: each fold's fit gives its KEPT (in-fold) wells a datum in the
    observation operator (``fit_flow``). A held-out well has no fitted datum, so it is
    predicted with ``d = 0``: the physical head at its cell. The k-fold gate therefore
    still tests absolute levels at unseen wells, which is what spatial interpolation
    needs -- a datum cannot be known where no well was ever observed. ``datum_mask``
    (``(W, T)``, the months that inform a datum; ``fit_flow``) is sliced to the fold.

    ``rivers`` is ``(RiverSet, layer, mode)`` or ``None``; it is attached to every fold's
    model. ``delay_storage``/``sw_field``/``fix_sw_scale`` pass through to ``fit_flow``.

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
            if ic_zone_of_cell is not None:
                # --ic-merged-proximal, from the fold's kept wells only (no leakage)
                h0_fold, _ = _merged_proximal_heads(
                    grid, h0_fold, well_xy[keep], np.asarray(obs_h0)[keep],
                    ic_zone_of_cell, ic_zone_of_cell[obs_idx.cpu().numpy()[keep]],
                    layer_of=obs_layer_np[keep] if ic_layered else None)
        m = FlowModel(grid, n_layers=n_layers, dt_days=30.0, device=device,
                      boundaries=boundaries)
        if rivers is not None:
            m.set_rivers(rivers[0], layer=rivers[1], mode=rivers[2])
        fit = fit_flow(m, obs_h[keep], obs_idx[keep], obs_layer[keep], recharge,
                       E=E, ground_elev=ground_elev, epochs=epochs, lr=lr,
                       param_mode=param_mode, h0=h0_fold, recharge_field=recharge_field,
                       pump_layer=pump_layer, recharge_layer=recharge_layer,
                       zone_of_cell=zone_of_cell, fix_eta=fix_eta,
                       fix_head_extra=fix_head_extra, pump_split=pump_split,
                       return_flow=return_flow, spread_km=spread_km, learn_spread=learn_spread,
                       loss_mode=loss_mode, level_weight=level_weight,
                       delay_storage=delay_storage, sw_field=sw_field, sw_layer=sw_layer,
                       fix_sw_scale=fix_sw_scale, delay_u0=delay_u0,
                       delay_layers=delay_layers, aquitard=aquitard, zone_w=zone_w,
                       proximal_layered=proximal_layered, well_datum=well_datum,
                       well_datum_sd=well_datum_sd,
                       datum_mask=None if datum_mask is None else datum_mask[keep])
        print(f"    fold {f + 1}/{n_folds}: n_held={len(held)} loss={fit['loss']:.4g} "
              f"({time.perf_counter() - t_fold:.1f}s)", flush=True)
        with torch.no_grad():
            # The model may live on CUDA (--device); every tensor entering the rollout and
            # every index tensor addressing its output has to follow it there.
            fdev = m.log_T.device
            h0_eval = (h0_fold if h0_fold is not None
                      else torch.zeros(n_layers, n_active, dtype=torch.float64))
            h0_eval = h0_eval.to(dtype=torch.float64, device=fdev)
            if (param_mode in ("homogeneous", "zonal")
                    and (E is not None or recharge_field is not None)):
                h = _predict_homogeneous(m, fit, h0_eval, n_steps, recharge=recharge,
                                         recharge_field=recharge_field, E=E,
                                         ground_elev=ground_elev,
                                         recharge_layer=recharge_layer, pump_layer=pump_layer,
                                         sw_field=sw_field, sw_layer=sw_layer)
            else:
                h = m(h0_eval, recharge,
                     torch.zeros(n_layers, n_active, n_steps, dtype=torch.float64,
                                 device=fdev), n_steps)
            # held-out wells: d = 0 under --well-datum fit (no datum without a well)
            p = h[obs_layer[held].to(fdev), obs_idx[held].to(fdev), 1:].cpu().numpy()
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
        # The fold's own parameter set is kept (2026-09-11): five fold models are the
        # cheapest honest ensemble the forward twin can draw its parameter spread from.
        per_fold.append({"fold": f, "n_held": len(held), "fit_loss": fit["loss"],
                         "r2_kfold": _r2(p, obs_h[held].numpy()),
                         "r2_idw": _r2(idw, obs_h[held].numpy()),
                         "bounds_hit": fit["bounds_hit"],
                         "theta": fit.get("theta", {})})
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


GROUND_ELEV_MODES = ("wells", "dem")


def _dem_on_grid(grid, dem_npz: str, polygon: str | None = None) -> np.ndarray:
    """The basemap's SRTM elevation (``basemap.npz["dem"]``) on ``grid``'s active cells.

    ``basemap.py`` samples SRTM bilinearly at the active-cell centroids of the dx=1000 m
    grid, in ``grid.centroids()`` order, so on that grid it is used as is (``rivers.
    load_dem`` checks the cell count). On any other dx it is resampled by nearest
    dx=1000 m centroid, which needs the polygon the grid was built from (``polygon``,
    default ``DEFAULT_PATHS["polygon"]``)."""
    from .rivers import load_dem

    with np.load(dem_npz) as z:
        if "dem" not in z:
            raise ValueError(f"{dem_npz} has no 'dem' (it was fetched with --no-dem)")
        n = int(z["dem"].shape[0])
    if n == grid.n_active:
        return load_dem(dem_npz, grid)[0]
    g1 = build_grid(polygon or DEFAULT_PATHS["polygon"], dx=1000.0)
    dem, _ = load_dem(dem_npz, g1)
    c1, c = g1.centroids(), grid.centroids()
    nn_idx = np.array([int(np.argmin(((c1 - p) ** 2).sum(1))) for p in c])
    return dem[nn_idx]


def _load_ground_elev(grid, stn: pd.DataFrame, mode: str = "wells",
                      dem_npz: str = "results/twin/basemap.npz",
                      log=print, polygon: str | None = None) -> torch.Tensor:
    """Ground-elevation field (m) for the pump lift, on every active cell.

    ``mode="wells"`` (default, the historical field): IDW of the fan stations'
    ``GroundHeight``. NaN is skipped but 0.0 is taken as a real elevation, although the
    audit of 2026-09-23 found 60 of 158 wells carry exactly 0.0 (a missing-value code: the
    SRTM surface is 30 m at some of them); the count is printed on every call so a run's
    log shows it. ``mode="dem"`` (``--ground-elev dem``, opt-in) takes the basemap's SRTM
    field instead (``_dem_on_grid``), which agrees with ``WellElevation`` to a median of
    about 2 m.
    """
    from .heads import _station_xy

    if mode not in GROUND_ELEV_MODES:
        raise ValueError(f"ground-elevation mode must be one of {GROUND_ELEV_MODES}, "
                         f"got {mode!r}")
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
    n_zero = int(sum(g == 0.0 for g in ge))
    if mode == "dem":
        dem = _dem_on_grid(grid, dem_npz, polygon=polygon)
        if log:
            log(f"ground elevation: SRTM from {dem_npz} (--ground-elev dem), "
                f"{np.min(dem):.1f}..{np.max(dem):.1f} m, median {np.median(dem):.1f} m; "
                f"station GroundHeight not used ({n_zero} of {len(ge)} finite values are 0.0)")
        return torch.tensor(np.asarray(dem, dtype="float64"), dtype=torch.float64)
    if not xy:
        raise ValueError("no station carried a finite GroundHeight -- cannot build a "
                         "ground-elevation field for the pumping driver")
    if log:
        log(f"ground elevation: IDW of {len(ge)} station GroundHeight values, {n_zero} of "
            "them exactly 0.0 and used as real elevations (--ground-elev dem replaces "
            "this field with SRTM)")
    return torch.tensor(_idw_field(grid, np.array(xy), np.array(ge)), dtype=torch.float64)


def _load_pumping_kwh(grid, pump_census_path: str, pump_kwh_path: str,
                      t0: str, t1: str, meter_filter: str = "dedupe-cap",
                      cap_duty: float = 1.0, eta_classes: bool = False
                      ) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
    """Monthly electricity census -> (A, T) kWh per active cell (Task 4's aggregate_pumps),
    or ``((C, A, T), class_names)`` with ``eta_classes=True``.

    ``meter_filter`` (2026-09-11, ``pumping.clean_census``): ``"none"`` is the raw census
    every result before that date used; ``"dedupe"`` counts each shared meter once;
    ``"dedupe-cap"`` also drops meters whose mean draw exceeds ``cap_duty`` x their rated
    motor capacity. On this census the three give 9.9, 6.6 and 2.2 TWh over 2012-2022.
    """
    for p, label in ((pump_census_path, "pump census"), (pump_kwh_path, "pump kWh")):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{label} parquet not found at {p!r} -- fix round 1's pumping driver "
                "needs it; stopping rather than silently skipping it. See "
                "task-5-report.md's fix-round-1 section for where the real 116,769-pump "
                "census currently lives (it has no stable in-repo path yet).")
    pumps = pd.read_parquet(pump_census_path)
    kwh = pd.read_parquet(pump_kwh_path)
    if meter_filter != "none":
        pumps, kwh, report = pumping_mod.clean_census(
            pumps, kwh, cap_duty=(cap_duty if meter_filter == "dedupe-cap" else None),
            t0=t0, t1=t1)
        print(f"pump census ({meter_filter}): raw {report['kwh_raw_GWh'].sum():.0f} GWh -> "
              f"de-duplicated {report['kwh_dedup_GWh'].sum():.0f} -> kept "
              f"{report['kwh_kept_GWh'].sum():.0f} GWh over {len(pumps)} pumps; dropped "
              f"{int(report['n_meters_dropped'].sum())} meters over capacity", flush=True)
    if eta_classes:
        from .scenario import CLASSES, energy_by_class

        by_cls, _dates = energy_by_class(pumps, kwh, grid, t0, t1)
        names = [c for c in CLASSES if c in by_cls]
        return torch.tensor(np.stack([by_cls[c] for c in names]), dtype=torch.float64), names
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


# Where the real inputs live once the data cache is in place (docs/DATA_FORMAT.md,
# ``hydrophysics.twin.fetch_amp``). Every path is relative to the repo root. The old
# defaults pointed at a doubled ``chou-shui-data/chou-shui-data/`` prefix and, for the
# census, at another machine's scratchpad, so every run had to spell all of them out.
DEFAULT_PATHS = {
    "polygon": "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json",
    "wells_dir": "AMP_V2/data/wells",
    "stations": "AMP_V2/data/fan_stations.parquet",
    "pump_census": "AMP_V2/data/tpc_pumps.parquet",
    "pump_kwh": "AMP_V2/data/pump_kwh_all.parquet",
    "rf_timeseries": "chou-shui-data/data/rf_timeseries.csv",
    "rf_stations": "chou-shui-data/data/rf_stations.csv",
    "gw_stations": "chou-shui-data/data/gw_stations.csv",
    "et_npz": "results/et/openmeteo_et0_2012_2022.npz",
}


def _write_theta(path: str, theta: dict, meta: dict) -> None:
    """Persist a calibrated parameter set with enough provenance to rebuild the model."""
    with open(path, "w") as fh:
        json.dump({"theta": theta, "meta": meta}, fh, indent=1)


_PARAM_MODES = ("homogeneous", "percell", "zonal")
_DEFAULT_ZONE_BOUNDARIES = "205,182"


def _parse_zone_boundaries(text: str, allow_split: bool = False) -> tuple:
    """``"205,182"`` -> ``(205.0, 182.0)``: the proximal/mid and mid/distal eastings in km.

    ``allow_split`` (round 3): also accept ``"205,182,S"`` with ``S > 205``, the opt-in
    split of the proximal zone at easting S, and return ``(proximal, distal, split)`` with
    ``split`` None for two values. Callers that only know three zones leave it False, so a
    four-zone model given to them fails here instead of being rebuilt as three zones.

    Spec §4.2 requires re-running the gate varying only the SECOND (mid/distal) value,
    at 178 and 186 km, because that boundary is an equal-width default with no
    independent justification -- the first (proximal/mid) value must stay 205. A
    transposed or equal pair would silently move the proximal boundary or empty the mid
    zone, so both are rejected rather than tolerated.
    """
    parts = [p.strip() for p in str(text).split(",")]
    split_km = None
    if allow_split and len(parts) == 3:
        try:
            split_km = float(parts[2])
        except ValueError as exc:
            raise ValueError(f"--zone-boundaries values must be numbers, got {text!r}") from exc
        parts = parts[:2]
    if len(parts) != 2:
        raise ValueError(
            f"--zone-boundaries wants PROXIMAL_KM,DISTAL_KM (e.g. '205,182'), got {text!r}"
            + ("" if allow_split else "; a third value (the proximal split) is not "
               "supported by this caller")
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
    if not allow_split:
        return proximal_km, distal_km
    if split_km is not None and not split_km > proximal_km:
        raise ValueError(f"--zone-boundaries third value (proximal split, {split_km}) must "
                         f"sit east of the proximal boundary ({proximal_km})")
    return proximal_km, distal_km, split_km


def _is_hit_entry(v) -> bool:
    """True for a single-parameter ``_clamp_`` result ``{"lo", "hi", "n"}`` -- the leaf
    of both the flat (homogeneous/percell) and nested (zonal) ``bounds_hit`` shapes.
    """
    return isinstance(v, dict) and {"lo", "hi", "n"} <= v.keys()


def _fmt_hit_entry(name: str, v: dict) -> str:
    return f"{name}: lo={v['lo']}/{v['n']} hi={v['hi']}/{v['n']}"


def _format_bounds_hit(hits: dict) -> str:
    """Flat dicts (homogeneous/percell: ``{param: {"lo","hi","n"}}``) print one line.
    Nested dicts (zonal: ``{zone: {param: {"lo","hi","n"}}}``) print one line per zone.

    Spec §6's primary rule reads this, and pooling would let an interior mid value mask
    a pinned proximal one -- so the zonal form is never collapsed into a single number.
    Printing ``lo/n`` and ``hi/n`` (Ruling P1) rather than a bare count is what lets a
    reader see "proximal log_T 1/1 at the lower bound" instead of mistaking a small
    absolute count for a minority.
    """
    if not hits:
        return "  bounds_hit={}"
    if all(_is_hit_entry(v) for v in hits.values()):
        parts = ", ".join(_fmt_hit_entry(k, v) for k, v in hits.items())
        return f"  bounds_hit={{{parts}}}"
    lines = []
    for zone, params in hits.items():
        parts = ", ".join(_fmt_hit_entry(k, v) for k, v in params.items())
        lines.append(f"    bounds_hit[{zone}]={{{parts}}}")
    return "\n".join(lines)


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
                         "'205,182'). The mid/distal (SECOND) value is an unjustified "
                         "default; spec 4.2 requires varying only that SECOND value -- "
                         "re-run at '205,178' and '205,186' -- to report whether the "
                         "verdict moves. Do NOT vary the first (proximal) value: e.g. "
                         "'186,178' moves the PROXIMAL boundary instead and is not the "
                         "sensitivity check spec 4.2 asks for. Opt-in third value "
                         "'205,182,S' (S > 205): split the proximal zone at easting S km "
                         "into 'proximal' (east) and 'proximal_w' (205..S), each one "
                         "merged aquifer (+2 parameters)")
    ap.add_argument("--log-t-min-proximal", type=float, default=None,
                    help="--param-mode zonal: raise the proximal log_T floor to this many "
                         "m2/day (both proximal parts when split). Default: the global "
                         "BOUNDS floor, 10 m2/day, for every zone")
    ap.add_argument("--zone-blend-km", type=float, default=0.0,
                    help="--param-mode zonal: blend the mid/distal parameters over a "
                         "logistic of this width (km) around the second boundary instead "
                         "of a step. The distal weight is sigmoid((distal_km - x)/w), and "
                         "log-parameters are mixed per cell. The proximal line stays "
                         "sharp. 0 (default) = the sharp zonation. Recorded in meta "
                         "(zone_blend_km), so twin.forward rebuilds the same field")
    ap.add_argument("--wells-dir", default=DEFAULT_PATHS["wells_dir"])
    ap.add_argument("--stations", default=DEFAULT_PATHS["stations"])
    ap.add_argument("--polygon", default=DEFAULT_PATHS["polygon"])
    ap.add_argument("--pump-census", default=DEFAULT_PATHS["pump_census"],
                    help="pump census parquet (sid, TWD97_X, TWD97_Y, PUMP_HP, PURPOSE)")
    ap.add_argument("--pump-kwh", default=DEFAULT_PATHS["pump_kwh"])
    ap.add_argument("--rf-timeseries", default=DEFAULT_PATHS["rf_timeseries"])
    ap.add_argument("--rf-stations", default=DEFAULT_PATHS["rf_stations"])
    ap.add_argument("--et-npz", default=DEFAULT_PATHS["et_npz"])
    ap.add_argument("--gw-stations", default=DEFAULT_PATHS["gw_stations"])
    ap.add_argument("--boundaries", choices=("none", "coast-apex"), default="coast-apex",
                    help="'coast-apex' (default since 2026-09-11) opens the basin with "
                         "general-head boundaries on the coast (h_b = 0) and at the apex "
                         "(h_b = initial head), each with a learnable conductance -- see "
                         "boundaries.py for why the closed basin pinned the forcing. "
                         "'none' reproduces every result recorded before that date.")
    ap.add_argument("--meter-filter", choices=("none", "dedupe", "dedupe-cap"),
                    default="dedupe-cap",
                    help="census cleaning (pumping.clean_census): count each shared "
                         "meter once, and (dedupe-cap) drop meters whose mean draw "
                         "exceeds --cap-duty x rated motor capacity. 'none' is the raw "
                         "census every result before 2026-09-11 used (9.9 TWh; the "
                         "default keeps 2.2).")
    ap.add_argument("--cap-duty", type=float, default=1.0)
    ap.add_argument("--eta-classes", action="store_true",
                    help="one wire-to-water efficiency per purpose class (irrigation, "
                         "aquaculture, livestock, domestic, industry, other) instead of "
                         "one for the whole census")
    ap.add_argument("--fix-eta", type=float, default=None,
                    help="hold the wire-to-water efficiency at this value (diagnostic; "
                         "every free fit so far drove it to the 0.05 floor)")
    ap.add_argument("--fix-head-extra", type=float, default=None,
                    help="hold the extra pumping head (m) at this value (diagnostic)")
    ap.add_argument("--pump-split", action="store_true",
                    help="learn the share of abstraction taken from layer 1 (index 0) "
                         "instead of --pump-layer; one logit parameter")
    ap.add_argument("--return-flow", action="store_true",
                    help="learn an irrigation return-flow fraction (<= 0.7) of the pumped "
                         "volume that re-enters layer 1 the same month; one logit")
    ap.add_argument("--pump-spread-km", type=float, default=None,
                    help="spread each cell's pumping energy over a Gaussian of this radius "
                         "(km, mass-conserving) before conversion -- spread.py")
    ap.add_argument("--learn-spread", action="store_true",
                    help="learn the spread radius (log, bounded 0.5-25 km) instead")
    ap.add_argument("--holdout-months", type=int, default=0,
                    help="temporal gate: fit on the record minus its last N months, then "
                         "score a free-running continuation over those N months against a "
                         "per-well month-of-year climatology built from the fitted months. "
                         "This scores the response to the forcing in time, which the "
                         "held-out-well gate cannot.")
    ap.add_argument("--no-backfill", action="store_true",
                    help="keep never-observed months as NaN (masked in the loss) instead "
                         "of interpolating across every gap with limit_direction='both', "
                         "which copies held-out heads of late-starting wells into the fit "
                         "target, climatology and initial condition; the initial "
                         "condition uses only the wells observed in month 0")
    ap.add_argument("--strict-coverage", action="store_true",
                    help="QC coverage as the fraction of calendar months observed in the "
                         "window, with leading/trailing gaps counted by the gap check "
                         "(default: len(samples)/n_hours, inflated by 10-minute data, and "
                         "gaps only between the first and last sample)")
    ap.add_argument("--ic-merged-proximal", action="store_true",
                    help="initial heads of EVERY layer in the proximal zone(s) (proximal, "
                         "and proximal_w with a split) from the IDW of that zone's own wells "
                         "regardless of layer code (one merged aquifer), instead of "
                         "per-layer IDW from distant mid-fan wells; the apex boundary head "
                         "inherits it through set_apex_heads")
    ap.add_argument("--ic-layered-proximal", action="store_true",
                    help="with --ic-merged-proximal: keep the layers inside the proximal "
                         "zone(s) -- layer k from the IDW of that zone's layer-k wells, the "
                         "merged value for a layer without one (proximal L3-L4). Pairs "
                         "with --proximal-layered; outside the proximal zone(s) nothing "
                         "changes")
    ap.add_argument("--proximal-layered", action="store_true",
                    help="--param-mode zonal: the proximal zone(s) get per-layer log_T and "
                         "log_S and a learnable log_L per interface (the mid/distal form, "
                         "same bounds incl. --log-t-min-proximal and --l-min) instead of "
                         "one merged aquifer with log_L pinned at its ceiling. Starts at "
                         "the merged values, so epoch 0 is the merged model. With "
                         "--ic-merged-proximal alone the initial heads stay merged per "
                         "zone and the layered model relaxes them")
    ap.add_argument("--ground-elev", choices=GROUND_ELEV_MODES, default="wells",
                    help="ground-elevation field for the pump lift: 'wells' = IDW of "
                         "station GroundHeight (historical; 0.0 codes used as real), "
                         "'dem' = SRTM per cell from --dem-npz")
    ap.add_argument("--loss", choices=("level", "anomaly"), default="level",
                    help="'anomaly' fits each well's departures from its own mean plus "
                         "--level-weight x the means (2026-09-18); 'level' is plain MSE")
    ap.add_argument("--level-weight", type=float, default=0.1)
    ap.add_argument("--well-datum", choices=WELL_DATUM_MODES, default="off",
                    help="'fit': each calibration well carries a datum d_i in the "
                         "observation operator (pred = model head + d_i), fitted with the "
                         "physical parameters under a N(0, --well-datum-sd^2) prior, in both "
                         "--loss variants. Not physical: never enters the flow solve, the "
                         "column, forward runs or subsidence; recorded per sid in theta meta "
                         "(well_datum). Temporal gate: fitted on the fitted months, applied "
                         "unchanged to the held-out ones. k-fold: held-out wells use d=0. "
                         "Homogeneous/zonal only. 'off' (default): no datum")
    ap.add_argument("--well-datum-sd", type=float, default=WELL_DATUM_SD_DEFAULT,
                    metavar="METRES",
                    help="prior sd of the per-well datum (default 5 m). The prior weight is "
                         "kappa = min(n, 12) s^2 / sd^2 months (s = within-well residual sd, "
                         "n = the well's observed fitted months), so a well keeps "
                         "n / (n + kappa) of its mean residual")
    ap.add_argument("--l-min", type=float, default=None,
                    help="leakance floor in 1/day (default 1e-8): raises BOUNDS['log_L']")
    # --- opt-in physics of 2026-09-23 (G1 drift, G6 rivers); defaults reproduce the past
    ap.add_argument("--delay-storage", choices=DELAY_MODES, default="off",
                    help="lumped delay bed (slow-release interbed storage exchanging with "
                         "each layer through a time constant): 'global' adds one S_d and "
                         "one tau, 'zonal' one pair per fan zone")
    ap.add_argument("--delay-tau-max-years", type=float, default=30.0,
                    help="ceiling of the delay-bed time constant")
    ap.add_argument("--delay-tau-min-days", type=float, default=DELAY_TAU_MIN_DAYS,
                    help="floor of the delay-bed time constant (default 30 = historical); "
                         "180 keeps the bed a slow release rather than extra instant "
                         "storage (S_eff and delay_is_elastic are reported)")
    ap.add_argument("--delay-u0", choices=DELAY_U0_MODES, default="eq",
                    help="initial state of the delay bed: 'eq' u0 = h0 (historical), "
                         "'learned' u0 = h0 + exp(log_du0[_zone]), the pre-2012 "
                         "disequilibrium of interbeds still draining after decades of "
                         "drawdown (0.01-50 m, init 1 m)")
    ap.add_argument("--delay-layers", default=None,
                    help="0-based layers that carry the delay bed, e.g. '1,2,3' (default "
                         "all); elsewhere S_d sits at its floor as a constant")
    ap.add_argument("--aquitard-storage", choices=DELAY_MODES, default="off",
                    help="a storage node on every layer interface (aquitard delayed "
                         "drainage; FlowModel.aqt_terms), in parallel with log_L: "
                         "'global' or per 'zonal' (S_a, G) pair")
    ap.add_argument("--spread-max-km", type=float, default=None,
                    help="ceiling of the learned stress radius (default 25 km; the "
                         "deliverable stage3_spreadL_gate was fitted at 10)")
    ap.add_argument("--rivers", choices=("none", "ghb", "riv"), default="none",
                    help="river cells from the channel polygons (rivers.py): 'ghb' linear "
                         "exchange, 'riv' MODFLOW RIV with a lagged connection switch "
                         "(refuses --compile-matvec). 'none' keeps rivers no-flow")
    ap.add_argument("--river-set", default="choushui,wu,beigang")
    ap.add_argument("--river-shp", default=None,
                    help="default $HYDROMIND_GW_DATA/water/river_TWD97.shp")
    ap.add_argument("--dem-npz", default="results/twin/basemap.npz",
                    help="SRTM per active cell from hydrophysics.twin.basemap (river stage)")
    ap.add_argument("--river-stage-depth", type=float, default=1.0,
                    help="river stage below the cell's ground elevation, m")
    ap.add_argument("--river-rbot-depth", type=float, default=3.0,
                    help="river-bed bottom below ground, m")
    ap.add_argument("--river-layer", type=int, default=0)
    ap.add_argument("--river-c-split", choices=("group", "zone"), default="group",
                    help="'zone': one conductance per fan zone for the Choushui (losing "
                         "proximal reach vs gaining distal reach); needs --param-mode zonal")
    ap.add_argument("--river-skip-apex", action=argparse.BooleanOptionalAction, default=True,
                    help="drop river cells that are also apex boundary cells, so the "
                         "Choushui's entry is not counted twice (default on)")
    ap.add_argument("--river-stage-season", default=None,
                    help="CSV of monthly connection factors per river group (month, "
                         "choushui, wu, beigang); default: always connected")
    ap.add_argument("--river-edge-buffer-m", type=float, default=None,
                    help="how far outside a cell the Wu/Beigang channel may lie and still "
                         "count as its boundary (default: dx)")
    ap.add_argument("--sw-recharge", default=None,
                    help="npz of canal irrigation deliveries per cell and month "
                         "(hydrophysics.twin.surface_water); adds a learnable recharge "
                         "fraction of it")
    ap.add_argument("--sw-layer", type=int, default=None,
                    help="layer that receives it (default --recharge-layer)")
    ap.add_argument("--sw-components", choices=SW_COMPONENT_CHOICES, default="delivered",
                    help="which surface-water fields of the --sw-recharge npz to use, each "
                         "with its own learned fraction: 'delivered' (canal deliveries, "
                         "historical), 'percolation' (paddy flood percolation, "
                         "surface_water.py v2) or 'both'")
    ap.add_argument("--fix-sw-scale", type=float, default=None,
                    help="hold the canal-recharge fraction at this value (diagnostic)")
    ap.add_argument("--policy-response", action="store_true",
                    help="after the run, score the policy response (policy_gate.py: "
                         "irrigation x1/0.85/0.7 for 120 months, reference column as a "
                         "proxy) and write scorecard.json into --out")
    ap.add_argument("--temporal-rmse-k", type=float, default=1.5,
                    help="temporal verdict: PASS needs shape R2 above climatology's and "
                         "RMSE below k x climatology's over the held-out months")
    ap.add_argument("--wells-from", default=None,
                    help="CSV with a 'sid' column: use only these wells. Every run "
                         "writes its own well list as stage3_wells.csv, so a --dx 500 "
                         "convergence check can be made like-for-like by passing the "
                         "1 km run's list here (the active-cell mask otherwise changes "
                         "the well set).")
    ap.add_argument("--pump-layer", type=int, default=1,
                    help="0-indexed layer that receives pumping (default 1 = layer 2, "
                         "the main production aquifer)")
    ap.add_argument("--recharge-layer", type=int, default=0,
                    help="0-indexed layer that receives rain-minus-ET0 recharge")
    ap.add_argument("--no-forcing", action="store_true",
                    help="skip the pumping/recharge drivers and h0 IC, reproducing the "
                         "original (degenerate, see module docstring) zero-forcing run")
    ap.add_argument("--out", default=None,
                    help="output directory (default: results/twin_runs/stage3_<UTC "
                         "timestamp>, so a new run never overwrites the published "
                         "results/twin/*.csv)")
    ap.add_argument("--log-every", type=int, default=0,
                    help="print in-sample R2 every N epochs during the fit")
    ap.add_argument("--dump-predictions", action="store_true",
                    help="write per-held-out-entry predictions (flow, IDW, obs) plus\neach entry's distance to the nearest training entry, for degradation-vs-distance\nanalysis without refitting.")
    ap.add_argument("--fit-only", action="store_true",
                    help="run the in-sample fit and skip the k-fold gate")
    ap.add_argument("--device", default=None,
                    help="'cuda', 'cpu', or omit to auto-select CUDA when available. "
                         "Until this flag existed both FlowModel construction sites "
                         "omitted device=, so every Stage-1/2/3 flow run silently used "
                         "CPU -- including the 4.3 h fit and 12,733 s/fold timings behind "
                         "the ~144 h sweep estimate. calibrate_mlcw has always "
                         "auto-selected CUDA, so the twin's two halves disagreed.")
    ap.add_argument("--compile-matvec", action="store_true",
                    help="torch.compile the conductance matvec (1.38x -> 1.91x on a "
                         "fan-scale rollout, ~8 s one-time warmup, head/gradient "
                         "unchanged to ~1e-9). Off by default because compiled kernel "
                         "selection is not bit-reproducible across inductor cache "
                         "states; recorded in the run's provenance when used.")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join("results", "twin_runs",
                                time.strftime("stage3_%Y%m%d-%H%M%S", time.gmtime()))
    set_compile_matvec(args.compile_matvec)
    set_l_min(args.l_min)
    set_delay_tau_max(args.delay_tau_max_years)
    set_delay_tau_min(args.delay_tau_min_days)
    set_spread_max_km(args.spread_max_km)
    if args.log_t_min_proximal is not None and args.param_mode != "zonal":
        raise SystemExit("--log-t-min-proximal needs --param-mode zonal")
    set_log_t_min_proximal(args.log_t_min_proximal)
    if args.proximal_layered and args.param_mode != "zonal":
        raise SystemExit("--proximal-layered needs --param-mode zonal")
    if args.ic_layered_proximal and not args.ic_merged_proximal:
        raise SystemExit("--ic-layered-proximal refines --ic-merged-proximal; pass both")
    if args.well_datum != "off" and args.param_mode == "percell":
        raise SystemExit("--well-datum fit needs --param-mode homogeneous or zonal")
    if args.well_datum != "off" and not args.well_datum_sd > 0.0:
        raise SystemExit("--well-datum-sd must be > 0")
    delay_layers = parse_delay_layers(args.delay_layers)
    if args.delay_u0 != "eq" and args.delay_storage == "off":
        raise SystemExit("--delay-u0 learned needs --delay-storage global or zonal")
    if args.river_stage_season and args.compile_matvec and args.rivers != "none":
        raise SystemExit("--river-stage-season rebuilds the operator every month; drop "
                         "--compile-matvec")
    if args.rivers == "riv" and args.compile_matvec:
        raise SystemExit("--rivers riv rebuilds the operator whenever the river switch "
                         "changes; drop --compile-matvec (or use --rivers ghb)")
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)

    from .heads import build_head_field

    grid = build_grid(args.polygon, dx=args.dx)

    zone_of_cell = zone_counts = proximal_km = distal_km = split_km = None
    if args.param_mode == "zonal":
        proximal_km, distal_km, split_km = _parse_zone_boundaries(args.zone_boundaries,
                                                                  allow_split=True)
        zone_of_cell = fan_zones(grid.centroids(), proximal_km=proximal_km,
                                 distal_km=distal_km, split_km=split_km)
        zone_counts = {name: int((zone_of_cell == i).sum())
                       for i, name in enumerate(zone_names(3 if split_km is None else 4))}
        print(f"zones: proximal/mid at {proximal_km:.0f} km, mid/distal at "
              f"{distal_km:.0f} km"
              + (f", proximal split at {split_km:g} km" if split_km is not None else "")
              + f" -> cells {zone_counts}", flush=True)
        empty = [n for n, c in zone_counts.items() if c == 0]
        if empty:
            raise SystemExit(
                f"zone(s) {empty} contain no active cells at these boundaries; "
                "the gate would score a model with fewer zones than it reports"
            )
    zone_w = None
    if args.zone_blend_km and args.zone_blend_km > 0.0:
        if args.param_mode != "zonal":
            raise SystemExit("--zone-blend-km needs --param-mode zonal")
        zone_w = zone_blend_weights(grid.centroids(), proximal_km, distal_km,
                                    args.zone_blend_km, split_km=split_km)
        print(f"zone blend: mid/distal mixed over {args.zone_blend_km:g} km; cells with "
              f"both weights > 5%: {int(((zone_w[1] > 0.05) & (zone_w[2] > 0.05)).sum())}",
              flush=True)

    stn = pd.read_parquet(args.stations)
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(args.wells_dir, stn, strict_coverage=args.strict_coverage)
    if args.strict_coverage:
        hf_legacy = build_head_field(args.wells_dir, stn)
        dropped = sorted(set(hf_legacy.sids) - set(hf.sids))
        added = sorted(set(hf.sids) - set(hf_legacy.sids))
        print(f"--strict-coverage: {len(hf.sids)} wells pass QC (legacy {len(hf_legacy.sids)}); "
              f"dropped {len(dropped)} {dropped}; added {len(added)} {added}", flush=True)

    allowed = None
    if args.wells_from:
        allowed = set(pd.read_csv(args.wells_from, dtype={"sid": str})["sid"])
    idx, lay, series, xy_used, sids_used, raw_series = [], [], [], [], [], []
    for w in range(len(hf)):
        if allowed is not None and str(hf.sids[w]) not in allowed:
            continue
        i = grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1]))
        if i is None:
            continue
        sids_used.append(str(hf.sids[w]))
        raw_series.append(np.asarray(hf.heads[w], dtype="float64"))
        s = prepare_series(hf.heads[w], backfill=not args.no_backfill)
        idx.append(i)
        lay.append(max(int(hf.layers[w]) - 1, 0))
        series.append(s)
        xy_used.append(hf.xy[w])
    obs_h_full = np.stack(series)                                  # (W, 132), raw heads
    obs_raw_full = np.stack(raw_series)          # (W, 132), NaN where never observed
    if args.no_backfill:
        print(f"--no-backfill: {int((~np.isfinite(obs_h_full)).sum())} never-observed "
              "month-cells masked from the loss and scoring; initial condition from the "
              f"{int(np.isfinite(obs_h_full[:, 0]).sum())} wells observed in month 0",
              flush=True)
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
    eta_class_names = None
    if not args.no_forcing:
        ground_elev = _load_ground_elev(grid, stn, mode=args.ground_elev,
                                        dem_npz=args.dem_npz, polygon=args.polygon,
                                        log=lambda m: print(m, flush=True))
        if args.ground_elev != "wells":
            ge_old = _load_ground_elev(grid, stn, log=None)
            d = (ground_elev - ge_old).numpy()
            print(f"--ground-elev {args.ground_elev}: change vs the station IDW field "
                  f"mean {d.mean():+.1f} m, median {np.median(d):+.1f} m, "
                  f"5-95% {np.percentile(d, 5):+.1f}..{np.percentile(d, 95):+.1f} m",
                  flush=True)
        E = _load_pumping_kwh(grid, args.pump_census, args.pump_kwh,
                              "2012-01-01", "2023-01-01", meter_filter=args.meter_filter,
                              cap_duty=args.cap_duty, eta_classes=args.eta_classes)
        if args.eta_classes:
            E, eta_class_names = E
            print(f"eta classes: {eta_class_names}", flush=True)
        E = E[..., 1:]
        recharge_field = _load_recharge_field(grid, args.rf_timeseries, args.rf_stations,
                                              args.et_npz, args.gw_stations,
                                              "2012-01-01", "2023-01-01")[:, 1:]
    sw_field = None
    if args.sw_recharge:
        from .inputs import load_sw_recharge

        sw_field = load_sw_recharge(
            args.sw_recharge, grid,
            pd.date_range("2012-01-01", "2023-01-01", freq="MS", inclusive="left"),
            components=args.sw_components)[..., 1:]
        comp_mm = " + ".join(
            f"{k} {float(f.mean()) * 365.25 * 1000:.0f}"
            for k, f in zip(SW_COMPONENT_KEYS[args.sw_components],
                            sw_field if sw_field.dim() == 3 else [sw_field], strict=True))
        print(f"surface-water forcing: {args.sw_recharge}, fan mean {comp_mm} mm/yr "
              f"(recharge fraction {'fixed ' + str(args.fix_sw_scale) if args.fix_sw_scale else 'learned'})",
              flush=True)

    # temporal gate: the fit sees the record minus its last --holdout-months
    T_full = obs_h.shape[1]
    T_fit = T_full - int(args.holdout_months) if args.holdout_months > 0 else T_full
    if T_fit < 24:
        raise SystemExit("--holdout-months leaves fewer than 24 months to fit")
    obs_h_full_t, obs_h = obs_h, obs_h[:, :T_fit]
    n_steps = T_fit
    recharge_dummy = torch.zeros(4, grid.n_active, n_steps, dtype=torch.float64)
    h0_all = _idw_initial_heads(grid, well_xy, obs_h0, obs_layer_np, n_layers=4)
    ic_zone_of_cell = None
    if args.ic_merged_proximal:
        ic_zone_of_cell = _ic_zone_map(grid, args.zone_boundaries)
        h0_all, ic_rep = _merged_proximal_heads(grid, h0_all, well_xy, obs_h0,
                                                ic_zone_of_cell,
                                                ic_zone_of_cell[np.asarray(idx)],
                                                layer_of=(obs_layer_np
                                                          if args.ic_layered_proximal
                                                          else None))
        print(f"--ic-merged-proximal{' --ic-layered-proximal' if args.ic_layered_proximal else ''}"
              f": {ic_rep}", flush=True)
    nan_frac = float(np.isnan(np.stack([hf.heads[w] for w in range(len(hf))])).mean())
    print(f"head field: {len(hf)} wells passed QC, {len(sids_used)} inside the grid, "
          f"{100 * nan_frac:.2f}% NaN month-cells before interpolation", flush=True)
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame({"sid": sids_used, "layer": obs_layer_np + 1,
                  "x": well_xy[:, 0], "y": well_xy[:, 1]}).to_csv(
        os.path.join(args.out, "stage3_wells.csv"), index=False)

    boundaries = None
    if args.boundaries == "coast-apex":
        boundaries = fan_boundaries(grid, proximal_km=(proximal_km if proximal_km
                                                       is not None else 205.0))
        print(f"boundaries: {boundaries.describe()}", flush=True)
    else:
        print("boundaries: none (closed basin)", flush=True)

    m = FlowModel(grid, n_layers=4, dt_days=30.0, device=device, boundaries=boundaries)
    river_meta: dict = {"rivers": args.rivers}
    rivers_arg = None
    if args.rivers != "none":
        from .rivers import build_river_set, default_river_shp

        shp = args.river_shp or default_river_shp()
        from .rivers import apex_overlap

        if args.river_c_split == "zone" and zone_of_cell is None:
            raise SystemExit("--river-c-split zone needs --param-mode zonal")
        skip = (boundaries.apex_idx if (args.river_skip_apex and boundaries is not None)
                else None)
        rs, dem_sha1 = build_river_set(grid, shp=shp, groups=args.river_set,
                                       dem_npz=args.dem_npz,
                                       stage_depth=args.river_stage_depth,
                                       rbot_depth=args.river_rbot_depth,
                                       edge_buffer_m=args.river_edge_buffer_m,
                                       zone_of_cell=(None if zone_of_cell is None
                                                     else collapse_zones(zone_of_cell)),
                                       c_split=args.river_c_split, exclude_idx=skip,
                                       season_csv=args.river_stage_season)
        n_overlap = 0
        if boundaries is not None:
            rs_all, _ = build_river_set(grid, shp=shp, groups=args.river_set,
                                        dem_npz=args.dem_npz,
                                        stage_depth=args.river_stage_depth,
                                        rbot_depth=args.river_rbot_depth,
                                        edge_buffer_m=args.river_edge_buffer_m)
            n_overlap = apex_overlap(rs_all, boundaries.apex_idx)
            print(f"rivers: {n_overlap} river cells coincide with apex boundary cells, "
                  f"{apex_overlap(rs, boundaries.apex_idx)} after skip-apex "
                  f"({'on' if args.river_skip_apex else 'off'})", flush=True)
        m.set_rivers(rs, layer=args.river_layer, mode=args.rivers)
        rivers_arg = (rs, args.river_layer, args.rivers)
        river_meta.update({"river_set": args.river_set, "river_shp": shp,
                           "dem_npz": args.dem_npz, "dem_sha1": dem_sha1,
                           "river_stage_depth": args.river_stage_depth,
                           "river_rbot_depth": args.river_rbot_depth,
                           "river_layer": args.river_layer,
                           "river_edge_buffer_m": args.river_edge_buffer_m,
                           "river_groups": list(rs.names), "river_n_cells": rs.n_cells,
                           "river_c_split": args.river_c_split,
                           "river_skip_apex": bool(args.river_skip_apex),
                           "river_stage_season": args.river_stage_season,
                           "river_apex_overlap": int(n_overlap)})
        print(f"rivers ({args.rivers}, layer {args.river_layer}): {rs.describe()}", flush=True)
    git_commit = _git_commit()
    _reset_cg_stats()
    t0 = time.perf_counter()
    E_full, recharge_full, sw_full = E, recharge_field, sw_field
    # --well-datum fit: only months actually observed within the fitted period inform a
    # datum -- a back-filled month copies a neighbouring value, for a late-starting well a
    # held-out one, which would leak held-out levels into d_i
    datum_mask = (torch.from_numpy(np.isfinite(obs_raw_full[:, 1:][:, :T_fit]))
                  if args.well_datum == "fit" else None)
    if E is not None:
        E = E[..., :T_fit]
    if recharge_field is not None:
        recharge_field = recharge_field[:, :T_fit]
    if sw_field is not None:
        sw_field = sw_field[..., :T_fit]
    ins = fit_flow(m, obs_h, obs_idx, obs_layer, recharge_dummy, E=E, ground_elev=ground_elev,
                   epochs=args.epochs, lr=args.lr, param_mode=args.param_mode, h0=h0_all,
                   recharge_field=recharge_field, pump_layer=args.pump_layer,
                   recharge_layer=args.recharge_layer, log_every=args.log_every,
                   zone_of_cell=zone_of_cell, fix_eta=args.fix_eta,
                   fix_head_extra=args.fix_head_extra, pump_split=args.pump_split,
                   return_flow=args.return_flow, spread_km=args.pump_spread_km,
                   learn_spread=args.learn_spread, loss_mode=args.loss,
                   level_weight=args.level_weight, delay_storage=args.delay_storage,
                   sw_field=sw_field, sw_layer=args.sw_layer, fix_sw_scale=args.fix_sw_scale,
                   delay_u0=args.delay_u0, delay_layers=delay_layers,
                   aquitard=args.aquitard_storage, zone_w=zone_w,
                   proximal_layered=args.proximal_layered, well_datum=args.well_datum,
                   well_datum_sd=args.well_datum_sd, datum_mask=datum_mask)
    t_fit = time.perf_counter() - t0
    datum_np = datum_meta = None
    if args.well_datum == "fit":
        # fitted on obs_h, i.e. the fitted months only (T_fit); held-out months never enter
        datum_np = np.asarray(ins["well_datum"], dtype="float64")
        dstats = well_datum_stats(datum_np, ins["well_datum_n"], ins["well_datum_kappa"])
        dstats["sigma_within_m"] = ins["well_datum_sigma_m"]
        dstats["n_wells_no_obs"] = int((np.asarray(ins["well_datum_n"]) == 0).sum())
        dstats["r2_insample_with_datum"] = ins["r2_with_datum"]
        datum_meta = {"well_datum": {sid: float(v) for sid, v in zip(sids_used, datum_np,
                                                                     strict=True)},
                      "well_datum_mode": "fit", "well_datum_sd": float(args.well_datum_sd),
                      "well_datum_months_per_obs": WELL_DATUM_MONTHS_PER_OBS,
                      "well_datum_fit_months": int(T_fit), "well_datum_stats": dstats}
        print(f"--well-datum fit (sd {args.well_datum_sd:g} m): {dstats['n']} wells, mean "
              f"{dstats['mean_m']:+.2f} m, mean |d| {dstats['mean_abs_m']:.2f} m, rms "
              f"{dstats['rms_m']:.2f} m, max |d| {dstats['max_abs_m']:.2f} m; long-record kappa "
              f"{dstats['kappa_months']:.2f} months (within-well sd "
              f"{dstats['sigma_within_m']:.2f} m, min shrink {dstats['shrink_min']:.3f}); "
              f"in-sample R2 {ins['r2']:+.3f} physical, {ins['r2_with_datum']:+.3f} with "
              "datum", flush=True)
    temporal = None
    if args.holdout_months > 0:
        temporal = temporal_gate(m, ins, h0_all, obs_h_full_t, obs_idx, obs_layer, T_fit,
                                 E_full, recharge_full, ground_elev,
                                 recharge_layer=args.recharge_layer, pump_layer=args.pump_layer,
                                 sw_full=sw_full, sw_layer=args.sw_layer,
                                 rmse_k=args.temporal_rmse_k, keep_arrays=True,
                                 datum=datum_np)
        arrs = temporal.pop("arrays")
        # fair verdict (drift_diag.fair_temporal_verdict): a pure scoring addition beside
        # the legacy one -- per-well datum from the fitted months, raw (never back-filled)
        # observations, short-fit wells excluded, climatology + trend baseline
        from .drift_diag import fair_temporal_verdict

        fair = fair_temporal_verdict(arrs["pred"], obs_raw_full[:, 1:], T_fit)
        temporal.update({f"fair_{k}": v for k, v in fair.items()})
        temporal["verdict_fair"] = fair["verdict_fair"]
        os.makedirs(args.out, exist_ok=True)
        np.savez_compressed(os.path.join(args.out, "stage3_temporal_pred.npz"),
                            pred=arrs["pred"], obs=arrs["obs"], clim=arrs["clim"],
                            T_fit=arrs["T_fit"], sids=np.array(sids_used),
                            obs_raw=obs_raw_full[:, 1:],
                            # pred is the PHYSICAL head; pred + well_datum[:, None] is the
                            # datum-shifted prediction the datum_* metrics score
                            **({"well_datum": datum_np} if datum_np is not None else {}))
        pd.DataFrame([{**temporal, "holdout_months": args.holdout_months,
                       "no_backfill": bool(args.no_backfill),
                       **_input_opts_record(args),
                       "delay_storage": args.delay_storage, "delay_u0": args.delay_u0,
                       "aquitard_storage": args.aquitard_storage, "rivers": args.rivers,
                       "river_c_split": args.river_c_split,
                       "sw_recharge": args.sw_recharge or "",
                       "sw_components": args.sw_components,
                       "spread_km": ins.get("theta", {}).get("spread_km"),
                       "epochs": args.epochs,
                       **_well_datum_record(datum_meta),
                       "git_commit": _git_commit()}]).to_csv(
            os.path.join(args.out, "stage3_temporal.csv"), index=False)
        print(f"  TEMPORAL VERDICT: {temporal['verdict']} -- shape R2 "
              f"{temporal['r2_shape_model']:+.3f} vs climatology "
              f"{temporal['r2_shape_clim']:+.3f}; RMSE {temporal['rmse_model_m']:.2f} m vs "
              f"climatology {temporal['rmse_clim_m']:.2f} m (ratio "
              f"{temporal['rmse_ratio']:.2f}, must be < {args.temporal_rmse_k:g}); bias "
              f"{temporal['bias_model_m']:+.2f} m", flush=True)
        if datum_np is not None:
            print(f"  with the datum (not comparable with other runs' verdicts): "
                  f"{temporal['datum_verdict']} -- RMSE {temporal['datum_rmse_model_m']:.2f} m "
                  f"(ratio {temporal['datum_rmse_ratio']:.2f}), bias "
                  f"{temporal['datum_bias_model_m']:+.2f} m", flush=True)
        if fair.get("n_cells"):
            print(f"  FAIR TEMPORAL VERDICT: {fair['verdict_fair']} -- datum RMSE "
                  f"{fair['rmse_datum_model_m']:.2f} m vs best baseline "
                  f"{fair['best_baseline']} {fair['rmse_best_baseline_m']:.2f} m (ratio "
                  f"{fair['rmse_ratio_fair']:.2f}, must be <= {fair['fair_k']:g}); shape R2 "
                  f"{fair['r2_shape_datum_model']:+.3f} vs climatology "
                  f"{fair['r2_shape_clim']:+.3f} (tol {fair['fair_shape_tol']:g}); level "
                  f"error {fair['level_err_mean_abs_m']:.2f} m on "
                  f"{fair['n_wells_scored']} wells", flush=True)
        print(f"  TEMPORAL GATE ({args.holdout_months} held-out months, free-running "
              f"continuation): pooled R2 flow {temporal['r2_model']:+.3f} / climatology "
              f"{temporal['r2_clim']:+.3f} / persistence {temporal['r2_persist']:+.3f}; "
              f"anomaly R2 flow {temporal['r2_anom_model']:+.3f} / climatology "
              f"{temporal['r2_anom_clim']:+.3f} / persistence {temporal['r2_anom_persist']:+.3f}; "
              f"per-well median R2 flow {temporal['r2_well_median_model']:+.3f} / climatology "
              f"{temporal['r2_well_median_clim']:+.3f}; shape R2 flow "
              f"{temporal['r2_shape_model']:+.3f} / climatology {temporal['r2_shape_clim']:+.3f} "
              f"at a mean level error of {temporal['level_err_model_m']:.2f} m -> "
              f"{'PASS' if temporal['r2_anom_model'] > temporal['r2_anom_clim'] else 'FAIL'} "
              "(rule: anomaly R2 beats climatology)", flush=True)

    # Spec 6's PRIMARY decision rule -- "does the transmissivity clamp release, per zone?" --
    # is computed from THIS fit, not from the k-fold gate that follows. So it is printed here,
    # the moment it exists, rather than in the end-of-run report: a run killed or crashed
    # during the folds must still yield the primary answer.
    #
    # This ordering was paid for. A zonal seed-0 run killed mid-fold on 2026-09-01 had already
    # spent 4.3 h computing this exact result and emitted nothing, because the report came
    # after kfold_wells returned. The fit is ~4 h of a ~24 h zonal run, and spec 6 reaches the
    # SECONDARY rule (the margin, which the folds measure) only if the clamp released -- so the
    # cheap half of the run answers the question that can stop the expensive half.
    cg_fit_nonconverged, cg_fit_worst = _cg_stats()
    print(f"wells={obs_h.shape[0]} cells={grid.n_active} dx={args.dx:.0f}m "
          f"param_mode={args.param_mode} n_params={ins['n_params']} "
          f"forcing={'off' if args.no_forcing else 'on'} boundaries={args.boundaries} "
          f"meter_filter={args.meter_filter} eta_classes={eta_class_names} "
          f"fixed={ins.get('fixed', [])} pump_split={args.pump_split} "
          f"return_flow={args.return_flow} l_min={args.l_min} "
          f"spread_km={ins.get('theta', {}).get('spread_km')} holdout_months="
          f"{args.holdout_months} epochs={args.epochs}")
    print(f"  in-sample R2={ins['r2']:+.3f}  fit_time={t_fit:.1f}s")
    print(_format_bounds_hit(ins["bounds_hit"]))
    if "theta" in ins:
        print(f"  theta={ins['theta']}")
    print(f"  cg_maxiter={_CG_MAXITER}  cg_nonconverged={cg_fit_nonconverged}  "
          f"cg_worst_residual={cg_fit_worst:.3e}  git_commit={git_commit!r}", flush=True)

    _write_theta(os.path.join(args.out, "stage3_theta.json"), ins.get("theta", {}),
                 {"param_mode": args.param_mode, "boundaries": args.boundaries,
                  "zone_boundaries": args.zone_boundaries, "dx": args.dx,
                  **({"zone_blend_km": float(args.zone_blend_km)} if zone_w is not None
                     else {}),
                  **({"log_t_min_proximal": float(args.log_t_min_proximal)}
                     if args.log_t_min_proximal is not None else {}),
                  "pump_layer": args.pump_layer, "recharge_layer": args.recharge_layer,
                  "epochs": args.epochs, "git_commit": git_commit, "r2_insample": ins["r2"],
                  "bounds_hit": ins["bounds_hit"], "n_wells": int(obs_h.shape[0]),
                  "meter_filter": args.meter_filter, "cap_duty": args.cap_duty,
                  "eta_classes": eta_class_names, "fix_eta": args.fix_eta,
                  "fix_head_extra": args.fix_head_extra, "pump_split": args.pump_split,
                  "return_flow": args.return_flow, "l_min": args.l_min,
                  "pump_spread_km": args.pump_spread_km, "learn_spread": args.learn_spread,
                  "holdout_months": args.holdout_months, "temporal_gate": temporal,
                  "loss": args.loss, "level_weight": args.level_weight,
                  "delay_storage": args.delay_storage,
                  "delay_tau_max_years": args.delay_tau_max_years,
                  "delay_tau_min_days": args.delay_tau_min_days,
                  "delay_u0": args.delay_u0,
                  "delay_layers": list(delay_layers) if delay_layers is not None else None,
                  "delay_initial_state": ("u0 = h0 + exp(log_du0) (learned pre-2012 "
                                          "disequilibrium)" if args.delay_u0 == "learned"
                                          else "u0 = h0 (equilibrium; pre-2012 "
                                               "disequilibrium not represented)"),
                  "aquitard_storage": args.aquitard_storage,
                  "spread_max_km": (args.spread_max_km if args.spread_max_km is not None
                                    else math.exp(BOUNDS["log_spread_km"][1])),
                  "sw_components": args.sw_components,
                  **river_meta,
                  "sw_recharge": args.sw_recharge, "sw_layer": args.sw_layer,
                  "fix_sw_scale": args.fix_sw_scale,
                  "temporal_rmse_k": args.temporal_rmse_k,
                  "no_backfill": bool(args.no_backfill),
                  **_input_opts_record(args),
                  # opt-in 2026-09-26: written only when set, so default meta is unchanged
                  **(datum_meta or {})})

    if args.fit_only:
        # Discriminator mode: the in-sample TRAJECTORY separates under-training from a
        # structurally inadequate parameterisation, and costs one fit instead of eleven.
        tr = ins.get("r2_trace") or []
        if tr:
            print("  trajectory: " + "  ".join(f"{e}:{r:+.3f}" for e, r in tr))
            last = [r for _, r in tr[-3:]]
            if len(last) >= 2:
                drift = last[-1] - last[0]
                print(f"  change over the last {len(last)} logged points: {drift:+.4f} "
                      f"-> {'STILL CLIMBING (under-trained)' if drift > 0.01 else 'PLATEAUED (structural)'}")
        os.makedirs(args.out, exist_ok=True)
        trace_df = pd.DataFrame(tr, columns=["epoch", "r2_insample"])
        trace_df["cg_maxiter"] = _CG_MAXITER
        trace_df["cg_check_every"] = _CG_CHECK_EVERY
        trace_df["compile_matvec"] = bool(args.compile_matvec)
        trace_df["git_commit"] = git_commit
        trace_df.to_csv(os.path.join(args.out, "stage3_fit_trace.csv"), index=False)
        print(f"wrote {os.path.join(args.out, 'stage3_fit_trace.csv')}")
        if args.policy_response:
            _run_policy_gate(args)
        return

    t0 = time.perf_counter()
    gate = kfold_wells(grid, obs_h, obs_idx, obs_layer, recharge_dummy, n_layers=4,
                       epochs=args.epochs, lr=args.lr, n_folds=args.n_folds,
                       seed=args.seed,
                       param_mode=args.param_mode, well_xy=well_xy, obs_h0=obs_h0,
                       ground_elev=ground_elev, E=E, recharge_field=recharge_field,
                       pump_layer=args.pump_layer, recharge_layer=args.recharge_layer,
                       device=device,
                       dump_path=(os.path.join(args.out, "stage3_per_entry.npz")
                                  if args.dump_predictions else None),
                       zone_of_cell=zone_of_cell, boundaries=boundaries,
                       fix_eta=args.fix_eta, fix_head_extra=args.fix_head_extra,
                       pump_split=args.pump_split, return_flow=args.return_flow,
                       spread_km=args.pump_spread_km, learn_spread=args.learn_spread,
                       loss_mode=args.loss, level_weight=args.level_weight,
                       delay_storage=args.delay_storage, rivers=rivers_arg,
                       sw_field=sw_field, sw_layer=args.sw_layer,
                       fix_sw_scale=args.fix_sw_scale, delay_u0=args.delay_u0,
                       delay_layers=delay_layers, aquitard=args.aquitard_storage,
                       zone_w=zone_w, ic_zone_of_cell=ic_zone_of_cell,
                       proximal_layered=args.proximal_layered,
                       ic_layered=args.ic_layered_proximal,
                       well_datum=args.well_datum, well_datum_sd=args.well_datum_sd,
                       datum_mask=datum_mask)
    t_gate = time.perf_counter() - t0
    with open(os.path.join(args.out, "stage3_fold_thetas.json"), "w") as fh:
        json.dump([{"fold": f["fold"], "n_held": f["n_held"], "r2_kfold": f["r2_kfold"],
                    "r2_idw": f["r2_idw"], "theta": f["theta"]} for f in gate["per_fold"]],
                  fh, indent=1)
    cg_nonconverged, cg_worst_residual = _cg_stats()
    # The in-sample block (R2, per-zone bounds_hit, theta) was printed before the gate began;
    # it is not repeated here. What follows is what only the gate can tell you.
    print(f"  {args.n_folds}-fold R2={gate['r2_kfold']:+.3f}   "
          f"IDW baseline R2={gate['r2_idw']:+.3f}  gate_time={t_gate:.1f}s")
    print(f"  folds grouped by site: {gate['n_wells']} entries / {gate['n_sites']} sites   "
          f"co-location rate={gate['colocation_rate']:.3f}")
    if not (gate["colocation_rate"] >= 0.0 and gate["colocation_rate"] < 1e-9):
        print("  WARNING: held-out entries sit at zero distance from training entries, so "
              "the IDW baseline is reading co-located screens rather than interpolating. "
              "Treat the comparison below as biased toward IDW.")
    # Fix wave I5: the primary rule (does the clamp release?) must be checkable on the
    # fold models too, not only the in-sample fit -- a PASS where every fold model is
    # bound-saturated must be visible, not silently discarded.
    print("  per-fold bounds_hit:")
    for f in gate["per_fold"]:
        print(f"    -- fold {f['fold']} (n_held={f['n_held']}, "
              f"r2_kfold={f['r2_kfold']:+.3f}, r2_idw={f['r2_idw']:+.3f}) --")
        print(_format_bounds_hit(f["bounds_hit"]))
    print(f"GATE ({args.n_folds}-fold): "
          f"{'PASS' if gate['r2_kfold'] > gate['r2_idw'] else 'FAIL'}")
    # Fix wave I1/I2: the CG cap and convergence evidence must travel with the result --
    # the maxiter ruling is "both arms at the same cap", and the previous corruption
    # (median true relative residual 4.955e-02) was caught only by ad-hoc log grepping.
    print(f"  cg_maxiter={_CG_MAXITER}  cg_nonconverged={cg_nonconverged}  "
          f"cg_worst_residual={cg_worst_residual:.3e}  git_commit={git_commit!r}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stage3_flow.csv")
    pd.DataFrame([{"n_wells": gate["n_wells"], "n_cells": grid.n_active, "dx": args.dx,
                   "param_mode": args.param_mode,
                   "zone_proximal_km": (proximal_km if args.param_mode == "zonal"
                                        else ""),
                   "zone_distal_km": (distal_km if args.param_mode == "zonal" else ""),
                   "zone_split_km": split_km if split_km is not None else "",
                   "log_t_min_proximal": (args.log_t_min_proximal
                                          if args.log_t_min_proximal is not None else ""),
                   "zone_blend_km": float(args.zone_blend_km or 0.0),
                   "zone_cell_counts": (str(zone_counts) if args.param_mode == "zonal"
                                        else ""),
                   "n_params": ins["n_params"],
                   "forcing": "off" if args.no_forcing else "on",
                   "boundaries": args.boundaries, "meter_filter": args.meter_filter,
                   "eta_classes": str(eta_class_names), "fix_eta": args.fix_eta,
                   "fix_head_extra": args.fix_head_extra, "pump_split": args.pump_split,
                   "return_flow": args.return_flow, "l_min": args.l_min,
                   "spread_km": ins.get("theta", {}).get("spread_km"),
                   "holdout_months": args.holdout_months, "loss_mode": args.loss,
                   "r2_temporal": temporal["r2_model"] if temporal else "",
                   "r2_temporal_clim": temporal["r2_clim"] if temporal else "",
                   "temporal_verdict": temporal["verdict"] if temporal else "",
                   "temporal_rmse_ratio": temporal["rmse_ratio"] if temporal else "",
                   "delay_storage": args.delay_storage, "rivers": args.rivers,
                   "river_set": args.river_set if args.rivers != "none" else "",
                   "sw_recharge": args.sw_recharge or "",
                   "delay_u0": args.delay_u0, "aquitard_storage": args.aquitard_storage,
                   "sw_components": args.sw_components,
                   "river_c_split": args.river_c_split,
                   "no_backfill": bool(args.no_backfill),
                   **_input_opts_record(args),
                   **_well_datum_record(datum_meta),
                   "epochs": args.epochs,
                   "n_folds": gate["n_folds"], "seed": args.seed,
                   "n_sites": gate["n_sites"],
                   "colocation_rate": gate["colocation_rate"], "loss": ins["loss"],
                   "r2_insample": ins["r2"], "r2_kfold": gate["r2_kfold"],
                   "r2_idw": gate["r2_idw"], "bounds_hit": str(ins["bounds_hit"]),
                   "fold_bounds_hit": str([f["bounds_hit"] for f in gate["per_fold"]]),
                   "theta": str(ins.get("theta", {})),
                   "cg_maxiter": _CG_MAXITER, "cg_check_every": _CG_CHECK_EVERY,
                   "compile_matvec": bool(args.compile_matvec),
                   "cg_nonconverged": cg_nonconverged,
                   "cg_worst_residual": cg_worst_residual, "git_commit": git_commit,
                   "fit_time_s": t_fit, "gate_time_s": t_gate}]).to_csv(path, index=False)
    print(f"wrote {path}")
    # The theta file was written before the folds ran; stamp the verdict into it now so
    # the forward twin and the viewer inherit it (they print it on every run).
    theta_path = os.path.join(args.out, "stage3_theta.json")
    try:
        with open(theta_path) as fh:
            obj = json.load(fh)
        obj["meta"]["gate"] = {"r2_kfold": gate["r2_kfold"], "r2_idw": gate["r2_idw"],
                               "margin": gate["r2_kfold"] - gate["r2_idw"],
                               "verdict": "PASS" if gate["r2_kfold"] > gate["r2_idw"] else "FAIL",
                               "n_folds": gate["n_folds"], "seed": args.seed}
        with open(theta_path, "w") as fh:
            json.dump(obj, fh, indent=1)
    except OSError as e:
        print(f"could not stamp the verdict into {theta_path}: {e}")
    if args.policy_response:
        _run_policy_gate(args)


def _input_opts_record(args) -> dict:
    """The opt-in input constructions of 2026-09-23, as recorded in the theta meta and the
    CSVs (``inputs.input_options`` reads them back for the forward path)."""
    return {"ic_merged_proximal": bool(args.ic_merged_proximal),
            "ground_elev": args.ground_elev,
            "strict_coverage": bool(args.strict_coverage),
            **({"ground_elev_dem_npz": args.dem_npz} if args.ground_elev == "dem" else {}),
            # opt-in 2026-09-25, written only when set so default outputs keep their columns
            **({"ic_layered_proximal": True} if args.ic_layered_proximal else {}),
            **({"proximal_layered": True} if args.proximal_layered else {})}


def _well_datum_record(datum_meta: dict | None) -> dict:
    """CSV columns for ``--well-datum fit`` (none when off, so default CSVs keep their
    columns): the mode, the prior sd and the fitted datum's summary statistics."""
    if not datum_meta:
        return {}
    return {"well_datum": datum_meta["well_datum_mode"],
            "well_datum_sd": datum_meta["well_datum_sd"],
            **{f"well_datum_{k}": v for k, v in datum_meta["well_datum_stats"].items()}}


def _run_policy_gate(args) -> None:
    """``--policy-response``: the scorecard for this run, with the reference column."""
    from .policy_gate import main as policy_main

    theta = os.path.join(args.out, "stage3_theta.json")
    pg = ["--theta", theta, "--out", os.path.join(args.out, "scorecard.json")]
    if args.device:
        pg += ["--device", args.device]
    if args.holdout_months > 0:
        pg += ["--temporal", os.path.join(args.out, "stage3_temporal.csv")]
    policy_main(pg)


if __name__ == "__main__":
    main()
