"""Stage 4 in practice: recalibrate the compaction column on the gated flow model's heads.

    python -m hydrophysics.twin.calibrate_coupled --theta <stage3_theta.json> \\
        --out results/twin_runs/coupled

Stage 2 fitted the visco-elasto-plastic column against the MLCW compaction rings with the
column driven by the *observed* IDW head. The forward twin drives the same column with the
*flow model's* layer heads, which is a different signal: layer-resolved, smoother, and
carrying the model's own pumping response. Reusing Stage-2's parameters under a different
driver silently re-scales the rheology (``coupled.py`` says as much). This script does
the honest thing: it refits the column against the MLCW rings with the flow model's heads
as the driver, in three configurations, and scores every one on the leveling network the
column never sees.

Configurations (the ablation the design spec asked for):

- ``shared``   -- one global parameter set, layer-mean head (what the forward twin uses).
- ``weighted`` -- one global set plus learnable softmax weights over the four aquifers, so
                 the data say which layers drive compaction (``CoupledTwin.driver=weighted``).
- ``zonal``    -- one parameter set per fan zone (proximal/mid/distal), layer-mean head.
                 Over a ``--zone-county-split`` flow model (rung R1c) every zone has a
                 Changhua copy, so the column has six parameter sets and the Changhua
                 leveling benchmarks and rings get their own (the JSON records
                 ``zone_county_split`` and the county map's hash).
- ``banded``   -- (opt-in, 2026-09-29) one parameter set per band along the down-fan
                 (easting) axis, ``--n-bands`` bands mixed per cell by hat weights, with a
                 smoothness penalty ``--band-lambda`` on neighbouring bands. It targets the
                 182 km step: the zonal column's parameters jump on that line and the field
                 steps with them, which leveling does not show.

``--target leveling`` (2026-09-14) calibrates against the leveling network instead: ~800
benchmarks with 5-site-grouped folds, scored out of fold, with the rings as the
independent check. Motivated by the first run on the gated model, where every ring-fitted
configuration scored worse on leveling than the Stage-2 column (+0.036 vs +0.299): 14
rings are too few and too local to constrain a fan-wide field.

The column must be fitted on the SAME head trajectory the twin will later run. Crossing
them costs a lot (measured 2026-09-19 on the leveling network): free-fit heads with their
own column +0.552 and the physical model's heads with theirs +0.599, but each with the
other's column +0.264 and +0.395. Nudging the hindcast toward observations is the same
mistake in time rather than in parameters: it costs +0.599 -> +0.391.

Gates. In-sample against the rings is not the number; two out-of-sample numbers are:
leave-one-site-out over the MLCW rings (pooled R², same statistic as Stage 2) and the
independent leveling R² over ~800 benchmarks (``explorer3d.validate_against_leveling``).
The configuration that wins on leveling is written as ``vep_<name>.json`` and can be
handed to ``twin.forward --vep-json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..config import Config
from ..subsidence import load_mlcw_stations, mlcw_compaction
from ..train import pick_device
from .calibrate_flow import mask_month_flags, set_compile_matvec
from .compaction import (
    AQUITARD_BOUNDS,
    CREEP_MODES,
    CREEP_ZONE_SETS,
    VEPColumn,
    aquitard_bounds,
    column_param_sets,
    zone_creep,
    zone_key_matches,
)
from .forward import (
    N_LAYERS,
    attach_sw_recharge,
    blended_column,
    build_model,
    extra_hist,
    hindcast_with_nudging,
    load_members,
    member_datum,
    rollout,
    sw_hist,
)
from .inputs import input_options, load_twin_inputs
from .zone_county import column_zone_names, column_zone_weights
from .zones import (
    ALL_ZONE_NAMES,
    BAND_KINDS,
    N_ZONES,
    band_centres_km,
    band_weights,
    fan_zones,
    zone_names,
)

TAU_MAX_YEARS: float | None = None      # set by --tau-max-years; None = the record length
# --hpc0-guard-days (fix A1, 2026-09-23): h_pc0 <= 0 in any column whose tau is below this
HPC0_GUARD_DAYS: float | None = None
# Opt-in column constraints (round 3, 2026-09-23); None = Stage-2's bounds unchanged.
# --tau-min-days: floor on the viscous time constant (default 1 d)
TAU_MIN_DAYS: float | None = None
# --ske-min: floor on the elastic skeletal storage Ske (default 1e-6)
SKE_MIN: float | None = None
# --ske-skv-max: cap on Ske/Skv. Enforced by lowering Ske, after Skv has been raised to at
# least SKE_MIN / SKE_SKV_MAX so that both constraints can hold at once
SKE_SKV_MAX: float | None = None
SKE_MAX = 1e-1
# --ske-cap-from-theta KAPPA (opt-in, M1 2026-10-24): each column's Ske may not exceed
# KAPPA times the flow model's own storativity summed over its layers (``ske_caps``).
# SKE_CAP_KAPPA is the recorded factor; SKE_CAPS the caps in Ske units, one per column of
# the config being fitted (``_make`` hands column ``i`` the cap ``SKE_CAPS[i]``). None = no
# cap, which is the historical clamp exactly.
SKE_CAP_KAPPA: float | None = None
SKE_CAPS: list[float] | None = None
SKE_CAP_ZONES: list[str] | None = None
# --configs banded (opt-in, 2026-09-29): number of bands and the smoothness weight
N_BANDS: int = 7
BAND_LAMBDA: float = 0.0
BAND_KIND: str = "hat"
# --band-hpc0-guard (review 2026-09-29): "band" applies --hpc0-guard-days per band (the
# screened behaviour); "cell" also caps slow bands next to fast ones so that no hat cell
# has tau < D with h_pc0 > 0 (``_guard_hpc0_banded``)
BAND_HPC0_GUARD: str = "band"
# --creep (opt-in, 2026-10-11): "aquitard" adds compaction.aquitard_compaction's delayed
# drainage term to every column (shared and zonal configs); "vep" is the historical column
CREEP: str = "vep"
# Slow-component column (opt-in, E1 2026-10-28; every default is the historical column).
# --creep-zones: the zonal column zones that carry the aquitard term (compaction
# .CREEP_ZONE_SETS); the others stay pure VEP. "all" = every column, as before
CREEP_ZONES: str = "all"
# --aquitard-tau-min-years / --aquitard-ha0-min: raised floors of the aquitard term
# (compaction.aquitard_bounds); None = AQUITARD_BOUNDS
AQUITARD_TAU_MIN_YEARS: float | None = None
AQUITARD_HA0_MIN: float | None = None
# --tau-min-days-zone ZONE=DAYS: a per-zone floor on the VEP tau, on top of TAU_MIN_DAYS
# (a name also floors its Changhua copy); None = no per-zone floor
TAU_MIN_DAYS_ZONE: dict[str, float] | None = None
# --deterministic: torch deterministic algorithms, for a bit-identity rerun
DETERMINISTIC: bool = False


def column_aquitard_bounds() -> dict | None:
    """The aquitard bounds of the columns ``_make`` builds: None (``AQUITARD_BOUNDS``,
    the historical clamp) unless a floor flag is given."""
    if AQUITARD_TAU_MIN_YEARS is None and AQUITARD_HA0_MIN is None:
        return None
    return aquitard_bounds(AQUITARD_TAU_MIN_YEARS, AQUITARD_HA0_MIN)


def zone_tau_floors(names) -> list[float | None]:
    """``--tau-min-days-zone``: each column zone's own tau floor in days (None = only
    the global ``TAU_MIN_DAYS``). A zone matched by two keys takes the higher floor."""
    out: list[float | None] = []
    for n in names:
        v = [float(d) for k, d in (TAU_MIN_DAYS_ZONE or {}).items()
             if n is not None and zone_key_matches(k, n)]
        out.append(max(v) if v else None)
    return out


def parse_tau_min_days_zone(items: list[str] | None) -> dict[str, float] | None:
    """``--tau-min-days-zone distal=3650`` (repeatable) -> ``{"distal": 3650.0}``. A key is
    a column zone name (``distal``, ``mid_changhua``, ...); None/empty -> None."""
    if not items:
        return None
    out: dict[str, float] = {}
    for it in items:
        key, sep, val = str(it).partition("=")
        key = key.strip()
        if not sep or key not in ALL_ZONE_NAMES:
            raise ValueError(f"--tau-min-days-zone needs ZONE=DAYS with ZONE one of "
                             f"{ALL_ZONE_NAMES}, got {it!r}")
        if key in out:
            raise ValueError(f"--tau-min-days-zone gives zone {key!r} twice")
        days = float(val)
        if not (math.isfinite(days) and days >= 1.0):
            raise ValueError(f"--tau-min-days-zone {key}: the floor must be >= 1 d, got {val}")
        out[key] = days
    return out


def column_constraints() -> dict:
    """The constraints ``_clamp_column`` applies, for the column JSON (None = default).
    The Ske cap's keys appear only on a capped column, and the slow-component keys (E1)
    only when their flags are given, so a column without them has the historical JSON."""
    out = {"tau_min_days": TAU_MIN_DAYS, "ske_min": SKE_MIN, "ske_skv_max": SKE_SKV_MAX}
    if CREEP_ZONES != "all":
        out["creep_zones"] = CREEP_ZONES
    bounds = column_aquitard_bounds()
    if bounds is not None:
        out.update({"aquitard_tau_min_years": AQUITARD_TAU_MIN_YEARS,
                    "aquitard_ha0_min": AQUITARD_HA0_MIN,
                    "aquitard_bounds": {k: [float(lo), float(hi)]
                                        for k, (lo, hi) in bounds.items()}})
    if TAU_MIN_DAYS_ZONE:
        out["tau_min_days_zone"] = {k: float(v) for k, v in TAU_MIN_DAYS_ZONE.items()}
    if DETERMINISTIC:
        out["deterministic"] = True
    if SKE_CAP_KAPPA is not None:
        out.update({"ske_cap_kappa": float(SKE_CAP_KAPPA),
                    "ske_cap": None if SKE_CAPS is None else [float(c) for c in SKE_CAPS],
                    "ske_cap_zones": None if SKE_CAP_ZONES is None else list(SKE_CAP_ZONES),
                    "ske_cap_source": "kappa x flow theta storativity exp(log_S) summed over "
                                      "the flow model's layers, area-weighted over the "
                                      "column zone's cells"})
    return out


def flow_zone_storage(theta: dict, meta: dict | None = None,
                      n_layers: int = N_LAYERS) -> dict[str, float]:
    """``{flow zone name: sum over the flow model's layers of exp(log_S)}`` from a theta.

    A zonal theta carries ``log_S_<zone>`` (four values, or one for the merged-aquifer
    proximal zones, which ``calibrate_flow._expand_zonal`` gives to every layer, so it
    counts ``n_layers`` times). A homogeneous theta's ``log_S`` is returned under the key
    ``""`` (every zone). A ``--county-tie``d Changhua copy is its parent
    (``apply_county_ties``)."""
    from .zone_county import apply_county_ties

    th = apply_county_ties(theta, meta)

    def total(v) -> float:
        a = np.asarray(v, dtype="float64").reshape(-1)
        if a.size == 1:
            a = np.repeat(a, n_layers)
        if a.size != n_layers:
            raise ValueError(f"log_S has {a.size} values, expected 1 or {n_layers}")
        return float(np.exp(a).sum())

    if "log_S" in th:
        return {"": total(th["log_S"])}
    out = {k[len("log_S_"):]: total(v) for k, v in th.items() if k.startswith("log_S_")}
    if not out:
        raise ValueError("the theta carries no log_S / log_S_<zone> storativity")
    return out


def ske_caps(storage: dict[str, float], flow_names: tuple[str, ...],
             flow_w: np.ndarray, col_zone: np.ndarray, n_col: int,
             kappa: float) -> list[float]:
    """``--ske-cap-from-theta``: the Ske cap of each column, ``kappa`` times the
    area-weighted mean over the column zone's cells of each cell's flow storativity
    summed over layers (``flow_zone_storage``).

    ``flow_w`` is ``(A, n_flow_zones)``: one-hot flow zone ids, or the flow model's
    ``--zone-blend-km`` weights. Cells are equal-area, so a column zone that unites several
    flow zones (the proximal split's ``proximal`` and ``proximal_w``, or a county split
    collapsed by ``--column-zones 3``) gets their cell-count-weighted mean. ``col_zone``
    (A,) is the column zone of every cell (all 0 for a shared/weighted column)."""
    if not float(kappa) > 0.0:
        raise ValueError(f"--ske-cap-from-theta must be > 0, got {kappa}")
    if "" in storage:                                    # homogeneous flow model
        s = np.full(len(flow_names), storage[""])
    else:
        missing = [n for n in flow_names if n not in storage]
        if missing:
            raise ValueError(f"the theta has no log_S for flow zone(s) {missing}")
        s = np.array([storage[n] for n in flow_names], dtype="float64")
    w = np.asarray(flow_w, dtype="float64")
    if w.shape[1] != len(flow_names):
        raise ValueError(f"flow weights have {w.shape[1]} zones, names {len(flow_names)}")
    per_cell = w @ s / w.sum(axis=1)
    col_zone = np.asarray(col_zone).reshape(-1)
    caps = []
    for c in range(int(n_col)):
        sel = col_zone == c
        if not sel.any():
            raise ValueError(f"column zone {c} has no cell; cannot set its Ske cap")
        caps.append(float(kappa) * float(per_cell[sel].mean()))
    return caps


def theta_ske_caps(theta: dict, meta: dict, grid, col_zone: np.ndarray, n_col: int,
                   kappa: float) -> list[float]:
    """``ske_caps`` on a calibrated flow model: its theta's storativity, its flow zone
    map (``zone_county.flow_zone_map``; with ``zone_blend_km`` its blend weights) and the
    column's zone of every cell."""
    from .zone_county import flow_zone_map, flow_zone_weights

    storage = flow_zone_storage(theta, meta)
    if (meta or {}).get("param_mode", "zonal") != "zonal":
        names: tuple[str, ...] = ("",)
        w = np.ones((grid.n_active, 1))
    else:
        z, names = flow_zone_map(grid, meta)
        if float((meta or {}).get("zone_blend_km") or 0.0) > 0.0:
            w = np.asarray(flow_zone_weights(grid, meta, float(meta["zone_blend_km"]))).T
        else:
            w = np.eye(len(names))[np.asarray(z, dtype="int64")]
    return ske_caps(storage, names, w, col_zone, n_col, kappa)


def _clamp_column(col: VEPColumn, T: int) -> None:
    """Stage-2's bounds, except that the viscous time constant may be allowed past the
    record (``TAU_MAX_YEARS``): the mid-zone column sits on the record-length ceiling
    when fitted to leveling, and a residual-clay time constant of decades is what the
    literature reports for this fan (Lees et al. 2022).

    Opt-in constraints (all None by default, which is the historical clamp exactly):
    ``TAU_MIN_DAYS`` raises the tau floor from 1 d, ``SKE_MIN`` raises the Ske floor from
    1e-6, and ``SKE_SKV_MAX`` caps Ske/Skv. The round-2 proximal column had tau 24 d and
    Ske at its 1e-6 floor. That is an instant, purely inelastic response, which a
    monthly model cannot tell apart from elastic storage.

    Per column (E1, opt-in, set by ``_make``): ``col.tau_min_days`` (``--tau-min-days-zone``)
    raises that column's tau floor further, and ``col.aquitard_bounds`` are its aquitard
    bounds (``AQUITARD_BOUNDS`` unless ``--aquitard-tau-min-years``/``--aquitard-ha0-min``)."""
    tau_max = tau_ceiling_days(T, col.dt_days)
    tau_min = 1.0 if TAU_MIN_DAYS is None else max(1.0, float(TAU_MIN_DAYS))
    if tau_min >= tau_max:
        raise ValueError(f"--tau-min-days {tau_min:g} is not below the tau ceiling "
                         f"{tau_max:g} d")
    zone_floor = getattr(col, "tau_min_days", None)      # --tau-min-days-zone (``_make``)
    if zone_floor is not None:
        tau_min = max(tau_min, float(zone_floor))
        if tau_min >= tau_max:
            raise ValueError(f"--tau-min-days-zone {tau_min:g} d is not below the tau "
                             f"ceiling {tau_max:g} d (raise --tau-max-years)")
    ske_lo = math.log(1e-6 if SKE_MIN is None else float(SKE_MIN))
    if ske_lo >= math.log(SKE_MAX):
        raise ValueError(f"--ske-min {SKE_MIN:g} is not below the Ske ceiling {SKE_MAX:g}")
    skv_lo = math.log(1e-5)
    if SKE_SKV_MAX is not None:
        if not float(SKE_SKV_MAX) > 0.0:
            raise ValueError(f"--ske-skv-max must be > 0, got {SKE_SKV_MAX}")
        skv_lo = max(skv_lo, ske_lo - math.log(float(SKE_SKV_MAX)))
        if skv_lo > 0.0:
            raise ValueError("--ske-min / --ske-skv-max needs Skv > 1, above its ceiling")
    ske_hi = math.log(SKE_MAX)
    cap = getattr(col, "ske_cap", None)              # --ske-cap-from-theta (``_make``)
    if cap is not None:
        ske_hi = min(ske_hi, math.log(float(cap)))
        if ske_lo >= ske_hi:
            raise ValueError(f"the Ske cap {float(cap):g} from --ske-cap-from-theta is not "
                             f"above the Ske floor {math.exp(ske_lo):g}")
    with torch.no_grad():
        col.log_ske.clamp_(min=ske_lo, max=ske_hi)
        col.log_skv.clamp_(min=skv_lo, max=math.log(1e0))
        if SKE_SKV_MAX is not None:
            col.log_ske.copy_(torch.minimum(col.log_ske,
                                            col.log_skv + math.log(float(SKE_SKV_MAX))))
        col.log_tau.clamp_(min=math.log(tau_min), max=math.log(tau_max))
        if col.creep == "aquitard":
            for k, (lo, hi) in getattr(col, "aquitard_bounds", AQUITARD_BOUNDS).items():
                getattr(col, k).clamp_(min=lo, max=hi)


class _WeightedColumn(nn.Module):
    """A shared column driven by a learnable softmax mix of the layer heads."""

    def __init__(self, n_layers: int, device=None):
        super().__init__()
        self.col = VEPColumn(n_sites=1, dt_days=30.0, device=device)
        self.logits = nn.Parameter(torch.zeros(n_layers, device=device))

    def forward(self, heads: torch.Tensor) -> torch.Tensor:        # heads (n, L, T)
        w = torch.softmax(self.logits, dim=0)
        return self.col((heads * w[None, :, None]).sum(dim=1))

    def weights(self) -> np.ndarray:
        return torch.softmax(self.logits.detach(), dim=0).cpu().numpy()


class _BandedColumns(nn.ModuleList):
    """One VEP column per down-fan band, mixed per cell by band weights (``_predict``
    passes ``zone`` as ``(n, n_bands)`` weights to ``blended_column``), with a smoothness
    penalty on neighbouring bands.

    ``penalty()`` is ``lam * sum_k sum_i (p_k[i+1] - p_k[i])^2`` over the four column
    parameters (``log_ske``, ``log_skv``, ``log_tau`` in log units, ``h_pc0`` in metres),
    added to the masked MSE (m^2) in ``_fit``. ``lam = 0`` is N free columns; a large
    ``lam`` drives the bands toward one shared column."""

    def __init__(self, n_bands: int, lam: float = 0.0, device=None, kind: str = "hat"):
        if int(n_bands) < 2:
            raise ValueError(f"a banded column needs at least 2 bands, got {n_bands}")
        if kind not in BAND_KINDS:
            raise ValueError(f"band kind must be one of {BAND_KINDS}, got {kind!r}")
        super().__init__([VEPColumn(n_sites=1, dt_days=30.0, device=device)
                          for _ in range(int(n_bands))])
        self.lam = float(lam)
        # 'hat' mixes neighbouring bands per cell, which the A1 guard has to see
        self.kind = kind

    def stacked(self, key: str) -> torch.Tensor:
        return torch.cat([getattr(c, key).reshape(1) for c in self])

    def penalty(self) -> torch.Tensor:
        keys = ("log_ske", "log_skv", "log_tau", "h_pc0")
        rough = sum(((self.stacked(k)[1:] - self.stacked(k)[:-1]) ** 2).sum() for k in keys)
        return self.lam * rough


def banded_from_zonal(zonal: nn.Module, band_zone: np.ndarray, lam: float,
                      device=None, kind: str = "hat") -> _BandedColumns:
    """A banded column whose band ``i`` starts at the fitted zonal column of zone
    ``band_zone[i]`` (``--band-init zonal``): the fit then starts in the zonal column's
    basin and the smoothness penalty only has to remove the jumps between zones."""
    model = _BandedColumns(len(band_zone), lam, device=device, kind=kind).to(device)
    with torch.no_grad():
        for c, z in zip(model, band_zone, strict=True):
            for k in ("log_ske", "log_skv", "log_tau", "h_pc0"):
                getattr(c, k).copy_(getattr(zonal[int(z)], k).detach())
    return model


def _fit(model: nn.Module, heads: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor,
         zone: torch.Tensor | None, epochs: int, lr: float, rezero: bool = False) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    T = heads.shape[-1]
    loss = torch.tensor(float("nan"))
    for _ in range(epochs):
        opt.zero_grad()
        pred = _predict(model, heads, zone)
        if rezero:
            pred = _rezero(pred, mask)
        loss = (((pred - obs) ** 2) * mask).sum() / mask.sum().clamp(min=1)
        if isinstance(model, _BandedColumns) and model.lam > 0.0:
            (loss + model.penalty()).backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        cols = [model.col] if isinstance(model, _WeightedColumn) else (
            list(model) if isinstance(model, nn.ModuleList) else [model])
        for c in cols:
            _clamp_column(c, T)
            with torch.no_grad():
                rng = float(heads.max() - heads.min())
                c.h_pc0.clamp_(min=-rng, max=rng)
                _guard_hpc0(c)
        if isinstance(model, _BandedColumns):
            with torch.no_grad():
                _guard_hpc0_banded(model)
    return float(loss.detach())


def _guard_hpc0(c: VEPColumn) -> None:
    """``--hpc0-guard-days D``: no positive ``h_pc0`` in a column with ``tau < D``.

    A positive offset is an instantaneous virgin load at t=1. With a short tau it is
    released before the first leveling survey, and ``_rezero`` then hides it from the
    loss. Adam still takes full steps along that flat direction, so the offset wanders
    until it reaches its clamp. That is how the proximal column reached +4.7 m, about
    121 cm of unseen start-up per cell (A1). A long-tau column releases its offset as
    creep that leveling does see, so D should be well below the first-survey gap times
    a few. 365 d is the recommended value."""
    if HPC0_GUARD_DAYS is None:
        return
    fast = torch.exp(c.log_tau) < float(HPC0_GUARD_DAYS)
    c.h_pc0.copy_(torch.where(fast, torch.clamp(c.h_pc0, max=0.0), c.h_pc0))


def _guard_hpc0_banded(model: _BandedColumns) -> None:
    """``--hpc0-guard-days D`` for a hat-banded column, per *cell* (review, 2026-09-29).

    ``_guard_hpc0`` holds per band, but a hat cell mixes two neighbouring bands (``log_tau``
    linearly in log space, ``h_pc0`` linearly), so between a slow band ``i`` with
    ``h_pc0 > 0`` and a fast neighbour ``j`` (``tau_j < D``, ``h_j <= 0``) there are cells
    with ``tau < D`` and ``h_pc0 > 0``: the A1 start-up the guard exists to remove. With
    ``f_D = ln(tau_i / D) / ln(tau_i / tau_j)`` (the mix at which ``tau`` reaches ``D``) no
    such cell exists iff ``h_i (1 - f_D) <= -f_D h_j``, so ``h_i`` is capped at
    ``max(0, -f_D h_j / (1 - f_D))``. Pairs of two slow or two fast bands need nothing.
    Opt-in (``--band-hpc0-guard cell``): on the datum model at lambda 1e-5 it moves the
    fit into the all-short-tau basin (every band tau 25-52 d, rings -3.1).
    A step-banded column does not mix bands; ``_guard_hpc0`` is already per cell there."""
    if HPC0_GUARD_DAYS is None or model.kind != "hat" or BAND_HPC0_GUARD != "cell":
        return
    d = float(HPC0_GUARD_DAYS)
    tau = [float(torch.exp(c.log_tau).reshape(-1)[0]) for c in model]
    h = [float(c.h_pc0.reshape(-1)[0]) for c in model]
    for i, c in enumerate(model):
        if tau[i] < d or h[i] <= 0.0:
            continue
        cap = h[i]
        for j in (i - 1, i + 1):
            if 0 <= j < len(model) and tau[j] < d:
                f_d = math.log(tau[i] / d) / math.log(tau[i] / tau[j])
                cap = min(cap, max(0.0, -f_d * min(h[j], 0.0) / (1.0 - f_d)))
        if cap < h[i]:
            c.h_pc0.clamp_(max=cap)


def _predict(model: nn.Module, heads: torch.Tensor, zone: torch.Tensor | None) -> torch.Tensor:
    """``heads`` is (n, L, T); returns (n, T) subsidence."""
    if isinstance(model, _WeightedColumn):
        return model(heads)
    drv = heads.mean(dim=1)
    if isinstance(model, nn.ModuleList) and zone is not None and zone.is_floating_point():
        # --zone-blend-km: ``zone`` is (n, N_ZONES) weights; each site mixes the zones
        return blended_column(model, zone.T, drv)
    if isinstance(model, nn.ModuleList):                      # zonal: one column per zone
        out = torch.zeros_like(drv)
        for z, col in enumerate(model):
            sel = zone == z
            if sel.any():
                out[sel] = col(drv[sel])
        return out
    return model(drv)


def _make(config: str, device, n_zones: int = N_ZONES) -> nn.Module:
    """A fresh column of ``config``; ``n_zones`` is the zonal config's number of parameter
    sets (3, or 6 over a ``--zone-county-split`` flow model).

    E1 (opt-in): a zonal column gives the aquitard term only to the ``CREEP_ZONES`` zones,
    and each column its ``--tau-min-days-zone`` floor (``_floored``); the zone names are
    ``zones.zone_names(n_zones)``. The aquitard floors apply to shared and zonal."""
    if CREEP != "vep" and config not in ("shared", "zonal"):
        raise ValueError(f"--creep {CREEP} is implemented for the shared and zonal configs only")
    per_zone = CREEP_ZONES != "all" or bool(TAU_MIN_DAYS_ZONE)
    if per_zone and config != "zonal":
        raise ValueError("--creep-zones and --tau-min-days-zone apply to the zonal config "
                         f"only, not to {config}")
    if CREEP_ZONES != "all" and CREEP != "aquitard":
        raise ValueError(f"--creep-zones {CREEP_ZONES} needs --creep aquitard")
    aq = column_aquitard_bounds()
    if config == "shared":
        return _capped([VEPColumn(n_sites=1, dt_days=30.0, device=device, creep=CREEP,
                                  aquitard_bounds=aq).to(device)])[0]
    if config == "weighted":
        model = _WeightedColumn(N_LAYERS, device=device).to(device)
        _capped([model.col])
        return model
    if config == "zonal":
        names = zone_names(int(n_zones)) if per_zone else (None,) * int(n_zones)
        creeps = zone_creep(CREEP, CREEP_ZONES, names)
        return nn.ModuleList(_floored(_capped([
            VEPColumn(n_sites=1, dt_days=30.0, device=device, creep=c,
                      aquitard_bounds=aq if c == "aquitard" else None)
            for c in creeps]), names)).to(device)
    if config == "banded":
        if SKE_CAPS is not None:
            raise ValueError("--ske-cap-from-theta applies to the shared, weighted and zonal "
                             "configs, not to banded")
        return _BandedColumns(N_BANDS, BAND_LAMBDA, device=device, kind=BAND_KIND).to(device)
    raise ValueError(config)


def _floored(cols: list[VEPColumn], names) -> list[VEPColumn]:
    """``--tau-min-days-zone``: give each column its zone's tau floor (read by
    ``_clamp_column``) and start it no lower than that floor. No-op without the flag."""
    if not TAU_MIN_DAYS_ZONE:
        return cols
    unmatched = [k for k in TAU_MIN_DAYS_ZONE
                 if not any(zone_key_matches(k, n) for n in names)]
    if unmatched:
        raise ValueError(f"--tau-min-days-zone names zone(s) {unmatched} that the column "
                         f"{tuple(names)} does not have")
    for c, f in zip(cols, zone_tau_floors(names), strict=True):
        if f is None:
            continue
        c.tau_min_days = float(f)
        with torch.no_grad():
            c.log_tau.clamp_(min=math.log(float(f)))
    return cols


def _capped(cols: list[VEPColumn]) -> list[VEPColumn]:
    """``--ske-cap-from-theta``: give column ``i`` the cap ``SKE_CAPS[i]`` (read by
    ``_clamp_column``) and start it no higher than that cap. No-op without a cap."""
    if SKE_CAPS is None:
        return cols
    if len(SKE_CAPS) != len(cols):
        raise ValueError(f"{len(SKE_CAPS)} Ske caps for {len(cols)} column(s)")
    for c, cap in zip(cols, SKE_CAPS, strict=True):
        c.ske_cap = float(cap)
        with torch.no_grad():
            c.log_ske.clamp_(max=math.log(float(cap)))
    return cols


def site_rows(ddir: str, inp, heads: np.ndarray, zone_of_cell: np.ndarray | None):
    """MLCW rings -> (heads (n, L, T), obs (n, T), mask (n, T), zone (n,), names)."""
    stations = load_mlcw_stations(os.path.join(ddir, "mlcw_stations.csv"))
    comp = mlcw_compaction(ddir)
    cent = inp.grid.centroids()
    H, OBS, M, Z, names = [], [], [], [], []
    for _, r in stations.iterrows():
        if r["sub_id"] not in comp:
            continue
        d2 = ((cent - np.array([r["x"], r["y"]])) ** 2).sum(1)
        cell = int(np.argmin(d2))
        if math.sqrt(d2[cell]) > inp.grid.dx:
            continue
        c = comp[r["sub_id"]].reindex(inp.dates, method="nearest",
                                      tolerance=pd.Timedelta("45D"))
        ok = c.notna().to_numpy()
        if ok.sum() < 24:
            continue
        H.append(heads[:, cell, :])
        OBS.append(np.nan_to_num(c.to_numpy(dtype="float64")))
        M.append(ok)
        Z.append(zone_of_cell[cell] if zone_of_cell is not None else 0)
        names.append(str(r["sub_id"]))
    if not H:
        raise SystemExit("no MLCW site fell on the grid with >= 24 samples")
    return (np.stack(H), np.stack(OBS), np.stack(M), np.array(Z), names)


def leveling_rows(ddir: str, inp, heads: np.ndarray, zone_of_cell: np.ndarray | None,
                  min_obs: int = 5, max_rate: float | None = 0.5):
    """Leveling benchmarks -> (heads (n, L, T), obs (n, T), mask (n, T), zone, names).

    Each benchmark's cumulative subsidence (re-zeroed to its first survey) is placed on
    the model's monthly axis at the nearest month; months without a survey are masked.
    The column's own re-zeroing at t=0 is matched by re-zeroing the prediction at each
    site's first surveyed month inside ``_predict_rezero``."""
    from .leveling import load_panel, site_subsidence, site_xy

    panel = load_panel(ddir)
    obs = site_subsidence(panel, str(inp.dates[0].date()), str(inp.dates[-1].date()),
                          min_obs=min_obs, max_rate=max_rate)
    xy = site_xy(panel)
    cent = inp.grid.centroids()
    T = len(inp.dates)
    H, OBS, M, Z, names = [], [], [], [], []
    for sid, series in obs.items():
        if sid not in xy:
            continue
        d2 = ((cent - np.array(xy[sid])) ** 2).sum(1)
        cell = int(np.argmin(d2))
        if math.sqrt(d2[cell]) > inp.grid.dx:
            continue
        o = np.zeros(T)
        m = np.zeros(T, dtype=bool)
        for t_obs, v in series.items():
            j = int(np.argmin(np.abs((inp.dates - t_obs).days)))
            if abs((inp.dates[j] - t_obs).days) <= 45 and np.isfinite(v):
                o[j], m[j] = float(v), True
        if m.sum() < 3:
            continue
        H.append(heads[:, cell, :])
        OBS.append(o)
        M.append(m)
        Z.append(zone_of_cell[cell] if zone_of_cell is not None else 0)
        names.append(str(sid))
    if not H:
        raise SystemExit("no leveling benchmark fell on the grid")
    return (np.stack(H), np.stack(OBS), np.stack(M), np.array(Z), names)


def withhold_rows(mask: np.ndarray, flags: np.ndarray | None,
                  min_left: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """--withhold-surveys: the (n, T) survey mask with the flagged months removed, and the
    rows that keep at least ``min_left`` surveys (``leveling_rows``' own minimum). With no
    flags the mask and every row are returned unchanged."""
    mask = np.asarray(mask, dtype=bool)
    if flags is None:
        return mask, np.ones(mask.shape[0], dtype=bool)
    out = mask & ~np.asarray(flags, dtype=bool)[None, :]
    return out, out.sum(axis=1) >= min_left


def collapse_zones(zone_of_cell: np.ndarray) -> np.ndarray:
    """--column-zones 3 over a --zone-county-split flow model: the county copy of each
    column zone (ids N_ZONES..2 N_ZONES - 1) mapped back onto its zone, so Changhua and the
    rest of the fan share one three-zone column (the column of R0, R4 and R*)."""
    return np.asarray(zone_of_cell) % N_ZONES


def _rezero(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract each row's prediction at its first observed month, so predictions and
    leveling series measure change from the same instant."""
    first = torch.argmax((mask > 0).to(torch.int64), dim=1)
    return pred - pred.gather(1, first[:, None])


def fold_seed_record(seed: int) -> dict:
    """The column JSON's record of the leveling fold seed: ``fold_seed`` when it is not
    the historical 0 (every column written before 2026-10-19 used 0 and carries no key)."""
    return {"fold_seed": int(seed)} if seed else {}


def fold_site_tol_record(tol_m: float) -> dict:
    """The column JSON's record of ``--fold-site-tol-m``: only when it is positive (every
    column written before 2026-10-24 used the per-benchmark partition and has no key)."""
    return {"fold_site_tol_m": float(tol_m)} if tol_m and tol_m > 0.0 else {}


def leveling_fold_groups(xy: np.ndarray | None, tol_m: float) -> np.ndarray | None:
    """``--fold-site-tol-m``: the leveling folds' grouping. ``tol_m <= 0`` is None, the
    historical per-benchmark partition (``kfold_sites``' seeded permutation, unchanged);
    a positive ``tol_m`` gives one label per physical site, benchmarks chained within
    ``tol_m`` metres of each other (single linkage, ``calibrate_flow._site_labels_tol``,
    the meaning ``calibrate_flow --fold-site-tol-m`` gives its well folds), so that no two
    benchmarks closer than the tolerance land in different folds."""
    if not (tol_m and tol_m > 0.0):
        return None
    if xy is None:
        raise ValueError("--fold-site-tol-m needs the benchmarks' coordinates")
    from .calibrate_flow import _site_labels_tol

    return _site_labels_tol(np.asarray(xy, dtype="float64").reshape(-1, 2), float(tol_m))


def leveling_xy(ddir: str, names: list[str]) -> np.ndarray:
    """(n, 2) EPSG:3826 coordinates of the leveling benchmarks ``names`` (the fit's rows,
    in order), from the panel ``leveling_rows`` reads."""
    from .leveling import load_panel, site_xy

    xy = site_xy(load_panel(ddir))
    return np.array([xy[str(n)] for n in names], dtype="float64").reshape(-1, 2)


def site_folds(n: int, n_folds: int, seed: int = 0,
               groups: np.ndarray | None = None) -> list[np.ndarray]:
    """The leveling k-fold partition of ``n`` rows: with no ``groups`` the historical
    seeded permutation split into ``n_folds`` (bit-for-bit what every column before
    2026-10-24 used); with ``groups`` (``leveling_fold_groups``) a grouped split,
    ``calibrate_flow._kfold_indices``, every row of a group in one fold."""
    if groups is None:
        order = np.random.default_rng(seed).permutation(n)
        return np.array_split(order, n_folds)
    from .calibrate_flow import _kfold_indices

    return _kfold_indices(n, n_folds, seed=seed, groups=groups)


def kfold_sites(config: str, H, OBS, M, Z, epochs, lr, device, n_folds: int = 5,
                seed: int = 0, rezero: bool = True, make=None,
                n_zones: int = N_ZONES, return_preds: bool = False,
                groups: np.ndarray | None = None):
    """Site-grouped k-fold R2. ``make(keep)`` (optional) returns the fold's starting
    model from the kept sites only (the banded column's zonal warm start); by default
    each fold starts from ``_make(config, n_zones=n_zones)``. ``return_preds``: also the
    out-of-fold predictions, ``(r2, preds)``. ``groups`` (``--fold-site-tol-m``,
    ``leveling_fold_groups``): keep each physical site's benchmarks in one fold; None is
    the historical per-benchmark partition."""
    n = H.shape[0]
    folds = site_folds(n, n_folds, seed, groups)
    preds = torch.zeros_like(OBS)
    for held in folds:
        keep = np.setdiff1d(np.arange(n), held)
        model = _make(config, device, n_zones) if make is None else make(keep)
        _fit(model, H[keep], OBS[keep], M[keep], Z[keep], epochs, lr, rezero=rezero)
        with torch.no_grad():
            p = _predict(model, H[held], Z[held])
            preds[held] = _rezero(p, M[held]) if rezero else p
    r2 = _r2(preds.cpu().numpy(), OBS.cpu().numpy(), M.cpu().numpy().astype(bool))
    return (r2, preds) if return_preds else r2


def oof_by_zone(preds: np.ndarray, obs: np.ndarray, mask: np.ndarray, zone: np.ndarray,
                names: tuple[str, ...]) -> dict:
    """Out-of-fold leveling scores per column zone and per county (a ``--zone-county-split``
    report beside the pooled gate R2, which Yunlin's larger signal dominates): benchmarks,
    R2 and RMSE (m) of the masked entries. ``zone`` is each benchmark's zone id (the
    weights' argmax for blended zones)."""
    zone = np.asarray(zone)
    if zone.ndim == 2:
        zone = zone.argmax(axis=1)
    mask = np.asarray(mask).astype(bool)

    def score(sel: np.ndarray) -> dict:
        m = mask & sel[:, None]
        if not m.any():
            return {"benchmarks": int(sel.sum()), "r2": None, "rmse_m": None}
        return {"benchmarks": int(sel.sum()), "r2": _r2(preds, obs, m),
                "rmse_m": float(np.sqrt(((preds[m] - obs[m]) ** 2).mean()))}

    out = {name: score(zone == i) for i, name in enumerate(names)}
    ch = np.array([names[int(z)].endswith("_changhua") for z in zone], dtype=bool)
    out["county:changhua"] = score(ch)
    out["county:rest"] = score(~ch)
    return out


def county_split_ids(zone: np.ndarray, changhua: np.ndarray,
                     names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Zone ids and names of an UNSPLIT column re-labelled by county, for the report only:
    ``zone`` (or blend weights, by argmax) plus ``len(names)`` where ``changhua``, and the
    names of the split column (``mid`` -> ``mid_changhua``), so an unsplit model's
    out-of-fold leveling scores line up with a ``--zone-county-split`` model's."""
    zone = np.asarray(zone)
    if zone.ndim == 2:
        zone = zone.argmax(axis=1)
    ch = np.asarray(changhua, dtype=bool)
    if ch.shape != zone.shape:
        raise ValueError(f"county flags {ch.shape} do not match the zone ids {zone.shape}")
    n = len(names)
    return (zone.astype(np.int64) + n * ch.astype(np.int64),
            tuple(names) + tuple(f"{z}_changhua" for z in names))


def county_report_unsplit(ddir: str, inp, heads: np.ndarray, zone: np.ndarray,
                          names: tuple[str, ...], preds: np.ndarray, obs: np.ndarray,
                          mask: np.ndarray, keep: np.ndarray | None = None) -> dict | None:
    """Per-county out-of-fold leveling scores of a column fitted WITHOUT the county split
    (rung R1c's like-for-like baseline: the same seeded folds, scored per county and
    zone with the default county map). Report only: never changes the fit, and a missing
    county map skips the report instead of failing the run."""
    from .zone_county import DEFAULT_MAP, county_flags_for_grid

    try:
        flags, source = county_flags_for_grid(inp.grid, DEFAULT_MAP)
    except Exception as exc:  # noqa: BLE001 - a report must not fail a column fit
        print(f"    county report skipped: {exc}", flush=True)
        return None
    # each benchmark's cell: the zone map replaced by the cell index (as the banded
    # column does), rows in the same order as the fit's
    cells = np.asarray(leveling_rows(ddir, inp, heads,
                                     np.arange(inp.grid.n_active))[3]).astype(np.int64)
    if keep is not None:                   # --withhold-surveys dropped some rows
        cells = cells[keep]
    ids, split_names = county_split_ids(zone, np.asarray(flags, dtype=bool)[cells], names)
    rep = oof_by_zone(preds, obs, mask, ids, split_names)
    rep["county_map"] = source
    rep["report_only"] = "unsplit column scored per county; not a gate"
    return rep


def _r2(p: np.ndarray, o: np.ndarray, m: np.ndarray) -> float:
    p, o = p[m], o[m]
    return 1.0 - float(((o - p) ** 2).sum()) / max(float(((o - o.mean()) ** 2).sum()), 1e-12)


def loso(config: str, H, OBS, M, Z, epochs, lr, device, n_zones: int = N_ZONES) -> float:
    preds = torch.zeros_like(OBS)                 # on OBS's device; numpy only at the end
    n = H.shape[0]
    for held in range(n):
        keep = [i for i in range(n) if i != held]
        model = _make(config, device, n_zones)
        _fit(model, H[keep], OBS[keep], M[keep], Z[keep] if Z is not None else None, epochs, lr)
        with torch.no_grad():
            preds[held] = _predict(model, H[held:held + 1], Z[held:held + 1])[0]
    return _r2(preds.cpu().numpy(), OBS.cpu().numpy(), M.cpu().numpy().astype(bool))


def column_json(model: nn.Module) -> dict:
    def one(c: VEPColumn) -> dict:
        return {k: float(getattr(c, k).detach().cpu().reshape(-1)[0]) for k in c.param_keys()}
    if isinstance(model, _WeightedColumn):
        return {**one(model.col), "layer_weights": model.weights().tolist()}
    if isinstance(model, _BandedColumns):
        return {"banded": [one(c) for c in model], "band_lambda": model.lam}
    if isinstance(model, nn.ModuleList):
        return {"zonal": [one(c) for c in model]}
    return one(model)


def tau_ceiling_days(T: int, dt_days: float = 30.0) -> float:
    """The viscous time-constant ceiling ``_clamp_column`` applies, in days."""
    return (TAU_MAX_YEARS * 365.25 if TAU_MAX_YEARS is not None else float(T) * dt_days)


def tau_at_ceiling(params: dict, T: int, rtol: float = 1e-3) -> list[bool]:
    """Per column (one, one per zone or one per band): does ``tau`` sit on its ceiling?
    A column that does has a creep time constant the record cannot identify (STATE §3.2) -- its
    decadal creep is a modelling choice, which ``twin.forward`` can carry as a rheology
    axis by running columns fitted under different ceilings side by side."""
    ceil = math.log(tau_ceiling_days(T))
    cols = column_param_sets(params)
    return [bool(abs(float(c["log_tau"]) - ceil) < rtol * max(abs(ceil), 1.0)) for c in cols]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="refit the VEP column on the flow model's heads")
    ap.add_argument("--theta", required=True, help="stage3_theta.json of the gated model")
    ap.add_argument("--configs", default="shared,weighted,zonal")
    ap.add_argument("--target", choices=("rings", "leveling"), default="rings",
                    help="calibration target: the 14 MLCW rings (Stage-2 convention) or "
                         "the ~800 leveling benchmarks with site-grouped 5-fold scoring")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed of the leveling site folds (default 0, every run before "
                         "2026-10-19); a fresh-partition confirmation passes a new one")
    ap.add_argument("--tau-max-years", type=float, default=None,
                    help="ceiling on the viscous time constant (default: the record "
                         "length); the mid-zone column sits on that ceiling")
    ap.add_argument("--zone-blend-km", type=float, default=0.0,
                    help="zonal config: blend the mid/distal column parameters over a "
                         "logistic of this width (km) around the mid/distal line, not a "
                         "step. Fix A3: the sharp line makes a 2.5 cm/yr hindcast step "
                         "that the leveling rates do not show. The parameter count is "
                         "unchanged. Recorded as zone_blend_km in the JSON, which "
                         "twin.forward reads. 0 = sharp (default)")
    ap.add_argument("--hpc0-guard-days", type=float, default=None,
                    help="fix A1: keep h_pc0 <= 0 in every column whose tau is below this "
                         "many days (recommended 365). A positive offset in a fast column "
                         "is released before the first survey, where the re-zeroed loss "
                         "cannot see it. Default: unguarded")
    ap.add_argument("--tau-min-days", type=float, default=None,
                    help="floor on the column's viscous time constant, days (default 1). "
                         "A tau of a few weeks is an instant response at a monthly step, "
                         "which the fit can use in place of elastic storage")
    ap.add_argument("--ske-min", type=float, default=None,
                    help="floor on the elastic skeletal storage Ske (default 1e-6)")
    ap.add_argument("--ske-skv-max", type=float, default=None,
                    help="cap on Ske/Skv (default: none). Skv is first raised to at least "
                         "ske_min/ratio so that the floor and the cap can both hold")
    ap.add_argument("--ske-cap-from-theta", type=float, default=None, metavar="KAPPA",
                    help="M1 (opt-in): cap each column's elastic skeletal storage Ske at "
                         "KAPPA times the flow model's own storativity, exp(log_S) of the "
                         "--theta summed over its layers and area-weighted over the column "
                         "zone (proximal = proximal and proximal_w by cell count). Keeps the "
                         "column mass-consistent with the head calibration. Recorded as "
                         "ske_cap_kappa / ske_cap in the column JSON. Default: no cap")
    ap.add_argument("--fold-site-tol-m", type=float, default=0.0, metavar="M",
                    help="--target leveling: group the k-fold partition by physical site "
                         "(benchmarks within M m, single linkage, as calibrate_flow's "
                         "--fold-site-tol-m for wells) so that no two benchmarks closer than "
                         "M land in different folds; 0 (default) keeps the historical "
                         "per-benchmark partition")
    ap.add_argument("--n-bands", type=int, default=7,
                    help="banded config: number of column bands along the easting "
                         "(default 7, about 10 km apart over the fan)")
    ap.add_argument("--band-lambda", type=float, default=0.0,
                    help="banded config: weight of the smoothness penalty, the sum of "
                         "squared differences of each column parameter between "
                         "neighbouring bands, added to the masked MSE in m^2 (0 = none)")
    ap.add_argument("--band-kind", choices=BAND_KINDS, default="hat",
                    help="banded config: 'hat' interpolates the parameters linearly "
                         "between band centres (continuous, no line where they jump); "
                         "'step' gives each cell its nearest band")
    ap.add_argument("--band-init", choices=("default", "zonal"), default="default",
                    help="banded config: 'default' starts every band at the column's "
                         "default parameters; 'zonal' first fits the zonal column on the "
                         "same sites (per fold in the k-fold, so nothing held out leaks) "
                         "and starts each band at the zone of its centre. Leveling target "
                         "only; doubles the fitting time")
    ap.add_argument("--band-hpc0-guard", choices=("band", "cell"), default="band",
                    help="banded config, with --hpc0-guard-days D: 'band' (default) guards "
                         "each band; 'cell' also caps h_pc0 in a slow band next to a fast "
                         "one so that no hat-mixed cell has tau < D and h_pc0 > 0")
    ap.add_argument("--band-range-km", default=None,
                    help="banded config: 'lo,hi' easting of the first and last band "
                         "centre, km (default: the westmost and eastmost cells)")
    ap.add_argument("--creep", choices=CREEP_MODES, default="vep",
                    help="column rheology: 'vep' (historical) or 'aquitard', which adds the "
                         "delayed-drainage term (compaction.aquitard_compaction) whose rate "
                         "depends on the head level; shared and zonal configs. Recorded as "
                         "'creep' in the column JSON, which twin.forward reads")
    ap.add_argument("--creep-zones", choices=tuple(CREEP_ZONE_SETS), default="all",
                    help="E1 (opt-in), with --creep aquitard and --configs zonal: the column "
                         "zones that carry the aquitard term ('distal', 'mid+distal'; a name "
                         "also selects its Changhua copy); the others stay pure VEP. "
                         "Recorded as creep_zones in the JSON. Default 'all' (historical)")
    ap.add_argument("--aquitard-tau-min-years", type=float, default=None,
                    help="E1 (opt-in), with --creep aquitard: floor on the aquitard time "
                         "constant tau_a, years (default 1). A floor of a decade stops the "
                         "term from falling back to extra elastic storage")
    ap.add_argument("--aquitard-ha0-min", type=float, default=None,
                    help="E1 (opt-in), with --creep aquitard: floor on the aquitard's "
                         "initial excess head h_a0, m (default -30); 0 makes the term "
                         "residual drainage only. Both floors are recorded with the "
                         "effective aquitard_bounds in the JSON")
    ap.add_argument("--tau-min-days-zone", action="append", default=None,
                    metavar="ZONE=DAYS",
                    help="E1 (opt-in), --configs zonal: floor on one zone's VEP tau, days, "
                         "on top of --tau-min-days (repeatable; e.g. distal=3650 for a slow "
                         "distal column; a name also floors its Changhua copy). Recorded as "
                         "tau_min_days_zone in the JSON")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch deterministic algorithms (and CUBLAS workspace config), so "
                         "a rerun of the column fit can be compared bit for bit; recorded "
                         "as deterministic in the JSON. Default off (historical)")
    ap.add_argument("--column-zones", choices=("flow", "3"), default="flow",
                    help="zones of the zonal column: 'flow' (default) follows the theta "
                         "(six with --zone-county-split); '3' fits the three-zone column "
                         "even over a county-split flow model (2026-10-23: the six-zone "
                         "column is what made the R1c family uplift after 2022). Recorded "
                         "as column_zones_collapsed in the JSON, which twin.forward reads")
    ap.add_argument("--withhold-surveys", default=None, metavar="START:END",
                    help="YYYY-MM:YYYY-MM, inclusive: leveling surveys and ring samples "
                         "whose model month lies in the window never enter the loss, the "
                         "k-fold, the in-run leveling check or the ring check (a temporal "
                         "hold-out for a registered test; default: none withheld)")
    ap.add_argument("--hindcast-gain", type=float, default=0.0,
                    help="fit the column on a hindcast nudged toward the observed heads by "
                         "this gain every --hindcast-every months (twin.forward's "
                         "hindcast_with_nudging, 2026-10-28), so a forward run with the same "
                         "nudging drives it with the heads it was fitted on. 0 = the free "
                         "hindcast (default)")
    ap.add_argument("--hindcast-every", type=int, default=12)
    ap.add_argument("--update-method", choices=("innovation", "field"), default="innovation",
                    help="observation update of the nudged hindcast (twin.forward's)")
    ap.add_argument("--apex-hold", choices=("restart", "calibrated"), default="restart",
                    help="apex boundary head of the nudged hindcast (twin.forward's)")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--data", default=None)
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile-matvec", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    set_compile_matvec(args.compile_matvec)
    global TAU_MAX_YEARS, HPC0_GUARD_DAYS, TAU_MIN_DAYS, SKE_MIN, SKE_SKV_MAX
    global N_BANDS, BAND_LAMBDA, BAND_KIND, BAND_HPC0_GUARD, CREEP
    global SKE_CAP_KAPPA, SKE_CAPS, SKE_CAP_ZONES
    global CREEP_ZONES, AQUITARD_TAU_MIN_YEARS, AQUITARD_HA0_MIN, TAU_MIN_DAYS_ZONE
    global DETERMINISTIC
    CREEP = args.creep
    N_BANDS, BAND_LAMBDA = int(args.n_bands), float(args.band_lambda)
    BAND_KIND, BAND_HPC0_GUARD = args.band_kind, args.band_hpc0_guard
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    if args.fold_site_tol_m < 0.0:
        raise SystemExit("--fold-site-tol-m must be >= 0")
    if args.fold_site_tol_m > 0.0 and args.target != "leveling":
        raise SystemExit("--fold-site-tol-m groups the leveling folds; it needs --target "
                         "leveling (the rings use leave-one-site-out)")
    if args.ske_cap_from_theta is not None:
        if not args.ske_cap_from_theta > 0.0:
            raise SystemExit(f"--ske-cap-from-theta must be > 0, got {args.ske_cap_from_theta}")
        if "banded" in configs:
            raise SystemExit("--ske-cap-from-theta applies to the shared, weighted and zonal "
                             "configs, not to banded")
    SKE_CAP_KAPPA = args.ske_cap_from_theta
    SKE_CAPS = SKE_CAP_ZONES = None
    band_range = (None, None)
    if "banded" in configs:
        if N_BANDS < 2:
            raise SystemExit(f"--n-bands must be >= 2, got {N_BANDS}")
        if BAND_LAMBDA < 0.0:
            raise SystemExit(f"--band-lambda must be >= 0, got {BAND_LAMBDA}")
        if args.band_init == "zonal" and args.target != "leveling":
            raise SystemExit("--band-init zonal needs --target leveling")
        if args.band_range_km:
            band_range = tuple(float(v) for v in args.band_range_km.split(","))
            if len(band_range) != 2 or not band_range[1] > band_range[0]:
                raise SystemExit(f"--band-range-km needs 'lo,hi' with hi > lo, got "
                                 f"{args.band_range_km!r}")
    TAU_MAX_YEARS = args.tau_max_years
    HPC0_GUARD_DAYS = args.hpc0_guard_days
    TAU_MIN_DAYS, SKE_MIN, SKE_SKV_MAX = args.tau_min_days, args.ske_min, args.ske_skv_max
    # E1 slow-component flags (opt-in; all off is the historical column)
    CREEP_ZONES = args.creep_zones
    AQUITARD_TAU_MIN_YEARS, AQUITARD_HA0_MIN = args.aquitard_tau_min_years, args.aquitard_ha0_min
    try:
        TAU_MIN_DAYS_ZONE = parse_tau_min_days_zone(args.tau_min_days_zone)
        column_aquitard_bounds()                         # validates the floors
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if args.creep != "aquitard" and (CREEP_ZONES != "all" or AQUITARD_TAU_MIN_YEARS is not None
                                     or AQUITARD_HA0_MIN is not None):
        raise SystemExit("--creep-zones, --aquitard-tau-min-years and --aquitard-ha0-min need "
                         "--creep aquitard")
    if (CREEP_ZONES != "all" or TAU_MIN_DAYS_ZONE) and configs != ["zonal"]:
        raise SystemExit("--creep-zones and --tau-min-days-zone apply to --configs zonal only")
    if args.creep == "aquitard" and "zonal" in configs and args.zone_blend_km \
            and args.zone_blend_km > 0.0:
        # forward.blended_column mixes the four VEP keys only: it would drop the term
        raise SystemExit("--creep aquitard cannot be fitted with --zone-blend-km (the blended "
                         "zonal column carries the four VEP parameters only)")
    DETERMINISTIC = bool(args.deterministic)
    if DETERMINISTIC:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # fail on an inconsistent set before the (long) hindcast rather than in the first step
    _clamp_column(VEPColumn(n_sites=1, dt_days=30.0), 132)
    for f in (TAU_MIN_DAYS_ZONE or {}).values():
        probe = VEPColumn(n_sites=1, dt_days=30.0)
        probe.tau_min_days = float(f)
        try:
            _clamp_column(probe, 132)
        except ValueError as e:
            raise SystemExit(str(e)) from None
    device = pick_device(args.device)
    cfg = Config(data_dir=args.data) if args.data else Config()
    ddir = str(cfg.data_dir)
    member = load_members([args.theta])[0]
    county = bool(member.meta.get("zone_county_split"))
    if county and "banded" in configs:
        # bands vary with easting only; they would hand Changhua the rest of the fan's
        # column, which is what the county split exists to test
        raise SystemExit("--configs banded cannot honour the theta's --zone-county-split "
                         "(its bands follow easting only); use shared/weighted/zonal")
    inp = load_twin_inputs(dx=args.dx, meter_filter=member.meta.get("meter_filter", "none"),
                           cap_duty=float(member.meta.get("cap_duty", 1.0)),
                           **input_options(member.meta))
    model, scalars, zone_of_cell = build_model(inp.grid, member, device)
    eta_classes = member.meta.get("eta_classes")
    if eta_classes:
        E = torch.tensor(np.stack([inp.E_by_class[c] for c in eta_classes]),
                         dtype=torch.float64)[..., 1:]
    else:
        E = inp.E_total[:, 1:]
    t0 = time.perf_counter()
    attach_sw_recharge(inp, member.meta)
    nudged = args.hindcast_gain > 0.0 and args.hindcast_every > 0
    if nudged:
        if eta_classes or inp.sw_field is not None or inp.extra_q is not None:
            raise SystemExit("--hindcast-gain: the nudged column fit supports the plain "
                             "forcing only (no eta classes, canal or extra abstraction)")
        h0 = inp.initial_heads(0, n_layers=N_LAYERS)
        heads = hindcast_with_nudging(
            model, scalars, inp, h0, E, inp.recharge_field[:, 1:], args.hindcast_gain,
            args.hindcast_every, datum=member_datum(member, inp.sids),
            apex_from=h0 if args.apex_hold == "calibrated" else None,
            update_method=args.update_method).cpu().numpy()  # (L, A, T)
        print(f"hindcast nudged toward the observed heads: gain {args.hindcast_gain:g} "
              f"every {args.hindcast_every} month(s), {args.update_method} update, apex "
              f"{args.apex_hold}", flush=True)
    else:
        heads = rollout(model, scalars, inp.initial_heads(0), E, inp.recharge_field[:, 1:],
                        inp.ground_elev, sw_field=sw_hist(inp),
                        q_extra=extra_hist(inp)).cpu().numpy()  # (L, A, T)
    print(f"hindcast heads in {time.perf_counter() - t0:.1f}s", flush=True)
    if zone_of_cell is None:
        zone_of_cell = fan_zones(inp.grid.centroids())
    # the column's zones: three, or with the flow model's --zone-county-split the
    # Changhua copy of each (build_model returns that map, verified against the meta)
    col_names = column_zone_names(member.meta)
    collapsed = args.column_zones == "3" and county
    if collapsed:
        if "banded" in configs:
            raise SystemExit("--column-zones 3 applies to the shared/weighted/zonal configs")
        zone_of_cell = collapse_zones(zone_of_cell)
        col_names = column_zone_names({})
        county = False                     # the column is unsplit from here on
        print(f"--column-zones 3: one three-zone column {col_names} over the county-split "
              "flow model", flush=True)
    n_col_zones = len(col_names)
    if TAU_MIN_DAYS_ZONE or CREEP_ZONES != "all":
        # the zone names _make derives from the zone count must be the run's own
        if tuple(zone_names(n_col_zones)) != tuple(col_names):
            raise SystemExit(f"--creep-zones/--tau-min-days-zone: column zones {col_names} "
                             f"are not the standard {zone_names(n_col_zones)}")
        bad = [k for k in (TAU_MIN_DAYS_ZONE or {})
               if not any(zone_key_matches(k, n) for n in col_names)]
        if bad:
            raise SystemExit(f"--tau-min-days-zone names zone(s) {bad} that the column "
                             f"{col_names} does not have")
        print(f"E1 per-zone column: creep {zone_creep(CREEP, CREEP_ZONES, col_names)}; tau "
              f"floors (d) {zone_tau_floors(col_names)}", flush=True)
    if int(np.max(zone_of_cell)) >= n_col_zones:
        raise SystemExit(f"zone map ids up to {int(np.max(zone_of_cell))} exceed the column's "
                         f"{n_col_zones} zones {col_names}")
    if county:
        print(f"--zone-county-split theta: the zonal column has {n_col_zones} zones "
              f"{col_names}", flush=True)
    # per-cell zone rows: the zone id, or with --zone-blend-km the (A, n_zones) weights
    zone_rows = zone_of_cell
    if args.zone_blend_km and args.zone_blend_km > 0.0:
        # the column is three-zone even over a flow model with the proximal split (and
        # split by county with the county split)
        zone_meta = ({k: v for k, v in member.meta.items() if k != "zone_county_split"}
                     if collapsed else member.meta)
        zone_rows = column_zone_weights(inp.grid, zone_meta)(args.zone_blend_km).T
        print(f"column zones blended over {args.zone_blend_km:g} km across the "
              f"mid/distal line", flush=True)
    ztype = torch.float32 if np.asarray(zone_rows).dtype.kind == "f" else torch.long

    wflags = mask_month_flags(args.withhold_surveys, inp.dates)
    keep = None
    Hr, OBSr, Mr, Zr, names_r = site_rows(ddir, inp, heads, zone_rows)
    Mr, _ = withhold_rows(Mr, wflags)        # every ring keeps its row; only months go
    print(f"MLCW sites on the grid: {len(names_r)}", flush=True)
    if args.target == "leveling":
        H, OBS, M, Z, names = leveling_rows(ddir, inp, heads, zone_rows)
        M, keep = withhold_rows(M, wflags)
        if wflags is not None:
            H, OBS, M, Z = H[keep], OBS[keep], M[keep], Z[keep]
            names = [n for n, k in zip(names, keep, strict=True) if k]
            print(f"--withhold-surveys {args.withhold_surveys}: {int(wflags.sum())} months "
                  f"withheld; {int((~keep).sum())} benchmarks left with < 3 surveys dropped",
                  flush=True)
        print(f"leveling benchmarks on the grid: {len(names)}", flush=True)
        fold_groups = leveling_fold_groups(
            leveling_xy(ddir, names) if args.fold_site_tol_m > 0.0 else None,
            args.fold_site_tol_m)
        if fold_groups is not None:
            print(f"--fold-site-tol-m {args.fold_site_tol_m:g}: {len(names)} benchmarks in "
                  f"{len(np.unique(fold_groups))} physical sites", flush=True)
    else:
        fold_groups = None
        H, OBS, M, Z, names = Hr, OBSr, Mr, Zr, names_r
    rezero = args.target == "leveling"
    Ht = torch.tensor(H, dtype=torch.float32, device=device)
    Ot = torch.tensor(OBS, dtype=torch.float32, device=device)
    Mt = torch.tensor(M, dtype=torch.float32, device=device)
    Zt = torch.tensor(Z, dtype=ztype, device=device)

    from .explorer3d import validate_against_leveling

    os.makedirs(args.out, exist_ok=True)
    rows = []
    heads_all = torch.tensor(heads, dtype=torch.float32, device=device).permute(1, 0, 2)  # (A, L, T)
    zone_all = torch.tensor(zone_rows, dtype=ztype, device=device)
    by_config = {None: (Zt, Zr, zone_all, ztype)}
    band_info = {}
    if "banded" in configs:
        # the rows' own cells (a second pass with the cell index as the "zone"), then the
        # (n, n_bands) band weights of each row and of every cell
        cent = inp.grid.centroids()
        centres = band_centres_km(cent[:, 0] / 1000.0, N_BANDS, *band_range)
        bw = band_weights(cent, centres, args.band_kind).T                     # (A, N)
        cells = np.arange(cent.shape[0])
        cells_r = site_rows(ddir, inp, heads, cells)[3]
        cells_f = (leveling_rows(ddir, inp, heads, cells)[3]
                   if args.target == "leveling" else cells_r)
        if args.target == "leveling" and keep is not None:
            cells_f = np.asarray(cells_f)[keep]
        by_config["banded"] = (torch.tensor(bw[cells_f], dtype=torch.float32, device=device),
                               bw[cells_r],
                               torch.tensor(bw, dtype=torch.float32, device=device),
                               torch.float32)
        band_info = {"band_centres_km": [float(c) for c in centres],
                     "band_kind": args.band_kind, "band_axis": "easting",
                     "band_init": args.band_init,
                     **({"band_hpc0_guard": "cell"} if args.band_hpc0_guard == "cell"
                        else {})}
        if args.band_init == "zonal":
            from .calibrate_flow import _parse_zone_boundaries

            zb3 = _parse_zone_boundaries(member.meta.get("zone_boundaries", "205,182"),
                                         allow_split=True)[:2]
            band_zone = fan_zones(np.column_stack([centres * 1000.0, np.zeros_like(centres)]),
                                  *zb3)
            band_info["band_init_zones"] = [int(z) for z in band_zone]
            Z0 = by_config[None][0]

            def warm(keep=None):
                """The zonal column fitted on ``keep`` (all sites if None), as bands."""
                idx = slice(None) if keep is None else keep
                zon = _make("zonal", device)
                _fit(zon, Ht[idx], Ot[idx], Mt[idx], Z0[idx], args.epochs, args.lr,
                     rezero=rezero)
                return banded_from_zonal(zon, band_zone, BAND_LAMBDA, device, BAND_KIND)
        print(f"banded column: {N_BANDS} bands ({args.band_kind}) centred at "
              f"{np.round(centres, 1).tolist()} km, smoothness lambda {BAND_LAMBDA:g}",
              flush=True)
    caps_by_config = {}
    if SKE_CAP_KAPPA is not None:
        # M1: per column zone for zonal, one fan-wide cap for shared/weighted
        cz = np.asarray(zone_of_cell, dtype="int64")
        caps_by_config["zonal"] = (theta_ske_caps(member.theta, member.meta, inp.grid, cz,
                                                  n_col_zones, SKE_CAP_KAPPA),
                                   list(col_names))
        caps_by_config["shared"] = (theta_ske_caps(member.theta, member.meta, inp.grid,
                                                   np.zeros_like(cz), 1, SKE_CAP_KAPPA),
                                    ["fan"])
        caps_by_config["weighted"] = caps_by_config["shared"]
        for cfg_name in configs:
            caps, zn = caps_by_config[cfg_name]
            print(f"--ske-cap-from-theta {SKE_CAP_KAPPA:g} ({cfg_name}): Ske caps "
                  + ", ".join(f"{z} {c:.3g}" for z, c in zip(zn, caps, strict=True)),
                  flush=True)
    for config in configs:
        t0 = time.perf_counter()
        if SKE_CAP_KAPPA is not None:
            SKE_CAPS, SKE_CAP_ZONES = caps_by_config[config]
        Zt, Zr, zone_all, ztype = by_config.get(config, by_config[None])
        band_warm = config == "banded" and args.band_init == "zonal"
        model_c = warm() if band_warm else _make(config, device, n_col_zones)
        loss = _fit(model_c, Ht, Ot, Mt, Zt, args.epochs, args.lr, rezero=rezero)
        with torch.no_grad():
            p_ins = _predict(model_c, Ht, Zt)
            p_ins = _rezero(p_ins, Mt) if rezero else p_ins
            ins = _r2(p_ins.cpu().numpy(), OBS, M.astype(bool))
        by_zone = None
        if args.target == "leveling":
            r2_loso, oof = kfold_sites(config, Ht, Ot, Mt, Zt, args.epochs, args.lr, device,
                                       n_folds=args.n_folds, seed=args.seed, rezero=True,
                                       make=warm if band_warm else None,
                                       n_zones=n_col_zones, return_preds=True,
                                       groups=fold_groups)
            if county and config == "zonal":
                # reported beside the pooled gate R2, never a gate of its own; the rings
                # (independent check) cover mid_changhua only
                by_zone = oof_by_zone(oof.cpu().numpy(), OBS, M, Zt.cpu().numpy(),
                                      col_names)
            elif config == "zonal":
                # an unsplit column's like-for-like baseline for R1c: same seeded folds,
                # scored per county with the default county map (report only)
                by_zone = county_report_unsplit(ddir, inp, heads, Zt.cpu().numpy(), col_names,
                                                oof.cpu().numpy(), OBS, M, keep=keep)
            for zn, sc in (by_zone or {}).items():
                if isinstance(sc, dict):
                    print(f"    out-of-fold {zn:22s} n={sc['benchmarks']:4d}  R2 "
                          + (f"{sc['r2']:+.3f}  RMSE {100 * sc['rmse_m']:.1f} cm"
                             if sc["r2"] is not None else "n/a"), flush=True)
            # the rings become the independent check
            Hrt = torch.tensor(Hr, dtype=torch.float32, device=device)
            Zrt = torch.tensor(Zr, dtype=ztype, device=device)
            with torch.no_grad():
                rings_r2 = _r2(_predict(model_c, Hrt, Zrt).cpu().numpy(), OBSr, Mr.astype(bool))
        else:
            r2_loso = loso(config, Ht, Ot, Mt, Zt, args.epochs, args.lr, device,
                           n_zones=n_col_zones)
            rings_r2 = float("nan")
        with torch.no_grad():
            field = np.concatenate([_predict(model_c, heads_all[i:i + 512], zone_all[i:i + 512])
                                    .cpu().numpy() for i in range(0, heads_all.shape[0], 512)])
        lev = validate_against_leveling(ddir, inp.grid, field, inp.dates,
                                        exclude_months=wflags)
        params = column_json(model_c)
        at_ceil = tau_at_ceiling(params, heads.shape[-1])
        if any(at_ceil):
            print(f"{config:>9}: tau on its ceiling ({tau_ceiling_days(heads.shape[-1]):.0f} d) "
                  f"for column(s) {[i for i, a in enumerate(at_ceil) if a]} -- not "
                  "identifiable; run twin.forward with columns fitted under two ceilings "
                  "(--vep-json a.json,b.json) to carry the rheology spread", flush=True)
        with open(os.path.join(args.out, f"vep_{config}_{args.target}.json"), "w") as fh:
            json.dump({**params, "config": config, "creep": CREEP, "driver": "flow-model heads",
                       "target": args.target, "tau_max_years": args.tau_max_years,
                       **({"zone_blend_km": float(args.zone_blend_km)}
                          if config == "zonal" and args.zone_blend_km else {}),
                       # rung R1c: the column's county split, checked by every consumer
                       **({"zone_county_split": True, "zone_names": list(col_names),
                           "zone_county_sha256": member.meta.get("zone_county_sha256"),
                           "zone_county_map": member.meta.get("zone_county_map")}
                          if config == "zonal" and county else {}),
                       **(band_info if config == "banded" else {}),
                       **({"column_zones_collapsed": True, "zone_names": list(col_names)}
                          if collapsed else {}),
                       **({"withhold_surveys": args.withhold_surveys}
                          if args.withhold_surveys else {}),
                       # opt-in 2026-10-28: the head field the column pairs with
                       **({"hindcast_gain": args.hindcast_gain,
                           "hindcast_every": args.hindcast_every,
                           "update_method": args.update_method,
                           "apex_hold": args.apex_hold} if nudged else {}),
                       "hpc0_guard_days": args.hpc0_guard_days,
                       **column_constraints(),
                       "tau_ceiling_days": tau_ceiling_days(heads.shape[-1]),
                       "tau_at_ceiling": at_ceil,
                       "theta": args.theta, "loss": loss,
                       "r2_insample": ins, "r2_outoffold": r2_loso,
                       **fold_seed_record(args.seed),
                       **fold_site_tol_record(args.fold_site_tol_m),
                       **({"r2_outoffold_by_zone": by_zone} if by_zone is not None else {}),
                       "rings_independent_r2": rings_r2, "leveling": lev}, fh, indent=1)
        rows.append({"config": config, "target": args.target,
                     "n_params": sum(p.numel() for p in model_c.parameters()),
                     "r2_insample": ins, "r2_outoffold": r2_loso, "rings_independent_r2": rings_r2,
                     "leveling_r2": lev["r2"], "leveling_bias_cm": lev["bias_cm"],
                     "leveling_rmse_cm": lev["rmse_cm"], "seconds": time.perf_counter() - t0})
        extra = (f"  weights {np.round(params['layer_weights'], 3).tolist()}"
                 if "layer_weights" in params else "")
        print(f"{config:>9}: {args.target} in-sample {ins:+.3f}  out-of-fold {r2_loso:+.3f}  "
              f"leveling(all, in-sample if target) {lev['r2']:+.3f} (bias {lev['bias_cm']:+.1f} "
              f"cm)  rings {rings_r2:+.3f}{extra}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "stage4_column.csv"), index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
