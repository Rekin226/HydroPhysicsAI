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
from .calibrate_flow import set_compile_matvec
from .compaction import VEPColumn
from .forward import (
    N_LAYERS,
    attach_sw_recharge,
    blended_column,
    build_model,
    load_members,
    rollout,
    sw_hist,
)
from .inputs import input_options, load_twin_inputs
from .zones import N_ZONES, fan_zones, zone_blend_weights

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


def column_constraints() -> dict:
    """The constraints ``_clamp_column`` applies, for the column JSON (None = default)."""
    return {"tau_min_days": TAU_MIN_DAYS, "ske_min": SKE_MIN, "ske_skv_max": SKE_SKV_MAX}


def _clamp_column(col: VEPColumn, T: int) -> None:
    """Stage-2's bounds, except that the viscous time constant may be allowed past the
    record (``TAU_MAX_YEARS``): the mid-zone column sits on the record-length ceiling
    when fitted to leveling, and a residual-clay time constant of decades is what the
    literature reports for this fan (Lees et al. 2022).

    Opt-in constraints (all None by default, which is the historical clamp exactly):
    ``TAU_MIN_DAYS`` raises the tau floor from 1 d, ``SKE_MIN`` raises the Ske floor from
    1e-6, and ``SKE_SKV_MAX`` caps Ske/Skv. The round-2 proximal column had tau 24 d and
    Ske at its 1e-6 floor. That is an instant, purely inelastic response, which a
    monthly model cannot tell apart from elastic storage."""
    tau_max = tau_ceiling_days(T, col.dt_days)
    tau_min = 1.0 if TAU_MIN_DAYS is None else max(1.0, float(TAU_MIN_DAYS))
    if tau_min >= tau_max:
        raise ValueError(f"--tau-min-days {tau_min:g} is not below the tau ceiling "
                         f"{tau_max:g} d")
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
    with torch.no_grad():
        col.log_ske.clamp_(min=ske_lo, max=math.log(SKE_MAX))
        col.log_skv.clamp_(min=skv_lo, max=math.log(1e0))
        if SKE_SKV_MAX is not None:
            col.log_ske.copy_(torch.minimum(col.log_ske,
                                            col.log_skv + math.log(float(SKE_SKV_MAX))))
        col.log_tau.clamp_(min=math.log(tau_min), max=math.log(tau_max))


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


def _make(config: str, device) -> nn.Module:
    if config == "shared":
        return VEPColumn(n_sites=1, dt_days=30.0, device=device).to(device)
    if config == "weighted":
        return _WeightedColumn(N_LAYERS, device=device).to(device)
    if config == "zonal":
        return nn.ModuleList([VEPColumn(n_sites=1, dt_days=30.0, device=device)
                              for _ in range(N_ZONES)]).to(device)
    raise ValueError(config)


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


def _rezero(pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Subtract each row's prediction at its first observed month, so predictions and
    leveling series measure change from the same instant."""
    first = torch.argmax((mask > 0).to(torch.int64), dim=1)
    return pred - pred.gather(1, first[:, None])


def kfold_sites(config: str, H, OBS, M, Z, epochs, lr, device, n_folds: int = 5,
                seed: int = 0, rezero: bool = True) -> float:
    n = H.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, n_folds)
    preds = torch.zeros_like(OBS)
    for held in folds:
        keep = np.setdiff1d(np.arange(n), held)
        model = _make(config, device)
        _fit(model, H[keep], OBS[keep], M[keep], Z[keep], epochs, lr, rezero=rezero)
        with torch.no_grad():
            p = _predict(model, H[held], Z[held])
            preds[held] = _rezero(p, M[held]) if rezero else p
    return _r2(preds.cpu().numpy(), OBS.cpu().numpy(), M.cpu().numpy().astype(bool))


def _r2(p: np.ndarray, o: np.ndarray, m: np.ndarray) -> float:
    p, o = p[m], o[m]
    return 1.0 - float(((o - p) ** 2).sum()) / max(float(((o - o.mean()) ** 2).sum()), 1e-12)


def loso(config: str, H, OBS, M, Z, epochs, lr, device) -> float:
    preds = torch.zeros_like(OBS)                 # on OBS's device; numpy only at the end
    n = H.shape[0]
    for held in range(n):
        keep = [i for i in range(n) if i != held]
        model = _make(config, device)
        _fit(model, H[keep], OBS[keep], M[keep], Z[keep] if Z is not None else None, epochs, lr)
        with torch.no_grad():
            preds[held] = _predict(model, H[held:held + 1], Z[held:held + 1])[0]
    return _r2(preds.cpu().numpy(), OBS.cpu().numpy(), M.cpu().numpy().astype(bool))


def column_json(model: nn.Module) -> dict:
    def one(c: VEPColumn) -> dict:
        return {k: float(getattr(c, k).detach().cpu().reshape(-1)[0])
                for k in ("log_ske", "log_skv", "log_tau", "h_pc0")}
    if isinstance(model, _WeightedColumn):
        return {**one(model.col), "layer_weights": model.weights().tolist()}
    if isinstance(model, nn.ModuleList):
        return {"zonal": [one(c) for c in model]}
    return one(model)


def tau_ceiling_days(T: int, dt_days: float = 30.0) -> float:
    """The viscous time-constant ceiling ``_clamp_column`` applies, in days."""
    return (TAU_MAX_YEARS * 365.25 if TAU_MAX_YEARS is not None else float(T) * dt_days)


def tau_at_ceiling(params: dict, T: int, rtol: float = 1e-3) -> list[bool]:
    """Per column (one, or one per zone): does ``tau`` sit on its ceiling? A column that
    does has a creep time constant the record cannot identify (STATE §3.2) -- its
    decadal creep is a modelling choice, which ``twin.forward`` can carry as a rheology
    axis by running columns fitted under different ceilings side by side."""
    ceil = math.log(tau_ceiling_days(T))
    cols = params.get("zonal", [params])
    return [bool(abs(float(c["log_tau"]) - ceil) < rtol * max(abs(ceil), 1.0)) for c in cols]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="refit the VEP column on the flow model's heads")
    ap.add_argument("--theta", required=True, help="stage3_theta.json of the gated model")
    ap.add_argument("--configs", default="shared,weighted,zonal")
    ap.add_argument("--target", choices=("rings", "leveling"), default="rings",
                    help="calibration target: the 14 MLCW rings (Stage-2 convention) or "
                         "the ~800 leveling benchmarks with site-grouped 5-fold scoring")
    ap.add_argument("--n-folds", type=int, default=5)
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
    TAU_MAX_YEARS = args.tau_max_years
    HPC0_GUARD_DAYS = args.hpc0_guard_days
    TAU_MIN_DAYS, SKE_MIN, SKE_SKV_MAX = args.tau_min_days, args.ske_min, args.ske_skv_max
    # fail on an inconsistent set before the (long) hindcast rather than in the first step
    _clamp_column(VEPColumn(n_sites=1, dt_days=30.0), 132)
    device = pick_device(args.device)
    cfg = Config(data_dir=args.data) if args.data else Config()
    ddir = str(cfg.data_dir)
    member = load_members([args.theta])[0]
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
    heads = rollout(model, scalars, inp.initial_heads(0), E, inp.recharge_field[:, 1:],
                    inp.ground_elev, sw_field=sw_hist(inp)).cpu().numpy()  # (L, A, T)
    print(f"hindcast heads in {time.perf_counter() - t0:.1f}s", flush=True)
    if zone_of_cell is None:
        zone_of_cell = fan_zones(inp.grid.centroids())
    # per-cell zone rows: the zone id, or with --zone-blend-km the (A, N_ZONES) weights
    zone_rows = zone_of_cell
    if args.zone_blend_km and args.zone_blend_km > 0.0:
        from .calibrate_flow import _parse_zone_boundaries

        # the column is three-zone even over a flow model with the proximal split
        zb = _parse_zone_boundaries(member.meta.get("zone_boundaries", "205,182"),
                                    allow_split=True)[:2]
        zone_rows = zone_blend_weights(inp.grid.centroids(), *zb, args.zone_blend_km).T
        print(f"column zones blended over {args.zone_blend_km:g} km across the "
              f"{zb[1]:g} km line", flush=True)
    ztype = torch.float32 if np.asarray(zone_rows).dtype.kind == "f" else torch.long

    Hr, OBSr, Mr, Zr, names_r = site_rows(ddir, inp, heads, zone_rows)
    print(f"MLCW sites on the grid: {len(names_r)}", flush=True)
    if args.target == "leveling":
        H, OBS, M, Z, names = leveling_rows(ddir, inp, heads, zone_rows)
        print(f"leveling benchmarks on the grid: {len(names)}", flush=True)
    else:
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
    for config in [c.strip() for c in args.configs.split(",") if c.strip()]:
        t0 = time.perf_counter()
        model_c = _make(config, device)
        loss = _fit(model_c, Ht, Ot, Mt, Zt, args.epochs, args.lr, rezero=rezero)
        with torch.no_grad():
            p_ins = _predict(model_c, Ht, Zt)
            p_ins = _rezero(p_ins, Mt) if rezero else p_ins
            ins = _r2(p_ins.cpu().numpy(), OBS, M.astype(bool))
        if args.target == "leveling":
            r2_loso = kfold_sites(config, Ht, Ot, Mt, Zt, args.epochs, args.lr, device,
                                  n_folds=args.n_folds, rezero=True)
            # the rings become the independent check
            Hrt = torch.tensor(Hr, dtype=torch.float32, device=device)
            Zrt = torch.tensor(Zr, dtype=ztype, device=device)
            with torch.no_grad():
                rings_r2 = _r2(_predict(model_c, Hrt, Zrt).cpu().numpy(), OBSr, Mr.astype(bool))
        else:
            r2_loso = loso(config, Ht, Ot, Mt, Zt, args.epochs, args.lr, device)
            rings_r2 = float("nan")
        with torch.no_grad():
            field = np.concatenate([_predict(model_c, heads_all[i:i + 512], zone_all[i:i + 512])
                                    .cpu().numpy() for i in range(0, heads_all.shape[0], 512)])
        lev = validate_against_leveling(ddir, inp.grid, field, inp.dates)
        params = column_json(model_c)
        at_ceil = tau_at_ceiling(params, heads.shape[-1])
        if any(at_ceil):
            print(f"{config:>9}: tau on its ceiling ({tau_ceiling_days(heads.shape[-1]):.0f} d) "
                  f"for column(s) {[i for i, a in enumerate(at_ceil) if a]} -- not "
                  "identifiable; run twin.forward with columns fitted under two ceilings "
                  "(--vep-json a.json,b.json) to carry the rheology spread", flush=True)
        with open(os.path.join(args.out, f"vep_{config}_{args.target}.json"), "w") as fh:
            json.dump({**params, "config": config, "driver": "flow-model heads",
                       "target": args.target, "tau_max_years": args.tau_max_years,
                       **({"zone_blend_km": float(args.zone_blend_km)}
                          if config == "zonal" and args.zone_blend_km else {}),
                       "hpc0_guard_days": args.hpc0_guard_days,
                       **column_constraints(),
                       "tau_ceiling_days": tau_ceiling_days(heads.shape[-1]),
                       "tau_at_ceiling": at_ceil,
                       "theta": args.theta, "loss": loss,
                       "r2_insample": ins, "r2_outoffold": r2_loso,
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
