"""The forward twin: a pumping policy in, heads and subsidence out, past the end of the data.

    python -m hydrophysics.twin.forward --theta results/twin_runs/<run>/stage3_theta.json \\
        --theta results/twin_runs/<run>/stage3_fold_thetas.json \\
        --scenario "cut30:irrigation=0.7@2026-01" --scenario "retire_aqua:aquaculture=0" \\
        --horizon 120 --out results/twin_forward/run

What one run does, in order:

1. **Hindcast** the observed record (2012-2022) with the calibrated flow model, from the
   IDW initial head field, under the recorded electricity and recharge -- the same
   rollout the Stage-3 gate scored, so the run carries its own in-sample skill.
2. **Restart from observations.** At the forecast origin (the last observed month) the
   model state is nudged toward the IDW field of that month's observed heads,
   ``h = h_model + gain * (h_obs - h_model)`` per layer, in layers that have wells. This
   is the assimilation step the architecture names as its second layer: the forecast
   starts from what was measured, not from wherever the hindcast drifted to. ``gain=1``
   is a full re-initialisation; ``gain=0`` is a free run.
3. **Run forward** for ``horizon`` months under assumed forcing: month-of-year climatology
   of the recorded electricity per policy class and of the recharge, with the scenario's
   multipliers applied (``scenario.PumpingScenario``). Pumping still feeds back on head
   through the lift term.
4. **Compact.** The Stage-2 visco-elasto-plastic column, one shared parameter set, is
   driven by the layer-mean head over hindcast + forecast, so the preconsolidation state
   the forecast inherits is the one the record built up.
5. **Spread.** Every parameter set passed (the in-sample fit and each fold's fit) is a
   member; optionally each is run from ``--ic-members`` perturbed observation fields.
   The output carries the member mean and standard deviation per scenario.

Honesty clause, printed on every run: the pumping -> head map is only as good as the
Stage-3 verdict recorded in the theta file's metadata. This tool does not decide whether
the model is trustworthy; it makes the model's consequences visible so the gate can.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..config import Config
from ..train import pick_device
from .boundaries import fan_boundaries
from .calibrate_flow import (
    _expand_zonal,
    _parse_zone_boundaries,
    _r2,
    _rollout,
    set_compile_matvec,
    set_l_min,
)
from .compaction import VEPColumn
from .flow import FlowModel
from .inputs import TwinInputs, load_twin_inputs
from .scenario import BASELINE, CLASSES, PumpingScenario, climatology
from .zones import fan_zones

N_LAYERS = 4


# ---------------------------------------------------------------------------------------
# calibrated parameter sets
# ---------------------------------------------------------------------------------------
@dataclass
class Member:
    """One calibrated parameter set with the metadata needed to rebuild its model."""

    label: str
    theta: dict
    meta: dict


def load_members(paths: list[str]) -> list[Member]:
    """Read ``stage3_theta.json`` files and ``stage3_fold_thetas.json`` lists.

    A fold list carries no metadata of its own; it inherits the metadata of the last
    in-sample file seen before it, which is how ``calibrate_flow`` writes them side by
    side. Pass the in-sample file first.
    """
    members: list[Member] = []
    meta: dict | None = None
    for p in paths:
        if p.endswith(".csv"):
            # a recorded gate artefact (stage3_flow.csv): theta is a dict repr and the
            # verdict travels with it, which is exactly what the honesty clause prints
            obj = _members_from_gate_csv(p)
        else:
            with open(p) as fh:
                obj = json.load(fh)
        if isinstance(obj, dict) and "theta" in obj:
            meta = obj.get("meta", {})
            members.append(Member(label=os.path.basename(os.path.dirname(p)) or p,
                                  theta=obj["theta"], meta=meta))
        elif isinstance(obj, list):
            if meta is None:
                raise ValueError(f"{p}: a fold list needs an in-sample theta file before "
                                 "it on the command line to supply metadata")
            for f in obj:
                members.append(Member(label=f"fold{f['fold']}", theta=f["theta"], meta=meta))
        else:
            raise ValueError(f"{p}: not a theta file")
    if not members:
        raise ValueError("no parameter sets given")
    return members


def _members_from_gate_csv(path: str) -> dict:
    """``stage3_flow.csv`` row -> the same ``{"theta", "meta"}`` shape as stage3_theta.json.
    Runs recorded before 2026-09-11 have no ``boundaries`` column: they were closed."""
    import ast

    row = pd.read_csv(path).iloc[-1]
    theta = ast.literal_eval(str(row["theta"]))
    meta = {"param_mode": str(row["param_mode"]), "git_commit": str(row.get("git_commit", "")),
            "boundaries": str(row["boundaries"]) if "boundaries" in row else "none",
            "zone_boundaries": (f"{float(row['zone_proximal_km']):g},"
                                f"{float(row['zone_distal_km']):g}"
                                if str(row.get("zone_proximal_km", "")) not in ("", "nan")
                                else "205,182"),
            "dx": float(row["dx"]), "epochs": int(row["epochs"]),
            "r2_insample": float(row["r2_insample"]),
            "gate": {"r2_kfold": float(row["r2_kfold"]), "r2_idw": float(row["r2_idw"]),
                     "margin": float(row["r2_kfold"]) - float(row["r2_idw"]),
                     "verdict": "PASS" if float(row["r2_kfold"]) > float(row["r2_idw"])
                     else "FAIL", "n_folds": int(row["n_folds"])}}
    return {"theta": theta, "meta": meta}


def build_model(grid, member: Member, device) -> tuple[FlowModel, dict, np.ndarray | None]:
    """Rebuild the calibrated ``FlowModel`` -> ``(model, scalars, zone_of_cell)``.

    ``scalars`` holds the driver parameters that live outside the model (``log_eta``,
    ``log_head_extra``, ``recharge_frac_logit``) as float64 tensors on ``device``.
    """
    meta, th = member.meta, member.theta
    mode = meta.get("param_mode", "zonal")
    proximal_km = 205.0
    zone_of_cell = None
    if mode == "zonal":
        proximal_km, distal_km = _parse_zone_boundaries(meta.get("zone_boundaries", "205,182"))
        zone_of_cell = fan_zones(grid.centroids(), proximal_km=proximal_km, distal_km=distal_km)
    boundaries = None
    if meta.get("boundaries", "none") == "coast-apex":
        boundaries = fan_boundaries(grid, proximal_km=proximal_km)
    model = FlowModel(grid, n_layers=N_LAYERS, dt_days=30.0, device=device,
                      boundaries=boundaries)
    A = grid.n_active

    def t(v):
        return torch.tensor(np.asarray(v, dtype="float64").reshape(-1, 1),
                            dtype=torch.float64, device=device)

    with torch.no_grad():
        if mode == "zonal":
            zt = torch.tensor(zone_of_cell, dtype=torch.long, device=device)
            keys = ("log_T_proximal", "log_S_proximal", "log_T_mid", "log_S_mid",
                    "log_T_distal", "log_S_distal", "log_L_mid", "log_L_distal")
            theta_t = {k: t(th[k]) for k in keys if k in th}
            log_T, log_S, log_L = _expand_zonal(theta_t, zt, N_LAYERS)
            model.log_T.copy_(log_T)
            model.log_S.copy_(log_S)
            model.log_L.copy_(log_L)
        elif mode == "homogeneous":
            model.log_T.copy_(t(th["log_T"]).expand(-1, A))
            model.log_S.copy_(t(th["log_S"]).expand(-1, A))
            model.log_L.copy_(t(th["log_L"]).expand(-1, A))
        else:
            raise ValueError(f"cannot rebuild param_mode={mode!r}")
        if boundaries is not None:
            if "log_C_coast" not in th:
                raise ValueError("meta says boundaries=coast-apex but theta has no "
                                 "log_C_coast -- the file is from a closed-basin run")
            model.log_C_coast.copy_(t(th["log_C_coast"]))
            model.log_C_apex.copy_(t(th["log_C_apex"]))
    set_l_min(meta.get("l_min"))
    scalars = {}
    for k in ("log_eta", "log_head_extra", "recharge_frac_logit", "pump_split_logit",
              "return_frac_logit"):
        if k in th:
            scalars[k] = torch.tensor(np.asarray(th[k], dtype="float64"),
                                      dtype=torch.float64, device=device)
    return model, scalars, zone_of_cell


def rollout(model: FlowModel, scalars: dict, h0: torch.Tensor, E: torch.Tensor,
            recharge_field: torch.Tensor, ground_elev: torch.Tensor,
            pump_layer: int = 1, recharge_layer: int = 0) -> torch.Tensor:
    """Run the calibrated model over a forcing sequence -> heads ``(L, A, T+1)`` (float64,
    on the model's device). ``E`` and ``recharge_field`` are ``(A, T)``."""
    with torch.no_grad():
        model.set_apex_heads(h0)
        return _rollout(
            model, model.log_T, model.log_S, model.log_L, h0, E.shape[-1],
            recharge_field=recharge_field, recharge_scale=scalars.get("recharge_frac_logit"),
            recharge_layer=recharge_layer, E=E, log_eta=scalars.get("log_eta"),
            log_head_extra=scalars.get("log_head_extra"), ground_elev=ground_elev,
            pump_layer=pump_layer, log_C_coast=model.log_C_coast, log_C_apex=model.log_C_apex,
            pump_split_logit=scalars.get("pump_split_logit"),
            return_frac_logit=scalars.get("return_frac_logit"))


# ---------------------------------------------------------------------------------------
# scenarios and forcing
# ---------------------------------------------------------------------------------------
def parse_scenario(text: str) -> tuple[PumpingScenario, float]:
    """``"name:irrigation=0.7,aquaculture=0[,rain=0.9]@2026-01#proximal/mid"`` -> scenario.

    ``rain`` is not a pumping class; it scales the recharge climatology and comes back as
    the second element. Everything after ``#`` names the zones the policy applies to.
    """
    if ":" not in text:
        raise ValueError(f"scenario {text!r}: expected 'name:class=factor,...'")
    name, rest = text.split(":", 1)
    zones = None
    if "#" in rest:
        rest, z = rest.split("#", 1)
        zones = tuple(s.strip() for s in z.split("/") if s.strip())
    start = None
    if "@" in rest:
        rest, start = rest.split("@", 1)
        start = start.strip()
    factors: dict[str, float] = {}
    rain = 1.0
    for item in rest.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=")
        k, v = k.strip(), float(v)
        if k == "rain":
            rain = v
        elif k in CLASSES:
            factors[k] = v
        else:
            raise ValueError(f"scenario {name!r}: unknown class {k!r}; expected one of "
                             f"{CLASSES} or 'rain'")
    return PumpingScenario(name.strip(), factors=factors, zones=zones, start=start), rain


def future_forcing(inp: TwinInputs, scenario: PumpingScenario, horizon: int,
                   rain_scale: float = 1.0, zone_of_cell: np.ndarray | None = None,
                   eta_classes: list[str] | None = None
                   ) -> tuple[torch.Tensor, torch.Tensor, pd.DatetimeIndex]:
    """Climatology of the recorded forcing, under the policy -> ``(E, recharge, dates)``."""
    fut_by_class = {}
    fut_dates = None
    for k, E in inp.E_by_class.items():
        fut_by_class[k], fut_dates = climatology(E, inp.dates, horizon)
    if eta_classes:
        # apply the policy class by class so each keeps its own efficiency
        E_fut = np.stack([scenario.apply({c: fut_by_class[c]}, fut_dates,
                                         zone_of_cell=zone_of_cell) for c in eta_classes])
    else:
        E_fut = scenario.apply(fut_by_class, fut_dates, zone_of_cell=zone_of_cell)
    r_fut, _ = climatology(inp.recharge_field.numpy(), inp.dates, horizon)
    return (torch.tensor(E_fut, dtype=torch.float64),
            torch.tensor(r_fut * rain_scale, dtype=torch.float64), fut_dates)


def nudge_to_observations(inp: TwinInputs, h_model: torch.Tensor, month: int,
                          gain: float, noise: np.ndarray | None = None) -> torch.Tensor:
    """``h_model + gain * (h_obs - h_model)`` in every layer that has an observation in
    ``month``; other layers keep the model state. ``h_model`` is ``(L, A)``."""
    if gain <= 0.0:
        return h_model
    h = inp.obs_h[:, month].copy()
    if noise is not None:
        h = h + noise
    finite = np.isfinite(h)
    out = h_model.clone()
    field = inp.initial_heads(month, n_layers=h_model.shape[0],
                              well_mask=finite, noise=noise).to(h_model)
    for k in range(h_model.shape[0]):
        if (finite & (inp.obs_layer == k)).any():
            out[k] = h_model[k] + gain * (field[k] - h_model[k])
    return out


# ---------------------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------------------
class ZonalColumn(torch.nn.Module):
    """One VEP parameter set per fan zone (``calibrate_coupled --configs zonal``), applied
    cell by cell through ``zone_of_cell``. Same call signature as ``VEPColumn`` on the
    layer-mean head, so ``compaction`` does not care which it gets."""

    def __init__(self, params: list[dict], zone_of_cell: np.ndarray, device=None):
        super().__init__()
        self.cols = torch.nn.ModuleList()
        for p in params:
            c = VEPColumn(n_sites=1, dt_days=30.0, device=device)
            with torch.no_grad():
                for k in ("log_ske", "log_skv", "log_tau", "h_pc0"):
                    getattr(c, k).fill_(float(p[k]))
            self.cols.append(c)
        self.register_buffer("zone", torch.as_tensor(zone_of_cell, dtype=torch.long,
                                                     device=device))

    def forward(self, drv: torch.Tensor) -> torch.Tensor:          # drv (A, T)
        out = torch.zeros_like(drv)
        for z, c in enumerate(self.cols):
            sel = self.zone == z
            if sel.any():
                out[sel] = c(drv[sel])
        return out


def load_or_fit_vep(vep_json: str | None, ddir: str | None, hf, device,
                    epochs: int = 2000, zone_of_cell: np.ndarray | None = None
                    ) -> tuple[torch.nn.Module, dict]:
    """The compaction column: read it from ``vep_json`` if that exists (a shared
    Stage-2 set, or a per-zone set from ``calibrate_coupled``), else fit the shared
    Stage-2 column on the MLCW sites (``explorer3d.fit_shared_vep``) and write it there."""
    if vep_json and os.path.exists(vep_json):
        with open(vep_json) as fh:
            p = json.load(fh)
        if "zonal" in p:
            if zone_of_cell is None:
                raise ValueError("a zonal column needs zone_of_cell")
            return ZonalColumn(p["zonal"], zone_of_cell, device=device).to(device), p
        col = VEPColumn(n_sites=1, dt_days=30.0, device=device)
        with torch.no_grad():
            for k in ("log_ske", "log_skv", "log_tau", "h_pc0"):
                getattr(col, k).fill_(float(p[k]))
        return col.to(device), p
    if ddir is None:
        raise SystemExit("no --vep-json to load and no data dir to fit one from")
    from .explorer3d import fit_shared_vep

    col, info = fit_shared_vep(ddir, hf, epochs=epochs, device=device)
    p = {k: float(getattr(col, k).detach().cpu().reshape(-1)[0])
         for k in ("log_ske", "log_skv", "log_tau", "h_pc0")}
    p.update({"n_sites_fitted": info["n_sites_fitted"], "loss": info["loss"],
              "epochs": epochs})
    if vep_json:
        os.makedirs(os.path.dirname(vep_json) or ".", exist_ok=True)
        with open(vep_json, "w") as fh:
            json.dump(p, fh, indent=1)
    return col.to(device), p


def compaction(col: torch.nn.Module, heads: torch.Tensor) -> np.ndarray:
    """Layer-mean head ``(L, A, T)`` -> cumulative subsidence ``(A, T)`` in metres."""
    with torch.no_grad():
        drv = heads.mean(dim=0).to(dtype=torch.float32)
        return col(drv).cpu().numpy()


# ---------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------
def hindcast_with_nudging(model: FlowModel, scalars: dict, inp: TwinInputs, h0: torch.Tensor,
                          E: torch.Tensor, R: torch.Tensor, gain: float, every: int,
                          pump_layer: int = 1, recharge_layer: int = 0) -> torch.Tensor:
    """Hindcast in segments of ``every`` months, nudging the state toward that month's
    observed IDW field by ``gain`` at each segment end (sequential assimilation through
    the record). ``gain=0`` or ``every<=0`` is the plain hindcast."""
    T = E.shape[-1]
    if gain <= 0.0 or every <= 0:
        return rollout(model, scalars, h0, E, R, inp.ground_elev, pump_layer, recharge_layer)
    h = h0.to(dtype=torch.float64, device=model.log_T.device)
    out = [h[..., None]]
    t = 0
    while t < T:
        n = min(every, T - t)
        seg = rollout(model, scalars, h, E[..., t:t + n], R[:, t:t + n], inp.ground_elev,
                      pump_layer, recharge_layer)
        out.append(seg[..., 1:])
        t += n
        # month index t in the record (E is the record from month 1 on)
        h = nudge_to_observations(inp, seg[..., -1], t, gain)
        if t < T:
            out[-1] = torch.cat([seg[..., 1:-1], h[..., None]], dim=-1)
    return torch.cat(out, dim=-1)


def run(inp: TwinInputs, members: list[Member], scenarios: list[tuple[PumpingScenario, float]],
        horizon: int, gain: float, ic_members: int, ic_sigma: float, seed: int,
        col: VEPColumn, device, pump_layer: int = 1, recharge_layer: int = 0,
        log=print, hindcast_gain: float = 0.0, hindcast_every: int = 0) -> dict:
    T = len(inp.dates)
    origin = T - 1
    eta_classes = members[0].meta.get("eta_classes")
    if eta_classes:
        # per-class efficiencies: the forcing keeps its class axis in the calibrated order
        E_hist = torch.tensor(np.stack([inp.E_by_class[c] for c in eta_classes]),
                              dtype=torch.float64)[..., 1:]
    else:
        E_hist = inp.E_total[:, 1:]
    r_hist = inp.recharge_field[:, 1:]
    h0 = inp.initial_heads(0, n_layers=N_LAYERS)
    rng = np.random.default_rng(seed)
    ic_noises = [None] + [rng.normal(0.0, ic_sigma, size=inp.obs_h.shape[0])
                          for _ in range(max(ic_members, 0))]

    all_dates = inp.dates.append(pd.date_range(inp.dates[-1] + pd.offsets.MonthBegin(1),
                                               periods=horizon, freq="MS"))
    L, A = N_LAYERS, inp.grid.n_active
    n_scen = len(scenarios)
    n_mem = len(members) * len(ic_noises)
    heads_sum = np.zeros((n_scen, L, A, T + horizon), dtype="float64")
    heads_sq = np.zeros_like(heads_sum)
    subs_sum = np.zeros((n_scen, A, T + horizon), dtype="float64")
    subs_sq = np.zeros_like(subs_sum)
    fan_mean_rows = []
    hindcast_r2 = []

    obs_target = inp.obs_h_filled[:, 1:]
    for mem in members:
        t_m = time.perf_counter()
        model, scalars, zone_of_cell = build_model(inp.grid, mem, device)
        h_hist = hindcast_with_nudging(model, scalars, inp, h0, E_hist, r_hist,
                                       hindcast_gain, hindcast_every,
                                       pump_layer=pump_layer, recharge_layer=recharge_layer)
        pred = h_hist[inp.obs_layer, inp.obs_idx, 1:].cpu().numpy()
        r2 = _r2(pred, obs_target)
        hindcast_r2.append(r2)
        log(f"  member {mem.label}: hindcast in-sample R2 {r2:+.3f} "
            f"({time.perf_counter() - t_m:.1f}s)")
        h_end = h_hist[..., -1]
        for ni, noise in enumerate(ic_noises):
            h_start = nudge_to_observations(inp, h_end, origin, gain, noise=noise)
            for si, (scen, rain) in enumerate(scenarios):
                E_fut, r_fut, _ = future_forcing(inp, scen, horizon, rain_scale=rain,
                                                 zone_of_cell=zone_of_cell,
                                                 eta_classes=eta_classes)
                h_fut = rollout(model, scalars, h_start, E_fut, r_fut, inp.ground_elev,
                                pump_layer=pump_layer, recharge_layer=recharge_layer)
                # hindcast months 0..T-1, then the forecast's months 1..horizon
                heads = torch.cat([h_hist, h_fut[..., 1:]], dim=-1)
                subs = compaction(col, heads)
                hn = heads.cpu().numpy()
                heads_sum[si] += hn
                heads_sq[si] += hn ** 2
                subs_sum[si] += subs
                subs_sq[si] += subs ** 2
                fan_mean_rows.append({
                    "scenario": scen.name, "member": mem.label, "ic": ni,
                    **{f"head_L{k + 1}_end_m": float(hn[k, :, -1].mean()) for k in range(L)},
                    **{f"head_L{k + 1}_change_m": float((hn[k, :, -1] - hn[k, :, origin]).mean())
                       for k in range(L)},
                    "subs_end_cm": float(subs[:, -1].mean() * 100.0),
                    "subs_forward_cm": float((subs[:, -1] - subs[:, origin]).mean() * 100.0),
                    "subs_forward_p95_cm": float(np.percentile(subs[:, -1] - subs[:, origin], 95)
                                                 * 100.0),
                })
    heads_mean = heads_sum / n_mem
    heads_std = np.sqrt(np.maximum(heads_sq / n_mem - heads_mean ** 2, 0.0))
    subs_mean = subs_sum / n_mem
    subs_std = np.sqrt(np.maximum(subs_sq / n_mem - subs_mean ** 2, 0.0))
    return {"dates": all_dates, "origin": origin, "heads_mean": heads_mean.astype("float32"),
            "heads_std": heads_std.astype("float32"), "subs_mean": subs_mean.astype("float32"),
            "subs_std": subs_std.astype("float32"),
            "scenarios": [s.describe() for s, _ in scenarios],
            "scenario_names": [s.name for s, _ in scenarios],
            "rain_scales": [r for _, r in scenarios], "n_members": n_mem,
            "member_labels": [m.label for m in members], "hindcast_r2": hindcast_r2,
            "rows": pd.DataFrame(fan_mean_rows)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Forward twin: pumping scenario -> heads + subsidence")
    ap.add_argument("--theta", action="append", required=True,
                    help="stage3_theta.json (repeatable) and/or stage3_fold_thetas.json; "
                         "every parameter set becomes an ensemble member")
    ap.add_argument("--scenario", action="append", default=[],
                    help="'name:class=factor,...[,rain=f][@YYYY-MM][#zone/zone]' "
                         "(repeatable). Classes: " + ", ".join(CLASSES) + ". The baseline "
                         "(no change) is always run first.")
    ap.add_argument("--horizon", type=int, default=120, help="months past the record")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="observation nudging gain at the forecast origin (1 = restart "
                         "from the observed field, 0 = free run)")
    ap.add_argument("--hindcast-gain", type=float, default=0.0,
                    help="sequential assimilation through the record: nudge the state "
                         "toward the observed field by this gain every --hindcast-every "
                         "months (0 = plain hindcast). The hindcast R2 printed per member "
                         "is then an assimilated score, not a free-running one.")
    ap.add_argument("--hindcast-every", type=int, default=12)
    ap.add_argument("--ic-members", type=int, default=0,
                    help="extra members per parameter set, each restarting from the "
                         "observed field perturbed by N(0, --ic-sigma) per well")
    ap.add_argument("--ic-sigma", type=float, default=0.5, help="metres")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vep-json", default="results/twin/stage2_vep_shared.json",
                    help="shared Stage-2 column parameters; fitted and written if missing")
    ap.add_argument("--vep-epochs", type=int, default=2000)
    ap.add_argument("--data", default=None, help="data dir with MLCW/leveling (else "
                                                 "HYDROMIND_GW_DATA); needed only to fit "
                                                 "the column or to validate against leveling")
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--wells-from", default=None)
    ap.add_argument("--pump-layer", type=int, default=1)
    ap.add_argument("--recharge-layer", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compile-matvec", action="store_true")
    ap.add_argument("--out", required=True, help="output stem: <out>.npz, <out>.summary.csv")
    for k in ("polygon", "wells_dir", "stations", "pump_census", "pump_kwh",
              "rf_timeseries", "rf_stations", "gw_stations", "et_npz"):
        ap.add_argument(f"--{k.replace('_', '-')}", default=None)
    args = ap.parse_args(argv)

    set_compile_matvec(args.compile_matvec)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    members = load_members(args.theta)
    meta = members[0].meta
    print(f"parameter sets: {len(members)} ({', '.join(m.label for m in members)}); "
          f"param_mode={meta.get('param_mode')} boundaries={meta.get('boundaries')} "
          f"git={meta.get('git_commit')} r2_insample={meta.get('r2_insample')}", flush=True)
    verdict = meta.get("gate")
    print("GATE STATUS: " + (str(verdict) if verdict else
          "no k-fold verdict recorded in this theta file -- the pumping -> head map is "
          "UNVALIDATED; treat every number below as a model consequence, not a forecast"),
          flush=True)

    scenarios = [(BASELINE, 1.0)] + [parse_scenario(s) for s in args.scenario]
    for s, rain in scenarios:
        print(f"scenario: {s.describe()}" + (f", rain x{rain:g}" if rain != 1.0 else ""))

    paths = {k: getattr(args, k) for k in ("polygon", "wells_dir", "stations", "pump_census",
                                           "pump_kwh", "rf_timeseries", "rf_stations",
                                           "gw_stations", "et_npz")}
    inp = load_twin_inputs(paths, dx=args.dx, wells_from=args.wells_from,
                           meter_filter=meta.get("meter_filter", "none"),
                           cap_duty=float(meta.get("cap_duty", 1.0)))
    print(f"census cleaning as calibrated: meter_filter={meta.get('meter_filter', 'none')}",
          flush=True)

    cfg = Config(data_dir=args.data) if args.data else Config()
    ddir = str(cfg.data_dir) if cfg.data_dir and os.path.isdir(str(cfg.data_dir)) else None
    from .zones import fan_zones as _fan_zones

    zoc = _fan_zones(inp.grid.centroids(),
                     *_parse_zone_boundaries(meta.get("zone_boundaries", "205,182")))
    col, vep = load_or_fit_vep(args.vep_json, ddir, inp.hf, device, epochs=args.vep_epochs,
                               zone_of_cell=zoc)
    if "zonal" in vep:
        print("VEP column: per-zone (" + "; ".join(
            f"{z}: Ske={math.exp(q['log_ske']):.2e} Skv={math.exp(q['log_skv']):.2e} "
            f"tau={math.exp(q['log_tau']):.0f} d" for z, q in zip(("prox", "mid", "dist"),
                                                                    vep["zonal"], strict=True))
            + f") from {args.vep_json}", flush=True)
    else:
        print(f"VEP column: Ske={math.exp(vep['log_ske']):.3e} Skv={math.exp(vep['log_skv']):.3e} "
              f"tau={math.exp(vep['log_tau']):.0f} d", flush=True)

    t0 = time.perf_counter()
    res = run(inp, members, scenarios, args.horizon, args.gain, args.ic_members,
              args.ic_sigma, args.seed, col, device, pump_layer=args.pump_layer,
              recharge_layer=args.recharge_layer, hindcast_gain=args.hindcast_gain,
              hindcast_every=args.hindcast_every)
    print(f"ran {res['n_members']} members x {len(scenarios)} scenarios x "
          f"{len(res['dates'])} months in {time.perf_counter() - t0:.1f}s", flush=True)

    if ddir is not None:
        from .explorer3d import validate_against_leveling

        T = len(inp.dates)
        skill = validate_against_leveling(ddir, inp.grid, res["subs_mean"][0][:, :T],
                                          inp.dates)
        if skill["n_sites"]:
            print(f"hindcast subsidence vs leveling: sites {skill['n_sites']} pairs "
                  f"{skill['n_pairs']} R2 {skill['r2']:+.3f} RMSE {skill['rmse_cm']:.1f} cm "
                  f"bias {skill['bias_cm']:+.1f} cm", flush=True)
            res["leveling_skill"] = skill

    rows = res.pop("rows")
    summary = rows.drop(columns=["member", "ic"]).groupby("scenario").agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    print("\n=== forward summary (fan mean at the horizon; ± = std across members) ===")
    for name, r in summary.iterrows():
        print(f"  {name:>12}: layer-2 head change {r['head_L2_change_m_mean']:+.2f} "
              f"± {np.nan_to_num(r['head_L2_change_m_std']):.2f} m; forward subsidence "
              f"{r['subs_forward_cm_mean']:.2f} ± {np.nan_to_num(r['subs_forward_cm_std']):.2f} cm "
              f"(p95 {r['subs_forward_p95_cm_mean']:.2f} cm)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out + ".npz", dates=np.array([d.isoformat() for d in res["dates"]]),
        origin=res["origin"], heads_mean=res["heads_mean"], heads_std=res["heads_std"],
        subs_mean=res["subs_mean"], subs_std=res["subs_std"],
        scenario_names=np.array(res["scenario_names"]), scenarios=np.array(res["scenarios"]),
        mask=inp.grid.mask, nx=inp.grid.nx, ny=inp.grid.ny, dx=inp.grid.dx,
        x0=inp.grid.x0, y0=inp.grid.y0, n_members=res["n_members"],
        member_labels=np.array(res["member_labels"]), hindcast_r2=np.array(res["hindcast_r2"]),
        gate=json.dumps(meta), horizon=args.horizon, gain=args.gain,
        hindcast_gain=args.hindcast_gain, hindcast_every=args.hindcast_every)
    rows.to_csv(args.out + ".members.csv", index=False)
    summary.to_csv(args.out + ".summary.csv")
    print(f"wrote {args.out}.npz, {args.out}.members.csv, {args.out}.summary.csv")


if __name__ == "__main__":
    main()
