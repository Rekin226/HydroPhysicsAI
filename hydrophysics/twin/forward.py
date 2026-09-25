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
    aqt_fields_from_theta,
    delay_du0_from_theta,
    delay_fields_from_theta,
    set_compile_matvec,
    set_delay_tau_max,
    set_delay_tau_min,
    set_l_min,
    sw_scale_tensor,
    zone_tensor,
)
from .compaction import VEPColumn
from .flow import FlowModel
from .inputs import TwinInputs, input_options, load_twin_inputs
from .scenario import BASELINE, CLASSES, PumpingScenario, climatology
from .zones import collapse_zones, fan_zones, zone_blend_weights

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
                                + (f",{float(row['zone_split_km']):g}"
                                   if str(row.get("zone_split_km", "")) not in ("", "nan")
                                   else "")
                                if str(row.get("zone_proximal_km", "")) not in ("", "nan")
                                else "205,182"),
            "dx": float(row["dx"]), "epochs": int(row["epochs"]),
            "r2_insample": float(row["r2_insample"]),
            "gate": {"r2_kfold": float(row["r2_kfold"]), "r2_idw": float(row["r2_idw"]),
                     "margin": float(row["r2_kfold"]) - float(row["r2_idw"]),
                     "verdict": "PASS" if float(row["r2_kfold"]) > float(row["r2_idw"])
                     else "FAIL", "n_folds": int(row["n_folds"])}}
    # the opt-in physics of 2026-09-23 must travel with the member, or build_model
    # silently runs a different model than the one calibrated (review 2026-09-23)
    def _cell(key: str) -> str:
        v = row.get(key, "")
        return "" if pd.isna(v) else str(v).strip()

    if _cell("l_min") not in ("", "None"):
        meta["l_min"] = float(_cell("l_min"))
    if _cell("log_t_min_proximal") not in ("", "None"):
        meta["log_t_min_proximal"] = float(_cell("log_t_min_proximal"))
    if _cell("zone_blend_km") not in ("", "None") and float(_cell("zone_blend_km")) > 0.0:
        meta["zone_blend_km"] = float(_cell("zone_blend_km"))
    # opt-in input constructions (2026-09-23): they change h0 / ground_elev / the wells
    for key in ("ic_merged_proximal", "strict_coverage", "no_backfill",
                "ic_layered_proximal", "proximal_layered"):
        if _cell(key) in ("True", "true", "1"):
            meta[key] = True
    if _cell("ground_elev") not in ("", "None", "wells"):
        meta["ground_elev"] = _cell("ground_elev")
        if _cell("ground_elev_dem_npz") not in ("", "None"):
            meta["ground_elev_dem_npz"] = _cell("ground_elev_dem_npz")
    for key in ("delay_storage", "rivers", "river_set", "sw_recharge", "delay_u0",
                "aquitard_storage", "sw_components", "river_c_split"):
        if _cell(key) not in ("", "None"):
            meta[key] = _cell(key)
    needs_recipe = (meta.get("rivers", "none") != "none"
                    or bool(meta.get("sw_recharge"))
                    or meta.get("delay_storage", "off") != "off"
                    or meta.get("aquitard_storage", "off") != "off")
    if needs_recipe:
        # the CSV holds the switches but not the full recipe (river stage/bottom depths,
        # layers, shapefile, DEM hash, sw_layer, tau ceiling): take it from the run's
        # stage3_theta.json, and refuse rather than guess when that file is missing
        sib = os.path.join(os.path.dirname(os.path.abspath(path)), "stage3_theta.json")
        if not os.path.exists(sib):
            raise ValueError(
                f"{path}: this run used opt-in physics (rivers={meta.get('rivers', 'none')}, "
                f"sw_recharge={meta.get('sw_recharge') or 'off'}, "
                f"delay_storage={meta.get('delay_storage', 'off')}) whose full recipe is not "
                "in the CSV; pass its stage3_theta.json instead")
        with open(sib) as fh:
            sib_meta = json.load(fh).get("meta", {})
        for key in ("rivers", "river_set", "sw_recharge", "delay_storage"):
            if key in meta and str(sib_meta.get(key) or "") not in ("", str(meta[key])):
                raise ValueError(f"{path}: {key}={meta[key]!r} disagrees with {sib} "
                                 f"({sib_meta.get(key)!r}); they are not the same run")
        for key, v in sib_meta.items():
            if key.startswith(("river", "dem_", "delay_", "sw_", "aquitard_")):
                meta.setdefault(key, v)
    return {"theta": theta, "meta": meta}


def build_model(grid, member: Member, device) -> tuple[FlowModel, dict, np.ndarray | None]:
    """Rebuild the calibrated ``FlowModel`` -> ``(model, scalars, zone_of_cell)``.

    ``scalars`` holds the driver parameters that live outside the model (``log_eta``,
    ``log_head_extra``, ``recharge_frac_logit``) as float64 tensors on ``device``.
    ``zone_of_cell`` is always the three-zone map: a flow model fitted with the proximal
    split (``zone_boundaries`` "P,D,S") is rebuilt on four zones, and its ids are collapsed
    for the callers (the compaction column, scenario zones), which stay three-zone.
    """
    meta, th = member.meta, member.theta
    mode = meta.get("param_mode", "zonal")
    proximal_km = 205.0
    zone_of_cell = None
    zone_w = None
    if mode == "zonal":
        proximal_km, distal_km, split_km = _parse_zone_boundaries(
            meta.get("zone_boundaries", "205,182"), allow_split=True)
        # with the opt-in proximal split the flow uses four zone ids; the returned
        # zone_of_cell (column, scenarios, river split) is the three-zone collapse
        zone_of_cell = fan_zones(grid.centroids(), proximal_km=proximal_km, distal_km=distal_km,
                                 split_km=split_km)
        if (split_km is not None) != ("log_T_proximal_w" in th):
            raise ValueError(f"zone_boundaries {meta.get('zone_boundaries')!r} and the theta "
                             "disagree on the proximal split (log_T_proximal_w)")
        if float(meta.get("zone_blend_km") or 0.0) > 0.0:
            # --zone-blend-km: the calibrated parameters are a per-cell mix across the
            # mid/distal line; rebuilding them as a step would be a different model
            zone_w = zone_blend_weights(grid.centroids(), proximal_km, distal_km,
                                        float(meta["zone_blend_km"]), split_km=split_km)
    boundaries = None
    if meta.get("boundaries", "none") == "coast-apex":
        boundaries = fan_boundaries(grid, proximal_km=proximal_km)
    model = FlowModel(grid, n_layers=N_LAYERS, dt_days=30.0, device=device,
                      boundaries=boundaries)
    A = grid.n_active
    if meta.get("rivers", "none") not in (None, "none"):
        # the river cells are rebuilt from the recorded recipe; a changed DEM is flagged
        from .rivers import build_river_set

        # second-round options (2026-09-23): absent keys are the historical river set
        skip = (boundaries.apex_idx if (meta.get("river_skip_apex") and boundaries is not None)
                else None)
        rs, sha = build_river_set(
            grid, shp=meta.get("river_shp"), groups=meta.get("river_set",
                                                             "choushui,wu,beigang"),
            dem_npz=meta.get("dem_npz", "results/twin/basemap.npz"),
            stage_depth=float(meta.get("river_stage_depth", 1.0)),
            rbot_depth=float(meta.get("river_rbot_depth", 3.0)),
            edge_buffer_m=meta.get("river_edge_buffer_m"),
            zone_of_cell=None if zone_of_cell is None else collapse_zones(zone_of_cell),
            c_split=meta.get("river_c_split") or "group", exclude_idx=skip,
            season_csv=meta.get("river_stage_season"))
        if meta.get("river_groups") and list(rs.names) != list(meta["river_groups"]):
            raise ValueError(f"rebuilt river groups {list(rs.names)} differ from the "
                             f"calibrated {meta['river_groups']}")
        if meta.get("dem_sha1") and sha != meta["dem_sha1"]:
            print(f"WARNING: the DEM behind the river stages changed since calibration "
                  f"({meta['dem_sha1'][:8]} -> {sha[:8]})", flush=True)
        model.set_rivers(rs, layer=int(meta.get("river_layer", 0)), mode=meta["rivers"])

    def t(v):
        return torch.tensor(np.asarray(v, dtype="float64").reshape(-1, 1),
                            dtype=torch.float64, device=device)

    with torch.no_grad():
        if mode == "zonal":
            zt = zone_tensor(zone_of_cell, device, zone_w)
            keys = ("log_T_proximal", "log_S_proximal", "log_T_mid", "log_S_mid",
                    "log_T_distal", "log_S_distal", "log_L_mid", "log_L_distal",
                    "log_T_proximal_w", "log_S_proximal_w",
                    # --proximal-layered (2026-09-25): per-layer proximal T/S come in as
                    # (L, 1) through t(); these learnable leakances replace the constant
                    "log_L_proximal", "log_L_proximal_w")
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
    set_delay_tau_max(meta.get("delay_tau_max_years"))
    set_delay_tau_min(meta.get("delay_tau_min_days"))
    scalars = {}
    # opt-in physics (2026-09-23): delay bed, river conductances, canal-recharge fraction
    dl = meta.get("delay_layers")
    d_Sd, d_tau = delay_fields_from_theta(th, zone_of_cell, N_LAYERS, A, device=device,
                                          layers=tuple(dl) if dl is not None else None,
                                          zone_w=zone_w)
    if d_Sd is not None:
        model.delay_log_Sd, model.delay_log_tau = d_Sd, d_tau
        scalars["delay_Sd"], scalars["delay_tau"] = d_Sd, d_tau
        du0 = delay_du0_from_theta(th, zone_of_cell, N_LAYERS, A, device=device,
                                   zone_w=zone_w)
        if du0 is not None:
            # --delay-u0 learned: the record starts with the slow store above the aquifer
            scalars["delay_du0"] = du0
            model.delay_log_du0 = torch.log(du0)
    a_Sa, a_G = aqt_fields_from_theta(th, zone_of_cell, N_LAYERS, A, device=device,
                                      zone_w=zone_w)
    if a_Sa is not None:
        model.aqt_log_Sa, model.aqt_log_G = a_Sa, a_G
        scalars["aqt_Sa"], scalars["aqt_G"] = a_Sa, a_G
    if model.has_rivers:
        if "log_C_riv" not in th:
            raise ValueError("meta says rivers but theta has no log_C_riv")
        scalars["log_C_riv"] = torch.tensor(np.ravel(th["log_C_riv"]), dtype=torch.float64,
                                            device=device)
    if "log_sw_scale" in th:
        scalars["log_sw_scale"] = sw_scale_tensor(th["log_sw_scale"], device=device)
        if meta.get("sw_layer") is not None:
            scalars["sw_layer"] = int(meta["sw_layer"])
    for k in ("log_eta", "log_head_extra", "recharge_frac_logit", "pump_split_logit",
              "return_frac_logit"):
        if k in th:
            scalars[k] = torch.tensor(np.asarray(th[k], dtype="float64"),
                                      dtype=torch.float64, device=device)
    if "spread_km" in th:
        from .spread import pairwise_d2_km, spread_matrix

        scalars["spread_W"] = spread_matrix(
            pairwise_d2_km(grid, device=device),
            torch.tensor(math.log(float(th["spread_km"])), dtype=torch.float64, device=device))
    if zone_of_cell is not None:
        zone_of_cell = collapse_zones(zone_of_cell)     # a no-op without the proximal split
    return model, scalars, zone_of_cell


def rollout(model: FlowModel, scalars: dict, h0: torch.Tensor, E: torch.Tensor,
            recharge_field: torch.Tensor, ground_elev: torch.Tensor,
            pump_layer: int = 1, recharge_layer: int = 0,
            sw_field: torch.Tensor | None = None, u0: torch.Tensor | dict | None = None,
            return_state: bool = False, month0: int | None = None):
    """Run the calibrated model over a forcing sequence -> heads ``(L, A, T+1)`` (float64,
    on the model's device). ``E`` and ``recharge_field`` are ``(A, T)``.

    Models calibrated with the opt-in physics of 2026-09-23 need more: ``sw_field``
    (A, T) of canal deliveries when the theta has ``log_sw_scale`` (refused if missing:
    silently dropping a recharge term the size of rainfall's would be a different model),
    and, with a delay bed, ``u0`` (the slow store's state; defaults to ``h0``, i.e.
    equilibrium). ``return_state=True`` returns ``(heads, u_end)`` so a projection can
    start from the record's slow state.

    Second round (2026-09-23): a ``--delay-u0 learned`` model starts its slow store at
    ``h0 + du0`` when ``u0`` is not given (the record's start; a projection passes the
    hindcast's ``u_end``); an aquitard-store model's state is ``{"u", "ua"}`` and is
    returned and accepted in that form; ``month0`` is the calendar month index (0 =
    January) of the first step, needed only by a river season table (default: the
    record's own start, February 2012)."""
    ext = {}
    if "delay_Sd" in scalars:
        if u0 is None and "delay_du0" in scalars:
            u0 = h0.to(scalars["delay_du0"].device) + scalars["delay_du0"]
        elif isinstance(u0, dict) and u0.get("u") is None and "delay_du0" in scalars:
            u0 = {**u0, "u": h0.to(scalars["delay_du0"].device) + scalars["delay_du0"]}
        ext.update(delay_Sd=scalars["delay_Sd"], delay_tau=scalars["delay_tau"], u0=u0)
    if "aqt_Sa" in scalars:
        ext.update(aqt_Sa=scalars["aqt_Sa"], aqt_G=scalars["aqt_G"])
        if "u0" not in ext and u0 is not None:
            ext["u0"] = u0
    if month0 is not None:
        ext["river_month0"] = int(month0)
    if "log_C_riv" in scalars:
        ext["log_C_riv"] = scalars["log_C_riv"]
    if "log_sw_scale" in scalars:
        if sw_field is None:
            raise ValueError("this model was calibrated with --sw-recharge: pass sw_field "
                             "(canal deliveries, (A, T)) to rollout")
        ext.update(sw_field=sw_field, log_sw_scale=scalars["log_sw_scale"],
                   sw_layer=scalars.get("sw_layer"))
    with torch.no_grad():
        model.set_apex_heads(h0)
        out = _rollout(
            model, model.log_T, model.log_S, model.log_L, h0, E.shape[-1], **ext,
            return_state=True,
            recharge_field=recharge_field, recharge_scale=scalars.get("recharge_frac_logit"),
            recharge_layer=recharge_layer, E=E, log_eta=scalars.get("log_eta"),
            log_head_extra=scalars.get("log_head_extra"), ground_elev=ground_elev,
            pump_layer=pump_layer, log_C_coast=model.log_C_coast, log_C_apex=model.log_C_apex,
            pump_split_logit=scalars.get("pump_split_logit"),
            return_frac_logit=scalars.get("return_frac_logit"),
            spread_W=scalars.get("spread_W"))
    return out if return_state else out[0]


# ---------------------------------------------------------------------------------------
# scenarios and forcing
# ---------------------------------------------------------------------------------------
def parse_scenario(text: str) -> tuple[PumpingScenario, float]:
    """``"name:irrigation=0.7,aquaculture=0[,rain=0.9][,sw=0.8]@2026-01#proximal/mid"``
    -> scenario.

    ``rain`` is not a pumping class; it scales the recharge climatology and comes back as
    the second element. ``sw`` scales the canal irrigation deliveries (a surface-water
    policy lever, ``PumpingScenario.sw_factor``; only models calibrated with
    ``--sw-recharge`` respond to it) and ``sw_sub`` is the share of the cut canal water
    replaced by irrigation pumping (``sw_substitution_energy``). Everything after ``#``
    names the zones the policy applies to.
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
    sw = 1.0
    sw_sub = 0.0
    for item in rest.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=")
        k, v = k.strip(), float(v)
        if k == "rain":
            rain = v
        elif k == "sw":
            sw = v
        elif k == "sw_sub":
            sw_sub = v
        elif k in CLASSES:
            factors[k] = v
        else:
            raise ValueError(f"scenario {name!r}: unknown class {k!r}; expected one of "
                             f"{CLASSES}, 'rain', 'sw' or 'sw_sub'")
    return PumpingScenario(name.strip(), factors=factors, zones=zones, start=start,
                           sw_factor=sw, sw_sub=sw_sub), rain


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


def attach_sw_recharge(inp: TwinInputs, meta: dict, log=print) -> None:
    """Load the canal deliveries a member was calibrated with into ``inp.sw_field`` (in
    place; no-op for a run without ``--sw-recharge``). Every caller of ``rollout`` that
    rebuilds a member from its theta file needs this, or an sw model refuses to run."""
    if meta.get("sw_recharge") and getattr(inp, "sw_field", None) is None:
        from .inputs import load_sw_recharge

        comp = meta.get("sw_components") or "delivered"
        inp.sw_field = load_sw_recharge(meta["sw_recharge"], inp.grid, inp.dates,
                                        components=comp)
        log(f"canal deliveries as calibrated: {meta['sw_recharge']} ({comp})")


def sw_hist(inp: TwinInputs) -> torch.Tensor | None:
    """The record's canal deliveries aligned with ``recharge_field[:, 1:]``, or ``None``.
    ``(A, T-1)``, or ``(K, A, T-1)`` for a multi-component (``--sw-components both``) run."""
    sw = getattr(inp, "sw_field", None)
    if sw is None:
        return None
    return torch.as_tensor(np.asarray(sw), dtype=torch.float64)[..., 1:]


def delay_state_path(model: FlowModel, scalars: dict, heads: torch.Tensor,
                     u0: torch.Tensor | None = None) -> torch.Tensor | None:
    """The delay bed's slow-store head ``u`` along a rollout ``heads (L, A, T+1)`` ->
    ``(L, A, T+1)``, or ``None`` for a model without one. It replays the solver's own
    update ``u' = (tau u + dt h')/(tau + dt)`` from ``u0`` (default ``heads[..., 0]``,
    equilibrium), so ``u[..., t]`` is exactly the state a rollout ending at ``t`` returns
    with ``return_state=True`` -- a projection started mid-record needs it as ``u0``.

    An aquitard-store model (``--aquitard-storage``, with or without the delay bed) is
    refused rather than answered with ``None``: ``None`` means "no slow state", and a
    caller that starts a projection from it would silently reset ``u_a`` to equilibrium.
    Carry ``rollout(..., return_state=True)`` states instead."""
    if "aqt_Sa" in scalars:
        raise NotImplementedError("delay_state_path replays the interbed store only; an "
                                  "aquitard-store model needs rollout(return_state=True)")
    if "delay_tau" not in scalars:
        return None
    tau = torch.exp(scalars["delay_tau"]).to(heads.device)
    dt = float(model.dt)
    if u0 is None:
        u = heads[..., 0]
        if "delay_du0" in scalars:           # --delay-u0 learned: the record's own start
            u = u + scalars["delay_du0"].to(heads.device)
    else:
        u = u0.to(heads.device)
    out = [u]
    for t in range(1, heads.shape[-1]):
        u = (tau * u + dt * heads[..., t]) / (tau + dt)
        out.append(u)
    return torch.stack(out, dim=-1)


def future_sw(inp: TwinInputs, scenario: PumpingScenario, horizon: int,
              zone_of_cell: np.ndarray | None = None) -> torch.Tensor | None:
    """Canal deliveries past the record: month-of-year climatology of ``inp.sw_field``
    under the scenario's ``sw_factor`` -> ``(A, horizon)``, or ``None`` without one."""
    if inp.sw_field is None:
        return None
    clim, fut_dates = climatology(np.asarray(inp.sw_field), inp.dates, horizon)
    return torch.tensor(scenario.apply_sw(clim, fut_dates, zone_of_cell=zone_of_cell),
                        dtype=torch.float64)


def sw_substitution_energy(inp: TwinInputs, scenario: PumpingScenario, horizon: int,
                           scalars: dict, h_ref: torch.Tensor, pump_layer: int = 1,
                           zone_of_cell: np.ndarray | None = None,
                           dt: float = 30.0) -> torch.Tensor | None:
    """``--scenario ...,sw=0.8,sw_sub=f``: the extra irrigation pumping energy (kWh,
    ``(A, horizon)``) that replaces ``f`` x the canal water a scenario cuts, cell by cell.

    A cut in canal deliveries is in practice met by pumping (Changhua opened 31 deep wells
    in the 2021 drought; Yunlin's independent canals switched to wells): without this,
    ``sw=0.8`` alone is optimistic. The lost *delivered* volume (``sw_m_per_day``, the
    first component) is converted to electricity with the member's own efficiency and
    total dynamic head at the reference head ``h_ref`` (``(L, A)``, the projection's
    start), i.e. the inverse of ``pumping.energy_to_volume``, so the pumped volume the
    solver sees at that head is exactly ``f`` x the lost delivery. The monthly volume is
    built with the solver step ``dt`` (``model.dt``, 30 days), not the calendar month
    length, because ``_rollout`` applies pumping as volume / dt while canal water enters
    as m/day x area: the substituted pumping RATE then equals the lost delivery RATE in
    every month. ``None`` when nothing is substituted."""
    from .pumping import J_PER_KWH, MIN_LIFT_M, RHO_G

    f = float(getattr(scenario, "sw_sub", 0.0) or 0.0)
    if f <= 0.0 or inp.sw_field is None or scenario.sw_factor == 1.0:
        return None
    if "log_eta" not in scalars:
        raise ValueError("sw_sub needs a member with a pumping conversion (log_eta)")
    sw = np.asarray(inp.sw_field)
    delivered = sw[0] if sw.ndim == 3 else sw
    clim, fut_dates = climatology(delivered, inp.dates, horizon)
    lost = clim - scenario.apply_sw(clim, fut_dates, zone_of_cell=zone_of_cell)  # m/day
    vol = lost * float(inp.grid.dx) ** 2 * float(dt) * f                   # m3 per solver step
    log_eta = scalars["log_eta"].detach().cpu().double().reshape(-1)[0]
    head_extra = (float(torch.exp(scalars["log_head_extra"]).reshape(-1)[0])
                  if "log_head_extra" in scalars else 0.0)
    lift = inp.ground_elev.double().cpu() - h_ref[pump_layer].detach().double().cpu()
    head = torch.clamp(lift + head_extra, min=MIN_LIFT_M).numpy()
    kwh = vol * RHO_G * head[:, None] / (float(torch.exp(log_eta)) * J_PER_KWH)
    return torch.tensor(kwh, dtype=torch.float64)


def add_irrigation_energy(E: torch.Tensor, extra: torch.Tensor | None,
                          eta_classes: list[str] | None) -> torch.Tensor:
    """Add ``extra`` (A, T) kWh to the irrigation class of ``E`` ((A, T) or, with
    per-class efficiencies, (C, A, T) in ``eta_classes`` order)."""
    if extra is None:
        return E
    if E.dim() == 3:
        if not eta_classes or "irrigation" not in eta_classes:
            raise ValueError("sw_sub with eta classes needs an 'irrigation' class")
        E = E.clone()
        E[eta_classes.index("irrigation")] += extra.to(E)
        return E
    return E + extra.to(E)


def restart_taper(inp: TwinInputs, month: int, n_layers: int, taper_km: float) -> np.ndarray:
    """``(L, A)`` weight ``exp(-(d / taper_km)^2)``, where ``d`` is each cell's distance to
    the nearest well of the same layer that has a head in ``month``. A layer with no well
    gets zero weight everywhere."""
    cent = inp.grid.centroids()
    fin = np.isfinite(inp.obs_h[:, month])
    w = np.zeros((n_layers, cent.shape[0]), dtype="float64")
    for k in range(n_layers):
        xy = inp.well_xy[fin & (inp.obs_layer == k)]
        if len(xy):
            d = np.sqrt(((cent[:, None, :] - xy[None]) ** 2).sum(-1)).min(1) / 1000.0
            w[k] = np.exp(-(d / float(taper_km)) ** 2)
    return w


def nudge_to_observations(inp: TwinInputs, h_model: torch.Tensor, month: int,
                          gain: float, noise: np.ndarray | None = None,
                          taper_km: float | None = None) -> torch.Tensor:
    """``h_model + gain * (h_obs - h_model)`` in every layer that has an observation in
    ``month``; other layers keep the model state. ``h_model`` is ``(L, A)``.

    ``taper_km`` (``--restart-taper-km``, fix A2, 2026-09-23) multiplies the correction by
    :func:`restart_taper`, so cells far from a well of the same layer keep the model's own
    state. Without it, a layer with any well anywhere is replaced wholesale by the IDW
    field. At the Dec-2022 origin that put heads carried in from mid and distal wells into
    the unmeasured proximal layers 3-4, a median of -21 and -25 m. The column turned that
    into a step of +21 cm per proximal cell, and the rebound that followed elsewhere."""
    if gain <= 0.0:
        return h_model
    h = inp.obs_h[:, month].copy()
    if noise is not None:
        h = h + noise
    finite = np.isfinite(h)
    out = h_model.clone()
    field = inp.initial_heads(month, n_layers=h_model.shape[0],
                              well_mask=finite, noise=noise).to(h_model)
    taper = (torch.as_tensor(restart_taper(inp, month, h_model.shape[0], taper_km)).to(h_model)
             if taper_km and taper_km > 0 else None)
    for k in range(h_model.shape[0]):
        if (finite & (inp.obs_layer == k)).any():
            g = gain if taper is None else gain * taper[k]
            out[k] = h_model[k] + g * (field[k] - h_model[k])
    return out


# ---------------------------------------------------------------------------------------
# compaction
# ---------------------------------------------------------------------------------------
class ZonalColumn(torch.nn.Module):
    """One VEP parameter set per fan zone (``calibrate_coupled --configs zonal``), applied
    cell by cell through ``zone_of_cell``. Same call signature as ``VEPColumn`` on the
    layer-mean head, so ``compaction`` does not care which it gets."""

    def __init__(self, params: list[dict], zone_of_cell: np.ndarray, device=None,
                 zone_w: np.ndarray | None = None):
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
        # --zone-blend-km: (N_ZONES, A) weights; each cell's parameters are the mix
        self.register_buffer("zone_w", None if zone_w is None else torch.as_tensor(
            np.asarray(zone_w, dtype="float64"), dtype=torch.float64, device=device))

    def forward(self, drv: torch.Tensor) -> torch.Tensor:          # drv (A, T)
        if self.zone_w is not None:
            return blended_column(self.cols, self.zone_w, drv)
        out = torch.zeros_like(drv)
        for z, c in enumerate(self.cols):
            sel = self.zone == z
            if sel.any():
                out[sel] = c(drv[sel])
        return out


COLUMN_KEYS = ("log_ske", "log_skv", "log_tau", "h_pc0")


def blended_column(cols, zone_w: torch.Tensor, drv: torch.Tensor) -> torch.Tensor:
    """Per-zone VEP columns mixed per cell by ``zone_w`` ``(N_ZONES, n)`` -> ``(n, T)``.

    ``log_ske``, ``log_skv`` and ``log_tau`` mix in log space (a geometric mean) and
    ``h_pc0`` mixes linearly, which is what ``cols @ W`` gives on the stored parameters.
    With one-hot weights it equals the sharp per-zone column exactly. It is
    differentiable in every zone's parameters, so ``calibrate_coupled`` fits through it.
    """
    from .compaction import vep_compaction

    W = zone_w.to(dtype=drv.dtype, device=drv.device)
    per = {k: torch.cat([getattr(c, k).reshape(1) for c in cols]).to(drv.dtype) @ W
           for k in COLUMN_KEYS}
    return vep_compaction(drv, per["log_ske"], per["log_skv"], per["log_tau"], per["h_pc0"],
                          float(cols[0].dt_days))


def release_fast_startup(p: dict, tau_days: float | None) -> tuple[dict, list[int]]:
    """Fix A1 (2026-09-23): set ``h_pc0`` to 0 in every column whose creep time constant
    is shorter than ``tau_days`` and whose ``h_pc0`` is positive.

    A positive ``h_pc0`` starts the column under-consolidated: at step 1 it loads
    ``Skv * h_pc0`` of pending compaction, and after that the offset has no effect. With
    a short tau, that load is released within a few months of the record's start, before
    the first leveling survey, which re-zeros every site. The calibration therefore
    cannot see it. The fitted proximal column (tau 24 d, Skv 0.257, h_pc0 +4.7 m) puts
    about 121 cm into every proximal cell in early 2012. Setting its offset to 0 changes
    the in-sample leveling R2 by -0.0003. A column with a long tau (mid, 3960 d) releases
    its offset as creep that leveling does see, so it is kept.

    Returns ``(params, changed column indices)``. ``tau_days`` ``None`` or <= 0 changes
    nothing.
    """
    if not tau_days or tau_days <= 0:
        return p, []
    out = json.loads(json.dumps(p))
    cols = out.get("zonal", [out])
    changed = []
    for i, c in enumerate(cols):
        if math.exp(float(c["log_tau"])) < float(tau_days) and float(c["h_pc0"]) > 0.0:
            c["h_pc0"] = 0.0
            changed.append(i)
    return out, changed


def load_or_fit_vep(vep_json: str | None, ddir: str | None, hf, device,
                    epochs: int = 2000, zone_of_cell: np.ndarray | None = None,
                    zone_weights=None, hpc0_fast_days: float | None = None,
                    blend_km: float | None = None) -> tuple[torch.nn.Module, dict]:
    """The compaction column: read it from ``vep_json`` if that exists (a shared
    Stage-2 set, or a per-zone set from ``calibrate_coupled``), else fit the shared
    Stage-2 column on the MLCW sites (``explorer3d.fit_shared_vep``) and write it there.

    Opt-in fixes (2026-09-23; defaults change nothing):
    - ``hpc0_fast_days`` applies :func:`release_fast_startup` (A1).
    - A per-zone column fitted with ``--zone-blend-km`` (``zone_blend_km`` in its JSON) is
      rebuilt blended. That needs ``zone_weights``, a function ``km -> (N_ZONES, A)``.
    - ``blend_km`` overrides the file's blend width. This is an unrefitted illustration,
      and the returned params record it."""
    if vep_json and os.path.exists(vep_json):
        with open(vep_json) as fh:
            p = json.load(fh)
        p, changed = release_fast_startup(p, hpc0_fast_days)
        if changed:
            p["hpc0_released"] = {"tau_days_below": float(hpc0_fast_days),
                                  "columns": changed}
        else:
            # A1 already clean at the source: a column refitted with
            # ``calibrate_coupled --hpc0-guard-days`` has no fast column with a positive
            # offset, so the release has nothing to do. Record that A1 holds, so the app
            # does not re-remove a start-up step that is not there (review D1).
            thr = hpc0_fast_days if hpc0_fast_days and hpc0_fast_days > 0 else \
                p.get("hpc0_guard_days")
            if thr and float(thr) > 0 and not any(
                    math.exp(float(c["log_tau"])) < float(thr) and float(c["h_pc0"]) > 0.0
                    for c in p.get("zonal", [p])):
                p["hpc0_released"] = {"tau_days_below": float(thr), "columns": [],
                                      "already_clean": True}
        if "zonal" in p:
            if zone_of_cell is None:
                raise ValueError("a zonal column needs zone_of_cell")
            w_km = float(p.get("zone_blend_km") or 0.0)
            if blend_km is not None:
                if abs(float(blend_km) - w_km) > 1e-9:
                    p["zone_blend_km_override"] = {"fitted": w_km, "used": float(blend_km)}
                w_km = float(blend_km)
            zw = None
            if w_km > 0.0:
                if zone_weights is None:
                    raise ValueError(f"{vep_json}: a column blended over {w_km:g} km needs "
                                     "zone_weights to rebuild its per-cell parameters")
                zw = zone_weights(w_km)
            return ZonalColumn(p["zonal"], zone_of_cell, device=device,
                               zone_w=zw).to(device), p
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


def load_columns(vep_jsons: str, ddir: str | None, hf, device, epochs: int = 2000,
                 zone_of_cell: np.ndarray | None = None, zone_weights=None,
                 hpc0_fast_days: float | None = None, blend_km: float | None = None
                 ) -> list[tuple[str, torch.nn.Module, dict]]:
    """A comma-separated ``--vep-json`` -> ``[(label, column, params)]``, the rheology
    axis of the ensemble (2026-09-23). The creep time constant is not identifiable from
    the 11-year record (STATE §3.2: a tau ceiling of 11 or 30 years fits leveling
    equally and diverges after it), so passing both columns makes every projection
    carry that spread. Labels: the file's ``tau_max_years`` if recorded, else its
    directory name."""
    out = []
    for path in [p.strip() for p in vep_jsons.split(",") if p.strip()]:
        col, p = load_or_fit_vep(path, ddir, hf, device, epochs=epochs,
                                 zone_of_cell=zone_of_cell, zone_weights=zone_weights,
                                 hpc0_fast_days=hpc0_fast_days, blend_km=blend_km)
        tmax = p.get("tau_max_years")
        label = (f"tau{float(tmax):g}y" if tmax is not None
                 else os.path.basename(os.path.dirname(path)) or os.path.basename(path))
        if any(label == q[0] for q in out):
            label = f"{label}_{len(out)}"
        out.append((label, col, p))
    return out


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
                          pump_layer: int = 1, recharge_layer: int = 0,
                          sw: torch.Tensor | None = None, return_state: bool = False,
                          return_u_path: bool = False):
    """Hindcast in segments of ``every`` months, nudging the state toward that month's
    observed IDW field by ``gain`` at each segment end (sequential assimilation through
    the record). ``gain=0`` or ``every<=0`` is the plain hindcast. ``sw`` (A, T) is the
    record's canal deliveries for a model calibrated with them; ``return_state`` also
    returns the delay bed's slow state at the end (``None`` without one). Nudging acts on
    the aquifer heads only: the slow store keeps the model's own state.

    ``return_u_path`` (implies ``return_state``) appends the delay bed's slow-store path
    ``(L, A, T+1)`` (``None`` without a delay bed), replayed segment by segment from the
    solver's own UN-nudged heads: replaying the returned (nudged) series would not be the
    state the solver carried, and would jump at the projection start."""
    T = E.shape[-1]
    if gain <= 0.0 or every <= 0:
        res = rollout(model, scalars, h0, E, R, inp.ground_elev, pump_layer, recharge_layer,
                      sw_field=sw, return_state=return_state or return_u_path)
        if not return_u_path:
            return res
        heads, u = res
        return heads, u, delay_state_path(model, scalars, heads)
    want_path = return_u_path and "delay_tau" in scalars
    if want_path and "aqt_Sa" in scalars:
        delay_state_path(model, scalars, h0[..., None])      # raises: not replayable
    h = h0.to(dtype=torch.float64, device=model.log_T.device)
    out = [h[..., None]]
    u_out = []
    t = 0
    u = None
    while t < T:
        n = min(every, T - t)
        u_in = u
        seg, u = rollout(model, scalars, h, E[..., t:t + n], R[:, t:t + n], inp.ground_elev,
                         pump_layer, recharge_layer,
                         sw_field=None if sw is None else sw[..., t:t + n], u0=u,
                         return_state=True, month0=(1 + t) % 12)
        if want_path:
            # seg is the solver's own (un-nudged) segment: its replay is the solver's u
            up = delay_state_path(model, scalars, seg, u0=u_in)
            u_out.append(up if not u_out else up[..., 1:])
        out.append(seg[..., 1:])
        t += n
        # month index t in the record (E is the record from month 1 on)
        h = nudge_to_observations(inp, seg[..., -1], t, gain)
        if t < T:
            out[-1] = torch.cat([seg[..., 1:-1], h[..., None]], dim=-1)
    heads = torch.cat(out, dim=-1)
    if return_u_path:
        return heads, u, (torch.cat(u_out, dim=-1) if want_path else None)
    return (heads, u) if return_state else heads


def run(inp: TwinInputs, members: list[Member], scenarios: list[tuple[PumpingScenario, float]],
        horizon: int, gain: float, ic_members: int, ic_sigma: float, seed: int,
        col: VEPColumn | list, device, pump_layer: int = 1, recharge_layer: int = 0,
        log=print, hindcast_gain: float = 0.0, hindcast_every: int = 0,
        dump_delay_state: bool = False, restart_taper_km: float | None = None,
        column_heads: str = "restart", save_members: str | None = None) -> dict:
    """Run every member through every scenario. Opt-in fixes (2026-09-23), defaults off:

    - ``restart_taper_km`` tapers the origin restart by distance to the wells (A2,
      :func:`nudge_to_observations`).
    - ``column_heads="free"`` drives the column with a projection started from the
      hindcast's own end state (gain 0), the trajectory the column was calibrated on. The
      initial-field perturbation of an ``ic`` member is still added, so that axis survives.
      Displayed heads keep the restart. This is fix A2. It costs one more flow rollout per
      member, initial field and scenario.
    - ``save_members="yearly"`` keeps every member's December fields of subsidence and
      layer-2 head (``members_yearly`` in the result; ``main`` writes the sidecar
      ``<out>.members.npz``).
    """
    if column_heads not in ("restart", "free"):
        raise ValueError(f"column_heads must be 'restart' or 'free', got {column_heads!r}")
    # ``col`` is one column or the rheology axis ``[(label, column), ...]``: every flow
    # member's heads drive every column, and the subsidence statistics pool them all
    cols = ([(c[0], c[1]) for c in col] if isinstance(col, list) else [("column", col)])
    n_rheo = len(cols)
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
    sw_rec = sw_hist(inp)
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
    subs_sum = np.zeros((n_rheo, n_scen, A, T + horizon), dtype="float64")
    # --dump-delay-state: the slow store's head along hindcast + projection, so a later
    # column variant can be driven by it (its drainage IS clay compaction)
    delay_u_sum = (np.zeros((n_scen, L, A, T + horizon), dtype="float64")
                   if dump_delay_state else None)
    subs_sq = np.zeros_like(subs_sum)
    fan_mean_rows = []
    hindcast_r2 = []
    ye_idx = np.array([i for i, d in enumerate(all_dates) if d.month == 12], dtype="int64")
    n_ic = len(ic_noises)
    mem_yr = None
    if save_members == "yearly":
        mem_yr = {"subs": np.zeros((n_scen, n_mem * n_rheo, A, len(ye_idx)), dtype="float32"),
                  "headL2": np.zeros((n_scen, n_mem, A, len(ye_idx)), dtype="float32"),
                  "member": [], "ic": [], "rheology": [], "head_member": [], "head_ic": []}
    elif save_members not in (None, "none"):
        raise ValueError(f"save_members must be None/'none' or 'yearly', got {save_members!r}")

    obs_target = inp.obs_h_filled[:, 1:]
    for mi, mem in enumerate(members):
        t_m = time.perf_counter()
        model, scalars, zone_of_cell = build_model(inp.grid, mem, device)
        dump_u = dump_delay_state and "delay_tau" in scalars and "aqt_Sa" not in scalars
        h_hist, u_end, u_hist = hindcast_with_nudging(
            model, scalars, inp, h0, E_hist, r_hist, hindcast_gain, hindcast_every,
            pump_layer=pump_layer, recharge_layer=recharge_layer, sw=sw_rec,
            return_u_path=True) if dump_u else (*hindcast_with_nudging(
                model, scalars, inp, h0, E_hist, r_hist, hindcast_gain, hindcast_every,
                pump_layer=pump_layer, recharge_layer=recharge_layer, sw=sw_rec,
                return_state=True), None)
        pred = h_hist[inp.obs_layer, inp.obs_idx, 1:].cpu().numpy()
        r2 = _r2(pred, obs_target)
        hindcast_r2.append(r2)
        log(f"  member {mem.label}: hindcast in-sample R2 {r2:+.3f} "
            f"({time.perf_counter() - t_m:.1f}s)")
        h_end = h_hist[..., -1]
        for ni, noise in enumerate(ic_noises):
            h_start = nudge_to_observations(inp, h_end, origin, gain, noise=noise,
                                            taper_km=restart_taper_km)
            h_col0 = None
            if column_heads == "free":
                # the model's own end state, plus only the initial-field perturbation this
                # ic member carries (zero for the unperturbed member)
                h_col0 = h_end
                if noise is not None:
                    h_col0 = h_end + (h_start - nudge_to_observations(
                        inp, h_end, origin, gain, noise=None, taper_km=restart_taper_km))
            for si, (scen, rain) in enumerate(scenarios):
                E_fut, r_fut, fut_dates = future_forcing(inp, scen, horizon, rain_scale=rain,
                                                         zone_of_cell=zone_of_cell,
                                                         eta_classes=eta_classes)
                # canal cuts met by pumping (scenario sw_sub; None when not asked for)
                E_fut = add_irrigation_energy(
                    E_fut, sw_substitution_energy(inp, scen, horizon, scalars, h_start,
                                                  pump_layer, zone_of_cell, dt=model.dt),
                    eta_classes)
                # the delay bed's slow state carries from the record into the projection
                h_fut, _ = rollout(model, scalars, h_start, E_fut, r_fut,
                                       inp.ground_elev, pump_layer=pump_layer,
                                       recharge_layer=recharge_layer,
                                       sw_field=future_sw(inp, scen, horizon, zone_of_cell),
                                       u0=u_end, return_state=True,
                                       month0=fut_dates[0].month - 1)
                if dump_u:
                    # the hindcast path is the solver's own (un-nudged) replay, so it
                    # joins the projection's replay from the real u_end without a jump
                    u_path = torch.cat([
                        u_hist,
                        delay_state_path(model, scalars, h_fut, u0=u_end)[..., 1:]], dim=-1)
                    delay_u_sum[si] += u_path.cpu().numpy()
                # hindcast months 0..T-1, then the forecast's months 1..horizon
                heads = torch.cat([h_hist, h_fut[..., 1:]], dim=-1)
                hn = heads.cpu().numpy()
                heads_sum[si] += hn
                heads_sq[si] += hn ** 2
                heads_col = heads
                if h_col0 is not None:
                    h_fc, _ = rollout(model, scalars, h_col0, E_fut, r_fut, inp.ground_elev,
                                      pump_layer=pump_layer, recharge_layer=recharge_layer,
                                      sw_field=future_sw(inp, scen, horizon, zone_of_cell),
                                      u0=u_end, return_state=True,
                                      month0=fut_dates[0].month - 1)
                    heads_col = torch.cat([h_hist, h_fc[..., 1:]], dim=-1)
                k_mem = mi * n_ic + ni
                if mem_yr is not None:
                    mem_yr["headL2"][si, k_mem] = hn[1][:, ye_idx]
                    if si == 0:
                        mem_yr["head_member"].append(mem.label)
                        mem_yr["head_ic"].append(ni)
                for ri, (rlabel, rcol) in enumerate(cols):
                    subs = compaction(rcol, heads_col)
                    subs_sum[ri, si] += subs
                    subs_sq[ri, si] += subs ** 2
                    if mem_yr is not None:
                        mem_yr["subs"][si, k_mem * n_rheo + ri] = subs[:, ye_idx]
                        if si == 0:
                            mem_yr["member"].append(mem.label)
                            mem_yr["ic"].append(ni)
                            mem_yr["rheology"].append(rlabel)
                    fan_mean_rows.append({
                        "scenario": scen.name, "member": mem.label, "ic": ni,
                        "rheology": rlabel,
                        **{f"head_L{k + 1}_end_m": float(hn[k, :, -1].mean())
                           for k in range(L)},
                        **{f"head_L{k + 1}_change_m":
                           float((hn[k, :, -1] - hn[k, :, origin]).mean()) for k in range(L)},
                        "subs_end_cm": float(subs[:, -1].mean() * 100.0),
                        "subs_forward_cm": float((subs[:, -1] - subs[:, origin]).mean()
                                                 * 100.0),
                        "subs_forward_p95_cm": float(
                            np.percentile(subs[:, -1] - subs[:, origin], 95) * 100.0),
                    })
    heads_mean = heads_sum / n_mem
    heads_std = np.sqrt(np.maximum(heads_sq / n_mem - heads_mean ** 2, 0.0))
    # pooled over flow members x rheologies: the rheology spread is part of subs_std
    subs_mean = subs_sum.sum(axis=0) / (n_mem * n_rheo)
    subs_std = np.sqrt(np.maximum(subs_sq.sum(axis=0) / (n_mem * n_rheo) - subs_mean ** 2,
                                  0.0))
    return {"dates": all_dates, "origin": origin, "heads_mean": heads_mean.astype("float32"),
            "heads_std": heads_std.astype("float32"), "subs_mean": subs_mean.astype("float32"),
            "subs_std": subs_std.astype("float32"),
            "rheology_labels": [c[0] for c in cols],
            "subs_mean_by_rheology": (subs_sum / n_mem).astype("float32"),
            "scenarios": [s.describe() for s, _ in scenarios],
            "scenario_names": [s.name for s, _ in scenarios],
            "rain_scales": [r for _, r in scenarios], "n_members": n_mem,
            "member_labels": [m.label for m in members], "hindcast_r2": hindcast_r2,
            "rows": pd.DataFrame(fan_mean_rows),
            **({"members_yearly": {**mem_yr, "ye_idx": ye_idx,
                                   "years": [int(all_dates[i].year) for i in ye_idx]}}
               if mem_yr is not None else {}),
            **({"delay_u_mean": (delay_u_sum / n_mem).astype("float32")}
               if delay_u_sum is not None and delay_u_sum.any() else {})}


SUBS_Q_M = 0.001        # sidecar quantum for subsidence: 1 mm (int16: +-32 m)
HEAD_Q_M = 0.01         # ... for layer-2 head: 1 cm (int16: +-327 m)


def write_members_sidecar(path: str, my: dict, scenario_names: list[str], origin: int,
                          dates) -> dict:
    """``--save-members yearly`` -> ``<out>.members.npz`` (app spec §3.4).

    ``subs_members_yr`` has shape (S, M*R, A, Y) and ``headL2_members_yr`` has shape
    (S, M, A, Y). Both are int16: subsidence in units of ``subs_scale_m`` (1 mm) and head
    in units of ``head_scale_m`` (1 cm), at December of each year. This is the "quantised"
    form the spec allows next to float16. At these magnitudes it is finer than float16
    (float16 has a 6 cm step at 100 m of head) and the same size. ``member``/``ic``/
    ``rheology`` label the subsidence axis, and ``head_member``/``head_ic`` label the head
    axis. Returns the saturation counts, which should be zero."""
    s = np.asarray(my["subs"], dtype="float64") / SUBS_Q_M
    h = np.asarray(my["headL2"], dtype="float64") / HEAD_Q_M
    lim = np.iinfo("int16").max
    sat = {"subs": int((np.abs(s) > lim).sum()), "head": int((np.abs(h) > lim).sum())}
    np.savez_compressed(
        path,
        subs_members_yr=np.clip(np.rint(s), -lim, lim).astype("int16"),
        headL2_members_yr=np.clip(np.rint(h), -lim, lim).astype("int16"),
        subs_scale_m=SUBS_Q_M, head_scale_m=HEAD_Q_M,
        years=np.asarray(my["years"], dtype="int64"), ye_idx=np.asarray(my["ye_idx"]),
        member=np.array(my["member"]), ic=np.asarray(my["ic"], dtype="int64"),
        rheology=np.array(my["rheology"]), head_member=np.array(my["head_member"]),
        head_ic=np.asarray(my["head_ic"], dtype="int64"),
        scenario_names=np.array(scenario_names), origin=int(origin),
        dates=np.array([d.isoformat() for d in dates]))
    return sat


def load_members_sidecar(path: str) -> dict:
    """Read ``<out>.members.npz`` back to float cm (subsidence) and m (layer-2 head)."""
    z = np.load(path, allow_pickle=False)
    return {"subs_cm": z["subs_members_yr"].astype("float32") * float(z["subs_scale_m"]) * 100.0,
            "headL2_m": z["headL2_members_yr"].astype("float32") * float(z["head_scale_m"]),
            "years": [int(y) for y in z["years"]], "ye_idx": z["ye_idx"],
            "member": [str(v) for v in z["member"]], "ic": z["ic"],
            "rheology": [str(v) for v in z["rheology"]],
            "head_member": [str(v) for v in z["head_member"]], "head_ic": z["head_ic"],
            "scenario_names": [str(v) for v in z["scenario_names"]],
            "origin": int(z["origin"])}


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
                         "months (0 = plain hindcast, the default). The hindcast R2 "
                         "printed per member is then an assimilated score, not a "
                         "free-running one. WARNING (2026-09-19): the compaction column "
                         "is calibrated on the FREE-RUNNING hindcast, so nudged heads are "
                         "out of its distribution and subsidence skill falls (measured: "
                         "leveling R2 0.599 -> 0.391). Use it only with a column refit on "
                         "nudged heads.")
    ap.add_argument("--hindcast-every", type=int, default=12)
    ap.add_argument("--ic-members", type=int, default=0,
                    help="extra members per parameter set, each restarting from the "
                         "observed field perturbed by N(0, --ic-sigma) per well")
    ap.add_argument("--ic-sigma", type=float, default=0.5, help="metres")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vep-json", default="results/twin/stage2_vep_shared.json",
                    help="shared Stage-2 column parameters; fitted and written if missing. "
                         "Comma-separated files make a rheology axis: every member runs "
                         "through every column (e.g. tau ceilings 11 and 30 years) and the "
                         "subsidence spread pools them")
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
    ap.add_argument("--dump-delay-state", action="store_true",
                    help="write the delay bed's slow-store head (ensemble mean per "
                         "scenario, hindcast + projection) as delay_u_mean in <out>.npz. "
                         "CAVEAT: the column is still driven by the aquifer heads; the "
                         "bed's drainage is clay compaction the column also represents, "
                         "so do not add the two")
    # --- opt-in artefact fixes (2026-09-23); every default reproduces the runs before ---
    ap.add_argument("--restart-taper-km", type=float, default=None,
                    help="A2: taper the origin restart by exp(-(d/R)^2), where d is the "
                         "distance to the nearest well of the same layer with a head in "
                         "the origin month. Without it, the restart replaces whole layers "
                         "with the IDW field, and the unmeasured proximal layers 3-4 take "
                         "heads from distant wells")
    ap.add_argument("--column-heads", choices=("restart", "free"), default="restart",
                    help="A2: which projection drives the compaction column. 'restart' "
                         "(default) uses the restarted heads that are also displayed. "
                         "'free' uses a projection from the hindcast's own end state plus "
                         "the ic perturbation, which is the trajectory the column was "
                         "calibrated on. It costs one more rollout per member x ic x "
                         "scenario")
    ap.add_argument("--column-hpc0-fast-days", type=float, default=None,
                    help="A1: set h_pc0 to 0 in every column whose creep tau is below this "
                         "many days (365 affects only the proximal column, tau 24 d). This "
                         "removes the ~121 cm per cell start-up load that leveling cannot "
                         "see, and needs no refit (in-sample leveling R2 -0.0003)")
    ap.add_argument("--column-zone-blend-km", type=float, default=None,
                    help="A3 illustration: blend a per-zone column over this width across "
                         "the mid/distal line even though it was fitted sharp. Parameters "
                         "are NOT refitted; for the real fix use calibrate_coupled "
                         "--zone-blend-km, whose JSON records its width and is read "
                         "automatically")
    ap.add_argument("--save-members", choices=("none", "yearly"), default="none",
                    help="'yearly': write <out>.members.npz with every member's December "
                         "fields of subsidence and layer-2 head, quantised to int16 (app "
                         "spec 3.4)")
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
                           cap_duty=float(meta.get("cap_duty", 1.0)),
                           **input_options(meta))
    print(f"census cleaning as calibrated: meter_filter={meta.get('meter_filter', 'none')}",
          flush=True)
    attach_sw_recharge(inp, meta, log=lambda m: print(m, flush=True))

    cfg = Config(data_dir=args.data) if args.data else Config()
    ddir = str(cfg.data_dir) if cfg.data_dir and os.path.isdir(str(cfg.data_dir)) else None
    from .zones import fan_zones as _fan_zones

    # the column is three-zone even over a flow model with the proximal split
    zb = _parse_zone_boundaries(meta.get("zone_boundaries", "205,182"), allow_split=True)[:2]
    zoc = _fan_zones(inp.grid.centroids(), *zb)
    cent = inp.grid.centroids()
    columns = load_columns(args.vep_json, ddir, inp.hf, device, epochs=args.vep_epochs,
                           zone_of_cell=zoc,
                           zone_weights=lambda km: zone_blend_weights(cent, *zb, km),
                           hpc0_fast_days=args.column_hpc0_fast_days,
                           blend_km=args.column_zone_blend_km)
    for rlabel, _, vep in columns:
        if vep.get("hpc0_released", {}).get("already_clean"):
            print(f"A1 fix [{rlabel}]: no column with tau < "
                  f"{vep['hpc0_released']['tau_days_below']:g} d has h_pc0 > 0 "
                  "(already clean, nothing released)", flush=True)
        elif vep.get("hpc0_released"):
            print(f"A1 fix [{rlabel}]: h_pc0 set to 0 in column(s) "
                  f"{vep['hpc0_released']['columns']} (tau < "
                  f"{vep['hpc0_released']['tau_days_below']:g} d)", flush=True)
        if vep.get("zone_blend_km_override"):
            print(f"WARNING [{rlabel}]: column blended over "
                  f"{vep['zone_blend_km_override']['used']:g} km across the mid/distal line "
                  "but NOT refitted -- an illustration, not a calibrated column", flush=True)
        elif float(vep.get("zone_blend_km") or 0.0) > 0.0:
            print(f"VEP column [{rlabel}]: blended over {float(vep['zone_blend_km']):g} km "
                  "(as fitted)", flush=True)
        if "zonal" in vep:
            print(f"VEP column [{rlabel}]: per-zone (" + "; ".join(
                f"{z}: Ske={math.exp(q['log_ske']):.2e} Skv={math.exp(q['log_skv']):.2e} "
                f"tau={math.exp(q['log_tau']):.0f} d"
                for z, q in zip(("prox", "mid", "dist"), vep["zonal"], strict=True))
                + ")", flush=True)
        else:
            print(f"VEP column [{rlabel}]: Ske={math.exp(vep['log_ske']):.3e} "
                  f"Skv={math.exp(vep['log_skv']):.3e} tau={math.exp(vep['log_tau']):.0f} d",
                  flush=True)
    col = columns[0][1] if len(columns) == 1 else [(c[0], c[1]) for c in columns]

    t0 = time.perf_counter()
    res = run(inp, members, scenarios, args.horizon, args.gain, args.ic_members,
              args.ic_sigma, args.seed, col, device, pump_layer=args.pump_layer,
              recharge_layer=args.recharge_layer, hindcast_gain=args.hindcast_gain,
              hindcast_every=args.hindcast_every, dump_delay_state=args.dump_delay_state,
              restart_taper_km=args.restart_taper_km, column_heads=args.column_heads,
              save_members=args.save_members)
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
        if len(res["rheology_labels"]) > 1:
            for rlabel, sub in zip(res["rheology_labels"], res["subs_mean_by_rheology"],
                                   strict=True):
                sk = validate_against_leveling(ddir, inp.grid, sub[0][:, :T], inp.dates)
                if sk["n_sites"]:
                    print(f"  rheology {rlabel}: leveling R2 {sk['r2']:+.3f} RMSE "
                          f"{sk['rmse_cm']:.1f} cm bias {sk['bias_cm']:+.1f} cm", flush=True)

    rows = res.pop("rows")
    if len(res["rheology_labels"]) > 1:
        print("\n=== rheology axis (fan-mean forward subsidence at the horizon, cm) ===")
        by_r = rows.groupby(["scenario", "rheology"])["subs_forward_cm"].agg(["mean", "std"])
        for (name, rlabel), r in by_r.iterrows():
            print(f"  {name:>12} [{rlabel}]: {r['mean']:.2f} ± {np.nan_to_num(r['std']):.2f}")
    summary = (rows.drop(columns=["member", "ic", "rheology"], errors="ignore")
               .groupby("scenario").agg(["mean", "std"]))
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
        rheology_labels=np.array(res["rheology_labels"]),
        **({"subs_mean_by_rheology": res["subs_mean_by_rheology"]}
           if len(res["rheology_labels"]) > 1 else {}),
        **({"delay_u_mean": res["delay_u_mean"]} if "delay_u_mean" in res else {}),
        gate=json.dumps(meta), horizon=args.horizon, gain=args.gain,
        hindcast_gain=args.hindcast_gain, hindcast_every=args.hindcast_every,
        # which artefact fixes (2026-09-23) this run used; the app reads it
        forward_options=json.dumps({
            "restart_taper_km": args.restart_taper_km, "column_heads": args.column_heads,
            "column_hpc0_fast_days": args.column_hpc0_fast_days,
            "hpc0_released": [c[2].get("hpc0_released") for c in columns],
            "hpc0_guard_days": [c[2].get("hpc0_guard_days") for c in columns],
            "column_zone_blend_km": [
                (c[2]["zone_blend_km_override"]["used"] if c[2].get("zone_blend_km_override")
                 else float(c[2].get("zone_blend_km") or 0.0)) for c in columns],
            "save_members": args.save_members}))
    rows.to_csv(args.out + ".members.csv", index=False)
    summary.to_csv(args.out + ".summary.csv")
    print(f"wrote {args.out}.npz, {args.out}.members.csv, {args.out}.summary.csv")
    if "members_yearly" in res:
        sat = write_members_sidecar(args.out + ".members.npz", res["members_yearly"],
                                    res["scenario_names"], res["origin"], res["dates"])
        print(f"wrote {args.out}.members.npz (per-member December fields"
              + (f"; SATURATED int16 values: {sat}" if any(sat.values()) else "") + ")")


if __name__ == "__main__":
    main()
