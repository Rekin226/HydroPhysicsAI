"""The policy-response gate and the candidate scorecard (G3).

    python -m hydrophysics.twin.policy_gate --theta RUN/stage3_theta.json \\
        [--vep-json RUN/coupled_leveling/vep_zonal_leveling.json] \\
        [--temporal RUN_HOLDOUT/stage3_temporal.csv] --out RUN/scorecard.json

The held-out-well gate measures spatial interpolation under the recorded forcing and is
blind to whether the forcing does any work: on 2026-09-20 a 21 km stress spread won it by
0.006 and could not tell a 30 % irrigation cut from doing nothing (STATE §3.3). A twin
whose purpose is policy response has to be scored on it. This module does that, and
merges every verdict a candidate has into one JSON the orchestrator ranks on:

- **head gate**: k-fold R2 against IDW, from the run's ``stage3_flow.csv`` or theta meta;
- **temporal verdict**: held-out years, from ``--temporal`` or theta meta
  (``calibrate_flow.temporal_verdict``);
- **policy response** (here): one member, free-running hindcast (no nudging, so the
  response is the model's own), then ``horizon`` months under ``irrigation=level`` for
  each level, compacted by the candidate's column. PASS needs all of
  ``0 < dh2(0.85) < dh2(0.7)`` (heads recover, monotonically),
  ``ds(0.7) < ds(0.85) < 0`` (less subsidence, monotonically),
  ``dh2(0.7) >= min_dh`` and ``|ds(0.7)| / s_base >= min_rel_ds``;
- **leveling hindcast**: the same baseline run's subsidence against the leveling network,
  when the data dir is available.

The column pairs with its head field (STATE §3.3): pass the candidate's own refit column
with ``--vep-json``. Without it the deliverable's column is used and the scorecard says
``column: reference (proxy)``.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .calibrate_flow import (
    _parse_zone_boundaries,
    temporal_verdict,  # noqa: F401  (re-exported for the gate)
)
from .forward import (
    N_LAYERS,
    build_model,
    compaction,
    future_forcing,
    future_sw,
    hindcast_with_nudging,
    load_members,
    load_or_fit_vep,
    rollout,
    sw_hist,
)
from .scenario import PumpingScenario
from .zones import zone_blend_weights

REFERENCE_VEP = "results/twin_runs/stage3_spreadL_gate/coupled_leveling/vep_zonal_leveling.json"
LEVELS = (1.0, 0.85, 0.7)
# The deliverable's own response under this module's protocol (one member, free-running,
# 120 months, its own column; measured 2026-09-23): the reference of the pre-registered
# non-inferiority rule (``--min-response-frac``). At 0.5 the floor is -0.62 cm, which the
# 21 km model's -0.44 cm fails and the deliverable passes.
REFERENCE_RESPONSE = {"ds_m": {"0.7": -0.0125}, "s_base_m": 0.0478,
                      "source": "stage3_spreadL_gate (10 km), free-running, own column"}


def rank_key(card: dict) -> list:
    """The orchestrator's ranking key, to sort DESCENDING: head gate PASS, temporal
    verdict PASS, policy PASS (with the non-inferiority rule when it was set), leveling
    chain R2 (out of fold; the free-running hindcast here), then the temporal RMSE ratio
    ascending (negated). Missing pieces rank last within their slot."""
    def _pass(d):
        return 1 if isinstance(d, dict) and d.get("verdict") == "PASS" else 0

    pr = card.get("policy_response") or {}
    lev = (pr.get("leveling_hindcast") or {}).get("r2")
    tmp = card.get("temporal") or {}
    ratio = tmp.get("rmse_ratio")
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = float("inf")
    return [_pass(card.get("head_gate")), _pass(tmp), _pass(pr),
            float(lev) if lev is not None else float("-inf"), -ratio]


def policy_verdict(dh2: dict, ds: dict, s_base: float, min_dh: float = 0.5,
                   min_rel_ds: float = 0.03, min_abs_ds: float = 0.0,
                   min_response_frac: float = 0.0,
                   reference_ds70: float | None = None) -> dict:
    """The pass rule on the three-level response. ``dh2``/``ds`` map level -> change
    versus the baseline (level 1.0) in fan-mean layer-2 head (m) and fan-mean forward
    subsidence (m); ``s_base`` is the baseline's fan-mean forward subsidence (m).

    ``min_abs_ds`` (m, default 0 = off) adds ``|ds(0.7)| >= min_abs_ds``. Measured
    2026-09-23, free-running, 120 months, each model with its own column: the 10 km
    deliverable scores ds(0.7) -1.25 cm (26 % of a 4.78 cm baseline) and the 21 km model
    -0.44 cm (11 % of 3.92 cm), so BOTH pass the relative rule; the free-running
    baseline is far smaller than the nudged ensemble's, which inflates the ratio. An
    absolute floor would separate them but would be set after seeing two points; it is
    left off so the pre-registered rule stands, and reported instead.

    ``min_response_frac`` (default 0 = off) is the non-inferiority rule proposed for the
    candidates that follow: ``|ds(0.7)| >= frac x |ds_ref(0.7)|``, with the reference the
    deliverable's own value under the same protocol (``REFERENCE_RESPONSE``). It is
    pre-registered against the deliverable, not tuned between two candidates."""
    d85, d70 = float(dh2[0.85]), float(dh2[0.7])
    s85, s70 = float(ds[0.85]), float(ds[0.7])
    rel = abs(s70) / abs(s_base) if s_base else float("inf")
    checks = {"heads_recover_monotonically": bool(0.0 < d85 < d70),
              "subsidence_falls_monotonically": bool(s70 < s85 < 0.0),
              "head_response_large_enough": bool(d70 >= min_dh),
              "subsidence_response_large_enough": bool(rel >= min_rel_ds)}
    if min_abs_ds > 0:
        checks["subsidence_response_absolute"] = bool(abs(s70) >= min_abs_ds)
    ref70 = (float(REFERENCE_RESPONSE["ds_m"]["0.7"]) if reference_ds70 is None
             else float(reference_ds70))
    if min_response_frac > 0:
        checks["non_inferior_to_reference"] = bool(abs(s70) >= min_response_frac * abs(ref70))
    return {"dh2_m": {str(k): float(v) for k, v in dh2.items()},
            "ds_m": {str(k): float(v) for k, v in ds.items()},
            "s_base_m": float(s_base), "rel_ds": float(rel), "min_dh_m": float(min_dh),
            "min_rel_ds": float(min_rel_ds), "min_abs_ds_m": float(min_abs_ds),
            "min_response_frac": float(min_response_frac), "reference_ds70_m": ref70,
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL"}


def policy_response(inp, member, vep_json: str | None = None, levels=LEVELS,
                    horizon: int = 120, device=None, min_dh: float = 0.5,
                    min_rel_ds: float = 0.03, min_abs_ds: float = 0.0,
                    min_response_frac: float = 0.0,
                    pump_layer: int = 1, recharge_layer: int = 0,
                    ddir: str | None = None, log=print) -> dict:
    """Run the three-level irrigation response for one member (see the module doc)."""
    levels = tuple(sorted({1.0, *map(float, levels)}, reverse=True))
    model, scalars, zoc = build_model(inp.grid, member, device)
    proxy = not vep_json
    vep_path = vep_json or REFERENCE_VEP
    # a column refitted with --zone-blend-km rebuilds its per-cell parameters from the
    # blend weights on the run's own zone lines
    # the column is three-zone even over a flow model with the proximal split
    zb = _parse_zone_boundaries(member.meta.get("zone_boundaries", "205,182"),
                                allow_split=True)[:2]
    cent = inp.grid.centroids()
    col, _ = load_or_fit_vep(vep_path, None, inp.hf, device, zone_of_cell=zoc,
                             zone_weights=lambda km: zone_blend_weights(cent, *zb, km))
    eta_classes = member.meta.get("eta_classes")
    if eta_classes:
        E_hist = torch.tensor(np.stack([inp.E_by_class[c] for c in eta_classes]),
                              dtype=torch.float64)[..., 1:]
    else:
        E_hist = inp.E_total[:, 1:]
    sw_rec = sw_hist(inp)
    h0 = inp.initial_heads(0, n_layers=N_LAYERS)
    t0 = time.perf_counter()
    h_hist, u_end = hindcast_with_nudging(model, scalars, inp, h0, E_hist,
                                          inp.recharge_field[:, 1:], 0.0, 0,
                                          pump_layer=pump_layer, recharge_layer=recharge_layer,
                                          sw=sw_rec, return_state=True)
    origin = len(inp.dates) - 1
    end_h2, fwd_s, base_subs = {}, {}, None
    for lev in levels:
        scen = PumpingScenario(f"irrigation{lev:g}", factors={"irrigation": lev})
        E_fut, r_fut, fut_dates = future_forcing(inp, scen, horizon, zone_of_cell=zoc,
                                                 eta_classes=eta_classes)
        # month0: the projection's own first month (river-stage season), as forward.run
        h_fut = rollout(model, scalars, h_hist[..., -1], E_fut, r_fut, inp.ground_elev,
                        pump_layer=pump_layer, recharge_layer=recharge_layer,
                        sw_field=future_sw(inp, scen, horizon, zoc), u0=u_end,
                        month0=fut_dates[0].month - 1)
        heads = torch.cat([h_hist, h_fut[..., 1:]], dim=-1)
        subs = compaction(col, heads)
        end_h2[lev] = float(heads[1, :, -1].mean())
        fwd_s[lev] = float((subs[:, -1] - subs[:, origin]).mean())
        if lev == 1.0:
            base_subs = subs
        log(f"  irrigation x{lev:g}: layer-2 head at the horizon {end_h2[lev]:+.2f} m, "
            f"forward subsidence {100 * fwd_s[lev]:.2f} cm")
    dh2 = {lev: end_h2[lev] - end_h2[1.0] for lev in levels if lev != 1.0}
    ds = {lev: fwd_s[lev] - fwd_s[1.0] for lev in levels if lev != 1.0}
    out = policy_verdict(dh2, ds, fwd_s[1.0], min_dh=min_dh, min_rel_ds=min_rel_ds,
                         min_abs_ds=min_abs_ds, min_response_frac=min_response_frac)
    out.update({"column": "reference (proxy)" if proxy else "candidate",
                "vep_json": vep_path, "horizon_months": int(horizon),
                "levels": list(levels), "member": member.label,
                "time_s": time.perf_counter() - t0})
    if ddir is not None and base_subs is not None:
        from .explorer3d import validate_against_leveling

        T = len(inp.dates)
        sk = validate_against_leveling(ddir, inp.grid, base_subs[:, :T], inp.dates)
        if sk.get("n_sites"):
            out["leveling_hindcast"] = {k: sk[k] for k in ("r2", "rmse_cm", "bias_cm",
                                                           "n_sites", "n_pairs")}
    return out


def _head_gate(theta_path: str, meta: dict) -> dict | None:
    csv = os.path.join(os.path.dirname(theta_path), "stage3_flow.csv")
    if os.path.exists(csv):
        row = pd.read_csv(csv).iloc[-1]
        return {"r2_kfold": float(row["r2_kfold"]), "r2_idw": float(row["r2_idw"]),
                "margin": float(row["r2_kfold"]) - float(row["r2_idw"]),
                "verdict": "PASS" if row["r2_kfold"] > row["r2_idw"] else "FAIL",
                "source": csv}
    return meta.get("gate")


def _temporal(path: str | None, meta: dict) -> dict | None:
    if path and os.path.exists(path):
        return {**pd.read_csv(path).iloc[-1].to_dict(), "source": path}
    return meta.get("temporal_gate")


def main(argv=None) -> None:
    from ..config import Config
    from ..train import pick_device
    from .forward import attach_sw_recharge
    from .inputs import input_options, load_twin_inputs

    ap = argparse.ArgumentParser(description="policy-response gate and candidate scorecard")
    ap.add_argument("--theta", required=True, help="the candidate's stage3_theta.json")
    ap.add_argument("--vep-json", default=None,
                    help="the candidate's own refit column (default: the deliverable's, "
                         "labelled a proxy)")
    ap.add_argument("--temporal", default=None,
                    help="stage3_temporal.csv of the candidate's held-out-years run")
    ap.add_argument("--hindcast", default=None,
                    help="optional forward-run summary to cite in the scorecard")
    ap.add_argument("--levels", default="1.0,0.85,0.7")
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--policy-min-dh", type=float, default=0.5)
    ap.add_argument("--policy-min-rel-ds", type=float, default=0.03)
    ap.add_argument("--policy-min-abs-ds-cm", type=float, default=0.0,
                    help="optional absolute floor on |ds(0.7)| in cm (default off; see "
                         "policy_verdict for why)")
    ap.add_argument("--min-response-frac", type=float, default=0.0,
                    help="non-inferiority: |ds(0.7)| >= this x the deliverable's "
                         "(REFERENCE_RESPONSE, -1.25 cm); proposed 0.5; default 0 = off")
    ap.add_argument("--device", default=None)
    ap.add_argument("--data", default=None, help="data dir for the leveling hindcast "
                                                 "(else HYDROMIND_GW_DATA)")
    ap.add_argument("--out", default=None,
                    help="scorecard JSON (default: scorecard.json next to --theta)")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    members = load_members([args.theta])
    mem = members[0]
    meta = mem.meta
    print(f"policy gate: {args.theta} (device {device})", flush=True)
    inp = load_twin_inputs(dx=float(meta.get("dx", 1000.0)),
                           meter_filter=meta.get("meter_filter", "none"),
                           cap_duty=float(meta.get("cap_duty", 1.0)),
                           **input_options(meta))
    attach_sw_recharge(inp, meta)
    cfg = Config(data_dir=args.data) if args.data else Config()
    ddir = str(cfg.data_dir) if cfg.data_dir and os.path.isdir(str(cfg.data_dir)) else None
    pr = policy_response(inp, mem, args.vep_json,
                         levels=tuple(float(x) for x in args.levels.split(",")),
                         horizon=args.horizon, device=device, min_dh=args.policy_min_dh,
                         min_rel_ds=args.policy_min_rel_ds,
                         min_abs_ds=args.policy_min_abs_ds_cm / 100.0,
                         min_response_frac=args.min_response_frac,
                         pump_layer=int(meta.get("pump_layer", 1)),
                         recharge_layer=int(meta.get("recharge_layer", 0)), ddir=ddir)
    card = {"theta": args.theta, "git_commit": meta.get("git_commit"),
            "options": {k: meta.get(k) for k in ("delay_storage", "rivers", "river_set",
                                                 "sw_recharge", "learn_spread", "fix_eta",
                                                 "fix_head_extra", "return_flow",
                                                 "pump_split", "l_min", "holdout_months",
                                                 "delay_u0", "delay_tau_min_days",
                                                 "delay_layers", "aquitard_storage",
                                                 "sw_components", "river_c_split",
                                                 "river_skip_apex", "spread_max_km")},
            "head_gate": _head_gate(args.theta, meta),
            "temporal": _temporal(args.temporal, meta),
            "policy_response": pr, "hindcast_source": args.hindcast}
    # a proxy column breaks "the column pairs with its head field": such a card is a
    # screen, not a ranking; the orchestrator refits the column for head+temporal passers
    card["ranking_eligible"] = pr.get("column") != "reference (proxy)"
    card["rank_key"] = rank_key(card)
    print(f"POLICY RESPONSE: {pr['verdict']} -- dh2(0.7) {pr['dh2_m'].get('0.7', 0):+.2f} m, "
          f"ds(0.7) {100 * pr['ds_m'].get('0.7', 0):+.2f} cm, rel {100 * pr['rel_ds']:.1f} % "
          f"of the baseline's {100 * pr['s_base_m']:.2f} cm; column: {pr['column']}",
          flush=True)
    if "leveling_hindcast" in pr:
        lh = pr["leveling_hindcast"]
        print(f"leveling hindcast (this member, free-running): R2 {lh['r2']:+.3f}, "
              f"RMSE {lh['rmse_cm']:.1f} cm, bias {lh['bias_cm']:+.1f} cm", flush=True)
    out = args.out or os.path.join(os.path.dirname(args.theta), "scorecard.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(card, fh, indent=1, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
