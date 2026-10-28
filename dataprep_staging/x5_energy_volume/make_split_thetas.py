"""I1.X5R1.v2_postfailure: write stage3_datum_gate_<arm>_cs/stage3_theta.json for arms q0,
q0c, x5r1 (every zonal flow key duplicated into its Changhua copy, all copies tied, the
county-split meta of the default county map), then check that the hindcast heads are those of
the unsplit theta (max |dh| < 1e-6 m) before any column is fitted. Refuses existing dirs.

Usage (repo root): results/twin_runs/par28/py.sh dataprep_staging/x5_energy_volume/make_split_thetas.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from hydrophysics.twin.calibrate_coupled import (
    N_LAYERS,
    attach_sw_recharge,
    build_model,
    extra_hist,
    input_options,
    load_members,
    load_twin_inputs,
    rollout,
    sw_hist,
)
from hydrophysics.twin.calibrate_flow import _key_zone
from hydrophysics.twin.zone_county import (
    COUNTY_SUFFIX,
    DEFAULT_MAP,
    county_meta,
    verify_county_cache,
)

RUNS = Path("results/twin_runs")
ARMS = ("q0", "q0c", "x5r1")
BASE_ZONES = ("proximal", "mid", "distal", "proximal_w")
TOL_M = 1e-6


def split_theta(src: dict, flags: np.ndarray) -> dict:
    th, meta = dict(src["theta"]), dict(src["meta"])
    for k in list(th):
        if _key_zone(k) in BASE_ZONES:
            th[f"{k}{COUNTY_SUFFIX}"] = th[k]
    meta.update(county_meta(DEFAULT_MAP, flags))
    meta["zone_county_tied"] = list(BASE_ZONES)
    meta["split_note"] = ("I1.X5R1.v2_postfailure: unsplit flow theta with tied Changhua "
                          "copies, for a six-zone column on unchanged heads")
    return {"theta": th, "meta": meta}


def hindcast(theta_path: Path, device) -> np.ndarray:
    member = load_members([str(theta_path)])[0]
    inp = load_twin_inputs(meter_filter=member.meta.get("meter_filter", "none"),
                           cap_duty=float(member.meta.get("cap_duty", 1.0)),
                           **input_options(member.meta))
    model, scalars, _ = build_model(inp.grid, member, device)
    eta_classes = member.meta.get("eta_classes")
    E = (torch.tensor(np.stack([inp.E_by_class[c] for c in eta_classes]),
                      dtype=torch.float64)[..., 1:] if eta_classes else inp.E_total[:, 1:])
    attach_sw_recharge(inp, member.meta)
    return rollout(model, scalars, inp.initial_heads(0), E, inp.recharge_field[:, 1:],
                   inp.ground_elev, sw_field=sw_hist(inp), q_extra=extra_hist(inp)).cpu().numpy()


def main() -> int:
    device = torch.device("cuda")
    out = {}
    for arm in ARMS:
        src_path = RUNS / f"stage3_datum_gate_{arm}/stage3_theta.json"
        dst_dir = RUNS / f"stage3_datum_gate_{arm}_cs"
        if dst_dir.exists():
            raise SystemExit(f"{dst_dir} exists")
        src = json.loads(src_path.read_text())
        if src["meta"].get("zone_county_split"):
            raise SystemExit(f"{src_path} is already county-split")
        h0 = hindcast(src_path, device)
        inp = load_twin_inputs(meter_filter=src["meta"].get("meter_filter", "none"),
                               cap_duty=float(src["meta"].get("cap_duty", 1.0)),
                               **input_options(src["meta"]))
        flags = verify_county_cache(inp.grid, DEFAULT_MAP)
        dst_dir.mkdir()
        dst = dst_dir / "stage3_theta.json"
        dst.write_text(json.dumps(split_theta(src, flags), indent=1) + "\n")
        h1 = hindcast(dst, device)
        dh = float(np.nanmax(np.abs(h1 - h0)))
        nan_same = bool(np.array_equal(np.isnan(h0), np.isnan(h1)))
        out[arm] = {"max_abs_dh_m": dh, "nan_pattern_same": nan_same, "shape": list(h0.shape),
                    "n_layers": N_LAYERS, "ok": bool(dh < TOL_M and nan_same)}
        print(arm, out[arm], flush=True)
        if not out[arm]["ok"]:
            raise SystemExit(f"{arm}: heads differ (max |dh| {dh:g} m); not a like-for-like "
                             "column refit")
    (RUNS / "par27f").mkdir(exist_ok=True)
    (RUNS / "par27f/heads_identity.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
