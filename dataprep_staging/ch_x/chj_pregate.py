"""CH-J engineering pre-gate (I1.CHX.v1 secondary; chj_forcing.CALIBRATE_FLOW_PATCH).

With every Changhua copy at Q0's value (the copies start at their parents' and no step is
taken before the epoch-0 rollout), the Changhua layer shares set to Q0's own placement
(``q0_rest_fractions``), and the surface-water field identically zero (four county-split
components, as --sw-components both --sw-county-split builds them), the PATCHED
``calibrate_flow.fit_flow``'s epoch-0 rollout (the first ``_rollout`` of its training loop,
--freeze-from Q0 --pump-layer-shares-county) must reproduce Q0's rollout to
``chx_const.CHJ_FREEZE_REPRO_TOL_M`` (1e-6 m) at every layer, cell and month.

Q0's rollout is the existing evaluation path ``calibrate_flow._predict_homogeneous`` on a
model carrying Q0's own parameters (the unsplit zonation, Q0's boundary conductances,
pump_split, return flow, spread radius, fixed eta and head_extra): code the hook does not
touch.

Model inputs only: the fan grid (``DEFAULT_PATHS['polygon']``, dx 1 km), R1c's county map
(zone_county_map v2, verified against a rebuild from the county polygons), Q0's theta and
meta pump_layer. Every head is synthetic (a smooth initial field, synthetic targets that
only feed the loss); the forcing is synthetic too (energy, recharge, ground elevation, seeded).
No observed head, leveling or township value is opened.

Beside the gate: a sensitivity control (the registered X3 shares in Changhua, same pipeline;
it must move the comparison by more than 100 x the tolerance, which the gate also requires,
or an exact zero could mean the comparison never sees a placement), and, report-only, the
same comparison for the rollout rebuilt from the fit's theta file form
(``_predict_homogeneous`` on the fit result: the county shares as pump_layer_logits /
pump_layer_logits_ch), the path k-fold scoring, forward.build_model and calibrate_coupled
use.

    python dataprep_staging/ch_x/chj_pregate.py --out <result.json> [--months 131]

The result JSON is written once (mode 'x'); its ``pregate`` block ({passed,
max_abs_diff_m, tol_m, ...}) is what the lead copies into chj_engineering.json. Its
``pregate_sha256`` is this script's hash; the record's ``producer_sha256`` is the panel
producer's (chj_produce.py), never this one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chx_const as K  # noqa: E402

HOOK = "hydrophysics/twin/calibrate_flow.py"
ZONE_BOUNDARIES = (205.0, 182.0, 208.0)      # Q0's meta zone_boundaries '205,182,208'


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def synthetic_inputs(grid, n_months: int, seed: int):
    """Seeded synthetic heads and forcing on ``grid`` (no data read)."""
    import torch

    D = torch.float64
    rng = np.random.default_rng(seed)
    xy = grid.centroids()
    A = grid.n_active
    xn = (xy[:, 0] - xy[:, 0].min()) / max(np.ptp(xy[:, 0]), 1.0)     # 0 coast .. 1 apex
    ground = 5.0 + 95.0 * xn
    h0 = np.stack([ground - 3.0 - 2.0 * k - 4.0 * (1.0 - xn) for k in range(4)])
    season = 1.0 + 0.6 * np.sin(2.0 * np.pi * np.arange(n_months) / 12.0)
    E = rng.lognormal(np.log(4e3), 0.8, A)[:, None] * season[None, :]          # kWh/month
    rf = np.clip(rng.normal(8e-4, 3e-4, (A, n_months)), 0.0, None)              # m/day
    n_w = 40
    idx = rng.choice(A, n_w, replace=False)
    lay = rng.integers(0, 4, n_w)
    obs = h0[lay, idx][:, None] + rng.normal(0.0, 0.5, (n_w, n_months))
    return {"h0": torch.tensor(h0, dtype=D), "E": torch.tensor(E, dtype=D),
            "rf": torch.tensor(rf, dtype=D), "ground": torch.tensor(ground, dtype=D),
            "idx": torch.tensor(idx, dtype=torch.long),
            "lay": torch.tensor(lay, dtype=torch.long),
            "obs": torch.tensor(obs, dtype=D)}


def run_pregate(q0_theta: dict, q0_meta: dict, grid, changhua: np.ndarray,
                n_months: int = 131, seed: int = 20261103, device: str = "cpu",
                boundaries=None) -> dict:
    """Max |patched epoch-0 rollout - Q0's rollout| over (layer, cell, month 0..n)."""
    import torch

    from hydrophysics.twin import calibrate_flow as CF
    from hydrophysics.twin.flow import FlowModel
    from hydrophysics.twin.zones import county_split, fan_zones, zone_labels

    D = torch.float64
    pump_layer = int(q0_meta["pump_layer"])
    syn = synthetic_inputs(grid, n_months, seed)
    A = grid.n_active
    z = fan_zones(grid.centroids(), *ZONE_BOUNDARIES[:2], split_km=ZONE_BOUNDARIES[2])
    zc = county_split(z, changhua, 4)
    ch_t = torch.as_tensor(np.asarray(changhua, bool))

    # Q0's rollout: the existing evaluation path on a model carrying Q0's parameters
    th = {k: torch.as_tensor(np.asarray(v, dtype="float64"), dtype=D)
          for k, v in q0_theta.items() if isinstance(v, (int, float, list))}
    for k in list(th):
        if k.startswith(("log_T_", "log_S_")) and th[k].dim() == 1:
            th[k] = th[k].reshape(-1, 1)
        if k.startswith("log_L_"):
            th[k] = th[k].reshape(-1, 1)
    m_ref = FlowModel(grid, n_layers=4, dt_days=30.0, device=device, boundaries=boundaries)
    with torch.no_grad():
        lT, lS, lL = CF._expand_zonal(th, torch.as_tensor(z, dtype=torch.long),
                                      m_ref.n_layers)
        m_ref.log_T.copy_(lT)
        m_ref.log_S.copy_(lS)
        m_ref.log_L.copy_(lL)
        if m_ref.has_boundaries:
            m_ref.log_C_coast.copy_(th["log_C_coast"].reshape(m_ref.log_C_coast.shape))
            m_ref.log_C_apex.copy_(th["log_C_apex"].reshape(m_ref.log_C_apex.shape))
    m_ref.set_apex_heads(syn["h0"])
    ref = CF._predict_homogeneous(m_ref, {"theta": q0_theta}, syn["h0"], n_months,
                                  recharge_field=syn["rf"], E=syn["E"],
                                  ground_elev=syn["ground"], recharge_layer=0,
                                  pump_layer=pump_layer).cpu()

    # the patched fit_flow, CH-J's configuration, epoch 0 captured from its training loop
    rest = CF.q0_rest_fractions(q0_theta.get("pump_split_logit"), pump_layer)
    sw0 = torch.zeros(4, A, n_months, dtype=D)       # [c0 CH, c0 rest, c1 CH, c1 rest] = 0
    captured: list = []
    orig = CF._rollout

    def _capture(*a, **kw):
        out = orig(*a, **kw)
        if not captured:
            captured.append(out.detach().cpu().clone())
        return out

    def _epoch0(shares) -> tuple:
        captured.clear()
        mm = FlowModel(grid, n_layers=4, dt_days=30.0, device=device, boundaries=boundaries)
        CF._rollout = _capture
        try:
            f = CF.fit_flow(
                mm, syn["obs"], syn["idx"], syn["lay"],
                torch.zeros(4, A, n_months, dtype=D), E=syn["E"], ground_elev=syn["ground"],
                epochs=1, lr=0.1, param_mode="zonal", h0=syn["h0"], zone_of_cell=zc,
                zonation=zone_labels(True, True), recharge_field=syn["rf"],
                pump_layer=pump_layer, recharge_layer=0,
                fix_eta=float(q0_meta["fix_eta"]),
                fix_head_extra=float(q0_meta["fix_head_extra"]), pump_split=True,
                return_flow=True, learn_spread=True, sw_field=sw0, sw_layer=K.CHJ_SW_LAYER,
                well_datum="fit", county_copy_sd=K.CHJ_COUNTY_COPY_SD, changhua_cells=ch_t,
                freeze_from=q0_theta, pump_layer_shares_county=(rest, shares))
        finally:
            CF._rollout = orig
        return captured[0], f

    t0 = time.perf_counter()
    cand, fit = _epoch0(rest)
    t_fit = time.perf_counter() - t0
    # sensitivity control (report-only): the registered X3 shares in Changhua must move the
    # same comparison far beyond the tolerance, or the gate could not see a placement change
    ctrl, _ = _epoch0(np.asarray(K.CHJ_SHARES_CHANGHUA, float))
    d_ctrl = float((ctrl - ref).abs().max())
    d = float((cand - ref).abs().max())
    # report-only: the theta-file form, as k-fold scoring / forward / calibrate_coupled read
    # it (epochs=0: the fit's state is epoch 0's)
    m0 = FlowModel(grid, n_layers=4, dt_days=30.0, device=device, boundaries=boundaries)
    fit0 = CF.fit_flow(
        m0, syn["obs"], syn["idx"], syn["lay"], torch.zeros(4, A, n_months, dtype=D),
        E=syn["E"], ground_elev=syn["ground"], epochs=0, param_mode="zonal", h0=syn["h0"],
        zone_of_cell=zc, zonation=zone_labels(True, True), recharge_field=syn["rf"],
        pump_layer=pump_layer, fix_eta=float(q0_meta["fix_eta"]),
        fix_head_extra=float(q0_meta["fix_head_extra"]), pump_split=True, return_flow=True,
        learn_spread=True, sw_field=sw0, sw_layer=K.CHJ_SW_LAYER, well_datum="fit",
        county_copy_sd=K.CHJ_COUNTY_COPY_SD, changhua_cells=ch_t, freeze_from=q0_theta,
        pump_layer_shares_county=(rest, rest))
    reb = CF._predict_homogeneous(m0, fit0, syn["h0"], n_months, recharge_field=syn["rf"],
                                  E=syn["E"], ground_elev=syn["ground"],
                                  pump_layer=pump_layer, sw_field=sw0,
                                  sw_layer=K.CHJ_SW_LAYER, changhua_cells=ch_t).cpu()
    d_reb = float((reb - ref).abs().max())
    pinned = fit["freeze_partition"]["pinned"]
    plain = [k for k in pinned if not k.startswith("log_sw_scale[")]
    pin_ok = all(np.array_equal(np.asarray(fit0["theta"][k], dtype="float64").ravel(),
                                np.asarray(q0_theta[k], dtype="float64").ravel())
                 for k in plain)
    ok = bool(np.isfinite(d) and d <= K.CHJ_FREEZE_REPRO_TOL_M and pin_ok
              and d_ctrl > 100 * K.CHJ_FREEZE_REPRO_TOL_M)
    return {"passed": ok, "max_abs_diff_m": d, "tol_m": K.CHJ_FREEZE_REPRO_TOL_M,
            "pinned_keys_equal_q0": bool(pin_ok), "n_pinned": len(pinned),
            "n_free": len(fit["freeze_partition"]["free"]),
            "free": fit["freeze_partition"]["free"],
            "report_only_theta_file_rebuild_max_abs_diff_m": d_reb,
            "report_only_control_x3_shares_max_abs_diff_m": d_ctrl,
            "control_sensitive": bool(d_ctrl > 100 * K.CHJ_FREEZE_REPRO_TOL_M),
            "shape": list(cand.shape), "months": int(n_months), "seed": int(seed),
            "fit_seconds": round(t_fit, 1), "device": device,
            "rest_fractions": [float(v) for v in rest],
            "n_changhua_cells": int(np.asarray(changhua, bool).sum()), "n_active": int(A)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="result JSON (written once)")
    ap.add_argument("--theta", default=K.Q0_THETA, help="Q0's stage3_theta.json")
    ap.add_argument("--months", type=int, default=131)
    ap.add_argument("--seed", type=int, default=20261103)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"{a.out} exists (written once)")
    import torch

    from hydrophysics.twin.boundaries import fan_boundaries
    from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS
    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.zone_county import DEFAULT_MAP, county_sha256, verify_county_cache

    torch.set_num_threads(a.threads)
    theta_path = REPO / a.theta
    doc = json.loads(theta_path.read_text())
    grid = build_grid(str(REPO / DEFAULT_PATHS["polygon"]), dx=float(doc["meta"]["dx"]))
    changhua = verify_county_cache(grid, str(REPO / DEFAULT_MAP))
    bnd = (fan_boundaries(grid, proximal_km=ZONE_BOUNDARIES[0])
           if doc["meta"].get("boundaries") == "coast-apex" else None)
    pg = run_pregate(doc["theta"], doc["meta"], grid, changhua, a.months, a.seed,
                     boundaries=bnd)
    hook = REPO / HOOK
    rec = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "what": "CH-J engineering pre-gate (chj_forcing.CALIBRATE_FLOW_PATCH): patched "
                   "fit_flow epoch-0 rollout vs Q0's, synthetic heads and forcing, CPU",
           "pregate": pg, "hook": HOOK, "hook_sha256": _sha256(hook),
           "code_head": _git("rev-parse", "HEAD"),
           "code_dirty": bool(_git("status", "--porcelain", "--untracked-files=no", HOOK)),
           "q0_theta": a.theta, "q0_theta_sha256": _sha256(theta_path),
           "county_map": DEFAULT_MAP, "county_sha256": county_sha256(changhua),
           # the pre-gate script's own hash; NOT the engineering record's producer_sha256,
           # which names the panel producer (dataprep_staging/ch_x/chj_produce.py)
           "pregate_script": "dataprep_staging/ch_x/chj_pregate.py",
           "pregate_sha256": _sha256(Path(__file__)),
           "observed_heads_read": False}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(rec, f, indent=1)
    print(f"pre-gate {'PASSED' if pg['passed'] else 'FAILED'}: max |diff| "
          f"{pg['max_abs_diff_m']:.3e} m (tol {pg['tol_m']:g}); theta-file rebuild "
          f"{pg['report_only_theta_file_rebuild_max_abs_diff_m']:.3e} m; X3-shares control "
          f"{pg['report_only_control_x3_shares_max_abs_diff_m']:.3e} m -> {a.out}")
    return 0 if pg["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
