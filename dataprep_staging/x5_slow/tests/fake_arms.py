"""Test helpers for I1.X5SC.gate0 (CPU; no calibrate_coupled fit is run).

``write_heads_npz`` writes a made-up head field in dump_fit_heads.py's npz format on the
twin's grid (built from the fan polygon only). ``write_arm`` writes one arm the way
run_e2.sh would: jobmeta.start (guard, ledger, job.json) under the synthetic
HYDROMIND_GW_DATA, the column JSON with calibrate_coupled's own constraint keys, then
jobmeta.end. ``fake_declaration`` writes a declaration-like JSON whose ``frozen`` block is
jobmeta.frozen_values with the test's runs dir.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

X = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(X))
import dump_fit_heads as D  # noqa: E402
import e2common as C  # noqa: E402
import evaluate_e2 as E  # noqa: E402
import jobmeta as J  # noqa: E402

T_RECORD = 132          # calibrate_coupled's months 2012-01..2022-12


def write_heads_npz(path: Path) -> Path:
    """A smooth made-up head field (no observation, no model output) in the dump format."""
    from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS
    from hydrophysics.twin.grid import build_grid

    g = build_grid(str(C.repo_path(DEFAULT_PATHS["polygon"])), dx=1000.0)
    cent = g.centroids()
    dates = pd.date_range("2012-01-01", periods=T_RECORD, freq="MS")
    yrs = np.arange(T_RECORD) / 12.0
    xk = (cent[:, 0] - cent[:, 0].min()) / 1000.0
    base = -12.0 + 0.15 * xk                                     # (A,)
    season = np.sin(2 * np.pi * np.arange(T_RECORD) / 12.0)
    lay = np.stack([base[:, None] - 0.12 * yrs[None, :] + (1.0 + 0.2 * k) * season[None, :]
                    - 0.5 * k for k in range(4)])               # (L, A, T)
    theta = C.repo_path(C.SPEC["theta"])
    prov = {"what": "test head field", "argv": D.hindcast_argv("d", "o"),
            "theta": C.SPEC["theta"], "theta_sha256": C.file_sha256(theta),
            "wells_dir": C.SPEC["heads"]["wells_dir"],
            "code": {"tree_sha256": C.code_state()["tree_sha256"]}}
    np.savez_compressed(path, heads_mean=lay[None], dates=np.array([str(d.date()) for d in dates]),
                        scenario_names=np.array(["baseline"]), mask=g.mask, x0=g.x0, y0=g.y0,
                        dx=g.dx, provenance_json=np.array(json.dumps(prov)))
    return path


def constraint_keys(form: str) -> dict:
    """The column JSON's constraint keys from calibrate_coupled.column_constraints() itself,
    its module globals set from the form's flags as calibrate_coupled.main sets them."""
    from hydrophysics.twin import calibrate_coupled as cc

    names = ("CREEP", "CREEP_ZONES", "AQUITARD_TAU_MIN_YEARS", "AQUITARD_HA0_MIN",
             "TAU_MIN_DAYS_ZONE", "TAU_MAX_YEARS", "DETERMINISTIC")
    saved = {n: getattr(cc, n) for n in names}
    flags = C.form_flags(form)
    try:
        cc.CREEP = E._flag(flags, "--creep") or "vep"
        cc.CREEP_ZONES = E._flag(flags, "--creep-zones") or "all"
        v = E._flag(flags, "--aquitard-tau-min-years")
        cc.AQUITARD_TAU_MIN_YEARS = None if v is None else float(v)
        v = E._flag(flags, "--aquitard-ha0-min")
        cc.AQUITARD_HA0_MIN = None if v is None else float(v)
        v = E._flag(flags, "--tau-min-days-zone")
        cc.TAU_MIN_DAYS_ZONE = cc.parse_tau_min_days_zone([v] if v else None)
        v = E._flag(flags, "--tau-max-years")
        cc.TAU_MAX_YEARS = None if v is None else float(v)
        cc.DETERMINISTIC = False
        out = {"creep": cc.CREEP, "tau_max_years": cc.TAU_MAX_YEARS,
               **cc.column_constraints(),
               "tau_ceiling_days": cc.tau_ceiling_days(T_RECORD)}
    finally:
        for n, v in saved.items():
            setattr(cc, n, v)
    return out


def column_json(truth: str, form: str, origin: str, sets: list[dict],
                withhold: str | None = None) -> dict:
    return {"zonal": sets, "config": "zonal", "driver": "flow-model heads",
            "target": "leveling",
            "withhold_surveys": withhold or C.SPEC["withhold"].format(year=origin[:4]),
            "hindcast_gain": 1.0, "hindcast_every": 1, "update_method": "innovation",
            "apex_hold": "calibrated", "hpc0_guard_days": 365.0,
            **constraint_keys(form),
            "tau_at_ceiling": [False, False, False], "theta": C.SPEC["theta"],
            "loss": 0.0, "r2_insample": 0.5, "r2_outoffold": 0.5, "fold_seed": 2027,
            "rings_independent_r2": 123.0,             # must never reach the report
            "leveling": {"n_sites": 0}}


def start_arm(runs: Path, synth: Path, truth: str, form: str, origin: str,
              data_dir: str | None = None, env: str | None = None,
              argv: list[str] | None = None) -> int:
    man = json.loads((synth / "generator_manifest.json").read_text())
    d = data_dir or man["truths"][truth]["data_dir"]
    arm = C.arm_name(truth, form, origin)
    argv = argv if argv is not None else C.fit_args(form, origin, d, str(runs.resolve() / arm))
    old = os.environ.get("HYDROMIND_GW_DATA")
    os.environ["HYDROMIND_GW_DATA"] = env or d
    try:
        return J.start(runs, arm, truth, form, origin, d, argv, synth=synth)
    finally:
        if old is None:
            os.environ.pop("HYDROMIND_GW_DATA", None)
        else:
            os.environ["HYDROMIND_GW_DATA"] = old


def write_arm(runs: Path, synth: Path, truth: str, form: str, origin: str, sets: list[dict],
              withhold: str | None = None, rc: int = 0, output: bool = True) -> Path:
    assert start_arm(runs, synth, truth, form, origin) == 0
    arm = runs.resolve() / C.arm_name(truth, form, origin)
    if output:
        (arm / J.COLUMN_JSON).write_text(json.dumps(
            column_json(truth, form, origin, sets, withhold), indent=1))
    J.end(runs, arm.name, rc)
    return arm


def fake_declaration(path: Path, runs: Path, synth: Path, created: str) -> Path:
    fz = J.frozen_values(synth)
    fz["runs_dir"] = str(runs.resolve())
    path.write_text(json.dumps({"id": C.DECL_ID, "created": created, "frozen": fz}))
    return path


def aq(ska: float, tau_a_years: float, h_a0: float) -> dict:
    return {"log_ska": math.log(ska), "log_tau_a": math.log(tau_a_years * 365.25),
            "h_a0": float(h_a0)}
