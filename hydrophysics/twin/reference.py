"""Independent MODFLOW 6 flow and CSUB elastic-limit verification.

Requires the optional [reference] extra and an explicitly supplied mf6 executable.
These synthetic verification cases do not certify the Choushui calibration or identify
its creep mechanism. No field data, driver changes or downloads are performed here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _simulation(workspace, executable, name, periods):
    import flopy

    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(workspace),
                                 exe_name=str(executable))
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=periods,
                         perioddata=[(30.0, 1, 1.0)] * periods)
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-10, inner_dvclose=1e-11,
                       rcloserecord=1e-9, outer_maximum=100, inner_maximum=1000,
                       linear_acceleration="CG")
    return sim, flopy.mf6.ModflowGwf(sim, modelname=name, save_flows=True)


def _run(sim):
    sim.write_simulation(silent=True)
    ok, log = sim.run_simulation(silent=True, report=True)
    if not ok:
        raise RuntimeError("MODFLOW failed: " + "\n".join(log[-20:]))


def flow_case(workspace: Path, executable: Path) -> dict:
    """Heterogeneous two-layer transient FV solve with identical GHB/WEL stresses."""
    import flopy
    import torch

    from .boundaries import Boundaries
    from .flow import FlowModel
    from .grid import FanGrid

    nx, ny, layers, steps, dx, thick = 9, 7, 2, 24, 1000.0, 100.0
    grid = FanGrid(nx, ny, dx, 0.0, 0.0, np.ones((ny, nx), dtype=bool))
    cells = nx * ny
    left, right = np.arange(ny) * nx, np.arange(ny) * nx + nx - 1
    boundaries = Boundaries(left, np.ones(ny), right, np.ones(ny))
    model = FlowModel(grid, layers, dt_days=30, boundaries=boundaries)
    tr = np.stack([np.linspace(300, 800, cells), np.linspace(100, 450, cells)])
    storage = np.stack([np.full(cells, 0.002), np.full(cells, 0.001)])
    leakance, coast_c, apex_c = 1e-4, 20.0, 25.0
    initial = np.full((layers, cells), 10.0)
    with torch.no_grad():
        model.log_T.copy_(torch.tensor(np.log(tr)))
        model.log_S.copy_(torch.tensor(np.log(storage)))
        model.log_L.fill_(np.log(leakance))
        model.log_C_coast.fill_(np.log(coast_c))
        model.log_C_apex.fill_(np.log(apex_c))
        model.set_apex_heads(torch.tensor(initial))
    recharge = np.zeros((layers, cells, steps))
    recharge[0] = 1e-5 * (1 + 0.4 * np.sin(np.arange(steps) * np.pi / 6))
    pumping = np.zeros_like(recharge)
    pumping[1, cells // 2] = np.r_[np.full(12, 500.0), np.full(12, 250.0)]
    with torch.no_grad():
        heads = model(torch.tensor(initial), torch.tensor(recharge),
                      torch.tensor(pumping), steps).numpy()[..., 1:]
    sim, gwf = _simulation(workspace, executable, "flow", steps)
    flopy.mf6.ModflowGwfdis(gwf, nlay=layers, nrow=ny, ncol=nx, delr=dx, delc=dx,
                          top=0.0, botm=[-thick, -2 * thick], length_units="METERS")
    flopy.mf6.ModflowGwfic(gwf, strt=10.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=(tr / thick).reshape(layers, ny, nx),
                          k33=leakance * thick)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=(storage / thick).reshape(layers, ny, nx),
                          sy=0, transient={0: True})
    ghb = []
    for layer in range(layers):
        for row in range(ny):
            ghb.extend([((layer, row, 0), 0.0, coast_c),
                        ((layer, row, nx - 1), 10.0, apex_c)])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data={0: ghb})
    forcing = recharge * dx**2 - pumping
    wells = {t: [((layer, cell // nx, cell % nx), forcing[layer, cell, t])
                 for layer in range(layers) for cell in range(cells)] for t in range(steps)}
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wells)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="flow.hds", budget_filerecord="flow.cbc",
                         saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    _run(sim)
    reference = gwf.output.head().get_alldata().reshape(steps, layers, cells).transpose(1, 2, 0)
    error = heads - reference
    previous = np.concatenate([initial[..., None], heads[..., :-1]], axis=-1)
    stored = ((heads - previous) * storage[..., None] * dx**2).sum(axis=(0, 1)) / 30
    boundary_flux = (coast_c * (0 - heads[:, left])).sum(axis=(0, 1))
    boundary_flux += (apex_c * (10 - heads[:, right])).sum(axis=(0, 1))
    residual = stored - forcing.sum(axis=(0, 1)) - boundary_flux
    denominator = np.maximum(1, np.abs(forcing).sum(axis=(0, 1))
                             + np.abs(coast_c * heads[:, left]).sum(axis=(0, 1))
                             + np.abs(apex_c * (10 - heads[:, right])).sum(axis=(0, 1)))
    result = {"head_rmse_m": float(np.sqrt(np.mean(error**2))),
              "head_max_abs_m": float(np.max(np.abs(error))),
              "max_relative_water_budget_error": float(np.max(np.abs(residual) / denominator)),
              "head_tolerance_m": 1e-5, "budget_tolerance": 1e-6,
              "domain": "synthetic 9x7 cells, two confined layers, 24 monthly steps",
              "mapping": "K=T/thickness, Ss=S/thickness, K33=leakance*thickness; identical GHB and net cell fluxes"}
    result["passed"] = result["head_max_abs_m"] < 1e-5 and result["max_relative_water_budget_error"] < 1e-6
    return result


def compaction_case(workspace: Path, executable: Path) -> dict:
    """A prescribed drawdown/recovery verifies CSUB and VEP's shared elastic limit."""
    import flopy
    import torch

    from .compaction import VEPColumn

    h = np.r_[10.0, np.linspace(9.5, 5.0, 10), np.linspace(5.5, 10.0, 10)]
    skeletal, thickness = 1e-3, 10.0
    sim, gwf = _simulation(workspace, executable, "elastic", len(h) - 1)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=1, delr=100, delc=100,
                          top=20, botm=-80)
    flopy.mf6.ModflowGwfic(gwf, strt=h[0])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10)
    flopy.mf6.ModflowGwfsto(gwf, ss=0, sy=0, transient={0: True})
    # CSUB skips constant-head cells. A strong GHB drives an active cell instead;
    # compare both constitutive laws on the actual solved cell heads below.
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data={t: [((0, 0, 0), v, 1e4)]
                                                   for t, v in enumerate(h[1:])})
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="elastic.hds", saverecord=[("HEAD", "ALL")])
    csub = flopy.mf6.ModflowGwfcsub(
        gwf, head_based=True, beta=0, cg_ske_cr=0, ninterbeds=1,
        packagedata=[(0, (0, 0, 0), "NODELAY", -100.0, thickness, 1.0,
                      skeletal / thickness, skeletal / thickness, 0.4, 1.0, 0.0)])
    csub.obs.initialize(filename="elastic.obs", digits=15,
                        continuous={"compaction.csv": [("compaction", "interbed-compaction", (0,))]})
    _run(sim)
    h = np.r_[h[0], gwf.output.head().get_alldata().ravel()]
    values = np.genfromtxt(workspace / "compaction.csv", delimiter=",", names=True)
    reference = np.atleast_1d(values["COMPACTION"])
    column = VEPColumn(1).double()
    with torch.no_grad():
        column.log_ske.fill_(np.log(skeletal))
        column.log_skv.fill_(-100.0)
        column.h_pc0.fill_(-100.0)
        predicted = column(torch.tensor(h[None, :])).numpy()[0, 1:]
    analytic = skeletal * (h[0] - h[1:])
    error = float(np.max(np.abs(reference - predicted)))
    analytic_error = float(np.max(np.abs(reference - analytic)))
    return {"max_abs_difference_m": error, "analytic_max_abs_m": analytic_error,
            "tolerance_m": 1e-8, "passed": error < 1e-8 and analytic_error < 1e-8,
            "scope": "elastic drawdown and recovery only; no claim of equivalent creep or plasticity"}


def main(argv=None):
    from .release import atomic_json, sha256

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mf6", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)
    exe = args.mf6.resolve()
    if not exe.is_file():
        ap.error("--mf6 must point to an existing MODFLOW 6 executable")
    args.out.mkdir(parents=True, exist_ok=True)
    report = {"schema": 1, "type": "synthetic-numerical-verification",
              "executable_sha256": sha256(exe), "field_validation": False,
              "flow": flow_case(args.out / "flow", exe),
              "compaction": compaction_case(args.out / "compaction", exe)}
    report["passed"] = report["flow"]["passed"] and report["compaction"]["passed"]
    atomic_json(args.out / "report.json", report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
