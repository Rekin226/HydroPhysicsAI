"""Device placement: the whole flow path must follow --device, evaluation included.

Two real failures motivate this file, both of which cost a multi-hour gate run:

1. `calibrate_flow` constructed `FlowModel(...)` with no `device=`, so every flow run in
   the project's history silently used CPU while `calibrate_mlcw` used CUDA. The measured
   penalty on the real problem was ~51x (3,452 s/epoch against 67).
2. After that was fixed, the k-fold *evaluation* path still built `h0_eval`, the pumping
   scalars and the output index tensors on CPU, so a CUDA run died with "Expected all
   tensors to be on the same device" -- 9.5 h into a gate.

Both are the same class of bug: the fit path moved its inputs internally and the
evaluation path did not, so nothing failed until the fold that used it. These tests run
the *evaluation* helpers directly, since that is the half that was never exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hydrophysics.twin.calibrate_flow import _predict_homogeneous, _rollout  # noqa: E402
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _grid(n=5):
    return FanGrid(nx=n, ny=n, dx=1000.0, x0=0.0, y0=0.0,
                   mask=np.ones((n, n), dtype=bool))


@pytest.mark.parametrize("device", DEVICES)
def test_rollout_accepts_cpu_forcing_against_a_model_on_device(device):
    """CPU-built forcing must be adopted, not rejected, and not silently pull the model off.

    Every loader in this module (`_idw_initial_heads`, `_ground_elev`,
    `_load_pumping_kwh`) returns CPU tensors, so this is the real calling convention.
    """
    g = _grid()
    A, L, steps = g.n_active, 4, 3
    m = FlowModel(g, n_layers=L, device=device).to(device)

    # Deliberately CPU, exactly as the loaders produce them.
    h0 = torch.zeros(L, A, dtype=torch.float64)
    recharge = torch.full((L, A, steps), 1e-6, dtype=torch.float64)
    pumping = torch.zeros(L, A, steps, dtype=torch.float64)

    h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, steps,
                 recharge=recharge, pumping=pumping)
    assert h.device.type == device, "rollout must run where the parameters live"
    assert h.shape == (L, A, steps + 1)
    assert torch.isfinite(h).all()


@pytest.mark.parametrize("device", DEVICES)
def test_rollout_with_dynamic_pumping_forcing(device):
    """The E / ground_elev / log_eta path is what actually crashed the gate."""
    g = _grid()
    A, L, steps = g.n_active, 4, 3
    m = FlowModel(g, n_layers=L, device=device).to(device)

    h0 = torch.zeros(L, A, dtype=torch.float64)
    E = torch.full((A, steps), 5e3, dtype=torch.float64)              # CPU
    ground_elev = torch.full((A,), 20.0, dtype=torch.float64)         # CPU
    log_eta = torch.tensor(float(np.log(0.05)), dtype=torch.float64)  # CPU scalar
    recharge_field = torch.full((A, steps), 1e-6, dtype=torch.float64)
    recharge_scale = torch.tensor(0.0, dtype=torch.float64)

    h = _rollout(m, m.log_T, m.log_S, m.log_L, h0, steps,
                 E=E, ground_elev=ground_elev, log_eta=log_eta,
                 recharge_field=recharge_field, recharge_scale=recharge_scale,
                 recharge_layer=0, pump_layer=1)
    assert h.device.type == device
    assert torch.isfinite(h).all()


@pytest.mark.parametrize("device", DEVICES)
def test_predict_homogeneous_from_cpu_theta(device):
    """`fit["theta"]` is rebuilt as CPU floats during evaluation; it must still run."""
    g = _grid()
    A, L, steps = g.n_active, 4, 3
    m = FlowModel(g, n_layers=L, device=device).to(device)
    fit = {"theta": {"log_eta": float(np.log(0.05)), "recharge_frac_logit": 0.0}}

    h = _predict_homogeneous(
        m, fit, torch.zeros(L, A, dtype=torch.float64), steps,
        recharge=None,
        recharge_field=torch.full((A, steps), 1e-6, dtype=torch.float64),
        E=torch.full((A, steps), 5e3, dtype=torch.float64),
        ground_elev=torch.full((A,), 20.0, dtype=torch.float64),
        recharge_layer=0, pump_layer=1,
    )
    assert h.device.type == device
    assert torch.isfinite(h).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpu_and_cuda_rollouts_agree():
    """The device must change only where the work happens, never the answer."""
    g = _grid()
    A, L, steps = g.n_active, 4, 3
    h0 = torch.zeros(L, A, dtype=torch.float64)
    rech = torch.full((L, A, steps), 1e-6, dtype=torch.float64)
    pump = torch.zeros(L, A, steps, dtype=torch.float64)
    pump[1] = 50.0

    out = {}
    for dev in ("cpu", "cuda"):
        torch.manual_seed(0)
        m = FlowModel(g, n_layers=L, device=dev).to(dev)
        out[dev] = _rollout(m, m.log_T, m.log_S, m.log_L, h0, steps,
                            recharge=rech, pumping=pump).cpu()
    diff = float((out["cpu"] - out["cuda"]).abs().max())
    assert diff < 1e-8, f"cpu/cuda rollouts diverge by {diff:.3e}"
