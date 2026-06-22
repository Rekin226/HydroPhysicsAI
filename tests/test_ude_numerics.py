"""Numeric-correctness tests for the PhysicsUDE core (rollout + collocation + LOWO leakage).

The smoke tests in ``test_models_smoke.py`` only check shapes/finiteness. These assert the
two physics claims the flagship rests on -- the semi-implicit integrator converges to the
ODE equilibrium and cannot blow up, and the mass-balance residual is zero on data that
exactly satisfies the discrete ODE -- plus the leave-one-well-out no-leakage guarantee.
They mirror the analytic-PDE test the PINN already has and decomp's poison-the-val test.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hydrophysics import Config, load_dataset
from hydrophysics.sample import write_sample

torch = pytest.importorskip("torch")  # whole module skips if torch is absent


@pytest.fixture()
def data(tmp_path):
    d = write_sample(tmp_path / "data", n_wells=4, seed=2)
    return load_dataset(Config(data_dir=d, baseline_results=d / "gw_fit_results.csv"))


def _model():
    from hydrophysics.models.ude import PhysicsUDE
    return PhysicsUDE(device="cpu")  # no fit needed: we call the ODE methods directly


# All white-box ODE tests use float64 so the assertions test the math, not float32 noise.
_D = torch.float64


def test_rollout_converges_to_ode_equilibrium():
    """Under constant forcing the semi-implicit rollout must converge to h* = s/g, the
    fixed point of its own update h = (h + s)/(1 + g)."""
    m = _model()
    torch.manual_seed(0)
    W, T = 3, 800
    params = torch.randn(W, m._n_params(), dtype=_D)
    dyn = torch.randn(W, 1, 4, dtype=_D).expand(W, T, 4).contiguous()   # constant over time
    anchor = torch.zeros(W, dtype=_D)

    g, s = m._decay_and_source(params, dyn, anchor)
    h_star = s[:, -1] / g                                      # ODE equilibrium

    sim = m._rollout(params, dyn, torch.full((W,), 5.0, dtype=_D), anchor)
    assert torch.isfinite(sim).all()
    assert torch.allclose(sim[:, -1], h_star, atol=1e-6)
    # starting far above and far below the equilibrium reaches the same level
    sim_lo = m._rollout(params, dyn, torch.full((W,), -50.0, dtype=_D), anchor)
    assert torch.allclose(sim_lo[:, -1], h_star, atol=1e-6)


def test_rollout_stable_for_large_decay():
    """The integrator's claim is it 'cannot blow up' for g >= 0. A huge recession rate
    (the case that diverges under explicit Euler) must stay finite and bounded."""
    m = _model()
    torch.manual_seed(1)
    W, T = 2, 400
    params = torch.randn(W, m._n_params(), dtype=_D)
    params[:, 0] = 25.0                                        # a = softplus(25) ~= 25 -> big g
    dyn = torch.randn(W, 1, 4, dtype=_D).expand(W, T, 4).contiguous()
    sim = m._rollout(params, dyn, torch.full((W,), 1e3, dtype=_D), torch.zeros(W, dtype=_D))
    assert torch.isfinite(sim).all()
    assert sim.abs().max() < 1e4


def test_physics_residual_zero_on_exact_ode():
    """Collocation residual must be ~0 when the observations exactly satisfy the discrete
    mass balance h[t+1]-h[t] = -g*h[t] + s_t."""
    m = _model()
    torch.manual_seed(2)
    W, T = 3, 150
    # Scale params down so g = softplus(a)+sigmoid(k) stays < 2: the explicit-Euler
    # recurrence used to *construct* the exactly-satisfying target is only stable there.
    params = torch.randn(W, m._n_params(), dtype=_D) * 0.1
    dyn = torch.randn(W, T, 4, dtype=_D)                       # time-varying forcing is fine
    anchor = torch.zeros(W, dtype=_D)
    g, s = m._decay_and_source(params, dyn, anchor)
    assert (g < 2.0).all()                                     # guard the construction's stability

    target = torch.zeros(W, T, dtype=_D)
    target[:, 0] = 3.0
    for t in range(T - 1):
        target[:, t + 1] = target[:, t] + (s[:, t] - g * target[:, t])
    assert torch.isfinite(target).all()

    resid = m._physics_residual(params, dyn, target, torch.ones(W, T, dtype=_D),
                                anchor, torch.ones(W, dtype=_D))
    assert resid.item() < 1e-12


def test_lowo_ignores_validation_observations(data):
    """Leave-one-well-out must never see validation-period observations: poisoning every
    well's validation target leaves the held-out predictions bit-identical (the loss uses
    only training days and the observable features are built on the training mask)."""
    from hydrophysics.lowo import leave_one_well_out
    from hydrophysics.models.ude import observable_features

    clean = leave_one_well_out(data, device="cpu", epochs=8, folds=2,
                               feature_fn=observable_features)

    poisoned_target = data.target.copy()
    poisoned_target[:, data.val_mask] = 999.0
    poisoned = dataclasses.replace(data, target=poisoned_target)
    dirty = leave_one_well_out(poisoned, device="cpu", epochs=8, folds=2,
                               feature_fn=observable_features)

    assert np.allclose(clean, dirty, equal_nan=True)
