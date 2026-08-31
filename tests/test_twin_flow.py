import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
# hydrophysics.twin.grid needs matplotlib.path + pyproj, which are optional extras.
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")
pytest.importorskip("scipy")

from scipy.special import exp1  # noqa: E402

from hydrophysics.twin.calibrate_flow import (  # noqa: E402
    BOUNDS,
    _idw_field,
    _idw_initial_heads,
    _rollout,
    fit_flow,
    kfold_wells,
)
from hydrophysics.twin.flow import FlowModel  # noqa: E402
from hydrophysics.twin.grid import FanGrid  # noqa: E402


def _uniform_grid(n=41, dx=100.0):
    """A square all-active grid, so the solver is tested without polygon masking."""
    return FanGrid(nx=n, ny=n, dx=dx, x0=0.0, y0=0.0,
                   mask=np.ones((n, n), dtype=bool))


def test_theis_drawdown_matches_the_analytical_solution():
    """Confined, homogeneous, single well: the solver must reproduce Theis.

    NOTE (brief defect, corrected here): the brief specifies a 41x41 grid at dx=100m
    (domain half-width 2050m). With T=500 m2/day, S=1e-4, the pressure-diffusion length
    scale sqrt(4*T*t/S) at t=10 days is ~14,142m -- nearly 7x the domain half-width, so
    the no-flow boundary is reached almost immediately and the finite grid behaves like a
    small closed tank rather than an infinite aquifer. Verified numerically: the 41x41
    grid gives numeric/analytic ratios of 4.5x-7.4x at t=10 days (a uniform-domain-wide
    "bathtub" drawdown of Q*t/(S*domain_area) ~ 5.9m dominates), which is not a 20%-level
    discretization error -- it is a different physical regime. Widening the domain while
    holding dx fixed (so near-well resolution is unchanged) converges cleanly onto Theis:
    n=81 -> ratio ~1.9x, n=151 -> ~1.12x, n=251 -> ~1.005x. n=161 is used here for a
    comfortable margin (~10% error at both probe radii) at a runtime of ~1s.

    Reviewer note (fix round 1): the remaining 9.3%/10.5% over-prediction at n=161 is
    boundary image wells, not scheme bias -- at t=10 days the diffusion length
    sqrt(4*T*t/S) ~ 14.1 km still exceeds the n=161 half-width of 8.05 km, and the
    n=41->251 sequence above decays monotonically to 1.005x. A future tightening of the
    rel=0.20 tolerance should be paired with widening the grid further, not read as a
    scheme regression.
    """
    g = _uniform_grid(n=161)
    T_val, S_val, Q = 500.0, 1e-4, 1000.0          # m2/day, -, m3/day
    m = FlowModel(g, n_layers=1, dt_days=1.0)
    with torch.no_grad():
        m.log_T.fill_(float(np.log(T_val)))
        m.log_S.fill_(float(np.log(S_val)))

    A = g.n_active
    centre = g.active_index(80.5 * g.dx, 80.5 * g.dx)
    steps = 10
    pump = torch.zeros(1, A, steps)
    pump[0, centre, :] = Q
    h = m(torch.zeros(1, A), torch.zeros(1, A, steps), pump, steps)

    xy = g.centroids()
    c = xy[centre]
    t_days = steps * 1.0
    for probe_r in (300.0, 500.0):
        d = np.linalg.norm(xy - c, axis=1)
        i = int(np.argmin(np.abs(d - probe_r)))
        r = float(d[i])
        u = r ** 2 * S_val / (4 * T_val * t_days)
        analytic = Q / (4 * np.pi * T_val) * exp1(u)
        numeric = float(-(h[0, i, -1].detach()))   # drawdown is positive
        assert numeric == pytest.approx(analytic, rel=0.20)


@pytest.mark.parametrize("dt_days", [1.0, 30.0])
def test_mass_balance_closes_each_step(dt_days):
    """NOTE (fix round 1): the original assertion omitted dt from `added`. The true
    identity is stored = dt * sum(recharge * area) -- each of the 5 steps injects
    recharge*area*dt of water, not recharge*area. It only passed before because the
    fixture used dt_days=1.0 (so the missing factor was silently 1); at dt_days=30 the
    ratio between the (wrong) old `added` and the correct one is exactly 30.0. The solver
    was already right; only the test's bookkeeping was wrong. Parametrized over two dt
    values so the dt-scaling of storage is actually exercised.
    """
    g = _uniform_grid(n=21)
    m = FlowModel(g, n_layers=1, dt_days=dt_days)
    A = g.n_active
    rech = torch.full((1, A, 5), 1e-3)
    h = m(torch.zeros(1, A), rech, torch.zeros(1, A, 5), 5)
    S = torch.exp(m.log_S)[0]
    cell_area = g.dx ** 2
    stored = float(((h[0, :, -1] - h[0, :, 0]) * S).sum() * cell_area)
    added = float(rech.sum() * cell_area * m.dt)   # no-flow boundaries: nothing leaves
    assert stored == pytest.approx(added, rel=1e-4)


def test_face_conductance_is_harmonic_not_arithmetic_mean():
    """NOTE (fix round 1): every other test in this file uses spatially uniform T, for
    which harmonic and arithmetic face-averaging agree in value and derivative alike --
    the reviewer confirmed empirically that substituting `0.5*(T[:,ia]+T[:,ib])` for the
    harmonic mean in `_matvec_from` still leaves all 4 original tests passing. Since
    log_T is exactly what Tasks 5/6 calibrate cell-by-cell, that gap matters. This test
    uses heterogeneous T and an independent (not code-derived) physical check to close it.

    Setup: a 1D chain of cells (ny=1) split into two uniform zones, T1 then T2, with a
    single point source +Q at one end and an equal point sink -Q at the other end and no
    other sources in between. A single backward-Euler step with a very large dt makes the
    storage term S*area/dt negligible next to the conductance term, so the step solves
    (approximately) the true steady-state elliptic problem K(T) h = q directly.

    In that steady state, mass conservation forces the flux through *every* internal face
    of the chain to equal exactly Q (there is nowhere else for the water to go), including
    the single face straddling the T1/T2 interface. So Tf_interface * (h_left - h_right)
    must equal Q, where Tf_interface is whatever face-conductance formula the operator
    actually uses. The test computes Tf_interface independently in test code as the
    harmonic mean 2*T1*T2/(T1+T2) -- the textbook result for steady flow across a
    conductivity discontinuity, i.e. resistors in series -- and checks that multiplying it
    by the *simulated* head drop reproduces Q. This is not circular: if the operator
    internally used a different formula (e.g. the arithmetic mean), the simulated head
    drop would instead satisfy Tf_other * drop = Q, so Tf_harmonic * drop would recover Q
    only up to the ratio Tf_harmonic/Tf_other -- a large, predictable miss whenever T1 and
    T2 differ substantially, not a subtle rounding difference.

    Verified directly (see fix-round-1 section of the report): with the real harmonic
    implementation, Tf_harmonic * drop matches Q to 7 significant figures; with the
    arithmetic mean monkeypatched into `_matvec_from` in its place, the same check misses
    Q by ~90%.
    """
    n, k = 20, 10                      # chain of 20 cells, interface between col 9 and 10
    T1, T2, Q = 2000.0, 50.0, 10.0     # m2/day, m2/day, m3/day
    g = FanGrid(nx=n, ny=1, dx=100.0, x0=0.0, y0=0.0, mask=np.ones((1, n), dtype=bool))
    m = FlowModel(g, n_layers=1, dt_days=1.0e6)   # huge dt: storage term ~negligible
    A = g.n_active
    with torch.no_grad():
        log_t = torch.full((1, A), float(np.log(T1)))
        log_t[0, k:] = float(np.log(T2))
        m.log_T.copy_(log_t)

    rech = torch.zeros(1, A, 1)
    pump = torch.zeros(1, A, 1)
    rech[0, 0, 0] = Q / m.area          # recharge is a rate (m/day); recharge*area == Q
    pump[0, n - 1, 0] = Q               # pumping is already volumetric (m3/day)
    with torch.no_grad():
        h = m(torch.zeros(1, A), rech, pump, 1)

    drop = float(h[0, k - 1, -1] - h[0, k, -1])
    tf_harmonic = 2.0 * T1 * T2 / (T1 + T2)
    assert tf_harmonic * drop == pytest.approx(Q, rel=1e-3)


def test_gradients_reach_the_parameters_and_are_finite():
    """NOTE (brief defect, corrected here): the brief uses spatially uniform recharge and
    a plain h.sum() loss. With uniform forcing and a uniform initial condition the head
    field is provably uniform at every step (the flux term Tf*(h_i - h_j) is exactly zero
    for any Tf when h_i == h_j everywhere), so log_T can have no effect and its true
    gradient is exactly zero -- verified numerically (grad.abs().sum() == 0.0 exactly, in
    float64). Independently, even with non-uniform forcing, h.sum() is invariant to T
    whenever S is spatially uniform: internal flux terms are added to one cell and
    subtracted from its neighbour, so they cancel exactly in any total sum, leaving
    sum(h) determined only by total forcing and S (this is the same telescoping identity
    the mass-balance test above checks). So neither the forcing pattern nor the loss
    alone was sufficient; both had to change. Fixed here with (a) a localized recharge
    cell, which breaks the spatial symmetry, and (b) a sum-of-squares loss, which is
    sensitive to spatial redistribution rather than only to the total volume balance.
    """
    g = _uniform_grid(n=15)
    m = FlowModel(g, n_layers=1, dt_days=1.0)
    A = g.n_active
    rech = torch.zeros(1, A, 3)
    rech[0, 0, :] = 1e-3
    h = m(torch.zeros(1, A), rech, torch.zeros(1, A, 3), 3)
    h.pow(2).sum().backward()
    for name in ("log_T", "log_S"):
        gr = getattr(m, name).grad
        assert gr is not None and torch.isfinite(gr).all() and gr.abs().sum() > 0


def test_adjoint_gradient_matches_finite_differences():
    """The implicit-differentiation backward must agree with a numerical gradient.

    NOTE (brief defect, corrected here): the brief's uniform recharge makes the true
    dL/d(log_T) exactly zero (see the note in
    test_gradients_reach_the_parameters_and_are_finite above), so the comparison is
    trivially 0 == 0 regardless of whether the adjoint's parameter term is even
    implemented -- verified by monkeypatching the brief's literal (broken) backward,
    which returns None for every parameter grad: the test's own `m.log_T.grad.sum()`
    then raises AttributeError rather than exercising the `rel=0.02` comparison at all,
    and with a non-degenerate forcing pattern the true gradient is provably nonzero, so a
    broken backward is no longer masked. Fixed by localizing the recharge to a single
    cell.

    NOTE (fix round 3): this test used to also build the model and its inputs in float64
    explicitly, because this problem is numerically stiff (the harmonic conductance term
    T~500 is far larger than the storage term S*area/dt=1, so the assembled operator is
    ill-conditioned) and float32's ~1e-7 relative precision left no headroom for an
    eps=1e-4 central difference once that ill-conditioning amplified it -- float32 gave
    pure noise (order-1 relative error, sign flips) while float64 agreed with the
    analytic gradient to 8 significant figures. `FlowModel` now runs in float64
    internally regardless of the caller's dtype (see the module docstring), so that
    workaround is no longer needed here -- plain float32 tensors are passed in below and
    the model casts them itself, the same as any other caller.
    """
    g = _uniform_grid(n=9)
    m = FlowModel(g, n_layers=1, dt_days=1.0)
    A = g.n_active
    rech = torch.zeros(1, A, 2)
    rech[0, 0, :] = 1e-3

    def loss_of(logT_delta: float) -> float:
        with torch.no_grad():
            m.log_T.fill_(float(np.log(500.0)) + logT_delta)
        return float(m(torch.zeros(1, A), rech, torch.zeros(1, A, 2), 2).pow(2).sum())

    with torch.no_grad():
        m.log_T.fill_(float(np.log(500.0)))
    out = m(torch.zeros(1, A), rech, torch.zeros(1, A, 2), 2).pow(2).sum()
    out.backward()
    analytic = float(m.log_T.grad.sum())
    eps = 1e-4
    numeric = (loss_of(eps) - loss_of(-eps)) / (2 * eps)
    assert analytic == pytest.approx(numeric, rel=0.02)


def test_leakage_moves_water_between_layers_and_conserves_it():
    """NOTE (brief defect, corrected here): the brief's `added` omits the `m.dt` factor
    that `test_mass_balance_closes_each_step` above established is required (stored =
    dt * sum(recharge*area)). It happens to still pass as written because this test uses
    dt_days=1.0, which makes the missing factor silently 1 -- the same trap the earlier
    fix-round-1 note describes. Corrected here for consistency with the established
    convention, even though it does not change the numeric result at dt=1.
    """
    g = _uniform_grid(n=15)
    m = FlowModel(g, n_layers=2, dt_days=1.0)
    A = g.n_active
    with torch.no_grad():
        m.log_S.fill_(float(np.log(1e-3)))
        m.log_L.fill_(float(np.log(1e-3)))
    # recharge the upper layer only
    rech = torch.zeros(2, A, 8)
    rech[0] = 1e-3
    h = m(torch.zeros(2, A), rech, torch.zeros(2, A, 8), 8)
    assert float(h[0, :, -1].mean()) > 0.0
    assert float(h[1, :, -1].mean()) > 0.0            # leakage reached the lower layer
    assert float(h[0, :, -1].mean()) > float(h[1, :, -1].mean())

    S = torch.exp(m.log_S)
    stored = float(((h[:, :, -1] - h[:, :, 0]) * S).sum() * g.dx ** 2)
    added = float(rech.sum() * g.dx ** 2 * m.dt)
    assert stored == pytest.approx(added, rel=1e-4)


def test_zero_leakance_decouples_the_layers():
    g = _uniform_grid(n=11)
    m = FlowModel(g, n_layers=3, dt_days=1.0)
    A = g.n_active
    with torch.no_grad():
        m.log_L.fill_(-40.0)                          # effectively zero
    rech = torch.zeros(3, A, 4)
    rech[0] = 1e-3
    h = m(torch.zeros(3, A), rech, torch.zeros(3, A, 4), 4)
    assert float(h[1, :, -1].abs().max()) < 1e-9
    assert float(h[2, :, -1].abs().max()) < 1e-9


def test_leakage_gradient_is_nonzero_and_finite():
    """Finding 1 (fix round 1): `test_zero_leakance_decouples_the_layers` alone cannot
    distinguish "log_L wired correctly but driven to ~0" from "log_L ignored entirely
    by the operator" -- it passes against a stub that never adds the leakage term at
    all, since with no coupling the lower layers are also exactly zero. This test
    closes that gap directly: at a *finite*, non-degenerate leakance, log_L must sit
    inside the operator M(log_T, log_S, log_L), so d(loss)/d(log_L) must be a real,
    finite, nonzero number by construction -- and getting a gradient at all here also
    exercises the n_layers > 1 branch of the conditional parameter tuple threaded
    through _op/_ImplicitSolve.apply.

    Verified (see fix-round-1 section of the report) that this test fails -- with an
    AssertionError on `gr is not None`, since log_L then never enters the autograd
    graph at all -- against a stub that reverts to Task 2's leakage-free `_matvec_from`
    (the pre-Task-3 marker-comment version), confirming the test is not vacuous.
    """
    g = _uniform_grid(n=11)
    m = FlowModel(g, n_layers=2, dt_days=1.0)
    A = g.n_active
    with torch.no_grad():
        m.log_L.fill_(float(np.log(1e-3)))            # finite, non-degenerate leakance
    rech = torch.zeros(2, A, 4)
    rech[0, 0, :] = 1e-3                               # localized: breaks spatial symmetry
    h = m(torch.zeros(2, A), rech, torch.zeros(2, A, 4), 4)
    h.pow(2).sum().backward()
    gr = m.log_L.grad
    assert gr is not None
    assert torch.isfinite(gr).all()
    assert float(gr.abs().sum()) > 0.0


def test_four_layers_run_and_stay_finite():
    g = _uniform_grid(n=11)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    A = g.n_active
    h = m(torch.zeros(4, A), torch.full((4, A, 6), 1e-4), torch.zeros(4, A, 6), 6)
    assert h.shape == (4, A, 7)
    assert torch.isfinite(h).all()


def _synthetic_case(n=13, steps=12, seed=0):
    """Heads generated BY the model, so calibration is checkable against a known truth."""
    g = _uniform_grid(n=n, dx=1000.0)
    A = g.n_active
    truth = FlowModel(g, n_layers=2, dt_days=30.0)
    with torch.no_grad():
        truth.log_T.fill_(float(np.log(800.0)))
        truth.log_S.fill_(float(np.log(5e-4)))
        truth.log_L.fill_(float(np.log(1e-4)))
    rng = torch.Generator().manual_seed(seed)
    rech = torch.rand(2, A, steps, generator=rng) * 1e-3
    rech[1] = 0.0
    h = truth(torch.zeros(2, A), rech, torch.zeros(2, A, steps), steps).detach()
    obs_idx = torch.arange(0, A, max(A // 20, 1))
    obs_layer = torch.zeros_like(obs_idx)
    return g, h, rech, obs_idx, obs_layer


def test_fit_flow_recovers_synthetic_heads():
    """Default param_mode ("homogeneous", Ruling 1): the synthetic truth above is itself
    spatially uniform, so a homogeneous fit should recover it at least as well as a
    per-cell fit would -- this is the brief's own acceptance bar (r2 > 0.9), unchanged by
    the ruling.
    """
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   E=None, ground_elev=None, epochs=300, lr=0.1)
    assert out["param_mode"] == "homogeneous"
    assert out["n_params"] == 2 + 2 + 1          # log_T, log_S per layer + 1 interface log_L
    assert out["r2"] > 0.9


def test_fit_flow_homogeneous_writes_a_spatially_constant_field_back_to_the_model():
    """Ruling 1 implementation detail: fit_flow optimises a small (k, 1) tensor and
    expands it, then copies the expanded field back into the model's own per-cell
    nn.Parameters so a plain `model(...)` call afterwards reproduces the fit. Verify the
    written-back field really is constant per layer (not accidentally left per-cell).
    """
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech, epochs=50, lr=0.1)
    for layer in range(2):
        assert torch.allclose(m.log_T[layer], m.log_T[layer, 0].expand_as(m.log_T[layer]))
        assert torch.allclose(m.log_S[layer], m.log_S[layer, 0].expand_as(m.log_S[layer]))
    assert torch.allclose(m.log_L[0], m.log_L[0, 0].expand_as(m.log_L[0]))


def test_fit_flow_percell_mode_runs_with_many_more_parameters():
    """param_mode="percell" is the brief's original (secondary, per Ruling 1) config:
    FlowModel's own per-cell parameters optimised directly, one (log_T, log_S) per
    (layer, cell) plus one log_L per (interface, cell).
    """
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    A = g.n_active
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=50, lr=0.1, param_mode="percell")
    assert out["param_mode"] == "percell"
    assert out["n_params"] == 2 * 2 * A + 1 * A   # log_T + log_S (2 layers) + log_L (1 interface)


def test_fit_flow_rejects_an_unknown_param_mode():
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with pytest.raises(ValueError):
        fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                 epochs=1, param_mode="bogus")


def test_kfold_wells_returns_a_finite_pooled_number():
    """Ruling 2: k-fold (not leave-one-out) cross-validation over wells. The IDW baseline
    is scored inside the identical fold loop as the flow model, on the identical
    held-out cells -- checked here indirectly via n_held summing to n_wells (every well
    is held out exactly once) and both R2 values being finite over the same pooled set.
    """
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    out = kfold_wells(g, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                      n_layers=2, epochs=120, lr=0.1, n_folds=3)
    assert np.isfinite(out["r2_kfold"])
    assert np.isfinite(out["r2_idw"])
    assert out["n_wells"] == len(obs_idx)
    assert out["n_folds"] == 3
    assert sum(f["n_held"] for f in out["per_fold"]) == out["n_wells"]
    assert len(out["per_fold"]) == 3


def test_kfold_wells_never_calls_itself_loso():
    """Naming discipline (Ruling 2): the gate must never be reported as leave-one-out."""
    import hydrophysics.twin.calibrate_flow as mod

    assert not hasattr(mod, "loso_wells")
    assert hasattr(mod, "kfold_wells")


def test_log_T_bound_is_tightened_to_the_liu_2002_range():
    """Ruling 3: log(10)..log(2e4) m2/day, not the brief's log(1)..log(1e5) -- Task 2
    measured non-convergence across 5 decades of T even in float64 with Jacobi
    preconditioning.
    """
    lo, hi = BOUNDS["log_T"]
    assert lo == pytest.approx(math.log(10.0))
    assert hi == pytest.approx(math.log(2e4))


def test_log_S_and_log_L_bounds_are_unchanged_from_the_brief():
    assert BOUNDS["log_S"] == (pytest.approx(math.log(1e-6)), pytest.approx(math.log(0.3)))
    assert BOUNDS["log_L"] == (pytest.approx(math.log(1e-8)), pytest.approx(math.log(1e-1)))


def test_bounds_hit_reports_per_entry_counts():
    """Clamping must actually bind and be counted when a parameter is pushed outside
    BOUNDS, not just silently clip.
    """
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with torch.no_grad():
        m.log_T.fill_(float(np.log(1e6)))   # far outside the tightened bound
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=1, lr=0.0, param_mode="percell")
    assert out["bounds_hit"]["log_T"] == m.log_T.numel()


def test_zero_forcing_is_provably_parameter_independent():
    """Headline-run finding (module docstring, not one of the three rulings): with h0=0
    and zero recharge/pumping, b=0 at every step whenever h is already 0, so the unique
    solution of the SPD system M @ h = 0 is h = 0 for ANY log_T/log_S/log_L. Pin this
    directly: two wildly different parameter settings must produce bit-identical (zero)
    rollouts under zero forcing, confirming the degeneracy is a property of the zero
    forcing input, not an artifact of one particular parameter choice.
    """
    g = _uniform_grid(n=9)
    A = g.n_active
    rech = torch.zeros(1, A, 4)
    pump = torch.zeros(1, A, 4)
    for log_t, log_s in ((np.log(10.0), np.log(1e-6)), (np.log(2e4), np.log(0.3))):
        m = FlowModel(g, n_layers=1, dt_days=30.0)
        with torch.no_grad():
            m.log_T.fill_(float(log_t))
            m.log_S.fill_(float(log_s))
        h = m(torch.zeros(1, A), rech, pump, 4)
        assert torch.equal(h, torch.zeros_like(h))


# ---------------------------------------------------------------------------------
# Fix round 1: h0 from IDW, dynamic pumping (electricity census) and recharge
# (rain - ET0), superseding the invalid zero-forcing headline run above.
# ---------------------------------------------------------------------------------

def test_idw_initial_heads_falls_back_to_all_wells_for_an_empty_layer():
    g = _uniform_grid(n=5, dx=1000.0)
    xy = np.array([[500.0, 500.0], [4500.0, 4500.0]])
    h0_vals = np.array([10.0, 20.0])
    layer_of = np.array([0, 1])
    out = _idw_initial_heads(g, xy, h0_vals, layer_of, n_layers=3)
    assert out.shape == (3, g.n_active)
    fallback = _idw_field(g, xy, h0_vals)
    assert torch.allclose(out[2], torch.tensor(fallback, dtype=torch.float64))
    # layer 0's own IDW (built from a single point) must differ from the all-wells
    # fallback, confirming the per-layer path isn't silently using the fallback for
    # every layer.
    assert not torch.allclose(out[0], out[2])


def test_dynamic_pumping_lowers_head_relative_to_no_pumping():
    """Fix round 1: pumping via the electricity census must be a real driver, not
    another route to the zero-forcing degeneracy pinned above.
    """
    g = _uniform_grid(n=9, dx=1000.0)
    A = g.n_active
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with torch.no_grad():
        m.log_T.fill_(float(np.log(500.0)))
        m.log_S.fill_(float(np.log(1e-3)))
        m.log_L.fill_(float(np.log(1e-4)))
    ground_elev = torch.full((A,), 20.0, dtype=torch.float64)
    E = torch.full((A, 3), 20000.0, dtype=torch.float64)
    h0 = torch.zeros(2, A, dtype=torch.float64)
    log_eta = torch.tensor(float(np.log(0.3)), dtype=torch.float64)
    h_pumped = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3,
                        E=E, log_eta=log_eta, ground_elev=ground_elev, pump_layer=1)
    h_dry = _rollout(m, m.log_T, m.log_S, m.log_L, h0, 3)
    assert torch.equal(h_dry, torch.zeros_like(h_dry))       # the old degenerate case
    assert float(h_pumped[1, :, -1].mean()) < float(h_dry[1, :, -1].mean())
    assert torch.isfinite(h_pumped).all()


def test_dynamic_recharge_raises_head_relative_to_no_recharge():
    g = _uniform_grid(n=9, dx=1000.0)
    A = g.n_active
    m = FlowModel(g, n_layers=1, dt_days=30.0)
    with torch.no_grad():
        m.log_T.fill_(float(np.log(500.0)))
        m.log_S.fill_(float(np.log(1e-3)))
    h0 = torch.zeros(1, A, dtype=torch.float64)
    recharge_field = torch.full((A, 3), 5e-3, dtype=torch.float64)
    scale = torch.tensor(2.0, dtype=torch.float64)            # sigmoid(2) ~ 0.88
    h_wet = _rollout(m, m.log_T, m.log_S, None, h0, 3,
                     recharge_field=recharge_field, recharge_scale=scale, recharge_layer=0)
    h_dry = _rollout(m, m.log_T, m.log_S, None, h0, 3)
    assert torch.equal(h_dry, torch.zeros_like(h_dry))
    assert float(h_wet[0, :, -1].mean()) > float(h_dry[0, :, -1].mean())


def test_fit_flow_homogeneous_with_forcing_exposes_eta_and_recharge_fraction():
    """fit_flow's homogeneous mode must add exactly the two scalar parameters the
    coordinator's fix-round-1 ruling names (log_eta, recharge_frac_logit), report them
    under BOUNDS/theta, and actually move them off their fixed initial values -- if they
    hadn't moved, the dynamic forcing path would be silently disconnected from the loss
    the same way the zero-forcing configuration was.
    """
    g = _uniform_grid(n=9, dx=1000.0)
    A = g.n_active
    n_layers, steps = 2, 4
    m = FlowModel(g, n_layers=n_layers, dt_days=30.0)
    ground_elev = torch.full((A,), 20.0, dtype=torch.float64)
    E = torch.full((A, steps), 5000.0, dtype=torch.float64)
    recharge_field = torch.full((A, steps), 5e-3, dtype=torch.float64)
    h0 = torch.zeros(n_layers, A, dtype=torch.float64)
    recharge_dummy = torch.zeros(n_layers, A, steps, dtype=torch.float64)

    obs_idx = torch.arange(0, A, max(A // 5, 1))
    obs_idx_all = torch.cat([obs_idx, obs_idx])
    obs_layer_all = torch.cat([torch.zeros_like(obs_idx), torch.ones_like(obs_idx)])
    obs_h = torch.zeros(len(obs_idx_all), steps, dtype=torch.float64)
    obs_h[: len(obs_idx)] = 2.0     # recharged layer should rise
    obs_h[len(obs_idx):] = -2.0     # pumped layer should fall

    out = fit_flow(m, obs_h, obs_idx_all, obs_layer_all, recharge_dummy,
                   E=E, ground_elev=ground_elev, epochs=60, lr=0.2,
                   h0=h0, recharge_field=recharge_field)
    assert out["n_params"] == n_layers + n_layers + (n_layers - 1) + 1 + 1   # +eta +rfrac
    assert "eta" in out["theta"] and "recharge_frac" in out["theta"]
    assert "log_eta" in out["bounds_hit"]
    assert math.isfinite(out["loss"]) and math.isfinite(out["r2"])
    assert out["theta"]["log_eta"] != pytest.approx(float(np.log(0.3)), abs=1e-9)
    assert out["theta"]["recharge_frac_logit"] != pytest.approx(0.0, abs=1e-9)


# --- grouped k-fold: co-located screens must never straddle a fold boundary -------------
# The 2026-08-27 retraction: the Choushui head field is 66 physical sites carrying
# layer-coded screens, and an ungrouped split put 69.9% of held-out entries at ZERO
# distance from a training entry. idw_interp weights by 1/(d^2 + 1e-6), so a co-located
# source outweighs a 1 km neighbour by 1e12 and the "baseline" becomes a near-oracle.

def test_kfold_indices_without_groups_is_unchanged():
    from hydrophysics.twin.calibrate_flow import _kfold_indices
    folds = _kfold_indices(20, 4, seed=0)
    assert sum(len(f) for f in folds) == 20
    assert sorted(np.concatenate(folds).tolist()) == list(range(20))


def test_kfold_indices_keeps_every_member_of_a_group_in_one_fold():
    from hydrophysics.twin.calibrate_flow import _kfold_indices
    # 12 entries over 4 sites, 3 screens each
    groups = np.repeat(np.arange(4), 3)
    folds = _kfold_indices(12, 2, seed=0, groups=groups)
    assert sorted(np.concatenate(folds).tolist()) == list(range(12))
    for f in folds:
        # every site represented in this fold must be *wholly* in it
        for site in np.unique(groups[f]):
            assert set(np.flatnonzero(groups == site)) <= set(f.tolist())


def test_kfold_indices_rejects_more_folds_than_groups():
    from hydrophysics.twin.calibrate_flow import _kfold_indices
    groups = np.repeat(np.arange(3), 2)
    with pytest.raises(ValueError):
        _kfold_indices(6, 5, seed=0, groups=groups)


def test_kfold_wells_groups_colocated_wells_so_the_baseline_cannot_peek():
    """Two screens per site: an ungrouped split leaks, a grouped one must not."""
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    # two co-located screens (different layers) at each observed cell
    obs_idx_all = torch.cat([obs_idx, obs_idx])
    obs_layer_all = torch.cat([torch.zeros_like(obs_idx), torch.ones_like(obs_idx)])
    cent = g.centroids()
    well_xy = np.concatenate([cent[obs_idx.numpy()], cent[obs_idx.numpy()]], axis=0)
    W = len(obs_idx_all)
    obs = h[obs_layer_all, obs_idx_all, 1:]
    out = kfold_wells(g, obs, obs_idx_all, obs_layer_all, rech, n_layers=2,
                      epochs=3, lr=0.1, n_folds=3, well_xy=well_xy,
                      obs_h0=np.zeros(W))
    assert "colocation_rate" in out, "gate must report how leaky its folds are"
    assert out["colocation_rate"] == pytest.approx(0.0), (
        f"grouped folds still leak: {out['colocation_rate']:.3f}")


def test_kfold_indices_seed_changes_the_split_but_keeps_groups_intact():
    """--seed exists to measure fold-assignment variance: the 2026-08-27 gate came down to
    a 0.033 margin, which a single split cannot resolve. Varying the seed must actually
    move the split, and must never break the site grouping while doing so."""
    from hydrophysics.twin.calibrate_flow import _kfold_indices
    groups = np.repeat(np.arange(12), 2)          # 12 sites, 2 screens each
    a = _kfold_indices(24, 3, seed=0, groups=groups)
    b = _kfold_indices(24, 3, seed=1, groups=groups)
    again = _kfold_indices(24, 3, seed=0, groups=groups)

    # same seed is reproducible
    for f, g in zip(a, again, strict=True):
        assert np.array_equal(f, g)
    # a different seed actually moves wells between folds
    assert any(not np.array_equal(f, g) for f, g in zip(a, b, strict=True))
    # and grouping survives either way
    for folds in (a, b):
        assert sorted(np.concatenate(folds).tolist()) == list(range(24))
        for f in folds:
            for site in np.unique(groups[f]):
                assert set(np.flatnonzero(groups == site)) <= set(f.tolist())


def test_kfold_wells_dumps_per_entry_predictions_with_distances(tmp_path):
    """The pooled gate R2 cannot say whether the physics model degrades more or less
    gracefully than IDW as held-out sites get isolated, which is most of the argument for
    building a physics model. The dump keeps the raw predictions so that question is
    answerable without refitting."""
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    obs_idx_all = torch.cat([obs_idx, obs_idx])
    obs_layer_all = torch.cat([torch.zeros_like(obs_idx), torch.ones_like(obs_idx)])
    cent = g.centroids()
    well_xy = np.concatenate([cent[obs_idx.numpy()], cent[obs_idx.numpy()]], axis=0)
    W = len(obs_idx_all)
    obs = h[obs_layer_all, obs_idx_all, 1:]
    out_npz = tmp_path / "per_entry.npz"

    out = kfold_wells(g, obs, obs_idx_all, obs_layer_all, rech, n_layers=2,
                      epochs=3, lr=0.1, n_folds=3, well_xy=well_xy,
                      obs_h0=np.zeros(W), dump_path=str(out_npz))

    d = np.load(out_npz)
    # every held-out entry appears exactly once, across all folds
    assert sorted(d["entry"].tolist()) == list(range(W))
    assert d["pred"].shape == d["obs"].shape == d["idw"].shape == (W, obs.shape[1])
    # distances are real, positive, and finite -- grouping means never zero
    assert np.isfinite(d["nn_dist"]).all()
    assert (d["nn_dist"] > 0).all(), "grouped folds must not leave a zero-distance neighbour"
    # the dumped rows reproduce the pooled numbers the gate reported
    assert _r2_ref(d["pred"], d["obs"]) == pytest.approx(out["r2_kfold"], abs=1e-9)
    assert _r2_ref(d["idw"], d["obs"]) == pytest.approx(out["r2_idw"], abs=1e-9)


def _r2_ref(pred, obs):
    from hydrophysics.twin.calibrate_flow import _r2
    return _r2(np.asarray(pred), np.asarray(obs))


def test_make_zonal_params_has_exactly_26_free_parameters():
    """Spec §5: proximal 2 (one merged aquifer) + mid 11 + distal 11 + global 2.
    Fewer than naive 3x uniform zoning's 35, and every one physically motivated.
    """
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=True, use_recharge=True)
    assert sum(p.numel() for p in theta.values()) == 26
    assert theta["log_T_proximal"].shape == (1, 1)
    assert theta["log_S_proximal"].shape == (1, 1)
    assert theta["log_T_mid"].shape == (4, 1)
    assert theta["log_L_mid"].shape == (3, 1)
    assert theta["log_T_distal"].shape == (4, 1)
    assert theta["log_L_distal"].shape == (3, 1)


def test_make_zonal_params_does_not_expose_a_proximal_log_L():
    """Spec §5: fixing proximal log_L at the top of its range IS the statement 'there is
    no aquitard here'. It must be a constant, never an optimised parameter -- if it
    appears in theta the optimiser can walk it away from the published geology.
    """
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=True, use_recharge=True)
    assert "log_L_proximal" not in theta


def test_make_zonal_params_without_drivers_drops_the_two_globals():
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=False, use_recharge=False)
    assert sum(p.numel() for p in theta.values()) == 24
    assert "log_eta" not in theta
    assert "recharge_frac_logit" not in theta


def test_base_param_name_strips_zone_suffixes_but_not_log_eta():
    from hydrophysics.twin.calibrate_flow import BOUNDS, _base_param_name

    assert _base_param_name("log_T_mid") == "log_T"
    assert _base_param_name("log_S_proximal") == "log_S"
    assert _base_param_name("log_L_distal") == "log_L"
    assert _base_param_name("log_eta") == "log_eta"
    assert _base_param_name("recharge_frac_logit") == "recharge_frac_logit"
    for name in ("log_T_mid", "log_S_proximal", "log_L_distal", "log_eta"):
        assert _base_param_name(name) in BOUNDS


def test_expand_zonal_gathers_each_zones_value_to_its_own_cells():
    """The expansion is an advanced index into a (k, 3) column stack, so a cell must
    receive its own zone's value and nothing else.
    """
    from hydrophysics.twin.calibrate_flow import _expand_zonal, _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m)
    with torch.no_grad():
        theta["log_T_proximal"].fill_(1.0)
        theta["log_T_mid"].fill_(2.0)
        theta["log_T_distal"].fill_(3.0)
    zone_t = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    log_T, log_S, log_L = _expand_zonal(theta, zone_t, n_layers=4)
    assert log_T.shape == (4, 5)
    assert log_S.shape == (4, 5)
    assert log_L.shape == (3, 5)
    assert log_T[0].tolist() == [1.0, 2.0, 3.0, 2.0, 1.0]
    assert torch.allclose(log_T[3], log_T[0])       # proximal value shared across layers


def test_expand_zonal_shares_one_value_across_all_proximal_layers():
    """Spec §5: the proximal zone is ONE merged aquifer, so its four layers must carry
    an identical log_T and log_S -- not four values that happen to start equal.
    """
    from hydrophysics.twin.calibrate_flow import _expand_zonal, _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m)
    with torch.no_grad():
        theta["log_T_mid"].copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float64))
    zone_t = torch.tensor([0, 1], dtype=torch.long)
    log_T, log_S, _ = _expand_zonal(theta, zone_t, n_layers=4)
    assert len(set(log_T[:, 0].tolist())) == 1       # proximal: one value, four layers
    assert log_T[:, 1].tolist() == [1.0, 2.0, 3.0, 4.0]   # mid: four distinct layers
    assert len(set(log_S[:, 0].tolist())) == 1


def test_expand_zonal_pins_proximal_log_L_at_the_upper_bound():
    """The merged-aquifer statement, checked at the value level: proximal leakage sits at
    the top of BOUNDS, and it is a constant, so it carries no gradient.
    """
    from hydrophysics.twin.calibrate_flow import BOUNDS, _expand_zonal, _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m)
    zone_t = torch.tensor([0, 1, 2], dtype=torch.long)
    _, _, log_L = _expand_zonal(theta, zone_t, n_layers=4)
    assert torch.allclose(log_L[:, 0],
                       torch.full((3,), BOUNDS["log_L"][1], dtype=torch.float64))


def test_expand_zonal_accumulates_gradient_onto_each_zones_parameter():
    """Advanced indexing must scatter-add the per-cell gradient back onto the small
    per-zone tensor, the same way homogeneous mode relies on expand-backward. If a zone
    receives a zero or None gradient it is being silently frozen -- this project has
    already shipped a backward() returning None for log_T once.
    """
    from hydrophysics.twin.calibrate_flow import _expand_zonal, _make_zonal_params

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=True, use_recharge=True)
    zone_t = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)
    log_T, log_S, log_L = _expand_zonal(theta, zone_t, n_layers=4)
    (log_T.sum() + log_S.sum() + log_L.sum()).backward()
    for name in ("log_T_proximal", "log_S_proximal", "log_T_mid", "log_S_mid",
                 "log_L_mid", "log_T_distal", "log_S_distal", "log_L_distal"):
        gr = theta[name].grad
        assert gr is not None, f"{name} received no gradient"
        assert torch.isfinite(gr).all(), f"{name} gradient is not finite"
        assert (gr != 0).all(), f"{name} gradient is zero -- the zone is frozen"
    # proximal log_T is shared across 4 layers x 2 cells -> gradient of 8
    assert float(theta["log_T_proximal"].grad) == pytest.approx(8.0)
    # mid log_T is per-layer over 3 cells -> gradient of 3 per layer
    assert theta["log_T_mid"].grad.flatten().tolist() == pytest.approx([3.0] * 4)


def test_zonal_bounds_hit_reports_per_zone_and_never_pools():
    """Spec §6 primary rule: a pinned proximal log_T must not be maskable by interior
    mid/distal values. Pin proximal at the lower clamp, leave the rest interior, and the
    report must still show it.
    """
    from hydrophysics.twin.calibrate_flow import (
        BOUNDS,
        _make_zonal_params,
        _zonal_bounds_hit,
    )

    g = _uniform_grid(n=8, dx=1000.0)
    m = FlowModel(g, n_layers=4, dt_days=30.0)
    theta = _make_zonal_params(m, use_pumping=True, use_recharge=True)
    with torch.no_grad():
        theta["log_T_proximal"].fill_(BOUNDS["log_T"][0])
        theta["log_T_mid"].fill_(math.log(500.0))
        theta["log_T_distal"].fill_(math.log(500.0))
    hits = _zonal_bounds_hit(theta)
    assert set(hits) == {"proximal", "mid", "distal", "global"}
    assert hits["proximal"]["log_T"] == 1
    assert hits["mid"]["log_T"] == 0
    assert hits["distal"]["log_T"] == 0
    assert "log_L" not in hits["proximal"]        # fixed, not a free parameter
    assert "log_eta" in hits["global"]


def _zoned_synthetic_case(seed=0):
    """The synthetic case, plus a zone assignment that splits the square grid into three
    west-east bands so all three zones carry cells.
    """
    from hydrophysics.twin.zones import fan_zones

    g, h, rech, obs_idx, obs_layer = _synthetic_case(seed=seed)
    xy = g.centroids()
    # the uniform grid is 13x13 at dx=1000 m; rescale the eastings onto 170..215 km so the
    # default boundaries (205 / 182) cut it into three non-empty bands
    span = xy[:, 0].max() - xy[:, 0].min()
    x_km = 170.0 + 45.0 * (xy[:, 0] - xy[:, 0].min()) / max(span, 1e-9)
    zoned = np.column_stack([x_km * 1000.0, xy[:, 1]])
    return g, h, rech, obs_idx, obs_layer, fan_zones(zoned)


def test_fit_flow_zonal_reports_the_mode_and_runs():
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=30, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    assert out["param_mode"] == "zonal"
    assert math.isfinite(out["loss"])
    assert math.isfinite(out["r2"])


def test_fit_flow_zonal_has_the_expected_free_parameter_count():
    """2 layers: proximal 2 + mid (2+2+1) + distal (2+2+1) = 12, no drivers here."""
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=1, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    assert out["n_params"] == 2 + 5 + 5


def test_fit_flow_zonal_requires_a_zone_assignment():
    """Running zonal without zones would silently fall back to something -- refuse."""
    g, h, rech, obs_idx, obs_layer, _ = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with pytest.raises(ValueError, match="zone_of_cell"):
        fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                 epochs=1, param_mode="zonal", zone_of_cell=None)


def test_fit_flow_zonal_rejects_a_zone_vector_of_the_wrong_length():
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with pytest.raises(ValueError):
        fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                 epochs=1, param_mode="zonal", zone_of_cell=zones[:-1])


def test_fit_flow_zonal_reports_bounds_hit_per_zone():
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=5, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    hits = out["bounds_hit"]
    assert set(hits) == {"proximal", "mid", "distal", "global"}
    assert "log_T" in hits["proximal"] and "log_T" in hits["mid"]
    assert "log_L" not in hits["proximal"]


def test_fit_flow_zonal_writes_a_piecewise_constant_field_back_to_the_model():
    """The copy-back must produce a field that is constant WITHIN each zone and
    (generally) different between them. This is what lets the existing evaluation path
    read model.log_T directly for zonal, exactly as it does for homogeneous.
    """
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
             epochs=20, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    for zone_id in (0, 1, 2):
        sel = torch.tensor(zones == zone_id)
        assert sel.any(), f"zone {zone_id} has no cells in this fixture"
        block = m.log_T[0][sel]
        assert torch.allclose(block, block[0].expand_as(block))


def test_fit_flow_zonal_gradients_reach_every_zone_and_are_finite():
    """The _ImplicitSolve.backward returning None for log_T is a bug this project has
    already shipped once. Check the real fit path, not just the expansion helper: after
    one step every zone's parameter must have moved.
    """
    from hydrophysics.twin.calibrate_flow import _make_zonal_params

    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    before = {k: v.detach().clone()
              for k, v in _make_zonal_params(m).items()}
    out = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=3, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    for name, start in before.items():
        moved = np.asarray(out["theta"][name], dtype="float64")
        assert np.isfinite(moved).all(), f"{name} went non-finite"
        assert not np.allclose(moved, start.squeeze(-1).numpy()), \
            f"{name} did not move -- no gradient reached this zone"


def test_fit_flow_zonal_proximal_layers_equilibrate():
    """Spec §7: with proximal log_L fixed high, heads across the proximal layers must
    actually converge -- assert it rather than assuming the fixed leakage did its job.
    Compare against the distal zone, where leakage is free and low.
    """
    g, h, rech, obs_idx, obs_layer, zones = _zoned_synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    fit = fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                   epochs=40, lr=0.1, param_mode="zonal", zone_of_cell=zones)
    assert fit["param_mode"] == "zonal"
    A = g.n_active
    h0 = torch.zeros(2, A, dtype=torch.float64)
    with torch.no_grad():
        hh = m(h0, rech, torch.zeros(2, A, rech.shape[-1], dtype=torch.float64),
               rech.shape[-1])
    prox = torch.tensor(zones == 0)
    dist = torch.tensor(zones == 2)
    spread_prox = (hh[0][prox] - hh[1][prox]).abs().mean()
    spread_dist = (hh[0][dist] - hh[1][dist]).abs().mean()
    assert spread_prox < spread_dist, (
        f"proximal layers did not equilibrate: spread {spread_prox:.4g} is not below "
        f"the distal spread {spread_dist:.4g}"
    )


def test_fit_flow_still_rejects_an_unknown_param_mode_after_zonal_lands():
    g, h, rech, obs_idx, obs_layer = _synthetic_case()
    m = FlowModel(g, n_layers=2, dt_days=30.0)
    with pytest.raises(ValueError):
        fit_flow(m, h[obs_layer, obs_idx, 1:], obs_idx, obs_layer, rech,
                 epochs=1, param_mode="zonal_typo")
