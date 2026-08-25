import numpy as np
import pytest
import torch
from scipy.special import exp1

from hydrophysics.twin.flow import FlowModel
from hydrophysics.twin.grid import FanGrid


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

    NOTE (brief defect, corrected here): two problems with the brief's version.

    1. Uniform recharge makes the true dL/d(log_T) exactly zero (see the note in
       test_gradients_reach_the_parameters_and_are_finite above), so the comparison is
       trivially 0 == 0 regardless of whether the adjoint's parameter term is even
       implemented -- verified by monkeypatching the brief's literal (broken) backward,
       which returns None for every parameter grad: the test's own `m.log_T.grad.sum()`
       then raises AttributeError rather than exercising the `rel=0.02` comparison at
       all, and with a non-degenerate forcing pattern the true gradient is provably
       nonzero, so a broken backward is no longer masked. Fixed by localizing the
       recharge to a single cell.
    2. This problem is numerically stiff: the harmonic conductance term T (~500) is far
       larger than the storage term S*area/dt (=1e-4*100^2/1=1), so the assembled operator
       is dominated by the discrete-Laplacian-like conductance block and is
       ill-conditioned. A small CG residual does not imply a small solution error for an
       ill-conditioned system, and float32's ~1e-7 relative precision is not enough
       headroom for an eps=1e-4 central difference once that error is amplified --
       verified numerically: in float32 the finite-difference estimate is pure noise
       (order-1 relative error, sign flips) whereas in float64 it agrees with the
       analytic gradient to 8 significant figures. Finite-difference gradient checks
       needing float64 for exactly this reason is standard practice (e.g. PyTorch's own
       `torch.autograd.gradcheck` defaults to double precision). Fixed by building the
       model and its inputs in float64 for this test only.
    """
    g = _uniform_grid(n=9)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        m = FlowModel(g, n_layers=1, dt_days=1.0)
    finally:
        torch.set_default_dtype(prev_dtype)
    A = g.n_active
    rech = torch.zeros(1, A, 2, dtype=torch.float64)
    rech[0, 0, :] = 1e-3

    def loss_of(logT_delta: float) -> float:
        with torch.no_grad():
            m.log_T.fill_(float(np.log(500.0)) + logT_delta)
        return float(m(torch.zeros(1, A, dtype=torch.float64), rech,
                       torch.zeros(1, A, 2, dtype=torch.float64), 2).pow(2).sum())

    with torch.no_grad():
        m.log_T.fill_(float(np.log(500.0)))
    out = m(torch.zeros(1, A, dtype=torch.float64), rech,
            torch.zeros(1, A, 2, dtype=torch.float64), 2).pow(2).sum()
    out.backward()
    analytic = float(m.log_T.grad.sum())
    eps = 1e-4
    numeric = (loss_of(eps) - loss_of(-eps)) / (2 * eps)
    assert analytic == pytest.approx(numeric, rel=0.02)
