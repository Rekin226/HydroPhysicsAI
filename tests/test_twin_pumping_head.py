"""Total dynamic head in the energy->volume conversion.

The Stage-3 gate (2026-09-09) failed with `log_eta` pinned on its lower clamp in every
fold. The cause was here: `energy_to_volume` divided energy by the *static* lift alone,
and on the Choushui fan static lift is median 6.65 m -- ground elevation is median 9.1 m
and heads sit near the surface. Since volume goes as 1/head, that implied ~25e9 m3/yr at
a physical eta = 0.45 against a published ~1.5-2.0e9, and efficiency was the only lever
the fit had to absorb it.

These tests pin the physics: the head a pump works against is static lift PLUS drawdown,
friction and discharge head, and an artesian cell must not imply an unbounded volume.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hydrophysics.twin.pumping import (  # noqa: E402
    J_PER_KWH,
    MIN_LIFT_M,
    RHO_G,
    energy_to_volume,
)

# float64 throughout: the solver works in float64 (see flow.py's module docstring) and a
# float32 log(0.45) alone shifts the result by ~5e-8 relative, which is enough to fail an
# exact closed-form comparison.
F64 = torch.float64
ETA = torch.log(torch.tensor(0.45, dtype=F64))
KWH = torch.tensor([1000.0], dtype=F64)
FAN_MEDIAN_LIFT = torch.tensor([6.65], dtype=F64)


def test_matches_the_closed_form():
    v = energy_to_volume(KWH, torch.tensor([50.0], dtype=F64), ETA)
    expect = 0.45 * 1000.0 * J_PER_KWH / (RHO_G * 50.0)
    assert float(v) == pytest.approx(expect, rel=1e-12)


def test_head_extra_enters_the_denominator():
    """Total dynamic head, not static lift, is what divides the energy."""
    v = energy_to_volume(KWH, torch.tensor([10.0], dtype=F64), ETA,
                         head_extra=torch.tensor(40.0, dtype=F64))
    expect = 0.45 * 1000.0 * J_PER_KWH / (RHO_G * 50.0)
    assert float(v) == pytest.approx(expect, rel=1e-12)


def test_omitting_head_extra_preserves_the_old_behaviour():
    """Results recorded before 2026-09-09 must still reproduce exactly."""
    a = energy_to_volume(KWH, FAN_MEDIAN_LIFT, ETA)
    b = energy_to_volume(KWH, FAN_MEDIAN_LIFT, ETA, head_extra=None)
    assert torch.equal(a, b)


def test_realistic_head_cuts_the_implied_volume_by_about_an_order_of_magnitude():
    """The specific defect that pinned log_eta: 6.65 m of head instead of ~55 m."""
    static_only = energy_to_volume(KWH, FAN_MEDIAN_LIFT, ETA)
    with_tdh = energy_to_volume(KWH, FAN_MEDIAN_LIFT, ETA,
                                head_extra=torch.tensor(48.2, dtype=F64))
    ratio = float(static_only / with_tdh)
    assert 7.0 < ratio < 10.0, f"expected ~8x, got {ratio:.1f}x"


def test_artesian_cell_is_not_floored_into_an_absurd_volume():
    """~20% of fan cell-months have static lift below MIN_LIFT_M; 1% are negative.

    Without a head_extra term those cells clamp to MIN_LIFT_M and imply the largest
    volumes anywhere on the fan -- exactly backwards, since an artesian cell needs the
    LEAST work to lift water.
    """
    artesian = torch.tensor([-14.8], dtype=F64)
    floored = energy_to_volume(KWH, artesian, ETA)
    physical = energy_to_volume(KWH, artesian, ETA, head_extra=torch.tensor(48.2, dtype=F64))
    assert float(floored) == pytest.approx(
        0.45 * 1000.0 * J_PER_KWH / (RHO_G * MIN_LIFT_M), rel=1e-12)
    assert float(physical) < float(floored) / 10.0


def test_still_clamped_when_total_head_would_be_nonpositive():
    """head_extra smaller than the artesian rise must not produce a negative volume."""
    v = energy_to_volume(KWH, torch.tensor([-30.0], dtype=F64), ETA, head_extra=torch.tensor(5.0, dtype=F64))
    assert float(v) > 0
    assert float(v) == pytest.approx(
        0.45 * 1000.0 * J_PER_KWH / (RHO_G * MIN_LIFT_M), rel=1e-12)


def test_volume_is_differentiable_in_head_extra():
    """Calibration learns this parameter, so the gradient has to exist and be signed."""
    lhe = torch.tensor(float(torch.log(torch.tensor(40.0, dtype=F64))),
                   dtype=F64, requires_grad=True)
    v = energy_to_volume(KWH, FAN_MEDIAN_LIFT, ETA, head_extra=torch.exp(lhe))
    v.sum().backward()
    assert lhe.grad is not None and torch.isfinite(lhe.grad).all()
    # More head means less water per kWh.
    assert float(lhe.grad) < 0


def test_bounds_are_registered_for_the_new_parameter():
    from hydrophysics.twin.calibrate_flow import BOUNDS
    lo, hi = BOUNDS["log_head_extra"]
    assert float(torch.exp(torch.tensor(lo))) == pytest.approx(1.0)
    assert float(torch.exp(torch.tensor(hi))) == pytest.approx(200.0)
