"""Synthetic depth and boundary cases independent of field-site selection."""
from __future__ import annotations

import numpy as np
import pytest

from hydrophysics.twin.boundary_evidence import cross_boundary_pairs, jump_diagnostic
from hydrophysics.twin.observation_operator import (
    DepthProfile,
    ring_interval,
    surface_displacement,
    total_column,
)


def test_depth_operator_keeps_interval_column_and_surface_distinct():
    profile = DepthProfile((0, 100, 200, 300),
                           np.array([[0, 0, 0], [.01, .02, .03]]), "2023-01")
    assert ring_interval(profile, 100, 200, reference_month="2023-01")[-1] == .02
    assert total_column(profile)[-1] == pytest.approx(.06)
    assert surface_displacement(profile, residual_m=np.array([0, .04]),
                                reference_month="2023-01")[-1] == pytest.approx(.10)
    with pytest.raises(ValueError, match="exactly resolved"):
        ring_interval(profile, 50, 200, reference_month="2023-01")
    with pytest.raises(ValueError, match="reference"):
        ring_interval(profile, 100, 200, reference_month="2024-01")
    with pytest.raises(ValueError, match="residual"):
        surface_displacement(profile, residual_m=None, reference_month="2023-01")


def test_boundary_excess_vanishes_for_linear_gradient_and_detects_step():
    mask = np.ones((2, 6), dtype=bool)
    pairs = cross_boundary_pairs(mask, 179000., 1000., 182.)
    assert pairs.tolist() == [[1, 2, 3, 4], [7, 8, 9, 10]]
    linear = np.tile(np.arange(6, dtype=float) * .01, 2)
    assert jump_diagnostic(linear, pairs)["excess"]["median_absolute_cm"] < 1e-10
    stepped = linear + np.tile(np.array([0, 0, 0, .025, .025, .025]), 2)
    assert jump_diagnostic(stepped, pairs)["excess"]["median_cm"] == pytest.approx(2.5)
    mask[:, 3] = False
    with pytest.raises(ValueError, match="No complete"):
        cross_boundary_pairs(mask, 179000., 1000., 182.)
