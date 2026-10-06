"""Synthetic checks for depth integration and four-driver identifiability."""
import numpy as np
import pytest

from hydrophysics.twin.site_geometry import load_tuku_geometry
from hydrophysics.twin.synthetic_depth_identifiability import (
    analyze,
    cumulative,
    designs,
    matrix,
    ring_depths,
    synthetic_heads,
)


def test_ring_shortening_telescopes_to_depth_displacement():
    g = load_tuku_geometry()
    depth = ring_depths(g)
    h = synthetic_heads()
    for structure in designs(g).values():
        k = np.full(len(structure.parameter_names), 2e-4)
        ring = (matrix(structure, h, depth) @ k).reshape(len(h), -1)
        from_depth = np.diff(cumulative(structure, h, k, depth), axis=1)
        np.testing.assert_allclose(ring, from_depth, rtol=0, atol=1e-15)
        np.testing.assert_allclose(ring.sum(axis=1),
                                   cumulative(structure, h, k, depth)[:, -1]
                                   - cumulative(structure, h, k, depth)[:, 0],
                                   rtol=0, atol=1e-15)


def test_rank_recovery_and_positive_null_pair():
    result = analyze()
    assert result["status"].startswith("SYNTHETIC")
    rows = result["designs"]
    wra = rows["WRA_3_confining_coefficients"]
    fine = rows["IOT_29_fine_interval_coefficients"]
    reduced = rows["IOT_fine_support_7_hydro_zone_coefficients"]
    assert wra["basis_rank"] == wra["practical_rank"] == 3
    assert wra["noiseless_recovery_max_abs"] < 1e-15
    assert wra["gate"] == "passes_synthetic_design_only"
    assert fine["basis_rank"] == fine["practical_rank"] == 24
    assert fine["nullity"] == 5
    assert fine["gate"] == "reject_structural_rank_deficiency"
    pair = fine["positive_indistinguishable_pair"]
    assert pair["min_alternative_positive_compliance"] > 0
    assert pair["max_coefficient_difference"] == pytest.approx(1e-4)
    assert pair["max_ring_prediction_difference_m"] < 1e-15
    assert reduced["basis_rank"] == reduced["practical_rank"] == 7
    assert reduced["noiseless_recovery_max_abs"] < 1e-15
    assert reduced["gate"] == "reject_practical_at_declared_noise"
    assert 0.5 < reduced["max_relative_95_halfwidth_at_synthetic_noise"] < 0.7
    assert all(row["ring_cumulative_identity_error_m"] < 1e-15
               and row["whole_column_identity_error_m"] < 1e-15
               for row in rows.values())


def test_practical_gate_is_explicit_about_assumptions():
    generous = analyze(relative_95_tolerance=0.7)["designs"]
    assert generous["IOT_fine_support_7_hydro_zone_coefficients"]["gate"] == (
        "passes_synthetic_design_only"
    )
    assert generous["IOT_29_fine_interval_coefficients"]["gate"] == (
        "reject_structural_rank_deficiency"
    )
    with pytest.raises(ValueError, match="Positive finite"):
        analyze(sigma_m=0)
