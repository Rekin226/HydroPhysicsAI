"""Public 土庫 site tables constrain geometry, not constitutive response."""
from __future__ import annotations

import numpy as np
import pytest

from hydrophysics.twin.site_geometry import (
    installed_ring_overlap,
    load_tuku_geometry,
    overlap_lengths,
)


def test_tuku_source_tables_have_checked_station_version_and_depths():
    site = load_tuku_geometry()
    assert site.source["ring_station_id"] == "A130600GWMWTKJS"
    assert site.source["location"]["crs"] == "EPSG:3826"
    assert site.soil_bounds_m_bgs.shape == (47, 2)
    assert site.installed_ring_depths_m_bgs.shape == (26,)
    assert site.source_layer_numbers[38] == 49  # printed typo kept, sequence is 39
    assert site.hydro_bounds_m_bgs[:, 0].tolist() == [0, 34, 51, 171, 193, 246, 260]


def test_actual_installed_ring_intervals_conserve_thickness_across_soil_contacts():
    site = load_tuku_geometry()
    first = installed_ring_overlap(site, 1, 2)
    assert first.sum() == pytest.approx(12.02 - 9.85)
    assert first[1] == pytest.approx(0.35)  # CL, 9.85–10.2 m
    assert first[2] == pytest.approx(1.82)  # SP, 10.2–12.02 m
    middle = installed_ring_overlap(site, 12, 13)
    assert middle[23] == pytest.approx(3.04)  # CL, 172.96–176 m
    assert middle[24] == pytest.approx(1.77)  # ML, 176–177.77 m
    hydro = installed_ring_overlap(site, 12, 13, basis="hydro")
    assert hydro[3] == pytest.approx(4.81)  # entirely in interpreted confining unit 2
    assert hydro.sum() == pytest.approx(middle.sum())


def test_logged_coverage_refuses_deepest_installed_interval_and_invalid_geometry():
    site = load_tuku_geometry()
    assert installed_ring_overlap(site, 24, 25).sum() == pytest.approx(2.25)
    with pytest.raises(ValueError, match="exceeds logged depth"):
        installed_ring_overlap(site, 25, 26)  # ring 26 is at 302.01 m, log ends at 300 m
    with pytest.raises(ValueError, match="Invalid installed ring"):
        installed_ring_overlap(site, 2, 1)
    with pytest.raises(ValueError, match="Bounds"):
        overlap_lengths(np.array([[0., 1.], [2., 3.]]), 0.5, 2.5)
