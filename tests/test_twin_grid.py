from hydrophysics.twin.grid import build_grid

POLY = ("chou-shui-data/chou-shui-data/data/Zhuoshui Alluvial Fan/"
        "Zhuoshui Alluvial Fan.json")


def test_grid_masks_the_fan_polygon():
    g = build_grid(POLY, dx=1000.0)
    assert g.mask.shape == (g.ny, g.nx)
    # the fan is ~2,144 km2; at 1 km cells that is ~2,100 active cells
    assert 1900 < g.n_active < 2400
    assert g.n_active == int(g.mask.sum())
    # the bounding box is not all fan: masking must actually reject cells
    assert g.n_active < g.nx * g.ny * 0.75


def test_cell_lookup_round_trips_through_centroids():
    g = build_grid(POLY, dx=1000.0)
    c = g.centroids()
    assert c.shape == (g.n_active, 2)
    for i in (0, g.n_active // 2, g.n_active - 1):
        x, y = c[i]
        assert g.active_index(float(x), float(y)) == i


def test_coordinates_outside_the_fan_return_none():
    g = build_grid(POLY, dx=1000.0)
    assert g.active_index(0.0, 0.0) is None
    assert g.cell_of(0.0, 0.0) is None


def test_finer_grid_gives_more_cells_covering_similar_area():
    coarse = build_grid(POLY, dx=1000.0)
    fine = build_grid(POLY, dx=500.0)
    area_c = coarse.n_active * 1.0        # km2
    area_f = fine.n_active * 0.25         # km2
    assert fine.n_active > coarse.n_active
    assert abs(area_f - area_c) / area_c < 0.05
