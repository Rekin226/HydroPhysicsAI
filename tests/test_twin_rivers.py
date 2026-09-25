"""River cells from channel polygons (rivers.py)."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("shapely")
pytest.importorskip("matplotlib")
pytest.importorskip("pyproj")

from shapely.geometry import box  # noqa: E402

from hydrophysics.twin.grid import FanGrid  # noqa: E402
from hydrophysics.twin.rivers import (  # noqa: E402
    RIVER_GROUPS,
    _flatten_stage,
    parse_river_set,
    river_cells,
)


def _grid():
    return FanGrid(nx=4, ny=3, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((3, 4), dtype=bool))


def test_a_box_channel_gets_exact_area_weights():
    g = _grid()
    # a 3 km x 0.5 km channel from x=500 to 3500 along the middle row's lower half
    poly = box(500.0, 1000.0, 3500.0, 1500.0)
    dem = np.arange(g.n_active, dtype="float64") + 10.0
    rs = river_cells(g, {"choushui": poly}, dem, stage_depth=1.0, rbot_depth=3.0)
    got = {int(i): float(w) for i, w in zip(rs.idx, rs.weight, strict=True)}
    row1 = [g.active_index(500.0 + 1000 * c, 1500.0) for c in range(4)]
    assert got == pytest.approx({row1[0]: 0.25, row1[1]: 0.5, row1[2]: 0.5, row1[3]: 0.25})
    assert np.allclose(rs.h_riv - rs.rbot, 2.0)
    assert rs.names == ("choushui",) and (rs.group == 0).all()


def test_tiny_slivers_are_dropped_and_edge_groups_get_unit_weight():
    g = _grid()
    sliver = box(990.0, 0.0, 1010.0, 3000.0)          # 1 % of each cell on either side
    with pytest.raises(ValueError, match="no river cell"):
        river_cells(g, {"choushui": sliver}, np.zeros(g.n_active) + 5.0)
    rs = river_cells(g, {"choushui": sliver, "wu": box(0.0, 2400.0, 4000.0, 2600.0)},
                     np.zeros(g.n_active) + 5.0, edge_buffer_m=0.0)
    assert (rs.weight >= 0.02).all() and rs.names == ("choushui", "wu")
    # an edge group lying just outside the grid still claims the bordering cells
    outside = box(0.0, -400.0, 4000.0, -100.0)
    rs = river_cells(g, {"beigang": outside}, np.zeros(g.n_active) + 5.0, edge_buffer_m=500.0)
    assert rs.n_cells == 4 and (rs.weight == 1.0).all()
    assert set(rs.idx.tolist()) == {g.active_index(500.0 + 1000 * c, 500.0) for c in range(4)}


def test_stage_never_rises_downstream_after_flattening():
    x = np.array([3500.0, 2500.0, 1500.0, 500.0, 3000.0])
    h = np.array([10.0, 12.0, 8.0, 9.0, 11.0])
    out = _flatten_stage(np.arange(5), h, x)
    order = np.argsort(-x)
    assert (np.diff(out[order]) <= 0).all()
    assert (out <= h).all()


def test_river_set_parsing():
    assert parse_river_set("choushui, wu") == ("choushui", "wu")
    with pytest.raises(ValueError):
        parse_river_set("nile")
    assert "minor" in RIVER_GROUPS


_DATA = os.environ.get("HYDROMIND_GW_DATA", os.path.join("chou-shui-data", "data"))
_SHP = os.path.join(_DATA, "water", "river_TWD97.shp")
_POLY = "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json"
_DEM = "results/twin/basemap.npz"
_STN = "AMP_V2/data/fan_stations.parquet"


@pytest.mark.skipif(not all(os.path.exists(p) for p in (_SHP, _POLY, _DEM)),
                    reason="real river layer / fan polygon / basemap DEM not cached")
def test_real_choushui_crosses_the_fan_and_srtm_agrees_with_the_collars():
    pytest.importorskip("geopandas")
    import pandas as pd

    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.rivers import build_river_set

    g = build_grid(_POLY)
    rs, sha = build_river_set(g, shp=_SHP, dem_npz=_DEM)
    assert len(sha) == 40
    c = g.centroids()[rs.idx]
    ch = rs.group == rs.names.index("choushui")
    in_band = (c[ch, 1] >= 2_630_000) & (c[ch, 1] <= 2_640_000)
    assert int(in_band.sum()) >= 20
    if os.path.exists(_STN):
        from hydrophysics.twin.calibrate_flow import _load_ground_elev

        stn = pd.read_parquet(_STN)
        stn = stn[stn.GroundwaterZoneIdentifier == 50]
        ge = _load_ground_elev(g, stn).numpy()
        with np.load(_DEM) as z:
            dem = z["dem"].astype("float64")
        # Registration check: SRTM and the collar-IDW ground elevation must describe the
        # same surface. Measured 2026-09-23 they agree in the distal fan (median |d| 3.2 m,
        # where the collar network is dense) and diverge toward the apex (SRTM +8.7 m in
        # the mid fan, +17.8 m in the proximal fan) -- the collar IDW flattens the apex,
        # which also feeds the pumping lift. So: strong correlation everywhere, and close
        # agreement where the collars are dense.
        assert float(np.corrcoef(dem, ge)[0, 1]) > 0.9          # measured 0.946
        distal = g.centroids()[:, 0] < 195_000
        assert float(np.median(np.abs(dem[distal] - ge[distal]))) < 3.5
