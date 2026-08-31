import os

import numpy as np
import pytest

from hydrophysics.twin.zones import (
    DISTAL,
    MID,
    N_ZONES,
    PROXIMAL,
    ZONE_NAMES,
    fan_zones,
)

POLYGON = ("chou-shui-data/chou-shui-data/data/Zhuoshui Alluvial Fan/"
           "Zhuoshui Alluvial Fan.json")
STATIONS = "AMP_V2/data/fan_stations.parquet"
WELLS_DIR = "AMP_V2/data/wells"


def _xy_km(*xs_km):
    """(n, 2) EPSG:3826 metres from a list of eastings in km; northing is irrelevant."""
    return np.array([[x * 1000.0, 2_640_000.0] for x in xs_km], dtype="float64")


def test_zone_constants_are_the_documented_ids():
    assert (PROXIMAL, MID, DISTAL) == (0, 1, 2)
    assert ZONE_NAMES == ("proximal", "mid", "distal")
    assert N_ZONES == 3


def test_fan_zones_assigns_each_zone_by_easting():
    z = fan_zones(_xy_km(214.0, 195.0, 170.0))
    assert z.tolist() == [PROXIMAL, MID, DISTAL]


def test_fan_zones_uses_half_open_intervals_inclusive_on_the_high_side():
    """205.0 is proximal, 204.999 is mid; 182.0 is mid, 181.999 is distal. The
    convention is documented in the spec and every count in this plan depends on it.
    """
    assert fan_zones(_xy_km(205.0))[0] == PROXIMAL
    assert fan_zones(_xy_km(204.999))[0] == MID
    assert fan_zones(_xy_km(182.0))[0] == MID
    assert fan_zones(_xy_km(181.999))[0] == DISTAL


def test_fan_zones_covers_every_point_with_exactly_one_zone_no_gaps_no_overlaps():
    xs = np.linspace(150.0, 240.0, 4001)
    z = fan_zones(_xy_km(*xs))
    assert z.shape == (xs.size,)
    assert set(np.unique(z).tolist()) <= {PROXIMAL, MID, DISTAL}
    assert np.isin(z, [PROXIMAL, MID, DISTAL]).all()


def test_fan_zones_is_monotone_west_to_east():
    """Zone id must decrease as easting increases: distal -> mid -> proximal, no
    interleaving. Guards against an off-by-one in the nested where().
    """
    xs = np.linspace(150.0, 240.0, 2001)
    z = fan_zones(_xy_km(*xs))
    assert (np.diff(z) <= 0).all()


def test_fan_zones_honours_custom_boundaries():
    z = fan_zones(_xy_km(190.0, 180.0), proximal_km=186.0, distal_km=178.0)
    assert z.tolist() == [PROXIMAL, MID]


def test_fan_zones_is_pure_and_deterministic():
    xy = _xy_km(214.0, 195.0, 170.0)
    before = xy.copy()
    a = fan_zones(xy)
    b = fan_zones(xy)
    assert np.array_equal(a, b)
    assert np.array_equal(xy, before)      # input not mutated
    assert a.dtype == np.int64


def test_fan_zones_returns_an_empty_array_for_empty_input():
    z = fan_zones(np.zeros((0, 2), dtype="float64"))
    assert z.shape == (0,)
    assert z.dtype == np.int64


def test_fan_zones_rejects_a_wrong_shaped_array():
    with pytest.raises(ValueError):
        fan_zones(np.zeros((5, 3), dtype="float64"))


def test_fan_zones_rejects_reversed_boundaries():
    """distal_km must sit west of proximal_km, or the mid zone is empty and every
    count downstream is silently wrong.
    """
    with pytest.raises(ValueError):
        fan_zones(_xy_km(200.0), proximal_km=180.0, distal_km=190.0)


@pytest.mark.skipif(not os.path.exists(POLYGON), reason="fan polygon not available")
def test_default_boundaries_split_the_real_grid_264_1235_649():
    """Spec §4/§7 pre-registered count, verified 2026-08-31 against the real polygon at
    dx=1000. A mismatch means the polygon or the boundary moved -- investigate it, do
    not edit this number.
    """
    from hydrophysics.twin.grid import build_grid

    grid = build_grid(POLYGON, dx=1000.0)
    z = fan_zones(grid.centroids())
    assert grid.n_active == 2148
    assert [int((z == k).sum()) for k in range(N_ZONES)] == [264, 1235, 649]


@pytest.mark.skipif(
    not (os.path.exists(POLYGON) and os.path.exists(STATIONS) and os.path.isdir(WELLS_DIR)),
    reason="fan well data not available",
)
def test_default_boundaries_split_the_66_calibration_sites_12_33_21():
    """Spec §4.1/§7 pre-registered count, verified 2026-08-31. The proximal zone holding
    only 12 of 66 sites is exactly why §5 gives it 2 free parameters and not 11.
    """
    import pandas as pd

    from hydrophysics.twin.grid import build_grid
    from hydrophysics.twin.heads import build_head_field

    grid = build_grid(POLYGON, dx=1000.0)
    stn = pd.read_parquet(STATIONS)
    stn = stn[stn.GroundwaterZoneIdentifier == 50].copy()
    stn["sid"] = stn["sid"].astype(str)
    hf = build_head_field(WELLS_DIR, stn)
    xy = np.array(
        [hf.xy[w] for w in range(len(hf))
         if grid.active_index(float(hf.xy[w, 0]), float(hf.xy[w, 1])) is not None],
        dtype="float64",
    )
    assert xy.shape[0] == 136
    sites = np.unique(np.round(xy, 3), axis=0)
    assert sites.shape[0] == 66
    z = fan_zones(sites)
    assert [int((z == k).sum()) for k in range(N_ZONES)] == [12, 33, 21]
