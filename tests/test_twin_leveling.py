import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin import leveling


def _panel():
    """Two sites: A sinks 1 cm/yr over 6 annual surveys, B is stable.

    Columns mirror ``load_panel``'s output schema (x/y, already renamed), which is what
    ``site_subsidence`` and ``site_xy`` consume.
    """
    rows = []
    for i, yr in enumerate(range(2012, 2018)):
        rows.append({"sid": "A", "datetime": pd.Timestamp(f"{yr}-06-15"),
                     "elev_m": 10.0 - 0.01 * i, "x": 180000.0, "y": 2620000.0})
        rows.append({"sid": "B", "datetime": pd.Timestamp(f"{yr}-06-15"),
                     "elev_m": 5.0, "x": 190000.0, "y": 2630000.0})
    return pd.DataFrame(rows)


def test_site_subsidence_is_positive_and_rezeroed():
    sub = leveling.site_subsidence(_panel(), "2012-01-01", "2018-01-01", min_obs=5)
    assert set(sub) == {"A", "B"}
    a = sub["A"]
    assert a.iloc[0] == pytest.approx(0.0)          # re-zeroed to first observation
    assert a.iloc[-1] == pytest.approx(0.05)        # 5 cm of sinking, positive
    assert np.allclose(sub["B"].to_numpy(), 0.0)    # stable site stays flat


def test_min_obs_filter_drops_short_records():
    p = _panel()
    p = p[~((p.sid == "B") & (p.datetime > pd.Timestamp("2013-01-01")))]
    sub = leveling.site_subsidence(p, "2012-01-01", "2018-01-01", min_obs=5)
    assert set(sub) == {"A"}


def test_site_xy_returns_epsg3826_metres():
    xy = leveling.site_xy(_panel())
    assert xy["A"] == (180000.0, 2620000.0)
