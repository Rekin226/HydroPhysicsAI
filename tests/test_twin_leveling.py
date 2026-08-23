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


def test_planar_tectonic_removes_a_regional_tilt_but_not_local_signal():
    xy, sub = {}, {}
    times = pd.DatetimeIndex([pd.Timestamp(f"{y}-06-15") for y in range(2012, 2020)])
    yrs = np.array([(t - times[0]).days / 365.25 for t in times])
    for i in range(40):
        x = 170000.0 + 1000.0 * i
        y = 2620000.0 + 500.0 * i
        xy[f"S{i}"] = (x, y)
        tect = 0.002 + 1e-8 * (x - 170000.0)              # regional tilt, m/yr
        local = 0.01 if i % 2 == 0 else 0.0               # site-specific compaction
        sub[f"S{i}"] = pd.Series((tect + local) * yrs, index=times)

    out, info = leveling.remove_tectonic(sub, xy, mode="planar")
    even = np.mean([out[f"S{i}"].iloc[-1] for i in range(0, 40, 2)])
    odd = np.mean([out[f"S{i}"].iloc[-1] for i in range(1, 40, 2)])
    assert even - odd == pytest.approx(0.01 * yrs[-1], rel=0.05)   # local signal survives
    assert abs(odd) < 0.2 * abs(even)                              # regional tilt removed
    assert 0.0 <= info["var_removed"] <= 1.0


def test_tectonic_mode_none_is_identity():
    p = _panel()
    sub = leveling.site_subsidence(p, "2012-01-01", "2018-01-01", min_obs=5)
    out, info = leveling.remove_tectonic(sub, leveling.site_xy(p), mode="none")
    assert np.allclose(out["A"].to_numpy(), sub["A"].to_numpy())
    assert info["var_removed"] == 0.0
