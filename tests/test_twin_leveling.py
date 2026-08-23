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
    """Sites on an 8x5 grid so the [1, x, y] design matrix is full rank.

    The plane is scaled to be comparable to the local signal, so a no-op implementation
    cannot pass: the uncorrected residual tracks easting strongly, the corrected one does
    not. The local signal is assigned at random (fixed seed) so it is not aligned with the
    coordinate axes and cannot be absorbed by the fitted plane.
    """
    rng = np.random.default_rng(0)
    xy, sub, local_flag = {}, {}, {}
    times = pd.DatetimeIndex([pd.Timestamp(f"{y}-06-15") for y in range(2012, 2020)])
    yrs = np.array([(t - times[0]).days / 365.25 for t in times])
    for i in range(40):
        x = 170000.0 + 1000.0 * (i % 8)
        y = 2620000.0 + 1000.0 * (i // 8)
        name = f"S{i}"
        xy[name] = (x, y)
        tect = 0.002 + 2e-6 * (x - 170000.0) + 1e-6 * (y - 2620000.0)   # regional plane
        local = 0.01 if rng.random() < 0.5 else 0.0                      # position-independent
        local_flag[name] = local
        sub[name] = pd.Series((tect + local) * yrs, index=times)

    out, info = leveling.remove_tectonic(sub, xy, mode="planar")
    raw, _ = leveling.remove_tectonic(sub, xy, mode="none")

    east = np.array([xy[n][0] for n in xy])
    corr_raw = abs(np.corrcoef(east, np.array([raw[n].iloc[-1] for n in xy]))[0, 1])
    corr_out = abs(np.corrcoef(east, np.array([out[n].iloc[-1] for n in xy]))[0, 1])

    # a no-op implementation cannot pass: the tilt must actually be detectable and removed
    assert corr_raw > 0.5
    assert corr_out < 0.2

    hi = [out[n].iloc[-1] for n in xy if local_flag[n] > 0]
    lo = [out[n].iloc[-1] for n in xy if local_flag[n] == 0]
    assert np.mean(hi) - np.mean(lo) == pytest.approx(0.01 * yrs[-1], rel=0.05)
    assert 0.2 < info["var_removed"] < 0.9


def test_tectonic_mode_none_is_identity():
    p = _panel()
    sub = leveling.site_subsidence(p, "2012-01-01", "2018-01-01", min_obs=5)
    out, info = leveling.remove_tectonic(sub, leveling.site_xy(p), mode="none")
    assert np.allclose(out["A"].to_numpy(), sub["A"].to_numpy())
    assert info["var_removed"] == 0.0


def test_planar_correction_preserves_the_basin_mean_rate():
    """The intercept is basin-mean compaction, not tectonics -- it must survive."""
    xy, sub = {}, {}
    times = pd.DatetimeIndex([pd.Timestamp(f"{y}-06-15") for y in range(2012, 2020)])
    yrs = np.array([(t - times[0]).days / 365.25 for t in times])
    for i in range(40):
        x = 170000.0 + 1000.0 * (i % 8)
        y = 2620000.0 + 1000.0 * (i // 8)
        name = f"S{i}"
        xy[name] = (x, y)
        # every site sinks at 2 cm/yr, plus a regional tilt centred on the grid's own mean
        # x (173500.0, since x spans 170000..177000): a tilt term must be zero-mean over
        # the site grid, or it leaks into the fitted intercept and the assertion below no
        # longer isolates the shared/basin-mean rate.
        rate = 0.02 + 2e-6 * (x - 173500.0)
        sub[name] = pd.Series(rate * yrs, index=times)

    out, _ = leveling.remove_tectonic(sub, xy, mode="planar")
    finals = np.array([out[n].iloc[-1] for n in xy])
    # the shared 2 cm/yr sinking must remain; only the tilt is removed
    assert np.mean(finals) == pytest.approx(0.02 * yrs[-1], rel=0.05)
    assert np.all(finals > 0.0)
