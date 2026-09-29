"""surface_water.py v3 (``--jiji-supply``, 2026-09-29): year-to-year canal deliveries from
the Jiji weir's supply to each district, with the documented 2020 (crop 2) and 2021
(crop 1) shortfalls. Pins annual conservation, the crop timing, the floor, the MOA level,
the percolation cap, and that v3 cannot be combined with the v2 rotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin import surface_water as swm

SHARE = {"changhua": 0.48, "yunlin": 0.33}


def _factors(years=range(2012, 2023), **kw):
    return swm.jiji_crop_factors(years, SHARE, **kw)


def test_published_table_values_and_reference_mean():
    assert swm.JIJI_SUPPLY[2021][swm.JIJI_COL["changhua"]] == pytest.approx(6.11)
    assert swm.JIJI_SUPPLY[2020][swm.JIJI_COL["yunlin"]] == pytest.approx(6.98)
    f = _factors()
    for d in swm.DISTRICTS:
        ref = f[d].loc[list(swm.JIJI_REF_YEARS), "ratio"]
        assert ref.mean() == pytest.approx(1.0)                   # ratio to its own mean


def test_shortfall_lands_on_the_documented_crop_and_conserves_the_year():
    f = _factors()
    for d in swm.DISTRICTS:
        t, s1 = f[d], SHARE[d]
        annual = s1 * t["crop1"] + (1 - s1) * t["crop2"]
        np.testing.assert_allclose(annual.to_numpy(), t["ratio"].to_numpy())
        assert t.loc[2020, "crop1"] == pytest.approx(1.0)         # 2020: crop 2 carries it
        assert t.loc[2020, "crop2"] < t.loc[2020, "ratio"]
        assert t.loc[2021, "crop2"] == pytest.approx(1.0)         # 2021: crop 1 carries it
        assert t.loc[2021, "crop1"] < t.loc[2021, "ratio"]
        assert t.loc[2022, "crop1"] == t.loc[2022, "crop2"] == t.loc[2022, "ratio"]
        assert t.loc[2021, "timing"] == "crop1" and t.loc[2019, "timing"] == "both"
    # Changhua crop 1 of 2021 against its documented intake (12-14 of 25-41 cms)
    assert 0.3 < f["changhua"].loc[2021, "crop1"] < 0.5


def test_floor_spills_the_remainder_onto_the_other_crop():
    table = {2012: (10.0, 10.0, 0.0), 2013: (10.0, 10.0, 0.0), 2014: (4.0, 4.0, 0.0)}
    f = swm.jiji_crop_factors([2014], SHARE, table=table, ref_years=(2012, 2013),
                              shortfall_crop={2014: 1}, min_factor=0.1)
    t = f["yunlin"]
    assert t.loc[2014, "crop1"] == pytest.approx(0.1)             # floored
    s1 = SHARE["yunlin"]
    assert s1 * 0.1 + (1 - s1) * t.loc[2014, "crop2"] == pytest.approx(0.4)
    assert t.loc[2014, "crop2"] < 1.0
    # a year above its mean is never redistributed, even if listed
    f2 = swm.jiji_crop_factors([2012], SHARE, table=table, ref_years=(2014,),
                               shortfall_crop={2012: 1})
    assert f2["changhua"].loc[2012, "crop1"] == f2["changhua"].loc[2012, "crop2"] == 2.5


def test_missing_years_are_refused():
    with pytest.raises(ValueError, match="no 2030"):
        _factors(years=[2030])


def test_jiji_seasons_keep_the_moa_reference_level():
    years = list(range(2012, 2023))
    seasons = {}
    for d in swm.DISTRICTS:
        t = pd.DataFrame({"crop1": np.linspace(8e8, 4e8, len(years)),
                          "crop2": np.linspace(9e8, 5e8, len(years)),
                          "split": "share"}, index=pd.Index(years, name="year"))
        t.attrs["share_crop1"] = SHARE[d]
        seasons[d] = t
    f = _factors()
    v3 = swm.jiji_seasons(seasons, f)
    ref = list(swm.JIJI_REF_YEARS)
    for d in swm.DISTRICTS:
        b1 = seasons[d].loc[ref, "crop1"].mean()
        assert v3[d].loc[2021, "crop1"] == pytest.approx(b1 * f[d].loc[2021, "crop1"])
        # the MOA trend is gone: a year at ratio 1 sits exactly at the MOA ref mean
        assert v3[d].loc[ref, "crop1"].mean() == pytest.approx(
            b1 * f[d].loc[ref, "crop1"].mean())
        assert v3[d].attrs["share_crop1"] == SHARE[d]
        assert (v3[d]["split"] == "jiji").all()
    with pytest.raises(ValueError, match="reference years"):
        swm.jiji_seasons({d: s.loc[[2021, 2022]] for d, s in seasons.items()}, f)


def test_percolation_duty_follows_the_crop_and_is_capped():
    f = _factors()
    duty = swm.jiji_percolation_duty(f)
    ch = f["changhua"]
    assert duty[("changhua", 2021)][4] == pytest.approx(ch.loc[2021, "crop1"])
    assert 8 not in duty[("changhua", 2021)]                     # crop 2 at 1: omitted
    assert duty[("changhua", 2020)][9] == pytest.approx(ch.loc[2020, "crop2"])
    assert ("yunlin", 2016) not in duty                          # 2016 > 1, capped to 1
    assert all(0.0 < k <= 1.0 for m_k in duty.values() for k in m_k.values())
    assert all(m in swm.F_FLOOD for m_k in duty.values() for m in m_k)


def test_v3_field_on_a_toy_fan_scales_the_cut_months_only():
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyproj")
    from hydrophysics.twin.grid import FanGrid

    g = FanGrid(nx=3, ny=1, dx=1000.0, x0=0.0, y0=0.0, mask=np.ones((1, 3), dtype=bool))
    A = g.n_active
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    farm = {d: {"paddy": np.full(A, 2.5e5), "dry": np.zeros(A),
                "paddy_total_m2": 2.5e5 * A, "dry_total_m2": 0.0} for d in swm.DISTRICTS}
    years = list(range(2012, 2023))
    seasons = {}
    for d in swm.DISTRICTS:
        t = pd.DataFrame({"crop1": 1e8, "crop2": 1e8, "split": "share"},
                         index=pd.Index(years, name="year"))
        t.attrs["share_crop1"] = SHARE[d]
        seasons[d] = t
    f = _factors()
    base = swm.build_field(g, seasons, farm, dates)["sw_m_per_day"]
    v3 = swm.build_field(g, swm.jiji_seasons(seasons, f), farm, dates)["sw_m_per_day"]
    i = {k: dates.get_loc(pd.Timestamp(k)) for k in ("2020-04-01", "2020-08-01",
                                                     "2021-04-01", "2021-08-01")}
    # a toy MOA level of 1e8 per crop and district: v3 = base x the crop factor, summed
    for key, crop, y in (("2020-04-01", 1, 2020), ("2020-08-01", 2, 2020),
                         ("2021-04-01", 1, 2021), ("2021-08-01", 2, 2021)):
        want = sum(f[d].loc[y, f"crop{crop}"] for d in swm.DISTRICTS) / 2
        assert v3[:, i[key]] == pytest.approx(base[:, i[key]] * want)
    assert v3[:, i["2021-04-01"]].sum() < v3[:, i["2021-08-01"]].sum()
    perc = swm.build_percolation(g, farm, dates, None, np.array([0, 1, 2]),
                                 duty=swm.jiji_percolation_duty(f))
    perc0 = swm.build_percolation(g, farm, dates, None, np.array([0, 1, 2]))
    k21 = sum(f[d].loc[2021, "crop1"] for d in swm.DISTRICTS) / 2
    assert perc[:, i["2021-04-01"]] == pytest.approx(perc0[:, i["2021-04-01"]] * k21)
    assert perc[:, i["2021-08-01"]] == pytest.approx(perc0[:, i["2021-08-01"]])


def test_jiji_supply_refuses_the_v2_rotation():
    pytest.importorskip("torch")
    with pytest.raises(SystemExit):
        swm.main(["--jiji-supply", "--drought-duty"])
