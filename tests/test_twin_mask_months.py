"""CPU tests for --mask-months (N1 masked refits)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from hydrophysics.twin.calibrate_flow import mask_month_flags, prepare_series  # noqa: E402

DATES = pd.date_range("2012-01-31", periods=132, freq="ME")


def test_no_window_is_none():
    assert mask_month_flags(None, DATES) is None
    assert mask_month_flags("", DATES) is None


def test_window_is_inclusive_and_exact():
    f = mask_month_flags("2015-01:2015-05", DATES)
    assert f.dtype == bool and f.shape == (132,)
    assert f.sum() == 5
    assert list(DATES[f].strftime("%Y-%m")) == [f"2015-0{m}" for m in range(1, 6)]


@pytest.mark.parametrize("spec", ["2015-05:2015-01", "2011-06:2012-03", "2022-11:2023-02",
                                  "2012-01:2012-02", "2015-1", "spring"])
def test_bad_windows_raise(spec):
    with pytest.raises(ValueError):
        mask_month_flags(spec, DATES)


def test_masked_months_stay_nan_after_backfill():
    """The main() order: mask the raw series, back-fill, then mask again, so the
    interpolation across the window never becomes a fit target."""
    f = mask_month_flags("2015-01:2015-05", DATES)
    raw = np.sin(np.arange(132) / 3.0) + 10.0
    raw[[40, 41]] = np.nan                       # an ordinary gap elsewhere is still filled
    m = raw.copy()
    m[f] = np.nan
    s = prepare_series(m, backfill=True)
    s[f] = np.nan
    assert np.isnan(s[f]).all()
    assert np.isfinite(s[~f]).all()
    np.testing.assert_array_equal(s[~f & np.isfinite(raw)], raw[~f & np.isfinite(raw)])
