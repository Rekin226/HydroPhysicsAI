"""Bounded tests for measured-input information diagnostics, without site fitting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.observed_head_information import (
    monthly_values,
    normalized_condition,
    scenario_table,
)
from hydrophysics.twin.site_geometry import TUKU_SOURCES

needs_tuku = pytest.mark.skipif(not TUKU_SOURCES.exists(),
                                reason="Tuku site tables are local-only, not in the public repo")


def test_monthly_coverage_filters_sentinels_and_sparse_month():
    times = pd.date_range("2023-01-01", "2023-02-28 23:50", freq="10min")
    series = pd.Series(2.0, index=times)
    series.loc["2023-01-01"] = -999998.0
    series.loc["2023-02-01":] = -999998.0
    monthly, count, invalid = monthly_values(series)
    assert monthly.loc["2023-01-01"] == 2.0
    assert np.isnan(monthly.loc["2023-02-01"])
    assert count.loc["2023-01-01"] == 30 * 144
    assert invalid == 29 * 144


def test_duplicate_head_timestamps_rejected():
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2023-01-01"] * 2))
    with pytest.raises(ValueError, match="unique datetime"):
        monthly_values(series)


@needs_tuku
def test_ordered_and_catalog_mapping_rank_bounds_are_conditional():
    t = np.arange(24.0)
    observations = np.stack([t, t + np.sin(t), 0.5 * t + np.cos(t)], axis=1)
    rows = scenario_table(observations)
    assert len(rows["scenarios"]) == 6
    assert rows["scenarios"]["ordered_slots_234"]["model_aquifer_slots_1based"] == [2, 3, 4]
    assert rows["scenarios"]["catalog_code2_well2_code3_well4"][
        "observed_well_indices_1based"] == [1, 3]
    assert rows["envelope"]["IOT_fine_support_7_hydro_zone_coefficients"]["max_rank"] < 7
    assert rows["envelope"]["WRA_3_confining_coefficients"]["min_rank"] == 3


def test_zero_column_is_rank_deficient():
    assert normalized_condition(np.array([[1.0, 0.0], [0.0, 0.0]])) == (1, None)
