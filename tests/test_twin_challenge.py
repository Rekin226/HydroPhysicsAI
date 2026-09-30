"""Frozen-forecast challenge must not learn from its new observations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.challenge import monthly_observations, score


def test_datum_and_baselines_do_not_use_holdout():
    old = np.tile(np.arange(48, dtype=float) % 12, (2, 1))
    old[:, 36:] = 999  # even if supplied, future historical-cache values are ignored
    prediction = np.tile(np.arange(48, dtype=float) % 12 - 2, (2, 1))
    observed = prediction + 2
    first, detail = score(prediction, old, observed, 36)
    assert first["metrics"]["model_datum"]["rmse_m"] == pytest.approx(0)
    assert not first["prospective_validation"]
    observed[:, 36:] += 100
    second, changed = score(prediction, old, observed, 36)
    assert second["metrics"]["model_datum"]["rmse_m"] == pytest.approx(100)
    np.testing.assert_array_equal(changed["offsets"], detail["offsets"])
    for name in detail["predictions"]:
        np.testing.assert_array_equal(changed["predictions"][name], detail["predictions"][name])


def test_monthly_qc_removes_sentinels_and_sparse_months(tmp_path):
    dates = pd.date_range("2023-01-01", "2023-02-10", freq="h")
    values = np.full(len(dates), 4.0)
    values[:24] = -999998
    path = tmp_path / "well.parquet"
    pd.DataFrame({"value": values}, index=dates).to_parquet(path)
    result = monthly_observations(path)
    assert result.loc["2023-01-01"] == 4
    assert np.isnan(result.loc["2023-02-01"])


def test_insufficient_new_record_cannot_be_scored():
    data = np.ones((2, 48))
    new = np.full_like(data, np.nan)
    new[:, -1] = 1
    with pytest.raises(ValueError, match="sufficient"):
        score(data, data, new, 36)
