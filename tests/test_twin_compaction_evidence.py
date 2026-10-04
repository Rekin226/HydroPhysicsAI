"""Validate the locked ring operator and matched annual compaction-rate scoring."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.compaction_evidence import (
    _annual_pairs,
    _equal_site_scores,
    evaluate,
    monthly_compaction,
)
from hydrophysics.twin.release import sha256


def test_fixed_ring_pair_yields_positive_shortening():
    dates = pd.to_datetime(["2023-01-10", "2023-02-10", "2024-01-10"])
    wide = pd.DataFrame({"NO1": [1.0, 1.0, 1.0],
                         "NO26": [10.0, 9.99, 9.90]}, index=dates)
    compaction = monthly_compaction(wide, ("NO1", "NO26"))
    assert compaction.iloc[-1] == pytest.approx(0.1)
    assert _annual_pairs(compaction)[pd.Period("2024-01")] == pytest.approx(10.0)


def test_rate_score_weights_sites_equally():
    frame = pd.DataFrame({"site": ["a", "b", "b", "b"],
                          "observed_cm": [10.0, 0.0, 0.0, 0.0],
                          "model": [0.0, 0.0, 0.0, 0.0]})
    assert _equal_site_scores(frame, ["model"])["model"]["rmse_cm_per_year"] == pytest.approx(
        np.sqrt(50.0))


def test_locked_compaction_evaluation_and_source_hash(tmp_path):
    pytest.importorskip("pyarrow")
    cache = tmp_path / "cache"
    cache.mkdir()
    old_path = cache / "ls-wra-mlcw-obs___41.parquet"
    old_dates = pd.date_range("2021-01-01", "2022-12-01", freq="MS")
    pd.DataFrame({"NO1": np.ones(24), "NO26": 10 - np.arange(24) * 0.001},
                 index=old_dates).to_parquet(old_path)
    new_dates = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
    ring_path = tmp_path / "B.parquet"
    pd.DataFrame({"datetime": np.repeat(new_dates, 2),
                  "depth": ["NO1", "NO26"] * 24,
                  "value": np.stack([np.ones(24), 10 - np.arange(24) * 0.01],
                                    axis=1).reshape(-1)}).to_parquet(ring_path)
    stations = tmp_path / "stations.parquet"
    pd.DataFrame({"sid": ["B"], "X_3826": [500.0],
                  "Y_3826": [500.0]}).to_parquet(stations)
    forward = tmp_path / "frozen.npz"
    model_dates = pd.date_range("2022-12-01", "2024-12-01", freq="MS")
    np.savez(forward, dates=np.array([str(date.date()) for date in model_dates]),
             origin=0, nx=1, ny=1, dx=1000.0, x0=0.0, y0=0.0,
             mask=np.ones((1, 1), dtype=bool), scenario_names=["baseline"],
             subs_mean=(np.arange(25) * 0.01)[None, None, :])
    files = {key: {"path": str(path), "sha256": sha256(path)}
             for key, path in {"stations": stations, "ring_0": ring_path,
                               "test_model": forward}.items()}
    protocol = {"schema": 1, "no_model_refit_or_selection": True,
                "measurement_operator": "fixed registered shallow/deep ring pair; shortening is positive compaction",
                "monthly_alignment": "last complete survey per calendar month; model state at same calendar month; no interpolation",
                "sites": ["B"], "ring_pairs": {"B": ["NO1", "NO26"]},
                "start_inclusive": "2023-01-01", "end_inclusive": "2024-12-01",
                "historical_cutoff_exclusive": "2023-01-01",
                "historical_cache_dir": str(cache),
                "historical_cache_files": {old_path.name: sha256(old_path)},
                "min_annual_pairs_per_site": 1, "files": files}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    result = evaluate(path)
    assert result["evaluation_annual_pairs"] == 12
    assert result["distinct_site_units"] == 1
    assert result["decision_ready"] is False
    assert result["by_year_exploratory"]["2024"]["annual_pairs"] == 12
    assert result["scores"]["test_model"]["rmse_cm_per_year"] < 1e-9
    assert result["scores"]["zero_cm"]["rmse_cm_per_year"] == pytest.approx(12)
    ring_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source hash changed"):
        evaluate(path)
