"""Foundation tests: data loading, metrics, baselines, eval. No torch needed."""

from __future__ import annotations

import numpy as np
import pytest

from hydrophysics import (
    Config,
    benchmark_table,
    evaluate_predictions,
    kge,
    load_dataset,
    load_graybox_baseline,
    nse,
    persistence_prediction,
    rmse,
)
from hydrophysics.baselines import climatology_prediction, last_value_prediction
from hydrophysics.sample import write_sample


@pytest.fixture()
def sample_cfg(tmp_path):
    d = write_sample(tmp_path / "data", n_wells=4, seed=1)
    return Config(data_dir=d, baseline_results=d / "gw_fit_results.csv")


def test_metrics_perfect_and_nan():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    assert kge(obs, obs) == pytest.approx(1.0)
    assert nse(obs, obs) == pytest.approx(1.0)
    assert rmse(obs, obs) == pytest.approx(0.0)
    assert np.isnan(rmse(np.array([np.nan, np.nan]), np.array([1.0, 2.0])))


def test_constant_prediction_has_undefined_kge():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) + 30.0
    const = np.full_like(obs, 33.0)
    assert np.isnan(kge(obs, const))
    assert nse(obs, const) < 1.0


def test_load_dataset_shapes(sample_cfg):
    data = load_dataset(sample_cfg)
    assert data.n_wells == 4
    assert data.target.shape == (data.n_wells, data.n_days)
    assert data.rainfall.shape == data.target.shape
    assert data.upstream.shape == data.target.shape
    assert not np.any(data.train_mask & data.val_mask)
    assert np.all(data.train_mask | data.val_mask)
    assert list(data.attrs.index) == data.well_ids
    assert "is_coastal" in data.attrs.columns


def test_baselines_and_benchmark(sample_cfg):
    data = load_dataset(sample_cfg)
    preds = {
        "persistence": persistence_prediction(data),
        "climatology": climatology_prediction(data),
        "last_value": last_value_prediction(data),
    }
    for p in preds.values():
        assert p.shape == data.target.shape
    gb = load_graybox_baseline(sample_cfg)
    table = benchmark_table(data, preds, graybox=gb)
    assert "graybox_ode" in table.index
    assert {"kge_median", "rmse_median", "n_wells"} <= set(table.columns)


def test_perfect_model_tops_table(sample_cfg):
    data = load_dataset(sample_cfg)
    table = benchmark_table(data, {"perfect": data.target.copy(),
                                   "persistence": persistence_prediction(data)})
    assert table.index[0] == "perfect"
    assert table.loc["perfect", "kge_median"] == pytest.approx(1.0, abs=1e-6)
