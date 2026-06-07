"""Smoke tests for the GPU models. Skipped automatically when torch is absent.

These do not check accuracy (a 3-epoch run on synthetic data proves nothing about
skill); they check that fit -> simulate produces correctly-shaped, finite predictions
that flow through the benchmark harness. Run on the GPU box for real training.
"""

from __future__ import annotations

import numpy as np
import pytest

from hydrophysics import Config, benchmark_table, load_dataset
from hydrophysics.sample import write_sample

torch = pytest.importorskip("torch")  # whole module skips if torch is not installed


@pytest.fixture()
def data(tmp_path):
    d = write_sample(tmp_path / "data", n_wells=4, seed=2)
    return load_dataset(Config(data_dir=d, baseline_results=d / "gw_fit_results.csv"))


@pytest.mark.parametrize("model_name", ["gru", "ude"])
def test_fit_simulate_shapes(data, model_name):
    from hydrophysics.train import build_model

    model = build_model(model_name, device="cpu", epochs=3)
    model.fit(data)
    pred = model.simulate(data)
    assert pred.shape == data.target.shape
    assert np.isfinite(pred).all()

    table = benchmark_table(data, {model.name: pred}, period="val")
    assert model.name in table.index
