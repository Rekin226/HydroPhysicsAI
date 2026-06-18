"""Smoke + leakage tests for the decomposition simulation model. No torch needed."""

from __future__ import annotations

import copy

import numpy as np

from hydrophysics import Config, evaluate_predictions, load_dataset
from hydrophysics.decomp import DecompConfig, DecompModel
from hydrophysics.sample import write_sample


def _data(tmp_path):
    d = write_sample(tmp_path / "data", n_wells=4, seed=5)
    return load_dataset(Config(data_dir=d))


def test_fit_simulate_shapes(tmp_path):
    data = _data(tmp_path)
    pred = DecompModel(DecompConfig()).fit(data).simulate(data)
    assert pred.shape == data.target.shape
    assert np.isfinite(pred).all()
    table = evaluate_predictions(data, pred, period="val")
    assert table["kge"].notna().any()


def test_leakage_free(tmp_path):
    """Corrupting every validation-period groundwater value must not change the
    validation predictions at all -- the model must never read the val target."""
    data = _data(tmp_path)
    cfg = DecompConfig(use_trend=True, trend_damping=0.99)
    clean = DecompModel(cfg).fit(data).simulate(data)

    corrupt = copy.deepcopy(data)
    vm = corrupt.val_mask
    corrupt.target[:, vm] = 999.0
    poisoned = DecompModel(cfg).fit(corrupt).simulate(corrupt)

    assert np.nanmax(np.abs(clean[:, vm] - poisoned[:, vm])) == 0.0

    # ...but it MUST react to forcing (else it is a disguised constant)
    norain = copy.deepcopy(data)
    norain.rainfall[:, vm] = 0.0
    changed = DecompModel(cfg).fit(norain).simulate(norain)
    assert np.nanmax(np.abs(clean[:, vm] - changed[:, vm])) > 0.0
