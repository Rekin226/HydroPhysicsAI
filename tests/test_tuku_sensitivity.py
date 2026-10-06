"""Bounded, synthetic checks for the one-cell head proxy."""
import numpy as np
import pytest

from hydrophysics.twin.tuku_sensitivity import analyze, support_weights


def test_source_supports_are_normalized_and_distinct():
    weights = support_weights()
    assert set(weights) == {"WRA_2003_confining_units", "IOT_2017_CL_ML"}
    for w in weights.values():
        assert w.shape == (4,)
        assert np.all(w >= 0)
        assert w.sum() == pytest.approx(1)
    assert not np.allclose(*weights.values())


def test_ranking_depends_on_layer_response(tmp_path):
    # The source site lies in active cell 0; all other cells are inactive.
    heads = np.zeros((3, 4, 1, 1), dtype=np.float32)
    heads[1, :, 0, 0] = [0, 2, 2, 2]
    heads[2, :, 0, 0] = [1, 1, 1, 1]
    path = tmp_path / "forward.npz"
    np.savez(path, heads_mean=heads, scenario_names=["baseline", "cut30", "retire_aqua"],
             dates=["2032-12-01"], mask=np.array([[True]]),
             x0=187000., y0=2620000., dx=1000.)
    result = analyze(path)
    assert result["grid_cell"] == 0
    assert result["nonnegative_weight_ranking"] == "depends_on_unknown_weights"
    assert result["cut30_minus_retire_aqua_by_aquifer_m"] == [-1., 1., 1., 1.]
    for support in result["source_supports"].values():
        assert support["cut30_proxy_m"] > support["retire_aqua_proxy_m"]
    with pytest.raises(ValueError, match="Date absent"):
        analyze(path, "2031-12-01")
