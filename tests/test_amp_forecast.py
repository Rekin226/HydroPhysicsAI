from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")

from AMP_V2.ampg import ampg, classify, detect_lines  # noqa: E402
from AMP_V2.forecast_experiment import (  # noqa: E402
    evaluate,
    load_series,
    monthly_heads,
    origin_features,
)


def signal():
    index = pd.date_range("2019-01-01", "2020-01-01", freq="h", inclusive="left")
    t = np.arange(len(index)) / 24
    return pd.Series(10 + .3 * np.cos(2 * np.pi * t) + .1 * np.cos(4 * np.pi * t), index=index)


def test_future_values_cannot_change_origin_features():
    s = signal()
    origin = pd.Timestamp("2019-09-01")
    a = origin_features(s, origin)
    s.loc[origin:] = 10000 * np.arange(len(s.loc[origin:]))
    b = origin_features(s, origin)
    assert a["status"] == "ok"
    for key in a:
        if isinstance(a[key], float):
            np.testing.assert_allclose(a[key], b[key], equal_nan=True)
        else:
            assert a[key] == b[key]
    assert .2 < a["amp_v1"] < .4


def test_long_gaps_are_not_compressed_or_interpolated():
    s = signal()
    s.loc["2019-08-01":"2019-08-02"] = np.nan
    assert origin_features(s, "2019-09-01")["status"] == "long_gap"


def test_missing_endpoint_is_not_extrapolated():
    s = signal()
    s.loc["2019-08-31 23:00"] = np.nan
    assert origin_features(s, "2019-09-01")["status"] == "edge_gap"


def test_short_window_preserves_adjacent_harmonics_and_recovers_duty():
    t = np.arange(180 * 24) / 24
    ns = np.arange(1, 4)
    duty, attenuation = .3, .2
    amplitudes = np.abs(np.sin(ns * np.pi * duty)) / ns * np.exp(-attenuation * ns)
    x = sum(a * np.cos(2 * np.pi * n * t) for a, n in zip(amplitudes, ns, strict=True))
    # Regression: the full-record default's 200-bin peak separation erased a harmonic.
    old = classify(detect_lines(x, 24))
    assert len(old["harmonics"]) < 2
    current = ampg(x, 24, floor_win=91)
    assert current["n_harm"] >= 2
    assert current["duty_fit"] == pytest.approx(duty, abs=.01)


def test_monthly_target_requires_coverage():
    s = signal()
    s.loc["2019-08-01":"2019-08-15"] = np.nan
    m = monthly_heads(s)
    assert np.isnan(m.loc["2019-08-01"])
    assert np.isfinite(m.loc["2019-07-01"])


def panel():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2014-01-01", "2025-12-01", freq="MS")
    y = np.sin(2 * np.pi * dates.month / 12) + rng.normal(0, .2, len(dates))
    return pd.DataFrame({"well": "synthetic", "origin": dates, "observed": y,
                         "previous": np.roll(y, 1), "previous2": np.roll(y, 2),
                         "energy": rng.uniform(100, 200, len(y)),
                         "amp_v1": rng.uniform(.1, .3, len(y)),
                         "amp_v2": rng.uniform(.1, .3, len(y)),
                         "amp_duty": np.nan, "status": "ok"})


def test_test_targets_do_not_change_predictions_or_model_selection():
    p = panel()
    report, first = evaluate(p)
    p.loc[p.origin >= "2023-01-01", "observed"] += 100
    changed, second = evaluate(p)
    assert report["selection"] == changed["selection"]
    for name in report["scores"]:
        np.testing.assert_allclose(first[name], second[name])
    assert report["duty_test_rows"] == 0


def test_identical_test_rows_for_every_arm():
    p = panel()
    p.loc[p.origin == "2023-03-01", "amp_v2"] = np.nan
    report, predictions = evaluate(p)
    assert report["counts"]["test"] == 35
    assert not predictions[list(report["scores"])].isna().any().any()


def test_conflicting_vintages_are_rejected(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    s = signal().iloc[:4]
    s.rename("value").to_frame().to_parquet(a)
    (s + 1).rename("value").to_frame().to_parquet(b)
    with pytest.raises(ValueError, match="Conflicting"):
        load_series([a, b])
