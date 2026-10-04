"""Regression checks for fair scoring and annual scenario-rate interpretation."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.evidence import METHODS, _metrics, evaluate
from hydrophysics.twin.scenario_evidence import evaluate_sidecar


def test_fixed_origin_evidence_reproduces_saved_equal_well_score(tmp_path):
    rows = pd.DataFrame({
        "station_id": ["a", "a", "b"],
        "date": ["2023-01-01", "2023-02-01", "2023-01-01"],
        "observed_m": [0.0, 0.0, 0.0],
        "model_raw": [2.0, 2.0, 0.0],
        "model_datum": [2.0, 2.0, 0.0],
        "clim": [1.0, 1.0, 0.0],
        "clim_trend": [1.0, 1.0, 0.0],
        "persist": [3.0, 3.0, 0.0],
    })
    rows.to_csv(tmp_path / "predictions.csv", index=False)
    pd.DataFrame({"station_id": rows.station_id, "date": rows.date,
                  "head_m": rows.observed_m, "layer": [1, 1, 2]}).to_csv(
        tmp_path / "observations.csv", index=False)
    report = {"task": "newly-acquired-retrospective-head-challenge",
              "prospective_validation": False, "forecast_origin": "2022-12-01",
              "scored_months": 3, "stations": 2, "metrics": _metrics(rows),
              "model_sha256": "example", "beats_best_baseline": False,
              "best_baseline": "clim", "rmse_ratio": 2.0}
    (tmp_path / "report.json").write_text(json.dumps(report))
    events = tmp_path / "events.json"
    events.write_text(json.dumps([{"name": "example", "start": "2023-01-01",
                                   "end": "2023-02-01"}]))
    result, sites = evaluate(tmp_path, events)
    assert result["aggregate"]["metrics"]["model_datum"]["rmse_m"] == pytest.approx(
        np.sqrt(2.0))
    assert result["by_horizon"]["1-6"]["well_months"] == 3
    assert result["by_layer"]["2"]["wells"] == 1
    assert result["predictive_interval_coverage"] is None
    assert result["by_event"]["example"]["well_months"] == 2
    assert set(sites.station_id) == {"a", "b"}
    assert set(METHODS) <= set(result["aggregate"]["metrics"])


def test_fixed_origin_evidence_rejects_unmatched_observation(tmp_path):
    rows = pd.DataFrame({"station_id": ["a"], "date": ["2023-01-01"],
                         "observed_m": [0.0], **{method: [0.0] for method in METHODS}})
    rows.to_csv(tmp_path / "predictions.csv", index=False)
    pd.DataFrame({"station_id": ["a"], "date": ["2023-01-01"],
                  "head_m": [1.0], "layer": [1]}).to_csv(
        tmp_path / "observations.csv", index=False)
    (tmp_path / "report.json").write_text(json.dumps({
        "task": "newly-acquired-retrospective-head-challenge", "prospective_validation": False,
        "forecast_origin": "2022-12-01", "scored_months": 1, "stations": 1}))
    with pytest.raises(ValueError, match="disagree"):
        evaluate(tmp_path)


def _sidecar(path, policy):
    baseline = np.array([0, 10, 20, 30, 40], dtype="int16")
    data = np.zeros((2, 2, 2, 5), dtype="int16")
    data[0] = baseline
    data[1] = np.asarray(policy, dtype="int16")
    np.savez_compressed(path, subs_members_yr=data, subs_scale_m=0.001,
                        years=np.arange(2024, 2029), scenario_names=["baseline", "cut30"],
                        member=["one", "two"], ic=[0, 0], rheology=["r1", "r1"])


def test_one_time_rebound_is_not_sustained_rate_reduction(tmp_path):
    path = tmp_path / "yearly.npz"
    _sidecar(path, [0, 10, 15, 25, 35])
    evidence = evaluate_sidecar(path, policy_start_year=2026, final_year=2028,
                                late_start_year=2027)["scenarios"]["cut30"]
    assert evidence["year_end"]["2026"]["avoided_cumulative_cm"]["mean"] == 0.5
    assert evidence["year_end"]["2028"]["avoided_cumulative_cm"]["mean"] == 0.5
    assert evidence["late_rate_reduction_cm_per_year"]["mean"] == 0


def test_scenario_comparison_rejects_prepolicy_difference(tmp_path):
    path = tmp_path / "yearly.npz"
    _sidecar(path, [1, 10, 15, 25, 35])
    with pytest.raises(ValueError, match="before"):
        evaluate_sidecar(path, policy_start_year=2026, final_year=2028,
                         late_start_year=2027)
