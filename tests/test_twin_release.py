"""Release boundaries, prospective scoring, and failed-update rollback on synthetic data."""
from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pandas as pd
import pytest

from hydrophysics.twin.app.prep import pack
from hydrophysics.twin.release import check_payload, publish, read_payload, sha256
from hydrophysics.twin.update_cycle import run
from hydrophysics.twin.validation import evaluate, observations


def payload():
    return {
        "meta": {"yObs": 2022, "yTested": 2022, "vintage": "2022-12",
                 "nA": 2, "L": 4, "years": [2022, 2023]},
        "modelcard": {"public": True}, "wells": None, "leveling": None,
        "release": {"schema": 1, "status": "research-scenario", "public": True,
                    "offline": True, "data_vintage": "2022-12",
                    "sources": {"forward": {"sha256": "a" * 64}}},
        "arrays": {"subsBase": pack(np.zeros((2, 2))),
                   "headBase": pack(np.zeros((4, 2, 2)))},
    }


def page(path, data):
    path.write_text('<script id="payload" type="application/json">'
                    + json.dumps(data) + '</script>')
    return path


def test_release_snapshot_and_rejected_candidate_leave_current_unchanged(tmp_path):
    source = page(tmp_path / "viewer.html", payload())
    root = tmp_path / "releases"
    manifest = publish(source, root)
    assert sha256(root / manifest["page"]) == manifest["page_sha256"]
    current = (root / "current.json").read_bytes()
    data = read_payload(source)
    data["meta"]["yTested"] = 2025
    page(source, data)
    with pytest.raises(ValueError, match="Unobserved"):
        publish(source, root)
    assert (root / "current.json").read_bytes() == current
    assert not list(root.glob(".candidate-*"))


@pytest.mark.parametrize("change", ["private", "observations", "cdn", "shape", "nan"])
def test_release_rejects_unsafe_payload(change):
    data = payload()
    if change == "private":
        data["modelcard"]["public"] = False
    elif change == "observations":
        data["wells"] = [{"station": "synthetic"}]
    elif change == "cdn":
        data["release"]["offline"] = False
    else:
        data["arrays"]["headBase"] = pack(np.full((4, 2, 2) if change == "nan" else (2,),
                                                  np.nan if change == "nan" else 0), step=0.01)
    assert check_payload(data)


def observation_file(tmp_path):
    path = tmp_path / "observations.csv"
    pd.DataFrame({"station_id": ["s1", "s2"], "date": ["2026-09-01"] * 2,
                  "head_m": [1.0, 2.0], "layer": [1, 2], "datum": ["declared"] * 2}).to_csv(
                      path, index=False)
    return path


def test_qc_checks_each_station_freshness_and_rejects_duplicates(tmp_path):
    path = observation_file(tmp_path)
    assert observations(path, as_of="2026-09-30")["fresh"]
    frame = pd.read_csv(path)
    frame.loc[0, "date"] = "2022-12-01"
    frame.to_csv(path, index=False)
    assert observations(path, as_of="2026-09-30")["stale_stations"] == 1
    pd.concat([frame, frame]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        observations(path)


def test_qc_rejects_mixed_datums_and_future_dates(tmp_path):
    path = observation_file(tmp_path)
    with pytest.raises(ValueError, match="future"):
        observations(path, as_of="2025-01-01")
    frame = pd.read_csv(path)
    frame.loc[0, "datum"] = "different"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="one declared"):
        observations(path)


def evaluation_files(tmp_path):
    protocol = {"test_start": "2023-01-01", "test_end": "2023-02-28",
                "registered_before": "2022-12-31", "model_sha256": "a" * 64,
                "baseline": "training-only climatology and trend", "max_rmse_ratio": 0.95,
                "min_stations": 2, "min_months": 2, "independent_holdout": True}
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(protocol))
    frame = pd.DataFrame({"station_id": ["a", "a", "b", "b"],
                          "date": ["2023-01-01", "2023-02-01"] * 2,
                          "prediction_origin": ["2022-12-31"] * 4,
                          "model_sha256": ["a" * 64] * 4,
                          "observed_m": [0, 1, 0, 1], "predicted_m": [0.1, 1.1, 0.1, 1.1],
                          "baseline_m": [0.5, 1.5, 0.5, 1.5]})
    predictions = tmp_path / "predictions.csv"
    frame.to_csv(predictions, index=False)
    return predictions, config, frame, protocol


def test_prospective_score_and_reused_holdout_gate(tmp_path):
    predictions, config, _, protocol = evaluation_files(tmp_path)
    result = evaluate(predictions, config)
    assert result["passed"] and result["rmse_ratio"] == pytest.approx(0.2)
    assert not result["causal_policy_validation"]
    protocol["independent_holdout"] = False
    config.write_text(json.dumps(protocol))
    assert not evaluate(predictions, config)["passed"]


@pytest.mark.parametrize("change", ["leakage", "wrong_model", "nonfinite", "duplicate"])
def test_evaluation_rejects_invalid_predictions(tmp_path, change):
    predictions, config, frame, _ = evaluation_files(tmp_path)
    if change == "leakage":
        frame.loc[1, "prediction_origin"] = "2023-01-02"
    elif change == "wrong_model":
        frame.loc[0, "model_sha256"] = "b" * 64
    elif change == "nonfinite":
        frame.loc[0, "predicted_m"] = np.inf
    else:
        frame = pd.concat([frame, frame.iloc[:1]])
    frame.to_csv(predictions, index=False)
    with pytest.raises(ValueError):
        evaluate(predictions, config)


def test_update_retains_previous_release_on_stale_or_failed_run(tmp_path):
    source = page(tmp_path / "viewer.html", payload())
    root = tmp_path / "releases"
    publish(source, root)
    before = (root / "current.json").read_bytes()
    obs = observation_file(tmp_path)
    config = {"release_root": str(root), "observations": str(obs), "page": str(source),
              "as_of": "2026-09-30", "steps": [[sys.executable, "-c", "raise SystemExit(2)"]]}
    status = tmp_path / "status.json"
    assert run(config, status)["state"] == "failed"
    config["as_of"] = "2027-09-30"
    result = run(config, status)
    assert "stale" in result["error"]
    assert (root / "current.json").read_bytes() == before


def test_update_requires_matching_observation_lineage(tmp_path):
    obs = observation_file(tmp_path)
    data = payload()
    data["meta"].update(vintage="2026-09", yObs=2026, yTested=2026)
    data["release"]["data_vintage"] = "2026-09"
    source = page(tmp_path / "viewer.html", data)
    config = {"cwd": str(tmp_path), "release_root": "releases", "observations": obs.name,
              "page": source.name, "as_of": "2026-09-30",
              "steps": [[sys.executable, "-c", "pass"]]}
    status = tmp_path / "status.json"
    assert run(config, status)["state"] == "failed"
    (tmp_path / "acceptance.json").write_text(json.dumps({"passed": True}))
    config["acceptance_reports"] = ["acceptance.json"]
    assert run(config, status)["state"] == "failed"
    valid = copy.deepcopy(data)
    valid["release"]["sources"]["observations"] = {"sha256": sha256(obs)}
    page(source, valid)
    assert run(config, status)["state"] == "published"
    assert (tmp_path / "releases" / "current.json").exists()
    current = (tmp_path / "releases" / "current.json").read_bytes()
    (tmp_path / "acceptance.json").write_text(json.dumps({"passed": False}))
    assert run(config, status)["state"] == "failed"
    assert (tmp_path / "releases" / "current.json").read_bytes() == current
