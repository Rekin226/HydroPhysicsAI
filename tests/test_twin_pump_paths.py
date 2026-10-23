"""A calibration's non-default pump census is recorded in its meta and read back by every
consumer of ``input_options`` (2026-10-14: the Changhua + Yunlin census)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from hydrophysics.twin.calibrate_flow import DEFAULT_PATHS, _pump_paths_record  # noqa: E402
from hydrophysics.twin.inputs import input_options, resolve_paths  # noqa: E402


def test_default_census_records_nothing():
    args = SimpleNamespace(pump_census=DEFAULT_PATHS["pump_census"],
                           pump_kwh=DEFAULT_PATHS["pump_kwh"])
    assert _pump_paths_record(args) == {}
    assert "pump_census" not in input_options({"ground_elev": "wells"})


def test_non_default_census_round_trips_through_the_meta():
    args = SimpleNamespace(pump_census="census.parquet", pump_kwh=DEFAULT_PATHS["pump_kwh"])
    meta = _pump_paths_record(args)
    assert meta == {"pump_census": "census.parquet", "pump_kwh": DEFAULT_PATHS["pump_kwh"]}
    opts = input_options(meta)
    assert opts["pump_census"] == "census.parquet"
    assert opts["pump_kwh"] == DEFAULT_PATHS["pump_kwh"]


def test_resolve_paths_precedence():
    assert resolve_paths() == DEFAULT_PATHS
    recorded = resolve_paths(None, "c.parquet", "k.parquet")
    assert (recorded["pump_census"], recorded["pump_kwh"]) == ("c.parquet", "k.parquet")
    # an explicit path (a CLI flag) wins over the recorded one; None never overrides
    explicit = resolve_paths({"pump_census": "cli.parquet", "pump_kwh": None},
                             "c.parquet", "k.parquet")
    assert explicit["pump_census"] == "cli.parquet"
    assert explicit["pump_kwh"] == "k.parquet"
    assert explicit["polygon"] == DEFAULT_PATHS["polygon"]


def test_non_default_head_inputs_round_trip_through_the_meta():
    """2026-10-21: a cleaned wells directory and a repaired station table are recorded and
    read back like the census, so every consumer rebuilds the same heads."""
    args = SimpleNamespace(pump_census=DEFAULT_PATHS["pump_census"],
                           pump_kwh=DEFAULT_PATHS["pump_kwh"],
                           wells_dir="clean/wells", stations=DEFAULT_PATHS["stations"])
    meta = _pump_paths_record(args)
    assert meta == {"wells_dir": "clean/wells"}
    opts = input_options(meta)
    assert opts["wells_dir"] == "clean/wells" and "stations" not in opts
    P = resolve_paths(None, None, None, opts["wells_dir"], None)
    assert P["wells_dir"] == "clean/wells" and P["stations"] == DEFAULT_PATHS["stations"]
