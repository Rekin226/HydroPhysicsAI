"""Configuration and data-path resolution for the physics-ML module.

The real Zhuoshui groundwater data is NOT shipped in this repo (agency-data terms).
The data directory is resolved in this order:

1. The ``HYDROMIND_GW_DATA`` environment variable, if set.
2. ``Config.data_dir`` passed explicitly in code.
3. A bundled synthetic sample under ``physics_ml/sample_data/`` (used by tests/CI
   and by anyone without the real data). Generate it with ``sample.write_sample``.

To run on the real data, point ``HYDROMIND_GW_DATA`` at the single_tankV2 ``data/``
directory, e.g.::

    export HYDROMIND_GW_DATA=~/Desktop/Postdoc/code_space/single_tankV2/data
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Module-relative bundled sample (synthetic, safe to publish).
_SAMPLE_DIR = Path(__file__).resolve().parent / "sample_data"

# Validation split: calibration before this date, validation on/after it.
# Matches the gray-box baseline split so comparisons are apples-to-apples.
DEFAULT_SPLIT_DATE = "2019-01-01"


@dataclass
class Config:
    """Paths and modeling constants.

    data_dir: directory holding the CSV inputs (gw_timeseries.csv, rf_timeseries.csv,
        gw_stations.csv, gray_box_input.csv, intermediate/gw_coastal_inland_class.csv).
    baseline_results: path to a gray-box ``gw_fit_results.csv`` (per-well kge_val etc.).
        Optional; if absent, only the persistence baseline is available.
    split_date: validation split (ISO date string).
    """

    data_dir: Path = field(default_factory=lambda: _resolve_data_dir())
    baseline_results: Path | None = None
    split_date: str = DEFAULT_SPLIT_DATE

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.baseline_results is not None:
            self.baseline_results = Path(self.baseline_results)

    # Convenience accessors for the individual input files.
    @property
    def gw_timeseries(self) -> Path:
        return self.data_dir / "gw_timeseries.csv"

    @property
    def rf_timeseries(self) -> Path:
        return self.data_dir / "rf_timeseries.csv"

    @property
    def gw_stations(self) -> Path:
        return self.data_dir / "gw_stations.csv"

    @property
    def gray_box_input(self) -> Path:
        return self.data_dir / "gray_box_input.csv"

    @property
    def coastal_inland(self) -> Path:
        return self.data_dir / "intermediate" / "gw_coastal_inland_class.csv"


def _resolve_data_dir() -> Path:
    """Resolve the data directory from the environment or fall back to the sample."""
    env = os.environ.get("HYDROMIND_GW_DATA")
    if env:
        return Path(env).expanduser()
    return _SAMPLE_DIR


def default_config() -> Config:
    """A Config using the resolved data dir, auto-wiring a sample baseline if present."""
    cfg = Config()
    sample_baseline = cfg.data_dir / "gw_fit_results.csv"
    if cfg.baseline_results is None and sample_baseline.exists():
        cfg.baseline_results = sample_baseline
    return cfg
