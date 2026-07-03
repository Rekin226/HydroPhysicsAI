"""HydroPhysicsAI: GPU physics-informed neural operators for groundwater.

The thesis: classical hydrology calibrates one ODE per well. HydroPhysicsAI trains a
SINGLE physics-informed neural operator across all wells at once, conditioned on static
well attributes, on the NVIDIA GPU stack (CUDA / PhysicsNeMo), and benchmarks it in
true simulation mode against the per-well gray-box ODE baseline.

Two layers:
  - Foundation (NumPy/pandas, no GPU): data loading, KGE/NSE/RMSE evaluation, baselines.
    Runs anywhere, fully tested. This is the measuring stick.
  - Models (PyTorch / PhysicsNeMo, GPU): the new method. See hydrophysics.models.
"""

from .baselines import (
    climatology_prediction,
    last_value_prediction,
    load_graybox_baseline,
    persistence_prediction,
)
from .config import Config, default_config
from .data import GWData, load_dataset, load_dataset_from_frames
from .eval import benchmark_table, evaluate_predictions, spatial_scores
from .metrics import all_metrics, kge, kge_components, nse, rmse

__all__ = [
    "Config",
    "GWData",
    "all_metrics",
    "benchmark_table",
    "climatology_prediction",
    "default_config",
    "evaluate_predictions",
    "kge",
    "kge_components",
    "last_value_prediction",
    "load_dataset",
    "load_dataset_from_frames",
    "load_graybox_baseline",
    "nse",
    "persistence_prediction",
    "rmse",
    "spatial_scores",
]

__version__ = "0.0.1"
