"""Generate a small synthetic dataset in the single_tankV2 CSV layout.

Used by tests/CI and by anyone without the real (unpublished) Zhuoshui data. The
synthetic series follow the same gray-box-style dynamics (recession + rainfall +
seasonal + upstream coupling) so the loader, baselines, and eval exercise realistic
shapes and value ranges. It is NOT real data and carries no agency-data restrictions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_sample(out_dir, n_wells: int = 4, seed: int = 0,
                 start: str = "2017-01-01", end: str = "2019-12-31") -> Path:
    """Write a synthetic dataset to ``out_dir`` and return the path."""
    out_dir = Path(out_dir)
    (out_dir / "intermediate").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start, end, freq="D")
    T = len(dates)
    doy = dates.dayofyear.to_numpy()

    well_ids = [f"st{i + 1}" for i in range(n_wells)]
    rf_ids = [f"rf{i + 1}" for i in range(n_wells)]

    # Rainfall: sparse positive daily totals.
    rain = (rng.random((n_wells, T)) < 0.2) * rng.gamma(2.0, 5.0, (n_wells, T))

    # Groundwater via a simple recession + rainfall + seasonal model, integrated daily.
    h = np.zeros((n_wells, T))
    season = 0.8 * np.sin(2 * np.pi * doy / 365.25)
    for i in range(n_wells):
        level = rng.normal(10, 3)
        z = rng.normal(8, 2)
        for t in range(T):
            level += -0.05 * (level - z) + 0.02 * rain[i, t] + 0.1 * season[t]
            level += rng.normal(0, 0.05)
            h[i, t] = level

    # Upstream coupling: each well's upstream is the previous well (ring).
    ups_ids = [well_ids[(i - 1) % n_wells] for i in range(n_wells)]

    # Coordinates on an arbitrary local grid.
    tm_x = 190000 + rng.uniform(0, 20000, n_wells)
    tm_y = 2630000 + rng.uniform(0, 30000, n_wells)
    groups = ["coastal" if i == 0 else "inland" for i in range(n_wells)]

    # --- write gw_timeseries.csv (daily timestamps; loader resamples idempotently) ---
    gw = pd.DataFrame({"date time": dates.strftime("%Y-%m-%d %H:%M:%S")})
    for i, w in enumerate(well_ids):
        gw[w] = np.round(h[i], 3)
    gw.to_csv(out_dir / "gw_timeseries.csv", index=False)

    # --- rf_timeseries.csv ---
    rf = pd.DataFrame({"date time": dates.strftime("%Y-%m-%d")})
    for i, r in enumerate(rf_ids):
        rf[r] = np.round(rain[i], 2)
    rf.to_csv(out_dir / "rf_timeseries.csv", index=False)

    # --- gw_stations.csv ---
    pd.DataFrame({
        "Station": [7010000 + i for i in range(n_wells)],
        "NAME_C": [f"well_{i + 1}" for i in range(n_wells)],
        "TM_X97": np.round(tm_x, 3),
        "TM_Y97": np.round(tm_y, 3),
        "st_id": well_ids,
    }).to_csv(out_dir / "gw_stations.csv", index=False)

    # --- gray_box_input.csv (all active) ---
    pd.DataFrame({
        "gw_st": [7010000 + i for i in range(n_wells)],
        "st_id": well_ids,
        "gw_TM_X97": np.round(tm_x, 3),
        "gw_TM_Y97": np.round(tm_y, 3),
        "ups_id": ups_ids,
        "ups_lag_days": 1.0,
        "rf_id": rf_ids,
        "lag_days": 1,
        "group": groups,
        "active": 1,
    }).to_csv(out_dir / "gray_box_input.csv", index=False)

    # --- coastal/inland classification ---
    pd.DataFrame({
        "st_id": well_ids,
        "dist_to_coast_m": np.round(rng.uniform(1000, 15000, n_wells), 1),
        "is_near_coast": [g == "coastal" for g in groups],
        "dom_freq_cpd": 1.0,
        "dom_amp": np.round(rng.uniform(0.01, 0.1, n_wells), 4),
        "m2_amp": np.round(rng.uniform(0, 0.005, n_wells), 5),
        "is_m2_like": [g == "coastal" for g in groups],
        "group": groups,
    }).to_csv(out_dir / "intermediate" / "gw_coastal_inland_class.csv", index=False)

    # --- a plausible gray-box baseline result table ---
    pd.DataFrame({
        "st_id": well_ids,
        "model": ["base_tz"] * n_wells,
        "group_name": groups,
        "kge_val": np.round(rng.uniform(0.4, 0.85, n_wells), 3),
        "rmse_val": np.round(rng.uniform(0.3, 1.2, n_wells), 3),
        "r2_val": np.round(rng.uniform(0.4, 0.9, n_wells), 3),
    }).to_csv(out_dir / "gw_fit_results.csv", index=False)

    return out_dir


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "sample_data"
    p = write_sample(Path(__file__).resolve().parent / target if not Path(target).is_absolute() else target)
    print(f"wrote synthetic sample to {p}")
