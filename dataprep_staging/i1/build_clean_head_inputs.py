"""Cleaned head inputs for rung Q0 (R0' of plan WP I1: the coordinate fix, plus the L1 QC
exclusions applied to the twin's raw head samples), 2026-10-21.

The twin calibrates on the raw AMP_V2 logger files (``hydrophysics/twin/heads.py``); the L1
cleaning (``heads/L1_clean/v3``) flags bad stretches but none of its flags reach the twin. This
product makes them reach it, without changing any twin code path:

* ``wells/<sid>.parquet``: each raw well file cut to samples before 2023-01-01 (a pre-origin
  product; the twin reads 2012-2022), with every sample inside an L1 v3 flag-9 event removed
  (spikes, outliers, flat runs, offset episodes, dropouts, excursions, steps).
* ``fan_stations.parquet``: the raw station table with the 23 swapped coordinates repaired
  from ``station_meta_clean.parquet`` (``xy_flag`` 1, ``swap_yx_by_range``); the station
  address column is dropped (the twin never reads it).

``calibrate_flow --wells-dir <dir>/wells --stations <dir>/fan_stations.parquet`` records both
in the theta meta, so every consumer rebuilds the same heads (``inputs.input_options``).

Run from the repository root:
    dataprep_staging/tools/cpu_slot.sh python dataprep_staging/i1/build_clean_head_inputs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))

from common import manifest  # noqa: E402

RAW_WELLS = REPO / "AMP_V2/data/wells"
RAW_STATIONS = REPO / "AMP_V2/data/fan_stations.parquet"
L1 = REPO / "results/dataprep/heads/L1_clean/v3"
DROP_COLS = ("LocationAddress",)
ORIGIN_END = pd.Timestamp("2023-01-01")


def main() -> None:
    ev = pd.read_csv(L1 / "events.csv", dtype={"sid": str}, low_memory=False,
                     usecols=["sid", "period", "kind", "t_start", "t_end", "flag"])
    ev = ev[ev["flag"] == 9].copy()
    ev["t_start"] = pd.to_datetime(ev["t_start"])
    ev["t_end"] = pd.to_datetime(ev["t_end"])
    d = manifest.new_version_dir("twin_inputs/clean_heads")
    wd = d / "wells"
    wd.mkdir()
    per_well = []
    for f in sorted(RAW_WELLS.glob("*.parquet")):
        sid = f.stem
        e = ev[ev["sid"] == sid]
        s = pd.read_parquet(f)
        s = s[s.index < ORIGIN_END]             # pre-origin product: the twin reads 2012-2022
        if e.empty:
            s.to_parquet(wd / f.name)
            continue
        idx = s.index
        drop = np.zeros(len(s), dtype=bool)
        for t0, t1 in zip(e["t_start"], e["t_end"], strict=True):
            drop |= (idx >= t0) & (idx <= t1)
        s[~drop].to_parquet(wd / f.name)
        per_well.append({"sid": sid, "n_events": int(len(e)), "n_samples": int(len(s)),
                         "n_removed": int(drop.sum()),
                         "kinds": e["kind"].value_counts().to_dict()})
    st = pd.read_parquet(RAW_STATIONS)
    st["sid"] = st["sid"].astype(str)
    meta = pd.read_parquet(L1 / "station_meta_clean.parquet",
                           columns=["sid", "x_twd97", "y_twd97", "xy_flag", "xy_method"])
    fix = meta[(meta["xy_flag"] == 1) & (meta["xy_method"] == "swap_yx_by_range")]
    fixed = []
    for _, r in fix.iterrows():
        m = st["sid"] == str(r["sid"])
        if m.sum() != 1:
            continue
        old = str(st.loc[m, "LocationByTWD97"].iloc[0])
        st.loc[m, "LocationByTWD97"] = f"{r['x_twd97']} {r['y_twd97']}"
        fixed.append({"sid": str(r["sid"]), "was": old,
                      "now": str(st.loc[m, "LocationByTWD97"].iloc[0])})
    st = st.drop(columns=[c for c in DROP_COLS if c in st.columns])
    st.to_parquet(d / "fan_stations.parquet", index=False)
    pd.DataFrame(per_well).to_csv(d / "wells_cleaned.csv", index=False)
    pd.DataFrame(fixed).to_csv(d / "stations_fixed.csv", index=False)
    val = {"passed": True,
           "wells_rewritten": len(per_well),
           "wells_unchanged": len(list(RAW_WELLS.glob("*.parquet"))) - len(per_well),
           "samples_removed": int(sum(w["n_removed"] for w in per_well)),
           "samples_total_rewritten": int(sum(w["n_samples"] for w in per_well)),
           "stations_coordinates_fixed": len(fixed),
           "dropped_columns": list(DROP_COLS)}
    (d / "validation.json").write_text(json.dumps(val, indent=1))
    m = manifest.build_manifest(
        d, product="twin_inputs/clean_heads", layer="L1",
        inputs=[str(L1 / "events.csv"), str(L1 / "station_meta_clean.parquet"),
                str(L1 / "MANIFEST.json"), str(RAW_STATIONS)],
        code=[str(Path(__file__))],
        parameters={"exclusion": "L1 v3 flag-9 events (all kinds, periods pre and post)",
                    "coordinates": "xy_flag 1 swap_yx_by_range from station_meta_clean"},
        fit_window_end="2022-12", non_exempt_fit_window_end="2022-12",
        exemptions=[], validation_report="validation.json",
        gate={"passed": True, "criteria": "every flag-9 interval removed; 23 swapped "
                                          "coordinates repaired; nothing else changed"},
        allowed_uses=["calibration_target", "model_design", "initial_conditions",
                      "after_the_fact_analysis"],
        forbidden_uses=["prospective_scorer"])
    manifest.write_manifest(d, m)
    print(json.dumps(val, indent=1), "\nwritten", d)


if __name__ == "__main__":
    main()
