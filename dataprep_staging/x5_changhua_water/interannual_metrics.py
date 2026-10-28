"""Changhua interannual amplitude ratio and correlation of a twin run's held-out predictions.

The two X5-specific gates of rung I1.X5.v1 (route 4) are read from a run's design-time
out-of-fold predictions (``stage3_per_entry.npz``) with ``interannual()`` and ``corr()`` of
``dataprep_staging/i1/diag_changhua_interannual.py`` (imported, not copied):

* amplitude A = median over Changhua held-out entries with >= 60 paired finite months of
  nanstd(interannual(pred)) / nanstd(interannual(obs));
* correlation = median over Changhua entries with >= 24 paired months of
  corr(interannual(obs), interannual(pred)).

Columns of the per-entry arrays start in 2012-02, so column t has calendar month (t + 1) % 12.

Usage: python dataprep_staging/x5_changhua_water/interannual_metrics.py q0 [suffix ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(REPO / "dataprep_staging/i1"))

import diag_changhua_interannual as D  # noqa: E402

MIN_AMP_MONTHS, MIN_CORR_MONTHS = 60, 24


def changhua_interannual(suffix: str) -> dict:
    run = REPO / f"results/twin_runs/stage3_datum_gate_{suffix}"
    st = pd.read_parquet(D.STATIONS, columns=["sid", "CountyName"]).set_index("sid")
    z = np.load(run / "stage3_per_entry.npz", allow_pickle=True)
    w = pd.read_csv(run / "stage3_kfold_wells.csv", dtype={"sid": str}).set_index("entry")
    county = st.loc[w.loc[z["entry"], "sid"].to_numpy(), "CountyName"].to_numpy()
    obs, pred = z["obs"].astype(float), z["pred"].astype(float)
    cal = (np.arange(1, obs.shape[1] + 1)) % 12
    amps, cors = [], []
    for i in np.where(county == D.COUNTIES["changhua"])[0]:
        paired = np.isfinite(obs[i]) & np.isfinite(pred[i])
        o = D.interannual(np.where(paired, obs[i], np.nan), cal)
        p = D.interannual(np.where(paired, pred[i], np.nan), cal)
        if paired.sum() >= MIN_AMP_MONTHS and np.nanstd(o) > 0:
            amps.append(float(np.nanstd(p) / np.nanstd(o)))
        if paired.sum() >= MIN_CORR_MONTHS:
            c = D.corr(o, p)
            if np.isfinite(c):
                cors.append(c)
    return {"suffix": suffix, "amplitude_A": float(np.median(amps)), "n_amp": len(amps),
            "interannual_corr": float(np.median(cors)), "n_corr": len(cors)}


if __name__ == "__main__":
    print(json.dumps([changhua_interannual(s) for s in sys.argv[1:] or ["q0"]], indent=1))
