"""Share of Changhua (and other) cells whose forward 2023-25 subsidence rate is below -tol,
i.e. ground rising after 2022 (the R1c six-zone artefact), from a forward run's npz, with
the same series and dates as score_rung_b.plausibility.

Usage: python dataprep_staging/x5_energy_volume/uplift.py q0 [suffix ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "dataprep_staging/i1"))

import score_rung_b as B  # noqa: E402


def uplift_share(suffix: str, tol: float = 0.0) -> dict:
    """Shares of cells with 2023-25 rate < -tol cm/yr, Changhua and elsewhere."""
    ch = np.load(B.CMAP)["changhua"].astype(bool)
    z = np.load(B.FWD / f"datum_gate_{suffix}.npz")
    d = pd.DatetimeIndex(z["dates"])
    s = (z["subs_mean_by_rheology"][0][0] if "subs_mean_by_rheology" in z.files
         else z["subs_mean"][0]) * 100
    i22 = np.where(d <= pd.Timestamp("2022-12-31"))[0][-1]
    i25 = np.where(d <= pd.Timestamp("2025-12-31"))[0][-1]
    rate = (s[:, i25] - s[:, i22]) / 3
    return {"uplift_share_ch": float(np.mean(rate[ch] < -tol)),
            "uplift_share_rest": float(np.mean(rate[~ch] < -tol))}


if __name__ == "__main__":
    print(json.dumps({s: uplift_share(s, 0.1) for s in sys.argv[1:] or ["q0"]}, indent=1))
