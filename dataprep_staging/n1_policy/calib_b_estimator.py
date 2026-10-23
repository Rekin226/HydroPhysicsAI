"""Report-only calibration of step B's estimator in the model's world (no observation read).

Step B (N1.policy_response.v1) estimated beta_obs from three survey campaigns with benchmark
and campaign fixed effects. On each model's 2013-21 hindcast the same regression gave far
less than the model's true pulse derivative (Yunlin 0.14 vs 0.26, Changhua ~0 vs 0.80 cm/yr
per GWh/yr). This runs the identical estimator on every window of three consecutive May-to-May
campaigns of each model's base hindcast, at the scored benchmarks' cells, to give the spread
of what the method recovers when the truth is known. Reads model outputs and the scored
intervals' sid/cell/county columns only. Never changes a registered verdict.
Usage: results/twin_runs/par24/py.sh dataprep_staging/n1_policy/calib_b_estimator.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import REPO, REPORTS_DIR  # noqa: E402

from n1_policy import policy_lib as PL  # noqa: E402
from n1_policy import run_step_b as B  # noqa: E402

OUT = REPORTS_DIR / "N1_policy" / "step_b" / "estimator_calibration.json"


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists")
    os.chdir(REPO)
    iv = pd.read_csv(REPORTS_DIR / "N1_policy/step_b/intervals.csv",
                     usecols=["sid", "cell", "changhua"]).drop_duplicates("sid")
    scored = json.loads((REPORTS_DIR / "N1_policy/step_b/results.json").read_text())["tests"]
    res = {}
    for county, flag in (("yunlin", False), ("changhua", True)):
        theta, col = B.MODELS[county]
        tm = PL.TwinModel(str(REPO / theta), str(REPO / col))
        cells = iv[iv.changhua == flag].cell.to_numpy()
        E = tm.energy()
        E = E.sum(axis=0) if E.ndim == 3 else E
        S = tm.subsidence(tm.heads())
        may = np.array([k for k, p in enumerate(PL.MONTHS) if p.month == 5])
        W3 = PL.ring_weights(tm.cent, cells, 0.0, B.R3)
        W10 = PL.ring_weights(tm.cent, cells, B.R3, B.R10)
        r, e3, e10 = [], [], []
        for a, b in zip(may[:-1], may[1:], strict=True):
            r.append(S[cells, b] - S[cells, a])
            tot = E[:, a:b].sum(axis=1)
            e3.append(W3 @ tot)
            e10.append(W10 @ tot)
        r, e3, e10 = (np.stack(x, 1) for x in (r, e3, e10))          # (n, 9) 2013..2021
        years = [PL.MONTHS[b].year for b in may[1:]]
        n = len(cells)
        wins = []
        for s in range(len(years) - 2):
            sl = slice(s, s + 3)
            sid = np.repeat(np.arange(n)[:, None], 3, 1).ravel()
            yr = np.repeat(np.arange(3)[None, :], n, 0).ravel()
            beta = PL.fe_ols(r[:, sl].ravel(), np.column_stack([e3[:, sl].ravel(),
                                                               e10[:, sl].ravel()]), [sid, yr])
            sd = float(np.std(e3[:, sl] - e3[:, sl].mean(1, keepdims=True)) / 1e6)
            wins.append({"campaigns": years[s:s + 3], "beta_E3_cm_per_GWh": float(beta[0] * 1e8),
                         "within_sd_E3_GWh": sd})
        truth = scored[B.TESTS[county]]["model"]["beta_E3"] * 1e8
        obs = scored[B.TESTS[county]]["observed"]["beta_E3"] * 1e8
        b = np.array([w["beta_E3_cm_per_GWh"] for w in wins])
        res[county] = {"model_true_pulse_beta": truth, "observed_beta": obs, "windows": wins,
                       "recovered_median": float(np.median(b)),
                       "recovered_range": [float(b.min()), float(b.max())],
                       "recovery_ratio_median": float(np.median(b) / truth)}
        print(county, json.dumps(res[county], indent=1))
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
