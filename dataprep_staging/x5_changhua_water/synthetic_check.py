"""X5 synthetic recovery check (declared in X5.changhua_water.v1, run once before the fit).

Truth: the REAL Yunlin indicators z (standardised on the calibration years), v3's published
crop calendar, b* = (coh 0.4, riv 0.3, head 0.2), multiplicative log-normal noise sd 0.2 per
month (seed 20261025), scaled to the Jiji Yunlin annual delivery of 2012-2022. The same
pipeline then fits calendar and b on the calibration years 2015-2021 and predicts the
held-out years 2012-2014 and 2022.

Pass (all): every |b_hat - b*| <= 0.5 |b*|; held-out monthly r(pred, truth) >= 0.9; held-out
anomaly r (pred minus the calendar-only prediction vs truth minus it) >= 0.5; annual sums
reproduced to 1e-9 relative. Writes results/dataprep/x5_changhua_water/synthetic/
synthetic_check.json once (refuses to overwrite).

Usage: python -I synthetic_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x5_changhua_water import pipeline as P  # noqa: E402
from x5_changhua_water import x5const as K  # noqa: E402
from x5_changhua_water import x5lib as L  # noqa: E402


def main() -> int:
    out = K.SYN_DIR / "synthetic_check.json"
    if out.exists():
        raise SystemExit(f"{out} exists; the synthetic check runs once")
    z, _ = L.standardise(L.indicators("yl"), K.CALIB_YEARS)
    years = P.PRE_BLOCK
    idx = z.index[np.isin(z.index.year, years)]
    cal = np.array([K.SYN_CALENDAR[m] for m in range(1, 13)], dtype=float)
    lin = sum(K.SYN_B[k] * np.nan_to_num(z.loc[idx, k].to_numpy()) for k in K.INDICATORS_T3)
    rng = np.random.default_rng(K.SYN_SEED)
    w = pd.Series(cal[idx.month - 1] * np.exp(lin + rng.normal(0, K.SYN_NOISE_SD, len(idx))),
                  index=idx)
    shares = w / w.groupby(w.index.year).transform("sum")
    ann = L.jiji_annual()["yunlin_irrigation_1e8_m3"]
    truth = shares * ann.reindex(shares.index.year).to_numpy()
    calib = truth[np.isin(truth.index.year, K.CALIB_YEARS)]
    f = P.fit(calib, z, K.INDICATORS_T3)
    pred = P.blocks(f, z, ann, years)
    base = P.blocks(f, z, ann, years, b_override=P.zero_b(K.INDICATORS_T3))
    ho = np.isin(pred.index.year, K.HELDOUT_YEARS)
    r = L.pearson(pred[ho], truth.reindex(pred.index)[ho])
    ra = L.pearson((pred - base)[ho], (truth.reindex(pred.index) - base)[ho])
    sums = pred.groupby(pred.index.year).sum()
    rel = float(np.max(np.abs(sums.to_numpy() / ann.reindex(sums.index).to_numpy() - 1)))
    brel = {k: abs(f["b"][k] - K.SYN_B[k]) / abs(K.SYN_B[k]) for k in K.INDICATORS_T3}
    crit = {"b_within_tol": all(v <= K.SYN_PASS["b_rel_tol"] for v in brel.values()),
            "heldout_r": r >= K.SYN_PASS["heldout_r_min"],
            "anomaly_r": ra >= K.SYN_PASS["anomaly_r_min"],
            "annual_sums": rel <= K.SYN_PASS["annual_rel_tol"]}
    rec = {"schema": "x5-synthetic/1", "prereg": K.PREREG_ID,
           "label": "synthetic recovery check (design-time; declared criterion)",
           "b_true": K.SYN_B, "b_hat": f["b"], "b_rel_error": brel,
           "heldout_months": int(ho.sum()), "heldout_r": r, "heldout_anomaly_r": ra,
           "annual_max_rel_error": rel, "criteria": K.SYN_PASS, "each": crit,
           "passed": bool(all(crit.values())), "noise_sd": K.SYN_NOISE_SD,
           "seed": K.SYN_SEED, "calendar": K.SYN_CALENDAR}
    K.SYN_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: rec[k] for k in ("b_hat", "heldout_r", "heldout_anomaly_r",
                                          "annual_max_rel_error", "each", "passed")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
