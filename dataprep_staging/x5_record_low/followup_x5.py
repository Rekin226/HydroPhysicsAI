"""X5 follow-up: post-failure, exploratory; never validation; cannot promote.

The one follow-up the retry rule allows after X5.record_low_tf.v1 failed (0 of 14 sites).
Question: could 2020-21 discriminate the preconsolidation model from the head-level model at
all? Per band (primary input, the same fits as the scorer, nothing refitted differently):

* ``sep_mm``: RMS over the 2020-21 ring months of (TF - linear-head) predicted change since
  the anchor survey; if it is small next to the error both models make, the window cannot
  tell them apart;
* ``D_test_new_m``: depth of new lows gained in 2020-21 (monthly C20 heads);
* the TF advantage ``rmse_lin - rmse_tf`` against ``D_test_new_m`` (Spearman).

Writes results/dataprep/x5_record_low/followup/ once.

Usage: python followup_x5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import x5const as K  # noqa: E402
import x5lib as L  # noqa: E402

OUT = K.OUT / "followup"
LABEL = "post-failure, exploratory; never validation; cannot promote"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "followup.json"
    if path.exists():
        raise SystemExit(f"{path} exists; the follow-up runs once")
    recs, months = L.build_inputs()
    rows = []
    for r in recs:
        a = L.analyse_band(r["h_primary"], r["obs"], months)
        if not a["scorable"]:
            continue
        ia = int(np.flatnonzero(months == pd.Period(a["anchor"], "M"))[0])
        m = L.in_window(months, K.TEST) & np.isfinite(r["obs"])
        d = (a["tf"]["pred"][m] - a["tf"]["pred"][ia]) - (a["lin"]["pred"][m]
                                                         - a["lin"]["pred"][ia])
        rows.append({"site": r["site"], "band": r["band"], "sep_mm": L.rmse(d) * 1000,
                     "rmse_tf_mm": L.rmse(a["errors"]["tf"]) * 1000,
                     "rmse_lin_mm": L.rmse(a["errors"]["lin"]) * 1000,
                     "D_test_new_m": a["D_test_new_m"], "D_fit_m": a["D_fit_m"]})
    df = pd.DataFrame(rows)
    df["tf_gain_mm"] = df.rmse_lin_mm - df.rmse_tf_mm
    df["sep_over_err"] = df.sep_mm / df[["rmse_tf_mm", "rmse_lin_mm"]].min(axis=1)
    df.to_csv(OUT / "followup_bands.csv", index=False)
    rho, p = spearmanr(df.D_test_new_m, df.tf_gain_mm)
    big = df.D_test_new_m >= 0.5
    res = {"label": LABEL, "declaration": K.PREREG_ID, "n_bands": int(len(df)),
           "median_sep_mm": float(df.sep_mm.median()),
           "median_min_rmse_mm": float(df[["rmse_tf_mm", "rmse_lin_mm"]].min(axis=1).median()),
           "n_sep_below_half_err": int((df.sep_over_err < 0.5).sum()),
           "n_D_test_new_ge_0.5m": int(big.sum()),
           "n_D_test_new_zero": int((df.D_test_new_m <= 1e-9).sum()),
           "tf_beats_lin_by_20pct": {
               "D_test_new_ge_0.5m": int((df[big].rmse_tf_mm <= 0.8 * df[big].rmse_lin_mm).sum()),
               "D_test_new_lt_0.5m": int((df[~big].rmse_tf_mm <= 0.8 * df[~big].rmse_lin_mm)
                                         .sum())},
           "spearman_D_test_new_vs_tf_gain": {"rho": float(rho), "p": float(p)},
           "file": str(OUT / "followup_bands.csv")}
    print(json.dumps(res, indent=1, ensure_ascii=False))
    path.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
