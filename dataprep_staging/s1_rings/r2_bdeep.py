"""Repair round (r2) of WP S1a: the b_deep line's error in the registered (e) rows.

Supplement to ``r2_diagnose`` (whose output rings_bridged/v2 hashes, so it is not edited):
the error of the site's deep-interval scale ``b_deep - b_true`` per true group, per window and
in the windows that exceed 1.5 cm, and how the window means track the scale and membership
components. After the fact; changes no rule. Logs its look first.

    tools/cpu_slot.sh python -m s1_rings.r2_bdeep       (from dataprep_staging/)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

SYN = K.EVIDENCE / "synthetic"
OUT = K.EVIDENCE / "repair"
THR_CM = 100 * K.WORST_WINDOW_RMSE_M


def summarize(x: pd.DataFrame) -> dict:
    g = x.groupby("true_group")["bd_err"]
    return {tg: {"n": int(v.size), "n_finite": int(v.notna().sum()),
                 "mean_ppm": round(float(v.mean()), 1), "sd_ppm": round(float(v.std()), 1)}
            for tg, v in g}


def analyse(R: pd.DataFrame) -> dict:
    x = R[R.variant == "primary"].copy()
    x["bd_err"] = x["b_deep_ppm"] - x["b_true_ppm"]
    x["memb"] = x["err_cm"] - x["err_oracle_cm"]
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
    per = x.groupby(["window", "start"]).agg(
        bd_err_mean_ppm=("bd_err", "mean"), rmse_cm=("err_cm", rms),
        bias_scale_cm=("err_oracle_cm", "mean"), bias_membership_cm=("memb", "mean")
    ).reset_index()
    fail = per[per.rmse_cm > THR_CM]
    worst = int(per.loc[per.rmse_cm.idxmax(), "window"])
    return {
        "status": "after the fact (diagnosis of the registered S1a.e_synthetic_mixed.v1 rows); "
                  "no rule changed",
        "definition": "b_deep error = site's deep-interval scale estimated in the realization "
                      "minus the injected scale (ppm)",
        "overall": summarize(x),
        "worst_window": {"start": per.loc[per.window == worst, "start"].iloc[0],
                         **summarize(x[x.window == worst])},
        "windows_over_threshold": {"n": int(len(fail)),
                                   **summarize(x[x.window.isin(fail.window)])},
        "windows_within_threshold": summarize(x[~x.window.isin(fail.window)]),
        "corr_window_mean_bd_err_vs": {
            "bias_scale_cm": round(float(per.bd_err_mean_ppm.corr(per.bias_scale_cm)), 3),
            "bias_membership_cm": round(float(per.bd_err_mean_ppm.corr(per.bias_membership_cm)),
                                        3),
            "rmse_cm": round(float(per.bd_err_mean_ppm.corr(per.rmse_cm)), 3)},
        "per_window": per.round(3).to_dict(orient="records"),
    }


def main() -> None:
    from harness import prereg3
    look = prereg3.record_look(
        ["S1a:synth.R.mixed_groupcond.v1"], "rings+piles:2014-05..2021-12", "S1a", "diagnose",
        test_id="S1a.e.mixed_groupcond", prereg_id="S1a.e_synthetic_mixed.v1",
        note=("S1a repair round (r2): b_deep error (estimated minus injected scale) per true "
              "group and window in the registered (e) rows; a first summary of it was read "
              "interactively at about 08:41Z, before this record; no rule changed"))
    R = pd.read_parquet(SYN / "synthetic_rows.parquet")
    out = analyse(R)
    out["look"] = {k: look.get(k) for k in ("time", "purpose", "test_id")}
    (OUT / "e_bdeep_bias.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                      default=str))
    print(json.dumps({k: out[k] for k in ("overall", "worst_window", "windows_over_threshold",
                                          "windows_within_threshold",
                                          "corr_window_mean_bd_err_vs")}, indent=1))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
