"""S1a r3: after-the-fact analysis of the scored re-test (look 2026-10-17T21:29:21Z).

Reads the re-test rows only (``repair_r3/retest/fresh_rows.parquet``,
``spent_second_look_rows.parquet``) and the registered rows of S1a.e_synthetic_mixed.v1, and
writes ``repair_r3/retest/after_the_fact.json``: per-window errors of both methods on the
fresh draws, the scale (oracle) and membership parts, the b_deep error of the pre-2012
surveys, the reference-fallback shares and the paired comparison on the spent windows. Not a
test; no rule or threshold changes.

    python -m s1_rings.r3_post        (from dataprep_staging/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

RT = K.EVIDENCE / "repair_r3" / "retest"
THR = 100 * K.WORST_WINDOW_RMSE_M


def rms(v) -> float:
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def split(x: pd.DataFrame) -> dict:
    tA, tB = x[x.true_group == "A"], x[x.true_group == "B"]
    memb = x.err_cm - x.err_oracle_cm
    return {
        "n": int(len(x)), "rmse_cm": rms(x.err_cm), "bias_cm": float(x.err_cm.mean()),
        "oracle_rmse_cm": rms(x.err_oracle_cm), "oracle_bias_cm": float(x.err_oracle_cm.mean()),
        "membership_rms_cm": rms(memb), "membership_bias_cm": float(memb.mean()),
        "unassigned_share": float((x.assigned == "").mean()),
        "fallback_share_A": float(x.fallback_A.mean()), "fallback_share_B": float(x.fallback_B.mean()),
        "oracle_trueA_fallback_cm": rms(tA.loc[tA.fallback_A, "err_oracle_cm"]),
        "oracle_trueA_pooled_cm": rms(tA.loc[~tA.fallback_A, "err_oracle_cm"]),
        "n_trueA_fallback": int(tA.fallback_A.sum()), "n_trueA_pooled": int((~tA.fallback_A).sum()),
        "oracle_trueB_fallback_cm": rms(tB.loc[tB.fallback_B, "err_oracle_cm"]),
        "oracle_trueB_pooled_cm": rms(tB.loc[~tB.fallback_B, "err_oracle_cm"]),
        "mean_pool_n_A": float(x.pool_A_n.mean()), "mean_pool_n_B": float(x.pool_B_n.mean())}


def per_window(x: pd.DataFrame) -> list[dict]:
    rows = []
    for (st, m), g in x.groupby(["start", "method"]):
        rows.append({"start": st, "method": m, "n_sites": int(g.site.nunique()),
                     "rmse_cm": round(rms(g.err_cm), 3), "bias_cm": round(float(g.err_cm.mean()), 3),
                     "oracle_rmse_cm": round(rms(g.err_oracle_cm), 3),
                     "fallback_share_A": round(float(g.fallback_A.mean()), 3)})
    return rows


def main() -> None:
    R = pd.read_parquet(RT / "fresh_rows.parquet")
    S = pd.read_parquet(RT / "spent_second_look_rows.parquet")
    REG = pd.read_parquet(K.EVIDENCE / "synthetic" / "synthetic_rows.parquet")
    out = {"status": "after the fact (look 2026-10-17T21:29:21Z, after scoring); not a test",
           "families": {}}
    for fam in ("pre2012", "late2019"):
        for var in ("primary", "stress"):
            for m in ("r3", "registered"):
                x = R[(R.family == fam) & (R.variant == var) & (R.method == m)]
                out["families"][f"{fam}.{m}.{var}"] = split(x)
    x = R[(R.family == "pre2012") & (R.variant == "primary")]
    out["pre2012_primary_per_window"] = per_window(x)
    y = x[x.method == "r3"].copy()
    y["bd_err"] = y.b_deep_ppm - y.b_true_ppm
    out["pre2012_bdeep_error_by_site"] = (
        y.groupby("site").agg(rows=("bd_err", "size"),
                              finite=("bd_err", lambda v: int(np.isfinite(v).sum())),
                              mean_ppm=("bd_err", "mean"), sd_ppm=("bd_err", "std"),
                              deep_len_median_m=("deep_len_m", "median"))
        .round(1).reset_index().to_dict(orient="records"))
    out["pre2012_bdeep_finite_share"] = float(np.isfinite(y.bd_err).mean())
    out["pre2012_bdeep_error_window_mean_ppm"] = {k: round(float(v), 1) for k, v in
                                                  y.groupby("start").bd_err.mean().items()}
    # paired comparison on the spent windows (same draws): r3 second look vs registered rows
    key = ["variant", "window", "k", "site"]
    reg = REG[key + ["err_cm", "err_oracle_cm", "start"]]
    s3 = S[key + ["err_cm", "err_oracle_cm"]]
    mm = reg.merge(s3, on=key, suffixes=("_reg", "_r3"))
    pw = []
    for (var, st), g in mm.groupby(["variant", "start"]):
        pw.append({"variant": var, "start": st, "rmse_registered_cm": rms(g.err_cm_reg),
                   "rmse_r3_cm": rms(g.err_cm_r3),
                   "oracle_registered_cm": rms(g.err_oracle_cm_reg),
                   "oracle_r3_cm": rms(g.err_oracle_cm_r3)})
    P = pd.DataFrame(pw)
    comp = {}
    for var, g in P.groupby("variant"):
        comp[var] = {"n_windows": int(len(g)), "n_matched_rows": int((mm.variant == var).sum()),
                     "worst_registered_cm": float(g.rmse_registered_cm.max()),
                     "worst_r3_cm": float(g.rmse_r3_cm.max()),
                     "n_over_registered": int((g.rmse_registered_cm > THR).sum()),
                     "n_over_r3": int((g.rmse_r3_cm > THR).sum()),
                     "median_registered_cm": float(g.rmse_registered_cm.median()),
                     "median_r3_cm": float(g.rmse_r3_cm.median()),
                     "oracle_worst_registered_cm": float(g.oracle_registered_cm.max()),
                     "oracle_worst_r3_cm": float(g.oracle_r3_cm.max()),
                     "windows_r3_better": int((g.rmse_r3_cm < g.rmse_registered_cm).sum()),
                     "worst_r3_start": str(g.loc[g.rmse_r3_cm.idxmax(), "start"])}
    out["spent_second_look_paired"] = comp
    out["spent_second_look_per_window"] = P.round(3).to_dict(orient="records")
    (RT / "after_the_fact.json").write_text(json.dumps(out, indent=1, ensure_ascii=False,
                                                       default=str))
    print(json.dumps({k: out[k] for k in ("spent_second_look_paired",
                                          "pre2012_bdeep_finite_share")}, indent=1))
    for k, v in out["families"].items():
        print(k, {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()
                  if kk in ("rmse_cm", "oracle_rmse_cm", "membership_rms_cm", "unassigned_share",
                            "fallback_share_A", "oracle_trueA_pooled_cm",
                            "oracle_trueA_fallback_cm")})


if __name__ == "__main__":
    main()
