"""Score the X5 Yunlin held-out agreement check once (X5.changhua_water.v1, X5C).

Needs fit/fit.json and a 'declared_check' look on the held-out unit logged by look_x5.py.
Primary: Pearson r over the 48 held-out months (2012-2014, 2022) between the T3 Denton
prediction and W1 pre-origin/v2 Yunlin monthly values; agreement if r >= 0.6. Reported beside
it (declared; never substituted): T2 Denton, T3 Chow-Lin, the calendar-only baseline, r
without the structural-zero months, the anomaly r against the baseline, per-year shape r and
RMSE. Exploration (U7, labelled): 2023-2025 against W1 full/v4. Design-time (in-sample,
already-seen targets): Jan-May sums 2015-2021 vs the observed briefing values, crop-1 share
2012-2015 vs the MOA office crop-season shares.

Writes results/dataprep/reports/X5.changhua_water.v1/report.json once.

Usage: python -I score_x5.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg3  # noqa: E402

from x5_changhua_water import pipeline as P  # noqa: E402
from x5_changhua_water import x5const as K  # noqa: E402
from x5_changhua_water import x5lib as L  # noqa: E402
from x5_changhua_water.fit_x5 import load_w1  # noqa: E402
from x5_changhua_water.look_x5 import HELDOUT_UNIT  # noqa: E402


def fitted(rec: dict, model: str, county: str) -> dict:
    m = rec["models"][model]
    return {"calendar": np.asarray(rec["calendar"][county]), "b": m["b"], "keys": m["keys"]}


def metrics(pred: pd.Series, ref: pd.Series, base: pd.Series) -> dict:
    ref = ref.reindex(pred.index)
    nz = ref.to_numpy() != 0
    per_year = {int(y): L.pearson(pred[pred.index.year == y], ref[ref.index.year == y])
                for y in sorted(set(pred.index.year))}
    return {"r": L.pearson(pred, ref), "r_nonzero_months": L.pearson(pred[nz], ref[nz]),
            "anomaly_r_vs_baseline": L.pearson(pred - base, ref - base),
            "rmse_1e8m3": float(np.sqrt(np.mean((pred - ref) ** 2))),
            "per_year_shape_r": per_year, "n_months": int(len(pred))}


def main() -> int:
    out = K.REPORT_DIR / "report.json"
    if out.exists():
        raise SystemExit(f"{out} exists; scored once")
    looks = [lk for lk in prereg3.read_chained(prereg3.PREREG_DIR / "looks.jsonl")
             if lk.get("prereg_id") == K.PREREG_ID and lk.get("test_id") == K.CHECK_ID
             and lk.get("purpose") == "declared_check" and HELDOUT_UNIT in lk.get("units", [])]
    if not looks:
        raise SystemExit("no declared_check look on the held-out unit; run look_x5.py check")
    doc, sha = prereg3.load_prereg(K.PREREG_ID)
    fit = json.loads((K.FIT_DIR / "fit.json").read_text(encoding="utf-8"))
    z = {d: L.standardise(L.indicators(d), K.CALIB_YEARS)[0] for d in ("yl", "chc")}
    jj = L.jiji_annual()
    ann_yl, ann_chc = jj["yunlin_irrigation_1e8_m3"], jj["changhua_irrigation_1e8_m3"]
    yrs = P.PRE_BLOCK
    t3, t2 = fitted(fit, "T3", "yl"), fitted(fit, "T2", "yl")
    pred = {"T3_denton": P.blocks(t3, z["yl"], ann_yl, yrs),
            "T2_denton": P.blocks(t2, z["yl"], ann_yl, yrs),
            "T3_chow_lin": P.monthly(t3, z["yl"], ann_yl, yrs, "chow_lin")[0],
            "calendar_only_denton": P.blocks(t3, z["yl"], ann_yl, yrs,
                                             b_override=P.zero_b(K.INDICATORS_T3))}
    _, cl_meta = P.monthly(t3, z["yl"], ann_yl, yrs, "chow_lin")
    # ---- the held-out reference is read only here, after the look
    ref = load_w1(K.W1_PRE, "yl", K.HELDOUT_YEARS)
    ho = lambda s: s[np.isin(s.index.year, K.HELDOUT_YEARS)]  # noqa: E731
    base = ho(pred["calendar_only_denton"])
    res = {k: metrics(ho(v), ref, base) for k, v in pred.items()}
    r = res["T3_denton"]["r"]
    verdict = "AGREES" if r >= K.THRESHOLD_R else "DOES NOT AGREE"
    # ---- exploration (U7): 2023-2025 against W1 full/v4
    ex_ref = load_w1(K.W1_FULL, "yl", K.EXPLORE_YEARS)
    ex = {}
    for k, f in (("T3_denton", t3), ("T2_denton", t2)):
        ex_pred = P.blocks(f, z["yl"], ann_yl, K.EXPLORE_YEARS)
        ex_base = P.blocks(f, z["yl"], ann_yl, K.EXPLORE_YEARS,
                           b_override=P.zero_b(f["keys"]))
        ex[k] = metrics(ex_pred, ex_ref, ex_base)
    # ---- design-time, in-sample: Jan-May 2015-2021 vs the observed briefing values
    jm = pd.read_csv(K.W1_PRE / "janmay_validation.csv").set_index("year")
    t3c = fitted(fit, "T3", "chc")
    jiji_pred = {"yl": pred["T3_denton"],
                 "chc": P.blocks(t3c, z["chc"], ann_chc, yrs)}
    d1 = {}
    for c in ("yl", "chc"):
        s = jiji_pred[c]
        jms = s[s.index.month <= 5].groupby(s[s.index.month <= 5].index.year).sum()
        yy = [y for y in jm.index if y in jms.index]
        o, p = jm.loc[yy, f"obs_{c}"].to_numpy(), jms.loc[yy].to_numpy()
        w = jm.loc[yy, f"supply_limited_denton_{c}"].to_numpy()
        d1[c] = {"years": [int(y) for y in yy], "rmse_x5": float(np.sqrt(np.mean((p - o) ** 2))),
                 "r_x5": L.pearson(p, o), "rmse_w1_same_file": float(np.sqrt(np.mean(
                     (w - o) ** 2))), "r_w1_same_file": L.pearson(w, o)}
    # ---- design-time: crop-1 share 2012-2015 vs MOA office crop seasons
    moa = pd.read_csv(K.MOA_CSV, encoding="utf-8-sig",
                      usecols=["處別", "年別", "灌溉用水量_億噸", "期作"])
    moa["year"] = moa["年別"].astype(int) + 1911
    d2 = {}
    for c, off in K.MOA_OFFICE.items():
        sub = moa[(moa["處別"].astype(str).str.strip() == off) & moa["year"].between(2012, 2015)]
        pv = sub.pivot_table(index="year", columns="期作", values="灌溉用水量_億噸", aggfunc="sum")
        obs = (pv["一"] / (pv["一"] + pv["二"]))
        s = jiji_pred[c]
        s = s[s.index.year <= 2015]
        prd = s[s.index.month <= 6].groupby(s[s.index.month <= 6].index.year).sum() / \
            s.groupby(s.index.year).sum()
        d2[c] = {"years": [int(y) for y in obs.index],
                 "observed_crop1_share": [float(v) for v in obs],
                 "x5_jan_jun_share": [float(prd.loc[y]) for y in obs.index],
                 "mean_abs_diff": float(np.mean(np.abs(prd.reindex(obs.index) - obs)))}
    rep = {"schema": "x5-report/1", "prereg": K.PREREG_ID, "prereg_sha256": sha,
           "check": K.CHECK_ID, "look_time": looks[-1]["time"],
           "label": "design-time agreement check against an INFERRED reference (W1 monthly, "
                    "flag 5; no monthly observation exists); never validation of timing "
                    "skill (U1)",
           "threshold_r": K.THRESHOLD_R, "primary": "T3_denton",
           "primary_r": r, "verdict": verdict,
           "heldout_years": K.HELDOUT_YEARS, "calib_years": K.CALIB_YEARS,
           "fitted_b": {k: v["b"] for k, v in fit["models"].items()},
           "chow_lin": cl_meta, "heldout": res,
           "exploration_2023_2025": {"label": "post-origin, exploration only (U7); W1 full/v4 "
                                              "2023-2025 is itself inferred; heads absent "
                                              "after 2022, so T3 = calendar+coh+riv",
                                     "metrics": ex},
           "designtime_janmay_2015_2021": {"label": "in-sample, already-seen observed "
                                                    "targets; reported, never gating",
                                           "by_district": d1},
           "designtime_crop1_share_2012_2015": {"label": "already-seen observed MOA crop "
                                                         "shares; reported, never gating",
                                                "by_office": d2},
           "attempt": 1}
    K.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    print(json.dumps({"primary_r": r, "verdict": verdict,
                      "heldout": {k: {kk: v[kk] for kk in ("r", "anomaly_r_vs_baseline")}
                                  for k, v in res.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
