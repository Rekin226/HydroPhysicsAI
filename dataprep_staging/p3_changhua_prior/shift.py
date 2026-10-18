"""WP P3 step 6 shift diagnostics, ported from design_ml_transfer/a1_domain_shift.py (feature
shift, adversarial AUC with 5 km block CV, Kish ESS of clipped density-ratio weights) on the
rebuilt frame, plus the crop-mix shift (township upland share, MOA 2014-16 as the pilot).
Features only: no census value of Changhua is read. Canal water per farm hectare uses the
twin's surface_water_v3 (2012-2022), as the pilot did.

Usage: python -I shift.py        (writes P3_evidence/shift.json)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frame  # noqa: E402
import lib  # noqa: E402
import models_port as M  # noqa: E402

SW = lib.REPO / "results/twin/surface_water_v3.npz"


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    df, aux = frame.load("prior")
    df = M.add_crop(df, aux["tz"])
    sw = np.load(SW, allow_pickle=True)
    df["sw_m_yr"] = sw["sw_m_per_day"].mean(1) * 365.25
    farm = df[["paddy", "dry", "orchard"]].sum(1)
    df["sw_per_farm"] = np.where(farm > 5, df["sw_m_yr"] * 100 / np.maximum(farm, 1e-9), np.nan)
    tr = df[frame.training_cells(df)].copy()
    ch = df[df.moi_chg > 0.5].copy()
    res = {"n_yunlin_train": int(len(tr)), "n_changhua": int(len(ch))}
    fs = {}
    for f in M.FEATS + M.COV + ["sw_m_yr", "sw_per_farm", "c_upland"]:
        a, b = tr[f].to_numpy(float), ch[f].to_numpy(float)
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        lo, hi = np.percentile(a, [1, 99])
        fs[f] = {"yun_median": float(np.median(a)), "chg_median": float(np.median(b)),
                 "chg_frac_outside_yun_p1_p99": float(np.mean((b < lo) | (b > hi)))}
    res["feature_shift"] = fs
    both = pd.concat([tr.assign(lab=0), ch.assign(lab=1)])
    grp = both["block5"].to_numpy()
    sets = {"A_landuse_geo": M.FEATS + M.COV,
            "A_minus_x": M.FEATS + [c for c in M.COV if c != "x"],
            "B_plus_canal": M.FEATS + M.COV + ["sw_m_yr"]}
    for nm, cols in sets.items():
        X = both[cols].copy()
        for c in M.FEATS:
            X[c] = X[c] / 100.0
        X = X.to_numpy(float)
        y = both["lab"].to_numpy()
        p = np.zeros(len(y))
        for a, b in GroupKFold(n_splits=5).split(X, y, grp):
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                               max_leaf_nodes=15, min_samples_leaf=20,
                                               l2_regularization=1.0,
                                               random_state=0).fit(X[a], y[a])
            p[b] = m.predict_proba(X[b])[:, 1]
        py, pc = p[y == 0], p[y == 1]
        w = (py / (1 - py + 1e-6)) * (len(py) / len(pc))
        w = np.clip(w, 0, np.percentile(w, 99))
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        res[f"adv_{nm}"] = {"auc_block_cv": float(roc_auc_score(y, p)),
                            "chg_frac_p_gt_0.9": float(np.mean(pc > 0.9)),
                            "kish_ess_fraction": ess / len(py)}
    # upland-crop share of planted area, township medians (MOA 2014-16, as the pilot)
    res["upland_share_cell_median"] = {"yunlin_train": fs["c_upland"]["yun_median"],
                                       "changhua": fs["c_upland"]["chg_median"]}
    a1 = lib.PILOT_ML / "a1_results.json"
    import json
    with open(a1, encoding="utf-8") as f:
        pil = json.load(f)
    res["pilot_a1"] = {"auc_A": pil["adv_A_landuse_geo"]["auc_block_cv"],
                       "ess_frac_A": pil["adv_A_landuse_geo"]["ess_frac"],
                       "sw_per_farm_outside": pil["feature_shift"]["sw_per_farm"]
                       ["chg_frac_outside_yun_p1_p99"]}
    lib.save_json(lib.EVID / "shift.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "feature_shift"}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
