"""WP S5 repair round 2, fix round: the after-the-fact checks declared in S5.repair.md F.2.

    python -I s5r2f_looks.py fix_after && tools/cpu_slot.sh python -I s5r2f_diag.py

Report only. Nothing here feeds a method, hyperparameter, threshold or gate; every unit is spent.

A. Exposure of the 2018-19 surveys: T2 increments (round 1 and round 2) with a start or end
   survey in 2018-01..2019-12, their benchmarks, the share of T1F benchmarks among them; leveling
   surveys that never entered a scored T1/T2 increment.
B. T2 and T2b calibration by distance from the target benchmark to the nearest training
   benchmark (raw 90 % coverage, rms z, median |z|, share |z| > 3).
C. Distance from every twin-grid cell to the nearest leveling benchmark (the product's regime).

Output: results/dataprep/reports/S5_evidence/repair_r2/fix_round/fix_after.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s5const as C  # noqa: E402
import s5data as D  # noqa: E402
import s5model as M  # noqa: E402
import s5r2_const as K  # noqa: E402
import s5tests as Ts  # noqa: E402

OUT = K.EVID / "fix_round"
W0, W1 = 2018.0, 2020.0          # 2018-01-01 .. 2019-12-31 in decimal years
BINS_T2 = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, np.inf)]
BINS_T2B = [(10.0, 12.0), (12.0, 14.0), (14.0, np.inf)]
GAUSS = {"median_abs_z": 0.6745, "share_abs_z_gt3": 0.0027, "share_abs_z_gt_z90": 0.10}


def in_window(t) -> np.ndarray:
    t = np.asarray(t, float)
    return (t >= W0) & (t < W1)


def exposure(lev: pd.DataFrame) -> dict:
    t1f = pd.read_parquet(K.EVID / "t1f_targets.parquet", columns=["sid", "t_new", "vr"])
    t1f_all = set(t1f.sid)
    t1f_sc = set(t1f.loc[t1f.vr >= C.VR_MIN, "sid"])
    out = {"t1f_benchmarks_all": len(t1f_all), "t1f_benchmarks_scored": len(t1f_sc),
           "t1f_targets_in_window": int(in_window(t1f.t_new).sum()), "n_t1f": int(len(t1f))}
    used = set()
    for name, p in (("t2_round1", C.EVIDENCE / "t2_targets.parquet"),
                    ("t2_round2", K.EVID / "t2_targets.parquet"),
                    ("t2b_round2", K.EVID / "t2b_targets.parquet")):
        T = pd.read_parquet(p, columns=["sid", "t0", "t1"])
        hit = in_window(T.t0) | in_window(T.t1)
        sids = set(T.loc[hit, "sid"])
        out[name] = {"increments": int(len(T)), "increments_with_2018_19_survey": int(hit.sum()),
                     "benchmarks_with_2018_19_increment": len(sids),
                     "t1f_scored_benchmarks_among_them": len(t1f_sc & sids),
                     "t1f_all_benchmarks_among_them": len(t1f_all & sids)}
        used |= {(s, round(float(t), 6)) for s, t in zip(T.sid, T.t0, strict=True)}
        used |= {(s, round(float(t), 6)) for s, t in zip(T.sid, T.t1, strict=True)}
    for _name, p in (("t1_round1", C.EVIDENCE / "t1_targets.parquet"),
                    ("t1_round2", K.EVID / "t1_targets.parquet")):
        T = pd.read_parquet(p, columns=["sid", "t_new"])
        used |= {(s, round(float(t), 6)) for s, t in zip(T.sid, T.t_new, strict=True)}
    keys = [(s, round(float(t), 6)) for s, t in zip(lev.sid, lev.t, strict=True)]
    never = np.array([k not in used for k in keys])
    lw = lev[never]
    out["leveling_surveys"] = int(len(lev))
    out["surveys_never_in_scored_t1_t2"] = int(never.sum())
    out["surveys_never_scored_by_year"] = {int(y): int(n) for y, n in
                                           lw.groupby(np.floor(lw.t).astype(int)).size().items()}
    out["benchmarks_with_never_scored_survey"] = int(lw.sid.nunique())
    return out


def zstats(T: pd.DataFrame) -> dict:
    z = (T.err_fused / T.sd_fused).to_numpy()
    a = np.abs(z)
    return {"n": int(len(T)), "benchmarks": int(T.sid.nunique()),
            "raw_coverage90": float(T.hit_fused.mean()),
            "rms_z": float(np.sqrt(np.mean(z ** 2))), "median_abs_z": float(np.median(a)),
            "share_abs_z_gt3": float(np.mean(a > 3.0)),
            "rmse_mm": float(np.sqrt(np.mean(T.err_fused ** 2))),
            "median_sd_mm": float(T.sd_fused.median()),
            "median_dist_km": float(T.dist_km.median()) if "dist_km" in T else None}


def by_distance(T: pd.DataFrame, bins) -> list[dict]:
    rows = []
    for lo, hi in bins:
        s = T[(T.dist_km >= lo) & (T.dist_km < hi)]
        r = {"bin_km": [lo, None if not np.isfinite(hi) else hi]}
        r.update(zstats(s) if len(s) else {"n": 0})
        rows.append(r)
    return rows


def calibration(lev: pd.DataFrame) -> dict:
    bxy = lev.groupby("sid")[["x_km", "y_km"]].first().sort_index()
    cl = Ts.benchmark_clusters(lev)
    out = {}
    for key, buffer in (("t2", None), ("t2b", K.T2B_BUFFER_KM)):
        T = pd.read_parquet(K.EVID / f"{key}_targets.parquet")
        # block membership as used by the test (file labels) vs recomputed on the QC'd table
        agree = []
        dist = np.full(len(T), np.nan)
        for _b, g in T.groupby("cluster"):
            sids = sorted(set(g.sid))
            lab = {cl.get(s) for s in sids}
            block = [s for s, c in cl.items() if c in lab]
            agree.append(set(block) == set(sids) or set(sids) <= set(block))
            if buffer is None:
                train = bxy.index[~bxy.index.isin(block)]
            else:
                dmin = M.dist(bxy.to_numpy(), bxy.loc[block].to_numpy()).min(1)
                train = bxy.index[dmin > buffer]
            txy = bxy.loc[train].to_numpy()
            gx = bxy.loc[g.sid].to_numpy()
            dist[g.index.to_numpy()] = M.dist(gx, txy).min(1)
        T["dist_km"] = dist
        out[key] = {"overall": zstats(T), "gaussian_reference": GAUSS,
                    "by_distance": by_distance(T, BINS_T2 if buffer is None else BINS_T2B),
                    "distance_quantiles_km": {str(q): float(np.quantile(dist, q))
                                              for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                    "blocks_membership_consistent": int(sum(agree)), "blocks": len(agree)}
        T[["sid", "cluster", "t0", "t1", "dist_km"]].to_parquet(OUT / f"{key}_target_dist.parquet",
                                                                index=False)
    return out


def product_regime(lev: pd.DataFrame, g: dict) -> dict:
    bxy = lev.groupby("sid")[["x_km", "y_km"]].first().to_numpy()
    xy = g["cell_xy_km"]
    d = np.concatenate([M.dist(xy[i:i + 500], bxy).min(1) for i in range(0, len(xy), 500)])
    shares = {f"<= {x} km": float(np.mean(d <= x)) for x in (2, 4, 6, 10)}
    return {"cells": int(len(d)), "benchmarks": int(len(bxy)),
            "distance_quantiles_km": {str(q): float(np.quantile(d, q))
                                      for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "share_of_cells": shares,
            "note": ("distance from each twin-grid cell centre to the nearest leveling benchmark "
                     "(any survey 2012-2021); the product fits every benchmark")}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    g = D.grid()
    lev, qc = D.leveling(g)
    res = {"note": ("S5 fix round F.2, after the fact, report only (declared in S5.repair.md "
                    "F.2 before computation); spent units"),
           "leveling_qc": qc, "A_exposure": exposure(lev), "B_calibration": calibration(lev),
           "C_product_regime": product_regime(lev, g)}
    (OUT / "fix_after.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
