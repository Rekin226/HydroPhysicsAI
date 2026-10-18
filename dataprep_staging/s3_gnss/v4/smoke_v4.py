"""WP S3 fix round of the S3r repair (2026-10-18): smoke check of R7 declared in
``results/dataprep/reports/S3.repair.md`` 7.4 item 1, before any product or score.

    dataprep_staging/tools/cpu_slot.sh python -I -B dataprep_staging/s3_gnss/v4/smoke_v4.py OUT_DIR

1. The v4 WRA family is run on both variants (no injection) with the pass structure
   unrolled, and the final detection common mode (R7) is compared with the plain v3 median
   of the same pass-2 residual frame on the days on which all 5 neighbours report
   (expected identical) and on the other days (distribution of the change).
2. The composition events of ``diag_cm_composition.py`` are re-evaluated with the R7 common
   mode (composition part = its jump minus the constant-subset jump).
3. The v4 product tables of both variants are built in memory (``build_tables.build``) and
   compared with the v3 product: steps added/removed/changed, flag-0/1 days per WRA station
   in 2023-01..2025-03 (full), and the NGL daily table (identical apart from
   ``product_version``?).
No injection, no A1/A2/A3 score.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parents[1]))   # dataprep_staging
sys.path.insert(0, str(HERE))              # v4 modules
import build_tables as BT  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402

TUNE = REPO / "results/dataprep/reports/S3_evidence/repair_2026-10-18/tune_v3c/tune.json"
PRODUCT = REPO / "results/dataprep/gnss_clean"
W, GAP = 10, 10


def load_tuned(variant: str) -> tuple[M.Params, list[str]]:
    from dataclasses import replace
    t = json.loads(TUNE.read_text())
    p = replace(M.Params(), **{k: t["chosen_params"][k] for k in M.Params().as_dict()})
    return p, list(t["reference"] if variant == "preorigin" else t["reference_full"])


def wra_unrolled(p: M.Params, end) -> dict:
    ser = P.wra_series(end)
    cat = {k: [(d, t) for d, t in v if end is None or pd.Timestamp(d) <= end]
           for k, v in P.WRA_CATALOGUE.items()}
    drop, flag = P.wra_policy(end)
    proc = {k: v for k, v in ser.items() if k not in drop}
    members = [k for k in proc if k not in flag]
    xy = P.wra_station_xy()
    fam = P.Family("wra", {})
    cm, R2, info = None, None, {}
    for ipass in range(3):
        fam.results = {k: M.process_station(s, p, cm=None if cm is None else cm.get(k),
                                            catalogue=cat.get(k)) for k, s in proc.items()}
        if ipass < 2:
            R = fam.resid_frame()
            cm, info = P._knn_median_r7(R, members, xy, p.win, p.min_obs)
            R2 = R
    return {"R2": R2, "cm": cm, "info": info, "members": members, "xy": xy,
            "stations": list(proc)}


def gaps_of(av: pd.Series) -> list[tuple]:
    a = av.to_numpy()
    obs = np.flatnonzero(a)
    out = []
    if len(obs) < 2:
        return out
    for j in np.flatnonzero(np.diff(obs) > GAP + 1):
        out.append((av.index[obs[j] + 1], av.index[obs[j + 1]]))
    return out


def check(run: dict, variant: str) -> tuple[dict, pd.DataFrame]:
    R = run["R2"].sort_index()
    idx = pd.date_range(R.index.min(), R.index.max(), freq="D")
    R = R.reindex(idx)
    base = [m for m in run["members"] if m in R]
    ev_days = {}
    for k in base:
        av = R[k].notna()
        first, last = av[av].index.min(), av[av].index.max()
        ev = [g1 for _, g1 in gaps_of(av.loc[first:last])] + \
             [g0 for g0, _ in gaps_of(av.loc[first:last])] + [first, last + pd.Timedelta(days=1)]
        ev_days[k] = [d for d in ev if idx[0] < d <= idx[-1]]
    diffs_full, diffs_part, rows = [], [], []
    for s in run["stations"]:
        nb = P.knn_neighbours(s, base, run["xy"])
        if not nb:
            continue
        plain = R[nb].median(axis=1)
        v4 = run["cm"][s].reindex(idx)
        own = R[s].notna()
        allp = R[nb].notna().all(axis=1)
        d = (v4 - plain).abs()
        diffs_full.append(float(d[allp & own].max()) if (allp & own).any() else 0.0)
        diffs_part.extend((1000 * d[~allp & own & plain.notna()]).tolist())
        days = sorted({e for k in nb for e in ev_days.get(k, [])})
        for e in days:
            lo, hi = e - pd.Timedelta(days=W), e + pd.Timedelta(days=W - 1)
            b, a = R.loc[lo:e - pd.Timedelta(days=1), nb], R.loc[e:hi, nb]
            both = [k for k in nb if b[k].notna().sum() >= 3 and a[k].notna().sum() >= 3]
            jc = np.nan
            if both:
                cs = R[both].median(axis=1)
                jc = float(cs.loc[e:hi].median() - cs.loc[lo:e - pd.Timedelta(days=1)].median())
            j4 = float(v4.loc[e:hi].median() - v4.loc[lo:e - pd.Timedelta(days=1)].median())
            j3 = float(plain.loc[e:hi].median() - plain.loc[lo:e - pd.Timedelta(days=1)].median())
            rows.append({"variant": variant, "station": s, "date": e.date(),
                         "n_constant_subset": len(both),
                         "v3_composition_part_mm": 1000 * (j3 - jc),
                         "v4_composition_part_mm": 1000 * (j4 - jc)})
    ev = pd.DataFrame(rows)
    out = {"max_abs_diff_full_composition_days_m": float(max(diffs_full) if diffs_full else 0),
           "partial_days_abs_change_mm": {
               "n": len(diffs_part),
               "median": float(np.median(diffs_part)) if diffs_part else None,
               "p90": float(np.percentile(diffs_part, 90)) if diffs_part else None,
               "max": float(np.max(diffs_part)) if diffs_part else None},
           "n_events": int(len(ev))}
    for v in ("v3", "v4"):
        x = pd.to_numeric(ev[f"{v}_composition_part_mm"], errors="coerce").abs()
        out[f"{v}_composition_part_abs_mm"] = {
            "median": float(np.nanmedian(x)), "p90": float(np.nanpercentile(x, 90)),
            "max": float(np.nanmax(x)), "n_ge_5": int((x >= 5).sum()),
            "n_ge_10": int((x >= 10).sum())}
    out["share_days_imputed_by_station"] = {k: round(v["share_days_imputed"], 4)
                                            for k, v in run["info"].items()}
    out["days_no_neighbour_interpolated"] = {k: v["days_no_neighbour_interpolated"]
                                             for k, v in run["info"].items()
                                             if v["days_no_neighbour_interpolated"]}
    return out, ev


def compare_tables(tab: dict, variant: str) -> dict:
    v3 = PRODUCT / variant / "v3"
    s3 = pd.read_csv(v3 / "steps.csv")
    s4 = tab["steps"].copy()
    key = ["family", "station", "date"]
    m = s3.merge(s4, on=key, how="outer", suffixes=("_v3", "_v4"), indicator=True)
    only3 = m[m._merge == "left_only"][key + ["size_m_v3", "basis_v3"]]
    only4 = m[m._merge == "right_only"][key + ["size_m_v4", "basis_v4"]]
    both = m[m._merge == "both"]
    dsize = (both.size_m_v4 - both.size_m_v3).abs() * 1000
    basis_changed = both[both.basis_v3 != both.basis_v4][key + ["basis_v3", "basis_v4"]]
    out = {"steps_v3": int(len(s3)), "steps_v4": int(len(s4)),
           "only_v3": only3.astype(str).to_dict("records"),
           "only_v4": only4.astype(str).to_dict("records"),
           "common_size_abs_change_mm": {"max": float(dsize.max()) if len(dsize) else None,
                                         "median": float(dsize.median()) if len(dsize) else None},
           "basis_changed": basis_changed.astype(str).to_dict("records")}
    n3 = pd.read_parquet(v3 / "ngl" / "daily.parquet").drop(columns=["product_version"])
    n4 = tab["ngl_daily"].drop(columns=["product_version"]).reset_index(drop=True)
    out["ngl_daily_identical_to_v3"] = bool(n3.shape == n4.shape and n3.equals(n4))
    w3 = pd.read_parquet(v3 / "wra" / "daily.parquet")
    w4 = tab["wra_daily"]
    out["wra_rows_v3_v4"] = [int(len(w3)), int(len(w4))]
    for nm, w in (("v3", w3), ("v4", w4)):
        x = w[w.date >= "2023-01-01"] if variant == "full" else w
        out[f"wra_flag_counts_{nm}"] = {str(k): int(v) for k, v in
                                        x.flag.value_counts().sort_index().items()}
    if variant == "full":
        rows = {}
        for nm, w in (("v3", w3), ("v4", w4)):
            x = w[(w.date >= "2023-01-01") & (w.date <= "2025-03-31")]
            rows[nm] = x.groupby("station").flag.apply(lambda f: int(f.isin([0, 1]).sum()))
        T = pd.DataFrame(rows).fillna(0).astype(int)
        out["flag01_days_2023_25"] = T.to_dict("index")
    else:
        merged = w3.merge(w4, on=["station", "date"], suffixes=("_v3", "_v4"))
        out["wra_flag_changed_days"] = int((merged.flag_v3 != merged.flag_v4).sum())
        out["wra_value_max_abs_change_mm"] = float(
            1000 * (merged.value_v3 - merged.value_v4).abs().max())
    return out


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    res = {}
    for variant, end in (("preorigin", P.ORIGIN_END), ("full", None)):
        p, ref = load_tuned(variant)
        t0 = time.time()
        run = wra_unrolled(p, end)
        chk, ev = check(run, variant)
        ev.to_csv(od / f"events_r7_{variant}.csv", index=False)
        t1 = time.time()
        tab = BT.build(p, ref, end, version=f"gnss_clean/{variant}/v4")
        t2 = time.time()
        st = tab["stations"]
        notes = {k: st[(st.family == "wra") & (st.station == k)].notes.iloc[0]
                 for k in ("ANES", "FRES", "HLES", "HNES", "KTES", "SSES", "YCES")
                 if ((st.family == "wra") & (st.station == k)).any()}
        res[variant] = {"composition": chk, "tables_vs_v3": compare_tables(tab, variant),
                        "wra_notes": notes,
                        "seconds_unrolled_wra": round(t1 - t0, 1),
                        "seconds_build": round(t2 - t1, 1)}
        c = chk
        print(variant, json.dumps({"max_diff_full_days_m": c["max_abs_diff_full_composition_days_m"],
                                   "v3": c["v3_composition_part_abs_mm"],
                                   "v4": c["v4_composition_part_abs_mm"],
                                   "ngl_identical": res[variant]["tables_vs_v3"][
                                       "ngl_daily_identical_to_v3"],
                                   "seconds": [res[variant]["seconds_unrolled_wra"],
                                               res[variant]["seconds_build"]]}), flush=True)
    (od / "smoke_v4.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
