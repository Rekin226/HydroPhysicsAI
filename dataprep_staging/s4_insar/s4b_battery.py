"""WP S4b: the P3 transfer battery as eight held-out subsets of Yunlin cells (plan WP P3
"Validation": easting terciles; the most Changhua-like towns; town halves by paddy share, both
directions; town halves by aquaculture area, both directions).

P3 has not built its product yet, so the subsets are rebuilt from the design pilot that defined
them (``_pilots/design_ml_transfer/a2_robust_transfer.py``: ``tr`` = Yunlin cells with county
share rules and complete township crop features; held-out sets per split). The pilot files are
read-only design-time inputs; the Changhua-like town list is the pilot's recorded
``chlike_towns`` (a HistGradientBoosting propensity, not refitted). Only cell memberships come
out of this module; no census value or InSAR value is used to form them. ``main`` checks the
rebuilt training set against the pilot's recorded size (1091 cells).

    python -I s4b_battery.py      (hydro env) -> results/dataprep/insar_tools/s4b_run/battery.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import s4b_config as C  # noqa: E402

LU = C.DATAPREP / "_pilots" / "landuse_crop_priors"
ML = C.DATAPREP / "_pilots" / "design_ml_transfer"
INPUTS = [LU / "features.npz", LU / "target.npz", LU / "township.npz",
          LU / "township_crops.csv", ML / "a2_results.json"]
SPLITS = ("EAST3_west", "EAST3_mid", "EAST3_east", "CHLIKE", "PADDY_test_high",
          "PADDY_test_low", "AQUA_test_high", "AQUA_test_low")
CROPF = ["c_upland", "c_veg", "c_rice1", "c_rice2", "c_fruit", "c_fallow", "c_intensity"]


def _cells_frame() -> tuple[pd.DataFrame, list[str]]:
    f = np.load(LU / "features.npz", allow_pickle=True)
    t = np.load(LU / "target.npz", allow_pickle=True)
    groups = list(f["groups"])
    A = f["area_m2"].shape[1]
    df = pd.DataFrame(index=np.arange(A))
    for k in ("paddy", "dry", "orchard", "aqua"):
        df[k] = f["area_m2"][groups.index(k)] / 1e4
    sh = f["county_share"]
    df["sh_chg"], df["sh_yun"], df["sh_nan"], df["sh_oth"] = sh
    df["x"] = f["cxy"][:, 0]
    # township of a cell: majority of pumps; empty cells take the nearest pump's township
    from scipy.spatial import cKDTree
    tid = t["town_id"].copy()
    miss = tid < 0
    _, j = cKDTree(t["pump_xy"]).query(f["cxy"][miss])
    tid[miss] = t["pump_town"][j]
    df["town"] = tid
    # township crop features (a2_robust_transfer.add_crop)
    tz = np.load(LU / "township.npz", allow_pickle=True)
    names = list(tz["names"])
    ct = pd.read_csv(LU / "township_crops.csv")
    ct = ct[ct["年度"].isin([103, 104, 105])].groupby("town").mean(numeric_only=True)
    planted = ct[["upland", "veg", "rice1", "rice2", "fruit", "flower", "forage", "special",
                  "other"]].sum(1)
    tgroups = list(tz["groups"])
    farm_town = tz["town_class_area_m2"][:, [tgroups.index(g) for g in
                                             ("paddy", "dry", "orchard")]].sum(1) / 1e4
    tab = pd.DataFrame(index=names)
    for c, src in (("c_upland", "upland"), ("c_veg", "veg"), ("c_rice1", "rice1"),
                   ("c_rice2", "rice2"), ("c_fruit", "fruit"), ("c_fallow", "fallow")):
        tab[c] = (ct[src] / planted).reindex(names)
    tot_area = tz["town_area_total_m2"] / 1e4
    ingrid = tz["town_class_area_m2"].sum(1) / 1e4
    frac_in = np.where(tot_area > 0, ingrid / np.maximum(tot_area, 1e-9), np.nan)
    tab["c_intensity"] = (planted.reindex(names).to_numpy() * frac_in
                          / np.maximum(farm_town, 1e-9))
    ctown = tz["cell_town"]
    for c in CROPF:
        v = tab[c].to_numpy()
        df[c] = np.where(ctown >= 0, v[np.clip(ctown, 0, None)], np.nan)
    return df, list(t["towns"])


def subsets() -> dict:
    df, towns = _cells_frame()
    tr = df[(df.sh_yun + df.sh_oth >= 0.95) & (df.sh_yun > 0.3) & (df.sh_chg < 0.02)].copy()
    tr = tr[tr[CROPF].notna().all(1)]
    out = {}
    q1, q2 = np.quantile(tr.x, [1 / 3, 2 / 3])
    for nm, msk in (("west", tr.x < q1), ("mid", (tr.x >= q1) & (tr.x < q2)),
                    ("east", tr.x >= q2)):
        out[f"EAST3_{nm}"] = tr.index[msk.to_numpy()].tolist()
    chl = json.loads((ML / "a2_results.json").read_text())["chlike_towns"]
    hold = [towns.index(n) for n in chl]
    out["CHLIKE"] = tr.index[tr.town.isin(hold).to_numpy()].tolist()
    g = tr.groupby("town")[["paddy", "dry", "orchard", "aqua"]].sum()
    g["pshare"] = g.paddy / g[["paddy", "dry", "orchard"]].sum(1)
    for key, col in (("PADDY", "pshare"), ("AQUA", "aqua")):
        hi_t = list(g[col].sort_values().index[len(g) // 2:])
        m_hi = tr.town.isin(hi_t).to_numpy()
        out[f"{key}_test_high"] = tr.index[m_hi].tolist()
        out[f"{key}_test_low"] = tr.index[~m_hi].tolist()
    return {"train_cells": tr.index.tolist(), "splits": {k: out[k] for k in SPLITS},
            "chlike_towns": chl}


def main() -> int:
    b = subsets()
    n = len(b["train_cells"])
    rec = {"n_train_cells": n, "pilot_n_train_cells": 1091, "matches_pilot": n == 1091,
           "split_sizes": {k: len(v) for k, v in b["splits"].items()}, **b}
    C.RUN_DIR.mkdir(parents=True, exist_ok=True)
    (C.RUN_DIR / "battery.json").write_text(json.dumps(rec, ensure_ascii=False))
    print(json.dumps({k: rec[k] for k in ("n_train_cells", "matches_pilot", "split_sizes")},
                     ensure_ascii=False))
    return 0 if n == 1091 else 1


if __name__ == "__main__":
    raise SystemExit(main())
