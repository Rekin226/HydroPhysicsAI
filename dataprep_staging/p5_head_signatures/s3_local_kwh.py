"""P5 stage 3: census electricity around every monitoring well (C0 v3, pre-origin months).

The pilot's ``local_kwh.py`` / ``local_kernel.py`` on the C0 census of both counties (plus the
in-grid Chiayi pumps): kept pumps, kWh as delivered by C0 ``preorigin/v3`` (missing readings
count as zero, as in the model; the flag-4 December-2022 fill is used as delivered), months
2012-01..2022-12 only (U7: no kWh statistic of P5 reads a post-origin month).

Per well and month: E_r1, E_r2, E_r3, E_r5 (sum within 1, 2, 3, 5 km), E_K0_1000 (periodic-Theis
kernel, pilot definition), and E_r2 by class group (no_domestic, irrigation). Per well:
pump counts and HP within radii, the number of occupied 45-degree sectors of the 3 km disk
(census coverage, pilot ``sectors3``; both counties' pumps count now), and the distance to the
nearest pump. Fan regional series (all in-grid kept pumps) by class group.

Usage: s3_local_kwh.py WORKDIR
Writes WORKDIR/local_kwh.parquet, WORKDIR/wells_kwh_meta.parquet, WORKDIR/regional_kwh.parquet
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import kv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p5_head_signatures import config as C  # noqa: E402
from p5_head_signatures import inputs as I  # noqa: E402

GROUPS = {"all": None, "nodom": "no_domestic", "irr": "irrigation"}


def load_pumps():
    cp = I.census_pumps()
    cp = cp[cp["kept"]].copy()
    vals, flags, uids, cls = [], [], [], []
    dates = None
    for cty in ("yunlin", "changhua", "chiayi"):
        z = I.pump_kwh(cty)
        d = [str(x)[:7] for x in z["dates"]]
        if dates is None:
            dates = d
        elif d != dates:
            raise ValueError("pump arrays disagree on dates")
        vals.append(np.asarray(z["value"], float))
        flags.append(np.asarray(z["flag"]))
        uids.append(np.asarray(z["uid"]).astype(str))
        cls.append(np.asarray(z["purpose_class"]).astype(str))
    V = np.vstack(vals)
    Fl = np.vstack(flags)
    U = np.concatenate(uids)
    K = np.concatenate(cls)
    months = pd.Index(dates)
    sel = (months >= C.KWH_MONTHS[0]) & (months <= C.KWH_MONTHS[1])
    V = V[:, sel]
    Fl = Fl[:, sel]
    months = months[sel]
    xy = cp.set_index("uid").reindex(U)
    info = {"n_pumps_arrays": int(len(U)), "n_without_census_row": int(xy["x_twd97"].isna().sum()),
            "flag_counts": {str(k): int(v) for k, v in
                            zip(*np.unique(Fl, return_counts=True), strict=True)},
            "nan_values": int(np.isnan(V).sum())}
    V = np.nan_to_num(V, nan=0.0)          # missing readings count as zero (model convention)
    x = xy["x_twd97"].to_numpy(float)
    y = xy["y_twd97"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 100000) & (y > 2500000)
    info["n_with_xy"] = int(ok.sum())
    return {"V": V[ok], "x": x[ok], "y": y[ok], "cls": K[ok],
            "in_grid": xy["in_grid"].to_numpy(bool)[ok],
            "hp": np.nan_to_num(xy["pump_hp"].to_numpy(float)[ok]),
            "county": xy["county"].to_numpy(str)[ok], "months": list(months), "info": info}


def main():
    work = Path(sys.argv[1])
    work.mkdir(parents=True, exist_ok=True)
    pm = load_pumps()
    V, px, py, cls = pm["V"], pm["x"], pm["y"], pm["cls"]
    months = pm["months"]
    tree = cKDTree(np.c_[px, py])
    meta = I.station_meta()
    meta = meta[np.isfinite(meta.x_twd97) & np.isfinite(meta.y_twd97)
                & (meta.x_twd97 > 100000)].copy()
    gmask = {"all": np.ones(len(cls), bool), "nodom": cls != "domestic",
             "irr": cls == "irrigation"}
    rows, wm = [], []
    for _, w in meta.iterrows():
        nb = np.asarray(tree.query_ball_point([w.x_twd97, w.y_twd97], 5000.0), int)
        d = np.hypot(px[nb] - w.x_twd97, py[nb] - w.y_twd97)
        m = {"sid": w.sid, "county": w.county, "layer": w.layer, "x": w.x_twd97,
             "y": w.y_twd97, "dmin_pump_m": float(d.min()) if len(d) else np.nan}
        ang = np.arctan2(py[nb] - w.y_twd97, px[nb] - w.x_twd97)
        sec = np.floor((ang + np.pi) / (np.pi / 4)).astype(int) % 8
        m["sectors3"] = int(len(set(sec[d < C.SECTOR_RADIUS_M]))) if len(d) else 0
        for r in C.LOCAL_RADII_M:
            s = d < r
            m[f"n_r{int(r // 1000)}"] = int(s.sum())
            m[f"hp_r{int(r // 1000)}"] = float(pm["hp"][nb[s]].sum())
        ser = {}
        for r in C.LOCAL_RADII_M:
            ser[f"E_r{int(r // 1000)}"] = V[nb[d < r]].sum(0)
        for g in ("nodom", "irr"):
            s = (d < 2000) & gmask[g][nb]
            ser[f"E_r2_{g}"] = V[nb[s]].sum(0)
        nk = np.asarray(tree.query_ball_point([w.x_twd97, w.y_twd97],
                                              4 * C.K0_DELTA_M + 500.0), int)
        dk = np.maximum(np.hypot(px[nk] - w.x_twd97, py[nk] - w.y_twd97), C.K0_RMIN_M)
        wk = np.abs(kv(0, dk / C.K0_DELTA_M * np.sqrt(1j)))
        ser["E_K0_1000"] = (V[nk] * wk[:, None]).sum(0) if len(nk) else np.zeros(len(months))
        s = gmask["nodom"][nk]
        ser["E_K0_1000_nodom"] = (V[nk[s]] * wk[s, None]).sum(0) if s.any() else \
            np.zeros(len(months))
        df = pd.DataFrame(ser)
        df.insert(0, "month", months)
        df.insert(0, "sid", w.sid)
        rows.append(df)
        wm.append(m)
    L = pd.concat(rows, ignore_index=True)
    L.to_parquet(work / "local_kwh.parquet", index=False)
    W = pd.DataFrame(wm)
    W.to_parquet(work / "wells_kwh_meta.parquet", index=False)
    ig = pm["in_grid"]
    reg = pd.DataFrame({"month": months,
                        "fan_all": V[ig].sum(0),
                        "fan_nodom": V[ig & gmask["nodom"]].sum(0),
                        "fan_irr": V[ig & gmask["irr"]].sum(0),
                        "fan_other": V[ig & ~gmask["irr"] & gmask["nodom"]].sum(0),
                        "fan_domestic": V[ig & ~gmask["nodom"]].sum(0)})
    for cty in ("yunlin", "changhua"):
        s = ig & (pm["county"] == cty)
        reg[f"{cty}_all"] = V[s].sum(0)
        reg[f"{cty}_irr"] = V[s & gmask["irr"]].sum(0)
        reg[f"{cty}_nodom"] = V[s & gmask["nodom"]].sum(0)
    reg.to_parquet(work / "regional_kwh.parquet", index=False)
    info = pm["info"] | {"n_wells": int(len(W)), "months": [months[0], months[-1]],
                         "fan_all_gwh_per_yr": float(reg.fan_all.sum() / 1e6 / 11.0)}
    (work / "local_kwh_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
