"""P2e build: cell x class arrays of the census extension 2026-07..2026-08 (recorded-forcing
arm only, U4) from the county totals of e2_extension.py, and the registered construction
checks P2e.k1.construction.

Spatial allocation (fixed in P2def.v1): a county's extended total is spread over class x cell
with that county's mean share of the same calendar month over 2023-2025 (C0 full v3 recorded
arm, cells_by_class value_by_county over the county-scope census total), so the out-of-grid
part of the county stays out. The in-grid Chiayi pumps (0.17 % of in-grid energy) take their
2023-2025 same-month ratio to the Yunlin county total times Yunlin's extension. Flag 5 (TPC
derived), sd = value x the county's monthly log RMS from backtest x2 (uncalibrated; Chiayi uses
Yunlin's), counties in quadrature.

Usage: python -I e3_build.py build <bt_dir> <out_dir>
       python -I e3_build.py check <bt_dir> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
import e2_extension as X  # noqa: E402

SHARE_YEARS = (2023, 2024, 2025)


def county_total(prod) -> dict:
    df = prod.read_table("county_class_monthly.parquet",
                         columns=["county", "scope", "class", "date", "value", "flag"])
    return {k: df[(df["county"] == k) & (df["scope"] == "county") & (df["class"] == "all")]
            .set_index("date")["value"] for k in D.COUNTIES}


def build(bt_dir: Path, out: Path) -> dict:
    ext = pd.read_csv(bt_dir / "P2e_county_extension.csv")
    bt = D.load_json(bt_dir / "P2e_backtest.json")
    prod = D.open_c0("full", X.USE)
    g = np.load(prod.path("cells_by_class.npz"), allow_pickle=False)
    vbk = g["value_by_county"]                     # (K, C, A, T) recorded arm
    tot = county_total(prod)
    K, C, A = vbk.shape[:3]
    months = list(X.EXT_MONTHS)
    val = np.zeros((K, C, A, len(months)))
    shares = {}
    for j, mo in enumerate(months):
        cal = int(mo[5:])
        ts = [D.tidx(f"{y}-{cal:02d}") for y in SHARE_YEARS]
        for ki, k in enumerate(D.COUNTIES):
            base = "yunlin" if k == "chiayi" else k
            den = np.array([tot[base][D.ym(t)] for t in ts])
            sh = np.mean([np.nan_to_num(vbk[ki, :, :, t]) / d for t, d in zip(ts, den, strict=False)], axis=0)
            hat = float(ext[(ext["county"] == base) & (ext["month"] == mo)]["census_hat_kwh"]
                        .iloc[0])
            val[ki, :, :, j] = hat * sh
            shares[(k, mo)] = float(sh.sum())
    sd_rel = {k: bt["yunlin" if k == "chiayi" else k]["sd_monthly_log"] for k in D.COUNTIES}
    sk = np.array([sd_rel[k] for k in D.COUNTIES])
    sdbk = val * sk[:, None, None, None]
    value = val.sum(0)
    sd = np.sqrt((sdbk ** 2).sum(0))
    flag = np.full(value.shape, 5, dtype=np.uint8)
    vt = value.sum(0)
    sdt = np.sqrt(((val.sum(1) * sk[:, None, None]) ** 2).sum(0))
    dom = D.CLASSES.index("domestic")
    vnd = vt - value[dom]
    sdnd = np.sqrt((((val.sum(1) - val[:, dom]) * sk[:, None, None]) ** 2).sum(0))
    out.mkdir(parents=True, exist_ok=True)
    meta = {x: g[x] for x in ("mask_hash", "grid_hash", "nx", "ny", "dx", "x0", "y0",
                              "n_active")}
    np.savez_compressed(
        out / "census_extension.npz", value=value, flag=flag, sd=sd,
        value_by_county=val, flag_by_county=np.broadcast_to(flag, val.shape).copy(),
        sd_by_county=sdbk, value_total_all=vt, flag_total_all=flag[0], sd_total_all=sdt,
        value_total_nodomestic=vnd, flag_total_nodomestic=flag[0],
        sd_total_nodomestic=sdnd, dates=np.array(months), classes=np.array(D.CLASSES),
        counties=np.array(D.COUNTIES),
        unit=np.array("kWh/month, model units (cleaned-census kWh), recorded-forcing arm"),
        method=np.array("tpc_farm_x_ratio_extension"),
        sd_label=np.array("uncalibrated (monthly log RMS of backtest x2, last 36 origins)"),
        **meta)
    # county table: extension totals, in-grid sums, and the census-based 2026-07 value of
    # P2abc full v2 (recorded arm with the July-2026 imputation) for comparison
    p2 = guard_p2abc_full()
    cmp = p2.read_table("county_class_monthly.parquet",
                        columns=["county", "scope", "class", "date", "arm", "value", "flag"])
    rows = []
    for ki, k in enumerate(D.COUNTIES):
        base = "yunlin" if k == "chiayi" else k
        for j, mo in enumerate(months):
            e = ext[(ext["county"] == base) & (ext["month"] == mo)].iloc[0]
            c26 = cmp[(cmp["county"] == k) & (cmp["scope"] == "in_grid") &
                      (cmp["class"] == "all") & (cmp["date"] == mo) & (cmp["arm"] == "recorded")]
            rows.append({"county": k, "month": mo, "candidate": e["candidate"],
                         "county_census_hat_kwh": float(e["census_hat_kwh"]) if k != "chiayi"
                         else np.nan,
                         "in_grid_extension_kwh": float(val[ki, :, :, j].sum()),
                         "in_grid_share": shares[(k, mo)],
                         "sd_rel": sd_rel[k], "flag": 5,
                         "p2abc_full_v2_recorded_in_grid_kwh": float(c26["value"].iloc[0])
                         if len(c26) else np.nan,
                         "p2abc_full_v2_flag": int(c26["flag"].iloc[0]) if len(c26) else -1})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "county_extension.csv", index=False)
    return {"shares": {f"{k}|{m}": v for (k, m), v in shares.items()}, "sd_rel": sd_rel}


def guard_p2abc_full():
    from common import guard
    return guard.open_product(D.P2ABC["full"], "forcing_recorded_arm")


def check(bt_dir: Path, out: Path) -> dict:
    res = {"items": {}, "pass": True}

    def item(name, ok, detail):
        res["items"][name] = {"pass": bool(ok), "detail": detail}
        res["pass"] = res["pass"] and bool(ok)

    z = np.load(out / "census_extension.npz", allow_pickle=False)
    g = np.load(D.C0["full"] / "cells_by_class.npz", allow_pickle=False)
    ext = pd.read_csv(bt_dir / "P2e_county_extension.csv")
    bt = D.load_json(bt_dir / "P2e_backtest.json")
    v, vbk = z["value"], z["value_by_county"]
    item("a_shapes_grid_dates",
         v.shape == (6, D.A_CELLS, 2) and vbk.shape == (3, 6, D.A_CELLS, 2) and
         str(z["mask_hash"]) == str(g["mask_hash"]) and
         str(z["grid_hash"]) == str(g["grid_hash"]) and list(z["dates"]) == list(X.EXT_MONTHS),
         {"shape": list(v.shape)})
    sk = np.array([bt["yunlin" if k == "chiayi" else k]["sd_monthly_log"]
                   for k in D.COUNTIES])
    ok_b = (bool(np.isfinite(v).all()) and bool((v >= 0).all()) and
            bool((z["flag"] == 5).all()) and
            np.allclose(z["sd_by_county"], vbk * sk[:, None, None, None], rtol=1e-12, atol=0))
    item("b_finite_nonneg_flag5_sd_rule", ok_b, {"sd_rel": sk.tolist()})
    prod = D.open_c0("full", X.USE)
    tot = county_total(prod)
    gv = np.load(prod.path("cells_by_class.npz"), allow_pickle=False)["value_by_county"]
    mx = 0.0
    for ki, k in enumerate(D.COUNTIES):
        base = "yunlin" if k == "chiayi" else k
        for j, mo in enumerate(X.EXT_MONTHS):
            cal = int(mo[5:])
            ts = [D.tidx(f"{y}-{cal:02d}") for y in SHARE_YEARS]
            frac = np.mean([np.nansum(gv[ki, :, :, t]) / tot[base][D.ym(t)] for t in ts])
            hat = float(ext[(ext["county"] == base) & (ext["month"] == mo)]["census_hat_kwh"]
                        .iloc[0])
            mx = max(mx, abs(vbk[ki, :, :, j].sum() / (hat * frac) - 1))
    item("c_ingrid_sum_equals_hat_x_share", mx <= 1e-9, {"max_rel": mx})
    item("d_aggregates", np.allclose(v, vbk.sum(0), rtol=1e-12, atol=1e-9) and
         np.allclose(z["value_total_all"], v.sum(0), rtol=1e-12, atol=1e-9), {})
    bad = [x for x in z.files if x.lower() in ("uid", "x", "y", "sid", "meter_idx")]
    item("e_privacy", not bad, {"bad_keys": bad})
    # f: county extension recomputed from the selected candidate (independent call)
    Yc = X.census_monthly(prod)
    Xc = {k: D.tpc_farm(k) for k in ("yunlin", "changhua")}
    for k in ("yunlin", "changhua"):
        Yc[k] = Yc[k][(Yc[k].index >= X.FIT0) & (Yc[k].index <= X.LAST_CENSUS)]
    best = {k: bt[k]["selected"] for k in ("yunlin", "changhua")}
    yl = X.fit_predict(best["yunlin"], Yc["yunlin"], Xc["yunlin"], X.LAST_CENSUS,
                       list(X.EXT_MONTHS))
    mx2 = 0.0
    for k in ("yunlin", "changhua"):
        kw = {"yl_pred": yl, "Y_yl": Yc["yunlin"]} if best[k] == "scaled" else {}
        p = X.fit_predict(best[k], Yc[k], Xc[k], X.LAST_CENSUS, list(X.EXT_MONTHS), **kw)
        for mo in X.EXT_MONTHS:
            h = float(ext[(ext["county"] == k) & (ext["month"] == mo)]["census_hat_kwh"].iloc[0])
            mx2 = max(mx2, abs(p[mo] / h - 1))
    item("f_county_extension_recomputed", mx2 <= 1e-12, {"max_rel": mx2})
    return res


if __name__ == "__main__":
    cmd, bt_dir, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if cmd == "build":
        print(build(bt_dir, out))
    else:
        r = check(bt_dir, out)
        D.save_json(r, out / "P2e_k1_construction.json")
        print(r)
