"""WP S4b: report-only rows registered in ``S4b.insar_acceptance.v1`` and computed AFTER the
registered tests were scored on run P (looks logged by ``s4b_looks.py`` beforehand).

    python -I s4b_after.py      (hydro env, one CPU slot)

Rows:
  F_g1          G1 on the full-network run F, same stations and pre-origin window
  pilot_window  run F over the pilot's post-origin window 2021-12-05 -> 2023-02-20
  ngl_onfan     NGL on-fan stations with the WRA tie (runs P and F), pre-origin window
  p_vs_f        run P against run F at the cells (pre-origin cumulative change)
  c1_all_pairs  C1 statistic over every pre-origin pair
  decomp_2025   ascending (F) / descending (D) decomposition of the 2025 LOS rates
Writes ``results/dataprep/reports/S4b_evidence/after.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyproj  # noqa: E402
import s4b_config as C  # noqa: E402
import s4b_evaluate as Ev  # noqa: E402
import s4b_gnss_inventory as GI  # noqa: E402
import s4b_lib as Lb  # noqa: E402

NGL_DIRS = [C.REPO / "data_fetch_api" / "new_measurements" / "ngl_gnss" / d
            for d in ("2026-10-12_s3", "2026-10-12_s3b", "2026-10-12")]
NGL_ONFAN = ("DNAN", "FUNY", "HUWE", "JSAN", "PUS2")
_MON = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG",
                                        "SEP", "OCT", "NOV", "DEC"])}


def _day(s: str) -> pd.Timestamp:
    yy = int(s[:2])
    return pd.Timestamp(year=(1900 + yy) if yy >= 90 else (2000 + yy), month=_MON[s[2:5]],
                        day=int(s[5:7]))


def read_tenv3(st: str) -> tuple[pd.Series, float, float] | None:
    """Daily up (m, u0 + up) of an NGL station, and its median lon/lat (pure text parsing)."""
    p = next((d / f"{st}.tenv3" for d in NGL_DIRS if (d / f"{st}.tenv3").is_file()), None)
    if p is None:
        return None
    rows = []
    with open(p, encoding="ascii", errors="replace") as f:
        hdr = next(f).split()
        ix = {h: i for i, h in enumerate(hdr)}
        for line in f:
            q = line.split()
            if len(q) < len(hdr):
                continue
            try:
                rows.append((_day(q[ix["YYMMMDD"]]),
                             float(q[ix["u0(m)"]]) + float(q[ix["____up(m)"]]),
                             float(q[ix["_latitude(deg)"]]),
                             ((float(q[ix["_longitude(deg)"]]) + 180.0) % 360.0) - 180.0))
            except (ValueError, KeyError):
                continue
    df = pd.DataFrame(rows, columns=["d", "up", "lat", "lon"]).drop_duplicates("d")
    s = df.set_index("d")["up"].sort_index()
    return s, float(df["lon"].median()), float(df["lat"].median())


def ngl_equipment_steps(st: str, lo, hi) -> list[str]:
    out = []
    p = NGL_DIRS[0] / "steps.txt"
    with open(p, encoding="ascii", errors="replace") as f:
        for line in f:
            q = line.split()
            if len(q) >= 3 and q[0] == st and q[2] == "1":
                d = _day(q[1])
                if lo <= d <= hi:
                    out.append(str(d.date()))
    return out


def g1_rows(run_name: str, inv: pd.DataFrame, t0: str, t1: str) -> dict:
    run, g, v, _ = Ev.processed(run_name)
    st = Ev.station_set(inv, "exclude")
    ins, gn = Ev.gnss_tables(run, g, v, inv, st, "exclude")
    return {"loso": Ev.loso_eval(ins, gn, t0, t1, "loso"),
            "tie_none": Ev.loso_eval(ins, gn, t0, t1, "none")}, (run, g, v, ins, gn)


def ngl_rows(ctx, ev_stations: list[str], t0: str, t1: str) -> dict:
    run, g, v, ins, gn = ctx
    lo = pd.Timestamp(t0) - pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
    hi = pd.Timestamp(t1) + pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
    sel = (ins.index >= lo) & (ins.index <= hi)
    dd = (ins[ev_stations] - gn[ev_stations]).loc[sel]
    off = (dd - dd.mean()).mean(axis=1)
    tr = pyproj.Transformer.from_crs(4326, 3826, always_xy=True)
    out = {}
    for st in NGL_ONFAN:
        r = read_tenv3(st)
        if r is None:
            out[st] = {"note": "no tenv3"}
            continue
        s, lon, lat = r
        x, y = tr.transform(lon, lat)
        i, j, dist = Lb.station_pixel(g, x, y)
        steps = ngl_equipment_steps(st, lo, hi)
        iv = pd.Series(Lb.station_series(v, run["mask"], i, j), index=run["dates"]).loc[sel]
        gv = Lb.at_epochs(Lb.smooth_daily(s * 1000.0), run["dates"]).loc[sel]
        cal = iv - off
        both = cal.notna() & gv.notna()
        c0, n0 = Lb.window_mean(cal.where(both), t0)
        c1, n1 = Lb.window_mean(cal.where(both), t1)
        g0, _ = Lb.window_mean(gv.where(both), t0)
        g1, _ = Lb.window_mean(gv.where(both), t1)
        ok = n0 >= C.GNSS_MIN_EPOCHS_PER_WINDOW and n1 >= C.GNSS_MIN_EPOCHS_PER_WINDOW
        out[st] = {"pix_dist_m": round(dist, 1), "equipment_steps_in_span": steps,
                   "n_ep_w0": n0, "n_ep_w1": n1, "evaluable": bool(ok and not steps),
                   "insar_change_mm": c1 - c0 if ok else None,
                   "gnss_change_mm": g1 - g0 if ok else None,
                   "err_mm": (c1 - c0) - (g1 - g0) if ok else None}
    e = [v["err_mm"] for v in out.values() if v.get("evaluable") and v["err_mm"] is not None]
    return {"stations": out, "n": len(e),
            "rms_mm": float(np.sqrt(np.mean(np.square(e)))) if e else None}


def p_vs_f() -> dict:
    A = Lb.n_active()
    res = {}
    vals = {}
    for nm in ("P", "F"):
        run, g, v, _ = Ev.processed(nm)
        common = run["dates"][run["dates"] <= pd.Timestamp("2022-12-31")]
        vals[nm] = (run, g, v, common)
    dp, df_ = vals["P"][3], vals["F"][3]
    common = dp.intersection(df_)
    last = common.max()
    for nm in ("P", "F"):
        run, g, v, _ = vals[nm]
        k0 = int(np.flatnonzero(run["dates"] == common.min())[0])
        k1 = int(np.flatnonzero(run["dates"] == last)[0])
        cv, _, _ = Lb.to_cells(v[k1] - v[k0], run["mask"], g, A)
        res[nm] = cv
    d = res["F"] - res["P"]
    ok = np.isfinite(d)
    return {"first": str(common.min().date()), "last": str(last.date()),
            "n_cells": int(ok.sum()), "median_diff_mm": float(np.median(d[ok])),
            "rms_diff_mm": float(np.sqrt(np.mean(d[ok] ** 2))),
            "p95_abs_diff_mm": float(np.percentile(np.abs(d[ok]), 95)),
            "corr": float(np.corrcoef(res["P"][ok], res["F"][ok])[0, 1]),
            "median_change_P_mm": float(np.median(res["P"][ok])),
            "median_change_F_mm": float(np.median(res["F"][ok]))}


def c1_all() -> dict:
    z = np.load(C.RUN_DIR / "coh_P_all_preorigin.npz", allow_pickle=False)
    U = Ev.units()
    med = z["cell_median_coh"]
    out = {"n_pairs": int(med.shape[0])}
    for zid in C.CH_ZONES:
        cells = np.array(U[zid]["cells"], int)
        coh = med[:, cells]
        frac = np.where(np.isfinite(coh), coh >= C.COH_MIN, False).mean(1)
        out[zid] = {"median_frac_coherent": float(np.median(frac))}
    return out


def rate_cells(name: str, span) -> tuple[np.ndarray, dict]:
    run, g, _, _ = Ev.processed(name)
    los, _ = Lb.reference(run, g)
    lo, hi = (pd.Timestamp(t) for t in span)
    sel = (run["dates"] >= lo) & (run["dates"] <= hi)
    t = Lb.decimal_years(run["dates"][sel])
    Y = los[sel].reshape(int(sel.sum()), -1).astype(float)
    ok = np.isfinite(Y).all(0)
    X = np.column_stack([np.ones_like(t), t - t.mean()])
    rate = np.full(Y.shape[1], np.nan)
    B, *_ = np.linalg.lstsq(X, Y[:, ok], rcond=None)
    rate[ok] = B[1]
    rate = rate.reshape(los.shape[1:])
    A = Lb.n_active()
    valid = run["mask"] & np.isfinite(rate)
    rc, _, _ = Lb.to_cells(rate, valid, g, A)
    E, _, _ = Lb.to_cells(g["E"], np.isfinite(g["E"]), g, A)
    Uc, _, _ = Lb.to_cells(g["U"], np.isfinite(g["U"]), g, A)
    return rc, {"E": E, "U": Uc, "n_epochs": int(sel.sum())}


def decomp_2025() -> dict:
    va, ga = rate_cells("F", C.DECOMP_SPAN)
    vd, gd = rate_cells("D", C.DECOMP_SPAN)
    ok = np.isfinite(va) & np.isfinite(vd) & np.isfinite(ga["E"]) & np.isfinite(gd["E"])
    det = ga["E"] * gd["U"] - gd["E"] * ga["U"]
    east = (va * gd["U"] - vd * ga["U"]) / det
    up = (ga["E"] * vd - gd["E"] * va) / det
    vert_eq = va / ga["U"]
    d = vert_eq - up
    np.savez_compressed(C.EVID / "decomp_2025_cells.npz", up=up, east=east, vert_eq_asc=vert_eq,
                        los_rate_asc=va, los_rate_desc=vd, ok=ok)
    return {"span": list(C.DECOMP_SPAN), "n_epochs_asc": ga["n_epochs"],
            "n_epochs_desc": gd["n_epochs"], "n_cells": int(ok.sum()),
            "median_up_mm_yr": float(np.median(up[ok])),
            "median_east_mm_yr": float(np.median(east[ok])),
            "p05_p95_east_mm_yr": [float(np.percentile(east[ok], 5)),
                                   float(np.percentile(east[ok], 95))],
            "median_vert_eq_minus_up_mm_yr": float(np.median(d[ok])),
            "rms_vert_eq_minus_up_mm_yr": float(np.sqrt(np.mean(d[ok] ** 2))),
            "frac_cells_abs_diff_gt_10mm_yr": float(np.mean(np.abs(d[ok]) > 10)),
            "corr_vert_eq_up": float(np.corrcoef(vert_eq[ok], up[ok])[0, 1]),
            "note": "rates referenced to each frame's own stable area (north component "
                    "ignored); after-the-fact analysis"}


def main() -> int:
    C.EVID.mkdir(parents=True, exist_ok=True)
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    out = {}
    out["F_g1"], ctxF = g1_rows("F", inv, C.GNSS_T0, C.GNSS_T1)
    _, ctxP = g1_rows("P", inv, C.GNSS_T0, C.GNSS_T1)
    invp = GI.inventory(C.PILOT_T0, C.PILOT_T1)
    out["pilot_window_F"], _ = g1_rows("F", invp, C.PILOT_T0, C.PILOT_T1)
    evP = json.loads((C.EVID / "test_gnss.json").read_text())["primary"]["stations"]
    evF = out["F_g1"]["loso"]["stations"]
    out["ngl_onfan_P"] = ngl_rows(ctxP, evP, C.GNSS_T0, C.GNSS_T1)
    out["ngl_onfan_F"] = ngl_rows(ctxF, evF, C.GNSS_T0, C.GNSS_T1)
    out["p_vs_f"] = p_vs_f()
    out["c1_all_pairs"] = c1_all()
    out["decomp_2025"] = decomp_2025()
    Lb.jdump(out, C.EVID / "after.json")
    brief = {"F_g1_rms": out["F_g1"]["loso"]["rms_mm"],
             "pilot_window_rms": out["pilot_window_F"]["loso"]["rms_mm"],
             "ngl_P_rms": out["ngl_onfan_P"]["rms_mm"], "ngl_F_rms": out["ngl_onfan_F"]["rms_mm"],
             "p_vs_f_rms": out["p_vs_f"]["rms_diff_mm"]}
    print(json.dumps(brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
