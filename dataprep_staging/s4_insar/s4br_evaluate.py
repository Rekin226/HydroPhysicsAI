"""WP S4b repair round 2 (S4br): the re-tests registered in ``S4b.insar_repair.v1`` and their
report-only rows. Every registered score calls ``prereg3.assert_can_score`` immediately before
it is computed (A1 rule). Gate run: the repaired pre-origin run P2 only (U7).

    python -I s4br_evaluate.py design      fresh NGL window and station set from metadata and
                                           InSAR epoch dates only (no height value is read);
                                           written before the pre-registration, which copies it
    python -I s4br_evaluate.py g1_ngl      G1 on the FRESH NGL family (registered, gate)
    python -I s4br_evaluate.py g1_wra      G1 on the WRA family (SECOND LOOK: declared in the
                                           pre-registration, logged before it is computed)
    python -I s4br_evaluate.py y           Y on run P2 (SECOND LOOK, same handling)

Outputs: ``results/dataprep/reports/S4b_evidence/repair_2026-10-18/test_<cmd>.json``.
Hydro env; one CPU slot.
"""
from __future__ import annotations

import argparse
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
import s4b_lib as Lb  # noqa: E402
import s4br_config as R  # noqa: E402
import s4br_lib as RL  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

T_NGL = "S4b.r2.G1_ngl_fresh_preorigin"
T_WRA = "S4b.r2.G1_wra_second_look_preorigin"
T_Y = "S4b.r2.Y_second_look_preorigin"
ROUND1_TERMS = C.REF_TERMS                     # offset + east/north plane + height


def _score(test_id: str) -> dict:
    from harness import prereg3
    return prereg3.assert_can_score(R.PREREG_ID, R.WP, test_id=test_id)


def _look(units: list[str], scope: str, purpose: str, note: str, members=None) -> dict:
    """Second looks and report-only rows on spent units: declared in S4b.insar_repair.v1
    (``second_looks``) and logged here immediately BEFORE they are computed."""
    from harness import prereg3
    return prereg3.record_look(units, scope, R.WP, purpose, prereg_id=R.PREREG_ID,
                               members=members, note=note)


def evid(name: str) -> Path:
    R.EVID.mkdir(parents=True, exist_ok=True)
    return R.EVID / name


# ----------------------------------------------------------------------------- design
def epoch_dates(name: str) -> pd.DatetimeIndex:
    import h5py
    with h5py.File(R.RUN_DIR / name / "TS_GEOCml2" / R.CUM_FILE, "r") as f:
        return pd.to_datetime([str(d) for d in f["imdates"][()]], format="%Y%m%d")


def fresh_window(dates: pd.DatetimeIndex) -> dict:
    """Rule (pre-registered): t0 = a month end, t1 = t0 + 14 month ends, both +-30 d windows
    inside [FRESH_FIRST, FRESH_LAST]; choose the t0 that maximises the smaller number of run
    epochs in the two windows; ties -> the latest t0."""
    h = pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
    first, last = pd.Timestamp(R.FRESH_FIRST), pd.Timestamp(R.FRESH_LAST)
    cands = []
    for t0 in pd.date_range(first, last, freq="ME"):
        t1 = t0 + pd.offsets.MonthEnd(R.CHANGE_MONTHS)
        if t0 - h < first or t1 + h > last:
            continue
        n0 = int(((dates >= t0 - h) & (dates <= t0 + h)).sum())
        n1 = int(((dates >= t1 - h) & (dates <= t1 + h)).sum())
        cands.append({"t0": str(t0.date()), "t1": str(t1.date()), "n_epochs_w0": n0,
                      "n_epochs_w1": n1, "min": min(n0, n1)})
    best = max(cands, key=lambda r: (r["min"], r["t0"]))
    lo_m = (pd.Timestamp(best["t0"]) - h).strftime("%Y-%m")
    hi_m = (pd.Timestamp(best["t1"]) + h).strftime("%Y-%m")
    return {"scope": f"{lo_m}..{hi_m}", "rule": "t0 = month end, t1 = t0 + 14 month ends, both +-30 d windows inside "
                    f"{R.FRESH_FIRST}..{R.FRESH_LAST}; maximise min(epochs in w0, w1) of "
                    "run P2; ties -> latest t0",
            "candidates": cands, "t0": best["t0"], "t1": best["t1"]}


def ngl_dates_only(st: str) -> pd.DatetimeIndex:
    """Dates of the daily solutions of an NGL station (date column only; no height parsed)."""
    p = RL.ngl_tenv3_path(st)
    out = []
    with open(p, encoding="ascii", errors="replace") as f:
        hdr = next(f).split()
        k = hdr.index("YYMMMDD")
        for line in f:
            q = line.split()
            if len(q) > k:
                try:
                    out.append(RL._day(q[k]))
                except (KeyError, ValueError):
                    continue
    return pd.DatetimeIndex(sorted(set(out)))


def ngl_station_set(t0: str, t1: str) -> dict:
    """Rule (pre-registered): NGL stations with a tenv3 file in L0 and a DataHoldings position
    inside an active twin cell and within NGL_MAX_PIX_DIST_M of a primary multilooked pixel
    centre; eligible when each +-30 d window holds >= 20 daily solutions and no equipment
    change (steps.txt type 1) lies in [t0 - 30 d, t1 + 30 d]."""
    h = RL.ngl_holdings()
    g = RL.load_geom(C.PRIMARY)
    mask, dx, x0, y0 = Lb.twin_grid()
    tr = pyproj.Transformer.from_crs(4326, 3826, always_xy=True)
    hw = pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
    lo, hi = pd.Timestamp(t0) - hw, pd.Timestamp(t1) + hw
    rows = {}
    for d in R.NGL_DIRS:
        for p in sorted(d.glob("*.tenv3")):
            st = p.stem
            if st in rows or st not in h.index:
                continue
            x, y = tr.transform(float(h.loc[st, "lon"]), float(h.loc[st, "lat"]))
            cell = int(Lb.cell_index(np.array([x]), np.array([y]), mask, dx, x0, y0)[0])
            i, j, dist = Lb.station_pixel(g, x, y)
            rec = {"cell": cell, "ml_i": i, "ml_j": j, "pix_dist_m": round(dist, 1),
                   "x": x, "y": y}
            if cell < 0 or dist > R.NGL_MAX_PIX_DIST_M:
                rec["status"] = "not on an active cell inside the grid"
                rows[st] = rec
                continue
            dd = ngl_dates_only(st)
            n0 = int(((dd >= pd.Timestamp(t0) - hw) & (dd <= pd.Timestamp(t0) + hw)).sum())
            n1 = int(((dd >= pd.Timestamp(t1) - hw) & (dd <= pd.Timestamp(t1) + hw)).sum())
            eq = RL.ngl_steps(st, lo, hi, types=R.NGL_STEP_TYPES)
            eq2 = RL.ngl_steps(st, lo, hi, types=("2",))
            rec.update({"n_days_w0": n0, "n_days_w1": n1, "equipment_steps_in_span": eq,
                        "earthquake_flags_in_span_not_applied": eq2})
            ok = n0 >= C.GNSS_MIN_DAYS_PER_WINDOW and n1 >= C.GNSS_MIN_DAYS_PER_WINDOW
            rec["status"] = ("eligible" if ok and not eq else
                             "equipment change in span" if eq else "too few daily solutions")
            rows[st] = rec
    return rows


def cmd_design(_a) -> dict:
    d = epoch_dates("P2")
    w = fresh_window(d)
    st = ngl_station_set(w["t0"], w["t1"])
    out = {"window": w, "stations": st,
           "eligible": sorted(k for k, v in st.items() if v["status"] == "eligible"),
           "p2_epochs": [str(x.date()) for x in d],
           "note": "metadata, InSAR epoch dates and NGL solution dates only; no InSAR or GNSS "
                   "displacement value is read"}
    return out


# ----------------------------------------------------------------------------- G1 core
def ngl_tables(run: dict, v: np.ndarray, stations: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    Ins, Gn = {}, {}
    for st, rec in stations.items():
        Ins[st] = Lb.station_series(v, run["mask"], int(rec["ml_i"]), int(rec["ml_j"]))
        s = RL.read_ngl_up(st) * 1000.0
        Gn[st] = Lb.at_epochs(Lb.smooth_daily(s), run["dates"]).to_numpy()
    return (pd.DataFrame(Ins, index=run["dates"]), pd.DataFrame(Gn, index=run["dates"]))


def g1(Ins: pd.DataFrame, Gn: pd.DataFrame, t0: str, t1: str) -> dict:
    res = Ev.loso_eval(Ins, Gn, t0, t1, "loso")
    n = res["n_evaluable"]
    rms = res.get("rms_mm")
    res["passed"] = bool(rms is not None and n >= C.GNSS_MIN_STATIONS
                         and rms <= C.GNSS_LOSO_RMS_MAX_MM)
    res["threshold"] = f"RMS <= {C.GNSS_LOSO_RMS_MAX_MM} mm with >= {C.GNSS_MIN_STATIONS} " \
                       "evaluable stations (fewer = fail)"
    return res


def run_variant(name: str, terms, cum_file: str = R.CUM_FILE):
    run = RL.load_run(name, cum_file)
    g = RL.load_geom(run["frame"])
    los, tab = Lb.reference(run, g, terms)
    return run, g, Lb.vertical(los, g), tab


def design_doc() -> dict:
    return json.loads(evid("design.json").read_text())


def cmd_g1_ngl(_a) -> dict:
    des = design_doc()
    t0, t1 = des["window"]["t0"], des["window"]["t1"]
    elig = {k: des["stations"][k] for k in des["eligible"]}
    sc = _score(T_NGL)
    run, g, v, tab = run_variant("P2", R.REF_TERMS)
    Ins, Gn = ngl_tables(run, v, elig)
    prim = g1(Ins, Gn, t0, t1)
    out = {"test_id": T_NGL, "run": "P2", "window": [t0, t1], "primary": prim,
           "prereg_sha256": sc["sha256"],
           "post_registration_looks": sc["post_registration_looks"],
           "unverifiable_prior_looks_n": len(sc["unverifiable_prior_looks"]),
           "reference_table_P2": tab.to_dict("records")}
    rep = {}
    # registered report-only rows (computed after the primary score)
    r1, g1_, v1, _ = run_variant("P2", ROUND1_TERMS)
    rep["round1_plane_reference"] = g1(*ngl_tables(r1, v1, elig), t0, t1)
    r2, g2, v2, _ = run_variant("P2", R.REF_TERMS, "cum.h5")
    rep["unfiltered_cum"] = g1(*ngl_tables(r2, v2, elig), t0, t1)
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    st_w = Ev.station_set(inv, "exclude")
    w_scope = f"insar_wra_gnss_tie:{des['window']['scope']}"
    _look(["s4b.r2.wra_gnss.tie_fresh_window"], w_scope, "report_only",
          "WRA stations as tie stations (never scored) in the fresh NGL window",
          members={"s4b.r2.wra_gnss.tie_fresh_window": {
              "description": "run-P2 InSAR and WRA L0 heights at the 18 round-1 G1 stations, "
                             "used only as tie stations of the report-only combined tie",
              "ids": [f"wra_gnss:{k}" for k in st_w], "window": des["window"]["scope"]}})
    Iw, Gw = Ev.gnss_tables(run, g, v, inv, st_w, "exclude")
    rep["combined_wra_ngl_tie"] = combined_tie(Ins, Gn, Iw, Gw, t0, t1)
    # second look in round 1's G1 window (round 1's NGL row looked there, after the fact):
    # the same station rule applied to that window
    st_old = ngl_station_set(C.GNSS_T0, C.GNSS_T1)
    elig_old = {k: v_ for k, v_ in st_old.items() if v_.get("status") == "eligible"}
    _look(["s4b.r2.ngl_gnss.round1_window"], "insar_ngl_gnss_change:2021-08..2022-12",
          "second_look_score", "NGL on-fan stations in round 1's G1 window, where round 1's "
          "after-the-fact NGL row already looked: SECOND LOOK, report only",
          members={"s4b.r2.ngl_gnss.round1_window": {
              "description": "run-P2 InSAR and NGL L0 heights at the NGL stations eligible "
                             "in round 1's G1 window",
              "ids": [f"ngl_gnss:{k}" for k in sorted(elig_old)],
              "window": "2021-08..2022-12"}})
    rep["ngl_round1_g1_window_second_look"] = g1(*ngl_tables(run, v, elig_old),
                                                 C.GNSS_T0, C.GNSS_T1)
    rep["ngl_round1_g1_window_second_look"]["station_rule"] = {
        k: v_.get("status") for k, v_ in st_old.items() if v_.get("cell", -1) >= 0}
    out["report_only"] = rep
    Ins.to_csv(evid("ngl_P2_insar_vertical_mm.csv"))
    Gn.to_csv(evid("ngl_P2_gnss_up_mm.csv"))
    return out


def combined_tie(Ins: pd.DataFrame, Gn: pd.DataFrame, Iw: pd.DataFrame, Gw: pd.DataFrame,
                 t0: str, t1: str) -> dict:
    """Report-only: the leave-one-out tie over every evaluable station of both families,
    errors scored at the NGL stations only."""
    I_all, G_all = pd.concat([Ins, Iw], axis=1), pd.concat([Gn, Gw], axis=1)
    first = Lb.loso(I_all, G_all, t0, t1, tie="none")
    ev = first.loc[first["evaluable"], "station"].tolist()
    res = Lb.loso(I_all[ev], G_all[ev], t0, t1, tie="loso")
    e = res.loc[res["station"].isin(Ins.columns) & res["evaluable"], "err_mm"].to_numpy(float)
    return {"n_evaluable_ngl": int(np.isfinite(e).sum()), "n_tie_stations": len(ev),
            "rms_mm": float(np.sqrt(np.nanmean(e ** 2))) if np.isfinite(e).any() else None,
            "rows": res[res["station"].isin(Ins.columns)].to_dict("records")}


def cmd_g1_wra(_a) -> dict:
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    st = Ev.station_set(inv, "exclude")
    sc = _look(["s4b.wra_gnss.loso_preorigin"], "insar_wra_gnss_change:2021-08..2022-12",
               "second_look_score", f"{T_WRA}: the registered G1 procedure on run P2 "
               "(M1-M3) at the spent WRA units; SECOND LOOK, conditional, no evidential "
               "weight")
    run, g, v, tab = run_variant("P2", R.REF_TERMS)
    Ins, Gn = Ev.gnss_tables(run, g, v, inv, st, "exclude")
    prim = g1(Ins, Gn, C.GNSS_T0, C.GNSS_T1)
    out = {"test_id": T_WRA, "label": "SECOND LOOK on spent units (conditional)",
           "run": "P2", "window": [C.GNSS_T0, C.GNSS_T1], "primary": prim,
           "look_logged": {k: sc.get(k) for k in ("time", "purpose", "units", "data_scope")},
           "network_gaps_P2": run["network_gaps"]}
    rep = {}
    r1, g1_, v1, _ = run_variant("P2", ROUND1_TERMS)
    rep["round1_plane_reference"] = g1(*Ev.gnss_tables(r1, g1_, v1, inv, st, "exclude"),
                                       C.GNSS_T0, C.GNSS_T1)
    r2, g2, v2, _ = run_variant("P2", R.REF_TERMS, "cum.h5")
    rep["unfiltered_cum"] = g1(*Ev.gnss_tables(r2, g2, v2, inv, st, "exclude"),
                               C.GNSS_T0, C.GNSS_T1)
    from common import guard
    p3 = guard.open_product(C.DATAPREP / "gnss_clean" / "full" / "v4", "after_the_fact_analysis")
    daily = p3.read_table("wra/daily.parquet")
    G3 = {}
    for k in st:
        d = daily[(daily["station"] == k) & daily["flag"].isin([0, 1])]
        s = pd.Series(d["value"].to_numpy(float) * 1000.0,
                      index=pd.DatetimeIndex(pd.to_datetime(d["date"]))).sort_index()
        G3[k] = Lb.at_epochs(Lb.smooth_daily(s), run["dates"]).to_numpy()
    rep["s3v4_step_corrected_heights"] = g1(Ins, pd.DataFrame(G3, index=run["dates"]),
                                            C.GNSS_T0, C.GNSS_T1)
    out["report_only"] = rep
    Ins.to_csv(evid("wra_P2_insar_vertical_mm.csv"))
    Gn.to_csv(evid("wra_P2_wra_height_mm.csv"))
    return out


# ----------------------------------------------------------------------------- Y
def cmd_y(_a) -> dict:
    U = Ev.units()
    yl = sorted(set().union(*[set(U[u]["cells"]) for u in C.YUNLIN_UNITS]))
    bat = json.loads((C.RUN_DIR / "battery.json").read_text())
    E = Ev.census_kwh_per_year()
    D = Ev.densities(E)
    T = Ev.thickness_proxies()
    _, _, r, c = Ev.cell_centres_km()
    blocks = (r // 5) * 1000 + (c // 5)
    sc = _look(["s4b.seasamp.yunlin"], "insar_069A_seasonal_amplitude:2020-01..2022-12",
               "second_look_score", f"{T_Y}: the registered Y procedure on run P2 (M1-M3) at "
               "the spent Yunlin cells; SECOND LOOK, conditional, no evidential weight")
    run, g, v, _ = run_variant("P2", R.REF_TERMS)
    lo, hi = (pd.Timestamp(t) for t in C.SEAS_WINDOW)
    sel = (run["dates"] >= lo) & (run["dates"] <= hi)
    hfit = Lb.harmonic_fit(v[sel], run["dates"][sel])
    valid = run["mask"] & np.isfinite(hfit["amp"])
    amp, _, _ = Lb.to_cells(hfit["amp"], valid, g, Lb.n_active())
    np.save(evid("seasonal_amp_cells_P2.npy"), amp)
    prim = f"dens_{C.DENSITY_PRIMARY_KM:g}km"
    comps = ["placebo_ring", "thick_ske_zonal", "thick_west_of_pinchout_km"]

    def rhos(cells):
        cells = np.array([k for k in cells if np.isfinite(amp[k])], int)
        row = {"n": int(cells.size)}
        for k, f in {**D, **T}.items():
            row[k] = float(spearmanr(amp[cells], f[cells])[0]) if cells.size > 5 else np.nan
        return row, cells

    y1row, yc = rhos(yl)
    boot = Ev.block_bootstrap_rho(amp[yc], D[prim][yc], blocks[yc], C.BOOT_N, C.BOOT_SEED)
    y1 = {"cells": y1row, "rho_primary": y1row[prim],
          "boot_p05": float(np.percentile(boot, 5)), "passed": float(np.percentile(boot, 5)) > 0}
    wins, splits = 0, {}
    for k, cells in bat["splits"].items():
        row, _ = rhos(cells)
        ca = np.array([abs(row[cc]) for cc in comps], float)
        best = float(np.nanmax(ca)) if np.isfinite(ca).any() else 0.0
        win = bool(np.isfinite(row[prim]) and row[prim] > best)
        wins += win
        row.update({"best_competitor_abs": best, "win": win})
        splits[k] = row
    y2 = {"wins": int(wins), "of": len(splits), "min_wins": C.BATTERY_MIN_SPLITS,
          "passed": wins >= C.BATTERY_MIN_SPLITS, "splits": splits}
    x, _, _, _ = Ev.cell_centres_km()
    return {"test_id": T_Y, "label": "SECOND LOOK on spent units (conditional)", "run": "P2",
            "n_epochs": int(sel.sum()), "n_cells_with_amplitude": int(np.isfinite(amp).sum()),
            "Y1_all_yunlin": y1, "Y2_battery": y2, "passed": y1["passed"] and y2["passed"],
            "look_logged": {k: sc.get(k) for k in ("time", "purpose", "units", "data_scope")},
            "amplitude_yunlin_mm_p10_p50_p90": [float(np.nanpercentile(amp[yc], q))
                                                for q in (10, 50, 90)],
            "report_only_rho_amp_easting": float(spearmanr(amp[yc], x[yc])[0])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["design", "g1_ngl", "g1_wra", "y"])
    a = ap.parse_args(argv)
    out = {"design": cmd_design, "g1_ngl": cmd_g1_ngl, "g1_wra": cmd_g1_wra,
           "y": cmd_y}[a.cmd](a)
    name = "design.json" if a.cmd == "design" else f"test_{a.cmd}.json"
    if a.cmd == "design" and evid(name).exists():
        raise SystemExit(f"{evid(name)} exists (the design is written once)")
    Lb.jdump(out, evid(name))
    brief = {k: out[k] for k in ("test_id", "eligible", "passed") if k in out}
    if "primary" in out:
        brief.update({k: out["primary"].get(k) for k in ("rms_mm", "n_evaluable", "passed")})
    if a.cmd == "design":
        brief["window"] = [out["window"]["t0"], out["window"]["t1"]]
    print(json.dumps(brief, default=Lb._js))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
