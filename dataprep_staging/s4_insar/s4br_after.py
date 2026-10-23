"""WP S4b repair round 2 (S4br): after-the-fact rows, computed AFTER the registered re-test and
the second looks, each logged in the A1 look log immediately before it is computed. Report
only: never a gate, a threshold or a design choice.

    python -I s4br_after.py network    LiCSBAS network structure of P2, F2, S2 against round
                                       1's P and F (interferograms kept, epochs, gaps, masked
                                       and nullified pixels): InSAR-internal
    python -I s4br_after.py g1_full    the registered G1 window at the 18 spent WRA stations
                                       on the full-product chain (F2 alone, and F2 merged with
                                       S2), and the NGL fresh window on F2 (post-origin pairs)
    python -I s4br_after.py ramp       after the registered score and the second looks: run P2,
                                       reference (round-1 plane / M3 offset + height) x window
                                       (fresh 2020-01-31 -> 2021-03-31 / round 1's 2021-09-30
                                       -> 2022-11-30) x family (WRA 18 / NGL 4), plus the
                                       stable-strip plane change between the windows: does the
                                       reference term flip between windows? Diagnosis only.
    python -I s4br_after.py ramp_fix   correction of the 'ramp' rows and of the registered
                                       report-only combined-tie row: their WRA station set came
                                       from round 1's inventory (catalogue steps judged in round
                                       1's window), so GFES (catalogued -5.6 cm step on
                                       2020-07-10, inside the fresh span) was wrongly kept; here
                                       the registered WRA station rule is applied to the fresh
                                       window (s4b_gnss_inventory.inventory, unchanged code)
    python -I s4br_after.py derived    numbers derived from the logged 'ramp' rows only (no new
                                       data read): error correlation with northing/easting and
                                       the RMS left after a best (x, y) plane, per row; the
                                       plane's east-term change at the NGL stations

Writes ``results/dataprep/reports/S4b_evidence/repair_2026-10-18/after.json`` (merged by key).
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
import s4b_config as C  # noqa: E402
import s4b_evaluate as Ev  # noqa: E402
import s4b_lib as Lb  # noqa: E402
import s4br_config as R  # noqa: E402
import s4br_evaluate as RE  # noqa: E402
import s4br_lib as RL  # noqa: E402
from harness import prereg3  # noqa: E402
from s4b_looks import all_cells  # noqa: E402

OUT = R.EVID / "after.json"


def params13(ts: Path) -> dict:
    out = {}
    for line in (ts / "info" / "13parameters.txt").read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def mask_stats(ts: Path, cell: np.ndarray) -> dict:
    L, W = cell.shape
    m = np.fromfile(ts / "results" / "mask", dtype=np.float32).reshape(L, W)
    act = cell >= 0
    out = {"frac_active_pixels_kept": float(np.mean((m == 1)[act]))}
    f = ts / "results" / "n_nullify_rat"
    if f.exists():
        r = np.fromfile(f, dtype=np.float32).reshape(L, W)
        out["median_nullified_ratio_active"] = float(np.nanmedian(r[act]))
    return out


def cmd_network(_a) -> dict:
    prereg3.record_look(
        ["s4br.after.network.cells"], "insar_licsbas_network:2020-01..2025-12", R.WP,
        "integrity_check", prereg_id=R.PREREG_ID,
        members={"s4br.after.network.cells": {
            "description": "LiCSBAS network structure and masks of the repaired runs P2, F2, "
                           "S2 and of round 1's P and F (no displacement, GNSS or census)",
            "cells": all_cells(), "window": "2020-01..2025-12"}},
        note="S4br after the fact: network structure of the repaired runs")
    g = RL.load_geom(C.PRIMARY)
    gS = RL.load_geom(R.SECOND)
    rows = {}
    for nm, base, geo in (("P", C.RUN_DIR, g), ("F", C.RUN_DIR, g), ("P2", R.RUN_DIR, g),
                          ("F2", R.RUN_DIR, g), ("S2", R.RUN_DIR, gS)):
        ts = base / nm / "TS_GEOCml2"
        p = params13(ts)
        rows[nm] = {"n_ifg_used": int(p.get("n_ifg", 0)), "n_ifg_all": int(p.get("n_ifg_all", 0)),
                    "n_epochs": int(p.get("n_im", 0)),
                    "network_gaps": RL.network_gaps(ts), **mask_stats(ts, geo["cell"])}
    return {"network": rows}


def cmd_g1_full(_a) -> dict:
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    st = Ev.station_set(inv, "exclude")
    prereg3.record_look(
        ["s4b.wra_gnss.loso_preorigin"], "insar_wra_gnss_change:2021-08..2022-12", R.WP,
        "after_the_fact", prereg_id=R.PREREG_ID,
        note="S4br after the fact: the registered G1 window at the spent WRA stations on the "
             "full-product chain (F2 alone; F2 merged with S2); report only")
    out = {}
    run, g, v, _ = RL.processed("F2")
    Ins, Gn = Ev.gnss_tables(run, g, v, inv, st, "exclude")
    out["wra_F2"] = RE.g1(Ins, Gn, C.GNSS_T0, C.GNSS_T1)
    runS, gS, vS, _ = RL.processed("S2")
    val, src, dates, tstar = RL.tie_and_merge(v, run["dates"], run["mask"], vS, runS["dates"],
                                              runS["mask"])
    prim = np.isin(src, (1, 2, 3))
    merged = {"cum": val, "dates": dates, "mask": run["mask"] | runS["mask"]}
    vm = np.where(prim, val, np.nan)
    Ins2, Gn2 = Ev.gnss_tables(merged, g, vm, inv, st, "exclude")
    out["wra_F2_S2_merged"] = RE.g1(Ins2, Gn2, C.GNSS_T0, C.GNSS_T1)
    out["wra_F2_S2_merged"]["tie_epoch"] = tstar
    # the fresh NGL window on F2 (F2 adds 2023+ interferograms: after the fact, U7)
    des = json.loads((R.EVID / "design.json").read_text())
    elig = {k: des["stations"][k] for k in des["eligible"]}
    prereg3.record_look(
        ["s4b.r2.ngl_gnss.loso_fresh"], f"insar_ngl_gnss_change:{des['window']['scope']}",
        R.WP, "after_the_fact", prereg_id=R.PREREG_ID,
        note="S4br after the fact: the fresh NGL window on the full-product run F2 (uses "
             "post-origin interferograms); report only, after the registered score")
    I3, G3 = RE.ngl_tables(run, v, elig)
    out["ngl_fresh_window_F2"] = RE.g1(I3, G3, des["window"]["t0"], des["window"]["t1"])
    return out


def cmd_ramp(_a) -> dict:
    des = json.loads((R.EVID / "design.json").read_text())
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    st = Ev.station_set(inv, "exclude")
    elig = {k: des["stations"][k] for k in des["eligible"]}
    wins = {"fresh": (des["window"]["t0"], des["window"]["t1"], des["window"]["scope"]),
            "round1": (C.GNSS_T0, C.GNSS_T1, "2021-08..2022-12")}
    prereg3.record_look(
        ["s4b.wra_gnss.loso_preorigin", "s4b.r2.wra_gnss.tie_fresh_window",
         "s4b.r2.ngl_gnss.loso_fresh", "s4b.r2.ngl_gnss.round1_window"],
        "insar_gnss_ramp_diagnosis:2020-01..2022-12", R.WP, "diagnose", prereg_id=R.PREREG_ID,
        note="S4br after the registered score and the declared second looks: reference x "
             "window x family table on run P2 (spent units only); diagnosis, never a design "
             "choice")
    out = {}
    for ref_nm, terms in (("plane", RE.ROUND1_TERMS), ("offset_height", R.REF_TERMS)):
        run, g, v, tab = RE.run_variant("P2", terms)
        Iw, Gw = Ev.gnss_tables(run, g, v, inv, st, "exclude")
        In, Gn = RE.ngl_tables(run, v, elig)
        for w_nm, (t0, t1, _sc) in wins.items():
            for fam, (I_, G_) in (("wra", (Iw, Gw)), ("ngl", (In, Gn))):
                r = RE.g1(I_, G_, t0, t1)
                out[f"{fam}_{w_nm}_{ref_nm}"] = {"rms_mm": r.get("rms_mm"),
                                                 "n": r["n_evaluable"],
                                                 "err_mm": {x["station"]: x["err_mm"]
                                                            for x in r["rows"]}}
        if ref_nm == "plane":
            t = tab.set_index(pd.to_datetime(tab["date"]))
            ch = {}
            for w_nm, (t0, t1, _sc) in wins.items():
                h = pd.Timedelta(days=C.GNSS_HALF_WINDOW_D)
                a0 = (t.index >= pd.Timestamp(t0) - h) & (t.index <= pd.Timestamp(t0) + h)
                a1 = (t.index >= pd.Timestamp(t1) - h) & (t.index <= pd.Timestamp(t1) + h)
                ch[w_nm] = {c: float(t.loc[a1, f"coef_{c}"].mean() - t.loc[a0, f"coef_{c}"]
                                     .mean()) for c in ("east_km", "north_km", "hgt_km")}
            out["plane_coef_change_w1_minus_w0_mm_per_km_los"] = ch
    return {"ramp_table": out}


def cmd_ramp_fix(_a) -> dict:
    import s4b_gnss_inventory as GI
    des = json.loads((R.EVID / "design.json").read_text())
    t0, t1, scope = des["window"]["t0"], des["window"]["t1"], des["window"]["scope"]
    inv_f = GI.inventory(t0, t1)
    st_f = Ev.station_set(inv_f, "exclude")
    u = "s4br.after.wra_gnss.fresh_window_rule"
    prereg3.record_look(
        [u], f"insar_wra_gnss_change:{scope}", R.WP, "diagnose", prereg_id=R.PREREG_ID,
        members={u: {"description": "WRA stations selected by the registered G1 station rule "
                                    "applied to the fresh window (availability and catalogue "
                                    "steps judged in that window); run P2",
                     "ids": [f"wra_gnss:{k}" for k in st_f], "window": scope}},
        note="S4br after the fact: corrected WRA station set for the fresh window (GFES's "
             "catalogued 2020-07-10 step lies inside it)")
    elig = {k: des["stations"][k] for k in des["eligible"]}
    out = {"wra_station_set_fresh_window": st_f,
           "excluded_by_catalogue": sorted(k for k in inv_f.index
                                           if len(inv_f.loc[k].get("catalogue_steps_in_span")
                                                  or []) > 0)}
    for ref_nm, terms in (("plane", RE.ROUND1_TERMS), ("offset_height", R.REF_TERMS)):
        run, g, v, _ = RE.run_variant("P2", terms)
        Iw, Gw = Ev.gnss_tables(run, g, v, inv_f, st_f, "exclude")
        r = RE.g1(Iw, Gw, t0, t1)
        out[f"wra_fresh_{ref_nm}"] = {"rms_mm": r.get("rms_mm"), "n": r["n_evaluable"],
                                      "err_mm": {x["station"]: x["err_mm"] for x in r["rows"]}}
        if ref_nm == "offset_height":
            In, Gn = RE.ngl_tables(run, v, elig)
            ct = RE.combined_tie(In, Gn, Iw, Gw, t0, t1)
            out["combined_wra_ngl_tie_corrected"] = {
                "rms_mm": ct["rms_mm"], "n_evaluable_ngl": ct["n_evaluable_ngl"],
                "n_tie_stations": ct["n_tie_stations"],
                "err_mm": {x["station"]: x["err_mm"] for x in ct["rows"]}}
    return {"ramp_fix": out}


def cmd_derived(_a) -> dict:
    cur = json.loads(OUT.read_text())
    rt = cur["ramp_table"]
    gm = json.loads((R.EVID / "diag_geometry.json").read_text())
    des = json.loads((R.EVID / "design.json").read_text())
    g = RL.load_geom(C.PRIMARY)
    stb = g["stable_geom"]
    xc, yc = float(np.mean(g["x"][stb])), float(np.mean(g["y"][stb]))
    pos = {k: ((float(g["x"][v["ml_i"], v["ml_j"]]) - xc) / 1000.0,
               (float(g["y"][v["ml_i"], v["ml_j"]]) - yc) / 1000.0,
               float(g["U"][v["ml_i"], v["ml_j"]]))
           for k, v in des["stations"].items() if k in des["eligible"]}
    for k, xr in gm["station_x_rel_km"].items():
        pos.setdefault(k, (xr, gm["station_y_rel_km"][k], 0.83))
    out = {}
    for key, v in rt.items():
        if "err_mm" not in v:
            continue
        e = pd.Series({k: x for k, x in v["err_mm"].items() if x is not None})
        e = e[[k for k in e.index if k in pos]]
        if len(e) < 4:
            continue
        X = np.column_stack([np.ones(len(e)), [pos[k][0] for k in e.index],
                             [pos[k][1] for k in e.index]])
        res = e.to_numpy() - X @ np.linalg.lstsq(X, e.to_numpy(), rcond=None)[0]
        out[key] = {"n": int(len(e)),
                    "corr_northing": float(np.corrcoef(e, X[:, 2])[0, 1]),
                    "corr_easting": float(np.corrcoef(e, X[:, 1])[0, 1]),
                    "rms_after_xy_plane_mm": float(np.sqrt(np.mean(res ** 2)))}
    ch = rt["plane_coef_change_w1_minus_w0_mm_per_km_los"]["fresh"]["east_km"]
    out["plane_east_term_fresh_window_vertical_mm"] = {
        k: float(-ch * pos[k][0] / pos[k][2]) for k in des["eligible"] if k in pos}
    return {"derived": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["network", "g1_full", "ramp", "ramp_fix", "derived"])
    a = ap.parse_args(argv)
    res = {"network": cmd_network, "g1_full": cmd_g1_full, "ramp": cmd_ramp,
           "ramp_fix": cmd_ramp_fix, "derived": cmd_derived}[a.cmd](a)
    cur = json.loads(OUT.read_text()) if OUT.exists() else {}
    cur.update(res)
    Lb.jdump(cur, OUT)
    print(json.dumps({k: (v.get("rms_mm") if isinstance(v, dict) and "rms_mm" in v else
                          "ok") for k, v in res.items()}, default=Lb._js))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
