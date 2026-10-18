"""WP S4b: the registered tests of ``S4b.insar_acceptance.v1`` and their report-only rows.

    python -I s4b_evaluate.py gnss        G1 (gate): WRA leave-one-station-out 14-month change
    python -I s4b_evaluate.py coherence   C1: Changhua zone coherence coverage, summer pairs
    python -I s4b_evaluate.py seasonal    Y1/Y2: seasonal amplitude vs census density, Yunlin

Every registered test calls ``prereg3.assert_can_score`` immediately before its score is
computed (A1 rule), on the pre-origin gate run P only (U7). Outputs go to
``results/dataprep/reports/S4b_evidence/``. Hydro env; single thread (cpu_slot).
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
import s4b_lib as Lb  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

T_G1 = "S4b.G1_gnss_loso_14mo_preorigin"
T_C1 = "S4b.C1_changhua_summer_coherence_preorigin"
T_Y = "S4b.Y_seasonal_amplitude_yunlin_preorigin"


def _score_gate(test_id: str) -> None:
    from harness import prereg3
    prereg3.assert_can_score(C.PREREG_ID, C.WP, test_id=test_id)


# ----------------------------------------------------------------------------- processing
def processed(name: str, terms=C.REF_TERMS, u_const=None):
    run = Lb.load_run(name)
    g = Lb.load_geom(run["frame"])
    los, tab = Lb.reference(run, g, terms)
    return run, g, Lb.vertical(los, g, u_const), tab


def station_set(inv: pd.DataFrame, step_rule: str) -> list[str]:
    ok = inv["on_fan"].astype(bool) & inv["gnss_ok"].astype(bool)
    if step_rule == "exclude":
        ok &= inv["catalogue_steps_in_span"].apply(len) == 0
    return sorted(inv.index[ok])


def gnss_tables(run, g, v, inv, stations, step_rule) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = Lb.wra_files()
    Ins, Gn = {}, {}
    for st in stations:
        Ins[st] = Lb.station_series(v, run["mask"], int(inv.loc[st, "ml_i"]),
                                  int(inv.loc[st, "ml_j"]))
        s = Lb.read_wra(files[st]) * 1000.0
        if step_rule == "correct":
            for d, amp in C.GNSS_STEP_CATALOGUE.get(st, []):
                s.loc[pd.Timestamp(d):] -= amp * 1000.0
        Gn[st] = Lb.at_epochs(Lb.smooth_daily(s), run["dates"]).to_numpy()
    return (pd.DataFrame(Ins, index=run["dates"]), pd.DataFrame(Gn, index=run["dates"]))


def loso_eval(Ins, Gn, t0, t1, tie="loso") -> dict:
    """Two passes: stations evaluable on their own (enough epochs with both values in each
    window), then the leave-one-out tie among the evaluable stations only."""
    first = Lb.loso(Ins, Gn, t0, t1, tie="none")
    ev = first.loc[first["evaluable"], "station"].tolist()
    if len(ev) < 3:
        return {"n_evaluable": len(ev), "rms_mm": None, "rows": first.to_dict("records")}
    res = Lb.loso(Ins[ev], Gn[ev], t0, t1, tie=tie)
    e = res.loc[res["evaluable"], "err_mm"].to_numpy(float)
    return {"n_evaluable": int(np.isfinite(e).sum()), "stations": ev,
            "rms_mm": float(np.sqrt(np.nanmean(e ** 2))),
            "mean_mm": float(np.nanmean(e)), "median_abs_mm": float(np.nanmedian(np.abs(e))),
            "max_abs_mm": float(np.nanmax(np.abs(e))),
            "not_evaluable": sorted(set(Ins.columns) - set(ev)),
            "rows": res.to_dict("records")}


def cmd_gnss(a) -> dict:
    inv = pd.read_json(C.RUN_DIR / "gnss_inventory.json").set_index("station")
    st_primary = station_set(inv, "exclude")
    _score_gate(T_G1)
    run, g, v, tab = processed("P")
    Ins, Gn = gnss_tables(run, g, v, inv, st_primary, "exclude")
    prim = loso_eval(Ins, Gn, C.GNSS_T0, C.GNSS_T1, "loso")
    out = {"test_id": T_G1, "run": "P", "window": [C.GNSS_T0, C.GNSS_T1],
           "threshold_rms_mm": C.GNSS_LOSO_RMS_MAX_MM, "primary": prim,
           "min_stations": C.GNSS_MIN_STATIONS,
           "passed": prim["rms_mm"] is not None and prim["n_evaluable"] >= C.GNSS_MIN_STATIONS
           and prim["rms_mm"] <= C.GNSS_LOSO_RMS_MAX_MM,
           "reference_table_P": tab.to_dict("records")}
    # registered report-only rows (same run, same window)
    rep = {}
    rep["tie_none"] = loso_eval(Ins, Gn, C.GNSS_T0, C.GNSS_T1, "none")
    st_corr = station_set(inv, "correct")
    Ins2, Gn2 = gnss_tables(run, g, v, inv, st_corr, "correct")
    rep["steps_corrected"] = loso_eval(Ins2, Gn2, C.GNSS_T0, C.GNSS_T1, "loso")
    for nm, terms in (("ref_offset_hgt", ("offset", "hgt_km")), ("ref_offset", ("offset",))):
        r2, g2, v2, _ = processed("P", terms)
        Ins3, Gn3 = gnss_tables(r2, g2, v2, inv, st_primary, "exclude")
        rep[nm] = loso_eval(Ins3, Gn3, C.GNSS_T0, C.GNSS_T1, "loso")
    r4, g4, v4, _ = processed("P", C.REF_TERMS, C.U_CONSTANT_PLAN)
    Ins4, Gn4 = gnss_tables(r4, g4, v4, inv, st_primary, "exclude")
    rep["u_constant_0.833"] = loso_eval(Ins4, Gn4, C.GNSS_T0, C.GNSS_T1, "loso")
    out["report_only"] = rep
    Ins.to_csv(C.EVID / "gnss_P_insar_vertical_mm.csv")
    Gn.to_csv(C.EVID / "gnss_P_wra_height_mm.csv")
    return out


# ----------------------------------------------------------------------------- coherence
def units() -> dict:
    u = json.loads(C.UNITS_V3.read_text())
    return {x["id"]: x for x in u["units"]}


def cmd_coherence(a) -> dict:
    U = units()
    _score_gate(T_C1)
    z = np.load(C.RUN_DIR / "coh_P_summer_preorigin.npz", allow_pickle=False)
    med = z["cell_median_coh"]
    pairs = [str(p) for p in z["pairs"]]
    years = np.array([int(p[:4]) for p in pairs])
    res = {"test_id": T_C1, "n_pairs": len(pairs), "pairs": pairs,
           "rule": {"coh_min": C.COH_MIN, "zone_min_frac": C.COH_ZONE_MIN_FRAC,
                    "statistic": "median over summer pairs of the zone's coherent-cell "
                                 "fraction; denominator = every active cell of the zone"},
           "zones": {}}
    for zid in C.CH_ZONES:
        cells = np.array(U[zid]["cells"], int)
        coh = med[:, cells]
        frac = np.where(np.isfinite(coh), coh >= C.COH_MIN, False).mean(1)
        cover = np.isfinite(coh).mean(1)
        stat = float(np.median(frac))
        res["zones"][zid] = {
            "n_cells": int(cells.size), "median_frac_coherent": stat,
            "covered": stat >= C.COH_ZONE_MIN_FRAC,
            "frac_coherent_p10_p90": [float(np.percentile(frac, 10)),
                                      float(np.percentile(frac, 90))],
            "median_frac_cells_with_data": float(np.median(cover)),
            "per_year_median": {int(y): float(np.median(frac[years == y]))
                                for y in sorted(set(years))},
            "cell_median_coherence_median": float(np.nanmedian(coh))}
    res["passed_all_zones"] = all(v["covered"] for v in res["zones"].values())
    return res


# ----------------------------------------------------------------------------- seasonal
def census_kwh_per_year() -> np.ndarray:
    sys.path.insert(0, str(C.STAGING))
    from common import guard
    p = guard.open_product(C.DATAPREP / "pump_electricity" / "preorigin" / "v1", "validator")
    z = np.load(p.path("cells_total.npz"), allow_pickle=False)
    dates = pd.to_datetime([str(d) for d in z["dates"]], format="%Y-%m")
    sel = (dates.year >= C.CENSUS_YEARS[0]) & (dates.year <= C.CENSUS_YEARS[1])
    return z["value_all"][:, sel].sum(1) / (sel.sum() / 12.0)


def cell_centres_km() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask, dx, x0, y0 = Lb.twin_grid()
    r, c = np.nonzero(mask)
    return (x0 + (c + 0.5) * dx) / 1000.0, (y0 + (r + 0.5) * dx) / 1000.0, r, c


def densities(E: np.ndarray) -> dict:
    x, y, _, _ = cell_centres_km()
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    out = {}
    for r in C.DENSITY_SCALES_KM:
        w = d <= r + 1e-9
        out[f"dens_{r:g}km"] = (w * E[None]).sum(1) / w.sum(1)
    lo, hi = C.PLACEBO_RING_KM
    w = (d > lo) & (d <= hi)
    out["placebo_ring"] = (w * E[None]).sum(1) / np.maximum(w.sum(1), 1)
    return out


def thickness_proxies() -> dict:
    x, _, _, _ = cell_centres_km()
    vep = json.loads((C.REPO / "results" / "twin_runs" / "stage3_datum_gate" / "coupled_leveling"
                      / "vep_zonal_leveling.json").read_text())
    ske = np.exp([z["log_ske"] for z in vep["zonal"]])      # zone ids 0 prox, 1 mid, 2 dist
    zone = np.where(x >= 205.0, 0, np.where(x >= 182.0, 1, 2))
    return {"thick_ske_zonal": ske[zone], "thick_west_of_pinchout_km": np.maximum(0.0, 205.0 - x)}


def block_bootstrap_rho(a: np.ndarray, b: np.ndarray, blocks: np.ndarray, n: int, seed: int
                        ) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ub = np.unique(blocks)
    idx = {k: np.flatnonzero(blocks == k) for k in ub}
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(ub, size=ub.size, replace=True)
        s = np.concatenate([idx[k] for k in pick])
        out[i] = spearmanr(a[s], b[s])[0]
    return out


def amplitude_cells(name: str = "P") -> tuple[np.ndarray, dict]:
    run, g, v, tab = processed(name)
    lo, hi = (pd.Timestamp(t) for t in C.SEAS_WINDOW)
    sel = (run["dates"] >= lo) & (run["dates"] <= hi)
    h = Lb.harmonic_fit(v[sel], run["dates"][sel])
    valid = run["mask"] & np.isfinite(h["amp"])
    A = Lb.n_active()
    amp_c, _, n_c = Lb.to_cells(h["amp"], valid, g, A)
    return amp_c, {"pixel": h, "valid": valid, "n_epochs": int(sel.sum()), "n_cells_valid":
                   int(np.isfinite(amp_c).sum())}


def cmd_seasonal(a) -> dict:
    U = units()
    yl = sorted(set().union(*[set(U[u]["cells"]) for u in C.YUNLIN_UNITS]))
    bat = json.loads((C.RUN_DIR / "battery.json").read_text())
    E = census_kwh_per_year()
    D = densities(E)
    T = thickness_proxies()
    _, _, r, c = cell_centres_km()
    blocks = (r // 5) * 1000 + (c // 5)
    _score_gate(T_Y)
    amp, info = amplitude_cells("P")
    np.save(C.EVID / "seasonal_amp_cells_P.npy", amp)
    prim = f"dens_{C.DENSITY_PRIMARY_KM:g}km"
    comps = ["placebo_ring", "thick_ske_zonal", "thick_west_of_pinchout_km"]

    def rhos(cells):
        cells = np.array([k for k in cells if np.isfinite(amp[k])], int)
        row = {"n": int(cells.size)}
        for k, f in {**D, **T}.items():
            row[k] = float(spearmanr(amp[cells], f[cells])[0]) if cells.size > 5 else np.nan
        return row, cells

    y1row, yc = rhos(yl)
    boot = block_bootstrap_rho(amp[yc], D[prim][yc], blocks[yc], C.BOOT_N, C.BOOT_SEED)
    y1 = {"cells": y1row, "rho_primary": y1row[prim],
          "boot_p05": float(np.percentile(boot, 5)), "boot_p95": float(np.percentile(boot, 95)),
          "n_blocks": int(np.unique(blocks[yc]).size),
          "passed": float(np.percentile(boot, 5)) > 0}
    splits = {}
    wins = 0
    for k, cells in bat["splits"].items():
        row, _ = rhos(cells)
        # an undefined competitor correlation (a proxy constant within the subset) counts 0
        ca = np.array([abs(row[cc]) for cc in comps], float)
        best_comp = float(np.nanmax(ca)) if np.isfinite(ca).any() else 0.0
        win = bool(np.isfinite(row[prim]) and row[prim] > best_comp)
        wins += win
        row.update({"best_competitor_abs": best_comp, "win": win})
        # report-only: the other scales against the same competitors
        row["win_2km"] = bool(row["dens_2km"] > best_comp)
        row["win_10km"] = bool(row["dens_10km"] > best_comp)
        splits[k] = row
    y2 = {"wins": int(wins), "of": len(splits), "min_wins": C.BATTERY_MIN_SPLITS,
          "passed": wins >= C.BATTERY_MIN_SPLITS, "splits": splits}
    return {"test_id": T_Y, "run": "P", "n_epochs": info["n_epochs"],
            "n_cells_with_amplitude": info["n_cells_valid"],
            "Y1_all_yunlin": y1, "Y2_battery": y2, "passed": y1["passed"] and y2["passed"],
            "amplitude_yunlin_mm_p10_p50_p90": [float(np.nanpercentile(amp[yc], q))
                                                for q in (10, 50, 90)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gnss", "coherence", "seasonal"])
    a = ap.parse_args(argv)
    C.EVID.mkdir(parents=True, exist_ok=True)
    out = {"gnss": cmd_gnss, "coherence": cmd_coherence, "seasonal": cmd_seasonal}[a.cmd](a)
    Lb.jdump(out, C.EVID / f"test_{a.cmd}.json")
    brief = {k: v for k, v in out.items() if k in ("test_id", "passed", "passed_all_zones")}
    print(json.dumps(brief))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
