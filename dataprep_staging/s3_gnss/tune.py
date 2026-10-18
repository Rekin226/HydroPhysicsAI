"""WP S3 design on PRE-ORIGIN data only (every series cut at 2022-12-31; U3/U7).

    dataprep_staging/tools/cpu_slot.sh python dataprep_staging/s3_gnss/tune.py OUT_DIR

Writes ``OUT_DIR/tune.json`` (and the trial tables) with:

1. **Detector hyperparameters.** Grid ``delta_min x z_min``; every other parameter at its
   ``model.Params`` default, except ``local_half`` = 30 (round r2: the r1 stage-2 choice).
   For each configuration: injected +-2 and +-3 cm steps (6 repetitions, seed 101) in the WRA (2020-01..2022-12) and NGL (record start..2022-12)
   families, and the false-detection rate = statistically detected steps per station-year on
   the un-injected off-fan reference stations (which should have no unexplained steps).
   Selection rule (fixed before the grid ran): maximise the pooled 2 cm recovery subject to
   a false-detection rate <= 0.05 per station-year; within 0.01 of the best, prefer the larger
   ``z_min``, then the larger ``delta_min``. Stage 2: with that detector, the local step-size
   half-window from ``GRID_HALF`` days (seed 202), max 2 cm recovery, ties to the longer.
2. **Off-fan reference stations**: the rule in ``pipeline.select_reference``.
3. **Common-mode design evidence for WRA**: correlations, on 2020-22, of the WRA network
   median residual split at several running-mean windows with the NGL off-fan and NGL
   fan-station medians; the high-pass window is the largest whose high-frequency part has
   |corr| <= 0.05 with both (the result fixed ``Params.hp_window``).

Round r2 (2026-10-12 afternoon): the scan no longer applies the robust-fit clip mask (a
step near a record end was clipped out of the scan; ``model.process_station``); grids
widened at the edges r1 chose. r1 output and code are kept in
``results/dataprep/reports/S3_evidence/tune/``.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import injection as INJ  # noqa: E402
import io_gnss as io  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402

END = P.ORIGIN_END
GRID_DELTA = [0.008, 0.010, 0.012, 0.015, 0.018]   # r2: 0.018 added (r1 chose the edge)
GRID_Z = [3.5, 4.0, 5.0, 6.0]   # r2: 8.0 dropped (never competitive in r1), 3.5 added
GRID_HALF = [20, 30, 45, 60]   # r2: 90 dropped (worst in r1), 20 added
FP_MAX = 0.05


def false_rate(fam: P.Family, stations: list[str]) -> tuple[float, int, float]:
    n_det, years = 0, 0.0
    for k in stations:
        r = fam.results[k]
        n_det += sum(1 for s in r.steps if s.evidence == "detected")
        years += r.ok.sum() / 365.25
    return (n_det / years if years else np.nan), n_det, years


def cm_band_evidence(wra: P.Family, ngl: P.Family, ref: list[str]) -> dict:
    Rw = wra.resid_frame(wra.info["members"])
    w = Rw.median(axis=1)
    Rn = ngl.resid_frame()
    off = Rn[ref].median(axis=1)
    fan_st = [k for k in ["HUWE", "DNAN", "FUNY", "PUS2", "LUKN"] if k in Rn]
    fan = Rn[fan_st].median(axis=1)
    j = pd.concat([w.rename("w"), off.rename("off"), fan.rename("fan")], axis=1).dropna()
    j = j.loc["2020-01-01":str(END.date())]
    rows = []
    for W in (5, 9, 15, 21, 31, 45, 61, 91, 181):
        lp = j.rolling(W, center=True, min_periods=max(3, W // 3)).mean()
        hp = j - lp
        rows.append({"window_d": W,
                     "hp_corr_off": float(hp.w.corr(hp.off)),
                     "hp_corr_fan": float(hp.w.corr(hp.fan)),
                     "hp_sd_mm": float(1000 * hp.w.std()),
                     "lp_corr_off": float(lp.w.corr(lp.off)),
                     "lp_corr_fan": float(lp.w.corr(lp.fan)),
                     "lp_sd_mm": float(1000 * lp.w.std())})
    ok = [r["window_d"] for r in rows
          if abs(r["hp_corr_off"]) <= 0.05 and abs(r["hp_corr_fan"]) <= 0.05]
    # scatter reduction on 2020-22 for the candidate WRA common modes
    def red(cm: pd.Series) -> float:
        sds0, sds1 = [], []
        for k in wra.info["members"]:
            a = pd.concat([Rw[k], cm], axis=1).dropna().loc["2020-01-01":str(END.date())]
            if len(a) < 300:
                continue
            sds0.append(a.iloc[:, 0].std())
            sds1.append((a.iloc[:, 0] - a.iloc[:, 1]).std())
        return float(1000 * np.median(sds0)), float(1000 * np.median(sds1))
    hpw = max(ok) if ok else None
    cand = {"none": None,
            "wra_onfan_fullband (pilot)": w,
            "ngl_offfan": off}
    if hpw:
        cand[f"wra_hf_{hpw}d"] = w - w.rolling(hpw, center=True,
                                               min_periods=max(3, hpw // 3)).mean()
    scat = {}
    for name, cm in cand.items():
        if cm is None:
            continue
        s0, s1 = red(cm)
        scat[name] = {"median_station_sd_mm_before": round(s0, 2),
                      "median_station_sd_mm_after": round(s1, 2)}
    return {"band_table": rows, "hp_window_rule": "largest window with |hp corr| <= 0.05 "
            "with both the NGL off-fan and NGL fan-station medians",
            "hp_window_selected": hpw, "fan_stations": fan_st, "reference": ref,
            "scatter_2020_22": scat, "n_days": int(len(j))}


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    grid_npz = np.load(io.REPO / "results/twin_forward/datum_gate_innov.npz")
    # r2: stage 1 runs with the r1 stage-2 choice local_half = 30 (pre-origin evidence)
    p0 = replace(M.Params(), local_half=30)
    wser = P.wra_series(END)
    nser, _ = P.ngl_series(END)
    n1 = P.run_ngl(p0, ref=P.NGL_REF_CANDIDATES, window_end=END, n_pass=1, series=nser)
    ref, reftab = P.select_reference(n1, grid_npz)
    print(f"reference {ref} ({time.time() - t0:.0f}s)", flush=True)
    results = []
    trials = []
    for dmin in GRID_DELTA:
        for zmin in GRID_Z:
            p = replace(p0, delta_min=dmin, z_min=zmin)
            bw = P.run_wra(p, window_end=END, series=wser)
            bn = P.run_ngl(p, ref=ref, window_end=END, series=nser)
            fr, ndet, yrs = false_rate(bn, ref)
            tw = INJ.run_trials("wra", p, bw, END, [0.02, 0.03], 6, 101, series=wser)
            tn = INJ.run_trials("ngl", p, bn, END, [0.02, 0.03], 6, 101, ref=ref,
                                series=nser)
            tt = pd.concat([tw, tn])
            tt["delta_min"], tt["z_min"] = dmin, zmin
            trials.append(tt)
            two = tt[tt.abs_size_m.round(3) == 0.02]
            res = {"delta_min": dmin, "z_min": zmin,
                   "recovered_2cm": float(two.recovered.mean()),
                   "recovered_2cm_wra": float(two[two.family == "wra"].recovered.mean()),
                   "recovered_2cm_ngl": float(two[two.family == "ngl"].recovered.mean()),
                   "recovered_3cm": float(tt[tt.abs_size_m.round(3) == 0.03].recovered.mean()),
                   "false_per_station_year_offfan": fr, "false_n": ndet,
                   "offfan_station_years": yrs,
                   "wra_detected_steps_base": int(sum(
                       sum(1 for s in r.steps if s.evidence == "detected")
                       for r in bw.results.values())),
                   "extra_steps_per_trial": float(tt.extra_steps.mean())}
            results.append(res)
            print(json.dumps(res), f"({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(results)
    feas = R[R.false_per_station_year_offfan <= FP_MAX]
    if len(feas):
        best = feas.recovered_2cm.max()
        near = feas[feas.recovered_2cm >= best - 0.01].sort_values(
            ["z_min", "delta_min"], ascending=False)
        chosen = near.iloc[0].to_dict()
    else:
        chosen = None
    # stage 2: local step-size window with the chosen detector (detection unaffected)
    pc = replace(p0, delta_min=chosen["delta_min"], z_min=chosen["z_min"]) if chosen else p0
    half_rows = []
    for half in GRID_HALF:
        ph = replace(pc, local_half=half)
        bw = P.run_wra(ph, window_end=END, series=wser)
        bn = P.run_ngl(ph, ref=ref, window_end=END, series=nser)
        tw = INJ.run_trials("wra", ph, bw, END, [0.02, 0.03], 6, 202, series=wser)
        tn = INJ.run_trials("ngl", ph, bn, END, [0.02, 0.03], 6, 202, ref=ref, series=nser)
        tt = pd.concat([tw, tn])
        tt["local_half"] = half
        trials.append(tt)
        two = tt[tt.abs_size_m.round(3) == 0.02]
        hr = {"local_half": half, "recovered_2cm": float(two.recovered.mean()),
              "recovered_3cm": float(tt[tt.abs_size_m.round(3) == 0.03].recovered.mean()),
              "size_err_mm_p90_found": float(np.nanpercentile(
                  1000 * tt[tt.found].size_err_m.abs(), 90))}
        half_rows.append(hr)
        print(json.dumps(hr), f"({time.time() - t0:.0f}s)", flush=True)
    H = pd.DataFrame(half_rows)
    best_h = H.recovered_2cm.max()
    chosen_half = int(H[H.recovered_2cm >= best_h - 0.01].sort_values(
        "local_half", ascending=False).iloc[0].local_half)
    pc = replace(pc, local_half=chosen_half)
    # common-mode evidence with the chosen detector
    bw = P.run_wra(pc, window_end=END, series=wser)
    bn = P.run_ngl(pc, ref=ref, window_end=END, series=nser)
    cmev = cm_band_evidence(bw, bn, ref)
    pd.concat(trials).to_csv(od / "tune_trials.csv", index=False)
    rep = {"window_end": str(END.date()), "grid": results, "selection_rule":
           f"max pooled 2 cm recovery s.t. off-fan false detections <= {FP_MAX} per "
           "station-year; within 0.01 prefer larger z_min, then larger delta_min",
           "chosen": chosen, "local_half_grid": half_rows,
           "local_half_rule": "max pooled 2 cm recovery; within 0.01 prefer the longer window",
           "chosen_local_half": chosen_half, "chosen_params": pc.as_dict(),
           "reference_table": reftab.to_dict(orient="records"),
           "reference": ref, "cm_evidence": cmev, "params_default": p0.as_dict(),
           "seconds": round(time.time() - t0, 1)}
    (od / "tune.json").write_text(json.dumps(rep, indent=1, default=str))
    print("chosen", chosen)
    print(json.dumps(cmev, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
