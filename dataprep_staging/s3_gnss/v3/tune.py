"""WP S3 v3 design (repair round 3 S3r, 2026-10-18), design run 3, on PRE-ORIGIN data only
(every series cut at 2022-12-31; U3/U7), with NEW design-time seeds 525 and 626.

    cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/tune.py reference OUT_DIR
    cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/tune.py stage1 --win W OUT_DIR
    cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/tune.py select1 OUT_DIR
    cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/tune.py stage2 --size ORDER:HALF OUT_DIR
    cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/tune.py select2 OUT_DIR

Design run 1 (quadratic size model on +-60 d, seed 505) was stopped after design-time evidence
showed its premise was wrong (repair report Part 2; ``S3_evidence/repair_2026-10-18/
tune_v3_aborted/``). Design run 2 (seeds 515/616, ``tune_v3b/``) completed with the rules
below and showed one more defect on its design units: the station-wide empirical null
over-detects in the noisy years of a non-stationary station and under-detects in its quiet
ones (ERLN, 0 of 12 steps found). Run 3 differs from run 2 only by the model's time-local null
scale (R2b, ``model.NULL_HALF`` = 365 d, fixed a priori) and by its seeds. Rules of runs 2 and 3,
fixed before run 2 ran:

* Every configuration uses the v3 model and pipeline (``model.py``: empirical-null z,
  gap-aware sharpness, |D| ranking, joint box fit with a quadratic trend, level-difference step
  size; ``pipeline.py``: 5-nearest-station WRA detection common mode). Parameters not listed
  keep their ``model.Params`` defaults.
* **Stage 1 (detector; seed 515 in run 2, 525 in run 3).** Grid: scan half-window ``win`` in {20, 30} days x
  ``z_min`` in {3.5, 4.0, 4.5, 5.0} (empirical-null units) x ``delta_min`` in {10, 15} mm, with
  the step size fixed at the level difference on +-10 days (``size_order`` 0, ``local_half``
  10). Per configuration: injected +-2 and +-3 cm steps, 6 repetitions, in the WRA
  (2020-01..2022-12) and NGL (record start..2022-12) families; the false-detection rate =
  detected steps per station-year on the un-injected off-fan reference stations. Rule (as in
  tune r3): maximise the pooled (both families, 2 and 3 cm) recovery subject to false
  detections <= 0.05 per station-year; within 0.01 of the best, prefer the larger ``z_min``,
  then the larger ``delta_min``, then the smaller ``win``.
* **Stage 2 (step-size model; seed 616 in run 2, 626 in run 3).** With the stage-1 detector: (size_order,
  local_half) in {(0, 7), (0, 10), (0, 15), (0, 20)} and the v2 model (1, 30); injected +-2,
  +-3, +-5 and +-10 cm steps, 6 repetitions, both families. Rule: maximise the pooled recovery
  over all sizes and both families; within 0.01 of the best, prefer the v2 model (1, 30), then
  the longer window.
* **Reference stations**: the unchanged pre-origin rule (``pipeline.select_reference``) on a
  pass-1 run.
* **wra_hf window**: the unchanged r3 rule (largest window whose high-frequency part has
  |corr| <= 0.05 with both the NGL off-fan and NGL fan-station medians on 2020-22; declared
  fallback: the smallest max |corr|), evaluated with the final detector.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import injection as INJ  # noqa: E402
import io_gnss as io  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402

END = P.ORIGIN_END
GRID_WIN = [20, 30]
GRID_Z = [3.5, 4.0, 4.5, 5.0]
GRID_DELTA = [0.010, 0.015]
#: stage-2 candidates (size_order, local_half), in tie-break preference order
GRID_SIZE = [(1, 30), (0, 20), (0, 15), (0, 10), (0, 7)]
FP_MAX = 0.05
SEED1, SEED2 = 525, 626     # design run 3 (run 2: 515, 616)
P0 = replace(M.Params(), local_half=10, trend_order=2, size_order=0)


def cpu_env() -> dict:
    return {**{k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                              "MKL_NUM_THREADS")}, "nice": os.nice(0)}


def false_rate(fam: P.Family, stations: list[str]) -> tuple[float, int, float]:
    n_det, years = 0, 0.0
    for k in stations:
        r = fam.results[k]
        n_det += sum(1 for s in r.steps if s.evidence == "detected")
        years += r.ok.sum() / 365.25
    return (n_det / years if years else np.nan), n_det, years


def reference(od: Path) -> tuple[list[str], list[str], dict]:
    """The pre-origin reference rule on a pass-1 run (cached in OUT_DIR/reference.json)."""
    f = od / "reference.json"
    if f.exists():
        d = json.loads(f.read_text())
        return d["reference"], d["reference_full"], d
    grid_npz = np.load(io.REPO / "results/twin_forward/datum_gate_innov.npz")
    nser, _ = P.ngl_series(END)
    n1 = P.run_ngl(P0, ref=P.NGL_REF_CANDIDATES_PRE, window_end=END, n_pass=1, series=nser)
    ref, tab = P.select_reference(n1, grid_npz, P.ngl_policy(END)[1])
    nall, _ = P.ngl_series(END, P.NGL_ALL)
    n1all = P.run_ngl(P0, ref=P.NGL_REF_CANDIDATES, window_end=END, n_pass=1, series=nall,
                      stations=P.NGL_ALL)
    ref_full, tab_full = P.select_reference(n1all, grid_npz, P.NGL_REF_CANDIDATES)
    d = {"reference": ref, "reference_full": ref_full,
         "reference_table": tab.to_dict(orient="records"),
         "reference_table_full": tab_full.to_dict(orient="records")}
    f.write_text(json.dumps(d, indent=1, default=str))
    return ref, ref_full, d


def stage1(win: int, od: Path) -> int:
    t0 = time.time()
    ref, _, _ = reference(od)
    wser = P.wra_series(END)
    nser, _ = P.ngl_series(END)
    rows, trials = [], []
    for zmin in GRID_Z:
        for dmin in GRID_DELTA:
            p = replace(P0, win=win, z_min=zmin, delta_min=dmin)
            bw = P.run_wra(p, window_end=END, series=wser)
            bn = P.run_ngl(p, ref=ref, window_end=END, series=nser)
            fr, ndet, yrs = false_rate(bn, ref)
            tw = INJ.run_trials("wra", p, bw, END, [0.02, 0.03], 6, SEED1, series=wser)
            tn = INJ.run_trials("ngl", p, bn, END, [0.02, 0.03], 6, SEED1, ref=ref,
                                series=nser)
            tt = pd.concat([tw, tn])
            tt["win"], tt["z_min"], tt["delta_min"] = win, zmin, dmin
            trials.append(tt)
            sz = tt.abs_size_m.round(3)
            res = {"win": win, "z_min": zmin, "delta_min": dmin,
                   "recovered_2_3cm": float(tt.recovered.mean()),
                   "recovered_2cm": float(tt[sz == 0.02].recovered.mean()),
                   "recovered_3cm": float(tt[sz == 0.03].recovered.mean()),
                   "recovered_wra": float(tt[tt.family == "wra"].recovered.mean()),
                   "recovered_ngl": float(tt[tt.family == "ngl"].recovered.mean()),
                   "found_within_3d": float(tt.found.mean()),
                   "false_per_station_year_offfan": fr, "false_n": ndet,
                   "offfan_station_years": yrs,
                   "wra_detected_steps_base": int(sum(
                       sum(1 for s in r.steps if s.evidence == "detected")
                       for r in bw.results.values())),
                   "ngl_detected_steps_base": int(sum(
                       sum(1 for s in r.steps if s.evidence == "detected")
                       for r in bn.results.values())),
                   "extra_steps_per_trial": float(tt.extra_steps.mean())}
            rows.append(res)
            print(json.dumps(res), f"({time.time() - t0:.0f}s)", flush=True)
    pd.concat(trials).to_csv(od / f"stage1_trials_win{win}.csv", index=False)
    (od / f"stage1_win{win}.json").write_text(json.dumps(
        {"rows": rows, "seed": SEED1, "params_base": replace(P0, win=win).as_dict(),
         "reference": ref, "cpu_env": cpu_env(), "seconds": round(time.time() - t0, 1)},
        indent=1))
    return 0


def select1(od: Path) -> int:
    rows = []
    for w in GRID_WIN:
        rows += json.loads((od / f"stage1_win{w}.json").read_text())["rows"]
    R = pd.DataFrame(rows)
    feas = R[R.false_per_station_year_offfan <= FP_MAX]
    if not len(feas):
        raise SystemExit("no stage-1 configuration meets the false-detection constraint")
    best = feas.recovered_2_3cm.max()
    near = feas[feas.recovered_2_3cm >= best - 0.01].sort_values(
        ["z_min", "delta_min", "win"], ascending=[False, False, True])
    chosen = near.iloc[0].to_dict()
    out = {"grid": rows, "rule": "max pooled 2+3 cm recovery s.t. off-fan false detections <= "
           f"{FP_MAX} per station-year; within 0.01 prefer larger z_min, then larger "
           "delta_min, then smaller win", "chosen": chosen}
    (od / "stage1_selected.json").write_text(json.dumps(out, indent=1))
    print(R.round(4).to_string())
    print("chosen", json.dumps(chosen))
    return 0


def stage1_params(od: Path) -> M.Params:
    c = json.loads((od / "stage1_selected.json").read_text())["chosen"]
    return replace(P0, win=int(c["win"]), z_min=float(c["z_min"]),
                   delta_min=float(c["delta_min"]))


def size_tag(order: int, half: int) -> str:
    return f"o{order}h{half}"


def stage2(order: int, half: int, od: Path) -> int:
    t0 = time.time()
    ref, _, _ = reference(od)
    p = replace(stage1_params(od), size_order=order, local_half=half)
    wser = P.wra_series(END)
    nser, _ = P.ngl_series(END)
    bw = P.run_wra(p, window_end=END, series=wser)
    bn = P.run_ngl(p, ref=ref, window_end=END, series=nser)
    fr, ndet, yrs = false_rate(bn, ref)
    amps = [0.02, 0.03, 0.05, 0.10]
    tw = INJ.run_trials("wra", p, bw, END, amps, 6, SEED2, series=wser)
    tn = INJ.run_trials("ngl", p, bn, END, amps, 6, SEED2, ref=ref, series=nser)
    tt = pd.concat([tw, tn])
    tt["size_order"], tt["local_half"] = order, half
    tt.to_csv(od / f"stage2_trials_{size_tag(order, half)}.csv", index=False)
    f = tt[tt.found]
    res = {"size_order": order, "local_half": half, "recovered": float(tt.recovered.mean()),
           "recovered_wra": float(tt[tt.family == "wra"].recovered.mean()),
           "recovered_ngl": float(tt[tt.family == "ngl"].recovered.mean()),
           "by_size": {f"{a * 100:.0f}cm": float(g.recovered.mean())
                       for a, g in tt.groupby(tt.abs_size_m.round(3))},
           "found_within_3d": float(tt.found.mean()),
           "size_err_mm_sd_found": float(1000 * f.size_err_m.std()),
           "size_err_mm_p90_found": float(np.nanpercentile(1000 * f.size_err_m.abs(), 90)),
           "false_per_station_year_offfan": fr, "false_n": ndet,
           "seconds": round(time.time() - t0, 1)}
    (od / f"stage2_{size_tag(order, half)}.json").write_text(json.dumps(
        {"row": res, "seed": SEED2, "params": p.as_dict(), "cpu_env": cpu_env()}, indent=1))
    print(json.dumps(res), flush=True)
    return 0


def cm_band_evidence(wra: P.Family, ngl: P.Family, ref: list[str]) -> dict:
    """The r3 wra_hf window rule, unchanged."""
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
        rows.append({"window_d": W, "hp_corr_off": float(hp.w.corr(hp.off)),
                     "hp_corr_fan": float(hp.w.corr(hp.fan)),
                     "hp_sd_mm": float(1000 * hp.w.std())})
    ok = [r["window_d"] for r in rows
          if abs(r["hp_corr_off"]) <= 0.05 and abs(r["hp_corr_fan"]) <= 0.05]
    if ok:
        hpw, how = max(ok), "rule"
    else:
        best = min(rows, key=lambda r: max(abs(r["hp_corr_off"]), abs(r["hp_corr_fan"])))
        hpw, how = best["window_d"], "fallback: smallest max(|hp corr|)"
    return {"band_table": rows, "hp_window_selected": hpw, "hp_window_selected_by": how,
            "fan_stations": fan_st, "reference": ref, "n_days": int(len(j))}


def select2(od: Path) -> int:
    t0 = time.time()
    rows = [json.loads((od / f"stage2_{size_tag(o, h)}.json").read_text())["row"]
            for o, h in GRID_SIZE]
    H = pd.DataFrame(rows)
    best = H.recovered.max()
    pick = [(o, h) for o, h in GRID_SIZE      # preference order; first within 0.01 of best
            if float(H[(H.size_order == o) & (H.local_half == h)].recovered.iloc[0])
            >= best - 0.01][0]
    chosen_order, chosen_half = pick
    ref, ref_full, reftab = reference(od)
    pc = replace(stage1_params(od), size_order=chosen_order, local_half=chosen_half)
    wser = P.wra_series(END)
    nser, _ = P.ngl_series(END)
    bw = P.run_wra(pc, window_end=END, series=wser)
    bn = P.run_ngl(pc, ref=ref, window_end=END, series=nser)
    cmev = cm_band_evidence(bw, bn, ref)
    pc = replace(pc, hp_window=int(cmev["hp_window_selected"]))
    s1 = json.loads((od / "stage1_selected.json").read_text())
    rep = {"window_end": str(END.date()), "round": "v3 (repair round 3 S3r)",
           "stage1": s1, "stage2_grid": rows,
           "stage2_rule": "max pooled recovery over 2/3/5/10 cm and both families; within "
                          "0.01 of the best prefer the v2 model (1, 30), then the longer window",
           "chosen_size_order": chosen_order, "chosen_local_half": chosen_half,
           "chosen_params": pc.as_dict(),
           "reference": ref, "reference_full": ref_full,
           "reference_table": reftab["reference_table"],
           "reference_table_full": reftab["reference_table_full"],
           "cm_evidence": cmev, "seeds": {"stage1": SEED1, "stage2": SEED2},
           "ngl_stations_preorigin": sorted(bn.results),
           "wra_stations_preorigin": sorted(bw.results),
           "wra_members_preorigin": list(bw.info["members"]),
           "base_steps": {f"{fam}:{k}": [(str(s.date.date()), round(1000 * s.size, 1),
                                          s.evidence) for s in r.steps]
                          for fam, run in (("wra", bw), ("ngl", bn))
                          for k, r in run.results.items() if r.steps},
           "cpu_env": cpu_env(), "seconds_select2": round(time.time() - t0, 1)}
    (od / "tune.json").write_text(json.dumps(rep, indent=1, default=str))
    print(H.round(4).to_string())
    print("chosen size model", pick, "params", json.dumps(pc.as_dict()))
    print(json.dumps({k: v for k, v in cmev.items() if k != "band_table"}, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["reference", "stage1", "select1", "stage2", "select2"])
    ap.add_argument("out")
    ap.add_argument("--win", type=int)
    ap.add_argument("--size", help="ORDER:HALF of a stage-2 candidate")
    a = ap.parse_args(argv)
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    if a.cmd == "reference":
        ref, ref_full, _ = reference(od)
        print("reference", ref, "full", ref_full)
        return 0
    if a.cmd == "stage1":
        if a.win not in GRID_WIN:
            raise SystemExit(f"--win must be one of {GRID_WIN}")
        return stage1(a.win, od)
    if a.cmd == "select1":
        return select1(od)
    if a.cmd == "stage2":
        o, h = (int(x) for x in str(a.size).split(":"))
        if (o, h) not in GRID_SIZE:
            raise SystemExit(f"--size must be one of {GRID_SIZE}")
        return stage2(o, h, od)
    return select2(od)


if __name__ == "__main__":
    raise SystemExit(main())
