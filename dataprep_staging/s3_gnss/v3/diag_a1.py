"""WP S3 repair round 3 (S3r): anatomy of the failed A1 trials (diagnosis only; no new draw).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/diag_a1.py OUT_DIR

Reads the archived trial tables of the v2 gate run (seed 404, pre-origin) and rebuilds the v2
pre-origin base runs (no injection) with the registered v2 tuned parameters (tune round r3),
using the unchanged v2 code. For every trial it attaches covariates of the background at the
injection day: station noise (sigma_hf, lag-1 rho), observations within +-30 days, distance to
the nearest gap longer than 10 days, to the nearest base-run step (and its size), to the
nearest base-run excursion, to the record ends; month. It then tabulates the failure classes
(not found / found with |size error| > 5 mm) against those covariates, and for the size
failures compares the size error with the estimator's own standard error.

Look logged beforehand: ``looks_r5.py diag_a1_trials diag_a1_background``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # dataprep_staging
sys.path.insert(0, str(HERE.parent / "v2"))       # unchanged v2 pipeline
import model as M  # noqa: E402
import pipeline as P  # noqa: E402
from common.paths import REPORTS_DIR  # noqa: E402

EV2 = REPORTS_DIR / "S3_evidence" / "repair_2026-10-13"
TUNE = EV2 / "tune_r3" / "tune.json"


def load_params() -> tuple[M.Params, list[str]]:
    t = json.loads(TUNE.read_text())
    cp = t["chosen_params"]
    return replace(M.Params(), **{k: cp[k] for k in M.Params().as_dict()}), list(t["reference"])


def covariates(r: M.StationResult, d: pd.Timestamp) -> dict:
    j = int(r.dates.searchsorted(d))
    n = len(r.dates)
    obs = np.isfinite(r.y)
    lo, hi = max(0, j - 30), min(n, j + 30)
    oi = np.flatnonzero(obs)
    # gaps longer than 10 days: (end of last obs before, first obs after)
    gaps = []
    if len(oi) > 1:
        dd = np.diff(oi)
        for k in np.flatnonzero(dd > 10):
            gaps.append((oi[k], oi[k + 1]))
    dgap = min([min(abs(j - a), abs(j - b)) for a, b in gaps], default=9999)
    st = [(abs((s.date - d).days), s) for s in r.steps]
    near = min(st, key=lambda q: q[0]) if st else (9999, None)
    tr = np.flatnonzero(r.transient)
    dtr = int(np.min(np.abs(tr - j))) if len(tr) else 9999
    return {"sigma_hf_mm": 1000 * r.sigma_hf, "rho": r.rho,
            "n_obs_pm30": int(obs[lo:hi].sum()), "n_obs_before30": int(obs[lo:j].sum()),
            "n_obs_after30": int(obs[j:hi].sum()),
            "days_to_gap10": int(dgap), "days_to_base_step": int(near[0]),
            "base_step_size_mm": 1000 * near[1].size if near[1] is not None else np.nan,
            "base_step_evidence": near[1].evidence if near[1] is not None else "",
            "days_to_excursion": dtr, "days_to_start": int(j), "days_to_end": int(n - 1 - j),
            "month": int(d.month)}


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    p, ref = load_params()
    end = P.ORIGIN_END
    wra = P.run_wra(p, window_end=end, series=P.wra_series(end))
    nser, _ = P.ngl_series(end)
    ngl = P.run_ngl(p, ref=ref, window_end=end, series=nser)
    base = {"wra": wra, "ngl": ngl}
    t = pd.read_csv(EV2 / "acceptance" / "injection_trials_preorigin.csv",
                    parse_dates=["inject_date"])
    rows = []
    for rec in t.itertuples():
        r = base[rec.family].results[rec.station]
        cv = covariates(r, rec.inject_date)
        cls = "ok" if rec.recovered else ("not_found" if not rec.found else "size")
        rows.append({"family": rec.family, "station": rec.station,
                     "inject_date": rec.inject_date.date(), "size_mm": 1000 * rec.size_m,
                     "abs_size_mm": 1000 * rec.abs_size_m, "cls": cls,
                     "err_mm": 1000 * rec.size_err_m, "excursion": rec.flagged_as_excursion,
                     **cv})
    T = pd.DataFrame(rows)
    T.to_csv(od / "a1_trials_with_covariates.csv", index=False)
    summ: dict = {}

    def rate(g):
        return {"n": int(len(g)), "fail": float((g.cls != "ok").mean()),
                "not_found": float((g.cls == "not_found").mean()),
                "size": float((g.cls == "size").mean())}
    for fam, g in T.groupby("family"):
        s: dict = {"all": rate(g)}
        s["by_sigma_hf_mm"] = {str(k): rate(x) for k, x in
                               g.groupby(pd.cut(g.sigma_hf_mm, [0, 3, 4, 5, 6, 8, 30]),
                                         observed=True)}
        s["by_days_to_base_step"] = {str(k): rate(x) for k, x in
                                     g.groupby(pd.cut(g.days_to_base_step,
                                                      [0, 90, 120, 180, 365, 10000]),
                                               observed=True)}
        s["by_days_to_gap10"] = {str(k): rate(x) for k, x in
                                 g.groupby(pd.cut(g.days_to_gap10,
                                                  [-1, 15, 30, 60, 120, 10000]), observed=True)}
        s["by_n_obs_pm30"] = {str(k): rate(x) for k, x in
                              g.groupby(pd.cut(g.n_obs_pm30, [0, 45, 50, 55, 58, 61]),
                                        observed=True)}
        s["by_days_to_excursion"] = {str(k): rate(x) for k, x in
                                     g.groupby(pd.cut(g.days_to_excursion,
                                                      [-1, 0, 60, 120, 240, 10000]),
                                               observed=True)}
        s["by_abs_size_mm"] = {str(k): rate(x) for k, x in g.groupby("abs_size_mm")}
        s["by_month"] = {str(k): rate(x) for k, x in g.groupby("month")}
        s["station_noise"] = {k: {"sigma_hf_mm": round(1000 * r.sigma_hf, 2),
                                  "rho": round(r.rho, 3),
                                  "n_steps": len(r.steps),
                                  "rate_mm_yr": round(1000 * r.fit.rate, 2)}
                              for k, r in base[fam].results.items()}
        s["excursion_flagged_not_found"] = int(((g.cls == "not_found") & g.excursion).sum())
        s["excursion_flagged_size"] = int(((g.cls == "size") & g.excursion).sum())
        summ[fam] = s
    (od / "a1_anatomy.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({f: summ[f]["all"] for f in summ}, indent=1))
    for fam in summ:
        print("==", fam)
        for key in ("by_sigma_hf_mm", "by_days_to_base_step", "by_days_to_gap10",
                    "by_n_obs_pm30", "by_days_to_excursion"):
            print(key)
            for k, v in summ[fam][key].items():
                print(f"   {k:>16} n={v['n']:4d} fail={v['fail']:.3f} nf={v['not_found']:.3f} "
                      f"size={v['size']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
