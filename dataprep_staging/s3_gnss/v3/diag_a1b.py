"""WP S3 repair round 3 (S3r): error budget of the v2 detector (diagnosis only; no new draw).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/diag_a1b.py OUT_DIR

1. **Size estimator at null positions.** The v2 local step size is an OLS-type estimate that
   is linear in the step, so its error at an injected step equals the estimate at the same
   position of the un-injected series. For every station of the v2 pre-origin base runs it is
   evaluated at every injection-eligible day (``injection.eligible_days``) on the series the
   pipeline uses (screened values minus the global seasonal term minus the pass-3 detection
   common mode); per station: sd, MAD-sd and the share beyond 5 mm. These are compared with
   the size errors of the failed seed-404 trials of the same station (validation of the error
   model).
2. **Scan z at null positions.** The scan statistic ``|D| / (sigma_hf infl sqrt(v))`` at the
   same days: its MAD-sd and the share above the threshold (the noise model behind z).
3. **Failed trials flagged as excursions** (seed 404, pre-origin): every such trial run is
   reproduced exactly (all injections of that family run, read from the archived table) with
   the v2 code, and the station's notes (excursion or transient decisions) are recorded.

Look logged beforehand: ``looks_r5.py diag_a1_trials diag_a1_background``.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))          # dataprep_staging
sys.path.insert(0, str(HERE.parent / "v2"))       # unchanged v2 pipeline
sys.path.append(str(HERE))                        # v3 diagnostics (after v2)
import diag_a1 as D  # noqa: E402
import injection as INJ  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402

TRIALS = D.EV2 / "acceptance" / "injection_trials_preorigin.csv"


def station_arrays(fam: P.Family, k: str, p: M.Params) -> tuple[np.ndarray, np.ndarray]:
    """(y_adj, rc) of the final pass: y_adj as in ``process_station``'s final local sizes,
    rc = residual minus the detection common mode, screened (as in the scan)."""
    r = fam.results[k]
    cm = fam.detect_cm.get(k)
    cmv = np.zeros(len(r.dates))
    if cm is not None:
        c = cm.reindex(r.dates).to_numpy(dtype=float)
        cmv = np.where(np.isfinite(c), c, 0.0)
    yv = np.where(r.ok, r.y - r.correction, np.nan)      # steps removed (null background)
    y_adj = yv - r.fit.seasonal - cmv
    rc = np.where(r.ok, r.fit.resid, np.nan) - cmv
    return y_adj, rc


def null_errors(fam: P.Family, p: M.Params) -> pd.DataFrame:
    rows = []
    for k, r in fam.results.items():
        y_adj, rc = station_arrays(fam, k, p)
        el = np.flatnonzero(INJ.eligible_days(r))
        D_, vc = M.scan(rc, p.win, p.min_obs)
        infl = math.sqrt((1 + r.rho) / (1 - r.rho))
        z = np.abs(D_) / (r.sigma_hf * infl * np.sqrt(vc))
        for j in el:
            c, se, n = M.local_step(y_adj, r.dates, int(j), p.local_half, [], r.rho)
            rows.append({"station": k, "day": r.dates[j].date(), "err_mm": 1000 * c,
                         "se_mm": 1000 * se, "scan_D_mm": 1000 * D_[j], "scan_z": z[j]})
    return pd.DataFrame(rows)


def summarise(N: pd.DataFrame, trials: pd.DataFrame, fam: str, z_min: float) -> pd.DataFrame:
    t = trials[(trials.family == fam) & trials.found]
    obs = t.groupby("station").size_err_m.agg(
        trial_sd_mm=lambda x: 1000 * x.std(),
        trial_share_gt5=lambda x: float((1000 * x.abs() > 5).mean()), trial_n="size")
    g = N.groupby("station")
    out = pd.DataFrame({
        "null_n": g.size(),
        "null_sd_mm": g.err_mm.std(),
        "null_madsd_mm": g.err_mm.agg(lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))),
        "null_share_gt5": g.err_mm.agg(lambda x: float((x.abs() > 5).mean())),
        "null_median_se_mm": g.se_mm.median(),
        "z_null_madsd": g.scan_z.agg(lambda x: 1.4826 * np.median(np.abs(x))),
        "z_null_share_gt_zmin": g.scan_z.agg(lambda x: float((x >= z_min).mean())),
        "D_null_share_ge15mm": g.scan_D_mm.agg(lambda x: float((x.abs() >= 15).mean()))})
    return out.join(obs).round(3)


def reproduce_excursions(p: M.Params, ref: list[str], trials: pd.DataFrame) -> list[dict]:
    end = P.ORIGIN_END
    wser = P.wra_series(end)
    nser, _ = P.ngl_series(end)
    out = []
    bad = trials[(~trials.recovered) & trials.flagged_as_excursion]
    for (fam, trial), _g in bad.groupby(["family", "trial"]):
        allinj = trials[(trials.family == fam) & (trials.trial == trial)]
        inj = {r.station: [(pd.Timestamp(r.inject_date), float(r.size_m))]
               for r in allinj.itertuples()}
        if fam == "wra":
            run = P.run_wra(p, window_end=end, inject=inj, stations=list(wser), series=wser)
        else:
            run = P.run_ngl(p, ref=ref, window_end=end, inject=inj, stations=list(nser),
                            series=nser)
        for r in _g.itertuples():
            res = run.results[r.station]
            out.append({"family": fam, "trial": int(trial), "station": r.station,
                        "inject_date": str(pd.Timestamp(r.inject_date).date()),
                        "size_mm": 1000 * r.size_m, "found": bool(r.found),
                        "steps": [(str(s.date.date()), round(1000 * s.size, 1), s.evidence)
                                  for s in res.steps],
                        "notes": [n for n in res.notes
                                  if "excursion" in n or "transient" in n or "dropped" in n]})
    return out


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    p, ref = D.load_params()
    end = P.ORIGIN_END
    wra = P.run_wra(p, window_end=end, series=P.wra_series(end))
    nser, _ = P.ngl_series(end)
    ngl = P.run_ngl(p, ref=ref, window_end=end, series=nser)
    trials = pd.read_csv(TRIALS, parse_dates=["inject_date"])
    res = {}
    for fam, f in (("wra", wra), ("ngl", ngl)):
        N = null_errors(f, p)
        N.to_csv(od / f"null_errors_{fam}.csv", index=False)
        S = summarise(N, trials, fam, p.z_min)
        S.to_csv(od / f"null_vs_trials_{fam}.csv")
        allN = {"null_sd_mm": float(N.err_mm.std()),
                "null_madsd_mm": float(1.4826 * np.median(np.abs(N.err_mm - N.err_mm.median()))),
                "null_share_gt5": float((N.err_mm.abs() > 5).mean()),
                "z_null_share_gt_zmin": float((N.scan_z >= p.z_min).mean()),
                "corr_station_share_null_vs_trials": float(
                    S[["null_share_gt5", "trial_share_gt5"]].corr().iloc[0, 1])}
        res[fam] = allN
        print("==", fam, json.dumps(allN))
        print(S.to_string())
    ex = reproduce_excursions(p, ref, trials)
    res["excursion_reproductions"] = ex
    for e in ex:
        print(json.dumps(e, ensure_ascii=False))
    (od / "a1_error_budget.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
