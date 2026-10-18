"""WP S3 repair round 3 (S3r): trace of the v2 greedy detection loop on failed seed-404 trials
(diagnosis only; failed units, no new draw).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/diag_loop_trace.py

Reproduces the pass-3 detection loop of ``v2/model.process_station`` step by step for six
not-found trials (TKJS, STES) and prints, per iteration, the max-z candidate, the scan values at
the injection day and why the loop accepted, rejected or stopped. The loop is restated here
only to print its decisions; the v2 functions do the arithmetic.

Look logged beforehand: ``looks_r5.py diag_a1_trials diag_a1_background``.
"""
from __future__ import annotations

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
import model as M  # noqa: E402
import pipeline as P  # noqa: E402

CASES = [(0, "TKJS"), (8, "TKJS"), (26, "TKJS"), (12, "STES"), (16, "STES"), (4, "STES")]


def trace(p: M.Params, wser: dict, inj: dict, st: str) -> list[str]:
    end = P.ORIGIN_END
    cat = {k: [(d, x) for d, x in v if pd.Timestamp(d) <= end]
           for k, v in P.WRA_CATALOGUE.items()}
    fam = P.Family("wra", {})
    cm = None
    for _ in range(2):          # passes 1-2 give the pass-3 detection common mode
        fam.results = {k: M.process_station(P._inject(s, inj.get(k)), p,
                                            cm=None if cm is None else cm.get(k),
                                            catalogue=cat.get(k))
                       for k, s in wser.items()}
        cm = P._loo_median(fam.resid_frame(), list(wser))
    s = M.daily(P._inject(wser[st], inj.get(st)))
    dates = pd.DatetimeIndex(s.index)
    y = s.to_numpy(float)
    n = len(y)
    c = cm[st].reindex(dates).to_numpy(float)
    cmv = np.where(np.isfinite(c), c, 0.0)
    yv = np.where(M.gross_ok(y, p), y, np.nan)
    steps: list[M.Step] = []
    fit = M.robust_fit(yv, dates, steps, np.isfinite(yv), p.clip_k, None)
    sig, rho = M.hf_noise(fit.resid - cmv, p.win)
    infl = math.sqrt((1 + rho) / (1 - rho))
    rejected = np.zeros(n, bool)
    d_inj, c_inj = inj[st][0]
    j = int(dates.searchsorted(d_inj))
    out = [f"=== {st} {d_inj.date()} {1000 * c_inj:+.0f} mm: in-loop sigma_hf "
           f"{1000 * sig:.2f} mm, rho {rho:.2f} (inflation {infl:.2f})"]
    for it in range(30):
        rc = np.where(np.isfinite(yv), fit.resid - cmv, np.nan)
        dd, vc = M.scan(rc, p.win, p.min_obs)
        z = np.abs(dd) / (sig * infl * np.sqrt(vc))
        forb = rejected.copy()
        for stp in steps:
            k = int(dates.searchsorted(stp.date))
            forb[max(0, k - p.exclusion_days): k + p.exclusion_days + 1] = True
        z[forb] = np.nan
        i0 = int(np.nanargmax(z))
        zi = z[j - 3:j + 4]
        zj = np.nanmax(zi) if np.isfinite(zi).any() else float("nan")
        msg = (f"  it{it}: max z {z[i0]:.2f} at {dates[i0].date()} |D| {1000 * abs(dd[i0]):.1f}"
               f" mm; at the injection z {zj:.2f} |D| "
               f"{1000 * np.nanmax(np.abs(dd[j - 3:j + 4])):.1f} mm")
        if not (abs(dd[i0]) >= p.delta_min and z[i0] >= p.z_min):
            why = "small D" if abs(dd[i0]) < p.delta_min else "low z"
            out.append(f"{msg} -> loop stops ({why})")
            break
        ib = M.refine(rc, i0, p.win, p.min_obs, forb)
        dsh, na, nb = M.sharpness(rc, ib, p.sharp_days)
        if not (np.isfinite(dsh) and dsh * dd[i0] > 0 and abs(dsh) >= p.sharp_min * abs(dd[i0])):
            out.append(f"{msg} -> rejected as a ramp at {dates[ib].date()} (7-day difference "
                       f"{1000 * dsh if np.isfinite(dsh) else float('nan'):.1f} mm, "
                       f"{na} obs before, {nb} after)")
            rejected[max(0, i0 - p.win // 2): i0 + p.win // 2] = True
            continue
        ie = M.box_or_step(rc, ib, p)
        if ie is not None:
            out.append(f"{msg} -> excursion to {dates[ie].date()}")
            yv[ib:ie] = np.nan
            fit = M.robust_fit(yv, dates, steps, np.isfinite(yv), p.clip_k, fit.t_ref)
            continue
        out.append(f"{msg} -> step at {dates[ib].date()}")
        steps.append(M.Step(date=dates[ib], evidence="detected"))
        steps.sort(key=lambda q: q.date)
        fit = M.robust_fit(yv, dates, steps, np.isfinite(yv), p.clip_k, fit.t_ref)
    return out


def main() -> int:
    p, _ = D.load_params()
    wser = P.wra_series(P.ORIGIN_END)
    t = pd.read_csv(D.EV2 / "acceptance" / "injection_trials_preorigin.csv",
                    parse_dates=["inject_date"])
    for tr, st in CASES:
        rows = t[(t.family == "wra") & (t.trial == tr)]
        inj = {r.station: [(pd.Timestamp(r.inject_date), float(r.size_m))]
               for r in rows.itertuples()}
        print("\n".join(trace(p, wser, inj, st)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
