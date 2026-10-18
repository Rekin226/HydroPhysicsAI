"""WP S3 repair round 3 (S3r): design-time error budget of candidate DETECTION common modes at
null positions (no injection; design-time evidence, not a test).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/design_null_cm.py OUT

On the v3 pre-origin base runs (stage-1 starting parameters), each station's step-free residual
(series minus its global model with every step) has a candidate common mode subtracted, and the
OLS local step-size model (linear trend + step, +-20 and +-30 days) is evaluated at every
injection-eligible day. Candidates, built from the OTHER stations' step-free residuals of the
same family (leave-one-out):

* ``none``: no common mode;
* ``median``: the leave-one-out median (WRA: every member; NGL: the off-fan reference
  stations), i.e. the v2/v3 detection common mode;
* ``median_scaled``: ``alpha x median`` with alpha the station's robust (Huber) regression slope
  of its residual on that median;
* ``knn5_median``: WRA only, median of the 5 nearest other stations;
* ``knn5_scaled``: WRA only, alpha x knn5_median.

Reports per family the share of null estimates beyond 5 mm and their sd.

Look logged beforehand: ``looks_r5.py design_null_cm``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import injection as INJ  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402
import products as PR  # noqa: E402

HALVES = (20, 30)


def huber_slope(y: np.ndarray, x: np.ndarray, k: float = 1.345) -> float:
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < 100:
        return 1.0
    yy, xx = y[m], x[m]
    a = float(np.dot(xx, yy) / np.dot(xx, xx))
    for _ in range(20):
        r = yy - a * xx
        s = M.mad_sd(r)
        w = np.minimum(1.0, k * s / np.maximum(np.abs(r), 1e-12))
        a_new = float(np.sum(w * xx * yy) / np.sum(w * xx * xx))
        if abs(a_new - a) < 1e-6:
            break
        a = a_new
    return a


def step_free_frame(fam: P.Family) -> pd.DataFrame:
    cols = {}
    for k, r in fam.results.items():
        cols[k] = pd.Series(np.where(r.ok & ~r.untied, r.fit.resid, np.nan), index=r.dates)
    return pd.DataFrame(cols)


def evaluate(fam: P.Family, members: list[str], xy: dict | None) -> pd.DataFrame:
    R = step_free_frame(fam)
    rows = []
    for k, r in fam.results.items():
        own = R[k].reindex(r.dates).to_numpy()
        others = [m for m in members if m != k and m in R]
        cands = {"none": np.zeros(len(r.dates))}
        med = R[others].median(axis=1).reindex(r.dates).to_numpy()
        cands["median"] = med
        cands["median_scaled"] = huber_slope(own, med) * med
        if xy is not None and k in xy:
            d = sorted(others, key=lambda m: np.hypot(xy[m][0] - xy[k][0], xy[m][1] - xy[k][1]))
            kn = R[d[:5]].median(axis=1).reindex(r.dates).to_numpy()
            cands["knn5_median"] = kn
            cands["knn5_scaled"] = huber_slope(own, kn) * kn
        el = np.flatnonzero(INJ.eligible_days(r))
        for name, cm in cands.items():
            res = own - np.where(np.isfinite(cm), cm, np.nan if name != "none" else 0.0)
            for h in HALVES:
                c, _ = M.scan(res, h, 15, 1)
                e = 1000 * c[el]
                e = e[np.isfinite(e)]
                rows.append({"station": k, "cm": name, "half": h, "n": int(e.size),
                             "sd_mm": float(np.std(e)) if e.size else np.nan,
                             "share_gt5": float(np.mean(np.abs(e) > 5)) if e.size else np.nan})
    return pd.DataFrame(rows)


def main(out: str) -> int:
    od = Path(out)
    od.mkdir(parents=True, exist_ok=True)
    p = replace(M.Params(), win=30, z_min=4.0, delta_min=0.015, local_half=60, trend_order=2)
    end = P.ORIGIN_END
    ref = json.loads((od.parent / "tune_v3" / "reference.json").read_text())["reference"]
    wra = P.run_wra(p, window_end=end, series=P.wra_series(end))
    nser, _ = P.ngl_series(end)
    ngl = P.run_ngl(p, ref=ref, window_end=end, series=nser)
    rep = {}
    for name, fam, mem, xy in (("wra", wra, wra.info["members"], PR.wra_xy()),
                               ("ngl", ngl, ref, None)):
        T = evaluate(fam, mem, xy)
        T.to_csv(od / f"null_cm_{name}.csv", index=False)
        agg = T.groupby(["cm", "half"]).apply(
            lambda g: pd.Series({"share_gt5": float(np.average(g.share_gt5, weights=g.n)),
                                 "sd_mm_rms": float(np.sqrt(np.average(g.sd_mm ** 2,
                                                                       weights=g.n)))}),
            include_groups=False)
        rep[name] = agg.round(4).reset_index().to_dict(orient="records")
        print("==", name)
        print(agg.round(4).to_string())
    (od / "null_cm.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
