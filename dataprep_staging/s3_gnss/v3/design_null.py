"""WP S3 repair round 3 (S3r): design-time error budget of candidate local step-size models at
null positions (no injection; design-time evidence, not a test).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/design_null.py OUT

On the v3 pre-origin base runs (stage-1 starting parameters), the OLS local step estimate of a
model "polynomial trend of order k + step" on +-h days is evaluated at every
injection-eligible day of the step-free residual (series minus the station's global model with
every step, minus the detection common mode). With no step present, the estimate is the error
an injected step would get from that model (exactly for OLS; the pipeline's estimator is the
same fit with a 4-MAD clip). Reports, per family, the sd, the robust sd and the share beyond
5 mm for (k, h) in {1, 2, 3} x {20, 30, 45, 60, 90}.

Look logged beforehand: ``looks_r5.py design_null``.
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

ORDERS = (1, 2, 3)
HALVES = (20, 30, 45, 60, 90)


def null_table(fam: P.Family) -> pd.DataFrame:
    rows = []
    for k, r in fam.results.items():
        cmv = r.cmv if r.cmv is not None else np.zeros(len(r.dates))
        res = np.where(r.ok & ~r.untied, r.fit.resid, np.nan) - cmv
        el = np.flatnonzero(INJ.eligible_days(r))
        for order in ORDERS:
            for h in HALVES:
                c, _ = M.scan(res, h, 15, order)
                e = 1000 * c[el]
                e = e[np.isfinite(e)]
                rows.append({"station": k, "order": order, "half": h, "n": int(e.size),
                             "sd_mm": float(np.std(e)) if e.size else np.nan,
                             "madsd_mm": float(M.mad_sd(e)) if e.size else np.nan,
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
    for name, fam in (("wra", wra), ("ngl", ngl)):
        T = null_table(fam)
        T.to_csv(od / f"null_models_{name}.csv", index=False)
        w = T.assign(wn=T.n)
        agg = w.groupby(["order", "half"]).apply(
            lambda g: pd.Series({"share_gt5": float(np.average(g.share_gt5, weights=g.wn)),
                                 "sd_mm_rms": float(np.sqrt(np.average(g.sd_mm ** 2,
                                                                       weights=g.wn))),
                                 "median_station_madsd_mm": float(g.madsd_mm.median())}),
            include_groups=False)
        rep[name] = agg.round(4).reset_index().to_dict(orient="records")
        print("==", name)
        print(agg.round(4).to_string())
    (od / "null_models.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
