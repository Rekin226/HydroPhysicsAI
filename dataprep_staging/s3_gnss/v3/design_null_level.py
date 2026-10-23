"""WP S3 repair round 3 (S3r): design-time error budget of level-difference size models under
candidate detection common modes, at null positions (no injection; design-time evidence).

    dataprep_staging/tools/cpu_slot.sh python -I dataprep_staging/s3_gnss/v3/design_null_level.py

On the v3 pre-origin base runs (stage-1 starting parameters of design run 1), each station's
step-free residual minus a candidate common mode is given the OLS local step estimate of
``model.scan`` at every injection-eligible day, for the v2 model (linear trend + step, +-30 d)
and the no-slope level difference (order 0) on +-7, 10, 15 and 20 days. Candidates (built from
the other stations' step-free residuals, leave-one-out): the current detection common mode (WRA:
median of every member; NGL: median of the off-fan reference stations), the median of the 3
and of the 5 nearest stations, and for NGL the median of every other NGL station. Writes
``design_null/null_level_cm_{wra,ngl}.csv`` and ``null_level_cm.json``.

Look: ``looks_r5.py design_null_level``, recorded about one minute AFTER this ran (a disclosed
slip; the run is design-time evidence on the same unit as ``design_null``).
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
sys.path.insert(1, str(HERE.parents[1]))           # dataprep_staging
import injection as INJ  # noqa: E402
import model as M  # noqa: E402
import pipeline as P  # noqa: E402
import products as PR  # noqa: E402
from common.paths import REPORTS_DIR  # noqa: E402

EV = REPORTS_DIR / "S3_evidence" / "repair_2026-10-18"
MODELS = ((1, 30), (0, 7), (0, 10), (0, 15), (0, 20))


def frame(fam: P.Family) -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(np.where(r.ok & ~r.untied, r.fit.resid, np.nan),
                                      index=r.dates) for k, r in fam.results.items()})


def evaluate(name: str, fam: P.Family, xy: dict, members: list[str]) -> pd.DataFrame:
    R = frame(fam)
    rows = []
    for k, r in fam.results.items():
        own = R[k].reindex(r.dates).to_numpy()
        others = [m for m in R if m != k]
        mem = [m for m in members if m != k]
        near = sorted(others, key=lambda m: np.hypot(xy[m][0] - xy[k][0], xy[m][1] - xy[k][1]))
        cands = {"loo_median(current)": R[mem].median(axis=1),
                 "knn3": R[near[:3]].median(axis=1), "knn5": R[near[:5]].median(axis=1)}
        if name == "ngl":
            cands["all_median"] = R[others].median(axis=1)
        el = np.flatnonzero(INJ.eligible_days(r))
        for cn, cm in cands.items():
            res = own - cm.reindex(r.dates).to_numpy()
            for order, h in MODELS:
                c, _ = M.scan(res, h, max(3, h // 2) if order == 0 else 15, order)
                e = 1000 * c[el]
                e = e[np.isfinite(e)]
                rows.append({"station": k, "cm": cn, "model": f"o{order}h{h}", "n": e.size,
                             "gt5": float(np.mean(np.abs(e) > 5)), "sd": float(np.std(e))})
    return pd.DataFrame(rows)


def main() -> int:
    od = EV / "design_null"
    od.mkdir(parents=True, exist_ok=True)
    p = replace(M.Params(), win=30, z_min=4.0, delta_min=0.015, local_half=60, trend_order=2)
    end = P.ORIGIN_END
    ref = json.loads((EV / "tune_v3" / "reference.json").read_text())["reference"]
    wra = P.run_wra(p, window_end=end, series=P.wra_series(end))
    nser, _ = P.ngl_series(end)
    ngl = P.run_ngl(p, ref=ref, window_end=end, series=nser)
    out = {}
    for name, fam, xy, mem in (("wra", wra, PR.wra_xy(), wra.info["members"]),
                               ("ngl", ngl, PR.ngl_xy(list(ngl.results)), ref)):
        T = evaluate(name, fam, xy, mem)
        T.to_csv(od / f"null_level_cm_{name}.csv", index=False)
        agg = T.groupby(["cm", "model"]).apply(
            lambda g: pd.Series({"share_gt5": np.average(g.gt5, weights=g.n),
                                 "sd": np.sqrt(np.average(g.sd ** 2, weights=g.n))}),
            include_groups=False).round(4)
        print("==", name)
        print(agg.unstack("model").to_string())
        out[name] = agg.reset_index().to_dict(orient="records")
    (od / "null_level_cm.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
