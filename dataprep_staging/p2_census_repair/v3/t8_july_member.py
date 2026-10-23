"""Registered, reported-only tests P2r3b.b2.july_syncfill_<county> (pre-registration
P2abc.r3.repair.v2): round 2's July backtest (v2/t3_july_backtest.backtest, unchanged: kept
code-M2 meters with full-variant detected phase A; truth (Jul + Aug) / 2 of year Y; estimate
the mean of the same over Y-1..Y-3; Y = 2013..2025) re-run on the recorded synchronized-zero
member of the full v2 product, i.e. with every synchronized-zero meter-month replaced by its
fill (sync_fill.parquet, recorded column). It shows how much the July verdicts depend on the
synchronized-zero months (2019-07/08, 2021-07/08, 2025-07/08 among them). Nothing ships or is
gated on it; the round-2 verdicts (P2r2.b1) stand.

Also reported: the member over Y = 2013..2024, the recorded arm over Y = 2013..2024 (pair mean)
and the plan's own variant (recorded July as read, Y = 2013..2024).

Usage: python -I t8_july_member.py <out_dir>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import p3lib as P3  # noqa: E402
import scoring  # noqa: E402
import t3_july_backtest as T3  # noqa: E402  (v2, read-only)
from p3lib import L  # noqa: E402

PREREG3 = "P2abc.r3.repair.v2"


def main(out_dir: str, staging: str | None = None) -> None:
    t0 = time.time()
    res = {"code_sha256": P3.code_sha(Path(__file__), L.CODE_DIR / "t3_july_backtest.py",
                                      L.CODE_DIR / "p2lib.py"), "counties": {}}
    res["authorization"] = scoring.authorize([f"P2r3b.b2.july_syncfill_{c}"
                                              for c in L.COUNTIES], prereg_id=PREREG3)
    d = Path(staging) if staging else P3.WORK3 / "assembled_full"
    fill = pd.read_parquet(d / "sync_fill.parquet")
    l1 = L.l1_pumps()
    prod = L.open_c0("full")
    for c in L.COUNTIES:
        cd = L.CountyData("full", c, prod=prod, l1=l1)
        isA = (cd.code == "M2") & (cd.phase == "A")
        X = cd.M.astype(np.float64).copy()
        fs = fill[fill["county"] == c]
        r = cd.mpos.reindex(fs["meter_idx"].to_numpy()).to_numpy().astype(int)
        t = np.array([L.tidx(m) for m in fs["month"]], int) if len(fs) else np.zeros(0, int)
        fv = fs["fill_recorded_kwh"].to_numpy()
        ok = np.isfinite(fv)
        X[r[ok], t[ok]] = fv[ok]
        cell = cd.m_cell[isA]
        out = {"n_meters": int(isA.sum()),
               "n_filled_meter_months_phaseA_julaug": int(np.sum(
                   isA[r[ok]] & np.isin(L.DATES[t[ok]].month, [7, 8])))}
        m = T3.backtest(X[isA], cell, T3.YEARS, pair_mean=True)
        out["member_2013_2025"] = {k: m[k] for k in ("per_year", "mean_abs_total_err_pct",
                                                     "max_abs_total_err_pct",
                                                     "cell_rel_rmse_mean")}
        yrs24 = [y for y in T3.YEARS if y <= 2024]
        m24 = T3.backtest(X[isA], cell, yrs24, pair_mean=True)
        out["member_2013_2024"] = {k: m24[k] for k in ("mean_abs_total_err_pct",
                                                       "max_abs_total_err_pct")}
        r24 = T3.backtest(cd.M[isA].astype(np.float64), cell, yrs24, pair_mean=True)
        out["recorded_2013_2024"] = {k: r24[k] for k in ("mean_abs_total_err_pct",
                                                         "max_abs_total_err_pct")}
        a24 = T3.backtest(cd.M[isA].astype(np.float64), cell, yrs24, pair_mean=False)
        out["recorded_as_read_july_2013_2024"] = {k: a24[k] for k in (
            "mean_abs_total_err_pct", "max_abs_total_err_pct")}
        out["criterion_reference"] = "mean |fan-total error| <= 20 % (plan P2b); reported only"
        res["counties"][c] = out
        print(c, round(out["member_2013_2025"]["mean_abs_total_err_pct"], 2),
              round(out["member_2013_2024"]["mean_abs_total_err_pct"], 2),
              round(out["recorded_2013_2024"]["mean_abs_total_err_pct"], 2), flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    L.save_json(res, Path(out_dir) / "t8_july_member.json")


if __name__ == "__main__":
    main(sys.argv[1])
