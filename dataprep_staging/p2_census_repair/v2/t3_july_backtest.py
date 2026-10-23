"""Registered test P2r2.b1 (P2b): rolling-origin July backtest of the meter's own 3-year
same-month mean, per county, on the C0 full v3 product (pre-registration
P2abc.r2.c0_based.v1).

Units: kept meters with billing code M2 and full-variant detected phase A (pilot s09 rule).
For each July Y in 2013..2025: truth = the meter's July under the even split of its Jul-Aug
bill, (Jul + Aug) / 2 (missing readings as 0, as the pilot); estimate = the mean of that value
over the three previous Julys. Metric: fan-total error 100 (sum est - sum truth) / sum truth;
criterion: mean |error| over 2013..2025 <= 20 % (plan P2b). A county that fails ships no
July-2026 imputation (its open phase-A July 2026 stays NaN, flag 8).

Reported (not gating): 2013-2022 only; max error; 1 km cell relative RMSE (first-pump cell);
the Chow-Lin arm (truth and base = Chow-Lin July values of the full-variant stage-1 split);
imputation sd: q90 of |est - truth| / est (est > 0) at meter and cell level, leave-one-year-out
coverage of the 90 % interval with clusters = years (harness.coverage3; 13 years < 20 clusters
is outside the calibrated envelope, so the label is 'uncalibrated' unless the envelope is met);
M0 July/June ratios 2022-2026 and the July zero shares by billing code.

Usage: python -I t3_july_backtest.py <out_dir>
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2lib as L  # noqa: E402
import scoring  # noqa: E402
from harness import coverage3  # noqa: E402

YEARS = list(range(2013, 2026))


def code_sha() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), L.CODE_DIR / "p2lib.py")}


def july(X: np.ndarray, y: int, pair_mean: bool) -> np.ndarray:
    j = L.tidx(f"{y}-07")
    if pair_mean:
        return (np.nan_to_num(X[:, j]) + np.nan_to_num(X[:, j + 1])) / 2
    return np.nan_to_num(X[:, j])


def backtest(X: np.ndarray, cell: np.ndarray, years, pair_mean: bool) -> dict:
    rows, met, cel = [], [], []
    for y in years:
        tr = july(X, y, pair_mean)
        est = np.mean([july(X, y - k, pair_mean) for k in (1, 2, 3)], axis=0)
        r = {"year": y, "meter_rmse": float(np.sqrt(np.mean((est - tr) ** 2))),
             "total_err_pct": float(100 * (est.sum() - tr.sum()) / tr.sum())}
        ok = cell >= 0
        tc = pd.Series(tr[ok]).groupby(cell[ok]).sum()
        ec = pd.Series(est[ok]).groupby(cell[ok]).sum()
        r["cell_rel_rmse"] = float(np.sqrt(np.mean((ec - tc) ** 2)) / tc.mean())
        rows.append(r)
        met.append(pd.DataFrame({"year": y, "est": est, "truth": tr}))
        cel.append(pd.DataFrame({"year": y, "est": ec.values, "truth": tc.values}))
    bt = pd.DataFrame(rows)
    a = bt["total_err_pct"].abs()
    pre = bt["year"] <= 2022
    out = {"per_year": bt.round(6).to_dict(orient="records"),
           "mean_abs_total_err_pct": float(a.mean()), "max_abs_total_err_pct": float(a.max()),
           "mean_abs_total_err_pct_2013_2022": float(a[pre].mean()),
           "max_abs_total_err_pct_2013_2022": float(a[pre].max()),
           "cell_rel_rmse_mean": float(bt["cell_rel_rmse"].mean())}
    sd = {}
    for lev, d in (("meter", pd.concat(met, ignore_index=True)),
                   ("cell", pd.concat(cel, ignore_index=True))):
        d = d[d["est"] > 0].copy()
        d["z"] = (d["est"] - d["truth"]).abs() / d["est"]
        hits = np.zeros(len(d), bool)
        for y in years:
            m = (d["year"] == y).to_numpy()
            q = np.quantile(d.loc[~m, "z"], 0.90)
            hits[m] = d.loc[m, "z"].to_numpy() <= q
        g = coverage3.coverage_gate(hits, d["year"].to_numpy())
        sd[lev] = {"q90_rel": float(np.quantile(d["z"], 0.90)), "loyo_coverage": g,
                   "label": "calibrated" if g["pass"] and g["calibrated"] else "uncalibrated"}
    out["imputation_sd"] = sd
    return out


def main(out_dir: str) -> None:
    t0 = time.time()
    res = {"code_sha256": code_sha(), "years": [YEARS[0], YEARS[-1]], "counties": {}}
    res["authorization"] = scoring.authorize([f"P2r2.b1.july_{c}" for c in L.COUNTIES])
    l1 = L.l1_pumps()
    prod = L.open_c0("full")
    for c in L.COUNTIES:
        cd = L.CountyData("full", c, prod=prod, l1=l1)
        code, ph = cd.code, cd.phase
        isA = (code == "M2") & (ph == "A")
        X = cd.M[isA]
        cell = cd.m_cell[isA]
        r = {"units": {"n_meters_M2_phaseA": int(isA.sum())}}
        r["recorded_pair_mean"] = backtest(X, cell, YEARS, pair_mean=True)
        r["criterion"] = "mean_abs_total_err_pct (2013-2025, pair-mean July) <= 20"
        r["passed"] = bool(r["recorded_pair_mean"]["mean_abs_total_err_pct"] <= 20.0)
        z = np.load(L.WORK / f"full_{c}_cl.npz")
        Mcl = cd.M.astype(np.float64).copy()
        Mcl[z["rows"]] = z["values"]
        r["chowlin_arm"] = backtest(Mcl[isA], cell, YEARS, pair_mean=False)
        r["recorded_as_read_july"] = {k: v for k, v in backtest(X, cell, YEARS,
                                                                 pair_mean=False).items()
                                      if k != "imputation_sd"}
        keep_m0 = code == "M0"
        ratio = {}
        for y in (2022, 2023, 2024, 2025, 2026):
            j = L.tidx(f"{y}-07")
            ratio[str(y)] = {
                "M0_jul_over_jun": float(np.nansum(cd.M[keep_m0, j])
                                         / np.nansum(cd.M[keep_m0, j - 1])),
                "zero_share_july_by_code": {b: float(np.mean(cd.M[code == b, j] == 0))
                                            for b in ("M0", "M1", "M2", "M9")
                                            if (code == b).any()}}
        r["m0_july_ratio"] = ratio
        res["counties"][c] = r
        print(c, {k: round(v, 3) for k, v in r["recorded_pair_mean"].items()
                  if isinstance(v, float)}, "passed", r["passed"], flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    L.save_json(res, Path(out_dir) / "t3_july_backtest.json")


if __name__ == "__main__":
    main(sys.argv[1])
