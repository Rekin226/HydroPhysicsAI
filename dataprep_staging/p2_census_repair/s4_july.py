"""P2b: July 2025 and record ends - rolling-origin backtest of the meter's own 3-year
same-month mean (pre-registered: P2b.july_backtest.v1; ports of pilot s09 and s10), the
imputation sd model, and the M0 July/June ratios behind the flag-9 decision.

Writes ``_work_p2abc/july_backtest.json`` and ``imputation_sd.json``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
import s3_validate as s3  # noqa: E402

YEARS = list(range(2013, 2025))


def own_3yr(X: np.ndarray, t: int) -> np.ndarray:
    return np.mean([X[:, t - 12 * j] for j in (1, 2, 3)], axis=0)


def main() -> None:
    mt, M, _ = lib.load_work()
    tab = pd.read_parquet(lib.WORK / "full_phase.parquet")
    phase = tab["phase"].to_numpy()
    keep = mt["kept"].to_numpy()
    bill = mt["bill"].to_numpy()
    X = np.nan_to_num(M.astype(float))
    isA = keep & (bill == "M2") & (phase == lib.PHASE_A)
    isM = keep & (phase == lib.MONTHLY)
    grid = lib.fan_grid()
    fx = np.isfinite(mt["x"].to_numpy())
    cellkey = (np.floor((mt["y"].to_numpy() - grid.y0) / 1000) * 1000
               + np.floor((mt["x"].to_numpy() - grid.x0) / 1000))
    isA_c = isA & fx                      # pilot s10 also requires finite coordinates
    rows, cell_rows = [], []
    for y in YEARS:
        t7 = lib.tidx(f"{y}-07")
        truth, est = X[isA, t7], own_3yr(X[isA], t7)
        rows.append({"year": y, "meter_rmse": float(np.sqrt(np.mean((est - truth) ** 2))),
                     "total_err_pct": float(100 * (est.sum() - truth.sum()) / truth.sum())})
        tc = pd.Series(X[isA_c, t7]).groupby(cellkey[isA_c]).sum()
        ec = pd.Series(own_3yr(X[isA_c], t7)).groupby(cellkey[isA_c]).sum()
        cell_rows.append(pd.DataFrame({"year": y, "cell": tc.index, "truth": tc.values,
                                       "est": ec.reindex(tc.index).values}))
        rows[-1]["cell_rel_rmse"] = float(np.sqrt(np.mean((ec - tc) ** 2)) / tc.mean())
    bt = pd.DataFrame(rows)
    cells = pd.concat(cell_rows, ignore_index=True)
    a = bt["total_err_pct"].abs()
    pre = bt["year"] <= 2022
    res = {
        "units": {"n_meters_phaseA_M2": int(isA.sum())},
        "per_year": bt.round(4).to_dict(orient="records"),
        "mean_abs_total_err_pct": float(a.mean()), "max_abs_total_err_pct": float(a.max()),
        "mean_abs_total_err_pct_2013_2022": float(a[pre].mean()),
        "max_abs_total_err_pct_2013_2022": float(a[pre].max()),
        "cell_rel_rmse_mean": float(bt["cell_rel_rmse"].mean()),
        "zero_as_recorded_err_pct": 100.0,
        "criterion": "mean_abs_total_err_pct <= 20", "pilot": {"mean": 14.003, "max": 29.202,
                                                               "cell_rel_rmse": 0.335},
    }
    res["passed"] = bool(res["mean_abs_total_err_pct"] <= 20.0)

    # ---- imputation sd: q90 of |est - truth| / est at meter and 1 km cell level,
    # leave-one-year-out coverage of the resulting 90 % intervals (clusters = years).
    # Variant 'preorigin' uses the pre-origin phase table and the Julys 2013-2022 only (U3).
    sdres = {}
    for var, years in (("full", YEARS), ("preorigin", [y for y in YEARS if y <= 2022])):
        ph_v = pd.read_parquet(lib.WORK / f"{var}_phase.parquet")["phase"].to_numpy()
        selA = keep & (bill == "M2") & (ph_v == lib.PHASE_A)
        selC = selA & fx
        met, cel = [], []
        for y in years:
            t7 = lib.tidx(f"{y}-07")
            met.append(pd.DataFrame({"year": y, "est": own_3yr(X[selA], t7),
                                     "truth": X[selA, t7]}))
            tc = pd.Series(X[selC, t7]).groupby(cellkey[selC]).sum()
            ec = pd.Series(own_3yr(X[selC], t7)).groupby(cellkey[selC]).sum()
            cel.append(pd.DataFrame({"year": y, "truth": tc.values,
                                     "est": ec.reindex(tc.index).values}))
        sdres[var] = {"years": [years[0], years[-1]]}
        for lev, d in (("meter", pd.concat(met, ignore_index=True)),
                       ("cell", pd.concat(cel, ignore_index=True))):
            d = d[d["est"] > 0].copy()
            d["z"] = (d["est"] - d["truth"]).abs() / d["est"]
            hits = np.zeros(len(d), bool)
            for y in years:
                q = np.quantile(d.loc[d["year"] != y, "z"], 0.90)
                m = (d["year"] == y).to_numpy()
                hits[m] = d.loc[m, "z"].to_numpy() <= q
            sdres[var][lev] = {"q90_rel": float(np.quantile(d["z"], 0.90)),
                               "loyo_coverage": s3.coverage_gate(hits, d["year"].to_numpy())}
    sdres["rule"] = ("sd = q90_rel / 1.645 x imputed value (meter level for meter arrays, "
                     "1 km cell level for cell arrays); also applied to interior gaps and the "
                     "pre-origin edge months of non-Chow-Lin meters (borrowed calibration)")
    res["imputation_sd"] = sdres

    # ---- M0 / monthly meters in July 2025 (flag 9 decision, plan P2b)
    ratio = {}
    for y in (2022, 2023, 2024, 2025):
        t7 = lib.tidx(f"{y}-07")
        m0 = keep & (bill == "M0")
        ratio[y] = {"M0_jul_over_jun": float(X[m0, t7].sum() / X[m0, t7 - 1].sum()),
                    "monthly_phase_jul_over_jun": float(X[isM, t7].sum() / X[isM, t7 - 1].sum()),
                    "M0_zero_share_july": float(np.mean(X[m0, t7] == 0))}
    res["m0_july_ratio"] = ratio
    t25 = lib.tidx("2025-07")
    res["july2025_zero_share_by_bill"] = {
        b: float(np.mean(X[keep & (bill == b), t25] == 0)) for b in ("M0", "M1", "M2", "M9")}
    lib.save_json(res, lib.WORK / "july_backtest.json")
    lib.save_json(sdres, lib.WORK / "imputation_sd.json")
    print({k: res[k] for k in ("mean_abs_total_err_pct", "max_abs_total_err_pct",
                               "mean_abs_total_err_pct_2013_2022", "cell_rel_rmse_mean",
                               "passed")})
    print(sdres)
    print(ratio, res["july2025_zero_share_by_bill"])


if __name__ == "__main__":
    main()
