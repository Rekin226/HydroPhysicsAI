"""Report facts of the P2abc repair round (look [P2-r3-facts], logged before this runs), after
every registered check was scored and both v2 products were published. Descriptive only.

Usage: python -I r3_facts.py <out_json>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parents[2], _HERE.parents[1], _HERE.parent / "v2", _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import p3lib as P3  # noqa: E402
from p3lib import L  # noqa: E402

V1 = {v: L.PRODUCT_ROOT / v / "v1" for v in ("preorigin", "full")}
V2 = {v: L.PRODUCT_ROOT / v / "v2" for v in ("preorigin", "full")}


def gwh(x) -> float:
    return float(np.nansum(x) / 1e6)


def main(out_json: str) -> None:
    res = {"zeros": L.load_json(P3.WORK3 / "zeros" / "zeros_build.json"),
           "zeros_first_build_v1code": L.load_json(P3.EVID3 / "zeros3_build_first_v1code.json")}
    for v in ("preorigin", "full"):
        r = {}
        T = L.VARIANTS[v]["T"]
        dates = pd.DatetimeIndex(L.DATES[:T])
        # v1 -> v2 identity of the primary arms
        for arm in ("recorded", "chowlin"):
            a1 = np.load(V1[v] / f"kwh_by_class_{arm}.npz")
            a2 = np.load(V2[v] / f"kwh_by_class_{arm}.npz")
            d = {}
            for k in ("value", "sd", "value_total_all", "value_by_county"):
                x, y = a1[k], a2[k]
                d[k] = {"identical": bool(np.array_equal(x, y, equal_nan=True)),
                        "max_abs_diff": float(np.nanmax(np.abs(x - y))) if x.size else 0.0}
            for k in ("flag", "flag_total_all"):
                diff = a1[k] != a2[k]
                d[k] = {"n_diff": int(diff.sum()),
                        "months": sorted({str(dates[t].strftime("%Y-%m"))
                                          for t in np.nonzero(diff)[-1]})}
            r[f"v1_v2_{arm}"] = d
        # members: in-grid totals by year and the synchronized months
        tot = {}
        for arm in ("recorded", "recorded_syncfill", "chowlin", "chowlin_syncfill"):
            z = np.load(V2[v] / f"kwh_by_class_{arm}.npz")
            vb = np.nan_to_num(z["value_by_county"])          # (K, C, A, T)
            tot[arm] = vb.sum(axis=(1, 2))                    # (K, T)
        yrs = range(2012, 2023 if v == "preorigin" else 2027)
        r["member_in_grid_GWh_by_year"] = {}
        for ki, c in enumerate(L.COUNTIES):
            rows = {}
            for y in yrs:
                m = (dates.year == y)
                if v == "full" and y == 2026:
                    m &= dates.month <= 6
                rows[str(y)] = {arm: round(gwh(tot[arm][ki, m]), 4) for arm in tot}
            r["member_in_grid_GWh_by_year"][c] = rows
        r["member_in_grid_2012_2022_GWh_per_yr"] = {
            c: {arm: round(gwh(tot[arm][ki, (dates.year >= 2012) & (dates.year <= 2022)]) / 11,
                           3) for arm in tot} for ki, c in enumerate(L.COUNTIES)}
        jj = {}
        for y in (2019, 2021, 2025):
            if v == "preorigin" and y > 2022:
                continue
            m = np.isin(np.arange(T), [L.tidx(f"{y}-07"), L.tidx(f"{y}-08")])
            jj[str(y)] = {c: {arm: round(gwh(tot[arm][ki, m]), 3) for arm in tot}
                          for ki, c in enumerate(L.COUNTIES)}
        r["member_in_grid_jul_aug_GWh"] = jj
        fill = pd.read_parquet(V2[v] / "sync_fill.parquet")
        r["sync_fill_meter_months"] = fill.groupby("county").size().to_dict()
        r["sync_fill_GWh_meter_level"] = (fill.groupby("county")[["fill_recorded_kwh",
                                                                  "fill_chowlin_kwh"]]
                                          .sum() / 1e6).round(4).to_dict()
        fm = fill.groupby(["county", "month"])["fill_recorded_kwh"].sum() / 1e6
        r["sync_fill_GWh_by_month_top"] = {c: fm.loc[c].sort_values(ascending=False).head(12)
                                           .round(4).to_dict() for c in L.COUNTIES
                                           if c in fm.index.get_level_values(0)}
        zt = pd.read_parquet(V2[v] / "zero_class.parquet")
        r["zero_rows"] = int(len(zt))
        r["zero_class_by_county"] = zt.groupby(["county", "zclass"], observed=True).size() \
            .unstack(fill_value=0).to_dict(orient="index")
        r["structural_rows_by_county"] = zt[zt["structural"]].groupby("county").size().to_dict()
        r["sync_rows_by_county"] = zt[zt["sync_suspect"]].groupby("county").size().to_dict()
        r["sync_months_by_county"] = {c: sorted(zt.loc[(zt["county"] == c) & zt["sync_suspect"],
                                                       "month"].unique().tolist())
                                      for c in L.COUNTIES}
        ms = pd.read_parquet(V2[v] / "meter_zero_summary.parquet")
        r["terminated_meters"] = ms.groupby("county")["terminated"].sum().astype(int).to_dict()
        r["terminated_last12_GWh"] = (ms.groupby("county")["last12_kwh_before_termination"]
                                      .sum() / 1e6).round(3).to_dict()
        r["agreement_with_c0_terminated"] = ms.groupby("county").apply(
            lambda g: float(np.mean(g["terminated"] == g["c0_terminated"]))).round(4).to_dict()
        mk = pd.read_parquet(V2[v] / "incomplete_pair_mask.parquet")
        r["mask_reasons"] = mk.groupby(["county", "reason"]).size().unstack(fill_value=0) \
            .to_dict(orient="index")
        if v == "full":
            t25 = L.tidx("2025-07")
            c0 = np.load(L.C0_DIR["full"] / "cells_by_class.npz")
            rec = np.load(V2[v] / "kwh_by_class_recorded.npz")
            f2, f0 = rec["flag"][:, :, t25], c0["flag"][:, :, t25]
            r["m0_2025_07_cells_flag9_new"] = int(((f2 == 9) & (f0 != 9)).sum())
            r["m0_2025_07_class_cells_with_energy"] = int((np.nan_to_num(
                rec["value"][:, :, t25]) > 0).sum())
            r["m0_2025_07_total_cells_flag9"] = int((rec["flag_total_all"][:, t25] == 9).sum())
            r["m0_2025_07_total_cells_flag9_c0"] = int((np.load(
                L.C0_DIR["full"] / "cells_total.npz")["flag_all"][:, t25] == 9).sum())
            e = {}
            for c in L.COUNTIES:
                z = np.load(V2[v] / f"pumps_{L.FILETAG[c]}.npz")
                s = z["method"][:, t25] == 12
                ing = z["cell"] >= 0
                e[c] = {"pumps": int(s.sum()), "GWh_2025_07": round(gwh(z["value"][s, t25]), 4),
                        "GWh_2025_07_in_grid": round(gwh(z["value"][s & ing, t25]), 4),
                        "county_GWh_2025_07_all_pumps": round(gwh(z["value"][:, t25]), 4)}
            r["m0_2025_07_energy"] = e
            jl = pd.read_parquet(V2[v] / "july2026_imputation.parquet")
            r["july2026"] = {c: {k: round(float(g[k].sum() / 1e6), 4) for k in (
                "imputed_recorded_kwh", "imputed_recorded_syncfill_kwh", "imputed_chowlin_kwh",
                "imputed_chowlin_syncfill_kwh")} | {"shipped": int(g["shipped"].sum()),
                                                     "n": int(len(g))}
                for c, g in jl.groupby("county")}
        res[v] = r
    res["t6"] = L.load_json(P3.EVID3 / "tests" / "t6_zero_shipped.json")
    res["t8"] = L.load_json(P3.EVID3 / "tests" / "t8_july_member.json")
    L.save_json(res, out_json)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
