"""P2d evidence that cannot weight the members (plan WP P2d), reported only:

* **TPC residual trend**: calendar-year TPC d007012 farm-sector sales minus the cleaned census
  (county scope; all classes, and the agricultural classes), OLS slope from 2016 and from 2017,
  pre-origin window (to 2022, C0 preorigin v3) and full window (to 2025, C0 full v3), with the
  synthetic layer's own annual growth for comparison;
* **survivorship**: terminated meters and their last-12-month energy (P2abc v2
  meter_zero_summary), cumulative by termination year, a lower bound on replacement
  abstraction now on unseen meters;
* **kWh per active meter** (risk of double counting in the high member): annual county energy
  of kept meters per meter with a positive reading in the year, 2012-2016 vs 2017-2022 (and
  2023-2025, full).

Usage: python -I d4_evidence.py <out_json> <layer_dir_preorigin> <layer_dir_full>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dlib as D  # noqa: E402
import looks_def as LK  # noqa: E402
from common import guard  # noqa: E402

AGRI = ("irrigation", "aquaculture", "livestock")


def ols_slope(y: np.ndarray, x: np.ndarray) -> dict:
    X = np.column_stack([np.ones(len(x)), x - x.mean()])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = r @ r / max(len(y) - 2, 1)
    se = float(np.sqrt(s2 / ((x - x.mean()) ** 2).sum()))
    return {"slope_GWh_per_yr": float(b[1]), "se": se, "n_years": int(len(y))}


def census_annual(variant: str, use: str) -> pd.DataFrame:
    p = guard.open_product(D.C0[variant], use)
    df = p.read_table("county_class_monthly.parquet",
                      columns=["county", "scope", "class", "date", "value", "flag"])
    df = df[df["scope"] == "county"].copy()
    df["year"] = df["date"].str[:4].astype(int)
    return df


def tpc_trend(out: dict, layer_dirs: dict) -> None:
    t = D.read_tpc()
    res = {}
    for variant, use, end in (("preorigin", "design_time_evidence", 2022),
                              ("full", "after_the_fact_analysis", 2025)):
        cen = census_annual(variant, use)
        lay = pd.read_parquet(Path(layer_dirs[variant]) / "county_class_monthly.parquet")
        lay["year"] = lay["date"].str[:4].astype(int)
        for k in ("yunlin", "changhua"):
            tp = t[(t["county_zh"] == D.COUNTY_ZH[k]) & (t["sector"] == "農林漁牧")].copy()
            tp["year"] = tp["month"].str[:4].astype(int)
            yrs = list(range(2016, end + 1))
            tpy = tp[tp["year"].isin(yrs)].groupby("year")["kwh"].sum() / 1e6
            c = cen[cen["county"] == k]
            all_y = c[c["class"] == "all"].groupby("year")["value"].sum().reindex(yrs) / 1e6
            agr_y = c[c["class"].isin(AGRI)].groupby("year")["value"].sum().reindex(yrs) / 1e6
            r_all = (tpy - all_y).dropna()
            r_agr = (tpy - agr_y).dropna()
            item = {"years": yrs, "tpc_GWh": tpy.round(3).tolist(),
                    "census_all_GWh": all_y.round(3).tolist(),
                    "resid_all_GWh": r_all.round(3).tolist(),
                    "resid_agri_GWh": r_agr.round(3).tolist(),
                    "coverage_all": (all_y / tpy).round(4).tolist()}
            for y0 in (2016, 2017):
                sel = r_all.index >= y0
                item[f"trend_all_from_{y0}"] = ols_slope(r_all[sel].to_numpy(),
                                                         np.asarray(r_all.index[sel], float))
                sel = r_agr.index >= y0
                item[f"trend_agri_from_{y0}"] = ols_slope(r_agr[sel].to_numpy(),
                                                          np.asarray(r_agr.index[sel], float))
            for m in ("low", "high"):
                ly = lay[(lay["county"] == k) & (lay["member"] == m)].groupby(
                    "year")["value_county_kwh"].sum().reindex(yrs).fillna(0.0) / 1e6
                full_years = [y for y in yrs if y >= 2017]
                item[f"synthetic_{m}_county_GWh"] = ly.round(3).tolist()
                item[f"synthetic_{m}_growth_2017_on"] = ols_slope(
                    ly.reindex(full_years).to_numpy(), np.asarray(full_years, float))
            res[f"{variant}.{k}"] = item
    out["tpc_residual_trend"] = res


def survivorship(out: dict) -> None:
    res = {}
    for variant, use in (("preorigin", "design_time_evidence"),
                         ("full", "after_the_fact_analysis")):
        p = guard.open_product(D.P2ABC[variant], use)
        z = p.read_table("meter_zero_summary.parquet")
        for k in ("yunlin", "changhua", "chiayi"):
            s = z[(z["county"] == k) & z["terminated"].astype(bool)].copy()
            s["ty"] = s["termination"].astype(str).str[:4]
            g = s.groupby("ty").agg(n=("meter_idx", "size"),
                                    gwh=("last12_kwh_before_termination",
                                         lambda v: float(np.nansum(v)) / 1e6))
            g["n_cum"] = g["n"].cumsum()
            g["gwh_cum"] = g["gwh"].cumsum()
            res[f"{variant}.{k}"] = {"by_termination_year": g.reset_index().to_dict(
                orient="records"), "total_meters": int(len(s)),
                "total_last12_GWh": float(np.nansum(s["last12_kwh_before_termination"]) / 1e6)}
    out["survivorship"] = res


def kwh_per_meter(out: dict) -> None:
    res = {}
    l1 = D.l1_table(use="model_design")
    for variant, use in (("preorigin", "model_design"), ("full", "after_the_fact_analysis")):
        for k in ("yunlin", "changhua"):
            p = D.Pumps(variant, k, use, l1=l1)
            E = p.energy()
            T = p.T
            yrs = list(range(2012, (2022 if variant == "preorigin" else 2025) + 1))
            dfm = pd.DataFrame({"m": p.meter_idx})
            rows = {}
            for y in yrs:
                cols = slice(D.tidx(f"{y}-01"), D.tidx(f"{y}-12") + 1)
                if cols.stop > T:
                    continue
                e_p = E[:, cols]
                tot = float(e_p.sum())
                # meter series = pump energy / share; positive reading in the year per meter
                pos = pd.Series((e_p > 0).any(1)).groupby(dfm["m"].to_numpy()).any()
                rows[y] = {"GWh": tot / 1e6, "active_meters": int(pos.sum()),
                           "kWh_per_active_meter": tot / max(int(pos.sum()), 1)}
            per = pd.DataFrame(rows).T
            item = {"by_year": per.round(3).reset_index().rename(
                columns={"index": "year"}).to_dict(orient="records")}
            for a, b in ((2012, 2016), (2017, 2022), (2023, 2025)):
                sel = [y for y in range(a, b + 1) if y in per.index]
                if sel:
                    item[f"{a}_{b}"] = [float(per.loc[sel, "kWh_per_active_meter"].min()),
                                        float(per.loc[sel, "kWh_per_active_meter"].max())]
            res[f"{variant}.{k}"] = item
            del p, E
    out["kwh_per_active_meter"] = res


def main(out_json: str, pre_dir: str, full_dir: str) -> None:
    LK.record(
        "P2def-d4-evidence",
        units={"P2def:d4_census_tpc_annual": "annual cleaned-census county totals (C0 v3) and "
                                             "TPC d007012 farm-sector sales 2016-2025; P2abc v2 "
                                             "terminated meters; kWh per active meter "
                                             "2012-2025"},
        scope="c0_pump_electricity_v3:2012-01..2026-06", purpose="after_the_fact",
        note="P2d evidence reported only (members stay unweighted): TPC residual trend "
             "(pre-origin and full windows), survivorship lower bound, kWh per active meter; "
             "run after the P2d back-test was scored and the layer built.")
    out = {"code": D.p2def_code()}
    tpc_trend(out, {"preorigin": pre_dir, "full": full_dir})
    survivorship(out)
    kwh_per_meter(out)
    D.save_json(out, out_json)
    for k, v in out["tpc_residual_trend"].items():
        print(k, {x: (round(v[x]["slope_GWh_per_yr"], 2), round(v[x]["se"], 2))
                  for x in v if x.startswith("trend") or x.endswith("growth_2017_on")})
    for k, v in out["survivorship"].items():
        print(k, v["total_meters"], round(v["total_last12_GWh"], 2))
    for k, v in out["kwh_per_active_meter"].items():
        print(k, {x: v[x] for x in v if x != "by_year"})


if __name__ == "__main__":
    main(*sys.argv[1:4])
