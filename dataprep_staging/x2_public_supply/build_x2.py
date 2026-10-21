"""Build the X2 products: public-supply (Taiwan Water) groundwater abstraction, Changhua and
Yunlin, from the public yearbooks and briefings; P6 comparison; gridded variants.

Products (new versioned directories, common.manifest):

* ``public_supply_gw/v<N>`` (L2, full 2007-2025): county annual and monthly tables with sd,
  flag and source page; the Changhua per-system table; every extracted yearbook number with
  its table and page; the WRA Jan-May reads; the P6 comparison; ``grid_full.npz`` with the
  variants county_uniform, p6_proxy and service_area (physical m3/day, 4 x 2148 x 228).
  Post-origin values: after-the-fact analysis, recorded-forcing arm, statistical-layer
  history and validation only (U4).
* ``public_supply_gw/model_npz_preorigin/v<N>`` (L3, 2007-01..2022-12): npz files in the
  ``calibrate_flow --extra-abstraction`` layout (q_m3d (4, 2148, 192) model units, dates, dx,
  n_active), per variant, in two vintages: 'strict' (documents published by 2022-12; 2022
  held at 2020) and 'exempt' (2022 measured, from documents published 2023-04 and 2024-04,
  declared as an exemption with the strict files as its sensitivity). The p6_proxy files carry
  P6's location exemptions. ``q_p6_proxy_plus_p6_other__*`` add P6's non-Taiwan-Water
  cohorts (irrigation agency, Taiwan Sugar, schools, factories, prison) to the measured Taiwan
  Water abstraction: the like-for-like replacement of R1e-pw's npz.

Usage: python -I build_x2.py [--dry-run <scratch_dir>] [--only-full --supersedes TEXT]
       python -I build_x2.py --use-full results/dataprep/public_supply_gw/v<N> --supersedes T
(a dry run writes the products and the ledger fragment under the scratch directory instead
of results/dataprep; --only-full builds the full product only, recording what it supersedes)
"""
from __future__ import annotations

import calendar
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger, manifest  # noqa: E402
from common.paths import ORIGIN, rel_to_repo  # noqa: E402

from x2_public_supply import drought_wells as DWL  # noqa: E402
from x2_public_supply import grid as G  # noqa: E402
from x2_public_supply import series as S  # noqa: E402
from x2_public_supply import x2const as K  # noqa: E402

VARIANTS = ("county_uniform", "p6_proxy", "service_area")
K1_RATIO = 2.469         # P6's pre-origin recomputation of the WRA-to-model ratio (sensitivity)
CODE = sorted(str(p) for p in K.CODE_DIR.glob("*.py"))


def months_axis(t0: str, t1: str) -> list[str]:
    return [str(p) for p in pd.period_range(t0, t1, freq="M")]


def county_rate(monthly: pd.DataFrame, county: str, months: list[str]) -> np.ndarray:
    s = monthly[monthly.county == county].set_index("month").rate_m3d
    return s.reindex(months).to_numpy(float)


def system_rates(res: dict, months: list[str]) -> dict[str, np.ndarray]:
    """Changhua per-system m3/day: the county monthly groundwater split by each year's
    system groundwater shares (表3, purchased raw water assigned as in series.changhua)."""
    cm = res["monthly"][res["monthly"].county == "changhua"].set_index("month").rate_m3d
    sy = res["systems"]
    out = {k: np.zeros(len(months)) for k in K.CHANGHUA_SYSTEMS}
    for i, mth in enumerate(months):
        y = int(mth[:4])
        rows = sy[sy.year == y]
        tot = rows.groundwater_m3.sum()
        for r in rows.itertuples():
            out[r.system][i] = cm[mth] * r.groundwater_m3 / tot
    return out


def weights(g: dict, m: dict, tid, info) -> dict:
    """County-uniform: per km2 over the county, the out-of-grid share dropped (weights sum to
    the county's active-area fraction; P6's loc_uniform_county put the whole county volume in
    grid). P6 proxy: P6's Taiwan Water cohort weights. Service area: Changhua per system,
    Yunlin as county-uniform."""
    sw = G.service_weights(g, tid, info)
    fr = {c: G.county_area_fraction(g, info, c) for c in ("changhua", "yunlin")}
    return {
        "county_uniform": {c: G.county_weights(g, c, fr[c]) for c in ("changhua", "yunlin")},
        "p6_proxy": {"changhua": G.tw_weights_p6(m, "changhua")[0],
                     "yunlin": G.tw_weights_p6(m, "yunlin")[0]},
        "service_area": {**sw, "yunlin": G.county_weights(g, "yunlin", fr["yunlin"])},
    }


def gridded(res: dict, months: list[str], W: dict) -> dict[str, np.ndarray]:
    rates = {c: county_rate(res["monthly"], c, months) for c in ("changhua", "yunlin")}
    sys_r = system_rates(res, months)
    out = {}
    for v in VARIANTS:
        r = {**sys_r, "yunlin": rates["yunlin"]} if v == "service_area" else rates
        out[v] = G.grid_rates(months, r, W[v])
    return out


def in_grid_annual(X: np.ndarray, months: list[str], cc: np.ndarray) -> pd.DataFrame:
    d = np.array([calendar.monthrange(int(t[:4]), int(t[5:]))[1] for t in months])
    yrs = np.array([int(t[:4]) for t in months])
    rows = []
    for county, zh in K.COUNTY_ZH.items():
        sel = cc == zh
        vol = (X[:, sel, :].sum((0, 1)) * d)
        for y in np.unique(yrs):
            rows.append({"county_of_cell": county, "year": int(y),
                         "Mm3": float(vol[yrs == y].sum() / 1e6)})
    other = ~np.isin(cc, list(K.COUNTY_ZH.values()))
    vol = (X[:, other, :].sum((0, 1)) * d)
    for y in np.unique(yrs):
        rows.append({"county_of_cell": "other_cells", "year": int(y),
                     "Mm3": float(vol[yrs == y].sum() / 1e6)})
    return pd.DataFrame(rows)


def p6_comparison(full: dict, m: dict, rb: dict, g: dict, Xfull: dict,
                  months_full: list[str]) -> tuple[pd.DataFrame, dict]:
    """County and in-grid volumes: measured (X2) against P6's Taiwan Water part."""
    a = full["annual"].set_index(["county", "year"]).value_Mm3
    ids = rb["ids"]
    vol = np.asarray(m["cohort_volume_1e4m3_per_yr"], float) / 100.0      # Mm3/yr
    yrs_c = np.asarray(m["cohort_year"], int)
    pend_vol = 53.30                     # P6: (7,172.25 - 1,842.3) x 10^4 m3/yr
    p6_dates = rb["dates"]
    yrs_m = np.array([int(t[:4]) for t in p6_dates])
    cc = g["cell_county"]
    pend = np.asarray(m["rate_phys_m3d__pending_tw_changhua"], float)
    rows = []
    for county in ("changhua", "yunlin"):
        sel_c = [i for i, c in enumerate(ids) if c.startswith(f"tw_{county}_")]
        zh = K.COUNTY_ZH[county]
        cells = cc == zh
        for y in range(2007, 2026):
            # county total of P6's Taiwan Water: cohorts still pumping (stop at July of the
            # disposal year -> mid-year mean), plus the pending Changhua wells to 2022
            frac = np.array([1.0 if y < yc else (0.5 if y == yc else 0.0)
                             for yc in yrs_c[sel_c]])
            p6_county = float((vol[sel_c] * frac).sum()) + (
                pend_vol if (county == "changhua" and y <= 2022) else 0.0)
            if y <= 2022:
                tw_in = float(rb["tw"][:, cells, :][:, :, yrs_m == y].sum((0, 1)).mean()
                              * 365.25 / 1e6)
                pe_in = float(pend[:, cells, :][:, :, yrs_m == y].sum((0, 1)).mean()
                              * 365.25 / 1e6)
            else:
                tw_in = pe_in = float("nan")
            row = {"county": county, "year": y, "measured_county_Mm3": a.get((county, y)),
                   "p6_tw_county_Mm3": p6_county if y <= 2022 else float("nan"),
                   "p6_tw_in_grid_Mm3": tw_in + (pe_in if county == "changhua" else 0.0)}
            for v in VARIANTS:
                df = in_grid_annual(Xfull[v], months_full, cc)
                row[f"measured_in_grid_{v}_Mm3"] = float(
                    df[(df.county_of_cell == county) & (df.year == y)].Mm3.iloc[0])
            row["measured_model_units_Mm3"] = row["measured_county_Mm3"] / K.WRA_TO_MODEL
            row["measured_model_units_k1_Mm3"] = row["measured_county_Mm3"] / K1_RATIO
            rows.append(row)
    df = pd.DataFrame(rows)
    summ = {}
    for county in ("changhua", "yunlin"):
        d = df[df.county == county].set_index("year")
        base_m = float(d.loc[2011:2014, "measured_county_Mm3"].mean())
        base_p = float(d.loc[2011:2014, "p6_tw_county_Mm3"].mean())

        def half_year(col: str, base: float, d: pd.DataFrame = d) -> int | None:
            s = d[col].dropna()
            hit = s[s < 0.5 * base]
            return int(hit.index[0]) if len(hit) else None

        summ[county] = {
            "measured_2011_2014_mean_Mm3": base_m, "p6_tw_2011_2014_mean_Mm3": base_p,
            "measured_2019_2022_mean_Mm3": float(d.loc[2019:2022,
                                                       "measured_county_Mm3"].mean()),
            "p6_tw_2019_2022_mean_Mm3": float(d.loc[2019:2022, "p6_tw_county_Mm3"].mean()),
            "measured_change_2011_2014_to_2019_2022_Mm3": float(
                d.loc[2019:2022, "measured_county_Mm3"].mean() - base_m),
            "p6_change_2011_2014_to_2019_2022_Mm3": float(
                d.loc[2019:2022, "p6_tw_county_Mm3"].mean() - base_p),
            "first_year_below_half_of_2011_2014_measured": half_year("measured_county_Mm3",
                                                                     base_m),
            "first_year_below_half_of_2011_2014_p6": half_year("p6_tw_county_Mm3", base_p),
            "measured_in_grid_2012_2021_mean_Mm3": {
                v: float(d.loc[2012:2021, f"measured_in_grid_{v}_Mm3"].mean())
                for v in VARIANTS},
            "p6_in_grid_2012_2021_mean_Mm3": float(d.loc[2012:2021,
                                                         "p6_tw_in_grid_Mm3"].mean()),
        }
    return df, summ


def extraction_table(tab: dict) -> pd.DataFrame:
    """Every yearbook number X2 uses, with yearbook, table, page and the check it passed."""
    rows = []
    for yb, t in sorted(tab.items()):
        f = f"twc_yearbook_{yb}.pdf"
        for d in (S.D11, S.D5):
            rec = t["monthly"]["district"].get(d)
            if rec:
                for i, v in enumerate(rec["months"]):
                    rows.append({"file": f, "table": "表2(1)", "pages": str(rec["pages"]),
                                 "row": d, "column": f"{i + 1}月", "data_year": yb + 1911,
                                 "value": v, "unit": "m3",
                                 "check": "months_sum_to_total; districts_sum_to_company"})
                rows.append({"file": f, "table": "表2(1)", "pages": str(rec["pages"]),
                             "row": d, "column": "總計", "data_year": yb + 1911,
                             "value": rec["total"], "unit": "m3", "check": "printed total"})
            sy = t["systems"]["districts"].get(d)
            if sy:
                rows.append({"file": f, "table": "表3", "pages": str(sy["page"]), "row": d,
                             "column": "出水量全年總計", "data_year": yb + 1911,
                             "value": sy["total"], "unit": "m3",
                             "check": "systems_sum_to_district; table3_equals_table2_1"})
                for k, v in sy["systems"].items():
                    rows.append({"file": f, "table": "表3", "pages": str(v["page"]), "row": k,
                                 "column": "出水量全年總計", "data_year": yb + 1911,
                                 "value": v["output"], "unit": "m3",
                                 "check": "systems_sum_to_district"})
            pr = t.get("purchases", {}).get("districts", {}).get(d)
            if pr:
                for k in ("raw_clear", "irrigation_raw", "central_raw", "south_raw",
                          "north_raw", "other_raw", "taipei_clear"):
                    if k in pr:
                        rows.append({"file": f, "table": "表4",
                                     "pages": str([pr.get("page1"), pr.get("page2")]),
                                     "row": d, "column": k, "data_year": yb + 1911,
                                     "value": pr[k], "unit": "1000 m3",
                                     "check": "purchases_sum_by_agency (rounding <= 3)"})
        for code, s in t["sources"]["systems"].items():
            rows.append({"file": f, "table": "表1", "pages": str(s["page"]),
                         "row": f"{code}{s['name']}", "column": "地下水 well count",
                         "data_year": yb + 1911, "value": s["well_count"], "unit": "wells",
                         "check": "groundwater_only=" + str(s["groundwater_only"])
                         + "; surface: " + (s["sw"] + " " + s["res"]).strip()})
        bd = t["by_district"]
        if yb == max(tab):
            for r, row in bd["values"].items():
                for d in (S.D11, S.D5):
                    if d in row and int(r) >= 96:
                        rows.append({"file": f, "table": "表2(2)",
                                     "pages": str(bd["column_pages"].get(d)), "row": d,
                                     "column": "annual", "data_year": int(r) + 1911,
                                     "value": row[d], "unit": "m3",
                                     "check": "cross_yearbook_same_value"})
    return pd.DataFrame(rows)


def save_model_npz(path: Path, q_phys: np.ndarray, months: list[str], g: dict, m: dict,
                   meta: dict) -> dict:
    q = q_phys / K.WRA_TO_MODEL
    if q.shape != (4, g["n_active"], len(months)):
        raise SystemExit(f"{path.name}: shape {q.shape}")
    if not (np.isfinite(q).all() and (q >= 0).all()):
        raise SystemExit(f"{path.name}: not finite and non-negative")
    np.savez_compressed(
        path, q_m3d=q.astype("float64"), dates=np.array(months), dx=np.float64(g["dx"]),
        n_active=np.int64(g["n_active"]), wra_to_model=np.float64(K.WRA_TO_MODEL),
        grid_hash=np.array(str(m["grid_hash"])), mask_hash=np.array(str(m["mask_hash"])),
        units=np.array("m3/day of abstraction per cell and twin layer, model units (physical "
                       "/ wra_to_model); positive = abstraction"),
        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    yrs = np.array([int(t[:4]) for t in months])
    d = np.array([calendar.monthrange(int(t[:4]), int(t[5:]))[1] for t in months])
    vol = (q.sum((0, 1)) * d)
    return {"file": path.name, "shape": list(q.shape),
            "model_Mm3_per_yr": {str(y): float(vol[yrs == y].sum() / 1e6)
                                 for y in (2010, 2012, 2015, 2018, 2019, 2020, 2021, 2022)},
            "dec2022_model_m3d_total": float(q[:, :, -1].sum())}


def l0_inputs() -> list[str]:
    files = sorted(str(p) for p in K.L0["twc_statistics"].glob("twc_yearbook_*.pdf"))
    files += [str(K.L0["twc_statistics"] / "SOURCE.json"),
              str(K.L0_MISC / "wra_cy_subsidence_results_110.pdf"),
              str(K.L0_MISC / "wra_cy_subsidence_results_112.pdf"),
              str(K.L0_MISC / "SOURCE.json"), str(K.TOWNS_TOPO),
              str(K.L0["wra_water_use_annual"] / "wra_wateruse_110_drought.pdf"),
              str(K.L0["wra_water_use_annual"] / "wra_wateruse_111_drought.pdf"),
              str(K.L0["wra_water_use_annual"] / "SOURCE.json"),
              str(K.L0_PLANS / "moea_yunchang_plan_phase2_main.pdf"),
              str(K.L0_PLANS / "moea_yunchang_plan_rev1_main.pdf"),
              str(K.L0_PLANS / "SOURCE.json")]
    return files


def main(root: Path | None = None, only_full: bool = False,
         supersedes: str | None = None, use_full: Path | None = None) -> None:
    prod_root = root if root is not None else manifest.DATAPREP
    g = G.engine_grid("model_design")
    m_after, man_after = G.p6_member("after_the_fact_analysis")
    rb = G.rebuild_p6(m_after)
    if rb["max_rel_err"] > 1e-6:
        raise SystemExit(f"P6 rebuild error {rb['max_rel_err']}")
    tid, info = G.town_of_cells(g)
    W = weights(g, m_after, tid, info)
    months_full = months_axis(K.FIRST_MONTH, K.LAST_MONTH)
    months_pre = months_axis(K.FIRST_MONTH, ORIGIN)
    if months_pre != rb["dates"]:
        raise SystemExit("pre-origin months differ from the P6 member's")
    res = {v: S.build(v) for v in ("full", "exempt", "strict")}
    Xfull = gridded(res["full"], months_full, W)
    # ---- checks
    checks = {"p6_rebuild_max_rel_err": rb["max_rel_err"]}
    for v in VARIANTS:
        X = Xfull[v]
        d = np.array([calendar.monthrange(int(t[:4]), int(t[5:]))[1] for t in months_full])
        tot = X.sum((0, 1)) * d
        mon = res["full"]["monthly"]
        want = np.zeros(len(months_full))
        for c in ("changhua", "yunlin"):
            frac = (sum(w.sum() for k, w in W[v].items() if k != "yunlin") if (
                v == "service_area" and c == "changhua") else W[v][c].sum())
            if v == "service_area" and c == "changhua":
                sysr = system_rates(res["full"], months_full)
                want += sum(sysr[k] * W[v][k].sum() for k in K.CHANGHUA_SYSTEMS) * d
                continue
            want += county_rate(mon, c, months_full) * frac * d
        checks[f"mass_balance_{v}_max_rel_err"] = float(np.max(np.abs(tot - want) /
                                                               np.maximum(want, 1.0)))
    for vin, r in res.items():
        a, mo = r["annual"], r["monthly"]
        s = mo.assign(year=mo.month.str[:4].astype(int)).groupby(["county", "year"]).value_m3.sum()
        diff = (s - a.set_index(["county", "year"]).value_m3).abs().max()
        checks[f"monthly_sums_to_annual_{vin}_max_abs_m3"] = float(diff)
        checks[f"non_negative_{vin}"] = bool((mo.value_m3 >= -1e-6).all())
    tw_tab = json.loads((K.EVID / "twc_tables.json").read_text(encoding="utf-8"))
    checks["twc_text_layer_checks"] = {"n": tw_tab["validation"]["n_checks"],
                                       "failed": tw_tab["validation"]["n_failed"]}
    chart = json.loads((K.EVID / "chart_reconciliation.json").read_text(encoding="utf-8"))
    checks["chart_reads"] = {k: chart[k] for k in ("n_points", "n_disagree",
                                                   "cross_vintage_disagree", "n1_disagree")}
    # external statements (validators)
    a = res["full"]["annual"].set_index(["county", "year"])
    ext = {
        "phase2_plan_2019_changhua_gw_Mm3": {"printed": 99.0, "x2": float(
            a.loc[("changhua", 2019), "value_Mm3"]),
            "source": "moea_yunchang_plan_phase2_main.pdf p.29-30: 108年 ... 地下水約27萬噸/日"
                      "(0.99億噸/年)"},
        "phase2_plan_2019_yunlin_gw_Mm3": {"printed": 3.0, "x2": float(
            a.loc[("yunlin", 2019), "value_Mm3"]),
            "source": "moea_yunchang_plan_phase2_main.pdf p.30: 地下水(0.9萬噸/日，0.03億噸/年，"
                      "備援水井維護之抽水量)"},
        "rev1_plan_baseline_changhua_gw_Mm3": {"printed": 103.0, "x2_2010_2012_mean": float(
            np.mean([a.loc[("changhua", y), "value_Mm3"] for y in (2010, 2011, 2012)])),
            "source": "moea_yunchang_plan_rev1_main.pdf (text line 1648): 每日28萬噸 ... 約為每年"
                      "1.03億噸"},
        "rev1_plan_baseline_yunlin_gw_Mm3": {"printed": 46.0, "x2_2011_2012_mean": float(
            np.mean([a.loc[("yunlin", y), "value_Mm3"] for y in (2011, 2012)])),
            "source": "moea_yunchang_plan_rev1_main.pdf (text line 416): 每日抽取12.6萬噸(0.46億噸"
                      "/年)地下水"},
    }
    dwp = DWL.parse()
    for r in dwp["values"]:
        if r["county"] == "yunlin":
            ext[f"drought_wells_{r['year']}_yunlin_Mm3_lower_bound"] = {
                "printed": r["volume_1e4m3"] / 100, "x2": float(
                    a.loc[("yunlin", r["year"]), "value_Mm3"]), "source": r["source"],
                "period": r["period"], "published": r["published"]}
        else:
            ext[f"drought_wells_{r['year']}_changhua_Mm3_part_of_output"] = {
                "printed": r["volume_1e4m3"] / 100, "x2": float(
                    a.loc[("changhua", r["year"]), "value_Mm3"]), "source": r["source"],
                "period": r["period"], "published": r["published"]}
    checks["drought_well_tables"] = {"checks": dwp["checks"], "issues": dwp["issues"]}
    for v in ext.values():
        x = v.get("x2", v.get("x2_2010_2012_mean", v.get("x2_2011_2012_mean")))
        v["rel_diff"] = (x - v["printed"]) / v["printed"]
    yl = res["full"]["annual"][res["full"]["annual"].county == "yunlin"]
    e12 = yl.dropna(subset=["E1_m3", "E2_m3"])
    e12 = e12[~e12.drought_year.astype(bool)]
    checks["yunlin_E1_minus_E2_normal_years_Mm3"] = {
        str(int(r.year)): float((r.E1_m3 - r.E2_m3) / 1e6) for r in e12.itertuples()}
    gate_ok = (checks["twc_text_layer_checks"]["failed"] == 0
               and checks["chart_reads"]["n_disagree"] == 0
               and checks["chart_reads"]["cross_vintage_disagree"] == 0
               and all(checks[f"mass_balance_{v}_max_rel_err"] < 1e-6 for v in VARIANTS)
               and all(checks[f"monthly_sums_to_annual_{v}_max_abs_m3"] < 1.0
                       for v in res)
               and all(checks[f"non_negative_{v}"] for v in res)
               and abs(ext["phase2_plan_2019_changhua_gw_Mm3"]["rel_diff"]) < 0.05
               and abs(ext["rev1_plan_baseline_changhua_gw_Mm3"]["rel_diff"]) < 0.05
               and all(c["ok"] for c in dwp["checks"] if "全台" not in c["check"]))
    comp, comp_summ = p6_comparison(res["full"], m_after, rb, g, Xfull, months_full)
    if use_full is not None:          # reuse a built full product (pre-origin rebuild)
        dA = use_full
        valA = json.loads((dA / "validation.json").read_text(encoding="utf-8"))
    else:
        # ---- product A (full)
        dA = manifest.new_version_dir("public_supply_gw", root=prod_root)
        res["full"]["annual"].to_csv(dA / "county_annual.csv", index=False)
        res["full"]["monthly"].to_csv(dA / "county_monthly.csv", index=False)
        res["full"]["systems"].to_csv(dA / "changhua_systems_annual.csv", index=False)
        extraction_table(res["full"]["tables"]).to_csv(dA / "twc_extracted_numbers.csv",
                                                       index=False)
        shutil.copy2(K.EVID / "wra_janmay_controlled_wells.csv",
                     dA / "wra_janmay_controlled_wells.csv")
        comp.to_csv(dA / "p6_comparison.csv", index=False)
        info.to_csv(dA / "township_grid_areas.csv", index=False)
        flagA = {c: res["full"]["monthly"][res["full"]["monthly"].county == c].set_index(
            "month").flag.reindex(months_full).to_numpy(np.uint8) for c in ("changhua", "yunlin")}
        np.savez_compressed(
            dA / "grid_full.npz", dates=np.array(months_full), dx=np.float64(g["dx"]),
            n_active=np.int64(g["n_active"]), mask_hash=np.array(str(m_after["mask_hash"])),
            grid_hash=np.array(str(m_after["grid_hash"])),
            **{f"rate_phys_m3d__{v}": Xfull[v].astype("float32") for v in VARIANTS},
            flag_changhua=flagA["changhua"], flag_yunlin=flagA["yunlin"],
            wra_to_model=np.float64(K.WRA_TO_MODEL),
            units=np.array("m3/day of abstraction (positive), physical; model units = physical / "
                           "wra_to_model (an assumption kept from P6)"),
            layer_weights=np.array(K.LAYER_WEIGHTS))
        in_grid = {v: in_grid_annual(Xfull[v], months_full, g["cell_county"]).to_dict("records")
                   for v in VARIANTS}
        valA = {"passed": gate_ok, "checks": checks, "external_statements": ext,
                "p6_comparison_summary": comp_summ,
                "in_grid_fraction": {v: {k: float(w.sum()) for k, w in W[v].items()}
                                     for v in VARIANTS},
                "county_area_fraction_in_active_grid": {
                    c: G.county_area_fraction(g, info, c) for c in ("changhua", "yunlin")},
                "in_grid_annual_Mm3": in_grid}
        (dA / "validation.json").write_text(json.dumps(valA, indent=1, ensure_ascii=False,
                                                       default=float))
        (dA / "PRODUCT_CARD.md").write_text(product_card(), encoding="utf-8")
        exA = []
        mA = manifest.build_manifest(
            dA, product="public_supply_gw", layer="L2",
            inputs=l0_inputs() + [str(K.P6_MEMBER / "sensitivity_member.npz"),
                                  str(K.P6_MEMBER / "MANIFEST.json"),
                                  str(K.ENGINE / "MANIFEST.json"),
                                  str(K.EVID / "twc_tables.json"),
                                  str(K.EVID / "chart_reconciliation.json")],
            input_trees=[str(K.API_CACHE)],
            code=CODE,
            parameters={"variants": list(VARIANTS), "layer_weights": list(K.LAYER_WEIGHTS),
                        "wra_to_model": K.WRA_TO_MODEL, "drought_years": list(K.DROUGHT_YEARS),
                        "sd_plant_frac": K.SD_PLANT_FRAC, "sd_riverbed_2021": K.SD_RIVERBED_2021,
                        "service_towns": {k: list(v) for k, v in K.SERVICE_TOWNS.items()},
                        "months": [months_full[0], months_full[-1]],
                        "scope": "Taiwan Water (台水) public-supply groundwater only; other public "
                                 "wells (irrigation agency, Taiwan Sugar, schools) are not in it"},
            fit_window_end=K.LAST_MONTH, validation_report="validation.json",
            gate={"passed": gate_ok, "criteria": "yearbook text-layer arithmetic (months, "
                  "districts, systems, purchases, cross-yearbook) all pass; chart reads agree "
                  "(two models, two briefings); grid mass balance < 1e-6; monthly sums equal "
                  "annual; Changhua 2019 and baseline within 5 % of the action plans' figures"},
            allowed_uses=["after_the_fact_analysis", "forcing_recorded_arm",
                          "statistical_layer_history", "validator"],
            forbidden_uses=["scoring_target", "baseline", "datum_fit", "prospective_scorer",
                            "calibration_target", "selection", "forcing_frozen", "prior",
                            "model_design", "initial_conditions", "idw_field", "prior_width_input",
                            "design_time_evidence"],
            exemptions=exA,
            extra={**({"supersedes": supersedes} if supersedes else {}),
                   "sources_publication": {"twc_yearbooks": "96-114 (data 2007-2025; 111 "
                                           "published 2023-04)", "wra_briefing_110": "2022-10",
                                           "wra_briefing_112": "2024-04"},
                   "note": "Measured Taiwan Water output and purchases; the Yunlin series is "
                           "inferred (flag 5). Post-origin values feed only the recorded-forcing "
                           "arm and after-the-fact analysis (U4)."})
        manifest.write_manifest(dA, mA)
    if only_full:
        out_dir = K.REPORT_DIR if root is None else root
        summ = {"product_full": rel_to_repo(dA), "gate_passed": gate_ok, "checks": checks,
                "external": ext, "p6_comparison": comp_summ,
                "in_grid_fraction": valA["in_grid_fraction"],
                "county_area_fraction": valA["county_area_fraction_in_active_grid"],
                "supersedes": supersedes}
        (out_dir / f"build_summary_{dA.name}.json").write_text(
            json.dumps(summ, indent=1, ensure_ascii=False, default=float))
        print(json.dumps({"product_full": summ["product_full"], "gate_passed": gate_ok}))
        return
    # ---- product B (pre-origin npz)
    m_pre, man_pre = G.p6_member("forcing_frozen")
    rb_pre = G.rebuild_p6(m_pre)
    dB = manifest.new_version_dir("public_supply_gw/model_npz_preorigin", root=prod_root)
    outs, files_info = [], []
    p6_ex = [{k: v for k, v in e.items() if k != "inherited_from"}
             | {"inherited_from": "model_ready/release_20261012b2/preorigin/"
                                  "public_wells_member/v1"} for e in man_pre["exemptions"]]
    p6_end = max(e["data_end"] for e in p6_ex)
    ex2022 = {"input": "data_fetch_api/new_measurements/twc_statistics/2026-10-21/"
                       "twc_yearbook_111.pdf (Taiwan Water 2022 output and purchases; listed "
                       "2023-04-28) and data_fetch_api/new_measurements/wra_reports_misc/"
                       "wra_cy_subsidence_results_112.pdf (Yunlin Taiwan Water Jan-May 2022; "
                       "created 2024-04) and data_fetch_api/new_measurements/"
                       "wra_water_use_annual/2026-10-21/wra_wateruse_111_drought.pdf (2020 and "
                       "2022 drought-well volumes, a lower bound for Yunlin; created 2023-12)",
              "data_end": "2024-04",
              "reason": "the only public records of 2022 Taiwan Water abstraction; the values "
                        "are 2022 facts printed in documents published after the origin",
              "ledger_number_id": "X2.v1.exemption.values_2022_published_after_origin",
              "sensitivity": "the '__strict' file of the same variant: 2022 held at 2020 "
                             "from documents published by 2022-12 (validation.json "
                             "exemption_sensitivity)",
              "uses": ["forcing_frozen", "model_design", "design_time_evidence"]}
    sens = {}
    for vin in ("strict", "exempt"):
        Xpre = gridded(res[vin], months_pre, W)
        for v in VARIANTS:
            meta = {"variant": v, "vintage": vin, "components": ["taiwan_water_measured"],
                    "source_product": "public_supply_gw (same build)"}
            name = f"q_{v}__{vin}.npz"
            files_info.append(save_model_npz(dB / name, Xpre[v], months_pre, g, m_pre, meta))
            fwe = ORIGIN if vin == "strict" else "2024-04"
            if v == "p6_proxy":
                fwe = max(fwe, p6_end)
            outs.append({"path": name, "fit_window_end": fwe})
        name = f"q_p6_proxy_plus_p6_other__{vin}.npz"
        meta = {"variant": "p6_proxy", "vintage": vin,
                "components": ["taiwan_water_measured", "p6_primary_non_taiwan_water_cohorts"]}
        files_info.append(save_model_npz(dB / name, Xpre["p6_proxy"] + rb_pre["other"],
                                         months_pre, g, m_pre, meta))
        outs.append({"path": name, "fit_window_end": max(
            ORIGIN if vin == "strict" else "2024-04", p6_end)})
        sens[vin] = {v: float(Xpre[v][:, :, -12:].sum() / 12) for v in VARIANTS}
    valB = {"passed": gate_ok, "files": files_info,
            "exemption_sensitivity": {
                "2022_mean_physical_m3d_in_grid": sens,
                "note": "strict holds 2022 at 2020; exempt uses the 2022 measurements"},
            "r1e_pw_reference": "public_wells/model_npz/v1 (P6 primary + pending, model units)",
            "loader_layout": "inputs.load_extra_abstraction: q_m3d (4, n_active, months), "
                             "dates, dx, n_active; months after 2022-12 hold 2022-12",
            "p6_rebuild_max_rel_err": rb_pre["max_rel_err"]}
    (dB / "validation.json").write_text(json.dumps(valB, indent=1, ensure_ascii=False))
    exB = [ex2022] + p6_ex
    mB = manifest.build_manifest(
        dB, product="public_supply_gw/model_npz_preorigin", layer="L3",
        inputs=l0_inputs() + [str(K.P6_MEMBER / "sensitivity_member.npz"),
                              str(K.P6_MEMBER / "MANIFEST.json"),
                              str(K.ENGINE / "MANIFEST.json"),
                              str(K.EVID / "twc_tables.json"),
                              str(K.EVID / "chart_reconciliation.json"),
                              str(dA / "MANIFEST.json")],
        code=CODE,
        parameters={"variants": list(VARIANTS), "vintages": ["strict", "exempt"],
                    "units": "model (physical / wra_to_model)", "wra_to_model": K.WRA_TO_MODEL,
                    "strict_hold_year": S.HOLD_YEAR, "months": [months_pre[0], months_pre[-1]],
                    "note": "measured volumes, proxy locations (variants); P6 restriction "
                            "(sensitivity member, never the primary model) applies to the "
                            "p6_proxy_plus_p6_other files' P6 part"},
        fit_window_end=max("2024-04", p6_end), non_exempt_fit_window_end=ORIGIN,
        exemptions=exB, outputs=outs + [{"path": "validation.json"}],
        validation_report="validation.json",
        gate={"passed": gate_ok, "criteria": "as public_supply_gw; shape (4, 2148, 192), "
              "finite, >= 0; months equal the P6 member's"},
        allowed_uses=["forcing_frozen", "model_design", "design_time_evidence",
                      "after_the_fact_analysis"],
        forbidden_uses=["scoring_target", "baseline", "datum_fit", "prospective_scorer",
                        "calibration_target", "selection", "statistical_layer_history",
                        "forcing_recorded_arm", "idw_field", "initial_conditions"],
        extra={"supersedes": supersedes} if supersedes else None)
    manifest.write_manifest(dB, mB)
    summary = {"product_full": rel_to_repo(dA), "product_preorigin": rel_to_repo(dB),
               "gate_passed": gate_ok, "checks": checks, "external": ext,
               "p6_comparison": comp_summ, "files_preorigin": files_info,
               "in_grid_fraction": valA["in_grid_fraction"],
               "county_area_fraction": valA["county_area_fraction_in_active_grid"]}
    out_dir = K.REPORT_DIR if root is None else root
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=1,
                                                           ensure_ascii=False, default=float))
    write_ledger(dA, dB, None if root is None else root / "ledger_rows")
    print(json.dumps({k: summary[k] for k in ("product_full", "product_preorigin",
                                              "gate_passed")}, indent=1))


def write_ledger(dA: Path, dB: Path, rows_dir: Path | None = None) -> None:
    vA, vB = rel_to_repo(dA), rel_to_repo(dB)
    yb = "data_fetch_api/new_measurements/twc_statistics/2026-10-21"
    misc = "data_fetch_api/new_measurements/wra_reports_misc"
    plans = "data_fetch_api/new_measurements/wra_subsidence_action_plans"
    rows = [
        {"number_id": "X2.changhua.tw_groundwater_annual", "source": yb, "role": "input",
         "product_version": vA, "note": "TWC yearbooks 表2/表3/表4 (11th district)"},
        {"number_id": "X2.yunlin.tw_groundwater_annual_E2",
         "source": f"{misc}/wra_cy_subsidence_results_110.pdf", "role": "input",
         "product_version": vA, "note": "Jan-May Taiwan Water public-well pumping 2011-2021"},
        {"number_id": "X2.yunlin.tw_groundwater_annual_E2",
         "source": f"{misc}/wra_cy_subsidence_results_112.pdf", "role": "input",
         "product_version": vA, "note": "Jan-May 2022-2023"},
        {"number_id": "X2.yunlin.tw_groundwater_annual_E2", "source": yb, "role": "input",
         "product_version": vA, "note": "Changhua annual/Jan-May ratio (seasonality transfer)"},
        {"number_id": "X2.yunlin.tw_groundwater_annual_E1",
         "source": f"{yb}/twc_yearbook_114.pdf", "role": "validator", "product_version": vA,
         "note": "E1 residual (systems output minus purchased raw water) checks E2"},
        {"number_id": "X2.check.phase2_plan_2019",
         "source": f"{plans}/moea_yunchang_plan_phase2_main.pdf", "role": "validator",
         "product_version": vA, "note": "2019 Changhua 0.99 and Yunlin 0.03 x 10^8 m3/yr"},
        {"number_id": "X2.check.rev1_plan_baseline",
         "source": f"{plans}/moea_yunchang_plan_rev1_main.pdf", "role": "validator",
         "product_version": vA, "note": "baseline Changhua 1.03, Yunlin 0.46 x 10^8 m3/yr"},
        {"number_id": "X2.check.chart_reads_n1",
         "source": "results/dataprep/reports/P6_evidence/after_the_fact_n1_janmay.json",
         "role": "validator", "product_version": vA, "note": "N1's reads of the 110 charts"},
        {"number_id": "X2.p6_comparison", "source": rel_to_repo(K.P6_MEMBER), "role": "input",
         "product_version": vA, "note": "P6 member (Taiwan Water cohorts, weights)"},
        {"number_id": "X2.grid.weights", "source": rel_to_repo(K.ENGINE), "role": "input",
         "product_version": vA, "note": "active cells and cell counties"},
        {"number_id": "X2.grid.weights", "source": rel_to_repo(K.TOWNS_TOPO), "role": "input",
         "product_version": vA, "note": "service-area variant townships"},
        {"number_id": "X2.v1.exemption.values_2022_published_after_origin",
         "source": f"{yb}/twc_yearbook_111.pdf", "role": "input", "product_version": vB,
         "note": "exemption: 2022 values published 2023-04"},
        {"number_id": "X2.v1.exemption.values_2022_published_after_origin",
         "source": f"{misc}/wra_cy_subsidence_results_112.pdf", "role": "input",
         "product_version": vB, "note": "exemption: 2022 Jan-May Yunlin, created 2024-04"},
    ]
    if rows_dir is None:
        ledger.append_rows("X2", rows)
    else:
        ledger.append_rows("X2", rows, rows_dir=rows_dir)


def product_card() -> str:
    return """# public_supply_gw: Taiwan Water groundwater abstraction, Changhua and Yunlin (WP X2)

What it is: the groundwater that Taiwan Water (台水) pumps for public supply, by county,
2007-2025, monthly and annual, with sd and flags, and gridded on the twin grid.

- Changhua: the 11th district's plant output (all its systems draw on wells until 2022)
  minus the raw water it buys (表4; the 鳥嘴潭 lake water from 2023). Flags 0-1 (measured).
- Yunlin: the WRA briefings' measured Jan-May pumping of the Yunlin Taiwan Water wells times
  the same year's Changhua annual/Jan-May ratio; in the drought years 2021 and 2023 the
  output-minus-purchases residual. Flag 5 (inferred). Before 2011: held at 2011 (flag 6).
- Grid variants (grid_full.npz, physical m3/day, layers 3-4 split 0.5/0.5):
  county_uniform (per km2 over the county, out-of-grid share dropped), p6_proxy (P6's
  Taiwan Water drought-well locations; drops the out-of-grid share), service_area
  (Changhua per system over its service townships; Yunlin county-uniform).
- Model units: divide by 2.58 (P6's stated assumption, unchanged; K1 2.47 as a sensitivity).
- Uses: after-the-fact analysis, recorded-forcing arm, statistical-layer history, validator.
  Pre-origin files are in public_supply_gw/model_npz_preorigin.
- Not included: irrigation-agency, Taiwan Sugar and school public wells (see P6), industrial
  self-supply.
"""


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", default=None)
    ap.add_argument("--only-full", action="store_true")
    ap.add_argument("--supersedes", default=None)
    ap.add_argument("--use-full", default=None, help="existing public_supply_gw/v<N> to reuse")
    a = ap.parse_args()
    main(Path(a.dry_run) if a.dry_run else None, a.only_full, a.supersedes,
         Path(a.use_full).resolve() if a.use_full else None)
