"""WP P1 round 2: score the frozen v1 inference against the measured census.

Implements exactly the definitions of the pre-registrations (written 2026-10-12T05:58Z,
before this script first ran):

* ``P1r2.census_confirmations`` (declaration): C1 a vs census, C2 misspecification link,
  C4 phi_in, C5 in-grid anchor, C8 coverage (validator role), C9 Yunlin consistency,
  T12 monthly shape, T13 ensemble rank;
* ``P1r2.census_tests`` (test): T10 Yunlin extension, T11 Changhua extension, 2025-07..2026-06
  under the pre-registered month rule (``harness.prereg.assert_can_score`` is called first,
  which also logs the scoring look).

Inputs: the census-truth build (``census_truth.py``), the v1 product files, and the observed
TPC table (opened through the loader guard for after-the-fact analysis).

Usage: python -I r2_evaluate.py <truth_dir> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lib  # noqa: E402
from common import guard  # noqa: E402
from common.paths import REPO  # noqa: E402
from harness import prereg as pr  # noqa: E402

CE = REPO / "results/dataprep/county_electricity"
V1P, V1F = CE / "preorigin/v1", CE / "full/v1"
OBS = CE / "tpc_observed/v1"
PRE_WIN, FULL_WIN = ("2016-01-01", "2022-12-01"), ("2016-01-01", "2025-06-01")
LONG = ("2012-01-01", "2022-12-01")
EXT = ("2025-07-01", "2026-06-01")
AGRI = ("irrigation", "aquaculture", "livestock")
Z90 = 1.6448536269514722


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def truth_series(tm: pd.DataFrame, county, variant, series="clean", scope="county",
                 classes=None, exclude=None) -> pd.Series:
    x = tm[(tm.county == county) & (tm.variant == variant) & (tm.series == series)
           & (tm.scope == scope)]
    if classes is not None:
        x = x[x["class"].isin(classes)]
    if exclude is not None:
        x = x[~x["class"].isin(exclude)]
    s = x.groupby("date")["gwh"].sum()
    s.index = pd.DatetimeIndex(s.index, name="date")
    return s.sort_index()


def wsum(s: pd.Series, w) -> float:
    return float(s.loc[w[0]:w[1]].sum())


def moy_cv(s: pd.Series) -> float:
    m = s.groupby(s.index.month).mean()
    return float(m.std(ddof=1) / m.mean())


def deseason(s: pd.Series) -> pd.Series:
    return s - s.groupby(s.index.month).transform("mean")


def compare(est: pd.Series, obs: pd.Series, w) -> dict:
    j = pd.concat([est.rename("e"), obs.rename("o")], axis=1).loc[w[0]:w[1]].dropna()
    if len(j) < 3:
        return {"n": int(len(j))}
    rel = (j.e - j.o) / j.o
    ann = lib.annual_err_pct(j.e, j.o)
    return {"n": int(len(j)), "window": [w[0][:7], w[1][:7]],
            "level_ratio": float(j.e.sum() / j.o.sum()),
            "monthly_rel_rms": float(np.sqrt(np.mean(rel ** 2))),
            "monthly_rel_bias": float(rel.mean()),
            "corr": float(np.corrcoef(j.e, j.o)[0, 1]),
            "corr_deseasonalised": float(np.corrcoef(deseason(j.e), deseason(j.o))[0, 1]),
            "moy_cv_est": moy_cv(j.e), "moy_cv_obs": moy_cv(j.o),
            "moy_cv_ratio": moy_cv(j.e) / moy_cv(j.o),
            "annual_err_pct": {str(k): float(v) for k, v in ann.items()},
            "annual_rms_pct": float(np.sqrt(np.mean(ann.to_numpy() ** 2)))}


def cand(path: Path, estimator: str, scope: str = "county", col: str = "value") -> pd.Series:
    c = pd.read_parquet(path)
    x = c[(c.estimator == estimator) & (c.scope == scope)].set_index("date")[col]
    x.index = pd.DatetimeIndex(x.index, name="date")
    return x.sort_index()


def wilson(k: int, n: int) -> list[float]:
    lo, hi = lib.wilson(k, n)
    return [lo, hi]


def ext_test(pred: pd.Series, sd: pd.Series, obs: pd.Series, months) -> dict:
    j = pd.concat([pred.rename("p"), sd.rename("sd"), obs.rename("o")], axis=1).loc[months]
    rel = (j.p - j.o) / j.o
    inside = (j.p - j.o).abs() <= Z90 * j.sd
    k, n = int(inside.sum()), int(len(j))
    wl = wilson(k, n)
    err = float(j.p.sum() / j.o.sum() - 1.0)
    return {"months": [str(m.date())[:7] for m in j.index], "n": n,
            "sum_pred": float(j.p.sum()), "sum_obs": float(j.o.sum()),
            "sum_err": err, "pass": bool(abs(err) <= 0.10),
            "monthly_rel_err": {str(m.date())[:7]: float(v) for m, v in rel.items()},
            "monthly_rel_rms": float(np.sqrt(np.mean(rel ** 2))),
            "coverage_1.645sd": {"inside": k, "n": n, "point": k / n if n else None,
                                 "wilson95": wl, "contains_0.90": bool(wl[0] <= 0.90 <= wl[1])}}


def main(truth_dir: str, out_dir: str) -> int:
    td, od = Path(truth_dir), Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    tm = pd.read_parquet(td / "truth_monthly.parquet")
    mc = pd.read_parquet(td / "meter_counts.parquet")
    tinfo = json.loads((td / "truth_info.json").read_text(encoding="utf-8"))
    dec = {v: json.loads((CE / v / "v1/decomposition.json").read_text(encoding="utf-8"))
           for v in ("preorigin", "full")}
    val = {v: json.loads((CE / v / "v1/validation.json").read_text(encoding="utf-8"))
           for v in ("preorigin", "full")}
    sep = {v: dec[v]["source_separation"] for v in dec}
    irr = {v: dec[v]["irrigation_only"] for v in dec}
    ev = {"prereg": {}, "truth_info": {k: tinfo[k] for k in ("grid",
                                                               "preorigin_vs_full_le_2022_12")}}
    for pid in ("P1r2.census_confirmations", "P1r2.census_tests"):
        doc, sha = pr.load_prereg(pid)
        ev["prereg"][pid] = {"sha256": sha, "created": doc["created"]}

    T = {}
    for cty in ("changhua", "yunlin"):
        for var in ("preorigin", "full"):
            T[cty, var, "all"] = truth_series(tm, cty, var)
            T[cty, var, "irr"] = truth_series(tm, cty, var, classes=["irrigation"])
            T[cty, var, "nodom"] = truth_series(tm, cty, var, exclude=["domestic"])
            T[cty, var, "in_all"] = truth_series(tm, cty, var, scope="in_grid")
            T[cty, var, "in_irr"] = truth_series(tm, cty, var, scope="in_grid",
                                                 classes=["irrigation"])
        T[cty, "full", "dedup_agri"] = truth_series(tm, cty, "full", "dedup", classes=AGRI)
        T[cty, "full", "dedup_all"] = truth_series(tm, cty, "full", "dedup")
        T[cty, "full", "dedup_in_all"] = truth_series(tm, cty, "full", "dedup", "in_grid")

    # ------------------------------------------------------------------ C9 consistency
    yl1 = pd.read_parquet(V1P / "yunlin_census_class_monthly.parquet")
    ylf = pd.read_parquet(V1F / "yunlin_census_class_monthly.parquet")

    def v1_series(df, name):
        s = df[df.series == name].set_index("date")["value"]
        s.index = pd.DatetimeIndex(s.index, name="date")
        return s.sort_index()
    c9 = {}
    for var, df, last in (("preorigin", yl1, "2022-12-01"), ("full", ylf, "2025-06-01")):
        for nm, key in (("clean_total", "all"), ("clean_irrigation", "irr")):
            a = v1_series(df, nm).loc[:last]
            b = T["yunlin", var, key].reindex(a.index).fillna(0.0)
            c9[f"{var}:{nm}"] = {"n": int(len(a)), "max_abs_GWh": float((a - b).abs().max())}
    c9["identical"] = all(v["max_abs_GWh"] <= 1e-6 for v in c9.values())
    ev["C9_yunlin_consistency"] = c9

    # ------------------------------------------------------------------ C1 a vs census
    def a_star(var, w, key="all", ykey=None):
        return wsum(T["changhua", var, key], w) / wsum(T["yunlin", var, ykey or key], w)

    def reading(x, raw, rem):
        if raw[0] <= x <= raw[1]:
            return "confirmed"
        if rem[0] <= x <= rem[1]:
            return "confirmed only by the reported remedy interval"
        return "not confirmed"
    pre_conf = json.loads((REPO / "results/dataprep/preregistration/"
                           "P1r2.census_confirmations.json").read_text(encoding="utf-8"))
    th = pre_conf["thresholds"]["C1_a_vs_census"]
    c1 = {}
    for var, w, key in (("preorigin", PRE_WIN, "v1_preorigin"), ("full", FULL_WIN, "v1_full")):
        x = a_star(var, w)
        raw, rem = th[key]["raw90"], th[key]["remedy90"]
        c1[var] = {"window": [w[0][:7], w[1][:7]], "a_star": x, "a_v1": sep[var]["a"],
                   "a_v1_over_a_star": sep[var]["a"] / x, "raw90": raw, "remedy90": rem,
                   "reading": reading(x, raw, rem),
                   "irrigation": {"a_star_irr": a_star(var, w, "irr"), "a_irr_v1": irr[var]["a"],
                                  "raw90_v1": [irr[var]["a_lo"], irr[var]["a_hi"]],
                                  "inside_raw90": bool(irr[var]["a_lo"] <= a_star(var, w, "irr")
                                                       <= irr[var]["a_hi"])},
                   "changhua_domestic_excluded_over_yunlin_all": a_star(var, w, "nodom", "all"),
                   "both_domestic_excluded": a_star(var, w, "nodom", "nodom")}
    c1["preorigin"]["window_2012_2022"] = a_star("preorigin", LONG)
    ev["C1_a_vs_census"] = c1

    # ------------------------------------------------------------------ C2
    mis = val["preorigin"]["tests"]["T9_MIS"]["preorigin"]["cases"]
    ch, yl = T["changhua", "preorigin", "all"].loc[LONG[0]:LONG[1]], \
        T["yunlin", "preorigin", "all"].loc[LONG[0]:LONG[1]]
    chi, yli = T["changhua", "preorigin", "irr"].loc[LONG[0]:LONG[1]], \
        T["yunlin", "preorigin", "irr"].loc[LONG[0]:LONG[1]]
    ev["C2_misspecification_link"] = {
        "a_star_minus_a_pre": c1["preorigin"]["a_star"] - sep["preorigin"]["a"],
        "moy_cv_all": {"changhua": moy_cv(ch), "yunlin": moy_cv(yl),
                       "ratio": moy_cv(ch) / moy_cv(yl)},
        "moy_cv_irrigation": {"changhua": moy_cv(chi), "yunlin": moy_cv(yli),
                              "ratio": moy_cv(chi) / moy_cv(yli)},
        "v1_T9_bias_a_hat_minus_a_eff": {k: v["bias_vs_a_eff"] for k, v in mis.items()},
        "note": "a negative T9 bias means the NNLS under-estimates a when Changhua's pump "
                "season is flatter (M1) or later (M3) than Yunlin's"}

    # ------------------------------------------------------------------ C4 phi_in
    phi = dec["preorigin"]["in_grid_anchor"]["phi_in"]
    sphi = dec["preorigin"]["in_grid_anchor"]["sigma_ln_phi"]
    phis = wsum(T["changhua", "preorigin", "in_all"], LONG) / \
        wsum(T["changhua", "preorigin", "all"], LONG)
    lnr = float(np.log(phis / phi))
    cc = tinfo["counties"]["changhua"]
    ev["C4_phi_in"] = {
        "phi_star": phis, "phi_in_v1": phi, "sigma_ln_phi": sphi, "ln_ratio": lnr,
        "z": lnr / sphi, "band_halfwidth_ln": Z90 * sphi,
        "inside_band": bool(abs(lnr) <= Z90 * sphi),
        "pump_count_fraction_all_pumps": cc["pumps_in_grid_frac"],
        "pump_count_fraction_kept": cc["preorigin"]["pumps_kept_in_grid"]
        / cc["preorigin"]["pumps_kept"],
        "dedup_electricity_fraction_2012_2022": wsum(T["changhua", "full", "dedup_in_all"],
                                                     LONG)
        / wsum(T["changhua", "full", "dedup_all"], LONG),
        "irrigation_fraction_2012_2022": wsum(T["changhua", "preorigin", "in_irr"], LONG)
        / wsum(T["changhua", "preorigin", "irr"], LONG),
        "yunlin_in_grid_fraction_2012_2022": wsum(T["yunlin", "preorigin", "in_all"], LONG)
        / wsum(T["yunlin", "preorigin", "all"], LONG)}

    # ------------------------------------------------------------------ C5 in-grid anchor
    mu_sd = dec["preorigin"]["mu_prior"]["log_sd"]
    c5 = {}
    for name, w in (("2012-01..2022-12", LONG), ("2013-01..2022-12",
                                                 ("2013-01-01", "2022-12-01"))):
        est_in = wsum(cand(V1P / "changhua_candidates.parquet", "scaled", "in_grid"), w)
        est_ct = wsum(cand(V1P / "changhua_candidates.parquet", "scaled", "county"), w)
        tr_in = wsum(T["changhua", "preorigin", "in_all"], w)
        tr_ct = wsum(T["changhua", "preorigin", "all"], w)
        ny = (pd.Timestamp(w[1]).year - pd.Timestamp(w[0]).year + 1)
        r_in, r_ct = est_in / tr_in, est_ct / tr_ct
        c5[name] = {"estimate_in_grid_GWh_yr": est_in / ny, "truth_in_grid_GWh_yr": tr_in / ny,
                    "ratio_in_grid": r_in, "ln_ratio_in_grid": float(np.log(r_in)),
                    "z_mu_prior": float(np.log(r_in)) / mu_sd,
                    "inside_mu90_0.549": bool(abs(np.log(r_in)) <= Z90 * mu_sd),
                    "inside_mu90_0.5": bool(abs(np.log(r_in)) <= Z90 * 0.5),
                    "estimate_county_GWh_yr": est_ct / ny, "truth_county_GWh_yr": tr_ct / ny,
                    "ratio_county": r_ct, "z_county_mu_prior": float(np.log(r_ct)) / mu_sd,
                    "yunlin_in_grid_GWh_yr": wsum(T["yunlin", "preorigin", "in_all"], w) / ny,
                    "yunlin_county_GWh_yr": wsum(T["yunlin", "preorigin", "all"], w) / ny}
    ev["C5_in_grid_anchor"] = c5

    # ------------------------------------------------------------------ T12 shape
    t12 = {}
    for nm in ("scaled", "shape", "transfer", "irrigation_only"):
        est = cand(V1P / "changhua_candidates.parquet", nm)
        obs = T["changhua", "preorigin", "irr" if nm == "irrigation_only" else "all"]
        t12[nm] = {"2016-2022": compare(est, obs, PRE_WIN),
                   "2012-2015": compare(est, obs, ("2012-01-01", "2015-12-01"))}
    ev["T12_monthly_shape"] = t12

    # ------------------------------------------------------------------ T13 rank
    t13 = {}
    for var, path, w in (("preorigin", V1P, PRE_WIN), ("full", V1F, FULL_WIN)):
        rows = {}
        for nm in ("scaled", "shape", "transfer", "irrigation_only",
                   "national_subindustry_weak"):
            est = cand(path / "changhua_candidates.parquet", nm).loc[w[0]:w[1]].dropna()
            obs = T["changhua", var, "irr" if nm == "irrigation_only" else "all"]
            o = obs.reindex(est.index)
            r = float(est.sum() / o.sum())
            rows[nm] = {"ratio_to_truth": r, "abs_ln": abs(float(np.log(r))),
                        "months": [str(est.index.min().date())[:7],
                                   str(est.index.max().date())[:7], int(len(est))]}
        order = sorted(rows, key=lambda k: rows[k]["abs_ln"])
        for i, k in enumerate(order):
            rows[k]["rank"] = i + 1
        t13[var] = rows
    t13["truth_z_in_mu_prior"] = float(np.log(c1["preorigin"]["a_star"]
                                              / sep["preorigin"]["a"]) / mu_sd)
    ev["T13_ensemble_rank"] = t13

    # ------------------------------------------------------------------ C8 coverage
    p = guard.open_product(OBS, "after_the_fact_analysis")
    obs_t = p.read_table("tpc_county_sector_monthly_observed.parquet")
    ag = obs_t[(obs_t.sector == lib.AG) & obs_t.county.isin([lib.CH, lib.YL])]
    ag = ag.pivot(index="date", columns="county", values="value")
    ag.index = pd.DatetimeIndex(ag.index)
    c8 = {}
    for cty, cn in (("changhua", lib.CH), ("yunlin", lib.YL)):
        tp = ag[cn]
        out = {}
        for yr in range(2016, 2027):
            m1 = f"{yr}-06-01" if yr == 2026 else f"{yr}-12-01"
            w = (f"{yr}-01-01", m1)
            tpc_y = float(tp.loc[w[0]:w[1]].sum())
            out[str(yr) + ("H1" if yr == 2026 else "")] = {
                "tpc_ag_GWh": tpc_y,
                "cov_dedup_agri": wsum(T[cty, "full", "dedup_agri"], w) / tpc_y,
                "cov_dedup_all": wsum(T[cty, "full", "dedup_all"], w) / tpc_y,
                "cov_clean_all": wsum(T[cty, "full", "all"], w) / tpc_y}
        c8[cty] = out
    c8["ratio_CH_over_YL"] = {
        k: {"tpc_ag": c8["changhua"][k]["tpc_ag_GWh"] / c8["yunlin"][k]["tpc_ag_GWh"],
            "census_dedup_agri": (c8["changhua"][k]["cov_dedup_agri"]
                                  * c8["changhua"][k]["tpc_ag_GWh"])
            / (c8["yunlin"][k]["cov_dedup_agri"] * c8["yunlin"][k]["tpc_ag_GWh"]),
            "census_clean_all": (c8["changhua"][k]["cov_clean_all"]
                                 * c8["changhua"][k]["tpc_ag_GWh"])
            / (c8["yunlin"][k]["cov_clean_all"] * c8["yunlin"][k]["tpc_ag_GWh"])}
        for k in c8["changhua"]}
    c8["opened_with"] = {"product": "county_electricity/tpc_observed/v1",
                         "use": "after_the_fact_analysis", "notes": p.notes}
    ev["C8_coverage_validator_role"] = c8

    # ------------------------------------------------------------------ T10 / T11 (test)
    scored = {}
    for tid in ("P1r2.T10_ext_yunlin", "P1r2.T11_ext_changhua"):
        scored[tid] = pr.assert_can_score("P1r2.census_tests", "P1", test_id=tid)
    rule = {}
    months = []
    for m in pd.date_range(EXT[0], EXT[1], freq="MS"):
        ok = True
        for cty in ("changhua", "yunlin"):
            x = mc[mc.county == cty].set_index("date")["meters_nonzero"]
            prev = x.get(m - pd.DateOffset(years=1), np.nan)
            r = float(x.get(m, np.nan) / prev) if prev and np.isfinite(prev) else float("nan")
            rule[f"{cty}:{str(m.date())[:7]}"] = r
            ok &= bool(np.isfinite(r) and r >= 0.75)
        if ok:
            months.append(m)
    jul26 = {cty: float(mc[(mc.county == cty) & (mc.date == pd.Timestamp("2026-07-01"))]
                        ["meters_nonzero"].iloc[0]
                        / mc[(mc.county == cty) & (mc.date == pd.Timestamp("2025-07-01"))]
                        ["meters_nonzero"].iloc[0]) for cty in ("changhua", "yunlin")}
    yx = pd.read_parquet(V1F / "yunlin_extension.parquet").set_index("date")
    yx.index = pd.DatetimeIndex(yx.index)
    t10 = ext_test(yx["value"], yx["sd"], T["yunlin", "full", "all"], months)
    t11 = {}
    for nm in ("scaled_apre", "scaled", "transfer"):
        v = cand(V1F / "changhua_candidates.parquet", nm)
        s = cand(V1F / "changhua_candidates.parquet", nm, col="sd")
        t11[nm] = ext_test(v, s, T["changhua", "full", "all"], months)
    t11["pass"] = t11["scaled_apre"]["pass"]
    ev["T10_T11_month_rule"] = {"meters_nonzero_ratio_to_m_minus_12": rule,
                                "scored_months": [str(m.date())[:7] for m in months],
                                "excluded_a_priori": ["2026-07"],
                                "jul_2026_ratio_reported": jul26}
    ev["T10_ext_yunlin"] = t10
    ev["T11_ext_changhua"] = t11
    ev["scoring_looks"] = {k: {"prereg_id": v["prereg"]["id"], "sha256": v["sha256"],
                               "prereg_created": v["prereg"]["created"],
                               "other_tests_prior_looks": v["other_tests_prior_looks"]}
                           for k, v in scored.items()}
    yl_t = T["yunlin", "full", "all"].loc[months].sum()
    ch_t = T["changhua", "full", "all"].loc[months].sum()
    ev["T11_decomposition"] = {
        "truth_CH_over_YL_scored_months": float(ch_t / yl_t),
        "a_pre": sep["preorigin"]["a"], "a_full": sep["full"]["a"],
        "identity": "1 + T11 err = (1 + T10 err) x a / (CH/YL truth ratio over the months)",
        "check_scaled_apre": float((1 + t10["sum_err"]) * sep["preorigin"]["a"]
                                   / (ch_t / yl_t) - 1)}

    # ------------------------------------------------------------------ tables
    rows = []
    for (cty, var, key), s in T.items():
        rows.append(pd.DataFrame({"date": s.index, "county": cty, "variant": var,
                                  "series": key, "gwh": s.to_numpy()}))
    pd.concat(rows, ignore_index=True).to_parquet(od / "truth_series_used.parquet",
                                                  index=False)
    yrs = []
    for cty, cn in (("changhua", "CH"), ("yunlin", "YL")):
        s = T[cty, "full", "all"]
        yrs.append(s.groupby(s.index.year).sum().rename(f"{cn}_clean_all_full"))
        s = T[cty, "full", "in_all"]
        yrs.append(s.groupby(s.index.year).sum().rename(f"{cn}_clean_in_grid_full"))
    for nm in ("scaled", "scaled_apre", "shape", "transfer"):
        s = cand(V1F / "changhua_candidates.parquet", nm)
        yrs.append(s.groupby(s.index.year).sum().rename(f"v1full_{nm}"))
    s = cand(V1P / "changhua_candidates.parquet", "scaled", "in_grid")
    yrs.append(s.groupby(s.index.year).sum().rename("v1pre_scaled_in_grid"))
    annual = pd.concat(yrs, axis=1)
    annual.index.name = "year"
    annual.reset_index().to_parquet(od / "annual_comparison.parquet", index=False)
    (od / "evaluation.json").write_text(json.dumps(js(ev), ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(json.dumps(js({k: ev[k] for k in ("C9_yunlin_consistency", "C1_a_vs_census",
                                            "C4_phi_in", "C5_in_grid_anchor")}),
                     ensure_ascii=False, indent=1))
    print(json.dumps(js({"T10": {k: t10[k] for k in ("n", "sum_err", "pass",
                                                     "monthly_rel_rms")},
                         "T11": {nm: {k: t11[nm][k] for k in ("n", "sum_err", "pass")}
                                 for nm in ("scaled_apre", "scaled", "transfer")}}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
