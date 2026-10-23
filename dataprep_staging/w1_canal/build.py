"""Build both variants of surface water v4 into a staging directory (then publish.py).

    cpu_slot.sh python -m w1_canal.build STAGE_DIR        (from dataprep_staging/)

Writes STAGE_DIR/{preorigin,full}/ with every output file of the product and
STAGE_DIR/common.json (parse QC, source checks, head check). Nothing is written under
results/dataprep/ by this script; publish.py copies a checked staging tree into new
version directories with manifests.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common.hashing import grid_hash, mask_hash  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from scipy import stats  # noqa: E402

from w1_canal import checks, composite, diversion, grid, sources, statespace, validate  # noqa: E402
from w1_canal.config import (  # noqa: E402
    BENCH_YEARS,
    CALENDAR,
    DISTRICT_NAME,
    DISTRICTS,
    GAUGE_NAMES,
    PRINTED_MONTHLY_TOL,
    PRODUCT_BASE,
    QC_EXCLUDE,
    SS_SEED,
    VARIANTS,
)
from w1_canal.parse_yearbook import parse_qc, parse_years  # noqa: E402

LEVEL = 0.90


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


# ----------------------------------------------------------------------------- gauges L1
def gauges_l1(daily: pd.DataFrame, mon: pd.DataFrame) -> pd.DataFrame:
    """Daily gauge table with flags: 0 parsed, 8 not printed, 9 suspect (plan exclusions,
    and months whose parsed days are all zero while the printed monthly mean is positive)."""
    d = daily.copy()
    d["date"] = pd.to_datetime(dict(year=d.year, month=d.month, day=d.day))
    full = []
    for (s, y), g in d.groupby(["station_no", "year"]):
        idx = pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D")
        g = g.set_index("date").reindex(idx)
        g["station_no"], g["year"] = s, y
        full.append(g)
    d = pd.concat(full).rename_axis("date").reset_index()
    d["flag"] = np.where(d.q_cms.isna(), 8, 0).astype("uint8")
    d["method"] = np.where(d.q_cms.isna(), "not_printed", "yearbook_daily_mean")
    for s, items in QC_EXCLUDE.items():
        for y, months in items:
            sel = (d.station_no == s) & (d.date.dt.year == y)
            if months:
                sel &= d.date.dt.month.isin(months)
            d.loc[sel & d.q_cms.notna(), "flag"] = 9
            d.loc[sel & d.q_cms.notna(), "method"] = "excluded_plan_W1_step1"
    m1 = mon[mon.page == mon.groupby(["station_no", "year"]).page.transform("min")]
    pm = m1.set_index(["station_no", "year", "month"]).q_printed_cms
    cm = d.groupby([d.station_no, d.date.dt.year.rename("year"),
                    d.date.dt.month.rename("month")]).q_cms.agg(["sum", "count"])
    zero = cm[(cm["sum"] == 0) & (cm["count"] > 0)].index
    for s, y, m in zero:
        if (s, y, m) in pm.index and pm.loc[(s, y, m)] > 0:
            sel = (d.station_no == s) & (d.date.dt.year == y) & (d.date.dt.month == m)
            d.loc[sel & d.q_cms.notna(), "flag"] = 9
            d.loc[sel & d.q_cms.notna(), "method"] = "zero_month_vs_positive_printed_mean"
    d["station"] = d.station_no.map(GAUGE_NAMES)
    d["date"] = d.date.dt.strftime("%Y-%m-%d")
    d["value"] = d.q_cms
    d["sd"] = np.nan
    d["product_version"] = ""
    d["source_ids"] = "wra_hydro_yearbook:" + d.year.astype(str)
    return d[["date", "station_no", "station", "value", "sd", "flag", "method",
              "product_version", "source_ids"]]


# ----------------------------------------------------------------------------- coverage
def coverage_summary(hits: list[bool], clusters: list, n_clusters: int,
                     level: float = 0.95) -> dict:
    """Pre-registered rule: Clopper-Pearson 95 % interval with effective n = number of year
    clusters (hits scaled by n_clusters / n) must contain 0.90. The A1 harness gate
    (dataprep_staging/harness/coverage.py, built in parallel by WP A1) is reported too."""
    n = len(hits)
    k = int(sum(hits))
    k_eff = k * n_clusters / n
    lo = 0.0 if k_eff == 0 else float(stats.beta.ppf((1 - level) / 2, k_eff,
                                                     n_clusters - k_eff + 1))
    hi = 1.0 if k_eff >= n_clusters else float(stats.beta.ppf((1 + level) / 2, k_eff + 1,
                                                              n_clusters - k_eff))
    out = {"n": n, "hits": k, "coverage": k / n, "n_eff": n_clusters,
           "clopper_pearson_95_neff": [lo, hi], "passed_prereg_rule": bool(lo <= 0.90 <= hi)}
    try:
        from harness import coverage as hc
        r = hc.coverage_gate(np.asarray(hits, dtype=bool), np.asarray(clusters))
        out["a1_harness_gate"] = {k2: (float(v) if isinstance(v, (float, np.floating)) else v)
                                  for k2, v in r.items() if not isinstance(v, np.ndarray)}
    except Exception as e:  # noqa: BLE001
        out["a1_harness_gate"] = {"available": False, "error": str(e)[:200]}
    return out


# ----------------------------------------------------------------------------- variant
def run_variant(name, cfg, daily, mon, jiji, bench, moa, v3m, alloc, stage: Path) -> dict:
    years = list(cfg["years"])
    pv = f"{PRODUCT_BASE}/{name}/(staged)"
    out = stage / name
    out.mkdir(parents=True, exist_ok=True)
    log(f"== {name}: years {years[0]}-{years[-1]}")
    # gauges L1 (variant years)
    gl1 = gauges_l1(daily[daily.year.isin(years)], mon)
    gl1.to_parquet(out / "gauges_daily_L1.parquet", index=False)
    # composite
    comp = composite.build_composite(daily, jiji, years, years)
    ctab = composite.composite_table(comp, pv)
    ctab.to_parquet(out / "inflow_composite_daily.parquet", index=False)
    dd = diversion.make_daily(comp["I_cms"], jiji, years)
    targets = diversion.annual_targets(jiji, years)
    # registered benchmark
    log("benchmark grid + leave-one-year-out")
    res, grid_list = validate.evaluate(dd, jiji, v3m, bench, years, log)
    validate.to_frame(res).to_csv(out / "janmay_validation.csv", index=False)
    fc = res["final_choice"]
    p = np.array(fc["beta_a"])
    s_c, s_y, w = fc["s_c"], fc["s_y"], fc["winter"]
    ind = dict(zip(DISTRICTS, diversion.simulate(dd, *p, s_c, s_y, w), strict=True))
    den = {d: diversion.denton_pfd(ind[d], targets[d]) for d in DISTRICTS}
    # structural-zero alternative (winter weight 0.5), for the sd of zero months
    alt = None
    if w == 0.0:
        p5, _ = diversion.fit(dd, targets, s_c, s_y, 0.5, True)
        i5 = dict(zip(DISTRICTS, diversion.simulate(dd, *p5, s_c, s_y, 0.5), strict=True))
        alt = {d: diversion.denton_pfd(i5[d], targets[d]) for d in DISTRICTS}
    # state space with all aggregates
    log("state space (all aggregates)")
    jm_all = {d: {int(y): float(bench.loc[y, d]) for y in BENCH_YEARS} for d in DISTRICTS}
    cs = {d: {int(y): float(v) for y, v in moa[d].dropna().items()} for d in DISTRICTS}
    ss = {}
    for k, d in enumerate(DISTRICTS):
        obs = statespace.build_obs(years, targets[d], jm_all[d], cs[d])
        ss[d] = statespace.run(ind[d], obs, seed=SS_SEED + k)
    # leave-one-year-out coverage of the state space
    log("state space leave-one-year-out coverage")
    cov_rows = []
    for k, (yr, sc_k, sy_k, w_k) in enumerate(res["chosen_supply_limited"]):
        g = next(g for g in grid_list if g["cap"] and g["s_c"] == sc_k and g["s_y"] == sy_k
                 and g["w"] == w_k)
        ik = dict(zip(DISTRICTS, diversion.simulate(dd, *g["p"], sc_k, sy_k, w_k),
                      strict=True))
        for j, d in enumerate(DISTRICTS):
            obs = statespace.build_obs(years, targets[d], jm_all[d], cs[d], holdout_year=yr)
            r = statespace.run(ik[d], obs, seed=SS_SEED + 100 * (k + 1) + j)
            jm = statespace.janmay_of(r["draws"], years, yr)
            lo, hi = statespace.interval(jm, LEVEL)
            o = jm_all[d][yr]
            cov_rows.append({"year": yr, "district": d, "obs": o, "p05": lo, "p50":
                             float(np.median(jm)), "p95": hi, "hit": bool(lo <= o <= hi),
                             "best_rho": r["best"]["rho"], "best_sigma": r["best"]["sigma"]})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(out / "statespace_loo_coverage.csv", index=False)
    cov_sum = coverage_summary(cov.hit.tolist(), cov.year.tolist(),
                               n_clusters=len(BENCH_YEARS))
    cov_sum["rmse_median"] = {d: float(np.sqrt(((cov[cov.district == d].p50
                                                  - cov[cov.district == d].obs) ** 2).mean()))
                              for d in DISTRICTS}
    cov_sum["mean_width_90"] = {d: float((cov[cov.district == d].p95
                                          - cov[cov.district == d].p05).mean())
                                for d in DISTRICTS}
    # monthly table
    months = dd.months
    rows = []
    for d in DISTRICTS:
        X = ss[d]["draws"]
        sd = X.std(0)
        zero = ind[d] <= 1e-12
        if alt is not None:
            sd = np.where(zero, alt[d], sd)
        crop = np.array([CALENDAR.get(m.month, (0, 0))[0] for m in months])
        rows.append(pd.DataFrame({
            "date": months.strftime("%Y-%m-%d"), "district": DISTRICT_NAME[d],
            "value": den[d], "sd": sd, "flag": np.uint8(5),
            "method": np.where(zero, "structural_zero_calendar", "supply_limited_denton_pfd"),
            "product_version": pv,
            "source_ids": "wra_hydro_yearbook;jiji_weir_annual;wra_briefing_110_janmay"
                          "(share choice);moa_691(state space)",
            "indicator": ind[d], "ss_mean": X.mean(0), "ss_p05": np.percentile(X, 5, 0),
            "ss_p50": np.median(X, 0), "ss_p95": np.percentile(X, 95, 0), "crop": crop,
            "structural_zero": zero}))
    monthly = pd.concat(rows, ignore_index=True)
    monthly.to_csv(out / "district_monthly.csv", index=False)
    np.savez_compressed(out / "district_draws.npz",
                        dates=np.array(months.strftime("%Y-%m-%d")),
                        draws_changhua=ss["chc"]["draws"], draws_yunlin=ss["yl"]["draws"],
                        units=np.array("1e8 m3 per month, Jiji weir delivery to the district"),
                        note=np.array("posterior draws of the state-space model (exact annual "
                                      "totals); equal weight; member i of both districts "
                                      "drawn independently"))
    # percolation basis for P4 E4: delivery by crop season
    basis = monthly[["date", "district", "value", "sd", "crop"]].copy()
    basis["crop1_delivery_1e8m3"] = np.where(basis.crop == 1, basis.value, 0.0)
    basis["crop2_delivery_1e8m3"] = np.where(basis.crop == 2, basis.value, 0.0)
    basis.to_csv(out / "percolation_basis.csv", index=False)
    # L3 grid
    vol = {d: den[d] for d in DISTRICTS}
    sw = grid.spread(vol, alloc, months)
    sw_sd = grid.spread_sd({d: monthly[monthly.district == DISTRICT_NAME[d]].sd.to_numpy()
                            for d in DISTRICTS}, alloc, months)
    flag = np.full(sw.shape, 5, dtype="uint8")
    g = grid.twin_grid()
    # v3-level variant (timing-only A/B): v4 x the constant ratio of v3's to Jiji's level
    v3a = v3m.resample("YS").sum()
    k_lvl, k_spread = {}, {}
    for d in DISTRICTS:
        v4a = pd.Series(den[d].reshape(len(years), 12).sum(1) * alloc[d]["infan_fraction"],
                        index=pd.to_datetime([f"{y}-01-01" for y in years]))
        common = v3a.index.intersection(v4a.index)
        ratio = v3a.loc[common, d] / v4a.loc[common]
        k_lvl[d] = float(ratio.mean())
        k_spread[d] = float(np.ptp(ratio))
    sw_v3l = grid.spread({d: den[d] * k_lvl[d] for d in DISTRICTS}, alloc, months)
    from common.flags import validate_grid
    gerr = validate_grid(sw, sw_sd, flag, months)
    np.savez_compressed(
        out / "grid_sw_v4.npz", value=sw, sd=sw_sd, flag=flag,
        dates=np.array(months.strftime("%Y-%m-%d")),
        mask_hash=np.array(mask_hash(g["mask"])),
        grid_hash=np.array(grid_hash(int(g["nx"]), int(g["ny"]), float(g["dx"]),
                                     float(g["x0"]), float(g["y0"]), g["mask"])),
        nx=g["nx"], ny=g["ny"], dx=g["dx"], x0=g["x0"], y0=g["y0"],
        share_changhua=alloc["chc"]["share"], share_yunlin=alloc["yl"]["share"],
        value_v3level=sw_v3l, k_v3level=np.array([k_lvl["chc"], k_lvl["yl"]]),
        districts=np.array(["changhua", "yunlin"]),
        units=np.array("m/day of the cell (mean daily rate of the month), delivered canal "
                       "water; not recharge"),
        note=np.array("value = Denton district series spread with the v3 farmland allocation "
                      "(LUIMAP 2024); value_v3level = the same timing at v3's (MOA 2012-2019) "
                      "level for a timing-only A/B; a draw on the grid = share_d (x) district "
                      "draw * 1e8 / dx^2 / days of the month (linear)"))
    # supporting: crop shares
    crop = checks.crop_share_check(den, years, moa)
    sens = grid.allocation_sensitivity(alloc, vol, months)
    val = {
        "variant": name, "years": [years[0], years[-1]],
        "composite": {"log": comp["log"], "median_ratio": comp["median_ratio"].to_dict(),
                      "scale_to_weir_inflow": comp["scale"].round(4).to_dict(),
                      "ratio_before_qc": comp["ratio_before"].round(4).to_dict(),
                      "daily_rel_sd": ctab.attrs["rel_sd"],
                      "flag_counts": ctab.flag.value_counts().to_dict()},
        "benchmark": {k: v for k, v in res.items()},
        "final_fit": {"beta": float(p[0]), "a_chc_cms": float(p[1]), "a_yl_cms": float(p[2]),
                      "s_chc": s_c, "s_yl": s_y, "winter": w,
                      "annual_fit_rms_1e8m3": fc["annual_fit_rms"],
                      "denton_negative_months": int(sum((den[d] < -1e-12).sum()
                                                        for d in DISTRICTS)),
                      "indicator_annual_r2": {
                          d: float(1 - ((ind[d].reshape(len(years), 12).sum(1) - targets[d])
                                        ** 2).sum() / ((targets[d] - targets[d].mean()) ** 2)
                                   .sum()) for d in DISTRICTS}},
        "state_space": {d: {"best": ss[d]["best"], "weights_top": sorted(
            ss[d]["weights"], key=lambda x: -x["w"])[:8], "ess": ss[d]["ess"],
            "obs": ss[d]["obs_labels"]} for d in DISTRICTS},
        "state_space_loo_coverage": cov_sum,
        "crop_share_vs_moa691": crop,
        "allocation": {"infan_fraction": {d: alloc[d]["infan_fraction"] for d in DISTRICTS},
                       "v3_rebuild_max_rel": alloc["v3_rebuild_max_rel"],
                       "luimap_sensitivity": sens,
                       "k_v3level": k_lvl, "k_v3level_range_over_years": k_spread},
        "grid_validation_errors": gerr,
        "annual_infan_1e8m3": {d: (den[d].reshape(len(years), 12).sum(1)
                                   * alloc[d]["infan_fraction"]).round(3).tolist()
                               for d in DISTRICTS},
    }
    with open(out / "validation_variant.json", "w") as f:
        json.dump(jsonable(val), f, indent=1, ensure_ascii=False)
    log(f"{name}: SL+Denton RMSE chc {res['scores']['supply_limited_denton']['chc']['rmse']:.3f}"
        f" yl {res['scores']['supply_limited_denton']['yl']['rmse']:.3f}; acceptance "
        f"{res['acceptance']['passed']}")
    return {"val": val, "den": den, "months": months, "comp": comp}


def main(stage: str):
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    log("parse yearbooks")
    daily_raw, mon, ann, pdfs = parse_years(log=log)
    pqc, daily = parse_qc(daily_raw, mon, ann, PRINTED_MONTHLY_TOL)
    mon.to_csv(stage / "printed_monthly.csv", index=False)
    ann.to_csv(stage / "printed_annual.csv", index=False)
    jiji, jchk = sources.jiji_annual()
    bench, bchk = sources.janmay_benchmark()
    moa = sources.moa_crop_share()
    v3m = sources.v3_monthly()
    alloc = grid.v3_allocation()
    results = {}
    for name, cfg in VARIANTS.items():
        results[name] = run_variant(name, cfg, daily, mon, jiji, bench, moa, v3m, alloc, stage)
    # supporting: head partial correlation on the pre-origin variant (2012-2022)
    log("head check")
    pre = results["preorigin"]
    months = pre["months"]
    comp_m = (pre["comp"]["I_cms"] * 86400 / 1e8).resample("MS").sum()
    series = {"v4_preorigin_denton": {DISTRICT_NAME[d]: pd.Series(pre["den"][d], index=months)
                                      for d in DISTRICTS},
              "v3": {DISTRICT_NAME[d]: v3m[d] for d in DISTRICTS},
              "inflow_only": {DISTRICT_NAME[d]: comp_m for d in DISTRICTS}}
    hc = checks.head_check(series, log)
    for g in hc.values():
        g["v4_minus_v3_partial"] = round(g["v4_preorigin_denton"]["partial_r_given_rain3"]
                                         - g["v3"]["partial_r_given_rain3"], 4)
    hc_pass = all(g["v4_minus_v3_partial"] >= 0 for g in hc.values())
    log("comparisons with the scout's parse and transcription")
    pvp = checks.parse_vs_pilot(daily)
    jvp = checks.janmay_vs_pilot(bench)
    common = {"parse_qc": pqc, "yearbook_pdfs": [rel_to_repo(p) for p in pdfs],
              "jiji_check": jchk, "janmay_check": bchk,
              "parse_vs_pilot": pvp, "janmay_vs_pilot": jvp,
              "janmay_benchmark_1e8m3": bench.round(6).to_dict(),
              "moa691_crop1_share": moa.round(4).to_dict(),
              "head_check": hc, "head_check_not_worse": hc_pass}
    with open(stage / "common.json", "w") as f:
        json.dump(jsonable(common), f, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
