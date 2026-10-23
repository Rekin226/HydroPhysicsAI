"""Stage surface_water_v4/full/v2: full/v1 plus the 2026 extension (plan W1 step 6).

    cpu_slot.sh python -I dataprep_staging/w1_canal/ext_build.py STAGE_DIR

Requires the pre-registration W1.ext2026_rain_runoff.v1 (ext_prereg.py) to exist and be
unchanged; logs the design-time look (annual totals 2012-2025) in the A1 look log; runs the
leave-one-year-out chain test and the pre-registered shipping rule; writes to STAGE_DIR
every output of full/v2: the full/v1 files (2012-2025 unchanged, product version updated
later by ext_publish.py), the series extended by 2026-01..last complete month (flag 6), the
rain index and the test tables. Nothing is written under results/dataprep/ here.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE.parents[1]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from harness import prereg  # noqa: E402

from w1_canal import diversion, grid  # noqa: E402
from w1_canal import ext_config as E  # noqa: E402
from w1_canal import ext_model as M  # noqa: E402
from w1_canal.build import jsonable  # noqa: E402
from w1_canal.config import CALENDAR, DISTRICT_NAME, DISTRICTS  # noqa: E402
from w1_canal.ext_prereg import PREREG_ID  # noqa: E402

COPY_UNCHANGED = ["gauges_daily_L1.parquet", "inflow_composite_daily.parquet",
                  "janmay_validation.csv", "statespace_loo_coverage.csv",
                  "printed_monthly.csv", "printed_annual.csv"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ext_months(P: pd.Series) -> pd.DatetimeIndex:
    """2026-01 .. the last month such that every month from 2025-11 on has an index (the
    lags of January 2026 reach back to November 2025)."""
    need = P.loc[f"{E.EXT_YEAR - 1}-11-01":]
    bad = need[need.isna()]
    first_bad = bad.index.min() if len(bad) else None
    last = (first_bad - pd.offsets.MonthBegin(1)) if first_bad is not None else need.index.max()
    if last < pd.Timestamp(f"{E.EXT_YEAR}-01-01"):
        raise ValueError(f"basin index incomplete before {E.EXT_YEAR}: first gap {first_bad}")
    return pd.date_range(f"{E.EXT_YEAR}-01-01", last, freq="MS"), first_bad


def main(stage: str):
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    doc, sha = prereg.load_prereg(PREREG_ID)
    log(f"pre-registration {PREREG_ID} created {doc['created']} sha {sha[:12]}")
    for rec in doc["inputs"]:
        p = M.FULL_V1.parents[3].parent / rec["path"]
        if sha256_file(p) != rec["sha256"]:
            raise RuntimeError(f"pre-registered input changed: {rec['path']}")
    base = M.load_full_v1()
    log("rain index")
    daily, src, qa = M.station_daily()
    pm, cnt = M.station_monthly(daily)
    P, sub, info = M.basin_index(pm)
    months26, first_gap = ext_months(P)
    log(f"extension months {months26[0]:%Y-%m}..{months26[-1]:%Y-%m}; first gap {first_gap}")
    eval_years = M.evaluation_years(P)
    log(f"evaluation years {eval_years}")
    prereg.record_look(
        [f"W1.annual.{d}.{y}" for d in DISTRICTS for y in eval_years],
        "jiji_annual:2012..2025", "W1", "score", test_id="W1.ext2026.shipping_rule",
        prereg_id=PREREG_ID,
        note="design-time leave-one-year-out chain test deciding the 2026 series (units not "
             "fresh: fit targets of full/v1)")
    log("chain test")
    test = M.chain_test(base, P)
    rule = M.shipping_rule(test)
    log(f"rule: {rule}")
    # ------------------------------------------------------------------ 2026 series
    r26 = M.chain_year(base, P, E.EXT_YEAR, list(E.FIT_YEARS), R_year=E.R_EXT_FROM_YEAR)
    if not r26["months"].equals(months26):
        raise RuntimeError("extension months mismatch")
    mon = test["monthly"]
    w_zero = {m for m in range(1, 13) if diversion.wvec(base["final_fit"]["s_chc"],
                                                        base["final_fit"]["winter"])[m] == 0}
    v1_sd_cal = {d: base["sd"][d].groupby(base["sd"][d].index.month).mean() for d in DISTRICTS}
    den_tab = {d: base["den"][d] for d in DISTRICTS}
    val26, sd26, ind26 = {}, {}, {}
    for d in DISTRICTS:
        cal = months26.month
        if rule["ships"] == "rain_chain":
            v = r26[d]
            mm = mon[mon.district == d]
            rm = ((mm.rain_chain - mm.denton) ** 2).groupby(mm.month).mean() ** 0.5
            s = np.array([rm.loc[m] for m in cal])
        else:
            dt = den_tab[d]
            clim = dt.groupby(dt.index.month).mean()
            v = np.array([clim.loc[m] for m in cal])
            sdy = dt.groupby(dt.index.month).std(ddof=1)
            s = np.array([sdy.loc[m] for m in cal])
        zero = np.isin(cal, sorted(w_zero))
        s = np.where(zero, [v1_sd_cal[d].loc[m] for m in cal], s)
        val26[d], sd26[d], ind26[d] = np.asarray(v, float), s, np.asarray(r26[d], float)
    # draws: block bootstrap of whole-year fold residuals (same year for both districts)
    rng = np.random.default_rng(E.DRAW_SEED)
    pick = rng.choice(list(test["evaluation_years"]), size=E.N_DRAWS, replace=True)
    k = len(months26)
    draws26 = {}
    for d in DISTRICTS:
        mm = mon[mon.district == d].set_index(["year", "month"])
        X = np.zeros((E.N_DRAWS, k))
        for j, y in enumerate(pick):
            if rule["ships"] == "rain_chain":
                res = np.array([mm.loc[(y, m), "denton"] - mm.loc[(y, m), "rain_chain"]
                                for m in months26.month])
                X[j] = np.clip(val26[d] + res, 0.0, None)
            else:
                X[j] = np.array([mm.loc[(y, m), "denton"] for m in months26.month])
        draws26[d] = X
    # ------------------------------------------------------------------ copy + extend
    for f in COPY_UNCHANGED:
        shutil.copy2(M.FULL_V1 / f, stage / f)
    v1m = pd.read_csv(M.FULL_V1 / "district_monthly.csv")
    rows = [v1m]
    method = ("rain_runoff_chain_extension" if rule["ships"] == "rain_chain"
              else "monthly_climatology_extension")
    for d in DISTRICTS:
        X = draws26[d]
        zero = np.isin(months26.month, sorted(w_zero))
        rows.append(pd.DataFrame({
            "date": months26.strftime("%Y-%m-%d"), "district": DISTRICT_NAME[d],
            "value": val26[d], "sd": sd26[d], "flag": np.uint8(6),
            "method": np.where(zero, "structural_zero_calendar", method),
            "product_version": "", "source_ids":
                "cwa_codis_rain_upper_choushui;surface_water_v4/full/v1;jiji_weir_annual(R 2025)",
            "indicator": ind26[d], "ss_mean": X.mean(0), "ss_p05": np.percentile(X, 5, 0),
            "ss_p50": np.median(X, 0), "ss_p95": np.percentile(X, 95, 0),
            "crop": [CALENDAR.get(m, (0, 0))[0] for m in months26.month],
            "structural_zero": zero}))
    monthly = pd.concat(rows, ignore_index=True)
    monthly = monthly.sort_values(["district", "date"], kind="stable").reset_index(drop=True)
    order = {"changhua": 0, "yunlin": 1}
    monthly = monthly.sort_values(["district", "date"], key=lambda s: s.map(order)
                                  if s.name == "district" else s, kind="stable")
    monthly.to_csv(stage / "district_monthly.csv", index=False)
    dv1 = base["draws"]
    dates_all = np.concatenate([dv1["dates"], np.array(months26.strftime("%Y-%m-%d"))])
    np.savez_compressed(
        stage / "district_draws.npz", dates=dates_all,
        draws_changhua=np.concatenate([dv1["draws_changhua"], draws26["chc"]], axis=1),
        draws_yunlin=np.concatenate([dv1["draws_yunlin"], draws26["yl"]], axis=1),
        units=dv1["units"],
        note=np.array("2012-2025: full/v1 state-space posterior draws (exact annual totals; "
                      "districts drawn independently); 2026 months (flag 6): block bootstrap "
                      "of whole-year leave-one-year-out residuals of the rain-driven chain, "
                      "member i of both districts from the same residual year"))
    basis = monthly[["date", "district", "value", "sd", "crop"]].copy()
    basis["crop1_delivery_1e8m3"] = np.where(basis.crop == 1, basis.value, 0.0)
    basis["crop2_delivery_1e8m3"] = np.where(basis.crop == 2, basis.value, 0.0)
    basis["flag_value"] = np.where(pd.to_datetime(basis.date).dt.year >= E.EXT_YEAR, 6, 5)
    basis.to_csv(stage / "percolation_basis.csv", index=False)
    # grid
    alloc = grid.v3_allocation()
    g1 = np.load(M.FULL_V1 / "grid_sw_v4.npz", allow_pickle=True)
    dates_v1 = pd.to_datetime(g1["dates"])
    rebuilt = grid.spread({d: base["den"][d].to_numpy() for d in DISTRICTS}, alloc, dates_v1)
    grid_rebuild_max_abs = float(np.abs(rebuilt - g1["value"]).max())
    sw26 = grid.spread(val26, alloc, months26)
    sd26g = grid.spread_sd(sd26, alloc, months26)
    k_lvl = {d: float(x) for d, x in zip(DISTRICTS, g1["k_v3level"], strict=True)}
    v3l26 = grid.spread({d: val26[d] * k_lvl[d] for d in DISTRICTS}, alloc, months26)
    out_grid = {key: g1[key] for key in g1.files}
    out_grid.update(
        value=np.concatenate([g1["value"], sw26], axis=1),
        sd=np.concatenate([g1["sd"], sd26g], axis=1),
        flag=np.concatenate([g1["flag"], np.full(sw26.shape, 6, dtype="uint8")], axis=1),
        dates=dates_all,
        value_v3level=np.concatenate([g1["value_v3level"], v3l26], axis=1),
        note=np.array(str(g1["note"]) + "; 2026 months (flag 6): rain-driven extension, W1 "
                      "step 6 (U4: recorded-forcing arm only)"))
    from common.flags import validate_grid
    gerr = validate_grid(out_grid["value"], out_grid["sd"], out_grid["flag"],
                         pd.to_datetime(dates_all))
    np.savez_compressed(stage / "grid_sw_v4.npz", **out_grid)
    # rain index + tables
    ri = pd.DataFrame({"date": P.index.strftime("%Y-%m-%d"), "value": P.to_numpy(),
                       "n_valid_records": info["n_valid_records"].reindex(P.index).to_numpy()})
    for sb in sub.columns:
        ri[f"subbasin_{sb}_mm"] = sub[sb].to_numpy()
    ri["sd"] = np.nan
    ri["flag"] = np.where(ri.value.isna(), 8, 3).astype("uint8")
    ri["method"] = np.where(ri.value.isna(), "fewer_than_8_valid_records",
                            "normal_ratio_area_weighted_station_mean")
    ri["product_version"] = ""
    ri["source_ids"] = "cwa_codis_rain_upper_choushui"
    ri["units"] = "mm/month (areal index)"
    rs = (ri.flag == 3)
    # sd of the index: between-record scatter of the normal ratio / sqrt(n), times the level
    ratio = pm / pm.loc[E.NORMAL_WINDOW[0]:E.NORMAL_WINDOW[1]].mean()
    rel = (ratio.std(axis=1, ddof=1) / ratio.mean(axis=1) /
           np.sqrt(pm.notna().sum(axis=1))).reindex(P.index)
    ri.loc[rs, "sd"] = (ri.value * rel.to_numpy())[rs]
    ri.to_csv(stage / "basin_rain_index_monthly.csv", index=False)
    stm = pm.copy()
    stm.columns = [f"{c}_mm" for c in stm.columns]
    stc = cnt.copy()
    stc.columns = [f"{c}_numeric_days" for c in stc.columns]
    pd.concat([stm, stc], axis=1).rename_axis("date").reset_index().assign(
        date=lambda t: t.date.dt.strftime("%Y-%m-%d")).to_csv(
        stage / "station_rain_monthly.csv", index=False)
    test["annual"].to_csv(stage / "ext2026_loo_annual.csv", index=False)
    test["monthly"].to_csv(stage / "ext2026_loo_monthly.csv", index=False)
    qm = mon[mon.district == "chc"]
    rq = ((qm.q_pred - qm.q_obs) ** 2).groupby(qm.month).mean() ** 0.5
    inflow = pd.DataFrame({
        "date": months26.strftime("%Y-%m-%d"), "value": r26["q_pred"],
        "sd": [rq.loc[m] for m in months26.month], "flag": np.uint8(6),
        "method": f"rain_runoff_regression_{r26['spec']}", "product_version": "",
        "source_ids": "cwa_codis_rain_upper_choushui;surface_water_v4/full/v1 composite",
        "units": "1e8 m3 per month (composite H075+H049+H064 rescaled to the weir inflow)",
        "mean_cms": r26["q_pred"] * 1e8 / (months26.days_in_month.to_numpy() * 86400.0)})
    inflow.to_csv(stage / "inflow_predicted_monthly.csv", index=False)
    v1val = json.loads((M.FULL_V1 / "validation.json").read_text())
    ext = {
        "preregistration": {"id": PREREG_ID, "created": doc["created"], "sha256": sha},
        "extension_months": [f"{months26[0]:%Y-%m}", f"{months26[-1]:%Y-%m}"],
        "first_index_gap_after_2025_11": None if first_gap is None else f"{first_gap:%Y-%m}",
        "months_without_index_2011_11_on": [f"{m:%Y-%m}" for m in
                                            P.loc["2011-11-01":months26[-1]][
                                                P.loc["2011-11-01":months26[-1]].isna()].index],
        "evaluation_years": test["evaluation_years"],
        "rain_index": {"normal_mm_per_month": info["normal_mm_per_month"],
                       "members": info["members"], "area_weights": info["area_weights"],
                       "n_valid_records_min_2025_11_on": int(info["n_valid_records"]
                                                             .loc["2025-11-01":months26[-1]]
                                                             .min()),
                       "qa_counts": qa,
                       "last_numeric_day": {c: (str(daily[c].last_valid_index().date())
                                                if daily[c].notna().any() else None)
                                            for c in daily.columns}},
        "chain_test": {"scores": test["scores"], "compare": test["compare"],
                       "inflow_monthly_rmse_loo": test["inflow_monthly_rmse"],
                       "inflow_monthly_rmse_climatology_insample":
                           test["inflow_monthly_rmse_climatology_insample"],
                       "spec_counts": test["spec_counts"]},
        "shipping_rule": rule,
        "fit_2026": {"spec": r26["spec"], "inner_loo_rmse": r26.get("inner_loo_rmse"),
                     "beta_a": r26["beta_a"], "R_from_year": E.R_EXT_FROM_YEAR,
                     "annual_fit_rms": r26["fit_rms"]},
        "series_2026": {d: {"value": val26[d], "sd": sd26[d], "indicator": ind26[d],
                            "draws_p05": np.percentile(draws26[d], 5, 0),
                            "draws_p95": np.percentile(draws26[d], 95, 0),
                            "jan_sep_total": float(val26[d].sum())} for d in DISTRICTS},
        "jan_sep_totals_2012_2025_denton": {d: {int(y): float(base["den"][d].loc[str(y)]
                                                         .iloc[:9].sum())
                                                for y in E.FIT_YEARS} for d in DISTRICTS},
        "draw_years": pick,
        "grid": {"v1_rebuild_max_abs_m_per_day": grid_rebuild_max_abs,
                 "validation_errors": gerr},
    }
    full_val = dict(v1val)
    full_val["ext2026"] = ext
    with open(stage / "validation_ext.json", "w") as f:
        json.dump(jsonable(full_val), f, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
