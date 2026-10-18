"""Stage surface_water_v4/full/v3 (W1 repair round): full/v2 with the repaired 2026
uncertainty, its coverage check, interval labels on every row and consistent structural zeros.

    cpu_slot.sh python -I dataprep_staging/w1_canal/rep_build.py STAGE_DIR

Requires the A1 declaration W1.ext2026_uncertainty.r1 (rep_prereg.py) to exist with
unchanged inputs; logs the design-time looks (annual totals, Jan-May totals, composite
months) in the A1 look log before computing any coverage; writes every output of full/v3 to
STAGE_DIR (rep_publish.py publishes it). Values are not changed: 2012-2025 values and members
are full/v1's, the 2026 values are the registered chain's (W1.ext2026_rain_runoff.v1).

What changes against full/v2 (verifier major issue 1 and minor issues):
* 2026 sd and members: rep_config error model (residuals against the full/v1 state-space
  draws, bias folded into the spread, small-sample inflation, smoothed bootstrap, centred);
* every row carries ``sd_basis`` and ``interval_calibration`` (plan 4.7);
* structural-zero months (Dec-Jan, winter weight 0) carry sd 0, matching their members; the
  winter-0.5 alternative moves to ``structural_alt_value`` (full/v1-v2 stored it as the sd);
* inflow_predicted_monthly.csv: sd = g * RMS of the leave-one-year-out monthly residuals, its
  90 % normal interval coverage-checked on months whose composite is observed (flag <= 1);
* validation: the RMSEs against the flag-5 Denton series are moved out of the scores into
  ``agreement_with_flag5_denton_not_scores`` (U1).
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
from harness import prereg2  # noqa: E402
from scipy import stats  # noqa: E402

from w1_canal import ext_model as M  # noqa: E402
from w1_canal import grid  # noqa: E402
from w1_canal import rep_config as RC  # noqa: E402
from w1_canal import rep_uncert as U  # noqa: E402
from w1_canal.build import jsonable  # noqa: E402
from w1_canal.config import DISTRICT_NAME, DISTRICTS, JIJI_COL  # noqa: E402
from w1_canal.sources import janmay_benchmark  # noqa: E402

DECL_ID = "W1.ext2026_uncertainty.r1"
FULL_V2 = M.FULL_V1.parent / "v2"
COPY = ["gauges_daily_L1.parquet", "inflow_composite_daily.parquet", "janmay_validation.csv",
        "statespace_loo_coverage.csv", "printed_monthly.csv", "printed_annual.csv",
        "station_rain_monthly.csv", "ext2026_loo_annual.csv", "ext2026_loo_monthly.csv",
        "basin_rain_index_monthly.csv"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check_declaration() -> tuple[dict, str]:
    doc, sha = prereg2.load_prereg(DECL_ID)
    bad = prereg2.changed_inputs(doc)
    if bad:
        raise RuntimeError(f"declared inputs changed: {bad}")
    return doc, sha


def main(stage: str):
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    doc, sha = check_declaration()
    log(f"declaration {DECL_ID} created {doc['created']} sha {sha[:12]}")
    base = M.load_full_v1()
    v2 = pd.read_csv(FULL_V2 / "district_monthly.csv")
    v2val = json.loads((FULL_V2 / "validation.json").read_text())
    ext = v2val["ext2026"]
    # ------------------------------------------------------------------ rain index (L0)
    log("rain index from L0")
    daily, _src, _qa = M.station_daily()
    pm, _cnt = M.station_monthly(daily)
    P, _sub, _info = M.basin_index(pm)
    pub = pd.read_csv(FULL_V2 / "basin_rain_index_monthly.csv", parse_dates=["date"])
    pub = pub.set_index("date")["value"]
    p_diff = float(np.nanmax(np.abs(P.reindex(pub.index).to_numpy() - pub.to_numpy())))
    if not (p_diff < 1e-9 and P.reindex(pub.index).isna().equals(pub.isna())):
        raise RuntimeError(f"rain index differs from full/v2: {p_diff}")
    eval_years = M.evaluation_years(P)
    if eval_years != ext["evaluation_years"]:
        raise RuntimeError("evaluation years differ from full/v2")
    # ------------------------------------------------------------------ looks, then folds
    jiji = base["jiji"]
    bench, _ = janmay_benchmark()
    obs_annual = {d: {y: float(jiji.loc[y, JIJI_COL[d]]) for y in eval_years} for d in DISTRICTS}
    obs_janmay = {d: {y: float(bench.loc[y, d]) for y in RC.JANMAY_YEARS} for d in DISTRICTS}
    comp = pd.read_parquet(M.FULL_V1 / "inflow_composite_daily.parquet",
                           columns=["date", "flag"])
    observed = U.inflow_observed_months(comp)
    inflow_years = sorted({m.year for m in observed if m.year in eval_years})
    note = ("W1 repair round: design-time coverage check of the repaired 2026 error model "
            "(double leave-one-year-out; units already looked at by W1.ext2026_rain_runoff.v1 "
            "or the Jan-May benchmark)")
    prereg2.record_look([f"W1.annual.{d}.{y}" for d in DISTRICTS for y in eval_years],
                        "jiji_annual:2012..2025", "W1", "design_time_coverage_check",
                        prereg_id=DECL_ID, note=note)
    prereg2.record_look([f"W1.janmay.{d}.{y}" for d in DISTRICTS for y in RC.JANMAY_YEARS],
                        "wra_briefing_110_janmay:2017-01..2021-05", "W1",
                        "design_time_coverage_check", prereg_id=DECL_ID, note=note)
    prereg2.record_look([f"W1.janmay.{d}.{y}" for d in DISTRICTS for y in range(2015, 2022)],
                        "wra_briefing_110_janmay:2015-01..2021-05", "W1",
                        "design_time_coverage_check", prereg_id=DECL_ID,
                        note=note + "; A1 gate-2 re-evaluation of full/v1's existing state-space "
                        "Jan-May coverage (statespace_loo_coverage.csv), historical-row label")
    prereg2.record_look([f"W1.composite.{y}" for y in inflow_years],
                        "inflow_composite_full_v1:2012..2025", "W1",
                        "design_time_coverage_check", prereg_id=DECL_ID, note=note)
    log(f"chain folds: {len(eval_years)} single + {len(eval_years) * (len(eval_years) - 1)} "
        f"double")
    single, double = U.chain_folds(base, P, eval_years)
    loo_pub = pd.read_csv(FULL_V2 / "ext2026_loo_monthly.csv")
    rep = max(float(np.abs(np.asarray(single[t][d]) - loo_pub[(loo_pub.year == t) & (
        loo_pub.district == d)].sort_values("month").rain_chain.to_numpy()).max())
        for t in eval_years for d in DISTRICTS)
    if rep > 1e-9:
        raise RuntimeError(f"single folds differ from full/v2 ext2026_loo_monthly: {rep}")
    # ------------------------------------------------------------------ coverage
    log("coverage (annual, Jan-May, inflow)")
    cov, jcov = U.coverage_rows(base, single, double, eval_years, obs_annual, obs_janmay)
    g_ann = U.gate(cov)
    g_jm = U.gate(jcov)
    icov = U.inflow_coverage(single, double, base["Q"], eval_years, observed)
    g_in = U.gate(icov)
    passed = bool(g_ann["a1_gate2"]["pass"])
    label26 = RC.LABEL_2026_PASS if passed else RC.LABEL_2026_FAIL
    # historical rows: A1 gate-2 re-evaluation of full/v1's Jan-May state-space coverage
    ssc = pd.read_csv(M.FULL_V1 / "statespace_loo_coverage.csv")
    g_hist = U.gate(ssc)
    gh = g_hist["a1_gate2"]
    label_hist = RC.LABEL_HIST_TEMPLATE.format(
        verdict="coverage-checked" if gh["pass"] else "uncalibrated", hits=g_hist["hits"],
        n=g_hist["n"], lo=gh["lo"], hi=gh["hi"], rel="contains" if gh["contains_target"]
        else "excludes")
    log(f"annual {g_ann['hits']}/{g_ann['n']} gate {passed}; Jan-May {g_jm['hits']}/"
        f"{g_jm['n']}; inflow {g_in['hits']}/{g_in['n']} gate {g_in['a1_gate2']['pass']}")
    # ------------------------------------------------------------------ 2026 members
    r26 = M.chain_year(base, P, RC.EXT_YEAR, list(RC.FIT_YEARS), R_year=M.E.R_EXT_FROM_YEAR)
    months26 = r26["months"]
    v2_26 = v2[pd.to_datetime(v2.date).dt.year == RC.EXT_YEAR]
    val26 = {d: v2_26[v2_26.district == DISTRICT_NAME[d]].sort_values("date").value.to_numpy()
             for d in DISTRICTS}
    rv = max(float(np.abs(np.asarray(r26[d]) - val26[d]).max()) for d in DISTRICTS)
    if rv > 1e-9:
        raise RuntimeError(f"2026 chain differs from full/v2: {rv}")
    ss, dates_v1 = U.ss_draws(base)
    r_pool = U.residual_pool(single, ss, dates_v1, eval_years)
    prm26 = {d: U.params(r_pool[d]) for d in DISTRICTS}
    iy, kk, eps = U.sampled_members(len(eval_years), ss["chc"].shape[0], RC.N_DRAWS,
                                    RC.DRAW_SEED)
    smask26 = U.structural_mask(base["final_fit"], months26)
    X26, X26_pre = U.members(val26, r_pool, prm26, iy, kk, eps, smask26)
    sd26 = {d: X26[d].std(0, ddof=0) for d in DISTRICTS}
    alt = U.structural_alt_2026(base, P)
    alt26 = {d: np.where(smask26, np.asarray(alt[d], float), np.nan) for d in DISTRICTS}
    member_check = {
        nm: {"pre_truncation_mean_minus_value_max_abs":
               float(np.abs(X26_pre[d].mean(0) - val26[d]).max()),
               "mean_minus_value_by_month": (X26[d].mean(0) - val26[d]).tolist(),
               "jan_sep_mean_minus_value": float(X26[d].sum(1).mean() - val26[d].sum()),
               "negative_members": int((X26[d] < 0).sum()),
               "structural_nonzero_members": int((X26[d][:, smask26] != 0).sum()),
               "distinct_trajectories": int(len(np.unique(np.round(X26[d], 12), axis=0))),
               "distinct_jan_sep_sums": int(len(np.unique(np.round(X26[d].sum(1), 9)))),
               "truncated_member_months": int((X26_pre[d] < 0).sum()),
               "jan_sep_p05_p50_p95": [float(x) for x in np.percentile(X26[d].sum(1),
                                                                       [5, 50, 95])]}
        for d, nm in DISTRICT_NAME.items()}
    inflow_rms26 = U.inflow_rms(single, base["Q"], eval_years)
    g11 = float(stats.t.ppf(RC.PRED_QUANTILE, len(eval_years))
                / stats.norm.ppf(RC.PRED_QUANTILE))
    # ------------------------------------------------------------------ tables
    for f in COPY:
        shutil.copy2(FULL_V2 / f, stage / f)
    mon = v2.copy()
    is26 = pd.to_datetime(mon.date).dt.year == RC.EXT_YEAR
    sz = mon.structural_zero.astype(bool)
    mon["structural_alt_value"] = np.where(sz & ~is26, mon.sd, np.nan)
    mon.loc[sz & ~is26, "sd"] = 0.0
    for d in DISTRICTS:
        sel = is26 & (mon.district == DISTRICT_NAME[d])
        idx = mon[sel].sort_values("date").index
        X = X26[d]
        mon.loc[idx, "sd"] = sd26[d]
        mon.loc[idx, "ss_mean"] = X.mean(0)
        mon.loc[idx, "ss_p05"] = np.percentile(X, 5, 0)
        mon.loc[idx, "ss_p50"] = np.median(X, 0)
        mon.loc[idx, "ss_p95"] = np.percentile(X, 95, 0)
        mon.loc[idx, "structural_alt_value"] = alt26[d]
    mon["sd_basis"] = np.where(
        sz, "structural_zero_members_0",
        np.where(is26, "sd_of_repaired_2026_members (rep_config error model)",
                 "sd_of_full_v1_state_space_draws"))
    mon["interval_calibration"] = np.where(
        sz, RC.LABEL_STRUCTURAL, np.where(is26, label26, label_hist))
    mon.to_csv(stage / "district_monthly.csv", index=False)
    dv1 = base["draws"]
    np.savez_compressed(
        stage / "district_draws.npz", dates=np.concatenate(
            [dv1["dates"], np.array(months26.strftime("%Y-%m-%d"))]),
        draws_changhua=np.concatenate([dv1["draws_changhua"], X26["chc"]], axis=1),
        draws_yunlin=np.concatenate([dv1["draws_yunlin"], X26["yl"]], axis=1),
        units=dv1["units"], member_pool_year=np.asarray(eval_years)[iy],
        member_ss_index=kk, member_shock=eps,
        note=np.array("2012-2025: full/v1 state-space posterior draws (exact annual totals; "
                      "districts drawn independently). 2026 months (flag 6), repair round: "
                      "value + f*g*(residual trajectory D_k(y,.) - chain_y(.) of pool year y "
                      "+ h*eps*pi)/sqrt(1+h^2), centred on the members' mean, truncated at 0, "
                      "structural-zero months 0 (rep_config.py); member i of both districts "
                      "shares y, k and eps (member_pool_year, member_ss_index, member_shock)"))
    basis = mon[["date", "district", "value", "sd", "crop"]].copy()
    basis["crop1_delivery_1e8m3"] = np.where(basis.crop == 1, basis.value, 0.0)
    basis["crop2_delivery_1e8m3"] = np.where(basis.crop == 2, basis.value, 0.0)
    basis["flag_value"] = np.where(pd.to_datetime(basis.date).dt.year >= RC.EXT_YEAR, 6, 5)
    basis.to_csv(stage / "percolation_basis.csv", index=False)
    # grid: same values; sd from the repaired table; structural alternative as its own field
    g2 = np.load(FULL_V2 / "grid_sw_v4.npz", allow_pickle=True)
    alloc = grid.v3_allocation()
    gdates = pd.to_datetime(g2["dates"])
    sd_tab = {d: mon[mon.district == DISTRICT_NAME[d]].sort_values("date").sd.to_numpy()
              for d in DISTRICTS}
    alt_tab = {d: np.nan_to_num(mon[mon.district == DISTRICT_NAME[d]].sort_values("date")
                                .structural_alt_value.to_numpy()) for d in DISTRICTS}
    sd_grid = grid.spread_sd(sd_tab, alloc, gdates)
    keep = ~(U.structural_mask(base["final_fit"], gdates)
             | np.asarray(gdates.year == RC.EXT_YEAR))
    sd_grid_check = float(np.abs(sd_grid[:, keep] - g2["sd"][:, keep]).max())
    out_grid = {k: g2[k] for k in g2.files}
    out_grid.update(
        sd=sd_grid, value_structural_alt=grid.spread(alt_tab, alloc, gdates),
        note=np.array(str(g2["note"]) + "; repair round (full/v3): sd from the repaired table "
                      "(structural-zero months 0, 2026 months from the repaired members); "
                      "value_structural_alt = winter-0.5 alternative of the structural-zero "
                      "months (0 elsewhere)"))
    from common.flags import validate_grid
    gerr = validate_grid(out_grid["value"], out_grid["sd"], out_grid["flag"], gdates)
    np.savez_compressed(stage / "grid_sw_v4.npz", **out_grid)
    inf = pd.read_csv(FULL_V2 / "inflow_predicted_monthly.csv")
    inf["sd"] = g11 * inflow_rms26[months26.month.to_numpy() - 1]
    inf["sd_basis"] = ("g * RMS over the 11 leave-one-year-out folds of (predicted - composite) "
                       "for the calendar month; 90 % interval = value +- 1.645 sd, floored at 0")
    inf["interval_calibration"] = (
        "monthly 90 % interval coverage-checked on observed composite months (flag <= 1): "
        f"{g_in['hits']}/{g_in['n']}, A1 gate-2 "
        f"{'pass' if g_in['a1_gate2']['pass'] else 'fail'}"
        + ("" if g_in["a1_gate2"]["pass"] else " -> uncalibrated"))
    inf.to_csv(stage / "inflow_predicted_monthly.csv", index=False)
    cov.to_csv(stage / "rep_coverage_annual.csv", index=False)
    jcov.to_csv(stage / "rep_coverage_janmay.csv", index=False)
    icov.to_csv(stage / "rep_coverage_inflow.csv", index=False)
    drows = []
    for (t, y), r in double.items():
        for i in range(12):
            drows.append({"heldout_year": t, "year": y, "month": i + 1,
                          "chain_chc": float(r["chc"][i]), "chain_yl": float(r["yl"][i]),
                          "q_pred": float(r["q_pred"][i]), "spec": r["spec"]})
    pd.DataFrame(drows).to_csv(stage / "rep_double_loo_monthly.csv", index=False)
    # ------------------------------------------------------------------ validation
    v3val = json.loads(json.dumps(v2val))
    e3 = v3val["ext2026"]
    agree = {}
    for d in DISTRICTS:
        sc = e3["chain_test"]["scores"][d]
        agree[d] = {k: sc.pop(k) for k in ("rain_chain_janSep_vs_denton_rmse",
                                           "monthly_rmse_vs_denton") if k in sc}
    e3["agreement_with_flag5_denton_not_scores"] = {
        "values": agree, "note": "RMS differences between the chain and the full/v1 Denton "
        "series (flag 5, inferred): agreement between two models, not scores (U1); moved out "
        "of chain_test.scores in full/v3"}
    sup = {}
    for d in DISTRICTS:
        s = e3["series_2026"][d]
        sup[d] = {k: s.pop(k) for k in ("sd", "draws_p05", "draws_p95") if k in s}
    e3["superseded_in_full_v3"] = {
        "series_2026": sup, "draw_years": e3.pop("draw_years", None),
        "note": "full/v2's 2026 sd (RMSE against the flag-5 Denton series) and members (block "
                "bootstrap of 11 fold residuals) are replaced by the repaired error model "
                "(repair_r1)"}
    v3val["repair_r1"] = {
        "declaration": {"id": DECL_ID, "created": doc["created"], "sha256": sha},
        "error_model": {"settings": "dataprep_staging/w1_canal/rep_config.py",
                        "pool_years": eval_years,
                        "params_2026": {DISTRICT_NAME[d]: {k: (v.tolist() if isinstance(
                            v, np.ndarray) else v) for k, v in prm26[d].items()}
                            for d in DISTRICTS}},
        "coverage_annual": g_ann, "coverage_janmay": g_jm, "coverage_inflow": g_in,
        "coverage_hist_statespace_janmay": g_hist, "label_hist": label_hist,
        "label_2026": label26, "annual_check_passed": passed,
        "series_2026": {DISTRICT_NAME[d]: {
            "value": val26[d], "sd": sd26[d], "p05": np.percentile(X26[d], 5, 0),
            "p95": np.percentile(X26[d], 95, 0), "structural_alt_value": alt26[d],
            "jan_sep_total": float(val26[d].sum())} for d in DISTRICTS},
        "member_checks": member_check,
        "inflow_2026_sd": {"g": g11, "rms_by_month": inflow_rms26.tolist()},
        "reproducibility": {"rain_index_vs_full_v2_max_abs": p_diff,
                            "single_folds_vs_full_v2_max_abs": rep,
                            "chain_2026_vs_full_v2_max_abs": rv,
                            "grid_sd_unchanged_months_max_abs": sd_grid_check},
        "grid_validation_errors": gerr,
        "structural_zero_convention": "sd = 0 (equal to the members' spread); the former sd "
                                      "(winter-0.5 alternative Denton value) is in "
                                      "structural_alt_value and grid value_structural_alt",
        "inflow_observed_years": inflow_years,
    }
    with open(stage / "validation_rep.json", "w") as f:
        json.dump(jsonable(v3val), f, indent=1, ensure_ascii=False)
    log("done")


if __name__ == "__main__":
    main(sys.argv[1])
