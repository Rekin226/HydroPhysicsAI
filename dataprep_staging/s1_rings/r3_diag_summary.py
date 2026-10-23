"""S1a r3: summary statistics of the diagnosis (reads the r3_diagnose outputs only).

Writes results/dataprep/reports/S1a_evidence/repair_r3/diag/summary.json. Every number the
r3 diagnosis quotes is in that file. No data file is read here except the diagnosis outputs
(rerun_rows.parquet, truth_bdeep.csv, truth_intervals.parquet, timing_survey_scale.csv), all
covered by the look ``r3_looks diagnose``.

    python -m s1_rings.r3_diag_summary        (from dataprep_staging/)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from s1_rings import s1const as K  # noqa: E402

D = K.EVIDENCE / "repair_r3" / "diag"
THR_CM = 100 * K.WORST_WINDOW_RMSE_M


def rms(v) -> float:
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def window_rmse(x: pd.DataFrame, col: str = "err_cm") -> pd.Series:
    return x.groupby("start")[col].apply(rms)


def forcing_check(R: pd.DataFrame) -> dict:
    out = {}
    for var in ("primary", "stress"):
        a = R[(R.variant == var) & (R["mode"] == "registered")].set_index(
            ["window", "k", "site"]).sort_index()
        b = R[(R.variant == var) & (R["mode"] == "noforce")].set_index(
            ["window", "k", "site"]).sort_index()
        d = (b.err_cm - a.err_cm).abs()
        wa, wb = window_rmse(a.reset_index()), window_rmse(b.reset_index())
        oa = window_rmse(a.reset_index(), "err_oracle_cm")
        ob = window_rmse(b.reset_index(), "err_oracle_cm")
        dw = wb - wa
        out[var] = {
            "max_abs_row_change_cm": float(d.max()), "n_rows_change_gt_0p07cm": int((d > 0.07).sum()),
            "n_rows_change_gt_0p5cm": int((d > 0.5).sum()),
            "window_rmse_change_cm_min": float(dw.min()), "window_rmse_change_cm_max": float(dw.max()),
            "window_min_change": str(dw.idxmin()),
            "rmse_at_min_change_cm": [float(wa[dw.idxmin()]), float(wb[dw.idxmin()])],
            "worst_window_cm": [float(wa.max()), float(wb.max())],
            "worst_window_start": [str(wa.idxmax()), str(wb.idxmax())],
            "n_windows_over": [int((wa > THR_CM).sum()), int((wb > THR_CM).sum())],
            "oracle_worst_cm": [float(oa.max()), float(ob.max())],
            "oracle_n_windows_over": [int((oa > THR_CM).sum()), int((ob > THR_CM).sum())],
            "pool_A_n_mean": [float(a.pool_A_n.mean()), float(b.pool_A_n.mean())],
            "pool_B_n_mean": [float(a.pool_B_n.mean()), float(b.pool_B_n.mean())]}
    return out


def line_calibration(x: pd.DataFrame, p_inj: float) -> dict:
    """Empirical vs nominal log10 likelihood ratio per line (primary variant)."""
    prior_odds = p_inj / (1 - p_inj)
    out = {}
    for line, col in (("b_deep", "lr_bdeep"), ("pile", "lr_pile"), ("offset", "lr_offset")):
        y = x[np.isfinite(x[col])].copy()
        y["l10"] = np.log10(y[col])
        bins = [-3.01, -2, -1, -0.3, 0.3, 1, 2, 3.01]
        y["bin"] = pd.cut(y.l10, bins)
        rows = []
        for b, g in y.groupby("bin", observed=True):
            nA = int((g.true_group == "A").sum())
            nB = int((g.true_group == "B").sum())
            emp = (math.log10(((nA + 0.5) / (nB + 0.5)) / prior_odds))
            rows.append({"bin_log10_lr": str(b), "n": int(len(g)), "n_true_A": nA,
                         "n_true_B": nB, "nominal_mean_log10_lr": round(float(g.l10.mean()), 3),
                         "empirical_log10_lr": round(emp, 3)})
        votes = y.groupby(["true_group", y[col.replace("lr_", "vote_")].fillna("-")]).size()
        out[line] = {"bins": rows, "n": int(len(y)),
                     "votes_by_true_group": {f"{a}|{v}": int(n) for (a, v), n in votes.items()}}
    return out


def reliability(x: pd.DataFrame) -> list[dict]:
    m = x[x.assigned == ""]
    rows = []
    for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)):
        g = m[(m.p_A >= lo) & (m.p_A < hi)]
        rows.append({"p_A_bin": f"{lo}-{min(hi, 1.0)}", "n": int(len(g)),
                     "share_true_A": round(float((g.true_group == "A").mean()), 3)
                     if len(g) else None,
                     "mean_p_A": round(float(g.p_A.mean()), 3) if len(g) else None})
    return rows


def worst_window_anatomy(x: pd.DataFrame, T: pd.DataFrame, start: str) -> dict:
    w = x[x.start == start]
    t = T[T.start == start].set_index("site")
    per = []
    for s, g in w.groupby("site"):
        per.append({"site": s, "n": int(len(g)), "n_true_A": int((g.true_group == "A").sum()),
                    "b_deep_truth_ppm": round(float(t.loc[s, "b_deep_truth_ppm"]), 1)
                    if s in t.index and np.isfinite(t.loc[s, "b_deep_truth_ppm"]) else None,
                    "sd_jk_ppm": round(float(t.loc[s, "sd_deep_ppm"]), 1)
                    if s in t.index and np.isfinite(t.loc[s, "sd_deep_ppm"]) else None,
                    "deep_len_m": round(float(t.loc[s, "deep_len_m"]), 1) if s in t.index else None,
                    "share_bdeep_vote_B_trueA": round(float(
                        ((g.true_group == "A") & (g.vote_bdeep == "B")).sum()
                        / max((g.true_group == "A").sum(), 1)), 3),
                    "mean_p_A_trueA": round(float(g.loc[g.true_group == "A", "p_A"].mean()), 3)
                    if (g.true_group == "A").any() else None,
                    "rmse_cm": round(rms(g.err_cm), 3), "bias_cm": round(float(g.err_cm.mean()), 3),
                    "rmse_oracle_cm": round(rms(g.err_oracle_cm), 3),
                    "sq_err_share": round(float(np.square(g.err_cm).sum()
                                                / np.square(w.err_cm).sum()), 3)})
    return {"start": start, "n": int(len(w)), "rmse_cm": rms(w.err_cm),
            "bias_cm": float(w.err_cm.mean()), "per_site": per}


def bdeep_error(T: pd.DataFrame, IV: pd.DataFrame) -> dict:
    f = T.dropna(subset=["b_deep_truth_ppm"]).copy()
    f["excess_mm"] = f.b_deep_truth_ppm * f.deep_len_m / 1000.0
    z = f.b_deep_truth_ppm / np.maximum(f.sd_deep_ppm, 1e6 * K.SD_FLOOR)
    wm = f.groupby("start").excess_mm.mean()
    by_len = []
    for lo, hi in ((0, 30), (30, 50), (50, 80), (80, 400)):
        g = f[(f.deep_len_m > lo) & (f.deep_len_m <= hi)]
        zz = z[g.index]
        by_len.append({"deep_len_m": f"{lo}-{hi}", "n": int(len(g)),
                       "sites": int(g.site.nunique()),
                       "ppm_mean": round(float(g.b_deep_truth_ppm.mean()), 1),
                       "ppm_sd": round(float(g.b_deep_truth_ppm.std()), 1),
                       "mm_mean": round(float(g.excess_mm.mean()), 2),
                       "mm_sd": round(float(g.excess_mm.std()), 2),
                       "jackknife_sd_median_ppm": round(float(g.sd_deep_ppm.median()), 1),
                       "z_sd": round(float(zz.std()), 2), "z_mean": round(float(zz.mean()), 2)})
    per_site = (f.groupby("site").agg(n=("start", "size"),
                                      ppm_mean=("b_deep_truth_ppm", "mean"),
                                      ppm_sd=("b_deep_truth_ppm", "std"),
                                      jackknife_sd_median=("sd_deep_ppm", "median"),
                                      deep_len_median=("deep_len_m", "median"),
                                      n_deep_median=("n_deep", "median"))
                .round(1).reset_index().to_dict(orient="records"))
    deep = IV[IV.deep]
    bad = deep.start.between("2016-07", "2017-06")
    return {
        "definition": "b_deep of the UNINJECTED surveys (true scale 0) per registered window and "
                      "site; excess_mm = b_deep x deep length (the deep section's shortening "
                      "beyond the mean-rate prediction, sign: negative = more shortening)",
        "n_site_windows": int(len(T)), "n_finite": int(len(f)),
        "n_nan_lt3_deep_intervals": int(T.b_deep_truth_ppm.isna().sum()),
        "ppm_mean": round(float(f.b_deep_truth_ppm.mean()), 1),
        "ppm_sd": round(float(f.b_deep_truth_ppm.std()), 1),
        "var_ppm2_total": round(float(f.b_deep_truth_ppm.var()), 0),
        "var_ppm2_window_means": round(float(f.groupby("start").b_deep_truth_ppm.mean().var()), 0),
        "var_ppm2_within": round(float((f.b_deep_truth_ppm - f.groupby("start")
                                        .b_deep_truth_ppm.transform("mean")).var()), 0),
        "excess_mm_mean": round(float(f.excess_mm.mean()), 2),
        "excess_mm_sd": round(float(f.excess_mm.std()), 2),
        "excess_mm_rms": round(rms(f.excess_mm), 2),
        "window_mean_excess_mm_sd": round(float(wm.std()), 2),
        "window_mean_excess_mm_min": round(float(wm.min()), 2),
        "window_mean_excess_mm_min_start": str(wm.idxmin()),
        "within_window_excess_mm_sd": round(float((f.excess_mm - f.groupby("start")
                                                   .excess_mm.transform("mean")).std()), 2),
        "z_jackknife_sd": round(float(z.std()), 2), "z_jackknife_mean": round(float(z.mean()), 2),
        "by_deep_length": by_len, "per_site": per_site,
        "deep_interval_excess_mm_mean_failing_era_2016_07_to_2017_06": round(
            float(deep[bad].excess_mm.mean()), 3),
        "deep_interval_excess_mm_mean_other": round(float(deep[~bad].excess_mm.mean()), 3),
        "deep_interval_ppm_equiv_failing_era": round(
            float(1e3 * deep[bad].excess_mm.sum() / deep[bad].L_k_m.sum()), 1),
        "deep_interval_ppm_equiv_other": round(
            float(1e3 * deep[~bad].excess_mm.sum() / deep[~bad].L_k_m.sum()), 1)}


def timing_summary(TM: pd.DataFrame) -> dict:
    d = TM[TM.set == "deep"].copy()
    med = float(d.slope_ppm.median())
    mad = float(1.4826 * np.median(np.abs(d.slope_ppm - med)))
    m = d.groupby("month").slope_ppm.agg(["size", "median"])
    big = d.groupby("month").slope_ppm.apply(lambda v: int((np.abs(v) > 40).sum()))
    d["t2"] = pd.to_datetime(d.t2)
    d["half"] = d.t2.dt.year.astype(str) + np.where(d.t2.dt.month <= 6, "H1", "H2")
    half = {}
    for s, g in d.groupby("site"):
        hs = g.groupby("half").slope_ppm.sum()
        half[s] = {k: round(float(v), 1) for k, v in hs.items() if "2015H1" <= k <= "2019H2"}
    h = pd.DataFrame(half).T[sorted({k for v in half.values() for k in v})]
    return {"definition": "per site and pair of consecutive surveys (<= 62 days apart), OLS "
                          "slope of the depth change of the rings below the site's median depth "
                          "on depth (ppm); an instrument scale step would be one large common "
                          "slope on one date",
            "n_pairs": int(len(d)), "median_ppm": round(med, 1), "mad_sd_ppm": round(mad, 1),
            "max_abs_monthly_median_ppm": round(float(m["median"].abs().max()), 1),
            "month_of_max": str(m["median"].abs().idxmax()),
            "max_sites_over_40ppm_in_one_month": int(big.max()),
            "half_year_sum_ppm_by_site": half,
            "half_year_median_over_sites_ppm": {str(c): round(float(h[c].median()), 1)
                                                for c in h.columns}}


def pools(x: pd.DataFrame) -> dict:
    out = {}
    for g in "AB":
        n = x[f"pool_{g}_n"]
        fb = x[f"fallback_{g}"]
        out[g] = {"mean_n": round(float(n.mean()), 2),
                  "share_n": {int(k): round(float(v), 3)
                              for k, v in n.value_counts(normalize=True).sort_index().items()},
                  "fallback_share": round(float(fb.mean()), 3)}
    tA = x[x.true_group == "A"]
    out["oracle_rmse_by_poolA_n_trueA"] = {
        int(k): {"n": int(len(g)), "rmse_oracle_cm": round(rms(g.err_oracle_cm), 3)}
        for k, g in tA.groupby("pool_A_n")}
    tB = x[x.true_group == "B"]
    out["oracle_rmse_by_poolB_n_trueB"] = {
        int(k): {"n": int(len(g)), "rmse_oracle_cm": round(rms(g.err_oracle_cm), 3)}
        for k, g in tB.groupby("pool_B_n")}
    out["oracle_rmse_trueB_fallback_vs_pooled_cm"] = {
        "fallback": round(rms(tB.loc[tB.fallback_B, "err_oracle_cm"]), 3),
        "pooled": round(rms(tB.loc[~tB.fallback_B, "err_oracle_cm"]), 3),
        "n_fallback": int(tB.fallback_B.sum()), "n_pooled": int((~tB.fallback_B).sum())}
    return out


def membership_parts(x: pd.DataFrame) -> dict:
    """Errors by evidence situation (primary, registered mode)."""
    def sit(r):
        off = r.vote_offset is not None and r.vote_offset == r.vote_offset
        bd = r.vote_bdeep is not None and r.vote_bdeep == r.vote_bdeep
        if r.assigned:
            return "assigned"
        if r.forced:
            return "forced"
        return f"unassigned_offset{'vote' if off else 'none'}_bdeep{'vote' if bd else 'none'}"
    y = x.copy()
    y["situation"] = y.apply(sit, axis=1)
    out = {}
    for s, g in y.groupby("situation"):
        out[s] = {"n": int(len(g)), "share": round(float(len(g) / len(y)), 3),
                  "rmse_cm": round(rms(g.err_cm), 3), "bias_cm": round(float(g.err_cm.mean()), 3),
                  "rmse_oracle_cm": round(rms(g.err_oracle_cm), 3),
                  "sq_err_share": round(float(np.square(g.err_cm).sum()
                                              / np.square(y.err_cm).sum()), 3)}
    # wrong-direction b_deep likelihood ratios for truly-A sites by deep length
    tA = y[y.true_group == "A"].copy()
    tA["lenbin"] = pd.cut(tA.deep_len_m, [0, 30, 50, 80, 400])
    wl = tA.groupby("lenbin", observed=True).apply(
        lambda g: pd.Series({"n": len(g), "share_lr_bdeep_le_0p1": float((g.lr_bdeep <= 0.1).mean()),
                             "share_vote_B": float((g.vote_bdeep == "B").mean()),
                             "rmse_cm": rms(g.err_cm)}), include_groups=False)
    out["trueA_by_deep_length"] = {str(k): {kk: (round(float(vv), 3) if kk != "n" else int(vv))
                                            for kk, vv in v.items()}
                                   for k, v in wl.to_dict(orient="index").items()}
    return out


def main() -> None:
    R = pd.read_parquet(D / "rerun_rows.parquet")
    T = pd.read_csv(D / "truth_bdeep.csv")
    IV = pd.read_parquet(D / "truth_intervals.parquet")
    TM = pd.read_csv(D / "timing_survey_scale.csv")
    p_inj = float(json.loads((K.EVIDENCE / "assignment.json").read_text())["p_A_rule"]["p_A"])
    x = R[(R.variant == "primary") & (R["mode"] == "registered")].copy()
    xs = R[(R.variant == "stress") & (R["mode"] == "registered")].copy()
    wr = window_rmse(x)
    out = {
        "source": "repair_r3/diag/rerun_rows.parquet (re-execution reproducing the registered "
                  "S1a.e_synthetic_mixed.v1 rows exactly) and the truth/timing diagnosis files",
        "status": "diagnosis after the fact on spent units; not a test",
        "registered_primary": {"worst_window_cm": float(wr.max()), "worst_start": str(wr.idxmax()),
                               "n_windows": int(len(wr)), "n_over": int((wr > THR_CM).sum()),
                               "median_cm": float(wr.median())},
        "forcing_full_rerun": forcing_check(R),
        "bdeep_error_truth": bdeep_error(T, IV),
        "timing": timing_summary(TM),
        "line_calibration_primary": line_calibration(x, p_inj),
        "line_calibration_stress": line_calibration(xs, p_inj),
        "p_A_reliability_mixture_rows_primary": reliability(x),
        "pools_primary": pools(x),
        "membership_situations_primary": membership_parts(x),
        "worst_window_anatomy_primary": worst_window_anatomy(x, T, str(wr.idxmax())),
        "by_true_group_primary": {g: {"n": int(len(v)), "rmse_cm": round(rms(v.err_cm), 3),
                                      "bias_cm": round(float(v.err_cm.mean()), 3),
                                      "rmse_oracle_cm": round(rms(v.err_oracle_cm), 3)}
                                  for g, v in x.groupby("true_group")},
    }
    (D / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({k: out[k] for k in ("registered_primary", "forcing_full_rerun")}, indent=1,
                     ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
