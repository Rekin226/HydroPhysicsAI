"""WP P1 registered tests T1-T9 (pre-registration P1_county_electricity_2026-10-12).

Writes ``<out_dir>/validation.json`` (all tests) and ``<out_dir>/misspecification.json``.
No head, ring, pile, GNSS or InSAR observation is read. T1 and T8-preorigin read only data
<= 2022-12 (the preorigin loader); the other tests read the full variant.

Usage: python -I validate.py <out_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as dmod  # noqa: E402
import lib  # noqa: E402
import production as prod  # noqa: E402

R_SYN, B_SYN, SEED0 = 1000, 1000, 7000
DROUGHT_YEARS = (2015, 2020, 2021)
DROUGHT_MONTHS = (1, 2, 3, 4, 5)


# ----------------------------------------------------------------------------- backcast
def matched_lead(d: pd.DataFrame, moy_w, trend_w, targets) -> dict:
    out = {}
    for c, col in (("CH", "CH_ag"), ("YL", "YL_ag")):
        sh = d[col] / d["nat_ag"]
        res = {}
        for tag, trend in (("trend", True), ("no_trend", False)):
            m = lib.share_model(sh, moy_w, trend_w, trend=trend)
            tgt = d.loc[f"{targets[0]}-01":f"{targets[-1]}-12"].dropna(subset=[col, "nat_ag"])
            pred = pd.Series(tgt["nat_ag"].to_numpy() * lib.share_predict(m, tgt.index),
                             index=tgt.index)
            ae = lib.annual_err_pct(pred, tgt[col])
            mrel = pred / tgt[col] - 1
            res[tag] = {"beta_per_yr": m["beta"], "annual_err_pct": {int(k): float(v) for k, v
                                                                    in ae.items()},
                        "mean_bias_pct": float(ae.mean()), "rms_pct": lib.rms(ae),
                        "monthly_rel_rms": lib.rms(mrel)}
        res["pass"] = bool(res["trend"]["rms_pct"] <= 5.0
                           and abs(res["trend"]["mean_bias_pct"]) <= 3.0)
        out[c] = res
    out["pass"] = bool(out["CH"]["pass"] and out["YL"]["pass"])
    out["design"] = {"moy_window": list(moy_w), "trend_window": list(trend_w),
                     "targets": list(targets)}
    return out


def loyo_backcast(d: pd.DataFrame) -> dict:
    out = {}
    for c, col in (("CH", "CH_ag"), ("YL", "YL_ag")):
        sh = (d[col] / d["nat_ag"]).dropna()
        errs = {}
        for yr in range(2016, 2026):
            tr = sh[sh.index.year != yr]
            s = tr.groupby(tr.index.month).mean()
            te = d.loc[str(yr)].dropna(subset=[col, "nat_ag"])
            pred = te["nat_ag"] * s.reindex(te.index.month).to_numpy()
            errs[yr] = float(100 * (pred.sum() / te[col].sum() - 1))
        out[c] = {"annual_err_pct": errs, "rms_pct": lib.rms(list(errs.values()))}
    return out


# ----------------------------------------------------------------------------- extension
def ext_tests(d: pd.DataFrame) -> dict:
    yl, ya = d["YL_clean"], d["YL_ag"]
    t0, t1 = "2016-01", "2025-06"
    base = d.loc[t0:t1].dropna(subset=["YL_clean", "YL_ag"])
    loyo, mrel = {}, []
    for yr in range(2016, 2025):
        r = lib.transfer_ratio(yl, ya, t0, t1, exclude_years=[yr])
        te = base.loc[str(yr)]
        p = lib.apply_ratio(r, te["YL_ag"])
        loyo[yr] = float(100 * (p.sum() / te["YL_clean"].sum() - 1))
        mrel += list((p / te["YL_clean"] - 1).to_numpy())
    r = lib.transfer_ratio(yl, ya, t0, "2024-12")
    te = base.loc["2025-01":"2025-06"]
    p = lib.apply_ratio(r, te["YL_ag"])
    h1_2025 = float(100 * (p.sum() / te["YL_clean"].sum() - 1))
    # quarters
    q = {}
    for yr in list(range(2016, 2025)) + [2025]:
        for qi in range(4):
            months = [3 * qi + 1, 3 * qi + 2, 3 * qi + 3]
            blk = base[(base.index.year == yr) & base.index.month.isin(months)]
            if len(blk) < 3:
                continue
            m = pd.Series(False, index=base.index)
            m.loc[blk.index] = True
            rr = lib.transfer_ratio(yl, ya, t0, t1, exclude_mask=m)
            p = lib.apply_ratio(rr, blk["YL_ag"])
            q[f"{yr}Q{qi + 1}"] = float(100 * (p.sum() / blk["YL_clean"].sum() - 1))
    qs = pd.Series(q)
    by_q = {f"Q{k}": lib.rms(qs[qs.index.str.endswith(f"Q{k}")]) for k in range(1, 5)}
    # forward-only rolling 14-month blocks
    roll = {}
    for yr in range(2018, 2024):
        start = pd.Timestamp(f"{yr}-07-01")
        stop = start + pd.DateOffset(months=13)
        if stop > pd.Timestamp("2025-06-01"):
            continue
        rr = lib.transfer_ratio(yl, ya, t0, str((start - pd.DateOffset(months=1)).date())[:7])
        blk = base.loc[start:stop]
        p = lib.apply_ratio(rr, blk["YL_ag"])
        roll[f"{yr}-07"] = {"err14_pct": float(100 * (p.sum() / blk["YL_clean"].sum() - 1)),
                            "err12_pct": float(100 * (p.iloc[:12].sum()
                                                      / blk["YL_clean"].iloc[:12].sum() - 1))}
    # added after the pre-registration (reported only): pre-origin-only LOYO of the transfer
    pre = {}
    for yr in range(2016, 2023):
        rr = lib.transfer_ratio(yl, ya, "2016-01", "2022-12", exclude_years=[yr])
        te = base.loc[str(yr)]
        p = lib.apply_ratio(rr, te["YL_ag"])
        pre[yr] = float(100 * (p.sum() / te["YL_clean"].sum() - 1))
    rms_all = lib.rms(list(loyo.values()))
    return {
        "T4_EXT_LOYO": {"annual_err_pct": loyo, "rms_pct_2016_2024": rms_all,
                        "rms_pct_2016_2022": lib.rms([loyo[y] for y in range(2016, 2023)]),
                        "monthly_rel_rms": lib.rms(mrel), "pass": bool(rms_all <= 10.0)},
        "T5_EXT_season": {"quarter_err_pct": q, "rms_pct_by_quarter": by_q,
                          "rms_pct_pooled": lib.rms(qs.to_numpy()), "n_blocks": int(len(qs))},
        "T6_EXT_2023_25": {"2023": loyo[2023], "2024": loyo[2024], "2025H1": h1_2025},
        "T7_EXT_rolling": roll,
        "T4b_preorigin_transfer_LOYO_not_preregistered": {
            "annual_err_pct": pre, "rms_pct": lib.rms(list(pre.values()))},
    }


# ----------------------------------------------------------------------------- synthetics
def _ar1(n, sd, phi, rng):
    return lib.ar1_noise(n, sd, phi, rng, 1)[0]


def _block_noise(resid: pd.Series, years: np.ndarray, rng) -> np.ndarray:
    e = resid - resid.groupby(resid.index.month).transform("mean")
    yrs = resid.index.year.to_numpy()
    full = [y for y in np.unique(yrs) if (yrs == y).sum() == 12]
    out = np.empty(len(resid))
    for y in np.unique(years):
        k = int((years == y).sum())
        src = full[rng.integers(0, len(full))]
        out[years == y] = e[yrs == src].to_numpy()[:k]
    return out


def syn_window(D: dict, F: dict, cases: bool) -> dict:
    d = D["d"]
    t0, t1 = F["window"]
    w, P, N, y, yrs = dmod.nnls_inputs(d, t0, t1)
    X = np.column_stack([P, N])
    alpha, beta = F["sep"]["a"], F["sep"]["b"]
    sd, phi = F["sep"]["resid_sd"], F["sep"]["resid_ac1"]
    uy, Gy, _ = lib.year_stats(X, y, yrs)
    yidx = np.searchsorted(uy, yrs)
    Pser = pd.Series(P, index=w.index)
    c = lib.seasonal_component(Pser)
    cm1 = pd.Series(c, index=w.index).groupby(w.index.month).first()
    shift = cm1.reindex(((w.index.month - 2) % 12) + 1).to_numpy()
    shift_early = cm1.reindex((w.index.month % 12) + 1).to_numpy()
    drought = np.isin(w.index.year, DROUGHT_YEARS) & np.isin(w.index.month, DROUGHT_MONTHS)
    shapes = {"M0_correct": P}
    if cases:
        shapes.update({"M1_cv_x0.5": P + (0.5 - 1) * c, "M2_cv_x1.5": P + (1.5 - 1) * c,
                       "M3_phase_+1m": P - c + shift, "M4_drought_springs_x1.5":
                       np.where(drought, 1.5 * P, P), "M3b_phase_-1m": P - c + shift_early})
    res = {}
    t_start = time.time()
    for name, Pp in shapes.items():
        base = alpha * Pp + beta * N
        a_eff = alpha * Pp.mean() / P.mean()
        ahat, cover, cover_blk, lo_all, hi_all = [], [], [], [], []
        n_rep = 2 * R_SYN if name == "M0_correct" else R_SYN
        for r in range(n_rep):
            rng = np.random.default_rng(SEED0 + r)
            yy = base + _ar1(len(base), sd, phi, rng)
            hy = np.zeros((len(uy), 2))
            np.add.at(hy, yidx, X * yy[:, None])
            a_hat = lib.nnls2_stats(Gy.sum(0), hy.sum(0))[0]
            ahat.append(a_hat)
            if name == "M0_correct":
                cnt = lib.block_bootstrap_counts(len(uy), B_SYN, rng)
                bo = lib.nnls2_stats(np.einsum("by,yij->bij", cnt, Gy),
                                     np.einsum("by,yi->bi", cnt, hy))[:, 0]
                lo, hi = np.percentile(bo, [5, 95])
                lo_all.append(lo)
                hi_all.append(hi)
                cover.append(lo <= alpha <= hi)
                if r < R_SYN:   # secondary noise model, reported only
                    rng2 = np.random.default_rng(SEED0 + r)
                    yb = base + _block_noise(F["resid"], yrs, rng2)
                    hb = np.zeros((len(uy), 2))
                    np.add.at(hb, yidx, X * yb[:, None])
                    cnt2 = lib.block_bootstrap_counts(len(uy), B_SYN, rng2)
                    bo2 = lib.nnls2_stats(np.einsum("by,yij->bij", cnt2, Gy),
                                          np.einsum("by,yi->bi", cnt2, hb))[:, 0]
                    l2, h2 = np.percentile(bo2, [5, 95])
                    cover_blk.append(l2 <= alpha <= h2)
        ahat = np.array(ahat)
        rec = {"alpha_true": alpha, "a_eff": float(a_eff), "n_rep": int(len(ahat)),
               "mean_a_hat": float(ahat[:R_SYN].mean()),
               "sd_a_hat": float(ahat[:R_SYN].std(ddof=1)),
               "bias_vs_alpha": float(ahat[:R_SYN].mean() - alpha),
               "bias_vs_a_eff": float(ahat[:R_SYN].mean() - a_eff),
               "rel_bias_vs_a_eff": float(ahat[:R_SYN].mean() / a_eff - 1),
               "ln_bias_vs_a_eff": float(np.log(ahat[:R_SYN].mean() / a_eff))}
        if name == "M0_correct":
            cov = np.array(cover)
            k = int(cov[:R_SYN].sum())
            lo_w, hi_w = lib.wilson(k, R_SYN)
            pt = k / R_SYN
            rec["coverage"] = {"point": pt, "wilson95": [lo_w, hi_w], "n_eff": R_SYN,
                               "design_effect": 1.0,
                               "pass": bool(lo_w <= 0.90 <= hi_w and 0.85 <= pt <= 0.95)}
            kb = int(np.sum(cover_blk))
            rec["coverage_secondary_block_noise"] = {"point": kb / R_SYN,
                                                     "wilson95": list(lib.wilson(kb, R_SYN))}
            # remedy (reported only): half-width factor fitted on 0..R-1, checked on R..2R-1
            ah = ahat
            lo_a, hi_a = np.array(lo_all), np.array(hi_all)
            ks = np.linspace(0.5, 4.0, 701)

            def cov_k(kk, sl):
                lo_k = ah[sl] - kk * (ah[sl] - lo_a[sl])
                hi_k = ah[sl] + kk * (hi_a[sl] - ah[sl])
                return np.mean((lo_k <= alpha) & (alpha <= hi_k))
            sA, sB = slice(0, R_SYN), slice(R_SYN, 2 * R_SYN)
            kfit = float(next((kk for kk in ks if cov_k(kk, sA) >= 0.90), np.nan))
            kB = int(round(cov_k(kfit, sB) * R_SYN)) if np.isfinite(kfit) else 0
            rec["remedy_reported_only"] = {
                "k_halfwidth_factor": kfit, "coverage_fit_set": float(cov_k(kfit, sA)),
                "coverage_fresh_set": kB / R_SYN,
                "wilson95_fresh": list(lib.wilson(kB, R_SYN)),
                "uncalibrated_point_coverage_fresh_set": float(np.mean(cov[sB]))}
        res[name] = rec
    return {"window": [t0, t1], "noise": {"type": "AR(1) Gaussian", "sd": sd, "phi": phi},
            "R": R_SYN, "B": B_SYN, "seeds": f"default_rng({SEED0} + r)", "cases": res,
            "seconds": round(time.time() - t_start, 1)}


def fold_mis(syn: dict, F: dict) -> dict:
    hw = 0.5 * (F["sep"]["a_hi"] - F["sep"]["a_lo"])
    rows, qual = {}, []
    for name, rec in syn["cases"].items():
        if name == "M0_correct":
            continue
        exceeds = abs(rec["bias_vs_a_eff"]) > hw
        rows[name] = {"bias_vs_a_eff": rec["bias_vs_a_eff"],
                      "ln_bias_vs_a_eff": rec["ln_bias_vs_a_eff"],
                      "bias_vs_alpha": rec["bias_vs_alpha"], "exceeds_halfwidth": exceeds,
                      "registered": name != "M3b_phase_-1m"}
        if exceeds and name != "M3b_phase_-1m":
            qual.append(abs(rec["ln_bias_vs_a_eff"]))
    b_mis = max(qual) if qual else 0.0
    sig_phi = F["phi"]["sigma_ln_phi"]
    return {"bootstrap_halfwidth": hw, "cases": rows, "b_mis_ln": b_mis,
            "sigma_ln_phi": sig_phi,
            "sigma_mu_ln": float(np.sqrt(0.5 ** 2 + sig_phi ** 2 + b_mis ** 2)),
            "rule": "sigma_mu = sqrt(0.5^2 + sigma_ln_phi^2 + b_mis^2) (pre-registered)"}


def main(out_dir: str) -> int:
    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Dp, Df = dmod.load("preorigin"), dmod.load("full")
    assert Dp["d"].index.max() <= pd.Timestamp("2022-12-01")
    Fp, Ff = prod.fits(Dp), prod.fits(Df)
    V = {"prereg": "results/dataprep/preregistration/P1_county_electricity_2026-10-12.json"}
    V["T1_BV1"] = matched_lead(Dp["d"], ("2019-01", "2022-12"), ("2019-01", "2022-12"),
                               [2016, 2017, 2018])
    V["T2_BV2"] = matched_lead(Df["d"], ("2020-01", "2023-12"), ("2020-01", "2026-08"),
                               [2016, 2017, 2018, 2019])
    V["T3_BV_LOYO"] = loyo_backcast(Df["d"])
    V.update(ext_tests(Df["d"]))
    syn_p = syn_window(Dp, Fp, cases=True)
    syn_f = syn_window(Df, Ff, cases=True)
    V["T8_SYN_coverage"] = {"preorigin": syn_p["cases"]["M0_correct"]["coverage"] | {
        "secondary": syn_p["cases"]["M0_correct"]["coverage_secondary_block_noise"],
        "remedy_reported_only": syn_p["cases"]["M0_correct"]["remedy_reported_only"],
        "noise": syn_p["noise"]},
        "full": syn_f["cases"]["M0_correct"]["coverage"] | {
        "secondary": syn_f["cases"]["M0_correct"]["coverage_secondary_block_noise"],
        "remedy_reported_only": syn_f["cases"]["M0_correct"]["remedy_reported_only"],
        "noise": syn_f["noise"]}}
    mis = {"preorigin": {"synthetic": syn_p, "fold": fold_mis(syn_p, Fp)},
           "full": {"synthetic": syn_f, "fold": fold_mis(syn_f, Ff)}}
    V["T9_MIS"] = {k: v["fold"] for k, v in mis.items()}
    V["production_fits"] = {k: {"sep": F["sep"], "irr": {x: F["irr"][x] for x in
                                                          ("a", "b", "r2", "a_lo", "a_hi")}}
                            for k, F in (("preorigin", Fp), ("full", Ff))}
    V["gates"] = {
        "preorigin": {"T1_BV1": V["T1_BV1"]["pass"],
                      "T8_SYN_coverage": V["T8_SYN_coverage"]["preorigin"]["pass"],
                      "T9_folded": True},
        "full": {"T2_BV2": V["T2_BV2"]["pass"], "T4_EXT_LOYO": V["T4_EXT_LOYO"]["pass"],
                 "T8_SYN_coverage": V["T8_SYN_coverage"]["full"]["pass"], "T9_folded": True},
    }
    for g in V["gates"].values():
        g["passed"] = bool(all(v for k, v in g.items()))
    V["seconds"] = round(time.time() - t0, 1)
    (od / "validation.json").write_text(json.dumps(V, ensure_ascii=False, indent=1,
                                                   default=float))
    (od / "misspecification.json").write_text(json.dumps(mis, ensure_ascii=False, indent=1,
                                                         default=float))
    print(json.dumps({k: V[k] for k in ("T1_BV1", "T2_BV2", "gates")}, indent=1,
                     default=float))
    print(json.dumps(V["T4_EXT_LOYO"], indent=1, default=float))
    print(json.dumps(V["T8_SYN_coverage"], indent=1, default=float))
    print(json.dumps(V["T9_MIS"], indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
