"""WP P1 production fits per variant (choices fixed in the pre-registration
``results/dataprep/preregistration/P1_county_electricity_2026-10-12.json``).

``fits(D)`` returns the source separation (both regressors), the backcast share models, the
transfer ratio, the national sub-industry member and the estimator ratios for one variant.
``series(D, F, V)`` assembles the monthly series (county and in-grid) with sd, flag and
method, given the validation summary ``V`` (for the backcast and extension sd).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as dmod  # noqa: E402
import lib  # noqa: E402

WIN = {"preorigin": ("2016-01", "2022-12"), "full": ("2016-01", "2025-06")}
TREND = {"preorigin": ("2016-01", "2022-12"), "full": ("2016-01", "2026-08")}
MOY_BC = ("2016-01", "2019-12")
BC = ("2012-01", "2015-12")
EXT = ("2025-07", "2026-08")
SUB_CODES = ("011", "012", "013", "032")
SUB_PUMP = ("011", "013", "032")
START = "2012-01"


def _subind(d: pd.DataFrame, sub: pd.DataFrame, t1: str) -> dict:
    """Weak member: NNLS of county ag on national sub-industries (2021-01 on)."""
    cols = [c for c in sub.columns if c[:3] in SUB_CODES]
    X = sub[cols].join(d[["CH_ag", "YL_ag", "YL_clean"]], how="inner").loc[:t1]
    X = X.dropna(subset=cols + ["CH_ag", "YL_ag"])
    out = {"window": [str(X.index.min().date())[:7], str(X.index.max().date())[:7]],
           "n_months": int(len(X)), "codes": [c[:3] for c in cols]}
    pump = {}
    for cty in ("CH_ag", "YL_ag"):
        A = X[cols].to_numpy()
        w, _ = nnls(A, X[cty].to_numpy())
        pred = A @ w
        yv = X[cty].to_numpy()
        r2 = 1 - np.sum((yv - pred) ** 2) / np.sum((yv - yv.mean()) ** 2)
        contrib = {c[:3]: float((w[j] * X[c]).sum() / yv.sum()) for j, c in enumerate(cols)}
        pl = sum(w[j] * X[c] for j, c in enumerate(cols) if c[:3] in SUB_PUMP)
        pump[cty] = pl
        out[cty] = {"weights": {c[:3]: float(w[j]) for j, c in enumerate(cols)},
                    "r2": float(r2), "share_of_county_ag": contrib,
                    "rel_resid_rms": float(np.sqrt(np.mean((yv - pred) ** 2)) / yv.mean())}
    out["ratio_CH_over_YL_subind"] = float(pump["CH_ag"].sum() / pump["YL_ag"].sum())
    yl = X["YL_clean"].dropna()
    out["ratio_CH_subind_over_YL_clean"] = (float(pump["CH_ag"].loc[yl.index].sum() / yl.sum())
                                            if len(yl) else None)
    out["series_CH"] = pump["CH_ag"]
    return out


def fits(D: dict) -> dict:
    d, v = D["d"], D["variant"]
    t0, t1 = WIN[v]
    w, P, N, y, yrs = dmod.nnls_inputs(d, t0, t1)
    sep = lib.nnls2_fit(P, N, y, yrs, n_boot=2000, seed=0)
    resid = pd.Series(y - (sep["a"] * P + sep["b"] * N), index=w.index)
    wi, Pi, Ni, yi, yrsi = dmod.nnls_inputs(d, t0, t1, "YL_clean_irr")
    irr = lib.nnls2_fit(Pi, Ni, yi, yrsi, n_boot=2000, seed=0)
    sh = {"CH": d["CH_ag"] / d["nat_ag"], "YL": d["YL_ag"] / d["nat_ag"]}
    bc = {c: lib.share_model(s, MOY_BC, TREND[v]) for c, s in sh.items()}
    bc_notrend = {c: lib.share_model(s, MOY_BC, TREND[v], trend=False) for c, s in sh.items()}
    r = lib.transfer_ratio(d["YL_clean"], d["YL_ag"], t0, t1)
    rr = (d["YL_clean"] / d["YL_ag"]).loc[t0:t1].dropna()
    r_cv = (rr.groupby(rr.index.month).std(ddof=1) / rr.groupby(rr.index.month).mean())
    sub = _subind(d, D["sub"], t1)
    return {"variant": v, "window": [t0, t1], "sep": sep, "resid": resid, "irr": irr,
            "bc": bc, "bc_notrend": bc_notrend, "r": r, "r_cv": r_cv, "sub": sub,
            "phi": D["phi"]}


def ag_with_backcast(D: dict, F: dict) -> pd.DataFrame:
    """CH_ag, YL_ag observed (2016 on) plus the national-share backcast 2012-15."""
    d = D["d"]
    idx = d.loc[BC[0]:BC[1]].index
    out = d[["CH_ag", "YL_ag", "nat_ag"]].copy()
    out["CH_bc"] = np.nan
    out["YL_bc"] = np.nan
    for c in ("CH", "YL"):
        out.loc[idx, f"{c}_bc"] = d.loc[idx, "nat_ag"].to_numpy() * lib.share_predict(
            F["bc"][c], idx)
    out["is_backcast"] = out.index.isin(idx)
    for c in ("CH", "YL"):
        out[f"{c}_ag_x"] = out[f"{c}_ag"].where(~out["is_backcast"], out[f"{c}_bc"])
    return out


def estimator_ratios(D: dict, F: dict) -> dict:
    """Ratio of each estimator to its own Yunlin counterpart over the NNLS window."""
    d = D["d"]
    t0, t1 = F["window"]
    w = d.loc[t0:t1].dropna(subset=["CH_ag", "YL_ag", "YL_clean"])
    yl = w["YL_clean"].sum()
    shape = (w["CH_ag"] - F["sep"]["b"] * (w["YL_ag"] - w["YL_clean"])).sum() / yl
    transfer = (w["CH_ag"] * F["r"].reindex(w.index.month).to_numpy()).sum() / yl
    return {"scaled": F["sep"]["a"], "shape": float(shape), "transfer": float(transfer),
            "irrigation_only": F["irr"]["a"],
            "national_subindustry_weak": F["sub"]["ratio_CH_over_YL_subind"],
            "CH_ag_over_YL_ag": float(w["CH_ag"].sum() / w["YL_ag"].sum()),
            "window": [t0, t1], "basis": "Changhua pump-like / own Yunlin counterpart"}


def series(D: dict, F: dict, V: dict) -> dict:
    """Monthly series 2012-01..end with sd, flag, method (county and in-grid)."""
    v = D["variant"]
    d = D["d"]
    ag = ag_with_backcast(D, F)
    end = D["end"]
    idx = pd.date_range(START, end, freq="MS", name="date")
    ag = ag.reindex(idx)
    a, b = F["sep"]["a"], F["sep"]["b"]
    cv_a, sd_b = F["sep"]["a_boot_sd"] / a, F["sep"]["b_boot_sd"]
    cv_bc = {c: V["backcast_monthly_rel_rms"][c] for c in ("CH", "YL")}
    phi, sig_phi = F["phi"]["phi_in"], F["phi"]["sigma_ln_phi"]
    yl = d["YL_clean"].reindex(idx)
    yl_flag = pd.Series(np.where(yl.notna(), 0, 8), index=idx)
    yl_method = pd.Series(np.where(yl.notna(), "census_clean_dedupe_cap1", ""), index=idx,
                          dtype=object)
    yl_sd = pd.Series(np.nan, index=idx)
    is_ext = pd.Series(False, index=idx)
    out = {}
    if v == "full":
        e_idx = pd.date_range(EXT[0] + "-01", end, freq="MS")
        r_full = F["r"]
        ext = lib.apply_ratio(r_full, d["YL_ag"].reindex(e_idx))
        yl.loc[e_idx] = ext.to_numpy()
        yl_flag.loc[e_idx] = 5
        yl_method.loc[e_idx] = "tpc_x_rmoy_extension"
        yl_sd.loc[e_idx] = ext.to_numpy() * V["extension_monthly_rel_rms"]
        is_ext.loc[e_idx] = True
    out["YL_clean"] = pd.DataFrame({"value": yl, "sd": yl_sd, "flag": yl_flag,
                                    "method": yl_method, "is_extension": is_ext})
    isbc = ag["is_backcast"].fillna(False).astype(bool)
    N = ag["YL_ag_x"] - yl
    cv_ext = V.get("extension_monthly_rel_rms", 0.0) if v == "full" else 0.0
    est = {}
    # scaled (primary): a * YL_clean; a_pre also carried in the full variant (no splice)
    a_list = [("scaled", a, cv_a)]
    if v == "full" and V.get("a_pre") is not None:
        a_list.append(("scaled_apre", V["a_pre"], V["a_pre_cv"]))
    for name, aa, cva in a_list:
        val = aa * yl
        sd = val * np.sqrt(cva ** 2 + np.where(is_ext, cv_ext ** 2, 0.0))
        est[name] = (val, sd, np.where(is_ext, f"{name}_x_extended_yunlin", f"{name}_a_x_YL_clean"))
    # shape: CH_ag - b N_YL (backcast months use backcast CH_ag and YL_ag)
    val = ag["CH_ag_x"] - b * N
    var = (sd_b * N) ** 2
    var = var + np.where(isbc, (cv_bc["CH"] * ag["CH_ag_x"]) ** 2
                         + (b * cv_bc["YL"] * ag["YL_ag_x"]) ** 2, 0.0)
    var = var + np.where(is_ext, (b * cv_ext * yl) ** 2, 0.0)
    est["shape"] = (val, np.sqrt(var), np.where(isbc, "shape_backcast", np.where(
        is_ext, "shape_extended_yunlin", "shape")))
    # transfer: CH_ag * r(moy)
    rm = F["r"].reindex(idx.month).to_numpy()
    rcv = F["r_cv"].reindex(idx.month).to_numpy()
    val = ag["CH_ag_x"] * rm
    sd = val * np.sqrt(rcv ** 2 + np.where(isbc, cv_bc["CH"] ** 2, 0.0))
    est["transfer"] = (val, sd, np.where(isbc, "transfer_backcast", "transfer"))
    # irrigation-only: a_irr * YL_clean_irr (census months only)
    yi = d["YL_clean_irr"].reindex(idx)
    val = F["irr"]["a"] * yi
    est["irrigation_only"] = (val, val * F["irr"]["a_boot_sd"] / F["irr"]["a"],
                              np.full(len(idx), "irrigation_only"))
    # national sub-industry (weak; 2021 on)
    s = F["sub"]["series_CH"].reindex(idx)
    est["national_subindustry_weak"] = (s, s * F["sub"]["CH_ag"]["rel_resid_rms"],
                                        np.full(len(idx), "national_subindustry_nnls"))
    out["estimators"] = {}
    for name, (val, sd, meth) in est.items():
        val = pd.Series(np.asarray(val, float), index=idx)
        sd = pd.Series(np.asarray(sd, float), index=idx)
        flag = pd.Series(np.where(val.notna(), 5, 8), index=idx)
        out["estimators"][name] = {
            "county": pd.DataFrame({"value": val, "sd": sd, "flag": flag, "method": meth}),
            "in_grid": pd.DataFrame({"value": val * phi,
                                     "sd": val * phi * np.sqrt((sd / val) ** 2 + sig_phi ** 2),
                                     "flag": flag, "method": np.char.add(
                                         np.asarray(meth, dtype=str), "_x_phi_in")}),
        }
    out["ag"] = ag
    out["is_backcast"] = isbc
    out["is_extension"] = is_ext
    return out
