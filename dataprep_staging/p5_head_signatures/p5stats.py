"""Test statistics of WP P5 (registered in ``P5.signatures.v1``; pre-origin data only).

Every function takes plain tables and returns plain dicts, so the registered runner
(``t_run.py``) and the synthetic tests share one implementation. Definitions:

* Well-level feature: median of the index over the well's pre-origin months (2012-01..2022-12)
  with a value; a well needs WELL_MIN_MONTHS such months.
* Target: log10(mean monthly kWh + 1) over the 132 pre-origin months (census, C0 v3).
* Well tables carry coordinates ``xc``, ``yc`` (TWD97 m) and the target ``y``.
* Cross-well placebo target of well i: median of the targets of the wells j of the
  census-covered universe (Yunlin and Changhua, sectors3 >= 7) with
  PLACEBO_MIN_M < d_ij <= PLACEBO_MAX_M (kWh more than 15 km away, plan WP P5).
* Model: OLS  y = b0 + b1 g(feature) + b2 [layer == 2], g = log10(max(x, 1e-4)) for amplitude
  indices (pilot ``cv_shallow.py``) and the identity for signed indices.
* Spatial-block CV: GroupKFold(5) over 5 km blocks (floor(x / 5 km), floor(y / 5 km)),
  deterministic; per-fold R2 = 1 - SSE / SS around the test fold's mean.
* Permutation null: the feature permuted across wells (layer kept), N_PERM draws, the same
  folds; the per-fold 95th percentile.
* Claim rule (plan WP P5 acceptance): fold k passes when R2_k > perm95_k and R2_k > the
  placebo-target R2_k; a claim needs >= FOLD_PASS_MIN of N_FOLDS folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from p5_head_signatures import config as C

LOG_FEATURES = ("A_S1c_strain", "A_S1c_eq", "A_S1", "dr_med", "dr_p90", "dsd_med", "A_daily",
                "A_S2c", "eemd_annual_amp_m")


def transform(x: np.ndarray, feature: str) -> np.ndarray:
    x = np.asarray(x, float)
    if feature in LOG_FEATURES:
        return np.log10(np.clip(x, C.FEATURE_LOG_FLOOR, None))
    return x


# ----------------------------------------------------------------------------- well tables
def well_features(F: pd.DataFrame, feature: str, months=C.PRE_MONTHS) -> pd.DataFrame:
    """Median of ``feature`` per well over months in the window, with the month count."""
    G = F[(F["month"] >= months[0]) & (F["month"] <= months[1])].dropna(subset=[feature])
    g = G.groupby("sid")[feature]
    return pd.DataFrame({"feature": g.median(), "n_months": g.size()}).reset_index()


def well_targets(L: pd.DataFrame, col: str = C.PRIMARY_TARGET) -> pd.DataFrame:
    Lp = L[(L["month"] >= C.KWH_MONTHS[0]) & (L["month"] <= C.KWH_MONTHS[1])]
    e = Lp.groupby("sid")[col].mean()
    return pd.DataFrame({"sid": e.index, "E_mean": e.to_numpy(),
                         "y": np.log10(e.to_numpy() + 1.0)})


def placebo_targets(W: pd.DataFrame, universe: pd.DataFrame) -> pd.Series:
    """Median universe target over the 15-25 km annulus of each well of ``W``."""
    ux, uy, uyv, us = (universe["xc"].to_numpy(), universe["yc"].to_numpy(),
                       universe["y_target"].to_numpy(), universe["sid"].to_numpy())
    out = []
    for _, w in W.iterrows():
        d = np.hypot(ux - w["xc"], uy - w["yc"])
        s = (d > C.PLACEBO_MIN_M) & (d <= C.PLACEBO_MAX_M) & (us != w["sid"])
        out.append(float(np.median(uyv[s])) if s.any() else np.nan)
    return pd.Series(out, index=W.index)


def blocks(W: pd.DataFrame) -> np.ndarray:
    return ((W["xc"] // C.BLOCK_M).astype(int).astype(str) + "_"
            + (W["yc"] // C.BLOCK_M).astype(int).astype(str)).to_numpy()


# ----------------------------------------------------------------------------- regression
def design(g: np.ndarray, layer2: np.ndarray, with_layer: bool) -> np.ndarray:
    cols = [np.ones(len(g)), g]
    if with_layer:
        cols.append(layer2.astype(float))
    return np.column_stack(cols)


def ols_fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    b, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    return Xte @ b


def r2(y: np.ndarray, p: np.ndarray) -> float:
    ss = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - float(np.sum((y - p) ** 2)) / ss if ss > 0 else np.nan


def fold_ids(groups: np.ndarray, n_folds: int = C.N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=n_folds)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, dummy, groups))


def cv_fold_r2(g, y, layer2, folds, with_layer: bool) -> tuple[np.ndarray, np.ndarray]:
    X = design(g, layer2, with_layer)
    pred = np.full(len(y), np.nan)
    out = []
    for tr, te in folds:
        p = ols_fit_predict(X[tr], y[tr], X[te])
        pred[te] = p
        out.append(r2(y[te], p))
    return np.array(out), pred


def cv_claim(W: pd.DataFrame, feature: str, seed: int, n_perm: int = C.N_PERM) -> dict:
    """The plan's claim rule on well table ``W`` (columns feature, y, y_placebo, layer, xc, yc,
    town)."""
    W = W.dropna(subset=["feature", "y", "y_placebo"]).reset_index(drop=True)
    n = len(W)
    if n < 2 * C.N_FOLDS:
        return {"n_wells": n, "status": "too_few_wells"}
    g = transform(W["feature"].to_numpy(), feature)
    y = W["y"].to_numpy(float)
    yp = W["y_placebo"].to_numpy(float)
    l2 = (W["layer"].astype(str) == "2").to_numpy()
    wl = bool(0 < l2.sum() < n)
    grp = blocks(W)
    folds = fold_ids(grp)
    r2k, pred = cv_fold_r2(g, y, l2, folds, wl)
    r2p, _ = cv_fold_r2(g, yp, l2, folds, wl)
    r2l, pred_l = cv_fold_r2(np.zeros(n), y, l2, folds, wl)    # layer-only baseline
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(folds)))
    for b in range(n_perm):
        null[b], _ = cv_fold_r2(g[rng.permutation(n)], y, l2, folds, wl)
    p95 = np.quantile(null, 0.95, axis=0)
    passes = (r2k > p95) & (r2k > r2p)
    # partial correlation of g with y given the placebo target and layer
    Z = np.column_stack([np.ones(n), yp, l2.astype(float)])
    rg = g - Z @ np.linalg.lstsq(Z, g, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rho, p_rho = stats.spearmanr(g, y)
    try:
        town = W["town"].astype(str).to_numpy()
        logo = list(LeaveOneGroupOut().split(np.zeros(n), np.zeros(n), town))
        _, pred_t = cv_fold_r2(g, y, l2, logo, wl)
        r2_town = r2(y, pred_t)
    except (KeyError, ValueError):
        r2_town = np.nan
    return {"n_wells": int(n), "n_blocks": int(len(set(grp))), "n_layer2": int(l2.sum()),
            "fold_sizes": [int(len(te)) for _, te in folds],
            "fold_r2": r2k.round(4).tolist(), "fold_perm95": p95.round(4).tolist(),
            "fold_placebo_r2": r2p.round(4).tolist(), "fold_layer_only_r2": r2l.round(4).tolist(),
            "fold_pass": passes.tolist(), "n_folds_pass": int(passes.sum()),
            "claim_supported": bool(passes.sum() >= C.FOLD_PASS_MIN),
            "pooled_cv_r2": r2(y, pred), "pooled_cv_r2_layer_only": r2(y, pred_l),
            "pooled_cv_r2_township_logo": r2_town,
            "spearman_feature_target": float(rho), "spearman_p": float(p_rho),
            "spearman_cv_pred": float(stats.spearmanr(pred, y)[0]),
            "partial_r_given_placebo_layer": float(np.corrcoef(rg, ry)[0, 1]),
            "slope_g": float(np.linalg.lstsq(design(g, l2, wl), y, rcond=None)[0][1]),
            "with_layer_dummy": wl,
            "seed": int(seed), "n_perm": int(n_perm)}


def masked_half(W: pd.DataFrame, feature: str, seed: int, n_perm: int = C.N_PERM) -> dict:
    """Changhua-sized masked half (plan): split at the median northing, train on one half,
    test on the other, both directions (reported, not gated)."""
    W = W.dropna(subset=["feature", "y", "y_placebo"]).reset_index(drop=True)
    g = transform(W["feature"].to_numpy(), feature)
    y = W["y"].to_numpy(float)
    yp = W["y_placebo"].to_numpy(float)
    l2 = (W["layer"].astype(str) == "2").to_numpy()
    wl = bool(0 < l2.sum() < len(l2))
    north = W["yc"].to_numpy(float)
    cut = np.median(north)
    halves = {"train_south_test_north": (north <= cut, north > cut),
              "train_north_test_south": (north > cut, north <= cut)}
    rng = np.random.default_rng(seed)
    out = {}
    for name, (tr, te) in halves.items():
        tr, te = np.flatnonzero(tr), np.flatnonzero(te)
        X = design(g, l2, wl)
        rr = r2(y[te], ols_fit_predict(X[tr], y[tr], X[te]))
        rp = r2(yp[te], ols_fit_predict(X[tr], yp[tr], X[te]))
        null = []
        for _ in range(n_perm):
            gp = g[rng.permutation(len(g))]
            Xp = design(gp, l2, wl)
            null.append(r2(y[te], ols_fit_predict(Xp[tr], y[tr], Xp[te])))
        out[name] = {"n_train": int(len(tr)), "n_test": int(len(te)), "r2": rr,
                     "placebo_r2": rp, "perm95": float(np.quantile(null, 0.95)),
                     "beats_both": bool(rr > rp and rr > np.quantile(null, 0.95))}
    return out


def transfer(Wtr: pd.DataFrame, Wte: pd.DataFrame, feature: str, seed: int,
             n_perm: int = C.N_PERM) -> dict:
    """Train on ``Wtr`` (Yunlin), predict ``Wte`` (Changhua): R2 against the test wells'
    own mean, Spearman, permutation null (test features permuted) and placebo."""
    a = Wtr.dropna(subset=["feature", "y", "y_placebo"]).reset_index(drop=True)
    b = Wte.dropna(subset=["feature", "y", "y_placebo"]).reset_index(drop=True)
    ga, gb = transform(a["feature"].to_numpy(), feature), transform(b["feature"].to_numpy(),
                                                                       feature)
    la = (a["layer"].astype(str) == "2").to_numpy()
    lb = (b["layer"].astype(str) == "2").to_numpy()
    wl = bool(0 < la.sum() < len(la))
    Xa, Xb = design(ga, la, wl), design(gb, lb, wl)
    ya, yb = a["y"].to_numpy(float), b["y"].to_numpy(float)
    pa, pb = a["y_placebo"].to_numpy(float), b["y_placebo"].to_numpy(float)
    beta = np.linalg.lstsq(Xa, ya, rcond=None)[0]
    pred = Xb @ beta
    rr = r2(yb, pred)
    beta_p = np.linalg.lstsq(Xa, pa, rcond=None)[0]
    rp = r2(pb, Xb @ beta_p)
    rng = np.random.default_rng(seed)
    null, null_rho = [], []
    for _ in range(n_perm):
        perm = rng.permutation(len(b))
        Xbp = Xb.copy()
        Xbp[:, 1] = Xb[perm, 1]
        null.append(r2(yb, Xbp @ beta))
        null_rho.append(stats.spearmanr(gb[perm], yb)[0])
    rho = float(stats.spearmanr(gb, yb)[0])
    p95 = float(np.quantile(null, 0.95))
    return {"n_train": int(len(a)), "n_test": int(len(b)), "r2_transfer": rr,
            "perm95": p95, "placebo_r2_transfer": rp,
            "claim_supported": bool(rr > p95 and rr > rp),
            "spearman_test": rho,
            "spearman_perm_p_one_sided": float((1 + np.sum(np.array(null_rho) >= rho))
                                               / (1 + n_perm)),
            "corr_pred_obs": float(np.corrcoef(pred, yb)[0, 1]),
            "median_pred_minus_obs_dex": float(np.median(pred - yb)),
            "implied_ratio_test_over_train_median_kwh": float(
                10 ** (np.median(pred) - np.median(Xa @ beta))),
            "observed_ratio_test_over_train_median_kwh": float(
                np.median(b["E_mean"]) / np.median(a["E_mean"])),
            "coef_train": beta.round(5).tolist()}


def rank_check(W: pd.DataFrame, feature: str, seed: int, n_perm: int = C.N_PERM,
               n_boot: int = C.N_BOOT) -> dict:
    W = W.dropna(subset=["feature", "y"]).reset_index(drop=True)
    g = transform(W["feature"].to_numpy(), feature)
    y = W["y"].to_numpy(float)
    rho = float(stats.spearmanr(g, y)[0])
    rng = np.random.default_rng(seed)
    null = np.array([stats.spearmanr(g[rng.permutation(len(g))], y)[0] for _ in range(n_perm)])
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(g), len(g))
        boot.append(stats.spearmanr(g[i], y[i])[0])
    return {"n_wells": int(len(W)), "spearman": rho,
            "perm_p_one_sided": float((1 + np.sum(null >= rho)) / (1 + n_perm)),
            "boot90": [float(np.nanquantile(boot, 0.05)), float(np.nanquantile(boot, 0.95))],
            "boot_note": "well bootstrap percentile interval, uncalibrated (plan 4.7)"}


# ----------------------------------------------------------------------------- within-well
EXPECT = {"mean": -1, "DN": 1, "A_S1c_strain": 1, "dr_med": 1}


def _anom(s: pd.Series) -> pd.Series:
    mo = pd.Index([int(m[5:7]) for m in s.index])
    return s - s.groupby(mo).transform("mean")


def _corr(a: pd.Series, b: pd.Series, min_n: int) -> float:
    m = a.notna() & b.notna()
    if m.sum() < min_n or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def within_well(Fp: pd.DataFrame, Ek: pd.DataFrame, reg: pd.Series, wl: pd.DataFrame,
                index: str, seed: int, n_perm: int = 200) -> dict:
    """Per well: anomaly correlations of ``index`` with local kWh (E_K0_1000), with distant
    wells' kWh (15-25 km, median), with the fan total, and the local-only partial r (fan
    total regressed out of both). ``Fp``: months x wells index table; ``Ek``: months x wells
    local kWh; ``reg``: fan total by month; ``wl``: well list with xc, yc, county."""
    months = Fp.index
    ra = _anom(reg.reindex(months))
    rows = []
    xy = wl.set_index("sid")[["xc", "yc"]]
    Ea = {s: _anom(Ek[s].reindex(months)) for s in Ek.columns if s in xy.index}
    sgn = EXPECT[index]
    rng = np.random.default_rng(seed)
    years = np.array([int(m[:4]) for m in months])
    uy = np.unique(years)
    for sid in wl["sid"]:
        if sid not in Fp or sid not in Ea:
            continue
        x = Fp[sid]
        if x.notna().sum() < C.WITHIN_MIN_MONTHS:
            continue
        xa = _anom(x)
        ea = Ea[sid]
        r_loc = _corr(xa, ea, C.WITHIN_MIN_MONTHS)
        if not np.isfinite(r_loc):
            continue
        d = np.hypot(xy["xc"] - xy.loc[sid, "xc"], xy["yc"] - xy.loc[sid, "yc"])
        far = d[(d > C.PLACEBO_MIN_M) & (d <= C.PLACEBO_MAX_M)].index
        rp = [_corr(xa, Ea[j], C.WITHIN_MIN_MONTHS) for j in far if j in Ea]
        rp = [v for v in rp if np.isfinite(v)]
        r_reg = _corr(xa, ra, C.WITHIN_MIN_MONTHS)
        Z = pd.DataFrame({"x": xa, "e": ea, "r": ra}).dropna()
        A = np.c_[np.ones(len(Z)), Z["r"].to_numpy()]
        xl = Z["x"].to_numpy() - A @ np.linalg.lstsq(A, Z["x"].to_numpy(), rcond=None)[0]
        el = Z["e"].to_numpy() - A @ np.linalg.lstsq(A, Z["e"].to_numpy(), rcond=None)[0]
        r_ls = float(np.corrcoef(xl, el)[0, 1]) if np.std(el) > 0 and np.std(xl) > 0 else np.nan
        # year-shuffle null: calendar years of the local kWh anomaly permuted
        null = []
        for _ in range(n_perm):
            perm = dict(zip(uy, rng.permutation(uy), strict=True))
            idx = pd.Index([f"{perm[int(m[:4])]}-{m[5:7]}" for m in months])
            es = pd.Series(ea.reindex(idx).to_numpy(), index=months)
            null.append(_corr(xa, es, C.WITHIN_MIN_MONTHS))
        rows.append({"sid": sid, "n": int(xa.notna().sum()), "r_local": r_loc,
                     "r_placebo_med": float(np.median(rp)) if rp else np.nan,
                     "n_placebo": len(rp), "r_regional": r_reg, "r_local_specific": r_ls,
                     "r_null_median": float(np.nanmedian(null))})
    T = pd.DataFrame(rows)
    if T.empty:
        return {"n_wells": 0}
    T = T.dropna(subset=["r_local", "r_placebo_med", "r_regional", "r_local_specific"])

    def sign_p(v):
        v = np.asarray(v)
        v = v[np.isfinite(v) & (v != 0)]
        if not len(v):
            return np.nan
        return float(stats.binomtest(int((v > 0).sum()), len(v), 0.5).pvalue)

    dpl = sgn * T["r_local"] - sgn * T["r_placebo_med"]
    drg = sgn * T["r_local"] - sgn * T["r_regional"]
    dnl = sgn * T["r_local"] - sgn * T["r_null_median"]
    ls = sgn * T["r_local_specific"]
    res = {"n_wells": int(len(T)), "expected_sign": sgn,
           "median_r_local": float(T["r_local"].median()),
           "median_r_placebo": float(T["r_placebo_med"].median()),
           "median_r_regional": float(T["r_regional"].median()),
           "median_r_local_specific": float(T["r_local_specific"].median()),
           "median_r_yearshuffle_null": float(T["r_null_median"].median()),
           "frac_local_beats_placebo": float((dpl > 0).mean()),
           "frac_local_beats_regional": float((drg > 0).mean()),
           "sign_p_local_vs_placebo": sign_p(dpl), "sign_p_local_vs_regional": sign_p(drg),
           "sign_p_local_vs_null": sign_p(dnl), "sign_p_local_specific": sign_p(ls),
           "frac_local_specific_expected_sign": float((ls > 0).mean())}
    res["local_signal_beyond_placebos"] = bool(
        np.median(ls) > 0 and res["sign_p_local_specific"] < 0.05
        and np.median(dpl) > 0 and res["sign_p_local_vs_placebo"] < 0.05
        and np.median(drg) > 0 and res["sign_p_local_vs_regional"] < 0.05
        and np.median(dnl) > 0 and res["sign_p_local_vs_null"] < 0.05)
    res["per_well"] = T.round(4).to_dict("list")
    return res


# ----------------------------------------------------------------------------- seasonal CV
def climatology_cv(s: pd.Series) -> float:
    """CV (population sd / mean) of the 12-value calendar-month climatology (median over
    years; each month needs CLIM_MIN_YEARS values); NaN if a month is short."""
    mo = pd.Index([int(m[5:7]) for m in s.index])
    g = s.groupby(mo)
    cnt = g.count()
    if len(cnt) < 12 or (cnt < C.CLIM_MIN_YEARS).any():
        return np.nan
    c = g.median().reindex(range(1, 13)).to_numpy()
    m = float(np.mean(c))
    return float(np.std(c) / m) if m > 0 else np.nan


def county_cv_ratio(cv: pd.DataFrame, seed: int, n_boot: int = C.N_BOOT) -> dict:
    """Median CV Changhua / median CV Yunlin with a well-cluster bootstrap (within county)."""
    a = cv.loc[cv.county == "Changhua", "cv"].dropna().to_numpy()
    b = cv.loc[cv.county == "Yunlin", "cv"].dropna().to_numpy()
    if len(a) < 3 or len(b) < 3:
        return {"n_changhua": int(len(a)), "n_yunlin": int(len(b)), "ratio": np.nan}
    rng = np.random.default_rng(seed)
    bs = [np.median(rng.choice(a, len(a))) / np.median(rng.choice(b, len(b)))
          for _ in range(n_boot)]
    return {"n_changhua": int(len(a)), "n_yunlin": int(len(b)),
            "median_cv_changhua": float(np.median(a)), "median_cv_yunlin": float(np.median(b)),
            "ratio": float(np.median(a) / np.median(b)),
            "boot90": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))],
            "boot_note": "well-cluster bootstrap percentile interval, uncalibrated (plan 4.7)"}


def series_cv(x: np.ndarray, months: list[str]) -> float:
    """CV of the 12-month climatology (mean over years) of a regional census series."""
    mo = np.array([int(m[5:7]) for m in months])
    c = pd.Series(x).groupby(mo).mean().reindex(range(1, 13)).to_numpy()
    return float(np.std(c) / np.mean(c))


# ----------------------------------------------------------------------------- extractor
def epoch_medians(F: pd.DataFrame, cols: list[str], years: tuple[int, int],
                  min_months: int = 6) -> pd.DataFrame:
    yr = F["month"].str[:4].astype(int)
    G = F[(yr >= years[0]) & (yr <= years[1])]
    out = {}
    for c in cols:
        g = G.dropna(subset=[c]).groupby("sid")[c]
        med = g.median()
        out[c] = med.where(g.size() >= min_months)
    return pd.DataFrame(out)


def o1_coherence(F: pd.DataFrame, min_months: int = 12) -> pd.DataFrame:
    G = F.dropna(subset=["gO1", "gO1_raw"])
    rows = []
    for sid, g in G.groupby("sid"):
        if len(g) < min_months:
            continue
        rows.append({"sid": sid, "n": int(len(g)),
                     "coh_corr": float(np.abs(np.exp(1j * g["gO1"].to_numpy()).mean())),
                     "coh_raw": float(np.abs(np.exp(1j * g["gO1_raw"].to_numpy()).mean()))})
    return pd.DataFrame(rows)
