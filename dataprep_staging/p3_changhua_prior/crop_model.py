"""WP P3 step 2: crop-statistics township model, fitted on YUNLIN townships only, pre-origin
survey years 2012-2022 (ROC 101-111); predictions for Changhua townships (no Changhua census
value is read here).

Target: whole-township irrigation-class kWh per calendar year of all kept Yunlin census pumps
(in grid or not), C0 pump_electricity/preorigin/v3 pumps_yunlin.npz joined to census_L1/v3
township names (normalised to the MOA spelling, lib.norm_town).
Regressors: MOA 038 / J51 crop-season areas (ha) rice1, rice2, upland, veg, fruit, special,
flower, forage (lib.crop_table, the pilot's grouping).

Variants:
- ``plan``: per-year NNLS shrunk towards the pooled 2012-2022 NNLS (plan P3 step 2). The
  penalty is lam * sum_k ||X_k||^2 (beta_k - beta_pool_k)^2 (augmented NNLS, beta >= 0);
  lam is chosen from LAMBDAS by Yunlin leave-one-township-out R2 pooled over town-years
  (township held out of both the pooled and the per-year fit). Design choice made on Yunlin
  only, before any Changhua look.
- ``pilot``: the pilot's township_crops.py exactly (per-year NNLS, no shrinkage, years
  2014/15/16/21, census names joined unnormalised so 臺西鄉 drops out), for reproduction.
- ``onerate``: one kWh per planted ha-season (the pilot's comparison).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frame  # noqa: E402
import lib  # noqa: E402

YEARS = list(range(2012, 2023))
LAMBDAS = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, np.inf]
FEATS = lib.CROP_FEATS


def town_kwh(county: str, use: str, normalise: bool = True) -> pd.DataFrame:
    """Whole-township kWh per (town, class, year) of all kept pumps of a county (in grid or
    not), plus an in-grid-only column; from C0 preorigin/v3 per-pump arrays."""
    p = frame.open_c0(lib.C0_PRE, use)
    z = np.load(p.path(f"pumps_{county}.npz"), allow_pickle=True)
    L = frame.census_attrs(county, use, ["uid", "town"]).set_index("uid")
    town = L.loc[z["uid"], "town"].to_numpy()
    if normalise:
        town = np.array([lib.norm_town(lib.COUNTY_ZH[county], t) for t in town], dtype=object)
    else:
        town = np.array([lib.COUNTY_ZH[county] + str(t) for t in town], dtype=object)
    dates = np.asarray(z["dates"]).astype(str)
    yrs = np.array([int(d[:4]) for d in dates])
    v = np.nan_to_num(z["value"].astype(float))
    rows = []
    cls = np.asarray(z["purpose_class"]).astype(str)
    ing = np.asarray(z["cell"]) >= 0
    for y in YEARS:
        e = v[:, yrs == y].sum(1)
        df = pd.DataFrame({"town": town, "cls": cls, "kwh": e, "kwh_ingrid": e * ing})
        g = df.groupby(["town", "cls"])[["kwh", "kwh_ingrid"]].sum().reset_index()
        g["year"] = y
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def design(t: pd.DataFrame, year: int, county_zh: str) -> pd.DataFrame:
    ct = t.loc[lib.roc(year)]
    return ct[ct.index.str.startswith(county_zh)][FEATS]


def fit_nnls(X, Y):
    b, _ = nnls(X, Y)
    return b


def fit_shrunk(X, Y, b_pool, lam):
    if lam == 0:
        return fit_nnls(X, Y)
    if not np.isfinite(lam):
        return b_pool.copy()
    s = np.sqrt((X ** 2).sum(0))
    Xa = np.vstack([X, np.sqrt(lam) * np.diag(s)])
    Ya = np.concatenate([Y, np.sqrt(lam) * s * b_pool])
    return fit_nnls(Xa, Ya)


def yunlin_panel(t: pd.DataFrame, tk: pd.DataFrame, years=YEARS):
    """{year: (towns, X (n, k), Y (n,))} for Yunlin townships present in both tables."""
    out = {}
    irr = tk[tk.cls == "irrigation"]
    for y in years:
        X = design(t, y, "雲林縣")
        yy = irr[irr.year == y].groupby("town")["kwh"].sum()
        j = X.join(yy.rename("kwh"), how="inner")
        out[y] = (list(j.index), j[FEATS].to_numpy(float), j["kwh"].to_numpy(float))
    return out


def pooled(panel, exclude=None, years=None):
    Xs, Ys = [], []
    for y, (towns, X, Y) in panel.items():
        if years is not None and y not in years:
            continue
        m = np.array([tn != exclude for tn in towns])
        Xs.append(X[m])
        Ys.append(Y[m])
    return fit_nnls(np.vstack(Xs), np.concatenate(Ys))


def loto_predictions(panel, lam):
    """{year: loo predictions per town} for the shrunk per-year model (held-out township
    removed from both fits)."""
    towns_all = sorted({tn for v in panel.values() for tn in v[0]})
    bp = {tn: pooled(panel, exclude=tn) for tn in towns_all}
    out = {}
    for y, (towns, X, Y) in panel.items():
        p = np.zeros(len(Y))
        for i, tn in enumerate(towns):
            m = np.arange(len(Y)) != i
            b = fit_shrunk(X[m], Y[m], bp[tn], lam)
            p[i] = X[i] @ b
        out[y] = p
    return out


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def select_lambda(panel, lambdas=LAMBDAS) -> tuple[float, dict]:
    scores = {}
    for lam in lambdas:
        pr = loto_predictions(panel, lam)
        Y = np.concatenate([panel[y][2] for y in panel])
        P = np.concatenate([pr[y] for y in panel])
        scores[str(lam)] = r2(Y, P)
    best = max(scores.values())
    lam = max(lam for lam in lambdas if scores[str(lam)] >= best - 1e-12)
    return lam, scores


def onerate_loo(X, Y):
    tot = X.sum(1)
    return np.array([tot[i] * np.delete(Y, i).sum() / np.delete(tot, i).sum()
                     for i in range(len(Y))])


def fit_all(t, tk, lam, panel=None):
    """Per-year shrunk betas fitted on all Yunlin townships (plus the pooled beta)."""
    panel = panel or yunlin_panel(t, tk)
    bp = pooled(panel)
    betas = {y: fit_shrunk(X, Y, bp, lam) for y, (tw, X, Y) in panel.items()}
    return betas, bp


def predict_changhua(t, betas) -> pd.DataFrame:
    """Whole-township irrigation kWh per Changhua township and year (crop model)."""
    rows = []
    for y, b in betas.items():
        X = design(t, y, "彰化縣")
        for tn, x in zip(X.index, X.to_numpy(float), strict=True):
            rows.append({"town": tn, "year": y, "kwh_irr_whole_pred": float(x @ b)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- pilot variant
PILOT_YEARS = (2014, 2015, 2016, 2021)


def pilot_variant(t: pd.DataFrame, tk_raw: pd.DataFrame) -> dict:
    """township_crops.py: per-year NNLS on Yunlin townships joined unnormalised (the census
    spelling 台西鄉 does not match MOA's 臺西鄉, so that township drops out), LOO R2 and the
    one-rate LOO R2; Changhua whole-township predictions."""
    out = {}
    irr = tk_raw[tk_raw.cls == "irrigation"]
    for y in PILOT_YEARS:
        ct = t.loc[lib.roc(y)]
        yy = irr[irr.year == y].groupby("town")["kwh"].sum()
        j = ct.join(yy.rename("kwh_irr"), how="inner")
        j = j[j.index.str.startswith("雲林縣")]
        X, Y = j[FEATS].to_numpy(float), j["kwh_irr"].to_numpy(float)
        beta = fit_nnls(X, Y)
        loo = np.zeros_like(Y)
        for i in range(len(Y)):
            m = np.arange(len(Y)) != i
            loo[i] = X[i] @ fit_nnls(X[m], Y[m])
        ch = ct[ct.index.str.startswith("彰化縣")]
        out[y] = {"n_towns": int(len(Y)), "towns": list(j.index), "beta": beta.tolist(),
                  "loo_r2": r2(Y, loo), "onerate_loo_r2": r2(Y, onerate_loo(X, Y)),
                  "changhua_whole_pred": dict(zip(ch.index, (ch[FEATS].to_numpy(float) @ beta)
                                                  .tolist(), strict=True))}
    return out


def main() -> None:
    t = lib.crop_table()
    tk = town_kwh("yunlin", "prior")
    tk_raw = town_kwh("yunlin", "prior", normalise=False)
    panel = yunlin_panel(t, tk)
    lam, lam_scores = select_lambda(panel)
    pr = loto_predictions(panel, lam)
    pr0 = loto_predictions(panel, 0.0)
    prp = loto_predictions(panel, np.inf)
    per_year = {}
    for y, (towns, X, Y) in panel.items():
        per_year[y] = {"n_towns": len(towns), "loo_r2_shrunk": r2(Y, pr[y]),
                       "loo_r2_per_year_nnls": r2(Y, pr0[y]), "loo_r2_pooled": r2(Y, prp[y]),
                       "loo_r2_onerate": r2(Y, onerate_loo(X, Y)),
                       "yunlin_irr_GWh": float(Y.sum() / 1e6)}
    betas, bp = fit_all(t, tk, lam, panel)
    pred = predict_changhua(t, betas)
    pil = pilot_variant(t, tk_raw)
    res = {"lambda_grid_loto_r2": lam_scores, "lambda": lam, "per_year": per_year,
           "beta_pooled": dict(zip(FEATS, bp.tolist(), strict=True)),
           "betas": {str(y): dict(zip(FEATS, b.tolist(), strict=True)) for y, b in betas.items()},
           "pilot_variant": {str(k): v for k, v in pil.items()},
           "yunlin_towns": panel[YEARS[0]][0]}
    lib.WORK.mkdir(parents=True, exist_ok=True)
    pred.to_csv(lib.WORK / "crop_model_changhua_pred.csv", index=False)
    tk.to_csv(lib.WORK / "yunlin_town_kwh.csv", index=False)
    lib.save_json(lib.EVID / "crop_model.json", res)
    print("lambda", lam, lam_scores)
    for y, v in per_year.items():
        print(y, {k: round(x, 3) if isinstance(x, float) else x for k, x in v.items()})
    for y, v in pil.items():
        print("pilot", y, v["n_towns"], round(v["loo_r2"], 3), round(v["onerate_loo_r2"], 3))


if __name__ == "__main__":
    main()
