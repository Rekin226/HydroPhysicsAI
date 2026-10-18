"""Land-use transfer models of the 2026-10-12 pilots, ported with their logic, features and
hyperparameters unchanged (design-time choices; nothing here was tuned on Changhua):

- landuse_crop_priors/models.py: NNLSArea (optional zone interaction), SK (poisson GLM,
  HGB-poisson, RF), the per-class models of apply.py;
- design_ml_transfer/a2_robust_transfer.py: Farm1 (uniform over farmland), NNLS (a2 form),
  Tree (HGB / RF / monotone HGB), Ens, metrics, add_crop, the eight transfer splits;
- a5_class_composite.py: Rate1, Composite (rf_total, compA, compB);
- a4_mlp_transfer.py: MLP (sklearn, 5 seeds).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib  # noqa: E402

warnings.filterwarnings("ignore")
FEATS, FARM, GEO, COV = lib.FEATS, lib.FARM, lib.GEO, lib.COV
CROPF = ["c_upland", "c_veg", "c_rice1", "c_rice2", "c_fruit", "c_fallow", "c_intensity"]
CLS4 = ["E_irrigation", "E_aquaculture", "E_livestock", "E_other"]


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


# ----------------------------------------------------------------- landuse_crop_priors models
class NNLSArea:
    """y = sum_k beta_k * area_k (ha), beta >= 0; optional zone interactions."""

    def __init__(self, feats, inter=None):
        self.feats, self.inter = feats, inter

    def design(self, d):
        cols = []
        for k in self.feats:
            if self.inter and k in FARM:
                for lvl in self.inter[1]:
                    cols.append(d[k].to_numpy() * (d[self.inter[0]].to_numpy() == lvl))
            else:
                cols.append(d[k].to_numpy())
        return np.stack(cols, 1)

    def fit(self, d, y):
        self.beta, _ = nnls(self.design(d), y)
        return self

    def predict(self, d):
        return self.design(d) @ self.beta


class SK:
    def __init__(self, kind, feats):
        self.kind, self.feats = kind, feats

    def X(self, d):
        X = d[self.feats].copy()
        for k in FEATS:
            if k in X:
                X[k] = X[k] / 100.0
        return X.to_numpy(float)

    def fit(self, d, y):
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        from sklearn.linear_model import PoissonRegressor
        from sklearn.preprocessing import StandardScaler
        X = self.X(d)
        if self.kind == "poisson":
            self.sc = StandardScaler().fit(X)
            self.m = PoissonRegressor(alpha=1e-3, max_iter=3000).fit(self.sc.transform(X), y / 1e4)
        elif self.kind == "hgb":
            self.m = HistGradientBoostingRegressor(
                loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(X, y / 1e4)
        elif self.kind == "rf":
            self.m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                           n_jobs=1, random_state=0).fit(X, y)
        return self

    def predict(self, d):
        X = self.X(d)
        if self.kind == "poisson":
            return self.m.predict(self.sc.transform(X)) * 1e4
        if self.kind == "hgb":
            return self.m.predict(X) * 1e4
        return self.m.predict(X)


def lu_models():
    """apply.py's five per-class models (summed over CLS4 for a total map)."""
    return {
        "nnls_area": lambda: NNLSArea(FEATS),
        "nnls_area_x_zone": lambda: NNLSArea(FEATS, inter=("zone", (0, 1, 2))),
        "poisson_glm": lambda: SK("poisson", FEATS + COV),
        "hgb_poisson": lambda: SK("hgb", FEATS + COV),
        "rf": lambda: SK("rf", FEATS + COV),
    }


def lu_cv_models():
    """models.py's seven CV models."""
    m = dict(lu_models())
    m["nnls_area_x_coastband"] = lambda: NNLSArea(FEATS, inter=("coastband", (0, 1, 2)))
    m["hgb_landuse_only"] = lambda: SK("hgb", FEATS)
    order = ["nnls_area", "nnls_area_x_zone", "nnls_area_x_coastband", "poisson_glm",
             "hgb_poisson", "rf", "hgb_landuse_only"]
    return {k: m[k] for k in order}


# ----------------------------------------------------------------- design_ml_transfer models
class NNLSa2:
    def __init__(self, feats):
        self.feats = feats

    def fit(self, d, y):
        self.b, _ = nnls(d[self.feats].to_numpy(float), y)
        return self

    def predict(self, d):
        return d[self.feats].to_numpy(float) @ self.b


class Farm1:
    """baseline: one kWh per farmland ha (uniform over farmland)."""

    def fit(self, d, y):
        a = d[FARM].sum(1).to_numpy()
        self.r = y.sum() / a.sum()
        return self

    def predict(self, d):
        return d[FARM].sum(1).to_numpy() * self.r


class Tree:
    def __init__(self, kind, feats, mono=False):
        self.kind, self.feats, self.mono = kind, feats, mono

    def X(self, d):
        X = d[self.feats].copy()
        for k in FEATS:
            if k in X:
                X[k] = X[k] / 100.0
        return X.to_numpy(float)

    def fit(self, d, y):
        from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
        X = self.X(d)
        if self.kind == "hgb":
            cst = [1 if (self.mono and f in FARM) else 0 for f in self.feats]
            self.m = HistGradientBoostingRegressor(
                loss="poisson", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0, monotonic_cst=cst,
                random_state=0).fit(X, y / 1e4)
        else:
            self.m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                           n_jobs=1, random_state=0).fit(X, y)
        return self

    def predict(self, d):
        p = self.m.predict(self.X(d))
        return p * 1e4 if self.kind == "hgb" else p


class Ens:
    def __init__(self, makers):
        self.makers = makers

    def fit(self, d, y):
        self.ms = [mk().fit(d, y) for mk in self.makers]
        return self

    def predict(self, d):
        return np.mean([np.maximum(m.predict(d), 0) for m in self.ms], axis=0)


class MLP:
    def __init__(self, feats, alpha=1e-2, seeds=5, log=True):
        self.feats, self.alpha, self.seeds, self.log = feats, alpha, seeds, log

    def X(self, d):
        X = d[self.feats].copy()
        for k in FEATS:
            if k in X:
                X[k] = X[k] / 100.0
        return X.to_numpy(float)

    def fit(self, d, y):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        X = self.X(d)
        self.sc = StandardScaler().fit(X)
        t = np.log1p(y / 1e3) if self.log else y / 1e5
        self.ms = [MLPRegressor(hidden_layer_sizes=(32, 16), alpha=self.alpha, max_iter=3000,
                                early_stopping=True, validation_fraction=0.15,
                                random_state=s).fit(self.sc.transform(X), t)
                   for s in range(self.seeds)]
        return self

    def predict(self, d):
        Xs = self.sc.transform(self.X(d))
        p = np.mean([m.predict(Xs) for m in self.ms], axis=0)
        return np.expm1(p) * 1e3 if self.log else p * 1e5


A2_MODELS = {
    "farm_uniform": lambda: Farm1(),
    "nnls_LU": lambda: NNLSa2(FEATS),
    "hgb_LU+GEO+X": lambda: Tree("hgb", FEATS + GEO + ["x"]),
    "rf_LU+GEO+X": lambda: Tree("rf", FEATS + GEO + ["x"]),
    "hgb_LU+GEO": lambda: Tree("hgb", FEATS + GEO),
    "rf_LU+GEO": lambda: Tree("rf", FEATS + GEO),
    "hgbmono_LU+GEO": lambda: Tree("hgb", FEATS + GEO, mono=True),
    "hgbmono_LU+GEO+CROP": lambda: Tree("hgb", FEATS + GEO + CROPF, mono=True),
    "rf_LU+GEO+CROP": lambda: Tree("rf", FEATS + GEO + CROPF),
    "ens_robust": lambda: Ens([lambda: NNLSa2(FEATS), lambda: Tree("hgb", FEATS + GEO, True),
                               lambda: Tree("rf", FEATS + GEO)]),
    "ens_scout(with_x)": lambda: Ens([lambda: NNLSa2(FEATS),
                                      lambda: Tree("hgb", FEATS + GEO + ["x"]),
                                      lambda: Tree("rf", FEATS + GEO + ["x"])]),
}
A4_MODELS = {
    "mlp_log_LU+GEO": lambda: MLP(FEATS + GEO),
    "mlp_lin_LU+GEO": lambda: MLP(FEATS + GEO, log=False),
    "mlp_log_LU+GEO+CROP": lambda: MLP(FEATS + GEO + CROPF),
}


class Rate1:
    def __init__(self, feat):
        self.f = feat

    def fit(self, d, y):
        a = d[self.f].to_numpy(float)
        self.b = max(float(a @ y / max(a @ a, 1e-9)), 0.0)
        return self

    def predict(self, d):
        return d[self.f].to_numpy(float) * self.b


class Composite:
    def __init__(self, makers):
        self.makers = makers

    def fit_cls(self, d):
        self.ms = {c: mk().fit(d, d[c].to_numpy(float)) for c, mk in self.makers.items()}
        return self

    def predict_cls(self, d):
        return {c: np.maximum(m.predict(d), 0) for c, m in self.ms.items()}

    def predict(self, d):
        return sum(self.predict_cls(d).values())


def _rf():
    return Tree("rf", FEATS + GEO)


A5_COMPS = {
    "rf_total": None,
    "compA_rf_per_class": {c: _rf for c in CLS4},
    "compB_rf_irr+rates": {"E_irrigation": _rf, "E_aquaculture": lambda: Rate1("aqua"),
                           "E_livestock": lambda: Rate1("livestock"), "E_other": _rf},
}


def a5_fit_predict(name, dtr, dte, cls: bool = False):
    if A5_COMPS[name] is None:
        p = np.maximum(_rf().fit(dtr, dtr["E_tot"].to_numpy(float)).predict(dte), 0)
        return {"E_tot": p} if cls else p
    c = Composite(A5_COMPS[name]).fit_cls(dtr)
    return c.predict_cls(dte) if cls else c.predict(dte)


# ----------------------------------------------------------------- metrics, crop features
def metrics(d, y, p):
    """a2_robust_transfer.metrics (ratio, cell R2/Spearman, 5 km density corr, AA 1/5 km)."""
    p = np.maximum(p, 0)
    out = {"ratio": float(p.sum() / y.sum()), "R2_cell": r2(y, p),
           "spearman_cell": float(spearmanr(y, p)[0])}
    b = d["block5"].to_numpy()
    yb, pb = pd.Series(y).groupby(b).sum(), pd.Series(p).groupby(b).sum()
    nb = pd.Series(1, index=np.arange(len(b))).groupby(b).sum().to_numpy()
    out["corr_5km_density"] = float(np.corrcoef(yb / nb, pb / nb)[0, 1]) if len(yb) > 2 else None
    ys, ps = y / y.sum(), p / max(p.sum(), 1e-12)
    out["AA_1km"] = float(1 - 0.5 * np.abs(ys - ps).sum())
    out["AA_5km"] = float(1 - 0.5 * np.abs(yb / yb.sum() - pb / max(pb.sum(), 1e-12)).sum())
    return out


def add_crop(df, tz, crop_years=("103", "104", "105")):
    """a2's township crop-mix features (pre-origin survey years 2014-16, as the pilot), from
    the P3 MOA download (identical to the pilot's rows for these years)."""
    names = [str(n) for n in tz["names"]]
    t = lib.crop_table()
    ct = t.loc[list(crop_years)].groupby(level=1).mean(numeric_only=True)
    planted = ct[["upland", "veg", "rice1", "rice2", "fruit", "flower", "forage", "special",
                  "other"]].sum(1)
    groups = [str(g) for g in tz["groups"]]
    farm_town = tz["town_class_area_m2"][:, [groups.index(g) for g in
                                             ("paddy", "dry", "orchard")]].sum(1) / 1e4
    tab = pd.DataFrame(index=names)
    for c, src in (("c_upland", "upland"), ("c_veg", "veg"), ("c_rice1", "rice1"),
                   ("c_rice2", "rice2"), ("c_fruit", "fruit"), ("c_fallow", "fallow")):
        tab[c] = (ct[src] / planted).reindex(names)
    tot_area = tz["town_area_total_m2"] / 1e4
    ingrid = tz["town_class_area_m2"].sum(1) / 1e4
    frac_in = np.where(tot_area > 0, ingrid / np.maximum(tot_area, 1e-9), np.nan)
    tab["c_intensity"] = (planted.reindex(names).to_numpy() * frac_in /
                          np.maximum(farm_town, 1e-9))
    ctown = tz["cell_town"]
    for c in CROPF:
        v = tab[c].to_numpy()
        df[c] = np.where(ctown >= 0, v[np.clip(ctown, 0, None)], np.nan)
    return df


def transfer_splits(tr: pd.DataFrame, chlike_towns: list[int]) -> dict:
    """The eight transfer splits of plan P3 (a2/a5): test masks over the training frame."""
    sp = {}
    q1, q2 = np.quantile(tr.x, [1 / 3, 2 / 3])
    sp["EAST3_west"] = (tr.x < q1).to_numpy()
    sp["EAST3_mid"] = ((tr.x >= q1) & (tr.x < q2)).to_numpy()
    sp["EAST3_east"] = (tr.x >= q2).to_numpy()
    sp["CHLIKE"] = tr.town.isin(chlike_towns).to_numpy()
    g = tr.groupby("town")[["paddy", "dry", "orchard", "aqua"]].sum()
    g["pshare"] = g.paddy / g[["paddy", "dry", "orchard"]].sum(1)
    for key, col in (("PADDY", "pshare"), ("AQUA", "aqua")):
        hi_t = list(g[col].sort_values().index[len(g) // 2:])
        m_hi = tr.town.isin(hi_t).to_numpy()
        sp[f"{key}_test_high"] = m_hi
        sp[f"{key}_test_low"] = ~m_hi
    return sp


def propensity(tr: pd.DataFrame, ch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """a2's Changhua-vs-Yunlin propensity (LU + GEO, no coordinates; block5 GroupKFold);
    returns (prop_train, prop_changhua, AUC)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    both = pd.concat([tr.assign(lab=0), ch.assign(lab=1)])
    X = both[FEATS + GEO].to_numpy(float)
    yb = both["lab"].to_numpy()
    prop = np.zeros(len(yb))
    for a, b in GroupKFold(n_splits=5).split(X, yb, both["block5"].to_numpy()):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=20, l2_regularization=1.0,
                                           random_state=0).fit(X[a], yb[a])
        prop[b] = m.predict_proba(X[b])[:, 1]
    return prop[yb == 0], prop[yb == 1], float(roc_auc_score(yb, prop))


def chlike_towns(tr: pd.DataFrame) -> list[int]:
    tp = tr.groupby("town")["prop"].mean().sort_values(ascending=False)
    tn = tr.groupby("town").size()
    cum = np.cumsum([tn[t] for t in tp.index])
    return list(tp.index[: int(np.searchsorted(cum, 0.3 * len(tr))) + 1])
