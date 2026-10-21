"""Depth -> aquifer (twin layer) at any location, learned from the monitoring wells.

The WRA monitoring stations carry a total depth and the CGS aquifer code (1-4 = twin layers
0-3). A gradient-boosting classifier on (depth, x, y) maps a well depth at a location to a
probability over aquifers; it is compared under spatial block cross-validation (10 km
blocks, whole blocks held out) with a location-free depth-threshold baseline (thresholds
fitted on the training folds). The fitted model translates the depth classes printed in
public tables (MOA irrigation-association classes, "deeper than 100 m") into twin layers
for each county, weighted by where the census electricity is.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import x3const as K
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

LAYERS = (1, 2, 3, 4)
BLOCK_M = 10_000.0
N_FOLDS = 5
SEED = 2027


def _xy(s: str) -> tuple[float, float] | None:
    try:
        a, b = (float(v) for v in str(s).split()[:2])
    except (ValueError, TypeError):
        return None
    x, y = (a, b) if a < b else (b, a)            # some records are stored as 'y x'
    if not (140000 <= x <= 240000 and 2580000 <= y <= 2700000):
        return None
    return x, y


def stations() -> pd.DataFrame:
    st = pd.read_parquet(K.STATIONS, columns=K.STATION_COLS)
    st = st[st["CountyName"].isin(list(K.COUNTY_ZH.values()))].copy()
    st["layer_code"] = pd.to_numeric(st["GroundwaterLayerCode"], errors="coerce")
    st["depth_m"] = pd.to_numeric(st["WellDepth"], errors="coerce")
    fd = pd.to_numeric(st["FinishDepth"], errors="coerce")
    st["depth_m"] = st["depth_m"].where(st["depth_m"] > 0, fd)
    xy = st["LocationByTWD97"].map(_xy)
    st = st[xy.notna() & st["layer_code"].isin(LAYERS) & (st["depth_m"] > 0)].copy()
    st["x"] = [v[0] for v in xy[st.index]]
    st["y"] = [v[1] for v in xy[st.index]]
    st["county"] = st["CountyName"].map({v: k for k, v in K.COUNTY_ZH.items()})
    st["site"] = st["sid"].str[:6] + st["sid"].str[7:]
    bx = np.floor(st["x"] / BLOCK_M).astype(int)
    by = np.floor(st["y"] / BLOCK_M).astype(int)
    st["block"] = bx.astype(str) + "_" + by.astype(str)
    return st[["sid", "site", "county", "TownName", "x", "y", "depth_m", "layer_code",
               "block"]].reset_index(drop=True)


def _folds(blocks: pd.Series, k: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    ub = np.array(sorted(blocks.unique()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ub))
    fold_of = {b: int(i % k) for i, b in zip(perm, ub[perm], strict=True)}
    return blocks.map(fold_of).to_numpy()


def _threshold_fit(d: np.ndarray, lay: np.ndarray) -> np.ndarray:
    """Three ordered depth cuts minimising misclassification (grid search, 5 m steps)."""
    grid = np.arange(20, 300, 5.0)
    best, cuts = None, None
    for c1 in grid:
        for c2 in grid[grid > c1]:
            for c3 in grid[grid > c2][::2]:
                pred = 1 + (d > c1) + (d > c2) + (d > c3)
                err = np.mean(pred != lay)
                if best is None or err < best:
                    best, cuts = err, (c1, c2, c3)
    return np.array(cuts)


def _threshold_proba(d: np.ndarray, cuts: np.ndarray, d_tr: np.ndarray,
                     l_tr: np.ndarray) -> np.ndarray:
    """Class frequencies of the training wells in the same depth bin (Laplace-smoothed)."""
    b = np.digitize(d, cuts)
    b_tr = np.digitize(d_tr, cuts)
    out = np.zeros((len(d), 4))
    for i in range(4):
        m = b_tr == i
        cnt = np.array([(l_tr[m] == L).sum() for L in LAYERS], dtype=float) + 0.5
        out[b == i] = cnt / cnt.sum()
    return out


def gbm() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                      subsample=0.8, random_state=SEED)


def cross_validate(st: pd.DataFrame) -> dict:
    X = st[["depth_m", "x", "y"]].to_numpy(float)
    y = st["layer_code"].to_numpy(int)
    folds = _folds(st["block"])
    p_g = np.zeros((len(st), 4))
    p_t = np.zeros((len(st), 4))
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        m = gbm().fit(X[tr], y[tr])
        pp = np.zeros((te.sum(), 4))
        for j, c in enumerate(m.classes_):
            pp[:, LAYERS.index(int(c))] = m.predict_proba(X[te])[:, j]
        p_g[te] = np.clip(pp, 1e-3, 1) / np.clip(pp, 1e-3, 1).sum(1, keepdims=True)
        cuts = _threshold_fit(X[tr, 0], y[tr])
        p_t[te] = _threshold_proba(X[te, 0], cuts, X[tr, 0], y[tr])
    out = {"n_wells": int(len(st)), "n_blocks": int(st["block"].nunique()),
           "n_folds": N_FOLDS, "block_km": BLOCK_M / 1000, "seed": SEED}
    for name, p in (("gbm_depth_xy", p_g), ("depth_thresholds", p_t)):
        pred = np.array(LAYERS)[p.argmax(1)]
        out[name] = {"accuracy": float(accuracy_score(y, pred)),
                     "log_loss": float(log_loss(y, p, labels=list(LAYERS))),
                     "accuracy_layer1_vs_rest": float(np.mean((pred == 1) == (y == 1))),
                     "by_county": {c: float(accuracy_score(y[st["county"] == c],
                                                           pred[st["county"] == c]))
                                   for c in ("changhua", "yunlin")}}
    # bootstrap over blocks for the skill difference
    rng = np.random.default_rng(SEED)
    ub = st["block"].unique()
    ll_g = -np.log(p_g[np.arange(len(y)), np.searchsorted(LAYERS, y)])
    ll_t = -np.log(p_t[np.arange(len(y)), np.searchsorted(LAYERS, y)])
    diffs = []
    for _ in range(1000):
        bs = rng.choice(ub, size=len(ub), replace=True)
        idx = np.concatenate([np.flatnonzero(st["block"].to_numpy() == b) for b in bs])
        diffs.append(ll_t[idx].mean() - ll_g[idx].mean())
    out["log_loss_gain_gbm_vs_thresholds_ci90"] = [float(np.percentile(diffs, 5)),
                                                    float(np.percentile(diffs, 95))]
    return out


class ThresholdModel:
    """Location-free depth -> aquifer model (the CV winner, see :func:`cross_validate`):
    three depth cuts and the smoothed aquifer frequencies of the wells in each depth bin."""

    def __init__(self, st: pd.DataFrame):
        self.d = st["depth_m"].to_numpy(float)
        self.lay = st["layer_code"].to_numpy(int)
        self.cuts = _threshold_fit(self.d, self.lay)

    def proba(self, depth_m: np.ndarray) -> np.ndarray:
        return _threshold_proba(np.asarray(depth_m, float), self.cuts, self.d, self.lay)

    def class_to_layers(self, d_lo: float, d_hi: float, n: int = 20) -> np.ndarray:
        """Aquifer mix of a depth class [d_lo, d_hi), depths uniform within the class."""
        ds = np.linspace(d_lo, d_hi, n, endpoint=False) + (d_hi - d_lo) / (2 * n)
        return self.proba(ds).mean(0)


def depth_layer_table(st: pd.DataFrame) -> pd.DataFrame:
    """Median, quartiles and count of monitoring-well depth by county, zone and aquifer."""
    st = st.assign(zone=st["x"].map(K.zone_of_x))
    g = st.groupby(["county", "zone", "layer_code"])["depth_m"]
    return g.agg(n="count", q25=lambda s: s.quantile(0.25), median="median",
                 q75=lambda s: s.quantile(0.75)).reset_index()
