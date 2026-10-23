"""Census pumps -> share of pumped electricity drawn from each twin layer (L2 product).

Join key: the township (鄉鎮市) of each census pump, a public administrative unit. No
personal column is read (CENSUS_COLS lists the columns; names, addresses and postal codes
are never opened). Each kept, in-grid pump installed by 2022-12 carries an energy weight
``duty_2012_2022 x pump_hp`` (proportional to its 2012-2022 mean draw) and its township's
private-well split over aquifers 1-4 from Lin (2012), two members:

* ``census_initial`` (Table 4-10): the 2000-2008 YunTech/ARDC well census;
* ``calibrated`` (Table 4-14): the same after MODFLOW calibration on 2005-2009 heads.

A learning-from-label-proportions (LLP) logistic model predicts each pump's probability of
drawing from aquifer 1 from its census attributes (horsepower, tube diameter, purpose,
install year, position) so that the energy-weighted mean over a township's pumps matches
that township's printed share. It is scored by leave-one-spatial-block-out CV (k-means
blocks of townships) against the county mean of the training townships. Its use in the
product is decided by that score (``choose_within_township``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import x3const as K
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor

SEED = 2027
N_BLOCKS = 6
L2 = 1.0                       # ridge penalty of the LLP logistic (standardised features)
FEATURES = ["log_hp", "tube_dia", "dia_missing", "aquaculture", "domestic", "livestock",
            "other", "install_year", "install_missing", "x_km", "y_km"]
ATTR_ONLY = [f for f in FEATURES if f not in ("x_km", "y_km")]


def load_pumps() -> pd.DataFrame:
    c = pd.read_parquet(K.CENSUS_L1 / "census_pumps.parquet", columns=K.CENSUS_COLS)
    c = c[c["county"].isin(["changhua", "yunlin"]) & c["kept"].astype(bool)
          & c["in_grid"].astype(bool) & c["cell"].notna()].copy()
    inst = pd.to_datetime(c["install_month"], format="%Y-%m", errors="coerce")
    c = c[inst.isna() | (inst <= pd.Timestamp(K.ORIGIN + "-01"))].copy()
    inst = pd.to_datetime(c["install_month"], format="%Y-%m", errors="coerce")
    c["w"] = (pd.to_numeric(c["meter_duty_2012_2022"], errors="coerce").fillna(0)
              * pd.to_numeric(c["pump_hp"], errors="coerce").fillna(0))
    c = c[c["w"] > 0].copy()
    c["town_key"] = c["town"].map(K.town_key)
    hp = pd.to_numeric(c["pump_hp"], errors="coerce")
    c["log_hp"] = np.log1p(hp.clip(0, 100))
    dia = pd.to_numeric(c["tube_dia_raw"], errors="coerce")
    dia = dia.where((dia > 0) & (dia <= 24))
    c["dia_missing"] = dia.isna().astype(float)
    c["tube_dia"] = dia.fillna(dia.median())
    for cls in ("aquaculture", "domestic", "livestock"):
        c[cls] = (c["purpose_class"] == cls).astype(float)
    c["other"] = c["purpose_class"].isin(["other", "industry"]).astype(float)
    yr = inst.dt.year + inst.dt.month / 12.0
    c["install_missing"] = yr.isna().astype(float)
    c["install_year"] = yr.fillna(yr.median())
    c["x_km"] = c["x_twd97"] / 1000.0
    c["y_km"] = c["y_twd97"] / 1000.0
    c["cell"] = c["cell"].astype(int)
    return c


def lin_table(evidence_csv) -> pd.DataFrame:
    t = pd.read_csv(evidence_csv)
    t = t[t["county"].isin(["changhua", "yunlin"])].copy()
    # renormalise the printed 2-decimal shares to sum to one
    a = t[["a1", "a2", "a3", "a4"]].to_numpy(float)
    t[["a1", "a2", "a3", "a4"]] = a / a.sum(1, keepdims=True)
    return t


# ----------------------------------------------------------------------------- LLP model
class LLP:
    """Logistic P(aquifer 1 | pump attributes); fitted so that each township's
    energy-weighted mean matches its share (squared error, equal township weights)."""

    def __init__(self, features: list[str], l2: float = L2):
        self.features, self.l2 = list(features), l2

    def _X(self, p: pd.DataFrame) -> np.ndarray:
        X = p[self.features].to_numpy(float)
        return (X - self.mu) / self.sd

    def fit(self, pumps: pd.DataFrame, shares: pd.Series) -> LLP:
        p = pumps[pumps["town_key"].isin(shares.index)]
        X0 = p[self.features].to_numpy(float)
        self.mu, self.sd = X0.mean(0), X0.std(0) + 1e-9
        X = self._X(p)
        towns = p["town_key"].to_numpy()
        ut = np.array(sorted(shares.index))
        idx = np.searchsorted(ut, towns)
        w = p["w"].to_numpy(float)
        wt = np.bincount(idx, weights=w, minlength=len(ut))
        s = shares.reindex(ut).to_numpy(float)
        s = np.clip(s, 1e-3, 1 - 1e-3)

        def obj(beta):
            z = beta[0] + X @ beta[1:]
            pr = 1.0 / (1.0 + np.exp(-z))
            sh = np.bincount(idx, weights=w * pr, minlength=len(ut)) / wt
            r = sh - s
            g_sh = 2 * r / len(ut)
            gpr = g_sh[idx] * w / wt[idx] * pr * (1 - pr)
            grad = np.r_[gpr.sum(), X.T @ gpr] + np.r_[0, 2 * self.l2 * beta[1:] / len(ut)]
            return float((r ** 2).mean() + self.l2 * (beta[1:] ** 2).sum() / len(ut)), grad

        b0 = np.zeros(X.shape[1] + 1)
        b0[0] = np.log(s.mean() / (1 - s.mean()))
        res = minimize(obj, b0, jac=True, method="L-BFGS-B")
        self.beta = res.x
        return self

    def predict_pump(self, pumps: pd.DataFrame) -> np.ndarray:
        z = self.beta[0] + self._X(pumps) @ self.beta[1:]
        return 1.0 / (1.0 + np.exp(-z))

    def predict_town(self, pumps: pd.DataFrame) -> pd.Series:
        pr = self.predict_pump(pumps)
        g = pd.DataFrame({"t": pumps["town_key"].to_numpy(), "wp": pumps["w"] * pr,
                          "w": pumps["w"]}).groupby("t")[["wp", "w"]].sum()
        return g["wp"] / g["w"]


def town_features(pumps: pd.DataFrame) -> pd.DataFrame:
    """Energy-weighted township means of the pump features (for the GBM benchmark)."""
    w = pumps["w"]
    return pumps[FEATURES].mul(w, axis=0).groupby(pumps["town_key"]).sum().div(
        w.groupby(pumps["town_key"]).sum(), axis=0)


def town_blocks(pumps: pd.DataFrame, towns: list[str], k: int = N_BLOCKS) -> pd.Series:
    cen = pumps.groupby("town_key")[["x_km", "y_km"]].mean().reindex(towns)
    lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(cen.to_numpy())
    return pd.Series(lab, index=towns)


def cross_validate(pumps: pd.DataFrame, table: pd.DataFrame, variant: str) -> dict:
    t = table[table["variant"] == variant].set_index("town")
    towns = [x for x in t.index if x in set(pumps["town_key"])]
    blocks = town_blocks(pumps, towns)
    share = t.loc[towns, "a1"]
    county = t.loc[towns, "county"]
    e_town = pumps.groupby("town_key")["w"].sum().reindex(towns)
    preds = {"county_mean": pd.Series(np.nan, index=towns),
             "llp_attributes": pd.Series(np.nan, index=towns),
             "llp_attributes_xy": pd.Series(np.nan, index=towns),
             "gbm_town_means": pd.Series(np.nan, index=towns)}
    tf = town_features(pumps).reindex(towns)
    for b in sorted(blocks.unique()):
        te = blocks.index[blocks == b]
        tr = blocks.index[blocks != b]
        for c in ("changhua", "yunlin"):
            trc = [x for x in tr if county[x] == c]
            tec = [x for x in te if county[x] == c]
            if tec:
                ww = e_town[trc]
                preds["county_mean"][tec] = float((share[trc] * ww).sum() / ww.sum())
        ptr = pumps[pumps["town_key"].isin(tr)]
        pte = pumps[pumps["town_key"].isin(te)]
        for name, feats in (("llp_attributes", ATTR_ONLY), ("llp_attributes_xy", FEATURES)):
            m = LLP(feats).fit(ptr, share[tr])
            preds[name][te] = m.predict_town(pte).reindex(te).to_numpy()
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                       subsample=0.8, random_state=SEED)
        gb.fit(tf.loc[tr].to_numpy(), share[tr].to_numpy(), sample_weight=None)
        preds["gbm_town_means"][te] = np.clip(gb.predict(tf.loc[te].to_numpy()), 0, 1)
    out = {"variant": variant, "n_towns": len(towns), "n_blocks": N_BLOCKS}
    for name, pr in preds.items():
        err = pr - share
        out[name] = {
            "rmse_equal_towns": float(np.sqrt((err ** 2).mean())),
            "rmse_energy_weighted": float(np.sqrt((err ** 2 * e_town).sum() / e_town.sum())),
            "county_total_error": {c: float(((pr - share) * e_town)[county == c].sum()
                                            / e_town[county == c].sum())
                                   for c in ("changhua", "yunlin")},
            "spearman_vs_truth": float(spearmanr(pr, share).statistic)}
    out["_preds"] = pd.DataFrame(preds).assign(truth=share, block=blocks, county=county)
    return out


def choose_within_township(cv: dict) -> str:
    """LLP within-township variation is used only if it beats the county mean out of block
    on both RMSE measures with a positive rank correlation (> 0.3) to the held-out townships;
    otherwise each pump takes its township's printed split."""
    base = cv["county_mean"]
    for name in ("llp_attributes", "llp_attributes_xy", "gbm_town_means"):
        m = cv[name]
        if (m["rmse_equal_towns"] < base["rmse_equal_towns"]
                and m["rmse_energy_weighted"] < base["rmse_energy_weighted"]
                and m["spearman_vs_truth"] > 0.3):
            return name
    return "township_table"


# ----------------------------------------------------------------------------- products
def cell_town_table(pumps: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Per pump: the township split of both members (rows of townships Lin does not list,
    e.g. in-grid pumps of neighbouring townships, take their county's energy-weighted mean)."""
    rows = []
    for v in ("census_initial", "calibrated"):
        t = table[table["variant"] == v].set_index("town")
        a = t[["a1", "a2", "a3", "a4"]]
        p = pumps[["town_key", "county", "cell", "w"]].copy()
        j = p.join(a, on="town_key")
        miss = j["a1"].isna()
        for c in ("changhua", "yunlin"):
            mc = (p["county"] == c) & ~miss
            cm = (j.loc[mc, ["a1", "a2", "a3", "a4"]].mul(j.loc[mc, "w"], axis=0).sum()
                  / j.loc[mc, "w"].sum())
            for k in ("a1", "a2", "a3", "a4"):
                j.loc[miss & (p["county"] == c), k] = cm[k]
        j["filled_from_county"] = miss
        j["variant"] = v
        rows.append(j)
    return pd.concat(rows)


def cell_product(pt: pd.DataFrame, n_active: int, sd_floor: float) -> dict:
    """(4, A) energy-weighted share of each twin layer per cell, mean of the two members,
    sd = sqrt((half the member difference)^2 + sd_floor^2), clipped to [0, 0.5]."""
    shares = {}
    for v, g in pt.groupby("variant"):
        w = g["w"].to_numpy(float)
        cell = g["cell"].to_numpy(int)
        tot = np.bincount(cell, weights=w, minlength=n_active)
        m = np.zeros((4, n_active))
        for k, col in enumerate(("a1", "a2", "a3", "a4")):
            m[k] = np.bincount(cell, weights=w * g[col].to_numpy(float), minlength=n_active)
        with np.errstate(invalid="ignore", divide="ignore"):
            shares[v] = np.where(tot > 0, m / tot, np.nan)
        energy = tot
    a, b = shares["census_initial"], shares["calibrated"]
    mean = 0.5 * (a + b)
    sd = np.sqrt((0.5 * (a - b)) ** 2 + sd_floor ** 2)
    sd = np.where(np.isfinite(mean), np.minimum(sd, 0.5), np.nan)
    filled = pt[pt["variant"] == "calibrated"].groupby("cell")["filled_from_county"].mean()
    ff = np.zeros(n_active)
    ff[filled.index.to_numpy(int)] = filled.to_numpy()
    return {"mean": mean, "sd": sd, "census_initial": a, "calibrated": b,
            "energy_weight": energy, "frac_energy_county_filled": ff}


def county_split(pt: pd.DataFrame, table: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """County private-well split per member, with a township bootstrap (90 % interval)."""
    rng = np.random.default_rng(SEED)
    out = []
    for v in ("census_initial", "calibrated"):
        g = pt[pt["variant"] == v]
        for c in ("changhua", "yunlin"):
            gc = g[g["county"] == c]
            by_t = gc.groupby("town_key").apply(
                lambda d: pd.Series({"w": d["w"].sum(),
                                     **{k: (d[k] * d["w"]).sum() for k in
                                        ("a1", "a2", "a3", "a4")}}), include_groups=False)
            est = by_t[["a1", "a2", "a3", "a4"]].sum() / by_t["w"].sum()
            boots = []
            for _ in range(n_boot):
                s = by_t.iloc[rng.integers(0, len(by_t), len(by_t))]
                boots.append((s[["a1", "a2", "a3", "a4"]].sum() / s["w"].sum()).to_numpy())
            boots = np.array(boots)
            rec = {"variant": v, "county": c, "n_towns": int(len(by_t))}
            for k, col in enumerate(("a1", "a2", "a3", "a4")):
                rec[col] = float(est[col])
                rec[col + "_lo90"] = float(np.percentile(boots[:, k], 5))
                rec[col + "_hi90"] = float(np.percentile(boots[:, k], 95))
            out.append(rec)
    return pd.DataFrame(out)


def wra_rank_check(pred_deep: pd.Series, wra: pd.DataFrame) -> dict:
    """Independent check: township deep share (1 - aquifer-1 share) against the WRA count
    share of wells deeper than 100 m (a different survey and definition: rank only)."""
    w = wra.set_index("town")
    j = pd.DataFrame({"pred": pred_deep}).join(w[["deep_count_share",
                                                 "deep_from_water_rights_register"]],
                                                how="inner")
    r_all = spearmanr(j["pred"], j["deep_count_share"]).statistic
    jj = j[~j["deep_from_water_rights_register"]]
    r_survey = spearmanr(jj["pred"], jj["deep_count_share"]).statistic if len(jj) > 4 \
        else np.nan
    return {"n_towns": int(len(j)), "spearman_all": float(r_all),
            "n_towns_survey_counts": int(len(jj)), "spearman_survey_counts": float(r_survey)}
