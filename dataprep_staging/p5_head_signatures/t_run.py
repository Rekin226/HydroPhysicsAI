"""Registered tests of WP P5 (pre-registration ``P5.signatures.v1``), run in a CPU slot.

Usage: t_run.py WORKDIR EVIDENCE_DIR TEST_ID [TEST_ID ...]   (or ``all``)

Each test calls ``prereg3.assert_can_score("P5.signatures.v1", "P5", test_id=...)`` immediately
before it computes anything that compares an index with electricity or with the pilot, and
writes EVIDENCE_DIR/tests/<test_id>.json. Every statistic uses months 2012-01..2022-12 only
(U7); the post-origin rows of the feature table are never read here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import prereg3  # noqa: E402

from p5_head_signatures import config as C  # noqa: E402
from p5_head_signatures import inputs as I  # noqa: E402
from p5_head_signatures import p5stats as S  # noqa: E402

PREREG_ID = "P5.signatures.v1"
DAILY_FEATURES = ("A_S1c_strain", "DN", "dr_med")
TESTS = ("P5.t1a_repro", "P5.t1b_epoch", "P5.t2_baro", "P5.t3_yunlin_claims",
         "P5.t4_changhua", "P5.t5_within", "P5.t6_seasonal_cv", "P5.t7_tf_claims",
         "P5.t8_eemd_claims")


def jdump(o) -> str:
    def conv(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.bool_,)):
            return bool(x)
        raise TypeError(type(x))
    return json.dumps(o, indent=1, default=conv)


class Data:
    def __init__(self, work: Path):
        self.work = work
        F = pd.read_parquet(work / "well_month_features.parquet")
        self.F = F[F["month"] <= C.ORIGIN].copy()            # pre-origin rows only (U7)
        meta = I.station_meta()
        km = pd.read_parquet(work / "wells_kwh_meta.parquet")
        km = km.rename(columns={"x": "xc", "y": "yc"})
        km = km.merge(meta[["sid", "TownName", "layer_flag"]], on="sid", how="left")
        km["town"] = km["TownName"].astype(str)
        self.km = km
        L = pd.read_parquet(work / "local_kwh.parquet")
        self.L = L[(L["month"] >= C.KWH_MONTHS[0]) & (L["month"] <= C.KWH_MONTHS[1])]
        reg = pd.read_parquet(work / "regional_kwh.parquet")
        self.reg = reg.set_index("month")
        uni = km[km.county.isin(["Yunlin", "Changhua"]) & (km.sectors3 >= C.SECTORS_MIN)]
        t = S.well_targets(self.L, C.PRIMARY_TARGET).rename(columns={"y": "y_target"})
        self.universe = uni.merge(t, on="sid", how="inner")

    def shallow(self, county: str) -> pd.DataFrame:
        k = self.km
        return k[(k.county == county) & k.layer.isin(list(C.LAYERS_SHALLOW))
                 & (k.layer_flag == 0) & (k.sectors3 >= C.SECTORS_MIN)].copy()

    def table(self, wells: pd.DataFrame, feat: pd.DataFrame, target: str = C.PRIMARY_TARGET
              ) -> pd.DataFrame:
        """Well table: feature (well-level), target y, placebo target, layer, xc, yc, town."""
        t = S.well_targets(self.L, target)
        W = wells.merge(feat, on="sid", how="inner").merge(t, on="sid", how="inner")
        if "n_months" in W:
            W = W[W["n_months"] >= C.WELL_MIN_MONTHS]
        uni = self.universe
        if target != C.PRIMARY_TARGET:
            tt = S.well_targets(self.L, target).rename(columns={"y": "y_target"})
            uni = uni.drop(columns=["y_target", "E_mean"]).merge(tt, on="sid", how="inner")
        W = W.reset_index(drop=True)
        W["y_placebo"] = S.placebo_targets(W, uni)
        return W

    def feat(self, name: str) -> pd.DataFrame:
        return S.well_features(self.F, name)


def score(test_id: str) -> dict:
    r = prereg3.assert_can_score(PREREG_ID, C.WP, test_id=test_id)
    return {"prereg_sha256": r["sha256"], "post_registration_looks": r["post_registration_looks"],
            "unverifiable_prior_looks": len(r["unverifiable_prior_looks"])}


# ----------------------------------------------------------------------------- tests
def t1a(D: Data) -> dict:
    out = {"scored": score("P5.t1a_repro")}
    pilot = pd.read_parquet(C.PILOT_DIR / "out" / "well_month_features.parquet",
                            columns=["sid", "month", "dr_med", "A_S1c_strain"])
    pilot["sid"] = pilot["sid"].astype(str)
    res = {}
    for ep in (C.EPOCH_HOURLY, C.EPOCH_10MIN):
        a = S.epoch_medians(pilot, ["dr_med", "A_S1c_strain"], ep)
        b = S.epoch_medians(D.F, ["dr_med_raw", "A_S1c_strain_raw", "dr_med", "A_S1c_strain"],
                            ep)
        j = a.join(b, how="inner", lsuffix="_pilot")
        for c_p, c_new, gated in (("dr_med_pilot", "dr_med_raw", True),
                                  ("A_S1c_strain_pilot", "A_S1c_strain_raw", True),
                                  ("dr_med_pilot", "dr_med", False),
                                  ("A_S1c_strain_pilot", "A_S1c_strain", False)):
            jj = j[[c_p, c_new]].dropna()
            rho = float(S.stats.spearmanr(jj[c_p], jj[c_new])[0])
            res[f"{ep[0]}-{ep[1]}:{c_new}"] = {
                "n_wells": int(len(jj)), "spearman": rho,
                "median_ratio_new_over_pilot": float((jj[c_new] / jj[c_p]).median()),
                "gated": gated, "pass": bool(rho >= C.REPRO_SPEARMAN_MIN) if gated else None}
    out["results"] = res
    out["passed"] = all(v["pass"] for v in res.values() if v["gated"])
    return out


def t1b(D: Data) -> dict:
    out = {"scored": score("P5.t1b_epoch")}
    res = {}
    for cols, gated in ((["dr_med", "A_S1c_strain"], True),
                        (["dr_med_raw", "A_S1c_strain_raw"], False)):
        e1 = S.epoch_medians(D.F, cols, C.EPOCH_HOURLY)
        e2 = S.epoch_medians(D.F, cols, C.EPOCH_10MIN)
        j = e1.join(e2, lsuffix="_h", rsuffix="_10")
        for c in cols:
            jj = j[[c + "_h", c + "_10"]].dropna()
            rho = float(S.stats.spearmanr(jj[c + "_h"], jj[c + "_10"])[0])
            res[c] = {"n_wells": int(len(jj)), "spearman": rho,
                      "median_ratio_10min_over_hourly": float((jj[c + "_10"]
                                                               / jj[c + "_h"]).median()),
                      "gated": gated,
                      "pass": bool(rho >= C.REPRO_SPEARMAN_MIN) if gated else None}
    out["results"] = res
    out["passed"] = all(v["pass"] for v in res.values() if v["gated"])
    return out


def t2(D: Data) -> dict:
    out = {"scored": score("P5.t2_baro")}
    be = I.be_table()
    acc = set(be.loc[be["accepted"].astype(bool), "sid"].astype(str))
    G = D.F[D.F["sid"].isin(acc)]
    co = S.o1_coherence(G)
    d = co["coh_corr"] - co["coh_raw"]
    nz = d[d != 0]
    p = float(S.stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue) if len(nz) else np.nan
    allw = S.o1_coherence(D.F)
    out["results"] = {
        "n_wells": int(len(co)), "median_coh_raw": float(co["coh_raw"].median()),
        "median_coh_corr": float(co["coh_corr"].median()),
        "median_paired_diff": float(d.median()), "n_better": int((d > 0).sum()),
        "n_worse": int((d < 0).sum()), "sign_test_p": p,
        "all_wells_median_coh_raw": float(allw["coh_raw"].median()),
        "all_wells_median_coh_corr": float(allw["coh_corr"].median()),
        "all_wells_n": int(len(allw)),
        "loading_S1_amp_median_m": float(G["A_S1_baro"].median()),
        "loading_S1_amp_p90_m": float(G["A_S1_baro"].quantile(0.9)),
        "S1c_rel_change_median": float(((G["A_S1c_strain"] - G["A_S1c_strain_raw"])
                                        / G["A_S1c_strain_raw"]).median()),
        "inherited_h1a_c6": "H1a v3 validation: 0 of 307 flagged objects on a >= 20 hPa fall "
                            "(P5 adds no flags; its sample set equals L1 v3's)"}
    out["passed"] = bool(d.median() > 0)
    return out


def claims_block(D: Data, wells: pd.DataFrame, feats: dict, seed0: int,
                 target: str = C.PRIMARY_TARGET, half: bool = False) -> dict:
    res = {}
    for i, (name, ftab) in enumerate(feats.items()):
        W = D.table(wells, ftab, target)
        r = S.cv_claim(W, name, seed=seed0 + i)
        if half:
            r["masked_half"] = S.masked_half(W, name, seed=seed0 + 100 + i)
        res[name] = r
    return res


def t3(D: Data) -> dict:
    out = {"scored": score("P5.t3_yunlin_claims")}
    feats = {f: D.feat(f) for f in DAILY_FEATURES}
    out["results"] = claims_block(D, D.shallow("Yunlin"), feats, C.SEED + 300, half=True)
    out["primary"] = {"feature": C.PRIMARY_FEATURE,
                      "claim_supported": out["results"][C.PRIMARY_FEATURE]["claim_supported"]}
    return out


def t4(D: Data) -> dict:
    out = {"scored": score("P5.t4_changhua")}
    feats = {f: D.feat(f) for f in DAILY_FEATURES}
    yl, ch = D.shallow("Yunlin"), D.shallow("Changhua")
    out["claims_changhua"] = claims_block(D, ch, feats, C.SEED + 400)
    tr, rk, sens = {}, {}, {}
    for i, f in enumerate(DAILY_FEATURES):
        Wy, Wc = D.table(yl, feats[f]), D.table(ch, feats[f])
        tr[f] = S.transfer(Wy, Wc, f, seed=C.SEED + 410 + i)
        rk[f] = {"changhua": S.rank_check(Wc, f, seed=C.SEED + 420 + i),
                 "yunlin": S.rank_check(Wy, f, seed=C.SEED + 430 + i)}
        Wcn = D.table(ch, feats[f], "E_r2_nodom")
        Wyn = D.table(yl, feats[f], "E_r2_nodom")
        sens[f] = {"rank_changhua_nodomestic": S.rank_check(Wcn, f, seed=C.SEED + 440 + i),
                   "transfer_nodomestic": S.transfer(Wyn, Wcn, f, seed=C.SEED + 450 + i)}
    out["transfer"] = tr
    out["rank_check"] = rk
    out["sensitivity_no_domestic"] = sens
    # magnitude: the pilot's implied Changhua/Yunlin ratio of local kernel kWh (0.81) vs census
    kk = S.well_targets(D.L, "E_K0_1000")
    m = {}
    for cty, w in (("Yunlin", yl), ("Changhua", ch)):
        ww = w.merge(feats[C.PRIMARY_FEATURE], on="sid").merge(kk, on="sid")
        ww = ww[ww["n_months"] >= C.WELL_MIN_MONTHS]
        m[cty] = float(ww["E_mean"].median())
    out["magnitude"] = {"pilot_implied_ratio_EK0_1000": 0.81,
                        "census_ratio_median_EK0_1000_changhua_over_yunlin":
                            m["Changhua"] / m["Yunlin"],
                        "median_EK0_1000_kwh_per_month": m}
    out["rank_signal_present_changhua_primary"] = bool(
        rk[C.PRIMARY_FEATURE]["changhua"]["spearman"] > 0
        and rk[C.PRIMARY_FEATURE]["changhua"]["perm_p_one_sided"] <= 0.05)
    return out


def t5(D: Data) -> dict:
    out = {"scored": score("P5.t5_within")}
    months = sorted(D.F["month"].unique())
    Ek = D.L.pivot(index="month", columns="sid", values="E_K0_1000").reindex(months)
    reg = D.reg["fan_all"].reindex(months)
    res = {}
    for cty in ("Yunlin", "Changhua"):
        wl = D.km[(D.km.county == cty) & (D.km.sectors3 >= C.SECTORS_MIN)
                  & (D.km.layer_flag == 0) & D.km.layer.isin(["1", "2", "3", "4"])]
        # placebo partners come from the whole census-covered universe (both counties)
        wl_all = D.km[D.km.county.isin(["Yunlin", "Changhua"])
                      & (D.km.sectors3 >= C.SECTORS_MIN)]
        for i, idx in enumerate(("mean", "DN", "A_S1c_strain", "dr_med")):
            Fp = D.F.pivot_table(index="month", columns="sid", values=idx).reindex(months)
            Fp = Fp[[c for c in Fp.columns if c in set(wl["sid"])]]
            r = S.within_well(Fp, Ek, reg, wl_all, idx, seed=C.SEED + 500 + i)
            res[f"{cty}:{idx}"] = r
    out["results"] = res
    return out


def t6(D: Data) -> dict:
    out = {"scored": score("P5.t6_seasonal_cv")}
    wells = pd.concat([D.shallow("Yunlin"), D.shallow("Changhua")])
    res = {}
    for i, idx in enumerate(("A_S1c_strain", "DN", "dr_med")):
        rows = []
        for sid, g in D.F[D.F["sid"].isin(set(wells["sid"]))].groupby("sid"):
            s = g.set_index("month")[idx].dropna()
            if len(s) < C.WELL_MIN_MONTHS:
                continue
            rows.append({"sid": sid, "cv": S.climatology_cv(s)})
        cv = pd.DataFrame(rows).merge(wells[["sid", "county", "layer"]], on="sid")
        res[idx] = S.county_cv_ratio(cv, seed=C.SEED + 600 + i)
        res[idx]["by_layer"] = {f"{c}:{lay}": float(g["cv"].median())
                                for (c, lay), g in cv.groupby(["county", "layer"])}
    reg = D.reg
    months = list(reg.index)
    census = {}
    for grp in ("irr", "all", "nodom"):
        cy = S.series_cv(reg[f"yunlin_{grp}"].to_numpy(), months)
        cc = S.series_cv(reg[f"changhua_{grp}"].to_numpy(), months)
        census[grp] = {"cv_yunlin": cy, "cv_changhua": cc, "alpha": cc / cy}
    # like-for-like: the same CV statistic on local kWh (E_r2) around the same wells
    rows = []
    for sid, g in D.L[D.L["sid"].isin(set(wells["sid"]))].groupby("sid"):
        rows.append({"sid": sid, "cv": S.climatology_cv(g.set_index("month")["E_r2"])})
    cvl = pd.DataFrame(rows).merge(wells[["sid", "county"]], on="sid")
    census["local_E_r2_same_wells"] = S.county_cv_ratio(cvl, seed=C.SEED + 650)
    out["head_proxy"] = res
    out["census"] = census
    a = res["A_S1c_strain"]
    truth = census["irr"]["alpha"]
    out["primary"] = {"proxy_ratio": a["ratio"], "proxy_boot90": a["boot90"],
                      "census_alpha_irrigation": truth,
                      "consistent": bool(a["boot90"][0] <= truth <= a["boot90"][1]),
                      "proxy_meets_p43_cut_0p75": bool(a["ratio"] >= C.P4_SEASONALITY_CUT),
                      "census_meets_p43_cut_0p75": bool(truth >= C.P4_SEASONALITY_CUT)}
    # timing (reported): county-median normalised climatology of the index vs census
    tim = {}
    for cty, w in (("Yunlin", D.shallow("Yunlin")), ("Changhua", D.shallow("Changhua"))):
        G = D.F[D.F["sid"].isin(set(w["sid"]))].copy()
        G["mon"] = G["month"].str[5:7].astype(int)
        for idx in ("A_S1c_strain", "DN"):
            G["n"] = G[idx] / G.groupby("sid")[idx].transform("median")
            c = G.groupby("mon")["n"].median().reindex(range(1, 13))
            mo = np.array([int(m[5:7]) for m in months])
            k = pd.Series(reg[f"{cty.lower()}_irr"].to_numpy()).groupby(mo).mean()
            k = (k / k.mean()).reindex(range(1, 13))
            loc = D.L[D.L["sid"].isin(set(w["sid"]))].copy()
            loc["mon"] = loc["month"].str[5:7].astype(int)
            kl = loc.groupby("mon")["E_r2"].mean()
            kl = (kl / kl.mean()).reindex(range(1, 13))
            tim[f"{cty}:{idx}"] = {"index_clim": c.round(3).tolist(),
                                   "census_irr_clim": k.round(3).tolist(),
                                   "census_local_E_r2_clim": kl.round(3).tolist(),
                                   "r_index_vs_census_irr": float(np.corrcoef(c, k)[0, 1]),
                                   "r_index_vs_census_local": float(np.corrcoef(c, kl)[0, 1]),
                                   "index_peak_months": [int(x) for x in
                                                         c.sort_values().index[-2:][::-1]],
                                   "census_local_peak_months": [int(x) for x in
                                                                kl.sort_values().index[-2:][::-1]]}
    out["timing"] = tim
    return out


def t7(D: Data) -> dict:
    out = {"scored": score("P5.t7_tf_claims")}
    T = pd.read_csv(D.work / "tf_gains.csv", dtype={"sid": str})
    ft = T[["sid", "tf_pump_gain"]].rename(columns={"tf_pump_gain": "feature"})
    ft["n_months"] = T["n_months"]
    yl, ch = D.shallow("Yunlin"), D.shallow("Changhua")
    out["yunlin"] = claims_block(D, yl, {"tf_pump_gain": ft}, C.SEED + 700)
    out["changhua"] = claims_block(D, ch, {"tf_pump_gain": ft}, C.SEED + 710)
    out["transfer"] = S.transfer(D.table(yl, ft), D.table(ch, ft), "tf_pump_gain",
                                 seed=C.SEED + 720)
    out["rank_check"] = {"yunlin": S.rank_check(D.table(yl, ft), "tf_pump_gain", C.SEED + 730),
                         "changhua": S.rank_check(D.table(ch, ft), "tf_pump_gain",
                                                  C.SEED + 731)}
    out["fit_summary"] = {"n_wells": int(len(T)), "median_evp": float(T["evp"].median()),
                          "share_gain_positive": float((T["tf_pump_gain"] > 0).mean()),
                          "median_rel_se": float((T["se_tf_pump_gain"]
                                                  / T["tf_pump_gain"].abs()).median())}
    out["claim_supported_yunlin"] = out["yunlin"]["tf_pump_gain"].get("claim_supported", False)
    return out


def t8(D: Data) -> dict:
    out = {"scored": score("P5.t8_eemd_claims")}
    E = pd.read_csv(D.work / "eemd_features.csv", dtype={"sid": str})
    feats = {}
    for f in ("eemd_annual_amp_m", "eof_pump_loading"):
        ft = E[["sid", f]].rename(columns={f: "feature"})
        ft["n_months"] = E["n_obs_months"].fillna(C.WELL_MIN_MONTHS)
        feats[f] = ft
    yl, ch = D.shallow("Yunlin"), D.shallow("Changhua")
    out["yunlin"] = claims_block(D, yl, feats, C.SEED + 800)
    out["changhua"] = claims_block(D, ch, feats, C.SEED + 810)
    out["claim_supported_yunlin"] = {f: out["yunlin"][f].get("claim_supported", False)
                                     for f in feats}
    return out


RUN = {"P5.t1a_repro": t1a, "P5.t1b_epoch": t1b, "P5.t2_baro": t2, "P5.t3_yunlin_claims": t3,
       "P5.t4_changhua": t4, "P5.t5_within": t5, "P5.t6_seasonal_cv": t6,
       "P5.t7_tf_claims": t7, "P5.t8_eemd_claims": t8}


def main():
    work, ev = Path(sys.argv[1]), Path(sys.argv[2])
    ids = list(TESTS) if sys.argv[3:] == ["all"] else sys.argv[3:]
    (ev / "tests").mkdir(parents=True, exist_ok=True)
    D = Data(work)
    for tid in ids:
        res = RUN[tid](D)
        res["test_id"] = tid
        (ev / "tests" / f"{tid}.json").write_text(jdump(res))
        print(tid, "done", flush=True)


if __name__ == "__main__":
    main()
