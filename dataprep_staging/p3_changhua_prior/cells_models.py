"""WP P3 Yunlin-side modelling (no Changhua census value is read here).

Subcommands (each writes into results/dataprep/reports/P3_evidence/):
  repro     landuse_crop_priors models.py spatial CV on the rebuilt frame (design-time
            reproduction against the pilot's cv_results.csv).
  battery   the eight transfer splits of plan P3 (+ LOTO) for the a2 / a4 / a5 models and the
            crop-dasymetric maps, scored on Yunlin E_tot; the plan's gate; the primary-map rule.
  predict   every map fitted on all Yunlin training cells and applied to every active cell;
            Changhua parts (x MOI Changhua share), class arrays where the model has classes;
            the crop-dasymetric maps for Changhua; NNLS township-bootstrap totals; shift
            diagnostics. Writes work/predictions.npz (frozen before the registered scoring).

Usage: python -I cells_models.py {repro|battery|predict}
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crop_model as CM  # noqa: E402
import frame  # noqa: E402
import lib  # noqa: E402
import models_port as M  # noqa: E402

RNG_SEED = 0
GATE = {"aa_mean_delta_min": 0.03, "aa_worst_delta_min": -0.02, "corr_min": 0.5,
        "corr_splits_min": 6}


# ----------------------------------------------------------------------------- frames
def frames():
    df, aux = frame.load("prior")
    df["coastband"] = np.digitize(df["d_coast_km"], [5.0, 15.0])
    df = M.add_crop(df, aux["tz"])
    tr_lu = df[frame.training_cells(df)].copy()                    # landuse pilot (1100)
    tr = tr_lu[tr_lu[M.CROPF].notna().all(1)].copy()               # a2/a4/a5 (crop features)
    chm = df["moi_chg"].to_numpy() > 0.5
    med = df.loc[chm, M.CROPF].median()
    dfp = df.copy()
    dfp[M.CROPF] = dfp[M.CROPF].fillna(med)                        # a2's Changhua fill
    return df, dfp, tr_lu, tr, aux


# ----------------------------------------------------------------------------- repro
def repro():
    from sklearn.model_selection import GroupKFold
    df, dfp, tr_lu, tr, aux = frames()
    tr = tr_lu.reset_index()
    rows = []
    for target in ["E_tot", "E_irrigation", "E_aquaculture", "E_livestock", "E_other",
                   "E_crop1", "E_crop2", "E_off", "E1416_tot"]:
        y = tr[target].to_numpy(float)
        for scheme in ("town", "block5"):
            g = tr[scheme].to_numpy()
            if scheme == "town":
                splits = [(np.where(g != u)[0], np.where(g == u)[0]) for u in np.unique(g)]
            else:
                splits = list(GroupKFold(n_splits=5).split(tr, y, g))
            for name, mk in M.lu_cv_models().items():
                p = np.zeros_like(y)
                for a, b in splits:
                    p[b] = mk().fit(tr.iloc[a], y[a]).predict(tr.iloc[b])
                p = np.maximum(p, 0)
                t_, b_ = tr["town"].to_numpy(), tr["block5"].to_numpy()
                yt, pt = pd.Series(y).groupby(t_).sum(), pd.Series(p).groupby(t_).sum()
                yb, pb = pd.Series(y).groupby(b_).sum(), pd.Series(p).groupby(b_).sum()
                rows.append({"target": target, "scheme": scheme, "model": name,
                             "R2_cell": M.r2(y, p), "R2_township_sum": M.r2(yt.values, pt.values),
                             "R2_5km_block_sum": M.r2(yb.values, pb.values),
                             "total_ratio": float(p.sum() / y.sum())})
        print(target, "done", flush=True)
    res = pd.DataFrame(rows)
    pil = pd.read_csv(lib.PILOT_LU / "cv_results.csv")
    m = res.merge(pil, on=["target", "scheme", "model"], suffixes=("", "_pilot"))
    for c in ("R2_cell", "R2_township_sum", "R2_5km_block_sum", "total_ratio"):
        m[f"d_{c}"] = m[c] - m[f"{c}_pilot"]
    m.to_csv(lib.EVID / "repro_landuse_cv.csv", index=False)
    summ = {c: float(m[f"d_{c}"].abs().max()) for c in
            ("R2_cell", "R2_township_sum", "R2_5km_block_sum", "total_ratio")}
    lib.save_json(lib.EVID / "repro_landuse_cv.json", {"n_rows": int(len(m)),
                                                       "max_abs_diff_vs_pilot": summ})
    print(json.dumps(summ))


# ----------------------------------------------------------------------------- crop-dasymetric
class CropDas:
    """Crop-dasymetric total map (plan P3 steps 2-4), trained on a set of Yunlin cells.

    irrigation: crop-statistics township model (per-year NNLS shrunk to the pooled fit, lam by
    LOTO on the training towns) applied to a target township's crop areas, times the
    farmland fraction of the township inside the target cells, spread over the target cells
    in proportion to the LUIMAP NNLS irrigation shape x the township's share of the cell
    (``variant='plan'``) or to farmland x share (``variant='croponly'``, plan step 8);
    aquaculture and livestock: per-hectare rates = the own-class coefficient of the NNLS
    class models (pond, livestock buildings; plan step 4); other: NNLS other-class model.
    Training townships: every Yunlin township, with crop areas scaled by its farmland
    fraction outside the target cells and energy of its pumps outside them.
    """

    def __init__(self, ctx, variant="plan"):
        self.ctx, self.variant = ctx, variant

    def fit_predict(self, dtr, dte_idx, dte, test_mask_cells, train_whole: bool = False):
        """``train_whole``: the target cells hold no training township (Changhua): every
        Yunlin township trains with its whole crop areas and all its pumps."""
        c = self.ctx
        S = c["S"]                             # (nT, A) MOI township share of cell
        farm_c = c["farm_ha"]                  # (A,) LUIMAP farmland ha per cell (whole cell)
        # farmland of each township inside the target cells / whole township
        in_t = np.zeros(S.shape[1], bool)
        in_t[test_mask_cells] = True
        nnls_irr = M.NNLSArea(M.FEATS).fit(dtr, dtr["E_irrigation"].to_numpy(float))
        nnls_aq = M.NNLSArea(M.FEATS).fit(dtr, dtr["E_aquaculture"].to_numpy(float))
        nnls_ls = M.NNLSArea(M.FEATS).fit(dtr, dtr["E_livestock"].to_numpy(float))
        nnls_ot = M.NNLSArea(M.FEATS).fit(dtr, dtr["E_other"].to_numpy(float))
        b_aq = nnls_aq.beta[M.FEATS.index("aqua")]
        b_ls = nnls_ls.beta[M.FEATS.index("livestock")]
        farm_t_in = (S * farm_c[None, :] * in_t[None, :]).sum(1)          # ha (approx.)
        farm_t_whole = c["farm_whole_ha"]
        frac_in = np.where(farm_t_whole > 0, farm_t_in / np.maximum(farm_t_whole, 1e-9), 0.0)
        frac_in = np.clip(frac_in, 0.0, 1.0)
        # crop model on Yunlin townships, energy and crop areas outside the target cells
        panel = {}
        for y in CM.YEARS:
            X = CM.design(c["crop"], y, "雲林縣")
            towns, rowsX, rowsY = [], [], []
            for tn, x in zip(X.index, X.to_numpy(float), strict=True):
                if tn not in c["tidx"]:
                    continue
                j = c["tidx"][tn]
                keep = 1.0 if train_whole else 1.0 - frac_in[j]
                if keep < 0.2:                 # township mostly inside the target cells
                    continue
                e = c["pump_e"][y]
                m = c["pump_town"] == tn
                if not train_whole:
                    m = m & ~c["pump_in_target"](in_t)
                towns.append(tn)
                rowsX.append(x * keep)
                rowsY.append(float(e[m].sum()))
            panel[y] = (towns, np.array(rowsX), np.array(rowsY))
        lam, _ = CM.select_lambda(panel)
        betas, _ = CM.fit_all(None, None, lam, panel)
        # predictions for the townships touching the target cells
        A = S.shape[1]
        irr = np.zeros(A)
        tgt = np.flatnonzero(in_t)
        shape = np.maximum(nnls_irr.predict(c["df_all"].iloc[tgt]), 0)
        for tn, j in c["tidx_target"].items():
            if frac_in[j] <= 0:
                continue
            pred_y = []
            for y in CM.YEARS:
                X = CM.design(c["crop"], y, c["county_zh"])
                if tn not in X.index:
                    continue
                pred_y.append(float(X.loc[tn].to_numpy(float) @ betas[y]))
            if not pred_y:
                continue
            total = np.mean(pred_y) * frac_in[j]
            w = farm_c[tgt] * S[j, tgt] if self.variant == "croponly" else shape * S[j, tgt]
            if w.sum() <= 0:
                w = S[j, tgt].copy()
            if w.sum() <= 0:
                continue
            irr[tgt] += total * w / w.sum()
        d_t = c["df_all"].iloc[tgt]
        aq = b_aq * d_t["aqua"].to_numpy()
        ls = b_ls * d_t["livestock"].to_numpy()
        ot = np.maximum(nnls_ot.predict(d_t), 0)
        self.last = {"lambda": lam, "b_aq": float(b_aq), "b_ls": float(b_ls),
                     "n_train_towns": {y: len(panel[y][0]) for y in CM.YEARS}}
        out = {"E_irrigation": irr[tgt], "E_aquaculture": aq, "E_livestock": ls,
               "E_other": ot}
        pos = {cc: k for k, cc in enumerate(tgt)}
        sel = np.array([pos[i] for i in dte_idx])
        return {k: v[sel] for k, v in out.items()}


def crop_context(df, aux, county_zh="雲林縣"):
    tz = aux["tz"]
    names = [str(n) for n in tz["names"]]
    groups = [str(g) for g in tz["groups"]]
    S = tz["share"]
    farm_c = df[["paddy", "dry", "orchard"]].sum(1).to_numpy()
    whole = tz["whole_class_area_m2"]
    farm_whole = np.nansum(whole[:, [groups.index(g) for g in ("paddy", "dry", "orchard")]],
                           1) / 1e4
    # Yunlin pumps: town, cell, annual irrigation kWh per year (all kept pumps)
    p = frame.open_c0(lib.C0_PRE, "prior")
    z = np.load(p.path("pumps_yunlin.npz"), allow_pickle=True)
    L = frame.census_attrs("yunlin", "prior", ["uid", "town"]).set_index("uid")
    ptown = np.array([lib.norm_town("雲林縣", t) for t in L.loc[z["uid"], "town"]],
                     dtype=object)
    irr = np.asarray(z["purpose_class"]).astype(str) == "irrigation"
    dates = np.asarray(z["dates"]).astype(str)
    yrs = np.array([int(d[:4]) for d in dates])
    v = np.nan_to_num(z["value"].astype(float))
    pump_e = {y: v[:, yrs == y].sum(1) * irr for y in CM.YEARS}
    pcell = np.asarray(z["cell"])

    def in_target(in_t):
        return np.where(pcell >= 0, in_t[np.clip(pcell, 0, None)], False)

    tidx = {n: i for i, n in enumerate(names)}
    return {"S": S, "farm_ha": farm_c, "farm_whole_ha": farm_whole, "crop": lib.crop_table(),
            "tidx": {n: i for n, i in tidx.items() if n.startswith("雲林縣")},
            "tidx_target": {n: i for n, i in tidx.items() if n.startswith(county_zh)},
            "county_zh": county_zh, "pump_e": pump_e, "pump_town": ptown,
            "pump_in_target": in_target, "df_all": df}


# ----------------------------------------------------------------------------- battery
def battery():
    df, dfp, tr_lu, tr, aux = frames()
    ch = dfp[dfp["moi_chg"] > 0.5].copy()
    p_tr, p_ch, auc = M.propensity(tr, ch)
    tr["prop"] = p_tr
    chl = M.chlike_towns(tr)
    splits = M.transfer_splits(tr, chl)
    y = tr["E_tot"].to_numpy(float)
    town = tr["town"].to_numpy()
    ctx = crop_context(df, aux)
    rows, das_info = [], {}
    total_models = {**M.A2_MODELS, **M.A4_MODELS}
    for sp, te in list(splits.items()) + [("LOTO", None)]:
        parts = ([(np.unique(town)[k], town == np.unique(town)[k])
                  for k in range(len(np.unique(town)))] if te is None else [(sp, te)])
        preds = {}
        for nm, mk in total_models.items():
            p = np.zeros_like(y)
            for _, m_ in parts:
                p[m_] = mk().fit(tr[~m_], y[~m_]).predict(tr[m_])
            preds[nm] = p
        for nm in M.A5_COMPS:
            if nm == "rf_total":
                continue
            p = np.zeros_like(y)
            for _, m_ in parts:
                p[m_] = M.a5_fit_predict(nm, tr[~m_], tr[m_])
            preds[nm] = p
        for var in ("plan", "croponly"):
            p = np.zeros_like(y)
            info = []
            for _, m_ in parts:
                cd = CropDas(ctx, var)
                idx = tr.index.to_numpy()[m_]
                o = cd.fit_predict(tr[~m_], idx, tr[m_], idx)
                p[m_] = sum(o.values())
                info.append(cd.last)
            preds[f"cropdas_{var}"] = p
            das_info[f"{sp}|{var}"] = info if te is None else info[0]
        for nm, p in preds.items():
            mm = te if te is not None else np.ones(len(y), bool)
            rows.append({"split": sp, "model": nm, **M.metrics(tr[mm], y[mm], p[mm])})
        print(sp, "done", flush=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(lib.EVID / "battery_scores.csv", index=False)
    # gate (plan P3 Validation) on the eight transfer splits
    t8 = tab[tab.split != "LOTO"]
    uni = t8[t8.model == "farm_uniform"].set_index("split")["AA_5km"]
    gate = {}
    for nm, g in t8.groupby("model"):
        g = g.set_index("split")
        d = g["AA_5km"] - uni.reindex(g.index)
        nc = int((g["corr_5km_density"] >= GATE["corr_min"]).sum())
        gate[nm] = {"aa_mean_delta": float(d.mean()), "aa_worst_delta": float(d.min()),
                    "corr_ge_0.5_splits": nc, "ratio_min": float(g["ratio"].min()),
                    "ratio_max": float(g["ratio"].max()),
                    "passes": bool(d.mean() >= GATE["aa_mean_delta_min"] and
                                   d.min() >= GATE["aa_worst_delta_min"] and
                                   nc >= GATE["corr_splits_min"])}
    primary = "cropdas_plan" if gate["cropdas_plan"]["passes"] else "rf_LU+GEO"
    # comparison with the pilot's a2/a4/a5 rows
    comp = []
    for f, key in (("a2_scores.csv", "E_tot|"), ("a4_scores.csv", ""), ("a5_scores.csv", "")):
        pil = pd.read_csv(lib.PILOT_ML / f)
        if key:
            pil = pil[pil.split.str.startswith(key)].copy()
            pil["split"] = pil.split.str[len(key):]
        pil = pil[pil.split != "CHANGHUA_GWh_yr"]
        pil["model"] = pil.model.replace({"rf_total": "rf_LU+GEO"})
        mm = tab.merge(pil, on=["split", "model"], suffixes=("", "_pilot"))
        for c in ("ratio", "AA_5km", "corr_5km_density"):
            mm[f"d_{c}"] = mm[c] - mm[f"{c}_pilot"]
        comp.append(mm.assign(pilot_file=f))
    comp = pd.concat(comp, ignore_index=True)
    comp.to_csv(lib.EVID / "battery_vs_pilot.csv", index=False)
    out = {"gate_rule": GATE, "gate": gate, "primary_by_plan_rule": primary,
           "propensity_auc_changhua_vs_yunlin_train": auc, "chlike_towns":
           [aux["towns"][int(t)] for t in chl], "n_train_cells": int(len(tr)),
           "n_changhua_cells_gt05": int(len(ch)), "cropdas_fit_info": das_info,
           "max_abs_diff_vs_pilot": {c: float(comp[f"d_{c}"].abs().max())
                                     for c in ("ratio", "AA_5km", "corr_5km_density")}}
    lib.save_json(lib.EVID / "battery.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "cropdas_fit_info"},
                     ensure_ascii=False, indent=1, default=lib._jdefault))


# ----------------------------------------------------------------------------- predict
def predict():
    df, dfp, tr_lu, tr, aux = frames()
    A = len(df)
    chg = df["moi_chg"].to_numpy()
    maps, cls_maps = {}, {}
    # landuse pilot per-class models (apply.py), fitted on the 1100 training cells
    for name, mk in M.lu_models().items():
        cl = {}
        for t in M.CLS4:
            cl[t] = np.maximum(mk().fit(tr_lu, tr_lu[t].to_numpy(float)).predict(df), 0)
        cls_maps[f"lu_{name}"] = cl
        maps[f"lu_{name}"] = sum(cl.values())
    # a2 / a4 total models (E_tot) and a5 composites, on the a2 training cells
    for nm, mk in {**M.A2_MODELS, **M.A4_MODELS}.items():
        maps[nm] = np.maximum(mk().fit(tr, tr["E_tot"].to_numpy(float)).predict(dfp), 0)
    for nm in M.A5_COMPS:
        if nm == "rf_total":
            continue
        cl = M.a5_fit_predict(nm, tr, dfp, cls=True)
        cls_maps[nm] = cl
        maps[nm] = sum(cl.values())
    # crop-dasymetric for Changhua (training: all Yunlin townships, whole)
    ctx = crop_context(df, aux, county_zh="彰化縣")
    target_cells = np.flatnonzero(chg > 0)
    das_info = {}
    for var in ("plan", "croponly"):
        cd = CropDas(ctx, var)
        # Changhua target: every Yunlin township trains whole (crop areas and all pumps)
        o = cd.fit_predict(tr_lu, target_cells, df.iloc[target_cells], target_cells,
                           train_whole=True)
        full = {}
        for k, v in o.items():
            a = np.zeros(A)
            a[target_cells] = v
            full[k] = a
        # aq/ls/other are whole-cell predictions; their Changhua part is taken by x moi_chg
        # below like every other map; irrigation is already Changhua-only (township-based)
        full["E_irrigation"] = np.where(chg > 0, full["E_irrigation"] / np.maximum(chg, 1e-12),
                                        0.0)
        cls_maps[f"cropdas_{var}"] = full
        maps[f"cropdas_{var}"] = sum(full.values())
        das_info[var] = cd.last
    # pilot-exact crop-dasymetric reproduction (combine.py: 2014-16 mean / 2021, land-area
    # fraction, NNLS irr shape, NNLS aq/ls/other), from the pilot-variant township predictions
    with open(lib.EVID / "crop_model.json", encoding="utf-8") as fh:
        cmj = json.load(fh)
    tz = aux["tz"]
    names = [str(n) for n in tz["names"]]
    S = tz["share"]
    infan = S.sum(1) * 1e6
    frac_area = np.minimum(infan / tz["town_area_total_m2"], 1.0)
    need = 1.0 - np.clip(df["moi_yun"].to_numpy(), 0, 1)
    irr_shape = cls_maps["lu_nnls_area"]["E_irrigation"] * need
    other = sum(cls_maps["lu_nnls_area"][c] * chg for c in
                ("E_aquaculture", "E_livestock", "E_other"))
    pil_das = {}
    for label, years in (("2014-16", (2014, 2015, 2016)), ("2021", (2021,))):
        town_pred = {}
        for y in years:
            for tn, v in cmj["pilot_variant"][str(y)]["changhua_whole_pred"].items():
                town_pred.setdefault(tn, []).append(v)
        das = np.zeros(A)
        for tn, vals in town_pred.items():
            if tn not in names:
                continue
            j = names.index(tn)
            if S[j].sum() <= 0:
                continue
            target = float(np.mean(vals)) * frac_area[j]
            w = np.where(chg > 0, irr_shape * S[j] / np.maximum(chg, 1e-9), 0.0)
            if w.sum() <= 0:
                w = S[j].copy()
            das += target * w / w.sum()
        # stored as whole-cell equivalents so that x moi_chg returns das + other
        tot_ch = das + other
        maps[f"pilotdas_{label}"] = np.where(chg > 0, tot_ch / np.maximum(chg, 1e-12), 0.0)
        pil_das[label] = {"irrigation_GWh": float(das.sum() / 1e6),
                          "aq_ls_other_GWh": float(other.sum() / 1e6),
                          "total_GWh": float(tot_ch.sum() / 1e6)}
    # NNLS township-block bootstrap (fast_nnls.py / apply.py): Changhua totals
    rng = np.random.default_rng(RNG_SEED)
    towns = tr_lu["town"].unique()
    towns_arr = tr_lu["town"].to_numpy()
    Xd = M.NNLSArea(M.FEATS).design(df)
    boot = {}
    for target in ["E_tot", "E_irrigation", "E_aquaculture", "E_livestock", "E_other"]:
        B = []
        for _ in range(500):
            pick = rng.choice(towns, size=len(towns), replace=True)
            idx = np.concatenate([np.where(towns_arr == g)[0] for g in pick])
            d = tr_lu.iloc[idx]
            B.append(M.NNLSArea(M.FEATS).fit(d, d[target].to_numpy(float)).beta)
        B = np.array(B)
        tot = (Xd @ B.T * chg[:, None]).sum(0) / 1e6
        boot[target] = {"p05": float(np.percentile(tot, 5)), "p50": float(np.median(tot)),
                        "p95": float(np.percentile(tot, 95))}
    # shift diagnostics (plan P3 step 6), Yunlin vs Changhua features only
    ch = dfp[dfp["moi_chg"] > 0.5].copy()
    p_tr, p_ch, auc = M.propensity(tr, ch)
    w = p_tr / np.maximum(1 - p_tr, 1e-6)
    kish = float(w.sum() ** 2 / (w ** 2).sum() / len(w))
    farm = df[M.FARM[:3]].sum(1)
    cpf = np.where(farm > 0, df["canal3_ha"] / np.maximum(farm, 1e-9), np.nan)
    lo, hi = np.nanpercentile(cpf[tr_lu.index], [1, 99])
    cpf_ch = cpf[ch.index]
    shift = {"adversarial_auc": auc, "kish_ess_fraction": kish,
             "canal3_per_farm_outside_yunlin_1_99": float(np.mean((cpf_ch < lo) | (cpf_ch > hi))),
             "dry_share_yunlin_train": float(tr_lu["dry"].sum() / farm[tr_lu.index].sum()),
             "dry_share_changhua": float(ch["dry"].sum() / farm[ch.index].sum()),
             "prop_gt_0.9_changhua_frac": float(np.mean(p_ch > 0.9))}
    np.savez_compressed(
        lib.WORK / "predictions.npz", map_names=np.array(list(maps)),
        maps=np.stack([maps[k] for k in maps]), moi_chg=chg,
        cls_names=np.array(list(cls_maps)),
        cls=np.stack([np.stack([cls_maps[k][c] for c in M.CLS4]) for k in cls_maps]),
        classes=np.array(M.CLS4), prop_changhua=p_ch, changhua_gt05_cells=ch.index.to_numpy())
    out = {"maps": list(maps), "class_maps": list(cls_maps),
           "changhua_part_GWh": {k: float((v * chg).sum() / 1e6) for k, v in maps.items()},
           "changhua_part_gt05_cells_GWh": {k: float((v * chg)[chg > 0.5].sum() / 1e6)
                                            for k, v in maps.items()},
           "cropdas_info": das_info, "pilot_das_repro": pil_das, "nnls_bootstrap": boot,
           "shift": shift}
    lib.save_json(lib.EVID / "predictions.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=lib._jdefault)[:6000])


if __name__ == "__main__":
    {"repro": repro, "battery": battery, "predict": predict}[sys.argv[1]]()
