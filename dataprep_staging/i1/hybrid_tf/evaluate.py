"""Task 2: design-time skill of the hybrid TF layer on 2012-2022 (spent units, U7).

Two time-blocked splits of R0's 158 scored wells (74 Changhua, 84 Yunlin), heads 2012-02..2022-12:

* ``loyo``: leave one calendar year out (11 folds, as ``I1_changhua_drivers``); the fit uses
  every other year, so it is not a forecast; it measures how well the drivers explain a year;
* ``fwd``:  forward chaining, the forecast mimic: fit on heads up to September of year Y (the
  origin), predict November Y to October Y+1, for Y = 2015..2021 (7 folds). This copies the
  draft test: heads fetched to 2026-09, window 2026-11..2027-10.

In both, the origin of the persistence-type forecasts is the last observation at most
``MAX_ORIGIN_GAP`` months before ``ORIGIN_LAG`` = 2 months ahead of the block start, and the
``*_fc`` variants cut each driver ``AVAIL_LAG`` months before the block start (the months not
yet recorded when the draft test freezes its predictions; see ``config``).

Models (seasonal = no head after the origin; onestep = the previous month's head is known):

* baselines: ``clim`` (month-of-year mean of the fit months), ``persist`` (origin head carried
  forward) / ``persist1`` (previous month), ``stat_ar1`` / ``stat_ar1_1`` (the existing
  statistical layer: climatology plus the origin anomaly decayed by its lag-1 coefficient),
  ``phys`` (R0's out-of-fold head plus an offset fitted on the fit months), ``phys_ar`` /
  ``phys_ar1`` (R0 plus its decayed origin residual);
* TF models: ``tf`` (pure), ``phys_tf`` (R0 + TF correction), each with ``_ar`` (decayed origin
  residual) or ``_ar1`` (one-month-ahead residual); driver sets: recorded (no suffix), ``_fc``
  (seasonal cut-offs), ``_fc1`` (seasonal cut-offs but rain recharge recorded),
  ``_fc_hold`` (cut-offs with hold-last for every driver; fill-rule sensitivity).

Scores per well on the months where every model of the setting has a prediction, then per
county: median R2 (1 - SSE/SST about the well's mean over the scored months), pooled RMSE,
pooled RMSE ratio against climatology, share of wells beating climatology.

Writes ``results/dataprep/reports/I1_hybrid_tf/{design_skill.json, design_wells.csv,
design_skill_table.md, data_roles.csv}`` and logs one look (after_the_fact).

Run from the repository root:
    dataprep_staging/tools/cpu_slot.sh ~/miniforge3/envs/hydro/bin/python \
        -m i1.hybrid_tf.evaluate      (with PYTHONPATH=dataprep_staging)
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from . import config as C
from .data import load_design
from .model import (
    Ridge,
    cut_series,
    features,
    lag1,
    month_clim,
    month_dummies,
    origin_index,
)

sys.path.insert(0, str(C.REPO / "dataprep_staging"))

SEASONAL = ("clim", "persist", "stat_ar1", "phys", "phys_ar", "tf", "tf_ar", "phys_tf",
            "phys_tf_ar", "tf_fc", "tf_ar_fc", "phys_tf_fc", "phys_tf_ar_fc", "tf_fc_hold",
            "tf_ar_fc_hold", "tf_fc_elec")
ONESTEP = ("clim", "persist1", "stat_ar1_1", "phys_ar1", "tf_ar1", "phys_tf_ar1",
           "tf_fc1_ar1", "phys_tf_fc1_ar1", "tf_fc_ar1", "arx1", "arx_fc1_1", "arx_fc_1",
           "arx_fc1elec_1")
ELEC = frozenset({"electricity_local_3km", "electricity_county"})


def blocks(split: str, hd: pd.DatetimeIndex) -> list[dict]:
    T = len(hd)
    t = np.arange(T)
    out = []
    if split == "loyo":
        for yr in sorted(set(hd.year)):
            test = np.asarray(hd.year == yr)
            start = int(np.flatnonzero(test)[0])
            out.append({"label": str(yr), "train": ~test, "test": test, "start": start,
                        "end": int(np.flatnonzero(test)[-1]),
                        "origin": start - C.ORIGIN_LAG})
    elif split == "fwd":
        for yr in range(2015, 2022):
            start = int(hd.get_loc(pd.Timestamp(f"{yr}-11-01")))
            end = min(start + 11, T - 1)
            origin = start - C.ORIGIN_LAG
            out.append({"label": f"{yr}-11..{yr + 1}-10", "train": t <= origin,
                        "test": (t >= start) & (t <= end), "start": start, "end": end,
                        "origin": origin})
    else:
        raise ValueError(split)
    return out


def variant_series(drv: dict, start_d: int, cal_d: np.ndarray, fit_d: np.ndarray,
                   mode: str) -> dict:
    """Driver series with the forecast-time cut-offs (driver index of the block start =
    ``start_d``). ``mode``: 'fc' (seasonal), 'fc1' (rain recorded), 'fc_hold', 'fc_elec'
    (electricity recorded), 'fc1elec' (rain and electricity recorded)."""
    out = {}
    for name, s in drv.items():
        recorded = ((mode in ("fc1", "fc1elec") and name in C.RECORDED_IN_WINDOW)
                    or (mode in ("fc_elec", "fc1elec") and name in ELEC))
        if recorded:
            out[name] = s
            continue
        hold = mode == "fc_hold" or name in C.HOLD_LAST
        out[name] = cut_series(s, start_d - C.AVAIL_LAG[name], cal_d, fit_d, hold)
    return out


def run_split(d, split: str) -> tuple[dict, dict]:
    hd, dd = d.head_dates, d.drv_dates
    W, T = d.y.shape
    cal_h, cal_d = np.asarray(hd.month - 1), np.asarray(dd.month - 1)
    dum = month_dummies(cal_h)
    preds = {"seasonal": {m: np.full((W, T), np.nan) for m in SEASONAL},
             "onestep": {m: np.full((W, T), np.nan) for m in ONESTEP}}
    tvec = np.arange(T)
    ref = [{k: float(v[i].mean()) for k, v in d.drivers.items()} for i in range(W)]
    Xrec = [np.c_[features({k: v[i] for k, v in d.drivers.items()}, C.DRIVERS, ref[i])[1:],
                  dum] for i in range(W)]
    for b in blocks(split, hd):
        rows = []
        for i in range(W):
            y, p = d.y[i], d.p[i]
            fit = b["train"] & np.isfinite(y)
            if fit.sum() < C.MIN_TRAIN:
                continue
            fit_d = np.r_[True, b["train"]]
            drv = {k: v[i] for k, v in d.drivers.items()}
            Xv = {}
            var_rows = (tvec > b["origin"]) & (tvec <= b["end"])
            for mode in ("fc", "fc1", "fc_hold", "fc_elec", "fc1elec"):
                ser = variant_series(drv, b["start"] + 1, cal_d, fit_d, mode)
                Xm = np.c_[features(ser, C.DRIVERS, ref[i])[1:], dum]
                X = Xrec[i].copy()
                X[var_rows] = Xm[var_rows]
                Xv[mode] = X
            Xv["rec"] = Xrec[i]
            m_tf = Ridge().fit(Xrec[i][fit], y[fit])
            m_pt = Ridge().fit(Xrec[i][fit], (y - p)[fit])
            f = {("tf", k): m_tf.predict(X) for k, X in Xv.items()}
            f.update({("phys_tf", k): p + m_pt.predict(X) for k, X in Xv.items()})
            clim = month_clim(y, fit, cal_h)
            phys = p + float(np.mean((y - p)[fit]))
            o = origin_index(y, b["origin"])
            rho = {"clim": lag1(y - clim, fit)[0], "phys": lag1(y - phys, fit)[0],
                   "tf": lag1(y - f[("tf", "rec")], fit)[0],
                   "phys_tf": lag1(y - f[("phys_tf", "rec")], fit)[0]}
            rows.append((i, y, fit, clim, phys, o, rho, f, Xv))
        # pooled lag-1 coefficients where a well has too few pairs
        pooled = {k: float(np.nanmedian([r[6][k] for r in rows])) for k in rows[0][6]}
        test = b["test"]
        for i, y, fit, clim, phys, o, rho, f, Xv in rows:
            rr = {k: (v if np.isfinite(v) else pooled[k]) for k, v in rho.items()}
            S = preds["seasonal"]
            S["clim"][i, test] = clim[test]
            S["phys"][i, test] = phys[test]
            S["tf"][i, test] = f[("tf", "rec")][test]
            S["phys_tf"][i, test] = f[("phys_tf", "rec")][test]
            S["tf_fc"][i, test] = f[("tf", "fc")][test]
            S["phys_tf_fc"][i, test] = f[("phys_tf", "fc")][test]
            S["tf_fc_hold"][i, test] = f[("tf", "fc_hold")][test]
            S["tf_fc_elec"][i, test] = f[("tf", "fc_elec")][test]
            if o >= 0:
                k = (tvec - o)[test].astype(float)
                S["persist"][i, test] = y[o]
                S["stat_ar1"][i, test] = clim[test] + rr["clim"] ** k * (y[o] - clim[o])
                S["phys_ar"][i, test] = phys[test] + rr["phys"] ** k * (y[o] - phys[o])
                for fam in ("tf", "phys_tf"):
                    r0 = y[o] - f[(fam, "rec")][o]
                    for mode, suf in (("rec", ""), ("fc", "_fc"), ("fc_hold", "_fc_hold")):
                        name = f"{fam}_ar{suf}"
                        if name in S:
                            S[name][i, test] = f[(fam, mode)][test] + rr[fam] ** k * r0
            O1 = preds["onestep"]
            tt = np.flatnonzero(test)
            tt = tt[(tt >= 1)]
            tt = tt[np.isfinite(y[tt - 1])]
            O1["clim"][i, tt] = clim[tt]
            O1["persist1"][i, tt] = y[tt - 1]
            O1["stat_ar1_1"][i, tt] = clim[tt] + rr["clim"] * (y[tt - 1] - clim[tt - 1])
            O1["phys_ar1"][i, tt] = phys[tt] + rr["phys"] * (y[tt - 1] - phys[tt - 1])
            for fam in ("tf", "phys_tf"):
                for mode, suf in (("rec", ""), ("fc1", "_fc1"), ("fc", "_fc")):
                    name = f"{fam}{suf}_ar1"
                    if name in O1:
                        g = f[(fam, mode)]
                        O1[name][i, tt] = g[tt] + rr[fam] * (y[tt - 1] - g[tt - 1])
            # ARX: the existing AR(1) statistical layer plus a TF model of its one-month
            # innovations (nests stat_ar1_1: ridge shrinks the TF term to zero if the drivers
            # add nothing). Added after design run 2.
            a = y - clim
            tr = np.flatnonzero(fit[1:] & fit[:-1]) + 1
            if tr.size >= C.MIN_TRAIN:
                u = a[tr] - rr["clim"] * a[tr - 1]
                m_arx = Ridge().fit(Xv["rec"][tr], u)
                base = clim[tt] + rr["clim"] * a[tt - 1]
                for mode, name in (("rec", "arx1"), ("fc1", "arx_fc1_1"), ("fc", "arx_fc_1"),
                                   ("fc1elec", "arx_fc1elec_1")):
                    O1[name][i, tt] = base + m_arx.predict(Xv[mode][tt])
    return preds, {"blocks": [b["label"] for b in blocks(split, hd)]}


def score(d, preds: dict, split: str, block_list: list[dict]) -> tuple[list, dict]:
    rows, summary = [], {}
    for setting, P in preds.items():
        models = list(P)
        allm = np.stack([P[m] for m in models])
        common = np.isfinite(d.y) & np.isfinite(allm).all(axis=0)
        for i in range(len(d.sids)):
            m = common[i]
            if m.sum() < C.MIN_SCORED:
                continue
            yo = d.y[i, m]
            sst = float(((yo - yo.mean()) ** 2).sum())
            e_clim = float(((P["clim"][i, m] - yo) ** 2).sum())
            for mod in models:
                sse = float(((P[mod][i, m] - yo) ** 2).sum())
                rows.append({"split": split, "setting": setting, "model": mod,
                             "sid": d.sids[i], "county": d.county[i], "n": int(m.sum()),
                             "sse": sse, "sse_clim": e_clim, "rmse": np.sqrt(sse / m.sum()),
                             "r2": 1 - sse / sst if sst > 0 else np.nan})
        df = pd.DataFrame([r for r in rows if r["setting"] == setting])
        out = {}
        for cty in ("changhua", "yunlin", "all"):
            g = df if cty == "all" else df[df["county"] == cty]
            res = {}
            for mod in models:
                gm = g[g["model"] == mod]
                res[mod] = {
                    "n_wells": int(len(gm)), "n_months": int(gm["n"].sum()),
                    "median_r2": float(gm["r2"].median()),
                    "pooled_rmse_m": float(np.sqrt(gm["sse"].sum() / gm["n"].sum())),
                    "rmse_ratio_vs_clim": float(np.sqrt(gm["sse"].sum() / gm["sse_clim"].sum())),
                    "median_well_rmse_ratio_vs_clim": float(
                        np.sqrt(gm["sse"] / gm["sse_clim"]).median()),
                    "share_wells_beat_clim": float((gm["sse"] < gm["sse_clim"]).mean())}
            out[cty] = res
        # per block pooled ratio against climatology (variability across years)
        per_block = {}
        for b in block_list:
            tb = b["test"]
            res = {}
            for cty in ("changhua", "yunlin"):
                sel = (d.county == cty)[:, None] & common & tb[None, :]
                if sel.sum() == 0:
                    continue
                e_c = ((P["clim"] - d.y) ** 2)[sel].sum()
                res[cty] = {mod: float(np.sqrt(((P[mod] - d.y) ** 2)[sel].sum() / e_c))
                            for mod in models}
            per_block[b["label"]] = res
        summary[setting] = {"by_county": out, "per_block_rmse_ratio_vs_clim": per_block}
    return rows, summary


def main() -> None:
    from harness import prereg3
    d = load_design()
    C.OUT.mkdir(parents=True, exist_ok=True)
    allrows, summ = [], {}
    for split in ("loyo", "fwd"):
        preds, meta = run_split(d, split)
        rows, s = score(d, preds, split, blocks(split, d.head_dates))
        allrows += rows
        summ[split] = {**meta, **s}
    wells = pd.DataFrame(allrows)
    wells.drop(columns=["sse", "sse_clim"]).to_csv(C.OUT / "design_wells.csv", index=False)
    rec = {
        "schema": "i1-hybrid-tf-design/1",
        "role": "design-time evidence on spent units (heads 2012-2022 at R0's 158 wells, "
                "already scored by R0..R1e and I1_changhua_drivers); never a gate, threshold "
                "or validation of its own choices (U7, plan 4.5)",
        "wells": {"changhua": int((d.county == "changhua").sum()),
                  "yunlin": int((d.county == "yunlin").sum())},
        "hyperparameters": {"taus_months": list(C.TAUS), "ridge_alpha": C.ALPHA,
                            "local_radius_m": C.LOCAL_RADIUS_M, "min_train": C.MIN_TRAIN,
                            "min_scored": C.MIN_SCORED, "max_origin_gap": C.MAX_ORIGIN_GAP,
                            "origin_lag": C.ORIGIN_LAG, "avail_lag": C.AVAIL_LAG,
                            "hold_last": sorted(C.HOLD_LAST),
                            "recorded_in_window": sorted(C.RECORDED_IN_WINDOW),
                            "source": "copied from I1_changhua_drivers; not tuned",
                            "ridge_guards": "added after design run 1 (logged look "
                                            "2026-10-20T01:20Z): columns constant over the "
                                            "fit rows dropped; standardised features clipped "
                                            "to their fit-row range"},
        "data_notes": d.notes, "reads": d.reads,
        "splits": summ,
    }
    (C.OUT / "design_skill.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    roles = [
        ("heads 2012-02..2022-12 (release preorigin/heads_observed, flags 0-1)",
         "design_time_evidence", "fit target and scoring target of the design folds"),
        ("R0 out-of-fold heads (stage3_datum_gate_cwa/stage3_per_entry.npz)", "input",
         "physics component and baseline"),
        ("C0 electricity (release preorigin/electricity_census)", "input", "driver"),
        ("W1 canal (release preorigin/surface_water)", "input", "driver"),
        ("P6 public wells (release preorigin/public_wells_member)", "input", "driver"),
        ("W2 ET0 (release preorigin/et0) and the twin rain cache to 2022-12", "input",
         "driver (rain recharge)"),
        ("W1 river gauges H057/H071 (surface_water_v4/preorigin/v2)", "input", "driver"),
        ("zone_county_map/v2", "input", "county of each cell"),
    ]
    pd.DataFrame([{"number_id": "I1.hybrid_tf.design.*", "source": s, "role": r, "note": n}
                  for s, r, n in roles]).to_csv(C.OUT / "data_roles.csv", index=False)
    look = prereg3.record_look(
        ["I1:design_oof_heads_R0_R1e"], "heads:2012-01..2022-12", "I1", "after_the_fact",
        note="Hybrid TF layer design study (task I1_hybrid_tf, 2026-10-20): leave-one-year-out "
             "and forward-chaining skill of per-well transfer-function models, pure and as an "
             "R0 correction, against climatology, persistence, the AR(1) statistical layer and "
             "R0; design-time evidence only. Run 4: canal availability lag corrected from 1 to "
             "10 months after the inventory (2026 canal months are flag-6 extrapolations). "
             "Run 2 added ridge extrapolation guards "
             "(constant columns dropped, features clipped to the fit range) after run 1 showed "
             "errors of tens to thousands of metres at short-record wells; run 3 adds the ARX "
             "one-step family (the AR(1) layer plus a TF model of its innovations) and the "
             "electricity-recorded variants, after run 2 showed the TF layer losing to the "
             "AR(1) layer once drivers are cut. "
             "results/dataprep/reports/I1_hybrid_tf/design_skill.json")
    print("look logged:", look["time"])


if __name__ == "__main__":
    main()
