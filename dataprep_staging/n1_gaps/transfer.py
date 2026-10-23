"""Transfer check: the page's model (datum_gate_innov) against R0's sibling (datum_gate_cwa).

Every N1 verdict (G1, A1, step B, step C) came from the stage3_datum_gate_cwa recipe
(theta 789d0e45, rebuilt rain, pump spread 0.5 km at its lower bound) or its masked refits;
the page displays stage3_datum_gate (theta 036a7a17, vendor rain, spread 2.71 km). Section
11 of the evidence document compares the two on head, leveling and pile scores only. This
script compares their POLICY RESPONSES, model output against model output:

- the solved ensembles (``datum_gate_{innov,cwa}.npz``, coupled_leveling rheology, the
  column the page shows): fan, county, township and cell-level avoided subsidence for
  cut30 and retire_aqua at Dec 2032 (forward from Dec 2022, as the page), top-decile cell
  Jaccard, headline townships, timing, and the layer-2 head change at the 158 well cells;
- the one-member response bases (``response_basis_{innov,cwa}.npz``): per-class fan,
  township and top-decile agreement and the lever orders (cm and per 100 GWh/yr).

Well cells come from the run's ``stage3_wells.csv`` locations only (sid, layer, x, y);
no head or subsidence observation is read. Nothing is scored or registered.
Usage: python -I dataprep_staging/n1_gaps/transfer.py   (CPU, under a minute)
Output: results/dataprep/reports/N1_gaps_2026-10-24/science-05/transfer_innov_vs_cwa.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from n1_gaps import gaps_lib as G  # noqa: E402

OUT = G.OUT_ROOT / "science-05"
MODELS = {"innov": "stage3_datum_gate", "cwa": "stage3_datum_gate_cwa"}
THETAS = ("stage3_datum_gate", "stage3_datum_gate_cwa", "stage3_datum_gate_m1618",
          "stage3_datum_gate_m15", "stage3_spreadL_gate")
SCEN = ("cut30", "retire_aqua")


def spearman(a, b) -> float:
    from scipy.stats import spearmanr

    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok]).statistic)


def ensemble(tag: str) -> dict:
    p = G.FWD / f"datum_gate_{tag}.npz"
    z = np.load(p, allow_pickle=False)
    names = [str(n) for n in z["scenario_names"]]
    labels = [str(r) for r in z["rheology_labels"]]
    S = z["subs_mean_by_rheology"][labels.index("coupled_leveling")] * 100.0   # cm
    H = z["heads_mean"]
    o = int(z["origin"])
    dates = pd.DatetimeIndex(z["dates"])
    b = names.index("baseline")
    fwd = S[..., :] - S[..., o:o + 1]                # forward from Dec 2022
    out = {"path": p, "dates": dates, "origin": o, "base_fwd": fwd[b]}
    for s in SCEN:
        i = names.index(s)
        out[s] = {"avoided": -(fwd[i] - fwd[b]),       # (A, T) cm, + = less sinking
                  "dh": H[i] - H[b]}                  # (L, A, T) m
    return out


def basis(tag: str) -> dict:
    from hydrophysics.twin.app.prep import basis_deltas, year_ends

    p = G.FWD / f"response_basis_{tag}.npz"
    z = np.load(p, allow_pickle=False)
    names = [str(n) for n in z["scenario_names"]]
    ye, years = year_ends(pd.DatetimeIndex(z["dates"]))
    bd = basis_deltas(z["subs_mean"] * 100.0, z["heads_mean"], names, ye)
    yr = years.index(2022)
    d = bd["dsub"] - bd["dsub"][..., yr:yr + 1]
    return {"path": p, "avoided": -d[..., -1], "dh2": bd["dheadL2"][..., -1],
            "members": [str(m) for m in z["member_labels"]]}


def compare_fields(a: np.ndarray, b: np.ndarray, town, nT, names, elig, ch) -> dict:
    ta, tb = G.town_means(a, town, nT), G.town_means(b, town, nT)
    da, db = G.top_decile(a), G.top_decile(b)
    diff = a - b
    return {
        "fan_mean_cm": [float(a.mean()), float(b.mean())],
        "yunlin_mean_cm": [float(a[~ch].mean()), float(b[~ch].mean())],
        "changhua_mean_cm": [float(a[ch].mean()), float(b[ch].mean())],
        "cell_pearson_r": float(np.corrcoef(a, b)[0, 1]),
        "cell_abs_diff_p50_p95_max_cm": [float(np.percentile(np.abs(diff), 50)),
                                         float(np.percentile(np.abs(diff), 95)),
                                         float(np.abs(diff).max())],
        "cells_abs_diff_gt_0p25cm": int((np.abs(diff) > 0.25).sum()),
        "top_decile_cell_jaccard": G.jaccard(da, db),
        "township_spearman": spearman(ta, tb),
        "township_abs_diff_max_cm": float(np.nanmax(np.abs(ta - tb))),
        "township_cm": {names[i]: [round(float(ta[i]), 4), round(float(tb[i]), 4)]
                        for i in range(nT)},
        "headline_towns": [[names[i] for i in G.top_towns(ta, elig)],
                           [names[i] for i in G.top_towns(tb, elig)]],
        "headline_leader_same": G.top_towns(ta, elig)[:1] == G.top_towns(tb, elig)[:1],
        "headline_top3_same_set": set(G.top_towns(ta, elig)) == set(G.top_towns(tb, elig)),
    }


def main() -> int:
    os.chdir(G.REPO)
    D = G.load_payload()
    town, names, elig = G.townships(D)
    nT = len(names)
    gg = G.grid_groups()
    ch = gg["changhua"]
    cent = gg["grid"].centroids()
    wells = pd.read_csv(G.RUNS / "stage3_datum_gate" / "stage3_wells.csv",
                        usecols=["sid", "layer", "x", "y"], dtype={"sid": str})
    from n1_policy.policy_lib import nearest_cell
    wc = nearest_cell(cent, wells[["x", "y"]].to_numpy(float), 1000.0)
    wc = wc[wc >= 0]
    G.log(f"{len(wells)} well locations, {len(wc)} inside the grid "
          f"({len(np.unique(wc))} distinct cells)")

    E = {k: ensemble(k) for k in MODELS}
    if not (E["innov"]["dates"] == E["cwa"]["dates"]).all():
        raise SystemExit("date axes differ")
    dates = E["innov"]["dates"]
    T_end = len(dates) - 1
    t26 = int(np.nonzero((dates.year == 2026) & (dates.month == 12))[0][0])
    res = {"what": "policy response of the page's model (innov) against R0's sibling (cwa); "
                   "pairs are [innov, cwa]; avoided = baseline minus policy forward "
                   "subsidence from Dec 2022, cm, coupled_leveling column",
           "horizon": str(dates[T_end].date())[:7], "ensemble": {}, "basis": {}}
    for s in SCEN:
        a, b = E["innov"][s]["avoided"], E["cwa"][s]["avoided"]
        blk = compare_fields(a[:, T_end], b[:, T_end], town, nT, names, elig, ch)
        blk["share_by_dec2026"] = [float(a[:, t26].mean() / a[:, T_end].mean()),
                                   float(b[:, t26].mean() / b[:, T_end].mean())]
        h_a = E["innov"][s]["dh"][1][wc, T_end]
        h_b = E["cwa"][s]["dh"][1][wc, T_end]
        blk["layer2_head_change_at_well_cells_m"] = {
            "n_cells": int(len(wc)), "mean": [float(h_a.mean()), float(h_b.mean())],
            "pearson_r": float(np.corrcoef(h_a, h_b)[0, 1]),
            "abs_diff_p50_p95_max": [float(np.percentile(np.abs(h_a - h_b), 50)),
                                     float(np.percentile(np.abs(h_a - h_b), 95)),
                                     float(np.abs(h_a - h_b).max())],
            "ratio_of_means_innov_over_cwa": float(h_a.mean() / h_b.mean())}
        res["ensemble"][s] = blk
        G.log(f"ensemble {s}: fan {blk['fan_mean_cm']}, r {blk['cell_pearson_r']:.3f}, "
              f"top-decile J {blk['top_decile_cell_jaccard']:.2f}, towns rho "
              f"{blk['township_spearman']:.3f}, headline {blk['headline_towns']}")
    bf = [E["innov"]["base_fwd"][:, T_end], E["cwa"]["base_fwd"][:, T_end]]
    res["ensemble"]["baseline_forward_cm"] = {
        "fan_mean": [float(bf[0].mean()), float(bf[1].mean())],
        "cell_pearson_r": float(np.corrcoef(bf[0], bf[1])[0, 1])}

    B = {k: basis(k) for k in MODELS}
    gwh = np.array([c["gwh"] for c in D["classes"]], float)
    K = len(G.CLASSES)
    for si, start in enumerate((2026, 2030)):
        per = {}
        fa, fb = [], []
        for k, c in enumerate(G.CLASSES):
            r = si * K + k
            a, b = B["innov"]["avoided"][r], B["cwa"]["avoided"][r]
            blk = compare_fields(a, b, town, nT, names, elig, ch)
            blk.pop("township_cm")
            ha, hb = B["innov"]["dh2"][r][wc], B["cwa"]["dh2"][r][wc]
            blk["layer2_head_change_at_well_cells_m_mean"] = [float(ha.mean()), float(hb.mean())]
            blk["layer2_head_well_cells_pearson_r"] = float(np.corrcoef(ha, hb)[0, 1])
            per[c] = blk
            fa.append(a.mean())
            fb.append(b.mean())
        fa, fb = np.array(fa), np.array(fb)
        res["basis"][str(start)] = {
            "per_class": per,
            "lever_order_cm": [[G.CLASSES[i] for i in np.argsort(-fa)],
                               [G.CLASSES[i] for i in np.argsort(-fb)]],
            "lever_order_per_100gwh": [[G.CLASSES[i] for i in np.argsort(-fa / gwh)],
                                       [G.CLASSES[i] for i in np.argsort(-fb / gwh)]],
            "per_100gwh_cm": {c: [float(100 * fa[k] / gwh[k]), float(100 * fb[k] / gwh[k])]
                              for k, c in enumerate(G.CLASSES)}}
    res["spread_km"] = {}
    for d in THETAS:
        p = G.RUNS / d / "stage3_theta.json"
        if p.exists():
            th = json.loads(p.read_text())
            bh = th["meta"].get("bounds_hit", {}).get("global", {}).get("log_spread_km", {})
            res["spread_km"][d] = {"spread_km": float(th["theta"]["spread_km"]),
                                   "at_lower_bound": bool(bh.get("lo", 0)),
                                   "at_upper_bound": bool(bh.get("hi", 0)),
                                   "theta_sha256_8": G.sha256(p)[:8]}
    res["provenance"] = {
        "code": "dataprep_staging/n1_gaps/transfer.py", "git_head": G.git_head(),
        "inputs": [G.source(E[k]["path"]) for k in MODELS] +
                  [G.source(B[k]["path"]) for k in MODELS] +
                  [G.source(G.PAGE), G.source(G.RUNS / "stage3_datum_gate" / "stage3_wells.csv")],
        "basis_members": {k: B[k]["members"] for k in MODELS},
        "status": "post-hoc model-only comparison; no observation read, nothing scored"}
    G.write_new(OUT / "transfer_innov_vs_cwa.json", res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
