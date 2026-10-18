"""WP H2 design stage (design-time realizations only; plan H2 steps 2-3, 4.5).

Reads the runs of the design-time realizations (families N, S, E, H, T r = 0..4; d r = 0..4;
D, the 2026 outage window in 2013-2022) and fixes, on clean truth only:

1. The factor classes. Gap classes with fewer than ``MIN_CELLS`` clean design cells or fewer
   than ``MIN_CLUSTERS`` clusters are pooled by a rule declared here, before any error is
   computed (``POOL``, followed until a class with enough cells is reached): internal
   classes pool with the next class by length, ``thin`` with ``int_6_11``, ``lead`` and
   ``trail`` into ``edge``, the two network-outage classes with each other.
2. The method per factor class: the router unless a baseline (backfill, clim+AR(1), same-layer
   IDW where defined, space-time kriging) has a lower pooled RMSE on the class's design cells;
   then that baseline (plan H2: "a family where a baseline wins falls back to that baseline").
   A fall-back without a native sd borrows the router's native sd as its scale.
3. The sd factor per class: the split-free conformal factor
   ``q_0.90(|err| / sd_ref) / z_0.95`` (finite-sample quantile) on the class's design cells,
   applied with the 0.05 m floor.
4. The bridge sd per bridged-share class (family d, design r = 0..4): ``q_0.90(|err|) /
   z_0.95``; a share class with fewer than ``MIN_CELLS`` cells pools with its neighbour.

Design-time revision (2026-10-13, after the first design pass, before any fresh realization
existed): the first version of this script also disabled the bridge in every share class
where the router beat it on family d. It disabled all three classes (router 0.22 / 0.25 /
0.26 m against bridge 0.23 / 0.29 / 0.58 m on scattered single-well gaps, where neighbours are
observed). That rule was dropped: the router borrows other wells (flag 3) and U2 never lets it
be a calibration target, while the same-instrument bridge (flag 2) is the only fill U2 admits;
the bridge beats every plan baseline (backfill, clim+AR(1), IDW, STK) on the design cells and
its accuracy is carried by its share-class sd. The router-versus-bridge comparison is
reported. In the same pass the network-outage class was split by run length (see
``methods.classify_cells``) and the daily runs gained the IDW-defined mask.

Second design pass (2026-10-14, resume of the WP after a cut-off; still before any fresh
realization or registration existed): the first pass's runs and parameters had been produced
by the code before the revision above (its parameters still disabled all three bridge classes
and pooled the network outages), so every design run was repeated with the revised code in a
new work directory. Two further design-time changes were made before those runs, on the
evidence of the first pass only: the plan's 3-23-month internal class was split at 6 and 12
months (in-sample coverage with one factor: 0.948 at 3 months, 0.876 at 12), and the bridge
form is re-decided on pre-origin realizations (``s1_bridge_form.py``; the pilot's choice used
2023-25 truth). The first pass's files are kept in ``H2_evidence/design_pass1/``.

Writes ``<workdir>/design/design_params.json`` (hashed by the pre-registration) and
``design_report.json`` (design-time tables: in-sample, optimistic by construction).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from harness.scorers import conformal_quantile  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import scoring  # noqa: E402
from h2_heads_fill.bridge import frac_class  # noqa: E402

MIN_CELLS = 30
MIN_CLUSTERS = 10
POOL = {"int_le2": "int_3_5", "int_3_5": "int_6_11", "int_6_11": "int_12_23",
        "int_12_23": "int_6_11", "int_ge24": "int_12_23", "thin": "int_6_11",
        "lead": "edge", "trail": "edge", "edge": "int_ge24",
        "no_obs_month": "no_obs_run2", "no_obs_run2": "no_obs_month"}
PLAN_FACTORS = {"int_le2": 1.0, "int_3_5": 1.1, "int_6_11": 1.1, "int_12_23": 1.1,
                "int_ge24": 1.25, "lead": 2.5, "trail": 2.5, "no_obs_month": 1.25,
                "no_obs_run2": 1.25}


def class_map(cal: pd.DataFrame) -> dict:
    """Pool classes by counts only (no error is used here): a sparse class follows ``POOL``
    until it reaches a class that is not sparse (edges pool together into ``edge``)."""
    sizes = {c: (len(g), g.cluster.nunique()) for c, g in cal.groupby("cls")}
    sparse = {c for c, (n, k) in sizes.items() if n < MIN_CELLS or k < MIN_CLUSTERS}
    if "lead" in sparse or "trail" in sparse:            # edges pool together
        sparse |= {"lead", "trail"}
    cmap = {}
    for c in sorted(sparse):
        k, seen = POOL.get(c, "int_6_11"), {c}
        while k in sparse and k not in seen and k != "edge":
            seen.add(k)
            k = POOL.get(k, "int_6_11")
        cmap[c] = k if (k == "edge" or k not in sparse) else "int_6_11"
    return cmap


def conformal_factor(err, sd):
    z = np.abs(np.asarray(err, float) / np.asarray(sd, float))
    z = z[np.isfinite(z)]
    return conformal_quantile(z, C.COVERAGE_LEVEL, finite_sample=True) / C.Z90


def main():
    bf = json.loads((C.EVIDENCE_DIR / "design" / "bridge_form.json").read_text())
    if bf["decision"] != C.BRIDGE_FORM:
        raise SystemExit(f"config.BRIDGE_FORM {C.BRIDGE_FORM!r} differs from the stage-1 "
                         f"decision {bf['decision']!r}")
    mon = scoring.load_runs("pre", ["N", "S", "E", "H", "T"], C.DESIGN_R)
    dly = scoring.load_runs("pre", ["d"], C.DESIGN_R)
    Dd = scoring.load_runs("pre", ["D"], [0])
    cal = mon[mon.clean.astype(bool)].copy()
    cmap = class_map(cal)
    cal["key"] = [cmap.get(c, c) for c in cal.cls]
    method_by_class, sd_ref, factor, rm_tab = {}, {}, {}, {}
    for k, g in cal.groupby("key"):
        r = {m: scoring.rmse(g[f"pred_{m}"] - g.truth) for m in ("router",) + scoring.BASELINES
             if m != "idw"}
        gi = g[g.idw_defined.astype(bool)]
        r["idw"] = scoring.rmse(gi.pred_idw - gi.truth) if len(gi) >= MIN_CELLS else np.nan
        r["router_on_idw_cells"] = scoring.rmse(gi.pred_router - gi.truth) if len(gi) else np.nan
        best = "router"
        for b in scoring.BASELINES:
            ref = r["router_on_idw_cells"] if b == "idw" else r["router"]
            if np.isfinite(r[b]) and r[b] < ref and r[b] < r.get(best, np.inf):
                best = b
        method_by_class[k] = best
        sd_ref[k] = best if best in scoring.NATIVE_SD else "router"
        factor[k] = conformal_factor(g[f"pred_{best}"] - g.truth, g[f"sd_{sd_ref[k]}"])
        rm_tab[k] = {"n": len(g), "n_clusters": int(g.cluster.nunique()), "rmse": r,
                     "method": best, "factor": factor[k]}
    # bridge
    dok = dly[dly.ok & dly.clean.astype(bool)].copy()
    dok["fc"] = [frac_class(f) for f in dok.frac]
    bsd, bmap, btab, bdis = {}, {}, {}, []
    counts = dok.fc.value_counts().to_dict()
    for k in range(1, len(C.BRIDGE_FRAC_CLASSES) + 1):
        bmap[k] = k
        if counts.get(k, 0) < MIN_CELLS:
            nb = k + 1 if k < len(C.BRIDGE_FRAC_CLASSES) else k - 1
            bmap[k] = nb
    for k in range(1, len(C.BRIDGE_FRAC_CLASSES) + 1):
        g = dok[dok.fc.map(bmap) == bmap[k]]
        if len(g) == 0:
            continue
        e = g.pred_bridge - g.truth
        bsd[str(k)] = conformal_quantile(np.abs(e.to_numpy()), C.COVERAGE_LEVEL,
                                         finite_sample=True) / C.Z90
        rr = {m: scoring.rmse(g[f"pred_{m}"] - g.truth) for m in
              ("bridge", "router", "backfill", "clim_ar", "stk")}
        gi = g[g.idw_defined.astype(bool)] if "idw_defined" in g else g.iloc[:0]
        rr["idw"] = scoring.rmse(gi.pred_idw - gi.truth) if len(gi) else np.nan
        rr["partial"] = scoring.rmse(g.partial - g.truth)
        btab[str(k)] = {"n": len(g), "n_clusters": int(g.sid.nunique()), "rmse": rr,
                        "sd": bsd[str(k)], "pooled_with": bmap[k]}
    params = {
        "schema": "h2-design-params/1",
        "design_realizations": {k: [v.start, v.stop - 1] for k, v in C.DESIGN_R.items()},
        "truth": "clean (io_h2.clean_truth)",
        "class_map": cmap, "method_by_class": method_by_class, "sd_ref_by_class": sd_ref,
        "factor_by_class": factor, "bridge_sd_by_class": bsd, "bridge_class_pool": bmap,
        "bridge_disabled": bdis, "router_vs_bridge_reported_only": True,
        "bridge_form": C.BRIDGE_FORM, "sd_floor": C.SD_FLOOR, "min_cells": MIN_CELLS,
        "min_clusters": MIN_CLUSTERS, "pool_rule": POOL,
        "plan_factors_for_comparison": PLAN_FACTORS}
    # design-time (in-sample) tables for the report
    pm = scoring.product_monthly(mon, params)
    pdly = scoring.product_daily(dly, params)
    pD = scoring.product_daily(Dd, params)
    rep = {"class_table": rm_tab, "bridge_table": btab,
           "design_rmse_clean": scoring.rmse_table(
               pm[pm.clean.astype(bool)], ("product",) + scoring.BASELINES + scoring.REPORTED
           ).to_dict("records"),
           "design_rmse_all": scoring.rmse_table(
               pm, ("product",) + scoring.BASELINES + scoring.REPORTED).to_dict("records"),
           "design_coverage_clean_insample": scoring.coverage_rows(pm[pm.clean.astype(bool)]),
           "design_daily_rmse_clean": scoring.rmse_table(
               pdly[pdly.clean.astype(bool)].assign(stratum="d"),
               ("product", "router", "backfill", "clim_ar", "idw", "stk")).to_dict("records"),
           "design_daily_coverage_clean_insample": scoring.coverage_rows(
               pdly[pdly.clean.astype(bool)].assign(stratum="d")),
           "D_febmar_rmse_clean": scoring.rmse_table(
               pD[pD.clean.astype(bool)].assign(stratum="D"),
               ("product", "backfill", "clim_ar", "stk")).to_dict("records"),
           "D_febmar_coverage_clean": scoring.coverage_rows(
               pD[pD.clean.astype(bool)].assign(stratum="D")),
           "D_bridgeable_share": float(Dd.ok.mean()) if len(Dd) else np.nan,
           "d_bridgeable_share": float(dly.ok.mean()) if len(dly) else np.nan,
           "n_cells": {"monthly": len(mon), "monthly_clean": len(cal), "d": len(dly),
                       "d_ok_clean": len(dok), "D": len(Dd)}}
    out = C.WORKDIR / "design"
    scoring.jdump(params, out / "design_params.json")
    scoring.jdump(rep, out / "design_report.json")
    scoring.jdump(params, C.EVIDENCE_DIR / "design" / "design_params.json")
    scoring.jdump(rep, C.EVIDENCE_DIR / "design" / "design_report.json")
    print(pd.Series(factor).round(3).to_string())
    print(method_by_class, cmap, bsd, bdis)


if __name__ == "__main__":
    main()
