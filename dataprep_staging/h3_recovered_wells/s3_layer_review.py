"""Step 3: review the layer codes proposed for the on-grid wells that have full observed
coverage but no layer code (plan WP H3 'Acquisition': "assign one from WellDepth, flagged and
reviewed (unverified)").

The proposals are H1a's (``station_meta_preorigin.parquet``: median layer of the 3
nearest-depth coded wells within 10 km; flag 5). This step reviews them, without changing
them, with two pre-origin checks:

1. **Leave-one-site-out accuracy of the proposal rule** on every coded well with a WellDepth
   (the rule applied with the well's own site removed), fan-wide and within 15 km of the
   reviewed wells.
2. **Head evidence** (L1 v3 C20 months 2012-2022): for each reviewed well, the correlation of
   its monthly anomalies (well mean removed) with its nested sibling and with the coded wells
   within 15 km, summarised per layer, and the mean level difference to its sibling.

The proposals stay flag 5 (unverified). No reviewed well enters the R5 rung in this WP: they
are listed as an optional sub-rung that needs the user's sign-off (see the report).

Writes ``H3_evidence/layer_review/layer_review.csv`` and ``layer_review.json``.

Usage: python -I s3_layer_review.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory  # noqa: E402

NEAR_M = 15000.0


def propose(coded: pd.DataFrame, d0: float, x: float, y: float) -> int:
    """H1a's rule (h1_heads_qc/meta.py): median layer of the 3 nearest-depth coded wells
    within 10 km (fan-wide if fewer than 3)."""
    c = coded
    if np.isfinite(x):
        dd = np.hypot(c.x - x, c.y - y)
        near = c[dd <= 10000.0]
        if len(near) >= 3:
            c = near
    k = c.assign(gap=(c.d - d0).abs()).nsmallest(3, "gap")
    return int(np.round(k["layer"].astype(int).median()))


def main():
    inv = pd.read_csv(C.EVIDENCE / "inventory" / "inventory.csv",
                      dtype={"sid": str, "layer_raw": str, "layer_code": str, "site_id": str})
    cnt = json.loads((C.EVIDENCE / "inventory" / "counts.json").read_text())
    rev = cnt["no_layer_full_coverage_on_grid"]["sids"]
    md = inventory.load_meta()
    md = md.assign(x=md.x_twd97, y=md.y_twd97, d=md.WellDepth.where(md.WellDepth > 0))
    coded = md[md.layer_code_raw.astype(str).isin(C.LAYERS) & md.d.notna()].copy()
    coded["layer"] = coded.layer_code_raw.astype(str)
    # 1. leave-one-site-out accuracy of the rule
    rows = []
    for _, r in coded.iterrows():
        others = coded[coded.site_id != r.site_id]
        p = propose(others, r.d, r.x, r.y)
        rows.append({"sid": r.sid, "true": int(r.layer), "pred": p, "x": r.x, "y": r.y})
    L = pd.DataFrame(rows)
    rv = md[md.sid.isin(rev)]
    dmin = np.min(np.hypot(L.x.to_numpy()[:, None] - rv.x.to_numpy()[None, :],
                           L.y.to_numpy()[:, None] - rv.y.to_numpy()[None, :]), axis=1)
    near = dmin <= NEAR_M
    acc = {"fan_wide": {"n": int(len(L)), "accuracy": float((L.true == L.pred).mean()),
                        "confusion": pd.crosstab(L.true, L.pred).to_dict()},
           "within_15km_of_reviewed": {"n": int(near.sum()),
                                       "accuracy": float((L.true[near] == L.pred[near]).mean())
                                       if near.any() else None,
                                       "wells": L[near].sid.tolist()}}
    # 2. head evidence (C20 months, pre-origin)
    hp = guard.open_product(C.L1_DIR, "selection", window_end=C.PRE_END)
    val = hp.read_table("heads_monthly.parquet", columns=["sid", "date", "flag", "c20", "value"])
    val = val[val.flag.isin([0, 1]) & val.c20.astype(bool)]
    Y = val.pivot(index="date", columns="sid", values="value")
    A = Y - Y.mean()
    out = []
    for sid in rev:
        r = md[md.sid == sid].iloc[0]
        sib = md[(md.site_id == r.site_id) & (md.sid != sid)]
        rec = {"sid": sid, "site_id": r.site_id, "town": r.TownName, "well_depth": r.WellDepth,
               "proposed_layer": str(r.layer_code), "layer_flag": int(r.layer_flag),
               "proposal_method": r.layer_method}
        if len(sib):
            s2 = sib.iloc[0].sid
            ok = A[[sid, s2]].dropna() if s2 in A else pd.DataFrame()
            rec.update({"sibling": s2, "sibling_depth": float(sib.iloc[0].WellDepth),
                        "sibling_proposed_layer": str(sib.iloc[0].layer_code),
                        "corr_with_sibling": float(ok[sid].corr(ok[s2])) if len(ok) > 24
                        else None,
                        "mean_level_minus_sibling_m": float((Y[sid] - Y[s2]).mean())
                        if s2 in Y else None})
        dd = np.hypot(coded.x - r.x, coded.y - r.y)
        nb = coded[(dd <= NEAR_M) & (coded.site_id != r.site_id)]
        per = {}
        for lay, g in nb.groupby("layer"):
            cs = []
            for s in g.sid:
                if s in A and sid in A:
                    ok = A[[sid, s]].dropna()
                    if len(ok) > 24:
                        cs.append(float(ok[sid].corr(ok[s])))
            per[lay] = {"n_wells": int(len(g)), "n_corr": len(cs),
                        "median_corr": float(np.median(cs)) if cs else None,
                        "depths": sorted(float(v) for v in g.d)}
        rec["neighbours_15km_by_layer"] = per
        best = {k: v["median_corr"] for k, v in per.items() if v["median_corr"] is not None}
        rec["layer_of_best_median_corr"] = max(best, key=best.get) if best else None
        out.append(rec)
    R = pd.DataFrame([{k: v for k, v in x.items() if k != "neighbours_15km_by_layer"}
                      for x in out])
    od = C.EVIDENCE / "layer_review"
    od.mkdir(parents=True, exist_ok=True)
    R.to_csv(od / "layer_review.csv", index=False)
    rep = {"reviewed": rev, "rule_loso_accuracy": acc, "wells": out,
           "verdict": "proposals stay flag 5 (unverified); not admitted to R5 in this WP",
           "inventory_rows": inv[inv.sid.isin(rev)][["sid", "county", "cell", "l1_c20",
                                                     "record_end_trail", "c20_max_internal",
                                                     "c20_lead"]].to_dict("records")}
    (od / "layer_review.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False,
                                                     default=str))
    pd.set_option("display.width", 250)
    print(json.dumps(acc, default=str)[:1500])
    print(R.to_string())
    for x in out:
        print(x["sid"], {k: (v["n_wells"], v["median_corr"]) for k, v in
                         x["neighbours_15km_by_layer"].items()})


if __name__ == "__main__":
    main()
