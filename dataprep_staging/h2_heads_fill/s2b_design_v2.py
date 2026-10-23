"""WP H2 repair round: re-estimate the gap-class sd factors (v2 design; see ``repair_v2``).

Design set: the monthly runs of families N, S, E, H, T at realizations r = 0..4 (v1 design)
and the v1 fresh realizations (N, S, E r = 10..19; H, T r = 10..29), clean truth. Family E
well-gaps are counted once (a well's leading or trailing gap is deterministic, so repeated
draws of the same well repeat the same cells): duplicates of (sid, stratum, month) keep the
first realization. Per gap class (v1 classes, all routed to the router): factor =
q_0.90(|err| / router sd) / z_0.95 (finite-sample conformal quantile, as v1). The method per
class, the class map, the bridge sds and every other parameter are copied from v1 unchanged.

Writes ``H2_evidence/design_v2/design_params_v2.json`` (hashed by the v2 registration) and
``design_report_v2.json`` (in-sample tables on the v2 design set: optimistic by construction).

Usage: python -I s2b_design_v2.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import pandas as pd  # noqa: E402

from h2_heads_fill import repair_v2 as R  # noqa: E402
from h2_heads_fill import scoring  # noqa: E402
from h2_heads_fill.s2_design import conformal_factor  # noqa: E402


def design_set() -> pd.DataFrame:
    parts = []
    for f in R.FAMILIES_V2:
        df = scoring.load_runs("pre", [f], {f: R.DESIGN_R_V2[f]})
        if f == "E":
            df = df.sort_values("r", kind="stable").drop_duplicates(
                ["sid", "stratum", "month"], keep="first")
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def main():
    p1 = json.loads(R.PARAMS_V1.read_text())
    mon = design_set()
    cal = mon[mon.clean.astype(bool)].copy()
    cal["key"] = [scoring.class_key(c, p1) for c in cal.cls]
    fac, tab = {}, {}
    for k, g in cal.groupby("key"):
        ref = p1["sd_ref_by_class"][k]
        meth = p1["method_by_class"][k]
        fac[k] = conformal_factor(g[f"pred_{meth}"] - g.truth, g[f"sd_{ref}"])
        tab[k] = {"n": int(len(g)), "n_clusters": int(g.cluster.nunique()),
                  "factor_v1": p1["factor_by_class"][k], "factor_v2": fac[k],
                  "rmse_router": scoring.rmse(g.pred_router - g.truth)}
    missing = sorted(set(p1["factor_by_class"]) - set(fac))
    if missing:
        raise SystemExit(f"classes without v2 design cells: {missing}")
    p2 = copy.deepcopy(p1)
    p2.update({"schema": "h2-design-params/2", "factor_by_class": fac,
               "design_realizations": {f: R.DESIGN_R_V2[f] for f in R.FAMILIES_V2} | {
                   "d": p1["design_realizations"]["d"]},
               "repair": "v2 (post-look): factors re-estimated on design r0-4 + v1 fresh "
                         "realizations; everything else copied from v1",
               "v1_params_file": str(R.PARAMS_V1)})
    pm = scoring.product_monthly(mon, p2)
    pc = pm[pm.clean.astype(bool)]
    rep = {"class_table": tab, "n_cells": {"monthly": int(len(mon)), "clean": int(len(cal))},
           "insample_coverage_clean_by_family_stratum": scoring.coverage_rows(pc),
           "note": "design-time (v2): in-sample by construction"}
    scoring.jdump(p2, R.PARAMS_V2)
    scoring.jdump(rep, R.DESIGN_DIR_V2 / "design_report_v2.json")
    print(pd.DataFrame(tab).T.round(3).to_string())


if __name__ == "__main__":
    main()
