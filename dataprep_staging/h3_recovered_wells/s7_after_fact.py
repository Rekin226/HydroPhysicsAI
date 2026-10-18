"""Step 7 (after the registered scoring): H2's registered run tables restricted to the H3 test
wells. After-the-fact analysis, never a gate (the look is logged first).

* Family d of ``H2.fresh.d`` (r = 10..19) and ``H2.fresh3.d`` (r = 60..69): coverage and RMSE
  of the frozen bridge at the test wells (clean truth; cluster = the well, and per well with
  cluster = the window).
* Family S of ``H2.fresh.S`` (r = 10..19; router with the v1 design factors): coverage of
  H2's router at the test wells per stratum (cluster = the well). The router (flag 3) is not
  released (H2 partial) and never a calibration target (U2); this is information for I1.

Writes ``H3_evidence/after_fact/h2_runs_test_wells.json``.

Usage: python -I s7_after_fact.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h3_recovered_wells import config as C  # noqa: E402

os.environ["H2_WORKDIR"] = str(C.H2_WORKDIR)

import numpy as np  # noqa: E402
from h2_heads_fill import scoring  # noqa: E402
from harness import coverage3, prereg3  # noqa: E402

Z90 = 1.6448536269514722
RUNS = C.H2_EVIDENCE / "run_tables"


def per_well(p, cluster_col):
    out = {}
    for sid, g in p.groupby("sid"):
        err = (g.pred_product - g.truth).to_numpy()
        hits = np.abs(err) <= Z90 * g.sd_product.to_numpy()
        gate = coverage3.coverage_gate(hits, g[cluster_col].astype(str).to_numpy())
        out[sid] = {"n": int(len(g)), "k": int(g[cluster_col].nunique()),
                    "coverage": gate["coverage"], "lo": gate["lo"], "hi": gate["hi"],
                    "pass": gate["pass"], "calibrated": gate["calibrated"],
                    "rmse": scoring.rmse(err)}
    return out


def main():
    reg = json.loads((C.EVIDENCE / "registered_wells.json").read_text())
    wells = reg["test_wells"]
    unit = "H3:after_fact_H2_runs_test_wells"
    prereg3.record_look([unit], "heads:2012-01..2022-12", C.WP, "after_the_fact",
                        members={unit: {"description": "H2 registered family-d (r 10-19, "
                                        "60-69) and family-S (r 10-19) run tables restricted "
                                        "to the 16 H3 test wells", "ids": wells,
                                        "window": "2012-01..2022-12"}},
                        note="after the H3.recovery.v1 scoring; never a gate")
    params = json.loads(C.H2_PARAMS.read_text())
    rep = {"test_wells": wells}
    for name, rs in (("H2.fresh.d", range(10, 20)), ("H2.fresh3.d", range(60, 70))):
        raw = scoring.load_runs("pre", ["d"], {"d": list(rs)}, workdir=RUNS)
        raw = raw[raw.sid.isin(wells)]
        p = scoring.product_daily(raw, params)
        p["cluster_window"] = p.sid + "_r" + p.r.astype(int).astype(str)
        clean = p[p.clean.astype(bool)].copy()
        clean["cluster"] = clean.sid
        rep[name] = {"pooled": scoring.coverage_rows(clean.assign(stratum="d")),
                     "rmse": scoring.rmse_table(clean.assign(stratum="d"),
                                                ("product",) + scoring.BASELINES)
                     .to_dict("records"),
                     "per_well": per_well(clean, "cluster_window")}
    raw = scoring.load_runs("pre", ["S"], {"S": list(range(10, 20))}, workdir=RUNS)
    raw = raw[raw.sid.isin(wells)]
    p = scoring.product_monthly(raw, params)
    clean = p[p.clean.astype(bool)].copy()
    rep["H2.fresh.S"] = {"by_stratum": scoring.coverage_rows(clean),
                         "rmse": scoring.rmse_table(clean, ("product",) + scoring.BASELINES)
                         .to_dict("records"),
                         "pooled_all_S": scoring.coverage_rows(clean.assign(stratum="S_all")),
                         "wells_present": sorted(clean.sid.unique())}
    od = C.EVIDENCE / "after_fact"
    od.mkdir(parents=True, exist_ok=True)
    scoring.jdump(rep, od / "h2_runs_test_wells.json")
    for name in ("H2.fresh.d", "H2.fresh3.d"):
        c = rep[name]["pooled"][0]
        print(name, round(c["coverage"], 3), c["k_clusters"], round(c["rmse_product"], 3))
    for r in rep["H2.fresh.S"]["by_stratum"]:
        print(r["stratum"], round(r["coverage"], 3), r["k_clusters"], round(r["rmse_product"], 3))


if __name__ == "__main__":
    main()
