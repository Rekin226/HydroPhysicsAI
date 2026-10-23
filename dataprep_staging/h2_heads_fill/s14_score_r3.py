"""Registered scoring of WP H2 repair round 3, test ``H2.fresh3.d`` (``H2.repair3.v1``).

A replication of ``H2.fresh.d`` on family-d realizations never generated before (r = 60..69):
the frozen daily bridge (form 'shape', bridged-share sd classes of ``design_params.json``)
scored exactly as ``s4_score`` scores H2.fresh.d (coverage rule gate-3 on clean truth with the
well as cluster; pooled clean-truth RMSE below the twin's backfill, climatology + AR(1),
space-time kriging and same-layer IDW on the IDW cells). ``prereg3.assert_can_score`` runs
first (refuses if the registration is not intact or postdates a look at the test data; logs
the scoring look). The run tables come from ``runner.py pre d 60 .. 69``.

Usage: python -I s14_score_r3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402
from h2_heads_fill import scoring  # noqa: E402
from h2_heads_fill.s4_score import split_conformal, stratum_verdicts  # noqa: E402


def main():
    ok = prereg3.assert_can_score(R3.PREREG_ID_R3, C.WP, test_id=R3.TEST_D3)
    params = json.loads(R3.PARAMS_V1.read_text())
    raw = scoring.load_runs("pre", ["d"], {"d": list(R3.FRESH_D3)})
    p = scoring.product_daily(raw, params)
    p["share_class"] = p["stratum"]
    p["stratum"] = "d"
    clean = p[p.clean.astype(bool)]
    verdict = stratum_verdicts(clean)
    rep = {"test_id": R3.TEST_D3, "family": "d",
           "realizations": [R3.FRESH_D3.start, R3.FRESH_D3.stop - 1],
           "assert_can_score": {"sha256": ok["sha256"], "inputs_verified": ok["inputs_verified"],
                                "post_registration_looks": ok["post_registration_looks"],
                                "unverifiable_prior_looks": len(ok["unverifiable_prior_looks"])},
           "strata_clean": verdict, "strata_missing": [s for s in ("d",) if s not in verdict],
           "pass": "d" in verdict and verdict["d"]["pass"],
           "reported": {
               "all_truth": {"coverage": scoring.coverage_rows(p),
                             "rmse": scoring.rmse_table(
                                 p, ("product",) + scoring.BASELINES + scoring.REPORTED
                             ).to_dict("records")},
               "model_wells_clean": {
                   "coverage": scoring.coverage_rows(clean[clean.in_model.astype(bool)]),
                   "rmse": scoring.rmse_table(clean[clean.in_model.astype(bool)],
                                              ("product",) + scoring.BASELINES
                                              ).to_dict("records")},
               "share_classes_clean": {
                   "coverage": scoring.coverage_rows(clean, by="share_class"),
                   "rmse": scoring.rmse_table(clean, ("product", "router") + scoring.BASELINES,
                                              by="share_class").to_dict("records")},
               "split_conformal_within_fresh_clean": split_conformal(clean),
               "bridgeable_share": float(raw.ok.mean()),
               "not_bridged_reasons": raw[~raw.ok].reason.value_counts().to_dict(),
               "partial_mean_rmse_clean": scoring.rmse(clean.partial - clean.truth),
               "bridge_forms_rmse_clean": {"shape": scoring.rmse(clean.pred_shape - clean.truth),
                                           "lin": scoring.rmse(clean.pred_lin - clean.truth)},
               "n_cells_all": int(len(raw)), "n_cells_clean_scored": int(len(clean))}}
    R3.REG_R3.mkdir(parents=True, exist_ok=True)
    scoring.jdump(rep, R3.REG_R3 / f"{R3.TEST_D3}.json")
    c = verdict["d"]["coverage_gate"]
    print(R3.TEST_D3, "PASS" if rep["pass"] else "FAIL", round(c["coverage"], 3),
          [round(c["lo"], 3), round(c["hi"], 3)], c["k_clusters"],
          {k: (round(x["rmse_product"], 3), round(x["rmse_baseline"], 3))
           for k, x in verdict["d"]["baselines"].items() if x["defined"]})


if __name__ == "__main__":
    main()
