"""Step 6: registered scoring of ``H3.recovery.v1`` (tests ``H3.d_wells`` and ``H3.snapshot``).

``prereg3.assert_can_score`` runs first for each test (refuses if the registration is not
intact or postdates a look at its test data; logs the scoring look).

``H3.d_wells`` (per-well daily-bridge coverage, the H2 stratum that applies to recovered
wells' calibration targets): the frozen bridge of H2 (form 'shape', sd of the bridged-share
class in ``design_params.json``) on the family-d cells of the test wells (r = 100..159),
clean truth (``io_h2.clean_truth``), only bridgeable cells (as H2's product). Per well: rule
gate-3 (``harness.coverage3.coverage_gate``) with cluster = the gap window (well,
realization). Pooled over the test wells (cluster = the well) and against the baselines:
reported, and the pooled RMSE rule decides whether any bridged month is admitted.

``H3.snapshot`` (well 09030212's once-daily era): sigma_snap from the other wells of its raw
layer code (2012-2018 hourly records; ``snapshot.sigma_snap``), then the coverage of
``|snapshot month - C20 month| <= 1.6449 sigma_snap`` on 09030212's own 2012-2018 months by
rule gate-3 with cluster = the calendar quarter.

Usage: python -I s6_score.py
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
import pandas as pd  # noqa: E402

from h3_recovered_wells import snapshot  # noqa: E402

sys.path.insert(0, str(C.REPO / "dataprep_staging"))
from h2_heads_fill import scoring  # noqa: E402
from harness import coverage3, prereg3  # noqa: E402

Z90 = 1.6448536269514722
RH_WELL = "09030212"


def jdump(obj, path: Path):
    scoring.jdump(obj, path)


def score_d(reg: dict) -> dict:
    ok = prereg3.assert_can_score(C.PREREG_ID, C.WP, test_id=C.TEST_D)
    params = json.loads(C.H2_PARAMS.read_text())
    parts = [pd.read_parquet(C.RUN_DIR / f"d_r{r}.parquet") for r in C.FRESH_D]
    raw = pd.concat(parts, ignore_index=True)
    p = scoring.product_daily(raw, params)
    p["cluster_window"] = p.sid + "_r" + p.r.astype(int).astype(str)
    clean = p[p.clean.astype(bool)].copy()
    wells = reg["test_wells"]
    per = {}
    for sid in wells:
        g = clean[clean.sid == sid]
        if g.empty:
            per[sid] = {"k_windows": 0, "n_cells": 0, "pass": False, "calibrated": False,
                        "reason": "no scored clean bridgeable cell"}
            continue
        err = (g.pred_product - g.truth).to_numpy()
        hits = np.abs(err) <= Z90 * g.sd_product.to_numpy()
        gate = coverage3.coverage_gate(hits, g.cluster_window.to_numpy())
        k = int(g.cluster_window.nunique())
        rec = {"k_windows": k, "n_cells": int(len(g)), "coverage_gate": gate,
               "rmse_bridge": scoring.rmse(err),
               "rmse_backfill": scoring.rmse(g.pred_backfill - g.truth),
               "rmse_clim_ar": scoring.rmse(g.pred_clim_ar - g.truth),
               "calibrated": bool(gate.get("calibrated")) and k >= C.MIN_WINDOWS_PER_WELL}
        rec["pass"] = bool(gate["pass"]) and rec["calibrated"]
        per[sid] = rec
    # pooled over the test wells (cluster = the well), and the baselines (H2's d rule)
    clean["cluster"] = clean.sid
    pooled_cov = scoring.coverage_rows(clean.assign(stratum="d_test_wells"))
    rm = scoring.rmse_table(clean.assign(stratum="d_test_wells"),
                            ("product",) + scoring.BASELINES).to_dict("records")[0]
    beats = {}
    for b in scoring.BASELINES:
        if b == "idw":
            ref, mine = rm.get("rmse_idw"), rm.get("rmse_product_on_idw_cells")
        else:
            ref, mine = rm.get(f"rmse_{b}"), rm.get("rmse_product")
        defined = ref is not None and np.isfinite(ref) and rm.get(f"n_{b}", 0) > 0
        beats[b] = {"defined": bool(defined), "rmse_baseline": ref, "rmse_bridge": mine,
                    "bridge_lower": bool(defined and mine < ref)}
    pooled_rmse_pass = all(v["bridge_lower"] for v in beats.values() if v["defined"])
    allc = p.copy()
    allc["cluster"] = allc.sid
    rep = {"test_id": C.TEST_D, "realizations": [C.FRESH_D.start, C.FRESH_D.stop - 1],
           "assert_can_score": {"sha256": ok["sha256"], "inputs_verified": ok["inputs_verified"],
                                "post_registration_looks": ok["post_registration_looks"],
                                "unverifiable_prior_looks": len(ok["unverifiable_prior_looks"])},
           "per_well": per,
           "n_wells_pass": int(sum(v["pass"] for v in per.values())),
           "n_wells": len(wells),
           "pooled": {"coverage_cluster_well": pooled_cov, "rmse": rm, "baselines": beats,
                      "rmse_pass": bool(pooled_rmse_pass)},
           "reported": {
               "all_truth_coverage_cluster_well": scoring.coverage_rows(
                   allc.assign(stratum="d_test_wells_all_truth")),
               "share_classes_clean": scoring.coverage_rows(
                   clean.assign(stratum=clean.frac_class.astype(str))),
               "bridgeable_share": float(raw.ok.mean()) if len(raw) else None,
               "not_bridged_reasons": raw[~raw.ok].reason.value_counts().to_dict(),
               "bridge_forms_rmse_clean": {"shape": scoring.rmse(clean.pred_shape - clean.truth),
                                           "lin": scoring.rmse(clean.pred_lin - clean.truth)},
               "n_cells_all": int(len(raw)), "n_cells_clean_scored": int(len(clean)),
               "expected_false_fails_if_calibrated": "about 0.05 per well from the interval "
                                                     "alone (more where the 0.85-0.95 window "
                                                     "applies; A1 report)"}}
    clean.to_parquet(C.EVIDENCE / "registered" / f"{C.TEST_D}_cells.parquet")
    return rep


def score_snapshot(reg: dict) -> dict:
    ok = prereg3.assert_can_score(C.PREREG_ID, C.WP, test_id=snapshot.TEST_ID)
    lay = reg["snapshot"]["layer"]
    same_layer = reg["snapshot"]["estimation_wells"]
    daily = snapshot.l1_daily(sorted(set(same_layer) | {RH_WELL}))
    monthly = snapshot.l1_monthly()
    err = snapshot.snapshot_errors(sorted(set(same_layer) | {RH_WELL}), daily, monthly)
    sig = snapshot.sigma_snap(err, same_layer, exclude=[RH_WELL])
    own = err[err.sid == RH_WELL].copy()
    own["quarter"] = pd.PeriodIndex(own.month, freq="M").asfreq("Q").astype(str)
    hits = np.abs(own.err.to_numpy()) <= Z90 * sig["sigma_snap"]
    gate = coverage3.coverage_gate(hits, own.quarter.to_numpy()) if len(own) else {}
    k = int(own.quarter.nunique())
    calibrated = bool(gate.get("calibrated")) and k >= 20
    months, scr = snapshot.snapshot_months(RH_WELL, daily)
    rep = {"test_id": snapshot.TEST_ID, "well": RH_WELL, "layer": lay,
           "assert_can_score": {"sha256": ok["sha256"], "inputs_verified": ok["inputs_verified"],
                                "post_registration_looks": ok["post_registration_looks"],
                                "unverifiable_prior_looks": len(ok["unverifiable_prior_looks"])},
           "sigma_snap": sig, "own_months": int(len(own)), "own_rmse": scoring.rmse(own.err),
           "own_bias": float(own.err.mean()) if len(own) else None,
           "coverage_gate": gate, "k_quarters": k, "calibrated": calibrated,
           "pass": bool(gate.get("pass")) and calibrated,
           "screen_2019_2022": scr, "snapshot_months_2019_2022": int(len(months)),
           "reported": {"sigma_by_layer_all_wells": {
               lv: snapshot.sigma_snap(err, reg["snapshot"]["by_layer"].get(lv, []),
                                       exclude=[RH_WELL])
               for lv in sorted(reg["snapshot"]["by_layer"])}}}
    own.to_csv(C.EVIDENCE / "registered" / "H3.snapshot_own_months.csv", index=False)
    months.to_csv(C.EVIDENCE / "registered" / "H3.snapshot_months_2019_2022.csv", index=False)
    return rep


def main():
    reg = json.loads((C.EVIDENCE / "registered_wells.json").read_text())
    (C.EVIDENCE / "registered").mkdir(parents=True, exist_ok=True)
    d = score_d(reg)
    jdump(d, C.EVIDENCE / "registered" / f"{C.TEST_D}.json")
    print(C.TEST_D, f"{d['n_wells_pass']}/{d['n_wells']} wells pass; pooled RMSE pass",
          d["pooled"]["rmse_pass"],
          {s: (v["k_windows"], round(v["coverage_gate"]["coverage"], 3) if v.get(
              "coverage_gate") else None, v["pass"]) for s, v in d["per_well"].items()})
    s = score_snapshot(reg)
    jdump(s, C.EVIDENCE / "registered" / f"{snapshot.TEST_ID}.json")
    print(snapshot.TEST_ID, "PASS" if s["pass"] else "FAIL", s["sigma_snap"],
          s["coverage_gate"].get("coverage"), s["k_quarters"], s["snapshot_months_2019_2022"])


if __name__ == "__main__":
    main()
