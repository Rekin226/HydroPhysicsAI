"""Repair round 1: checks of ``heads/recovered_wells/v<N>`` (default v3) (guard, identity, acceptance record,
redaction, segment rule, registered inputs). Writes ``H3_evidence/repair1/checks_v2.json``.

Usage: python -I r1_checks.py [v<N>]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from common import guard, manifest  # noqa: E402
from harness import prereg3  # noqa: E402

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells import inventory  # noqa: E402

VER = sys.argv[1] if len(sys.argv) > 1 else "v3"
V2 = C.REPO / "results/dataprep/heads/recovered_wells" / VER
V1 = C.REPO / "results/dataprep/heads/recovered_wells/v1"
GRANT = ["calibration_target", "datum_fit", "validator", "after_the_fact_analysis"]
REFUSE = ["scoring_target", "baseline", "initial_conditions", "idw_field", "prior",
          "model_design", "selection", "forcing_frozen", "statistical_layer_history",
          "prospective_scorer"]


def main():
    out = {}
    m, _ = manifest.load_manifest(V2 / "MANIFEST.json")
    out["manifest_schema_errors"] = manifest.validate_manifest(m, V2)
    g = {}
    for u in GRANT:
        p = guard.open_product(V2, u)
        t = p.read_table("targets_preorigin.parquet")
        g[u] = {"rows": int(len(t)), "flags": sorted(int(x) for x in t.flag.unique())}
    r = {}
    for u in REFUSE:
        try:
            guard.open_product(V2, u)
            r[u] = "GRANTED (unexpected)"
        except Exception as e:  # noqa: BLE001
            r[u] = type(e).__name__
    out["guard_granted"], out["guard_refused"] = g, r
    t2 = pd.read_parquet(V2 / "targets_preorigin.parquet")
    t1 = pd.read_parquet(V1 / "targets_preorigin.parquet")
    cols = ["sid", "month", "value", "sd", "flag", "method", "sigma_obs_m", "u2_weight",
            "head_m", "head_sd_m"]
    a = t2[cols].sort_values(["sid", "month"]).reset_index(drop=True)
    b = t1[cols].sort_values(["sid", "month"]).reset_index(drop=True)
    num = ["value", "sd", "sigma_obs_m", "u2_weight", "head_m", "head_sd_m"]
    out["identity_v1"] = {
        "rows_v2": int(len(a)), "rows_v1": int(len(b)),
        "keys_equal": bool(a[["sid", "month", "flag", "method"]].equals(
            b[["sid", "month", "flag", "method"]])),
        "max_abs_diff": float(np.nanmax(np.abs(a[num].to_numpy(float)
                                               - b[num].to_numpy(float))))}
    out["flag2_methods"] = sorted(set(t2.loc[t2.flag == 2, "method"]))
    out["latest_month"] = str(t2.month.max())
    out["admitted_in_model"] = sorted(set(t2.sid) & inventory.model_wells())
    val = json.loads((V2 / "validation.json").read_text())
    out["acceptance_record"] = {
        "validation_plan_criterion": val["acceptance_plan"][
            "each_recovered_well_passes_h2_coverage_stratum"],
        "validation_n_pass": val["acceptance_plan"]["n_admitted_passing_H3.d_wells"],
        "manifest_gate_passed": m["gate"]["passed"],
        "manifest_plan_acceptance_passed": m["gate"].get("plan_acceptance_passed"),
        "consistent_with_wp_report": (
            val["acceptance_plan"]["each_recovered_well_passes_h2_coverage_stratum"] is False
            and m["gate"].get("plan_acceptance_passed") is False)}
    txt = (V2 / "rh_probe_preorigin.json").read_text()
    late = [d for d in re.findall(r"\b(20\d{2})-(\d{2})", txt)
            if (int(d[0]), int(d[1])) > (2022, 12)]
    late8 = [d for d in re.findall(r"\b(20\d{6})\b", txt) if d > "20221231"]
    out["redaction"] = {"post_origin_dates_left": len(late) + len(late8)}
    W = pd.read_csv(V2 / "recovered_wells.csv", dtype={"sid": str})
    sc = W[W.step_verdict.notna()]
    out["segment_rule"] = {
        "screened": int(len(sc)),
        "pre_segment_passes": sorted(sc[sc.segment_pre_passes_envelope.astype(bool)].sid),
        "post_segment_passes": sorted(sc[sc.segment_post_passes_envelope.astype(bool)].sid)}
    out["held_back_screened"] = {s: x for s, x in zip(sc[~sc.admitted.astype(bool)].sid,
                                                      sc[~sc.admitted.astype(bool)].reason,
                                                      strict=True)}
    out["registered_inputs_changed"] = {
        pid: prereg3.changed_inputs(prereg3.load_prereg(pid)[0])
        for pid in (C.PREREG_ID, "H3.repair.v1")}
    sup = json.loads((C.REPO / "results/dataprep" / C.PRODUCT / "SUPERSEDED.json").read_text())
    out["superseded_record"] = {"current": sup["current"],
                                "superseded": [s["version"] for s in sup["superseded"]]}
    ok = (not out["manifest_schema_errors"]
          and all(v != "GRANTED (unexpected)" for v in r.values())
          and g["calibration_target"]["flags"] == [0, 1, 2]
          and g["datum_fit"]["flags"] == [0, 1]
          and out["identity_v1"]["keys_equal"] and out["identity_v1"]["max_abs_diff"] == 0.0
          and out["flag2_methods"] == ["daily_bridge_partial"]
          and out["latest_month"] <= C.PRE_END and not out["admitted_in_model"]
          and out["acceptance_record"]["consistent_with_wp_report"]
          and out["redaction"]["post_origin_dates_left"] == 0
          and not any(out["registered_inputs_changed"].values()))
    out["all_ok"] = bool(ok)
    p = C.EVIDENCE / "repair1" / f"checks_{VER}.json"
    p.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
