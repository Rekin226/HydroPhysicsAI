"""Write the DRAFT N2 prospective policy-response declaration (never registers it).

The draft is a JSON file for the user to approve. This script does not call any prereg
tool, does not append to INDEX.jsonl or looks.jsonl, does not fetch or read any head,
leveling or pile value, and reads no electricity month after 2026-08. It hashes the frozen
model files, counts the existing looks that touch the window (to show the unit is unspent)
and copies the power-check numbers from ``n2_power.py`` / ``n2_power_rollout.py``.

Usage: python -I dataprep_staging/n1_gaps/n2_draft.py
Output: results/dataprep/reports/N1_gaps_2026-10-24/science-15/
        N2.policy_response.prospective_2026-11.v1.DRAFT.json
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from n1_gaps import gaps_lib as G  # noqa: E402

OUT = G.OUT_ROOT / "science-15"
PREREG = G.REPO / "results" / "dataprep" / "preregistration"
UNIT = "I1:prospective_heads_2026-11_2027-10_R0wells"
WINDOW = ("2026-11", "2027-10")
R = G.RUNS
F = G.FWD
P = G.REPO / "results" / "twin" / "prospective_2026-11"


def window_looks() -> dict:
    """Looks whose scope reaches the window (any month 2026-11 or later), by text."""
    rows = [json.loads(line) for line in (PREREG / "looks.jsonl").read_text().splitlines()
            if line.strip()]
    pat = re.compile(r"(2026-1[12]|202[7-9]-\d\d)")
    hits, unit_hits = [], []
    for r in rows:
        scope = " ".join(str(r.get(k) or "") for k in ("data_scope", "scope_windows"))
        if pat.search(scope):
            hits.append({k: r.get(k) for k in ("logged", "prereg_id", "test_id", "data_scope")})
        if UNIT in json.dumps(r.get("units") or r.get("members") or {}):
            unit_hits.append({k: r.get(k) for k in ("logged", "prereg_id", "test_id")})
    return {"n_looks": len(rows), "looks_with_scope_in_or_after_2026-11": hits,
            "looks_naming_the_unit": unit_hits}


def main() -> int:
    os.chdir(G.REPO)
    pw = json.loads((OUT / "n2_power_check.json").read_text())
    pr = json.loads((OUT / "n2_power_rollout.json").read_text())
    tr = json.loads((G.OUT_ROOT / "science-05" / "transfer_innov_vs_cwa.json").read_text())
    el = json.loads((G.OUT_ROOT / "science-13" / "elastic_split.json").read_text())
    lk = window_looks()
    if lk["looks_with_scope_in_or_after_2026-11"] or lk["looks_naming_the_unit"]:
        raise SystemExit(f"the window carries looks: {lk}; the draft's premise fails")
    files = {
        "page_theta": R / "stage3_datum_gate" / "stage3_theta.json",
        "page_column": R / "stage3_datum_gate" / "coupled_leveling" / "vep_zonal_leveling.json",
        "page_forward": F / "datum_gate_innov.npz",
        "page_basis": F / "response_basis_innov.npz",
        "zr_predictions": P / "predictions.csv",
        "zr_protocol": P / "protocol.json",
        "zr_manifest": P / "manifest.json",
        "r0_theta": R / "stage3_datum_gate_cwa" / "stage3_theta.json",
        "r0_column": R / "stage3_datum_gate_cwa" / "coupled_leveling" / "vep_zonal_leveling.json",
        "r0_forward": F / "datum_gate_cwa.npz",
        "x1_county_monthly": G.REPO / "results/dataprep/electricity_x1_tpc/full/v1/"
                                      "county_monthly.parquet",
        "x1_manifest": G.REPO / "results/dataprep/electricity_x1_tpc/full/v1/MANIFEST.json",
        "wells": R / "stage3_datum_gate" / "stage3_wells.csv",
        "units_registry": G.REPO / "results/dataprep/validation_units/v3/units.json",
        "power_check": OUT / "n2_power_check.json",
        "power_rollout": OUT / "n2_power_rollout.json",
        "transfer_check": G.OUT_ROOT / "science-05" / "transfer_innov_vs_cwa.json",
        "elastic_split": G.OUT_ROOT / "science-13" / "elastic_split.json"}
    inputs = {k: G.source(v) for k, v in files.items() if Path(v).exists()}
    missing = [k for k, v in files.items() if not Path(v).exists()]
    s = pr["summary"]
    d = {
        "id": "N2.policy_response.prospective_2026-11.v1",
        "status": "draft, not registered",
        "drafted": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "drafted_by": "dataprep_staging/n1_gaps/n2_draft.py (analyst agent, N1 gaps "
                      "2026-10-24); for the user's approval",
        "not_registered_statement": (
            "This file is a draft. It is not in preregistration/INDEX.jsonl, no prereg tool "
            "was called, no look was recorded, and no head, leveling, pile or post-2026-08 "
            "electricity value was read to write it. It becomes a registration only if the "
            "user approves it and the standard tool writes it to preregistration/ before "
            "the deadline."),
        "registration_deadline": "2026-10-31T16:00:00Z (2026-11-01 00:00 Asia/Taipei), "
                                 "before the first window month exists",
        "title": "Prospective head response of the frozen page model to recorded Yunlin "
                 "pumping anomalies, 2026-11..2027-10 (with a report-only deep-pile arm)",
        "kind": "test",
        "why": (
            "No clean, non-design-time test of the policy response is possible on pre-2023 "
            "data: 2007-10 and 2016-18 are spent (N1.policy_response.v1 steps B and C), step "
            "B's exploratory follow-up is spent (N1.changhua_fix.v1), 2020-22 is spent, and "
            "2023+ never gates (U7). The registered prospective tests on this window score "
            "head LEVELS (prospective_2026-11, free run) or statistical layers (I1 T1/T2, X1 "
            "T3), not the physics' response to a pumping contrast against a zero-response "
            "arm. This draft adds that contrast."),
        "preconditions_before_registration": [
            "USER RULING U4: U4 says post-origin forcing never feeds the physics model's "
            "frozen or prospective forecasts (ruling 2026-10-21 allows recorded drivers only "
            "in a separate statistical layer). This test feeds recorded post-origin pumping "
            "(X1, allowed use 'forcing_recorded_arm') into a SEPARATE counterfactual arm of "
            "the frozen physics model; the registered forecast (ZR) stays exactly as "
            "registered. It needs the user's explicit ruling that such an arm is allowed.",
            "ENGINEERING: the PR-arm hook (a month-by-month pumping multiplier in an "
            "otherwise frozen forward run) is built and unit-tested in an isolated worktree, "
            "and its code sha256 is written into this file, before registration. "
            "dataprep_staging/n1_gaps/n2_power_rollout.py shows the record-period version "
            "with the frozen code (policy_lib.TwinModel rollouts).",
            "the user confirms the primary model (page model, below) or switches it to R0 "
            "before registration; the other becomes the declared secondary"],
        "attempt_policy": (
            "User rule 2026-10-23 ('trying another way must never threaten the test'): "
            "engineering failures (crash, bug, missing file) may be fixed and re-run before "
            "the scoring look, each logged; the test is scored ONCE after the window closes; "
            "a failed, inconclusive or not-informative result is reported as such and never "
            "re-scored on the same units; the alternatives under 'declared_secondary' are "
            "fixed now and reported beside the primary, never in its place; after a failure "
            "at most ONE follow-up registration (new test ids, label 'post-failure, "
            "exploratory', never validation), and it cannot reuse this window; the final "
            "report lists every attempt, failures included."),
        "fresh_units": [UNIT],
        "unit": {
            "id": UNIT,
            "description": "monthly heads 2026-11..2027-10 (flags 0-1) at R0's 158 scored "
                           "wells; 84 Yunlin wells score the primary, 74 Changhua wells are "
                           "report-only (the model carries no Changhua pumping)",
            "window": list(WINDOW),
            "why_unspent": {
                "looks_jsonl": f"{lk['n_looks']} looks; none has a data scope or scope "
                               "window in or after 2026-11 and none names this unit "
                               "(checked by n2_draft.window_looks at drafting)",
                "data": "no head of the window exists at drafting (the last fetched head "
                        "month is 2026-09) and the first window month begins 2026-11-01",
                "check": lk},
            "shared_with": [
                "twin.prospective free-running head test results/twin/prospective_2026-11 "
                "(registered 2026-10-11; same 158 wells, same window; its predictions are "
                "this test's ZR arm and are used unchanged)",
                "I1.hybrid_tf.prospective_2026-11.v1 T1 and T2",
                "X1.hybrid_tf_tpc.prospective_2026-11.v1 T3"],
            "multiplicity": "with this test, 5 comparisons score the same heads; each is "
                            "reported separately with its own decision, none is selected "
                            "post hoc, and no family-wise claim is made"},
        "u7": "2023+ never gates: this test gates only on the fresh 2026-11..2027-10 heads; "
              "2023-2026 electricity enters only as recorded forcing of the PR arm; the "
              "deep-pile arm is report-only",
        "primary_model": {
            "name": "the page model: stage3_datum_gate + coupled_leveling column, forward "
                    "datum_gate_innov (the registered prospective forecast's model)",
            "theta_sha256": inputs["page_theta"]["sha256"],
            "column_sha256": inputs["page_column"]["sha256"],
            "forward_sha256": inputs["page_forward"]["sha256"],
            "member_for_the_contrast": "stage3_datum_gate (main parameter set), CPU "
                                       "rollouts; the 18-set ensemble contrast is secondary",
            "transfer_note": (
                "every N1 verdict came from the stage3_datum_gate_cwa recipe (R0); the "
                "model-only transfer check (science-05) finds the two policy responses "
                "nearly identical: cut30 fan 0.942 vs 0.958 cm, cell r "
                f"{tr['ensemble']['cut30']['cell_pearson_r']:.4f}, layer-2 head change at "
                "the well cells 0.908 vs 0.918 m")},
        "declared_secondary_model": {
            "name": "R0: stage3_datum_gate_cwa + its coupled_leveling column, forward "
                    "datum_gate_cwa",
            "theta_sha256": inputs["r0_theta"]["sha256"],
            "column_sha256": inputs.get("r0_column", {}).get("sha256"),
            "forward_sha256": inputs["r0_forward"]["sha256"],
            "note": "R0 has no registered free-run predictions on this window; its ZR arm "
                    "would be generated from datum_gate_cwa.npz by the same rule before "
                    "registration and hashed here"},
        "arms": {
            "ZR": "zero-response arm: the registered free-running forecast "
                  "(results/twin/prospective_2026-11/predictions.csv, unchanged): climatology "
                  "pumping after the 2022-12 origin",
            "PR": "policy-response arm: ZR + the model contrast c(w, t) = H_PR - H_ZR at "
                  "each well's cell and layer, where H_PR and H_ZR are two rollouts of the "
                  "frozen flow model from the same origin state (the registered forecast's "
                  "restart, innovation update, restart taper 5 km) with identical recharge "
                  "and other forcings, differing only in pumping energy",
            "CL": "the registered prospective test's baseline: month-of-year climatology "
                  "of each well's fitted months before the origin"},
        "forcing_rule_frozen_now": {
            "ratio": "r(t) = X1 Yunlin in-grid county value of month t (product "
                     "electricity_x1_tpc/full; flags 0 census or 5 TPC-downscaled) / its "
                     "2012-2022 mean for that month of year",
            "PR_energy": "E_PR(cell, t) = E_ZR(cell, t) x r(t) for every cell (the model's "
                         "pumping is Yunlin-only), months 2023-01..2027-10",
            "unrecorded": "a month not published in X1 (or a later X1 version whose "
                          "manifest grants forcing_recorded_arm) by the scoring deadline "
                          "takes r = 1",
            "x1_version": "electricity_x1_tpc/full/v1 or the first later version extending "
                          "it with the same frozen model (parameters.json unchanged); the "
                          "version used is hashed at scoring"},
        "primary_metric": {
            "name": "beta_response",
            "definition": "OLS slope of e = obs - ZR on c = PR - ZR over Yunlin well-months "
                          "of the window (flags 0-1, the registered prospective test's "
                          "per-well datum), with well fixed effects (both demeaned per "
                          "well), so a level offset of ZR cannot create the slope",
            "interval": "90 % from a 5 km block cluster bootstrap, B 2000, seed 20261101",
            "co_reported": ["RMSE of PR, ZR and CL (Yunlin; the registered datum)",
                            "the level component: window-mean e on window-mean c across "
                            "wells (confounded with ZR's level bias; report only)",
                            "Changhua wells: the same statistics, report only"]},
        "informative_gate": {
            "rule": "evaluated from the model and the forcing only, before any window head "
                    "is read: RMS of the well-demeaned contrast over Yunlin well-months >= "
                    "0.05 m (N1 thresholds.informative_head_m) and >= 5 bootstrap clusters; "
                    "otherwise the result is 'not informative' and no verdict is given",
            "ruling_if_x1_missing": "if fewer than 6 window months carry a recorded r, the "
                                    "result is 'not informative'"},
        "decision_rule": {
            "response_supported": "beta 90 % interval excludes 0 AND the point estimate "
                                  "lies in [1/1.5, 1.5] (N1 ratio_band) AND RMSE(PR) < "
                                  "RMSE(ZR)",
            "direction_supported_size_not": "beta 90 % interval excludes 0 and beta > 0, "
                                            "but the point estimate is outside "
                                            "[1/1.5, 1.5] or RMSE(PR) >= RMSE(ZR)",
            "not_supported": "beta 90 % interval includes 0, or beta <= 0",
            "not_informative": "the informative gate fails",
            "skill_wording": "a claim that the model forecasts heads better than "
                             "climatology additionally needs RMSE(PR) < RMSE(CL); "
                             "otherwise the page may cite only the response verdict"},
        "baselines": {"CL": "per-well month-of-year climatology (registered prospective)",
                      "ZR": "same frozen model, climatology pumping (registered predictions)"},
        "declared_secondary": {
            "ensemble": "the contrast from the 18-set ensemble mean (GPU or CPU, frozen "
                        "code) instead of the main parameter set",
            "window_only": "r(t) applied only from 2026-11 (r = 1 for 2023-01..2026-10)",
            "r0": "the same test on R0 (declared_secondary_model)",
            "x1_band": "PR with r(t) at the X1 lo90 and hi90 bounds (forcing uncertainty)"},
        "power_check": {
            "reads_the_unit": False,
            "files": ["power_check", "power_rollout"],
            "basis_sensitivity": "all modelled Yunlin pumping off raises layer-2 heads at the "
                                 "84 Yunlin well cells by "
                                 f"{pw['sensitivity_all_yunlin_pumping_off']['layer2_head_rise_at_yunlin_wells_m_mean_p10_p90'][0]:.2f} m "
                                 "on average (page basis, horizon)",
            "anomaly_sizes": "X1 Yunlin Nov-Oct annual anomalies 2012-2025: |a| p50 "
                             f"{pw['annual_anomaly_abs_p50_p90'][0]:.3f}, p90 "
                             f"{pw['annual_anomaly_abs_p50_p90'][1]:.3f}; last 12 recorded "
                             f"months ({pw['last_12_recorded_months']['window']}) "
                             f"{pw['last_12_recorded_months']['annual_anomaly']:+.3f}",
            "model_dynamics_replay": "frozen page model, record 2012-2022, PR vs ZR: "
                                     "Yunlin window-mean contrast |c| p50 "
                                     f"{s['yunlin_abs_mean_contrast_m_p50_p90'][0]:.2f} m "
                                     f"(p90 {s['yunlin_abs_mean_contrast_m_p50_p90'][1]:.2f}); "
                                     "within-window (well-demeaned) RMS p50 "
                                     f"{s['yunlin_within_window_rms_m_p50_p90'][0]:.2f} m "
                                     f"(p90 {s['yunlin_within_window_rms_m_p50_p90'][1]:.2f}); "
                                     f"{s['slope_mean_contrast_per_unit_annual_anomaly_m']:.1f} m "
                                     "per unit annual anomaly",
            "reading": "the contrast is 10-15 times the informative threshold in a typical "
                       "year, so a 'not informative' outcome is unlikely; detectability of "
                       "beta then rests on the residual noise of ZR (post-origin head RMSE "
                       "1.5-2.0 m on the page's published scores)",
            "caveat": "X1's Yunlin TPC seasonal profile correlates 0.37 with the census and "
                      "its 2026H1 fold is its worst; the within-window contrast therefore "
                      "tests the model and the X1 forcing jointly, and a 'not supported' "
                      "result cannot separate the two"},
        "priors_disclosed": {
            "G1": "N1.phase12_response.v1 G1 (masked 2015): obs-on-model slope 2.13 "
                  "(model head response about half the observed)",
            "step_C": "N1.policy_response.v1 step C Yunlin (masked 2016-18): slope 1.21 "
                      "[1.06, 1.36]",
            "expectation": "beta between about 1.2 and 2.1 if those carry over, so "
                           "'direction supported, size not' is a plausible outcome"},
        "subsidence_arm_report_only": {
            "status": "report-only under U7; never gates, never a verdict",
            "data": "WRA 地層下陷監測整合分析月報 table 3 deep benchmarks (printed monthly "
                    "compression), the piles >= 200 m inside the grid, months "
                    "2026-11..2027-10 as published by the scoring date; transcribed with "
                    "twin.report_extract as for the 2024-25 archive",
            "statistic": "window compression at each pile cell: observed vs the PR and ZR "
                         "column outputs (page column, total depth)",
            "power": "the replay's fan subsidence contrast is about "
                     f"{s['fan_subsidence_contrast_per_year_cm_p50_abs']:.2f} cm per year "
                     "(p50); the column's 2024-25 RMSE on the >= 200 m piles was 0.50 cm; "
                     "the arm is unlikely to separate PR from ZR",
            "mechanism_note": "in the page's run the avoided subsidence is entirely elastic "
                              "(science-13: inelastic part "
                              f"{el['policies']['cut30']['by_year']['2032']['A_linear']['inelastic']['fan']:+.3f} cm "
                              "for cut30), so the pile contrast is an elastic rebound"},
        "what_a_pass_supports": "the frozen model's head response to real pumping "
                                "anomalies at Yunlin wells on fresh months; nothing about "
                                "the 1-3 yr subsidence size, the long-run cumulative "
                                "subsidence or Changhua levers",
        "scoring": {"not_before": "2028-01-15 (window closed, heads fetched, TPC/X1 month "
                                  "2027-10 published with its 2-7 week lag)",
                    "with": "the same scoring session as prospective_2026-11 is allowed; "
                            "the order of scoring within the session does not change any "
                            "decision"},
        "inputs": inputs,
        "missing_inputs_at_drafting": missing,
    }
    G.write_new(OUT / "N2.policy_response.prospective_2026-11.v1.DRAFT.json", d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
