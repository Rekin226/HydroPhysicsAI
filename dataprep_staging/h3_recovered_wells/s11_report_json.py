"""Step 11: assemble ``results/dataprep/reports/H3.json`` (acceptance table, key numbers) from
the evidence files. The prose report ``H3.md`` is written by hand from the same files.

Usage: python -I s11_report_json.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h3_recovered_wells import config as C  # noqa: E402
from h3_recovered_wells.s9_ledger import product_version  # noqa: E402

EV = C.EVIDENCE
DEVIATIONS = [
    "Gate on the daily-bridge stratum (d) only, per well (cluster = gap window, 60 "
    "realizations r 100-159), instead of all H2 strata: recovered wells' targets can only be "
    "flags 0-2 (U2) and H2's router fill was not released; router strata reported after the "
    "fact only",
    "Declared fallback for wells failing the per-well test: observed months only (flags 0-1); "
    "the plan does not say what happens to a failing well",
    "Candidate list on L1 v3 is 15 wells, not the plan's 16 (coordinate and county-label "
    "repairs, flag-9 removal); 19 on-grid wells stop in 2018-12, not 18",
    "R-h find (09030212, once-daily record in the twin's cache) handled inside H3 with a "
    "registered representativeness test (H3.snapshot) and a persistence screen; L1 unchanged",
    "Post-look tightening: 09030212 held back despite its registered pass, on an "
    "after-the-fact neighbour check (+0.87 m offset); removes data, disclosed",
    "60 instead of 40 family-d realizations, chosen after a counts-only dry run on H2's design "
    "realization r = 0; the dry run's metric outputs were deleted unread (look H3:dryrun_paths)",
    "No secondary 2023-26 test set: the recovered wells are outside the new-network fetch",
]
ISSUES = [
    "I1 R5: read heads/recovered_wells/v1 with calibration_target in R5 only; likelihood "
    "variance sigma_obs_m^2 + head_sd_m^2; a Phase-1 hook must read the targets from the "
    "product (the twin's build_head_field excludes these wells by its 180-day gap rule or the "
    "raw 'y x' coordinates); keep the initial-condition construction unchanged in R5",
    "H2's class-based bridge sd is not calibrated per well: too wide at smooth wells (3 "
    "Changhua candidates and 09150131 at coverage 1.0), too narrow at high-variability Yunlin "
    "wells (0.51-0.79); a per-well scale needs a new registration and also concerns H2 v4's "
    "model wells",
    "09030212: real once-daily record 2019-22 but offset about +0.9 to +1.3 m from its "
    "neighbour relation from mid-2019; needs metadata on the 2019 instrument change before "
    "use, otherwise a with/without sensitivity in R5",
    "chou-shui-data/data/raw/gw_timeseries_raw.csv (and gw_timeseries.csv) is fully "
    "gap-filled by an unknown method (0 NaN, 2.19 M imputed hours incl. multi-year edges): "
    "never use as observations",
    "18 wells stopping 2018-12 have no 2019-22 record in any existing route; vendor 10-minute "
    "records restart 2023-12/2024-08 (post-origin only); recovery would need a WRA request "
    "(letters not approved)",
    "Secondary 2023-26 test set for the 15 recovered wells needs a credentialed vendor fetch "
    "(fetch_amp wells --start 2023-01-01, separate directory)",
    "Six apex no-code wells: layer proposals (flag 5) need user or WRA confirmation; rule "
    "accuracy about 0.70",
    "R5 rung gate (plan acceptance part 2) is evaluated by I1 after the freeze",
]


def load(p):
    return json.loads(Path(p).read_text())


def main():
    cnt = load(EV / "inventory" / "counts.json")
    probe = load(EV / "rh_probe" / "rh_probe.json")
    lr = load(EV / "layer_review" / "layer_review.json")
    d = load(EV / "registered" / f"{C.TEST_D}.json")
    s = load(EV / "registered" / f"{C.TEST_SNAP}.json")
    chk = load(EV / "checks.json")
    pv = product_version()
    val = load(C.REPO / "results/dataprep" / pv / "validation.json")
    af = load(EV / "after_fact" / "h2_runs_test_wells.json")
    per = d["per_well"]
    fails = {k: v for k, v in per.items() if not v["pass"]}
    pooled = d["pooled"]
    acc = [
        {"criterion": "Reconcile counts: one coverage definition settled and recorded",
         "threshold": "definition chosen before candidate selection; every definition in the "
                      "plan recomputed",
         "measured": f"settled on L1 v3 C20 observed-month coverage (2012-01..2022-12); "
                     f">=30 % on-grid excluded wells CH/YL/other: twin rule "
                     f"{cnt['reconcile_ge_0.30']['twin_hour>=0.3:excluded_on_grid']}, C20 "
                     f"{cnt['reconcile_ge_0.30']['l1_c20>=0.3:excluded_on_grid']} (raw "
                     f"coordinates: {cnt['reconcile_ge_0.30']['l1_c20>=0.3:excluded_on_grid_raw_xy']})",
         "passed": True},
        {"criterion": "Inclusion on observed months only (no fill counts; >= 0.80 coverage, "
                      "layer 1-4, on grid, H2 envelope, record to 2022-12)",
         "threshold": "candidates identified by the declared rule",
         "measured": f"{cnt['on_grid_layer14_ge_0.80_l1_c20']['n']} wells reach 0.80 "
                     f"({cnt['on_grid_layer14_ge_0.80_l1_c20']['by_county']}); "
                     f"{cnt['candidates']['n']} meet the full envelope "
                     f"({cnt['candidates']['by_county']}); 07050142 fails (internal gap 25 "
                     f"months)",
         "passed": True},
        {"criterion": "No extrapolated record in calibration (flag 6 beyond a record end)",
         "threshold": "0 target rows after a well's record end; 0 rows after 2022-12; 0 flag "
                      "3-6 rows",
         "measured": f"targets max month {chk['max_month']}; flags "
                     f"{val['targets']['by_flag']}; every admitted well's record reaches "
                     f"2022-12",
         "passed": chk["max_month"] <= C.PRE_END},
        {"criterion": "R-h probe of post-2018 records (existing access only)",
         "threshold": "every existing route checked; no new credentials, no letters",
         "measured": f"{probe['wells']['on_grid']} on-grid wells end 2018-12; post-2018 "
                     f"pre-origin record found for {probe['found_preorigin_post2018_observed']} "
                     f"(09030212, once-daily readings in the twin's cache); none for the other "
                     f"{probe['wells']['on_grid'] - probe['found_preorigin_post2018_observed']}"
                     f"; credentialed refetch not run (no WISENVR_* variables)",
         "passed": True},
        {"criterion": "No-code wells: layer assigned from WellDepth, flagged and reviewed",
         "threshold": "proposal flag 5 with a documented review; not used without sign-off",
         "measured": f"6 apex wells (Nantou); H1a proposals kept flag 5; rule leave-one-site-"
                     f"out accuracy {lr['rule_loso_accuracy']['fan_wide']['accuracy']:.2f} "
                     f"fan-wide, "
                     f"{lr['rule_loso_accuracy']['within_15km_of_reviewed']['accuracy']:.2f} "
                     f"within 15 km; not admitted",
         "passed": True},
        {"criterion": "Each recovered well passes the H2 coverage stratum that applies to its "
                      "targets (daily bridge, per well, rule gate-3, k >= 20 windows)",
         "threshold": "every admitted well's per-well gate passes and is calibrated",
         "measured": f"{d['n_wells_pass']}/{d['n_wells']} test wells pass (09030212, 09080251, "
                     f"09180121, 09200121); 12 of the 15 admitted wells fail: over-coverage "
                     f"(sd too wide) at "
                     + ", ".join(f"{k} {v['coverage_gate']['coverage']:.2f}"
                                 for k, v in fails.items()
                                 if v["coverage_gate"]["coverage"] > 0.9)
                     + "; under-coverage at "
                     + ", ".join(f"{k} {v['coverage_gate']['coverage']:.2f}"
                                 for k, v in fails.items()
                                 if v["coverage_gate"]["coverage"] <= 0.9)
                     + f"; pooled {pooled['coverage_cluster_well'][0]['coverage']:.3f} "
                     f"[{pooled['coverage_cluster_well'][0]['lo']:.3f}, "
                     f"{pooled['coverage_cluster_well'][0]['hi']:.3f}] (16 clusters, "
                     f"uncalibrated)",
         "passed": False},
        {"criterion": "Fallback: no fill whose coverage failed enters a target (declared rule)",
         "threshold": "wells failing the per-well test enter with flags 0-1 only; bridged "
                      "months kept only where the per-well test and the pooled RMSE rule pass",
         "measured": f"pooled bridge RMSE {pooled['rmse']['rmse_product']:.3f} m below STK "
                     f"{pooled['rmse']['rmse_stk']:.3f}, IDW {pooled['rmse']['rmse_idw']:.3f}, "
                     f"clim+AR(1) {pooled['rmse']['rmse_clim_ar']:.3f}, backfill "
                     f"{pooled['rmse']['rmse_backfill']:.3f}; 12 wells observed-only (19 "
                     f"bridged months of v4 dropped), 3 wells keep 6 bridged months",
         "passed": True},
        {"criterion": "Once-daily continuation of 09030212 (H3.snapshot)",
         "threshold": "gate-3 coverage of +-1.6449 sigma_snap on its own 2012-2018 months, "
                      "calibrated (k >= 20 quarters)",
         "measured": f"sigma_snap {s['sigma_snap']['sigma_snap']:.3f} m "
                     f"({s['sigma_snap']['n_wells']} wells, {s['sigma_snap']['n_months']} "
                     f"months); coverage {s['coverage_gate'].get('coverage', float('nan')):.3f} "
                     f"[{s['coverage_gate'].get('lo', float('nan')):.3f}, "
                     f"{s['coverage_gate'].get('hi', float('nan')):.3f}], k {s['k_quarters']}; "
                     f"registered rule would admit the well (coverage with once-daily months "
                     f"{val['envelope_09030212_with_snapshot']['coverage']:.3f}), but it is "
                     f"held back post-look: {val['post_look_hold_back_09030212']}",
         "passed": bool(s["pass"])},
        {"criterion": "Product heads/recovered_wells released with manifest, guard and "
                      "identity checks",
         "threshold": "checker 0 problems; guard grants exactly the declared uses; rows equal "
                      "their v4 source",
         "measured": f"{pv}: hash problems {len(chk['manifest_hash_problems'])}; guard as "
                     f"expected {chk['expected_grants_ok']}; identity max diff "
                     f"{max(v for k, v in chk['identity_with_v4'].items() if isinstance(v, float)):.1e}"
                     f", flags equal {chk['identity_with_v4']['flag_equal']}",
         "passed": (not chk["manifest_hash_problems"]) and chk["expected_grants_ok"]},
        {"criterion": "R5 passes its rung gate (plan acceptance, part 2)",
         "threshold": "I1 ladder rung R5 gate (GPU, after the freeze)",
         "measured": "not evaluated: Phase 1 work (I1), outside Phase 0",
         "passed": False},
    ]
    out = {"wp": "H3", "acceptance": acc, "product_version": pv,
           "admitted": val["admitted"], "targets": val["targets"],
           "registered": {"H3.d_wells": {"n_pass": d["n_wells_pass"], "n": d["n_wells"],
                                         "pooled": pooled},
                          "H3.snapshot": {k: s[k] for k in ("pass", "sigma_snap", "own_rmse",
                                                            "own_bias", "k_quarters",
                                                            "coverage_gate",
                                                            "snapshot_months_2019_2022")}},
           "after_fact": {"H2.fresh.d": af["H2.fresh.d"]["pooled"],
                          "H2.fresh3.d": af["H2.fresh3.d"]["pooled"],
                          "H2.fresh.S": af["H2.fresh.S"]["by_stratum"]},
           "post_look_hold_back_09030212": val["post_look_hold_back_09030212"],
           "status": "partial",
           "deviations_from_plan": DEVIATIONS, "issues_for_downstream": ISSUES,
           "claude_api_tokens": 0, "gpu": "none"}
    (C.REPO / "results/dataprep/reports/H3.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False, default=str))
    for a in acc:
        print("PASS" if a["passed"] else "----", a["criterion"][:70], "|", a["measured"][:160])


if __name__ == "__main__":
    main()
