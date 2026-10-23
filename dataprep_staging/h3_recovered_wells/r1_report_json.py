"""Repair round 1: rewrite ``results/dataprep/reports/H3.json`` as the WP's structured result as
it now stands (the first-round file is kept in ``H3_evidence/repair1/first_round/``). Numbers
are read from the evidence and product files; the prose is ``H3.md``'s Repair round section.

Usage: python -I r1_report_json.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h3_recovered_wells import config as C  # noqa: E402

EV = C.EVIDENCE
R1 = EV / "repair1"
DP = C.REPO / "results/dataprep"
V3 = DP / "heads/recovered_wells/v3"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    stp = json.loads((R1 / "registered" / "H3.step_screen.json").read_text())
    chk = json.loads((R1 / "checks_v3.json").read_text())
    val = json.loads((V3 / "validation.json").read_text())
    dres = json.loads((EV / "registered" / f"{C.TEST_D}.json").read_text())
    pw = {r["sid"]: r for r in stp["per_well"]}
    adm = val["admitted"]["sids"]
    zs = [pw[s]["z"] for s in adm]
    ps = [pw[s]["p"] for s in adm]
    sh = [pw[s]["null_z_q95"] * pw[s]["rmse_loyo"] for s in adm]
    nb = stp["null_by_split"].values()
    rh = pw["09030212"]
    acceptance = [
        {"criterion": "Reconcile counts: one coverage definition settled and recorded",
         "threshold": "definition taken before candidate selection; every definition the plan "
                      "names recomputed",
         "measured": "L1 v3 C20; on-grid excluded wells >= 0.30 (CH/YL/other): twin rule "
                     "17/37/6, C20 10/32/6, raw coordinates 8/32/6; the claim 'fixed before "
                     "any coverage was computed' is withdrawn as unverifiable (config.py's "
                     "final bytes postdate the inventory looks)",
         "passed": True},
        {"criterion": "Inclusion on observed months only",
         "threshold": "declared envelope (>=0.80 C20, layer 1-4, on grid, H2 envelope, record "
                      "to 2022-12); fills never count",
         "measured": "16 wells reach 0.80 (YL 12, CH 4); 15 meet the envelope (YL 12, CH 3); "
                     "07050142 fails (25-month internal gap)",
         "passed": True},
        {"criterion": "No extrapolated record in calibration",
         "threshold": "0 target rows after a record end or after 2022-12; 0 flag 3-6 rows",
         "measured": f"latest month {chk['latest_month']}; flags "
                     f"{val['targets']['by_flag']}; flag 2 only {chk['flag2_methods']}",
         "passed": True},
        {"criterion": "One admission rule for every recovered well (repair of major issue 1)",
         "threshold": "registered level-shift screen applied to all admitted wells with a "
                      "null-calibrated threshold (p <= 0.05)",
         "measured": f"H3.step_screen (H3.repair.v1): {len(adm)}/{len(adm)} screened, "
                     f"{len(stp['admitted_v1']['flagged'])} flagged, "
                     f"{len(stp['admitted_v1']['untestable'])} untestable; z "
                     f"{min(zs):.2f}-{max(zs):.2f}, p {min(ps):.2f}-{max(ps):.2f}; 09030212 z "
                     f"{rh['z']:.2f}, p {rh['p']:.3f} (not flagged; reported only)",
         "passed": True},
        {"criterion": "Product records the plan verdict (repair of major issue 2)",
         "threshold": "validation report and manifest gate agree with the WP report",
         "measured": f"v3 acceptance_plan = "
                     f"{val['acceptance_plan']['each_recovered_well_passes_h2_coverage_stratum']}"
                     f" ({val['acceptance_plan']['n_admitted_passing_H3.d_wells']} of "
                     f"{val['acceptance_plan']['n_admitted']}); gate plan_acceptance_passed "
                     f"{chk['acceptance_record']['manifest_plan_acceptance_passed']}; "
                     f"consistent {chk['acceptance_record']['consistent_with_wp_report']}",
         "passed": bool(chk["acceptance_record"]["consistent_with_wp_report"])},
        {"criterion": "R-h probe of post-2018 records (existing access only)",
         "threshold": "every existing route checked; no new credentials, no letters",
         "measured": "19 on-grid wells end 2018-12; record found for 1 (09030212 once-daily, "
                     "held back); none for 18 in the cache, ls_cache, delivery files, "
                     "new-network fetch or vendor list; the credentialed vendor route (existing "
                     "access in data_fetch_api/) was NOT run: the session's permission system "
                     "refused loading the stored access; r1_vendor_probe.py is ready for the user",
         "passed": False},
        {"criterion": "No-code wells: layer from WellDepth, flagged and reviewed",
         "threshold": "proposals flag 5 with a documented review; not used without sign-off",
         "measured": "6 apex wells; proposal rule accuracy 0.70 fan-wide, 0.73 within 15 km; "
                     "not admitted",
         "passed": True},
        {"criterion": "Each recovered well passes the H2 coverage stratum that applies to its "
                      "targets (plan acceptance part 1; daily bridge, per well, gate-3, k >= 20)",
         "threshold": "every admitted well passes H3.d_wells",
         "measured": f"{val['acceptance_plan']['n_admitted_passing_H3.d_wells']} of "
                     f"{len(adm)} admitted wells pass ({dres['n_wells_pass']}/"
                     f"{dres['n_wells']} test wells); 5 over-cover (0.96-1.00), 7 under-cover "
                     "(0.51-0.79); pooled 0.859 [0.740, 0.929], 16 clusters, uncalibrated",
         "passed": False},
        {"criterion": "Fallback: no fill whose coverage failed enters a target",
         "threshold": "failing wells enter with flags 0-1 only; bridged months only where both "
                      "the per-well test and the pooled RMSE rule pass",
         "measured": "12 wells observed only; 6 bridged months at 3 passing wells; pooled "
                     "bridge RMSE 0.486 m below STK 0.585, IDW 0.750, clim+AR(1) 0.864, "
                     "backfill 1.220",
         "passed": True},
        {"criterion": "Once-daily continuation of 09030212 (H3.snapshot)",
         "threshold": "gate-3 coverage of +-1.6449 sigma_snap on its own 2012-2018 months, "
                      "calibrated (k >= 20 quarters)",
         "measured": "registered result passed (sigma_snap 0.051 m; coverage 1.000 [0.857, "
                     "1.000], k 26) but uninformative (own RMSE 0.0025 m, point window off; "
                     "24/52 estimation wells fail the same test); well held back by "
                     "H3.repair.v1 rule 3",
         "passed": True},
        {"criterion": "Product released with manifest, guard and identity checks",
         "threshold": "checker 0 problems on the current version; guard grants exactly the "
                      "declared uses; rows equal their source",
         "measured": f"v3: checker ok (67 files), schema errors "
                     f"{len(chk['manifest_schema_errors'])}; guard grants "
                     f"{sorted(chk['guard_granted'])} and refuses "
                     f"{len(chk['guard_refused'])} probed uses; rows equal v1 (max diff "
                     f"{chk['identity_v1']['max_abs_diff']}); v2: 1 expected code mismatch "
                     "(r1_package.py corrected for v3; archived bytes match v2's manifest)",
         "passed": bool(chk["all_ok"])},
        {"criterion": "R5 passes its rung gate (plan acceptance part 2)",
         "threshold": "I1 ladder rung R5 gate",
         "measured": "not evaluated: I1 work, after the freeze, on the GPU",
         "passed": False},
    ]
    products = [
        {"path": str(V3) + "/", "layer": "L2",
         "manifest": f"MANIFEST.json sha256 {sha(V3 / 'MANIFEST.json')}; checker ok (67 "
                     "files). Gate: passed = release rule (H3.recovery.v1 envelope/bridge "
                     "rule with fallback + H3.repair.v1 step screen); plan_acceptance_passed "
                     "= false (3 of 15 pass H3.d_wells). Files: targets_preorigin.parquet "
                     "(1,839 rows, 15 wells, flags 0/1/2 = 1211/622/6, identical to v1), "
                     "recovered_wells.csv (with H3.d_wells and H3.step_screen columns and the "
                     "segment check), inventory.csv, rh_probe_preorigin.json (post-origin "
                     "dates redacted), layer_review.csv, validation.json, PRODUCT_CARD.md",
         "allowed_uses": "calibration_target (pre-declared I1 rung R5 only), datum_fit (flags "
                         "0-1), validator (flags 0-1 only, per card), after_the_fact_analysis; "
                         "scoring_target, baseline, initial_conditions, idw_field, prior, "
                         "model_design, selection and the rest forbidden",
         "fit_window_end": "2022-12"},
        {"path": str(DP / "heads/recovered_wells/SUPERSEDED.json"), "layer": "version record",
         "manifest": "dataprep-superseded/1: current v3; v1 and v2 superseded "
                     "(after_the_fact_analysis); checker note on v2's expected code mismatch "
                     "with the archived packager bytes",
         "allowed_uses": "provenance record", "fit_window_end": "n/a"},
        {"path": str(DP / "heads/recovered_wells/v2") + "/", "layer": "L2 (superseded)",
         "manifest": f"MANIFEST.json sha256 {sha(DP / 'heads/recovered_wells/v2/MANIFEST.json')}"
                     "; same rows, gate and acceptance record as v3; its rh_probe_preorigin."
                     "json kept two post-origin dates as keys; 1 expected code-hash mismatch",
         "allowed_uses": "superseded by v3; use v3", "fit_window_end": "2022-12"},
        {"path": str(DP / "heads/recovered_wells/v1") + "/", "layer": "L2 (superseded)",
         "manifest": "MANIFEST.json (checker ok, 46 files); superseded by v3 (contradictory "
                     "acceptance record, single-well post-look hold-back, post-origin dates, "
                     "missing probe sources)",
         "allowed_uses": "superseded by v3; use v3", "fit_window_end": "2022-12"},
        {"path": str(DP / "preregistration/H3.repair.v1.json"), "layer": "preregistration",
         "manifest": f"sha256 {sha(DP / 'preregistration/H3.repair.v1.json')}; written "
                     "2026-10-14T01:46:06Z by harness.prereg3; test H3.step_screen; 24 hashed "
                     "inputs unchanged at scoring (01:46:11Z); no post-registration looks",
         "allowed_uses": "registration of the repair-round test and admission rule",
         "fit_window_end": "n/a"},
        {"path": str(DP / "preregistration/H3.recovery.v1.json"), "layer": "preregistration",
         "manifest": "sha256 f3c17f78641386f9c2cad23facc7b111edba4a24f5cd44f0a002feaabed6dcd5 "
                     "(first round; tests H3.d_wells, H3.snapshot; inputs unchanged)",
         "allowed_uses": "registration of the first-round tests and admission rule",
         "fit_window_end": "n/a"},
        {"path": str(R1) + "/", "layer": "evidence",
         "manifest": "none. registered/ (H3.step_screen.json, _wells.csv, _null.parquet), "
                     "checks_v3.json, checks_v2_first.json, check_recovered_wells.json, "
                     "code_as_packaged_v2/r1_package.py, first_round/ (H3.md, H3.json)",
         "allowed_uses": "provenance only", "fit_window_end": "n/a"},
    ]
    code = sorted(str(p) for p in C.CODE_DIR.glob("*.py"))
    key = [
        f"Current product heads/recovered_wells/v3: {len(adm)} wells (YL 12, CH 3; layers "
        "1/2/3/4 = 1/5/7/2), 1,839 targets (flags 0/1/2 = 1211/622/6), identical to v1 rows; "
        "tiers: 3 bridged, 12 observed only",
        "Plan acceptance part 1 FAILED: 3 of 15 admitted wells (4 of 16 test wells) pass "
        "H3.d_wells; recorded as false in v3 validation.json and gate.plan_acceptance_passed",
        f"H3.step_screen (H3.repair.v1): 0 of 15 admitted wells flagged; z "
        f"{min(zs):.2f}-{max(zs):.2f}, p {min(ps):.2f}-{max(ps):.2f}; null 142-151 model "
        "wells per split",
        f"09030212: z {rh['z']:.2f} (bias +{rh['bias']:.2f} m, LOYO RMSE "
        f"{rh['rmse_loyo']:.3f} m), p {rh['p']:.3f}: not a well-specific shift; first-round "
        "hold-back reason withdrawn; held back by rule 3 (uninformative snapshot test, "
        "post-look screen)",
        "Null share of model wells with z >= 2 (the first round's hold-back rule): "
        f"{min(v['share_z_ge_2'] for v in nb):.2f}-{max(v['share_z_ge_2'] for v in nb):.2f} "
        "(0.19-0.26 at the 2019 splits); null z95 "
        f"{min(v['z_q95'] for v in nb):.2f}-{max(v['z_q95'] for v in nb):.2f}",
        f"Shift the screen can leave unflagged (null z95 x LOYO RMSE): "
        f"{min(sh):.2f}-{max(sh):.2f} m across the admitted wells",
        "H3.d_wells unchanged: pooled coverage 0.859 [0.740, 0.929] (uncalibrated); bridge "
        "RMSE 0.486 m vs STK 0.585, IDW 0.750, clim+AR(1) 0.864, backfill 1.220",
        "R-h: 1 of 19 on-grid wells ending 2018-12 has a 2019-22 record (09030212, held "
        "back); credentialed vendor route not run (permission refused)",
        "Looks: 12 H3 (3 in the repair round: 2 verification disclosures, 1 score); "
        "ledger 61 H3 rows, 0 conflicts; A1 audit refuses no H3 test",
        "GPU none; Claude API 0 tokens",
    ]
    deviations = [
        "Only the daily-bridge stratum (d) is gated, per well (cluster = gap window, 60 "
        "realizations r 100-159); router strata after the fact only (H2's router not released)",
        "Declared fallback for wells failing the per-well test: observed months only "
        "(flags 0-1); the plan does not say what happens to a failing well",
        "Candidate list on L1 v3 is 15 wells, not the plan's 16; 19 on-grid wells stop in "
        "2018-12, not 18",
        "60 instead of 40 family-d realizations after a counts-only dry run (disclosed)",
        "Repair round: a level-shift screen (H3.step_screen) not in the plan, registered after "
        "the first round's results and the verifier's per-well numbers were seen; not blind "
        "for 07260241, 09080241, 09150131, 09130321",
        "09030212 stays held back under H3.repair.v1 rule 3 (uninformative admission test, "
        "post-look screen) although the first-round registered rule admitted it: a disclosed "
        "tightening; its neighbour-offset reason is withdrawn",
        "Manifest gate keeps passed = true for the release rule and records the failed plan "
        "acceptance as plan_acceptance_passed = false (the A0 guard refuses non-design uses of "
        "a product whose gate has not passed)",
        "Credentialed vendor route of R-h not run (session not permitted to load the stored "
        "access); no secondary 2023-26 test set",
        "v2 superseded within the repair round (two post-origin dates left as JSON keys, an "
        "overstated resolution statement); v2 reports 1 expected code-hash mismatch",
    ]
    issues = [
        "I1 R5: read heads/recovered_wells/v3 with calibration_target in R5 only; likelihood "
        "variance sigma_obs_m^2 + head_sd_m^2; a Phase-1 hook must read the targets from the "
        "product (build_head_field excludes these wells); keep the initial-condition "
        "construction unchanged",
        f"I1 R5: add a sensitivity member with separate datum offsets before and after each "
        f"well's split (2019 for 12 Yunlin wells, 2020-21 for 3 Changhua wells); the step "
        f"screen leaves shifts of {min(sh):.1f}-{max(sh):.1f} m unflagged",
        "Validators must filter to flags 0-1; A0 could add validator: {0, 1} to "
        "common/uses.py FLAGS_BY_USE (other WPs may not change that behaviour)",
        "R-h / secondary test set: the user can run dataprep_staging/h3_recovered_wells/"
        "r1_vendor_probe.py with WISENVR_BASE_URL, WISENVR_USERNAME, WISENVR_PASSWORD, "
        "WISENVR_TOKEN_URL in the environment (2019-22 for the 23 wells ending 2018-12; "
        "2023-26 availability for the 16 test wells; L0 + SOURCE.json); expected R-h yield low",
        "09030212: real once-daily record 2019-22; its neighbour offset (p 0.14) is within the "
        "model-well null; admission needs a powered representativeness test not tuned on the "
        "series (sub-daily vendor data or WRA metadata on the 2019 change); at most a "
        "with/without sensitivity in R5",
        "H2's class-based bridge sd is not calibrated per well (3 of 15 pass); a per-well scale "
        "needs a new registration and also concerns H2 v4's model wells",
        "chou-shui-data raw/gw_timeseries_raw.csv and gw_timeseries.csv are fully gap-filled; "
        "never observations",
        "Six apex no-code wells need layer confirmation (rule accuracy about 0.70)",
        "R5 rung gate (plan acceptance part 2) is evaluated by I1 after the freeze",
        "Before any commit of dataprep_staging: Rule 1 scan; r1_vendor_probe.py holds no "
        "credential or host (environment only)",
    ]
    summary = (
        "H3 repair round. Both major findings were confirmed and fixed under a new "
        "registration, H3.repair.v1. First, the hand-set post-look hold-back, applied to "
        "09030212 alone, was replaced by one level-shift screen. H3.step_screen tests every "
        "admitted well against a null of 142-151 continuous model wells, with the same "
        "calendar split and non-sibling model-well neighbours. It flags 0 of the 15 admitted "
        f"wells. 09030212's +0.87 m offset gives z {rh['z']:.2f} and p {rh['p']:.2f}, so the "
        "first round's reason for holding it back is withdrawn. The first round's 2x rule "
        "would flag 19-26 % of continuous model wells at the 2019 splits. 09030212 stays held "
        "back under rule 3, registered before the screen ran: its admission test was "
        "uninformative and its persistence screen was tuned after the series was viewed. The "
        "screen is coarse and can leave shifts of "
        f"{min(sh):.1f}-{max(sh):.1f} m unflagged, so I1 should add a per-segment datum "
        "sensitivity member in R5. Second, the product now records the plan acceptance "
        "honestly. heads/recovered_wells/v3 has the same 15 wells and the same 1,839 rows as "
        "v1, and its validation.json and manifest gate say that 3 of 15 admitted wells pass "
        "H3.d_wells and that the plan criterion fails. The minor fixes are in v3: validators "
        "are told to use flags 0-1 only, flag-2 rows are restricted to partial-month bridges, "
        "the R-h sources are listed as manifest inputs, post-origin dates are redacted, and "
        "the unverifiable config-timing claim is withdrawn. v2 was superseded within the "
        "round for an incomplete redaction, and its manifest reports one expected code "
        "mismatch with the archived bytes. The credentialed vendor route of R-h was not run: "
        "the session's permission system refused loading the stored access, so 'every "
        "existing route checked' is now marked not met. The script is ready for the user. "
        "Status stays partial: plan acceptance part 1 fails for 12 of 15 wells, the R5 gate "
        "is I1 work, 18 of 19 records ending in 2018 are not recovered, and the vendor route "
        "is unchecked. No GPU, 0 API tokens, no writes to hydrophysics/, tests/ or L0, "
        "nothing staged.")
    out = {"wp": "H3", "status": "partial", "summary": summary, "acceptance": acceptance,
           "products": products, "code_paths": code, "key_numbers": key,
           "deviations_from_plan": deviations, "issues_for_downstream": issues,
           "report_path": str(DP / "reports/H3.md"),
           "repair_round": {"date": "2026-10-14", "prereg": "H3.repair.v1",
                            "current_version": "heads/recovered_wells/v3",
                            "first_round_copy": str(R1 / "first_round" / "H3.json")},
           "claude_api_tokens": 0, "gpu": "none"}
    (DP / "reports/H3.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("status",)}), len(acceptance), "criteria;",
          sum(a["passed"] for a in acceptance), "passed")


if __name__ == "__main__":
    main()
