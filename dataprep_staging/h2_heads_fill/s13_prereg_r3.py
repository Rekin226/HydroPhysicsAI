"""Pre-registration of WP H2 repair round 3 (``H2.repair3.v1``; A1 ``prereg3``).

Written after the post-look analyses of this round (``s10_ic_twin.py jan2012``,
``s11_e_fresh.py``, ``s12_edge_ic.py``; all logged) and before any of its fresh data exist:

* ``H2.fresh3.d``  replication of the daily-bridge test on family-d realizations r = 60..69
  (never generated). Its packaging rule, declared here, releases the full-variant daily-bridge
  component (Feb-Mar 2026 bridged) for the statistical layer's history (plan WP H2, Use 4).
* ``H2.ic.twin``   the initial-condition field against the twin's actual stage-3
  construction (backfill + merged proximal), leave one well out on January 2013..2022 (never
  scored by any initial-condition method). Primary candidate ``gp_bf`` (the GP field on the
  twin's information set), chosen on the post-look January 2012 evidence.

No complete fill is re-tested: the family-E pool is exhausted (``e_freshness.json``: 1 of 172
lead-eligible and 2 of 180 trail-eligible well-gaps were never used), so no fresh E stratum
can be formed on the pre-origin matrix, and 2023+ truth cannot gate (U7).

Usage: python -I s13_prereg_r3.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from harness import prereg3  # noqa: E402

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import io_h2  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402

PRIMARY_IC = "gp_bf"
CODE = ["__init__.py", "config.py", "io_h2.py", "methods.py", "kriging.py", "bridge.py",
        "runner.py", "scoring.py", "s4_score.py", "ic_field.py", "r3_common.py",
        "looks_r3.py", "s10_ic_twin.py", "s11_e_fresh.py", "s12_edge_ic.py",
        "s13_prereg_r3.py", "s14_score_r3.py"]
LIB = ["dataprep_staging/harness/gaps.py", "dataprep_staging/harness/gaps_guarded.py",
       "dataprep_staging/harness/coverage3.py", "dataprep_staging/harness/scorers.py",
       "dataprep_staging/harness/prereg3.py", "dataprep_staging/common/guard.py",
       "dataprep_staging/common/paths.py", "hydrophysics/twin/grid.py",
       "hydrophysics/twin/calibrate_flow.py", "hydrophysics/twin/zones.py",
       "hydrophysics/subsidence.py"]
EV = "results/dataprep/reports/H2_evidence"
DATA = ["results/dataprep/heads/L1_clean/v3/MANIFEST.json",
        "results/dataprep/heads/L1_clean/v3/heads_monthly.parquet",
        "results/dataprep/heads/L1_clean/v3/heads_daily.parquet",
        "results/dataprep/heads/L1_clean/v3/station_meta_preorigin.parquet",
        "results/dataprep/validation_units/v3/wells.csv",
        "results/dataprep/validation_units/v3/MANIFEST.json",
        "results/twin/basemap.npz", "results/twin_forward/datum_gate_innov.npz",
        "results/twin_runs/stage3_datum_gate_cwa/stage3_fit_summary.json",
        f"{EV}/design/design_params.json", f"{EV}/design/bridge_form.json",
        f"{EV}/port_check.json", f"{EV}/registered/H2.fresh.d.json",
        f"{EV}/registered/H2.ic.lowo.json", f"{EV}/repair3/ic_jan2012_twin.json",
        f"{EV}/repair3/edge_ic_jan2012.json", f"{EV}/repair3/e_freshness.json",
        "results/dataprep/preregistration/H2.fill_acceptance.v1.json",
        "results/dataprep/preregistration/H2.fill_acceptance.v2.json"]
D_UNIT = f"H2:fresh3_d_r{R3.FRESH_D3.start}-{R3.FRESH_D3.stop - 1}"
IC_UNIT = "H2:ic_twin_jan2013-2022"
IC_SCOPE = "heads:" + ",".join(R3.IC_MONTHS)


def doc(h):
    ids = sorted(h.sids)
    model_ids = sorted(s for s, m in zip(h.sids, h.meta.in_model, strict=True) if m)
    j12 = json.loads((R3.EV_R3 / "ic_jan2012_twin.json").read_text())
    edge = json.loads((R3.EV_R3 / "edge_ic_jan2012.json").read_text())
    tests = [
        {"test_id": R3.TEST_D3, "data_scope": "heads:2012-01..2022-12", "fresh_units": [D_UNIT],
         "external_units": {D_UNIT: {
             "description": "family d (one random daily gap of 5-60 days per eligible well, "
                            "seed 1000*ord('d') + r) realizations r = 60..69 on the "
                            "pre-origin L1 v3 matrix; never generated or scored before this "
                            "registration; truth cells are the C20 months the gaps make "
                            "non-C20", "ids": ids, "window": "2012-01..2022-12"}},
         "metric": "as H2.fresh.d: stratum d (all bridged months pooled), clean truth: (a) "
                   "coverage of the bridge's 90 % interval (pred +- 1.6449 sd, sd of the "
                   "bridged-share class in design_params.json) by rule gate-3 "
                   "(harness.coverage3.coverage_gate, cluster = the well); (b) pooled RMSE "
                   "of the bridge below the twin's backfill, climatology + AR(1), space-time "
                   "kriging, and same-layer IDW on the IDW cells (bridge on the same cells)",
         "threshold": "coverage gate passes AND the bridge's RMSE is strictly lower than "
                      "every defined baseline's (s14_score_r3.py with s4_score."
                      "stratum_verdicts)"},
        {"test_id": R3.TEST_IC, "data_scope": IC_SCOPE, "fresh_units": [IC_UNIT],
         "external_units": {IC_UNIT: {
             "description": "leave one well out at every model well with an L1 flag-0/1 "
                            "monthly value (C20 mean, else the partial-month mean) in January "
                            "of 2013..2022 (about 1,465 well-months). No initial-condition "
                            "method has been scored on these months; the values were truth "
                            "cells of H2's router tests (families S, N, E, H, T, scored "
                            "2026-10-13) but no initial-condition choice was made on them",
             "ids": model_ids, "window": ",".join(R3.IC_MONTHS)}},
         "metric": "pooled LOWO RMSE over every (well, January) cell at the held-out well's "
                   "model cell (the cell value a run starts from) of the primary candidate "
                   "'gp_bf' (ic_field.gp_layered fitted on the twin's information set: the "
                   "other model wells' backfilled values of the month, calibrate_flow."
                   "prepare_series(backfill=True) over 2012-01..2022-12) against 'twin' (the "
                   "stage-3 construction: calibrate_flow._idw_initial_heads on the same "
                   "values, then _merged_proximal_heads with _ic_zone_map('205,182,208'), "
                   "layer_of=None; r3_common.TwinIC); difference interval from a 2000-draw "
                   "site-cluster bootstrap (sites = wells within 100 m; seed 20261013)",
         "threshold": "decision rule: 'confirmed' if RMSE(gp_bf) < RMSE(twin) and the upper "
                      "end of the 90 % site-bootstrap interval of RMSE(gp_bf) - RMSE(twin) "
                      "is below 0; 'point_only' if RMSE(gp_bf) < RMSE(twin) otherwise; "
                      "'not_supported' if RMSE(gp_bf) >= RMSE(twin)"}]
    return {
        "id": R3.PREREG_ID_R3, "wp": C.WP, "kind": "test",
        "title": "WP H2 repair round 3: daily-bridge replication with the full-variant "
                 "packaging rule; initial-condition field against the twin's actual "
                 "construction on fresh months",
        "plan_section": "docs/DATA_REMEDIATION_PLAN_2026-10.md WP H2 (technique step 7, "
                        "validation, Use), 2.4-2.5, 4.4, 4.5",
        "tests": tests,
        "zones": prereg3.not_applicable("point records; the twin's IC zones are part of the "
                                        "baseline construction (205,182,208)"),
        "priors": prereg3.not_applicable("no priors"),
        "burn_in": prereg3.not_applicable("no model run (the twin's 12-month burn-in is "
                                          "kept either way, plan H2 step 7)"),
        "folds": prereg3.not_applicable("artificial-gap realizations (d) and leave one well "
                                        "out (IC); clusters given in each metric"),
        "fresh_units": [D_UNIT, IC_UNIT],
        "design_time_units": [
            "H2:design_{N,S,E,H,T,d}_r0-4", "H2:D_febmar_2013-2022",
            "H2:fresh_{N,S,E,d}_r10-19, H2:fresh_{H,T}_r10-29 (scored under "
            "H2.fill_acceptance.v1)",
            "H2:fresh2_{N,S}_r30-39, H2:fresh2_E_r30-59, H2:fresh2_{H,T}_r30-49 (scored under "
            "H2.fill_acceptance.v2)",
            "H2:ic_lowo_2012-01 (scored under v1; re-scored post-look against the twin's "
            "construction on 2026-10-13T20:5xZ: ic_jan2012_twin.json)",
            "January 2012 leading-edge cells of every H2 E realization (edge_ic_jan2012.json; "
            "its look declared the E-lead window 2012-01..2017-06, but only January 2012 "
            "values were computed)",
            "pilot head_gapfill_pilot N r0-2, S r0-3, E r0-1, H r0-2, T r0-2, D 2013-2025"],
        "ic_primary_candidate": PRIMARY_IC,
        "ic_candidates_reported": ["twin_obs (the twin's construction without backfill)",
                                   "gp_obs (the registered H2.ic.lowo field: observed wells "
                                   "only)", "idw (the H2.ic.lowo baseline: plain per-layer "
                                   "IDW at the well)", "every candidate at the well location "
                                   "as well as at the cell"],
        "thresholds": {R3.TEST_D3: "as H2.fresh.d (gate-3 coverage and RMSE below every "
                                   "defined baseline)",
                       R3.TEST_IC: "decision categories confirmed / point_only / "
                                   "not_supported (see the test)"},
        "gate_scores": [{"name": R3.TEST_D3, "rule": "pass/fail", "weight": 0.5},
                        {"name": R3.TEST_IC, "rule": "decision", "weight": 0.5}],
        "gate_rule": (
            "Two separate registered decisions; equal weights only satisfy the schema. No "
            "complete fill is re-tested or released in this round. (1) H2.fresh3.d: if it "
            "passes, the full-variant daily-bridge component (the L1 flag 0/1 observations "
            "and the flag-2 daily-bridged months of heads/L2_filled_full built with the frozen "
            "bridge of design_params.json, Feb-Mar 2026 bridged) is released as a new "
            "heads/L2_filled_full version whose gate is H2.fresh.d and H2.fresh3.d, for "
            "statistical_layer_history and after_the_fact_analysis only (never calibration, "
            "initial conditions, design or selection of a model scored on 2023-26: U3); its "
            "after-the-fact checks on 2023-25 truth are reported and never gate it (U7). If "
            "it fails, no full-variant component is released and the pre-origin bridge "
            "release is reported as not replicated. (2) H2.ic.twin: if 'confirmed', the "
            "January 2012 gp_bf field (4 layers on the model grid with its native sd; fitted "
            "on the January 2012 L1 flag-0/1 values of the model wells observed then and the "
            "backfilled first observed pre-origin values of the model wells not observed "
            "then; no flag 2-6 fill enters it) is released for initial_conditions in a new "
            "heads/L2_filled_preorigin version whose card states the late-starting-well "
            "evidence of edge_ic_jan2012.json; if 'point_only' or 'not_supported', no "
            "initial-condition field is released and I1 is advised to keep the twin's "
            "construction. In every case that new pre-origin version also carries the "
            "daily-bridge component (flags 0-2) for calibration_target under the v1 packaging "
            "rule (gate H2.fresh.d), with a corrected product card."),
        "primary_model": (
            "daily bridge: form 'shape', bridged-share sds "
            f"{json.loads(R3.PARAMS_V1.read_text())['bridge_sd_by_class']} "
            "(design_params.json, unchanged since H2.fill_acceptance.v1); initial condition: "
            f"primary candidate '{PRIMARY_IC}' (ic_field.gp_layered, unchanged, on the twin's "
            "information set) against the stage-3 construction"),
        "inputs": [str((C.CODE_DIR / f).relative_to(C.REPO)) for f in CODE] + LIB + DATA,
        "reported_not_gated": [
            "H2.fresh3.d: all-truth tables, model-well subset, bridged-share classes (bridge "
            "and router), bridgeable share, both bridge forms, split-conformal multiplier",
            "H2.ic.twin: every candidate at cell and well location, per month, per layer, "
            "proximal / non-proximal, native 90 % coverage of the GP sds, the site-bootstrap "
            "difference of every candidate against twin"],
        "interpretation": (
            "Repair round 3 after an independent verification of H2 (2026-10-14 local). "
            "(a) The registered H2.ic.lowo compared the GP field with plain per-layer IDW, "
            "but the calibrated stage-3 runs start from the backfilled, merged-proximal "
            "construction (stage3_fit_summary.json: no_backfill false, ic_merged_proximal "
            "true, zones 205,182,208). Post-look on January 2012 (121 wells, 58 sites; "
            "cell level): twin "
            f"{j12['cell']['rmse']['twin']:.2f} m, twin_obs "
            f"{j12['cell']['rmse']['twin_obs']:.2f}, gp_obs "
            f"{j12['cell']['rmse']['gp_obs']:.2f}, gp_bf {j12['cell']['rmse']['gp_bf']:.2f}; "
            "the gp_obs - twin site-bootstrap interval includes 0. gp_bf is the primary "
            "candidate because it changes only the interpolator of the twin's construction "
            "(same information set) and had the lower January 2012 RMSE. At late-starting "
            "wells (design-time, edge_ic_jan2012.json, "
            f"{edge['n_wells']} wells) the twin's own construction "
            f"({edge['rmse']['twin_own']:.2f} m) beats gp_bf with the well's own backfill "
            f"({edge['rmse']['gp_bf_own']:.2f}) and H2's leading-edge router fill is best "
            f"({edge['rmse']['router']:.2f}); LOWO at observed wells cannot measure that, so "
            "it is reported with any released field. (b) The pilot's E r0-1 well-gaps were "
            "not excluded from the registered E gates; e_freshness.json recomputes both "
            "rounds with them excluded (v1 E strata pass, v2 E_lead66 fails) and shows the E "
            "pool exhausted, so no complete fill is re-tested. (c) Family d r >= 60 has never "
            "been generated. prereg3 cannot verify freshness of external units (A1 open "
            "issue); the evidence is the seed ranges and the look log. No observation dated "
            "2023-01 or later enters any test, threshold or choice here (U7): the d test and "
            "the backfill use the pre-origin matrix only."),
    }


def main(argv):
    h = io_h2.load("pre", with_daily=False)
    d = doc(h)
    errs = prereg3.validate_prereg(d)
    if errs:
        raise SystemExit("; ".join(errs))
    if argv and argv[0] == "--dry-run":
        print("valid; tests:", [t["test_id"] for t in d["tests"]], "inputs", len(d["inputs"]))
        return
    path, sha = prereg3.write_prereg(d)
    print(path, sha)


if __name__ == "__main__":
    main(sys.argv[1:])
