"""Registration I1.CHX.v1 (harness.prereg3, kind 'test', label design-time spatial holdout;
round 7 rank 13), written BEFORE any hindcast prediction at the external screens, any CH-J fit
and any G0 run.

Primary: R1c's Changhua pumping response direction at the 23 never-calibrated external
Changhua screens (N3F's kept list): two R1c free-run hindcasts 2012-22 (A recorded; B
Changhua at its 2012 monthly pattern, Yunlin recorded), c = A - B at each screen's cell and
layer, e = obs - B over 2013-01..2022-12; e and c residualised on per-screen intercepts and
linear trends AND month-of-sample effects (chx_const.CHX_PRIMARY_FORM 'twoway'), then the
pinned scorer's per-well fixed-effects slope with its 5 km block bootstrap (the
07100111-141 nest one block) and decide(); 'direction' = response_supported or
direction_supported_size_not; the full verdict is reported. The judge's per-well detrended
slope is report-only: on the stand-in its confounded false pass is 1.0 (g0_expected).
Placebo (report only): the Yunlin-only contrast A - C at the same screens. The informative
gate (on the transformed contrast) and G0 (common confounder with heterogeneous loading,
common trend) must pass before any head is opened.

Secondary: CH-J, one pre-declared arm (co-located shallow pumping and all Changhua surface
water, Yunlin frozen at Q0), with no-harm, forward-plausibility and external gates, scored in
the same session. Declared fallback: CH-J needs an opt-in calibrate_flow hook that no builder
may add to an existing twin module; unless its engineering record (hook sha256, rank-8
forcing version, passed freeze pre-gate, panel producer) is frozen within 21 days of this
registration and names the scored CH-J panel, the primary is scored alone and CH-J's gates
are report-only (score_chx.chj_provenance enforces it).

``--validate`` writes nothing: it prints the document's canonical sha256, prereg3's
validation, the dry-run blocking check against the registry, the readiness list and this
script's sha256. ``--write`` (the lead only) registers it once, with ``--expect-self-sha``.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY -I dataprep_staging/ch_x/prereg_chx.py --validate --quiet
    $PY -I dataprep_staging/ch_x/prereg_chx.py --write --quiet --expect-self-sha <sha256>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chj_forcing as CJ  # noqa: E402
import chx_const as K  # noqa: E402
import chx_io as IO  # noqa: E402
import score_chx as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
THIS = f"{K.X}/prereg_chx.py"
TESTS_FILE = f"{K.X}/tests/test_chx.py"
RUNNER = f"{K.X}/run_chx.sh"
RUNNER_LINES = (f"ID={K.CHX_ID}", f"WT_HEAD={K.WT_HEAD}", f'WT="$HOME/{K.WT_NAME}"',
                "HC=results/twin_runs/chx/hindcast", "G0D=results/twin_runs/chx/g0")
N1_ID = "N1.cpu_response.v1"
N3F_ID = "N3F.ext_level.prospective_2026-11.v1"
FAILED_RUNGS = ("I1.R1.changhua_census.v1", "I1.R1b.census_canal.v1",
                "I1.R1c.county_zones.v1", "I1.R1d_dom.v1", "I1.R1d_sw.v1", "I1.R1e_cn.v1",
                "I1.R1e_pw.v1", "I1.Rstar.v1", "I1.R4p.v1", "I1.X5R1.v1",
                "I1.X5R1.v2_postfailure", "I1.X5CW.v1")


def frozen_code() -> dict:
    return {p: IO.sha_or_missing(CODE / p) for p in S.FROZEN + (THIS, TESTS_FILE)}


def screens() -> dict:
    ch = IO.changhua_ext()
    yl = IO.county_rows(IO.ext_wells(), "yunlin")
    return {"ids": list(ch["station_id"]), "rows_sha256": IO.rows_sha(ch),
            "yunlin_ids": list(yl["station_id"]), "yunlin_rows_sha256": IO.rows_sha(yl),
            "wells_csv": K.EXT_WELLS, "wells_csv_sha256": IO.sha256(IO.rp(K.EXT_WELLS)),
            "columns_read": list(K.EXT_COLUMNS),
            "g0_mask_column": K.EXT_COUNT_COLUMN,
            "rule": "N3F's kept external wells (active forward-grid cell, layer code 1-4, >= 24 "
                    "pre-origin months, residual sd from >= 12 months; applied by "
                    "export_baselines.py ext to pre-origin rows only) with county Changhua; "
                    "each screen at its own cell and layer (layer_code - 1)"}


def recipe() -> dict:
    return {
        "theta": K.R1C_THETA, "theta_sha256": IO.sha_or_missing(IO.rp(K.R1C_THETA)),
        "worktree": f"$HOME/{K.WT_NAME} (clean, detached)", "worktree_head": K.WT_HEAD,
        "worktree_reason": "N2C's registered code commit; R1c's own theta records 709f660 but "
                           "was fitted with the R1c hooks uncommitted on top of it (they "
                           "landed in 8d51b46), so the clean commit at which R1c is known to "
                           "reproduce (X5.changhua_lever_model.v1 precheck, X5.n2c_reading.v1 "
                           "reproduction) is used",
        "hindcasts": f"bash {RUNNER} hindcast -> python {K.X}/hindcast.py chx --out "
                     "results/twin_runs/chx/hindcast --check-twinmodel (CPU or GPU; heads "
                     "only, no column, no assimilation)",
        "arms": {"A": "recorded energy in both counties (R1c's own forcing, model_census v1)",
                 "B": f"ZR: every Changhua cell (R1c's county map, zone_county_sha256 "
                      f"verified) at its own {K.PATTERN_YEAR} monthly energy pattern from "
                      f"{K.PATTERN_YEAR + 1}-01 on, other cells recorded",
                 "C": f"placebo: every Yunlin cell (centroid in {K.YUNLIN_ZH} by "
                      f"zone_county.county_of_points on {K.COUNTY_GPKG}, and not in R1c's "
                      f"Changhua map) at its {K.PATTERN_YEAR} pattern, every other cell "
                      "(Changhua and the in-grid Chiayi/Nantou cells) recorded"},
        "engineering_checks": ["export cell == R1c grid cell for every screen",
                               "at least one Yunlin cell; cell counts per county recorded",
                               "arm A == policy_lib.TwinModel(theta, column).heads() to 1e-9 m",
                               "finite heads", "panel written once; sha256 in chx_panel.json",
                               "panel code_head == worktree_head; theta sha256 as declared"],
        "g0": f"bash {RUNNER} g0 -> python {K.X}/g0_chx.py --panel <panel> --out "
              "results/twin_runs/chx/g0 (CPU, model output and declared parameters only)",
        "score": f"bash {RUNNER} score (once)",
        "frozen_code_check": "the scorer refuses unless every frozen_code sha256 matches"}


#: G0 on the stand-in panel (g0_chx.synthetic_panel: c = -U(0.8, 2.5) x lagged a_CH, RMS of
#: the well-demeaned contrast 0.16 m at x1), 200 replicates per cell, B 400, produced by
#: ``g0_chx.py --smoke --reps 200 --form F --scale S`` (no model output, no observation)
G0_EXPECTED = {
    "twoway_x1": "transformed contrast RMS 0.038 m < 0.05: not_informative (G0 not run)",
    "twoway_x2": "informative (0.076 m); power 0.44, false pass max 0.07: not decidable",
    "twoway_x3": "informative (0.113 m); power 0.625, false pass max 0.07: decidable; "
                 "report-only: loading corr 0.5 with the contrast amplitude gives false pass "
                 "0.715 (power 1.0); trend reversed power 0.66; no trend false 0.055",
    "detrended_x1": "power 1.0, confounded false pass P(k=0, rho=-0.83) 0.99 (median slope "
                    "1.49); P(k=0, rho=0) 0.125: not decidable",
    "detrended_x3": "power 1.0, confounded false pass 0.99 (median slope 0.50): not decidable "
                    "at any contrast size (a confounder perfectly correlated over time with "
                    "a_CH biases the slope and the block bootstrap cannot see it)",
    "reading": "with the frozen 'twoway' form the outcome turns on how much R1c's contrast "
               "differs ACROSS the 23 screens: below about 0.05 m RMS after the transform it "
               "is 'not_informative' (no head read); about 0.1 m RMS is needed for 'decidable'. "
               "The design cannot exclude a confounder whose spatial loading tracks R1c's "
               "pumping response (report-only cell: false pass about 0.7 at a loading "
               "correlation of 0.5); a direction pass is conditional on that."}


def g0_spec() -> dict:
    return {"world": "e = k c + block AR(1) + well AR(1) + kappa g_t (1 + z_i) + s h_t on "
                     "R1c's own contrast c at the 23 screens (g0_chx.py docstring)",
            "form": K.CHX_PRIMARY_FORM,
            "params_source": K.G0_PARAMS_FROM, "grid": {"k": list(K.G0_K),
                                                        "rho": list(K.G0_RHO)},
            "heterogeneous_loading": {"mult": K.G0_HET_MULT, "corr": K.G0_HET_CORR,
                                      "z": "N(0, 1) per screen per replicate"},
            "report_only_cells": list(K.G0_REPORT_CELLS),
            "replicates_per_cell": K.G0_REPLICATES, "seed": K.G0_SEED,
            "bootstrap": "the pinned scorer's (B 2000, seed 20261101)",
            "common_trend": {"form": "hinge 0 before the break, linear to amp at 2022-12",
                             "break": K.TREND_BREAK, "amp_m": K.TREND_AMP_M,
                             "sign": "the one biasing the per-well detrended slope upward "
                                     "(absorbed exactly by the twoway form)"},
            "order": "informative gate on the transformed contrast first (fails -> "
                     "'not_informative', no cell run, no head read), then exact plumbing, "
                     "then the grid",
            "expected_on_stand_in": G0_EXPECTED,
            "mask": "each screen observed on its last min(n_fit_months, 120) window months "
                    "(N3F's logged count; an approximation, disclosed)",
            "gate": {"power": f"P(direction | k = 1, rho = -0.83) >= {K.G0_POWER_MIN}",
                     "informative": "R1c's transformed contrast passes the pinned gate",
                     "false_pass": f"max over rho in {list(K.G0_RHO)} of P(direction | k = 0)"
                                   f" <= {K.G0_FALSE_MAX}",
                     "else": "'not decidable': stop, no head opened, report written"},
            "plumbing": "no noise: the detrended slope equals k to 1e-9; failure = "
                        "engineering (code fix only)",
            "power_caveat": "noise parameters come from the 74 calibration screens; if the "
                            "externals are noisier G0 overstates power"}


def chj_spec() -> dict:
    return {
        "arm": "ONE pre-declared arm, no selection: chj_forcing.py docstring",
        "physics": {"changhua_pump_shares": list(K.CHJ_SHARES_CHANGHUA),
                    "rest_pump_placement": "exactly Q0's own: sigmoid(pump_split_logit) in "
                                           "layer 0, the rest in Q0's pump layer 1 "
                                           "(chj_forcing.q0_rest_fractions from Q0's theta; "
                                           "pump_split_logit pinned)",
                    "sw": "W1 v4 deliveries + x5_changhua_water preorigin v1 non-Jiji "
                          "(service placement on canal_x4 shares) + W1 percolation, both "
                          "ZERO outside R1c's Changhua cells (Q0 has no sw; Yunlin unchanged), "
                          "layer 0, --sw-county-split, Changhua log_sw_scale entries free "
                          "(fractions in [0.01, 1]), rest-of-fan entries pinned (zero field)",
                    "freeze": CJ.FREE_RULE,
                    "county_copy_sd": K.CHJ_COUNTY_COPY_SD, "seeds": list(K.CHJ_SEEDS),
                    "columns": "both county columns refitted on CH-J's own heads; Changhua "
                               f"Ske/Skv <= {K.CHJ_COLUMN_CAP}",
                    "hypothesis": "co-location of canal/paddy recharge with shallow pumping "
                                  "(not the pump layer)"},
        "code": {"chj_forcing.py": "frozen here", "calibrate_flow_hook":
                 "NOT available at registration (CALIBRATE_FLOW_PATCH in chj_forcing.py is "
                 "the spec); it must implement that spec exactly, be committed at a frozen "
                 "commit and pass the engineering pre-gate; its engineering record "
                 "(results/twin_runs/chx/chj/chj_engineering.json, keys "
                 f"{list(S.ENGINEERING_KEYS)}) is written before any CH-J fit, and the "
                 "panel producer writes chj_panel.json (keys "
                 f"{list(S.PANEL_SUMMARY_KEYS)}) beside chj_panel.npz",
                 "gating_condition": "score_chx.chj_provenance: record present, frozen "
                                     f"within {K.CHJ_ENGINEERING_DEADLINE_DAYS} days of this "
                                     "registration, pre-gate passed to "
                                     f"{K.CHJ_FREEZE_REPRO_TOL_M} m; chj_panel.json names the "
                                     "record, its producer, the panel and metrics sha256 and "
                                     "postdates the freeze; seeds exactly "
                                     f"{list(K.CHJ_SEEDS)}; otherwise report-only"},
        "forcing_version": "Changhua census forcing as fixed by round-7 rank 8 (the step-0 "
                           "census audit of the bimonthly even split and duty-cap retention); "
                           "its product version and sha256 are logged with the hook's sha256 "
                           "before any CH-J fit; the primary keeps R1c's own forcing "
                           "(model_census v1, bound by R1c's theta)",
        "engineering_pregate": "with Changhua copies pinned, the freeze hook reproduces Q0 to "
                               f"{K.CHJ_FREEZE_REPRO_TOL_M} m on synthetic heads",
        "registered_outcome": f"Changhua sw fraction ending at {K.CHJ_SW_FRACTION_BOUND} "
                              "(either seed): 'co-located recharge rejected'",
        "gates": {
            "noharm": "Yunlin k-fold and Yunlin county_corr >= Q0 - 0.01; out-of-fold "
                      "leveling >= Q0 - 0.02; Changhua leveling not below Q0's (spent units, "
                      "design-time)",
            "forward": "model-only 2023-25: county-mean subsidence >= 0 in both counties; "
                       "Changhua rising share <= Q0's",
            "externals": f"median CH seasonal amplitude ratio (model/obs, 2-month smoothed "
                         f"climatology of detrended series) in {list(K.CHJ_AMP_BAND)}; CH/YL "
                         f"ratio of medians in the same band; county-mean correlation >= Q0's "
                         f"- {K.CHJ_NOHARM['county_mean_corr_ext']}; direction by the "
                         "primary estimator (same form) on CH-J's own contrast",
            "conjunction": "both seeds pass every gate and the sw fraction is off its bound"},
        "report_only": ["the F reading (X5.changhua_lever_model.v1's definition)",
                        "fitted Changhua confined-layer S against x5_storage_priors v1b Ss "
                        "(product values, no pinning; v1b has no allowed_uses)"],
        "fallback": f"if the calibrate_flow hook is not frozen within "
                    f"{K.CHJ_ENGINEERING_DEADLINE_DAYS} days of registration, the primary "
                    "is scored alone and CH-J's external gates are report-only whenever CH-J "
                    "runs (they can never gate after the primary's session)",
        "predictions_hashed": "every CH-J arm's heads at the externals go into chj_panel.npz "
                              "whose sha256 is recorded before the single scoring session"}


def disclosures() -> dict:
    return {
        "external_screens": [
            f"{N1_ID} scored R0 event slopes (E2015, E2020, E2021) at N1:ext_wells, which hold "
            "these 23 screens, over 2013-22",
            "N3F's export (export_baselines.py ext) computed datum, residual sd and "
            f"climatology from their pre-origin rows ({N3F_ID})",
            "the H3 recovered-wells inventory and step screens touched 07170211/221 and "
            "07260241",
            "X5EV loaded 2010-11 hourly series of 07100111, 07170211, 07240212 and 07250111 "
            "(a disjoint window)",
            "round-7 planners listed the 23 ids and the N3F kept_by_county count (metadata)",
            "published I1_changhua_local summary statistics (Changhua amplitude 0.35 vs 1.25 "
            "m; R1c ratio 1.13) were read by earlier sessions (74 calibration wells, not "
            "these screens)"],
        "failed_changhua_rungs": list(FAILED_RUNGS),
        "gate_redesign": "the amplitude-free gate structure (direction, amplitude bands, "
                         "ratios, no-harm) was adopted on 2026-10-18/19 after the "
                         "amplitude-sensitive RMSE gates rewarded shrinkage (RC2); it was "
                         "never applied to the earlier rungs",
        "model_side": "no hindcast with the B or C pattern arm exists for R1c; R1c's heads at "
                      "the external screens have never been compared with observations",
        "design_change_before_registration": "the round-7 review found the judge's per-well "
                                             "detrended slope cannot pass G0 (confounded false "
                                             "pass 0.99-1.0 on the stand-in); the primary form "
                                             "was changed to 'twoway' before registration on "
                                             "stand-in numbers only (G0_EXPECTED); no model "
                                             "output at the externals and no head was seen",
        "drafter": "the drafter (round-7 builder) opened no head, no N3F prediction or "
                   "baseline value and no sealed file; it read MANIFESTs, theta meta keys, "
                   "registration metadata and code"}


def outcomes() -> dict:
    return {
        "direction": {"E_row": "new: \"R1c's Changhua pumping response has the right "
                               "direction at 23 never-calibrated screens (design-time)\"",
                      "C20_C21": "gain the clause"},
        "not_supported": {"E_row": "new: 'not supported' (design-time)",
                          "C20_C21": "gain the narrowing clause"},
        "not_informative": {"E_row": "recorded 'not informative' (no head read)",
                            "C20_C21": "unchanged"},
        "not_decidable": {"E_row": "recorded 'not decidable' (G0)", "C20_C21": "unchanged"},
        "chj_pass": "a second E-row: 'a census-forced Changhua model with Yunlin frozen and "
                    "co-located shallow pumping plus canal recharge passes amplitude-free "
                    "gates (design-time)'",
        "chj_fail": "the C20/C21 narrowing retires the co-location hypothesis "
                    "('co-located recharge rejected' when the fraction hit its bound)",
        "unchanged_whatever_the_outcome": ["C13a", "E20 stays D", "the F < 0.5 cap",
                                           "N2C's before_scoring and never clauses",
                                           "no Changhua lever displayed before N2C"],
        "page_ruling": "the new E-rows and the C20/C21 wording need the user's approval "
                       "(requested in round 7; not in the sixth set, user_rulings_2026-11-01b"
                       "); until then outcomes go to the evidence record and the ledger only"}


def doc() -> dict:
    sc = screens()
    fz = frozen_code()
    ext_ch = {"description": "the 23 never-calibrated external Changhua screens (N3F kept, "
                             "county Changhua); heads 2013-01..2022-12; design-time spatial "
                             "holdout (prior looks disclosed), never fresh",
              "ids": sc["ids"], "window": "2013-01..2022-12"}
    ext_yl = {"description": "N3F's kept external Yunlin screens; heads 2013-01..2022-12; read "
                             "only for CH-J's CH/YL amplitude-ratio gate; design-time (N1 "
                             "T1_ext), NOT declared fresh",
              "ids": sc["yunlin_ids"], "window": "2013-01..2022-12"}
    test_primary = {
        "test_id": K.CHX_TEST, "data_scope": K.CHX_SCOPE, "fresh_units": [K.CHX_UNIT],
        "external_units": {K.CHX_UNIT: ext_ch},
        "label": "design-time spatial holdout",
        "what": "Does R1c's Changhua-only pumping contrast have the right sign at screens no "
                "model was fitted or selected on?",
        "estimand": "beta = pinned per-well FE slope (bootstrap_scorer.beta_response) of e~ "
                    "on c~, e = obs - B, c = A - B, both residualised on per-screen "
                    "intercepts and linear trends and month-of-sample effects (exact least "
                    f"squares, chx_lib.fe_basis form {K.CHX_PRIMARY_FORM!r}); rows flags 0-1 "
                    f"with >= {K.MIN_WELL_MONTHS} scored window months per screen; identified "
                    "from differences across screens in R1c's contrast",
        "interval": "90 %, 5 km block bootstrap (bootstrap_scorer: B 2000, seed 20261101), "
                    f"the {'/'.join(K.NEST)} nest one block '{K.NEST_BLOCK}'",
        "informative_gate": "bootstrap_scorer.informative_gate on the TRANSFORMED contrast c~ "
                            "at all 23 screens x 120 window months, before any head (RMS of "
                            "the well-demeaned c~ >= 0.05 m, >= 5 blocks); fails -> "
                            "'not_informative', no head read",
        "decision_rule": "verdict = bootstrap_scorer.decide(informative, beta, lo, hi, "
                         "RMSE(PR), RMSE(ZR)) with RMSE(ZR) = equal-well RMS of e~ and "
                         "RMSE(PR) of e~ - c~ (the same transformed rows); DIRECTION iff "
                         f"verdict in {list(K.DIRECTION_VERDICTS)}; the full verdict is "
                         "reported",
        "precondition": "informative, then G0 decidable (g0_spec); otherwise "
                        "'not_informative' / 'not decidable' with no head read",
        "report_only": ["placebo: the same estimator on c_YL = A - C with e = obs - C",
                        "the per-well detrended slope (the judge's form; confounded on the "
                        "stand-in)", "the demeaned slope, N2C's own primary form",
                        "level component", "interval_below_zero_with_positive_beta flag"],
        "never_substituted": "no report-only variant replaces the primary"}
    test_chj = {
        "test_id": K.CHJ_TEST, "data_scope": K.CHX_SCOPE,
        "fresh_units": [K.CHX_UNIT],
        "external_units": {K.CHX_UNIT: ext_ch, K.CHX_UNIT_YL: ext_yl},
        "label": "design-time spatial holdout (secondary; report-only under the fallback)",
        "what": "Does one co-located CH-J arm pass amplitude-free gates at the externals?",
        "gates": chj_spec()["gates"]}
    inputs = ([{"path": p, "sha256": h} for p, h in fz.items() if h != "missing"]
              + [K.R1C_THETA, K.R1C_COLUMN, K.Q0_THETA, K.COUNTY_MAP, K.COUNTY_GPKG,
                 K.EXT_WELLS,
                 f"results/dataprep/preregistration/{N1_ID}.json",
                 f"results/dataprep/preregistration/{N3F_ID}.json", K.RULINGS])
    return {
        "id": K.CHX_ID, "wp": K.WP, "kind": "test",
        "title": "I1.CHX.v1: R1c's Changhua pumping response direction at the 23 "
                 "never-calibrated external Changhua screens (primary), with one co-located "
                 "shallow-offset companion CH-J, Yunlin frozen (secondary); design-time",
        "priors": "Step C's Changhua support belongs to the masked refit and does not isolate "
                  "Changhua pumping (R0, without Changhua pumping, also passed it). No test "
                  "has scored R1c itself at screens it was never fitted or selected on, nor "
                  "isolated its Changhua-only contrast. Expected: on the stand-in G0 is "
                  "decidable only when R1c's contrast differs across screens by about 0.1 m "
                  "RMS (G0_EXPECTED), so 'not_informative' or 'not decidable' are likely; "
                  "direction about 0.25 overall; CH-J conjunction about 0.10-0.15 if it runs.",
        "tests": [test_primary, test_chj],
        "fresh_units": [K.CHX_UNIT],
        "zones": "external screens at their own forward-grid cell and twin layer; Changhua "
                 "cells = R1c's county map (zone_county_map v2, sha verified)",
        "burn_in": f"free run from R1c's own month-0 field (2012-01); {K.PATTERN_YEAR} is the "
                   "pattern year (not scored); window 2013-01..2022-12",
        "folds": NA("spatial holdout; no folds (CH-J's no-harm gates use Q0's seed folds)"),
        "gate_scores": NA("declared gates; no weighted score"),
        "primary_model": "R1c (stage3_datum_gate_ccz, main parameter set), the N2C primary",
        "screens": sc, "recipe": recipe(), "g0_spec": g0_spec(), "chj": chj_spec(),
        "scorer": {"path": K.SCORER, "sha256": K.SCORER_SHA, "pinned": True},
        "thresholds": {"min_well_months": K.MIN_WELL_MONTHS,
                       "direction_verdicts": list(K.DIRECTION_VERDICTS),
                       "g0_power_min": K.G0_POWER_MIN, "g0_false_max": K.G0_FALSE_MAX,
                       "chj_amp_band": list(K.CHJ_AMP_BAND), "chj_noharm": K.CHJ_NOHARM,
                       "bounds": "inclusive, unrounded"},
        "frozen_code": fz,
        "prior_looks_disclosed": disclosures(),
        "evidence_and_ledger_mapping": outcomes(),
        "attempt_policy": {
            "scored_runs": "one scoring session; report.json written once (mode 'x')",
            "engineering_retries": "free before any head is read (crash, OOM, missing file, a "
                                   "refused step 0-2); each logged in attempts.jsonl; a change "
                                   "to frozen code needs a v2 declaration disclosing them",
            "pre_declared_alternatives": [], "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; cannot promote",
            "externals_after_scoring": "spent for selection",
            "thresholds_frozen": True, "report": "every attempt, failures included"},
        "u7": "every input is pre-origin (2012-22); L1_clean v3 read with window_end 2022-12",
        "interpretation": "A direction pass says screens where R1c's Changhua-only contrast "
                          "is larger also show the larger observed response, at screens R1c "
                          "never saw (design-time); a confounder common to every screen and "
                          "the common trend are absorbed, but one whose spatial loading tracks "
                          "R1c's pumping response is not excluded. It says nothing about the "
                          "sustained lever (F = 0.49) or its size. CH-J tests one physical "
                          "explanation of Changhua's weak response.",
        "inputs": inputs,
        "no_prior_look_statement": "no R1c B/C-arm hindcast and no R1c prediction at the 23 "
                                   "screens exists at registration; no CH-J fit exists",
        "written_by": THIS,
    }


def readiness(d: dict) -> list[str]:
    out = []
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not IO.rp(p).exists() and not (CODE / p).exists():
            out.append(f"input {p} is missing")
    if IO.sha_or_missing(CODE / K.SCORER) != K.SCORER_SHA:
        out.append("the pinned scorer's sha256 differs from f8990d61")
    sh = (CODE / RUNNER).read_text() if (CODE / RUNNER).exists() else ""
    lines = set(sh.splitlines())
    out += [f"{RUNNER} lacks {s!r}" for s in RUNNER_LINES if s not in lines]
    if "/" + "home/" in sh:
        out.append(f"{RUNNER} hardcodes a home path")
    if IO.rp(f"results/dataprep/preregistration/{K.CHX_ID}.json").exists():
        out.append(f"{K.CHX_ID} is already registered")
    for p in ("results/twin_runs/chx/hindcast/chx_panel.npz", "results/twin_runs/chx/g0/g0.json",
              S.REPORT, "results/twin_runs/chx/chj/chj_engineering.json"):
        if IO.rp(p).exists():
            out.append(f"{p} exists before the registration")
    wt = Path.home() / K.WT_NAME
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if head != K.WT_HEAD:
            out.append(f"worktree $HOME/{K.WT_NAME} is at {head}, not {K.WT_HEAD}")
    except (OSError, subprocess.CalledProcessError):
        out.append(f"worktree $HOME/{K.WT_NAME} missing (bash {RUNNER} worktree)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", IO.canonical_sha(d))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    idx = prereg3.membership_index()
    for t in d["tests"]:
        for u, v in t["external_units"].items():
            m = prereg3._declared_members(v)
            if m is not None and not idx.known(u):
                idx.members[idx.canon(u)] = m
    block = prereg3.blocking_looks(d["tests"], None, idx=idx,
                                   test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    self_sha = IO.sha256(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
    bad = bool(errs or block or pending)
    if a.validate:
        print("valid; not written" if not bad else "NOT READY: fix the problems above")
        return 1 if bad else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --write needs --expect-self-sha equal to this script's sha256")
        return 1
    if bad:
        print("REFUSED: not ready to register")
        return 1
    path, h = prereg3.write_prereg(d)
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
