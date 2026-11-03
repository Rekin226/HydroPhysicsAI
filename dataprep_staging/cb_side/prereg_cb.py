"""Draft registration of CB.coverage_side.v1 (harness.prereg3 schema, kind 'declaration'),
built in code because dataprep_staging/ ignores *.json. Nothing here writes to the registry:
the lead reviews the draft, picks the seed and registers it with prereg3.write_prereg after
the code is merged (R7.commit_first), before 2026-12-15 and before any window product is
read (hard limit 2028-01-15).

  prereg_cb.py --seed N [--verify-npz]   prints the draft JSON, its sha256, validate_prereg
                                         and readiness (validate only; never writes)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cb_side as C  # noqa: E402  (puts FV v2, FV v1, E02c, U1.HB and staging on sys.path)

O, V2, S, U = C.O, C.V2, C.S, C.U  # noqa: E741
MISSING = "MISSING"
REGISTER_BY = "2026-12-15T23:59:59+00:00"
HARD_LIMIT = "2028-01-15T00:00:00+00:00"
RULINGS = "results/dataprep/rulings/user_rulings_2026-11-03.json"
RULINGS_SHA = "ec610b692835d955141e9261e6d6046ee1984f39891bcf1c2e77591aad1c69eb"
RULING_ID = "R8.8"
E02C_DECL = f"results/dataprep/preregistration/{S.ID}.json"
E02C_DECL_SHA = "cc5b970f4bd009cd1f84ae3c702e7d057bd6325f724c38ecdd710ab2a62dd468"
FV_EXT_DECL = "results/dataprep/preregistration/FV.ext.v1.json"
FV_EXT_DECL_SHA = "5010831534996cd4a68cc6613731c9e1b97851d71f1925dd7fe2e5a5abab549e"
U1HB_RESULT = f"results/dataprep/reports/{C.U1HB_ID}/result.json"
U1HB_RESULT_SHA = "7dd2bcdb416f650f78a6d1c92ec42a2d930cd1812ba3fcb5bf80ed686447ce3c"
SEAL_LOOK = "R8:fe_encompass_fixer_reads_disclosure"
BUILDER_LOOKS = (   # (unit, purpose) as the lead logged them, 2026-11-03
    ("R8:cb_coverage_side_builder_reads_disclosure", "design_time"),     # 09:23:48Z
    ("R8:cb_coverage_side_builder_reads_supplement", "disclosure"))      # 09:31:21Z
SEALED_STDOUT = "results/twin_runs/fv2/sealed/"
KILLED_ROUTE = ("Mapping the E02c Changhua coverage side to C20/C21 or 'Changhua pumping "
                "matters'. N2C alone decides response class. (round-8 plan, killed routes)")
BUILDER = (
    "The CB builder (round 8, 2026-11-03) read code (osse_fv2.py, osse_fv.py, score_e02c.py, "
    "power_u1hb.py header, harness.prereg/2/3, fe_encompass prereg_fe.py / decide_fe.py / "
    "fe_lib.py, tests/test_fe_encompass.py header) and these non-code files: "
    "docs/TWIN_ROUND8_PLAN.local.md (entry #10, the audit finding '[high] bands + Changhua "
    "(FV.decidability.v2 reading 1)', the killed-routes list, execution notes, the R8.x "
    "ruling list and the F2 disclosure line); results/twin_runs/fv2/attempts.jsonl (all rows "
    "printed: events, rc, truth, batch, npz and sealed-stdout sha256s; the sealed stdout files "
    "were never opened); DR.relative_drift.v1's registration (printed in full, including 30 of "
    "its unit's well ids) and report.json (printed: tau^2, ICCs, grouping, leads, page "
    "numbers; pre-origin derived statistics, design_time); the header and the first two rows "
    "of wells_e02d.csv (two Changhua wells: layer, zone, cell, n_fit, datum_m, "
    "sd_abs_m, groups, sd_fit_rel_m, t_centre_year, phi; pre-origin derived values); "
    "user_rulings_2026-11-03.json (in full); the looks.jsonl row of "
    f"{SEAL_LOOK}; FV v2's and U1.HB.power.v1's registrations (key lists and the fields "
    "tests, fresh_units, external_units, zones, folds, burn_in, gate_scores, priors, "
    "attempt_policy, sealed_until_scoring, inputs_frozen, v1_forwards; U1.HB's code_sha256 "
    "and design head); FV v2's report_worlds.json (key list, status, seed, k_truth, "
    "P.coverage.<county>.by_truth for both counties, truth_worlds.Q0 key list and its "
    "coverage_supported rates); hydrophysics/twin/claims_v1.json rows C14a and C14a-r "
    "(including C14a's round-8 notes with FV v2's per-truth and U1.HB numbers); directory "
    "listings of results/, the FV / DR / U1.HB / E02c / FV.ext report directories, "
    "results/twin_forward (fv2 names) and the preregistration directory; sha256 only of the "
    "20 FV v2 batch npz, fv_zr0.npz (and its size), the DR, U1.HB, FV v2, FV.ext and E02c "
    "registrations, DR's report and wells table, U1.HB's result.json, report_worlds.json and "
    "the attempts log. It opened no npz array, read no head or observation value, computed no "
    "statistic on any real input or real seed, and ran CB's engine only on SYNTHETIC inputs "
    "(osse_fv2.build_synthetic; smoke seed) and E02c's coverage code on a synthetic world for "
    "timing. Logged by the lead as two looks: "
    f"{BUILDER_LOOKS[0][0]} (purpose '{BUILDER_LOOKS[0][1]}', 2026-11-03T09:23:48Z: the reads "
    f"above) and {BUILDER_LOOKS[1][0]} (purpose '{BUILDER_LOOKS[1][1]}', "
    "2026-11-03T09:31:21Z: the supplement covering the printed truth_worlds.Q0 "
    "coverage_supported rates of report_worlds.json and the claims_v1.json C14a / C14a-r "
    "rows). The reviewer's fixes (2026-11-03) read no further non-code file beyond these "
    "looks' rows and e02dlib.py / stat_v2.py seed constants (code).")


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _sha(p) -> str:
    q = O.rp(p)
    return O.sha256(q) if q.is_file() else MISSING


def _bind(path: str, expected: str) -> dict:
    return {"path": path, "sha256": _sha(path), "expected_sha256": expected}


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def ruling_binding() -> dict:
    q = O.rp(RULINGS)
    r = json.loads(q.read_text(encoding="utf-8")) if q.is_file() else {}
    hit = [x for x in r.get("rulings", []) if x.get("id") == RULING_ID]
    return {**_bind(RULINGS, RULINGS_SHA), "id": RULING_ID,
            "decision": hit[0].get("decision") if hit else MISSING}


def page_mapping() -> dict:
    w = C.WORDING
    return {
        "ruling": f"{RULING_ID} (user_rulings_2026-11-03.json sha {RULINGS_SHA[:8]}) named "
                  "neither CB nor C14a and kept the FV v2 per-truth split OFF the page "
                  "(ledger / README / MODEL_CARD only; R7.fv_v2_page wording unchanged). CB's "
                  "reading is itself a per-family split",
        "page": C.PAGE_RULE,
        "claim": "C14a, Changhua only",
        "status": "no status change in any branch (C14a stays as E02c decides it)",
        "on_pass": {"where": "ledger / README / MODEL_CARD only",
                    "ledger": w["side_separates"]["ledger"],
                    "placeholders": "a = min P(over) page family (gated arm), b = max P(over) "
                                    "Changhua recipes, c = page-family bias-cancelled-by-shock "
                                    "max; all from report.json; the datum-centre suffix is "
                                    "appended when that report-only arm's gate-A minimum "
                                    "misses 0.8",
                    "datum_centre_suffix": C.DATUM_CENTRE_SUFFIX},
        "on_fail": {"where": "ledger / README / MODEL_CARD only",
                    "ledger": w["side_does_not_separate"]["ledger"],
                    "datum_centre_suffix": C.DATUM_CENTRE_SUFFIX},
        "on_not_evaluable": {"where": "ledger / README / MODEL_CARD only",
                             "ledger": w["not_evaluable"]["ledger"]},
        "never": ["any page line or page status change from CB (a page line needs a new "
                  "user ruling)",
                  "any C20/C21 mapping or 'Changhua pumping matters' (" + KILLED_ROUTE + ")",
                  "a reading of a 2028 pass or an under-coverage fail (one-sided)",
                  "Yunlin (not computed; C14a Changhua only)", "the word 'calibrated'"],
        "lead_confirms": "LEAD decision (2026-11-03, adversarial review B1): CB gets no page "
                         "line in any branch; every branch is ledger / README / MODEL_CARD "
                         "only; a page line needs a new user ruling"}


def draft_prereg(seed: int) -> dict:
    """CB's registration draft for ``seed`` (the lead's choice; refused seeds raise)."""
    seed = C.check_seed(seed)
    batches = C.fv2_batches_from_attempts(C.FV2_ATTEMPTS)
    frozen = {p: _sha(p) for p in C.FROZEN}
    inputs = sorted({*C.FROZEN, C.E02_PRED, C.E02_PROTOCOL, C.WELLS_CSV, C.NOISE_CSV,
                     C.MASK_CSV, C.ZR0_NPZ, C.FV2_DECL, C.FV2_REPORT, C.DR_DECL, C.DR_REPORT,
                     C.DR_WELLS, C.U1HB_DECL, RULINGS, E02C_DECL, FV_EXT_DECL})
    doc = {
        "id": C.ID, "wp": C.WP, "kind": "declaration",
        "title": "CB: on which side does an E02c Changhua coverage fail fall, by truth family, "
                 "once DR's per-well drift and U1.HB.power's year / month shocks are overlaid "
                 "on FV v2's 1000 full-forward worlds (model-only, CPU, fresh seed)",
        "label": "new estimand (the side distribution of E02c's Changhua coverage outcome); "
                 "not FV v3; FV v2's readings and wording are unchanged",
        "why": "Audit finding '[high] bands + Changhua (FV.decidability.v2 reading 1)': "
               "Changhua's pooled P = 0.50 hides a split by truth family, and FV's worlds omit "
               "DR's drift and year shocks. CB asks whether the SIDE of a 2028 Changhua E02c "
               "coverage fail separates the families once drift and shocks are in: if a "
               "page-family truth fails by over-coverage (>= 0.8 in every year/month ICC cell) "
               "and a "
               "Changhua-recipe truth rarely does (<= 0.2), a 2028 over-coverage fail is a "
               "one-sided ledger reading for C14a Changhua (never on the page). Round-8 "
               "plan entry #10 (rewritten).",
        "tests": _na("model-only declaration: synthetic worlds; no observation is scored and "
                     "no fresh unit is consumed"),
        "fresh_units": _na("no observation is scored"),
        "zones": _na("no fit; the frozen forward runs' own zones"),
        "folds": _na("synthetic worlds; no folds"),
        "burn_in": _na("no fit"),
        "gate_scores": _na("one gate by declared thresholds; no weighted score"),
        "estimand": {
            "name": "side distribution of E02c's Changhua coverage outcome",
            "definition": "P(over | T, cell, arm) = share of truth T's 250 worlds whose E02c "
                          "Changhua coverage outcome is 'coverage_not_supported' with E02c's "
                          "frozen side 'above' the registered range (unconditional: the "
                          "probability of the 2028-observable outcome 'fail by "
                          "over-coverage'); the side among non-passes is report-only",
            "outcomes": {"pass": "coverage_supported",
                         "over": "coverage_not_supported, side 'above' (score_e02c.cov_outcome)",
                         "under": "coverage_not_supported, side 'below'",
                         "pooled_only": "pooled_gate_failed_county_report_only (no side)",
                         "not_evaluable": "any other verdict"},
            "side_source": "score_e02c.cov_outcome's wording (cov_not_supported, side = 'below' "
                           "if coverage < 0.90 else 'above'), read back and verified per world; "
                           "score_e02c.py imported unchanged (hash-bound, never edited)",
            "not_fv_v3": "FV v2's P, readings and wording are not re-estimated or changed; E0 "
                         "only reproduces its published rates as an engineering identity"},
        "design": {
            "seed": seed, "seed_chosen_by": "the lead, at registration",
            "refused_seeds": {str(k): v for k, v in C.REFUSED_SEEDS.items()},
            "smoke_seed": C.SMOKE_SEED, "streams": C.STREAMS,
            "stream_rule": "SeedSequence([seed, stream, truth index (FV order R0, Q0, R1c, "
                           "X5R1), world k]); common random numbers across ICC cells and "
                           "drift arms",
            "k_per_truth": C.K, "truths": list(C.TAGS),
            "families": {"page": list(C.PAGE_FAMILY), "changhua_recipes": list(C.ALT_FAMILY)},
            "county": C.COUNTY,
            "world": "obs = ZR + D_T + R_T,k + n + drift on 2026-11..2027-10 at E02c's 158 "
                     "wells; ZR, D_T, R_T,k exactly FV v2's (osse_fv2.score_worlds: D_T from "
                     "batch 0's baseline and fv_zr0.npz, anchored on obs_mask.csv months; R_T,k "
                     "= world k minus its own batch baseline)",
            "noise": "n = sqrt(a_y) sd C + sqrt(1 - a_y) (sqrt(a_m) sd B_t + sqrt(1 - a_m) e); "
                     "e = FV v2's AR(1) (noise_ar1.csv sd, phi; osse_fv.simulate_ar1, fresh "
                     "stream); C one N(0, 1) per world (every well-month, both counties); B_t "
                     "one N(0, 1) per month (every well); marginal sd kept; (0, 0) gives e bit "
                     "for bit",
            "icc_grid": {"icc_year": list(C.ICC_YEAR), "icc_month": list(C.ICC_MONTH),
                         "cells": C.CELLS,
                         "source": f"{C.U1HB_ID} (power_u1hb.ICC_YEAR / ICC_MONTH, imported; "
                                   f"registration sha {C.U1HB_DECL_SHA[:8]}); ICC_well is not "
                                   "overlaid (per-well persistence is the AR(1) and the "
                                   "drift)"},
            "drift": {
                "source": f"{C.DR_ID} (registration sha {C.DR_DECL_SHA[:8]}; report sha "
                          f"{C.PINS['dr_report'][:8]}; wells_e02d.csv sha "
                          f"{C.PINS['dr_wells'][:8]})",
                "tau2_per_year2": "DR report decisions.<county>.tau2_per_year2 (Changhua "
                                  "0.0200, Yunlin 0.0185), grouping 'zone' and include_drift "
                                  "true checked",
                "model": "b_w (t - 2023.0), t the mid-month decimal year; b = sqrt(tau_c^2) z, "
                         "z ~ N(0, 1) per well centred within its DR zone group (wells_e02d.csv "
                         "'group') and rescaled by sqrt(n_g / (n_g - 1)); only station_id, "
                         "county, grouping and group are read from wells_e02d.csv",
                "arms": {"none": "no drift",
                         "origin": "DR's registered lead definition (gating)",
                         "datum_centre_report_only": "DR's report-only lead from the datum "
                                                     "centre (+5.5 years; report-only)"}},
            "fv2_batches": batches,
            "fv2_batches_rule": "the last successful forward of each (truth, batch) in FV v2's "
                                "attempts.jsonl; the runner re-derives them and hashes every "
                                "npz before opening it",
            "run": "python dataprep_staging/cb_side/cb_side.py run --seed <registered> "
                   "--workers 4 (tmux, CPU only, about 15-30 min; writes "
                   f"{C.REPORT} once and appends {C.ATTEMPTS})"},
        "gates": {
            "E0": "engineering, before any fresh draw: FV v2's own seed and noise stream, cell "
                  "(0, 0), no drift -> report_worlds.json P.coverage.<county>.by_truth "
                  f"reproduced for all four truths and both counties within {C.E0_TOL}; "
                  "failure writes nothing (engineering retry; no new statistic exists)",
            "E1": f"engineering, fresh seed: Q0 Changhua pass rate at (0, 0), no drift, within "
                  f"{C.E1_N_SE} Monte Carlo SE of FV v2's published {C.V2_Q0_CHANGHUA} (SE of "
                  "the difference of two binomial rates at the pooled rate; equality if that "
                  "SE is 0); failure writes a FINAL not_evaluable report (no other seed)",
            "CB": f"min over T in {list(C.PAGE_FAMILY)}, every ICC cell, drift 'origin', of "
                  f"P(over) >= {C.P_OVER_MIN} AND max over T in {list(C.ALT_FAMILY)}, every "
                  f"ICC cell, drift 'none' and 'origin', of P(over) <= {C.P_OVER_MAX}; point "
                  "estimates; each margin is reported in MC SE units (no SE allowance)",
            "order": "E0, E1, then the grid (3 drift arms x 12 cells x 1000 worlds)"},
        "readings": {
            "side_separates": "CB gate passed: a one-sided ledger / README / MODEL_CARD line "
                              "for C14a Changhua (only a 2028 over-coverage fail is read); no "
                              "page line, no status change",
            "side_does_not_separate": "CB gate failed: no side reading; the FV v2 per-truth "
                                      "disclosure stands; no page line",
            "not_evaluable": "E1 failed, or any failure after E1 passed: no side reading; "
                             "no further attempt",
            "applies_to": "C14a Changhua only; no C20/C21 mapping (N2C governs response "
                          "class)",
            "one_sided": "a 2028 pass or under-coverage fail gains no reading from CB"},
        "report_only": [
            "bias-cancelled-by-shock probability (reported explicitly): P(non-pass at (0, 0) "
            "and pass at the cell; same world and drift arm), per truth, arm and cell, and its "
            "max over cells",
            "over-lost-to-shock and side-flipped-by-shock (paired), bias-cancelled-by-drift",
            "side among non-passes (conditional)", "every outcome rate per truth, arm, cell",
            "Changhua coverage quantiles", "the datum-centre drift arm",
            "P(over) of the page family without drift"],
        "priors": "The audit's unverified guess: page-family truths fail by over-coverage "
                  "(the interval is wider than their noise). The plan puts the pass chance at "
                  "about 0.2 in Changhua: shocks and drift lower coverage, so a year shock can "
                  "move a page-family world into the window (a pass) or below it, and drift "
                  "spreads wells. Known before design (disclosed): FV v2's per-truth Changhua "
                  "pass rates (Q0 0.004, R0 0.012, R1c 0.992, X5R1 0.996) and E02-own rates; "
                  "U1.HB.power's false-fail 0.82-0.83 at ICC_year 0.2 with ICC_month 0.3; DR's "
                  "tau. No side rate of any FV v2 world is known to the builder.",
        "thresholds": {"p_over_min_page_family": C.P_OVER_MIN,
                       "p_over_max_changhua_recipes": C.P_OVER_MAX, "e1_n_se": C.E1_N_SE,
                       "e0_tol": C.E0_TOL, "k_per_truth": C.K},
        "primary_model": {"e02_predictions_sha256": S.E02_PRED_SHA,
                          "e02_forward_sha256": S.MODEL_SHA,
                          "truth_worlds": "FV.decidability.v2_postfailure's 20 batch forwards "
                                          "(4 truths x 250 worlds)"},
        "interpretation": "Design-time, model-only: the probability that E02c's Changhua "
                          "coverage outcome is a fail by over-coverage, by truth family, under "
                          "drift and shocks. It supports no claim about the aquifer or about "
                          "Changhua pumping; on a pass it lets a 2028 Changhua over-coverage "
                          "fail be read one-sidedly for C14a.",
        "binds": {
            "fv_decidability_v2": {
                "declaration": _bind(C.FV2_DECL, C.FV2_DECL_SHA),
                "report_worlds": {**_bind(C.FV2_REPORT, C.PINS["fv2_report"]),
                                  "keys_used": "P.coverage.<county>.by_truth (E0 and E1)"},
                "attempts": {"path": C.FV2_ATTEMPTS, "sha256_at_draft": _sha(C.FV2_ATTEMPTS),
                             "use": "batch npz sha256s and the score_worlds / score_ext rows "
                                    "(seal lapse); not an input (FV v2 owns the file)"},
                "shared_code": "osse_fv2.py, osse_fv.py, score_e02c.py, coverage3.py and "
                               "bands.py must equal FV v2's registered frozen_code (checked "
                               "by the runner)"},
            "fv_ext": _bind(FV_EXT_DECL, FV_EXT_DECL_SHA),
            "dr": {"declaration": _bind(C.DR_DECL, C.DR_DECL_SHA),
                   "report": _bind(C.DR_REPORT, C.PINS["dr_report"]),
                   "wells": _bind(C.DR_WELLS, C.PINS["dr_wells"])},
            "u1_hb_power": {"declaration": _bind(C.U1HB_DECL, C.U1HB_DECL_SHA),
                            "result_hashed_only": _bind(U1HB_RESULT, U1HB_RESULT_SHA)},
            "e02c_registration_hashed_only": _bind(E02C_DECL, E02C_DECL_SHA),
            "inputs": {k: {"path": getattr(C.Paths(), k), "sha256": _sha(getattr(C.Paths(), k)),
                           "expected_sha256": v} for k, v in C.PINS.items()},
            "ruling": ruling_binding()},
        "seal": {
            "fv2_clause": "FV v2's sealed_until_scoring: 'forward stdout (sealed file, sha "
                          "logged); batch npz values are read only by the scorers (this one "
                          "and FV.ext.v1)'",
            "status": "both scorers ran (attempts rows score_worlds and score_ext, rc 0), so "
                      "the clause's stated scope has lapsed: lead decision 2026-11-03, "
                      f"recorded with look {SEAL_LOOK}",
            "cb": f"CB opens the batch npz (synthetic model output) only in 'run', after "
                  f"verifying each sha256, and never the sealed stdout ({SEALED_STDOUT})"},
        "page_mapping": page_mapping(),
        "attempt_policy": {
            "scored_runs": "the study runs once on the registered seed; report.json is "
                           "written once (mode 'x'); never re-drawn on another seed or grid",
            "engineering_retries": "free only while E0 or an input / registration check "
                                   "fails (nothing beyond FV v2's published numbers has been "
                                   "computed); each attempt appended to " + C.ATTEMPTS,
            "e1_failure": "final not_evaluable report",
            "after_e1": "LEAD decision: any failure after E1 passed (an exception in the grid "
                        "or its statistics) writes a FINAL not_evaluable report; the fresh "
                        "draws were made, so nothing is re-run",
            "attempts_rows": "every invocation of 'run' appends one attempts row (refusals "
                             "before registration, refused re-runs and interrupts included; "
                             "BaseException is logged and re-raised)",
            "exploratory_followups_max": 0,
            "thresholds_frozen": True,
            "report": "every attempt, failures included"},
        "rulings": {RULING_ID: "page wording (sha in binds.ruling); see page_mapping"},
        "disclosures": {
            "builder": BUILDER,
            "optimism": "rain at climatology after 2022-12 (FV v2's worlds); the AR(1) part "
                        "is independent between wells; shocks are Gaussian and shared by both "
                        "counties (U1.HB's generator); every well-month is observed",
            "drift": "DR's tau is a LOWER bound (in-sample damping); the gated arm uses DR's "
                     "'origin' lead (4.4 years mid-window), the smaller drift; a larger drift "
                     "lowers coverage, so the gated page-family condition is the favourable "
                     "one for over-coverage. The datum-centre lead (+5.5 years) is reported "
                     "beside it, report-only; that arm shifts every well by its county's "
                     "median shift (DR's lead_years_datum_centre_report_only - lead_years, "
                     "a county median over wells, +5.5 years), not by each well's own datum "
                     "centre. Common-mode drift: DR's tau^2 is group-relative only (each "
                     "well against its zone group's leave-one-out median), so no drift shared "
                     "by a whole zone group or county is overlaid; shared drift would shift "
                     "a county's wells together and lower coverage, so leaving it out favours "
                     "over-coverage (the gated page-family condition). DR's L (4.375 years) is "
                     "the origin-to-2027-05 lead (E02d window A's middle); CB uses each "
                     "window month's own lead (3.875-4.792 years, mean 4.333)",
            "icc_well": "ICC_well is not overlaid: U1.HB.power's well-persistence axis is "
                        "represented only by the AR(1) persistence and the drift; the gate "
                        "covers every year/month ICC cell, not U1.HB's ICC_well axis",
            "estimand_choice": "P(over) is unconditional (the 2028-observable outcome); the "
                               "side conditional on a fail is report-only because Changhua-"
                               "recipe worlds rarely fail without shocks",
            "families": "the gate is per truth (min / max over truths), not a pooled mean",
            "e0": "E0 recomputes FV v2's published coverage rates with FV v2's own seed as an "
                  "engineering identity; it is not a new statistic",
            "scope": "Changhua only; Yunlin's side is not computed",
            "multiplicity": "FV v2, FV.ext, FV.stat.v2, U1.HB.power, U1.decomp_power and E02c "
                            "share these wells and worlds; CB decides only its own reading",
            "tests": {"path": C.TESTS, "sha256_at_declaration": _sha(C.TESTS),
                      "binding": "disclosure only, not an input"}},
        "deadline": {"register_by": REGISTER_BY, "hard_limit": HARD_LIMIT,
                     "rule": "before any window product is read and before 2028-01-15; seed, "
                             "ICC grid, gate and readings fixed before the run"},
        "frozen_code": frozen,
        "registration_script": {"path": C.PREREG, "sha256": _sha(C.PREREG),
                                "binding": "report-only: not frozen; the registered "
                                           "document's sha binds what this script wrote"},
        "written_by": C.PREREG,
        "inputs": inputs,
    }
    return json.loads(json.dumps(doc, default=O._jsonable))


# ----------------------------------------------------------------------------- checks
def _git_problems(paths) -> list[str]:
    out = []
    for p in paths:
        tracked = subprocess.run(["git", "-C", str(C.REPO), "ls-files", "--error-unmatch", p],
                                 capture_output=True, text=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(C.REPO), "diff", "--quiet", "HEAD", "--", p],
                               capture_output=True).returncode == 0
        if not tracked or not clean:
            out.append(f"{p} is not committed as-is (R7.commit_first)")
    r = subprocess.run(["git", "-C", str(C.REPO), "merge-base", "--is-ancestor", "HEAD",
                        "origin/main"], capture_output=True, text=True)
    if r.returncode != 0:
        out.append("HEAD is not contained in origin/main (merge before registration)")
    return out


def logged_look(unit: str, purpose: str, prereg_dir=None) -> bool:
    from harness import prereg3

    p = (Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR) / "looks.jsonl"
    return p.is_file() and any(lk.get("units") == [unit] and lk.get("purpose") == purpose
                               for lk in prereg3.read_chained(p))


def readiness(d: dict, prereg_dir=None, now=None, verify_npz: bool = False) -> list[str]:
    """Everything that must hold before the lead registers the draft (empty = ready)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    if now >= dt.datetime.fromisoformat(HARD_LIMIT):
        out.append("past 2028-01-15: too late")
    elif now > dt.datetime.fromisoformat(REGISTER_BY):
        out.append("past the plan's 2026-12-15 deadline (lead decides)")
    for name, b in (("FV v2 registration", d["binds"]["fv_decidability_v2"]["declaration"]),
                    ("report_worlds.json", d["binds"]["fv_decidability_v2"]["report_worlds"]),
                    ("FV.ext registration", d["binds"]["fv_ext"]),
                    ("DR registration", d["binds"]["dr"]["declaration"]),
                    ("DR report", d["binds"]["dr"]["report"]),
                    ("DR wells", d["binds"]["dr"]["wells"]),
                    ("U1.HB registration", d["binds"]["u1_hb_power"]["declaration"]),
                    ("E02c registration", d["binds"]["e02c_registration_hashed_only"]),
                    ("rulings", d["binds"]["ruling"]),
                    *((f"input {k}", v) for k, v in d["binds"]["inputs"].items())):
        if b["sha256"] != b["expected_sha256"]:
            out.append(f"{name}: sha256 {b['sha256'][:12]} != {b['expected_sha256'][:12]}")
    if MISSING in d["frozen_code"].values():
        out.append("a frozen code file is missing")
    want = {f"{t}_b{b}" for t in C.TAGS for b in range(V2.PLAN.n_batch)}
    if set(d["design"]["fv2_batches"]) != want:
        out.append("FV v2 attempts.jsonl does not give all 20 batch forwards")
    if not C.seal_lapsed(C.FV2_ATTEMPTS)["lapsed"]:
        out.append("FV v2 or FV.ext not scored: the seal has not lapsed")
    if verify_npz:
        for key, sha in d["design"]["fv2_batches"].items():
            t, b = key.rsplit("_b", 1)
            if _sha(V2.batch_npz(t, int(b))) != sha:
                out.append(f"{key}: npz sha256 differs from FV v2's attempts.jsonl")
    for unit, purpose in BUILDER_LOOKS:
        if not logged_look(unit, purpose, prereg_dir=prereg_dir):
            out.append(f"builder look {unit} (purpose '{purpose}') not logged yet (lead)")
    out += _git_problems(C.FROZEN)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--verify-npz", action="store_true",
                    help="hash the 20 FV v2 batch npz (about 8 GB) against attempts.jsonl")
    a = ap.parse_args(argv)
    from harness import prereg3

    d = draft_prereg(a.seed)
    print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print(f"\ndraft sha256 {sha_text(d)}")
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "OK")
    todo = readiness(d, verify_npz=a.verify_npz)
    print("readiness:", "READY" if not todo else "\n  - " + "\n  - ".join(todo))
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
