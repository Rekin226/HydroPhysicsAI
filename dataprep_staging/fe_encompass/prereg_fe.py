"""Draft registration of FE.encompass_2026-12.v1 (harness.prereg3 schema, kind 'test'), built in
code because dataprep_staging/ ignores *.json. Nothing here writes to the registry: the lead
reviews the draft, picks the seed and registers it with prereg3.write_prereg after the code
is committed (R7.commit_first), before 2026-11-30T16:00Z. A registration created after
2026-11-25T23:59:59Z makes FE report-only (review B2 hard rule, enforced in decide_fe and
score_fe from the INDEX.jsonl creation time).

  prereg_fe.py --seed N [--verify-npz]   prints the draft JSON, its sha256, validate_prereg,
                                         the blocking probe and readiness
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

import decide_fe as D  # noqa: E402  (puts FV.stat.v2 and the staging tree on sys.path)
import fe_lib as L  # noqa: E402

O, S, V2 = D.O, D.S, D.V2  # noqa: E741
MISSING = "MISSING"
PENDING = "PENDING"
FE_BUILDER_LOOK = ("R8:fe_encompass_builder_reads_disclosure", "disclosure")
FE_FIXER_LOOK = ("R8:fe_encompass_fixer_reads_disclosure", "design_time")
FE_LOOKS = (FE_BUILDER_LOOK, FE_FIXER_LOOK)
GOV_DECL = "results/dataprep/preregistration/GOV.heads_window_product.v1.json"
GOV_DECL_SHA = "141c012899a67c69eecfd630a6e9a92de3c057f5cfa2d67412adf42608167855"
GOV_CODE_SHA8 = "9a953039"
E02C_NOT_SCORABLE_RULE = (
    "Pre-declared: if E02c's 2028 outcome is a record without e02_step5 "
    "['evaluation.csv_sha256'] (a lead-written not-scorable record, or any E02c report that "
    "scored no E02 step-(5) rows), FE writes a FINAL report 'not_evaluable (E02c not "
    "scorable)': no row is read, no look is logged, FE never waits for a later E02c and never "
    "falls back to an unchained product")
SESSION_ORDER = (
    "the 2028-01-15..31 session, after the E02c session order of score_e02c.SESSION_ORDER: "
    "(1) E02 step (5) and E04 step (5) (python -m hydrophysics.twin.prospective score on "
    "results/twin/prospective_2026-11 and results/twin/prospective_2026-11_hybrid, --new-wells "
    "inside the window product); (2) E02c.prospective_2026-11.v1 scored (or a lead-written "
    "not-scorable record); (3) E02d.decomposed_prospective_2026-12.v1 window A scored (or a "
    "record); (4) FE last: window-product and chain checks, E02d's window-A precondition "
    "through the GOV module, order check, assert_can_score, then E02's and E04's step-(5) rows "
    "are read. FE never runs a forward and never re-runs E04's or E02's scorer.")
ORDER_DEVIATION = (
    "The plan states the order E02c -> E02d -> E04 -> FE. E02c's own registered session order "
    "(score_e02c.SESSION_ORDER) scores E04's step (5) (the hybrid manifest) at step 6, before "
    "E02c at step 9, so the literal order cannot hold. FE enforces 'E02c, E02d and E04 all "
    "done, FE last' instead: a disclosed deviation from the plan's order.")
MULTIPLICITY = (
    "E02 (pooled), E02c (T-cov, T-level), E02d, E04 (the hybrid manifest), I1.hybrid_tf T1/T2, "
    "X1 T3a/T3b, N2, N2C, N2CS and E02x score overlapping well-months of the same wells; FE "
    "decides only its own claim (which first-year forecast carries the information) and no "
    "family-wise claim is made")
BUILDER = (
    "The FE builder and its fixer (round 8, 2026-11-03) read code (fv_stat_v2, "
    "fv_decidability(_v2), e02c, e02d, gov_heads_window, harness.prereg/2/3, twin.prospective, "
    "common.manifest) and these non-code files: docs/TWIN_ROUND8_PLAN.local.md (entry #9, "
    "rank 3, execution notes); hydrophysics/twin/claims_v1.json rows C30, E02, E04; the E02 "
    "and E04 predictions.csv (model output: header, ids, origins, counts, interval widths and "
    "baseline_m; E04's first prediction row printed by a head command; no observation); both "
    "manifest.json and protocol.json (the E04 protocol's printed retrospective_basis carries "
    "its registered 2023-26 rmse_ratio and picp); FV.stat.v2's registration (keys, "
    "frozen_code, design seed and streams) and its report's status, readings.e04, P and null "
    "P fields; user_rulings_2026-11-03.json (ids, decisions, R8.7 in full); "
    "FV.decidability.v1's inputs/summary.json in full (it printed the pre-origin pumping "
    "clim_pool_gwh values, design_time_evidence, already in FV v1's logged look) and its "
    "registration's noise / inputs_frozen fields; FV.decidability.v2's registration (key "
    "list; the noise, inputs_frozen, zr0, sealed_until_scoring and truths fields); the key "
    "list (no array) of results/twin_forward/fv2_truth_R0_b0.npz; directory listings of "
    "results/ (twin_forward, twin_runs/fv2, the FV report dirs); the E02c / E02d / FV "
    "registrations, FV v2's report_worlds.json, the FV.stat.v2 erratum, noise_ar1.csv, "
    "obs_mask.csv and wells_158.csv hashed only (the code reads the last three at run time); "
    "looks.jsonl, INDEX.jsonl and the units registries through prereg3 (nothing printed). "
    "It computed no lambda, interval, rate or contrast statistic on real inputs, read no head, "
    "opened no npz array and no sealed log, and ran the study code only on FABRICATED inputs "
    "in tests. A second fixer pass read E02c's registration (key list, scoring field, report-path "
    "grep), the reviewer's scratch script b1_exp2.py (code) and nothing else non-code. "
    f"Logged by the lead as two looks: {FE_BUILDER_LOOK[0]} (the builder's reads) "
    f"and {FE_FIXER_LOOK[0]} (logged 2026-11-03T04:53:37Z; the fixer's reads, including the "
    "FV v1 summary.json print).")


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _sha(p) -> str:
    q = O.rp(p)
    return O.sha256(q) if q.is_file() else MISSING


def _json(p) -> dict | None:
    q = O.rp(p)
    return json.loads(q.read_text(encoding="utf-8")) if q.is_file() else None


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def _bind(path: str, expected: str) -> dict:
    return {"path": path, "sha256": _sha(path), "expected_sha256": expected}


def fv_stat_v2_binding() -> dict:
    """FV.stat.v2's registration and report by sha256; its reading and P / P0 taken from the
    report only when the report's sha256 is the registered one."""
    decl = _json(L.FV_STAT_V2_DECL)
    rep_ok = _sha(L.FV_STAT_V2_REPORT) == L.FV_STAT_V2_REPORT_SHA
    rep = _json(L.FV_STAT_V2_REPORT) if rep_ok else None
    reg_fc = (decl or {}).get("frozen_code") or {}
    gen = {p: {"registered": reg_fc.get(p, MISSING), "now": _sha(p)} for p in S.FROZEN}
    p = ((rep or {}).get("P") or {}).get("p")
    p0 = (((rep or {}).get("null_reading") or {}).get("P") or {}).get("p")
    return {"id": L.FV_STAT_V2_ID,
            "declaration": _bind(L.FV_STAT_V2_DECL, L.FV_STAT_V2_DECL_SHA),
            "report": _bind(L.FV_STAT_V2_REPORT, L.FV_STAT_V2_REPORT_SHA),
            "erratum_county_layout": _bind(L.FV_STAT_V2_ERRATUM, L.FV_STAT_V2_ERRATUM_SHA),
            "status": (rep or {}).get("status", PENDING),
            "e04_reading": ((rep or {}).get("readings") or {}).get("e04", PENDING),
            "P": p, "P0_null": p0,
            "generator_code_unchanged": all(v["registered"] == v["now"] != MISSING
                                            for v in gen.values()),
            "generator_code": gen,
            "use": "the AR world generator (stat_v2.world -> stat_fv.anomalies) and E04's "
                   "layer re-run (stat_fv.e04_frame) are imported unchanged with FE's own "
                   "layout (81 Yunlin / 66 Changhua by station code, E04's own origins); "
                   "FV.stat.v2's layout constant (78 / 69, see its erratum) is NOT used; fresh "
                   "FE seed and streams 72-85, so no FV.stat.v2 world is reused"}


def fv2_binding() -> dict:
    return {"id": L.FV2_ID, "declaration": _bind(L.FV2_DECL, L.FV2_DECL_SHA),
            "report_worlds": _bind(L.FV2_REPORT, L.FV2_REPORT_SHA),
            "sealed_until_scoring_clause": "FV v2's registration: 'batch npz values are read "
                                           "only by the scorers (this one and FV.ext.v1)'; "
                                           "both are scored (report_worlds.json, FV.ext.v1 "
                                           "report.json), so the clause's stated scope has "
                                           "lapsed (lead's confirmation, 2026-11-03; the npz "
                                           "are synthetic model worlds); FE opens them only in "
                                           "decide_fe run and never the sealed forward stdout; "
                                           f"look {FE_FIXER_LOOK[0]} (logged "
                                           "2026-11-03T04:53:37Z)",
            "attempts": L.FV2_ATTEMPTS}


def ruling_binding() -> dict:
    r = _json(L.RULINGS) or {}
    hit = [x for x in r.get("rulings", []) if x.get("id") == L.RULING_ID]
    return {**_bind(L.RULINGS, L.RULINGS_SHA), "id": L.RULING_ID,
            "decision": hit[0].get("decision") if hit else MISSING}


def _reg(pid: str) -> dict:
    p = f"results/dataprep/preregistration/{pid}.json"
    return {"id": pid, "path": p, "sha256": _sha(p)}


def _files(d: str) -> dict:
    return {n: {"path": f"{d}/{n}", "sha256": _sha(f"{d}/{n}")}
            for n in ("predictions.csv", "protocol.json", "manifest.json")}


def unit_ids() -> list[str]:
    return list(D.load_inputs().ids)


def external_unit(ids: list[str]) -> dict:
    return {L.UNIT: {"description": f"the {len(ids)} wells E02 (158) and E04 (147) share, "
                                    f"heads {L.FIRST}..{L.LAST} (FE, fresh, year 1)",
                     "ids": list(ids), "window": f"{L.FIRST}..{L.LAST}"}}


def planned_role(now=None) -> dict:
    now = now or dt.datetime.now(dt.timezone.utc)
    late = now > dt.datetime.fromisoformat(L.GATING_ARM_BY)
    return {"if_registered_now": "report_only" if late else "gating_if_g0_passes",
            "gating_arm_by": L.GATING_ARM_BY,
            "rule": "decide_fe.role_cap on the INDEX.jsonl creation time: after "
                    "2026-11-25T23:59:59Z FE is report-only whatever G0 says"}


def draft_prereg(seed: int, ids: list[str] | None = None, now=None) -> dict:
    """FE's registration draft for ``seed`` (the lead's choice; refused seeds raise)."""
    seed = D.check_seed(seed)
    ids = list(ids) if ids is not None else unit_ids()
    fv = fv_stat_v2_binding()
    fvp = (f"P {fv['P']:.3f}; null P0 {fv['P0_null']:.3f}"
           if isinstance(fv["P"], float) and isinstance(fv["P0_null"], float)
           else "P and P0 not available (report missing or not the registered one)")
    test = {
        "test_id": L.TEST, "data_scope": L.DATA_SCOPE, "fresh_units": [L.UNIT],
        "external_units": external_unit(ids),
        "label": "prospective: the window's heads did not exist at registration",
        "claim": "forecast of record (R8.7): does the frozen physics forecast E02 add "
                 "information beyond the frozen statistical layer E04 (its climatology AND its "
                 "persisted anomaly) on first-year within-well head anomalies, 2026-12..2027-10?",
        "metric": {
            "rows": "E02's and E04's step-(5) evaluation rows (verified against their "
                    "registered predictions by score_e02c.load_e02_step5), identical well-months "
                    "and observed_m at the unit's wells (1e-9 m); E02's evaluation.csv is the one "
                    "E02c's report scored; months 2026-12..2027-10; >= 6 months per well",
            "y": "obs - STAT (STAT = E04 predicted_m)",
            "x_P": "PHYS - STAT (PHYS = E02 predicted_m)",
            "x_C": "CLIM - STAT (CLIM = E04 baseline_m, the layer's climatology)",
            "primary": "encompassing coefficient lambda_P of the within-well two-contrast "
                       "least-squares fit y = lambda_P x_P + lambda_C x_C + e (each well's "
                       "window means removed; rows weighted equally)",
            "interval": "90 % percentile interval [L_P, U_P] of a well x quarter (pigeonhole) "
                        f"block bootstrap, B = {L.BOOT_B}: wells and the four quarters "
                        f"{list(L.QUARTERS)} drawn with replacement, row weight = well copies x "
                        "quarter copies, within-well means recomputed per replicate; draws from "
                        f"SeedSequence([seed, {D.STREAMS['score']}, 0, 0])",
            "informative_gate": f"partial physics contrast RMS (x_P within wells net of its "
                                f"projection on x_C) over the scored rows >= {L.X_RMS_MIN_M} m "
                                "(model output only), else not_informative",
            "report_only": ["lambda_C", "one-contrast lambda (x_P alone)",
                            "raw-level two-contrast fit (no demeaning, no intercept)",
                            "per-county two-contrast fits", "rows and wells scored"]},
        "decision_rule": {"rule": L.READING_RULE, "readings": list(L.READINGS),
                          "delta": L.DELTA}}
    frozen = {p: _sha(p) for p in D.FROZEN}
    inputs = sorted({*D.FROZEN, *(f"{d}/{n}" for d in (L.E02_DIR, L.E04_DIR)
                                  for n in ("predictions.csv", "protocol.json",
                                            "manifest.json")),
                     L.FV_STAT_V2_DECL, L.FV_STAT_V2_REPORT, L.RULINGS, L.FV2_DECL, L.FV2_REPORT,
                     L.FV1_DECL, L.NOISE_CSV, L.MASK_CSV, L.WELLS_CSV, GOV_DECL,
                     f"results/dataprep/preregistration/{L.E02C_ID}.json",
                     f"results/dataprep/preregistration/{L.E02D_ID}.json"})
    doc = {
        "id": L.ID, "wp": L.WP, "kind": "test",
        "title": "FE: does the frozen physics forecast (E02) add to the frozen statistical "
                 "layer (E04)? Within-well two-contrast encompassing coefficient on the shared "
                 "wells, 2026-12..2027-10, well x quarter block bootstrap",
        "why": "R8.7 names E04 the first-year forecast of record and labels physics levels "
               "after 2023-11 'average-weather scenario'. FE is the one prospective test of "
               "that choice on anomaly dynamics: lambda_P > 0 means the physics forecast "
               "carries information the statistical layer (climatology and persisted anomaly) "
               "lacks; an upper bound below the declared margin 0.25 means the layer "
               "encompasses it.",
        "tests": [test],
        "design": {"seed": seed, "seed_chosen_by": "the lead, at registration",
                   "refused_seeds": {str(k): v for k, v in D.REFUSED_SEEDS.items()},
                   "smoke_seed": D.SMOKE_SEED, "streams": D.STREAMS,
                   "window": [L.FIRST, L.LAST], "quarters": list(L.QUARTERS),
                   "wells": len(ids), "B": L.BOOT_B, "ci": list(L.CI), "delta": L.DELTA,
                   "min_months": L.MIN_MONTHS, "min_wells": L.MIN_WELLS,
                   "planned_role": planned_role(now)},
        "decidability": {
            "what": "model-only synthetic study (decide_fe.py run), CPU, no forward run, no "
                    "observation; run ONCE under this registration, before "
                    f"{L.REGISTER_BY}; written to {L.DECIDABILITY}",
            "streams": D.STREAMS, "B": D.B_STUDY,
            "scale": {"source": L.NOISE_CSV, "sha256": L.NOISE_SHA,
                      "what": "per-well sd_m and phi of the page model's 2012-01..2022-12 "
                              "fitted residuals (FV.decidability.v1 export; L1_clean v3 flags "
                              "0-1; window_end 2022-12; use design_time_evidence; look "
                              "FV:preorigin_residuals_R0wells logged 2026-11-01T03:06Z), "
                              "registered in FV.decidability.v1 (inputs_frozen) and reused by "
                              "FV v2; every synthetic error in every arm is scaled by it"},
            "arms": {
                "ar": {"role": "gating (lambda_P = 0)", "K": D.K_AR, "cells": D.ar_cells(),
                       "world": "FV.stat.v2 AR world x sd on the 147 wells; E04's layer re-run; "
                                "PHYS = CLIM_world + dC_j, dC_j = PHYS - CLIM of PHYS-arm world "
                                "j (an analogue built from 2012-22 inputs only; j = (offset + "
                                "cell K + k) mod N, offsets " + json.dumps(D.POOL_OFFSET) + ")",
                       "streams": "mask_ar, subset_ar, boot_ar"},
                "pe": {"role": "gating (persistence error)", "K": D.K_PE,
                       "cells": D.pe_cells(), "shapes": {k: v for k, v in
                                                         D.PE_SHAPES.items()},
                       "shape_amplitude": D.PE_AMP, "share": D.PE_SHARE,
                       "world": "AR(1) at rho_hist (FV.stat.v2's generator, what E04's re-run "
                                "layer fits) up to each well's origin, AR(1) at rho_win after it "
                                "(from the origin value, same share); PHYS = CLIM_world + dC_j "
                                "+ 0.5 a0 r^h (a0 = the world's anomaly at the origin, h = "
                                "months after it, r in {none, 0.6, 0.99}); gated per (cell, "
                                "shape)"},
                "phys": {"role": "gating", "K": D.K_PHYS, "truths": list(D.TRUTHS),
                         "cells": D.phys_cells(), "lambda_points": list(D.LAMBDA_POINTS),
                         "world": "FV v2 batch forward h_T,k at the wells' cell/layer "
                                  "(wells_158.csv) + FV v2-rule noise (noise_ar1 sd, phi; "
                                  "independent) over 2012-01..2027-10; E04's layer re-run; "
                                  "PHYS = h_ZR0 (fv_zr0.npz) + E02's datum fitted on the world's "
                                  "observed 2012-22 months (obs_mask.csv); lambda-world = obs - "
                                  "(1 - lambda)(PHYS - STAT)",
                         "zr0": {"path": L.ZR0_NPZ, "sha256": L.ZR0_SHA}},
                "sigma": {"role": "report-only", "K": D.K_SIG, "cells": D.sig_cells(),
                          "world": "y = x_P,j + sd x FV.stat.v2 AR world on the window, the "
                                   "analogue regressors of PHYS world j; true lambda_P 1, "
                                   "lambda_C 0"},
                "order": "PHYS first (it builds the analogue pool), then AR, PE, SIGMA"},
            "u7": "no real contrast (E02 / E04 predicted_m, E04 baseline_m) enters any arm or "
                  "any G0 condition; gate (e) is the median analogue partial contrast RMS over "
                  "the PHYS full-variant worlds; the real contrasts enter only at scoring "
                  "(the scoring-time informative gate, then the fit), as N2CS does",
            "traceability": "decidability.json records truth_files_used (every FV v2 batch npz "
                            "and the ZR0 sha256 read); score_fe verifies them against "
                            "fv2_batches and the pinned ZR0 sha, and that decidability.json is "
                            "the file its attempts.jsonl success row hashed under this "
                            "registration",
            "fv2_batches": D.fv2_batches_registered(),
            "g0": D.G0, "g0_rule": D.G0_RULE,
            "fallback": "if decidability.json is not written under this registration before "
                        f"{L.REGISTER_BY}, or the registration is created after "
                        f"{L.GATING_ARM_BY}, FE is report-only",
            "optimism": "history fully observed in every world; no drift or datum step; AR "
                        "errors exactly AR(1); rain at climatology after 2022-12 in the PHYS "
                        "worlds (FV v2's optimism), so E04's re-run layer sees little "
                        "real-weather persistence there after 2022 and PHYS-arm power is "
                        "conditional on that; PHYS noise independent between wells"},
        "binds": {"e02": _files(L.E02_DIR), "e04": _files(L.E04_DIR),
                  "read_only_registrations": [_reg(L.E02C_ID), _reg(L.E02D_ID)],
                  "fv_stat_v2": fv, "fv_decidability_v2": fv2_binding(),
                  "fv_decidability_v1": {"declaration": _bind(L.FV1_DECL, L.FV1_DECL_SHA),
                                         "noise_ar1": _bind(L.NOISE_CSV, L.NOISE_SHA),
                                         "obs_mask": _bind(L.MASK_CSV, L.MASK_SHA)},
                  "wells_158": _bind(L.WELLS_CSV, L.WELLS_SHA),
                  "gov": {**_bind(GOV_DECL, GOV_DECL_SHA), "id": "GOV.heads_window_product.v1",
                          "module": D.GOV},
                  "ruling": ruling_binding()},
        "frozen_code": frozen,
        "registration_script": {"path": L.PREREG, "sha256": _sha(L.PREREG),
                                "binding": "report-only: not frozen; the registered document's "
                                           "sha binds what this script wrote"},
        "scoring": {"not_before": L.NOT_BEFORE[:10], "no_later_than": L.NO_LATER_THAN,
                    "order": SESSION_ORDER, "order_deviation": ORDER_DEVIATION,
                    "e02c_not_scorable": E02C_NOT_SCORABLE_RULE,
                    "e02c_report": f"canonical {L.E02C_REPORT} (E02c's documented --out; a "
                                   "lead-written not-scorable record goes there too) or a "
                                   "byte-identical copy, and E02c's own done-status (its score "
                                   "looks when it claims a score) must pass; otherwise "
                                   "REFUSED; E02c's report must name its window_product",
                    "e02d_report": "an E02d record must name window 'A' (lead-written records "
                                   "too); otherwise REFUSED",
                    "precondition": "the window product exists, its MANIFEST is readable and "
                                    "it is E02c's product (otherwise REFUSED, nothing "
                                    "written); E02d's window-A rule through the GOV module "
                                    f"(heads_window.check_precondition('{L.E02D_ID}:A')): "
                                    "prospective_scorer, and a declared "
                                    "prospective_scorer_window covers 2026-12..2027-10; "
                                    "otherwise not_evaluable, nothing read, no look",
                    "scorer": f"{L.SCORER} score --seed <registered> --prereg-id {L.ID} "
                              "--window-product <dir> --e02-score-dir <dir> --e04-score-dir "
                              "<dir> --e02c-report <json> --e02d-report <json>",
                    "role": "score_fe.role_from: gating iff decidability.json on time with G0 "
                            "passed and the registration created by 2026-11-25",
                    "no_forwards": True},
        "sealed_until_scoring": "every head of 2026-12..2027-10 at the unit's wells; no "
                                "within-well lambda of E02 against E04 on any past window is "
                                "computed before scoring",
        "page_mapping": {
            "ruling": "R8.7 only (page line, ledger, README)",
            "ledger_row": {"id": "E28 (proposed; the lead assigns the id)",
                           "claim": "Forecast of record for first-year heads (2026-12.."
                                    "2027-10) at the wells E02 and E04 share",
                           "status_current": "D",
                           "wording_en": "decided by FE.encompass_2026-12.v1 (prospective "
                                         "encompassing test), not before 2028-01-15",
                           "cites": {"registrations": [L.ID, L.FV_STAT_V2_ID, L.FV2_ID,
                                                       L.E02C_ID, L.E02D_ID]}},
            "before_scoring": L.WORDING["before_scoring"],
            "gating": {"physics_informative": "a page line beside E04 (E04 stays forecast of "
                                              "record; no blend); C30-fc stays D via E04",
                       "stat_encompasses": "the pre-written narrowing beside E04 and on "
                                           "C30-scen",
                       "undecided": "E04's line gains the undecided sentence; no status "
                                    "change",
                       "not_informative": "the not-informative sentence; no status change",
                       "not_evaluable": "the not-evaluable sentence; no status change"},
            "report_only": "the reading with the report-only prefix in notes only; no status "
                           "change on the page",
            "wording": L.WORDING,
            "never": "no combined forecast or correction factor is built from lambda_P; "
                     "lambda_P is called the 'encompassing coefficient' (涵蓋係數), never a "
                     "weight"},
        "disclosures": {
            "builder": BUILDER,
            "multiplicity": MULTIPLICITY,
            "order_deviation": ORDER_DEVIATION,
            "U7": "No post-2022 data in the design: FE's design reads no observation, no "
                  "real contrast and no post-2022-12 head-derived scale. From the frozen "
                  "predictions it uses only the well ids and E04's per-well origin months. "
                  "Every synthetic error is scaled by FV v1's pre-origin noise_ar1.csv "
                  "(window_end 2022-12); every physics contrast in the study (the AR and PE "
                  "arms' dC, gate (e), SIGMA's regressors) is a PHYS-world analogue built in FV "
                  "v2's forward worlds from 2012-22 inputs only. The real contrasts (E02 and "
                  "E04 predicted_m, E04 baseline_m) enter only at scoring: the scoring-time "
                  "informative gate on the scored rows, then the fit (as N2CS).",
            "county": "county by station code (E02c's COUNTY map: 81 Yunlin, 66 Changhua); "
                      "FV.stat.v2's layout constant (78 / 69) is not used (FV.stat.v2 erratum "
                      f"{L.FV_STAT_V2_ERRATUM}, sha {L.FV_STAT_V2_ERRATUM_SHA[:8]})",
            "window": "FE's window starts 2026-12 (round-8 rule for registrations after "
                      "2026-11-01): E04's and E02's 2026-11 rows are not scored here",
            "b1": "a one-contrast lambda (x_P alone) can be driven by the layer's own "
                  "persistence error (x_P contains the layer's persisted anomaly); the "
                  "two-contrast fit puts that into lambda_C. The PE arm (history rho 0.93 / "
                  "window 0.80 and the reverse, with origin-anomaly-shaped dC) gates lambda_P "
                  "against it; the one-contrast rates are reported beside it. Correction: the "
                  "earlier 'over-persist' arm (y + (1 - kappa) x_C) was an algebraic copy of the "
                  "AR arm for lambda_P (it moves only lambda_C) and is withdrawn",
            "linearity": "lambda_P is linear in y for fixed regressors, so the lambda = 0.5 "
                         "and 0.25 PHYS worlds are the lambda = 1 worlds shifted by "
                         "-(1 - lambda) in every replicate",
            "gov_binding": "FE freezes dataprep_staging/gov_heads_window/heads_window.py "
                           f"(sha {GOV_CODE_SHA8}..., the module of the registered "
                           f"GOV.heads_window_product.v1, registration sha {GOV_DECL_SHA[:8]}) "
                           "and E02d's score_e02d.py through it; FE uses E02d's window-A "
                           "PRECONDITIONS key, whose months equal FE's (no PRECONDITIONS edit)",
            "phys_history": "in the PHYS worlds 2023-01..2026-09 is a free run at "
                            "climatological rain plus pumping draws and noise, so E04's re-run "
                            "layer sees little real-weather persistence there after 2022; the "
                            "PHYS-arm power is conditional on that",
            "fv_stat_v2_reading": f"FV.stat.v2 was scored {fv['e04_reading']} ({fvp}, from "
                                  "its sha-verified report): a no-persistence world passes "
                                  "E04's own rule about half the time; FE asks a different "
                                  "question (added information, not E04 vs climatology)",
            "tests": {"path": L.TESTS, "sha256_at_declaration": _sha(L.TESTS),
                      "binding": "disclosure only, not an input"}},
        "attempt_policy": {"scored_runs": "decidability once, score once (mode 'x' each)",
                           "engineering_retries": "free before any statistic, every attempt "
                                                  f"appended to {L.REPORT_DIR}/"
                                                  f"{L.ATTEMPTS_NAME}",
                           "exploratory_followups_max": 1,
                           "followup_alternatives": "only the report-only forms already "
                                                    "declared (one-contrast, raw-level, "
                                                    "per-county)",
                           "thresholds_frozen": True},
        "rulings": {"R8.7": "page line, ledger, README; C30 split; post-2028 wording "
                            "pre-declared (sha in binds.ruling)"},
        "priors": "No direction is claimed. The physics forecast is an average-weather free "
                  "run from 2022-12 (FM: start memory about 11 months); the layer is "
                  "climatology plus a decaying persisted anomaly. 'stat_encompasses' is the "
                  "likelier outcome if E02's seasonal shape adds nothing to the wells' own "
                  "climatology; its wording is pre-written.",
        "primary_model": {"phys": f"E02 frozen forward {L.E02_MODEL_SHA[:8]} "
                                  f"({L.E02_DIR})", "stat": f"E04 hybrid layer ({L.E04_DIR})"},
        "thresholds": {"min_months": L.MIN_MONTHS, "min_wells": L.MIN_WELLS, "B": L.BOOT_B,
                       "ci": list(L.CI), "x_rms_min_m": L.X_RMS_MIN_M, "delta": L.DELTA,
                       "max_bad_replicates": L.MAX_BAD_REPLICATES, "g0": D.G0},
        "interpretation": "Which of two frozen forecasts carries the within-well first-year "
                          "anomaly information at the shared wells; not the level, not the "
                          "bands, not policy response, not any other year.",
        "written_by": L.PREREG,
        "zones": _na("no fit"), "burn_in": _na("no fit"), "folds": _na("prospective window"),
        "gate_scores": _na("one reading by declared rules; no weighted score"),
        "fresh_units": [L.UNIT], "inputs": inputs,
    }
    return json.loads(json.dumps(doc, default=O._jsonable))


# ----------------------------------------------------------------------------- checks
def blocking_probe(doc: dict, prereg_dir=None) -> list:
    """prereg3's blocking rule against the look log, with FE's external unit's members added
    exactly as write_prereg adds them (metadata only)."""
    from harness import prereg3

    d = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    idx = prereg3.membership_index(d)
    for _, ext in prereg3._external_dicts(doc):
        for u, v in ext.items():
            m = prereg3._declared_members(v)
            if m is not None and not idx.known(u):
                idx.members[idx.canon(u)] = m
    now = dt.datetime.now(dt.timezone.utc)
    return prereg3.blocking_looks(doc["tests"], None, d, idx, test_time=now)


def _git_problems(paths) -> list[str]:
    out = []
    for p in paths:
        tracked = subprocess.run(["git", "-C", str(D.REPO), "ls-files", "--error-unmatch", p],
                                 capture_output=True, text=True).returncode == 0
        clean = subprocess.run(["git", "-C", str(D.REPO), "diff", "--quiet", "HEAD", "--", p],
                               capture_output=True).returncode == 0
        if not tracked or not clean:
            out.append(f"{p} is not committed as-is (R7.commit_first)")
    return out


def _merged_problems() -> list[str]:
    """'Merged, not just committed': HEAD must be contained in origin/main."""
    r = subprocess.run(["git", "-C", str(D.REPO), "merge-base", "--is-ancestor", "HEAD",
                        "origin/main"], capture_output=True, text=True)
    if r.returncode != 0:
        return ["HEAD is not contained in origin/main (the code must be merged before "
                "registration)"]
    return []


def logged_look(unit: str, purpose: str, prereg_dir=None) -> bool:
    from harness import prereg3

    p = (Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR) / "looks.jsonl"
    return p.is_file() and any(lk.get("units") == [unit] and lk.get("purpose") == purpose
                               for lk in prereg3.read_chained(p))


def readiness(d: dict, prereg_dir=None, now=None, verify_npz: bool = False) -> list[str]:
    """Everything that must hold before the lead registers the draft (empty = ready)."""
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    if now >= dt.datetime.fromisoformat(L.REGISTER_BY):
        out.append(f"registration deadline {L.REGISTER_BY} has passed: too late")
    if now > dt.datetime.fromisoformat(L.GATING_ARM_BY):
        out.append("after 2026-11-25: registering now makes FE report-only (review B2)")
    for tag, sha in (("e02", L.E02_PRED_SHA), ("e04", L.E04_PRED_SHA)):
        if d["binds"][tag]["predictions.csv"]["sha256"] != sha:
            out.append(f"{tag} predictions.csv missing or not the registered one")
    if d["binds"]["e04"]["manifest.json"]["sha256"] != L.E04_MANIFEST_SHA:
        out.append("E04 manifest.json missing or changed")
    if d["binds"]["e04"]["protocol.json"]["sha256"] != L.E04_PROTOCOL_SHA:
        out.append("E04 protocol.json missing or changed")
    if d["binds"]["e02"]["protocol.json"]["sha256"] != L.E02_PROTOCOL_SHA:
        out.append("E02 protocol.json missing or changed")
    b = d["binds"]
    for name, rec in (("FV.stat.v2 registration", b["fv_stat_v2"]["declaration"]),
                      ("FV.stat.v2 report", b["fv_stat_v2"]["report"]),
                      ("FV.decidability.v2 registration", b["fv_decidability_v2"]
                       ["declaration"]),
                      ("FV.decidability.v2 report_worlds", b["fv_decidability_v2"]
                       ["report_worlds"]),
                      ("FV.decidability.v1 registration", b["fv_decidability_v1"]
                       ["declaration"]),
                      ("noise_ar1.csv", b["fv_decidability_v1"]["noise_ar1"]),
                      ("obs_mask.csv", b["fv_decidability_v1"]["obs_mask"]),
                      ("wells_158.csv", b["wells_158"]),
                      ("GOV.heads_window_product.v1 registration", b["gov"])):
        if rec["sha256"] != rec["expected_sha256"]:
            out.append(f"{name} missing or changed")
    fv = b["fv_stat_v2"]
    if fv["e04_reading"] not in ("not_shown_weak", "weak_by_design"):
        out.append("FV.stat.v2 has no evaluable reading: FE registers report-only")
    if not fv["generator_code_unchanged"]:
        out.append("FV.stat.v2's generator code differs from its registration")
    batches = d["decidability"]["fv2_batches"]
    want = {f"{t}_b{i}" for t in D.TRUTHS for i in range(V2.PLAN.n_batch)}
    if set(batches) != want:
        out.append(f"FV v2 batches missing from attempts.jsonl: {sorted(want - set(batches))}")
    elif verify_npz:
        bad = [k for k, r in batches.items() if _sha(r["path"]) != r["npz_sha256"]]
        if bad:
            out.append(f"FV v2 batch npz differ from their attempts sha256: {bad}")
    if _sha(L.ZR0_NPZ) != L.ZR0_SHA and verify_npz:
        out.append("fv_zr0.npz missing or changed")
    rb = b["ruling"]
    if rb["sha256"] != L.RULINGS_SHA or not str(rb["decision"]).startswith("Approved"):
        out.append("ruling R8.7 missing, changed or not approved")
    for r in b["read_only_registrations"]:
        if r["sha256"] == MISSING:
            out.append(f"{r['id']} is not registered")
    if len(d["tests"][0]["external_units"][L.UNIT]["ids"]) != L.N_SHARED:
        out.append("the unit does not hold the shared wells")
    for p in (L.REPORT, L.DECIDABILITY):
        if O.rp(p).exists():
            out.append(f"{p} exists before the registration")
    out += [f"frozen code {p} missing" for p, h in d["frozen_code"].items() if h == MISSING]
    out += [f"input {p} missing" for p in d["inputs"] if not O.rp(p).exists()]
    out += _git_problems(list(D.FROZEN))
    out += _merged_problems()
    for look in FE_LOOKS:
        if not logged_look(*look, prereg_dir):
            out.append(f"look {look[0]} is not logged")
    if d["frozen_code"].get(D.GOV, "")[:8] != GOV_CODE_SHA8:
        out.append(f"{D.GOV} is not the registered GOV module ({GOV_CODE_SHA8})")
    if D.index_record(L.ID, prereg_dir):
        out.append(f"{L.ID} is already registered")
    try:
        hits = blocking_probe(d, prereg_dir)
    except Exception as err:  # noqa: BLE001
        out.append(f"blocking probe failed: {err}")
    else:
        if hits:
            out.append(f"looks block the unit: {hits[:5]}")
    return out


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True, help="the lead's seed")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verify-npz", action="store_true", help="hash the 21 forward npz (slow)")
    a = ap.parse_args(argv)
    try:
        d = draft_prereg(a.seed)
    except (L.EngineeringFailure, L.Refused) as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 1
    from harness import prereg3

    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", sha_text(d))
    print("validate_prereg:", errs or "ok")
    print("readiness:", readiness(d, prereg_dir, verify_npz=a.verify_npz) or "ready to register")
    print("this script sha256:", O.sha256(Path(__file__)))
    print("not written: the lead registers with prereg3.write_prereg after review")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
