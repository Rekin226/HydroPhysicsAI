"""Declaration drafts of round-8 rank 16's FB part (harness.prereg3 schema). Nothing here
writes a registration: ``--validate`` prints the draft's canonical sha256, prereg3.
validate_prereg's errors and the readiness problems. The LEAD registers with
prereg3.write_prereg after review; the builder never called it.

  power  FB.family_power.v1 (kind 'declaration'): ``draft_prereg_power(seed)``. The synthetic
         power study (fb_lib), registered BEFORE it runs.
  bf     FB.family_bf_2026-12.v1 (kind 'test'): ``draft_prereg_bf(pairs)``. The prospective
         Bayes-factor test of {R0,Q0} vs {R1c,X5R1}, ONLY for the family pair of a county
         that passed the registered power study (lead ruling B1), registered before
         2026-11-30T16:00Z, scored after N2C's report, ledger only (E29), never moving C20/C21.

    python -I dataprep_staging/fb_family/prereg_fb.py power --validate --seed <int>
    python -I dataprep_staging/fb_family/prereg_fb.py bf --validate --pairs R0Q0_vs_R1cX5R1@Changhua
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fb_inputs as FI  # noqa: E402
import fb_io as io  # noqa: E402
import fb_lib as L  # noqa: E402
import power_fb as PW  # noqa: E402
import score_fb as SC  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
THIS = f"{io.X}/prereg_fb.py"
TESTS = "tests/test_fb_family.py"
SCORER = SC.SELF
BF_FROZEN = (SCORER, f"{io.X}/fb_lib.py", f"{io.X}/fb_io.py", f"{io.X}/fb_inputs.py",
             "dataprep_staging/e02d/e02dlib.py", "dataprep_staging/e02d/power_e02d.py",
             "dataprep_staging/fv_decidability/osse_fv.py",
             "dataprep_staging/n2_policy/bootstrap_scorer.py", SC.GOV, SC.E02C_SCORER)
PLAN_ENTRY = ("round-8 plan #16 (docs/TWIN_ROUND8_PLAN.local.md): 'FB.family_power.v1: CPU "
              "power study only, using U1.decomp_power plus FV D_k. Register the BF test before "
              "2026-11-30 only for pairs that pass, scored after N2C, and never moving "
              "C20/C21.' Gate: 'FB power gate P(correct strong) >= 0.8, P(wrong) <= 0.05'. "
              "Lead rulings on the FB review: B1 family pair only; B2 no_later_than "
              "2028-01-31; B3 sigma_idio scale; S1 recipe-id names and pre-worded line; S2 "
              f"ledger only ({L.LEDGER_ROW}).")
NOT_STACKING = (
    "FB is not the killed route 'Bayesian stacking of 6 Changhua members and the 0-1.3x lever "
    "spread row' (round-8 plan, killed routes): (1) stacking fits member weights to "
    "maximise in-sample predictive density on the wells and months the members were fitted "
    "on (2012-22); FB fits nothing: two pre-declared recipe families {R0,Q0} and {R1c,X5R1} "
    "with equal prior odds, equal member weights and a uniform prior over U1's registered "
    "noise grid, scored once on fresh months 2026-12..2027-10 that no theta, column or noise "
    "parameter has seen; (2) stacking outputs a weighted Changhua lever and a spread row for "
    "the page before N2C is scored, which N2C's before_scoring clause and R7.changhua_page "
    "forbid; FB outputs no lever, size, factor, forecast or member weight, shows nothing on the "
    "page in any branch, is scored only after N2C's report exists, and writes one ledger line "
    f"({L.LEDGER_ROW}) that never moves C20/C21 and never reads as a response class; (3) FB is "
    "gated by a pre-registered synthetic power study and a family pair that fails is never "
    "registered")
BUILDER_READS = [
    "docs/TWIN_ROUND8_PLAN.local.md (entry #16, judge's summary, findings on FV v2 and "
    "CB #10, killed routes, execution notes)",
    "results/dataprep/preregistration/U1.decomp_power.v1.json (design, derivation, decision; "
    "the registered grid, groups, sub-groups and relative scales)",
    "results/dataprep/preregistration/FV.decidability.v2_postfailure.json (design, readings, "
    "thresholds, truths incl. v1_forward sha256, disclosures, frozen_code)",
    "results/dataprep/preregistration/FV.decidability.v1.json (inputs_frozen, zr0, input "
    "list with sha256)",
    "results/dataprep/preregistration/N2C.changhua_response.prospective_2026-11.v1.json "
    "(page_mapping incl. before_scoring and never, scoring incl. same_session, tests' units)",
    "results/dataprep/preregistration/E02d.decomposed_prospective_2026-12.v1.json (tests' unit "
    "layout; a grep for sigma_idio_A in its inputs/files: path and sha256)",
    "results/dataprep/preregistration/INDEX.jsonl (sha256 and created of U1, FV v1/v2, N2C, "
    "E02c)",
    "results/dataprep/rulings/user_rulings_2026-11-03.json (ruling ids and decisions; none "
    "concerns FB) and a grep hit for R7.changhua_page in user_rulings_2026-11-02.json",
    "header lines only: DR.relative_drift.v1/wells_e02d.csv, E02c/baselines/wells_158.csv, "
    "FV.decidability.v1/inputs/obs_mask.csv, twin/e02d_relative/sigma_idio_A.csv",
    "a grep for 'fv_zr0' in results/twin_runs/fv_decidability/attempts.jsonl (no hit)",
    "sha256 only (no value): results/twin_forward/fv_zr0.npz (a1833d5f) and every input",
    "directory listings: results/twin_forward/fv_truth_*.npz, fv_zr0.npz, "
    "datum_gate_innov.npz (sizes), results/dataprep/reports/ (names), FV v1 inputs/, "
    "results/twin/e02d_relative/",
    "NOT read: any head, residual, sd_fit_rel, sigma_idio or datum value, any forward's heads, "
    "U1's or FV's result/report files, FV v2's sealed stdout, any 2023+ observation"]
DISCLOSURES = {
    "known_to_designers": "FV v2's per-truth Changhua E02c coverage split (Q0 0.004, R0 0.012, "
                          "R1c 0.992, X5R1 0.996; published in the round-8 plan and F2) implies "
                          "the Changhua departures differ materially between the two families; "
                          "this motivated the family pair. No D_k value, separation or power "
                          "number was computed or read before this declaration",
    "optimism": "working independence between wells in the likelihood (the worlds carry U1's "
                "sub-group shocks); hypothesis means at climatology pumping and rain (the "
                "realised-forcing arm adds FV's Yunlin pumping response only; no rain-year "
                "term, no Changhua pumping anomaly); the truths share the model family (FV v2's "
                "disclosure)",
    "pessimism": "U1's grid is bounded upward by DR's bootstrap upper-90 % plus margins; the "
                 "scale is E02d's sigma_idio (ensemble and drift included); the gate takes the "
                 "worst truth x worst gating row, never a pooled rate",
    "wrong_side": L.WRONG_SIDE_DISCLOSURE,
    "scale_cells": f"report-only worlds at x{list(L.SCALE_CELLS)} sigma_idio with the "
                   "likelihood unchanged show the cost of an understated scale; they do not gate",
    "plug_in_mean": "hypothesis means are computed on the realised mask (the relative "
                    "operator's medians see the same wells as the data); the median is still "
                    "nonlinear in mu + x, which the worlds measure",
    "offset": "mu_k = FV's D_k (against the same-code page reference ZR0) + FV's ZR0 -> ZR "
              "offset (code drift and IC-member average), because the 2028 scorer subtracts "
              "E02's REGISTERED forecast; the offset is common to every hypothesis; its RMS is "
              "reported (report-only)",
    "window": "FV's departure covers 2026-11..2027-10; FB uses 2026-12..2027-10 (the round-8 "
              "rule for registrations created after 2026-11-01)",
    "multiplicity": "only the family pair {R0,Q0} vs {R1c,X5R1} can gate or register, at most "
                    "once per county (two tests at most); the six single-recipe pairs are "
                    "computed report-only and are never registered; the two counties are read "
                    "separately, never combined, and P(wrong) is per county"}


def _check_seed(seed) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int (no default)")
    return seed


# ----------------------------------------------------------------------------- power
def draft_prereg_power(seed: int, prereg_dir=None, *, u1: tuple | None = None,
                       input_sha: dict | None = None) -> dict:
    """FB.family_power.v1 (declaration). ``seed`` is required (no default). ``u1`` (doc,
    sha) and ``input_sha`` are for tests only: any injection turns the id into TEST.*."""
    _check_seed(seed)
    injected = u1 is not None or input_sha is not None
    u1_doc, u1_sha = u1 if u1 is not None else io.registered(L.U1_ID, prereg_dir)
    shas = input_sha if input_sha is not None else FI.input_sha()
    design = L.design_spec(u1_doc, u1_sha, shas, seed)
    n = {c: {"working": design["counties"][c]["n_working_cells"],
             "world": design["counties"][c]["n_world_cells"],
             "gating_world": design["counties"][c]["n_gating_world_cells"]}
         for c in L.COUNTIES}
    pid = ("TEST." if injected else "") + L.POWER_ID
    return {
        "id": pid, "wp": L.WP, "kind": "declaration",
        "title": "Synthetic power study of a prospective Bayes-factor test between the recipe "
                 "families {R0,Q0} and {R1c,X5R1} on 2026-12..2027-10 at E02's wells, per "
                 "county; U1.decomp_power.v1's noise structure at E02d's sigma_idio scale plus "
                 "FV D_k; CPU only",
        "why": "Round-8 rank 16 (FB). A BF test is registered only for a county whose family "
               "pair is decisive by design: P(correct strong) >= "
               f"{L.P_CORRECT_MIN} and P(wrong strong) <= {L.P_WRONG_MAX} at BF >= "
               f"{L.B_STRONG:g}, worst truth x worst gating row. Registered before any "
               "computation.",
        "plan_entry": PLAN_ENTRY,
        "design": design, "n_cells": n,
        "hypotheses": {"truths": list(L.TRUTHS),
                       "families": {k: list(v) for k, v in L.FAMILIES.items()},
                       "gating_pair": L.FAMILY_PAIR,
                       "report_only_pairs": [p for p, _, _ in L.PAIRS if p != L.FAMILY_PAIR],
                       "names": "recipe ids only; no mechanism is attached to a family"},
        "derivation": {
            "noise_structure": f"{L.U1_ID}'s registered a_s, a_w, phi and dropout grid per "
                               "county, DR's groups and sub-groups; its world draw "
                               "(power_e02d.draw's structure and RNG order, fb_lib.draw_scaled)",
            "noise_scale": f"E02d's registered window-A sigma_idio ({L.E02D_ID}, "
                           "sigma_idio_A.csv, sha b70da8b0): per well-month, metres, "
                           "sd_fit_rel + ensemble + drift terms; gated for units and lead "
                           "consistency (fb_inputs.sigma_scale) after this registration",
            "means": "mu_k = osse_fv.departure(h_k, h_ZR, obs_mask) on 2026-12..2027-10: FV "
                     "v1's truth forward k (baseline) minus E02's registered forward, datum-"
                     "anchored on each well's observed 2012-22 months; = FV's D_k + FV's "
                     "ZR0 -> ZR offset",
            "forcing_arm": "osse_fv.basis of truth k's 58 Yunlin impulse arms times "
                           "osse_fv.draw_multipliers of FV v1's anomaly_pool.csv "
                           "(pump_electricity preorigin v3, design_time_evidence)",
            "statistic": "fb_lib module docstring and fb_lib.score_pair (one implementation "
                         "for this study and the 2028 scorer, score_fb.py)"},
        "decision_to_registration": {
            "rule": f"per county, the family pair {L.FAMILY_PAIR} only: pass iff min over "
                    f"truths x gating rows (arms {list(L.GATING_ARMS)}) of P(correct strong) "
                    f">= {L.P_CORRECT_MIN} AND max of P(wrong strong) <= {L.P_WRONG_MAX}",
            "pass": f"that county's family pair MAY enter {L.BF_ID}, registered before "
                    f"{L.REG_DEADLINE}",
            "fail": "never registered (no re-run, no new grid, no other B)",
            "truth_pairs": "report-only in every branch; never registrable (lead ruling B1)",
            "none": f"no BF test; {L.LEDGER_ROW} records 'family BF not decidable by design "
                    f"({L.POWER_ID})'; C20/C21 and every page line unchanged"},
        "bf_commitments": {
            "window": f"{L.WINDOW[0]}..{L.WINDOW[1]} ({L.T_WIN} months)",
            "scored": f"not before {L.NOT_BEFORE}, no later than {L.NO_LATER_THAN}, and only "
                      f"after N2C's report ({L.N2C_REPORT}) exists",
            "ledger": f"{L.LEDGER_ROW}; no page line in any branch",
            "precedence": L.PRECEDENCE,
            "never": "never moves C20/C21; no response-class reading (N2C alone decides); no "
                     "lever, size, factor, member weight or stacked forecast; no page line"},
        "not_the_killed_stacking_route": NOT_STACKING,
        "disclosures": DISCLOSURES,
        "builder_reads": BUILDER_READS,
        "smoke_disclosure": "the builder ran fb_lib on SYNTHETIC inputs only "
                            "(synthetic_u1 / synthetic_inputs, seeds "
                            f"{list(PW.BUILDER_SEEDS)}) in the test suite and timing smokes; "
                            "no real D_k, scale or power number was computed",
        "refused_seeds": list(PW.REFUSED_SEEDS),
        "self_checks": "per county, on the run's own design and stream 2: a county-month shock "
                       "cancels (1e-9 m); the vectorised log-likelihood with realised-mask "
                       "means equals a scalar loop; log BF antisymmetric; identical hypotheses "
                       "give 0. A failure is an engineering failure: nothing is written, the "
                       "attempt is logged",
        "code_sha256": PW.code_sha(),
        "run": {"command": f"python -I {io.X}/power_fb.py run --seed {seed}",
                "result": f"{PW.RESULT_DIR}/result.json", "attempts": PW.ATTEMPTS,
                "runs_once": True,
                "cost": "CPU only, about 50-70 minutes (2 counties x 4 truths x "
                        f"{n['Yunlin']['world']} world cells x {L.REPLICATES} replicates, "
                        "realised-mask means); 0 GPU-h"},
        "attempt_policy": "run once under this registration; engineering failures (input "
                          "identity, sigma_idio gate, forward metadata, self-check) write "
                          "nothing, are logged in attempts.jsonl (crashes too) and may be "
                          "retried; no exploratory follow-up; a change to bound code needs a "
                          "new id",
        "zones": NA("synthetic draws"), "priors": NA("synthetic draws"),
        "burn_in": NA("synthetic draws"), "folds": NA("synthetic draws"),
        "tests": NA("declaration of a synthetic study; no observation is scored"),
        "fresh_units": NA("no observation is involved"),
        "thresholds": design["thresholds"], "gate_scores": NA("no promoting gate"),
        "primary_model": NA("no fitted model: FV's truth forwards as hypothesis means and U1's "
                            "synthetic noise at E02d's scale"),
        "inputs": sorted({f"results/dataprep/preregistration/{L.U1_ID}.json",
                          f"results/dataprep/preregistration/{L.E02D_ID}.json",
                          f"results/dataprep/preregistration/{L.FV1_ID}.json",
                          f"results/dataprep/preregistration/{L.FV2_ID}.json",
                          f"results/dataprep/preregistration/{L.N2C_ID}.json",
                          *FI.INPUT_FILES.values(), THIS, TESTS, *PW.CODE_FILES}),
        "interpretation": "Decidability by design of a recipe-family Bayes-factor test on the "
                          "2026-12..2027-10 window; it estimates nothing from heads and decides "
                          "nothing about C20/C21.",
        "written_by": THIS}


def power_problems(doc: dict, prereg_dir=None) -> list[str]:
    bad = []
    if doc["id"].startswith("TEST."):
        bad.append("a TEST draft (injected inputs) is never registrable")
    for pid in (L.U1_ID, L.E02D_ID, L.FV1_ID, L.FV2_ID, L.N2C_ID):
        if io.index_record(pid, prereg_dir) is None:
            bad.append(f"{pid} is not registered")
    if io.index_record(L.POWER_ID, prereg_dir) is not None:
        bad.append(f"{L.POWER_ID} is already registered")
    why = PW.seed_problem(doc["design"]["seed"])
    if why:
        bad.append(why)
    if io.repo_path(f"{PW.RESULT_DIR}/result.json").exists():
        bad.append("the power result already exists")
    if io.now() >= io.parse_utc(L.REG_DEADLINE):
        bad.append(f"the BF registration deadline {L.REG_DEADLINE} has passed")
    if not doc["id"].startswith("TEST."):
        try:
            u1, _ = io.registered(L.U1_ID, prereg_dir)
            fv2, _ = io.registered(L.FV2_ID, prereg_dir)
            e02d, _ = io.registered(L.E02D_ID, prereg_dir)
            bad += FI.upstream_pin_problems(doc["design"]["input_sha256"], u1, fv2, e02d)
            bad += PW.bound_problems(u1, fv2)
        except io.Refused as exc:
            bad.append(str(exc))
    return bad


# ----------------------------------------------------------------------------- bf
def _slug(county: str) -> str:
    return county.lower()


def bf_unit(county: str, n: int) -> str:
    return f"FB:heads_{_slug(county)}_2026-12_2027-10_wells{n}"


def draft_prereg_bf(pairs: list[str], prereg_dir=None, *, power: dict | None = None,
                    wells: dict | None = None) -> dict:
    """FB.family_bf_2026-12.v1 (test) for the family-pair candidates ``pairs`` (keys
    'R0Q0_vs_R1cX5R1@<county>', each of which must have PASSED the registered power study;
    ``bf_problems`` checks that). Any other key is refused (lead ruling B1). ``power``
    ({path, sha256, declaration_sha256}) and ``wells`` ({county: ids}) are for tests only:
    any injection turns the id into TEST.*."""
    if not pairs:
        raise ValueError("no candidate: the BF test is registered only for passing pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate candidate")
    cands = [L.candidate(k) for k in pairs]
    L.require_family(pairs)
    injected = power is not None or wells is not None
    if power is None:
        rp = io.repo_path(f"{PW.RESULT_DIR}/result.json")
        rec = io.index_record(L.POWER_ID, prereg_dir)
        power = {"path": f"{PW.RESULT_DIR}/result.json",
                 "sha256": io.sha256(rp) if rp.exists() else "MISSING",
                 "declaration_sha256": rec["sha256"] if rec else "MISSING"}
    ids = wells if wells is not None else FI.county_ids()
    tests, units = [], []
    for c in cands:
        unit = bf_unit(c["county"], len(ids[c["county"]]))
        units.append(unit)
        tests.append({
            "test_id": f"FBP.bf_{c['pair']}_{_slug(c['county'])}_2026-12",
            "data_scope": L.DATA_SCOPE, "fresh_units": [unit],
            "external_units": {unit: {
                "description": f"monthly heads {L.WINDOW[0]}..{L.WINDOW[1]} (flags 0-1) at the "
                               f"{len(ids[c['county']])} {c['county']} wells of E02's 158 "
                               f"(U1.decomp_power.v1's {c['county']} wells"
                               + ("; N2C's 74" if c["county"] == "Changhua" else "") + ")",
                "ids": list(ids[c["county"]]), "window": f"{L.WINDOW[0]}..{L.WINDOW[1]}"}},
            "candidate": c["key"], "county": c["county"],
            "hypotheses": {"A": c["A"], "B": c["B"]},
            "statistic": "log BF(A:B) = fb_lib.score_pair(obs - ZR, groups, mu, v, 11, A, B) "
                         "with mu and v from the registered power result (score_fb.py)",
            "reading": {
                "closer_to_R0Q0": f"log BF >= ln {L.B_STRONG:g}",
                "closer_to_R1cX5R1": f"log BF <= -ln {L.B_STRONG:g}",
                "not_decisive": "otherwise",
                "not_evaluable": f"< {L.MIN_WELLS} scored wells, or the window product fails "
                                 "the MANIFEST precondition"}})
    pid = ("TEST." if injected else "") + L.BF_ID
    return {
        "id": pid, "wp": L.WP, "kind": "test",
        "title": "Prospective Bayes-factor test between the recipe families {R0,Q0} and "
                 f"{{R1c,X5R1}} on {L.WINDOW[0]}..{L.WINDOW[1]} at E02's wells, for the "
                 f"counties whose family pair passed {L.POWER_ID}",
        "why": "Which family's climatology forwards the window's per-well relative head "
               f"pattern is closer to, where that is decisive by design. {L.LEDGER_ROW}; never "
               "a response-class or C20/C21 reading.",
        "plan_entry": PLAN_ENTRY,
        "candidates": [c["key"] for c in cands],
        "power_study": {"id": L.POWER_ID, **power,
                        "rule": "every candidate here is a family pair that passed "
                                "(bf_problems and score_fb check result.json)"},
        "tests": tests, "fresh_units": units,
        "primary_model": {
            "residual": f"obs - ZR, ZR = E02's registered predicted_m ({SC.E02_PRED}, sha256 "
                        f"{SC.E02_PRED_SHA}, datum included); model-only datum check: "
                        "predicted_m - h_ZR constant per well within 1e-6 m (count only)",
            "hypothesis_means": "mu_k from result.json (osse_fv.departure(h_k, h_ZR, obs_mask), "
                                "2026-12..2027-10, FV v1's truth forwards); the relative "
                                "operator and window mean on the realised mask",
            "variance": "v from result.json 'calibration' (U1 working cells at E02d's "
                        "sigma_idio scale), v * T / n for a well with n < 11 months",
            "marginal": "equal member weights within a family; uniform over U1's working cells",
            "code": f"fb_lib.score_pair wrapped by {SCORER} (frozen_code)"},
        "priors": {"hypotheses": "equal prior odds (BF = posterior odds)",
                   "members": "equal weights within a family; nothing is fitted",
                   "noise": "uniform over U1.decomp_power.v1's a_s x a_w x phi cells"},
        "thresholds": {"b_strong": L.B_STRONG, "min_wells": L.MIN_WELLS,
                       "min_months": L.MIN_MONTHS, "min_others": L.MIN_OTHERS},
        "readings": {k: f"{L.LEDGER_ROW}: {v}" for k, v in L.LEDGER_TEXT.items()},
        "precedence": L.PRECEDENCE,
        "page_mapping": {
            "ledger_row": L.LEDGER_ROW,
            "before_scoring": f"nothing on the page; {L.LEDGER_ROW} lists FB as 'registered, "
                              "scored not before N2C's report (2028)'",
            "after_scoring": f"{L.LEDGER_ROW} only, beside N2C's verdict; no page line in any "
                             "branch",
            "never": "no change to C20/C21; no response-class reading (N2C alone decides "
                     "Changhua response class); no Changhua lever, size, correction factor or "
                     "pumping volume; no member weights or stacked forecast; no page line; no "
                     "claim about the page model's own forecast skill"},
        "not_the_killed_stacking_route": NOT_STACKING,
        "scoring": {
            "not_before": L.NOT_BEFORE,
            "after": f"N2C's step-(2) report {L.N2C_REPORT} exists, names {L.N2C_ID} and "
                     "carries a pinned-scorer verdict (its sha256 is recorded in FB's report); "
                     "on Changhua wells no FB statistic is computed or read before it",
            "no_later_than": L.NO_LATER_THAN,
            "window_product": f"the governed heads window product ({SC.GOV_ID}; "
                              "product_grants_scorer AND covering_rule for 2026-12..2027-10), "
                              "flags 0-1, the feed N2/N2C use",
            "once": "scored once; report written with open mode 'x'; a session after "
                    f"{L.NO_LATER_THAN} is flagged as a breach"},
        "attempt_policy": "engineering failures before any statistic are fixed, logged and "
                          "re-run; each test is scored once; a not-decisive or not-evaluable "
                          "result is reported as such; no follow-up reuses this window",
        "multiplicity": DISCLOSURES["multiplicity"],
        "disclosures": DISCLOSURES,
        "frozen_code": {f: (io.sha256(io.repo_path(f)) if io.repo_path(f).exists()
                            else "MISSING") for f in BF_FROZEN},
        "u7": "no 2023+ observation enters the design: the means are model forwards, the noise "
              "is U1's DR-bounded structure at E02d's pre-registered scale; only the fresh "
              "2026-12..2027-10 heads score",
        "zones": NA("no zone split beyond DR's groups in the relative operator"),
        "burn_in": NA("no fit"), "folds": NA("no folds; one prospective window"),
        "gate_scores": NA("readings by one threshold; no weighted score"),
        "inputs": sorted({power["path"], f"results/dataprep/preregistration/{L.POWER_ID}.json",
                          f"results/dataprep/preregistration/{L.N2C_ID}.json", SC.E02_PRED,
                          FI.WELLS_158, FI.ZR_FORWARD, *BF_FROZEN, THIS, TESTS}),
        "interpretation": "A prospective comparison of two pre-declared recipe families on "
                          "fresh months; it supports no response class, lever or factor.",
        "written_by": THIS}


def bf_problems(doc: dict, pairs: list[str], prereg_dir=None) -> list[str]:
    bad = []
    try:
        L.require_family(pairs)
    except ValueError as exc:
        bad.append(str(exc))
    if doc["id"].startswith("TEST."):
        bad.append("a TEST draft (injected inputs) is never registrable")
    rec = io.index_record(L.POWER_ID, prereg_dir)
    if rec is None:
        bad.append(f"{L.POWER_ID} is not registered")
    rp = io.repo_path(f"{PW.RESULT_DIR}/result.json")
    if not rp.exists():
        bad.append("the power result does not exist")
    else:
        res = json.loads(rp.read_text(encoding="utf-8"))
        if res.get("smoke") or res.get("synthetic"):
            bad.append("the power result is a smoke")
        if rec is not None and res.get("declaration", {}).get("sha256") != rec["sha256"]:
            bad.append("the power result does not cite the registered declaration")
        for k in pairs:
            if not res.get("decisions", {}).get(k, {}).get("pass"):
                bad.append(f"{k} did not pass the power study")
    if io.index_record(L.N2C_ID, prereg_dir) is None:
        bad.append(f"{L.N2C_ID} is not registered")
    if io.index_record(L.BF_ID, prereg_dir) is not None:
        bad.append(f"{L.BF_ID} is already registered")
    fc = doc.get("frozen_code") or {}
    if "MISSING" in fc.values():
        bad.append("frozen_code has a MISSING file")
    if fc.get(SC.GOV) != SC.GOV_SHA:
        bad.append(f"{SC.GOV} is not at {SC.GOV_SHA[:8]}")
    if fc.get(SC.E02C_SCORER) != SC.E02C_SCORER_SHA:
        bad.append(f"{SC.E02C_SCORER} is not at {SC.E02C_SCORER_SHA[:8]}")
    if io.now() >= io.parse_utc(L.REG_DEADLINE):
        bad.append(f"the registration deadline {L.REG_DEADLINE} has passed")
    return bad


# ----------------------------------------------------------------------------- cli
def _report(doc: dict, problems: list[str]) -> None:
    errs = prereg3.validate_prereg(doc)
    print(json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True, default=str))
    print(f"draft canonical sha256 {io.sha_json(doc)}")
    print(f"validate_prereg: {errs or 'OK'}")
    print(f"readiness problems: {problems or 'none'}")
    print("nothing written (the lead registers with prereg3.write_prereg after review)")


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("power")
    p.add_argument("--validate", action="store_true", required=True)
    p.add_argument("--seed", type=int, required=True)
    b = sub.add_parser("bf")
    b.add_argument("--validate", action="store_true", required=True)
    b.add_argument("--pairs", required=True, help="comma-separated family-pair keys")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "power":
            doc = draft_prereg_power(a.seed, prereg_dir)
            _report(doc, power_problems(doc, prereg_dir))
        else:
            pairs = [x.strip() for x in a.pairs.split(",") if x.strip()]
            doc = draft_prereg_bf(pairs, prereg_dir)
            _report(doc, bf_problems(doc, pairs, prereg_dir))
    except (io.Refused, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
