"""Draft declaration of C14f.coverage_clause.v1 (harness.prereg3 schema, kind 'declaration'),
built in code because dataprep_staging/ ignores *.json. Nothing here writes to the registry.

STATUS: draft (user rulings R11.7, R11.7a-d). The lead registers it with
harness.prereg3.write_prereg and runs it from the SAME build checkout at merged main (r7, which
links results/twin/prospective_2026-11_ext, or r7b after that symlink is added), only after this
file and run_c14f.py are merged as-is, using the precedent rel_to_repo wrapper: patch only
harness.prereg3.rel_to_repo to a non-resolving version, so the INDEX path, units_registry and
ledger source are repo-relative while inputs under symlinked results/* stay absolute (as for
E02z, PB and FW); DATAPREP_REPO is not set. ``readiness`` lists what is still open.
No statistic is computed here: the only things read are code sha256s, FV.ext.v1's
declaration (frozen_code) and report (sha256 only),
the rulings files (the cited rulings) and the export protocol (rule fields).

  prereg_c14f.py [--quiet] [--write PATH]   prints the draft, its sha256, validate_prereg and
                                            readiness; --write saves the draft JSON (a local,
                                            git-ignored file; never under results/)
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

import run_c14f as R  # noqa: E402  (puts the FV and staging directories on sys.path)

O = R.O  # noqa: E741  (osse_fv, frozen)
XF = R.XF
PENDING = "PENDING"
MISSING = "MISSING"
#: N3F's window opens 2026-11 and its data start arriving in 2026-12; the reading reads no
#: window data, but a reading registered after N3F is scored would be after the fact
TOO_LATE = dt.datetime(2027, 11, 1, tzinfo=dt.timezone.utc)

#: the code history this declaration binds (frozen_code pins the bytes)
CODE_NOTE = ("Code 51d7908 (rebased onto origin/main 0941ba6 as a65b190, content identical) "
             "plus review fixes b6145ae; worlds, seed, streams, K, nulls and thresholds "
             "unchanged (the fixes: outcome decided on integer pass counts as exact fractions, "
             "R11.7a-d cited and hashed, registration-path note, design-time look recorded, "
             "3-decimal wording, weak-pair provenance, versions and atomic report write).")
PRIORS = (
    "Known before this design, disclosure only (nothing here chose K, the seed, the worlds, "
    "the null, the thresholds or the wording; all are FV.ext.v1's or R9.4's): FV.ext.v1's "
    "whole-rule P 0.592 (Q0 0.568, R0 0.544, R1c 0.620, X5R1 0.636) and null false pass 0.0, "
    "and its median pooled coverage per truth (Q0 0.946, R0 0.944, R1c 0.941, X5R1 0.936). On "
    "the same worlds the clause passes whenever the whole rule passes, so P >= 0.592 and "
    "'weak' can come only from P0 > 0.5. The medians sit just below the clause's upper bound "
    "0.95, so its binding side is over-coverage and P reaches 0.8 only if the across-world "
    "spread of pooled coverage is small; no stored output holds that spread or any per-world "
    "clause outcome. FV.ext.v1's null (truth at the registered climatology, RMS 1.25 m from "
    "the forecast) is expected to under-cover, so P0 is expected low. Expected outcome: "
    "'unchanged' or 'measured', no direction claimed; 'weak' is not expected.")
U7 = (
    "No observation after 2022-12 enters selection, calibration or design. The worlds are "
    "FV.ext.v1's (FV v2's registered full forwards read at N3F's 66 external wells, rain at "
    "climatology, pumping anomalies from the pre-origin pool); the noise sd, datum and "
    "registered baseline are N3F's export values derived from pre-origin heads (2012-22). "
    "N3F's window (2026-11..2027-10) is not read: no head file is opened, and the window "
    "heads are not yet available.")
MODEL_ONLY = (
    "Model-only: the reading reads model forwards (FV v2's truth batches, FV v1's truth "
    "forwards, ZR0, the registered forward datum_gate_innov), N3F's registered export "
    "(predictions.csv: forecast, interval, registered baseline; wells.csv: cell, layer, datum, "
    "residual sd; protocol.json) and FV v1's anomaly_pool.csv and noise_ar1.csv. It never "
    "reads an observed head, in the window or before it. Its result is a statement about N3F's "
    "coverage clause in the model's own worlds, never evidence about the aquifer.")
BUILDER = (
    "The builder (round 11, 2026-11-09) read code (ext_fv.py, osse_fv2.py, osse_fv.py, "
    "score_e02c.py annotate, e02_decid/*, harness.prereg3), FV.ext.v1's declaration and "
    "report.json in full (aggregates only: per-truth pass rates, median ratio and coverage, "
    "null false pass; no per-world value exists there), N3F's registration (tests, "
    "thresholds), the N3F export protocol.json, the C14f ledger row and progress-map entry, the "
    "rulings R11.7, R10.5, R9.4 and R9.4b, INDEX.jsonl and looks.jsonl rows naming FV.ext, "
    "U1.ext_transfer_power.v1's history disclosure and result.json key list (another estimand), "
    "and file names under results/twin_forward and results/twin_runs/fv2. It opened no npz, "
    "no predictions.csv or wells.csv row and no observation, and ran the reading only on "
    "synthetic worlds (ext_fv.build_synthetic).")
THRESHOLD_NOTE = [
    "'measured' needs P >= 0.8 AND P0 <= 0.2: the pair R10.5 applied to C14f on FV.ext.v1's "
    "whole-rule reading (R10.5's text states only this 'measured' pair).",
    "The 'weak' pair (P < 0.5 or P0 > 0.5; otherwise 'unchanged') comes from R9.4's "
    "pre-declared rule in the unregistered dataprep_staging/e02_decid/run_e02d1.py "
    "(E02.decidability.v1, never registered by R9.4b); this declaration is the first "
    "registered statement of the weak pair.",
    "Point estimates decide, compared as exact fractions of integer pass counts (P = total "
    "clause passes / (4 x 250), P0 = null passes / 500); floats and the MC SE are display only.",
    "P is pooled exactly as FV.ext.v1 pools its P (equal-weight mean over the four truths), so "
    "the clause's P is comparable with the 0.592 R10.5 graded.",
    "P0 uses FV.ext.v1's own null world (the one R10.5 read as 'null pass rate 0.0'); the "
    "midpoint null is report-only and never decides.",
]


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _sha(p) -> str:
    q = O.rp(p)
    return O.sha256(q) if q.is_file() else MISSING


def _json(p) -> dict | None:
    q = O.rp(p)
    return json.loads(q.read_text(encoding="utf-8")) if q.is_file() else None


def _ruling(path: str, rid: str) -> dict | None:
    d = _json(path)
    if not d:
        return None
    return next((r for r in d.get("rulings", []) if r.get("id") == rid), None)


def _reg(prereg_id: str, prereg_dir=None) -> dict | str:
    r = R.index_record(prereg_id, prereg_dir)
    return {"id": prereg_id, "sha256": r["sha256"], "created": r.get("created")} if r else PENDING


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def input_paths(P: R.ClausePaths) -> list[str]:
    """Every file the run reads, as given (repo-relative for the real paths); write_prereg
    hashes them. Code is pinned through frozen_code."""
    e = P.ext
    return sorted({*R.FROZEN, R.PREREG, P.fvext_decl, P.fvext_report,
                   *[f"{e.ext_dir}/{n}" for n in XF.EXT_FILES],
                   f"{e.v2.v1.inputs_dir}/anomaly_pool.csv", f"{e.v2.v1.inputs_dir}/noise_ar1.csv",
                   e.v2.v1.zr0, e.v2.v1.zr_forward, *e.v2.v1_truths.values(),
                   *[p for ps in e.v2.batches.values() for p in ps],
                   f"{e.v2.arms_dir}/arms.json", *[p for p, _ in P.rulings.values()]})


def draft_prereg(prereg_dir=None, P: R.ClausePaths | None = None) -> dict:
    """C14f.coverage_clause.v1's declaration (not written)."""
    P = P or R.ClausePaths()
    fv = _json(P.fvext_decl) or {}
    proto = _json(f"{P.ext.ext_dir}/protocol.json") or {}
    rulings = {rid: {"path": path, "sha256": _sha(path), "expected_sha256": sha,
                     "decision": (_ruling(path, rid) or {}).get("decision", PENDING)}
               for rid, (path, sha) in P.rulings.items()}
    doc = {
        "id": R.ID, "wp": R.WP, "kind": "declaration",
        "title": "C14f's coverage clause alone: how often N3F's pooled 90 % coverage clause "
                 "([0.85, 0.95]) passes in FV.ext.v1's own right-kind worlds (P) and null "
                 "worlds (P0) at the 66 never-calibrated external wells; R9.4's three outcomes; "
                 "post-failure exploratory, model-only",
        "label": R.LABEL,
        "status_note": f"Draft. {CODE_NOTE} C14f's grade moves only if a later user ruling "
                       "accepts this reading (R11.7, R11.7a).",
        "claim": {"id": "C14f", "ledger": "hydrophysics/twin/claims_v1.json",
                  "claim": "The frozen forecast's per-well 90 % head interval (combined spread, "
                           "datum and residual sd from each well's own pre-origin record) at "
                           "N3F's 66 never-calibrated external wells, 2026-11..2027-10",
                  "grade_now": "D, decidability unknown (R10.5: FV.ext.v1 whole-rule P 0.592 "
                               "< 0.8, null 0.0)"},
        "estimand": {
            "clause": R.CLAUSE_TEXT,
            "P": "equal-weight mean over FV.ext.v1's truths (R0, Q0, R1c, X5R1) of P(clause "
                 "passes | obs = ZR_ext + D_T + R_T,k + e_k), 250 worlds each, the worlds and "
                 "noise draws exactly FV.ext.v1's",
            "P0": "P(clause passes | obs0 = registered baseline + e_k), FV.ext.v1's 500 null "
                  "worlds",
            "interpretation": "the operating characteristic of N3F's coverage clause alone: "
                              "how often a right-kind model's interval meets it, and how often "
                              "it is met when the truth is the registered climatology"},
        "rule": {"protocol": f"{P.ext.ext_dir}/protocol.json",
                 "stated": {k: proto.get(k, PENDING) for k in R.N3F_RULE},
                 "declared": dict(R.N3F_RULE),
                 "n3f_registration": {"id": XF.N3F_ID, "test_id": "N3FP.ext_level_2026-11",
                                      "decision_rule_passed": "RMSE ratio <= 1.0 AND coverage "
                                                              "in [0.85, 0.95]",
                                      "not_informative": "fewer than 30 wells with >= 6 window "
                                                         "months"},
                 "check": "run_c14f.protocol_rule: the export protocol must state exactly the "
                          "declared rule, else engineering failure"},
        "worlds": {
            "source_declaration": R.FVEXT_ID, "registration": _reg(R.FVEXT_ID, prereg_dir),
            "declaration_sha256": _sha(P.fvext_decl),
            "expected_declaration_sha256": R.FVEXT_DECL_SHA,
            "report": P.fvext_report, "report_sha256": _sha(P.fvext_report),
            "expected_report_sha256": R.FVEXT_REPORT_SHA,
            "how": "FV.ext.v1's registered scorer ext_fv.score_ext runs unchanged (its "
                   "frozen_code pinned) with its own seed, streams, plan, K and declaration; a "
                   "pass-through wrapper of ext_fv.rule records each call's pass, ratio and "
                   "pooled coverage in memory (run_c14f.recording)",
            "gate_I": "the scorer's report on this run must equal FV.ext.v1's stored report on "
                      f"every key except {list(R.IDENTITY_SKIP)} (floats within "
                      f"{R.IDENTITY_TOL}); otherwise engineering failure, nothing written",
            "fresh": "nothing fresh: same forwards, same noise draws as FV.ext.v1 (by design: "
                     "the clause is read on the very worlds R10.5 graded; user ruling R11.7b)"},
        "null": {"primary": "FV.ext.v1's null: obs0 = registered baseline + e_k (stream 22, "
                            "500 worlds); decides (user ruling R11.7c)",
                 "report_only": "midpoint null obs = (ZR_ext + registered baseline) / 2 + e_k on "
                                "the same draws (E02.decidability.v1's equal-skill form); "
                                "reported, never decides"},
        "reused_code": {"from": R.FVEXT_ID, "pins": dict(R.REUSED),
                        "pins_match_declaration": (fv.get("frozen_code") or {}) == R.REUSED,
                        "current_files_match": not R.reused_code_problems(),
                        "functions": ["ext_fv.score_ext", "ext_fv.rule", "ext_fv.load_ext",
                                      "osse_fv.g0_e02", "osse_fv.simulate_ar1",
                                      "osse_fv.rng_for", "osse_fv._rate",
                                      "osse_fv2.load_registered", "osse_fv2.check_attempts"]},
        "design": R.design(),
        "readings": {"rule": R.OUTCOME_RULE,
                     "outcomes": {"measured": "P >= 0.8 and P0 <= 0.2",
                                  "weak": "P < 0.5 or P0 > 0.5 (when not measured)",
                                  "unchanged": "otherwise",
                                  "not_evaluable": "1.645 SE > 0.07 on P or P0, or a non-finite "
                                                   "estimate (mapped as unchanged)"},
                     "threshold_note": THRESHOLD_NOTE,
                     "wording": R.WORDING, "weak_why": R.WEAK_WHY},
        "ledger_mapping": {**R.LEDGER_MAPPING, "proposed_grade_keys": dict(R.PROPOSED_GRADE),
                           "page": "none"},
        "outputs": {
            "report": f"{R.REPORT} (written once: temporary file, then a no-clobber link)",
            "attempts": f"{R.REPORT_DIR}/{R.ATTEMPTS_NAME} (start / engineering_failure / "
                        "error / success; no statistic)",
            "report_only_note": "coverage quantiles, fail_side_share and the midpoint null "
                                "never change the outcome or the proposal",
            "report_keys": ["status", "outcome", "P (n_by_truth, n_total, k_total, exact, p, "
                            "mc_se, mc_half_width_90, by_truth, min_over_truths)",
                            "P0 (n, k, exact, p, mc_se, mc_half_width_90)",
                            "design_time_look (unit, new_look_logged False, ruling R11.7d, "
                            "logged_look)", "engineering.versions (python, numpy, pandas)",
                            "truth_worlds[T] (clause_passed, coverage_quantiles, "
                            "fail_side_share, whole_rule_passed_fvext)",
                            "null_world (clause_passed, coverage_quantiles, fail_side_share)",
                            "report_only.midpoint_null", "distance_in_mc_se", "mc_gate",
                            "wording (en, zh)", "ledger_mapping", "engineering", "known_before",
                            "declaration_sha256", "frozen_code"]},
        "gates": {
            "E": "this declaration registered and intact (INDEX.jsonl, load_prereg, frozen "
                 "code, inputs, design); FV.ext.v1 registered and intact at sha256 50108315 "
                 "(osse_fv2.load_registered with ext_fv.FROZEN); its report at sha256 "
                 "b29f6cfc; FV v2 batches the outputs of their last forward attempts; FV.ext.v1's "
                 "own gates inside score_ext (export shas, kept wells, arms, batch metadata, "
                 "hindcast identity, null tolerance, evaluate self-check); the export protocol "
                 "states N3F's rule; >= 30 wells x >= 6 months",
            "I": "identity with FV.ext.v1's report (above); every recorded pass equals N3F's "
                 "rule of its own ratio and coverage; a world's two calls share coverage; the "
                 "recorded calls reproduce the scorer's per-truth pass rates and median "
                 "coverage",
            "MC": {"rule": f"1.645 x Monte Carlo SE of P and of P0 <= {R.MC_HALF_WIDTH}, else "
                           "not_evaluable",
                   "note": "non-binding: largest possible half-widths 0.026 (P, 4 x 250) and "
                           "0.037 (P0, 500)"}},
        "attempt_policy": {
            "label": "post-failure, exploratory (retry rule, user rule 2026-10-23): the one "
                     "follow-up after R10.5's regrade of C14f; never validation",
            "engineering_retries": "free while gate E or I fails, before any clause statistic "
                                   "leaves memory; each attempt logged in attempts.jsonl",
            "scored_runs": "one (user ruling R11.7a; report.json written once through a "
                           "temporary file and a no-clobber link); the outcome stands as "
                           "computed",
            "failures": "reported (every attempt in attempts.jsonl and in the report's attempt "
                        "number)",
            "exploratory_followups_max": 0, "thresholds_frozen": True},
        "can_change": ["a proposal for C14f's D qualifier (D-measured, D-weak or unchanged "
                       "D-unknown), effective only through a later user ruling (R11.7)",
                       "C14f's ledger notes may cite the report"],
        "cannot_change": ["C14f's status (D)", "E22 or N3F (FV.ext.v1's whole-rule reading "
                          "stands)", "any page text", "progress_map_v1.json without a ruling",
                          "anything about the aquifer"],
        "u7": U7,
        "model_only": {"statement": MODEL_ONLY,
                       "asserted_by": f"{R.TESTS}::test_the_reading_is_model_only"},
        "runtime": "CPU only, no forward and no twin solver: FV.ext.v1's scorer reads about "
                   "10 GB of existing npz at 66 cells (one pass) and evaluates 2,500 + 500 rule "
                   "frames of 792 rows; minutes, a few GB of RAM",
        "rulings": rulings,
        "disclosures": {
            "builder": BUILDER,
            "after_the_fact_known": PRIORS,
            "design_time_look": "the run re-reads N3F's export values derived from pre-origin "
                                "heads (residual_sd_m, datum_m, baseline_m), FV.ext.v1's "
                                "logged look FV:ext_preorigin_derived_N3Fwells (2012-22, "
                                "design_time_evidence); spent months, no fresh unit; no new "
                                "look entry (R11.7d): the run checks that look is logged for "
                                "the export's files (ext_fv.look_logged) and records it in "
                                "report.design_time_look",
            "code": CODE_NOTE,
            "registration_paths": "register and run from the same build checkout at merged "
                                  "main (r7, which links results/twin/prospective_2026-11_ext, "
                                  "or r7b after adding that symlink), with the precedent "
                                  "rel_to_repo wrapper: only harness.prereg3.rel_to_repo is "
                                  "patched to a non-resolving version, so the INDEX path, "
                                  "units_registry and ledger source are repo-relative and "
                                  "inputs under symlinked results/* stay absolute (as for "
                                  "E02z, PB and FW); DATAPREP_REPO is not set",
            "tests": {"path": R.TESTS, "sha256_at_declaration": _sha(R.TESTS),
                      "binding": "disclosure only, not an input"}},
        "primary_model": {"forecast": "N3F's registered external predictions (forward "
                                      "cf2c7154)",
                          "truths": "FV.ext.v1's (FV v2 full forwards R0, Q0, R1c, X5R1)"},
        "priors": PRIORS,
        "thresholds": {**R.design()["thresholds"], "K_TRUTH": R.K_TRUTH, "K_NULL": R.K_NULL},
        "interpretation": "A model-only operating characteristic of N3F's coverage clause "
                          "alone. It proposes C14f's D qualifier and changes nothing until a "
                          "user ruling accepts it; it is not evidence about the aquifer.",
        "written_by": R.PREREG,
        "zones": _na("no fit"), "burn_in": _na("no fit"),
        "folds": _na("synthetic worlds; no folds"),
        "gate_scores": _na("one outcome by fixed thresholds; no weighted score"),
        "tests": _na("model-only declaration: synthetic worlds, no observation is scored and "
                     "no fresh unit is consumed"),
        "fresh_units": _na("no observation is scored"),
        "frozen_code": {p: _sha(p) for p in R.FROZEN},
        "inputs": input_paths(P),
    }
    return json.loads(json.dumps(doc, default=O._jsonable))


def _git_problems(paths) -> list[str]:
    out = []
    for p in paths:
        try:
            tracked = subprocess.run(["git", "-C", str(R.REPO), "ls-files", "--error-unmatch",
                                      p], capture_output=True, text=True).returncode == 0
            clean = subprocess.run(["git", "-C", str(R.REPO), "diff", "--quiet", "HEAD", "--",
                                    p], capture_output=True).returncode == 0
        except OSError as err:
            return [f"git: {err}"]
        if not tracked or not clean:
            out.append(f"{p} is not committed as-is (R7.commit_first)")
    return out


def readiness(d: dict, P: R.ClausePaths | None = None, prereg_dir=None, now=None) -> list[str]:
    """Everything that must hold before the lead registers the draft (empty = ready)."""
    P = P or R.ClausePaths()
    now = now or dt.datetime.now(dt.timezone.utc)
    out = []
    if now >= TOO_LATE:
        out.append("N3F's window has closed: a reading now would be after the fact")
    w = d["worlds"]
    if w["registration"] == PENDING:
        out.append(f"{R.FVEXT_ID} is not in the registry")
    if w["declaration_sha256"] != R.FVEXT_DECL_SHA:
        out.append(f"{R.FVEXT_ID}'s declaration is missing or not sha256 {R.FVEXT_DECL_SHA[:8]}")
    if w["report_sha256"] != R.FVEXT_REPORT_SHA:
        out.append(f"{R.FVEXT_ID}'s report is missing or not sha256 {R.FVEXT_REPORT_SHA[:8]}")
    if not d["reused_code"]["pins_match_declaration"]:
        out.append("run_c14f.REUSED differs from FV.ext.v1's frozen_code")
    if not d["reused_code"]["current_files_match"]:
        out.append(f"reused code differs from its pins: {R.reused_code_problems()}")
    if d["rule"]["stated"] != d["rule"]["declared"]:
        out.append("the export protocol states another rule than N3F's")
    for rid, r in d["rulings"].items():
        if r["sha256"] != r["expected_sha256"] or r["decision"] == PENDING:
            out.append(f"ruling {rid} not found at sha256 {r['expected_sha256'][:8]}")
    if O.rp(P.report).exists():
        out.append(f"{P.report} exists before the declaration")
    out += [f"frozen code {p} missing" for p, h in d["frozen_code"].items() if h == MISSING]
    out += [f"input {p} missing" for p in d["inputs"] if not O.rp(p).exists()]
    out += _git_problems([*R.FROZEN, R.PREREG])
    if R.index_record(R.ID, prereg_dir):
        out.append(f"{R.ID} is already registered")
    return out


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--write", help="save the draft JSON here (local; never under results/)")
    a = ap.parse_args(argv)
    d = draft_prereg(prereg_dir)
    from harness import prereg3

    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    if a.write:
        if R._under_results(a.write):
            print("refused: the draft is never written under results/", file=sys.stderr)
            return 1
        with open(a.write, "x", encoding="utf-8") as fh:
            fh.write(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
        print("wrote draft:", a.write)
    print("draft sha256:", sha_text(d))
    print("validate_prereg:", errs or "ok")
    print("readiness:", readiness(d, prereg_dir=prereg_dir) or "ready to register")
    print("this script sha256:", O.sha256(Path(__file__)))
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
