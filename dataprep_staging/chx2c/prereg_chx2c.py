"""Draft declaration of I1.CHX.v1.twocontrast (harness.prereg3 schema, kind 'declaration'),
built in code because dataprep_staging/ ignores *.json. Nothing here writes to the registry
and nothing here reads an observation or computes a statistic: the lead reviews the draft,
commits the code (R7.commit_first) and registers it with prereg3.write_prereg.

  prereg_chx2c.py           prints the draft JSON, its sha256, validate_prereg and readiness
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import chx2c_lib as C  # noqa: E402
import chx_io as IO  # noqa: E402

CODE_ROOT = C.STAGING.parent
MISSING = "MISSING"
#: sha256 the parent's own records give (report.json prereg_sha256 / panel_sha256)
PARENT_PREREG_SHA = "794f6a7ef70686f2b773a76aba7228660c75871eb59c5e63711a3c98d2190e53"
PANEL_SHA = "53726565f993b2bf294ab9bfbc2911104bb73712b2679c6f41cc0210003167fc"

BUILDER = (
    "The round-8 rank-8 builder (2026-11-03) read code, registrations and published reports "
    "only: CLAUDE.md, docs/TWIN_ROUND8_PLAN.local.md (rank #8 and the audit finding '[medium] "
    "Changhua (E25, I1.CHX.v1 placebo and looks)'), results/dataprep/rulings/"
    "user_rulings_2026-11-03.json, I1.CHX.v1's declaration and its report.json (published "
    "primary, placebo and report-only numbers, informative gates, post_registration_looks), "
    "chx_panel.json (metadata: shas, arms, key shapes, cell counts), the array HEADERS of "
    "chx_panel.npz (key names, dtypes and shapes; no array value was loaded), INDEX.jsonl "
    "(id, kind, created, test_ids), and of looks.jsonl: the purpose counts over all rows; the "
    "fields time, wp, purpose, units, data_scope, prereg_id and test_id (NOT the note or "
    "members) of the 'after_the_fact' rows printed from 2026-10-12T10:32Z to "
    "2026-10-17T02:35Z (W2W4, P1, S2, S3, S4b, H2, S1b, H3, P2abc, P3, P2def, H4); three rows "
    "WITH their notes and only the keys of their members: X5 post_failure_followup "
    "2026-10-25T14:56:42Z, X5 post_failure_exploratory 2026-10-25T14:59:15Z and I1 score "
    "2026-11-01T14:27:29Z (I1.CHX.v1's scoring look); and the raw first 600 bytes of the file "
    "(the P1 design_time_seed row of 2026-10-12T05:44Z with its note, and the start of the next "
    "row, census_yunlin 2025-07..2026-06). Also the declarations "
    "I1.X5R1.v2_postfailure, FV.decidability.v2_postfailure and "
    "X5.t1_postfailure_gnss_leveling.v1 (format precedents), and the code of ch_x, "
    "n2_policy/bootstrap_scorer.py, harness.prereg/prereg2/prereg3, fv_stat_v2/prereg_v2.py "
    "and x5_insar_anchor/followup_t1.py. It opened no observed head, computed no correlation "
    "of the two contrasts and no statistic on real data, and ran only synthetic tests.")

PRIORS = (
    "Seen before writing (I1.CHX.v1 report.json, published 2026-11-01T14:27Z): primary "
    "(twoway) beta 0.167 [0.024, 0.437], verdict direction_supported_size_not (size clause "
    "failed: point outside [0.67, 1.5], RMSE(PR) not below RMSE(ZR)); placebo (the same "
    "estimator on c_YL = A - C with e = obs - C) beta 1.01 [-0.16, 2.15], not_supported on "
    "precision only, its report-only detrended 2.61 [1.41, 3.93] and demeaned 1.24 "
    "[0.44, 3.32]; the transformed-contrast RMS over 23 x 120 is 0.259 m (c_CH) and 0.068 m "
    "(c_YL), both model-only. The correlation of the two contrasts was NOT computed. The "
    "round-8 planner puts 'not separable' at about 0.65: the Yunlin contrast's variation "
    "across screens is a quarter of Changhua's, the placebo point is 6 times the primary's, "
    "and a joint regression usually widens the Changhua interval, which already sits close "
    "to 0. 'Survives' is still possible if the two contrasts are nearly orthogonal across "
    "screens.")

INTERPRETATION = (
    "CHX's primary regressed obs - B on c_CH; under linear superposition that equals fixing "
    "the response to Yunlin's pumping pattern at the model's own scale (k_YL = 1). The "
    "placebo regressed obs - C on c_YL with the Changhua response fixed at the model's scale. "
    "Each therefore leaves the other contrast in its error. The joint regression frees both: "
    "'survives' says the observed response at the 23 screens still rises with R1c's "
    "Changhua-only contrast when a free Yunlin spill-over term is fitted beside it; 'not "
    "separable' says the 23 screens cannot tell Changhua pumping from Yunlin spill-over (or "
    "from a regional pattern whose spatial loading tracks either). Neither reading says "
    "anything about size; a confounder whose loading tracks both contrasts is still not "
    "excluded. Units are spent: this is after-the-fact and exploratory, it cannot promote.")


def sha_text(obj) -> str:
    return hashlib.sha256((json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def _sha_code(p: str) -> str:
    q = CODE_ROOT / p
    return IO.sha256(q) if q.is_file() else MISSING


def _sha_data(p: str) -> str:
    q = IO.rp(p)
    return IO.sha256(q) if q.is_file() else MISSING


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def _inputs() -> list:
    out = []
    for p, pinned in ((C.PARENT_PREREG, PARENT_PREREG_SHA), (C.PARENT_REPORT, None),
                      (C.PANEL, PANEL_SHA), (C.PANEL_SUMMARY, None), (C.RULINGS, None)):
        h = pinned or _sha_data(p)
        out.append({"path": p, "sha256": h} if h != MISSING else p)
    return out


def draft_prereg() -> dict:
    """The declaration (no 'created'; write_prereg adds it). Reads no observation."""
    frozen = {p: _sha_code(p) for p in (*C.OWN_FROZEN, *C.PARENT_FROZEN)}
    doc = {
        "id": C.ID, "wp": C.WP, "kind": "declaration",
        "title": "I1.CHX.v1's single exploratory follow-up (R8.10): does the direction pass at "
                 "the 23 spent Changhua screens survive a Yunlin spill-over control? A joint "
                 "regression on R1c's Changhua contrast and the Yunlin placebo contrast, CHX "
                 "estimator and bootstrap unchanged; narrowing only, cannot promote",
        "label": "after-the-fact, exploratory; I1.CHX.v1's single exploratory follow-up; "
                 "narrowing only; cannot promote",
        "followup_of": C.PARENT_ID,
        "ruling": {"id": C.RULING_ID, "file": C.RULINGS, "decision": "spend CHX's single "
                   "exploratory slot on the two-contrast narrowing (cannot promote); not on "
                   "CH-R"},
        "addresses": {
            "size_clause": "I1.CHX.v1's size clause failed (verdict "
                           "direction_supported_size_not: point 0.167 outside [0.67, 1.5]). "
                           "This follow-up does not re-test size and has no size reading; the "
                           "joint Changhua point beside the band is report-only",
            "placebo": "I1.CHX.v1's placebo was not_supported on precision only (point about "
                       "6 times the primary, report-only forms strongly positive), so it is "
                       "not evidence of specificity; this follow-up asks whether the "
                       "direction pass separates Changhua pumping from Yunlin spill-over",
            "not_a_primary_failure": "the parent's primary passed its direction clause; this "
                                     "is not a post-failure follow-up of the primary and it "
                                     "can only narrow E25's wording"},
        "cannot_promote": C.CANNOT_PROMOTE,
        "units": {"spent": C.UNIT, "scored_by_parent": "2026-11-01T14:27:29Z (score look)",
                  "window": C.SCOPE},
        "look": {"purpose": C.LOOK_PURPOSE, "units": list(C.LOOK_UNITS),
                 "data_scope": C.LOOK_SCOPE, "wp": C.WP, "prereg_id": C.ID,
                 "note": C.LOOK_NOTE,
                 "when": "score_chx2c.py look, run by the lead AFTER this declaration is "
                         "registered: it first computes the model-only collinearity (the "
                         "parent's order: score_chx decides from the panel alone before any "
                         "look); if |r| > 0.9 or r is not finite it writes the 'not separable' "
                         "report and logs NO look; otherwise it logs this look, before any "
                         "head is read. score refuses without the look. Heads are read once on "
                         "a successful run; a crash re-run, if one happens, is a second read "
                         "recorded in the attempts log"},
        "design": {
            "dependent": "e = obs - B (CHX's own: B = R1c with every Changhua cell at its 2012 "
                         "monthly pattern from 2013-01)",
            "regressors": {"changhua": "c_CH = A - B (panel c_ch; CHX's primary contrast)",
                           "yunlin": "c_YL = A - C (panel c_placebo_ch; the Yunlin placebo "
                                     "contrast at the same 23 screens)"},
            "rows": "CHX's: chx_lib.long_panel/eligible on the re-read heads (flags 0-1, "
                    "finite obs, >= 12 scored window months per screen), both regressors "
                    "attached on exactly those rows",
            "fixed_effects": f"'{C.FORM}' (chx_lib.fe_basis: per-screen intercept and linear "
                             "trend + month-of-sample effects), projected out of e and of BOTH "
                             "contrasts by exact least squares (chx_lib.fe_resid), then the "
                             "pinned per-well demeaning (bootstrap_scorer.within)",
            "estimator": "least squares of e~ on [c~_CH, c~_YL] from the summed per-well "
                         "cross-products; with the Yunlin column dropped it is "
                         "bootstrap_scorer.beta_response on the same rows (CHX's estimator, "
                         "unchanged)",
            "interval": "90 %, CHX's 5 km block bootstrap unchanged: CHX's blocks (07100111-141 "
                        "nest one block), B 2000, seed 20261101, default_rng(seed).integers(0, "
                        "n_blocks, (B, n_blocks)) over the sorted block ids; each draw solves "
                        "the 2 x 2 normal equations of the resampled block sums; draws with "
                        f"det <= {C.SINGULAR_DET_REL} x the product of the diagonal dropped "
                        "and counted; 5 % / 95 % quantiles (numpy default)",
            "offset_invariance": "under superposition the both-at-2012 offset is B + C - A and "
                                 "obs - B = (obs - (B + C - A)) - c_YL, so b_CH and its "
                                 "interval are identical with either offset; b_YL = k_YL - 1, "
                                 "and k_YL = b_YL + 1 is reported (report only)",
            "heads": "L1_clean v3 through chx_io.l1_monthly (CHX's loader: use "
                     "design_time_evidence, window_end 2022-12), the 23 screens, "
                     "2013-01..2022-12, after the look",
            "panel_keys_read": list(C.PANEL_KEYS),
            "panel_keys_note": "model output and metadata only; chx_panel.npz holds no "
                               "observation"},
        "gates": {
            "collinearity": {
                "declared": "before any computation, in this declaration; computed by "
                            "'look' after registration, before the look is logged",
                "statistic": "r = cosine of the two-way-transformed c_CH and c_YL over every "
                             "screen x window month (23 x 120, the full grid of CHX's "
                             "informative gate; = Pearson r, both have mean 0); model only, "
                             "reads no observation",
                "rule": f"|r| > {C.COLLINEAR_MAX} -> reading fixed to 'not separable' "
                        "without fitting and without reading a head",
                "scored_rows": f"after the heads are read, the same cosine on the scored rows "
                               f"(it depends on the observation mask, not on values): |r| > "
                               f"{C.COLLINEAR_MAX} also fixes 'not separable' (narrowing "
                               "only; the joint numbers are then report-only)",
                "singular_draws": f"more than {C.SINGULAR_SHARE_MAX:.0%} of the B draws "
                                  "singular -> 'not separable' (narrowing only; practically "
                                  "never fires once the model-only |r| <= 0.9)",
                "non_finite": "a non-finite r (model-only or scored-row) fails toward 'not "
                              "separable'; non-finite numbers are written as null"},
            "pregate": {
                "rule": "with the Yunlin regressor dropped, reproduce I1.CHX.v1's published "
                        f"primary beta 0.16689938 and its 90 % interval to {C.PREGATE_TOL} by "
                        "two routes: CHX's frozen score_chx.score_panel on the re-read heads "
                        "(which must also reproduce the published placebo to the same "
                        "tolerance) and this code's joint_fit on c_CH alone; the routes agree "
                        f"to {C.ROUTE_TOL}; rows 1205 and wells 23 identical",
                "when": "after the look and the heads (the 23 screens are spent units)",
                "on_failure": "the report says 'not evaluable', no joint coefficient is "
                              "computed, E25 keeps only the wording fix; final, no retry",
                "why_final": "the frozen route is the parent's own scoring function on the "
                             "parent's own loader, panel and code (all sha-bound), so a "
                             "frozen-route mismatch means an input changed, which no code fix "
                             "here may repair; the own route equals the frozen route exactly "
                             "(measured difference 0.0 in beta and both interval ends on the "
                             "synthetic test panels), so it cannot be the cause either; a code "
                             "change would also need a new declaration and the slot is spent"}},
        "readings": {
            "survives": {"rule": "the Changhua coefficient's 90 % interval lies wholly above 0 "
                                 "(lo > 0) and its point is > 0, every gate passed",
                         "e25_note": C.READING_SURVIVES},
            "not_separable": {"rule": "anything else past the pre-gate, or a collinearity / "
                                      "singular-draw rule above",
                              "e25_note": C.READING_NOT_SEPARABLE},
            "not_evaluable": {"rule": "the reproduction pre-gate failed",
                              "e25_note": C.READING_NOT_EVALUABLE},
            "report_only": ["the Yunlin coefficient b_YL and k_YL = b_YL + 1 with intervals",
                            "the Changhua point against the ratio band (no size reading)",
                            "the model-only and scored-row |r|", "singular draws",
                            "the transformed RMS of each contrast"],
            "one_sided": "no reading supports, promotes or adds an E-row; 'survives' is a "
                         "wording clause on E25 only"},
        "evidence_and_ledger_mapping": {
            "E25": {"status": C.E25_STATUS,
                    "survives": f"E25 notes gain '{C.READING_SURVIVES}'",
                    "not_separable": f"E25 notes gain '{C.READING_NOT_SEPARABLE}'",
                    "not_evaluable": "E25 notes keep the placebo rewording only"},
            "page": "E25 line gains the separability clause (page wording under R8.8)",
            "unchanged_whatever_the_outcome": ["E25's status", "C20/C21", "E26", "CH-J's "
                                               "registration and gates", "N2C"]},
        "outputs": {"report": C.REPORT, "attempts_log": C.ATTEMPTS,
                    "never_written": C.PARENT_REPORT},
        "attempt_policy": {
            "scored_runs": "one scoring session; twocontrast_report.json written once (mode "
                           "'x'); I1.CHX.v1's report.json is never written",
            "engineering_retries": "free while a step refuses before the heads are read "
                                   "(as the parent's); a crash AFTER the heads are read (no "
                                   "statistic printed, no report written) may be re-run at "
                                   "most once with unchanged code; that re-run is a second "
                                   "read of the same heads and its attempt says so. This "
                                   "departs from the parent's 'free before any head is read' "
                                   "because the pre-gate needs the heads; a pre-gate failure "
                                   "is final",
            "repair_ids": [],
            "attempts_log": f"{C.ATTEMPTS}: every invocation (start; refusal, failure or "
                            "collinear stop with its step and message; success with the "
                            "report sha256); never a statistic",
            "exploratory_followups_max": 0, "thresholds_frozen": True,
            "pre_declared_alternatives": [], "report": "every attempt, failures included",
            "followup_label": "after-the-fact, exploratory; cannot promote"},
        "disclosures": {
            "builder": BUILDER,
            "parent_post_registration_look": "I1.CHX.v1's report lists FV.ext's design-time "
                                              "read of the same screens' pre-origin residual "
                                              "sd (2026-11-01T14:19:55Z) as a "
                                              "post-registration look; it predates this "
                                              "follow-up and cannot have set anything here",
            "label_departure": "the parent pre-declared followup_label 'post-failure, "
                               "exploratory; cannot promote'. This follow-up is labelled "
                               "'after-the-fact, exploratory; ... cannot promote' because the "
                               "parent's primary did not fail (direction passed); it follows "
                               "the failed size clause and the not-supported placebo. The "
                               "'cannot promote' clause is kept unchanged",
            "not_an_alternative": "the parent's attempt_policy.pre_declared_alternatives is []; "
                                  "this is not an alternative to the parent's design but the "
                                  "parent's single exploratory slot "
                                  "(exploratory_followups_max 1), granted by user ruling R8.10",
            "spec_departures": {
                "abs_r": "the plan says 'corr(contrasts) > 0.9'; the rule uses |r| > 0.9, "
                         "which is stricter (a strongly negative r is equally inseparable) and "
                         "can only narrow",
                "pregate_value": "the plan's '0.1670 to 1e-6' is read as the published "
                                 "full-precision beta 0.16689938333659432 (and its interval) "
                                 "to 1e-6; 0.1670 is its rounding",
                "heads_use": "heads are read with use design_time_evidence ONLY to reproduce "
                             "the parent's read exactly (same loader, use and window_end); "
                             "L1_clean v3 allows both design_time_evidence and "
                             "after_the_fact_analysis. The reading is after-the-fact: "
                             "R7.changhua_page's 'S (design-time)' label does not carry over "
                             "to it"},
            "rulings_file": f"{C.RULINGS} is final (hash-bound by GOV, read-only); it is "
                            "hashed as an input",
            "tests": {"path": C.TESTS, "sha256_at_declaration": _sha_code(C.TESTS),
                      "binding": "disclosure only, not frozen"}},
        "priors": PRIORS,
        "interpretation": INTERPRETATION,
        "primary_model": {"model": "R1c (stage3_datum_gate_ccz), I1.CHX.v1's hindcast panel "
                                   "arms A, B, C, unchanged", "panel_sha256": PANEL_SHA},
        "thresholds": {"collinear_max_abs_r": C.COLLINEAR_MAX, "pregate_tol": C.PREGATE_TOL,
                       "route_tol": C.ROUTE_TOL, "singular_det_rel": C.SINGULAR_DET_REL,
                       "singular_share_max": C.SINGULAR_SHARE_MAX, "level": C.BS.LEVEL,
                       "B": C.BS.B_BOOT, "seed": C.BS.SEED, "bounds": "strict as written"},
        "zones": _na("no fit; the parent's arms at the 23 screens"),
        "burn_in": _na("no model run; the parent's hindcast panel"),
        "folds": _na("spatial screens, no folds (the parent's)"),
        "gate_scores": _na("a declared reading; no weighted score"),
        "tests": _na("after-the-fact exploratory narrowing on the spent unit "
                     f"{C.UNIT}; never validation; cannot promote"),
        "fresh_units": _na(f"spent unit {C.UNIT} (scored by {C.PARENT_ID})"),
        "frozen_code": frozen,
        "inputs": _inputs(),
        "written_by": C.PREREG,
    }
    return json.loads(json.dumps(doc))


def _git_problems(paths) -> list[str]:
    out = []
    for p in paths:
        try:
            tracked = subprocess.run(["git", "-C", str(CODE_ROOT), "ls-files",
                                      "--error-unmatch", p], capture_output=True,
                                     text=True).returncode == 0
            clean = subprocess.run(["git", "-C", str(CODE_ROOT), "diff", "--quiet", "HEAD",
                                    "--", p], capture_output=True).returncode == 0
        except OSError as err:
            return [f"git: {err}"]
        if not tracked or not clean:
            out.append(f"{p} is not committed as-is (R7.commit_first)")
    return out


def readiness(d: dict, prereg_dir=None) -> list[str]:
    """What must hold before the lead registers (empty = ready). Reads no observation."""
    from harness import prereg3

    pdir = Path(prereg_dir) if prereg_dir else prereg3.PREREG_DIR
    out = []
    for p, h in d["frozen_code"].items():
        if h == MISSING:
            out.append(f"frozen code {p} missing")
        elif p in C.PARENT_FROZEN and h != C.PARENT_FROZEN[p]:
            out.append(f"{p} differs from the sha256 I1.CHX.v1 froze")
    for x in d["inputs"]:
        if not isinstance(x, dict):
            out.append(f"input {x} missing")
    if _sha_data(C.PARENT_PREREG) != PARENT_PREREG_SHA:
        out.append("I1.CHX.v1's declaration is missing or changed")
    if _sha_data(C.PANEL) != PANEL_SHA:
        out.append("chx_panel.npz is missing or not the scored panel")
    rep = IO.rp(C.PARENT_REPORT)
    if rep.is_file():
        r = json.loads(rep.read_text(encoding="utf-8"))
        if r.get("prereg_sha256") != PARENT_PREREG_SHA or r.get("panel_sha256") != PANEL_SHA:
            out.append("the parent report does not name the declared prereg/panel")
    for p in (C.REPORT, C.ATTEMPTS):
        if IO.rp(p).exists():
            out.append(f"{p} exists before the declaration")
    idx = prereg3.read_chained(pdir / "INDEX.jsonl")
    if any(r["id"] == C.ID for r in idx):
        out.append(f"{C.ID} is already registered")
    import score_chx2c as S

    others = [i for i in S.followups_of_parent(pdir) if i != C.ID
              and i not in (d["attempt_policy"].get("repair_ids") or [])]
    pp = pdir / f"{C.PARENT_ID}.json"
    limit = (json.loads(pp.read_text(encoding="utf-8"))["attempt_policy"]
             ["exploratory_followups_max"] if pp.is_file() else None)
    if limit is None:
        out.append("I1.CHX.v1's declaration (exploratory_followups_max) is not readable")
    elif len(others) + 1 > int(limit):
        out.append(f"I1.CHX.v1 allows {limit} exploratory follow-up(s); registered: {others}")
    out += _git_problems([*C.OWN_FROZEN])
    return out


def main(argv=None, prereg_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    from harness import prereg3

    d = draft_prereg()
    errs = prereg3.validate_prereg(d, prereg_dir=prereg_dir or prereg3.PREREG_DIR)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", sha_text(d))
    print("validate_prereg:", errs or "ok")
    print("readiness:", readiness(d, prereg_dir) or "ready to register")
    print("not written: the lead registers with prereg3.write_prereg after review")
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
