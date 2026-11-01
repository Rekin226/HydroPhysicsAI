"""Declaration U1.no_band.v1 (harness.prereg3, kind 'declaration'): one merged narrowing for
every band no held or arriving data can calibrate, with a machine-checked reason per decision.

Round 5, rank 3 (U1-N + U1.SN + the spread-removal part of U1.CB, merged by the review). It
fixes ONE sub-claim scheme for the page's '90 % band' claim (ledger C14), used by every band
action:

- C14a  per-well registered 90 % head interval at the monitoring wells, in 2026-11..2027-10
- C14b  the same recipe outside that window
- C14c  aggregate run spreads (N2-unit mean, township, fan layer 2, cells without a well)
- C14d  head-contrast (policy-response) spreads on the answer tiles and series
- C14e  subsidence level and contrast bands

and records seven decisions, each with reasons a checker resolves against the files:
a MANIFEST path plus field (``manifest_use`` / ``manifest_field``), a prereg3 registration id
(in INDEX.jsonl with its sha) plus a result path (``registration``), a published result file
plus field (``result``, for post-hoc reports that have no registration), a code fact
(``code``, supplementary) or a ``context`` note (never sufficient alone). It is a narrowing,
not evidence: nothing is scored, no data value is read (manifests, registrations and report
metadata only), and the page share covered by a band is reported before and after.

Because it cannot change once registered, ``--validate`` also checks it against the ledger
and page it is cited by (:data:`PAGE_ALIGNMENT`, :func:`page_alignment_problems`): every
declared status equals the ledger's ``status_current``, every sub-claim and related claim
cites this id, and the declared texts appear in the ledger wording and in the template's
EN and ZH strings. Registration is refused until they agree.

User rulings 2026-10-29 (third set) (d): C14 is split into C14a-e; subsidence bands are
permanently Model internals; the paired run spread moves off the head-response tiles into
Model internals. Ruling (a) labels C14a's interval family with the U7 disclosure.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY -I dataprep_staging/u1_bands/prereg_u1_no_band.py --validate
    $PY -I dataprep_staging/u1_bands/prereg_u1_no_band.py --write --expect-self-sha <sha>

Register it before E02c (rank 4) is written and before any band-related score.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO / "dataprep_staging") not in sys.path:
    sys.path.insert(0, str(REPO / "dataprep_staging"))

from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
ID = "U1.no_band.v1"
THIS = "dataprep_staging/u1_bands/prereg_u1_no_band.py"
DEADLINE = dt.datetime(2026, 10, 31, 16, 0, tzinfo=dt.timezone.utc)
STATUSES = frozenset("SDNLW")


def status_ok(s) -> bool:
    """A status code or a '+'-joined combination of distinct codes (as the ledger writes)."""
    parts = str(s).split("+")
    return bool(s) and all(x in STATUSES for x in parts) and len(set(parts)) == len(parts)


def ledger_status(sc: dict) -> str:
    """The ledger status_current a declared (sub-)claim corresponds to."""
    return sc["status"] + (f"+{sc['internals_status']}" if sc.get("internals_status") else "")

MACHINE = ("manifest_use", "manifest_field", "registration", "result")
REASON_TYPES = MACHINE + ("code", "context")

DP = "results/dataprep"
REL = f"{DP}/model_ready/release_20261012b2"
E02 = "results/twin/prospective_2026-11"
A0 = f"{DP}/reports/N1_gaps_2026-10-24/A0"
M1 = f"{DP}/reports/N1_gaps_2026-10-24/M1"
LEDGER = "hydrophysics/twin/claims_v1.json"
TEMPLATE = "hydrophysics/twin/app/template.html"
N2C_LABEL = "untested (N2C registered, scored not before 2028-01-15)"
N2C_LABEL_ZH = "未經檢驗（N2C 已登錄，最早 2028-01-15 評分）"


def _mu(manifest: str, use: str, expect: str, **kw) -> dict:
    return {"type": "manifest_use", "manifest": f"{DP}/{manifest}/MANIFEST.json", "use": use,
            "expect": expect, **kw}


def _mf(manifest: str, field: str, **expect) -> dict:
    return {"type": "manifest_field", "manifest": f"{DP}/{manifest}/MANIFEST.json",
            "field": field, **expect}


def _ctx(text: str, source: str) -> dict:
    return {"type": "context", "text": text, "source": source}


# ----------------------------------------------------------------------------- reasons
SUBSIDENCE_SCORERS = [
    _mu("model_ready/release_20261012b2/preorigin/piles_printed/v1", "prospective_scorer",
        "forbidden", allowed_equals=["validator", "after_the_fact_analysis"],
        why="current-release pre-origin printed piles: validator and after_the_fact_analysis "
            "only (2026-10-18 ruling)"),
    _mu("model_ready/release_20261012b2/full/piles_printed/v1", "prospective_scorer",
        "forbidden", why="current-release full printed piles forbid prospective_scorer"),
    _mu("deep_benchmarks_printed/v2", "prospective_scorer", "unlisted",
        why="deep_benchmarks_printed v2 allows scoring_target and design_time_evidence but "
            "does not list prospective_scorer: not in allowed_uses, so not allowed "
            "(common/guard.py check_use); it is not 'forbidden'"),
    *[_mu(f"gnss_clean/full/v{v}", "prospective_scorer", "forbidden",
          why=f"gnss_clean full v{v} forbids prospective_scorer") for v in (1, 2, 3, 4)],
    _mf("gnss_clean/full/v4", "gate.passed", equals=False,
        why="gnss_clean full v4 did not pass its gate"),
    _mu("gnss_clean/preorigin/v4", "prospective_scorer", "forbidden",
        why="gnss_clean pre-origin v4 forbids prospective_scorer"),
    _ctx("GNSS over-reads fast sinking about 1.4x against leveling and steps by cm (vertical "
         "reference = leveling, 2026-10 decision)", "project memory: vertical reference "
                                                   "decision; not machine-checked"),
    _mu("subsidence_public_reports/postorigin/v1", "prospective_scorer", "forbidden",
        allowed_equals=["validator", "after_the_fact_analysis"],
        why="WRA township summaries after the origin: validator-only"),
    _mu("subsidence_public_reports/preorigin/v1", "scoring_target", "forbidden",
        allowed_equals=["validator", "after_the_fact_analysis", "design_time_evidence"],
        why="WRA township summaries before the origin: validator-only (no scoring_target)"),
    *[_mu(m, "prospective_scorer", "forbidden",
          allowed_equals=["validator", "after_the_fact_analysis"],
          why=f"{m}: validator-only") for m in ("insar_069A/v3", "insar_licsar_subsets/v1")],
    _mu("x5_insar_anchor/v1", "validator", "forbidden",
        allowed_equals=["after_the_fact_analysis"],
        why="InSAR anchor: after_the_fact_analysis only"),
    _mu("subsidence_fused/v5", "prospective_scorer", "forbidden",
        allowed_equals=["after_the_fact_analysis"],
        why="subsidence_fused v5: after_the_fact_analysis only"),
]
U7_RULE = _mf("subsidence_public_reports/postorigin/v1", "rule_u7",
              contains="never a gate, ladder decision, member choice, hyperparameter or "
                       "pre-registered threshold",
              why="U7: post-origin (2023-26) observations never calibrate or select anything, "
                  "so 2023-26 residuals cannot calibrate a band")
SUBSIDENCE_TESTS = [
    {"type": "registration", "id": "N1.policy_response.v1",
     "result": f"{DP}/reports/N1_policy/step_b/results.json",
     "result_field": "prereg", "result_equals": "N1.policy_response.v1",
     "why": "step B (dose, subsidence) result of the registered policy-response test"},
    {"type": "registration", "id": "N1.policy_response.v1",
     "result": f"{DP}/reports/N1_policy/step_c/results.json",
     "result_field": "prereg", "result_equals": "N1.policy_response.v1",
     "why": "step C (masked 2016-18 leveling) result; steps B and C disagree in the direction "
            "of the 1-3-year subsidence response (docs/TWIN_EVIDENCE_2026-10.md section 13)"},
    {"type": "result", "path": f"{M1}/verdict.json", "field": "verdict", "startswith": "FAIL",
     "why": "M1 storage-capped column failed its synthetic gate 0 (post hoc, model-only; no "
            "registration)"},
    {"type": "registration", "id": "I1.X5SC.gate0.v1",
     "result": "results/twin_runs/par32/runs/e2_report.json",
     "result_field": ["verdict", "eligible_forms_for_I1.X5SC"], "result_equals": [],
     "why": "no slow-compaction form is identifiable from held leveling, so no level-dependent "
            "coastal response (or its band) can be stated"},
    U7_RULE,
    {"type": "result", "path": "results/twin/deep_pile_holdout/report.json",
     "field": "scores.n_sites", "equals": 4,
     "why": "pile count from the published report key only: 4 printed piles in the 2024-25 "
            "window"},
    {"type": "result", "path": "results/twin/deep_pile_holdout/report.json",
     "field": "scores.cum_deep.n_sites", "equals": 3,
     "why": "3 of them anchored below the declared depth: far below 20 clusters or 19 units"},
    {"type": "result", "path": "results/twin/deep_pile_holdout/report.json",
     "field": "prospective_validation", "equals": False,
     "why": "the pile comparison is not a prospective validation"},
]
THRESHOLDS = [
    {"type": "registration", "id": "A1r2.harness_rules.v1",
     "result": f"{DP}/reports/A1_evidence/repair2/coverage3_design.json",
     "reg_field": "thresholds.coverage_gate.calibrated_envelope.min_k", "reg_equals": 20,
     "why": "a coverage verdict is 'calibrated' only with at least 20 clusters (gate-3)"},
    {"type": "code", "check": "bands_min_units_signed_090", "equals": 19,
     "why": "a signed 0.90 conformal band needs at least 19 exchangeable units "
            "(hydrophysics/twin/bands.py min_units)"},
    {"type": "code", "check": "coverage3_envelope_min_k", "equals": 20,
     "why": "harness.coverage3 ENVELOPE min_k"},
    {"type": "code", "check": "guard_unlisted_use_refused", "equals": True,
     "why": "common/guard.py check_use refuses a use that is not in allowed_uses"},
]
E02_REASONS = [
    {"type": "result", "path": f"{E02}/protocol.json", "field": "test_start",
     "equals": "2026-11-01", "why": "E02's registered window starts 2026-11-01"},
    {"type": "result", "path": f"{E02}/protocol.json", "field": "test_end",
     "equals": "2027-10-31", "why": "and ends 2027-10-31"},
    {"type": "result", "path": f"{E02}/protocol.json", "field": "interval",
     "equals": "central 90 %, combined sigma", "why": "the registered interval"},
    {"type": "result", "path": f"{E02}/protocol.json", "field": "coverage_bounds",
     "equals": [0.85, 0.95], "why": "its only coverage check (pooled point window)"},
    {"type": "result", "path": f"{E02}/manifest.json", "field": "predictions_sha256",
     "equals": "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32",
     "why": "the frozen predictions (manifest-hash registration of 2026-10-11, made before "
            "the current registry)"},
]

SUBCLAIMS = {
    "C14a": {
        "what": "per-well registered 90 % head interval (combined sigma, E02) at the 158 "
                "monitoring wells, 2026-11..2027-10",
        "status": "N+D",
        "status_reading": "N: the page names the interval in text only (at the monitoring "
                          "wells, in the window, with the U7 disclosure); D: its coverage is "
                          "decided by E02 and, once registered, E02c; this declaration "
                          "decides nothing about it",
        "answer": "text only, no band: the evidence line names the registered per-well 90 % "
                  "interval (combined spread, at the monitoring wells, in the window only) "
                  "with the U7 disclosure, and only when the build bound E02's manifest, "
                  "predictions and protocol to the page's forward run; no aggregate "
                  "half-width summary and no band is shown",
        "decided_by": "E02 (manifest-hash registration 2026-10-11, pooled [0.85, 0.95]) and, "
                      "once registered, E02c T-cov (round 5, rank 4); not decided here",
        "label_u7": {"en": "the interval family was chosen after its 2023-26 coverage was "
                           "seen; the 2026-11..2027-10 window is untouched",
                     "zh": "此區間形式是在看過其 2023-26 年涵蓋率之後選定的；2026-11..2027-10 "
                           "的檢驗期間未被觸及"},
        "ruling": "2026-10-29 user, third set (a)",
        "reasons": E02_REASONS},
    "C14b": {
        "what": "the same interval recipe at the monitoring wells outside 2026-11..2027-10 "
                "(cells without a well belong to C14c)",
        "status": "N",
        "wording": {"en": "the same recipe outside 2026-11..2027-10: untested",
                    "zh": "同一方法在 2026-11..2027-10 以外：未檢驗"},
        "reasons": [E02_REASONS[0], E02_REASONS[1],
                    _ctx("no other registered coverage test of this recipe exists; the "
                         "2023-26 coverage is an after-the-fact measurement (U7)",
                         "hydrophysics/twin/claims_v1.json C14 notes")]},
    "C14c": {
        "what": "aggregate run spreads: N2-unit mean, township, fan-average layer 2 (bn_head), "
                "cells without a well",
        "status": "N",
        "wording": {"en": "run spread, a parameter spread, not a predictive interval",
                    "zh": "模擬分布，是參數分布，不是預測區間"},
        "answer": "bn_head (the fan-average layer-2 head run spread) stays on the answer's "
                  "head series, labelled with this wording; the page's 'no band' sentence "
                  "must not list the fan average among outputs without a band",
        "reasons": [
            {"type": "result", "path": f"{E02}/manifest.json", "field": "rows", "equals": 1896,
             "why": "E02 scores per-well-month rows (158 x 12), never a mean over wells; with "
                    "correlated well errors an interval for a mean is untested"},
            {"type": "result", "path": f"{E02}/protocol.json", "field": "datum",
             "equals": "per-well mean fitted-period residual; fitted months only",
             "why": "the interval is anchored on a per-well datum: a cell without a well has "
                    "no datum and no interval"},
            {"type": "code", "check": "template_has_keys", "keys": ["bn_head", "cov_line"],
             "equals": True, "why": "the page elements this wording applies to"}]},
    "C14d": {
        "what": "paired run spread of the head contrast (rng_p, dn_yr, dn_con for heads) on "
                "the answer tiles and policy series",
        "status": "L",
        "answer": "removed from the answer; shown only under Model internals",
        "wording": {"en": "a parameter spread; it excludes response-size error",
                    "zh": "參數分布，不含反應大小的誤差"},
        "plan_label": "parameter spread; excludes response-size error",
        "c01_range_line": "C01 keeps N2's existing verdict wording ('point slope', not "
                          "'response'); no x0.67-1.5 band is drawn",
        "ruling": "2026-10-29 user, third set (d)",
        "reasons": [
            {"type": "result", "path": f"{A0}/verdict.json",
             "field": "findings.head_response_cut30",
             "contains": "response SIZE is not pinned by the calibration data",
             "why": "A0 (post hoc identifiability audit; replaces the killed U1-L): the "
                    "calibration data do not pin the head-response size, so no held data "
                    "bound the contrast size"},
            {"type": "registration", "id": "N2.policy_response.prospective_2026-11.v1",
             "result": None, "result_pending": "N2 is unscored (not before 2028-01-15)",
             "reg_field": "thresholds.ratio_band", "reg_equals": 1.5,
             "why": "N2's registered rule (pending, no result): the within-window anomaly "
                    "slope is judged as a point in [1/1.5, 1.5]; its verdict wording, not a "
                    "band, is what C01 carries"},
            {"type": "code", "check": "template_has_keys", "keys": ["rng_p", "dn_yr", "dn_con"],
             "equals": True, "why": "the page elements moved"},
            _ctx("model_averaging synthetic (post hoc, model output only): the paired-member "
                 "spread covered 0 % of the true contrast in every scenario; level conformal, "
                 "stacking and envelopes 0.16-0.41", "round-5 exploration "
                                                     "explore/model_averaging/sim_results.json "
                                                     "(session scratch; not machine-checked)")]},
    "C14e": {
        "what": "subsidence level and contrast bands (bn_runs, bn_constructed, dn_yr / dn_con "
                "and rng_p on cm tiles, rng_scaled, rng_h_only)",
        "status": "W",
        "internals_status": "L",
        "answer": "withdrawn from the answer; the internals copy is labelled",
        "wording": {"en": "a parameter spread, not a predictive interval; no held data can "
                          "calibrate a subsidence band",
                    "zh": "參數分布，不是預測區間；現有資料無法校準地層下陷的帶狀"},
        "scope_note": "the page says 'held'; arriving data cannot calibrate one either, since "
                      "no subsidence product allows prospective_scorer (decision 4)",
        "matches": "C06 / C07 (Model internals)",
        "ruling": "2026-10-29 user, third set (d): subsidence bands permanently Model "
                  "internals",
        "reasons": SUBSIDENCE_SCORERS + SUBSIDENCE_TESTS},
}
DECISIONS = [
    {"n": 1, "subclaim": "C14b", "decision": "N: the recipe is untested outside the window",
     "reasons": SUBCLAIMS["C14b"]["reasons"]},
    {"n": 2, "subclaim": "C14c", "decision": "N: 'run spread, a parameter spread, not a "
                                             "predictive interval'; an interval for a mean of "
                                             "correlated wells is untested and a cell without "
                                             "a well has no datum",
     "reasons": SUBCLAIMS["C14c"]["reasons"]},
    {"n": 3, "subclaim": "C14d", "decision": "L: the paired run spread leaves the answer tiles "
                                             "for Model internals, labelled 'parameter spread; "
                                             "excludes response-size error'",
     "reasons": SUBCLAIMS["C14d"]["reasons"]},
    {"n": 4, "subclaim": "C14e", "decision": "W in the answer, L in Model internals (as C06 / "
                                             "C07); every subsidence scorer is closed for a "
                                             "band by its current-release allowed_uses or by a "
                                             "registered failure",
     "reasons": SUBCLAIMS["C14e"]["reasons"]},
    {"n": 5, "subclaim": "closed_route",
     "decision": "rolling-origin conformal bands are closed, for two different reasons. "
                 "Subsidence and leveling: the held leveling allows at most 6 rolling "
                 "origins (2016-21, I1.X5NH.v1), below bands.min_units = 19 for a signed "
                 "0.90 band. Heads: not too few units, but spent windows: 2016-22 head "
                 "windows are design-time, and the interval family was chosen on its "
                 "2023-26 coverage (U7; ruling (a)), so a head rolling-origin coverage would "
                 "be design-time evidence, never a calibration; only E02 and E02c decide the "
                 "head interval",
     "reasons": [THRESHOLDS[1],
                 {"type": "registration", "id": "I1.X5NH.v1",
                  "result": f"{DP}/reports/I1_X5NH/report.json",
                  "reg_contains": "rolling origins 2016-04, 2017-04, 2018-04, 2019-04, "
                                  "2020-04, 2021-04",
                  "why": "subsidence and leveling: the rolling origins the record allows "
                         "(six, 2016-21)"},
                 {"type": "code", "check": "file_contains",
                  "file": "hydrophysics/twin/prospective.py", "text": "covered 92 % of the",
                  "equals": True,
                  "why": "heads: the combined family was chosen because it covered 92 % of "
                         "the 2023-26 observations (U7 design selection)"},
                 U7_RULE]},
    {"n": 6, "subclaim": "future_band_condition",
     "decision": "a future band needs a new registration naming a product whose MANIFEST "
                 "allows prospective_scorer, at least 20 clusters for a calibrated gate-3 "
                 "verdict (A1r2.harness_rules.v1) and at least 19 units for a signed 0.90 "
                 "conformal band",
     "reasons": THRESHOLDS},
    {"n": 7, "subclaim": "changhua_contrast",
     "decision": "no band; the Changhua head lever appears only through N2C's ruled mapping "
                 "(this does not prejudge N2C); wherever it is mentioned it carries "
                 f"'{N2C_LABEL}'",
     "wording": {"en": f"no band; the Changhua head lever appears only through N2C's ruled "
                       f"mapping: {N2C_LABEL}",
                 "zh": f"無帶狀範圍；彰化水頭槓桿只依 N2C 已裁定的對應出現：{N2C_LABEL_ZH}"},
     "reasons": [{"type": "registration", "id": "N2C.changhua_response.prospective_2026-11.v1",
                  "result": "results/twin/prospective_2026-11_n2c_r1c/manifest.json",
                  "reg_field": "page_mapping.before_scoring", "reg_contains": N2C_LABEL,
                  "why": "N2C's ruled page mapping (registered, pending)"}]},
]
RELATED = {
    "C06": {"status": "L", "reason": "no registered test of differential settlement, and no "
                                     "subsidence band can be tested (decision 4)",
            "cites": "U1.no_band.v1 decision 4"},
    "C07": {"status": "L", "reason": "sign agreement of parameter sets counts a withdrawn "
                                     "subsidence quantity: a parameter spread, not "
                                     "uncertainty (decisions 3-4)",
            "cites": "U1.no_band.v1 decisions 3-4"},
    "C09": {"status": "W", "reason": "the fan baseline size is untested and the distal column "
                                     "does not sink (I1.X5NH.v1); its band is a run spread "
                                     "(decision 4)",
            "cites": "U1.no_band.v1 decision 4; I1.X5NH.v1",
            "reasons": [{"type": "registration", "id": "I1.X5NH.v1",
                         "result": f"{DP}/reports/I1_X5NH/report.json",
                         "why": "registered coastal rung and its report"}]},
    "C30": {"status": "N+D", "legend_band_sentence": {
        "en": "Projected value ({y}), not validated; no uncertainty band, since no held data "
              "can calibrate one.",
        "zh": "推估值（{y}），未驗證；沒有不確定性帶狀，因為現有資料無法校準。"},
            "status_reading": "N: the legend sentence above (template lg_proj); D: the heads "
                              "part is decided by E02c's strict level rule (rank 4), not by "
                              "this declaration",
            "cites": "U1.no_band.v1 decisions 2 and 4"},
}
# page band elements: which answer output each decorates before and after (template keys)
BAND_ELEMENTS = [
    {"keys": ["dn_yr", "dn_con"], "on": "C01", "kind": "paired run spread, head series",
     "before": "answer", "after": "internals", "subclaim": "C14d"},
    {"keys": ["bn_head"], "on": "C30", "kind": "fan-average layer-2 head run spread",
     "before": "answer", "after": "answer (C14c wording)", "subclaim": "C14c"},
    {"keys": ["cov_line", "evc_cov"], "on": "C30", "kind": "coverage line of the run spread",
     "before": "answer", "after": "answer: names the registered per-well interval (C14a, D) "
                                  "and the run spread (C14c)", "subclaim": "C14a/C14c"},
    {"keys": ["bn_runs", "bn_constructed"], "on": "C30", "kind": "subsidence baseline band",
     "before": "answer", "after": "internals", "subclaim": "C14e"},
    {"keys": ["rng_p", "rng_scaled", "rng_h_only"], "on": "C03",
     "kind": "cm tile run range (tile already in internals)", "before": "internals",
     "after": "internals", "subclaim": "C14e"},
]
# answer outputs that carry a band display / a band whose coverage a registered test decides
SHARE_BEFORE = {"any_band": ["C01", "C30"], "registered_coverage": []}
SHARE_AFTER = {"any_band": ["C30"], "registered_coverage": ["E02"]}
SHARE_NOTE = ("'registered_coverage' 0 -> 1 is a LABELLING change under ruling (a), not new "
              "band coverage: E02's interval and its pooled coverage check were registered on "
              "2026-10-11, no band is drawn for it, and this declaration adds no test; only "
              "the page now names that interval as decided by its registered test")

# What the ledger and the page must say for this declaration to be registered (checked by
# page_alignment_problems in --validate; recorded in the document). Template strings are
# compared after mapping the en dash to '-'; EN case-insensitively.
_ALIGNED_IDS = ("C14a", "C14b", "C14c", "C14d", "C14e", "C06", "C07", "C09", "C30")
PAGE_ALIGNMENT = {
    "ledger": {
        "status_current": "every C14 sub-claim and related claim: equals the declared status "
                          "(status + internals_status as 'X+Y')",
        "cites": f"every one of {', '.join(_ALIGNED_IDS)} lists {ID} in cites.registrations",
        "wording_contains": {
            "C14a": ["the interval family was chosen after its 2023-26 coverage was seen"],
            "C14b": ["outside that window the recipe is untested"],
            "C14c": [SUBCLAIMS["C14c"]["wording"]["en"]],
            "C14d": [SUBCLAIMS["C14d"]["wording"]["en"]],
            "C14e": [SUBCLAIMS["C14e"]["wording"]["en"]],
            "C30": ["no uncertainty band, since no held data can calibrate one"]},
        "notes_contains": {"C14a": ["half-width summary is not on the page"]}},
    "template": {
        "contains": [
            {"key": "evc_pro_iv", "subclaim": "C14a",
             "en": ["the interval family was chosen after its 2023-26 coverage was seen",
                    "outside that window the recipe is untested"],
             "zh": ["在看過其 2023-26 年涵蓋率之後選定", "未經檢驗"]},
            {"key": "bn_head", "subclaim": "C14c",
             "en": ["a run spread, a parameter spread, not a predictive interval"],
             "zh": ["屬參數分布，不是預測區間"]},
            {"key": "dn_yr", "subclaim": "C14d",
             "en": ["model internals", SUBCLAIMS["C14d"]["wording"]["en"]],
             "zh": ["模型內部", "不含反應大小的誤差"]},
            {"key": "dn_con", "subclaim": "C14d",
             "en": ["model internals", SUBCLAIMS["C14d"]["wording"]["en"]],
             "zh": ["模型內部", "不含反應大小的誤差"]},
            {"key": "bn_runs", "subclaim": "C14e",
             "en": ["model internals", SUBCLAIMS["C14e"]["wording"]["en"]],
             "zh": ["模型內部", "不是預測區間；現有資料無法校準地層下陷的帶狀"]},
            {"key": "lg_proj", "subclaim": "C30",
             "en": [RELATED["C30"]["legend_band_sentence"]["en"]],
             "zh": [RELATED["C30"]["legend_band_sentence"]["zh"]]}],
        "not_contains": [
            {"key": "evc_band_none", "subclaim": "C14c",
             "en": ["the fan"], "zh": ["全扇"],
             "why": "bn_head (fan-average layer-2 head run spread) stays on the answer's head "
                    "series under C14c's wording, so the 'no band' sentence must not list the "
                    "fan average among outputs that show no band"}]},
}


# ----------------------------------------------------------------------------- checker
def _get(d, dotted):
    """Value at a dotted path, or at a list of keys (for keys that contain a dot)."""
    cur = d
    for k in (dotted if isinstance(dotted, (list, tuple)) else dotted.split(".")):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            raise KeyError(dotted)
    return cur


def _expect(value, r: dict, prefix: str = "") -> str | None:
    """None if ``value`` meets the expectation keys of ``r`` (with ``prefix``)."""
    if prefix + "equals" in r and value != r[prefix + "equals"]:
        return f"{value!r} != {r[prefix + 'equals']!r}"
    if prefix + "contains" in r:
        want = r[prefix + "contains"]
        hay = value if isinstance(value, (str, list)) else json.dumps(value)
        if want not in hay:
            return f"{want!r} not in value"
    if prefix + "startswith" in r and not str(value).startswith(r[prefix + "startswith"]):
        return f"{str(value)[:40]!r} does not start with {r[prefix + 'startswith']!r}"
    return None


def _load(path: str):
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def _code_check(r: dict):
    c = r["check"]
    if c == "bands_min_units_signed_090":
        sys.path.insert(0, str(HERE))
        import power_u1hb
        return power_u1hb.bands_constants()["min_units_signed_0.90"]
    if c == "coverage3_envelope_min_k":
        from harness import coverage3
        return coverage3.ENVELOPE["min_k"]
    if c == "guard_unlisted_use_refused":
        from common import guard
        m = {"product": "probe", "version": "v0", "allowed_uses": ["validator"],
             "forbidden_uses": [], "fit_window_end": "2022-12"}
        try:
            guard.check_use(m, "prospective_scorer")
        except guard.UseNotAllowed:
            return True
        return False
    if c == "file_contains":
        return r["text"] in (REPO / r["file"]).read_text(encoding="utf-8")
    if c == "template_has_keys":
        txt = (REPO / TEMPLATE).read_text(encoding="utf-8")
        return all(re.search(rf"\b{re.escape(k)}:\s*\"", txt) for k in r["keys"])
    raise KeyError(f"unknown code check {c!r}")


def check_reason(r: dict, prereg_dir: Path | None = None) -> dict:
    """{"status": ok | fail | missing | context, "detail"} for one reason."""
    from common.paths import PREREG_DIR

    t = r.get("type")
    try:
        if t == "context":
            ok = bool(r.get("text")) and bool(r.get("source"))
            return {"status": "context" if ok else "fail", "detail": "not machine-checked"}
        if t == "code":
            got = _code_check(r)
            err = _expect(got, r)
            return {"status": "fail" if err else "ok", "detail": err or f"{r['check']} = {got}"}
        if t in ("manifest_use", "manifest_field"):
            p = REPO / r["manifest"]
            if not p.exists():
                return {"status": "missing", "detail": r["manifest"]}
            m = _load(r["manifest"])
            if t == "manifest_field":
                err = _expect(_get(m, r["field"]), r)
                return {"status": "fail" if err else "ok", "detail": err or r["field"]}
            al, fb = m.get("allowed_uses", []), m.get("forbidden_uses", [])
            u, ex = r["use"], r["expect"]
            ok = {"forbidden": u in fb and u not in al,
                  "unlisted": u not in al and u not in fb,
                  "not_allowed": u not in al,
                  "allowed": u in al}.get(ex)
            if ok is None:
                return {"status": "fail", "detail": f"unknown expect {ex!r}"}
            if "allowed_equals" in r and sorted(al) != sorted(r["allowed_equals"]):
                return {"status": "fail", "detail": f"allowed_uses {al} != {r['allowed_equals']}"}
            return {"status": "ok" if ok else "fail",
                    "detail": f"{u}: expect {ex}; allowed {al}"}
        if t == "registration":
            d = Path(prereg_dir) if prereg_dir is not None else PREREG_DIR
            if not (d / "INDEX.jsonl").exists():
                return {"status": "missing", "detail": f"{d}/INDEX.jsonl"}
            rows = [x for x in prereg3.read_chained(d / "INDEX.jsonl") if x["id"] == r["id"]]
            if not rows:
                return {"status": "fail", "detail": f"{r['id']} not in INDEX.jsonl"}
            f = d / f"{r['id']}.json"
            if hashlib.sha256(f.read_bytes()).hexdigest() != rows[-1]["sha256"]:
                return {"status": "fail", "detail": f"{r['id']} file sha differs from INDEX"}
            doc = json.loads(f.read_text(encoding="utf-8"))
            if "reg_field" in r:
                err = _expect(_get(doc, r["reg_field"]), r, "reg_")
                if err:
                    return {"status": "fail", "detail": f"{r['reg_field']}: {err}"}
            elif "reg_contains" in r:
                if r["reg_contains"] not in json.dumps(doc, ensure_ascii=False):
                    return {"status": "fail", "detail": "reg_contains not found"}
            if r.get("result") is None:
                if not r.get("result_pending"):
                    return {"status": "fail", "detail": "no result and no result_pending"}
                return {"status": "ok", "detail": f"{r['id']} sha {rows[-1]['sha256'][:8]}; "
                                                  f"result pending: {r['result_pending']}"}
            if not (REPO / r["result"]).exists():
                return {"status": "missing", "detail": r["result"]}
            if "result_field" in r:
                err = _expect(_get(_load(r["result"]), r["result_field"]), r, "result_")
                if err:
                    return {"status": "fail", "detail": f"{r['result_field']}: {err}"}
            return {"status": "ok", "detail": f"{r['id']} sha {rows[-1]['sha256'][:8]}"}
        if t == "result":
            if not (REPO / r["path"]).exists():
                return {"status": "missing", "detail": r["path"]}
            err = _expect(_get(_load(r["path"]), r["field"]), r)
            return {"status": "fail" if err else "ok", "detail": err or r["field"]}
    except KeyError as exc:
        return {"status": "fail", "detail": f"field {exc} not found"}
    return {"status": "fail", "detail": f"unknown reason type {t!r}"}


def _all_reason_sets(d: dict):
    for k, s in d["subclaims"].items():
        yield f"subclaims.{k}", s.get("reasons") or []
    for x in d["decisions"]:
        yield f"decisions.{x['n']}", x.get("reasons") or []
    for k, s in d["related_claims"].items():
        if "reasons" in s:
            yield f"related_claims.{k}", s["reasons"]


def check_reasons(d: dict, prereg_dir: Path | None = None) -> list[dict]:
    """Every reason of the declaration, checked; plus a structural row per reason set that
    fails when it has no machine-checked reason."""
    out = []
    for where, rs in _all_reason_sets(d):
        if not any(r.get("type") in MACHINE for r in rs):
            out.append({"where": where, "i": None, "type": "structure", "status": "fail",
                        "detail": "no machine-checked reason (manifest or registration or "
                                  "result)"})
        for i, r in enumerate(rs):
            if r.get("type") not in REASON_TYPES:
                out.append({"where": where, "i": i, "type": r.get("type"), "status": "fail",
                            "detail": "unknown type"})
                continue
            out.append({"where": where, "i": i, "type": r["type"], **check_reason(r, prereg_dir)})
    return out


def structure_problems(d: dict) -> list[str]:
    out = []
    if sorted(d["subclaims"]) != ["C14a", "C14b", "C14c", "C14d", "C14e"]:
        out.append("sub-claims must be exactly C14a-e")
    for k, s in d["subclaims"].items():
        if not status_ok(s.get("status")):
            out.append(f"{k}: status {s.get('status')!r} not in S/D/N/L/W")
        if s.get("internals_status") is not None and not status_ok(s["internals_status"]):
            out.append(f"{k}: internals_status not in S/D/N/L/W")
    if [x["n"] for x in d["decisions"]] != list(range(1, 8)):
        out.append("decisions must be numbered 1-7")
    for k, s in d["related_claims"].items():
        if not status_ok(s.get("status")):
            out.append(f"{k}: status not in S/D/N/L/W")
    texts = json.dumps(d, ensure_ascii=False)
    if "calibrated interval" in texts.replace("not a calibrated interval", ""):
        out.append("the declaration calls a band a 'calibrated interval'")
    return out


# ----------------------------------------------------------------------------- page alignment
def _norm(t: str, lang: str) -> str:
    t = t.replace("\u2013", "-")
    return t.lower() if lang == "en" else t


def template_strings(template_path: Path | None = None) -> dict:
    """{key: [en, zh]} of the i18n string literals ``key: "..."`` in the page template (the
    EN block comes first); keys that do not occur exactly twice map to what was found."""
    p = Path(template_path) if template_path is not None else REPO / TEMPLATE
    txt = p.read_text(encoding="utf-8")
    keys = {x["key"] for g in PAGE_ALIGNMENT["template"].values() for x in g}
    out = {}
    for k in sorted(keys):
        out[k] = [json.loads(f'"{m}"') for m in
                  re.findall(rf'(?<![\w$]){re.escape(k)}:\s*"((?:[^"\\]|\\.)*)"', txt)]
    return out


def page_alignment_problems(ledger_path: Path | None = None,
                            template_path: Path | None = None) -> list[str]:
    """Where the ledger or the page disagrees with this declaration (empty = aligned)."""
    lp = Path(ledger_path) if ledger_path is not None else REPO / LEDGER
    led = {c["id"]: c for c in json.loads(lp.read_text(encoding="utf-8"))["claims"]}
    out = []
    declared = {**{k: ledger_status(v) for k, v in SUBCLAIMS.items()},
                **{k: v["status"] for k, v in RELATED.items()}}
    for k in _ALIGNED_IDS:
        c = led.get(k)
        if c is None:
            out.append(f"ledger: {k} missing")
            continue
        if c.get("status_current") != declared[k]:
            out.append(f"ledger: {k} status_current {c.get('status_current')!r} != declared "
                       f"{declared[k]!r}")
        if ID not in (c.get("cites") or {}).get("registrations", []):
            out.append(f"ledger: {k} does not cite {ID}")
    for field, spec in (("wording_en", PAGE_ALIGNMENT["ledger"]["wording_contains"]),
                        ("notes", PAGE_ALIGNMENT["ledger"]["notes_contains"])):
        for k, phrases in spec.items():
            have = _norm(str((led.get(k) or {}).get(field, "")), "en")
            for ph in phrases:
                if _norm(ph, "en") not in have:
                    out.append(f"ledger: {k} {field} lacks {ph!r}")
    strings = template_strings(template_path)
    for rule, want in (("contains", True), ("not_contains", False)):
        for x in PAGE_ALIGNMENT["template"][rule]:
            got = strings.get(x["key"], [])
            if len(got) != 2:
                out.append(f"template: {x['key']} found {len(got)} times, expected EN + ZH")
                continue
            for lang, text in zip(("en", "zh"), got, strict=True):
                for ph in x[lang]:
                    if (_norm(ph, lang) in _norm(text, lang)) != want:
                        out.append(f"template: {x['key']} ({lang}) "
                                   f"{'lacks' if want else 'must not contain'} {ph!r} "
                                   f"({x['subclaim']})")
    return out


# ----------------------------------------------------------------------------- band share
def _answer(status: str) -> bool:
    return any(s in STATUSES - {"L", "W"} for s in str(status).replace("+", ""))


def band_share(ledger_path: Path | None = None) -> dict:
    """Share of the ledger's answer outputs (status with S, D or N; the C14 family itself
    excluded) that carry a band display, and a band whose coverage a registered test decides,
    before and after this declaration (``SHARE_BEFORE`` / ``SHARE_AFTER``)."""
    p = Path(ledger_path) if ledger_path is not None else REPO / LEDGER
    led = json.loads(p.read_text(encoding="utf-8"))
    ids = [c["id"] for c in led["claims"]
           if not c["id"].startswith("C14") and _answer(c.get("status_current", ""))]
    n = len(ids)
    miss = sorted({c for s in (SHARE_BEFORE, SHARE_AFTER) for v in s.values() for c in v}
                  - set(ids))
    out = {"ledger": str(p.relative_to(REPO)) if REPO in p.parents else str(p),
           "ledger_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
           "answer_outputs": n, "not_answer_outputs_in_map": miss}
    for name, s in (("before", SHARE_BEFORE), ("after", SHARE_AFTER)):
        out[name] = {k: {"claims": v, "share": round(len(v) / n, 4) if n else None}
                     for k, v in s.items()}
    out["reading"] = ("a narrowing, not evidence: after it, the only band on an answer "
                      "output is C30's fan-average head run spread, labelled a parameter "
                      "spread (C14c, N); the registered per-well interval is named in text "
                      "only (C14a, N+D, pending)")
    out["registered_coverage_note"] = SHARE_NOTE
    return out


# ----------------------------------------------------------------------------- document
def doc() -> dict:
    inputs = {THIS, LEDGER, TEMPLATE, "hydrophysics/twin/bands.py",
              "dataprep_staging/harness/coverage3.py", "dataprep_staging/common/guard.py"}
    for _, rs in _all_reason_sets({"subclaims": SUBCLAIMS, "decisions": DECISIONS,
                                   "related_claims": RELATED}):
        for r in rs:
            for k in ("manifest", "path", "result"):
                if r.get(k):
                    inputs.add(r[k])
            if r.get("type") == "registration":
                inputs.add(f"{DP}/preregistration/{r['id']}.json")
    return {
        "id": ID, "wp": "U1", "kind": "declaration",
        "title": "Narrowing of every page band no held or arriving data can calibrate: C14 "
                 "split into C14a-e, with a machine-checked reason per decision",
        "label": "a narrowing, not evidence: it scores nothing and supports no world claim",
        "why": "The page's band claim C14 was one 'N' with a generic reason. Round 5 merged "
               "U1-N, U1.SN and the spread-removal part of U1.CB into one declaration with "
               "one sub-claim scheme, so every band the page shows or withdrew has an "
               "auditable reason (a manifest use rule or a registered result), and the 'no "
               "band' answer can be checked.",
        "subclaims": copy.deepcopy(SUBCLAIMS),
        "decisions": copy.deepcopy(DECISIONS),
        "related_claims": copy.deepcopy(RELATED),
        "band_elements": copy.deepcopy(BAND_ELEMENTS),
        "page_alignment": copy.deepcopy(PAGE_ALIGNMENT),
        "band_share": {"before": SHARE_BEFORE, "after": SHARE_AFTER, "note": SHARE_NOTE,
                       "rule": "share = claims listed / answer outputs of the ledger (status "
                               "with S, D or N; C14 family excluded), computed by "
                               "prereg_u1_no_band.band_share from the ledger as written "
                               "(its sha256 is recorded)",
                       "computed": band_share()},
        "reason_types": {
            "manifest_use": "MANIFEST path + use: 'forbidden' (in forbidden_uses), 'unlisted' "
                            "(in neither list: not allowed under guard.py), 'not_allowed', "
                            "'allowed'; optional allowed_equals",
            "manifest_field": "MANIFEST path + dotted field + equals/contains/startswith",
            "registration": "prereg3 id in INDEX.jsonl with the same file sha + an existing "
                            "result path, or result null with 'result_pending' for an "
                            "unscored registration (+ optional reg_ / result_ field checks)",
            "result": "published result file + dotted field + expectation (post-hoc reports "
                      "with no registration: A0, M1, E02 manifest-hash files, pile report)",
            "code": "a code fact (supplementary)",
            "context": "a note that cannot be machine-checked; never sufficient alone"},
        "page_effect": "C14 becomes C14a-e. C14a: N+D (registered interval named in text "
                       "only, U7 label; decided by E02 / E02c). C14b, C14c: N with the wording "
                       "above (bn_head stays on the answer's head series as a labelled run "
                       "spread). C14d: L (moved to Model internals). C14e: W in the answer, L "
                       "in internals. C06, C07, C09 gain the cited reasons; C30's legend "
                       "carries 'no uncertainty band, since no held data can calibrate one' "
                       "(its D part is E02c's). Nothing is promoted. Registration is refused "
                       "until the ledger and template match 'page_alignment'.",
        "data_read": "MANIFEST allowed/forbidden uses and gate flags, registration JSONs, and "
                     "named keys of published reports (A0 and M1 verdicts, N1 step B/C "
                     "results 'prereg', the X5SC verdict list, the pile report's n_sites and "
                     "prospective_validation, E02's protocol and manifest), the ledger and "
                     "the page template's strings. No observation value is read; pile counts "
                     "come from report keys only.",
        "attempt_policy": "User rule 2026-10-23: a declaration, not a test; nothing is scored "
                          "or re-scored. A future band for any sub-claim needs a NEW "
                          "registration meeting decision 6, labelled as such; this "
                          "declaration is never edited after registration (a change needs a "
                          "new id).",
        "u7": "no post-origin observation and no 2023-26 residual is used; U7 is itself a "
              "reason (decision 4)",
        "rulings": {
            "2026-10-29 user, third set (a)": "C14a's interval family carries the EN/ZH U7 "
                                              "disclosure",
            "2026-10-29 user, third set (d)": "C14 split into C14a-e; subsidence bands "
                                              "permanently Model internals; paired run spread "
                                              "off the head-response tiles into Model "
                                              "internals",
            "2026-10-29 user (4)": "no calibrated correction factor follows (unchanged)"},
        "order": "registered before E02c (rank 4) is written and before any band-related "
                 "score; by 2026-10-31T16:00Z",
        "zones": NA("a ledger narrowing; no model zones"),
        "priors": NA("no estimate"),
        "burn_in": NA("no fit"),
        "folds": NA("no fit"),
        "tests": NA("a narrowing declaration; nothing is scored"),
        "fresh_units": NA("no observation is involved; nothing is sealed"),
        "thresholds": {"min_clusters_calibrated_gate3": 20, "min_units_signed_conformal_090": 19,
                       "max_rolling_origins": 6},
        "gate_scores": NA("no gate"),
        "primary_model": NA("no model is assessed"),
        "inputs": sorted(inputs),
        "interpretation": "Which band claims the record can and cannot support, and why, per "
                          "sub-claim. It raises the honesty of the 'no band' answer; it is "
                          "not evidence for any world claim and promotes nothing.",
        "written_by": THIS,
    }


def canonical_sha(d: dict) -> str:
    return hashlib.sha256((json.dumps(d, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="dry run: print, check, write nothing")
    g.add_argument("--write", action="store_true", help="register once (the lead only)")
    ap.add_argument("--expect-self-sha", default=None,
                    help="sha256 of this script as reviewed; required with --write")
    ap.add_argument("--allow-late", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="do not print the document")
    a = ap.parse_args(argv)
    d = doc()
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", canonical_sha(d))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    probs = structure_problems(d)
    rows = check_reasons(d)
    bad_rows = [r for r in rows if r["status"] in ("fail", "missing")]
    n_ok = sum(r["status"] == "ok" for r in rows)
    n_ctx = sum(r["status"] == "context" for r in rows)
    print(f"reasons: {n_ok} ok, {n_ctx} context, {len(bad_rows)} failing or missing")
    for r in bad_rows:
        print("  ", r)
    missing_inputs = [p for p in d["inputs"] if not (REPO / p).exists()]
    print("missing inputs:", missing_inputs or "none")
    print("band share:", json.dumps(band_share(), ensure_ascii=False))
    print("structure:", probs or "ok")
    align = page_alignment_problems()
    print("page alignment (ledger + template):", "ok" if not align else "")
    for x in align:
        print("  ", x)
    block = prereg3.blocking_looks([], None, test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none (no tests, no fresh units)")
    if (REPO / f"{DP}/preregistration/{ID}.json").exists():
        probs.append(f"{ID} is already registered")
    self_sha = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    print("this script sha256:", self_sha)
    late = dt.datetime.now(dt.timezone.utc) >= DEADLINE
    if late:
        print(f"DEADLINE PASSED ({DEADLINE:%Y-%m-%d %H:%M} UTC)")
    bad = bool(errs or probs or bad_rows or missing_inputs or block or align)
    if a.validate:
        print("valid; not written" if not bad else "NOT READY: fix the problems above")
        return 1 if bad else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --write needs --expect-self-sha equal to this script's sha256")
        return 1
    if late and not a.allow_late:
        print("REFUSED: past the deadline")
        return 1
    if bad:
        print("REFUSED: not ready to register")
        return 1
    path, h = prereg3.write_prereg(d)
    print(f"registered {path.relative_to(REPO)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
