"""Write the two prospective test registrations of round 7 ranks 5-6 (harness.prereg3, kind
'test'):

  e02d  E02d.decomposed_prospective_2026-12.v1: the per-well (relative) 90 % band test on
        2026-12..2027-10 at E02's 158 wells under a NEW unit id; register before
        2026-11-30T16:00Z
  e02y  E02y.years23_2027-11_2029-10.v1: years 2 and 3 (2027-11..2028-10, 2028-11..2029-10)
        of the same frozen forward, relative (primary per county where the 12-month P-gate
        passed) and absolute (secondary, pass_weak / fail_shock wording); user ruling
        R7.heads_years23; register before 2027-11-01 (planned: the same week as E02d)

Both read nothing from heads. ``--write`` (the lead only) refuses unless DR.relative_drift.v1
ran under its registration, U1.decomp_power.v1's registered result exists (for e02d: both
counties pass at 11 months), the artefact (build_artefact.py) and the absolute registrations
are frozen and hash-consistent, no look blocks the units, the deadline has not passed and
``--expect-self-sha`` is this file's sha256.

    python -I dataprep_staging/e02d/prereg_e02d.py e02d --validate
    python -I dataprep_staging/e02d/prereg_e02d.py e02d --write --expect-self-sha <sha>
    python -I dataprep_staging/e02d/prereg_e02d.py e02y --validate
    python -I dataprep_staging/e02d/prereg_e02d.py e02y --write --expect-self-sha <sha>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_artefact as BA  # noqa: E402
import e02dio as io  # noqa: E402
import e02dlib as L  # noqa: E402
import members_e02d as ME  # noqa: E402
import power_e02d as PW  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
THIS = f"{io.X}/prereg_e02d.py"
SCORER = f"{io.X}/score_e02d.py"
# this script is NOT frozen: its job ends at --write and the registered document's own sha
# binds what it wrote (an edit between the E02d and E02y writes leaves E02d scorable); its
# sha256 is recorded report-only under registration_script
FROZEN = (SCORER, f"{io.X}/e02dlib.py", f"{io.X}/e02dio.py", f"{io.X}/build_artefact.py",
          f"{io.X}/members_e02d.py", f"{io.X}/dr_drift.py", f"{io.X}/power_e02d.py",
          f"{io.X}/run_e02d.sh")
BOUND = ("dataprep_staging/e02c/score_e02c.py", "dataprep_staging/harness/coverage3.py",
         "dataprep_staging/harness/coverage2.py", "dataprep_staging/harness/coverage.py")
# recorded, report-only (a routine edit must not leave E02d unscorable; E02c's rule)
CODE_REPORT_ONLY = ("hydrophysics/twin/prospective.py", "hydrophysics/twin/validation.py",
                    "hydrophysics/twin/challenge.py", "dataprep_staging/harness/prereg3.py")
ENGINEERING_RESTORE = ("a git-tracked module listed in code_at_registration_report_only may be "
                       "restored to its registered sha256 (git checkout of that blob) before "
                       "the scoring run as an engineering step, logged in the report; frozen "
                       "and bound code is never edited")
Y23_DEADLINE = dt.datetime(2027, 11, 1, 0, 0, tzinfo=io.UTC)
E02C_ID = "E02c.prospective_2026-11.v1"
U7 = {"en": "the interval family was chosen after its 2023-26 coverage was seen; the relative "
            "(per-well) form is the same family with the shared part removed; the scored "
            "windows are untouched",
      "zh": "此區間形式是在看過其 2023-26 年涵蓋率之後選定的；相對（逐井）形式為同一形式扣除"
            "共同部分；計分期間未被觸及"}
MULTIPLICITY = ("E02 (pooled coverage and RMSE ratio), E02c (T-cov, T-level), the hybrid "
                "manifest, I1.hybrid_tf T1/T2, X1 T3a/T3b, N2 and N2C (Changhua subset) all "
                "score overlapping well-months of the same 158 wells; each decides only its own "
                "claim and no family-wise claim is made")
WORDING = {
    "rel_pass": "{county}: per-well spread around the county-wide departure passed its "
                "registered gate on {first}..{last} (coverage {cov:.2f}; {test}, prospective)",
    "rel_fail": "{county}: per-well spread around the county-wide departure failed its "
                "registered gate on {first}..{last} (coverage {cov:.2f}, {side}); the per-well "
                "wording is withdrawn for {county}",
    "rel_not_evaluable": "{county}: per-well band undecided ({why})",
    "rel_report_only": "{county}: report-only (the 12-month P-gate did not pass for "
                       "{county}); coverage {cov:.2f}",
    "abs_pass_weak": "{county}: no gross under-coverage detected on {first}..{last} "
                     "(absolute 90 % interval, coverage {cov:.2f}; one year, not decisive)",
    "abs_fail_shock": "{county}: the absolute 90 % interval failed its registered gate on "
                      "{first}..{last} (coverage {cov:.2f}); outside the registered range; may "
                      "reflect one year's shock",
    "never": ["calibrated", "校準", "the full interval passed"]}


def _rulings_ok(key: str) -> tuple[bool, str]:
    p = io.REPO / L.RULINGS
    if not p.is_file():
        return False, f"{L.RULINGS} missing"
    r = json.loads(p.read_text(encoding="utf-8"))
    rid = L.RULING_IDS[key]
    hit = [x for x in r.get("rulings", []) if x.get("id") == rid]
    if not hit or not str(hit[0].get("decision", "")).lower().startswith("yes"):
        return False, f"ruling {rid} is not a 'Yes'"
    return True, "ok"


def _unit(win: str, ids: list[str]) -> dict:
    w = L.WINDOWS[win]
    return {w["unit"]: {"description": f"E02's 158 wells (I1's member list, a new unit id), "
                                       f"heads {w['first']}..{w['last']} ({w['role']})",
                        "ids": ids, "window": f"{w['first']}..{w['last']}"}}


def _scope(win: str) -> str:
    w = L.WINDOWS[win]
    return f"heads:{w['first']}..{w['last']}"


def _slots() -> dict:
    """Readiness of every upstream artefact (no observation is read)."""
    out = {}
    drp = io.REPO / BA.DR_DIR / "report.json"
    if drp.is_file():
        dr = json.loads(drp.read_text(encoding="utf-8"))
        out["dr"] = {"status": "ok" if not dr.get("engineering_failure") else "failed",
                     "report": io.rec(drp), "wells": io.rec(io.REPO / BA.DR_DIR /
                                                             "wells_e02d.csv"),
                     "decisions": dr.get("decisions"),
                     "declaration": dr.get("declaration"), "look": dr.get("look")}
    else:
        out["dr"] = {"status": "PENDING"}
    pwp = io.REPO / PW.RESULT_DIR / "result.json"
    if pwp.is_file():
        r = json.loads(pwp.read_text(encoding="utf-8"))
        out["power"] = {"status": "ok" if not (r.get("smoke") or r.get("engineering_failure"))
                        else "invalid", "result": io.rec(pwp),
                        "declaration": r.get("declaration"),
                        "decisions": r.get("decisions"),
                        "e02d_registrable": r.get("e02d_registrable"),
                        "years23_relative_primary": r.get("years23_relative_primary"),
                        "dr_report_sha256": (r.get("design") or {}).get("dr_report_sha256")}
    else:
        out["power"] = {"status": "PENDING"}
    gp = io.REPO / ME.GATE_JSON
    out["members"] = ({"status": "ok" if json.loads(gp.read_text()).get("ok") else "failed",
                       "gate": io.rec(gp), "npz": io.rec(io.REPO / ME.MEMBERS_NPZ)}
                      if gp.is_file() else {"status": "PENDING"})
    sp = io.REPO / BA.OUT / "summary.json"
    if sp.is_file():
        s = json.loads(sp.read_text(encoding="utf-8"))
        files = {n: io.rec(io.REPO / BA.OUT / n) for n in s.get("files", {})}
        bad = [n for n, h in s.get("files", {}).items() if files[n]["sha256"] != h]
        out["artefact"] = {"status": "ok" if not bad else "mismatch", "summary": io.rec(sp),
                           "files": files, "dr_report_sha256":
                           (s.get("inputs") or {}).get("dr_report", {}).get("sha256"),
                           "abs_check": s.get("abs_check")}
    else:
        out["artefact"] = {"status": "PENDING"}
    for win in ("B", "C"):
        d = io.REPO / L.ABS_DIRS[win]
        out[f"abs_{win}"] = ({n: io.rec(d / n) for n in ("predictions.csv", "protocol.json",
                                                         "manifest.json")}
                             if (d / "manifest.json").is_file() else {"status": "PENDING"})
    return out


def _unit_ids() -> list[str]:
    import pandas as pd

    ids = sorted(pd.read_csv(io.REPO / L.E02_DIR / "predictions.csv",
                             dtype={"station_id": str})["station_id"].unique())
    if len(ids) != L.N_UNIT_WELLS:
        raise SystemExit(f"E02 has {len(ids)} wells, expected {L.N_UNIT_WELLS}")
    return ids


def _registration_script() -> dict:
    return {"path": THIS, "sha256": io.sha256(Path(__file__)),
            "binding": "report-only: not frozen; the registered document's sha binds what this "
                       "script wrote"}


def _relative_metric(win: str) -> dict:
    w = L.WINDOWS[win]
    src = ("E02's step-(5) rows (python -m hydrophysics.twin.prospective score on "
           f"{L.E02_DIR}, --new-wells inside the window product, run in E02/E02c's own "
           f"session), restricted to {w['first']}..{w['last']}" if win == "A" else
           f"the step-(5) rows of {L.ABS_DIRS[win]}, produced by {SCORER} itself AFTER "
           "prereg3.assert_can_score has logged both score looks of the window "
           "(hydrophysics.twin.prospective.score called in-process into <out>/step5, "
           "--new-wells inside the window product; nothing printed)")
    return {
        "observations": src + "; verified against the registered rows "
                              "(score_e02c.load_e02_step5: manifest, predicted/lower/upper/"
                              "baseline to 1e-9, validation.evaluate reproduced); no heads "
                              "table of its own",
        "residual": "r = observed_m - predicted_m (the frozen centre, no correction factor)",
        "relative": f"e = r - median of r over the OTHER scored wells of the same DR group in "
                    f"the same month (>= {L.MIN_OTHERS} others; otherwise the row is not "
                    "scored); groups = wells_e02d.csv 'group' (DR's rule, frozen)",
        "hit": f"|e| <= {L.Z90} * sigma_idio (sigma_idio_{win}.csv, frozen)",
        "gate": f"per county: wells with >= {L.MIN_MONTHS} scored months; undecided below "
                f"{L.MIN_WELLS_COUNTY}; harness.coverage3.coverage_gate (gate-3, method "
                f"{L.GATE_METHOD}, clusters = wells, target {L.TARGET}, level {L.GATE_LEVEL}) "
                f"AND point coverage within {list(L.POINT_WINDOW)}",
        "report_only": ["signed misses (below / above)", "month-cluster gate-3 "
                        "('uncalibrated, k < 20')", "coverage by DR group"]}


def doc_e02d() -> dict:
    ids = _unit_ids()
    s = _slots()
    win = L.WINDOWS["A"]
    unit = _unit("A", ids)
    test = {
        "test_id": L.E02D_TEST, "data_scope": _scope("A"), "fresh_units": [win["unit"]],
        "external_units": unit, "label": "prospective: the window's heads did not exist at "
                                         "registration",
        "claim": "C14a-r (new sub-claim): the per-well spread of heads around the county-wide "
                 "departure, at the 158 monitoring wells, 2026-12..2027-10",
        "what": "Does the frozen relative 90 % band (sigma_idio) cover the per-well part of "
                "the head error, per county, under a cluster-aware gate?",
        "metric": _relative_metric("A"),
        "decision_rule": {
            "pass": "the county's gate passes -> C14a-r S (prospective) for that county",
            "fail": "the gate fails -> the per-well wording is withdrawn for that county",
            "not_evaluable": f"< {L.MIN_WELLS_COUNTY} wells with >= {L.MIN_MONTHS} months, or "
                             "no window product granting prospective_scorer",
            "per_county": "Yunlin decides only Yunlin wording, Changhua only Changhua"},
        "ext_report_only": "N3F's never-calibrated wells (if their step-(5) rows exist): "
                           f"relative errors with the median over ext wells only (>= "
                           f"{L.MIN_EXT_WELLS_MONTH} reporting per county-month) and the "
                           "share inside N3F's own frozen half-width; report-only, never a "
                           "gate (an absolute width over a relative error)"}
    return {
        "id": L.E02D_ID, "wp": L.E02D_WP, "kind": "test",
        "title": "E02d: decomposed per-well (relative) head-band test, per county, "
                 "2026-12..2027-10, frozen forward cf2c7154",
        "why": "E02c's absolute coverage gate is weak by design: one shared year or month "
               "shock decides it (U1.HB.power.v1). Removing the county-wide (or group-wide) "
               "departure leaves a per-well estimand whose gate U1.decomp_power.v1 shows to be "
               "decisive. It merges bands P1-R2, P2-R1 and P3-R2 (round 7 judge rank 5).",
        "tests": [test],
        "interval": {
            "formula": "sigma_idio = sqrt(sd_fit_rel^2 + sigma_ens_rel^2 [+ tau^2 dt^2 where "
                       "DR's rule triggered])",
            "sd_fit_rel": "DR.relative_drift.v1 per-well RMS of the relative fitted-month "
                          "residual (same LOO operator and groups)",
            "sigma_ens_rel": "relative spread of the 36 rerun members (18 parameter sets x 2 "
                             "initial fields); heads_std is not used",
            "centre": f"E02's registered predicted_m ({L.E02_PRED_SHA[:8]}), no correction",
            "artefact": s["artefact"], "members": s["members"]},
        "binds": {"e02": {n: io.rec(io.REPO / L.E02_DIR / n)
                          for n in ("predictions.csv", "protocol.json", "manifest.json")},
                  "forward": {"path": L.FORWARD, "sha256": L.MODEL_SHA},
                  "dr": s["dr"], "power": s["power"],
                  "bound_code": {f: io.sha256(io.REPO / f) for f in BOUND},
                  "code_at_registration_report_only": {f: io.sha256(io.REPO / f)
                                                       for f in CODE_REPORT_ONLY},
                  "engineering": ENGINEERING_RESTORE},
        "frozen_code": {f: io.sha256(io.REPO / f) for f in FROZEN},
        "scoring": {
            "not_before": win["not_before"], "no_later_than": win["no_later_than"],
            "precondition": "the window heads product's MANIFEST lists prospective_scorer "
                            "(flags 0-1) and, if it declares parameters."
                            "prospective_scorer_window, that window covers 2026-12..2027-10; "
                            "checked from the MANIFEST alone; otherwise not_evaluable, "
                            "nothing read, no look",
            "order": f"in the 2028-01-15 session AFTER E02c ({E02C_ID}): E02c's report (or a "
                     "lead-written not-scorable record) must exist; then assert_can_score; "
                     "then E02's step-(5) rows are read",
            "scorer": f"{SCORER} score --window A --window-product <dir> --step5-dir <E02 "
                      "step-(5) out> --e02c-report <E02c report.json> --out <dir>"},
        "sealed_until_scoring": "every head of 2026-12..2027-10 at the 158 wells; NO 2023-26 "
                                "coverage of the relative recipe may ever be computed before "
                                "scoring",
        "registration_script": _registration_script(),
        "page_mapping": {
            "before_scoring": {"C14a-r": "N+D: 'decided by a registered test that is decisive "
                                         "by design (U1.decomp_power.v1); scored not before "
                                         "2028-01-15'",
                               "C14a": "E02c's absolute wording untouched (pass_weak / "
                                       "fail_shock)"},
            "pass": "C14a-r S (prospective) for that county: 'per-well spread around the "
                    "county-wide departure', never the full interval, never 'calibrated'",
            "fail": "the per-well wording is withdrawn for that county (N, coverage stated)",
            "not_evaluable": "C14a-r N, 'undecided'",
            "wording": WORDING},
        "disclosures": {
            "U7": "the combined family (ensemble spread and in-sample sd in quadrature) was "
                  "chosen after its 2023-26 coverage was seen; E02d is the same family moved "
                  "into relative space. Label (en/zh): " + json.dumps(U7, ensure_ascii=False),
            "multiplicity": MULTIPLICITY + "; E02d adds one",
            "design_time": "DR.relative_drift.v1 read the 2012-22 fitted residuals once "
                           "(logged) to set the grouping and drift term by declared rules",
            "members": "the fold members were rerun (model only, sealed stdout); the gate "
                       "reproduced the stored member mean within 1e-3 m",
            "n3f_ext": "the ext stratum is report-only; post-origin records may exist only at "
                       "the 158 wells, so it is likely not informative"},
        "attempt_policy": "engineering failures may be fixed and re-run before the score look, "
                          "each logged; the test is scored ONCE after the window closes; a "
                          "failed or not-evaluable result is reported as such and never "
                          "re-scored on this unit; at most one post-failure exploratory "
                          "follow-up (new ids, never validation); every attempt is listed",
        "rulings": {"R7.e02d_residual_read": "DR may set E02d's drift and grouping by the "
                                             "pre-declared rule",
                    "precedent": "the 2026-12 window is inside the prospective_scorer "
                                 "vocabulary ('Nov 2026 - Oct 2027'); the N3F-style December "
                                 "fallback precedent sets the 2026-11-30T16:00Z deadline"},
        "zones": NA("no fit"), "priors": NA("no prior"), "burn_in": NA("no fit"),
        "folds": NA("prospective window"), "fresh_units": [win["unit"]],
        "thresholds": {"target": L.TARGET, "gate_level": L.GATE_LEVEL,
                       "point_window": list(L.POINT_WINDOW), "min_months": L.MIN_MONTHS,
                       "min_wells_county": L.MIN_WELLS_COUNTY, "min_others": L.MIN_OTHERS,
                       "z": L.Z90},
        "gate_scores": NA("no weighted gate"),
        "primary_model": {"name": "datum_gate_innov (frozen forward), E02's registered "
                                  "centre", "forward_sha256": L.MODEL_SHA},
        "inputs": _inputs(s, ("A",)),
        "interpretation": "Per-well spread around the county-wide departure only: not the "
                          "level, not the policy contrast, not the full interval.",
        "written_by": THIS}


def _inputs(s: dict, wins) -> list[str]:
    out = {f"{L.E02_DIR}/{n}" for n in ("predictions.csv", "protocol.json", "manifest.json")}
    out |= {L.FORWARD, *FROZEN, *BOUND}
    for k in ("dr", "power"):
        for kk in ("report", "wells", "result"):
            if isinstance(s[k].get(kk), dict):
                out.add(s[k][kk]["path"])
    if s["members"].get("gate"):
        out |= {s["members"]["gate"]["path"], s["members"]["npz"]["path"]}
    if s["artefact"].get("summary"):
        out.add(s["artefact"]["summary"]["path"])
        out |= {r["path"] for n, r in s["artefact"]["files"].items()
                if any(n.endswith(f"_{w}.csv") for w in wins)}
    for w in wins:
        if w in ("B", "C") and "manifest.json" in s[f"abs_{w}"]:
            out |= {r["path"] for r in s[f"abs_{w}"].values()}
    if any(w in ("B", "C") for w in wins):
        out.add(L.RULINGS)
    return sorted(out)


def doc_e02y() -> dict:
    ids = _unit_ids()
    s = _slots()
    prim = (s["power"].get("years23_relative_primary") or {}) if s["power"]["status"] == "ok" \
        else {}
    tests = []
    for win in ("B", "C"):
        w = L.WINDOWS[win]
        unit = _unit(win, ids)
        tag = w["first"]
        tests.append({
            "test_id": f"E02yP.rel_{tag}", "data_scope": _scope(win),
            "fresh_units": [w["unit"]], "external_units": unit,
            "claim": f"C14b: per-well band (relative) at {w['first']}..{w['last']} "
                     f"({w['role']})",
            "metric": _relative_metric(win),
            "role_by_county": {c: ("primary" if prim.get(c) else "report-only")
                               for c in L.COUNTIES},
            "decision_rule": {"pass": "C14b S-limited for that county and these leads",
                              "fail": "C14b N with the coverage stated for that county",
                              "not_evaluable": "undecided"}})
        tests.append({
            "test_id": f"E02yP.abs_{tag}", "data_scope": _scope(win),
            "fresh_units": [w["unit"]], "external_units": unit,
            "claim": f"C14b secondary: absolute 90 % interval at {w['first']}..{w['last']}",
            "metric": {"rows": "the same step-(5) rows; hit = lower_m <= observed_m <= "
                               "upper_m of the registered absolute rows",
                       "wells": "every well of the absolute registration (E02's recipe, the "
                                "158-well unit) with scored rows; NOT restricted to wells "
                                "that have a relative artefact row (wells without "
                                "sd_fit_rel stay in the absolute test)",
                       "gate": "per county, the same gate as the relative test"},
            "decision_rule": {"pass": "pass_weak wording only ('no gross under-coverage "
                                      "detected'); never mapped to S",
                              "fail": "fail_shock wording",
                              "not_evaluable": "undecided"}})
    return {
        "id": L.E02Y_ID, "wp": L.E02Y_WP, "kind": "test",
        "title": "Years 2 and 3 of the frozen head forecast cf2c7154 (2027-11..2028-10, "
                 "2028-11..2029-10): relative per-well band per county (primary where the "
                 "12-month P-gate passed) and absolute interval (secondary)",
        "why": "C14b ('outside the first window the recipe is untested') has no decider. "
               "User ruling R7.heads_years23 extends heads prospective_scorer to "
               "2027-11..2029-10 for the same frozen forecast. Registered now, before any "
               "year-1 result, so year 1 cannot influence it (judge rank 6; bands P2-R2 and "
               "P3-R2 windows B/C).",
        "tests": tests,
        "artefacts": {"absolute": {w: s[f"abs_{w}"] for w in ("B", "C")},
                      "absolute_recipe": "python -m hydrophysics.twin.prospective register "
                                         "--test-start/--test-end on the same forward; the "
                                         "same call for 2026-12..2027-10 reproduced E02's "
                                         "rows of those months to 1e-9 m (abs_check)",
                      "relative": s["artefact"], "members": s["members"]},
        "binds": {"forward": {"path": L.FORWARD, "sha256": L.MODEL_SHA}, "dr": s["dr"],
                  "power": s["power"], "rulings": io.rec(io.REPO / L.RULINGS)
                  if (io.REPO / L.RULINGS).is_file() else "PENDING",
                  "bound_code": {f: io.sha256(io.REPO / f) for f in BOUND},
                  "code_at_registration_report_only": {f: io.sha256(io.REPO / f)
                                                       for f in CODE_REPORT_ONLY},
                  "engineering": ENGINEERING_RESTORE},
        "frozen_code": {f: io.sha256(io.REPO / f) for f in FROZEN},
        "report_only": {"lead_growth": "per county, coverage(year 2) - coverage(year 1) and "
                                       "(year 3) - (year 2), relative and absolute",
                        "twin_prospective_protocol": "the protocol.json of each year-2/3 "
                                                     "absolute registration (max_rmse_ratio "
                                                     "1.0 vs climatology, pooled "
                                                     "coverage_bounds [0.85, 0.95]) yields a "
                                                     "pass/fail from validation.evaluate in "
                                                     "step (5); it is report-only, never "
                                                     "page-mapped and not a test of this "
                                                     "registration (no multiplicity)",
                        "common_mode": "k = 2-3 year clusters: the shared (county-wide) part "
                                       "is declared NOT decidable",
                        "p_gate_12_months": "U1.decomp_power.v1 at 12 months, per county"},
        "scoring": {"B": {"not_before": L.WINDOWS["B"]["not_before"],
                          "no_later_than": L.WINDOWS["B"]["no_later_than"]},
                    "C": {"not_before": L.WINDOWS["C"]["not_before"],
                          "no_later_than": L.WINDOWS["C"]["no_later_than"]},
                    "order": "one session per window, all inside score_e02d.py: (1) date; "
                             "(2) the window product MANIFEST check (metadata only); (3) "
                             "prereg3.assert_can_score for the relative, then the absolute "
                             "test (both looks logged before any window row is read); (4) "
                             "hydrophysics.twin.prospective.score on the window's absolute "
                             "registration into <out>/step5, in-process, nothing printed; "
                             "(5) the relative test; (6) the absolute test. Nobody runs "
                             "'python -m hydrophysics.twin.prospective score' on the "
                             "year-2/3 registrations by hand",
                    "precondition": "a window product whose MANIFEST grants "
                                    "prospective_scorer AND declares parameters."
                                    "prospective_scorer_window = [first, last] (YYYY-MM) "
                                    "covering the window's months (R7.heads_years23; the "
                                    "vocabulary text 'Nov 2026 - Oct 2027' does not cover "
                                    "years 2-3 by itself); otherwise not_evaluable, nothing "
                                    "read, no look",
                    "scorer": f"{SCORER} score --window B|C --window-product <dir> "
                              "--new-wells <dir inside the window product> --out <dir> "
                              "[--previous-report <the previous window's report.json>]"},
        "page_mapping": {
            "now": "C14b N -> N+D: 'outside the first window, decided by registered tests for "
                   "2027-11..2029-10; untested beyond'",
            "relative_pass": "C14b S-limited per county for those leads",
            "relative_fail": "C14b N with the coverage stated; no interval wording at those "
                             "leads for that county",
            "absolute": "pass_weak / fail_shock only; never S", "wording": WORDING},
        "disclosures": {"U7": U7, "multiplicity": MULTIPLICITY + "; years 2-3 are new "
                                                                 "windows of the same "
                                                                 "forecast",
                        "same_origin": "both windows share the 2022-12 origin state; year "
                                       "clusters are not independent forecasts"},
        "attempt_policy": "as E02d: scored once per window, never re-scored; engineering "
                          "fixes logged",
        "registration_script": _registration_script(),
        "rulings": {"R7.heads_years23": "Yes (2026-11-01, sixth set)"},
        "zones": NA("no fit"), "priors": NA("no prior"), "burn_in": NA("no fit"),
        "folds": NA("prospective windows"),
        "fresh_units": [L.WINDOWS["B"]["unit"], L.WINDOWS["C"]["unit"]],
        "thresholds": {"target": L.TARGET, "gate_level": L.GATE_LEVEL,
                       "point_window": list(L.POINT_WINDOW), "min_months": L.MIN_MONTHS,
                       "min_wells_county": L.MIN_WELLS_COUNTY, "min_others": L.MIN_OTHERS},
        "gate_scores": NA("no weighted gate"),
        "primary_model": {"name": "datum_gate_innov (frozen forward)",
                          "forward_sha256": L.MODEL_SHA},
        "inputs": _inputs(s, ("B", "C")),
        "interpretation": "Lead-aware readings of the same frozen band; the relative part can "
                          "decide per county, the absolute part cannot.",
        "written_by": THIS}


def problems_for_write(which: str, d: dict, expect_self_sha, prereg_dir) -> list[str]:
    bad = []
    s = _slots()
    if s["dr"]["status"] != "ok":
        bad.append(f"DR: {s['dr']['status']}")
    elif not (s["dr"].get("declaration") or {}).get("sha256"):
        bad.append("DR's report was not written under its registration")
    if s["power"]["status"] != "ok":
        bad.append(f"power: {s['power']['status']}")
    else:
        reg = io.index_record(L.POWER_ID, prereg_dir)
        if reg is None or (s["power"]["declaration"] or {}).get("sha256") != reg["sha256"]:
            bad.append("the power result was not run under the registered declaration")
        if s["power"].get("dr_report_sha256") != (s["dr"].get("report") or {}).get("sha256"):
            bad.append("the power design was derived from a different DR report")
        if which == "e02d" and not s["power"].get("e02d_registrable"):
            bad.append("U1.decomp_power.v1 did not pass in both counties at 11 months: E02d "
                       "is NOT registered; record C14a-r N ('undecidable even after removing "
                       "the common mode')")
    if s["members"]["status"] != "ok":
        bad.append(f"members: {s['members']['status']}")
    if s["artefact"]["status"] != "ok":
        bad.append(f"artefact: {s['artefact']['status']}")
    elif s["artefact"]["dr_report_sha256"] != (s["dr"].get("report") or {}).get("sha256"):
        bad.append("the artefact was built from a different DR report")
    for n, h in (("predictions.csv", L.E02_PRED_SHA), ("protocol.json", L.E02_PROTOCOL_SHA)):
        if io.sha256(io.REPO / L.E02_DIR / n) != h:
            bad.append(f"E02 {n} changed")
    if io.sha256(io.REPO / L.FORWARD) != L.MODEL_SHA:
        bad.append("forward npz changed")
    if which == "e02y":
        ok, why = _rulings_ok("years23")
        if not ok:
            bad.append(why)
        for w in ("B", "C"):
            if "manifest.json" not in s[f"abs_{w}"]:
                bad.append(f"absolute registration {w} missing")
        if io.now() >= Y23_DEADLINE:
            bad.append("2027-11-01 has passed: year 2 may no longer be registered")
    elif io.now() >= io.DEADLINE:
        bad.append("DEADLINE PASSED (2026-11-30T16:00Z): do not register E02d")
    self_sha = io.sha256(Path(__file__))
    if expect_self_sha != self_sha:
        bad.append(f"--expect-self-sha {expect_self_sha!r} != {self_sha}")
    return bad


def check_blocking(d: dict, prereg_dir) -> list:
    idx = prereg3.membership_index(prereg_dir)
    for t in d["tests"]:
        for u, v in t["external_units"].items():
            if not idx.known(u):
                idx.members[idx.canon(u)] = prereg3._declared_members(v)
    return prereg3.blocking_looks(d["tests"], None, prereg_dir, idx, test_time=io.now())


def main(argv=None, prereg_dir=None, rows_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=("e02d", "e02y"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    pdir = Path(prereg_dir) if prereg_dir else io.prereg_dir_default()
    d = doc_e02d() if a.which == "e02d" else doc_e02y()
    errs = prereg3.validate_prereg(d, prereg_dir=pdir)
    block = check_blocking(d, pdir)
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("draft sha256:", io.sha_json(d))
    print("validate:", errs or "ok")
    print("blocking looks:", block or "none")
    print("this script sha256:", io.sha256(Path(__file__)))
    if not a.write:
        return 1 if errs else 0
    bad = problems_for_write(a.which, d, a.expect_self_sha, pdir)
    if errs or block or bad:
        print("REFUSED:", errs + [str(b) for b in block] + bad)
        return 1
    kw = {"prereg_dir": pdir}
    if rows_dir is not None:
        kw["rows_dir"] = rows_dir
    path, h = prereg3.write_prereg(d, **kw)
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
