"""Draft registration of E02z.level_years23.v1 (harness.prereg3 schema, kind 'test'). DRAFT
ONLY: nothing here writes a registration (write_prereg is never called); the lead validates,
runs the blocking probe and registers before 2027-11-01 (target December 2026), after the code
is merged and the pre-origin baselines are exported and verified.

``draft_prereg(seed)`` takes the lead's fresh seed (no default; previewed, smoked and
elsewhere-registered seeds are refused). The seed is E02z's block-bootstrap seed (the month
secondary uses seed + 1, as E02c) and the decidability reading's world seed.

    python -I dataprep_staging/e02z/prereg_e02z.py --seed N --validate
    python -I dataprep_staging/e02z/prereg_e02z.py --seed N --probe     (read-only)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "dataprep_staging" / "e02c"), str(REPO / "dataprep_staging"),
           str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e02z_const as K  # noqa: E402
import e02z_lib as L  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

THIS = f"{K.X}/prereg_e02z.py"
OWN = (f"{K.X}/e02z_const.py", f"{K.X}/e02z_lib.py", f"{K.X}/decide_e02z.py",
       f"{K.X}/score_e02z.py", f"{K.X}/export_baselines_e02z.py", f"{K.X}/run_e02z.sh")
FROZEN = (*OWN, K.E02C_SCORER, K.E02C_EXPORT, K.GOV,
          "dataprep_staging/fv_decidability/osse_fv.py",
          "dataprep_staging/fv_decidability_v2/osse_fv2.py")
REPORT_ONLY_CODE = ("hydrophysics/twin/validation.py", "hydrophysics/twin/prospective.py",
                    "hydrophysics/twin/drift_diag.py", "dataprep_staging/harness/prereg3.py",
                    "dataprep_staging/common/guard.py", "dataprep_staging/common/manifest.py",
                    "dataprep_staging/common/uses.py")
PREREG = "results/dataprep/preregistration"

MULTIPLICITY = (
    "the 158 wells' heads of 2027-11..2029-10 are also scored by E02y (relative per-well band, "
    "absolute interval, and the report-only twin.prospective protocol verdict inside step (5)); "
    "the same frozen forecast's year 1 is scored by E02, E02c (T-cov, T-level), E02d, N2, N2C, "
    "N2CS and the hybrid/X1/I1 tests; E02z adds one level decision per county and window; "
    "each test decides only its own claim, none is selected post hoc, and no family-wise or "
    "combined claim is made")
U7 = ("nothing from 2023-26 enters the rule, the baselines, the wells, the seed or the "
      "thresholds: CL/CLT are fitted on 2012-01..2022-12 only (E02c's code), ZR is E02y's "
      "registered rows, and no 2023-26 residual is read or recomputed. The published RC5 numbers "
      "(pooled over both counties: frozen run 2.048 m against CL 1.667 m on 2023-26) are "
      "quoted only to pre-word the expected narrowing; the rule is E02c's, fixed 2026-10-29 "
      "before those numbers were summarised for this purpose")
NON_CODE_READS = (
    "while building (2026-11-03) the builder read: docs/TWIN_ROUND8_PLAN.local.md (the #17 entry, "
    "the 'C30 years 2-3' audit finding, R8.7, R8.14 and the F2/RC5 line); the full registration "
    "E02y.years23_2027-11_2029-10.v1 (including its unit ids and its DR/power design-time "
    "decisions); E02c.prospective_2026-11.v1 (lists shortened); the key names, frozen_code, "
    "created and seed fields of GOV.heads_window_product.v1, FV.decidability.v1/v2, FV.ext.v1, "
    "FV.stat.v2, N2CS.osse_design.v1 and FM.origin_memory.v1, and FV v1's input path list; "
    "user_rulings_2026-11-03.json (R8.1a-R8.4a, R8.7, R8.14) and user_rulings_2026-11-01b.json "
    "(R7.heads_years23, R7.n3_g0prime); the header and first rows of results/twin/prospective_"
    "2027-11_y2/predictions.csv (one predicted_m value and part of a second: model output) and "
    "that directory's protocol.json and manifest.json; directory listings of results/; sha256 "
    "sums of the y2/y3 files, the rulings, the registrations above, E02c's wells_158.csv and FV "
    "v1's input CSVs (hashes only). No observation value, window head, post-origin datum or "
    "2023-26 residual was read, and no statistic was computed on a real seed (the engine ran "
    "only on its synthetic smoke context)")


def _na(reason: str) -> dict:
    return {"not_applicable": reason}


def unit_ids(prereg_dir: Path | None = None) -> list[str]:
    """E02y's 158 ids (the same wells, new unit ids), from E02y's pinned registration."""
    d = Path(prereg_dir) if prereg_dir else L.rp(PREREG)
    p = d / f"{K.E02Y_ID}.json"
    if L.sha256(p) != K.E02Y_SHA:
        raise ValueError(f"{p} is not E02y's registered file")
    doc = json.loads(p.read_text(encoding="utf-8"))
    sets = {tuple(sorted(map(str, t["external_units"][u]["ids"])))
            for t in doc["tests"] for u in t["external_units"]}
    if len(sets) != 1:
        raise ValueError("E02y's units do not hold one well list")
    ids = list(next(iter(sets)))
    if len(ids) != K.N_WELLS:
        raise ValueError(f"E02y lists {len(ids)} wells, expected {K.N_WELLS}")
    return ids


def baselines_record(check_files: bool) -> dict:
    """The exported pre-origin baselines (export_baselines_e02z.py, verified)."""
    d = L.rp(K.BASE_DIR)
    if not check_files:
        return {"dir": K.BASE_DIR, "status": "pending export (draft built without files)",
                "files": {}}
    miss = [n for n in K.BASE_FILES if not (d / n).is_file()]
    if miss:
        raise ValueError(f"export the baselines first ({miss} missing in {K.BASE_DIR})")
    summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    ver = json.loads((d / "verify.json").read_text(encoding="utf-8"))
    if not ver.get("identical"):
        raise ValueError("verify.json does not report a bit-identical rebuild")
    if summ["files"].get(K.BASE_FILE) != L.sha256(d / K.BASE_FILE):
        raise ValueError("baselines_e02z.csv differs from its summary")
    return {"dir": K.BASE_DIR, "status": "ok",
            "files": {K.BASE_FILE: L.sha256(d / K.BASE_FILE),
                      "summary.json": L.sha256(d / "summary.json"),
                      "verify.json": L.sha256(d / "verify.json")},
            "reference": {w: v.get("chosen") for w, v in summ["reference"].items()},
            "verify": {"identical": True, "verified": ver.get("verified")}}


def _inputs(check_files: bool) -> list[str]:
    paths = [f"{PREREG}/{K.E02Y_ID}.json", f"{PREREG}/{K.E02C_ID}.json",
             f"{PREREG}/{K.GOV_ID}.json", f"{PREREG}/{K.FM_ID}.json",
             f"{PREREG}/{K.FV1_ID}.json", f"{PREREG}/{K.FV2_ID}.json", K.RULINGS, K.RULINGS_R7,
             *[f"{K.ABS_DIRS[w]}/{n}" for w in K.WINDOWS for n in K.ABS_SHA[w]],
             K.E02C_WELLS, *[f"{K.FV_INPUTS}/{n}" for n in K.FV_INPUT_SHA],
             *[f"results/twin_runs/{K.ZR0_DIR}/{t}" for t in K.ZR0_THETAS],
             f"results/twin_runs/{K.ZR0_DIR}/{K.COLUMN}",
             *[f"results/twin_runs/{d}/{n}" for _, d in K.TRUTHS
               for n in ("stage3_theta.json", K.COLUMN)],
             *FROZEN]
    if check_files:
        paths += [f"{K.BASE_DIR}/{n}" for n in K.BASE_FILES]
        paths += [K.fv1_npz(t) for t in (K.ZR0, *K.TAGS)]        # S4: hashed by prereg3
    return list(dict.fromkeys(paths))


def _test(win: str, ids: list[str], seed: int) -> dict:
    first, last = K.WINDOWS[win]
    unit = {K.UNIT[win]: {
        "description": f"monthly heads {first}..{last} (flags 0-1) at E02's 158 wells (E02y's "
                       f"member list, a new unit id beside {K.E02Y_UNIT[win]})",
        "ids": ids, "window": f"{first}..{last}"}}
    return {
        "test_id": K.TEST[win], "data_scope": K.SCOPE[win], "fresh_units": [K.UNIT[win]],
        "external_units": unit,
        "claim": f"C30-scen, heads part: the frozen model's average-weather level on "
                 f"{first}..{last} (year {2 if win == 'B' else 3}), per county",
        "label": "prospective: the window's heads did not exist at registration",
        "what": "Does the frozen forecast's average-weather level (E02y's registered "
                "predicted_m, forward cf2c7154) beat BOTH the pre-origin climatology CL and "
                "climatology-plus-trend CLT, in each county on its own?",
        "role_by_county": {c: "primary" for c in K.COUNTIES},
        "metric": {
            "rows": "exactly E02y's step-(5) rows of this window (<E02y out>/step5: "
                    "evaluation.csv and report.json written by score_e02d.py in-process, "
                    "--new-wells inside the window product), verified by score_e02c."
                    "load_e02_step5 against the registered predictions/protocol/manifest "
                    "(sha256 pinned); E02z reads no heads table of its own",
            "county": "station code 09 Yunlin, 07 Changhua (score_e02c.county_of)",
            "baselines": "CL, CLT = twin.drift_diag.fair_baselines (clim, clim_trend; "
                         "month_offset 0) of heads/L1_clean/v3 flags 0-1 opened with use "
                         "'baseline' and window_end 2022-12 (pre-origin months only), on this "
                         "window's months, through E02c's export code (obs_matrix, "
                         "baselines_table, reference_rule; imported read-only); frozen in "
                         "baselines_e02z.csv (rebuilt bit-identically, hashed under "
                         "'baselines')",
            "rmse": "equal-well RMSE (E02's convention) over the county's rows with finite CL "
                    "and CLT, wells with >= 6 such months (score_e02c.with_baselines, _rmse3)",
            "interval": f"score_e02c.paired_bootstrap of MSE(ZR) - MSE(baseline): wells "
                        f"resampled by 5 km TWD97 blocks (E02c's wells_158.csv block5) within "
                        f"the county, B 2000, seed {seed}, central 90 %; declared secondary: "
                        f"months resampled (seed {seed + 1}), report-only"},
        "decision_rule": {
            "level_supported": "for a county: RMSE(ZR) < RMSE(CL) AND RMSE(ZR) < RMSE(CLT) AND "
                               "both block-bootstrap 90 % intervals of the MSE difference lie "
                               "below 0",
            "level_not_supported": "otherwise (the expected outcome: a pre-worded narrowing)",
            "level_not_evaluable": f"fewer than {K.MIN_WELLS_COUNTY} wells of the county with "
                                   f">= {K.MIN_MONTHS} window months and finite CL/CLT, or "
                                   f"fewer than {K.MIN_BLOCKS_COUNTY} distinct 5 km blocks "
                                   "among them (review S5, lead); no "
                                   "window product granting the window (precondition); "
                                   "E02y's window not evaluable or not scorable",
            "per_county": "Yunlin decides only Yunlin wording, Changhua only Changhua wording; "
                          "no pooled verdict"},
        "informative_gate": f">= {K.MIN_WELLS_COUNTY} wells with >= {K.MIN_MONTHS} months and "
                            f">= {K.MIN_BLOCKS_COUNTY} distinct 5 km blocks per county",
        "report_only": ["the month-resampled interval", "RMSE ZR / CL / CLT per county",
                        "E02y's step-(5) protocol verdict (passed, rmse_ratio, coverage), "
                        "unchanged"],
        "seed": seed}


def fv1_forward_problems() -> list[str]:
    """Review S4: FV v1's gate-E reference forwards at their pinned sha256."""
    out = []
    for t in (K.ZR0, *K.TAGS):
        p = L.rp(K.fv1_npz(t))
        if not p.is_file() or L.sha256(p) != K.FV1_FWD_SHA[t]:
            out.append(f"{K.fv1_npz(t)} missing or not at {K.FV1_FWD_SHA[t][:8]}")
    return out


def draft_prereg(seed, *, prereg_dir: Path | None = None, ids: list[str] | None = None,
                 check_files: bool = True, now=None) -> dict:
    """The draft registration. ``seed`` is required (the lead's fresh seed). Refuses at or
    after REG_DEADLINE (review S6). ``check_files`` hashes the frozen code and the exported
    baselines (and refuses if the GOV module, E02c's scorer or FV v1's reference forwards are
    not at their pinned sha256)."""
    seed = L.check_seed(seed)
    now = now or L.now_utc()
    if now >= L.parse_utc(K.REG_DEADLINE):
        raise ValueError(f"E02z must be registered before {K.REG_DEADLINE} (review S6)")
    if check_files:
        bad = fv1_forward_problems()
        if bad:
            raise ValueError("FV v1's reference forwards: " + "; ".join(bad))
        for f, want in ((K.GOV, K.GOV_SHA), (K.E02C_SCORER, K.E02C_SCORER_SHA),
                        (K.E02C_EXPORT, K.E02C_EXPORT_SHA)):
            if L.sha256(L.rp(f)) != want:
                raise ValueError(f"{f} is not at its pinned sha256 {want[:8]}")
    ids = sorted(map(str, ids)) if ids is not None else unit_ids(prereg_dir)
    if len(ids) != K.N_WELLS:
        raise ValueError(f"expected {K.N_WELLS} ids")
    nb = {w: K.NOT_BEFORE[w][:10] for w in K.WINDOWS}
    scoring = {
        w: {"not_before": K.NOT_BEFORE[w], "no_later_than": K.NO_LATER_THAN[w],
            "as": f"E02y's window {w} dates (not before the window's end plus the session "
                  "date); the same session, after E02y's window report"} for w in K.WINDOWS}
    scoring.update({
        "registration_deadline": K.REG_DEADLINE,
        "register_before": "2027-11-01 (target December 2026), before any window product is "
                           "read and before E02c's year-1 level verdict (2028-01-15), so no "
                           "window result can inform the design",
        "order": "one session per window, after E02y's scorer: (1) date, deadline and "
                 "registration checks; (2) E02y's window report and score looks; (3) the "
                 "MANIFEST precondition (its sha256 recorded); (4) pre-look work: the "
                 "baselines, wells, blocks, seed and E02y's prediction files are read and "
                 "checked, then the step-(5) rows' contract (metadata, station_id and date "
                 "only); (5) prereg3.assert_can_score (the window's test); (6) the rows are "
                 "verified and each county decided; the report written once",
        "precondition": f"E02z's own, from the MANIFEST alone (R8.14; N2CS precedent): E02c's "
                        f"frozen score_e02c.product_grants_scorer (sha "
                        f"{K.E02C_SCORER_SHA[:8]}) AND {K.GOV_ID}'s public "
                        f"heads_window.covering_rule(m, window) (sha {K.GOV_SHA[:8]}); both "
                        "files in frozen_code and checked at scoring; uses.py is not edited. "
                        "If E02y did not look at the window, a failure -> level_not_evaluable, "
                        "nothing read, no look; if E02y looked, a failure (a wrong or "
                        "non-covering product) is REFUSED as an engineering error (review "
                        "S3). The MANIFEST's sha256 is recorded (review Q6)",
        "e02y_order": "E02y's window report naming E02y and the window must exist. If E02y's "
                      "score looks of the window are in looks.jsonl, the report must be a "
                      "scored one and <E02y out>/step5's report.json and evaluation.csv must "
                      "match its rows_provenance sha256 (review Q6); a 'not_scorable' or "
                      "'not_evaluable' record is then REFUSED (review S3). Without those "
                      "looks, E02y's own 'not_evaluable' record or a lead-written "
                      "'not_scorable' record -> level_not_evaluable without a look, and a "
                      "scored report is REFUSED",
        "pre_look_checks": "every non-observation read before the look (review S7): the "
                           "baselines' sha256 and table, E02c's well table and blocks, the "
                           "registered seed, E02y's prediction files; then step-(5) "
                           "report.json carries the registered manifest; evaluation.csv has "
                           "station_id, date, predicted_m, lower_m, upper_m, baseline_m, "
                           "observed_m; station ids are 8-character strings with their "
                           "leading zeros and all among the 158; every date parses to a "
                           "month; >= 1 row in the window. The rows are read from the header "
                           "and the station_id / date columns only; a failure is REFUSED "
                           "(engineering, before the look; N2CS B2 precedent)",
        "after_look": "an environment failure before any statistic (a missing file, an "
                      "OSError) is logged and the window rerun with unchanged code; any other "
                      "exception after assert_can_score writes the final report "
                      "'engineering_failure_after_look' (disclosed, never re-scored; review "
                      "Q4)",
        "if_order_breached": "scored once whatever; a session after no_later_than is flagged "
                             "beside the verdicts and the delay reported",
        "scorer": f"{K.X}/score_e02z.py score --window B|C --window-product <dir> "
                  "--e02y-report <E02y out>/report.json",
        "write_once": f"{K.SCORE_REPORT['B']} and {K.SCORE_REPORT['C']} (mode 'x'); attempts in "
                      f"{K.SCORE_ATTEMPTS}"})
    decidability = {
        "label": "fresh first-attempt synthetic decidability reading (model-only, CPU scoring); "
                 "FV v1 / v2 code imported read-only, FV.stat.v2's fresh-seed and attempts "
                 "pattern",
        "seed": seed, "k_truth": K.K_TRUTH, "k_null": K.K_NULL, "p_weak": K.P_WEAK,
        "streams": {"noise": K.STREAM_NOISE, "null": K.STREAM_NULL,
                    "selfcheck": K.STREAM_SELFCHECK},
        "truths": [list(t) for t in K.TRUTHS],
        "world": "obs = ZR + D_T + e on 2027-11..2029-10: ZR = E02y's registered rows; D_T = "
                 "(h_T - h_ZR0) at each well's cell and layer minus its mean over the well's "
                 "observed 2012-22 months (FV v1 obs_mask.csv); e = per-well AR(1) (FV v1 "
                 "noise_ar1.csv), one 24-month path per world shared by both windows",
        "forwards": {"commit": K.WT_HEAD, "worktree": f"$HOME/{K.WT_NAME}",
                     "flags": K.FWD_FLAGS,
                     "zr0": f"results/twin_runs/{K.ZR0_DIR} {list(K.ZR0_THETAS)} (the page "
                            "recipe's 18 parameter sets), baseline only",
                     "truths": "one stage3_theta.json each, baseline only, its column",
                     "outputs": {t: K.fwd_npz(t) for t in (K.ZR0, *K.TAGS)},
                     "hindcast_identity": "each 2012-01..2022-12 at the 158 wells equals FV "
                                          "v1's forward of the same run to 0.0 m (gate E)",
                     "reference_forwards_sha256": {K.fv1_npz(t): K.FV1_FWD_SHA[t]
                                                   for t in (K.ZR0, *K.TAGS)},
                     "device": K.CUDA_ONLY},
        "gates": {"E": "registration, frozen code, input sha256s, forward metadata, hindcast "
                       "identity, the engine's world path equal to the scorer's on "
                       f"{K.N_SELFCHECK} self-check worlds; a failure writes nothing",
                  "N": "per county and window, null worlds obs = CL + e (K_NULL): false-pass "
                       "rate <= FV v1's null_tolerance on that county's ZR - CL, else "
                       "not_evaluable",
                  "MC": "1.645 SE <= 0.07 (non-binding at this K; bug check)"},
        "reading_rule": "per county and window: P = equal-weight mean over the four truths' "
                        "pass rates; P <= 0.5 -> 'weak_by_design', otherwise 'not_shown_"
                        "weak'; applied once, one-sided",
        "role": "adds only the registered suffix (DECIDE_SUFFIX) to E02z's line and the "
                "plan's grading (not_shown_weak 0.7, otherwise 0.5; bookkeeping); it never "
                "changes the decision rule, a threshold or a well",
        "when": f"after registration and before {K.NOT_BEFORE['B']} (window B's session); "
                "decide_e02z.run refuses at or after that instant and once any E02z score "
                "report exists, and the scorer reads a decidability report created at or "
                "after it (or without 'created') as 'not_run' (suffix and grade declared; "
                "review B1)",
        "runner": f"bash {K.X}/run_e02z.sh setup | forward [ZR0|R0|Q0|R1c|X5R1] | decide "
                  "SEED",
        "report": K.DECIDE_REPORT, "attempts": K.DECIDE_ATTEMPTS,
        "limits": ["no pumping-anomaly arm and no recorded weather: interannual weather "
                   "departures enter only through e (optimism)",
                   "noise independent between wells (optimism)",
                   "the noise is hindcast misfit that already holds structural error D_T adds "
                   "(pessimism)",
                   "every well observed in every window month (optimism)",
                   "FV v2's worlds were less climatological than the published 2023-26 record "
                   "(CL RMSE 2.21-2.23 m against 1.667 m): the truths may sit farther from "
                   "CL than reality, so P may be optimistic for a pass"]}
    first_b, last_c = K.WINDOWS["B"][0], K.WINDOWS["C"][1]
    doc = {
        "id": K.ID, "wp": K.WP, "kind": "test", "title": K.TITLE,
        "why": "The 2026-11 audit ('forecast start and validation, C30 years 2-3'): C30's heads "
               "level is decided only for 2026-11..2027-10 (E02c T-level); E02y registers "
               "only intervals for 2027-11..2029-10 and its level reading is report-only, "
               "never page-mapped. E02z gives the baseline head projection beyond the first "
               "year a decider under the existing ruling R7.heads_years23 (R8.14).",
        "tests": [_test(w, ids, seed) for w in K.WINDOWS],
        "fresh_units": [K.UNIT[w] for w in K.WINDOWS],
        "scoring": scoring,
        "baselines": baselines_record(check_files),
        "governance": {"ruling": "R8.14 (confirmed 2026-11-03T01:26Z): covered by "
                                 "R7.heads_years23 through the registered "
                                 f"{K.GOV_ID} without editing uses.py",
                       "gov_registration_sha256": K.GOV_REG_SHA,
                       "heads_window_py": {"path": K.GOV, "sha256": K.GOV_SHA},
                       "precedent": "N2CS score_n2cs.precondition (lead ruling B1)"},
        "decidability": decidability,
        "page_mapping": {
            "claim": "C30-scen (R8.7 split: C30-scen = the physics average-weather scenario, "
                     "N; C30-fc = the first-year forecast of record, D via E04)",
            "before_scoring": {
                "C30-scen": f"N with D sub-label 'strict level rule years 2-3 (E02z), per "
                            f"county; scored not before {nb['B']} ({first_b}..2028-10) and "
                            f"{nb['C']} (2028-11..{last_c})'",
                "ledger": "E02z row, status D"},
            "level_supported": "C30-scen heads part -> S-limited for that county and window "
                               "('average-weather scenario level beat CL and CLT')",
            "level_not_supported": "C30-scen stays N, decided (narrowed): the pre-worded line; "
                                   "the expected outcome",
            "level_not_evaluable": "C30-scen stays N, 'undecided'",
            "wording": K.WORDING, "decidability_suffix": K.DECIDE_SUFFIX,
            "grade_bookkeeping": K.GRADE,
            "expected_outcome": {
                "reading": "narrowing: level_not_supported in both counties is the expected "
                           "outcome",
                "why": "RC5: the published 2023-26 record shows the frozen run at 2.048 m "
                       "against CL 1.667 m (pooled over both counties, equal-well RMSE)",
                "rc5": K.RC5},
            "never": ["C30-fc is never moved by E02z (it is E04's)",
                      "physics head levels after 2023-11 are never called a forecast "
                      "(R8.7 'average-weather scenario, not a forecast')",
                      "never 'calibrated' / '校準'",
                      "no interval, subsidence, policy-response or aquifer claim",
                      "E02c's year-1 T-level verdict and E02y's verdicts are unchanged",
                      "a pass in year 2 or 3 does not make the physics run the forecast of "
                      "record"]},
        "rulings": {"R8.14": {"file": K.RULINGS, "sha256": K.RULINGS_SHA,
                              "decision": "Confirmed"},
                    "R8.7": {"file": K.RULINGS, "sha256": K.RULINGS_SHA,
                             "decision": "C30 -> C30-scen (N) and C30-fc (D via E04)"},
                    "R7.heads_years23": {"file": K.RULINGS_R7, "sha256": K.RULINGS_R7_SHA,
                                         "decision": "Yes (2026-11-01, sixth set)"}},
        "reading_with_fm": {
            "note": f"{K.FM_ID} (sha {K.FM_SHA[:8]}): past the start memory L*, a level "
                    "'forecast' is the model's average-weather scenario (its parameters, "
                    "datum and the climatology forcing), not the 2022-12 observation update. "
                    "At leads 59-82 months E02z therefore tests the model's average-weather "
                    "level and datum",
            "case_dependence": "if FM's case is 'longer_than_record', the line adds 'start "
                               "memory not shown ended in this model'; the verdict and rule "
                               "do not change"},
        "disclosures": {
            "multiplicity": MULTIPLICITY, "U7": U7,
            "same_origin": "both windows share the 2022-12 origin state and the same wells; "
                           "the two window verdicts are not independent forecasts",
            "same_rows_as_e02y": "E02z scores E02y's step-(5) rows; E02y's score look of the "
                                 "window precedes E02z's in the same session (a "
                                 "post-registration look, reported, not blocking)",
            "window_product": "E02y's scorer and E02z each check the product they are given; "
                              "the lead passes the same window product directory to both "
                              "(E02z records its path)",
            "pre_look_read": "the scorer reads the step-(5) header and the station_id / date "
                             "columns seconds before assert_can_score; no observed_m value",
            "baseline_reference_rule": "E02c's declared rule is re-applied per window "
                                       "(pre-origin climatologies only); its choice is "
                                       "recorded under 'baselines'",
            "per_county_minimum": f"{K.MIN_WELLS_COUNTY} wells (E02c's per-county envelope, "
                                  "E02y's per-county minimum), not E02c's pooled 100, and "
                                  f"{K.MIN_BLOCKS_COUNTY} distinct 5 km blocks (review S5)",
            "builder_reads": NON_CODE_READS},
        "attempt_policy": "engineering failures (crash, bug, missing file) may be fixed and "
                          "re-run before the window's score look, each logged; each window is "
                          "scored ONCE; a not-supported, not-evaluable or after-look "
                          "engineering result is reported as such and never re-scored on "
                          "these units; at most ONE post-failure exploratory follow-up (new "
                          "ids, pre-declared alternatives only, never validation, never this "
                          "window); the decidability reading runs once (engineering retries "
                          "logged in its attempts.jsonl); after the score look (review Q4) an "
                          "environment failure before any statistic is computed (a missing "
                          "file, an OSError) is logged and the window is rerun with unchanged "
                          "code (the rerun logs its own look, listed); every other failure "
                          "after the look is a final 'engineering_failure_after_look' report; "
                          "a registration created at or after the deadline is refused by the "
                          "draft, the engine and the scorer (review S6); every attempt is "
                          "listed",
        "environment": {"numpy": np.__version__, "pandas": pd.__version__,
                        "python": sys.version.split()[0],
                        "note": "recorded at drafting (review Q4); the scorer records its own"},
        "zones": _na("no fit"), "priors": _na("no prior enters the statistic"),
        "burn_in": _na("no fit"), "folds": _na("prospective windows"),
        "thresholds": {"min_wells_county": K.MIN_WELLS_COUNTY,
                       "min_blocks_county": K.MIN_BLOCKS_COUNTY, "min_months": K.MIN_MONTHS,
                       "boot_B": K.BOOT_B, "boot_ci": K.BOOT_CI, "boot_seed": seed,
                       "block_km": K.BLOCK_KM, "p_weak": K.P_WEAK},
        "gate_scores": _na("no weighted gate; each county's verdict is reported as it falls"),
        "primary_model": {"name": "datum_gate_innov (frozen forward), E02y's registered "
                                  "year-2/3 predictions", "forward_sha256": K.MODEL_SHA,
                          "predictions_sha256": {w: K.ABS_SHA[w]["predictions.csv"]
                                                 for w in K.WINDOWS}},
        "frozen_code": ({f: L.sha256(L.rp(f)) for f in FROZEN} if check_files else {}),
        "code_at_registration_report_only": (
            {f: L.sha256(L.rp(f)) for f in REPORT_ONLY_CODE if L.rp(f).is_file()}
            if check_files else {}),
        "engineering": "a git-tracked module listed in code_at_registration_report_only may be "
                       "restored to its registered sha256 before the scoring run as an "
                       "engineering step, logged in the report; frozen code is never edited",
        "inputs": _inputs(check_files),
        "interpretation": "Whether the frozen model's average-weather head level beats both "
                          "simple pre-origin baselines in years 2 and 3, per county. A pass "
                          "is about the average-weather scenario, never the forecast of "
                          "record; the expected outcome is a narrowing. Nothing about "
                          "intervals, subsidence, policy responses or the aquifer.",
        "written_by": THIS}
    return doc


def validate(doc: dict) -> list[str]:
    from harness import prereg3

    return prereg3.validate_prereg(doc)


def blocking_probe(doc: dict) -> list:
    """Read-only: the looks that would block this draft now (write_prereg's own check)."""
    from harness import prereg3

    idx = prereg3.membership_index()
    for t in doc["tests"]:
        for u, v in (t.get("external_units") or {}).items():
            m = prereg3._declared_members(v)
            if m is not None and not idx.known(u):
                idx.members[idx.canon(u)] = m
    return prereg3.blocking_looks(doc["tests"], None, idx=idx,
                                  test_time=dt.datetime.now(dt.timezone.utc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, required=True, help="the lead's fresh seed")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--without-files", action="store_true",
                    help="preview: no frozen-code hashes, no baselines (never registrable)")
    a = ap.parse_args(argv)
    try:
        doc = draft_prereg(a.seed, check_files=not a.without_files)
    except (ValueError, L.Refused) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    missing = [p for p in doc["inputs"] if not L.rp(p).exists()]
    if missing:
        print(json.dumps({"inputs_missing": missing}, indent=1))
    if a.validate:
        print(json.dumps({"errors": validate(doc)}, indent=1))
    if a.probe:
        print(json.dumps({"blocking": blocking_probe(doc)}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
