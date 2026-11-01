"""Declaration FM.origin_memory.v1, kind 'declaration', before any curve is computed: for how
many months does the page model's head level remember its December 2022 start?

Model-only, heads only, no GPU. It reads the forward runs FS.start_invariance.v1 already wrote
and scored (results/twin_forward/fs_start_S0/S1/S2.npz), through a key whitelist (heads_mean,
dates, scenario_names, mask, origin). No observation of any family is read or scored. It
moves no claim's status; it feeds E23's horizon, C30's heads wording and (under a page-line
ruling) an E02/E21 interpretation clause (user ruling 2026-10-29: the ledger governs page
wording).

``--validate`` (dry run) prints the document, its canonical sha256 and this script's sha256,
runs prereg3.validate_prereg, checks the scorer's binding, runs prereg3.blocking_looks as a
dry run and reports readiness; nothing is written. ``--write`` registers it once
(prereg3.write_prereg; the lead only). From the repository root:

    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fm_origin_memory/prereg_fm.py \\
        --validate
    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fm_origin_memory/prereg_fm.py \\
        --write --expect-self-sha <sha256 of this file as reviewed>

then score once (CPU, a few minutes):

    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fm_origin_memory/score_fm.py \\
        --out results/dataprep/reports/FM.origin_memory.v1/report.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(HERE))

import fm_common as C  # noqa: E402
import score_fm as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable

PREREG_ID = S.PREREG_ID
X = C.X
FROZEN = S.FROZEN
TESTS = f"{X}/tests/test_fm_origin_memory.py"
REPORT = S.DEFAULT_OUT
X5_WELLS = "dataprep_staging/x5_class_share/wells_frozen.json"
X5_WELLS_ROWS_SHA256 = "7ce6a13dd3b4381f972f180c8bad8ea6ddfe9680bf0a9089c80eecf8152cd335"
FS_DECL = ("results/dataprep/preregistration/FS.start_invariance.v1.json",
           "9f8a71155cc9be83c0e23a9d61212ff95203d6d52934dab60d7f77b6a035d7b4")
FS_REPORT = ("results/dataprep/reports/FS.start_invariance.v1/report.json",
             "6130a3bf88a3889d09a982067418db69bdd144e1280c776d4c133f8179584ecd")
SCORE_CMD = (f"~/miniforge3/envs/hydro/bin/python -I {X}/score_fm.py --out {REPORT}")


def sha256(p: Path) -> str:
    return C.sha256(p) if p.exists() else "missing"


def frozen_code() -> dict:
    return {p: sha256(REPO / p) for p in FROZEN}


def canonical_sha(d: dict) -> str:
    return hashlib.sha256((json.dumps(d, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


INPUT_SHA256 = {**{p: h for p, h in S.FILES.values()}, S.REF[0]: S.REF[1],
                FS_DECL[0]: FS_DECL[1], FS_REPORT[0]: FS_REPORT[1]}

DISCLOSED = {
    "fs_endpoints_seen": (
        "FS.start_invariance.v1's report (report-only block baseline_level_shift_m) printed "
        "the baseline level shift, arm minus S0, at 2023-01, 2026-12 and 2032-12 for S1 and "
        "S2 (N2 unit and fan layer 2); the round-7 plan quotes it as about +0.77 m in "
        "2023-01 and about 1e-6 m by 2026-12 (an endpoint e-fold of about 3.5 months). "
        "These endpoints were seen before this declaration; the per-lead curve between them "
        "and the per-screen p90 have not been computed by anyone"),
    "fs_lever_seen": (
        "FS's gated result (the policy lever does not depend on the start, largest |R - 1| "
        "9.74e-06 %) is in the evidence record; it is about the policy difference, not the "
        "level, and is not used here"),
    "fs_files": (
        "the three npz files are the ones FS.start_invariance.v1 scored (its report lists "
        "these sha256; FS registered 9f8a7115, report 6130a3bf). FS's reproduction gate "
        "recorded max |S0 - page run| = 0.0 m for the baseline at the 158 screens"),
}


def doc() -> dict:
    frozen = frozen_code()
    return {
        "id": PREREG_ID, "wp": "FM", "kind": "declaration",
        "title": "Origin memory of the page model's head level: the first month after which "
                 "the observation update at the December 2022 start moves heads by less than "
                 "0.05 m at 90 % of the 158 frozen screens, for both the free-run start (S1) "
                 "and the old field restart (S2), at every later month through 2026-12 "
                 "(existing FS forwards; model-only, moves no claim status)",
        "priors": (
            "Model-only. The head solver is linear in the start state, so a start difference "
            "decays with the aquifer's diffusion times and does not depend on the scenario. "
            "From FS's endpoints (disclosed) the N2-unit mean falls below 0.05 m after about "
            "10 months; the per-screen p90 may decay more slowly in the confined deep layers "
            "(3-4) and under S2, whose untapered restart wrote a median -21/-25 m into the "
            "unmeasured proximal layers. Planners expected L* about 8-14 months (P3-A) or "
            "12-24 months (P2-R1). No prior is used: the outcome wordings are fixed for "
            "L* <= 12, 12 < L* <= 47 and L* > 47 months (or undefined)."),
        "disclosed": DISCLOSED,
        "power": (
            "Not a test; the chances concern which pre-declared wording applies. G-def "
            "(L* <= 47) passes with about 0.9 (the judge's estimate): FS's N2-unit endpoints "
            "(about 1e-6 m by 2026-12) make it near-certain for S1's mean, but not by "
            "construction, because G-def needs the 90th percentile over single screens of "
            "both arms, and S2's untapered restart wrote -21/-25 m into confined deep "
            "layers whose slow modes FS's endpoints do not resolve: if more than 10 % of "
            "the screens sit where a 25 m offset decays with a time constant of about 7.5 "
            "months or more (25 m x exp(-46/8) is about 0.08 m), p90_S2 stays >= 0.05 m at "
            "lead 47. Between the two passing cases there is "
            "no favourite: the planners' ranges (8-14 and 12-24 months) straddle the "
            "12-month bound. The window clause is 'beyond' whenever G-def passes unless a "
            "difference re-grows after 2026-12, which the linear solver makes unlikely"),
        "arms": {
            "S0": "fs_start_S0.npz: the page recipe rerun (innovation restart at the "
                  "2022-12 origin, gain 1, 18 parameter sets x 2 initial fields); the "
                  "reference of every difference",
            "S1": "fs_start_S1.npz: S0 with gain 0 (each member starts the projection from "
                  "its own free-running Dec-2022 state)",
            "S2": "fs_start_S2.npz: S0 with the old untapered IDW field restart",
            "no_new_forward": "no forward is run: the monthly per-cell curve is already in "
                              "the FS files (heads_mean, 5 scenarios x 4 layers x 2148 cells "
                              "x 252 months, 2012-01..2032-12); the fallback rerun of plan "
                              "P3-A is dropped"},
        "recipe": {
            "scorer": f"{X}/score_fm.py (sha256 {frozen[FROZEN[1]]}) with {X}/fm_common.py "
                      f"(sha256 {frozen[FROZEN[0]]}); run once: {SCORE_CMD}",
            "key_whitelist": (
                f"every npz is opened through fm_common.WhitelistedNpz with keys "
                f"{list(C.FM_KEYS)} only; loading any other key raises (an explicit check "
                f"that survives python -O); {list(C.NEVER_KEYS)} can never be whitelisted. "
                "The report lists the keys each file was read for. The fs_start CSVs "
                "(summary, members), the sealed forward stdout and the attempt log are not "
                "opened"),
            "binding": "the scorer refuses unless this declaration is registered and intact "
                       "(prereg3.load_prereg re-hashes every declared input) and the sha256 "
                       "of score_fm.py and fm_common.py equal frozen_code; each file it is "
                       "given must carry the declared sha256; report.json records this "
                       "declaration's sha256 and is written once (open mode 'x')",
            "gpu": "none"},
        "frozen_code": frozen,
        "zones": (
            "Frozen well screens (cell, layer, county): N2's and E02's 158 registered stations "
            "(results/twin/prospective_2026-11/predictions.csv station ids; 84 Yunlin, 74 "
            f"Changhua), cells and layers as in {X5_WELLS} (rows sha256 "
            f"{X5_WELLS_ROWS_SHA256}), embedded in fm_common.py without station ids, screens "
            f"sha256 {C.SCREENS_SHA256} (the same tuples as FS's scorer). Each screen is read "
            "in its own layer. The N2 unit is the 84 Yunlin screens"),
        "burn_in": "unchanged (forward origin 2022-12; lead 1 = 2023-01)",
        "folds": NA("model-only comparison of existing forward arms of one frozen "
                    "calibration; no folds"),
        "gate_scores": NA("no science gate; one definitional gate (G-def) and one "
                          "engineering gate (G-eng)"),
        "primary_model": "the page model (theta stage3_datum_gate 036a7a17 with its fold and "
                         "posterior sets, page run datum_gate_innov), as rerun by FS (S0, "
                         "S1, S2 at code 4c2723e)",
        "inputs": (
            [{"path": p, "sha256": frozen[p]} for p in FROZEN]
            + [{"path": p, "sha256": h} for p, h in INPUT_SHA256.items()]
            + [f"{X}/prereg_fm.py", TESTS, X5_WELLS]),
        "tests": NA("model-only declaration: the estimand is a property of one model's "
                    "forward runs; no observation is scored, no fresh unit exists and none "
                    "is consumed"),
        "fresh_units": NA("no observation is read or scored"),
        "thresholds": {
            "estimand": (
                "baseline scenario; for k in {S1, S2} and lead L = 1..48 (2023-01..2026-12): "
                "d_k(i, L) = h_k(i, L) - h_S0(i, L), heads_mean (the arm's ensemble mean) at "
                "screen i's cell in its own layer; p90_k(L) = numpy.percentile(|d_k(., L)|, "
                "90) over the 158 screens (linear). S0's heads_mean averages its two initial "
                "fields per parameter set (the innovation restart and the same with sigma "
                "0.5 m seed-0 noise on the observations); S1's two fields are identical (gain "
                "0); S2's two fields are its own restart with and without that noise. So "
                "d_S1 is the page ensemble mean's start (the observation update plus half the "
                "seed-0 perturbation's effect) against the free run, and d_S2 the page's "
                "start against the old restart's, each as the page shows them"),
            "lstar": (
                f"L* = the smallest lead L such that p90_S1(L') < {C.FLOOR_M} m and "
                f"p90_S2(L') < {C.FLOOR_M} m (strict) for every L' in L..48 (2026-12); "
                "undefined if either is >= 0.05 m at lead 48"),
            "g_def": (
                "L* is defined and L* <= 47, i.e. a 'stays below' month exists before 2026-12; "
                "otherwise the wording is 'start memory longer than the record; no horizon "
                "stated'"),
            "g_eng": (
                "before any curve: S0's baseline heads at the 158 screens equal the page run "
                f"datum_gate_innov.npz's in every month 2012-01..2032-12 (max |diff| <= "
                f"{S.REPRO_TOL_M} m, as FS's reproduction recorded); otherwise exit 3, "
                "nothing computed"),
            "outcome_cases": (
                f"'within_first_year' if L* <= {S.CASE_BOUNDS[0]}; 'beyond_first_year' if "
                f"{S.CASE_BOUNDS[0]} < L* <= {S.CASE_BOUNDS[1]}; 'longer_than_record' if L* "
                f"> {S.CASE_BOUNDS[1]} or undefined"),
            "wordings": S.WORDINGS,
            "e02_e21_window": (
                f"the E02/E21 interpretation clause covers the registered 2026-11..2027-10 "
                f"window, leads {S.WINDOW_E02_FIRST_LEAD}..{S.WINDOW_E02_LAST_LEAD}, past the "
                f"decision window: 'window_beyond_memory' if L* is defined and p90_S1 and "
                f"p90_S2 are < {C.FLOOR_M} m at every lead from L* through "
                f"{S.WINDOW_E02_LAST_LEAD} (2027-10; the same files' continuation); "
                "'window_partly_within_memory' if L* is defined but either reaches the floor "
                "again by 2027-10; 'window_within_memory' if L* is undefined or > 47. The "
                "clause case never changes L*, G-def or the E23/C30 wordings"),
            "e02_e21_clauses": S.E02_E21_CLAUSES,
            "efold": (
                "e-folding time of a_k(L) = |mean over the N2 unit's 84 screens of d_k(i, L)|, "
                "k in {S1, S2}, in months after lead 1: tau_1e where a_k first falls to "
                "a_k(1)/e (log-linear interpolation), and tau_fit = -1/slope of the "
                f"least-squares line of ln a_k on L over the leads with a_k >= "
                f"{C.EFOLD_FIT_MIN_M} m; reported, never gating"),
            "bounds": "strict below the floor, unrounded",
            "reported_not_gating": [
                "p90_S1 and p90_S2 at every lead 1..48, and per arm the 'stays below' lead",
                "the N2-unit curves |mean d_k| (84 Yunlin screens), their e-folds and their "
                "own 'stays below' lead",
                "p90 per county (84 Yunlin, 74 Changhua) and per screened layer (L1..L4) at "
                "leads 1, 3, 6, 12, 24, 36, 48, with their 'stays below' leads",
                "|fan layer-2 mean difference| at the same leads",
                "continuation 2027-01..2032-12: the 'stays below' lead over leads 1..120 and "
                "the largest p90 after L*",
                "the other four scenarios (cut30, retire_aqua, cut30_2030, retire_aqua_2030): "
                "their L* and the largest |d_k(scenario) - d_k(baseline)| over screens and "
                "leads (start memory is expected scenario-independent)"]},
        "evidence_and_ledger_mapping": {
            "scope": (
                "Model-only; no outcome moves any claim's status (C30 stays N+D, E02 and E21 "
                "stay D, E23 stays L, C01 unchanged). Every sentence says 'in this model'. "
                "The heads are the page model's; nothing is said about the aquifer"),
            "within_first_year_or_beyond_first_year": [
                "E23 (L) gains the horizon: report.json primary.wording.E23",
                "C30's heads part is narrowed to primary.wording.C30_heads and stays N+D "
                "(under user ruling 2026-10-29 (1))",
                "E02 and E21 gain primary.wording.E02_E21_clause (its case in "
                "primary.wording.E02_E21_clause_case: beyond the window or partly within it) "
                "as an interpretation note ONLY if the user gives the page-line ruling for "
                "it; without the ruling it goes to the ledger notes and README only"],
            "longer_than_record": [
                "E23 records 'start memory longer than the record; no horizon stated'; C30 "
                "is not narrowed; the E02/E21 note (under the same ruling) reads "
                "primary.wording.E02_E21_clause (case 'window_within_memory')"],
            "on_engineering_failure": [
                "exit 1-3 with nothing computed; the ledger records 'not evaluable "
                "(engineering)'; a fix of the scorer needs a v2 that discloses it"]},
        "attempt_policy": {
            "scored_runs": "one scoring run; the first run that passes steps 0-2 computes the "
                           "curves, writes report.json (open mode 'x') and is final",
            "engineering_retries": (
                "free before report.json exists: a missing or drifted file or a scorer crash "
                "in steps 0-2 (which print only the G-eng maximum difference). A fix that "
                "changes the scorer or fm_common.py cannot run under this declaration (the "
                "sha256 binding refuses it): it needs a v2 that discloses the attempt"),
            "pre_declared_alternatives": NA("none: the estimand is deterministic on files "
                                            "that already exist"),
            "exploratory_followups_max": 1,
            "followup_label": "post-result, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included"},
        "interpretation": (
            "L* says for how long, in this model, the observation update at the December "
            "2022 start still moves the simulated heads at the monitored screens by 0.05 m "
            "or more at the 90th percentile. After L*, the model's head levels are set by its "
            "parameters, datum and the average-weather (climatology) forcing of the "
            "projection, not by the observations at the start: a level 'forecast' past L* is "
            "the model's average-weather scenario. The start compared is the page's own: "
            "S0's ensemble mean includes half the effect of its seed-0 perturbed initial "
            "field, so |S1 - S0| is the page start's memory, not the observation update "
            "alone. This bears on what E02c's level rule can test in the 2026-11..2027-10 "
            "window (the average-weather level and datum, if the start memory has ended "
            "before 2026-11 and stays ended through 2027-10). It says nothing about whether "
            "those levels are right (E02/E02c decide that from 2028-01-15) and is not "
            "evidence about the aquifer."),
        "rule_risks_disclosure": [
            "the drafter listed the fs_start npz member names and sizes (zip metadata, no "
            "array loaded) and read FS's report blocks 'files', 'reproduction' (the "
            "max |S0 - page| values only), 'verdict' and 'registration' to bind the files; "
            "no curve or level value was computed or printed",
            "the drafter printed a few selected metadata keys of the page theta "
            "(sw_recharge, extra_abstraction, eta_classes, delay_initial_state and similar "
            "switches) to check the forward's record handling for FH; that printout also "
            "showed the first entries of the theta's fitted well_datum dictionary (model "
            "parameters fitted in 2026-09 on pre-origin heads). Nothing was computed from "
            "them and they are not used here",
            "the plan's planners and reviewers read only npz headers (shapes) of these "
            "files"],
        "no_prior_look_statement": (
            "No per-lead curve of |S1 - S0| or |S2 - S0| at the screens, no per-screen p90 "
            "and no e-fold of these runs has been computed; only FS's three endpoint shifts "
            "(disclosed) are known."),
        "written_by": f"{X}/prereg_fm.py",
    }


def scorer_problems() -> list[str]:
    out = []
    if (S.PREREG_ID, S.FROZEN) != (PREREG_ID, FROZEN):
        out.append("scorer id or frozen list differs from the declaration")
    if C._screens_sha() != C.SCREENS_SHA256:
        out.append("screen list differs from its sha")
    if set(C.FM_KEYS) & set(C.NEVER_KEYS):
        out.append("whitelist holds a never-key")
    w = REPO / X5_WELLS
    if w.exists():
        rows = json.loads(w.read_text())
        if rows.get("rows_sha256") != X5_WELLS_ROWS_SHA256:
            out.append(f"{X5_WELLS} rows sha changed")
        got = [(r["cell"], r["layer"], r["county"][0].upper()) for r in rows["rows"]]
        if got != list(zip(C.CELLS, C.LAYERS, "".join(C.COUNTY), strict=True)):
            out.append(f"screens differ from {X5_WELLS}")
    return out


def readiness() -> list[str]:
    """What stands between validation and registration (empty = ready)."""
    out = []
    for rec in doc()["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (REPO / p).exists():
            out.append(f"input {p} is missing")
        elif isinstance(rec, dict) and sha256(REPO / p) != rec["sha256"]:
            out.append(f"input {p} changed since drafting")
    if (REPO / f"results/dataprep/preregistration/{PREREG_ID}.json").exists():
        out.append(f"{PREREG_ID} is already registered")
    if (REPO / REPORT).exists():
        out.append(f"{REPORT} exists")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="dry run: print, check, write nothing")
    g.add_argument("--write", action="store_true", help="register once (the lead only)")
    ap.add_argument("--expect-self-sha", default=None,
                    help="sha256 of this script as reviewed; required with --write")
    ap.add_argument("--quiet", action="store_true", help="do not print the document")
    a = ap.parse_args(argv)
    d = doc()
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", canonical_sha(d))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    probs = scorer_problems()
    print("scorer/binding checks:", probs or "ok")
    block = prereg3.blocking_looks([], None, test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none (declaration with no tests and no "
          "fresh units)")
    pending = readiness()
    print("readiness:", pending or "ready to register")
    self_sha = sha256(Path(__file__).resolve())
    print("this script sha256:", self_sha)
    print("frozen_code:", json.dumps(d["frozen_code"], indent=1))
    bad = bool(errs or probs or block or pending)
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
    print(f"registered {path.relative_to(REPO)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
