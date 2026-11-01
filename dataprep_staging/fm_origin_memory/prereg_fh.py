"""Declaration FH.forcing_sensitivity.v1, kind 'declaration', before either new forward runs:
in the page model, from how many months after a start does the head level depend more on the
weather and pumping that follow than on what is known at the start?

Model-only, heads only. Two new forwards of the page recipe from a December 2019 origin
(--record-end 2019-12; gain 0 and gain 1) and the existing free-run hindcast fs_start_S1.npz
(FS.start_invariance.v1) share one free-running December 2019 state. No observation of any
family is read or scored by the scorer; the forwards read the 2012-19 inputs as every page
forward does and their stdout is sealed. It moves no claim's status; it adds context to the
C30, E01 and E07 notes (user ruling 2026-10-29: the ledger governs page wording).

``--validate`` (dry run) prints the document, its canonical sha256 and this script's sha256,
runs prereg3.validate_prereg, checks that the runner and the scorer carry the declared recipe,
runs prereg3.blocking_looks as a dry run and reports readiness; nothing is written.
``--write`` registers it once (prereg3.write_prereg; the lead only). From the repository root:

    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fm_origin_memory/prereg_fh.py \\
        --validate
    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fm_origin_memory/prereg_fh.py \\
        --write --expect-self-sha <sha256 of this file as reviewed>
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
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(HERE))

import fm_common as C  # noqa: E402
import score_fh as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable

PREREG_ID = S.PREREG_ID
X = C.X
FROZEN = S.FROZEN
RUNNER = f"{X}/run_fh.sh"
TESTS = f"{X}/tests/test_fh_forcing.py"
REPORT = S.DEFAULT_OUT
LOGDIR = "results/twin_runs/fh_force"
NOGW = f"{LOGDIR}/no_gw_data"            # HYDROMIND_GW_DATA for the forwards: must not exist
CODE = "4c2723e"
FM_ID = "FM.origin_memory.v1"
G = "results/twin_runs/stage3_datum_gate"
PAGE_THETAS = (f"--theta {G}/stage3_theta.json --theta {G}/stage3_fold_thetas.json "
               f"--theta {G}/stage3_posterior_tn.json")
VEP = (f"{G}/coupled_leveling/vep_zonal_leveling.json,"
       f"{G}/coupled_leveling_tau30/vep_zonal_leveling.json")
RF = "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"
RF_SHA256 = "ef5a525f5ea51697af81f2a9cd4a0fce8cad074f7cfe10121920b8e4f6a63714"
COMMON_SH = ("--horizon 120 --apex-hold calibrated --column-heads free "
             "--column-hpc0-fast-days 365 --rf-timeseries $RF --device cuda --compile-matvec")
COMMON = COMMON_SH.replace("$RF", RF)
RECORD_END = "--record-end 2019-12"
ARM_FLAGS = {
    "C19": "--gain 0.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation "
           "--restart-taper-km 5 --save-members none",
    "N19": "--gain 1.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation "
           "--restart-taper-km 5 --save-members none",
}
# the forward's inputs at the page theta, as FS.start_invariance.v1 declared them (sha256 at
# FS's drafting; tests check they equal dataprep_staging/fs_start/prereg_fs.py's)
FORWARD_INPUTS = {
    "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json":
        "b1dd3bd324d24c495b214003ff5ef688476c6737b961a230899f7abbd78efbc8",
    "AMP_V2/data/fan_stations.parquet":
        "ba1506b93491442862df7bb580e3fa9ae0165c8a58e3bcf09193ec2e58f4e0d6",
    "AMP_V2/data/tpc_pumps.parquet":
        "3725ae3bfa88bbb8011a14a7eec62ae1b75721c88e3159fd316b554d5299fa3c",
    "AMP_V2/data/pump_kwh_all.parquet":
        "72fe813b6a508a299823ab48e0a86019795c84f29d48b508a08e743e6a035215",
    RF: RF_SHA256,
    "chou-shui-data/data/rf_stations.csv":
        "d45ecd82c0c743c022a54e6fd7382c84857e8153d9b8ac1e430be43193e1d386",
    "chou-shui-data/data/gw_stations.csv":
        "ea875d727a9ef3abb7d0edac67650a7a1bdfc0a9e4e97c23946fc8049ee352ae",
    "results/et/openmeteo_et0_2012_2022.npz":
        "54c031db747ad07a5d5d789b604a0c4295e1540114b3eea9ba0c59ac4c62568b",
    "results/twin/basemap.npz":
        "99bbe12f4b0a74b69a89737ee372a4720759a0635b5173b0ca4e199b12362a63",
}
WELLS_DIR = "AMP_V2/data/wells"
INPUT_DIGESTS = [{"path": WELLS_DIR, "glob": "*.parquet", "kind": S.DIGEST_KINDS[0],
                  "n_files": 344,
                  "sha256": "614d5c74c1bb622d225fb9d03fcfa449d10fd863e0aea2f0575bb941384c28e2"}]
THETA_INPUTS = {
    f"{G}/stage3_theta.json": "036a7a17dcbd383fba9eaab6a12e716ab0cdffcdae0385adaf5349a57c64de77",
    f"{G}/stage3_fold_thetas.json":
        "c70632858febe2bbe57f5fe4f17646e66c64a56812699f952cf505fcdf498a89",
    f"{G}/stage3_posterior_tn.json":
        "5b194e1e7bde28b72c18d977f0321e20e12aca411b14ae2ef408d749018ec601",
    f"{G}/coupled_leveling/vep_zonal_leveling.json":
        "d290c59209f758389b7133c3f5b9aadcc71b42ad55f9089d56645fafc547b162",
    f"{G}/coupled_leveling_tau30/vep_zonal_leveling.json":
        "7f799b590fb8e82043204424aa9af001206bb6a542415197606baf08d8518d81",
}
FS_DECL = ("results/dataprep/preregistration/FS.start_invariance.v1.json",
           "9f8a71155cc9be83c0e23a9d61212ff95203d6d52934dab60d7f77b6a035d7b4")
FS_REPORT = ("results/dataprep/reports/FS.start_invariance.v1/report.json",
             "6130a3bf88a3889d09a982067418db69bdd144e1280c776d4c133f8179584ecd")
INPUT_SHA256 = {**{p: h for p, h in S.FS_FILES.values()}, FS_DECL[0]: FS_DECL[1],
                FS_REPORT[0]: FS_REPORT[1], **THETA_INPUTS, **FORWARD_INPUTS}

RUNNER_LINES = (
    f"ID={PREREG_ID}", "DECL=results/dataprep/preregistration/$ID.json", f"CODE={CODE}",
    "R=results/twin_runs", "G=$R/stage3_datum_gate", "LOG=$R/fh_force",
    "NOGW=$LOG/no_gw_data",
    "REPORT=results/dataprep/reports/$ID/report.json",
    'PAGE_THETAS="--theta $G/stage3_theta.json --theta $G/stage3_fold_thetas.json '
    '--theta $G/stage3_posterior_tn.json"',
    'VEP="$G/coupled_leveling/vep_zonal_leveling.json,'
    '$G/coupled_leveling_tau30/vep_zonal_leveling.json"',
    f"RF={RF}", f'COMMON="{COMMON_SH}"', f'RECORD_END="{RECORD_END}"',
    *(f'F_{a}="{f}"' for a, f in ARM_FLAGS.items()))
RUNNER_FRAGMENTS = (
    '[ ! -e "$NOGW" ] || die "$NOGW exists; HYDROMIND_GW_DATA must name an absent path"\n'
    '    env HYDROMIND_GW_DATA="$NOGW" PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.forward',
    '"$HPY" -I "$SCORER" --verify-registration',
    '"$HPY" -I "$SCORER" --attempts "$ATT" --out "$REPORT"',
    '$(arm_flags "$a") $COMMON --vep-json "$VEP" --out "$out" > "$sealed" 2> "$err"',
    'WT="$HOME/HydroPhysicsAI-fhrun-$CODE"',
    'C19) echo "$PAGE_THETAS $F_C19 $RECORD_END" ;;',
    'N19) echo "$PAGE_THETAS $F_N19 $RECORD_END" ;;')


def command(arm: str, wt: str = f"$HOME/HydroPhysicsAI-fhrun-{CODE}") -> str:
    """The forward command line of ``arm`` (as the runner's DRY=1 prints it)."""
    return (f"env HYDROMIND_GW_DATA={NOGW} PYTHONPATH={wt} "
            f"$HOME/miniforge3/envs/hydro/bin/python -P -m hydrophysics.twin.forward "
            f"{PAGE_THETAS} {ARM_FLAGS[arm]} {RECORD_END} "
            f"{COMMON} --vep-json {VEP} --out results/twin_forward/fh_force_{arm}")


def sha256(p: Path) -> str:
    return C.sha256(p) if p.exists() else "missing"


def frozen_code() -> dict:
    return {p: sha256(REPO / p) for p in FROZEN}


def canonical_sha(d: dict) -> str:
    return hashlib.sha256((json.dumps(d, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


DISCLOSED = {
    "fs_s1_hindcast": (
        "fs_start_S1.npz's hindcast months (2012-01..2022-12) are the page model's free run "
        "under the recorded inputs; FS computed only policy levers and three baseline level "
        "shifts (2023-01, 2026-12, 2032-12) from it. No 2020-22 hindcast head of it at the "
        "screens has been computed or printed"),
    "fm_shares_code": (
        f"the scorer shares fm_common.py with {FM_ID} (round-7 rank 2), which reads the "
        "same FS files for the 2022-12 origin memory; FM's result, if scored first, is "
        "known when FH's forwards run, and FH's 2022-12 M is reported only, never compared "
        "with F"),
    "record_end_mechanics": (
        "forward --record-end 2019-12 cuts the record (inputs.truncate_record: dates, "
        "observed heads, pumping energy by class, recharge) after 2019-12, so the origin "
        "moves to 2019-12 and the projection's month-of-year climatology is that of "
        "2012-01..2019-12. The page theta has no canal (sw_recharge) or extra-abstraction "
        "field, so the cut applies. The hindcast is causal and starts from the same month-0 "
        "field, so its 2012-19 months should equal fs_start_S1's (gate E1). With gain 0 the "
        "innovation update returns the state unchanged, so C19's two initial fields per "
        "parameter set are identical (as in S1); N19, like S0, has the page's second, "
        "perturbed initial field (ic sigma 0.5 m, seed 0)"),
    "gw_data_env": (
        "forward.main (at 4c2723e and now) sets its data dir from --data, else "
        "$HYDROMIND_GW_DATA, else the bundled synthetic hydrophysics/sample_data, and when that "
        "is an existing directory it validates the hindcast against leveling after the run "
        "and before the npz is written. A first draft of this declaration ran the forwards "
        "with the variable unset; review found (by resolving the data dir of the 4c2723e "
        "package on CPU) that the fallback sample_data exists but has no leveling panel, so "
        "every forward would have crashed after its GPU run. No forward was run. The "
        f"forwards now run without --data and with HYDROMIND_GW_DATA={NOGW}, a path the "
        "runner checks is absent before every forward: the data dir resolves to None, no "
        "leveling file is read and no leveling R2 is printed. The variable's only other "
        "reader on this path is the river layer, which the page theta does not use (rivers "
        "'none'), and the column JSONs exist, so no column is fitted; heads do not depend "
        "on it"),
    "forcing_source": (
        "no new product is used: the 2020-22 recorded forcing is the forward's own "
        "calibrated inputs (pump census and kWh, the vendor rain cache the page theta was "
        "fitted on, ET0), exactly as fs_start_S1 read them; the climatology is computed from "
        "the same inputs' 2012-19 months"),
}


def doc() -> dict:
    frozen = frozen_code()
    return {
        "id": PREREG_ID, "wp": "FH", "kind": "declaration",
        "title": "Forcing sensitivity of the page model's head level under the 2020-22 "
                 "history: the first lead after a December 2019 start at which the 90th "
                 "percentile over the 158 frozen screens of |recorded-forcing run - "
                 "climatology run| exceeds that of |innovation-restart run - free run| "
                 "(model-only, moves no claim status)",
        "priors": (
            "Model-only. M (origin memory) decays with the aquifer's diffusion times; FM "
            "(rank 2) measures it at the 2022-12 origin, and FS's endpoints suggest an "
            "e-fold of a few months. F (the head departure under the recorded 2020-22 "
            "forcing from the climatology run) grows as weather and pumping anomalies "
            "accumulate; 2020-22 includes the 2021 drought, so F is large for an anomalous "
            "period and L*_force is the forcing horizon under that history, not a "
            "climatological average. Expected: L*_force within the first year, but no "
            "prior is used; the wordings are fixed for L*_force <= 12, 12 < L*_force <= 36, "
            "not reached in 36 months, and not evaluable."),
        "disclosed": DISCLOSED,
        "power": (
            "Not a test; the chances concern which pre-declared wording applies. Evaluable "
            "(p90_M(1) >= 0.05 m) with about 0.85 (the judge's estimate): at the 2022-12 "
            "origin FS's start shift was about +0.77 m at lead 1 on the N2 unit's mean, far "
            "above the floor, but it is not certain by construction at 2019-12, because "
            "the update's size is the free run's misfit to the December 2019 observations, "
            "which no one has computed, and the p90 over single screens needs the update "
            "to reach more than 10 % of the screens. Given evaluable, L*_force within 36 "
            "months is likely (F grows with the 2020-22 anomalies including the 2021 "
            "drought while M decays with an e-fold of a few months), and within the first "
            "year more likely than not; 'not_reached' stays possible if the 2020-22 "
            "forcing sat near its 2012-19 month-of-year averages at most screens"),
        "arms": {
            "C19": "NEW: the page recipe (FS's S1 flags: page theta and its 17 fold and "
                   "posterior sets, gain 0, innovation, taper 5 km, 1 extra initial field) "
                   "with --record-end 2019-12, baseline only: projection 2020-01..2029-12 "
                   "under the 2012-19 climatology from the free-run Dec-2019 state",
            "N19": "NEW: the same with gain 1 (FS's S0 flags): the page's innovation restart "
                   "at 2019-12 from the observed Dec-2019 heads (pre-origin, as initial "
                   "conditions), then the same climatology",
            "R1": "EXISTING fs_start_S1.npz, baseline scenario, hindcast months 2020-01.."
                  "2022-12: the same free run under the recorded 2020-22 inputs (gain 0 at "
                  "2022-12 does not touch them)",
            "no_2015_arm": "the optional 2015-12 origin with recorded 2016-18 inputs is not "
                           "run under v1 (one forcing history, declared as such)"},
        "recipe": {
            "code": (f"the forwards run from a clean detached worktree $HOME/HydroPhysicsAI-"
                     f"fhrun-{CODE} via PYTHONPATH and python -P, launched from the "
                     f"repository root; {CODE} is the code fs_start_S1 ran with (FS's "
                     "attempt log), so E1 compares like with like. Both arms at that code; "
                     "the scorer refuses otherwise"),
            **{f"forward_{a}": command(a) for a in ARM_FLAGS},
            "runner": f"bash {RUNNER} forward C19, forward N19 (tmux; both may run at once "
                      f"under MPS), score (once); sha256 {frozen[RUNNER]}",
            "scorer": f"{X}/score_fh.py (sha256 {frozen[FROZEN[1]]}) with {X}/fm_common.py "
                      f"(sha256 {frozen[FROZEN[0]]})",
            "key_whitelist": (
                f"every npz is opened through fm_common.WhitelistedNpz: fs_start_S0/S1 with "
                f"{list(C.FM_KEYS)}; the new runs with those plus the recipe metadata "
                f"{list(C.FH_META_KEYS)}; {list(C.NEVER_KEYS)} can never be loaded; the "
                "report lists the keys each file was read for"),
            "binding": "the runner and the scorer refuse unless this declaration is "
                       "registered and intact and the sha256 of fm_common.py, score_fh.py "
                       "and run_fh.sh equal frozen_code; every declared input (thetas, "
                       "columns, forward inputs, the FS files) and the head-well directory "
                       "digest are re-checked before every forward and at scoring; the "
                       "scorer refuses npz files that are not the outputs of each arm's last "
                       "successful forward attempt, or arms run on different code",
            "gpu": "C19 and N19: 18 parameter sets x 2 initial fields x 1 scenario x 2 "
                   "rollouts (column-heads free), 96 hindcast and 120 projection months; "
                   "about 15-25 min each on the T4 (FS's 5-scenario arms took about an "
                   "hour); about 40 min in all, or about 25 min wall time with both under "
                   "MPS. The runner refuses below 5000 MiB free; no twin-module edit while "
                   "they run (code freeze)"},
        "frozen_code": frozen,
        "zones": (
            "the 158 frozen screens of N2/E02 (cell, layer, county; fm_common.py, screens "
            f"sha256 {C.SCREENS_SHA256}, the same tuples as FS's scorer), each read in its "
            "own layer; the N2 unit is the 84 Yunlin screens"),
        "burn_in": "unchanged (hindcast from 2012-01; origin 2019-12; lead 1 = 2020-01)",
        "folds": NA("model-only comparison of forward runs of one frozen calibration"),
        "gate_scores": NA("no science gate; engineering gates E1 and E2 and one "
                          "definitional gate"),
        "primary_model": "the page model (theta stage3_datum_gate 036a7a17 with its fold and "
                         "posterior sets), page recipe of FS's S0/S1 at code 4c2723e",
        "inputs": (
            [{"path": p, "sha256": frozen[p]} for p in FROZEN]
            + [{"path": p, "sha256": h} for p, h in INPUT_SHA256.items()]
            + [f"{X}/prereg_fh.py", TESTS]),
        "input_digests": INPUT_DIGESTS,
        "tests": NA("model-only declaration: the estimand is a property of one model's "
                    "forward runs; no observation is scored and no fresh unit is consumed"),
        "fresh_units": NA("no observation is read or scored"),
        "thresholds": {
            "definitions": (
                "baseline scenario, heads_mean (the ensemble mean over 36 members) at screen "
                "i's cell in its own layer; lead L after 2019-12. F(i, L) = R1(i, L) - "
                "C19(i, L) for L = 1..36 (2020-01..2022-12); M(i, L) = N19(i, L) - C19(i, L) "
                "for L = 1..120; p90_F(L), p90_M(L) = numpy.percentile(|.|, 90) over the 158 "
                "screens (linear)"),
            "lstar_force": "L*_force = the smallest L in 1..36 with p90_F(L) > p90_M(L) "
                           "(strict); 'not reached' if none",
            "evaluable_if": (f"p90_M(1) >= {C.FLOOR_M} m (the 2019-12 start carries "
                             "information); otherwise 'not_evaluable'"),
            "outcome_cases": (
                f"'within_first_year' if L*_force <= {S.CASE_BOUND}; 'within_record' if "
                f"{S.CASE_BOUND} < L*_force <= 36; 'not_reached'; 'not_evaluable'"),
            "wordings": S.WORDINGS,
            "engineering_checks_before_F_or_M_is_computed": [
                "step 0: this declaration registered and intact (inputs re-hashed, the "
                "well-directory digest re-computed); frozen code matches; each new npz is "
                "its arm's last successful forward output in attempts.jsonl; both arms at "
                "one code",
                "step 1: every file: scenario names, dates (new runs 2012-01..2029-12, "
                "origin 2019-12; FS files 2012-01..2032-12, origin 2022-12), 2148 active "
                "cells, heads_mean shape, finite heads; new runs: horizon 120, gain 0 (C19) "
                "or 1 (N19), hindcast_gain 0, 36 members of the page's 18 sets, scenarios "
                "'baseline' only, forward_options record_end 2019-12, innovation, taper 5 "
                "km, column-heads free, apex hold calibrated, hpc0 fast days 365, no "
                "climatology_end or extend_record_to",
                f"step 2, E1: C19's and N19's heads 2012-01..2019-12 equal fs_start_S1's in "
                f"every layer and cell (max |diff| < {S.E1_TOL_M} m, 50 times below the "
                "floor; a numeric band for float32 storage, where one ulp of a head of tens "
                "of metres is 2-8e-6 m, and any unordered CUDA index_add accumulation; the "
                "report records per arm whether the match is bit-identical, as FS's 0.0 m "
                "reproduction was); otherwise exit 3 and nothing more runs under v1",
                "E2: the scorer loads no observation array (key whitelist above); the "
                "forwards' stdout is sealed and they run without --data and with "
                f"HYDROMIND_GW_DATA={NOGW}, a path the runner checks is absent (the data dir "
                "resolves to None: no leveling file is read, no leveling R2 printed)"],
            "bounds": "strict, unrounded",
            "reported_not_gating": [
                "p90_F and p90_M at every lead; the 'stays above' lead (F > M at every later "
                "lead through 36)",
                "the same comparison for the N2 unit (|mean| over the 84 Yunlin screens), "
                "Changhua's 74 screens (|mean|), the fan layer-2 mean and the per-county "
                "p90, each with its first-exceeds and stays-above leads",
                "M's e-fold at 2019-12 (N2 unit, as FM defines it) and its 'stays below "
                "0.05 m' lead over 1..120",
                "the 2022-12 origin memory from the FS forwards (|S1 - S0|: p90 and N2 unit "
                "at leads 1, 3, 6, 12, 24, 36), reported separately and never compared with "
                "F (another origin)"]},
        "evidence_and_ledger_mapping": {
            "scope": (
                "Model-only; no claim's status moves. Every sentence says 'in this model' "
                "and 'under the 2020-22 forcing history'"),
            "within_first_year_or_within_record": [
                "C30, E01 and E07 notes gain primary.wording (context only): in this model, "
                "from about L*_force months the level depends more on future weather and "
                "pumping than on anything known at the start"],
            "not_reached": ["the ledger notes record primary.wording; no page text"],
            "not_evaluable": ["a ledger note with primary.wording; nothing else changes"],
            "on_engineering_failure": [
                "E1 failure (exit 3): ledger note 'not evaluable (the runs do not share the "
                "2019-12 state)'; nothing else; a v2 must disclose the attempt"]},
        "attempt_policy": {
            "scored_runs": "one forward per arm and one scoring run; the first scorer run "
                           "that passes steps 0-2 computes F, M and L*_force, writes "
                           "report.json (open mode 'x') and is final",
            "values_written_before_scoring": (
                "hydrophysics.twin.forward prints per-member in-sample hindcast R2 "
                "(2012-19), the '=== forward summary ===' block and the theta's gate status "
                "to stdout and writes results/twin_forward/fh_force_<arm>.npz, .summary.csv "
                f"and .members.csv. The runner sends stdout to {LOGDIR}/sealed/"
                "<arm>_forward.stdout.SEALED (mode 400) and logs its sha256 and the outputs' "
                f"in {LOGDIR}/attempts.jsonl. Nobody opens, tails, greps or loads the sealed "
                "logs, the CSVs or the new npz before report.json exists; the runner prints "
                "only exit codes, sha256s and free GPU memory; stderr may be read to "
                "diagnose a crash"),
            "engineering_retries": (
                f"free before report.json exists: a forward crash, OOM or missing file, or "
                f"a scorer failure in step 0 or 1; each attempt is logged; `bash {RUNNER} "
                f"archive '<reason>' [arm ...]` moves the outputs unopened to {LOGDIR}/"
                "archive/<UTC stamp>/ before a rerun. An E1 failure (step 2, a difference of "
                f"{S.E1_TOL_M} m or more, beyond any numeric noise) means the runs do not "
                "share the 2019-12 state; it is not retried under v1 (the runner refuses to "
                "archive after it). A fix that changes the scorer, fm_common.py, the runner "
                "or the recipe needs a v2 that discloses the attempts"),
            "pre_declared_alternatives": NA("none"),
            "exploratory_followups_max": 1,
            "followup_label": "post-result, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included (attempts.jsonl and the report)"},
        "interpretation": (
            "L*_force says, in this model and under the 2020-22 forcing history, from which "
            "lead the head departures caused by the recorded weather and pumping (against "
            "their 2012-19 month-of-year averages) are larger, at the 90th percentile of the "
            "monitored screens, than what the observation update at the start adds. Past it, "
            "the model's level depends more on the future forcing than on the start, so a "
            "forecast's skill rests on the forcing scenario. It is one history including a "
            "drought year, not an average, and it is not evidence about the aquifer; it "
            "explains, as model context, why a forecast from a 2022-12 start can lose to "
            "climatology in later years (E01, E07), without testing it."),
        "rule_risks_disclosure": [
            "the drafter read FS's report blocks 'files', 'reproduction' (max |S0 - page| "
            "only), 'verdict' and 'registration' and listed the fs_start npz member names "
            "(zip metadata); no head value of any run was computed or printed",
            "the drafter printed selected metadata switches of the page theta (sw_recharge, "
            "extra_abstraction, eta_classes, delay_initial_state and similar) to check that "
            "--record-end applies; the printout also showed the first entries of the theta's "
            "fitted well_datum dictionary (model parameters fitted on pre-origin heads); "
            "nothing was computed from them",
            "the forwards read the observed Dec-2019 heads (N19's restart; pre-origin, "
            "clean_heads allows initial_conditions) and the 2012-19 forcing, as every page "
            "forward reads its origin heads; per-member in-sample hindcast R2 over 2012-19 "
            "is printed to the sealed stdout only",
            "no 2020-22 head observation, no leveling and no post-origin value is read"],
        "no_prior_look_statement": (
            "No forward from a 2019-12 origin of the page recipe exists (no fh_force output); "
            "no difference of fs_start_S1's 2020-22 hindcast against any climatology run "
            "has been computed."),
        "written_by": f"{X}/prereg_fh.py",
    }


def runner_problems() -> list[str]:
    sh = (REPO / RUNNER).read_text()
    lines = set(sh.splitlines())
    out = [f"{RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    out += [f"{RUNNER} does not carry {s!r}" for s in RUNNER_FRAGMENTS if s not in sh]
    if "/home/" in sh:
        out.append(f"{RUNNER} hardcodes a /home path")
    return out


def scorer_problems() -> list[str]:
    out = []
    if (S.PREREG_ID, S.FROZEN) != (PREREG_ID, FROZEN):
        out.append("scorer id or frozen list differs from the declaration")
    if C._screens_sha() != C.SCREENS_SHA256:
        out.append("screen list differs from its sha")
    if set(C.FM_KEYS + C.FH_META_KEYS) & set(C.NEVER_KEYS):
        out.append("whitelist holds a never-key")
    record_end = RECORD_END.split()[-1]
    if record_end != S.NEW_ORIGIN:
        out.append("scorer origin differs from the runner's --record-end")
    horizon = int(COMMON.split("--horizon ")[1].split()[0])
    if horizon != S.N_M:
        out.append("scorer horizon differs from the runner's")
    if not 0.0 < S.E1_TOL_M <= C.FLOOR_M / 50:
        out.append("E1 band is not at least 50 times below the floor")
    return out


def readiness() -> list[str]:
    """What stands between validation and registration (empty = ready)."""
    out = []
    d = doc()
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (REPO / p).exists():
            out.append(f"input {p} is missing")
        elif isinstance(rec, dict) and sha256(REPO / p) != rec["sha256"]:
            out.append(f"input {p} changed since drafting")
    try:
        S.check_input_digests(d, REPO)
    except C.EngineeringError as err:
        out.append(str(err))
    if (REPO / f"results/dataprep/preregistration/{PREREG_ID}.json").exists():
        out.append(f"{PREREG_ID} is already registered")
    for a in S.ARMS:
        for s in ("npz", "summary.csv", "members.csv"):
            if (REPO / f"results/twin_forward/fh_force_{a}.{s}").exists():
                out.append(f"results/twin_forward/fh_force_{a}.{s} exists before the "
                           "declaration")
    if (REPO / REPORT).exists():
        out.append(f"{REPORT} exists")
    wt = Path.home() / f"HydroPhysicsAI-fhrun-{CODE}"
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return out + [f"worktree $HOME/HydroPhysicsAI-fhrun-{CODE} missing (git worktree "
                      f"add --detach $HOME/HydroPhysicsAI-fhrun-{CODE} {CODE})"]
    if head != CODE:
        out.append(f"worktree $HOME/HydroPhysicsAI-fhrun-{CODE} is at {head}")
    if dirty:
        out.append(f"worktree $HOME/HydroPhysicsAI-fhrun-{CODE} is dirty")
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
    probs = runner_problems() + scorer_problems()
    print("recipe/binding checks:", probs or "ok")
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
