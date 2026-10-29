"""Declaration X5.lever_class_share.v1, kind 'declaration', before either all-class forward
run exists: is the all-class x0.7 head lever of the two calibrations that differ only in the
irrigation energy-to-volume conversion (q0c, x5r1) invariant to that conversion?

Model-only. No observation of any family is read or scored. The declaration feeds only the
evidence record (docs/TWIN_EVIDENCE_2026-10.md) and the claims ledger
(hydrophysics/twin/claims_v1.json, row C13); it adds no page lever and changes no page wording
by itself (user ruling 2026-10-29: the ledger governs page wording; unsupported outputs are
narrowed or withdrawn).

By default it VALIDATES only (prereg3.validate_prereg), checks that the runner carries the
declared recipe, runs prereg3.blocking_looks as a dry run and reports whether it is ready to
register; nothing is written. ``--register`` writes it once (prereg3.write_prereg). Run from
the repository root, under the X5 lock:

    ~/miniforge3/envs/hydro/bin/python dataprep_staging/x5_class_share/prereg_class_share.py
    flock results/dataprep/preregistration/.x5.lock ~/miniforge3/envs/hydro/bin/python \\
        dataprep_staging/x5_class_share/prereg_class_share.py --register \\
        --expect-self-sha <sha256 of this file as reviewed>
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

from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable

PREREG_ID = "X5.lever_class_share.v1"
X = "dataprep_staging/x5_class_share"
SCORER = f"{X}/score_class_share.py"
RUNNER = f"{X}/run_class_share.sh"
WELLS = f"{X}/wells_frozen.json"
FROZEN = (SCORER, RUNNER, WELLS)
TESTS = f"{X}/tests/test_x5_class_share.py"
PY_SH = "results/twin_runs/par28/py.sh"
WT_HEAD = "5f3af18"
REPORT = f"results/dataprep/reports/{PREREG_ID}/report.json"

BAND = (0.9, 1.1)
STRICT_BAND = (0.95, 1.05)
WINDOWS = {"2026-27": "2026-01..2027-12 (forward months 168-191)",
           "2030-32": "2030-01..2032-12 (forward months 216-251)"}
SCEN_CUT30 = "cut30:irrigation=0.7@2026-01"
SCEN_ALL30 = ("all30:irrigation=0.7,aquaculture=0.7,livestock=0.7,domestic=0.7,industry=0.7,"
              "other=0.7@2026-01")
FWD_FLAGS = ("--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation "
             "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
             "--column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members none")
# the runner must carry these lines verbatim (one source of the recipe, checked here)
RUNNER_LINES = (f"CUT30='{SCEN_CUT30}'", f"ALL30='{SCEN_ALL30}'", f'FLAGS="{FWD_FLAGS}"',
                f"WT_HEAD={WT_HEAD}", f"PY={PY_SH}", 'WT="$HOME/HydroPhysicsAI-x5run"',
                "DECL=results/dataprep/preregistration/$ID.json", f"ID={PREREG_ID}",
                "REPORT=results/dataprep/reports/$ID/report.json")
RUNNER_FRAGMENTS = (
    '--scenario "$CUT30" --scenario "$ALL30" $FLAGS',
    '--vep-json "$R/stage3_datum_gate_$a/coupled_leveling/vep_zonal_leveling.json"',
    '--theta "$R/stage3_datum_gate_$a/stage3_theta.json"',
    "--out \"$out\" > \"$sealed\" 2> \"$err\"")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def frozen_code() -> dict:
    return {p: sha256(REPO / p) for p in FROZEN}


def command(arm: str) -> str:
    r = "results/twin_runs"
    return (f"{PY_SH} -m hydrophysics.twin.forward --theta {r}/stage3_datum_gate_{arm}/"
            f"stage3_theta.json --scenario '{SCEN_CUT30}' --scenario '{SCEN_ALL30}' "
            f"{FWD_FLAGS} --vep-json {r}/stage3_datum_gate_{arm}/coupled_leveling/"
            f"vep_zonal_leveling.json --out results/twin_forward/class_share_{arm} "
            f"> results/twin_runs/class_share/sealed/{arm}_forward.stdout.SEALED "
            f"2> results/twin_runs/class_share/{arm}_forward.stderr.log")


# Disclosed before declaration (model outputs and pre-origin forcing only).
DISCLOSED = {
    "cut30_lever_ratio_x5r1_over_q0c": {
        "estimator": "ratio of means: mean over the county's frozen screens and the window's "
                     "months of (cut30 - baseline) head, x5r1 over q0c; batch-27 files "
                     "results/twin_forward/datum_gate_{q0c,x5r1}.npz",
        "changhua": {"2026-27": 1.3893, "2030-32": 1.3859},
        "yunlin": {"2026-27": 1.2730, "2030-32": 1.2717},
        "by_horizon_changhua": {"m1-3": 1.408, "m4-6": 1.375, "m7-12": 1.391,
                                "m13-24": 1.389, "y5-7": 1.386},
        "by_horizon_yunlin": {"m1-3": 1.266, "m4-6": 1.256, "m7-12": 1.284,
                              "m13-24": 1.274, "y5-7": 1.272},
        "q0c_cut30_mean_lever_changhua_m": {"m1-3": 0.677, "m4-6": 1.05, "m7-12": 1.14,
                                            "m13-24": 1.265, "y5-7": 1.318}},
    "hindcast_indistinguishability_2012_22": {
        "what": "free-run model hindcast heads (model output, not observations) at the frozen "
                "screens, x5r1 against q0c; medians over screens",
        "changhua": {"amplitude_ratio_anomaly": 1.002, "corr_anomaly": 0.981,
                     "amplitude_ratio_interannual": 1.083, "corr_interannual": 0.992},
        "yunlin": {"amplitude_ratio_anomaly": 1.012, "corr_anomaly": 0.984,
                   "amplitude_ratio_interannual": 1.015, "corr_interannual": 0.995}},
    "class_share_arithmetic": {
        "what": "predicted cut30 lever ratio = irrigation share of effective volume in x5r1 "
                "over that in q0c (county census energy 2012-22, pumping family, "
                "pre-origin; explorer changhua_inference)",
        "predicted": {"changhua": 1.33, "yunlin": 1.29},
        "found": {"changhua": 1.39, "yunlin": 1.27},
        "fan_check_from_batch27_logs_GWh_2012_22": {
            "q0c": {"irrigation": 2303.4, "other_classes": 1838.2, "irrigation_share": 0.556},
            "x5r1": {"irrigation": 4988.0, "other_classes": 1838.2, "irrigation_share": 0.731},
            "share_ratio": 1.314},
        "implied_all30_ratio_found_over_predicted": {"changhua": 1.044, "yunlin": 0.987},
        "control_world": "cdm (domestic class removed) shows the same class-share effect: "
                         "x1.42 over ccz"},
    "explorer": "session scratchpad explore/changhua_inference (exp5_volume_ident, "
                "exp7_lever_by_horizon, exp8_class_split); judge pick 4"}


def doc() -> dict:
    wells = json.loads((REPO / WELLS).read_text())
    assert (wells["n"], wells["n_changhua"], wells["n_yunlin"]) == (158, 74, 84)
    frozen = frozen_code()
    return {
        "id": PREREG_ID, "wp": "X5", "kind": "declaration",
        "title": "Class share of the irrigation-cut head lever: is an all-class x0.7 pumping "
                 "cut invariant to the irrigation energy-to-volume conversion? (x5r1 over q0c "
                 "lever ratio in [0.9, 1.1] at the 74 Changhua and 84 Yunlin well screens, "
                 "2026-27 and 2030-32 means; model-only, feeds the evidence record and the "
                 "claims ledger only)",
        "priors": (
            "Model-only (no observation is read or scored). q0c = Q0 + the four-county census "
            "at the twin's conversion (3.937 m3/kWh); x5r1 = the same with irrigation-class "
            "kWh x 2.1085 (Changhua) / 2.2220 (Yunlin), I1.X5R1.v1's primary arm (that rung "
            "failed its gates; Q0 stays the page model, and Q0 carries no Changhua pumping). "
            "Both were calibrated on the same 2012-22 heads and fit them alike (hindcast "
            "anomaly amplitude ratio 1.00, correlation 0.98, interannual correlation 0.99 at "
            "the Changhua screens), yet their irrigation-only cut30 levers differ x1.389 "
            "(Changhua) / x1.273 (Yunlin) in 2026-27 and x1.386 / x1.272 in 2030-32, at every "
            "horizon (1.37-1.41 Changhua). Class-share arithmetic predicted 1.33 / 1.29: the "
            "heads pin the response to total pumping, and an irrigation-only cut scales with "
            "irrigation's share of the effective volume (fan: 0.556 in q0c, 0.731 in x5r1, "
            "ratio 1.31). Volume and irrigation share are not identifiable from the held "
            "heads (explorer changhua_inference: Bayesian volume, datum DC gain confounded "
            "with lambda < 0 in Changhua, transfer-function and two-way FE levers "
            "uninformative at sigma 1 m, class-split regressors' spread 0.5-0.85 > the 0.44 "
            "difference). The explorer's found/predicted ratios imply an all-class ratio near "
            "1.04 (Changhua) and 0.99 (Yunlin); that expectation is disclosed here, and the "
            "all-class forward has never been run. LOW-RISK PREDICTION: this arithmetic "
            "already puts R inside the gating band [0.9, 1.1] with margin, so the gate is a "
            "consistency check of the class-share decomposition, not a stretch prediction, "
            "and a pass is not support for any Changhua lever. A stricter band [0.95, 1.05] "
            "is pre-declared as report-only: it is reported for all four county-windows and "
            "never changes the verdict."),
        "disclosed": DISCLOSED,
        "arms": {
            "q0c": "theta results/twin_runs/stage3_datum_gate_q0c/stage3_theta.json (batch 27, "
                   "reported-only arm of I1.X5R1.v1), its leveling column "
                   "coupled_leveling/vep_zonal_leveling.json",
            "x5r1": "theta results/twin_runs/stage3_datum_gate_x5r1/stage3_theta.json (batch "
                    "27, I1.X5R1.v1 primary), its leveling column"},
        "scenarios": {
            "baseline": "climatology of the 2012-22 record, no change",
            "cut30": f"'{SCEN_CUT30}' (reproduction check only)",
            "all30": f"'{SCEN_ALL30}' (every policy class x0.7 from 2026-01: "
                     "forward.parse_scenario has no all-class keyword, so all six "
                     "scenario.CLASSES are named; neither theta carries a fixed extra "
                     "abstraction, sw recharge or eta classes, so this scales all pumping "
                     "energy the model sees)"},
        "recipe": {
            "code": f"main {WT_HEAD} in the clean detached worktree $HOME/HydroPhysicsAI-x5run "
                    "(the code batch 27 wrote datum_gate_{q0c,x5r1}.npz with), read-only, "
                    f"launched by {PY_SH} from the repository root; the worktree stays at "
                    f"{WT_HEAD} until the report is written",
            "forward_q0c": command("q0c"),
            "forward_x5r1": command("x5r1"),
            "runner": f"bash {RUNNER} forward (both arms, sequential, in tmux), then bash "
                      f"{RUNNER} score (once); sha256 {frozen[RUNNER]}",
            "scorer": f"{SCORER} (sha256 {frozen[SCORER]}), reads the two class_share npz, "
                      "the two batch-27 npz (reproduction), the registered declaration and "
                      "the attempt log; plumbing-checked on the batch-27 files (baseline, "
                      "cut30) before any all30 output exists",
            "binding": "the runner and the scorer refuse unless this declaration is "
                       "registered and intact and the sha256 of the scorer, the runner and "
                       "wells_frozen.json equal frozen_code; the scorer refuses npz files "
                       "that are not the outputs of the runner's last successful forward "
                       "attempt (attempts.jsonl); report.json records this declaration's "
                       "sha256",
            "gpu": "two forwards of about 4-5 min each on the T4 (about 10 min, under 0.2 "
                   "GPU-h); after or alongside (MPS) the current GPU jobs"},
        "frozen_code": frozen,
        "zones": (
            "Frozen well screens (cell, layer, county): the 158 screens of the explorer's "
            "extract (74 Changhua, 84 Yunlin; 72 distinct cells; layers L1 37, L2 79, L3 32, "
            "L4 10), county by zone_county_map v2; list embedded in the scorer, rows sha256 "
            f"{wells['rows_sha256']}"),
        "burn_in": "unchanged (forward origin 2022-12; the policy starts 2026-01)",
        "folds": NA("model-only comparison of two frozen calibrations; no folds"),
        "gate_scores": NA("one pass/fail gate on four ratios; no weighted score"),
        "primary_model": "the pair (q0c, x5r1); the gated quantity is the x5r1/q0c all30 "
                         "lever ratio",
        "inputs": (
            [{"path": p, "sha256": frozen[p]} for p in FROZEN]
            + [f"{X}/prereg_class_share.py", TESTS, PY_SH,
               "results/twin_runs/stage3_datum_gate_q0c/stage3_theta.json",
               "results/twin_runs/stage3_datum_gate_x5r1/stage3_theta.json",
               "results/twin_runs/stage3_datum_gate_q0c/coupled_leveling/"
               "vep_zonal_leveling.json",
               "results/twin_runs/stage3_datum_gate_x5r1/coupled_leveling/"
               "vep_zonal_leveling.json",
               "results/dataprep/pump_electricity/model_census/v1/kwh.parquet",
               "results/dataprep/x5_energy_volume/stage2_inputs/county/kwh.parquet",
               "results/twin_forward/datum_gate_q0c.npz",
               "results/twin_forward/datum_gate_x5r1.npz",
               "results/dataprep/zone_county_map/v2/zone_county_map.npz",
               "results/dataprep/preregistration/I1.X5R1.v1.json"]),
        "tests": NA("model-only declaration: the prediction is about two model runs, not "
                    "about observations; no fresh unit exists and none is consumed"),
        "fresh_units": NA("no observation is scored"),
        "thresholds": {
            "units": (
                "for county c in {changhua (74 screens), yunlin (84)} and window w in "
                f"{WINDOWS}: R(c, w) = mean over c's screens and w's months of (all30 - "
                "baseline) head in x5r1, divided by the same mean in q0c (ratio of means, "
                "heads_mean at each screen's own cell and layer)"),
            "engineering_checks_before_any_all30_value_is_read": [
                "step 0: this declaration registered and intact; frozen_code matches; the "
                "npz files are the last successful forward outputs in attempts.jsonl",
                "step 1: all four files (two new, two batch 27): scenario names and "
                "describe() strings exact; dates 2012-01..2032-12; origin 2022-12; horizon "
                "120; heads_mean (n_scen, 4, 2148, 252); member label stage3_datum_gate_<arm>; "
                f"gate meta pump_kwh and meter_filter of the arm; theta git {WT_HEAD}; forward "
                "options as batch 27; finite heads at the screens",
                "step 2 (reproduction): baseline and cut30 heads at the 158 screens match "
                "datum_gate_{q0c,x5r1}.npz (max |diff| <= 0.01 m), and the cut30 ratio "
                "reproduces the disclosed values (|diff| <= 0.01) in all four "
                "county-windows; only cut30 values are printed"],
            "evaluable_if": "q0c's all30 mean lever >= 0.05 m and x5r1's > 0 in every "
                            "county-window; otherwise 'not_evaluable'",
            "passes_if_all": [
                f"R(changhua, 2026-27) in [{BAND[0]}, {BAND[1]}]",
                f"R(changhua, 2030-32) in [{BAND[0]}, {BAND[1]}]",
                f"R(yunlin, 2026-27) in [{BAND[0]}, {BAND[1]}]",
                f"R(yunlin, 2030-32) in [{BAND[0]}, {BAND[1]}]"],
            "bounds": "inclusive, unrounded",
            "report_only_strict_band": (
                f"[{STRICT_BAND[0]}, {STRICT_BAND[1]}], inclusive: whether each of the four R "
                "lies inside, and whether all four do; pre-declared, never gating"),
            "reported_not_gating": [
                "the step-2 reproduction values",
                "the measured R range over the four county-windows and, per county, the "
                "residual factor [min(1, R), max(1, R)] over its two windows",
                "all30 and cut30 ratios by horizon (m1-3, m4-6, m7-12, m13-24, y5-7)",
                "cut30/all30 lever within each model (irrigation fraction of the all-class "
                "lever) against the census irrigation shares 0.556 / 0.731",
                "R by layer; per-screen ratios (median, p10, p90, share inside the band)",
                "all30 subsidence lever at the same cells, x5r1 over q0c"]},
        "evidence_and_ledger_mapping": {
            "scope": (
                "This declaration feeds only the evidence record "
                "(docs/TWIN_EVIDENCE_2026-10.md) and the claims ledger "
                "(hydrophysics/twin/claims_v1.json). It adds no page lever and changes no "
                "page wording by itself: the ledger governs page wording, and unsupported "
                "outputs are narrowed or withdrawn on the page (user ruling 2026-10-29). Row "
                "C13 ('Changhua class levers: not available in this model') stays as it is "
                "under every outcome; any change to it needs a later user ruling. Every "
                "statement below is about the q0c/x5r1 pair at the frozen screens (mean "
                "head lever over a county's screens), not about the page model Q0, which has "
                "no Changhua pumping."),
            "irrigation_only_lever_all_outcomes": (
                "the irrigation-only x0.7 head lever differs between the two conversions by "
                "the measured cut30 ratio, x5r1 over q0c: 1.39 (Changhua) and 1.27 (Yunlin), "
                "as reproduced in step 2. That figure is a ratio between the q0c (3.937 "
                "m3/kWh) and x5r1 (irrigation kWh x2.11 Changhua / x2.22 Yunlin) "
                "conversions only; it is not a bound on the true conversion, and it is never "
                "compounded with R (it already contains the all-class residual: 1.39 is "
                "about 1.33 class share x 1.04)."),
            "on_pass": [
                "record and ledger note: in the q0c/x5r1 pair the all-class x0.7 head lever "
                "is invariant to the conversion within +-10 %, quoted with the measured "
                "residual factor (the R range over the four county-windows, e.g. 'x5r1/q0c "
                "= 0.97-1.05'); never the bare word 'conversion-invariant'",
                "if all four R also lie in the report-only band [0.95, 1.05], the note may "
                "say 'within +-5 % (report-only band)'",
                "the pass is labelled a low-risk consistency check of the class-share "
                "decomposition; it is not cited as support for the Changhua head lever, "
                "which has only design-time support (step C) and is decided on observations "
                "only by N2/N2C",
                "absolute Changhua volume stays 'not identifiable from held data'"],
            "on_fail": [
                "record and ledger note, for each failing county: total-pumping (all-class) "
                "levers in the q0c/x5r1 pair carry the residual conversion factor "
                "[min(1, R), max(1, R)] over that county's two windows",
                "the statement 'heads pin the response to total pumping' is not used for "
                "that county in the record or the ledger, and is withdrawn wherever it "
                "already appears there",
                "the irrigation-only factor stays the measured cut30 ratio (above), not "
                "compounded with R",
                "absolute volume stays 'not identifiable from held data'"],
            "on_not_evaluable": [
                "record and ledger note 'not evaluable (q0c all-class lever below 0.05 m "
                "in <county-window>)'; no factor is quoted and nothing else changes"]},
        "attempt_policy": {
            "scored_runs": "one forward per arm and one scoring run; the first scorer run "
                           "that passes steps 0-2 computes the verdict, writes report.json "
                           "(open mode 'x') and is final; no rerun after that, whatever the "
                           "verdict",
            "values_written_before_scoring": (
                "hydrophysics.twin.forward computes the all30 scenario and writes its values "
                "before the scorer runs: its stdout prints a '=== forward summary ===' block "
                "(fan-mean layer-2 head change and subsidence for every scenario, all30 "
                "included) and a rheology-axis block, and it writes results/twin_forward/"
                "class_share_<arm>.npz, .summary.csv and .members.csv. The runner redirects "
                "the forward stdout to results/twin_runs/class_share/sealed/"
                "<arm>_forward.stdout.SEALED and records its sha256, and those of the npz "
                "and both CSVs, in results/twin_runs/class_share/attempts.jsonl. Nobody "
                "opens, tails, greps or loads the sealed logs, the class_share CSVs or the "
                "all30 slice of the npz before report.json exists; the runner prints only "
                "exit codes and sha256s. The forward stderr (tracebacks and Python warnings; "
                "forward.py prints no value there) goes to <arm>_forward.stderr.log and may "
                "be read to diagnose a crash."),
            "engineering_retries": (
                "free before any all30 value is READ (all30 values exist on disk after each "
                "forward, sealed as above): a forward crash, OOM or missing file, or a scorer "
                "failure in step 0, 1 or 2. Each attempt is logged in attempts.jsonl. Before "
                "a rerun, the cause is diagnosed from the stderr log, the scorer's messages "
                "(which print only cut30 values) and metadata, and fixed; then `bash "
                f"{RUNNER} archive '<reason>'` moves the attempt's outputs, unopened, to "
                "results/twin_runs/class_share/archive/<UTC stamp>/ and logs it, and the "
                "forwards are rerun. A fix that changes the scorer, the runner, "
                "wells_frozen.json, the worktree or the recipe cannot be run under this "
                "declaration (the sha256 binding refuses it): it needs a new version (v2) "
                "that discloses the failed attempts. Archived outputs are opened only after "
                "report.json exists."),
            "pre_declared_alternatives": [],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included (attempts.jsonl and the report)"},
        "interpretation": (
            "A pass says that in this model family the head response to a cut of ALL "
            "pumping is set by the heads the models were calibrated on and depends on the "
            "unresolved conversion by no more than the measured residual factor (within "
            "+-10 %). It is a low-risk consistency check: the disclosed arithmetic already "
            "expected about 1.04 and 0.99. It does not say any lever is right: both models "
            "share the structure, the Changhua head lever has only design-time support "
            "(step C), and N2/N2C remain the only route that can decide it on observations. "
            "A fail says the two calibrations differ beyond class share (storage or "
            "transmissivity traded against volume, I1.X5R1.v1's warning), so total-pumping "
            "levers in this pair inherit the measured conversion factor too. Being "
            "model-only, neither outcome is evidence about the aquifer; it decides how the "
            "levers are described in the evidence record and the claims ledger, and nothing "
            "on the page changes without a ledger ruling."),
        "rule_risks_disclosure": [
            "the cut30 ratios, the hindcast indistinguishability and the class-share "
            "arithmetic were computed before this declaration (all model output or "
            "pre-origin forcing); they are disclosed above and only the all30 ratio is "
            "predicted",
            "the explorer read fitted per-well datums (model parameters fitted on spent "
            "2012-22 heads) for a closed side check (exp6); no head value enters this "
            "declaration",
            "the drafter of this declaration also printed the first ~20 fitted per-well "
            "datums (well_datum) of both thetas while inspecting the forward npz gate meta "
            "(model parameters fitted on spent 2012-22 heads); nothing was computed from "
            "them",
            "the drafter and the reviewer re-read statistics batch 27 had already printed "
            "(stored gate status r2_kfold 0.70 vs IDW 0.71; in-sample calibration R2 0.845 "
            "q0c / 0.826 x5r1; hindcast subsidence vs leveling R2 +0.343, 2012-22); spent, "
            "not recomputed, not used",
            "the well-screen list came from an inputs extract whose observed-head arrays "
            "were nulled on load; only sids, cells and layers were kept; county by "
            "zone_county_map v2 (allowed use model_design)",
            "x5r1's forcing (results/dataprep/x5_energy_volume/stage2_inputs/county) has a "
            "build_info.json but no MANIFEST with allowed_uses; its forcing role rests on the "
            "I1.X5R1.v1 registration (an input here), and it is cited as that",
            "model_census v1 (q0c's forcing) allows forcing_frozen and model_design; its "
            "Jan-2023 half-bill exemption is already registered",
            "the forward log prints the theta's stored gate status and a 2012-22 hindcast "
            "leveling R2 (spent; printed by batch 27 already); it is sealed with the rest of "
            "the stdout and neither number is used"],
        "no_prior_look_statement": (
            "No forward run with an all-class scenario exists for q0c or x5r1 (or for any "
            "model); no all30 lever has been computed. No observation is read by the runs' "
            "scored output or by the scorer."),
        "written_by": f"{X}/prereg_class_share.py",
    }


def runner_problems() -> list[str]:
    sh = (REPO / RUNNER).read_text()
    lines = set(sh.splitlines())
    out = [f"{RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    out += [f"{RUNNER} does not carry {s!r}" for s in RUNNER_FRAGMENTS if s not in sh]
    if "/home/" in sh:
        out.append(f"{RUNNER} hardcodes a /home path")
    return out


def readiness() -> list[str]:
    """What stands between validation and registration (empty = ready)."""
    out = []
    for rec in doc()["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (REPO / p).exists():
            out.append(f"input {p} is missing")
    if (REPO / f"results/dataprep/preregistration/{PREREG_ID}.json").exists():
        out.append(f"{PREREG_ID} is already registered")
    for s in ("npz", "summary.csv", "members.csv"):
        for a in ("q0c", "x5r1"):
            if (REPO / f"results/twin_forward/class_share_{a}.{s}").exists():
                out.append(f"results/twin_forward/class_share_{a}.{s} exists before the "
                           "declaration")
    if (REPO / REPORT).exists():
        out.append(f"{REPORT} exists")
    wt = Path.home() / "HydroPhysicsAI-x5run"
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        return out + [f"worktree $HOME/HydroPhysicsAI-x5run: {err}"]
    if head != WT_HEAD:
        out.append(f"worktree $HOME/HydroPhysicsAI-x5run is at {head}, not {WT_HEAD}")
    if dirty:
        out.append("worktree $HOME/HydroPhysicsAI-x5run is dirty")
    py = REPO / PY_SH
    if not py.exists() or f"export PYTHONPATH={wt}" not in py.read_text().splitlines():
        out.append(f"{PY_SH} does not pin the worktree")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--register", action="store_true",
                    help="write the declaration once (prereg3.write_prereg), under the X5 lock")
    ap.add_argument("--expect-self-sha", default=None,
                    help="sha256 of this script as reviewed; required with --register")
    ap.add_argument("--json", action="store_true", help="print the document")
    a = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    probs = runner_problems()
    wells = json.loads((REPO / WELLS).read_text())
    sys.path.insert(0, str(HERE))
    import score_class_share as S  # noqa: E402

    if S._rows_sha() != wells["rows_sha256"]:
        probs.append("scorer well list differs from wells_frozen.json")
    if (S.BAND, S.STRICT_BAND, S.PREREG_ID) != (BAND, STRICT_BAND, PREREG_ID):
        probs.append("scorer bands or id differ from the declaration")
    print("recipe/binding checks:", probs or "ok")
    now = dt.datetime.now(dt.timezone.utc)
    tests = d["tests"] if isinstance(d["tests"], list) else []
    block = prereg3.blocking_looks(tests, None, test_time=now)
    if isinstance(d.get("fresh_units"), list):
        block += prereg3.blocking_looks([{"test_id": "<declaration>", "data_scope": "*",
                                          "fresh_units": d["fresh_units"]}], None,
                                        test_time=now)
    print("blocking looks (dry run):", block or "none (declaration with no tests and no "
          "fresh units)")
    pending = readiness()
    print("readiness:", pending or "ready to register")
    self_sha = sha256(Path(__file__).resolve())
    print(f"this script sha256 {self_sha}")
    print("frozen_code:", json.dumps(d["frozen_code"], indent=1))
    if a.json:
        print(json.dumps(d, indent=1, ensure_ascii=False))
    bad = bool(errs or probs or block or pending)
    if not a.register:
        print("valid; not written (pass --register to write)" if not bad else
              "NOT READY: fix the problems above")
        return 1 if bad else 0
    if a.expect_self_sha != self_sha:
        print("REFUSED: --register needs --expect-self-sha equal to this script's sha256")
        return 1
    if bad:
        print("REFUSED: not ready to register")
        return 1
    path, h = prereg3.write_prereg(d)
    print(f"registered {path.relative_to(REPO)} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
