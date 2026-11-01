"""Declaration FS.start_invariance.v1, kind 'declaration', before any arm's forward run exists:
does the page's head lever (C01: policy minus baseline, mean over N2's 84 Yunlin wells, each in
its own screened layer) depend on the forecast's 2022-12 start state?

Model-only, heads only. No observation of any family is read or scored. The declaration feeds
the evidence record (docs/TWIN_EVIDENCE_2026-10.md) and the claims ledger
(hydrophysics/twin/claims_v1.json: C01 wording and one new model-internal row); it moves no
claim's status (user ruling 2026-10-29: the ledger governs page wording).

``--validate`` (dry run) prints the document, its canonical sha256 and this script's sha256,
runs prereg3.validate_prereg, checks that the runner and the scorer carry the declared recipe,
runs prereg3.blocking_looks as a dry run and reports readiness; nothing is written.
``--write`` registers it once (prereg3.write_prereg; the lead only). From the repository root:

    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fs_start/prereg_fs.py --validate
    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fs_start/prereg_fs.py --write \\
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
sys.path.insert(0, str(HERE))

import score_fs as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable

PREREG_ID = "FS.start_invariance.v1"
X = "dataprep_staging/fs_start"
SCORER = f"{X}/score_fs.py"
RUNNER = f"{X}/run_fs.sh"
FROZEN = (SCORER, RUNNER)
TESTS = f"{X}/tests/test_fs_start.py"
REPORT = f"results/dataprep/reports/{PREREG_ID}/report.json"
LOGDIR = "results/twin_runs/fs_start"
X5_WELLS = "dataprep_staging/x5_class_share/wells_frozen.json"
X5_WELLS_ROWS_SHA256 = "7ce6a13dd3b4381f972f180c8bad8ea6ddfe9680bf0a9089c80eecf8152cd335"
N2_PRED = "results/twin/prospective_2026-11/predictions.csv"

# the code ladder: main at drafting, then 730f65b (the first commit with --update-method and
# the last before the page run was written). fa5cca2 (the theta's commit) has no
# --update-method, so it cannot run the page recipe.
CODES = ("4c2723e", "730f65b")
G = "results/twin_runs/stage3_datum_gate"
PAGE_THETAS = (f"--theta {G}/stage3_theta.json --theta {G}/stage3_fold_thetas.json "
               f"--theta {G}/stage3_posterior_tn.json")
ONE_THETA = f"--theta {G}/stage3_theta.json"
VEP = (f"{G}/coupled_leveling/vep_zonal_leveling.json,"
       f"{G}/coupled_leveling_tau30/vep_zonal_leveling.json")
SCENARIOS = ("cut30:irrigation=0.7@2026-01", "retire_aqua:aquaculture=0@2026-01",
             "cut30_2030:irrigation=0.7@2030-01", "retire_aqua_2030:aquaculture=0@2030-01")
# the page run (2026-10-11 01:08 +0800) predates the CWA CODIS rain rebuild (18:39 the same
# day), so it read the vendor rain cache; the default rf_timeseries.csv is now the rebuild
RF = "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"
RF_SHA256 = "ef5a525f5ea51697af81f2a9cd4a0fce8cad074f7cfe10121920b8e4f6a63714"
COMMON_SH = ("--horizon 120 --apex-hold calibrated --column-heads free "
             "--column-hpc0-fast-days 365 --rf-timeseries $RF --device cuda --compile-matvec")
COMMON = COMMON_SH.replace("$RF", RF)
# every other file the forward reads at the page theta (calibrate_flow.DEFAULT_PATHS minus
# the rain, the DEM the theta's ground_elev 'dem' names); sha256 at drafting, all older than
# the page run. The head-well directory is declared as a digest (input_digests)
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
ARM_FLAGS = {
    "S0": "--gain 1.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation "
          "--restart-taper-km 5 --save-members none",
    "S1": "--gain 0.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation "
          "--restart-taper-km 5 --save-members none",
    "S2": "--gain 1.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method field "
          "--save-members none",
    "S3": "--gain 1.0 --ic-members 8 --ic-sigma 2.0 --seed 0 --update-method innovation "
          "--restart-taper-km 5 --save-members yearly",
    "S3ref": "--gain 1.0 --ic-members 0 --update-method innovation --restart-taper-km 5 "
             "--save-members none",
}
ARM_THETAS = {"S0": PAGE_THETAS, "S1": PAGE_THETAS, "S2": PAGE_THETAS, "S3": ONE_THETA,
              "S3ref": ONE_THETA}
RUNNER_VAR = {"S0": "F_S0", "S1": "F_S1", "S2": "F_S2", "S3": "F_S3", "S3ref": "F_S3REF"}
# the runner must carry these lines verbatim (one source of the recipe, checked here)
RUNNER_LINES = (
    f"ID={PREREG_ID}", "DECL=results/dataprep/preregistration/$ID.json",
    f"CODES=({' '.join(CODES)})", "R=results/twin_runs", "G=$R/stage3_datum_gate",
    "LOG=$R/fs_start", "REPORT=results/dataprep/reports/$ID/report.json",
    'PAGE_THETAS="--theta $G/stage3_theta.json --theta $G/stage3_fold_thetas.json '
    '--theta $G/stage3_posterior_tn.json"',
    'ONE_THETA="--theta $G/stage3_theta.json"',
    'VEP="$G/coupled_leveling/vep_zonal_leveling.json,'
    '$G/coupled_leveling_tau30/vep_zonal_leveling.json"',
    "SCENARIOS=(" + " ".join(f"'{s}'" for s in SCENARIOS) + ")",
    f"RF={RF}",
    f'COMMON="{COMMON_SH}"',
    *(f'{RUNNER_VAR[a]}="{f}"' for a, f in ARM_FLAGS.items()))
RUNNER_FRAGMENTS = (
    'PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.forward $(arm_flags "$a")',
    '--vep-json "$VEP" --out "$out" > "$sealed" 2> "$err"',
    'WT="$HOME/HydroPhysicsAI-fsrun-$CODE"',
    'k=$(count_records event=repro rc=3)')


def command(arm: str, wt: str = "$HOME/HydroPhysicsAI-fsrun-<code>") -> str:
    """The forward command line of ``arm`` (as the runner's DRY=1 prints it)."""
    scen = "".join(f" --scenario '{s}'" for s in SCENARIOS)
    return (f"PYTHONPATH={wt} $HOME/miniforge3/envs/hydro/bin/python -P -m "
            f"hydrophysics.twin.forward {ARM_THETAS[arm]} {ARM_FLAGS[arm]}{scen} {COMMON} "
            f"--vep-json {VEP} --out results/twin_forward/fs_start_{arm}")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def frozen_code() -> dict:
    return {p: sha256(REPO / p) for p in FROZEN}


def canonical_sha(d: dict) -> str:
    return hashlib.sha256((json.dumps(d, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


INPUT_SHA256 = {
    S.REF: S.REF_SHA256,
    f"{G}/stage3_theta.json": "036a7a17dcbd383fba9eaab6a12e716ab0cdffcdae0385adaf5349a57c64de77",
    f"{G}/stage3_fold_thetas.json":
        "c70632858febe2bbe57f5fe4f17646e66c64a56812699f952cf505fcdf498a89",
    f"{G}/stage3_posterior_tn.json":
        "5b194e1e7bde28b72c18d977f0321e20e12aca411b14ae2ef408d749018ec601",
    f"{G}/coupled_leveling/vep_zonal_leveling.json":
        "d290c59209f758389b7133c3f5b9aadcc71b42ad55f9089d56645fafc547b162",
    f"{G}/coupled_leveling_tau30/vep_zonal_leveling.json":
        "7f799b590fb8e82043204424aa9af001206bb6a542415197606baf08d8518d81",
    "results/twin/prospective_2026-11/manifest.json":
        "d7286551245644b7932fac98751bcd0e837611ffdfbc9a403dd096c8c51af056",
    N2_PRED: "16ac3c3783cea9f3c94560448d06c64ec8dc8fbb450effa28b465469a931cc32",
    **FORWARD_INPUTS,
}

DISCLOSED = {
    "page_run_n2_unit_dec2032_m": {
        "what": "the page run's C01 numbers recomputed by score_fs.py --check-reference from "
                "results/twin_forward/datum_gate_innov.npz (model output, already on the page "
                "as +1.19 / +1.41 m); no other value of any run was computed",
        "cut30": 1.1911, "retire_aqua": 1.4116},
    "code_provenance": (
        "the page run datum_gate_innov.npz (sha256 cf2c7154..., written 2026-10-11 01:08 "
        "+0800) records forward_options update_method 'innovation', column_hpc0_fast_days 365 "
        "and restart_taper_km 5; the first forward.py with --update-method is 730f65b "
        "(2026-10-11 00:27 +0800), also the last commit before the file was written, so the "
        "theta's commit fa5cca2 cannot run the page recipe. No diff of the run was saved, so "
        "the code that wrote it is not verified. The 'worktree at fa5cca2' fallback of the "
        "round-5 plan is replaced by the ladder 4c2723e (main at drafting) -> 730f65b, "
        "declared here before any run; both parse every arm's command, the rain flag "
        "included (tests/test_fs_start.py::test_ladder_parses_every_arm extracts each rung "
        "with git archive and stops each forward right after its argument parse)"),
    "rain_input": (
        "the forward takes its rain from calibrate_flow.DEFAULT_PATHS['rf_timeseries'] "
        "(chou-shui-data/data/rf_timeseries.csv) unless --rf-timeseries is given. That file "
        "was replaced by the CWA CODIS rebuild on 2026-10-11 at 18:39 +0800; the page run "
        "was written at 01:08 the same day, on the vendor cache, which is kept as "
        f"{RF} (sha256 {RF_SHA256[:8]}; docs/TWIN_RELEASE.md: the page theta was fitted on "
        "it). Every arm therefore passes --rf-timeseries with that file (the flag exists at "
        "both rungs). The kept file's bytes cannot be proved identical to what the page run "
        "read (no input hash was recorded in the run); the S0 reproduction gate tests it. "
        "At drafting the file was checked for structure only (4018 days 2012-01-01.."
        "2022-12-31, the same 26 gauge columns as the rebuild, all in rf_stations.csv, no "
        "empty cell, monthly means finite under 730f65b's loader logic); no rain value was "
        "printed"),
    "forward_inputs": (
        "every file the forward reads at the page theta is declared with its sha256 in "
        "inputs (the polygon, stations, pump census and kWh, rain, rain stations, gw "
        "stations, ET0, the DEM basemap) and the head-well directory as a digest in "
        "input_digests; all were older than the page run at drafting (mtimes). "
        "prereg3.load_prereg re-hashes the files and the scorer re-computes the digest at "
        "every step 0, which the runner calls before every forward, so drift refuses before "
        "GPU time is spent"),
    "page_recipe": (
        "parameter sets: stage3_theta.json + stage3_fold_thetas.json (5) + "
        "stage3_posterior_tn.json (12) = 18 (labels stage3_datum_gate, fold0-4, "
        "foldpost0-11, as recorded in the page run); --ic-members 1 (36 head members, as "
        "recorded); --ic-sigma and --seed were not recorded and are taken at the CLI "
        "defaults 0.5 and 0 (the same at every code of the ladder); rheology axis "
        "coupled_leveling + coupled_leveling_tau30 (recorded labels coupled_leveling, "
        f"tau30y); rain {RF} via --rf-timeseries (see rain_input); the S0 gate tests this "
        "reconstruction"),
}


def doc() -> dict:
    frozen = frozen_code()
    return {
        "id": PREREG_ID, "wp": "FS", "kind": "declaration",
        "title": "Start-state invariance of the page's head lever: is the N2-unit lever "
                 "(cut30, retire_aqua; starts 2026-01 and 2030-01) of the page model the "
                 "same from a free-run start (gain 0) and from the old untapered field "
                 "restart as from the page's innovation restart? (ratio in [0.95, 1.05] and "
                 "per-well p90 deviation <= 0.10 at N2's 84 Yunlin wells; heads only; "
                 "model-only, moves no claim status)",
        "priors": (
            "Model-only (no observation is read or scored). In this model the head lever "
            "depends on the start state only through nonlinear terms, chiefly the lift term "
            "of the energy-to-volume conversion (pumped volume proportional to E / (ground - "
            "h + head_extra), head_extra fixed at 40 m, evaluated on the simulated head), "
            "plus the apex hold and boundary conductances. A start-state difference dh in "
            "the pumped layer changes the lever by roughly dh / (lift + 40 m): about 2-4 % "
            "for the 1-2 m differences expected between the innovation restart and the "
            "free run (S1). S2, the untapered field restart that wrote a median -21/-25 m "
            "into the unmeasured proximal layers 3-4 (and replaces every layer that has a "
            "well wholesale), is a stress case: a fail there is plausible. S3 (independent "
            "zero-mean noise per well, sigma 2 m) mostly cancels in the 84-well mean and "
            "decays by diffusion, so it is report-only. Predicted |ratio - 1|: 0.02-0.04 "
            "for S1; not predicted for S2. This is not a stretch prediction for S1; a pass "
            "is a fact about the model, not about the aquifer."),
        "disclosed": DISCLOSED,
        "arms": {
            "S0": "rerun of the page recipe (page theta stage3_datum_gate with its 17 fold "
                  "and posterior sets, innovation restart at the Dec-2022 origin, gain 1, "
                  "5 km taper); reproduction gate against the page run",
            "S1": "S0 with --gain 0: every member starts the projection from its own "
                  "free-running Dec-2022 state (no observation update); GATING",
            "S2": "S0 with --update-method field and no --restart-taper-km: the old "
                  "untapered IDW-field restart (the -21/-25 m proximal step of the "
                  "superseded datum_gate.npz); --apex-hold calibrated kept, so only the "
                  "start state differs; GATING",
            "S3": "the page theta alone (no fold or posterior sets), --ic-members 8 "
                  "--ic-sigma 2.0 --seed 0, innovation restart; REPORT-ONLY",
            "S3ref": "the page theta alone, --ic-members 0 (S3's unperturbed member); the "
                     "base of S3's ratios; REPORT-ONLY",
            "S5_dropped": "the recorded-pumping (X1) start arm is dropped: no ruling was "
                          "asked for it and X1's only twin use is forcing_recorded_arm",
            "no_column_heads_restart_arm": "every arm keeps --column-heads free: the "
                                           "compaction column stays driven by the free-running "
                                           "heads it was fitted on (the column pairs with its "
                                           "head field); a --column-heads restart arm is "
                                           "excluded and no subsidence quantity is scored "
                                           "(with free column heads it is identical across "
                                           "S0, S1 and S2 by construction)"},
        "scenarios": {
            "baseline": "climatology of the 2012-22 record, no change",
            "cut30": f"'{SCENARIOS[0]}' (the page's)",
            "retire_aqua": f"'{SCENARIOS[1]}' (the page's)",
            "cut30_2030": f"'{SCENARIOS[2]}' (the page's 2030 start, from the basis)",
            "retire_aqua_2030": f"'{SCENARIOS[3]}'"},
        "recipe": {
            "code": (
                "the forward runs from a clean detached worktree $HOME/HydroPhysicsAI-fsrun-"
                f"<code> via PYTHONPATH and python -P, launched from the repository root. "
                f"Ladder: {CODES[0]} (main at drafting) first; after a logged S0 "
                f"reproduction failure {CODES[1]} (the first commit with --update-method and "
                "the last before the page run was written). Every arm of the scoring runs "
                "the same code; the runner refuses otherwise"),
            **{f"forward_{a}": command(a) for a in ARM_FLAGS},
            "runner": f"bash {RUNNER} forward S0, repro, forward S1/S2/S3/S3ref (tmux, MPS "
                      f"allowed), score (once); sha256 {frozen[RUNNER]}",
            "scorer": f"{SCORER} (sha256 {frozen[SCORER]}); reads the five fs_start npz, "
                      "S3's members npz, the page run, the registered declaration and the "
                      "attempt log",
            "binding": "the runner and the scorer refuse unless this declaration is "
                       "registered and intact and the sha256 of the scorer and the runner "
                       "equal frozen_code; the scorer refuses npz files that are not the "
                       "outputs of each arm's last successful forward attempt, arms run on "
                       "different code, or an S0 file without a logged passing repro; "
                       "report.json records this declaration's sha256",
            "gpu": "S0, S1, S2: 18 parameter sets x 2 initial fields x 5 scenarios x 2 "
                   "rollouts (column-heads free), about 50-60 min each on the T4 (the page "
                   "run, 3 scenarios, took 34 min under MPS); S3 about 20-30 min; S3ref "
                   "about 5 min; about 3 GPU-h in all. S0 and its repro must finish before "
                   "any other arm starts (the runner refuses otherwise), so the wall time is "
                   "about 1 h for S0 plus about 1.5 h with S1, S2, S3 and S3ref under MPS, "
                   "about 2.5 h. Check free GPU memory first (the runner refuses below "
                   "5000 MiB); no twin-module edit while it runs (code freeze)"},
        "frozen_code": frozen,
        "zones": (
            "Frozen well screens (cell, layer, county): N2's 158 registered stations (station "
            "ids of results/twin/prospective_2026-11/predictions.csv; 84 Yunlin, county code "
            "09, the gated unit; 74 Changhua, report-only), cells and layers as in "
            f"{X5_WELLS} (rows sha256 {X5_WELLS_ROWS_SHA256}); embedded in the scorer "
            f"without station ids, screens sha256 {S.SCREENS_SHA256}. The page's C01 "
            "computation on the page run with these screens gives +1.1911 / +1.4116 m "
            "(published +1.19 / +1.41)"),
        "burn_in": "unchanged (forward origin 2022-12; policies start 2026-01 or 2030-01)",
        "folds": NA("model-only comparison of forward arms of one frozen calibration; no "
                    "folds"),
        "gate_scores": NA("one pass/fail gate on two co-primaries; no weighted score"),
        "primary_model": "the page model (theta stage3_datum_gate, 036a7a17, with its fold "
                         "and posterior sets, page run datum_gate_innov); the gated "
                         "quantities are arm/S0 ratios of its head lever",
        "inputs": (
            [{"path": p, "sha256": frozen[p]} for p in FROZEN]
            + [{"path": p, "sha256": h} for p, h in INPUT_SHA256.items()]
            + [f"{X}/prereg_fs.py", TESTS, X5_WELLS]),
        "input_digests": INPUT_DIGESTS,
        "tests": NA("model-only declaration: the prediction is about forward runs of one "
                    "model, not about observations; no fresh unit exists and none is "
                    "consumed"),
        "fresh_units": NA("no observation is scored"),
        "thresholds": {
            "unit": (
                "the N2 unit: for scenario s, l_i(s, w) = mean over window w's months of "
                "(s - baseline) head at well i's cell in its own screened layer "
                "(heads_mean, the ensemble mean of the arm); the N2-unit lever is the mean "
                "of l_i over N2's 84 Yunlin wells (the page's C01 quantity)"),
            "gated_cells": (
                "levers cut30 and retire_aqua; starts 2026-01 and 2030-01; windows 2026-27 "
                "(months 168-191), 2030-32 (216-251) and Dec 2032 (251); the 2026-27 window "
                "is not applicable to a 2030 start (no lever yet): 10 cells per arm, arms "
                "S1 and S2 (20 cells)"),
            "co_primary_1": (
                f"R = N2-unit lever of the arm / that of S0, inside [{S.BAND[0]}, "
                f"{S.BAND[1]}] (inclusive, unrounded)"),
            "co_primary_2": (
                "p90 over the 84 wells (numpy.percentile, linear) of |l_i(arm) - l_i(S0)| / "
                f"max(|l_i(S0)|, {S.LEVER_FLOOR_M} m) <= {S.P90_MAX} (inclusive), so that "
                "cancellation in the mean cannot hide start dependence; the count of wells "
                "under the floor is reported"),
            "evaluable_if": (
                f"S0's N2-unit lever >= {S.MIN_LEVER_M} m in every gated cell; otherwise the "
                "verdict is 'not_evaluable'"),
            "passes_if_all": "both co-primaries hold in all 20 gated cells (S1 and S2, both "
                             "levers, both starts, every applicable window)",
            "engineering_checks_before_any_gated_value_is_read": [
                "step 0: this declaration registered and intact; frozen_code matches; every "
                "declared input file (the forward's included) and the head-well directory "
                "digest unchanged; every "
                "npz is its arm's last successful forward output in attempts.jsonl; all "
                "arms ran the same code; a passing repro is logged for this S0 file",
                "step 1: every file: scenario names and describe() strings exact; dates "
                "2012-01..2032-12; origin 2022-12; horizon 120; heads_mean (n_scen, 4, 2148, "
                "252); member labels, n_members, gain and forward options of the arm; theta "
                "git fa5cca2; rheology axis (coupled_leveling, tau30y); finite heads",
                f"step 2 (reproduction, run as `{RUNNER} repro` right after S0 and again "
                "at scoring): S0's baseline, cut30 and retire_aqua heads at the 158 screens "
                f"match the page run in every month (max |diff| <= {S.REF_TOL_M} m), S0's "
                f"N2-unit Dec-2032 levers match the page run's within {S.REF_TOL_M} m, and the "
                "page run's match the published +1.19 / +1.41 m within "
                f"{S.PUBLISHED_TOL_M} m (half the rounding unit); only these page-published "
                "quantities are printed"],
            "bounds": "inclusive, unrounded",
            "report_only_band": (
                f"[{S.REPORT_BAND[0]}, {S.REPORT_BAND[1]}] on R, per cell; pre-declared, "
                "never gating"),
            "reported_not_gating": [
                "the step-2 reproduction values",
                "the maximum |R - 1| (as x %) and the maximum p90 over the gated cells; "
                "failing cells and arms",
                "the same R and p90 at Changhua's 74 wells and R of the fan layer-2 mean",
                "R by layer at the Yunlin wells",
                "the N2-unit lever at every December 2026-2032 for S0, S1 and S2 (the page "
                "tile's series)",
                "the baseline level shift, arm minus S0, at 2023-01, 2026-12 and 2032-12 "
                "(N2 unit and fan layer 2): 'the start moves levels'",
                "S3: the ensemble-mean R and p90 against S3ref; per initial field, the "
                "layer-2 December lever at the Yunlin screens' cells over the unperturbed "
                "member's (from the quantised members file, about +-1 % quantisation) and "
                "the share inside [0.95, 1.05]"]},
        "evidence_and_ledger_mapping": {
            "scope": (
                "Model-only: no outcome moves C01's status (S+D) or any other claim's. The "
                "declaration feeds the evidence record and the claims ledger; any page "
                "wording follows the ledger (user ruling 2026-10-29). The heads are the "
                "page model's; the compaction column is not tested here."),
            "on_pass": [
                "C01 gains: 'does not depend on the 2022-12 start state (within 5 %, "
                "model-only declaration FS.start_invariance.v1)', quoted with the measured "
                "maximum |R - 1|",
                "a new ledger row (status L, model-internal check) for the start "
                "invariance, cited by C01",
                "the start note (renderStartNote) may say that the start moves the levels "
                "(the baseline level shift is reported), not the policy differences"],
            "on_fail": [
                "the verdict stays 'fail' under the gate as declared (all 20 cells); the "
                "wording is chosen by report.json summary.wording_case, and x for an arm is "
                "its per_arm maximum |R - 1| in % (or its maximum p90 in % if only "
                "co-primary 2 fails for that arm)",
                "wording_case 'S2_only' (S1 passes, S2 fails): C01 adds 'does not depend on "
                "the start under the page's restart family (free-run start S1 within 5 %, "
                "model-only); it depends on the start by up to x % only under the superseded "
                "untapered field restart (S2, model-only declaration FS.start_invariance."
                "v1)', x from S2; the S1 result is stated separately with its own maximum "
                "|R - 1|",
                "wording_case 'S1_only' or 'S1_and_S2': C01 adds 'depends on the 2022-12 "
                "start state by up to x % (model-only)', x the largest over the failing "
                "arms, naming them (S1 free-run start, S2 old field restart) with each "
                "arm's own x",
                "the new ledger row records the failure and the wording case (status L); "
                "the start is decided only by the prospective tests (E02, N2)"],
            "on_not_evaluable": [
                "a ledger note 'not evaluable (S0 lever below 0.05 m in <cell>)'; nothing "
                "else changes"],
            "independent_of_this_run": (
                "the code fact that the subsidence fields start from the model's own "
                "simulated Dec-2022 heads (column-heads free) is ledger wording for C29/C30 "
                "under rank 9; it needs no run and is not decided here")},
        "attempt_policy": {
            "scored_runs": "one forward per arm (per code rung) and one scoring run; the "
                           "first scorer run that passes steps 0-2 computes the verdict, "
                           "writes report.json (open mode 'x') and is final; no rerun after "
                           "that, whatever the verdict",
            "code_ladder": (
                f"S0 runs first at {CODES[0]}; `repro` (scorer steps 0-2 on S0 only) is "
                "logged with its exit code. Exit 3 (reproduction failure) moves the runner "
                f"to {CODES[1]}: the S0 outputs are archived unopened beyond the printed "
                "reproduction values, a clean worktree at that code is created, and S0 is "
                "rerun. Each rung is an engineering attempt, logged in attempts.jsonl. S1, "
                "S2, S3 and S3ref run only after a passing repro at the current code. If S0 "
                "fails at both, nothing more runs under v1; a v2 must disclose these "
                "attempts"),
            "values_written_before_scoring": (
                "hydrophysics.twin.forward writes every scenario's values before the scorer "
                "runs: its stdout prints a '=== forward summary ===' block (fan-mean layer-2 "
                "head change and subsidence per scenario), a rheology-axis block, the "
                "theta's stored gate status and a 2012-22 hindcast R2 per member, and it "
                f"writes results/twin_forward/fs_start_<arm>.npz, .summary.csv, "
                ".members.csv (S3 also .members.npz). The runner redirects stdout to "
                f"{LOGDIR}/sealed/<arm>_forward.stdout.SEALED and records its sha256, and "
                f"those of the outputs, in {LOGDIR}/attempts.jsonl. Nobody opens, tails, "
                "greps or loads the sealed logs, the fs_start CSVs or any arm's npz before "
                "report.json exists, except S0's page-published quantities through `repro`; "
                "the runner prints only exit codes, sha256s and free GPU memory. The "
                "forward stderr (tracebacks and warnings; forward.py prints no value there) "
                "goes to <arm>_forward.stderr.log and may be read to diagnose a crash"),
            "engineering_retries": (
                "free before report.json exists: a forward crash, OOM or missing file, or a "
                "scorer failure in step 0, 1 or 2. Each attempt is logged. Before a rerun "
                "the cause is diagnosed from the stderr log, the scorer's messages (which "
                "print only page-published quantities) and metadata, and fixed; then `bash "
                f"{RUNNER} archive '<reason>' [arm ...]` moves the outputs, unopened, to "
                f"{LOGDIR}/archive/<UTC stamp>/ and logs it, and the forward is rerun. A fix "
                "that changes the scorer, the runner or the recipe cannot be run under this "
                "declaration (the sha256 binding refuses it): it needs a v2 that discloses "
                "the failed attempts. Archived outputs are opened only after report.json "
                "exists"),
            "pre_declared_alternatives": [
                f"the code ladder {' -> '.join(CODES)} (engineering only)"],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included (attempts.jsonl and the report)"},
        "interpretation": (
            "A pass says that in the page model the head response to a pumping cut, in N2's "
            "unit, is set by the model's parameters and forcing and not by how the Dec-2022 "
            "state was started (innovation restart, free run or the old field restart), "
            "within 5 % on the mean and 10 % per well: the start moves the level, not the "
            "difference. It says nothing about whether the response size is right (that "
            "stays with N2) or about the level (E02c). A fail says the page's lever carries "
            "a start-state dependence of the measured size, which C01 then states; the "
            "start itself is decided only by the prospective tests. Being model-only, "
            "neither outcome is evidence about the aquifer."),
        "rule_risks_disclosure": [
            "the drafter printed the page run's stored gate status line (r2_kfold, r2_idw, "
            "anomaly k-fold, verdicts; spent statistics already in the evidence record) "
            "while reading the head of datum_gate_innov.log for the recipe; nothing was "
            "computed from it",
            "the drafter read the page run's metadata (scenario names, member labels, "
            "forward options, gate keys, shapes) and the basis run's metadata, and "
            "computed only the page run's published N2-unit Dec-2032 levers (disclosed "
            "above); no other head or subsidence value of any run was read",
            "the screen list's station ids were compared with N2's registered predictions "
            "(station_id and layer columns only); no prediction value was read",
            "the fixer hashed the forward's input files and the head-well directory (bytes "
            "hashed, no file parsed except the vendor rain cache, which was checked for "
            "structure only: shape, dates, gauge columns, empty-cell count and finiteness of "
            "monthly means; no rain value printed) and read only metadata keys of the page "
            "theta (input paths, ground_elev, rivers)",
            "the forward reads the 2012-22 inputs (forcing and the pre-origin heads it "
            "restarts from) as every page forward does; it may read the 2012-22 leveling "
            "to print a hindcast skill line, which is sealed with the stdout and not used",
            "the arms differ from the page only in the start-state flags and the two "
            "added 2030-start scenarios (which do not change the other scenarios' "
            "rollouts: the initial-field noise is drawn once per run from the seed)"],
        "no_prior_look_statement": (
            "No forward run of the page model with --gain 0, with the field update, or with "
            "--ic-members 8 --ic-sigma 2.0 exists; no S1, S2 or S3 lever has been computed. "
            "No observation is read by the scored outputs or by the scorer."),
        "written_by": f"{X}/prereg_fs.py",
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
    if S.SCEN_NAMES[1:] != tuple(s.split(":")[0] for s in SCENARIOS):
        out.append("scorer scenario names differ from the runner's")
    if S._screens_sha() != S.SCREENS_SHA256:
        out.append("scorer screen list differs from its sha")
    w = REPO / X5_WELLS
    if w.exists():
        rows = json.loads(w.read_text())
        if rows.get("rows_sha256") != X5_WELLS_ROWS_SHA256:
            out.append(f"{X5_WELLS} rows sha changed")
        got = [(r["cell"], r["layer"], r["county"][0].upper()) for r in rows["rows"]]
        if got != list(zip(S.CELLS, S.LAYERS, "".join(S.COUNTY), strict=True)):
            out.append(f"scorer screens differ from {X5_WELLS}")
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
    except S.EngineeringError as err:
        out.append(str(err))
    if (REPO / f"results/dataprep/preregistration/{PREREG_ID}.json").exists():
        out.append(f"{PREREG_ID} is already registered")
    for a in S.ARMS:
        for s in ("npz", "summary.csv", "members.csv", "members.npz"):
            if (REPO / f"results/twin_forward/fs_start_{a}.{s}").exists():
                out.append(f"results/twin_forward/fs_start_{a}.{s} exists before the "
                           "declaration")
    if (REPO / REPORT).exists():
        out.append(f"{REPORT} exists")
    wt = Path.home() / f"HydroPhysicsAI-fsrun-{CODES[0]}"
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return out + [f"worktree $HOME/HydroPhysicsAI-fsrun-{CODES[0]} missing (git worktree "
                      f"add --detach $HOME/HydroPhysicsAI-fsrun-{CODES[0]} {CODES[0]})"]
    if head != CODES[0]:
        out.append(f"worktree $HOME/HydroPhysicsAI-fsrun-{CODES[0]} is at {head}")
    if dirty:
        out.append(f"worktree $HOME/HydroPhysicsAI-fsrun-{CODES[0]} is dirty")
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
