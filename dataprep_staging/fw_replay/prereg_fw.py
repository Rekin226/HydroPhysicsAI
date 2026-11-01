"""Declaration FW.weather_replay.v1, kind 'declaration', before any replay forward exists:
does a dry year expose the compaction column's permanent (inelastic) channel in the page
model? Model-only; no observation of any family is read or scored.

The engine is the new opt-in module hydrophysics/twin/replay.py (forward.py untouched); it
must be committed first, and its commit is the run code (``--run-code``). The replayed fields
are fixed here, at registration: 'energy+weather' when a pre-origin rain manifest allowing
forcing_frozen or design_time_evidence exists for the forward's rain input (manifest_rain.py,
a lead step), otherwise 'energy' (pumping anomalies only; rain, recharge and canal water at
climatology), which the wording then says.

``--validate`` (dry run) prints the document, its canonical sha256 and this script's sha256,
runs prereg3.validate_prereg, checks the runner and scorer carry the declared recipe, runs
prereg3.blocking_looks as a dry run and reports readiness; nothing is written. ``--write``
registers it once (prereg3.write_prereg; the lead only). From the repository root:

    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fw_replay/prereg_fw.py --validate \\
        --run-code <7-char commit holding hydrophysics/twin/replay.py>
    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fw_replay/prereg_fw.py --write \\
        --run-code <commit> --expect-self-sha <sha256 of this file as reviewed>
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(HERE))

import score_fw as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable

PREREG_ID = S.PREREG_ID
X = S.X
SCORER = f"{X}/score_fw.py"
RUNNER = f"{X}/run_fw.sh"
FROZEN = (SCORER, RUNNER)
TESTS = f"{X}/tests/test_fw_replay.py"
REPORT = f"results/dataprep/reports/{PREREG_ID}/report.json"
LOGDIR = S.LOGDIR
G = "results/twin_runs/stage3_datum_gate"
PAGE_THETAS = (f"--theta {G}/stage3_theta.json --theta {G}/stage3_fold_thetas.json "
               f"--theta {G}/stage3_posterior_tn.json")
ONE_THETA = f"--theta {G}/stage3_theta.json"
VEP = (f"{G}/coupled_leveling/vep_zonal_leveling.json,"
       f"{G}/coupled_leveling_tau30/vep_zonal_leveling.json")
SCENARIOS = ("cut30:irrigation=0.7@2026-01", "retire_aqua:aquaculture=0@2026-01",
             "cut30_lift:irrigation=0.7@2026-01..2029-12")
RF = "chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv"
RF_SHA256 = "ef5a525f5ea51697af81f2a9cd4a0fce8cad074f7cfe10121920b8e4f6a63714"
COMMON_SH = ("--horizon 120 --apex-hold calibrated --column-heads free "
             "--column-hpc0-fast-days 365 --rf-timeseries $RF --device cuda --compile-matvec")
COMMON = COMMON_SH.replace("$RF", RF)
F_PAGE = ("--gain 1.0 --ic-members 1 --ic-sigma 0.5 --seed 0 --update-method innovation "
          "--restart-taper-km 5 --save-members none")
F_RED = ("--gain 1.0 --ic-members 0 --update-method innovation --restart-taper-km 5 "
         "--save-members none")
RAIN_MANIFEST = "results/dataprep/twin_rain/preorigin/v1/MANIFEST.json"
X5_RECORD_LOW = "results/dataprep/reports/X5.record_low_tf.v1/report.json"
COUNTY_MAP = "results/dataprep/zone_county_map/v2/zone_county_map.npz"
PREGATE_LOG = f"{S.LOGDIR}/design_time/pregate.jsonl"
# every file the forward reads at the page theta, sha256 as declared by FS.start_invariance.v1
# (re-hashed at --validate and by prereg3 at --write)
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
MODEL_FILES = {
    f"{G}/stage3_theta.json": "036a7a17dcbd383fba9eaab6a12e716ab0cdffcdae0385adaf5349a57c64de77",
    f"{G}/stage3_fold_thetas.json":
        "c70632858febe2bbe57f5fe4f17646e66c64a56812699f952cf505fcdf498a89",
    f"{G}/stage3_posterior_tn.json":
        "5b194e1e7bde28b72c18d977f0321e20e12aca411b14ae2ef408d749018ec601",
    f"{G}/coupled_leveling/vep_zonal_leveling.json":
        "d290c59209f758389b7133c3f5b9aadcc71b42ad55f9089d56645fafc547b162",
    f"{G}/coupled_leveling_tau30/vep_zonal_leveling.json":
        "7f799b590fb8e82043204424aa9af001206bb6a542415197606baf08d8518d81",
    S.REF: S.REF_SHA256,
}
WELLS_DIR = "AMP_V2/data/wells"
INPUT_DIGESTS = [{"path": WELLS_DIR, "glob": "*.parquet", "kind": S.DIGEST_KINDS[0],
                  "n_files": 344,
                  "sha256": "614d5c74c1bb622d225fb9d03fcfa449d10fd863e0aea2f0575bb941384c28e2"}]
RUNNER_LINES = (
    f"ID={PREREG_ID}", "DECL=results/dataprep/preregistration/$ID.json",
    "R=results/twin_runs", "G=$R/stage3_datum_gate", "LOG=$R/fw_replay",
    "OUTD=results/twin_forward/fw_replay",
    "REPORT=results/dataprep/reports/$ID/report.json",
    'PAGE_THETAS="--theta $G/stage3_theta.json --theta $G/stage3_fold_thetas.json '
    '--theta $G/stage3_posterior_tn.json"',
    'ONE_THETA="--theta $G/stage3_theta.json"',
    'VEP="$G/coupled_leveling/vep_zonal_leveling.json,'
    '$G/coupled_leveling_tau30/vep_zonal_leveling.json"',
    "SCENARIOS=(" + " ".join(f"'{s}'" for s in SCENARIOS) + ")",
    f"RF={RF}", f'COMMON="{COMMON_SH}"', f'F_PAGE="{F_PAGE}"', f'F_RED="{F_RED}"',
    "YEARS=(" + " ".join(str(y) for y in S.SEQ_STARTS) + ")",
    "PREGATE_LOG=$LOG/design_time/pregate.jsonl")
RUNNER_FRAGMENTS = (
    'PYTHONPATH="$WT" "$HPY" -P -m hydrophysics.twin.replay $(arm_flags "$a")',
    '--vep-json "$VEP" --out "$out" > "$sealed" 2> "$err"',
    'WT="$HOME/HydroPhysicsAI-fwrun-$CODE"', "CODE=$(decl_get run_code)",
    "FIELDS=$(decl_get replay_fields)", "[ \"$a\" = C ] || need_gate_and_repro",
    "no_failed_repro", 'RW20[12][0-9]) echo "--projection-forcing replay:$(seq_of "${a:2}") '
    '--replay-pumping climatology --replay-fields energy+weather -- $ONE_THETA $F_RED"')


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"


def frozen_code() -> dict:
    return {p: sha256(REPO / p) for p in FROZEN}


def canonical_sha(d: dict) -> str:
    return hashlib.sha256((json.dumps(d, sort_keys=True, ensure_ascii=False, indent=1)
                           + "\n").encode()).hexdigest()


def engine_code(run_code: str | None) -> dict:
    """sha256 of the engine files at the run code (``git show``), or of the working files
    when no code is given (drafting)."""
    out = {}
    for name, path in S.ENGINE.items():
        if run_code is None:
            out[name] = sha256(REPO / path)
            continue
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"{run_code}:{path}"],
                              capture_output=True, check=True).stdout
        out[name] = hashlib.sha256(blob).hexdigest()
    return out


RAIN_USES = ("forcing_frozen", "design_time_evidence")


def replay_fields(repo: Path = REPO) -> tuple[str, str]:
    """('energy+weather', why) when a pre-origin rain manifest for the forward's rain input
    allows forcing_frozen or design_time_evidence (the plan's condition), else ('energy',
    why)."""
    p = repo / RAIN_MANIFEST
    if not p.exists():
        return "energy", f"no rain manifest at {RAIN_MANIFEST} at registration"
    m = json.loads(p.read_text())
    ins = {r.get("path"): r.get("sha256") for r in m.get("inputs", [])}
    uses = sorted(set(RAIN_USES) & set(m.get("allowed_uses") or []))
    if not uses:
        return "energy", f"{RAIN_MANIFEST} allows neither {' nor '.join(RAIN_USES)}"
    if ins.get(RF) != RF_SHA256:
        return "energy", f"{RAIN_MANIFEST} does not name {RF} at sha256 {RF_SHA256[:8]}"
    if m.get("fit_window_end", "9999-99") > "2022-12":
        return "energy", f"{RAIN_MANIFEST} fit_window_end is after the origin"
    return "energy+weather", f"{RAIN_MANIFEST} allows {', '.join(uses)} for {RF}"


def command(arm: str, fields: str, wt: str = "$HOME/HydroPhysicsAI-fwrun-<code>") -> str:
    spec = S.arm_spec(arm, fields)
    if spec["replay"] is None:
        rflags = "--projection-forcing climatology"
    else:
        rflags = (f"--projection-forcing {spec['projection_forcing']} --replay-pumping "
                  f"{spec['replay']['pumping']} --replay-fields {spec['replay']['fields']}")
    th, fl = (PAGE_THETAS, F_PAGE) if spec["n_members"] == 36 else (ONE_THETA, F_RED)
    scen = "".join(f" --scenario '{s}'" for s in SCENARIOS)
    return (f"PYTHONPATH={wt} $HOME/miniforge3/envs/hydro/bin/python -P -m "
            f"hydrophysics.twin.replay {rflags} -- {th} {fl}{scen} {COMMON} --vep-json {VEP} "
            f"--out {S.OUT_STEM.format(arm=arm)}")


def pregate_record(run_code: str | None, repo: Path = REPO) -> dict | None:
    """The last design-time forcing pregate logged at ``run_code`` (run_fw.sh pregate), or
    None. It printed booleans only (the gate reads forcing, no outcome)."""
    p = repo / PREGATE_LOG
    if run_code is None or not p.exists():
        return None
    recs = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    recs = [r for r in recs if r.get("event") == "pregate" and r.get("code") == run_code]
    return recs[-1] if recs else None


def county_map_record(repo: Path = REPO) -> dict:
    return {"path": COUNTY_MAP, "sha256": sha256(repo / COUNTY_MAP),
            "allowed_uses_basis": (
                "zone_county_map v2 allows model_design and after_the_fact_analysis; here it "
                "only aggregates model output (no observation) for report-only county tables "
                "after the forwards ran, never feeding a model or the wording rule")}


DISCLOSED = {
    "page_statement": (
        "C15 'avoided subsidence all elastic' and the evidence record (section 13: 'a dry "
        "year that drew heads below that margin would add inelastic loading that a cut "
        "could avoid; that case is not in the page's runs') rest on the page run under "
        "month-of-year climatology, where no cell sets a new head minimum after 2026-01 "
        "(margin 0.24 m at the 5th percentile, 0.03 m at its smallest; page_posthoc). The "
        "drafter knew these published numbers; no replay forward exists"),
    "x5_record_low": (
        "X5.record_low_tf.v1 (design-time) did not meet its declared success: the rings did "
        "not show the running-minimum response shape. It is cited beside any 'inelastic "
        "present' wording; it is evidence about the aquifer, this declaration is not"),
    "step_c_not_cited": (
        "step C's positive year-to-year timing (slope 0.22) scores a mostly elastic "
        "partition and is not evidence for a preconsolidation channel; it is not cited"),
    "rain_input": (
        f"the page theta was fitted on the vendor rain cache {RF} (sha256 {RF_SHA256[:8]}); "
        "it is the forward's rain input here as in every page forward (FS.start_invariance."
        "v1). The cache zero-fills gauge rf11 for all of 2021-22 and rf15 in 2015 H2 "
        "(CWA station inventory note): a replayed 2015, 2021 or 2022 carries that "
        "artefact, exactly as the hindcast and the calibration did. Without a pre-origin "
        "rain manifest allowing forcing_frozen or design_time_evidence the replay keeps "
        "recharge and canal water at "
        "climatology and replays the pumping only (recipe.replay_fields 'energy')"),
    "canal": (
        "the page theta has no --sw-recharge: the page model carries no canal deliveries, "
        "so the canal replay of the engine is inactive here (it is tested on synthetic "
        "inputs)"),
    "engine": (
        "the engine is a new opt-in module (hydrophysics/twin/replay.py) that swaps five "
        "module-level names of an unchanged forward.py inside a context manager; forward.py, "
        "scenario.py and compaction.py are byte-identical to the code that reproduced the "
        "page run in FS.start_invariance.v1 (4c2723e; sha256 f073eccc, 3efa3f2a, cc4aacf0). "
        "The registered scorers of N2 and N3S-L keep running from their pinned checkout"),
    "county_map": (
        f"the capture aggregates model output by county with {COUNTY_MAP} (allowed uses "
        "model_design, after_the_fact_analysis): an after-the-fact, report-only aggregate of "
        "model output, never fed back into a model and not used by the wording rule; the "
        "capture records the map's source and sha256 and the scorer refuses a split built "
        "from any other map (a capture without a split is kept and the reason reported)"),
    "pumping_steps": (
        "the detrend is a straight line per class over 2012-22; a step in the record (e.g. "
        "the pump census frozen in mid-2016) is not removed and replays as an anomaly year. "
        "The capture records Y_c(y)/L_c(y) per class and source year, and the report shows it "
        "beside the weather-only (RW) and pumping arms, so a crossing can be attributed. The "
        "gated wordings say 'replayed forcing years', never 'dry year'"),
    "wording_ruling": (
        "the judge's rank 11 lists needs_user_ruling for the C15/C25/C27 ledger wording; the "
        "2026-11-01b rulings (R7.*) hold none. Registration does not wait for it, but the "
        "page step does: the filled wordings (in particular C25's 'Y cm of lasting benefit', "
        "a fan avoided-subsidence number after the 2026-10-29 narrowing of avoided-"
        "subsidence headlines) go to the user before any page edit"),
}


def doc(run_code: str | None = None, repo: Path = REPO) -> dict:
    frozen = frozen_code()
    fields, why = replay_fields(repo)
    eng = engine_code(run_code)
    weather = fields == "energy+weather"
    cmds = {a: command(a, fields) for a in ("C", "W2015", "RD2015", "RR2015")
            + (("RW2015",) if weather else ())}
    pre = pregate_record(run_code, repo)
    ph = S.PLACEHOLDER_FILL
    w_in = S.wording("inelastic_present", ph, fields)
    w_in_none = S.wording("inelastic_present", {**ph, "years": ""}, fields)
    w_me = S.wording("mostly_elastic", ph, fields)
    w_ae = S.wording("mostly_elastic", {**ph, "all_zero": True}, fields)
    extra_inputs = [X5_RECORD_LOW, COUNTY_MAP] + ([RAIN_MANIFEST] if fields ==
                                                  "energy+weather" else [])
    return {
        "id": PREREG_ID, "wp": "FW", "kind": "declaration",
        "title": "Weather replay: under the 2012-22 forcing years replayed in order instead of "
                 "mean climatology, is at least 10 % of the page model's cut30 avoided "
                 "subsidence by Dec 2032 inelastic (permanent)? (median over the 11 cyclic "
                 "orders of 2012..2022; model-only, moves no claim status)",
        "priors": (
            "Model-only. The column is threshold-nonlinear: inelastic strain loads only while "
            "the driver head is below its running minimum. Under mean forcing no cell crosses "
            "after 2026-01, so the cut's avoided subsidence is all elastic. The replayed years "
            "include the public drought years 2015 and 2021 (an expectation from the public "
            "drought record; no hindcast new-minimum count or head was read), so replayed "
            "heads may fall below the running minimum and the baseline may cross where the "
            "cut does not. The mid zone's Skv "
            "is about 5x Ske with tau 3960 d, so much of a dry-year load lands after 2032; the "
            "distal zone (tau 57 d) responds within the year but carries the page's inflated "
            "distal column (Ske/Skv 0.63, RC2). Expected median share 0.05-0.20; the "
            "probability of >= 0.10 is about 0.5. Either outcome is an honest wording change."),
        "disclosed": DISCLOSED,
        "arms": {
            "C": "climatology (the page recipe through the engine's climatology path, "
                 "bit-identical to forward.py); reference; reproduces the page run",
            "W2012..W2022": "PRIMARY: replay of the cyclic order of 2012..2022 starting at "
                            "that year (2023 replays the start year, 2024 the next, ...; each "
                            "projects 10 of the 11 years), detrended pumping, the page "
                            "ensemble (18 parameter sets x 2 initial fields)",
            "RD2012..RD2022": "report-only: the same sequences, detrended pumping, the page "
                              "theta alone with --ic-members 0",
            "RR2012..RR2022": "report-only: raw (metered, not detrended) pumping, the page "
                              "theta alone with --ic-members 0; RR against RD is the raw-"
                              "pumping arm",
            "RW2012..RW2022": ("report-only, 'energy+weather' mode only: weather alone "
                               "(--replay-pumping climatology), the page theta alone with "
                               "--ic-members 0; RW against RD attributes a crossing to "
                               "weather or to pumping anomalies"
                               + ("" if weather else "; NOT RUN in this declaration "
                                  "(pumping-only mode: nothing to replay)")),
            "not_run": "SC (rank 17) is not built at drafting and is not an arm; tau30 is the "
                       "second rheology of every arm (report-only); no forecast-mode arm"},
        "scenarios": {
            "baseline": "no change",
            "cut30": f"'{SCENARIOS[0]}' (the page's)",
            "retire_aqua": f"'{SCENARIOS[1]}' (the page's)",
            "cut30_lift": f"'{SCENARIOS[2]}': the cut from 2026-01 through 2029-12, lifted "
                          "2030-01 (replay.WindowedScenario)"},
        "recipe": {
            "run_code": run_code or "<to be given at --write>",
            "engine_code_sha256": eng,
            "replay_fields": fields,
            "replay_fields_reason": why,
            "arms_run": list(S.arms(fields)),
            "county_map": county_map_record(repo),
            "sequences": {f"W{y}": S.years_text(S.cyclic(y)) for y in S.SEQ_STARTS},
            "pumping_detrend": (
                "per policy class, Y_c(y) = fan-total energy of calendar year y (cells x "
                "months of the twin's own E_by_class); OLS line L_c(y) = a + b (y - 2017) over "
                "2012..2022 (a = mean Y_c); g_c(y) = L_c(y) / a (mean 1); a replayed month "
                "from source year y is E_c(y, m) / g_c(y); a zero class keeps g = 1; any "
                f"g < {0.25} refuses the arm (engineering); recharge (rain - ET0) and canal "
                "water are never detrended; the capture records Y_c(y)/L_c(y) per class and "
                "source year (the replayed anomaly) for the report"),
            "seed": "0 (--seed 0; the initial-field noise is drawn once per run from it, so "
                    "every arm's members share it)",
            **{f"forward_{a}": c for a, c in cmds.items()},
            "forward_all_arms": "the same command per arm with the arm's sequence (runner "
                                "arm_flags; DRY=1 prints each)",
            "runner": f"bash {RUNNER} gate, forward C, repro, launch primary, launch reduced"
                      + (", launch weather" if weather else "")
                      + f", score (once); sha256 {frozen[RUNNER]}",
            "scorer": f"{SCORER} (sha256 {frozen[SCORER]})",
            "binding": "the runner and the scorer refuse unless this declaration is registered "
                       "and intact, the scorer and runner hash to frozen_code, the worktree "
                       "is at run_code and its engine files hash to engine_code_sha256, and "
                       "every capture records that engine code; report.json records this "
                       "declaration's sha256",
            "gpu": ("C and W*: 18 sets x 2 initial fields x 4 scenarios x 2 rollouts (column-"
                    "heads free), about 45 min each (the page run, 3 scenarios, took 34 min "
                    "under MPS); RD/RR/RW about 5 min each; about 10-11 GPU-h in all"
                    + (" plus about 1 GPU-h for RW" if weather else "")
                    + ", about 4-4.5 h wall (C alone first, then 3 sessions under MPS, "
                    "2.86x). Over the plan's 4-7 GPU-h because every arm keeps the page "
                    "ensemble's 36 members and the lift scenario. Cuts the lead may make "
                    "BEFORE registration only (each changes this recipe): drop RD and RR "
                    "(about 2 GPU-h; loses the raw-pumping comparison), drop cut30_lift "
                    "(about a quarter of every C/W arm, about 2.5 GPU-h; loses the lift "
                    "test), or run W with --ic-members 0 (18 runs, about half of W, about 4 "
                    "GPU-h; leaves the page ensemble). The runner refuses below 5000 MiB free; "
                    "no twin-module edit in the run worktree while jobs run (code freeze)")},
        "frozen_code": frozen,
        "zones": ("the compaction column's three zones (proximal, mid, distal; the page "
                  "column's zone map) and the county split of the twin grid "
                  f"({COUNTY_MAP}: Changhua vs the rest); report aggregates only"),
        "burn_in": "unchanged (origin 2022-12; the cut starts 2026-01)",
        "folds": NA("model-only forward arms of one frozen calibration; no folds"),
        "gate_scores": NA("one wording rule on one median; no weighted score"),
        "primary_model": ("the page model: theta stage3_datum_gate (036a7a17) with its 17 fold "
                          "and posterior sets, page column coupled_leveling d290c592, the page "
                          "run's flags (datum_gate_innov), --column-heads free"),
        "inputs": ([{"path": p, "sha256": frozen[p]} for p in FROZEN]
                   + [{"path": S.ENGINE[n], "sha256": h} for n, h in eng.items()]
                   + [{"path": p, "sha256": h} for p, h in MODEL_FILES.items()]
                   + [{"path": p, "sha256": h} for p, h in FORWARD_INPUTS.items()]
                   + extra_inputs + [f"{X}/prereg_fw.py", TESTS, f"{X}/manifest_rain.py"]),
        "input_digests": INPUT_DIGESTS,
        "tests": NA("model-only declaration about forward runs of one model; no observation "
                    "is scored, no fresh unit exists and none is consumed"),
        "fresh_units": NA("no observation is scored"),
        "thresholds": {
            "decomposition": (
                "the column's recurrence split exactly (replay.vep_parts; the parts' sum "
                "equals the column bit for bit): elastic = Ske (h0 - h_t); inelastic = the "
                "relaxed eps_i (plus the aquitard term if a column had one; the page column "
                "has none); driver h = layer-mean head of the --column-heads free projection, "
                "as the forward computes it. Member-mean parts at each December"),
            "primary_statistic": (
                "for each W arm k: share_k = I_k / T_k, with T_k the fan mean over all active "
                "cells of (baseline - cut30) total compaction at Dec 2032 and I_k the same for "
                f"the inelastic part, rheology coupled_leveling; share_k is not evaluable if "
                f"T_k < {S.MIN_AVOIDED_M} m. The statistic is the median (numpy.median) over "
                f"the evaluable arms; fewer than {S.MIN_EVALUABLE} evaluable arms -> "
                "'not_evaluable'"),
            "wording_rule": (f"median >= {S.SHARE_THRESHOLD} (inclusive, unrounded) -> case "
                             "'inelastic_present'; otherwise 'mostly_elastic', worded 'All "
                             f"elastic' only when every evaluable |share_k| < "
                             f"{S.ALL_ELASTIC_ABS} (0.0 % at one decimal), else 'Mostly "
                             "elastic (median X %, below the 10 % rule)'"),
            "crossings": (
                "a cell crosses preconsolidation in a member run when its driver sets a new "
                "running minimum (the column's below > 0) in a projected month; reported per "
                "arm for baseline and cut30, after 2022-12 and from 2026-01, as cells crossing "
                "in at least one run and in at least half the runs (fan, by column zone and "
                "by county), plus cells whose baseline "
                "crosses from 2026 in at least half the runs and whose cut30 does not; "
                "report-only: baseline new-minimum cell-months 2023-32 by replayed source "
                "year"),
            "source_years_for_C27": (
                f"per W arm, run, cell and projected year from {S.FROM_YEAR} (the cut's "
                "start): the baseline sets a new head minimum and cut30 does not; these "
                "cell-run-years are counted by the source year the projected year replayed, "
                f"and C27 lists the years holding >= {S.SOURCE_YEAR_MIN_SHARE} of the count, "
                "largest first (none: C27 says the cut makes crossings shallower rather than "
                "preventing them)"),
            "lift": ("cut30_lift: avoided subsidence (total and inelastic) at Dec 2029 (when "
                     "lifted) and Dec 2032 (three years later), and their ratio (remaining "
                     "fraction), for the fan, each column zone and each county (the cut30 "
                     "share's masks); report-only"),
            "engineering_checks_before_any_value_is_read": [
                "step 0: registration intact; frozen code; declared inputs and the head-well "
                "digest unchanged; every npz and capture is its arm's last successful output; "
                "one code; a passing forcing gate and a passing C reproduction logged",
                "gate (before any forward but C may use it): replay of 2012..2022 from "
                "2012-01 equals the record's energy by class, recharge (and canal water if "
                "present) exactly; detrended x factor equals the record (rel 1e-12); every "
                "factor >= 0.25; and the pipeline layer: the patched forward.future_forcing "
                "(baseline scenario, per class stacked and summed, rain_scale 1, torch "
                "conversion) and forward.future_sw on the real inputs, started at 2012-01, "
                "return the record exactly (detrended: x factor, rel 1e-12). A design-time "
                "dry run of this gate (run_fw.sh pregate <code>, booleans only) is required "
                "before registration so a factor refusal cannot first appear after it",
                f"step 1: scenario strings, dates 2012-01..2032-12, origin, members, options, "
                f"rheology labels, replay record (sequence, pumping, fields), engine code; "
                f"capture parts total vs the column <= {S.PARTS_TOL_M} m and vs the forward's "
                f"own subsidence by rheology <= {S.CAPTURE_TOL_M} m at every December",
                f"step 2: C's heads and subsidence (baseline, cut30, retire_aqua) equal the "
                f"page run datum_gate_innov (sha256 {S.REF_SHA256[:8]}) within "
                f"{S.REF_TOL_HEAD_M} m / {S.REF_TOL_SUBS_M} m at every cell and month; only "
                "these maxima are printed"],
            "reported_not_gating": [
                "every sequence's share, I_k and T_k; the median, min and max",
                "shares by column zone (proximal, mid, distal; the distal share inherits RC2's "
                "inflated distal column) and by county (Changhua carries no pumping in the "
                "page model)",
                "retire_aqua's share; the tau30 column's share",
                "C's share and crossings (mean-forcing reference; the page says no crossing "
                "after 2026-01)",
                "RD vs RR medians (the raw-pumping arm); RD vs RW (weather only, "
                "'energy+weather' mode) shares and prevented-crossing source years; the "
                "replayed pumping anomaly Y_c/L_c per class and source year",
                "the lift test; the forcing spread of baseline subsidence 2023-32 across "
                "the 11 orders (ledger and README only, 'forcing spread, model-internal, not "
                "an interval')",
                "the sampling note: the 11 orders reuse the same 11 years; their range is not "
                "an interval and not 11 independent draws"]},
        "evidence_and_ledger_mapping": {
            "scope": ("Model-only: no claim status moves (C15 stays L, C25 N, C27 N); the "
                      "wording follows the ledger (user ruling 2026-10-29). The weather spread "
                      "stays off the page tile: ledger and README Limits only, unless the user "
                      "rules otherwise. Step C is not cited."),
            "replay_mode_in_every_text": (
                f"recipe.replay_fields is '{fields}': every wording names it "
                f"('{S.under_text(fields)}')"
                + ("" if weather else "; in this pumping-only mode no wording says weather or "
                   "dry year, and the closed-caveat text reads 'closed for pumping anomalies "
                   "only; a rain-year replay was not run'")),
            "case_inelastic_present": w_in,
            "case_inelastic_present_if_no_source_year": {"C27": w_in_none["C27"]},
            "case_mostly_elastic": w_me,
            "case_mostly_elastic_if_all_shares_zero": w_ae,
            "fill": "X = 100 x the median share (1 decimal), min-max over the evaluable "
                    "sequences (n of 11), Y = 100 x the median I_k in cm (1 decimal), <years> "
                    "the C27 source years; the texts above are rendered by score_fw.wording "
                    "with placeholders and report.json carries the same function's filled "
                    "texts; TWIN_EVIDENCE_s13 replaces section 13's 'that case is not in the "
                    "page's runs' and the matching MODEL_CARD caveat",
            "user_ruling_before_page": DISCLOSED["wording_ruling"],
            "on_not_evaluable": "a ledger note 'FW.weather_replay.v1 not evaluable (n of 11 "
                                "sequences)'; nothing else changes",
            "feeds": "rank 12 (FV.decidability.v2_postfailure) may reuse the W arms' replay "
                     "engine (its own declaration)"},
        "attempt_policy": {
            "scored_runs": "one forward per arm and one scoring run; the first scorer run that "
                           "passes steps 0-2 writes report.json (mode 'x') and is final",
            "order": "gate, forward C, repro (exit 3 = C does not reproduce the page run: "
                     "nothing more runs under v1), then the W, RD and RR arms in any order "
                     "(the runner refuses them before a passing gate and repro), then score",
            "values_written_before_scoring": (
                "forward.py prints a '=== forward summary ===' block, a rheology-axis block "
                "and per-member hindcast R2 lines, and may print a 2012-22 leveling skill "
                "line; it writes <arm>.npz, .summary.csv, .members.csv, and the engine "
                f"writes <arm>.fw.npz under {S.OUT_STEM.format(arm='')[:-1]}. The runner seals "
                f"stdout to {LOGDIR}/sealed/<arm>_forward.stdout.SEALED and logs the sha256 of "
                "every output; nobody opens them before report.json exists except the "
                "reproduction maxima printed by `repro`; stderr may be read after a crash"),
            "engineering_retries": (
                "free before report.json exists (crash, OOM, missing file, a scorer failure "
                f"in steps 0-2); each is logged; `bash {RUNNER} archive '<reason>' [arm ...|"
                "gate]` moves outputs unopened; a fix to the scorer, runner, engine or recipe "
                "needs a v2 that discloses the attempts"),
            "pre_declared_alternatives": [
                "replay_fields 'energy' (pumping only) when no rain manifest exists at "
                "registration; fixed in this document, never switched after a look"],
            "failed_reproduction": (
                "a repro exit 3 ends v1 for good: the runner refuses every further forward "
                "(C included), repro, gate and launch once any repro exit 3 is logged, and the "
                "scorer refuses to score; archiving C and rerunning does not reopen it"),
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included (attempts.jsonl and the report)"},
        "interpretation": (
            "A share >= 0.10 says that, in this model, mean forcing hides a permanent part of "
            "a cut's benefit that dry years expose; below 0.10 says the cut's benefit is "
            "elastic also under the record's own weather, closing the mean-forcing caveat. "
            "Either is a statement about the model's form (Q0's column on free-run heads), "
            "not about the aquifer: the rings did not show the running-minimum response "
            "(X5.record_low_tf.v1). The spread across orders is a forcing spread of one "
            "model, not a predictive interval."),
        "rule_risks_disclosure": [
            "the drafter read the page theta's meta keys and flags (sw_recharge, eta_classes, "
            "extra_abstraction, zone_county_split, meter_filter, boundaries); the printout "
            "also showed the theta's fitted per-well datum offsets (model parameters fitted "
            "on pre-origin heads, stored in the theta); nothing was computed from them",
            "the drafter read the key structure of the two column JSONs (three zones, VEP "
            "keys only, no blend) and the exit codes of FS.start_invariance.v1's attempts; "
            "no head, leveling, ring, pile or GNSS value and no forward output was read",
            "the forward reads the 2012-22 inputs (forcing and the pre-origin heads it "
            "restarts from) as every page forward does; its stdout is sealed",
            "the gate loads the same inputs and prints booleans only",
            ("the design-time forcing pregate (run_fw.sh pregate, booleans only, forcing "
             "arrays only) " + ("has not been run at drafting" if pre is None else
                                f"ran at {pre.get('code')} on {pre.get('time')}, exit "
                                f"{pre.get('rc')}, record sha256 "
                                f"{str(pre.get('gate_sha256'))[:16]}")),
            "the priors' drought years 2015 and 2021 are from the public drought record, not "
            "from a model output or head read"],
        "no_prior_look_statement": (
            "No replay forward of any model exists; no inelastic share, crossing count under "
            "replay, or replayed-forcing statistic has been computed. The engine was tested "
            "on synthetic inputs only."),
        "written_by": f"{X}/prereg_fw.py",
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
    return out


def readiness(run_code: str | None, repo: Path = REPO) -> list[str]:
    out = []
    d = doc(run_code, repo)
    if run_code is None:
        out.append("--run-code not given (the commit holding hydrophysics/twin/replay.py)")
    elif not re.fullmatch(r"[0-9a-f]{7}", run_code):
        out.append(f"--run-code {run_code!r}: expected 7 hex characters")
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (repo / p).exists():
            out.append(f"input {p} is missing")
        elif isinstance(rec, dict) and sha256(repo / p) != rec["sha256"]:
            out.append(f"input {p} changed since drafting (or differs from the run code)")
    try:
        S.check_input_digests(d, repo)
    except S.EngineeringError as err:
        out.append(str(err))
    if run_code and re.fullmatch(r"[0-9a-f]{7}", run_code):
        pre = pregate_record(run_code, repo)
        if pre is None or int(pre.get("rc", -1)) != 0:
            out.append(f"no passing design-time forcing pregate at {run_code} (bash {RUNNER} "
                       f"pregate {run_code}; a trend-factor refusal after registration would "
                       "end v1)")
    if (repo / f"results/dataprep/preregistration/{PREREG_ID}.json").exists():
        out.append(f"{PREREG_ID} is already registered")
    od = repo / S.OUT_STEM.format(arm="").rstrip("/")
    if od.exists() and any(od.iterdir()):
        out.append(f"{od.relative_to(repo)} holds outputs before the declaration")
    if (repo / REPORT).exists():
        out.append(f"{REPORT} exists")
    if run_code:
        wt = Path.home() / f"HydroPhysicsAI-fwrun-{run_code}"
        try:
            head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return out + [f"worktree $HOME/HydroPhysicsAI-fwrun-{run_code} missing (git "
                          f"worktree add --detach $HOME/HydroPhysicsAI-fwrun-{run_code} "
                          f"{run_code})"]
        if head != run_code:
            out.append(f"worktree $HOME/HydroPhysicsAI-fwrun-{run_code} is at {head}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="dry run: print, check, write nothing")
    g.add_argument("--write", action="store_true", help="register once (the lead only)")
    ap.add_argument("--run-code", default=None, help="7-char commit holding the engine")
    ap.add_argument("--expect-self-sha", default=None,
                    help="sha256 of this script as reviewed; required with --write")
    ap.add_argument("--quiet", action="store_true", help="do not print the document")
    a = ap.parse_args(argv)
    try:
        d = doc(a.run_code)
    except subprocess.CalledProcessError as err:
        print(f"REFUSED: git show at {a.run_code} failed ({err})")
        return 1
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", canonical_sha(d))
    print("replay fields:", d["recipe"]["replay_fields"], "-", d["recipe"]["replay_fields_reason"])
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    probs = runner_problems() + scorer_problems()
    print("recipe/binding checks:", probs or "ok")
    block = prereg3.blocking_looks([], None, test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none (declaration with no tests and no "
          "fresh units)")
    pending = readiness(a.run_code)
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
