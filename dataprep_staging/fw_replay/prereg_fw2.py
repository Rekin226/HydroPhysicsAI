"""Declaration FW.weather_replay.v2, kind 'declaration', before any v2 replay forward exists:
does a dry year expose the compaction column's permanent (inelastic) channel in the page
model? Model-only; 'energy+weather' mode; no observation of any family is read or scored.

v2 is FW.weather_replay.v1's recipe with the weather replayed (user ruling R7.fw_both, rulings
2026-11-01f). v1 was registered pumping-only by the rain manifest's absolute-path bug and runs
as registered (R7.fw_v1_pumping, R7.fw_both): it feeds FV.decidability.v2_postfailure's
forcing-year reading, and v1 is NOT withdrawn. v2 is a separate, concurrent declaration in
separate files (prereg_fw2.py, score_fw2.py, run_fw2.sh, manifest_rain2.py) and directories
(results/twin_forward/fw_replay_v2, results/twin_runs/fw_replay_v2, reports/FW.weather_replay.
v2); v1's files are imported read-only and never changed.

The replay mode is fixed at 'energy+weather' by the ruling; readiness refuses to register
unless the pre-origin rain manifest v2 (manifest_rain2.py, a lead step) names the forward's
rain input by its repo-relative path at its sha256 and allows forcing_frozen or
design_time_evidence. An absolute or home path in that manifest is refused with a message,
never silently mismatched (v1's failure).

``--validate`` (dry run) prints the document, its canonical sha256 and this script's sha256,
runs prereg3.validate_prereg, checks the runner and scorer carry the declared recipe, runs
prereg3.blocking_looks as a dry run and reports readiness; nothing is written. ``--write``
registers it once (prereg3.write_prereg; the lead only). From the repository root:

    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fw_replay/prereg_fw2.py --validate \\
        --run-code <7-char commit holding hydrophysics/twin/replay.py>
    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fw_replay/prereg_fw2.py --write \\
        --run-code <commit> --expect-self-sha <sha256 of this file as reviewed>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(HERE))

import prereg_fw as P1  # noqa: E402  (v1's declaration script, read-only: shared constants)
import score_fw2 as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
S1 = S.S1

PREREG_ID = S.PREREG_ID
V1_ID = S.V1_ID
X = S.X
FIELDS = S.FIELDS
SCORER = f"{X}/score_fw2.py"
RUNNER = f"{X}/run_fw2.sh"
FROZEN = S.FROZEN
TESTS = f"{X}/tests/test_fw_replay_v2.py"
REPORT = S.REPORT
LOGDIR = S.LOGDIR
PREGATE_LOG = f"{LOGDIR}/design_time/pregate.jsonl"
WT_TEMPLATE = "$HOME/HydroPhysicsAI-fwrun-<code>"
# v1's shared recipe (one definition, imported)
G, PAGE_THETAS, ONE_THETA, VEP = P1.G, P1.PAGE_THETAS, P1.ONE_THETA, P1.VEP
SCENARIOS, RF, RF_SHA256 = P1.SCENARIOS, P1.RF, P1.RF_SHA256
COMMON_SH, COMMON, F_PAGE, F_RED = P1.COMMON_SH, P1.COMMON, P1.F_PAGE, P1.F_RED
X5_RECORD_LOW, COUNTY_MAP = P1.X5_RECORD_LOW, P1.COUNTY_MAP
FORWARD_INPUTS, MODEL_FILES, INPUT_DIGESTS = P1.FORWARD_INPUTS, P1.MODEL_FILES, P1.INPUT_DIGESTS
RAIN_USES = P1.RAIN_USES
sha256, canonical_sha, engine_code = P1.sha256, P1.canonical_sha, P1.engine_code
county_map_record = P1.county_map_record
# the rain product v2 (manifest_rain2.py; repo-relative paths)
RAIN_PRODUCT, RAIN_VERSION = "twin_rain/preorigin", "v2"
RAIN_MANIFEST = f"results/dataprep/{RAIN_PRODUCT}/{RAIN_VERSION}/MANIFEST.json"
V1_RAIN_MANIFEST = P1.RAIN_MANIFEST
# rulings
RULING_ID = "R7.fw_both"
RULING_V1 = "R7.fw_v1_pumping"
RULING_WORDING = "R7.fw_wording"
RULINGS_C = "results/dataprep/rulings/user_rulings_2026-11-01c.json"
RULINGS_E = "results/dataprep/rulings/user_rulings_2026-11-01e.json"
RULINGS_F = "results/dataprep/rulings/user_rulings_2026-11-01f.json"
RULINGS = {RULINGS_C: RULING_WORDING, RULINGS_E: RULING_V1, RULINGS_F: RULING_ID}
# v1, read-only: its registration at its sha256, and the code v2 imports from it
V1_DECL = f"results/dataprep/preregistration/{V1_ID}.json"
V1_SHA256 = "91faaf6b5e0cec54ce36acabb6685c71b2f4ab3faa39ba5355a069869ac3462b"
V1_CREATED = "2026-11-01T14:22:15.576760Z"
V1_RUN_CODE = "cebb789"
V1_FROZEN_CODE = {
    f"{X}/run_fw.sh": "e7c2c93c4d868536c2ebed2a1e065d46db817caf652a2749d98fb2e2d2f01218",
    f"{X}/score_fw.py": "b10a7ad013204de5dcd07153f2d0402f88cb03abe0268201b08edbbaea4dd060"}
V1_MANIFEST_RAIN_SHA256 = "5f1031bbf0e9bacd348d64e05b03a9c281b13b01581207963656defaaa729f98"
V1_LOG, V1_OUTD, V1_REPORT = S1.LOGDIR, S1.OUT_STEM.format(arm="").rstrip("/"), \
    f"results/dataprep/reports/{V1_ID}/report.json"
V1_CAUSE = (
    "registered pumping-only (recipe.replay_fields 'energy') by the absolute-path bug: "
    "manifest_rain.py wrote results/dataprep/twin_rain/preorigin/v1/MANIFEST.json with "
    "absolute input paths (common.paths.rel_to_repo resolved the worktree's chou-shui-data "
    "symlink to a path outside the repository), while prereg_fw.py compared the repo-relative "
    f"path {RF} at sha256 {RF_SHA256[:8]}; the file and its sha256 were correct, so the "
    "weather mode the plan intended was dropped by a string comparison, not by the data")

RUNNER_LINES = (
    f"ID={PREREG_ID}", f"FIELDS={FIELDS}", f"TMUX_PREFIX={S.TMUX_PREFIX}",
    f"SELF={RUNNER}", f"SCORER={SCORER}", f"SCORER_V1={X}/score_fw.py",
    "DECL=results/dataprep/preregistration/$ID.json",
    "R=results/twin_runs", "G=$R/stage3_datum_gate", "LOG=$R/fw_replay_v2",
    "OUTD=results/twin_forward/fw_replay_v2",
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
    '[ "$(decl_get replay_fields)" = "$FIELDS" ]', "[ \"$a\" = C ] || need_gate_and_repro",
    'for f in "$SCORER" "$SELF" "$SCORER_V1"; do',
    'tmux new-session -d -s "$s" "bash $SELF forward $which $i/$n; exec bash"',
    'for i in $(seq 1 "$1"); do echo "$TMUX_PREFIX$i"; done',
    "no_failed_repro", 'RW20[12][0-9]) echo "--projection-forcing replay:$(seq_of "${a:2}") '
    '--replay-pumping climatology --replay-fields energy+weather -- $ONE_THETA $F_RED"')


def repo_relative(path) -> bool:
    """True for a repo-relative POSIX path (not absolute, not home, no '..', not empty)."""
    t = "" if path is None else str(path)
    pp = PurePosixPath(t.replace("\\", "/"))
    return bool(t) and not pp.is_absolute() and not t.startswith("~") and \
        ".." not in pp.parts and not re.match(r"^[A-Za-z]:", t)


def rain_manifest_problems(m: dict) -> list[str]:
    """Why the rain manifest cannot carry this declaration's weather replay (empty: it can).
    The forward's rain input must be named by its repo-relative path at RF_SHA256; an
    absolute or home path is refused with a message, never matched (v1's failure was a silent
    mismatch)."""
    out = []
    recs = [r for r in m.get("inputs") or [] if isinstance(r, dict)]
    bad = [str(r.get("path")) for r in recs if not repo_relative(r.get("path"))]
    if bad:
        out.append(f"{RAIN_MANIFEST} records input paths that are not repo-relative ({bad}): "
                   "absolute and home paths are refused; rebuild the product with "
                   f"{X}/manifest_rain2.py (it writes repo-relative paths)")
    if (m.get("product"), m.get("version")) != (RAIN_PRODUCT, RAIN_VERSION):
        out.append(f"{RAIN_MANIFEST} is product {m.get('product')!r} {m.get('version')!r}, not "
                   f"{RAIN_PRODUCT!r} {RAIN_VERSION!r}")
    if not (m.get("gate") or {}).get("passed"):
        out.append(f"{RAIN_MANIFEST}: its structure gate did not pass")
    ins = {PurePosixPath(str(r["path"])).as_posix(): r.get("sha256") for r in recs
           if repo_relative(r.get("path"))}
    if RF not in ins:
        out.append(f"{RAIN_MANIFEST} does not name {RF} (repo-relative)")
    elif ins[RF] != RF_SHA256:
        out.append(f"{RAIN_MANIFEST} names {RF} at sha256 {str(ins[RF])[:8]}, not "
                   f"{RF_SHA256[:8]}")
    if not set(RAIN_USES) & set(m.get("allowed_uses") or []):
        out.append(f"{RAIN_MANIFEST} allows neither {' nor '.join(RAIN_USES)}")
    if str(m.get("fit_window_end", "9999-99")) > "2022-12":
        out.append(f"{RAIN_MANIFEST} fit_window_end is after the origin")
    return out


def rain_problems(repo: Path = REPO) -> list[str]:
    p = repo / RAIN_MANIFEST
    if not p.exists():
        return [f"no rain manifest at {RAIN_MANIFEST} (manifest_rain2.py --write)"]
    try:
        m = json.loads(p.read_text())
    except (OSError, ValueError) as err:
        return [f"{RAIN_MANIFEST} unreadable: {err}"]
    return rain_manifest_problems(m)


def replay_fields(repo: Path = REPO) -> tuple[str, str]:
    """('energy+weather', why) when the rain manifest v2 carries the weather replay, else
    ('energy', the problems): a diagnosis only. v2's recipe is 'energy+weather' by ruling and
    readiness refuses to register while this says 'energy'; there is no fallback."""
    probs = rain_problems(repo)
    if probs:
        return "energy", "; ".join(probs)
    return FIELDS, (f"ruling {RULING_ID} fixes '{FIELDS}'; {RAIN_MANIFEST} names {RF} "
                    f"(repo-relative) at sha256 {RF_SHA256[:8]} and allows "
                    f"{' or '.join(RAIN_USES)}")


def rulings_problems(repo: Path = REPO) -> list[str]:
    """The rulings this declaration rests on are present and hold the cited ids."""
    out = []
    for f, rid in RULINGS.items():
        p = repo / f
        try:
            rs = json.loads(p.read_text()).get("rulings", [])
        except (OSError, ValueError):
            out.append(f"{f} missing or unreadable (ruling {rid})")
            continue
        hit = [r for r in rs if r.get("id") == rid]
        if len(hit) != 1:
            out.append(f"{f} holds no single ruling {rid}")
        elif rid == RULING_ID and PREREG_ID not in (hit[0].get("applies_to") or []):
            out.append(f"ruling {rid} does not apply to {PREREG_ID}")
    return out


def v1_problems(repo: Path = REPO) -> list[str]:
    """v1's registration is at its sha256 and the v1 code v2 imports is byte-identical to
    v1's frozen code (v1 runs as registered; nothing of v1's is changed)."""
    out = []
    if sha256(repo / V1_DECL) != V1_SHA256:
        out.append(f"{V1_DECL} missing or not at sha256 {V1_SHA256[:16]}")
    for p, h in {**V1_FROZEN_CODE, f"{X}/manifest_rain.py": V1_MANIFEST_RAIN_SHA256}.items():
        if sha256(REPO / p) != h:
            out.append(f"{p} differs from {V1_ID}'s registered sha256 {h[:16]}: v1's files "
                       "must stay byte-identical")
    return out


def command(arm: str, wt: str = WT_TEMPLATE) -> str:
    spec = S.arm_spec(arm, FIELDS)
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
    """The last design-time forcing pregate logged at ``run_code`` by run_fw2.sh (v2's own
    log; v1's pregate is not reused), or None."""
    p = repo / PREGATE_LOG
    if run_code is None or not p.exists():
        return None
    recs = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    recs = [r for r in recs if r.get("event") == "pregate" and r.get("code") == run_code]
    return recs[-1] if recs else None


GAUGE = ("the vendor cache zero-fills gauge rf11 for all of 2021-22 and rf15 in 2015 H2 "
         "(zero-filled, not dry)")
DISCLOSED = {
    **{k: P1.DISCLOSED[k] for k in ("page_statement", "x5_record_low", "step_c_not_cited")},
    "v1_concurrent": (
        f"{V1_ID} was registered at {V1_CREATED} (registration sha256 {V1_SHA256[:16]}, run "
        f"code {V1_RUN_CODE}) and was {V1_CAUSE}. v1 runs as registered, pumping-only, under "
        f"{RULING_V1} (rulings 2026-11-01e, which superseded R7.fw_v2_weather of 2026-11-01d) "
        f"and {RULING_ID} (rulings 2026-11-01f): it feeds FV.decidability.v2_postfailure's "
        "forcing-year reading, and an inelastic channel found there is a lower bound. v1 is "
        "not withdrawn. v2 is a separate, concurrent declaration with the same recipe (arms, "
        "sequences, scenarios, flags, pumping detrend, engine code, gates, thresholds, "
        "statistic, wording function) except the replay mode ('energy+weather', which adds "
        "the RW weather-only arms v1's mode does not run), the ids, the output and log "
        "directories, the tmux session names (fwv2_1..fwv2_n) and the rain-caveat clause in "
        "the wordings. v2 runs its own pregate and gate; v1's records are not reused, and "
        "v2's runner and scorer refuse v1's directories. v1's files (prereg_fw.py, "
        "score_fw.py, run_fw.sh, manifest_rain.py, the engine) are byte-identical; v2 imports "
        "score_fw.py, prereg_fw.py and manifest_rain.py read-only and binds score_fw.py in "
        "its frozen code"),
    "rain_input": (
        f"the page theta was fitted on the vendor rain cache {RF} (sha256 {RF_SHA256[:8]}); "
        "it is the forward's rain input here as in every page forward (FS.start_invariance."
        f"v1). v2 replays the weather (recipe.replay_fields '{FIELDS}': rain-driven recharge "
        "(and canal water, inactive here) from the source years) under the pre-origin rain "
        f"manifest {RAIN_MANIFEST} (product {RAIN_PRODUCT} {RAIN_VERSION}, allowed uses "
        "forcing_frozen and design_time_evidence), which names the cache by its repo-relative "
        f"path at that sha256; product v1 ({V1_RAIN_MANIFEST}) names the same file at the same "
        "sha256 by an absolute path and is kept unchanged"),
    "gauge_caveat": (
        f"{GAUGE} (CWA station inventory note). In this mode a replayed 2015, 2021 or 2022 "
        "carries that artefact, exactly as the hindcast and the calibration did. Every "
        "weather-mode wording that cites a replayed-year (dry-year) result carries the "
        f"clause '{S.GAUGE_CAVEAT}' (score_fw2.GAUGE_CAVEAT); the not-evaluable ledger note "
        "cites no result and does not"),
    **{k: P1.DISCLOSED[k] for k in ("canal", "engine", "county_map", "pumping_steps")},
    "wording_ruling": (
        "the judge's rank 11 lists needs_user_ruling for the C15/C25/C27 ledger wording. "
        f"{RULING_WORDING} (rulings 2026-11-01c) puts FW's 'X cm of lasting benefit' in the "
        "ledger and README only (the page gets only the qualitative dry-year inelastic-channel "
        f"note under Model internals), and {RULING_ID} extends that to both declarations. "
        "Registration does not wait for anything else, but the page step does: the filled "
        "wordings of v1 and v2 go to the user side by side before any page edit"),
}


def doc(run_code: str | None = None, repo: Path = REPO) -> dict:
    frozen = {p: sha256(REPO / p) for p in FROZEN}
    got_fields, why = replay_fields(repo)
    eng = engine_code(run_code)
    cmds = {a: command(a) for a in ("C", "W2015", "RD2015", "RR2015", "RW2015")}
    pre = pregate_record(run_code, repo)
    ph = S.PLACEHOLDER_FILL
    w_in = S.wording("inelastic_present", ph)
    w_in_none = S.wording("inelastic_present", {**ph, "years": ""})
    w_me = S.wording("mostly_elastic", ph)
    w_ae = S.wording("mostly_elastic", {**ph, "all_zero": True})
    return {
        "id": PREREG_ID, "wp": "FW", "kind": "declaration",
        "title": "Weather replay (energy+weather): under the 2012-22 forcing years replayed in "
                 "order instead of mean climatology, is at least 10 % of the page model's cut30 "
                 "avoided subsidence by Dec 2032 inelastic (permanent)? (median over the 11 "
                 "cyclic orders of 2012..2022; model-only, moves no claim status; concurrent "
                 f"with the pumping-only {V1_ID})",
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
            "probability of >= 0.10 is about 0.5. Either outcome is an honest wording change. "
            f"(The priors are {V1_ID}'s; no v1 output exists to update them.)"),
        "disclosed": DISCLOSED,
        "arms": {
            "C": "climatology (the page recipe through the engine's climatology path, "
                 "bit-identical to forward.py); reference; reproduces the page run",
            "W2012..W2022": "PRIMARY: replay of the cyclic order of 2012..2022 starting at "
                            "that year (2023 replays the start year, 2024 the next, ...; each "
                            "projects 10 of the 11 years), weather and detrended pumping, the "
                            "page ensemble (18 parameter sets x 2 initial fields)",
            "RD2012..RD2022": "report-only: the same sequences, weather and detrended pumping, "
                              "the page theta alone with --ic-members 0",
            "RR2012..RR2022": "report-only: weather and raw (metered, not detrended) pumping, "
                              "the page theta alone with --ic-members 0; RR against RD is the "
                              "raw-pumping arm",
            "RW2012..RW2022": "report-only: weather alone (--replay-pumping climatology), the "
                              "page theta alone with --ic-members 0; RW against RD attributes "
                              "a crossing to weather or to pumping anomalies",
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
            "replay_fields": FIELDS,
            "replay_fields_reason": (why if got_fields == FIELDS else
                                     f"ruling {RULING_ID} fixes '{FIELDS}'; NOT READY: {why}"),
            "arms_run": list(S.arms()),
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
            "runner": f"bash {RUNNER} gate, forward C, repro, launch primary, launch reduced, "
                      f"launch weather, score (once); sha256 {frozen[RUNNER]}",
            "tmux_sessions": (f"{S.TMUX_PREFIX}1..{S.TMUX_PREFIX}n (DRY=1 bash {RUNNER} launch "
                              "<group> [n] prints them); never v1's fw1..fwn"),
            "scorer": f"{SCORER} (sha256 {frozen[SCORER]}), which imports {X}/score_fw.py "
                      f"(sha256 {frozen[f'{X}/score_fw.py']}, {V1_ID}'s frozen scorer)",
            "outputs": {"forward": S.OUT_STEM.format(arm="<arm>"), "logs": LOGDIR,
                        "report": REPORT,
                        "never": [V1_OUTD, V1_LOG, f"results/dataprep/reports/{V1_ID}"]},
            "binding": "the runner and the scorer refuse unless this declaration is registered "
                       "and intact, the scorer, runner and the imported v1 scorer hash to "
                       "frozen_code, recipe.replay_fields is 'energy+weather', the worktree "
                       "is at run_code and its engine files hash to engine_code_sha256, and "
                       "every capture records that engine code; report.json records this "
                       "declaration's sha256",
            "gpu": ("C and W*: 18 sets x 2 initial fields x 4 scenarios x 2 rollouts (column-"
                    "heads free), about 45 min each (the page run, 3 scenarios, took 34 min "
                    "under MPS); RD/RR/RW about 5 min each; about 10-11 GPU-h in all plus "
                    "about 1 GPU-h for RW, about 4-4.5 h wall (C alone first, then 3 sessions "
                    "under MPS, 2.86x). Run after v1's forwards finish (R7.fw_both); no "
                    "change to the memory check. The runner refuses below 5000 MiB free; no "
                    "twin-module edit in the run worktree while jobs run (code freeze)")},
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
                   + [{"path": V1_DECL, "sha256": V1_SHA256}]
                   + [X5_RECORD_LOW, COUNTY_MAP, RAIN_MANIFEST, *RULINGS]
                   + [f"{X}/prereg_fw2.py", f"{X}/prereg_fw.py", TESTS, f"{X}/manifest_rain2.py",
                      f"{X}/manifest_rain.py"]),
        "input_digests": INPUT_DIGESTS,
        "tests": NA("model-only declaration about forward runs of one model; no observation "
                    "is scored, no fresh unit exists and none is consumed"),
        "fresh_units": NA("no observation is scored"),
        "thresholds": {
            "same_as": f"{V1_ID} (score_fw.py constants, imported unchanged)",
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
                             f"{S1.ALL_ELASTIC_ABS} (0.0 % at one decimal), else 'Mostly "
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
                f"per W arm, run, cell and projected year from {S1.FROM_YEAR} (the cut's "
                "start): the baseline sets a new head minimum and cut30 does not; these "
                "cell-run-years are counted by the source year the projected year replayed, "
                f"and C27 lists the years holding >= {S1.SOURCE_YEAR_MIN_SHARE} of the count, "
                "largest first (none: C27 says the cut makes crossings shallower rather than "
                "preventing them)"),
            "lift": ("cut30_lift: avoided subsidence (total and inelastic) at Dec 2029 (when "
                     "lifted) and Dec 2032 (three years later), and their ratio (remaining "
                     "fraction), for the fan, each column zone and each county (the cut30 "
                     "share's masks); report-only"),
            "engineering_checks_before_any_value_is_read": [
                "step 0: registration intact; frozen code (score_fw2.py, run_fw2.sh and the "
                "imported score_fw.py); recipe.replay_fields 'energy+weather'; declared inputs "
                "and the head-well digest unchanged; every npz and capture is its arm's last "
                "successful output in v2's own attempt log; one code; a passing forcing gate "
                "and a passing C reproduction logged under v2; no path in v1's directories",
                "gate (before any forward but C may use it): replay of 2012..2022 from "
                "2012-01 equals the record's energy by class, recharge (and canal water if "
                "present) exactly; detrended x factor equals the record (rel 1e-12); every "
                "factor >= 0.25; and the pipeline layer: the patched forward.future_forcing "
                "(baseline scenario, per class stacked and summed, rain_scale 1, torch "
                "conversion) and forward.future_sw on the real inputs, started at 2012-01, "
                "return the record exactly (detrended: x factor, rel 1e-12). A design-time "
                f"dry run of this gate (run_fw2.sh pregate <code>, booleans only, logged to "
                f"{PREGATE_LOG}) is required before registration",
                f"step 1: scenario strings, dates 2012-01..2032-12, origin, members, options, "
                f"rheology labels, replay record (sequence, pumping, fields), engine code; "
                f"capture parts total vs the column <= {S1.PARTS_TOL_M} m and vs the forward's "
                f"own subsidence by rheology <= {S1.CAPTURE_TOL_M} m at every December",
                f"step 2: C's heads and subsidence (baseline, cut30, retire_aqua) equal the "
                f"page run datum_gate_innov (sha256 {S.REF_SHA256[:8]}) within "
                f"{S1.REF_TOL_HEAD_M} m / {S1.REF_TOL_SUBS_M} m at every cell and month; only "
                "these maxima are printed"],
            "reported_not_gating": [
                "every sequence's share, I_k and T_k; the median, min and max",
                "shares by column zone (proximal, mid, distal; the distal share inherits RC2's "
                "inflated distal column) and by county (Changhua carries no pumping in the "
                "page model)",
                "retire_aqua's share; the tau30 column's share",
                "C's share and crossings (mean-forcing reference; the page says no crossing "
                "after 2026-01)",
                "RD vs RR medians (the raw-pumping arm); RD vs RW (weather only) shares and "
                "prevented-crossing source years; the replayed pumping anomaly Y_c/L_c per "
                "class and source year",
                "the lift test; the forcing spread of baseline subsidence 2023-32 across "
                "the 11 orders (ledger and README only, 'forcing spread, model-internal, not "
                "an interval')",
                "the sampling note: the 11 orders reuse the same 11 years; their range is not "
                "an interval and not 11 independent draws",
                f"beside {V1_ID}'s report when both exist (the side-by-side table is the "
                "page step's, not this scorer's: v2's scorer reads none of v1's files)"]},
        "evidence_and_ledger_mapping": {
            "scope": ("Model-only: no claim status moves (C15 stays L, C25 N, C27 N); the "
                      "wording follows the ledger (user ruling 2026-10-29). The weather spread "
                      f"and every avoided-subsidence number stay off the page ({RULING_WORDING}"
                      f", {RULING_ID}): ledger and README Limits only. Step C is not cited."),
            "replay_mode_in_every_text": (
                f"recipe.replay_fields is '{FIELDS}': every wording names it "
                f"('{S.under_text(FIELDS)}') and every text citing a replayed-year result "
                f"carries the gauge caveat ('{S.GAUGE_CAVEAT}')"),
            "relation_to_v1": (
                f"{V1_ID} (pumping-only; its texts say 'pumping anomalies only' and 'a "
                "rain-year replay was not run') and v2 (weather; its texts say 'weather and "
                "detrended pumping' and carry the gauge caveat) are separate results; neither "
                "report overwrites or rewords the other, and the filled texts of both go to "
                "the user before any page edit"),
            "case_inelastic_present": w_in,
            "case_inelastic_present_if_no_source_year": {"C27": w_in_none["C27"]},
            "case_mostly_elastic": w_me,
            "case_mostly_elastic_if_all_shares_zero": w_ae,
            "fill": "X = 100 x the median share (1 decimal), min-max over the evaluable "
                    "sequences (n of 11), Y = 100 x the median I_k in cm (1 decimal), <years> "
                    "the C27 source years; the texts above are rendered by score_fw2.wording "
                    "(score_fw.wording with v2's id and the gauge caveat) with placeholders and "
                    "report.json carries the same function's filled texts; TWIN_EVIDENCE_s13 "
                    "replaces section 13's 'that case is not in the page's runs' and the "
                    "matching MODEL_CARD caveat",
            "user_ruling_before_page": DISCLOSED["wording_ruling"],
            "on_not_evaluable": f"a ledger note '{PREREG_ID} not evaluable (n of 11 "
                                "sequences)'; nothing else changes",
            "feeds": (f"nothing registered: FV.decidability.v2_postfailure reads {V1_ID}'s "
                      "report path, not v2's (R7.fw_v1_pumping)")},
        "attempt_policy": {
            "scored_runs": "one forward per arm and one scoring run; the first scorer run that "
                           "passes steps 0-2 writes report.json (mode 'x') and is final",
            "order": f"gate, forward C, repro (exit 3 = C does not reproduce the page run: "
                     f"nothing more runs under {PREREG_ID}), then the W, RD, RR and RW arms in "
                     "any order (the runner refuses them before a passing gate and repro), "
                     "then score",
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
                "needs a v3 that discloses the attempts"),
            "pre_declared_alternatives": [
                f"none for the replay mode: '{FIELDS}' is fixed by ruling {RULING_ID}; this "
                "declaration refuses to register in any other mode (v1's fallback to 'energy' "
                "without a usable rain manifest does not exist here)"],
            "failed_reproduction": (
                f"a repro exit 3 ends {PREREG_ID} for good: the runner refuses every further "
                "forward (C included), repro, gate and launch once any repro exit 3 is logged "
                "in v2's log, and the scorer refuses to score; archiving C and rerunning does "
                "not reopen it. v2 runs and reproduces its own C; a v1 repro outcome is v1's "
                "and is neither read nor reused by v2's runner or scorer"),
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
            f"model, not a predictive interval. The replayed rain carries {GAUGE}."),
        "rule_risks_disclosure": [
            "the drafter read the page theta's meta keys and flags (sw_recharge, eta_classes, "
            "extra_abstraction, zone_county_split, meter_filter, boundaries); the printout "
            "also showed the theta's fitted per-well datum offsets (model parameters fitted "
            "on pre-origin heads, stored in the theta); nothing was computed from them "
            f"({V1_ID}'s drafting, inherited)",
            "the drafter read the key structure of the two column JSONs (three zones, VEP "
            "keys only, no blend) and the exit codes of FS.start_invariance.v1's attempts; "
            "no head, leveling, ring, pile or GNSS value and no forward output was read",
            "the forward reads the 2012-22 inputs (forcing and the pre-origin heads it "
            "restarts from) as every page forward does; its stdout is sealed",
            "the gate loads the same inputs and prints booleans only",
            ("v2's design-time forcing pregate (run_fw2.sh pregate, booleans only, forcing "
             "arrays only) " + ("has not been run at drafting" if pre is None else
                                f"ran at {pre.get('code')} on {pre.get('time')}, exit "
                                f"{pre.get('rc')}, record sha256 "
                                f"{str(pre.get('gate_sha256'))[:16]}")),
            "the priors' drought years 2015 and 2021 are from the public drought record, not "
            "from a model output or head read",
            f"the v2 drafter read {V1_DECL} (keys, created, recipe.run_code, replay_fields and "
            f"its reason, frozen_code, sha256), the event, exit code, time, code and sha256 "
            f"fields of {V1_LOG}'s attempt and pregate logs (v1's pregate and gate, both exit "
            f"0: booleans on forcing arrays), {V1_RAIN_MANIFEST}'s input paths and sha256, and "
            "the rulings files; no v1 forward existed at drafting and no v1 output was read"],
        "no_prior_look_statement": (
            "No replay forward existed at drafting; no inelastic share, crossing count under "
            "replay, or replayed-forcing statistic had been computed. The engine was tested on "
            f"synthetic inputs only. {V1_ID} runs concurrently (pumping-only); v2's drafter "
            "and code read none of its outputs, and v2 is registered before any v2 forward."),
        "written_by": f"{X}/prereg_fw2.py",
    }


def runner_problems() -> list[str]:
    sh = (REPO / RUNNER).read_text()
    lines = set(sh.splitlines())
    out = [f"{RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    out += [f"{RUNNER} does not carry {s!r}" for s in RUNNER_FRAGMENTS if s not in sh]
    for bad in ("LOG=$R/fw_replay", "OUTD=results/twin_forward/fw_replay",
                f"ID={V1_ID}"):
        if bad in lines:
            out.append(f"{RUNNER} carries v1's line {bad!r}")
    if re.search(r'-s "fw\$i"|-t "fw\$i"', sh):
        out.append(f"{RUNNER} names v1's tmux sessions fw$i")
    if "/home/" in sh:
        out.append(f"{RUNNER} hardcodes a /home path")
    return out


def scorer_problems() -> list[str]:
    out = []
    if (S.PREREG_ID, S.FROZEN, S.FIELDS) != (PREREG_ID, FROZEN, FIELDS):
        out.append("scorer id, frozen list or mode differs from the declaration")
    if S.SCEN_NAMES[1:] != tuple(s.split(":")[0] for s in SCENARIOS):
        out.append("scorer scenario names differ from the runner's")
    if S.OUT_STEM.split("{")[0].rstrip("/") in S.V1_DIRS or S.LOGDIR in S.V1_DIRS:
        out.append("scorer output or log directory is v1's")
    return out


def readiness(run_code: str | None, repo: Path = REPO) -> list[str]:
    out = []
    d = doc(run_code, repo)
    f, why = replay_fields(repo)
    if f != FIELDS:
        out.append(f"ruling {RULING_ID}: {PREREG_ID} registers in '{FIELDS}' mode only; the "
                   f"rain check says: {why}")
    out += rulings_problems(repo) + v1_problems(repo)
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
        S1.check_input_digests(d, repo)
    except S.EngineeringError as err:
        out.append(str(err))
    if run_code and re.fullmatch(r"[0-9a-f]{7}", run_code):
        pre = pregate_record(run_code, repo)
        if pre is None or int(pre.get("rc", -1)) != 0:
            out.append(f"no passing design-time forcing pregate at {run_code} in {PREGATE_LOG} "
                       f"(bash {RUNNER} pregate {run_code}; a trend-factor refusal after "
                       f"registration would end {PREREG_ID})")
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
