"""Declaration X5.changhua_lever_model.v1 (harness.prereg3, kind 'declaration'; round 5 rank 8),
registered before any all30 or replay forward: R1c's all-class Changhua head lever beside its
Changhua-pumping family (model-internal spread, L), beside its 2016-18 masked refit (T, an L
comparison), and F, the gain of N2C's own per-well-demeaned slope estimator on R1c's response
to replayed pre-origin county anomalies relative to R1c's steady-state lever per unit
fractional cut. F < 0.5 caps an N2C 'response supported' at N for C13a (user ruling
2026-11-01, fourth set; tightening only).

Model-only: no observation is read or scored. ``--validate`` writes nothing: it prints the
document, its canonical sha256, prereg3's validation, the runner/scorer binding checks, the
dry-run blocking check, the readiness list and this script's sha256. ``--write`` registers it
once (the lead only), with ``--expect-self-sha``. Before it is ready the lead runs, from the
repository root: ``run_chlever.sh replays``, ``run_chlever.sh prebuild`` (GPU, baseline only)
and ``run_chlever.sh precheck``; both outputs are frozen into the declaration.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY -I dataprep_staging/x5_changhua_lever/prereg_chlever.py --validate --quiet
    $PY -I dataprep_staging/x5_changhua_lever/prereg_chlever.py --write --quiet \\
        --expect-self-sha <sha256 printed by --validate>
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
for p in (str(REPO / "dataprep_staging"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import replay as RP  # noqa: E402
import score_chlever as S  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
ID = S.PREREG_ID
X = S.X
THIS = f"{X}/prereg_chlever.py"
TESTS = f"{X}/tests/test_chlever.py"
RUNNER = f"{X}/run_chlever.sh"
WT_HEAD = "1b08fb6"
PY_SH = "results/twin_runs/r6ch/py.sh"
FLAGS = ("--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation "
         "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
         "--column-hpc0-fast-days 365 --device cuda --compile-matvec --save-members yearly")
ALL30 = ("all30:irrigation=0.7,aquaculture=0.7,livestock=0.7,domestic=0.7,industry=0.7,"
         "other=0.7@2026-01")
RUNNER_LINES = (f"ID={ID}", f"WT_HEAD={WT_HEAD}", f"PY={PY_SH}", 'WT="$HOME/HydroPhysicsAI-r6ch"',
                f"ALL30='{ALL30}'", f'FLAGS="{FLAGS}"',
                "ARMS=(ccz ccz_m1618 chc chs cdm csw ccs rs x5r1t cpw b1)",
                "YEARS=(2012 2013 2014 2015 2016 2017 2018)",
                "DECL=results/dataprep/preregistration/$ID.json",
                "REPORT=results/dataprep/reports/$ID/report.json",
                'export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"')
RUNNER_FRAGMENTS = ('eval "$(fwd_cmd "$a")" > "$sealed" 2> "$err"', 'chmod 400 "$sealed"',
                    "@2026-01#changhua", "@2026-01#yunlin", "@2023-01")
N2C_ID = "N2C.changhua_response.prospective_2026-11.v1"
CLASS_SHARE_ID = "X5.lever_class_share.v1"
CLASS_SHARE_REPORT = f"results/dataprep/reports/{CLASS_SHARE_ID}/report.json"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "missing"


def canonical_sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def frozen_code() -> dict:
    return {p: sha(REPO / p) for p in S.FROZEN}


def arm_files() -> dict:
    out = {}
    for arm, spec in S.ARMS.items():
        th = f"results/twin_runs/stage3_datum_gate_{arm}/stage3_theta.json"
        col = f"results/twin_runs/stage3_datum_gate_{arm}/{spec['column']}/vep_zonal_leveling.json"
        rec = {**spec, "theta": th, "theta_sha256": sha(REPO / th), "column_json": col,
               "column_sha256": sha(REPO / col)}
        if arm == "ccz_m1618":
            cb = f"results/twin_runs/stage3_datum_gate_{arm}/{S.COL_B}/vep_zonal_leveling.json"
            rec.update(column_b_for_the_independence_check=cb, column_b_sha256=sha(REPO / cb))
        out[arm] = rec
    return out


def frozen_inputs() -> dict:
    """The replays and the precheck as written before registration (or 'missing')."""
    rj = REPO / S.REPLAYS / "replays.json"
    pj = REPO / S.PRECHECK
    rep = json.loads(rj.read_text()) if rj.exists() else None
    pre = json.loads(pj.read_text()) if pj.exists() else None
    return {"replays": {"replays_json": f"{S.REPLAYS}/replays.json",
                        "replays_json_sha256": sha(rj), "content": rep},
            "precheck": ({"precheck_json": S.PRECHECK, "precheck_json_sha256": sha(pj), **pre}
                         if pre else {"precheck_json": S.PRECHECK, "precheck_json_sha256":
                                      "missing", "band_arms": [], "arms": {}})}


def command(arm: str) -> str:
    extra = f"--scenario '{ALL30}'"
    if arm == "ccz":
        extra += (f" --recorded-pumping 'CH30={S.REPLAYS}/ch30.csv@2026-01#changhua'"
                  f" --recorded-pumping 'YL30={S.REPLAYS}/yl30.csv@2026-01#yunlin'"
                  + "".join(f" --recorded-pumping 'RP{y}={S.REPLAYS}/rp{y}.npz@2023-01'"
                            for y in S.REPLAY_YEARS))
    col = S.ARMS[arm]["column"]
    return (f"{PY_SH} -m hydrophysics.twin.forward --theta results/twin_runs/"
            f"stage3_datum_gate_{arm}/stage3_theta.json {extra} {FLAGS} --vep-json "
            f"results/twin_runs/stage3_datum_gate_{arm}/{col}/vep_zonal_leveling.json "
            f"--out results/twin_forward/chlever_{arm}")


def doc() -> dict:
    fz = frozen_code()
    fi = frozen_inputs()
    arms = arm_files()
    w = json.loads((REPO / S.WELLS).read_text())
    return {
        "id": ID, "wp": "X5", "kind": "declaration",
        "title": "R1c's all-class Changhua head lever: model-internal spread across the "
                 "Changhua-pumping family (L), the masked-refit comparison T (L), and F, the "
                 "share of the steady-state lever that N2C's own estimator exercises on "
                 "replayed pre-origin county anomalies (F < 0.5 caps C13a at N); model-only",
        "priors": (
            "Model-only (no observation is read or scored). N2C (registered 2026-10-29) decides "
            "R1c's Changhua head response on 2026-11..2027-10 by the per-well-demeaned slope "
            "of e = obs - ZR on c = PR - ZR, with PR forced by both counties' recorded "
            "anomalies from 2023-01. That estimator sees only within-window variation of c, "
            "which is a filtered, partly lagged response to monthly anomalies; the page's "
            "Changhua lever (C13a) is a sustained x0.7 all-class cut. F measures how much of "
            "the sustained lever per unit fractional cut the estimator's own operation "
            "exercises on R1c, using pre-origin anomaly replays (no X1). X5.lever_class_share"
            ".v1 (PASS) found the all-class lever invariant to the energy-to-volume "
            "conversion within about 5 % (q0c/x5r1 1.048/1.052 Changhua, 1.034/1.036 "
            "Yunlin), so the family spread here measures T/S/forcing-variant structure, not "
            "conversion. ccz and ccz_m1618 share recipe, structure and forcing with 3 of 11 "
            "years masked, so T near 1 is expected and is reported with no gate (review "
            "P2.R2). Every family arm failed at least one registered gate; the spread mixes "
            "failed rungs and is worded as model-internal only. The fitted canal/water terms "
            "sit at their bounds in csw/ccs, so the spread says nothing about the missing "
            "non-Jiji water."),
        "rulings": {
            "2026-11-01 user (fourth set), rank 8": "F < 0.5 (share of R1c's Changhua lever "
                                                    "exercised by N2C's estimator) caps an N2C "
                                                    "response_supported at N for C13a: "
                                                    "APPROVED, pre-declared (tightening only)",
            "2026-10-29 user (third set) (d)": "C13 split; only R1c's all-class lever may be "
                                               "shown under an N2C pass",
            N2C_ID: "page_mapping.never: no Changhua subsidence lever, no absolute volume, no "
                    "claim about the page model follows from any verdict (unchanged here)"},
        "disclosed": {
            "class_share_all30_ratio_x5r1_over_q0c": {"changhua": {"2026-27": 1.048,
                                                                   "2030-32": 1.052},
                                                      "yunlin": {"2026-27": 1.034,
                                                                 "2030-32": 1.036},
                                                      "source": CLASS_SHARE_REPORT},
            "q0c_x5r1_all30_levers": "known from X5.lever_class_share.v1 (scored); read here "
                                     "from results/twin_forward/class_share_{q0c,x5r1}.npz "
                                     "as disclosed arms, never re-run",
            "exploration": "the 2026-10-28 explorer (exp1/exp7) printed irrigation-only cut30 "
                           "levers for 12 Changhua-pumping worlds; the drafter of this "
                           "declaration did not open exp*.json",
            "archive_metadata": "the drafter read only the archived forwards' metadata "
                                "(scenario names, shapes, member labels, forward options, "
                                "theta git/pump_kwh/meter_filter); no head value"},
        "arms": arms,
        "disclosed_arms": S.DISCLOSED_ARMS,
        "disclosed_files_sha256": {a: sha(REPO / q) for a, q in S.DISCLOSED_NPZ.items()},
        "disclosed_files_checks": {"scenario_names": list(S.DISCLOSED_SCEN),
                                   "forward_options": S.DISCLOSED_OPTIONS,
                                   "member_label": "stage3_datum_gate_<arm>",
                                   "dates": "2012-01..2032-12, origin 2022-12, horizon 120"},
        "scenarios": {
            "baseline": "climatology of the 2012-22 record after the 2022-12 origin",
            "all30": f"'{ALL30}' (every class x0.7 from 2026-01; describe() "
                     f"{S.SCEN_DESC[S.ALL30]!r})",
            "CH30/YL30 (ccz only)": "recorded-pumping arms, multiplier 0.7 on all classes "
                                    "from 2026-01 to 2032-12 in Changhua (#changhua) or "
                                    "Yunlin (#yunlin) cells only: R1c's steady-state lever "
                                    "per county",
            "RP2012..RP2018 (ccz only)": "recorded-pumping arms, per-cell field from 2023-01: "
                                         "replay.py's contiguous pre-origin anomaly blocks "
                                         "(target 2023-01+j <- source Y-01+j, j=0..57)"},
        "replays": fi["replays"],
        "precheck": fi["precheck"],
        "recipe": {
            "code": f"main {WT_HEAD} (N2C's registered code commit) in the clean detached "
                    "worktree $HOME/HydroPhysicsAI-r6ch, read-only until the report exists",
            "launcher": f"{PY_SH}: 'export PYTHONPATH=<worktree>' then exec the hydro env "
                        "python -P; HYDROMIND_GW_DATA=<repo>/chou-shui-data/data is exported "
                        "by the runner",
            "forwards": {a: command(a) for a in S.ARMS},
            "runner": f"bash {RUNNER} forward (eleven arms, sequential, tmux, under MPS), then "
                      f"bash {RUNNER} score (once); sha256 {fz[RUNNER]}",
            "scorer": f"{S.FROZEN[0]} sha256 {fz[S.FROZEN[0]]}",
            "binding": "runner and scorer refuse unless this declaration is registered and "
                       "intact and the frozen code shas match; the scorer refuses npz files "
                       "that are not the last successful forward outputs in attempts.jsonl, "
                       "pre-builds that differ from precheck.prebuild_sha256, a replays.json "
                       "other than replays.replays_json_sha256, and recorded arms whose file "
                       "sha256 or county scope differ from the declared ones",
            "gpu": "eleven forwards (one with 11 scenarios) of about 2-5 min each on the T4 "
                   "after the pre-builds (about 0.5-1 GPU-h in all); code freeze while they "
                   "run"},
        "frozen_code": fz,
        "zones": (f"N2C's 74 Changhua screens (and 84 Yunlin, report only) of {S.WELLS} "
                  f"(sha256 {S.WELLS_SHA}; rows sha256 {w['rows_sha256']}), each at its own "
                  "cell and layer"),
        "burn_in": "unchanged (forward origin 2022-12; the cut starts 2026-01; replays 2023-01)",
        "folds": NA("model-only declaration; no folds"),
        "gate_scores": NA("declared reading bands; no weighted score"),
        "primary_model": "R1c (stage3_datum_gate_ccz, main parameter set), the N2C primary",
        "inputs": ([{"path": p, "sha256": fz[p]} for p in S.FROZEN]
                   + [THIS, TESTS, PY_SH, S.PRECHECK, f"{S.REPLAYS}/replays.json",
                      RP.PRODUCT + "/MANIFEST.json", RP.PRODUCT + "/" + RP.TABLE,
                      RP.COUNTY_MAP, CLASS_SHARE_REPORT,
                      f"results/dataprep/preregistration/{N2C_ID}.json",
                      *S.DISCLOSED_NPZ.values()]
                   + [r["theta"] for r in arms.values()] + [r["column_json"] for r in arms.values()]
                   + [arms["ccz_m1618"]["column_b_for_the_independence_check"]]),
        "tests": NA("model-only declaration: the readings are about model runs; no fresh unit "
                    "exists and none is consumed"),
        "fresh_units": NA("no observation is scored"),
        "thresholds": {
            "units": "L(arm, w) = mean over the 74 Changhua screens and the months of window w "
                     "(2026-27 = 2026-01..2027-12, 2030-32 = 2030-01..2032-12) of (all30 - "
                     "baseline) head of the arm's own forward",
            "evaluable_if": f"per window: L(ccz, w) >= {S.MIN_LEVER_M} m (the judge's "
                            "'L_ccz >= 0.05 m at the 74 screens'); the spread and T lines of "
                            "window w are written only where it holds; a window where it fails "
                            "is reported 'not evaluable in w' with no ratio line. It gates no "
                            "F wording (F has its own evaluable_if) and caps nothing: the "
                            "2026-11-01 ruling approved only the F < 0.5 cap",
            "spread_L": "ratio L(arm, w) / L(ccz, w) for the band arms (chc, chs, cdm, csw, "
                        "ccs, rs, x5r1t that passed the precheck; cpw and b1 never; arms "
                        "failing the precheck are listed with precheck.drift_excluded_reason, "
                        "'baseline not reproduced (code or ic-random-stream drift)' for the "
                        "single archives rs and x5r1t, '(code drift)' for the 18-member "
                        "archives, report only); the "
                        "band is [min, max] per window, report only, worded 'model-internal "
                        "lever spread (L)'; the same with the disclosed q0c and x5r1 beside "
                        "it",
            "T_L": "T(w) = L(ccz, w) / L(ccz_m1618, w), report only, no gate",
            "F": {
                "definition": (
                    "lambda_c(well) = mean over 2030-32 of (CH30 or YL30 - baseline) head at "
                    "the screen / 0.3 (steady-state lever per unit fractional cut of county "
                    "c's pumping); for replay j, window months t = 2026-11..2027-10: a_c,j(t) "
                    "= r_c,j(t) - 1 (replays.json), s_j(well, t) = -(lambda_CH a_CH,j(t) + "
                    "lambda_YL a_YL,j(t)) (the steady-state response), c_j(well, t) = "
                    "RPj - baseline head (R1c's dynamic response, the quantity N2C's "
                    "contrast is); F = N2C's estimator (bootstrap_scorer.slope_fe) of c on s "
                    "pooled over the 7 replays x 74 screens, one fixed effect per (replay, "
                    "screen): F = sum c~ s~ / sum s~^2"),
                "evaluable_if": f"RMS of s~ pooled >= {S.F_MIN_RMS_M} m and F finite; "
                                "otherwise no F wording and no cap",
                "replay_source": f"{RP.PRODUCT}/{RP.TABLE} via common.guard, use "
                                 f"{RP.USE}, in_grid all-class, Changhua and Yunlin, months "
                                 f"<= {RP.SLICE_END}; never X1",
                "reading": {
                    "F >= 0.8": S.WORDING["F_ge_0p8"],
                    "0.5 <= F < 0.8": S.WORDING["F_0p5_0p8"],
                    "F < 0.5": S.WORDING["F_lt_0p5"],
                    "not evaluable": S.WORDING["F_not_evaluable"]},
                "report_only": "F per replay and its range; RMS of c~ per replay (N2C's "
                               "informative quantity under replayed anomalies); lambda "
                               "medians; superposition (CH30 + YL30) / all30"},
            "engineering_checks_before_any_lever_is_read": [
                "step 0: registered and intact; frozen code; attempts.jsonl binds every npz; "
                "replays.json as registered",
                "step 1: scenario names and baseline/all30 strings exact; ccz's recorded arms "
                "carry the declared file sha256, scope and start; dates 2012-01..2032-12; "
                "origin 2022-12; horizon 120; heads_mean (n_scen, 4, 2148, 252); member "
                "label stage3_datum_gate_<arm>; forward options; theta meta equal to the "
                "pre-build's",
                f"step 2: baseline heads at the 158 screens reproduce the pre-build within "
                f"{S.REF_TOL_M} m (the pre-builds reproduce the archives: precheck)"],
            "bounds": "inclusive, unrounded"},
        "evidence_and_ledger_mapping": {
            "scope": "Feeds the claims ledger (C13a) and the evidence record only. No status "
                     "moves to S or D. There is no 'N2C speaks for the family' clause: N2C "
                     "scores R1c's within-window anomaly response, not the sustained lever.",
            "C13a_spread": "C13a's 'untested (N2C registered, scored not before 2028-01-15)' "
                           "label gains the spread line (L) when evaluable",
            "C13a_F": "F sets how an N2C 'response supported' may be worded (bands above); "
                      "F < 0.5 caps C13a at N even after a pass (the only cap; user ruling "
                      "2026-11-01). F not evaluable or the lever not evaluable gives no F or "
                      "ratio wording and no cap: C13a then follows N2C's page_mapping and "
                      "X5.n2c_reading.v1. This never loosens N2C's page_mapping",
            "T": "the Changhua head evidence may cite T as an L comparison; step C's "
                 "design-time support stays attributed to the masked refit (ccz_m1618)",
            "combination": "where X5.n2c_reading.v1 also restricts the display, the stricter "
                           "reading applies"},
        "attempt_policy": {
            "scored_runs": "one forward per arm and one scoring run; the first scorer run "
                           "that passes steps 0-2 writes report.json (mode 'x') and is final",
            "values_written_before_scoring": (
                "forward.py writes every scenario's values before the scorer runs (stdout "
                "'=== forward summary ===', results/twin_forward/chlever_<arm>.npz, "
                ".members.npz, .summary.csv, .members.csv). The runner sends stdout to "
                "results/twin_runs/x5_chlever/sealed/<arm>_forward.stdout.SEALED (mode 400) "
                "and records every sha256 in attempts.jsonl; nobody opens, tails, greps or "
                "loads them, or any all30/CH30/YL30/RP slice, before report.json exists. "
                "stderr may be read to diagnose a crash."),
            "engineering_retries": (
                "free before any lever is READ: a crash, OOM, missing file or a scorer "
                f"failure in steps 0-2; each logged; `bash {RUNNER} archive '<reason>'` "
                "moves an attempt's outputs unopened; a fix that changes the frozen code, "
                "the worktree, the replays, the pre-builds or the recipe needs a v2 "
                "declaration disclosing the failed attempts"),
            "pre_declared_alternatives": [],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included"},
        "interpretation": (
            "The spread and T say how much R1c's sustained all-class Changhua lever depends "
            "on which failed Changhua-pumping variant one picks (L, model-internal). F says "
            "how much of that sustained lever N2C's within-window estimator can see on R1c "
            "under realistic pre-origin anomalies: F near 1 means the window's anomaly "
            "response is close to the steady-state lever, so an N2C pass bears on the "
            "lever's size; F well below 1 means N2C decides mainly the fast, first-year part "
            "and the sustained lever stays a model extrapolation. Nothing here is evidence "
            "about the aquifer."),
        "rule_risks_disclosure": [
            "the replay source's MANIFEST forbids forcing_recorded_arm: the replays use "
            "twin.forward's recorded-pumping mechanism as a design device for synthetic "
            "future months under the allowed use design_time_evidence; no replay output is "
            "ever joined to an observation or used as a recorded arm of a scored test",
            "every family arm failed at least one registered gate; the spread is labelled so",
            "the single archives (rs, x5r1t, b1) are reproduced on heads_mean, which averages "
            "the unperturbed member and one perturbed ic member; if the ic random stream "
            "changed between the archive's commit and 1b08fb6, such an arm fails the precheck "
            "with unchanged physics and is excluded ('baseline not reproduced (code or "
            "ic-random-stream drift)'), so the band may shrink for that reason; the precheck "
            "runs before any lever exists, so the exclusion cannot depend on the outcome",
            "the q0c and x5r1 class-share files are bound by sha256 and checked for scenario "
            "names and strings, dates, origin, horizon, member label, gain and forward options "
            "before any lever is read",
            "ccz's six-zone county column was suspended as a subsidence fallback (ruling "
            "2026-10-21); only heads are used here, and under --column-heads free heads do "
            "not depend on the column (checked on ccz_m1618 in the precheck)",
            "the 18-member archives are reproduced on their unperturbed main member at "
            "December layer-2 heads in 1 cm quanta (the only per-member heads archived)",
            "T near 1 is expected (same recipe, 3 of 11 years masked); it carries no gate",
            "q0c/x5r1 all30 values are known (class-share report) and enter only as disclosed "
            "arms beside the band"],
        "no_prior_look_statement": (
            "No forward with an all30 scenario or a replay arm exists for any arm of this "
            "declaration (the only all30 runs are q0c and x5r1, disclosed). The pre-builds "
            "are baseline-only. No observation is read by the forwards' scored output or by "
            "the scorer."),
        "written_by": THIS,
    }


def runner_problems() -> list[str]:
    sh = (REPO / RUNNER).read_text()
    lines = set(sh.splitlines())
    out = [f"{RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    out += [f"{RUNNER} does not carry {s!r}" for s in RUNNER_FRAGMENTS if s not in sh]
    if "/home/" in sh:
        out.append(f"{RUNNER} hardcodes a /home path")
    dry = subprocess.run(["bash", str(REPO / RUNNER), "forward"], cwd=REPO,
                         env={"DRY": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
                         capture_output=True, text=True)
    if dry.stdout.splitlines() != [command(a) for a in S.ARMS]:
        out.append("runner DRY=1 forward lines differ from the declared commands")
    return out


def readiness(d: dict) -> list[str]:
    out = []
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (REPO / p).exists():
            out.append(f"input {p} is missing")
    rep = d["replays"]["content"]
    if rep is None:
        out.append("replays not built: bash run_chlever.sh replays")
    else:
        for r in {**rep["replays"], **rep["cuts"]}.values():
            if sha(REPO / S.REPLAYS / r["file"]) != r["sha256"]:
                out.append(f"replay file {r['file']} differs from replays.json")
        if sorted(rep["replays"]) != [f"rp{y}" for y in S.REPLAY_YEARS]:
            out.append("replays.json does not hold rp2012..rp2018")
    pre = d["precheck"]
    if not pre.get("arms"):
        out.append("precheck missing: bash run_chlever.sh prebuild, then precheck")
    else:
        if not pre.get("ok_reference"):
            out.append("precheck: ccz or ccz_m1618 does not reproduce (lead decision needed)")
        for arm in S.ARMS:
            r = pre["arms"].get(arm)
            if r is None or sha(REPO / S.PRE / f"{arm}.npz") != r["prebuild_sha256"]:
                out.append(f"pre-build {arm} differs from precheck.json")
    if (REPO / f"results/dataprep/preregistration/{ID}.json").exists():
        out.append(f"{ID} is already registered")
    for a in S.ARMS:
        for s in ("npz", "members.npz", "summary.csv", "members.csv"):
            if (REPO / f"results/twin_forward/chlever_{a}.{s}").exists():
                out.append(f"results/twin_forward/chlever_{a}.{s} exists before the declaration")
    if (REPO / S.REPORT).exists():
        out.append(f"{S.REPORT} exists")
    wt = Path.home() / "HydroPhysicsAI-r6ch"
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        return out + [f"worktree $HOME/HydroPhysicsAI-r6ch: {err}"]
    if head != WT_HEAD:
        out.append(f"worktree $HOME/HydroPhysicsAI-r6ch is at {head}, not {WT_HEAD}")
    if dirty:
        out.append("worktree $HOME/HydroPhysicsAI-r6ch is dirty")
    py = REPO / PY_SH
    if not py.exists() or f"export PYTHONPATH={wt}" not in py.read_text().splitlines():
        out.append(f"{PY_SH} does not pin the worktree")
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
    probs = runner_problems()
    print("recipe/binding checks:", probs or "ok")
    block = prereg3.blocking_looks([], None, test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none (declaration with no tests and no fresh "
          "units)")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    self_sha = sha(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
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
