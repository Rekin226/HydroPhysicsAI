"""Declaration of I1.X5SC.gate0.v1 (coast step E2, synthetic identifiability gate 0), kind
'declaration', before any of its 12 synthetic column fits exists.

By default it VALIDATES only (prereg3.validate_prereg) and reports which frozen inputs are
still missing; nothing is written. ``--register`` writes it once (prereg3.write_prereg),
after checking that the code is the frozen worktree, the truth heads and synthetic panels
exist and were made by that code, and every input hashes. Run it from the repository root
through the frozen-code launcher, under the X5 lock:

    results/twin_runs/par32/py.sh dataprep_staging/x5_slow/prereg_x5sc_gate0.py
    flock results/dataprep/preregistration/.x5.lock \\
        results/twin_runs/par32/py.sh dataprep_staging/x5_slow/prereg_x5sc_gate0.py --register
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import e2common as C  # noqa: E402
import jobmeta as J  # noqa: E402

sys.path.insert(0, str(C.REPO / "dataprep_staging"))
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
X = "dataprep_staging/x5_slow"
SCRIPTS = ("e2common.py", "dump_fit_heads.py", "make_synthetic.py", "e2flags.py",
           "jobmeta.py", "run_e2.sh", "evaluate_e2.py", "prereg_x5sc_gate0.py",
           "tests/test_x5sc_gate0.py", "tests/fake_arms.py")
AQ_COLUMN_2026_10 = "results/twin_runs/stage3_datum_gate/coupled_leveling_aquitard"


def synthetic_inputs() -> list[str]:
    out = [C.HEADS_NPZ, f"{C.SYNTH_DIR}/generator_manifest.json"]
    for t in C.SPEC["primary_truths"]:
        out += [f"{C.SYNTH_DIR}/truth_{t}/data/{C.SPEC['panel']}",
                f"{C.SYNTH_DIR}/truth_{t}/truth_column.json"]
    return out


def input_paths() -> list[str]:
    return ([f"{X}/{s}" for s in SCRIPTS]
            + [C.SPEC["q0_nh_column"], C.SPEC["theta"], f"{C.SPEC['data_src']}/{C.SPEC['panel']}",
               C.PY_SH, C.CODE_DIFF] + synthetic_inputs())


def run_instructions() -> list[str]:
    wt = "$HOME/HydroPhysicsAI-x5sc"
    return [
        "cd <repository root>; mkdir -p results/twin_runs/par32",
        f"git diff {C.BASE_COMMIT[:7]} --binary -- hydrophysics > {C.CODE_DIFF}; for f in "
        "$(git ls-files --others --exclude-standard hydrophysics); do git diff --no-index "
        f"--binary /dev/null \"$f\" >> {C.CODE_DIFF}; done",
        f"git worktree add --detach {wt} {C.BASE_COMMIT[:7]}; git -C {wt} apply "
        f"\"$PWD/{C.CODE_DIFF}\"",
        f"{C.PY_SH}: export PYTHONPATH={wt}; exec \"$HOME/miniforge3/envs/hydro/bin/python\" "
        "-P \"$@\" (chmod +x)",
        f"HYDROMIND_GW_DATA=$PWD/{C.SPEC['data_src']} {C.PY_SH} {X}/dump_fit_heads.py   (GPU, "
        "about 2 min)",
        f"{C.PY_SH} {X}/make_synthetic.py   (CPU, about 1 min)",
        f"flock results/dataprep/preregistration/.x5.lock {C.PY_SH} {X}/prereg_x5sc_gate0.py "
        "--register",
        "tmux new -s x5sc; export CUDA_MPS_PIPE_DIRECTORY=$HOME/.mps/pipe "
        "CUDA_MPS_LOG_DIRECTORY=$HOME/.mps/log; nvidia-smi; nvidia-cuda-mps-control -d (only "
        f"if not running and no other GPU process); bash {X}/run_e2.sh",
        f"after the batch, once: {C.PY_SH} {X}/evaluate_e2.py",
    ]


def doc(frozen: dict | None = None) -> dict:
    g = C.SPEC["gates"]
    frozen = frozen if frozen is not None else J.frozen_values(C.repo_path(C.SYNTH_DIR))
    hashes = ", ".join(f"{s} {C.file_sha256(HERE / s)[:16]}" for s in SCRIPTS
                       if (HERE / s).exists())
    arms = {a: f"truth {t}, form {f}, origin {o}: calibrate_coupled on truth {t}'s synthetic "
               f"panel, surveys {C.SPEC['withhold'].format(year=o[:4])} withheld"
            for a, t, f, o in C.all_arms()}
    return {
        "id": C.DECL_ID, "wp": "I1", "kind": "declaration",
        "title": "Coast E2: can a leveling fit on the 2012-21 survey mask recover a slow "
                 "distal component? Synthetic gate 0 of the slow-component rung I1.X5SC",
        "priors": (
            "The 2026-10 leveling-fitted aquitard column put its term on the bounds in every "
            "zone (TWIN_EVIDENCE_2026-10 section 6), so a slow distal term may not be "
            "identifiable from 2012-21 leveling. The E1 floors (tau_a >= 10 yr, h_a0 >= 0; "
            "VEP distal tau >= 3650 d) stop the fallback to extra elastic storage. Truth "
            f"parameters (background {C.BG_TARGET_CM} cm/yr in 2016, Ska/tau_a "
            f"{C.SENS_TRUTH}/yr, tau_a and VEP tau {C.TAU_A_YEARS:g} yr) come from spent "
            "X5NH bias and ring results; acceptable for a synthetic design, and disclosed "
            "because they tune the pass region toward the hypothesis. Draft-time synthetic "
            "separation (noise-free, q0_nh forward heads, superseded): distal A-N 8.5/14.6 cm "
            "RMS, V-N 2.9/4.5 cm at 2016-04/2019-04 against a 5.4 cm residual sd, so G0-N and "
            "G0-V are informative; an oracle VEP column with tau on its 3650 d floor already "
            "showed a truth-N background above 0.5 cm/yr, so vt may fail G0-N."),
        "arms": arms,
        "recipe": (
            "truth heads: dump_fit_heads.py runs calibrate_coupled.main with the theta and "
            "the common flags and captures its hindcast_with_nudging result (clean_heads v1 "
            "via the theta meta, 2012-01..2022-12), written <= 2021-12. Synthetic panels: "
            "make_synthetic.py (truths N, A, V; e2common.SPEC; self-checked against "
            "calibrate_coupled._predict). Fits: run_e2.sh, calibrate_coupled "
            + " ".join(C.SPEC["calibrate_common"]) + " plus the form flags aq: "
            + " ".join(C.SPEC["forms"]["aq"]) + "; vt: " + " ".join(C.SPEC["forms"]["vt"])
            + "; --withhold-surveys <YYYY>-05:2022-12 at origins "
            + ", ".join(C.SPEC["origins"]) + "; --data and HYDROMIND_GW_DATA = the truth's "
            "synthetic dir (guarded). Code: main " + C.BASE_COMMIT[:7] + " plus "
            + C.CODE_DIFF + " in a detached worktree, run through " + C.PY_SH + " (frozen "
            "values in 'frozen'). Evaluator evaluate_e2.py, report " + C.REPORT
            + f". Spec sha256 {C.spec_sha256()}; scripts (sha256 prefixes) {hashes}"),
        "frozen": frozen,
        "run_instructions": run_instructions(),
        "zones": "Q0's three column zones; distal = column zone distal (easting < 182 km)",
        "burn_in": "unchanged",
        "folds": "leveling site folds seed 2027 (synthetic panels; the k-fold R2 is "
                 "reported only)",
        "gate_scores": NA("pass/fail gates on ratios of synthetic quantities"),
        "primary_model": "aq (constrained aquitard, mid+distal); vt is the pre-declared "
                         "secondary form. No other form (no distal-only aquitard arm)",
        "inputs": input_paths(),
        "tests": NA("synthetic only: no observation unit is scored; the leveling panel is "
                    "used as an inventory (benchmark ids, coordinates, survey dates)"),
        "fresh_units": NA("no observation unit is scored"),
        "thresholds": {
            "units": C.SPEC["background_definition"],
            "passes_if_all": [g["G0-A1"], g["G0-A2"], g["G0-A3"], g["G0-N"]],
            "at_bound_tolerance": f"{g['at_bound_frac']:g} of the bound interval's width in "
                                  "fit coordinates",
            "on_pass": g["eligible_for_b"],
            "on_fail": "the form is not run in I1.X5SC; if both forms fail, I1.X5SC is not "
                       "run and D1's fail wording applies",
            "only_vt_passes": g["only_vt_passes"],
            "binding_on_I1.X5SC": g["binding_on_b"],
            "claim_gate": g["G0-V (claim gate, aquitard form)"],
            "reported_not_gating": g["reported_only"],
            "never_read": g["never_read"]},
        "noise_rule": C.SPEC["noise"],
        "attempt_policy": {
            "scored_runs": "one completed attempt per arm; the evaluator scores once and "
                           f"writes {C.REPORT} only (no other output path)",
            "engineering_retries": "free for crashed attempts only (rc != 0, no column JSON, "
                                   "or abandoned by a killed batch); a completed arm is never "
                                   "rerun. Every start, end, refusal and abandonment is "
                                   f"logged in {C.RUNS_DIR}/{C.LEDGER} (append-only, hash "
                                   "chained, outside the arm dirs); a retry archives the "
                                   "earlier job.json and outputs, deleting nothing",
            "evaluator_refuses": "while the runner holds its lock, while any arm has not "
                                 "ended or never started, when an arm has more than one "
                                 "completed attempt, when the report exists or the ledger "
                                 "records an evaluation, or when the spec, code tree, "
                                 "generator manifest or runs dir differ from 'frozen'",
            "pre_declared_alternatives": NA("none: aq and vt only"),
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True, "report": "every attempt, failures included"},
        "interpretation": (
            "A1 and A3 passing with N passing says the 2012-21 survey mask and noise can "
            "recover a slow distal background without inventing one. A2 says the aquitard "
            "form also recovers the level sensitivity when it exists; G0-V says it does not "
            "invent one when the background is level-independent. A fail closes that form "
            "for I1.X5SC before any real-data E1 slow-component fit exists. A pass is "
            "necessary, not sufficient: the synthetic truths are in the fitted model family "
            "(correctly specified) and the noise is white per survey, while the real 5.4 cm "
            "residual is mostly time-correlated model error that does not average away over "
            "about 250 distal benchmarks, so gate 0 is optimistic."),
        "notes_for_I1.X5SC": [
            "calibrate_coupled's leveling_rows/site_subsidence apply min_obs 5 and the 0.5 "
            "m/yr screen on every survey 2012-01..2022-12 before --withhold-surveys drops "
            "months, so which benchmarks a rolling-origin fit includes depends on surveys it "
            "withholds (design-time 2012-21 data; no post-2022 leak); disclose it there",
            "the vt flags (--tau-max-years 100 also lifts the mid and proximal tau ceilings "
            "above q0_nh's 3960 d) and the at-bound rule are binding on nh_vt and G6"],
        "no_prior_look_statement": (
            "No observation unit is scored. A slow-component column HAS been fitted on real "
            f"data before: the 2026-10 leveling-fitted aquitard column ({AQ_COLUMN_2026_10}, "
            "default bounds, all zones; TWIN_EVIDENCE_2026-10 section 6) collapsed onto its "
            "bounds, and its stage4_column.csv row (leveling R2, rings_independent_r2, bias, "
            "RMSE, run time) was read on 2026-10-28 while drafting E2, to estimate GPU time. "
            "No column under the E1 flags (aquitard floors, per-zone creep, per-zone tau "
            "floor) has been fitted on real or synthetic data before this declaration. "
            "Design-time reads, no observation value: the leveling panel inventory (sid, "
            "datetime, x_3826, y_3826; rows 2012-05..2021-09; elev_m never read); the q0_nh "
            "column JSON's parameters, settings, in-sample rmse_cm/bias_cm and n_sites (model "
            "statistics); the theta meta; the clean_heads v1/v2 MANIFEST text; par31 run-log "
            "timings. The draft's synthetic panels were built on the q0_nh forward npz "
            "(clean_heads v2 nudged, sliced to 2021-12, decompressed in memory); they and "
            "that npz are superseded and are not inputs here."),
        "design_time_disclosures_to_log": [
            "leveling panel inventory read for E2 design (columns sid, datetime, x_3826, "
            "y_3826; rows 2012-05..2021-09; elev_m never read); purpose design (synthetic "
            "survey mask)",
            f"{AQ_COLUMN_2026_10}/stage4_column.csv row read 2026-10-28 (an existing "
            "column's model statistics, including its rings_independent_r2 on the 2015-21 "
            "rings, already looked); purpose design (GPU time)"],
        "written_by": f"{X}/prereg_x5sc_gate0.py",
    }


def missing_inputs() -> list[str]:
    return [p for p in input_paths() if not C.repo_path(p).exists()]


def register_checks(d: dict) -> list[str]:
    """Why the declaration cannot be written yet (empty = ready)."""
    bad = [f"missing input {p}" for p in missing_inputs()]
    fz = d["frozen"]
    if fz["code"]["worktree_is_main_repo"]:
        bad.append("hydrophysics is imported from the main repository: run through "
                   f"{C.PY_SH} (frozen worktree)")
    if fz["code"]["worktree_head"] != C.BASE_COMMIT:
        bad.append(f"worktree HEAD {fz['code']['worktree_head']} is not {C.BASE_COMMIT}")
    for k in ("py_sh_sha256", "code_diff_sha256"):
        if fz["code"][k] is None:
            bad.append(f"frozen code {k} missing")
    if bad:
        return bad
    man = J.check_synth(C.repo_path(C.SYNTH_DIR))
    if man["heads_sha256"] != fz["heads_npz_sha256"]:
        bad.append("the synthetic panels were not generated from the current fit_heads.npz")
    if man.get("code_tree_sha256") != fz["code"]["tree_sha256"]:
        bad.append("the generator ran on another code tree")
    if man.get("heads_code_tree_sha256") != fz["code"]["tree_sha256"]:
        bad.append("dump_fit_heads.py ran on another code tree")
    if C.repo_path(C.RUNS_DIR).exists() and any(C.repo_path(C.RUNS_DIR).iterdir()):
        bad.append(f"{C.RUNS_DIR} is not empty: arms must not exist before registration")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--register", action="store_true",
                    help="write the declaration (prereg3.write_prereg); default: validate")
    args = ap.parse_args(argv)
    d = doc()
    errs = prereg3.validate_prereg(d)
    if errs:
        print("INVALID:", *errs, sep="\n  ")
        return 1
    pending = register_checks(d)
    if not args.register:
        print("valid; not written (pass --register to write)")
        if pending:
            print("not ready to register:", *pending, sep="\n  ")
        return 0
    if pending:
        print("refused:", *pending, sep="\n  ")
        return 1
    print(*prereg3.write_prereg(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
