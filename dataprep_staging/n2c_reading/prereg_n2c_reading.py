"""Declaration X5.n2c_reading.v1 (harness.prereg3, kind 'declaration'; round 5 rank 10):
synthetic operating characteristics of N2C's decision rule, registered BEFORE any replicate
is drawn and before the replay forward runs, and the hash binding of the shared N2/N2C
block-bootstrap scorer (dataprep_staging/n2_policy/bootstrap_scorer.py) that N2C's step (1)
and N2 use in the 2028 session.

User ruling 2026-11-01 (fourth set, rank 10): RUN; the reading may tighten N2C's display
('reported, not shown' on a high confounded false-pass rate); write and hash the shared
scorer now. Model-only: no observation is read; every noise and confounder parameter cites a
published report field (checked against the files by ``--validate``).

    PY=~/miniforge3/envs/hydro/bin/python
    bash dataprep_staging/n2c_reading/run_n2c_reading.sh replays
    $PY -I dataprep_staging/n2c_reading/prereg_n2c_reading.py --validate --quiet
    $PY -I dataprep_staging/n2c_reading/prereg_n2c_reading.py --write --quiet \\
        --expect-self-sha <sha256 printed by --validate>
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
for p in (str(REPO / "dataprep_staging"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import synth_n2c as S  # noqa: E402
from harness import prereg3  # noqa: E402

BS = S.BS
NA = prereg3.not_applicable
ID = S.DECL_ID
THIS = f"{S.X}/prereg_n2c_reading.py"
TESTS = f"{S.X}/tests/test_n2c_reading.py"
RUNNER = f"{S.X}/run_n2c_reading.sh"
WT_HEAD = "1b08fb6"
PY_SH = "results/twin_runs/r6ch/py.sh"
DEADLINE = dt.datetime(2028, 1, 15, 0, 0, tzinfo=dt.timezone.utc)
N2_ID = "N2.policy_response.prospective_2026-11.v1"
N2C_ID = "N2C.changhua_response.prospective_2026-11.v1"
FLAGS = ("--horizon 120 --gain 1.0 --ic-members 1 --update-method innovation "
         "--restart-taper-km 5 --apex-hold calibrated --column-heads free "
         "--column-hpc0-fast-days 365 --device cpu --save-members yearly")
RUNNER_LINES = (f"ID={ID}", f"WT_HEAD={WT_HEAD}", f"PY={PY_SH}", 'WT="$HOME/HydroPhysicsAI-r6ch"',
                f'FLAGS="{FLAGS}"', "YEARS=(2012 2013 2014 2015 2016 2017 2018)",
                "DECL=results/dataprep/preregistration/$ID.json",
                'export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"')


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).exists() else "missing"


def canonical_sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def command() -> str:
    r = "results/twin_runs"
    return (f"{PY_SH} -m hydrophysics.twin.forward --theta {r}/stage3_datum_gate_ccz/"
            f"stage3_theta.json --vep-json {r}/stage3_datum_gate_ccz/coupled_leveling/"
            f"vep_zonal_leveling.json {FLAGS}"
            + "".join(f" --recorded-pumping 'RP{y}={S.REPLAYS}/rp{y}.npz@2023-01'"
                      for y in S.REPLAY_YEARS) + f" --out {S.FWD_STEM}")


def replays_block() -> dict:
    rj = REPO / S.REPLAYS / "replays.json"
    return {"replays_json": f"{S.REPLAYS}/replays.json", "replays_json_sha256": sha(rj),
            "content": json.loads(rj.read_text()) if rj.exists() else None}


def doc() -> dict:
    code = S.code_sha256()
    scorer = "dataprep_staging/n2_policy/bootstrap_scorer.py"
    return {
        "id": ID, "wp": "X5", "kind": "declaration",
        "title": "Synthetic operating characteristics of N2C's decision rule (confounded "
                 "false-pass rate, wrong-size pass rates, 90 % interval coverage, power) run "
                 "with the hashed shared N2/N2C scorer; tightens N2C's display only; "
                 "model-only",
        "priors": (
            "N2C decides R1c's Changhua head response on 2026-11..2027-10 with the "
            "per-well-demeaned slope beta of e = obs - ZR on c = PR - ZR, a 5 km block "
            "bootstrap 90 % interval and the rule 'response supported' iff the interval "
            "excludes 0, beta in [1/1.5, 1.5] and RMSE(PR) < RMSE(ZR). Changhua heads are "
            "noisy for R1c (anomaly RMSE 1.318 m), the observed regional head anomaly "
            "follows census energy (-0.83) and canal deliveries (+0.83), which co-move in "
            "drought years, and N2C has no power check. This declaration measures, before "
            "any N2C statistic exists, how often that rule says 'response supported' when "
            "the model's response is absent but a canal-like omitted term co-moves with "
            "pumping, when the response is half or double the model's, and how well the "
            "interval covers a correct response. Model-only; no X1."),
        "rulings": {
            "2026-11-01 user (fourth set), rank 10": "RUN; a model-only reading may tighten "
                                                     "N2C's display ('reported, not shown' "
                                                     "on a high confounded false-pass rate); "
                                                     "write and hash the shared N2/N2C "
                                                     "block-bootstrap scorer now",
            "2026-10-29 user (3)": "the N2 companion is dropped: this adds no test and never "
                                   "changes a verdict word"},
        "shared_scorer_binding": {
            "path": scorer, "sha256": code[scorer], "version": BS.VERSION,
            "binds": [f"{N2C_ID}: scoring.same_session 'The scorer code for beta_response and "
                      "its 5 km block bootstrap is one implementation shared with N2, hashed "
                      "and logged before step (1)' is this file at this sha256 (user ruling "
                      "2026-11-01, fourth set)",
                      f"{N2_ID}: beta_response, its interval, the informative gate, the RMSEs "
                      "and the decision rule are computed by the same file"],
            "implements": BS.REGISTERED,
            "interpretations_fixed_here": [
                "RMSE of PR, ZR and CL: equal weight per well (the registered prospective "
                "test's twin.validation convention), pooled reported beside",
                "rows: flags 0-1 and finite obs, ZR and c; no per-well month minimum (none "
                "is registered)",
                "the informative gate uses the contrast over every eligible well and window "
                "month, before any head is read",
                "percentile interval (5 %, 95 %, numpy linear), blocks resampled with "
                "replacement as many as hold scored rows, draws with a zero denominator "
                "dropped and counted",
                "level component: OLS with intercept across wells (report only)",
                "the decision rule is applied literally: an interval wholly below 0 with "
                "beta > 0 matches neither registered not_supported clause and goes on to the "
                "size test; the scorer does not override a registered rule and flags the case "
                "(interval_below_zero_with_positive_beta) for the user"]},
        "design": S.design_spec(),
        "code_sha256": code,
        "replays": replays_block(),
        "recipe": {
            "code": f"main {WT_HEAD} (N2C's code commit) in the clean detached worktree "
                    "$HOME/HydroPhysicsAI-r6ch (shared with X5.changhua_lever_model.v1)",
            "forward": command(),
            "forward_vs_n2c": "N2C's registered primary forward command (CPU, R1c main "
                              "set, --ic-members 1, column-heads free) with the PR arm "
                              "replaced by RP2012..RP2018 and --save-members yearly (no "
                              "effect on heads_mean)",
            "runner": f"bash {RUNNER} forward (CPU, tmux), extract, run",
            "cpu": "one forward with 8 scenarios on CPU (hours) plus about 1 min of replicates"},
        "zones": "N2C's 74 Changhua wells (screens of dataprep_staging/x5_class_share/"
                 "wells_frozen.json; 5 km blocks from the x, y of "
                 f"{S.WELLS_XY})",
        "burn_in": NA("synthetic; the AR(1) noise starts stationary"),
        "folds": NA("synthetic declaration; no folds"),
        "gate_scores": NA("declared reading thresholds; no weighted score"),
        "primary_model": "R1c (stage3_datum_gate_ccz, main set) supplies the contrast; the "
                         "characterised object is N2C's decision rule as implemented by the "
                         "shared scorer",
        "inputs": ([{"path": p, "sha256": h} for p, h in code.items()]
                   + [THIS, TESTS, PY_SH, f"{S.REPLAYS}/replays.json", S.WELLS_XY,
                      S.ARCHIVE_MEMBERS,
                      "results/dataprep/reports/I1_changhua_local/report.json",
                      "results/dataprep/reports/I1_changhua_interannual/report.json",
                      "results/dataprep/reports/I1_changhua_drivers/report.json",
                      "results/dataprep/reports/I1_changhua_drivers/report.md",
                      "results/twin/hybrid_datum_gate_ccz/report.json",
                      f"results/dataprep/preregistration/{N2_ID}.json",
                      f"results/dataprep/preregistration/{N2C_ID}.json",
                      "results/twin_runs/stage3_datum_gate_ccz/stage3_theta.json",
                      "results/twin_runs/stage3_datum_gate_ccz/coupled_leveling/"
                      "vep_zonal_leveling.json"]),
        "tests": NA("synthetic declaration: no fresh unit exists and none is consumed"),
        "fresh_units": NA("no observation is scored"),
        "thresholds": {
            "engineering": [
                "exact plumbing (no noise, kappa 0): beta == k for every informative replay; "
                "a failure is a code defect: engineering failure, no reading",
                f"plumbing cell (k = 1, rho = 0, kappa 0, noise x{S.PLUMBING_NOISE_SCALE}, "
                f"{S.PLUMBING_REPLICATES} replicates): median beta in {list(S.PLUMBING_BAND)}; "
                "outside it (exact plumbing passing) the TERMINAL reading 'not_estimable' is "
                "written to result.json (display DISPLAY.not_estimable: no rate printed, "
                "N2C's display neither narrowed nor widened) and is final",
                "extract: the run's unperturbed main member reproduces datum_gate_ccz's "
                "archived one within one 1 cm quantum (December layer-2 heads at the 74 "
                "screens' cells); every replay arm ran on the declared file (sha256)"],
            "reading": S.design_spec()["reading"],
            "published_fields": {k: {kk: v[kk] for kk in ("value", "source", "field")}
                                 for k, v in S.PARAMS.items()}},
        "evidence_and_ledger_mapping": {
            "scope": "No claim changes status now. The reading conditions C13a's 2028 "
                     "transition only, and only by tightening.",
            "reported_not_shown": "if R1 fires, an N2C 'response supported' leaves C13a at W "
                                  "('reported, not shown')",
            "rates_printed": "if R2 fires, any shown Changhua head lever carries the measured "
                             "rates",
            "power": "beside any non-pass N2C verdict the power is printed",
            "never": "the N2C verdict word never changes; nothing here shows more than "
                     "N2C's page_mapping allows",
            "combination": "with X5.changhua_lever_model.v1 (F cap), the stricter reading "
                           "applies"},
        "attempt_policy": {
            "scored_runs": "one forward, one extract, one registered run; result.json is "
                           "written once (mode 'x') and is final, whatever it shows",
            "sealing": "the forward's stdout and outputs and the panel are not opened, "
                       "printed or summarised before result.json exists (the runner prints "
                       "shas; extract prints the panel shape, block count and the "
                       "reproduction check only)",
            "engineering_retries": "free before the reading exists: a crash, a missing "
                                   "file, a failed reproduction or a failed EXACT plumbing "
                                   "check; each logged (attempts.jsonl, "
                                   "engineering_failure_*.json, which carries no panel "
                                   "summary); a fix that changes the bound code needs a v2 "
                                   "that discloses the failed attempts",
            "v2_scope": "a v2 after an engineering failure may change code only; it may not "
                        "change PARAMS, GRID, REPLICATES, SEED, the plumbing-cell settings, "
                        "READING, DISPLAY or the replays",
            "terminal_outcomes": ["not_estimable: the low-noise plumbing cell's median beta "
                                  "outside [0.9, 1.1] with exact plumbing passing; written "
                                  "once to result.json, final; no v2 may follow it"],
            "pre_declared_alternatives": [],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "deadline": "registered before 2028-01-15 and before any N2C statistic exists"},
        "interpretation": (
            "A high confounded false-pass rate means an N2C pass could come from the "
            "omitted canal co-movement alone; the pass is then reported, not shown. High "
            "wrong-size pass rates or a poorly covering interval mean a shown lever must "
            "carry those rates. Power tells a reader what a 'not supported' or 'not "
            "informative' N2C verdict can mean. All of this is about the decision rule under "
            "declared synthetic worlds; none of it is evidence about the aquifer."),
        "rule_risks_disclosure": [
            "every noise and confounder parameter is a published design-time statistic of "
            "spent 2012-22 heads (I1_changhua_local, I1_changhua_interannual, "
            "I1_changhua_drivers, R1c's hybrid report); the drafter read those fields "
            "(and only those report blocks) while drafting",
            "no per-well canal ridge coefficient is published; kappa is derived from the "
            "canal-only out-of-sample R2 and R0's Changhua RMSE and R2 (PARAMS.kappa_m)",
            "rho = corr(g, a_CH): the judge's 'rho = -0.83 to the contrast's time profile' "
            "is read as -0.83 to the pumping anomaly (c ~ -a), the drought-year direction "
            "that can produce a false pass; the literal reading would bias beta down and "
            "make the false-pass reading vacuous",
            "the block noise model correlates wells in the same 5 km block (0.44) and no "
            "others; the published residual correlation at 5-10 km is 0.36 and beyond 20 km "
            "-0.08",
            "the synthetic panel is complete (74 wells x 12 months, 12 recorded months); the "
            "real window may lack months or wells",
            "the replays use twin.forward's recorded-pumping mechanism for synthetic future "
            "months under the census product's allowed use design_time_evidence (its "
            "MANIFEST forbids forcing_recorded_arm, which is not the use made here)",
            "the drafter ran the driver's smoke mode on a synthetic stand-in panel (no model "
            "output) to test the plumbing; no parameter was changed after it",
            "the cited I1_changhua_drivers and I1_changhua_interannual reports label "
            "themselves after-the-fact analysis (never a gate or a rung choice); here they "
            "only parameterise the synthetic worlds (design_time_evidence, as the judge "
            "required); they set no gate and choose no rung",
            "phi is the median lag-1 autocorrelation of OBSERVED head anomalies over 158 "
            "wells in both counties (R1c's hybrid report), not of R1c's Changhua residuals",
            "rho: the cited field is the observed regional head's correlation with census "
            "energy (-0.83) and canal deliveries (+0.83), used as a proxy for the canal-vs-"
            "pumping co-movement; no canal-vs-energy correlation is published",
            "while drafting, the builder accidentally printed the header and first row of "
            "R1c's registered prospective_2026-11_n2c_r1c/predictions.csv (predicted_m and "
            "baseline_m; baseline_m is CL, a pre-origin climatology derived from "
            "observations); nothing used it",
            "in a later drafting pass (review fixes), the drafter printed the regional_vs_drivers_r and "
            "regional_residual_vs_drivers_r fields (all counties and rungs) of the "
            "I1_changhua_interannual report to look for a published canal-vs-energy "
            "correlation; none exists; no parameter changed",
            "the plumbing cell runs at x0.02 noise with kappa 0 so that it tests the "
            "plumbing, not Monte Carlo luck; the full-noise (k = 1, rho = 0) cell is an "
            "ordinary grid cell",
            "R1 and R2 use the stricter of the unconditional pass rate and the rate among "
            "replicates that pass N2C's informative gate (N2C can only pass when it does)",
            "no X1 value enters any parameter"],
        "no_prior_look_statement": (
            "No replay forward of R1c exists; no synthetic replicate on R1c's contrast has "
            "been drawn; no N2C window head exists."),
        "written_by": THIS,
    }


def published_problems() -> list[str]:
    """The cited report fields carry the declared values."""
    out = []

    def get(path: str, keys: list[str]):
        d = json.loads((REPO / path).read_text())
        for k in keys:
            d = d[k]
        return float(d)

    chk = [("sigma_tot_m", "results/dataprep/reports/I1_changhua_local/report.json",
            ["part_a_local", "changhua", "rmse_r1c_mean"]),
           ("block_share", "results/dataprep/reports/I1_changhua_interannual/report.json",
            ["counties", "changhua", "residual", "R1c", "by_distance", "0-5 km", "median_r"]),
           ("phi", "results/twin/hybrid_datum_gate_ccz/report.json", ["rho", "median_own"])]
    for name, path, keys in chk:
        try:
            if get(path, keys) != S.PARAMS[name]["value"]:
                out.append(f"{name}: {path} {'.'.join(keys)} differs")
        except (OSError, KeyError, ValueError) as err:
            out.append(f"{name}: {err}")
    try:
        k = S.PARAMS["kappa_m"]["inputs"]
        if get("results/dataprep/reports/I1_changhua_drivers/report.json",
               ["summary", "changhua", "median_cv_r2", "raw:canal_w1_cell"]) != k["canal_r2"]:
            out.append("kappa: canal R2 differs")
        if get("results/dataprep/reports/I1_changhua_local/report.json",
               ["part_a_local", "changhua", "rmse_r0_mean"]) != k["rmse_r0_m"]:
            out.append("kappa: R0 RMSE differs")
        md = (REPO / "results/dataprep/reports/I1_changhua_drivers/report.md").read_text()
        if not re.search(r"\|\s*R0\s*\|\s*0\.14\s*\|", md):
            out.append("kappa: R0 Changhua R2 0.14 not in report.md")
        rr = json.loads((REPO / "results/dataprep/reports/I1_changhua_interannual/report.json")
                        .read_text())["counties"]["changhua"]["observed"]["regional_vs_drivers_r"]
        if round(-rr["census_energy"], 2) != 0.83 or round(rr["canal_deliveries"], 2) != 0.83:
            out.append("rho: published 0.83 magnitudes differ")
    except (OSError, KeyError, ValueError) as err:
        out.append(f"kappa/rho: {err}")
    return out


def scorer_problems() -> list[str]:
    """The shared scorer's constants against the N2 and N2C registrations."""
    out = []
    for pid in (N2_ID, N2C_ID):
        p = REPO / f"results/dataprep/preregistration/{pid}.json"
        if not p.exists():
            out.append(f"{pid} not found")
            continue
        d = json.loads(p.read_text())
        interval = (d["primary_metric"]["interval"] if pid == N2_ID
                    else d["tests"][0]["metric"]["interval"])
        if interval != "90 % from a 5 km block cluster bootstrap, B 2000, seed 20261101":
            out.append(f"{pid}: interval wording changed")
        rule = d["decision_rule"] if pid == N2_ID else d["tests"][0]["decision_rule"]
        if rule["not_supported"] != "beta 90 % interval includes 0, or beta <= 0":
            out.append(f"{pid}: decision rule wording changed")
        if d.get("thresholds") != {"informative_head_m": BS.INFORMATIVE_HEAD_M,
                                                 "level": BS.LEVEL,
                                                 "min_clusters": BS.MIN_CLUSTERS,
                                                 "min_months": BS.MIN_MONTHS,
                                                 "ratio_band": BS.RATIO_BAND}:
            out.append(f"{pid}: thresholds differ from the scorer")
    return out


def runner_problems() -> list[str]:
    sh = (REPO / RUNNER).read_text()
    lines = set(sh.splitlines())
    out = [f"{RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    if "/home/" in sh:
        out.append(f"{RUNNER} hardcodes a /home path")
    dry = subprocess.run(["bash", str(REPO / RUNNER), "forward"], cwd=REPO,
                         env={"DRY": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())},
                         capture_output=True, text=True)
    if dry.stdout.splitlines() != [command()]:
        out.append("runner DRY=1 forward line differs from the declared command")
    return out


def readiness(d: dict) -> list[str]:
    out = []
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (REPO / p).exists():
            out.append(f"input {p} is missing")
    rep = d["replays"]["content"]
    if rep is None:
        out.append("replays not built: bash run_n2c_reading.sh replays")
    else:
        for r in rep["replays"].values():
            if sha(REPO / S.REPLAYS / r["file"]) != r["sha256"]:
                out.append(f"replay file {r['file']} differs from replays.json")
    if (REPO / f"results/dataprep/preregistration/{ID}.json").exists():
        out.append(f"{ID} is already registered")
    for p in (S.FWD_STEM + ".npz", S.PANEL, f"{S.OUT_DIR}/result.json"):
        if (REPO / p).exists():
            out.append(f"{p} exists before the declaration")
    if dt.datetime.now(dt.timezone.utc) >= DEADLINE:
        out.append("past 2028-01-15: the declaration must precede N2C's session")
    wt = Path.home() / "HydroPhysicsAI-r6ch"
    try:
        head = subprocess.run(["git", "-C", str(wt), "rev-parse", "--short=7", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        return out + [f"worktree $HOME/HydroPhysicsAI-r6ch: {err}"]
    if head != WT_HEAD:
        out.append(f"worktree $HOME/HydroPhysicsAI-r6ch is at {head}, not {WT_HEAD}")
    try:
        dirty = subprocess.run(["git", "-C", str(wt), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as err:
        dirty = str(err)
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
    probs = runner_problems() + published_problems() + scorer_problems()
    print("binding, published-field and scorer checks:", probs or "ok")
    block = prereg3.blocking_looks([], None, test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none (no tests, no fresh units)")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    print(f"shared scorer {d['shared_scorer_binding']['path']} sha256 "
          f"{d['shared_scorer_binding']['sha256']}")
    self_sha = sha(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
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
