"""Declaration SD.cut_response_spread.v1 (harness.prereg3, kind 'declaration'; round 7 rank 9):
the cut30 avoided subsidence across the columns that fit leveling equally well, by zone and
county, each column on its own paired heads, main parameter set only (model-only).

Merges subsidence lens B R3 and lens C R2 with the review fixes: per-column numbers come from
single-member, single-column runs (the archives pool 18 members and 2 rheologies); E0 checks
the pooled page figure against the FAN average 0.942 cm (not a Yunlin figure); the q0_nh
column runs on a fresh origin-2022-12 forward (its archive assimilated heads to 2025-06 and is
never read, so q0_nh is labelled 'unverified pairing'); the pairing gate reproduces each
column's stored hindcast subsidence on its own heads and reads no leveling; columns without
an independent stored trajectory are labelled and every reading is given without them too.

Nothing is read or written by ``--validate`` except the printed document, its canonical
sha256, prereg3's validation, the runner check, the dry-run blocking check, the readiness
list and this script's sha256. ``--write`` registers it once (the lead only) with
``--expect-self-sha``. It must be registered BEFORE any forward of run_sd.sh runs.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY -I dataprep_staging/sd_column/prereg_sd.py --validate --quiet
    $PY -I dataprep_staging/sd_column/prereg_sd.py --write --quiet \\
        --expect-self-sha <sha256 printed by --validate>
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parent), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import score_sd as S  # noqa: E402
import sdlib as L  # noqa: E402
from harness import prereg3  # noqa: E402

NA = prereg3.not_applicable
ID = L.SD_ID
THIS = f"{L.X}/prereg_sd.py"
TESTS = f"{L.X}/tests/test_sd_column.py"
RUNNER = f"{L.X}/run_sd.sh"
PAGE_HTML = L.PAGE_HTML
ZONE_MAP = "results/dataprep/zone_county_map/v2/zone_county_map.npz"
POLYGON = "chou-shui-data/data/Zhuoshui Alluvial Fan/Zhuoshui Alluvial Fan.json"
RUNNER_LINES = (f"ID={ID}", f"CODE={L.CODE}", f'WT="$HOME/{L.WT_DIR}"',
                "KEYS=(page q0 q0cs cwa q0nh ccz q0c x5r1cs)",
                "DECL=results/dataprep/preregistration/$ID.json",
                "REPORT=results/dataprep/reports/$ID/report.json",
                'export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"')
RUNNER_FRAGMENTS = ('> "$sealed" 2> "$err"', 'chmod 400 "$sealed"')


def repo() -> Path:
    return S.repo()


def frozen_code() -> dict:
    return {p: L.sha(repo() / p) for p in S.FROZEN}


def archives() -> list[str]:
    out = {S.E0_ARCHIVE}
    for _run, _sub, _tier, chk in L.COLUMNS.values():
        if chk and chk[0] == "single":
            out.add(f"{L.FWD}/{chk[1]}.npz")
        elif chk and chk[0] == "members":
            out.add(f"{L.FWD}/{chk[1]}.members.npz")
    return sorted(out)


def column_records(r: Path) -> dict:
    inc = L.inclusion(r)
    out = {}
    for c, (run, _sub, tier, chk) in L.COLUMNS.items():
        p = r / L.column_path(c)
        rec = {"tier": tier, "forward": L.forward_of(c), "column_json": L.column_path(c),
               "column_sha256": L.sha(p), "theta": L.theta_path(run),
               "theta_sha256": L.sha(r / L.theta_path(run)),
               "forward_theta": L.theta_path(L.FORWARDS[L.forward_of(c)]["run"]),
               "archive_check": list(chk) if chk else None,
               **inc["columns"][c]}
        if p.exists():
            rec["param_hash"] = L.param_hash(json.loads(p.read_text()))
        out[c] = rec
    return out


def doc() -> dict:
    r = repo()
    fz = frozen_code()
    cols = column_records(r)
    inc = L.inclusion(r)
    thetas = sorted({L.theta_path(L.FORWARDS[k]["run"]) for k in L.FORWARDS}
                    | {v["theta"] for v in cols.values()})
    return {
        "id": ID, "wp": "SD", "kind": "declaration",
        "title": "Cut30 avoided subsidence across the columns that fit leveling equally well, "
                 "by zone and county, each on its own paired heads, main parameter set only "
                 "(model-form spread, not an interval; model-only)",
        "priors": (
            "Model-only: no observation is read or scored. The page's cut30 benefit is all "
            "elastic in the page column (science-13), so its size inherits the column's Ske, "
            "which leveling does not pin: columns with out-of-fold R2 within 0.02 of the page "
            "differ about 4.5x in distal and 2x in mid Ske (planner and review figures, from "
            "the published column JSONs). Expected reading: inland about 2x and coast about "
            "4-5x column dependence (judge pass chance about 0.25 for 'inland robust'). The "
            "page column was chosen knowing the 2024-25 pile seasonal R2 of 0.92 (U7, E06); "
            "that is disclosed, and nothing here selects a column."),
        "rulings": {
            "judge rank 9": "Tier A / Tier B split, equal weights, label 'model-form spread, "
                            "not an interval'; displaying the spread on the page tile needs a "
                            "user ruling (C14e, U1.no_band.v1); ledger and README Limits need "
                            "none",
            "2026-10-29 user (4)": "no correction factor follows from any spread"},
        "columns": cols,
        "dropped_at_declaration": L.DROPPED_AT_DECLARATION,
        "inclusion": {"rule": f"out-of-fold leveling R2 (published r2_outoffold) >= page - "
                              f"{L.BAR_DELTA}; Changhua tier listed apart with no bar",
                      "page_r2_outoffold": inc["page_r2_outoffold"], "bar": inc["bar"],
                      "note": "spent fit metrics used only as the inclusion rule they already "
                              "were; the bar was fixed by the judge before these values were "
                              "aggregated; q0_s29 has the same parameters as q0 (only the "
                              "fold seed differs) and counts once (distinct_of); cwa_tau30 "
                              "enters by the rule (R0's tau30 rheology, as Tier A carries "
                              "the page's)"},
        "forwards": {k: {"command": L.forward_command(k), "theta": L.theta_path(f["run"]),
                         "rain": f["rf"], "columns": list(f["columns"]),
                         "out": f"{L.out_stem(k)}.npz"} for k, f in L.FORWARDS.items()},
        "recipe": {
            "code": f"main {L.CODE} in the clean detached worktree $HOME/{L.WT_DIR}, "
                    "imported through PYTHONPATH with python -P from the repository root; "
                    "hydrophysics/twin is unchanged from it to origin/main fc0ea4a except "
                    "the page files",
            "runner": f"bash {RUNNER} forward all (or one key per tmux window under MPS), "
                      f"then bash {RUNNER} score once; sha256 {fz[RUNNER]}",
            "flags": L.FLAGS, "scenario": L.CUT30,
            "q0nh": f"adds {L.NUDGE_FLAGS} (the hindcast nudging its column was fitted on), "
                    "origin 2022-12, never --extend-record-to",
            "why_gain_0": "with --column-heads free the subsidence does not depend on the "
                          "restart gain; gain 0 makes the saved heads the column's driver "
                          "heads, so the elastic split is exact and no origin restart reads "
                          "a head",
            "gpu": "8 forwards of 1 member x 2 scenarios (about 2-5 min each on the T4); code "
                   "freeze while they run"},
        "frozen_code": fz,
        "zones": ("zones.fan_zones on the 1 km grid cell centres (proximal east of 205 km, "
                  "distal west of 182 km); county from zone_county DEFAULT_MAP (Yunlin = not "
                  f"Changhua); townships from the page payload ({PAGE_HTML}, eligible = the "
                  "page's headline filter)"),
        "burn_in": "unchanged: hindcast 2012-01..2022-12 from the calibration's month-0 field; "
                   "the cut starts 2026-01; avoided subsidence counted from Dec 2022",
        "folds": NA("model-only declaration; no folds"),
        "gate_scores": NA("declared readings; no weighted score"),
        "primary_model": f"the page column {L.PAGE_COLUMN_SHA8} (stage3_datum_gate/"
                         "coupled_leveling) on the page theta's main parameter set",
        "inputs": ([{"path": p, "sha256": s} for p, s in fz.items()]
                   + [THIS, TESTS, PAGE_HTML, ZONE_MAP, POLYGON, L.RF_VENDOR, L.RF_DEFAULT]
                   + thetas + [v["column_json"] for v in cols.values()
                               if v["column_sha256"] != "missing"]
                   + archives()),
        "tests": NA("model-only declaration: the readings are about model runs; no fresh unit "
                    "exists and none is consumed"),
        "fresh_units": NA("no observation is scored"),
        "thresholds": {
            "quantities": {
                "avoided": "cut30 minus baseline avoided subsidence (cm, + = less sinking), "
                           "from Dec 2022, at 2026-12, 2028-12 and 2032-12, cell means by "
                           "group: fan, county, zone x county, inland Yunlin (proximal + mid)",
                "split": "elastic = Ske (h_cut - h_base) on the layer-mean driver head per "
                         "cell; inelastic = avoided - elastic (VEP delayed term plus the "
                         "aquitard term where present)",
                "distal_share": "sum of avoided over distal cells / sum over the fan; and "
                                "over Yunlin distal / Yunlin",
                "towns": "Yunlin township top-5 by mean avoided (page-eligible townships), at "
                         "each horizon; the reading uses 2032-12 (C11's horizon)",
                "cm_per_m": "group mean avoided (cm) / group mean avoided driver head (m), "
                            "where |avoided head| >= 1 mm",
                "storage_ratio": "column Ske (Yunlin distal, mid) / sum over layers of the "
                                 "flow theta's exp(log_S_<zone>) (page about 11x)",
                "f_subs": f"Yunlin avoided at {L.F_SUBS_EARLY} / at {L.F_SUBS_LATE} (page "
                          "column; a declaration for X5.gnss_seasonal_column.v1, never a "
                          "gate)"},
            "readings": {
                "R_inland": "over the distinct passing columns of a set, at each horizon "
                            "(primary 2028-12): every inland Yunlin avoided value of the same "
                            "sign and max/min <= 1.5 -> 'inland Yunlin robust within x1.5' "
                            "(in this model family, not a test); else 'column-dependent x r "
                            "(a-b cm)'",
                "R_rank": "top-5 Yunlin township overlap with the page column >= 4 of 5 in "
                          "every distinct column -> 'ranking column-robust', else 'ranking "
                          "column-dependent'; the leader the same in every column -> C11 "
                          "keeps its leader wording, else loses it",
                "ranges": "Yunlin avoided range (cm) and Yunlin distal share range (%) "
                          "across the set",
                "sets": "'A' (Tier A, the page's own heads: the headline spread) and 'A+B' "
                        "(adds the Q0 family and R0 on their own heads: head-field plus "
                        "column spread), reported apart, each also 'verified only' (without "
                        "the UNVERIFIED columns page_aquitard and q0_nh); the Changhua tier "
                        "is reported separately and enters no Yunlin reading",
                "equal_weights": "every distinct column counts once; no observation "
                                 "weighting of any kind",
                "evaluable_if": f"page Yunlin avoided at {L.PRIMARY_HORIZON} >= "
                                f"{S.MIN_PAGE_YUNLIN_CM} cm"},
            "engineering_checks_before_any_cut30_quantity_is_formed": [
                "step 0: registered and intact; frozen code; inputs; attempts.jsonl binds "
                "every npz",
                f"step 1 (E0): {S.E0_ARCHIVE} coupled_leveling rheology gives the published "
                f"fan-average {S.E0_PUBLISHED_CM} cm within {S.E0_TOL_CM} cm (Yunlin "
                f"{S.E0_YUNLIN_REPORT_ONLY_CM} cm report only)",
                "step 2: layout (baseline + cut30 strings, 2012-01..2032-12, origin 2022-12, "
                "one member, gain 0, rheology labels, forward options), grid, and the rain "
                "file against the theta's recorded sha256",
                f"step 3: hindcast subsidence reproduces the stored trajectory (one-theta "
                f"archives: monthly, {L.TOL_SINGLE_M} m; 18-member archives: the main "
                f"member's December fields, half a quantum + {L.TOL_MEMBERS_EXTRA_M} m; "
                "page_aquitard: through the page column of the same forward), the column "
                f"rebuilt on the CPU reproduces the forward within {L.TOL_REBUILD_M} m in "
                "both scenarios (maximum absolute difference only; no cut30 quantity is "
                "formed before step 4), and the column's own theta hashes as the forward's; "
                "the page column failing is an engineering failure; any other failing "
                "column is dropped and listed, never tuned",
                "UNVERIFIED columns (no independent stored trajectory) pass on pairing and "
                "rebuild only and carry their label: " + "; ".join(
                    f"{c}: {v}" for c, v in L.UNVERIFIED.items())],
            "bounds": "inclusive, unrounded"},
        "evidence_and_ledger_mapping": {
            "scope": "A new L evidence row (id assigned by the lead), cited by the C03, C29, "
                     "C14e, C11 and C27 internals notes. No status moves; C03 and C09 stay W. "
                     "Showing the spread on the tA tile needs a user ruling.",
            "line_template": "across N leveling-equivalent columns <set words> the Yunlin "
                             "cut30 avoided subsidence by 2028-12 is a-b cm; coast c-d % of "
                             "it; <R_inland> (model-form spread, not an interval); one line "
                             "for Tier A (page_line_L_A) and one for A+B (page_line_L), "
                             "with the verified-only lines beside them",
            "C11": "keeps or loses its leader wording by R_rank",
            "C27": "may cite the R_inland reading as model-form dependence (not a test)",
            "changhua_tier": "C13c wording only; C13c stays W",
            "label": S.LABEL},
        "attempt_policy": {
            "scored_runs": "one forward per key and one scoring run; the first scorer run that "
                           "passes steps 0-3 writes report.json (mode 'x') and is final",
            "values_written_before_scoring": (
                "forward.py prints the fan-mean summary (and a hindcast-vs-leveling line) and "
                "writes results/twin_forward/sd_<key>.{npz,summary.csv,members.csv}; stdout "
                "is sealed (mode 400) under results/twin_runs/sd_column/sealed/ and nobody "
                "opens it, the CSVs or any cut30 slice before report.json exists"),
            "engineering_retries": "free before any cut30 quantity is formed (crash, OOM, missing "
                                   "file, scorer exit 1-3), each logged; archive moves a "
                                   "failed attempt unopened; a fix that changes the frozen "
                                   "code, the worktree commit or the recipe needs a v2 "
                                   "declaration disclosing the attempts",
            "pre_declared_alternatives": [],
            "exploratory_followups_max": 1,
            "followup_label": "post-failure, exploratory; never validation; cannot promote",
            "thresholds_frozen": True,
            "report": "every attempt, failures included"},
        "interpretation": (
            "The spread says how much the model's cut30 avoided subsidence depends on which "
            "leveling-equivalent column (and, in A+B, which head field) one picks. It is a "
            "model-form spread over a declared, non-exhaustive set, not an interval, not a "
            "bound and not evidence about the aquifer; the response size stays untested."),
        "rule_risks_disclosure": [
            "column-pairing rule: every column runs only on the heads it was fitted on; a "
            "column without its own forward is never substituted",
            "q0_nh: the archive datum_gate_q0_nh.npz (origin 2025-06, extend_record_to "
            "2025-06) is never read (judge rank 9); the scored forward is fresh with origin "
            "2022-12, so q0_nh has no clean reference trajectory and is labelled 'unverified "
            "pairing'; A+B is reported with and without it",
            "page_aquitard: no archive stores its trajectory; it is checked only through the "
            "page column of the same forward and the CPU rebuild (same load_columns code), "
            "so it is labelled 'unverified trajectory (shared heads)'; A is reported with "
            "and without it",
            "the page column was chosen knowing the 2024-25 pile seasonal R2 0.92 (U7, E06)",
            "the Changhua tier: R1c's six-zone county column was suspended as a subsidence "
            "fallback (ruling 2026-10-21) and county-split Changhua columns are "
            "elastic-dominated with a rebound artefact; reported apart, C13c wording only",
            "the planners read column JSON fit metrics and archive metadata only; no cut30 "
            "aggregate of any column on main-member heads existed at drafting",
            "the forwards print a hindcast-vs-leveling line (HYDROMIND_GW_DATA is needed for "
            "the inputs); it goes to the sealed stdout and is never read for this "
            "declaration (spent calibration data)"],
        "no_prior_look_statement": (
            "No sd_<key> forward exists. Pooled cut30 values of the page, R0 and R1c archives "
            "are published (0.942 cm fan, 0.958 cm R0); the Q0-family archives' cut30 "
            "values pool two ic members and have not been aggregated by zone for any of the "
            "declared readings by the drafter."),
        "written_by": THIS,
    }


def runner_problems() -> list[str]:
    r = repo()
    sh = (r / RUNNER).read_text()
    lines = set(sh.splitlines())
    out = [f"{RUNNER} does not carry the line {s!r}" for s in RUNNER_LINES if s not in lines]
    out += [f"{RUNNER} does not carry {s!r}" for s in RUNNER_FRAGMENTS if s not in sh]
    if "/home/" in sh:
        out.append(f"{RUNNER} hardcodes a /home path")
    env = {"DRY": "1", "PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    if os.environ.get("DATAPREP_REPO"):
        env["DATAPREP_REPO"] = os.environ["DATAPREP_REPO"]
    dry = subprocess.run(["bash", str(r / RUNNER), "forward", "all"], cwd=r, env=env,
                         capture_output=True, text=True)
    if dry.stdout.splitlines() != [L.forward_command(k) for k in L.FORWARDS]:
        out.append("runner DRY=1 forward lines differ from the declared commands")
    return out


def readiness(d: dict) -> list[str]:
    r = repo()
    out = []
    for rec in d["inputs"]:
        p = rec["path"] if isinstance(rec, dict) else rec
        if not (r / p).exists():
            out.append(f"input {p} is missing")
    for c, v in d["columns"].items():
        if v["column_sha256"] == "missing":
            out.append(f"column {c} JSON missing")
    if d["columns"][L.PAGE_COLUMN]["column_sha256"][:8] != L.PAGE_COLUMN_SHA8:
        out.append("the page column is not d290c592")
    if (r / f"results/dataprep/preregistration/{ID}.json").exists():
        out.append(f"{ID} is already registered")
    for k in L.FORWARDS:
        for s in ("npz", "summary.csv", "members.csv"):
            if (r / f"{L.out_stem(k)}.{s}").exists():
                out.append(f"{L.out_stem(k)}.{s} exists before the declaration")
    if (r / S.REPORT).exists():
        out.append(f"{S.REPORT} exists")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true", help="dry run: print, check, write nothing")
    g.add_argument("--write", action="store_true", help="register once (the lead only)")
    ap.add_argument("--expect-self-sha", default=None)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    d = doc()
    if not a.quiet:
        print(json.dumps(d, indent=1, ensure_ascii=False, sort_keys=True))
    print("document canonical sha256:", L.canonical_sha(d))
    errs = prereg3.validate_prereg(d)
    print("validate_prereg:", errs or "valid")
    probs = runner_problems()
    print("recipe/binding checks:", probs or "ok")
    block = prereg3.blocking_looks([], None, test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none (declaration with no tests and no fresh "
          "units)")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    self_sha = L.sha(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
    print("columns included:", sorted(c for c, v in d["columns"].items() if v["included"]))
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
    print(f"registered {path} sha256 {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
