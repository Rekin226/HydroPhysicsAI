"""Declaration SD.cut_response_spread.v2 (harness.prereg3, kind 'declaration'; round 7 rank 9):
v1's declaration with the forward's own rheology labelling rule, scored on v1's eight sealed
forwards (reused, bound by sha256).

The document is v1's (prereg_sd.doc(), unchanged code) with only these fields replaced:
id, frozen_code, inputs, recipe, attempt_policy, the step-0 and step-2 engineering-check
texts, no_prior_look_statement and written_by; and these added: supersedes,
v1_engineering_failure, reused_forwards, label_mapping. ``--validate`` checks that every
other field equals v1's registered declaration.

Nothing is read or written by ``--validate`` except the printed document, its canonical
sha256, prereg3's validation, the runner checks (v1's and v2's DRY=1 forward lines), the
dry-run blocking check, the readiness list (which hashes the reused npz files and reads
their rheology_labels and key list only) and this script's sha256. ``--write`` registers it
once (the lead only) with ``--expect-self-sha``, after v1's engineering closure is recorded.

    PY=~/miniforge3/envs/hydro/bin/python
    $PY -I dataprep_staging/sd_column/prereg_sd2.py --validate --quiet
    $PY -I dataprep_staging/sd_column/prereg_sd2.py --write --quiet \\
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

import prereg_sd as PS  # noqa: E402
import score_sd2 as S2  # noqa: E402
import sd2lib as L2  # noqa: E402
import sdlib as L  # noqa: E402
from harness import prereg3  # noqa: E402

ID = L2.SD2_ID
THIS = f"{L.X}/prereg_sd2.py"
TESTS = (PS.TESTS, f"{L.X}/tests/test_sd_v2.py")
RUNNER = f"{L.X}/run_sd2.sh"
RUNNER_LINES = (f"ID={ID}", f"V1={L2.V1_ID}", f"CODE={L.CODE}", f'WT="$HOME/{L.WT_DIR}"',
                f"FORWARD_PY_SHA256={L2.FORWARD_PY_SHA256}",
                "DECL=results/dataprep/preregistration/$ID.json",
                "REPORT=results/dataprep/reports/$ID/report.json",
                "ATT_V1=$LOG/attempts.jsonl", "ATT=$LOG/attempts_v2.jsonl",
                'export HYDROMIND_GW_DATA="$REPO/chou-shui-data/data"')
RUNNER_FRAGMENTS = ('score --attempts "$ATT_V1" --out "$REPORT"',)
#: fields v2 replaces or adds; every other field must equal v1's registered declaration
REPLACED = ("id", "frozen_code", "inputs", "recipe", "attempt_policy", "thresholds",
            "no_prior_look_statement", "written_by")
ADDED = ("supersedes", "v1_engineering_failure", "reused_forwards", "label_mapping")
#: fields prereg3.write_prereg adds to a registered file (not part of the drafted document)
WRITER_FIELDS = ("schema", "created", "tool", "tool_version", "harness_version",
                 "blocking_rule", "scope_windows", "unverifiable_prior_looks", "units_registry")
ENG = "engineering_checks_before_any_cut30_quantity_is_formed"
STEP0_V1 = "step 0: registered and intact; frozen code; inputs; attempts.jsonl binds every npz"
STEP0_V2 = ("step 0: registered and intact (v2; and v1 0d05eb53 registered, intact and "
            "unscored); frozen code (v2's score_sd2.py, sd2lib.py, run_sd2.sh and v1's "
            "score_sd.py, sdlib.py, run_sd.sh at their v1 sha256); inputs; every key's "
            "registered v1 command is v2's command; v1's attempts.jsonl binds every npz to the "
            "sha256 of v1's forward of the key (reused_forwards) and no later forward or "
            "archive event touches it")
LABELS_V1 = "rheology labels"
LABELS_V2 = ("rheology labels mapped to the declared columns by forward.py's own labelling "
             "rule at main 4c2723e (label_mapping; verified against the npz labels and the "
             "--vep-json files of v1's registered command; any ambiguity refuses; the map "
             "must be the declared column order)")


def repo() -> Path:
    return S2.repo()


def frozen_code() -> dict:
    return {p: L.sha(repo() / p) for p in S2.FROZEN}


def v1_registered() -> dict | None:
    p = repo() / L2.V1_DECL
    return json.loads(p.read_text()) if p.exists() else None


def reused_forwards(v1: dict | None) -> dict:
    fw = (v1 or {}).get("forwards", {})
    out = {}
    for k in L.FORWARDS:
        rec = L2.V1_FORWARDS[k]
        out[k] = {"command": L.forward_command(k),
                  "v1_command_identical": fw.get(k, {}).get("command") == L.forward_command(k),
                  "npz": f"{L.out_stem(k)}.npz",
                  **{f: rec[f] for f in ("attempt", "rc", "start", "time", "code",
                                         "worktree_head", "declaration_sha256", "npz_sha256",
                                         "summary_csv_sha256", "members_csv_sha256",
                                         "stdout_sealed_sha256")}}
    return out


def label_block() -> dict:
    exp = {}
    for k, f in L.FORWARDS.items():
        exp[k] = {"columns": list(f["columns"]),
                  "rheology_labels": L2.V1_OBSERVED_LABELS[k],
                  "rheology_index": {c: i for i, c in enumerate(f["columns"])},
                  "v1_expected_labels": L.expected_labels(k)}
    return {"rule": L2.LABEL_RULE, "forward_py": L2.FORWARD_PY,
            "forward_py_sha256_at_code": L2.FORWARD_PY_SHA256,
            "rule_source_lines": list(L2.RULE_SOURCE_LINES),
            "ambiguity_refuses": list(L2.AMBIGUITY_REFUSALS),
            "per_forward": exp,
            "code": "sd2lib.label_mapping / check_forward_npz2 (v1's other layout checks are "
                    "sdlib.check_forward_npz, unchanged; only its label line is replaced)"}


def doc() -> dict:
    d = PS.doc()
    fz = frozen_code()
    v1 = v1_registered()
    rest = [x for x in d["inputs"] if not isinstance(x, dict) and x not in (PS.THIS, PS.TESTS)]
    plan = L2.reuse_plan((v1 or {}).get("forwards", {}))
    d["id"] = ID
    d["frozen_code"] = fz
    d["inputs"] = ([{"path": p, "sha256": s} for p, s in fz.items()]
                   + [THIS, PS.THIS, *TESTS] + rest + [L2.V1_DECL, L2.V1_CLOSURE]
                   + [{"path": f"{L.out_stem(k)}.npz", "sha256": L2.V1_FORWARDS[k]["npz_sha256"]}
                      for k in L.FORWARDS])
    rc = dict(d["recipe"])
    rc["runner"] = (f"no forward runs under v2: v1's eight forwards are reused (bash "
                    f"{PS.RUNNER} forward all ran them under {L2.V1_ID}; sha256 "
                    f"{L2.V1_FROZEN[PS.RUNNER]}); then bash {RUNNER} score once; sha256 "
                    f"{fz[RUNNER]}")
    rc["gpu"] = "none under v2 (the 8 forwards of 1 member x 2 scenarios ran under v1)"
    rc["reuse"] = ("each key's npz is bound by the sha256 v1's runner recorded "
                   "(reused_forwards); a key whose v1 command were not v2's would need a "
                   f"rerun of that key only; none does ({len(plan['reuse'])} of "
                   f"{len(L.FORWARDS)} identical)")
    d["recipe"] = rc
    th = json.loads(json.dumps(d["thresholds"]))
    eng = th[ENG]
    if eng[0] != STEP0_V1 or LABELS_V1 not in eng[2]:
        raise RuntimeError("v1's engineering-check texts are not the expected ones")
    eng[0] = STEP0_V2
    eng[2] = eng[2].replace(LABELS_V1, LABELS_V2, 1)
    d["thresholds"] = th
    ap = dict(d["attempt_policy"])
    ap["scored_runs"] = ("no forward runs under v2 (v1's eight are reused, bound by sha256); "
                         "one scoring run: the first scorer run that passes steps 0-3 writes "
                         "report.json (mode 'x') and is final")
    ap["values_written_before_scoring"] = (
        ap["values_written_before_scoring"] + "; unchanged since v1: v1's sealed stdouts, "
        "the sd_* CSVs and every cut30 slice stay unopened until v2's report.json exists")
    ap["engineering_retries"] = (
        "free before any cut30 quantity is formed (scorer exit 1-3), each logged in "
        "results/twin_runs/sd_column/attempts_v2.jsonl; a fix that changes the frozen code, "
        "the worktree commit or the recipe needs a new declaration disclosing the attempts; "
        "a reused forward is never rerun or archived under v2")
    d["attempt_policy"] = ap
    d["no_prior_look_statement"] = (
        "v1's eight sd_<key> forwards exist (exit 0) and are unopened: their stdout is sealed "
        "(mode 400) and no sealed log, sd_* CSV or cut30 slice has been opened by anyone. "
        "v1's single scoring ran steps 0 and 1 (E0 reproduces the already published archived "
        "page figure) and stopped in step 2, which compares layout fields only (scenario "
        "strings, dates, origin, member, gain, labels, forward options, the heads_mean shape, "
        "the grid, the rain hash) and printed only the label mismatch; steps 3 and 4 never "
        "ran, so no cut30 quantity of any sd forward was formed or printed. The v2 builder "
        "opened the npz files only for rheology_labels, scenario_names, the dates shape and "
        "the key list, and v1's attempts.jsonl (hashes and times). Pooled cut30 values of the "
        "page, R0 and R1c archives are published (0.942 cm fan, 0.958 cm R0); the Q0-family "
        "archives' cut30 values pool two ic members and have not been aggregated by zone for "
        "any of the declared readings by the drafter.")
    d["written_by"] = THIS
    d["supersedes"] = {
        "id": L2.V1_ID, "sha256": L2.V1_SHA256, "created": L2.V1_CREATED,
        "relation": "engineering v2 under v1's attempt_policy ('a fix that changes the frozen "
                    "code, the worktree commit or the recipe needs a v2 declaration disclosing "
                    "the attempts'); v1 is closed unscored; not a post-failure follow-up (no "
                    "cut30 quantity of v1 exists), so it does not spend v1's exploratory slot",
        "changed": "only the rheology label mapping (and the id, the reuse binding, and the "
                   "fields that name v2's files); estimands, gates, thresholds, readings, "
                   "wordings and the columns are v1's"}
    d["v1_engineering_failure"] = {
        "attempts": {"forwards": "8 of 8 exit 0 (attempt 1 each; reused_forwards)",
                     "score": L2.V1_SCORE_ATTEMPT},
        "step": "step 2 (layout), after step 0 (registration, binding) and step 1 (E0) passed",
        "closure_note": f"{L2.V1_CLOSURE} (the lead's record: status "
                        "engineering_failure_v1_closed, 2026-11-01T15:15Z; a hashed input)",
        "message_reported": L2.V1_FAILURE_REPORTED,
        "message_rebuilt": L2.v1_failure_text(),
        "message_note": "v1's scorer stdout was not archived; message_rebuilt is score_sd's "
                        "message format applied to the npz rheology_labels (metadata) and "
                        "sdlib.expected_labels; q0 and q0cs matched v1's rule (shared "
                        "directory names, index suffix)",
        "cause": "sdlib.expected_labels labels every column by its directory name; "
                 "forward.py labels a column with tau_max_years 'tau<N>y' and a single column "
                 "'column' (label_mapping.rule)",
        "read": "no cut30 quantity was formed and no sealed value (stdout, CSV, cut30 slice, "
                "subsidence or head array value) was read under v1 or by the v2 builder"}
    d["reused_forwards"] = reused_forwards(v1)
    d["label_mapping"] = label_block()
    return d


def identity_problems(d: dict, v1: dict | None) -> list[str]:
    """Fields v2 must carry unchanged from v1's registered declaration."""
    if v1 is None:
        return [f"{L2.V1_DECL} missing"]
    out = []
    for k in sorted(set(d) | set(v1)):
        if k in REPLACED or k in ADDED or k in WRITER_FIELDS:
            continue
        if d.get(k) != v1.get(k):
            out.append(f"field {k!r} differs from v1's registered declaration")
    th, t1 = d["thresholds"], v1["thresholds"]
    if {k: v for k, v in th.items() if k != ENG} != {k: v for k, v in t1.items() if k != ENG}:
        out.append("thresholds (other than the engineering-check texts) differ from v1's")
    e2, e1 = th[ENG], t1[ENG]
    if len(e1) != len(e2) or e1[1] != e2[1] or e1[3:] != e2[3:]:
        out.append("engineering checks other than steps 0 and 2 differ from v1's")
    for k in ("code", "flags", "scenario", "q0nh", "why_gain_0"):
        if d["recipe"].get(k) != v1["recipe"].get(k):
            out.append(f"recipe[{k!r}] differs from v1's")
    for k in ("pre_declared_alternatives", "exploratory_followups_max", "followup_label",
              "thresholds_frozen", "report"):
        if d["attempt_policy"].get(k) != v1["attempt_policy"].get(k):
            out.append(f"attempt_policy[{k!r}] differs from v1's")
    if d["frozen_code"].get(PS.RUNNER) != v1["frozen_code"].get(PS.RUNNER):
        out.append("v1's runner is not frozen at its v1 sha256")
    for p, s in L2.V1_FROZEN.items():
        if d["frozen_code"].get(p) != s or v1["frozen_code"].get(p) != s:
            out.append(f"{p} is not at its v1 sha256")
    return out


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
        out.append("v2 runner DRY=1 forward lines differ from the reused commands")
    return out + PS.runner_problems()           # v1's runner: its lines and DRY output


def forward_py_problems() -> list[str]:
    """forward.py at CODE (git object) is the pinned rule source."""
    r = repo()
    g = subprocess.run(["git", "-C", str(r), "show", f"{L.CODE}:{L2.FORWARD_PY}"],
                       capture_output=True)
    if g.returncode != 0:
        return [f"git cannot show {L.CODE}:{L2.FORWARD_PY}"]
    import hashlib

    out = []
    if hashlib.sha256(g.stdout).hexdigest() != L2.FORWARD_PY_SHA256:
        out.append(f"{L.CODE}:{L2.FORWARD_PY} is not the pinned sha256")
    return out + L2.rule_source_problems(g.stdout.decode())


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
    if (r / S2.REPORT).exists():
        out.append(f"{S2.REPORT} exists")
    try:
        S2.verify_v1()
    except S2.Engineering as e:
        out.append(str(e))
    v1 = v1_registered() or {"forwards": {}}
    out += [f"v1 input {c['path']}: {c['problem']}" for c in prereg3.changed_inputs(v1)]
    plan = L2.reuse_plan((v1_registered() or {}).get("forwards", {}))
    out += [f"forward {k}: {why}; rerun that key only (new declaration)"
            for k, why in plan["rerun"].items()]
    files, e = L2.bind_v1_forwards(r / L2.V1_ATTEMPTS, r)
    out += [f"reuse binding: {x}" for x in e]
    for k, p in files.items():
        m = L2.npz_label_metadata(p)
        _, le = L2.label_mapping(k, m["labels"], m["has_by_rheology"], r,
                                 v1["forwards"].get(k, {}).get("command"))
        out += [f"label mapping: {x}" for x in le]
        if m["labels"] != L2.V1_OBSERVED_LABELS[k]:
            out.append(f"{k}: npz labels are not the disclosed v1 labels")
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
    ident = identity_problems(d, v1_registered())
    print("identity with v1 (all but the replaced/added fields):", ident or "identical")
    probs = runner_problems() + forward_py_problems()
    print("recipe/binding checks:", probs or "ok")
    block = prereg3.blocking_looks([], None, test_time=dt.datetime.now(dt.timezone.utc))
    print("blocking looks (dry run):", block or "none (declaration with no tests and no fresh "
          "units)")
    pending = readiness(d)
    print("readiness:", pending or "ready to register")
    self_sha = L.sha(Path(__file__).resolve())
    print(f"this script sha256: {self_sha}")
    print("columns included:", sorted(c for c, v in d["columns"].items() if v["included"]))
    print("reused forwards:", sorted(d["reused_forwards"]))
    bad = bool(errs or ident or probs or block or pending)
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
