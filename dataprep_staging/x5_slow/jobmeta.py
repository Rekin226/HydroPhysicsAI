"""Per-arm records, the attempt ledger and the guards of run_e2.sh (hashes and times only;
no data value is read or written).

    jobmeta.py preflight RUNS SYNTH          once per batch, before any worker starts:
                                             the frozen code (py.sh worktree at BASE_COMMIT),
                                             the registered declaration and its frozen hashes,
                                             the synthetic panels; closes attempts a dead
                                             batch left open (event "abandoned")
    jobmeta.py start RUNS ARM TRUTH FORM ORIGIN DATA_DIR -- ARGV...
                                             exit 0: go (job.json + ledger "start");
                                             exit 2: the arm completed before, never rerun;
                                             exit 3: refused (ledger "refused")
    jobmeta.py end RUNS ARM RC               job.json rc/end/code_end + ledger "end"
    jobmeta.py check-synth SYNTH             refuses panels whose hash moved
    ... | jobmeta.py redact > LOG            drops calibrate_coupled's real-ring numbers

Ledger: RUNS/attempts.jsonl, append-only (flock), each line chained to the previous one by
its sha256. It lives outside every arm dir, so no rerun can erase it. Retry rule: an arm may
be rerun only when every earlier attempt crashed (rc != 0, no column JSON, or abandoned);
a completed arm (rc 0 with its column JSON) is never rerun. A retry never deletes anything:
the earlier job.json becomes job.attempt<k>.json and the earlier outputs move to
attempt<k>/.

Real-data guard (``guard``): the arm's --data and HYDROMIND_GW_DATA must both resolve to
the generator manifest's synthetic data dir of the arm's truth, the argv must equal the
spec's exactly, and the panel there must be a regular file (not a symlink) whose sha256 is
the generator's and differs from the real panel's. Otherwise the arm is refused before
calibrate_coupled starts.

``end`` also replaces rings_independent_r2 in the arm's column JSON and stage4_column.csv by
a note: it scores the synthetic-fitted column on the REAL rings, and E2 never reads it.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2common as C  # noqa: E402

REDACTED = "redacted: real rings vs a synthetic-fitted column; E2 never reads it"
COLUMN_JSON = "vep_zonal_leveling.json"
EXIT_COMPLETED, EXIT_REFUSED = 2, 3


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------------------
def ledger_path(runs) -> Path:
    return Path(runs) / C.LEDGER


def read_ledger(runs) -> list[dict]:
    p = ledger_path(runs)
    if not p.exists():
        return []
    rows, prev = [], None
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
        rec = json.loads(line)
        if rec.get("prev_sha256") != prev:
            raise SystemExit(f"{p}: line {i + 1} breaks the hash chain (ledger edited?)")
        prev = hashlib.sha256(line.encode("utf-8")).hexdigest()
        rows.append(rec)
    return rows


def append(runs, rec: dict) -> dict:
    """Append ``rec`` (plus time and the previous line's sha256) under an exclusive lock."""
    p = ledger_path(runs)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            lines = fh.read().splitlines()
            prev = hashlib.sha256(lines[-1].encode("utf-8")).hexdigest() if lines else None
            out = {"time": now(), **rec, "prev_sha256": prev}
            fh.seek(0, os.SEEK_END)
            fh.write(json.dumps(out, sort_keys=True, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return out


def attempts(rows: list[dict], arm: str) -> dict[int, dict]:
    """attempt number -> {"start": rec, "end": rec or None} of ``arm``."""
    out: dict[int, dict] = {}
    for r in rows:
        if r.get("arm") != arm:
            continue
        if r["event"] == "start":
            out[int(r["attempt"])] = {"start": r, "end": None}
        elif r["event"] in ("end", "abandoned"):
            out[int(r["attempt"])]["end"] = r
    return out


def completed(a: dict) -> bool:
    e = a["end"]
    return e is not None and e["event"] == "end" and e.get("rc") == 0 and bool(e.get("output"))


def arm_status(rows: list[dict], arm: str) -> dict:
    at = attempts(rows, arm)
    return {"attempts": len(at), "completed": [k for k, a in at.items() if completed(a)],
            "open": [k for k, a in at.items() if a["end"] is None],
            "last": max(at) if at else None}


# ---------------------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------------------
def synth_manifest(synth) -> dict:
    return json.loads((Path(synth) / "generator_manifest.json").read_text())


def guard(arm_dir: Path, truth: str, form: str, origin: str, data_dir: str,
          argv: list[str], man: dict) -> dict:
    """The real-data guard (module docstring); returns what job.json records, raises
    SystemExit with the reason otherwise."""
    from hydrophysics.config import Config

    d = Path(data_dir).resolve()
    tv = man["truths"].get(truth)
    if tv is None or Path(tv["data_dir"]).resolve() != d:
        raise SystemExit(f"data dir {d} is not the generator's truth {truth} dir")
    want = C.fit_args(form, origin, tv["data_dir"], str(arm_dir))
    if list(argv) != want:
        raise SystemExit(f"argv differs from the spec's: {argv} != {want}")
    if argv.count("--data") != 1:
        raise SystemExit("--data must appear exactly once")
    env = os.environ.get("HYDROMIND_GW_DATA")
    if not env or Path(env).resolve() != d:
        raise SystemExit(f"HYDROMIND_GW_DATA={env!r} is not the synthetic dir {d}")
    if Path(Config(data_dir=argv[argv.index("--data") + 1]).data_dir).resolve() != d \
            or Path(Config().data_dir).resolve() != d:
        raise SystemExit("hydrophysics.config.Config resolves another data dir")
    panel = d / C.SPEC["panel"]
    real = C.repo_path(C.SPEC["data_src"]) / C.SPEC["panel"]
    if panel.is_symlink() or not panel.is_file():
        raise SystemExit(f"{panel} is not a regular file (a symlink would be the real panel)")
    if panel.resolve() == real.resolve():
        raise SystemExit(f"{panel} is the real panel")
    sha = C.file_sha256(panel)
    if sha != tv["panel_sha256"]:
        raise SystemExit(f"{panel}: sha256 {sha} is not the generator's")
    real_sha = C.file_sha256(real)
    if sha == real_sha:
        raise SystemExit(f"{panel} has the real panel's content")
    return {"data_dir": str(d), "env_HYDROMIND_GW_DATA": env, "panel": str(panel),
            "panel_sha256": sha, "real_panel_sha256": real_sha, "guard": "passed"}


def launcher_state() -> dict:
    out = {}
    for k, rel in (("py_sh", C.PY_SH), ("code_diff", C.CODE_DIFF)):
        p = C.repo_path(rel)
        out[k] = {"path": rel, "sha256": C.file_sha256(p) if p.exists() else None}
    return out


def check_synth(synth) -> dict:
    man = synth_manifest(synth)
    if man["spec_sha256"] != C.spec_sha256():
        raise SystemExit("the spec changed since the synthetic panels were generated")
    if C.file_sha256(man["heads_npz"]) != man["heads_sha256"]:
        raise SystemExit("the truth heads npz changed since generation")
    for t, v in man["truths"].items():
        got = C.file_sha256(Path(v["data_dir"]) / C.SPEC["panel"])
        if got != v["panel_sha256"]:
            raise SystemExit(f"truth {t}: panel hash moved ({got})")
    return man


def frozen_values(synth) -> dict:
    """The values the declaration freezes and the batch must still match."""
    synth = Path(synth)
    gm = synth / "generator_manifest.json"
    cs = C.code_state()
    lau = launcher_state()
    return {"spec_sha256": C.spec_sha256(),
            "code": {"base_commit": C.BASE_COMMIT, "worktree_head": cs["git_head"],
                     "worktree_is_main_repo": cs["is_main_repo"],
                     "tree_sha256": cs["tree_sha256"], "n_files": cs["n_files"],
                     "py_sh_sha256": lau["py_sh"]["sha256"],
                     "code_diff_sha256": lau["code_diff"]["sha256"]},
            "generator_manifest_sha256": C.file_sha256(gm) if gm.exists() else None,
            "heads_npz_sha256": (C.file_sha256(C.repo_path(C.HEADS_NPZ))
                                 if C.repo_path(C.HEADS_NPZ).exists() else None),
            "runs_dir": C.RUNS_DIR, "report": C.REPORT, "ledger": f"{C.RUNS_DIR}/{C.LEDGER}",
            "arms": [a for a, *_ in C.all_arms()],
            "at_bound_frac": C.SPEC["gates"]["at_bound_frac"]}


def load_declaration(path=None) -> dict:
    p = C.repo_path(path or C.DECL_PATH)
    if not p.exists():
        raise SystemExit(f"{p} does not exist: register {C.DECL_ID} first "
                         "(prereg_x5sc_gate0.py --register)")
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("id") != C.DECL_ID or "frozen" not in d:
        raise SystemExit(f"{p} is not the {C.DECL_ID} declaration")
    return d


def preflight(runs, synth, decl_path=None) -> dict:
    runs, synth = Path(runs), Path(synth)
    rows = read_ledger(runs)
    if any(r["event"] == "evaluated" for r in rows):
        raise SystemExit("the gate was already evaluated; no arm may run after scoring")
    decl = load_declaration(decl_path)
    fz = decl["frozen"]
    cur = frozen_values(synth)
    bad = []
    if cur["code"]["worktree_is_main_repo"]:
        bad.append("hydrophysics is imported from the main repository, not the frozen "
                   "worktree (run through results/twin_runs/par32/py.sh)")
    if cur["code"]["worktree_head"] != C.BASE_COMMIT:
        bad.append(f"worktree HEAD {cur['code']['worktree_head']} is not {C.BASE_COMMIT}")
    for k in ("spec_sha256", "generator_manifest_sha256", "heads_npz_sha256"):
        if cur[k] != fz.get(k):
            bad.append(f"{k} {cur[k]} != declared {fz.get(k)}")
    for k in ("tree_sha256", "py_sh_sha256", "code_diff_sha256", "worktree_head"):
        if cur["code"][k] != fz["code"].get(k):
            bad.append(f"code {k} {cur['code'][k]} != declared {fz['code'].get(k)}")
    if runs.resolve() != C.repo_path(fz["runs_dir"]).resolve():
        bad.append(f"runs dir {runs} is not the declared {fz['runs_dir']}")
    if bad:
        raise SystemExit("preflight refused:\n  " + "\n  ".join(bad))
    check_synth(synth)
    closed = []
    for a, *_ in C.all_arms():
        for k, v in attempts(rows, a).items():
            if v["end"] is None:
                append(runs, {"event": "abandoned", "arm": a, "attempt": k, "rc": None,
                              "output": (runs / a / COLUMN_JSON).exists(),
                              "note": "open attempt of an earlier batch (killed or crashed)"})
                closed.append(f"{a}#{k}")
    return {"declaration_created": decl.get("created"), "abandoned_closed": closed,
            "tree_sha256": cur["code"]["tree_sha256"]}


# ---------------------------------------------------------------------------------------
# start / end
# ---------------------------------------------------------------------------------------
def archive_attempt(arm_dir: Path, k: int) -> None:
    """Keep a crashed attempt's files: job.json -> job.attempt<k>.json, the rest ->
    attempt<k>/ (nothing is deleted)."""
    if not arm_dir.exists():
        return
    jj = arm_dir / "job.json"
    if jj.exists():
        jj.rename(arm_dir / f"job.attempt{k}.json")
    rest = [p for p in arm_dir.iterdir()
            if not re.fullmatch(r"job\.attempt\d+\.json|attempt\d+", p.name)]
    if rest:
        dst = arm_dir / f"attempt{k}"
        dst.mkdir()
        for p in rest:
            shutil.move(str(p), str(dst / p.name))


def start(runs, arm: str, truth: str, form: str, origin: str, data_dir: str,
          argv: list[str], synth=None) -> int:
    runs = Path(runs)
    arm_dir = runs.resolve() / arm
    st = arm_status(read_ledger(runs), arm)
    if st["completed"]:
        print(f"{arm}: completed in attempt {st['completed'][0]}; never rerun")
        return EXIT_COMPLETED
    if st["open"]:
        print(f"{arm}: attempt {st['open']} is still open; refused")
        return EXIT_REFUSED
    man = synth_manifest(synth or C.repo_path(C.SYNTH_DIR))
    try:
        g = guard(arm_dir, truth, form, origin, data_dir, argv, man)
    except SystemExit as e:
        append(runs, {"event": "refused", "arm": arm, "reason": str(e)})
        print(f"{arm}: refused: {e}")
        return EXIT_REFUSED
    k = (st["last"] or 0) + 1
    if st["last"]:
        archive_attempt(arm_dir, st["last"])
    cs = C.code_state()
    rec = {"arm": arm, "truth": truth, "form": form, "origin": origin, "attempt": k,
           **g, "spec_sha256": C.spec_sha256(), "code_start": cs, "launcher": launcher_state(),
           "start": now(), "cmd": list(argv)}
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / "job.json").write_text(json.dumps(rec, indent=1))
    append(runs, {"event": "start", "arm": arm, "attempt": k,
                  "tree_sha256": cs["tree_sha256"], "panel_sha256": g["panel_sha256"],
                  "data_dir": g["data_dir"],
                  "argv_sha256": hashlib.sha256(json.dumps(argv).encode()).hexdigest()})
    return 0


def redact_outputs(arm: Path) -> None:
    j = arm / COLUMN_JSON
    if j.exists():
        p = json.loads(j.read_text())
        if "rings_independent_r2" in p:
            p["rings_independent_r2"] = REDACTED
            j.write_text(json.dumps(p, indent=1))
    c = arm / "stage4_column.csv"
    if c.exists():
        import pandas as pd

        df = pd.read_csv(c)
        if "rings_independent_r2" in df.columns:
            df["rings_independent_r2"] = REDACTED
            df.to_csv(c, index=False)


def end(runs, arm: str, rc: int) -> int:
    runs = Path(runs)
    arm_dir = runs.resolve() / arm
    p = arm_dir / "job.json"
    rec = json.loads(p.read_text())
    redact_outputs(arm_dir)
    out = (arm_dir / COLUMN_JSON).exists()
    rec.update(rc=int(rc), end=now(), output=out, code_end=C.code_state())
    rec["code_changed_during_job"] = rec["code_end"] != rec["code_start"]
    p.write_text(json.dumps(rec, indent=1))
    append(runs, {"event": "end", "arm": arm, "attempt": rec["attempt"], "rc": int(rc),
                  "output": out, "code_changed": rec["code_changed_during_job"]})
    return 0


# ---------------------------------------------------------------------------------------
# log redaction
# ---------------------------------------------------------------------------------------
def redact_line(line: str) -> str:
    """'... rings +0.123 ...' -> 'rings <redacted>'; table rows are dropped by the caller."""
    return re.sub(r"(rings\S*)\s+[-+]?(nan|\d+\.?\d*(e[-+]?\d+)?)", r"\1 <redacted>", line)


def redact_stream(src, dst) -> None:
    table = False
    for line in src:
        if "rings_independent_r2" in line:      # the final DataFrame print: drop it all
            table = True
            dst.write("[stage4 table dropped: it carries rings_independent_r2]\n")
            continue
        if table:
            continue
        dst.write(redact_line(line))
        dst.flush()


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "preflight":
        print(json.dumps(preflight(argv[1], argv[2]), indent=1))
        return 0
    if cmd == "start":
        runs, arm, truth, form, origin, ddir = argv[1:7]
        if argv[7:8] != ["--"]:
            raise SystemExit("start: the argv must follow '--'")
        return start(runs, arm, truth, form, origin, ddir, argv[8:])
    if cmd == "end":
        return end(argv[1], argv[2], int(argv[3]))
    if cmd == "check-synth":
        man = check_synth(argv[1])
        print("synthetic panels intact:", ", ".join(sorted(man["truths"])))
        return 0
    if cmd == "redact":
        redact_stream(sys.stdin, sys.stdout)
        return 0
    raise SystemExit(f"unknown command {cmd!r}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
