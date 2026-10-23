"""P4.0 repair round: is ``stage3_datum_gate_cwa/stage3_theta.json`` batch 8's merged theta?

    cd dataprep_staging
    python -m p4_changhua.repair_theta_status [OUT.json]

engine/v1 was labelled "provisional: batch 8's merged theta does not exist yet" (stale task
context). The verifier showed that batch 8 had finished about 10 h before the v1 build. This
script collects the evidence read-only (file times, the batch's dependency declaration, the
controller's queue log, the theta's meta and its identity with the full fit) and returns it
as a dict; ``build_engine_v2`` records it in the product's validation.json.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.hashing import sha256_file  # noqa: E402
from common.paths import REPO, rel_to_repo  # noqa: E402

RUN = REPO / "results/twin_runs/stage3_datum_gate_cwa"
THETA = RUN / "stage3_theta.json"
FIT_SUMMARY = RUN / "stage3_fit_summary.json"
FOLD_THETAS = RUN / "stage3_fold_thetas.json"
PAR8 = REPO / "results/twin_runs/par8"
JOBS = PAR8 / "jobs.sh"
QLOG = REPO / "results/twin_runs/gap_queue.log"
V1_BUILD_START_UTC = "2026-10-12T06:40:57Z"       # build_engine.log of engine/v1 (14:40:57 +08)
V1_SHA = "789d0e459a3b35eb3e16002869405c889a86131815fcebfef84a9ffd8370704c"


def _utc(p: Path) -> str:
    return _dt.datetime.fromtimestamp(p.stat().st_mtime, _dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ")


def _max_abs_diff(a, b) -> float:
    if isinstance(a, dict):
        if set(a) != set(b):
            return float("inf")
        return max((_max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((_max_abs_diff(x, y) for x, y in zip(a, b, strict=True)), default=0.0)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b))
    return 0.0 if a == b else float("inf")


def collect() -> dict:
    th = json.loads(THETA.read_text())
    fs = json.loads(FIT_SUMMARY.read_text())
    fit_theta = fs["fit"]["theta"]
    if isinstance(fit_theta, str):        # fit_flow stores the full fit's theta as a repr
        fit_theta = ast.literal_eval(fit_theta)
    meta = th["meta"]
    markers = {}
    for name in ("fit_full", "fold0", "fold1", "fold2", "fold3", "fold4", "merge", "posterior",
                 "column", "forward", "ring"):
        p = PAR8 / "done" / name
        markers[name] = _utc(p) if p.exists() else None
    deps = [ln.strip() for ln in JOBS.read_text().splitlines()
            if ln.strip().startswith("[merge]=") or "STAGE_M=" in ln
            or ln.strip().startswith("job_merge()")]
    qlines = [ln.rstrip() for ln in QLOG.read_text(errors="replace").splitlines()
              if "batch 8" in ln or "par8/merge.log" in ln or "merge FAILED" in ln]
    merge_log = (PAR8 / "merge.log").read_text(errors="replace").strip().splitlines()[-1:]
    try:
        n_twin = len(subprocess.run(["pgrep", "-f", "hydrophysics.twin"], capture_output=True,
                                    text=True).stdout.split())
    except OSError:
        n_twin = None
    theta_mtime = _utc(THETA)
    finished = [ln for ln in qlines if "##### batch 8 finished" in ln]
    t_v1 = _dt.datetime.strptime(V1_BUILD_START_UTC, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=_dt.timezone.utc)
    t_theta = _dt.datetime.strptime(theta_mtime, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=_dt.timezone.utc)
    merge_done = markers["merge"]
    t_merge = (_dt.datetime.strptime(merge_done, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=_dt.timezone.utc) if merge_done else None)
    ring_done = markers["ring"]
    t_ring = (_dt.datetime.strptime(ring_done, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=_dt.timezone.utc) if ring_done else None)
    out = {
        "theta": {"path": rel_to_repo(THETA), "sha256": sha256_file(THETA),
                  "sha256_equals_engine_v1": sha256_file(THETA) == V1_SHA,
                  "mtime_utc": theta_mtime,
                  "meta_gate": meta.get("gate"), "meta_git_commit": meta.get("git_commit"),
                  "meta_holdout_months": meta.get("holdout_months"),
                  "params_max_abs_diff_vs_full_fit_summary": _max_abs_diff(
                      th["theta"], fit_theta),
                  "fold_thetas_file": {"path": rel_to_repo(FOLD_THETAS),
                                       "mtime_utc": _utc(FOLD_THETAS),
                                       "n_folds": len(json.loads(FOLD_THETAS.read_text()))}},
        "batch8": {"done_markers_utc": markers,
                   "merge_dependencies_in_jobs_sh": deps,
                   "queue_log_lines": qlines,
                   "merge_log_last_line": merge_log,
                   "finished_line": finished[-1] if finished else None,
                   "twin_processes_running_now": n_twin},
        "timeline_hours": {
            "theta_written_to_v1_build": round((t_v1 - t_theta).total_seconds() / 3600, 2),
            "merge_marker_to_v1_build": (round((t_v1 - t_merge).total_seconds() / 3600, 2)
                                         if t_merge else None),
            "batch_end_to_v1_build": (round((t_v1 - t_ring).total_seconds() / 3600, 2)
                                      if t_ring else None)},
        "v1_build_start_utc": V1_BUILD_START_UTC,
        "checked_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ok = (out["theta"]["sha256_equals_engine_v1"] and meta.get("gate", {}).get("n_folds") == 5
          and out["theta"]["params_max_abs_diff_vs_full_fit_summary"] == 0.0
          and merge_done is not None and t_merge >= t_theta and finished
          and t_merge < t_v1)
    out["verdict"] = ("batch 8's merged theta: the merge job (deps fit_full + folds 0-4) "
                      "wrote it and its done marker before the v1 build; its meta carries the "
                      "5-fold gate and its parameters equal the full fit"
                      if ok else "NOT confirmed: see the fields above")
    out["is_batch8_merged_theta"] = bool(ok)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    res = collect()
    txt = json.dumps(res, ensure_ascii=False, indent=1) + "\n"
    if argv:
        Path(argv[0]).write_text(txt)
    print(txt)
    return 0 if res["is_batch8_merged_theta"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
