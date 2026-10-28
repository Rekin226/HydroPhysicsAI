"""Write results/twin_runs/par28 for rung I1.X5CW.v1, choosing the base by the registered rule
(``base_rule`` in results/dataprep/preregistration/I1.X5CW.v1.json) from the written-once
report of I1.X5R1.v1. Refuses to run before that report exists or if par28 exists.

Usage (repo root): python dataprep_staging/x5_changhua_water/make_batch28.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "results/twin_runs"
P27, P28 = RUNS / "par27", RUNS / "par28"
REPORT = REPO / "results/dataprep/reports/I1_X5R1/report.json"
CW = "results/dataprep/x5_changhua_water/preorigin/v1"


def base_recipe() -> tuple[str, str, str]:
    rep = json.loads(REPORT.read_text())
    sha = hashlib.sha256(REPORT.read_bytes()).hexdigest()
    p = rep["passes_all"]
    base = "x5r1" if p["x5r1"] else ("x5r1t" if p["x5r1t"] else "q0")
    jobs27 = (P27 / "jobs.sh").read_text()
    if base == "q0":
        rec = '("${BASE[@]}")'
    else:
        m = re.search(rf"^RECIPE_{base}=(\(.*\))$", jobs27, re.M)
        rec = m.group(1)
    return base, rec, sha


def main() -> int:
    if P28.exists():
        raise SystemExit(f"{P28} exists")
    base, rec, sha = base_recipe()
    jobs27 = (P27 / "jobs.sh").read_text()
    head = jobs27[jobs27.index("R=results/twin_runs"):jobs27.index("RECIPE_q0c=")]
    tail = jobs27[jobs27.index("_rec() {"):]
    P28.mkdir()
    for f in ("controller.sh", "worker.sh"):
        (P28 / f).write_text((P27 / f).read_text().replace("par27", "par28")
                             .replace("ctl27", "ctl28").replace("batch 27", "batch 28"))
    (P28 / "py.sh").write_text((P27 / "py.sh").read_text())
    jobs = (
        "# Batch 28: rung I1.X5CW.v1 (route 4, Changhua water outside the Jiji supply).\n"
        f"# Base chosen by the registered rule: {base} (I1_X5R1/report.json sha256 {sha}).\n"
        "# cws = base + service-map field (primary); cwp = base + all-paddy field (alternative).\n"
        "# Code: main 5f3af18, worktree ~/HydroPhysicsAI-x5run (py.sh). CODE FREEZE until\n"
        "# '[ctl28] ##### batch 28 finished' appears in results/twin_runs/gap_queue.log.\n"
        + head + f"CW={CW}\nRECIPE_base={rec}\n"
        'RECIPE_cws=("${RECIPE_base[@]}" --sw-recharge $CW/x5_t2_service.npz '
        "--sw-components delivered)\n"
        'RECIPE_cwp=("${RECIPE_base[@]}" --sw-recharge $CW/x5_t2_paddy.npz '
        "--sw-components delivered)\n"
        "ARMS=(cws cwp)\nSTAGE_A=()\n"
        'for a in "${ARMS[@]}"; do STAGE_A+=(${a}_fit ${a}_f0 ${a}_f1 ${a}_f2 ${a}_f3 ${a}_f4 '
        "${a}_temporal); done\n"
        "STAGE_M=(cws_merge cwp_merge)\nSTAGE_B=(cws_column cwp_column)\n"
        "STAGE_C=(cws_forward cwp_forward)\nSTAGE_D=()\ndeclare -A DEPS=()\n"
        'for a in "${ARMS[@]}"; do\n'
        '    DEPS[${a}_merge]="${a}_fit ${a}_f0 ${a}_f1 ${a}_f2 ${a}_f3 ${a}_f4"\n'
        '    DEPS[${a}_column]="${a}_merge"; DEPS[${a}_forward]="${a}_column"\ndone\n' + tail)
    (P28 / "jobs.sh").write_text(jobs)
    (P28 / "base.json").write_text(json.dumps({"base": base, "x5r1_report_sha256": sha},
                                              indent=1) + "\n")
    for f in P28.glob("*.sh"):
        f.chmod(0o755)
    print(base, P28)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
