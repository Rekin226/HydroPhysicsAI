"""WP P1 repair round, resume check (2026-10-12, after a usage-limit interruption).

The repair-round session wrote every product, look and report file (10:41Z-10:52Z) but was cut
off before it returned its result. This script re-verifies, read-only, that what it left is
still sound, logs the two repair-round computations that the look log did not yet name, and
records the outcome:

* the A0 checker over ``results/dataprep/county_electricity`` (every manifest re-hashed);
* the sha256 of the P1 pre-registrations against ``INDEX.jsonl``;
* both A1 tools (``harness.prereg`` and the repaired ``harness.prereg2``, which A1 published at
  about 11:00Z, after the repair round) asked with ``log=False`` whether T10 and T11 could still
  be scored as pre-registered tests (they must refuse);
* ``harness.prereg2.audit`` entries that name P1;
* ledger conflicts over every fragment, and the P1 product_version normalisation;
* ruff findings in the repair-round files (left unfixed when a manifest hashes the file).

Nothing is re-scored and no product is written. Output: one evidence JSON.

Usage: python -I r3_resume_check.py <evidence_json> [--log-looks]
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import check, ledger  # noqa: E402
from common.paths import PREREG_DIR, REPO  # noqa: E402
from harness import prereg as pr1  # noqa: E402
from harness import prereg2 as pr2  # noqa: E402

PID = "P1r2.census_tests"
TESTS = ("P1r2.T10_ext_yunlin", "P1r2.T11_ext_changhua")
CE = REPO / "results/dataprep/county_electricity"
R3_FILES = sorted(HERE.glob("r3_*.py"))
TAG_T3 = "[P1-r3-late-4]"
TAG_RESUME = "[P1-r3-resume-1]"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_checker() -> dict:
    reports = [check.check_manifest(p) for p in sorted(CE.glob("*/v*/MANIFEST.json"))]
    return {"manifests": [{"path": str(r.get("path")), "status": r.get("status"),
                           "n_hashed": r.get("n_hashed"),
                           "problems": (r.get("schema_errors") or []) + (r.get("hash_problems")
                                                                         or [])}
                          for r in reports],
            "n_manifests": len(reports),
            "n_failed": sum(r.get("status") != "ok" for r in reports),
            "files_rehashed": sum(r.get("n_hashed") or 0 for r in reports)}


def prereg_hashes() -> dict:
    res = {}
    for row in pr1.read_chained(Path(PREREG_DIR) / "INDEX.jsonl"):
        if row["wp"] != "P1":
            continue
        f = REPO / row["path"]
        res[row["id"]] = {"sha256": row["sha256"],
                          "matches": hashlib.sha256(f.read_bytes()).hexdigest() == row["sha256"]}
    return res


def tool_verdicts() -> dict:
    res = {}
    for name, tool in (("harness.prereg", pr1), ("harness.prereg2", pr2)):
        for tid in TESTS:
            try:
                tool.assert_can_score(PID, "P1", test_id=tid, log=False)
                res[f"{name}:{tid}"] = {"refused": False}
            except pr1.PreregistrationError as e:
                res[f"{name}:{tid}"] = {"refused": True, "error": str(e)[:240]}
    return res


def audit_p1() -> list:
    a = pr2.audit()
    hits = []
    for key, val in a.items():
        if isinstance(val, list):
            for x in val:
                s = json.dumps(x, ensure_ascii=False, default=str)
                if "P1" in s:
                    hits.append({"section": key, "entry": x})
    return hits


def ledger_state() -> dict:
    rows = ledger.read_all()
    p1 = [r for r in rows if r.get("wp") == "P1"]
    return {"rows_all_fragments": len(rows), "conflicts": len(ledger.conflicts(rows)),
            "p1_rows": len(p1),
            "p1_repo_relative_product_version": sum(
                r.get("product_version", "").startswith("results/") for r in p1)}


def ruff_state() -> dict:
    out = subprocess.run([sys.executable, "-m", "ruff", "check", "--line-length", "100",
                          "--output-format", "concise", *map(str, R3_FILES)],
                         capture_output=True, text=True, cwd=REPO)
    found = [ln for ln in out.stdout.splitlines() if ln.startswith(str(HERE)) or ".py:" in ln]
    return {"files": [p.name for p in R3_FILES], "findings": found}


def log_looks() -> list:
    """Append the two records once (idempotent on the note tag)."""
    existing = {(lk.get("note") or "")[:24] for lk in pr1.read_chained(
        Path(PREREG_DIR) / "looks.jsonl") if lk.get("wp") == "P1"}
    done = []
    if not any(n.startswith(TAG_T3) for n in existing):
        pr2.record_look(
            ["tpc_d007012_ag_changhua_yunlin"], "tpc_d007012:2016-01..2025-12", "P1",
            "after_the_fact",
            note=f"{TAG_T3} logged late (resume check): r3_preorigin_v2.py at about "
                 "2026-10-12T10:45Z recomputed T3_BV_LOYO (backcast leave-one-year-out, "
                 "reported only) on TPC 農林漁牧 2016-2022 for preorigin/v2, and r3_build.py at "
                 "10:50Z recomputed the v1 T3 on 2016-2025 full-variant data to check the port "
                 "(identical to full/v1). TPC data, no registered unit; no gate, parameter or "
                 "threshold depends on either.")
        done.append(TAG_T3)
    if not any(n.startswith(TAG_RESUME) for n in existing):
        pr2.record_look(
            ["changhua_census_months_2025-07..2026-06", "changhua_census_vs_P1_v1",
             "yunlin_census_new_months_2025-07..2026-06", "yunlin_census_vs_P1_v1"],
            "*", "P1", "integrity_check",
            note=f"{TAG_RESUME} resume check after a usage-limit interruption: re-hash of the "
                 "county_electricity manifests, pre-registration hashes, both A1 tools asked "
                 "with log=False about T10/T11, ledger conflicts, ruff; the already published "
                 "evaluation numbers were re-read for the report; nothing re-scored.")
        done.append(TAG_RESUME)
    return done


def main(argv: list[str]) -> int:
    dest = Path(argv[0])
    started = _now()
    res = {"started": started,
           "checker": run_checker(),
           "prereg_hashes": prereg_hashes(),
           "tool_verdicts": tool_verdicts(),
           "prereg2_audit_P1": audit_p1(),
           "ledger": ledger_state(),
           "ruff_r3": ruff_state()}
    if "--log-looks" in argv:
        res["looks_appended"] = log_looks()
    res["finished"] = _now()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("ledger", "tool_verdicts")}, indent=1,
                     ensure_ascii=False)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
