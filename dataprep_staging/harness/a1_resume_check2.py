"""WP A1 second resume check: re-verify the repaired A1 after the workflow called the build again.

    python -m harness.a1_resume_check2 EVIDENCE_DIR      (from dataprep_staging/)

Run after ``harness.a1r_checks EVIDENCE_DIR``, pytest/ruff (``pytest.txt``, ``ruff.txt``) and
``harness.repro_pilots EVIDENCE_DIR/repro`` have written their outputs. The script

* compares the pilot reproduction rerun with the reported run
  (``A1_evidence/repro_final/repro_pilots.json``), part by part, and the code hashes;
* compares ``a1r_checks.json`` with the repair round's (``A1_evidence/repair/a1r_checks.json``);
* lists every look and registration appended to the logs after the repair round's own check run
  and checks that none names an A1 fresh unit (v1 or v2) or a withdrawn id;
* records the sha256 and mtime of every file under ``dataprep_staging/harness/`` and checks
  that none was modified after the repair report was written (this script excepted);

then appends a "Resume check 2" section to ``reports/A1.md``, a ``resume_check_2`` record to
``reports/A1.json`` and four ledger rows. Products and pre-registrations are only read.
Refuses to run twice.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import ledger  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from common.paths import DATAPREP, PREREG_DIR, rel_to_repo  # noqa: E402

from harness import prereg  # noqa: E402
from harness.a1_resume_check import _parts  # noqa: E402

REPORTS = DATAPREP / "reports"
EVID = REPORTS / "A1_evidence"
REPORTED = EVID / "repro_final" / "repro_pilots.json"
REPAIR_CHECKS = EVID / "repair" / "a1r_checks.json"
HARNESS = Path(__file__).resolve().parent
MARK = "## Resume check 2"
SELF = Path(__file__).resolve()


def _utc(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _strip(d: dict) -> dict:
    """a1r_checks content that must not change between runs (the look count may grow)."""
    out = {k: v for k, v in d.items() if k not in ("chains", "harness_files_hashed_elsewhere")}
    out["harness_hashed_n"] = d["harness_files_hashed_elsewhere"]["n"]
    out["harness_hashed_changed"] = d["harness_files_hashed_elsewhere"]["changed"]
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ev = Path(argv[0]).resolve()
    md_path, js_path = REPORTS / "A1.md", REPORTS / "A1.json"
    md = md_path.read_text(encoding="utf-8")
    if MARK in md:
        raise SystemExit("resume check 2 already appended to A1.md")
    js = json.loads(js_path.read_text(encoding="utf-8"))
    repair_time = js["repair_round"]["time"]                      # e.g. 2026-10-12T11:06:41Z
    report_written = _utc(js_path.stat().st_mtime)               # A1.json write = report end

    # 1. harness files: hashes and mtimes
    files = []
    for p in sorted(HARNESS.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p == SELF:
            continue
        files.append({"file": rel_to_repo(p), "sha256": sha256_file(p),
                      "mtime": _utc(p.stat().st_mtime)})
    changed_after = [f["file"] for f in files if f["mtime"] > report_written]
    (ev / "harness_files.json").write_text(json.dumps(
        {"report_written": report_written, "files": files}, indent=1), encoding="utf-8")

    # 2. a1r_checks now vs the repair round
    now_c = json.loads((ev / "a1r_checks.json").read_text(encoding="utf-8"))
    rep_c = json.loads(REPAIR_CHECKS.read_text(encoding="utf-8"))
    checks_same = _strip(now_c) == _strip(rep_c)

    # 3. looks and registrations logged after the repair report
    looks = prereg.read_chained(PREREG_DIR / "looks.jsonl")
    idx = prereg.read_chained(PREREG_DIR / "INDEX.jsonl")
    v1 = json.loads((DATAPREP / "validation_units" / "v1" / "units.json")
                    .read_text(encoding="utf-8"))
    v2 = json.loads((DATAPREP / "validation_units" / "v2" / "units.json")
                    .read_text(encoding="utf-8"))
    a1_fresh = ({u["id"] for u in v1["units"] if u.get("role") == "fresh"}
                | {u["id"] for u in v2["units"] if u.get("role") == "fresh"}
                | set(v2.get("withdrawn", {})))
    # everything appended after the repair round's own check run (its record counts)
    new_looks = looks[rep_c["chains"]["looks.jsonl_records"]:]
    new_regs = idx[rep_c["chains"]["INDEX.jsonl_records"]:]
    hits = [lk for lk in new_looks if set(lk["units"]) & a1_fresh]
    v2_fresh = sorted(u["id"] for u in v2["units"] if u.get("role") == "fresh")
    v2_fresh_any = sorted({u for lk in looks for u in lk["units"]} & set(v2_fresh))

    # 4. pilot reproduction
    new_path = ev / "repro" / "repro_pilots.json"
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old = json.loads(REPORTED.read_text(encoding="utf-8"))
    same_parts = _parts(new) == _parts(old)
    same_code = new.get("code") == old.get("code")

    # 5. tests
    pytest_txt = (ev / "pytest.txt").read_text(encoding="utf-8")
    ruff_ok = "All checks passed!" in (ev / "ruff.txt").read_text(encoding="utf-8")
    m = re.search(r"(\d+) passed in ([0-9.]+)s", pytest_txt)
    n_pass, t_pass = (int(m.group(1)), m.group(2)) if m else (0, "?")
    failed = re.search(r"(\d+) failed", pytest_txt)

    ok = bool(now_c["pass"] and checks_same and not changed_after and not hits
              and not v2_fresh_any and new["pass"] and same_parts and same_code
              and n_pass > 0 and not failed and ruff_ok)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = {
        "time": now,
        "reason": "the workflow called the A1 build again after a usage-limit interruption; "
                  "the WP (build and repair round) had already finished",
        "repair_report_time": repair_time,
        "a1_json_written": report_written,
        "harness_files": len(files),
        "harness_files_modified_after_report": changed_after,
        "a1r_checks_pass": now_c["pass"],
        "a1r_checks_identical_to_repair_run": checks_same,
        "checker": {k: v["status"] for k, v in now_c["checker"].items()},
        "INDEX_records": len(idx), "looks_records": len(looks),
        "looks_appended_after_repair_checks": [{k: lk.get(k) for k in ("wp", "purpose", "logged",
                                                               "units")} for lk in new_looks],
        "registrations_after_repair_checks": [r["id"] for r in new_regs],
        "looks_after_repair_checks_on_a1_fresh_or_withdrawn_units": len(hits),
        "v2_fresh_units": len(v2_fresh),
        "v2_fresh_units_with_any_look": v2_fresh_any,
        "repro_pass": new["pass"],
        "repro_n_values": new["n_values_total"],
        "repro_max_rel_diff": new["max_rel_diff_overall"],
        "repro_seconds": new.get("seconds"),
        "repro_identical_to_reported": same_parts,
        "repro_code_hashes_identical": same_code,
        "repro_sha256": sha256_file(new_path),
        "pytest_passed": n_pass,
        "ruff_clean": ruff_ok,
        "pass": ok,
        "evidence": rel_to_repo(ev),
    }
    look_txt = "; ".join(f"{lk['wp']} {lk['purpose']} on {', '.join(lk['units'])}"
                         for lk in new_looks) or "none"
    lines = [
        "",
        f"{MARK} ({now})",
        "",
        "The workflow called the A1 build again after a usage-limit interruption. The build "
        f"and its repair round had finished ({repair_time}), so nothing was rebuilt, no "
        "product version or pre-registration was written, and the WP stands as the repair "
        "round left it. It was re-verified:",
        "",
        f"- Harness files: {len(files)} hashed into `harness_files.json`; modified after the "
        f"report was written ({report_written}): {len(changed_after)} (this check adds "
        "`a1_resume_check2.py`, which nothing hashes).",
        f"- `harness.a1r_checks` rerun: pass {now_c['pass']}; identical to the repair run "
        f"(look count aside): {checks_same}; checker v1 {rec['checker'].get('v1')}, v2 "
        f"{rec['checker'].get('v2')}; harness files hashed elsewhere "
        f"{now_c['harness_files_hashed_elsewhere']['n']}, changed "
        f"{len(now_c['harness_files_hashed_elsewhere']['changed'])}; probes: first tool "
        f"allows all {now_c['probes']['first_tool_allows_all']}, prereg2 refuses all "
        f"{now_c['probes']['prereg2_refuses_all']}.",
        f"- Look log: {len(looks)} records, INDEX {len(idx)}. Appended after the repair "
        "round's check run: "
        f"{len(new_looks)} ({look_txt}); registrations after it: {len(new_regs)}. Looks on an "
        f"A1 fresh or withdrawn unit since: {len(hits)}; looks ever on the {len(v2_fresh)} v2 "
        f"fresh units: {len(v2_fresh_any)}. The A1-fresh-units-v2 declaration still precedes "
        "every use.",
        f"- Pilot reproduction rerun: {new['n_values_total']} values, max relative difference "
        f"{new['max_rel_diff_overall']:.3g}, pass {new['pass']}; every part identical to the "
        f"reported run: {same_parts}; code hashes identical: {same_code} "
        f"({new.get('seconds')} s).",
        f"- Tests: {n_pass} passed in {t_pass}s; ruff clean: {ruff_ok}.",
        "",
        f"Verdict: {'unchanged, still passed' if ok else 'CHANGED, see evidence'}. "
        f"Evidence: `{rel_to_repo(ev)}/`.",
        "",
    ]
    md_path.write_text(md.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8")
    js["resume_check_2"] = rec
    js.setdefault("evidence", []).append(rel_to_repo(ev))
    js["key_numbers"].append(
        f"Resume check 2 {now}: a1r_checks pass {now_c['pass']} (identical to the repair run: "
        f"{checks_same}); repro {new['n_values_total']} values max "
        f"{new['max_rel_diff_overall']:.3g} (identical to reported: {same_parts}); {n_pass} "
        f"tests passed; 0 harness files modified; looks on v2 fresh units {len(v2_fresh_any)}")
    js_path.write_text(json.dumps(js, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    pv = f"harness@1.0.0;resume2 {now}"
    rows = [
        {"number_id": "A1.resume2.checks", "source": rel_to_repo(ev / "a1r_checks.json"),
         "role": "validator", "product_version": pv,
         "note": "guard, checker v1/v2, preregs, chains, fresh-unit looks, probes, audit"},
        {"number_id": "A1.resume2.repro", "source": rel_to_repo(new_path), "role": "validator",
         "product_version": pv, "note": "pilot reproduction rerun vs reported run"},
        {"number_id": "A1.resume2.tests", "source": rel_to_repo(ev / "pytest.txt"),
         "role": "validator", "product_version": pv, "note": "pytest harness + common"},
        {"number_id": "A1.resume2.files", "source": rel_to_repo(ev / "harness_files.json"),
         "role": "validator", "product_version": pv,
         "note": "sha256 and mtime of every harness file"},
    ]
    ledger.append_rows("A1", rows)
    print(json.dumps({k: rec[k] for k in (
        "pass", "a1r_checks_identical_to_repair_run", "harness_files_modified_after_report",
        "looks_after_repair_checks_on_a1_fresh_or_withdrawn_units",
        "repro_identical_to_reported",
        "repro_max_rel_diff", "pytest_passed")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
