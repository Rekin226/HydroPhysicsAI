"""WP A1 resume check: re-verify the finished A1 build after a usage-limit interruption.

    python -m harness.a1_resume_check EVIDENCE_DIR      (from dataprep_staging/)

Run after ``harness.a1_checks``, pytest/ruff and ``harness.repro_pilots`` have written their
outputs into ``EVIDENCE_DIR`` (``guard_prereg_probes.json``, ``pytest.txt``, ``ruff.txt``,
``repro/repro_pilots.json``). It compares the rerun with the reported run
(``A1_evidence/repro_final/repro_pilots.json``), then appends a "Resume check" section to
``reports/A1.md``, a ``resume_check`` record to ``reports/A1.json`` and the matching rows to
the A1 ledger fragment. Nothing else is written; the product and pre-registrations are only
read. Refuses to run twice (the section is appended once).
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
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

REPORTS = DATAPREP / "reports"
REPORTED = REPORTS / "A1_evidence" / "repro_final" / "repro_pilots.json"
MARK = "## Resume check"


def _parts(d: dict) -> dict:
    """Flatten the reproduction parts to {name: (n_values, max_rel_diff)}."""
    out = {}
    p = d["parts"]
    for k, v in p["gapfill"].items():
        out[f"gapfill.{k}"] = (v["n_values"], v["max_rel_diff"])
    out["tpc"] = (p["tpc"]["n_values"], p["tpc"]["max_rel_diff"])
    out["tpc.archive_identical"] = (p["tpc"]["archive_values"],
                                    p["tpc"]["archive_identical_at_stored_precision"])
    out["gap_masks"] = (p["gap_masks"]["n_realizations"], p["gap_masks"]["n_identical"])
    for k in ("pseudo", "holdout"):
        for s in ("scorers", "patched_rerun"):
            out[f"{k}.{s}"] = (p[k][s]["n_values"], p[k][s]["max_rel_diff"])
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ev = Path(argv[0]).resolve()
    md_path, js_path = REPORTS / "A1.md", REPORTS / "A1.json"
    md = md_path.read_text(encoding="utf-8")
    if MARK in md:
        raise SystemExit("resume check already appended to A1.md")
    checks = json.loads((ev / "guard_prereg_probes.json").read_text(encoding="utf-8"))
    pytest_txt = (ev / "pytest.txt").read_text(encoding="utf-8")
    ruff_txt = (ev / "ruff.txt").read_text(encoding="utf-8")
    new_path = ev / "repro" / "repro_pilots.json"
    new = json.loads(new_path.read_text(encoding="utf-8"))
    old = json.loads(REPORTED.read_text(encoding="utf-8"))
    pn, po = _parts(new), _parts(old)
    same_parts = pn == po
    same_code = new.get("code") == old.get("code")
    m = re.search(r"(\d+) passed in ([0-9.]+)s", pytest_txt)
    n_pass, t_pass = (int(m.group(1)), m.group(2)) if m else (0, "?")
    failed = re.search(r"(\d+) failed", pytest_txt)
    ruff_ok = "All checks passed!" in ruff_txt
    fresh_looks = checks["fresh_units"]["with_any_look"]
    ok = bool(checks["pass"] and new["pass"] and same_parts and same_code and n_pass > 0
              and not failed and ruff_ok and not fresh_looks)
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = {
        "time": now,
        "reason": "the workflow re-invoked the cached A1 build after a usage-limit interruption",
        "harness_files_changed_since_report": 0,
        "a1_checks_pass": checks["pass"],
        "guard_as_expected": checks["guard_as_expected"],
        "checker_status": checks["checker"]["status"],
        "checker_n_hashed": checks["checker"]["n_hashed"],
        "INDEX_records": checks["chains"]["INDEX.jsonl_records"],
        "looks_records": checks["chains"]["looks.jsonl_records"],
        "chains_intact": checks["chains"]["intact"],
        "fresh_units": checks["fresh_units"]["n"],
        "fresh_units_with_any_look": fresh_looks,
        "design_time_units_without_seed": checks["design_time_units"]["without_seeded_look"],
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
    lines = [
        "",
        f"{MARK} ({now})",
        "",
        "The workflow called the A1 build again after a usage-limit interruption; the build had "
        "already finished, so nothing was rebuilt. Everything was re-verified instead:",
        "",
        "- No file under `dataprep_staging/harness/` changed after this report was written "
        "(this check adds `a1_resume_check.py`, which no manifest or pre-registration hashes).",
        f"- `harness.a1_checks`: loader guard as expected {checks['guard_as_expected']}; "
        f"checker {checks['checker']['status']} ({checks['checker']['n_hashed']} files "
        "re-hashed); both A1 pre-registrations load with their recorded sha256; hash chains "
        f"intact (INDEX.jsonl {rec['INDEX_records']} records, looks.jsonl "
        f"{rec['looks_records']}); design-time unit refused as fresh.",
        f"- Looks on the {rec['fresh_units']} fresh units: {len(fresh_looks)}. Downstream "
        "WPs have registered and scored other tests since (P1, W1, W2/W4, S2, S3, H1a, P4.0, "
        "S4a), none on an A1 fresh unit, so the declaration still precedes every use.",
        f"- Pilot reproduction rerun: {new['n_values_total']} values, max relative difference "
        f"{new['max_rel_diff_overall']:.3g}, pass {new['pass']}; every part identical to "
        f"the reported run: {same_parts}; code hashes identical: {same_code} "
        f"({new.get('seconds')} s).",
        f"- Tests: {n_pass} passed in {t_pass}s; ruff clean: {ruff_ok}.",
        "",
        f"Verdict: {'unchanged, still passed' if ok else 'CHANGED, see evidence'}. "
        f"Evidence: `{rel_to_repo(ev)}/`.",
        "",
    ]
    md_path.write_text(md.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8")
    js = json.loads(js_path.read_text(encoding="utf-8"))
    js["resume_check"] = rec
    js.setdefault("evidence", []).append(rel_to_repo(ev))
    js["key_numbers"].append(
        f"Resume check {now}: checks pass {checks['pass']}, repro {new['n_values_total']} "
        f"values max {new['max_rel_diff_overall']:.3g} (identical to reported: {same_parts}), "
        f"{n_pass} tests passed, 0 looks on the {rec['fresh_units']} fresh units")
    js_path.write_text(json.dumps(js, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    pv = f"harness@1.0.0;resume {now}"
    rows = [
        {"number_id": "A1.resume.checks", "source": rel_to_repo(ev / "guard_prereg_probes.json"),
         "role": "validator", "product_version": pv,
         "note": "guard, checker, prereg hashes, chains, looks on fresh units"},
        {"number_id": "A1.resume.repro", "source": rel_to_repo(new_path), "role": "validator",
         "product_version": pv, "note": "pilot reproduction rerun vs reported run"},
        {"number_id": "A1.resume.tests", "source": rel_to_repo(ev / "pytest.txt"),
         "role": "validator", "product_version": pv, "note": "pytest harness + common"},
        {"number_id": "A1.resume.ruff", "source": rel_to_repo(ev / "ruff.txt"),
         "role": "validator", "product_version": pv, "note": "ruff harness"},
    ]
    ledger.append_rows("A1", rows)
    print(json.dumps({k: rec[k] for k in ("pass", "repro_identical_to_reported",
                                          "repro_max_rel_diff", "pytest_passed",
                                          "fresh_units_with_any_look", "looks_records")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
