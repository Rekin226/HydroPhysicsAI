"""WP P1 repair round: add the resume check to ``reports/P1.json`` (idempotent).

``r3_report.py`` wrote P1.json at 10:50Z; the session was then cut off by a usage limit before
it returned. This script adds the resume check (``P1_evidence/r3/resume_check.json``, written by
``r3_resume_check.py``) and corrects two round-2 downstream notes that the repair round had
superseded (the domestic sensitivity and the ledger row count). Nothing else changes.

Usage: python -I r3_report_resume.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.paths import REPO  # noqa: E402

REPORT = REPO / "results/dataprep/reports/P1.json"
EVID = REPO / "results/dataprep/reports/P1_evidence/r3/resume_check.json"
TAG = "R3 resume re-verification"


def main() -> int:
    d = json.loads(REPORT.read_text(encoding="utf-8"))
    ev = json.loads(EVID.read_text(encoding="utf-8"))
    if any(a["criterion"].startswith(TAG) for a in d["acceptance"]):
        print("already patched")
        return 0
    ck, lg, tv = ev["checker"], ev["ledger"], ev["tool_verdicts"]
    refused = all(v["refused"] for v in tv.values())
    hashes_ok = all(v["matches"] for v in ev["prereg_hashes"].values())
    ruff = ev["ruff_r3"]["findings"]
    d["acceptance"].append({
        "criterion": f"{TAG} (after the usage-limit interruption, {ev['started']}): products, "
                     "pre-registrations, look log and ledger as the repair round left them",
        "threshold": "0 hash failures; P1 pre-registration sha256 equal INDEX; harness.prereg "
                     "and harness.prereg2 both refuse T10 and T11 (log=False); 0 ledger "
                     "conflicts",
        "measured": f"{ck['n_manifests']} manifests, {ck['files_rehashed']} files, "
                    f"{ck['n_failed']} failures; pre-registration hashes equal: {hashes_ok}; "
                    f"refused by both tools: {refused}; ledger {lg['rows_all_fragments']} rows, "
                    f"{lg['conflicts']} conflicts, P1 {lg['p1_rows']} rows",
        "passed": bool(ck["n_failed"] == 0 and hashes_ok and refused and lg["conflicts"] == 0)})
    d["key_numbers"].append(
        f"Resume check {ev['started']}: checker {ck['n_manifests']} manifests / "
        f"{ck['files_rehashed']} files / {ck['n_failed']} failures; T10 and T11 refused by "
        f"harness.prereg and harness.prereg2; ledger {lg['rows_all_fragments']} rows, "
        f"{lg['conflicts']} conflicts (P1 {lg['p1_rows']})")
    d["deviations_from_plan"].insert(
        6, f"Repair round: {len(ruff)} ruff findings (I001, import order after the sys.path "
           "set-up) in r3_build.py, r3_census_sensitivity.py and r3_preorigin_v2.py are left "
           "unfixed because census_check/v2 and preorigin/v2 hash these files; fix them at the "
           "Phase-1 port.")
    d["deviations_from_plan"].insert(
        7, "Repair round: the session was cut off by a usage limit after writing every file; "
           "a resume check (r3_resume_check.py) re-verified the state read-only and logged "
           "two more look records: [P1-r3-late-4] (the T3 recomputations of r3_preorigin_v2.py "
           "and r3_build.py on TPC data, not logged before) and [P1-r3-resume-1].")
    iss = d["issues_for_downstream"]
    for i, s in enumerate(iss):
        if s.startswith("Phase-0 status report") and "[repair round" not in s:
            iss[i] = s + (" [repair round: 0.75 removes domestic from Changhua only; like for "
                          "like (both counties without domestic, 2016-22) the ratio is 0.795]")
        if s.startswith("ledger/data_roles.csv") and "[repair round" not in s:
            iss[i] = s + (f" [repair round: rows/P1.csv now has {lg['p1_rows']} rows, "
                          f"{lg['conflicts']} conflicts over {lg['rows_all_fragments']} rows of "
                          "all fragments; data_roles.csv still not reassembled]")
    iss.insert(7, "Repair round, WP A1 / every scorer: harness.prereg2 (A1 repair, about 11:00Z) "
                  "also refuses T10 and T11, and its audit lists the round-1 pre-registration "
                  "P1_county_electricity_2026-10-12 as written outside the tool (the tool did "
                  "not exist at 02:36Z; disclosed in round 1). Its scores (BV1, BV2, T4, T8, "
                  "T9) stand as made; any re-score needs a new registration under prereg2.")
    for name in ("r3_resume_check.py", "r3_report_resume.py"):
        p = str(HERE / name)
        if p not in d["code_paths"]:
            d["code_paths"].insert(5, p)
    d["summary"] += (" A resume check after a usage-limit interruption (" + ev["started"] +
                     ") re-hashed the 6 county_electricity manifests (251 files, 0 failures), "
                     "confirmed that both A1 tools (prereg and the repaired prereg2) refuse T10 "
                     "and T11, found 0 ledger conflicts, and logged the two repair-round "
                     "computations the look log had not named.")
    REPORT.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("patched", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
