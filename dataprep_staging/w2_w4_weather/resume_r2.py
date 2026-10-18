"""Resume round 2 of WP W2W4 (2026-10-12, after a second usage-limit interruption).

The first rounds built et0/v1 and rain_check/v1 and wrote the report at about 10:34Z, but did
not return. Since then WP A1's repair round (report 11:07Z) replaced the pre-registration tool
by ``harness.prereg2`` and its audit flagged two W2W4 items: looks logged under the id
``W4.rain_kriging_vs_idw.v1`` (a file written by this WP's prereg.py, not in INDEX.jsonl), and
three W2/W4 pre-registration files written outside the tool. This script, read-only except for
the look log and one new evidence file:

1. re-runs this WP's checker (check_w2w4.main) into a new evidence file;
2. applies ``prereg2`` to the W2W4 records: ``assert_can_score(..., log=False)`` on
   W2.et0_overlap.v3 (no new scoring look: nothing is re-scored), ``load_prereg`` and
   ``changed_inputs`` on the W4 declaration, sha256 of the three files outside INDEX against
   the ledger, and the W2W4 subset of ``prereg2.audit()``;
3. with ``--apply``: re-attributes each look logged under the unregistered id to the A1
   declaration ``W4.rain_kriging_vs_idw.v1.record`` with ``prereg2.record_look`` (purpose
   ``unregistered_score`` for the scores, ``reproduce`` for the rerun; the note names the
   original record and its real time), and logs one ``resume_review`` look for this round's
   re-reading of the products (report regeneration re-computes the probe-variant table and the
   review-probe check; no test is scored). Idempotent: a re-attribution already in the log is
   not written again.

Usage: python -I isorun.py resume_r2.py --out <evidence json> [--apply]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys

import check_w2w4
from common import ledger
from harness import prereg2 as P2
from wcommon import PREREG_DIR, REPORTS_DIR, WP, sha256_file, write_json_new

EVID = REPORTS_DIR / "W2W4_evidence"
W2_TEST = "W2.et0_overlap.v3"
W4_DECL = "W4.rain_kriging_vs_idw.v1.record"
W4_OLD = "W4.rain_kriging_vs_idw.v1"
OUTSIDE = ("W2.et0_overlap.v1", "W2.et0_overlap.v2", "W4.rain_kriging_vs_idw.v1")
PURPOSE_MAP = {"score": "unregistered_score", "reproduce": "reproduce"}
GAUGES = [f"rain_gauge_rf{i}" for i in range(1, 27)]
TAG = "re-attribution of look record"


def _line_hashes() -> dict:
    """sha256 of each raw line of looks.jsonl -> its record (the chain's 'prev' of the next)."""
    out = {}
    path = PREREG_DIR / "looks.jsonl"
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            out[hashlib.sha256(ln.encode()).hexdigest()] = json.loads(ln)
    return out


def w2w4_audit() -> dict:
    a = P2.audit()
    mine = {W2_TEST, W4_DECL, *OUTSIDE}
    return {
        "index_records": a["index_records"], "looks_records": a["looks_records"],
        "tests_refused_under_repaired_rule": [r for r in a["tests_refused_under_repaired_rule"]
                                              if r["wp"] == WP],
        "tests_with_changed_inputs": [r for r in a["tests_with_changed_inputs"]
                                      if r["wp"] == WP],
        "index_records_not_loadable": [r for r in a["index_records_not_loadable"]
                                       if r["id"] in mine],
        "looks_with_unknown_prereg_id": [r for r in a["looks_with_unknown_prereg_id"]
                                         if r["wp"] == WP],
        "prereg_files_not_in_index": [f for f in a["prereg_files_not_in_index"]
                                      if f.startswith(("W2.", "W4."))],
    }


def prereg_checks() -> dict:
    out = {}
    try:
        r = P2.assert_can_score(W2_TEST, WP, log=False)
        out[W2_TEST] = {"assert_can_score_prereg2": "ok (log=False: no scoring look written)",
                        "sha256": r["sha256"], "tests": r["tests"],
                        "inputs_verified": r["inputs_verified"],
                        "post_registration_looks_other_tests": r["post_registration_looks"]}
    except P2.PreregistrationError as exc:
        out[W2_TEST] = {"assert_can_score_prereg2": f"refused: {exc}"}
    try:
        doc, sha = P2.load_prereg(W4_DECL)
        out[W4_DECL] = {"load_prereg": "ok", "kind": doc["kind"], "sha256": sha,
                        "changed_inputs": P2.changed_inputs(doc)}
    except P2.PreregistrationError as exc:
        out[W4_DECL] = {"load_prereg": f"refused: {exc}"}
    rows = ledger.read_all()
    for pid in OUTSIDE:
        p = PREREG_DIR / f"{pid}.json"
        sha = sha256_file(p)
        rec = [r["product_version"] for r in rows if r["number_id"] == f"{WP}.prereg.{pid}"]
        doc = json.loads(p.read_text(encoding="utf-8"))
        out[pid] = {"written_by": "dataprep_staging/w2_w4_weather/prereg.py (outside the A1 "
                                  "tool)", "sha256": sha,
                    "ledger_sha256_matches": f"prereg-sha256:{sha}" in rec,
                    "changed_inputs": P2.changed_inputs(doc)}
    return out


def reattribute(apply: bool) -> list[dict]:
    lines = _line_hashes()
    looks = P2.read_chained(PREREG_DIR / "looks.jsonl")
    done = {lk.get("note", "").split(TAG + " ", 1)[1].split(" ", 1)[0]
            for lk in looks if TAG in (lk.get("note") or "")}
    # each record's own line hash = the 'prev' of the record that follows it
    own = {json.dumps(v, sort_keys=True): k for k, v in lines.items()}
    out = []
    for lk in looks:
        if lk.get("wp") != WP or lk.get("prereg_id") != W4_OLD:
            continue
        h = own[json.dumps(lk, sort_keys=True)]
        purpose = PURPOSE_MAP.get(lk["purpose"], lk["purpose"])
        note = (f"{TAG} {h} (looks.jsonl line sha256): originally logged at {lk['logged']} "
                f"with purpose {lk['purpose']!r}, test id {lk.get('test_id')!r} and "
                f"pre-registration id {W4_OLD!r}, a file written by this WP's prereg.py that "
                f"is not in INDEX.jsonl (A1 repair-round audit). Same look, now attributed to "
                f"the A1 declaration {W4_DECL!r}; real time as in the original note: "
                f"{lk.get('note', '')}")
        item = {"original_line_sha256": h, "original_time": lk["time"],
                "original_purpose": lk["purpose"], "data_scope": lk["data_scope"],
                "new_purpose": purpose, "already_logged": h in done}
        if apply and h not in done:
            rec = P2.record_look(lk["units"], lk["data_scope"], WP, purpose,
                                 prereg_id=W4_DECL, note=note)
            item["logged"] = rec["logged"]
        out.append(item)
    return out


def resume_look(apply: bool) -> dict | None:
    note = ("resume round 2 (after a usage-limit interruption): re-read the published et0/v1 "
            "and rain_check/v1 outputs and the first rounds' report, re-ran the hash checker, "
            "and regenerated reports/W2W4.{md,json} with report.py, which re-computes the "
            "January-2022 probe-variant table and the review-probe check from L0. No test was "
            "re-scored and no product was rebuilt.")
    looks = P2.read_chained(PREREG_DIR / "looks.jsonl")
    if any(lk.get("wp") == WP and lk.get("purpose") == "resume_review" for lk in looks):
        return {"already_logged": True}
    units = ["openmeteo_et0_main_vs_fe_gridcells_61pts",
             "openmeteo_et0_main_2025-01..2025-06_61pts",
             "openmeteo_et0_refetch_2026-07..2026-09_61pts",
             "et0_splice_2023-01..2026-09_61pts",
             "openmeteo_et0_probe_st1_2022_local_and_variants", *GAUGES]
    if not apply:
        return {"would_log": True, "units": len(units)}
    rec = P2.record_look(units, "w2w4_products:2012-01..2026-09", WP, "resume_review",
                         note=note)
    return {"logged": rec["logged"], "units": len(units)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    chk_path = EVID / "checker_v1_resume2.json"
    rc = 0
    if not chk_path.exists():
        rc = check_w2w4.main(["--et0", "results/dataprep/et0/v1", "--out", str(chk_path)])
    chk = json.loads(chk_path.read_text())
    before = w2w4_audit()
    pre = prereg_checks()
    re_att = reattribute(a.apply)
    rl = resume_look(a.apply)
    rep = {"label": "W2W4 resume round 2: checker rerun, prereg2 checks, look re-attribution",
           "harness_prereg2": {"tool": P2.TOOL, "version": P2.TOOL_VERSION},
           "checker": {"path": chk_path.relative_to(REPORTS_DIR.parents[2]).as_posix(),
                       "status": chk["status"],
                       "manifests": [(m["path"], m["status"]) for m in chk["manifests"]],
                       "n_sources": len(chk["sources"]),
                       "orphans_openmeteo_et0": chk["orphans_openmeteo_et0"]},
           "audit_w2w4_before": before, "prereg_checks": pre,
           "reattributed_looks": re_att, "resume_look": rl, "applied": a.apply}
    if a.apply:
        rep["audit_w2w4_after"] = w2w4_audit()
    write_json_new(a.out, rep)
    print(json.dumps(rep, indent=1, default=str)[:6000])
    return rc


if __name__ == "__main__":
    sys.exit(main())
