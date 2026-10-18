"""WP C0 repair round, resume after a usage-limit interruption (2026-10-14): read-only check of
the v3 work left by the interrupted session, before its products are finalised.

The interrupted session built v3 (``repair3_build.py``, ``repair3_assemble.py``), registered
``C0.census_product.v3`` and scored every v3 test (``repair3_checks.py``); it stopped before
``repair3_finalize.py`` (no MANIFEST, card or validation report in the v3 directories), the
ledger rows and the report. This script re-verifies, without rebuilding or re-scoring anything:

1. ``harness.prereg3.assert_can_score(..., log=False)`` on C0.census_product.v3: the registered
   file and every pinned input (L0 parquet, code, v1/v2/v3 product files) are unchanged and no
   look predates the registration on its test units;
2. every v3 check file: test id, pre-registration sha256, ``inputs_verified``, the sha256 of
   the checks code at scoring time equal to the current ``repair3_checks.py``, pass value equal
   to ``summary.json``; and the archived checks code equals the current file;
3. every file of the v3 product directories is pinned in the pre-registration with the same
   sha256, and no MANIFEST/validation/card exists yet (nothing published);
4. the v1 and v2 manifests still re-hash (``common.check.check_manifest``);
5. ``prereg3.audit()``: C0 tests refused under the repaired rule.

``python -I repair3_resume.py log``    log the resume look (before reading anything further)
``python -I repair3_resume.py check``  run the checks; writes
                                        reports/C0_evidence/repair3/resume_check.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from common.check import check_manifest  # noqa: E402
from common.hashing import sha256_file  # noqa: E402
from harness import prereg3  # noqa: E402

PREREG_ID = "C0.census_product.v3"
EV3 = lib.EVIDENCE / "repair3"
CHECKS = EV3 / "checks"
OUT = EV3 / "resume_check.json"
PE = lib.PRODUCT_ROOT
V3 = {k: PE / k / "v3" for k in ("census_L1", "preorigin", "full")}
PUBLISHED = [PE / k / v for k in ("census_L1", "preorigin", "full") for v in ("v1", "v2")]
TESTS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "T5")
UNITS = ["C0r3:census_snapshot_sensitivity", "C0r3:dec2022_fill_vs_jan2023_half",
         "C0r3:fill_backtest_loyo_2012_2021", "C0r3:perturbation_invariance",
         "C0r3:v3_county_set", "C0r3:v3_descriptive", "C0r3:v3_masks_internal",
         "C0r3:v3_privacy", "C0r3:v3_variant_consistency", "C0r3:v3_vs_v1_decomposition",
         "C0r3:v3_yunlin_vs_cached_forcing"]


def log_look() -> None:
    prereg3.record_look(
        UNITS, "c0_products_v3:2007-01..2026-07", lib.WP, "integrity_check",
        note="[C0-repair3-resume-1] resume session 2026-10-14 after a usage-limit interruption "
             "(the v3 tests were all scored by the interrupted session, 08:34-08:43Z on "
             "2026-10-13). Before this record the session listed the evidence tree and read "
             "repair/checks/summary.json + T5.json (v2) and repair3/checks/summary.json, T5.json "
             "and R3.json (v3), the v2 manifest/card and the v3 pre-registration. From here: "
             "read-only integrity check (repair3_resume.py check), then the remaining v3 check "
             "files are read to write validation.json, the product cards and the report; "
             "nothing is rebuilt or re-scored.")
    print("logged [C0-repair3-resume-1]")


def check() -> dict:
    t0 = time.time()
    res: dict = {"time_utc": prereg3._now(), "prereg": PREREG_ID,
                 "purpose": "resume integrity check (read-only) before finalising v3"}
    try:
        ok = prereg3.assert_can_score(PREREG_ID, lib.WP, log=False, verify_inputs=True)
        res["prereg_status"] = {
            "allowed": True, "sha256": ok["sha256"], "tests": ok["tests"],
            "inputs_verified": ok["inputs_verified"],
            "post_registration_looks_other_wp": [x for x in ok["post_registration_looks"]
                                                 if x["wp"] != lib.WP],
            "n_post_registration_looks": len(ok["post_registration_looks"]),
            "n_unverifiable_prior_looks": len(ok["unverifiable_prior_looks"])}
        sha = ok["sha256"]
        doc = ok["prereg"]
    except prereg3.PreregistrationError as exc:
        res["prereg_status"] = {"allowed": False, "error": str(exc)[:800]}
        doc, sha = prereg3.load_prereg(PREREG_ID)
    code_sha = sha256_file(lib.CODE_DIR / "repair3_checks.py")
    summ = lib.load_json(CHECKS / "summary.json")
    per = {}
    for k in TESTS:
        p = CHECKS / f"{k}.json"
        if not p.exists():
            per[k] = {"exists": False}
            continue
        t = lib.load_json(p)
        per[k] = {"exists": True, "test_id": t.get("test_id"),
                  "prereg_sha_ok": t.get("prereg_sha256") == sha,
                  "inputs_verified": t.get("inputs_verified"),
                  "code_sha_ok": t.get("checks_code_sha256") == code_sha,
                  "pass": t.get("pass"), "summary_pass": summ["tests"].get(k),
                  "pass_matches_summary": t.get("pass") == summ["tests"].get(k),
                  "time_utc": t.get("time_utc")}
    res["checks"] = per
    res["checks_ok"] = all(v.get("exists") and v["prereg_sha_ok"] and v["inputs_verified"]
                           and v["code_sha_ok"] and v["pass_matches_summary"]
                           for v in per.values())
    arch = sorted((CHECKS / "code_archive").glob("*.py"))
    res["code_archive"] = {a.name: sha256_file(a) == code_sha for a in arch}
    res["summary_gates"] = {g: v["passed"] for g, v in summ["gates"].items()}
    res["sd_label"] = summ.get("sd_label")

    pinned = {i["path"]: i["sha256"] for i in doc["inputs"]}
    v3 = {}
    for name, d in V3.items():
        files = sorted(p for p in d.iterdir() if p.is_file())
        rows = {lib.rel(p): (pinned.get(lib.rel(p)) == sha256_file(p)) for p in files}
        v3[name] = {"files": len(files), "all_pinned_and_equal": all(rows.values()),
                    "not_pinned_or_changed": [k for k, v in rows.items() if not v],
                    "already_finalised": [f for f in ("MANIFEST.json", "validation.json",
                                                      "PRODUCT_CARD.md") if (d / f).exists()]}
    res["v3_dirs"] = v3

    man = [check_manifest(d / "MANIFEST.json") for d in PUBLISHED]
    res["published_manifests"] = [{k: m.get(k) for k in ("path", "status", "n_hashed",
                                                         "schema_errors", "hash_problems")}
                                  for m in man]
    res["published_manifests_ok"] = all(m["status"] == "ok" for m in man)

    a = prereg3.audit()
    res["audit"] = {"index_records": a["index_records"], "looks_records": a["looks_records"],
                    "c0_refused": [x for x in a["tests_refused_under_prereg3"]
                                   if x["wp"] == lib.WP]}
    res["seconds"] = round(time.time() - t0, 1)
    lib.save_json(res, OUT)
    return res


def main(argv: list[str]) -> int:
    if argv == ["log"]:
        log_look()
        return 0
    if argv == ["check"]:
        r = check()
        print({k: r[k] for k in ("prereg_status", "checks_ok", "summary_gates", "sd_label",
                                 "published_manifests_ok", "seconds")})
        print("v3 dirs:", r["v3_dirs"])
        print("archive:", r["code_archive"])
        print("c0 refused:", len(r["audit"]["c0_refused"]))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
