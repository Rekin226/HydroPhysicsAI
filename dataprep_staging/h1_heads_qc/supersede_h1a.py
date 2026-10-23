"""H1a repair round (2026-10-13): write ``results/dataprep/heads/L1_clean/SUPERSEDED.json``
(the convention W3 introduced for ground_elevation) naming the current version and why the
earlier ones are superseded (python -I).

* v1: gate not passed (H1a.qc_acceptance.v1); its flag-1 step and episode corrections are
  largely spurious (round-1 verifier; repair-round level check).
* v2: the registered build of H1a.qc_acceptance.v2 (all 7 criteria passed), published for
  provenance only (after_the_fact_analysis, validator); its 8 co-located-backed step
  corrections were not confirmed by the sibling-difference check, hence the reviewed v3.
  Three repair-round scripts listed in v2's manifest (``s2c_review_v3.py``,
  ``s6_build_product_v2b.py``, ``looks_h1a_repair.py``) had wording edits after v2 was
  packaged (a card sentence, the v3 backing note and a look note: "all 8 refuted" became
  "7 not confirmed, 1 untestable"), so the checker reports their sha256 for v2. The exact
  bytes v2 recorded were reconstructed and archived in
  ``results/dataprep/reports/H1a_evidence/v2/code_as_packaged_in_v2/``; this script
  verifies them against v2's manifest.

Usage: supersede_h1a.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import REPO  # noqa: E402

PROD = REPO / "results" / "dataprep" / "heads" / "L1_clean"
ARCH = REPO / "results" / "dataprep" / "reports" / "H1a_evidence" / "v2" / "code_as_packaged_in_v2"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def outputs_equal(v: str, ref: str) -> tuple[list[str], list[str]]:
    a = json.loads((PROD / v / "MANIFEST.json").read_text())
    b = json.loads((PROD / ref / "MANIFEST.json").read_text())
    hb = {o["path"]: o["sha256"] for o in b["outputs"]}
    same = [o["path"] for o in a["outputs"] if hb.get(o["path"]) == o["sha256"]]
    diff = [o["path"] for o in a["outputs"] if hb.get(o["path"]) != o["sha256"]]
    return same, diff


def main():
    m2 = json.loads((PROD / "v2" / "MANIFEST.json").read_text())
    edited, archive = [], []
    for c in m2["code"]:
        p = REPO / c["path"]
        if sha(p) != c["sha256"]:
            edited.append(c["path"])
            a = ARCH / Path(c["path"]).name
            archive.append({"path": c["path"], "archived_copy": str(a.relative_to(REPO)),
                            "recorded_sha256": c["sha256"],
                            "archived_copy_sha256": sha(a) if a.exists() else None,
                            "archived_copy_matches": a.exists() and sha(a) == c["sha256"],
                            "current_sha256": sha(p)})
    s2, d2 = outputs_equal("v2", "v3")
    s1, d1 = outputs_equal("v1", "v3")
    doc = {"schema": "dataprep-superseded/1", "product": "heads/L1_clean",
           "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "writer": "dataprep_staging/h1_heads_qc/supersede_h1a.py", "current": "v3",
           "superseded": [
               {"version": "v1", "superseded_by": "v3",
                "why": "gate not passed (H1a.qc_acceptance.v1: paired recall 0.889, 1 of 3 "
                       "real events flagged, 3 episodes on >= 20 hPa falls); flag-1 step and "
                       "episode corrections largely spurious (pre-origin level check: 18 of 39 "
                       "flag-1 steps and 125 of 210 flag-1 episodes fail)",
                "outputs_byte_identical_to_v3": s1, "outputs_differing_from_v3": d1,
                "allowed_uses_note": "gate failed: the guard grants only after_the_fact_"
                                     "analysis and design_time_evidence (<= 2022-12)"},
               {"version": "v2", "superseded_by": "v3",
                "why": "registered build of H1a.qc_acceptance.v2 (7 of 7 criteria passed); "
                       "its 8 co-located-backed step corrections were not confirmed by the "
                       "after-the-fact sibling-difference check (7 not confirmed, 1 "
                       "untestable); v3 flags them instead (post-look review)",
                "allowed_uses": m2["allowed_uses"],
                "outputs_byte_identical_to_v3": s2, "outputs_differing_from_v3": d2,
                "code_files_edited_since_build": edited,
                "code_archive": archive,
                "code_equals_preregistration_code": "registered inputs of "
                                                    "H1a.qc_acceptance.v2 unchanged; the "
                                                    "edited files are repair-round scripts, "
                                                    "not registered inputs"}]}
    (PROD / "SUPERSEDED.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"edited": edited,
                      "archive_ok": all(a["archived_copy_matches"] for a in archive)}))


if __name__ == "__main__":
    main()
