"""Record that ground_elevation v1 and v2 are superseded by v3 (WP W3, continuation round).

v1 (built with the code recorded in the pre-registration), v2 and v3 were built minutes apart
from the same inputs and design. The builder files were then edited in place (product-card
wording, labels, one reported-only metric), so the code hashes in the v1 and v2 manifests no
longer match the files and the checker reports both as failed. Version directories are never
rewritten, so this script writes a product-level ``SUPERSEDED.json`` instead. It re-hashes
every output of the three versions and lists which are byte-identical to v3, so a reader can
see that the edits changed documentation only.

Usage: python -I supersede.py      (refuses to overwrite an existing SUPERSEDED.json)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from common import hashing  # noqa: E402
from common.paths import DATAPREP, rel_to_repo  # noqa: E402

PROD = DATAPREP / "ground_elevation"


def main() -> int:
    out = PROD / "SUPERSEDED.json"
    if out.exists():
        raise SystemExit(f"{out} exists; never overwritten")
    man = {v: json.loads((PROD / v / "MANIFEST.json").read_text(encoding="utf-8"))
           for v in ("v1", "v2", "v3")}
    now = {v: {r["path"]: hashing.sha256_file(PROD / v / r["path"]) for r in man[v]["outputs"]}
           for v in man}
    pre = json.loads((DATAPREP / "preregistration/W3.dtm_heldout_bias.json")
                     .read_text(encoding="utf-8"))
    reg_code = {r["path"]: r["sha256"] for r in pre["code_at_registration"]}
    recs = []
    for v in ("v1", "v2"):
        rec_out = {r["path"]: r["sha256"] for r in man[v]["outputs"]}
        same = sorted(p for p in rec_out if now["v3"].get(p) == now[v][p])
        diff = sorted(p for p in rec_out if now["v3"].get(p) != now[v][p])
        code = {r["path"]: r["sha256"] for r in man[v]["code"]}
        cur = {p: hashing.sha256_file(HERE.parents[2] / p) for p in code}
        recs.append({
            "version": v, "superseded_by": "v3",
            "outputs_rehashed_equal_manifest": all(rec_out[p] == now[v][p] for p in rec_out),
            "outputs_byte_identical_to_v3": same,
            "outputs_differing_from_v3": diff,
            "code_files_edited_since_build": sorted(p for p in code if cur[p] != code[p]),
            "code_equals_preregistration_code": all(reg_code.get(p) == code[p] for p in code),
            "gate": man[v]["gate"],
        })
    doc = {
        "schema": "dataprep-superseded/1",
        "product": "ground_elevation",
        "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "writer": rel_to_repo(HERE),
        "current": "v3",
        "superseded": recs,
        "why": "v1 and v2 were built with earlier in-place editions of dataprep_staging/w3_dtm/"
               "build.py, mosaic.py and validation.py (v1 with the editions hashed in the "
               "pre-registration). Those editions were not kept, so the common.check code-hash "
               "failure of v1 and v2 is expected and permanent. Every data output of v1 and v2 "
               "is byte-identical to v3; only PRODUCT_CARD.md and validation.json differ "
               "(wording, sd labels, and v3's added leave-one-block-out coverage of sd_epoch). "
               "Consumers open v3.",
        "v3_build_note": man["v3"].get("build_note"),
    }
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    out.chmod(0o444)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
