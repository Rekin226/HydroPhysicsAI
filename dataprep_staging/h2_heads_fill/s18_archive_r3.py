"""Archive the WP H2 run tables and the exact bytes of the registered code (repair round 3).

* ``H2_evidence/run_tables/``: every artificial-gap run table behind a registered or
  design-time H2 verdict (``<WORKDIR>/runs/pre/*``), the cached base matrices and bridged
  tables, the design directory and the run logs, copied from the session work directory (which
  lives under /tmp) into the gitignored results tree, with ``INDEX.json`` (path, bytes,
  sha256). The tables are deterministic (the verifier reproduced S r10, d r10 and E r30 to
  0.0), so this is a convenience copy, not the only route to them.
* ``H2_evidence/code_as_registered/``: the H2 staging modules with the sha256 each
  pre-registration recorded (``H2.fill_acceptance.v1``, ``v2``, ``H2.repair3.v1``), checked
  against the files on disk. ``config.py`` carries a session-scratchpad default for
  ``WORKDIR`` whose path names the local user; before any commit of ``dataprep_staging`` it
  must be replaced (CLAUDE.md Rule 1), which will make the registered hash differ: these copies
  keep the registrations and manifests verifiable after that edit.

Usage: python -I s18_archive_r3.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

from h2_heads_fill import config as C  # noqa: E402
from h2_heads_fill import r3_common as R3  # noqa: E402

PREREGS = ("H2.fill_acceptance.v1", "H2.fill_acceptance.v2", R3.PREREG_ID_R3)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def archive_runs() -> dict:
    dst = C.EVIDENCE_DIR / "run_tables"
    src = C.WORKDIR
    files = [p for p in sorted(src.rglob("*")) if p.is_file()
             and (p.parts[len(src.parts)] in ("runs", "design") or p.suffix in (".npz", ".txt")
                  or p.name.startswith(("bridged_", "after_fact_")))]
    index = []
    for p in files:
        rel = p.relative_to(src)
        q = dst / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        if not q.exists() or sha(q) != sha(p):
            shutil.copy2(p, q)
        index.append({"path": str(rel), "bytes": p.stat().st_size, "sha256": sha(p)})
    doc = {"source": "session work directory (C.WORKDIR, under /tmp; not named here)",
           "n_files": len(index), "bytes": sum(x["bytes"] for x in index), "files": index}
    R3.jdump(doc, dst / "INDEX.json")
    return {"n_files": doc["n_files"], "bytes": doc["bytes"]}


def archive_code() -> dict:
    dst = C.EVIDENCE_DIR / "code_as_registered"
    dst.mkdir(parents=True, exist_ok=True)
    rec = {}
    for pid in PREREGS:
        doc = json.loads((C.REPO / "results/dataprep/preregistration" / f"{pid}.json"
                          ).read_text())
        for inp in doc["inputs"]:
            if not inp["path"].startswith("dataprep_staging/h2_heads_fill/"):
                continue
            p = C.REPO / inp["path"]
            now = sha(p)
            r = rec.setdefault(inp["path"], {"sha256": inp["sha256"], "registrations": [],
                                             "on_disk_matches": now == inp["sha256"]})
            if r["sha256"] != inp["sha256"]:
                r.setdefault("other_registered_sha256", []).append(inp["sha256"])
            r["registrations"].append(pid)
            q = dst / inp["sha256"][:12] / p.name
            if not q.exists() and now == inp["sha256"]:
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
            r["archived_as"] = str(q.relative_to(C.EVIDENCE_DIR)) if q.exists() else None
    R3.jdump(rec, dst / "INDEX.json")
    return {"n_files": len(rec), "all_match": all(r["on_disk_matches"] for r in rec.values())}


def main():
    print(json.dumps({"runs": archive_runs(), "code": archive_code()}))


if __name__ == "__main__":
    main()
