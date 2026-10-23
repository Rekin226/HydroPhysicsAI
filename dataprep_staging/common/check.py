"""Manifest checker CLI: validate every MANIFEST.json and SOURCE.json and recompute every hash.

    python -m common.check                      # all products, pilots and L0 sources
    python -m common.check results/dataprep/heads  # only manifests under a path
    python -m common.check --json out.json --no-sources
    python -m common.check --fail-on-orphans    # also fail on L0 files no record covers

(run from ``dataprep_staging/`` or with it on PYTHONPATH; ``python
dataprep_staging/common/check.py`` also works). Exit status 1 if any manifest fails its
schema or any hash is not reproduced (or, with ``--fail-on-orphans``, if an L0 file is
covered by no SOURCE.json and no L0-registry manifest).

Orphans (repair round): every file under ``data_fetch_api/new_measurements/`` should be
listed by a ``SOURCE.json`` (plan 2.1). Directories that predate the convention and are
read-only for the data-prep work are covered instead by the L0 registry
(``results/dataprep/l0_registry/v<N>/MANIFEST.json``, layer ``registry``), whose inputs are
those files with their sha256, so a later change to any of them fails the check. Files
covered by neither are reported as orphans.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.manifest import (
    find_manifests,
    load_manifest,
    validate_manifest,
    verify_hashes,  # noqa: E402
)
from common.paths import DATAPREP, L0_NEW, from_repo, rel_to_repo  # noqa: E402
from common.source import SOURCE_NAME, check_source_file, listed_files  # noqa: E402


def check_manifest(path: Path) -> dict:
    t0 = time.time()
    try:
        m, d = load_manifest(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": rel_to_repo(path), "status": "fail", "problems": [f"unreadable: {exc}"]}
    schema = validate_manifest(m, d)
    hashes = verify_hashes(m, d)
    n = sum(len(m.get(k) or []) for k in ("inputs", "code", "outputs"))
    return {"path": rel_to_repo(path), "product": m.get("product"), "version": m.get("version"),
            "layer": m.get("layer"), "status": "fail" if (schema or hashes) else "ok",
            "n_hashed": n, "schema_errors": schema, "hash_problems": hashes,
            "seconds": round(time.time() - t0, 2)}


def find_orphans(sources_root: Path, source_files: list[Path],
                 registry_manifests: list[Path]) -> list[str]:
    """Files under ``sources_root`` covered by no SOURCE.json and no registry manifest."""
    covered = set()
    for p in source_files:
        try:
            covered.update(listed_files(p))
        except (OSError, json.JSONDecodeError):
            covered.add(p.resolve())
    for mp in registry_manifests:
        try:
            m, _ = load_manifest(mp)
        except (OSError, json.JSONDecodeError):
            continue
        covered.update(from_repo(r["path"]).resolve() for r in m.get("inputs") or [])
    out = []
    for q in sorted(Path(sources_root).rglob("*")):
        if q.is_file() and "__pycache__" not in q.parts and q.resolve() not in covered:
            out.append(rel_to_repo(q))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-hash every dataprep manifest and L0 record.")
    ap.add_argument("roots", nargs="*", help="directories to scan (default: results/dataprep)")
    ap.add_argument("--no-sources", action="store_true", help="skip L0 SOURCE.json files")
    ap.add_argument("--sources-root", default=str(L0_NEW))
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--registry-root", default=str(DATAPREP / "l0_registry"),
                    help="L0-registry manifests that cover pre-convention L0 files")
    ap.add_argument("--fail-on-orphans", action="store_true",
                    help="exit 1 if an L0 file is covered by no SOURCE.json or registry")
    a = ap.parse_args(argv)
    roots = [from_repo(r) for r in a.roots] or [DATAPREP]
    rep = {"manifests": [], "sources": []}
    for r in roots:
        for p in find_manifests(r):
            res = check_manifest(p)
            rep["manifests"].append(res)
            print(f"{res['status']:5s} {res.get('n_hashed', 0):5d} files  {res['path']}")
            for e in res.get("schema_errors", []) + res.get("problems", []):
                print("      schema:", e)
            for h in res.get("hash_problems", []):
                print("      hash:", h)
    orphans: list[str] = []
    if not a.no_sources:
        source_files = sorted(Path(a.sources_root).rglob(SOURCE_NAME))
        for p in source_files:
            res = check_source_file(p)
            res["path"] = rel_to_repo(p)
            rep["sources"].append(res)
            print(f"{res['status']:6s} {res['n_files']:4d} files  {res['path']}")
            for e in res["problems"]:
                print("      ", e)
        registries = {from_repo(r["path"]) for r in rep["manifests"]
                      if r.get("layer") == "registry"}
        if Path(a.registry_root).is_dir():                  # scoped runs too
            registries.update(find_manifests(a.registry_root))
        registries = sorted(registries)
        orphans = find_orphans(Path(a.sources_root), source_files, registries)
        rep["orphans"] = orphans
        tops = {}
        for o in orphans:
            rel = Path(o).relative_to(rel_to_repo(Path(a.sources_root)))
            top = rel.parts[0] if len(rel.parts) > 1 else "(root files)"
            tops[top] = tops.get(top, 0) + 1
        for t, n in sorted(tops.items()):
            print(f"orphan {n:5d} files  {t}")
    nm = len(rep["manifests"])
    fm = sum(r["status"] != "ok" for r in rep["manifests"])
    ns = len(rep["sources"])
    fs = sum(r["status"] == "fail" for r in rep["sources"])
    lg = sum(r["status"] == "legacy" for r in rep["sources"])
    hashed = sum(r.get("n_hashed", 0) for r in rep["manifests"]) + sum(
        r["n_files"] for r in rep["sources"] if r["status"] != "legacy")
    rep["summary"] = {"manifests": nm, "manifests_failed": fm, "sources": ns,
                      "sources_failed": fs, "sources_legacy": lg, "files_hashed": hashed,
                      "orphans": len(orphans)}
    print(f"\nmanifests {nm} (failed {fm}); sources {ns} (failed {fs}, legacy {lg}); "
          f"files re-hashed {hashed}; L0 orphans {len(orphans)}")
    if a.json:
        Path(a.json).write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8")
    return 1 if (fm or fs or (a.fail_on_orphans and orphans)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
