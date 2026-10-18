"""Compare two builds of the release and verify a release in place (plan 2.8 items 1 and 5).

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release.verify compare A B --json OUT
    PYTHONPATH=<repo>/dataprep_staging python -m l3_release.verify check RELEASE_DIR --json OUT

``compare`` hashes every file of two release trees (relative paths) and lists identical,
differing and one-sided files. ``check`` re-validates every component manifest, re-hashes
every output, checks every file RELEASE.json lists, and opens every component for every use
(allowed uses must open, the others must be refused).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import guard, hashing, manifest
from common import uses as U

from l3_release import io_det

NOT_HASHED = ("REBUILD_CHECK.json",)


def tree_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    return {p.relative_to(root).as_posix(): hashing.sha256_file(p)
            for p in hashing.iter_files(root) if p.name not in NOT_HASHED}


def compare(a: Path, b: Path) -> dict:
    ha, hb = tree_hashes(a), tree_hashes(b)
    common = sorted(set(ha) & set(hb))
    diff = [k for k in common if ha[k] != hb[k]]
    return {"n_files_a": len(ha), "n_files_b": len(hb), "n_common": len(common),
            "n_identical": len(common) - len(diff), "differing": diff,
            "only_in_a": sorted(set(ha) - set(hb)), "only_in_b": sorted(set(hb) - set(ha)),
            "identical": not diff and set(ha) == set(hb),
            "tree_digest_a": _digest(ha), "tree_digest_b": _digest(hb),
            "files": [{"path": k, "sha256": ha[k]} for k in sorted(ha)]}


def _digest(h: dict[str, str]) -> str:
    return hashing.sha256_bytes("".join(f"{k}\0{v}\n" for k, v in sorted(h.items())).encode())


def check(release_dir: Path) -> dict:
    release_dir = Path(release_dir)
    rel = json.loads((release_dir / "RELEASE.json").read_text(encoding="utf-8"))
    out = {"manifests": {}, "release_json_files": {"checked": 0, "problems": []},
           "guard_probe": {}, "passed": True}
    mans = sorted(release_dir.rglob(manifest.MANIFEST_NAME))
    for mp in mans:
        m, d = manifest.load_manifest(mp)
        key = d.parent.relative_to(release_dir).as_posix()
        schema = manifest.validate_manifest(m, d)
        hashes = manifest.verify_hashes(m, d)
        res, bad = {}, []
        for use in sorted(U.USES):
            try:
                guard.open_product(d, use, verify=False,
                                   require_gate=True if use in U.GATE_EXEMPT_USES else None)
                res[use] = "granted"
            except (guard.UseNotAllowed, guard.LeakageError, guard.GateNotPassed) as exc:
                res[use] = f"refused:{type(exc).__name__}"
            if (res[use] == "granted") != (use in m["allowed_uses"]):
                bad.append(use)
        out["manifests"][key] = {"schema_errors": schema, "hash_problems": hashes,
                                 "guard_mismatches": bad, "n_outputs": len(m["outputs"])}
        out["guard_probe"][key] = res
        if schema or hashes or bad:
            out["passed"] = False
    for v in rel["variants"].values():
        for c in v["components"]:
            for o in c["outputs"]:
                p = release_dir / o["path"]
                out["release_json_files"]["checked"] += 1
                if not p.is_file() or hashing.sha256_file(p) != o["sha256"]:
                    out["release_json_files"]["problems"].append(o["path"])
            mp = release_dir / c["path"] / manifest.MANIFEST_NAME
            if hashing.sha256_file(mp) != c["manifest_sha256"]:
                out["release_json_files"]["problems"].append(str(mp.relative_to(release_dir)))
    if out["release_json_files"]["problems"]:
        out["passed"] = False
    out["n_manifests"] = len(mans)
    return out


def record(release_dir: Path, compare_json: Path, check_json: Path, note: str) -> dict:
    """Write REBUILD_CHECK.json into the release (not part of the hashed content): the file
    comparison of the two builds and the check of the release after its move into place."""
    import datetime as _dt
    cmp_ = json.loads(Path(compare_json).read_text(encoding="utf-8"))
    chk = json.loads(Path(check_json).read_text(encoding="utf-8"))
    final = tree_hashes(release_dir)
    same_as_build_a = {f["path"]: f["sha256"] for f in cmp_["files"]} == final
    rec = {
        "schema": "dataprep-release-rebuild/1", "plan": "2.8 item 5",
        "written_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "procedure": note,
        "compare": {k: cmp_[k] for k in ("n_files_a", "n_files_b", "n_common", "n_identical",
                                         "differing", "only_in_a", "only_in_b", "identical",
                                         "tree_digest_a", "tree_digest_b")},
        "final_equals_build_a": same_as_build_a,
        "final_tree_digest": _digest(final),
        "post_move_check": {"passed": chk["passed"], "n_manifests": chk["n_manifests"],
                            "release_json_files": chk["release_json_files"],
                            "manifest_problems": {k: v for k, v in chk["manifests"].items()
                                                  if v["schema_errors"] or v["hash_problems"]
                                                  or v["guard_mismatches"]}},
        "files": cmp_["files"],
    }
    p = Path(release_dir) / "REBUILD_CHECK.json"
    if p.exists():
        raise FileExistsError(p)
    p.write_text(io_det.dumps(rec), encoding="utf-8")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c1 = sub.add_parser("compare")
    c1.add_argument("a")
    c1.add_argument("b")
    c1.add_argument("--json")
    c2 = sub.add_parser("check")
    c2.add_argument("release_dir")
    c2.add_argument("--json")
    c3 = sub.add_parser("record")
    c3.add_argument("release_dir")
    c3.add_argument("compare_json")
    c3.add_argument("check_json")
    c3.add_argument("--note", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        rec = record(Path(a.release_dir), Path(a.compare_json), Path(a.check_json), a.note)
        print(f"REBUILD_CHECK.json: identical {rec['compare']['identical']}; final equals "
              f"build A {rec['final_equals_build_a']}; post-move check "
              f"{rec['post_move_check']['passed']}")
        return 0 if (rec["compare"]["identical"] and rec["final_equals_build_a"]
                     and rec["post_move_check"]["passed"]) else 1
    if a.cmd == "compare":
        res = compare(Path(a.a), Path(a.b))
        print(f"files {res['n_files_a']} / {res['n_files_b']}; identical {res['n_identical']}; "
              f"differing {len(res['differing'])}; only_a {len(res['only_in_a'])}; only_b "
              f"{len(res['only_in_b'])}; identical trees {res['identical']}")
        for k in res["differing"][:20]:
            print("  DIFF", k)
        ok = res["identical"]
    else:
        res = check(Path(a.release_dir))
        print(f"manifests {res['n_manifests']}; release files checked "
              f"{res['release_json_files']['checked']}; problems "
              f"{len(res['release_json_files']['problems'])}; passed {res['passed']}")
        for k, v in res["manifests"].items():
            if v["schema_errors"] or v["hash_problems"] or v["guard_mismatches"]:
                print("  FAIL", k, v)
        ok = res["passed"]
    if a.json:
        Path(a.json).write_text(io_det.dumps(res), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
