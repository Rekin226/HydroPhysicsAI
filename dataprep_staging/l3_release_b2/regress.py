"""Compare release_20261012b2 with release_20261012b file by file and check the two rulings.

    PYTHONPATH=<repo>/dataprep_staging python -m l3_release_b2.regress NEW_RELEASE_DIR --json OUT

Reads both releases (never writes into them). Every file of the two trees is hashed
(REBUILD_CHECK.json aside). A file that differs is explained:

* npz: compared array by array; the key ``release_component`` names the release and is
  expected to differ (``equal_except_release_name``);
* JSON (RELEASE.json, MANIFEST.json, validation.json): compared after replacing the release
  and package names of b2 by b's; the remaining differing keys are listed and split into
  mechanical ones (code hashes and git state of the b2 package, the HEAD commit, the hashes of
  re-packed npz and of other manifests) and content ones;
* PRODUCT_CARD.md: the lines that differ after the same name replacement.

Then the rulings are checked: pre-origin piles allow exactly validator and
after_the_fact_analysis and refuse design_time_evidence and scoring_target; every read of a
component uses an allowed use; the W1 pre-origin component keeps b's uses and data; both rulings
and the open decision are in RELEASE.json; content differences appear only in the two affected
components; every data file (table, csv, copy) is byte-identical to b's.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
from common import guard, hashing
from common import uses as U
from common.paths import REPO

from l3_release_b2 import config as C
from l3_release_b2 import io_det

OLD = REPO / C.RELEASE_DIR / C.SUPERSEDES
NOT_HASHED = ("REBUILD_CHECK.json",)
NAME_KEYS = {"release_component"}
AFFECTED = {"preorigin/surface_water", "preorigin/piles_printed"}
RENAMES = ((C.RELEASE, C.SUPERSEDES), ("l3_release_b2", "l3_release_b"))
#: differing JSON keys that follow from the new release name, package and commit, not from
#: a ruling (index-collapsed paths)
MECHANICAL = [
    r"^code\[\*\]\.(sha256|bytes|git_state)$",              # manifests: b2 package files
    r"^git_commit$",                                          # HEAD at build time
    r"^code\[\*\]\.sha256$",                                  # RELEASE.json code list
    r"^outputs\[\*\]\.(sha256|bytes)$",                       # re-packed npz (release name)
    r"^variants\.\w+\.components\[\*\]\.outputs\[\*\]\.(sha256|bytes)$",
    r"^variants\.\w+\.components\[\*\]\.manifest_sha256$",
    r"^grid\.manifest_sha256$",
    r"^release_gate\.item4_product_card\.sha256$",
]


def _norm(x):
    if isinstance(x, str):
        for a, b in RENAMES:
            x = x.replace(a, b)
        return x
    if isinstance(x, list):
        return [_norm(v) for v in x]
    if isinstance(x, dict):
        return {_norm(k): _norm(v) for k, v in x.items()}
    return x


def _json_diff(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if k not in a or k not in b:
                out.append(p)
            else:
                out += _json_diff(a[k], b[k], p)
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [path]
        out = []
        for i, (u, v) in enumerate(zip(a, b, strict=True)):
            out += _json_diff(u, v, f"{path}[{i}]")
        return out
    return [] if a == b else [path]


def _collapse(p: str) -> str:
    return re.sub(r"\[\d+\]", "[*]", p)


def _npz_equal(a: Path, b: Path) -> tuple[bool, list[str]]:
    xa, sa = io_det.read_npz(a)
    xb, sb = io_det.read_npz(b)
    diff = sorted((set(xa) ^ set(xb)) | set(sa) ^ set(sb))
    for k in sorted(set(xa) & set(xb)):
        if k in NAME_KEYS:
            continue
        u, v = xa[k], xb[k]
        same = (u.shape == v.shape and u.dtype == v.dtype
                and np.array_equal(u, v, equal_nan=u.dtype.kind == "f"))
        if not same:
            diff.append(k)
    return not diff, diff


def _component_of(rel: str) -> str | None:
    parts = rel.split("/")
    return "/".join(parts[:2]) if parts[0] in ("preorigin", "full") else None


def compare_trees(new: Path) -> dict:
    ho = {p.relative_to(OLD).as_posix(): hashing.sha256_file(p)
          for p in hashing.iter_files(OLD) if p.name not in NOT_HASHED}
    hn = {p.relative_to(new).as_posix(): hashing.sha256_file(p)
          for p in hashing.iter_files(new) if p.name not in NOT_HASHED}
    files = []
    for rel in sorted(set(ho) | set(hn)):
        rec = {"path": rel, "component": _component_of(rel)}
        if rel not in ho or rel not in hn:
            rec["status"] = "only_in_new" if rel in hn else "only_in_old"
            files.append(rec)
            continue
        if ho[rel] == hn[rel]:
            rec["status"] = "identical"
            files.append(rec)
            continue
        rec["status"] = "differs"
        if rel.endswith(".npz"):
            ok, d = _npz_equal(OLD / rel, new / rel)
            rec["kind"] = "equal_except_release_name" if ok else "npz_content"
            rec["differing_keys"] = d
        elif rel.endswith(".json"):
            a = _norm(json.loads((OLD / rel).read_text(encoding="utf-8")))
            b = json.loads((new / rel).read_text(encoding="utf-8"))
            nb = _norm(b)
            paths = _json_diff(a, nb)
            coll = sorted({_collapse(p) for p in paths})
            mech = [p for p in coll if any(re.match(m, p) for m in MECHANICAL)]
            rec["kind"] = "json"
            rec["mechanical_keys"] = mech
            rec["content_keys"] = [p for p in coll if p not in mech]
            rec["only_name_differs"] = not paths
        elif rel.endswith(".md"):
            a = _norm((OLD / rel).read_text(encoding="utf-8")).splitlines()
            b = _norm((new / rel).read_text(encoding="utf-8")).splitlines()
            d = [x for x in difflib.unified_diff(a, b, lineterm="", n=0)
                 if x[:1] in "+-" and not x.startswith(("+++", "---"))]
            rec["kind"] = "text"
            rec["lines_removed"] = [x[1:] for x in d if x.startswith("-")]
            rec["lines_added"] = [x[1:] for x in d if x.startswith("+")]
        else:
            rec["kind"] = "data_file_differs"
        files.append(rec)
    return {"files": files,
            "counts": {s: sum(f["status"] == s for f in files)
                       for s in ("identical", "differs", "only_in_new", "only_in_old")}}


def _probe(comp_dir: Path) -> dict:
    res = {}
    for use in sorted(U.USES):
        try:
            guard.open_product(comp_dir, use, verify=False,
                               require_gate=True if use in U.GATE_EXEMPT_USES else None)
            res[use] = "granted"
        except (guard.UseNotAllowed, guard.LeakageError, guard.GateNotPassed) as exc:
            res[use] = f"refused:{type(exc).__name__}"
    return res


def rulings_check(new: Path) -> dict:
    rel = json.loads((new / "RELEASE.json").read_text(encoding="utf-8"))
    pm = json.loads((new / "preorigin/piles_printed/v1/MANIFEST.json").read_text(
        encoding="utf-8"))
    pm_old = json.loads((OLD / "preorigin/piles_printed/v1/MANIFEST.json").read_text(
        encoding="utf-8"))
    wm = json.loads((new / "preorigin/surface_water/v1/MANIFEST.json").read_text(
        encoding="utf-8"))
    wm_old = json.loads((OLD / "preorigin/surface_water/v1/MANIFEST.json").read_text(
        encoding="utf-8"))
    probe = _probe(new / "preorigin/piles_printed/v1")
    reads_bad = []
    for v in rel["variants"].values():
        for c in v["components"]:
            m = json.loads((new / c["path"] / "MANIFEST.json").read_text(encoding="utf-8"))
            for rd in m["parameters"]["reads"]:
                if rd["use"] not in m["allowed_uses"]:
                    reads_bad.append(f"{c['variant']}/{c['name']}: {rd['use']}")
    ids = [r["id"] for r in rel.get("user_rulings", [])]
    card = (new / "PRODUCT_CARD.md").read_text(encoding="utf-8")
    w1_notes = " ".join(wm["parameters"]["notes"])
    return {
        "piles_allowed_uses": pm["allowed_uses"],
        "piles_allowed_uses_old": pm_old["allowed_uses"],
        "piles_reads": pm["parameters"]["reads"],
        "piles_reads_old": pm_old["parameters"]["reads"],
        "piles_guard_probe": probe,
        "piles_ok": (pm["allowed_uses"] == ["validator", "after_the_fact_analysis"]
                     and probe["design_time_evidence"].startswith("refused")
                     and probe["scoring_target"].startswith("refused")
                     and probe["validator"] == "granted"
                     and probe["after_the_fact_analysis"] == "granted"
                     and all(r["use"] == "validator" for r in pm["parameters"]["reads"])
                     and bool(pm["parameters"].get("user_rulings"))),
        "w1_allowed_uses_unchanged": wm["allowed_uses"] == wm_old["allowed_uses"],
        "w1_reads_unchanged": wm["parameters"]["reads"] == wm_old["parameters"]["reads"],
        "w1_ruling_in_manifest": (bool(wm["parameters"].get("user_rulings"))
                                  and "USER RULING 2026-10-18 (a)" in w1_notes),
        "reads_with_disallowed_use": reads_bad,
        "release_json_rulings": ids,
        "release_json_open_decisions": [d["id"] for d in rel.get("open_user_decisions", [])],
        "card_has_rulings": ("## User rulings (2026-10-18)" in card
                             and "## Open user decisions" in card),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("new")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    new = Path(a.new)
    cmp_ = compare_trees(new)
    rl = rulings_check(new)
    rb = json.loads((OLD / "REBUILD_CHECK.json").read_text(encoding="utf-8"))
    files = {p.relative_to(OLD).as_posix(): hashing.sha256_file(p)
             for p in hashing.iter_files(OLD) if p.name not in NOT_HASHED}
    old_unchanged = files == {f["path"]: f["sha256"] for f in rb["files"]}
    diffs = [f for f in cmp_["files"] if f["status"] != "identical"]
    unexpected = []
    for f in diffs:
        if f["status"] != "differs":
            unexpected.append(f"{f['path']}: {f['status']}")
        elif f["kind"] in ("npz_content", "data_file_differs"):
            unexpected.append(f"{f['path']}: {f['kind']}")
        elif f["kind"] == "json" and f["content_keys"] and f["component"] is not None \
                and f["component"] not in AFFECTED:
            unexpected.append(f"{f['path']}: content keys {f['content_keys']}")
    content_components = sorted({f["component"] for f in diffs if f.get("kind") == "json"
                                 and f["content_keys"] and f["component"]})
    res = {"old": f"{C.RELEASE_DIR}/{C.SUPERSEDES}", "old_tree_unchanged": old_unchanged,
           "compare": cmp_, "rulings": rl, "unexpected": unexpected,
           "components_with_content_changes": content_components}
    res["passed"] = bool(
        old_unchanged and not unexpected and rl["piles_ok"] and rl["w1_allowed_uses_unchanged"]
        and rl["w1_reads_unchanged"] and rl["w1_ruling_in_manifest"]
        and not rl["reads_with_disallowed_use"]
        and rl["release_json_rulings"] == [r["id"] for r in C.USER_RULINGS]
        and rl["release_json_open_decisions"] == [d["id"] for d in C.OPEN_USER_DECISIONS]
        and rl["card_has_rulings"] and set(content_components) <= AFFECTED)
    Path(a.json).write_text(io_det.dumps(res), encoding="utf-8")
    print(f"files {cmp_['counts']}; components with content changes {content_components}; "
          f"unexpected {len(unexpected)}; piles ok {rl['piles_ok']}; old release unchanged "
          f"{old_unchanged}; passed {res['passed']}")
    for u in unexpected:
        print("  UNEXPECTED", u)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
