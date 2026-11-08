"""Make a claim-honesty grader workspace outside the repository (user ruling R11.4).

R11.4 (user rulings 2026-11-08, twenty-fourth set): the graders work in a fresh clone with no
``*.local.md`` and no HONESTY files, do not read the honesty-fix git history or PRs, receive
GRADER_BRIEF.md word for word, and read statements through the committed mechanical mask. This
script makes that workspace from the hashed inventory, in one run, at a path outside the
repository:

    DEST/inventory.json      the masked grader view (honesty.grader_view; schema
                             twin-honesty-grader-view/1; its sha256 is the inventory's)
    DEST/GRADER_BRIEF.md     the brief, byte for byte as committed at the pinned commit
    DEST/sample.json         only with --sample (the second grader's statement ids)
    DEST/repo/               the record the brief lets graders consult:
                             - the tracked tree at the pinned commit (``git archive``: no .git,
                               so no history, no branches and no PR refs);
                             - the gitignored record beside it: ``DEFAULT_RECORDS``
                               (registrations, reports, rulings, validation units and the twin
                               results with the pinned page; ``--record`` replaces them) plus
                               every other ``results/`` path a statement's ``check_against``
                               cites (a file, a directory, or a name prefix such as
                               ``results/twin_forward/all30_page``: the files it starts).

Excluded everywhere (``EXCLUDE``): ``*.local.md`` (planning notes), any path component starting
with ``HONESTY`` (earlier audit outputs, round-9 findings and views), stray findings, view and
sample files, ``.git``, and the analyst eval reports and answers (the brief excludes analyst
answers). Files over ``--max-mb`` are not copied (listed).

Then every text file of DEST/repo except the brief passes through the same committed mask
(``honesty_mask.mask_file_text``: Markdown, Python and text by block, HTML, JS and logs by
line, JSON by string value and key path; data tables and binary files are copied as they are),
so the docs a grader opens to check a statement show the
same placeholders as the view. Masked files are for reading, not running.

Before masking, the inventory's input files in DEST/repo must hash to the inventory's
``inputs`` (the pinned commit plus the copied record reproduce the inventory), or the script
stops. The operator's summary (copied, excluded, masked files, residual literal hits) is
printed, or written with ``--manifest`` to a path outside DEST; nothing about it is left in
DEST.

    python dataprep_staging/honesty/make_grader_workspace.py --inventory INVENTORY.json \\
        --dest /path/outside/the/repo [--commit SHA] [--sample SAMPLE.json] \\
        [--check-literal TEXT ...] [--manifest SUMMARY.json]
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hydrophysics.twin import claims as C  # noqa: E402
from hydrophysics.twin import honesty as H  # noqa: E402
from hydrophysics.twin import honesty_mask as M  # noqa: E402

BRIEF = "dataprep_staging/honesty/GRADER_BRIEF.md"
DEFAULT_RECORDS = ("results/dataprep/preregistration", "results/dataprep/reports",
                   "results/dataprep/rulings", "results/dataprep/validation_units",
                   "results/twin")
EXCLUDE_NAMES = ("*.local.md", "HONESTY*", "honesty_findings*", "grader_view*",
                 "second_grader*", ".git")
EXCLUDE_PATHS = ("docs/analyst/EVAL_*", "results/analyst", "results/analyst/*")


class WorkspaceError(RuntimeError):
    """The workspace cannot be made as R11.4 asks."""


def excluded(rel: str) -> bool:
    parts = rel.split("/")
    return any(fnmatch.fnmatchcase(p, pat) for p in parts for pat in EXCLUDE_NAMES) or \
        any(fnmatch.fnmatchcase(rel, pat) for pat in EXCLUDE_PATHS)


def _inside(p: Path, root: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _archive(repo: Path, commit: str, out: Path) -> None:
    out.mkdir(parents=True)
    git = subprocess.Popen(["git", "-C", str(repo), "archive", "--format=tar", commit],
                           stdout=subprocess.PIPE)
    tar = subprocess.run(["tar", "-x", "-C", str(out)], stdin=git.stdout, check=False)
    git.stdout.close()
    if git.wait() != 0 or tar.returncode != 0:
        raise WorkspaceError(f"git archive {commit} failed")


def _copy_record(src: Path, dst: Path, rel0: str, max_bytes: int, log: dict) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        rel = f"{rel0}/{p.relative_to(src).as_posix()}"
        if excluded(rel):
            log["excluded"].append(rel)
            continue
        if any(excluded(f"{rel0}/{a.as_posix()}") for a in p.relative_to(src).parents):
            continue
        if not p.is_file():
            continue
        if p.stat().st_size > max_bytes:
            log["too_large"].append(rel)
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, q)
        n += 1
    return n


def cited_records(inv: dict, repo: Path, records: tuple[str, ...]) -> list[str]:
    """The ``results/`` paths the statements' ``check_against`` cite outside ``records``, as
    files or directories of ``repo`` (a cited name prefix gives the files it starts)."""
    out: set[str] = set()
    cites = {p for r in inv["records"] for p in r["check_against"]["reports"]}
    for rel in sorted(cites):
        if not rel.startswith("results/") or any(
                rel == d or rel.startswith(d + "/") for d in records):
            continue
        p = repo / rel
        if p.exists():
            out.add(rel)
        elif p.parent.is_dir():
            out.update(f"{Path(rel).parent.as_posix()}/{q.name}"
                       for q in p.parent.glob(p.name + "*"))
    return sorted(out)


def _prune(tree: Path, log: dict) -> None:
    for p in sorted(tree.rglob("*"), key=lambda q: -len(q.parts)):
        rel = p.relative_to(tree).as_posix()
        if (p.exists() or p.is_symlink()) and excluded(rel):
            log["excluded"].append(rel)
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()


def _scrub(tree: Path, skip: set[str], max_bytes: int, log: dict) -> None:
    for p in sorted(tree.rglob("*")):
        rel = p.relative_to(tree).as_posix()
        if rel in skip or not p.is_file() or p.is_symlink() or p.stat().st_size > max_bytes:
            continue
        try:
            raw = p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        res = M.mask_file_text(p.name, raw)
        if res is not None and res[1]:
            p.write_text(res[0], encoding="utf-8")
            log["masked_files"][rel] = res[1]


def make_workspace(inventory: Path, dest: Path, repo: Path = REPO, commit: str | None = None,
                   records: tuple[str, ...] = DEFAULT_RECORDS, sample: Path | None = None,
                   max_mb: float = 64.0, check_literals: tuple[str, ...] = ()) -> dict:
    repo, dest = Path(repo).resolve(), Path(dest).resolve()
    if _inside(dest, repo) or _inside(repo, dest):
        raise WorkspaceError(f"{dest}: the workspace must lie outside the repository {repo}")
    if dest.exists() and any(dest.iterdir()):
        raise WorkspaceError(f"{dest}: not empty")
    inv = H.load_inventory(inventory)
    commit = commit or inv.get("commit")
    if not commit:
        raise WorkspaceError("the inventory records no commit; pass --commit")
    for rel in records:
        if _inside(dest, repo / rel):
            raise WorkspaceError(f"{dest}: inside the record {rel}")
    max_bytes = int(max_mb * 1e6)
    log: dict = {"excluded": [], "too_large": [], "records": {}, "masked_files": {}}
    tree = dest / "repo"
    dest.mkdir(parents=True, exist_ok=True)
    _archive(repo, commit, tree)
    for rel in records:
        src = repo / rel
        log["records"][rel] = _copy_record(src, tree / rel, rel, max_bytes, log) \
            if src.is_dir() else "missing"
    for rel in cited_records(inv, repo, records):
        src = repo / rel
        if excluded(rel):
            log["excluded"].append(rel)
        elif src.is_dir():
            log["records"][rel] = _copy_record(src, tree / rel, rel, max_bytes, log)
        elif src.stat().st_size > max_bytes:
            log["too_large"].append(rel)
        else:
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, tree / rel)
            log["records"][rel] = 1
    _prune(tree, log)
    for rel, h in inv["inputs"].items():
        p = tree / rel
        if not p.is_file() or C.sha256(p) != h:
            raise WorkspaceError(f"{rel}: missing from the workspace or not the inventory's "
                                 "input (sha256 differs); pin the commit the inventory was "
                                 "built on and copy the record that holds the page")
    view = H.grader_view(inv, tree)
    if not (tree / BRIEF).is_file():
        raise WorkspaceError(f"{BRIEF} is not in the tree at {commit}")
    shutil.copyfile(tree / BRIEF, dest / "GRADER_BRIEF.md")
    (dest / "inventory.json").write_text(json.dumps(view, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    if sample is not None:
        shutil.copyfile(sample, dest / "sample.json")
    _scrub(tree, {BRIEF}, max_bytes, log)
    hits = {}
    for lit in check_literals:
        hits[lit] = {}
        for p in sorted(dest.rglob("*")):
            if p.is_file() and not p.is_symlink() and p.stat().st_size <= max_bytes:
                k = p.read_bytes().count(lit.encode())
                if k:
                    hits[lit][p.relative_to(dest).as_posix()] = k
    return {"dest": str(dest), "commit": commit, "inventory_sha256": inv["sha256"],
            "view_sha256": view["view_sha256"], "mask_rules_sha256": view["mask_rules_sha256"],
            "masked_statements": view["masked"], "records": log["records"],
            "excluded": sorted(set(log["excluded"])), "too_large": log["too_large"],
            "masked_files": log["masked_files"], "literal_hits": hits}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inventory", type=Path, required=True, help="the hashed inventory")
    ap.add_argument("--dest", type=Path, required=True, help="new directory outside the repo")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--commit", default=None, help="default: the inventory's commit")
    ap.add_argument("--record", action="append", default=None,
                    help=f"gitignored record path to copy (repeatable; default "
                         f"{', '.join(DEFAULT_RECORDS)})")
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--max-mb", type=float, default=64.0)
    ap.add_argument("--check-literal", action="append", default=[],
                    help="report files of the workspace that still contain this text")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="write the operator summary here (outside the workspace)")
    a = ap.parse_args(argv)
    if a.manifest is not None and _inside(a.manifest, a.dest):
        raise WorkspaceError("the operator summary must not be written into the workspace")
    res = make_workspace(a.inventory, a.dest, a.repo, a.commit,
                         tuple(a.record) if a.record else DEFAULT_RECORDS, a.sample, a.max_mb,
                         tuple(a.check_literal))
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.manifest is not None:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(text + "\n", encoding="utf-8")
        print(f"workspace {res['dest']}: view {res['view_sha256']}, mask rules "
              f"{res['mask_rules_sha256']}; summary {a.manifest}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
