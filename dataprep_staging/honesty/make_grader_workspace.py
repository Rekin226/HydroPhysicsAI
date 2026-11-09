"""Make a claim-honesty grader workspace outside the repository (user ruling R11.4).

R11.4 (user rulings 2026-11-08, twenty-fourth set): the graders work in a fresh clone with no
``*.local.md`` and no HONESTY files, do not read the honesty-fix git history or PRs, receive
GRADER_BRIEF.md word for word, and read statements through the committed mechanical mask. This
script makes that workspace from the hashed inventory, in one run, at a path outside the
repository:

    DEST/inventory.json      the masked grader view (honesty.grader_view; schema
                             twin-honesty-grader-view/1; its sha256 is the inventory's; its
                             statement ids are opaque, S0001... in inventory order)
    DEST/GRADER_BRIEF.md     the brief, byte for byte as committed at the pinned commit
    DEST/MASKED_FILES.txt    the files of repo/ the mask changed (their hashes no longer match
                             a printed hash; printed prefixes of their hashes are masked too)
    DEST/sample.json         only with --sample (the second grader's ids, as view ids)
    DEST/repo/               the record the brief lets graders consult:
                             - the tracked tree at the pinned commit (``git archive``: no .git,
                               so no history, no branches and no PR refs);
                             - the gitignored record beside it: ``DEFAULT_RECORDS``
                               (registrations, reports, rulings, validation units; ``--record``
                               replaces them), the pinned page, and every other ``results/``
                               path a statement's ``check_against`` cites (a file, a directory,
                               or a name prefix such as ``results/twin_forward/all30_page``).
                               Earlier pages are not copied (their differences show earlier
                               fixes). Rulings files dated after the pinned commit are left out.

Left out everywhere (``EXCLUDE_NAMES``, ``EXCLUDE_PATHS``): ``*.local.md``, any path component
starting with ``HONESTY``, stray findings, view and sample files, ``.git``, the tests (graders
check statements against records, not code; tests pin figures and name fixes), the analyst
(code, eval sets, eval reports and answers), and the progress recount (its module and map turn
a figure into another).

The pin manifest (``--write-pin-manifest``) refuses record files modified after the pinned
commit's commit time; the build records the manifest's sha256 in its summary. The progress
recount runs without writing bytecode, and ``__pycache__`` / ``*.pyc`` are never copied; a gate
checks that DEST/repo holds exactly the tracked files at the commit and the manifest's files,
less the exclusions.

Then, in DEST/repo:

1. every text file passes through the committed mask (``honesty_mask.mask_file_text``), except
   raw text layers of public tables (``DATA_LAYERS``), which are read like CSV tables (only
   the literal gate runs on them);
2. a ruling on the claim-honesty audit itself (``applies_to`` names claim honesty, or its text
   names the audit report or expected or round-9 findings) keeps its id and ``applies_to`` and
   has its other text replaced by ``REDACTED``; a file-level ``context`` naming the audit, a
   grader, honesty or findings is redacted the same way;
3. every printed hex prefix (7+ characters) of the sha256 or sha1 of a file changed by 1 or 2
   is masked, in the tree and in the view, repeated until no file changes.

Two gates run last and stop the script (exit 1, the workspace left for inspection) on a hit:
the derivation gate (``honesty_mask.derivable_pairs`` on every unit of every text file and on
every view statement) and the literal gate: the operator's current figures (``--figures``, a
JSON file kept outside the repository: a list, or ``{"hidden": [...], "visible": [...]}``; or
``--check-literal``; hidden figures are required), each with its rounding neighbours, must not
occur within ``LITERAL_WINDOW`` characters of an honesty, 6-part, headline, overall, audit or
grader term in any text file or view statement, unless a physical unit follows (``71.1
cm/yr``); in JSON only string values are searched (a number is masked by its key path).
Visible figures (science trust and parts, which the mask leaves by design) are only reported,
and a hidden figure with the same digits as a visible one is not gated (counted). A hit the
operator has reviewed (``--allow``: file, ``literal`` value or ``derivable: true``, and a
reason, e.g. the same digits as a well-disposal rate) is reported as allowed.

Before masking, the inventory's input files must hash to the inventory's ``inputs`` (the
pinned commit plus the copied record reproduce the inventory). The operator's summary (copied,
left out, masked and redacted files, gates, and the view-id map) is printed, or written with
``--manifest`` to a path outside DEST; nothing about it is left in DEST.

    python dataprep_staging/honesty/make_grader_workspace.py --inventory INVENTORY.json \\
        --dest /path/outside/the/repo --figures /path/outside/figures.json \\
        [--commit SHA] [--sample SAMPLE.json] [--manifest SUMMARY.json]
"""
from __future__ import annotations

import argparse
import bisect
import fnmatch
import json
import re
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
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
                   "results/dataprep/rulings", "results/dataprep/validation_units")
RULINGS = "results/dataprep/rulings"
EXCLUDE_NAMES = ("*.local.md", "HONESTY*", "honesty_findings*", "grader_view*",
                 "second_grader*", ".git", "__pycache__", "*.pyc", "*.pyo")
DATA_LAYERS = ("*/text_layer/*",)      # raw text layers of public tables: tables, not prose
EXCLUDE_PATHS = ("tests", "tests/*", "hydrophysics_analyst", "hydrophysics_analyst/*",
                 "dataprep_staging/honesty", "dataprep_staging/honesty/*",
                 "docs/analyst", "docs/analyst/*", "results/analyst", "results/analyst/*",
                 "hydrophysics/twin/progress.py", "hydrophysics/twin/progress_map_v1.json")
REDACTED = "[redacted: ruling on the claim-honesty audit itself (R11.4)]"
REDACT_RULING = re.compile(r"audit report|expected findings?|round-9 findings?", re.I)
REDACT_CONTEXT = re.compile(r"audit|honesty|grader|finding", re.I)
LITERAL_WINDOW = 400
LITERAL_TERMS = re.compile(r"honesty|(?:6|six)[-_ ]part|headline|progress|overall|audit|grader"
                           r"|誠實|六部分|總分|頭條|進度|評分者", re.I)
PHYSICAL_UNIT = re.compile(r"\s*(?:cm|mm|m|km|Mm3|m3|kWh|GWh|MW|kg|t|yr|mm/yr|cm/yr|°|days?|"
                           r"wells?|公尺|公分|公釐)(?![A-Za-z0-9])")
MASKED_NOTE = ("Files of repo/ changed by the claim-honesty mask of user ruling R11.4 (figures "
               "replaced by placeholders, rulings on the audit itself redacted, or printed hash "
               "prefixes of such files masked). Their hashes do not match a hash printed for "
               "them elsewhere; that mismatch comes from the mask, not from the record.")


PIN_SCHEMA = "twin-honesty-pin-manifest/1"
OPERATOR_SECTION = "\n## Operator section"
MAPPINGS = ("official", "pre-R10.5")
_PROGRESS = """
import json, sys
from hydrophysics.twin import progress as P
led = json.load(open(sys.argv[1], encoding="utf-8"))
mp = None if sys.argv[3] == "official" else sys.argv[3]
res = P.recount(led, P.with_mapping(P.load_map(), mp), float(sys.argv[2]))
keys = ("official", "intended", "end_goal")
print(json.dumps({"six": [res["six_part"][k] for k in keys],
                  "science": [res["science"][k] for k in keys],
                  "parts": [v[k] for v in res["parts"].values() for k in keys]}))
"""


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


def _commit_date(repo: Path, commit: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%cs", commit],
                          check=True, capture_output=True, text=True).stdout.strip()


def _ruling_date(name: str) -> str | None:
    m = re.match(r"user_rulings_(\d{4}-\d{2}-\d{2})", name)
    return m.group(1) if m else None


def _copy_file(src: Path, dst: Path, rel: str, max_bytes: int, log: dict) -> int:
    if excluded(rel):
        log["excluded"].append(rel)
        return 0
    if src.stat().st_size > max_bytes:
        log["too_large"].append(rel)
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return 1


def _copy_record(src: Path, dst: Path, rel0: str, max_bytes: int, log: dict,
                 pin_date: str | None = None) -> int:
    n = 0
    for p in sorted(src.rglob("*")):
        sub = p.relative_to(src)
        rel = f"{rel0}/{sub.as_posix()}"
        if any(excluded(f"{rel0}/{a.as_posix()}") for a in sub.parents if a.parts):
            continue
        if not p.is_file():
            if excluded(rel):
                log["excluded"].append(rel)
            continue
        if rel0 == RULINGS and pin_date and (_ruling_date(p.name) or "") > pin_date:
            log["after_pin"].append(rel)
            continue
        n += _copy_file(p, dst / sub, rel, max_bytes, log)
    return n


def cited_records(inv: dict, repo: Path, records: tuple[str, ...]) -> list[str]:
    """The pinned page and the ``results/`` paths the statements' ``check_against`` cite
    outside ``records``, as files or directories of ``repo`` (a cited name prefix gives the
    files it starts)."""
    out: set[str] = {rel for rel in inv["inputs"] if rel.startswith("results/")}
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


def _text_files(tree: Path, max_bytes: int, skip: set[str]):
    for p in sorted(tree.rglob("*")):
        rel = p.relative_to(tree).as_posix()
        if rel in skip or not p.is_file() or p.is_symlink() or p.stat().st_size > max_bytes:
            continue
        try:
            yield rel, p, p.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue


def redact_rulings(raw: str) -> tuple[str, int]:
    """A rulings file with the rulings on the claim-honesty audit itself redacted (step 2)."""
    doc = json.loads(raw)
    if not isinstance(doc, dict):
        return raw, 0
    n = 0

    def blank(d: dict, keep: tuple[str, ...]) -> None:
        nonlocal n
        for k, v in d.items():
            if k not in keep and isinstance(v, str) and v != REDACTED:
                d[k] = REDACTED
                n += 1

    for r in doc.get("rulings") or []:
        if not isinstance(r, dict):
            continue
        applies = " ".join(r.get("applies_to") or []).lower()
        if "claim honesty" in applies or REDACT_RULING.search(json.dumps(r, ensure_ascii=False)):
            blank(r, ("id", "applies_to"))
    ctx = doc.get("context")
    if isinstance(ctx, str) and REDACT_CONTEXT.search(ctx):
        doc["context"], n = REDACTED, n + 1
    if not n:
        return raw, 0
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n", n


def literal_variants(lits) -> list[str]:
    """Each literal with its rounding neighbours: one more decimal within half a unit of its
    last digit, and one decimal fewer when two or more are given."""
    out: set[str] = set()
    for lit in lits:
        lit = str(lit).strip()
        out.add(lit)
        if not re.fullmatch(r"\d+(?:\.\d+)?", lit):
            continue
        d = Decimal(lit)
        k = -d.as_tuple().exponent
        step = Decimal(1).scaleb(-(k + 1))
        for i in range(-5, 6):
            out.add(f"{d + i * step:.{k + 1}f}")
        if k >= 2:
            out.add(f"{d:.{k - 1}f}")
    return sorted(out)


_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d])")


def _literal_hits(text: str, lits) -> list[str]:
    """The literals found within LITERAL_WINDOW characters of a term (no physical unit)."""
    lits = set(lits)
    if not lits:
        return []
    terms = [m.start() for m in LITERAL_TERMS.finditer(text)]
    if not terms:
        return []
    hits = []
    for m in _NUMBER.finditer(text):
        if m.group(0) not in lits:
            continue
        k = bisect.bisect_left(terms, m.start() - LITERAL_WINDOW)
        if k >= len(terms) or terms[k] > m.end() + LITERAL_WINDOW:
            continue
        if PHYSICAL_UNIT.match(text, m.end()):
            continue
        hits.append(f"{m.group(0)}: ...{text[max(0, m.start() - 60):m.end() + 30]}...")
    return hits


def grader_brief(raw: str) -> str:
    """The grader-facing part of the brief: everything before its operator section."""
    i = raw.find(OPERATOR_SECTION)
    return raw if i < 0 else raw[:i]


def _commit_time(repo: Path, commit: str) -> int:
    return int(subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%ct", commit],
                              check=True, capture_output=True, text=True).stdout.strip())


def data_layer(rel: str) -> bool:
    return any(fnmatch.fnmatchcase(rel, pat) for pat in DATA_LAYERS)


def pin_manifest(inv: dict, repo: Path, commit: str, records: tuple[str, ...] = DEFAULT_RECORDS,
                 max_mb: float = 64.0) -> dict:
    """The record files (path and sha256) the workspace may hold, listed at pin time: run it
    right after pinning, keep the file outside the repository, build from it. A record file
    modified after the pinned commit's commit time is refused (WorkspaceError lists them)."""
    repo = Path(repo).resolve()
    t_pin = _commit_time(repo, commit)
    late: list[str] = []
    files: dict[str, str] = {}
    log: dict = {"excluded": [], "too_large": [], "after_pin": []}
    max_bytes = int(max_mb * 1e6)
    pin_date = _commit_date(repo, commit)
    paths = [r for r in records] + cited_records(inv, repo, records)
    for rel in paths:
        src = repo / rel
        cands = [src] if src.is_file() else sorted(q for q in src.rglob("*") if q.is_file()) \
            if src.is_dir() else []
        for q in cands:
            r = q.relative_to(repo).as_posix()
            if excluded(r) or any(excluded(a.as_posix()) for a in Path(r).parents if a.parts):
                log["excluded"].append(r)
            elif r.startswith(RULINGS + "/") and (_ruling_date(q.name) or "") > pin_date:
                log["after_pin"].append(r)
            elif q.stat().st_size > max_bytes:
                log["too_large"].append(r)
            else:
                if q.stat().st_mtime > t_pin:
                    late.append(r)
                files[r] = C.sha256(q)
    if late:
        raise WorkspaceError(f"{len(late)} record files were modified after the pinned "
                             f"commit {commit[:12]}: " + ", ".join(late[:10])
                             + ("..." if len(late) > 10 else "")
                             + "; pin again after the record settles")
    return {"schema": PIN_SCHEMA, "commit": commit, "inventory_sha256": inv["sha256"],
            "files": dict(sorted(files.items())), "excluded": sorted(set(log["excluded"])),
            "after_pin": log["after_pin"], "too_large": log["too_large"]}


def progress_figures(tree: Path, repo: Path, commit: str, honesty: float,
                     mapping: str) -> dict:
    """The progress recount (the pinned tree's code) of the ledger at ``commit``."""
    led = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{H.LEDGER_PATH}"],
                         check=True, capture_output=True).stdout
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        fh.write(led)
        tmp = Path(fh.name)
    try:
        out = subprocess.run([sys.executable, "-B", "-c", _PROGRESS, str(tmp), str(honesty),
                              mapping],
                             cwd=tree, env={"PYTHONPATH": str(tree), "PATH": "/usr/bin:/bin",
                                            "PYTHONDONTWRITEBYTECODE": "1"},
                             capture_output=True, text=True)
    finally:
        tmp.unlink()
    if out.returncode:
        raise WorkspaceError(f"progress at {commit[:12]} ({mapping}) failed: "
                             f"{out.stderr.strip()[-300:]}")
    return json.loads(out.stdout)


def figure_literals(xs) -> list[str]:
    """Two- and one-decimal forms of each figure, with their rounding neighbours."""
    out: list[str] = []
    for x in xs:
        if x is None:
            continue
        out += [f"{float(x):.2f}", f"{float(x):.1f}"]
    return literal_variants(out)


def pinned_figures(tree: Path, repo: Path, commit: str, honesty_input: dict,
                   score_summary: dict | None) -> tuple[list[str], list[str], dict]:
    """(hidden literals, visible literals, counts) built at the pin: the honesty input and the
    per-source audit figures (``score_summary``: its numeric summary only), every 6-part
    figure (official, intended, end-goal) of the pinned ledger and of each earlier pinned
    state, under each mapping; science and part figures are visible."""
    hidden: list[float] = [honesty_input["honesty"]]
    if score_summary is not None:
        hidden.append(score_summary.get("honesty"))
        hidden += [v.get("honesty") for v in (score_summary.get("per_source") or {}).values()]
        sc = score_summary.get("sample_check") or {}
        hidden += [sc.get("honesty_on_sample_auditor"), sc.get("honesty_on_sample_second")]
    visible: list[float] = []
    states = [(commit, honesty_input["honesty"])] + [
        (e["commit"], e["honesty"]) for e in honesty_input.get("earlier", [])]
    runs = 0
    for c, h in states:
        hidden.append(h)
        for mp in MAPPINGS:
            try:
                fig = progress_figures(tree, repo, c, h, mp)
            except WorkspaceError:
                if mp == "official":
                    raise
                continue
            hidden += fig["six"]
            visible += fig["science"] + fig["parts"]
            runs += 1
    hid, vis = figure_literals(hidden), figure_literals(visible)
    shared = set(hid) & set(vis)              # the same digits as a science or part figure
    return [h for h in hid if h not in shared], [v for v in vis if v not in shared], {
        "states": len(states), "recounts": runs, "hidden": len(hid) - len(shared),
        "visible": len(vis) - len(shared), "shared_not_gated": len(shared)}


def tree_mismatch(repo: Path, commit: str, tree: Path, pin: dict) -> dict:
    """The shipped tree against (tracked files at the commit + the pin manifest's files) minus
    the declared exclusions, file for file: {'extra': [...], 'missing': [...]}."""
    tracked = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    want = {r for r in set(tracked) | set(pin["files"])
            if not excluded(r) and not any(excluded(a.as_posix()) for a in Path(r).parents
                                           if a.parts)}
    have = {p.relative_to(tree).as_posix() for p in tree.rglob("*")
            if p.is_file() or p.is_symlink()}
    return {"extra": sorted(have - want), "missing": sorted(want - have)}


def make_workspace(inventory: Path, dest: Path, repo: Path = REPO, commit: str | None = None,
                   records: tuple[str, ...] = DEFAULT_RECORDS, sample: Path | None = None,
                   max_mb: float = 64.0, literals: tuple[str, ...] = (),
                   visible: tuple[str, ...] = (), allow: tuple[dict, ...] = (),
                   pin: dict | None = None, honesty_input: dict | None = None,
                   score_summary: dict | None = None, pin_sha256: str | None = None,
                   free_window: tuple[float, float] = (90.0, 100.0)) -> dict:
    repo, dest = Path(repo).resolve(), Path(dest).resolve()
    if not literals and honesty_input is None:
        raise WorkspaceError("the literal gate needs the current figures (--honesty-input, "
                             "--figures or --check-literal)")
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
    pin_date = _commit_date(repo, commit)
    max_bytes = int(max_mb * 1e6)
    log: dict = {"excluded": [], "too_large": [], "after_pin": [], "records": {},
                 "masked_files": {}, "redacted": {}, "hash_masked": {}}
    tree = dest / "repo"
    dest.mkdir(parents=True, exist_ok=True)
    _archive(repo, commit, tree)
    if pin is not None:
        if pin.get("schema") != PIN_SCHEMA or pin.get("commit") != commit or \
                pin.get("inventory_sha256") != inv["sha256"]:
            raise WorkspaceError("the pin manifest is not for this commit and inventory")
        for rel, h in pin["files"].items():
            src = repo / rel
            if not src.is_file() or C.sha256(src) != h:
                raise WorkspaceError(f"{rel}: changed or gone since the pin manifest was made")
            (tree / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, tree / rel)
        log["records"] = {"pinned files": len(pin["files"])}
    else:
        for rel in records:
            src = repo / rel
            log["records"][rel] = _copy_record(src, tree / rel, rel, max_bytes, log,
                                               pin_date) if src.is_dir() else "missing"
        for rel in cited_records(inv, repo, records):
            src = repo / rel
            log["records"][rel] = _copy_record(src, tree / rel, rel, max_bytes, log,
                                               pin_date) if src.is_dir() else \
                _copy_file(src, tree / rel, rel, max_bytes, log)
    for rel, h in inv["inputs"].items():
        p = tree / rel
        if not p.is_file() or C.sha256(p) != h:
            raise WorkspaceError(f"{rel}: missing from the workspace or not the inventory's "
                                 "input (sha256 differs); pin the commit the inventory was "
                                 "built on and copy the record that holds the page")
    if not (tree / BRIEF).is_file():
        raise WorkspaceError(f"{BRIEF} is not in the tree at {commit}")
    (dest / "GRADER_BRIEF.md").write_bytes(
        grader_brief((tree / BRIEF).read_text(encoding="utf-8")).encode("utf-8"))
    fig_counts, pin_hid, pin_vis = None, [], []
    if honesty_input is not None:       # already with their roundings: not expanded again
        pin_hid, pin_vis, fig_counts = pinned_figures(tree, repo, commit, honesty_input,
                                                      score_summary)
    ctx = dest / ".view_ctx"
    for rel in (H.README, H.MODEL_CARD, H.EVIDENCE):
        (ctx / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tree / rel, ctx / rel)
    _prune(tree, log)

    # 1-2: mask and redact; keep the digests of every changed file's original content
    digs: list[str] = []
    for rel, p, raw in _text_files(tree, max_bytes, {BRIEF}):
        new = raw
        res = None if data_layer(rel) else M.mask_file_text(p.name, raw)
        if res is not None and res[1]:
            new = res[0]
            log["masked_files"][rel] = res[1]
        if rel.startswith(RULINGS + "/") and p.suffix == ".json":
            try:
                new, k = redact_rulings(new)
            except ValueError:
                k = 0
            if k:
                log["redacted"][rel] = k
        if new != raw:
            digs += M.digests(raw.encode("utf-8"))
            p.write_text(new, encoding="utf-8")
    # 3: printed hash prefixes of changed files, until no file changes
    pending = list(digs)
    while pending:
        new_digs: list[str] = []
        for rel, p, raw in _text_files(tree, max_bytes, {BRIEF}):
            new, k = M.mask_hashes(raw, pending)
            if k:
                log["hash_masked"][rel] = log["hash_masked"].get(rel, 0) + k
                new_digs += M.digests(raw.encode("utf-8"))
                p.write_text(new, encoding="utf-8")
        digs += new_digs
        pending = new_digs

    # the view: masked with the unmasked docs as context, then the hash mask
    try:
        view = H.grader_view(inv, ctx, post=lambda s: M.mask_hashes(s, digs),
                             excluded=excluded)
    finally:
        shutil.rmtree(ctx)
    (dest / "inventory.json").write_text(json.dumps(view, ensure_ascii=False, indent=1) + "\n",
                                         encoding="utf-8")
    changed = sorted(set(log["masked_files"]) | set(log["redacted"]) | set(log["hash_masked"]))
    (dest / "MASKED_FILES.txt").write_text(
        MASKED_NOTE + "\n\n" + "\n".join(f"repo/{r}" for r in changed) + "\n", encoding="utf-8")
    ids = H.view_ids(inv)
    if sample is not None:
        got = json.loads(Path(sample).read_text(encoding="utf-8"))
        got = got.get("sample", got) if isinstance(got, dict) else got
        vids = set(ids.values())
        out = [s if s in vids else ids.get(s) for s in got]
        if None in out:
            raise WorkspaceError("the sample names an id neither the inventory nor the view has")
        (dest / "sample.json").write_text(json.dumps({"sample": out}, indent=1) + "\n",
                                          encoding="utf-8")

    # gates
    lits = sorted(set(literal_variants(literals)) | set(pin_hid))
    vis = sorted((set(literal_variants(visible)) | set(pin_vis)) - set(lits))
    derivable, literal_hits, visible_hits, allowed, free, ids_left = [], [], [], [], [], []
    def allowed_by(rel: str, key: str, value: str = "") -> bool:
        return any(x.get("file") in (rel, rel.removeprefix("repo/")) and (
            x.get(key) is True or (value and value.startswith(str(x.get(key)) + ":")))
            for x in allow)

    for rel, p, raw in _text_files(dest, max_bytes, {"inventory.json"}):
        units = M.units(p.name, raw) if p.suffix.lower() in M.JSON_SUFFIXES else [raw]
        for u in ([] if data_layer(rel.removeprefix("repo/")) else M.units(p.name, raw)):
            for a, b, h in M.derivable_pairs(u):
                line = f"{rel}: 6x{a:g} - 5x{b:g} = {h:g}"
                (allowed if allowed_by(rel, "derivable") else derivable).append(line)
            free += [f"{rel}: 6x{a:g} - 5x{b:g} = {h:g}" for a, b, h in
                     M.free_pairs(u, free_window)]
        ids_left += [f"{rel}: {m.group(0)}" for m in M._ID.finditer(raw)]
        for u in units:
            for h in _literal_hits(u, lits):
                (allowed if allowed_by(rel, "literal", h) else literal_hits).append(
                    f"{rel}: {h}")
            visible_hits += [f"{rel}: {h}" for h in _literal_hits(u, vis)]
    for r in view["records"]:
        for t in filter(None, r["text"].values()):
            where = f"inventory.json:{r['id']}"
            derivable += [f"{where}: 6x{a:g} - 5x{b:g} = {h:g}" for a, b, h in
                          M.derivable_pairs(t)]
            literal_hits += [f"{where}: {h}" for h in _literal_hits(t, lits)]
            visible_hits += [f"{where}: {h}" for h in _literal_hits(t, vis)]
            free += [f"{where}: 6x{a:g} - 5x{b:g} = {h:g}" for a, b, h in
                     M.free_pairs(t, free_window)]
            ids_left += [f"{where}: {m.group(0)}" for m in M._ID.finditer(t)]
    tree_diff = tree_mismatch(repo, commit, tree, pin) if pin is not None else None
    pages = sorted(r for r in (log["records"] if pin is None else pin["files"])
                   if re.fullmatch(r"results/twin/twin_app_\w+\.html", r)
                   and r not in inv["inputs"])
    return {"dest": str(dest), "commit": commit, "pin_date": pin_date,
            "inventory_sha256": inv["sha256"], "view_sha256": view["view_sha256"],
            "mask_rules_sha256": view["mask_rules_sha256"], "masked_statements": view["masked"],
            "unverifiable": view["unverifiable"], "bias_disclosure": view["bias_disclosure"],
            "pinned": pin is not None, "pin_manifest_sha256": pin_sha256,
            "figures": fig_counts, "records": log["records"],
            "disclosures": {
                "R11.4a": "Left out of the workspace, against the brief's wording 'the "
                          "repository at the pinned commit' (user ruling R11.4a): "
                          + ", ".join(x for x in EXCLUDE_PATHS if not x.endswith("/*")) + ".",
                "R11.4b": "Earlier pages the statements cite were in the workspace, so the "
                          "page history including round-9 fixes was visible to graders (user "
                          "ruling R11.4b): " + (", ".join(pages) or "none") + "."},
            "excluded": sorted(set(log["excluded"])),
            "after_pin": log["after_pin"] if pin is None else pin.get("after_pin", []),
            "too_large": log["too_large"], "masked_files": log["masked_files"],
            "redacted": log["redacted"], "hash_masked": log["hash_masked"],
            "gates": {"literals_checked": len(lits), "derivable": derivable,
                      "literal_hits": literal_hits, "visible_literal_hits": visible_hits,
                      "allowed_literal_hits": allowed, "statement_ids_left": ids_left,
                      "term_free_pairs": free, "free_window": list(free_window),
                      "tree_mismatch": tree_diff,
                      "passed": not derivable and not literal_hits and not ids_left
                      and not (tree_diff and (tree_diff["extra"] or tree_diff["missing"]))},
            "view_ids": {v: k for k, v in ids.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inventory", type=Path, required=True, help="the hashed inventory")
    ap.add_argument("--dest", type=Path, default=None, help="new directory outside the repo")
    ap.add_argument("--repo", type=Path, default=REPO)
    ap.add_argument("--commit", default=None, help="default: the inventory's commit")
    ap.add_argument("--record", action="append", default=None,
                    help=f"gitignored record directory (repeatable; default "
                         f"{', '.join(DEFAULT_RECORDS)})")
    ap.add_argument("--write-pin-manifest", type=Path, default=None,
                    help="at pin time: list the record files and their sha256 here and stop")
    ap.add_argument("--pin-manifest", type=Path, default=None,
                    help="build from this pin manifest (required unless --unpinned)")
    ap.add_argument("--unpinned", action="store_true",
                    help="copy the live record tree (testing only)")
    ap.add_argument("--sample", type=Path, default=None,
                    help="the second grader's statement ids (inventory or view ids)")
    ap.add_argument("--max-mb", type=float, default=64.0)
    ap.add_argument("--honesty-input", type=Path, default=None,
                    help='JSON {"honesty": X, "earlier": [{"commit": C, "honesty": Y}, ...]} '
                         "kept outside the repository: the literal gate's figures are built "
                         "from it at the pin")
    ap.add_argument("--score-summary", type=Path, default=None,
                    help="an earlier audit's score.json: only its numeric honesty summary "
                         "is read (overall, per source, sample)")
    ap.add_argument("--figures", type=Path, default=None,
                    help='extra figures: a JSON list, or {"hidden": [...], "visible": [...]}')
    ap.add_argument("--check-literal", action="append", default=[])
    ap.add_argument("--free-window", type=float, nargs=2, default=(90.0, 100.0),
                    metavar=("LO", "HI"), help="6a - 5b window of the term-free review list")
    ap.add_argument("--allow", type=Path, default=None,
                    help="JSON list of reviewed gate hits to accept: {file (under repo/ or the "
                         "workspace root), literal (a value) or derivable (true), reason}; keep "
                         "it outside the repository")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="write the operator summary here (outside the workspace)")
    a = ap.parse_args(argv)
    for opt in (a.honesty_input, a.figures, a.allow, a.write_pin_manifest):
        if opt is not None and _inside(opt, a.repo):
            raise WorkspaceError(f"{opt}: keep operator files outside the repository")
    records = tuple(a.record) if a.record else DEFAULT_RECORDS
    inv = H.load_inventory(a.inventory)
    if a.write_pin_manifest is not None:
        man = pin_manifest(inv, a.repo, a.commit or inv["commit"], records, a.max_mb)
        a.write_pin_manifest.parent.mkdir(parents=True, exist_ok=True)
        a.write_pin_manifest.write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
        print(f"pin manifest {a.write_pin_manifest}: {len(man['files'])} files at "
              f"{man['commit'][:12]}")
        return 0
    if a.dest is None:
        ap.error("--dest is required to build a workspace")
    if a.pin_manifest is None and not a.unpinned:
        ap.error("--pin-manifest is required (or --unpinned for a test build)")
    if a.manifest is not None and _inside(a.manifest, a.dest):
        raise WorkspaceError("the operator summary must not be written into the workspace")
    lits, vis = list(a.check_literal), []
    if a.figures is not None:
        fig = json.loads(a.figures.read_text(encoding="utf-8"))
        if isinstance(fig, dict):
            lits += [str(x) for x in fig.get("hidden", [])]
            vis += [str(x) for x in fig.get("visible", [])]
        else:
            lits += [str(x) for x in fig]
    hon = json.loads(a.honesty_input.read_text(encoding="utf-8")) if a.honesty_input else None
    score = None
    if a.score_summary is not None:
        full = json.loads(a.score_summary.read_text(encoding="utf-8"))
        score = {"honesty": full.get("honesty"),
                 "per_source": {k: {"honesty": v.get("honesty")}
                                for k, v in (full.get("per_source") or {}).items()},
                 "sample_check": {k: (full.get("sample_check") or {}).get(k) for k in
                                  ("honesty_on_sample_auditor", "honesty_on_sample_second")}}
        del full
    pin = json.loads(a.pin_manifest.read_text(encoding="utf-8")) if a.pin_manifest else None
    pin_sha = C.sha256(a.pin_manifest) if a.pin_manifest else None
    allow = tuple(json.loads(a.allow.read_text(encoding="utf-8"))) if a.allow else ()
    res = make_workspace(a.inventory, a.dest, a.repo, a.commit, records, a.sample, a.max_mb,
                         tuple(lits), tuple(vis), allow, pin, hon, score, pin_sha,
                         tuple(a.free_window))
    if a.manifest is not None:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    g = res["gates"]
    print(f"workspace {res['dest']}: view {res['view_sha256']}, mask rules "
          f"{res['mask_rules_sha256']}; gates {'passed' if g['passed'] else 'FAILED'} "
          f"({len(g['derivable'])} derivable pairs, {len(g['literal_hits'])} literal hits, "
          f"{len(g['statement_ids_left'])} statement ids; {len(g['term_free_pairs'])} "
          "term-free pairs to review)" + (f"; summary {a.manifest}" if a.manifest else ""))
    if not g["passed"]:
        td = g["tree_mismatch"] or {"extra": [], "missing": []}
        for line in (g["derivable"] + g["literal_hits"] + g["statement_ids_left"]
                     + [f"extra file {x}" for x in td["extra"]]
                     + [f"missing file {x}" for x in td["missing"]])[:60]:
            print("  GATE", line[:300])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
