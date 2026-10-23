"""MANIFEST.json schema, writer, validator and hash verification (plan 2.3).

Every L1-L3 product version directory ``results/dataprep/<product>/v<N>/`` holds a
``MANIFEST.json``. Paths: ``inputs`` and ``code`` are repository-relative (absolute if outside
the repository); ``outputs`` and ``validation_report`` are relative to the product directory.
Pilot directories under ``results/dataprep/_pilots/`` use the same schema with
``layer = "pilot"``; the L0 registry (``results/dataprep/l0_registry/v<N>/``, hashes of L0
files that predate SOURCE.json) uses ``layer = "registry"``.

Schema ``dataprep-manifest/2`` (repair round, 2026-10-12) adds, all optional:

* ``non_exempt_fit_window_end`` - required whenever ``exemptions`` is non-empty: the fit
  window end ignoring the exempt inputs. ``fit_window_end`` must equal
  ``max(non_exempt_fit_window_end, exemptions[*].data_end)``. An exemption names the uses
  it covers explicitly (no ``*``), only from :data:`common.uses.EXEMPTABLE_USES`.
* ``time_slicing`` - ``{"column": <date column>, "params_fit_window_end": "YYYY-MM"}`` for
  a product whose rows are dated and whose own fitted parameters saw data only up to
  ``params_fit_window_end``; a consumer may then read rows up to a ``window_end`` it
  declares (the guard filters them). Not combinable with exemptions.
* ``outputs[i].fit_window_end`` - a per-output window, at most the product's, for products
  (pilots) that mix pre-origin and post-origin outputs.
* ``input_trees`` - ``[{"path", "sha256_tree", "n_files"}]`` for input directories read as
  a whole (``hashing.sha256_tree``); re-hashed by the checker.

Pre-origin consistency no longer depends on the product name alone: every pre-origin use in
``allowed_uses`` must be reachable (``fit_window_end <= ORIGIN``, a covering exemption, a time
slice, or at least one pre-origin output), and a product whose name says pre-origin
(``preorigin``, ``pre_origin``, ``pre-origin``) must have ``fit_window_end <= ORIGIN`` or full
exemption coverage.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path

from common import gitinfo
from common.hashing import file_record, is_sha256, iter_files, sha256_file, sha256_tree
from common.paths import DATAPREP, ORIGIN, from_repo, rel_to_repo
from common.uses import EXEMPTABLE_USES, PRE_ORIGIN_USES, USES

MANIFEST_NAME = "MANIFEST.json"
SCHEMA_VERSION = "dataprep-manifest/2"
SCHEMAS = ("dataprep-manifest/1", "dataprep-manifest/2")
LAYERS = ("L1", "L2", "L3", "pilot", "registry")
#: Layers whose version need not be v<N> and whose validation_report may be null.
RELAXED_LAYERS = ("pilot", "registry")
REQUIRED = ("product", "version", "created", "layer", "inputs", "code", "git_commit",
            "parameters", "fit_window_end", "exemptions", "outputs", "validation_report",
            "gate", "allowed_uses", "forbidden_uses", "personal_data")
EXEMPTION_KEYS = ("input", "data_end", "reason", "ledger_number_id", "sensitivity", "uses")
_YM = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_YMD = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VN = re.compile(r"^v[1-9]\d*$")
_PREORIGIN_NAME = re.compile(r"pre[_\-]?origin", re.I)


class ManifestError(ValueError):
    """The manifest does not satisfy the schema."""


class HashMismatch(RuntimeError):
    """A file's sha256 differs from the one recorded in its manifest."""


# ----------------------------------------------------------------------------- versions
def new_version_dir(product: str, root: str | os.PathLike = DATAPREP) -> Path:
    """Create and return ``<root>/<product>/v<N+1>`` (exclusive mkdir; never reuses a version)."""
    base = Path(root) / product
    base.mkdir(parents=True, exist_ok=True)
    while True:
        ns = [int(p.name[1:]) for p in base.iterdir() if p.is_dir() and _VN.match(p.name)]
        d = base / f"v{max(ns, default=0) + 1}"
        try:
            d.mkdir()
            return d
        except FileExistsError:   # a concurrent builder took this number
            continue


# ----------------------------------------------------------------------------- records
def _rows(path: Path) -> int | None:
    try:
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows)
        if path.suffix == ".csv":
            with open(path, "rb") as f:
                n = sum(1 for _ in f)
            return max(n - 1, 0)
    except Exception:  # noqa: BLE001 - row count is informative only
        return None
    return None


def _input_record(x) -> dict:
    if isinstance(x, dict):
        rec = dict(x)
        p = from_repo(rec["path"])
    else:
        rec, p = {}, from_repo(x)
    if p.is_dir():
        raise ManifestError(f"input {p} is a directory; list its files")
    rec.update({"path": rel_to_repo(p), "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return rec


def build_manifest(product_dir: str | os.PathLike, *, product: str, layer: str,
                   inputs: list, code: list, parameters: dict, fit_window_end: str,
                   validation_report: str | None, gate: dict, allowed_uses: list[str],
                   forbidden_uses: list[str], personal_data: str = "none",
                   exemptions: list[dict] | None = None, outputs: list | None = None,
                   version: str | None = None, created: str | None = None,
                   non_exempt_fit_window_end: str | None = None,
                   time_slicing: dict | None = None, input_trees: list | None = None,
                   extra: dict | None = None) -> dict:
    """Assemble a manifest, hashing every input, code file, input tree and output now.

    ``outputs=None`` lists every file in ``product_dir`` except the manifest itself. Output
    records given as dicts keep their extra keys (``fit_window_end``, ``l0_copy``, ...).
    ``input_trees`` lists directories read as a whole (hashed with ``sha256_tree``).
    """
    d = Path(product_dir)
    if outputs is None:
        outputs = [p.relative_to(d).as_posix() for p in iter_files(d, (MANIFEST_NAME,))]
    outs = []
    for o in outputs:
        rec = dict(o) if isinstance(o, dict) else {"path": str(o)}
        p = d / rec["path"]
        rec.update(file_record(p, relto=d))
        if "rows" not in rec:
            n = _rows(p)
            if n is not None:
                rec["rows"] = n
        outs.append(rec)
    code_recs = [_input_record(c) for c in code]
    state = gitinfo.code_state([from_repo(c["path"]) for c in code_recs])
    for c in code_recs:
        c["git_state"] = state.get(c["path"], "unknown")
    m = {
        "schema": SCHEMA_VERSION,
        "product": product,
        "version": version or d.name,
        "created": created or _dt.date.today().isoformat(),
        "layer": layer,
        "inputs": [_input_record(x) for x in inputs],
        "code": code_recs,
        "git_commit": gitinfo.head_commit(),
        "parameters": parameters,
        "fit_window_end": fit_window_end,
        "exemptions": list(exemptions or []),
        "outputs": outs,
        "validation_report": validation_report,
        "gate": gate,
        "allowed_uses": list(allowed_uses),
        "forbidden_uses": list(forbidden_uses),
        "personal_data": personal_data,
    }
    if non_exempt_fit_window_end is not None:
        m["non_exempt_fit_window_end"] = non_exempt_fit_window_end
    if time_slicing is not None:
        m["time_slicing"] = dict(time_slicing)
    if input_trees:
        trees = []
        for t in input_trees:
            rec = dict(t) if isinstance(t, dict) else {"path": str(t)}
            p = from_repo(rec["path"])
            if not p.is_dir():
                raise ManifestError(f"input tree {p} is not a directory")
            rec.update({"path": rel_to_repo(p), "sha256_tree": sha256_tree(p),
                        "n_files": len(iter_files(p))})
            trees.append(rec)
        m["input_trees"] = trees
    if extra:
        clash = set(extra) & set(m)
        if clash:
            raise ManifestError(f"extra keys clash with schema keys: {sorted(clash)}")
        m.update(extra)
    return m


# ----------------------------------------------------------------------------- pre-origin
def is_preorigin_name(product: str) -> bool:
    """True when a product name says it is a pre-origin variant (any spelling)."""
    return bool(_PREORIGIN_NAME.search(str(product)))


def exemption_coverage(m: dict, use: str) -> tuple[bool, list[dict], str]:
    """(covered, covering exemptions, reason) for a pre-origin ``use`` of a product whose
    ``fit_window_end`` is later than the origin (U3/U5).

    Covered only if: the use is exemptable; ``non_exempt_fit_window_end`` is present and
    <= ORIGIN; and every exemption whose ``data_end`` is later than the origin lists the use
    explicitly (one uncovered post-origin input leaks into the use)."""
    ex = [x for x in (m.get("exemptions") or []) if isinstance(x, dict)]
    if not ex:
        return False, [], "no exemption listed"
    if use not in EXEMPTABLE_USES:
        return False, [], f"use {use!r} can never be exempted (targets, fits, selection)"
    ne = m.get("non_exempt_fit_window_end")
    if not (isinstance(ne, str) and _YM.match(ne)):
        return False, [], "non_exempt_fit_window_end missing"
    if ne > ORIGIN:
        return False, [], f"non_exempt_fit_window_end {ne} > {ORIGIN}"
    late = [x for x in ex if str(x.get("data_end", "")) > ORIGIN]
    uncovered = [x.get("input") for x in late if use not in (x.get("uses") or [])]
    if uncovered:
        return False, [], f"post-origin exempt inputs {uncovered} do not cover use {use!r}"
    return True, late, "covered"


def preorigin_routes(m: dict, use: str) -> dict:
    """How a pre-origin ``use`` may read this product:

    ``whole``   fit_window_end <= ORIGIN, or full exemption coverage;
    ``slice``   the product declares time_slicing with params_fit_window_end <= ORIGIN
                (the consumer must read a window ending by the origin);
    ``outputs`` outputs whose own fit_window_end <= ORIGIN (pilots mixing windows).
    """
    fwe = str(m.get("fit_window_end", "9999-12"))
    covered, late, why = (True, [], "fit_window_end <= origin") if fwe <= ORIGIN else \
        exemption_coverage(m, use)
    ts = m.get("time_slicing") or {}
    slice_ok = bool(ts) and str(ts.get("params_fit_window_end", "9999-12")) <= ORIGIN
    outs = [o["path"] for o in (m.get("outputs") or [])
            if isinstance(o, dict) and "fit_window_end" in o
            and str(o["fit_window_end"]) <= ORIGIN]
    return {"whole": covered, "exemptions": late, "reason": why, "slice": slice_ok,
            "outputs": outs}


# ----------------------------------------------------------------------------- validation
def validate_manifest(m: dict, product_dir: str | os.PathLike | None = None) -> list[str]:
    """Schema errors (empty list = valid). With ``product_dir``, also checks that the outputs
    and the validation report exist and that ``version`` matches the directory name."""
    e = []
    if not isinstance(m, dict):
        return ["manifest is not a JSON object"]
    for k in REQUIRED:
        if k not in m:
            e.append(f"missing key {k!r}")
    if e:
        return e
    if "schema" in m and m["schema"] not in SCHEMAS:
        e.append(f"schema {m['schema']!r} not in {SCHEMAS}")
    layer = m["layer"]
    if layer not in LAYERS:
        e.append(f"layer {layer!r} not in {LAYERS}")
    relaxed = layer in RELAXED_LAYERS
    if not isinstance(m["product"], str) or not m["product"]:
        e.append("product must be a non-empty string")
    if not relaxed and not (isinstance(m["version"], str) and _VN.match(m["version"])):
        e.append(f"version {m['version']!r} must look like v<N>")
    if not (isinstance(m["created"], str) and _YMD.match(m["created"])):
        e.append(f"created {m['created']!r} must be YYYY-MM-DD")
    fwe = m["fit_window_end"]
    if not (isinstance(fwe, str) and _YM.match(fwe)):
        e.append(f"fit_window_end {fwe!r} must be YYYY-MM")
        fwe = None
    for key in ("inputs", "code", "outputs"):
        recs = m[key]
        if not isinstance(recs, list):
            e.append(f"{key} must be a list")
            continue
        for i, r in enumerate(recs):
            if not isinstance(r, dict) or "path" not in r or "sha256" not in r:
                e.append(f"{key}[{i}] needs path and sha256")
            elif not is_sha256(r["sha256"]):
                e.append(f"{key}[{i}] sha256 is not 64 hex chars")
    for i, r in enumerate(m["outputs"] if isinstance(m["outputs"], list) else []):
        if isinstance(r, dict) and "fit_window_end" in r:
            ofw = r["fit_window_end"]
            if not (isinstance(ofw, str) and _YM.match(ofw)):
                e.append(f"outputs[{i}].fit_window_end must be YYYY-MM")
            elif fwe and ofw > fwe:
                e.append(f"outputs[{i}].fit_window_end {ofw} > product fit_window_end {fwe}")
    trees = m.get("input_trees")
    if trees is not None:
        if not isinstance(trees, list):
            e.append("input_trees must be a list")
        else:
            for i, t in enumerate(trees):
                if not (isinstance(t, dict) and "path" in t and is_sha256(t.get("sha256_tree"))):
                    e.append(f"input_trees[{i}] needs path and a sha256_tree")
    if layer != "pilot" and not m["code"]:
        e.append("code must list at least one file")
    if layer != "pilot" and not m["outputs"]:
        e.append("outputs must list at least one file")
    if not isinstance(m["parameters"], dict):
        e.append("parameters must be an object")
    gate = m["gate"]
    if not (isinstance(gate, dict) and isinstance(gate.get("passed"), bool)
            and isinstance(gate.get("criteria"), str)):
        e.append("gate must be {'passed': bool, 'criteria': str}")
    vr = m["validation_report"]
    if not relaxed and not (isinstance(vr, str) and vr):
        e.append("validation_report must name a file")
    au, fu = m["allowed_uses"], m["forbidden_uses"]
    uses_ok = isinstance(au, list) and isinstance(fu, list)
    if not uses_ok:
        e.append("allowed_uses and forbidden_uses must be lists")
    else:
        unknown = sorted((set(au) | set(fu)) - set(USES))
        if unknown:
            e.append(f"unknown uses {unknown}")
        both = sorted(set(au) & set(fu))
        if both:
            e.append(f"uses both allowed and forbidden: {both}")
    if not (isinstance(m["personal_data"], str) and m["personal_data"]):
        e.append("personal_data must be a non-empty string ('none' if none)")
    # ---- exemptions (U3)
    ex = m["exemptions"]
    if not isinstance(ex, list):
        e.append("exemptions must be a list")
        ex = []
    ends = []
    for i, x in enumerate(ex):
        if not isinstance(x, dict):
            e.append(f"exemptions[{i}] must be an object")
            continue
        miss = [k for k in EXEMPTION_KEYS if not x.get(k)]
        if miss:
            e.append(f"exemptions[{i}] missing {miss} (U3: ledger row and sensitivity run)")
        if x.get("data_end"):
            if _YM.match(str(x["data_end"])):
                ends.append(str(x["data_end"]))
            else:
                e.append(f"exemptions[{i}].data_end must be YYYY-MM")
        xu = x.get("uses")
        if xu is not None and not isinstance(xu, list):
            e.append(f"exemptions[{i}].uses must be a list of uses")
            xu = []
        xu = xu or []
        if "*" in xu:
            e.append(f"exemptions[{i}] uses '*': name the covered uses explicitly")
        bad = sorted(set(xu) - set(USES) - {"*"})
        if bad:
            e.append(f"exemptions[{i}] unknown uses {bad}")
        never = sorted((set(xu) & set(USES)) - EXEMPTABLE_USES)
        if never:
            e.append(f"exemptions[{i}] covers uses that can never be exempted: {never}")
    ne = m.get("non_exempt_fit_window_end")
    if ne is not None and not (isinstance(ne, str) and _YM.match(ne)):
        e.append("non_exempt_fit_window_end must be YYYY-MM")
        ne = None
    if ex:
        if ne is None:
            e.append("exemptions listed but non_exempt_fit_window_end is missing")
        elif fwe:
            if ne > fwe:
                e.append(f"non_exempt_fit_window_end {ne} > fit_window_end {fwe}")
            want = max([ne] + ends)
            if fwe != want:
                e.append(f"fit_window_end {fwe} != max(non_exempt_fit_window_end, "
                         f"exemptions' data_end) = {want}")
    # ---- time slicing
    ts = m.get("time_slicing")
    if ts is not None:
        if not (isinstance(ts, dict) and isinstance(ts.get("column"), str) and ts["column"]
                and isinstance(ts.get("params_fit_window_end"), str)
                and _YM.match(ts["params_fit_window_end"])):
            e.append("time_slicing must be {'column': str, 'params_fit_window_end': 'YYYY-MM'}")
        elif fwe and ts["params_fit_window_end"] > fwe:
            e.append("time_slicing.params_fit_window_end > fit_window_end")
        if ex:
            e.append("time_slicing cannot be combined with exemptions")
    # ---- pre-origin consistency (U3/U5), independent of the product name
    if fwe and uses_ok:
        for use in sorted(set(au) & PRE_ORIGIN_USES):
            if fwe <= ORIGIN:
                continue
            r = preorigin_routes(m, use)
            if not (r["whole"] or r["slice"] or r["outputs"]):
                e.append(f"allowed pre-origin use {use!r} can never pass the guard: "
                         f"fit_window_end {fwe} > {ORIGIN} and {r['reason']}, no time slice, "
                         f"no pre-origin output")
    if fwe and is_preorigin_name(m["product"]) and fwe > ORIGIN:
        if not ex:
            e.append(f"pre-origin product with fit_window_end {fwe} > {ORIGIN} and no exemption")
        elif ne is not None and ne > ORIGIN:
            e.append(f"pre-origin product with non_exempt_fit_window_end {ne} > {ORIGIN}")
    if product_dir is not None:
        d = Path(product_dir)
        if not relaxed and d.name != m["version"]:
            e.append(f"version {m['version']!r} != directory name {d.name!r}")
        for i, r in enumerate(m["outputs"] if isinstance(m["outputs"], list) else []):
            if isinstance(r, dict) and "path" in r and not (d / r["path"]).is_file():
                e.append(f"outputs[{i}] {r['path']} does not exist")
        if isinstance(vr, str) and vr and not (d / vr).is_file():
            e.append(f"validation_report {vr} does not exist")
    return e


def write_manifest(product_dir: str | os.PathLike, m: dict) -> Path:
    """Validate and write ``MANIFEST.json``; refuses to overwrite an existing manifest."""
    d = Path(product_dir)
    errs = validate_manifest(m, d)
    if errs:
        raise ManifestError("; ".join(errs))
    p = d / MANIFEST_NAME
    with open(p, "x", encoding="utf-8") as f:     # 'x': never overwrite a version
        json.dump(m, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return p


def load_manifest(path: str | os.PathLike) -> tuple[dict, Path]:
    """(manifest, product_dir) from a product directory or a MANIFEST.json path."""
    p = Path(path)
    if p.is_dir():
        p = p / MANIFEST_NAME
    with open(p, encoding="utf-8") as f:
        return json.load(f), p.parent


# ----------------------------------------------------------------------------- hashes
def verify_hashes(m: dict, product_dir: str | os.PathLike,
                  sections: tuple[str, ...] = ("inputs", "code", "outputs")) -> list[dict]:
    """Recompute every hash; returns one dict per problem (empty = all reproduced)."""
    d = Path(product_dir)
    probs = []
    if "inputs" in sections:
        for t in m.get("input_trees") or []:
            p = from_repo(t["path"])
            if not p.is_dir():
                probs.append({"section": "input_trees", "path": t["path"], "problem": "missing"})
                continue
            got = sha256_tree(p)
            if got != t["sha256_tree"]:
                probs.append({"section": "input_trees", "path": t["path"],
                              "problem": "sha256_tree", "expected": t["sha256_tree"],
                              "got": got})
    for sec in sections:
        for r in m.get(sec) or []:
            p = (d / r["path"]) if sec == "outputs" else from_repo(r["path"])
            if not p.is_file():
                probs.append({"section": sec, "path": r["path"], "problem": "missing"})
                continue
            got = sha256_file(p)
            if got != r["sha256"]:
                probs.append({"section": sec, "path": r["path"], "problem": "sha256",
                              "expected": r["sha256"], "got": got})
    return probs


def assert_inputs_match(m: dict, product_dir: str | os.PathLike) -> None:
    """Raise :class:`HashMismatch` if any recorded input or code file changed (2.3: a product
    refuses to build if an input hash differs from its manifest)."""
    probs = verify_hashes(m, product_dir, ("inputs", "code"))
    if probs:
        raise HashMismatch(json.dumps(probs, ensure_ascii=False))


def find_manifests(root: str | os.PathLike = DATAPREP) -> list[Path]:
    """Every MANIFEST.json under ``root`` (sorted)."""
    return sorted(Path(root).rglob(MANIFEST_NAME))
