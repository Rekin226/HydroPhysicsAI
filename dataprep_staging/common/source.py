"""L0 download records: ``data_fetch_api/new_measurements/<source>/SOURCE.json`` (plan 2.1).

Extends the existing convention (fetch date, publisher, URL, bytes per file) with a sha256
and a licence note per source. A rolling source (a file the publisher replaces, such as TPC
d007012) is archived in a dated subdirectory per fetch, each with its own SOURCE.json.
L0 is never modified: :func:`write_source_json` refuses to replace an existing record.
Downloaded files are untrusted data: nothing here opens them except to hash their bytes.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from common.hashing import is_sha256, sha256_file

SOURCE_NAME = "SOURCE.json"
SCHEMA_VERSION = "dataprep-source/1"
REQUIRED = ("source_id", "title", "publisher", "licence", "fetched", "files")
FILE_REQUIRED = ("file", "url", "bytes", "sha256")


class SourceError(ValueError):
    pass


def copy_into_l0(src: str | os.PathLike, dest: str | os.PathLike) -> Path:
    """Copy one file into an L0 directory. Refuses to replace an existing file whose bytes
    differ; an identical existing file is left untouched. The copy is made read-only."""
    src, dest = Path(src), Path(dest)
    if dest.exists():
        if sha256_file(dest) != sha256_file(src):
            raise SourceError(f"{dest} exists with different content; L0 is never modified")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    os.chmod(dest, 0o444)
    return dest


def build_source(directory: str | os.PathLike, *, source_id: str, title: str, publisher: str,
                 licence: str, fetched: str, files: list[dict], **extra) -> dict:
    """Record for ``directory``; each entry of ``files`` needs 'file' (relative to the
    directory) and 'url' (None if unknown, with a 'note'); bytes and sha256 are computed."""
    d = Path(directory)
    recs = []
    for f in files:
        rec = dict(f)
        p = d / rec["file"]
        if not p.is_file():
            raise SourceError(f"{p} missing")
        rec["bytes"] = p.stat().st_size
        rec["sha256"] = sha256_file(p)
        recs.append(rec)
    m = {"schema": SCHEMA_VERSION, "source_id": source_id, "title": title,
         "publisher": publisher, "licence": licence, "fetched": fetched}
    m.update(extra)
    m["files"] = recs
    return m


def write_source_json(directory: str | os.PathLike, record: dict) -> Path:
    """Write SOURCE.json; if one exists it must be identical (L0 is never modified)."""
    d = Path(directory)
    errs = validate_source(record, d)
    if errs:
        raise SourceError("; ".join(errs))
    p = d / SOURCE_NAME
    text = json.dumps(record, ensure_ascii=False, indent=1) + "\n"
    if p.exists():
        if p.read_text(encoding="utf-8") != text:
            raise SourceError(f"{p} exists with different content; L0 is never modified")
        return p
    with open(p, "x", encoding="utf-8") as f:
        f.write(text)
    os.chmod(p, 0o444)
    return p


def validate_source(rec: dict, directory: str | os.PathLike | None = None,
                    rehash: bool = True) -> list[str]:
    """Errors for a dataprep-source/1 record (and, with ``directory``, its files' bytes and
    sha256)."""
    e = [f"missing key {k!r}" for k in REQUIRED if k not in rec]
    files = rec.get("files") or []
    if not isinstance(files, list) or not files:
        e.append("files must be a non-empty list")
        return e
    for i, f in enumerate(files):
        miss = [k for k in FILE_REQUIRED if k not in f]
        if miss:
            e.append(f"files[{i}] missing {miss}")
            continue
        if f["url"] is None and not f.get("note"):
            e.append(f"files[{i}] has no url and no note explaining why")
        if not is_sha256(f["sha256"]):
            e.append(f"files[{i}] sha256 malformed")
        if directory is not None:
            p = Path(directory) / f["file"]
            if not p.is_file():
                e.append(f"files[{i}] {f['file']} missing")
            elif p.stat().st_size != f["bytes"]:
                e.append(f"files[{i}] {f['file']} bytes {p.stat().st_size} != {f['bytes']}")
            elif rehash and sha256_file(p) != f["sha256"]:
                e.append(f"files[{i}] {f['file']} sha256 mismatch")
    return e


def listed_files(path: str | os.PathLike) -> list[Path]:
    """Absolute paths of the files a SOURCE.json lists (dataprep or legacy layout), plus the
    SOURCE.json itself. Used by the checker to find L0 files no record covers."""
    p = Path(path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    files = rec.get("files")
    names = (list(files) if isinstance(files, dict)
             else [f.get("file") for f in files or [] if isinstance(f, dict)])
    return [p.resolve()] + [(p.parent / str(n)).resolve() for n in names if n]


def check_source_file(path: str | os.PathLike) -> dict:
    """{'path', 'status': 'ok'|'legacy'|'fail', 'problems': [...]} for one SOURCE.json.
    Legacy records (written before dataprep-source/1, without sha256) are checked for file
    presence and bytes only and reported as 'legacy'."""
    p = Path(path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("schema") == SCHEMA_VERSION:
        probs = validate_source(rec, p.parent)
        return {"path": str(p), "status": "fail" if probs else "ok", "problems": probs,
                "n_files": len(rec.get("files") or [])}
    probs, n = [], 0
    files = rec.get("files")
    items = (files.items() if isinstance(files, dict)
             else ((f.get("file"), f) for f in files or []))
    for name, f in items:
        n += 1
        q = p.parent / str(name)
        if not q.is_file():
            probs.append(f"{name} missing")
        elif isinstance(f, dict) and "bytes" in f and q.stat().st_size != f["bytes"]:
            probs.append(f"{name} bytes differ")
    return {"path": str(p), "status": "fail" if probs else "legacy", "problems": probs,
            "n_files": n, "note": "pre-dataprep record: no sha256/licence (read-only, kept)"}
