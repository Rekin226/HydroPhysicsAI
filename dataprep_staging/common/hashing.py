"""Content hashing for manifests, SOURCE.json files and grid masks (plan 2.3)."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
CHUNK = 1 << 20


def is_sha256(s: object) -> bool:
    return isinstance(s, str) and bool(_HEX64.match(s))


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str | os.PathLike, chunk: int = CHUNK) -> str:
    """Streaming sha256 of a file's bytes (symlinks are followed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_files(root: str | os.PathLike, exclude_names: tuple[str, ...] = ()) -> list[Path]:
    """Sorted regular files under ``root`` (recursive), skipping ``__pycache__`` and any
    file whose name is in ``exclude_names``."""
    root = Path(root)
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for fn in sorted(filenames):
            if fn in exclude_names:
                continue
            p = Path(dirpath) / fn
            if p.is_file():
                out.append(p)
    return out


def sha256_tree(root: str | os.PathLike, exclude_names: tuple[str, ...] = ()) -> str:
    """Deterministic digest of a directory: sha256 over sorted 'relpath\\0sha256\\n' lines."""
    root = Path(root)
    h = hashlib.sha256()
    for p in iter_files(root, exclude_names):
        rel = p.relative_to(root).as_posix()
        h.update(f"{rel}\0{sha256_file(p)}\n".encode())
    return h.hexdigest()


def file_record(path: str | os.PathLike, relto: str | os.PathLike | None = None,
                **extra) -> dict:
    """{'path', 'sha256', 'bytes', **extra}; ``path`` relative to ``relto`` when given."""
    p = Path(path)
    rel = p.resolve().relative_to(Path(relto).resolve()).as_posix() if relto else str(p)
    rec = {"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size}
    rec.update(extra)
    return rec


def array_hash(a: np.ndarray) -> str:
    """sha256 over dtype, shape and C-order bytes of an array."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def mask_hash(mask: np.ndarray) -> str:
    """Hash of a grid mask (plan 2.1 L3: arrays carry the mask hash). Booleanised first, so
    a 0/1 integer mask and a boolean mask with the same cells hash identically."""
    return array_hash(np.asarray(mask).astype(bool).astype(np.uint8))


def grid_hash(nx: int, ny: int, dx: float, x0: float, y0: float, mask: np.ndarray) -> str:
    """Hash of the grid parameters plus its mask (for L3 arrays and RELEASE.json)."""
    h = hashlib.sha256()
    h.update(f"nx={int(nx)}|ny={int(ny)}|dx={float(dx)!r}|x0={float(x0)!r}|y0={float(y0)!r}|"
             .encode())
    h.update(mask_hash(mask).encode())
    return h.hexdigest()
