"""Read-only git queries for manifests (``git_commit`` and whether code files are committed).

Uses GIT_OPTIONAL_LOCKS=0 so that no query refreshes or locks the index. Never stages,
commits or writes anything.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from common.paths import REPO

_ENV = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                           env=_ENV, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def head_commit() -> str | None:
    return _git("rev-parse", "HEAD")


def code_state(paths: list[str | os.PathLike]) -> dict[str, str]:
    """{repo-relative path: 'committed' | 'modified' | 'untracked' | 'outside_repo'}."""
    out = {}
    for p in paths:
        ap = Path(p).resolve()
        try:
            rel = ap.relative_to(REPO).as_posix()
        except ValueError:
            out[ap.as_posix()] = "outside_repo"
            continue
        tracked = _git("ls-files", "--error-unmatch", "--", rel)
        if tracked is None:
            out[rel] = "untracked"
            continue
        diff = _git("diff", "--name-only", "HEAD", "--", rel)
        out[rel] = "modified" if diff else "committed"
    return out
