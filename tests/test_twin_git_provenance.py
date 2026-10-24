"""calibrate_flow records the commit of the package that runs, not of the working directory.

Review code-07 (2026-10-24): batches launched from the main checkout with a worktree on
PYTHONPATH recorded the main checkout's HEAD, a commit that did not hold the code that ran.
"""

from __future__ import annotations

import re
import subprocess

import pytest

pytest.importorskip("torch")

from hydrophysics.twin import calibrate_flow as cf  # noqa: E402


def _package_head() -> str:
    out = subprocess.run(["git", "-C", cf._PACKAGE_DIR, "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip("the package is not in a git checkout")
    return out.stdout.strip()


def test_commit_is_the_packages_whatever_the_working_directory(tmp_path, monkeypatch):
    head = _package_head()
    monkeypatch.chdir(tmp_path)                 # not a git checkout at all
    short = cf._git_commit()
    assert short and head.startswith(short)
    prov = cf._git_provenance()
    assert prov["git_commit_full"] == head


def test_provenance_records_a_dirty_flag_and_the_diff_hash():
    _package_head()
    prov = cf._git_provenance()
    assert set(prov) == {"git_commit_full", "git_dirty", "git_diff_sha256", "git_untracked"}
    assert isinstance(prov["git_dirty"], bool) and isinstance(prov["git_untracked"], int)
    # a clean package has no diff; a dirty one names its diff by sha256
    assert prov["git_diff_sha256"] == "" or re.fullmatch(r"[0-9a-f]{64}", prov["git_diff_sha256"])
    if not prov["git_dirty"]:
        assert prov["git_diff_sha256"] == "" and prov["git_untracked"] == 0


def test_provenance_never_raises_without_git(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("git")

    monkeypatch.setattr(cf.subprocess, "run", boom)
    assert cf._git_commit() == "" and cf._git_provenance() == {}
