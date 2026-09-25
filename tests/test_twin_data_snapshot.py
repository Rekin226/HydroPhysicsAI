from __future__ import annotations

import json

import pytest

from hydrophysics.twin import data_snapshot as ds


def _tree(root):
    (root / "chou-shui-data" / "data" / "a b").mkdir(parents=True)
    (root / "chou-shui-data" / "data" / "a b" / "fan.json").write_text("{}")
    (root / "AMP_V2" / "data" / "wells").mkdir(parents=True)
    (root / "AMP_V2" / "data" / "wells" / "w.parquet").write_bytes(b"\x00\x01")


def test_snapshot_verify_restore_roundtrip(tmp_path):
    repo, out = tmp_path / "repo", tmp_path / "bk"
    _tree(repo)
    tar, side = ds.snapshot(repo, out)
    man = json.loads(side.read_text())
    assert man["n_files"] == 2 and "tarball_sha256" in man
    assert not any(str(tmp_path) in k for k in man["files"])  # repo-relative paths only
    assert ds.verify(tar, tree_root=repo) == []

    (repo / "AMP_V2" / "data" / "wells" / "w.parquet").write_bytes(b"changed")
    assert ds.verify(tar, tree_root=repo) == ["tree: changed AMP_V2/data/wells/w.parquet"]

    with pytest.raises(FileExistsError):
        ds.restore(tar, repo)
    assert ds.restore(tar, repo, force=True) == 2
    assert ds.verify(tar, tree_root=repo) == []
    assert ds.restore(tar, tmp_path / "fresh", only=("AMP_V2",)) == 1


def test_corrupt_archive_detected(tmp_path):
    repo = tmp_path / "repo"
    _tree(repo)
    tar, _ = ds.snapshot(repo, tmp_path / "bk")
    raw = bytearray(tar.read_bytes())
    raw[-10] ^= 0xFF
    tar.write_bytes(bytes(raw))
    assert any("tarball sha256" in p for p in ds.verify(tar))


def test_refuses_backup_inside_repo(tmp_path):
    repo = tmp_path / "repo"
    _tree(repo)
    with pytest.raises(ValueError):
        ds.snapshot(repo, repo / "backups")
