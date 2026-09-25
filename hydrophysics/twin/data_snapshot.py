"""Back up, verify and restore the twin's gitignored data cache.

    python -m hydrophysics.twin.data_snapshot snapshot            # -> ~/twin_data_backups/
    python -m hydrophysics.twin.data_snapshot list
    python -m hydrophysics.twin.data_snapshot verify  <tarball>   # archive integrity
    python -m hydrophysics.twin.data_snapshot verify  <tarball> --tree   # live tree drift
    python -m hydrophysics.twin.data_snapshot restore <tarball> --dest <repo-root>

The twin reads two trees that are never committed: ``chou-shui-data/`` (the original data
delivery: fan polygon, rain gauges, curated wells, leveling panel) and ``AMP_V2/data/``
(the WiseEnvr cache). Parts of both cannot be re-fetched: the API does not serve the fan
polygon or the leveling panel, and ``ls_cache/`` itself came from a backup. This module
makes that backup reproducible.

A snapshot is two files written **outside the repository** (default
``~/twin_data_backups/``, or ``$HYDRO_TWIN_BACKUP_DIR``, or ``--out``):

- ``twin_data_<UTC stamp>.tar.gz``: the trees, stored under their repo-relative paths,
  plus ``MANIFEST.json`` as the archive's first member;
- ``twin_data_<UTC stamp>.manifest.json``: the same manifest beside the tarball, holding
  the tarball's own SHA256 and one SHA256 + size per file.

The manifest records repo-relative paths only (no home directory, user or host), so it is
safe to read anywhere. ``verify`` re-hashes the tarball and every member against it;
``--tree`` also compares the live working tree, which is how a silent cache change (the
147-well vs 174-well episode in ``docs/DATA_FORMAT.md``) is caught. ``restore`` refuses to
overwrite an existing file unless ``--force`` and verifies each file as it lands.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

SOURCES = ("chou-shui-data", "AMP_V2/data")
MANIFEST_NAME = "MANIFEST.json"
FORMAT_VERSION = 1
_CHUNK = 1 << 20


def default_backup_dir() -> Path:
    """``$HYDRO_TWIN_BACKUP_DIR`` if set, else ``~/twin_data_backups``."""
    env = os.environ.get("HYDRO_TWIN_BACKUP_DIR")
    return Path(env).expanduser() if env else Path.home() / "twin_data_backups"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_stream(fh) -> str:
    h = hashlib.sha256()
    for chunk in iter(lambda: fh.read(_CHUNK), b""):
        h.update(chunk)
    return h.hexdigest()


def _walk(root: Path, sources: tuple[str, ...]) -> list[str]:
    """Repo-relative POSIX paths of every regular file under ``sources``, sorted."""
    out: list[str] = []
    for src in sources:
        base = root / src
        if not base.is_dir():
            raise FileNotFoundError(f"source tree missing: {src}")
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames.sort()
            for name in filenames:
                p = Path(dirpath) / name
                if p.is_symlink() or not p.is_file():
                    continue
                out.append(p.relative_to(root).as_posix())
    return sorted(out)


def build_manifest(root: Path, sources: tuple[str, ...] = SOURCES) -> dict:
    """Hash every file under ``sources`` (paths relative to ``root``)."""
    files = {}
    for rel in _walk(root, sources):
        p = root / rel
        files[rel] = {"sha256": _sha256_file(p), "size": p.stat().st_size}
    return {
        "format": FORMAT_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": list(sources),
        "n_files": len(files),
        "total_bytes": sum(f["size"] for f in files.values()),
        "files": files,
    }


def snapshot(root: Path, out_dir: Path, sources: tuple[str, ...] = SOURCES,
             compresslevel: int = 6) -> tuple[Path, Path]:
    """Write ``<out_dir>/twin_data_<stamp>.tar.gz`` and its sidecar manifest."""
    root = root.resolve()
    out_dir = out_dir.expanduser().resolve()
    if out_dir == root or root in out_dir.parents:
        raise ValueError("backup directory must be outside the repository")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(root, sources)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    tar_path = out_dir / f"twin_data_{stamp}.tar.gz"
    side_path = out_dir / f"twin_data_{stamp}.manifest.json"
    tmp = tar_path.with_suffix(".gz.partial")
    body = json.dumps(manifest, indent=1, ensure_ascii=False).encode()
    with tarfile.open(tmp, "w:gz", compresslevel=compresslevel) as tar:
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size, info.mtime = len(body), int(time.time())
        tar.addfile(info, io.BytesIO(body))
        for rel in manifest["files"]:
            ti = tar.gettarinfo(str(root / rel), arcname=rel)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""          # no local account names in the archive
            with open(root / rel, "rb") as fh:
                tar.addfile(ti, fh)
    tmp.rename(tar_path)
    side = dict(manifest, tarball=tar_path.name, tarball_sha256=_sha256_file(tar_path),
                tarball_bytes=tar_path.stat().st_size)
    side_path.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    return tar_path, side_path


def _sidecar(tar_path: Path) -> Path:
    return tar_path.with_name(tar_path.name.replace(".tar.gz", ".manifest.json"))


def _load_manifest(tar_path: Path) -> dict:
    side = _sidecar(tar_path)
    if side.exists():
        return json.loads(side.read_text())
    with tarfile.open(tar_path, "r:gz") as tar:
        return json.loads(tar.extractfile(MANIFEST_NAME).read())


def verify(tar_path: Path, tree_root: Path | None = None) -> list[str]:
    """Problems found (empty list = verified).

    Checks the tarball hash against the sidecar, every member against the manifest, that
    no manifest file is missing from the archive and, with ``tree_root``, that the live
    tree matches the snapshot (missing, changed and new files are all reported).
    """
    problems: list[str] = []
    man = _load_manifest(tar_path)
    if "tarball_sha256" in man:
        got = _sha256_file(tar_path)
        if got != man["tarball_sha256"]:
            problems.append(f"tarball sha256 {got[:12]} != manifest {man['tarball_sha256'][:12]}")
    expected = man["files"]
    seen: set[str] = set()
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            if m.name == MANIFEST_NAME or not m.isfile():
                continue
            if m.name not in expected:
                problems.append(f"unexpected member: {m.name}")
                continue
            seen.add(m.name)
            digest = _sha256_stream(tar.extractfile(m))
            if digest != expected[m.name]["sha256"]:
                problems.append(f"member hash mismatch: {m.name}")
    problems += [f"missing from archive: {rel}" for rel in sorted(set(expected) - seen)]
    if tree_root is not None:
        root = tree_root.resolve()
        live = set(_walk(root, tuple(man["sources"])))
        for rel in sorted(set(expected) - live):
            problems.append(f"tree: missing {rel}")
        for rel in sorted(live - set(expected)):
            problems.append(f"tree: new file {rel}")
        for rel in sorted(live & set(expected)):
            p = root / rel
            if (p.stat().st_size != expected[rel]["size"]
                    or _sha256_file(p) != expected[rel]["sha256"]):
                problems.append(f"tree: changed {rel}")
    return problems


def _safe_target(dest: Path, name: str) -> Path:
    pp = PurePosixPath(name)
    if pp.is_absolute() or ".." in pp.parts:
        raise ValueError(f"unsafe member path: {name}")
    return dest / Path(*pp.parts)


def restore(tar_path: Path, dest: Path, force: bool = False,
            only: tuple[str, ...] = ()) -> int:
    """Extract (optionally only paths under ``only`` prefixes) and verify each file."""
    problems = verify(tar_path)
    if problems:
        raise RuntimeError("archive failed verification: " + "; ".join(problems[:5]))
    man = _load_manifest(tar_path)
    dest = dest.resolve()

    def wanted(name: str) -> bool:
        return not only or any(name == o or name.startswith(o.rstrip("/") + "/") for o in only)

    if not force:  # refuse before writing anything, not half-way through
        clash = [rel for rel in man["files"] if wanted(rel) and _safe_target(dest, rel).exists()]
        if clash:
            raise FileExistsError(f"{len(clash)} files already exist (e.g. {clash[0]}); "
                                  "pass --force to overwrite")
    n = 0
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar:
            if m.name == MANIFEST_NAME or not m.isfile() or not wanted(m.name):
                continue
            target = _safe_target(dest, m.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".restore-partial")
            with tar.extractfile(m) as src, open(tmp, "wb") as dst:
                for chunk in iter(lambda: src.read(_CHUNK), b""):
                    dst.write(chunk)
            if _sha256_file(tmp) != man["files"][m.name]["sha256"]:
                tmp.unlink()
                raise RuntimeError(f"hash mismatch after extraction: {m.name}")
            os.utime(tmp, (m.mtime, m.mtime))
            tmp.replace(target)
            n += 1
    return n


def _list(out_dir: Path) -> None:
    rows = sorted(out_dir.glob("twin_data_*.tar.gz")) if out_dir.is_dir() else []
    if not rows:
        print(f"no snapshots in {out_dir}")
        return
    for t in rows:
        try:
            man = _load_manifest(t)
            print(f"{t.name}  {man['n_files']} files  "
                  f"{man['total_bytes'] / 1e6:.0f} MB raw  "
                  f"{t.stat().st_size / 1e6:.0f} MB packed  created {man['created_utc']}")
        except Exception as exc:  # noqa: BLE001 - listing should survive one bad archive
            print(f"{t.name}  unreadable manifest: {exc}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot", help="write a dated tarball + SHA256 manifest")
    s.add_argument("--root", default=".", help="repository root (default: cwd)")
    s.add_argument("--out", default=None, help="backup directory (outside the repo)")
    s.add_argument("--sources", nargs="+", default=list(SOURCES))
    s.add_argument("--level", type=int, default=6, help="gzip level")
    s.add_argument("--no-verify", action="store_true", help="skip the post-write verify")
    v = sub.add_parser("verify", help="check a snapshot's integrity")
    v.add_argument("tarball")
    v.add_argument("--tree", nargs="?", const=".", default=None,
                   help="also compare the live tree at this root (default: cwd)")
    r = sub.add_parser("restore", help="extract a verified snapshot")
    r.add_argument("tarball")
    r.add_argument("--dest", default=".", help="repository root to restore into")
    r.add_argument("--force", action="store_true", help="overwrite existing files")
    r.add_argument("--only", nargs="*", default=[], help="restore only these path prefixes")
    ls = sub.add_parser("list", help="list snapshots in the backup directory")
    ls.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "snapshot":
        out = Path(a.out) if a.out else default_backup_dir()
        t0 = time.time()
        tar_path, side = snapshot(Path(a.root), out, tuple(a.sources), a.level)
        man = json.loads(side.read_text())
        print(f"wrote {tar_path}\n      {side}\n{man['n_files']} files, "
              f"{man['total_bytes'] / 1e6:.0f} MB -> {man['tarball_bytes'] / 1e6:.0f} MB, "
              f"sha256 {man['tarball_sha256']}  ({time.time() - t0:.0f} s)")
        if not a.no_verify:
            problems = verify(tar_path, tree_root=Path(a.root))
            print("verify: OK" if not problems else "verify: FAILED\n  " + "\n  ".join(problems))
            return 0 if not problems else 1
        return 0
    if a.cmd == "verify":
        problems = verify(Path(a.tarball), Path(a.tree) if a.tree else None)
        if problems:
            print(f"FAILED ({len(problems)} problems)\n  " + "\n  ".join(problems[:50]))
            return 1
        man = _load_manifest(Path(a.tarball))
        print(f"OK: {man['n_files']} files match"
              + (" the archive and the live tree" if a.tree else " the archive"))
        return 0
    if a.cmd == "restore":
        try:
            n = restore(Path(a.tarball), Path(a.dest), a.force, tuple(a.only))
        except (FileExistsError, RuntimeError, ValueError) as exc:
            print(f"restore refused: {exc}")
            return 1
        print(f"restored and verified {n} files into {Path(a.dest).resolve()}")
        return 0
    _list(Path(a.out) if a.out else default_backup_dir())
    return 0


if __name__ == "__main__":
    sys.exit(main())
