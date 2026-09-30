"""Content-addressed local research releases; failed checks never replace current.json.

    python -m hydrophysics.twin.release inspect results/twin/twin_app.html
    python -m hydrophysics.twin.release publish results/twin/twin_app.html --root results/releases

Publication here means a local artifact pointer, not uploading or deploying a site.
Research status is deliberate: numerical checks do not certify policy predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = 1


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(sources: dict, vintage: str, public: bool, offline: bool) -> dict:
    """Use logical input names, never local absolute paths or credentials."""
    inputs = {}
    for role, name in sorted(sources.items()):
        if name and name not in ("auto", "none") and Path(name).is_file():
            path = Path(name)
            inputs[role] = {"name": path.name, "sha256": sha256(path),
                            "bytes": path.stat().st_size}
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".html", ".js", ".csv", ".txt"):
            digest.update(path.relative_to(package).as_posix().encode() + b"\0")
            digest.update(bytes.fromhex(sha256(path)))
    versions = {}
    for name in ("numpy", "pandas", "torch", "nvidia-physicsnemo", "flopy"):
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return {"schema": SCHEMA, "status": "research-scenario", "data_vintage": vintage,
            "public": public, "offline": offline, "sources": inputs,
            "code_sha256": digest.hexdigest(), "software": versions,
            "uncertainty": "conditional model ensemble; not calibrated predictive coverage",
            "geometry": "schematic aquifers; screen-depth medians and assumed thicknesses"}


def read_payload(path: str | Path) -> dict:
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                      Path(path).read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError("No twin payload in HTML")
    return json.loads(match.group(1))


def check_payload(payload: dict) -> list[str]:
    """Hard release checks, separate from observational scientific validation."""
    import numpy as np

    from .app.prep import unpack

    errors = []
    meta, release = payload["meta"], payload.get("release", {})
    if release.get("schema") != SCHEMA or not release.get("sources", {}).get("forward"):
        errors.append("Missing versioned forward-input provenance")
    if release.get("status") != "research-scenario":
        errors.append("This publisher supports research scenarios only")
    if not release.get("public") or not payload["modelcard"].get("public"):
        errors.append("Individual observations cannot enter a public release")
    if payload.get("wells") is not None or payload.get("leveling") is not None:
        errors.append("Public payload contains individual observations")
    if not release.get("offline"):
        errors.append("Release must include its renderer or explicitly omit 3D")
    if meta.get("yTested", meta["yObs"]) > meta["yObs"]:
        errors.append("Unobserved calendar years are labelled tested")
    if release.get("data_vintage") != meta.get("vintage"):
        errors.append("Data vintage disagrees with the release manifest")
    shapes = {"subsBase": (meta["nA"], len(meta["years"])),
              "headBase": (meta["L"], meta["nA"], len(meta["years"]))}
    for key, shape in shapes.items():
        values = unpack(payload["arrays"][key])
        if values.shape != shape or not np.isfinite(values).all():
            errors.append(f"Invalid shape or non-finite values in {key}")
    return errors


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def publish(page: str | Path, root: str | Path) -> dict:
    """Validate an immutable copy before atomically switching the local pointer."""
    page, root = Path(page), Path(root)
    raw = page.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    # Parse exactly the bytes being published (the producer could be rebuilding page).
    root.mkdir(parents=True, exist_ok=True)
    fd, scratch = tempfile.mkstemp(suffix=".html", prefix=".candidate-", dir=root)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
        payload = read_payload(scratch)
        errors = check_payload(payload)
        if errors:
            raise ValueError("; ".join(errors))
        target = root / f"{digest}.html"
        os.replace(scratch, target)
        manifest = {**payload["release"], "page": target.name, "page_sha256": digest,
                    "published_utc": datetime.now(timezone.utc).isoformat()}
        atomic_json(root / f"{digest}.json", manifest)
        atomic_json(root / "current.json", manifest)
        return manifest
    finally:
        if os.path.exists(scratch):
            os.unlink(scratch)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("inspect", "publish"))
    ap.add_argument("page")
    ap.add_argument("--root", default="results/releases")
    args = ap.parse_args(argv)
    if args.action == "publish":
        print(json.dumps(publish(args.page, args.root), indent=2))
    else:
        payload = read_payload(args.page)
        errors = check_payload(payload)
        print(json.dumps({"release": payload.get("release"), "errors": errors}, indent=2))
        if errors:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
