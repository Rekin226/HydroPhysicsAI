"""Pre-origin MANIFEST v2 for the forward's own rain input (FW.weather_replay.v2 prerequisite).

Same product as manifest_rain.py (``twin_rain/preorigin``, layer L1: a pointer to the vendor
rain cache chou-shui-data/data/rf_timeseries_vendor_2012_2022.csv the page theta was fitted
on, with a structure record and a validation record, no rain value), written as the NEW
version v2. The structure check and its gate are manifest_rain.py's own functions, imported
read-only; manifest_rain.py and product v1 are never changed.

Why v2: v1 (written 2026-11-01) recorded its inputs as ABSOLUTE paths. The manifest builder
(common.paths.rel_to_repo) resolves symlinks, and in a worktree whose chou-shui-data/ is a
symlink to another checkout the resolved path falls outside the repository, so v1 names the
cache by a home path. prereg_fw.py compared the repo-relative path, did not match, and
registered FW.weather_replay.v1 pumping-only (the file and its sha256 were right). v1 runs as
registered (rulings R7.fw_v1_pumping, R7.fw_both); v1's product is kept unchanged.

This builder records every input and code path repo-relative AS GIVEN (the file is hashed
through ``repo / path``, a symlinked directory is followed but never recorded), refuses an
absolute, home ('~'), '..' or drive-letter path with a message, and refuses to write any
version directory but v2 (an existing one is refused too).

Allowed uses (as v1): forcing_frozen and design_time_evidence; every other use is forbidden.
Known defects carried into every use: the vendor cache zero-fills gauge rf11 for all of
2021-22 and rf15 in 2015 H2 (zero-filled, not dry).

    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fw_replay/manifest_rain2.py \\
        --validate --scratch <dir outside the repository>
    ~/miniforge3/envs/hydro/bin/python -I dataprep_staging/fw_replay/manifest_rain2.py --write

``--validate`` builds the product in the scratch directory and prints the manifest and its
checks; ``--write`` (the lead only; this file must be committed first, the manifest records
its git state) writes results/dataprep/twin_rain/preorigin/v2/ once. Neither prints a rain
value.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "dataprep_staging"))
sys.path.insert(0, str(HERE))

import manifest_rain as MR1  # noqa: E402  (v1 builder, read-only: structure, gate, constants)
from common import manifest as M  # noqa: E402
from common.hashing import sha256_file  # noqa: E402

PRODUCT = MR1.PRODUCT
VERSION = "v2"
V1_VERSION = MR1.VERSION
RF, RF_STATIONS = MR1.RF, MR1.RF_STATIONS
FIRST, LAST = MR1.FIRST, MR1.LAST
ALLOWED, FORBIDDEN = MR1.ALLOWED, MR1.FORBIDDEN
SELF = "dataprep_staging/fw_replay/manifest_rain2.py"
CODE = (SELF, MR1.SELF)
CONSUMER = "FW.weather_replay.v2"
structure, gate = MR1.structure, MR1.gate
GAUGE_NOTE = ("Known vendor-cache defects, carried into every use: gauge rf11 is zero for all "
              "of 2021-22 and rf15 in 2015 H2 (zero-filled, not dry); the page model was "
              "calibrated with them, so a replay of those years reproduces the hindcast's own "
              "forcing, artefact included.")


def repo_relative(path) -> str:
    """``path`` as a repo-relative POSIX string; refuses an absolute, home, '..' or empty
    path (the v1 manifest named its inputs by absolute paths)."""
    s = "" if path is None else str(path)
    pp = PurePosixPath(s.replace("\\", "/"))
    if not s or pp.is_absolute() or s.startswith("~") or ".." in pp.parts or \
            re.match(r"^[A-Za-z]:", s):
        raise M.ManifestError(f"input path {s!r} is not repo-relative (absolute, home, '..' "
                              "and empty paths are refused: name the file as it sits in the "
                              "repository)")
    return pp.as_posix()


def path_problems(m: dict) -> list[str]:
    """Input and code paths of a manifest that are not repo-relative (empty: none)."""
    out = []
    for key in ("inputs", "code"):
        for r in m.get(key) or []:
            try:
                repo_relative(r.get("path") if isinstance(r, dict) else r)
            except M.ManifestError as err:
                out.append(f"{key}: {err}")
    return out


def records(rels, repo: Path = REPO) -> list[dict]:
    """{'path' (repo-relative, as given), 'sha256', 'bytes'} of each file, hashed through
    ``repo / path`` (a symlinked directory is followed but never recorded)."""
    out = []
    for rel in rels:
        rel = repo_relative(rel)
        p = Path(repo) / rel
        out.append({"path": rel, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return out


def build(product_dir: Path, repo: Path = REPO) -> dict:
    """Build the product in ``product_dir`` (created; must not exist; must be a v2 directory)
    -> the manifest. Inputs: RF and RF_STATIONS under ``repo``, recorded repo-relative."""
    product_dir = Path(product_dir)
    if product_dir.name != VERSION:
        raise M.ManifestError(f"{product_dir} is not a {VERSION} directory ({V1_VERSION} is "
                              "never rewritten)")
    ins = records((RF, RF_STATIONS), repo)
    rain, stations = (Path(repo) / r["path"] for r in ins)
    product_dir.mkdir(parents=True, exist_ok=False)
    rec = structure(rain, stations)
    g = gate(rec)
    (product_dir / "rain_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    (product_dir / "validation.json").write_text(json.dumps(g, indent=1) + "\n")
    m = M.build_manifest(
        product_dir, product=PRODUCT, layer="L1", version=VERSION,
        inputs=[str(rain), str(stations)], code=list(CODE),
        parameters={"source": "vendor rain cache the page theta was fitted on",
                    "window": [FIRST.isoformat(), LAST.isoformat()],
                    "consumer": f"{CONSUMER} (hydrophysics.twin.replay, the forward's own "
                                "rain input, energy+weather mode)",
                    "relation_to_v1": (
                        f"{PRODUCT} {V1_VERSION} names the same cache at the same sha256 by "
                        "an absolute path (a symlinked chou-shui-data resolved outside the "
                        f"repository); {V1_VERSION} is kept unchanged. {VERSION} records the "
                        "repo-relative paths")},
        fit_window_end="2022-12", validation_report="validation.json",
        gate={"passed": g["passed"], "criteria": g["criteria"]},
        allowed_uses=ALLOWED, forbidden_uses=FORBIDDEN,
        personal_data="none (public rain gauge ids and daily totals)",
        extra={"notes": [
            "Pointer product: the cache is read in place and not copied; its sha256 is in "
            "inputs, and any change refuses every consumer that checks it.",
            "Input and code paths are repository-relative (the builder refuses any other).",
            GAUGE_NOTE,
            "forcing_frozen here means the page model's frozen forcing as calibrated "
            "(2012-22); no post-origin month exists in the cache."]})
    if [r["sha256"] for r in m["inputs"]] != [r["sha256"] for r in ins]:
        raise M.ManifestError("input sha256 differs between the repo-relative records and the "
                              "manifest builder")
    m["inputs"] = ins                       # repo-relative as given, never resolved
    for c, r in zip(m["code"], records(CODE, REPO), strict=True):    # code: this checkout
        if c["sha256"] != r["sha256"]:
            raise M.ManifestError(f"code {r['path']} sha256 differs between records")
        c["path"] = r["path"]
    errs = M.validate_manifest(m, product_dir) + path_problems(m)
    if errs:
        raise M.ManifestError("; ".join(errs))
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate", action="store_true")
    g.add_argument("--write", action="store_true")
    ap.add_argument("--scratch", default=None, help="--validate: a directory outside the "
                    "repository to build the product in")
    a = ap.parse_args(argv)
    if a.validate:
        if not a.scratch:
            ap.error("--validate needs --scratch")
        d = Path(a.scratch).resolve() / PRODUCT / VERSION
        m = build(d)
        print(json.dumps(m, indent=1, ensure_ascii=False))
        print(f"built in {d}; gate passed: {m['gate']['passed']}; paths repo-relative: "
              f"{not path_problems(m)}; code git state: "
              f"{[c.get('git_state') for c in m['code']]}; not written to results/")
        return 0 if m["gate"]["passed"] else 3
    d = M.DATAPREP / PRODUCT / VERSION
    if d.exists():
        print(f"REFUSED: {d} exists")
        return 1
    m = build(d)
    if not m["gate"]["passed"]:
        print("REFUSED: the structure gate failed; the product directory is left for "
              "inspection without a manifest")
        return 3
    p = M.write_manifest(d, m)
    print(f"wrote {p} (allowed uses {ALLOWED}; inputs {[r['path'] for r in m['inputs']]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
