"""WP S4b repair round 2 (S4br): LiCSBAS work directory of symlinks into the new L0 fetch of
the second ascending frame 069A_06632_162010 (``s4br_fetch_06632.py``). Same layout as S4a's
``make_licsbas_workdir.py`` (which only knows S4a's fetch directories):

  <out>/GEOC/baselines, metadata.txt                        -> <l0>/metadata/
  <out>/GEOC/<frame>.geo.{E,N,U,hgt,landmask}.tif           -> <l0>/GEOC/
  <out>/GEOC/<pair>/<pair>.geo.{unw,cc}.tif                 -> <l0>/GEOC/<pair>/

for the fetched pairs clipped with the frame's reference index window
(``s4br_run/pairs_<frame>.csv`` from ``s4br_fetch_06632.py validate``) inside --start/--end.
Never copies, never writes into L0.

Usage (any env): python -I s4br_workdir.py --frame 069A_06632_162010 --out DIR
                 [--start yyyymmdd] [--end yyyymmdd]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import s4br_config as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=[R.SECOND])
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="00000000")
    ap.add_argument("--end", default="99999999")
    a = ap.parse_args()
    l0 = R.L0_SECOND
    with open(R.RUN_DIR / f"pairs_{a.frame}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    pairs = [r["pair"] for r in rows if r["fetched"] == "True"
             and r["on_reference_grid"] == "True"
             and r["epoch1"] >= a.start and r["epoch2"] <= a.end]
    geoc = Path(a.out) / "GEOC"
    if geoc.exists():
        raise SystemExit(f"{geoc} exists; choose a new --out")
    geoc.mkdir(parents=True)
    for name in ("baselines", "metadata.txt"):
        os.symlink(l0 / "metadata" / name, geoc / name)
    linked = []
    for m in ("E", "N", "U", "hgt", "landmask"):
        src = l0 / "GEOC" / f"{a.frame}.geo.{m}.tif"
        if src.exists():
            os.symlink(src, geoc / src.name)
            linked.append(m)
    for p in pairs:
        (geoc / p).mkdir()
        for e in ("unw", "cc"):
            os.symlink(l0 / "GEOC" / p / f"{p}.geo.{e}.tif", geoc / p / f"{p}.geo.{e}.tif")
    print(json.dumps({"out": str(geoc), "n_pairs": len(pairs), "metadata_rasters_linked": linked,
                      "first": pairs[0] if pairs else None,
                      "last": pairs[-1] if pairs else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
