"""WP S4a helper (also for S4b): build a LiCSBAS work directory of symlinks into the L0 subsets.

LiCSBAS02_ml_prep.py reads one ``GEOC/`` directory holding ``yyyymmdd_yyyymmdd/`` pair
directories plus ``*.geo.{E,N,U,hgt}.tif``, ``baselines`` and ``metadata.txt``; it stacks the
rasters by array index and takes the geotransform from the first ``*.tif`` in GEOC. The L0
subsets of a frame were all clipped with one index window of the frame's reference grid
(``fetch_subsets.py``), so they stack by index whatever geotransform variant each file carries;
pairs clipped by their own geotransform (a different shape; none expected) are left out. This
script links (never copies, never writes into L0):

  <out>/GEOC/baselines, metadata.txt                        -> <l0>/metadata/
  <out>/GEOC/<frame>.geo.{E,N,U,hgt,landmask}.tif           -> <l0>/GEOC/
  <out>/GEOC/<pair>/<pair>.geo.{unw,cc}.tif                 -> <l0>/GEOC/<pair>/

for the fetched, on-reference-grid pairs of ``pairs_<frame>.csv`` (``validate.py``),
optionally limited to a date range or the first ``--max-pairs``.

Usage (any env): python -I make_licsbas_workdir.py --frame F --out DIR
                 [--start yyyymmdd] [--end yyyymmdd] [--max-pairs N] [--georef-variant V]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FRAMES, RUN_DIR  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=sorted(FRAMES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="00000000")
    ap.add_argument("--end", default="99999999")
    ap.add_argument("--max-pairs", type=int, default=0)
    ap.add_argument("--georef-variant", default="",
                    help="keep only pairs whose unw carries this georeference variant "
                         "(column georef_variant of pairs_<frame>.csv)")
    a = ap.parse_args()

    l0 = Path(FRAMES[a.frame]["l0_dir"])
    with open(RUN_DIR / f"pairs_{a.frame}.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    pairs = [r["pair"] for r in rows if r["fetched"] == "True"
             and r.get("on_reference_grid") == "True"
             and r["epoch1"] >= a.start and r["epoch2"] <= a.end
             and (not a.georef_variant or r.get("georef_variant") == a.georef_variant)]
    if a.max_pairs:
        pairs = pairs[:a.max_pairs]
    geoc = Path(a.out) / "GEOC"
    if geoc.exists():
        raise SystemExit(f"{geoc} exists; choose a new --out")
    geoc.mkdir(parents=True)
    for name in ("baselines", "metadata.txt"):
        os.symlink(l0 / "metadata" / name, geoc / name)
    linked_meta = []
    for m in ("E", "N", "U", "hgt", "landmask"):
        src = l0 / "GEOC" / f"{a.frame}.geo.{m}.tif"
        if src.exists():
            os.symlink(src, geoc / src.name)
            linked_meta.append(m)
    for p in pairs:
        (geoc / p).mkdir()
        for e in ("unw", "cc"):
            os.symlink(l0 / "GEOC" / p / f"{p}.geo.{e}.tif", geoc / p / f"{p}.geo.{e}.tif")
    print(json.dumps({"out": str(geoc), "n_pairs": len(pairs),
                      "metadata_rasters_linked": linked_meta,
                      "first": pairs[0] if pairs else None,
                      "last": pairs[-1] if pairs else None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
