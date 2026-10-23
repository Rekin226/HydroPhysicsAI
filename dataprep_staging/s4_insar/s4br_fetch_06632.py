"""WP S4b repair round 2 (S4br): fetch the approved second ascending frame 069A_06632_162010
(fan-clipped LiCSAR unw/cc subsets, about 1.1 GB) into a NEW L0 fetch directory, re-using
S4a's fetch code unchanged (those files are hashed by insar_licsar_subsets/v1).

  data_fetch_api/new_measurements/licsar_069A_06632/fetch_2026-10-18_s4br/
    listings/{interferograms,epochs,metadata}.html, listings/ifg_dirs/<pair>.html
    metadata/{baselines,metadata.txt,<frame>-poly.txt,...}
    GEOC/<pair>/<pair>.geo.{unw,cc}.tif, GEOC/<frame>.geo.{E,N,U,hgt,landmask}.tif
    SOURCE.json (write-once, last step)

S4a's modules read the frame table and the run directory from ``config``; this wrapper
narrows ``config.FRAMES`` (in place) to this frame with the new fetch directory and points
the request logs to ``results/dataprep/insar_tools/s4br_run/``. The clip box is S4a's
(``s4a_run/run_config.json``, read only). Downloads are untrusted data: listings are parsed
with regexes, GeoTIFFs only by GDAL as rasters; nothing downloaded is executed.

Usage:
  python -I s4br_fetch_06632.py listings     (hydro env: listings, metadata text, pair pages)
  python -I s4br_fetch_06632.py subsets      (licsbas env: GeoTIFF subsets, 8 threads, cap)
  python -I s4br_fetch_06632.py validate     (licsbas env: pairs table for the work dir)
  python -I s4br_fetch_06632.py source       (hydro env: SOURCE.json, write-once)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import config as S4A  # noqa: E402  (S4a's config, read only; narrowed in place below)

FRAME = "069A_06632_162010"
FETCH_DATE = "2026-10-18"
TAG = f"fetch_{FETCH_DATE}_s4br"
L0 = S4A.L0_NEW / "licsar_069A_06632" / TAG
RUN = S4A.DATAPREP / "insar_tools" / "s4br_run"
CAP_GB = 1.25          # approved: about 1.1 GB (S4a's estimate 1.093 GB); stop beyond 1.25 GB
SPEC = {"role": "second ascending frame (S4br: overlap averaging and coverage from 2022-11)",
        "start": "20221101", "end": "20260131", "l0_dir": L0}


def narrow() -> None:
    S4A.FRAMES.clear()
    S4A.FRAMES[FRAME] = SPEC
    RUN.mkdir(parents=True, exist_ok=True)


def cmd_listings() -> int:
    narrow()
    import fetch_listings as FL
    import fetch_pair_dirs as FP
    FL.LOG = RUN / "fetch_log_listings_06632.jsonl"
    FP.LOG = RUN / "fetch_log_pairdirs_06632.jsonl"
    FL.RUN_DIR = RUN
    FP.RUN_DIR = RUN
    rc = FL.main()
    if rc:
        return rc
    sys.argv = [sys.argv[0], "--threads", "4"]
    return FP.main()


def cmd_subsets() -> int:
    narrow()
    import fetch_subsets as FS
    FS.RUN_DIR = RUN
    sys.argv = [sys.argv[0], "--frame", FRAME, "--threads", "8", "--max-gb", str(CAP_GB)]
    return FS.main()


def cmd_validate() -> int:
    """Pairs table of the fetched subsets: one row per in-window pair with both files read
    back by GDAL, the shape and whether the files were clipped with the frame's reference
    index window (fetch log), the valid fraction of unw and cc in the window."""
    narrow()
    import numpy as np
    from osgeo import gdal
    gdal.UseExceptions()
    log = {}
    with open(RUN / f"fetch_log_{FRAME}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                log[Path(r["file"]).name] = r
    rows = []
    geoc = L0 / "GEOC"
    for d in sorted(p for p in geoc.iterdir() if p.is_dir()):
        pair = d.name
        rec = {"pair": pair, "epoch1": pair[:8], "epoch2": pair[9:], "fetched": False}
        fu, fc = d / f"{pair}.geo.unw.tif", d / f"{pair}.geo.cc.tif"
        if fu.exists() and fc.exists():
            u = gdal.Open(str(fu)).ReadAsArray().astype("f4")
            c = gdal.Open(str(fc)).ReadAsArray()
            lu, lc = log.get(fu.name, {}), log.get(fc.name, {})
            rec.update({"fetched": True, "shape": f"{u.shape[1]}x{u.shape[0]}",
                        "on_reference_grid": bool(lu.get("window_rule") == "frame_index_window"
                                                  and lc.get("window_rule")
                                                  == "frame_index_window"),
                        "unw_valid_frac_window": float(np.mean(np.isfinite(u) & (u != 0))),
                        "cc_valid_frac_window": float(np.mean(c > 0))})
        rows.append(rec)
    out = RUN / f"pairs_{FRAME}.csv"
    keys = ["pair", "epoch1", "epoch2", "fetched", "shape", "on_reference_grid",
            "unw_valid_frac_window", "cc_valid_frac_window"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    ok = sum(1 for r in rows if r.get("on_reference_grid"))
    print(json.dumps({"pairs": len(rows), "on_reference_grid": ok, "table": str(out)}))
    return 0


def cmd_source() -> int:
    narrow()
    import write_sources as WS
    from common.paths import rel_to_repo
    WS.RUN_DIR = RUN
    WS.FETCH_DATE = FETCH_DATE
    # S4a's writer reads its request logs under fixed names; give it this round's names
    for src, dst in (("fetch_log_listings_06632.jsonl", "fetch_log_listings.jsonl"),
                     ("fetch_log_pairdirs_06632.jsonl", "fetch_log_pairdirs.jsonl")):
        s, d = RUN / src, RUN / dst
        if not d.exists():
            d.write_bytes(s.read_bytes())
    WS.WP = "S4b (repair round 2, S4br)"
    sys.argv = [sys.argv[0], "--frame", FRAME]
    rc = WS.main()
    print(json.dumps({"source_json": rel_to_repo(L0 / "SOURCE.json")}))
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["listings", "subsets", "validate", "source"])
    a = ap.parse_args()
    return {"listings": cmd_listings, "subsets": cmd_subsets, "validate": cmd_validate,
            "source": cmd_source}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
