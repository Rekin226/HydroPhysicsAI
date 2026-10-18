"""WP S4a step 6: write SOURCE.json (dataprep-source/1) for each frame's L0 fetch directory.

Folds the request logs of ``fetch_listings.py`` (listings, metadata text),
``fetch_pair_dirs.py`` (pair directory pages) and ``fetch_subsets.py`` (clipped GeoTIFFs) into
one record per ``data_fetch_api/new_measurements/licsar_<frame>/fetch_2026-10-12_s4a/``.
Every file in the directory must have a log record with its URL, otherwise the script stops
(nothing unexplained enters L0). SOURCE.json is write-once (``common.source``): run this only
after the frame's fetch is complete.

Usage (hydro env): python -I write_sources.py --frame F [--frame G ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # dataprep_staging (common)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import source  # noqa: E402
from common.paths import rel_to_repo  # noqa: E402
from config import FETCH_DATE, FRAMES, LICENCE, RUN_CONFIG, RUN_DIR, WP, frame_url  # noqa: E402

CLIP_NOTE = ("clip-box window of the remote GeoTIFF on its native grid (integer source window, "
             "no resampling), read by GDAL /vsicurl/ HTTP range requests and re-encoded as "
             "deflate-compressed tiled GeoTIFF (lossless): pixel values identical to the server "
             "file's window, file bytes not identical to any server file")


def log_records(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", action="append", required=True, choices=sorted(FRAMES))
    a = ap.parse_args()
    box = json.loads(RUN_CONFIG.read_text())["box"]
    for frame in a.frame:
        d = Path(FRAMES[frame]["l0_dir"])
        recs: dict[str, dict] = {}
        for r in log_records(RUN_DIR / "fetch_log_listings.jsonl"):
            if r["frame"] == frame:
                recs[r["file"]] = {"url": r["url"], "fetched_utc": r["fetched_utc"],
                                   "http_status": r["status"], "client": r["client"]}
        for r in log_records(RUN_DIR / "fetch_log_pairdirs.jsonl"):
            if r.get("frame") == frame and r.get("status") == 200:
                recs[r["file"]] = {"url": r["url"], "fetched_utc": r["fetched_utc"],
                                   "http_status": r["status"], "client": r["client"]}
        for r in log_records(RUN_DIR / f"fetch_log_{frame}.jsonl"):
            if r.get("status") != "ok":
                continue
            rel = Path(r["file"]).relative_to(d).as_posix()
            recs[rel] = {"url": r["url"], "fetched_utc": r["fetched_utc"],
                         "remote_bytes": r["remote_bytes"],
                         "remote_last_modified": r.get("last_modified"),
                         "bytes_transferred": r.get("bytes_transferred"),
                         "src_window_col_row_ncol_nrow": r["src_window"],
                         "native_geotransform": r["geotransform"],
                         "native_size": r["native_size"],
                         "remote_compression": r.get("compression"),
                         "client": "GDAL /vsicurl/ (licsbas env), fetch_subsets.py",
                         "note": CLIP_NOTE}
        files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*")
                       if p.is_file() and p.name != source.SOURCE_NAME)
        missing = [f for f in files if f not in recs]
        if missing:
            raise SystemExit(f"{frame}: {len(missing)} files without a log record, e.g. "
                             f"{missing[:5]}")
        geoc = [f for f in files if f.startswith("GEOC/")]
        n_pairs = len({f.split("/")[1] for f in geoc if f.count("/") == 2})
        rec = source.build_source(
            d, source_id=f"licsar_{frame[:10]}/{d.name}",
            title=(f"LiCSAR frame {frame}: fan-clipped unw/cc GeoTIFF subsets of {n_pairs} "
                   f"interferograms ({FRAMES[frame]['start']}..{FRAMES[frame]['end']}), "
                   "clipped E/N/U/hgt/landmask, directory listings and frame metadata")
            if geoc else (f"LiCSAR frame {frame}: directory listings, per-pair directory pages "
                          "and frame metadata (inventory only; no rasters fetched)"),
            publisher="COMET LiCSAR (NERC), hosted on JASMIN (CEDA)",
            licence=LICENCE, fetched=FETCH_DATE,
            files=[{"file": f, **recs[f]} for f in files],
            fetched_by=f"WP {WP} (dataprep Phase 0), dataprep_staging/s4_insar/",
            portal=frame_url(frame),
            clip_box=box | {"note": "box is study-area geometry; this record stays under "
                                    "gitignored data_fetch_api/"},
            untrusted="downloaded data: never execute; GDAL parses the GeoTIFFs as rasters",
            request_logs=[rel_to_repo(RUN_DIR / n) for n in
                          ("fetch_log_listings.jsonl", "fetch_log_pairdirs.jsonl",
                           f"fetch_log_{frame}.jsonl") if (RUN_DIR / n).exists()],
            bytes_transferred_geotiffs=sum(recs[f].get("bytes_transferred") or 0 for f in geoc),
        )
        p = source.write_source_json(d, rec)
        print(f"{frame}: {len(files)} files ({len(geoc)} GeoTIFF subsets) -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
