"""WP S4a step 3: price the fan clip before fetching (user decision D1: report sizes).

Reads only GeoTIFF headers (HTTP range reads; no pixel data) of ``<pair>.geo.unw.tif`` and
``<pair>.geo.cc.tif`` for a stratified sample of the in-window pairs of each frame. Strata are
the portal's processing batches (directory modification month in the saved listing), because
the file layout (tiled/deflate, strip/deflate, strip/uncompressed) changes between batches.
For each sampled file it records the full size and two clip costs:

* ``naive``: bytes of every tile or strip the clip box touches (what a tile-level reader such
  as GDAL ``/vsicurl/`` without direct I/O transfers);
* ``smart``: the same for tiled or compressed files, but for uncompressed strips only the
  clip-box columns of each row (contiguous byte ranges, fetched by multi-range requests).

Only pairs whose saved directory page lists both GeoTIFFs are counted (``fetch_pair_dirs.py``;
some listed pair directories hold no GeoTIFF any more). Per-stratum means times the stratum's
pair count give the expected totals per frame.
Output: ``results/dataprep/insar_tools/s4a_run/size_estimate.json``.

Run with the hydro env: python -I estimate_sizes.py [per_stratum]
"""
from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import network as nw  # noqa: E402
import requests  # noqa: E402
from config import FRAMES, IFG_EXTS, RUN_CONFIG, RUN_DIR, frame_url  # noqa: E402
from tiffhead import RangeReader, box_window, read_header, window_tile_bytes  # noqa: E402

SEED = 2026
HEADER_COST = 16384 + 2048          # header range + IFD slack per file
RANGE_OVERHEAD = 90                 # multipart/byteranges part header per range (bytes)


def smart_bytes(h: dict, win: tuple[int, int, int, int]) -> int:
    c0, r0, nc, nr = win
    if nc == 0 or nr == 0:
        return 0
    if h["tiled"] or h["comp"] != 1:
        return window_tile_bytes(h, win)[0]
    bpp = h["bits"] // 8
    return nr * (nc * bpp + RANGE_OVERHEAD)


def probe(s: requests.Session, url: str, box: dict) -> dict:
    rr = RangeReader(s, url)
    h = read_header(rr)
    win = box_window(h, box)
    naive, nt = window_tile_bytes(h, win)
    return {"size": rr.size, "naive": naive, "smart": smart_bytes(h, win), "touched": nt,
            "n_blocks": len(h["cnts"]), "W": h["W"], "H": h["H"], "tw": h["tw"],
            "th": h["th"], "tiled": h["tiled"], "comp": h["comp"], "pred": h["pred"],
            "bits": h["bits"], "sf": h["sf"], "sx": h["sx"], "sy": h["sy"], "x0": h["x0"],
            "y0": h["y0"], "win": list(win), "header_bytes_read": rr.nbytes,
            "header_requests": rr.nreq}


def main() -> int:
    per_stratum = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    box = json.loads(RUN_CONFIG.read_text())["box"]
    s = requests.Session()
    out = {"box": box, "seed": SEED, "per_stratum": per_stratum, "frames": {}}
    hdr_total = 0
    for frame, spec in FRAMES.items():
        d = Path(spec["l0_dir"])
        html = d / "listings/interferograms.html"
        listed = nw.window_pairs(nw.parse_pairs(html), spec["start"], spec["end"])
        pairs = [p for p in listed if all(nw.pair_products(
            d / "listings" / "ifg_dirs" / f"{p}.html", p, IFG_EXTS).values())]
        mt = nw.listing_mtimes(html)
        strata = collections.defaultdict(list)
        for p in pairs:
            strata[mt.get(p, "?")[:7]].append(p)
        rng = random.Random(SEED)
        fr = {"n_pairs_listed": len(listed), "n_pairs_window": len(pairs), "strata": {}}
        tot = collections.Counter()
        for key in sorted(strata):
            members = strata[key]
            sample = sorted(rng.sample(members, min(per_stratum, len(members))))
            rows = []
            for p in sample:
                row = {"pair": p}
                for ext in IFG_EXTS:
                    u = f"{frame_url(frame)}/interferograms/{p}/{p}.geo.{ext}.tif"
                    row[ext] = probe(s, u, box)
                    hdr_total += row[ext]["header_bytes_read"]
                rows.append(row)
                print(frame, key, p, {e: (row[e]["size"], row[e]["naive"], row[e]["smart"],
                                          "T" if row[e]["tiled"] else "S", row[e]["comp"])
                                      for e in IFG_EXTS}, flush=True)
            mean = {k: sum(sum(r[e][k] for e in IFG_EXTS) for r in rows) / len(rows)
                    for k in ("size", "naive", "smart")}
            n = len(members)
            fr["strata"][key] = {"n_pairs": n, "sample": rows,
                                 "per_pair_mean": mean,
                                 "expected": {k: v * n for k, v in mean.items()}}
            for k, v in mean.items():
                tot[k] += v * n
            tot["header"] += 2 * n * HEADER_COST
        fr["expected_full_bytes"] = tot["size"]
        fr["expected_naive_clip_bytes"] = tot["naive"] + tot["header"]
        fr["expected_smart_clip_bytes"] = tot["smart"] + tot["header"]
        out["frames"][frame] = fr
        print(f"{frame}: {len(pairs)} pairs; full {tot['size']/1e9:.2f} GB; naive clip "
              f"{fr['expected_naive_clip_bytes']/1e9:.2f} GB; smart clip "
              f"{fr['expected_smart_clip_bytes']/1e9:.2f} GB", flush=True)
    out["survey_header_bytes"] = hdr_total
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "size_estimate.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"header bytes read by the survey: {hdr_total/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
