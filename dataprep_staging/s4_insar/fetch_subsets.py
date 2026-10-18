"""WP S4a step 4: fetch fan-clipped LiCSAR GeoTIFF subsets by HTTP range reads (L0).

Runs in the separate ``licsbas`` conda env (GDAL); never in ``hydro``. For every available
in-window pair of a frame (both ``.geo.unw.tif`` and ``.geo.cc.tif`` listed in the pair's saved
directory page, see ``fetch_pair_dirs.py``), opens the remote file through GDAL's ``/vsicurl/``
driver and copies the clip-box window on the file's own native grid (integer source window,
no resampling; values are copied bit for bit and re-encoded with lossless deflate) to

  <l0_dir>/GEOC/<pair>/<pair>.geo.<ext>.tif      (LiCSBAS01 layout)
  <l0_dir>/GEOC/<frame>.geo.<E|N|U|hgt|landmask>.tif

Transfer. Compressed files (deflate tiles or strips): GDAL fetches only the blocks the window
touches, with ``GDAL_HTTP_MULTIRANGE=SERIAL`` (consecutive blocks merged into one request,
requests issued one after another; the portal answers HTTP 429 to bursts of parallel range
requests, observed 2026-10-12 with GDAL's default parallel multi-range and direct I/O).
Uncompressed strip files (several portal batches store ``unw`` uncompressed in full-width
strips, 33.8 MB per file for 069A_06694_131402): only the window's columns of each row are
requested, as contiguous byte ranges grouped into multi-range requests of at most
``MAX_RANGES`` ranges (``multipart/byteranges`` responses), about a fifth of the strip bytes;
the row offsets come from GDAL's ``TIFF`` metadata domain (``BLOCK_OFFSET_0_<strip>``).
Bytes transferred per file = GDAL's network statistics for the file plus the raw range bytes.
``--max-gb`` stops submitting new files once the cumulative transfer of this run reaches the
cap (user decision D1: the budget is reported, not exceeded silently).

Window rule. All products of a frame are written on one geocoding grid (same width and height
as the frame's ``geo.E.tif``), but some batches' GeoTIFFs carry a different geotransform: e.g.
the 2022-05 batch ``unw`` of 069A_06694_131402 has dx = 0.001 * W / (W - 1) = 0.00100034 deg
and an origin half a pixel west/north of the ``cc``, ``E`` and ``landmask`` files (linspace of
the pixel edges used as pixel centres; the misregistration runs from -0.5 px at the west edge
to +0.5 px at the east edge). Clipping each file by its own geotransform put that ``unw`` one
pixel off its ``cc`` (S4a run 1, removed). So every file whose shape equals the reference
shape is clipped with the reference index window (as LiCSBAS itself stacks full frames by
array index); a file of another shape would fall back to its own geotransform and is flagged.
Each output keeps its source's geotransform (shifted by the window); the log records whether it
matches the reference, and S4b assigns the reference geotransform by index (plan S4 step 7).
Every written file gets one JSON line in
``results/dataprep/insar_tools/s4a_run/fetch_log_<frame>.jsonl``. Existing files are skipped
(L0 is write-once; outputs are read-only). Downloaded files are untrusted data: GDAL only
parses them as rasters.

Usage (licsbas env): python -I fetch_subsets.py --frame F [--limit N] [--threads 4]
                     [--meta-only] [--pairs-file list.txt] [--max-gb G] [--dry-run]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import network as nw  # noqa: E402
from config import FRAMES, IFG_EXTS, RUN_CONFIG, RUN_DIR, frame_url  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
GDAL_CONFIG = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",   # never list the remote directory
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "6",
    "GDAL_HTTP_RETRY_DELAY": "5",
    # r3 (resume, 2026-10-12): the total-request timeout was 180 s, and a frame metadata raster
    # (E/N/U/hgt: deflate 1-row strips) reads its 735-row window as one merged ~7.5 MB range,
    # which at 25-40 KB/s per connection outlasted it (4 truncated reads, "TIFFFillStrip: Read
    # error ... got 1395 bytes, expected 10274"). Stalls are caught by the low-speed abort.
    "GDAL_HTTP_TIMEOUT": "1800",
    "GDAL_HTTP_LOW_SPEED_TIME": "120",
    "GDAL_HTTP_LOW_SPEED_LIMIT": "512",
    "CPL_VSIL_CURL_CACHE_SIZE": "16777216",
    "CPL_VSIL_CURL_NON_CACHED": "",
    "CPL_VSIL_NETWORK_STATS_ENABLED": "YES",
    "GTIFF_DIRECT_IO": "NO",
    "GDAL_HTTP_MULTIRANGE": "SERIAL",
    # HTTP/1.1: one TCP connection per concurrent request. With HTTP/2 GDAL multiplexes every
    # thread's requests over one connection, and the portal path delivers about 25-60 KB/s per
    # connection (RTT 245 ms), so 6 threads ran no faster than one (2026-10-12).
    "GDAL_HTTP_VERSION": "1.1",
    "GDAL_HTTP_MULTIPLEX": "NO",
    "GDAL_NUM_THREADS": "1",
}
META_RASTERS = ("E", "N", "U", "hgt", "landmask")
CREATION = ["COMPRESS=DEFLATE", "TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"]
MAX_RANGES = 100          # byte ranges per multi-range request (Apache MaxRanges default 200)
NUMPY_DTYPE = {"Float32": "f4", "Float64": "f8", "Byte": "u1", "UInt16": "u2", "Int16": "i2",
               "UInt32": "u4", "Int32": "i4"}

_lock = threading.Lock()
_local = threading.local()


def _session():
    import requests
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def multirange(url: str, ranges: list[tuple[int, int]]) -> tuple[dict[int, bytes], int]:
    """GET ``ranges`` (inclusive byte pairs) in one request; {start: bytes}, bytes received.

    Refuses a 200 (full-file) answer without reading its body, so a server that ignores the
    ranges can never trigger a whole-file download."""
    hdr = {"Range": "bytes=" + ",".join(f"{a}-{b}" for a, b in ranges)}
    last = None
    for k in range(6):
        try:
            r = _session().get(url, headers=hdr, timeout=300, stream=True)
            if r.status_code == 206:
                body = r.content
                ctype = r.headers.get("Content-Type", "")
                break
            r.close()
            last = f"HTTP {r.status_code}"
            if r.status_code == 200:
                raise RuntimeError(f"{url}: server ignored the ranges (HTTP 200); aborted")
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 - transient network error, retried
            last = repr(e)
        time.sleep(10 * (k + 1))
    else:
        raise RuntimeError(f"{url}: multi-range request failed: {last}")
    out: dict[int, bytes] = {}
    if "multipart/byteranges" not in ctype:
        a, b = ranges[0]
        if len(ranges) != 1 or len(body) != b - a + 1:
            raise RuntimeError(f"{url}: single-part answer to {len(ranges)} ranges")
        return {a: body}, len(body)
    boundary = ctype.split("boundary=", 1)[1].strip().strip('"').encode()
    pos = 0
    while True:
        i = body.find(b"--" + boundary, pos)
        if i < 0 or body[i + 2 + len(boundary):i + 4 + len(boundary)] == b"--":
            break
        h_end = body.find(b"\r\n\r\n", i)
        head = body[i:h_end].decode("latin-1")
        cr = [x for x in head.split("\r\n") if x.lower().startswith("content-range:")][0]
        a, b = (int(v) for v in cr.split()[2].split("/")[0].split("-"))
        out[a] = body[h_end + 4:h_end + 4 + (b - a + 1)]
        if len(out[a]) != b - a + 1:
            raise RuntimeError(f"{url}: truncated part {a}-{b}")
        pos = h_end + 4 + (b - a + 1)
    missing = [a for a, _ in ranges if a not in out]
    if missing:
        raise RuntimeError(f"{url}: {len(missing)} ranges missing from the multipart answer")
    return out, len(body)


def read_uncompressed_window(ds, url: str, win: tuple[int, int, int, int]):
    """Window of an uncompressed strip GeoTIFF: one contiguous byte range per row."""
    import numpy as np
    band = ds.GetRasterBand(1)
    c0, r0, nc, nr = win
    bh = band.GetBlockSize()[1]
    dname = ds.GetRasterBand(1).DataType
    from osgeo import gdal
    dt = np.dtype(NUMPY_DTYPE[gdal.GetDataTypeName(dname)])
    bpp, W = dt.itemsize, ds.RasterXSize
    head, nbytes = multirange(url, [(0, 3)])
    order = {b"II": "<", b"MM": ">"}[head[0][:2]]
    ranges = []
    for r in range(r0, r0 + nr):
        off = int(band.GetMetadataItem(f"BLOCK_OFFSET_0_{r // bh}", "TIFF"))
        a = off + ((r % bh) * W + c0) * bpp
        ranges.append((a, a + nc * bpp - 1))
    got: dict[int, bytes] = {}
    for k in range(0, len(ranges), MAX_RANGES):
        part, n = multirange(url, ranges[k:k + MAX_RANGES])
        got.update(part)
        nbytes += n
    arr = np.frombuffer(b"".join(got[a] for a, _ in ranges), dtype=dt.newbyteorder(order))
    return arr.reshape(nr, nc).astype(dt.newbyteorder("="), copy=False), nbytes


def window(gt: tuple, W: int, H: int, box: dict) -> tuple[int, int, int, int]:
    """(col0, row0, ncols, nrows) of the box on the native grid, clipped to the image."""
    x0, sx, _, y0, _, sy = gt
    sy = -sy
    c0 = max(0, int((box["lon0"] - x0) // sx))
    c1 = min(W, int(-(-(box["lon1"] - x0) // sx)))
    r0 = max(0, int((y0 - box["lat1"]) // sy))
    r1 = min(H, int(-(-(y0 - box["lat0"]) // sy)))
    return c0, r0, max(0, c1 - c0), max(0, r1 - r0)


def frame_reference(gdal, frame: str, box: dict) -> dict:
    """The frame's reference grid (its ``geo.E.tif``: shape and geotransform) and the clip
    box's index window on it. Every product of a frame with the same shape is clipped with
    this one index window (see the module notes on stretched geotransforms)."""
    ds = gdal.Open(f"/vsicurl/{frame_url(frame)}/metadata/{frame}.geo.E.tif")
    gt = ds.GetGeoTransform()
    W, H = ds.RasterXSize, ds.RasterYSize
    return {"url": f"{frame_url(frame)}/metadata/{frame}.geo.E.tif", "shape": [W, H],
            "geotransform": list(gt), "window": list(window(gt, W, H, box))}


def clip_one(gdal, url: str, dest: Path, box: dict, ref: dict) -> dict:
    t0 = time.time()
    vsi = "/vsicurl/" + url
    st = gdal.VSIStatL(vsi)
    if st is None:
        return {"url": url, "file": str(dest), "status": "missing"}
    ds = gdal.Open(vsi)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    if [ds.RasterXSize, ds.RasterYSize] == ref["shape"]:
        c0, r0, nc, nr = ref["window"]
        rule = "frame_index_window"
    else:
        c0, r0, nc, nr = window(gt, ds.RasterXSize, ds.RasterYSize, box)
        rule = "own_geotransform_window (shape differs from the frame reference)"
    if nc == 0 or nr == 0:
        return {"url": url, "file": str(dest), "status": "outside"}
    md = gdal.GetFileMetadata(vsi, "HEADERS") or {}
    rec = {"url": url, "file": str(dest), "remote_bytes": int(st.size),
           "remote_mtime": int(st.mtime),
           "last_modified": md.get("Last-Modified") or md.get("last-modified"),
           "native_size": [ds.RasterXSize, ds.RasterYSize],
           "native_block": band.GetBlockSize(),
           "compression": ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE"),
           "dtype": gdal.GetDataTypeName(band.DataType),
           "nodata": band.GetNoDataValue(), "geotransform": list(gt),
           "area_or_point": ds.GetMetadataItem("AREA_OR_POINT"),
           "src_window": [c0, r0, nc, nr], "window_rule": rule,
           "geotransform_matches_reference": bool(
               all(abs(a - b) < 1e-9 for a, b in zip(gt, ref["geotransform"], strict=True)))}
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = (rec["compression"] in (None, "", "NONE")
           and band.GetBlockSize()[0] == ds.RasterXSize)
    if raw:
        arr, rec["raw_range_bytes"] = read_uncompressed_window(ds, url, (c0, r0, nc, nr))
    else:
        # one RasterIO call for the whole window, so GDAL fetches exactly the touched blocks
        # in one multi-range pass (block-by-block reads trigger /vsicurl/ sequential
        # read-ahead, which ran to the end of a strip file in the 2026-10-12 probe)
        arr = band.ReadAsArray(c0, r0, nc, nr)
    mem = gdal.GetDriverByName("MEM").Create("", nc, nr, 1, band.DataType)
    mem.SetGeoTransform([gt[0] + c0 * gt[1], gt[1], 0.0, gt[3] + r0 * gt[5], 0.0, gt[5]])
    mem.SetProjection(ds.GetProjection())
    if ds.GetMetadataItem("AREA_OR_POINT"):
        mem.SetMetadataItem("AREA_OR_POINT", ds.GetMetadataItem("AREA_OR_POINT"))
    mb = mem.GetRasterBand(1)
    if band.GetNoDataValue() is not None:
        mb.SetNoDataValue(band.GetNoDataValue())
    mb.WriteArray(arr)
    out = gdal.GetDriverByName("GTiff").CreateCopy(str(tmp), mem, options=CREATION)
    mem = None
    out.FlushCache()
    out = None
    ds = None
    os.replace(tmp, dest)
    dest.chmod(0o444)
    rec.update({"status": "ok", "reader": "raw_multirange" if raw else "gdal_vsicurl",
                "out_bytes": dest.stat().st_size,
                "seconds": round(time.time() - t0, 2),
                "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
    return rec


def netstats_bytes(gdal) -> tuple[int, dict[str, int]]:
    """Total downloaded bytes and per-file downloaded bytes from GDAL's network statistics."""
    js = json.loads(gdal.NetworkStatsGetAsSerializedJSON() or "{}")
    files = ((js.get("handlers") or {}).get("vsicurl") or {}).get("files") or {}
    per = {k: int(((v.get("methods") or {}).get("GET") or {}).get("downloaded_bytes", 0))
           for k, v in files.items()}
    total = int(((js.get("methods") or {}).get("GET") or {}).get("downloaded_bytes", 0))
    return total, per


def jobs_for(frame: str, a) -> tuple[list[tuple[str, Path]], list[str]]:
    spec = FRAMES[frame]
    l0 = Path(spec["l0_dir"])
    base = frame_url(frame)
    geoc = l0 / "GEOC"
    jobs = [(f"{base}/metadata/{frame}.geo.{m}.tif", geoc / f"{frame}.geo.{m}.tif")
            for m in META_RASTERS]
    unavailable: list[str] = []
    if a.meta_only:
        return jobs, unavailable
    if a.pairs_file:
        pairs = [p.strip() for p in Path(a.pairs_file).read_text().splitlines() if p.strip()]
    else:
        pairs = nw.window_pairs(nw.parse_pairs(l0 / "listings/interferograms.html"),
                                spec["start"], spec["end"])
    keep = []
    for p in pairs:
        page = l0 / "listings" / "ifg_dirs" / f"{p}.html"
        if not page.exists():
            raise SystemExit(f"{page} missing: run fetch_pair_dirs.py first")
        have = nw.pair_products(page, p, IFG_EXTS)
        (keep if all(have.values()) else unavailable).append(p)
    if a.limit:
        keep = keep[:a.limit]
    jobs += [(f"{base}/interferograms/{p}/{p}.geo.{e}.tif", geoc / p / f"{p}.geo.{e}.tif")
             for p in keep for e in IFG_EXTS]
    return jobs, unavailable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, choices=sorted(FRAMES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--meta-only", action="store_true")
    ap.add_argument("--pairs-file")
    ap.add_argument("--max-gb", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    jobs, unavailable = jobs_for(a.frame, a)
    todo = [(u, d) for u, d in jobs if not d.exists()]
    print(f"{a.frame}: {len(jobs)} files ({len(unavailable)} listed pairs lack unw or cc), "
          f"{len(todo)} to fetch, threads {a.threads}, cap {a.max_gb or 'none'} GB", flush=True)
    if a.dry_run:
        return 0

    from osgeo import gdal
    gdal.UseExceptions()
    for k, v in GDAL_CONFIG.items():
        gdal.SetConfigOption(k, v or None)

    box = json.loads(RUN_CONFIG.read_text())["box"]
    ref = frame_reference(gdal, a.frame, box)
    print(f"reference grid {ref['shape']} gt {ref['geotransform']} window {ref['window']}",
          flush=True)
    (RUN_DIR / f"reference_grid_{a.frame}.json").write_text(json.dumps(ref, indent=1) + "\n")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / f"fetch_log_{a.frame}.jsonl"
    gdal.NetworkStatsReset()
    cap = a.max_gb * 1e9 if a.max_gb else float("inf")
    t0, n_ok, n_bad, stopped = time.time(), 0, 0, False
    raw_total = [0]
    it = iter(todo)
    with open(log_path, "a", encoding="utf-8") as log, \
            cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs: dict = {}

        def submit_next() -> bool:
            nonlocal stopped
            if stopped:
                return False
            if netstats_bytes(gdal)[0] + raw_total[0] >= cap:
                stopped = True
                return False
            nxt = next(it, None)
            if nxt is None:
                return False
            futs[ex.submit(clip_one, gdal, nxt[0], nxt[1], box, ref)] = nxt
            return True

        for _ in range(a.threads * 2):
            submit_next()
        i = 0
        while futs:
            done, _ = cf.wait(list(futs), return_when=cf.FIRST_COMPLETED)
            for f in done:
                u, d = futs.pop(f)
                i += 1
                try:
                    rec = f.result()
                except Exception as e:  # noqa: BLE001 - log and continue; retried on rerun
                    rec = {"url": u, "file": str(d), "status": "error", "error": repr(e)[:500]}
                    tmp = d.with_name(d.name + ".part")
                    if tmp.exists():
                        tmp.unlink()
                rec["frame"] = a.frame
                rec["gdal_bytes"] = netstats_bytes(gdal)[1].get("/vsicurl/" + u, 0)
                rec["bytes_transferred"] = rec["gdal_bytes"] + rec.get("raw_range_bytes", 0)
                raw_total[0] += rec.get("raw_range_bytes", 0)
                with _lock:
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                n_ok += rec["status"] == "ok"
                n_bad += rec["status"] != "ok"
                if i % 50 == 0:
                    tot = netstats_bytes(gdal)[0] + raw_total[0]
                    print(f"{i}/{len(todo)} ok {n_ok} bad {n_bad} {time.time()-t0:.0f}s "
                          f"{tot/1e6:.1f} MB", flush=True)
                submit_next()
    total = netstats_bytes(gdal)[0] + raw_total[0]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RUN_DIR / f"netstats_{a.frame}_{stamp}.json").write_text(
        gdal.NetworkStatsGetAsSerializedJSON() or "{}")
    print(f"done ok {n_ok} bad {n_bad} in {time.time()-t0:.0f}s; transferred "
          f"{total/1e6:.1f} MB; {'STOPPED AT CAP' if stopped else 'complete'}; netstats -> "
          f"netstats_{a.frame}_{stamp}.json", flush=True)
    return 0 if (n_bad == 0 and not stopped) else 1


if __name__ == "__main__":
    raise SystemExit(main())
