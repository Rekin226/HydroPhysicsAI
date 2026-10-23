"""WP S4a acceptance S4a-6: the raw multi-range reader of ``fetch_subsets.py`` is exact.

For every portal batch whose files were read by the raw multi-range reader (uncompressed
full-width strips), this takes one fetched L0 file (the batch's first pair in date order) and
compares, bit for bit, its first and last 12 window rows with GDAL's own read of the same rows
from the remote file (``/vsicurl/``, the GTiff driver, no direct I/O). A batch is the portal's
processing month of the pair directory (modification month in the saved frame listing, the
strata of ``estimate_sizes.py``). Comparison is on the raw bytes (NaN payloads included).

Runs in the ``licsbas`` env. Writes ``results/dataprep/insar_tools/s4a_run/spot_checks.json``
(with the bytes transferred, which ``tally.py`` adds to the WP total). Read-only on L0.

Usage (licsbas env): python -I check_raw_reader.py [--frame F ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import network as nw  # noqa: E402
import numpy as np  # noqa: E402
from config import FRAMES, RUN_DIR  # noqa: E402
from fetch_subsets import GDAL_CONFIG  # noqa: E402

N_ROWS = 12


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", action="append", choices=sorted(FRAMES))
    a = ap.parse_args()
    frames = a.frame or [f for f in FRAMES if (RUN_DIR / f"fetch_log_{f}.jsonl").exists()]
    from osgeo import gdal
    gdal.UseExceptions()
    for k, v in GDAL_CONFIG.items():
        gdal.SetConfigOption(k, v or None)
    gdal.NetworkStatsReset()
    checks = []
    for frame in frames:
        l0 = Path(FRAMES[frame]["l0_dir"])
        mt = nw.listing_mtimes(l0 / "listings" / "interferograms.html")
        recs = [json.loads(x) for x in
                (RUN_DIR / f"fetch_log_{frame}.jsonl").read_text().splitlines()]
        raw = [r for r in recs if r.get("status") == "ok" and r.get("reader") == "raw_multirange"]
        by_batch: dict[str, list[dict]] = {}
        for r in raw:
            pair = Path(r["file"]).parent.name
            by_batch.setdefault(mt.get(pair, "?")[:7], []).append(r)
        for batch in sorted(by_batch):
            r = sorted(by_batch[batch], key=lambda x: Path(x["file"]).name)[0]
            c0, r0, nc, nr = r["src_window"]
            ds_loc = gdal.Open(r["file"])          # datasets kept alive while bands are read
            ds_rem = gdal.Open("/vsicurl/" + r["url"])
            local, remote = ds_loc.GetRasterBand(1), ds_rem.GetRasterBand(1)
            res = {"frame": frame, "batch": batch, "n_raw_files_in_batch": len(by_batch[batch]),
                   "file": str(Path(r["file"]).relative_to(l0)), "url": r["url"],
                   "src_window": r["src_window"], "blocks": []}
            for name, lr in (("first", 0), ("last", nr - N_ROWS)):
                a_loc = local.ReadAsArray(0, lr, nc, N_ROWS)
                a_rem = remote.ReadAsArray(c0, r0 + lr, nc, N_ROWS)
                same = (a_loc.dtype == a_rem.dtype and a_loc.shape == a_rem.shape
                        and a_loc.tobytes() == a_rem.tobytes())
                res["blocks"].append({"block": name, "window_rows": [lr, lr + N_ROWS],
                                      "dtype": str(a_loc.dtype), "shape": list(a_loc.shape),
                                      "bitwise_equal": bool(same),
                                      "n_finite": int(np.isfinite(a_loc).sum())})
            res["bitwise_equal"] = all(b["bitwise_equal"] for b in res["blocks"])
            checks.append(res)
            print(frame, batch, res["file"], res["bitwise_equal"], flush=True)
    js = json.loads(gdal.NetworkStatsGetAsSerializedJSON() or "{}")
    nbytes = int(((js.get("methods") or {}).get("GET") or {}).get("downloaded_bytes", 0))
    out = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "criterion": "S4a-6", "n_rows": N_ROWS, "checks": checks,
           "n_batches": len(checks), "all_equal": bool(checks) and all(
               c["bitwise_equal"] for c in checks), "bytes": nbytes,
           "bytes_note": "GDAL network statistics of the remote reads"}
    (RUN_DIR / "spot_checks.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(checks)} batches, all equal {out['all_equal']}, {nbytes/1e6:.2f} MB")
    return 0 if out["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
