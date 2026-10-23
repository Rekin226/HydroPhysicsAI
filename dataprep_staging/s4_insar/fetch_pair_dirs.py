"""WP S4a step 2b: save every in-window interferogram directory page of each frame (L0).

The frame-level listing names a pair directory even when its GeoTIFFs were removed from the
portal (for example 069A_06694_131402/interferograms/20200205_20200211/ holds only
``*.tif.metadata``/``*.tif.xml``; a GET of its ``.geo.unw.tif`` returns HTTP 403). So the
availability of ``.geo.unw.tif`` and ``.geo.cc.tif`` is read from each pair's own Apache index,
saved byte for byte to ``<l0_dir>/listings/ifg_dirs/<pair>.html`` (write-once, read-only).
One JSON line per request goes to ``results/dataprep/insar_tools/s4a_run/
fetch_log_pairdirs.jsonl``. Responses are untrusted data: saved and parsed with regexes only.

Run with the hydro env: python -I fetch_pair_dirs.py [--threads 4]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import network as nw  # noqa: E402
import requests  # noqa: E402
from config import FRAMES, RUN_DIR, frame_url  # noqa: E402
from fetch_listings import get, save  # noqa: E402

LOG = RUN_DIR / "fetch_log_pairdirs.jsonl"
_lock = threading.Lock()
_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def one(frame: str, pair: str, dest: Path) -> dict:
    url = f"{frame_url(frame)}/interferograms/{pair}/"
    t = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    r = get(_session(), url)
    save(dest, r.content)
    return {"frame": frame, "file": dest.relative_to(dest.parents[2]).as_posix(), "url": url,
            "status": r.status_code, "bytes": len(r.content), "fetched_utc": t,
            "client": f"python-requests/{requests.__version__}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for frame, spec in FRAMES.items():
        d = Path(spec["l0_dir"])
        pairs = nw.window_pairs(nw.parse_pairs(d / "listings/interferograms.html"),
                                spec["start"], spec["end"])
        jobs += [(frame, p, d / "listings" / "ifg_dirs" / f"{p}.html") for p in pairs]
    todo = [j for j in jobs if not j[2].exists()]
    print(f"{len(jobs)} pair pages, {len(todo)} to fetch", flush=True)
    n_err = 0
    with open(LOG, "a", encoding="utf-8") as log, \
            cf.ThreadPoolExecutor(max_workers=a.threads) as ex:
        futs = {ex.submit(one, *j): j for j in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            try:
                rec = f.result()
            except Exception as e:  # noqa: BLE001 - logged; rerun fetches what is missing
                fr, p, d = futs[f]
                rec = {"frame": fr, "pair": p, "status": "error", "error": repr(e)[:300]}
                n_err += 1
            with _lock:
                log.write(json.dumps(rec) + "\n")
            if i % 100 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} errors {n_err}", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
