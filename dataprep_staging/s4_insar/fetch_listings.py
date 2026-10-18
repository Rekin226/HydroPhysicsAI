"""WP S4a step 2: fetch the LiCSAR directory listings and frame metadata text files (L0).

For each frame in ``config.FRAMES`` this saves, into the frame's L0 fetch directory
(``data_fetch_api/new_measurements/licsar_<frame>/fetch_2026-10-12_s4a/``):

  listings/interferograms.html   Apache index of <frame>/interferograms/
  listings/epochs.html           Apache index of <frame>/epochs/
  listings/metadata.html         Apache index of <frame>/metadata/
  metadata/baselines             perpendicular baselines (LiCSBAS input)
  metadata/metadata.txt          frame metadata (LiCSBAS input)
  metadata/<frame>-poly.txt      frame footprint polygon (coverage of the clip box)
  metadata/gaps.txt, lackifg.txt portal's own network-gap notes

Existing files are never replaced (L0). One JSON line per request goes to
``results/dataprep/insar_tools/s4a_run/fetch_log_listings.jsonl``; ``write_sources.py`` folds
it into SOURCE.json. The responses are untrusted data and are only saved, never executed.

Run with the hydro env: python -I fetch_listings.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402
from config import FRAMES, META_TEXT, RUN_DIR, frame_url  # noqa: E402

LOG = RUN_DIR / "fetch_log_listings.jsonl"


def get(session: requests.Session, url: str) -> requests.Response:
    last = None
    for k in range(4):
        try:
            r = session.get(url, timeout=120)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:  # transient network error
            last = repr(e)
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"{url}: {last}")


def save(path: Path, content: bytes) -> bool:
    """Write once; an existing identical file is kept, a differing one is an error."""
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"{path} exists with different content (L0 is never modified)")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as f:
        f.write(content)
    path.chmod(0o444)
    return True


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    with open(LOG, "a", encoding="utf-8") as log:
        for frame, spec in FRAMES.items():
            base = frame_url(frame)
            jobs = [(f"{base}/interferograms/", "listings/interferograms.html"),
                    (f"{base}/epochs/", "listings/epochs.html"),
                    (f"{base}/metadata/", "listings/metadata.html")]
            jobs += [(f"{base}/metadata/{m}", f"metadata/{m}") for m in META_TEXT]
            listed = Path(spec["l0_dir"]) / "listings/metadata.html"
            names = listed.read_text(errors="replace") if listed.exists() else ""
            jobs += [(f"{base}/metadata/{m}", f"metadata/{m}")
                     for m in (f"{frame}-poly.txt", "gaps.txt", "lackifg.txt")
                     if f'href="{m}"' in names]
            for url, rel in jobs:
                dest = Path(spec["l0_dir"]) / rel
                if dest.exists():
                    print("exists", dest)
                    continue
                t = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
                r = get(s, url)
                save(dest, r.content)
                rec = {"frame": frame, "file": rel, "url": url, "status": r.status_code,
                       "bytes": len(r.content), "fetched_utc": t,
                       "last_modified": r.headers.get("Last-Modified"),
                       "client": f"python-requests/{requests.__version__}"}
                log.write(json.dumps(rec) + "\n")
                print(frame, rel, len(r.content))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
