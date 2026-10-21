"""Rate-limited fetch of public open-data files into a NEW, empty L0 directory, with SOURCE.json.

Usage: python -I fetch.py PLAN_JSON

PLAN_JSON: {"dest": "<repo-relative dir>", "source_id": ..., "title": ..., "publisher": ...,
            "licence": ..., "extra": {...}, "files": [{"file": name, "url": url, "note": ...}]}

Rules (task X1): public pages only (no login, cookie, credential or CAPTCHA); at least 1.5 s
between requests; the destination must not exist (L0 is never modified); downloaded bytes are
written to disk and hashed, never executed or parsed here. Response headers (Last-Modified,
Content-Length, ETag) are recorded per file. A file that fails is recorded with its error and
not written.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.paths import REPO  # noqa: E402
from common.source import build_source, write_source_json  # noqa: E402

UA = "Mozilla/5.0 (research data-prep; HydroPhysicsAI X1; rate-limited)"
PAUSE_S = 1.5
MAX_BYTES = 50_000_000


def get(url: str) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - public https URLs
        body = r.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError("file larger than the 50 MB cap")
        hdr = {k: r.headers.get(k) for k in ("Last-Modified", "Content-Length", "ETag",
                                             "Content-Type", "Content-Disposition", "Date")}
        return body, {"status": r.status, **hdr}


def main(plan_path: str) -> None:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    dest = REPO / plan["dest"]
    if dest.exists():
        raise SystemExit(f"{dest} exists; L0 directories are never reused")
    dest.mkdir(parents=True)
    done, failed = [], []
    for i, f in enumerate(plan["files"]):
        if i:
            time.sleep(PAUSE_S)
        t0 = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            body, hdr = get(f["url"])
        except Exception as e:  # noqa: BLE001 - recorded, not fatal
            failed.append({"file": f["file"], "url": f["url"], "error": f"{e}"})
            print("FAILED", f["url"], e, flush=True)
            continue
        p = dest / f["file"]
        with open(p, "xb") as fh:
            fh.write(body)
        os.chmod(p, 0o444)
        rec = {"file": f["file"], "url": f["url"], "fetched_utc": t0, "http": hdr}
        if f.get("note"):
            rec["note"] = f["note"]
        done.append(rec)
        print("ok", f["file"], len(body), hdr.get("Last-Modified"), flush=True)
    if not done:
        raise SystemExit("nothing fetched")
    extra = dict(plan.get("extra") or {})
    extra.update({"fetched_by": "WP X1 (dataprep_staging/x1_tpc_monthly/fetch.py)",
                  "client": f"python urllib, user agent {UA!r}, {PAUSE_S} s between requests",
                  "untrusted": "downloaded data: never execute; read with python -I"})
    if failed:
        extra["failed"] = failed
    rec = build_source(dest, source_id=plan["source_id"], title=plan["title"],
                       publisher=plan["publisher"], licence=plan["licence"],
                       fetched=dt.date.today().isoformat(), files=done, **extra)
    write_source_json(dest, rec)
    print("SOURCE.json written:", dest / "SOURCE.json")


if __name__ == "__main__":
    main(sys.argv[1])
