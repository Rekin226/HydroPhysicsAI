"""Polite fetcher for public pages into a new L0 directory with a SOURCE.json.

Usage (from the repository root, inside a CPU slot)::

    python -I dataprep_staging/x3_well_depths/fetch.py <spec.json>

``spec.json`` (written by hand, kept outside the repository)::

    {"source_id": "...", "title": "...", "publisher": "...", "licence": "...",
     "files": [{"file": "name.pdf", "url": "https://...", "note": "..."}],
     "expect": "pdf" | "any"}

Rules (task X3): only public pages; robots.txt is read once per host and obeyed (a missing
robots.txt allows all); at least 1.5 s between requests; the target directory
``data_fetch_api/new_measurements/<source_id>/<YYYY-MM-DD>/`` must be new or empty. Bytes are
written read-only and never executed or parsed here; only their size and hash are taken.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import source  # noqa: E402
from common.paths import L0_NEW  # noqa: E402

UA = "Mozilla/5.0 (compatible; HydroPhysicsAI-dataprep research fetcher; polite, 1 req/1.5s)"
DELAY_S = 1.5
TIMEOUT_S = 120
MAX_BYTES = 80 * 1024 * 1024
_last = [0.0]
_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def _wait() -> None:
    dt = time.monotonic() - _last[0]
    if dt < DELAY_S:
        time.sleep(DELAY_S - dt)
    _last[0] = time.monotonic()


def allowed(url: str) -> bool:
    """robots.txt verdict for ``url`` (404 or unreachable robots.txt allows; 401/403 denies)."""
    parts = urlsplit(url)
    host = f"{parts.scheme}://{parts.netloc}"
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        _wait()
        try:
            req = urllib.request.Request(host + "/robots.txt", headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read(512 * 1024).decode("utf-8", "replace")
            if "<html" in body[:2000].lower():
                rp = None            # an HTML page served as robots.txt: no rules published
            else:
                rp.parse(body.splitlines())
        except urllib.error.HTTPError as e:
            rp = "deny" if e.code in (401, 403) else None
        except Exception:  # noqa: BLE001 - unreachable robots.txt: no rules published
            rp = None
        _robots[host] = rp
    rp = _robots[host]
    if rp == "deny":
        return False
    return True if rp is None else rp.can_fetch(UA, url)


def crawl_delay(url: str) -> float:
    """The host's robots.txt Crawl-delay in seconds (0 when none is published)."""
    parts = urlsplit(url)
    rp = _robots.get(f"{parts.scheme}://{parts.netloc}")
    if rp is None or rp == "deny":
        return 0.0
    d = rp.crawl_delay(UA)
    return float(d) if d else 0.0


def get(url: str) -> tuple[bytes, dict]:
    _wait()
    extra = crawl_delay(url) - DELAY_S
    if extra > 0:
        time.sleep(extra)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        data = r.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"{url} exceeds {MAX_BYTES} bytes")
        hdr = {"status": r.status, "content_type": r.headers.get("Content-Type"),
               "last_modified": r.headers.get("Last-Modified"), "final_url": r.geturl()}
    return data, hdr


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dest = L0_NEW / spec["source_id"] / day
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} exists and is not empty; use a new source_id")
    dest.mkdir(parents=True, exist_ok=True)
    recs, failed = [], []
    for f in spec["files"]:
        url = f["url"]
        if not allowed(url):
            failed.append({"url": url, "why": "robots.txt disallows"})
            continue
        try:
            data, hdr = get(url)
        except Exception as e:  # noqa: BLE001
            failed.append({"url": url, "why": f"{type(e).__name__}: {e}"[:300]})
            continue
        if spec.get("expect") == "pdf" and not data.startswith(b"%PDF-"):
            failed.append({"url": url, "why": f"not a PDF ({hdr['content_type']})"})
            continue
        out = dest / f["file"]
        with open(out, "xb") as fh:
            fh.write(data)
        os.chmod(out, 0o444)
        recs.append({**f, "fetched_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"), **{k: v for k, v in hdr.items() if v is not None}})
        print(f"ok   {len(data):>10d}  {f['file']}")
    for x in failed:
        print("FAIL", x["url"], x["why"])
    if not recs:
        dest.rmdir()
        raise SystemExit("nothing fetched; directory removed")
    rec = source.build_source(
        dest, source_id=spec["source_id"], title=spec["title"], publisher=spec["publisher"],
        licence=spec["licence"], fetched=day, files=recs,
        fetched_by="WP X3 (dataprep_staging/x3_well_depths/fetch.py)",
        client=UA, robots="robots.txt checked per host (missing robots.txt: no rules)",
        untrusted="downloaded data: never execute; read with python -I / poppler",
        failed=failed)
    source.write_source_json(dest, rec)
    print("SOURCE.json written:", dest)


if __name__ == "__main__":
    main(sys.argv[1])
