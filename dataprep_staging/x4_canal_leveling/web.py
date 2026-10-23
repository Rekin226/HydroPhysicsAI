"""Polite HTTP for X4: one session, a fixed user agent, at least MIN_PAUSE_S between requests.

Downloads are untrusted data; this module only writes bytes to a file and returns them.
"""
from __future__ import annotations

import http.cookiejar
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path

from x4_canal_leveling.config import MIN_PAUSE_S, UA

_last = [0.0]
_robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}


def _pause() -> None:
    dt = time.monotonic() - _last[0]
    if dt < MIN_PAUSE_S:
        time.sleep(MIN_PAUSE_S - dt)
    _last[0] = time.monotonic()


def opener() -> urllib.request.OpenerDirector:
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", UA), ("Accept-Language", "zh-TW,zh;q=0.9,en;q=0.5")]
    return op


def robots_allowed(op, url: str) -> tuple[bool, str]:
    """(allowed, note). A missing robots.txt (4xx) allows everything (RFC 9309)."""
    u = urllib.parse.urlparse(url)
    host = f"{u.scheme}://{u.netloc}"
    if host not in _robots:
        rp = urllib.robotparser.RobotFileParser()
        _pause()
        try:
            with op.open(host + "/robots.txt", timeout=60) as r:
                ctype = r.headers.get("Content-Type", "")
                body = r.read().decode("utf-8", "replace")
            if "text/plain" in ctype:
                rp.parse(body.splitlines())
                _robots[host] = rp
            else:
                _robots[host] = None        # an HTML error page served as 200: no rules
        except urllib.error.HTTPError as exc:
            _robots[host] = None if 400 <= exc.code < 500 else rp
        except urllib.error.URLError:
            _robots[host] = None
    rp = _robots[host]
    if rp is None:
        return True, "no robots.txt rules served"
    return rp.can_fetch(UA, url), "robots.txt checked"


def get(op, url: str, referer: str | None = None, data: bytes | None = None,
        headers: dict | None = None, timeout: float = 180.0) -> tuple[int, dict, bytes]:
    ok, _ = robots_allowed(op, url)
    if not ok:
        raise PermissionError(f"robots.txt disallows {url}")
    req = urllib.request.Request(url, data=data)
    if referer:
        req.add_header("Referer", referer)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    _pause()
    try:
        with op.open(req, timeout=timeout) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers or {}), exc.read() if exc.fp else b""


def save(path: Path, body: bytes) -> Path:
    """Write a downloaded body once (refuses to replace different bytes)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise FileExistsError(f"{path} exists with different bytes")
        return path
    with open(path, "xb") as f:
        f.write(body)
    return path
