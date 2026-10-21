"""Polite fetcher for WP X2: public pages only, robots.txt respected, >= 1.5 s per request.

Every request is appended to ``reports/X2_public_supply/fetch_log.jsonl`` (url, method,
status, bytes, sha256, UTC time, destination). Files are written only into the new X2 L0
directories of :data:`x2const.L0` (created by this module; never a directory that holds a
file X2 did not write) and made read-only. Downloaded bytes are untrusted data: they are
hashed and written, never executed; HTML forms are parsed with regular expressions only.

Usage (CLI):
    python -I fetch.py get <source_key> <file_name> <url> [--note TEXT]
    python -I fetch.py ajax <source_key> <file_name> <url> [--note TEXT]
    python -I fetch.py postback <source_key> <file_name> <page_url> <event_target>
                               <field=value> ... [--note TEXT]
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from x2_public_supply import x2const as K  # noqa: E402

_STAMP = K.REPORT_DIR / ".last_request"
_ROBOTS: dict[str, list[tuple[str, str]] | None] = {}


def parse_robots(text: str) -> list[tuple[str, str]]:
    """Rules of every group that applies to '*' (RFC 9309 2.2.1-2.2.2: a leading BOM is
    ignored, all groups matching the user agent are merged)."""
    rules, agents, in_rules = [], [], False
    for raw in text.lstrip("﻿").splitlines():
        ln = raw.split("#", 1)[0].strip()
        if ":" not in ln:
            continue
        k, v = (s.strip() for s in ln.split(":", 1))
        k = k.lower()
        if k == "user-agent":
            if in_rules:
                agents, in_rules = [], False
            agents.append(v.lower())
        elif k in ("allow", "disallow"):
            in_rules = True
            if "*" in agents and v:
                rules.append((k, v))
    return rules


def rules_allow(rules: list[tuple[str, str]], path: str) -> bool:
    """Longest matching rule wins; an Allow equivalent to a Disallow wins (RFC 9309 2.2.2)."""
    best = None
    for k, v in rules:
        pat = re.escape(v).replace(r"\*", ".*")
        if pat.endswith(r"\$"):
            pat = pat[:-2] + "$"
        if re.match(pat, path):
            key = (len(v), k == "allow")
            if best is None or key > best[0]:
                best = (key, k)
    return best is None or best[1] == "allow"


def _wait() -> None:
    K.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if _STAMP.exists():
        dt = time.time() - _STAMP.stat().st_mtime
        if dt < K.MIN_INTERVAL_S:
            time.sleep(K.MIN_INTERVAL_S - dt)
    _STAMP.touch()


def _log(rec: dict) -> None:
    K.FETCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(K.FETCH_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _request(url: str, data: bytes | None = None, headers: dict | None = None,
             timeout: int = 120) -> tuple[int, bytes, dict]:
    _wait()
    h = {"User-Agent": K.USER_AGENT}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b"", dict(e.headers or {})


def robots_allows(url: str) -> bool:
    """robots.txt check (RFC 9309). A host whose /robots.txt is not a robots file (an HTML
    error page or 404) has no rules. Groups for '*' are merged and an Allow equivalent to a
    Disallow wins (RFC 9309 2.2.2); see :func:`parse_robots`."""
    u = urllib.parse.urlsplit(url)
    host = f"{u.scheme}://{u.netloc}"
    if host not in _ROBOTS:
        st, body, hd = _request(host + "/robots.txt", timeout=30)
        text = body.decode("utf-8", "replace")
        is_robots = st == 200 and "<html" not in text.lower()[:2000] and (
            "user-agent" in text.lower())
        _log({"utc": _now(), "method": "GET", "url": host + "/robots.txt", "status": st,
              "bytes": len(body), "purpose": "robots", "robots_file": is_robots})
        _ROBOTS[host] = parse_robots(text) if is_robots else None
    rules = _ROBOTS[host]
    path = u.path or "/"
    if u.query:
        path += "?" + u.query
    return True if rules is None else rules_allow(rules, path)


def _dest(source_key: str, name: str) -> Path:
    d = K.L0[source_key]
    d.mkdir(parents=True, exist_ok=True)
    mine = {json.loads(ln).get("dest") for ln in K.FETCH_LOG.read_text(
        encoding="utf-8").splitlines() if ln.strip()} if K.FETCH_LOG.exists() else set()
    for p in d.iterdir():
        if p.name != "SOURCE.json" and str(p) not in mine:
            raise SystemExit(f"{d} holds {p.name}, which X2 did not write: refusing")
    out = d / name
    if out.exists():
        raise SystemExit(f"{out} exists; L0 is never modified")
    return out


def _save(out: Path, body: bytes, rec: dict) -> dict:
    out.write_bytes(body)
    os.chmod(out, 0o444)
    rec.update({"dest": str(out), "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest()})
    _log(rec)
    return rec


def get(source_key: str, name: str, url: str, note: str = "") -> dict:
    if not robots_allows(url):
        raise SystemExit(f"robots.txt disallows {url}")
    out = _dest(source_key, name)
    st, body, hd = _request(url)
    rec = {"utc": _now(), "method": "GET", "url": url, "status": st, "source_key": source_key,
           "file": name, "note": note, "content_type": hd.get("Content-Type", ""),
           "last_modified": hd.get("Last-Modified", "")}
    if st != 200:
        _log({**rec, "bytes": len(body), "saved": False})
        raise SystemExit(f"HTTP {st} for {url}")
    return _save(out, body, rec)


def post_ajax(source_key: str, name: str, url: str, note: str = "") -> dict:
    """POST with an empty body and ``X-Requested-With`` (a public list pager that the page's
    own links call this way; no login, no token)."""
    if not robots_allows(url):
        raise SystemExit(f"robots.txt disallows {url}")
    out = _dest(source_key, name)
    st, body, hd = _request(url, data=b"", headers={"X-Requested-With": "XMLHttpRequest"})
    rec = {"utc": _now(), "method": "POST", "url": url, "status": st, "source_key": source_key,
           "file": name, "note": note, "content_type": hd.get("Content-Type", "")}
    if st != 200:
        _log({**rec, "bytes": len(body), "saved": False})
        raise SystemExit(f"HTTP {st} for POST {url}")
    return _save(out, body, rec)


def form_fields(page: str) -> dict:
    """Hidden ASP.NET fields of a public form page (regex only)."""
    f = {}
    for m in re.finditer(r'<input[^>]*type="hidden"[^>]*>', page):
        tag = m.group(0)
        n = re.search(r'name="([^"]+)"', tag)
        v = re.search(r'value="([^"]*)"', tag)
        if n:
            f[html.unescape(n.group(1))] = html.unescape(v.group(1)) if v else ""
    return f


def postback(source_key: str, name: str, page_url: str, event_target: str,
             fields: dict, note: str = "", page_html: str | None = None) -> dict:
    """Submit a public ASP.NET form (no login, no CAPTCHA): GET the page for its hidden
    fields (unless given), then POST them with ``fields``; saves the POST response."""
    if not robots_allows(page_url):
        raise SystemExit(f"robots.txt disallows {page_url}")
    if page_html is None:
        st, body, _ = _request(page_url)
        _log({"utc": _now(), "method": "GET", "url": page_url, "status": st,
              "bytes": len(body), "purpose": "form state for a postback (not saved)"})
        page_html = body.decode("utf-8", "replace")
    data = form_fields(page_html)
    data["__EVENTTARGET"] = event_target
    data["__EVENTARGUMENT"] = ""
    data.update(fields)
    out = _dest(source_key, name)
    st, body, hd = _request(page_url, data=urllib.parse.urlencode(data).encode(),
                            headers={"Content-Type": "application/x-www-form-urlencoded"})
    rec = {"utc": _now(), "method": "POST", "url": page_url, "status": st,
           "source_key": source_key, "file": name, "note": note,
           "post_fields": {k: v for k, v in fields.items()}, "event_target": event_target,
           "content_type": hd.get("Content-Type", "")}
    if st != 200:
        _log({**rec, "bytes": len(body), "saved": False})
        raise SystemExit(f"HTTP {st} for POST {page_url}")
    return _save(out, body, rec)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("get")
    g.add_argument("source_key", choices=sorted(K.L0))
    g.add_argument("name")
    g.add_argument("url")
    g.add_argument("--note", default="")
    p = sub.add_parser("postback")
    p.add_argument("source_key", choices=sorted(K.L0))
    p.add_argument("name")
    p.add_argument("page_url")
    p.add_argument("event_target")
    p.add_argument("fields", nargs="*")
    p.add_argument("--note", default="")
    j = sub.add_parser("ajax")
    j.add_argument("source_key", choices=sorted(K.L0))
    j.add_argument("name")
    j.add_argument("url")
    j.add_argument("--note", default="")
    a = ap.parse_args(argv)
    if a.cmd == "get":
        r = get(a.source_key, a.name, a.url, a.note)
    elif a.cmd == "ajax":
        r = post_ajax(a.source_key, a.name, a.url, a.note)
    else:
        r = postback(a.source_key, a.name, a.page_url, a.event_target,
                     dict(x.split("=", 1) for x in a.fields), a.note)
    print(json.dumps({k: r[k] for k in ("file", "status", "bytes", "sha256")}))


if __name__ == "__main__":
    main()
