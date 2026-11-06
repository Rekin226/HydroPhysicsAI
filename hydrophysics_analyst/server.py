"""The local web shell: chat on the left, the bound page on the right (same-origin iframe).

    python -m hydrophysics_analyst serve --page results/twin/twin_app_20261107a.html
        [--port 8765] [--no-llm] [--config analyst_users.json]
        [--allow-remote --behind-proxy]

Standard library only. Binds 127.0.0.1 by default; a non-loopback bind needs ``--allow-remote``
and either a user config (hashed per-user tokens) or ``--behind-proxy`` (trusts the proxy's
``X-Remote-User``). Routes:

  GET  /                 the shell                 GET /api/meta     classes, towns, years, presets
  GET  /p/<page_sha>/    the bound page (bytes)    POST /api/ask     question -> routed answer
  GET  /static/<file>    shell assets              POST /api/answer  edited route -> answer (no LLM)

The shell drives the page only through its URL hash (the page listens for ``hashchange``); the
page itself is served unchanged. CSP: strict on shell routes; the page gets the policy it needs
for its inline script, styles and ``data:`` images, and no network.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import mimetypes
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import presets as PR
from . import query as Q
from .page import BindingError, PageBundle
from .render import S, strings
from .session import Session

STATIC = Path(__file__).with_name("static")
CSP_SHELL = "default-src 'self'; object-src 'none'; base-uri 'none'"
CSP_PAGE = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
            "img-src data: blob:; connect-src 'none'; worker-src blob:")
# Shared (non-loopback) installs also forbid framing by other origins (clickjacking). On a
# loopback bind it is left off so an editor's built-in browser (VS Code "Preview in Editor",
# which frames the page from its own origin) can show the shell instead of a blank tab.
FRAME_SELF = "; frame-ancestors 'self'"
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def hash_token(tok: str) -> str:
    return hashlib.sha256(tok.encode("utf-8")).hexdigest()


class App:
    def __init__(self, b: PageBundle, *, use_llm: bool = True, llm=None, users: dict | None = None,
                 behind_proxy: bool = False, audit_dir: Path | None = Path("results/analyst/audit"),
                 bundles: Path | None = None, llm_config: dict | None = None):
        self.b, self.use_llm, self.users, self.behind_proxy = b, use_llm, users or {}, behind_proxy
        self.audit_dir, self.bundles, self.llm_config = audit_dir, bundles, llm_config or {}
        self._llm = llm
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()
        self._cur_mtime = self._mtime()
        self.embeddable = True            # make_server: False for non-loopback binds

    def llm(self):
        if self._llm is None and self.use_llm:
            from .ask import make_llm
            cfg = self.llm_config
            self._llm = make_llm(self.b, model=cfg.get("model"), fallbacks=cfg.get("fallbacks", True),
                                 inference_geo=cfg.get("inference_geo"))
        return self._llm

    # ---- bundles: serve the current one, reload when current.json changes ------------
    def _mtime(self) -> float | None:
        if not self.bundles:
            return None
        f = self.bundles / "current.json"
        return f.stat().st_mtime if f.exists() else None

    def maybe_reload(self) -> None:
        m = self._mtime()
        if self.bundles is None or m is None or m == self._cur_mtime:
            return
        with self.lock:
            from .bundle import current
            d = current(self.bundles)
            self.b = PageBundle.load(d / "page.local.html", d / "claims_v1.json", mode=self.b.mode)
            self._llm, self.sessions, self._cur_mtime = None, {}, m

    def page_bytes(self, sha: str) -> bytes | None:
        if sha == self.b.page_sha256:
            return self.b.page_path.read_bytes()
        if self.bundles and len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha):
            f = self.bundles / sha / "page.local.html"
            if f.exists():                     # old bundles stay served: old links still resolve
                return f.read_bytes()
        return None

    def session(self, sid: str) -> Session:
        with self.lock:
            return self.sessions.setdefault(sid or "default", Session())

    def meta(self) -> dict:
        from .router import routed_presets
        c = Q.ctx(self.b)
        return {"classes": [{"key": cl["key"], "en": cl["en"], "zh": cl["zh"]} for cl in c.classes],
                "towns": [{"i": i, "en": t["en"], "zh": t["zh"], "pooled": bool(t.get("pooled"))}
                          for i, t in enumerate(c.towns)],
                "years": c.years[c.YR + 1:], "starts": c.starts, "pct": PR.PCT,
                "presets": list(routed_presets()), "page": f"/p/{self.b.page_sha256}/",
                "examples": PR.examples(self.b),
                "strings": strings(), "llm": bool(self.use_llm)}

    def view(self, ans, rec: dict | None, violations=()) -> dict:
        from .jsport import strip_tags
        vals = {}
        for s in ans.spans:
            if s.kind == "value":
                vals.setdefault(s.key, s.text)
        blocks: list[dict] = []
        for s in ans.spans:
            if s.kind == "value" and s.inline:
                continue
            txt = strip_tags(s.text) + (f" {s.unit}" if s.kind == "value" and s.unit else "")
            if blocks and blocks[-1]["block"] == s.block:
                blocks[-1]["text"] += " " + txt
            else:
                blocks.append({"block": s.block, "text": txt})
        from .binding import topics
        tp = topics()                      # digit-free tags; never the ledger one-liners (W numbers)
        from .render import resolve
        charts = []
        for ch in ans.charts:
            ser = []
            for sr in ch.series:
                pts = []
                for xi, ref in enumerate(sr.refs):
                    v = resolve(ans, ref)            # the answer's own value: number + page text
                    if v.value is None or v.value != v.value:
                        continue
                    pts.append({"x": ch.x[xi] if xi < len(ch.x) else xi, "y": float(v.value),
                                "text": v.text})
                ser.append({"label": sr.label, "points": pts})
            charts.append({"kind": ch.kind, "title": strip_tags(ch.title), "unit": ch.unit,
                           "block": ch.block, "x": list(ch.x),
                           "marker": [{"x": m[0], "label": m[1]} for m in ch.marker], "series": ser})
        return {"preset": ans.preset, "params": ans.params, "lang": ans.lang, "charts": charts,
                "echo": [strip_tags(s.text) for s in ans.echo], "blocks": blocks,
                "citations": [{"id": c, "status": s, "topic": tp.get(c, "")} for c, s in ans.citations],
                "table": [list(r) for r in ans.table], "deep_link": ans.deep_link, "values": vals,
                "banners": [S(ans.lang, k) for k in ans.banners], "refusal": ans.refusal,
                "clarify": ans.clarify, "fallback": ans.fallback,
                "audit_id": (rec or {}).get("id"), "stage": (rec or {}).get("stage"),
                "violations": [v.__dict__ for v in violations]}

    def ask(self, body: dict, user: str) -> dict:
        from .ask import ask
        ans, v, rec = ask(self.b, str(body.get("question", ""))[:600], lang=body.get("lang") or None,
                          use_llm=self.use_llm, llm=self.llm(), audit_dir=self.audit_dir, user=user,
                          session=self.session(body.get("session", "")))
        return self.view(ans, rec, v)

    def answer(self, body: dict, user: str) -> dict:
        from . import audit
        from .validate import validate
        lang = body.get("lang") or "en"
        preset = body.get("preset") or ""
        if preset not in PR.PRESETS or preset == "refuse":
            return {"error": "unknown preset"}
        ans = PR.run(self.b, preset, dict(body.get("params") or {}), lang)
        v = validate(ans, self.b)
        if v:
            import dataclasses as dc
            ans = dc.replace(PR.refuse(self.b, {"reason": "no_run"}, lang), fallback="validator")
        else:
            self.session(body.get("session", "")).update(ans.preset, ans.params)
        rec = {"user": user, "lang": lang, "question": None, "route": {"preset": preset, "edited": True},
               "stage": "edit", "answer_preset": ans.preset if not ans.clarify else "clarify",
               "answer_params": ans.params, "answer_text": ans.text(), "citations": ans.citations,
               "refusal": ans.refusal, "clarify": ans.clarify, "fallback": ans.fallback,
               "violations": [x.__dict__ for x in v], "deep_link": ans.deep_link,
               "page_sha256": self.b.page_sha256, "ledger_sha256": self.b.ledger_sha256,
               "strings_sha256": self.b.strings_sha256}
        if self.audit_dir is not None:
            rec["id"] = audit.write(rec, self.audit_dir)
        return self.view(ans, rec, v)


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "analyst"
        sys_version = ""

        def log_message(self, fmt, *args):  # quiet; the audit log is the record
            pass

        def _user(self) -> str | None:
            if app.behind_proxy:
                return self.headers.get("X-Remote-User") or None
            if not app.users:
                return "local"
            tok = self.headers.get("X-Analyst-Token") or ""
            if not tok:
                ck = self.headers.get("Cookie") or ""
                for part in ck.split(";"):
                    k, _, v = part.strip().partition("=")
                    if k == "analyst_token":
                        tok = v
            h = hash_token(tok) if tok else ""
            for name, want in app.users.items():
                if h and secrets.compare_digest(h, want):
                    return name
            return None

        def _send(self, code: int, body: bytes, ctype: str, csp: str = CSP_SHELL, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", csp + ("" if app.embeddable else FRAME_SELF))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self):  # noqa: N802
            self._guard(self._get)

        def do_POST(self):  # noqa: N802
            self._guard(self._post)

        def _guard(self, fn):
            """A handler error answers 500 with a message (the shell shows it) instead of a dropped
            connection that leaves the browser waiting; the traceback goes to the server log."""
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc(file=sys.stderr)
                with contextlib.suppress(Exception):
                    self._json(500, {"error": f"{type(e).__name__}: {str(e)[:200]}"})

        def _get(self):
            app.maybe_reload()
            path = self.path.split("?", 1)[0]
            user = self._user()
            if user is None:
                return self._send(401, b"unauthorized", "text/plain; charset=utf-8")
            if path in ("/", "/index.html"):
                return self._send(200, (STATIC / "shell.html").read_bytes(), "text/html; charset=utf-8")
            if path.startswith("/static/"):
                name = path[len("/static/"):]
                f = (STATIC / name).resolve()
                if f.parent != STATIC.resolve() or not f.is_file():
                    return self._send(404, b"not found", "text/plain; charset=utf-8")
                ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                return self._send(200, f.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text") or ctype.endswith("javascript") else ""))
            if path.startswith("/p/"):
                data = app.page_bytes(path[3:].strip("/"))
                if data is None:
                    return self._send(404, b"not found", "text/plain; charset=utf-8")
                return self._send(200, data, "text/html; charset=utf-8", csp=CSP_PAGE)
            if path == "/api/meta":
                return self._json(200, app.meta())
            return self._send(404, b"not found", "text/plain; charset=utf-8")

        def _post(self):
            app.maybe_reload()
            user = self._user()
            if user is None:
                return self._send(401, b"unauthorized", "text/plain; charset=utf-8")
            n = min(int(self.headers.get("Content-Length") or 0), 65536)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, {"error": "bad json"})
            if self.path == "/api/ask":
                return self._json(200, app.ask(body, user))
            if self.path == "/api/answer":
                return self._json(200, app.answer(body, user))
            return self._json(404, {"error": "not found"})
    return H


def prune_audit(directory: Path, days: int) -> int:
    """Delete audit day-files older than the retention period (deployment config)."""
    import datetime as dt
    cutoff = dt.date.today() - dt.timedelta(days=days)
    n = 0
    for f in Path(directory).glob("*.jsonl"):
        try:
            if dt.date.fromisoformat(f.stem) < cutoff:
                f.unlink()
                n += 1
        except ValueError:
            continue
    return n


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765, *, allow_remote: bool = False):
    if host not in LOOPBACK and not (allow_remote and (app.users or app.behind_proxy)):
        raise SystemExit("refusing a non-loopback bind without --allow-remote and auth "
                         "(--config users or --behind-proxy)")
    app.embeddable = host in LOOPBACK
    return ThreadingHTTPServer((host, port), make_handler(app))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--page", type=Path)
    src.add_argument("--bundles", type=Path, help="a bundle root (python -m hydrophysics_analyst bundle)")
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--config", type=Path, help="deployment config JSON (see docs/analyst/OPS.md): "
                    '{"users": {name: sha256(token)}, "llm": {...}, "audit": {...}}')
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--behind-proxy", action="store_true")
    ap.add_argument("--mode", choices=["dev", "deploy"], default="dev")
    ap.add_argument("--allow-pending", action="store_true")
    ap.add_argument("--unbound", action="store_true")
    a = ap.parse_args(argv)
    cfg = json.loads(a.config.read_text(encoding="utf-8")) if a.config else {}
    try:
        if a.bundles:
            from .bundle import current
            d = current(a.bundles)
            b = PageBundle.load(d / "page.local.html", d / "claims_v1.json", mode="deploy")
        else:
            b = PageBundle.load(a.page, a.ledger, mode=a.mode, allow_pending=a.allow_pending,
                                check_forward=not a.unbound, check_template=not a.unbound)
    except BindingError as e:
        print(f"binding failed: {e}", file=sys.stderr)
        return 3
    llm_cfg = cfg.get("llm", {})
    audit_cfg = cfg.get("audit", {})
    audit_dir = Path(audit_cfg.get("dir", "results/analyst/audit"))
    if audit_cfg.get("retention_days"):
        prune_audit(audit_dir, int(audit_cfg["retention_days"]))
    use_llm = not a.no_llm and llm_cfg.get("enabled", True)
    app = App(b, use_llm=use_llm, users=cfg.get("users", {}), behind_proxy=a.behind_proxy,
              audit_dir=audit_dir, bundles=a.bundles, llm_config=llm_cfg)
    srv = make_server(app, a.host, a.port, allow_remote=a.allow_remote)
    print(f"analyst shell on http://{a.host}:{srv.server_address[1]}/  (page {b.page_sha256[:12]})",
          flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
