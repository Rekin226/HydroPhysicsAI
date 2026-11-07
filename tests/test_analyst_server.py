"""The local web shell (stdlib server) on a synthetic page; only loopback traffic."""
from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request

import pytest

from hydrophysics_analyst import testing
from hydrophysics_analyst.llm.backend import FakeBackend
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.router import routed_presets
from hydrophysics_analyst.router.llm import LLMRouter
from hydrophysics_analyst.server import (
    CSP_PAGE,
    CSP_SHELL,
    FRAME_SELF,
    App,
    hash_token,
    make_server,
)


@pytest.fixture(autouse=True)
def loopback_only(monkeypatch):
    real = socket.socket.connect

    def guard(self, addr):
        host = addr[0] if isinstance(addr, tuple) else addr
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise AssertionError(f"no network beyond localhost: {addr}")
        return real(self, addr)
    monkeypatch.setattr(socket.socket, "connect", guard)


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("srv")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


def _serve(app):
    srv = make_server(app, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"} | (headers or {}))
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_routes_and_csp(b, tmp_path):
    app = App(b, llm=testing.echo_llm(b), audit_dir=tmp_path)       # presets only via the LLM
    srv, base = _serve(app)
    try:
        st, h, body = _get(base + "/")
        # loopback: no frame-ancestors, so an editor's built-in browser can show the shell
        assert st == 200 and h["Content-Security-Policy"] == CSP_SHELL and b"shell.js" in body
        assert "frame-ancestors" not in h["Content-Security-Policy"]
        st, h, body = _get(base + f"/p/{b.page_sha256}/")
        assert h["Content-Security-Policy"] == CSP_PAGE and b'id="payload"' in body
        assert h["X-Content-Type-Options"] == "nosniff"
        meta = json.loads(_get(base + "/api/meta")[2])
        assert meta["starts"] == [2026, 2030] and len(meta["classes"]) == 6
        with pytest.raises(urllib.error.HTTPError):
            _get(base + "/static/../page.py")
        v = _post(base + "/api/ask", {"question": "Retire aquaculture from 2026", "session": "s1"})
        assert v["preset"] == "policy_head" and v["values"]["head"].startswith("+") and v["audit_id"]
        assert all(any(ch.isdigit() for ch in c["topic"]) is False for c in v["citations"])
    finally:
        srv.shutdown()


def test_edits_never_call_the_llm(b, tmp_path):
    llm = LLMRouter(b, FakeBackend([]), presets=routed_presets())       # empty script: a call fails
    app = App(b, use_llm=True, llm=llm, audit_dir=tmp_path)
    srv, base = _serve(app)
    try:
        v = _post(base + "/api/answer", {"preset": "policy_head", "lang": "zh", "session": "s",
                                         "params": {"cuts": [{"class": "irrigation", "pct": 30}],
                                                    "start": 2030, "year": 2032}})
        assert v["preset"] == "policy_head" and v["lang"] == "zh" and not v["violations"]
    finally:
        srv.shutdown()


def test_non_loopback_needs_auth(b):
    with pytest.raises(SystemExit):
        make_server(App(b, use_llm=False), "0.0.0.0", 0, allow_remote=True)
    with pytest.raises(SystemExit):
        make_server(App(b, use_llm=False, users={"a": hash_token("t")}), "0.0.0.0", 0)


def test_token_auth(b, tmp_path):
    app = App(b, use_llm=False, users={"alice": hash_token("letmein")}, audit_dir=tmp_path)
    srv, base = _serve(app)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/api/meta")
        assert e.value.code == 401
        st, _, _ = _get(base + "/api/meta", {"X-Analyst-Token": "letmein"})
        assert st == 200
    finally:
        srv.shutdown()


def test_shared_installs_forbid_framing(b, tmp_path):
    app = App(b, use_llm=False, audit_dir=tmp_path, behind_proxy=True)
    srv, base = _serve(app)
    app.embeddable = False                       # what make_server sets for a non-loopback bind
    try:
        _, h, _ = _get(base + "/", {"X-Remote-User": "u"})
        assert h["Content-Security-Policy"] == CSP_SHELL + FRAME_SELF
        _, h, _ = _get(base + f"/p/{b.page_sha256}/", {"X-Remote-User": "u"})
        assert h["Content-Security-Policy"] == CSP_PAGE + FRAME_SELF
    finally:
        srv.shutdown()


def test_a_handler_error_answers_500_with_a_message(b, tmp_path, monkeypatch, capsys):
    app = App(b, use_llm=False, audit_dir=tmp_path)
    monkeypatch.setattr(app, "ask", lambda body, user: 1 / 0)
    srv, base = _serve(app)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/api/ask", {"question": "x"})
        assert e.value.code == 500 and "ZeroDivisionError" in json.loads(e.value.read())["error"]
    finally:
        srv.shutdown()
