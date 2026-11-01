"""Phase 5: how_sure, the MCP stdio server, and private bundles with their gates."""
from __future__ import annotations

import io
import json

import pytest

from hydrophysics_analyst import bundle as BU
from hydrophysics_analyst import presets as PR
from hydrophysics_analyst import testing
from hydrophysics_analyst.mcp_server import RELAY, MCPServer
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.router.rules import RulesRouter
from hydrophysics_analyst.validate import validate


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("p5")
    page = testing.build_synthetic_page(d)
    led = testing.synthetic_ledger(d, page)
    return page, led, PageBundle.load(page, led, allow_pending=True)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_how_sure_is_page_text_and_validates(built, lang):
    b = built[2]
    a = PR.run(b, "how_sure", {}, lang)
    assert validate(a, b) == []
    assert any(s.key.startswith("ev_") for s in a.spans)
    st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
    assert not any("W" in st[c].split("+") for c, _ in a.citations)
    assert not any(s.key == "evc_pro_iv" for s in a.spans)          # a band key: kept out


def test_how_sure_after_the_scoring_date(built, monkeypatch):
    b = built[2]
    monkeypatch.setenv("ANALYST_TODAY", "2028-02-01")
    a = PR.run(b, "how_sure", {}, "en")
    assert "banner_stale_wording" in a.banners and any(s.key == "frame_scoring_due" for s in a.spans)
    assert validate(a, b) == []


def test_how_sure_is_routed(built):
    for q in ("How sure are we about this?", "這個結果可信嗎？"):
        assert RulesRouter(built[2]).route(q).preset == "how_sure"


def _rpc(srv, msgs):
    out = io.StringIO()
    srv.serve(io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def test_mcp_initialize_list_and_call(built, tmp_path):
    srv = MCPServer(built[2], use_llm=False, audit_dir=tmp_path)
    r = _rpc(srv, [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                   {"jsonrpc": "2.0", "method": "notifications/initialized"},
                   {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                   {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "ask_twin", "arguments": {"question": "Retire aquaculture from 2026"}}},
                   {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "policy_head", "arguments": {"cuts": [{"class": "irrigation", "pct": 30}]}}},
                   {"jsonrpc": "2.0", "id": 5, "method": "nope"}])
    assert [x["id"] for x in r] == [1, 2, 3, 4, 5]                       # no reply to the notification
    assert r[0]["result"]["serverInfo"]["name"] == "twin-analyst"
    names = {t["name"] for t in r[1]["result"]["tools"]}
    assert {"ask_twin", "policy_head", "how_sure"} <= names and "refuse" not in names
    for i in (2, 3):
        txt = r[i]["result"]["content"][0]["text"]
        assert txt.startswith(RELAY) and "C01" in txt and not r[i]["result"]["isError"]
    assert r[4]["error"]["code"] == -32601


def test_bundle_gates_and_atomic_switch(built, tmp_path):
    page, led, b = built
    root = tmp_path / "bundles"
    with pytest.raises(BU.GateFailed):                                    # no parity report
        BU.build_bundle(page, root, ledger=led)
    assert not (root / "current.json").exists()
    par = tmp_path / "parity.json"
    par.write_text(json.dumps({"page_sha256": b.page_sha256, "n_cases": 200, "n_mismatch": 0, "sha": "x"}))
    ev = tmp_path / "summary.json"
    ev.write_text(json.dumps({"page_sha256": b.page_sha256, "decision": "opus-low",
                              "summaries": {"opus-low": {"accuracy": 95.0, "unsafe": 0}}}))
    with pytest.raises(BU.GateFailed, match="eval"):
        BU.build_bundle(page, root, ledger=led, parity=par, evalsum=ev)
    assert not (root / "current.json").exists()
    ev.write_text(json.dumps({"page_sha256": b.page_sha256, "decision": "opus-low",
                              "summaries": {"opus-low": {"accuracy": 98.1, "unsafe": 0}}}))
    meta = BU.build_bundle(page, root, ledger=led, parity=par, evalsum=ev)
    cur = json.loads((root / "current.json").read_text())
    assert cur["page_sha256"] == b.page_sha256 == meta["page_sha256"]
    assert (root / b.page_sha256 / "page.local.html").read_bytes() == page.read_bytes()
    assert meta["wording_valid_until"] == "2028-01-15"
    loaded = PageBundle.load(BU.current(root) / "page.local.html", BU.current(root) / "claims_v1.json",
                             mode="deploy")
    assert loaded.page_sha256 == b.page_sha256
