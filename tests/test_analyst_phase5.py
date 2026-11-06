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


def _with_results(tmp_path, results: dict | None):
    from hydrophysics.twin.release import read_payload
    page = testing.build_synthetic_page(tmp_path)
    p = read_payload(page)
    mc = p["modelcard"]
    mc["registrations"] = dict(mc.get("registrations") or {}, fv=True, fs=True)
    mc["prospective"] = {"registered_on": "2026-10-11", "stations": 158, "test_start": "2026-11-01",
                         "test_end": "2027-10-31", "result": "pending"}
    if results is not None:
        mc["results"] = results
    K = len(p["classes"])
    p["n2Unit"]["basis"][K:] = [list(r) for r in p["n2Unit"]["basis"][:K]]    # equal starts: note shows
    testing._write_payload(page, p)
    return PageBundle.load(page, testing.synthetic_ledger(tmp_path, page), allow_pending=True)


def test_fv_not_evaluable_line_mirrors_the_page(tmp_path):
    """FV.decidability.v1: the fixed-beforehand line until its report exists, then the declared
    not-evaluable line with the page's esc() of the reason (round-6 page keys)."""
    from hydrophysics_analyst import query as Q
    b = _with_results(tmp_path, {"fv": {"why": "a <b> & c"}})
    if "evc_pro_fv_ne" not in b.T["en"]:
        pytest.skip("template without the FV not-evaluable line")
    sn = {s.key: s.text for s in Q.evidence(b, "en").snippets}
    assert "evc_pro_fv_ne" in sn and "a &lt;b&gt; &amp; c" in sn["evc_pro_fv_ne"]
    assert validate(PR.run(b, "how_sure", {}, "en"), b) == []


def test_fs_pass_extends_the_start_note(tmp_path):
    from hydrophysics_analyst import query as Q
    b = _with_results(tmp_path, {"fs": {"wording_case": "pass"}})
    if "start_note_fs" not in b.T["en"]:
        pytest.skip("template without the FS start-note clause")
    r = Q.start_compare(b, Q.Policy.from_cuts({0: 50}, 2026).f, 2032, "en")
    note = next(s for s in r.snippets if s.key.startswith("start_note"))
    assert note.key == "start_note+start_note_fs" and b.T["en"]["start_note_fs"].strip() in note.text
    a = PR.run(b, "start_compare", {"cuts": [{"class": "irrigation", "pct": 50}], "year": 2032}, "en")
    assert validate(a, b) == []


R7_RESULTS = {
    "dr": {"Yunlin": {"evaluable": True, "excess_drift_6y_m": 0.21, "include_drift": False}},
    "fm": {"case": "within_first_year", "clause_case": "window_beyond_memory", "lstar_months": 11,
           "lstar_month": "2023-11", "floor_m": 0.05},
    "fvext": {"status": "scored", "reading": "weak_by_design", "p": 0.42},
    "fv": {"why": "superseded"},
    "fv2": {"status": "scored", "level": "not_shown_weak", "p_level": 0.03,
            "coverage": {"Yunlin": {"reading": "weak_by_design", "p": 0.4},
                         "Changhua": {"reading": "not_shown_weak", "p": 0.01}}},
    "fv2fy": {"status": "scored", "rain": "climatology",
              "rc": {"Yunlin": {"reading": "omits", "rc": 0.8, "t_lo": 0.85, "t_hi": 0.95, "lo": 0.7, "hi": 0.9},
                     "Changhua": {"reading": "omits", "rc": 0.6, "t_lo": 0.85, "t_hi": 0.95, "lo": 0.5, "hi": 0.7}}},
    "chx": {"beta": 0.31, "lo": 0.1, "hi": 0.52, "n_wells": 40},
    "chb": {"beta": 0.27, "lo": 0.05, "hi": 0.49, "n_wells": 40},
}


def test_round7_evidence_lines_mirror_the_page(tmp_path):
    """Round 7: E02d, FM (declared case only), FV.ext, FV v2 level and coverage, the Changhua FV v2
    sentence, E25/E26 direction rows. DR drift and FV v2 forcing-year lines are band keys and stay
    out of answers (page_scoping.answer_must_not_use_band_keys)."""
    from hydrophysics.twin.release import read_payload
    from hydrophysics_analyst import query as Q
    page = testing.build_synthetic_page(tmp_path)
    p = read_payload(page)
    mc = p["modelcard"]
    mc["registrations"] = dict(mc.get("registrations") or {}, e02c=True, e02d=True, ext=True, fm=True,
                               fv=True, fv2=True, fvext=True, dr=True, chx=True, chb=True)
    mc["prospective"] = {"registered_on": "2026-10-11", "stations": 158, "test_start": "2026-11-01",
                         "test_end": "2027-10-31", "result": "pending", "interval_bound": True}
    mc["results"] = R7_RESULTS
    testing._write_payload(page, p)
    b = PageBundle.load(page, testing.synthetic_ledger(tmp_path, page), allow_pending=True)
    if "evc_fv2_lvl_nsw" not in b.T["en"]:
        pytest.skip("template without the round-7 evidence lines")
    keys = [s.key for s in Q.evidence(b, "en").snippets]
    for k in ("evc_pro_e02d", "evc_pro_fm", "evc_fvext_wbd", "evc_fv2_lvl_nsw", "evc_fv2_cov_wbd",
              "evc_fv2_cov_nsw", "evc_fv2_ch_clim", "evc_chx", "evc_chb"):
        assert k in keys, k
    banned = set(b.ledger["page_scoping"]["answer_must_not_use_band_keys"])
    assert not banned & set(keys) and "evc_pro_fv_ne" not in keys      # v2 level replaces v1's line
    for lang in ("en", "zh"):
        assert validate(PR.run(b, "how_sure", {}, lang), b) == []


R87_HYBRID = {"rmse_stat": 1.0, "rmse_clim": 1.2, "rmse_model": 2.0, "ratio_stat": 0.9,
              "ratio_phys": 1.0, "per_year": {}, "forecast_origin": "2022-12",
              "end_exclusive": "2026-09",
              "registration": {"registered_on": "2026-10-11", "stations": 147,
                               "test_start": "2026-11-01", "test_end": "2027-10-31"}}
R87_KEYS = ("evc_pro_e02z", "evc_hyb_reg", "evc_hyb_for", "evc_hyb_fvs", "evc_hyb_fvs_pass")


def _r87_page(tmp_path):
    from hydrophysics.twin.release import read_payload
    page = testing.build_synthetic_page(tmp_path)
    p = read_payload(page)
    mc = p["modelcard"]
    mc["registrations"] = dict(mc.get("registrations") or {}, e02c=True, e02z=True, fm=True,
                               fvs2=True)
    mc["prospective"] = {"registered_on": "2026-10-11", "stations": 158,
                         "test_start": "2026-11-01", "test_end": "2027-10-31",
                         "result": "pending", "interval_bound": False}
    mc["hybrid"] = R87_HYBRID
    mc["results"] = {"fm": R7_RESULTS["fm"],
                     "fvs2": {"status": "scored", "reading": "not_shown_weak", "p": 0.9296,
                              "p0": 0.4753}}
    testing._write_payload(page, p)
    return page, PageBundle.load(page, testing.synthetic_ledger(tmp_path, page),
                                 allow_pending=True)


def test_r8_7_forecast_of_record_lines_mirror_the_page(tmp_path):
    """Ruling R8.7: the analyst's evidence carries the page's forecast-of-record lines (E04
    named, FV.stat.v2's two lines, E02z's sub-label) in the page's order and cites the same
    rows (C30-fc, C30-scen, E04, E30); no answer cites the pre-split C30."""
    from hydrophysics.twin import claims as CL
    from hydrophysics_analyst import binding as BD
    from hydrophysics_analyst import query as Q
    _, b = _r87_page(tmp_path)
    if "evc_hyb_for" not in b.T["en"]:
        pytest.skip("template without the R8.7 lines")
    for lang in ("en", "zh"):
        sn = Q.evidence(b, lang).snippets
        keys = [s.key for s in sn]
        for k in R87_KEYS:
            assert k in keys, (lang, k)
        assert keys.index("evc_pro_e02c") < keys.index("evc_pro_e02z")
        assert [keys.index(k) for k in R87_KEYS[1:]] == sorted(keys.index(k) for k in R87_KEYS[1:])
        txt = {s.key: s.text for s in sn}
        if lang == "en":
            assert txt["evc_hyb_for"].startswith("Forecast of record for Nov 2026–Oct 2027: this "
                                                 "registered statistical layer (E04)")
            assert "passes with probability 0.93" in txt["evc_hyb_fvs"]
            assert "probability 0.48" in txt["evc_hyb_fvs"]
            assert "a fail has probability about 0.07" in txt["evc_hyb_fvs_pass"]
        assert validate(PR.run(b, "how_sure", {}, lang), b) == []
    # the claim set of these lines is the page's own (display_map), and never the old C30
    page_keys = CL.function_keys("evidenceItems", follow=True) | CL.function_keys("fvsLines")
    assert set(R87_KEYS) <= page_keys
    cited = {c for k in R87_KEYS for c in BD.snippet_claims(k, b.ledger)}
    assert cited == {"C30-fc", "C30-scen", "E04", "E30"}
    a = PR.run(b, "how_sure", {}, "en")
    got = {c for c, _ in a.citations}
    assert {"C30-fc", "E04", "C30-scen"} <= got and "C30" not in got
    ids = {c["id"] for c in b.ledger["claims"]}
    rules_ids = {c for r in BD.rules().values() for c in r.claims} | set(BD.topics()) | set(
        BD.scoring())
    assert rules_ids <= ids and "C30" not in rules_ids


def test_r8_7_forecast_of_record_text_equals_the_page(tmp_path):
    """The rendered page's evidenceItems() and the analyst give the same R8.7 sentences (EN and
    ZH), in headless Chromium."""
    sync = pytest.importorskip("playwright.sync_api")
    page, b = _r87_page(tmp_path)
    if "evc_hyb_for" not in b.T["en"]:
        pytest.skip("template without the R8.7 lines")
    from hydrophysics.twin.browser_check import _exe
    from hydrophysics_analyst import query as Q
    from hydrophysics_analyst.jsport import strip_tags
    pw = sync.sync_playwright().start()
    try:
        browser = pw.chromium.launch(**_exe())
    except Exception as e:  # noqa: BLE001
        pw.stop()
        pytest.skip(f"no Chromium for Playwright: {type(e).__name__}")
    try:
        pg = browser.new_page()
        pg.goto(page.as_uri())
        pg.wait_for_function("document.body.dataset.ready === '1'", timeout=30000)
        got = pg.evaluate("() => { const out = {}; for (const l of ['en', 'zh']) { LANG = l; "
                          "out[l] = evidenceItems().map(x => x[1]); } LANG = 'en'; return out; }")
    finally:
        browser.close()
        pw.stop()
    for lang in ("en", "zh"):
        mine = {s.key: strip_tags(s.text) for s in Q.evidence(b, lang).snippets}
        page_txt = [strip_tags(t) for t in got[lang]]
        for k in R87_KEYS:
            assert mine[k] in page_txt, (lang, k, mine[k][:80])
