"""Round 8 (plan #19): the analyst mirrors every line of the final page's trust drawer except its
band keys, the R8.13 two-part head band with its coverage line, and E24's internals line.

The synthetic page carries every block the drawer reads (made-up numbers, no real data); the
Chromium tests compare the port with the page's own ``evidenceItems()``, ``#tsNote``/``#pbWrap``
and ``e24Line()``, line by line and claim by claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from hydrophysics_analyst import presets as PR
from hydrophysics_analyst import testing
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.validate import validate

MC = {
    "new_data_challenge": {"metrics": {"model_datum": {"rmse_m": 2.1}, "clim": {"rmse_m": 1.7}},
                           "best_baseline": "clim", "beats_best_baseline": False, "stations": 120,
                           "forecast_origin": "2022-12-01", "end_exclusive": "2026-09-01"},
    "coverage": {"picp_90": 0.74, "inflation_90": 1.9, "combined_picp_90": 0.93, "stations": 120,
                 "scored_months": 4321, "forecast_origin": "2022-12-01",
                 "end_exclusive": "2026-09-01"},
    "prospective": {"registered_on": "2026-10-11", "stations": 120, "test_start": "2026-11-01",
                    "test_end": "2027-10-31", "result": "pending", "interval_bound": True},
    "hybrid": {"rmse_stat": 1.5, "rmse_clim": 1.7, "rmse_model": 2.1, "ratio_stat": 0.9,
               "ratio_phys": 1.0, "per_year": {"2024": {"stat": 1.4, "clim": 1.6},
                                               "2023": {"stat": 1.3, "clim": 1.8}},
               "forecast_origin": "2022-12-01", "end_exclusive": "2026-09-01",
               "registration": {"registered_on": "2026-10-11", "stations": 110,
                                "test_start": "2026-11-01", "test_end": "2027-10-31"}},
    "ring_holdout": {"rings": {"n_sites": 9, "cum": {"climatology": {"r2": -0.2, "rmse": 1.5},
                                                     "base_continue": {"r2": -1.1, "rmse": 2.2}}},
                     "window": ["2023-01-01", "2025-12-31"], "max_ref_age_days": 60},
    "deep_piles": {"n_sites": 4, "window": ["2024-01", "2025-12"], "n_verified_values": 99,
                   "cum": {"model": {"rmse": 1.2}, "base_continue": {"rmse": 1.1}},
                   "cum_deep": {"n_sites": 3, "model": {"rmse": 0.5, "n_inside": 2},
                                "base_continue": {"rmse": 1.3}},
                   "after_feeds": {"window": ["2025-06", "2025-12"], "model": {"rmse": 0.7},
                                   "base_continue": {"rmse": 2.0}},
                   "selection": {"date": "2026-10-21", "ruling": "U7"}},
    "policy_response": {
        "window": "2016-01..2018-12 (masked refits; design-time window)",
        "heads": {"yunlin": {"rmse_model_m": 1.4, "rmse_cl_m": 2.2, "rmse_zr_m": 2.3,
                             "verdict_cl": "model_better", "verdict_zr": "model_better",
                             "slope_obs_on_model": 1.2, "slope_ci": [1.05, 1.35],
                             "model": {"id": "R0-m1618", "deliverable_recipe": True}},
                  "changhua": {"rmse_model_m": 1.1, "rmse_cl_m": 1.2, "rmse_zr_m": 1.2,
                               "verdict_cl": "model_better", "verdict_zr": "model_better",
                               "slope_obs_on_model": 0.85, "slope_ci": [0.6, 1.1],
                               "model": {"id": "R1c-m1618", "deliverable_recipe": False}}},
        "subsidence": {"yunlin": {"rms_model_cm": 1.9, "rms_cl_cm": 1.0, "rms_zr_cm": 1.1,
                                  "verdict_cl": "baseline_better", "verdict_zr": "baseline_better",
                                  "slope_obs_on_model": 0.22, "slope_ci": [0.17, 0.27],
                                  "model": {"id": "R0-m1618", "deliverable_recipe": True}},
                       "changhua": {"rms_model_cm": 0.6, "rms_cl_cm": 0.58, "rms_zr_cm": 0.7,
                                    "verdict_cl": "baseline_better", "verdict_zr": "model_better",
                                    "slope_obs_on_model": 0.4, "slope_ci": [0.3, 0.5],
                                    "model": {"id": "R1c-m1618", "deliverable_recipe": False}}},
        "step_b": {"yunlin": {"beta_obs_cm_per_gwh": 0.5, "beta_model_cm_per_gwh": 0.2,
                              "ratio_ci": [1.1, 4.0], "verdict": "inconclusive"}},
        "changhua": {"status": "not_supported", "dose_obs_cm_per_gwh": -0.5,
                     "dose_model_cm_per_gwh": 0.8}},
    "pb": {"ribbon": {"months": ["2026-12", "2027-01", "2027-02"], "small": 1,
                      "label": {"en": "decided by a registered test",
                                "zh": "由已登錄之檢定判定"},
                      "zones": [{"zone": "Yunlin:prox", "county": "Yunlin", "n": 20,
                                 "hw": [1.5, None, 1.8]},
                                {"zone": "Changhua:mid", "county": "Changhua", "n": 12,
                                 "hw": [2.0, 2.1, 2.2]}]},
           "shared": {"label": {"en": "cannot be tested before 2030",
                                "zh": "2030 年前無法檢定"},
                      "counties": {"Yunlin": {"months": ["2026-12", "2027-01"], "lo": [-1.2, -1.5],
                                              "hi": [1.1, 1.4]},
                                   "Changhua": {"months": ["2026-12", "2027-01"],
                                                "lo": [-2.0, -1.0], "hi": [0.3, 0.4]}}},
           "lofo": {"Yunlin": {"inside": 40, "n": 44}, "Changhua": {"inside": 41, "n": 44}}},
}
# the round-8 page lines the port now mirrors (plan #19), by their own page string
PORTED = ("ev_status", "evc_nc_fail", "evc_cov", "evc_pro_iv", "evc_pro_dr", "evc_hyb",
          "evc_ring_ref", "evc_pile", "evc_pr_heads", "evc_pr_ch_head", "evc_pr_heads_size",
          "evc_pr_subs", "evc_pr_ch_subs", "evc_pr_ch_class", "evc_n2_rd", "ev_counts_long")
BAND = ("bn_head_pb", "cov_line", "pb_title", "pb_ribbon_h", "pb_wells", "pb_shared_h",
        "pb_range", "bn_head_fy_pb", "pb_lofo")
# how_sure carries no head-band line: the band is left to the page link (bn_head_pb says "the
# band below", pb_title is a heading; neither reads alone)
BAND_ANSWER: list[str] = []
# the lead's rulings on drawer lines kept out of answers (B1: D2; S1: spec §13)
NOT_IN_ANSWERS = {"evc_pr_ch_head", "evc_pr_ch_subs", "evc_pr_ch_class", "evc_ring",
                  "evc_ring_ref", "evc_pile", "evc_hyb", "evc_cov", "cov_line", "bn_aux"}
# every R8.13 key that carries a number is a ledger band key (B2)
PB_NUMERIC = {"bn_head_fy_pb", "pb_ribbon_h", "pb_ribbon_what", "pb_wells", "pb_small",
              "pb_shared_h", "pb_shared_what", "pb_range", "pb_lofo", "pb_lofo_c"}


def _page(tmp_path, *, pb=True, members=True, fy_agg=True, all30=False):
    from hydrophysics.twin.release import read_payload
    page = testing.build_synthetic_page(tmp_path, all30=all30)
    p = read_payload(page)
    p["modelcard"].update(MC)
    if not pb:
        p["modelcard"].pop("pb")
    if not fy_agg:
        p["modelcard"]["results"]["fv2fy"].pop("agg", None)
    if members:
        ny = len(p["meta"]["years"])
        p["memberFields"] = dict(p.get("memberFields") or {}, n=4,
                                 baseHeadBandYr=[[9.0 + 0.1 * q] * ny for q in range(5)])
    testing._write_payload(page, p)
    b = PageBundle.load(page, testing.synthetic_ledger(tmp_path, page), allow_pending=True)
    if "bn_head_fy_pb" not in b.T["en"]:
        pytest.skip("template without the round-8 lines")
    return page, b


def _first(keys):
    return [k.split("+")[0] for k in keys]


def _parts(keys):
    return {k for key in keys for k in key.split("+")}


def test_round8_drawer_lines_are_mirrored_in_the_page_order(tmp_path):
    from hydrophysics_analyst import query as Q
    _, b = _page(tmp_path)
    banned = Q.band_keys(b)
    for lang in ("en", "zh"):
        keys = [s.key for s in Q.evidence(b, lang, full=True).snippets]
        first = _first(keys)
        for k in PORTED:
            assert k in first, (lang, k)
        # the page's order: status before the drawer's tests, the counts last
        assert first.index("ev_status") < first.index("evc_nc_fail") < first.index("evc_cov")
        assert first.index("evc_cov") < first.index("evc_pro") < first.index("evc_pro_iv")
        assert first.index("evc_pro_iv") < first.index("evc_pro_dr") < first.index("evc_pro_e02c")
        assert first.index("evc_hyb") < first.index("evc_ring_ref") < first.index("evc_pile")
        assert first.index("evc_pile") < first.index("evc_pr_heads") < first.index("evc_n2_rd")
        assert first.index("evc_n2_rd") < first.index("evc_chx") < first.index("evc_subs_decomp")
        assert first[-1] == "ev_counts_long"
        # E25's two-contrast clause and E26's synthetic rates ride on their lines (R8.10, R8.8)
        assert "evc_chx+chx_2c_ns" in keys and "evc_chb+chb_syn" in keys
        # the answer layer's evidence drops every band-key line
        assert not banned & _parts(s.key for s in Q.evidence(b, lang).snippets)
        bk = _first(s.key for s in Q.head_band(b, lang, full=True).snippets)
        for k in BAND:
            assert k in bk, (lang, k)
        assert not banned & _parts(s.key for s in Q.head_band(b, lang).snippets)


def test_how_sure_answers_only_the_lines_the_rulings_allow(tmp_path):
    """B1 (D2): no Changhua-lever line; S1 (spec §13): no ring, pile, hybrid or coverage line;
    B2 and the final pass: no head-band line at all (the band is left to the page link)."""
    _, b = _page(tmp_path)
    for lang in ("en", "zh"):
        a = PR.run(b, "how_sure", {}, lang)
        assert validate(a, b) == []
        parts = _parts(s.key for s in a.spans if s.kind == "snippet")
        assert not parts & NOT_IN_ANSWERS, parts & NOT_IN_ANSWERS
        assert not parts & PB_NUMERIC
        band = [s.key for s in a.spans if s.key.split("+")[0].startswith(("bn_", "pb_", "cov_"))]
        assert band == BAND_ANSWER, band
        assert "tab=trust" in a.deep_link            # the page carries the band
        st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
        assert not any("W" in st[c] for c, _ in a.citations)
        for k in ("evc_pro", "evc_hyb_for", "evc_n2_rd", "evc_chx", "evc_pr_heads",
                  "ev_counts_long"):
            assert k in {s.key.split("+")[0] for s in a.spans}, k


def test_numeric_r8_13_keys_are_ledger_band_keys():
    from hydrophysics_analyst.page import LEDGER
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    banned = set(led["page_scoping"]["answer_must_not_use_band_keys"])
    assert banned >= PB_NUMERIC
    assert not set(BAND_ANSWER) & banned


def test_w_governed_lines_are_flagged_before_their_citations_are_stripped(tmp_path):
    """B1: the validator (and so the eval) judges a sentence on the claims of its own page
    strings, not on the W-stripped citations: the real Changhua-lever keys are flagged; a
    registered test of a withdrawn quantity beside other claims (evc_pile_n3, C09 W+D) is not."""
    import dataclasses as dc

    from hydrophysics_analyst import eval as EV
    from hydrophysics_analyst import query as Q
    from hydrophysics_analyst import validate as V
    from hydrophysics_analyst.render import Span
    _, b = _page(tmp_path)
    assert set(V.w_governed("evc_pr_ch_subs", b.ledger)) == {"C13c"}
    assert set(V.w_governed("evc_pr_ch_class", b.ledger)) == {"C13b"}
    assert set(V.w_governed("evc_pr_ch_head", b.ledger)) == {"C13a"}
    assert V.w_governed("evc_pile_n3", b.ledger) == ()
    assert V.w_governed("evc_hyb_for", b.ledger) == ()
    a = PR.run(b, "how_sure", {}, "en")
    full = {s.key.split("+")[0]: s for s in Q.evidence(b, "en", full=True).snippets}
    for k in ("evc_pr_ch_subs", "evc_pr_ch_class", "evc_pr_ch_head"):
        s = full[k]          # as how_sure would show it: its W citations stripped
        bad = dc.replace(a, spans=a.spans + (Span("snippet", s.text, key=s.key, claims=(),
                                                  block="evidence"),))
        v = validate(bad, b)
        assert any(x.rule == "withheld" and k in x.detail for x in v), (k, v)
        item = {"expect": {"preset": "how_sure", "params": {}, "answer": {"no_withheld": True}}}
        assert EV.score(b, item, bad)["unsafe"], k


def test_ev_counts_follow_test_rows(tmp_path):
    """``testCounts``: the synthetic page's rows are the new-data check (FAIL), the unseen-well
    level screen (``verdict`` PASS) and the N1 rows of the deliverable recipe (heads PASS,
    subsidence FAIL); the R1c rows are not the deliverable recipe and are not counted."""
    from hydrophysics_analyst import query as Q
    _, b = _page(tmp_path)
    c = Q.ctx(b)
    assert c.mc.get("temporal") is None and c.mc.get("r2_anom") is None
    assert c.mc.get("r2_kfold") is not None and c.mc.get("verdict") == "PASS"
    assert Q.test_counts(c) == (2, 2)
    txt = {s.key: s.text for s in Q.evidence(b, "en").snippets}
    assert txt["ev_counts_long"] == c.tr("en", "ev_counts_long", {"p": 2, "f": 2})


def test_e24_is_internals_only_and_never_in_an_answer(tmp_path):
    from hydrophysics_analyst import query as Q
    _, b = _page(tmp_path)
    c = Q.ctx(b)
    irr = Q.Policy(Q.IRR30[:c.K], 2026)
    assert [s.key.split("+")[0] for s in Q.e24_line(b, irr, "en").snippets] == ["def_tA_e24"]
    assert not Q.e24_line(b, Q.Policy((0.7,) * c.K, 2026), "en").snippets
    assert not Q.e24_line(b, Q.Policy(Q.IRR30[:c.K], 2030), "en").snippets
    for lang in ("en", "zh"):
        a = PR.run(b, "policy_head", {"cuts": [{"class": c.classes[0]["key"], "pct": 30}],
                                       "start": 2026, "year": c.years[-1]}, lang)
        assert not any("def_tA_e24" in s.key or "e24_" in s.key for s in a.spans)


def _chromium():
    sync = pytest.importorskip("playwright.sync_api")
    from hydrophysics.twin.browser_check import _exe
    pw = sync.sync_playwright().start()
    try:
        return pw, pw.chromium.launch(**_exe())
    except Exception as e:  # noqa: BLE001
        pw.stop()
        pytest.skip(f"no Chromium for Playwright: {type(e).__name__}")


def _open(browser, page):
    pg = browser.new_page()
    pg.goto(page.as_uri())
    pg.wait_for_function("document.body.dataset.ready === '1'", timeout=30000)
    return pg


def _default(pg, b, lang, n):
    from hydrophysics_analyst import parity as PA
    from hydrophysics_analyst import query as Q
    c = Q.ctx(b)
    pg.evaluate(PA._SET_HASH, Q.deep_link(b, Q.Policy((1.0,) * c.K, c.starts[0]), c.years[-1],
                                          lang=lang, nonce=n))


def test_round8_port_equals_the_page_line_by_line_and_claim_by_claim(tmp_path):
    """In headless Chromium at the default view: the evidence line and tile reading, the
    drawer and the head band are the port's line by line, both ways, text and claims; E24
    equals ``e24Line()`` on its one policy and is empty on another; and each check catches a
    dropped line, a wrong claim, a dropped zone label, a dropped lofo line, a truncated range."""
    from hydrophysics_analyst import parity as PA
    from hydrophysics_analyst import query as Q
    page, b = _page(tmp_path)
    c = Q.ctx(b)
    pw, browser = _chromium()
    try:
        pg = _open(browser, page)
        n = 0
        for lang in ("en", "zh"):
            n += 1
            _default(pg, b, lang, n)
            ev = Q.evidence(b, lang, full=True).snippets
            assert PA._evidence_items(b, ev, pg.evaluate(PA._EV_ITEMS), lang) == []
            assert PA._top_items(b, ev, pg.evaluate(PA._TOP), lang) == []
            band = Q.head_band(b, lang, full=True).snippets
            assert PA._band_items(b, band, pg.evaluate(PA._BAND), lang) == []
            for pol, want in ((Q.Policy(Q.IRR30[:c.K], 2026), True),
                              (Q.Policy((0.7,) * c.K, 2026), False)):
                n += 1
                pg.evaluate(PA._SET_HASH, Q.deep_link(b, pol, c.years[-1], lang=lang, nonce=n,
                                                      mi=True))
                js = pg.evaluate("() => e24Line()")
                assert js == "".join(s.text for s in Q.e24_line(b, pol, lang).snippets)
                assert bool(js) is want
        # the checks are not vacuous
        n += 1
        _default(pg, b, "en", n)
        ev = list(Q.evidence(b, "en", full=True).snippets)
        items = pg.evaluate(PA._EV_ITEMS)
        drop = next(i for i, s in enumerate(ev) if s.key.startswith("evc_n2_rd"))
        got = PA._evidence_items(b, ev[:drop] + ev[drop + 1:], items, "en")
        assert [m["what"] for m in got] == ["evidence_missing"]
        dec = next(s for s in ev if s.key.startswith("evc_subs_decomp+"))   # carries C31
        bad = [Q.Snippet("evc_subs_decomp", s.lang, s.text) if s is dec else s for s in ev]
        assert [m["what"] for m in PA._evidence_items(b, bad, items, "en")] == ["evidence_claims"]
        assert [m["what"] for m in PA._evidence_items(b, ev + [ev[3]], items, "en")] == [
            "evidence_extra"]
        top = pg.evaluate(PA._TOP)
        assert [m["what"] for m in PA._top_items(b, ev[1:], top, "en")] == ["top_missing"]
        keyed = pg.evaluate(PA._BAND)
        band = list(Q.head_band(b, "en", full=True).snippets)
        zone = next(i for i, s in enumerate(band) if s.key == "pb_wells")
        assert [m["what"] for m in PA._band_items(b, band[:zone] + band[zone + 1:], keyed, "en")
                ] == ["head_band_missing"]
        lofo = [s for s in band if not s.key.startswith("pb_lofo")]
        assert [m["what"] for m in PA._band_items(b, lofo, keyed, "en")] == ["head_band_missing"]
        rng = [Q.Snippet(s.key, s.lang, s.text[:-3]) if s.key == "pb_range" else s for s in band]
        got = [m["what"] for m in PA._band_items(b, rng, keyed, "en")]
        assert got.count("head_band_missing") == 2 and got.count("head_band_extra") == 2
    finally:
        browser.close()
        pw.stop()


@pytest.mark.parametrize("variant,first", [({"pb": False}, "bn_head"),
                                           ({"pb": False, "members": False}, "bn_head_none"),
                                           ({"pb": False, "fy_agg": True}, "bn_head_fy")])
def test_head_band_without_pb_equals_the_page(tmp_path, variant, first):
    """The PB-absent branches of ``seriesFor``'s band note: the run-spread note (bn_head), the
    one with FV v2's forcing-year range (bn_head_fy, a band key: mirrored, never answered) and
    none (bn_head_none), both ways against the page, with the coverage line when a band shows."""
    from hydrophysics_analyst import parity as PA
    from hydrophysics_analyst import query as Q
    v = {"fy_agg": False} | variant
    page, b = _page(tmp_path, **v)
    band = Q.head_band(b, "en", full=True).snippets
    assert band[0].key.split("+")[0] == first
    assert ("cov_line" in _first(s.key for s in band)) is (first != "bn_head_none")
    shown = _parts(s.key for s in PR.run(b, "how_sure", {}, "en").spans)
    assert not {"bn_head_fy", "cov_line"} & shown
    pw, browser = _chromium()
    try:
        pg = _open(browser, page)
        for i, lang in enumerate(("en", "zh")):
            _default(pg, b, lang, i + 1)
            assert PA._band_items(b, Q.head_band(b, lang, full=True).snippets,
                                  pg.evaluate(PA._BAND), lang) == []
    finally:
        browser.close()
        pw.stop()


def test_default_view_tile_reading_is_the_all30_line(tmp_path):
    """With the all30 run (R8.9) the page's default view shows tW_w30 under the head tile, and
    how_sure carries that line (with its claims), both ways against the page."""
    from hydrophysics_analyst import parity as PA
    from hydrophysics_analyst import query as Q
    page, b = _page(tmp_path, all30=True)
    a = PR.run(b, "how_sure", {}, "en")
    assert "tW_w30" in {s.key for s in a.spans} and "tW_r" not in {s.key for s in a.spans}
    assert validate(a, b) == []
    pw, browser = _chromium()
    try:
        pg = _open(browser, page)
        for i, lang in enumerate(("en", "zh")):
            _default(pg, b, lang, i + 1)
            assert PA._top_items(b, Q.evidence(b, lang, full=True).snippets,
                                 pg.evaluate(PA._TOP), lang) == []
    finally:
        browser.close()
        pw.stop()


def test_page_pins_name_the_ledger_page():
    """The usage pins (cli, page, server, mcp_server, parity, bundle) name the page the ledger
    binds (``page.html``), so they cannot drift from the binding again."""
    import re

    import hydrophysics_analyst as HA
    from hydrophysics_analyst.page import LEDGER
    led = json.loads(LEDGER.read_text(encoding="utf-8"))
    want = Path(led["page"]["html"]).name
    pkg = Path(HA.__file__).parent
    for mod in ("cli", "page", "server", "mcp_server", "parity", "bundle"):
        head = "".join((pkg / f"{mod}.py").read_text(encoding="utf-8").splitlines(True)[:6])
        got = re.findall(r"twin_app_\d{8}[a-z]?(?:\.local)?\.html", head)
        assert got and {g.replace(".local", "") for g in got} == {want}, (mod, got, want)


@pytest.mark.skipif(not os.environ.get("ANALYST_PAGE"), reason="ANALYST_PAGE not set")
def test_real_page_binds_strictly_and_how_sure_validates():
    """The final round-8 page binds in deploy mode (forward and template pins of the ledger) and
    how_sure on it validates in both languages, citing no W claim."""
    b = PageBundle.load(Path(os.environ["ANALYST_PAGE"]), mode="deploy")
    st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
    for lang in ("en", "zh"):
        a = PR.run(b, "how_sure", {}, lang)
        assert validate(a, b) == []
        assert not any("W" in st[c] for c, _ in a.citations)
