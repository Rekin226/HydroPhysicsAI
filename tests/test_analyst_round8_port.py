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
# the round-8 page lines the port now carries (plan #19), by their own page string
PORTED = ("ev_status", "evc_nc_fail", "evc_cov", "evc_hyb", "evc_ring_ref", "evc_pile",
          "evc_pr_heads", "evc_pr_ch_head", "evc_pr_heads_size", "evc_pr_subs", "evc_pr_ch_subs",
          "evc_pr_ch_class", "evc_n2_rd", "ev_counts_long")
BAND = ("bn_head_pb", "cov_line", "pb_title", "pb_ribbon_h", "pb_wells", "pb_shared_h",
        "pb_range", "bn_head_fy_pb", "pb_lofo")


def _page(tmp_path):
    from hydrophysics.twin.release import read_payload
    page = testing.build_synthetic_page(tmp_path)
    p = read_payload(page)
    p["modelcard"].update(MC)
    p["memberFields"] = dict(p.get("memberFields") or {}, n=4, baseHeadBandYr=[[0.0]] * 5)
    testing._write_payload(page, p)
    b = PageBundle.load(page, testing.synthetic_ledger(tmp_path, page), allow_pending=True)
    if "bn_head_fy_pb" not in b.T["en"]:
        pytest.skip("template without the round-8 lines")
    return page, b


def _first(keys):
    return [k.split("+")[0] for k in keys]


def test_round8_drawer_lines_are_ported_in_the_page_order(tmp_path):
    from hydrophysics_analyst import query as Q
    _, b = _page(tmp_path)
    banned = set(b.ledger["page_scoping"]["answer_must_not_use_band_keys"])
    for lang in ("en", "zh"):
        keys = [s.key for s in Q.evidence(b, lang).snippets]
        first = _first(keys)
        for k in PORTED:
            assert k in first, (lang, k)
        # the page's order: status before the drawer's tests, the counts last
        assert first.index("ev_status") < first.index("evc_nc_fail") < first.index("evc_cov")
        assert first.index("evc_cov") < first.index("evc_pro") < first.index("evc_hyb")
        assert first.index("evc_hyb") < first.index("evc_ring_ref") < first.index("evc_pile")
        assert first.index("evc_pile") < first.index("evc_pr_heads") < first.index("evc_n2_rd")
        assert first.index("evc_n2_rd") < first.index("evc_chx") < first.index("evc_subs_decomp")
        assert first[-1] == "ev_counts_long"
        # E25's two-contrast clause and E26's synthetic rates ride on their lines (R8.10, R8.8)
        assert "evc_chx+chx_2c_ns" in keys and "evc_chb+chb_syn" in keys
        assert not banned & {k for key in keys for k in key.split("+")}
        bk = _first(s.key for s in Q.head_band(b, lang).snippets)
        for k in BAND:
            assert k in bk, (lang, k)
        assert "bn_head_fy" not in bk
        a = PR.run(b, "how_sure", {}, lang)
        assert validate(a, b) == []
        st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
        assert not any("W" in st[c] for c, _ in a.citations)


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


def test_round8_port_equals_the_page_line_by_line_and_claim_by_claim(tmp_path):
    """In headless Chromium: every drawer line that is not a band-key line is the port's, with
    the same text and the same claims (and nothing else is); the band lines are in the head
    series' notes; E24 equals ``e24Line()`` on its one policy and is empty on another."""
    from hydrophysics_analyst import binding as BD
    from hydrophysics_analyst import parity as PA
    from hydrophysics_analyst import query as Q
    from hydrophysics_analyst.jsport import strip_tags
    page, b = _page(tmp_path)
    c = Q.ctx(b)
    pw, browser = _chromium()
    try:
        pg = browser.new_page()
        pg.goto(page.as_uri())
        pg.wait_for_function("document.body.dataset.ready === '1'", timeout=30000)
        n = 0
        for lang in ("en", "zh"):
            n += 1
            pg.evaluate(PA._SET_HASH, Q.deep_link(b, Q.Policy((1.0,) * c.K, c.starts[0]),
                                                  c.years[-1], lang=lang, nonce=n))
            ev = Q.evidence(b, lang).snippets
            items = pg.evaluate(PA._EV_ITEMS)
            assert PA._evidence_items(b, ev, items, lang) == []
            # the claim set: the page's drawer (band-key lines aside) is the port's, and the port
            # adds only the head tile reading's claims
            dm = b.ledger["display_map"]["i18n"]
            banned = set(b.ledger["page_scoping"]["answer_must_not_use_band_keys"])
            page_claims = {cl for _, ks in items if not banned & set(ks) for k in ks
                           for cl in dm.get(k, [])}
            mine = {cl for s in ev if s.key.split("+")[0] not in PA.NOT_DRAWER
                    for cl in BD.snippet_claims(s.key, b.ledger)}
            assert mine == page_claims, (sorted(mine ^ page_claims))
            band = pg.evaluate(PA._BAND)
            for s in Q.head_band(b, lang).snippets:
                assert strip_tags(s.text) in band, (lang, s.key)
            for pol, want in ((Q.Policy(Q.IRR30[:c.K], 2026), True),
                              (Q.Policy((0.7,) * c.K, 2026), False)):
                n += 1
                pg.evaluate(PA._SET_HASH, Q.deep_link(b, pol, c.years[-1], lang=lang, nonce=n,
                                                      mi=True))
                js = pg.evaluate("() => e24Line()")
                assert js == "".join(s.text for s in Q.e24_line(b, pol, lang).snippets)
                assert bool(js) is want
        # the check is not vacuous: a dropped line and a wrong claim are both caught
        ev = list(Q.evidence(b, "en").snippets)
        pg.evaluate("() => { LANG = 'en'; }")
        items = pg.evaluate(PA._EV_ITEMS)
        drop = next(i for i, s in enumerate(ev) if s.key.startswith("evc_n2_rd"))
        got = PA._evidence_items(b, ev[:drop] + ev[drop + 1:], items, "en")
        assert [m["what"] for m in got] == ["evidence_missing"]
        dec = next(s for s in ev if s.key.startswith("evc_subs_decomp+"))   # carries C31
        bad = [Q.Snippet("evc_subs_decomp", s.lang, s.text) if s is dec else s for s in ev]
        got = PA._evidence_items(b, bad, items, "en")
        assert [m["what"] for m in got] == ["evidence_claims"]
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
