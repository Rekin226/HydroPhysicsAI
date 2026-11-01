"""The claims ledger (hydrophysics/twin/claims_v1.json) against the page template, the docs and,
where the gitignored files exist, the pre-registration index and the built page.

What counts as "a number the page displays" is enumerated from the code, not by hand:
  1. every dictionary key whose EN or ZH string shows a number ({placeholder} or a digit);
  2. every top-level JS function that formats a number or interpolates a payload value;
  3. every model-card detail row (kv_*), whose numbers are written inline;
  4. every tile id;
  5. every packed field array and every numeric payload leaf (key paths only: the payload is
     parsed with each number replaced by a placeholder, so no value is ever read).
Each must map to a ledger id, and no id may be U (unsupported) or X (inconsistent with the
record). The scoping gate reads the TEMPLATE, not the ledger: the answer row and the headline
must not carry a withdrawn, unsupported or inconsistent claim.

CPU only, no torch/scipy. Registry and page checks skip cleanly when those gitignored files are
absent (CI). While any cited registration's hash is still PENDING, exactly one test fails
loudly with claims.load_ledger's own message (test_no_cited_registration_is_pending); every
other test reads the ledger with allow_pending=True, so a real regression stays visible.
"""
from __future__ import annotations

import json
import re

import pytest

from hydrophysics.twin import claims as C

INDEX = C.PREREG / "INDEX.jsonl"
NEED_REGISTRY = pytest.mark.skipif(not INDEX.exists(), reason="the pre-registration index "
                                   "(results/dataprep/preregistration) is gitignored, absent here")
BAD = {"U", "X"}
_CACHE: dict = {}


def L() -> dict:
    """The ledger, read past the PENDING guard. The guard itself is asserted once, loudly, by
    test_no_cited_registration_is_pending; reading past it here keeps the other checks live
    while registrations are pending."""
    if "ledger" not in _CACHE:
        _CACHE["ledger"] = C.load_ledger(allow_pending=True)
    return _CACHE["ledger"]


def CL() -> dict:
    return {c["id"]: c for c in L()["claims"]}


def DM() -> dict:
    return L()["display_map"]


def _page():
    page = C.REPO / L()["page"]["html"]
    if not page.exists():
        pytest.skip(f"{L()['page']['html']} is a gitignored build, absent here")
    return page


def _st(cid: str) -> set[str]:
    return set(C.codes(CL()[cid]["status_current"]))


# ---- the ledger itself -----------------------------------------------------------------------
def test_ids_unique_and_status_codes_valid():
    ids = [c["id"] for c in L()["claims"]]
    assert len(ids) == len(set(ids))
    for c in L()["claims"]:
        for col in ("status_before", "status_current"):
            parts = C.codes(c[col])
            assert parts and all(p in C.CODES for p in parts), (c["id"], col, c[col])
        assert c["wording_en"].strip(), c["id"]
        assert c["text_keys"], f"{c['id']} has no displayed text key"


def test_no_claim_is_unsupported_or_inconsistent_with_the_record():
    bad = sorted(c["id"] for c in L()["claims"] if set(C.codes(c["status_current"])) & BAD)
    assert not bad, f"claims still U or X: {bad}"


def test_a_status_change_is_a_wording_change():
    for c in L()["claims"]:
        if c["status_before"] != c["status_current"] or set(C.codes(c["status_before"])) & BAD:
            assert c["changes"], f"{c['id']} changed status but records no wording change"
            assert not c["wording_en"].startswith("Unchanged"), c["id"]
        if not c["changes"]:
            assert c["wording_en"].startswith(("Unchanged", "Not a claim", "Not displayed")), c["id"]


def test_status_changes_rest_on_a_user_ruling():
    """Every status change (not only a withdrawal) cites a recorded ruling."""
    rulings = {r["id"] for r in L()["rulings"]}
    for c in L()["claims"]:
        if c["status_before"] != c["status_current"]:
            assert c["ruling"] in rulings, f"{c['id']} changed status without a ruling"
        for r in (c.get("ruling"), c.get("ruling_also")):
            assert r is None or r in rulings, (c["id"], r)


def test_supported_and_decided_claims_cite_their_registration():
    for c in L()["claims"]:
        st, ci = _st(c["id"]), c["cites"]
        if "S" in st:
            assert ci["tests"], f"{c['id']} is S but cites no registered test id"
        if "D" in st:
            assert ci["registrations"] or ci["manifests"], f"{c['id']} is D but cites nothing"


def test_ledger_is_safe_to_commit():
    text = C.LEDGER.read_text(encoding="utf-8")
    assert not re.search(r"/home/|/Users/|/tmp/", text)
    assert not re.search(r"(\d{1,3}\.){3}\d{1,3}", text.replace("0.0048-0.0071", ""))


@NEED_REGISTRY
def test_citations_exist_in_the_registry_and_are_unchanged():
    idx, tests = C.prereg_index()
    hashes = L()["registry_hashes"]
    pend = set(C.pending_registrations(L()))   # failed once, loudly, by the PENDING test
    for c in L()["claims"]:
        ci = c["cites"]
        for r in ci["registrations"]:
            if r in pend:
                continue
            assert r in idx, (c["id"], r)
            assert (C.PREREG / f"{r}.json").exists(), (c["id"], r)
            assert hashes[r] == idx[r]["sha256"], f"{r} changed since the ledger was built"
        for t in ci["tests"]:
            assert t in tests, (c["id"], t)
        for m in ci["manifests"]:
            assert C.sha256(C.REPO / m) == hashes[m], f"manifest {m} changed"
        for p in ci["reports"]:
            assert (C.REPO / p).exists(), (c["id"], p)


@NEED_REGISTRY
def test_the_head_claim_is_decided_on_the_page_model():
    n2 = json.loads((C.PREREG / "N2.policy_response.prospective_2026-11.v1.json").read_text())
    assert n2["primary_model"]["forward_sha256"] == L()["page"]["n2_primary_forward_sha256"]
    assert L()["page"]["forward_sha256"] == L()["page"]["n2_primary_forward_sha256"]


# ---- every display site maps to a ledger id --------------------------------------------------
def _ids_ok(where: str, mapping: dict):
    bad = {k: v for k, v in mapping.items() if not v or any(i not in CL() for i in v)}
    assert not bad, f"{where}: unknown or empty ids {bad}"


def test_ledger_is_pinned_to_this_template():
    assert C.sha256(C.TEMPLATE) == L()["page"]["template_sha256"], (
        "the page template changed: re-review the claims ledger and update template_sha256")


def test_every_numeric_dictionary_key_is_mapped():
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    numeric, m = C.displayed_numeric_keys(), DM()["i18n"]
    assert not sorted(numeric - set(m)), f"unmapped numeric keys: {sorted(numeric - set(m))}"
    assert not sorted(set(m) - numeric), f"stale keys in the ledger: {sorted(set(m) - numeric)}"
    assert set(m) <= set(zh) and set(m) <= set(en), "EN/ZH key parity broken for mapped keys"
    _ids_ok("i18n", m)


def test_every_number_formatting_function_is_mapped():
    fns, m = C.number_functions(), DM()["functions"]
    assert not sorted(set(fns) - set(m)), f"unmapped functions: {sorted(set(fns) - set(m))}"
    assert not sorted(set(m) - set(fns)), f"stale functions: {sorted(set(m) - set(fns))}"
    _ids_ok("functions", m)


def test_every_model_card_row_and_tile_is_mapped():
    rows = C.card_rows()
    assert not sorted(rows - set(DM()["card_rows"])), sorted(rows - set(DM()["card_rows"]))
    ids, _ = C.tiles()
    assert set(ids) == set(DM()["tiles"]), (ids, sorted(DM()["tiles"]))
    _ids_ok("card_rows", DM()["card_rows"])
    _ids_ok("tiles", DM()["tiles"])


def test_mapped_dictionary_keys_are_used_by_the_page():
    src = C.template_src()
    t0 = src.index("const T = {")
    body = src[:t0] + src[src.index("\n}};", t0):]          # markup + code, dictionary removed
    computed = ("ly_", "pl_", "pr_")                         # tr("ly_" + id), tr("pl_" + key), ...
    for p in computed:
        assert f'"{p}"' in body, f"computed prefix {p} no longer used"
    unused = [k for k in DM()["i18n"]
              if not re.search(r'"' + re.escape(k) + r'"', body) and not k.startswith(computed)]
    assert not unused, f"mapped keys the page never asks for: {unused}"


# ---- the scoping gate (reads the template) ---------------------------------------------------
def test_the_answer_row_carries_no_withdrawn_or_unsupported_claim():
    ids, primary = C.tiles()
    allowed = set(L()["page_scoping"]["answer_tiles_allowed"])   # L: the labelled cost tile
    assert not allowed & {"W", "U", "X"}, allowed
    for tid in ids[:primary]:
        for cid in DM()["tiles"][tid]:
            assert _st(cid) <= allowed, f"answer tile {tid} rests on {cid} ({CL()[cid]['status_current']})"
    for tid in ids[primary:]:
        assert any(_st(c) & {"W", "L"} for c in DM()["tiles"][tid]), tid
    # every withdrawn claim that owns a tile sits in the model-internals row
    for cid in CL():
        if "W" in _st(cid):
            for tid, cl in DM()["tiles"].items():
                if cid in cl:
                    assert ids.index(tid) >= primary, f"withdrawn {cid} is in the answer row ({tid})"


def test_the_headline_uses_no_withdrawn_or_unsupported_claim():
    """The answer functions (headline, evidence line, start note, policy bar), with every
    top-level function they call, ask for no key of a W/U/X claim."""
    sc = L()["page_scoping"]
    computed = sc["answer_computed_keys"]
    for fn in sc["answer_functions"]:
        keys = C.function_keys(fn, follow=True)
        for expr, ks in computed.items():
            if expr in C.computed_tr_calls(C.function_source(fn, follow=True)):
                keys |= set(ks)
        for k in keys:
            for cid in DM()["i18n"].get(k, []):
                assert not _st(cid) & {"W", "U", "X"}, f"{fn} shows {k} -> {cid}"
        bad = keys & set(sc["headline_must_not_use"])
        assert not bad, (fn, bad)


def test_answer_functions_have_no_unreadable_keys():
    """The documented limit of the key scan: a tr() key built at run time (a lookup table or a
    concatenation) is invisible to claims.tr_keys. Each such call in the answer functions must
    be listed in page_scoping.answer_computed_keys with the keys it can take."""
    sc = L()["page_scoping"]
    for fn in sc["answer_functions"]:
        for expr in C.computed_tr_calls(C.function_source(fn, follow=True)):
            assert expr in sc["answer_computed_keys"], f"{fn}: unlisted run-time key tr({expr})"
    en = C.i18n(lang="en")
    for ks in sc["answer_computed_keys"].values():
        assert set(ks) <= set(en), ks


def test_answer_functions_do_not_read_withdrawn_quantities():
    sc = L()["page_scoping"]
    for fn in sc["answer_functions"]:
        body = C.function_source(fn)
        for name in sc["answer_must_not_read"]:
            assert not re.search(r"(?<![A-Za-z0-9_$])" + re.escape(name) + r"(?![A-Za-z0-9_$])",
                                 body), f"{fn} reads {name}"


def test_key_scan_sees_chosen_and_called_keys():
    """Regression for the round-3 review: ternary-chosen keys and callees are seen."""
    assert {"ev_nc_pass", "ev_nc_fail"} <= C.function_keys("evLineText")
    assert {"less_metrics", "more_metrics"} <= C.function_keys("tileLabels")
    assert "hw_n2" not in C.function_keys("renderHeadline")
    assert {"hw_n2", "hw_yun", "hw_fan"} <= C.function_keys("renderHeadline", follow=True)
    assert C.computed_tr_calls('tr("ly_" + id); tr(a ? "x" : "y"); tr(t)') == ['"ly_" + id', "t"]


def test_withdrawn_numbers_outside_the_tiles_carry_the_label():
    """Keys that show a withdrawn quantity outside the internals tile row say so."""
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    for k in L()["page_scoping"]["internals_labelled_keys"]:
        assert "Model internals" in en[k], k
        assert "模型內部" in zh[k], k


def test_template_anchors_hold():
    """Ruling (4): 'no calibrated correction factor follows' stays in the head definition."""
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    for k, rule in L()["page_scoping"]["template_anchors"].items():
        assert rule["en"] in en[k], (k, rule["en"])
        assert rule["zh"] in zh[k], (k, rule["zh"])


def test_the_page_opens_on_the_tested_quantity():
    js = C.template_js()
    sc = L()["page_scoping"]
    assert re.search(r'const S = \{[^}]*layer: "' + sc["default_layer"] + '"', js)
    assert re.search(r'\bts: "' + sc["default_series"] + '"', js)
    en = C.i18n(lang="en")
    assert "Model internals" in en["more_metrics"] and en["mi_note"].startswith("Model internals")


def test_no_registered_beforehand_wording_remains():
    src = C.template_src()
    for bad in ("registered beforehand", "pre-registered", "preregistered", "事先登錄"):
        assert bad not in src, bad
    # and the round-3 overclaims: N2 does not decide the tile's size; the start note is no
    # longer in fan-average cm
    for bad in L()["page_scoping"]["template_absent"]:
        assert bad not in src, bad
    for p in ("hydrophysics/twin/viewer_app.py", "hydrophysics/twin/kfold_scores.py"):
        assert "pre-registered 2026-09-2" not in (C.REPO / p).read_text(encoding="utf-8"), p


def test_doc_anchors_hold():
    for c in L()["claims"]:
        for doc, rule in c.get("doc_anchors", {}).items():
            text = (C.REPO / doc).read_text(encoding="utf-8")
            for s in rule.get("present", []):
                assert s in text, f"{c['id']}: {doc} lost {s!r}"
            for s in rule.get("absent", []):
                assert s not in text, f"{c['id']}: {doc} still says {s!r}"


# ---- the built page (gitignored) ------------------------------------------------------------
def test_every_payload_array_is_mapped():
    D = C.payload_structure(_page())
    pats = DM()["arrays"]
    _ids_ok("arrays", pats)
    for k in D["arrays"]:
        assert any(re.fullmatch(re.escape(p).replace(r"\*", ".*"), k) for p in pats), k


def test_every_numeric_payload_path_is_mapped_exactly():
    D = C.payload_structure(_page())
    paths = {p for p in C.numeric_paths(D) if not p.startswith("arrays.")}
    m = DM()["payload"]
    assert not sorted(paths - set(m)), f"payload numbers with no ledger id: {sorted(paths - set(m))}"
    assert not sorted(set(m) - paths), f"stale payload paths: {sorted(set(m) - paths)}"
    _ids_ok("payload", m)
    js = C.template_js()
    named = [p for p, v in m.items() if v == ["Z04"] and C.js_names_leaf(js, p)]
    assert not named, f"paths marked 'not displayed' that the page code names: {named}"


def test_page_is_built_from_the_pinned_forward_run():
    D = C.payload_structure(_page())
    assert D["release"]["sources"]["forward"]["sha256"] == L()["page"]["forward_sha256"]


# ---- round 5 (user rulings 2026-10-29, third set) --------------------------------------------
N2C = "N2C.changhua_response.prospective_2026-11.v1"
U1 = "U1.no_band.v1"


def test_no_cited_registration_is_pending():
    """Fails loudly, with the guard's own message, until every registration the page cites is
    in the registry and its hash replaces PENDING."""
    try:
        C.load_ledger()
    except C.PendingRegistrationError as exc:
        pytest.fail(str(exc), pytrace=False)


def test_pending_guard_structure(tmp_path):
    """The ledger's structure while registrations are pending, and the guard itself on a tmp
    copy (no registry is read or written)."""
    raw = C.load_ledger(allow_pending=True)
    hashes, pend = raw["registry_hashes"], C.pending_registrations(raw)
    cited = {r for c in raw["claims"] for r in c["cites"]["registrations"]}
    assert set(pend) <= cited, f"PENDING hashes nobody cites: {sorted(set(pend) - cited)}"
    for r in cited:
        assert r in hashes, f"{r} is cited but has no registry_hashes entry"
        assert hashes[r] == C.PENDING or re.fullmatch(r"[0-9a-f]{64}", hashes[r]), (r, hashes[r])
    for c in raw["claims"]:             # a pending registration can only back a pending claim
        if set(c["cites"]["registrations"]) & set(pend):
            assert set(C.codes(c["status_current"])) & {"D", "N", "L", "W"}, c["id"]
    # the guard: refuses a pending copy, names every pending id, accepts it once filled
    path = tmp_path / "claims.json"
    path.write_text(json.dumps({"registry_hashes": {"A.x.v1": "PENDING", "B.y.v1": "f" * 64}}))
    with pytest.raises(C.PendingRegistrationError, match=r"A\.x\.v1.*INDEX\.jsonl"):
        C.load_ledger(path)
    assert C.load_ledger(path, allow_pending=True)["registry_hashes"]["A.x.v1"] == "PENDING"
    path.write_text(json.dumps({"registry_hashes": {"A.x.v1": "e" * 64}}))
    assert C.pending_registrations(C.load_ledger(path)) == []


def test_c13_is_split_into_head_class_and_subsidence_claims():
    cl = CL()
    assert "C13" not in cl and {"C13a", "C13b", "C13c"} <= set(cl)
    for cid in ("C13a", "C13b", "C13c"):            # withdrawn; never a bare D on a lever
        assert _st(cid) == {"W"}, (cid, cl[cid]["status_current"])
    assert N2C in cl["C13a"]["cites"]["registrations"]
    table = cl["C13a"]["outcome_display"]
    for k in ("before_scoring", "response_supported", "direction_supported_size_not",
              "not_supported", "not_informative", "session_order_breached", "never"):
        assert table.get(k), k
    assert "all-class" in table["response_supported"] and "C13b and C13c stay W" in table["never"]
    e20 = cl["E20"]                                   # the pending decision is an evidence row
    assert _st("E20") == {"D"} and e20["cites"]["registrations"] == [N2C]
    assert re.fullmatch(r"[0-9a-f]{64}", e20["units_registry_sha256"])
    assert "X5.lever_class_share.v1" in cl["C13b"]["cites"]["registrations"]
    assert N2C in cl["C13c"]["cites"]["registrations"]


def test_c14_is_split_and_every_band_claim_cites_its_record():
    cl = CL()
    subs = [f"C14{x}" for x in "abcde"]
    assert "C14" not in cl and set(subs) <= set(cl)
    for cid in subs:
        st = _st(cid)
        assert st and st <= set("SDNLW"), (cid, st)
        ci = cl[cid]["cites"]
        assert U1 in ci["registrations"] or ci["tests"], f"{cid} cites neither {U1} nor a test"
    assert _st("C14d") == {"L"} and "W" in _st("C14e") and "D" in _st("C14a")
    assert "E02c.prospective_2026-11.v1" in cl["C14a"]["cites"]["registrations"]
    assert "E02c.prospective_2026-11.v1" in cl["E21"]["cites"]["registrations"]
    assert cl["E22"]["cites"]["registrations"] and _st("E22") == {"D"}


def test_no_band_text_in_an_answer_function():
    sc = L()["page_scoping"]
    banned = set(sc["answer_must_not_use_band_keys"])
    for fn in sc["answer_functions"]:
        bad = C.function_keys(fn, follow=True) & banned
        assert not bad, f"{fn} shows band text {sorted(bad)}"
    ids, primary = C.tiles()
    for tid in ids[:primary]:                         # no answer tile rests on a band claim
        assert not set(DM()["tiles"][tid]) & {"C14c", "C14d", "C14e"}, tid


def test_changhua_lever_mentions_carry_the_label_en_zh():
    """N2C page_mapping.before_scoring: every page string (EN and ZH) and every sentence of the
    README, the MODEL_CARD and the whole evidence update that mentions a Changhua lever carries
    the N2C untested label, the C13b/C13c withdrawal wording or C02's spillover label."""
    rule = L()["page_scoping"]["changhua_lever_rule"]
    for lang in ("en", "zh"):
        d = C.i18n(lang=lang)
        bad = {k: C.lever_mentions(s, rule, lang) for k, s in d.items()
               if C.lever_mentions(s, rule, lang)}
        assert not bad, f"{lang}: unlabelled Changhua-lever strings {bad}"
        for k in ("evc_pr_ch_head", "pr_ch_untested", "pol_caveat_ch", "pol_caveat_ch_np"):
            assert rule["labels"][lang][0] in d[k], (lang, k)
    assert "docs/TWIN_EVIDENCE_2026-10.md" in rule["docs"], "the evidence update is checked whole"
    for doc in rule["docs"]:
        text = (C.REPO / doc).read_text(encoding="utf-8")
        assert rule["labels"]["en"][0] in text, f"{doc} never carries the N2C label"
        bad = C.lever_mentions(text, rule)
        assert not bad, f"{doc}: unlabelled Changhua-lever mentions {bad}"


def test_lever_mentions_checks_each_sentence():
    """Regression for the round-5 review: one labelled mention does not cover an unlabelled one
    in the same paragraph; a decimal ('x0.7') does not end a sentence; 'cut' names a lever."""
    rule = L()["page_scoping"]["changhua_lever_rule"]
    lab = rule["labels"]["en"][0]
    para = (f"The Changhua head lever is {lab}. A Changhua class-specific lever\n"
            "would differ x1.39 between conversions.\n\nNothing about levers here.")
    assert C.lever_mentions(para, rule) == [
        "A Changhua class-specific lever would differ x1.39 between conversions."]
    assert C.lever_mentions(f"A x0.7 Changhua cut raises heads; {lab}.", rule) == []
    assert C.lever_mentions("A x0.7 Changhua cut raises heads.", rule)
    assert C.lever_mentions("- Yunlin only.\n- Changhua levers: shown.", rule) == [
        "- Changhua levers: shown."]
    zlab = rule["labels"]["zh"][0]
    assert C.lever_mentions(f"彰化水位槓桿：{zlab}。彰化分類槓桿：顯示。", rule, "zh") == [
        "彰化分類槓桿：顯示。"]


def test_the_e02_interval_carries_its_u7_disclosure_en_zh():
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert "chosen after its 2023–26 coverage was seen" in en["evc_pro_iv"]
    assert "window is untouched" in en["evc_pro_iv"]
    assert "看過其 2023–26 年涵蓋率之後選定" in zh["evc_pro_iv"] and "未被觸及" in zh["evc_pro_iv"]
    js = C.template_js()
    assert "q.interval_bound" in js, "the interval line must wait for the build's hash binding"
    for k in ("evc_pro_e02c", "evc_pro_ext"):
        assert "2028-01-15" in en[k] and "2028-01-15" in zh[k], k


def test_the_paired_head_spread_is_drawn_only_under_model_internals():
    body = C.function_source("renderTS")
    assert re.search(r"const headOff = isHead && !S\.more", body)
    assert 'tr("dn_head_off")' in body
    en = C.i18n(lang="en")
    for k in ("dn_yr", "dn_con"):
        assert en[k].startswith("Model internals") and "excludes response-size error" in en[k]
    assert 'invalidate("ts")' in C.template_js().split('$("moreBtn").onclick', 1)[1].split("\n")[0]


def test_every_subsidence_band_note_says_model_internals():
    """Ruling 2026-10-29 third set (d): subsidence bands are permanently Model internals. Every
    band note of the subsidence series and of the difference panel (C14e's band keys) is in
    internals_labelled_keys, so test_withdrawn_numbers_outside_the_tiles_carry_the_label checks
    its EN and ZH prefix; and the series code draws no subsidence band note outside them."""
    notes = {"bn_runs", "bn_constructed", "bn_normal", "dn_yr", "dn_con", "dn_scaled"}
    sc = L()["page_scoping"]
    assert notes <= set(sc["internals_labelled_keys"]), notes - set(sc["internals_labelled_keys"])
    assert notes <= set(sc["answer_must_not_use_band_keys"])
    assert {f"i18n:{k}" for k in notes} <= set(CL()["C14e"]["text_keys"])
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    for k in notes:
        assert en[k].startswith("Model internals") and zh[k].startswith("模型內部"), k
    body = C.function_source("seriesFor")
    subs = body[body.index('if (S.ts === "subs")'):body.index("const lay =")]  # subsidence branch
    assert set(re.findall(r'tr\("(bn_[a-z_]+)"', subs)) <= notes


def test_pending_lines_wait_for_their_registration():
    """The page names E02c, the external-well test and the U1 narrowing only when the build
    found their ids in the registry (viewer_app PAGE_REGISTRATIONS -> modelcard.registrations)."""
    from hydrophysics.twin import viewer_app as va

    body = C.function_source("evidenceItems")
    for flag, key in (("e02c", "evc_pro_e02c"), ("ext", "evc_pro_ext"),
                      ("no_band", "evc_band_none")):
        assert re.search(r"if \(rg\." + flag + r'\) out\.push\(\["", tr\("' + key + r'"\)\]\)',
                         body), key
        assert body.count(f'"{key}"') == 1, key
    regs = {r for ids in va.PAGE_REGISTRATIONS.values() for r in ids}
    cited = {r for c in L()["claims"] for r in c["cites"]["registrations"]}
    pend = set(C.pending_registrations(L()))
    assert pend <= regs, f"a pending registration the page does not gate: {sorted(pend - regs)}"
    assert {"E02c.prospective_2026-11.v1", "U1.no_band.v1"} <= cited
    assert va.PAGE_REGISTRATIONS["ext"][0] in cited or va.PAGE_REGISTRATIONS["ext"][1] in cited
