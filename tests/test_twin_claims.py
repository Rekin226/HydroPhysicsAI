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
        also = c.get("ruling_also")                 # one ruling id, or a list of them
        for r in [c.get("ruling")] + (also if isinstance(also, list) else [also]):
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
                      ("no_band", "evc_band_none"), ("n3sl", "evc_n3sl")):
        assert re.search(r"if \(rg\." + flag + r'\) out\.push\(\["", tr\("' + key + r'"\)\]\)',
                         body), key
        assert body.count(f'"{key}"') == 1, key
    # FV's line waits for its registration too, then reads its report (fvLine)
    assert 'if (rg.fv) out.push(["", fvLine()]);' in body and body.count("fvLine()") == 1
    # N3's line also carries its conversion window, worded from the registered role
    assert re.search(r'if \(rg\.n3\) out\.push\(\["", tr\("evc_pile_n3", \{conv: ', body)
    assert body.count('"evc_pile_n3"') == 1
    regs = {r for ids in va.PAGE_REGISTRATIONS.values() for r in ids}
    cited = {r for c in L()["claims"] for r in c["cites"]["registrations"]}
    pend = set(C.pending_registrations(L()))
    assert pend <= regs, f"a pending registration the page does not gate: {sorted(pend - regs)}"
    assert {"E02c.prospective_2026-11.v1", "U1.no_band.v1"} <= cited
    assert va.PAGE_REGISTRATIONS["ext"][0] in cited or va.PAGE_REGISTRATIONS["ext"][1] in cited


# ---- round 6 (user rulings 2026-11-01, fourth set) -------------------------------------------
FOURTH = "user ruling 2026-11-01 (fourth set)"
FIFTH = "user ruling 2026-11-01 (fifth set)"
N3_SUB = "pile-anchored status-quo compaction at 2 Yunlin cells decided by N3"


def _reg(flag: str) -> str:
    """The ledger's id for a page registration flag (the first of viewer_app's ids it cites)."""
    from hydrophysics.twin import viewer_app as va

    cited = {r for c in L()["claims"] for r in c["cites"]["registrations"]}
    hit = [r for r in va.PAGE_REGISTRATIONS[flag] if r in cited]
    assert hit, f"the ledger cites none of viewer_app's ids for {flag}"
    return hit[0]


def _also(c: dict) -> set:
    a = c.get("ruling_also")
    return {c.get("ruling")} | set(a if isinstance(a, list) else [a])


def test_fourth_set_rulings_are_recorded():
    ids = {r["id"] for r in L()["rulings"]}
    assert {f"{FOURTH} ({x})" for x in "abcdef"} <= ids
    # and the fifth set (results/dataprep/rulings/user_rulings_2026-11-01.json)
    assert {f"{FIFTH} ({x})" for x in ("N3.level_only", "N3SL.design", "scorer.literal_rule",
                                         "chlever.per_window")} <= ids
    text = {r["id"]: r["text"] for r in L()["rulings"]}
    assert "after_the_fact_analysis" in text[f"{FOURTH} (e)"]
    assert "ROC115 excluded" in text[f"{FOURTH} (f)"]


def test_c31_is_decided_by_n3_and_no_cut_number_can_become_s():
    """Fourth set (a): the narrowed page mapping. Only C31 can become S on a joint N2 + N3G
    pass; the avoided-subsidence number stays W ('links tested, cut size untested'); the 溪州
    arm never gates; C09 and C30 carry the N3 sub-label for the 2-cell sub-claim only."""
    cl, n3 = CL(), _reg("n3")
    c31 = cl["C31"]
    # N3 is registered level-only (G0 failed; user ruling 2026-11-01, fifth set): C31 is N
    assert _st("C31") == {"N"} and c31["cites"]["registrations"] == [n3]
    assert f"{FIFTH} (N3.level_only)" in _also(c31) and f"{FIFTH} (N3.level_only)" in _also(
        cl["C27"])
    assert "registered level-only" in c31["wording_en"] and "C31 is N" in c31["wording_en"]
    assert "C31 is N" in c31["outcome_display"]["registered_state"]
    assert "compaction per metre of head change at the 2 Yunlin deep-pile cells" in c31["wording_en"]
    t = c31["outcome_display"]
    for k in ("before_scoring", "g0_failed", "joint_pass", "not_supported", "not_informative",
              "level_pass", "level_fail", "never"):
        assert t.get(k), k
    assert "C31 becomes S" in t["joint_pass"] and "no cut number becomes S" in t["joint_pass"]
    # equal to or narrower than N3's page mapping: only the joint pass reaches S
    for k, v in t.items():
        if k != "joint_pass":
            assert not re.search(r"C31 becomes S|C31 -> S|S for", v), k
    assert "C31 stays N" in t["conversion_pass_only"] and "head link: N2" in t["conversion_pass_only"]
    assert "'passed N3 (prospective)' at status N" in t["level_pass"]
    assert "only C31 can become S" in t["level_pass"]
    assert "makes only the 2-cell sub-claim S" not in CL()["C09"]["wording_en"]
    g0 = t["g0_failed"]
    for k in ("24-month conversion test gates", "'untested: the synthetic gate G0 failed'",
              "'conversion not identifiable at the piles by design (N3 G0)'",
              "keep the N3 sub-label", "rg.n3_conv12"):
        assert k in g0, k
    assert "C31 stays N" in t["report_only_role"]
    for rep in ("results/twin/mechanism/mechanism.json",
                "results/twin/deep_pile_holdout/report.json"):
        for cid in ("C27", "C31"):
            assert rep in cl[cid]["cites"]["reports"], (cid, rep)
    assert "links tested, cut size untested" in t["joint_pass"]
    assert "never gates" in t["never"] and "no Changhua subsidence lever" in t["never"]
    assert "no correction factor" in t["not_supported"]
    assert {"intended", "scoped"} <= set(c31["coverage"]) and "2 " in c31["coverage"]["scoped"]
    assert _st("C09") == {"W", "D"} and _st("C30") == {"N", "D"} and _st("C03") == {"W"}
    for cid in ("C09", "C30"):
        assert N3_SUB in cl[cid]["wording_en"] and n3 in cl[cid]["cites"]["registrations"], cid
        assert f"{FOURTH} (a)" in _also(cl[cid]), cid
    assert "links tested, cut size untested" in cl["C03"]["wording_en"]
    # no other claim can be promoted by N3: every claim citing N3 is D, N or W now
    for c in L()["claims"]:
        if n3 in c["cites"]["registrations"]:
            assert "S" not in _st(c["id"]), c["id"]
    assert "context only" in cl["C15"]["wording_en"] and _st("C15") == {"L"}


def test_c27_names_n3_only_through_the_registry_flag():
    """Rank 9: the decomposition line names N3 as the conversion decider only when the build
    found N3 in the registry; otherwise the conversion reads 'untested'."""
    body = C.function_source("evidenceItems")
    assert "const n3c = rg.n3_conv12 || rg.n3_conv24;" in body
    assert ('tr(!rg.n3 ? "decomp_conv_un" : n3c ? "decomp_conv_n3" : "decomp_conv_g0")'
            in body)
    assert ('tr(rg.n3_conv12 ? "pile_conv_12" : rg.n3_conv24 ? "pile_conv_24" : '
            '"pile_conv_ro")' in body)
    assert body.count('"evc_subs_decomp"') == 1
    for lang in ("en", "zh"):
        d = C.i18n(lang=lang)
        named = {k for k, v in d.items() if re.search(r"(?<![A-Za-z0-9])N3(?![A-Za-z0-9])", v)}
        assert named == {"evc_pile_n3", "decomp_conv_n3", "decomp_conv_g0"}, (lang, named)
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert "untested" in en["decomp_conv_un"] and "未經檢驗" in zh["decomp_conv_un"]
    assert "head rise times the column's compaction per metre" in en["evc_subs_decomp"]
    assert "neither factor decides a cut's subsidence size" in en["decomp_conv_n3"]
    assert "by design" in en["decomp_conv_g0"] and "reported only" in en["pile_conv_ro"]
    assert "December 2026 to November 2027" in en["evc_pile_n3"]     # the level window, always
    assert "December 2026 to November 2028" in en["pile_conv_24"]
    assert "2029-01-15" in en["pile_conv_24"] and "same months" in en["pile_conv_12"]
    assert "2026 年 12 月至 2028 年 11 月" in zh["pile_conv_24"]
    c27 = CL()["C27"]
    assert _st("C27") == {"N"} and _reg("n3") in c27["cites"]["registrations"]
    assert "N2.policy_response.prospective_2026-11.v1" in c27["cites"]["registrations"]
    assert {"i18n:evc_subs_decomp", "i18n:decomp_conv_un",
            "i18n:decomp_conv_g0"} <= set(c27["text_keys"])
    assert {"intended", "scoped"} <= set(c27["coverage"])
    assert "model form uncertainty, not a bound" in CL()["C25"]["wording_en"]


def test_c13a_caps_can_only_narrow_an_n2c_pass():
    """Fourth set (b), (c): F < 0.5 caps an N2C pass at N; a high confounded false-pass rate
    makes it 'reported, not shown'; the verdict word never changes. The page clauses wait for
    their registrations."""
    c13a = CL()["C13a"]
    t = c13a["outcome_display"]
    assert "F < 0.5 caps an N2C 'response supported' at N for C13a" in t["cap_F"]
    # fifth set (chlever.per_window): F < 0.5 is the only cap; F not evaluable caps nothing
    assert "F < 0.5 is the only cap" in t["cap_F"] and "neither caps anything" in t["cap_F"]
    assert "at N for C13a exactly as F < 0.5 does" not in t["cap_F"]
    assert "reported, not shown" in t["reading"] and "verdict word never changes" in t["reading"]
    assert "shared N2/N2C block-bootstrap scorer" in t["reading"]
    assert "model-internal lever spread (L)" in t["model_internal_spread"]
    assert "can only narrow it" in t["response_supported"]
    for flag in ("x5clm", "x5n2c"):
        assert _reg(flag) in c13a["cites"]["registrations"]
    assert {f"{FOURTH} (b)", f"{FOURTH} (c)"} <= _also(c13a)
    assert _st("C13a") == {"W"}
    body = C.function_source("policyResponseLines")
    assert 'rg.x5clm ? chCapF() : ""' in body and 'rg.x5n2c ? chCapRd() : ""' in body
    en = C.i18n(lang="en")
    assert "less than half" in en["pr_ch_cap_f"] and "reported, not shown" in en["pr_ch_cap_rd"]
    # each clause states the rule until its report exists, then the measured value (both
    # branches stay in the template; the result branch never loosens the rule)
    f, rd = C.function_source("chCapF"), C.function_source("chCapRd")
    assert 'if (!r) return tr("pr_ch_cap_f");' in f and '"pr_ch_cap_f_res"' in f
    assert "r.caps_c13a_at_N" in f
    assert 'if (!r) return tr("pr_ch_cap_rd");' in rd
    assert 'tr(r.false_pass >= .25 ? "pr_ch_rd_held" : "pr_ch_rd_res"' in rd
    assert "below 0.5" in en["pr_ch_cap_f_res"] and "first-year response only" in en[
        "pr_ch_cap_f_res"]
    assert "reported, not shown" in en["pr_ch_rd_held"] and "below 0.25" in en["pr_ch_rd_res"]
    # the scored results are in the ledger: F 0.49 caps C13a at N; the confounded false-pass
    # rate 0.116 does not fire; the rates and the power are recorded for a later N2C display
    n = c13a["notes"]
    for k in ("F = 0.49", "x1.04-1.30", "x0.96-1.27", "x1.31 (2026-27) and x1.33 (2030-32)",
              "0.116", "0.744", "0.828", "0.93, 0.60 with the canal confounder",
              "the N2C verdict word never changes"):
        assert k in n, k
    assert "SLOT (rank 8" not in n


def test_the_decidability_reading_is_one_sided():
    """Fourth set (d): P <= 0.5 means 'weak by design'; there is no 'a fail is informative'."""
    cl, fv = CL(), _reg("fv")
    for cid in ("E02", "E21", "C30"):
        assert fv in cl[cid]["cites"]["registrations"], cid
    for cid in ("E02", "E21"):
        w = cl[cid]["wording_en"]
        assert "<= 0.5" in w and "weak by design" in w and "no 'a fail is informative'" in w
        # the study's scope: E02c's level rule and E02c's county gates, not E02's own coverage
        assert "E02c's level rule (also carried on E02's level line)" in w, cid
        assert "E02's own pooled coverage check is report-only" in w, cid
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert "0.5 or less" in en["evc_pro_fv"] and "weak by design" in en["evc_pro_fv"]
    assert "E02c's county coverage gates" in en["evc_pro_fv"]
    assert "E02's and E02c's level and coverage verdicts" not in en["evc_pro_fv"]
    assert "E02 與 E02c 的水位與涵蓋率判定" not in zh["evc_pro_fv"]
    assert "informative" not in en["evc_pro_fv"] and "0.5" in zh["evc_pro_fv"]


def test_the_decidability_study_was_not_evaluable_and_gives_no_reading():
    """FV.decidability.v1 was scored not evaluable (superposition failed): by its page_mapping
    there is no interpretation line and E02/E02c are read as registered. The page shows the
    declared not-evaluable sentence once the report exists and the rule fixed beforehand only
    without it; the ledger keeps the registered rule and records the result."""
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    want = {"en": "Synthetic decidability study (FV.decidability.v1) not evaluable ({why}); "
                  "E02 and E02c gain no interpretation line from it.",
            "zh": "合成可判定性研究（FV.decidability.v1）無法評估（{why}）；E02 與 E02c "
                  "不因此增加解讀說明。"}
    decl = C.PREREG / "FV.decidability.v1.json"
    if decl.exists():                          # the registered wording, verbatim
        w = json.loads(decl.read_text(encoding="utf-8"))["readings"]["wording"]
        assert {k: w["not_evaluable"][k] for k in ("en", "zh")} == want
    assert en["evc_pro_fv_ne"] == want["en"] and zh["evc_pro_fv_ne"] == want["zh"]
    for d in (en, zh):
        assert "weak by design" not in d["evc_pro_fv_ne"] and "informative" not in d[
            "evc_pro_fv_ne"]
    body = C.function_source("fvLine")
    assert 'if (!r) return tr("evc_pro_fv");' in body and 'tr("evc_pro_fv_ne"' in body
    assert "s.pass === false" in body and '"fv_why_sup"' in body
    cl = CL()
    assert "Rank 11 result" in cl["E02"]["notes"] and "SLOT (rank 11" not in cl["E02"]["notes"]
    for k in ("not_evaluable", "superposition failed", "0.2845 m", "0.02 m",
              "'no interpretation line; E02/E02c read as registered'"):
        assert k in cl["E02"]["notes"], k
    for cid in ("E02", "E21", "C30"):
        c = cl[cid]
        assert "i18n:evc_pro_fv_ne" in c["text_keys"], cid
        assert "results/dataprep/reports/FV.decidability.v1/report.json" in c["cites"]["reports"]
        assert "not evaluable" in c["wording_en"], cid
    # the registered rule stays written; the result adds no reading of it
    assert "weak by design" in cl["E02"]["wording_en"] and "<= 0.5" in cl["E02"]["wording_en"]
    assert "clause (d) does not apply" in cl["C30"]["wording_en"]


def test_fs_wording_follows_its_declaration_and_waits_for_its_report():
    """FS.start_invariance.v1 (rank 7): registered, result pending. The page words C01 from the
    report's summary.wording_case only (evidence_and_ledger_mapping); no report, no FS line; FS
    is not cited until it is scored."""
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert ("does not depend on the 2022-12 start state (within 5 %, model-only declaration "
            "FS.start_invariance.v1)") in en["fs_pass"]
    assert ("does not depend on the start under the page's restart family (free-run start S1 "
            "within 5 %, model-only") in en["fs_s2_only"]
    assert ("depends on the start by up to {x} %{q} only under the superseded untapered field "
            "restart (S2, model-only declaration FS.start_invariance.v1)") in en["fs_s2_only"]
    assert "depends on the 2022-12 start state by up to {x} %{q} (model-only" in en["fs_fail"]
    # the level shift is gone by 2026-12 (report baseline_level_shift_m): only the early levels
    assert "first months after it (gone by December 2026), not these policy differences" in en[
        "start_note_fs"]
    for k in ("fs_pass", "fs_s2_only", "fs_fail", "start_note_fs"):
        assert "FS.start_invariance.v1" in zh[k], k
    gate = C.function_source("fsResult")
    assert "return rg.fs && r ? r : null;" in gate
    words = C.function_source("fsWords")
    for case in ("pass", "S2_only", "S1_only", "S1_and_S2"):
        assert f'"{case}"' in words, case
    assert '"not_evaluable"' not in words                  # ledger note only
    assert 'fs && fs.wording_case === "pass" ? tr("start_note_fs")' in C.function_source(
        "renderStartNote")
    assert "const fs = fsWords(); if (fs) p(fs);" in C.function_source("tileDefHtml")
    # scored 2026-11-01: pass; C01 cites FS and the new L row (E23) the declaration calls for
    cl, fs = CL(), "FS.start_invariance.v1"
    assert _reg("fs") == fs and _st("E23") == {"L"} and _st("C01") == {"S", "D"}
    for cid in ("C01", "E23"):
        assert fs in cl[cid]["cites"]["registrations"], cid
        assert "results/dataprep/reports/FS.start_invariance.v1/report.json" in cl[cid][
            "cites"]["reports"], cid
    for k in ("verdict pass", "9.74e-06 %", "numerical precision",
              "linear in the start state", "FV.decidability.v1's failed pumping superposition"):
        assert k in cl["E23"]["notes"], k
    assert "archived unopened" in cl["E23"]["notes"]
    for k in ("fs_pass", "start_note_fs"):
        assert DM()["i18n"][k] == ["C01", "E23"], k
    # a departure below the display unit is a bound, never a rounded 0.0
    assert 'v < .001 ? "< 0.001"' in words


def test_county_area_and_coast_get_the_n3sl_sub_label():
    cl, sl = CL(), _reg("n3sl")
    assert _st("C05") == {"W", "D"} and _st("C26") == {"N", "D"}
    for cid in ("C05", "C26"):
        assert sl in cl[cid]["cites"]["registrations"] and "i18n:evc_n3sl" in cl[cid]["text_keys"]
    assert "SL's page role is fixed now, before registration, as report_only" in cl["C26"]["wording_en"]
    assert "C26b" not in cl and "--sl-role report_only" in cl["C26"]["wording_en"]
    assert "ROC115 excluded" in cl["C26"]["notes"]
    # N3S-L's before_scoring mapping also names C29 (ledger only, no page text)
    assert "'coastal cells: see N3S-L'" in cl["C29"]["wording_en"] and _st("C29") == {"L"}
    assert sl in cl["C29"]["cites"]["registrations"] and f"{FOURTH} (f)" in _also(cl["C29"])
    # timing: ROC116's interval starts before registration; only the summaries postdate it
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert "registered before the summaries for the survey years it scores exist" in en["evc_n3sl"]
    assert "在其評分的測量年度摘要出現之前登錄" in zh["evc_n3sl"]
    readme = (C.REPO / "README.md").read_text(encoding="utf-8")
    for bad in ("registered before the survey years it scores", "registered before those surveys"):
        assert bad not in en["evc_n3sl"] and bad not in " ".join(readme.split()), bad
    assert "已在其評分的測量年度之前登錄" not in zh["evc_n3sl"]


def test_the_start_code_fact_is_in_the_internals_note():
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert "free-running heads" in en["mi_note"] and "December 2022 heads" in en["mi_note"]
    assert "2022 年 12 月水頭" in zh["mi_note"]
    for cid in ("C29", "C30"):
        assert "i18n:mi_note" in CL()[cid]["text_keys"], cid
        assert cid in DM()["i18n"]["mi_note"], cid


def _paragraphs(text: str) -> list[str]:
    return [" ".join(p.split()) for p in re.split(r"\n\s*\n|\n(?=\s*[-*]\s)", text)]


def test_pile_context_is_cited_together_and_only_in_the_docs():
    """Fourth set (e): the ring-vs-column seasonal slope and the 2024-25 pile R² appear only
    together, with the after-the-fact label, and never on the page."""
    for doc in ("README.md", "MODEL_CARD.md", "docs/TWIN_EVIDENCE_2026-10.md"):
        for par in _paragraphs((C.REPO / doc).read_text(encoding="utf-8")):
            ring, pile = "9.3 mm/m" in par, "R² 0.92" in par
            if ring or pile:
                assert ring and pile, (doc, par[:160])
                assert "after-the-fact" in par and "no correction applied" in par, (doc, par[:160])
                assert "not a test" in par, (doc, par[:160])
    src = C.template_src()
    assert "9.3 mm" not in src and "R² 0.92" not in src


def test_result_placeholders_stay_in_the_ledger():
    """Ranks 7, 8, 11 and 12 are scored and their slots filled; the page, the README and the
    model card never show a slot or a placeholder."""
    cl = CL()
    for cid in ("C27", "C13c"):
        assert "coefficient not identifiable from held piles" in cl[cid]["notes"], cid
        assert "Capped at N, no correction applied" in cl[cid]["notes"], cid
        assert "PLACEHOLDER (rank 12" not in cl[cid]["notes"], cid
    assert "Rank 7 result" in cl["C01"]["notes"] and "SLOT (rank 7" not in cl["C01"]["notes"]
    assert "Rank 8 result" in cl["C13a"]["notes"] and "Rank 11 result" in cl["E02"]["notes"]
    for c in L()["claims"]:                      # never in a displayed wording
        assert "PLACEHOLDER" not in c["wording_en"] and "SLOT (" not in c["wording_en"], c["id"]
    for p in (C.TEMPLATE, C.REPO / "README.md", C.REPO / "MODEL_CARD.md"):
        text = p.read_text(encoding="utf-8")
        for bad in ("PLACEHOLDER", "SLOT (", "<k>"):
            assert bad not in text, (p.name, bad)


# the 12-month window is worded statically in these places; G0 may make the 24-month unit gating
# or leave the conversion report-only, and then each must be edited before the commit
N3_CONV12_ANCHORS = {
    "README.md": "is decided by `N3.piles_prospective.v1` on the deep piles printed in the WRA "
                 "monthly reports for December 2026 to November 2027",
    "MODEL_CARD.md": "is decided by `N3.piles_prospective.v1` only at the 2 Yunlin deep-pile "
                     "cells (土庫國中, 中科虎尾; window 2026-12 to 2027-11",
    "ledger:C31": "on the 2026-12..2027-11 window (not before 2028-01-15) when G0 makes the "
                  "12-month conversion test gating",
}


def _n3_anchor_texts() -> dict[str, str]:
    out = {d: " ".join((C.REPO / d).read_text(encoding="utf-8").split())
           for d in ("README.md", "MODEL_CARD.md")}
    out["ledger:C31"] = CL()["C31"]["wording_en"]
    return out


# G0 failed at both horizons and N3 is registered level-only (user ruling 2026-11-01, fifth set):
# these places word that registration statically; the next test checks it against the roles
N3_LEVEL_ONLY_ANCHORS = {
    "README.md": "`N3.piles_prospective.v1` was registered level-only",
    "MODEL_CARD.md": "`N3.piles_prospective.v1` is registered level-only",
    "ledger:C31": "N3.piles_prospective.v1 is registered level-only",
}


def test_n3_docs_word_the_level_only_registration():
    """The static level-only wording exists where the next test expects it (so that test
    bites), and the 12-month conversion wording is gone."""
    texts = _n3_anchor_texts()
    for where, anchor in N3_LEVEL_ONLY_ANCHORS.items():
        assert anchor in texts[where], where
    for where, anchor in N3_CONV12_ANCHORS.items():
        assert anchor not in texts[where], where


@NEED_REGISTRY
def test_n3_docs_follow_the_registered_conversion_role():
    """Fourth set (a), G0: once N3 is registered, the README, the model card and C31 may word the
    conversion as decided on December 2026 to November 2027 only if the registered document makes
    the 12-month conversion test gating (the page follows the role by itself)."""
    idx, _ = C.prereg_index()
    n3 = _reg("n3")
    if n3 not in idx:
        pytest.skip(f"{n3} is not registered yet")
    roles = json.loads((C.PREREG / f"{n3}.json").read_text(encoding="utf-8")).get("roles") or {}
    texts = _n3_anchor_texts()
    gating = [t for t in ("N3P.conversion_12m", "N3P.conversion_24m") if roles.get(t) == "gating"]
    if gating:
        level_only = [w for w, a in N3_LEVEL_ONLY_ANCHORS.items() if a in texts[w]]
        assert not level_only, f"{gating} gate; {level_only} still word N3 level-only"
    if roles.get("N3P.conversion_12m") == "gating":
        return
    stale = [w for w, a in N3_CONV12_ANCHORS.items() if a in texts[w]]
    assert not stale, (f"N3's 12-month conversion test is {roles.get('N3P.conversion_12m')!r}; "
                       f"reword the conversion window or role in {stale} (C31 g0_failed row)")


# ---- round 7 (user rulings 2026-11-01, sixth to tenth sets) ----------------------------------
R7_SETS = {"sixth set": ("R7.gnss_seasonal_S", "R7.e02d_residual_read", "R7.heads_years23",
                         "R7.n3_g0prime", "R7.l1_pre2012", "R7.fv_v2_page", "R7.forecast_mode_S"),
           "seventh set": ("R7.fm_page_line", "R7.fw_wording", "R7.fw_budget", "R7.commit_first"),
           "eighth set": ("R7.fw_v2_weather",), "ninth set": ("R7.fw_v1_pumping",),
           "tenth set": ("R7.fw_both",)}


def _r7(rid: str) -> str:
    return next(f"user ruling 2026-11-01 ({s}) ({rid})" for s, ids in R7_SETS.items()
                if rid in ids)


def test_round7_rulings_are_recorded():
    ids = {r["id"] for r in L()["rulings"]}
    want = {f"user ruling 2026-11-01 ({s}) ({i})" for s, v in R7_SETS.items() for i in v}
    assert want <= ids, sorted(want - ids)
    text = {r["id"]: r["text"] for r in L()["rulings"]}
    assert "Superseded" in text[_r7("R7.fw_v2_weather")]
    assert "Not approved" in text[_r7("R7.forecast_mode_S")]


def test_round7_interval_rows_follow_their_registrations():
    """E02d: new sub-claim C14a-r (N+D, decisive by design); E02y: C14b N -> N+D on the
    year-2/3 ruling; C14f: N3F's interval check relabelled (D, no new page text)."""
    cl = CL()
    a = cl["C14a-r"]
    assert _st("C14a-r") == {"N", "D"}
    assert a["cites"]["registrations"][0] == "E02d.decomposed_prospective_2026-12.v1"
    assert a["cites"]["tests"] == ["E02dP.G_idio_2026-12"]
    assert "decided by a registered test that is decisive by design (U1.decomp_power.v1)" in a[
        "wording_en"]
    assert a["text_keys"] == ["i18n:evc_pro_e02d"] and "calibrated" in a["notes"]
    b = cl["C14b"]
    assert b["status_before"] == "N" and _st("C14b") == {"N", "D"}
    assert b["ruling"] == _r7("R7.heads_years23")
    assert "E02y.years23_2027-11_2029-10.v1" in b["cites"]["registrations"]
    assert {"E02yP.rel_2027-11", "E02yP.rel_2028-11"} <= set(b["cites"]["tests"])
    assert "a lower bound" in b["wording_en"] and "report-only, never on the" in b["notes"]
    f = cl["C14f"]
    assert _st("C14f") == {"D"} and f["text_keys"] == ["i18n:evc_pro_ext"]
    assert f["cites"]["tests"] == ["N3FP.ext_level_2026-11"] and "Relabel only" in f["wording_en"]
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert "untested beyond" in en["pro_out_e02y"] and "2029-01-15" in zh["pro_out_e02y"]
    assert en["pro_out_un"] == "Outside that window the recipe is untested."
    body = C.function_source("evidenceItems")
    assert 'if (rg.e02d) out.push(["", tr("evc_pro_e02d")]);' in body
    assert 'tr(rg.e02y ? "pro_out_e02y" : "pro_out_un")' in body


def test_round7_model_only_readings_are_worded_as_declared():
    """FM (ruling R7.fm_page_line), FV v2 (post-failure, exploratory; R7.fv_v2_page) and FV.ext
    carry their registered wording; FV v2's lines say 'post-failure exploratory' and no S
    follows; the one-sided reading has no 'a fail is informative' line."""
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    cl = CL()
    for k in ("evc_fv2_lvl_nsw", "evc_fv2_lvl_wbd", "evc_fv2_cov_nsw", "evc_fv2_cov_wbd",
              "evc_fv2_rc_nso", "evc_fv2_rc_omits", "bn_head_fy"):
        assert "post-failure exploratory" in en[k] and "FV.decidability.v2" in zh[k], k
        assert "informative" not in en[k], k
    assert "lies beyond the start memory" in en["evc_pro_fm"]
    assert "FM.origin_memory.v1" in zh["evc_pro_fm"]
    decl = C.PREREG / "FV.decidability.v2_postfailure.json"
    if decl.exists():                                   # the registered text, verbatim
        w = json.loads(decl.read_text(encoding="utf-8"))["readings"]["reading_1"]["wording"]
        for lang, d in (("en", en), ("zh", zh)):
            assert d["evc_fv2_lvl_nsw"] == w["level"]["not_shown_weak"][lang].replace(
                "{p:.2f}", "{p}"), lang
            assert d["evc_fv2_cov_wbd"] == w["coverage"]["weak_by_design"][lang].replace(
                "{p:.2f}", "{p}").replace("{county}", "{cty}"), lang
    for cid in ("E02", "E21", "C30"):
        assert "FV.decidability.v2_postfailure" in cl[cid]["cites"]["registrations"], cid
        assert "FM.origin_memory.v1" in cl[cid]["cites"]["registrations"], cid
        assert _r7("R7.fv_v2_page") in _also(cl[cid]), cid
    assert "FV.ext.v1" in cl["E22"]["cites"]["registrations"] and _st("E22") == {"D"}
    assert "FV.stat.v1" in cl["E04"]["cites"]["registrations"]
    assert "not evaluable (engineering)" in cl["E04"]["notes"]
    assert "start memory" in cl["E23"]["wording_en"] and _st("E23") == {"L"}
    for c in L()["claims"]:                            # model-only studies support nothing
        regs = set(c["cites"]["registrations"])
        if regs & {"FV.decidability.v2_postfailure", "FV.ext.v1", "FM.origin_memory.v1",
                   "FH.forcing_sensitivity.v1", "FW.weather_replay.v1", "FW.weather_replay.v2",
                   "SD.cut_response_spread.v2"}:
            assert "S" not in _st(c["id"]), c["id"]


def test_avoided_subsidence_numbers_of_round7_stay_off_the_page():
    """FW's numbers and SD's spread are ledger and README only (rulings R7.fw_wording,
    R7.fw_both; the SD tile display was not ruled): the page has only FW's qualitative note."""
    src = C.template_src()
    for bad in ("0.47-1.42", "x2.02", "2.4-3.2 cm", "lasting benefit", "SD.cut_response_spread"):
        assert bad not in src, bad
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    for k in ("mi_fw_both", "mi_fw_v1", "mi_fw_v2"):
        assert not re.search(r"\d\s*cm", en[k]) and "公分" not in zh[k], k
    for k in ("mi_fw_both", "mi_fw_v2"):              # the gauge caveat on the weather mode
        assert "rf11 for all of 2021–22 and rf15 in 2015 H2" in en[k], k
        assert "rf11" in zh[k] and "rf15" in zh[k], k
    assert "dry" not in en["mi_fw_v1"].replace("not dry", "")   # pumping-only: no dry year
    cl = CL()
    e24 = cl["E24"]
    assert _st("E24") == {"L"} and e24["text_keys"] == ["doc:README.md"]
    assert "model-form spread, not an interval" in e24["wording_en"]
    for cid in ("C03", "C29", "C14e", "C11", "C27"):
        assert "row E24" in cl[cid]["notes"], cid
    assert _st("C03") == {"W"} and _st("C15") == {"L"} and _st("C25") == {"N"}
    for cid in ("C15", "C25"):
        assert {"FW.weather_replay.v1", "FW.weather_replay.v2"} <= set(
            cl[cid]["cites"]["registrations"]), cid


def test_changhua_direction_rows_follow_the_eleventh_set():
    """User rulings 2026-11-02 (eleventh set): E25/E26 and the C20/C21 direction clause on the
    page, README and model card as 'S (design-time), direction only; size not supported', with
    the slopes; the Changhua lever stays untested and capped; C11 drops the leader wording
    (Tier A+B); the census audit gets one README sentence and no page text."""
    ids = {r["id"] for r in L()["rulings"]}
    for rid in ("R7.changhua_page", "R7.sd_leader_tier", "R7.audit_readme"):
        assert f"user ruling 2026-11-02 (eleventh set) ({rid})" in ids, rid
    cl = CL()
    for cid, test, key in (("E25", "I1CHX.direction_R1c_ext23", "evc_chx"),
                           ("E26", "I1CHB.direction_R1c_2010_2011", "evc_chb")):
        assert _st(cid) == {"S"} and cl[cid]["cites"]["tests"] == [test], cid
        assert f"i18n:{key}" in cl[cid]["text_keys"] and DM()["i18n"][key][0] == cid, cid
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    for k in ("evc_chx", "evc_chb"):
        assert "S (design-time), direction only; size not supported" in en[k], k
        assert "僅方向；大小未獲支持" in zh[k], k
    assert "untested (N2C registered, scored not before 2028-01-15) and capped" in en["evc_chx"]
    body = C.function_source("chDirLines")
    assert "rg[k] && r" in body and '"evc_chx"' in body and '"evc_chb"' in body
    readme = " ".join((C.REPO / "README.md").read_text(encoding="utf-8").split())
    card = " ".join((C.REPO / "MODEL_CARD.md").read_text(encoding="utf-8").split())
    for text in (readme, card):
        assert "0.17 [0.02, 0.44]" in text and "0.37 [0.15, 1.35]" in text
        assert "direction only; size not supported" in text and "12 wells" in text
    assert _st("C13a") == {"W"} and _st("E20") == {"D"}
    for cid in ("C20", "C21"):
        assert {"I1.CHX.v1", "I1.CH_backcast.v1"} <= set(cl[cid]["cites"]["registrations"])
    # C11: Tier A+B decides; no township is named as leading
    assert "column-dependent" in en["tn_pol"] and "no township is named as leading" in en["tn_pol"]
    assert "{bias}" not in en["tn_pol"] and "headline" not in en["tn_pol"]
    assert "居首" in zh["tn_pol"] and "R7.sd_leader_tier" in " ".join(_also(cl["C11"]))
    # the audit: one README sentence, no page text
    assert "`model_census` v1 as delivered" in readme and "(D1b)" in readme
    assert "model_census" not in C.template_src()
    assert "N3.G0prime.v1" in cl["C31"]["notes"] and _st("C31") == {"N"}
