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
    """Exact both ways. A conditional block (ruling R8.9: the all30 file, viewer_app --all30)
    is mapped only when the page names its source in release.sources, and then exactly; a page
    built without that source may carry none of its paths."""
    D = C.payload_structure(_page())
    paths = {p for p in C.numeric_paths(D) if not p.startswith("arrays.")}
    m = dict(DM()["payload"])
    srcs = set(D["release"]["sources"])
    for name, blk in DM().get("payload_conditional", {}).items():
        _ids_ok(f"payload_conditional.{name}", blk["paths"])
        assert not set(blk["paths"]) & set(m), name
        if blk["source"] in srcs:
            m.update(blk["paths"])
        else:
            assert not set(blk["paths"]) & paths, f"{name}: paths without the source {srcs}"
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
    arm never gates; C09 and C30-scen (C30 before the R8.7 split) carry the N3 sub-label for the
    2-cell sub-claim only."""
    cl, n3 = CL(), _reg("n3")
    c31 = cl["C31"]
    # N3 is registered level-only (G0 failed; user ruling 2026-11-01, fifth set): C31 is N
    # N3.G0prime.v1 (failed for every estimator) is cited beside it since round 8 (F1)
    assert _st("C31") == {"N"} and c31["cites"]["registrations"] == [n3, "N3.G0prime.v1"]
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
    assert _st("C09") == {"W", "D"} and _st("C30-scen") == {"N"} and _st("C03") == {"W"}
    for cid in ("C09", "C30-scen"):
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
    for cid in ("E02", "E21", "C30-scen"):
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
    for cid in ("E02", "E21", "C30-scen"):
        c = cl[cid]
        assert "i18n:evc_pro_fv_ne" in c["text_keys"], cid
        assert "results/dataprep/reports/FV.decidability.v1/report.json" in c["cites"]["reports"]
        assert "not evaluable" in c["wording_en"], cid
    # the registered rule stays written; the result adds no reading of it
    assert "weak by design" in cl["E02"]["wording_en"] and "<= 0.5" in cl["E02"]["wording_en"]
    assert "clause (d) does not apply" in cl["C30-scen"]["wording_en"]


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
    # scored 2026-11-01: pass; C01 cites FS and the new L row (E23) the declaration calls for.
    # Round 8 (ruling R8.8) splits C01: C01 keeps the direction (S), C01-w the window (D) and
    # C01-sz the size (N)
    cl, fs = CL(), "FS.start_invariance.v1"
    assert _reg("fs") == fs and _st("E23") == {"L"} and _st("C01") == {"S"}
    assert _st("C01-w") == {"D"} and _st("C01-sz") == {"N"}
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
    for cid in ("C29", "C30-scen"):
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
    for cid in ("E02", "E21", "C30-scen"):
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
    """FW's numbers are ledger and README only (rulings R7.fw_wording, R7.fw_both): the page has
    only FW's qualitative note. SD's spread (E24) reaches the page only in the tA definition under
    Model internals (ruling R8.11, 2026-11-03), worded from SD v2's report, never written into
    the template and never in an answer function."""
    src = C.template_src()
    for bad in ("0.47-1.42", "0.47–1.42", "x2.02", "×2.02", "2.4-3.2 cm", "lasting benefit"):
        assert bad not in src, bad
    en0 = C.i18n(lang="en")
    assert [k for k, v in en0.items() if "SD.cut_response_spread" in v] == ["def_tA_e24"]
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    for k in ("mi_fw_both", "mi_fw_v1", "mi_fw_v2"):
        assert not re.search(r"\d\s*cm", en[k]) and "公分" not in zh[k], k
    for k in ("mi_fw_both", "mi_fw_v2"):              # the gauge caveat on the weather mode
        assert "rf11 for all of 2021–22 and rf15 in 2015 H2" in en[k], k
        assert "rf11" in zh[k] and "rf15" in zh[k], k
    assert "dry" not in en["mi_fw_v1"].replace("not dry", "")   # pumping-only: no dry year
    cl = CL()
    e24 = cl["E24"]
    assert _st("E24") == {"L"} and e24["text_keys"] == ["doc:README.md", "i18n:def_tA_e24"]
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


# ---- round 8 (user rulings 2026-11-03, twelfth set) ------------------------------------------
def _r8(rid: str) -> str:
    return f"user ruling 2026-11-03 (twelfth set) ({rid})"


# status_current at main 711447b, before the round-8 results, of every row that was S then
PRE_ROUND8_STATUS = {"C01": "S+D", "E25": "S", "E26": "S"}
R8_MODEL_ONLY = ("FV.stat.v2", "X5.n2_reading.v1", "COLF.column_family_2028.v1",
                 "FE.encompass_2026-12.v1", "CB.coverage_side.v1", "FB.family_power.v1",
                 "U1.ext_transfer_power.v1", "X5.gnss_seasonal_family.gates.v1",
                 "N2CS.osse_design.v1", "GOV.heads_window_product.v1", "FW.rider.v1")


def test_round8_rulings_and_rows_follow_the_scored_reports():
    """The twelfth set is recorded; each scored round-8 study moves only what its registration
    maps: E04 and N2's window row C01-w get measured decidability (C01 split into C01, C01-w and
    C01-sz under R8.8, following C14a -> C14a-r), E02z is row E30 (D), E02x's failed P-gate is
    row C14f-r (N), FB is row E29 (N), and no model-only study supports anything."""
    ids = {r["id"] for r in L()["rulings"]}
    for rid in ("R8.1a", "R8.1b", "R8.1c", "R8.1d", "R8.1e", "R8.1f", "R8.2", "R8.3", "R8.4a",
                "R8.5", "R8.7", "R8.8", "R8.9", "R8.10", "R8.11", "R8.14"):
        assert _r8(rid) in ids, rid
    # R8.7 (forecast of record, C30 split) is wired since 2026-11-06 (test_r8_7_*); R8.9 (the
    # all30 headline lever) since 2026-11-06 (test_r8_9_*)
    for rid in ("R8.6a", "R8.6b", "R8.6c", "R8.13"):   # wired separately
        assert _r8(rid) not in ids, rid
    assert all("user_rulings_2026-11-03.json (sha256 ec610b69)" in r["text"]
               for r in L()["rulings"] if r["id"].startswith("user ruling 2026-11-03"))
    cl = CL()
    c01, w, sz = cl["C01"], cl["C01-w"], cl["C01-sz"]
    assert c01["status_before"] == "U" and _st("C01") == {"S"} and c01["ruling"] == _r8("R8.8")
    assert "S+D from that scoping" in c01["notes"] and "split on 2026-11-06" in c01["notes"]
    assert "X5.n2_reading.v1" not in c01["cites"]["registrations"]
    assert c01["cites"]["tests"] == ["N1P.C_heads_yunlin", "N1.G1_mask_E2015"]
    assert _st("C01-w") == {"D"} and w["cites"]["tests"] == ["N2P.heads_yunlin_response"]
    assert set(w["cites"]["registrations"]) == {"N2.policy_response.prospective_2026-11.v1",
                                                "X5.n2_reading.v1"}
    assert "'informative'" in w["notes"] and "cannot decide" in w["notes"]
    assert "0.752 at rho -0.87" in w["notes"] and "0.002 at rho -0.87" in w["notes"]
    assert _st("C01-sz") == {"N"} and "size likely too small; not gated" in sz["wording_en"]
    assert "X5.lever_class_share.v1" in sz["cites"]["registrations"]
    assert "No calibrated correction factor follows." in sz["doc_anchors"]["README.md"]["present"]
    assert DM()["tiles"]["tW"] == ["C01", "C01-w", "C01-sz"]
    assert DM()["i18n"]["evc_n2_rd"] == ["C01-w"] and DM()["functions"]["n2rdLine"] == ["C01-w"]
    e04 = cl["E04"]
    assert _st("E04") == {"D"} and "FV.stat.v2" in e04["cites"]["registrations"]
    assert "'not_shown_weak'" in e04["notes"] and "decidability measured" in e04["notes"]
    assert "81 Yunlin / 66 Changhua" in e04["notes"]                     # the erratum
    assert "report-only" in e04["notes"] and "FE.encompass_2026-12.v1" in e04["notes"]
    e30 = cl["E30"]
    assert _st("E30") == {"D"} and e30["ruling"] == _r8("R8.14")
    assert e30["cites"]["registrations"][0] == "E02z.level_years23.v1"
    assert set(e30["cites"]["tests"]) == {"E02zP.level_2027-11", "E02zP.level_2028-11"}
    assert "GOV.heads_window_product.v1" in e30["cites"]["registrations"]
    assert "E02z.level_years23.v1" in cl["C30-scen"]["cites"]["registrations"]
    assert "GOV.heads_window_product.v1" in cl["C14b"]["cites"]["registrations"]
    f = cl["C14f-r"]
    assert _st("C14f-r") == {"N"} and f["cites"]["registrations"] == ["U1.ext_transfer_power.v1"]
    assert "'undecidable by design at the external wells'" in f["wording_en"]
    e29 = cl["E29"]
    assert _st("E29") == {"N"} and e29["cites"]["registrations"] == ["FB.family_power.v1"]
    assert "family BF not decidable by design (FB.family_power.v1)" in e29["wording_en"]
    assert not {"E27", "E28"} & set(cl), "E27/E28 were conditional on registrations that failed"
    for c in L()["claims"]:                 # a model-only study never adds an S
        if set(c["cites"]["registrations"]) & set(R8_MODEL_ONLY) and "S" in _st(c["id"]):
            assert "S" in C.codes(PRE_ROUND8_STATUS.get(c["id"], "")), c["id"]
    for cid in ("E25", "E26"):                    # the S rests on the primary tests only
        assert len(cl[cid]["cites"]["tests"]) == 1, cid


def test_round8_narrowings_and_closures_are_worded_as_registered():
    """The two-contrast narrowing on E25, CH-J's report-only line and N2CS's N line (with its
    failing condition) on C20/C21, COLF's narrowing on E24/C09/C11, the GNSS closure, CB on
    C14a; verbatim against the reports when they exist."""
    cl = CL()
    assert "direction not separable from Yunlin spill-over" in cl["E25"]["notes"]
    assert "0.148 [-0.038, 0.448]" in cl["E25"]["notes"] and _st("E25") == {"S"}
    chj = ("CH-J (report-only): Changhua sw fraction off its 0.01 bound; median Changhua "
           "seasonal amplitude ratio 0.78, 0.78 (seeds 2027, 2029); not support")
    rep = C.REPO / "results/dataprep/reports/I1.CHX.v1/chj_report.json"
    if rep.exists():
        assert json.loads(rep.read_text(encoding="utf-8"))["r8_1b"]["wording"] == chj
    for cid in ("C20", "C21"):
        n = cl[cid]["notes"]
        assert chj in n and "co-located canal/paddy recharge is NOT rejected" in n, cid
        assert "one replicate" in n and "only the no-harm" in n, cid      # the seed disclosure
        assert "the --ske-skv-max 0.1 cap applies globally" in n, cid      # R8.1f
        assert "Q0's own three-zone uncapped column" in n, cid
        assert "'size class not decidable prospectively (OSSE power 0.89)'" in n, cid
        assert "false 'not overstated' rate" in n and "0.47-0.63" in n and "<= 0.20" in n, cid
        assert _st(cid) == {"N"}, cid
        assert "co-located canal/paddy recharge rejected" not in cl[cid]["wording_en"], cid
    for cid in ("E24", "C09", "C11"):
        assert "cannot separate these columns" in cl[cid]["notes"], cid
        assert "COLF.column_family_2028.v1" in cl[cid]["cites"]["registrations"], cid
    assert "tie artefact" in cl["E24"]["notes"]
    for cid in ("E24", "C27", "C03", "C11", "C29"):
        assert "the GNSS seasonal route is closed" in cl[cid]["notes"], cid
    assert "side_does_not_separate" in cl["C14a"]["notes"]
    assert _st("C03") == {"W"} and _st("E24") == {"L"} and _st("C27") == {"N"}


def test_round8_page_lines_follow_r8_8_and_r8_11_only():
    """R8.8: the E25/E26 caveats and the two-contrast clause, C27 in registered numbers, the
    x1.27 clause on C01/C12, the C01 split, and the N2 and FV.stat.v2 operating-characteristic
    lines. R8.11: E24 on tA under Model internals, never in an answer function. Not on the page:
    the FV v2 per-truth split."""
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert "imprecise and not evidence of specificity" in en["evc_chx"]
    assert "不構成專一性的證據" in zh["evc_chx"]
    assert "not separable from Yunlin spill-over" in en["chx_2c_ns"] and "{lo}" in zh["chx_2c_ns"]
    assert "not independent of the screen test" in en["evc_chb"] and "{k0}" in en["chb_syn"]
    for k in ("evc_chx", "evc_chb"):            # the analyst renders these: no new placeholder
        assert set(re.findall(r"\{(\w+)\}", en[k])) == {"n", "s", "lo", "hi"}, k
    body = C.function_source("chDirLines")
    assert "chxSepClause()" in body and '"chb_syn"' in body
    assert 'if (!(mc.registrations || {}).chx2c || !r || r.reading !== "not_separable")' in (
        C.function_source("chxSepClause"))
    decl = C.PREREG / "FV.stat.v2.json"
    if decl.exists():                           # the registered wording, verbatim
        rd = json.loads(decl.read_text(encoding="utf-8"))["readings"]
        for lang, d in (("en", en), ("zh", zh)):
            sub = {"{p:.2f}": "{p}", "{p0:.2f}": "{p0}", "{q:.2f}": "{q}"}
            want = rd["wording"]["not_shown_weak"][lang]
            pas = rd["what_a_2028_pass_shows"][lang]
            for a, b in sub.items():
                want, pas = want.replace(a, b), pas.replace(a, b)
            assert d["evc_hyb_fvs"] == want and d["evc_hyb_fvs_pass"] == pas, lang
    assert "out.push(...fvsLines());" in C.function_source("evidenceItems")
    assert "const nl = n2rdLine(); if (nl)" in C.function_source("evidenceItems")
    decl = C.PREREG / "X5.n2_reading.v1.json"
    if decl.exists():                           # the registered wording, verbatim
        w = json.loads(decl.read_text(encoding="utf-8"))["thresholds"]["reading"]["wording"]
        want = w["informative"]
        for k in ("fp", "ws05", "ws2", "cov", "pw", "pwc", "pwf"):
            want = want.replace(f"{{{k}:.2f}}", f"{{{k}}}")
        assert en["evc_n2_rd"] == want
    assert en["evc_n2_rd"].endswith("lambda 0.8: power {pwf})")      # no gloss after it
    assert "不符合其門檻（< 0.25 / ≥ 0.80）" in zh["evc_n2_rd"] and "未達" not in zh["evc_n2_rd"]
    for d in (en, zh):
        assert not any("full-size response" in v or "完整大小" in v for v in d.values())
    for k in ("hl_pol", "def_tW", "rank_limit"):
        assert "×1.27" in en[k] and "×1.27" in zh[k], k
    assert "(S, design-time" in en["tW_r"] and "(D)" in en["tW_r"] and "(N;" in en["tW_r"]
    assert "Size (N, model-only)" in en["def_tW"] and "(D):" in en["def_tW"]
    # C27: registered numbers; the 'other way' / 'constant rate' readings are gone
    assert "{syl}" in en["evc_pr_subs"] and "{scu}" in zh["evc_pr_subs"]
    assert "ratio interval [{lo}, {hi}]" in en["pr_stepb"]
    # on leveling CL is a constant sinking rate, not 'no change' (N1.policy_response.v1)
    for k in ("pr_s_lost_both", "pr_s_lost_cl", "pr_s_lost_zr", "pr_s_beat"):
        assert "a constant sinking rate" in en[k] and "固定下陷速率" in zh[k], k
        assert "no change" not in en[k] and "不變" not in zh[k], k
    assert "against a constant sinking rate (CL)" in CL()["C27"]["wording_en"]
    for gone in ("pr_way_other", "pr_way_same", "pr_cl_better"):
        assert gone not in en and gone not in zh, gone
    assert not any("too large" in v for d in (en, zh) for v in d.values())
    # E24 (R8.11): Model internals, L, not an interval; tA's definition only
    assert en["def_tA_e24"].startswith("Model internals (L)") and zh["def_tA_e24"].startswith(
        "模型內部數值（L）")
    assert "(model-form spread, not an interval)" in en["def_tA_e24"]
    # SD v2's registered line_template: range, coast share, inland reading; Tier A beside A+B
    assert "coast (distal zone) {crng} % of it; {inl}" in en["def_tA_e24"]
    assert "{ta}" in en["def_tA_e24"] and "coast {crng} % of it" in en["def_tA_e24_a"]
    assert en["e24_inl_rob"] == "inland Yunlin robust within ×1.5"
    assert "const e24 = e24Line(); if (e24) p(e24);" in C.function_source("tileDefHtml")
    e24 = C.function_source("e24Line")                 # only for the run SD v2 measured
    assert 'PRESETS.find(q => q[0] === "irr30")' in e24 and "M.starts[S.start] !== 2026" in e24
    assert 'r.lo <= 0 ? "chx_2c_ns" : "chx_2c_ns_gen"' in C.function_source("chxSepClause")
    assert "reaching 0" not in en["chx_2c_ns_gen"]
    sc = L()["page_scoping"]
    assert "def_tA_e24" in sc["internals_labelled_keys"] and "def_tA_e24" in sc[
        "headline_must_not_use"]
    for fn in sc["answer_functions"]:
        assert "e24Line" not in C.function_closure(fn), fn
    # the FV v2 per-truth split stays off the page
    for bad in ("Q0 0.004", "R1c 0.992", "X5R1 0.996", "per truth"):
        assert not any(bad in v for v in en.values()), bad


# ---- ruling R8.7 (2026-11-06): forecast of record, C30 split, post-2028 wording ------------
POST2028 = {
    "C30-scen": ("before_scoring", "e02c_level_forecast_supported",
                 "e02c_level_forecast_not_supported", "e02c_not_evaluable",
                 "e02z_level_supported", "e02z_level_not_supported", "e02z_level_not_evaluable",
                 "n3_and_n3sl", "rider_caveat", "never"),
    "C30-fc": ("before_scoring", "e04_pass", "e04_fail", "e04_not_evaluable", "never"),
    "E02": ("before_scoring", "pass", "fail", "not_evaluable", "never"),
    "E21": ("before_scoring", "level_forecast_supported", "level_forecast_not_supported",
            "level_not_evaluable", "coverage_supported", "coverage_supported_weak_power",
            "coverage_not_supported", "coverage_not_evaluable",
            "pooled_gate_failed_county_report_only", "rider_caveat", "never"),
}


def test_r8_7_c30_is_split_into_the_scenario_and_the_forecast_of_record():
    """R8.7: C30 -> C30-scen (the physics average-weather scenario, N; C30's D sub-labels kept
    as notes and decided in their own rows) and C30-fc (E04 named the first-year forecast of
    record, D via E04). Lead decision on the review: plain N, status_before C30's own N."""
    cl, ids = CL(), {r["id"] for r in L()["rulings"]}
    assert _r8("R8.7") in ids and "C30" not in cl
    scen, fc = cl["C30-scen"], cl["C30-fc"]
    assert _st("C30-scen") == {"N"} and _st("C30-fc") == {"D"}
    assert scen["ruling"] == fc["ruling"] == _r8("R8.7")
    assert scen["status_before"] == fc["status_before"] == "N"
    assert "status N (ruling R8.7: 'C30-scen (N)')" in scen["wording_en"]
    for cid in ("C26", "C29", "C31", "E02", "E21", "E30"):        # the alias note
        assert "'C30' in this row means C30-scen" in cl[cid]["notes"], cid
    assert "C30-fc" in DM()["functions"]["renderTS"]
    assert "results/twin/prospective_2026-11_hybrid/protocol.json" in fc["cites"]["reports"]
    # the scenario keeps every C30 citation; the forecast row cites E04's manifest and FV.stat.v2
    for r in ("E02c.prospective_2026-11.v1", "E02z.level_years23.v1", "N3.piles_prospective.v1",
              "N3SL.township_prospective_roc116-117.v1", "FM.origin_memory.v1",
              "FV.decidability.v2_postfailure", "GOV.heads_window_product.v1"):
        assert r in scen["cites"]["registrations"], r
    assert fc["cites"]["manifests"] == ["results/twin/prospective_2026-11_hybrid/manifest.json"]
    assert "FV.stat.v2" in fc["cites"]["registrations"]
    assert "Wording before the split (C30)" in scen["wording_en"]
    assert "never makes the physics run a forecast or the forecast of record" in scen[
        "wording_en"]
    dm = DM()
    assert not any("C30" in v for sec in ("i18n", "functions", "card_rows", "arrays", "payload")
                   for v in dm[sec].values()), "a display site still names C30"
    for k in ("lg_proj", "lg_proj_scen", "ts_fm_mark", "evc_pro_e02z", "evc_pro_fm", "kv_proj"):
        where = dm["card_rows"] if k.startswith("kv_") else dm["i18n"]
        assert "C30-scen" in where[k] and "C30-fc" not in where[k], k
    for k in ("evc_hyb_for", "evc_hyb_fvs", "evc_hyb_fvs_pass"):
        assert dm["i18n"][k] == ["E04", "C30-fc"], k
    assert dm["i18n"]["evc_pro_e02z"] == ["E30", "C30-scen"]
    assert "i18n:evc_hyb_for" in fc["text_keys"] and "i18n:evc_hyb_for" in cl["E04"]["text_keys"]
    assert "i18n:evc_pro_e02z" in cl["E30"]["text_keys"] and "R8.7" in cl["E30"]["where"]


def test_r8_7_page_says_which_line_is_a_forecast_and_which_a_scenario_en_zh():
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert "an average-weather scenario after {lm}, not a forecast" in en["lg_proj_scen"]
    assert "平均天氣情境，並非預測" in zh["lg_proj_scen"] and "{lm}" in zh["lg_proj_scen"]
    for d in (en, zh):                      # the earlier legend wording is kept, with FM added
        assert d["lg_proj"].split("{y}")[0] in d["lg_proj_scen"]
    assert en["ts_fm_mark"] == "{lm}: start memory ends, in this model"
    assert "本模型中" in zh["ts_fm_mark"]
    assert "average-weather scenario, not a forecast" in en["ts_fm_note"]
    assert "forecast of record is the registered statistical layer (E04" in en["ts_fm_note"]
    assert en["evc_hyb_for"].startswith("Forecast of record for {a}–{b}: this registered "
                                        "statistical layer (E04), not the physics model")
    assert "正式預測" in zh["evc_hyb_for"] and "（E04）" in zh["evc_hyb_for"]
    for d in (en, zh):
        assert "R8.7" in d["evc_hyb_for"] and "{a}" in d["evc_hyb_for"]
        assert not re.search(r"\d", re.sub(r"\{[a-z]+\}|R8\.7|E04|–", "", d["evc_hyb_for"]))
    # the E02z sub-label is its registered before_scoring text
    sub = ("strict level rule years 2-3 (E02z), per county; scored not before 2029-01-15 "
           "(2027-11..2028-10) and 2030-01-15 (2028-11..2029-10)")
    assert sub in en["evc_pro_e02z"]
    decl = C.PREREG / "E02z.level_years23.v1.json"
    if decl.exists():
        pm = json.loads(decl.read_text(encoding="utf-8"))["page_mapping"]
        assert f"'{sub}'" in pm["before_scoring"]["C30-scen"]
    # wiring: each line waits for its record; the start memory is read from FM's report
    fm = C.function_source("fmMemory")
    assert "(mc.registrations || {}).fm" in fm and 'r.case !== "within_first_year"' in fm
    assert 'r.clause_case !== "window_beyond_memory"' in fm
    assert "2023-11" not in C.template_src()          # never typed in: lstar_month from FM
    lg = C.function_source("legend")
    assert "fmMemory()" in lg and 'tr("lg_proj_scen"' in lg and 'tr("lg_proj"' in lg
    ts = C.function_source("renderTS")
    assert 'S.ts === "head" && !fit ? fmMemory() : null' in ts and '"ts_fm_mark"' in ts
    assert '"ts_fm_note"' in ts
    ev = C.function_source("evidenceItems")
    assert ev.index('tr("evc_hyb_reg"') < ev.index('tr("evc_hyb_for"') < ev.index(
        "out.push(...fvsLines());")
    assert 'if (rg.e02z) out.push(["", tr("evc_pro_e02z")]);' in ev
    # the physics level is never called a forecast in the page's own (non-registered) words
    for k in ("evc_pro_e02c", "evc_pro_ext", "cv9_b_split"):
        assert "frozen run" in en[k] or "projection's mean forcing" in en[k], k
        assert not re.search(r"(?i)level forecast|the forecast passes|forecast's mean", en[k]), k
        assert "水位預測" not in zh[k] and "預測必須" not in zh[k] and "預報的平均" not in zh[k], k
    assert 'fd.unit === "m"' in lg                               # head fields only
    fr = C.function_source("forecastOfRecord")
    assert 'g.result === "failed"' in fr and 'g.result === "not_evaluable"' in fr
    assert "forecastOfRecord()" in ts and 'const fr = forecastOfRecord(); if (fr)' in ev
    assert "not drawn on the map or the series" in en["evc_hyb_for"]
    for fn in L()["page_scoping"]["answer_functions"]:     # the answer row is unchanged
        keys = C.function_keys(fn, follow=True)
        assert not keys & {"evc_hyb_for", "lg_proj_scen", "ts_fm_note", "evc_pro_e02z"}, fn


def test_r8_7_post_2028_wording_is_predeclared_for_every_branch():
    """R8.7: C30 (both rows), E02 and E21 carry the post-2028 wording of every branch, pass
    included, quoting each registration's own wording where one exists; none of it is shown
    before its outcome exists (no template key, no README or model-card sentence)."""
    cl = CL()
    for cid, keys in POST2028.items():
        od = cl[cid]["outcome_display"]
        assert set(keys) <= set(od), (cid, sorted(set(keys) - set(od)))
        assert "before any E02, E02c, E04 or E02z score" in od["declared"], cid
        assert "nothing here is shown" in od["declared"], cid
        assert od["never"], cid
        assert any(x.startswith("declared exception: registered artefact names") for x in
                       od["never"]), cid
    scen, fc, e21 = (cl[c]["outcome_display"] for c in ("C30-scen", "C30-fc", "E21"))
    assert "S-limited" in scen["e02c_level_forecast_supported"]
    assert "not a forecast" in scen["e02c_level_forecast_supported"]
    assert "the expected outcome" in scen["e02c_level_forecast_not_supported"]
    assert fc["e04_pass"].startswith("C30-fc -> S (prospective) for 2026-11..2027-10 only")
    assert fc["e04_fail"].startswith("C30-fc -> N, decided")
    assert any("never moves C30-fc" in x for x in fc["never"])
    assert any("never called a forecast" in x for x in scen["never"])
    decl = C.PREREG / "E02c.prospective_2026-11.v1.json"
    if decl.exists():                         # E02c's registered wording, verbatim
        pm = json.loads(decl.read_text(encoding="utf-8"))["page_mapping"]
        lw, cw = pm["T_level"]["wording"], pm["T_cov"]["wording"]
        assert f"'{lw['level_supported']}' [verbatim]" in scen["e02c_level_forecast_supported"]
        assert f"'{lw['level_not_supported']}' [verbatim]" in scen[
            "e02c_level_forecast_not_supported"]
        assert f"'{lw['level_not_evaluable']}' [verbatim]" in scen["e02c_not_evaluable"]
        assert f"'{lw['level_supported']}' [verbatim]" in e21["level_forecast_supported"]
        for k, w in (("coverage_supported", "cov_supported"),
                     ("coverage_supported_weak_power", "cov_supported_weak_power"),
                     ("coverage_not_supported", "cov_not_supported"),
                     ("coverage_not_evaluable", "cov_not_evaluable")):
            assert f"'{cw[w]}' [verbatim]" in e21[k], k
    decl = C.PREREG / "E02z.level_years23.v1.json"
    if decl.exists():
        pm = json.loads(decl.read_text(encoding="utf-8"))["page_mapping"]
        zw, zs = pm["wording"], pm["decidability_suffix"]
        for k in ("level_supported", "level_not_supported", "level_not_evaluable"):
            assert f"'{zw[k]}' [verbatim]" in scen[f"e02z_{k}"], k
            if k != "level_not_evaluable":
                assert f"'{zs['not_shown_weak']}'" in scen[f"e02z_{k}"], k
        assert f"'{zs['not_run']}' [verbatim]" in scen["e02z_level_not_evaluable"]
        assert "the same registered suffix" in scen["e02z_level_not_evaluable"]
    for k in ("e02z_level_supported", "e02z_level_not_supported"):
        assert "decidability_suffix.not_shown_weak" in scen[k], k
    assert "ledger, README and MODEL_CARD only, never on the page (ruling R8.8)" in e21[
        "coverage_supported"]
    decl = C.PREREG / "FV.stat.v2.json"
    if decl.exists():
        pas = json.loads(decl.read_text(encoding="utf-8"))["readings"][
            "what_a_2028_pass_shows"]["en"]
        assert pas.split(" What it cannot show")[0].split(": ")[0] in fc["e04_pass"]
        assert "a 2028 fail is the more informative outcome" in pas and \
            "a 2028 fail is the more informative outcome" in fc["e04_fail"]
    # nothing pre-declared is shown before its outcome exists
    shown = [C.template_src(), (C.REPO / "README.md").read_text(encoding="utf-8"),
             (C.REPO / "MODEL_CARD.md").read_text(encoding="utf-8")]
    for bad in ("passed its registered test on 2026-11..2027-10",
                "did not pass its registered test", "registered prospective test E02 passed",
                "registered prospective test E02 did not pass", "beat climatology and "
                "climatology-plus-trend on 2026-11..2027-10", "FW.rider.v1"):
        assert not any(bad in t for t in shown), bad


def test_r8_7_rain_year_rider_result_is_a_ledger_caveat_only():
    """FW.rider.v1, registered and scored once (2026-11-06): its declared caveat, verbatim,
    sits in the E21 and C30-scen notes and outcome_display only. It is cited as a model-only
    study, adds no S and no status change, and no page key, README or model-card sentence
    carries it (the R8.7 page wording is unchanged)."""
    cl, hashes = CL(), L()["registry_hashes"]
    rid, rep = "FW.rider.v1", "results/dataprep/reports/FW.rider.v1/report.json"
    assert hashes[rid].startswith("20e8839d")
    wording = ("Rain-year caveat (model-only rain-year rider FW.rider.v1): in the model's own "
               "replayed 2012-22 rain and detrended pumping years (FW.weather_replay.v2, the "
               "page ensemble), a model of the right kind passes E02c's level rule with "
               "probability 1.00 (90 % interval over the 11 recorded years 1.00-1.00; single "
               "sequences 0.99-1.00); the replayed rain years do not show the level test weak. "
               "One-sided: this adds no other reading, and P is not the chance of a 2028 pass.")
    path = C.REPO / rep
    if path.exists():                       # the scored report, verbatim and at its sha
        r = json.loads(path.read_text(encoding="utf-8"))
        assert r["wording"]["en"] == wording and r["status"] == "scored"
        assert r["reading"] == "not_shown_weak_in_replayed_years"
        assert C.sha256(path).startswith("377afa62")
        assert r["declaration_sha256"] == hashes[rid]
    for cid in ("E21", "C30-scen"):
        c = cl[cid]
        for blob in (c["notes"], c["outcome_display"]["rider_caveat"]):
            assert f"'{wording}'" in blob, cid
            for k in ("sha 377afa62", "registered sha 20e8839d",
                      "'not_shown_weak_in_replayed_years'", "P 0.9989",
                      "FWR:accidental_smoke_compute_20261106",
                      "Ledger only; never on the page; no status change"):
                assert k in blob, (cid, k)
            assert "SLOT (" not in blob, cid
        assert rid in c["cites"]["registrations"] and rep in c["cites"]["reports"], cid
    assert _st("E21") == {"D"} and _st("C30-scen") == {"N"}          # no status change
    for c in L()["claims"]:
        assert rid not in c["wording_en"], c["id"]
        if rid in c["cites"]["registrations"]:
            assert "S" not in _st(c["id"]), c["id"]
            assert c["id"] in ("E21", "C30-scen"), c["id"]
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    assert not any(rid in v or "Rain-year caveat" in v for d in (en, zh) for v in d.values())
    assert rid not in C.template_src() and not any(rid in k for k in DM()["i18n"])


def test_round8_changhua_2008_09_window_is_closed_as_not_decidable():
    """Round-8 #18: the count-only look on Changhua ls_cache 2008-09 stopped; its declared
    wording is recorded on the rows that track the Changhua head windows (E26, C20, C21) and in
    README Limits, with no status change and no page line."""
    cl, wording = CL(), "'last possible Changhua head window checked, not decidable'"
    rep = "results/twin_runs/chx/backcast_2008_09/avail/avail.json"
    for cid in ("E26", "C20", "C21"):
        n = cl[cid]["notes"]
        for k in (wording, "I1CHB0:ls_cache_rowcounts_2008_2009", "found 0 of 24 Changhua",
                  ">= 18 available months in 2008-01..2009-12", "constants sha256 d4b9ab77",
                  "sha 508fd440", "P6:moea_public_well_tables",
                  "moea_public_well_tables:2008-01..2024-12", "no status change"):
            assert k in n, (cid, k)
        assert rep in cl[cid]["cites"]["reports"], cid
    assert _st("E26") == {"S"} and _st("C20") == {"N"} and _st("C21") == {"N"}
    readme = " ".join((C.REPO / "README.md").read_text(encoding="utf-8").split())
    assert ("so the last possible Changhua head window is checked and not decidable" in readme)
    src = C.template_src()
    assert "I1CHB0" not in src and "2008-09" not in src and "2008–09" not in src
    path = C.REPO / rep
    if path.exists():                        # the stopped result, as recorded
        f = json.loads(path.read_text(encoding="utf-8"))["feasibility"]
        assert C.sha256(path).startswith("508fd440")
        assert f["passed"] is False and f["n_eligible"] == 0 and f["nest_included"]["n"] == 0
        assert f["n_changhua_candidates"] == 24 and f["constants_sha256"].startswith("d4b9ab77")
        assert f["on_stop"] == wording.strip("'")


def test_round8_2010_11_backcast_is_closed_by_ruling_r8_15():
    """Round-8 #12: the 2010-11 backcast's synthetic G0 failed (power 0.53 < 0.6), so by user
    ruling R8.15 it stopped before any 2010-11 head was read. The declared line is on E04, E07
    and E26 and in README Limits; R8.15 is a recorded ruling; no status changes."""
    cl, rid = CL(), "user ruling 2026-11-06 (thirteenth set) (R8.15)"
    line = ("I1.backcast_2010_11.v1 not decidable by design: its primary's synthetic G0 gives "
            "power 0.53 < 0.6 (false pass 0.06), so the whole test would be report-only; by "
            "user ruling R8.15 it was stopped before any 2010-11 head was read (window unspent).")
    rulings = {r["id"]: r["text"] for r in L()["rulings"]}
    assert rid in rulings and "user_rulings_2026-11-06.json (sha256 c2d3b0f3)" in rulings[rid]
    assert "Stop before any 2010-11 head is read" in rulings[rid]
    g0 = "results/twin_runs/chx/backcast_2010_11/g0/g0_g1.json"
    for cid in ("E04", "E07", "E26"):
        c = cl[cid]
        for k in (line, "sha b0622bc9", "sha 1f592b31", "sha c2d3b0f3", "seed 71120611",
                  "I1BC11:ls_cache_dates_2009_2011", "no status change"):
            assert k in c["notes"], (cid, k)
        assert g0 in c["cites"]["reports"] and rid in c["ruling_also"], cid
    assert _st("E04") == {"D"} and _st("E07") == {"N"} and _st("E26") == {"S"}
    readme = " ".join((C.REPO / "README.md").read_text(encoding="utf-8").split())
    assert line.split(": ", 1)[1] in readme and "`I1.backcast_2010_11.v1`" in readme
    assert "I1.backcast_2010_11" not in C.template_src()
    assert "I1.backcast_2010_11.v1" not in L()["registry_hashes"]       # never registered
    path = C.REPO / g0
    if path.exists():                      # the failed gate, as recorded
        g = json.loads(path.read_text(encoding="utf-8"))
        assert C.sha256(path).startswith("b0622bc9") and g["gate"]["decidable"] is False
        assert round(g["gate"]["power"], 2) == 0.53 and round(g["gate"]["false_pass"], 2) == 0.06


# ---- ruling R8.9 (2026-11-06 wiring): all30 is the headline head lever ----------------------
ALL30_SIDECAR = C.REPO / "results/twin_forward/all30_page.provenance.json"
CLASS_CLAUSE = {"en": "size depends on the pumping conversion (×1.27 Yunlin; "
                      "X5.lever_class_share.v1, report-only)",
                "zh": "大小取決於抽水換算（雲林 ×1.27；X5.lever_class_share.v1，僅供報告）"}


def test_r8_9_all30_is_the_headline_lever_and_the_class_levers_are_labelled_en_zh():
    """R8.9: all30 (all-class x0.7) at N2's unit is the headline; cut30 and retire_aqua are
    class scenarios labelled 'size depends on the pumping conversion'. No status moves."""
    cl, ids = CL(), {r["id"] for r in L()["rulings"]}
    assert _r8("R8.9") in ids
    for cid, st in (("C01", {"S"}), ("C01-w", {"D"}), ("C01-sz", {"N"}), ("C12", {"L"})):
        assert _r8("R8.9") in cl[cid]["ruling_also"], cid
        assert _st(cid) == st, cid                                   # nothing promoted
    assert "x1.27 Yunlin; X5.lever_class_share.v1, report-only" in cl["C01-sz"]["wording_en"]
    assert "all30" in cl["C01"]["wording_en"] and "all30" in cl["C12"]["notes"]
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    for k in ("tW_cls", "hl_pol", "def_tW", "rank_limit"):
        assert CLASS_CLAUSE["en"] in en[k] and CLASS_CLAUSE["zh"] in zh[k], k
    for k in ("hl_w30", "tW_w30", "def_tW", "hl_pol"):          # the headline lever, both languages
        assert ("all pumping" in en[k] or "a cut of all pumping" in en[k]) and "全部抽水" in zh[k], k
    assert "×1.03–1.05" in en["hl_w30"] and "×1.03–1.05" in zh["hl_w30"]
    assert "N2 decides the within-window response, not this size" in en["hl_w30"]
    js = C.template_js()
    assert 'const PRESETS = [["all30", [0.7,0.7,0.7,0.7,0.7,0.7]],' in js
    assert "isW30(f, startYear)" in C.function_source("policyInto")
    assert "w30Sentence" in C.function_closure("renderHeadline")
    assert {"hl_w30", "hw_n2"} <= C.function_keys("renderHeadline", follow=True)
    assert {"tW_w30", "tW_cls", "tW_w30_src"} <= C.function_keys("renderTiles")
    blk = DM()["payload_conditional"]["all30"]
    assert blk["source"] == "all30" and "n2Unit.all30.series[]" in blk["paths"]
    # the bound forward stays the page's and N2's
    assert L()["page"]["forward_sha256"] == L()["page"]["n2_primary_forward_sha256"] == (
        "cf2c7154160d9e4626d1acc5b5c3abd7866e09c3539a9249c657dd6b39007228")


def test_r8_9_the_all30_number_is_recorded_from_its_file_once_it_exists():
    """The all30 number is model output: C01-sz's notes carry a SLOT until the lead's forward
    exists, then its Dec-2032 value at N2's unit and the file's sha256 from the sidecar."""
    notes = CL()["C01-sz"]["notes"]
    side = json.loads(ALL30_SIDECAR.read_text()) if ALL30_SIDECAR.exists() else None
    if side and side.get("status") == "ok":
        v, sha = side["n2_unit"]["dec2032_m"]["all30"], side["outputs_sha256"]["npz"]
        assert "SLOT (rank 14" not in notes, "fill the all30 number from its sidecar"
        assert "Rank 14 result" in notes and f"{v:+.2f} m" in notes and sha[:8] in notes
    else:
        assert "SLOT (rank 14" in notes or "Rank 14 result" in notes


# ---- round 8 #20 (2026-11-06): the head-contrast model-form spread note on C14d --------------
HC_ARMS = ("chc", "chs", "cdm", "csw", "ccs", "rs", "x5r1t", "ccz", "q0c", "x5r1")
HC_REPORT = C.REPO / "results/dataprep/reports/X5.changhua_lever_model.v1/report.json"


def _hc_text(rep: dict) -> dict[str, str]:
    """The ranges the note quotes, recomputed from the scored report (window means)."""
    out = {}
    for county, key in (("yunlin", "L_yunlin_m_report_only"), ("changhua", "L_m")):
        for w in ("2026-27", "2030-32"):
            v = [rep[key][a][w] for a in HC_ARMS]
            out[f"{county}/{w}"] = f"{min(v):.2f}-{max(v):.2f}"
            out[f"{county}/{w}/x"] = f"x{max(v) / min(v):.2f}"
    return out


def test_c14d_head_contrast_spread_is_a_ledger_and_readme_note_only():
    """#20: the model-form spread of the all30 head contrast over the existing forwards, with
    the x1.27 / x1.39 conversion disclosure, in C14d's notes and the README; C14d stays L, the
    page carries none of it, and the numbers are the scored report's."""
    c = CL()["C14d"]
    assert _st("C14d") == {"L"} and c["status_before"] == "N"
    n = c["notes"]
    assert "model-form spread, not an interval" in n and "excludes response-size error" in n
    assert "x1.27 in Yunlin and x1.39 in Changhua" in n and "x1.03-1.05 for all30" in n
    assert "untested (N2C registered, scored not before 2028-01-15)" in n
    assert "withdrawn under every outcome" in n and "the page's own all30 run is not" in n
    assert {"X5.changhua_lever_model.v1", "X5.lever_class_share.v1"} <= set(
        c["cites"]["registrations"])
    readme = (C.REPO / "README.md").read_text(encoding="utf-8")
    quoted = {"yunlin/2026-27": "2.21-2.60", "yunlin/2030-32": "2.44-2.85",
              "changhua/2026-27": "2.10-2.73", "changhua/2030-32": "2.48-3.28"}
    flat = " ".join(readme.split())
    for k, v in quoted.items():
        assert v in n and v in flat, k
    assert "x1.18 and x1.17" in flat and "(x1.30, x1.33)" in flat
    src = C.template_src()                               # no page change
    assert not any(v in src for v in quoted.values()) and "x1.39" not in src
    if HC_REPORT.exists():                               # verbatim against the scored report
        got = _hc_text(json.loads(HC_REPORT.read_text(encoding="utf-8")))
        for k, v in quoted.items():
            assert got[k] == v, (k, got[k])
        assert [got[f"{k}/x"] for k in quoted] == ["x1.18", "x1.17", "x1.30", "x1.33"]
