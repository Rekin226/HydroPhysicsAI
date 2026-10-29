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
absent (CI).
"""
from __future__ import annotations

import re

import pytest

from hydrophysics.twin import claims as C

LEDGER = C.load_ledger()
CLAIMS = {c["id"]: c for c in LEDGER["claims"]}
DMAP = LEDGER["display_map"]
PAGE = C.REPO / LEDGER["page"]["html"]
INDEX = C.PREREG / "INDEX.jsonl"
NEED_PAGE = pytest.mark.skipif(not PAGE.exists(), reason=f"{LEDGER['page']['html']} is a "
                               "gitignored build, absent here")
NEED_REGISTRY = pytest.mark.skipif(not INDEX.exists(), reason="the pre-registration index "
                                   "(results/dataprep/preregistration) is gitignored, absent here")
BAD = {"U", "X"}


def _st(cid: str) -> set[str]:
    return set(C.codes(CLAIMS[cid]["status_current"]))


# ---- the ledger itself -----------------------------------------------------------------------
def test_ids_unique_and_status_codes_valid():
    ids = [c["id"] for c in LEDGER["claims"]]
    assert len(ids) == len(set(ids))
    for c in LEDGER["claims"]:
        for col in ("status_before", "status_current"):
            parts = C.codes(c[col])
            assert parts and all(p in C.CODES for p in parts), (c["id"], col, c[col])
        assert c["wording_en"].strip(), c["id"]
        assert c["text_keys"], f"{c['id']} has no displayed text key"


def test_no_claim_is_unsupported_or_inconsistent_with_the_record():
    bad = sorted(c["id"] for c in LEDGER["claims"] if set(C.codes(c["status_current"])) & BAD)
    assert not bad, f"claims still U or X: {bad}"


def test_a_status_change_is_a_wording_change():
    for c in LEDGER["claims"]:
        if c["status_before"] != c["status_current"] or set(C.codes(c["status_before"])) & BAD:
            assert c["changes"], f"{c['id']} changed status but records no wording change"
            assert not c["wording_en"].startswith("Unchanged"), c["id"]
        if not c["changes"]:
            assert c["wording_en"].startswith(("Unchanged", "Not a claim", "Not displayed")), c["id"]


def test_status_changes_rest_on_a_user_ruling():
    """Every status change (not only a withdrawal) cites a recorded ruling."""
    rulings = {r["id"] for r in LEDGER["rulings"]}
    for c in LEDGER["claims"]:
        if c["status_before"] != c["status_current"]:
            assert c["ruling"] in rulings, f"{c['id']} changed status without a ruling"
        for r in (c.get("ruling"), c.get("ruling_also")):
            assert r is None or r in rulings, (c["id"], r)


def test_supported_and_decided_claims_cite_their_registration():
    for c in LEDGER["claims"]:
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
    hashes = LEDGER["registry_hashes"]
    for c in LEDGER["claims"]:
        ci = c["cites"]
        for r in ci["registrations"]:
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
    import json

    n2 = json.loads((C.PREREG / "N2.policy_response.prospective_2026-11.v1.json").read_text())
    assert n2["primary_model"]["forward_sha256"] == LEDGER["page"]["n2_primary_forward_sha256"]
    assert LEDGER["page"]["forward_sha256"] == LEDGER["page"]["n2_primary_forward_sha256"]


# ---- every display site maps to a ledger id --------------------------------------------------
def _ids_ok(where: str, mapping: dict):
    bad = {k: v for k, v in mapping.items() if not v or any(i not in CLAIMS for i in v)}
    assert not bad, f"{where}: unknown or empty ids {bad}"


def test_ledger_is_pinned_to_this_template():
    assert C.sha256(C.TEMPLATE) == LEDGER["page"]["template_sha256"], (
        "the page template changed: re-review the claims ledger and update template_sha256")


def test_every_numeric_dictionary_key_is_mapped():
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    numeric, m = C.displayed_numeric_keys(), DMAP["i18n"]
    assert not sorted(numeric - set(m)), f"unmapped numeric keys: {sorted(numeric - set(m))}"
    assert not sorted(set(m) - numeric), f"stale keys in the ledger: {sorted(set(m) - numeric)}"
    assert set(m) <= set(zh) and set(m) <= set(en), "EN/ZH key parity broken for mapped keys"
    _ids_ok("i18n", m)


def test_every_number_formatting_function_is_mapped():
    fns, m = C.number_functions(), DMAP["functions"]
    assert not sorted(set(fns) - set(m)), f"unmapped functions: {sorted(set(fns) - set(m))}"
    assert not sorted(set(m) - set(fns)), f"stale functions: {sorted(set(m) - set(fns))}"
    _ids_ok("functions", m)


def test_every_model_card_row_and_tile_is_mapped():
    rows = C.card_rows()
    assert not sorted(rows - set(DMAP["card_rows"])), sorted(rows - set(DMAP["card_rows"]))
    ids, _ = C.tiles()
    assert set(ids) == set(DMAP["tiles"]), (ids, sorted(DMAP["tiles"]))
    _ids_ok("card_rows", DMAP["card_rows"])
    _ids_ok("tiles", DMAP["tiles"])


def test_mapped_dictionary_keys_are_used_by_the_page():
    src = C.template_src()
    t0 = src.index("const T = {")
    body = src[:t0] + src[src.index("\n}};", t0):]          # markup + code, dictionary removed
    computed = ("ly_", "pl_", "pr_")                         # tr("ly_" + id), tr("pl_" + key), ...
    for p in computed:
        assert f'"{p}"' in body, f"computed prefix {p} no longer used"
    unused = [k for k in DMAP["i18n"]
              if not re.search(r'"' + re.escape(k) + r'"', body) and not k.startswith(computed)]
    assert not unused, f"mapped keys the page never asks for: {unused}"


# ---- the scoping gate (reads the template) ---------------------------------------------------
def test_the_answer_row_carries_no_withdrawn_or_unsupported_claim():
    ids, primary = C.tiles()
    allowed = set(LEDGER["page_scoping"]["answer_tiles_allowed"])   # L: the labelled cost tile
    assert not allowed & {"W", "U", "X"}, allowed
    for tid in ids[:primary]:
        for cid in DMAP["tiles"][tid]:
            assert _st(cid) <= allowed, f"answer tile {tid} rests on {cid} ({CLAIMS[cid]['status_current']})"
    for tid in ids[primary:]:
        assert any(_st(c) & {"W", "L"} for c in DMAP["tiles"][tid]), tid
    # every withdrawn claim that owns a tile sits in the model-internals row
    for cid in CLAIMS:
        if "W" in _st(cid):
            for tid, cl in DMAP["tiles"].items():
                if cid in cl:
                    assert ids.index(tid) >= primary, f"withdrawn {cid} is in the answer row ({tid})"


def test_the_headline_uses_no_withdrawn_or_unsupported_claim():
    """The answer functions (headline, evidence line, start note, policy bar), with every
    top-level function they call, ask for no key of a W/U/X claim."""
    sc = LEDGER["page_scoping"]
    computed = sc["answer_computed_keys"]
    for fn in sc["answer_functions"]:
        keys = C.function_keys(fn, follow=True)
        for expr, ks in computed.items():
            if expr in C.computed_tr_calls(C.function_source(fn, follow=True)):
                keys |= set(ks)
        for k in keys:
            for cid in DMAP["i18n"].get(k, []):
                assert not _st(cid) & {"W", "U", "X"}, f"{fn} shows {k} -> {cid}"
        bad = keys & set(sc["headline_must_not_use"])
        assert not bad, (fn, bad)


def test_answer_functions_have_no_unreadable_keys():
    """The documented limit of the key scan: a tr() key built at run time (a lookup table or a
    concatenation) is invisible to claims.tr_keys. Each such call in the answer functions must
    be listed in page_scoping.answer_computed_keys with the keys it can take."""
    sc = LEDGER["page_scoping"]
    for fn in sc["answer_functions"]:
        for expr in C.computed_tr_calls(C.function_source(fn, follow=True)):
            assert expr in sc["answer_computed_keys"], f"{fn}: unlisted run-time key tr({expr})"
    en = C.i18n(lang="en")
    for ks in sc["answer_computed_keys"].values():
        assert set(ks) <= set(en), ks


def test_answer_functions_do_not_read_withdrawn_quantities():
    sc = LEDGER["page_scoping"]
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
    for k in LEDGER["page_scoping"]["internals_labelled_keys"]:
        assert "Model internals" in en[k], k
        assert "模型內部" in zh[k], k


def test_template_anchors_hold():
    """Ruling (4): 'no calibrated correction factor follows' stays in the head definition."""
    en, zh = C.i18n(lang="en"), C.i18n(lang="zh")
    for k, rule in LEDGER["page_scoping"]["template_anchors"].items():
        assert rule["en"] in en[k], (k, rule["en"])
        assert rule["zh"] in zh[k], (k, rule["zh"])


def test_the_page_opens_on_the_tested_quantity():
    js = C.template_js()
    sc = LEDGER["page_scoping"]
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
    for bad in LEDGER["page_scoping"]["template_absent"]:
        assert bad not in src, bad
    for p in ("hydrophysics/twin/viewer_app.py", "hydrophysics/twin/kfold_scores.py"):
        assert "pre-registered 2026-09-2" not in (C.REPO / p).read_text(encoding="utf-8"), p


def test_doc_anchors_hold():
    for c in LEDGER["claims"]:
        for doc, rule in c.get("doc_anchors", {}).items():
            text = (C.REPO / doc).read_text(encoding="utf-8")
            for s in rule.get("present", []):
                assert s in text, f"{c['id']}: {doc} lost {s!r}"
            for s in rule.get("absent", []):
                assert s not in text, f"{c['id']}: {doc} still says {s!r}"


# ---- the built page (gitignored) ------------------------------------------------------------
@NEED_PAGE
def test_every_payload_array_is_mapped():
    D = C.payload_structure(PAGE)
    pats = DMAP["arrays"]
    _ids_ok("arrays", pats)
    for k in D["arrays"]:
        assert any(re.fullmatch(re.escape(p).replace(r"\*", ".*"), k) for p in pats), k


@NEED_PAGE
def test_every_numeric_payload_path_is_mapped_exactly():
    D = C.payload_structure(PAGE)
    paths = {p for p in C.numeric_paths(D) if not p.startswith("arrays.")}
    m = DMAP["payload"]
    assert not sorted(paths - set(m)), f"payload numbers with no ledger id: {sorted(paths - set(m))}"
    assert not sorted(set(m) - paths), f"stale payload paths: {sorted(set(m) - paths)}"
    _ids_ok("payload", m)
    js = C.template_js()
    named = [p for p, v in m.items() if v == ["Z04"] and C.js_names_leaf(js, p)]
    assert not named, f"paths marked 'not displayed' that the page code names: {named}"


@NEED_PAGE
def test_page_is_built_from_the_pinned_forward_run():
    D = C.payload_structure(PAGE)
    assert D["release"]["sources"]["forward"]["sha256"] == LEDGER["page"]["forward_sha256"]
