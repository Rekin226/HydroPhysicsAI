"""CPU tests of U1.no_band.v1 (prereg_u1_no_band.py): the sub-claim scheme, the reason
checker (on the real manifests and registrations when present, and on synthetic files in
pytest's tmp dirs) and the band share. Nothing is registered in the real registry.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
X = TESTS.parent
REPO = X.parents[1]
for p in (str(X), str(REPO / "dataprep_staging")):
    if p not in sys.path:
        sys.path.insert(0, p)
import prereg_u1_no_band as N  # noqa: E402

HAVE_RESULTS = (REPO / "results/dataprep/preregistration/INDEX.jsonl").exists()
needs_results = pytest.mark.skipif(not HAVE_RESULTS, reason="gitignored results/ absent")


def test_subclaim_scheme_and_statuses():
    d = N.doc()
    assert N.structure_problems(d) == []
    s = d["subclaims"]
    assert sorted(s) == ["C14a", "C14b", "C14c", "C14d", "C14e"]
    assert [s[k]["status"] for k in sorted(s)] == ["N+D", "N", "N", "L", "W"]
    assert s["C14e"]["internals_status"] == "L" and N.ledger_status(s["C14e"]) == "W+L"
    assert all(N.status_ok(v["status"]) for v in d["related_claims"].values())
    assert d["related_claims"]["C30"]["status"] == "N+D"
    assert "cell" not in s["C14b"]["what"].split("(")[0]          # cells belong to C14c
    assert "half-width" in s["C14a"]["answer"] and "no aggregate" in s["C14a"]["answer"]
    assert [N.status_ok(x) for x in ("N", "N+D", "W+L", "N+N", "X", "", "N+")] == [
        True, True, True, False, False, False, False]


def test_band_wording_is_never_an_interval_claim():
    s = N.SUBCLAIMS
    assert s["C14c"]["wording"]["en"] == "run spread, a parameter spread, not a predictive interval"
    assert s["C14d"]["wording"]["en"] == "a parameter spread; it excludes response-size error"
    assert "not a predictive interval" in s["C14e"]["wording"]["en"]
    assert "untested" in s["C14b"]["wording"]["en"]
    for k in ("C14b", "C14c", "C14d", "C14e"):
        assert s[k]["wording"]["zh"], k
    assert s["C14a"]["label_u7"]["en"] == ("the interval family was chosen after its 2023-26 "
                                           "coverage was seen; the 2026-11..2027-10 window is "
                                           "untouched")
    c30 = N.RELATED["C30"]["legend_band_sentence"]
    assert "no uncertainty band, since no held data can calibrate one" in c30["en"]
    assert c30["zh"].startswith("推估值")
    assert "x0.67" not in json.dumps(N.doc()).replace("no x0.67-1.5 band is drawn", "")


def test_changhua_decision_carries_the_n2c_label():
    d7 = [x for x in N.DECISIONS if x["n"] == 7][0]
    assert N.N2C_LABEL == "untested (N2C registered, scored not before 2028-01-15)"
    assert N.N2C_LABEL in d7["wording"]["en"] and N.N2C_LABEL in d7["decision"]
    assert "does not prejudge N2C" in d7["decision"]


def test_every_reason_set_has_a_machine_checked_reason():
    rows = N.check_reasons(N.doc())
    assert not [r for r in rows if r["type"] == "structure"]
    assert not [r for r in rows if r["status"] == "fail" and r["type"] in ("code", "context")]


def test_subsidence_scorers_cite_current_release_and_unlisted_vs_forbidden():
    rs = N.SUBCLAIMS["C14e"]["reasons"]
    ms = {r["manifest"]: r for r in rs if r["type"] == "manifest_use"}
    pre = "results/dataprep/model_ready/release_20261012b2/preorigin/piles_printed/v1/MANIFEST.json"
    assert ms[pre]["allowed_equals"] == ["validator", "after_the_fact_analysis"]
    deep = "results/dataprep/deep_benchmarks_printed/v2/MANIFEST.json"
    assert ms[deep]["expect"] == "unlisted" and "not in allowed_uses" in ms[deep]["why"]
    assert {r["id"] for r in rs if r["type"] == "registration"} >= {"N1.policy_response.v1",
                                                                    "I1.X5SC.gate0.v1"}
    a0 = [r for r in N.SUBCLAIMS["C14d"]["reasons"] if r.get("path", "").endswith(
        "A0/verdict.json")]
    assert a0, "C14d cites A0"


@needs_results
def test_every_reason_resolves_on_the_held_files():
    rows = N.check_reasons(N.doc())
    bad = [r for r in rows if r["status"] in ("fail", "missing")]
    assert bad == []
    assert sum(r["status"] == "ok" for r in rows) >= 40


# ----------------------------------------------------------------------------- checker
def _manifest(tmp: Path, allowed, forbidden, **extra) -> str:
    p = tmp / "m" / "MANIFEST.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"allowed_uses": allowed, "forbidden_uses": forbidden, **extra}))
    return str(p)


def test_checker_on_synthetic_manifests(tmp_path):
    m = _manifest(tmp_path, ["validator"], ["prospective_scorer"], gate={"passed": False})
    ok = lambda r: N.check_reason(r)["status"]  # noqa: E731
    base = {"type": "manifest_use", "manifest": m}
    assert ok({**base, "use": "prospective_scorer", "expect": "forbidden"}) == "ok"
    assert ok({**base, "use": "prospective_scorer", "expect": "allowed"}) == "fail"
    assert ok({**base, "use": "scoring_target", "expect": "unlisted"}) == "ok"
    assert ok({**base, "use": "prospective_scorer", "expect": "unlisted"}) == "fail"
    assert ok({**base, "use": "validator", "expect": "allowed",
               "allowed_equals": ["validator"]}) == "ok"
    assert ok({**base, "use": "validator", "expect": "allowed",
               "allowed_equals": ["validator", "x"]}) == "fail"
    assert ok({"type": "manifest_field", "manifest": m, "field": "gate.passed",
               "equals": False}) == "ok"
    assert ok({"type": "manifest_field", "manifest": m, "field": "gate.nope",
               "equals": False}) == "fail"
    assert ok({**base, "manifest": str(tmp_path / "none.json"), "use": "x",
               "expect": "forbidden"}) == "missing"


def test_checker_on_results_and_lists(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"a": {"b.c": [], "s": "FAIL: x"}, "n": 4}))
    ok = lambda r: N.check_reason(r)["status"]  # noqa: E731
    assert ok({"type": "result", "path": str(p), "field": "n", "equals": 4}) == "ok"
    assert ok({"type": "result", "path": str(p), "field": "n", "equals": 5}) == "fail"
    assert ok({"type": "result", "path": str(p), "field": ["a", "b.c"], "equals": []}) == "ok"
    assert ok({"type": "result", "path": str(p), "field": "a.s", "startswith": "FAIL"}) == "ok"
    assert ok({"type": "result", "path": str(p), "field": "a.s", "contains": "PASS"}) == "fail"
    assert ok({"type": "context", "text": "t", "source": "s"}) == "context"
    assert ok({"type": "bogus"}) == "fail"


def test_checker_on_a_tmp_registry(tmp_path):
    from harness import prereg3

    reg = tmp_path / "prereg"
    reg.mkdir()
    body = (json.dumps({"id": "T.x.v1", "thresholds": {"k": 20}, "note": "rolling a b"},
                       sort_keys=True) + "\n").encode()
    (reg / "T.x.v1.json").write_bytes(body)
    prereg3._v1._append_chained(reg / "INDEX.jsonl", {
        "id": "T.x.v1", "sha256": hashlib.sha256(body).hexdigest(), "test_ids": []})
    res = tmp_path / "res.json"
    res.write_text(json.dumps({"prereg": "T.x.v1"}))
    r = {"type": "registration", "id": "T.x.v1", "result": str(res)}
    st = lambda x: N.check_reason(x, reg)["status"]  # noqa: E731
    assert st(r) == "ok"
    assert st({**r, "reg_field": "thresholds.k", "reg_equals": 20}) == "ok"
    assert st({**r, "reg_field": "thresholds.k", "reg_equals": 19}) == "fail"
    assert st({**r, "reg_contains": "rolling a b"}) == "ok"
    assert st({**r, "reg_contains": "nope"}) == "fail"
    assert st({**r, "result_field": "prereg", "result_equals": "T.x.v1"}) == "ok"
    assert st({**r, "id": "T.missing.v1"}) == "fail"
    assert st({**r, "result": str(tmp_path / "gone.json")}) == "missing"
    (reg / "T.x.v1.json").write_bytes(body + b" ")            # tampered file
    assert st(r) == "fail"


def test_a_reason_set_without_a_machine_reason_fails_the_structure():
    d = N.doc()
    d["subclaims"]["C14b"] = dict(d["subclaims"]["C14b"],
                                  reasons=[{"type": "context", "text": "t", "source": "s"}])
    rows = N.check_reasons(d)
    assert [r["where"] for r in rows if r["type"] == "structure"] == ["subclaims.C14b"]


def test_code_checks():
    for r in N.THRESHOLDS:
        if r["type"] == "code":
            assert N.check_reason(r)["status"] == "ok", r
    assert N.check_reason({"type": "code", "check": "template_has_keys",
                           "keys": ["no_such_key_xyz"], "equals": True})["status"] == "fail"


# ----------------------------------------------------------------------------- share, CLI
def test_band_share_is_reported_before_and_after():
    s = N.band_share()
    assert s["answer_outputs"] > 10 and s["not_answer_outputs_in_map"] == []
    assert s["before"]["registered_coverage"]["share"] == 0.0
    assert s["after"]["any_band"]["share"] < s["before"]["any_band"]["share"]
    assert "not evidence" in s["reading"]
    assert "LABELLING change" in s["registered_coverage_note"]
    assert "not new band coverage" in N.doc()["band_share"]["note"]


@needs_results
def test_validate_cli_is_a_dry_run(capsys):
    if (REPO / "results/dataprep/preregistration/U1.no_band.v1.json").exists():
        pytest.skip("U1.no_band.v1 is registered; the dry-run check applies before registration")
    before = sorted((REPO / "results/dataprep/preregistration").glob("U1.*"))
    aligned = N.page_alignment_problems() == []
    assert N.main(["--validate", "--quiet"]) == (0 if aligned else 1)
    out = capsys.readouterr().out
    assert "validate_prereg: valid" in out and "0 failing or missing" in out
    assert "page alignment (ledger + template):" in out
    assert N.main(["--write", "--expect-self-sha", "0" * 64]) == 1
    assert sorted((REPO / "results/dataprep/preregistration").glob("U1.*")) == before


@needs_results
def test_write_into_a_tmp_registry(tmp_path):
    from harness import prereg3

    path, h = prereg3.write_prereg(N.doc(), prereg_dir=tmp_path / "p",
                                   rows_dir=tmp_path / "rows", registries=[])
    got = json.loads(path.read_text())
    assert got["id"] == "U1.no_band.v1" and got["kind"] == "declaration"
    assert {x["path"] for x in got["inputs"]} >= {N.LEDGER, N.TEMPLATE}


# ----------------------------------------------------------------------------- page alignment
def _aligned_files(tmp: Path) -> tuple[Path, Path]:
    """A minimal ledger and template that agree with the declaration."""
    claims = []
    for k in N._ALIGNED_IDS:
        sc = N.SUBCLAIMS.get(k) or N.RELATED[k]
        claims.append({"id": k, "status_current": N.ledger_status(sc),
                       "cites": {"registrations": [N.ID]},
                       "wording_en": " ".join(N.PAGE_ALIGNMENT["ledger"]["wording_contains"]
                                              .get(k, [])),
                       "notes": " ".join(N.PAGE_ALIGNMENT["ledger"]["notes_contains"]
                                         .get(k, []))})
    led = tmp / "claims.json"
    led.write_text(json.dumps({"claims": claims}, ensure_ascii=False), encoding="utf-8")
    en, zh = [], []
    for x in N.PAGE_ALIGNMENT["template"]["contains"]:
        en.append(f'  {x["key"]}: {json.dumps(" ".join(x["en"]))},')
        zh.append(f'  {x["key"]}: {json.dumps(" ".join(x["zh"]), ensure_ascii=False)},')
    en.append('  evc_band_none: "No band is shown for subsidence or policy contrasts.",')
    zh.append('  evc_band_none: "地層下陷與政策差值都不顯示帶狀。",')
    tpl = tmp / "template.html"
    tpl.write_text("const EN = {\n" + "\n".join(en) + "\n};\nconst ZH = {\n"
                   + "\n".join(zh) + "\n};\n", encoding="utf-8")
    return led, tpl


def test_page_alignment_passes_on_aligned_files_and_names_each_mismatch(tmp_path):
    led, tpl = _aligned_files(tmp_path)
    assert N.page_alignment_problems(led, tpl) == []
    d = json.loads(led.read_text(encoding="utf-8"))
    for c in d["claims"]:
        if c["id"] == "C14a":
            c["status_current"] = "D"                       # the old declared status
        if c["id"] == "C30":
            c["wording_en"] = "Bands on projected values are run spreads."
        if c["id"] == "C14c":
            c["cites"]["registrations"] = []
    led.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    t = tpl.read_text(encoding="utf-8").replace(
        "No band is shown for subsidence", "No band is shown for the fan or subsidence")
    tpl.write_text(t.replace("  lg_proj:", "  lg_projX:", 1), encoding="utf-8")
    probs = N.page_alignment_problems(led, tpl)
    assert any("C14a status_current 'D' != declared 'N+D'" in x for x in probs)
    assert any("C30 wording_en lacks" in x for x in probs)
    assert any("C14c does not cite" in x for x in probs)
    assert any("evc_band_none (en) must not contain 'the fan'" in x for x in probs)
    assert any("lg_proj found 1 times" in x for x in probs)


def test_template_strings_split_en_and_zh_on_the_real_page():
    got = N.template_strings()
    for k, v in got.items():
        assert len(v) == 2, k
        assert not re.search(r"[\u4e00-\u9fff]", v[0]) and re.search(r"[\u4e00-\u9fff]", v[1]), k


def test_pending_registration_result_and_file_contains(tmp_path):
    from harness import prereg3

    reg = tmp_path / "prereg"
    reg.mkdir()
    body = (json.dumps({"id": "T.p.v1", "thresholds": {"ratio_band": 1.5}}) + "\n").encode()
    (reg / "T.p.v1.json").write_bytes(body)
    prereg3._v1._append_chained(reg / "INDEX.jsonl", {
        "id": "T.p.v1", "sha256": hashlib.sha256(body).hexdigest(), "test_ids": []})
    r = {"type": "registration", "id": "T.p.v1", "result": None,
         "reg_field": "thresholds.ratio_band", "reg_equals": 1.5}
    assert N.check_reason(r, reg)["status"] == "fail"             # needs result_pending
    assert N.check_reason({**r, "result_pending": "unscored"}, reg)["status"] == "ok"
    n2 = [x for x in N.SUBCLAIMS["C14d"]["reasons"] if x.get("id", "").startswith("N2.")][0]
    assert n2["result"] is None and "unscored" in n2["result_pending"]
    fc = {"type": "code", "check": "file_contains", "file": "hydrophysics/twin/prospective.py",
          "equals": True}
    assert N.check_reason({**fc, "text": "covered 92 % of the"})["status"] == "ok"
    assert N.check_reason({**fc, "text": "no such sentence xyz"})["status"] == "fail"


def test_decision_5_separates_leveling_and_head_reasons():
    d5 = [x for x in N.DECISIONS if x["n"] == 5][0]
    assert "Subsidence and leveling" in d5["decision"] and "Heads" in d5["decision"]
    assert "spent windows" in d5["decision"]
    assert any(r.get("check") == "file_contains" for r in d5["reasons"])
    assert N.U7_RULE in d5["reasons"]


def test_no_home_paths_or_user_names_in_staged_files():
    pat = re.compile(r"(/home/|-home-)[A-Za-z0-9_]")
    for p in sorted(X.rglob("*.py")):
        assert not pat.search(p.read_text(encoding="utf-8")), p
