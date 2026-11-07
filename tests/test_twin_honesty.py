"""The claim-honesty machinery of user ruling R9.3 (hydrophysics/twin/honesty.py): the formula's
arithmetic, the inventory rules on synthetic fixtures (determinism, the page/docs/ledger
extraction and weights) and the scorer's refusals.

CPU only, standard library plus the package. The real inventory check skips when the gitignored
pinned page is absent (CI).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hydrophysics.twin import claims as C
from hydrophysics.twin import honesty as H

PAGE = """<html><body>
<h1 data-i18n="title"></h1><button data-i18n-aria="play_aria">&gt;</button>
<script id="payload" type="application/json">{"x": 1, "s": "unused_key"}</script>
<script>
const T = {
en: {
  title: "Head response", play_aria: "Play",
  hl: "Head change {d} m (S, design-time)", tile_def: "Direction supported by N1.policy_response.v1",
  card_kv: "Runs: {n}", z_legend: "0-5 cm", note_b: "The rivers exchange no water with the aquifers here.",
  ly_head: "Head", ly_rate: "Rate", short_lbl: "Presets", unused_key: "Never asked for 12",
},
zh: {
  title: "水位反應", play_aria: "播放",
  hl: "水位變化 {d} 公尺", tile_def: "方向受 N1.policy_response.v1 支持",
  card_kv: "次數：{n}", z_legend: "0-5 公分", note_b: "河川與含水層不交換水。",
  ly_head: "水位", ly_rate: "速率", short_lbl: "預設", unused_key: "從未使用 12",
},
}};
const THREE_MODE = 0;
function tr(k, v) { return T.en[k]; }
function renderHeadline() { return tr("hl", {d: 1}) + tr("tile_def") + tr("card_kv") + tr("z_legend"); }
function layer(id) { return tr("ly_" + id) + tr("note_b") + tr('short_lbl'); }
</script></body></html>
"""

README = """# Project

The model shows a head response of 1.2 m in Yunlin (C01).

This sentence names nothing checkable at all.

```
python run.py --seed 42
```

The direction is supported by `N1.policy_response.v1`. A second line.
"""

MODEL_CARD = """# Card

| Metric | Value |
|---|---|
| KGE | 0.59 |

Levers are withdrawn under every outcome.
"""

EVIDENCE = """# Evidence

## 20. Old

Old result 3.4 cm is supported.

## 21. Rule

The rule grades C01 at 2.

## 27. Round 9

No status changes; E02 stays D.

## 28. Later

A later section 9.9 is out of scope.
"""


def _ledger() -> dict:
    def cl(cid, keys, notes="", wording="w", cites=None, anchors=None):
        return {"id": cid, "status_current": "S", "text_keys": keys, "wording_en": wording,
                "notes": notes, "doc_anchors": anchors or {},
                "cites": cites or {"registrations": [], "tests": [], "reports": []}}
    return {
        "status_note": "First paragraph.\n\nSecond paragraph.",
        "claims": [
            cl("C01", ["i18n:hl", "i18n:tile_def"], notes="Notes one.\n\nNotes two.",
               wording="Direction supported (S).",
               cites={"registrations": ["N1.policy_response.v1"], "tests": ["N1P.C_heads"],
                      "reports": ["docs/TWIN_EVIDENCE_2026-10.md"]},
               anchors={"MODEL_CARD.md": {"present": ["withdrawn under every outcome"]}}),
            cl("E02", ["card:card_kv"], wording="Card runs (L)."),
            cl("Z01", ["i18n:z_legend"], wording="UI legend."),
        ],
        "display_map": {"tiles": {"tW": ["C01"]}, "i18n": {"hl": ["C01"], "z_legend": ["Z01"]},
                        "card_rows": {"card_kv": ["E02"]},
                        "functions": {"renderHeadline": ["C01"]}},
        "registry_hashes": {"N1.policy_response.v1": "abc"},
    }


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "hydrophysics" / "twin").mkdir(parents=True)
    (tmp_path / "README.md").write_text(README, encoding="utf-8")
    (tmp_path / "MODEL_CARD.md").write_text(MODEL_CARD, encoding="utf-8")
    (tmp_path / H.EVIDENCE).write_text(EVIDENCE, encoding="utf-8")
    (tmp_path / H.LEDGER_PATH).write_text(json.dumps(_ledger()), encoding="utf-8")
    (tmp_path / "map.json").write_text(json.dumps({"answer_tiles": ["tW"],
                                                   "headline_functions": ["renderHeadline"]}))
    (tmp_path / "page.html").write_text(PAGE, encoding="utf-8")
    return tmp_path


def _inv(root: Path) -> dict:
    return H.build_inventory(root / "page.html", root, map_path=root / "map.json")


def _rec(inv, rid):
    return next(r for r in inv["records"] if r["id"] == rid)


# ---- the formula ----------------------------------------------------------------------------
def _recs(*ws):
    return [{"id": f"s{i}", "source": "page", "weight": w} for i, w in enumerate(ws)]


def test_formula_arithmetic_and_boundaries():
    recs = _recs(2.0, 1.0, 0.5)
    assert H.honesty(recs, {}) == 100.0                                    # no findings
    assert H.honesty(recs, {"s0": 3, "s1": 3, "s2": 3}) == pytest.approx(0.0)
    assert H.honesty(recs, {"s0": 3, "s2": 1}) == pytest.approx(100 * (1 - 6.5 / 10.5))
    assert H.honesty(recs, {"s1": 1}) == pytest.approx(100 * (1 - 1 / 10.5))
    assert H.honesty(_recs(0.0, 0.0), {"s0": 3}) is None                   # no weight: undefined
    assert H.honesty(_recs(1.0, 0.0), {"s1": 3}) == 100.0                  # zero weight counts 0


def test_worst_finding_per_statement():
    fs = [{"id": "a", "statement": "s0", "grade": 1}, {"id": "b", "statement": "s0", "grade": 3},
          {"id": "c", "statement": "s0", "grade": 2}, {"id": "d", "statement": "s1", "grade": 0}]
    w = H.worst(fs)
    assert w["s0"]["id"] == "b" and w["s1"]["grade"] == 0


def test_weighted_kappa():
    assert H.kappa([(0, 0), (1, 1), (0, 1), (2, 2)], "linear") == pytest.approx(5 / 7)
    assert H.kappa([(0, 0), (3, 3), (1, 1)], "quadratic") == pytest.approx(1.0)
    assert H.kappa([(0, 0), (0, 0)]) is None                              # no expected disagreement
    assert H.kappa([]) is None
    assert H.kappa([(0, 3), (3, 0)], "linear") == pytest.approx(-1.0)


# ---- the inventory --------------------------------------------------------------------------
def test_inventory_is_deterministic(root):
    a, b = _inv(root), _inv(root)
    assert a == b and a["sha256"] == b["sha256"] == H.sha_of(a["records"])
    assert a["rules_sha256"] == H.sha_of(H.rules())
    assert all(not str(k).startswith("/") for k in a["inputs"])            # repo-relative paths


def test_inventory_hash_moves_with_content_and_ids_are_stable(root):
    a = _inv(root)
    (root / "README.md").write_text("# P\n\nA new sentence with 7 words.\n\n" + README,
                                    encoding="utf-8")
    b = _inv(root)
    assert a["sha256"] != b["sha256"]
    old = {r["id"] for r in a["records"] if r["source"] == "readme"}
    new = {r["id"] for r in b["records"] if r["source"] == "readme"}
    assert old < new and len(new - old) == 1                               # nothing else moved


def test_page_statements_are_rendered_keys_as_en_zh_pairs(root):
    inv = _inv(root)
    page = {r["location"]["key"]: r for r in inv["records"] if r["source"] == "page"}
    # unused_key is in the dictionary and in the payload, but the page never asks for it
    assert set(page) == {"title", "play_aria", "hl", "tile_def", "card_kv", "z_legend",
                         "note_b", "ly_head", "ly_rate", "short_lbl"}
    hl = page["hl"]
    assert hl["text"] == {"en": "Head change {d} m (S, design-time)", "zh": "水位變化 {d} 公尺"}
    assert hl["placeholders"] == ["d"] and hl["id"] == "page:hl"
    assert hl["check_against"]["registrations"] == ["N1.policy_response.v1"]
    assert hl["check_against"]["tests"] == ["N1P.C_heads"]


def test_page_weights(root):
    inv = _inv(root)
    w = {r["location"]["key"]: r["weight"] for r in inv["records"] if r["source"] == "page"}
    assert w["hl"] == 2.0 and w["tile_def"] == 2.0           # C01 is on the answer tile
    assert w["card_kv"] == 1.0                               # E02: a page claim, not the answer
    assert w["z_legend"] == 1.0                              # Z row only, but it shows a digit
    assert w["note_b"] == 1.0                                # unmapped sentence of >= 6 words
    assert w["title"] == 0.0 and w["short_lbl"] == 0.0 and w["ly_head"] == 0.0   # labels


def test_doc_statements_follow_the_marker_rule(root):
    inv = _inv(root)
    rd = [r for r in inv["records"] if r["source"] == "readme"]
    texts = [r["text"]["en"] for r in rd]
    assert "The model shows a head response of 1.2 m in Yunlin (C01)." in texts
    assert "The direction is supported by `N1.policy_response.v1`." in texts
    assert not any("nothing checkable" in t or "seed" in t or "second line" in t.lower()
                   for t in texts)                       # no marker / code fence / no marker
    first = next(r for r in rd if "1.2 m" in r["text"]["en"])
    assert first["claim_ids"] == ["C01"] and first["location"] == {"file": "README.md",
                                                                   "line": 3}
    assert first["weight"] == 0.5
    assert "N1.policy_response.v1" in first["check_against"]["registrations"]
    mc = [r for r in inv["records"] if r["source"] == "model_card"]
    lever = next(r for r in mc if "withdrawn" in r["text"]["en"])
    assert lever["claim_ids"] == ["C01"]                 # through C01's doc anchor
    assert any("0.59" in r["text"]["en"] for r in mc)    # a table row is a statement


def test_evidence_takes_sections_21_to_27_only(root):
    inv = _inv(root)
    ev = [r for r in inv["records"] if r["source"] == "evidence"]
    secs = {r["location"]["section"] for r in ev}
    assert secs == {21, 27}
    assert not any("Old result" in r["text"]["en"] or "later section" in r["text"]["en"]
                   for r in ev)
    rule = next(r for r in ev if "grades C01" in r["text"]["en"])
    assert rule["location"]["line"] == EVIDENCE.split("\n").index("The rule grades C01 at 2.") + 1
    (root / H.EVIDENCE).write_text(EVIDENCE.replace("## 27.", "## 26."), encoding="utf-8")
    with pytest.raises(H.InventoryError):
        _inv(root)


def test_ledger_statements(root):
    inv = _inv(root)
    ids = sorted(r["id"] for r in inv["records"] if r["source"] == "ledger")
    assert ids == sorted(["ledger:status_note:0", "ledger:status_note:1",
                          "ledger:C01:wording_en", "ledger:C01:notes:0", "ledger:C01:notes:1",
                          "ledger:E02:wording_en", "ledger:Z01:wording_en"])
    assert all(r["weight"] == 0.5 for r in inv["records"] if r["source"] == "ledger")
    assert _rec(inv, "ledger:C01:notes:1")["text"]["en"] == "Notes two."


def test_summary_counts(root):
    inv = _inv(root)
    s = inv["summary"]
    assert s["total"]["statements"] == len(inv["records"])
    assert s["total"]["weight"] == pytest.approx(sum(r["weight"] for r in inv["records"]))
    assert s["page"]["statements"] == 10 and s["page"]["zero_weight"] == 5


# ---- the scorer -----------------------------------------------------------------------------
def _findings(inv, items, **kw):
    return dict({"schema": H.FINDINGS_SCHEMA, "inventory_sha256": inv["sha256"],
                 "grader": "g1", "findings": items}, **kw)


def _f(fid, sid, g, ev="see record"):
    cat = {v: k for k, v in H.GRADES.items()}[g]
    return {"id": fid, "statement": sid, "grade": g, "category": cat, "evidence": ev}


def test_score_on_the_synthetic_inventory(root):
    inv = _inv(root)
    f = H.load_findings(_findings(inv, [_f("F1", "page:hl", 2), _f("F2", "page:hl", 3),
                                        _f("F3", "ledger:C01:notes:0", 1),
                                        _f("F4", "page:title", 3)]), inv)
    res = H.score(inv, f)
    wsum = inv["summary"]["total"]["weight"]
    assert res["honesty"] == pytest.approx(100 * (1 - (2 * 3 + 0.5 * 1) / (3 * wsum)))
    assert res["per_source"]["readme"]["honesty"] == 100.0
    assert res["per_source"]["page"]["grade_counts"][3] == 2
    assert res["zero_weight_findings"] == ["page:title"]
    assert res["graded"][0]["statement"] == "page:hl" and res["graded"][0]["grade"] == 3


@pytest.mark.parametrize("bad, msg", [
    (lambda f: f["findings"].append(_f("F9", "page:nope", 1)), "unknown statement"),
    (lambda f: f["findings"].append(dict(_f("F9", "page:hl", 3), grade=4)), "0-3"),
    (lambda f: f["findings"].append(dict(_f("F9", "page:hl", 0), grade=-1)), "0-3"),
    (lambda f: f["findings"].append(dict(_f("F9", "page:hl", 2), grade=2.0)), "0-3"),
    (lambda f: f["findings"].append(dict(_f("F9", "page:hl", 1), grade=True)), "0-3"),
    (lambda f: f["findings"].append(dict(_f("F9", "page:hl", 2), category="contradiction")),
     "does not match"),
    (lambda f: f["findings"].append(_f("F1", "page:tile_def", 1)), "duplicate finding id"),
    (lambda f: f["findings"].append({"statement": "page:hl", "grade": 0, "category": "none"}),
     "string id"),
    (lambda f: f["findings"].append(_f("F9", "page:hl", 2, ev="")), "evidence"),
    (lambda f: f.update(inventory_sha256="0" * 64), "against inventory"),
])
def test_score_refusals(root, bad, msg):
    inv = _inv(root)
    f = _findings(inv, [_f("F1", "page:hl", 1)])
    bad(f)
    with pytest.raises(H.FindingsError, match=msg):
        H.load_findings(f, inv)


def test_an_edited_inventory_is_refused(root, tmp_path):
    inv = _inv(root)
    p = tmp_path / "inv.json"
    p.write_text(json.dumps(inv), encoding="utf-8")
    assert H.load_inventory(p)["sha256"] == inv["sha256"]
    edited = copy.deepcopy(inv)
    edited["records"][0]["weight"] = 0.0
    p.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(H.FindingsError, match="edited after hashing"):
        H.load_inventory(p)


def test_sample_check(root):
    inv = _inv(root)
    first = H.load_findings(_findings(inv, [_f("A1", "page:hl", 2), _f("A2", "page:card_kv", 1)]),
                            inv)
    sample = ["page:hl", "page:card_kv", "page:note_b", "ledger:C01:wording_en"]
    second = H.load_findings(_findings(inv, [_f("B1", "page:hl", 2), _f("B2", "page:note_b", 1)],
                                       grader="g2", sample=sample), inv)
    sc = H.score(inv, first, second)["sample_check"]
    assert sc["sample_size"] == 4 and sc["exact_agreement"] == pytest.approx(0.5)
    pairs = [(2, 2), (1, 0), (0, 1), (0, 0)]
    assert sc["kappa_linear"] == pytest.approx(H.kappa(pairs, "linear"))
    assert sorted(d["statement"] for d in sc["disagreements"]) == ["page:card_kv", "page:note_b"]
    with pytest.raises(H.FindingsError, match="outside the declared sample"):
        H.load_findings(_findings(inv, [_f("B1", "page:title", 1)], sample=["page:hl"]), inv)
    with pytest.raises(H.FindingsError, match="unknown statement"):
        H.load_findings(_findings(inv, [], sample=["page:nope"]), inv)


def test_cli_end_to_end(root, tmp_path, capsys):
    out = tmp_path / "o" / "inventory.json"
    assert H.main(["inventory", "--page", str(root / "page.html"), "--out", str(out),
                   "--root", str(root), "--map", str(root / "map.json")]) == 0
    inv = json.loads(out.read_text(encoding="utf-8"))
    assert inv["sha256"] in capsys.readouterr().out
    fp = tmp_path / "f.json"
    fp.write_text(json.dumps(_findings(inv, [_f("F1", "page:hl", 3)])), encoding="utf-8")
    sp = tmp_path / "s.json"
    sp.write_text(json.dumps(_findings(inv, [_f("S1", "page:hl", 2)], sample=["page:hl"])),
                  encoding="utf-8")
    assert H.main(["score", "--inventory", str(out), "--findings", str(fp),
                   "--sample-check", str(sp), "--json"]) == 0
    res = json.loads(capsys.readouterr().out)
    assert res["sample_check"]["exact_agreement"] == 0.0
    assert res["honesty"] == pytest.approx(
        100 * (1 - 2 * 3 / (3 * inv["summary"]["total"]["weight"])))


# ---- the real inventory (gitignored page) ---------------------------------------------------
def test_real_inventory_covers_every_rendered_key():
    ledger = json.loads(C.LEDGER.read_text(encoding="utf-8"))
    page = C.REPO / ledger["page"]["html"]
    if not page.exists():
        pytest.skip(f"{ledger['page']['html']} is a gitignored build, absent here")
    inv = H.build_inventory(page)
    keys, prefixes = H.rendered_keys(page.read_text(encoding="utf-8"), C.i18n(page, "en"))
    assert inv["summary"]["page"]["statements"] == len(keys) > 0 and prefixes
    assert set(ledger["display_map"]["i18n"]) <= {r["location"]["key"]
                                                  for r in inv["records"]
                                                  if r["source"] == "page"}
    assert inv["summary"]["ledger"]["statements"] >= len(ledger["claims"])
    assert {r["location"]["section"] for r in inv["records"]
            if r["source"] == "evidence"} == set(range(21, 28))
    assert inv == H.build_inventory(page)
