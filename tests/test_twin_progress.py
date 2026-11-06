"""The progress recount (hydrophysics/twin/progress.py): the rule arithmetic of section 21 on a
synthetic ledger, and the committed mapping (progress_map_v1.json) against the real ledger.

CPU only, standard library plus the package; reads no observation value, report or registry.
"""
from __future__ import annotations

import copy
import json

import pytest

from hydrophysics.twin import claims as C
from hydrophysics.twin import progress as P

PARTS = ["head_response_yunlin", "forecast_start_validation", "uncertainty_bands",
         "subsidence_cut_response", "changhua"]


def _map(claims: dict) -> dict:
    real = P.load_map()
    return {"parts": PARTS, "grades": real["grades"], "end_goal_grade": 1.0,
            "answer_tiles": ["tW"], "headline_functions": ["renderHeadline"],
            "claims": claims}


def _claim(cid, status, text_keys=()):
    return {"id": cid, "status_current": status, "text_keys": list(text_keys),
            "doc_anchors": {}}


def _synthetic():
    """One part (heads) with every weight class, one composite in bands, a W+D in subsidence,
    a docs-only S in Changhua, a forecast D whose decidability depends on the registry."""
    ledger = {
        "claims": [
            _claim("A1", "S", ["tile:tW"]),            # answer tile -> 2
            _claim("A2", "N", ["i18n:x"]),             # headline function -> 2
            _claim("A3", "L", ["i18n:y"]),             # page -> 1
            _claim("A4", "W", ["doc:README.md"]),      # docs only -> 0.5
            _claim("Z9", "L", ["i18n:z"]),             # non-claim row -> 0
            _claim("B1", "N+D", ["i18n:b"]),
            _claim("B2", "L", ["i18n:b2"]),
            _claim("F1", "D", ["i18n:f"]),
            _claim("F2", "N", ["i18n:f2"]),
            _claim("S1", "W+D", ["i18n:s"]),
            _claim("S2", "N", ["i18n:s2"]),
            _claim("H1", "S", ["doc:README.md"]),
            _claim("H2", "W", ["i18n:h"]),
        ],
        "display_map": {"tiles": {"tW": ["A1"], "tA": ["S1"]},
                        "functions": {"renderHeadline": ["A2"]}},
        "registry_hashes": {"REG.measured.v1": "abc"},
    }
    claims = {
        "A1": {"part": PARTS[0], "grade": {"S": {"S": "S-dt"}}, "reason": "r"},
        "A2": {"part": PARTS[0], "reason": "r"},
        "A3": {"part": PARTS[0], "reason": "r"},
        "A4": {"part": PARTS[0], "reason": "r",
               "end_goal": {"kind": "withdrawn_with_reason", "by": "x"}},
        "Z9": {"part": "none", "reason": "r"},
        "B1": {"part": PARTS[2], "grade": {"N+D": {"D": {"measured_by": "REG.measured.v1",
                                                         "otherwise": "D-unknown"}}},
               "reason": "r"},
        "B2": {"part": PARTS[2], "reason": "r",
               "end_goal": {"kind": "no_target", "by": "x", "requires": "REG.absent.v1"}},
        "F1": {"part": PARTS[1], "grade": {"D": {"D": {"measured_by": "REG.absent.v1",
                                                       "otherwise": "D-unknown"}}},
               "reason": "r"},
        "F2": {"part": PARTS[1], "reason": "r"},
        "S1": {"part": PARTS[3], "reason": "r"},
        "S2": {"part": PARTS[3], "reason": "r"},
        "H1": {"part": PARTS[4], "grade": {"S": {"S": "S-dt-dir-size"}}, "reason": "r"},
        "H2": {"part": PARTS[4], "reason": "r"},
    }
    return ledger, _map(claims)


def _row(res, cid):
    return next(r for r in res["rows"] if r["id"] == cid)


def test_weights_follow_the_ledger():
    ledger, pmap = _synthetic()
    res = P.recount(ledger, pmap)
    assert [_row(res, c)["weight"] for c in ("A1", "A2", "A3", "A4", "Z9")] == [2, 2, 1, 0.5, 0]


def test_part_arithmetic_on_a_synthetic_ledger():
    ledger, pmap = _synthetic()
    res = P.recount(ledger, pmap, honesty=90.0)
    heads = (2 * 0.9 + 2 * 0.5 + 1 * 0.25 + 0.5 * 0.15) / 5.5 * 100
    fc = (0.6 + 0.5) / 2 * 100          # F1's decider is not cited: D-unknown
    bands = (0.7 + 0.25) / 2 * 100      # N+D -> max(N, D-measured)
    subs = (0.15 + 0.5) / 2 * 100       # W+D -> W
    ch = (0.5 * 0.75 + 1 * 0.15) / 1.5 * 100
    parts = res["parts"]
    for p, want in zip(PARTS, (heads, fc, bands, subs, ch), strict=True):
        assert parts[p]["official"] == pytest.approx(want)
    sci = (heads + fc + bands + subs + ch) / 5
    assert res["science"]["official"] == pytest.approx(sci)
    assert res["six_part"]["official"] == pytest.approx((5 * sci + 90.0) / 6)
    # intended output: W and L graded 0
    assert parts[PARTS[0]]["intended"] == pytest.approx((2 * 0.9 + 2 * 0.5) / 5.5 * 100)
    assert parts[PARTS[2]]["intended"] == pytest.approx(0.7 / 2 * 100)
    # end goal: A4 counts 1.0; B2's mark needs a registration the ledger does not cite
    assert parts[PARTS[0]]["end_goal"] == pytest.approx(
        (2 * 0.9 + 2 * 0.5 + 0.25 + 0.5 * 1.0) / 5.5 * 100)
    assert parts[PARTS[2]]["end_goal"] == pytest.approx(bands)
    assert res["composite_rows"] == ["B1", "S1"]


def test_composite_first_rule_and_w_dominates():
    ledger, pmap = _synthetic()
    res = P.recount(ledger, pmap, composite="first")
    assert _row(res, "B1")["grade"] == 0.5 and _row(res, "B1")["graded_code"] == "N"
    assert _row(res, "S1")["grade"] == 0.15      # W wins under either rule
    assert _row(P.recount(ledger, pmap), "B1")["grade"] == 0.7


def test_measured_by_follows_the_registry():
    ledger, pmap = _synthetic()
    assert _row(P.recount(ledger, pmap), "F1")["grade_key"] == "D-unknown"
    ledger["registry_hashes"]["REG.absent.v1"] = "def"
    res = P.recount(ledger, pmap)
    assert _row(res, "F1")["grade_key"] == "D-measured"
    assert _row(res, "B2")["end_goal"] == "no_target"


def test_a_status_change_without_a_mapping_change_refuses():
    ledger, pmap = _synthetic()
    ledger["claims"][1]["status_current"] = "S"          # A2: N -> S, no qualifier mapped
    with pytest.raises(P.MappingError, match="A2"):
        P.recount(ledger, pmap)
    ledger, pmap = _synthetic()
    ledger["claims"].append(_claim("NEW", "N"))
    with pytest.raises(P.MappingError, match="NEW: no mapping entry"):
        P.recount(ledger, pmap)


def test_weight_override_needs_a_reason():
    ledger, pmap = _synthetic()
    pmap["claims"]["A2"]["weight"] = 1
    with pytest.raises(P.MappingError, match="weight override"):
        P.recount(ledger, pmap)
    pmap["claims"]["A2"]["weight_reason"] = "why"
    res = P.recount(ledger, pmap)
    assert _row(res, "A2")["weight"] == 1 and res["weight_overrides"] == ["A2"]


# ---- the committed mapping against the real ledger -------------------------------------------
def _ledger():
    return C.load_ledger(allow_pending=True)


def test_every_ledger_claim_has_a_mapping_entry():
    pmap = P.load_map()
    missing = [c["id"] for c in _ledger()["claims"] if c["id"] not in pmap["claims"]]
    assert not missing, f"no progress mapping for {missing}"


def test_the_map_has_no_unknown_ids():
    pmap = P.load_map()
    ids = {c["id"] for c in _ledger()["claims"]}
    active = {k for k, v in pmap["claims"].items() if not v.get("retired")}
    retired = {k for k, v in pmap["claims"].items() if v.get("retired")}
    assert active == ids, f"unknown ids {sorted(active - ids)}"
    assert not retired & ids, f"ids marked retired are still in the ledger: {retired & ids}"
    for name, cand in pmap["candidates"].items():
        assert set(cand["claims"]) <= set(pmap["claims"]), name


def test_map_values_are_well_formed():
    pmap = P.load_map()
    assert pmap["parts"] == PARTS
    assert set(pmap["grades"]) == {"S", "S-dt", "S-dt-dir-size", "D-measured", "D-unknown",
                                   "D-weak", "N", "L", "W"}
    for cid, e in pmap["claims"].items():
        assert e["part"] in PARTS + ["none"], cid
        assert e.get("reason"), cid
        if "weight" in e:
            assert e["weight"] in (2, 1, 0.5, 0) and e.get("weight_reason"), cid
        for status, quals in e.get("grade", {}).items():
            codes = status.split("+")
            for code, spec in quals.items():
                assert code in codes and code in ("S", "D"), (cid, status, code)
                key = spec if isinstance(spec, str) else spec["otherwise"]
                assert key in pmap["grades"], (cid, key)
        if e.get("end_goal"):
            assert e["end_goal"]["kind"] in P.END_GOAL_KINDS and e["end_goal"]["by"], cid


def test_weights_and_grades_on_the_real_ledger():
    ledger = _ledger()
    for cand in (None, *P.load_map()["candidates"]):
        pmap = P.with_candidate(P.load_map(), cand)
        for rule in ("max", "first"):
            res = P.recount(ledger, pmap, composite=rule)
            assert {r["weight"] for r in res["rows"]} <= {2.0, 1.0, 0.5, 0.0}
            assert all(0 <= r["grade"] <= 1 for r in res["rows"])
            assert all(v is not None and 0 <= v <= 100
                       for s in res["parts"].values() for v in (s["official"], s["intended"],
                                                                s["end_goal"]))


def test_answer_tiles_match_the_template():
    ids, primary = C.tiles()
    assert P.load_map()["answer_tiles"] == ids[:primary]


def test_non_claim_rows_weigh_zero_and_sit_in_no_part():
    res = P.recount(_ledger(), P.load_map())
    for r in res["rows"]:
        if r["id"].startswith("Z"):
            assert r["weight"] == 0 and r["part"] == "none", r["id"]


def test_cli_json_round_trip(capsys, tmp_path):
    ledger, pmap = _synthetic()
    lp, mp = tmp_path / "ledger.json", tmp_path / "map.json"
    lp.write_text(json.dumps(ledger))
    mp.write_text(json.dumps(pmap))
    assert P.main(["--ledger", str(lp), "--map", str(mp), "--json", "--honesty", "90"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["six_part"]["official"] == pytest.approx(
        P.recount(copy.deepcopy(ledger), pmap, honesty=90.0)["six_part"]["official"])
    assert P.main(["--at", str(lp), "--map", str(mp)]) == 0
    assert "not computed" in capsys.readouterr().out
