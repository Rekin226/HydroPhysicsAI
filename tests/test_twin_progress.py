"""The progress recount (hydrophysics/twin/progress.py): the rule arithmetic of section 21 on a
synthetic ledger, and the committed mapping (progress_map_v1.json) against the real ledger.

CPU only, standard library plus the package; reads no observation value, report or registry.
"""
from __future__ import annotations

import copy
import json
import subprocess

import pytest

from hydrophysics.twin import claims as C
from hydrophysics.twin import progress as P

PARTS = ["head_response_yunlin", "forecast_start_validation", "uncertainty_bands",
         "subsidence_cut_response", "changhua"]


def _map(claims: dict) -> dict:
    real = P.load_map()
    return {"parts": PARTS, "grades": real["grades"], "end_goal_grade": 1.0, "composite": "max",
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
    with pytest.raises(P.MappingError,
                       match="A2: status 'S' carries an S code the mapping gives no qualifier"):
        P.recount(ledger, pmap)
    ledger, pmap = _synthetic()
    ledger["claims"][1]["status_current"] = "L"          # a change to N/L/W does not refuse
    assert _row(P.recount(ledger, pmap), "A2")["grade"] == 0.25
    ledger["claims"][0]["status_current"] = "S+D"         # A1: a new S/D-coded status refuses
    with pytest.raises(P.MappingError, match="A1: status 'S\\+D' carries an S code"):
        P.recount(ledger, pmap)
    ledger, pmap = _synthetic()
    ledger["claims"].append(_claim("NEW", "N"))
    with pytest.raises(P.MappingError, match="NEW: no mapping entry"):
        P.recount(ledger, pmap)


def test_a_pending_hash_refuses():
    ledger, pmap = _synthetic()
    ledger["registry_hashes"]["REG.measured.v1"] = "PENDING"
    with pytest.raises(P.MappingError, match="REG.measured.v1 is cited with a PENDING hash"):
        P.recount(ledger, pmap)
    ledger, pmap = _synthetic()
    ledger["registry_hashes"]["REG.absent.v1"] = "PENDING"           # B2's end-goal requires
    with pytest.raises(P.MappingError, match="PENDING"):
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
    for name, alt in pmap["mappings"].items():
        assert set(alt["claims"]) <= set(pmap["claims"]), name


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
    for name in (None, *P.load_map()["mappings"]):
        pmap = P.with_mapping(P.load_map(), name)
        for rule in ("max", "first"):
            res = P.recount(ledger, pmap, composite=rule)
            assert {r["display_weight"] for r in res["rows"]} <= {2.0, 1.0, 0.5, 0.0}
            for parent, ids in res["split_groups"].items():
                ps = [r for r in res["rows"] if r["id"] in ids]
                disp = [r["display_weight"] for r in ps]
                assert [r["weight"] for r in ps] == pytest.approx(
                    [max(disp) * d / sum(disp) for d in disp]), parent
                if pmap.get("split_cap"):
                    mean = sum(r["weight"] * r["grade"] for r in ps) / max(disp)
                    assert mean <= ps[0]["parent_grade"] + 1e-12, parent
            if not pmap.get("split_weights"):
                assert all(r["weight"] == r["display_weight"] for r in res["rows"])
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


# ---- user rulings 2026-11-07, fourteenth set (R8.16a/b/c) -------------------------------------
# results/dataprep/rulings/user_rulings_2026-11-07.json (sha256 c0886864)
def _split_synthetic():
    """A1 (status S+D when split) split into A1 (S), A1-w (D) and A1-sz (N); the pieces' own
    display weights are 2, 2 and 1."""
    ledger, pmap = _synthetic()
    pmap.update(split_weights=True, split_cap=True)
    ledger["claims"] += [_claim("A1-w", "D", ["tile:tW"]), _claim("A1-sz", "N", ["i18n:q"])]
    ledger["display_map"]["tiles"]["tW"].append("A1-w")
    def so(st):
        return {"parent": "A1", "parent_status": "S+D", "piece_status_at_split": st}
    pmap["claims"]["A1"].update(split_of=so("S"), split_into=["A1", "A1-w", "A1-sz"])
    pmap["claims"]["A1"]["grade"]["S+D"] = {"S": "S-dt-dir-size", "D": "D-unknown"}
    pmap["claims"]["A1-w"] = {"part": PARTS[0], "grade": {"D": {"D": "D-measured"}},
                              "reason": "r", "split_of": so("D")}
    pmap["claims"]["A1-sz"] = {"part": PARTS[0], "grade": {"S": {"S": "S-dt"}}, "reason": "r",
                               "split_of": so("N")}
    return ledger, pmap


def test_split_pieces_share_the_parent_weight_in_proportion():
    """R8.16b and R8.17d: piece i weighs W_P * d_i / sum d, W_P the largest display weight among
    the pieces, so the pieces together weigh exactly the parent's weight."""
    ledger, pmap = _split_synthetic()
    pmap["split_cap"] = False
    res = P.recount(ledger, pmap)
    assert [_row(res, c)["weight"] for c in ("A1", "A1-w", "A1-sz")] == pytest.approx(
        [2 * 2 / 5, 2 * 2 / 5, 2 * 1 / 5])
    assert _row(res, "A1-sz")["display_weight"] == 1
    assert res["parts"][PARTS[0]]["weight"] == pytest.approx(5.5)    # unchanged by the split
    # before the split only the parent is present: it keeps its weight and its grade
    ledger, _ = _synthetic()
    ledger["claims"][0]["status_current"] = "S+D"
    r = _row(P.recount(ledger, pmap), "A1")
    assert r["weight"] == 2 and r["grade"] == 0.75 and r["grade_capped"] is None
    # pieces of one parent in two parts, or a piece its parent does not list, refuse
    ledger, pmap = _split_synthetic()
    pmap["claims"]["A1-w"]["part"] = PARTS[1]
    with pytest.raises(P.MappingError, match="split pieces of A1"):
        P.recount(ledger, pmap)
    ledger, pmap = _split_synthetic()
    pmap["claims"]["A1"]["split_into"] = ["A1", "A1-w"]
    with pytest.raises(P.MappingError, match="A1-sz: its parent A1 must list it"):
        P.recount(ledger, pmap)


def _set_parent_status(pmap, status):
    for c in ("A1", "A1-w", "A1-sz"):
        pmap["claims"][c]["split_of"]["parent_status"] = status


W_PIECES = (0.8, 0.8, 0.4)                                  # 2 * (2, 2, 1) / 5


def _wmean(grades):
    return sum(w * g for w, g in zip(W_PIECES, grades, strict=True)) / 2


def test_split_uplift_is_removed():
    """R8.17a: the uplift the split itself created, u = max(0, m0 - g_P), is removed by scaling
    every piece by (m1 - u) / m1; with no later change (m1 = m0) the pieces' mean is g_P."""
    ledger, pmap = _split_synthetic()
    raw = (0.9, 0.7, 0.5)                                    # S-dt, D-measured, N at the split
    res = P.recount(ledger, pmap)                            # parent S+D -> 0.75 > m0 0.74
    assert [_row(res, c)["grade"] for c in ("A1", "A1-w", "A1-sz")] == pytest.approx(raw)
    assert _row(res, "A1")["split_uplift"] == 0 and _row(res, "A1")["grade_capped"] is None
    pmap["claims"]["A1"]["grade"]["N+D"] = {"D": "D-measured"}
    _set_parent_status(pmap, "N+D")                          # parent graded N: 0.5 < 0.74
    res = P.recount(ledger, pmap)
    got = [_row(res, c)["grade"] for c in ("A1", "A1-w", "A1-sz")]
    m0 = _wmean(raw)
    assert _row(res, "A1")["parent_grade"] == 0.5
    assert _row(res, "A1")["split_uplift"] == pytest.approx(m0 - 0.5)
    assert got == pytest.approx([g * 0.5 / m0 for g in raw])
    assert _wmean(got) == pytest.approx(0.5)
    assert _row(res, "A1")["intended_grade"] == pytest.approx(got[0])
    # display only: a piece can show below its own grade (A1-sz 0.5 -> 0.34)
    assert got[2] < raw[2]


def test_a_later_status_change_counts_in_full():
    """A piece's status change after the split moves the part by exactly that change's
    weighted grade gain; the split's own uplift stays removed."""
    ledger, pmap = _split_synthetic()
    pmap["claims"]["A1"]["grade"]["N+D"] = {"D": "D-measured"}
    _set_parent_status(pmap, "N+D")
    before = P.recount(ledger, pmap)["parts"][PARTS[0]]
    ledger["claims"][-1]["status_current"] = "S"            # A1-sz: N (0.5) -> S-dt (0.9)
    after = P.recount(ledger, pmap)
    assert after["parts"][PARTS[0]]["weight"] == pytest.approx(before["weight"])
    gain = 0.4 * (0.9 - 0.5)                                # w_i * delta g
    assert after["parts"][PARTS[0]]["official"] - before["official"] == pytest.approx(
        100 * gain / before["weight"])
    assert _wmean([_row(after, c)["grade"] for c in ("A1", "A1-w", "A1-sz")]) == \
        pytest.approx(_wmean((0.9, 0.7, 0.9)) - (_wmean((0.9, 0.7, 0.5)) - 0.5))


def test_split_rule_edge_cases():
    # before the split only the parent is present: weighted and graded as itself, uncapped
    ledger, pmap = _split_synthetic()
    pre, _ = _synthetic()
    pre["claims"][0]["status_current"] = "S+D"
    r = _row(P.recount(pre, pmap), "A1")
    assert (r["weight"], r["grade"], r["split_of"]) == (2, 0.75, None)
    # an at-split status with a conditional qualifier refuses (m0 must be fixed)
    pmap["claims"]["A1-w"]["grade"]["D"]["D"] = {"measured_by": "REG.measured.v1",
                                                 "otherwise": "D-unknown"}
    with pytest.raises(P.MappingError, match="A1-w: the qualifier of its at-split status"):
        P.recount(ledger, pmap)
    ledger, pmap = _split_synthetic()
    del pmap["claims"]["A1-sz"]["split_of"]["piece_status_at_split"]
    with pytest.raises(P.MappingError, match="A1-sz: split_of needs piece_status_at_split"):
        P.recount(ledger, pmap)


def test_a_reading_report_must_be_cited_by_the_row():
    ledger, pmap = _synthetic()
    pmap["claims"]["F1"]["grade"] = {"D": {"D": {
        "measured_by": "REG.measured.v1", "reading_report": "reports/R/decidability.json",
        "reading_sha256": "00000000", "otherwise": "D-unknown"}}}
    assert _row(P.recount(ledger, pmap), "F1")["grade_key"] == "D-unknown"
    ledger["claims"][7]["cites"] = {"reports": ["reports/R/decidability.json"]}
    assert _row(P.recount(ledger, pmap), "F1")["grade_key"] == "D-measured"


def test_the_default_mapping_is_the_ruled_one():
    pmap = P.load_map()
    assert pmap["composite"] == "first" and pmap["split_weights"] is True           # R8.16a/b
    assert pmap["split_cap"] is True                                                # R8.17a
    assert "R8.16" in pmap["rule"] and "c0886864" in pmap["rule"]
    assert "R8.17" in pmap["rule"] and "7602157b" in pmap["rule"]
    cl = pmap["claims"]
    so = {c: cl[c].get("split_of") for c in ("C01", "C01-w", "C01-sz", "C30-scen", "C30-fc")}
    at = {"C01": "S", "C01-w": "D", "C01-sz": "N", "C30-scen": "N", "C30-fc": "D"}
    assert so == {c: {"parent": c[:3], "parent_status": "S+D" if c.startswith("C01") else "N+D",
                      "piece_status_at_split": at[c]} for c in at}
    assert cl["C01"]["split_into"] == ["C01", "C01-w", "C01-sz"]
    assert cl["C30"]["split_into"] == ["C30-scen", "C30-fc"] and cl["C30"]["retired"]
    assert cl["C01-w"]["grade"] == {"D": {"D": "D-unknown"}}                         # R8.16c
    assert cl["E21"]["grade"] == {"D": {"D": "D-weak"}}
    assert cl["E25"]["grade"] == cl["E26"]["grade"] == {"S": {"S": "S-dt-dir-size"}}
    for c in ("E04", "C30-fc"):                                                     # R8.17b
        assert cl[c]["grade"] == {"D": {"D": "D-unknown"}}, c
    e30 = cl["E30"]["grade"]["D"]["D"]                      # not ruled: its registered reading
    assert e30["measured_by"] == "E02z.level_years23.v1" and e30["reading_sha256"] == "0e5e83d3"
    assert e30["reading_report"] == ("results/dataprep/reports/E02z.level_years23.v1/"
                                     "decidability/report.json")
    assert "weight" not in cl["C28"]                                                # R8.17c
    assert not any("plan" in e["reason"] for e in cl.values())
    assert P.with_mapping(pmap, None) is pmap and P.with_mapping(pmap, "official") is pmap
    old = P.with_mapping(pmap, "default-v1")
    assert old["composite"] == "max" and old["split_weights"] is False
    assert old["split_cap"] is False and old["claims"]["C28"]["weight"] == 1
    res = P.recount(_ledger(), pmap)
    assert res["composite_rule"] == "first" and res["weight_overrides"] == []
    assert res["split_groups"] == {"C01": ["C01", "C01-sz", "C01-w"], "C30": ["C30-fc", "C30-scen"]}
    w = {r["id"]: r["weight"] for r in res["rows"]}
    assert (w["C30-scen"], w["C30-fc"]) == pytest.approx((4 / 3, 2 / 3))             # R8.17d
    assert (w["C01"], w["C01-w"], w["C01-sz"]) == pytest.approx((2 / 3,) * 3)


AUDIT_STATE, TODAY = "fd40796", "08a6aa7"


def _at(commit):
    ok = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=P.REPO,
                        capture_output=True).returncode == 0
    if not ok:
        pytest.skip(f"commit {commit} not in this clone (shallow checkout)")
    return P.load_ledger(at=commit)[0]


# Official figures (rulings R8.16a-c, R8.17a-d) as published in evidence section 26.
PINNED = {
    AUDIT_STATE: {"honesty": 95.0,
                  "parts": (50.9091, 51.1538, 41.875, 35.814, 45.625),
                  "science": (45.0754, 37.8435, 56.6219), "six": (53.3961, 63.0182),
                  "default_v1": (48.36, 56.13)},
    TODAY: {"honesty": 98.0,
            "parts": (49.3939, 52.5, 42.7778, 35.5682, 45.625),
            "science": (45.173, 38.0873, 57.2008), "six": (53.9775, 64.0006),
            "default_v1": (49.16, 57.30)},
}


@pytest.mark.parametrize("commit", [AUDIT_STATE, TODAY])
def test_ruled_figures_are_reproduced(commit):
    """The official figures at the 2026-11-02 audit state (main fd40796) and at main 08a6aa7,
    pinned, with the earlier default (default-v1) beside them."""
    ledger, pin = _at(commit), PINNED[commit]
    res = P.recount(ledger, P.load_map(), honesty=pin["honesty"])
    for p, v in zip(PARTS, pin["parts"], strict=True):
        assert res["parts"][p]["official"] == pytest.approx(v, abs=1e-3), p
    sci = res["science"]
    assert (sci["official"], sci["intended"], sci["end_goal"]) == pytest.approx(
        pin["science"], abs=1e-3)
    assert (res["six_part"]["official"], res["six_part"]["end_goal"]) == pytest.approx(
        pin["six"], abs=1e-3)
    old = P.recount(ledger, P.with_mapping(P.load_map(), "default-v1"), honesty=pin["honesty"])
    assert (old["science"]["official"], old["six_part"]["official"]) == pytest.approx(
        pin["default_v1"], abs=0.01)


def test_every_main_state_from_the_audit_to_today_recounts():
    _at(AUDIT_STATE)
    revs = subprocess.run(["git", "rev-list", "--first-parent", f"{AUDIT_STATE}^..{TODAY}"],
                          cwd=P.REPO, capture_output=True, text=True, check=True).stdout.split()
    assert len(revs) >= 2
    for rev in revs:
        ledger = P.load_ledger(at=rev)[0]
        for name in (None, "default-v1"):
            P.recount(ledger, P.with_mapping(P.load_map(), name))


# ---- user ruling R9.4 (sixteenth set): a reading with three pre-declared outcomes -------------
# results/dataprep/rulings/user_rulings_2026-11-07c.json (sha256 338f2de4)
REP3 = "reports/R3/report.json"
OUTCOMES3 = {"measured": "D-measured", "weak": "D-weak", "unchanged": "D-unknown",
             "not_evaluable": "D-unknown"}


def _three(outcome=None, cite_reg=True, cite_rep=True):
    """F1 (D, forecast part) decided by REG.three.v1's recorded outcome."""
    ledger, pmap = _synthetic()
    pmap["claims"]["F1"]["grade"] = {"D": {"D": {
        "measured_by": "REG.three.v1", "reading_report": REP3,
        "reading_outcomes": dict(OUTCOMES3), "otherwise": "D-unknown"}}}
    f1 = ledger["claims"][7]
    assert f1["id"] == "F1"
    if cite_reg:
        ledger["registry_hashes"]["REG.three.v1"] = "f" * 64
    if cite_rep:
        f1["cites"] = {"reports": [REP3]}
    if outcome is not None:
        f1["decidability_reading"] = {"registration": "REG.three.v1", "report": REP3,
                                      "outcome": outcome}
    return ledger, pmap


def test_a_three_outcome_reading_grades_by_its_recorded_outcome():
    fc = {}
    for out, key in OUTCOMES3.items():
        ledger, pmap = _three(out)
        res = P.recount(ledger, pmap)
        assert _row(res, "F1")["grade_key"] == key, out
        assert "REG.three.v1 outcome" in _row(res, "F1")["grade_why"]
        fc[out] = res["parts"][PARTS[1]]["official"]
    # F1 (weight 1) and F2 (N, weight 1): the outcome moves the part by +-0.1 / 2
    assert fc["measured"] - fc["unchanged"] == pytest.approx(5.0)
    assert fc["weak"] - fc["unchanged"] == pytest.approx(-5.0)
    assert fc["not_evaluable"] == fc["unchanged"]
    # before the ledger cites the registration and the report: 'otherwise', whatever happened
    for kw in ({"cite_reg": False, "cite_rep": False}, {"cite_rep": False},
               {"cite_reg": False}):
        ledger, pmap = _three(None, **kw)
        r = _row(P.recount(ledger, pmap), "F1")
        assert r["grade_key"] == "D-unknown" and "no outcome recorded" in r["grade_why"], kw


def test_a_three_outcome_reading_refuses_an_inconsistent_ledger():
    ledger, pmap = _three(None)                   # both cited, outcome not recorded
    with pytest.raises(P.MappingError, match="F1: cites REG.three.v1 and its report but "
                                             "records no outcome"):
        P.recount(ledger, pmap)
    for kw in ({"cite_rep": False}, {"cite_reg": False}):      # an outcome cannot hide
        ledger, pmap = _three("weak", **kw)
        with pytest.raises(P.MappingError, match="does not cite both"):
            P.recount(ledger, pmap)
    ledger, pmap = _three("decisive")
    with pytest.raises(P.MappingError, match="'decisive' is not one REG.three.v1 declared"):
        P.recount(ledger, pmap)
    ledger, pmap = _three("measured")
    ledger["claims"][7]["decidability_reading"]["registration"] = "REG.other.v1"
    with pytest.raises(P.MappingError, match="another reading"):
        P.recount(ledger, pmap)
    ledger, pmap = _three("measured")
    ledger["registry_hashes"]["REG.three.v1"] = "PENDING"
    with pytest.raises(P.MappingError, match="PENDING"):
        P.recount(ledger, pmap)
    ledger, pmap = _three("measured")
    pmap["claims"]["F1"]["grade"]["D"]["D"]["reading_outcomes"]["weak"] = "D-feeble"
    with pytest.raises(P.MappingError, match="reading_outcomes needs a reading_report and "
                                             "grade keys"):
        P.recount(ledger, pmap)


def test_e02_follows_its_r9_4_reading_and_is_unchanged_now():
    """E02's D qualifier comes from E02.decidability.v1's pre-declared outcome once the ledger
    cites it; on the committed ledger it is still D with decidability unknown."""
    pmap = P.load_map()
    rep = "results/dataprep/reports/E02.decidability.v1/report.json"
    assert pmap["claims"]["E02"]["grade"] == {"D": {"D": {
        "measured_by": "E02.decidability.v1", "reading_report": rep,
        "reading_outcomes": OUTCOMES3, "otherwise": "D-unknown"}}}
    assert "R9.4" in pmap["rule"] and "338f2de4" in pmap["rule"]
    assert "R9.4" in pmap["claims"]["E02"]["reason"]
    ledger = _ledger()
    base = P.recount(ledger, pmap)
    assert _row(base, "E02")["grade_key"] == "D-unknown" and _row(base, "E02")["weight"] == 1
    fc0 = base["parts"]["forecast_start_validation"]
    for out, key in OUTCOMES3.items():           # what-if: the ledger change after the run
        lg = copy.deepcopy(ledger)
        lg["registry_hashes"]["E02.decidability.v1"] = "f" * 64
        e = next(c for c in lg["claims"] if c["id"] == "E02")
        e["cites"]["registrations"].append("E02.decidability.v1")
        e["cites"]["reports"].append(rep)
        e["decidability_reading"] = {"registration": "E02.decidability.v1", "report": rep,
                                     "outcome": out}
        res = P.recount(lg, pmap)
        assert _row(res, "E02")["grade_key"] == key, out
        delta = {"D-measured": 0.1, "D-weak": -0.1, "D-unknown": 0.0}[key]
        assert res["parts"]["forecast_start_validation"]["official"] - fc0["official"] == \
            pytest.approx(100 * delta / fc0["weight"]), out
