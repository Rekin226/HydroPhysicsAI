"""Claim binding agrees with the committed ledger (structure only: allow_pending)."""
from __future__ import annotations

import json
import re

from hydrophysics.twin import claims
from hydrophysics_analyst import binding as BD

LEDGER = claims.load_ledger(allow_pending=True)
T_EN, T_ZH = claims.i18n(lang="en"), claims.i18n(lang="zh")


def test_every_rule_cites_claims_its_paths_give():
    paths = {   # the paths/functions each query value carries (query.py)
        "head_change": (("n2Unit.solved[][]", "n2Unit.basis[][]", "arrays.dHeadL2"),
                        ("headWhere", "renderStartNote")),
        "energy_forgone": (("classes[].gwh",), ("renderTiles",)),
        "class_rank": (("n2Unit.basis[][]", "arrays.dHeadL2"), ("renderRank",)),
        "avoided_subsidence": (("arrays.dSubs",), ("renderTowns",)),
        "rail_angular_distortion": (("hsr.ch[]",), ("renderHsr", "railLine")),
    }
    for r in BD.rules().values():
        if not r.offered or not r.claims:
            continue
        p, f = paths[r.quantity]
        assert set(r.claims) <= BD.ledger_ids(p, f, LEDGER), r


def test_required_labels_exist_in_both_languages():
    for r in BD.rules().values():
        for k in r.required_label:
            assert k in T_EN and k in T_ZH, k


def test_scoring_dates_appear_in_the_ledger_wording():
    """Checked for every scoring claim the bound ledger has (the ledger moves: C31 arrives in
    round 6). At least the N2 head claim must be present: since the round-8 split (ruling R8.8)
    it is C01-w, N2's within-window response (C01 keeps the direction, S)."""
    by = {c["id"]: c for c in LEDGER["claims"]}
    present = [cid for cid in BD.scoring() if cid in by]
    assert "C01-w" in present
    for cid in present:
        s = BD.scoring()[cid]
        blob = by[cid]["wording_en"] + json.dumps(by[cid].get("outcome_display", {}))
        assert s["not_before"] in blob, cid
    # a claim whose registration gives no scoring date says so, in the rules and the ledger
    for cid, s in BD.scoring_undated().items():
        assert cid in by and "no registered scoring date" in s["note"], cid
        assert "no registered scoring date" in by[cid]["wording_en"] + by[cid]["notes"], cid


def test_classes_follow_the_status_codes():
    for r in BD.rules().values():
        if r.claims:
            assert BD.class_of(r.claims, LEDGER) == r.answer_class, r
        if r.answer_class == "answer":
            st = {c["id"]: c["status_current"] for c in LEDGER["claims"]}
            assert not any(set(claims.codes(st[c])) & {"W", "L"} for c in r.claims)
    labelled = [r for r in BD.rules().values() if r.answer_class == "labelled"]
    assert labelled and all(r.claims == ("C08",) for r in labelled)


def test_topics_are_digit_free_and_known():
    ids = {c["id"] for c in LEDGER["claims"]}
    for cid, t in BD.topics().items():
        assert cid in ids and not re.search(r"\d", t)


def test_no_withheld_rule_is_offered():
    for r in BD.rules().values():
        if r.answer_class == "withheld":
            assert not r.offered
