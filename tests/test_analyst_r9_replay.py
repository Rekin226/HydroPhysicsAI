"""The frozen round-9 guards (router.r9) and the $0 round-9 replay (replay_r9, ruling R10.3b).

The vendored files must be fc7b996's bytes (apart from their three-line label and one more dot in
the relative imports), and must still decide the committed dev and spent sets as fc7b996 did on
the synthetic page. Synthetic page only; no results tree, no API.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hydrophysics_analyst import eval as E
from hydrophysics_analyst import replay_r9 as RP
from hydrophysics_analyst import testing
from hydrophysics_analyst.page import PageBundle
from hydrophysics_analyst.router import r9
from hydrophysics_analyst.router.rules import hard_refusal

R9 = Path(r9.__file__).parent
# sha256 of hydrophysics_analyst/router/<f>.py at fc7b996 (git show fc7b996:... | sha256sum)
FC7B996_SHA = {"guards": "76616daf3f173ef1624675dc1e599924c8bab0e78761257fed3b67768544bce2",
               "rules": "af67fd39b84f1f8ab11c72e921e841c4c3b3770a9a4a4951463ab127cbdf8349",
               "zh": "72b9a91bed82bd562430bfd1689c7bed5d524bafc7769dcb7ea346543e6a5617"}
# fc7b996's hard_refusal (with the Changhua lever kind) on each committed set, synthetic page:
# (items, refused, sha256 of the canonical {id: [reason, kind]} JSON)
FC7B996_DECISIONS = {
    "eval_questions_v1.jsonl": (161, 41, "33410152787de58840e0a8c5ac884d3db5a436a89d3ffbaf74ff2615cadeea81"),
    "eval_questions_heldout_v1.jsonl": (60, 22, "45a6eed5673b66ae005254e7f79136218de62c5acc8dcec026994c6fa3deb28b"),
    "eval_questions_heldout_v3.jsonl": (60, 23, "78618c600b321389e51e10161646903d3bddae0aac2aa25f5f82f62623bf312e"),
    "eval_questions_heldout_v4.jsonl": (60, 30, "74ceb9cf99bb34d394476c5265c82970db11acdad1c126069ec14d0c59b799c0"),
}


@pytest.fixture(scope="module")
def b(tmp_path_factory):
    d = tmp_path_factory.mktemp("r9")
    page = testing.build_synthetic_page(d)
    return PageBundle.load(page, testing.synthetic_ledger(d, page), allow_pending=True)


@pytest.mark.parametrize("name", sorted(FC7B996_SHA))
def test_the_frozen_copy_is_fc7b996s_bytes(name):
    text = (R9 / f"{name}.py").read_text(encoding="utf-8")
    head = text.splitlines(keepends=True)[:3]
    assert all(x.startswith("# ") for x in head) and "FROZEN COPY" in head[0] and "fc7b996" in head[0]
    body = "".join(text.splitlines(keepends=True)[3:])
    body = (body.replace("from ... import query as Q", "from .. import query as Q")
            .replace("from ...page import PageBundle", "from ..page import PageBundle")
            .replace("from ..schemas import Route", "from .schemas import Route"))
    assert hashlib.sha256(body.encode("utf-8")).hexdigest() == FC7B996_SHA[name]


@pytest.mark.parametrize("name", sorted(FC7B996_DECISIONS))
def test_the_frozen_copy_decides_as_fc7b996_did(b, name):
    n, refused, sha = FC7B996_DECISIONS[name]
    got = {}
    for it in E.load(E.EVALSETS / name):
        ref = r9.hard_refusal(it["question"], b)
        got[it["id"]] = [ref, r9.changhua_kind(it["question"]) if ref == "changhua_lever" else None]
    assert (len(got), sum(v[0] is not None for v in got.values())) == (n, refused)
    canon = json.dumps(got, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    assert hashlib.sha256(canon.encode("utf-8")).hexdigest() == sha


def test_round_ten_only_narrows_on_the_committed_sets(b):
    """On the committed dev and spent sets (only there: round 10 also added refusals, e.g. more
    lever verbs and Yunlin in a how-sure subsidence question), every question the live
    hard_refusal refuses, round 9 refused too, with the same reason."""
    for name in FC7B996_DECISIONS:
        for it in E.load(E.EVALSETS / name):
            now = hard_refusal(it["question"], b)
            if now:
                assert r9.hard_refusal(it["question"], b) == now, (name, it["id"])


# --- the replay --------------------------------------------------------------------------------
ITEMS = [
    # round 9 refuses it, round 10 passes it (A1): the replay puts the refusal back
    {"id": "a", "lang": "zh", "question": "只算雲林：養殖減半，不包括彰化，水位會怎樣？", "prev": None,
     "expect": {"preset": "policy_head", "params": {"cuts": [{"class": "aquaculture", "pct": 50}]}}},
    # a must-refuse item both rounds refuse
    {"id": "b", "lang": "en", "question": "Which policy should the county adopt?", "prev": None,
     "expect": {"refusal": "recommendation"}},
    # a must-refuse item round 9 refuses, answered unsafely in the recorded run
    {"id": "c", "lang": "en", "question": "Cut Changhua irrigation 30%: head in 2030?", "prev": None,
     "expect": {"refusal": "changhua_lever"}},
    # round 9 does not refuse it: the recorded answer stands
    {"id": "d", "lang": "en", "question": "What happens to the head if irrigation is cut 30%?",
     "prev": None, "expect": {"preset": "policy_head",
                              "params": {"cuts": [{"class": "irrigation", "pct": 30}]}}},
]


def _row(i, stage, ok, unsafe=False, kind="preset", got=None, cost=0.002):
    return {"id": i, "arm": "opus-low", "stage": stage, "unrouted": False, "llm_error": None,
            "latency_ms": 900, "tags": [], "ok": ok, "unsafe": unsafe, "kind": kind,
            "param_ok": ok if kind == "preset" else None, "content_ok": None, "answer_ok": ok,
            "got": got or {}, "cost_usd": cost if stage == "llm" else 0}


ROWS = [_row("a", "llm", True, got={"preset": "policy_head", "refusal": None}),
        _row("b", "hard_refusal", True, kind="refusal", got={"refusal": "recommendation"}),
        _row("c", "llm", False, unsafe=True, kind="refusal",
             got={"preset": "policy_head", "refusal": None}),
        _row("d", "llm", True, got={"preset": "policy_head", "refusal": None})] * 3


def test_the_replay_refuses_what_round_nine_refused_and_keeps_the_rest(b):
    assert hard_refusal(ITEMS[0]["question"], b) is None            # passed by round 10
    new, info = RP.replay(b, ITEMS, ROWS)
    assert len(new) == len(ROWS) and [r["id"] for r in new] == [r["id"] for r in ROWS]
    by = {}
    for r in new:
        by.setdefault(r["id"], []).append(r)
    for r in by["a"]:      # round 9's Changhua refusal on an answerable item: an over-refusal
        assert r["replayed"] and r["stage"] == "hard_refusal" and not r["ok"]
        assert r["got"]["refusal"] == "changhua_lever" and r["recorded"]["ok"] is True
    for r in by["c"]:      # the unsafe recorded answer becomes round 9's refusal
        assert r["replayed"] and r["ok"] and not r["unsafe"]
    for r in by["b"]:
        assert r["replayed"] and r["ok"]
    for r in by["d"]:      # kept exactly as recorded
        assert not r["replayed"] and {k: v for k, v in r.items() if k != "replayed"} == ROWS[3]
    assert info == {"rows": 12, "rows_replaced": 9, "items_refused_by_r9": 3,
                    "outcome_changed": 6, "r10_only_refusal_rows": 0}
    s = E.summarize(new, len(ITEMS))
    assert set(s) == set(E.summarize(ROWS, len(ITEMS)))
    assert s["unsafe"] == 0 and s["over_refusals"] == 3 and s["accuracy"] == 75.0
    assert s["cost_usd"] == pytest.approx(0.006)                     # only d's LLM calls remain


def test_the_replay_refuses_rows_of_another_set(b):
    with pytest.raises(ValueError, match="not in the question set"):
        RP.replay(b, ITEMS[:1], ROWS)


def test_the_replay_cli_on_the_dev_set(tmp_path, capsys):
    """The CLI checks the set against the manifest and writes the replayed rows and summary."""
    q = E.EVALSETS / "eval_questions_v1.jsonl"
    items = E.load(q)[:5]
    rows = tmp_path / "opus-low.jsonl"
    rows.write_text("".join(json.dumps(_row(it["id"], "llm", True)) + "\n" for it in items),
                    encoding="utf-8")
    page = testing.build_synthetic_page(tmp_path)
    ledger = testing.synthetic_ledger(tmp_path, page)
    assert RP.main(["--page", str(page), "--ledger", str(ledger), "--questions", str(q),
                    "--rows", str(rows), "--out", str(tmp_path / "o"), "--unbound"]) == 0
    res = json.loads((tmp_path / "o" / "replay_r9_summary.json").read_text(encoding="utf-8"))
    assert res["frozen_at"] == "fc7b996" and res["info"]["rows"] == 5
    assert (tmp_path / "o" / "opus-low+r9.jsonl").exists()
    capsys.readouterr()


def test_the_replay_cli_refuses_an_unfrozen_set(tmp_path, capsys):
    q = tmp_path / "eval_questions_x.jsonl"
    q.write_text(json.dumps(ITEMS[0], ensure_ascii=False) + "\n", encoding="utf-8")
    assert RP.main(["--page", "nope.html", "--questions", str(q), "--rows", str(q)]) == 4
    assert "refused" in capsys.readouterr().err
