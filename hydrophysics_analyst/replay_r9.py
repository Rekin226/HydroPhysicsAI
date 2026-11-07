"""Deterministic $0 replay of the ROUND-9 guards on a held-out set's recorded rows (ruling R10.3b).

    python -m hydrophysics_analyst replay-r9 --page results/twin/twin_app.html \\
        --questions hydrophysics_analyst/evalsets/eval_questions_heldout_v5.jsonl \\
        --rows results/analyst/eval/heldout_v5/opus-low.jsonl [--out <dir>]

Pre-declared rule (R10.3b): every item the round-9 ``hard_refusal`` (the frozen copy in
``router.r9``, as at fc7b996) would refuse gets that refusal, scored as the eval scores it; every
other item keeps the deployed arm's recorded answer, row by row (all repeats). The summary has
the same metrics as ``eval.summarize``; it is reported beside the deployed arm's score, never in
its place. No API call: replayed refusals cost nothing.

Only ``hard_refusal`` is replayed. The round-10 rules-stage changes (A4's confident
ambiguous_town clarification, A5's "as of <year>") are not undone by the replay; ``info`` counts
the rows the round-10 hard_refusal refused that round 9 would not have (expected 0: round 10
only narrows).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import eval as E
from .ask import answer_route
from .page import PageBundle
from .router import r9
from .router.rules import detect_lang
from .router.schemas import Route


def r9_route(b: PageBundle, question: str) -> Route | None:
    """The round-9 hard refusal of ``question`` as a route (with the Changhua lever label), or
    None when round 9 would not refuse it at that stage."""
    ref = r9.hard_refusal(question, b)
    if not ref:
        return None
    params = {"reason": ref}
    if ref == "changhua_lever":
        params["lever"] = r9.changhua_kind(question)
    return Route(preset="refuse", params=params, refusal=ref, lang=detect_lang(question),
                 confidence=1.0)


def replay(b: PageBundle, items: list[dict], rows: list[dict]) -> tuple[list[dict], dict]:
    """(replayed rows, info). Rows keep their order; a replaced row carries ``replayed: True``
    and the recorded outcome under ``recorded``."""
    by_id = {it["id"]: it for it in items}
    unknown = sorted({r["id"] for r in rows} - set(by_id))
    if unknown:
        raise ValueError(f"rows for items not in the question set: {unknown[:5]}")
    decided = {i: r9_route(b, it["question"]) for i, it in by_id.items()}
    out, info = [], {"rows": len(rows), "rows_replaced": 0, "items_refused_by_r9": 0,
                     "outcome_changed": 0, "r10_only_refusal_rows": 0}
    info["items_refused_by_r9"] = sum(r is not None for r in decided.values())
    for row in rows:
        it, rr = by_id[row["id"]], decided[row["id"]]
        if rr is None:
            if row.get("stage") == "hard_refusal":
                info["r10_only_refusal_rows"] += 1
            out.append(dict(row, replayed=False))
            continue
        lang = rr.lang or it["lang"]
        s = E.score(b, it, answer_route(b, rr, lang))
        new = {"id": row["id"], "arm": f"{row.get('arm')}+r9", "stage": "hard_refusal",
               "unrouted": False, "llm_error": None, "latency_ms": 0, "tags": it.get("tags", []),
               **s, "replayed": True,
               "recorded": {k: row.get(k) for k in ("stage", "ok", "unsafe", "got")}}
        info["rows_replaced"] += 1
        info["outcome_changed"] += (new["ok"], new["unsafe"]) != (row.get("ok"), row.get("unsafe"))
        out.append(new)
    return out, info


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--rows", type=Path, required=True, help="the deployed arm's recorded rows")
    ap.add_argument("--out", type=Path, default=None, help="write <arm>+r9.jsonl and summary here")
    ap.add_argument("--unbound", action="store_true", help="dev pages only")
    a = ap.parse_args(argv)
    try:
        entry = E.check_set(a.questions)          # the frozen bytes (no attempt is logged)
    except E.EvalSetError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 4
    b = PageBundle.load(a.page, a.ledger, allow_pending=a.unbound, check_forward=not a.unbound,
                        check_template=not a.unbound)
    items = E.load(a.questions)
    rows = load_rows(a.rows)
    new, info = replay(b, items, rows)
    res = {"questions": entry["name"], "questions_sha256": entry["sha256"],
           "page_sha256": b.page_sha256, "ledger_sha256": b.ledger_sha256,
           "rows_from": a.rows.name, "frozen_at": r9.FROZEN_AT,
           "rule": "R10.3b: round-9 hard_refusal refusals replace the recorded answer; every other "
                   "row keeps the deployed arm's recorded answer; reported beside the score",
           "info": info, "summary": E.summarize(new, len(items)),
           "recorded_summary": E.summarize(rows, len(items))}
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        arm = (rows[0].get("arm") if rows else "arm") or "arm"
        (a.out / f"{arm}+r9.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in new) + "\n", encoding="utf-8")
        (a.out / "replay_r9_summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False),
                                                      encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
