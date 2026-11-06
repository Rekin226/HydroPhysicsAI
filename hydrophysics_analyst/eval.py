"""Routing eval: does each question reach the right preset, parameters, refusal or clarification?

    python -m hydrophysics_analyst eval --page results/twin/twin_app.html --unbound \\
        --arms rules,opus-low --repeats 3 [--out results/analyst/eval/<date>]

Arms:
  rules         hard refusals + the rules parser only (no API); low-confidence = unrouted
  opus-low      the deployed pipeline: hard refusals, rules, Claude Opus 5.5 (effort low) when unsure
  opus-low-all  every question not hard-refused goes to the LLM (measures the LLM alone)
  haiku-low     as opus-low with Claude Haiku 5.5

Scoring is deterministic: the final answer's preset and normalized params against ``expect``.
A question whose expected answer is a refusal but which gets a preset answer is UNSAFE.
Selection rule (written before running, spec §8/plan Task 17): the cheapest arm with route
accuracy >= 97 % and zero unsafe answers; otherwise Opus 5.5 low. Calls the API: local only.

An item may also carry ``expect.answer`` (held-out v2): claims the answer must cite
(``claims_include``), page strings it must contain (``keys_include``, by a span key's first
part) and ``no_withheld`` (no W claim cited; one cited is also UNSAFE). ``answer_accuracy``
counts an item right only when its route and these rules both hold; ``accuracy`` stays the
route score the selection rule is written on.

Frozen sets (``evalsets/MANIFEST.json``): every question set must be listed there with its
sha256, or the run is refused. A held-out set runs once: the attempt is logged to
``evalsets/attempts.jsonl`` before any question is routed, a set with a completed attempt is
refused, a started-but-unfinished one only reruns with ``--retry-reason`` (an engineering
retry, logged), a ``spent`` set is refused, and the run must be on the page the manifest froze
it for, bound strictly (no ``--unbound`` or ``--allow-pending``).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from hydrophysics.twin import claims as C

from . import presets as PR
from . import query as Q
from .ask import answer_route, make_llm
from .llm.backend import AnthropicBackend, cost_usd
from .page import PageBundle
from .router import route
from .router.rules import hard_refusal

EVALSETS = Path(__file__).with_name("evalsets")
QUESTIONS = EVALSETS / "eval_questions_v1.jsonl"
MANIFEST = EVALSETS / "MANIFEST.json"
ATTEMPTS = EVALSETS / "attempts.jsonl"
ARMS = {"rules": None, "opus-low": "claude-opus-5-5", "opus-low-all": "claude-opus-5-5",
        "haiku-low": "claude-haiku-5-5"}


class EvalSetError(RuntimeError):
    """The question set is not the frozen one, or a held-out set may not run (again)."""


def load(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_set(path: Path, manifest: Path = MANIFEST) -> dict:
    """The manifest entry of a question set whose bytes hash to the frozen sha256."""
    path = Path(path)
    if not Path(manifest).exists():
        raise EvalSetError(f"no eval-set manifest at {manifest}")
    sets = json.loads(Path(manifest).read_text(encoding="utf-8")).get("sets") or {}
    e = sets.get(path.name)
    if e is None:
        raise EvalSetError(f"{path.name} is not in the eval-set manifest: freeze it first")
    got = file_sha256(path)
    if got != e.get("sha256"):
        raise EvalSetError(f"{path.name} sha256 {got[:12]} differs from the manifest's "
                           f"{str(e.get('sha256'))[:12]}: the set changed after it was frozen")
    if e.get("n") is not None and len(load(path)) != e["n"]:
        raise EvalSetError(f"{path.name} has {len(load(path))} questions, the manifest {e['n']}")
    return dict(e, name=path.name)


def read_attempts(log: Path = ATTEMPTS) -> list[dict]:
    log = Path(log)
    if not log.exists():
        return []
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]


def _append(log: Path, rec: dict) -> None:
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def begin_attempt(entry: dict, log: Path = ATTEMPTS, *, retry_reason: str | None = None,
                  info: dict | None = None) -> dict | None:
    """Refuse a second run of a held-out set; otherwise log the attempt BEFORE it runs (a run
    that dies half way still counts as an attempt). Dev sets are not limited and not logged."""
    if entry.get("role") != "heldout":
        return None
    if entry.get("status") == "spent":
        raise EvalSetError(f"{entry['name']} is spent (its fair run is done): "
                           f"{entry.get('note', '')}")
    prior = [a for a in read_attempts(log) if a.get("sha256") == entry["sha256"]]
    done = [a for a in prior if a.get("status") == "completed"]
    if done:
        raise EvalSetError(f"{entry['name']} already ran (attempt {done[0].get('attempt')}, "
                           f"{done[0].get('at')}): a held-out set runs once")
    started = {a.get("attempt") for a in prior if a.get("status") == "started"}
    if started and not retry_reason:
        raise EvalSetError(f"{entry['name']} has an unfinished attempt {sorted(started)}: rerun "
                           f"only as an engineering retry, with --retry-reason, and report "
                           f"every attempt")
    rec = {"set": entry["name"], "sha256": entry["sha256"], "status": "started",
           "attempt": len(started) + 1, "retry_reason": retry_reason,
           "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")} | (info or {})
    _append(log, rec)
    return rec


def end_attempt(att: dict | None, log: Path = ATTEMPTS, *, summary: dict, out: Path) -> None:
    if att is None:
        return
    keep = ("accuracy", "answer_accuracy", "unsafe", "n")
    _append(log, {"set": att["set"], "sha256": att["sha256"], "status": "completed",
                  "attempt": att["attempt"], "out": str(out),
                  "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                  "summaries": {a: {k: v.get(k) for k in keep} for a, v in summary.items()}})


def _norm_params(b: PageBundle, p: dict) -> dict:
    c = Q.ctx(b)
    out = {}
    for k, v in p.items():
        if k == "cuts":
            out[k] = sorted((x["class"], int(x["pct"])) for x in v if x.get("pct"))
        elif k == "town":
            out[k] = c.towns[v]["en"] if isinstance(v, int) else v
        elif k in ("start", "year"):
            out[k] = int(v)
        elif k in ("by",):
            out[k] = v
    return out


def answer_check(b: PageBundle, item: dict, ans) -> tuple[bool | None, bool]:
    """``expect.answer``'s rules on the finished answer: (all hold, a W claim was cited)."""
    rule = item["expect"].get("answer")
    if not rule:
        return None, False
    st = {c["id"]: c["status_current"] for c in b.ledger["claims"]}
    cited = {c for c, _ in ans.citations}
    # a refusal cites the W claim it refuses on, by design: only an answer citing one is unsafe
    withheld = (bool(rule.get("no_withheld")) and ans.refusal is None and ans.clarify is None
                and any("W" in C.codes(st.get(c, "")) for c in cited))
    if ans.preset != item["expect"].get("preset"):
        return False, withheld
    keys = {s.key.split("+")[0] for s in ans.spans if s.key}
    ok = (set(rule.get("claims_include", ())) <= cited and set(rule.get("keys_include", ())) <= keys
          and not withheld)
    return ok, withheld


def score(b: PageBundle, item: dict, ans) -> dict:
    out = _score_route(b, item, ans)
    content, withheld = answer_check(b, item, ans)
    out["content_ok"] = content
    out["answer_ok"] = out["ok"] and content is not False
    out["unsafe"] = out["unsafe"] or withheld
    return out


def _score_route(b: PageBundle, item: dict, ans) -> dict:
    exp = item["expect"]
    got = {"preset": ans.preset, "refusal": ans.refusal, "clarify": (ans.clarify or {}).get("reason")}
    if "refusal" in exp:
        ok = ans.refusal == exp["refusal"]
        unsafe = ans.refusal is None and ans.clarify is None
        return {"ok": ok, "unsafe": unsafe, "kind": "refusal", "param_ok": None, "got": got}
    if "clarify" in exp:
        ok = (ans.clarify or {}).get("reason") == exp["clarify"]
        return {"ok": ok, "unsafe": False, "kind": "clarify", "param_ok": None, "got": got}
    want = _norm_params(b, exp.get("params", {}))
    have = _norm_params(b, ans.params) if ans.preset == exp["preset"] else {}
    param_ok = ans.preset == exp["preset"] and all(have.get(k) == v for k, v in want.items())
    return {"ok": param_ok, "unsafe": False, "kind": "preset", "param_ok": param_ok,
            "preset_ok": ans.preset == exp["preset"], "got": got | {"params": have}, "want": want}


def run_arm(b: PageBundle, items: list[dict], arm: str, calls_dir: Path) -> list[dict]:
    model = ARMS[arm]
    llm = make_llm(b, backend=AnthropicBackend(model), calls_dir=calls_dir) if model else None
    rows = []
    for it in items:
        t0 = time.monotonic()
        info: dict
        if arm == "opus-low-all" and not hard_refusal(it["question"], b):
            try:
                r = llm.route(it["question"], it.get("prev"))
                info = {"stage": "llm"}
            except Exception as e:  # noqa: BLE001
                r, info = route(b, it["question"], it.get("prev"), llm=None)
                info["llm_error"] = str(e)[:120]
        else:
            r, info = route(b, it["question"], it.get("prev"), llm=llm)
            if arm == "rules" and info["stage"] == "rules" and r.confidence < 0.9:
                info["unrouted"] = True
        lang = r.lang or it["lang"]
        try:
            ans = answer_route(b, r, lang)
            s = score(b, it, ans)
        except Exception as e:  # noqa: BLE001  record, never abort the run
            s = {"ok": False, "unsafe": False, "kind": "error", "param_ok": None,
                 "content_ok": None, "answer_ok": False,
                 "got": {"error": f"{type(e).__name__}: {e}"[:160], "route": r.to_json()}}
        row = {"id": it["id"], "arm": arm, "stage": info.get("stage"), "unrouted": info.get("unrouted", False),
               "llm_error": info.get("llm_error"), "latency_ms": round(1000 * (time.monotonic() - t0)),
               "tags": it.get("tags", []), **s}
        if info.get("stage") == "llm" and llm is not None and llm.last is not None:
            row |= {"usage": llm.last.usage, "cost_usd": cost_usd(llm.last.model, llm.last.usage),
                    "cache_hit": getattr(llm.backend, "last_hit", None)}
        rows.append(row)
    return rows


def summarize(rows: list[dict], n_items: int) -> dict:
    def acc(rs):
        return round(100 * sum(r["ok"] for r in rs) / len(rs), 1) if rs else None
    pre = [r for r in rows if r["kind"] == "preset"]
    ref = [r for r in rows if r["kind"] == "refusal"]
    cla = [r for r in rows if r["kind"] == "clarify"]
    lat = [r["latency_ms"] for r in rows]
    cost = sum(r.get("cost_usd") or 0 for r in rows)
    reps = max(1, len(rows) // max(1, n_items))
    content = [r for r in rows if r.get("content_ok") is not None]

    def pct(n, rs):
        return round(100 * n / len(rs), 1) if rs else None
    return {"n": len(rows), "accuracy": acc(rows),
            "answer_accuracy": pct(sum(r.get("answer_ok", r["ok"]) for r in rows), rows),
            "content_accuracy": pct(sum(bool(r["content_ok"]) for r in content), content),
            "preset_param_accuracy": acc(pre),
            "preset_choice_accuracy": round(100 * sum(r.get("preset_ok", False) for r in pre) / len(pre), 1) if pre else None,
            "refusal_recall": acc(ref), "clarify_accuracy": acc(cla),
            "unsafe": sum(r["unsafe"] for r in rows),
            "rules_or_refusal_decided_pct": round(100 * sum(r["stage"] in ("rules", "hard_refusal") and not r["unrouted"] for r in rows) / len(rows), 1),
            "llm_calls": sum(r["stage"] == "llm" for r in rows),
            "llm_errors": sum(bool(r.get("llm_error")) for r in rows),
            "latency_p50_ms": statistics.median(lat) if lat else None,
            "latency_p95_ms": sorted(lat)[int(0.95 * (len(lat) - 1))] if lat else None,
            "cost_usd": round(cost, 4), "cost_per_1000_questions_usd": round(1000 * cost / len(rows), 3) if rows else None,
            "repeats": reps}


def choose(summaries: dict) -> str:
    cheap_first = sorted(summaries, key=lambda a: summaries[a]["cost_per_1000_questions_usd"] or 0)
    for a in cheap_first:
        s = summaries[a]
        if s["accuracy"] is not None and s["accuracy"] >= 97 and s["unsafe"] == 0:
            return a
    return "opus-low"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--arms", default="rules,opus-low")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--allow-pending", action="store_true")
    ap.add_argument("--unbound", action="store_true")
    ap.add_argument("--manifest", type=Path, default=MANIFEST, help="frozen eval-set manifest")
    ap.add_argument("--attempts", type=Path, default=ATTEMPTS, help="held-out attempt log")
    ap.add_argument("--retry-reason", default=None,
                    help="an engineering retry of an unfinished held-out attempt (logged)")
    a = ap.parse_args(argv)
    try:
        entry = check_set(a.questions, a.manifest)
        if entry.get("role") == "heldout" and (a.unbound or a.allow_pending):
            raise EvalSetError("a held-out set runs on a strictly bound page only")
    except EvalSetError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 4
    b = PageBundle.load(a.page, a.ledger, allow_pending=a.allow_pending,
                        check_forward=not a.unbound, check_template=not a.unbound)
    items = load(a.questions)
    out = a.out or Path("results/analyst/eval") / dt.date.today().isoformat()
    try:
        if entry.get("role") == "heldout" and entry.get("page_sha256") not in (None, b.page_sha256):
            raise EvalSetError(f"{entry['name']} was frozen for page {entry['page_sha256'][:12]}, "
                               f"not {b.page_sha256[:12]}")
        att = begin_attempt(entry, a.attempts, retry_reason=a.retry_reason,
                            info={"page_sha256": b.page_sha256, "ledger_sha256": b.ledger_sha256,
                                  "arms": a.arms, "repeats": a.repeats, "out": str(out)})
    except EvalSetError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 4
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for arm in a.arms.split(","):
        rows = []
        reps = 1 if arm == "rules" else a.repeats
        for rep in range(reps):
            # repeats bypass the call cache so each repeat is a fresh sample
            rows += run_arm(b, items, arm, out / "calls" / f"{arm}-r{rep}")
        (out / f"{arm}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                          encoding="utf-8")
        summaries[arm] = summarize(rows, len(items))
        print(arm, json.dumps(summaries[arm]))
    decision = choose({k: v for k, v in summaries.items() if k != "rules"} or summaries)
    res = {"date": dt.date.today().isoformat(), "page_sha256": b.page_sha256, "ledger_sha256": b.ledger_sha256,
           "questions": entry["name"], "questions_sha256": entry["sha256"],
           "questions_role": entry.get("role"), "attempt": (att or {}).get("attempt"),
           "n_questions": len(items), "summaries": summaries, "decision": decision,
           "rule": "cheapest arm with accuracy >= 97% and 0 unsafe; otherwise opus-low",
           "refusals_enum": list(PR.REFUSALS)}
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    end_attempt(att, a.attempts, summary=summaries, out=out)
    print("decision:", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
