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
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import time
from pathlib import Path

from . import presets as PR
from . import query as Q
from .ask import answer_route, make_llm
from .llm.backend import AnthropicBackend, cost_usd
from .page import PageBundle
from .router import route
from .router.rules import hard_refusal

QUESTIONS = Path(__file__).with_name("data") / "eval_questions_v1.jsonl"
ARMS = {"rules": None, "opus-low": "claude-opus-5-5", "opus-low-all": "claude-opus-5-5",
        "haiku-low": "claude-haiku-5-5"}


def load(path: Path = QUESTIONS) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


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


def score(b: PageBundle, item: dict, ans) -> dict:
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
    return {"n": len(rows), "accuracy": acc(rows), "preset_param_accuracy": acc(pre),
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
    a = ap.parse_args(argv)
    b = PageBundle.load(a.page, a.ledger, allow_pending=a.allow_pending,
                        check_forward=not a.unbound, check_template=not a.unbound)
    items = load(a.questions)
    out = a.out or Path("results/analyst/eval") / dt.date.today().isoformat()
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
           "n_questions": len(items), "summaries": summaries, "decision": decision,
           "rule": "cheapest arm with accuracy >= 97% and 0 unsafe; otherwise opus-low",
           "refusals_enum": list(PR.REFUSALS)}
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("decision:", decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
