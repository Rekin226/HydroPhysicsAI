"""Command line: answer a preset with explicit parameters, or a plain-language question.

    python -m hydrophysics_analyst preset policy_head --cut aquaculture=30 --start 2030 \\
        --year 2032 --lang zh --page results/twin/twin_app_20261108c.html
    python -m hydrophysics_analyst ask "停止養殖抽水對2032年水位影響？" --page ... [--no-llm]

``--allow-pending`` and ``--unbound`` are development switches (a banner marks every answer);
``--mode deploy`` refuses both. ``--json`` prints the full answer object.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .page import BindingError, PageBundle


def _common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, default=None)
    ap.add_argument("--lang", choices=["en", "zh"], default=None,
                    help="answer language (default: the question's; en for presets)")
    ap.add_argument("--mode", choices=["dev", "deploy"], default="dev")
    ap.add_argument("--allow-pending", action="store_true")
    ap.add_argument("--unbound", action="store_true", help="skip the page/ledger pins (dev only)")
    ap.add_argument("--json", action="store_true")


def load_bundle(a) -> PageBundle:
    return PageBundle.load(a.page, a.ledger, mode=a.mode, allow_pending=a.allow_pending,
                           check_forward=not a.unbound, check_template=not a.unbound)


def _print(ans, as_json: bool, violations) -> int:
    if as_json:
        d = ans.to_json() | {"violations": [v.__dict__ for v in violations]}
        print(json.dumps(d, ensure_ascii=False, indent=1))
    else:
        print(ans.text())
        for v in violations:
            print(f"VALIDATOR: {v.rule}: {v.detail}", file=sys.stderr)
    return 1 if violations else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="hydrophysics_analyst", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("preset")
    pp.add_argument("name")
    pp.add_argument("--cut", action="append", default=[], help="class=pct, e.g. aquaculture=30")
    pp.add_argument("--start", type=int)
    pp.add_argument("--year", type=int)
    pp.add_argument("--town")
    pp.add_argument("--by", choices=["head", "gwh"])
    pp.add_argument("--target-m", type=float)
    pp.add_argument("--reason")
    _common(pp)
    pa = sub.add_parser("ask")
    pa.add_argument("question")
    pa.add_argument("--no-llm", action="store_true")
    pa.add_argument("--no-api", action="store_true", help="replay LLM calls from the cache only")
    pa.add_argument("--model", default=None)
    pa.add_argument("--audit-dir", type=Path, default=Path("results/analyst/audit"))
    pa.add_argument("--user", default="cli")
    _common(pa)
    a = ap.parse_args(argv)
    try:
        b = load_bundle(a)
    except BindingError as e:
        print(f"binding failed: {e}", file=sys.stderr)
        return 3
    from . import presets as PR
    from .validate import validate
    if a.cmd == "preset":
        params: dict = {}
        if a.cut or a.name in ("policy_head", "start_compare", "township_internals", "rail_internals"):
            params["cuts"] = [{"class": k, "pct": int(v)} for k, v in (c.split("=") for c in a.cut)]
        for k in ("start", "year", "town", "by", "reason"):
            if getattr(a, k) is not None:
                params[k] = getattr(a, k)
        if a.target_m is not None:
            params["target_m"] = a.target_m
        ans = PR.run(b, a.name, params, a.lang or "en")
        return _print(ans, a.json, validate(ans, b))
    from .ask import ask
    ans, violations, _ = ask(b, a.question, lang=a.lang, use_llm=not a.no_llm, no_api=a.no_api,
                             model=a.model, audit_dir=a.audit_dir, user=a.user)
    return _print(ans, a.json, violations)


if __name__ == "__main__":
    raise SystemExit(main())
