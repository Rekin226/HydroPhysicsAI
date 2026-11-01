"""Append-only audit log and replay.

    python -m hydrophysics_analyst replay <id> --page ... [--audit-dir results/analyst/audit]

Each answer appends one JSON line to ``<audit-dir>/YYYY-MM-DD.jsonl``: id, time, user, language,
question, the route and which stage decided it, the LLM model/request id/usage/cost when used,
the answer text, claims and validator verdict, the deep link, and the page/ledger/strings shas
and code version. ``replay`` re-runs the logged route through the presets — no LLM — and must
reproduce the answer text exactly on the same page and ledger. The API key is never logged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import uuid
from pathlib import Path

from . import __version__
from .page import PageBundle

DEFAULT_DIR = Path("results/analyst/audit")


def write(record: dict, directory: Path = DEFAULT_DIR) -> str:
    now = dt.datetime.now(dt.timezone.utc)
    rid = record.get("id") or f"{now:%Y%m%d}-{uuid.uuid4().hex[:10]}"
    record = {"id": rid, "ts": now.isoformat(), "code_version": __version__} | record
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return rid


def find(rid: str, directory: Path = DEFAULT_DIR) -> dict:
    for f in sorted(Path(directory).glob("*.jsonl"), reverse=True):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("id") == rid:
                    return d
    raise KeyError(rid)


class ReplayMismatch(RuntimeError):
    pass


def replay(rid: str, b: PageBundle, directory: Path = DEFAULT_DIR):
    from . import presets as PR
    rec = find(rid, directory)
    if rec["page_sha256"] != b.page_sha256 or rec["ledger_sha256"] != b.ledger_sha256:
        raise ReplayMismatch("the audit record was made on a different page or ledger")
    preset, params, lang = rec["answer_preset"], rec["answer_params"], rec["lang"]
    if preset == "clarify":
        raise ReplayMismatch("a clarification has no answer to replay")
    ans = PR.run(b, preset, params, lang)
    if ans.text() != rec["answer_text"]:
        raise ReplayMismatch("replayed text differs from the logged answer")
    return ans


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id")
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--audit-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--allow-pending", action="store_true")
    ap.add_argument("--unbound", action="store_true")
    a = ap.parse_args(argv)
    b = PageBundle.load(a.page, a.ledger, allow_pending=a.allow_pending,
                        check_forward=not a.unbound, check_template=not a.unbound)
    try:
        ans = replay(a.id, b, a.audit_dir)
    except (KeyError, ReplayMismatch) as e:
        print(f"replay failed: {e}", file=sys.stderr)
        return 1
    print(ans.text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
