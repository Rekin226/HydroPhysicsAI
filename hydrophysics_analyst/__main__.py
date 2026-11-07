"""python -m hydrophysics_analyst <command> ...

Commands:
  preset   answer one preset with explicit parameters (no LLM)
  ask      answer a question in plain language (rules router, then the LLM router)
  serve    the local web shell (chat beside the page)
  replay   re-run a logged answer from the audit log, without the LLM
  eval     the routing eval (local; calls the API)
  parity   JS-vs-Python parity report for a page (local; needs Playwright)
  mcp      an MCP stdio server exposing the presets (team-internal)
  bundle   build a private deployment bundle for a page
  replay-r9  the round-9 guards replayed at $0 on a held-out set's recorded rows (R10.3b)
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {"preset": "cli", "ask": "cli", "serve": "server", "replay": "audit", "eval": "eval",
            "parity": "parity", "mcp": "mcp_server", "bundle": "bundle",
            "replay-r9": "replay_r9"}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        return 2
    mod = importlib.import_module(f"hydrophysics_analyst.{COMMANDS[cmd]}")
    if COMMANDS[cmd] == "cli":
        return int(mod.main([cmd, *rest]) or 0)
    return int(mod.main(rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
