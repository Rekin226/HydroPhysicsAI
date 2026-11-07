"""MCP stdio server exposing the analyst to Claude Code / Claude Desktop. TEAM-INTERNAL ONLY.

    python -m hydrophysics_analyst mcp --page results/twin/twin_app_20261107a.html

Register in Claude Code, e.g. ``claude mcp add twin-analyst -- python -m hydrophysics_analyst mcp
--page <page>``. Tools: ``ask_twin`` (a plain-language question: the rules may refuse or clarify,
the LLM router chooses every preset) and one tool per preset with its enum-only schema. Every tool returns
a finished answer that already passed the validator and was written to the audit log, prefixed
with an instruction to relay it verbatim.

Limit (spec D4): the host model's own prose around the tool output is NOT validated, so this
surface is for the expert team only — never for the water authority, who use the web shell.
Hand-rolled JSON-RPC 2.0 over stdio (newline-delimited), no extra dependency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from . import presets as PR
from .page import BindingError, PageBundle
from .session import Session

PROTOCOL = "2025-06-18"
RELAY = ("Relay this answer to the user verbatim. Do not add numbers, conclusions, verdicts or "
         "recommendations, and do not paraphrase capability statements. Claim IDs and the page "
         "link must stay.\n\n")
INSTRUCTIONS = ("Answers about the HydroPhysicsAI twin decision page. Every number and sentence is the "
                "page's own; relay tool output verbatim. Withdrawn quantities (subsidence size, area) "
                "and Changhua levers are refused by design: do not estimate them yourself.")
TOOL_DOCS = {
    "policy_head": "Head change at the registered Yunlin wells under a policy (cuts per class, start, year).",
    "start_compare": "The same cuts started in each start year, compared.",
    "lever_rank": "Ranking of water-use classes by head rise or per energy given up (model internals).",
    "township_internals": "One township's labelled model-internal subsidence row under a policy.",
    "rail_internals": "The head answer plus the rail's model-internal note (no verdict).",
    "solved_compare": "The page's full-model runs side by side.",
    "how_sure": "How far the result is tested and when the registered tests are scored.",
}


class MCPServer:
    def __init__(self, b: PageBundle, *, use_llm: bool = True, llm=None,
                 audit_dir: Path | None = Path("results/analyst/audit")):
        self.b, self.use_llm, self.llm, self.audit_dir = b, use_llm, llm, audit_dir
        self.session = Session()

    def tools(self) -> list[dict]:
        sch = PR.schemas(self.b) | {"how_sure": {"type": "object", "additionalProperties": False,
                                                 "properties": {}, "required": []}}
        out = [{"name": "ask_twin", "description": "Ask the twin decision page a question in plain "
                "language (English or Traditional Chinese).",
                "inputSchema": {"type": "object", "additionalProperties": False, "required": ["question"],
                                "properties": {"question": {"type": "string"},
                                               "lang": {"type": "string", "enum": ["en", "zh"]}}}}]
        for name, doc in TOOL_DOCS.items():
            s = dict(sch[name])
            s["properties"] = dict(s["properties"]) | {"lang": {"type": "string", "enum": ["en", "zh"]}}
            s["required"] = [r for r in s.get("required", []) if r not in ("start", "year", "by")]
            out.append({"name": name, "description": doc, "inputSchema": s})
        return out

    def call(self, name: str, args: dict) -> dict:
        from . import audit
        from .ask import ask
        from .validate import validate
        args = dict(args or {})
        lang = args.pop("lang", None)
        if name == "ask_twin":
            ans, v, _ = ask(self.b, str(args.get("question", ""))[:600], lang=lang, use_llm=self.use_llm,
                            llm=self.llm, audit_dir=self.audit_dir, user="mcp", session=self.session)
        elif name in TOOL_DOCS:
            ans = PR.run(self.b, name, args, lang or "en")
            v = validate(ans, self.b)
            if v:
                import dataclasses as dc
                ans = dc.replace(PR.refuse(self.b, {"reason": "no_run"}, lang or "en"), fallback="validator")
            elif self.audit_dir is not None:
                audit.write({"user": "mcp", "lang": ans.lang, "question": None, "route": {"preset": name},
                             "stage": "mcp_tool", "answer_preset": ans.preset if not ans.clarify else "clarify",
                             "answer_params": ans.params, "answer_text": ans.text(),
                             "citations": ans.citations, "page_sha256": self.b.page_sha256,
                             "ledger_sha256": self.b.ledger_sha256, "strings_sha256": self.b.strings_sha256},
                            self.audit_dir)
        else:
            return {"content": [{"type": "text", "text": f"unknown tool {name}"}], "isError": True}
        return {"content": [{"type": "text", "text": RELAY + ans.text()}], "isError": bool(v)}

    def handle(self, msg: dict) -> dict | None:
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:                 # a notification: no response
            return None
        try:
            if method == "initialize":
                res = {"protocolVersion": PROTOCOL, "capabilities": {"tools": {"listChanged": False}},
                       "serverInfo": {"name": "twin-analyst", "version": __version__},
                       "instructions": INSTRUCTIONS}
            elif method == "ping":
                res = {}
            elif method == "tools/list":
                res = {"tools": self.tools()}
            elif method == "tools/call":
                p = msg.get("params") or {}
                res = self.call(p.get("name", ""), p.get("arguments") or {})
            else:
                return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
        except Exception as e:  # noqa: BLE001
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": f"{type(e).__name__}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": res}

    def serve(self, rin=sys.stdin, rout=sys.stdout) -> None:
        for line in rin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                rout.write(json.dumps({"jsonrpc": "2.0", "id": None,
                                       "error": {"code": -32700, "message": "parse error"}}) + "\n")
                rout.flush()
                continue
            out = self.handle(msg)
            if out is not None:
                rout.write(json.dumps(out, ensure_ascii=False) + "\n")
                rout.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--ledger", type=Path)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--allow-pending", action="store_true")
    ap.add_argument("--unbound", action="store_true")
    a = ap.parse_args(argv)
    try:
        b = PageBundle.load(a.page, a.ledger, allow_pending=a.allow_pending,
                            check_forward=not a.unbound, check_template=not a.unbound)
    except BindingError as e:
        print(f"binding failed: {e}", file=sys.stderr)
        return 3
    llm = None
    if not a.no_llm:
        from .ask import make_llm
        llm = make_llm(b)
    MCPServer(b, use_llm=not a.no_llm, llm=llm).serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
