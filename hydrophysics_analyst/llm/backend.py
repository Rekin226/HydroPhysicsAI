"""LLM backends behind one protocol.

- :class:`AnthropicBackend`: first-party Claude API (or a cloud provider client by
  ``client_factory``). Checks ``stop_reason == "refusal"`` before reading content. Server-side
  ``fallbacks: "default"`` (beta ``server-side-fallback-2026-07-01``) only on the first-party API
  with Opus 5.5 and only when enabled — never with Haiku 5.5, never on Bedrock/Vertex/Foundry,
  and off in a ZDR deployment where the model list must be approved.
- :class:`FakeBackend`: scripted responses for tests (no network).
- :class:`NullBackend`: no LLM at all (rules parser + form only).

The API key comes from the environment, or from the repo-root ``.env`` via :func:`load_key`
(the hand-rolled pattern of the dataprep scripts); it is never printed or logged.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class NormalizedResponse:
    text: str
    stop_reason: str
    usage: dict = field(default_factory=dict)
    model: str = ""
    request_id: str = ""


class LLMBackend(Protocol):
    name: str

    def complete(self, system: list[dict], messages: list[dict], *, output_format: dict | None,
                 effort: str, max_tokens: int) -> NormalizedResponse: ...


class BackendUnavailable(RuntimeError):
    pass


def load_key() -> None:
    """Put ``ANTHROPIC_API_KEY`` from the repo-root ``.env`` into the environment if unset."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            return


class AnthropicBackend:
    def __init__(self, model: str = DEFAULT_MODEL, client_factory=None, *, fallbacks: bool = True,
                 first_party: bool = True, inference_geo: str | None = None, timeout: float = 60.0):
        self.model, self.name = model, f"anthropic:{model}"
        self.client_factory = client_factory
        self.fallbacks = fallbacks and first_party and model.startswith("claude-opus-5-5")
        self.inference_geo = inference_geo
        self.timeout = timeout
        self._client = None

    def client(self):
        if self._client is None:
            if self.client_factory is not None:
                self._client = self.client_factory()
            else:
                try:
                    import anthropic
                except ImportError as e:            # optional extra
                    raise BackendUnavailable("pip install 'hydrophysics-ai[analyst]'") from e
                load_key()
                if not os.environ.get("ANTHROPIC_API_KEY"):
                    raise BackendUnavailable("no ANTHROPIC_API_KEY (environment or .env)")
                self._client = anthropic.Anthropic(timeout=self.timeout, max_retries=2)
        return self._client

    def request(self, system, messages, *, output_format, effort, max_tokens) -> dict:
        req: dict = {"model": self.model, "max_tokens": max_tokens, "system": system,
                     "messages": messages, "output_config": {"effort": effort}}
        if output_format is not None:
            req["output_config"]["format"] = output_format
        if self.inference_geo:
            req["inference_geo"] = self.inference_geo
        if self.fallbacks:
            req["betas"] = [FALLBACK_BETA]
            req["fallbacks"] = "default"
        return req

    def complete(self, system, messages, *, output_format=None, effort="low", max_tokens=1024):
        req = self.request(system, messages, output_format=output_format, effort=effort,
                           max_tokens=max_tokens)
        cl = self.client()
        resp = cl.beta.messages.create(**req) if "betas" in req else cl.messages.create(**req)
        stop = getattr(resp, "stop_reason", "") or ""
        text = "" if stop == "refusal" else "".join(
            getattr(blk, "text", "") for blk in resp.content if getattr(blk, "type", "") == "text")
        u = getattr(resp, "usage", None)
        usage = {k: getattr(u, k, None) for k in ("input_tokens", "output_tokens",
                                                   "cache_read_input_tokens",
                                                   "cache_creation_input_tokens")} if u else {}
        return NormalizedResponse(text=text, stop_reason=stop, usage=usage,
                                  model=getattr(resp, "model", self.model),
                                  request_id=getattr(resp, "_request_id", "") or "")


class FakeBackend:
    """Scripted responses (dicts are JSON-encoded as the text); records every request."""

    def __init__(self, script: list):
        self.script, self.calls, self.name = list(script), [], "fake"

    def complete(self, system, messages, *, output_format=None, effort="low", max_tokens=1024):
        self.calls.append({"system": system, "messages": messages, "output_format": output_format})
        if not self.script:
            raise AssertionError("FakeBackend called with an empty script")
        r = self.script.pop(0)
        if isinstance(r, NormalizedResponse):
            return r
        return NormalizedResponse(text=json.dumps(r) if isinstance(r, dict) else str(r),
                                  stop_reason="end_turn", model="fake")


class NullBackend:
    name = "none"

    def complete(self, *a, **k):
        raise BackendUnavailable("no LLM backend configured (rules parser and form only)")


PRICES = {  # USD per million tokens (input, output, cache read)
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
}


def cost_usd(model: str, usage: dict) -> float:
    p = PRICES.get(model)
    if not p or not usage:
        return 0.0
    i, o, cr = (usage.get("input_tokens") or 0, usage.get("output_tokens") or 0,
                usage.get("cache_read_input_tokens") or 0)
    cw = usage.get("cache_creation_input_tokens") or 0
    return (i * p[0] + cw * p[0] * 1.25 + o * p[1] + cr * p[2]) / 1e6
