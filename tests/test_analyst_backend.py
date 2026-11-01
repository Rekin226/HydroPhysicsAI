"""Backend request shape (no network), the per-call cache and --no-api replay."""
from __future__ import annotations

import pytest

from hydrophysics_analyst.llm.backend import AnthropicBackend, FakeBackend, cost_usd
from hydrophysics_analyst.llm.cache import CachedBackend, CacheMiss

SYS = [{"type": "text", "text": "route"}]
MSG = [{"role": "user", "content": "q"}]


def test_request_shape_opus_and_haiku():
    r = AnthropicBackend("claude-opus-5-5").request(SYS, MSG, output_format={"type": "json_schema", "schema": {}},
                                                    effort="low", max_tokens=1024)
    assert r["model"] == "claude-opus-5-5" and r["output_config"]["effort"] == "low"
    assert r["output_config"]["format"]["type"] == "json_schema"
    assert "thinking" not in r and "tool_choice" not in r and "temperature" not in r
    assert r["fallbacks"] == "default" and r["betas"] == ["server-side-fallback-2026-07-01"]
    h = AnthropicBackend("claude-haiku-5-5").request(SYS, MSG, output_format=None, effort="low", max_tokens=64)
    assert "fallbacks" not in h and "betas" not in h
    z = AnthropicBackend("claude-opus-5-5", fallbacks=False, inference_geo="us").request(
        SYS, MSG, output_format=None, effort="low", max_tokens=64)
    assert "fallbacks" not in z and z["inference_geo"] == "us"


def test_sdk_adapter_with_a_mocked_client():
    pytest.importorskip("anthropic")

    class Blk:
        type, text = "text", '{"preset": null}'

    class Resp:
        stop_reason, content, model, _request_id = "end_turn", [Blk()], "claude-opus-5-5", "req_1"

        class usage:
            input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens = 10, 5, 0, 0

    seen = {}

    class Msgs:
        def create(self, **kw):
            seen.update(kw)
            return Resp()

    class Client:
        messages = Msgs()

        class beta:
            messages = Msgs()

    be = AnthropicBackend("claude-opus-5-5", client_factory=Client)
    r = be.complete(SYS, MSG, output_format={"type": "json_schema", "schema": {}})
    assert r.text == '{"preset": null}' and r.request_id == "req_1" and seen["fallbacks"] == "default"
    Resp.stop_reason = "refusal"
    assert be.complete(SYS, MSG).text == ""


def test_cache_hit_and_no_api(tmp_path):
    fb = FakeBackend([{"preset": "policy_head"}])
    cb = CachedBackend(fb, tmp_path)
    a = cb.complete(SYS, MSG)
    b = cb.complete(SYS, MSG)
    assert a.text == b.text and len(fb.calls) == 1 and cb.last_hit
    with pytest.raises(CacheMiss):
        CachedBackend(FakeBackend([]), tmp_path / "empty", no_api=True).complete(SYS, MSG)


def test_cost():
    assert cost_usd("claude-opus-5-5", {"input_tokens": 1_000_000, "output_tokens": 0}) == pytest.approx(4.0)
    assert cost_usd("claude-haiku-5-5", {"input_tokens": 0, "output_tokens": 1_000_000}) == pytest.approx(0.5)
