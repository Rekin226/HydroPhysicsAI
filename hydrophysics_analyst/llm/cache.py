"""Per-call JSON cache keyed by the canonical request (the repo's report_extract pattern).

``results/analyst/calls/<sha256>.json`` holds the request, the normalized response, usage,
model and the page/ledger shas. ``no_api=True`` replays from the cache only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .backend import NormalizedResponse


class CacheMiss(LookupError):
    pass


class CachedBackend:
    def __init__(self, inner, directory: Path = Path("results/analyst/calls"), *, no_api: bool = False,
                 tag: dict | None = None):
        self.inner, self.dir, self.no_api, self.tag = inner, Path(directory), no_api, tag or {}
        self.name = f"cached:{getattr(inner, 'name', '?')}"
        self.last_hit = False

    def key(self, system, messages, output_format, effort, max_tokens) -> str:
        blob = json.dumps({"backend": getattr(self.inner, "name", "?"), "system": system,
                           "messages": messages, "output_format": output_format, "effort": effort,
                           "max_tokens": max_tokens}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(self, system, messages, *, output_format=None, effort="low", max_tokens=1024):
        k = self.key(system, messages, output_format, effort, max_tokens)
        p = self.dir / f"{k}.json"
        if p.exists():
            self.last_hit = True
            d = json.loads(p.read_text(encoding="utf-8"))
            return NormalizedResponse(**d["response"])
        self.last_hit = False
        if self.no_api:
            raise CacheMiss(f"no cached response for request {k[:12]} (--no-api)")
        r = self.inner.complete(system, messages, output_format=output_format, effort=effort,
                                max_tokens=max_tokens)
        self.dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"key": k, "messages": messages, "output_format": output_format,
                                 "effort": effort, "response": asdict(r), "tag": self.tag,
                                 "ts": dt.datetime.now(dt.timezone.utc).isoformat()},
                                ensure_ascii=False, indent=1), encoding="utf-8")
        return r
