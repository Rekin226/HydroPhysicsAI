"""Per-conversation state: the last resolved route (preset + params), not a transcript."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Session:
    prev: dict | None = None

    def update(self, preset: str, params: dict) -> None:
        if preset not in ("refuse", "clarify"):
            self.prev = {"preset": preset, "params": dict(params)}
