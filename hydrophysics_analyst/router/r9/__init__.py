"""FROZEN round-9 decision code (ruling R10.3b): the deterministic ``hard_refusal`` and its guards
exactly as at commit fc7b996 (main before the round-10 narrowing), vendored so the round-9
guards can be replayed at $0 on a held-out set's recorded rows (``hydrophysics_analyst.replay_r9``).

Never edit these files and never route live questions through them: the live router is
``hydrophysics_analyst.router``. Only the relative imports differ from fc7b996 (one more dot);
tests/test_analyst_r9_replay.py checks the copy still reproduces fc7b996's decisions.
"""
from __future__ import annotations

from .guards import changhua_kind
from .rules import hard_refusal

FROZEN_AT = "fc7b996"

__all__ = ["FROZEN_AT", "changhua_kind", "hard_refusal"]
