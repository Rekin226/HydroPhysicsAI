"""Ask the twin decision page in plain language.

Every answer is a page state rendered with the page's own strings: numbers come from a numpy
port of the page's runtime math on the built page's payload, sentences from the page's own
bilingual dictionary, claim IDs from the claims ledger. The LLM, when used, only routes a
question to a preset with parameters (enum-valued); it never writes user-visible text.

Lives outside ``hydrophysics/`` on purpose: ``release.provenance`` hashes that whole package.
Spec: docs/superpowers/specs/2026-11-01-twin-analyst-agent-design.md
"""
from __future__ import annotations

__version__ = "0.1.0"
