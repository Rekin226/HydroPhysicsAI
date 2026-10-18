"""WP P4 (Changhua abstraction) staging code. P4.0: the linear-response engine (engine.py).

Import with ``sys.path.insert(0, "<repo>/dataprep_staging")`` and ``from p4_changhua import
engine``. Nothing under ``hydrophysics/`` imports this package; it imports repository modules
read-only. Phase 1 ports it to ``hydrophysics/dataprep/`` with tests (plan 2.7).
"""
from __future__ import annotations

__version__ = "0.1.0"   # WP P4.0, 2026-10-12
