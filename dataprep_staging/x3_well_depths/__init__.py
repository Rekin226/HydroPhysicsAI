"""X3: public well-depth evidence for Changhua and Yunlin, and the pumping split by layer.

Searches public sources for pump or well depths, builds depth distributions mapped to the
twin's aquifers, learns a depth class from census attributes where a legitimate key exists,
and estimates independently, from design-time heads only, which aquifer the local pumping
stress hits. The result is a prior for the GPU rung that gives Changhua its own split.
"""
from __future__ import annotations

__version__ = "1.0.0"
