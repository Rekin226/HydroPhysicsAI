"""WP S1a (plan section 3, WP S1): ring instrument-change correction and 2022 bridge.

Phase-0 staging code (user decision D12). Builds the data product
``results/dataprep/rings_bridged/v<N>/``; nothing under ``hydrophysics/`` imports it, and it
imports repository modules read-only only. See ``README.md`` in this directory.
"""
from __future__ import annotations
