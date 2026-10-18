"""Shared staging library for the October 2026 data-remediation plan (WP A0).

Modules:
  paths     fixed layout (results/dataprep, ledger, L0 new_measurements, origin 2022-12)
  hashing   sha256 of files, trees, arrays, grid masks
  flags     flag table and long-table/grid validators (plan 2.4)
  uses      use vocabulary, pre-origin uses, flags allowed per use, ledger roles (2.5, 4.1)
  manifest  MANIFEST.json schema, writer, validator, hash verification, new_version_dir (2.3)
  guard     loader guard: open_product(dir, use[, window_end]) enforcing U1-U3, U5, U7
            (overrides only tighten; time slices with declared encodings iso/yyyymm/year;
            per-output windows)
  ledger    per-WP ledger fragments and data_roles.csv assembly (4.1, U6)
  source    L0 SOURCE.json writer/validator and read-only copy into L0 (2.1)
  check     CLI re-hashing every manifest and SOURCE.json; reports L0 orphans

Import with ``sys.path.insert(0, "<repo>/dataprep_staging")`` and then
``from common import manifest``. Nothing under ``hydrophysics/`` imports this package.
"""
from __future__ import annotations

__version__ = "0.2.1"   # A0fix: type-safe time slicing (declared encodings); manifest schema 2
