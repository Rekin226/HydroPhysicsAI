"""First L3 model-ready release of the October 2026 data-remediation plan (plan 2.1 L3, 2.3, 2.8).

Modules:
  config      release name, component specifications, exclusions with reasons, known limits
  io_det      deterministic writers (npz, parquet, json) and a pickle-free npz reader
  components  component builders (byte copy through the guard, repack, guarded table read, grid)
  build       CLI: build the release into an output root
  verify      CLI: compare two builds file by file, verify a release, write REBUILD_CHECK.json
  inventory   classify every product manifest under results/dataprep (included or excluded)
  ledger_rows CLI: the release's ledger fragment (rows/L3.csv) and data_roles.csv assembly
  report      CLI: the WP report (results/dataprep/reports/L3.md and L3.json)

Import with ``sys.path`` holding ``<repo>/dataprep_staging`` (``from l3_release import ...``).
Nothing under ``hydrophysics/`` imports this package; it imports ``hydrophysics.twin.grid``
read-only for the grid convention.
"""
from __future__ import annotations

__version__ = "1.0.0"
