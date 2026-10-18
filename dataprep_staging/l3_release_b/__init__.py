"""L3 release ``release_20261012b``: repair of ``release_20261012`` (plan 2.1 L3, 2.3, 2.8).

A copy of ``l3_release`` (left unedited, because the component manifests of release_20261012
record its sha256) with the repair of the L3 verifier's three majors (results/dataprep/reports/
L3.repair.md): one inclusion rule R for every product, a publication filter for the pre-origin
piles and narrowed pile uses, the post-origin stations dropped from the pre-origin station table,
publication checks for pre-origin tables, and the repair round's products classified.

Modules:
  config      release name, rule R, component specifications, exclusions, known limits
  io_det      deterministic writers (npz, parquet, json) and a pickle-free npz reader
  components  component builders (byte copy through the guard, repack, guarded table read, grid)
  build       CLI: build the release into an output root
  verify      CLI: compare two builds file by file, verify a release, write REBUILD_CHECK.json
  inventory   classify every product manifest under results/dataprep (included or excluded)
  ledger_rows CLI: the release's ledger fragment (rows/L3.csv) and data_roles.csv assembly
  diag        read-only diagnosis evidence (L3.repair.md Part 1)

Import with ``sys.path`` holding ``<repo>/dataprep_staging`` (``from l3_release_b import ...``).
Nothing under ``hydrophysics/`` imports this package; it imports ``hydrophysics.twin.grid``
read-only for the grid convention.
"""
from __future__ import annotations

__version__ = "1.1.0"
