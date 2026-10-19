"""L3 release ``release_20261012b2``: ``release_20261012b`` with two user rulings (2026-10-18).

A copy of ``l3_release_b`` (left unedited, because it is committed and the component
manifests of release_20261012b record its sha256). Release b's verifier left two majors and
three user decisions open (results/dataprep/reports/REPAIR_ROUND_2026-10-18.json, key
"release"). The user ruled on two of them on 2026-10-18:

(a) W1 pre-origin (``surface_water_v4/preorigin/v2``) stays included. This is a ruling, not a
    gate pass; it is recorded in RELEASE.json, the W1 component's notes and manifest, and the
    product card.
(b) The pre-origin printed piles (S2, ``deep_benchmarks_printed/v2``) are narrowed to the uses
    validator and after_the_fact_analysis (the plan's S2 role "never in a gate or choice");
    design_time_evidence and scoring_target are dropped, and the component's guard read uses
    validator instead of design_time_evidence (same rows: both uses keep flags 0-6).

The third decision (which publication date counts for the 2022 pile reports) was not ruled on;
b's rule (the agency listing's 發布年月) is kept and listed as an open user decision. Nothing
else changes: same upstream versions, rule R, thresholds, gates and products.

Modules:
  config        release name, rule R, user rulings, component specifications, exclusions, limits
  io_det        deterministic writers (npz, parquet, json) and a pickle-free npz reader
  components    component builders (byte copy through the guard, repack, guarded table read, grid)
  build         CLI: build the release into an output root
  verify        CLI: compare two builds file by file, verify a release, write REBUILD_CHECK.json
  inventory     classify every product manifest under results/dataprep (included or excluded)
  ledger_rows   CLI: the release's ledger fragment (rows/L3.csv) and data_roles.csv assembly
  regress       CLI: compare release_20261012b2 with release_20261012b file by file
  repair_ledger CLI: ledger rows for the numbers quoted about this release

Import with ``sys.path`` holding ``<repo>/dataprep_staging`` (``from l3_release_b2 import ...``).
Nothing under ``hydrophysics/`` imports this package; it imports ``hydrophysics.twin.grid``
read-only for the grid convention.
"""
from __future__ import annotations

__version__ = "1.2.0"
