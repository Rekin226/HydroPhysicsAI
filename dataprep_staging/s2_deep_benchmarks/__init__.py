"""WP S2: deep benchmarks (深層樁) as a first-class series (plan section 3, WP S2).

Modules:
  s2lib               pure functions: feed loading, vendor/raw identity, pre-origin change
                      envelope, step detection and correction, daily and monthly aggregation,
                      printed-value merge, cumulative reconstruction, acceptance statistics
  archive_opendata    copies the 2026-10-11 open-data deep-pile dumps (data.gov.tw 58339) into
                      L0 with a SOURCE.json
  probe_reports       checks the WRA monthly-report address for reports newer than 2026-01
  portal_check        reads the portal listing of series 007003 and re-probes the address
  prereg              round-1 acceptance pre-registration (written before the A1 tool existed)
  prereg_r2           round-2 core declaration with the A1 tool (S2r2.deep_benchmarks_core)
  fetch_007001        fetches the Changhua-Yunlin monthly reports (series 007001) into L0
  cy007001            parses Table 1 of the 007001 reports (deterministic read)
  transcribe_007001   builds results/dataprep/deep_benchmarks_printed/v<N>/ (two reads per row;
                      Claude API used only to copy printed numbers)
  prereg_c            registers the fresh test C (S2r2.c_fresh_cy007001)
  c_test              monthly-mean increments and the scorer of test C
  build               builds results/dataprep/deep_benchmarks/v<N>/ with MANIFEST.json
  ledger_rows         appends the S2 ledger fragment
  r3_diagnostics      round 3: agency probe-adjustment records against the step catalogue,
                      failure patterns of tests A and C1 (after-the-fact, no rule change)
  r3_repro_check      round 3: scratch rebuild against v1; pandas non-ns index probe
  report              writes results/dataprep/reports/S2.json (S2.md is written by hand)
  printed_v2          round 4: deep_benchmarks_printed/v<N> (one primary printing per
                      sensor-month; integrated-report values under a manifest; sd labelled)
  label_confirm       round 4: labels of the vendor series from pre-origin data only
  prereg_r4           round 4: declaration S2r4.deep_benchmarks_v2 (A1 tool)
  build_v2            round 4: deep_benchmarks/v<N> with pre-origin rows from a pre-origin run
  r4_checks           round 4: guard access, checker, reproduction and look-log disclosures
  ledger_rows_r4      round 4: rewrites the S2 ledger fragment (round-3 file superseded)
  report_r4           round 4: writes results/dataprep/reports/S2.json

Nothing under hydrophysics/ imports this package. Data never lives here.
"""
from __future__ import annotations
