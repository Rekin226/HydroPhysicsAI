# s1_rings: WP S1a (ring instrument-change correction and 2022 bridge)

Plan: `docs/DATA_REMEDIATION_PLAN_2026-10.md`, section 3, WP S1 (part a) and 5.5.
Product: `results/dataprep/rings_bridged/v<N>/` (gitignored). Code only here.

Rules: `results/dataprep/preregistration/S1a.rings_bridged_rules.v1.json` (declaration,
written before the rebuilt evidence was computed). Synthetic test (e):
`S1a.e_synthetic_mixed.v1` (registered before it ran).

| Module | Role |
|---|---|
| `s1const.py` | constants fixed by the declaration (windows, evidence thresholds, piles, tests) |
| `s1lib.py` | model `z_post = a + (1 + b) z_true`; window fits, `b_deep`, pile-implied scale, likelihood ratios and votes, assignment, LOO pooling, members, mixture, monthly series, 2022 fills |
| `s1io.py` | guarded reads: deep_benchmarks/v2 (after_the_fact_analysis, or design_time_evidence through the 2022-12 slice for pre-origin tests), deep_benchmarks_printed/v2 (validator), gnss_clean/full/v1 (after_the_fact_analysis) |
| `seed_looks.py` | logs the pilot's earlier looks and this WP's design-time look (run once) |
| `prereg_s1a.py` | writes the two pre-registrations |
| `path_test.py` | family Y (masked years 2017-2020, pre-origin): the path terms of the 2022 sd |
| `synthetic.py` | check (e): mixed A/B pseudo-gaps in every 13-month window 2015-2021, full pipeline |
| `build.py` | the product (`--assign-only` writes the assignment evidence used for p_A) |
| `gnss_sensitivity.py` | after the fact: GNSS-implied scales as a fourth line (not used by the product) |
| `checks.py` | manifest re-hash, guard probe per use, flag/sd consistency |
| `report.py` | WP report and ledger rows |
| `tests/` | CPU unit tests on synthetic ring sites (`test_s1lib.py`) and of the repair modules (`test_r2.py`) |

Repair round (r2, after the verifier's review of v1; declaration
`S1a.rings_bridged_rules.v2`, written before v2 was built). These modules are new; none of the
files hashed by rings_bridged/v1 or by the v1 pre-registrations was edited.

| Module | Role |
|---|---|
| `r2_diagnose.py` | after the fact: (e) rows split into scale and membership components, counterfactual primaries, window coverage; (a) power decomposition |
| `r2_bdeep.py` | after the fact: b_deep error per group and window in the (e) rows |
| `prereg_r2.py` | declaration S1a.rings_bridged_rules.v2 (feed-check day screen, allowed-uses rule, provenance, card) |
| `build_r2.py` | rings_bridged/v2 (reuses `build.py` and `s1lib.py` unchanged) |
| `r2_checks.py` | v2 against v1 (only the declared change) and the registered code unchanged |
| `report_r2.py` | ledger fragment (round 1 kept in `ledger/superseded/S1a.r1.csv`; two roles corrected) |
| `r2_reverify.py` | resumed session: re-hash of v1/v2, pre-registration sha256, guard probe, ledger and look-log checks, report numbers against the evidence; `--ledger-only` appends its ledger rows |

Order: `seed_looks` -> `prereg_s1a rules` -> `path_test` -> `build --assign-only` ->
`prereg_s1a synthetic` -> `synthetic` -> `gnss_sensitivity` -> `build` -> `checks` -> `report`;
repair round: `r2_diagnose` -> `prereg_r2` -> `build_r2` -> `checks <v2>` -> `r2_checks` ->
`r2_bdeep` -> `report_r2` -> `r2_reverify` -> `r2_reverify --ledger-only`.
Run long steps through `tools/cpu_slot.sh`. No GPU; nothing here writes under `hydrophysics/`,
`tests/` or L0.

Repair round r3 (S1ar, 2026-10-18; report `results/dataprep/reports/S1a.repair.md`; declaration
`S1a.rings_bridged_rules.v3`, fresh test `S1a.e_synthetic_fresh.v3`). New modules only; no file
hashed by rings_bridged/v1, v2 or their pre-registrations was edited.

| Module | Role |
|---|---|
| `r3_looks.py` | look records of the round (availability, diagnose, second looks) |
| `r3_availability.py` | pre-2012 window entry counts from survey dates and read masks only |
| `r3_diagnose.py` | re-execution of the registered (e) pipeline with internals; b_deep of the uninjected surveys; deep-ring timing |
| `r3_diag_summary.py` | every diagnosis number (`repair_r3/diag/summary.json`) |
| `r3_calibrate.py` | sigma_x, sigma_c of the length-aware b_deep error model |
| `r3_lib.py` | rules v3: length-aware b_deep sd, no pile line, EM prior, posterior-threshold pools, common-mode sd, corrected (a) |
| `r3_prereg.py` | the two r3 pre-registrations |
| `r3_synthetic.py` | the fresh re-test (pre-2012 gating, late-2019 reported, registered-method control) and the second look on the spent windows |
| `r3_post.py` | after-the-fact analysis of the scored re-test |
| `r3_build.py` | rings_bridged/v3 |
| `r3_report.py` | ledger rows, `S1a.repair.json`, check of the quoted numbers |
| `tests/test_r3.py` | CPU unit tests of r3_lib and the build helpers on toy sites |

Order: `r3_looks availability|diagnose` -> `r3_diagnose timing|truth|rerun` -> `r3_diag_summary`
-> (Part 1 written and hashed) -> `r3_calibrate` -> `r3_prereg rules` -> `r3_prereg test` ->
`r3_synthetic fresh` -> `r3_synthetic spent` -> `r3_build` -> `checks <v3>` -> `r3_post` ->
`r3_report` -> `r3_report --json-only`.
