# WP S1b: independent checks of the ring bridge (staging code)

Phase-0 staging code (user decision D12) for plan WP S1, part b: checks (c) GNSS and (d)
InSAR of `results/dataprep/rings_bridged/v2` (S1a). Validators only: nothing here writes to
S1a's product or changes an S1a parameter. Code only; no data in this directory.

| File | Role |
|---|---|
| `s1b_const.py` | constants fixed in the pre-registration `S1b.ring_bridge_checks.v1` |
| `s1b_core.py` | pure functions: window fits, excess over the gap, bracket, family rule, correlation |
| `s1b_io.py` | guarded reads (`after_the_fact_analysis`) of rings_bridged/v2, gnss_clean/full/v2, insar_069A/v2 |
| `s1b_looks.py` | look-log records written before the registration (design, availability runs) |
| `s1b_synth.py` | availability-preserving synthetic operating characteristics (design-time) |
| `s1b_prereg.py` | writes the pre-registration (harness.prereg3) |
| `s1b_run.py` | `registered` (the registered checks) and `after` (registered report-only variants) |
| `s1b_diag.py` | after-the-fact diagnostics (SJES offset, InSAR vs GNSS rates, scales) |
| `s1b_report.py` | ledger rows, provenance record, `reports/S1b.json` |
| `tests/test_s1b_core.py` | CPU unit tests on synthetic series |

Order of use (each heavy step through `dataprep_staging/tools/cpu_slot.sh`, hydro env,
`python -I`, `CUDA_VISIBLE_DEVICES=""`):

1. `s1b_looks.py design`, `s1b_looks.py availability2`, `s1b_synth.py`
2. `s1b_prereg.py --dry-run`, then `s1b_prereg.py`
3. `s1b_run.py registered`, then `s1b_run.py after`, then `s1b_diag.py`
4. `s1b_report.py`

The registered inputs (the code files listed in the pre-registration among them) must stay
unchanged for `s1b_run.py registered` to run again (`prereg3.assert_can_score`). Port to
`hydrophysics/dataprep/` after the freeze with the S1 port (plan 5.5).

## Repair round (2026-10-14)

After-the-fact scripts answering the independent verification (the registered files above stay
unchanged; each script logs its look before reading anything):

| File | Role |
|---|---|
| `s1b_r2_core.py` | pure helpers: plan-literal bracket readings, minimum k, Williams' t, bootstrap, bridge increments, longest data gap |
| `s1b_r2_looks.py` | look records (`verifier` disclosure; the scripts below call `readings`, `synth`, `u1`) |
| `s1b_r2_readings.py` | plan-literal and k-sigma readings, per-window RMSE, (d1) dependent-correlation test, from the registered rows only |
| `s1b_r2_synth.py` | synthetic operating characteristics of those readings (round-1 generator, seed 1803) |
| `s1b_r2_u1.py` | GNSS flag composition, single-epoch bridge split (superseded), ring-residual-sd variant, GNSS data gaps, SJES endpoints |
| `s1b_r2_bridge.py` | InSAR model-set windows (36 d epoch sets) against GNSS and a within-segment null; U1-strict readings; epoch noise |
| `s1b_r2_report.py` | ledger rows, `S1b_evidence/repair/PROVENANCE.json`, `reports/S1b.json` |
| `tests/test_s1b_r2.py` | unit tests of `s1b_r2_core.py` |

Order: `s1b_r2_looks.py verifier`, then `s1b_r2_readings.py`, `s1b_r2_synth.py`,
`s1b_r2_u1.py`, `s1b_r2_bridge.py` (heavy steps through `tools/cpu_slot.sh`), then
`s1b_r2_report.py`.

## Repair round r3 (S1br, 2026-10-18)

Second look of checks (c) and (d) on the REPAIRED products (rings_bridged/v3, gnss_clean/full/v4,
insar_069A/v3) under the plan-literal readings, with S1b's 2-sigma rule labelled beside them,
and fresh-seed synthetic operating characteristics of every reading. The registered round-1
and r2 files above stay unchanged; the registered computation functions of `s1b_run` run
unchanged through `s1b_r3_io.Adapter`. Report: `results/dataprep/reports/S1b.repair.md`.

| File | Role |
|---|---|
| `s1b_r3_looks.py` | look records of this round (disclosures, diagnose, design, synthetic, second look, after) |
| `s1b_r3_diagnose.py` | diagnosis numbers from S1b's round-1 and r2 evidence (no product read) |
| `s1b_r3_snapshot.py` | hashed snapshots of the report parts |
| `s1b_r3_io.py` | guarded reads of the repaired products; the InSAR pixel rule restated for v3; `Adapter` |
| `s1b_r3_synth.py` | synthetic operating characteristics, ring noise N1 (sd_s) and N2 (residual sd), fresh seeds |
| `s1b_r3_prereg.py` | the declaration `S1b.r3.rings_v3_second_look.v1` (harness.prereg3; hashes code and inputs) |
| `s1b_r3_run.py` | the declared second look, variants and decompositions |
| `s1b_r3_after.py` | real counts against the synthetic count distributions (declared) |
| `s1b_r3_diag_after.py`, `s1b_r3_diag_after2.py` | logged after-the-fact diagnostics (not declared) |
| `s1b_r3_report.py` | numbers, quote check, provenance, ledger rows, `reports/S1b.repair.json` |
| `tests/test_s1b_r3.py` | 8 CPU unit tests (synthetic values; a fake-input run of the whole pipeline) |

Order: `s1b_r3_looks.py disclose_session|disclose_r2_verifier|diagnose`, `s1b_r3_diagnose.py`,
`s1b_r3_snapshot.py part1_diagnosis`, `s1b_r3_looks.py design`, `s1b_r3_synth.py smoke`,
`s1b_r3_snapshot.py part2_design`, `s1b_r3_prereg.py`, `s1b_r3_synth.py N1_sd_s` and `N2_resid`,
`s1b_r3_run.py`, `s1b_r3_after.py`, the two diagnostics, `s1b_r3_report.py`. Heavy steps through
`tools/cpu_slot.sh`. The run scripts refuse to run if any declared input or code file changed.

## Repair round r3, fix round (after the r3 verification)

The r3 verifier found the r3 "both GNSS artefacts removed: 8/9" statement one-sided. The fix
round declares (`S1b.r3f.step_policy_symmetry.v1`) and computes, after the fact on spent
units, a symmetric grid: GNSS v2 / v4 crossed with the step policies product /
catalogue_in_gap / catalogue_all (SJES excluded), through the registered round-1 pipeline.

    python -I dataprep_staging/s1_rings/s1b/s1b_r3f_looks.py disclose_r3_verifier|fix_diagnose
    python -I dataprep_staging/s1_rings/s1b/s1b_r3f_prereg.py
    tools/cpu_slot.sh python -I dataprep_staging/s1_rings/s1b/s1b_r3f_symmetry.py
    python -I dataprep_staging/s1_rings/s1b/s1b_r3f_report.py

Tests: `tests/test_s1b_r3f.py` (catalogue_all add-back; equality with the registered
in-gap rule inside the gap). Report: `results/dataprep/reports/S1b.repair.md`, Fix round.
