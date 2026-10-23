# s5_fusion - WP S5: space-time subsidence fusion

Phase-0 staging code for plan WP S5 (`docs/DATA_REMEDIATION_PLAN_2026-10.md`). Code only: data,
evidence and products live under the gitignored `results/dataprep/`.

## Model

Vertical displacement u(x, t) (mm, + = sinking, 0 at 2012-01-01) on the twin grid:

    u(x, t) = mu_v (t - 2012) + phi(x) . w(t) + phi(x) . (a cos 2 pi t + b sin 2 pi t)

- space: predictive-process basis phi(x) = k(x, Z) K_ZZ^-1 on a 4 km node lattice, Matern 3/2;
- time: integrated random walk on velocity at annual knots (state u_k, v_k per node);
- separable prior precision blockdiag(J_t, I/sig_s^2) (x) K_ZZ^-1;
- instrument operators (`s5fit.py`): leveling (diffuse offset, random velocity, white noise;
  eliminated by restricted likelihood), GNSS (offset, station excess rate, family frame rate,
  family-month common mode), InSAR (4 km block x LiCSBAS network segment offsets, free plane
  per month projected out, exponential APS + white), rings and piles (depth fraction f times u,
  per-segment offsets, random velocity);
- exact Gaussian posterior (dense Cholesky), hyperparameters by restricted marginal
  likelihood on pre-origin data only (U7).

## Files

| file | role |
|---|---|
| `s5const.py` | constants, windows, QC rules, thresholds (hashed by the pre-registration) |
| `s5model.py` | kernels, lattice, IRW time basis, restricted projections, assembler, solver |
| `s5data.py` | observation tables of every instrument (`pre` and `full` variants) |
| `s5insarP.py` | twin-cell values of S4b's pre-origin LiCSBAS run P (for the pre-origin runs) |
| `s5fit.py` | the fusion problem, contributions, fit, predictions, screening, f, optimiser |
| `s5hyper.py` | registered hyperparameter procedure H (stages A-D; T1 stage A) |
| `s5tests.py` | registered tests T1, T2 (gate), T3 and R2 (report only) |
| `s5product.py` | builds `results/dataprep/subsidence_fused/v<N>/` |
| `s5looks.py` | look log helpers (A1 / harness.prereg3) |
| `s5prereg.py` | writes the pre-registration `S5.subsidence_fusion.v1` |
| `s5dryrun.py` | the pre-registration dry run on synthetic values |
| `s5looks2.py`, `s5looks3.py` | look log of the resumed sessions (2026-10-14 evening, 2026-10-17) |
| `s5diag.py` | after-the-fact diagnostics (breakdowns, depth-fraction trade-off) |
| `s5report.py` | collects the report numbers and the ledger number list |
| `s5ledger.py` | ledger rows (plan 4.1) for every quoted number |
| `tests/test_s5model.py` | CPU unit tests (synthetic) |

## Order

```bash
python -I s5looks.py data_build                     # look logged before the loaders ran
tools/cpu_slot.sh python -I s5insarP.py
python -I s5prereg.py                               # registration, before any real fit
tools/cpu_slot.sh python -I s5hyper.py main         # -> S5_evidence/hyper_pre.json
tools/cpu_slot.sh python -I s5hyper.py t1           # -> S5_evidence/hyper_t1.json
tools/cpu_slot.sh python -I s5tests.py t1|t2|t3|r2
python -I s5looks3.py product_build && tools/cpu_slot.sh python -I s5product.py
python -I s5looks3.py diagnostics && tools/cpu_slot.sh python -I s5diag.py
python -I s5report.py && python -I s5ledger.py
```

Run with the hydro env, single-threaded (`tools/cpu_slot.sh`), `CUDA_VISIBLE_DEVICES=""`
(nothing here imports torch). Before any commit: CLAUDE.md Rule 1 scan.

## Repair round (2026-10-17): subsidence_fused/v2

The round-1 modules above are hashed by `S5.subsidence_fusion.v1` and by subsidence_fused/v1
and stay unchanged; the repair modules import them read-only.

| file | role |
|---|---|
| `s5r_const.py` | constants of stage 2 and of the registered test `S5.monthly_field.v1` (frozen) |
| `s5r_vconst.py` | constants of the v2 product rules and the repair diagnostics |
| `s5r_core.py` | v2 operators: GNSS offset segments at daily-record gaps > 30 d, stepped benchmarks |
| `s5r_stage2.py` | stage 2: InSAR monthly anomaly (AR(1) x Matern, block-tridiagonal inference, plane-free support) |
| `s5r_mtest.py` | registered stage-2 procedure (`hyper`) and test M1 (`score`) |
| `s5r_prereg_m1.py` | writes the registration `S5.monthly_field.v1` (before the stage-2 fit) |
| `s5r_prereg_v2.py` | writes the declaration `S5.subsidence_fused_rules.v2` (before v2 is built) |
| `s5r_product.py` | builds `subsidence_fused/v<N>` (v2); `--dry <dir>/v9999` runs it on synthetic values |
| `s5r_dryrun.py` | dry runs of the monthly procedure and test on synthetic values |
| `s5r_diag.py` | after-the-fact diagnostics: stage-A continuation, T2 with a buffer, campaign offsets |
| `s5r_looks.py` | look log of the repair round |
| `s5r_report.py`, `s5r_ledger.py` | report numbers; ledger (round-1 roles corrected, repair numbers) |
| `s5r_write_json.py` | writes `reports/S5.json` from the evidence |
| `s5r_prereg_v3.py` | writes the declaration `S5.subsidence_fused_rules.v3` (label amendment, before v3) |
| `tests/test_s5r_stage2.py`, `tests/test_s5r_core.py` | CPU unit tests (synthetic) |

```bash
python -I s5r_looks.py verifier_disclosure; python -I s5r_looks.py repair_inspect
python -I s5r_looks.py dryrun_monthly && tools/cpu_slot.sh python -I s5r_dryrun.py anomaly|null
python -I s5r_ledger.py rebuild_r1                  # round-1 rows with corrected roles
python -I s5r_prereg_m1.py                          # registration, before the stage-2 fit
python -I s5r_looks.py stage2_hyper && tools/cpu_slot.sh python -I s5r_mtest.py hyper
python -I s5r_prereg_v2.py                          # declaration, before v2 and before M1
tools/cpu_slot.sh python -I s5r_mtest.py score      # M1 (assert_can_score logs the look)
python -I s5r_looks.py product_build_v2 && tools/cpu_slot.sh python -I s5r_product.py
python -I s5r_looks.py repair_diagnostics && tools/cpu_slot.sh python -I s5r_diag.py stageA|t2buffer|campaign
python -I s5r_prereg_v3.py && python -I s5r_looks.py product_build_v3 && tools/cpu_slot.sh python -I s5r_product.py
python -I s5r_looks.py report_facts && python -I s5r_report.py && python -I s5r_ledger.py repair
python -I s5r_write_json.py
```

subsidence_fused/v3 is current (values = v2; corrected monthly sd label). v2's manifest fails the
checker on `s5r_product.py`, edited for v3; the byte-exact v2 build code is archived in
`results/dataprep/reports/S5_evidence/repair/code_at_v2/`. Do not edit any module hashed by v3's
manifest (every `s5*.py` above except s5r_looks, s5r_dryrun, s5r_diag, s5r_report, s5r_ledger,
s5r_write_json, s5diag, s5report, s5ledger, s5looks2/3): build a v4 instead.

## Repair round 2 (2026-10-18): subsidence_fused/v4

Diagnosis, design and results: `results/dataprep/reports/S5.repair.md` (+ `.json`). Registration
`S5.repair_r2.v1` (fresh test T1F; second looks declared). The modules above stay unchanged;
the round-2 modules import them read-only.

| file | role |
|---|---|
| `s5r2_const.py` | repaired-model constants (R1-R6), H2 budgets, T1F windows (registered) |
| `s5r2_model.py` | reverting temporal prior, campaign offsets, correlated GNSS / ring-pile noise, InSAR unit rates, fit, optimiser, predictions (registered) |
| `s5r2_data.py` | loaders with gnss_clean v4, run P2 / insar_069A v3, campaigns, pseudo-origin ends (registered) |
| `s5r2_insarP2.py` | twin-cell values of S4br's run P2 (registered) |
| `s5r2_hyper.py` | procedure H2: chains main, t1, t1f (registered) |
| `s5r2_tests.py` | T1F (fresh, gating), T1 / T2 / T2b second looks, T3 report (registered) |
| `s5r2_prereg.py` | writes `S5.repair_r2.v1` (registered) |
| `s5r2_dryrun.py` | synthetic-value dry runs (registered) |
| `s5r2_diag.py` | diagnosis (before any change) and after-the-fact attribution |
| `s5r2_looks.py` | look log of round 2 |
| `s5r2_product.py` | builds subsidence_fused/v4 |
| `s5r2_report.py`, `s5r2_ledger.py`, `s5r2_write_json.py` | numbers check, ledger rows, S5.repair.json |
| `tests/test_s5r2.py` | CPU unit tests (temporal prior moments, InSAR assembly vs dense, covariances) |

```bash
python -I s5r2_looks.py diagnose_lev; python -I s5r2_looks.py diagnose_inst
tools/cpu_slot.sh python -I s5r2_diag.py t1_decomp|persistence|insar_runP|t2_split|resid_ac|gnss_v2v4
python -I s5r2_looks.py data_build_insarP2 && tools/cpu_slot.sh python -I s5r2_insarP2.py
python -I s5r2_looks.py dryrun_r2 && tools/cpu_slot.sh python -I s5r2_hyper.py t1f --dry <dir> --cap 12
python -I s5r2_prereg.py                                     # registration before any real fit
python -I s5r2_looks.py hyper_t1f_lev; python -I s5r2_looks.py hyper_t1f_inst
tools/cpu_slot.sh --slots 2 python -I s5r2_hyper.py t1f && tools/cpu_slot.sh python -I s5r2_tests.py t1f
python -I s5r2_looks.py hyper_main && tools/cpu_slot.sh --slots 2 python -I s5r2_hyper.py main
tools/cpu_slot.sh --slots 2 python -I s5r2_hyper.py t1
tools/cpu_slot.sh python -I s5r2_tests.py t2|t2b|t3|t1      # second looks log themselves
python -I s5r2_looks.py after_t1sl && tools/cpu_slot.sh python -I s5r2_diag.py t1sl_decomp
python -I s5r2_looks.py product_build_v4 && tools/cpu_slot.sh --slots 2 python -I s5r2_product.py
python -I s5r2_looks.py report_facts && python -I s5r2_report.py subsidence_fused/v4
python -I s5r2_ledger.py && python -I s5r2_write_json.py v4
```

Do not edit any registered module or any module hashed by v4's manifest; build a v5 instead.

## Repair round 2, fix round (2026-10-18)

No method change and no new test. Report: `results/dataprep/reports/S5.repair.md`, Fix round.

- `s5r2f_looks.py`: look log entries (`fix_after`, `fix_report`).
- `s5r2f_diag.py`: the after-the-fact checks declared in F.2: exposure of the 2018-19
  surveys, T2/T2b calibration by distance to training leveling, product regime.
- `s5r2f_product.py`: `subsidence_fused/v5`, which is v4's values (byte copies) with corrected
  labels. T1F reads design-exposed, pass conditional (second look). The T2 over-coverage
  reading is corrected.
- `s5r2f_report.py`: checks the fix-round numbers in the report and writes their ledger rows.
- `tests/test_s5r2f.py`: labelling and statistics unit tests.
