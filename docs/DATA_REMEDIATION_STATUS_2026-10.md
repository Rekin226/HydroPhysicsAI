# Data remediation, Phase 0 status report

Written 2026-10-17 (state at about 14:30 local time). This report closes Phase 0 of
[DATA_REMEDIATION_PLAN_2026-10.md](DATA_REMEDIATION_PLAN_2026-10.md) (revised 2026-10-12 after the
Changhua pump census was obtained), which acts on [DATA_AUDIT_2026-10-11.md](DATA_AUDIT_2026-10-11.md).

- **Sources.** Each WP's report and verifier verdict (`results/dataprep/reports/<WP>.md` and
  `.json`), the L3 release and its verification, the Phase-1 run records under
  `results/twin_runs/par9`, `par10` and `par11`, and one whole-tree manifest check run for this
  report (section 7).
- **Machine-readable version:** `results/dataprep/reports/PHASE0_STATUS.json`. A byte copy of
  this text is `results/dataprep/reports/PHASE0_STATUS.md`.
- **What "final status" means.** It is each WP's status after its last repair round and the
  independent verification of that round. "(open issues)" means the verifier left at least one
  major finding open. No WP had an open critical finding.
- **This report ran no new test.** Its new numbers are the Phase-1 rung scores read from the
  run files (5.3), the checker run (7) and the Claude API token totals (11). They are ledgered in
  `results/dataprep/ledger/rows/PHASE0_STATUS.csv`. Every other number is quoted from a WP report,
  and that WP ledgered it.

## 1. Summary

- **Scorecard.** There are 28 WP entries: the plan's WPs as revised (C0 and P4m added), plus the
  guard fix A0fix. Then there is the L3 release.
  - 13 passed, 4 of them with an open verifier major (A1, C0, P4m, H1a).
  - 14 are partial and 1 failed (H4).
  - The release is built and rebuilds byte-identically, but its verifier raised 3 major issues
    that need a corrected release or explicit waivers (section 6).
- **Changhua pumping is measured, not inferred.** C0's cleaned TPC census puts in-grid Changhua
  at 187.36 GWh/yr over 2012-22, the same as Yunlin's 188.44. Of that, 23.0 % is the domestic
  class (Yunlin 5.5 %), which may not be pumping. Its irrigation seasonal CV is 0.141, against
  Yunlin's 0.234.
- **Heads do not identify the magnitude of that series (P4m, frozen θ, in-sample).**
  - Metered forcing lifts the Changhua amplitude ratio from 0.38 to 1.02 without lowering the
    anomaly RMSE (1.191 → 1.244 m, interval covers 0).
  - The amplitude-matched scale (0.97) stays at 1.02-1.15 when the series is shifted by 2 months
    or scrambled across cells, so it identifies nothing.
  - No physically valid indicator supports a Changhua-specific response multiplier.
  - The observed 2012 initial condition is out of equilibrium with metered pumping (start-up
    transient −0.94 m).
- **Phase 1 has started, ahead of the planned code window.**
  - Rung R1 (batch 9, Changhua census added) and R1b (batch 10, plus canal v4) each failed 2 of
    their 4 pre-registered gates: the Changhua out-of-fold anomaly RMSE (R1 1.074 → 1.290 m) and
    the leveling column's out-of-fold R² (0.649 → 0.248).
  - R1c (county-split mid zone and county column) is running as batch 11.
- **Model-ready forcing and heads exist; subsidence validation does not.** The release carries
  electricity, canal, ET0, heads targets, an initial-condition field and the printed deep piles.
  Rings (S1a), GNSS (S3), InSAR (S4b), the fused field (S5) and the DTM (W3) failed their gates,
  so 2023-26 subsidence validation rests on the printed piles alone. H4's sub-grid operator
  failed.
- **The freeze has not lifted (decision D13).** Evidence doc §11 is written (commit 709f660), but
  a `par*` GPU job is running (`hydro-par11`, batch 11, started 2026-10-17 11:42). Nothing under
  `hydrophysics/` or `tests/` may change until `##### batch 11 finished` appears in
  `results/twin_runs/par11/controller.log`.
- **Compliance.**
  - No Phase-0 WP edited `hydrophysics/` or `tests/`, used the GPU or committed anything.
  - Claude API use was extraction only: 324,540 input and 86,659 output tokens (S2, N1, P6).
  - Two disclosed CPU-slot slips: P3 ran a 2-minute job outside the wrapper, and W2W4 ran its
    network-bound fetch outside it.

## 2. Status per WP

| WP | Final status | Headline (key number) | Verifier |
|---|---|---|---|
| A0 | passed (open issue closed by A0fix) | Checker over 16 manifests + 31 SOURCE.json: 0 failures, 0 L0 orphans. Three pilot spot reruns reproduce to ≤ 1.6e-12. Pre-origin ρ(k) 0.331 / 0.494 / 0.651 m (k = 20/15/10) replaces the plan's values, which used 2023-26 heads | issues → fixed by A0fix |
| A0fix | passed | Guard time-slice leak closed. On 29 synthetic probes the old guard leaked in 12 cases (25 rows), the fixed one in 0. 31/31 tests pass | confirmed |
| A1 | passed (open issues) | coverage3 "upper90" rule: ≥ 0.89 actual coverage in all 205 configurations (min 0.9094) inside its envelope (k ≥ 20 clusters). prereg3 refuses 7/7 probe scenarios | 2 majors open: prereg3 does not enforce the look rule on external units; gate-3 has no pooling helper |
| C0 | passed (open issues) | Census v3, in grid 2012-22: Yunlin 188.44, Changhua 187.36, Chiayi 0.63 GWh/yr. The December-2022 half-bill leak is removed (R2 invariance: 0 violations) | 1 major open: the fill sd is labelled "calibrated" but fails the plan's gate-3 (coverage 0.808) |
| P1 | partial | Repurposed as a census validator. Pre-origin a 0.959 confirmed against the census a* 0.990. Yunlin backcast fails (RMS 5.19 %, bias −5.04 %); bootstrap coverage 0.826 / 0.840; after-the-fact Yunlin extension +15.8 % | confirmed |
| P2abc | passed | Billing products v2 for both counties. Zero classes within 0.47 % of the pilot on 7/7 items. Meter skill ≥ 0.232 (Yunlin) / 0.205 (Changhua). July backtest 15.4 % / 8.7 %. Head placebo check not significant | confirmed |
| P2def | partial | The post-2016 synthetic layer fails its back-test: Yunlin township count error 0.579 (≤ 0.30), kWh +30.4 % (≤ 20 %). The extension to 2026-08 passes (annual RMS 0.070 / 0.054) | confirmed |
| P3 | passed | Methods test against the Changhua census. The LUIMAP random forest transfers the shape (ΔAA₅ₖₘ +0.044, corr 0.81, township R² 0.92), but not the level (0.84 ×) or the class split (0/9 maps). The crop-statistics model gives R² 0.20 | confirmed (CPU-slot slip, disclosed) |
| P40 | passed | Engine v2 replicates batch 8's merged cwa θ (sha256 789d0e45…): sparse vs torch 1.66e-6 m. The tangent-linear variance matches a stepwise Monte Carlo (median ratios 1.015 and 1.0000016), where the frozen-lift K Σ Kᵀ gives 0.693 and 0.877. RIV check case 2.71e-5 m (fails 1e-5) | confirmed |
| P4m | passed (open issues) | Metered Changhua forcing on the frozen θ: amplitude 0.38 → 1.02, RMSE 1.191 → 1.244 m (no detectable change). s_LS 0.292; s_AM 0.969 does not identify the magnitude. No valid multiplier evidence. 24-month burn-in recommended | 1 major open: the sign of E9 depends on which energy classes enter the regressor |
| P5 | passed | Heads give Changhua rank and timing only: S1c Spearman 0.27 (p 0.022), timing r 0.76-0.78. Every Changhua magnitude claim and transfer fails (R² −0.7 to −1.9) | confirmed |
| P6 | passed | Inventory reconciles with the MOEA annex: 775 vs 774 wells, volume within 0.0006 %. 129 Changhua Taiwan Water wells (53.3 Mm³/yr) still pumped through 2022 and are in neither the census nor the model | confirmed |
| N1 | passed | 64 sourced intervention rows. The 2015 shortfall is reproduced at held-out wells better than climatology (1.28 vs 1.77 m), at about half the observed amplitude. The 2020/21 primary tests could not be evaluated | confirmed |
| H1a | passed (open issues) | L1 heads v3: 7/7 registered criteria pass (recall 0.920 / 0.920; leakage 0 of 907,684 rows). Pre-origin corrected well-months fall from 714 (v1) to 0 | 1 major open: `events.csv` comes back empty through the guard, silently |
| H1b | partial | DTM fills for 0.0 / NaN GroundHeight: registered conformal coverage 0.900. All 60 + 32 model wells replaced. Release is blocked by W3's failed gate | confirmed |
| H2 | partial (open issues) | The complete gap fill fails in both rounds (12/14, then 11/14 strata). The daily bridge passes (coverage 0.908, RMSE 0.436 vs ≥ 0.847 m) and releases the targets. The GP initial-condition field beats the twin's construction narrowly (6.27 vs 8.42 m) | 1 major (disclosed): a required part is missing |
| H3 | partial | 15 recovered wells (1,839 targets), for rung R5 only. Plan acceptance fails: 3 of 15 wells pass the per-well coverage test. R-h found post-2018 records for 1 of 19 wells | confirmed |
| H4 | failed | Fails F2 (shuffled-position placebo p 0.090) and C2 (site-held-out Q4 excess +1.11 m²). The fitted correction's sd is 0.037 m | confirmed |
| S1a | partial (open issues) | Ring bridge v2: (e) worst window 2.79 cm (≤ 1.5) and (a) power 0.721 (≥ 0.8) fail; (b) 5/5 and the residual offset 13/14 pass. After-the-fact use only | 1 major open: the claim that the forcing rule is "not a lever" is wrong in magnitude |
| S1b | partial (open issues) | Registered checks pass under S1b's own 2-sigma rule (9/10, 14/14, r 0.714). The plan's literal bracket fails (3/10) | 1 major open: the argument for discounting the literal failure is not robust |
| S2 | partial | Feed piles fail plan test A (58/61 within 2 mm, max 3.33 mm). The printed piles v2 pass their transcription gate (0.975 / 1.000). Fresh test C1 fails (170/207) | confirmed |
| S3 | partial | GNSS gate fails twice: step recovery 0.863 (WRA) / 0.8625 (NGL) against 0.90; A2 same-sign share 0.50 against 0.75 | confirmed |
| S4a | partial | 929/929 primary-frame pairs fetched (206 epochs, 27 in 2022), 2.898 GB under the 3 GB cap. The primary subset is 2.43 GB, against D1's ~1.4 GB. No descending partner before 2025 | issues (minor only) |
| S4b | partial | G1 14-month RMS 13.2 mm (≤ 10) fails. Changhua Z2 summer coherence 0.297: not covered. Seasonal-amplitude validator 0/8. 51 % of values are bridged across network gaps (flag 4) | confirmed |
| S5 | partial | Fusion gate fails (T1 ratio 1.042, coverage 0.631; T2 coverage 0.940). Monthly field shipped on 0.72 of cell-months after a marginal M1 pass (0.972, P 0.086) | confirmed |
| W1 | partial (open issues) | Canal v4 (pre-origin) beats v3 and climatology on Jan-May in both districts (Yunlin bound −0.002). The head check fails in Yunlin layer 1 (−0.0087). Full variant 2/4. Forward archive blocked | 1 major (disclosed): the forward archive is missing |
| W2W4 | passed | ET0 extended to 2026-09-30 (2022 overlap MAE ≤ 0.063 mm/d; the cache uses GMT days). Kriging not adopted (−1.1 % against the 10 % needed) | confirmed (fetch outside slot, disclosed) |
| W3 | partial | The MOI 20 m DTM fails its held-out bias gate: −0.842 m (90 % −0.908..−0.783) against ≤ 0.30. The twin keeps SRTM (+1.25 m) | confirmed |
| L3 | built (open issues) | `release_20261012`: 9 pre-origin and 8 full components plus the grid, from 13 gate-passed versions. 77/77 files identical on rebuild | 3 majors open (section 6) |

## 3. Acceptance criteria: what passed and what failed

Counts are the WP's own acceptance rows (registered, plan and repair criteria). The failed rows
are listed with the measured value and the threshold. Rows a WP declared report-only or
after-the-fact are marked as such.

| WP | Rows passed | Failed criteria (measured vs threshold) |
|---|---|---|
| A0 | 12/12 | — (the guard defect the verifier found is fixed in A0fix) |
| A0fix | 11/11 | — |
| A1 | 15/19 | All informational or superseded: outside the envelope, upper90 falls below 0.89 once in 29 configurations (k10 × 50, ICC 0.7: 0.8823). No candidate in the design sweep met 0.89 everywhere (design-time). The level-0.90 sensitivity falls below 0.89 in 12/205. The first fresh-unit declaration v1 failed for 2 of 17 units (superseded by v2). The verifier's 2 majors are open |
| C0 | 17/19 | Task (d) taken literally ("within rounding" of the model's forcing) is not met, by design: max 3.15e-4 per month before 2022-12, −3.35 % in 2022-12. The v2 gates failed (v2 superseded). Verifier: the R7 "calibrated" label fails gate-3 (0.808, interval excludes 0.90) |
| P1 | 11/20 | BV1 Yunlin: RMS 5.19 %, bias −5.04 % (≤ 5 %, ≤ 3 %). T8 coverage 0.826 pre-origin, 0.840 full (0.85-0.95). T10 +15.8 % (≤ 10 %, after the fact). G3: 4 of 1,024 rows. tpc_observed gate failed. C1 full-window a not confirmed. C4 φ z = +2.06. C9 3.3e-4 GWh (≤ 1e-6) |
| P2abc | 13/18 | None gating. The first full assembly failed on 1 pump-month (repaired under a registered rule). Chiayi July backtest 33.8 % (reported). Chiayi meter skill not evaluable for n = 30-300. Changhua Chow-Lin sd coverage fails at n ≥ 10, so it is labelled uncalibrated. Head placebo p 0.051 (Yunlin) and 0.786 (Changhua), not gating |
| P2def | 12/16 | P2d back-test: Yunlin count error 0.579 (≤ 0.30) and kWh +0.304 (≤ 0.20); Changhua count error 0.332. P2f placebo: Changhua layer 1 p 0.660 |
| P3 | 11/17 | Registered: crop-statistics township R² 0.201 (≥ 0.5); class split, 0 of 9 maps at TV ≤ 0.10; heads-only block Spearman −0.285 (≥ 0.5). Declared comparisons: 12 of 25 scout totals within \|ln\| 0.30; heads-only I2 at 0.232 × truth; the φ_in estimators do not bracket the census 0.959 |
| P40 | 16/18 | RIV check case 2.71e-5 m (≤ 1e-5; 2.71e-9 m at a CG tolerance of 1e-12). v2 vs v1: 3 eigensolver estimates differ by ≤ 8.4e-15 relative |
| P4m | 18/18 | — (the verifier's E9 class-dependence finding is open) |
| P5 | 7/13 | Changhua claims t4a 1/5, 1/5, 0/5 (≥ 4/5). Transfer t4b R² −1.56. t5 local signal: Yunlin S1c only. t6 seasonal-CV α proxy 0.936 [0.778, 1.070] vs census 0.604. t7 transfer-function gains 1/5. t8 EEMD 3/5 (negative), EOF 0/5 |
| P6 | 7/7 | — |
| N1 | 6/6 | — (the 2020 and 2021 primary tests were not evaluable, with 3 and 2 sites) |
| H1a | 10/10 | — (the verifier's `events.csv` guard finding is open) |
| H1b | 10/11 | Release: the upstream W3 gate failed and no user decision is recorded |
| H2 | 11/15 | v1 coverage 12/14 strata after the family-E correction. v2 coverage 11/14 and RMSE in N_net2 0.783 vs clim+AR(1) 0.764. Initial conditions and IDW fields from flag ≤ 4 fills: not released |
| H3 | 9/12 | H3.d_wells: 3 of 15 admitted wells pass (plan acceptance part 1). R-h: 1 of 19. The R5 rung gate is not evaluated (Phase 1) |
| H4 | 10/13 | F2: placebo p 0.090 (≤ 0.05). C2: Q4 excess −24.304 → −23.194 m², a reduction of −1.111 (must be > 0). The v1 gate failed (A8 1.0207) |
| S1a | 6/9 | (e) 2.79 cm (≤ 1.5, threshold post hoc). (a) power 0.721 (≥ 0.8). Product gate failed |
| S1b | 6/11 | NGL family not judgeable (n = 1). Plan-literal (c) 3/10 unwidened (and the plan-literal conjunction). d1 bridged-over-raw p 0.072 (not registered). Power against a 25 % scale error 0.06-0.12 |
| S2 | 11/14 | A: 58/61 within 2 mm (max 3.33 mm). Product gate failed. Registered C1: 170/207, r 0.750-0.864 |
| S3 | 8/13 | A1: WRA 0.863, NGL 0.8625 (≥ 0.90). A2 same sign 0.50 for both common modes (≥ 0.75). Full-record A1 0.819 / 0.868 (reported) |
| S4a | 8/9 registered | S4a-7 against D1: primary subsets 2.430 GB (about 1.4; fail above 1.5) |
| S4b | 6/12 | G1 13.2 mm (≤ 10). C1 Z2 0.297 (≥ 0.30). Y2 0/8 (≥ 6/8). Asc/desc decomposition not possible. Round-1 look logging failed. Plan validation rows not run |
| S5 | 12/16 | T1 ratio 1.042 (< 1) with coverage 0.631. T2 coverage 0.940, whose interval 0.911-0.960 excludes 0.90. Gate failed |
| W1 | 16/20 | Full variant 2/4. Head check: Yunlin layer 1 −0.0087 (must be ≥ 0). 2026 chain "not distinguishable from climatology". Repair round 1 labelled too few intervals (fixed in round 2) |
| W2W4 | 8/10 | W4 adoption: −1.1 % (≥ 10 %), and 2 gauges above 1.20 × (rf25 1.828, rf3 1.265). Masked IDW stays, as the plan's rule says |
| W3 | 8/11 | Gate: −0.842 m (\|bias\| ≤ 0.30). sd_point coverage 0.689. The v1 and v2 manifests fail the checker (code edited in place) |
| L3 | 12/12 (builder) | Verifier: plan 2.8 item 2 is applied inconsistently (S2 and W1 included, H3 excluded), and the pre-origin piles carry values first published in 2023 (section 6) |

## 4. Deviations from the plan

### 4.1 Plan-level deviations

- **Revision of 2026-10-12.** C0 was added. P4.1 (E1-E7, E11), the P4.2 members, P4.3's strict
  pseudo tests and level discrimination, and the μ table were superseded by P4m. E8, E9 and E10
  survive inside P4m. P1 became a census validator and P3 an out-of-sample methods test, so
  neither ships a prior.
- **Code location (D12).** Code lives in `dataprep_staging/`, not `results/dataprep/_staging_code/`.
  When the Phase 0 products were built it was not committed: their manifests record the
  repository commit of the day (709f660 for the N1 inputs) with each staging file marked
  `git_state: untracked`, so provenance rests on sha256 hashes. `dataprep_staging/` has since
  been committed (68ee347, 2026-10-18; 949 files at the PR head).
- **Ledger.** One fragment per WP (`ledger/rows/<WP>.csv`, with columns `wp` and `note` added). L3
  reassembled `data_roles.csv` (16,161 rows, 0 conflicts). This report added 16 rows; all 16,177
  rows have 0 conflicts. `data_roles.csv` has not been reassembled since.
- **Pre-registration tooling changed during Phase 0.** prereg became prereg2 and then prereg3.
  coverage became coverage2 and then coverage3 (rule gate-3), whose envelope is k ≥ 20 clusters;
  strata with fewer clusters are labelled uncalibrated. Twelve early registrations were written
  outside the tool (P1, P2a-c, S2, W1, W2, W3, W4). The tool cannot verify them; several were
  later indexed through declarations. The look log is self-reported.
- **U2 σ_obs.** ρ(k) was recomputed on 2012-22 data (0.331 / 0.494 / 0.651 / 0.804 / 0.862 m), because
  the plan's values had used 2023-26 heads (A0).
- **Repair rounds.** Independent verification and one to four repair rounds per WP went beyond
  the plan. Every repair made a new version; superseded versions carry `SUPERSEDED.json`.
- **Post-look builds.** In H1a (v3), H1b (v2), H2 (repair 3) and S5 (v3) the product for use is a
  post-look reviewed build, labelled as such. The registered builds are kept for provenance.
- **Phase order.** S5 ran in Phase 0 (the plan puts it in Phase 1), and an L3 release was built in
  Phase 0.
- **Phase 1 started outside the planned code window.**
  - The GPU rungs began on 2026-10-14 from separate worktrees with uncommitted hooks, before the
    port, the hooks, the coordinate fix and the ring-loader fixes.
  - The rung gates use the full 5-fold k-fold on the design-time partition, instead of the plan's
    temporal screen plus a CPU held-out-site proxy.
  - The rungs read `pump_electricity/model_census/v1` (Yunlin rows equal to the registered
    forcing; December-2022 half bills kept under a declared exemption), not C0 `preorigin/v3`.

### 4.2 Plan facts corrected in Phase 0

| Plan statement | Phase-0 finding | WP |
|---|---|---|
| φ_in ≈ 0.75-0.82 scales the Changhua county series to the grid | 0.959 of Changhua pump electricity lies in active cells; area scaling would understate in-grid Changhua by about 22 % | P1, P3 |
| a ≈ 0.94-0.96 × Yunlin, "confirmed" | census a* 0.990 (2016-22) and 1.031 (2012-22); P1's pre-origin a 0.959 is 3.1 % low; like-for-like without domestic 0.795 | P1 |
| Changhua seasonality unknown (α) | irrigation CV 0.141 vs 0.234 (ratio about 0.6); all classes 0.117 vs 0.141 | C0, P4m |
| The 2022 ET0 difference of 0.048 mm/d is GMT vs local days | the cache uses GMT days; the difference is an Open-Meteo archive offset (0.049 mm/d), constant within 0.003 mm/d over 2012-21 | W2W4 |
| WRA-to-model volume ratio 2.58 | pre-origin 2.47, falling from 2.74 (2013) to 2.09 (2021) | P6 |
| Heads-only Changhua magnitude ≈ 0.25 × Yunlin | an artefact: I2 is 0.23 × the census, and its block pattern anti-correlates with it (−0.28) | P3, P5 |
| P2c: 878,738 structural zeros | an all-meter statistic, recomputed exactly; the kept-meter count is 902,981 | P2abc |
| H3: 16 candidates; 18 wells stop in 2018-12 | 15 candidates on L1 v3; 19 on-grid wells stop in 2018-12 | H3 |
| The 2025-26 census decline may be missing readings | missing share ≤ 0.05 %; TPC posts whole bimonthly bills to the second month; the census/TPC ratio fell from 0.50-0.53 to 0.425 | P2def, P1 |
| Public deep wells: a missing deep stress | confirmed and sized: public-supply purposes carry 4.2 GWh/yr in the census against about 15.7 GWh/yr census-equivalent for the 2011-19 cut alone | P6 |

### 4.3 WP-level deviations that change how products may be used

| WP | Deviation |
|---|---|
| C0 | The pre-origin forcing holds a flag-4 fill in 2022-12 for Dec-Jan-paired meters. Two exemptions: the 2026 census attribute table (≤ 3.7e-4 of energy), and residual unpaired halves (0.68 / 0.57 / 0.001 % of December-2022 energy, added post-look). The pre-install coding is not harmonised (Changhua NaN, Yunlin 0) |
| P2abc | The name `yunlin_electricity` is kept, although the product covers both counties and in-grid Chiayi. The Yunlin July-August 2025 zero bills are treated as synchronized zeros (flag 9) with sensitivity members |
| P2def | The post-2016 layer, extension and layer response are separate products. The extension covers 2026-07..08 only. P2f is a declaration, not a test |
| P40 | Kernels hold 180 lags. The RIV verification uses a synthetic configuration. The bound-pinned profiles are convergence-limited (refits stopped at the cap) |
| P4m | E8 was rebuilt pre-origin from L0 hourly heads. E9 was rerun with the declared site definition. A post-look revision withdrew the hashed multiplier recommendation and the R1m rung |
| P6 | The primary member uses all 2011-2019 cohorts (1,056 wells); the plan's 2011-18 ramp is a variant. The Tuku-Huwei component was added after the declaration |
| N1 | Held-out non-calibration wells serve as external units, and prereg3 does not check their freshness. Masked-year refits are declared for Phases 1-2 |
| H1a | v3 (a post-look review: co-located backing alone no longer corrects a step) is the product for use. v2 is kept for provenance only |
| H1b | Separate metadata product. v2 also replaces 11 agency values that are inconsistent with WellElevation (post-look) |
| H2 | Trailing envelope of 48 months, not 66. No complete fill is released. The initial-condition field comes from the twin's own information set |
| H3 | Only the daily-bridge stratum is gated, per well. Failing wells enter with flags 0-1 only |
| H4 | The v2 operator subtracts the cell-mean kernel (a correction of the plan's formula). It is tested on held-out non-calibration wells |
| S1a | GNSS is not an evidence line, because S3 failed. The offset clusters were widened by 0.5 cm after the pilot tables had been seen |
| S1b | Check (c) was made operational as the excess-rate bracket widened by 2σ. The plan's literal bracket fails |
| S2 | TKJS_283 is excluded as defective. The 2023 printed 007001 values serve a fresh test C |
| S3 | Pre-origin and full variants. A1 gates on the pre-origin variant. A2 must hold for each offered common mode |
| S4a/S4b | The whole twin grid plus 2 km is clipped. G1 is scored on the pre-origin run P. The plan's leveling, ring/pile and printed-pile validation rows were not run |
| S5 | Two-stage monthly field. Exact Gaussian inference. T2 has no buffer (a buffered rerun after the fact gives coverage 0.876) |
| W1 | State space over 2012-22 (pre-origin) and 2012-25 (full); the yearbook gauges start in 2012. v4 is at the weir level: v3 is 1.615 × (Changhua) and 1.122 × (Yunlin) higher |
| W2W4 | GMT days. The main array is level-matched to the cache from 2023-01 (flag 1). The raw splice is kept |
| W3 | The product is v3, with an epoch correction from WRA leveling. The time-varying z(t) for 2022-23 is extrapolated (flag 6) |
| L3 | Each component is its own L3 product version. Gridded upstream files are byte copies. ET0 stays at its 61 native points. `prior_width_input` is not granted on electricity |

## 5. Changhua: the C0 census product and the P4m diagnostics

### 5.1 C0, the cleaned pump-electricity census (`pump_electricity/*/v3`)

- **Source and cleaning.** C0 reads only the census's `parquet/` derivatives; the name and
  address columns were never read. Cleaning is `hydrophysics.twin.pumping.clean_census` exactly as
  the model applies it to Yunlin: cap_duty 1.0, duty window 2012-2022, shared meters counted once.
  On the model's inputs it reproduces the model's Yunlin forcing to 0.0 kWh (T1a).
- **Products.**
  - `census_L1/v3`: per pump, with coordinates and well numbers, so `personal_data` is declared
    and the product stays out of the release.
  - `preorigin/v3`: 2007-01..2022-12; class × cell × month value / sd / flag, per county; a
    domestic-excluded member (`cells_total.npz` `value_nodomestic`).
  - `full/v3`: 2007-01..2026-07, with 2026-07 masked.

| County | Kept pumps in active cells | In-grid GWh/yr 2012-22 (pre-origin v3) | Domestic share | Irrigation seasonal CV |
|---|---|---|---|---|
| Yunlin | 114,060 | 188.44 | 5.5 % | 0.234 |
| Changhua | 113,108 | 187.36 | 23.0 % | 0.141 |
| Chiayi (in grid) | 316 | 0.63 | 0 % | — |

- **The leak and its fix.**
  - In v1 (as in the model's own cached forcing), the 2022-12 value of every Dec-Jan-paired
    meter was half of a bill read in January 2023: 25.7 / 33.0 / 59.4 % of in-grid December-2022
    energy (Yunlin / Changhua / Chiayi).
  - v3 detects the pairing on pre-origin pairs only and fills that month from the Oct-Nov 2022
    base times a 2012-21 ratio (flag 4).
  - The perturbation test R2 changes every month from 2023-01 on and finds 0 differences in the
    pre-origin products.
  - The fill's sd back-test reaches 0.808 coverage. That fails the plan's gate-3 (C0 verifier),
    so the sd must be read as **uncalibrated**; the release already treats it so.
  - The model's cached forcing differs from v3 by −3.35 % in 2022-12.
- **Limits.**
  - The census is frozen at mid-2016 installs; P2d's synthetic layer failed its back-test.
  - It excludes the public and industrial deep wells (P6).
  - TPC scaling is not valid (coverage about 0.4, not stationary).
  - July 2026 is partial.
  - The bimonthly even split stays the primary arm: the Chow-Lin arm shows no placebo-controlled
    gain at cell scale.
- **Checks against independent data.**
  - P1: the census confirms the TPC-based pre-origin share, but the Yunlin extension
    over-predicted by 15.8 % after the fact (ratio drift).
  - P3: land use transfers the within-county shape (township share R² 0.92) but not the level
    (0.84 ×) or the class split. No Yunlin-trained map sees Changhua's domestic class.

### 5.2 P4m, metered-forcing diagnostics

Setup: the frozen batch-8 cwa θ, the P4.0 engine (stepwise, lift feedback), and Changhua plus
in-grid Chiayi from C0 `preorigin/v3` (187.99 GWh/yr; 144.91 without domestic). All of it is
in-sample design-time evidence on the 158 calibration wells, 2013-2022. The intervals are
cluster bootstraps and are uncalibrated.

| Changhua wells (74) | A0, zero Changhua | A1, metered | A2, metered without domestic |
|---|---|---|---|
| Anomaly RMSE (m) | 1.191 | 1.244 (+4.5 %, interval −0.014..+0.136) | 1.186 |
| Amplitude ratio | 0.38 | 1.02 | 0.95 |
| Median correlation | 0.53 | 0.63 | 0.67 |
| Yunlin anomaly RMSE (84 wells) | 1.888 | 1.916 (+1.5 %) | 1.912 |

- **Scale the heads accept.**
  - Least squares gives 0.292 over all wells (attenuated). The amplitude-matched scale is 0.969,
    but it stays at 1.02-1.15 when the series is delayed or advanced by 2 months or permuted
    across cells, so it cannot identify the magnitude.
  - In the CPU emulation of an R1 refit, the heads prefer about half the metered series:
    s_θ 0.48, Changhua RMSE 1.112 m at s = 0.5 against 1.153 m at s = 1. θ bias explains about one
    third of the gap.
  - The rest is response or conversion, and these data cannot separate the two.
- **Seasonal cycle and pattern.** The frozen θ accepts 0.10 (OLS) / 0.23 (GLS) of the metered
  seasonal part. Removing the seasonal part fits better (1.130 m), and a cell-scrambled series
  fits as well as the true one (1.223 m).
- **Initial condition.** Metered forcing creates a start-up transient of −0.94 m (−1.36..−0.54).
  A 2007-11 spin-up moves layer-1 Changhua heads by −6.25 m. The level (mean) channel stays
  negative under GLS in 3 of 4 E10 variants, so it is logged for plan 7.3. A 24-month burn-in is
  the only treatment that passes under both OLS and GLS.
- **Indicators for a Changhua multiplier.**
  - BE: the layer-1 medians are negative in both counties, which is physically invalid. The
    valid-epoch fallback gives 0.94.
  - E8: not robust to the easting adjustment (2.23 → 1.35; raw pump-layer ratio 0.85).
  - E9: with the all-class regressor, 10 of 13 Changhua slopes are positive. The verifier found
    that with irrigation energy only, 9 of 13 are negative (median −1.89 against the model's
    −22.1 m/GWh). E9 is therefore class-dependent and inconclusive, not "sign-reversed".
  - Either way, no indicator is both physically valid and robust, so P4m recommends no
    multiplier and no R1m rung.
- **Recorded arm 2023-25 (U4; reported, never gating).** Metered forcing on the frozen θ worsens
  Changhua by 20.5 % and Yunlin by 2.6 %.
- **Recommendation for R1.**
  - Primary arm: C0 `preorigin/v3`, all classes; the domestic-excluded member as the declared
    sensitivity.
  - No multiplier.
  - A spin-up or a 24-month burn-in.
  - Rerun the P4m diagnostics on R1's θ: s_LS in [0.8, 1.25], and E10 not significantly negative
    under both OLS and GLS.

### 5.3 The first Phase-1 rungs (GPU)

The k-fold, leveling and in-sample values below were read from the run files for this report.
The county anomaly RMSEs, and the amplitudes, correlations and θ shifts in the reading below,
are as recorded by the Phase-1 session and were not recomputed here. All comparisons use the
design-time 5-fold partition of R0 (batch 8, cwa).

| Gate (registered) | R0 (cwa) | R1 (batch 9: + census) | R1b (batch 10: + canal v4 and paddy percolation) |
|---|---|---|---|
| k-fold out-of-fold R² ≥ R0 − 0.01 | 0.643 | 0.652 pass | 0.642 pass |
| Changhua out-of-fold anomaly RMSE ≤ R0 | 1.074 m | 1.290 m **fail** | 1.264 m **fail** |
| Yunlin out-of-fold anomaly RMSE ≤ R0 × 1.02 | 1.856 m | 1.848 m pass | 1.777 m pass |
| Leveling column out-of-fold R² ≥ R0 − 0.02 | 0.649 | 0.248 **fail** | 0.335 **fail** |
| In-sample R² (reported) | 0.865 | 0.836 | 0.827 |

- **Reading.**
  - Adding the metered Changhua census makes the Changhua amplitude roughly right (0.39 → 1.13)
    without improving its correlation (0.46 → 0.48).
  - The optimiser compensates fan-wide: return fraction 0.42 → 0.63, storage up, mid/distal
    layer-2 T down about 30 ×, spread 0.5 → 3.5 km.
  - The column collapses.
  - This repeats P4m's in-sample finding at full calibration: Changhua draws Yunlin's
    electricity, yet swings about a third as much and subsides about a quarter as much, and
    easting-only zones cannot hold both.
- **R1c (batch 11, running).** It splits the mid zone by county and makes the column a 6-zone
  county split (declaration `I1.R1c.county_zones.v1`). A design-time pass must be confirmed on a
  fresh partition (seed 2027, 200 m site tolerance) before R1c can become the deliverable.

### 5.4 Inputs to the Changhua decision

- **Established.**
  - Changhua's in-grid metered electricity equals Yunlin's (187.36 vs 188.44 GWh/yr).
  - Its irrigation seasonality is about 0.6 of Yunlin's.
  - About a quarter of it is a domestic class that may be household load.
  - The frozen θ cannot use the series' seasonal cycle or its spatial pattern.
  - A recalibration with it (R1, R1b) fails the Changhua head and leveling gates.
- **Not established.**
  - Whether the metered magnitude is right in model units. Heads do not identify it, and the
    conversion (η, head_extra) is confounded with response.
  - Whether Changhua's aquifer or column responds differently (R1c tests this).
  - Whether part of the load is not groundwater pumping.
- **To declare before any R1-family score is read (U7).**
  - The electricity arm: all classes or no domestic.
  - The census source: C0 v3 or model_census/v1 with its exemption.
  - The burn-in or spin-up treatment.
  - The rule for promoting a rung.
- **Evidence only a data request can supply.** R-b (WRA's water-to-electricity table and the
  Changhua registered-well register) and R-g (the water-to-electricity calibration) would measure
  the conversion. Drafting them is not approved.

## 6. The L3 release `release_20261012`

Location: `results/dataprep/model_ready/release_20261012/`, with `RELEASE.json`, `PRODUCT_CARD.md`
and `REBUILD_CHECK.json`. Each component is a product version with its own manifest, so the
guard enforces its uses.

| Component | Upstream | Allowed uses |
|---|---|---|
| preorigin/electricity_census | pump_electricity/preorigin/v3 (C0) | forcing_frozen, model_design, prior, design_time_evidence, validator, after_the_fact_analysis |
| preorigin/electricity_billing_arms | yunlin_electricity/preorigin/v2 (P2abc) | same as above |
| preorigin/surface_water | surface_water_v4/preorigin/v2 (W1) | forcing_frozen, model_design, design_time_evidence, after_the_fact_analysis |
| preorigin/public_wells_member | public_wells/v2 (P6; sensitivity member) | forcing_frozen, model_design, prior, design_time_evidence, validator, after_the_fact_analysis |
| preorigin/et0 | et0/v1 pre-origin output (W2W4) | forcing_frozen, statistical_layer_history, after_the_fact_analysis |
| preorigin/heads_observed | heads/L1_clean/v3 sliced to 2022-12 (H1a) | calibration_target, datum_fit, baseline, initial_conditions, idw_field, model_design, selection, scoring_target, validator, … |
| preorigin/heads_targets | heads/L2_filled_preorigin/v4 (H2 bridge rows) | calibration_target |
| preorigin/heads_initial_conditions | heads/L2_filled_preorigin/v4 (January-2012 GP field) | initial_conditions |
| preorigin/piles_printed | deep_benchmarks_printed/v2 sliced to 2022-12 (S2) | calibration_target, datum_fit, baseline, design_time_evidence, validator, scoring_target, after_the_fact_analysis |
| full/electricity_census, full/electricity_billing_arms | C0 full/v3, P2abc full/v2 | forcing_recorded_arm, statistical_layer_history, validator, after_the_fact_analysis |
| full/electricity_extension | census_extension/full/v1 (P2e) | forcing_recorded_arm, after_the_fact_analysis |
| full/surface_water, full/et0 | W1 full/v4, et0/v1 | forcing_recorded_arm, statistical_layer_history, after_the_fact_analysis |
| full/heads_observed | heads/L1_clean/v3 (H1a) | scoring_target, validator, statistical_layer_history, after_the_fact_analysis |
| full/heads_filled | heads/L2_filled_full/v2 (H2 bridge) | statistical_layer_history, after_the_fact_analysis |
| full/piles_printed | deep_benchmarks_printed/v2 (S2) | validator, scoring_target, after_the_fact_analysis |
| grid/v1 | results/twin_forward/datum_gate_innov.npz | geometry only (nx 59, ny 77, dx 1 km, 2,148 active cells; mask hash fae17e0a…) |

- **Excluded: 73 versions or WP-level items, each with a reason.**

  | Reason | Versions |
  |---|---|
  | Gate failed (P1, P2d, H1b, H2 complete fills, W3, S3, S4b, S1a, S2 feed piles, S5, H4) | 31 |
  | Superseded | 17 |
  | Not model data (P4.0 engine, P4m diagnostics, P5, N1, P2f, W4, A1 units, LiCSAR subsets) | 13 |
  | Phase-1 artefacts (model_census, sw model npz, zone_county_map) | 4 |
  | Not a product (L0 registry, S5 dry run, P3, pilots) | 4 |
  | Privacy (C0 census_L1) | 3 |
  | Plan acceptance failed (H3 recovered wells v3) | 1 |

- **Checks.**
  - The builder's acceptance passes 12/12: plan 2.8 items 1-5; 306 guard probes with 0
    mismatches; 77/77 files identical over two builds.
  - The verifier's independent third rebuild was byte-identical too.
  - Every component manifest re-hashes in today's checker run (section 7).
- **The verifier's three major issues, all open.**
  1. The S2 printed piles are included although plan S2's acceptance failed (A 58/61; fresh C1
     170/207). That is the same pattern the release used to exclude H3. `preorigin/piles_printed`
     also grants calibration_target, datum_fit and baseline, whereas S2 limits the printed
     product to validator and scoring uses.
  2. W1 `preorigin/v2` is included although its pre-registered "not worse" head check failed
     (Yunlin layer 1 −0.0087), and the inclusion basis misstates W1's status.
  3. `preorigin/piles_printed` holds 21 primary values (2022-02, 2022-03, 2022-10) that are known
     only from 2023 re-printings, plus 49 rows with 2023 reprint columns. It lists no exemption,
     yet grants uses that cannot be exempted.
- **Minor issues.**
  - 13 stations established in 2023-24 appear in the pre-origin station table.
  - U1 is enforced only through `read_table`, not through `path()`.
  - Three upstreams are L1, whereas plan 2.1 says L2.
  - The "latest dated value" check covered time axes only.
  - Whether the exceptions were declared before any trial build cannot be verified.
- **Consequence.** Until a corrected `release_20261012b` (section 9, step 8) resolves the three
  majors:
  - the electricity, ET0 and heads components are unaffected, except that users of the
    pre-origin station table should drop the 13 stations established after 2022;
  - the canal component can be used with W1's failed head check stated;
  - the pile components should not be used for any fitting purpose.

## 7. Integrity of the product tree (checker run of 2026-10-17)

`python -m common.check` over the whole tree, run in a CPU slot:

- **Manifests.** 117 checked, 109 ok. All 18 release components and every current product version
  re-hash. 12,555 files were re-hashed in all.
- **The 8 manifests that fail, all expected.**
  - `ground_elevation/v1` and `v2`: W3 edited the code in place; v3 is current.
  - `heads/L1_clean/v2`: provenance only; the exact bytes are archived.
  - `heads/recovered_wells/v2`: superseded within its round; bytes archived.
  - `public_wells/v1`: superseded.
  - `subsidence_fused/v2`: superseded; bytes archived.
  - S5's dry-run `v9999`: not a product.
  - `zone_county_map/v1`: a Phase-1 artefact whose code path points outside the repository, into
    a Phase-1 worktree; superseded by v2.
- **SOURCE.json.** 52 checked: 49 ok, 2 legacy, 1 failed. The TPC census SOURCE.json uses
  `files[].path` where the checker expects `file`, so its 19 files count as L0 orphans. Fixing
  the key is the owner's edit; none of the census's raw or extracted files needs to be read.

Evidence: `results/dataprep/reports/PHASE0_STATUS_evidence/checker_2026-10-17.{json,log}`.

## 8. Open problems

**Changhua and forcing**

1. R1 and R1b fail the Changhua head and leveling gates; R1c is pending. If R1c fails, the
   remaining candidates are a Changhua conversion (η, head_extra), the seasonal timing and layer
   allocation of the forcing, and non-pumping load. Only R-b or R-g data could separate them.
2. The Phase-1 rungs use `model_census/v1`, whose 2022-12 carries January-2023 half bills
   (exemption `R1.model_census.dec2022_halfbill`). A rung that becomes primary needs C0 v3 or a
   sensitivity run of that exemption.
3. The domestic class (23 % of Changhua) may be household load. The arm must be chosen before
   scores are read.
4. Post-2016 pumps are missing. The P2d layer failed (counts +40 %, kWh +30 % in Yunlin), so the
   zero member stays the default and low/high are upper-side bounds at most.
5. Public deep wells are missing, among them Changhua Taiwan Water at about 53 Mm³/yr through
   2022. Only the P6 sensitivity member exists, and R-i was not drafted.
6. Most forcing sds are labelled uncalibrated: C0's December-2022 fill; P2abc's Changhua and
   Chiayi Chow-Lin, imputation and member sds; P2def; W1's monthly and 2026 intervals. Only the
   Yunlin Chow-Lin sd and W1's Jan-May aggregate passed a coverage rule, the latter weakly. The
   plan 4.8 propagation must not take the others at face value. The `forcing_uncertainty`
   product does not exist, and `common/uses.py` has no use for it.
7. The 2026 Yunlin canal values are probably biased high: chain residuals for 2020-25 are all
   negative.

**Heads**

8. No complete gap fill passed, so there is no IDW head field and no flag ≤ 4 initial
   condition. The GP initial-condition gain is narrow, and it is worse at late-starting wells.
9. H4 failed. The Q4 high-swing deficit is a forcing or parameter question (the pumping spread
   sits at its 0.5 km bound), not an observation-operator fix.
10. H1a `events.csv` reads back empty through the guard (the `flag` column has event semantics).
    P5 used a raw `path()` read.
11. GroundHeight is unreliable at 159 of 344 stations. The DTM replacement waits for W3.

**Subsidence**

12. Every subsidence product except the printed piles failed its gate: rings (e) 2.79 cm, GNSS
    0.863, InSAR 13.2 mm, fusion T1 1.042. The printed 2019-21 pile months are not corroborated
    by the feeds.
13. S1's checks rest inherently on post-gap (2023+) data, and S5's GNSS gap rule was motivated
    by 2023+ evidence. Both are U7 tensions awaiting a user ruling.
14. InSAR summer coherence does not cover Changhua Z2 (0.297). Before 2025 there is no
    descending geometry. 51 % of InSAR values are bridged across network gaps.

**Harness and process**

15. prereg3 does not enforce the look rule for external units. Gate-3 has no pooling step for
    strata with fewer than 20 clusters.
16. 72 of 105 fresh-unit pairs overlap. In-sample looks by P40, P4m, H2 and H4, and S4b's InSAR
    looks on all active cells, can block later registrations on fresh units. A registration
    order and a data-type ruling are needed.
17. The 14-stratum H2 gate fails a perfectly calibrated product at least once with probability
    0.51. A multiplicity-aware rule is undecided.

**Provenance and housekeeping**

18. *(Resolved.)* `dataprep_staging/`, the plan and the audit were committed in 68ee347; the
    three worktree code lines (pump paths, county zones, the ring_holdout reference fix) in
    8d51b46.
19. Four staging files and the plan's Appendix A contain a local home path or user name.
20. Disk to recover by hand:
    - about 1.2 GB in a P40 staging directory (deletion was refused by the session's permission
      classifier);
    - C0's 198 MB build cache;
    - about 3.7 GB of H1a scratch pickles;
    - a 62 MB P3 classification cache.

## 9. Phase 1 checklist

Status marks: **[done]**, **[running]**, **[open]**, **[decision]** (needs the user).

**Step 0. Freeze-lift trigger (D13)**

- [done] Evidence doc §11 is written (commit 709f660).
- [running] Batch 11 (rung R1c; `hydro-par11`; `results/twin_runs/par11/`) runs from the
  Phase-1c worktree through `par11/py.sh`.
  - Stage A (full fit plus 5 folds) started at 11:42; the full fit was at epoch 400 of 500 at
    14:08.
  - By the batch header's stage times it should finish around midnight on 2026-10-17/18.
  - Wait for `##### batch 11 finished` in `par11/controller.log`, and confirm that no other `par*`
    controller is active.
- Until then, touch neither the main tree's `hydrophysics/` and `tests/` nor the Phase-1c
  worktree's twin modules.

**Step 1. Read R1c as registered** (CPU, right after the batch)

- [open] Score R1c's four gates against R0 on the design-time folds.
- If all four pass, run the registered fresh-partition confirmation (R0 and R1c re-run
  k-fold, seed 2027, 200 m site tolerance) before R1c can become the deliverable recipe.
- If it fails, follow the declared interpretation:
  - head gate fails: the Changhua forcing is wrong (conversion, seasonality or layer
    allocation);
  - only the leveling gate fails: the column needs more than a county split.
- Record the verdict in `ledger/rows/I1.csv` and in the evidence doc.

**Step 2. Code window** (no GPU job running; every change bit-identical when its option is
absent)

1. [done, commit 8d51b46] **Merge the three uncommitted code lines** onto one branch from 709f660:
   - the pump-path hooks (branch `twin/phase1-changhua-census`);
   - the county zones (branch `twin/phase1c-county-zones`; 12 twin modules, `zone_county.py`,
     2 tests);
   - the ring_holdout reference fix (branch `claude/objective-hofstadter-b8e942`).

   `mechanism.py`, `ring_holdout.py` and `viewer_app.py` are edited in both of the last two. The
   Phase-1c change to `ring_holdout.py` passes the county-map hash to `load_columns`; it needs
   the ring_holdout session's agreement (plan 5.5). Then:
   - run `pytest -q` and ruff;
   - run the Rule 1 scan;
   - commit only when the user asks.

   From then on every rung records a git commit instead of a `code.diff` hash. *(Not followed:
   batches 14-25 still recorded a `code.diff`, and par23-25 no `code.base`; see "N1
   policy-response tests" below for the reconstructed commits.)*
2. [open] **Port the staging library to `hydrophysics/dataprep/`.**
   - Modules: `common/` (manifest, guard, uses, flags, ledger, source, check, paths, hashing) and
     `harness/` (scorers, coverage3, prereg3, gaps_guarded, placebos, units), with their tests
     (31 and 64-76).
   - CI rules: `from __future__ import annotations`; ruff at line length 100;
     `pytest.importorskip("torch")` where torch is imported; no `tests.` imports;
     Python ≥ 3.10.
   - Copy, do not move: every manifest hashes the staging paths, so `dataprep_staging/` stays
     byte-identical as the provenance record.
   - Fix the lint findings that were left in hashed files (P1, N1, W2W4 `rain_check.py`, H1a) in
     the ported copies only.
3. [open] **Sanitise before committing `dataprep_staging/` (D12).** Remove the home path or user
   name from these files:
   - `h2_heads_fill/config.py` (hashed by H2's manifests: the checker will then flag those
     versions; the registered bytes are archived);
   - `p1_county_electricity/r3_report.py`;
   - `p2_census_repair/v3/r3_report.py`;
   - `p6_public_wells/report_text.py`.

   Also:
   - S4a's `notes_for_s4b.py` holds a typed coordinate band; replacing it needs
     `insar_licsar_subsets/v2`;
   - replace the scratchpad path in the plan's Appendix A;
   - fix the TPC SOURCE.json key;
   - then run the Rule 1 scan.
4. [open] **I1 hooks** (plan I1; path and sha256 of every input recorded in the run metadata):
   - **Census source.** `--pump-census` / `--pump-kwh` exist. Switch rungs that may become primary
     to C0 `pump_electricity/preorigin/v3` (or keep model_census/v1 with its exemption and run
     the sensitivity). Declare the domestic-excluded member.
   - **`--electricity-product`** (does not exist yet): P2abc `yunlin_electricity/preorigin/v2`
     arms, recorded or Chow-Lin. The synthetic post-2016 members stay out (P2d gate failed).
   - **Canal v4.** `--sw-recharge` exists; it reads `surface_water_v4/model_npz/v1`, built from W1
     `preorigin/v2` deliveries plus v3 percolation, with `fit_window_end` 2024-12 through W1's
     LUIMAP exemption. v4 sits below v3's level (1.615 × and 1.122 × lower), and `value_v3level`
     gives a timing-only A/B.
   - **`--heads-product`** (does not exist yet): H2 `L2_filled_preorigin/v4` targets (flags 0-2,
     U2 weights with the pre-origin ρ(k)). Adopt the GP initial-condition field only after
     comparing it with the stage-3 construction on I1's own pre-origin burn-in.
   - **`--well-set`** (does not exist yet): H3 `recovered_wells/v3`, rung R5 only, with a
     per-segment datum member. It needs a waiver first (plan acceptance failed).
   - **`--mask-months START:END`** (done, commit d3b365d): for N1 G1/C3 (masked 2015, 2020 and 2021
     refits) and the fresh 2016-18 block.
   - **Coordinate fix.** `heads._station_xy` reads H1a's repair table (swapped "y x" stations; the
     9 stations at "0 0" stay missing). This gives a new well set, so a new baseline.
   - **Burn-in or spin-up.** A 24-month burn-in in the loss and diagnostics, or a 2007-11 spin-up
     with metered Changhua pumping (P4m).
   - **Changhua in the forward chain.** Changhua classes in `forward` and `scenario`, so the
     policy levers act there, and a Changhua-aware `forcing.npz`, so the ring continuation job
     runs again (batches 9-11 skipped it).
   - **For rung R4:** free per-layer abstraction fractions and a lagged return flow.
   - **`--ground-elev dtm`:** only after a W3 decision.
   - **Ring loaders.** `subsidence.mlcw_compaction` uses the deepest ring, which is no longer read
     after the gap. `mechanism.band_compaction` carries the step. One agreed session fixes both,
     coordinated with ring_holdout (plan 5.5). `rings_bridged/v2` stays after-the-fact only.
   - **Secondary QC.** L1 v3 flags in `challenge.monthly_observations` and the prospective scorer,
     reported as secondary scores only.
   - **Product access.** Every data product is opened through `guard.open_product` with a
     declared use.
   - **Not needed:** the H4 observation hook (H4 failed).
5. [open] **Tests** (plan I1): the sparse operator equals the matvec; the adjoint equals a
   forward impulse; census-path and extra-input bit-identity when absent; loader use guards,
   including `fit_window_end` and the time-slice encodings; the ported harness tests.

**Step 3. New baseline R0'**

- [open] Rerun batch 8's merged cwa recipe with the coordinate fix (new well set).
- Re-base the R1-family comparisons on R0'; R0, R1, R1b and R1c were scored against the old R0.
- Declare per rung either the plan's binding tests (temporal screen `--holdout-months 36` within
  1 %, leveling out of fold, CPU held-out-site proxy) or why the full k-fold replaces them.

**Step 4. Rebuild P4.0 on R0' and on R\***

- [done] P4.0 is built on the merged θ: engine/v2 replicates batch 8's merged θ
  (sha256 789d0e45…).
- [open] Build engine v3 on R0', and again on R\*.
- Rerun the P4m diagnostics on the R1-family θ (s_LS in [0.8, 1.25]; E10 not significantly
  negative under OLS and GLS).
- Use stepwise rollouts for added Changhua load, `engine_tlm` for forcing variance, and a CG
  tolerance of 1e-12 for RIV checks.
- E1 and E3 are superseded, so there is nothing to rerun there.

**Step 5. Ladder**, each rung pre-registered before its scores

| Rung | Status |
|---|---|
| R1 (+ Changhua census) | [done] batch 9 failed 2/4 gates; R1b (+ canal v4, batch 10) failed the same 2 |
| R1c (county zones) | [running] |
| R2 (+ Chow-Lin electricity arm) | [open]; read as a timing sensitivity (no placebo-controlled gain) |
| R3 (+ sw v4, pre-origin) | [open] on R0'. R1b tested canal v4 on top of R1. N1's G3 test starts at R3 |
| R4 (on R3: per-layer fractions and lagged return flow) | [open]; P2f test: model Yunlin local-only L1/L2 inside [0.52, 0.95] (cwa model −0.02) |
| R5 (+ recovered wells) | [decision] waiver needed (H3 3/15) |
| R6 (+ deep public-well class) | only if R-i data arrive (not drafted); the P6 member is a sensitivity |
| Old R1 (DTM ground elevation) | [decision] W3 |
| R\* | every rung that passed: fit + 5 folds + temporal screen; the full k-fold gate binds only here |

The 2023-25 forcing experiment and every 2023-26 subsidence score are reported, never gating
(U7).

**Step 6. After R\***

- [open] Refit the column on leveling per zone; the column pairs with its head field.
- [open] Propagate forcing uncertainty (plan 4.8, P40 `engine_tlm`). A0 or its successor must
  first add a use for it.
- [done 2026-10-23] Run N1's registered Phase 1-2 tests: G1 model better than CL and ZR; G3 not run (no delivery term); G4 not evaluable (erratum). C3 and C5 were never registered.
- [open] Pre-register one primary model before any 2023-26 re-score, then apply the I1
  re-scoring protocol (steps 0-7).
- [done 2026-10-23] Update the evidence doc (section 12) and the model card, with Changhua and
  Yunlin reported separately.
- P4.3/P4.4 as written are superseded. The R1-family decision takes their place.

**Step 7. Harness prerequisites** (before any Phase-1 test on fresh units)

- [open] A1 repair round 3: prereg3 must enforce the look rule for external units, and gate-3
  needs a pooling helper.
- [open] Agree a registration order across WPs.
- [decision] Rule whether non-head looks block fresh head units.

**Step 8. Release**

- [open] Build `release_20261012b`:
  - resolve the L3 verifier's three majors: waive or narrow the S2 printed piles and W1
    pre-origin under plan 2.8 item 2; drop or exempt the pile values published in 2023 and the
    reprint columns; drop the 13 stations established after 2022 from the pre-origin station
    table;
  - add the recovered wells if R5 is waived.
- [open] Make new versions where products change: C0 v4 relabelling the fill sd "uncalibrated";
  H1a v4 or a guard change so `events.csv` opens.
- [open] Reassemble `data_roles.csv`.

**Step 9. Housekeeping**

- Delete the large caches listed in section 8 by hand.
- Add SOURCE.json files for the pre-convention L0 directories.
- Close the plan-text corrections of 4.2 in a plan revision.

## 10. Completeness check: Phase-0 items no WP covered

| Plan item (section) | Status | Note |
|---|---|---|
| R1 data requests: "Send R1" in Phase 0 (3.0, 6.2) | not covered (user decision) | R-a is answered by the census. R-b, R-c, R-d, R-e, R-f, R-g and R-i are open. The Phase-1 failures point at conversion and response, which only R-b and R-g would measure |
| P4.1 E1-E7 and E11, P4.2 members, μ table (P4) | superseded | E8, E9 and E10 continue inside P4m. Three are still relevant and were not re-scoped: E2 (policy invariance under response variants), E3 (response-ridge identifiability, i.e. can heads identify a county-specific response at metered pumping, the question R1c asks) and E7. E7's chart (Changhua 0.80 × Yunlin Jan-May irrigation pumping) was never compared with the census, and the D6(a) Claude API approval for it went unused |
| P3 outputs: π maps, 200-member bootstrap, `phi_in.json` (P3) | superseded | Replaced by the census; P3 ran as a methods test |
| S4 validation rows: leveling 2020-21 withheld, rings/piles 2023-25 net of the GNSS excess, printed piles after 2025-05 (S4) | not covered | S4b ran G1, C1 and Y only. S1b covered the bridged-ring gap correlation. S5's T1 withheld 2020-21 leveling for the fused field, not for InSAR alone |
| S4 asc/desc decomposition from 2022-11 (S4 step 6) | blocked (data) | 105D has GeoTIFFs only for 2025-01..2026-01. 069A_06632 was not fetched (1.093 GB, beyond the 3 GB cap; user decision) |
| W1 step 6: forward archive of the WRA real-time discharge and weir feeds | blocked | Endpoints are unreachable from this VM. It needs another route and a decision on a recurring job |
| H3 probe R-h, credentialed vendor route | partial | Not run (permission). The user can run `dataprep_staging/h3_recovered_wells/r1_vendor_probe.py` with credentials in the environment |
| Plan 4.8: `forcing_uncertainty/v1` product | not covered | P40 built and verified the machinery (`engine_tlm`). There is no product and no use for it in `common/uses.py`. It needs the Phase-1 θ |
| H2 step 2 Kalman smoother with rain; step 6 deep-imputer benchmark | not covered (declared deviation) | GP only; the deep imputer is optional |
| P5 step 4: tidal T and S with an ocean-loading model | not covered | Amplitudes, phases and coherence only |
| P6: digitising the phase-2 plan maps of the pending Changhua Taiwan Water wells | not covered (noted by P6) | Would locate those wells better than the drought-well proxy |
| P1 extension validation: whole-season hold-outs, 2023-25 scored separately | not found in P1's report | P1 reports leave-one-year-out 2016-24 and 2016-22; P1 is now a validator |
| S2: monthly extension as WRA publishes | done once; not scheduled | No report newer than 2026-01 existed on 2026-10-12 |
| A0 task 2: SOURCE.json for pre-convention L0 directories | covered differently | `l0_registry/v1` hashes them; their owner should add SOURCE.json files |
| D12: commit staging code after the Rule 1 scan | not done | Waits for the user; sanitisation list in section 9, step 2.3 |
| Plan 2.1: L3 built only from L2 products | deviation | Three L1 upstreams (H1a L1 v3, S2 printed, ET0) |
| Phase 1-2 items (I1, P4.3/P4.4, N1 masked refits, P2d optional GPU ramp, H4 hook) | outside Phase 0 | See section 9; the H4 hook is not needed |

## 11. Resources and compliance

- **GPU.** None in any Phase-0 WP. The Phase-1 batches 9-11 are separate GPU work by the
  Phase-1 session.
- **Claude API, extraction only.**

  | WP | Input tokens | Output tokens |
  |---|---|---|
  | S2 round 2 | 263,527 | 56,309 |
  | N1 | 34,149 | 16,840 |
  | P6 | 26,864 | 13,510 |
  | **Total** | **324,540** | **86,659** |

  No physical value was filled by the API.
- **CPU budget.** Two disclosed slips: P3 ran a 2-minute job outside the wrapper and killed it,
  and W2W4 ran its first-round downloads and its network-bound fetch under nice 10 outside the
  wrapper.
- **Repository.**
  - No Phase-0 WP wrote under `hydrophysics/` or `tests/`.
  - `git status` shows only untracked `dataprep_staging/`, the plan, the audit and `.claude/`.
    This report adds one untracked docs file.
  - Nothing is staged or committed.
- **Privacy.** No product or report carries census names or addresses, or station contact
  fields. Per-pump tables are gitignored and declared as personal data.

## Repair round (2026-10-18)

Appended 2026-10-18. Sections 1-11 above are unchanged and describe the state before this round.
The round re-opened six WPs (W3, S3, S1a, S4b, S1b, S5) and rebuilt the L3 release. Every WP
followed the repair discipline: the diagnosis was written (and for most WPs hashed) before any
method change, thresholds were never changed, and re-tests on spent units are labelled "second
look", which cannot open a gate. Each WP was then verified independently. Machine-readable
version: `results/dataprep/reports/REPAIR_ROUND_2026-10-18.json`. Per-WP detail is in
`results/dataprep/reports/<WP>.repair.md` and `.json`. Every number below is quoted from those
reports, where the WP ledgered it.

**Outcome in one line.** No WP moved to passed. W3 stays failed, the other five stay partial, no
new product version is releasable, and `release_20261012b` corrects the three release majors but
its verifier left two majors open.

### R.1 Per-WP results

| WP | Final status | Verifier |
|---|---|---|
| W3 | failed (no new version) | confirmed (minors only) |
| S3 | partial (open issues) | issues: 1 new major (pre-existing QC gap) |
| S1a | partial | confirmed (minors only) |
| S4b | partial | issues (minors only, no major) |
| S1b | partial | confirmed (minors only) |
| S5 | partial | confirmed (minors only) |

#### W3 (DTM, `ground_elevation`)

- **Diagnosis.** The registered test compares a 1 km cell mean of bare ground with leveling
  benchmark marks. About half of the −0.842 m is that aggregation (−0.42 m, a DTM-only quantity
  that W3's own pre-registration predicted for a perfect DTM). The remaining point residual of
  about −0.38 m depends on site type (bridges −0.61, schools −0.27 m). The verifier's wording
  applies: it cannot be split between sub-20 m siting of the marks and a DTM-to-WRA height-system
  offset.
- **Repair.** None to the product: no defect was diagnosed, so no method changed and no v4 was
  built. A corrected test (agency GroundHeight at wells surveyed 2005-2022, same 0.30 m
  threshold, ≥ 20 sites) was designed and declared (`W3r.corrected_test_design.v1`) before any
  residual.
- **Re-test.** None evaluable. Only 2 sites (3 wells) qualify, all already looked at, and 0 fresh
  sites exist. The older GroundHeight values have an undocumented survey date, and leveling puts
  a median 0.328 m of subsidence between the two candidate dates (56 % of stations above
  0.30 m), more than the threshold. The registered test stands at −0.842 m (90 %
  −0.908..−0.783) against |median| ≤ 0.30 m: failed. A declared after-the-fact analysis (69
  sites, never gating) gives DTM minus well ground −0.046 m if surveyed at establishment and
  −0.362 m (90 % −0.582..−0.163) if surveyed at the 2009-2011 modification.
- **Final status.** Failed. `ground_elevation/v3` stays excluded from every gated use; the twin
  keeps SRTM. Verifier minors: the diagnosis claimed more attribution than the evidence shows;
  some metadata was computed before the diagnosis was written (the "before the first new
  computation" row is false as worded, rule (a) still holds); the snapshot hash proves content,
  not the 17:14Z time; 197 never-compared benchmarks could have given a fresh, non-gating
  replication of the site-type pattern.
- **Data needed.** A fresh bare-ground reference in TWVD2001 with known survey epoch at ≥ 20
  sites: WRA survey dates and datum for GroundHeight, a new leveling of ground at post-2005 wells,
  NLSC LiDAR ground check points or accuracy reports, benchmark point-of-record descriptions
  (monument type and mark height), or e-GNSS/ICESat-2 heights. Items need a data request or an
  account.

#### S3 (GNSS, `gnss_clean`)

- **Diagnosis.** The verifier's major was real and wider than stated: v3's WRA detection common
  mode was the median over whichever of 5 neighbours reported, so it jumped by up to 54.7 mm when
  neighbours stopped or resumed (mainly the January 2023 outage). In full/v3 this made
  common-mode-only steps and lost flag-0/1 days at ANES, HLES, HNES and YCES, and made the S3r
  A3 and twin comparisons compare different station sets.
- **Repair.** R7': absent neighbours are filled from pairwise neighbour offsets, so the common
  mode no longer depends on who reports (approximately; the verifier notes it is not exact). A
  first fix (R7) failed its own smoke check and was withdrawn. New versions
  `gnss_clean/preorigin/v4` and `gnss_clean/full/v4` (full/v4 replaces full/v3, which carries the
  artefacts). The v3 A3 and twin comparisons are withdrawn and redone on matched station sets.
- **Re-test.** No fresh units remain; no new registered test. Second looks on spent seed-707
  units: A1 WRA 0.900 (v3 registered 0.897) against ≥ 0.90, conditional, no evidential weight;
  A1 NGL 0.8729, identical to v3, fails; A2 same sign 0.857, pooled ratio 0.746 / 1.039,
  conditional. The verifier re-ran all 1,480 trials and reproduced them exactly. Flag-0/1 days in
  2023-25 return to v2 levels (HLES 739, ANES 430, HNES 707, YCES 775). Twin R² on the 12 common
  stations: −2.74 (v2), −3.06 (v3), −2.53 (v4).
- **Final status.** Partial (open issues). Gate failed on fresh units in both families. New
  verifier major, pre-existing and not introduced by this round: metre-scale gross errors carry
  flag 0 in the daily tables (XPES 2022-08-17..21, residual about −1.08 m, all versions; HLES
  2025-01-21, +2.38 m, full/v4), because the 31-day running-median screen cannot reject a lone
  sample inside a data gap. Monthly products, A2, A3, twin rates and the gate are unaffected;
  daily users through the guard are not.
- **Data needed.** New GNSS backgrounds never touched by design or test; a noise-model (GLS,
  power-law) step-size study; lower-noise re-processed solutions; WRA equipment and monument logs
  (KTES, TKJS, GFES, SJES); or a user ruling on a tolerance matching the measured size precision,
  as a new registration. A fix for the isolated-sample screen is needed before daily use.

#### S1a (ring bridge, `rings_bridged`)

- **Diagnosis.** The verifier major was confirmed by re-execution: forcing changes window RMSEs
  by up to 0.275 cm through pool size. The worst-window failures came from b_deep errors that
  grow as 1/deep-section length, short deep sections giving confident wrong group votes, a
  miscalibrated Xiutan pile line and pools holding assigned sites only. The verifier notes that
  the "2017 dry-season pulse" attribution of the failing windows is not cleanly supported (failing
  starts are 2016-10..2017-06 plus 2017-08); the repairs do not depend on it.
- **Repair.** Five changes (length-aware b_deep sd, pile line out of membership, empirical-Bayes
  prior, posterior-threshold pools independent of forcing, common-mode sd), registered with a
  fresh test. Product `rings_bridged/v3`.
- **Re-test (fresh units).** `S1a.e3.pre2012_r3_groupcond` on 16 pre-2012 windows never used by
  an S1a test, with new seeds: worst-window RMSE 3.23 cm against ≤ 1.5 cm (11 of 16 above):
  failed. Registered method on the same draws 2.94 cm. Second look on the 42 spent windows:
  2.08 cm (registered 2.79 cm), conditional, still failing. Check (a): power 0.561 registered,
  0.748 corrected (second look), against ≥ 0.8: failed. Check (b) 5/5 and residual offset 13/14
  pass.
- **Final status.** Partial; all versions after_the_fact_analysis only. Verifier minors: five
  unscreened 9999 sentinels in pre-2012 L0 surveys (no conclusion changes; the reported-only
  stress worst window moves from 4.08 to 4.84 cm), the pre-2012 surveys were partly exposed to a
  pilot script (not a WP-level freshness violation), and check (b) "shows" a 2022 anomaly should
  read "is consistent with".
- **Data needed.** The WRA record of the 2022 instrument change with scale calibration and zero
  offset (plan R-d), ideally an overlap survey; an independent across-2022 compaction record at
  more ring sites; or a non-exposed pseudo-gap era with long deep sections (post-2023 windows
  need a U7 ruling).

#### S4b (InSAR, `insar_069A`)

- **Diagnosis** (hashed before method code). The largest term was network gaps from a
  processing slip: LiCSBAS step 12 ran without multi-prime, so its first loop pass removed 163 of
  164 interferograms, including every one spanning the 2022 gaps. Other terms: a stable-strip
  reference plane extrapolated 17-46 km west, atmosphere about 4 mm, floors about 3 mm.
- **Repair.** Multi-prime loop masking with per-pixel nullification (no network gap remains;
  bridged values 51.2 % → 3.64 %), LiCSBAS16 filter at defaults, an off-fan offset + height
  reference with no plane, and the approved second ascending frame 069A_06632_162010 (765 files,
  1134.1 MB, new L0 with SOURCE.json), merged from 2022-11 (Z3 coverage 0.64 → 0.99). Product
  `insar_069A/v3`.
- **Re-test.** Fresh units, registered (`S4b.insar_repair.v1`): G1 on the NGL family gave
  7.28 mm but over 4 evaluable stations, against ≤ 10.0 mm with ≥ 5 stations: failed (HUWE
  pixels masked by coherence, LUKN outside the frame). Second looks: WRA G1 17.66 mm (fail), Y2
  0/8 (fail). C1 Z2 stays 0.297 (physical summer decorrelation, not re-tested). After the fact,
  the reference term flips between windows, so no reference chosen in advance is shown to stay
  within 10 mm in both windows on the NGL network (on WRA alone the old plane gives 8.43 / 7.25
  mm).
- **Final status.** Partial; verifier verdict "issues" with minor findings only. Main minors: the
  registration's statement that NGL was never compared with InSAR is false at project level (S1b
  and S5 did; not a rule-(d) violation within S4b); the v3 manifest omits the LiCSBAS15 `-i 500`
  amendment from its recorded parameters; report-only rows that wrongly kept GFES carry no
  contamination flag in the JSON and ledger; HUWE's exclusion depends on the coh_avg threshold,
  so any later change admitting HUWE would be tuning against the failed test.
- **Data needed.** A ramp reference spanning the fan (a plan decision making one GNSS family the
  calibration network, or GACOS/ERA5 or ionospheric corrections, which need accounts); fresh
  validation units with ≥ 5 stations (more on-fan GNSS, the 2020-21 leveling row after an A1
  ruling, or new GNSS for a prospective test); 6-day pairs or L-band for C1 Z2.

#### S1b (bridged-ring validation, no product)

- **Diagnosis.** The verifier major was confirmed: r3's "8/9 with both GNSS artefacts removed"
  removed only the effects that put the product outside the bracket and kept v4 step decisions
  that put Tuku inside.
- **Repair.** A symmetric step-policy grid (GNSS v2/v4 × three policies), declared before it was
  computed (`S1b.r3f.step_policy_symmetry.v1`) with the rule that the r3 statement stands only if
  all six cells pass. Errata for the minors.
- **Re-test.** None fresh (the 2022 gap happens once); all after the fact on spent units. Range
  ±30/40 ppm: 6-8/9 across the grid, family rule (8 of 9) passing in 1 of 6 cells; 7/9 under
  catalogue-only steps on both GNSS versions. The r3 statement is withdrawn. Plan-literal k0
  stays 5/10 against 8/10: failed. S1b's own k = 2 rule 9/10 is a second look.
- **Final status.** Partial. Verifier minors only (a bracket misquote that changes no count;
  "detector-free" should read "catalogue-only").
- **Data needed.** The WRA 2022 probe or cable change record; WRA GNSS equipment logs (KTES,
  SJES, GFES, and TKJS July-August 2023); an InSAR reference validated on fresh units; ≥ 3
  continuous GNSS within 5 km of ring sites; a user ruling on which reading governs.

#### S5 (fused subsidence, `subsidence_fused`)

- **Diagnosis.** Both verifier majors confirmed. T1F was not fresh: its 2018-19 surveys were
  scored targets of round-1 T2, training data of round-1 T1 and inputs to the round-1 campaign
  diagnostic. The reading of T2's over-coverage as a design artefact was wrong.
- **Repair.** Labels only: T1F relabelled "design-exposed, second look", the T2 reading withdrawn,
  T2b to stand beside T2 rather than replace it. No method change, because recalibrating the sd
  would tune it against T2 and no fresh units exist. Product `subsidence_fused/v5` (v4 values,
  corrected labels).
- **Re-test.** None. Only 28 of 7,924 leveling surveys were never scored, all removed by
  screening. T1F ratio 0.939, coverage 0.903: pass, conditional (second look). T2 coverage 0.948
  over-covers: failed. T2b 0.920: conditional pass. After-the-fact distance bins show T2
  over-covers at all distances (0.938 / 0.950 / 0.991) with heavy-tailed z.
- **Final status.** Partial; gate failed on T2, 0 passes on fresh units, after_the_fact_analysis
  only. Verifier minors: the "sd grows with distance" reading is confounded by subsidence rate
  (the core-versus-periphery contrast), and the JSON's test count (6) is wrong (20 pass).
- **Data needed.** Fresh leveling (the 112年 per-benchmark appendix and later years, request
  R-e); then a pre-registered heavy-tailed error model; atmospheric corrections or a pre-2025
  descending track; TKJS/GFES equipment logs; a user ruling to keep T2 and T2b as two rows.

### R.2 `release_20261012b`

- **What changed.** Built under one written inclusion rule (rule R: a product enters only if its
  own registered gate passed with hashes reproduced and every plan acceptance criterion assigned
  to it passed). Against `release_20261012`, only the pre-origin pile table and the pre-origin
  station table changed:
  - S2 printed piles stay in, with pre-origin uses narrowed to design_time_evidence, validator,
    scoring_target and after_the_fact_analysis (calibration_target, datum_fit and baseline
    removed). H3 stays excluded.
  - Pre-origin piles are dated by the agency listing's publication month: 35 primary values
    published after 2022-12 dropped and 49 second printings published in 2023 removed; 231 of
    266 rows kept, values and flags identical.
  - 13 stations established after 2022-12 dropped from the pre-origin station table.
  - The W1 basis text now states the failed head check.
  - No repair-round product was added (none passed); every new version is listed as excluded.
- **Build.** 17 components plus the grid from 13 upstream versions; two builds from scratch
  identical (77/77 files); 18 manifests pass the checker; 306 guard probes, 0 mismatches.
  `release_20261012` is unchanged. Four upstreams (C0 `pump_electricity/{preorigin,full}/v3`, H2
  `heads/L2_filled_{preorigin/v4,full/v2}`) no longer pass the shared checker against the
  current code (commit 8d51b46 and the D12 sanitisation); the release builder accepts them only
  where the changed file matches the recorded hash at commit 709f660 or an archived copy
  (amendment 1, declared before the rebuild).
- **Verifier verdict: issues, 2 majors open.**
  1. W1 `surface_water_v4/preorigin/v2` is still included, but the plan says the head-anomaly
     check "must not get worse" while marking only the crop-season-share check as not gated, so
     rule R appears to exclude it. Exclude it, or the user rules explicitly on whether that check
     is gated.
  2. Pre-origin printed piles still grant design_time_evidence, which lets 2019-2021 pile months
     inform twin design choices; the plan's S2 role says "never fitted, never in a gate or
     choice".
  Minors: amendment 1 cannot be audited by diff (v1 bytes not archived); C0 and H2 are refused by
  any other consumer until re-versioned or the guard learns the same pinning; supersession exists
  only in the new release (the old release and `deep_benchmarks_printed/v2` still grant fitting
  uses through the guard); two aborted-attempt logs under gitignored `results/` hold absolute
  paths and must not be committed.
- **User decisions.** Whether W1 pre-origin stays in; whether design_time_evidence is allowed for
  the printed piles; which publication date counts for the 2022 pile reports (the server copies
  are all dated 2023-05-11; using that date would drop 53 more pre-origin rows in a
  release_20261012c).

### R.3 Items for the harness owner

- Verifier looks not recorded in `preregistration/looks.jsonl`: the S1a verifier read pre-2012
  L0 ring values after scoring; the S4b verifier made post-registration looks on spent units
  (2026-10-17 22:47-23:03Z); the S1b verifier read grid outputs and GNSS step rows. Each should
  be logged on their behalf.
- The S4b registration needs a correcting disclosure note on NGL exposure, and `insar_069A/v3`
  needs a disclosure note (or v4) recording the LiCSBAS15 `-i 500` amendment.
- The S3 daily gross-error screen should be fixed before any daily use of `gnss_clean`.
- Resources: CPU only through the shared slots; no GPU; 0 Claude API tokens; one approved
  download (S4b second frame); nothing staged or committed; no file under `hydrophysics/` or
  `tests/` changed.

## Release b2 (2026-10-18): user rulings

`results/dataprep/model_ready/release_20261012b2/` is `release_20261012b` with two user rulings
applied. Release b's verifier had left two majors and three user decisions open. The user ruled
on two of them. Release b is not modified.

- **Ruling (a): W1 pre-origin stays included.** `surface_water_v4/preorigin/v2` stays in. This
  is a ruling, not a gate pass. The basis given: the ladder rung R1b (R1 plus W1 canal
  recharge) did not make heads worse in either county. Out-of-fold anomaly RMSE went from
  1.290 m to 1.264 m in Changhua and from 1.848 m to 1.777 m in Yunlin, on R0's design-time
  folds. The canal rungs R1b, R1d-sw and R1e-cn use this product. Context: this compares R1b
  with R1. Against R0 (1.074 m), R1b still fails its own Changhua head gate. W1's supporting
  head check still failed in Yunlin layer 1 (-0.0087) and was not re-scored.
- **Ruling (b): pre-origin printed piles narrowed.** The pre-origin pile component now allows
  only validator and after_the_fact_analysis, following the plan's S2 role ("never in a gate
  or choice"). design_time_evidence and scoring_target are dropped. The builder now reads the
  upstream table as validator instead of design_time_evidence. Both uses keep flags 0-6, so the
  same 231 rows come through and the table is byte-identical to release b's. The full-variant
  piles are unchanged.
- **Where the rulings are recorded.** `RELEASE.json` (`user_rulings`, `open_user_decisions`),
  `PRODUCT_CARD.md` (two new sections), and the notes, manifest parameters and
  `validation.json` of `preorigin/surface_water` and `preorigin/piles_printed`. The definition
  was declared before the first build: `preregistration/L3.release_20261012b2.definition.v1`
  (sha256 68ca545c…). One build-time check was added: every read must use one of its
  component's allowed uses. No threshold, gate, upstream version or product changed.
- **What differs from release b.** 77 files in each release. 33 are byte-identical: every
  table, csv and byte-copied array. 44 differ:
  - `RELEASE.json` and `PRODUCT_CARD.md`: the rulings, the open decision, the new name.
  - 18 `MANIFEST.json` and 18 `validation.json`: the release name, the code hashes of the new
    package `dataprep_staging/l3_release_b2/` and the git commit at build time. Only the two
    components above have content changes (uses, the read's use, notes, the ruling record).
  - 6 re-packed npz files: equal to b's array by array; only the `release_component` key, which
    holds the release name, differs.
- **Rebuild check.** Two builds from scratch were identical, 77 of 77 files
  (`REBUILD_CHECK.json`). The final tree equals build A.
- **Verifier verdict: passed.** `l3_release_b2.verify check`: 18 manifests valid, every hash
  reproduced, 306 guard probes with 0 mismatches. The shared checker passes all 18 manifests
  (385 files re-hashed) in b2 and in b. The comparison with b (`l3_release_b2.regress`) passed:
  no unexpected difference, and the pile component refuses design_time_evidence and
  scoring_target. Inventory: 13 upstream versions included, 81 excluded, 0 unclassified.
  Ledger: 171 new L3 rows, 0 conflicts. Unit tests: 11 passed; ruff clean.
- **Not fixed by b2.** The guard still grants the old pile uses through release b,
  `release_20261012` and `deep_benchmarks_printed/v2` itself; the narrowing exists only in b2.
  Release b's other verifier minors stand (amendment 1 not auditable by diff; C0 and H2 pinned
  to commit 709f660; two aborted-attempt logs with absolute paths under `results/`).
- **Open user decision (c), not ruled on.** Which publication date counts for the 2022 pile
  reports. b2 keeps release b's rule: the agency listing's publication month (發布年月). The
  server copies of every 2022 report are dated 2023-05-11. Using that date would drop 53 more
  pre-origin pile rows (data months 2022-04 to 2022-11, leaving 178 rows that end in 2021-11)
  and 45 more second printings. That would need a new release.

## Phase 1 ladder results (2026-10-19)

Every rung was declared before its run (A1 harness, `results/dataprep/preregistration/I1.*`) and
scored on R0's design-time folds. The four gates are fixed by R0:
- k-fold out-of-fold R2 at least 0.633;
- Changhua out-of-fold head-anomaly RMSE at most 1.074 m (74 wells);
- Yunlin RMSE at most 1.893 m;
- leveling column out-of-fold R2 at least 0.629.

A pass would then need a fresh-partition confirmation. **No rung passed. R0 stays the deliverable.**

| Rung | What it adds | k-fold R2 | Changhua RMSE (m) | Yunlin RMSE (m) | Leveling R2 | Gates failed |
|---|---|---|---|---|---|---|
| R0 | registered model | 0.643 | 1.074 | 1.856 | 0.649 | none (reference) |
| R1 | Changhua + Chiayi pump census | 0.652 | 1.290 | 1.848 | 0.248 | Changhua, leveling |
| R1b | R1 + canal recharge (W1 v4) | 0.642 | 1.264 | 1.777 | 0.335 | Changhua, leveling |
| R1c | R1 + county-split mid zone and county column | 0.646 | 1.318 | 1.820 | 0.663 | Changhua |
| R1d-dom | R1c without the domestic class | 0.622 | 1.398 | 1.820 | 0.678 | k-fold, Changhua |
| R1d-sw | R1c + canal recharge | 0.613 | 1.381 | 1.760 | 0.684 | k-fold, Changhua |
| R1e-pw | R1c + public-supply wells (P6 member, fixed) | 0.638 | 1.322 | 1.785 | 0.621 | Changhua, leveling |
| R1e-cn | R1d-sw with canal fractions split by county | 0.597 | 1.419 | 1.756 | 0.641 | k-fold, Changhua |

**What the ladder shows.**
- **The census fixes the size of Changhua's seasonal swing.** The predicted-to-observed amplitude goes from R0's 0.39 to 1.07-1.40. R0 scores better only because a small prediction loses less when the correlation is low (about 0.45).
- **Most of Changhua's misfit is year to year.** 85 % of the Changhua head-anomaly variance is interannual (Yunlin 58 %). The interannual error is 1.2-1.7 m2 against 0.3-0.7 m2 seasonal. No rung moved the interannual correlation above 0.52, except canal water alone (R1b, 0.63). The best lag is 0 or +1 month, so timing is not the cause.
- **The public-supply wells barely change Changhua heads.** R1e-pw adds the 2011-2019 closure ramp (about 125 to 0 Mm3/yr, WRA) and the 129 pending Taiwan Water wells. The interannual correlation goes from 0.46 to 0.49, and the leveling column drops below its gate.
- **The model wants no canal recharge in Changhua.** R1e-cn put both Changhua fractions (deliveries and percolation) at their 0.01 lower bound. The rest of the fan took 0.13 of deliveries. Canal recharge improves Yunlin heads (1.756 m, the best of any rung) but not Changhua.
- **The domestic class is not the cause.** Removing it made Changhua worse (1.398 m).
- **The county-split column is the one lasting gain.** It raises Changhua leveling R2 from 0.02 (R0) to 0.18-0.29 and the overall out-of-fold leveling R2 to 0.64-0.68.

**Reading.** Every forcing available without new data has now been tried. None reproduces Changhua's year-to-year head changes; they change the amplitude, not the shape. What remains is either structural or needs data:
- structural: how pumping is split across layers and how much returns to the shallow aquifer in Changhua, both still shared with Yunlin;
- data: metered public-well volumes (request R-i), well screen depths, and the heads' own long-term drift.

Code for R1e: commit d32768c on the PR branch (at run time the uncommitted branch
`twin/phase1e-pw-canal`; `results/twin_runs/par14/code.diff`, sha 184867d7). Batches 14 and 15 ran 2026-10-18 15:09 to
2026-10-19 03:59 with no failed job.

## Fresh-partition comparison under gate set G2 (2026-10-20)

Declared before any fresh fold was run (`I1.G2.fresh_partition.v1`). Its gates were designed
after the design-time results, so they apply only to a partition never scored before: well
folds `--seed 2027 --fold-site-tol-m 200` and leveling site folds `--seed 2027`. The partition
is fresh; the data are the same 2012-2022 wells and months. Batches 16-18 ran 2026-10-19 11:12
to 2026-10-20 11:45 with no failed job. Scored by `dataprep_staging/i1/score_g2.py`
(`results/dataprep/reports/I1_G2_fresh_partition/report.json`).

| G2 gate (threshold against R0) | R0 | R1c | R1e-pw |
|---|---|---|---|
| k-fold R2 (>= R0 - 0.01) | 0.718 | 0.646 (fail) | 0.657 (fail) |
| Changhua median correlation (>= R0 - 0.02) | 0.423 | 0.437 | 0.474 |
| Yunlin median correlation (>= R0 - 0.02) | 0.738 | 0.735 | 0.769 |
| Changhua county-mean anomaly RMSE, m (<= R0 x 1.02) | 0.515 | 0.643 (fail) | 0.657 (fail) |
| Yunlin county-mean anomaly RMSE, m (<= R0 x 1.02) | 1.082 | 1.036 | 0.941 |
| Leveling out-of-fold R2 (>= R0 - 0.02) | 0.647 | 0.661 | 0.619 (fail) |

Reported, not gating: Changhua single-well RMSE 1.10 / 1.34 / 1.34 m; Changhua amplitude
0.42 / 1.06 / 1.15; Changhua leveling R2 0.007 / 0.175 / 0.163.

**Decision, by the declared rule: no rung passes.** The deliverable keeps R0's heads and takes
its subsidence (the VEP column) from R1c. R1c's leveling R2 (0.661) beats R0's (0.647), and
its Changhua leveling gain holds on fresh folds (0.175 against 0.007). Each output comes from
its own consistent model: the column pairs with its own head field.

**Notes.**
- **The k-fold score depends on the partition.** R0 rises from 0.643 (design folds) to 0.718
  here, while R1c stays at 0.646. Any single-partition k-fold comparison carries an
  uncertainty of that size.
- **Changhua stays unsolved in heads.** Every recipe with the census fits Changhua's
  county-mean anomaly worse than R0.
- **The public wells help Yunlin, not Changhua:** R1e-pw has the best Yunlin county-mean
  RMSE (0.941 m) and correlation (0.769). R1e-pw stays diagnostic in any case (P6 is a
  sensitivity member).

**User ruling (2026-10-21): the declared fallback is suspended; R0 stays the deliverable for
heads and subsidence.** R1c's column fits the 2012-2022 leveling slightly better, but its
forecast collapses after the origin:

| | R0 | R1c | "no subsidence" |
|---|---|---|---|
| Fan-mean subsidence 2023-2032 | 2.88 cm | 0.13 cm | – |
| Rest of the fan (mostly Yunlin), 2023-2032 | 2.91 cm | -1.59 cm (uplift) | – |
| Measured deep piles 2024-25, RMSE | 1.27 cm | 3.55 cm | 3.53 cm |
| Deep piles >= 200 m, R2 | 0.92 | -3.95 | -3.89 |

The R0 column of this table is the registered page run (`datum_gate_innov`, leveling column;
the sources are listed in `deviation.json`), not `stage3_datum_gate_cwa`, which gives 2.86 cm
and a deep-pile RMSE of 1.30 cm. 2.88 cm is the page's basis (December 2022 to December
2032, the leveling column's 36 runs); evidence section 1 explains the forward summary's
3.03 cm.

The piles kept compressing 3-4 cm; R1c predicts what a zero forecast predicts. The composite
would have shipped a subsidence forecast the measurements contradict.

This is a disclosed deviation: it uses post-origin, report-only evidence (U7), after the G2
scores were known (`results/dataprep/reports/I1_G2_fresh_partition/deviation.json`). Lesson for
the next declaration: a column gate on hindcast leveling alone is not enough; it needs a
forward-plausibility check, declared before it is scored.

**Consequence under U7.** The deep piles were used to choose between columns, so by the plan's
own rule R0's 2023-26 subsidence scores (deep piles R2 0.92, RMSE 1.27 cm) are now
selection-contaminated: they stay in the record but are no longer reported as skill. A
subsidence skill claim needs a new registered window.

## User rulings and registrations (2026-10-21)

- **U4 and statistical layers.** U4 protects the physics model's frozen and prospective
  forecasts. A separate statistical layer may use recorded post-origin drivers (manifest use
  `statistical_layer_history`). This covers the registered hybrid test
  `I1.hybrid_tf.prospective_2026-11.v1`.
- **T3 registered** on that ruling: `X1.hybrid_tf_tpc.prospective_2026-11.v1`. It is the
  hybrid layer with Taiwan Power county electricity (downscaled) as its electricity driver,
  scored one month ahead on heads from 2026-11 to 2027-10 against the AR(1) layer. All 17 code
  hashes were verified, and one frozen fit was made on heads 2012-02..2026-09 (parameters
  sha256 2563512a). The design expects a narrow pass in Changhua only.
- **water.gov.tw robots.txt.** It is ambiguous; the user ruled for the RFC 9309 reading, so
  fetching was allowed. The Taiwan Water yearbooks and the measured Changhua public-supply
  series (`public_supply_gw/v3`) stand.

Record: `results/dataprep/rulings/user_rulings_2026-10-21.json`.

## Rungs R4, R4p and Q0 (2026-10-22)

Declared as `I1.R4Q0.v1` and `I1.R4p.v1` before they ran. All runs use the seed-2027 partition.
Numbers: `results/dataprep/reports/I1_R4Q0/report.json`. **No rung passes; R0 stays the deliverable.**

| | R0 | R1c (base) | R4 (shares learned) | R4p (shares measured) | Q0 (cleaned R0) |
|---|---|---|---|---|---|
| k-fold R2 | 0.718 | 0.646 | 0.622 | 0.615 | 0.648 |
| Changhua correlation | 0.423 | 0.437 | 0.489 | 0.335 | 0.468 |
| Changhua county-mean RMSE (m) | 0.515 | 0.643 | 0.731 | 0.697 | 0.531 |
| Yunlin county-mean RMSE (m) | 1.082 | 1.036 | 1.012 | 1.047 | 0.980 |
| Leveling R2 | 0.647 | 0.661 | 0.608 | **0.686** | 0.653 |
| Temporal fair ratio (2020-22) | - | 1.085 | 1.108 | 1.082 | **1.066** |
| 2023-25 county-mean subsidence, rest of fan (cm/yr) | +0.11 | -0.53 | -0.40 | -0.35 | +0.17 |
| P2f Yunlin layer-1/2 ratio (observed 0.85, 0.52-0.95) | -0.02 | 0.02 | 0.20 | 1.01 | -0.01 |

**Findings:**

- **R4 fails all but one gate.** Its free fit sent 98.6 % of Changhua's pumping to aquifer 3, the opposite of the well census (84 % in aquifer 1). The optimiser uses depth to damp Changhua's response.
- **R4p uses the measured shares.** It moves the layer response the right way (0.02 to 1.01) but overshoots. It also gives the best leveling of any rung, while Changhua's correlation falls.
- **Q0 is a new baseline and not gated,** because its well set differs. It has the best temporal ratio and passes the plausibility check, and its Changhua single-well error is 1.008 m (R0 1.100).

**The uplift is a column artefact.** Every rung built on R1c forecasts uplift in the rest of the fan after 2022; R0 and Q0 do not. All models' forecast heads recover alike after the 2020-21 drought. The difference is the 6-zone county-split VEP column that R1c introduced: it responds elastically to the head recovery, where R0's 3-zone column keeps compacting. Future rungs should keep the 3-zone column, or constrain the split column, before any subsidence claim.

## Combined candidate R* (2026-10-22)

Declared as `I1.Rstar.v1` before it ran. Recipe: Q0's cleaned inputs + the Changhua census +
the measured layer shares (84/11/2.5/2.5 %) + lagged return flow + a learned Changhua
pump-efficiency factor, with R0's 3-zone column. Seed-2027 partition. Numbers:
`results/dataprep/reports/I1_Rstar/report.json`. **R* fails 5 of 7 gates; no seed-2029
confirmation is run; R0 stays the deliverable.**

| | Q0 | R* | gate | pass |
|---|---|---|---|---|
| k-fold R2 | 0.648 | 0.562 | >= 0.638 | no |
| Changhua / Yunlin correlation | 0.468 / 0.740 | 0.454 / 0.692 | >= 0.448 / 0.720 | yes / no |
| Changhua / Yunlin county-mean RMSE (m) | 0.531 / 0.980 | 0.855 / 1.111 | <= 0.542 / 1.000 | no / no |
| Leveling R2 | 0.653 | 0.365 | >= 0.633 | no |
| Temporal fair ratio (2020-22) | 1.066 | **1.053** (PASS) | <= 1.077 | yes |
| 2023-25 subsidence, Changhua / rest (cm/yr) | +0.27 / +0.17 | +0.79 / +0.13 | >= 0 | yes |
| P2f Yunlin layer-1/2 ratio (observed 0.85, 0.52-0.95) | -0.01 | 1.00 | inside | no |

**Findings:**

- **The efficiency factor went the wrong way: 1.55, not below 1.** Once 84 % of pumping is fixed in aquifer 1 with an almost instantaneous return (0.17 months), the fit raised Changhua's net abstraction. Changhua's simulated amplitude rose from 0.36 to 1.17 of observed, and its county-mean error from 0.53 to 0.85 m.
- **So Changhua's half response is not an energy-to-volume (efficiency) effect.** The model cannot absorb it by scaling pumping.
- **Leveling collapses** (0.65 to 0.37; Changhua -0.62), and the measured shares again overshoot the P2f layer response (1.00).
- **The time direction is fine:** R* has the best temporal ratio of any rung and passes the plausibility check. The failure is in out-of-fold spatial skill.
- **Next suspect, per the declared interpretation:** a source term the model does not hold. The most likely is canal water outside the Jiji supply (X4: W1 counts about 35 % of the Changhua irrigation office's water), which would add recharge and damp Changhua's response.

## N1 Phase 1-2 tests (2026-10-23)

Registration `N1.phase12_response.v1`. Each test was scored once. Numbers:
`results/dataprep/reports/N1_evidence/phase12_tests/`. The full write-up is in
`docs/TWIN_EVIDENCE_2026-10.md`, section 12.

- **G1, masked 2015 refit (batch 23, `stage3_datum_gate_m15`, 135 of 158 wells).**
  - Model better than CL: RMSE 1.490 against 2.062 m, 90 % interval of the difference -0.67 to -0.46.
  - Model better than ZR: 1.749 m, interval -0.32 to -0.20. ZR is informative (RMS 0.63 m).
  - By county: Changhua ratio 0.89, Yunlin 0.68.
  - Reported, not gating: the response slope is 2.13, the damping seen out of sample.
- **G3: not run.** R0 has no canal-delivery term.
- **G4: not evaluable.** No sensor has all five Jan-May 2022 months. `g4/erratum.json` corrects the written verdict.
- **Code.** `--mask-months` is commit d3b365d on the PR branch (at run time the uncommitted branch `twin/n1-mask-months`); the frozen diff is `results/twin_runs/par23/code.diff`.

## N1 policy-response tests (2026-10-23)

Registrations `N1.policy_response.v1` (sha 6c3a4e0c, steps A-C) and `N1.changhua_fix.v1`
(sha be85ab6b, fixes F1 and F2, step B's one exploratory follow-up). Each test was scored
once. The full write-up is in `docs/TWIN_EVIDENCE_2026-10.md`, section 13. Every attempt,
including the engineering re-runs, is listed in `results/dataprep/reports/N1_policy/SUMMARY.md`.
`SUMMARY.md` is a result file and is left as written. The declared secondaries and step A
corrections it omits are below and in evidence section 13; the code provenance is in
[`N1_CODE_PROVENANCE_2026-10.md`](N1_CODE_PROVENANCE_2026-10.md).

- **Models.** Yunlin is scored on R0 (`stage3_datum_gate_cwa`). Changhua is scored on R1c with a three-zone leveling column, because R0 has no Changhua pumping. The page's levers come from the registered run `datum_gate_innov` (theta `stage3_datum_gate`), which section 11 of the evidence doc found within noise of R0.
- **Heads: supported in both counties** on masked 2016-18 refits (a design-time window). Yunlin: RMSE 1.38 m against 2.21 (no change) and 2.29 m (reference-year pumping); observed-on-modelled slope 1.21 (1.06-1.36), so the modelled response is too small. Changhua (R1c): 1.07 against 1.19 and 1.21 m; slope 0.85 (0.63-1.05).
- **Subsidence, 1-3-year size: not supported.** On 2016-18 a constant rate partitions the leveling better than the columns (slopes 0.22 and 0.40). In 2007-10, across places, Yunlin sank more than modelled (observed +0.66 against +0.26 cm/yr per GWh/yr, inconclusive). The two disagree in direction, so the label is "not supported", with no direction attached. The column is fitted to the cumulative leveling record, not to a policy difference. In this model the avoided subsidence is an elastic response that arrives within about a year; its 1-3-year size is not supported by the registered tests, and whether a cut changes long-run sinking is untested.
- **Changhua levers: not supported.** Step B: observed -0.50 against the census model's +0.80 cm/yr per GWh/yr, below every model-world window. F1 failed its design gate (rest-of-fan out-of-fold leveling R2 0.601 < 0.655) and was not scored. F2 changed nothing (heads ratio 1.002 against R1c, leveling 1.014). No further Changhua fix may be scored on 2007-10 or 2016-18.
- **Step A.** A1 (on R0 and R1c): the leveling-fitted column compensates fan-wide (0.93 at k = 2) but not by zone (proximal Yunlin 0.75, mid Changhua 0.77). A2 was computed on the superseded `response_basis.npz` (2026-09-20, `2f21f81`), not on the page's or R0's basis, so its "ranking robust" is withdrawn as a statement about the page; A2's result is left as written. A2b (post hoc, model-only, not a test; below) shows what holds on the page's basis.
- **2010-12 leveling judge unit: not spent** (its pre-declared power rule failed for Changhua: share positive 0.889 against 0.95). It is unspent only by `looks.jsonl`, which never names it, and it is not a clean judge (science-16 of the 2026-10-24 gap audit):
  - the W3 looks of 2026-10-12 read leveling displacement 2011-01..2021-12 at the 1030 active-cell benchmarks, which overlap its 2011-12 surveys;
  - step B's 2008-10 campaigns include the May 2010 survey;
  - step B's one exploratory follow-up is spent (`N1.changhua_fix.v1`), so a Yunlin 2010-12 test would be a second follow-up, which the registered attempt policy forbids unless the user amends it; and it has no power check;
  - the R0 column is fitted on leveling from 2012.

**Declared secondaries** (reported, never gating; full table in evidence section 13):

| secondary | Yunlin | Changhua |
|---|---|---|
| step B, E10 coefficient, observed / model (cm/yr per GWh/yr) | +0.045 / +0.221 | -0.221 / +0.165 |
| step B, model hindcast fixed-effects E3 | +0.141 | +0.004 |
| step C, heads scored with the other county's model: ratio to CL, slope | 0.64, 1.16 (R1c) | 0.955, 0.77 (R0) |
| step C, leveling scored with the other county's model: ratio to CL, slope | 2.47, 0.18 (R1c) | 1.29, 0.10 (R0) |
| pooled observed-on-modelled slopes | declared, not computed | declared, not computed |

**Post-hoc model-only checks (2026-10-24).** Each is post hoc, model-only and not a test:
model outputs only, no observation read, nothing scored, no registration. Files:
`results/dataprep/reports/N1_gaps_2026-10-24/science-{04,05,10,13}/`; full write-up in
evidence section 13 ("Post-hoc model-only checks").
- **A2b** (`science-04/`; post hoc, model-only, not a test): the A2 procedure rerun unchanged reproduces `step_a/a2.json` exactly on A2's own basis. On the page's basis, the page payload and R0's basis the cm lever ranking holds in 100 % of draws at both starts (industry above other); the per-100-GWh/yr ranking is a non-robust near-tie (2.84-3.10 cm per 100 GWh/yr on the page, order kept in 3 % of draws on the page payload, 10-21 % on the raw bases); the irrigation lever's top-decile hot spots are fragile (Jaccard 5th percentile 0.00, median 0.71-0.72).
- **Transfer** (`science-05/transfer_innov_vs_cwa.json`; post hoc, model-only, not a test): the page's model and R0 avoid 0.942 and 0.958 cm (cut30) and 1.306 and 1.261 cm (retire_aqua) fan-mean by 2032, within 2-4 %; cell correlation 0.9999 and 0.9996.
- **Hot-spot stability** (`science-10/hotspot_stability.json`; post hoc, model-only, not a test): the cut30 leader Baozhong leads in 83 % of the page's runs, 72 % of R0's and 57 % of the A2b draws; retire_aqua's Taixi in 61 %, 83 % and 100 %. Neither is stable by the file's rule.
- **Elastic split** (`science-13/`; post hoc, model-only, not a test): under mean forcing on ensemble-mean heads no cell falls below its preconsolidation head after Jan 2026, so the avoided subsidence is all elastic (inelastic 0.000 cm) and 98-99 % of it arrives by Dec 2026. A dry year below the 0.24 m margin (5th percentile) is not in the runs.
- **Page labels from them**: `dataprep_staging/n1_policy/page_posthoc.py` copies these numbers into `page_posthoc.json` (same directory, written once), which `viewer_app --posthoc` (off by default) puts on the page, refused unless it names the page's forward run and basis.
- `science-15/` holds a draft registration only; it is not cited and nothing in it is registered.

**Re-running cited results: not from the sibling worktrees.** No cited result may be re-run
from the worktrees `HydroPhysicsAI-integration`, `-phase1`, `-phase1c`, `-r4p` or `-rstar`.
`integration` (branch `twin/r4-layer-allocation`), `phase1` and `phase1c` hold uncommitted
edits to twin modules (3, 2 and 12 files under `hydrophysics/twin/`, plus an untracked
module in `phase1c`) that match no recorded batch, except that `phase1` equals par9's diff;
`r4p` and `rstar` are clean but checked out at older commits (`5f44cd2`, `a3d6eed`) without
the later fixes. Re-run from the PR branch, or from a batch's recorded commit plus its
`code.diff` as tabled below. Archiving or removing those worktrees is the user's decision.

**Code for the N1 batches.** The batches recorded a diff without a base commit. The bases,
reconstructed by comparing each recorded diff restricted to `hydrophysics/` with a commit
diff:

| batch | used by | recorded `code.diff` | equivalent commits | on the PR branch |
|---|---|---|---|---|
| par23 | G1 refit (`_m15`) | af53ef55 | ccb6ff2..d3b365d | yes |
| par24 | step C refits and columns; step A1 and step B runs | 2618484a | c48e858..294ed2e | no (`twin/n1-policy-response`) |
| par25 | F1, F2 | 277d6a50 | c48e858..8a96b21 | no (`twin/n1-changhua-fix`) |

`--withhold-surveys`, `--column-zones 3` and `--return-rice-share` are therefore not on the PR
branch yet. Merging those two branches is a decision for the user. The scorers' usage lines
say `py.sh -I`, which drops `PYTHONPATH`; run them through `par24/py.sh` or `par25/py.sh`
without `-I`. The G1 and G4 scorers are not hash-bound by `N1.phase12_response.v1`. They
were written after it and are tied to the looks only by their file times and commit ba76923
(evidence section 12).

**Phase 1 rungs: which code reproduces them.** From the batch records and a reconstruction
of each recorded diff on its base:

| rung | batch | code record | reproducible from |
|---|---|---|---|
| R1 | par9 | base 709f660 + `code.diff` (sha c5abcfc0; no `code.diff.sha256`) | only that diff on 709f660. 8d51b46 refactored the code, and 69 lines of `calibrate_flow.py` differ. The phase1 worktree still holds the diff uncommitted |
| R1b | par10 | none (`py.sh` points at the phase1 worktree) | presumably par9's diff; unverified |
| R1c | par11 | base 709f660 + `code.diff` (308d2ce0) | 8d51b46: zero differing lines in `calibrate_flow`, `forward`, `inputs`, `zones`, `calibrate_coupled`, `scenario`, `uncertainty`, `policy_gate` and `residual_kriging`. `mechanism`, `ring_holdout` and `viewer_app` differ but no fit uses them. `zone_county.py` was untracked at run time and is not recorded (`code.status` lists it as untracked; `code.diff` omits it), so this is partly verified (see `TWIN_THETA_PROVENANCE_2026-10.md`) |
| R1d-dom, R1d-sw | par12, par13 | `jobs.sh` names 8d51b46; no `code.status` | 8d51b46, if the tree was clean |
| R1e-pw, R1e-cn | par14, par15 | base 8d51b46 + `code.diff` (184867d7) | d32768c (equal on `hydrophysics/`) |

R1c's theta (`stage3_datum_gate_ccz`) is a registered input of N1 steps A-C. Removing the
phase1 worktree would lose the only copy of R1's code outside `par9/code.diff`, so ask the
user first.

**Release-b2 products behind the cited tests.** Each manifest records commit 709f660. Its
code hashes compared with the PR head:

| product | used by | code files equal at the PR head |
|---|---|---|
| `heads/L1_clean/v3` | G1, step C heads (observations) | 48 of 48 |
| `pump_electricity/preorigin/v3` | `N1.policy_response.v1` input (step B exposure) | 15 of 16; `hydrophysics/twin/scenario.py` matches 709f660, not the PR head |
| `deep_benchmarks_printed/v2` | G4 | 13 of 13 |
| `surface_water_v4/preorigin/v2` | `N1.phase12_response.v1` input (G3, not run) | 15 of 15 |
| `n1_policy/rice_intensity/v1` | F2 | built by `n1_policy/build_rice_intensity.py`, hash-bound by `N1.changhua_fix.v1` and equal at the PR head |

All N1 and I1 code hashes match the PR head. Some older Phase 0 registrations do not,
because their code was edited after they were registered (H1a, H2, H3, the L3 b and b2
definitions, P2abc, S1a, S2, S3, S5, W3). That is expected, but it means the registered
versions exist only as their hashes and as archived copies. The b2 release definition
(`L3.release_20261012b2.definition.v1`) depends on
`results/dataprep/reports/H2_evidence/code_as_registered/015c1980819a/config.py`, a
gitignored copy outside git. Whether to commit a sanitised copy under a new path, after the
Rule 1 scan, is open for the user.
