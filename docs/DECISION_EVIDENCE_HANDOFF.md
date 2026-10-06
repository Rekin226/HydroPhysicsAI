# HydroPhysicsAI decision evidence handoff — 2026-10-06

## Executive verdict

**Research scenario explorer; no validated subsidence intervention recommendation.**
The selected model loses the saved, retrospectively scored head challenge to
seasonal climatology, and the saved column models disagree on lasting rate
benefit. The viewer can show conditional consequences, with provenance and
warnings. It cannot identify which pumping action will reduce observed future
subsidence. No code was pushed, merged or deployed.

This audit and bounded engineering are on a separate worktree (`HydroPhysicsAI-decision-evidence`),
branch `audit/decision-evidence`, through commit `fd37c76`. The original
main checkout remained clean, and its existing
Claude daemon and localhost preview were not interrupted. Public source
downloads and private evaluation outputs remain under ignored
`results/decision_evidence/`.

## What is measured today

| Evidence | Actual result | Interpretation |
|---|---|---|
| Frozen December 2022-origin head challenge through August 2026 | 158 wells, 6,186 well-months: datum-adjusted model RMSE **2.072 m**, seasonal climatology **1.667 m**; model loses in every horizon bucket and layer | Retrospectively scored after acquisition; not prospective. Source hashes, split and scores: [readiness](DECISION_READINESS.md), ignored `results/twin/new_data_challenge/` in original checkout. |
| Unseen-well spatial screen | Selected R² **0.643**, inverse-distance interpolation **0.702** | Spatial baseline not beaten. [Readiness](DECISION_READINESS.md) and existing calibration scorecards. |
| Two-site ring change proxy | 30 overlapping annual pairs: selected 1.528, pre-2023 historical-rate baseline 1.538 cm/year RMSE | The archived total column was compared with a specific ring interval; this is **not depth-matched skill**. [Protocol and scores](DECISION_READINESS.md). |
| Hypothetical pumping scenarios | Selected model: 2028–2032 rate reductions below 0.001 cm/year for both interventions. Previous model: 0.0793 irrigation and 0.0752 aquaculture cm/year. | Structural disagreement; neither action was observed or verified feasible. The 72 member paths per model are sensitivity samples, not calibrated probabilities. [Scenario table](DECISION_READINESS.md). |
| 土庫 GNSS surface lead | [Public WRA dataset 36694](https://data.gov.tw/dataset/36694) has 2,461 土庫 daily elevations through 2025-12-31. The current public data and local cache differ on overlapping values and contain abrupt site-specific steps. | New retrospective raw contrasts are **not a field-accuracy score**. See [site audit](SITE_GEOLOGY_SOURCE_AUDIT.md) and ignored `tuku_gnss_public_raw_contrast.json`. |

The GNSS script compares same-month, within-segment changes after re-zeroing
and excludes flagged and missing months. At the exploratory 2.5 cm flag
threshold, public-source selected-model raw RMSE is 1.17 versus zero 0.67 cm
for Aug–Dec 2023; 1.06 versus 1.29 for Feb–Oct 2024; 0.12 versus 0.82 for
Mar–May 2025; and 1.12 versus 0.45 for Aug–Dec 2025. These short, correlated,
post hoc segments give mixed results. The 2 cm threshold flags 30 dates,
2.5–3 cm flags eight, and 5 cm flags two; no threshold is selected by model
score. The official description calls the field elevation in metres but
does not specify its vertical height component or reference frame.
The model predicts one positive whole-column compaction value; GNSS elevation
also carries deeper, tectonic, reference and instrument motion. No correction
for those terms was invented. Tracked twin code does not directly read GNSS;
independence from human selection is unverified.

## Completed engineering versus remaining defect

- **Completed:** saved-evidence diagnostics, source-hashed 土庫 geometry and
  interval-overlap checks, a zero-right-hand-side adjoint fix, a four-layer
  head structural sensitivity script, and an observation-only GNSS QC and
  raw-contrast script. The GNSS script validates station identity, units
  contract, duplicate timestamps, finite values, gross metre-scale plausibility,
  monthly coverage, step
  boundaries and missing-month breaks. It always returns an unvalidated
  status. Code and tests: `hydrophysics/twin/site_geometry.py`,
  `tuku_sensitivity.py`, `gnss_surface_evidence.py`, plus tests.
- **Completed viewer safeguard:** `hydrophysics/twin/app/template.html` now
  labels scenario rankings as conditional and parameter agreement as sample
  sign agreement. An offline generated page using synthetic inputs is in
  ignored `results/decision_evidence/viewer_check/conditional_viewer.html`
  (SHA-256 `92db6ed78c7cdc68e67355a5b126418b1c6ab0a9821409887d500ffc785f53e4`).
  The running original preview is unchanged.
- **Model defect:** the operational VEP output is one scalar column per cell
  driven by a combined head. No stored depth profile can be integrated against
  a ring interval. A vertical constitutive model and tested depth operator
  remain to be implemented, with identifiable pressure and material inputs.
- **Model evidence defect:** the selected head model loses to simple temporal
  and spatial baselines, while plausible model structures produce conflicting
  later subsidence-rate responses. Retuning against already inspected outcomes
  would not repair this evidence.
- **External observation gaps:** 土庫 ring NO1 installed depth, NO25/NO26 API
  crosswalk, 48–51 m lithology conflict, pressure screen tops and bottoms,
  head/ring/GNSS vertical references, GNSS maintenance and revision history,
  deep reference motion, and measured pumping intervention records. Official
  [2003 WRA](https://web.wra.gov.tw/wralib/WraLib/wSite/books_41979) and
  [2017 IOT](https://www.iot.gov.tw/uploads/asset/data/661982b7367376304acd4ed0/B1068760.pdf)
  tables are preserved separately; no false single geology was chosen.
  The exact [unsent records request](UNSENT_WRA_RECORDS_REQUEST.md) is ready
  for owner review.

## Next experiments and decision gates

| Priority and dependency | Bounded next work | Acceptance test and resource estimate |
|---|---|---|
| 1. WRA station/reference package; external owner | Resolve GNSS height type, antenna/reference changes and version differences, plus ring maintenance and pressure screens. Freeze QC before new scoring. | Station crosswalk explains or explicitly bounds 2022-06-17, 2024-01-01 and 2024-11-02/03 changes and the 2025-02-01 5.55 cm discrepancy; independent datum and uncertainty documented. Desk review is days once records arrive; no compute expense. |
| 2. Depth-resolved one-site model; depends on item 1 and deeper material bounds | Implement explicit stress driver and strain history per surveyed interval; compare WRA confining-unit and IOT soil-class structures without forcing one contact. Use synthetic mass/strain-sum and elastic-limit tests before any fit. | Ring interval operator and total-column sum agree on synthetic columns; parameters recover on synthetic data; observed ring fit assessed only on pre-registered dates. Small geometry checks run about 2 ms per 24 intervals; full fit runtime is **unmeasured** and must be benchmarked on one CPU column before training. |
| 3. Fixed-origin forecast challenge; depends on registered data split | Compare heads and surface/ring rates with seasonal, persistence and site-rate baselines by site, layer, lead and event; lock preprocessing before evaluation. | Model beats strongest registered baseline on genuinely held-out periods with calibrated interval coverage. CPU evaluation is cheap relative to simulation; existing relevant tests ran 113 in 6.96 s. No training is authorized here. |
| 4. Intervention decision; depends on measured implementation | Obtain pumping volumes, spatial scope, timing, substitution supply and control sites. Compare immediate rebound separately from sustained rate change across model structures. | Observed treated-minus-control later rate is distinguishable from zero and robust to reasonable confounder/structure sensitivity. This requires field records and later measurements; no meaningful runtime estimate substitutes for them. |

The strongest supported public statement is that the app illustrates how two
*assumed* pumping changes propagate through its present model, with the
published evaluation limits visible. Claims of accurate future subsidence,
a superior intervention, protected infrastructure, uncertainty coverage or
causal policy benefit must wait.

## Reproduction and verification

Run `python -m hydrophysics.twin.gnss_surface_evidence --help` and
`python -m hydrophysics.twin.tuku_sensitivity --help` in the hydro
environment for bounded read-only diagnostics. The site audit gives exact
input paths and hashes. A broader CPU suite covering column, observation
boundary, leveling, forward, policy, evidence, geology, GNSS and viewer tests
passed **113 tests in 6.96 s**; the two known 800-epoch training tests were
excluded. Ruff, staged whitespace, and credential-pattern checks passed. No paid compute,
lengthy simulation, package install or external publication was used.
