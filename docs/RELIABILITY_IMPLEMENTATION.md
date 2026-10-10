# Reliability fixes following the repository audit

October 7, 2026. Implements the software reliability stage of
[the repository audit](REPOSITORY_DECISION_AUDIT.md) and
[the development roadmap](SUBSIDENCE_DECISION_ROADMAP.md).

## Implemented

| Problem | Change | Evidence |
|---|---|---|
| Forward heads were reused as adjoint initial guesses; zero right-hand sides could produce misleading convergence warnings | Integrated the audit worktree's zero-start adjoint and exact zero-RHS handling | Previously failing anomaly-loss and coupled-gradient cases now pass with convergence warnings treated as errors; added zero-RHS and finite-difference gradient regressions |
| Neural training targets crossed the holdout boundary | Require the entire supervised history/target window to belong to the allowed training period | Changing all held-out targets and forcing leaves fitted weights identical with the same seed |
| Neural inference required future measured heads | Separate inference history requirements from supervised target construction; allow future targets to be missing | Point and probabilistic forecasts at the last observed origin are unchanged when future targets become NaN; no valid history returns an unavailable forecast |
| Point scores silently used different rows | Apply common support across model/persistence/climatology; publish observed, available and scored counts | Synthetic missing-forecast case scores the same three rows for every method; probabilistic comparisons also exclude missing baselines consistently |
| An unrelated passing report could approve an update | Recompute the supported validation gate and match forward, code, observation, protocol and prediction hashes; retain accepted identities | Mismatched model/code/data/protocol/period/predictions/task, boolean-only reports and missing bindings are rejected; the previous release remains unchanged |
| Candidate page could change between checking and publication | Compare the expected hash against the exact page bytes being published | A synthetic page replacement after checking is rejected without changing the release pointer |
| Corrected observation updates were only available experimentally | New `forward.run`/CLI runs default to innovation updates; propagate the choice through hindcast, restart and perturbation handling | Routing regressions exercise both methods; existing perfect-observation and datum invariants remain |
| Viewer wording overstated evidence | Integrated the audit branch's literal sample-agreement labels, conditional rankings, total-column proxy and total-well-depth descriptions | Current template rendered against an archived research payload for desktop/mobile browser checks |

The earlier AMP portability fix remains in place: its tests explicitly skip when
SciPy is unavailable in the foundation installation.

The field-acquisition follow-up added conservative public-data imports and repaired
weather handling: date alignment for ET0, failure on a day with no ET0 source,
recorded spatial-imputation masks, validation of cached arrays and rejection of
missing weather in the climatic water-balance transform. See
[the acquisition and input-impact evidence](FIELD_DATA_ACQUISITION.md).
Its separate focused verification passed 32 tests; the earlier 572-test full-suite
result below predates these additions and is not a full-suite check of them.

## Changes callers must make

- To reproduce an old forward configuration, explicitly pass `--update-method field`.
  New runs use `innovation`, with the choice in saved `forward_options`. Direct
  low-level helpers retain their historical defaults; explicit arguments are preferred.
- To forecast at the last observed date with the neural model, provide the future
  date/forcing horizon with future **target heads left missing**. The model does not
  forecast its own rainfall/upstream inputs. Historical zero-fill of missing forcing
  remains; document supplied/imputed values and their availability at issuance.
- Update acceptance configuration from report filenames to entries naming `report`,
  `protocol` and `predictions`. Add code/observation hashes to the protocol and render
  with the matching validation input flags. See [the release contract](TWIN_RELEASE.md).
- Retrain affected LSTM checkpoints and reevaluate them; rescoring old weights alone
  cannot remove training leakage. Do not reuse their old skill or interval-coverage
  numbers as results of the repaired implementation. [The model card](../MODEL_CARD.md)
  marks those results as historical. Archived scientific predictions were not altered.

## Verification

The initial focused suite passed 56 tests; the two additional update-routing cases
also passed. The complete suite in the hydro environment passed **572 tests**, with
**7 skips and 20 warnings**, in 674.90 seconds (exit zero). CPU thread counts were
limited; available optional CUDA smoke tests also ran. All skips concern unavailable
real-data fixtures. The run emitted **no conjugate-gradient nonconvergence warnings**,
in contrast to the audited pre-fix run. Remaining warnings concern empty metric
summaries, a test tensor conversion, library deprecations and single-process setup.
Lint, CLI construction and diff whitespace checks passed. Full logs and source hashes
are recorded locally under `results/reliability_fixes/`.

The browser check passed at desktop and mobile sizes with no JavaScript errors or
remote requests, working language switching and a working no-WebGL fallback. It
used this project's cached Chromium executable because the installed Playwright's
default executable was absent; no new browser was installed. The rendered page uses
the archived research payload with the current template, not new scientific results.
Software-rendered timing does not certify interactive performance on a user's GPU.

## Scientific gates still outstanding

These repairs establish more reliable software and evidence handling. They do not
establish improved field subsidence accuracy, calibrated probabilities or causal
benefits of a pumping intervention. The release gate presently supports head
validation only; publication remains explicitly a local research artifact.

The next field-calibrated stage requires:

1. Exact, dated monitoring-well screen intervals and signed head units/datum;
   instrument references and stable ring/surface measurement definitions.
2. Paired electricity, discharge and total dynamic head measurements, plus historical
   meter/well mapping, to independently constrain the existing volume conversion.
3. A frozen subsidence target and evaluation protocol, untouched site/time records
   and action-specific error tolerance before ranking interventions.

The isolated audit's geometry/evidence modules and synthetic depth designs remain
available for the subsequent measurement-mapping stage; this change does not import
that entire experimental branch. Surrogate expansion and basin-wide refits remain
deferred as the audit recommends. No field metadata, passing scientific result or
winning scenario has been invented to bypass these dependencies.

No release was deployed or data request sent. Code and documentation remain local
and uncommitted. Verification jobs have finished.
