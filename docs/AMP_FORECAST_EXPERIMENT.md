# AMP forecast integration experiment

## Result — October 6, 2026

A bounded, retrospective three-well 土庫 pilot does **not** support adding AMP
to the deployed forecast. Original AMP, HGT/version two, and amplitude with
inferred operating duration all failed to improve the electricity-based comparator
on identical eligible rows. This is an initial integration experiment, not a
basin-wide rejection of AMP or a test of subsidence/intervention accuracy.

| One-month well forecast | Equal-well RMSE |
|---|---:|
| Seasonal climatology | 1.35046 |
| Previous-month persistence | 1.23402 |
| Seasonal persistence | **0.63712** |
| Seasonal-persistence correction using head history and electricity | 0.64721 |
| Same correction + original AMP | 0.64752 |
| Same correction + HGT AMP v2 | 0.64941 |
| Same correction + AMP and operating-duration proxy | 0.65665 |

Scores use **native head input units**. The audit's unresolved exact-well units,
screens and vertical references still apply; these numbers are not certified
metres or validated physical pumping estimates. They must not be compared directly
with the earlier 158-well, multiyear fixed-origin challenge: sites, dates, forecast
horizon and observation access differ.

Only **18 of 132 candidate test well-months** survived the fixed QC and matching
rules: three colocated wells in July–December 2023. There were 252 training rows
and 19 validation rows. This small, correlated subset cannot establish general
forecast skill. Duty was recoverable in seven test rows; the remaining rows used
the documented training-only imputation and missingness indicator.

## What was repaired and added

- The old monthly AMP exports filter full records, then aggregate. FFT, Hilbert
  transforms and symmetric smoothing let later values influence earlier features.
  Merely shifting those exports by one month does not remove that leakage.
- `AMP_V2/forecast_experiment.py` recomputes all transforms inside each origin's
  previous 180 days. QC and interpolation also see only that past window. It
  retains hourly timestamps, requires 95% hourly coverage, fills only interior
  gaps of at most six hours, and rejects remaining gaps. Features summarize a
  30-day envelope ending seven days before the origin to reduce endpoint effects.
- Long-record harmonic detector settings suppressed adjacent harmonics in the
  short window: the old 200-bin minimum peak distance exceeded the 180-bin
  spacing of daily harmonics. The new caller uses a 0.5-cycle/day local-background
  width expressed in window bins. Default full-record behavior is preserved.
  A three-harmonic synthetic test reproduces the defect and verifies duration
  recovery with the corrected setting.
- Operating duration requires a fitted ladder of at least three harmonics.
  The experiment does not substitute the historical AMP-G fallback of half-day
  operation when recovery fails. Amplitude times duty remains a signature,
  **not measured water volume**; recovery depends on the assumed pulse and
  attenuation model and the unresolved attribution of daily signals.
- `AMP_V2/forecast_energy.py` prepares local irrigation electricity with the
  existing shared-meter deduplication. It retains all meter siblings while
  allocating energy, freezes meter screening using pre-2021 data, and rejects
  months with less than 90% coverage of retained local meters. The pilot retained
  457 meters within one kilometre of the site. Static registry metadata and
  final-vintage histories remain retrospective inputs.

The first pilot, retained separately, used the long-record detector and recovered
no duty values. After the independently reproduced synthetic failure, only the
short-window detector was corrected; dates, QC, model features and tuning grid
were unchanged. The table above is the corrected run. This correction occurred
after seeing initial results, so neither run is a preregistered prospective trial.

## Evaluation contract

Training origins are January 2014–December 2020. Ridge strengths of 0.1, 1, 10 and
100 are selected separately per arm using January 2021–December 2022 only.
Test origins are January 2023–August 2026. The training seasonal means,
standardization, imputation and fitted coefficients stay frozen during testing.
Every comparator uses the same eligible rows; no test result selects a parameter.

The electricity correction contains previous-head seasonal anomaly, one-month
head change, seasonal sine/cosine, and log electricity. AMP arms add the indicated
features. This is a **statistical forecast at monitoring wells**, not a replacement
for the aquifer solver or its physical pumping forcing. In particular, the AMP
proxy is not added to electricity-derived abstraction as a second water volume.

Electricity is lagged by two calendar months. Actual release dates are unavailable:
this is an explicit availability assumption, not verified historical availability.
AMP uses measurements timestamped before the origin, but historical revision
times are also unverified. Monthly targets require 80% hourly coverage and are
never interpolated. New monthly measurements can inform later one-step origins;
this does not evaluate a multiyear forecast from one fixed origin.

Raw inputs, private per-well scores, predictions, source hashes and the protocol
remain in ignored local results. Each run requires a fresh output directory.
Eight focused tests cover future-value invariance, test-target independence of
model selection/predictions, harmonic recovery, gaps, target coverage, paired
scoring rows and conflicting overlapping source vintages.

## Reproduce from the repository root

Use the existing research environment with NumPy, pandas, SciPy, PyArrow and
PyTorch (the latter is imported by the existing electricity-cleaning module).
No GPU is used. Run longer jobs in tmux as required by the repository instructions.

```bash
python -m AMP_V2.forecast_energy \
  --census AMP_V2/data/tpc_pumps.parquet \
  --kwh AMP_V2/data/pump_kwh_all.parquet \
  --x 187777.4 --y 2620597.81 \
  --out results/amp_forecast/energy.csv

python -m AMP_V2.forecast_experiment \
  --historical AMP_V2/data/wells \
  --recent data_fetch_api/new_measurements/heads_202210_202610 \
  --energy results/amp_forecast/energy.csv \
  --well 09050321 --well 09050331 --well 09050341 \
  --out results/amp_forecast/reproduction

python -m pytest -q tests/test_amp_forecast.py
```

The corrected local artifacts are in `results/amp_forecast/tuku_pilot_corrected/`;
the original run is in `results/amp_forecast/tuku_pilot/`. Do not ship those raw
artifacts or interpret the software tests as field validation.

## Next evidence needed

Choose an expanded site set using record continuity and irrigation coverage,
before examining forecast scores. The strict continuous-window requirement lost
most recent 土庫 periods; investigate gaps and measurement availability rather
than deleting missing timestamps or relaxing QC to improve the score. Evaluate
any alternative window/gap method as a separate, frozen sensitivity experiment.
Verify publication lags and site units/reference, then reserve a new evaluation
period. Only a robust incremental gain over seasonal persistence and the same
electricity comparator would justify further integration. Better short-term head
predictions would still leave depth-resolved subsidence validation outstanding.

## Methodology references

- Ouédraogo, A. R., Hsu, S. M., and Wang, Y. (2023),
  [Estimating the Average Magnitude of Pumping Surrounding Monitoring Wells Using
  Signal Processing](https://ascelibrary.com/doi/10.1061/JHYEFF.HEENG-5760),
  *Journal of Hydrologic Engineering*, 28(4). Original AMP source; the publisher's
  citation and abstract were verified, not unrestricted full-text access. The
  experiment reuses the repository implementation and bandwidth choices, rather
  than claiming a new line-by-line full-paper reproduction.
- Lin, Y.-D., Tan, Y.-K., Ku, T., and Tian, B. (2023),
  [A Frequency Estimation Scheme Based on Gaussian Average Filtering Decomposition
  and Hilbert Transform: With Estimation of Respiratory Rate as an Example](https://www.mdpi.com/1424-8220/23/8/3785),
  *Sensors*, 23(8), 3785. The HGT signal-processing source adapted for AMP v2;
  its application is respiratory signals, not field pumping validation.
