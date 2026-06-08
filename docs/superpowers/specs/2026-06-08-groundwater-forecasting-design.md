# Groundwater operational forecasting (Task B) — design

## Goal

Accurately predict groundwater-level fluctuation in **operational forecast mode**: at a
forecast origin day `t0`, predict the level at `t0 + h` for horizons `h ∈ {1, 7, 30}`
days, allowed to use observed levels up to `t0` (data assimilation) plus forcing. This is
distinct from the existing free-running *simulation* benchmark, and is how real
monitoring / early-warning systems work. Decided with the user 2026-06-08 (voice + chat):
do forecasting first and measure the accuracy.

## Method (chosen)

**Global attribute-aware seq2seq LSTM**, one network across all 61 wells.

- **Encoder input** (lookback `L = 90` days ending at `t0`), per day:
  observed level, rainfall, upstream level, sin(doy), cos(doy).
- **Static conditioning**: the well's standardized attributes, concatenated to the
  encoder output (entity-aware, EA-LSTM style).
- **Decoder / head**: produce the next `H = 30` daily levels in one shot (direct
  multi-horizon). Horizons 1/7/30 are read off the 30-vector.
- **Future forcing**: rainfall, upstream level, sin/cos(doy) for `t0+1..t0+H` are fed to
  the head (perfect-forcing assumption — actual future forcing in hindcast). This matches
  the project convention of treating upstream level and rainfall as external drivers, and
  mirrors operational use of weather forecasts. Documented as an assumption.
- Output is the level **anomaly** relative to the last observed level `y[t0]` (predict the
  change), so the model assimilates the current state directly and only learns dynamics.

## Baselines (honest references at each horizon)

- **Persistence**: `pred(t0+h) = y(t0)`. The forecast-mode no-skill reference (already in
  `baselines.py`, generalized to horizon h here).
- **Climatology**: per-well seasonal mean by day-of-year (training only).
- (Optional) lightweight AR / gradient-boosted lagged regression as a stronger baseline.

## Evaluation protocol (no leakage)

- **Split**: train on pre-2019; evaluate forecasts whose origin `t0` is in 2019+.
  Assimilation uses observed levels up to `t0` only — never on/after the predicted day.
- **Scoring**: for each horizon h, assemble the series of h-ahead predictions across the
  val period per well and score KGE / NSE / RMSE (existing `metrics.py`), reported as the
  per-well distribution (median + mean), vs persistence and climatology at the SAME h.
- **Tuning**: all hyperparameters (L, hidden, layers, epochs, lr) chosen on an inner split
  carved from the pre-2019 training window (inner-train < 2018, inner-val 2018). The 2019+
  set is scored once. Same discipline as the UDE work.

## Components / files

- `hydrophysics/models/forecast_lstm.py` — `GlobalForecastLSTM` (`fit`, `forecast`):
  builds sliding-window samples, standardizes, trains the LSTM, produces (well, origin, h)
  forecasts. Self-contained; does NOT use the simulation-mode `GroundwaterModel` contract.
- `hydrophysics/forecast_eval.py` — windowed forecast harness + horizon-wise scoring
  against persistence/climatology; a `python -m hydrophysics.forecast_eval` CLI.
- `tests/test_forecast_smoke.py` — shape/finiteness smoke test (skips without torch).
- Results: `results/forecast/` aggregate horizon-wise skill tables (no raw level series).

## Success criteria

- Beats persistence and climatology at the 7- and 30-day horizons (1-day persistence is
  near-trivial, reported for context). Higher overall accuracy than the simulation-mode
  numbers, as expected for assimilated forecasting.
- Honest, reproducible, leakage-free; README/CONTRIBUTING note the new forecast track as
  separate from the simulation benchmark.

## Out of scope (later)

- Physics + Kalman/nudging assimilation (approach ②) as a comparison.
- Probabilistic / uncertainty forecasts.
- Real (uncertain) weather-forecast forcing instead of perfect forcing.
