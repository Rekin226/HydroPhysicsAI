# Round 12 (exploratory): statistical subsidence layer

Status: exploratory, design-time. Leveling 2012-2021 is a spent unit, so nothing here is a
test and no number below is a registered result. Outputs are written outside the repository
(`HYDRO_R12_OUT`); no data is committed.

| script | what it does |
|---|---|
| `panel.py`, `baselines.py` | rebuilds the column's leveling target and scores model-free predictors on it (random and 10 km block folds) |
| `increments.py` | May-to-May benchmark increments and yearly head statistics at benchmarks |
| `backtest.py`, `layer.py` | rolling-origin backtest (1, 3, 5 years) of a spatially smoothed trailing rate against persistence and the page column |
| `nowcast.py` | first head-driven increment test on the 2012+ head record |
| `longrecord.py` | head-driven rate modulation on the 2001-2026 head record (public yearbook monthly means before 2012, extracted locally) |
| `../r12_dataset/build.py` | analysis-ready tables and a QC report |

Run from the repository root, e.g. `HYDRO_RESULTS=results python -m dataprep_staging.r12_leveling.layer`.

Recorded on 2026-11-12 (page model `datum_gate_innov`, which was fitted on these surveys):

- Column gate target, out of fold: same-period neighbour interpolation R2 0.89 (0.74 under
  10 km blocks); pre-2012 rate continued 0.22; the column's recorded value is 0.65.
- Cumulative drop from origins 2009-2020, RMSE in cm, layer against column on common
  pairs: 1 year 1.68 / 1.98; 3 years 2.90 / 3.41; 5 years 3.65 / 4.84. Changhua 3 years
  2.32 / 3.26; Yunlin 3 years 3.24 / 3.51.
- Head-driven term, coefficient fitted on earlier years only, test years 2013-2021: MSE
  skill over the layer +10.7 % (Yunlin +15.5 %, Changhua +2.1 %); coefficient about -0.12
  per metre of year-on-year head change, year-bootstrap 90 % interval -0.30 to +0.02.

Limits: one layer family was chosen after looking at these scores; the head term's interval
includes zero; the 2022-2025 county figures were read as an after-the-fact check only.
