# Twin evidence update, October 2026

Written 2026-10-11 from the audit of the same day. Every number below is read from a
result file named beside it; nothing here is a projection of what a fix might give.
Sections marked *pending* are filled in as their runs finish.

## 1. The corrected observation update does not change the subsidence numbers

The published projection (`results/twin_forward/datum_gate.npz`, built 2026-09-27)
predates the innovation update and restarted its displayed heads with the superseded
field update. Rerun with `--update-method innovation` and otherwise identical flags
(`results/twin_forward/datum_gate_innov.npz`, response basis
`response_basis_innov.npz`):

| quantity | field update (published) | innovation update |
|---|---|---|
| fan-mean forward subsidence 2023-32, baseline | 3.03 cm | 3.03 cm |
| avoided by 2032, irrigation −30 % | 0.94 cm | 0.94 cm |
| avoided by 2032, aquaculture retired | 1.30 cm | 1.30 cm |
| frozen-forecast head RMSE 2023-26, 158 wells (`new_data_challenge*/report.json`) | 2.072 m (ratio 1.243 vs climatology) | 2.048 m (ratio 1.229) |

The subsidence is identical because the column is driven by the free-running heads
(`--column-heads free`), which no observation update touches; only the displayed heads
and their 2023-26 score move. The page is rebuilt on the innovation run so that what it
displays is what the corrected method gives.

## 2. The page's uncertainty bands were too narrow

`twin.coverage` scores how often 2023-26 observations fall inside the run's intervals
(`results/twin/coverage_datum_gate_innov/report.json`, 158 wells, 6,186 well-months,
datum offsets and spreads from fitted months only):

| interval family | nominal 90 % covers | inflation needed | mean 90 % width |
|---|---|---|---|
| ensemble spread (what the page showed) | 74.7 % | 1.86 | 4.8 m |
| well's fitted-period residual sd | 80.2 % | 1.27 | 4.8 m |
| **combined, in quadrature** | **92.6 %** | 0.86 | 7.1 m |
| climatology with its residual sd | 84.5 % | 1.16 | 4.5 m |

The combined interval is the one registered in the prospective protocol. The page keeps
the run spread as its band but labels it as spread, not probability, and prints this
measured coverage beside it.

## 3. A prospective test is registered

`results/twin/prospective_2026-11/` (manifest, protocol, predictions; SHA-256 in the
manifest, registered 2026-10-11): the innovation run's datum-adjusted heads at 158 wells
for November 2026 to October 2027, the climatology baseline, and the combined 90 %
interval. Gate: equal-well RMSE ratio ≤ 1 against climatology, ≥ 100 stations with ≥ 6
months, 90 % interval coverage within 0.85-0.95. Scored with
`python -m hydrophysics.twin.prospective score` once the window closes. No other score
in this repository is prospective.

## 4. The damped head swings are least-squares shrinkage, not a timing defect

`results/twin/diagnostics/pumping_lag.json` (89 wells with usable local electricity
variance): observed and modelled head anomalies respond to the 3 km local electricity
anomaly at the same lag (0 months) with near-identical lag profiles. In sample the
model's anomaly amplitude is 0.73 of observed at correlation 0.65; the MSE-optimal
amplitude for that correlation is about 0.89 of observed, so the fit sits near its
optimum and a larger amplitude would raise the error. Storage variants (×0.25 to ×32,
`update_experiment/dynamics_extended.json`) move the 2023-25 free run by < 0.02 m. The
amplitude can only be recovered by forcing information the model does not have (canal
deliveries, metered abstraction, gauge rainfall) or by a statistical layer on top of the
physics; another calibration sweep is not the next step.

## 5. First out-of-sample subsidence test: 2023-25 rings and GNSS

`twin.ring_holdout` drives the frozen flow model (18 parameter sets) and the deliverable's
column from the free-running record end state through January 2023 to March 2025 under
three forcings and scores each site's cumulative displacement (from December 2022) and
OLS rate against the 14 compaction rings (13 with enough months) and the 19 on-grid WRA
GNSS stations (`results/twin/ring_holdout/report.json`, `rings_sites.csv`,
`gnss_sites.csv`). It is not a clean holdout (the report lists six reasons); it is the
only subsidence observation past the forecast origin.

| 2023-01 to 2025-03 | rings, cumulative (cm) R² / RMSE / bias | rings, rate (cm/yr) R² / RMSE / bias | GNSS, rate (cm/yr) R² / RMSE / bias |
|---|---|---|---|
| model, climatology forcing (the page's projection) | **+0.23 / 2.65 / +1.29** | −2.06 / 1.02 / −0.35 | −1.51 / 2.18 / −1.85 |
| model, recorded pumping | +0.24 / 2.63 / +0.80 | −5.64 / 1.50 / −0.77 | −3.40 / 2.89 / −2.56 |
| model, recorded pumping and weather | +0.23 / 2.65 / +0.86 | −5.26 / 1.45 / −0.68 | −3.19 / 2.82 / −2.49 |
| continue the site's own prior rate | −0.79 / 4.04 / +3.15 | −3.05 / 1.17 / +0.97 | −1.33 / 2.10 / +1.52 |
| no change | −0.45 / 3.63 / −2.02 | −5.23 / 1.45 / −1.33 | −2.56 / 2.59 / −2.20 |
| member p10-p90 covers the observed site value | 15 % (climatology) | 31 % | 5 % |

Reading: the rings slowed sharply after 2022 (site rates 1.7-4.0 cm/yr in 2017-22 fell to
0.6-2.0), so continuing the prior rate overshoots and no-change undershoots; the column's
cumulative displacement sits between them with weak but real skill. The spatial pattern
of rates is not captured by the model or by either baseline. Two signatures matter for
the policy question: the column under-predicts the ongoing rate, and more so when the
recorded (lower) pumping is used, because recovering heads set no new minima and the
VEP column then only creeps; the rings kept compacting. That is the same deficit
section 6 finds. The member spread is far too narrow to be read as a range. GNSS rates
carry full-column and tectonic motion the model does not represent; the model does not
beat the continued-rate baseline there.

## 6. Policy mechanism: the aquitard column does not resolve it (negative result)

The page's policy-timing statement (a pumping cut gives a one-time rebound and does not
slow the ongoing sinking) rests on the VEP column. The 14 compaction rings show a
sustained next-year rate sensitivity to head level of +0.0059 per m (95 % CI
0.0048-0.0071, `results/twin/mechanism/mechanism.json`); the VEP column's own hindcast
gives −0.0005 (CI −0.0013..0.0002). The hypothesis tested here: a zonal column with the
delayed-drainage term (`calibrate_coupled --creep aquitard`), fitted on leveling exactly
as the deliverable's column, keeps the leveling skill and carries that sensitivity.

| column (leveling target, 11-yr tau ceiling) | leveling R² out of fold | rings R² | lev1 on its hindcast (CI) | +1 m step in the projection: sustained rate change |
|---|---|---|---|---|
| VEP (deliverable) | 0.652 | 0.009 | −0.0005 (−0.0013..0.0002) | 0 |
| VEP + aquitard term | 0.656 | 0.003 | −0.0005 (−0.0013..0.0003) | −5e-6 m/yr |
| observed rings | | | **+0.0059 (0.0048..0.0071)** | |

Sources: `results/twin_runs/stage3_datum_gate/coupled_leveling{,_aquitard}/vep_zonal_leveling.json`,
`results/twin/mechanism_aquitard/mechanism.json` (`--recompute-column`). The leveling
fit pushed the new term to its bounds in every zone (mid zone: time constant at the
100-year ceiling with coefficient 0.001; distal zone: at the 1-year floor, acting as
extra elastic storage). The leveling network does not identify a level-dependent rate,
so a column fitted to it cannot carry one, whatever its rheology.

Consequence for the page: the avoided-subsidence number by 2032 stands (both columns
give it), but the statement that the benefit arrives within months and the sinking
rate is unchanged is a property of a column the ring record contradicts. The page
now carries that as an unresolved question beside the number rather than as a
finding. Resolving it needs a target that sees the rate response: the 2023-25 ring
and GNSS continuation (section 5) and the Tuku trial records requested in
[FIELD_DATA_ACQUISITION.md](FIELD_DATA_ACQUISITION.md). Fitting the column on the
rings themselves would make the judge part of the fit and was not done.
