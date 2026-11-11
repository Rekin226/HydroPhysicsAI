# Round 12: frozen head-driven subsidence layer and its declaration

| file | role |
|---|---|
| `model.py` | `SL.subsidence_layer.v1`: rate field from the 2019-2021 leveling, head-change driver from the 2001+ head record, prediction on the registered 1 km cells |
| `freeze.py` | writes the coefficient, input hashes and predictions under `results/dataprep/reports/SL/freeze` (local only) |
| `power.py` | synthetic power study of the area test, settings fixed before the run |
| `prereg_sl.py` | writes the declaration `SL.head_layer_companion.v1` |
| `../r12_yearbook/` | extraction of monthly heads 2001-2011 from the public hydrological yearbooks and the product `heads/L1_yearbook` |

Recorded on 2026-11-12:

- Coefficient: -0.127 per metre of year-on-year head change, fitted on 15 leveling years
  (2007-2021).
- Power study (4,000 worlds, three survey years): pass rate 0.65 if the fitted sensitivity is
  true and 0.19 if there is none, at the central year-shock setting of 0.5 cm/yr. The bar was
  0.80, so the area test is report-only (user ruling R12.3).
- The declaration is a report-only companion of N3S-L on survey years ROC116-117. Survey year
  2026 (ROC115) is recorded and never scored, because N3S-L seals it.

Nothing here changes a claim status or the page.
