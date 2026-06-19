# Per-site Sk via Distance-to-Coast Regression — Design

**Date:** 2026-06-19
**Status:** Approved (brainstorming), pending spec review
**Author:** brainstormed with Claude

## Summary

A leave-one-site-out-gated research attempt to rescue the head→subsidence coupling in the
Choushui explorer. The single basin-wide `Sk` failed (pooled R² = **−0.28**) and a
spatially-interpolated per-site `Sk` failed worse at leave-one-site-out (R² = **−2.40**).
This spec models the per-site compaction coefficient `Sk` as a function of **one physical,
leakage-safe, grid-evaluable predictor — distance-to-coast** — and **only ships a
continuous subsidence surface if it clears an honest leave-one-site-out gate (R² > 0)**.
If it fails the gate, the outcome is documented as a second negative result and no surface
is shipped.

This is explicitly framed as a likely-modest research attempt, not a promised win. n = 14
sites; the relationship is real but moderate.

## Motivation / verified feasibility (measured, not assumed)

Per-site coefficients `Sk_i` (compaction m per m of drawdown) vary ~28× across the 14 MLCW
sites (0.004–0.115) yet each site is individually linear (corr(D,C) 0.76–0.95). The spread
tracks distance-to-coast:

```
corr(distance-to-coast, Sk)     = -0.73
corr(distance-to-coast, log Sk) = -0.68     # explains ~46% of log-Sk variance
corr(column-depth, log Sk)      = -0.14     # weak AND borehole-only -> rejected
```

Closer to the coast → higher `Sk` (新生 at 6.4 km: 0.115; 嘉興 at 25.7 km: 0.004), the
expected marine-clay gradient. This is why pure spatial IDW failed (no coast axis) and why
a coast-axis regression has a genuine chance.

**Stated risk:** r = −0.68 is moderate, not strong; the nearest-coast site (新生, 6.4 km)
is alone with a 7 km gap to the next, so it is high-leverage. Leave-one-site-out is the
verdict and may land only weakly positive or fail. Accepted outcome.

## Scope decisions (locked)

- **Single predictor: distance-to-coast.** Justified by: dominant physical control on
  Choushui compaction; independent of the compaction signal (no target leakage);
  independent of head/drawdown (no `Sk = C/D` circularity); and evaluable at every grid
  cell (required for a continuous surface). Column-depth rejected (weak + borehole-only).
  Head/drawdown features rejected (circularity + parameter budget at n = 14).
- **Two parameters only** (`β₀, β₁`) for 14 sites — deliberately minimal so leave-one-
  site-out is meaningful.
- **Honest gate decides shipping.** A continuous subsidence surface is built ONLY if the
  leave-one-site-out pooled R² on compaction is > 0. Otherwise: documented negative result,
  no surface.
- **Out of scope (YAGNI):** multi-feature regression, ML models, GNSS/InSAR cross-check
  (blocked on wisenvr), retuning the head field or the explorer's head layer.

## Model

Model compaction **directly** (avoids the `C/D` ratio instability):

```
C_i(t) = Sk(distcoast_i) · D_i(t),   with   Sk(dc) = exp(β₀ + β₁ · dc)
```

`exp(...)` keeps `Sk` positive. Fitting procedure:

1. Per site, compute `Sk_i = (D·C)/(D·D)` (drawdown-weighted slope through origin), where
   `D` is the IDW-head cumulative drawdown at the site and `C` the re-zeroed measured
   compaction (both monthly, from the existing `calibrate_sk` pairing).
2. Weighted least-squares of `log Sk_i` on `[1, dc_i]`, weighting each site by `Σ D_i²`
   (sites with more drawdown determine their `Sk_i` more reliably).
3. `predict_sk(dc) = exp(β₀ + β₁·dc)`.

## The honest gate — leave-one-site-out (LOSO)

For each site `j`: refit `β` on the other 13 sites (steps 1–2 above), predict
`Sk_j = predict_sk(dc_j)`, and accumulate squared error of `Sk_j · D_j(t)` against
`C_j(t)`. Pooled `R² = 1 − SS_res / SS_tot`, `SS_tot` about the global compaction mean.

Reference points (reported alongside, computed by this repo):
- single basin-wide `Sk`: R² = −0.28
- spatial-IDW per-site `Sk` (LOSO): R² = −2.40
- per-site own `Sk` (in-sample, an upper bound, not a gate): R² = 0.81

**Pass bar: LOSO R² > 0.** Reported verbatim regardless of value.

## Components / integration

- **`hydrophysics/subsidence.py`** (append):
  - `site_distance_to_coast(stations, coast_shp) -> dict[sub_id -> float]` — shapely
    distance from each MLCW `(x,y)` to the coastline geometry (EPSG:3826 meters).
  - `fit_sk_regression(per_site, dist) -> {b0, b1, r2_insample, predict_sk}` — weighted
    log-linear fit (step 2). `predict_sk` is a plain callable `dc -> Sk`.
  - `loso_sk_regression(per_site, dist) -> {r2, per_site_pred, baseline_single, baseline_idw}`
    — the gate.
- **`hydrophysics/explorer.py`** (append, used only when the gate passes):
  - `coast_distance_grid(XX, YY, coast_shp) -> (n,n)` — distance-to-coast per cell.
  - `subsidence_grid_from_sk_field(HH, sk_field) -> (Tm,n,n)` — `sk_field(x,y) ·
    cumulative_drawdown`, where `sk_field` comes from `predict_sk(coast_distance_grid)`.
  - `build_explorer` gains a `coupling="single"|"coast"` switch; `"coast"` uses the
    regression surface and shows LOSO predicted-vs-observed in the validation panel
    (title annotated with the LOSO R²). Default stays `"single"` so nothing changes
    unless the gate passes and the caller opts in.
- **CLI:** `python -m hydrophysics.subsidence_report` prints the four R² numbers
  (single, IDW-LOSO, per-site-insample, coast-LOSO) and `β₀,β₁` — the verdict table.

## Data flow

```
stations(x,y) + coast_shp ──► site_distance_to_coast ─┐
calibrate_sk pairs (D_i, C_i) ───────────────────────┼─► fit_sk_regression (β)
                                                      └─► loso_sk_regression ─► LOSO R²  (GATE)
                                                                                   │
                          (only if R²>0)  coast_distance_grid ─► predict_sk ──► Sk(x,y)
                                                      └─► subsidence_grid_from_sk_field ─► explorer surface
```

## Error handling & failure modes

- **Gate fails (LOSO R² ≤ 0):** do not build the surface; the report prints the negative
  number; README/spec record it as a second documented negative. This is the expected
  branch if the coast signal doesn't generalize.
- **High-leverage near-coast site:** report the LOSO error for 新生 specifically (it will
  dominate); if removing it flips the verdict, say so explicitly rather than silently.
- **Distance computation:** assert all 14 distances are positive and within the fan's
  coastal range (0–30 km); coastline is the same EPSG:3826 CRS as the sites.
- **No extrapolation claims:** `Sk(x,y)` is only meaningful within the well/coast envelope;
  the existing fan-polygon mask already bounds the surface.

## Testing

- `fit_sk_regression` recovers a known `(β₀, β₁)` on synthetic `Sk = exp(β₀+β₁·dc)` data.
- `loso_sk_regression` on synthetic data with a true coast gradient returns **positive**
  R²; on synthetic data with `Sk` independent of `dc` returns **≤ 0** (the gate actually
  discriminates).
- `site_distance_to_coast` returns 14 positive distances in the expected range.
- Leakage guard: assert the predictor path never references compaction `C` (the fit takes
  `dist` + `per_site` pairs; the predictor is a function of `dc` only).
- If the real gate passes: an explorer smoke test with `coupling="coast"` writes a valid
  HTML.

## Deliverable

A verdict, honestly reported: the four R² numbers and `β`. **Then, conditionally**, either
a coast-calibrated subsidence surface in the explorer (validated by LOSO) or a documented
second negative result. Either way the README's explorer section is updated with the real
LOSO number, replacing the current "single-coefficient R²=−0.28, illustrative only" note
with whichever outcome occurs.
