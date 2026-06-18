---
title: HydroPhysicsAI Demo
emoji: 💧
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.18.0
app_file: app.py
pinned: false
license: mit
---

# HydroPhysicsAI — live demo

Interactive demo of a GPU physics-informed neural operator for groundwater levels across
the **61 real monitoring stations of Taiwan's Zhuoshui alluvial fan**.
**Pick a station** from the descriptive dropdown (its true location is highlighted on the
map), then choose a forecast origin, to see:

1. a **free-running PhysicsUDE simulation** vs the truth (the hard "simulation mode"), and
2. a **probabilistic 30-day forecast** with a calibrated 90% interval.

The **map shows real geography**: every station at its actual TWD97 coordinate on the real
alluvial fan, with the fan outline, rivers, and coastline as a backdrop, markers colored
coastal vs inland, real names on hover, and the current selection starred — so you can see
*where* each station sits. All charts are **interactive Plotly figures** (hover for date +
value, drag to zoom, double-click to reset), rendered through Gradio's `gr.Plot`.

> **Data policy.** Station **locations, names, coastal/inland group, and the GIS map
> layers are REAL** (publishing this location/boundary metadata was approved). The plotted
> **time series are SYNTHETIC illustrations** — the real Zhuoshui agency
> groundwater/rainfall series are not redistributable and never ship to this public Space.
> The synthetic generator mirrors the real signal structure so the exact model code paths
> run; **nothing plotted here is a real measurement.** Real-data headline scores live in
> the [GitHub repo](https://github.com/Rekin226/HydroPhysicsAI).

Startup loads a small precomputed artifact (`app/artifact.npz`, ~5 MB: synthetic series +
cached PhysicsUDE simulation and probabilistic-forecast cubes, built once on GPU by
`app/demo_data.py`), so the Space comes up in seconds with no live training. Changing the
station or moving the slider only re-plots.
