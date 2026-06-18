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

Interactive demo of a GPU physics-informed neural operator for groundwater levels.
**Pick a well** from the descriptive dropdown (its location is highlighted on the map),
then choose a forecast origin, to see:

1. a **free-running PhysicsUDE simulation** vs observed (the hard "simulation mode"), and
2. a **probabilistic 30-day forecast** with a calibrated 90% interval.

The well **map** scatters every well on its coordinates, colored coastal vs inland, with
the current selection starred — so you can see *where* each well sits. All charts are
**interactive Plotly figures** (hover for date + value, drag to zoom, double-click to
reset), rendered through Gradio's `gr.Plot`.

> The demo runs on a **synthetic / illustrative** dataset generated at startup —
> coordinates, coastal/inland labels and series are all fabricated. The real Zhuoshui
> agency series cannot be redistributed and this Space is public, so **nothing shown here
> is a real measurement.** It exercises the exact same models and code paths; the
> real-data headline scores live in the
> [GitHub repo](https://github.com/Rekin226/HydroPhysicsAI).

Models train once at startup (CPU, ~1 min) and are cached; changing the well or moving the
slider only re-plots (no retraining).
