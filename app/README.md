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
Pick a well and a forecast origin to see:

1. a **free-running PhysicsUDE simulation** vs observed (the hard "simulation mode"), and
2. a **probabilistic 30-day forecast** with a calibrated 90% interval.

> The demo runs on a **synthetic** dataset generated at startup — the real Zhuoshui
> agency series cannot be redistributed, and this Space is public. It exercises the exact
> same models and code paths; the real-data headline scores live in the
> [GitHub repo](https://github.com/Rekin226/HydroPhysicsAI).

Models train once at startup (CPU, ~1 min) and are cached; moving the controls only
re-plots.
