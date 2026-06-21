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
the **61 real monitoring stations of Taiwan's Zhuoshui alluvial fan**. **Pick a station**
by clicking the map or the descriptive dropdown, then explore five linked panels:

1. a **free-running PhysicsUDE simulation** vs the truth (the hard "simulation mode"),
2. a **probabilistic 30-day forecast** with a calibrated 90% interval,
3. **interactive what-if physics** — drag the *Rainfall ×* and *ET ×* sliders and the
   trained operator **re-simulates live** (a real forward pass through the recharge-memory
   ODE): cut the rain to create a drought and watch the table sink below baseline with the
   aquifer's natural lag,
4. a **basin groundwater surface** — the operator's table interpolated across the whole
   fan, with a day slider to scrub six years of seasonal recharge and drawdown, and
5. a **live recharge-stress monitor** — pull **real-time Open-Meteo weather** at the
   station's true coordinates and see whether the aquifer is **recharging** or under
   **drought stress** right now, via the same 100-day recharge-memory the operator uses.

The **map shows real geography**: every station at its actual TWD97 coordinate on the real
alluvial fan, with the fan outline, rivers, and coastline as a backdrop, markers colored
coastal vs inland, real names on hover, and the current selection starred — so you can see
*where* each station sits. All charts are **interactive Plotly figures** (hover for date +
value, drag to zoom, double-click to reset), rendered through Gradio's `gr.Plot`.

> **Data policy.** Station **locations, names, coastal/inland group, the GIS map layers,
> and all weather (precipitation / ET₀, Open-Meteo public ERA5) are REAL.** The plotted
> **groundwater time series are SYNTHETIC illustrations** — the real Zhuoshui agency
> groundwater series are not redistributable and never ship to this public Space. The
> synthetic generator mirrors the real signal structure so the exact model code paths run;
> **no groundwater value plotted here is a real measurement.** The live recharge-stress
> monitor shows only real public weather, never a groundwater level. Real-data headline
> scores live in the [GitHub repo](https://github.com/Rekin226/HydroPhysicsAI).

Startup loads a small precomputed artifact (`app/artifact.npz`, ~5 MB: synthetic series +
cached PhysicsUDE simulation and probabilistic-forecast cubes) plus the **trained operator**
(`app/models/operator.pt`, ~27 KB), built once on GPU by `app/demo_data.py`, so the Space
comes up in seconds. The what-if and surface panels run the operator forward live on CPU
(~20 ms); the recharge-stress monitor fetches Open-Meteo on demand (cached per station).
