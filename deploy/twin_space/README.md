---
title: Choushui Fan Twin
emoji: 💧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Groundwater policy scenarios + analyst, Choushui fan
---

# Choushui Fan Twin: decision page + analyst

A research scenario page for groundwater pumping policy on Taiwan's Choushui River alluvial
fan, with a plain-language analyst beside it (English or Traditional Chinese).

- **The page** (right) opens on its answer: the Yunlin head response to a pumping cut, with a
  map, a time series, and a trust tab listing which tests passed and which did not. Numbers
  whose size is not supported by the registered tests sit in a labelled "Model internals"
  section.
- **The analyst** (left) answers only with the page's own numbers and sentences, cites the
  claim IDs behind each line, and links to the page in the same state. It refuses questions
  the evidence does not cover (for example Changhua levers, subsidence size, years after the
  page's horizon). Example questions answer instantly without the language model.

**Status: research beta.** The analyst's routing is still below its own accuracy bar on the
latest held-out evaluation (0 unsafe answers, accuracy short of the 97 % target). This is a
research scenario page, not a forecast or an agency product.

Only the public build ships: aggregates, no individual wells or benchmarks. Source, tests and
the evidence record: <https://github.com/Rekin226/HydroPhysicsAI>. Feedback: open an issue
there with the "Decision page feedback" form.
