# Twin decision app: minimal redesign (2026-10-11)

Replaces the layout of `2026-09-23-twin-decision-app-redesign.md`; every feature of that spec is
kept. Source: `hydrophysics/twin/app/template.html` (one self-contained page, same three build
markers, same payload field names). The payload only grew: `modelcard.coverage`,
`modelcard.prospective`, `modelcard.ring_holdout` (all optional, built from `--coverage`,
`--prospective`, `--ring-holdout`; absent argument means absent line). `--three inline` now ships
Three.js in `<script type="text/plain" id="three-src">`, evaluated when the 3D drawer first opens.

## Layout

Desktop (>= 1180 px): header (title + one facts line, Evidence badge, EN/中文, theme, Settings, `?`,
Story, Model card) / headline + evidence line + 3 primary numbers (avoided subsidence with range,
area sinking faster than the threshold, parameter sets agreeing) with the other three (rail
gradient, layer-2 head, pumping cut) behind "3 more metrics" / one timeline (play, slider, "End of
horizon") / map 60 % + time series 40 % / policy bar (6 sliders, start toggle, Presets, reset, pin,
share) / tabs (Townships, Rail, Section, Compare, Trust, Ranking) / 3D drawer (closed).
Mobile: one column, map first, policy bar sticky at the bottom (collapsed to a summary + Edit).

## Decisions, one per audited problem

| # | Decision |
|---|---|
| 1 | Answer first; hero and auto-opened drawer removed; its four facts are the header line. |
| 2 | One evidence layer: header badge -> model card (evidence list, tests, details); one line under the headline; the twelve limits are written once (`caveats()`), rendered in the Trust tab and, one at a time with their own title, in a dialog from a few contextual links. |
| 3 | Tiles are number, unit, range, one qualifier; definitions in a `?` popover; text nodes updated in place. |
| 4 | Everything follows the year slider; headline says "by <year>"; ranges and set-agreement that exist only at the horizon say so. |
| 5 | Story is an overlay only: it snapshots the reader's state and restores it on Skip/Esc; Finish keeps the story's policy. No preset forces the start year; dropped pins toast. |
| 6 | One Settings popover: rate threshold, rail threshold, hatch, base map, overlays, play-runs-one-by-one. |
| 7 | Glossary: "Business as usual", "Policy", "Presets", "Hide changes below 1 mm or with low agreement". |
| 8 | Draw-a-section: instruction in the map header, Esc cancels. |
| 9 | Policy fields into preallocated typed arrays, one rAF-coalesced flush, reused offscreen canvas, static map layers cached, hover reads the cached field, Three.js evaluated on first open. |
| 10 | Skeleton + progress bar while decoding; side-by-side uses the full column width; error box with reload. |
| 11 | Real light/dark theme (CSS variables; canvas/SVG read them); header toggle; no `!important` beyond `[hidden]`. |
| 12 | One dictionary `T[lang][key]`; `browser_check` asserts equal key sets and no missing keys. |
| 13 | Base text >= 13 px, AA contrast, skip link, `?` keyboard sheet, focus trap in the story, swipe divider on plain arrows, roving tabs, town cards are buttons with inline content, hover readout not live (a separate polite region speaks on keyboard use). |

## Feature checklist (browser_check unless noted)

Difference map on a fixed snapped diverging scale with hatching (screenshots); linked selection
(map, series, township cards, section, 3D); fitted/tested/projected zones; Story 6 steps by keyboard
(walk); 3D explainer with exaggeration label (viewer screenshots); offline single file (0 remote
requests); palette unchanged; CSV/PNG/table exports (manual); share link + pins (walk); EN/中文
(walk); reduced motion (context flag); software-GL fallback (`--disable-webgl` run);
constructed vs runs band labels; townships, rail, section, compare, trust, ranking (walk).

## i18n key plan

Prefixes: `tA..tC` tiles and `def_*` popovers; `hl_*` headline; `ev_*`/`evc_*` evidence; `ly_*`,
`lg_*` map; `bn_*`, `d_*`, `dn_*` series; `pr_*` presets; `cv1..cv12_*` limits; `tt_*`, `th_*`, `kv_*`,
`pl_*` tests and card; `d3_*`; `st1..st6_*` story; `k_*` keyboard.
