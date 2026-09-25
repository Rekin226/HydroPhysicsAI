# Twin decision app: redesign spec (G9)

**Date:** 2026-09-23 · **Owner:** twin decision app · **Replaces the page built by:**
`hydrophysics/twin/viewer_app.py` (commit a6855ca) · **Output:** `results/twin/twin_app.html`

Inputs to this spec: the literature review in the G9 workflow (refs [1]-[23] below keep its
numbering), the audit of the current page (screenshots in the auditor's scratch, not
committed), and `docs/superpowers/STATE.md` section 3 (the verdict ledger).

---

## 0. The problem in one paragraph

The current page answers "what does the fan look like". A water authority asks something
else: *what does this policy change, where, by when, how sure are we, and does it protect
the high-speed rail and my township*. The audit found that the page cannot answer any of
those in two minutes:

- The default 3D view is broken by a units bug. `y = metres * exag/40` with 1 unit = 1 km,
  so the label "40x" is really 1000x.
- The policy signal (about 1 cm) is drawn on a 0-330 cm colour ramp.
- The headline "worst township" is an apex cell (林內鄉) that may be an artefact.
- The only ± shown is the baseline's spread, placed next to a policy delta that comes from
  a single member.

The literature [1]-[3], [13]-[14] and [19] points the same way: lead with the **difference
from a named baseline**, read numbers from **linked 2D views**, show uncertainty as
**agreement and bands**, and keep 3D as a context view.

## 1. Design principles and where they come from

| # | Decision | Motivated by |
|---|---|---|
| P1 | The default map is **scenario minus business-as-usual**, on a fixed, symmetric, diverging scale in cm (or m for head). Absolute fields are secondary. | [13] L'Yi 2021, [14] Gleicher 2011, [11] Bannister 2021 |
| P2 | **2D is the main reading surface.** The plan map and the time series are large. The 3D block is a smaller linked orientation view, closed until opened. No value is ever read off the 3D view. | [1] St John 2001, [2] Tory 2006, [3] Munzner 2014, [19] Lotteraner 2023 |
| P3 | **The answer comes first, in plain units.** A one-sentence headline and five KPI tiles sit above everything, and each tile carries its own uncertainty. | [4] Padilla 2018, [20] Tidwell 2008, [21] Herrera-García 2021 |
| P4 | **Uncertainty is shown as agreement and bands, not spaghetti.** Time series use a fan chart (median, 50 % and 80 % bands), with an optional "play members" hypothetical outcome plot (HOP). Maps are hatched where members disagree on the sign of the change. | [5] Padilla 2017, [6] Padilla 2020, [7] Kale 2019, [8] Zou 2024, [9] Correll 2018, [11] |
| P5 | **Structural caveats are named in plain words.** They sit next to the number they qualify, not at the bottom of a rail. The ensemble band does not contain them, and the page says so. | [10] Hullman 2019, [12] Potter 2025 |
| P6 | **Sections and fences explain the depth.** Fixed fences run along the HSR and along the fan axis, and the user can draw a section. The 3D view is an exploded layer view with honest exaggeration. | [18] Kessler 2009, [16] Malard 2023, [2] |
| P7 | **The HSR corridor is a first-class view:** a longitudinal profile with the differential-settlement gradient. | [23] Chiu 2026, [21] |
| P8 | **Township small multiples** on shared axes, sortable by benefit, with names in Chinese and English. | [13], [14] |
| P9 | **The page opens as a guided story** (5 steps). The user then gets an analyst mode ("martini glass"). | [15] Segel & Heer 2010, [16] |
| P10 | **Model and observation sit side by side.** Leveling benchmarks and monitoring wells are on the map. The fit period and the projection are divided visibly, and the skill is stated. | [17] Cox 2013, [20] |
| P11 | **One year control drives every view.** Animation is used in the story only. Comparison uses static difference maps. | [13], [17], [19] |
| P12 | **Scenarios can be named, pinned (up to 4) and shared in the URL.** | [20] |

## 2. Wireframe

Desktop is 1440 px wide. The grid is two columns, 60/40. The 3D view sits in a drawer.

```
+------------------------------------------------------------------------------------------+
| Choushui Fan Twin - policy explorer          [Story] [Analyst]   Model card (i)  EN|中文  |
| ! Projections beyond 2025 are the model's own trend, which has not been validated (why?) |
+------------------------------------------------------------------------------------------+
| HEADLINE (aria-live):                                                                    |
| "Cutting irrigation pumping 30% from 2026 avoids 0.78 cm (0.64-0.87) of fan-average      |
|  subsidence by 2032, most of it in 虎尾 Huwei, 土庫 Tuku and 元長 Yuanchang. All 36 model |
|  runs agree the effect is a benefit. Cost: 120 GWh/yr less pumping energy (~X Mm3/yr)."  |
+-----------+------------+-------------+--------------+---------------+--------------------+
| AVOIDED   | AREA > 1   | HSR MAX     | LAYER-2 HEAD | PUMPING CUT   | CONFIDENCE         |
| SUBSID.   | cm/yr 2032 | GRADIENT    | RECOVERY     | (the cost)    | 36/36 runs agree   |
| 0.78 cm   | 208->190   | 1/2400 ->   | +0.82 m      | -18% energy   | [robust]           |
| p10-p90   | km2        | 1/2600      | 0.71-0.88    | GWh/yr        | drift caveat (i)   |
| .64-.87   | -18 km2    | seg. flags 3|              |               |                    |
+-----------+------------+-------------+--------------+---------------+--------------------+
| MAP (2D, primary)            [Diff][Side-by-side][Swipe]   | TIME SERIES (linked)         |
| layer: [dSubs cm][dHead m][rate cm/yr][absolute]           | Fan-average cumulative subs. |
| +--------------------------------------------------------+ | since 2012-01, cm            |
| | orthophoto (muted 40%) + dField (diverging, fixed      | |  __ baseline median+band     |
| | +/-1 cm), hatching where <80% of runs agree,           | |  __ scenario median+band     |
| | township outlines + labels, HSR line (bold),           | |  | fit | 2023-25 | 2026-32 | |
| | rivers (dashed, "no-flow in model"), leveling dots     | |  | obs | anchored| model   | |
| | (skill-coloured ring), wells (triangles), north arrow, | |  |     |         | trend   | |
| | 10 km scale bar, legend with units                     | |  [ ] play runs one by one    |
| +--------------------------------------------------------+ |------------------------------|
| year  2012 |------[====o]------| 2032   [>] play          | Selected: 虎尾 Huwei / cell   |
|                                                            | / benchmark: obs vs model    |
+------------------------------------------------------------+------------------------------+
| POLICY LEVERS                         | LEVER RANKING (what each lever buys)             |
| Irrigation    [----o-----] -30%       | cm avoided by 2032 per 10% cut, per GWh saved   |
| Aquaculture   [--------o-] -0%        | Aquaculture  ########   0.18 cm/10% ...         |
| Livestock / Domestic / Industry / Oth | Irrigation   #####      0.08 ...                |
| Starts [2026 | 2030] (solved runs)    | Livestock    ##                              |
| [Pin as A] [Pin as B] [Share link]    | ...                                             |
+---------------------------------------+-------------------------------------------------+
| TABS: [Townships] [HSR corridor] [Section] [Compare pinned] [Where the model is trusted] |
|  Townships: 4x5 sparklines 2012-2032, baseline vs scenario, band; sort by benefit/risk   |
|  HSR: chainage x-axis, subsidence (top) and gradient (bottom), threshold line, flags     |
|  Section: fan-axis / HSR / user-drawn; 4 aquifers, heads (base vs scen), compaction bar  |
|  Compare: up to 4 pinned policies x 5 KPIs table + small-multiple difference maps        |
|  Trusted: hindcast skill per township, leveling residual map, known limits list         |
+------------------------------------------------------------------------------------------+
| [3D context view  v]  (drawer, closed by default; lazy-built on open)                    |
+------------------------------------------------------------------------------------------+
```

The wireframe's numbers only show layout. The measured ones are 0.78 cm (0.64-0.87),
+0.82 m, 36/36 and the 208 km² baseline area above 1 cm/yr. The HSR gradient, the energy
figures, the "-18 %" and the ranking bars are placeholders. The builder computes them.

Phone (below 760 px): one column, in this order: headline, KPI tiles (2x3), map (full
width, square), year slider, time series, levers, tabs as an accordion, 3D drawer. Nothing
lives inside a `position:relative; overflow:hidden` stage. That is the audit's mobile bug.

## 3. Required elements

### 3.1 Impact strip (headline and KPI tiles) [P3, P4, P5]

The strip answers what, how much, where, when and how sure. It always compares the
**current policy** with the **named baseline** "business as usual, 2023-2032 pumping at
the 2012-2022 class mix". Every tile shows a value, its unit, a p10-p90 range, and a
one-line definition on hover or tap.

| Tile | Definition (exact) | Unit | Source |
|---|---|---|---|
| Avoided subsidence | fan-average of `subs(policy, Dec 2032) - subs(policy, Dec 2022)` minus the same quantity for the baseline, with the sign flipped so that a benefit is positive. Also shows the peak cell and the share of baseline forward subsidence avoided. | cm, % | basis superposition. Band from §3.4 |
| Area sinking faster than threshold | cells (1 km² each) whose rate over the final 12 months (Dec 2031 to Dec 2032) exceeds the threshold, shown baseline to policy with the difference. Threshold selector 1/2/3 cm/yr, **default 1**: the model's baseline maximum is 3.1 cm/yr and only 4 cells pass 3, so a 3 cm/yr default would read "0 to 0". | km² | subs fields |
| HSR corridor | the maximum differential-settlement gradient along the rail over 2023-2032 (see §3.6), baseline and policy, as angular distortion 1/N. Also counts the segments above threshold. | 1/N, segments | subs fields sampled on the HSR polyline |
| Layer-2 head recovery | fan-average `head_L2(policy) - head_L2(baseline)` at Dec 2032 (layer 2 is the main production aquifer), plus the max cell | m | basis heads. Band from §3.4 |
| Pumping cut (the cost) | `sum_c (1 - f_c) * E_c`, the class energy for 2023-2032. It is shown as GWh/yr and as a percentage of total pumping energy. A Mm³/yr conversion appears **only if** the builder has a published kWh-to-m³ factor per class, labelled "approximate". Otherwise GWh only. | GWh/yr, % | class energies (§4, new small array) |
| Confidence | "N of 36 runs agree the policy reduces subsidence" plus a word: **robust** (≥ 90 %), **likely** (≥ 66 %), **uncertain** (< 66 %). Beneath it, always, sits the drift caveat link. | count | §3.4 |

**Headline sentence**, generated in JS from the tiles by a template with no free text:
`"{Policy phrase} from {year} avoids {X} cm ({p10}-{p90}) of fan-average subsidence by
2032, {share}% of what the baseline would add. Largest benefit: {top 3 townships}.
{N}/36 runs agree it is a benefit."` It updates on slider `change` (release), not on
every `input` event, so that screen readers are not flooded. It lives in an
`aria-live="polite"` region.

**Worked values for the solved runs**, from `physical_spread_nonudge.members.csv`, 36
paired members (the builder must reproduce these, see §8):

| scenario | Δ forward subsidence, fan mean, p10 / p50 / p90 (cm) | Δ layer-2 head, p10 / p50 / p90 (m) | runs agreeing |
|---|---|---|---|
| irrigation -30 % from 2026 | -0.87 / -0.78 / -0.64 | +0.71 / +0.82 / +0.88 | 36/36 |
| aquaculture retired from 2026 | -1.83 / -1.77 / -1.26 | +0.98 / +1.35 / +1.40 | 36/36 |

### 3.2 Linked 2D map and time series (primary), 3D as context [P2, P11, P10]

**Map** (`<canvas>`, 2D context, no WebGL):

- The 59x77 grid is drawn as nearest-neighbour blocks upscaled to the canvas, over the
  orthophoto at 40 % opacity. The orthophoto can be switched to EMAP or to none.
- **Layers** (radio): Δ subsidence since Dec 2022 (default) · Δ layer-2 head · subsidence
  rate for the year · absolute cumulative subsidence since Jan 2012 · absolute layer-2
  head. Absolute layers use a perceptually uniform sequential ramp (viridis or cividis)
  whose limits are the **p2-p98 of the baseline**, not the maximum. The audit found that
  a maximum-based scale puts 75 % of the fan in the palest 10 % of the ramp.
- **Δ scale:** diverging, **fixed and symmetric**, snapped to the nearest of
  {±0.25, ±0.5, ±1, ±2, ±5} cm (or {±0.5, ±1, ±2, ±5} m for head). It is chosen once per
  session from the largest single-lever response (so ±2 cm for this deliverable), and
  does **not** change when the policy changes. Otherwise a small effect would look as
  large as a big one. The legend states the units and "blue = less sinking than
  baseline".
- **Overlays** (checkboxes): township outlines and labels (default on), HSR (on), rivers
  Choushui/Xinhuwei/Beigang as a dashed line labelled "treated as no-flow in the model"
  (on), leveling benchmarks (off in Story, on in Analyst), monitoring wells (off), fan
  zone boundaries (off). The fan zone boundaries overlay is how a viewer spots the
  zonal-column step the audit saw.
- **Agreement hatching** (P4, [9], [11]): diagonal hatch over cells where fewer than 80 %
  of runs agree on the sign of Δ. A toggle switches it to a value-suppressing bivariate
  palette. Where agreement data is unavailable (see §3.4 fallback), the hatch marks cells
  with |Δ| < 0.1 cm and the legend says "change below 1 mm, not meaningful".
- North arrow, 10 km scale bar, and grid coordinates (TWD97) in the hover readout.
- **Pick:** click or tap a cell to select it and its township. Shift-click selects the
  township only. Clicking a leveling dot selects that benchmark. The selection is shared
  with the time series, the township grid, the section and the 3D drawer.

**Time series** (inline SVG, drawn by hand, no charting library):

- The default is the fan average. With a selection it shows the cell, township or
  benchmark.
- It plots cumulative subsidence since Jan 2012 (cm), with a second tab for layer-2 head
  (m).
- Baseline and scenario are each drawn as a median line with a 50 % band (darker) and an
  80 % band (lighter).
- Time is split into three vertical zones, each with a label:
  - **2012-2022 "fitted to observations"**, with observed leveling or head points drawn
    for a selected benchmark or well;
  - **2023-2025 "projection, within the tested horizon"**. The temporal gate tested 36
    months;
  - **2026-2032 "projection, the model's own trend (not validated)"**, drawn over a
    light hatch.
- A policy-start marker sits at 2026 or 2030.
- A "play runs one by one" toggle (HOP, [7]) animates the member trajectories one at a
  time at 2 per second, never all at once [6]. It is available only for the fan average
  and the townships, where member data exists (§4).

**3D context view** (drawer, closed by default, built lazily on first open) [P2, P6]:

- This is an exploded block model. The four aquifers and three aquitards are separated
  by fixed gaps, standing on the SRTM surface, with the orthophoto draped on top.
- **Vertical scale fix:** `y = metres * exag / 1000` (1 unit = 1 km). The exaggeration
  selector offers {10x, 25x, 50x}, default 25x, and the current value is shown on screen
  at all times. The builder unit-tests this (§8).
- The top surface is coloured by the same Δ field and palette as the 2D map. The head
  surfaces can be toggled (baseline or scenario).
- There is one clip plane. It follows the section line drawn in the Section tab, so the
  2D section and the 3D cut are the same object [2].
- Shadows are on as a depth cue. The view includes a compass and a "reset view" button,
  and a hint line: "drag = rotate · right-drag or two-finger = pan · wheel or pinch =
  zoom".
- There are **no numbers in 3D**. Hovering shows the value in the 2D readout, not in the
  3D view.
- The Story step 3 explainer runs here: animated drawdown, then aquitard compaction,
  then the ground surface lowering, in the style of [16].

### 3.3 Scenario comparison [P1, P12]

- **Diff** (default): the map shows policy minus baseline.
- **Side by side:** two maps with the same scale and a synchronised pick. The left map is
  the baseline or pinned A, the right map is the current policy or pinned B.
- **Swipe:** one map with a draggable vertical divider between A and B. It is keyboard
  operable (arrow keys move it 5 %).
- **Lever ranking** (always visible under the levers). A horizontal bar chart of the
  response to retiring each class fully, from the basis. There are two metrics (toggle):
  "cm avoided by 2032 (fan average)" and "cm avoided per 100 GWh/yr of pumping energy
  forgone". The second is where aquaculture stands out; the audit found it is the most
  effective lever per unit of energy. Values for the 2026 and 2030 start are shown as
  paired bars.
- **Compare pinned** tab: up to 4 pinned policies (A-D), stored in the URL hash and in
  `localStorage`, with every read and write in `try/catch`. It shows:
  - a table of the six KPI tiles for each policy;
  - a row of small-multiple Δ maps on the shared fixed scale.

  The reference presets are "Irrigation -30 %", "Aquaculture retired", "Both" and "All
  classes -20 %". They appear as unpinned suggestions only.
- **Superposition honesty:** policies are formed from the response basis, which is exact
  for a linear model. When the chosen policy equals a **solved** scenario (cut30 or
  retire_aqua from 2026), the page switches to the solved 36-member fields and shows a
  chip "full model, 36 runs". Otherwise the chip reads "fast estimate from single-lever
  runs (error vs full model: X %)". X comes from the basis `check_irr50_2026` run and
  from the cut30 comparison, both computed by the builder [12].

### 3.4 Uncertainty non-experts read correctly [P4, P5]

Two kinds of uncertainty are shown, and kept visibly separate.

**A. Ensemble spread (quantified).** The forward npz holds 36 members, but only as mean
and std fields. Paired per-member deltas exist only as fan scalars in
`<forward>.members.csv`. The spec therefore needs one small addition upstream:

- `twin.forward` gains `--save-members yearly`. It writes `subs_members_yr` of shape
  (S, M, A, Y) and `headL2_members_yr` of shape (S, M, A, Y), as float16 at December of
  each year 2012-2032 (Y = 21). For the deliverable that is 3 × 36 × 2148 × 21 × 2 B ≈
  9.7 MB per array before compression. **It is not shipped to the page raw.** The
  builder reduces it (§4).
- **This is a GPU re-run** of the deliverable forward run. The orchestrator queues it; it
  is not started from the builder.
- **Fallback until that run exists**, which the builder must support:
  - The tile bands come from `members.csv` (the paired fan-scalar deltas above).
  - For a slider policy, the band is scaled from the nearest solved scenario, and the
    tile says "range scaled from the full-model runs". The ratios are p10/p50 and
    p90/p50 of the solved delta: irrigation from cut30, aquaculture from retire_aqua,
    and other classes from the wider of the two.
  - Township bands use the baseline `subs_std` only for the absolute series, never for Δ.
  - Map hatching falls back to the |Δ| < 0.1 cm rule and says so.

**B. Structural caveats (not quantified).** These are listed in plain words. Each one is
attached to what it qualifies, and a single "Why is the range not wider?" link collects
them. Wording follows STATE.md section 3, with numbers quoted, not re-derived:

1. **Decade-scale trend not validated.** "When the model runs on its own for 3 years it
   drifts: its error at the 158 wells is 6.3 m, against 2.0 m for simply repeating the
   past average. Before 2026 the model is held close to observations. After that, the
   trend is the model's own." Attached to the 2026-2032 hatch and the Confidence tile.
2. **Pumping stress spread over 10 km.** "The model spreads each area's pumping over
   about 10 km. This fits the wells best, but the reason is not yet known (meter vs well
   location, or irrigation water moved between districts). A wider spread fits heads
   equally well and shows almost no policy effect." Attached to the lever ranking.
3. **Slow clay creep is uncertain.** "The clay's creep time constant cannot be pinned
   down from 11 years of data. A longer constant gives more subsidence after 2032 than
   this model shows." Attached to the Avoided tile.
4. **Some parameters sit at their limits** (10 of 32), so they are held, not varied.
   The range therefore understates the uncertainty. Attached to the Confidence tile.
5. **Rivers are treated as no-flow** (Choushui, Xinhuwei, Beigang). This matters near
   the rivers, where real heads recover faster. Attached to the river overlay legend.
6. **The head test does not test policy response.** The +0.80 head score measures
   interpolation between wells, not whether pumping changes are right. Attached to the
   model card verdict. The badge must not show +0.804 alone.

The **model card** (the (i) button) holds the gate JSON fields: `fix_eta` 0.5,
`fix_head_extra` 40 m, `return_flow`, `pump_spread_km`, `bounds_hit`, `n_wells` 158,
`git_commit`, `temporal_gate`. It also holds the member count and per-member
`hindcast_r2`. It gives three skills: head k-fold +0.804 vs IDW +0.702, column out-of-fold
+0.589, full-chain leveling hindcast +0.579 (RMSE 6.2 cm, bias -0.8 cm). It states the
data vintage.

### 3.5 Story mode and analyst mode [P9]

**Story** is the default on first load. It skips straight to Analyst if the URL hash
carries a policy. It is a stepper with Back, Next and Skip, and each step sets the view
state:

1. **The fan and its layers.** The 3D drawer opens, exploded, and slowly rotates once.
   Text: 4 aquifers, clay between them, a fan built by the Choushui River.
2. **Pumping lowers the water.** The 2D map shows the absolute layer-2 head and the time
   series shows the fan-average head, 2012-2022, with observations. Text: where the
   heads are lowest, and that pumping is ~60 % irrigation by energy.
3. **Clay squeezes and the ground sinks.** The 3D explainer animation runs [16]. The map
   switches to cumulative subsidence and the time series to the fan-average and
   benchmark hindcast, with a skill of +0.58. Text: sinking is permanent, and the lost
   storage does not come back ([22]).
4. **The rail.** The HSR tab opens with the baseline profile and the gradient threshold.
   Text: what matters to the railway is how uneven the sinking is along the track, not
   the total ([23]).
5. **Try a policy.** The irrigation slider is preset to -30 % and pulses. The map
   switches to Diff and the headline appears. Text: what the numbers mean and the drift
   caveat in one sentence. Then the levers unlock and the page enters Analyst.

**Analyst** unlocks:

- all layers and overlays;
- the Section tab with user-drawn lines;
- the rate threshold selector and the 2030 start;
- the Where-trusted tab;
- CSV export of the township and HSR tables, and PNG export of the map.

### 3.6 Township and HSR-corridor summaries [P7, P8]

**Townships:**

- The grid is 4 × 5 small multiples, one per township, with a shared y-axis. Each panel
  shows cumulative subsidence from 2012 to 2032, a baseline band with its median, and the
  scenario median.
- Each panel shows the Δ at 2032 as a number with a robust/likely/uncertain marker.
- Sort by: benefit (cm), baseline forward subsidence, or name.
- Labels are bilingual, e.g. "虎尾鎮 Huwei". The builder carries a 20-entry romanisation
  table in code. It is public administrative names, not data.
- **Coverage gap:** `cell_townships.csv` labels 1,091 of 2,148 cells, all in Yunlin
  (county code 1 for 1,063). The Changhua half of the fan is unlabelled. Until a
  Changhua township file exists, the grid shows the 20 Yunlin townships plus one panel,
  "Changhua (not yet split by township)", built from the unlabelled cells, flagged as
  such. The builder must not call a Yunlin township the fan's "worst" without saying
  Changhua is pooled.
- **Artefact guard (audit item 3).** Each panel also carries the hindcast skill against
  the leveling sites inside the township (n, bias, R² when n ≥ 5). If a township's
  projected value leads the ranking but it has fewer than 5 benchmarks or a hindcast bias
  greater than 5 cm, the panel shows "low confidence: few or poorly matched benchmarks",
  and the headline's "largest benefit" list skips it. This is how the 林內鄉 Linnei apex
  hotspot is handled until the column is checked. It is not hidden, only not headlined.

**HSR corridor:**

- **Geometry.** Add a new committed file `hydrophysics/twin/app/geodata/thsr_choushui.csv`
  (`chainage_km, x_twd97, y_twd97`), traced by `hydrophysics/twin/app/geo.py`, with
  `thsr_stations.csv` and `rivers_choushui.csv` beside it. It is the THSR centreline across the fan, digitised
  from OpenStreetMap (ODbL, attributed on the page) and resampled every 250 m. It is
  public infrastructure, not project data. The builder takes `--hsr PATH`, defaulting to
  that file. If the file is missing, the HSR tile and tab show "rail alignment not
  loaded" and nothing else changes.
- **Profile:**
  - x is chainage (km), with station markers for Changhua (彰化) and Yunlin (雲林).
  - Top panel: cumulative subsidence since 2012 along the line, baseline and policy
    (median and 80 % band), sampled bilinearly from the cell field.
  - Bottom panel: the gradient along the line over 2023-2032, as angular distortion
    `|Δs| / L` over a 1 km moving window, with a dashed threshold line and flagged
    segments shaded.
- **Threshold.** The default is 1/1000 over 2023-2032, labelled "illustrative screening
  value, not the operator's criterion". It can be edited in Analyst. This spec does not
  invent an operator threshold. The builder may replace the default only with a cited
  value.
- **Caveat.** The model is a 1 km grid, so gradients shorter than about 2 km are not
  resolved. The profile says so, and the gradient is smoothed to 2 km before it is
  flagged.

### 3.7 Section tab [P6]

- Three presets: fan axis from apex to coast, the HSR, and west to east through Huwei.
  In Analyst the user can also draw a line with two clicks on the map.
- The section is inline SVG at full tab width (at least 600 px, 12 px labels):
  - depth (m) on y, distance (km) on x;
  - the 4 aquifer bands at per-zone depths, with aquitards hatched;
  - baseline head in each aquifer as a solid line and scenario head as dashed, on a head
    axis on the right;
  - a compaction bar strip along the top showing Δ subsidence.
- Each aquifer band is coloured by the **per-cell** head along the line, not the
  midpoint's head (audit item 8).

### 3.8 Where the model is trusted (tab) [P10]

- A map of the leveling residual at the benchmarks: hindcast minus observed, as a
  diverging dot fill with a colour-blind-safe palette.
- A per-township skill table.
- The six caveats of §3.4-B.
- The provenance chip: "fields from the full model" or "fast estimate".

## 4. Data contract: what each panel needs, and from where

The existing CLI stays compatible: `--forward`, `--basis` and `--out` are required and
optional exactly as today, and `--townships`, `--quarter`, `--delta-step` and `--basemap`
are kept. The new optional flags are `--hsr`, `--members-csv` (defaulting to
`<forward stem>.members.csv`), `--leveling` (default: the panel under
`$HYDROMIND_GW_DATA/ls_cache`, via `twin/leveling.py`), `--wells` (default: the
calibration's station list), `--temporal` (held-out-years predictions npz behind caveat 1
and the "tested horizon"), `--column-csv` (column skill table), `--theta` (the calibration's
theta json, for the learned stress radius in caveat 2), `--three {cdn,inline,none}`
(default `cdn`) and `--private` (§11.1). Every new input degrades gracefully when absent.

Notation: A = 2148 cells, K = 6 classes, Y = 21 year-ends (Dec 2012 to Dec 2032),
M = members. Quantisation is int16 at the stated step unless noted.

| Page array | Built from | Shape shipped | Step | ~Size b64 | Used by |
|---|---|---|---|---|---|
| `mask, nx, ny, dx, x0, y0` | forward | scalars + bitmask | - | <1 KB | all |
| `ground` | basemap `dem` | A | 0.1 m | 6 KB | 3D, section |
| `ortho` | basemap `PHOTO2`, **re-encoded at JPEG q60, 1200 px long side** | 1 image | - | ≤ 700 KB | map, 3D drape |
| `emap` | basemap `EMAP`, optional, same re-encode | 1 image | - | ≤ 500 KB | map |
| `zone, town_idx, town_names(zh,en)` | `fan_zones`, townships csv | A (uint8) | - | 6 KB | map, townships |
| `subsBase_yr` | forward `subs_mean[0]` at year-ends | A × Y | 0.01 cm | 120 KB | absolute layer, rate, time series |
| `subsBand_yr` | per-member if present, else `subs_mean[0] ± 1.2816·subs_std[0]` | 4 × A × Y (p10,p25,p75,p90) | 0.01 cm | 480 KB | bands, absolute uncertainty |
| `dSubs_basis_yr` | basis `subs_mean[k] - subs_mean[0]`, k = 6 classes × {2026, 2030} | 12 × A × Y | 0.001 cm | 720 KB | Δ map, all tiles |
| `dHeadL2_basis_yr` | basis `heads_mean[k, 1] - heads_mean[0, 1]` | 12 × A × Y | 1 mm | 720 KB | head tile, Δ head layer |
| `headBase_L_yr` | forward `heads_mean[0]`, **year-ends only** (was quarterly) | 4 × A × Y | 1 cm | 480 KB | section, head layer, 3D |
| `dHead_basis_L_2032` | basis Δ heads, all 4 layers, **2032 only** | 12 × 4 × A | 1 mm | 140 KB | section scenario heads |
| `solved` | forward scenarios 1-2 (`cut30`, `retire_aqua`): Δsubs and ΔheadL2 median, p10 and p90 across members, year-ends | 2 × 2 × 3 × A × Y | as above | 1.4 MB with members, 240 KB with medians only | solved-scenario chip, map agreement, bands |
| `agree_frac` | per-member paired Δsubs sign agreement at 2032, solved scenarios | 2 × A (uint8, %) | 1 % | 6 KB | hatching |
| `fanMembers` | members.csv paired deltas + per-member fan-average series if `--save-members` | 3 × M × Y | 0.01 cm | 4 KB | HOP, fan bands |
| `townMembers` | per-member township means (if `--save-members`) | 3 × M × 21 towns × Y | 0.01 cm | 110 KB | township bands, HOP |
| `classEnergy` | `inp.E_by_class` 2023-2032 annual mean (builder reads it via the same input loader as `forward`; CPU only) | K | GWh/yr | <1 KB | cost tile, lever ranking |
| `hsr` | `--hsr` polyline + precomputed cell weights (bilinear) | N_pts × (4 idx + 4 w) | - | 30 KB | HSR tile, profile |
| `leveling` | benchmarks in the fan: xy, annual observed cumulative since 2012, hindcast residual | ~800 × (2 + ≤ 11) | 0.1 cm | 90 KB | map dots, trusted tab, time series obs |
| `wells` | 158 calibration wells: xy, layer, annual observed head | 158 × (3 + 11) | 1 cm | 15 KB | map, time series obs |
| `rivers, county_line, fan_outline` | polygon json + river centrelines (OSM, simplified to 200 m) | polylines | 1 m | 20 KB | map overlays |
| `gate, modelcard` | forward `gate` JSON + skills | JSON | - | 3 KB | model card |
| `basis_error` | builder-computed: superposition vs `check_irr50_2026` and vs solved cut30, fan mean and p95 cell | 4 floats | - | <1 KB | provenance chip |

**Dropped from the current payload:**

- the quarterly 4-layer `headBase` (1.95 MB; replaced by year-ends);
- the full `dHead` (C × 4 × time, 3.0 MB; replaced by layer 2 through time plus 4 layers
  at 2032);
- the per-quarter `subsStd` (0.49 MB).

Monthly resolution is not needed for a decision page, because the time control steps by
year. The Story animation interpolates linearly between year-ends.

**Solved 2030 runs** (audit item 7): the "Starts 2030" option uses the basis `*_2030`
rows directly. It must not shift the 2026 response in time.

## 5. Computation in the page

- The policy vector is `f` (K factors) plus the start year s ∈ {2026, 2030}. It gives
  `Δ(f, s) = Σ_k (1 - f_k) · dSubs_basis[k, s]`. That is O(K · A) for the **current year
  only** (12.9 K multiply-adds) on `input`, plus O(K · A · Y) (270 K) on `change`, which
  feeds the time series and the townships.
- Township and HSR aggregates are precomputed per basis row by the builder (`town_dSubs`
  of shape 12 × 21 × Y, and `hsr_dSubs` of shape 12 × N_pts × Y). Tiles are then O(K)
  per update.
- Budget: an `input` tick in under 8 ms on a mid phone, and a `change` in under 50 ms.
- There is no mesh rebuild while the 3D drawer is closed. When it is open, only the top
  surface's colour attribute updates on `input`. Geometry rebuilds only when the
  exaggeration changes. This removes the audit's O(A · C) rebuild of 7 instanced meshes
  per tick.

## 6. Performance budget

| Metric | Target | Current |
|---|---|---|
| HTML size | **≤ 5 MB** with `--three cdn`, ≤ 5.7 MB with `--three inline` (three.min.js is about 0.6 MB) | 10.7 MB |
| Time to headline visible, desktop, cold | ≤ 1.5 s | 5 s (3D first) |
| Time to headline visible, phone (software GL) | ≤ 4 s. WebGL is not needed until the 3D drawer opens. | 20 s |
| JS heap | ≤ 60 MB with the 3D drawer open | 35 MB |
| Offline | the whole page works except the 3D drawer when `--three cdn` and offline, which shows "3D view needs the internet or a build with --three inline". **No Google Fonts**: use a system font stack with `"Noto Sans TC", "PingFang TC", "Microsoft JhengHei"` for Chinese. | fails offline |

The builder prints the payload breakdown per array and **fails** (non-zero exit) if the
page exceeds 8 MB, as a guard against regressions.

three.js, when `--three inline`, is read from a pinned copy in the environment if present.
Otherwise the builder falls back to `cdn` with a warning. The CDN tag is
`https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js`, the same pin as
today, with an `onerror` handler that swaps the drawer for the message above. Plotly is
**not** used. The charts are simple, and inline SVG keeps the page small and printable.

## 7. Accessibility and units

- **Palettes.**
  - Δ fields use a diverging blue to white to orange scale (ColorBrewer `RdYlBu` reversed
    or `PuOr`), which is colour-blind safe. Blue means benefit (less sinking or higher
    head).
  - Absolute fields use `cividis` (subsidence) and `viridis` (head).
  - The palette choice is validated in the builder with a deuteranopia simulation check
    in the test (ΔE between endpoints ≥ 20 under simulation).
  - Meaning is never carried by colour alone: agreement uses hatching, flagged HSR
    segments use a pattern, and robust/likely/uncertain uses a word as well as a
    marker.
- **Units on everything.** Every axis, tile, legend and readout carries a unit and a
  reference: "cm since Jan 2012", "cm/yr (Dec 2031 to Dec 2032)", "m, layer 2 (main
  production aquifer)", "GWh/yr". A projection value always carries "projected".
- **Keyboard.** Every control is a native `<input>` or `<button>`. The map pick is
  keyboard driven: arrow keys move a focus cell, Enter selects. The swipe divider works
  with the arrow keys. The Story stepper responds to ←/→.
- **Screen readers.** The headline is in an `aria-live` region. Each chart has a "Show as
  table" toggle that renders the same numbers as an HTML table. The map has an `aria-label`
  summary sentence that is regenerated on change.
- **Contrast.** Text is WCAG AA (4.5:1). Tiles are at least 44 px touch targets on phone.
- **Language.** An EN|中文 toggle switches UI strings. Township names are always shown
  in both scripts.
- `prefers-reduced-motion` disables the Story rotation, the HOP autoplay and the
  explainer animation, and replaces them with stepped frames.

## 8. Build, tests, and acceptance

**Module layout.** `hydrophysics/twin/viewer_app.py` is rewritten in place and keeps the
CLI; it orchestrates only. Everything that needs judgement (percentiles, calibration,
artefact removal, skill, packing) lives in `hydrophysics/twin/app/prep.py`, the OSM
geometry tracer in `hydrophysics/twin/app/geo.py` (network only when run as its own CLI,
never at build time), and the HTML, CSS and JS in `hydrophysics/twin/app/template.html`
(package data). A pinned three.js for `--three inline` is looked up at
`hydrophysics/twin/app/vendor/three.min.js`. `from __future__ import annotations`, ruff
line length 100.

**New test file `tests/test_viewer_app.py`** (CPU only, under 30 s, synthetic 6 × 5 grid,
3 scenarios, 4 members, basis with 13 rows):

1. `build()` accepts the current CLI arguments, writes one HTML file, and makes no network
   fetch at build time.
2. The payload JSON parses, and every array in §4 has the documented shape.
3. Superposition: the page-side formula ported to numpy reproduces the basis delta
   exactly for a single-lever policy. `basis_error` is reported.
4. The 2030 start uses the `*_2030` rows, not a shifted 2026 row.
5. Vertical scale: the emitted JS constant for the metres-to-units factor is `exag/1000`.
6. The fixed Δ scale is symmetric and snapped to the allowed set, and it does not depend
   on the policy.
7. The size guard triggers above the limit, using a monkeypatched limit.
8. Missing optional inputs (`--hsr`, leveling, wells, members.csv) still build, and the
   corresponding panels carry their "not loaded" state.
9. The members.csv paired-delta p10/p50/p90 computation matches a hand-computed value.

**Acceptance on the real deliverable** (the orchestrator runs it, CPU only, using the
existing npz files):

- The builder's printed summary reproduces §3.1's solved-scenario table to ±0.01.
- The page is ≤ 5 MB with `--three cdn`.
- A headless render (the auditor's `render.py` recipe) at 1440 × 900 and 390 × 844 shows
  the headline and tiles above the fold. The Diff map for irrigation -30 % is visibly
  non-uniform. The phone layout has no clipped panels.
- The two-minute test from the audit is re-scored: what, how much, where, when, how sure
  and compared to what each have a named on-screen answer.
- The builder prints the top 5 cells and townships by baseline forward subsidence, each
  with its leveling support (n, bias). A reviewer checks 林內鄉 Linnei and the zone-step
  before sharing.

## 9. Dependencies on other gaps

| Needs | From gap | If absent |
|---|---|---|
| Per-member yearly fields (`--save-members yearly`) for map agreement, township bands and HOPs | new, small change to `twin.forward` plus one queued GPU re-run of the deliverable | fallback of §3.4-A, labelled |
| Retrained surrogate | G8 | not used by the page. The page uses the solver basis. The provenance chip never says "surrogate". |
| A policy-response criterion | G3 | the model card shows the head gate with caveat 6 |
| Lifted rivers boundary | G6 | caveat 5 stays |
| Changhua township labels | new data task | pooled "Changhua" panel |
| THSR polyline | new committed file (OSM, ODbL) | HSR panels show "not loaded" |

## 10. Out of scope

- Running the solver from the page.
- A server.
- Real-time data feeds.
- Monthly animation.
- Showing the 21 km model. It is the recorded non-deliverable per STATE.md.
- Local LLM features.

## 11. Amendments made while building (2026-09-23)

These override the sections above where they conflict.

### 11.1 Public page by default: observations only as aggregates

The repository is public, and the committed `results/twin/twin_app.html` is public too.
The benchmark and well locations and series are project data. So:

- **Default (public):** leveling and wells enter the page **only as aggregates**. These
  are the chain and per-township skill (n, bias, R²) and the fan-average series of the
  layer-2 wells. There are no benchmark dots, no well triangles, and no per-point residual
  map. §3.2's "leveling dots / wells" overlays and §3.8's residual map become township
  skill choropleths in the public build.
- **`--private`:** embeds every benchmark and well with its location and observed series.
  This is for a local page only, and its output must never be committed. The builder
  prints a warning when this flag is set.
- A test (`test_public_page_ships_observations_only_as_aggregates`) guards the default.

### 11.2 Basis rescaled to the solved ensemble runs

The response basis is one parameter set, but the solved scenarios are 36-member means. In
§3.3 the page would jump when a slider lands on a solved scenario. Instead,
`prep.calibrate_basis` computes a per-class factor, which is the solved fan-mean response
divided by the basis response at the horizon, for subsidence and layer-2 head separately.
Classes without a solved run (livestock, domestic, industry, other) take the mean factor
and are marked **assumed** on the page. The builder still reports the raw superposition
error (vs `check_irr50_2026` and the solved runs) before rescaling. That error is what
the provenance chip quotes.

### 11.3 Column artefact steps removed; forward change measured after the restart

The audit's 林內鄉 Linnei apex hotspot (item 3) comes largely from two artefacts of the
proximal column, not from physics:

- a jump as the column settles onto the first heads, about 130 cm per cell;
- a jump after the restart from observed heads at the origin, about 33 cm.

`prep.artefact_steps` removes both steps from every field, only in the cells it selects.
`prep.restart_bridge` bridges what the restart leaves elsewhere at year-end resolution.
"Forward change" (tiles, Δ maps) is measured from the first year-end **after** the
restart (`meta.yRef`), not from Dec 2022.

Policy differences are unaffected, because the steps are identical in every scenario. The
builder prints the removed step sizes. The root cause stays open under the column
calibration (STATE.md §3), and the page's model card names it.

### 11.4 Tested horizon comes from the held-out-years test

The boundary between the 2023-2025 zone and the untested zone in §3.2 is not hard-coded.
It is read from `--temporal` (the held-out-years predictions): the tested horizon is the
last observed year-end plus the free-run length of that test (`meta.yTested`). Caveat 1's
numbers (6.3 m vs 2.0 m) come from the same file. Without the file there is no tested
zone: everything after the last observation is drawn as "not validated".

### 11.5 Measured budget

The built page is **1.73 MB** with `--three cdn` (target ≤ 5 MB, guard 8 MB). Arrays are
quantised, byte-shuffled and gzipped (`prep.pack`), and decompressed in the page with
`DecompressionStream`. The template is about 190 KB.

### 11.6 Artefacts fixed at the source; per-member sidecar (2026-09-23, opt-in)

The root causes of §11.3 now have opt-in fixes upstream. Every default reproduces the
earlier runs.

- **A1, start-up.** Fixed with `twin.forward --column-hpc0-fast-days 365`. It sets
  `h_pc0 = 0` in every column with tau < 365 d, which is the proximal column only.
  `calibrate_coupled --hpc0-guard-days 365` stops a refit from bringing the offset back.
- **A2, restart.** Fixed with `twin.forward --column-heads free`: the column is driven by
  the projection from the hindcast's own end state, plus the ic perturbation.
  `--restart-taper-km R` tapers the displayed restart by distance to the wells.
- **A3, the 182 km line.** `--zone-blend-km w` is available in `calibrate_flow`,
  `calibrate_coupled` and `forward` (read from the theta meta and the column JSON). The
  sharp column must be refitted before it is used: blending it without a refit costs
  leveling R² (0.599 → 0.502 at 2 km).
- **Builder.** When the forward npz records both A1 and A2 (`forward_options`), the builder
  skips `artefact_steps` and `restart_bridge`.
- **Sidecar.** `--save-members yearly` writes `<out>.members.npz` with
  `subs_members_yr` (S, M·R, A, Y) and `headL2_members_yr` (S, M, A, Y). They are **int16**
  with explicit scales, 1 mm and 1 cm, rather than float16: the file is the same size and
  finer at 100 m of head. When the sidecar is present, the builder ships per-cell agreement
  (`solvedAgree<i>`: the percentage of parameter sets that agree on the sign). The map
  also hatches cells below 80 % agreement. The builder also ships the real yearly fan
  bands (`bandYr`) and per-township agreement (`townAgree`). Without the sidecar the
  fallbacks in §3.4 stay.

## References

[1] St John et al. 2001, Human Factors 43(1), doi:10.1518/001872001775992534 ·
[2] Tory et al. 2006, IEEE TVCG 12(1), doi:10.1109/TVCG.2006.17 ·
[3] Munzner 2014, Visualization Analysis and Design, ch. 6 ·
[4] Padilla et al. 2018, CRPI, doi:10.1186/s41235-018-0120-9 ·
[5] Padilla et al. 2017, CRPI, doi:10.1186/s41235-017-0076-1 ·
[6] Padilla et al. 2020, JEP: Applied, doi:10.1037/xap0000245 ·
[7] Kale et al. 2019, IEEE TVCG 25(1), doi:10.1109/TVCG.2018.2864909 ·
[8] Zou et al. 2024, arXiv:2411.02576 ·
[9] Correll et al. 2018, CHI, doi:10.1145/3173574.3174216 ·
[10] Hullman et al. 2019, IEEE TVCG, doi:10.1109/TVCG.2018.2864889 ·
[11] Bannister et al. 2021, Sci. Rep., doi:10.1038/s41598-021-98290-4 ·
[12] Potter et al. 2025, IEEE CG&A, doi:10.1109/MCG.2025.3549665 ·
[13] L'Yi et al. 2021, IEEE TVCG, doi:10.1109/TVCG.2020.3030419 ·
[14] Gleicher et al. 2011, Inf. Vis. 10(4), doi:10.1177/1473871611416549 ·
[15] Segel & Heer 2010, IEEE TVCG 16(6), doi:10.1109/TVCG.2010.179 ·
[16] Malard et al. 2023, C. R. Géoscience 355(S1), doi:10.5802/crgeos.152 ·
[17] Cox et al. 2013, J. Hydrology 491:56-72 ·
[18] Kessler et al. 2009, Computers & Geosciences 35(6) (DOI not tool-verified) ·
[19] Lotteraner et al. 2023, IEEE CG&A, doi:10.1109/MCG.2023.3309090 ·
[20] Tidwell & van den Brink 2008, Ground Water 46(2) (DOI not tool-verified) ·
[21] Herrera-García et al. 2021, Science 371 (DOI not tool-verified) ·
[22] Hasan et al. 2023, Nat. Commun., doi:10.1038/s41467-023-41933-z ·
[23] Chiu et al. 2026, Remote Sensing 18(15):2612.
