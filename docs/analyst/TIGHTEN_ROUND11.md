# Analyst, round 11: the tighten-only post-route step (ruling R11.1)

Ruling R11.1 allows tighten-only analyst fixes for the v5 LLM routing errors: changes that can only
add clarifications or correct a clarification's reason. They are for development only and are
unscored. Held-out v5 ([`EVAL_2026-11-08_v5.md`](EVAL_2026-11-08_v5.md)) found four LLM routing
errors. Two of them are in scope.

## What

[`router/tighten.py`](../../hydrophysics_analyst/router/tighten.py) runs after every route:
hard refusals, the rules, the LLM, and the no-LLM fallback. `router.route` calls it, and it logs
what it changed as `tighten` in the audit record and in eval rows. It can do two things.

- **(a) A preset becomes `clarify missing_year`.** This happens when all of these hold:
  - the preset reports at a year (every routed preset except `how_sure`);
  - the question gives a lead of an indefinite number of years ("a few years from now", "in
    the next few years", "several years later", 過幾年之後, 未來幾年, 數年後);
  - the question gives no reporting year. A calendar, spelled, two-digit or ROC year, or a
    definite lead ("in 3 years", 三年後), counts as giving one.

  It fires only when the preset would really be answered as that preset. If the run would refuse
  it (no policy, no run, disabled, past the page), clarify it (bad parameters), or swap it for a
  validator refusal, the route is left unchanged. This fixes v5_034 (灌溉減兩成，過幾年之後的水位會怎樣？).
- **(b) A `clarify unknown` becomes `clarify out_of_grid`.** This happens when the question names
  a start or a cut size off the page grid:
  - a start the rules read as off the page;
  - a late start the rules do not read: 拖到2029年才實施, 2029年才開始, "not until 2028",
    "delay ... until 2027";
  - a percentage off the 5 % grid;
  - a third, sixth, seventh, ninth or odd eighth (三分之一, 1/3).

  Only the catch-all `unknown` is corrected. A specific reason (missing_cut, missing_year,
  ambiguous_town, no_llm) is left as given. This fixes v5_061.

Both changes return a bare clarification, with no preset and no parameters (like
`router.bare_clarification`). An off-grid start lists the page's starts as candidates.

**Why (a) agrees with OPS.** OPS.md's page default ("start 2026, year 2032") applies "when the
question gives none". The LLM prompt says the default applies "only when the question gives no
year at all". A question about "a few years from now" asks about a year but does not name it.
Answering for 2032 would answer a different question. The rule is the indefinite form of the
guards' leads: `guards.beyond_horizon` reads "in N years" or N年後 as start + N, and here N is
not given. Every committed set labels such questions `missing_year`: dev q138, v3_060 and
v5_034. On the dev set and v1-v5, (a) fires on exactly those three items and (b) only on items
labelled `out_of_grid`. v5_034's label is therefore consistent with OPS. OPS.md now spells this
out in the schema section.

## Proof that it only tightens

By construction (`tighten.tighten`):
- A route with a refusal (`refusal` set or preset `refuse`) is returned as the same object. So
  is a clarification shown as a refusal (`beyond_horizon`).
- A clarification can only become another clarification, and only `unknown` → `out_of_grid`.
- A preset can only become a bare `missing_year` clarification, and only if it would have been
  answered as that preset.
- No path creates a preset, parameters or a refusal.

By test ([`tests/test_analyst_tighten_r11.py`](../../tests/test_analyst_tighten_r11.py), synthetic
page):
- **Property test.** Every committed question (dev set and held-out v1-v5, including the
  withdrawn v2: 491 questions) plus 35 synthetic ones is crossed with 71 route kinds:
  - every routed preset;
  - every refusal reason, alone and with a clarification;
  - a refusal reason on a preset;
  - every clarification reason, bare and with a preset;
  - preset routes that are refused or clarified at run time;
  - empty routes.

  Each pair is judged on the answer the user would get: `answer_route`, then the validator's
  refusal fallback. The test checks that a refused answer is never changed, that a
  clarification stays one (only `unknown` → `out_of_grid`), that a preset answer stays as it is
  or becomes `missing_year`, and that nothing becomes a preset.
- **Router test.** `router.route` with no LLM, run over every committed item (with its `prev`):
  the outcome after the step equals the outcome before it.
- **Targeted tests.**
  - (a) and (b) through `router.route` with a scripted LLM, including a follow-up.
  - Boundary twins: a given year or definite lead keeps the preset; `how_sure` is untouched;
    on-grid values keep `unknown`; specific reasons are never re-labelled; refusals are
    untouched; a guard-refused off-grid question stays refused.
  - Labels: on every committed item the step fires only where the item's label agrees.
  - Code pins: `router/tighten.py` is in `eval.ROUTE_CODE`, and its functions are in
    `eval.ROUTE_FUNCS`.

## Effect on recorded rows (spent sets, not a score)

The step was applied to the recorded routes, with no new LLM call ($0). The table counts the
opus-low rows. The dev set has no recorded LLM rows. On every set, the no-LLM rules arm routed
live gives no change.

| set | rows | before | after | changed rows | each |
|---|---|---|---|---|---|
| v3 | 180 | 76.7 % | 78.3 % | v3_060 × 3: policy_head → missing_year (rules-stage preset, pre-R9.2 pipeline) | wrong → right |
| v4 | 180 | 85.6 % | 85.6 % | none | — |
| v5 | 270 | 86.7 % | 88.9 % | v5_034 × 3: policy_head → missing_year; v5_061 × 3: unknown → out_of_grid | wrong → right |

- No changed row went from right to wrong, and the unsafe counts are unchanged (v3 12, v4 0,
  v5 0). opus-low-all calls the LLM directly and does not pass through `router.route`. On its
  rows the same replay would change v3_060 once, v4_056 once (unknown → out_of_grid), and
  v5_034 and v5_061 3 times each, all wrong → right.
- **This is tuned on spent sets.** The step was written after reading v5's errors and the
  v3/v4/v5 items and rows. The figures above are development numbers, not a score. A fair score
  needs a new independently written held-out set. The 97 % bar is unchanged and still not met.

## Left out

- **v5_003** (policy_head expected, LLM refused `observations`) and **v5_035**
  (township_internals expected, LLM refused `subsidence_size`) are over-refusals. Fixing either
  means turning a refusal into an answer, which loosens a guard. That is out of scope under
  R11.1.
- **v4_059** ("the year our new reservoir comes online", missing_year) is a year tied to an
  unnamed event, not an indefinite lead. (a) does not read it. The LLM already clarified it.
- (b) does not read a reporting year before the page's years. That is history, not a grid
  miss. A year after the page is refused upstream (`beyond_horizon`).
- A preset routed despite an off-grid start or size is not turned into `out_of_grid`. That
  would be a tightening R11.1 allows, but it was not one of the v5 errors.
