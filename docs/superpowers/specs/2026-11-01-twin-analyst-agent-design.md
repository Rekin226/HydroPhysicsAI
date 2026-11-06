# Twin analyst agent: ask the decision page in plain language

**Date:** 2026-11-01 · **Owner:** project lead · **Replaces:** nothing (new layer; the page is unchanged) ·
**Output:** new top-level package `hydrophysics_analyst/` (CLI, local web shell, optional MCP adapter), a
private-deployment bundle format. Implementation plan:
[`../plans/2026-11-01-twin-analyst-agent.md`](../plans/2026-11-01-twin-analyst-agent.md).

**Inputs to this spec.** Three read-only surveys (page compute surface, claims ledger, LLM/packaging
conventions) and two independent architecture designs (correctness-first; UX-and-deployment-first), merged
here. Every file:line below was checked on 2026-11-01 against the working tree (which carries uncommitted
round-6 edits to the ledger, template and viewer — line numbers can drift by a few lines; search for the
function name).

---

## 0. The problem in one paragraph

The decision page (`hydrophysics/twin/viewer_app.py` → one offline HTML) answers *what does this policy change,
where, by when, how sure are we* — but only by driving sliders, tabs and drawers. Analysts (the lead and team
now; a water authority later, privately, en/zh) want to type the question and get the answer. The risk is the
opposite of the benefit: a fluent LLM sentence ("retiring aquaculture protects the rail") undoes months of claim
scoping in `claims_v1.json`. So the design is: **every answer is a page state rendered with page strings.** The
LLM routes free text to a fixed, tested preset with parameters and never writes a number or a capability
sentence. Numbers come from a numpy port of the page's own JS math running on the built page's payload;
sentences come from the page's own bilingual dictionary `T`; every answer cites claim IDs and links back into
the page at the exact state that shows the same numbers.

## 1. Design principles

- **P1 — The page is the source of truth.** The analyst binds to one built page + one ledger. It reproduces
  what that page shows (same numbers to the displayed digit, same sentences). It never recomputes from the npz
  inputs (rounding in the payload would make it disagree with the page).
- **P2 — The LLM routes; code answers.** The LLM emits only an **enum-valued** structured route (`preset`,
  `params`, `refusal` reason, `clarify` reason plus candidates). No field holds free text, so no LLM-written
  string ever reaches the user. The interpretation echo, the answer text, the numbers and the citations are
  produced by code. Project rule kept: *the API is for extraction/discovery, never for filling physical values*.
- **P2b — Where sentences come from.** Answer sentences come from the bound page's `T` dictionary. The only
  exceptions live in `hydrophysics_analyst/strings.json` (en and zh), which the lead reviews:
  - echo field labels;
  - clarify texts;
  - refusal frames;
  - banners.

  Any other sentence needs a `T` key or a ledger ruling. Preset outputs that would need new capability wording
  stay off until ruled (D6).
- **P3 — Status decides what may be said.** S/D/N quantities are answers (with the page's sentence for that
  claim); L quantities only inside a "Model internals / 模型內部數值" span the renderer labels, except the cost
  tile C08, which the ledger allows in the answer row with its own page label (`answer_tiles_allowed_note`); W quantities are
  never stated (withheld at the query layer, not just filtered); Changhua lever mentions always carry a
  `changhua_lever_rule` label. U/X do not exist in the ledger.
- **P4 — Withhold at the source.** Withdrawn quantities (`fieldAvoid`, `avoid`, area, subsidence bands, cm
  ranking) exist only inside `query._internal`; no tool schema can reach them.
- **P5 — Everything is reproducible.** Each answer records the route, tool calls, page sha, ledger sha, code
  version and model id; `replay` reproduces the answer byte-identically without the LLM.
- **P6 — The LLM is optional and swappable.** A rules parser handles most questions with no network; the LLM
  backend is a protocol (Claude API now; ZDR / cloud region / local model / none later). A deployment with no
  LLM still works through the form UI.
- **P7 — Don't touch the physics stack.** The package lives outside `hydrophysics/` (see §2). No GPU use (CLAUDE.md
  Rule 3). The public static page is not changed and gets no chat.

## 2. Where the code lives, and why

`release.provenance` hashes every `.py/.html/.js/.csv/.txt` under `Path(__file__).resolve().parents[1]`, i.e. the
whole `hydrophysics/` package (`hydrophysics/twin/release.py:41-45`), and `update_cycle`/`validation` compare that
`code_sha256` against acceptance protocols (`update_cycle.py:43`, `validation.py:131`). Any module under
`hydrophysics/` would change the code hash of every later page and break frozen protocols; twin modules are also
frozen while GPU jobs run (project memory). Therefore:

- New top-level package **`hydrophysics_analyst/`** beside `hydrophysics/`. `pyproject.toml`
  `[tool.setuptools.packages.find] include = ["hydrophysics*"]` already matches it; add package-data for its
  JSON/static files and a new extra `analyst = ["anthropic>=1.13"]`.
- It **imports** from the twin read-only: `release.read_payload` (release.py:58), `app.prep.unpack`
  (prep.py:99), `claims.load_ledger/codes/i18n/sentences/lever_mentions/numeric_paths` (claims.py:33-285). It
  never writes into `hydrophysics/`.
- Two optional later changes **do** touch `hydrophysics/` and need the lead's approval and a no-GPU-jobs window
  (§10 D5): extra URL-hash keys in the template, and `check_payload(allow_private=True)` in release.py.

## 3. Interaction surfaces

| Surface | Phase | Role | Guardrail coverage |
|---|---|---|---|
| **CLI** `python -m hydrophysics_analyst ask "…" [--lang zh] [--no-llm] [--preset … --cut aquaculture=30 …]` | 1 | owner/team; also the eval and replay harness | full (validator) |
| **Local web shell** `python -m hydrophysics_analyst serve` → `http://127.0.0.1:8765`: chat panel left, the bound private page right in a same-origin iframe driven by the URL hash | 4 | the product for the team and later the authority | full, plus a "page agrees" badge |
| **MCP server** (stdio) exposing the presets as tools for Claude Code/Desktop | 5 (optional) | expert team convenience | partial: tools return finished, validated answer blocks with "relay verbatim"; the host LLM's own prose is *not* validated → internal use only (D4) |
| Chat panel inside the page | — | rejected | breaks the offline single-file page, every new i18n key enters `display_map`, leak risk into the public build, changes `code_sha256` |

The page already accepts state: it listens for `hashchange` (`template.html:3241`) and `hashRead` takes
`p,s,y,l,lang,pins` (`template.html:643-654`). The shell drives the iframe through the hash only — no
postMessage, no network code in the page.

**A web-shell session end to end (Phase 4).** The analyst types 「養殖減30%，2030年開始，2032年水位？」 ("Aquaculture cut 30%,
starting 2030 — water level in 2032?"). The rules parser resolves it with no LLM call (class names from
`D.classes`, years, percentages, start). Code renders the echo chip *"Preset: policy head · aquaculture −30% ·
from Jan 2030 · Dec 2032 [edit]"*; each segment is a dropdown or stepper, so a correction takes one click and makes
no LLM call. The answer card shows:

- `hl_pol` filled with the N2-unit number [C01 S+D];
- `tW_r`;
- the cost line `tC_r` behind its label [C08 L];
- the provenance chip `prov_fast` (no solved run exists for 2030) [C28 L];
- claim chips with hover wording;
- "Open in page", which sets the iframe hash;
- "page agrees ✓": the shell reads `#headline` from the same-origin iframe and compares its numbers.

The "page agrees" check must wait for the page to re-render. `dataset.ready` is set only once, at boot
(`template.html:3266`), and `hashchange` → `update` → `schedule` → `requestAnimationFrame(flush)` (≈2968). So the
shell has to set the hash and then wait for `hashchange` plus two animation frames before reading the DOM, as in
the parity harness (§5).

A follow-up "same for 2026" patches `start` only. "Does it protect the HSR?" runs the rail preset, which shows
labelled internals and states that the page gives no protect/better/worse verdict.

## 4. Architecture

```
hydrophysics_analyst/
  __init__.py
  __main__.py      CLI: preset | ask | serve | replay | eval | parity (+ mcp | bundle in phase 5)
  page.py          PageBundle.load(page_html, ledger_json, mode): read payload, unpack arrays (float32),
                   T[en]/T[zh] via claims.i18n(page), ledger, binding checks (§4.1)
  jsport.py        exact ports of the JS formatters (f1, f2, sg, pctf, inv, r3, toFixed, Math.round,
                   number->string, toLocaleString) and tr(T, key, **kw)
  query.py         deterministic query layer (§5); W quantities only in query._internal
  binding.py       claim binding: display_map lookup + rules_v1.json; answer_class per value (§6)
  rules_v1.json    hand table (quantity, region, aggregation) -> claims, required_label, verdict_ban,
                   answer_class; scoring dates
  presets.py       preset registry: param JSON schema, run(bundle, params) -> [QueryResult], render(lang)
  render.py        Answer = spans (value | snippet | prose) + citations + deep link + labels
  validate.py      answer validator (§8)
  router/
    rules.py       deterministic parser (regex + dictionaries built from the payload, en/zh)
    llm.py         LLM router: one structured-output call -> Route
    schemas.py     Route schema (enum-only), per-preset param schemas
  llm/
    backend.py     LLMBackend protocol; AnthropicBackend; FakeBackend; NullBackend
    cache.py       per-call JSON cache keyed by sha256(canonical request); --no-api replay
  session.py       per-session structured state (last resolved params) — not a transcript
  audit.py         append-only JSONL audit + replay
  server.py        stdlib ThreadingHTTPServer: /, /p/<page_sha>/, /api/route, /api/answer
  static/          shell.html, shell.js, shell.css (no external requests; strict CSP on shell routes, §9)
  mcp_server.py    (phase 5, optional)
  bundle.py        private bundle build + gates (§9)
  eval.py          local-only eval harness; evalsets/eval_questions_v1.jsonl (+ held-out set)
  parity.py        Playwright parity report (local; bundle gate)
tests/test_analyst_*.py
```

**Data flow.**

1. The question goes to `router.rules`. If the rules are not confident, or the question is a follow-up they
   cannot patch, it goes to `router.llm`.
2. The router returns a `Route{preset, params, inherit, refusal, clarify}`.
3. Params are validated against the preset schema and the payload (start ∈ `meta.starts`, year ∈ `meta.years`,
   town ∈ `D.towns`, cuts on the 5 % grid).
4. Code renders the echo.
5. `preset.run` executes the query functions on the bundle.
6. `render` uses page strings to build the answer.
7. `validate` checks it, then the audit is written and the answer is returned with its deep link.

### 4.1 Binding checks (refuse to start on failure)

- (a) `payload.release.sources.forward.sha256 == ledger.page.forward_sha256` (as test_twin_claims.py:328).
- (b) The built page reverses to the pinned template. `_render` (viewer_app.py:1804-1824) makes four
  substitutions. Undo all four on the built HTML, then check that the sha equals `ledger.page.template_sha256`:
  - the payload body goes back to `__PAYLOAD__`;
  - the inline three.js `<script id="three-src">…</script>`, or the empty string, goes back to
    `<!--__THREE_INLINE__-->` immediately after the payload `</script>`;
  - the two literals in `const THREE_MODE = "…", THREE_CDN = "…"` (template.html:554) go back to
    `__THREE_MODE__` and `__THREE_CDN__`.

  `claims.i18n` takes a `Path` and reads the file itself (claims.py:107-117), so it cannot be pointed at a slice
  of a page. `page.py` parses `T` from a string with the same regex, applied only to the text **after**
  `const THREE_MODE`. In an inline build the three.js source comes before the main script and could contain a
  stray `const T = {`. A test asserts that this parser equals `claims.i18n(TEMPLATE)` on the template.

  Assert that the T literals contain no backslashes: `claims.i18n` returns raw literals with no JS unescaping.
- (c) The ledger loads strictly with `load_ledger()`. `--allow-pending` exists for development only: it adds a
  "DEV: ledger has PENDING registrations" banner to every answer and is refused in `--mode deploy`.
- (d) The query layer never reads `wells`/`leveling`, and a test asserts it. Private pages can be bound;
  individual observations are out of scope for v1.

**State on 2026-11-01**, from the lead's round-6 work in progress (not defects of this design; it changed
during the writing of this spec):

- The ledger pins `results/twin/twin_app_20261101a.html`, which is not built yet; `results/twin/twin_app.html`
  is older.
- The X5 registry hash was `PENDING` in the morning and was resolved by the evening, so strict load passes now.
- Survey noted: C31 `status_current` N vs `tests/test_twin_claims.py:546` expecting D, and a "fifth set" ruling
  cited but absent. The forward-sha check in the ledger tests is at :328.

Phase 0 develops against synthetic pages built by `va.build` (§11). The real binding is made once the lead builds
the 20261101a page.

## 5. Deterministic query layer (`query.py`)

**Port list** (JS in `hydrophysics/twin/app/template.html`; about 250 JS lines become about 300 numpy lines).

| Python | JS |
|---|---|
| `jsport.*` formatters, `tr` | 615-619, `tr`, `Number.prototype.toFixed` |
| `unpack_f32` | `unpack` 583-598 (`prep.unpack(...).astype(float32)`) |
| `solved_match` | `solvedMatch` 1357 (all factors within 1e-6, same start) |
| `policy_fields` | `policyInto` 1366 (base / solved / basis / none; `dW` from `n2Unit`) |
| `kpis` (answer subset) | `kpis` 1435-1516: `n2Head`, `yunHead`, `fieldHead`, `gwh`, `gwhPct`, `town`, `townBase`, `hsr*`, `agree`/`nRuns`, `noRun` |
| `head_where` | `headWhere` 1673 |
| `policy_phrase` | `policyPhrase` 1524 |
| `ev_line` | `evLineText` 1653 |
| `headline` | `renderHeadline` 1625 branches `hl_fit/hl_base/hl_none/hl_early/hl_pol` |
| `tiles_tW_tC` | `renderTiles` 1698 (answer row only) |
| `rank_rows` | `renderRank` 2216 (data part) + `gwhTieWords` |
| `town_rows` | `renderTowns` 2245. v1 ports only the `tw_avoided_val` row plus the `tw_low`/`tw_pooled_np` labels. The full `tn_pol` sentence needs `biasClause`, `topTowns`, `lowConf`, `pooledClause` and `townAgreeNote`; port it later, or not at all |
| `prov_chip` | `prov_full`/`prov_fast` plus `prov_nl`/`prov_assumed` (read `D.basisError`, `D.calib`) |
| `boff_words` | `boffWords`, needed by `hl_none` |
| `rail` | `hsrProfile`/`angDist` 1406-1418, `railLine`; prep.py already has `angular_distortion`, `hsr_max_by_set` |
| `start_note` | `renderStartNote` 3003 (2026 vs 2030 within max(0.01, 3 %)) |
| `deep_link` | `hashWrite` 636 (`p` as `+v.toFixed(2)` with JS number-to-string) |
| `_internal.*` | `fieldAvoid`, `avoid`, `band`, `share`, area: for the validator lexicon only; never exposed |

**API.**

```python
@dataclass(frozen=True)
class Policy:
    f: tuple[float, ...]          # f_k = float(js_num_str(+to_fixed(1 - cut_k/100, 2))): the hash-rounded value
                                  # (hashWrite, TPL:638); differs by 1 ulp from 1 - cut for cuts 55,70,80,85,90,95
    start: int                    # in meta.starts

@dataclass(frozen=True)
class Value:
    key: str
    value: float | int | None
    text: str                     # page-formatted string
    unit: str | None
    paths: tuple[str, ...]        # display_map paths, e.g. "n2Unit.basis[][]"
    functions: tuple[str, ...]    # JS functions ported, e.g. ("headWhere",)
    quantity: str                 # rule-table key: quantity, region, aggregation
    region: str
    aggregation: str
    claims: tuple[str, ...]       # filled by binding.bind()
    status: dict[str, str]        # {"C01": "S+D"}
    answer_class: Literal["answer", "labelled", "internal", "withheld", "context"]  # labelled = C08 only

@dataclass(frozen=True)
class Snippet:
    key: str                      # i18n key
    lang: str
    text: str
    claims: tuple[str, ...]

@dataclass(frozen=True)
class QueryResult:
    values: tuple[Value, ...]
    snippets: tuple[Snippet, ...]
    src: Literal["base", "solved", "basis", "none"]
    policy: Policy
    year: int
    deep_link: str
    page_sha256: str
    ledger_sha256: str
    code_version: str

def policy_fields(b, p: Policy) -> PolicyFields      # dS, dH: f32[A*Y]; dW: f32[Y] (JS Float32Array, TPL:1365); src; solved idx
# `year` arguments are calendar years (2012..2032), converted with meta.years.index(); meta.yRef is an INDEX.
def headline(b, p, year, lang="en") -> QueryResult
def head_n2(b, p, year) -> QueryResult               # tile tW
def cost(b, p, year) -> QueryResult                  # tile tC
def start_compare(b, f, year) -> QueryResult         # both starts + start_note when the page shows it
def rank(b, year, by: Literal["head", "gwh"]) -> QueryResult   # "cm" -> Withheld(C03)
def township(b, p, year, town: int | str) -> QueryResult       # C11 internals + E09 skill flag
def rail(b, p, year, thr_n=1000) -> QueryResult      # C06 internals
def evidence(b, today: date) -> QueryResult
def cheapest_for_head(b, target_m, year, start) -> QueryResult # search over page states
def deep_link(b, p, year, layer="dhead", lang="en", pins=()) -> str
```

**Matching the browser to the bit.**

- **Decoding.** Decode as the JS `unpack` does (583-598). `prep.unpack` returns float64, and it treats the
  dtype minimum as NaN for u16 and i32 too, whereas the JS does so only for i16. So `unpack_f32` mirrors the JS,
  then casts to float32.
- **Superposition.** Basis superposition runs one class at a time in JS order:
  `dS = (dS.astype(f64) + w*bS[off].astype(f64)).astype(f32)`, which mirrors `+=` into a Float32Array. **`dW`
  works the same way.** It is a `Float32Array(Y)` (TPL:1365), so `n2Unit.basis` and `n2Unit.solved` values are
  stored to float32.
- **Sums.** KPI sums are float64 in JS **sequential** order (`np.cumsum(x, dtype=f64)[-1]`, not pairwise
  `np.sum`).
- **Rounding and formatting.**
  - `toFixed` uses the exact binary value with half-up on |x|, and prefixes "-" when x < 0.
  - `Math.round` is `f = floor(x); f + (x - f >= 0.5)`. Plain `floor(x + 0.5)` is wrong at
    0.49999999999999994.
  - Minus sign is U+2212.
  - Thousands commas follow `toLocaleString("en")`.
  - Parity compares **formatted strings**, not floats.
- **`kpis` is not pure.** It reads the globals `S.thr` (area threshold, which feeds a W quantity) and `S.hsrThr`
  (rail). Neither is in the URL hash. v1 fixes `hsrThr = 1000`, the page default, and never exposes the area. The
  parity harness sets both globals explicitly before each case.

**Parity tests.**

- **CI, Python only.** Build a **real synthetic page** with `va.build`: copy `_write_inputs`/`_build` from
  tests/test_viewer_app.py (22-141; that file runs in the plain CI job, since viewer_app imports only numpy and
  pandas at module level). The default `_build` leaves most presets with nothing to test (`wells="none"` gives
  null `classes[k].gwh` and `n2Unit`; `hsr_csv=None` gives null `D.hsr`; `townships_csv=None` pools every cell
  into town 0). So:
  - pass a synthetic `hsr_csv` (pattern at test_viewer_app.py:290-301) and a synthetic `townships_csv`;
  - inject into the payload: `n2Unit` (mirror `_n2_unit_block`, viewer_app.py:681-693, with `n_registered`,
    4-decimal rounding, and `solved` ordered like `D.solved`), `classes[k].gwh`, and
    `modelcard.pumping = {"changhua": false}`.

  This gives the real template, the real `T` and real payload shapes. Use hand-computed expectations and formatter
  golden cases.
- **Local, Playwright.**
  - Setup: `importorskip("playwright.sync_api")`, reusing `_browser` from tests/test_viewer_app.py:1315-1343.
    The template is a classic `<script>` with top-level `function`/`const` names, so `kpis`, `policyFields`
    (const at TPL:1381), `D` and `S` can be called from `page.evaluate`:
    `page.evaluate("([f,si,y]) => JSON.stringify(kpis(policyFields(f,si),y))", [f, si, yIndex])`. `kpis` takes a
    year **index**.
  - Sync: set the hash, then wait for the re-render, not for `dataset.ready`:
    `h => new Promise(r => { addEventListener("hashchange", () => requestAnimationFrame(() => requestAnimationFrame(r)), {once: true}); location.hash = h; })`.
    Setting an unchanged hash fires no `hashchange`, so every case adds a nonce key `&n=<i>` (`hashRead` ignores
    unknown keys). Add a timeout, and dedupe the grid: `PRESETS` irr30/aqua0 equal the solved cut30/retire_aqua.
  - `#rankTbl` renders only on the rank tab (`renderActivePanel`, ≈TPL:2985) and depends on `S.rank`, which is
    not in the hash. The harness calls `setTab("rank")` and sets `S.rank`; Python renders the table text
    separately from the preset prose.
  - Checks: the numbers must match the Python port to within 1e-9. The DOM `textContent` of `#headline`,
    `#tW .vn`, `#tC .vn`, `#startNote` and `#rankTbl` must match the Python render with tags stripped (a few `T`
    strings contain `<b>`).
  - Grid: page presets, both solved runs and 30 seeded lever vectors on the 5 % grid; both starts; the horizon
    plus two earlier years; en and zh. That is about 300 cases, not 8k.
  - This is a **bundle gate** (§9) and runs on the synthetic page too.

**Known page quirks, kept as-is.** The analyst reproduces the page, it doesn't fix it:

- the cross-section always uses basis `dHeadEnd`;
- the alt-model answer is null for a 2030 start;
- township agreement exists only at the horizon and only for solved runs;
- only `cut30` and `retire_aqua` are solved (both from 2026), so every 2030 answer is the fast estimate and is
  labelled `prov_fast`.

## 6. Claim binding (`binding.py`, `rules_v1.json`)

Two sources must agree mechanically:

1. **`display_map`**: the union of `payload[path]`, `arrays[name]` (with `*` patterns) and `functions[fn]` over
   a value's `paths` and `functions`. Normalise paths as `claims.numeric_paths()` does (claims.py:77).
2. **`rules_v1.json`**, keyed by `(quantity, region, aggregation)`. Each entry holds `claims`,
   `required_label` (i18n keys whose rendered text must appear in the same sentence), `verdict_ban` and
   `answer_class`.

`test_rule_claims_subset_of_display_map` fails if a rule cites a claim that the display map doesn't give for
those paths. That is a ledger gap: the quantity is not answerable until the lead rules on it.

**Initial rows.**

| quantity / region / aggregation | claims | class · required label |
|---|---|---|
| head_change / n2_yunlin_wells / mean_own_layer | C01 (S+D) | answer · `tW_r` |
| head_change / yunlin_l2 or fan_l2 (headWhere fallbacks `hw_yun`/`hw_fan`) | C01 | answer, only when `n2Unit` is null; flagged |
| energy_forgone / policy / sum_classes | C08 (L) | **labelled** (answer row allowed by `page_scoping.answer_tiles_allowed_note`; `TILE_IDS…PRIMARY = 2`, TPL:1671) · the page's cost-tile title + `tC_r` |
| class_rank / head or per_gwh | C12 (L), C01 | internal · `rank_limit` (always) |
| avoided_subsidence / township / cell_mean | C11 (L) | internal · `mi_note` + `tw_avoided_val`; `tw_low` flag (E09); `tw_pooled_np` for the pooled Changhua row |
| rail_angular_distortion | C06 (L) | internal · `rail_note`; verdict_ban protect/safe/better/worse/risk |
| sign_agreement | C07 (L) | internal · provenance chip only |
| superposition_error / fast estimate | C28 (L) | internal · `prov_fast` |
| avoided_subsidence / fan; area; subsidence bands; cm rank; baseline cm | C03, C05, C14e, C09 (W) | **withheld** |
| any Changhua lever/cut | C13a/b/c (W), E20 (D) | **withheld** · C13a label |
| township head change | — | **not offered** until ruled (D3) |

`required_label` is always **the page's own sentence next to that number**. `mi_note` ("These numbers come from the
model's subsidence column…") is used only for subsidence-column values (C11, C29). It would be false for cost,
head ranking or the fast-estimate chip.

**Status → speech.**

- **S, D, N**: state the value with the page sentence for that claim.
- **L**: emit only inside a span the renderer has already labelled.
- **W**: the query returns `Withheld(claim, snippet)`, never a number.
- **Compound codes**: run `codes()`. Any W makes the value withheld; else any L makes it internal; else it is an
  answer.

**Dates and the registry.** `rules_v1.json.scoring` lists each D claim's registration and `not_before` date:
2028-01-15 for N2/N2C/E02c/N3F/N3, and 2029-01-15 for N3's 24-month test. A test asserts each date string appears
in that claim's ledger wording.

- `evidence(today)` uses `status_current`.
- If `today >= not_before` and the status is still D, the answer says *"scoring due; no result recorded in the
  ledger"* and makes no confidence statement.
- Bundles carry `wording_valid_until` (the earliest pending date), and after it every answer shows a stale-wording
  banner.
- Registry-dependent page lines are cited only if the bound page's `modelcard.registrations` flag is true
  (mirrors viewer_app.py `_registrations_block`).

## 7. Presets (v1 catalogue)

Every number comes from `query` and every sentence from the page's `T`; all are cited. The echo is code-rendered
with `policy_phrase` and a month-year (as `renderPolicySummary`).

| # | Preset | Example questions (en / zh) | Params | Output | Claims | Refuses |
|---|---|---|---|---|---|---|
| 1 | `policy_head` | "What does a 30% aquaculture cut do by 2030?" / 「養殖減30%，到2030年水位？」 ("Aquaculture cut 30%: water level by 2030?") | cuts{class: 0-100 step 5}, start, year | `hl_pol` (or `hl_early`/`hl_base`/`hl_none`), `tW_r`, cost line (labelled), `prov_full`/`prov_fast`, deep link | C01, C08, C28 | subsidence cm, area |
| 2 | `start_compare` | "Does waiting until 2030 matter?" / 「2030年才開始有差嗎？」 ("Does it make a difference if we only start in 2030?") | cuts, year | N2 head for both starts at year + horizon; `start_note` exactly when the page shows it; two-pin deep link | C01 (C10 timing wording) | "subsidence lost by waiting" |
| 3 | `lever_rank` | "Which class gives the most head rise, and per GWh?" / 「哪個類別每度電效益最高？」 ("Which class gives the most benefit per kWh?") | year, by∈{head,gwh} | rank rows, `rank_unit_*`, `rank_limit` always; `ev_decision` on "which should we choose" | C12 (L), C01 | cm ranking (C03) |
| 4 | `cheapest_for_head` (**flagged off until D6**) | "Cheapest lever for +0.5 m at N2's wells by 2032?" / 「最省電的方案讓水位上升0.5公尺？」 ("Which option uses the least electricity to raise the level 0.5 m?") | target_m, year, start | min-GWh setting over single-class 5 % cuts and uniform cuts with `n2Head ≥ target`; page value, cost, deep link. Its framing sentence is new capability wording and so needs a ledger ruling | C01, C08, C12 | "N cm" targets (C03/C27) → offers the head form |
| 5 | `township_internals` | "What happens to Tuku?" / 「土庫鎮會怎樣？」 ("What happens to Tuku township?") | cuts, start, year, town (fuzzy en/zh; ask if ambiguous) | `mi_note` + `tw_avoided_val`, `tw_low` flag, `tw_pooled_np` for the Changhua pooled row (full `tn_pol` deferred, §5) | C11 (L), E09 | township head (D3), Changhua levers |
| 6 | `rail_internals` | "Does it protect the HSR?" / 「對高鐵有保護嗎？」 ("Does it protect the high-speed rail?") | cuts, start, year (`hsrThr` fixed at 1000) | head answer (#1) then `rail_note` + rail line, labelled | C06 (L) | any yes/no or better/worse verdict |
| 7 | `how_sure` (**after v1**: needs `testRows`/`renderCard`, TPL:2575 on, ported first) | "How sure are we? When is it tested?" / 「可信度多高？」 ("How credible is it?") | — | `evLineText`, `tW_r`, `ev_decision`, the evidence-drawer rows (`evc_pro_*`, `pr_ch_untested`, …) with dates (E02, E20, E21, E22), test counts (E17) | E01, E07, E17, D rows | probabilities, "validated/accurate" |
| 8 | `solved_compare` | "Compare cut30 and retiring aquaculture." / 「比較兩個方案」 ("Compare the two options") | two solved names | `head_n2` + cost for both, `prov_full`; agreement only labelled | C01, C08, C07 (L) | avoided cm |
| 9 | `refuse` | — | reason | a refusal frame from `strings.json`, plus the page's own sentence where one exists, plus claim IDs: `changhua_lever` (C13a label), `subsidence_size` (C03/C27), `area` (C05), `recommendation` (`ev_decision`), `volumes_money`, `observations`, `beyond_2032`, `out_of_domain` (frame only), `provider_refusal` (frame only: the model provider declined) | per reason | — |

**v1 for the team** ships presets 1, 2, 3, 5, 6, 8 and 9. Preset 7 follows once `testRows` is ported, and preset
4 after ruling D6.

## 8. LLM layer and validator

**Router.**

- Rules first. Measure their coverage; the estimate is 60-80 %.
- LLM fallback is **one stateless structured-output call** per turn:
  `output_config={"format": {"type": "json_schema", …}, "effort": "low"}`. There is no tool loop and no forced
  `tool_choice` (it returns a 400 on Opus 5.5).
- Schema:
  `{preset: enum, params: anyOf per preset (additionalProperties: false), inherit_from_previous: [field], refusal: enum|null, clarify: {reason: enum[ambiguous_town, missing_year, missing_cut, out_of_grid], candidates: [enum]}|null, answer_lang: "en"|"zh"}`.
  **Every field is an enum**, and code renders the clarify text from `strings.json`. Structured outputs
  supports `anyOf` but not `oneOf`, `minimum`/`maximum`/`multipleOf`, and every object needs
  `additionalProperties: false`. Hence:
  - cut percents are `enum [0, 5, …, 100]`;
  - years are an enum of `meta.years`;
  - cuts are an array of `{class: enum, pct: enum}`, not an object with free keys.

  Verify these limits against the current structured-outputs docs when implementing.
- **Hard refusal triggers override the LLM.** They run on the raw question and win over any route, rules or
  LLM:
  - **any** Changhua mention: `\bChanghua\b|彰化` or a pooled-town name;
  - subsidence cm, area;
  - money, volumes;
  - years after 2032.

  The only exemption is `township_internals` asking for the pooled row, which carries `tw_pooled_np`. The ledger's
  `changhua_lever_rule.mention` regex needs "lever/cut" (槓桿/減抽) in the same sentence, so it would miss "reduce
  Changhua aquaculture 30 %". It is used to check **output** only, not to trigger refusals.
- **The echo always names the quantity and its scope**, so a misroute is visible. The scope is rendered as a
  page snippet span (`hw_n2`), because a digit such as "84" in an echo span would fail validator rule 1. For a
  pooled Changhua row, the town chip carries the C02 label (「並非彰化槓桿」 / "not a Changhua lever"), and a
  sentence break separates the policy and town segments. Otherwise the zh `pp_cut` (「{c}減抽{p}%」) plus 「彰化縣」
  would trip `lever_mentions`.
- The user message is the question plus the **previous resolved params as JSON**. So "now for 2030" is a patch,
  no transcript is replayed, and preserved-thinking history rules don't arise.
- The router sees no payload numbers.
- System prompt (frozen bytes, cached):
  - the role ("route; never compute");
  - the project rule;
  - status-code semantics;
  - preset docs with examples;
  - enums from the payload (classes, towns, years);
  - refusal reasons;
  - for each claim: id, status and a **code-written, digit-free topic tag** from `rules_v1.json`.
- What stays out of the prompt: the ledger's `claim` one-liners, which carry W numbers (C03 "cut30 0.94 cm…",
  C09 "2.9 cm"); `wording_en`; `notes`.
- A test asserts the system prompt's digits are only `meta.years`, `starts`, claim ids (`[CEZ]\d+[a-e]?`) and
  grid percents, and that no number token from any ledger `claim` or `wording_en` (e.g. "0.94", "2.9") appears.
- Today's date and the page and ledger shas go in a trailing block so the cache prefix holds.

**Backends.**

- `LLMBackend.complete(system, messages, tools, output_format, effort) -> NormalizedResponse(text, tool_calls, stop_reason, usage, model, request_id)`.
- Implementations:
  - `AnthropicBackend(client_factory)`: first-party `anthropic.Anthropic()`, or `AnthropicBedrockMantle` /
    `AnthropicVertex` / `AnthropicFoundry` chosen by config.
  - `FakeBackend(script)` for tests.
  - `NullBackend`: rules plus the form only.
  - A local-model backend later, behind the same protocol.
- The key comes from the environment or from the repo's hand-rolled `load_key()` reading `.env` (pattern in
  `dataprep_staging/x4_canal_leveling/lev_extract.py:122-130`). It is never logged.
- Check `stop_reason == "refusal"` before reading content.
- Server `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) is on for team use on the first-party
  API with Opus 5.5.
  - Omit it for Haiku 5.5, which has no server-side fallback, and on Bedrock, Vertex and Foundry, which use client
    middleware instead.
  - Turn it off in deployment, where the model list must be approved.
- Default model: `claude-opus-5-5` at low effort, until the eval (Phase 3) says otherwise.
- **Egress:** in preset mode only the question text and enum state leave the machine, never tool outputs or
  payload data.

**Free-form answers: deferred (not in this plan).** A digit tokeniser cannot guard LLM prose:

- number words ("thirty", 三十, 一半, "twice") pass;
- verdict synonyms ("shields the rail", "fine for the HSR", "best choice") pass.

If free-form is ever wanted, its prose must come from a closed list of connectives, and it needs its own design
and a ledger ruling. Out-of-preset questions get a refusal that lists the presets.

**Validator** `validate(answer, results, bundle, lang) -> list[Violation]`. It runs on every answer, presets
included, as a regression net.

1. **Numbers.** Value and snippet spans are trusted by construction. Every other span (`strings.json` frames,
   labels, echo) may contain **no** number tokens.
   - Tokeniser: `_LIT`-style (claims.py:120), extended for U+2212, %, `1/12,345`, exponents, commas and
     full-width digits.
   - Exempt: years in `meta.years`/`starts`, and lever percents equal to the params.
2. **Banned strings.**
   - `page_scoping.template_absent` applies to **all** spans, along with the rendered text of
     `headline_must_not_use` and `answer_must_not_use_band_keys`.
   - The prose lexicon (validated, accurate, proven, guarantee, will stop, predict, recommend, safe, protect,
     better/worse) applies **only to non-page spans** (`strings.json`, echo, labels). About 33 page `T` strings
     legitimately contain those words in negated or scoped form (`ev_decision`, `rank_limit`, `rail_note`, …).
3. **Withheld.** No withheld value is referenced, and no sentence pairs a withheld quantity name with a number.
4. **Labels.** Every sentence with an internal value contains its `required_label` text.
5. **Changhua.** `claims.lever_mentions(...) == []` on the **whole composed text, echo included**, not span by
   span.
6. **Citations.** The cited ids equal the union of the claims of the values and snippets used. Decorative
   citations are rejected, e.g. an S claim beside an L number.
7. **Dates.** No stale D claim is presented as current.
8. **Reproduction.** Re-running the **logged route** reproduces the numbers. The hash carries only
   `p, s, y, lang`; township, rank-by and the like come from the route.

On failure: a preset render that fails is a bug. The answer becomes a refusal plus the deep link, marked
`fallback: validator`, and every violation is logged. A failure is never silent.

**Cache, audit, cost.**

- LLM calls are cached in `results/analyst/calls/<sha>.json` with request, response, usage, model, effort and the
  page and ledger shas. `--no-api` replays from the cache, mirroring `report_extract.py:293-311`.
- The audit log `results/analyst/audit/YYYY-MM-DD.jsonl` records:
  - request id, user, language, question;
  - route and any corrections;
  - each tool call (name, args, output sha);
  - answer, claims, validator verdict, deep link;
  - page sha, ledger sha, strings sha, code version;
  - model id, provider request id, usage, latency.
- `replay <id>` re-runs the tools from the logged route without the LLM and must reproduce the answer
  byte-identically.
- `results/` is gitignored. Questions may be sensitive, so there is a retention setting.
- Targets:
  - Rules path: under 300 ms.
  - LLM-routed: p50 under 3 s on Haiku 5.5, under 5 s on Opus 5.5 at low effort.
  - Cost per routed question: about $0.01-0.02 on Opus 5.5 with a cached prefix, about $0.001 on Haiku 5.5.

## 9. Private deployment

**What ships.**

- `bundles/<page_sha>/` with `page.local.html` (the `--private` build), `claims_v1.json` and `bundle.json`.
  `bundle.json` holds: page sha, release block, ledger sha, `T` strings sha, parity report sha, eval report sha,
  `built_utc` and `wording_valid_until`.
- A `current.json` pointer, switched atomically as in `release.publish`.
- The service: a wheel of `hydrophysics-ai[analyst]` (numpy and pandas; no torch), or a `python:3.11-slim`
  container with the bundle mounted read-only.

**`bundle` gates**, all of which must pass:

- binding checks (§4.1) in deploy mode;
- strict ledger;
- Playwright parity report;
- the golden eval at or above its threshold;
- the claims tests green.

A failed gate keeps the previous bundle (the same rule as release.py: failed checks never replace
`current.json`). Publishing a private page through release.py needs `check_payload(allow_private=True)` (D5). Until
then `bundle` does its own private-aware checks and leaves release.py untouched.

**Hosting.**

- An on-prem VM; never internet-facing.
- It binds 127.0.0.1 by default. A team install sits behind the authority's reverse proxy (TLS, AD/OIDC SSO,
  `X-Remote-User`); the fallback is hashed per-user tokens in `config.toml`.
- CSP: strict `default-src 'self'` on the shell routes. On `/p/<sha>/` it is
  `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; frame-ancestors 'self'`,
  because the page needs its inline script, styles and `data:` images (TPL:1806).
- Old bundles stay served so old links and audits still resolve.

**Data governance by backend**, decided with the authority (D7):

- **Anthropic API:** under `router_only` egress, only the question text and state leave. `inference_geo` is
  global or us only.
- **Anthropic zero-data-retention agreement:** confirm model eligibility. Fable models are not ZDR-eligible.
- **Bedrock or Vertex in an approved region:** availability in a Taiwan region is **unverified**.
- **Local model on CPU:** routing only; never on the physics GPU.
- **None:** the rules parser plus the form.

**Update cycle.**

1. New build.
2. `browser_check` plus the claims tests.
3. `release publish` (public) and/or `hydrophysics_analyst bundle` (private).
4. The server reloads when `current.json` changes.

## 10. Open decisions for the lead (defaults in **bold**)

- **D1 — Package location:** **`hydrophysics_analyst/`, outside the hashed `hydrophysics/`**, versus
  `hydrophysics/analyst/` (which changes `code_sha256`).
- **D2 — W numbers:** **never stated; link to the page's Model internals**, versus stated with the page's
  withdrawn label.
- **D3 — Township head change:** **not offered until the ledger maps it** (C02 or C14c, or a new row).
- **D4 — MCP surface without the validator on host prose:** **internal team only**, or not built.
- **D5 — Changes inside `hydrophysics/`:** extra hash keys `sel`, `tab`, `mi` in the template (about 10
  lines, no new i18n keys; **not** `thr`, which is the W area threshold), and
  `check_payload(allow_private=True)`. **Deferred to Phase 5, in a no-GPU-jobs window, with browser_check.**
- **D6 — `cheapest_for_head` wording** ("smallest setting reaching a target"): needs a ledger ruling. **Ship it
  behind a flag until ruled.**
- **D7 — Authority deployment:** LLM backend, egress mode, audit retention, who reviews the zh terminology.
- **D8 — Development binding:** **synthetic `va.build` pages until the 20261101a page is built**, then bind to
  it. Keep `--allow-pending` for days when the ledger is mid-edit.
- **D9 — zh from Phase 1:** **yes.** It is free because sentences come from the page's `T[zh]`, and the rules
  parser needs zh names anyway.

## 11. Build, tests and acceptance

- **CI-safe tests** (no anthropic and no playwright in CI; helpers copied into each file; no `tests.` imports;
  no conftest):
  - `test_analyst_jsport.py`: formatter golden cases (`toFixed(0.125, 2) == "0.13"`,
    `sg(-0.004) == "−0.00"`, …).
  - `test_analyst_query.py`: synthetic `va.build` page with injected `n2Unit`; superposition by hand; solved snapping; float32 accumulation;
    deep-link round trip; no reads of `wells`/`leveling`.
  - `test_analyst_binding.py`: against the committed ledger, loaded with `allow_pending=True` and checked for
    structure only. Checks that rule claims ⊂ display_map; that every `required_label` exists in en and zh; that
    scoring dates appear in the wording; that no withheld quantity is reachable from any schema; and that the
    Changhua refusal passes `lever_mentions`.
  - `test_analyst_validate.py`: a red-team corpus that must be 100 % rejected, and every preset render on the
    synthetic grid, which must have 0 false rejects.
  - `test_analyst_router.py`: rules parser cases in en and zh; `FakeBackend` scripts for routing, bad params
    leading to a clarification, an LLM route that a hard refusal trigger must override (Changhua, cm), a refusal `stop_reason`,
    and cache replay; `importorskip("anthropic")` only for the SDK adapter.
  - `test_analyst_server.py`: stdlib server with no network beyond localhost; both CSP variants; the routes.
- **Local only:** `test_analyst_parity.py` (Playwright; runs on the synthetic page, and on the real page when
  `ANALYST_PAGE` is set) and
  `python -m hydrophysics_analyst eval`.
- **Acceptance per phase:** see the plan. Headline gates:
  - parity is string-identical on the full grid;
  - the validator rejects 100 % of the red-team corpus with 0 false rejects;
  - router accuracy is at least 97 % and adversarial refusal recall is 100 % for the chosen model;
  - every audited answer replays byte-identically.

## 12. Out of scope

- Any change to the public page, or a chat inside any page.
- LLM-produced physical values; LLM-written prose of any kind, including free-form answers (§8).
- New physics quantities.
- Individual well or benchmark answers.
- Money and volumes.
- Answers beyond Dec 2032, or starts other than `meta.starts`.
- Multi-class optimisation beyond the preset-4 search.
- GPU-hosted or local LLMs before the authority asks for one.
- Internet deployment.
- Editing the ledger: the analyst consumes it; rulings stay with the lead.

## 13. Amendments made while building (2026-11-01)

**Rulings.** The lead decided:
- D2: never state W numbers; link only.
- D3: township head not offered.
- D4: MCP built, internal only.
- D5: both `hydrophysics/` changes approved.
- D6: cheapest built, flagged off.
- D7: the authority deployment targets the Anthropic API under ZDR.
- D8: wait for the lead's 20261101a build.
- Eval arms: rules plus Opus 5.5 low.
- D1 and D9: defaults kept.

**Citations.** These are the claims governing what is *stated*: the values shown, a refusal's own
sentence, and the evidence lines of `how_sure`. W claims are excluded. Caveat sentences that only
mention claims (`mi_note`, `rank_limit`) add none.

**Clarifications.** These carry `preset: "clarify"`, so the reproduction rule does not re-run them
as a preset.

**Missing parameters.** These take the page's own defaults (`S.start` = first start,
`S.year` = horizon, `S.rank` = head), the same for the rules and the LLM. A rail, township or
start-compare question with no cut is `clarify missing_cut`, never the baseline.

**Validator.**
- The verdict and withheld lexicons are checked in both languages whatever the answer language.
- A refusal frame may name the withdrawn quantity it refuses.

**`how_sure`.** It ports a stable subset of `evidenceItems`: the evidence line, `ev_decision`,
`tW_r`, the registered prospective-test lines and the registry-flag lines.
- `evc_pro_iv` is excluded: it is a band key the ledger keeps out of answers.
- Ring, pile, hybrid and coverage lines stay in the page's drawer.
- Parity checks that every `how_sure` sentence appears in the page's evidence line, drawer or tiles.
- *Round 8 (plan #19, 2026-11-07):* the query layer now mirrors the whole drawer and the head
  band for parity, line by line both ways with claims; answers drop the ledger's band keys and
  `presets.DRAWER_ONLY` (ring, pile, hybrid, coverage as above; Changhua's withdrawn levers,
  ruling D2; the wells' note). The tile line is the default view's (`tW_w30` with the all30
  run). Of the R8.13 band, answers say only that it is the two-part band.

**D5.** The patches were prepared and tested in a scratch copy of the working tree, not applied:
a twin forward job was running. They are in `docs/analyst/d5/`, with `apply_d5.sh`, which refuses
while a twin job runs. The analyst already emits `sel`/`tab`/`mi`, which older pages ignore.

**Later rulings (2026-11-01, after the first build).**
- D5: the lead applies `apply_d5.sh` themselves.
- D3: township water-level questions are refused (`township_head`) by the rules and the LLM
  alike.
- The question sets are versioned under `evalsets/`.
- D6: cheapest stays off.

**Shell UX (lead's request, 2026-11-01).**
- **Example-question chips** come from `presets.examples(b)`: preset plus params, with labels
  built from page strings. They are answered without the LLM.
- **Clarification candidates** are clickable.
- **In-chat SVG charts** are a `Chart` that holds only references (`r<i>:<value key>`) into the
  answer's own bound values.
  - Head change at N2's wells by year (one line per policy or start; start-year hairlines).
  - Ranking bars (internals box).
  - Full-run comparison bars (head, cost).
  - A cost share bar.
- **What charts never show:** subsidence (W) and bands (`evc_band_none`).
- **Validator rule 9:** points resolve to answer values, no withheld value, claims cited, titles
  are page strings, internals only in the internals block.
- **Style:** the dataviz method. Palette slots 1-2 validated (CVD ΔE 24.7, contrast ≥ 3:1), 2 px
  lines, ringed end-dots, bars ≤ 24 px with 4 px data-ends, a legend for ≥ 2 series, hover
  tooltips, a screen-reader table per chart.

**Deployment config.** It is a JSON file (`docs/analyst/analyst_config.example.json`), not TOML,
so Python 3.10 needs no extra parser. Users map to the sha256 of their token. Audit retention is
applied at server start.

**Rules router.** It was tuned on `eval_questions_v1.jsonl`, so its numbers on that set are
optimistic. The independent held-out set `eval_questions_heldout_v1.jsonl` was written without
seeing v1 or the code. The eval report gives both (`docs/analyst/EVAL_2026-11-01.md`).

## 14. References

- Page: `hydrophysics/twin/viewer_app.py`, `hydrophysics/twin/app/template.html`, `hydrophysics/twin/app/prep.py`.
- Specs: `docs/superpowers/specs/2026-09-23-twin-decision-app-redesign.md`, `2026-10-11-twin-app-minimal-redesign.md`.
- Ledger: `hydrophysics/twin/claims_v1.json`, `hydrophysics/twin/claims.py`, `tests/test_twin_claims.py`.
- Release: `hydrophysics/twin/release.py`, `docs/TWIN_RELEASE.md`, `hydrophysics/twin/browser_check.py`.
- Existing LLM code: `hydrophysics/twin/data_scout.py`, `hydrophysics/twin/report_extract.py`.
- Hardware and stack constraints: `docs/GPU_SERVER.md`.
