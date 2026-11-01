# Twin Analyst Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for
> tracking. Read the spec first; it holds the reasoning, this plan holds the order and the contracts.

**Goal:** Let analysts ask the twin decision page questions in plain language (en/zh) and get answers that are
string-identical to what the page shows at a deep-linked state, cite the claims ledger, and contain no
LLM-written text at all. That rules out an LLM number, an LLM capability sentence and an LLM clarification.

**Architecture:**
- A new top-level package `hydrophysics_analyst/`. It sits outside `hydrophysics/` so the page and release
  `code_sha256` don't change.
- It binds to one built page and one ledger, and ports the page's runtime JS math to numpy (`query.py`).
- It renders answers from the page's own `T` dictionary, binds every value to claim IDs (`display_map` +
  `rules_v1.json`), and runs a validator on every answer.
- A rules parser routes most questions. One Claude structured-output call with **enum-only** fields routes the
  rest. Hard refusal triggers override any route.
- Surfaces: CLI (v1) → local web shell (chat + iframe driven by URL hash) → optional MCP / private bundle.

**Tech Stack:** Python ≥3.10, numpy, stdlib (`http.server`, `json`, `decimal`, `hashlib`), `anthropic>=1.13`
(optional extra `analyst`), pytest, ruff, Playwright (local parity only).

**Spec:** `docs/superpowers/specs/2026-11-01-twin-analyst-agent-design.md`

**Status (2026-11-01, branch `twin/analyst-agent`):**
- **Done:** Phases 0–5, plus the Phase-6 deployment pieces that need no authority input (ZDR
  config, data-governance memo template, OPS, Containerfile).
- **Results:** parity shows 0 mismatches on synthetic and real pages. Eval:
  `docs/analyst/EVAL_2026-11-01.md` (held-out pipeline 90.0 % with 3 unsafe before the fixes; one
  post-fix follow-up at 93.3 % with 0 unsafe).
- **Waiting:**
  - Tasks 23–24 (D5): prepared in `docs/analyst/d5/`, not applied, because a twin GPU job was
    running.
  - Real-page binding: waits for the lead's 20261101a build.
  - Task 26: needs the authority.

## Global Constraints

- **Never create or edit files under `hydrophysics/`** in Phases 0-4. `release.provenance` hashes the whole
  package (`hydrophysics/twin/release.py:41`).
  - Phase-5 tasks that touch it are marked **[HYDROPHYSICS EDIT]**. They need the lead's approval (spec D5).
  - They also wait for a window with no twin GPU jobs:
    `nvidia-smi --query-compute-apps=pid,process_name --format=csv`.
- **No LLM-written string reaches the user.**
  - Route fields are enums or numbers on the slider grid.
  - Value and snippet spans come from code.
  - Frames come from `strings.json`.
  - If a change needs the LLM to write user-visible text, the design is wrong.
- **W-status quantities never leave `query._internal`.** No tool schema, preset output or render path may
  reference them.
- **Where sentences come from.** Answer sentences come from the bound page's `T` (`claims.i18n`). Exceptions,
  all in `hydrophysics_analyst/strings.json` (en + zh) and reviewed by the lead: echo labels, clarify texts,
  refusal frames, banners. Any other sentence needs a `T` key or a ledger ruling.
- **CI has no `anthropic` and no `playwright`** (`.github/workflows/ci.yml` installs `.[dev,explorer]`).
  - Tests use `FakeBackend` and synthetic pages built with `va.build`. `viewer_app`, `claims`, `release` and
    `app/prep` import only stdlib, numpy and pandas, so they run in the plain CI job.
  - SDK-adapter tests use `pytest.importorskip("anthropic")`.
  - Parity tests use `importorskip("playwright.sync_api")`.
- **Test conventions:**
  - no `tests/__init__.py`, no `conftest.py`, no `from tests.` imports; copy helpers into each test file;
  - tests are CPU-only and fast (analyst suite under 60 s, including `va.build` of tiny grids);
  - block outbound sockets. The pattern in `tests/test_viewer_app.py:115-124` blocks **all** `connect` calls, so
    server tests need a variant that allows `127.0.0.1`.
- **Every module starts with `from __future__ import annotations`.** ruff line-length 100, rules
  `E4,E7,E9,F,I,B,UP,SIM`. CLI modules expose `main(argv=None)`, use `argparse` with `description=__doc__`, and
  put usage examples in the module docstring. Every subpackage has an `__init__.py`, otherwise `packages.find`
  skips it.
- **Secrets and data:**
  - the API key comes from the environment or the `load_key()` pattern
    (`dataprep_staging/x4_canal_leveling/lev_extract.py:122-130`); never printed or logged;
  - caches and audit logs go under `results/analyst/` (gitignored);
  - private pages (`*.local.html`) and bundles are never committed;
  - run the CLAUDE.md Rule 1 secrets scan before every commit.
- **Model ids:** `claude-opus-5-5` (default, `effort: "low"`) and `claude-haiku-5-5` (eval arm).
  - No `budget_tokens`, no `thinking: disabled` on Opus 5.5, no forced `tool_choice` (400).
  - Structured output via `output_config.format`.
  - Check `stop_reason == "refusal"` first.
  - `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) is used only on the first-party API with Opus
    5.5. Never with Haiku 5.5, and never on Bedrock, Vertex or Foundry.
- **Reproduce the page, don't fix it.** Known quirks are kept (spec §5). If the port disagrees with the page, the
  port is wrong.
- **The ledger is read-only for this work.** It is being edited actively by the lead. If something needs a claim
  row or wording, stop and list it in spec §10. Never edit `claims_v1.json`.
- **APIs that take a `Path`:** `claims.load_ledger`, `claims.i18n` and `release.read_payload` take
  `pathlib.Path`.

## Phase map

| Phase | Tasks | Acceptance gate | Rough effort |
|---|---|---|---|
| 0: Page binding and query layer | 1-6 | CI green; Playwright parity string-identical on the grid (synthetic page now, real page once built) | 4-5 d |
| 1: Binding, presets, validator, CLI without LLM | 7-11 | `preset …` works en/zh; red-team 100 % rejected; 0 false rejects | 4 d |
| 2: Routers, backend, cache, audit (**v1 for the team ends here**) | 12-15 | FakeBackend tests in CI; `ask "…"` over the API locally | 3 d |
| 3: Eval and model choice | 16-17 | written model decision with numbers | 1-2 d + API cost (~$5-15) |
| 4: Local web shell (once the CLI shows demand) | 18-20 | one-click correction without LLM; "page agrees" badge; CSP; localhost | 3-4 d |
| 5: Optional: `how_sure`, MCP, hash keys, private bundle | 21-25 | per task, each needs a go-ahead | 4-7 d |
| 6: Authority pilot | 26 | authority eval ≥ threshold; data-governance memo | external |

---

## Phase 0: Page binding and query layer

### Task 1: Package skeleton and packaging

**Files:**
- Create: `hydrophysics_analyst/__init__.py`, `hydrophysics_analyst/__main__.py` (subcommand dispatcher with
  stubs), `hydrophysics_analyst/router/__init__.py`, `hydrophysics_analyst/llm/__init__.py`.
- Modify: `pyproject.toml`:
  - `[project.optional-dependencies] analyst = ["anthropic>=1.13"]`;
  - `[tool.setuptools.package-data] hydrophysics_analyst = ["rules_v1.json", "strings.json", "static/*", "data/*.jsonl"]`.
  - Leave `packages.find include = ["hydrophysics*"]` alone; it already matches.
- Test: `tests/test_analyst_package.py`

**Interfaces:** `hydrophysics_analyst.__version__: str`. `python -m hydrophysics_analyst --help` lists
`preset, ask, serve, replay, eval, parity` (plus `mcp` and `bundle` once Phase 5 lands).

- [ ] Step 1: Write failing tests:
  - the package imports with `sys.modules["anthropic"] = None`;
  - `main(["--help"])` exits 0 and lists the subcommands;
  - **path containment**: the package dir is not inside `Path(release.__file__).resolve().parents[1]`, so it
    cannot enter `code_sha256`;
  - `setuptools.find_packages(include=["hydrophysics*"])` contains `hydrophysics_analyst.router` and
    `hydrophysics_analyst.llm`.
- [ ] Step 2: Implement; `pip install -e ".[dev,analyst]"` in env `hydro`.
- [ ] Step 3: `ruff check . && pytest -q tests/test_analyst_package.py`; commit.

### Task 2: JS formatter port (`jsport.py`)

**Files:** Create `hydrophysics_analyst/jsport.py`. Test: `tests/test_analyst_jsport.py`.

**Interfaces:**
- `to_fixed(x, d) -> str`
- `js_round(x) -> int`
- `js_num_str(x) -> str` (JS `String(number)`)
- `locale_en(x) -> str`
- `f1, f2, sg, pctf, inv, r3`, mirroring `template.html:615-619` exactly (read those lines first)
- `tr(T, lang, key, **kw) -> str`, mirroring the page's `tr` (find `function tr(`):
  - a missing language falls back to `T.en`, then to the key itself;
  - an undefined substitution value becomes `"undefined"`;
  - numbers are inserted with JS `String()`.

Reference semantics:

```python
from decimal import Decimal, ROUND_HALF_UP
def to_fixed(x: float, d: int) -> str:          # Number.prototype.toFixed for |x| < 1e21
    # ECMA-262: if x < 0, emit "-" and format -x; the digits come from the exact binary value,
    # ties rounded up. So (-0.004).toFixed(2) == "-0.00" but (-0).toFixed(2) == "0.00".
    q = Decimal(abs(x)).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return ("-" if x < 0 else "") + f"{q:.{d}f}"

def js_round(x: float) -> int:                   # Math.round; floor(x + 0.5) fails at 0.49999999999999994
    f = math.floor(x)
    return f + (1 if x - f >= 0.5 else 0)
```

- [ ] Step 1: Write failing golden tests:
  - `to_fixed(0.125, 2) == "0.13"`, `to_fixed(1.005, 2) == "1.00"`, `to_fixed(-0.004, 2) == "-0.00"`,
    `to_fixed(-0.0, 2) == "0.00"`;
  - `js_round(0.49999999999999994) == 0`, `js_round(-0.5) == 0`, `js_round(2.5) == 3`;
  - `js_num_str(0.7) == "0.7"`, `js_num_str(1.0) == "1"`;
  - `locale_en(12345.6)`;
  - `sg` uses U+2212 (copy what the page does for tiny negatives);
  - the `tr` fallbacks.
  - Mark any case not provable from the JS source with `# confirm in Task 6`.
- [ ] Step 2: Implement; run; commit.

### Task 3: Synthetic page builder for tests

**Files:** Create `tests/test_analyst_page.py`. The builder is a module-level helper, copied into later test
files that need it.

**Interfaces:**
- `_build_page(tmp_path, monkeypatch) -> Path`. Copy `tests/test_viewer_app.py:22-141`: the constants, `_mask`,
  `_write_inputs`, `_build`, and the body of the `quiet` fixture (monkeypatches of
  `va._ground_idw`/`va._load_inputs` and the socket block). `quiet` is a fixture, so the helper takes
  `monkeypatch`.
- Build with `va.build(...)` and **also pass** a synthetic `hsr_csv` (pattern at test_viewer_app.py:290-301) and a
  synthetic `townships_csv`. Otherwise `D.hsr` is null and every cell pools into town 0.
- Then **inject** into the payload `<script id="payload">`:
  - `n2Unit`, mirroring `_n2_unit_block` (viewer_app.py:681-693): `n`, `n_registered`, `solved` ordered like
    `D.solved`, `basis`, all rounded to 4 decimals;
  - `classes[k].gwh` (null under `wells="none"`, viewer_app.py:604);
  - `modelcard.pumping = {"changhua": false}`.
- Re-serialise the payload with the same `json.dumps(..., allow_nan=False)` and `</` escaping as `_render`.
- The result is a real page with the real template, real `T`, real payload shapes and both solved runs.

- [ ] Step 1: Write the helper plus tests:
  - `release.read_payload(page)` has `n2Unit`;
  - `D.hsr`, `classes[0].gwh` and at least two non-pooled towns exist;
  - no `T` literal contains a backslash.
- [ ] Step 2: Commit.

### Task 4: `PageBundle` (`page.py`)

**Files:** Create `hydrophysics_analyst/page.py`. Test: extend `tests/test_analyst_page.py`.

**Interfaces:**

```python
@dataclass
class PageBundle:
    payload: dict
    T: dict[str, dict[str, str]]
    ledger: dict
    claims_by_id: dict[str, dict]
    page_sha256: str
    ledger_sha256: str
    strings_sha256: str
    mode: Literal["dev", "deploy"]
    banners: list[str]
    def array(self, name: str) -> np.ndarray: ...  # unpack_f32, mirrors JS unpack (583-598); raises for wells/leveling
    @classmethod
    def load(cls, page_html: Path, ledger_json: Path | None = None, mode="dev", allow_pending=False,
             check_forward=True) -> PageBundle: ...
class BindingError(RuntimeError): ...
```

Binding checks (spec §4.1):
- **(a) Forward sha.** It matches `ledger.page.forward_sha256`.
- **(b) Template reversal.** Reverse `_render`'s four substitutions (viewer_app.py:1804-1824):
  - the payload body back to `__PAYLOAD__`;
  - the `three-src` script or "" back to `<!--__THREE_INLINE__-->`;
  - the two literals at template.html:554 back to placeholders.

  The sha must then equal `ledger.page.template_sha256`.
- **(c) Strict ledger.** Load it strictly. `allow_pending` adds a banner in dev and is refused in deploy.
- **(d) No individual observations.** `array("wells")` and `array("leveling")` raise.
- **`T` extraction.** `parse_T(text, lang)` uses the regex of `claims.i18n` (claims.py:107-117), which itself
  takes a `Path`, on the text after `const THREE_MODE`. It asserts there are no backslashes in the literals.

- [ ] Step 1: Write failing tests on the synthetic page:
  - loads; `array()` dtype is float32 and matches `prep.unpack` except for the NaN rule on non-i16;
  - `parse_T(TEMPLATE text) == claims.i18n(TEMPLATE)` for en and zh;
  - check (b) passes on an untouched page and fails after a one-byte edit to the JS. Use a ledger copy whose
    `template_sha256` is the working-tree template's sha (today they match, as test_twin_claims.py:151 asserts);
  - a mismatched forward sha raises (the synthetic page passes `check_forward=False` in dev only; deploy mode
    forbids it);
  - `allow_pending` in deploy mode raises;
  - `array("wells")` raises.
- [ ] Step 2: Implement.
- [ ] Step 3: Against the **real** ledger, load a synthetic page with `check_forward=False`. If the
  working-tree template has drifted from the pin, check (b) fails there; skip that one assertion with a clear
  reason rather than weakening the check. Commit.

### Task 5: Query layer (`query.py`)

**Files:** Create `hydrophysics_analyst/query.py`. Test: `tests/test_analyst_query.py`.

**Interfaces:** exactly as spec §5. Produces:
- the types `Policy`, `Value`, `Snippet`, `QueryResult`, `PolicyFields`, `Withheld`;
- the functions `policy_fields`, `headline`, `head_n2`, `cost`, `start_compare`, `rank`, `township`, `rail`,
  `deep_link`, `prov_chip`, `boff_words`;
- `_internal.*`;
- `evidence` and `cheapest_for_head` as stubs that raise `NotImplementedError` (Phase 5 and D6).

Rules for the port:
- Before writing a function, read its JS (line refs in spec §5) and port it line by line, in the same summation
  order.
- **Years:** API `year` arguments are calendar years, converted with `meta.years.index(year)`. `meta.yRef` is
  already an index.
- **Factors:** `Policy.f` holds the hash-rounded value `float(js_num_str(+to_fixed(1 - cut/100, 2)))` (hashWrite,
  TPL:638). For cuts 55, 70, 80, 85, 90 and 95 it differs by 1 ulp from `1 - cut/100`. Answers claim parity with
  the linked state, so use the hash value.
- **Answer class:** C08 values get `answer_class = "labelled"`.
- **Globals:** `kpis` reads globals. Pass `hsr_thr=1000` (the page default) explicitly, and never compute the
  area KPI outside `_internal`.
- **Township:** in v1 return only the `tw_avoided_val` value with `mi_note`, and the `tw_low` and
  `tw_pooled_np` labels. Not the full `tn_pol` sentence.
- **Claims:** leave `Value.claims` and `status` empty; Task 7 fills them. Fill `paths` and `functions` now with
  `display_map` canonical spellings (`[]` for list elements; claims.py:77).

Float32 rule:

```python
def _accumulate(acc: np.ndarray, w: float, row: np.ndarray) -> np.ndarray:
    # JS: Float32Array acc; acc[i] += w * row[i]  -> double math, float32 store. Used for dS, dH AND dW.
    return (acc.astype(np.float64) + w * np.asarray(row, dtype=np.float64)).astype(np.float32)

def _jsum(x: np.ndarray) -> float:               # JS sequential double sum, not pairwise
    return float(np.cumsum(x, dtype=np.float64)[-1]) if x.size else 0.0
```

- [ ] Step 1: Write failing tests on the synthetic page:
  - all factors 1 → `src == "base"`;
  - a solved match within 1e-6 → `src == "solved"` and `dW == np.float32(n2Unit.solved[m])`;
  - a basis policy equals the float32 hand superposition;
  - no basis → `src == "none"`;
  - `head_where` picks `n2Head`, then `yunHead`, then `fieldHead`;
  - `cost.gwh == Σ(1-f_k)·gwh_k`;
  - `rank("cm")` returns `Withheld("C03")`;
  - `deep_link` round-trips through a Python port of `hashRead` (length K, entries in [0, 1]);
  - `start_compare` emits `start_note` exactly inside the page tolerance;
  - walking every field of every public `QueryResult` finds no `_internal` value.
- [ ] Step 2: Implement until green. `ruff`; commit.

### Task 6: Playwright parity gate (local)

**Files:** Create `hydrophysics_analyst/parity.py` (`python -m hydrophysics_analyst parity --page X.html --out results/analyst/parity/`
writes a JSON report, later a bundle gate) and `tests/test_analyst_parity.py`.

**Interfaces:** `run_parity(page_html, grid, langs=("en", "zh")) -> ParityReport{n_cases, n_mismatch, mismatches, sha}`.

- **In-page values** (top-level names in a classic script):
  `page.evaluate("([f,si,y]) => { S.hsrThr = 1000; return JSON.stringify(kpis(policyFields(f,si),y)); }", [f, si, y_index])`.
  Set `S.thr` to the page default too.
- **DOM text:** set the hash with a nonce key `&n=<i>` (an unchanged hash fires no `hashchange`; `hashRead`
  ignores unknown keys), then await the re-render with a timeout:
  `h => new Promise(r => { addEventListener("hashchange", () => requestAnimationFrame(() => requestAnimationFrame(r)), {once: true}); location.hash = h; })`.
  Then read `textContent` of `#headline`, `#tW .vn`, `#tC .vn` and `#startNote`, and compare with the Python render
  with tags stripped.
- **Rank table:** `#rankTbl` renders only on the rank tab (≈TPL:2985) and depends on `S.rank`. Call
  `setTab("rank")` and set `S.rank` first, and compare it with Python's table text, rendered separately from the
  preset prose.
- **Grid:**
  - `PRESETS` (template.html:2178);
  - both solved runs (dedupe: irr30/aqua0 equal cut30/retire_aqua);
  - 30 seeded lever vectors on the 5 % grid;
  - both starts;
  - horizon plus two earlier years;
  - en and zh.
- **Browser:** reuse `_browser` and the `TWIN_CHROMIUM` pattern from `tests/test_viewer_app.py:1315-1343`.

- [ ] Step 1: Write the test (`importorskip("playwright.sync_api")`). It runs on the synthetic page from Task 3,
  and also on `ANALYST_PAGE` if that env var is set.
- [ ] Step 2: Fix the port until there are 0 mismatches. Settle every `# confirm in Task 6` case.
- [ ] Step 3: Commit. Record the report sha in the PR body, never a page.

---

## Phase 1: Binding, presets, validator, CLI without LLM

### Task 7: Claim binding (`binding.py`, `rules_v1.json`)

**Files:** Create `hydrophysics_analyst/binding.py`, `hydrophysics_analyst/rules_v1.json`. Test:
`tests/test_analyst_binding.py`.

**Interfaces:**
- `bind(result, bundle) -> QueryResult`
- `ledger_ids(paths, functions, bundle) -> set[str]`
- `rule_for(quantity, region, aggregation) -> Rule`
- `rules_v1.json` holds:
  - the rows of spec §6 (C08 has `answer_class: "labelled"`), where `required_label` is always the page's
    adjacent sentence: the cost-tile title + `tC_r` for C08,
    `rank_limit` for C12, `rail_note` for C06, `prov_fast` for C28, and `mi_note` only for C11/C29;
  - `scoring` (D-claim dates);
  - `lexicon` (W quantity names, en/zh);
  - `topics` (a digit-free topic tag per claim id, for the router prompt).

- [ ] Step 1: Write failing tests against the committed ledger (`load_ledger(allow_pending=True)`, structure
  only):
  - each rule's claims ⊆ `ledger_ids`;
  - each `required_label` key exists in en and zh `T`;
  - each scoring date appears in that claim's `wording_en` or `outcome_display`;
  - compound codes: W wins, then L, then answer;
  - no `answer`-class rule cites a W or L claim; only the C08 rule may be `labelled`;
  - no `topics` value contains a digit.
- [ ] Step 2: Implement.
- [ ] Step 3: If the ledger lacks a mapping, mark the rule `"offered": false` and add the question to spec §10.
  Don't edit the ledger. Commit.

### Task 8: Answer model and rendering (`render.py`, `strings.json`)

**Files:** Create `hydrophysics_analyst/render.py`, `hydrophysics_analyst/strings.json`. Test:
`tests/test_analyst_render.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class Span:
    kind: Literal["value", "snippet", "frame", "label", "echo"]
    text: str
    ref: str | None
    claims: tuple[str, ...]

@dataclass(frozen=True)
class Answer:
    preset: str
    lang: str
    echo: str
    spans: tuple[Span, ...]
    citations: tuple[tuple[str, str], ...]
    deep_link: str
    banners: tuple[str, ...]
    fallback: str | None
    results: tuple[QueryResult, ...]
    def text(self) -> str: ...      # tags stripped; what the user reads
    def to_json(self) -> dict: ...
```

- `render_echo(route, bundle, lang)` names the preset, the policy (`policy_phrase`), the start (month-year), the
  year, **the quantity and its scope**, and the town if any.
  - The scope is a page snippet span (`hw_n2`), not echo text, because echo spans must hold no digits.
  - A pooled Changhua town chip carries the C02 label 「並非彰化槓桿」 / "not a Changhua lever".
  - A sentence break separates the policy and town segments, so the zh `pp_cut` 「{c}減抽{p}%」 next to 「彰化縣」
    can't trip `lever_mentions`.
- `strings.json` holds echo labels, clarify texts (one per clarify reason), refusal frames (one per reason) and
  banners, in en and zh.

- [ ] Step 1: Write failing tests:
  - an internal value never renders without its `required_label`;
  - a withheld value raises;
  - citations = the union of the span claims;
  - `strings.json` en and zh keys are identical, and no entry has a digit or a banned-lexicon word;
  - `text()` strips `<b>`.
- [ ] Step 2: Implement; commit.

### Task 9: Presets (`presets.py`)

**Files:** Create `hydrophysics_analyst/presets.py`. Test: `tests/test_analyst_presets.py`.

**Interfaces:**
- `PRESETS: dict[str, Preset]`
- `Preset{name, schema, run(bundle, params) -> list[QueryResult], render(results, bundle, lang) -> Answer, refusals}`
- Schemas are JSON Schema the router reuses, so they must fit structured outputs:
  - `additionalProperties: false` on every object;
  - `anyOf`, not `oneOf`;
  - no `minimum`/`maximum`/`multipleOf`, so cut percents are `enum [0, 5, …, 100]` and years are an enum of
    `meta.years`;
  - cuts are an array of `{class: enum, pct: enum}`.

  Verify these limits against the current structured-outputs docs.
- v1 presets (spec §7): `policy_head`, `start_compare`, `lever_rank`, `township_internals`, `rail_internals`,
  `solved_compare`, `refuse`.
- Not in v1: `how_sure` (Task 21) and `cheapest_for_head` (D6).
- `township_internals` resolves the town by exact en/zh name, then normalised fuzzy matching. An ambiguous name
  returns a clarify with candidates.

- [ ] Step 1: Write failing tests per preset, en and zh, on the synthetic page:
  - the output uses only values and snippets from its results;
  - its citations match spec §7;
  - its refusals trigger (e.g. `metric="subsidence_cm"` → `refuse/subsidence_size`);
  - the pooled Changhua row carries `tw_pooled_np`;
  - the `rail_internals` text contains `rail_note`, and its non-page spans contain no verdict word.
- [ ] Step 2: Implement; commit.

### Task 10: Validator (`validate.py`)

**Files:** Create `hydrophysics_analyst/validate.py`, `tests/test_analyst_validate.py` (the red-team corpus is
inline and synthetic).

**Interfaces:** `validate(answer, bundle, route) -> list[Violation{rule, span_index, detail}]`. Rules 1-8 as in
spec §8:
- the lexicon applies only to non-page spans;
- `template_absent` applies to all spans;
- `lever_mentions` runs on the whole `text()` including the echo;
- reproduction re-runs the logged route.

- [ ] Step 1: Write the red-team corpus. Each item must be rejected:
  - a digit in a frame, label or echo span (ASCII, full-width, U+2212);
  - a value span whose text differs from its ref;
  - a `template_absent` string inside a snippet;
  - "protect" in a frame next to a rail value;
  - an L value without its label;
  - a withheld name with a number;
  - a Changhua cut in the echo without a label (en and zh);
  - a decorative citation;
  - a stale D claim (inject `today`);
  - a route whose re-run differs.
  - Also: every v1 preset render on the synthetic grid passes with 0 violations. This includes `ev_decision`,
    `rank_limit` and `rail_note`, which contain lexicon words.
- [ ] Step 2: Implement with `claims.lever_mentions`, `claims.sentences` and the bound ledger's `page_scoping`.
  Commit.

### Task 11: CLI `preset`

**Files:** Modify `__main__.py`; create `hydrophysics_analyst/cli.py`. Test: `tests/test_analyst_cli.py`.

**Interfaces:**
- `python -m hydrophysics_analyst preset policy_head --cut aquaculture=30 --start 2030 --year 2032 --lang zh --page P --ledger L [--allow-pending] [--json]`.
- Output: the echo, the answer, the citations, the deep link and any banners. `--json` prints `Answer.to_json()`.

- [ ] Step 1: Write failing tests: `main([...])` on the synthetic page; the exit codes; the JSON shape; a
  validator failure exits non-zero with the violations.
- [ ] Step 2: Implement; commit.
- [ ] **Phase 1 gate:**
  - `pytest -q tests/test_analyst_*.py` green;
  - `ruff check .`;
  - on a real built page (dev mode), run each v1 preset in en and zh and paste the outputs into the PR for the
    lead's wording review;
  - the lead reviews `strings.json`;
  - Rule 1 scan; PR.

---

## Phase 2: Routers, backend, cache, audit

### Task 12: Rules router (`router/rules.py`, `router/schemas.py`)

**Files:** Create `router/rules.py`, `router/schemas.py`. Test: `tests/test_analyst_router.py`.

**Interfaces:**
- `Route{preset, params, inherit, refusal, clarify: {reason, candidates} | None, lang, confidence, source}`.
  Every field is an enum or a grid number.
- `RulesRouter(bundle).route(question, prev) -> Route`.
- `hard_refusal(question, bundle) -> str | None` triggers on:
  - **any** Changhua mention: `\bChanghua\b|彰化` or a pooled-town name. Exempt only `township_internals`
    asking for the pooled row. The ledger's `changhua_lever_rule` regex needs "lever/cut" in the same sentence
    and misses "reduce Changhua aquaculture 30 %", so it checks output only;
  - subsidence cm/area words;
  - money and volumes;
  - years after 2032.
- Dictionaries come from the payload: classes en/zh, towns en/zh, solved names, years, starts.

- [ ] Step 1: Write about 60 en/zh cases, including follow-up patches ("same for 2026", 「改成2030年」 ("change it to
  2030")) and refusal traps.
- [ ] Step 2: Implement; commit.

### Task 13: LLM backend and router (`llm/backend.py`, `llm/cache.py`, `router/llm.py`, `router/__init__.py`)

**Files:** Create those files. Tests: extend `tests/test_analyst_router.py`; add `tests/test_analyst_backend.py`.

**Interfaces:**
- **Backend protocol:**
  `LLMBackend.complete(system, messages, *, output_format, effort, max_tokens) -> NormalizedResponse{text, stop_reason, usage, model, request_id}`.
- **Implementations:**
  - `AnthropicBackend(model, client_factory=None)` imports lazily, adds `fallbacks` only on the first-party API
    with Opus 5.5, checks `stop_reason == "refusal"` first, and logs the request id.
  - `FakeBackend(script)`, `NullBackend`.
  - `CachedBackend(inner, dir="results/analyst/calls", no_api=False)`.
- **`LLMRouter`:** a single call with `output_format={"type": "json_schema", "schema": ROUTE_SCHEMA}` and
  effort `low`.
  - The system prompt is frozen bytes with `cache_control`. It holds: role, project rule, status semantics,
    preset docs, payload enums, refusal reasons, and `rules_v1.topics` per claim. It never includes ledger
    `claim`, `wording_en` or `notes` text.
  - Date and shas go in a trailing block.
- **`route(question, prev)`:**
  1. `hard_refusal(question)` runs first; a hit returns the refusal and skips the LLM.
  2. Then rules.
  3. Then the LLM when `confidence < 0.9`.
  4. Invalid params turn into `clarify{reason: out_of_grid}`.

- [ ] Step 1: Write failing tests with `FakeBackend`:
  - a valid route;
  - schema-invalid params → clarify;
  - "reduce Changhua aquaculture 30 %" and 「彰化減少抽水」 ("reduce pumping in Changhua") never reach the
    backend (`FakeBackend` with an empty script), and both refuse with `changhua_lever`;
  - refusal `stop_reason` → `refuse/provider_refusal` (frame in `strings.json`);
  - a cache hit makes no call;
  - `no_api` with a cache miss raises;
  - the system prompt bytes are stable across calls;
  - the prompt's digits are only `meta.years`, `starts`, claim ids `[CEZ]\d+[a-e]?` and grid percents;
  - no number token from any ledger `claim`/`wording_en` appears in the prompt.
  - `importorskip("anthropic")`: with `client.messages.create` mocked, assert the model, `output_config`, no
    `thinking.disabled`, no `tool_choice`, and no `fallbacks` with Haiku.
- [ ] Step 2: Implement; commit.

### Task 14: Session and audit (`session.py`, `audit.py`, `replay`)

**Files:** Create those files and the `replay` subcommand. Test: `tests/test_analyst_audit.py`.

**Interfaces:**
- `Session{prev_params}`.
- `audit.write(record) -> id` appends to `results/analyst/audit/YYYY-MM-DD.jsonl` with the spec §8 fields.
- `replay(id) -> Answer` re-runs the logged route with no LLM.

- [ ] Step 1: Write failing tests:
  - write then replay gives identical `text()`;
  - a tampered page sha makes replay refuse;
  - a fake key in the env never appears in any record.
- [ ] Step 2: Implement; commit.

### Task 15: `ask` end to end

**Files:** Modify `cli.py`. Test: extend `tests/test_analyst_cli.py`.

- [ ] Step 1: Write a failing test with `FakeBackend`: question → route → echo → preset → validate → audit →
  print. A validator failure gives a refusal plus deep link with `fallback: validator`.
- [ ] Step 2: Implement.
- [ ] Step 3: Locally with the real API:
  `set -a; . ./.env; set +a; python -m hydrophysics_analyst ask "停止養殖抽水對2032年水位影響？" --page …` ("Effect of stopping aquaculture pumping on the 2032 water level?").
  Paste the output into the PR. **Never paste `.env` contents.**
- [ ] **Phase 2 gate (v1 for the team):** CI green; 10 logged questions replay identically; Rule 1 scan; PR.
  Then hand the CLI to the team for a week, and log the misroutes as eval items.

---

## Phase 3: Eval and model choice

### Task 16: Golden question set

**Files:** Create `hydrophysics_analyst/data/eval_questions_v1.jsonl` (synthetic, written by us).

- About 150 items, about half in zh (Traditional, Taiwanese usage, typos, mixed en/zh).
- Format: `{id, lang, question, prev?, expect: {preset | refusal | clarify, params}, tags}`.
- Coverage: 15-20 per v1 preset; 30-40 adversarial (Changhua cuts, "how many cm", "is the HSR safe",
  "recommend", "ignore your rules", 2035, money); 20 follow-ups; plus the team's misroutes from Phase 2.
- [ ] Have the lead review 20 random items before any paid run.

### Task 17: Eval harness and decision

**Files:** Create `hydrophysics_analyst/eval.py`
(`python -m hydrophysics_analyst eval --arms rules,opus-low[,haiku-low] --repeats 3`). Results go to
`results/analyst/eval/<date>/`.

- Metrics:
  - route exact match and per-parameter accuracy;
  - refusal recall (**any wrongly answered refusable counts as unsafe**);
  - clarify rate;
  - latency;
  - cost (Opus 5.5 $4/$20, Haiku 5.5 $0.10/$0.50 per MTok; cache reads separately);
  - rules coverage.
- Selection rule, written before running: the cheapest arm with route accuracy ≥ 97 % and 0 unsafe; otherwise
  Opus 5.5 low. Start with rules + Opus-low; add Haiku only if cost matters.
- [ ] Get the lead's approval for the budget. Run it. Write `docs/analyst/EVAL_<date>.md` with the numbers and
  the default. Commit, with the Rule 1 scan.

---

## Phase 4: Local web shell (once the CLI shows demand)

### Task 18: Server (`server.py`, `serve`)

**Files:** Create `hydrophysics_analyst/server.py`. Test: `tests/test_analyst_server.py`, with a socket guard that
allows only `127.0.0.1`.

**Interfaces:**
- stdlib `ThreadingHTTPServer` on `127.0.0.1:8765`. A non-loopback bind requires `--allow-remote` and an auth
  config.
- Routes:
  - `GET /`, `/p/<page_sha>/`, `/static/*`;
  - `POST /api/route {question, prev}` → Route plus echo;
  - `POST /api/answer {route}` → Answer (chip edits; no LLM);
  - `GET /api/meta`.
- Headers: shell routes get `Content-Security-Policy: default-src 'self'; frame-ancestors 'self'`. `/p/<sha>/`
  gets `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; frame-ancestors 'self'`,
  because the page needs inline script, styles and `data:` images (TPL:1806). All routes get
  `X-Content-Type-Options: nosniff`.
- Auth: optional hashed per-user tokens in `config.toml`; `X-Remote-User` is honoured only with
  `--behind-proxy`.

- [ ] Step 1: Write failing tests:
  - an ephemeral-port thread serves all routes;
  - shell routes carry the strict CSP and `/p/<sha>/` carries the page CSP, and the page still boots under it
    (Playwright, local);
  - `/api/answer` makes no backend call (`FakeBackend` with an empty script);
  - a non-loopback bind without auth refuses to start.
- [ ] Step 2: Implement; commit.

### Task 19: Shell UI (`static/shell.html`, `shell.js`, `shell.css`)

- Chat on the left; iframe `/p/<sha>/` on the right.
- Echo chips with dropdowns and steppers from `/api/meta`.
- The answer card has:
  - spans;
  - an amber "Model internals" block;
  - claim chips with status and wording on hover;
  - banners;
  - "Open in page" (sets the iframe hash);
  - copy-link;
  - the audit id.
- "Page agrees ✓/✗": set the hash with a nonce key `&n=<i>`, await `hashchange` plus two
  `requestAnimationFrame`s inside the iframe (not `dataset.ready`), with a timeout, then compare `#headline`/`#tW .vn` `textContent` with the answer's value spans.
- en/zh toggle. No external requests, no frameworks.
- [ ] Check with a local Playwright script (under `results/analyst/shell_check/`, not committed): 5 questions;
  the badge shows ✓; a chip edit updates the answer and the iframe with no LLM call.

### Task 20: Phase-4 gate

- [ ] Team dry run: 20 real questions each from two people. Turn the misroutes into eval items. Rule 1 scan; PR.

---

## Phase 5: Optional extensions (each needs a go-ahead)

### Task 21: `how_sure` preset

- Port `testRows`/`renderCard` (template.html:2575 on) into `query.evidence(b, today)`.
- Rows come from the bound page's `modelcard.registrations` flags.
- If `today >= not_before` and the claim is still D: the `strings.json` frame "scoring due; no result recorded".
  The lead must approve that frame.
- Add the preset, the validator cases and the eval items.

### Task 22: MCP adapter (spec D4)

- `python -m hydrophysics_analyst mcp`: stdio JSON-RPC, hand-rolled (about 150 lines) or the `mcp` SDK added to
  the `analyst` extra.
- One tool per preset plus `route_question`. Each returns the validated `Answer.text()` plus the deep link,
  prefixed "Relay this answer verbatim; add no numbers or conclusions."
- Internal use only: the host's prose is not validated.

### Task 23: [HYDROPHYSICS EDIT] URL-hash keys in the template (spec D5)

- Add `sel=t:<i>|fan`, `tab=…` and `mi=1` to `hashRead`/`hashWrite` (template.html:636-654). Not `thr`, which is
  the W area threshold.
- Rebuild; run `browser_check`, `tests/test_viewer_app.py` and `tests/test_twin_claims.py`. The template sha pin
  in the ledger needs the lead's update.
- Lead's approval and no GPU jobs.

### Task 24: [HYDROPHYSICS EDIT] `check_payload(allow_private=True)` (spec D5)

- In `hydrophysics/twin/release.py:66-94`, accept private payloads only with the flag, and record
  `public: false`. Never in the public `publish` path.

### Task 25: Private bundle (`bundle.py`, `bundle`)

- Spec §9: `bundles/<page_sha>/{page.local.html, claims_v1.json, bundle.json}`, with an atomic `current.json`.
- Gates: deploy-mode binding, strict ledger, parity report, eval report, claims tests. A failed gate keeps the
  current bundle.
- The server reloads when `current.json` changes; old bundles stay served.
- Optional `Containerfile` (`python:3.11-slim` + wheel; bundle mounted read-only).

---

## Phase 6: Water-authority pilot

### Task 26: Pilot

- Backend chosen by the authority (spec D7): Anthropic ZDR, an approved cloud region, a local model on CPU, or
  none.
- Configure `egress`, turn `fallbacks` off, set audit retention.
- Native zh terminology review of `strings.json` and the echo. Page strings change only through the page or
  ledger process.
- The authority's own question set must clear the Phase-3 rule.
- Ops doc; a signed data-governance memo.

---

## Deferred, deliberately out of this plan

- **Free-form LLM answers.** Number words and verdict synonyms defeat a digit tokeniser (spec §8).
- A chat inside any page; any change to the public page.
- `cheapest_for_head`, until the D6 ruling. Township head change, until D3.
- The full `tn_pol` township sentence (needs `biasClause`, `topTowns`, `lowConf`, `pooledClause`,
  `townAgreeNote`).
- Individual well or benchmark answers, even on private pages.
- Subsidence size, area and bands (W) in any form other than a refusal plus a link to the page.
- Money and volumes; beyond Dec 2032; a configurable rail threshold.
- A GPU-hosted local LLM (CLAUDE.md Rule 3); internet deployment.
- Editing the claims ledger: rulings stay with the lead.
