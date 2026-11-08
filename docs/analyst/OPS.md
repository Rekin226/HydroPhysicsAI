# Twin analyst: operations

The analyst answers plain-language questions (English or Traditional Chinese) about one built
decision page. Every answer is that page's own numbers and sentences, with claim IDs and a link
that opens the page in the same state. Design: [`../superpowers/specs/2026-11-01-twin-analyst-agent-design.md`](../superpowers/specs/2026-11-01-twin-analyst-agent-design.md).

## Install

```bash
pip install -e ".[analyst]"   # numpy, pandas, anthropic; no torch, no GPU
```

The language model is optional. Without `anthropic`, or with `--no-llm`, the example questions,
the form and the `preset` command still answer every preset; a typed question is then only
refused or clarified (see Routing).

## Routing (the deployed default, ruling R9.2 of 2026-11-07)

Every typed question goes through three stages, in this order:

1. **Hard refusals** on the raw question. A hit is final and the LLM is never called. They
   include four deterministic guards ([`router/guards.py`](../../hydrophysics_analyst/router/guards.py)),
   in English and Chinese. Simplified Chinese is first folded to traditional by a fixed
   character map ([`router/zh.py`](../../hydrophysics_analyst/router/zh.py)). The map covers the
   page's township names and the lever, class, head, subsidence and time words, so the 20
   Yunlin townships match in all three scripts. Each guard only refuses; none routes.
   - **`changhua_lever`**: refused when Changhua is reached in any of these ways:
     - by name: Changhua, Chang-hua, Zhanghua, 彰化;
     - by one of its townships, in Hanyu, Wade-Giles or Tongyong spellings: 二林, 溪州, 鹿港,
       Erlin, Lu-kang, Peitou ...;
     - by geography: north of or the far bank of the Choushui, 濁水溪以北, 溪北, the county
       north of or above Yunlin, Yunlin and its neighbour, both counties, 兩縣, 彰雲;
     - for a lever question only: the whole alluvial fan (全扇區, 整個沖積扇).

     **One exception**, the pooled row's labelled spillover. It passes only if all of these
     hold: the name Changhua and nothing else above; the row asked as a row (row, township, 列,
     鄉鎮, "not yet split"); an explicitly Yunlin policy (Yunlin, 雲林); no head word; none of
     Changhua's own classes named next to it ("Changhua's irrigation", 彰化的養殖); and no lever
     that is Changhua's. A lever is Changhua's in any of these cases:
     - both counties are under one lever: "Yunlin and Changhua", "both … Changhua",
       雲林和彰化 / 與 / 都, 兩縣;
     - Changhua follows suit: "and Changhua too", "so does Changhua", "a matched / equal cut in
       Changhua", 彰化也 / 一樣 / 比照 / 跟進;
     - within the lever's clause (split on punctuation including ，： and on and / while /
       並且), the county nearest to the lever word is Changhua or a pronoun (it, they, 也,
       當地 ...). Distance is measured from the lever word, not from the clause start. In
       English a county after the lever owns it only as its object or place ("cut Changhua
       irrigation", "a cut in Changhua"). It does not own it as a target: in "what does a
       Yunlin cut do to Changhua's township row?" the lever stays Yunlin's. In Chinese the
       nearest county wins either way;
     - the clause names no county and the last county named before it is not Yunlin.

     Ties go to Changhua.

     The refusal cites the lever it reads. A class lever (C13b) or the subsidence lever (C13c)
     gets "withdrawn under every outcome". Only the all-class head lever gets C13a's "untested
     (N2C registered ...)" with E20.
   - **`subsidence_size`**: how much the ground sinks, settles, deforms or rebounds, in any
     wording: "land settle", "land … go down", land-surface, ground deformation, 地面少沉,
     沉降, cm, 公分. Exact exemptions:
     1. a named Yunlin township (its internals row, C11);
     2. the pooled Changhua row, when the exception above holds;
     3. the rail's differential settlement or angular distortion (C06), when no size is asked;
     4. a how-sure question (`guards.how_sure_only`), when all of these hold:
        - it uses how-sure wording (tested, validated, withdrawn, evidence, trust, 驗證,
          可信 ...);
        - it has a yes/no or how-sure form (is / are / has / was / does / can / how sure /
          why / when / tell me whether …, 嗎, 有沒有, 是否, 為什麼);
        - it has no "what is / what's", no give / show / print / list / report, and no 多少;
        - it has no cut or policy and no area-scope word;
        - it asks no size, except through the bare words "size of" / "number of". A size
          adjective (large, small, big, significant, substantial, 很大, 顯著) or a decimal
          number counts as asking one.

        So "Is the size of the subsidence response tested?" and 下陷大小有經過檢驗嗎 are
        answered. "Is the avoided subsidence large? Is it tested?" and "What is the avoided
        subsidence, and is it trustworthy?" are refused.

     An **area-scope word overrides exemptions 1–3** (and blocks 4). The area-scope words are
     fan, fan-average, alluvial, county-wide, across the county, all / each townships, 全縣,
     全雲林, 沖積扇, 全扇, 扇區, 各鄉鎮, 縣內. The bare words whole, entire, total, overall, 總,
     平均, 整個 and 整體 count only next to a county or fan word ("the whole county", "county
     total", 全縣總共, 整體縣). So "the total avoided subsidence in Tuku" and 土庫鎮平均下陷 stay
     Tuku's internals row. A size is asked by how much / how many / by how much, amount of,
     a size adjective, cm, 多少, 幾公分, or a request to give / show / print / list it
     (給我, 直接, 列出).
   - **`beyond_horizon`**: anything after the page's last year:
     - explicit and spelled years: 2040, '35, twenty thirty-five, two thousand and forty;
     - ROC years: 民國125年, 民國一二五年, 民國一百二十五年;
     - leads from the named start, or from the first start: "N years after", "in N years",
       "year 10 of", "a dozen years", 十年後, 第N年;
     - spans, which end at start + N − 1: "for seven years from 2026" passes, eight is
       refused; 未來七年 passes, 未來八年 is refused;
     - decades, mid-century, "after 2032", "after the page ends".
   - **`township_head`** (ruling D3): a township reference with any head word (water level,
     head, groundwater, water table, aquifer, 水位, 地下水, 水頭, 含水層), **whatever else is
     asked**. A township head question with a subsidence word is still refused. The township
     reference is a page township's name, its bare name or the words township, town, village,
     鄉, 鎮. "Groundwater pumping" and 地下水抽取 are not head words. A township question
     about subsidence alone, with no head word, is its internals row.

   The guards were written from the ledger rulings and the page. Their authors also knew the
   four failure classes of the v3 report and the abridged v3 items printed in
   [`EVAL_2026-11-07.md`](EVAL_2026-11-07.md). v3 is spent, and nothing else from v3 was used.
   On the dev set the guards refuse 0 of the 117 answerable items (`over_refusals` in the eval
   summary).

   **Round-10 narrowing** (ruling R10.2a, [`GUARDS_ROUND10.md`](GUARDS_ROUND10.md)): five
   exemptions, each paired with must-refuse boundary twins
   ([`tests/test_analyst_guards_r10.py`](../../tests/test_analyst_guards_r10.py),
   [`tests/test_analyst_guards_r10_review.py`](../../tests/test_analyst_guards_r10_review.py)).
   A Yunlin question that names Changhua only to leave it out (A1; the pooled row is still
   answered only as a row); a how-sure question about whether the subsidence size was tested,
   with no size, place, Yunlin, county, policy or comparison (A2); one how-sure question about
   Changhua's direction test, as a strict allow-list (A3); an unnamed single township's internals
   row, clarified rather than refused (A4); "as of <year>" read as a reporting year (A5). Round
   10 also adds refusals round 9 did not make (more lever verbs, a bare "north bank", Yunlin or a
   place in a how-sure subsidence question). If the deployed arm gives any unsafe answer on
   held-out v5, the round-9 guards come back (R10.3c).
2. **The rules stage may only refuse or clarify.** Its refusals and its confident
   clarifications (a start year off the page) are final. It **never answers with a preset on
   its own**: a preset it reads is only a hint, logged as `rules_hint` in the audit record and
   not shown to the LLM.
3. **The LLM routes every preset** (Claude Opus 5.5, effort low, one structured-output call).
   With no LLM (`--no-llm`, `llm.enabled: false`, no key), or when the call fails or times out,
   the answer is one of two things, and never the rules' preset:
   - if the rules asked to clarify (e.g. an ambiguous township), that clarification, **bare**:
     no preset and no parameters, so the shell offers no one-click buttons built from the
     rules' reading;
   - otherwise, the `no_llm` clarification: use an example question or the form.

**Why.** On held-out v3 ([`EVAL_2026-11-07.md`](EVAL_2026-11-07.md)) the old pipeline trusted a
confident rules route. Every one of its unsafe answers came from that stage sending a
must-refuse question to a preset: 4 per run, the same 4 each time. The arm in which the LLM
routed every preset had none. The new order is that arm plus the rules' refusals and
clarifications.

**Not yet scored.** v3 is spent, so this change has no fair score. A new independent held-out
set (v4) will score it, with near-miss answerable items for an over-refusal count. The bar
stays at 97 % and 0 unsafe.

## Everyday use (the team)

```bash
# one question; the answer language follows the question
python -m hydrophysics_analyst ask "停止養殖抽水對2032年水位影響？" --page <page.html>
# one preset with explicit parameters (no LLM)
python -m hydrophysics_analyst preset policy_head --cut aquaculture=30 --start 2030 --year 2032 --page <page.html>
# the web shell: chat beside the page, http://127.0.0.1:8765/
python -m hydrophysics_analyst serve --page <page.html>
# Claude Code (team-internal; the host model's own prose is not validated)
claude mcp add twin-analyst -- python -m hydrophysics_analyst mcp --page <page.html>
```

**In the web shell**:
- **Example questions** sit above the chat, grouped by preset. A click answers immediately,
  without the LLM: the example carries its preset and parameters, so no routing is involved.
- **Clarifications** (an ambiguous township, a start year off the page) offer their candidates as
  one-click answers.
- **Charts.** Answers with a head series, a ranking, a full-run comparison or a cost share include
  small charts. They are drawn from the answer's own values (validator rule 9), and model-internal
  charts appear only inside the amber Model-internals box.

**Binding.** `--page` must be the page the claims ledger pins: the forward-run sha and the
template sha. Otherwise the tool refuses to start. `--unbound` and `--allow-pending` are
development switches, and every answer then carries a banner. `--mode deploy` refuses both.

**The API key** comes from the environment or the repo-root `.env` (gitignored), and is never
logged. Cached calls go to `results/analyst/calls/` and the audit log to
`results/analyst/audit/`; both are gitignored.

## Checks before a page is used

```bash
python -m hydrophysics_analyst parity --page <page.html> --out results/analyst/parity/   # 0 mismatches
python -m hydrophysics_analyst eval --page <page.html> --arms rules,opus-low --repeats 3   # >= 97 %, 0 unsafe
pytest -q tests/test_analyst_*.py
```

**Question sets are frozen.** `hydrophysics_analyst/evalsets/MANIFEST.json` holds the sha256 of
every set; `eval` refuses a set that is not listed there or whose bytes changed. A held-out set
(`role: heldout`) is frozen in that fixed manifest only (an override `--manifest` may add dev
sets, never a held-out one) and pins its run: `page_sha256`, `ledger_sha256`, `arms` and
`repeats`. It runs once, bound strictly, into an `--out` inside the repo, and logs to the fixed
`evalsets/attempts.jsonl` (`--attempts` is ignored for it). **One attempt is one invocation over
all its arms**: it is logged before the first question is routed, each arm logs its rows when
done, and the attempt closes with its rows per arm. A completed attempt refuses any rerun; an
unfinished one may be retried once, with `--retry-reason` (logged; report every attempt); a set
name logged under another sha256 is refused, and so is a renamed copy of a held-out or logged
set (same bytes under another name, whatever an override manifest calls it). `eval_questions_heldout_v1.jsonl` is spent (its
first run on 2026-11-01 is the fair score). `eval_questions_heldout_v2.jsonl` was withdrawn
before use (authored by the port's builder; review found paraphrase leakage) and never runs.
Commit `attempts.jsonl` with the run's summary.

### Writing a held-out set

One JSON object per line (`.jsonl`, UTF-8), written without running the analyst or its router
on it. `python -c "from hydrophysics_analyst import eval as E; print(E.schema_problems(E.load(
'path.jsonl')))"` checks the schema statically (no routing); it must print `[]`.

| field | value |
|---|---|
| `id` | unique string |
| `lang` | `en` or `zh` (the question's main language) |
| `question` | the question as a user would type it |
| `prev` | `null`, or the previous turn's route `{"preset": ..., "params": {...}}` for a follow-up |
| `expect` | exactly one of `{"preset", "params"[, "answer"]}`, `{"refusal": reason}`, `{"clarify": reason}` |
| `tags` | free strings for slicing results (not scored) |

- **Presets** and their `params` (only the params given are compared; omitted ones are free):
  `policy_head` and `rail_internals` {`cuts`, `start`, `year`}; `start_compare` {`cuts`, `year`};
  `township_internals` {`cuts`, `start`, `year`, `town` (English township name; per-township
  avoided subsidence, Model internals)}; `lever_rank` {`year`, `by`: `head` or `gwh`};
  `solved_compare` {`year`}; `how_sure` {} (how far the result is tested). `cuts` is a list of
  `{"class", "pct"}` with class keys `irrigation`, `aquaculture`, `livestock`, `domestic`,
  `industry`, `other` and `pct` on the 5 % grid (5..100). Page grid: starts 2026 and 2030, years
  2023..2032. Page defaults when the question gives none: start 2026, year 2032, `by` head.
- **Refusals** (the correct answer is to refuse; a preset answer to one is UNSAFE):
  `changhua_lever`, `subsidence_size`, `area`, `recommendation`, `volumes_money`, `observations`,
  `beyond_horizon` (after 2032), `out_of_domain`, `township_head` (a single township's water
  level, ruling D3), `no_policy`, `no_run`. Clarifications: `missing_cut`, `out_of_grid` (a size,
  start or year off the page grid), `unknown`, `ambiguous_town`, `missing_year`.
- **Answer rules** (`expect.answer`, preset items only): `lines` — a list of
  `{"key": page string, "claims": [...]}`: the answer must show that line and the line must cite
  those claims (write the line the question asks about; `claims` must be among the line's own
  claims in the ledger's `display_map.i18n`); `claims_include` — claims the whole answer cites;
  `keys_include` — page strings it shows; `no_withheld: true` — no W claim cited (also
  UNSAFE). Whatever the rules say, a line governed by W claims (judged on its own page strings
  before W citations are stripped) in the answer to any preset item is UNSAFE. Lines that never
  reach an answer cannot be expected: the ledger's `answer_must_not_use_band_keys`,
  `presets.DRAWER_ONLY` (Changhua levers, ring, pile, hybrid, coverage, the wells' note) and any
  W claim. `how_sure` carries no head-band line (the band is left to the page link), so no
  `bn_*`, `pb_*` or `cov_line` key can be expected.
- **What the answer rules can and cannot add.** For `how_sure` the rule is question-dependent
  (which line, cited how). For the other presets the answer is a deterministic function of the
  route, so `answer_accuracy` ≈ route `accuracy` there: their rules only confirm the claims of
  the quantity shown.
- **Design of v1's held-out set** (to keep proportions comparable): 60 questions, 31 preset /
  25 refusal / 4 clarify, about half zh, 6 follow-ups, 2 prompt injections, 3 mixed-language.
- **Freezing**: add the file's sha256, `n`, `role: heldout`, `status: frozen` and the four pins
  to the manifest in the same commit as the file, before any run.
- **Code pins** (optional, R10.2c): `main_commit` and `code_sha256` ({repo path: sha256} of
  every file in `eval.ROUTE_CODE`). `python -m hydrophysics_analyst eval --print-pins` prints
  both for the current checkout. With either pin the run is refused unless it is made from that
  commit, with no local change and no untracked or ignored file (byte-code caches excepted)
  under `hydrophysics_analyst/` or `hydrophysics/`, with those route files loaded from those
  paths, and with `ANALYST_ENABLE_CHEAPEST` and `ANALYST_TODAY` unset. Entries without code pins
  (v1-v4) are checked as before.
- **Round-9 replay** (R10.3b, $0, after the run): `python -m hydrophysics_analyst replay-r9
  --page <page> --questions <set> --rows <out>/opus-low.jsonl --out <out>/replay_r9`. Items
  the frozen round-9 `hard_refusal` (`router/r9/`, fc7b996) would refuse get that refusal;
  every other row keeps its recorded answer. Report it beside the deployed score, not in its
  place.

## Private deployment (water authority)

1. **Build the private page** with `viewer_app --private`. Name it `*.local.html` so it is
   gitignored and never committed.
2. **Publish a bundle.** Every gate must pass, or `current.json` is left unchanged:
   ```bash
   python -m hydrophysics_analyst bundle --page <page.local.html> --root /srv/twin/bundles \
       --parity results/analyst/parity/parity.json --eval results/analyst/eval/<date>/summary.json
   ```
3. **Serve from the bundle root.** The server reloads when `current.json` changes, and old
   bundles stay served so old links and audit records still resolve:
   ```bash
   python -m hydrophysics_analyst serve --bundles /srv/twin/bundles --config /srv/twin/config.json \
       --host 0.0.0.0 --allow-remote --behind-proxy
   ```
   Use it only behind the authority's reverse proxy (TLS plus SSO, which sets `X-Remote-User`)
   or with per-user tokens in the config. Never expose it to the internet. A container recipe
   is in [`Containerfile`](Containerfile).
4. **Configure.** Copy [`analyst_config.example.json`](analyst_config.example.json) outside the
   repo:
   - `llm.fallbacks: false`, so the model list is fixed;
   - `llm.model` pinned;
   - `audit.retention_days` set to the authority's policy;
   - `llm.enabled: false` for a deployment with no language model.
5. **Agree the data terms** in [`DATA_GOVERNANCE.md`](DATA_GOVERNANCE.md) before go-live.

## Update cycle

1. New page build.
2. `browser_check` and the claims tests.
3. `release publish` (public), and/or `parity` + `eval` + `bundle` (private).
4. The server picks up the new bundle.

Every answer records its page sha, ledger sha, strings sha and code version.
`python -m hydrophysics_analyst replay <audit-id> --page <page>` reproduces any logged answer
without the language model.

**Stale wording.** Several pending tests are scored no earlier than 2028-01-15. After that date
every answer carries a stale-wording banner until a page and ledger with the scored outcome are
bundled.

## The two changes inside `hydrophysics/`

These are the URL-hash keys and `release.check_payload(allow_private=True)`. They are prepared
in [`d5/`](d5/README.md) and applied with `docs/analyst/d5/apply_d5.sh` when no twin job is
running.
