# Twin analyst: data-governance memo (template for the authority deployment)

**Status:** template. Fill in, review with the authority's information-security officer, sign
before go-live. **Backend chosen (2026-11-01):** the Anthropic API under a zero-data-retention
(ZDR) agreement. Everything stays swappable: a no-LLM mode needs no external service at all.

## 1. What the language model is used for

Routing only. The model reads the analyst's question and returns a choice from fixed lists:
- a preset;
- class cuts on a 5 % grid;
- a start year and a reporting year;
- a township from the page's list;
- or a refusal or clarification reason.

It never computes a number and never writes text shown to the user. Every number and sentence
comes from the bound decision page; a validator checks every answer. A question the rules
parser can read is never sent to the model: 73–83 % of the evaluation questions
(`EVAL_2026-11-01.md`).

## 2. What leaves the authority's network (per routed question)

| Sent | Not sent |
|---|---|
| The question text | Any number from the page or the model's data |
| The previous question's resolved parameters (enum values) | Individual wells, benchmarks, observations |
| A fixed system prompt: preset descriptions; water-use class and township names; start and reporting years; refusal reasons; digit-free claim topics | The claims ledger text, the payload, the answer |
| The first 16 hex characters of the page and ledger hashes | User names, audit records, API keys |

The router prompt is checked by a test: it may contain no number except years, start years,
claim IDs and grid percents.

## 3. Provider terms to confirm

- **ZDR.** A ZDR agreement with Anthropic covering the model in use (`claude-opus-5-5`).
  Fable-tier models are not ZDR-eligible; the config pins the model.
- **No server-side fallback** (`llm.fallbacks: false`), so no other model ever receives a
  question.
- **Region.** `inference_geo` offers `global` or `us` on the first-party API. If the authority
  requires a specific region, a cloud provider's Claude endpoint in an approved region is the
  alternative, configured through `AnthropicBackend(client_factory=...)`. Availability in a
  Taiwan region is **unverified**.
- **No language model at all.** Set `llm.enabled: false`. The rules parser and the form answer
  every preset; unparsed questions get a clarification.

## 4. Records kept on premises

The audit log (`audit.dir`, one JSON line per answer) holds:
- the user, question, route and stage;
- the model id, provider request id, token usage and cost;
- the answer text, claim IDs, validator verdict and deep link;
- the page, ledger and strings shas and the code version.

Retention: `audit.retention_days` = ______ (authority policy). Files older than this are
deleted at server start. Questions may be sensitive; access to `audit.dir` is restricted to
______.

## 5. Language review

Answer sentences are the page's own Chinese strings, which go through the page and ledger
process. The only analyst-specific Chinese text is `hydrophysics_analyst/strings.json`:
interpretation labels, clarifications, refusal frames and banners.
Native terminology reviewer: ______.

## 6. Sign-off

| Role | Name | Date |
|---|---|---|
| Project lead | | |
| Authority information security | | |
| Authority data owner | | |
