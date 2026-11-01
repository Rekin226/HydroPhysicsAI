# Twin analyst: operations

The analyst answers plain-language questions (English or Traditional Chinese) about one built
decision page. Every answer is that page's own numbers and sentences, with claim IDs and a link
that opens the page in the same state. Design: [`../superpowers/specs/2026-11-01-twin-analyst-agent-design.md`](../superpowers/specs/2026-11-01-twin-analyst-agent-design.md).

## Install

```bash
pip install -e ".[analyst]"   # numpy, pandas, anthropic; no torch, no GPU
```

The language model is optional. Without `anthropic`, or with `--no-llm`, the rules parser and
the form still answer every preset.

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
  without the LLM.
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
