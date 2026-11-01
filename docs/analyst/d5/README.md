# D5: the two analyst changes inside `hydrophysics/` (prepared, not yet applied)

**Approved:** 2026-11-01 (spec D5). **Not applied yet.** A `hydrophysics.twin.forward` job was
running when these were prepared, and the project rule is that twin modules are never edited
while a twin job runs. Run `docs/analyst/d5/apply_d5.sh` in a quiet window. It refuses to run
while any `hydrophysics.twin.*` process is alive.

| Patch | Change | Why |
|---|---|---|
| `01-template-hash-keys.patch` | `hashRead`/`hashWrite` gain `sel=t:<i>`, `tab=<towns\|hsr\|sec\|cmp\|trust\|rank>`, `mi=1`; a `hashApplyView()` applies them at boot and on `hashchange`. No new i18n keys. | Analyst deep links open the exact township, tab and Model-internals state. `thr` is deliberately absent: it is the withdrawn area threshold. |
| `02-release-allow-private.patch` | `check_payload(payload, *, allow_private=False)`. The flag accepts a `--private` payload only if it records `public: false`; the public `publish` path never passes it. | The private deployment bundle (`hydrophysics_analyst.bundle`). |

**Tested in a scratch copy of the current working tree** (round-6 edits included):

- `tests/test_viewer_app.py`: 70 passed.
- `tests/test_twin_claims.py`: the only new failure is `test_ledger_is_pinned_to_this_template`, which is expected. The other failures already exist on the unpatched tree (the round-6 ledger work in progress).
- Browser check on a synthetic page:
  - a link with `sel=t:1&tab=rank&mi=1` boots into that state;
  - `hashchange` to `tab=hsr` applies it;
  - `hashWrite` writes the keys;
  - invalid values are ignored;
  - no console errors.

**Not covered by the patches:** pages built before them ignore the new keys, because `hashRead`
skips unknown keys. So the analyst already emits them, and its links degrade gracefully.
