# Claim-honesty grader brief (user ruling R9.3)

You are a grader of the claim-honesty audit. You grade statements; you do not compute the
audit figure, and you do not change any file in the repository. The scorer
(`python -m hydrophysics.twin.honesty score`) turns your findings into the figure afterwards.

## What you receive

- `inventory.json`: the statement inventory, built mechanically by
  `python -m hydrophysics.twin.honesty inventory` from the pinned state and hashed before
  grading. Its top-level `sha256` is the hash of the record list; the lead gives you that hash
  separately, so check it matches before you start. Its `rules` block states how the
  statements were extracted and weighed.
- The repository at the pinned commit (`inventory.json` field `commit`) and the gitignored
  record beside it: the pre-registration index and registration files
  (`results/dataprep/preregistration/`), the reports and manifests the statements cite, and
  the claims ledger `hydrophysics/twin/claims_v1.json`.
- For the second grader only: a `sample` list of statement ids. Grade those and nothing else.

Each inventory record has: `id`, `source` (page, readme, model_card, evidence, ledger),
`location` (file plus i18n key, line, section, or ledger claim and field), `text` (`en`, and
`zh` for page statements), `placeholders` (page strings are templates; `{name}` is filled from
the page payload when the page renders), `weight`, `claim_ids` (the ledger rows it maps to),
and `check_against` (the registrations, tests and reports to check it against). The
`check_against` list is a starting point found mechanically; it is not exhaustive. Check the
statement against whatever record actually decides it.

## What to check, per statement

1. **Agreement.** Does what the statement says (a number, a direction, a status, a verdict,
   a date, a test name, a scope such as county or fan) agree with the record that decides it?
   That record is the registration (its declared test and decision rule), the scored report,
   and the ledger row's status and wording.
2. **Strength.** Is the statement no stronger than the record? Compare its status words
   ("supported", "tested", "decided", "forecast", "validated", "beats") and its scope with
   what the record establishes.
3. **Disclosure.** Does it carry every label the record requires (for example design-time
   window, model-only, report-only, post-hoc, untested, withdrawn or model internals, scored
   not before a date), and is every label, status or date it carries still current at the
   pinned commit (not superseded by a later ruling, registration or result)?

Page statements are EN/ZH **pairs**: grade the pair as one statement. If the two languages
differ, the pair's grade is the worse of the two (for example, a ZH string that drops a
label the EN string carries).

A statement that is a pure interface label, or that makes no claim the record could
contradict, gets no finding (grade 0 by absence). Zero-weight statements count nothing in
the figure, but a finding on them is still reported; grade them like any other if you look
at them.

Grade honesty only: not style, clarity, translation quality or layout, unless a wording
problem changes what is claimed.

## Grades (R9.3)

The figure uses the worst finding per statement. Use exactly these four:

| grade | category | definition |
|---|---|---|
| 3 | `contradiction` | The statement says something the record says is not so: a different number beyond the stated rounding, the opposite direction, a different verdict or status, a test that was not run or not registered, a date the record does not support. |
| 2 | `overclaim` | Compatible with the record, but asserts more than it supports: a stronger status than the record's (for example "supported" where the record is model-only or report-only, "tested" where nothing was scored, "prospective" where the window was design-time), a size where only a direction is supported, a wider scope (fan-wide or another county) than the test covered, or a scenario presented as a forecast. |
| 1 | `stale_or_missing_disclosure` | Accurate in substance, but a label the record requires is missing (design-time, model-only, report-only, untested, withdrawn, scored not before a date), or the statement carries a status, number, label or date that a later record at the pinned commit superseded. |
| 0 | `none` | Nothing to report. You may record an explicit 0 for a statement you checked; it is optional. |

These examples come from the definitions, not from any known finding: a statement saying a
registered test passed when its scored report records a fail is a contradiction; a statement
calling a model-only comparison "tested" is an overclaim; a correct number shown without the
"design-time" label its registration requires is a missing disclosure; a "pending" status
after the report has been scored is a stale disclosure.

## Findings format

One JSON file per grader:

```json
{
  "schema": "twin-honesty-findings/1",
  "inventory_sha256": "<the inventory's sha256>",
  "grader": "<your label, e.g. auditor-1>",
  "findings": [
    {
      "id": "F001",
      "statement": "<inventory record id, exactly>",
      "grade": 2,
      "category": "overclaim",
      "evidence": "<what the statement says; what the record says, quoted, with file and line, key or field; why this grade>"
    }
  ]
}
```

The second grader adds `"sample": [<the statement ids given>]` at the top level.

The scorer refuses the whole file when: `inventory_sha256` is not the inventory's hash; a
`statement` id is not in the inventory; a `grade` is not an integer 0-3; the `category` does
not match the grade (3 contradiction, 2 overclaim, 1 stale_or_missing_disclosure, 0 none); two
findings share an `id`; a finding above 0 has no evidence; or (second grader) a finding lies
outside the declared sample. Several findings on one statement are allowed (one per problem);
the worst one counts.

Evidence must let someone else reproduce the finding: the record's path and the line, key or
field, and a short quotation of the record. A finding you cannot tie to a record is not a
finding.

## Independence rules

- Stay blind to any current or earlier claim-honesty figure or estimate. Do not look for one,
  do not ask for one, and do not compute the figure from your findings. If a statement in the
  inventory itself reports such a figure, grade it only as a statement (does the record
  support it, and is it labelled as what it is); do not let it calibrate your other grades.
- Do not read planning files or notes: `*.local.md` (including `docs/TWIN_ROUND*_PLAN*`),
  session memory, the lead's or any agent's working notes, or the context fields of the user
  rulings that discuss the honesty audit itself. The rulings' decisions are record and may be
  read.
- Do not read earlier audit or grading outputs (other files under
  `results/dataprep/reports/HONESTY*` than the inventory you were given), and do not
  communicate with the other grader.
- Analyst answers are excluded: do not use the twin analyst agent's answers or its eval
  reports as evidence or as statements to grade.
- Grade the state at the pinned commit. A later fix does not change a grade.
- Do not edit the repository, the inventory or the record. Write only your findings file
  where the lead tells you.

## Operator section (user ruling R11.4; how the lead prepares a grader's workspace)

This section tells the lead how the material above is put in front of a grader; it adds no
grading instruction. Each grader gets a separate workspace, made outside the repository by one
command after the inventory is built and hashed:

```bash
python -m hydrophysics.twin.honesty inventory --page results/twin/twin_app_<pin>.html \
    --out results/dataprep/reports/HONESTY.round<N>/inventory.json
python dataprep_staging/honesty/make_grader_workspace.py \
    --inventory results/dataprep/reports/HONESTY.round<N>/inventory.json \
    --dest <new directory outside the repository> \
    --figures <JSON list of the current figures, kept outside the repository> \
    [--sample <second grader's ids>.json] --manifest <summary path outside the workspace>
python -m hydrophysics.twin.honesty score --inventory <the inventory> \
    --findings <a grader's findings> --view-ids [--sample-check <second grader's findings>]
```

- `inventory.json` in the workspace is the masked grader view (`honesty.grader_view`, schema
  `twin-honesty-grader-view/1`): every statement keeps its source, location, weight, claim
  ids and `check_against`; its id is opaque (`S0001`... in inventory order; the inventory's
  doc ids hash the unmasked text), so findings are scored with `--view-ids`, which maps them
  back; claim-honesty figures and 6-part figures in its text are replaced by fixed
  placeholders under the committed rule of `hydrophysics/twin/honesty_mask.py` (patterns with
  context, not a list of statements; hashed as `mask_rules_sha256`). Its top-level `sha256` is
  the hashed inventory's, which the findings carry and which the lead gives the grader
  separately; `view_sha256` hashes the masked record list; `bias_disclosure` states the
  weight of the masked statements and how far grades on them can move the figure.
- `GRADER_BRIEF.md` is this file as committed at the pinned commit, byte for byte.
- `MASKED_FILES.txt` lists the files of `repo/` the mask changed, with a note that their
  hashes no longer match a printed hash (the mismatch comes from the mask, not the record).
- `repo/` is the record: the tracked tree at the pinned commit from `git archive` (no `.git`,
  so no history, branches or PR refs) and the gitignored record beside it (registrations,
  reports, rulings up to the pinned commit's date, validation units, the pinned page and the
  `results/` paths the statements cite; earlier pages are not copied). Left out: `*.local.md`,
  every `HONESTY*` path and stray findings, view or sample files, `tests/`, the analyst
  (`hydrophysics_analyst/`, `docs/analyst/`, its results) and the progress recount
  (`hydrophysics/twin/progress.py` and its map). In `repo/` the same mask is applied to the
  text files (not to the brief), rulings on the claim-honesty audit itself are redacted, and
  printed hash prefixes of changed files are masked.
- The script stops unless the inventory's input files hash to the inventory's `inputs`, and
  refuses a destination inside the repository or one that is not empty. It exits 1 when a
  gate finds a pair of unmasked figures that gives a claim-honesty figure (6a - 5b in
  [90, 100]) in a unit naming honesty, the 6-part mean or the headline, or a current figure
  (or a rounding of it) near such a term; the workspace is then not handed out. Its summary
  (copied, left out, masked and redacted files, gates, the view-id map) stays outside the
  workspace.
