# N1 code provenance

Written 2026-10-24, after the scores. It records which code produced the N1 policy-response
results (`docs/TWIN_EVIDENCE_2026-10.md`, sections 12 and 13). Nothing was re-run or
re-scored for it, and no hash-bound file was edited. Commit ids are from the repository's
object store; the file hashes are sha256.

## Batches that built the N1 models

The batches recorded a diff without a base commit (no `code.base`). The base was
reconstructed by comparing each recorded `code.diff`, restricted to `hydrophysics/`, with
the commit's diff over `hydrophysics/`, index lines ignored.

| batch | models | recorded `code.diff` sha256 | equivalent commits | check | branch |
|---|---|---|---|---|---|
| par23 | `stage3_datum_gate_m15` (G1) | `af53ef559f4277a8987e16140977ab323cc8143053a5319fba23d315ebb72fe1` | `ccb6ff2..d3b365d` | `calibrate_flow.py` identical. The recorded diff also appends the then-untracked test file | on the PR branch |
| par24 | `stage3_datum_gate_m1618` and `stage3_datum_gate_ccz_m1618` with their withheld-survey columns (step C), plus the step A1 and step B runs | `2618484a818a7f401bc9536aaa26eaf5a040fec8f555df1c5eace83b3738531f` | `c48e858..294ed2e` | `calibrate_coupled.py`, `explorer3d.py` and `forward.py` identical | `twin/n1-policy-response`, not on the PR branch |
| par25 | F1 and F2 refits and columns | `277d6a509be59f39928ccbcbb4ed25d7bcd64e53f430204e9959d8b765869e5a` | `c48e858..8a96b21` | the four `hydrophysics/` files identical; the two test files were untracked | `twin/n1-changhua-fix`, not on the PR branch |

`N1.policy_response.v1` hashes the par24 diff, and `N1.changhua_fix.v1` hashes the par25
diff. Neither `step_c/results.json` nor `fix_c/results.json` records the tree the scorer
imported. The only evidence that the right tree was imported is that the runs completed,
since the PR branch's twin modules lack `--withhold-surveys`, `--column-zones 3` and
`--return-rice-share`, together with the logged `-I` failure below. Until those two branches
are merged, step C and the fixes can be reproduced only from `294ed2e` and `8a96b21`.

## How to run the scorers

Use `results/twin_runs/par24/py.sh` (steps A-C) or `par25/py.sh` (fixes) **without `-I`**.
Each sets `PYTHONPATH` to a checkout of `294ed2e` or `8a96b21` and runs `python -P`. The
usage lines in `run_step_a.py`, `run_step_b.py` and `run_step_c.py` say `py.sh -I`. That
flag drops `PYTHONPATH` and imports the main tree's twin modules, which is the logged
engineering failure of step B (Changhua) and A1 (`step_b_run.failed1.log`,
`step_a1_run.failed1.log` under `results/dataprep/reports/N1_policy/`). Those files are
hash-bound by the registration and stay as they are. `page_labels.py` imports no twin
code, so `-I` is harmless there.

## G1 and G4 scorers (`N1.phase12_response.v1`)

The registration (created 2026-10-13T20:00:43Z) hashes `n1lib.py` and `n1const.py`, not the
scorers, which were written later:

| file | sha256 | mtime (UTC) | look it produced (`looks.jsonl`) |
|---|---|---|---|
| `dataprep_staging/n1_natural_experiments/run_g4_piles.py` | `85834cca8327c253992a4d3ced2b861e27e63edf6788f932520fe8490bd2b199` | 2026-10-22 12:37:22 | G4, piles 2020-01..2022-05, 12:39:23 |
| `dataprep_staging/n1_natural_experiments/run_phase12_tests.py` | `90a15a4d53eeb98ef81bd6107124d92b5b449c76bae46f7c759868fbd0748ffc` | 2026-10-22 12:31:54 | G1, heads 2015-01..2015-05, 16:13:37 |

Both files are unchanged from commit `ba76923` (2026-10-23) to the PR head.
`phase12_tests/results.json` records no scorer hash. The file times and that commit are the
only link between the committed scorers and the G1 result cited in the model card.

## Page model against tested model

- **Page.** Theta `results/twin_runs/stage3_datum_gate/stage3_theta.json` (`036a7a17…`), commit `fa5cca2`, vendor rainfall, fitted before the adjoint fix `730f65b`. Forward run `datum_gate_innov.npz` (`cf2c7154…`), basis `response_basis_innov.npz` (`f2e7c536…`).
- **Tested.** The N1 tests scored `stage3_datum_gate_cwa` (theta `789d0e45…`; recorded commit `9715f58`; the code was at or after `13860ee`, see `TWIN_THETA_PROVENANCE_2026-10.md`) in Yunlin and R1c (`stage3_datum_gate_ccz`) in Changhua.
- **Step A2.** It used `results/twin_forward/response_basis.npz` (`df534c04…`, commit `2f21f81`, built 2026-09-20). That is neither the page's basis nor R0's (`response_basis_cwa.npz`, `2b4b1d05…`).
