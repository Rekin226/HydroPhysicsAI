# Which code produced each cited theta (2026-10-24)

`calibrate_flow` recorded `git rev-parse --short HEAD` of the working directory as the
theta's `meta.git_commit`. Every batch since Phase 1 ran from the main checkout with a
worktree on `PYTHONPATH`, often with uncommitted changes, so the recorded commit is the main
checkout's HEAD, not the code that ran. From this fix on, `calibrate_flow` records the commit
of the package that runs (`git -C hydrophysics/`), its full SHA, a dirty flag, the count of
untracked files and the sha256 of `git diff HEAD` over the package (`git_commit_full`,
`git_dirty`, `git_untracked`, `git_diff_sha256`). Thetas written before the fix keep their
old value. This table says what the record supports for each one.

How each row was checked:
- **meta keys**: every meta key and theta key the file carries is written somewhere in that
  commit's `calibrate_flow.py` or `zone_county.py`. A key missing at the recorded commit
  proves the recorded commit is not the code that ran.
- **batch diff**: the batch saved `code.diff` (and `code.status`) under
  `results/twin_runs/parNN/`. The diff's `index pre..post` blob ids were compared with the
  blobs at a base commit and at a candidate commit; every file must match on both sides.

| theta (`results/twin_runs/...`) | role | recorded `git_commit` | code that ran | status |
|---|---|---|---|---|
| `stage3_datum_gate` | the forward run on the decision page (`datum_gate_innov.npz`) | `fa5cca2` | `fa5cca2` | consistent: every meta key is written at `fa5cca2`; no diff was saved, so an uncommitted change cannot be excluded |
| `stage3_datum_gate_cwa` | R0, the deliverable (batch `par8`) | `9715f58` | not `9715f58`: its meta carries `rf_timeseries_sha256`, which `calibrate_flow` first writes at `13860ee` (a descendant of `9715f58`) | `13860ee` is the first commit consistent with every meta key; no diff was saved, so not verified |
| `stage3_datum_gate_ccz` | R1c, the census model (batch `par11`) | `709f660` | `709f660` plus `par11/code.diff` (sha `308d2ce0`) plus the untracked `zone_county.py` | partly verified: the post-image of the nine fitting modules in the diff (`calibrate_flow`, `calibrate_coupled`, `forward`, `inputs`, `zones`, `uncertainty`, `scenario`, `policy_gate`, `residual_kriging`) equals commit `22682ce`; `zone_county.py` was untracked and not saved, so its content is not recorded. No commit holds the whole batch code (`mechanism.py`, `ring_holdout.py` and `viewer_app.py` differ from every commit; none is imported by the fit) |
| `stage3_datum_gate_m15` | G1 masked-2015 refit (batch `par23`) | `ccb6ff2` | `d3b365d` | verified: `par23/code.diff` (sha `af53ef55`) is exactly `ccb6ff2..d3b365d` (`calibrate_flow.py`) |
| `stage3_datum_gate_m1618` | N1 step C, Yunlin flow (batch `par24`) | `c48e858` | flow fit `c48e858`; column `coupled_leveling_w1618` from `294ed2e` | verified: `par24/code.diff` (sha `2618484a`) is exactly `c48e858..294ed2e` and does not touch `calibrate_flow.py` |
| `stage3_datum_gate_ccz_m1618` | N1 step C, Changhua flow (R1c, batch `par24`) | `c48e858` | flow fit `c48e858`; three-zone column `coupled_leveling_z3_w1618` from `294ed2e` | verified as the row above |
| `stage3_datum_gate_ccr_m1618` | N1 fix F2 (batch `par25`) | `c48e858` | `8a96b21` | verified: `par25/code.diff` (sha `277d6a50`) is exactly `c48e858..8a96b21`, tests included; its meta keys `return_rice_share*` and theta key `return_rice_b` are never written at `c48e858` |

Reproducing the N1 Changhua results and fix F2 needs branch code that is not on the PR
branch: the three-zone column (`column_zones_collapsed`, commit `294ed2e`, branch
`twin/n1-policy-response`) and the paddy-weighted return flow (`return_rice_b`, commit
`8a96b21`, branch `twin/n1-changhua-fix`). `twin.forward` on the PR branch now refuses both
(`forward.check_supported`, `check_supported_column`) instead of silently rolling out a
different model.
