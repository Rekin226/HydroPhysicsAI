"""WP P4.0: record the WP's looks at its own results in the A1 look log (logged late).

    cd dataprep_staging
    python -m p4_changhua.looks_p40

The pre-registration ``P40.engine_acceptance.v1`` (2026-10-12T06:23:42Z) is a declaration:
numerical verification against the repository solver, no fresh data units. The looks below
record every computation of its acceptance numbers, plus the in-sample computations on the
2012-22 heads that the build makes (datum replica, R2, level-loss slices, Fisher, profiles).
Each look is recorded once; a rerun skips looks already in the log (matched on the note key).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import PREREG_DIR  # noqa: E402
from harness import prereg  # noqa: E402

WP = "P40"
PRE = "P40.engine_acceptance.v1"
THETA8 = "cwa_theta_789d0e45"
A1_UNITS = ["p40.A1.base", "p40.A1.extra", "p40.A1.riv"]
A2_UNITS = ["p40.A2.changhua_uniform", "p40.A2.single_cells", "p40.A2.random_field_131",
            "p40.A2.frozen_lift_energy"]
INS_UNITS = ["p40.insample.datum_replica", "p40.insample.r2", "p40.insample.level_slices",
             "p40.insample.fisher_interior", "p40.insample.pinned_profiles"]
A1_SCOPE = f"numerical:sparse_vs_torch:{THETA8}:2012-01..2022-12"
A2_SCOPE = f"numerical:adjoint_vs_forward:{THETA8}"
INS_SCOPE = "heads_insample_replication:2012-01..2022-12"

RUNS = [
    ("P40-look-1-trial", "2026-10-12T06:29:57Z..06:35:38Z",
     "scratch trial of build_engine.py (--scratch in the session scratchpad, p40_trial/; "
     "reduced settings: 24 kernel lags, 4 Monte Carlo draws). Printed A1 base 1.662e-06, "
     "extra 2.052e-06, riv 2.713e-05 m; A2 float64 1.66e-15/1.09e-15 on the impulse cases and "
     "3.49e-03/1.70e-03 on the 131-month cases (kernel truncated at 24 lags < 131 steps); "
     "build_engine.py was edited afterwards (mtime 06:36Z; the diff is not recoverable). "
     "A1, the impulse A2 cases, the datum and the Jacobian are identical between this run "
     "and the product; only the profile-refit diagnostic differs (reference refit dloss "
     "-1.84e-2 here, -9.55e-2 in the product). Cases and thresholds are fixed by the "
     "pre-registration of 06:23:42Z"),
    ("P40-look-2-product", "2026-10-12T06:40:57Z..06:49:32Z",
     "product build changhua_abstraction/engine/v1 (180 lags, 200 draws): A1 base 1.662e-06, "
     "extra 2.052e-06, riv 2.713e-05 m (riv fails the reported threshold 1e-5); A2 all cases "
     "<= 2.7e-15 (float64) and <= 4.9e-08 (float32)"),
    ("P40-look-3-rivdiag", "2026-10-12T10:25:38Z..10:28:10Z",
     "riv diagnostic (diag_riv.py, after the product): torch per-step CG error and a "
     "comparison with the reference's CG tolerance tightened in-process to 1e-12 (base and "
     "riv); post hoc, explains the riv shortfall, changes no threshold or product"),
]


def _have(key: str) -> bool:
    _, _, looks = prereg._paths(PREREG_DIR)
    return any(lk.get("wp") == WP and key in (lk.get("note") or "")
               for lk in prereg.read_chained(looks))


def main() -> int:
    n = 0
    for key, when, what in RUNS:
        specs = [(A1_UNITS, A1_SCOPE, "score", "P40.A1", PRE)]
        if key != "P40-look-3-rivdiag":
            specs += [(A2_UNITS, A2_SCOPE, "score", "P40.A2", PRE),
                      (INS_UNITS, INS_SCOPE, "insample_replication", None, None)]
        for units, scope, purpose, test_id, pid in specs:
            tag = f"[{key}:{test_id or 'insample'}]"
            if _have(tag):
                continue
            note = (f"{tag} logged late (2026-10-12T10:3xZ, P40 resume session); computed "
                    f"{when}: {what}")
            if test_id is None:
                note += ("; in-sample replication on the calibration heads 2012-01..2022-12 "
                         "(the theta's own fit window): datum prior, R2 0.8651 (= the theta's "
                         "recorded value), level-loss slices, Fisher and profiles of the "
                         "pinned entries; no choice was made from them")
            prereg.record_look(units, scope, WP, purpose, test_id=test_id, prereg_id=pid,
                               note=note)
            n += 1
    print(f"recorded {n} looks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
