"""WP P4.0 repair round: record the repair round's looks in the A1 look log (harness.prereg3).

    cd dataprep_staging
    python -m p4_changhua.looks_p40_repair

Every run of the repair round (``P40_evidence/repair/runs.json``) is recorded: its numerical
verification looks (sparse twin vs torch, adjoint vs forward, tangent-linear vs stepwise, the
RIV diagnostic) and its in-sample computations on the calibration heads 2012-01..2022-12
(datum replica, R2, level-loss slices, profiles, Fisher, residual autocorrelation). Each look is
recorded once (matched on its tag in the note).

harness.prereg3 (A1 repair round 2) needs members for every unit outside the registries:

* numerical units read no observation (model output against model output on the model's own
  forcing), so they carry ``{"none": ...}``;
* in-sample units read the heads of the 158 calibration wells of ``build_head_field`` over
  2012-01..2022-12 (back-filled and observed months), so they carry those well ids and that
  window. Under prereg3's rule such a look, made after the A1 fresh units were declared, is
  eligible to block later registrations on every fresh unit whose members and window it
  overlaps; the report flags this for A1 (see issues for downstream).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.paths import DATAPREP, PREREG_DIR, REPORTS_DIR  # noqa: E402
from harness import prereg3  # noqa: E402

WP = "P40"
PRE_A = "P40.engine_acceptance.v1"
PRE_R = "P40.repair_linearisation.v1"
THETA8 = "cwa_theta_789d0e45"
A_UNITS = ["p40.A1.base", "p40.A1.extra", "p40.A1.riv", "p40.A2.changhua_uniform",
           "p40.A2.single_cells", "p40.A2.random_field_131", "p40.A2.frozen_lift_energy"]
R_UNITS = ["p40.R1.energy", "p40.R1.volumetric", "p40.R2.energy_variance",
           "p40.R3.volumetric_variance", "p40.riv_diagnostic"]
INS_UNITS = ["p40.insample.datum_replica", "p40.insample.r2", "p40.insample.level_slices",
             "p40.insample.profiles_v2", "p40.insample.fisher_reference",
             "p40.insample.residual_autocorrelation"]
EVID = REPORTS_DIR / "P40_evidence" / "repair"
ARRAYS_V1 = DATAPREP / "changhua_abstraction" / "engine" / "v1" / "engine_arrays.npz"
NUMERICAL_NONE = ("numerical verification on the calibrated twin's own forcing: model output "
                  "against model output (sparse LU vs torch CG, adjoint vs forward, "
                  "tangent-linear vs stepwise rollout); no head, leveling or other "
                  "observation is read")
INSAMPLE_WINDOW = "2012-01..2022-12"


def _runs() -> list[tuple[str, str, str]]:
    runs = json.loads((EVID / "runs.json").read_text())
    return [(r["key"], r["when"], r["what"]) for r in runs]


def _have(tag: str) -> bool:
    looks = Path(PREREG_DIR) / "looks.jsonl"
    return any(lk.get("wp") == WP and tag in (lk.get("note") or "")
               for lk in prereg3.read_chained(looks))


def _calibration_wells() -> list[str]:
    with np.load(ARRAYS_V1, allow_pickle=False) as z:
        sids = [str(s) for s in z["sids"]]
    if len(sids) != 158 or len(set(sids)) != 158:
        raise SystemExit(f"expected 158 distinct calibration wells, got {len(sids)}")
    return sids


def main() -> int:
    sids = _calibration_wells()
    ins_members = {"ids": sids, "window": INSAMPLE_WINDOW,
                   "description": "heads of the 158 calibration wells (build_head_field) over "
                                  "2012-01..2022-12, back-filled and observed months, read for "
                                  "in-sample replication of the theta's own level loss (its "
                                  "fit window); pooled statistics only, no hold-out split"}
    n = 0
    for key, when, what in _runs():
        specs = [(A_UNITS, f"numerical:sparse_vs_torch_and_adjoint:{THETA8}:{INSAMPLE_WINDOW}",
                  "verify", PRE_A, {u: {"none": NUMERICAL_NONE} for u in A_UNITS}),
                 (R_UNITS, f"numerical:tangent_linear_vs_stepwise:{THETA8}:{INSAMPLE_WINDOW}",
                  "verify", PRE_R, {u: {"none": NUMERICAL_NONE} for u in R_UNITS}),
                 (INS_UNITS, f"heads_insample_replication:{INSAMPLE_WINDOW}",
                  "insample_replication", None, {u: dict(ins_members) for u in INS_UNITS})]
        for units, scope, purpose, pid, members in specs:
            tag = f"[{key}:{pid or 'insample'}]"
            if _have(tag):
                continue
            note = f"{tag} P40 repair round; computed {when}: {what}"
            if pid is None:
                note += ("; in-sample on the calibration heads 2012-01..2022-12 (the theta's "
                         "own fit window); no choice was made from them")
            prereg3.record_look(units, scope, WP, purpose, members=members, prereg_id=pid,
                                note=note)
            n += 1
    print(f"recorded {n} looks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
