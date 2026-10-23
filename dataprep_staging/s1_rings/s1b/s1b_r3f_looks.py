"""WP S1b repair round r3, fix round (S1br fix): look-log records (``harness.prereg3``).

    python -I dataprep_staging/s1_rings/s1b/s1b_r3f_looks.py <record>

Records (each logged before the read or computation it covers, unless its note says otherwise):

* ``disclose_r3_verifier``: a ``verification_disclosure`` record of the r3 verifier's unlogged
  re-run of ``s1b_run.compute`` (its minor issue 5 asked for it);
* ``fix_diagnose``: before the fix round reads the per-site rows of the r3 decomposition
  ``v3_product_round1_validators`` and of the declared variants in
  ``repair_r3/retest/second_look.json`` (already computed, spent units) to diagnose the
  verifier's major issue;
* ``fix_symmetry``: before the after-the-fact step-policy symmetry run declared in
  ``S1b.r3f.step_policy_symmetry.v1`` (spent units, report only).

No record here scores anything; every real-data number of the fix round is after the fact on
S1b's spent round-1 units.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from harness import prereg3  # noqa: E402

from s1_rings.s1b import s1b_const as K  # noqa: E402

REG_UNITS = ["s1b.c.ngl_near", "s1b.c.wra_colocated", "s1b.d.insar_ring_sites"]
DECL_R3 = "S1b.r3.rings_v3_second_look.v1"
DECL_FIX = "S1b.r3f.step_policy_symmetry.v1"
SCOPE_REAL = "rings+gnss+insar:2020-01..2025-12"


def _rec(units, scope, purpose, members=None, note="", prereg_id=None) -> dict:
    return prereg3.record_look(sorted(units), scope, K.WP, purpose, members=members or None,
                               prereg_id=prereg_id, note=note)


def disclose_r3_verifier() -> dict:
    note = (
        "Disclosure by the S1b r3 fix round (2026-10-18): the independent verifier of S1b's "
        "repair round r3 re-ran s1b_run.compute(PRIMARY) and the plane-cell variant read-only "
        "on rings_bridged/v3, gnss_clean/full/v4 and insar_069A/v3 (spent S1b units) in a CPU "
        "slot, writing only to a temporary directory outside the repository, after the r3 "
        "second look (2026-10-17T23:40Z) and before this record. Its minor issue 5 asked for "
        "this record. Nothing re-scored; no number of the verifier enters the report.")
    return _rec(REG_UNITS, SCOPE_REAL, "verification_disclosure", None, note, DECL_R3)


def fix_diagnose() -> dict:
    m = {"S1b:r3f.fix_diagnose_2026-10-18": {
        "none": "S1b r3 evidence already computed (second_look.json rows of the primary run, "
                "the declared variants and the two decompositions); no product is read",
        "description": "S1b r3 fix round: diagnosis of the r3 verifier's major issue (the "
                       "one-sided artefact-removed variant)"}}
    note = (
        "S1b r3 fix round, logged BEFORE the reads: the per-site rows (C_pre, C_post, "
        "containment by reading) of repair_r3/retest/second_look.json for the primary run, the "
        "variants validator_artefacts_removed and gnss_catalogue_steps_only_in_gap, and the "
        "decomposition v3_product_round1_validators, plus the v2 and v4 step catalogues of the "
        "10 co-located WRA stations (dates, sizes, basis), to diagnose whether the 8/9 "
        "artefact-removed count depends on v4-only step handling (TKJS 2023-24, GFES 2020). "
        "Spent units, numbers already computed in r3; nothing new is scored.")
    return _rec(REG_UNITS + sorted(m), SCOPE_REAL, "diagnose", m, note, DECL_R3)


def fix_symmetry() -> dict:
    m = {"S1b:r3f.step_policy_symmetry_2026-10-18": {
        "none": "S1b's spent round-1 units (10 WRA pairs) re-read under declared symmetric "
                "GNSS step policies on gnss_clean/full/v2 and v4; no registered member",
        "description": "S1b r3 fix round: after-the-fact step-policy symmetry of the "
                       "artefact-removed reading"}}
    note = (
        "S1b r3 fix round, logged BEFORE the computation: the after-the-fact variants declared "
        "in S1b.r3f.step_policy_symmetry.v1 (the r3 artefact removal on the round-1 "
        "validators; catalogue-only steps in every window on v4 and on v2; every cell with "
        "SJES excluded, catalogue_all also with SJES kept) through the registered round-1 "
        "pipeline with rings_bridged/v3. After the fact on spent units; "
        "report only; never a gate; nothing fed back into S1a.")
    return _rec(REG_UNITS + sorted(m), SCOPE_REAL, "after_the_fact", m, note, DECL_FIX)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    what = argv[0] if argv else ""
    fns = {"disclose_r3_verifier": disclose_r3_verifier, "fix_diagnose": fix_diagnose}
    if what not in fns:
        raise SystemExit(f"usage: s1b_r3f_looks.py {'|'.join(fns)}")
    rec = fns[what]()
    print(json.dumps({k: rec.get(k) for k in ("time", "purpose", "units", "prereg_id")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
