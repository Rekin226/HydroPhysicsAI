"""WP P1 repair round (r3): correct the A1 look log and show what the tool now decides.

The verifier's major issue: the round-2 design-time seed (``looks.jsonl``, logged
2026-10-12T05:58:43Z) recorded the main session's census profile under the unit
``changhua_census_vs_P1_v1`` only. ``inspection/profile2.json`` (file time 05:44Z), which P1
read before writing ``P1r2.census_tests``, also holds

* ``changhua_last14_cleaned_GWh``: Changhua county cleaned totals 2025-06..2026-07, the truth of
  T11 (P1's pre-registration says P1 saw them), and
* ``yunlin_last14_cleaned_GWh``: Yunlin county cleaned totals 2025-06..2026-07, the truth of T10
  (P1 declared it did not read that part; this cannot be verified).

Under the A1 convention, a unit whose test data were computed or seen before a
pre-registration carries a design-time seed dated to that evidence, so it can never serve as
a fresh unit. This script appends the two missing seeds (dated 05:44Z, the file time), logs
late the round-2 looks that were not logged (the declaration readings and the post hoc
diagnostics), logs the repair round's own after-the-fact computations, and then asks
``harness.prereg.assert_can_score`` (with ``log=False``, so no scoring look is added) whether
T10 and T11 could still be scored as pre-registered tests. Nothing is re-scored here; the
pre-registration file stays as written.

Idempotent: a record whose note starts with an existing tag is not appended again.

Usage: python -I r3_looks.py <evidence_json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.paths import PREREG_DIR  # noqa: E402
from harness import prereg as pr  # noqa: E402

PID = "P1r2.census_tests"
T10, T11 = "P1r2.T10_ext_yunlin", "P1r2.T11_ext_changhua"
U10, U11 = "yunlin_census_new_months_2025-07..2026-06", "changhua_census_months_2025-07..2026-06"
S10, S11 = "census_yunlin:2025-07..2026-06", "census_changhua:2025-07..2026-06"
PROFILE_TIME = "2026-10-12T05:44:00.000000Z"
PROFILE = ("data_fetch_api/new_measurements/tpc_changhua_yunlin_chiayi_pingtung_2026/"
           "inspection/profile2.json")

RECORDS = [
    dict(tag="[P1-r3-seed-1]", units=[U11], data_scope=S11, purpose="design_time_seed",
         seeded_time=PROFILE_TIME,
         note=f"corrective seed (repair round, verifier major issue): {PROFILE} (main session, "
              "file time 05:44Z) holds changhua_last14_cleaned_GWh, the Changhua county cleaned "
              "totals 2025-06..2026-07 = the T11 truth (the file covers 2025-06..2026-07; the "
              "scope names the test window). P1 read it before writing P1r2.census_tests "
              "(05:58:43Z), as that pre-registration's prior_exposure states. The round-2 seed "
              "named only 'changhua_census_vs_P1_v1', so assert_can_score could not block "
              "T11. T11 is relabelled an after-the-fact check."),
    dict(tag="[P1-r3-seed-2]", units=[U10], data_scope=S10, purpose="design_time_seed",
         seeded_time=PROFILE_TIME,
         note=f"corrective seed (repair round): the same file {PROFILE} holds "
              "yunlin_last14_cleaned_GWh, the Yunlin county cleaned totals 2025-06..2026-07 = "
              "the T10 truth, computed by the main session before P1's pre-registration and "
              "present in a file P1 opened. P1 declared it did not read the Yunlin part; that "
              "cannot be verified. Under the A1 convention a unit whose truth was computed "
              "before a pre-registration is design-time, so T10 is relabelled an after-the-fact "
              "check (candidate, threshold and month rule were still fixed before scoring)."),
    dict(tag="[P1-r3-late-1]", units=["changhua_census_vs_P1_v1"],
         data_scope="census_changhua:2012-01..2026-07", purpose="after_the_fact",
         note="logged late (repair round): r2_evaluate.py runs at 2026-10-12T06:06:05Z and "
              "06:08:36Z also computed the declaration readings of P1r2.census_confirmations "
              "(C1 a*, C2, C4 phi*, C5 anchor, C8 coverage, T12 shape, T13 rank: scores of the "
              "v1 candidates against the Changhua census 2012-2025). Declarations, no fresh "
              "units; the scripts were edited between the two runs (06:08:28Z); the second "
              "run's numbers are the ones published."),
    dict(tag="[P1-r3-late-2]", units=["yunlin_census_vs_P1_v1"],
         data_scope="census_yunlin:2012-01..2026-07", purpose="after_the_fact",
         note="logged late (repair round): the same r2_evaluate.py runs computed the Yunlin "
              "side of the declarations (C9 consistency with v1 YL_clean, the Yunlin terms of "
              "C1/C5/C8)."),
    dict(tag="[P1-r3-late-3]", units=[U10, U11, "changhua_census_vs_P1_v1",
                                      "yunlin_census_vs_P1_v1"],
         data_scope="*", purpose="after_the_fact",
         note="logged late (repair round): r2_diagnostics.py at about 2026-10-12T06:09:15Z "
              "(post hoc, after T10/T11 were scored) computed census/TPC ratios by July-June "
              "year 2016/17-2025/26, the year-on-year meter ratios, and the C9 row comparison "
              "(new Yunlin sheet vs AMP_V2)."),
    dict(tag="[P1-r3-look-1]", units=["changhua_census_vs_P1_v1"],
         data_scope="census_changhua:2012-01..2025-06", purpose="after_the_fact",
         note="repair round r3_census_sensitivity.py: C1/C4 readings (a*, phi*, in-grid level) "
              "recomputed under two pre-install codings of the Changhua kWh table inside the "
              "duty window (NaN -> 0, all or leading only); no reading or gate depends on it."),
    dict(tag="[P1-r3-look-2]", units=[U10, U11], data_scope="*", purpose="after_the_fact",
         note="repair round r3_census_sensitivity.py: cleaned census energy per meter with a "
              "non-zero reading, July-June years 2016/17-2025/26, both counties (truth only, "
              "no candidate scored), for the T10 wording; and the C9 extra-row pump count."),
]


def main(evidence: str) -> int:
    _, _, looks = pr._paths(PREREG_DIR)
    have = [lk.get("note", "") for lk in pr.read_chained(looks)]
    appended, skipped = [], []
    for r in RECORDS:
        note = f"{r['tag']} {r['note']}"
        if any(n.startswith(r["tag"]) for n in have):
            skipped.append(r["tag"])
            continue
        rec = pr.record_look(r["units"], r["data_scope"], "P1", r["purpose"],
                             seeded_time=r.get("seeded_time"), note=note)
        appended.append({"tag": r["tag"], "time": rec["time"], "logged": rec["logged"]})
    probes = {}
    for tid in (T10, T11):
        try:
            res = pr.assert_can_score(PID, "P1", test_id=tid, log=False)
            probes[tid] = {"can_score_as_preregistered": True,
                           "other_tests_prior_looks": res["other_tests_prior_looks"]}
        except pr.PreregistrationError as exc:
            probes[tid] = {"can_score_as_preregistered": False,
                           "error": type(exc).__name__, "message": str(exc)[:900]}
    blocking = pr.blocking_looks([t for t in pr.load_prereg(PID)[0]["tests"]], None)
    out = {"appended": appended, "skipped_already_present": skipped,
           "assert_can_score_log_false": probes,
           "blocking_looks_any_time": [{"test_id": b["test_id"], "units": b["units"],
                                        "look_time": b["look"]["time"],
                                        "look_logged": b["look"]["logged"],
                                        "look_purpose": b["look"]["purpose"],
                                        "look_test_id": b["look"]["test_id"]}
                                       for b in blocking],
           "prereg_file_unchanged": pr.load_prereg(PID)[1]}
    Path(evidence).parent.mkdir(parents=True, exist_ok=True)
    Path(evidence).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
