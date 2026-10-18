"""H1a repair round (2026-10-13): disclosure and diagnostic looks in the A1 look log
(``harness.prereg2``). The registered scoring steps are gated by ``looks_h1a_v2.py``; this
script logs only the looks that are not scores.

Usage:
  looks_h1a_repair.py history     disclose the unlogged pre-registration dry run of 2026-10-12
  looks_h1a_repair.py diagnose    log the after-the-fact verification of the v2 product
                                  (level check of every correction, false-flag proxies,
                                  O1 over all wells, harness artefacts, BE plausibility)
  looks_h1a_repair.py review      log the co-located sibling-difference check of the v2
                                  corrections that motivated the post-look v3 review, and the
                                  recomputation of the leakage probe and conformance checks on
                                  the v3 build
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import prereg2  # noqa: E402

PREREG_ID = "H1a.qc_acceptance.v2"
WP = "H1a"

HISTORY = [
    (["H1a:v2_preorigin_rows", "H1a:offset_episodes_2012_2022", "H1a:post_origin_reported"],
     "heads:2012-01..2026-09",
     "H1a repair-round history (logged 2026-10-13, not logged at the time): at about "
     "2026-10-12T16:19Z-16:21Z, two to four minutes BEFORE the pre-registration "
     "H1a.qc_acceptance.v2 (16:23:09Z), the previous session ran s2b_decide_v2.py and "
     "s3_assemble_v2.py on the FULL stage 1 (pre- and post-origin) as a dry run. The logs "
     "printed only counts of v2 decisions by period, kind, status and flag (post-origin "
     "included: 4 accepted steps, 6 offset episodes, 2 dropouts, 15 rejected-step runs) and "
     "the table sizes. No leakage comparison of full vs pre-only rows was computed and no "
     "family O realization r >= 10 was generated. decide_v2.py, levelcheck.py, assemble_v2.py, "
     "s2b_decide_v2.py, s3_assemble_v2.py, s5_inject_v2.py, s6_build_product_v2.py and "
     "s7_leakprobe.py were last modified before the dry run; s4_validate_v2.py was edited at "
     "16:22Z (after the pre-only smoke test). The pre-registration's statement 'the v2 builds "
     "did not exist before this registration' is therefore inaccurate for the full build; the "
     "repair round discloses it and reruns every stage from L0 in a fresh work directory."),
]

DIAGNOSE = [
    (["H1a:offset_episodes_2012_2022", "H1a:post_origin_reported"], "heads:2012-01..2026-09",
     "H1a repair round (2026-10-13): after-the-fact verification of the v2 product, after "
     "all registered scores: independent level check (10-day medians, 0.5|j|) of every "
     "flag-1/3 correction and of every flagged object, the verifier's two examples "
     "(07230332, 07260122), false-flag proxies on real data, O1 over all wells, BE "
     "plausibility, rescore attribution. Reported only; no rule or threshold changed."),
    (["H1a:inject_O_r10-19_paired", "H1a:inject_O_r10-19_step"], "heads:2012-01..2022-12",
     "H1a repair round (2026-10-13): after scoring, diagnosis of the r = 10..19 misses "
     "(harness artefacts such as effective durations beyond 365 days, kinds at the jump). "
     "Reported only; the registered recall stands."),
]

REVIEW = [
    (["H1a:offset_episodes_2012_2022", "H1a:post_origin_reported"], "heads:2012-01..2026-09",
     "H1a repair round (2026-10-13): co-located sibling-difference check (daily well minus "
     "nested sibling, 30/90/365 days each side) of the 9 step corrections of the v2 build "
     "(6 pre-origin, 3 post-origin), after all registered scores; none of the 8 co-located-"
     "backed corrections is confirmed (7 not confirmed over 365 days, 1 with only 7 sibling "
     "days after the jump). Motivates the "
     "post-look v3 review (co-located backing alone no longer corrects), which has no tunable "
     "constant and changes only the flag of those 8 steps."),
    (["H1a:v2_preorigin_rows", "H1a:real_events_2024", "H1a:offset_episodes_2012_2022",
      "H1a:audit_artefact_events_hist", "H1a:o1_corrected_wells_2012_2022"],
     "heads:2012-01..2026-09",
     "H1a repair round (2026-10-13): the leakage probe (full vs pre-only rows) and the "
     "conformance checks C3-C6 recomputed on the post-look v3 build, as reported checks after "
     "the registered scores (the registered results on the v2 build stand)."),
]


def log(entries, purpose):
    for units, scope, note in entries:
        rec = prereg2.record_look(units, scope, WP, purpose, note=note)
        print(json.dumps({k: rec[k] for k in ("time", "purpose", "units")}))


if __name__ == "__main__":
    if sys.argv[1:] == ["history"]:
        log(HISTORY, "history_disclosure")
    elif sys.argv[1:] == ["diagnose"]:
        log(DIAGNOSE, "diagnose")
    elif sys.argv[1:] == ["review"]:
        log(REVIEW, "diagnose")
    else:
        raise SystemExit(__doc__)
