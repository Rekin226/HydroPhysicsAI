"""Record in the A1 look log (harness.prereg.record_look) the W2W4 looks at results that the
scoring calls did not log themselves (resume round, 2026-10-12).

Already logged elsewhere: the W2 scoring looks of W2.et0_overlap.v3 (assert_can_score in
build_et0.py, twice: scratch build 10:29:32Z and the et0/v1 build 10:30:04Z) and the W4
LOGO scores of the first round (prereg_a1.py W4look). This script adds:

* design-time seeds, back-dated to documented evidence (pre-registration v2 or the L0 file
  times): the 2022 overlap cache vs forcing experiment (T1, computed by the plan and by the
  first round), the one-point local-day probe and the endpoint/model variants (GMT
  convention), and the 2012-2021 offset history at st1 and st42 (level-match stability);
* late-logged looks with the real time in the note: a possible first-round look at the 18
  main-fetch responses fetched before the interruption, the W4 residual-gap flag
  diagnostic, the W4 reproducibility rerun, the climatology-bias secondary of the et0
  builds and the post-hoc station check at CWA 467270.

Usage: python -I isorun.py look_log.py [--dry]
"""
from __future__ import annotations

import argparse
import json
import sys

from harness import prereg as P

WP = "W2W4"
GAUGES = [f"rain_gauge_rf{i}" for i in range(1, 27)]

SEEDS = [
    dict(units=["openmeteo_et0_overlap_2022_cache_vs_fe_61pts"],
         data_scope="openmeteo_et0:2022-01..2022-12", seeded_time="2026-10-12T02:35:37Z",
         note="T1 (2022 daily ET0, calibration cache vs forcing-experiment GMT fetch, 61 "
              "points) was computed by the plan's review ('checked here', median MAE 0.048, "
              "max 0.063) and by the first W2W4 round; its values appear in "
              "W2.et0_overlap.v2 (created 02:35:37Z), so the look is no later than that. "
              "Re-computed by build_et0.py at 10:29:32Z and 10:30:04Z (disclosed "
              "re-computation, not a fresh test)."),
    dict(units=["openmeteo_et0_probe_st1_2022_local_and_variants"],
         data_scope="openmeteo_et0:st1:2022", seeded_time="2026-10-12T02:35:37Z",
         note="one-point local-day probe (openmeteo_et0/w2_probe_st1_2022, fetched 02:31:36Z) "
              "and the January-2022 endpoint/model/timezone variants at st1 (archived in "
              "w2_variant_probes_2026-10-12) compared with the cache; quoted in "
              "W2.et0_overlap.v2 (02:35:37Z) as design-time evidence for the GMT convention. "
              "The variant table was re-computed in the resume round (archive_probes.py "
              "--summary, about 10:26Z) for the report."),
    dict(units=["openmeteo_et0_offset_history_2012_2021_st1_st42"],
         data_scope="openmeteo_et0:2012-01..2021-12:st1,st42",
         seeded_time="2026-10-12T02:40:03Z",
         note="annual offsets cache minus a fresh 2012-2021 GMT fetch at st1 and st42 "
              "(fetched 02:38:49Z-02:40:01Z, after the level-match rule was fixed in v2 at "
              "02:35:37Z); the first round's report draft states that the rule held, so a "
              "look happened between 02:40Z and the interruption at about 02:49Z. "
              "Re-computed by build_et0.py at 10:29:32Z and 10:30:04Z."),
]

LATE = [
    dict(units=["openmeteo_et0_main_2025-01..2025-06_61pts"],
         data_scope="openmeteo_et0:2025-01..2025-06", purpose="possible_unlogged_look",
         note="logged late; possible only. W2.et0_overlap.v3 (prior_exposure) discloses that "
              "a scratch look by the first round at the 18 main-fetch responses st1..st18 "
              "(fetched 02:35:51Z-02:49Z) cannot be excluded. If it happened, it came after "
              "the T2 design was fixed in W2.et0_overlap.v2 (02:35:37Z) and before its A1 "
              "record v3 (06:13:47Z). T2 has no tunable choice; its result was 100 % "
              "identical values."),
    dict(units=GAUGES, data_scope="rain_daily_marks:2012-01..2026-09",
         purpose="after_the_fact",
         note="logged late: W4 residual-gap diagnostic rain_flags.py (CODIS missing-value "
              "marks; kept gauge-months strict vs marks 1-2 accepted), computed at about "
              "2026-10-12T02:45Z (W2W4_evidence/rain_flag_diagnostic.json); report only, "
              "no decision depends on it"),
    dict(units=GAUGES, data_scope="rain_monthly:2012-01..2026-09", purpose="reproduce",
         prereg_id="W4.rain_kriging_vs_idw.v1", test_id="W4.v1.logo_ked_vs_idw",
         note="logged late: reproducibility rerun of rain_check.py into the scratchpad at "
              "about 2026-10-12T06:41Z (W2W4_evidence/rain_check_v1_rerun.json); all 8 "
              "outputs byte-identical to rain_check/v1"),
    dict(units=["W2.et0_climatology_bias_2023_2026_fanmean"],
         data_scope="et0_splice:2023-01..2026-09", purpose="secondary_report",
         prereg_id="W2.et0_overlap.v3",
         note="registered secondary (reported, not gating): fan-mean monthly ET0 2023-26 "
              "minus the 2012-2021 cache climatology, computed by the scratch build "
              "(10:29:32Z) and the et0/v1 build (10:30:04Z)"),
    dict(units=["W2.station467270_vs_st7_monthly_2020_2026"],
         data_scope="station_et0:2020-05..2026-09", purpose="after_the_fact",
         note="post-hoc optional station check (station_et0.py, not in the pre-registration, "
              "not gating): CWA 467270 FAO-56 ET0 and Class-A pan vs et0/v1 at st7, "
              "computed at about 10:30Z; no decision depends on it"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    recs = []
    for s in SEEDS:
        recs.append(("seed", s))
    for x in LATE:
        recs.append(("late", x))
    if a.dry:
        print(json.dumps([r for _, r in recs], indent=1, ensure_ascii=False))
        return 0
    out = []
    for kind, r in recs:
        if kind == "seed":
            out.append(P.record_look(r["units"], r["data_scope"], WP, "design_time_seed",
                                     seeded_time=r["seeded_time"], note=r["note"]))
        else:
            out.append(P.record_look(r["units"], r["data_scope"], WP, r["purpose"],
                                     test_id=r.get("test_id"), prereg_id=r.get("prereg_id"),
                                     note=r["note"]))
    print(json.dumps([{k: o.get(k) for k in ("time", "purpose", "data_scope")} for o in out],
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
