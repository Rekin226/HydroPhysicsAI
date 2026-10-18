"""WP S3: record, in the A1 look log, the looks at S3 test data made before the
pre-registration (one-off; refuses to record the same look twice).

    python -I dataprep_staging/s3_gnss/record_looks.py

1. Design-time seeds: the pre-origin tuning rounds r1 (2026-10-12 10:46-11:13 CST) and r2
   (from 14:14 CST) scored injected-step recovery on synthetic injections with seeds 101
   and 202 over the pre-origin series. Their units are those synthetic injection sets; the
   acceptance test uses new synthetic units (seed 303 and the harness family G seeds) on the
   same pre-origin background (disclosed in the pre-registration).
2. The pilot ``subsidence_gap_sources/scripts/gnss_stepfix.py`` computed GNSS-minus-ring
   excess rates over 2020-01..2021-12 at seven co-located pairs (design-time seed; the A3
   "pre" statistic).
3. An unregistered look by this WP: a smoke test of ``run_validation.py`` (2026-10-12 about
   14:21 CST, round-r1 hyperparameters, pre-origin variant) computed and printed the A2
   statistic (2021 extra compaction after the common-mode removal) on the co-located pairs
   before any pre-registration was written. A2's operationalisation and threshold were fixed
   earlier in ``results/dataprep/reports/S3_evidence/acceptance_spec.json`` (10:47 CST,
   sha256 d385a20a...), and nothing was changed after the look; A2 is therefore declared,
   not registered as a blind test.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import prereg  # noqa: E402

A2_UNITS = ["s3.pair.GFES-光復國小", "s3.pair.HLES-宏崙國小", "s3.pair.HNES-湖南國小",
            "s3.pair.JMES-拯民國小", "s3.pair.KTES-客厝國小", "s3.pair.SJES-溪州國小",
            "s3.pair.SSES-新生國小", "s3.pair.STES-秀潭國小", "s3.pair.TKJS-土庫國中",
            "s3.pair.YCES-元長國小", "s3.pair.GFES-NTUH_282_29", "s3.pair.SJES-SJES",
            "s3.pair.STES-STES", "s3.pair.TKJS-TKJS_281_29"]
PILOT_A3_UNITS = ["s3.pair.TKJS-土庫國中", "s3.pair.SJES-溪州國小", "s3.pair.GFES-光復國小",
                  "s3.pair.KTES-客厝國小", "s3.pair.SSES-新生國小", "s3.pair.HNES-湖南國小",
                  "s3.pair.HLES-宏崙國小"]

LOOKS = [
    dict(units=["s3.injection.seed101.wra.preorigin", "s3.injection.seed101.ngl.preorigin",
                "s3.injection.seed202.wra.preorigin", "s3.injection.seed202.ngl.preorigin"],
         data_scope="gnss_injection:preorigin_background:tune_r1", purpose="design_time_seed",
         seeded_time="2026-10-12T02:46:00.000000Z",
         note="S3 tune round r1 (tune.py, results/dataprep/reports/S3_evidence/tune/): "
              "2/3 cm recovery for 16 detector configs and 4 local windows"),
    dict(units=["s3.injection.seed101.wra.preorigin", "s3.injection.seed101.ngl.preorigin",
                "s3.injection.seed202.wra.preorigin", "s3.injection.seed202.ngl.preorigin"],
         data_scope="gnss_injection:preorigin_background:tune_r2", purpose="design_time_seed",
         seeded_time="2026-10-12T06:14:00.000000Z",
         note="S3 tune round r2 (tune.py, results/dataprep/reports/S3_evidence/tune_r2/): "
              "2/3 cm recovery for 20 detector configs and 4 local windows"),
    dict(units=PILOT_A3_UNITS, data_scope="gnss_vs_rings_excess_rate:2020-01..2021-12",
         purpose="design_time_seed", seeded_time="2026-10-12T00:00:00.000000Z",
         note="pilot subsidence_gap_sources gnss_stepfix.py: pre-gap (2020-21) GNSS-minus-"
              "ring excess rates at 7 pairs (gnss_vs_ring_stepfixed.csv)"),
    dict(units=A2_UNITS, data_scope="gnss_wra+rings+piles:2020-01..2021-12:extra_2021",
         purpose="unregistered_smoke_test",
         note="S3 run_validation.py smoke test (--skip-injection, r1 hyperparameters, "
              "pre-origin variant) printed the A2 statistic before the pre-registration "
              "(evaluable 5 pairs; same sign 0.8; pooled ratio 0.89 with wra_hf). A2's "
              "threshold predates it (acceptance_spec.json 10:47 CST); nothing changed after"),
]


def main() -> int:
    existing = prereg.read_chained(prereg.PREREG_DIR / "looks.jsonl")
    for lk in LOOKS:
        dup = [e for e in existing if e["wp"] == "S3" and e["data_scope"] == lk["data_scope"]
               and e["note"] == lk["note"]]
        if dup:
            print("already recorded:", lk["data_scope"])
            continue
        rec = prereg.record_look(lk["units"], lk["data_scope"], "S3", lk["purpose"],
                                 seeded_time=lk.get("seeded_time"), note=lk["note"])
        print("recorded", rec["time"], rec["data_scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
