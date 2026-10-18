"""WP S3 repair round: record in the A1 look log (``harness.prereg2``) the looks at S3 data
made since the resume round and before the v2 pre-registration (one-off; a look already
recorded with the same scope and note is skipped).

    python -I dataprep_staging/s3_gnss/v2/record_looks_r3.py

1. The independent verifier's re-runs (2026-10-12, after the resume round; the verifier did
   not log them): all 1,600 seed-303 A1 trials re-run from the current code, A2 recomputed
   with its own code on the published pre-origin tables (also with a 21-day ``wra_hf``), and
   the pre-origin jump and residual statistics of LNJS, YSLL and ANES for 2020-22 and 2023-25.
2. This round's diagnostics before any v2 design: day-to-day jump counts (> 3 cm) and
   residual spread of every WRA station for 2020-22 and 2023-25 and by month, the v1 step
   tables, and a smoke run of the v2 pipeline (pre-origin and full WRA family, round-r2
   detector) that printed the steps and the re-test notes. No score was computed.
3. Tune round r3 (``v2/tune.py``): design on pre-origin data, seeds 101 and 202 (design-time
   units, as rounds r1 and r2).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness import prereg2  # noqa: E402

V1 = "S3.gnss_clean_acceptance.v1"
A2_UNITS = ["s3.pair.GFES-光復國小", "s3.pair.HLES-宏崙國小", "s3.pair.HNES-湖南國小",
            "s3.pair.JMES-拯民國小", "s3.pair.KTES-客厝國小", "s3.pair.SJES-溪州國小",
            "s3.pair.SSES-新生國小", "s3.pair.STES-秀潭國小", "s3.pair.TKJS-土庫國中",
            "s3.pair.YCES-元長國小", "s3.pair.GFES-NTUH_282_29", "s3.pair.SJES-SJES",
            "s3.pair.STES-STES", "s3.pair.TKJS-TKJS_281_29"]
TWIN_WRA = "s3.twin_score.gnss_rates.wra.2023-01..2025-03"
SEED_UNITS = ["s3.injection.seed101.wra.preorigin", "s3.injection.seed101.ngl.preorigin",
              "s3.injection.seed202.wra.preorigin", "s3.injection.seed202.ngl.preorigin"]

LOOKS = [
    dict(units=["s3.injection.seed303.wra.preorigin", "s3.injection.seed303.ngl.preorigin"],
         data_scope="gnss_injection:preorigin_background:seed303", purpose="reproduce",
         prereg_id=V1, test_id="S3.A1_injected_steps_preorigin",
         note="independent verifier (2026-10-12, after the resume round; not logged by the "
              "verifier, recorded here by the S3 repair round): all 1,600 seed-303 trials re-run "
              "from the v1 code inside cpu_slot.sh; every injection date and recovered flag "
              "identical; WRA 0.816, NGL 0.891; post-hoc +-10 mm 0.917/0.953 reproduced"),
    dict(units=A2_UNITS, data_scope="gnss_wra+rings+piles:2020-01..2021-12:extra_2021",
         purpose="reproduce", prereg_id=V1,
         note="independent verifier: A2 recomputed with its own code on gnss_clean/preorigin/v1 "
              "(same sign 0.571, pooled ratio 0.737 wra_hf, 0.635 uncorrected, 0.402 on-fan, "
              "0.349 ngl_offfan; unique GNSS stations 3/5) and with a 21-day wra_hf (0.571/0.742)"),
    dict(units=A2_UNITS + [TWIN_WRA],
         data_scope="gnss_wra:2020-01..2025-03:qc_jumps_resid_sd",
         purpose="diagnose",
         note="independent verifier: day-to-day jumps > 3 cm and residual sd of LNJS, YSLL and "
              "ANES in 2020-22 (5-6/10-12/6-7 jumps) and 2023-25 (134-140/131-133/69-74); no score"),
    dict(units=A2_UNITS + [TWIN_WRA],
         data_scope="gnss_wra:2020-01..2025-03:qc_jumps_v1_steps_v2_smoke",
         purpose="diagnose",
         note="S3 repair round, before any v2 design: jump counts (> 3 cm) and 31-day high-pass "
              "spread of every WRA station for 2020-22 and 2023-25 and by month; the v1 steps.csv "
              "of both variants; a smoke run of the v2 pipeline (pre-origin and full WRA family, "
              "round-r2 detector) printing steps and re-test notes; no score computed"),
    dict(units=SEED_UNITS, data_scope="gnss_injection:preorigin_background:tune_r3",
         purpose="design_time_seed",
         note="S3 tune round r3 (dataprep_staging/s3_gnss/v2/tune.py, results/dataprep/reports/"
              "S3_evidence/repair_r3/tune_r3/): 2/3 cm recovery for 20 detector configs and 4 "
              "local windows on the v2 pipeline, pre-origin series only"),
]


def main() -> int:
    existing = prereg2.read_chained(prereg2.PREREG_DIR / "looks.jsonl")
    for lk in LOOKS:
        dup = [e for e in existing if e["wp"] == "S3" and e["data_scope"] == lk["data_scope"]
               and e["note"] == lk["note"]]
        if dup:
            print("already recorded:", lk["data_scope"])
            continue
        rec = prereg2.record_look(lk["units"], lk["data_scope"], "S3", lk["purpose"],
                                  prereg_id=lk.get("prereg_id"), test_id=lk.get("test_id"),
                                  note=lk["note"])
        print("recorded", rec["time"], rec["purpose"], rec["data_scope"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
