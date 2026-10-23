"""WP S3 v3 (repair round 3 S3r): look-log unit ids and their members (``harness.prereg3``
requires members for units outside the validation-unit registries). Frozen with the v3
pre-registration: the acceptance run imports it, so it is a registered input.

The A2 pair units keep the ids of the v1/v2 rounds (they are spent: every v3 A2 score on them
is a second look).
"""
from __future__ import annotations

SYN = ("synthetic step injections on GNSS height series; no registered well, leveling "
       "benchmark, fold test set or head-data cell")
A2_PAIRS = [("GFES", "光復國小"), ("HLES", "宏崙國小"), ("HNES", "湖南國小"), ("JMES", "拯民國小"),
            ("KTES", "客厝國小"), ("SJES", "溪州國小"), ("SSES", "新生國小"), ("STES", "秀潭國小"),
            ("TKJS", "土庫國中"), ("YCES", "元長國小"), ("GFES", "NTUH_282_29"), ("SJES", "SJES"),
            ("STES", "STES"), ("TKJS", "TKJS_281_29")]
A2_UNITS = [f"s3.pair.{g}-{r}" for g, r in A2_PAIRS]
SEED_TEST = 707
FRESH_A1 = [f"s3.injection.seed{SEED_TEST}.wra.preorigin",
            f"s3.injection.seed{SEED_TEST}.ngl.preorigin"]
#: earlier draws: tune r1-r3 (101, 202), failed tests (303, 404; diagnosed in S3r), v3 design
#: run 1 (505, stopped), run 2 (515, 616) and run 3 (525, 626)
DESIGN_SEEDS = (101, 202, 303, 404, 505, 515, 616, 525, 626)
DESIGN_UNITS = [f"s3.injection.seed{s}.{f}.preorigin" for s in DESIGN_SEEDS
                for f in ("wra", "ngl")]
HARNESS_G = ["s3.injection.harnessG.r20-29.wra.preorigin",
             "s3.injection.harnessG.r20-29.ngl.preorigin"]
FULL_INJ = [f"s3.injection.seed{SEED_TEST}.wra.full", f"s3.injection.seed{SEED_TEST}.ngl.full",
            "s3.injection.harnessG.r20-29.wra.full", "s3.injection.harnessG.r20-29.ngl.full"]


def a2_members(window: str) -> dict:
    """Members of the A2 pair units: the GNSS station and its ring or pile, in ``window``."""
    return {f"s3.pair.{g}-{r}": {"ids": [f"gnss:{g}", f"ring_or_pile:{r}"], "window": window,
                                 "description": "co-located WRA GNSS station and compaction "
                                                "ring or deep pile (2021 extra compaction)"}
            for g, r in A2_PAIRS}


def seed_members(units: list[str]) -> dict:
    """Members of synthetic injection units: none (simulation draws)."""
    return {u: {"none": SYN, "description": SYN} for u in units}
